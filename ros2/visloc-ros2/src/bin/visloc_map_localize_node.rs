//! `visloc_map_localize_node`: robot-style localization against a saved
//! Basalt map, with the odometry corrected into the map frame.
//!
//! Loads a map saved by `basalt_euroc_online_slam_demo --save-map`,
//! subscribes `image` (the camera the map's localization uses, raw) and
//! `odom` (`nav_msgs/Odometry`, gravity-aligned: a VIO such as
//! `visloc_vio_node`, or a ground robot's wheel odometry on level floors),
//! and publishes `/tf` (`map -> odom`, REP-105),
//! `map_pose` (the body pose in the map,
//! `geometry_msgs/PoseWithCovarianceStamped`; not `pose`, which
//! `visloc_vio_node` already publishes), `map_fix` (each localization),
//! `odom_fix` (each fix the correction used, in the odometry frame, for
//! `visloc_vio_node`'s `map_prior`) and `/diagnostics`. Run with `--help` for parameters.

use std::path::PathBuf;
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use visloc_ros2::dds::{
    latched_qos, output_qos, publish_logged, sensor_qos, spawn_receiver, RosNode, StaticTfListener,
};
use visloc_ros2::image::decode_luma;
use visloc_ros2::map_localize::{
    fix_outcome_name, load_basalt_map, map_diagnostics_msg, map_fix_msg, map_pose_msg,
    map_to_odom_tf, odom_fix_msg, odometry_noise_by_name, se3_from_csv, se3_from_ros, Fix,
    FixEvent, FusionOptions, ImageCadence, MapLocalizeOptions, MapLocalizer, Matching,
    OdomCorrector, OdomLookup,
};
use visloc_ros2::msgs::{
    DiagnosticArray, Header, Image, Int32, KeyValue, Odometry, PointCloud2,
    PoseWithCovarianceStamped, TfMessage, Time,
};
use visloc_ros2::params::NodeArgs;
use visloc_ros2::queue::DropOldestQueue;
use visloc_ros2::tf_tree::StaticTransforms;

const USAGE: &str = "\
visloc_map_localize_node -- localization against a saved Basalt map, correcting odometry into it

USAGE:
  visloc_map_localize_node --map <dir from basalt_euroc_online_slam_demo --save-map> [options]

PARAMETERS (--name value  or  -p name:=value):
  map                    saved map directory (map.json, sparse/, landmark_descriptors.txt) [required]
  camera                 Basalt camera index of the image topic                       [0]
  map_frame              frame of the map; parent of the published map -> odom       [map]
  extrinsics             where the camera sits on the odometry's body (child_frame_id):
                         map = the map's IMU-from-camera (body = the map's IMU, as
                         from visloc_vio_node); tf = /tf_static from the image's
                         frame to the body frame (e.g. base_link)                   [map]
  camera_frame           with extrinsics tf: the camera's frame       [the image's frame_id]
  publish_tf             publish /tf map_frame -> odometry frame                      [true]
  publish_map_cloud      publish the map's landmarks once on map_cloud (latched)     [true]
  localize_every         localize every N-th received image                           [5]
  initial_localize_every ... and every N-th until the first map pose (which needs two
                         agreeing fixes)                                              [1]
  initial_window_s       ... within this many seconds of the first image or the
                         latest localization; then every localize_every-th            [30]
  min_inliers            minimum PnP inliers to accept a fix                          [30]
  weak_min_inliers       a map-wide localization with this many inliers (but fewer than
                         min_inliers) only helps set the first map pose, with three
                         agreeing fixes; >= min_inliers turns it off                 [15]
  weak_gate_m            once the odometry has a map pose, a weak localization (also by
                         projection) this close (m) to the corrected odometry is used
                         too; for robots whose images rarely reach min_inliers     [0: off]
  min_inlier_cells       turn down a localization whose inliers occupy fewer cells of a
                         4x4 image grid (bunched on one patch, e.g. the rest of the view
                         hidden by people); 6 was measured on TUM walking          [0: off]
  init_warmup_s          before the first map pose, wait this long, then start from the
                         strongest fixes (most inliers) that agree, not the first two
                         that agree; 3 was measured on TUM walking                 [0: off]
  fusion                 also subscribe odom2 (a second odometry, e.g. wheels next to a
                         VIO on odom) and fuse both with the fixes in a sliding-window
                         pose graph; the map pose follows the fusion             [false]
  odom_noise             drift model of odom: vio, wheel or legged                    [vio]
  odom2_noise            ... of odom2                                                 [wheel]
  odom2_body_offset      odom2's body in odom's body frame (T_body_body2),
                         tx,ty,tz,qx,qy,qz,qw, e.g. a VIO's IMU <- base_link   [identity]
  fusion_source_check_s  leave out an odometry whose motion over the last this many
                         seconds disagrees with the fixes (people dragging a VIO)
                         while it does; 1 was measured in RNE                    [0: off]
  fusion_seed_s          when the map pose starts (or resets), start the fusion from
                         the fixes of the last this many seconds that agree with that
                         fix under one odometry, so the source check judges the VIO
                         from the start; 3 (with init_warmup_s 3) measured in RNE [0: off]
  max_hamming            largest ORB Hamming distance of a match                      [70]
  ratio                  nearest / second-nearest ratio (<= 0 disables)               [0.8]
  ransac_iterations      P3P RANSAC iterations                                        [1000]
  p3p_solver             P3P solver in RANSAC: grunert, or lambda-twist (more accurate
                         when most landmarks in view are on walls)                    [grunert]
  inlier_refits          rounds of re-collecting the refined pose's inliers and refining
                         again (0: refine once on the RANSAC winner's inliers)        [0]
  reprojection_threshold PnP inlier threshold in pixels                               [3.0]
  projection_radius      with a map pose, match each landmark only with keypoints this
                         close (px) to its projection from the predicted pose; on
                         failure, and with <= 0 always, match map-wide             [20]
  projection_fresh_s     ... only within this many seconds of the newest fix          [5]
  projection_fresh_m     ... and this many metres of odometry travel from it          [2]
  reset_prior_s          after a VIO reset (odom_reset), the last map pose is the
                         prior for this many seconds (0: off)                         [30]
  dump_inliers           CSV file: each accepted localization's PnP inliers,
                         `timestamp_ns,x,y,landmark_id` (keypoint px), for visualizations [none]
  prior_wait_ms          longest wait for the odometry of an image's stamp before
                         matching map-wide instead                                    [300]
  match_threads          threads of the map-wide (and radius) search                  [1]
  prior_radius           with a prior whose projection search fails, first match the
                         landmarks within this many metres of it (0: off; dense maps) [0]
  no_global_with_prior   with a prior, never match map-wide (dense maps)              [false]
  duplicate_radius       map-wide search: landmarks within this many metres of the best
                         one count as the same point in the ratio test (0: off)      [0]
  extra_descriptors      extra descriptors per landmark to match, from the map's
                         landmark_descriptors_extra.txt (0: the main one only)        [0]
  max_points             ORB keypoints per image                                      [the map's]
  position_stddev        published 1-sigma position at a fix (m)                      [0.03]
  position_stddev_per_m  ... plus this per metre of VIO travel since the fix          [0.025]
  orientation_stddev     published 1-sigma orientation (rad)                          [0.02]
  image_queue            images buffered while localizing (drop-oldest)              [1]
  domain_id              DDS domain (default: $ROS_DOMAIN_ID or 0)
  node_name              node name                                                    [visloc_map_localize]

TOPICS (remap with --remap from=to or -r from:=to):
  in : image (sensor_msgs/Image, raw), odom (nav_msgs/Odometry, gravity-aligned;
       its header.frame_id and child_frame_id name the odometry and body frames),
       /tf_static (tf2_msgs/TFMessage, with extrinsics tf), odom_reset (std_msgs/Int32,
       from visloc_vio_node: when it changes, the odometry restarted in a new frame, and
       the map pose starts over from new fixes), odom2 (nav_msgs/Odometry, with fusion)
  out: /tf (map_frame -> odometry frame), map_pose (geometry_msgs/PoseWithCovarianceStamped,
       body pose in map_frame, after two agreeing fixes), map_fix (geometry_msgs/
       PoseWithCovarianceStamped, the body pose of each image localized with min_inliers,
       e.g. for visloc_wheel_calibration_node), odom_fix (geometry_msgs/
       PoseWithCovarianceStamped, each fix the correction used, as the body pose it
       implies in the odometry frame; for visloc_vio_node's map_prior), /diagnostics (diagnostic_msgs/DiagnosticArray),
       map_cloud (sensor_msgs/PointCloud2, the map's landmarks in map_frame, transient-local)
";

fn main() {
    if let Err(error) = run() {
        eprintln!("visloc_map_localize_node: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = NodeArgs::parse(std::env::args().skip(1))?;
    if args.help {
        print!("{USAGE}");
        return Ok(());
    }
    let map_dir = PathBuf::from(args.required("map")?);
    let camera = args.parsed_or("camera", 0u16)?;
    let map_frame = args.string_or("map_frame", "map");
    let extrinsics_from_tf = match args.string_or("extrinsics", "map").as_str() {
        "map" => false,
        "tf" => true,
        other => return Err(format!("extrinsics: expected map or tf, got {other:?}")),
    };
    let camera_frame = args.string("camera_frame");
    let mut dump_inliers = match args.string("dump_inliers") {
        Some(path) => {
            let mut file = std::io::BufWriter::new(
                std::fs::File::create(&path).map_err(|e| format!("dump_inliers {path}: {e}"))?,
            );
            std::io::Write::write_all(&mut file, b"timestamp_ns,x,y,landmark_id\n")
                .map_err(|e| format!("dump_inliers {path}: {e}"))?;
            Some(file)
        }
        None => None,
    };
    let publish_tf = args.flag("publish_tf", true)?;
    let publish_map_cloud = args.flag("publish_map_cloud", true)?;
    let localize_every = args.parsed_or("localize_every", 5u64)?.max(1);
    let initial_localize_every = args.parsed_or("initial_localize_every", 1u64)?.max(1);
    let initial_window_s = args.parsed_or("initial_window_s", 30.0f64)?;
    if !(initial_window_s >= 0.0 && initial_window_s.is_finite()) {
        return Err("initial_window_s must be a non-negative number".into());
    }
    let mut options = MapLocalizeOptions::default();
    options.min_inliers = args.parsed_or("min_inliers", options.min_inliers)?;
    options.weak_min_inliers = args.parsed_or("weak_min_inliers", options.weak_min_inliers)?;
    options.correction.weak_gate_m =
        args.parsed_or("weak_gate_m", options.correction.weak_gate_m)?;
    if !(options.correction.weak_gate_m >= 0.0 && options.correction.weak_gate_m.is_finite()) {
        return Err("weak_gate_m must be a non-negative number".into());
    }
    options.min_inlier_cells = args.parsed_or("min_inlier_cells", options.min_inlier_cells)?;
    if options.min_inlier_cells > 16 {
        return Err("min_inlier_cells must be between 0 and 16".into());
    }
    options.correction.init_warmup_s =
        args.parsed_or("init_warmup_s", options.correction.init_warmup_s)?;
    if !(options.correction.init_warmup_s >= 0.0 && options.correction.init_warmup_s.is_finite()) {
        return Err("init_warmup_s must be a non-negative number".into());
    }
    if args.flag("fusion", false)? {
        let mut fusion = FusionOptions::default();
        if let Some(name) = args.string("odom_noise") {
            fusion.noise = odometry_noise_by_name(&name).map_err(|e| format!("odom_noise: {e}"))?;
        }
        if let Some(name) = args.string("odom2_noise") {
            fusion.second_noise =
                odometry_noise_by_name(&name).map_err(|e| format!("odom2_noise: {e}"))?;
        }
        if let Some(text) = args.string("odom2_body_offset") {
            fusion.body_from_second_body =
                se3_from_csv(&text).map_err(|e| format!("odom2_body_offset: {e}"))?;
        }
        fusion.source_check_window_s =
            args.parsed_or("fusion_source_check_s", fusion.source_check_window_s)?;
        if !(fusion.source_check_window_s >= 0.0 && fusion.source_check_window_s.is_finite()) {
            return Err("fusion_source_check_s must be a non-negative number".into());
        }
        fusion.seed_s = args.parsed_or("fusion_seed_s", fusion.seed_s)?;
        if !(fusion.seed_s >= 0.0 && fusion.seed_s.is_finite()) {
            return Err("fusion_seed_s must be a non-negative number".into());
        }
        options.fusion = Some(fusion);
    } else {
        for key in [
            "odom_noise",
            "odom2_noise",
            "odom2_body_offset",
            "fusion_source_check_s",
            "fusion_seed_s",
        ] {
            if args.string(key).is_some() {
                return Err(format!("{key} needs fusion:=true"));
            }
        }
    }
    options.max_hamming = args.parsed_or("max_hamming", options.max_hamming)?;
    let ratio = args.parsed_or("ratio", options.ratio.unwrap_or(0.0))?;
    options.ratio = (ratio > 0.0).then_some(ratio);
    options.ransac_iterations = args.parsed_or("ransac_iterations", options.ransac_iterations)?;
    options.p3p_solver = args.parsed_or("p3p_solver", options.p3p_solver)?;
    options.inlier_refits = args.parsed_or("inlier_refits", options.inlier_refits)?;
    options.reprojection_px = args.parsed_or("reprojection_threshold", options.reprojection_px)?;
    let radius = args.parsed_or(
        "projection_radius",
        options.projection_radius_px.unwrap_or(0.0),
    )?;
    options.projection_radius_px = (radius > 0.0).then_some(radius);
    options.projection_fresh_s =
        args.parsed_or("projection_fresh_s", options.projection_fresh_s)?;
    options.projection_fresh_m =
        args.parsed_or("projection_fresh_m", options.projection_fresh_m)?;
    options.reset_prior_s = args.parsed_or("reset_prior_s", options.reset_prior_s)?;
    let prior_wait = Duration::from_millis(args.parsed_or("prior_wait_ms", 300u64)?);
    options.match_threads = args.parsed_or("match_threads", options.match_threads)?;
    let prior_radius = args.parsed_or("prior_radius", 0.0f64)?;
    options.prior_radius_m = (prior_radius > 0.0).then_some(prior_radius);
    options.no_global_with_prior = args.flag("no_global_with_prior", false)?;
    let duplicate_radius = args.parsed_or("duplicate_radius", 0.0f64)?;
    options.duplicate_radius_m = (duplicate_radius > 0.0).then_some(duplicate_radius);
    options.extra_descriptors = args.parsed_or("extra_descriptors", options.extra_descriptors)?;
    options.max_points = args.parsed::<usize>("max_points")?;
    options.position_stddev_m = args.parsed_or("position_stddev", options.position_stddev_m)?;
    options.position_stddev_per_m =
        args.parsed_or("position_stddev_per_m", options.position_stddev_per_m)?;
    options.orientation_stddev_rad =
        args.parsed_or("orientation_stddev", options.orientation_stddev_rad)?;
    let image_queue = args.parsed_or("image_queue", 1usize)?;
    let domain_id = args.domain_id()?;
    let node_name = args
        .node_name
        .clone()
        .unwrap_or_else(|| args.string_or("node_name", "visloc_map_localize"));
    let namespace = args.namespace.clone().unwrap_or_else(|| "/".into());
    let image_topic = args.topic("image");
    let odom_topic = args.topic("odom");
    let odom2_topic = args.topic("odom2");
    let tf_topic = args.topic("/tf");
    let tf_static_topic = args.topic("/tf_static");
    let pose_topic = args.topic("map_pose");
    let fix_topic = args.topic("map_fix");
    let odom_fix_topic = args.topic("odom_fix");
    let diagnostics_topic = args.topic("/diagnostics");
    let map_cloud_topic = args.topic("map_cloud");
    let odom_reset_topic = args.topic("odom_reset");
    args.finish()?;

    let map = load_basalt_map(&map_dir)?;
    eprintln!(
        "visloc_map_localize_node: map {} with {} images, {} landmarks, {} descriptors",
        map_dir.display(),
        map.map.keyframes.len(),
        map.map.landmarks.len(),
        map.descriptors.len(),
    );
    // The map's landmarks, for RViz; published once below.
    let map_cloud = publish_map_cloud.then(|| {
        PointCloud2::xyz(
            Header::new(Time::from_nanos(0), map_frame.clone()),
            map.map.landmarks.values().map(|landmark| {
                let p = landmark.position;
                [p.x as f32, p.y as f32, p.z as f32]
            }),
        )
    });
    let localizer = MapLocalizer::new(map, camera, &options)?;
    if localizer.extra_descriptor_count() > 0 {
        eprintln!(
            "visloc_map_localize_node: matching {} extra landmark descriptors",
            localizer.extra_descriptor_count()
        );
    }
    let core = Arc::new(Mutex::new(OdomCorrector::new(&options)));
    // Notified on every odometry message, for images waiting for theirs.
    let odom_arrived = Arc::new(Condvar::new());

    let mut ros = RosNode::new(domain_id, &namespace, &node_name)?;
    let image_sub = ros.subscription::<Image>(&image_topic, sensor_qos(2))?;
    let odom_sub = ros.subscription::<Odometry>(&odom_topic, sensor_qos(50))?;
    let odom2_sub = options
        .fusion
        .is_some()
        .then(|| ros.subscription::<Odometry>(&odom2_topic, sensor_qos(50)))
        .transpose()?;
    let odom_reset_sub = ros.subscription::<Int32>(&odom_reset_topic, latched_qos(1))?;
    let tf_pub = ros.publisher::<TfMessage>(&tf_topic, output_qos(10))?;
    let pose_pub = ros.publisher::<PoseWithCovarianceStamped>(&pose_topic, output_qos(10))?;
    let fix_pub = ros.publisher::<PoseWithCovarianceStamped>(&fix_topic, output_qos(10))?;
    let odom_fix_pub =
        ros.publisher::<PoseWithCovarianceStamped>(&odom_fix_topic, output_qos(10))?;
    let diagnostics_pub = ros.publisher::<DiagnosticArray>(&diagnostics_topic, output_qos(10))?;
    // Kept by the transient-local writer for subscribers that join later.
    let _map_cloud_pub = match map_cloud {
        Some(cloud) => {
            let publisher = ros.publisher::<PointCloud2>(&map_cloud_topic, latched_qos(1))?;
            publish_logged(&publisher, cloud, "map_cloud");
            Some(publisher)
        }
        None => None,
    };
    // With extrinsics from tf: the static transforms, and the odometry's
    // body frame.
    let mut tf_static = extrinsics_from_tf
        .then(|| StaticTfListener::new(&mut ros, &tf_static_topic))
        .transpose()?;
    let transforms = tf_static.as_ref().map_or_else(
        || Arc::new(Mutex::new(StaticTransforms::new())),
        |listener| Arc::clone(listener.transforms()),
    );
    let body_frame = Arc::new(Mutex::new(None::<String>));

    // A VIO reset: the odometry frame starts over, so the correction does.
    {
        let core = Arc::clone(&core);
        let topic = odom_reset_topic.clone();
        let mut last = None::<i32>;
        spawn_receiver("odom-reset-rx", odom_reset_sub, move |msg: Int32| {
            if last.is_some_and(|count| count != msg.data) {
                core.lock().unwrap_or_else(|e| e.into_inner()).reset();
                eprintln!(
                    "visloc_map_localize_node: odometry reset ({} on {topic}); map pose unset \
                     until new fixes",
                    msg.data
                );
            }
            last = Some(msg.data);
        })?;
    }

    // Odometry: pair with fixes, then publish map -> odom and the map pose.
    let odom_messages = Arc::new(AtomicU64::new(0));
    {
        let core = Arc::clone(&core);
        let odom_arrived = Arc::clone(&odom_arrived);
        let odom_messages = Arc::clone(&odom_messages);
        let body_frame = Arc::clone(&body_frame);
        let map_frame = map_frame.clone();
        let options = options.clone();
        spawn_receiver("odom-rx", odom_sub, move |msg: Odometry| {
            odom_messages.fetch_add(1, Ordering::Relaxed);
            if extrinsics_from_tf {
                let mut body = body_frame.lock().unwrap_or_else(|e| e.into_inner());
                if body.as_deref() != Some(msg.child_frame_id.as_str()) {
                    *body = Some(msg.child_frame_id.clone());
                }
            }
            let stamp_ns = msg.header.stamp.to_nanos();
            let (events, map_pose, odom_fixes) = {
                let mut core = core.lock().unwrap_or_else(|e| e.into_inner());
                let (events, map_pose) = core.add_odom(stamp_ns, se3_from_ros(&msg.pose.pose));
                // Also the fixes the image thread applied since the last
                // odometry message: at most one message late.
                (events, map_pose, core.take_odom_fixes())
            };
            odom_arrived.notify_all();
            log_events(&events);
            for (fix_ns, odom_from_body) in &odom_fixes {
                publish_logged(
                    &odom_fix_pub,
                    odom_fix_msg(*fix_ns, odom_from_body, &msg.header.frame_id, &options),
                    "odom_fix",
                );
            }
            if let Some(map_pose) = map_pose {
                if publish_tf {
                    publish_logged(
                        &tf_pub,
                        map_to_odom_tf(&map_pose, &map_frame, &msg.header.frame_id),
                        "tf",
                    );
                }
                let mut pose = map_pose_msg(&map_pose, &map_frame, &options);
                pose.header.stamp = msg.header.stamp;
                publish_logged(&pose_pub, pose, "map_pose");
            }
        })?;
    }

    // The second odometry, for the fusion: only buffered; the first one's
    // messages publish.
    if let Some(odom2_sub) = odom2_sub {
        let core = Arc::clone(&core);
        spawn_receiver("odom2-rx", odom2_sub, move |msg: Odometry| {
            core.lock()
                .unwrap_or_else(|e| e.into_inner())
                .add_second_odom(msg.header.stamp.to_nanos(), se3_from_ros(&msg.pose.pose));
        })?;
        eprintln!("visloc_map_localize_node: fusing {odom_topic} and {odom2_topic} with the fixes");
    }

    // Images: keep every N-th (every M-th until the first map pose, while
    // images localize), localize the latest one when free.
    let queue = Arc::new(DropOldestQueue::<Image>::new(image_queue));
    {
        let queue = Arc::clone(&queue);
        let core = Arc::clone(&core);
        let mut cadence =
            ImageCadence::new(localize_every, initial_localize_every, initial_window_s);
        spawn_receiver("image-rx", image_sub, move |msg: Image| {
            let (has_map_pose, latest_fix_ns) = {
                let core = core.lock().unwrap_or_else(|e| e.into_inner());
                (core.is_set(), core.latest_fix_ns())
            };
            if cadence.take(msg.header.stamp.to_nanos(), has_map_pose, latest_fix_ns) {
                queue.push(msg);
            }
        })?;
    }
    eprintln!(
        "visloc_map_localize_node: domain {domain_id}, node {}; in: {image_topic} {odom_topic}; \
         out: {}{pose_topic} {fix_topic} {diagnostics_topic}; localizing every {localize_every} image(s), \
         every {initial_localize_every} until the first map pose \
         (within {initial_window_s} s of a localization)",
        visloc_ros2::dds::qualified_name(&namespace, &node_name),
        if publish_tf {
            format!("{tf_topic} ")
        } else {
            String::new()
        },
    );

    let (mut attempts, mut localized, mut projected) = (0u64, 0u64, 0u64);
    loop {
        let Some(msg) = queue.pop_timeout(Duration::from_secs(1)) else {
            continue;
        };
        attempts += 1;
        let stamp = msg.header.stamp;
        let started = Instant::now();
        let mut tf_missing = false;
        let camera_from_body = if extrinsics_from_tf {
            let camera = camera_frame.as_deref().unwrap_or(&msg.header.frame_id);
            let body = body_frame.lock().unwrap_or_else(|e| e.into_inner()).clone();
            body.ok_or_else(|| "no odometry yet to name the body frame".to_owned())
                .and_then(|body| {
                    let found = transforms
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .lookup(camera, &body);
                    tf_missing = found.is_none();
                    found.ok_or_else(|| format!("no /tf_static transform {camera} <- {body}"))
                })
        } else {
            Ok(localizer.camera_from_imu().clone())
        };
        if tf_missing {
            if let Some(listener) = tf_static.as_mut() {
                match listener.missing(&mut ros) {
                    Ok(true) => {
                        eprintln!("visloc_map_localize_node: subscribed to /tf_static again")
                    }
                    Ok(false) => {}
                    Err(error) => eprintln!("visloc_map_localize_node: {error}"),
                }
            }
        }
        let features = camera_from_body.and_then(|camera_from_body| {
            decode_luma(&msg)
                .and_then(|luma| luma.to_basalt())
                .map_err(|error| format!("image decode: {error}"))
                .and_then(|image| localizer.extract(&image))
                .map(|features| (features, camera_from_body))
        });
        let mut fix = match features {
            Ok((features, camera_from_body)) => {
                // The odometry of this frame usually arrives while the
                // features are extracted; wait for it briefly rather than
                // lose the prior.
                let prior = wait_for_prior(&core, &odom_arrived, stamp.to_nanos(), prior_wait);
                localizer.localize_features_with(&features, prior.as_ref(), &camera_from_body)
            }
            Err(failure) => Fix::failed(failure),
        };
        fix.elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        if fix.matching == Some(Matching::Projection) {
            projected += 1;
        }
        let (map_pose, pending, excluded) = {
            let mut core = core.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(map_from_body) = &fix.map_from_body {
                localized += 1;
                if let Some(file) = dump_inliers.as_mut() {
                    let rows: String = fix
                        .inlier_matches
                        .iter()
                        .map(|(p, id)| format!("{},{:.2},{:.2},{id}\n", stamp.to_nanos(), p.x, p.y))
                        .collect();
                    if let Err(error) = std::io::Write::write_all(file, rows.as_bytes())
                        .and_then(|()| std::io::Write::flush(file))
                    {
                        eprintln!("visloc_map_localize_node: dump_inliers: {error}");
                    }
                }
                publish_logged(
                    &fix_pub,
                    map_fix_msg(stamp.to_nanos(), map_from_body, &map_frame, &options),
                    "map_fix",
                );
                let events =
                    core.add_scored_fix(stamp.to_nanos(), map_from_body.clone(), fix.inliers);
                log_events(&events);
            } else if let Some(map_from_body) = &fix.weak_map_from_body {
                let events =
                    core.add_scored_weak_fix(stamp.to_nanos(), map_from_body.clone(), fix.inliers);
                log_events(&events);
            }
            (
                core.latest_map_pose(),
                core.pending_fixes(),
                core.fusion_excluded_edges(),
            )
        };
        let mut diagnostics =
            map_diagnostics_msg(&fix, stamp, map_pose.as_ref(), pending, &node_name);
        if let (Some(excluded), Some(status)) = (&excluded, diagnostics.status.first_mut()) {
            // Per odometry (odom, odom2): stretches the source check left out.
            status.values.push(KeyValue {
                key: "fusion_excluded_edges".into(),
                value: format!("{excluded:?}"),
            });
        }
        publish_logged(&diagnostics_pub, diagnostics, "diagnostics");
        if attempts <= 3 || attempts.is_multiple_of(50) {
            eprintln!(
                "visloc_map_localize_node: attempt {attempts}: localized={} inliers={} \
                 matches={} keypoints={} matching={} {:.1} ms (localized {localized}/{attempts}, \
                 {projected} by projection, dropped images {}, odom messages {}, map pose {}){}",
                fix.map_from_body.is_some(),
                fix.inliers,
                fix.matches,
                fix.keypoints,
                fix.matching.map_or("none", |m| m.name()),
                fix.elapsed_ms,
                queue.dropped(),
                odom_messages.load(Ordering::Relaxed),
                map_pose.is_some(),
                fix.failure
                    .as_ref()
                    .map(|reason| format!(" reason={reason}"))
                    .unwrap_or_default(),
            );
        }
    }
}

/// The corrected odometry's prediction of the body pose at `stamp_ns`,
/// waiting up to `timeout` for the odometry of that stamp. `None` before the
/// correction is set, or
/// when the odometry does not arrive in time.
fn wait_for_prior(
    core: &Mutex<OdomCorrector>,
    odom_arrived: &Condvar,
    stamp_ns: i64,
    timeout: Duration,
) -> Option<visloc_core::geometry::SE3> {
    let deadline = Instant::now() + timeout;
    let mut core = core.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        match core.prior(stamp_ns) {
            OdomLookup::Pose(prior) => return Some(prior),
            OdomLookup::Unavailable => return None,
            OdomLookup::NotYet => {
                let now = Instant::now();
                if now >= deadline {
                    return None;
                }
                core = odom_arrived
                    .wait_timeout(core, deadline - now)
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
        }
    }
}

/// Logs changes of the correction (initialization, rejections, resets and
/// fixes with no odometry); accepted fixes are the quiet normal case.
fn log_events(events: &[FixEvent]) {
    for event in events {
        match event {
            FixEvent::Applied { stamp_ns, outcome } => {
                let name = fix_outcome_name(*outcome);
                if !matches!(name, "accepted" | "pending" | "unused") {
                    eprintln!("visloc_map_localize_node: fix at {stamp_ns} ns {name}");
                }
            }
            FixEvent::Dropped { stamp_ns } => eprintln!(
                "visloc_map_localize_node: fix at {stamp_ns} ns dropped: no odometry for its stamp"
            ),
        }
    }
}
