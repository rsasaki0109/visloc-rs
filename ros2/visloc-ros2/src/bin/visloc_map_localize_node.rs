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
//! `visloc_vio_node` already publishes), `map_fix` (each localization) and
//! `/diagnostics`. Run with `--help` for parameters.

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
    map_to_odom_tf, se3_from_ros, Fix, FixEvent, ImageCadence, MapLocalizeOptions, MapLocalizer,
    Matching, OdomCorrector, OdomLookup,
};
use visloc_ros2::msgs::{
    DiagnosticArray, Header, Image, Odometry, PointCloud2, PoseWithCovarianceStamped, TfMessage,
    Time,
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
  max_hamming            largest ORB Hamming distance of a match                      [70]
  ratio                  nearest / second-nearest ratio (<= 0 disables)               [0.8]
  ransac_iterations      P3P RANSAC iterations                                        [1000]
  reprojection_threshold PnP inlier threshold in pixels                               [3.0]
  projection_radius      with a map pose, match each landmark only with keypoints this
                         close (px) to its projection from the predicted pose; on
                         failure, and with <= 0 always, match map-wide             [20]
  projection_fresh_s     ... only within this many seconds of the newest fix          [5]
  projection_fresh_m     ... and this many metres of odometry travel from it          [2]
  prior_wait_ms          longest wait for the odometry of an image's stamp before
                         matching map-wide instead                                    [300]
  match_threads          threads of the map-wide search                               [1]
  duplicate_radius       map-wide search: landmarks within this many metres of the best
                         one count as the same point in the ratio test (0: off)      [0]
  extra_descriptors      extra descriptors per landmark to match, from the map's
                         landmark_descriptors_extra.txt (0: the main one only)      [all]
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
       /tf_static (tf2_msgs/TFMessage, with extrinsics tf)
  out: /tf (map_frame -> odometry frame), map_pose (geometry_msgs/PoseWithCovarianceStamped,
       body pose in map_frame, after two agreeing fixes), map_fix (geometry_msgs/
       PoseWithCovarianceStamped, the body pose of each image localized with min_inliers,
       e.g. for visloc_wheel_calibration_node), /diagnostics (diagnostic_msgs/DiagnosticArray),
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
    options.max_hamming = args.parsed_or("max_hamming", options.max_hamming)?;
    let ratio = args.parsed_or("ratio", options.ratio.unwrap_or(0.0))?;
    options.ratio = (ratio > 0.0).then_some(ratio);
    options.ransac_iterations = args.parsed_or("ransac_iterations", options.ransac_iterations)?;
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
    let prior_wait = Duration::from_millis(args.parsed_or("prior_wait_ms", 300u64)?);
    options.match_threads = args.parsed_or("match_threads", options.match_threads)?;
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
    let tf_topic = args.topic("/tf");
    let tf_static_topic = args.topic("/tf_static");
    let pose_topic = args.topic("map_pose");
    let fix_topic = args.topic("map_fix");
    let diagnostics_topic = args.topic("/diagnostics");
    let map_cloud_topic = args.topic("map_cloud");
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
    let tf_pub = ros.publisher::<TfMessage>(&tf_topic, output_qos(10))?;
    let pose_pub = ros.publisher::<PoseWithCovarianceStamped>(&pose_topic, output_qos(10))?;
    let fix_pub = ros.publisher::<PoseWithCovarianceStamped>(&fix_topic, output_qos(10))?;
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
            let (events, map_pose) = core
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .add_odom(stamp_ns, se3_from_ros(&msg.pose.pose));
            odom_arrived.notify_all();
            log_events(&events);
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
        let (map_pose, pending) = {
            let mut core = core.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(map_from_body) = &fix.map_from_body {
                localized += 1;
                publish_logged(
                    &fix_pub,
                    map_fix_msg(stamp.to_nanos(), map_from_body, &map_frame, &options),
                    "map_fix",
                );
                let events = core.add_fix(stamp.to_nanos(), map_from_body.clone());
                log_events(&events);
            } else if let Some(map_from_body) = &fix.weak_map_from_body {
                let events = core.add_weak_fix(stamp.to_nanos(), map_from_body.clone());
                log_events(&events);
            }
            (core.latest_map_pose(), core.pending_fixes())
        };
        publish_logged(
            &diagnostics_pub,
            map_diagnostics_msg(&fix, stamp, map_pose.as_ref(), pending, &node_name),
            "diagnostics",
        );
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
