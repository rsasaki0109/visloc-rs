//! `visloc_map_localize_node`: robot-style localization against a saved
//! Basalt map, anchored to VIO odometry.
//!
//! Loads a map saved by `basalt_euroc_online_slam_demo --save-map`,
//! subscribes `image` (the camera the map's localization uses, raw) and
//! `odom` (`nav_msgs/Odometry` of the gravity-aligned VIO, e.g.
//! `visloc_vio_node`), and publishes `/tf` (`map -> odom`, REP-105),
//! `map_pose` (the body pose in the map,
//! `geometry_msgs/PoseWithCovarianceStamped`; not `pose`, which
//! `visloc_vio_node` already publishes) and `/diagnostics`. Run with `--help` for parameters.

use std::path::PathBuf;
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use visloc_ros2::dds::{output_qos, publish_logged, sensor_qos, spawn_receiver, RosNode};
use visloc_ros2::image::decode_luma;
use visloc_ros2::map_localize::{
    fix_outcome_name, load_basalt_map, map_diagnostics_msg, map_pose_msg, map_to_odom_tf,
    se3_from_ros, AnchorCore, Fix, FixEvent, MapLocalizeOptions, MapLocalizer,
};
use visloc_ros2::msgs::{DiagnosticArray, Image, Odometry, PoseWithCovarianceStamped, TfMessage};
use visloc_ros2::params::NodeArgs;
use visloc_ros2::queue::DropOldestQueue;

const USAGE: &str = "\
visloc_map_localize_node -- localization against a saved Basalt map, anchored to VIO odometry

USAGE:
  visloc_map_localize_node --map <dir from basalt_euroc_online_slam_demo --save-map> [options]

PARAMETERS (--name value  or  -p name:=value):
  map                    saved map directory (map.json, sparse/, landmark_descriptors.txt) [required]
  camera                 Basalt camera index of the image topic                       [0]
  map_frame              frame of the map; parent of the published map -> odom       [map]
  publish_tf             publish /tf map_frame -> odometry frame                      [true]
  localize_every         localize every N-th received image                           [5]
  min_inliers            minimum PnP inliers to accept a fix                          [30]
  max_hamming            largest ORB Hamming distance of a match                      [70]
  ratio                  nearest / second-nearest ratio (<= 0 disables)               [0.8]
  ransac_iterations      P3P RANSAC iterations                                        [1000]
  reprojection_threshold PnP inlier threshold in pixels                               [3.0]
  max_points             ORB keypoints per image                                      [the map's]
  position_stddev        published 1-sigma position at a fix (m)                      [0.03]
  position_stddev_per_m  ... plus this per metre of VIO travel since the fix          [0.025]
  orientation_stddev     published 1-sigma orientation (rad)                          [0.02]
  image_queue            images buffered while localizing (drop-oldest)              [1]
  domain_id              DDS domain (default: $ROS_DOMAIN_ID or 0)
  node_name              node name                                                    [visloc_map_localize]

TOPICS (remap with --remap from=to or -r from:=to):
  in : image (sensor_msgs/Image, raw), odom (nav_msgs/Odometry, gravity-aligned VIO;
       its header.frame_id and child_frame_id name the odometry and body frames)
  out: /tf (map_frame -> odometry frame), map_pose (geometry_msgs/PoseWithCovarianceStamped,
       body pose in map_frame, once anchored), /diagnostics (diagnostic_msgs/DiagnosticArray)
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
    let publish_tf = args.flag("publish_tf", true)?;
    let localize_every = args.parsed_or("localize_every", 5u64)?.max(1);
    let mut options = MapLocalizeOptions::default();
    options.min_inliers = args.parsed_or("min_inliers", options.min_inliers)?;
    options.max_hamming = args.parsed_or("max_hamming", options.max_hamming)?;
    let ratio = args.parsed_or("ratio", options.ratio.unwrap_or(0.0))?;
    options.ratio = (ratio > 0.0).then_some(ratio);
    options.ransac_iterations = args.parsed_or("ransac_iterations", options.ransac_iterations)?;
    options.reprojection_px = args.parsed_or("reprojection_threshold", options.reprojection_px)?;
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
    let pose_topic = args.topic("map_pose");
    let diagnostics_topic = args.topic("/diagnostics");
    args.finish()?;

    let map = load_basalt_map(&map_dir)?;
    eprintln!(
        "visloc_map_localize_node: map {} with {} images, {} landmarks, {} descriptors",
        map_dir.display(),
        map.map.keyframes.len(),
        map.map.landmarks.len(),
        map.descriptors.len(),
    );
    let localizer = MapLocalizer::new(map, camera, &options)?;
    let core = Arc::new(Mutex::new(AnchorCore::new(&options)));

    let mut ros = RosNode::new(domain_id, &namespace, &node_name)?;
    let image_sub = ros.subscription::<Image>(&image_topic, sensor_qos(2))?;
    let odom_sub = ros.subscription::<Odometry>(&odom_topic, sensor_qos(50))?;
    let tf_pub = ros.publisher::<TfMessage>(&tf_topic, output_qos(10))?;
    let pose_pub = ros.publisher::<PoseWithCovarianceStamped>(&pose_topic, output_qos(10))?;
    let diagnostics_pub = ros.publisher::<DiagnosticArray>(&diagnostics_topic, output_qos(10))?;

    // Odometry: pair with fixes, then publish map -> odom and the map pose.
    let odom_messages = Arc::new(AtomicU64::new(0));
    {
        let core = Arc::clone(&core);
        let odom_messages = Arc::clone(&odom_messages);
        let map_frame = map_frame.clone();
        let options = options.clone();
        spawn_receiver("odom-rx", odom_sub, move |msg: Odometry| {
            odom_messages.fetch_add(1, Ordering::Relaxed);
            let stamp_ns = msg.header.stamp.to_nanos();
            let (events, anchored) = core
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .add_odom(stamp_ns, se3_from_ros(&msg.pose.pose));
            log_events(&events);
            if let Some(anchored) = anchored {
                if publish_tf {
                    publish_logged(
                        &tf_pub,
                        map_to_odom_tf(&anchored, &map_frame, &msg.header.frame_id),
                        "tf",
                    );
                }
                let mut pose = map_pose_msg(&anchored, &map_frame, &options);
                pose.header.stamp = msg.header.stamp;
                publish_logged(&pose_pub, pose, "map_pose");
            }
        })?;
    }

    // Images: keep every N-th, localize the latest one when free.
    let queue = Arc::new(DropOldestQueue::<Image>::new(image_queue));
    {
        let queue = Arc::clone(&queue);
        let mut received = 0u64;
        spawn_receiver("image-rx", image_sub, move |msg: Image| {
            if received.is_multiple_of(localize_every) {
                queue.push(msg);
            }
            received += 1;
        })?;
    }
    eprintln!(
        "visloc_map_localize_node: domain {domain_id}, node {}; in: {image_topic} {odom_topic}; \
         out: {}{pose_topic} {diagnostics_topic}; localizing every {localize_every} image(s)",
        visloc_ros2::dds::qualified_name(&namespace, &node_name),
        if publish_tf {
            format!("{tf_topic} ")
        } else {
            String::new()
        },
    );

    let (mut attempts, mut localized) = (0u64, 0u64);
    loop {
        let Some(msg) = queue.pop_timeout(Duration::from_secs(1)) else {
            continue;
        };
        attempts += 1;
        let stamp = msg.header.stamp;
        let fix = match decode_luma(&msg).and_then(|luma| luma.to_basalt()) {
            Ok(image) => localizer.localize(&image),
            Err(error) => Fix {
                map_from_body: None,
                keypoints: 0,
                matches: 0,
                inliers: 0,
                elapsed_ms: 0.0,
                failure: Some(format!("image decode: {error}")),
            },
        };
        let (anchored, pending) = {
            let mut core = core.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(map_from_body) = &fix.map_from_body {
                localized += 1;
                let events = core.add_fix(stamp.to_nanos(), map_from_body.clone());
                log_events(&events);
            }
            (core.latest_anchored(), core.pending_fixes())
        };
        publish_logged(
            &diagnostics_pub,
            map_diagnostics_msg(&fix, stamp, anchored.as_ref(), pending, &node_name),
            "diagnostics",
        );
        if attempts <= 3 || attempts.is_multiple_of(50) {
            eprintln!(
                "visloc_map_localize_node: attempt {attempts}: localized={} inliers={} \
                 matches={} keypoints={} {:.1} ms (localized {localized}/{attempts}, \
                 dropped images {}, odom messages {}, anchored {}){}",
                fix.map_from_body.is_some(),
                fix.inliers,
                fix.matches,
                fix.keypoints,
                fix.elapsed_ms,
                queue.dropped(),
                odom_messages.load(Ordering::Relaxed),
                anchored.is_some(),
                fix.failure
                    .as_ref()
                    .map(|reason| format!(" reason={reason}"))
                    .unwrap_or_default(),
            );
        }
    }
}

/// Logs anchor state changes (initialization, rejections, re-anchoring and
/// fixes with no odometry); accepted fixes are the quiet normal case.
fn log_events(events: &[FixEvent]) {
    for event in events {
        match event {
            FixEvent::Applied { stamp_ns, outcome } => {
                let name = fix_outcome_name(*outcome);
                if name != "accepted" && name != "pending" {
                    eprintln!("visloc_map_localize_node: fix at {stamp_ns} ns {name}");
                }
            }
            FixEvent::Dropped { stamp_ns } => eprintln!(
                "visloc_map_localize_node: fix at {stamp_ns} ns dropped: no odometry for its stamp"
            ),
        }
    }
}
