//! `visloc_localize_node`: localize camera images against a COLMAP map.
//!
//! Loads a COLMAP model plus landmark descriptor store at start, subscribes
//! `image` (`sensor_msgs/Image`) and optionally `camera_info`, and publishes
//! `pose` (`geometry_msgs/PoseWithCovarianceStamped`), `inlier_count`
//! (`std_msgs/Int32`) and `/diagnostics` (`diagnostic_msgs/DiagnosticArray`).
//! Run with `--help` for parameters.

use std::path::PathBuf;
use std::process;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use visloc_core::types::Camera;
use visloc_ros2::dds::{output_qos, publish_logged, sensor_qos, spawn_receiver, RosNode};
use visloc_ros2::image::decode_luma;
use visloc_ros2::localize::{camera_from_info, diagnostics_msg, LocalizeCore, LocalizeOptions};
use visloc_ros2::msgs::{CameraInfo, DiagnosticArray, Image, Int32, PoseWithCovarianceStamped};
use visloc_ros2::params::NodeArgs;
use visloc_ros2::queue::DropOldestQueue;
use visloc_vision::features::sift::SiftNormalization;

const USAGE: &str = "\
visloc_localize_node -- map-based visual localization over ROS 2 (pure-Rust DDS)

USAGE:
  visloc_localize_node --map <colmap_model_dir> --descriptors <landmark_descriptors.txt> [options]

PARAMETERS (--name value  or  -p name:=value):
  map                    COLMAP model dir (cameras/images/points3D .bin or .txt)  [required]
  descriptors            landmark descriptor store (<id> <floats...> per line)    [optional]
  camera_id              map camera used when no CameraInfo is received          [smallest id]
  use_camera_info        use intrinsics from camera_info when available          [true]
  map_frame              header.frame_id of the published pose                   [map]
  min_inliers            minimum PnP inliers for success                         [12]
  ransac_iterations      PnP RANSAC iterations                                    [128]
  reprojection_threshold PnP inlier threshold in pixels                          [4.0]
  ratio                  descriptor ratio test (<= 0 disables)                   [0.8]
  sift_max_keypoints     SIFT keypoints per image                                 [4000]
  sift_root              RootSIFT (L1-root) descriptor normalization              [false]
  position_stddev        1-sigma position in the published covariance (m)        [0.1]
  orientation_stddev     1-sigma orientation in the published covariance (rad)   [0.05]
  image_queue            images buffered while localizing (drop-oldest)          [1]
  domain_id              DDS domain (default: $ROS_DOMAIN_ID or 0)
  node_name              node name                                               [visloc_localize]

TOPICS (remap with --remap from=to or -r from:=to):
  in : image (sensor_msgs/Image), camera_info (sensor_msgs/CameraInfo)
  out: pose (geometry_msgs/PoseWithCovarianceStamped, camera pose in map_frame),
       inlier_count (std_msgs/Int32), /diagnostics (diagnostic_msgs/DiagnosticArray)
";

fn main() {
    if let Err(error) = run() {
        eprintln!("visloc_localize_node: {error}");
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
    let descriptors = args.string("descriptors").map(PathBuf::from);
    let camera_id = args.parsed::<u64>("camera_id")?;
    let use_camera_info = args.flag("use_camera_info", true)?;
    let map_frame = args.string_or("map_frame", "map");
    let mut options = LocalizeOptions::default();
    options.config.min_inliers = args.parsed_or("min_inliers", options.config.min_inliers)?;
    options.config.ransac_iterations =
        args.parsed_or("ransac_iterations", options.config.ransac_iterations)?;
    options.config.reprojection_threshold = args.parsed_or(
        "reprojection_threshold",
        options.config.reprojection_threshold,
    )?;
    let ratio = args.parsed_or("ratio", options.config.ratio.unwrap_or(0.0))?;
    options.config.ratio = (ratio > 0.0).then_some(ratio);
    options.sift.max_keypoints =
        args.parsed_or("sift_max_keypoints", options.sift.max_keypoints)?;
    if args.flag("sift_root", false)? {
        options.sift.normalization = SiftNormalization::L1Root;
    }
    options.position_stddev_m = args.parsed_or("position_stddev", options.position_stddev_m)?;
    options.orientation_stddev_rad =
        args.parsed_or("orientation_stddev", options.orientation_stddev_rad)?;
    let image_queue = args.parsed_or("image_queue", 1usize)?;
    let domain_id = args.domain_id()?;
    let node_name = args
        .node_name
        .clone()
        .unwrap_or_else(|| args.string_or("node_name", "visloc_localize"));
    let namespace = args.namespace.clone().unwrap_or_else(|| "/".into());
    let image_topic = args.topic("image");
    let info_topic = args.topic("camera_info");
    let pose_topic = args.topic("pose");
    let inlier_topic = args.topic("inlier_count");
    let diagnostics_topic = args.topic("/diagnostics");
    args.finish()?;

    let core = LocalizeCore::load(&map_dir, descriptors.as_deref(), camera_id, options)?;
    eprintln!(
        "visloc_localize_node: map {} with {} cameras, {} landmarks, {} descriptors; default camera {}",
        map_dir.display(),
        core.map().cameras.len(),
        core.map().landmarks.len(),
        core.descriptor_count(),
        core.default_camera().id
    );

    let mut ros = RosNode::new(domain_id, &namespace, &node_name)?;
    let image_sub = ros.subscription::<Image>(&image_topic, sensor_qos(2))?;
    let pose_pub = ros.publisher::<PoseWithCovarianceStamped>(&pose_topic, output_qos(10))?;
    let inlier_pub = ros.publisher::<Int32>(&inlier_topic, output_qos(10))?;
    let diagnostics_pub = ros.publisher::<DiagnosticArray>(&diagnostics_topic, output_qos(10))?;

    let info_camera: Arc<Mutex<Option<Camera>>> = Arc::new(Mutex::new(None));
    if use_camera_info {
        let info_sub = ros.subscription::<CameraInfo>(&info_topic, sensor_qos(1))?;
        let info_camera = Arc::clone(&info_camera);
        let camera_id = core.default_camera().id;
        let mut warned = false;
        spawn_receiver(
            "camera-info-rx",
            info_sub,
            move |msg: CameraInfo| match camera_from_info(&msg, camera_id) {
                Ok(camera) => {
                    *info_camera.lock().unwrap_or_else(|e| e.into_inner()) = Some(camera);
                }
                Err(error) if !warned => {
                    eprintln!("visloc_localize_node: ignoring camera_info: {error}");
                    warned = true;
                }
                Err(_) => {}
            },
        )?;
    }

    let queue = Arc::new(DropOldestQueue::<Image>::new(image_queue));
    {
        let queue = Arc::clone(&queue);
        spawn_receiver("image-rx", image_sub, move |msg: Image| {
            queue.push(msg);
        })?;
    }
    eprintln!(
        "visloc_localize_node: domain {domain_id}, node {}; \
         in: {image_topic}{}; out: {pose_topic} {inlier_topic} {diagnostics_topic}",
        visloc_ros2::dds::qualified_name(&namespace, &node_name),
        if use_camera_info {
            format!(" {info_topic}")
        } else {
            String::new()
        }
    );

    let mut localized = 0u64;
    let mut attempts = 0u64;
    loop {
        let Some(msg) = queue.pop_timeout(Duration::from_secs(1)) else {
            continue;
        };
        attempts += 1;
        let stamp = msg.header.stamp;
        let outcome = match decode_luma(&msg) {
            Ok(luma) => {
                let camera = info_camera
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone();
                core.localize(&luma, camera.as_ref())
            }
            Err(error) => visloc_ros2::localize::LocalizeOutcome {
                success: false,
                camera_to_map: None,
                feature_count: 0,
                match_count: 0,
                correspondence_count: 0,
                inlier_count: 0,
                reprojection_error: None,
                failure: Some(format!("image decode: {error}")),
                elapsed_ms: 0.0,
            },
        };
        if let Some(pose) = core.pose_msg(&outcome, stamp, &map_frame) {
            localized += 1;
            publish_logged(&pose_pub, pose, "pose");
        }
        publish_logged(
            &inlier_pub,
            Int32 {
                data: i32::try_from(outcome.inlier_count).unwrap_or(i32::MAX),
            },
            "inlier_count",
        );
        publish_logged(
            &diagnostics_pub,
            diagnostics_msg(&outcome, stamp, &node_name),
            "diagnostics",
        );
        if attempts <= 3 || attempts.is_multiple_of(50) {
            eprintln!(
                "visloc_localize_node: frame {attempts}: success={} inliers={} matches={} \
                 features={} {:.1} ms (localized {localized}/{attempts}, dropped {}){}",
                outcome.success,
                outcome.inlier_count,
                outcome.match_count,
                outcome.feature_count,
                outcome.elapsed_ms,
                queue.dropped(),
                outcome
                    .failure
                    .as_ref()
                    .map(|reason| format!(" reason={reason}"))
                    .unwrap_or_default(),
            );
        }
    }
}
