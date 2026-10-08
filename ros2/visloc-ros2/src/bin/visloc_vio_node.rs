//! `visloc_vio_node`: stereo-inertial VIO (Basalt port) as a ROS 2 node.
//!
//! Subscribes `left/image_raw`, `right/image_raw` (`sensor_msgs/Image`) and
//! `imu` (`sensor_msgs/Imu`); publishes `odom` (`nav_msgs/Odometry`), `pose`
//! (`geometry_msgs/PoseStamped`), `path` (`nav_msgs/Path`) and optionally
//! `/tf`. Run with `--help` for parameters.

use std::process;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use visloc_basalt::RawU16Image;
use visloc_ros2::dds::{output_qos, publish_logged, sensor_qos, spawn_receiver, RosNode};
use visloc_ros2::image::decode_luma;
use visloc_ros2::msgs::{Image, Imu, Odometry, Path, PoseStamped, TfMessage};
use visloc_ros2::params::NodeArgs;
use visloc_ros2::sync::VioInputs;
use visloc_ros2::vio::{
    imu_sample_from_msg, odometry_msg, pose_stamped_msg, tf_msg, PathAccumulator, VioCore,
    VioFrames, VioOptions,
};

const USAGE: &str = "\
visloc_vio_node -- Basalt stereo-inertial VIO over ROS 2 (pure-Rust DDS)

USAGE:
  visloc_vio_node --calibration <calib.json> --config <config.json> [options]
  visloc_vio_node --ros-args -p calibration:=<..> -p config:=<..> [-r from:=to ...]

PARAMETERS (--name value  or  -p name:=value):
  calibration            Basalt calibration JSON (same file as the EuRoC demo)   [required]
  config                 Basalt VIO config JSON (same file as the EuRoC demo)    [required]
  odom_frame             header.frame_id of the outputs                          [odom]
  body_frame             child_frame_id (Basalt body = IMU frame)                [imu_link]
  publish_tf             publish odom_frame -> body_frame on /tf                 [true]
  publish_path           publish nav_msgs/Path                                   [true]
  path_max_length        poses kept in the published path (drop-oldest)          [2000]
  stereo_sync_tolerance_ms  max |t_left - t_right| for a stereo pair             [2.0]
  frame_queue            paired frames buffered for the estimator (drop-oldest)  [4]
  imu_buffer             IMU samples buffered (drop-oldest)                      [4000]
  image_qos_depth        subscription history depth for images                  [5]
  imu_qos_depth          subscription history depth for IMU                     [400]
  urgent_keyframes       demo's urgent-keyframe spacing defaults                 [true]
  imu_seed_klt           seed KLT with the IMU rotation                          [false]
  domain_id              DDS domain (default: $ROS_DOMAIN_ID or 0)
  node_name              node name                                              [visloc_vio]

TOPICS (remap with --remap from=to or -r from:=to):
  in : left/image_raw, right/image_raw (sensor_msgs/Image), imu (sensor_msgs/Imu)
  out: odom (nav_msgs/Odometry), pose (geometry_msgs/PoseStamped),
       path (nav_msgs/Path), /tf (tf2_msgs/TFMessage)
";

fn main() {
    if let Err(error) = run() {
        eprintln!("visloc_vio_node: {error}");
        process::exit(1);
    }
}

struct Shared {
    inputs: Mutex<VioInputs<RawU16Image>>,
    ready: Condvar,
}

impl Shared {
    fn with<R>(&self, f: impl FnOnce(&mut VioInputs<RawU16Image>) -> R) -> R {
        let mut guard = self.inputs.lock().unwrap_or_else(|e| e.into_inner());
        let result = f(&mut guard);
        drop(guard);
        self.ready.notify_one();
        result
    }
}

/// Logs at most once per `period` per call site.
struct RateLimitedLog {
    last: Option<Instant>,
    period: Duration,
    suppressed: u64,
}

impl RateLimitedLog {
    const fn new(period: Duration) -> Self {
        Self {
            last: None,
            period,
            suppressed: 0,
        }
    }

    fn log(&mut self, message: impl FnOnce() -> String) {
        let now = Instant::now();
        if self
            .last
            .is_none_or(|last| now.duration_since(last) >= self.period)
        {
            let suffix = if self.suppressed > 0 {
                format!(" ({} similar messages suppressed)", self.suppressed)
            } else {
                String::new()
            };
            eprintln!("visloc_vio_node: {}{suffix}", message());
            self.last = Some(now);
            self.suppressed = 0;
        } else {
            self.suppressed += 1;
        }
    }
}

fn run() -> Result<(), String> {
    let args = NodeArgs::parse(std::env::args().skip(1))?;
    if args.help {
        print!("{USAGE}");
        return Ok(());
    }
    let calibration_path = args.required("calibration")?;
    let config_path = args.required("config")?;
    let frames = VioFrames {
        odom_frame: args.string_or("odom_frame", "odom"),
        body_frame: args.string_or("body_frame", "imu_link"),
    };
    let publish_tf = args.flag("publish_tf", true)?;
    let publish_path = args.flag("publish_path", true)?;
    let path_max_length = args.parsed_or("path_max_length", 2000usize)?;
    let tolerance_ms = args.parsed_or("stereo_sync_tolerance_ms", 2.0f64)?;
    let frame_queue = args.parsed_or("frame_queue", 4usize)?;
    let imu_buffer = args.parsed_or("imu_buffer", 4000usize)?;
    let image_depth = args.parsed_or("image_qos_depth", 5usize)?;
    let imu_depth = args.parsed_or("imu_qos_depth", 400usize)?;
    let options = VioOptions {
        urgent_keyframes: args.flag("urgent_keyframes", true)?,
        imu_seed_klt: args.flag("imu_seed_klt", false)?,
    };
    let domain_id = args.domain_id()?;
    let node_name = args
        .node_name
        .clone()
        .unwrap_or_else(|| args.string_or("node_name", "visloc_vio"));
    let namespace = args.namespace.clone().unwrap_or_else(|| "/".into());
    let left_topic = args.topic("left/image_raw");
    let right_topic = args.topic("right/image_raw");
    let imu_topic = args.topic("imu");
    let odom_topic = args.topic("odom");
    let pose_topic = args.topic("pose");
    let path_topic = args.topic("path");
    let tf_topic = args.topic("/tf");
    args.finish()?;

    let read = |path: &str| {
        std::fs::read_to_string(path).map_err(|error| format!("reading {path}: {error}"))
    };
    let mut core = VioCore::from_json(&read(&calibration_path)?, &read(&config_path)?, &options)?;
    let expected = [core.expected_resolution(0), core.expected_resolution(1)];
    let shared = Arc::new(Shared {
        inputs: Mutex::new(VioInputs::new(
            (tolerance_ms * 1e6).round() as i64,
            8,
            frame_queue,
            imu_buffer,
            core.cam_time_offset_ns(),
        )),
        ready: Condvar::new(),
    });

    let mut ros = RosNode::new(domain_id, &namespace, &node_name)?;
    let left_sub = ros.subscription::<Image>(&left_topic, sensor_qos(image_depth))?;
    let right_sub = ros.subscription::<Image>(&right_topic, sensor_qos(image_depth))?;
    let imu_sub = ros.subscription::<Imu>(&imu_topic, sensor_qos(imu_depth))?;
    let odom_pub = ros.publisher::<Odometry>(&odom_topic, output_qos(10))?;
    let pose_pub = ros.publisher::<PoseStamped>(&pose_topic, output_qos(10))?;
    let path_pub = if publish_path {
        Some(ros.publisher::<Path>(&path_topic, output_qos(1))?)
    } else {
        None
    };
    let tf_pub = if publish_tf {
        Some(ros.publisher::<TfMessage>(&tf_topic, output_qos(100))?)
    } else {
        None
    };

    for (side, subscription) in [(0usize, left_sub), (1usize, right_sub)] {
        let shared = Arc::clone(&shared);
        let expected = expected[side];
        let mut errors = RateLimitedLog::new(Duration::from_secs(5));
        spawn_receiver(&format!("cam{side}-rx"), subscription, move |msg: Image| {
            let stamp_ns = msg.header.stamp.to_nanos();
            let image = decode_luma(&msg).and_then(|luma| {
                if let Some(expected) = expected {
                    if (luma.width, luma.height) != expected {
                        return Err(visloc_ros2::image::ImageConversionError::SizeMismatch {
                            expected,
                            actual: (luma.width, luma.height),
                        });
                    }
                }
                luma.to_basalt()
            });
            match image {
                Ok(image) if side == 0 => shared.with(|inputs| inputs.push_left(stamp_ns, image)),
                Ok(image) => shared.with(|inputs| inputs.push_right(stamp_ns, image)),
                Err(error) => errors.log(|| format!("dropping cam{side} image: {error}")),
            }
        })?;
    }
    {
        let shared = Arc::clone(&shared);
        let mut errors = RateLimitedLog::new(Duration::from_secs(5));
        spawn_receiver(
            "imu-rx",
            imu_sub,
            move |msg: Imu| match imu_sample_from_msg(&msg) {
                Some(sample) => {
                    if !shared.with(|inputs| inputs.push_imu(sample)) {
                        errors.log(|| "dropping non-increasing IMU stamp".into());
                    }
                }
                None => errors.log(|| "dropping non-finite IMU sample".into()),
            },
        )?;
    }

    eprintln!(
        "visloc_vio_node: domain {domain_id}, node {}; \
         in: {left_topic} {right_topic} {imu_topic}; out: {odom_topic} {pose_topic}{}{}",
        visloc_ros2::dds::qualified_name(&namespace, &node_name),
        if publish_path {
            format!(" {path_topic}")
        } else {
            String::new()
        },
        if publish_tf {
            format!(" {tf_topic}")
        } else {
            String::new()
        },
    );

    let mut path = PathAccumulator::new(path_max_length);
    let mut processed = 0u64;
    let mut waiting_log = RateLimitedLog::new(Duration::from_secs(5));
    let mut error_log = RateLimitedLog::new(Duration::from_secs(2));
    let mut last_status = Instant::now();
    loop {
        let packet = {
            let guard = shared.inputs.lock().unwrap_or_else(|e| e.into_inner());
            let (mut guard, timeout) = shared
                .ready
                .wait_timeout_while(guard, Duration::from_secs(1), |inputs| {
                    !inputs.front_ready()
                })
                .unwrap_or_else(|e| e.into_inner());
            if timeout.timed_out() && guard.queued_frames() > 0 {
                waiting_log
                    .log(|| "stereo frames queued but IMU has not reached their stamp yet".into());
            }
            guard.pop_ready()
        };
        if last_status.elapsed() >= Duration::from_secs(10) {
            let stats = shared.with(|inputs| inputs.stats());
            eprintln!(
                "visloc_vio_node: processed={processed} pairs={} dropped_behind={} \
                 sync_dropped={} imu_out_of_order={} imu_evicted={}",
                stats.pairs,
                stats.frames_dropped_behind,
                stats.sync_dropped,
                stats.imu_out_of_order,
                stats.imu_evicted
            );
            last_status = Instant::now();
        }
        let Some(packet) = packet else {
            continue;
        };
        match core.process(packet) {
            Ok(estimate) => {
                processed += 1;
                if processed == 1 {
                    eprintln!("visloc_vio_node: first pose published");
                }
                publish_logged(&odom_pub, odometry_msg(&estimate, &frames), "odom");
                let pose = pose_stamped_msg(&estimate, &frames);
                if let Some(path_pub) = &path_pub {
                    path.push(pose.clone());
                    publish_logged(path_pub, path.msg(&frames.odom_frame), "path");
                }
                publish_logged(&pose_pub, pose, "pose");
                if let Some(tf_pub) = &tf_pub {
                    publish_logged(tf_pub, tf_msg(&estimate, &frames), "tf");
                }
            }
            Err(error) => {
                error_log.log(|| format!("{error}"));
                if error.needs_reset {
                    core.reset()?;
                    shared.with(|inputs| inputs.reset_stream());
                    path.clear();
                    eprintln!("visloc_vio_node: estimator reset; re-initializing from IMU");
                }
            }
        }
    }
}
