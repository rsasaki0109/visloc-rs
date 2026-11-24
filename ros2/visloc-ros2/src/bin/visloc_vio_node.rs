//! `visloc_vio_node`: stereo- or monocular-inertial VIO (Basalt port) as a ROS 2 node.
//!
//! Subscribes `left/image_raw`, `right/image_raw` (`sensor_msgs/Image`; with
//! `mono:=true` only `left/image_raw`) and
//! `imu` (`sensor_msgs/Imu`); publishes `odom` (`nav_msgs/Odometry`), `pose`
//! (`geometry_msgs/PoseStamped`), `path` (`nav_msgs/Path`) and optionally
//! `/tf`. Run with `--help` for parameters.
//!
//! KLT tracking runs on its own thread, one frame ahead of the estimator
//! (the main thread), as upstream Basalt's optical-flow and VIO threads do.
//!
//! With `mapper:=true` the estimator also feeds Basalt's online mapper (loop
//! closure, global BA) on a third thread, and the node publishes the
//! `map -> odom` correction, `map_pose` and `map_path`; on SIGINT/SIGTERM it
//! runs the mapper's final optimization and can write the corrected
//! trajectory (see [`visloc_ros2::vio_mapper`]).
//!
//! With `lost_min_observations` or `lost_max_speed` set, the node watches
//! for lost tracking ([`visloc_ros2::tracking`]): it then resets the
//! estimator and publishes nothing until tracking holds again. Every
//! estimator reset bumps the count on `odom_reset` (`std_msgs/Int32`,
//! transient-local): the odometry frame starts over.
//!
//! With `wheel_odom_topic` (and `base_frame`) set as well, a lost-tracking
//! reset does not stop the outputs: the pose is carried on by the wheel
//! odometry's motion until tracking holds again, and the new estimator's
//! poses continue from there ([`visloc_ros2::odom_bridge`]). The odometry
//! frame stays the same, so no `odom_reset` is published and the robot can
//! keep driving.
//!
//! Two inputs reach into the estimator itself ([`visloc_ros2::vio_external`],
//! docs/odometry_fusion.md): with `map_prior:=true`, each fix that
//! `visloc_map_localize_node` uses (`odom_fix`) becomes a pose prior on its
//! frame, so the map corrects the VIO's velocity and biases too; with
//! `wheel_velocity:=true` and `wheel_odom_topic`, the wheel odometry's
//! twist becomes a body-velocity factor on every frame.

use std::process;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use nalgebra::Vector3;

use visloc_basalt::mapper_correction::tum_string;
use visloc_basalt::RawU16Image;
use visloc_ros2::dds::latched_qos;
use visloc_ros2::dds::{
    output_qos, publish_logged, sensor_qos, spawn_receiver, RosNode, StaticTfListener,
};
use visloc_ros2::image::decode_luma;
use visloc_ros2::map_localize::se3_from_ros;
use visloc_ros2::msgs::{
    Image, Imu, Int32, Odometry, Path, PoseStamped, PoseWithCovarianceStamped, TfMessage,
};
use visloc_ros2::odom_bridge::{BridgeState, OdomBridge};
use visloc_ros2::params::NodeArgs;
use visloc_ros2::sync::{SensorPacket, VioInputs};
use visloc_ros2::tf_tree::StaticTransforms;
use visloc_ros2::tracking::{TrackingConfig, TrackingMonitor, TrackingVerdict};
use visloc_ros2::vio::{
    imu_sample_from_msg, odometry_msg, pose_stamped_msg, rebase, tf_msg, PathAccumulator,
    TrackedFrame, VioCore, VioError, VioEstimate, VioFrames, VioFrontend, VioOptions,
};
use visloc_ros2::vio_external::{
    map_prior_factor, wheel_velocity_factor, ExternalNoise, FrameStamps,
};
use visloc_ros2::vio_mapper::{
    map_pose_msg, map_to_odom_transform, new_online_mapper, MapperParams, MapperSession,
};

const USAGE: &str = "\
visloc_vio_node -- Basalt stereo- or monocular-inertial VIO over ROS 2 (pure-Rust DDS)

USAGE:
  visloc_vio_node --calibration <calib.json> --config <config.json> [options]
  visloc_vio_node --ros-args -p calibration:=<..> -p config:=<..> [-r from:=to ...]

PARAMETERS (--name value  or  -p name:=value):
  calibration            Basalt calibration JSON (same file as the EuRoC demo)   [required]
  config                 Basalt VIO config JSON (same file as the EuRoC demo)    [required]
  odom_frame             header.frame_id of the outputs                          [odom]
  body_frame             the IMU's frame (Basalt's body); child_frame_id of the
                         outputs unless base_frame is set                        [imu_link]
  base_frame             publish this frame's pose instead (e.g. base_link, REP-105),
                         placed relative to body_frame by /tf_static             [none]
  publish_tf             publish odom_frame -> child frame on /tf                [true]
  publish_path           publish nav_msgs/Path                                   [true]
  path_max_length        poses kept in the published path (drop-oldest)          [2000]
  stereo_sync_tolerance_ms  max |t_left - t_right| for a stereo pair             [2.0]
  mono                   monocular-inertial VIO: subscribe left/image_raw only   [false]
  mono_camera            with mono, the calibration's camera that left/image_raw
                         carries (0 or 1)                                        [0]
  frame_queue            paired frames waiting for the frontend (drop-oldest)    [1]
  frontend_threads       worker threads of the KLT frontend (0: shared with the
                         estimator's pool)                                       [2]
  imu_buffer             IMU samples buffered (drop-oldest)                      [4000]
  image_qos_depth        subscription history depth for images                  [5]
  imu_qos_depth          subscription history depth for IMU                     [400]
  urgent_keyframes       demo's urgent-keyframe spacing defaults                 [true]
  imu_seed_klt           seed KLT with the IMU rotation                          [false]
  max_iterations         LM iterations per frame; overrides config.vio_max_iterations
                         (0: keep the config's)                                  [3]
  mapper                 run Basalt's online mapper (loop closure + global BA)
                         and publish the map_frame -> odom_frame correction      [false]
  map_frame              frame of the corrected outputs (with mapper)           [map]
  mapper_queue           mapper packets queued before new ones are dropped      [64]
  mapper_optimize_every_k  new keyframes between background global BA passes   [100]
  mapper_trajectory_out  on shutdown (SIGINT/SIGTERM), write the loop-closed
                         trajectory of every frame here (TUM, IMU frame)        [none]
  lost_min_observations  lost tracking: fewer tracked observations per frame than
                         this for lost_after_s resets the estimator and stops the
                         outputs until tracking holds for recover_after_s (0: off) [0]
  lost_after_s           ... for this long (s)                                   [0.5]
  recover_after_s        ... and good again for this long (s)                    [1.0]
  lost_max_speed         lost tracking: an estimated speed above this (m/s) (0: off) [0]
  wheel_odom_topic       while tracking is lost, carry the outputs on with this
                         wheel odometry (nav_msgs/Odometry), in the same odometry
                         frame; its child frame is placed relative to the output
                         frame by /tf_static; needs the lost-tracking monitor;
                         not with the mapper (empty: off)                        []
  wheel_velocity         also feed wheel_odom_topic's twist into the estimator as a
                         body velocity on every frame (wheel-inertial odometry);
                         its child frame is placed relative to body_frame by
                         /tf_static; does not need the lost-tracking monitor     [false]
  wheel_velocity_stddev  1-sigma of that velocity (m/s), plus 2% of the speed    [0.02]
  map_prior              feed each fix visloc_map_localize_node used (odom_fix) into
                         the estimator as a pose prior on its frame               [false]
  map_prior_stddev_m     1-sigma position of such a prior (m)                     [0.03]
  map_prior_stddev_deg   1-sigma orientation of such a prior (deg)                [1.0]
  domain_id              DDS domain (default: $ROS_DOMAIN_ID or 0)
  node_name              node name                                              [visloc_vio]

TOPICS (remap with --remap from=to or -r from:=to):
  in : left/image_raw, right/image_raw (sensor_msgs/Image), imu (sensor_msgs/Imu),
       /tf_static (tf2_msgs/TFMessage, with base_frame), with map_prior: odom_fix
       (geometry_msgs/PoseWithCovarianceStamped, from visloc_map_localize_node)
  out: odom (nav_msgs/Odometry), pose (geometry_msgs/PoseStamped),
       path (nav_msgs/Path), /tf (tf2_msgs/TFMessage),
       odom_reset (std_msgs/Int32, estimator resets so far; transient-local)
       with mapper: map_pose (geometry_msgs/PoseStamped), map_path (nav_msgs/Path),
       map_frame -> odom_frame on /tf
";

fn main() {
    if let Err(error) = run() {
        eprintln!("visloc_vio_node: {error}");
        process::exit(1);
    }
}

struct Shared {
    inputs: Mutex<Inputs>,
    ready: Condvar,
    /// Gyro bias of the latest estimate, for the frontend's `imu_seed_klt`.
    gyro_bias: Mutex<Vector3<f64>>,
    /// Frontend tracking times since the last status line.
    frontend_time: Mutex<StageTime>,
}

/// Processing times of one pipeline stage over a status period.
#[derive(Default)]
struct StageTime {
    count: u32,
    total: Duration,
    max: Duration,
}

impl StageTime {
    fn add(&mut self, elapsed: Duration) {
        self.count += 1;
        self.total += elapsed;
        self.max = self.max.max(elapsed);
    }

    /// `avg/max ms` and resets the period.
    fn take(&mut self) -> String {
        let summary = format!(
            "{:.1}/{:.1} ms",
            self.total.as_secs_f64() * 1e3 / f64::from(self.count.max(1)),
            self.max.as_secs_f64() * 1e3
        );
        *self = Self::default();
        summary
    }
}

/// Tracked observations per frame over a status period.
#[derive(Default)]
struct ObservationStats {
    sum: u64,
    min: Option<usize>,
    frames: u64,
}

impl ObservationStats {
    fn add(&mut self, count: usize) {
        self.sum += count as u64;
        self.min = Some(self.min.map_or(count, |m| m.min(count)));
        self.frames += 1;
    }

    /// `min/avg` and resets the period.
    fn take(&mut self) -> String {
        let text = match self.min {
            Some(min) => format!("{min}/{:.0}", self.sum as f64 / self.frames as f64),
            None => "-".into(),
        };
        *self = Self::default();
        text
    }
}

/// Sensor inputs plus the stream generation: a VIO reset bumps the
/// generation together with restarting the input stream, so every packet
/// popped afterwards belongs to the new stream and both threads can tell
/// stale work from current work.
struct Inputs {
    vio: VioInputs<RawU16Image>,
    generation: u64,
}

impl Shared {
    fn with<R>(&self, f: impl FnOnce(&mut VioInputs<RawU16Image>) -> R) -> R {
        let mut guard = self.inputs.lock().unwrap_or_else(|e| e.into_inner());
        let result = f(&mut guard.vio);
        drop(guard);
        self.ready.notify_one();
        result
    }

    /// Restarts the input stream and returns the new generation.
    fn restart_stream(&self) -> u64 {
        let mut guard = self.inputs.lock().unwrap_or_else(|e| e.into_inner());
        guard.vio.reset_stream();
        guard.generation += 1;
        *self.gyro_bias.lock().unwrap_or_else(|e| e.into_inner()) = Vector3::zeros();
        guard.generation
    }
}

/// A wheel odometry pose as the pose of the node's output frame, through
/// /tf_static (`None` until the two frames are connected).
fn wheel_in_output(
    transforms: &Mutex<StaticTransforms>,
    sample: &WheelSample,
    output_body: &str,
) -> Option<visloc_core::geometry::SE3> {
    let wheel_from_output = transforms
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .lookup(&sample.child_frame, output_body)?;
    Some(sample.pose.compose(&wheel_from_output))
}

/// One wheel odometry message, for the lost-tracking bridge.
#[derive(Clone, Debug)]
struct WheelSample {
    seq: u64,
    child_frame: String,
    stamp_ns: i64,
    pose: visloc_core::geometry::SE3,
    velocity_body: Vector3<f64>,
    angular_body: Vector3<f64>,
}

/// Frontend-thread output: a tracked frame, or a tracking error for the
/// estimator thread to handle, tagged with the stream generation.
type FrontendResult = (u64, Result<TrackedFrame, VioError>);

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
    let base_frame = args.string("base_frame").filter(|frame| !frame.is_empty());
    // The frames of the outputs: the base frame's pose when there is one.
    let output_frames = VioFrames {
        odom_frame: frames.odom_frame.clone(),
        body_frame: base_frame
            .clone()
            .unwrap_or_else(|| frames.body_frame.clone()),
    };
    let publish_tf = args.flag("publish_tf", true)?;
    let publish_path = args.flag("publish_path", true)?;
    let path_max_length = args.parsed_or("path_max_length", 2000usize)?;
    let tolerance_ms = args.parsed_or("stereo_sync_tolerance_ms", 2.0f64)?;
    let frame_queue = args.parsed_or("frame_queue", 1usize)?;
    let frontend_threads = args.parsed_or("frontend_threads", 2usize)?;
    let imu_buffer = args.parsed_or("imu_buffer", 4000usize)?;
    let image_depth = args.parsed_or("image_qos_depth", 5usize)?;
    let imu_depth = args.parsed_or("imu_qos_depth", 400usize)?;
    let mapper_params = MapperParams::from_args(&args)?;
    let mono_camera = args
        .flag("mono", false)?
        .then(|| args.parsed_or("mono_camera", 0usize))
        .transpose()?;
    let tracking = TrackingConfig {
        min_observations: args.parsed_or("lost_min_observations", 0usize)?,
        lost_after_s: args.parsed_or("lost_after_s", 0.5f64)?,
        recover_after_s: args.parsed_or("recover_after_s", 1.0f64)?,
        max_speed_m_s: args.parsed_or("lost_max_speed", 0.0f64)?,
    };
    let wheel_odom_topic = args
        .string("wheel_odom_topic")
        .filter(|topic| !topic.is_empty());
    let wheel_velocity = args.flag("wheel_velocity", false)?;
    let map_prior = args.flag("map_prior", false)?;
    let external_noise = ExternalNoise {
        fix_position_m: args.parsed_or("map_prior_stddev_m", 0.03f64)?,
        fix_orientation_rad: args.parsed_or("map_prior_stddev_deg", 1.0f64)?.to_radians(),
        velocity_m_s: args.parsed_or("wheel_velocity_stddev", 0.02f64)?,
        ..ExternalNoise::default()
    };
    if wheel_velocity && wheel_odom_topic.is_none() {
        return Err("wheel_velocity needs wheel_odom_topic".into());
    }
    if wheel_odom_topic.is_some() {
        if !tracking.enabled() && !wheel_velocity {
            return Err(
                "wheel_odom_topic needs the lost-tracking monitor (lost_min_observations \
                 or lost_max_speed)"
                    .into(),
            );
        }
        if mapper_params.is_some() {
            return Err("wheel_odom_topic is not supported with the online mapper".into());
        }
    }
    let options = VioOptions {
        urgent_keyframes: args.flag("urgent_keyframes", true)?,
        imu_seed_klt: args.flag("imu_seed_klt", false)?,
        max_iterations: Some(args.parsed_or("max_iterations", 3u32)?).filter(|&n| n > 0),
        retain_marg_data: mapper_params.is_some(),
        mono_camera,
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
    let tf_static_topic = args.topic("/tf_static");
    let map_pose_topic = args.topic("map_pose");
    let map_path_topic = args.topic("map_path");
    let odom_reset_topic = args.topic("odom_reset");
    let odom_fix_topic = args.topic("odom_fix");
    args.finish()?;

    let read = |path: &str| {
        std::fs::read_to_string(path).map_err(|error| format!("reading {path}: {error}"))
    };
    let core = VioCore::from_json(&read(&calibration_path)?, &read(&config_path)?, &options)?;
    let expected = [core.expected_resolution(0), core.expected_resolution(1)];
    let shared = Arc::new(Shared {
        inputs: Mutex::new(Inputs {
            vio: if mono_camera.is_some() {
                VioInputs::new_mono(frame_queue, imu_buffer, core.cam_time_offset_ns())
            } else {
                VioInputs::new(
                    (tolerance_ms * 1e6).round() as i64,
                    8,
                    frame_queue,
                    imu_buffer,
                    core.cam_time_offset_ns(),
                )
            },
            generation: 0,
        }),
        ready: Condvar::new(),
        gyro_bias: Mutex::new(Vector3::zeros()),
        frontend_time: Mutex::new(StageTime::default()),
    });
    // The mapper is built from the same calibration and config, and again
    // after every VIO reset.
    let mapper_source = mapper_params.as_ref().map(|params| {
        (
            core.calibration().clone(),
            core.config().clone(),
            params.clone(),
        )
    });
    let start_mapper = || {
        mapper_source
            .as_ref()
            .map(|(calibration, config, params)| {
                MapperSession::spawn(
                    new_online_mapper(calibration, config, params)?,
                    params.queue_capacity,
                )
            })
            .transpose()
    };
    let mut mapper = start_mapper()?;
    let (frontend, mut backend) = core.into_stages();

    let mut ros = RosNode::new(domain_id, &namespace, &node_name)?;
    let left_sub = ros.subscription::<Image>(&left_topic, sensor_qos(image_depth))?;
    // A monocular node subscribes to `left/image_raw` only.
    let right_sub = mono_camera
        .is_none()
        .then(|| ros.subscription::<Image>(&right_topic, sensor_qos(image_depth)))
        .transpose()?;
    let imu_sub = ros.subscription::<Imu>(&imu_topic, sensor_qos(imu_depth))?;
    let odom_pub = ros.publisher::<Odometry>(&odom_topic, output_qos(10))?;
    let pose_pub = ros.publisher::<PoseStamped>(&pose_topic, output_qos(10))?;
    let odom_reset_pub = ros.publisher::<Int32>(&odom_reset_topic, latched_qos(1))?;
    publish_logged(&odom_reset_pub, Int32 { data: 0 }, "odom_reset");
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
    let map_pose_pub = mapper_params
        .is_some()
        .then(|| ros.publisher::<PoseStamped>(&map_pose_topic, output_qos(10)))
        .transpose()?;
    let map_path_pub = (mapper_params.is_some() && publish_path)
        .then(|| ros.publisher::<Path>(&map_path_topic, output_qos(1)))
        .transpose()?;
    // Only with the mapper: the first SIGINT/SIGTERM ends the loop below so
    // the mapper can run its final optimization; a second one exits at once.
    // Without it the default signal handling is left alone.
    let shutdown = mapper_params
        .is_some()
        .then(|| {
            let signals = Arc::new(AtomicU32::new(0));
            let handler_signals = Arc::clone(&signals);
            ctrlc::set_handler(move || {
                if handler_signals.fetch_add(1, Ordering::SeqCst) > 0 {
                    eprintln!(
                        "visloc_vio_node: second signal; exiting without the final mapper optimization"
                    );
                    process::exit(130);
                }
                eprintln!(
                    "visloc_vio_node: shutting down after the final mapper optimization \
                     (signal again to skip it)"
                );
            })
            .map(|()| signals)
            .map_err(|error| format!("installing the signal handler: {error}"))
        })
        .transpose()?;

    // Map fixes in the published odometry frame: stamp and output-frame pose.
    let (fix_tx, fix_rx) = mpsc::channel::<(i64, visloc_core::geometry::SE3)>();
    if map_prior {
        let fix_sub =
            ros.subscription::<PoseWithCovarianceStamped>(&odom_fix_topic, sensor_qos(20))?;
        spawn_receiver(
            "odom-fix-rx",
            fix_sub,
            move |msg: PoseWithCovarianceStamped| {
                let _ = fix_tx.send((msg.header.stamp.to_nanos(), se3_from_ros(&msg.pose.pose)));
            },
        )?;
        eprintln!("visloc_vio_node: map fixes on {odom_fix_topic} go into the estimator");
    }
    let mut tf_static = (base_frame.is_some() || wheel_odom_topic.is_some())
        .then(|| StaticTfListener::new(&mut ros, &tf_static_topic))
        .transpose()?;
    let transforms = tf_static.as_ref().map_or_else(
        || Arc::new(Mutex::new(StaticTransforms::new())),
        |listener| Arc::clone(listener.transforms()),
    );

    // The newest wheel odometry: (sequence number, stamp, pose, body velocity, yaw rate).
    let wheel_latest: Arc<Mutex<Option<WheelSample>>> = Arc::new(Mutex::new(None));
    if let Some(topic) = &wheel_odom_topic {
        let wheel_sub = ros.subscription::<Odometry>(topic, sensor_qos(50))?;
        let latest = Arc::clone(&wheel_latest);
        let mut seq = 0u64;
        spawn_receiver("wheel-rx", wheel_sub, move |msg: Odometry| {
            seq += 1;
            *latest.lock().unwrap_or_else(|e| e.into_inner()) = Some(WheelSample {
                seq,
                child_frame: msg.child_frame_id.clone(),
                stamp_ns: msg.header.stamp.to_nanos(),
                pose: se3_from_ros(&msg.pose.pose),
                velocity_body: Vector3::new(
                    msg.twist.twist.linear.x,
                    msg.twist.twist.linear.y,
                    msg.twist.twist.linear.z,
                ),
                angular_body: Vector3::new(
                    msg.twist.twist.angular.x,
                    msg.twist.twist.angular.y,
                    msg.twist.twist.angular.z,
                ),
            });
        })?;
        eprintln!("visloc_vio_node: through lost tracking, the outputs follow {topic}");
    }

    let subscriptions = std::iter::once((0usize, left_sub))
        .chain(right_sub.map(|subscription| (1usize, subscription)));
    for (side, subscription) in subscriptions {
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
    if tracking.enabled() {
        eprintln!(
            "visloc_vio_node: lost-tracking monitor on: fewer than {} observations for {} s{}; \
             recovery after {} s; resets on {odom_reset_topic}",
            tracking.min_observations,
            tracking.lost_after_s,
            if tracking.max_speed_m_s > 0.0 {
                format!(" or faster than {} m/s", tracking.max_speed_m_s)
            } else {
                String::new()
            },
            tracking.recover_after_s,
        );
    }
    if let Some(params) = &mapper_params {
        eprintln!(
            "visloc_vio_node: online mapper on; out: {map_pose_topic}{} and {} -> {} on /tf",
            if publish_path {
                format!(" {map_path_topic}")
            } else {
                String::new()
            },
            params.map_frame,
            frames.odom_frame,
        );
    }

    // Frontend thread: KLT-track frames in order and hand them over. The
    // hand-over blocks rather than drops: every tracked frame carries the
    // IMU since the previous one, so the estimator must see all of them.
    // It is a rendezvous, so the frontend tracks at most one frame ahead;
    // when the estimator falls behind, frames wait (and drop-oldest) in the
    // inputs instead, before tracking, which keeps the newest frame next
    // and bounds the latency.
    let (tracked_tx, tracked_rx) = mpsc::sync_channel::<FrontendResult>(0);
    {
        let shared = Arc::clone(&shared);
        // Its own rayon pool keeps the frontend's parallel loops from
        // queueing behind the estimator's in the global pool.
        let pool = (frontend_threads > 0)
            .then(|| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(frontend_threads)
                    .thread_name(|index| format!("vio-klt-{index}"))
                    .build()
            })
            .transpose()
            .map_err(|error| format!("frontend thread pool: {error}"))?;
        std::thread::Builder::new()
            .name("vio-frontend".into())
            .spawn(move || match pool {
                Some(pool) => pool.install(|| run_frontend(frontend, &shared, &tracked_tx)),
                None => run_frontend(frontend, &shared, &tracked_tx),
            })
            .map_err(|error| format!("spawning the frontend thread: {error}"))?;
    }

    let mut path = PathAccumulator::new(path_max_length);
    let mut map_path = PathAccumulator::new(path_max_length);
    let mut processed = 0u64;
    let mut generation = 0u64;
    let mut error_log = RateLimitedLog::new(Duration::from_secs(2));
    let mut tf_log = RateLimitedLog::new(Duration::from_secs(5));
    let mut last_status = Instant::now();
    let mut estimator_time = StageTime::default();
    let mut monitor = TrackingMonitor::new(tracking);
    let mut resets = 0i32;
    let mut held = 0u64;
    let mut observations = ObservationStats::default();
    let mut bridge = OdomBridge::new();
    let mut wheel_published = 0u64;
    let mut outages = 0u32;
    let mut frame_stamps = FrameStamps::new(64);
    let (mut priors_used, mut priors_missed, mut velocities_used) = (0u64, 0u64, 0u64);
    // While tracking is lost, publishes the wheel-bridged pose of each new
    // wheel odometry message.
    let mut publish_bridged = |bridge: &mut OdomBridge, path: &mut PathAccumulator| {
        if bridge.state() != BridgeState::Wheels {
            return;
        }
        let Some(sample) = wheel_latest
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
        else {
            return;
        };
        if sample.seq == wheel_published {
            return;
        }
        let Some(pose) = wheel_in_output(&transforms, &sample, &output_frames.body_frame) else {
            return;
        };
        wheel_published = sample.seq;
        let Some(pose) = bridge.wheels(&pose) else {
            return;
        };
        let Some(wheel_from_out) = transforms
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .lookup(&sample.child_frame, &output_frames.body_frame)
        else {
            return;
        };
        let out_from_wheel = wheel_from_out.rotation.inverse();
        let estimate = VioEstimate {
            frame_id: 0,
            stamp_ns: sample.stamp_ns,
            header_stamp_ns: sample.stamp_ns,
            velocity_world: pose.rotation * (out_from_wheel * sample.velocity_body),
            imu_to_world: pose,
            gyro_bias: Vector3::zeros(),
            accel_bias: Vector3::zeros(),
            angular_velocity_body: Some(out_from_wheel * sample.angular_body),
            is_keyframe: false,
            observation_count: 0,
        };
        publish_logged(&odom_pub, odometry_msg(&estimate, &output_frames), "odom");
        let pose_msg = pose_stamped_msg(&estimate, &output_frames);
        if let Some(path_pub) = &path_pub {
            path.push(pose_msg.clone());
            publish_logged(path_pub, path.msg(&output_frames.odom_frame), "path");
        }
        publish_logged(&pose_pub, pose_msg, "pose");
        if let Some(tf_pub) = &tf_pub {
            publish_logged(tf_pub, tf_msg(&estimate, &output_frames), "tf");
        }
    };
    loop {
        publish_bridged(&mut bridge, &mut path);
        if shutdown
            .as_ref()
            .is_some_and(|signals| signals.load(Ordering::SeqCst) > 0)
        {
            break;
        }
        let received = tracked_rx.recv_timeout(Duration::from_secs(1));
        if last_status.elapsed() >= Duration::from_secs(10) {
            let stats = shared.with(|inputs| inputs.stats());
            let frontend_time = shared
                .frontend_time
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take();
            eprintln!(
                "visloc_vio_node: processed={processed} pairs={} dropped_behind={} \
                 sync_dropped={} imu_out_of_order={} imu_evicted={} \
                 frontend={frontend_time} estimator={} (avg/max) observations={} resets={resets} \
                 held={held} map_priors={priors_used}/{} wheel_velocities={velocities_used}{}{}",
                stats.pairs,
                stats.frames_dropped_behind,
                stats.sync_dropped,
                stats.imu_out_of_order,
                stats.imu_evicted,
                estimator_time.take(),
                observations.take(),
                priors_used + priors_missed,
                if monitor.is_tracking() {
                    ""
                } else {
                    " (tracking lost)"
                },
                mapper.as_ref().map_or_else(String::new, |session| format!(
                    " {}",
                    session.status().summary()
                )),
            );
            last_status = Instant::now();
        }
        let (frame_generation, tracked) = match received {
            Ok(received) => received,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err("the frontend thread stopped".into())
            }
        };
        if frame_generation != generation {
            // Tracked before the last reset.
            continue;
        }
        let started = Instant::now();
        let estimated = tracked.and_then(|frame| backend.estimate_with_marg_data(frame));
        estimator_time.add(started.elapsed());
        // Why the estimator must start over, if it must.
        let mut estimated = estimated;
        if let Ok((estimate, _)) = &estimated {
            *shared.gyro_bias.lock().unwrap_or_else(|e| e.into_inner()) = estimate.gyro_bias;
            observations.add(estimate.observation_count);
            match monitor.update(
                estimate.header_stamp_ns,
                estimate.observation_count,
                estimate.velocity_world.norm(),
            ) {
                TrackingVerdict::Publish => {}
                TrackingVerdict::Hold => {
                    held += 1;
                    publish_bridged(&mut bridge, &mut path);
                    continue;
                }
                TrackingVerdict::Lost(reason) => {
                    estimated = Err(VioError {
                        message: format!("tracking lost ({reason}); resetting the estimator"),
                        needs_reset: true,
                    });
                }
            }
        }
        match estimated {
            Ok((estimate, marg_data)) => {
                processed += 1;
                if processed == 1 {
                    eprintln!("visloc_vio_node: first pose published");
                }
                frame_stamps.record(estimate.header_stamp_ns, estimate.frame_id);
                if wheel_velocity {
                    let sample = wheel_latest
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .clone();
                    let imu_from_wheel = sample.as_ref().and_then(|sample| {
                        transforms
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .lookup(&frames.body_frame, &sample.child_frame)
                    });
                    if let (Some(sample), Some(imu_from_wheel), Some(angular)) =
                        (sample, imu_from_wheel, estimate.angular_velocity_body)
                    {
                        // Only a recent twist describes this frame.
                        if (sample.stamp_ns - estimate.header_stamp_ns).abs() <= 100_000_000 {
                            backend.add_external_factor(wheel_velocity_factor(
                                estimate.frame_id,
                                imu_from_wheel,
                                angular,
                                sample.velocity_body,
                                &external_noise,
                            ));
                            velocities_used += 1;
                        }
                    }
                }
                if map_prior {
                    let out_from_imu = match &base_frame {
                        None => Some(visloc_core::geometry::SE3::identity()),
                        Some(base) => transforms
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .lookup(&frames.body_frame, base)
                            .map(|imu_from_base| imu_from_base.inverse()),
                    };
                    let odom_from_vio = bridge.offset().clone();
                    for (stamp_ns, odom_from_out) in fix_rx.try_iter() {
                        match (frame_stamps.frame_at(stamp_ns, 1_000_000), &out_from_imu) {
                            (Some(frame_id), Some(out_from_imu)) => {
                                backend.add_external_factor(map_prior_factor(
                                    frame_id,
                                    &odom_from_out,
                                    out_from_imu,
                                    &odom_from_vio,
                                    &external_noise,
                                ));
                                priors_used += 1;
                            }
                            _ => priors_missed += 1,
                        }
                    }
                }
                if let Some(session) = mapper.as_mut() {
                    // The IMU pose, as the mapper's keyframe poses are.
                    session.record_vio_pose(
                        estimate.frame_id,
                        estimate.header_stamp_ns,
                        &estimate.imu_to_world,
                    );
                    if let Some(packet) = marg_data {
                        session.offer(packet);
                    }
                }
                let estimate = match &base_frame {
                    None => estimate,
                    Some(base) => {
                        let imu_from_base = transforms
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .lookup(&frames.body_frame, base);
                        let Some(imu_from_base) = imu_from_base else {
                            tf_log.log(|| {
                                format!(
                                    "no /tf_static transform {} <- {base} yet; not publishing",
                                    frames.body_frame
                                )
                            });
                            if let Some(listener) = tf_static.as_mut() {
                                match listener.missing(&mut ros) {
                                    Ok(true) => {
                                        eprintln!("visloc_vio_node: subscribed to /tf_static again")
                                    }
                                    Ok(false) => {}
                                    Err(error) => eprintln!("visloc_vio_node: {error}"),
                                }
                            }
                            continue;
                        };
                        rebase(&estimate, &imu_from_base)
                    }
                };
                let estimate = if wheel_odom_topic.is_some() {
                    let resumed = bridge.state() == BridgeState::Wheels;
                    let pose = bridge.vio(&estimate.imu_to_world);
                    if resumed {
                        eprintln!(
                            "visloc_vio_node: tracking holds again after {:.2} m on wheel odometry; \
                             VIO continues from there (outage {outages})",
                            bridge.bridged_m()
                        );
                    }
                    VioEstimate {
                        velocity_world: bridge.offset().rotation * estimate.velocity_world,
                        imu_to_world: pose,
                        ..estimate
                    }
                } else {
                    estimate
                };
                publish_logged(&odom_pub, odometry_msg(&estimate, &output_frames), "odom");
                let pose = pose_stamped_msg(&estimate, &output_frames);
                if let Some(path_pub) = &path_pub {
                    path.push(pose.clone());
                    publish_logged(path_pub, path.msg(&output_frames.odom_frame), "path");
                }
                publish_logged(&pose_pub, pose, "pose");
                let map_from_odom = mapper.as_ref().map(MapperSession::map_from_odom);
                if let (Some(map_from_odom), Some(params)) = (&map_from_odom, &mapper_params) {
                    let map_pose = map_pose_msg(
                        map_from_odom,
                        &estimate.imu_to_world,
                        estimate.header_stamp_ns,
                        &params.map_frame,
                    );
                    if let Some(map_path_pub) = &map_path_pub {
                        map_path.push(map_pose.clone());
                        publish_logged(map_path_pub, map_path.msg(&params.map_frame), "map_path");
                    }
                    if let Some(map_pose_pub) = &map_pose_pub {
                        publish_logged(map_pose_pub, map_pose, "map_pose");
                    }
                }
                if let Some(tf_pub) = &tf_pub {
                    let mut tf = tf_msg(&estimate, &output_frames);
                    if let (Some(map_from_odom), Some(params)) = (&map_from_odom, &mapper_params) {
                        tf.transforms.push(map_to_odom_transform(
                            map_from_odom,
                            estimate.header_stamp_ns,
                            &params.map_frame,
                            &output_frames.odom_frame,
                        ));
                    }
                    publish_logged(tf_pub, tf, "tf");
                }
            }
            Err(error) => {
                if error.needs_reset && !monitor.is_tracking() && monitor.config().enabled() {
                    // Lost tracking: always logged, it stops the robot.
                    eprintln!("visloc_vio_node: {error}");
                } else {
                    error_log.log(|| format!("{error}"));
                }
                if error.needs_reset {
                    // The frontend resets itself when it pops the first
                    // frame of the new generation.
                    generation = shared.restart_stream();
                    backend.reset()?;
                    frame_stamps.clear();
                    // Fixes of the old VIO frame no longer apply.
                    priors_missed += fix_rx.try_iter().count() as u64;
                    monitor.restart();
                    resets += 1;
                    let wheel_pose = wheel_odom_topic.as_ref().and_then(|_| {
                        wheel_latest
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .clone()
                    });
                    // Lost again before tracking held: still bridging from the first loss.
                    let new_outage = bridge.state() == BridgeState::Vio;
                    let bridged = wheel_pose.is_some_and(|sample| {
                        match wheel_in_output(&transforms, &sample, &output_frames.body_frame) {
                            Some(pose) => bridge.lost(&pose),
                            None => {
                                eprintln!(
                                    "visloc_vio_node: no /tf_static transform {} <- {}; \
                                     cannot drive on wheel odometry",
                                    sample.child_frame, output_frames.body_frame
                                );
                                false
                            }
                        }
                    });
                    if bridged {
                        if new_outage {
                            outages += 1;
                        }
                        eprintln!(
                            "visloc_vio_node: driving on wheel odometry until tracking holds \
                             again (outage {outages}); the odometry frame stays"
                        );
                    } else {
                        path.clear();
                        publish_logged(&odom_reset_pub, Int32 { data: resets }, "odom_reset");
                    }
                    eprintln!(
                        "visloc_vio_node: estimator reset {resets}; re-initializing from IMU{}",
                        if bridge.state() == BridgeState::Wheels {
                            "; outputs on wheel odometry until tracking holds again"
                        } else if monitor.config().enabled() {
                            "; outputs held until tracking holds again"
                        } else {
                            ""
                        }
                    );
                    if mapper.is_some() {
                        // The new VIO world frame is unrelated to the old
                        // one, so the map starts over too.
                        drop(mapper.take());
                        mapper = start_mapper()?;
                        map_path.clear();
                        eprintln!("visloc_vio_node: online mapper restarted with the estimator");
                    }
                }
            }
        }
    }
    // Only a shutdown signal leaves the loop, and only with the mapper.
    match (mapper, &mapper_params) {
        (Some(session), Some(params)) => finish_mapper(session, params),
        _ => Ok(()),
    }
}

/// Runs the mapper's final global optimization and writes the corrected
/// trajectory if asked to.
fn finish_mapper(session: MapperSession, params: &MapperParams) -> Result<(), String> {
    eprintln!(
        "visloc_vio_node: final mapper optimization ({})",
        session.status().summary()
    );
    let started = Instant::now();
    let finished = session.finish()?;
    eprintln!(
        "visloc_vio_node: mapper finished in {:.1} s: {} keyframes, {} frames corrected",
        started.elapsed().as_secs_f64(),
        finished.keyframe_count,
        finished.trajectory.len()
    );
    if let Some(path) = &params.trajectory_out {
        std::fs::write(path, tum_string(&finished.trajectory))
            .map_err(|error| format!("writing {}: {error}", path.display()))?;
        eprintln!(
            "visloc_vio_node: wrote the loop-closed trajectory to {}",
            path.display()
        );
    }
    Ok(())
}

/// Frontend-thread body: pops synchronized frames, tracks them and sends
/// them to the estimator thread in order. Returns when that thread hangs up.
fn run_frontend(
    mut frontend: VioFrontend,
    shared: &Shared,
    tracked_tx: &mpsc::SyncSender<FrontendResult>,
) {
    let mut generation = 0u64;
    let mut waiting_log = RateLimitedLog::new(Duration::from_secs(5));
    loop {
        let (packet, packet_generation): (Option<SensorPacket<RawU16Image>>, u64) = {
            let guard = shared.inputs.lock().unwrap_or_else(|e| e.into_inner());
            let (mut guard, timeout) = shared
                .ready
                .wait_timeout_while(guard, Duration::from_secs(1), |inputs| {
                    !inputs.vio.front_ready()
                })
                .unwrap_or_else(|e| e.into_inner());
            if timeout.timed_out() && guard.vio.queued_frames() > 0 {
                waiting_log
                    .log(|| "stereo frames queued but IMU has not reached their stamp yet".into());
            }
            (guard.vio.pop_ready(), guard.generation)
        };
        let Some(packet) = packet else {
            continue;
        };
        if packet_generation != generation {
            if let Err(error) = frontend.reset() {
                eprintln!("visloc_vio_node: {error}");
                return;
            }
            generation = packet_generation;
        }
        let gyro_bias = *shared.gyro_bias.lock().unwrap_or_else(|e| e.into_inner());
        // Errors go to the estimator thread too, which owns resets.
        let started = Instant::now();
        let tracked = frontend.track(packet, gyro_bias);
        shared
            .frontend_time
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .add(started.elapsed());
        if tracked_tx.send((generation, tracked)).is_err() {
            return;
        }
    }
}
