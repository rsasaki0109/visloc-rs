//! `visloc_vio_node`: stereo-inertial VIO (Basalt port) as a ROS 2 node.
//!
//! Subscribes `left/image_raw`, `right/image_raw` (`sensor_msgs/Image`) and
//! `imu` (`sensor_msgs/Imu`); publishes `odom` (`nav_msgs/Odometry`), `pose`
//! (`geometry_msgs/PoseStamped`), `path` (`nav_msgs/Path`) and optionally
//! `/tf`. Run with `--help` for parameters.
//!
//! KLT tracking runs on its own thread, one frame ahead of the estimator
//! (the main thread), as upstream Basalt's optical-flow and VIO threads do.

use std::process;
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use nalgebra::Vector3;

use visloc_basalt::RawU16Image;
use visloc_ros2::dds::{
    output_qos, publish_logged, sensor_qos, spawn_receiver, RosNode, StaticTfListener,
};
use visloc_ros2::image::decode_luma;
use visloc_ros2::msgs::{Image, Imu, Odometry, Path, PoseStamped, TfMessage};
use visloc_ros2::params::NodeArgs;
use visloc_ros2::sync::{SensorPacket, VioInputs};
use visloc_ros2::tf_tree::StaticTransforms;
use visloc_ros2::vio::{
    imu_sample_from_msg, odometry_msg, pose_stamped_msg, rebase, tf_msg, PathAccumulator,
    TrackedFrame, VioCore, VioError, VioFrames, VioFrontend, VioOptions,
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
  body_frame             the IMU's frame (Basalt's body); child_frame_id of the
                         outputs unless base_frame is set                        [imu_link]
  base_frame             publish this frame's pose instead (e.g. base_link, REP-105),
                         placed relative to body_frame by /tf_static             [none]
  publish_tf             publish odom_frame -> child frame on /tf                [true]
  publish_path           publish nav_msgs/Path                                   [true]
  path_max_length        poses kept in the published path (drop-oldest)          [2000]
  stereo_sync_tolerance_ms  max |t_left - t_right| for a stereo pair             [2.0]
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
  domain_id              DDS domain (default: $ROS_DOMAIN_ID or 0)
  node_name              node name                                              [visloc_vio]

TOPICS (remap with --remap from=to or -r from:=to):
  in : left/image_raw, right/image_raw (sensor_msgs/Image), imu (sensor_msgs/Imu),
       /tf_static (tf2_msgs/TFMessage, with base_frame)
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
    let options = VioOptions {
        urgent_keyframes: args.flag("urgent_keyframes", true)?,
        imu_seed_klt: args.flag("imu_seed_klt", false)?,
        max_iterations: Some(args.parsed_or("max_iterations", 3u32)?).filter(|&n| n > 0),
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
    args.finish()?;

    let read = |path: &str| {
        std::fs::read_to_string(path).map_err(|error| format!("reading {path}: {error}"))
    };
    let core = VioCore::from_json(&read(&calibration_path)?, &read(&config_path)?, &options)?;
    let expected = [core.expected_resolution(0), core.expected_resolution(1)];
    let shared = Arc::new(Shared {
        inputs: Mutex::new(Inputs {
            vio: VioInputs::new(
                (tolerance_ms * 1e6).round() as i64,
                8,
                frame_queue,
                imu_buffer,
                core.cam_time_offset_ns(),
            ),
            generation: 0,
        }),
        ready: Condvar::new(),
        gyro_bias: Mutex::new(Vector3::zeros()),
        frontend_time: Mutex::new(StageTime::default()),
    });
    let (frontend, mut backend) = core.into_stages();

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

    let mut tf_static = base_frame
        .is_some()
        .then(|| StaticTfListener::new(&mut ros, &tf_static_topic))
        .transpose()?;
    let transforms = tf_static.as_ref().map_or_else(
        || Arc::new(Mutex::new(StaticTransforms::new())),
        |listener| Arc::clone(listener.transforms()),
    );

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
    let mut processed = 0u64;
    let mut generation = 0u64;
    let mut error_log = RateLimitedLog::new(Duration::from_secs(2));
    let mut tf_log = RateLimitedLog::new(Duration::from_secs(5));
    let mut last_status = Instant::now();
    let mut estimator_time = StageTime::default();
    loop {
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
                 frontend={frontend_time} estimator={} (avg/max)",
                stats.pairs,
                stats.frames_dropped_behind,
                stats.sync_dropped,
                stats.imu_out_of_order,
                stats.imu_evicted,
                estimator_time.take(),
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
        let estimated = tracked.and_then(|frame| backend.estimate(frame));
        estimator_time.add(started.elapsed());
        match estimated {
            Ok(estimate) => {
                processed += 1;
                if processed == 1 {
                    eprintln!("visloc_vio_node: first pose published");
                }
                *shared.gyro_bias.lock().unwrap_or_else(|e| e.into_inner()) = estimate.gyro_bias;
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
                publish_logged(&odom_pub, odometry_msg(&estimate, &output_frames), "odom");
                let pose = pose_stamped_msg(&estimate, &output_frames);
                if let Some(path_pub) = &path_pub {
                    path.push(pose.clone());
                    publish_logged(path_pub, path.msg(&output_frames.odom_frame), "path");
                }
                publish_logged(&pose_pub, pose, "pose");
                if let Some(tf_pub) = &tf_pub {
                    publish_logged(tf_pub, tf_msg(&estimate, &output_frames), "tf");
                }
            }
            Err(error) => {
                error_log.log(|| format!("{error}"));
                if error.needs_reset {
                    // The frontend resets itself when it pops the first
                    // frame of the new generation.
                    generation = shared.restart_stream();
                    backend.reset()?;
                    path.clear();
                    eprintln!("visloc_vio_node: estimator reset; re-initializing from IMU");
                }
            }
        }
    }
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
