//! `visloc_nav_sim_node`: a simulated camera robot localized by visloc-rs,
//! for driving with Nav2 without a robot.
//!
//! The robot is `visloc_navigation::sim`'s differential-drive robot in the
//! simulated apartment; its map is the one `nav_sim_build_map` saved (the
//! COLMAP model and landmark descriptors). The node takes Nav2's `cmd_vel`,
//! and publishes the wheel odometry (`odom`, `/tf` `odom -> base_link`), a
//! depth-camera scan (`scan`) and, from map-based localization of the
//! camera's detections, `/tf` `map -> odom`: what a robot driver plus a
//! localization node give Nav2 in place of AMCL. See `docs/navigation.md`.

use std::fs::File;
use std::io::{BufWriter, Write as _};
use std::path::PathBuf;
use std::process;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use visloc_io::colmap::read_colmap_text_model;
use visloc_io::descriptors::read_landmark_descriptors_txt;
use visloc_navigation::sim::{CameraRig, Pose2, Texture, World};
use visloc_navigation::sim_robot::{LocalizationMode, SimRobot, SimRobotConfig};
use visloc_ros2::dds::{output_qos, publish_logged, sensor_qos, spawn_receiver, RosNode};
use visloc_ros2::msgs::{Int32, LaserScan, Odometry, PoseStamped, TfMessage, Time, Twist};
use visloc_ros2::nav_sim::{
    ground_truth_msg, map_pose_msg, odom_msg, parse_box, record_row, scan_msg, tf_msg,
    NavSimFrames, RECORD_HEADER,
};
use visloc_ros2::params::NodeArgs;

const USAGE: &str = "\
visloc_nav_sim_node -- a simulated camera robot localized by visloc-rs, for Nav2

USAGE:
  visloc_nav_sim_node --map_dir DIR [options]

PARAMETERS (--name value  or  -p name:=value):
  map_dir        map saved by `nav_sim_build_map` (sparse/ + landmark_descriptors.txt) [required]
  mode           where map -> odom comes from: visloc (map-based localization of the
                 camera), odometry (the start pose, then wheel odometry only) or
                 ground_truth                                                    [visloc]
  world_seed     seed of the simulated apartment; must be the map's               [1]
  start_x, start_y, start_yaw   true start pose in the map                [1.5, 2.5, 0]
  extra_box      x0,y0,x1,y1,height: an obstacle that is not in the map           [none]
  rate_hz        simulation and publishing rate                                   [20]
  camera_rate_hz localization attempts per second                                 [5]
  cmd_timeout_s  stop when no cmd_vel came for this long                          [0.5]
  record         CSV file: time, true pose, believed pose, collisions per step    [none]
  map_frame, odom_frame, base_frame                                   [map, odom, base_link]
  domain_id      DDS domain (default: $ROS_DOMAIN_ID or 0)
  node_name      node name                                                [visloc_nav_sim]

TOPICS (remap with --remap from=to or -r from:=to):
  in : cmd_vel (geometry_msgs/Twist)
  out: odom (nav_msgs/Odometry), scan (sensor_msgs/LaserScan, frame base_link),
       /tf (odom -> base_link; map -> odom once localized), ground_truth
       (nav_msgs/Odometry, true pose in map), map_pose (geometry_msgs/PoseStamped,
       the believed pose), collisions (std_msgs/Int32, steps a wall stopped the robot)
";

fn main() {
    if let Err(error) = run() {
        eprintln!("visloc_nav_sim_node: {error}");
        process::exit(1);
    }
}

fn now_time() -> Time {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as i64);
    Time::from_nanos(nanos)
}

fn run() -> Result<(), String> {
    let args = NodeArgs::parse(std::env::args().skip(1))?;
    if args.help {
        print!("{USAGE}");
        return Ok(());
    }
    let map_dir = PathBuf::from(args.required("map_dir")?);
    let mode: LocalizationMode = args.string_or("mode", "visloc").parse()?;
    let world_seed: u64 = args.parsed_or("world_seed", 1)?;
    let start = Pose2::new(
        args.parsed_or("start_x", 1.5)?,
        args.parsed_or("start_y", 2.5)?,
        args.parsed_or("start_yaw", 0.0)?,
    );
    let extra_box = args
        .string("extra_box")
        .map(|text| parse_box(&text))
        .transpose()?;
    let rate_hz: f64 = args.parsed_or("rate_hz", 20.0)?;
    let camera_rate_hz: f64 = args.parsed_or("camera_rate_hz", 5.0)?;
    let cmd_timeout = Duration::from_secs_f64(args.parsed_or("cmd_timeout_s", 0.5)?);
    let record = args.string("record");
    let frames = NavSimFrames {
        map: args.string_or("map_frame", "map"),
        odom: args.string_or("odom_frame", "odom"),
        base: args.string_or("base_frame", "base_link"),
    };
    if !(rate_hz > 0.0 && camera_rate_hz > 0.0) {
        return Err("rate_hz and camera_rate_hz must be positive".into());
    }
    let domain_id = args.domain_id()?;
    let node_name = args
        .node_name
        .clone()
        .unwrap_or_else(|| args.string_or("node_name", "visloc_nav_sim"));
    let namespace = args.namespace.clone().unwrap_or_else(|| "/".into());
    let cmd_topic = args.topic("cmd_vel");
    let odom_topic = args.topic("odom");
    let scan_topic = args.topic("scan");
    let tf_topic = args.topic("/tf");
    let truth_topic = args.topic("ground_truth");
    let map_pose_topic = args.topic("map_pose");
    let collisions_topic = args.topic("collisions");
    args.finish()?;

    let map = read_colmap_text_model(map_dir.join("sparse"))
        .map_err(|error| format!("reading {}/sparse: {error}", map_dir.display()))?;
    let descriptors = read_landmark_descriptors_txt(map_dir.join("landmark_descriptors.txt"))
        .map_err(|error| format!("reading the descriptors in {}: {error}", map_dir.display()))?;
    let mut world = World::apartment(world_seed);
    if let Some((min, max, height)) = extra_box {
        world.add_box(min, max, height, Some(Texture::default()));
    }
    let config = SimRobotConfig {
        mode,
        camera_rate_hz,
        ..SimRobotConfig::default()
    };
    eprintln!(
        "visloc_nav_sim_node: map {} ({} landmarks, {} keyframes), mode {mode:?}, start ({:.2}, {:.2}, {:.2}){}",
        map_dir.display(),
        map.landmarks.len(),
        map.keyframes.len(),
        start.x,
        start.y,
        start.yaw,
        extra_box.map_or(String::new(), |b| format!(", extra box {b:?}"))
    );
    let mut robot = SimRobot::new(
        world,
        CameraRig::forward(0.5),
        map,
        descriptors,
        start,
        config,
    );
    let scan_config = robot.config.scan;

    let mut ros = RosNode::new(domain_id, &namespace, &node_name)?;
    let cmd_sub = ros.subscription::<Twist>(&cmd_topic, sensor_qos(10))?;
    let odom_pub = ros.publisher::<Odometry>(&odom_topic, output_qos(20))?;
    let scan_pub = ros.publisher::<LaserScan>(&scan_topic, output_qos(10))?;
    let tf_pub = ros.publisher::<TfMessage>(&tf_topic, output_qos(100))?;
    let truth_pub = ros.publisher::<Odometry>(&truth_topic, output_qos(20))?;
    let map_pose_pub = ros.publisher::<PoseStamped>(&map_pose_topic, output_qos(20))?;
    let collisions_pub = ros.publisher::<Int32>(&collisions_topic, output_qos(10))?;

    let command: Arc<Mutex<(Twist, Instant)>> =
        Arc::new(Mutex::new((Twist::default(), Instant::now() - cmd_timeout)));
    {
        let command = Arc::clone(&command);
        spawn_receiver("cmd-rx", cmd_sub, move |msg: Twist| {
            *command.lock().unwrap_or_else(|e| e.into_inner()) = (msg, Instant::now());
        })?;
    }
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = Arc::clone(&stop);
        ctrlc::set_handler(move || stop.store(true, Ordering::SeqCst))
            .map_err(|error| format!("signal handler: {error}"))?;
    }
    let mut recorder = record
        .as_ref()
        .map(|path| -> Result<BufWriter<File>, String> {
            let mut file = BufWriter::new(
                File::create(path).map_err(|error| format!("creating {path}: {error}"))?,
            );
            writeln!(file, "{RECORD_HEADER}").map_err(|error| error.to_string())?;
            Ok(file)
        })
        .transpose()?;

    let dt = 1.0 / rate_hz;
    let period = Duration::from_secs_f64(dt);
    let mut next_tick = Instant::now();
    let mut localized_logged = false;
    let mut last_collisions = usize::MAX;
    let mut last_status = Instant::now();
    let mut step_index: u64 = 0;
    while !stop.load(Ordering::SeqCst) {
        let (cmd, received) = *command.lock().unwrap_or_else(|e| e.into_inner());
        let (v, w) = if received.elapsed() <= cmd_timeout {
            (cmd.linear.x, cmd.angular.z)
        } else {
            (0.0, 0.0)
        };
        let step = robot.step(v, w, dt);
        let stamp = now_time();
        publish_logged(&tf_pub, tf_msg(stamp, &frames, &step), "tf");
        publish_logged(&odom_pub, odom_msg(stamp, &frames, &step), "odom");
        publish_logged(
            &truth_pub,
            ground_truth_msg(stamp, &frames, &step),
            "ground_truth",
        );
        if let Some(pose) = map_pose_msg(stamp, &frames, &step) {
            publish_logged(&map_pose_pub, pose, "map_pose");
        }
        if step_index.is_multiple_of(2) {
            let ranges = robot.scan();
            publish_logged(
                &scan_pub,
                scan_msg(stamp, &frames.base, &scan_config, &ranges),
                "scan",
            );
        }
        let collisions = robot.robot.collisions;
        if collisions != last_collisions || step_index.is_multiple_of((rate_hz as u64).max(1)) {
            publish_logged(
                &collisions_pub,
                Int32 {
                    data: collisions as i32,
                },
                "collisions",
            );
            last_collisions = collisions;
        }
        if let Some(file) = recorder.as_mut() {
            writeln!(file, "{}", record_row(&step, collisions)).map_err(|e| e.to_string())?;
        }
        if !localized_logged && step.map_from_odom.is_some() {
            localized_logged = true;
            eprintln!(
                "visloc_nav_sim_node: map -> odom set after {:.1} s",
                step.time_s
            );
        }
        if last_status.elapsed() >= Duration::from_secs(10) {
            last_status = Instant::now();
            let c = robot.counts;
            let error = step.estimate().map_or(f64::NAN, |e| {
                (e.x - step.ground_truth.x).hypot(e.y - step.ground_truth.y)
            });
            eprintln!(
                "visloc_nav_sim_node: t {:.0} s, pose error {:.3} m, collisions {collisions}, \
                 localizations {} (fixes {}, accepted {}, rejected {}), mean {:.1} ms",
                step.time_s,
                error,
                c.attempts,
                c.fixes,
                c.accepted,
                c.rejected,
                1e3 * c.latency_sum_s / c.attempts.max(1) as f64
            );
        }
        step_index += 1;
        next_tick += period;
        let now = Instant::now();
        if next_tick > now {
            std::thread::sleep(next_tick - now);
        } else {
            next_tick = now;
        }
    }
    if let Some(mut file) = recorder {
        file.flush().map_err(|error| error.to_string())?;
    }
    Ok(())
}
