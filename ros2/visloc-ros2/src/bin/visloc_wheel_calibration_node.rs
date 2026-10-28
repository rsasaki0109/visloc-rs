//! `visloc_wheel_calibration_node`: a robot's wheel odometry, calibrated
//! against `visloc_vio_node` while it drives.
//!
//! Wheel odometry reports the point the robot turns about, in a level frame,
//! at the wheels' scale. The robot's description may put its body
//! (`base_link`) centimetres from that point and a degree or two off level,
//! and the VIO the maps are built from may be a few percent off the wheels'
//! scale. The node pairs the two odometries, fits all three once the robot
//! has turned and travelled enough (`visloc_localization::WheelCalibration`),
//! and from then on publishes the wheel odometry as the body's pose at the
//! VIO's scale, continuous where the calibration takes over. Before that it
//! passes the wheel odometry through. With `calibration_file`, a saved
//! calibration applies from the start, and a new one is saved there.

use std::process;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use visloc_core::geometry::SE3;
use visloc_localization::WheelCalibrationConfig;
use visloc_ros2::dds::{
    output_qos, publish_logged, sensor_qos, spawn_receiver, RosNode, StaticTfListener,
};
use visloc_ros2::map_localize::se3_from_ros;
use visloc_ros2::msgs::Odometry;
use visloc_ros2::params::NodeArgs;
use visloc_ros2::vio::ros_pose;
use visloc_ros2::wheel_odom::{SavedCalibration, WheelOdomCalibrator, WheelOutput};

const USAGE: &str = "\
visloc_wheel_calibration_node -- wheel odometry calibrated against the VIO while driving

USAGE:
  visloc_wheel_calibration_node [options]

PARAMETERS (--name value  or  -p name:=value):
  calibration_file     JSON calibration: applied from the start if it exists, written
                       when a new one is fitted                                     [none]
  min_turn_deg         fit once the wheel odometry has turned this much in total ...  [180]
  min_travel_m         ... and travelled this far ...                                [1.0]
  min_pairs            ... over this many wheel/VIO pose pairs                       [200]
  level                level the body frame against the VIO's gravity (level floor) [true]
  apply_scale          scale the output to the VIO's scale (the maps'); the VIO's scale
                       varies by a few percent between sessions                    [true]
  publish_uncalibrated pass the wheel odometry through until calibrated             [true]
  domain_id            DDS domain (default: $ROS_DOMAIN_ID or 0)
  node_name            node name                                       [visloc_wheel_calibration]

TOPICS (remap with --remap from=to or -r from:=to):
  in : wheel_odom (nav_msgs/Odometry, the wheel odometry: planar, child_frame_id = the
       robot's body, e.g. base_link), vio_odom (nav_msgs/Odometry, visloc_vio_node's
       odom), /tf_static (tf2_msgs/TFMessage: the VIO's child frame <- the body)
  out: odom (nav_msgs/Odometry, the body's pose at the VIO's scale, in the wheel
       odometry's frame; e.g. visloc_map_localize_node's odom with extrinsics:=tf)
";

fn main() {
    if let Err(error) = run() {
        eprintln!("visloc_wheel_calibration_node: {error}");
        process::exit(1);
    }
}

/// What the receive threads share.
struct State {
    calibrator: Option<WheelOdomCalibrator>,
    /// Child frames of the wheel and VIO odometries, once seen.
    wheel_frame: Option<String>,
    vio_frame: Option<String>,
}

fn run() -> Result<(), String> {
    let args = NodeArgs::parse(std::env::args().skip(1))?;
    if args.help {
        print!("{USAGE}");
        return Ok(());
    }
    let calibration_file = args.string("calibration_file");
    let defaults = WheelCalibrationConfig::default();
    let config = WheelCalibrationConfig {
        min_turn_deg: args.parsed_or("min_turn_deg", defaults.min_turn_deg)?,
        min_travel_m: args.parsed_or("min_travel_m", defaults.min_travel_m)?,
        min_pairs: args.parsed_or("min_pairs", defaults.min_pairs)?,
        level: args.flag("level", defaults.level)?,
        ..defaults
    };
    let apply_scale = args.flag("apply_scale", true)?;
    let publish_uncalibrated = args.flag("publish_uncalibrated", true)?;
    let domain_id = args.domain_id()?;
    let node_name = args
        .node_name
        .clone()
        .unwrap_or_else(|| args.string_or("node_name", "visloc_wheel_calibration"));
    let namespace = args.namespace.clone().unwrap_or_else(|| "/".into());
    let wheel_topic = args.topic("wheel_odom");
    let vio_topic = args.topic("vio_odom");
    let out_topic = args.topic("odom");
    let tf_static_topic = args.topic("/tf_static");
    args.finish()?;

    let saved = match &calibration_file {
        Some(path) if std::path::Path::new(path).exists() => {
            let text = std::fs::read_to_string(path)
                .map_err(|error| format!("reading {path}: {error}"))?;
            let saved = SavedCalibration::from_json(&text)?;
            eprintln!(
                "visloc_wheel_calibration_node: calibration from {path}: offset ({:.1}, {:.1}) cm, \
                 scale {:.4}, tilt {:.2} deg",
                saved.offset_m[0] * 100.0,
                saved.offset_m[1] * 100.0,
                saved.scale,
                saved.tilt_deg
            );
            Some(saved)
        }
        _ => None,
    };
    let state = Arc::new(Mutex::new(State {
        calibrator: saved.map(|saved| {
            WheelOdomCalibrator::from_estimate(saved.estimate()).with_scale(apply_scale)
        }),
        wheel_frame: None,
        vio_frame: None,
    }));

    let mut ros = RosNode::new(domain_id, &namespace, &node_name)?;
    let wheel_sub = ros.subscription::<Odometry>(&wheel_topic, sensor_qos(50))?;
    let vio_sub = ros.subscription::<Odometry>(&vio_topic, sensor_qos(50))?;
    let odom_pub = ros.publisher::<Odometry>(&out_topic, output_qos(50))?;
    let mut tf_static = StaticTfListener::new(&mut ros, &tf_static_topic)?;
    let transforms = Arc::clone(tf_static.transforms());

    {
        let state = Arc::clone(&state);
        spawn_receiver("vio-rx", vio_sub, move |msg: Odometry| {
            let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
            if state.vio_frame.is_none() {
                state.vio_frame = Some(msg.child_frame_id.clone());
            }
            if let Some(calibrator) = state.calibrator.as_mut() {
                calibrator.add_vio(msg.header.stamp.to_nanos(), &se3_from_ros(&msg.pose.pose));
            }
        })?;
    }
    {
        let state = Arc::clone(&state);
        spawn_receiver("wheel-rx", wheel_sub, move |msg: Odometry| {
            let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
            if state.wheel_frame.is_none() {
                state.wheel_frame = Some(msg.child_frame_id.clone());
            }
            let wheel = se3_from_ros(&msg.pose.pose);
            let output = match state.calibrator.as_mut() {
                Some(calibrator) => calibrator.wheel(msg.header.stamp.to_nanos(), &wheel),
                None => WheelOutput::PassThrough(wheel),
            };
            if let WheelOutput::Fixed(_, estimate) = &output {
                eprintln!(
                    "visloc_wheel_calibration_node: calibrated over {} pairs, {:.0} deg turned: \
                     the body sits ({:.1}, {:.1}) cm from the wheel odometry's point, leans \
                     {:.2} deg against the VIO's gravity; VIO scale {:.4} of the wheels; \
                     residual {:.2} cm (without the offset {:.2} cm)",
                    estimate.pairs,
                    estimate.turned_deg,
                    estimate.offset.x * 100.0,
                    estimate.offset.y * 100.0,
                    estimate.tilt_deg,
                    estimate.scale,
                    estimate.rms_m * 100.0,
                    estimate.rms_without_offset_m * 100.0,
                );
                if let Some(path) = &calibration_file {
                    match std::fs::write(path, SavedCalibration::from(estimate).to_json()) {
                        Ok(()) => eprintln!("visloc_wheel_calibration_node: saved {path}"),
                        Err(error) => {
                            eprintln!("visloc_wheel_calibration_node: writing {path}: {error}")
                        }
                    }
                }
            }
            if matches!(output, WheelOutput::PassThrough(_)) && !publish_uncalibrated {
                return;
            }
            let mut out = msg;
            out.pose.pose = ros_pose(output.pose());
            publish_logged(&odom_pub, out, "odom");
        })?;
    }

    eprintln!(
        "visloc_wheel_calibration_node: domain {domain_id}, node {}; in: {wheel_topic} {vio_topic}; \
         out: {out_topic}",
        visloc_ros2::dds::qualified_name(&namespace, &node_name)
    );
    let mut last_status = Instant::now();
    loop {
        std::thread::sleep(Duration::from_millis(500));
        let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
        if state.calibrator.is_none() {
            // Collecting needs where the VIO's frame sits on the body.
            if let (Some(wheel), Some(vio)) = (state.wheel_frame.clone(), state.vio_frame.clone()) {
                let vio_from_body = if wheel == vio {
                    Some(SE3::identity())
                } else {
                    transforms
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .lookup(&vio, &wheel)
                };
                match vio_from_body {
                    Some(vio_from_body) => {
                        state.calibrator = Some(
                            WheelOdomCalibrator::new(config.clone(), vio_from_body)
                                .with_scale(apply_scale),
                        );
                        eprintln!("visloc_wheel_calibration_node: collecting ({vio} <- {wheel} from /tf_static)");
                    }
                    None => match tf_static.missing(&mut ros) {
                        Ok(true) => eprintln!(
                            "visloc_wheel_calibration_node: no /tf_static {vio} <- {wheel} yet; \
                             subscribed again"
                        ),
                        Ok(false) => {}
                        Err(error) => eprintln!("visloc_wheel_calibration_node: {error}"),
                    },
                }
            }
        }
        if last_status.elapsed() >= Duration::from_secs(10) {
            last_status = Instant::now();
            match state.calibrator.as_ref().and_then(WheelOdomCalibrator::progress) {
                Some((pairs, turned)) => eprintln!(
                    "visloc_wheel_calibration_node: collecting: {pairs} pairs, {turned:.0} deg turned"
                ),
                None if state.calibrator.is_none() => eprintln!(
                    "visloc_wheel_calibration_node: waiting for wheel_odom, vio_odom and /tf_static"
                ),
                None => {}
            }
        }
    }
}
