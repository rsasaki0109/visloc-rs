//! `visloc_wheel_calibration_node`: a robot's wheel odometry, calibrated
//! while it drives, against `visloc_vio_node` and the map.
//!
//! Wheel odometry reports the point the robot turns about, in a level frame,
//! at the wheels' scale. The robot's description may put its body
//! (`base_link`) centimetres from that point and a degree or two off level,
//! and the map, built by a VIO, may be a few percent off the wheels' scale.
//! The node pairs the wheel odometry with the VIO, which shows the offset
//! once the robot has turned and travelled enough, and with
//! `visloc_map_localize_node`'s localizations, which carry the map's scale
//! and gravity (`visloc_localization::WheelCalibration`). It publishes the
//! wheel odometry as the body's pose, levelled and at the map's scale,
//! continuous wherever a new estimate takes over, and refits every
//! `refine_s` seconds. Before the first estimate it passes the wheel
//! odometry through. With `calibration_file`, a saved calibration applies
//! from the start, and each new estimate is saved there. It also publishes
//! the calibrated odometry on `/tf`, so that `map -> odom -> base_link`
//! agrees with `map_pose`: the wheel driver's own `odom -> base_link` must
//! then be off.

use std::process;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use visloc_core::geometry::SE3;
use visloc_localization::{WheelCalibrationConfig, WheelCalibrationEstimate};
use visloc_ros2::dds::{
    output_qos, publish_logged, sensor_qos, spawn_receiver, RosNode, StaticTfListener,
};
use visloc_ros2::map_localize::se3_from_ros;
use visloc_ros2::msgs::{Odometry, PoseWithCovarianceStamped, TfMessage};
use visloc_ros2::params::NodeArgs;
use visloc_ros2::vio::ros_pose;
use visloc_ros2::wheel_odom::{odom_tf, SavedCalibration, WheelOdomCalibrator, WheelOutput};

const USAGE: &str = "\
visloc_wheel_calibration_node -- wheel odometry calibrated against the VIO and the map while driving

USAGE:
  visloc_wheel_calibration_node [options]

PARAMETERS (--name value  or  -p name:=value):
  calibration_file     JSON calibration: applied from the start if it exists, written
                       at each new estimate                                         [none]
  min_turn_deg         the offset is fitted once the wheel odometry has turned this
                       much in total ...                                            [180]
  min_travel_m         ... and travelled this far (the scale: map localizations
                       spanning this far) ...                                       [1.0]
  min_pairs            ... over this many wheel/VIO pose pairs                      [200]
  min_fixes            the map's scale and level need this many localizations       [20]
  refine_s             refit this often (s); 0 keeps the first estimate (or the
                       saved one)                                                     [5]
  window_s             fit over the last this many seconds; 0 keeps every pair     [120]
  level                level the body frame against gravity (level floor)          [true]
  apply_scale          scale the output to the map's (from its localizations)      [true]
  publish_uncalibrated pass the wheel odometry through until calibrated             [true]
  publish_tf           publish the output's frame -> child frame on /tf (turn the wheel
                       driver's off)                                               [true]
  domain_id            DDS domain (default: $ROS_DOMAIN_ID or 0)
  node_name            node name                                       [visloc_wheel_calibration]

TOPICS (remap with --remap from=to or -r from:=to):
  in : wheel_odom (nav_msgs/Odometry, the wheel odometry: planar, child_frame_id = the
       robot's body, e.g. base_link), vio_odom (nav_msgs/Odometry, visloc_vio_node's
       odom; optional), map_fix (geometry_msgs/PoseWithCovarianceStamped,
       visloc_map_localize_node's localizations of the body, with extrinsics:=tf;
       optional), /tf_static (tf2_msgs/TFMessage: the VIO's child frame <- the body)
  out: odom (nav_msgs/Odometry, the body's pose at the map's scale, in the wheel
       odometry's frame; e.g. visloc_map_localize_node's odom with extrinsics:=tf),
       /tf (the same pose: wheel odometry frame -> body)
";

fn main() {
    if let Err(error) = run() {
        eprintln!("visloc_wheel_calibration_node: {error}");
        process::exit(1);
    }
}

/// What the receive threads share.
struct State {
    calibrator: WheelOdomCalibrator,
    /// Child frames of the wheel and VIO odometries, once seen.
    wheel_frame: Option<String>,
    vio_frame: Option<String>,
    /// Sources of the last estimate logged, and when.
    logged: Option<(String, Instant)>,
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
        min_fixes: args.parsed_or("min_fixes", defaults.min_fixes)?,
        window_s: args.parsed_or("window_s", 120.0)?,
        level: args.flag("level", defaults.level)?,
        ..defaults
    };
    let refine_s = args.parsed_or("refine_s", 5.0f64)?;
    if !(refine_s >= 0.0 && refine_s.is_finite() && config.window_s >= 0.0) {
        return Err("refine_s and window_s must be non-negative numbers".into());
    }
    let apply_scale = args.flag("apply_scale", true)?;
    let publish_uncalibrated = args.flag("publish_uncalibrated", true)?;
    let publish_tf = args.flag("publish_tf", true)?;
    let domain_id = args.domain_id()?;
    let node_name = args
        .node_name
        .clone()
        .unwrap_or_else(|| args.string_or("node_name", "visloc_wheel_calibration"));
    let namespace = args.namespace.clone().unwrap_or_else(|| "/".into());
    let wheel_topic = args.topic("wheel_odom");
    let vio_topic = args.topic("vio_odom");
    let fix_topic = args.topic("map_fix");
    let out_topic = args.topic("odom");
    let tf_topic = args.topic("/tf");
    let tf_static_topic = args.topic("/tf_static");
    args.finish()?;

    let mut calibrator = WheelOdomCalibrator::new(config)
        .with_scale(apply_scale)
        .with_refine_s(refine_s);
    if let Some(path) = calibration_file
        .as_ref()
        .filter(|path| std::path::Path::new(path).exists())
    {
        let text =
            std::fs::read_to_string(path).map_err(|error| format!("reading {path}: {error}"))?;
        let saved = SavedCalibration::from_json(&text)?;
        eprintln!(
            "visloc_wheel_calibration_node: calibration from {path}: offset ({:.1}, {:.1}) cm, \
             scale {:.4}, tilt {:.2} deg",
            saved.offset_m[0] * 100.0,
            saved.offset_m[1] * 100.0,
            saved.scale,
            saved.tilt_deg
        );
        calibrator = calibrator.with_prior(saved.estimate());
    }
    let state = Arc::new(Mutex::new(State {
        calibrator,
        wheel_frame: None,
        vio_frame: None,
        logged: None,
    }));

    let mut ros = RosNode::new(domain_id, &namespace, &node_name)?;
    let wheel_sub = ros.subscription::<Odometry>(&wheel_topic, sensor_qos(50))?;
    let vio_sub = ros.subscription::<Odometry>(&vio_topic, sensor_qos(50))?;
    let fix_sub = ros.subscription::<PoseWithCovarianceStamped>(&fix_topic, sensor_qos(50))?;
    let odom_pub = ros.publisher::<Odometry>(&out_topic, output_qos(50))?;
    let tf_pub = publish_tf
        .then(|| ros.publisher::<TfMessage>(&tf_topic, output_qos(100)))
        .transpose()?;
    let mut tf_static = StaticTfListener::new(&mut ros, &tf_static_topic)?;
    let transforms = Arc::clone(tf_static.transforms());

    {
        let state = Arc::clone(&state);
        spawn_receiver("vio-rx", vio_sub, move |msg: Odometry| {
            let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
            if state.vio_frame.is_none() {
                state.vio_frame = Some(msg.child_frame_id.clone());
            }
            state
                .calibrator
                .add_vio(msg.header.stamp.to_nanos(), &se3_from_ros(&msg.pose.pose));
        })?;
    }
    {
        let state = Arc::clone(&state);
        spawn_receiver("fix-rx", fix_sub, move |msg: PoseWithCovarianceStamped| {
            let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
            state
                .calibrator
                .add_fix(msg.header.stamp.to_nanos(), &se3_from_ros(&msg.pose.pose));
        })?;
    }
    {
        let state = Arc::clone(&state);
        spawn_receiver("wheel-rx", wheel_sub, move |msg: Odometry| {
            let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
            if state.wheel_frame.is_none() {
                state.wheel_frame = Some(msg.child_frame_id.clone());
            }
            let output = state
                .calibrator
                .wheel(msg.header.stamp.to_nanos(), &se3_from_ros(&msg.pose.pose));
            if let WheelOutput::Updated(_, estimate) = &output {
                log_estimate(&mut state, estimate);
                if let Some(path) = &calibration_file {
                    if let Err(error) =
                        std::fs::write(path, SavedCalibration::from(estimate).to_json())
                    {
                        eprintln!("visloc_wheel_calibration_node: writing {path}: {error}");
                    }
                }
            }
            if matches!(output, WheelOutput::PassThrough(_)) && !publish_uncalibrated {
                return;
            }
            let mut out = msg;
            out.pose.pose = ros_pose(output.pose());
            if let Some(tf_pub) = &tf_pub {
                publish_logged(tf_pub, odom_tf(&out), "tf");
            }
            publish_logged(&odom_pub, out, "odom");
        })?;
    }

    eprintln!(
        "visloc_wheel_calibration_node: domain {domain_id}, node {}; in: {wheel_topic} \
         {vio_topic} {fix_topic}; out: {out_topic}{}; refitting every {refine_s} s",
        visloc_ros2::dds::qualified_name(&namespace, &node_name),
        if publish_tf {
            format!(" {tf_topic}")
        } else {
            String::new()
        },
    );
    let mut last_status = Instant::now();
    loop {
        std::thread::sleep(Duration::from_millis(500));
        let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
        // Pairing with the VIO needs where its frame sits on the body.
        if !state.calibrator.has_imu_from_body() {
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
                        state.calibrator.set_imu_from_body(vio_from_body);
                        eprintln!(
                            "visloc_wheel_calibration_node: pairing with the VIO \
                             ({vio} <- {wheel} from /tf_static)"
                        );
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
            if state.wheel_frame.is_none() {
                eprintln!("visloc_wheel_calibration_node: waiting for wheel_odom");
            } else if state.calibrator.estimate().is_none() {
                let progress = state.calibrator.progress();
                eprintln!(
                    "visloc_wheel_calibration_node: collecting: {} VIO pairs, {:.0} deg turned; \
                     {} map localizations over {:.2} m",
                    progress.pairs, progress.turned_deg, progress.fixes, progress.fix_travel_m
                );
            }
        }
    }
}

/// Logs the first estimate, then one whose sources differ from the last
/// logged or a minute after it.
fn log_estimate(state: &mut State, estimate: &WheelCalibrationEstimate) {
    let sources = format!(
        "offset from {}, scale from {}, level from {}",
        estimate.offset_from.name(),
        estimate.scale_from.name(),
        estimate.level_from.name()
    );
    if state
        .logged
        .as_ref()
        .is_some_and(|(logged, at)| *logged == sources && at.elapsed() < Duration::from_secs(60))
    {
        return;
    }
    eprintln!(
        "visloc_wheel_calibration_node: estimate over {} VIO pairs ({:.0} deg turned) and {} map \
         localizations: the body sits ({:.1}, {:.1}) cm from the wheel odometry's point and \
         leans {:.2} deg; scale {:.4} of the wheels; {sources}",
        estimate.pairs,
        estimate.turned_deg,
        estimate.fixes,
        estimate.offset.x * 100.0,
        estimate.offset.y * 100.0,
        estimate.tilt_deg,
        estimate.scale,
    );
    state.logged = Some((sources, Instant::now()));
}
