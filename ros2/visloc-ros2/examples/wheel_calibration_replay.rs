//! Replays recorded odometries through `WheelOdomCalibrator`, as
//! `visloc_wheel_calibration_node` would see them, without ROS:
//!
//! ```text
//! wheel_calibration_replay --wheel odom.tum --imu-from-body tx,ty,tz,qx,qy,qz,qw \
//!   [--imu imu0/data.csv] [--vio trajectory_vio.tum] [--fixes localization.tum] \
//!   [--prior calibration.json] \
//!   [--refine-s 5] [--window-s 120] [--segment-s 5] [--fix-segment-s 30] \
//!   [--fix-delay-s 0.2] [--no-scale] \
//!   --out calibrated.tum [--estimate final.json]
//! ```
//!
//! `--wheel` holds the body's (e.g. `base_link`) wheel odometry poses. With
//! `--imu` (EuRoC's `imu0/data.csv`: stamp, gyro, accelerometer), their
//! heading comes from the gyro, as the node's with an `imu` topic. `--vio`
//! the VIO's IMU poses and `--fixes` map localizations of the IMU
//! (`basalt_euroc_localize_demo`'s `trajectory_vio.tum` and
//! `localization.tum`), turned into body poses with `--imu-from-body`
//! (`T_imu_body`). Each localization is handed over `--fix-delay-s` after its
//! image, as the map node publishes it once localized. The output is the
//! calibrated body pose per wheel pose (TUM, seconds), for
//! `basalt_euroc_localize_demo --odometry`; each new estimate goes to stderr.

use std::collections::VecDeque;
use std::fs;
use std::process;

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use visloc_core::geometry::SE3;
use visloc_localization::{GyroWheelConfig, GyroWheelOdometry, WheelCalibrationConfig};
use visloc_ros2::wheel_odom::{SavedCalibration, WheelOdomCalibrator, WheelOutput};

struct Args {
    wheel: String,
    imu: Option<String>,
    vio: Option<String>,
    fixes: Option<String>,
    prior: Option<String>,
    imu_from_body: SE3,
    refine_s: f64,
    window_s: f64,
    segment_s: f64,
    fix_segment_s: f64,
    fix_delay_s: f64,
    apply_scale: bool,
    out: String,
    estimate: Option<String>,
}

fn parse_pose(text: &str) -> Result<SE3, String> {
    let values = text
        .split(',')
        .map(|value| value.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("{text:?}: {error}"))?;
    let [tx, ty, tz, qx, qy, qz, qw]: [f64; 7] = values
        .try_into()
        .map_err(|_| format!("{text:?}: expected tx,ty,tz,qx,qy,qz,qw"))?;
    Ok(SE3::new(
        UnitQuaternion::from_quaternion(Quaternion::new(qw, qx, qy, qz)),
        Vector3::new(tx, ty, tz),
    ))
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        wheel: String::new(),
        imu: None,
        vio: None,
        fixes: None,
        prior: None,
        imu_from_body: SE3::identity(),
        refine_s: 5.0,
        window_s: 120.0,
        segment_s: WheelCalibrationConfig::default().segment_s,
        fix_segment_s: WheelCalibrationConfig::default().fix_segment_s,
        fix_delay_s: 0.2,
        apply_scale: true,
        out: String::new(),
        estimate: None,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(option) = iter.next() {
        let mut value = || iter.next().ok_or(format!("{option} needs a value"));
        let number = |text: String| {
            text.parse::<f64>()
                .map_err(|_| format!("{option}: expected a number"))
        };
        match option.as_str() {
            "--wheel" => args.wheel = value()?,
            "--imu" => args.imu = Some(value()?),
            "--vio" => args.vio = Some(value()?),
            "--fixes" => args.fixes = Some(value()?),
            "--prior" => args.prior = Some(value()?),
            "--imu-from-body" => args.imu_from_body = parse_pose(&value()?)?,
            "--refine-s" => args.refine_s = number(value()?)?,
            "--window-s" => args.window_s = number(value()?)?,
            "--segment-s" => args.segment_s = number(value()?)?,
            "--fix-segment-s" => args.fix_segment_s = number(value()?)?,
            "--fix-delay-s" => args.fix_delay_s = number(value()?)?,
            "--no-scale" => args.apply_scale = false,
            "--out" => args.out = value()?,
            "--estimate" => args.estimate = Some(value()?),
            other => return Err(format!("unknown option {other}")),
        }
    }
    if args.wheel.is_empty() || args.out.is_empty() {
        return Err("--wheel and --out are required".into());
    }
    Ok(args)
}

/// TUM poses (`t tx ty tz qx qy qz qw`, seconds) with nanosecond stamps.
fn read_tum(path: &str) -> Result<Vec<(i64, SE3)>, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let mut poses = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 8 {
            return Err(format!("{path}: bad line {line:?}"));
        }
        let (seconds, fraction) = fields[0].split_once('.').unwrap_or((fields[0], "0"));
        let fraction = format!("{:0<9}", &fraction[..fraction.len().min(9)]);
        let stamp = seconds.parse::<i64>().map_err(|e| e.to_string())? * 1_000_000_000
            + fraction.parse::<i64>().map_err(|e| e.to_string())?;
        let pose = parse_pose(&fields[1..8].join(","))?;
        poses.push((stamp, pose));
    }
    Ok(poses)
}

/// Stamp (ns), angular velocity (rad/s), specific force (m/s^2).
type ImuRow = (i64, Vector3<f64>, Vector3<f64>);

/// EuRoC IMU rows (`t [ns], gyro xyz, accelerometer xyz`).
fn read_imu(path: &str) -> Result<Vec<ImuRow>, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let mut samples = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let bad = || format!("{path}: bad line {line:?}");
        let (stamp, rest) = line.split_once(',').ok_or_else(bad)?;
        let values = rest
            .split(',')
            .map(|field| field.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| bad())?;
        let [wx, wy, wz, ax, ay, az]: [f64; 6] = values.try_into().map_err(|_| bad())?;
        samples.push((
            stamp.trim().parse::<i64>().map_err(|_| bad())?,
            Vector3::new(wx, wy, wz),
            Vector3::new(ax, ay, az),
        ));
    }
    Ok(samples)
}

fn tum_line(stamp: i64, pose: &SE3) -> String {
    let q = pose.rotation.quaternion();
    let t = pose.translation;
    format!(
        "{}.{:09} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9}\n",
        stamp / 1_000_000_000,
        stamp % 1_000_000_000,
        t.x,
        t.y,
        t.z,
        q.i,
        q.j,
        q.k,
        q.w
    )
}

fn main() {
    if let Err(error) = run() {
        eprintln!("wheel_calibration_replay: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let wheel = read_tum(&args.wheel)?;
    let imu = args.imu.as_deref().map(read_imu).transpose()?;
    let vio = args.vio.as_deref().map(read_tum).transpose()?;
    let fixes = args.fixes.as_deref().map(read_tum).transpose()?;
    let config = WheelCalibrationConfig {
        window_s: args.window_s,
        segment_s: args.segment_s,
        fix_segment_s: args.fix_segment_s,
        ..WheelCalibrationConfig::default()
    };
    let mut calibrator = WheelOdomCalibrator::new(config)
        .with_scale(args.apply_scale)
        .with_refine_s(args.refine_s);
    if let Some(path) = &args.prior {
        let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
        calibrator = calibrator.with_prior(SavedCalibration::from_json(&text)?.estimate());
    }
    if vio.is_some() {
        calibrator.set_imu_from_body(args.imu_from_body.clone());
    }
    let mut vio: VecDeque<(i64, SE3)> = vio.unwrap_or_default().into();
    let mut fixes: VecDeque<(i64, SE3)> = fixes
        .unwrap_or_default()
        .into_iter()
        .map(|(stamp, map_from_imu)| (stamp, map_from_imu.compose(&args.imu_from_body)))
        .collect();
    let mut imu: VecDeque<_> = imu.unwrap_or_default().into();
    let mut gyro = GyroWheelOdometry::new(GyroWheelConfig::default());
    let use_gyro = !imu.is_empty();
    let delay_ns = (args.fix_delay_s * 1e9) as i64;
    let mut out = String::from("# timestamp tx ty tz qx qy qz qw (calibrated body poses)\n");
    let mut last = None;
    for (stamp, odom_from_wheel) in &wheel {
        while vio.front().is_some_and(|(t, _)| t <= stamp) {
            let (t, pose) = vio.pop_front().expect("front");
            calibrator.add_vio(t, &pose);
        }
        while fixes.front().is_some_and(|(t, _)| t + delay_ns <= *stamp) {
            let (t, pose) = fixes.pop_front().expect("front");
            calibrator.add_fix(t, &pose);
        }
        while imu.front().is_some_and(|(t, _, _)| t <= stamp) {
            let (t, angular_velocity, acceleration) = imu.pop_front().expect("front");
            gyro.add_imu(t, &angular_velocity, &acceleration);
        }
        let odom_from_wheel = if use_gyro {
            gyro.wheel(*stamp, odom_from_wheel)
        } else {
            odom_from_wheel.clone()
        };
        let output = calibrator.wheel(*stamp, &odom_from_wheel);
        if let WheelOutput::Updated(_, estimate) = &output {
            eprintln!(
                "{:.1} s: offset ({:.1}, {:.1}) cm from {}, scale {:.4} from {}, tilt {:.2} deg from {} \
                 ({} VIO pairs, {:.0} deg; {} fixes)",
                (stamp - wheel[0].0) as f64 * 1e-9,
                estimate.offset.x * 100.0,
                estimate.offset.y * 100.0,
                estimate.offset_from.name(),
                estimate.scale,
                estimate.scale_from.name(),
                estimate.tilt_deg,
                estimate.level_from.name(),
                estimate.pairs,
                estimate.turned_deg,
                estimate.fixes,
            );
        }
        out.push_str(&tum_line(*stamp, output.pose()));
        last = calibrator.estimate().cloned();
    }
    if let Some(bias) = gyro.bias() {
        eprintln!(
            "gyro bias about the vertical {:.3} deg/min",
            bias.to_degrees() * 60.0
        );
    }
    fs::write(&args.out, out).map_err(|error| format!("{}: {error}", args.out))?;
    if let (Some(path), Some(estimate)) = (&args.estimate, last) {
        fs::write(path, SavedCalibration::from(&estimate).to_json())
            .map_err(|error| format!("{path}: {error}"))?;
    }
    Ok(())
}
