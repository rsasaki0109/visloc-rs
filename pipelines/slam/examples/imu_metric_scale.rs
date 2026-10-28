//! Metric scale of a COLMAP model from an IMU log
//! ([`visloc_slam::estimate_imu_scale`]).
//!
//! cargo run --release -p visloc-slam --example imu_metric_scale -- \
//!     <model dir> <imu.csv> [--frames TEMPLATE] [--divisor N] [--remainder R] \
//!     [--period S [--offset S]] [--no-centre-term] [--runs]
//!
//! The model is `images.bin` or `images.txt` (only the poses are read). The
//! frames used are the images whose names match `TEMPLATE`, where `{n}` stands
//! for the frame number's digits (default `{n}` anywhere: the last run of
//! digits in the name); with `--divisor N`, only numbers `n` with
//! `n % N == R`, as frame number `(n - R) / N`. On OverMaps360_50's
//! perspective model, `--frames '{n}_perspective_00000000.jpg' --divisor 2`
//! takes view 0 of lens 0 (`--remainder 1`: lens 1). The IMU is an Insta360
//! `.insv` (its trailer) or a CSV file (SI, EuRoC, or OverMaps360's raw
//! counts; `visloc_io::imu_csv::read_imu_file`). Without `--period`, the
//! frame timing is searched.

use std::path::Path;

use nalgebra::Matrix3;
use visloc_io::colmap::{parse_images_bin, parse_images_bin_names, parse_images_txt};
use visloc_io::imu_csv::read_imu_file;
use visloc_slam::{estimate_imu_scale, FrameTiming, ImuSamples, ImuScaleConfig, ScaleFrame};

/// The frame number in `name` per `template` (`{n}` = digits), if it matches.
fn frame_number(name: &str, template: &str) -> Option<i64> {
    let file = name.rsplit('/').next().unwrap_or(name);
    if template == "{n}" {
        let digits: String = file
            .chars()
            .rev()
            .skip_while(|c| !c.is_ascii_digit())
            .take_while(|c| c.is_ascii_digit())
            .collect();
        return digits.chars().rev().collect::<String>().parse().ok();
    }
    let (prefix, suffix) = template.split_once("{n}")?;
    let middle = file.strip_prefix(prefix)?.strip_suffix(suffix)?;
    (!middle.is_empty() && middle.bytes().all(|b| b.is_ascii_digit()))
        .then(|| middle.parse().ok())
        .flatten()
}

/// `(name, world-from-camera rotation, centre)` of every posed image.
fn read_poses(dir: &Path) -> Vec<(String, Matrix3<f64>, nalgebra::Vector3<f64>)> {
    let bin = dir.join("images.bin");
    let (keyframes, names) = if bin.exists() {
        let bytes = std::fs::read(&bin).expect("read images.bin");
        let kf = parse_images_bin(&bytes).expect("parse images.bin");
        let names = parse_images_bin_names(&bytes).expect("parse images.bin");
        (kf, names)
    } else {
        let text = std::fs::read_to_string(dir.join("images.txt")).expect("read images.txt");
        let kf = parse_images_txt(&text).expect("parse images.txt");
        // Every other non-comment line: IMAGE_ID ... NAME.
        let names = text
            .lines()
            .filter(|l| !l.starts_with('#'))
            .step_by(2)
            .filter_map(|l| {
                let t: Vec<&str> = l.split_whitespace().collect();
                (t.len() >= 10).then(|| (t[0].parse().unwrap(), t[9].to_string()))
            })
            .collect();
        (kf, names)
    };
    let names: std::collections::HashMap<u64, String> = names.into_iter().collect();
    keyframes
        .iter()
        .filter_map(|kf| {
            let pose = kf.frame.pose.as_ref()?;
            let r_wc = pose
                .world_to_camera
                .rotation
                .to_rotation_matrix()
                .matrix()
                .transpose();
            Some((
                names.get(&kf.frame.id)?.clone(),
                r_wc,
                pose.camera_center_world().coords,
            ))
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: imu_metric_scale <model dir> <imu.csv> [--frames TEMPLATE] [--divisor N] [--remainder R] [--period S [--offset S]] [--no-centre-term] [--runs]");
        std::process::exit(2);
    }
    let opt = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let flag = |name: &str| args.iter().any(|a| a == name);
    let template = opt("--frames").unwrap_or_else(|| "{n}".into());
    let divisor: i64 = opt("--divisor").map_or(1, |v| v.parse().expect("--divisor"));
    let remainder: i64 = opt("--remainder").map_or(0, |v| v.parse().expect("--remainder"));

    let t = std::time::Instant::now();
    let poses = read_poses(Path::new(&args[1]));
    let frames: Vec<ScaleFrame> = poses
        .iter()
        .filter_map(|(name, rotation, centre)| {
            let n = frame_number(name, &template)?;
            (n.rem_euclid(divisor) == remainder).then(|| ScaleFrame {
                number: (n - remainder) / divisor,
                rotation: *rotation,
                centre: *centre,
            })
        })
        .collect();
    let (csv, source) = read_imu_file(Path::new(&args[2])).expect("read IMU");
    println!(
        "{} posed images, {} frames match; {} IMU samples ({source}, {:.1} s); read in {:.1}s",
        poses.len(),
        frames.len(),
        csv.time.len(),
        csv.time.last().unwrap() - csv.time[0],
        t.elapsed().as_secs_f64()
    );
    let imu = ImuSamples {
        time: csv.time,
        gyro: csv.gyro,
        accel: csv.accel,
    };
    let mut cfg = ImuScaleConfig {
        rolling_shutter_term: !flag("--no-centre-term"),
        ..Default::default()
    };
    if let Some(period) = opt("--period") {
        let period: f64 = period.parse().expect("--period");
        cfg.timing = match opt("--offset") {
            Some(o) => FrameTiming::Known {
                period,
                offset: o.parse().expect("--offset"),
            },
            None => FrameTiming::Search {
                nominal_period: Some(period),
            },
        };
    }
    let t = std::time::Instant::now();
    match estimate_imu_scale(&imu, &frames, &cfg) {
        Ok(e) => {
            println!(
                "timing: every {:.7} s from {:+.4} s, drift {:+.1} / {:+.1} ms; rotation residual {:.3} deg; {} speed jumps",
                e.period, e.offset, e.drift_ms[0], e.drift_ms[1], e.rotation_residual_deg, e.speed_jumps
            );
            if flag("--runs") {
                for r in &e.runs {
                    println!(
                        "  run {:7.1}-{:7.1} s  {:3} intervals  scale {:.4}  |g| {:.3}{}{}",
                        r.start,
                        r.end,
                        r.intervals,
                        r.scale,
                        r.gravity,
                        r.centre_term.map_or(String::new(), |c| format!(
                            "  c ({:.4}, {:.4}, {:.4})",
                            c.x, c.y, c.z
                        )),
                        if r.kept { "" } else { "  rejected" }
                    );
                }
            }
            println!(
                "scale {:.4} metres per model unit (SD {:.4} over {}/{} runs), |g| {:.3}; {:.1}s",
                e.scale,
                e.scale_std,
                e.runs_kept(),
                e.runs.len(),
                e.gravity_median,
                t.elapsed().as_secs_f64()
            );
        }
        Err(err) => {
            eprintln!("IMU scale failed: {err}");
            std::process::exit(1);
        }
    }
}
