//! GNSS + stereo-VO joint fusion evaluated on a KITTI raw drive.
//!
//! Ground truth is the drive's RTK-grade OXTS GPS/INS (`oxts/`), converted to a
//! local ENU frame at the first sample. The VO input is a KITTI-format pose
//! file from `online_slam_stereo_vo_kitti_demo` run on the drive's rectified
//! grey stereo pair (`image_00` / `image_01`). The GNSS input is either the
//! OXTS positions at their native accuracy (`--gnss native`) or the same
//! positions degraded to consumer-receiver quality (`--gnss degraded`, the
//! default): subsampled, with white plus first-order Gauss–Markov
//! (time-correlated) noise, multipath jumps, and a dropout, all from a fixed
//! seed.
//!
//! ```text
//! # 1. Rectified stereo projections as an odometry-style calib.txt (P0/P1):
//! cargo run --release --example kitti_raw_gnss_vo_fusion -- \
//!     --date-dir <raw>/2011_09_30 --write-vo-calib calib.txt
//! # 2. Stereo VO (see docs/gnss_fusion.md for the exact flags), then:
//! cargo run --release --example kitti_raw_gnss_vo_fusion -- \
//!     --date-dir <raw>/2011_09_30 --drive 2011_09_30_drive_0027 \
//!     --vo-poses <vo-out>/vo_poses.txt [--window 300] [--gnss native] \
//!     [--seed 1] [--out-dir <dir>]
//! ```
//!
//! Reported (all RMS over camera centres vs OXTS ground truth, metres):
//! VO alone after the best-fit SE(3) alignment onto the ground truth; the raw
//! GNSS fixes against the true antenna (OXTS) position; and the fused ENU
//! trajectory with no alignment at all (batch, and sliding window with
//! `--window`).

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use nalgebra::{Isometry3, Point3, Rotation3, UnitQuaternion, Vector3};
use visloc_rs::fusion::{GnssMeasurement, MeasurementBuffer, Timed, TimedPose, Timestamp};
use visloc_rs::io::kitti_imu::{parse_kitti_oxts_timestamps_txt, read_kitti_oxts_dir};
use visloc_rs::io::kitti_raw_gnss::{
    kitti_oxts_body_to_local_level, kitti_oxts_gnss_measurements, KittiRawStereoCalibration,
    LocalEnuFrame,
};
use visloc_rs::slam::gnss_fusion::{
    aligned_position_rmse, fuse_gnss_with_visual_odometry, position_rmse, transform_trajectory,
    AlignmentRotationDof, FrameGnssStatus, GnssRobustMode, GnssVoFusionConfig, GnssVoFusionResult,
};
use visloc_rs::slam::{RobustKernel, GNSS_CHI2_3DOF_999};
use visloc_rs::tracking::PoseTrajectory;
use visloc_rs::{Pose, SE3};

#[derive(Debug, Clone)]
struct Args {
    date_dir: PathBuf,
    drive: Option<String>,
    write_vo_calib: Option<PathBuf>,
    vo_poses: Option<PathBuf>,
    vo_start_frame: usize,
    vo_stride: usize,
    gnss_native: bool,
    gnss_rate_hz: f64,
    sigma_h: f64,
    sigma_v_ratio: f64,
    correlated_fraction: f64,
    gm_tau_s: f64,
    outlier_rate: f64,
    outlier_min_m: f64,
    outlier_max_m: f64,
    dropout_start_fraction: f64,
    dropout_duration_s: f64,
    seed: u64,
    gravity_yaw: bool,
    window: Option<usize>,
    robust: String,
    vo_translation_ratio: Option<f64>,
    vo_rotation_sigma: Option<f64>,
    gate_at_fix_accuracy: bool,
    out_dir: Option<PathBuf>,
}

fn usage() -> ! {
    eprintln!(
        "usage: kitti_raw_gnss_vo_fusion --date-dir <raw/2011_09_30> \
         (--write-vo-calib <calib.txt> | --drive <2011_09_30_drive_0027> --vo-poses <vo_poses.txt>)\n\
         \x20 [--vo-start-frame 0] [--vo-stride 1] [--gnss degraded|native] [--gnss-rate-hz 1]\n\
         \x20 [--sigma-h 2.0] [--sigma-v-ratio 2.0] [--correlated-fraction 0.8] [--gm-tau-s 45]\n\
         \x20 [--outlier-rate 0.03] [--outlier-min-m 10] [--outlier-max-m 30]\n\
         \x20 [--dropout-start-fraction 0.5] [--dropout-duration-s 20] [--seed 1]\n\
         \x20 [--alignment full|gravity-yaw] [--window <frames>] [--robust gnc|huber|none]\n\
         \x20 [--vo-translation-ratio <r>] [--vo-rotation-sigma <rad>]\n\
         \x20 [--gate-at-fix-accuracy] [--out-dir <dir>]"
    );
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut args = Args {
        date_dir: PathBuf::new(),
        drive: None,
        write_vo_calib: None,
        vo_poses: None,
        vo_start_frame: 0,
        vo_stride: 1,
        gnss_native: false,
        gnss_rate_hz: 1.0,
        sigma_h: 2.0,
        sigma_v_ratio: 2.0,
        correlated_fraction: 0.8,
        gm_tau_s: 45.0,
        outlier_rate: 0.03,
        outlier_min_m: 10.0,
        outlier_max_m: 30.0,
        dropout_start_fraction: 0.5,
        dropout_duration_s: 20.0,
        seed: 1,
        gravity_yaw: false,
        window: None,
        robust: "gnc".to_owned(),
        vo_translation_ratio: None,
        vo_rotation_sigma: None,
        gate_at_fix_accuracy: false,
        out_dir: None,
    };
    let mut iter = env::args().skip(1);
    while let Some(flag) = iter.next() {
        let mut value = || iter.next().unwrap_or_else(|| usage());
        macro_rules! num {
            () => {
                value().parse().unwrap_or_else(|_| usage())
            };
        }
        match flag.as_str() {
            "--date-dir" => args.date_dir = PathBuf::from(value()),
            "--drive" => args.drive = Some(value()),
            "--write-vo-calib" => args.write_vo_calib = Some(PathBuf::from(value())),
            "--vo-poses" => args.vo_poses = Some(PathBuf::from(value())),
            "--vo-start-frame" => args.vo_start_frame = num!(),
            "--vo-stride" => args.vo_stride = num!(),
            "--gnss" => match value().as_str() {
                "degraded" => args.gnss_native = false,
                "native" => args.gnss_native = true,
                _ => usage(),
            },
            "--gnss-rate-hz" => args.gnss_rate_hz = num!(),
            "--sigma-h" => args.sigma_h = num!(),
            "--sigma-v-ratio" => args.sigma_v_ratio = num!(),
            "--correlated-fraction" => args.correlated_fraction = num!(),
            "--gm-tau-s" => args.gm_tau_s = num!(),
            "--outlier-rate" => args.outlier_rate = num!(),
            "--outlier-min-m" => args.outlier_min_m = num!(),
            "--outlier-max-m" => args.outlier_max_m = num!(),
            "--dropout-start-fraction" => args.dropout_start_fraction = num!(),
            "--dropout-duration-s" => args.dropout_duration_s = num!(),
            "--seed" => args.seed = num!(),
            "--alignment" => match value().as_str() {
                "full" => args.gravity_yaw = false,
                "gravity-yaw" => args.gravity_yaw = true,
                _ => usage(),
            },
            "--window" => args.window = Some(num!()),
            "--robust" => args.robust = value(),
            "--vo-translation-ratio" => args.vo_translation_ratio = Some(num!()),
            "--vo-rotation-sigma" => args.vo_rotation_sigma = Some(num!()),
            "--gate-at-fix-accuracy" => args.gate_at_fix_accuracy = true,
            "--out-dir" => args.out_dir = Some(PathBuf::from(value())),
            _ => usage(),
        }
    }
    if args.date_dir.as_os_str().is_empty()
        || (args.write_vo_calib.is_none() && (args.drive.is_none() || args.vo_poses.is_none()))
        || args.vo_stride == 0
        || !(args.correlated_fraction >= 0.0 && args.correlated_fraction <= 1.0)
        || args.gnss_rate_hz <= 0.0
    {
        usage();
    }
    args
}

/// SplitMix64: a tiny deterministic generator, so the degraded GNSS stream is
/// reproducible from `--seed` with no extra dependency.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal (Box–Muller).
    fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// One GNSS fix with its true antenna position and injected-error bookkeeping.
struct EvalFix {
    measurement: GnssMeasurement,
    truth: Vector3<f64>,
    multipath: bool,
}

/// OXTS positions subsampled to `rate_hz`, with per-axis noise of standard
/// deviation `sigma_h` (east/north) and `sigma_h·sigma_v_ratio` (up), of which
/// `correlated_fraction` of the variance is a first-order Gauss–Markov process
/// with time constant `gm_tau_s` and the rest white; multipath bursts (1–3
/// consecutive fixes offset by `outlier_min_m..outlier_max_m` horizontally) on
/// about `outlier_rate` of the fixes; and no fixes for `dropout_duration_s`
/// starting at `dropout_start_fraction` of the drive. Each fix reports
/// `sigma_h` / `sigma_h·sigma_v_ratio` as its accuracy.
fn degraded_gnss(fixes: &[GnssMeasurement], args: &Args) -> Vec<EvalFix> {
    let mut rng = SplitMix64(args.seed);
    let t0 = fixes[0].timestamp.as_nanoseconds();
    let t_end = fixes[fixes.len() - 1].timestamp.as_nanoseconds();
    // Subsample on the timestamp grid (the OXTS stream is ~10 Hz).
    let period_ns = (1e9 / args.gnss_rate_hz).round() as i128;
    let mut selected = Vec::new();
    let mut next_ns = t0;
    for fix in fixes {
        let t = fix.timestamp.as_nanoseconds();
        if t + period_ns / 20 >= next_ns {
            selected.push(fix);
            next_ns += period_ns;
            while next_ns <= t {
                next_ns += period_ns;
            }
        }
    }
    let dropout_start = t0 + ((t_end - t0) as f64 * args.dropout_start_fraction) as i128;
    let dropout_end = dropout_start + (args.dropout_duration_s * 1e9) as i128;

    let sigma = Vector3::new(
        args.sigma_h,
        args.sigma_h,
        args.sigma_h * args.sigma_v_ratio,
    );
    let sigma_gm = sigma * args.correlated_fraction.sqrt();
    let sigma_white = sigma * (1.0 - args.correlated_fraction).sqrt();
    let mut gm = Vector3::new(
        rng.normal() * sigma_gm.x,
        rng.normal() * sigma_gm.y,
        rng.normal() * sigma_gm.z,
    );
    let mut previous_t = t0;
    let mut burst_left = 0usize;
    let mut burst_offset = Vector3::zeros();
    let mut output = Vec::new();
    for fix in selected {
        let t = fix.timestamp.as_nanoseconds();
        let dt = (t - previous_t) as f64 * 1e-9;
        previous_t = t;
        if dt > 0.0 {
            let phi = (-dt / args.gm_tau_s.max(1e-9)).exp();
            let drive = (1.0 - phi * phi).sqrt();
            for axis in 0..3 {
                gm[axis] = phi * gm[axis] + drive * sigma_gm[axis] * rng.normal();
            }
        }
        let white = Vector3::new(
            rng.normal() * sigma_white.x,
            rng.normal() * sigma_white.y,
            rng.normal() * sigma_white.z,
        );
        // Mean burst length is 2 fixes, so start bursts at half the rate.
        let start_burst = rng.uniform() < args.outlier_rate / 2.0;
        let burst_length = 1 + (rng.uniform() * 3.0) as usize;
        let magnitude =
            args.outlier_min_m + rng.uniform() * (args.outlier_max_m - args.outlier_min_m);
        let heading = rng.uniform() * std::f64::consts::TAU;
        let vertical = rng.normal() * magnitude / 3.0;
        if burst_left == 0 && start_burst {
            burst_left = burst_length;
            burst_offset = Vector3::new(
                magnitude * heading.cos(),
                magnitude * heading.sin(),
                vertical,
            );
        }
        let multipath = burst_left > 0;
        let error = gm
            + white
            + if multipath {
                burst_offset
            } else {
                Vector3::zeros()
            };
        burst_left = burst_left.saturating_sub(1);
        if (dropout_start..dropout_end).contains(&t) {
            continue;
        }
        let truth = fix.position_world.coords;
        output.push(EvalFix {
            measurement: GnssMeasurement::new(fix.timestamp, Point3::from(truth + error))
                .with_accuracy(Some(args.sigma_h), Some(args.sigma_h * args.sigma_v_ratio)),
            truth,
            multipath,
        });
    }
    output
}

fn native_gnss(fixes: &[GnssMeasurement], args: &Args) -> Vec<EvalFix> {
    let period_ns = (1e9 / args.gnss_rate_hz).round() as i128;
    let t0 = fixes[0].timestamp.as_nanoseconds();
    let mut next_ns = t0;
    let mut output = Vec::new();
    for fix in fixes {
        let t = fix.timestamp.as_nanoseconds();
        if t + period_ns / 20 >= next_ns {
            output.push(EvalFix {
                measurement: fix.clone(),
                truth: fix.position_world.coords,
                multipath: false,
            });
            next_ns += period_ns;
            while next_ns <= t {
                next_ns += period_ns;
            }
        }
    }
    output
}

fn isometry_to_se3(iso: &Isometry3<f64>) -> SE3 {
    SE3::new(iso.rotation, iso.translation.vector)
}

fn timed_camera_to_world(timestamp: Timestamp, camera_to_world: SE3) -> TimedPose {
    Timed::new(
        timestamp,
        Pose {
            world_to_camera: camera_to_world.inverse(),
        },
    )
}

fn rmse_over(estimate: &[TimedPose], truth: &[TimedPose], mask: &[bool]) -> Option<f64> {
    let (sum, count) = estimate
        .iter()
        .zip(truth)
        .zip(mask)
        .filter(|(_, selected)| **selected)
        .fold((0.0, 0usize), |(sum, count), ((a, b), _)| {
            let d = a.value.camera_center_world() - b.value.camera_center_world();
            (sum + d.norm_squared(), count + 1)
        });
    (count > 0).then(|| (sum / count as f64).sqrt())
}

fn horizontal_rmse(estimate: &[TimedPose], truth: &[TimedPose]) -> Option<f64> {
    let n = estimate.len().min(truth.len());
    let sum: f64 = estimate
        .iter()
        .zip(truth)
        .map(|(a, b)| {
            (a.value.camera_center_world() - b.value.camera_center_world())
                .xy()
                .norm_squared()
        })
        .sum();
    (n > 0).then(|| (sum / n as f64).sqrt())
}

fn max_error(estimate: &[TimedPose], truth: &[TimedPose]) -> f64 {
    estimate
        .iter()
        .zip(truth)
        .map(|(a, b)| (a.value.camera_center_world() - b.value.camera_center_world()).norm())
        .fold(0.0, f64::max)
}

fn json_number(value: Option<f64>) -> String {
    match value {
        Some(v) if v.is_finite() => format!("{v:.4}"),
        _ => "null".to_owned(),
    }
}

struct RunReport {
    label: &'static str,
    result: GnssVoFusionResult,
    seconds: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let calibration = KittiRawStereoCalibration::read(&args.date_dir)?;
    if let Some(path) = &args.write_vo_calib {
        fs::write(path, calibration.odometry_calib_txt())?;
        println!("wrote {}", path.display());
        return Ok(());
    }
    let drive = args.drive.clone().expect("checked");
    let drive_dir = args.date_dir.join(format!("{drive}_sync"));

    // Ground truth: OXTS IMU poses in ENU at the first sample.
    let oxts = read_kitti_oxts_dir(drive_dir.join("oxts"))?;
    if oxts.is_empty() {
        return Err("no OXTS records".into());
    }
    let enu = LocalEnuFrame::at_oxts_sample(&oxts[0].sample);
    let imu_to_cam = calibration.imu_to_rectified_cam0();
    let cam_to_imu = isometry_to_se3(&imu_to_cam.inverse());
    let lever_arm = calibration.imu_origin_in_rectified_cam0();
    let image_times = parse_kitti_oxts_timestamps_txt(&fs::read_to_string(
        drive_dir.join("image_00").join("timestamps.txt"),
    )?)?;
    if image_times.len() != oxts.len() {
        return Err(format!(
            "image_00 has {} timestamps but oxts has {} records",
            image_times.len(),
            oxts.len()
        )
        .into());
    }

    // VO poses (camera-to-world, KITTI format), re-expressed relative to the
    // first VO frame.
    let vo_file = PoseTrajectory::read_kitti_poses(args.vo_poses.as_ref().expect("checked"))?;
    let vo_samples = vo_file.samples();
    if vo_samples.len() < 3 {
        return Err("need at least 3 VO poses".into());
    }
    let first_inverse = vo_samples[0].pose.camera_to_world().inverse();
    // `--alignment gravity-yaw`: put the VO map frame level using the first
    // frame's OXTS roll/pitch (what an accelerometer would give; the yaw stays
    // unknown), so the fusion only has to estimate yaw + translation.
    let level_from_cam0 = if args.gravity_yaw {
        let raw0 = &oxts[args.vo_start_frame].sample;
        let level_from_imu = kitti_oxts_body_to_local_level(raw0.roll_rad, raw0.pitch_rad, 0.0);
        let imu_from_cam = cam_to_imu.rotation.to_rotation_matrix();
        let rotation: Rotation3<f64> = level_from_imu * imu_from_cam;
        SE3::new(
            UnitQuaternion::from_rotation_matrix(&rotation),
            Vector3::zeros(),
        )
    } else {
        SE3::identity()
    };
    let mut vo = Vec::with_capacity(vo_samples.len());
    let mut truth = Vec::with_capacity(vo_samples.len());
    for (k, sample) in vo_samples.iter().enumerate() {
        let raw = args.vo_start_frame + k * args.vo_stride;
        if raw >= oxts.len() {
            return Err(format!("VO pose {k} maps to raw frame {raw} beyond the drive").into());
        }
        let timestamp = Timestamp::from_nanoseconds(image_times[raw]);
        let camera_to_vo = first_inverse.compose(&sample.pose.camera_to_world());
        vo.push(timed_camera_to_world(
            timestamp,
            level_from_cam0.compose(&camera_to_vo),
        ));
        let imu_to_enu = isometry_to_se3(&enu.imu_to_enu(&oxts[raw].sample));
        truth.push(timed_camera_to_world(
            timestamp,
            imu_to_enu.compose(&cam_to_imu),
        ));
    }
    let path_length: f64 = truth
        .windows(2)
        .map(|w| (w[1].value.camera_center_world() - w[0].value.camera_center_world()).norm())
        .sum();

    // GNSS stream: the OXTS positions, native or degraded.
    let oxts_fixes = kitti_oxts_gnss_measurements(&oxts, &enu);
    let fixes = if args.gnss_native {
        native_gnss(&oxts_fixes, &args)
    } else {
        degraded_gnss(&oxts_fixes, &args)
    };
    let fix_errors: Vec<f64> = fixes
        .iter()
        .map(|f| (f.measurement.position_world.coords - f.truth).norm())
        .collect();
    let rms = |values: &mut dyn Iterator<Item = f64>| {
        let (sum, n) = values.fold((0.0, 0usize), |(s, n), v| (s + v * v, n + 1));
        (n > 0).then(|| (sum / n as f64).sqrt())
    };
    let gnss_rmse = rms(&mut fix_errors.iter().copied());
    let gnss_clean_rmse = rms(&mut fix_errors
        .iter()
        .zip(&fixes)
        .filter(|(_, f)| !f.multipath)
        .map(|(e, _)| *e));
    let gnss_horizontal_rmse = rms(&mut fixes.iter().map(|f| {
        let d = f.measurement.position_world.coords - f.truth;
        d.xy().norm()
    }));
    let multipath_fixes = fixes.iter().filter(|f| f.multipath).count();
    let buffer = MeasurementBuffer::from_measurements(fixes.iter().map(|f| f.measurement.clone()));

    let mut config = GnssVoFusionConfig::metric_gravity_aligned();
    config.optimizer.rotation_dof = if args.gravity_yaw {
        AlignmentRotationDof::YawOnly
    } else {
        AlignmentRotationDof::Full
    };
    config.lever_arm = lever_arm;
    config.gate_at_fix_accuracy = args.gate_at_fix_accuracy;
    if let Some(ratio) = args.vo_translation_ratio {
        config.vo_noise.translation_sigma_ratio = ratio;
    }
    if let Some(sigma) = args.vo_rotation_sigma {
        config.vo_noise.rotation_sigma = sigma;
    }
    config.optimizer.robust = match args.robust.as_str() {
        "gnc" => GnssRobustMode::default(),
        "huber" => GnssRobustMode::Kernel(RobustKernel::Huber {
            delta: GNSS_CHI2_3DOF_999.sqrt(),
        }),
        "none" => {
            config.optimizer.refine_without_outliers = false;
            GnssRobustMode::None
        }
        _ => usage(),
    };

    let mut runs = Vec::new();
    let started = Instant::now();
    let batch = fuse_gnss_with_visual_odometry(&vo, &buffer, &config)?;
    runs.push(RunReport {
        label: "batch",
        result: batch,
        seconds: started.elapsed().as_secs_f64(),
    });
    if let Some(window) = args.window {
        let mut window_config = config.clone();
        window_config.window = Some(window);
        let started = Instant::now();
        let windowed = fuse_gnss_with_visual_odometry(&vo, &buffer, &window_config)?;
        runs.push(RunReport {
            label: "window",
            result: windowed,
            seconds: started.elapsed().as_secs_f64(),
        });
    }

    let vo_se3_ate = aligned_position_rmse(&vo, &truth, false);
    // Relative rotation error over 10-frame deltas: a check of the
    // camera/IMU calibration chain and the OXTS attitude convention (a wrong
    // extrinsic rotation shows up as a large error whenever the car turns).
    let vo_rotation_rpe_deg = relative_rotation_rmse_deg(&vo, &truth, 10);
    let vo_bootstrap = transform_trajectory(&runs[0].result.bootstrap.alignment, &vo);
    let vo_bootstrap_ate = position_rmse(&vo_bootstrap, &truth);
    let fix_times: Vec<i128> = fixes
        .iter()
        .map(|f| f.measurement.timestamp.as_nanoseconds())
        .collect();
    // Frames more than one GNSS period from any fix (the dropout).
    let gap_ns = (1.5e9 / args.gnss_rate_hz) as i128;
    let dropout_mask: Vec<bool> = vo
        .iter()
        .map(|pose| {
            let t = pose.timestamp.as_nanoseconds();
            let i = fix_times.partition_point(|&ft| ft < t);
            let before = i.checked_sub(1).map(|j| t - fix_times[j]);
            let after = fix_times.get(i).map(|&ft| ft - t);
            before.is_none_or(|d| d > gap_ns) && after.is_none_or(|d| d > gap_ns)
                || before.zip(after).is_some_and(|(b, a)| b + a > gap_ns)
        })
        .collect();
    // Frames within one GNSS period of an injected multipath fix (they may
    // use it directly or through interpolation).
    let period_ns = (1e9 / args.gnss_rate_hz) as i128;
    let near_multipath: Vec<bool> = vo
        .iter()
        .map(|pose| {
            let t = pose.timestamp.as_nanoseconds();
            fixes.iter().any(|f| {
                f.multipath && (f.measurement.timestamp.as_nanoseconds() - t).abs() < period_ns
            })
        })
        .collect();
    let vo_se3_dropout = {
        // VO with its global best fit, restricted to the dropout frames.
        let aligned = best_fit_se3(&vo, &truth);
        aligned.and_then(|aligned| rmse_over(&aligned, &truth, &dropout_mask))
    };

    println!(
        "drive={drive} frames={} path={path_length:.1} m gnss={} fixes={} (multipath={multipath_fixes}) \
         alignment={} lever_arm=[{:.3}, {:.3}, {:.3}]",
        vo.len(),
        if args.gnss_native { "native" } else { "degraded" },
        fixes.len(),
        if args.gravity_yaw { "gravity-yaw" } else { "full" },
        lever_arm.x,
        lever_arm.y,
        lever_arm.z,
    );
    println!(
        "ATE vo_se3_aligned={} m vo_bootstrap_aligned={} m; VO rotation RPE (10 frames)={} deg",
        json_number(vo_se3_ate),
        json_number(vo_bootstrap_ate),
        json_number(vo_rotation_rpe_deg)
    );
    println!(
        "GNSS raw fixes: rmse={} m horizontal={} m without_multipath={} m",
        json_number(gnss_rmse),
        json_number(gnss_horizontal_rmse),
        json_number(gnss_clean_rmse)
    );

    let mut json = String::new();
    writeln!(json, "{{")?;
    writeln!(json, "  \"drive\": \"{drive}\",")?;
    writeln!(json, "  \"frames\": {},", vo.len())?;
    writeln!(json, "  \"path_length_m\": {path_length:.2},")?;
    writeln!(
        json,
        "  \"gnss\": {{\"mode\": \"{}\", \"rate_hz\": {}, \"sigma_h\": {}, \"sigma_v_ratio\": {}, \
         \"correlated_fraction\": {}, \"gm_tau_s\": {}, \"outlier_rate\": {}, \"outlier_m\": [{}, {}], \
         \"dropout_start_fraction\": {}, \"dropout_duration_s\": {}, \"seed\": {}, \"fixes\": {}, \
         \"multipath_fixes\": {multipath_fixes}, \"rmse_m\": {}, \"horizontal_rmse_m\": {}, \
         \"rmse_without_multipath_m\": {}}},",
        if args.gnss_native { "native" } else { "degraded" },
        args.gnss_rate_hz,
        args.sigma_h,
        args.sigma_v_ratio,
        args.correlated_fraction,
        args.gm_tau_s,
        args.outlier_rate,
        args.outlier_min_m,
        args.outlier_max_m,
        args.dropout_start_fraction,
        args.dropout_duration_s,
        args.seed,
        fixes.len(),
        json_number(gnss_rmse),
        json_number(gnss_horizontal_rmse),
        json_number(gnss_clean_rmse),
    )?;
    writeln!(
        json,
        "  \"alignment\": \"{}\",",
        if args.gravity_yaw {
            "gravity-yaw"
        } else {
            "full"
        }
    )?;
    writeln!(
        json,
        "  \"gate_at_fix_accuracy\": {},",
        args.gate_at_fix_accuracy
    )?;
    writeln!(
        json,
        "  \"lever_arm_cam\": [{:.4}, {:.4}, {:.4}],",
        lever_arm.x, lever_arm.y, lever_arm.z
    )?;
    writeln!(
        json,
        "  \"vo\": {{\"se3_aligned_ate_m\": {}, \"bootstrap_aligned_ate_m\": {}, \"se3_aligned_dropout_ate_m\": {}, \
         \"rotation_rpe_10_frames_deg\": {}}},",
        json_number(vo_se3_ate),
        json_number(vo_bootstrap_ate),
        json_number(vo_se3_dropout),
        json_number(vo_rotation_rpe_deg)
    )?;
    writeln!(json, "  \"fused\": {{")?;
    for (index, run) in runs.iter().enumerate() {
        let fused = &run.result;
        let ate = position_rmse(&fused.trajectory_enu, &truth);
        let se3 = aligned_position_rmse(&fused.trajectory_enu, &truth, false);
        let dropout = rmse_over(&fused.trajectory_enu, &truth, &dropout_mask);
        let max = max_error(&fused.trajectory_enu, &truth);
        let horizontal = horizontal_rmse(&fused.trajectory_enu, &truth);
        let flagged: Vec<bool> = fused
            .frames
            .iter()
            .map(|frame| matches!(frame.gnss, FrameGnssStatus::Outlier { .. }))
            .collect();
        let flagged_near = flagged
            .iter()
            .zip(&near_multipath)
            .filter(|(f, n)| **f && **n)
            .count();
        let flagged_elsewhere = flagged
            .iter()
            .zip(&near_multipath)
            .filter(|(f, n)| **f && !**n)
            .count();
        let near_count = near_multipath.iter().filter(|n| **n).count();
        println!(
            "fused[{}]: ate={} m (horizontal {} m, se3-aligned {} m, max {max:.2} m, dropout frames {} m) \
             inlier_frames={} outlier_frames={} (near multipath {flagged_near}/{near_count}, elsewhere \
             {flagged_elsewhere}) no_fix_frames={} solves={} time={:.1}s",
            run.label,
            json_number(ate),
            json_number(horizontal),
            json_number(se3),
            json_number(dropout),
            fused.summary.frames_with_inlier_fix,
            fused.summary.frames_with_outlier_fix,
            fused.summary.frames_without_fix,
            fused.summary.optimizer_runs,
            run.seconds,
        );
        writeln!(
            json,
            "    \"{}\": {{\"ate_m\": {}, \"horizontal_ate_m\": {}, \"se3_aligned_ate_m\": {}, \
             \"max_error_m\": {max:.4}, \"dropout_ate_m\": {}, \"inlier_frames\": {}, \"outlier_frames\": {}, \
             \"outlier_frames_near_multipath\": {flagged_near}, \"frames_near_multipath\": {near_count}, \
             \"outlier_frames_elsewhere\": {flagged_elsewhere}, \"no_fix_frames\": {}, \
             \"optimizer_runs\": {}, \"seconds\": {:.2}}}{}",
            run.label,
            json_number(ate),
            json_number(horizontal),
            json_number(se3),
            json_number(dropout),
            fused.summary.frames_with_inlier_fix,
            fused.summary.frames_with_outlier_fix,
            fused.summary.frames_without_fix,
            fused.summary.optimizer_runs,
            run.seconds,
            if index + 1 < runs.len() { "," } else { "" }
        )?;
    }
    writeln!(json, "  }}")?;
    writeln!(json, "}}")?;

    if let Some(dir) = &args.out_dir {
        fs::create_dir_all(dir)?;
        fs::write(dir.join("summary.json"), &json)?;
        let vo_aligned = best_fit_se3(&vo, &truth).unwrap_or_else(|| vo.clone());
        let mut csv = String::from(
            "timestamp_ns,truth_x,truth_y,truth_z,vo_se3_x,vo_se3_y,vo_se3_z,batch_x,batch_y,batch_z,batch_gnss",
        );
        if runs.len() > 1 {
            csv.push_str(",window_x,window_y,window_z,window_gnss");
        }
        csv.push('\n');
        for (i, gt) in truth.iter().enumerate() {
            let g = gt.value.camera_center_world();
            let v = vo_aligned[i].value.camera_center_world();
            write!(
                csv,
                "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                gt.timestamp.as_nanoseconds(),
                g.x,
                g.y,
                g.z,
                v.x,
                v.y,
                v.z
            )?;
            for run in &runs {
                let f = run.result.trajectory_enu[i].value.camera_center_world();
                let status = match run.result.frames[i].gnss {
                    FrameGnssStatus::Inlier { .. } => "inlier",
                    FrameGnssStatus::Outlier { .. } => "outlier",
                    FrameGnssStatus::NoFix | FrameGnssStatus::NotOptimized => "no_fix",
                };
                write!(csv, ",{:.4},{:.4},{:.4},{status}", f.x, f.y, f.z)?;
            }
            csv.push('\n');
        }
        fs::write(dir.join("trajectory.csv"), csv)?;
        let mut gnss_csv = String::from("timestamp_ns,x,y,z,truth_x,truth_y,truth_z,multipath\n");
        for fix in &fixes {
            let p = fix.measurement.position_world;
            writeln!(
                gnss_csv,
                "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{}",
                fix.measurement.timestamp.as_nanoseconds(),
                p.x,
                p.y,
                p.z,
                fix.truth.x,
                fix.truth.y,
                fix.truth.z,
                u8::from(fix.multipath)
            )?;
        }
        fs::write(dir.join("gnss.csv"), gnss_csv)?;
        println!("wrote {}", dir.display());
    }
    Ok(())
}

/// RMS angle (degrees) between estimated and true relative rotations over
/// `delta`-frame steps.
fn relative_rotation_rmse_deg(
    estimate: &[TimedPose],
    truth: &[TimedPose],
    delta: usize,
) -> Option<f64> {
    let n = estimate.len().min(truth.len());
    if n <= delta {
        return None;
    }
    let sum: f64 = (0..n - delta)
        .map(|i| {
            let relative = |poses: &[TimedPose]| {
                poses[i].value.camera_to_world().rotation.inverse()
                    * poses[i + delta].value.camera_to_world().rotation
            };
            relative(estimate)
                .angle_to(&relative(truth))
                .to_degrees()
                .powi(2)
        })
        .sum();
    Some((sum / (n - delta) as f64).sqrt())
}

/// The trajectory moved by its best-fit rigid transform onto `truth`.
fn best_fit_se3(estimate: &[TimedPose], truth: &[TimedPose]) -> Option<Vec<TimedPose>> {
    use visloc_rs::slam::{fit_gnss_alignment, AlignmentCorrespondence, GnssScaleMode};
    let correspondences: Vec<AlignmentCorrespondence> = estimate
        .iter()
        .zip(truth)
        .map(|(a, b)| AlignmentCorrespondence {
            camera_to_map: a.value.camera_to_world(),
            position_enu: b.value.camera_center_world().coords,
            horizontal_sigma: 1.0,
            vertical_sigma: 1.0,
        })
        .collect();
    let alignment = fit_gnss_alignment(
        &correspondences,
        AlignmentRotationDof::Full,
        GnssScaleMode::Metric,
        &Vector3::zeros(),
    )?;
    Some(transform_trajectory(&alignment, estimate))
}
