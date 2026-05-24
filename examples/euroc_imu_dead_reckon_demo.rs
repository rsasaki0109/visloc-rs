//! IMU dead-reckoning baseline on the EuRoC MAV dataset.
//!
//! Reads an EuRoC recording (`mav0/imu0/data.csv`,
//! `mav0/state_groundtruth_estimate0/data.csv`, `mav0/cam0/data.csv`), seeds
//! the propagator with the first ground-truth state that lies inside the IMU
//! window (position, velocity, orientation, IMU biases), and then propagates
//! `(R, v, p)` forward using nothing but the body-frame gyro + accel stream.
//! At every cam0 frame timestamp the integrated state is compared against the
//! nearest ground-truth sample to expose orientation drift (degrees) and
//! position drift (metres) over the trajectory.
//!
//! The example is intentionally a "lower bound" reference: no visual aiding,
//! no bias re-estimation, no zero-velocity updates. Any visual-inertial
//! pipeline added later should drive these errors down — having the
//! IMU-only number on disk makes it easy to quantify the improvement.
//!
//! Usage:
//!
//! ```sh
//! cargo run --release --example euroc_imu_dead_reckon_demo -- \
//!     --euroc-dir /path/to/MH_01_easy \
//!     --out-dir target/euroc_imu_baseline \
//!     --max-frames 800
//! ```
//!
//! Writes `imu_dead_reckon.csv` (timestamp, position xyz, quaternion wxyz),
//! `imu_dead_reckon_errors.csv` (per-cam0-frame position / orientation error
//! against ground truth) and `summary.txt` to the output directory.

use std::env;
use std::fs;
use std::path::PathBuf;

use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_rs::io::euroc::{read_euroc_dataset_dir, EurocGroundTruthSample, EurocImuSample};
use visloc_rs::{
    umeyama_similarity_transform, TrajectorySimilarityTransform, VisualInertialInitializer,
    VisualInertialInitializerConfig,
};

/// Where to source the propagator's initial rotation. `Vi` and the two
/// hybrid variants require `--run-vi-init` (or are coerced into setting
/// it on parse). The hybrids decompose each rotation into a yaw component
/// about the world-up axis and the orthogonal roll/pitch tilt, then
/// recombine.
#[derive(Debug, Clone, Copy, PartialEq)]
enum SeedRotationSource {
    /// Use the GT orientation at the seed timestamp (the "cheating" baseline).
    Gt,
    /// Use the VI-init recovered rotation (gravity-aligned; yaw is 0 gauge).
    Vi,
    /// Mix: roll/pitch from VI-init, yaw from GT. Isolates "does the
    /// gravity-only init's missing yaw cause the drift?" — if this matches
    /// `Gt`, yaw was the limiter; if it matches `Vi`, roll/pitch leakage
    /// or downstream gyro drift dominates.
    ViRollPitchGtYaw,
    /// Mix: roll/pitch from GT, yaw from VI-init (i.e. yaw = 0). Isolates
    /// "does the gravity-alignment residual (roll/pitch leakage) cause the
    /// drift?" — if this matches `Gt`, roll/pitch was fine; if it matches
    /// `Vi`, the gravity residual was the limiter.
    GtRollPitchViYaw,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum SeedBiasSource {
    Gt,
    Vi,
    Zero,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum SeedVelocitySource {
    Gt,
    /// VI-init returns `0` for stationary bootstraps; this is equivalent to
    /// `Zero` for the static flavour but tracks intent more clearly.
    Vi,
    Zero,
}

#[derive(Debug)]
struct CliArgs {
    euroc_dir: PathBuf,
    out_dir: PathBuf,
    /// Cap the number of cam0 frames processed (useful for quick smoke
    /// runs). `0` keeps the full sequence.
    max_frames: usize,
    /// Gravity in the EuRoC world frame; the dataset uses `(0, 0, −9.81)`
    /// (z-up world frame, gravity points down).
    gravity_world: Vector3<f64>,
    /// When `true`, run [`VisualInertialInitializer`] over a leading
    /// window of IMU samples and log the residual against the GT
    /// state. When `--seed-from-vi-init` is also `true`, the demo
    /// uses the recovered `(orientation, bias_gyro, bias_acc,
    /// velocity = 0)` instead of GT for the propagator — that is the
    /// honest "no ground-truth cheating" baseline.
    run_vi_init: bool,
    /// Per-component seed source for the ablation harness. Independent
    /// of `--seed-from-vi-init`, so callers can mix sources (e.g. GT
    /// rotation + VI biases, or VI roll/pitch + GT yaw + GT biases) to
    /// attribute drift to specific components.
    seed_rotation_source: SeedRotationSource,
    seed_velocity_source: SeedVelocitySource,
    seed_bias_gyro_source: SeedBiasSource,
    seed_bias_acc_source: SeedBiasSource,
    /// Shift IMU sample timestamps by this offset (positive = IMU
    /// timestamps move forward relative to cam0) before they are
    /// compared to GT / cam0 timestamps. Used to sweep IMU/cam time
    /// alignment as a potential source of ATE drift.
    imu_cam_time_offset_ns: i64,
    /// Duration (seconds) of the leading IMU window pushed into the
    /// VI initialiser when `--run-vi-init` is on. EuRoC MH-sequences
    /// have a ~1 s stationary period at the start of the recording.
    vi_init_window_seconds: f64,
    /// Override the per-axis gyro standard deviation threshold for
    /// the VI initialiser (rad / s). EuRoC hand-held release periods
    /// often miss the conservative default 0.05; values around
    /// 0.1–0.2 admit the leading "lift-off" window on those runs.
    vi_init_gyro_std_limit: Option<f64>,
    /// Override the per-axis accel standard deviation threshold for
    /// the VI initialiser (m / s²).
    vi_init_accel_std_limit: Option<f64>,
}

fn parse_seed_rotation_source(s: &str) -> Result<SeedRotationSource, String> {
    match s {
        "gt" => Ok(SeedRotationSource::Gt),
        "vi" => Ok(SeedRotationSource::Vi),
        "vi_rollpitch_gt_yaw" => Ok(SeedRotationSource::ViRollPitchGtYaw),
        "gt_rollpitch_vi_yaw" => Ok(SeedRotationSource::GtRollPitchViYaw),
        other => Err(format!(
            "--seed-rotation-source expects gt|vi|vi_rollpitch_gt_yaw|gt_rollpitch_vi_yaw, got '{other}'"
        )),
    }
}

fn parse_seed_bias_source(s: &str, flag: &str) -> Result<SeedBiasSource, String> {
    match s {
        "gt" => Ok(SeedBiasSource::Gt),
        "vi" => Ok(SeedBiasSource::Vi),
        "zero" => Ok(SeedBiasSource::Zero),
        other => Err(format!("{flag} expects gt|vi|zero, got '{other}'")),
    }
}

fn parse_seed_velocity_source(s: &str) -> Result<SeedVelocitySource, String> {
    match s {
        "gt" => Ok(SeedVelocitySource::Gt),
        "vi" => Ok(SeedVelocitySource::Vi),
        "zero" => Ok(SeedVelocitySource::Zero),
        other => Err(format!(
            "--seed-velocity-source expects gt|vi|zero, got '{other}'"
        )),
    }
}

/// Decompose a rotation `R` into `(R_yaw, R_tilt)` such that
/// `R = R_tilt · R_yaw`, where `R_yaw` is a rotation about `world_up`
/// (i.e. `R_yaw · world_up = world_up`) and `R_tilt` is the orthogonal
/// roll/pitch tilt that lifts `world_up` into `R · world_up`. Used by
/// the ablation harness to mix yaw and roll/pitch from different sources
/// (GT vs VI). The matrix order `R = R_tilt · R_yaw` is the only one
/// that makes `R_yaw` a true rotation about `world_up`; swapping the
/// order silently breaks the gauge invariance, which is why this is
/// implemented as a dedicated helper with explicit tests.
fn split_yaw_tilt(
    rotation: &UnitQuaternion<f64>,
    world_up: Vector3<f64>,
) -> (UnitQuaternion<f64>, UnitQuaternion<f64>) {
    let body_up_in_world = rotation * world_up;
    let tilt = UnitQuaternion::rotation_between(&world_up, &body_up_in_world)
        .unwrap_or_else(UnitQuaternion::identity);
    // `yaw = tilt^-1 · rotation`. By construction `yaw · world_up = world_up`,
    // so `yaw` is a rotation purely about `world_up`.
    let yaw = tilt.inverse() * rotation;
    (yaw, tilt)
}

/// Recompose a rotation from a yaw source and a roll/pitch source. Both
/// inputs are full rotations; only their yaw / tilt components are kept.
/// Mirrors the decomposition order in [`split_yaw_tilt`]: the result is
/// `R_tilt · R_yaw`.
fn compose_yaw_tilt(
    yaw_source: &UnitQuaternion<f64>,
    tilt_source: &UnitQuaternion<f64>,
    world_up: Vector3<f64>,
) -> UnitQuaternion<f64> {
    let (yaw, _) = split_yaw_tilt(yaw_source, world_up);
    let (_, tilt) = split_yaw_tilt(tilt_source, world_up);
    tilt * yaw
}

fn parse_args() -> Result<CliArgs, Box<dyn std::error::Error>> {
    let mut euroc_dir: Option<PathBuf> = None;
    let mut out_dir = PathBuf::from("target/euroc_imu_dead_reckon");
    let mut max_frames: usize = 0;
    let mut gravity_world = Vector3::new(0.0, 0.0, -9.81);
    let mut run_vi_init = false;
    let mut seed_from_vi_init = false;
    let mut seed_rotation_source = SeedRotationSource::Gt;
    let mut seed_velocity_source = SeedVelocitySource::Gt;
    let mut seed_bias_gyro_source = SeedBiasSource::Gt;
    let mut seed_bias_acc_source = SeedBiasSource::Gt;
    let mut imu_cam_time_offset_ns: i64 = 0;
    let mut vi_init_window_seconds: f64 = 1.0;
    let mut vi_init_gyro_std_limit: Option<f64> = None;
    let mut vi_init_accel_std_limit: Option<f64> = None;

    let mut args: Vec<String> = env::args().skip(1).collect();
    let i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--euroc-dir" => {
                euroc_dir = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--out-dir" => {
                out_dir = PathBuf::from(args.remove(i + 1));
                args.remove(i);
            }
            "--max-frames" => {
                max_frames = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--gravity" => {
                let xyz: Vec<f64> = args
                    .remove(i + 1)
                    .split(',')
                    .map(|tok| tok.trim().parse::<f64>())
                    .collect::<Result<_, _>>()?;
                if xyz.len() != 3 {
                    return Err("--gravity expects 'gx,gy,gz'".into());
                }
                gravity_world = Vector3::new(xyz[0], xyz[1], xyz[2]);
                args.remove(i);
            }
            "--run-vi-init" => {
                run_vi_init = true;
                args.remove(i);
            }
            "--seed-from-vi-init" => {
                run_vi_init = true;
                seed_from_vi_init = true;
                args.remove(i);
            }
            "--seed-rotation-source" => {
                let value = args.remove(i + 1);
                seed_rotation_source = parse_seed_rotation_source(&value)?;
                args.remove(i);
            }
            "--seed-velocity-source" => {
                let value = args.remove(i + 1);
                seed_velocity_source = parse_seed_velocity_source(&value)?;
                args.remove(i);
            }
            "--seed-bias-gyro-source" => {
                let value = args.remove(i + 1);
                seed_bias_gyro_source = parse_seed_bias_source(&value, "--seed-bias-gyro-source")?;
                args.remove(i);
            }
            "--seed-bias-acc-source" => {
                let value = args.remove(i + 1);
                seed_bias_acc_source = parse_seed_bias_source(&value, "--seed-bias-acc-source")?;
                args.remove(i);
            }
            "--imu-cam-time-offset-ms" => {
                let value: f64 = args.remove(i + 1).parse()?;
                imu_cam_time_offset_ns = (value * 1.0e6) as i64;
                args.remove(i);
            }
            "--vi-init-window-seconds" => {
                vi_init_window_seconds = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--vi-init-gyro-std-limit" => {
                vi_init_gyro_std_limit = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--vi-init-accel-std-limit" => {
                vi_init_accel_std_limit = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    // `--seed-from-vi-init` is a shorthand for "all components from VI".
    // Per-component flags can override individual axes after that.
    if seed_from_vi_init {
        if seed_rotation_source == SeedRotationSource::Gt {
            seed_rotation_source = SeedRotationSource::Vi;
        }
        if seed_velocity_source == SeedVelocitySource::Gt {
            seed_velocity_source = SeedVelocitySource::Vi;
        }
        if seed_bias_gyro_source == SeedBiasSource::Gt {
            seed_bias_gyro_source = SeedBiasSource::Vi;
        }
        if seed_bias_acc_source == SeedBiasSource::Gt {
            seed_bias_acc_source = SeedBiasSource::Vi;
        }
    }
    // Any non-`Gt` source needs the VI initialiser to be run.
    let any_vi_source = seed_rotation_source != SeedRotationSource::Gt
        || seed_velocity_source == SeedVelocitySource::Vi
        || seed_bias_gyro_source == SeedBiasSource::Vi
        || seed_bias_acc_source == SeedBiasSource::Vi;
    if any_vi_source {
        run_vi_init = true;
    }
    let euroc_dir = euroc_dir.ok_or("--euroc-dir <path/to/MH_01_easy> is required")?;
    Ok(CliArgs {
        euroc_dir,
        out_dir,
        max_frames,
        gravity_world,
        run_vi_init,
        seed_rotation_source,
        seed_velocity_source,
        seed_bias_gyro_source,
        seed_bias_acc_source,
        imu_cam_time_offset_ns,
        vi_init_window_seconds,
        vi_init_gyro_std_limit,
        vi_init_accel_std_limit,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let dataset = read_euroc_dataset_dir(&args.euroc_dir)?;
    println!(
        "loaded euroc cam0_frames={} cam1_frames={} imu_samples={} gt_samples={} cam0_rate={:.1}Hz imu_rate={:.1}Hz",
        dataset.cam0_images.len(),
        dataset.cam1_images.len(),
        dataset.imu_samples.len(),
        dataset.ground_truth.len(),
        dataset.cam0_calibration.rate_hz,
        dataset.imu_calibration.rate_hz,
    );
    if dataset.ground_truth.is_empty() {
        return Err(format!(
            "ground truth missing under {}/mav0/state_groundtruth_estimate0/data.csv",
            args.euroc_dir.display()
        )
        .into());
    }
    if dataset.imu_samples.len() < 10 {
        return Err(format!(
            "too few IMU samples ({}) — is this an EuRoC recording?",
            dataset.imu_samples.len()
        )
        .into());
    }

    // Seed the propagator from the first GT sample that lies inside the IMU
    // stream window. EuRoC GT generally starts a few ms before the IMU on the
    // MH sequences, so this skips the warm-up gap and lets us read both
    // initial velocity and biases off the ground-truth row.
    let imu_first_ts = dataset.imu_samples.first().unwrap().timestamp_nanoseconds;
    let imu_last_ts = dataset.imu_samples.last().unwrap().timestamp_nanoseconds;
    let seed = dataset
        .ground_truth
        .iter()
        .find(|gt| {
            gt.timestamp_nanoseconds >= imu_first_ts && gt.timestamp_nanoseconds <= imu_last_ts
        })
        .ok_or("ground truth and IMU streams do not overlap")?;
    println!(
        "seed_gt_t_ns={} pos=[{:.3}, {:.3}, {:.3}] q_wxyz=[{:.3}, {:.3}, {:.3}, {:.3}] velocity={} bias_gyro={} bias_acc={}",
        seed.timestamp_nanoseconds,
        seed.position_world.x,
        seed.position_world.y,
        seed.position_world.z,
        seed.orientation_world.w,
        seed.orientation_world.i,
        seed.orientation_world.j,
        seed.orientation_world.k,
        seed.velocity_world.is_some(),
        seed.bias_gyro.is_some(),
        seed.bias_acc.is_some(),
    );

    let mut rotation = seed.orientation_world;
    let mut velocity = seed.velocity_world.unwrap_or_else(Vector3::zeros);
    let mut position = seed.position_world;
    let mut bias_gyro = seed.bias_gyro.unwrap_or_else(Vector3::zeros);
    let mut bias_acc = seed.bias_acc.unwrap_or_else(Vector3::zeros);

    // Optional: run the visual-inertial initializer over the leading
    // IMU window and log the residual vs ground truth. When
    // `--seed-from-vi-init` is set, also use the recovered values as
    // the propagator's bootstrap instead of the GT-cheating seed
    // above — this is the honest "no ground truth state" baseline.
    if args.run_vi_init {
        let mut vi_init_config = VisualInertialInitializerConfig {
            gravity_world: args.gravity_world,
            ..Default::default()
        };
        if let Some(limit) = args.vi_init_gyro_std_limit {
            vi_init_config.max_gyro_std = limit;
        }
        if let Some(limit) = args.vi_init_accel_std_limit {
            vi_init_config.max_accel_std = limit;
        }
        let mut initializer = VisualInertialInitializer::new(vi_init_config);
        let window_ns = (args.vi_init_window_seconds * 1.0e9) as i128;
        let window_end_ts = imu_first_ts as i128 + window_ns;
        let mut prev_ts = imu_first_ts;
        for sample in &dataset.imu_samples {
            if sample.timestamp_nanoseconds > window_end_ts {
                break;
            }
            let dt_ns = sample.timestamp_nanoseconds - prev_ts as i128;
            prev_ts = sample.timestamp_nanoseconds;
            if dt_ns <= 0 {
                continue;
            }
            let dt = dt_ns as f64 * 1.0e-9;
            initializer.push_sample(sample.gyro, sample.accel, dt);
        }
        match initializer.try_initialize() {
            Ok(result) => {
                let bias_gyro_residual =
                    (result.bias_gyro - seed.bias_gyro.unwrap_or_else(Vector3::zeros)).norm();
                let bias_acc_residual =
                    (result.bias_acc - seed.bias_acc.unwrap_or_else(Vector3::zeros)).norm();
                let rotation_residual_deg = seed
                    .orientation_world
                    .rotation_to(&result.initial_rotation_body_to_world)
                    .angle()
                    .to_degrees();
                // VI init cannot observe yaw from gravity alone, so
                // the full quaternion residual above includes a yaw
                // mismatch the initialiser has no way to fix. The
                // fair metric is the angle between the body-frame
                // "up" directions implied by the two orientations —
                // i.e. how close the recovered roll / pitch match
                // ground truth. Implemented on the result type so the
                // metric ships with the same yaw-gauge contract the
                // unit tests pin.
                let gravity_alignment_residual_deg =
                    result.gravity_alignment_residual_deg(&seed.orientation_world);
                println!(
                    "vi_init=success samples={} duration_s={:.3} gyro_std=[{:.4},{:.4},{:.4}] accel_std=[{:.3},{:.3},{:.3}] mean_accel_mag={:.4}",
                    result.samples_consumed,
                    result.duration_seconds,
                    result.gyro_std.x,
                    result.gyro_std.y,
                    result.gyro_std.z,
                    result.accel_std.x,
                    result.accel_std.y,
                    result.accel_std.z,
                    result.mean_accel_magnitude,
                );
                println!(
                    "vi_init bias_gyro=[{:.6},{:.6},{:.6}] bias_acc=[{:.4},{:.4},{:.4}] rotation_residual_vs_gt_deg={:.4} gravity_alignment_residual_deg={:.4} bias_gyro_residual_vs_gt={:.6} bias_acc_residual_vs_gt={:.4}",
                    result.bias_gyro.x,
                    result.bias_gyro.y,
                    result.bias_gyro.z,
                    result.bias_acc.x,
                    result.bias_acc.y,
                    result.bias_acc.z,
                    rotation_residual_deg,
                    gravity_alignment_residual_deg,
                    bias_gyro_residual,
                    bias_acc_residual,
                );
                // Per-component seed override for the ablation harness.
                // Each axis can independently come from GT or VI (and
                // rotation has two extra hybrid sources that split into
                // yaw and roll/pitch). Position is always anchored at
                // GT because VI init never observes absolute position.
                let world_up = -args.gravity_world.normalize();
                let r_gt = seed.orientation_world;
                let r_vi = result.initial_rotation_body_to_world;
                rotation = match args.seed_rotation_source {
                    SeedRotationSource::Gt => r_gt,
                    SeedRotationSource::Vi => r_vi,
                    SeedRotationSource::ViRollPitchGtYaw => {
                        compose_yaw_tilt(&r_gt, &r_vi, world_up)
                    }
                    SeedRotationSource::GtRollPitchViYaw => {
                        compose_yaw_tilt(&r_vi, &r_gt, world_up)
                    }
                };
                velocity = match args.seed_velocity_source {
                    SeedVelocitySource::Gt => seed.velocity_world.unwrap_or_else(Vector3::zeros),
                    SeedVelocitySource::Vi => result.initial_velocity_world,
                    SeedVelocitySource::Zero => Vector3::zeros(),
                };
                bias_gyro = match args.seed_bias_gyro_source {
                    SeedBiasSource::Gt => seed.bias_gyro.unwrap_or_else(Vector3::zeros),
                    SeedBiasSource::Vi => result.bias_gyro,
                    SeedBiasSource::Zero => Vector3::zeros(),
                };
                bias_acc = match args.seed_bias_acc_source {
                    SeedBiasSource::Gt => seed.bias_acc.unwrap_or_else(Vector3::zeros),
                    SeedBiasSource::Vi => result.bias_acc,
                    SeedBiasSource::Zero => Vector3::zeros(),
                };
                position = seed.position_world;
                println!(
                    "vi_init=seeded propagator rotation={:?} velocity={:?} bias_gyro={:?} bias_acc={:?}; position anchored at GT",
                    args.seed_rotation_source,
                    args.seed_velocity_source,
                    args.seed_bias_gyro_source,
                    args.seed_bias_acc_source,
                );
            }
            Err(reason) => {
                println!("vi_init=failed reason={reason:?}");
            }
        }
    }

    // The IMU is rigidly mounted to the body, so its readings live in the
    // sensor frame. EuRoC's imu0/sensor.yaml `T_BS` is identity for every
    // public release (the IMU IS the body origin), so we treat ω, a directly
    // as body-frame quantities. Capture that fact in the log so anyone re-
    // running on a different EuRoC release notices when it stops holding.
    let imu_t_bs = &dataset.imu_calibration.t_body_sensor;
    let imu_t_bs_is_identity = (0..4).all(|r| {
        (0..4).all(|c| {
            let expected = if r == c { 1.0 } else { 0.0 };
            (imu_t_bs[(r, c)] - expected).abs() < 1.0e-6
        })
    });
    println!("imu_t_bs_is_identity={imu_t_bs_is_identity}");

    fs::create_dir_all(&args.out_dir)?;
    let traj_path = args.out_dir.join("imu_dead_reckon.csv");
    let err_path = args.out_dir.join("imu_dead_reckon_errors.csv");

    let mut traj_csv = String::from("timestamp_ns,px,py,pz,qw,qx,qy,qz,vx,vy,vz\n");
    let mut err_csv = String::from(
        "timestamp_ns,frame_idx,gt_px,gt_py,gt_pz,est_px,est_py,est_pz,position_error_m,orientation_error_deg\n",
    );

    let mut sample_index = 0usize;
    let mut prev_ts = seed.timestamp_nanoseconds;
    // Snap the IMU iterator to the seed timestamp so the first integrated
    // step matches the seed state exactly. Any pre-seed samples are simply
    // not consumed.
    while sample_index < dataset.imu_samples.len()
        && dataset.imu_samples[sample_index].timestamp_nanoseconds < seed.timestamp_nanoseconds
    {
        sample_index += 1;
    }

    // Cam0 frame iterator — we record per-frame errors at the cam0 cadence
    // because that's the natural ATE cadence for a VIO benchmark.
    let mut frame_iter = dataset
        .cam0_images
        .iter()
        .enumerate()
        .filter(|(_, frame)| frame.timestamp_nanoseconds >= seed.timestamp_nanoseconds);
    let mut next_frame = frame_iter.next();
    let frame_cap = if args.max_frames == 0 {
        usize::MAX
    } else {
        args.max_frames
    };
    let mut frames_recorded = 0usize;

    let mut sum_position_sq = 0.0_f64;
    let mut sum_orientation_sq_deg = 0.0_f64;
    let mut max_position_err = 0.0_f64;
    let mut max_orientation_err_deg = 0.0_f64;
    let mut error_samples = 0usize;
    // Buffer (estimated, ground-truth) camera-centre pairs so we can run a
    // proper Umeyama-aligned ATE evaluation after the dead-reckon sweep
    // finishes. ATE is the standard metric reported by ORB-SLAM3 /
    // VINS-Mono / OKVIS / Kimera-VIO; without similarity alignment the
    // raw drift numbers are not comparable to the published values.
    let mut aligned_estimated_positions: Vec<Point3<f64>> = Vec::new();
    let mut aligned_reference_positions: Vec<Point3<f64>> = Vec::new();

    while sample_index < dataset.imu_samples.len() {
        let sample = &dataset.imu_samples[sample_index];
        let dt_ns = sample.timestamp_nanoseconds - prev_ts;
        // Skip out-of-order or zero-step samples; the IMU CSV occasionally
        // has repeated timestamps near the boundaries.
        if dt_ns <= 0 {
            prev_ts = sample.timestamp_nanoseconds;
            sample_index += 1;
            continue;
        }
        let dt = dt_ns as f64 * 1.0e-9;
        propagate(
            &mut rotation,
            &mut velocity,
            &mut position,
            &args.gravity_world,
            &bias_gyro,
            &bias_acc,
            sample,
            dt,
        );
        prev_ts = sample.timestamp_nanoseconds;
        sample_index += 1;

        // Emit a trajectory CSV row at every IMU step — keeps the data dense
        // for plots and ATE evaluation tooling.
        traj_csv.push_str(&format!(
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            sample.timestamp_nanoseconds,
            position.x,
            position.y,
            position.z,
            rotation.w,
            rotation.i,
            rotation.j,
            rotation.k,
            velocity.x,
            velocity.y,
            velocity.z,
        ));

        // Drain every cam0 frame that the current IMU step has now reached.
        // When `--imu-cam-time-offset-ms` is non-zero, the IMU's view of
        // wall-clock time is shifted by the offset before being compared
        // to cam0 / GT timestamps. Positive offset = IMU clock is ahead
        // of cam0 (the IMU "sees" the world before the camera does), so
        // a given IMU state corresponds to an earlier cam0 frame.
        let imu_ts_in_cam_frame = sample.timestamp_nanoseconds as i64 + args.imu_cam_time_offset_ns;
        while let Some((frame_idx, frame)) = next_frame {
            if (frame.timestamp_nanoseconds as i64) > imu_ts_in_cam_frame {
                break;
            }
            if frames_recorded >= frame_cap {
                next_frame = None;
                break;
            }
            let gt = nearest_ground_truth(&dataset.ground_truth, frame.timestamp_nanoseconds);
            let position_error = (gt.position_world - position).norm();
            let orientation_error_rad = rotation.rotation_to(&gt.orientation_world).angle();
            let orientation_error_deg = orientation_error_rad.to_degrees();
            err_csv.push_str(&format!(
                "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                frame.timestamp_nanoseconds,
                frame_idx,
                gt.position_world.x,
                gt.position_world.y,
                gt.position_world.z,
                position.x,
                position.y,
                position.z,
                position_error,
                orientation_error_deg,
            ));
            sum_position_sq += position_error * position_error;
            sum_orientation_sq_deg += orientation_error_deg * orientation_error_deg;
            if position_error > max_position_err {
                max_position_err = position_error;
            }
            if orientation_error_deg > max_orientation_err_deg {
                max_orientation_err_deg = orientation_error_deg;
            }
            aligned_estimated_positions.push(Point3::from(position));
            aligned_reference_positions.push(Point3::from(gt.position_world));
            error_samples += 1;
            frames_recorded += 1;
            next_frame = frame_iter.next();
        }
        if frames_recorded >= frame_cap {
            break;
        }
    }

    fs::write(&traj_path, traj_csv)?;
    fs::write(&err_path, err_csv)?;

    let (rmse_pos, rmse_rot_deg) = if error_samples > 0 {
        (
            (sum_position_sq / error_samples as f64).sqrt(),
            (sum_orientation_sq_deg / error_samples as f64).sqrt(),
        )
    } else {
        (0.0, 0.0)
    };

    // Apples-to-apples ATE numbers: rigid SE(3) Umeyama matches what
    // ORB-SLAM3 / OKVIS / VINS-Mono report on monocular-inertial runs;
    // similarity Sim(3) with scale matches the protocol used by purely
    // monocular SLAM benchmarks. We report both so the dead-reckon
    // residual is comparable to the published EuRoC numbers under
    // either convention.
    let aligned_rigid = umeyama_similarity_transform(
        &aligned_estimated_positions,
        &aligned_reference_positions,
        false,
    )
    .unwrap_or_else(TrajectorySimilarityTransform::identity);
    let aligned_similarity = umeyama_similarity_transform(
        &aligned_estimated_positions,
        &aligned_reference_positions,
        true,
    )
    .unwrap_or_else(TrajectorySimilarityTransform::identity);

    let mut rmse_sq_rigid = 0.0_f64;
    let mut max_rigid = 0.0_f64;
    let mut rmse_sq_sim = 0.0_f64;
    let mut max_sim = 0.0_f64;
    for (est, gt) in aligned_estimated_positions
        .iter()
        .zip(aligned_reference_positions.iter())
    {
        let rigid_err = (aligned_rigid.apply(est) - gt).norm();
        let sim_err = (aligned_similarity.apply(est) - gt).norm();
        rmse_sq_rigid += rigid_err * rigid_err;
        rmse_sq_sim += sim_err * sim_err;
        if rigid_err > max_rigid {
            max_rigid = rigid_err;
        }
        if sim_err > max_sim {
            max_sim = sim_err;
        }
    }
    let (ate_rmse_rigid, ate_rmse_sim) = if !aligned_estimated_positions.is_empty() {
        let n = aligned_estimated_positions.len() as f64;
        ((rmse_sq_rigid / n).sqrt(), (rmse_sq_sim / n).sqrt())
    } else {
        (0.0, 0.0)
    };
    let duration_s = if frames_recorded > 0 {
        let first = dataset
            .cam0_images
            .iter()
            .find(|f| f.timestamp_nanoseconds >= seed.timestamp_nanoseconds)
            .map(|f| f.timestamp_nanoseconds)
            .unwrap_or(seed.timestamp_nanoseconds);
        let last = prev_ts;
        (last - first) as f64 * 1.0e-9
    } else {
        0.0
    };

    let summary = format!(
        "euroc_dir={}\n\
         frames_recorded={frames_recorded}\n\
         duration_s={duration_s:.3}\n\
         imu_samples_consumed={sample_index}\n\
         ate_position_rmse_m={rmse_pos:.4}\n\
         ate_position_max_m={max_position_err:.4}\n\
         ate_orientation_rmse_deg={rmse_rot_deg:.4}\n\
         ate_orientation_max_deg={max_orientation_err_deg:.4}\n\
         ate_rigid_rmse_m={ate_rmse_rigid:.4}\n\
         ate_rigid_max_m={max_rigid:.4}\n\
         ate_similarity_rmse_m={ate_rmse_sim:.4}\n\
         ate_similarity_max_m={max_sim:.4}\n\
         ate_similarity_scale={scale:.6}\n",
        args.euroc_dir.display(),
        scale = aligned_similarity.scale,
    );
    println!("{summary}");
    fs::write(args.out_dir.join("summary.txt"), summary)?;
    println!(
        "wrote {} and {} (+ summary.txt)",
        traj_path.display(),
        err_path.display(),
    );
    Ok(())
}

/// One forward Euler step of the IMU strapdown propagator.
///
/// EuRoC stores gyro / accel readings in the IMU sensor frame; for every
/// EuRoC release the IMU frame coincides with the body frame (T_BS is
/// identity), so we treat the samples as body-frame quantities. The standard
/// strapdown equations under gravity in the world frame are:
///
/// * `R_{k+1} = R_k · Exp((ω − b_g) · Δt)`
/// * `v_{k+1} = v_k + (R_k · (a − b_a) + g_world) · Δt`
/// * `p_{k+1} = p_k + v_k · Δt + 0.5 · (R_k · (a − b_a) + g_world) · Δt²`
///
/// EuRoC's `gravity_world = (0, 0, -9.81)`: the world z-axis points up so
/// gravity acceleration adds the −z component to the velocity ODE.
// Canonical strapdown step: the mutable state (R, v, p), gravity, the two
// biases, the sample, and dt. Bundling would obscure the integration equations.
#[allow(clippy::too_many_arguments)]
fn propagate(
    rotation: &mut UnitQuaternion<f64>,
    velocity: &mut Vector3<f64>,
    position: &mut Vector3<f64>,
    gravity_world: &Vector3<f64>,
    bias_gyro: &Vector3<f64>,
    bias_acc: &Vector3<f64>,
    sample: &EurocImuSample,
    dt: f64,
) {
    let gyro_unbiased = sample.gyro - bias_gyro;
    let accel_unbiased = sample.accel - bias_acc;
    let accel_world = rotation.transform_vector(&accel_unbiased) + gravity_world;
    let new_position = *position + *velocity * dt + 0.5 * accel_world * dt * dt;
    let new_velocity = *velocity + accel_world * dt;
    let delta_rot = UnitQuaternion::from_scaled_axis(gyro_unbiased * dt);
    *rotation *= delta_rot;
    *velocity = new_velocity;
    *position = new_position;
}

fn nearest_ground_truth(
    samples: &[EurocGroundTruthSample],
    target_ts: i128,
) -> &EurocGroundTruthSample {
    let idx = samples
        .binary_search_by_key(&target_ts, |sample| sample.timestamp_nanoseconds)
        .unwrap_or_else(|insert| {
            // Pick the closer of the two neighbours; clamp at the ends.
            if insert == 0 {
                0
            } else if insert >= samples.len() {
                samples.len() - 1
            } else {
                let before = samples[insert - 1].timestamp_nanoseconds;
                let after = samples[insert].timestamp_nanoseconds;
                if (target_ts - before).abs() <= (after - target_ts).abs() {
                    insert - 1
                } else {
                    insert
                }
            }
        });
    &samples[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn z_up() -> Vector3<f64> {
        Vector3::new(0.0, 0.0, 1.0)
    }

    #[test]
    fn split_yaw_tilt_identity() {
        let (yaw, tilt) = split_yaw_tilt(&UnitQuaternion::identity(), z_up());
        assert!(yaw.angle() < 1.0e-12);
        assert!(tilt.angle() < 1.0e-12);
    }

    #[test]
    fn split_yaw_tilt_pure_yaw() {
        let yaw_angle = 0.7;
        let r = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw_angle);
        let (yaw, tilt) = split_yaw_tilt(&r, z_up());
        assert!((yaw.angle() - yaw_angle).abs() < 1.0e-9);
        assert!(tilt.angle() < 1.0e-9);
    }

    #[test]
    fn split_yaw_tilt_pure_pitch() {
        let pitch_angle = 0.3;
        let r = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), pitch_angle);
        let (yaw, tilt) = split_yaw_tilt(&r, z_up());
        assert!(yaw.angle() < 1.0e-9);
        assert!((tilt.angle() - pitch_angle).abs() < 1.0e-9);
        let recombined = yaw * tilt;
        assert!((recombined.rotation_to(&r).angle()).abs() < 1.0e-9);
    }

    #[test]
    fn compose_yaw_tilt_mixed() {
        let yaw_src = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.5);
        let tilt_src = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.3);
        let mixed = compose_yaw_tilt(&yaw_src, &tilt_src, z_up());
        // Decompose the mixed rotation and check both parts match the sources.
        let (yaw, tilt) = split_yaw_tilt(&mixed, z_up());
        assert!((yaw.angle() - 0.5).abs() < 1.0e-9);
        assert!((tilt.angle() - 0.3).abs() < 1.0e-9);
    }

    #[test]
    fn compose_yaw_tilt_roundtrip_preserves_full_rotation() {
        // Mixing yaw and tilt from the same rotation should reconstruct it
        // exactly — this is the "identity ablation" that proves the
        // decomposition is consistent.
        let r = UnitQuaternion::from_euler_angles(0.2, -0.4, 0.7); // roll, pitch, yaw
        let recombined = compose_yaw_tilt(&r, &r, z_up());
        assert!((recombined.rotation_to(&r).angle()).abs() < 1.0e-9);
    }

    #[test]
    fn vi_rollpitch_gt_yaw_keeps_yaw_only_from_gt() {
        // R_gt has yaw=0.7, R_vi has yaw=0.0 (gravity-only init). The mix
        // should have yaw=0.7 and the roll/pitch from R_vi.
        let r_gt = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.7)
            * UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.1);
        let r_vi = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.3);
        let mixed = compose_yaw_tilt(&r_gt, &r_vi, z_up());
        let (yaw, tilt) = split_yaw_tilt(&mixed, z_up());
        assert!((yaw.angle() - 0.7).abs() < 1.0e-9);
        assert!((tilt.angle() - 0.3).abs() < 1.0e-9);
    }
}
