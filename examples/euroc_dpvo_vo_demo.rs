//! DPVO visual-odometry EuRoC runner — Milestone M4 of
//! `docs/dpvo_droid_port_plan.md`.
//!
//! Wires `visloc_rs::slam::dpvo_vo::DpvoOdometry` (the ported DPVO frame
//! loop) over a real EuRoC `mav0/cam0` sequence: reads images, undistorts
//! them with the sensor.yaml radial-tangential model at full resolution
//! (`dpvo/stream.py::image_stream` does the same — see `dpvo_vo.rs`'s
//! module doc, "no resolution downscaling happens"), runs
//! [`DpvoOdometry::process_frame`] per frame, writes a trajectory CSV, and
//! computes ATE (rigid + similarity, Umeyama-aligned) against ground truth
//! when available — the same evaluation pattern
//! `examples/euroc_imu_dead_reckon_demo.rs` already uses.
//!
//! # Model artifacts
//!
//! Reads the four ONNX graphs + the `SoftAgg` weight `.npz` from
//! `--model-dir` (default `E:/visloc_archive/dpvo_onnx_m1`, the M1/M2
//! artifact directory — read-only, never written here). Regenerate them
//! with `scripts/export_dpvo_onnx.py` if missing (see that script's own
//! `--help`).
//!
//! # Config: `config/default.yaml`, not `config.py`'s bare defaults
//!
//! `E:/tools/DPVO/config/default.yaml` is the config that actually produced
//! DPVO's published EuRoC number (`PATCHES_PER_FRAME=96`,
//! `REMOVAL_WINDOW=22`, `OPTIMIZATION_WINDOW=10`, `PATCH_LIFETIME=13`,
//! `KEYFRAME_THRESH=15.0`) — see `dpvo_vo.rs`'s module doc for the full
//! citation. This demo defaults to those values (overridable via CLI) rather
//! than `crate::dpvo_patch_graph::DpvoVoConfig::default`'s bare `config.py`
//! numbers.
//!
//! Usage:
//!
//! ```sh
//! cargo run --release --features "image-io onnx-inference" \
//!     --example euroc_dpvo_vo_demo -- \
//!     --euroc-dir /path/to/MH_01_easy \
//!     --model-dir E:/visloc_archive/dpvo_onnx_m1 \
//!     --out-dir E:/visloc_archive/dpvo_m4_20260717 \
//!     --max-frames 400
//! ```
//!
//! Set `ORT_DYLIB_PATH` to the onnxruntime shared library, as with every
//! other ONNX-backed example in this repo.
//!
//! # `--imu` (Milestone M5, `docs/dpvo_droid_port_plan.md`)
//!
//! Feeds `mav0/imu0/data.csv` into `DpvoOdometry::push_imu`, interleaved by
//! timestamp with the camera frames (see the main loop's `imu_cursor`), and
//! builds `DpvoImuConfig::body_to_camera` directly from `cam0/sensor.yaml`'s
//! own `T_BS` via [`se3_from_t_bs`] (a verbatim copy of
//! `examples/euroc_online_slam_vi_demo.rs`'s own helper of the same name —
//! see `pipelines/slam/src/dpvo_vi_ba.rs`'s module doc for exactly which
//! direction this extrinsic must map). Off by default — omitting `--imu`
//! reproduces M4/M4-perf's visual-only behavior exactly. The summary echoes
//! the bootstrap chain's own diagnostics (`imu_bootstrapped`,
//! `imu_gravity_world_*`, `imu_bias_*`) alongside the usual ATE/scale
//! numbers, so a run's recovered `ate_similarity_scale` can be compared
//! directly against the M4-perf baseline (`1.266`) to see whether IMU
//! coupling actually pulled scale back toward `1.0`.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use nalgebra::{Matrix4, Point2, Point3, UnitQuaternion, Vector3};
use ndarray::Array2;

use visloc_rs::core::geometry::SE3;
use visloc_rs::io::euroc::{read_euroc_dataset_dir, EurocGroundTruthSample, EurocImuSample};
use visloc_rs::io::images::read_common_image;
use visloc_rs::slam::dpvo_patch_graph::DpvoVoConfig;
use visloc_rs::slam::dpvo_vo::{DpvoImuConfig, DpvoOdometry, DpvoOdometryConfig};
use visloc_rs::slam::{DpvoIntrinsics, ImuNoiseModel};
use visloc_rs::vision::distortion::RadialTangential;
use visloc_rs::vision::features::superpoint_onnx::OnnxBackend;
use visloc_rs::{umeyama_similarity_transform, TrajectorySimilarityTransform};

#[derive(Debug)]
struct CliArgs {
    euroc_dir: PathBuf,
    model_dir: PathBuf,
    out_dir: PathBuf,
    max_frames: usize,
    /// `evaluate_euroc.py`'s own `--stride` default (temporal subsampling,
    /// not spatial downscaling — see the module doc).
    stride: usize,
    seed: u64,
    patches_per_frame: usize,
    removal_window: usize,
    optimization_window: usize,
    patch_lifetime: usize,
    keyframe_index: usize,
    keyframe_thresh: f64,
    motion_damping: f64,
    onnx_cpu: bool,
    /// Milestone M5 (`docs/dpvo_droid_port_plan.md`): feed `mav0/imu0/data.csv`
    /// into `DpvoOdometry::push_imu` and enable the IMU-coupled joint solve
    /// once its bootstrap chain succeeds. Default off — visual-only, exactly
    /// M4/M4-perf's own behavior.
    imu: bool,
    imu_gravity_norm_deviation_ratio: f64,
    imu_min_bootstrap_factors: usize,
    /// Multiplier on `mav0/imu0/sensor.yaml`'s own noise densities (default
    /// `1.0`, i.e. the real sensor numbers, unmodified). A diagnostic/tuning
    /// knob added while investigating real-data joint-solve behavior — see
    /// the "M5 results" section of `docs/dpvo_droid_port_plan.md`.
    imu_noise_scale: f64,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            euroc_dir: PathBuf::new(),
            model_dir: PathBuf::from("E:/visloc_archive/dpvo_onnx_m1"),
            out_dir: PathBuf::from("target/euroc_dpvo_vo"),
            max_frames: 0,
            stride: 2,
            seed: 0,
            // config/default.yaml (see module doc).
            patches_per_frame: 96,
            removal_window: 22,
            optimization_window: 10,
            patch_lifetime: 13,
            keyframe_index: 4,
            keyframe_thresh: 15.0,
            motion_damping: 0.5,
            onnx_cpu: false,
            imu: false,
            imu_gravity_norm_deviation_ratio: 0.3,
            imu_min_bootstrap_factors: 10,
            imu_noise_scale: 1.0,
        }
    }
}

fn parse_args() -> Result<CliArgs, Box<dyn std::error::Error>> {
    let mut args = CliArgs::default();
    let mut euroc_dir: Option<PathBuf> = None;
    let mut raw: Vec<String> = env::args().skip(1).collect();
    let i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--euroc-dir" => euroc_dir = Some(PathBuf::from(raw.remove(i + 1))),
            "--model-dir" => args.model_dir = PathBuf::from(raw.remove(i + 1)),
            "--out-dir" => args.out_dir = PathBuf::from(raw.remove(i + 1)),
            "--max-frames" => args.max_frames = raw.remove(i + 1).parse()?,
            "--stride" => args.stride = raw.remove(i + 1).parse()?,
            "--seed" => args.seed = raw.remove(i + 1).parse()?,
            "--patches-per-frame" => args.patches_per_frame = raw.remove(i + 1).parse()?,
            "--removal-window" => args.removal_window = raw.remove(i + 1).parse()?,
            "--optimization-window" => args.optimization_window = raw.remove(i + 1).parse()?,
            "--patch-lifetime" => args.patch_lifetime = raw.remove(i + 1).parse()?,
            "--keyframe-index" => args.keyframe_index = raw.remove(i + 1).parse()?,
            "--keyframe-thresh" => args.keyframe_thresh = raw.remove(i + 1).parse()?,
            "--motion-damping" => args.motion_damping = raw.remove(i + 1).parse()?,
            "--onnx-cpu" => {
                args.onnx_cpu = true;
                raw.remove(i);
                continue;
            }
            "--imu" => {
                args.imu = true;
                raw.remove(i);
                continue;
            }
            "--imu-gravity-norm-deviation-ratio" => {
                args.imu_gravity_norm_deviation_ratio = raw.remove(i + 1).parse()?
            }
            "--imu-min-bootstrap-factors" => args.imu_min_bootstrap_factors = raw.remove(i + 1).parse()?,
            "--imu-noise-scale" => args.imu_noise_scale = raw.remove(i + 1).parse()?,
            other => return Err(format!("unknown argument: {other}").into()),
        }
        raw.remove(i);
    }
    args.euroc_dir = euroc_dir.ok_or("--euroc-dir <path/to/MH_01_easy> is required")?;
    Ok(args)
}

/// Decompose an EuRoC `T_BS` 4×4 matrix into an [`SE3`]. EuRoC's own
/// convention (confirmed by cross-referencing this repo's existing
/// `examples/euroc_online_slam_vi_demo.rs::se3_from_t_bs`, which this is a
/// verbatim copy of — see `pipelines/slam/src/dpvo_vi_ba.rs`'s module doc,
/// "jacobian convention conversion" section, for exactly which direction
/// this must map): the rotation block is a proper rotation matrix, the last
/// column is the translation, and the last row is `(0, 0, 0, 1)` — exactly
/// the SE(3) layout this codebase uses, taken literally with no inversion.
fn se3_from_t_bs(t_bs: &Matrix4<f64>) -> SE3 {
    let rotation_matrix = t_bs.fixed_view::<3, 3>(0, 0).into_owned();
    let translation = Vector3::new(t_bs[(0, 3)], t_bs[(1, 3)], t_bs[(2, 3)]);
    let rotation = UnitQuaternion::from_matrix(&rotation_matrix);
    SE3::new(rotation, translation)
}

/// Undistort a full grayscale image at the *same* intrinsics (matching
/// `cv2.undistort(image, K, dist)` — `dpvo/stream.py::image_stream`'s own
/// preprocessing): for every output (pinhole) pixel, map forward through
/// the distortion model to find the corresponding source pixel, then
/// bilinearly sample the original (distorted) image. Zero-pads samples
/// that land outside the source image.
fn undistort_image(
    source: &Array2<u8>,
    intrinsics: [f64; 4],
    distortion: &RadialTangential,
) -> Array2<u8> {
    let (h, w) = source.dim();
    if distortion.is_identity() {
        return source.clone();
    }
    let (fx, fy, cx, cy) = (intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]);
    let mut out = Array2::<u8>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let normalized = Point2::new((x as f64 - cx) / fx, (y as f64 - cy) / fy);
            let distorted_normalized = distortion.distort_normalized(normalized);
            let src_x = fx * distorted_normalized.x + cx;
            let src_y = fy * distorted_normalized.y + cy;
            out[(y, x)] = bilinear_sample_u8(source, src_x, src_y);
        }
    }
    out
}

fn bilinear_sample_u8(image: &Array2<u8>, x: f64, y: f64) -> u8 {
    let (h, w) = image.dim();
    if x < 0.0 || y < 0.0 || x >= (w - 1) as f64 || y >= (h - 1) as f64 {
        return 0;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;
    let v00 = image[(y0, x0)] as f64;
    let v01 = image[(y0, x0 + 1)] as f64;
    let v10 = image[(y0 + 1, x0)] as f64;
    let v11 = image[(y0 + 1, x0 + 1)] as f64;
    let value = v00 * (1.0 - fx) * (1.0 - fy)
        + v01 * fx * (1.0 - fy)
        + v10 * (1.0 - fx) * fy
        + v11 * fx * fy;
    value.round().clamp(0.0, 255.0) as u8
}

fn nearest_ground_truth(samples: &[EurocGroundTruthSample], target_ts: i128) -> Option<&EurocGroundTruthSample> {
    if samples.is_empty() {
        return None;
    }
    let idx = samples.binary_search_by_key(&target_ts, |s| s.timestamp_nanoseconds).unwrap_or_else(|insert| {
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
    Some(&samples[idx])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    fs::create_dir_all(&args.out_dir)?;

    let dataset = read_euroc_dataset_dir(&args.euroc_dir)?;
    println!(
        "loaded euroc cam0_frames={} gt_samples={} resolution={:?} intrinsics={:?} distortion={:?}",
        dataset.cam0_images.len(),
        dataset.ground_truth.len(),
        dataset.cam0_calibration.resolution,
        dataset.cam0_calibration.intrinsics,
        dataset.cam0_calibration.distortion_coefficients,
    );

    let (width, height) = (dataset.cam0_calibration.resolution.0 as usize, dataset.cam0_calibration.resolution.1 as usize);
    let intrinsics = dataset.cam0_calibration.intrinsics;
    let distortion = RadialTangential::from_euroc_coefficients(&dataset.cam0_calibration.distortion_coefficients)
        .unwrap_or(RadialTangential::IDENTITY);

    let backend = if args.onnx_cpu { OnnxBackend::Cpu } else { OnnxBackend::default() };
    let odometry_config = DpvoOdometryConfig {
        vo: DpvoVoConfig {
            buffer_size: 4096,
            patches_per_frame: args.patches_per_frame,
            removal_window: args.removal_window,
            optimization_window: args.optimization_window,
            patch_lifetime: args.patch_lifetime,
            keyframe_index: args.keyframe_index,
            keyframe_thresh: args.keyframe_thresh,
            motion_damping: args.motion_damping,
        },
        width,
        height,
        intrinsics: DpvoIntrinsics { fx: intrinsics[0], fy: intrinsics[1], cx: intrinsics[2], cy: intrinsics[3] },
        ba_lmbda: 1.0e-4,
        ba_ep: 100.0,
        motion_probe_min_flow: 2.0,
        seed: args.seed,
        // Milestone M5 (`docs/dpvo_droid_port_plan.md`): `--imu` couples
        // `mav0/imu0/data.csv` into the joint solve via
        // `crate::dpvo_vi_ba`; omitting the flag reproduces M4/M4-perf's
        // visual-only behavior exactly (`imu: None`).
        imu: args.imu.then(|| DpvoImuConfig {
            body_to_camera: se3_from_t_bs(&dataset.cam0_calibration.t_body_sensor),
            noise: ImuNoiseModel {
                gyroscope_noise_density: dataset.imu_calibration.gyroscope_noise_density * args.imu_noise_scale,
                accelerometer_noise_density: dataset.imu_calibration.accelerometer_noise_density * args.imu_noise_scale,
            },
            gravity_magnitude: 9.81,
            gravity_norm_deviation_ratio: args.imu_gravity_norm_deviation_ratio,
            min_bootstrap_factors: args.imu_min_bootstrap_factors,
        }),
    };

    if args.imu {
        println!(
            "imu enabled: samples={} gyro_noise_density={:.6e} accel_noise_density={:.6e} \
             body_to_camera_t=[{:.4},{:.4},{:.4}]",
            dataset.imu_samples.len(),
            dataset.imu_calibration.gyroscope_noise_density,
            dataset.imu_calibration.accelerometer_noise_density,
            odometry_config.imu.as_ref().unwrap().body_to_camera.translation.x,
            odometry_config.imu.as_ref().unwrap().body_to_camera.translation.y,
            odometry_config.imu.as_ref().unwrap().body_to_camera.translation.z,
        );
    }

    let mut odometry = DpvoOdometry::new(
        odometry_config,
        args.model_dir.join("fnet.onnx"),
        args.model_dir.join("inet.onnx"),
        args.model_dir.join("dpvo_update_pre_agg.onnx"),
        args.model_dir.join("dpvo_update_post_agg.onnx"),
        args.model_dir.join("fixtures").join("softagg_weights_fixture.npz"),
        backend,
    )?;

    let frame_cap = if args.max_frames == 0 { usize::MAX } else { args.max_frames };
    let frames: Vec<_> = dataset.cam0_images.iter().step_by(args.stride.max(1)).take(frame_cap).collect();
    println!("processing {} frames (stride={})", frames.len(), args.stride);

    let mut traj_csv = String::from("timestamp_ns,tx,ty,tz,qw,qx,qy,qz\n");
    let mut aligned_estimated: Vec<Point3<f64>> = Vec::new();
    let mut aligned_reference: Vec<Point3<f64>> = Vec::new();
    let mut tracked_frames = 0usize;

    // Coarse timing split for everything *outside* `DpvoOdometry` itself
    // (`DpvoOdometryStats` only covers ONNX/BA time inside `process_frame`;
    // this repo's own decode/undistort path turned out to dominate total
    // wall time on this machine — see the M4 results section of
    // `docs/dpvo_droid_port_plan.md` for the measured split — so it is
    // timed here rather than silently folded into "everything else").
    let mut io_ms_total = 0.0_f64;
    let mut undistort_ms_total = 0.0_f64;
    // Milestone M5: running cursor into `dataset.imu_samples` (file-order,
    // i.e. already timestamp-sorted per `read_euroc_imu_csv`'s own doc
    // comment) — every sample up to and including each camera frame's own
    // timestamp is pushed just before that frame is processed, mirroring
    // how a real-time streaming caller would interleave the two sensors.
    let mut imu_cursor = 0usize;

    let run_start = Instant::now();
    for (idx, entry) in frames.iter().enumerate() {
        let image_path = dataset.cam0_image_dir.join(&entry.filename);
        let io_start = Instant::now();
        let grayscale = read_common_image(&image_path)?;
        // `GrayscaleImage` stores normalized `[0,1]` f32 samples
        // (`crates/vision/src/features/mod.rs`); DPVO's own contract is raw
        // `[0,255]` pixels (`dpvo_vo.rs`'s `grayscale_to_input_tensor` doc),
        // so convert back to `u8` here at the loader boundary.
        let mut raw = Array2::<u8>::zeros((grayscale.height(), grayscale.width()));
        for y in 0..grayscale.height() {
            for x in 0..grayscale.width() {
                let normalized = grayscale.get(x, y).unwrap_or(0.0);
                raw[(y, x)] = (normalized * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
        io_ms_total += io_start.elapsed().as_secs_f64() * 1000.0;

        let undistort_start = Instant::now();
        let undistorted = undistort_image(&raw, intrinsics, &distortion);
        undistort_ms_total += undistort_start.elapsed().as_secs_f64() * 1000.0;

        let timestamp_seconds = entry.timestamp_nanoseconds as f64 * 1.0e-9;

        if args.imu {
            while imu_cursor < dataset.imu_samples.len()
                && dataset.imu_samples[imu_cursor].timestamp_nanoseconds <= entry.timestamp_nanoseconds
            {
                let sample: &EurocImuSample = &dataset.imu_samples[imu_cursor];
                odometry.push_imu(sample.timestamp_nanoseconds as f64 * 1.0e-9, sample.gyro, sample.accel);
                imu_cursor += 1;
            }
        }

        let pose = odometry.process_frame(undistorted.view(), timestamp_seconds)?;

        if let Some(pose_world_to_camera) = pose {
            tracked_frames += 1;
            // DPVO poses are `T_world_to_camera` (see `dpvo_patch_ba.rs`'s
            // convention-mapping doc) — the camera center in world is the
            // inverse's translation.
            let camera_in_world = pose_world_to_camera.inverse();
            let q = camera_in_world.rotation.quaternion();
            traj_csv.push_str(&format!(
                "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                entry.timestamp_nanoseconds,
                camera_in_world.translation.x,
                camera_in_world.translation.y,
                camera_in_world.translation.z,
                q.w,
                q.i,
                q.j,
                q.k,
            ));
            if let Some(gt) = nearest_ground_truth(&dataset.ground_truth, entry.timestamp_nanoseconds) {
                aligned_estimated.push(Point3::from(camera_in_world.translation));
                aligned_reference.push(Point3::from(gt.position_world));
            }
        }

        if idx % 10 == 0 || idx + 1 == frames.len() {
            let stats = odometry.stats();
            let n = stats.frames_processed.max(1) as f64;
            let imu_diag = odometry.imu_diagnostics();
            println!(
                "frame {}/{} tracked={} frames_graph_n={} io_ms_avg={:.2} undistort_ms_avg={:.2} encode_ms_avg={:.2} corr_ms_avg={:.2} update_ms_avg={:.2} ba_ms_avg={:.2} imu_bootstrapped={}",
                idx + 1,
                frames.len(),
                tracked_frames,
                odometry.graph().n_frames(),
                io_ms_total / n,
                undistort_ms_total / n,
                stats.encode_ms_total / n,
                stats.correlation_ms_total / n,
                stats.update_ms_total / n,
                stats.ba_ms_total / n,
                imu_diag.bootstrapped,
            );
        }
    }
    let total_elapsed_s = run_start.elapsed().as_secs_f64();

    let traj_path = args.out_dir.join("dpvo_trajectory.csv");
    fs::write(&traj_path, &traj_csv)?;

    let stats = odometry.stats();
    let ms_per_frame = if stats.frames_processed > 0 { total_elapsed_s * 1000.0 / stats.frames_processed as f64 } else { 0.0 };

    let (ate_rigid_rmse, ate_rigid_max, ate_sim_rmse, ate_sim_max, ate_sim_scale) = if aligned_estimated.len() >= 3 {
        let rigid = umeyama_similarity_transform(&aligned_estimated, &aligned_reference, false)
            .unwrap_or_else(TrajectorySimilarityTransform::identity);
        let similarity = umeyama_similarity_transform(&aligned_estimated, &aligned_reference, true)
            .unwrap_or_else(TrajectorySimilarityTransform::identity);
        let mut rmse_rigid_sq = 0.0;
        let mut max_rigid = 0.0_f64;
        let mut rmse_sim_sq = 0.0;
        let mut max_sim = 0.0_f64;
        for (est, gt) in aligned_estimated.iter().zip(aligned_reference.iter()) {
            let rigid_err = (rigid.apply(est) - gt).norm();
            let sim_err = (similarity.apply(est) - gt).norm();
            rmse_rigid_sq += rigid_err * rigid_err;
            rmse_sim_sq += sim_err * sim_err;
            max_rigid = max_rigid.max(rigid_err);
            max_sim = max_sim.max(sim_err);
        }
        let n = aligned_estimated.len() as f64;
        (
            (rmse_rigid_sq / n).sqrt(),
            max_rigid,
            (rmse_sim_sq / n).sqrt(),
            max_sim,
            similarity.scale,
        )
    } else {
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    };

    let tracked_fraction = if !frames.is_empty() { tracked_frames as f64 / frames.len() as f64 } else { 0.0 };
    let imu_diag = odometry.imu_diagnostics();
    let (gravity_x, gravity_y, gravity_z) = imu_diag
        .gravity_world
        .map(|g| (g.x, g.y, g.z))
        .unwrap_or((f64::NAN, f64::NAN, f64::NAN));

    let summary = format!(
        "euroc_dir={}\n\
         model_dir={}\n\
         frames_requested={frame_count}\n\
         frames_tracked={tracked_frames}\n\
         tracked_fraction={tracked_fraction:.4}\n\
         total_elapsed_s={total_elapsed_s:.2}\n\
         ms_per_frame_total={ms_per_frame:.2}\n\
         ms_per_frame_io={io_ms:.2}\n\
         ms_per_frame_undistort={undistort_ms:.2}\n\
         ms_per_frame_encode={encode_ms:.2}\n\
         ms_per_frame_correlation={corr_ms:.2}\n\
         ms_per_frame_update={update_ms:.2}\n\
         ms_per_frame_ba={ba_ms:.2}\n\
         ate_rigid_rmse_m={ate_rigid_rmse:.4}\n\
         ate_rigid_max_m={ate_rigid_max:.4}\n\
         ate_similarity_rmse_m={ate_sim_rmse:.4}\n\
         ate_similarity_max_m={ate_sim_max:.4}\n\
         ate_similarity_scale={ate_sim_scale:.6}\n\
         gt_matched_samples={matched}\n\
         imu_enabled={imu_enabled}\n\
         imu_bootstrapped={imu_bootstrapped}\n\
         imu_gravity_world_x={gravity_x:.4}\n\
         imu_gravity_world_y={gravity_y:.4}\n\
         imu_gravity_world_z={gravity_z:.4}\n\
         imu_bias_gyro_x={bias_gyro_x:.6}\n\
         imu_bias_gyro_y={bias_gyro_y:.6}\n\
         imu_bias_gyro_z={bias_gyro_z:.6}\n\
         imu_bias_accel_x={bias_accel_x:.6}\n\
         imu_bias_accel_y={bias_accel_y:.6}\n\
         imu_bias_accel_z={bias_accel_z:.6}\n",
        args.euroc_dir.display(),
        args.model_dir.display(),
        frame_count = frames.len(),
        io_ms = io_ms_total / stats.frames_processed.max(1) as f64,
        undistort_ms = undistort_ms_total / stats.frames_processed.max(1) as f64,
        encode_ms = stats.encode_ms_total / stats.frames_processed.max(1) as f64,
        corr_ms = stats.correlation_ms_total / stats.frames_processed.max(1) as f64,
        update_ms = stats.update_ms_total / stats.frames_processed.max(1) as f64,
        ba_ms = stats.ba_ms_total / stats.frames_processed.max(1) as f64,
        matched = aligned_estimated.len(),
        imu_enabled = args.imu,
        imu_bootstrapped = imu_diag.bootstrapped,
        bias_gyro_x = imu_diag.bias_gyro.x,
        bias_gyro_y = imu_diag.bias_gyro.y,
        bias_gyro_z = imu_diag.bias_gyro.z,
        bias_accel_x = imu_diag.bias_accel.x,
        bias_accel_y = imu_diag.bias_accel.y,
        bias_accel_z = imu_diag.bias_accel.z,
    );
    println!("{summary}");
    fs::write(args.out_dir.join("summary.txt"), &summary)?;
    println!("wrote {} and summary.txt to {}", traj_path.display(), args.out_dir.display());
    Ok(())
}
