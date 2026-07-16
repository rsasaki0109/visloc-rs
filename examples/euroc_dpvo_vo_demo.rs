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

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use nalgebra::{Point2, Point3};
use ndarray::Array2;

use visloc_rs::io::euroc::{read_euroc_dataset_dir, EurocGroundTruthSample};
use visloc_rs::io::images::read_common_image;
use visloc_rs::slam::dpvo_patch_graph::DpvoVoConfig;
use visloc_rs::slam::dpvo_vo::{DpvoOdometry, DpvoOdometryConfig};
use visloc_rs::slam::DpvoIntrinsics;
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
            other => return Err(format!("unknown argument: {other}").into()),
        }
        raw.remove(i);
    }
    args.euroc_dir = euroc_dir.ok_or("--euroc-dir <path/to/MH_01_easy> is required")?;
    Ok(args)
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
    };

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
            println!(
                "frame {}/{} tracked={} frames_graph_n={} io_ms_avg={:.2} undistort_ms_avg={:.2} encode_ms_avg={:.2} corr_ms_avg={:.2} update_ms_avg={:.2} ba_ms_avg={:.2}",
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
         gt_matched_samples={matched}\n",
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
    );
    println!("{summary}");
    fs::write(args.out_dir.join("summary.txt"), &summary)?;
    println!("wrote {} and summary.txt to {}", traj_path.display(), args.out_dir.display());
    Ok(())
}
