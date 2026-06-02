//! Metric stereo visual odometry on KITTI.
//!
//! Loads a rectified KITTI grayscale stereo pair (`<KITTI>/sequences/00/image_0`
//! for the left camera, `image_1` for the right) plus its `calib.txt`, runs
//! per-frame stereo triangulation with the recovered baseline `b = −tx / fx`,
//! and integrates a fully metric VO trajectory by stitching consecutive
//! stereo frames together with PnP RANSAC. Unlike the monocular companion
//! demo (`online_slam_image_vo_loop_demo`) this trajectory needs no scale-
//! recovery pass — the baseline anchors metric scale on every frame.
//!
//! Optionally refines the trajectory and the per-frame triangulations with
//! the new rectified-stereo BA (`BaStereoObservation`), which keeps the
//! metric gauge while reducing reprojection residuals. Pass
//! `--no-stereo-ba` to skip the BA pass.
//!
//! Usage:
//!
//! ```sh
//! cargo run --release --features image-io \
//!     --example online_slam_stereo_vo_kitti_demo -- \
//!     --image-left  /path/to/KITTI_odometry/sequences/00/image_0 \
//!     --image-right /path/to/KITTI_odometry/sequences/00/image_1 \
//!     --calib       /path/to/KITTI_odometry/sequences/00/calib.txt \
//!     --max-frames 200 \
//!     --frame-stride 4 \
//!     --out-dir target/kitti_stereo_vo_demo
//! ```
//!
//! Writes `vo.csv` (id, x, y, z) and, when BA runs, `ba.csv` to the output
//! directory plus a one-line `summary.txt` with the integrated trajectory
//! length and the BA cost reduction (when applicable).

#[cfg(not(feature = "image-io"))]
fn main() {
    eprintln!(
        "this example requires the `image-io` feature; rebuild with \
         `cargo run --release --features image-io --example online_slam_stereo_vo_kitti_demo`"
    );
    std::process::exit(2);
}

#[cfg(feature = "image-io")]
use std::env;
#[cfg(feature = "image-io")]
use std::fs;
#[cfg(feature = "image-io")]
use std::path::{Path, PathBuf};
#[cfg(feature = "image-io")]
use std::time::Instant;

#[cfg(feature = "image-io")]
use nalgebra::Point2;
#[cfg(feature = "image-io")]
use visloc_rs::core::geometry::Pose;
#[cfg(feature = "image-io")]
use visloc_rs::core::types::Camera;
#[cfg(feature = "image-io")]
use visloc_rs::io::calibration::parse_kitti_calibration_txt;
#[cfg(feature = "image-io")]
use visloc_rs::io::kitti::read_kitti_image_sequence_dir;
#[cfg(feature = "image-io")]
use visloc_rs::slam::gnc::{GncConfig, GncKernel, AUTO_SCALE_K};

/// Deterministic distinct-index sampler (inline LCG, no `rand` dependency —
/// the same convention as `pgo_g2o_robust_benchmark`). Returns `k` distinct
/// indices in `0..n` reproducibly from `seed`, so an outlier-injection run is
/// bit-identical across the Huber and GNC invocations being compared.
#[cfg(feature = "image-io")]
fn sample_distinct_indices(n: usize, k: usize, seed: u64) -> std::collections::HashSet<usize> {
    let mut out = std::collections::HashSet::new();
    if n == 0 {
        return out;
    }
    let target = k.min(n);
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    while out.len() < target {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        out.insert(((state >> 33) as usize) % n);
    }
    out
}

/// GNC outlier-rejection report for the stereo BA. The injected-* fields are
/// populated only when `--ba-inject-outliers` is active and give the recall /
/// false-positive of the GNC classification against the known injected labels.
#[cfg(feature = "image-io")]
struct GncBaReport {
    inlier_scale: f64,
    inliers: usize,
    outliers: usize,
    observations: usize,
    injected_total: usize,
    injected_rejected: usize,
    clean_rejected: usize,
}
#[cfg(feature = "image-io")]
use visloc_rs::vision::features::{
    FeatureSet, GrayscaleImage, HogLikeFeatureConfig, HogLikeFeatureExtractor,
};
#[cfg(feature = "image-io")]
use visloc_rs::vision::matching::{
    BruteForceMatcher, DescriptorMatch, Matcher, MutualSoftmaxConfig, MutualSoftmaxMatcher,
};
#[cfg(feature = "image-io")]
use visloc_rs::vision::stereo_vo::{
    estimate_relative_pose_kabsch_ransac, extend_stereo_tracks_via_projection, KabschRansacConfig,
    StereoPairCorrespondence, StereoRelativePoseMode, StereoRelativePoseSource, StereoVoFrontend,
    StereoVoFrontendConfig, StereoVoPairDiagnostics, TrackExtensionConfig,
};
#[cfg(feature = "image-io")]
use visloc_rs::vision::stereo_vo::{StereoFeature, StereoVoError};
#[cfg(feature = "image-io")]
use visloc_rs::vision::two_view::TwoViewCorrespondence;
#[cfg(feature = "image-io")]
use visloc_rs::{
    refine_stereo_vo_with_ba, relative_world_to_camera, scan_pairwise_loop_closures,
    write_colmap_binary_model_for_3dgs, write_colmap_text_model_for_3dgs, BaConfig,
    BaStereoObservation, BundleAdjustment, EssentialMatrixLoopClosureVerifier, LinearSolver,
    LoopClosureCandidate, LoopClosureConstraint, LoopClosureVerifier, PairwiseKeyframeView,
    PairwiseLoopClosureScannerConfig, PoseGraph, PoseGraphSe3Config, PoseTrajectory, RobustKernel,
    StereoVoBaConfig, TrackingEvent, TrackingState, TrajectoryAlignment, TrajectorySample,
};

#[cfg(feature = "image-io")]
#[derive(Debug)]
struct CliArgs {
    image_left: PathBuf,
    image_right: PathBuf,
    calib: PathBuf,
    out_dir: PathBuf,
    projection_left: String,
    projection_right: String,
    start_frame: usize,
    max_frames: usize,
    frame_stride: usize,
    min_pnp_inliers: usize,
    run_stereo_ba: bool,
    /// When `> 0`, replace the one-shot global BA with sliding-window local BA
    /// of this window size (overlapping windows, each fixing its first pose —
    /// the previous window's refined boundary). Scales to long sequences the
    /// one-shot BA cannot, and propagates drift correction window-to-window via
    /// [`refine_stereo_vo_with_ba`]'s `window_size`.
    window_ba: usize,
    /// After the `--window-ba` sweep, run one final GLOBAL BA over all frames
    /// seeded from the drift-suppressed windowed result (`window_size: None`).
    /// No effect unless `--window-ba` is also set.
    final_global_ba: bool,
    /// Run the stereo BA refinement with Graduated Non-Convexity outlier
    /// rejection (`BundleAdjustment::optimize_gnc`) instead of the default
    /// Huber M-estimator. GNC anneals from a convex (all-inlier) surrogate to
    /// the true robust cost, so wrong VO-chaining correspondences are switched
    /// off at the back-end rather than merely down-weighted.
    ba_gnc: bool,
    /// GNC surrogate family when `--ba-gnc` is set: truncated-least-squares
    /// (hard 0/1 verdict, exact inlier recovery — the default) or Geman-McClure
    /// (smooth weights, decisive identification but loosens weak directions).
    ba_gnc_kernel: GncKernel,
    /// GNC inlier scale `c` in pixels (reprojection-residual threshold). Under
    /// `--ba-gnc-auto-c` this is only a floor; otherwise it is used verbatim.
    ba_gnc_c: f64,
    /// Auto-estimate the GNC inlier scale from the residual MAD
    /// (`GncConfig::auto_scale`) so the pixel threshold tracks the run's own
    /// reprojection-noise spread instead of a hand-set value.
    ba_gnc_auto_c: bool,
    /// Re-estimate the GNC auto inlier scale at every μ level instead of once
    /// (`GncConfig::auto_scale_readapt`). Implies `--ba-gnc-auto-c`. Lets the
    /// scale contract as outliers are suppressed, recovering recall on heavily
    /// contaminated data where the one-shot estimate inflates.
    ba_gnc_readapt: bool,
    /// Inject this many gross outliers into the stereo BA observations before
    /// optimizing (controlled contamination of the real KITTI tracks). `0`
    /// disables injection. Deterministic in `--ba-inject-seed`, so the Huber
    /// and GNC runs being compared see identical corruption.
    ba_inject_outliers: usize,
    /// Seed for the deterministic outlier-injection sampler.
    ba_inject_seed: u64,
    /// Pixel offset applied to each injected outlier (added to the left `u`
    /// and the right `u`), simulating a gross wrong temporal data association.
    ba_inject_offset_px: f64,
    /// Optional KITTI ground-truth poses file (e.g. `dataset/poses/00.txt`).
    /// Used only for ATE evaluation; subsampled to match the frame indices
    /// the demo actually consumes.
    gt_poses: Option<PathBuf>,
    /// Original-stream stride that the loaded image sequence was already
    /// subsampled by (e.g. `8` if `scripts/fetch_kitti_seq00_images.py` ran
    /// with `--stride 8`). The demo subsamples GT poses by this stride
    /// before its own `--frame-stride` so estimated and reference frame
    /// indices line up.
    gt_original_stride: usize,
    /// When `--gt-poses` is provided, also derive a synthetic loop-closure
    /// edge between the first and the last keyframe from ground truth and
    /// run SE(3) PGO. Demonstrates the full SLAM stack (VO → BA →
    /// loop closure → PGO) without needing a physical visual loop in the
    /// data window.
    run_synthetic_loop_closure: bool,
    /// VO frontend selection: `classical` keeps the original
    /// `CornerFeatureExtractor` + `BruteForceMatcher` path, `deep` swaps
    /// in `HogLikeFeatureExtractor` + `MutualSoftmaxMatcher` through the
    /// generic `StereoVoFrontend::new_with` constructor.
    frontend: StereoFrontendChoice,
    /// Deep frontend feature cap. Lower values trade some matching
    /// redundancy for much faster long-sequence runs.
    deep_max_features: usize,
    /// Lowe-style HOG descriptor clipping threshold.
    deep_descriptor_clip: f32,
    /// Mutual-softmax confidence floor for deep temporal/stereo matching.
    deep_min_confidence: f32,
    /// Mutual-softmax inverse temperature for deep temporal/stereo matching.
    deep_temperature: f32,
    /// RANSAC iterations for consecutive-frame relative pose estimation.
    relative_pose_iterations: usize,
    /// PnP RANSAC reprojection threshold in pixels.
    pnp_reprojection_threshold_px: f64,
    /// Optional max depth for 3D points passed into consecutive-frame PnP.
    pnp_max_depth_m: Option<f64>,
    /// Optional comma-separated max-depth hypotheses for guarded PnP
    /// candidate selection.
    pnp_depth_hypotheses_m: Vec<f64>,
    /// Refine the selected relative pose against current left/right stereo
    /// reprojection residuals.
    stereo_pose_refinement: bool,
    stereo_vertical_alignment: bool,
    /// Optional override for the motion-scale rescue collapse threshold.
    motion_scale_rescue_min_translation_ratio: Option<f64>,
    /// Optional override for the rotation-vector rescue history length.
    rotation_vector_rescue_min_history: Option<usize>,
    /// Optional override for the rotation-vector rescue trigger delta.
    rotation_vector_rescue_max_delta_deg: Option<f64>,
    /// Consecutive-frame relative-pose source: `pnp` keeps the default
    /// 2D-3D PnP path, `kabsch` prefers metric 3D-3D stereo alignment.
    relative_pose_mode: StereoRelativePoseMode,
    /// Optional maximum vertical displacement for temporal matches.
    temporal_max_row_delta_px: Option<f64>,
    /// Print frontend progress every N processed frames. `0` disables
    /// periodic progress logs except for the first three pairs.
    progress_every: usize,
    /// Optional COLMAP text model output dir for 3DGS / NeRF bootstrap.
    /// When set, the demo writes `cameras.txt` / `images.txt` /
    /// `points3D.txt` under this directory using the final refined
    /// poses (BA-refined if `--no-stereo-ba` is not passed, else the
    /// raw VO poses) and the left-camera stereo features lifted via
    /// `pose.camera_to_world()`.
    colmap_export_dir: Option<PathBuf>,
    /// Binary counterpart of `--colmap-export`. Independent of it, so a
    /// single run can emit both text and binary forms.
    colmap_export_binary_dir: Option<PathBuf>,
    /// Prefix for the image NAME field embedded in `images.{txt,bin}`.
    /// The final NAME is `<prefix><6-digit frame_idx><suffix>`.
    colmap_image_prefix: String,
    /// Suffix for the image NAME field embedded in `images.{txt,bin}`.
    /// Defaults to `.png` to match the KITTI image filenames.
    colmap_image_suffix: String,
}

#[cfg(feature = "image-io")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StereoFrontendChoice {
    Classical,
    Deep,
}

#[cfg(feature = "image-io")]
impl StereoFrontendChoice {
    fn parse(value: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match value {
            "classical" | "corner" => Ok(Self::Classical),
            "deep" | "hog" | "lightglue" => Ok(Self::Deep),
            other => Err(format!("--frontend must be classical|deep, got {other}").into()),
        }
    }
}

#[cfg(feature = "image-io")]
fn parse_args() -> Result<CliArgs, Box<dyn std::error::Error>> {
    let mut image_left: Option<PathBuf> = None;
    let mut image_right: Option<PathBuf> = None;
    let mut calib: Option<PathBuf> = None;
    let mut out_dir: PathBuf = PathBuf::from("target/kitti_stereo_vo_demo");
    let mut projection_left = String::from("P0");
    let mut projection_right = String::from("P1");
    let mut start_frame: usize = 0;
    let mut max_frames: usize = 200;
    let mut frame_stride: usize = 4;
    let mut min_pnp_inliers: usize = 12;
    let mut run_stereo_ba = true;
    let mut window_ba: usize = 0;
    let mut final_global_ba = false;
    let mut ba_gnc = false;
    let mut ba_gnc_kernel = GncKernel::TruncatedLeastSquares;
    let mut ba_gnc_c: f64 = 4.0;
    let mut ba_gnc_auto_c = false;
    let mut ba_gnc_readapt = false;
    let mut ba_inject_outliers: usize = 0;
    let mut ba_inject_seed: u64 = 1;
    let mut ba_inject_offset_px: f64 = 60.0;
    let mut gt_poses: Option<PathBuf> = None;
    let mut gt_original_stride: usize = 1;
    let mut run_synthetic_loop_closure = false;
    let mut frontend = StereoFrontendChoice::Classical;
    let mut deep_max_features: usize = 1500;
    let mut deep_descriptor_clip: f32 = 0.2;
    let mut deep_min_confidence: f32 = 0.15;
    let mut deep_temperature: f32 = 25.0;
    let mut relative_pose_iterations: usize = 4000;
    let mut pnp_reprojection_threshold_px: f64 = 3.32;
    let mut pnp_max_depth_m: Option<f64> = None;
    let mut pnp_depth_hypotheses_m: Vec<f64> = Vec::new();
    let mut stereo_pose_refinement = false;
    let mut stereo_vertical_alignment = false;
    let mut motion_scale_rescue_min_translation_ratio: Option<f64> = None;
    let mut rotation_vector_rescue_min_history: Option<usize> = None;
    let mut rotation_vector_rescue_max_delta_deg: Option<f64> = None;
    let mut relative_pose_mode = StereoRelativePoseMode::PnpThenKabsch;
    let mut temporal_max_row_delta_px: Option<f64> = None;
    let mut progress_every: usize = 25;
    let mut colmap_export_dir: Option<PathBuf> = None;
    let mut colmap_export_binary_dir: Option<PathBuf> = None;
    let mut colmap_image_prefix: String = String::new();
    let mut colmap_image_suffix: String = String::from(".png");

    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--image-left" => {
                image_left = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--image-right" => {
                image_right = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--calib" => {
                calib = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--out-dir" => {
                out_dir = PathBuf::from(args.remove(i + 1));
                args.remove(i);
            }
            "--projection-left" => {
                projection_left = args.remove(i + 1);
                args.remove(i);
            }
            "--projection-right" => {
                projection_right = args.remove(i + 1);
                args.remove(i);
            }
            "--start-frame" => {
                start_frame = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--max-frames" => {
                max_frames = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--frame-stride" => {
                frame_stride = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--min-pnp-inliers" => {
                min_pnp_inliers = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--window-ba" => {
                let value = args.remove(i + 1);
                window_ba = value.parse().map_err(|_| {
                    format!("--window-ba expects a positive window size, got {value}")
                })?;
                args.remove(i);
            }
            "--final-global-ba" => {
                final_global_ba = true;
                args.remove(i);
            }
            "--no-stereo-ba" => {
                run_stereo_ba = false;
                args.remove(i);
            }
            "--ba-gnc" => {
                ba_gnc = true;
                args.remove(i);
            }
            "--ba-gnc-kernel" => {
                let value = args.remove(i + 1);
                args.remove(i);
                ba_gnc_kernel = match value.to_ascii_lowercase().as_str() {
                    "tls" | "truncated" | "truncated-least-squares" => {
                        GncKernel::TruncatedLeastSquares
                    }
                    "gm" | "geman-mcclure" | "gemanmcclure" => GncKernel::GemanMcClure,
                    other => {
                        return Err(format!(
                            "unknown --ba-gnc-kernel '{other}' (expected tls or gm)"
                        )
                        .into());
                    }
                };
            }
            "--ba-gnc-c" => {
                ba_gnc_c = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--ba-gnc-auto-c" => {
                ba_gnc_auto_c = true;
                args.remove(i);
            }
            "--ba-gnc-readapt" => {
                ba_gnc_readapt = true;
                ba_gnc_auto_c = true;
                args.remove(i);
            }
            "--ba-inject-outliers" => {
                ba_inject_outliers = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--ba-inject-seed" => {
                ba_inject_seed = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--ba-inject-offset-px" => {
                ba_inject_offset_px = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--gt-poses" => {
                gt_poses = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--gt-original-stride" => {
                gt_original_stride = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--synthetic-loop-closure" => {
                run_synthetic_loop_closure = true;
                args.remove(i);
            }
            "--frontend" => {
                let value = args.remove(i + 1);
                args.remove(i);
                frontend = StereoFrontendChoice::parse(&value)?;
            }
            "--deep-max-features" => {
                deep_max_features = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--deep-descriptor-clip" => {
                deep_descriptor_clip = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--deep-min-confidence" => {
                deep_min_confidence = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--deep-temperature" => {
                deep_temperature = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--relative-pose-iterations" => {
                relative_pose_iterations = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--pnp-reprojection-threshold" => {
                pnp_reprojection_threshold_px = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--pnp-max-depth" => {
                pnp_max_depth_m = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--pnp-depth-hypotheses" => {
                pnp_depth_hypotheses_m = parse_depth_hypotheses(&args.remove(i + 1))?;
                args.remove(i);
            }
            "--stereo-pose-refinement" => {
                stereo_pose_refinement = true;
                args.remove(i);
            }
            "--stereo-vertical-alignment" => {
                stereo_vertical_alignment = true;
                args.remove(i);
            }
            "--motion-scale-rescue-min-translation-ratio" => {
                let value = args.remove(i + 1).parse::<f64>()?;
                if !value.is_finite() || value < 0.0 {
                    return Err(format!(
                        "--motion-scale-rescue-min-translation-ratio must be finite and non-negative, got {value}"
                    )
                    .into());
                }
                motion_scale_rescue_min_translation_ratio = Some(value);
                args.remove(i);
            }
            "--rotation-vector-rescue-min-history" => {
                let value = args.remove(i + 1).parse::<usize>()?;
                if value == 0 {
                    return Err("--rotation-vector-rescue-min-history must be positive".into());
                }
                rotation_vector_rescue_min_history = Some(value);
                args.remove(i);
            }
            "--rotation-vector-rescue-max-delta-deg" => {
                let value = args.remove(i + 1).parse::<f64>()?;
                if !value.is_finite() || value < 0.0 {
                    return Err(format!(
                        "--rotation-vector-rescue-max-delta-deg must be finite and non-negative, got {value}"
                    )
                    .into());
                }
                rotation_vector_rescue_max_delta_deg = Some(value);
                args.remove(i);
            }
            "--relative-pose-mode" => {
                let value = args.remove(i + 1);
                args.remove(i);
                relative_pose_mode = parse_relative_pose_mode(&value)?;
            }
            "--temporal-max-row-delta" => {
                temporal_max_row_delta_px = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--progress-every" => {
                progress_every = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--colmap-export" => {
                colmap_export_dir = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--colmap-export-binary" => {
                colmap_export_binary_dir = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--colmap-image-prefix" => {
                colmap_image_prefix = args.remove(i + 1);
                args.remove(i);
            }
            "--colmap-image-suffix" => {
                colmap_image_suffix = args.remove(i + 1);
                args.remove(i);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    let image_left = image_left.ok_or("--image-left <path> is required")?;
    let image_right = image_right.ok_or("--image-right <path> is required")?;
    let calib = calib.ok_or("--calib <path/to/calib.txt> is required")?;
    Ok(CliArgs {
        image_left,
        image_right,
        calib,
        out_dir,
        projection_left,
        projection_right,
        start_frame,
        max_frames,
        frame_stride,
        min_pnp_inliers,
        run_stereo_ba,
        window_ba,
        final_global_ba,
        ba_gnc,
        ba_gnc_kernel,
        ba_gnc_c,
        ba_gnc_auto_c,
        ba_gnc_readapt,
        ba_inject_outliers,
        ba_inject_seed,
        ba_inject_offset_px,
        gt_poses,
        gt_original_stride,
        run_synthetic_loop_closure,
        frontend,
        deep_max_features,
        deep_descriptor_clip,
        deep_min_confidence,
        deep_temperature,
        relative_pose_iterations,
        pnp_reprojection_threshold_px,
        pnp_max_depth_m,
        pnp_depth_hypotheses_m,
        stereo_pose_refinement,
        stereo_vertical_alignment,
        motion_scale_rescue_min_translation_ratio,
        rotation_vector_rescue_min_history,
        rotation_vector_rescue_max_delta_deg,
        relative_pose_mode,
        temporal_max_row_delta_px,
        progress_every,
        colmap_export_dir,
        colmap_export_binary_dir,
        colmap_image_prefix,
        colmap_image_suffix,
    })
}

#[cfg(feature = "image-io")]
fn parse_depth_hypotheses(value: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let mut depths = Vec::new();
    for token in value.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        let depth = token.parse::<f64>()?;
        if !depth.is_finite() || depth <= 0.0 {
            return Err(
                format!("--pnp-depth-hypotheses values must be positive, got {token}").into(),
            );
        }
        if !depths.iter().any(|existing| {
            let existing: f64 = *existing;
            (existing - depth).abs() <= 1.0e-9
        }) {
            depths.push(depth);
        }
    }
    Ok(depths)
}

#[cfg(feature = "image-io")]
fn parse_relative_pose_mode(
    value: &str,
) -> Result<StereoRelativePoseMode, Box<dyn std::error::Error>> {
    match value {
        "pnp" | "pnp-then-kabsch" => Ok(StereoRelativePoseMode::PnpThenKabsch),
        "kabsch" | "kabsch-then-pnp" => Ok(StereoRelativePoseMode::KabschThenPnp),
        other => Err(format!("--relative-pose-mode must be pnp|kabsch, got {other}").into()),
    }
}

#[cfg(feature = "image-io")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    println!(
        "image_left={} image_right={} calib={} start_frame={} stride={} max_frames={}",
        args.image_left.display(),
        args.image_right.display(),
        args.calib.display(),
        args.start_frame,
        args.frame_stride,
        args.max_frames,
    );

    // Both eyes share the same intrinsics on KITTI rectified pairs, but the
    // demo loads them independently so a future caller can experiment with
    // diverging parametrizations.
    let left_seq =
        read_kitti_image_sequence_dir(&args.image_left, &args.calib, &args.projection_left, 0)?;
    let right_seq =
        read_kitti_image_sequence_dir(&args.image_right, &args.calib, &args.projection_right, 1)?;
    if left_seq.frames.len() != right_seq.frames.len() {
        return Err(format!(
            "left ({} frames) and right ({} frames) sequences must have equal length",
            left_seq.frames.len(),
            right_seq.frames.len()
        )
        .into());
    }
    let camera = left_seq.camera.clone();

    // Recover the baseline analytically from the projection matrices:
    // KITTI stores `P_i = [R_rect | t_i]` so `t_x = -fx · b`, hence
    // `b = -tx / fx`. `stereo_baseline_from` returns `Some(b)` only when
    // the pair shares intrinsics — exactly what rectified-stereo expects.
    let calib_text = fs::read_to_string(&args.calib)?;
    let projections = parse_kitti_calibration_txt(&calib_text)?;
    let p_left = projections
        .iter()
        .find(|p| p.label == args.projection_left)
        .ok_or_else(|| format!("calib missing {}", args.projection_left))?;
    let p_right = projections
        .iter()
        .find(|p| p.label == args.projection_right)
        .ok_or_else(|| format!("calib missing {}", args.projection_right))?;
    let baseline = p_right.stereo_baseline_from(p_left).ok_or_else(|| {
        format!(
            "calib pair {}↔{} did not yield a positive stereo baseline (intrinsics mismatch?)",
            args.projection_left, args.projection_right,
        )
    })?;
    println!(
        "camera {}x{} fx={:.2} cx={:.2} cy={:.2}; baseline={:.6} m",
        camera.width,
        camera.height,
        camera.params.first().copied().unwrap_or(0.0),
        camera.params.get(2).copied().unwrap_or(0.0),
        camera.params.get(3).copied().unwrap_or(0.0),
        baseline,
    );

    let left_frames: Vec<&GrayscaleImage> = left_seq
        .frames
        .iter()
        .skip(args.start_frame)
        .step_by(args.frame_stride.max(1))
        .take(args.max_frames)
        .map(|f| &f.image)
        .collect();
    let right_frames: Vec<&GrayscaleImage> = right_seq
        .frames
        .iter()
        .skip(args.start_frame)
        .step_by(args.frame_stride.max(1))
        .take(args.max_frames)
        .map(|f| &f.image)
        .collect();
    let n = left_frames.len();
    if n < 2 {
        return Err(format!("need at least 2 frames, got {n}").into());
    }
    println!("loaded n_frames={n}");

    // Drive the reusable rectified-stereo VO frontend over the loaded
    // image pairs. The frontend extracts features, triangulates each
    // pair, runs the row-restricted L↔R matcher + confidence-aware relative
    // pose estimation, and composes the metric trajectory; per-frame state
    // is exposed via its public fields for the BA pass below.
    let mut frontend_config = StereoVoFrontendConfig {
        kabsch: KabschRansacConfig {
            iterations: args.relative_pose_iterations,
            min_inliers: StereoVoFrontendConfig::default().kabsch.min_inliers,
            ..StereoVoFrontendConfig::default().kabsch
        },
        pnp_reprojection_threshold_px: args.pnp_reprojection_threshold_px,
        pnp_min_inliers: args.min_pnp_inliers,
        pnp_max_depth_m: args.pnp_max_depth_m,
        pnp_depth_hypotheses_m: args.pnp_depth_hypotheses_m.clone(),
        stereo_pose_refinement: args.stereo_pose_refinement,
        stereo_vertical_alignment: args.stereo_vertical_alignment,
        relative_pose_mode: args.relative_pose_mode,
        temporal_max_row_delta_px: args.temporal_max_row_delta_px,
        ..StereoVoFrontendConfig::default()
    };
    if let Some(ratio) = args.motion_scale_rescue_min_translation_ratio {
        frontend_config.motion_scale_rescue_min_translation_ratio = ratio;
    }
    if let Some(history) = args.rotation_vector_rescue_min_history {
        frontend_config.rotation_vector_rescue_min_history = history;
    }
    if let Some(delta_deg) = args.rotation_vector_rescue_max_delta_deg {
        frontend_config.rotation_vector_rescue_max_delta_deg = delta_deg;
    }
    let frontend_label = match args.frontend {
        StereoFrontendChoice::Classical => "classical (Corner + BF)",
        StereoFrontendChoice::Deep => "deep (HogLike + MutualSoftmax)",
    };
    println!(
        "frontend: {} (deep_max_features={} deep_descriptor_clip={} deep_min_confidence={} deep_temperature={} relative_pose_iterations={} pnp_reprojection_threshold_px={} pnp_max_depth_m={:?} pnp_depth_hypotheses_m={:?} pnp_adaptive_depth_hypothesis_m={:?} pnp_adaptive_depth_min_primary_inlier_ratio={} pnp_early_stop_min_iterations={} pnp_early_stop_inlier_ratio={:?} pnp_kabsch_challenge_max_inlier_ratio={} pnp_kabsch_challenge_min_3d_inlier_gain={} pnp_kabsch_challenge_max_residual_ratio={} motion_scale_rescue_min_history={} motion_scale_rescue_min_median_translation_m={} motion_scale_rescue_min_translation_ratio={} motion_scale_rescue_max_translation_ratio={} motion_scale_rescue_target_percentile={} motion_scale_rescue_max_pnp_inlier_ratio={} translation_direction_rescue_min_history={} translation_direction_rescue_min_median_translation_m={} translation_direction_rescue_max_angle_deg={} translation_direction_rescue_max_pnp_inlier_ratio={} rotation_spike_rescue_min_history={} rotation_spike_rescue_min_angle_deg={} rotation_spike_rescue_max_angle_ratio={} rotation_spike_rescue_max_pnp_inlier_ratio={} rotation_vector_rescue_min_history={} rotation_vector_rescue_min_median_translation_m={} rotation_vector_rescue_max_delta_deg={} rotation_vector_rescue_max_pnp_inlier_ratio={} stereo_pose_refinement={} stereo_vertical_alignment={} stereo_vertical_alignment_min_pairs={} stereo_vertical_alignment_max_correction_m={} stereo_pose_refinement_auto_min_history={} stereo_pose_refinement_auto_min_median_translation_m={} stereo_pose_refinement_auto_max_pnp_inlier_ratio={} relative_pose_mode={:?} temporal_max_row_delta_px={:?} temporal_auto_max_row_delta_px={:?} temporal_auto_min_history={} temporal_auto_min_median_translation_m={} temporal_auto_min_confidence={:?} temporal_auto_confidence_min_history={} temporal_auto_confidence_min_median_translation_m={} temporal_auto_confidence_curve_min_median_translation_m={:?} temporal_auto_confidence_curve_min_median_rotation_deg={:?} temporal_auto_confidence_max_median_rotation_deg={:?} progress_every={})",
        frontend_label,
        args.deep_max_features,
        args.deep_descriptor_clip,
        args.deep_min_confidence,
        args.deep_temperature,
        args.relative_pose_iterations,
        args.pnp_reprojection_threshold_px,
        args.pnp_max_depth_m,
        args.pnp_depth_hypotheses_m,
        frontend_config.pnp_adaptive_depth_hypothesis_m,
        frontend_config.pnp_adaptive_depth_min_primary_inlier_ratio,
        frontend_config.pnp_early_stop_min_iterations,
        frontend_config.pnp_early_stop_inlier_ratio,
        frontend_config.pnp_kabsch_challenge_max_inlier_ratio,
        frontend_config.pnp_kabsch_challenge_min_3d_inlier_gain,
        frontend_config.pnp_kabsch_challenge_max_residual_ratio,
        frontend_config.motion_scale_rescue_min_history,
        frontend_config.motion_scale_rescue_min_median_translation_m,
        frontend_config.motion_scale_rescue_min_translation_ratio,
        frontend_config.motion_scale_rescue_max_translation_ratio,
        frontend_config.motion_scale_rescue_target_percentile,
        frontend_config.motion_scale_rescue_max_pnp_inlier_ratio,
        frontend_config.translation_direction_rescue_min_history,
        frontend_config.translation_direction_rescue_min_median_translation_m,
        frontend_config.translation_direction_rescue_max_angle_deg,
        frontend_config.translation_direction_rescue_max_pnp_inlier_ratio,
        frontend_config.rotation_spike_rescue_min_history,
        frontend_config.rotation_spike_rescue_min_angle_deg,
        frontend_config.rotation_spike_rescue_max_angle_ratio,
        frontend_config.rotation_spike_rescue_max_pnp_inlier_ratio,
        frontend_config.rotation_vector_rescue_min_history,
        frontend_config.rotation_vector_rescue_min_median_translation_m,
        frontend_config.rotation_vector_rescue_max_delta_deg,
        frontend_config.rotation_vector_rescue_max_pnp_inlier_ratio,
        args.stereo_pose_refinement,
        args.stereo_vertical_alignment,
        frontend_config.stereo_vertical_alignment_min_pairs,
        frontend_config.stereo_vertical_alignment_max_correction_m,
        frontend_config.stereo_pose_refinement_auto_min_history,
        frontend_config.stereo_pose_refinement_auto_min_median_translation_m,
        frontend_config.stereo_pose_refinement_auto_max_pnp_inlier_ratio,
        args.relative_pose_mode,
        args.temporal_max_row_delta_px,
        frontend_config.temporal_auto_max_row_delta_px,
        frontend_config.temporal_auto_min_history,
        frontend_config.temporal_auto_min_median_translation_m,
        frontend_config.temporal_auto_min_confidence,
        frontend_config.temporal_auto_confidence_min_history,
        frontend_config.temporal_auto_confidence_min_median_translation_m,
        frontend_config.temporal_auto_confidence_curve_min_median_translation_m,
        frontend_config.temporal_auto_confidence_curve_min_median_rotation_deg,
        frontend_config.temporal_auto_confidence_max_median_rotation_deg,
        args.progress_every
    );
    let frontend_state = run_stereo_vo_frontend(
        args.frontend,
        &camera,
        baseline,
        &frontend_config,
        &left_frames,
        &right_frames,
        n,
        args.deep_max_features,
        args.deep_descriptor_clip,
        args.deep_min_confidence,
        args.deep_temperature,
        args.progress_every,
    )?;
    let FrontendState {
        poses: vo_poses,
        stereo_per_frame,
        left_features,
        right_features,
        per_pair_translation_m,
        kabsch_inlier_counts,
        pair_diagnostics,
    } = frontend_state;
    let stereo_counts: Vec<usize> = stereo_per_frame.iter().map(|s| s.len()).collect();
    println!(
        "stereo triangulations per frame: min={} median={} max={}",
        stereo_counts.iter().copied().min().unwrap_or(0),
        median(&stereo_counts),
        stereo_counts.iter().copied().max().unwrap_or(0),
    );
    println!(
        "relative-pose RANSAC inliers: mean={:.1} median={}",
        mean(&kabsch_inlier_counts),
        median(&kabsch_inlier_counts),
    );
    let (pnp_count, pnp_fallback_count, kabsch_count, kabsch_fallback_count) =
        relative_pose_source_counts(&pair_diagnostics);
    println!(
        "relative-pose source counts: pnp={} pnp_fallback={} kabsch={} kabsch_fallback={}",
        pnp_count, pnp_fallback_count, kabsch_count, kabsch_fallback_count,
    );
    println!("stereo VO: edges={}", vo_poses.len().saturating_sub(1));

    // Trajectory length: sum of consecutive camera-center deltas. With
    // monocular VO this would be in arbitrary units; here it's in meters.
    let centers: Vec<nalgebra::Point3<f64>> =
        vo_poses.iter().map(|p| p.camera_center_world()).collect();
    let total_length: f64 = centers.windows(2).map(|w| (w[1] - w[0]).norm()).sum();
    println!("stereo VO trajectory length: {:.2} m", total_length);

    fs::create_dir_all(&args.out_dir)?;
    write_pair_diagnostics_csv(
        &args.out_dir.join("frontend_pair_diagnostics.csv"),
        &pair_diagnostics,
    )?;
    write_stereo_ground_profile_csv(
        &args.out_dir.join("stereo_ground_profile.csv"),
        &stereo_per_frame,
        &left_features,
        &camera,
    )?;
    let vo_traj = build_pose_trajectory(&vo_poses);
    write_trajectory_csv(&args.out_dir.join("vo.csv"), &centers)?;
    vo_traj.write_kitti_poses(args.out_dir.join("vo_poses.txt"))?;

    // Optional: stereo BA refinement. Build a BA problem with all VO poses
    // (anchor the first), every triangulated point as a landmark (one ID
    // per (frame_index, stereo_feature_index) pair), and one
    // BaStereoObservation per stereo measurement. Frame i's points are in
    // frame-i's camera frame, so they need to be lifted into the world
    // frame using vo_poses[i]^{-1} before being added as landmarks.
    let mut ba_summary: Option<(f64, f64, usize)> = None;
    // GNC-only stats, populated when `--ba-gnc` is set. `None` for Huber.
    let mut ba_gnc_summary: Option<GncBaReport> = None;
    let mut ba_poses: Option<Vec<Pose>> = None;
    if args.run_stereo_ba {
        let mut ba = BundleAdjustment::new(camera.clone());
        ba.set_stereo_baseline(baseline);
        for (id, pose) in vo_poses.iter().enumerate() {
            ba.add_pose(id as u64, pose.clone());
        }
        ba.fix_pose(0);

        #[allow(clippy::type_complexity)]
        let outcome: Result<
            (f64, f64, usize, bool, Option<GncBaReport>),
            Box<dyn std::error::Error>,
        > = if args.window_ba > 0 {
            // Sliding-window local BA: overlapping `window_ba`-frame windows,
            // each fixing its first pose (the prior window's refined boundary).
            // Scales to long sequences the one-shot global BA cannot, and
            // propagates drift correction window-to-window. Temporal matches are
            // brute-force descriptor matches between consecutive left frames.
            let matcher = BruteForceMatcher { ratio: Some(0.8) };
            let temporal_matches: Vec<Vec<DescriptorMatch>> = (0..vo_poses.len().saturating_sub(1))
                .map(|i| {
                    matcher.match_descriptors(
                        &left_features[i].descriptors,
                        &left_features[i + 1].descriptors,
                    )
                })
                .collect();
            let cfg = StereoVoBaConfig {
                window_size: Some(args.window_ba),
                ba_config: BaConfig {
                    max_iterations: 12,
                    robust_kernel: RobustKernel::Huber { delta: 3.0 },
                    linear_solver: LinearSolver::Sparse,
                    ..BaConfig::default()
                },
                ..StereoVoBaConfig::default()
            };
            println!(
                "stereo BA back-end: sliding-window local BA (window={})",
                args.window_ba,
            );
            refine_stereo_vo_with_ba(
                &camera,
                baseline,
                &vo_poses,
                &left_features,
                &right_features,
                &stereo_per_frame,
                &temporal_matches,
                &cfg,
            )
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
            .and_then(|r_win| {
                // Optional final GLOBAL BA pass, seeded from the windowed sweep.
                // The one-shot global BA fails from the raw VO init (too large /
                // non-convex), but the windowed sweep already removed most of the
                // drift, so a joint solve from that good seed can converge and
                // attack the residual gap a purely-local window cannot reach.
                let refined = if args.final_global_ba {
                    let global_cfg = StereoVoBaConfig {
                        window_size: None,
                        ba_config: BaConfig {
                            max_iterations: 30,
                            robust_kernel: RobustKernel::Huber { delta: 3.0 },
                            linear_solver: LinearSolver::Sparse,
                            ..BaConfig::default()
                        },
                        ..StereoVoBaConfig::default()
                    };
                    println!(
                        "final global BA pass over {} frames (seeded from the windowed sweep)",
                        r_win.refined_poses.len(),
                    );
                    refine_stereo_vo_with_ba(
                        &camera,
                        baseline,
                        &r_win.refined_poses,
                        &left_features,
                        &right_features,
                        &stereo_per_frame,
                        &temporal_matches,
                        &global_cfg,
                    )
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?
                } else {
                    r_win
                };
                // Load the refined poses into `ba` so the shared read-back /
                // CSV / GT-eval path below works unchanged.
                for (i, p) in refined.refined_poses.iter().enumerate() {
                    ba.add_pose(i as u64, p.clone());
                }
                Ok((
                    refined.ba_result.initial_cost,
                    refined.ba_result.final_cost,
                    refined.ba_result.iterations.len(),
                    refined.ba_result.converged,
                    None,
                ))
            })
        } else {
            // Build multi-frame tracks via projection-guided extension. For
            // each frame's stereo features, project the landmark forward
            // through subsequent VO poses and search for descriptor-matching
            // stereo features near the projected pixel. This produces tracks
            // that span many more frames than the pair-chain alternative, so
            // BA gets long-baseline constraints to attack rotation drift.
            //
            // Search radius scales with observed per-pair translation, with a
            // 20 px floor that holds even when motion is small. The
            // frontend computes this from its `per_pair_translation_m` log.
            let search_radius_px = adaptive_track_search_radius_px(&per_pair_translation_m);
            println!("track-extension: search_radius_px={:.1}", search_radius_px,);
            let track_cfg = TrackExtensionConfig {
                search_radius_px,
                ratio: 0.8,
                max_extension: 30,
                deduplicate: true,
            };
            let tracks = extend_stereo_tracks_via_projection(
                &vo_poses,
                &left_features,
                &stereo_per_frame,
                &camera,
                &track_cfg,
            );
            let long_tracks: Vec<_> = tracks
                .iter()
                .filter(|t| t.observations.len() >= 3)
                .collect();
            let track_lengths: Vec<usize> =
                long_tracks.iter().map(|t| t.observations.len()).collect();
            println!(
                "stereo tracks (projection-guided): total={} long(≥3)={} mean_len={:.1} max_len={}",
                tracks.len(),
                long_tracks.len(),
                if track_lengths.is_empty() {
                    0.0
                } else {
                    track_lengths.iter().sum::<usize>() as f64 / track_lengths.len() as f64
                },
                track_lengths.iter().copied().max().unwrap_or(0),
            );

            // Collect every stereo observation first so the optional outlier
            // injection can address them by their insertion index — which, because
            // the demo adds no monocular observations, equals the index into the
            // GNC `observation_weights` vector. That alignment is what lets us score
            // the GNC classification against the known injected labels below.
            let mut stereo_obs: Vec<BaStereoObservation> = Vec::new();
            for (lm_id, track) in (0_u64..).zip(long_tracks.iter()) {
                ba.add_landmark(lm_id, track.landmark_world);
                for obs in &track.observations {
                    let l = left_features[obs.frame_index].keypoints[obs.left_index];
                    let r = right_features[obs.frame_index].keypoints[obs.right_index];
                    stereo_obs.push(BaStereoObservation {
                        keyframe_id: obs.frame_index as u64,
                        landmark_id: lm_id,
                        xy: Point2::new(l.x, l.y),
                        u_right: r.x,
                    });
                }
            }

            // Optional controlled contamination: corrupt N randomly-chosen stereo
            // observations by a large fixed pixel offset on both the left `u` and
            // the right `u`, simulating gross wrong temporal data associations. The
            // sampler is deterministic in the seed, so the Huber and GNC runs being
            // compared see *identical* corruption — the standard robust-SLAM
            // protocol (cf. `pgo_g2o_robust_benchmark`), here on real KITTI tracks.
            let injected: std::collections::HashSet<usize> =
                if args.ba_inject_outliers > 0 && !stereo_obs.is_empty() {
                    let set = sample_distinct_indices(
                        stereo_obs.len(),
                        args.ba_inject_outliers,
                        args.ba_inject_seed,
                    );
                    for &idx in &set {
                        let o = &mut stereo_obs[idx];
                        o.xy.coords.x += args.ba_inject_offset_px;
                        o.u_right += args.ba_inject_offset_px;
                    }
                    println!(
                        "BA outlier injection: corrupted {} / {} stereo observations \
                     (offset={:.1}px seed={})",
                        set.len(),
                        stereo_obs.len(),
                        args.ba_inject_offset_px,
                        args.ba_inject_seed,
                    );
                    set
                } else {
                    std::collections::HashSet::new()
                };

            for o in stereo_obs {
                ba.add_stereo_observation(o);
            }

            // Sparse Cholesky scales to the n_frames × stereo_per_frame size.
            // Robust kernel down-weights residual outliers (e.g., points that
            // were correctly triangulated within their own frame but matched
            // to the wrong temporal target during VO chaining).
            // Bound BA iterations: each LM step builds the Schur reduction over
            // every (pose-pair, landmark) cross block. For ~3000 landmarks /
            // ~6000 stereo observations on 100 keyframes the per-iteration cost
            // is dominated by the inner loop in `build_normal_equations`, so 5
            // LM iterations is enough to consume most of the cost reduction
            // without spending many minutes on diminishing returns.
            // 15 LM iterations: with long-track landmarks the per-iteration cost
            // is dominated by the Schur reduction over (pose-pair, landmark)
            // cross blocks; 5 iterations consume most of the cost reduction
            // but BA still has work to do on rotation drift.
            let ba_config = BaConfig {
                max_iterations: 15,
                linear_solver: LinearSolver::Sparse,
                robust_kernel: RobustKernel::Huber { delta: 4.0 },
                ..BaConfig::default()
            };

            // Both back-ends mutate `ba.poses` in place, so the pose read-back and
            // ATE/CSV plumbing below is shared. Normalize the two result types to
            // one tuple: (initial_cost, final_cost, iterations, converged, GNC
            // extras). The Huber path keeps `RobustKernel::Huber`; the GNC path
            // ignores `ba_config.robust_kernel` and anneals its own surrogate, so
            // wrong VO-chaining correspondences are switched off, not down-weighted.
            (if args.ba_gnc {
                let gnc_cfg = GncConfig {
                    kernel: args.ba_gnc_kernel,
                    c: args.ba_gnc_c,
                    auto_scale: if args.ba_gnc_auto_c {
                        Some(AUTO_SCALE_K)
                    } else {
                        None
                    },
                    auto_scale_readapt: args.ba_gnc_readapt,
                    ..GncConfig::default()
                };
                println!(
                    "stereo BA back-end: GNC (kernel={:?} c={:.2}px auto_c={} readapt={})",
                    gnc_cfg.kernel, gnc_cfg.c, args.ba_gnc_auto_c, args.ba_gnc_readapt,
                );
                ba.optimize_gnc(&ba_config, &gnc_cfg).map(|r| {
                    let inliers = r.inlier_count(0.5);
                    let outliers = r.outlier_count(0.5);
                    // `observation_weights` is monocular-first then stereo; the demo
                    // adds no monocular observations, so weight index i maps to the
                    // i-th injected/clean stereo observation. Score the w<0.5
                    // rejections against the known injected labels.
                    let mut injected_rejected = 0usize;
                    let mut clean_rejected = 0usize;
                    for (i, w) in r.observation_weights.iter().enumerate() {
                        if w.is_finite() && *w < 0.5 {
                            if injected.contains(&i) {
                                injected_rejected += 1;
                            } else {
                                clean_rejected += 1;
                            }
                        }
                    }
                    (
                        r.initial_cost,
                        r.final_cost,
                        r.outer_iterations,
                        r.converged,
                        Some(GncBaReport {
                            inlier_scale: r.inlier_scale,
                            inliers,
                            outliers,
                            observations: r.observation_count,
                            injected_total: injected.len(),
                            injected_rejected,
                            clean_rejected,
                        }),
                    )
                })
            } else {
                ba.optimize(&ba_config).map(|r| {
                    (
                        r.initial_cost,
                        r.final_cost,
                        r.iterations.len(),
                        r.converged,
                        None,
                    )
                })
            })
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
        };
        match outcome {
            Ok((initial_cost, final_cost, iters, converged, gnc_extra)) => {
                let refined_centers: Vec<nalgebra::Point3<f64>> = (0..vo_poses.len())
                    .map(|i| ba.poses[&(i as u64)].camera_center_world())
                    .collect();
                let refined_length: f64 = refined_centers
                    .windows(2)
                    .map(|w| (w[1] - w[0]).norm())
                    .sum();
                println!(
                    "stereo BA: cost {:.2} → {:.2} ({} iter, converged={}); length {:.2} m",
                    initial_cost, final_cost, iters, converged, refined_length,
                );
                if let Some(rep) = &gnc_extra {
                    println!(
                        "stereo BA GNC: inlier_scale={:.2}px inliers={}/{} outliers={} \
                         ({:.1}% rejected)",
                        rep.inlier_scale,
                        rep.inliers,
                        rep.observations,
                        rep.outliers,
                        if rep.observations > 0 {
                            100.0 * rep.outliers as f64 / rep.observations as f64
                        } else {
                            0.0
                        },
                    );
                    if rep.injected_total > 0 {
                        let clean = rep.observations.saturating_sub(rep.injected_total);
                        println!(
                            "stereo BA GNC outlier benchmark: injected={} \
                             recall={}/{} ({:.1}%) false_positives={}/{} ({:.2}%)",
                            rep.injected_total,
                            rep.injected_rejected,
                            rep.injected_total,
                            100.0 * rep.injected_rejected as f64 / rep.injected_total as f64,
                            rep.clean_rejected,
                            clean,
                            if clean > 0 {
                                100.0 * rep.clean_rejected as f64 / clean as f64
                            } else {
                                0.0
                            },
                        );
                    }
                }
                ba_gnc_summary = gnc_extra;
                write_trajectory_csv(&args.out_dir.join("ba.csv"), &refined_centers)?;
                ba_summary = Some((initial_cost, final_cost, iters));
                let refined_poses: Vec<Pose> = (0..vo_poses.len())
                    .map(|i| ba.poses[&(i as u64)].clone())
                    .collect();
                build_pose_trajectory(&refined_poses)
                    .write_kitti_poses(args.out_dir.join("ba_poses.txt"))?;
                ba_poses = Some(refined_poses);
            }
            Err(e) => {
                println!("stereo BA skipped: {e}");
            }
        }
    }

    let mut summary = format!(
        "n_frames={n} baseline_m={baseline:.6} vo_length_m={total_length:.2}\n\
         relative_pose_source_pnp={pnp_count} \
         relative_pose_source_pnp_fallback={pnp_fallback_count} \
         relative_pose_source_kabsch={kabsch_count} \
         relative_pose_source_kabsch_fallback={kabsch_fallback_count}\n",
    );
    if let Some((c0, c1, iters)) = ba_summary {
        summary.push_str(&format!(
            "stereo_ba_cost_initial={c0:.2} stereo_ba_cost_final={c1:.2} stereo_ba_iterations={iters}\n",
        ));
    }
    if let Some(rep) = &ba_gnc_summary {
        summary.push_str(&format!(
            "stereo_ba_gnc_inlier_scale_px={:.4} stereo_ba_gnc_inliers={} \
             stereo_ba_gnc_outliers={} stereo_ba_gnc_observations={}\n",
            rep.inlier_scale, rep.inliers, rep.outliers, rep.observations,
        ));
        if rep.injected_total > 0 {
            let clean = rep.observations.saturating_sub(rep.injected_total);
            summary.push_str(&format!(
                "stereo_ba_gnc_injected={} stereo_ba_gnc_injected_rejected={} \
                 stereo_ba_gnc_clean_rejected={} stereo_ba_gnc_clean_total={}\n",
                rep.injected_total, rep.injected_rejected, rep.clean_rejected, clean,
            ));
        }
    }

    // Optional ATE evaluation against KITTI ground truth. Both estimated
    // and reference trajectories start at the world origin (frame 0 =
    // identity), so `TrajectoryAlignment::None` is the right choice — no
    // Procrustes alignment is needed.
    if let Some(gt_path) = &args.gt_poses {
        let gt_full = PoseTrajectory::read_kitti_poses(gt_path)?;
        // Subsample GT to match the demo's per-frame indexing. The fetched
        // image subset already had `--stride gt_original_stride` applied,
        // and the demo additionally takes every `frame_stride`-th frame
        // from that. `start_frame` skips into the fetched image subset, so
        // the effective GT row for our frame `i` is at original index
        // `(start_frame + i * frame_stride) * gt_original_stride`.
        let effective_stride = args.frame_stride.max(1) * args.gt_original_stride.max(1);
        let gt_start = args.start_frame * args.gt_original_stride.max(1);
        let mut gt_traj = PoseTrajectory::new();
        for i in 0..n {
            let gt_idx = gt_start + i * effective_stride;
            if let Some(sample) = gt_full.samples().get(gt_idx) {
                gt_traj.push_sample(TrajectorySample {
                    frame_id: i as u64,
                    pose: sample.pose.clone(),
                    state: TrackingState::Tracking,
                    event: TrackingEvent::Tracked,
                    inlier_count: 0,
                    inlier_ratio: 0.0,
                    reprojection_error: None,
                });
            }
        }
        let gt_centers: Vec<nalgebra::Point3<f64>> = gt_traj
            .samples()
            .iter()
            .map(|s| s.camera_center_world())
            .collect();
        let gt_length: f64 = gt_centers.windows(2).map(|w| (w[1] - w[0]).norm()).sum();

        let vo_summary = vo_traj
            .translation_error_summary_against_with_alignment(&gt_traj, TrajectoryAlignment::None);
        println!(
            "GT eval VO: matched={} mean={:.2} m rmse={:.2} m max={:.2} m gt_length={:.2} m",
            vo_summary.matched_pose_count,
            vo_summary.mean_translation_error.unwrap_or(f64::NAN),
            vo_summary.rmse_translation_error.unwrap_or(f64::NAN),
            vo_summary.max_translation_error.unwrap_or(f64::NAN),
            gt_length,
        );
        summary.push_str(&format!(
            "gt_length_m={gt_length:.2} \
             vo_ate_mean_m={:.4} vo_ate_rmse_m={:.4} vo_ate_max_m={:.4}\n",
            vo_summary.mean_translation_error.unwrap_or(f64::NAN),
            vo_summary.rmse_translation_error.unwrap_or(f64::NAN),
            vo_summary.max_translation_error.unwrap_or(f64::NAN),
        ));
        write_trajectory_csv(&args.out_dir.join("gt.csv"), &gt_centers)?;
        gt_traj.write_kitti_poses(args.out_dir.join("gt_poses.txt"))?;
        let rel_diag = write_relative_pose_errors_csv(
            &args.out_dir.join("relative_pose_errors.csv"),
            &vo_poses,
            &gt_traj,
        )?;
        println!(
            "relative pose diagnostics: pairs={} mean_t_mag_err={:.4} m max_t_mag_err={:.4} m \
             mean_abs_ty_err={:.4} m max_abs_ty_err={:.4} m \
             mean_rot_err={:.4} deg max_rot_err={:.4} deg",
            rel_diag.pair_count,
            rel_diag.mean_translation_magnitude_error_m,
            rel_diag.max_translation_magnitude_error_m,
            rel_diag.mean_abs_ty_error_m,
            rel_diag.max_abs_ty_error_m,
            rel_diag.mean_rotation_error_deg,
            rel_diag.max_rotation_error_deg,
        );
        summary.push_str(&format!(
            "relative_pose_pairs={} relative_pose_mean_t_mag_err_m={:.4} \
             relative_pose_max_t_mag_err_m={:.4} relative_pose_mean_abs_ty_err_m={:.4} \
             relative_pose_max_abs_ty_err_m={:.4} relative_pose_mean_rot_err_deg={:.4} \
             relative_pose_max_rot_err_deg={:.4}\n",
            rel_diag.pair_count,
            rel_diag.mean_translation_magnitude_error_m,
            rel_diag.max_translation_magnitude_error_m,
            rel_diag.mean_abs_ty_error_m,
            rel_diag.max_abs_ty_error_m,
            rel_diag.mean_rotation_error_deg,
            rel_diag.max_rotation_error_deg,
        ));

        if let Some(refined) = &ba_poses {
            let ba_traj = build_pose_trajectory(refined);
            let ba_summary = ba_traj.translation_error_summary_against_with_alignment(
                &gt_traj,
                TrajectoryAlignment::None,
            );
            println!(
                "GT eval BA: matched={} mean={:.2} m rmse={:.2} m max={:.2} m",
                ba_summary.matched_pose_count,
                ba_summary.mean_translation_error.unwrap_or(f64::NAN),
                ba_summary.rmse_translation_error.unwrap_or(f64::NAN),
                ba_summary.max_translation_error.unwrap_or(f64::NAN),
            );
            summary.push_str(&format!(
                "ba_ate_mean_m={:.4} ba_ate_rmse_m={:.4} ba_ate_max_m={:.4}\n",
                ba_summary.mean_translation_error.unwrap_or(f64::NAN),
                ba_summary.rmse_translation_error.unwrap_or(f64::NAN),
                ba_summary.max_translation_error.unwrap_or(f64::NAN),
            ));
        }

        // Synthetic loop-closure + SE(3) PGO. We try the real essential-
        // matrix verifier first on the cross-frame descriptor matches —
        // even if it rejects (likely on a non-looping window), printing
        // the verdict shows the verifier is wired up. When it accepts,
        // we ALSO run cross-frame stereo Kabsch RANSAC to recover the
        // metric magnitude that the essential pose is missing, and use
        // that as the loop edge. Otherwise the demo falls back to a
        // GT-derived synthetic edge (the previous behaviour) so the
        // PGO numbers stay reproducible on test data without a real
        // physical loop.
        if args.run_synthetic_loop_closure && n >= 3 {
            let seed_poses: &[Pose] = ba_poses.as_deref().unwrap_or(&vo_poses);
            let mut graph = PoseGraph::new();
            for (id, pose) in seed_poses.iter().enumerate() {
                graph.add_pose(id as u64, pose.clone());
            }
            graph.anchor(0);
            for i in 0..(n - 1) {
                let measurement = relative_world_to_camera(&seed_poses[i], &seed_poses[i + 1]);
                graph.add_sequential_edge(i as u64, (i + 1) as u64, measurement);
            }

            // 1. Loop *detection* path. Run the pairwise scanner over every
            //    keyframe pair `(i, j)` with a sufficient frame-id gap; each
            //    pair gets brute-force descriptor matched and handed to
            //    `EssentialMatrixLoopClosureVerifier`. The scanner only emits
            //    pairs the verifier accepts, so we can pick the strongest
            //    candidate as the loop edge — no more hard-coded `(0, n-1)`
            //    probe. On windows without a physical loop the scanner
            //    correctly returns 0 candidates and the demo falls back to
            //    the GT-derived edge.
            let scanner_views: Vec<PairwiseKeyframeView> = (0..n)
                .map(|i| PairwiseKeyframeView {
                    frame_id: i as u64,
                    keypoints: &left_features[i].keypoints,
                    descriptors: &left_features[i].descriptors,
                })
                .collect();
            let scanner_matcher = BruteForceMatcher { ratio: Some(0.85) };
            let scanner_verifier = EssentialMatrixLoopClosureVerifier::default();
            // `min_keyframe_id_gap` is intentionally below half the window so
            // short demos still exercise the scanner; production callers
            // pick a gap commensurate with their local-mapping window.
            let scan_gap = std::cmp::min(20, (n / 3).max(1)) as u64;
            let scanner_cfg = PairwiseLoopClosureScannerConfig {
                min_keyframe_id_gap: scan_gap,
                min_matches: 30,
            };
            let scanner_candidates = scan_pairwise_loop_closures(
                &scanner_views,
                &scanner_matcher,
                &scanner_verifier,
                &camera,
                &scanner_cfg,
            );
            let strongest: Option<&LoopClosureCandidate> =
                scanner_candidates.iter().max_by(|a, b| {
                    a.score
                        .partial_cmp(&b.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            println!(
                "loop scanner: candidates={} (gap≥{}, n={})",
                scanner_candidates.len(),
                scan_gap,
                n,
            );
            if let Some(cand) = strongest {
                let v = cand
                    .verification
                    .as_ref()
                    .expect("scanner populates verification on accepted pairs");
                println!(
                    "loop scanner strongest: ({}, {}) inliers={} ratio={:.3} mean_sampson={:.4}",
                    cand.matched_keyframe_id,
                    cand.query_frame_id,
                    v.inlier_count,
                    v.inlier_ratio,
                    v.mean_sampson_error,
                );
            }

            // 2. If a candidate exists, recover a metric loop edge by
            //    running cross-frame stereo Kabsch RANSAC over the
            //    descriptor-match subset that has stereo triangulations at
            //    both ends. Essential RANSAC inside the verifier
            //    establishes geometric plausibility; Kabsch supplies the
            //    metric translation magnitude that essential's relative
            //    pose is missing. We re-match descriptors here because the
            //    scanner discards matches after verification — cheap
            //    relative to the per-pair RANSAC cost.
            let mut metric_loop_edge: Option<(
                visloc_rs::core::geometry::SE3,
                &'static str,
                u64,
                u64,
            )> = None;
            let mut chosen_inlier_count = 0usize;
            let mut chosen_inlier_ratio = 0.0_f64;
            let mut chosen_sampson = 0.0_f64;
            let mut chosen_score = 0.0_f64;
            if let Some(cand) = strongest {
                let from_idx = cand.matched_keyframe_id as usize;
                let to_idx = cand.query_frame_id as usize;
                let cross_matches = scanner_matcher.match_descriptors(
                    &left_features[from_idx].descriptors,
                    &left_features[to_idx].descriptors,
                );
                let stereo_a_lookup: std::collections::HashMap<
                    usize,
                    &visloc_rs::vision::stereo_vo::StereoFeature,
                > = stereo_per_frame[from_idx]
                    .iter()
                    .map(|f| (f.left_index, f))
                    .collect();
                let stereo_b_lookup: std::collections::HashMap<
                    usize,
                    &visloc_rs::vision::stereo_vo::StereoFeature,
                > = stereo_per_frame[to_idx]
                    .iter()
                    .map(|f| (f.left_index, f))
                    .collect();
                let mut pair_corrs: Vec<StereoPairCorrespondence> = Vec::new();
                for m in &cross_matches {
                    if let (Some(sa), Some(sb)) = (
                        stereo_a_lookup.get(&m.query_index),
                        stereo_b_lookup.get(&m.train_index),
                    ) {
                        pair_corrs.push(StereoPairCorrespondence {
                            a: sa.point_cam,
                            b: sb.point_cam,
                            confidence: m.confidence,
                        });
                    }
                }
                let kabsch_cfg = KabschRansacConfig {
                    iterations: 4000,
                    inlier_threshold_m: 2.0,
                    min_inliers: 12,
                    max_depth_m: 30.0,
                    seed: 11,
                };
                if let Some(report) = estimate_relative_pose_kabsch_ransac(&pair_corrs, &kabsch_cfg)
                {
                    let v = cand.verification.as_ref().unwrap();
                    println!(
                        "loop edge (kabsch): pair=({}, {}) pair_corrs={} inliers={} \
                         mean_residual={:.3} m t={:.2} m",
                        from_idx,
                        to_idx,
                        pair_corrs.len(),
                        report.inliers.len(),
                        report.mean_residual_m,
                        report.relative_pose.world_to_camera.translation.norm(),
                    );
                    chosen_inlier_count = v.inlier_count;
                    chosen_inlier_ratio = v.inlier_ratio;
                    chosen_sampson = v.mean_sampson_error;
                    chosen_score = v.score;
                    metric_loop_edge = Some((
                        report.relative_pose.world_to_camera,
                        "scanner",
                        cand.matched_keyframe_id,
                        cand.query_frame_id,
                    ));
                } else {
                    println!(
                        "loop edge (kabsch): rejected ({} cross-stereo correspondences for \
                         pair ({}, {}))",
                        pair_corrs.len(),
                        from_idx,
                        to_idx,
                    );
                }
            }

            // 3. Fall back to GT-derived edge between the first and last
            //    keyframe when the scanner found nothing or Kabsch
            //    rejected — keeps the demo's PGO numbers reproducible on
            //    test windows that don't actually loop. We also run the
            //    essential verifier on the (0, n-1) pair purely as a
            //    diagnostic so the GT-fallback edge can borrow its
            //    inlier-count for the PGO edge weight (without it the
            //    fallback weight collapses to 1.0 and the loop edge stops
            //    having pull commensurate with the chain).
            let gt_samples = gt_traj.samples();
            if metric_loop_edge.is_none() && gt_samples.len() == n {
                let cross_matcher = BruteForceMatcher { ratio: Some(0.85) };
                let cross_matches = cross_matcher.match_descriptors(
                    &left_features[0].descriptors,
                    &left_features[n - 1].descriptors,
                );
                let two_view: Vec<TwoViewCorrespondence> = cross_matches
                    .iter()
                    .map(|m| TwoViewCorrespondence {
                        previous_xy: left_features[0].keypoints[m.query_index],
                        current_xy: left_features[n - 1].keypoints[m.train_index],
                    })
                    .collect();
                let probe_verifier = EssentialMatrixLoopClosureVerifier::default();
                let probe = probe_verifier.verify(&two_view, &camera);
                println!(
                    "loop verifier probe (0, {}): verified={} inliers={} ratio={:.3} \
                     mean_sampson={:.4}",
                    n - 1,
                    probe.verified,
                    probe.inlier_count,
                    probe.inlier_ratio,
                    probe.mean_sampson_error,
                );
                chosen_inlier_count = probe.inlier_count;
                chosen_inlier_ratio = probe.inlier_ratio;
                chosen_sampson = probe.mean_sampson_error;
                chosen_score = probe.score;
                let loop_measurement =
                    relative_world_to_camera(&gt_samples[0].pose, &gt_samples[n - 1].pose);
                metric_loop_edge = Some((loop_measurement, "synthetic-gt", 0, (n - 1) as u64));
            }

            if let Some((relative_pose, source, from_id, to_id)) = metric_loop_edge {
                println!("loop edge source: {source} (from={from_id}, to={to_id})");
                // Render the loop edge endpoints for the plotter. The
                // endpoints come from the seed (BA-refined or raw VO)
                // trajectory because that's what the PGO operates on
                // before correction; the dashed red overlay then sits on
                // top of the BA / VO line in the top-down view, making
                // the geometric pull of the loop constraint visible.
                let from_pose = &seed_poses[from_id as usize];
                let to_pose = &seed_poses[to_id as usize];
                let from_c = from_pose.camera_center_world();
                let to_c = to_pose.camera_center_world();
                write_loop_edges_csv(
                    &args.out_dir.join("loop_edges.csv"),
                    &[(from_id, to_id, source, from_c, to_c)],
                )?;
                graph.add_loop_closure_constraint(&LoopClosureConstraint {
                    from_keyframe_id: from_id,
                    to_keyframe_id: to_id,
                    relative_pose,
                    inlier_count: chosen_inlier_count.max(1),
                    inlier_ratio: chosen_inlier_ratio.max(1.0),
                    mean_sampson_error: chosen_sampson,
                    score: chosen_score.max(1.0),
                });
                let pgo_result = graph.optimize_se3_iterative(&PoseGraphSe3Config {
                    initial_lambda: Some(1e-4),
                    robust_kernel: RobustKernel::Huber { delta: 0.1 },
                    linear_solver: LinearSolver::Sparse,
                    ..PoseGraphSe3Config::default()
                });
                match pgo_result {
                    Ok(r) => {
                        let pgo_poses: Vec<Pose> =
                            (0..n as u64).map(|id| graph.poses[&id].clone()).collect();
                        let pgo_centers: Vec<nalgebra::Point3<f64>> =
                            pgo_poses.iter().map(|p| p.camera_center_world()).collect();
                        let pgo_traj = build_pose_trajectory(&pgo_poses);
                        let pgo_summary = pgo_traj
                            .translation_error_summary_against_with_alignment(
                                &gt_traj,
                                TrajectoryAlignment::None,
                            );
                        println!(
                            "PGO (synthetic loop): se3_cost {:.4} → {:.4} ({} iter, converged={}); \
                             ATE mean={:.2} m rmse={:.2} m max={:.2} m",
                            r.initial_cost,
                            r.final_cost,
                            r.iterations.len(),
                            r.converged,
                            pgo_summary.mean_translation_error.unwrap_or(f64::NAN),
                            pgo_summary.rmse_translation_error.unwrap_or(f64::NAN),
                            pgo_summary.max_translation_error.unwrap_or(f64::NAN),
                        );
                        summary.push_str(&format!(
                            "pgo_se3_cost_initial={:.4} pgo_se3_cost_final={:.4} \
                             pgo_iterations={} pgo_ate_mean_m={:.4} pgo_ate_rmse_m={:.4} \
                             pgo_ate_max_m={:.4}\n",
                            r.initial_cost,
                            r.final_cost,
                            r.iterations.len(),
                            pgo_summary.mean_translation_error.unwrap_or(f64::NAN),
                            pgo_summary.rmse_translation_error.unwrap_or(f64::NAN),
                            pgo_summary.max_translation_error.unwrap_or(f64::NAN),
                        ));
                        write_trajectory_csv(&args.out_dir.join("pgo.csv"), &pgo_centers)?;
                        pgo_traj.write_kitti_poses(args.out_dir.join("pgo_poses.txt"))?;
                    }
                    Err(e) => {
                        println!("PGO skipped: {e}");
                    }
                }
            } else {
                println!(
                    "PGO skipped: no metric loop edge available (verifier rejected and \
                     GT subsample length {} != n_frames {})",
                    gt_samples.len(),
                    n,
                );
            }
        }
    }

    // Optional COLMAP 3DGS bootstrap export. Uses the BA-refined poses
    // when stereo BA produced a refinement, else the raw VO poses. PGO
    // is intentionally NOT consumed here because it is gated behind a
    // synthetic loop-closure edge that isn't always available; callers
    // who want a PGO-aware export can re-run with the same vo_poses.txt
    // and `--no-stereo-ba` to take the raw VO path.
    let export_poses: &[Pose] = ba_poses.as_deref().unwrap_or(&vo_poses);
    if let Some(colmap_dir) = &args.colmap_export_dir {
        let prefix = args.colmap_image_prefix.clone();
        let suffix = args.colmap_image_suffix.clone();
        let summary_colmap = write_colmap_text_model_for_3dgs(
            colmap_dir,
            &camera,
            export_poses,
            &left_features,
            &stereo_per_frame,
            |idx| format!("{prefix}{idx:06}{suffix}"),
        )?;
        println!(
            "COLMAP 3DGS export: frames={} landmarks={} observations={} dir={}",
            summary_colmap.frame_count,
            summary_colmap.landmark_count,
            summary_colmap.observation_count,
            colmap_dir.display(),
        );
    }
    if let Some(colmap_dir) = &args.colmap_export_binary_dir {
        let prefix = args.colmap_image_prefix.clone();
        let suffix = args.colmap_image_suffix.clone();
        let summary_colmap = write_colmap_binary_model_for_3dgs(
            colmap_dir,
            &camera,
            export_poses,
            &left_features,
            &stereo_per_frame,
            |idx| format!("{prefix}{idx:06}{suffix}"),
        )?;
        println!(
            "COLMAP 3DGS binary export: frames={} landmarks={} observations={} dir={}",
            summary_colmap.frame_count,
            summary_colmap.landmark_count,
            summary_colmap.observation_count,
            colmap_dir.display(),
        );
    }

    fs::write(args.out_dir.join("summary.txt"), &summary)?;
    println!("wrote {}/vo.csv and summary.txt", args.out_dir.display());

    Ok(())
}

#[cfg(feature = "image-io")]
struct FrontendState {
    poses: Vec<Pose>,
    stereo_per_frame: Vec<Vec<StereoFeature>>,
    left_features: Vec<FeatureSet>,
    right_features: Vec<FeatureSet>,
    per_pair_translation_m: Vec<f64>,
    kabsch_inlier_counts: Vec<usize>,
    pair_diagnostics: Vec<StereoVoPairDiagnostics>,
}

#[cfg(feature = "image-io")]
#[allow(clippy::too_many_arguments)]
fn run_stereo_vo_frontend(
    choice: StereoFrontendChoice,
    camera: &visloc_rs::Camera,
    baseline: f64,
    config: &StereoVoFrontendConfig,
    left_frames: &[&GrayscaleImage],
    right_frames: &[&GrayscaleImage],
    n: usize,
    deep_max_features: usize,
    deep_descriptor_clip: f32,
    deep_min_confidence: f32,
    deep_temperature: f32,
    progress_every: usize,
) -> Result<FrontendState, StereoVoError> {
    match choice {
        StereoFrontendChoice::Classical => {
            let mut frontend = StereoVoFrontend::new(camera.clone(), baseline, config.clone());
            run_frontend_loop(&mut frontend, left_frames, right_frames, n, progress_every)?;
            Ok(FrontendState {
                poses: frontend.poses,
                stereo_per_frame: frontend.stereo_per_frame,
                left_features: frontend.left_features,
                right_features: frontend.right_features,
                per_pair_translation_m: frontend.per_pair_translation_m,
                kabsch_inlier_counts: frontend.kabsch_inlier_counts,
                pair_diagnostics: frontend.pair_diagnostics,
            })
        }
        StereoFrontendChoice::Deep => {
            // KITTI is forward-driving with no in-plane camera rotation, so
            // axis-aligned HOG (`orient: false`) gives the best matcher
            // signal. The cap is configurable because long KITTI runs spend
            // most of their time in dense deep descriptor matching.
            let extractor = HogLikeFeatureExtractor::new(HogLikeFeatureConfig {
                max_features: deep_max_features,
                min_corner_score: 0.02,
                descriptor_clip: deep_descriptor_clip,
                orient: false,
            });
            let matcher = MutualSoftmaxMatcher::new(MutualSoftmaxConfig {
                temperature: deep_temperature,
                min_confidence: deep_min_confidence,
                emit_ratio_metadata: false,
            });
            let mut frontend = StereoVoFrontend::new_with(
                camera.clone(),
                baseline,
                config.clone(),
                extractor,
                matcher,
            );
            run_frontend_loop(&mut frontend, left_frames, right_frames, n, progress_every)?;
            Ok(FrontendState {
                poses: frontend.poses,
                stereo_per_frame: frontend.stereo_per_frame,
                left_features: frontend.left_features,
                right_features: frontend.right_features,
                per_pair_translation_m: frontend.per_pair_translation_m,
                kabsch_inlier_counts: frontend.kabsch_inlier_counts,
                pair_diagnostics: frontend.pair_diagnostics,
            })
        }
    }
}

#[cfg(feature = "image-io")]
fn run_frontend_loop<E, M>(
    frontend: &mut StereoVoFrontend<E, M>,
    left_frames: &[&GrayscaleImage],
    right_frames: &[&GrayscaleImage],
    n: usize,
    progress_every: usize,
) -> Result<(), StereoVoError>
where
    E: visloc_rs::FeatureExtractor<Image = GrayscaleImage>,
    E::Error: std::error::Error + Send + Sync + 'static,
    M: Matcher,
{
    let started = Instant::now();
    for i in 0..n {
        frontend.process_pair(left_frames[i], right_frames[i])?;
        if i > 0 && (i <= 3 || (progress_every > 0 && i % progress_every == 0) || i + 1 == n) {
            let translation_m = *frontend.per_pair_translation_m.last().unwrap();
            let inliers = *frontend.kabsch_inlier_counts.last().unwrap();
            let source = frontend
                .pair_diagnostics
                .last()
                .map(|d| relative_pose_source_label(d.source))
                .unwrap_or("unknown");
            println!(
                "  pair {}→{}: source={} pose_inliers={} t={:.2} m elapsed={:.1}s",
                i - 1,
                i,
                source,
                inliers,
                translation_m,
                started.elapsed().as_secs_f64(),
            );
        }
    }
    Ok(())
}

#[cfg(feature = "image-io")]
fn adaptive_track_search_radius_px(per_pair_translation_m: &[f64]) -> f64 {
    if per_pair_translation_m.is_empty() {
        return 25.0;
    }
    let mut sorted = per_pair_translation_m.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = sorted[sorted.len() / 2];
    (12.0 + 2.0 * med).clamp(20.0, 35.0)
}

#[cfg(feature = "image-io")]
fn build_pose_trajectory(poses: &[Pose]) -> PoseTrajectory {
    let mut traj = PoseTrajectory::new();
    for (i, pose) in poses.iter().enumerate() {
        traj.push_sample(TrajectorySample {
            frame_id: i as u64,
            pose: pose.clone(),
            state: TrackingState::Tracking,
            event: TrackingEvent::Tracked,
            inlier_count: 0,
            inlier_ratio: 0.0,
            reprojection_error: None,
        });
    }
    traj
}

#[cfg(feature = "image-io")]
fn write_trajectory_csv(
    path: &Path,
    centers: &[nalgebra::Point3<f64>],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut text = String::from("id,x,y,z\n");
    for (i, c) in centers.iter().enumerate() {
        text.push_str(&format!("{i},{:.6},{:.6},{:.6}\n", c.x, c.y, c.z));
    }
    fs::write(path, text)?;
    Ok(())
}

#[cfg(feature = "image-io")]
fn write_pair_diagnostics_csv(
    path: &Path,
    diagnostics: &[StereoVoPairDiagnostics],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut text = String::from(
        "from_id,to_id,source,temporal_matches,temporal_row_gate_px,temporal_confidence_gate,pnp_correspondences,\
stereo_pair_correspondences,inliers,raw_translation_m,raw_rotation_deg,\
translation_m,rotation_deg,motion_scale_rescued,translation_direction_rescued,\
stereo_vertical_aligned,rotation_spike_rescued,rotation_vector_rescued,pnp_mean_reprojection_error_px,\
kabsch_mean_residual_m\n",
    );
    for row in diagnostics {
        text.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{},{}\n",
            row.from_frame,
            row.to_frame,
            relative_pose_source_label(row.source),
            row.temporal_match_count,
            optional_f64(row.temporal_row_gate_px),
            optional_f32(row.temporal_confidence_gate),
            row.pnp_correspondence_count,
            row.stereo_pair_correspondence_count,
            row.inlier_count,
            row.raw_translation_m,
            row.raw_rotation_deg,
            row.translation_m,
            row.rotation_deg,
            row.motion_scale_rescued,
            row.translation_direction_rescued,
            row.stereo_vertical_aligned,
            row.rotation_spike_rescued,
            row.rotation_vector_rescued,
            optional_f64(row.pnp_mean_reprojection_error_px),
            optional_f64(row.kabsch_mean_residual_m),
        ));
    }
    fs::write(path, text)?;
    Ok(())
}

#[cfg(feature = "image-io")]
#[derive(Debug, Clone, Copy)]
struct StereoGroundProfile {
    total_stereo_points: usize,
    profile_points: usize,
    median_z_m: f64,
    median_y_m: f64,
    slope_y_per_z: f64,
    intercept_y_m: f64,
    rms_y_m: f64,
}

#[cfg(feature = "image-io")]
fn write_stereo_ground_profile_csv(
    path: &Path,
    stereo_per_frame: &[Vec<StereoFeature>],
    left_features: &[FeatureSet],
    camera: &Camera,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut text = String::from(
        "frame_id,total_stereo_points,profile_points,median_z_m,median_y_m,\
slope_y_per_z,intercept_y_m,rms_y_m,pitch_deg,y_at_10m,y_at_20m,y_at_30m\n",
    );
    for (frame_id, stereo) in stereo_per_frame.iter().enumerate() {
        let Some(left) = left_features.get(frame_id) else {
            continue;
        };
        if let Some(profile) = estimate_stereo_ground_profile(stereo, left, camera) {
            let pitch_deg = profile.slope_y_per_z.atan().to_degrees();
            text.push_str(&format!(
                "{frame_id},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                profile.total_stereo_points,
                profile.profile_points,
                profile.median_z_m,
                profile.median_y_m,
                profile.slope_y_per_z,
                profile.intercept_y_m,
                profile.rms_y_m,
                pitch_deg,
                ground_profile_y_at(&profile, 10.0),
                ground_profile_y_at(&profile, 20.0),
                ground_profile_y_at(&profile, 30.0),
            ));
        } else {
            text.push_str(&format!("{frame_id},{},0,,,,,,,,,\n", stereo.len()));
        }
    }
    fs::write(path, text)?;
    Ok(())
}

#[cfg(feature = "image-io")]
fn estimate_stereo_ground_profile(
    stereo: &[StereoFeature],
    left: &FeatureSet,
    camera: &Camera,
) -> Option<StereoGroundProfile> {
    let min_row = camera.height as f64 * 0.55;
    let mut yz = Vec::new();
    for feature in stereo {
        let Some(kp) = left.keypoints.get(feature.left_index) else {
            continue;
        };
        let point = feature.point_cam;
        if kp.y < min_row
            || point.z < 3.0
            || point.z > 45.0
            || point.y < -1.0
            || point.y > 4.0
            || !point.coords.iter().all(|v| v.is_finite())
        {
            continue;
        }
        yz.push((point.y, point.z));
    }
    if yz.len() < 20 {
        return None;
    }
    let mean_y = yz.iter().map(|(y, _)| *y).sum::<f64>() / yz.len() as f64;
    let mean_z = yz.iter().map(|(_, z)| *z).sum::<f64>() / yz.len() as f64;
    let mut cov_yz = 0.0;
    let mut var_z = 0.0;
    for (y, z) in &yz {
        cov_yz += (z - mean_z) * (y - mean_y);
        var_z += (z - mean_z).powi(2);
    }
    if var_z <= 1.0e-9 {
        return None;
    }
    let slope_y_per_z = cov_yz / var_z;
    let intercept_y_m = mean_y - slope_y_per_z * mean_z;
    let rms_y_m = (yz
        .iter()
        .map(|(y, z)| {
            let residual = y - (slope_y_per_z * z + intercept_y_m);
            residual * residual
        })
        .sum::<f64>()
        / yz.len() as f64)
        .sqrt();
    let ys = yz.iter().map(|(y, _)| *y).collect::<Vec<_>>();
    let zs = yz.iter().map(|(_, z)| *z).collect::<Vec<_>>();
    Some(StereoGroundProfile {
        total_stereo_points: stereo.len(),
        profile_points: yz.len(),
        median_z_m: median_f64(&zs).unwrap_or(mean_z),
        median_y_m: median_f64(&ys).unwrap_or(mean_y),
        slope_y_per_z,
        intercept_y_m,
        rms_y_m,
    })
}

#[cfg(feature = "image-io")]
fn ground_profile_y_at(profile: &StereoGroundProfile, z_m: f64) -> f64 {
    profile.slope_y_per_z * z_m + profile.intercept_y_m
}

#[cfg(feature = "image-io")]
fn relative_pose_source_label(source: StereoRelativePoseSource) -> &'static str {
    match source {
        StereoRelativePoseSource::Pnp => "pnp",
        StereoRelativePoseSource::PnpFallback => "pnp_fallback",
        StereoRelativePoseSource::Kabsch => "kabsch",
        StereoRelativePoseSource::KabschFallback => "kabsch_fallback",
    }
}

#[cfg(feature = "image-io")]
fn relative_pose_source_counts(
    diagnostics: &[StereoVoPairDiagnostics],
) -> (usize, usize, usize, usize) {
    let mut pnp = 0;
    let mut pnp_fallback = 0;
    let mut kabsch = 0;
    let mut kabsch_fallback = 0;
    for row in diagnostics {
        match row.source {
            StereoRelativePoseSource::Pnp => pnp += 1,
            StereoRelativePoseSource::PnpFallback => pnp_fallback += 1,
            StereoRelativePoseSource::Kabsch => kabsch += 1,
            StereoRelativePoseSource::KabschFallback => kabsch_fallback += 1,
        }
    }
    (pnp, pnp_fallback, kabsch, kabsch_fallback)
}

#[cfg(feature = "image-io")]
fn optional_f64(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.6}"))
        .unwrap_or_else(|| String::from(""))
}

#[cfg(feature = "image-io")]
fn optional_f32(value: Option<f32>) -> String {
    value
        .map(|v| format!("{v:.6}"))
        .unwrap_or_else(|| String::from(""))
}

#[cfg(feature = "image-io")]
#[derive(Debug, Clone, Copy)]
struct RelativePoseDiagnosticsSummary {
    pair_count: usize,
    mean_translation_magnitude_error_m: f64,
    max_translation_magnitude_error_m: f64,
    mean_abs_ty_error_m: f64,
    max_abs_ty_error_m: f64,
    mean_rotation_error_deg: f64,
    max_rotation_error_deg: f64,
}

#[cfg(feature = "image-io")]
fn write_relative_pose_errors_csv(
    path: &Path,
    estimated: &[Pose],
    reference: &PoseTrajectory,
) -> Result<RelativePoseDiagnosticsSummary, Box<dyn std::error::Error>> {
    let n = estimated.len().min(reference.len());
    let reference_samples = reference.samples();
    let mut text = String::from(
        "from_id,to_id,estimated_translation_m,reference_translation_m,\
translation_magnitude_error_m,translation_vector_error_m,\
estimated_tx_m,estimated_ty_m,estimated_tz_m,\
reference_tx_m,reference_ty_m,reference_tz_m,\
translation_error_x_m,translation_error_y_m,translation_error_z_m,rotation_error_deg\n",
    );
    let mut sum_t = 0.0;
    let mut max_t = 0.0_f64;
    let mut sum_abs_ty = 0.0;
    let mut max_abs_ty = 0.0_f64;
    let mut sum_r = 0.0;
    let mut max_r = 0.0_f64;
    let pair_count = n.saturating_sub(1);

    for i in 0..pair_count {
        let estimated_relative = relative_world_to_camera(&estimated[i], &estimated[i + 1]);
        let reference_relative =
            relative_world_to_camera(&reference_samples[i].pose, &reference_samples[i + 1].pose);
        let estimated_t = estimated_relative.translation.norm();
        let reference_t = reference_relative.translation.norm();
        let translation_vector_error =
            estimated_relative.translation - reference_relative.translation;
        let translation_error = (estimated_t - reference_t).abs();
        let abs_ty_error = translation_vector_error.y.abs();
        let rotation_error = (estimated_relative.rotation.inverse() * reference_relative.rotation)
            .angle()
            .to_degrees();
        sum_t += translation_error;
        max_t = max_t.max(translation_error);
        sum_abs_ty += abs_ty_error;
        max_abs_ty = max_abs_ty.max(abs_ty_error);
        sum_r += rotation_error;
        max_r = max_r.max(rotation_error);
        text.push_str(&format!(
            "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            reference_samples[i].frame_id,
            reference_samples[i + 1].frame_id,
            estimated_t,
            reference_t,
            translation_error,
            translation_vector_error.norm(),
            estimated_relative.translation.x,
            estimated_relative.translation.y,
            estimated_relative.translation.z,
            reference_relative.translation.x,
            reference_relative.translation.y,
            reference_relative.translation.z,
            translation_vector_error.x,
            translation_vector_error.y,
            translation_vector_error.z,
            rotation_error,
        ));
    }
    fs::write(path, text)?;

    let denom = pair_count.max(1) as f64;
    Ok(RelativePoseDiagnosticsSummary {
        pair_count,
        mean_translation_magnitude_error_m: sum_t / denom,
        max_translation_magnitude_error_m: max_t,
        mean_abs_ty_error_m: sum_abs_ty / denom,
        max_abs_ty_error_m: max_abs_ty,
        mean_rotation_error_deg: sum_r / denom,
        max_rotation_error_deg: max_r,
    })
}

#[cfg(feature = "image-io")]
type LoopEdgeRow<'a> = (
    u64,
    u64,
    &'a str,
    nalgebra::Point3<f64>,
    nalgebra::Point3<f64>,
);

#[cfg(feature = "image-io")]
fn write_loop_edges_csv(
    path: &Path,
    edges: &[LoopEdgeRow<'_>],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut text = String::from("from_id,to_id,source,from_x,from_y,from_z,to_x,to_y,to_z\n");
    for (from_id, to_id, source, from_c, to_c) in edges {
        text.push_str(&format!(
            "{from_id},{to_id},{source},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            from_c.x, from_c.y, from_c.z, to_c.x, to_c.y, to_c.z,
        ));
    }
    fs::write(path, text)?;
    Ok(())
}

#[cfg(feature = "image-io")]
fn mean(xs: &[usize]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<usize>() as f64 / xs.len() as f64
    }
}

#[cfg(feature = "image-io")]
fn median(xs: &[usize]) -> usize {
    if xs.is_empty() {
        return 0;
    }
    let mut sorted = xs.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

#[cfg(feature = "image-io")]
fn median_f64(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    let mut sorted = xs
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(f64::total_cmp);
    Some(sorted[sorted.len() / 2])
}
