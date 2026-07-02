//! Metric stereo VO from precomputed external deep feature/match files.
//!
//! This is the Rust-side consumer for a Python SuperPoint/LightGlue pipeline.
//! It does not load images or run a neural model. Instead it reads:
//!
//! ```text
//! frame_000000_left_features.txt
//! frame_000000_right_features.txt
//! frame_000000_stereo_matches.txt
//! frame_000001_left_features.txt
//! frame_000001_right_features.txt
//! frame_000001_stereo_matches.txt
//! frame_000001_temporal_matches.txt  # previous left -> current left
//! ...
//! ```
//!
//! Feature rows use `X Y SCORE D0 D1 ...`.
//! Match rows use `QUERY_IDX TRAIN_IDX CONFIDENCE [DISTANCE]`.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use nalgebra::Vector3;
use visloc_rs::{
    close_loops_on_vo_trajectory, close_loops_on_vo_trajectory_with_globals,
    close_loops_on_vo_trajectory_with_globals_and_loop_matches,
    close_loops_on_vo_trajectory_with_loop_matches, parse_kitti_calibration_txt,
    parse_kitti_oxts_timestamps_txt, parse_stereo_vo_imu_samples_txt,
    read_external_deep_features_txt, read_external_deep_matches_txt, read_kitti_oxts_dir,
    reconstruct_stereo_vo_with_ba, refine_stereo_vo_with_ba, slice_imu_samples_for_keyframes,
    write_colmap_binary_model_for_3dgs, write_colmap_reconstruction_for_3dgs,
    write_colmap_text_model_for_3dgs, write_online_ba_imu_state_csv, BaConfig, Camera,
    DescriptorMatch, GravityPrior, KabschRansacConfig, LandmarkInit, LinearSolver,
    LoopCandidatePair, LoopCandidateVerificationDiagnostic, OnlineStereoVoBa,
    OnlineStereoVoBaConfig, PerPoseGravityObservation, PerPoseGravityPrior, Pose, PoseTrajectory,
    PositionPrior, PositionPriorObservation, RobustKernel, StereoDepthGate,
    StereoDepthGateDiagnostics, StereoRelativePoseMode, StereoVoBaConfig, StereoVoBaImuInput,
    StereoVoBaImuSample, StereoVoFrontend, StereoVoFrontendConfig, StereoVoPairDiagnostics,
    TrackingEvent, TrackingState, TrajectorySample, VoLoopClosureConfig,
};

type LoopMatchesByPair = HashMap<(usize, usize), Vec<DescriptorMatch>>;

#[cfg(all(feature = "onnx-inference", feature = "image-io"))]
use visloc_rs::io::images::read_common_image;
#[cfg(all(feature = "onnx-inference", feature = "image-io"))]
use visloc_rs::vision::features::deep::DeepFeatureExtractor;
#[cfg(all(feature = "onnx-inference", feature = "image-io"))]
use visloc_rs::vision::features::lightglue_onnx::LightGlueOnnxMatcher;
#[cfg(all(feature = "onnx-inference", feature = "image-io"))]
use visloc_rs::vision::features::superpoint_onnx::{
    OnnxBackend, SuperPointOnnxConfig, SuperPointOnnxExtractor,
};
#[cfg(all(feature = "onnx-inference", feature = "image-io"))]
use visloc_rs::vision::features::FeatureSet;

/// In-process deep front-end: SuperPoint + LightGlue run via ONNX Runtime per
/// frame, producing the same `FeatureSet` / `DescriptorMatch` data the
/// file-based path reads from `--features-dir`. The match confidence carried in
/// `DescriptorMatch::confidence` is the LightGlue score, so the same
/// `--min-stereo-confidence` / `--min-temporal-confidence` gates apply.
#[cfg(all(feature = "onnx-inference", feature = "image-io"))]
struct InProcessFrontend {
    superpoint: SuperPointOnnxExtractor,
    lightglue: LightGlueOnnxMatcher,
    left_dir: PathBuf,
    right_dir: PathBuf,
    // Previous left-frame SuperPoint keypoints + descriptors, for the temporal
    // (previous-left -> current-left) match.
    prev_left: Option<KeypointsWithDescriptors>,
}

#[cfg(all(feature = "onnx-inference", feature = "image-io"))]
type FrameInputs = (
    FeatureSet,
    FeatureSet,
    Vec<DescriptorMatch>,
    Option<Vec<DescriptorMatch>>,
);

/// Previous left-frame SuperPoint keypoints paired with their descriptors.
#[cfg(all(feature = "onnx-inference", feature = "image-io"))]
type KeypointsWithDescriptors = (Vec<nalgebra::Point2<f64>>, Vec<Vec<f32>>);

#[cfg(all(feature = "onnx-inference", feature = "image-io"))]
impl InProcessFrontend {
    fn new(args: &CliArgs) -> Result<Self, Box<dyn std::error::Error>> {
        let backend = if args.onnx_backend_cpu {
            OnnxBackend::Cpu
        } else {
            OnnxBackend::CudaThenCpu
        };
        let config = SuperPointOnnxConfig {
            max_keypoints: args.onnx_max_keypoints,
            ..Default::default()
        };
        let superpoint = SuperPointOnnxExtractor::load_from_path_with_backend(
            args.superpoint_model.as_ref().unwrap(),
            config,
            backend,
        )?;
        let lightglue = LightGlueOnnxMatcher::load_from_path_with_backend(
            args.lightglue_model.as_ref().unwrap(),
            backend,
        )?;
        let images_dir = args.images_dir.as_ref().unwrap();
        Ok(Self {
            superpoint,
            lightglue,
            left_dir: images_dir.join(&args.left_subdir),
            right_dir: images_dir.join(&args.right_subdir),
            prev_left: None,
        })
    }

    fn process_frame(
        &mut self,
        frame_id: usize,
    ) -> Result<FrameInputs, Box<dyn std::error::Error>> {
        let name = format!("{frame_id:06}.png");
        let left_img = read_common_image(self.left_dir.join(&name))?;
        let right_img = read_common_image(self.right_dir.join(&name))?;
        let left = self.superpoint.extract_deep(&left_img)?;
        let right = self.superpoint.extract_deep(&right_img)?;

        let stereo = self
            .lightglue
            .match_features(
                &left.keypoints,
                &left.descriptors,
                &right.keypoints,
                &right.descriptors,
            )?
            .into_iter()
            .map(to_descriptor_match)
            .collect::<Vec<_>>();

        let temporal = match self.prev_left.take() {
            None => None,
            Some((prev_kpts, prev_desc)) => Some(
                self.lightglue
                    .match_features(&prev_kpts, &prev_desc, &left.keypoints, &left.descriptors)?
                    .into_iter()
                    .map(to_descriptor_match)
                    .collect::<Vec<_>>(),
            ),
        };

        let left_features = FeatureSet::new(left.keypoints.clone(), left.descriptors.clone())?;
        let right_features = FeatureSet::new(right.keypoints, right.descriptors)?;
        self.prev_left = Some((left.keypoints, left.descriptors));
        Ok((left_features, right_features, stereo, temporal))
    }
}

/// Read precomputed per-frame global descriptors written by
/// `vpr_global_descriptor_demo`: one line per frame, each a whitespace-separated
/// list of float32 values (already L2-normalised).
fn load_global_descriptors(path: &Path) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut globals = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row: Result<Vec<f32>, _> = line.split_whitespace().map(|v| v.parse::<f32>()).collect();
        let row = row.map_err(|e| format!("{}:{}: {e}", path.display(), line_no + 1))?;
        globals.push(row);
    }
    Ok(globals)
}

#[cfg(all(feature = "onnx-inference", feature = "image-io"))]
fn to_descriptor_match(
    m: visloc_rs::vision::features::lightglue_onnx::LightGlueMatch,
) -> DescriptorMatch {
    DescriptorMatch {
        query_index: m.query_index,
        train_index: m.train_index,
        distance: 1.0 - m.score,
        second_best_distance: None,
        ratio: None,
        confidence: Some(m.score),
    }
}

#[derive(Debug)]
// Several fields are only read under the `onnx-inference` + `image-io` in-process
// path; the default file-based build does not touch them.
#[allow(dead_code)]
struct CliArgs {
    features_dir: PathBuf,
    // In-process ONNX deep front-end (SuperPoint + LightGlue) instead of the
    // pre-exported `--features-dir`. Requires building with
    // `--features "image-io onnx-cuda"` (or `onnx-inference` for CPU).
    in_process_onnx: bool,
    superpoint_model: Option<PathBuf>,
    lightglue_model: Option<PathBuf>,
    images_dir: Option<PathBuf>,
    left_subdir: String,
    right_subdir: String,
    onnx_backend_cpu: bool,
    onnx_max_keypoints: usize,
    out_dir: PathBuf,
    calib: Option<PathBuf>,
    projection_left: String,
    projection_right: String,
    frames: usize,
    width: u32,
    height: u32,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    baseline: f64,
    relative_pose_mode: StereoRelativePoseMode,
    pnp_reprojection_threshold_px: f64,
    min_pnp_inliers: usize,
    stereo_vertical_alignment: bool,
    stereo_vertical_alignment_min_pairs: usize,
    stereo_vertical_alignment_max_correction_m: f64,
    rescue_min_median_translation_m: Option<f64>,
    motion_scale_rescue_max_inlier_ratio: Option<f64>,
    min_depth_m: Option<f64>,
    ba_exclude_rescued_pairs: bool,
    min_stereo_confidence: Option<f32>,
    min_temporal_confidence: Option<f32>,
    enable_ba: bool,
    ba_min_track_length: usize,
    ba_max_initial_depth_m: f64,
    ba_max_iterations: usize,
    ba_huber_delta_px: f64,
    ba_max_seed_row_fraction: Option<f64>,
    ba_max_init_residual_px: Option<f64>,
    ba_min_temporal_confidence: Option<f32>,
    ba_min_track_count: Option<usize>,
    ba_landmark_init: LandmarkInit,
    ba_window_size: Option<usize>,
    online_ba: bool,
    online_ba_window: usize,
    online_ba_trigger_every: usize,
    online_ba_history: usize,
    final_global_ba: bool,
    final_global_ba_iterations: usize,
    loop_closure: bool,
    loop_min_frame_gap: usize,
    loop_min_path_length: Option<f64>,
    loop_min_similarity: f32,
    loop_vocab_k: usize,
    loop_max_candidates_per_frame: usize,
    loop_max_verifications: Option<usize>,
    loop_two_view_ba: bool,
    loop_edge_information: bool,
    loop_global_descriptor_file: Option<PathBuf>,
    loop_matches_dir: Option<PathBuf>,
    loop_pnp_essential_inlier_filter: bool,
    loop_pnp_confidence_weighted_sampling: bool,
    loop_min_inlier_ratio: Option<f64>,
    loop_min_inliers: Option<usize>,
    ba_gravity_prior_weight: Option<f64>,
    ba_per_pose_gravity_prior_observations: Option<PathBuf>,
    ba_per_pose_gravity_prior_weight: f64,
    ba_per_pose_gravity_prior_g_world: Vector3<f64>,
    ba_position_prior_poses: Option<PathBuf>,
    ba_position_prior_weights: Vector3<f64>,
    ba_position_prior_couple_rotation: bool,
    post_ba_position_projection_poses: Option<PathBuf>,
    post_ba_position_projection_axes: Vector3<f64>,
    post_ba_position_projection_blend: Vector3<f64>,
    post_ba_rotation_projection_poses: Option<PathBuf>,
    imu_windows_dir: Option<PathBuf>,
    kitti_oxts_dir: Option<PathBuf>,
    kitti_image_timestamps: Option<PathBuf>,
    imu_gravity: Vector3<f64>,
    imu_weight_position: f64,
    imu_weight_velocity: f64,
    imu_weight_rotation: f64,
    imu_bias_gyro_init: Vector3<f64>,
    imu_bias_acc_init: Vector3<f64>,
    imu_bias_random_walk_weight: Option<f64>,
    imu_fix_first_bias: bool,
    imu_fix_first_velocity: bool,
    colmap_export_dir: Option<PathBuf>,
    colmap_export_binary_dir: Option<PathBuf>,
    sfm_colmap_export_dir: Option<PathBuf>,
    colmap_image_prefix: String,
    colmap_image_suffix: String,
    online_ba_imu_csv: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    if args.frames < 2 {
        return Err("--frames must be at least 2".into());
    }
    fs::create_dir_all(&args.out_dir)?;

    let (fx, fy, cx, cy, baseline) = if let Some(calib_path) = &args.calib {
        kitti_stereo_calibration(calib_path, &args.projection_left, &args.projection_right)?
    } else {
        (args.fx, args.fy, args.cx, args.cy, args.baseline)
    };
    let camera = Camera::pinhole(0, args.width, args.height, fx, fy, cx, cy);
    let mut config = StereoVoFrontendConfig {
        kabsch: KabschRansacConfig {
            iterations: 4000,
            min_inliers: StereoVoFrontendConfig::default().kabsch.min_inliers,
            ..StereoVoFrontendConfig::default().kabsch
        },
        pnp_reprojection_threshold_px: args.pnp_reprojection_threshold_px,
        pnp_min_inliers: args.min_pnp_inliers,
        stereo_vertical_alignment: args.stereo_vertical_alignment,
        stereo_vertical_alignment_min_pairs: args.stereo_vertical_alignment_min_pairs,
        stereo_vertical_alignment_max_correction_m: args.stereo_vertical_alignment_max_correction_m,
        relative_pose_mode: args.relative_pose_mode,
        ..StereoVoFrontendConfig::default()
    };
    if let Some(min_median) = args.rescue_min_median_translation_m {
        config.translation_direction_rescue_min_median_translation_m = min_median;
        config.rotation_spike_rescue_min_median_translation_m = min_median;
        config.rotation_vector_rescue_min_median_translation_m = min_median;
    }
    if let Some(max_ratio) = args.motion_scale_rescue_max_inlier_ratio {
        config.motion_scale_rescue_max_pnp_inlier_ratio = max_ratio;
    }
    if let Some(min_depth) = args.min_depth_m {
        config.stereo.min_depth_m = min_depth;
        config.stereo.depth_gate = StereoDepthGate::fixed();
    }
    let config = config;
    if args.online_ba && args.enable_ba {
        return Err(
            "--online-ba and --enable-ba are mutually exclusive: online BA interleaves the same \
             refiner inside the VO loop, post-process BA runs after. Pick one."
                .into(),
        );
    }
    if args.imu_windows_dir.is_some() {
        if !args.enable_ba && !args.online_ba {
            return Err(
                "--imu-windows-dir requires --enable-ba (post-process BA path) or --online-ba \
                 (streaming sliding-window BA)"
                    .into(),
            );
        }
        if args.kitti_oxts_dir.is_some() {
            return Err(
                "--imu-windows-dir and --kitti-oxts-dir are mutually exclusive: both build \
                 `imu_input.windows`. Pick one source."
                    .into(),
            );
        }
    }
    if args.kitti_oxts_dir.is_some() {
        if !args.enable_ba && !args.online_ba {
            return Err(
                "--kitti-oxts-dir requires --enable-ba (post-process BA path) or --online-ba \
                 (streaming sliding-window BA)"
                    .into(),
            );
        }
        if args.kitti_image_timestamps.is_none() {
            return Err(
                "--kitti-oxts-dir requires --kitti-image-timestamps (KITTI raw \
                 image_NN/timestamps.txt) to supply the keyframe times"
                    .into(),
            );
        }
    }
    if args.online_ba_imu_csv.is_some() && !args.online_ba {
        return Err(
            "--online-ba-imu-csv requires --online-ba (the CSV captures the per-trigger IMU state \
             produced by streaming sliding-window BA)"
                .into(),
        );
    }
    if args.final_global_ba && args.enable_ba {
        return Err(
            "--final-global-ba and --enable-ba are both global one-shot passes: --enable-ba \
             already runs a single dense global BA. Use --final-global-ba together with \
             --online-ba (a streaming windowed sweep followed by one final dense global pass), \
             or use --enable-ba on its own."
                .into(),
        );
    }
    if args.online_ba && args.ba_position_prior_poses.is_some() {
        return Err(
            "--ba-position-prior-poses is currently wired for one-shot --enable-ba only".into(),
        );
    }
    if args.online_ba && args.post_ba_position_projection_poses.is_some() {
        return Err(
            "--post-ba-position-projection-poses is currently wired for one-shot --enable-ba only"
                .into(),
        );
    }
    if args.online_ba && args.post_ba_rotation_projection_poses.is_some() {
        return Err(
            "--post-ba-rotation-projection-poses is currently wired for one-shot --enable-ba only"
                .into(),
        );
    }

    // Load IMU windows once (used by either the streaming online BA path or
    // the post-process BA path below; the two paths are still mutually
    // exclusive via the --online-ba / --enable-ba gate above).
    let imu_windows: Option<Vec<Vec<StereoVoBaImuSample>>> =
        if let Some(dir) = &args.imu_windows_dir {
            Some(load_imu_windows(dir, args.frames)?)
        } else if let Some(oxts_dir) = &args.kitti_oxts_dir {
            let kf_path = args
                .kitti_image_timestamps
                .as_ref()
                .expect("--kitti-image-timestamps validated above");
            Some(load_imu_windows_from_kitti_oxts(
                oxts_dir,
                kf_path,
                args.frames,
            )?)
        } else {
            None
        };
    let build_imu_input = |windows: Vec<Vec<StereoVoBaImuSample>>| StereoVoBaImuInput {
        windows,
        gravity_world: args.imu_gravity,
        bias_gyro_init: args.imu_bias_gyro_init,
        bias_acc_init: args.imu_bias_acc_init,
        weight_position: args.imu_weight_position,
        weight_velocity: args.imu_weight_velocity,
        weight_rotation: args.imu_weight_rotation,
        bias_random_walk_weight: args.imu_bias_random_walk_weight,
        fix_first_bias: args.imu_fix_first_bias,
        fix_first_velocity: args.imu_fix_first_velocity,
    };
    if let Some(w) = &imu_windows {
        let total_samples: usize = w.iter().map(|x| x.len()).sum();
        println!(
            "IMU pre-integration: windows={} total_samples={} gravity=({:.3},{:.3},{:.3}) \
             weight_p={} weight_v={} weight_R={} bias_random_walk_weight={:?} \
             fix_first_bias={} fix_first_velocity={} streaming={}",
            w.len(),
            total_samples,
            args.imu_gravity.x,
            args.imu_gravity.y,
            args.imu_gravity.z,
            args.imu_weight_position,
            args.imu_weight_velocity,
            args.imu_weight_rotation,
            args.imu_bias_random_walk_weight,
            args.imu_fix_first_bias,
            args.imu_fix_first_velocity,
            args.online_ba,
        );
    }

    let inner_frontend = StereoVoFrontend::new(camera, baseline, config);
    // KITTI-style level-world gravity prior: gravity points down in world
    // (+y on KITTI's y-down convention), and a level camera observes the
    // same direction. When `--ba-gravity-prior-weight 0` (default off) the
    // prior is `None` and BA runs without it.
    let gravity_prior = args.ba_gravity_prior_weight.map(|w| GravityPrior {
        g_world: Vector3::new(0.0, 9.81, 0.0),
        g_camera_observed: Vector3::new(0.0, 9.81, 0.0),
        weight: w,
    });
    let position_prior = match &args.ba_position_prior_poses {
        Some(path) => Some(read_position_prior_from_kitti_poses(
            path,
            args.frames,
            args.ba_position_prior_weights,
            args.ba_position_prior_couple_rotation,
        )?),
        None => None,
    };
    let per_pose_gravity_prior = match &args.ba_per_pose_gravity_prior_observations {
        Some(path) => Some(read_per_pose_gravity_prior_observations(
            path,
            args.frames,
            args.ba_per_pose_gravity_prior_g_world,
            args.ba_per_pose_gravity_prior_weight,
        )?),
        None => None,
    };
    // Always wrap in OnlineStereoVoBa; `trigger_every_frames = 0` disables
    // auto-firing so the wrapper behaves as a transparent pass-through when
    // `--online-ba` is not set.
    let online_config = OnlineStereoVoBaConfig {
        trigger_every_frames: if args.online_ba {
            args.online_ba_trigger_every
        } else {
            0
        },
        window_size: args.online_ba_window,
        ba_config: StereoVoBaConfig {
            min_track_length: args.ba_min_track_length,
            max_initial_depth_m: args.ba_max_initial_depth_m,
            max_seed_row_fraction: args.ba_max_seed_row_fraction,
            max_init_residual_px: args.ba_max_init_residual_px,
            min_temporal_confidence: args.ba_min_temporal_confidence,
            min_track_count: args.ba_min_track_count,
            landmark_init: args.ba_landmark_init,
            // The OnlineStereoVoBa wrapper already slices the trailing
            // window; the refiner should treat that slice as a single
            // joint BA (no nested sub-windows), so leave `window_size`
            // at None inside the inner config.
            window_size: None,
            gravity_prior: gravity_prior.clone(),
            position_prior: None,
            per_pose_gravity_prior: per_pose_gravity_prior.clone(),
            // The wrapper-level `imu_input` (set below for `--online-ba`)
            // is sliced per trigger and injected as the inner refiner's
            // `imu_input`. Leave the inner config's `imu_input` as `None`.
            imu_input: None,
            fix_pose_prefix: 1,
            ba_config: BaConfig {
                max_iterations: args.ba_max_iterations,
                robust_kernel: RobustKernel::Huber {
                    delta: args.ba_huber_delta_px,
                },
                linear_solver: LinearSolver::Sparse,
                ..BaConfig::default()
            },
        },
        // Wrapper-level IMU input (sliced per trigger to align with the
        // trailing BA window). Only set when `--online-ba` is on, since
        // `--enable-ba` runs a one-shot post-process BA that consumes the
        // global IMU input directly through the inner config below.
        imu_input: if args.online_ba {
            imu_windows.clone().map(build_imu_input)
        } else {
            None
        },
        local_map_history: args.online_ba_history,
        exclude_rescued_pair_matches: args.ba_exclude_rescued_pairs,
    };
    if args.online_ba {
        println!(
            "online BA enabled: window_size={} trigger_every_frames={} huber_delta={}px \
             min_track_count={:?}",
            online_config.window_size,
            online_config.trigger_every_frames,
            args.ba_huber_delta_px,
            online_config.ba_config.min_track_count,
        );
    }
    if let Some(w) = args.ba_gravity_prior_weight {
        println!(
            "BA gravity prior enabled: weight={w} g_world=(0, 9.81, 0) \
             g_camera_observed=(0, 9.81, 0) [KITTI y-down level-world]"
        );
    }
    if let Some(path) = &args.ba_position_prior_poses {
        println!(
            "BA position prior enabled: poses={} axis_weights=({:.3},{:.3},{:.3})",
            path.display(),
            args.ba_position_prior_weights.x,
            args.ba_position_prior_weights.y,
            args.ba_position_prior_weights.z,
        );
        if !args.ba_position_prior_couple_rotation {
            println!("BA position prior rotation coupling disabled");
        }
    }
    if let Some(path) = &args.ba_per_pose_gravity_prior_observations {
        let n = per_pose_gravity_prior
            .as_ref()
            .map(|p| p.observations.len())
            .unwrap_or(0);
        println!(
            "BA per-pose gravity prior enabled: observations={} weight={} g_world=({:.3},{:.3},{:.3}) source={}",
            n,
            args.ba_per_pose_gravity_prior_weight,
            args.ba_per_pose_gravity_prior_g_world.x,
            args.ba_per_pose_gravity_prior_g_world.y,
            args.ba_per_pose_gravity_prior_g_world.z,
            path.display(),
        );
    }
    let mut online_runner = OnlineStereoVoBa::new(inner_frontend, online_config);

    // Retain per-pair temporal matches when BA is enabled so they can drive
    // multi-frame track building after the frame-by-frame VO completes.
    let mut all_temporal_matches: Vec<Vec<DescriptorMatch>> = Vec::new();

    #[cfg(all(feature = "onnx-inference", feature = "image-io"))]
    let mut inprocess = if args.in_process_onnx {
        Some(InProcessFrontend::new(&args)?)
    } else {
        None
    };
    #[cfg(not(all(feature = "onnx-inference", feature = "image-io")))]
    if args.in_process_onnx {
        return Err("--in-process-onnx requires building this example with \
                    --features \"image-io onnx-cuda\" (or \"image-io onnx-inference\" for CPU)"
            .into());
    }

    for frame_id in 0..args.frames {
        // Feature/match inputs come from either the in-process ONNX front-end
        // or the pre-exported `--features-dir`; both yield the same types.
        let left_features;
        let right_features;
        let stereo_matches;
        let temporal_matches;
        #[cfg(all(feature = "onnx-inference", feature = "image-io"))]
        let from_inprocess = inprocess.is_some();
        #[cfg(not(all(feature = "onnx-inference", feature = "image-io")))]
        let from_inprocess = false;
        if from_inprocess {
            #[cfg(all(feature = "onnx-inference", feature = "image-io"))]
            {
                let (lf, rf, sm, tm) = inprocess.as_mut().unwrap().process_frame(frame_id)?;
                left_features = lf;
                right_features = rf;
                stereo_matches = filter_matches_by_confidence(sm, args.min_stereo_confidence);
                temporal_matches =
                    tm.map(|m| filter_matches_by_confidence(m, args.min_temporal_confidence));
            }
            #[cfg(not(all(feature = "onnx-inference", feature = "image-io")))]
            {
                unreachable!();
            }
        } else {
            left_features = read_external_deep_features_txt(
                args.features_dir.join(left_features_name(frame_id)),
            )?
            .into_feature_set()?;
            right_features = read_external_deep_features_txt(
                args.features_dir.join(right_features_name(frame_id)),
            )?
            .into_feature_set()?;
            let sm = read_external_deep_matches_txt(
                args.features_dir.join(stereo_matches_name(frame_id)),
            )?
            .into_descriptor_matches();
            stereo_matches = filter_matches_by_confidence(sm, args.min_stereo_confidence);
            temporal_matches = if frame_id == 0 {
                None
            } else {
                let matches = read_external_deep_matches_txt(
                    args.features_dir.join(temporal_matches_name(frame_id)),
                )?
                .into_descriptor_matches();
                Some(filter_matches_by_confidence(
                    matches,
                    args.min_temporal_confidence,
                ))
            };
        }
        if args.enable_ba || args.final_global_ba || args.sfm_colmap_export_dir.is_some() {
            if let Some(ref tm) = temporal_matches {
                all_temporal_matches.push(tm.clone());
            }
        }

        online_runner.process_pair_with_matches(
            left_features,
            right_features,
            Some(&stereo_matches),
            temporal_matches.as_deref(),
        )?;

        if frame_id > 0 {
            let diagnostics = online_runner.frontend.pair_diagnostics.last().unwrap();
            println!(
                "pair {}->{} source={:?} temporal_matches={} stereo_pairs={} inliers={} t={:.3}m",
                diagnostics.from_frame,
                diagnostics.to_frame,
                diagnostics.source,
                diagnostics.temporal_match_count,
                diagnostics.stereo_pair_correspondence_count,
                diagnostics.inlier_count,
                diagnostics.translation_m,
            );
        }
    }

    if args.enable_ba {
        let imu_input = imu_windows.clone().map(build_imu_input);
        let ba_config = StereoVoBaConfig {
            min_track_length: args.ba_min_track_length,
            max_initial_depth_m: args.ba_max_initial_depth_m,
            max_seed_row_fraction: args.ba_max_seed_row_fraction,
            max_init_residual_px: args.ba_max_init_residual_px,
            min_temporal_confidence: args.ba_min_temporal_confidence,
            min_track_count: args.ba_min_track_count,
            landmark_init: args.ba_landmark_init,
            window_size: args.ba_window_size,
            gravity_prior: gravity_prior.clone(),
            position_prior: position_prior.clone(),
            per_pose_gravity_prior: per_pose_gravity_prior.clone(),
            imu_input,
            fix_pose_prefix: 1,
            ba_config: BaConfig {
                max_iterations: args.ba_max_iterations,
                robust_kernel: RobustKernel::Huber {
                    delta: args.ba_huber_delta_px,
                },
                linear_solver: LinearSolver::Sparse,
                ..BaConfig::default()
            },
        };
        println!(
            "running multi-frame BA refinement: min_track_length={}, max_initial_depth={}m, \
             max_iterations={}, huber_delta={}px",
            ba_config.min_track_length,
            ba_config.max_initial_depth_m,
            ba_config.ba_config.max_iterations,
            args.ba_huber_delta_px,
        );
        match refine_stereo_vo_with_ba(
            &online_runner.frontend.camera,
            online_runner.frontend.baseline,
            &online_runner.frontend.poses,
            &online_runner.frontend.left_features,
            &online_runner.frontend.right_features,
            &online_runner.frontend.stereo_per_frame,
            &all_temporal_matches,
            &ba_config,
        ) {
            Ok(refinement) => {
                println!(
                    "BA refined: tracks={} observations={} cost {:.6} -> {:.6} ({} iters, \
                     converged={})",
                    refinement.track_count,
                    refinement.observation_count,
                    refinement.ba_result.initial_cost,
                    refinement.ba_result.final_cost,
                    refinement.ba_result.iterations.len(),
                    refinement.ba_result.converged,
                );
                if let Some(imu) = &refinement.imu_refinement {
                    write_imu_state_csv(
                        &args.out_dir.join("ba_imu_state.csv"),
                        &imu.refined_velocities,
                        &imu.refined_bias_gyro,
                        &imu.refined_bias_acc,
                    )?;
                    if let Some(last) = imu.refined_velocities.last() {
                        println!(
                            "BA IMU state: last_velocity=({:.3},{:.3},{:.3}) m/s",
                            last.x, last.y, last.z,
                        );
                    }
                }
                online_runner.frontend.poses = refinement.refined_poses;
            }
            Err(err) => {
                eprintln!("BA refinement skipped: {err}");
            }
        }
    }

    if args.online_ba {
        let triggered = online_runner.trigger_history.len();
        let ok_triggers = online_runner
            .trigger_history
            .iter()
            .filter(|s| s.result.is_ok())
            .count();
        let total_obs: usize = online_runner
            .trigger_history
            .iter()
            .filter_map(|s| s.result.as_ref().ok())
            .map(|r| r.observation_count)
            .sum();
        let total_tracks: usize = online_runner
            .trigger_history
            .iter()
            .filter_map(|s| s.result.as_ref().ok())
            .map(|r| r.track_count)
            .sum();
        println!(
            "online BA summary: triggers={} ok={} aggregate_tracks={} aggregate_observations={}",
            triggered, ok_triggers, total_tracks, total_obs,
        );
        if let Some(csv_path) = &args.online_ba_imu_csv {
            let rows = write_online_ba_imu_state_csv(csv_path, &online_runner.trigger_history)?;
            println!(
                "online BA IMU state CSV: rows={} path={}",
                rows,
                csv_path.display(),
            );
        }
    }
    // Final dense global BA pass: after the streaming windowed sweep
    // (`--online-ba`) has produced a locally-consistent trajectory, run one
    // joint bundle adjustment over EVERY pose and landmark at once
    // (`window_size: None`). The windowed sweep keeps drift small frame-to-
    // frame but never couples distant frames; this global pass closes the
    // residual long-range error the per-window BA cannot see.
    if args.final_global_ba {
        let global_cfg = StereoVoBaConfig {
            min_track_length: args.ba_min_track_length,
            max_initial_depth_m: args.ba_max_initial_depth_m,
            max_seed_row_fraction: args.ba_max_seed_row_fraction,
            max_init_residual_px: args.ba_max_init_residual_px,
            min_temporal_confidence: args.ba_min_temporal_confidence,
            min_track_count: args.ba_min_track_count,
            landmark_init: args.ba_landmark_init,
            window_size: None,
            gravity_prior: gravity_prior.clone(),
            position_prior: position_prior.clone(),
            per_pose_gravity_prior: per_pose_gravity_prior.clone(),
            imu_input: None,
            fix_pose_prefix: 1,
            ba_config: BaConfig {
                max_iterations: args.final_global_ba_iterations,
                robust_kernel: RobustKernel::Huber {
                    delta: args.ba_huber_delta_px,
                },
                linear_solver: LinearSolver::Sparse,
                ..BaConfig::default()
            },
        };
        println!(
            "running FINAL GLOBAL BA pass: poses={} temporal_match_sets={} max_iterations={} \
             huber_delta={}px",
            online_runner.frontend.poses.len(),
            all_temporal_matches.len(),
            args.final_global_ba_iterations,
            args.ba_huber_delta_px,
        );
        match refine_stereo_vo_with_ba(
            &online_runner.frontend.camera,
            online_runner.frontend.baseline,
            &online_runner.frontend.poses,
            &online_runner.frontend.left_features,
            &online_runner.frontend.right_features,
            &online_runner.frontend.stereo_per_frame,
            &all_temporal_matches,
            &global_cfg,
        ) {
            Ok(refinement) => {
                println!(
                    "FINAL GLOBAL BA: tracks={} observations={} cost {:.6} -> {:.6} ({} iters, \
                     converged={})",
                    refinement.track_count,
                    refinement.observation_count,
                    refinement.ba_result.initial_cost,
                    refinement.ba_result.final_cost,
                    refinement.ba_result.iterations.len(),
                    refinement.ba_result.converged,
                );
                online_runner.frontend.poses = refinement.refined_poses;
            }
            Err(err) => {
                eprintln!("FINAL GLOBAL BA skipped: {err}");
            }
        }
    }
    // Loop-closure pose-graph optimization. Unlike BA (which deforms an open
    // trajectory toward a local reprojection minimum without ever coupling a
    // revisit to its earlier observation), this detects revisited places by
    // appearance, verifies them with PnP into metric relative-pose constraints,
    // and re-distributes the accumulated drift with a robust GNC SE(3) solve.
    if args.loop_closure {
        let mut verifier = VoLoopClosureConfig::default().verifier;
        if let Some(ratio) = args.loop_min_inlier_ratio {
            verifier.min_inlier_ratio = ratio;
        }
        if let Some(min_inliers) = args.loop_min_inliers {
            verifier.min_inliers = min_inliers;
        }
        let loop_cfg = VoLoopClosureConfig {
            min_frame_gap: args.loop_min_frame_gap,
            min_path_length: args.loop_min_path_length,
            min_similarity: args.loop_min_similarity,
            vocab_k: args.loop_vocab_k,
            max_candidates_per_frame: args.loop_max_candidates_per_frame,
            max_verifications: args.loop_max_verifications,
            refine_loops_two_view: args.loop_two_view_ba,
            loop_edge_information: args.loop_edge_information,
            pnp_essential_inlier_filter: args.loop_pnp_essential_inlier_filter,
            pnp_confidence_weighted_sampling: args.loop_pnp_confidence_weighted_sampling,
            verifier,
            ..VoLoopClosureConfig::default()
        };
        // Optional: drive retrieval with a *learned* global descriptor
        // (EigenPlaces / CosPlace) precomputed by `vpr_global_descriptor_demo`,
        // one L2-normalised vector per frame, instead of the built-in k-means
        // VLAD over local SuperPoint descriptors. The geometric verification and
        // PGO that follow are identical, so this is a clean VLAD-vs-learned A/B.
        let learned_globals = match &args.loop_global_descriptor_file {
            Some(path) => {
                let globals = load_global_descriptors(path)?;
                if globals.len() != online_runner.frontend.poses.len() {
                    return Err(format!(
                        "global descriptor count {} != pose count {} (file {})",
                        globals.len(),
                        online_runner.frontend.poses.len(),
                        path.display()
                    )
                    .into());
                }
                println!(
                    "loaded {} learned global descriptors (dim {}) from {}",
                    globals.len(),
                    globals.first().map(|g| g.len()).unwrap_or(0),
                    path.display(),
                );
                Some(globals)
            }
            None => None,
        };
        let retrieval_label = match &learned_globals {
            Some(_) => "learned-vpr".to_string(),
            None => format!("vlad(k={})", args.loop_vocab_k),
        };
        let loop_matches = match &args.loop_matches_dir {
            Some(dir) => {
                let matches = load_loop_matches_dir(dir)?;
                println!(
                    "loaded {} external loop-match files from {}",
                    matches.len(),
                    dir.display()
                );
                Some(matches)
            }
            None => None,
        };
        println!(
            "running LOOP-CLOSURE PGO: poses={} min_frame_gap={} min_similarity={:.2} \
             retrieval={} max_candidates_per_frame={}",
            online_runner.frontend.poses.len(),
            args.loop_min_frame_gap,
            args.loop_min_similarity,
            retrieval_label,
            args.loop_max_candidates_per_frame,
        );
        let loop_result = match (&learned_globals, &loop_matches) {
            (Some(globals), Some(matches)) => {
                close_loops_on_vo_trajectory_with_globals_and_loop_matches(
                    &online_runner.frontend.camera,
                    &online_runner.frontend.poses,
                    &online_runner.frontend.left_features,
                    &online_runner.frontend.stereo_per_frame,
                    globals,
                    matches,
                    &loop_cfg,
                )
            }
            (Some(globals), None) => close_loops_on_vo_trajectory_with_globals(
                &online_runner.frontend.camera,
                &online_runner.frontend.poses,
                &online_runner.frontend.left_features,
                &online_runner.frontend.stereo_per_frame,
                globals,
                &loop_cfg,
            ),
            (None, Some(matches)) => close_loops_on_vo_trajectory_with_loop_matches(
                &online_runner.frontend.camera,
                &online_runner.frontend.poses,
                &online_runner.frontend.left_features,
                &online_runner.frontend.stereo_per_frame,
                matches,
                &loop_cfg,
            ),
            (None, None) => close_loops_on_vo_trajectory(
                &online_runner.frontend.camera,
                &online_runner.frontend.poses,
                &online_runner.frontend.left_features,
                &online_runner.frontend.stereo_per_frame,
                &loop_cfg,
            ),
        };
        match loop_result {
            Ok(result) => {
                let candidate_csv = args.out_dir.join("loop_candidates.csv");
                write_loop_candidates_csv(
                    &candidate_csv,
                    &result.candidate_pairs,
                    &retrieval_label,
                )?;
                println!("wrote {}", candidate_csv.display());
                let verification_csv = args.out_dir.join("loop_candidate_verifications.csv");
                write_loop_candidate_verifications_csv(
                    &verification_csv,
                    &result.verification_diagnostics,
                    &retrieval_label,
                )?;
                println!("wrote {}", verification_csv.display());
                match &result.gnc {
                    Some(gnc) => println!(
                        "LOOP-CLOSURE PGO: candidates={} verified_loops={} cost {:.6} -> {:.6} \
                         ({} outer iters, converged={})",
                        result.candidate_count,
                        result.verified_count(),
                        gnc.initial_cost,
                        gnc.final_cost,
                        gnc.outer_iterations,
                        gnc.converged,
                    ),
                    None => println!(
                        "LOOP-CLOSURE PGO: candidates={} verified_loops=0 (no loop verified; \
                         trajectory unchanged)",
                        result.candidate_count,
                    ),
                }
                online_runner.frontend.poses = result.refined_poses;
            }
            Err(err) => {
                eprintln!("LOOP-CLOSURE PGO skipped: {err}");
            }
        }
    }
    if let Some(path) = &args.post_ba_position_projection_poses {
        let projected = project_pose_centers_from_kitti_poses(
            &mut online_runner.frontend.poses,
            path,
            args.post_ba_position_projection_axes,
            args.post_ba_position_projection_blend,
        )?;
        println!(
            "post-BA position projection: poses={} axes=({:.0},{:.0},{:.0}) \
             blend=({:.3},{:.3},{:.3}) projected={}",
            path.display(),
            args.post_ba_position_projection_axes.x,
            args.post_ba_position_projection_axes.y,
            args.post_ba_position_projection_axes.z,
            args.post_ba_position_projection_blend.x,
            args.post_ba_position_projection_blend.y,
            args.post_ba_position_projection_blend.z,
            projected,
        );
    }
    if let Some(path) = &args.post_ba_rotation_projection_poses {
        let projected =
            project_pose_rotations_from_kitti_poses(&mut online_runner.frontend.poses, path)?;
        println!(
            "post-BA rotation projection: poses={} projected={}",
            path.display(),
            projected,
        );
    }
    let frontend = &online_runner.frontend;
    let centers: Vec<_> = frontend
        .poses
        .iter()
        .map(Pose::camera_center_world)
        .collect();
    write_trajectory_csv(&args.out_dir.join("vo.csv"), &centers)?;
    build_pose_trajectory(&frontend.poses).write_kitti_poses(args.out_dir.join("vo_poses.txt"))?;
    write_pair_diagnostics_csv(
        &args.out_dir.join("frontend_pair_diagnostics.csv"),
        &frontend.pair_diagnostics,
    )?;
    write_depth_gate_diagnostics_csv(
        &args.out_dir.join("frontend_depth_gate_diagnostics.csv"),
        &frontend.stereo_depth_gate_diagnostics,
    )?;
    fs::write(
        args.out_dir.join("summary.txt"),
        format!(
            "frames={} pairs={} trajectory_length_m={:.6}\n\
             stereo_depth_gate={}\n\
             stereo_min_depth_m={:.6}\n",
            frontend.frame_count(),
            frontend.pair_diagnostics.len(),
            frontend.trajectory_length_m(),
            stereo_depth_gate_label(&frontend.config.stereo.depth_gate),
            frontend.config.stereo.min_depth_m,
        ),
    )?;

    if let Some(colmap_dir) = &args.colmap_export_dir {
        let prefix = args.colmap_image_prefix.clone();
        let suffix = args.colmap_image_suffix.clone();
        let summary = write_colmap_text_model_for_3dgs(
            colmap_dir,
            &frontend.camera,
            &frontend.poses,
            &frontend.left_features,
            &frontend.stereo_per_frame,
            |idx| format!("{prefix}{idx:06}{suffix}"),
        )?;
        println!(
            "COLMAP 3DGS export: frames={} landmarks={} observations={} dir={}",
            summary.frame_count,
            summary.landmark_count,
            summary.observation_count,
            colmap_dir.display(),
        );
    }

    if let Some(colmap_dir) = &args.colmap_export_binary_dir {
        let prefix = args.colmap_image_prefix.clone();
        let suffix = args.colmap_image_suffix.clone();
        let summary = write_colmap_binary_model_for_3dgs(
            colmap_dir,
            &frontend.camera,
            &frontend.poses,
            &frontend.left_features,
            &frontend.stereo_per_frame,
            |idx| format!("{prefix}{idx:06}{suffix}"),
        )?;
        println!(
            "COLMAP 3DGS binary export: frames={} landmarks={} observations={} dir={}",
            summary.frame_count,
            summary.landmark_count,
            summary.observation_count,
            colmap_dir.display(),
        );
    }

    // SfM-grade COLMAP export: build merged multi-view tracks from the temporal
    // matches, run one global bundle adjustment over every pose and landmark,
    // and write a COLMAP model whose POINT3D TRACK[] tails span every frame that
    // observed each point. This is the structure a per-frame stereo lift lacks
    // and the form a downstream 3DGS optimizer needs to converge crisply.
    if let Some(sfm_dir) = &args.sfm_colmap_export_dir {
        let sfm_config = StereoVoBaConfig {
            min_track_length: args.ba_min_track_length,
            max_initial_depth_m: args.ba_max_initial_depth_m,
            max_seed_row_fraction: args.ba_max_seed_row_fraction,
            max_init_residual_px: args.ba_max_init_residual_px,
            min_temporal_confidence: args.ba_min_temporal_confidence,
            min_track_count: args.ba_min_track_count,
            landmark_init: args.ba_landmark_init,
            window_size: None,
            gravity_prior: None,
            position_prior: None,
            per_pose_gravity_prior: None,
            imu_input: None,
            fix_pose_prefix: 1,
            ba_config: BaConfig {
                max_iterations: args.final_global_ba_iterations,
                robust_kernel: RobustKernel::Huber {
                    delta: args.ba_huber_delta_px,
                },
                linear_solver: LinearSolver::Sparse,
                ..BaConfig::default()
            },
        };
        println!(
            "running SfM reconstruction for COLMAP export: min_track_length={} max_iterations={}",
            sfm_config.min_track_length, sfm_config.ba_config.max_iterations,
        );
        match reconstruct_stereo_vo_with_ba(
            &online_runner.frontend.camera,
            online_runner.frontend.baseline,
            &online_runner.frontend.poses,
            &online_runner.frontend.left_features,
            &online_runner.frontend.right_features,
            &online_runner.frontend.stereo_per_frame,
            &all_temporal_matches,
            &sfm_config,
        ) {
            Ok(recon) => {
                println!(
                    "SfM reconstruction: tracks={} observations={} reproj_px {:.4} -> {:.4} \
                     (BA cost {:.4} -> {:.4}, {} iters, converged={})",
                    recon.landmarks.len(),
                    recon.observation_count,
                    recon.mean_reproj_before_px,
                    recon.mean_reproj_after_px,
                    recon.ba_result.initial_cost,
                    recon.ba_result.final_cost,
                    recon.ba_result.iterations.len(),
                    recon.ba_result.converged,
                );
                let landmarks: Vec<visloc_rs::io::colmap::ReconstructionLandmark> = recon
                    .landmarks
                    .iter()
                    .map(|l| (l.position, l.observations.clone()))
                    .collect();
                let prefix = args.colmap_image_prefix.clone();
                let suffix = args.colmap_image_suffix.clone();
                let summary = write_colmap_reconstruction_for_3dgs(
                    sfm_dir,
                    &online_runner.frontend.camera,
                    &recon.refined_poses,
                    &online_runner.frontend.left_features,
                    &landmarks,
                    |idx| format!("{prefix}{idx:06}{suffix}"),
                )?;
                println!(
                    "SfM COLMAP export: frames={} landmarks={} observations={} dir={}",
                    summary.frame_count,
                    summary.landmark_count,
                    summary.observation_count,
                    sfm_dir.display(),
                );
            }
            Err(err) => {
                eprintln!("SfM reconstruction skipped: {err}");
            }
        }
    }

    println!("wrote {}", args.out_dir.display());

    Ok(())
}

fn filter_matches_by_confidence(
    matches: Vec<visloc_rs::DescriptorMatch>,
    min_confidence: Option<f32>,
) -> Vec<visloc_rs::DescriptorMatch> {
    let Some(min_confidence) = min_confidence else {
        return matches;
    };
    matches
        .into_iter()
        .filter(|descriptor_match| {
            descriptor_match
                .confidence
                .is_some_and(|confidence| confidence.is_finite() && confidence >= min_confidence)
        })
        .collect()
}

/// Rectified stereo intrinsics recovered from a KITTI calib file:
/// `(fx, fy, cx, cy, baseline_m)`.
type KittiStereoCalibration = (f64, f64, f64, f64, f64);

fn kitti_stereo_calibration(
    calib_path: &Path,
    projection_left: &str,
    projection_right: &str,
) -> Result<KittiStereoCalibration, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(calib_path)?;
    let projections = parse_kitti_calibration_txt(&text)?;
    let left = projections
        .iter()
        .find(|projection| projection.label == projection_left)
        .ok_or_else(|| format!("calib missing {projection_left}"))?;
    let right = projections
        .iter()
        .find(|projection| projection.label == projection_right)
        .ok_or_else(|| format!("calib missing {projection_right}"))?;
    let baseline = right.stereo_baseline_from(left).ok_or_else(|| {
        format!(
            "calib pair {projection_left}<->{projection_right} did not yield a positive baseline"
        )
    })?;
    Ok((left.fx(), left.fy(), left.cx(), left.cy(), baseline))
}

fn build_pose_trajectory(poses: &[Pose]) -> PoseTrajectory {
    let mut trajectory = PoseTrajectory::new();
    for (frame_id, pose) in poses.iter().enumerate() {
        trajectory.push_sample(TrajectorySample {
            frame_id: frame_id as u64,
            pose: pose.clone(),
            state: TrackingState::Tracking,
            event: TrackingEvent::Tracked,
            inlier_count: 0,
            inlier_ratio: 0.0,
            reprojection_error: None,
        });
    }
    trajectory
}

/// Load per-keyframe gravity-in-camera-frame observations from a plain
/// text file. The format is whitespace-separated, one observation per
/// line, with lines beginning `#` and blank lines treated as comments:
///
/// ```text
/// # keyframe_id gx gy gz
/// 0 0.05 9.79 0.51
/// 1 0.05 9.79 0.52
/// ...
/// ```
///
/// `g_world` and `weight` are caller-supplied. Observations whose
/// `keyframe_id >= frames` are silently dropped (so a 1000-frame OXTS
/// export can be sliced by `--frames 260` without preprocessing). The
/// returned prior has no extra metadata beyond what the file provides;
/// this is intentional, the file format is the boundary between
/// "online sensor source" and "BA-side prior consumer".
fn read_per_pose_gravity_prior_observations(
    path: &Path,
    frames: usize,
    g_world: Vector3<f64>,
    weight: f64,
) -> Result<PerPoseGravityPrior, Box<dyn std::error::Error>> {
    if !weight.is_finite() || weight < 0.0 {
        return Err("per-pose gravity prior weight must be finite and non-negative".into());
    }
    if !g_world.x.is_finite() || !g_world.y.is_finite() || !g_world.z.is_finite() {
        return Err("per-pose gravity prior g_world components must all be finite".into());
    }
    let text =
        fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let mut prior = PerPoseGravityPrior::new(g_world, weight);
    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        // Accept either 4 (legacy) or 5 (with per-obs weight) tokens.
        if parts.len() != 4 && parts.len() != 5 {
            return Err(format!(
                "{}:{}: expected 4 or 5 whitespace-separated tokens (keyframe_id gx gy gz [weight]), got {}",
                path.display(),
                lineno + 1,
                parts.len()
            )
            .into());
        }
        let keyframe_id: u64 = parts[0].parse().map_err(|e| {
            format!(
                "{}:{}: failed to parse keyframe_id `{}`: {e}",
                path.display(),
                lineno + 1,
                parts[0]
            )
        })?;
        if (keyframe_id as usize) >= frames {
            continue;
        }
        let gx: f64 = parts[1]
            .parse()
            .map_err(|e| format!("{}:{}: gx parse error: {e}", path.display(), lineno + 1))?;
        let gy: f64 = parts[2]
            .parse()
            .map_err(|e| format!("{}:{}: gy parse error: {e}", path.display(), lineno + 1))?;
        let gz: f64 = parts[3]
            .parse()
            .map_err(|e| format!("{}:{}: gz parse error: {e}", path.display(), lineno + 1))?;
        if !gx.is_finite() || !gy.is_finite() || !gz.is_finite() {
            return Err(format!(
                "{}:{}: g_camera components must all be finite",
                path.display(),
                lineno + 1
            )
            .into());
        }
        let obs_weight: f64 = if parts.len() == 5 {
            let w: f64 = parts[4].parse().map_err(|e| {
                format!("{}:{}: weight parse error: {e}", path.display(), lineno + 1)
            })?;
            if !w.is_finite() || w < 0.0 {
                return Err(format!(
                    "{}:{}: per-obs weight must be finite and non-negative",
                    path.display(),
                    lineno + 1
                )
                .into());
            }
            w
        } else {
            1.0
        };
        prior.push(PerPoseGravityObservation {
            keyframe_id,
            g_camera_observed: Vector3::new(gx, gy, gz),
            weight: obs_weight,
        });
    }
    if prior.observations.is_empty() {
        return Err(format!(
            "{}: no per-pose gravity observations parsed within frames=0..{frames}",
            path.display()
        )
        .into());
    }
    Ok(prior)
}

fn read_position_prior_from_kitti_poses(
    path: &Path,
    frames: usize,
    axis_weights: Vector3<f64>,
    couple_rotation: bool,
) -> Result<PositionPrior, Box<dyn std::error::Error>> {
    if axis_weights.x < 0.0
        || axis_weights.y < 0.0
        || axis_weights.z < 0.0
        || !axis_weights.x.is_finite()
        || !axis_weights.y.is_finite()
        || !axis_weights.z.is_finite()
    {
        return Err("--ba-position-prior-weights must be finite and non-negative".into());
    }
    if axis_weights == Vector3::zeros() {
        return Err("--ba-position-prior-weights must enable at least one axis".into());
    }

    let trajectory = PoseTrajectory::read_kitti_poses(path)?;
    if trajectory.len() < frames {
        return Err(format!(
            "--ba-position-prior-poses has {} poses, but --frames requires {frames}",
            trajectory.len()
        )
        .into());
    }

    let mut prior = PositionPrior::new().with_rotation_coupling(couple_rotation);
    for sample in trajectory.samples().iter().take(frames) {
        prior.push(PositionPriorObservation {
            keyframe_id: sample.frame_id,
            camera_center_world: sample.camera_center_world(),
            axis_weights,
        });
    }
    Ok(prior)
}

fn project_pose_centers_from_kitti_poses(
    poses: &mut [Pose],
    path: &Path,
    axes: Vector3<f64>,
    blend: Vector3<f64>,
) -> Result<usize, Box<dyn std::error::Error>> {
    if axes.x < 0.0
        || axes.y < 0.0
        || axes.z < 0.0
        || !axes.x.is_finite()
        || !axes.y.is_finite()
        || !axes.z.is_finite()
    {
        return Err("--post-ba-position-projection-axes must be finite and non-negative".into());
    }
    if axes == Vector3::zeros() {
        return Err("--post-ba-position-projection-axes must enable at least one axis".into());
    }
    if blend.x < 0.0
        || blend.y < 0.0
        || blend.z < 0.0
        || !blend.x.is_finite()
        || !blend.y.is_finite()
        || !blend.z.is_finite()
    {
        return Err("--post-ba-position-projection-blend must be finite and non-negative".into());
    }

    let trajectory = PoseTrajectory::read_kitti_poses(path)?;
    if trajectory.len() < poses.len() {
        return Err(format!(
            "--post-ba-position-projection-poses has {} poses, but VO has {}",
            trajectory.len(),
            poses.len()
        )
        .into());
    }

    for (pose, sample) in poses.iter_mut().zip(trajectory.samples()) {
        let mut center = pose.camera_center_world();
        let target = sample.camera_center_world();
        if axes.x > 0.0 {
            center.x += blend.x * (target.x - center.x);
        }
        if axes.y > 0.0 {
            center.y += blend.y * (target.y - center.y);
        }
        if axes.z > 0.0 {
            center.z += blend.z * (target.z - center.z);
        }
        let rotation = pose.world_to_camera.rotation;
        pose.world_to_camera.translation = -(rotation.transform_vector(&center.coords));
    }
    Ok(poses.len())
}

fn project_pose_rotations_from_kitti_poses(
    poses: &mut [Pose],
    path: &Path,
) -> Result<usize, Box<dyn std::error::Error>> {
    let trajectory = PoseTrajectory::read_kitti_poses(path)?;
    if trajectory.len() < poses.len() {
        return Err(format!(
            "--post-ba-rotation-projection-poses has {} poses, but VO has {}",
            trajectory.len(),
            poses.len()
        )
        .into());
    }

    for (pose, sample) in poses.iter_mut().zip(trajectory.samples()) {
        let center = pose.camera_center_world();
        pose.world_to_camera.rotation = sample.pose.world_to_camera.rotation;
        pose.world_to_camera.translation = -(pose
            .world_to_camera
            .rotation
            .transform_vector(&center.coords));
    }
    Ok(poses.len())
}

fn parse_args() -> Result<CliArgs, Box<dyn std::error::Error>> {
    let mut features_dir: Option<PathBuf> = None;
    let mut in_process_onnx = false;
    let mut superpoint_model: Option<PathBuf> = None;
    let mut lightglue_model: Option<PathBuf> = None;
    let mut images_dir: Option<PathBuf> = None;
    let mut left_subdir = String::from("image_0");
    let mut right_subdir = String::from("image_1");
    let mut onnx_backend_cpu = false;
    let mut onnx_max_keypoints: usize = 1500;
    let mut out_dir = PathBuf::from("target/stereo_vo_external_deep_files");
    let mut calib: Option<PathBuf> = None;
    let mut projection_left = String::from("P0");
    let mut projection_right = String::from("P1");
    let mut frames: Option<usize> = None;
    let mut width: u32 = 1241;
    let mut height: u32 = 376;
    let mut fx: f64 = 718.856;
    let mut fy: f64 = 718.856;
    let mut cx: f64 = 607.193;
    let mut cy: f64 = 185.216;
    let mut baseline: f64 = 0.537150888;
    let mut relative_pose_mode = StereoRelativePoseMode::PnpThenKabsch;
    let mut pnp_reprojection_threshold_px =
        StereoVoFrontendConfig::default().pnp_reprojection_threshold_px;
    let mut min_pnp_inliers = StereoVoFrontendConfig::default().pnp_min_inliers;
    let mut stereo_vertical_alignment = false;
    let mut stereo_vertical_alignment_min_pairs =
        StereoVoFrontendConfig::default().stereo_vertical_alignment_min_pairs;
    let mut stereo_vertical_alignment_max_correction_m =
        StereoVoFrontendConfig::default().stereo_vertical_alignment_max_correction_m;
    let mut rescue_min_median_translation_m: Option<f64> = None;
    let mut motion_scale_rescue_max_inlier_ratio: Option<f64> = None;
    let mut min_depth_m: Option<f64> = None;
    let mut ba_exclude_rescued_pairs = false;
    let mut min_stereo_confidence: Option<f32> = Some(0.5);
    let mut min_temporal_confidence: Option<f32> = Some(0.5);
    let mut enable_ba: bool = false;
    let mut ba_min_track_length: usize = 3;
    let mut ba_max_initial_depth_m: f64 = 60.0;
    let mut ba_max_iterations: usize = 12;
    let mut ba_huber_delta_px: f64 = 3.0;
    let mut ba_max_seed_row_fraction: Option<f64> = None;
    let mut ba_max_init_residual_px: Option<f64> = None;
    let mut ba_min_temporal_confidence: Option<f32> = None;
    let mut ba_min_track_count: Option<usize> = None;
    let mut ba_landmark_init: LandmarkInit = LandmarkInit::StereoSingleFrame;
    let mut ba_window_size: Option<usize> = None;
    let mut online_ba: bool = false;
    let mut online_ba_window: usize = 30;
    let mut online_ba_trigger_every: usize = 10;
    let mut online_ba_history: usize = 0;
    let mut final_global_ba: bool = false;
    let mut final_global_ba_iterations: usize = 30;
    let mut loop_closure: bool = false;
    let mut loop_two_view_ba: bool = false;
    let mut loop_edge_information: bool = false;
    let mut loop_global_descriptor_file: Option<PathBuf> = None;
    let mut loop_matches_dir: Option<PathBuf> = None;
    let mut loop_pnp_essential_inlier_filter: bool = false;
    let mut loop_pnp_confidence_weighted_sampling: bool = false;
    let mut loop_min_inlier_ratio: Option<f64> = None;
    let mut loop_min_inliers: Option<usize> = None;
    let mut loop_min_frame_gap: usize = 50;
    let mut loop_min_path_length: Option<f64> = Some(5.0);
    let mut loop_min_similarity: f32 = 0.20;
    let mut loop_vocab_k: usize = 64;
    let mut loop_max_candidates_per_frame: usize = 3;
    let mut loop_max_verifications: Option<usize> = Some(400);
    let mut ba_gravity_prior_weight: Option<f64> = None;
    let mut ba_per_pose_gravity_prior_observations: Option<PathBuf> = None;
    let mut ba_per_pose_gravity_prior_weight: f64 = 1.0;
    let mut ba_per_pose_gravity_prior_g_world = Vector3::new(0.0, 9.81, 0.0);
    let mut ba_position_prior_poses: Option<PathBuf> = None;
    let mut ba_position_prior_weights = Vector3::<f64>::zeros();
    let mut ba_position_prior_couple_rotation = true;
    let mut post_ba_position_projection_poses: Option<PathBuf> = None;
    let mut post_ba_position_projection_axes = Vector3::<f64>::zeros();
    let mut post_ba_position_projection_blend = Vector3::new(1.0, 1.0, 1.0);
    let mut post_ba_rotation_projection_poses: Option<PathBuf> = None;
    let mut imu_windows_dir: Option<PathBuf> = None;
    let mut kitti_oxts_dir: Option<PathBuf> = None;
    let mut kitti_image_timestamps: Option<PathBuf> = None;
    let mut imu_gravity = Vector3::new(0.0, 9.81, 0.0);
    let mut imu_weight_position: f64 = 1.0;
    let mut imu_weight_velocity: f64 = 1.0;
    let mut imu_weight_rotation: f64 = 1.0;
    let mut imu_bias_gyro_init = Vector3::<f64>::zeros();
    let mut imu_bias_acc_init = Vector3::<f64>::zeros();
    let mut imu_bias_random_walk_weight: Option<f64> = None;
    let mut imu_fix_first_bias: bool = true;
    let mut imu_fix_first_velocity: bool = false;
    let mut colmap_export_dir: Option<PathBuf> = None;
    let mut colmap_export_binary_dir: Option<PathBuf> = None;
    let mut sfm_colmap_export_dir: Option<PathBuf> = None;
    let mut colmap_image_prefix: String = String::new();
    let mut colmap_image_suffix: String = String::from(".png");
    let mut online_ba_imu_csv: Option<PathBuf> = None;

    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--features-dir" => {
                features_dir = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--in-process-onnx" => {
                in_process_onnx = true;
                args.remove(i);
            }
            "--superpoint-model" => {
                superpoint_model = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--lightglue-model" => {
                lightglue_model = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--images-dir" => {
                images_dir = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--left-subdir" => {
                left_subdir = args.remove(i + 1);
                args.remove(i);
            }
            "--right-subdir" => {
                right_subdir = args.remove(i + 1);
                args.remove(i);
            }
            "--onnx-cpu" => {
                onnx_backend_cpu = true;
                args.remove(i);
            }
            "--onnx-max-keypoints" => {
                onnx_max_keypoints = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--out-dir" => {
                out_dir = PathBuf::from(args.remove(i + 1));
                args.remove(i);
            }
            "--calib" => {
                calib = Some(PathBuf::from(args.remove(i + 1)));
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
            "--frames" => {
                frames = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--width" => {
                width = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--height" => {
                height = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--fx" => {
                fx = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--fy" => {
                fy = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--cx" => {
                cx = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--cy" => {
                cy = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--baseline" => {
                baseline = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--relative-pose-mode" => {
                relative_pose_mode = parse_relative_pose_mode(&args.remove(i + 1))?;
                args.remove(i);
            }
            "--pnp-reprojection-threshold" => {
                pnp_reprojection_threshold_px = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--min-pnp-inliers" => {
                min_pnp_inliers = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--stereo-vertical-alignment" => {
                stereo_vertical_alignment = true;
                args.remove(i);
            }
            "--stereo-vertical-alignment-min-pairs" => {
                stereo_vertical_alignment_min_pairs = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--stereo-vertical-alignment-max-correction" => {
                stereo_vertical_alignment_max_correction_m = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--rescue-min-median-translation" => {
                rescue_min_median_translation_m = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--ba-exclude-rescued-pairs" => {
                ba_exclude_rescued_pairs = true;
                args.remove(i);
            }
            "--motion-scale-rescue-max-inlier-ratio" => {
                motion_scale_rescue_max_inlier_ratio = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--min-depth" => {
                let value = args.remove(i + 1).parse::<f64>()?;
                if !value.is_finite() || value <= 0.0 {
                    return Err(
                        format!("--min-depth must be finite and positive, got {value}").into(),
                    );
                }
                min_depth_m = Some(value);
                args.remove(i);
            }
            "--min-stereo-confidence" => {
                min_stereo_confidence = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--min-temporal-confidence" => {
                min_temporal_confidence = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--enable-ba" => {
                enable_ba = true;
                args.remove(i);
            }
            "--ba-min-track-length" => {
                ba_min_track_length = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--ba-max-initial-depth" => {
                ba_max_initial_depth_m = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--ba-max-iterations" => {
                ba_max_iterations = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--ba-huber-delta" => {
                ba_huber_delta_px = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--ba-max-seed-row-fraction" => {
                ba_max_seed_row_fraction = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--ba-max-init-residual" => {
                ba_max_init_residual_px = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--ba-min-temporal-confidence" => {
                ba_min_temporal_confidence = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--ba-min-track-count" => {
                ba_min_track_count = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--ba-landmark-init" => {
                let v = args.remove(i + 1);
                ba_landmark_init = match v.as_str() {
                    "single" | "stereo-single-frame" => LandmarkInit::StereoSingleFrame,
                    "dlt" | "multi-view-dlt" => LandmarkInit::MultiViewDlt,
                    other => return Err(format!("unknown --ba-landmark-init {other}").into()),
                };
                args.remove(i);
            }
            "--ba-window-size" => {
                ba_window_size = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--online-ba" => {
                online_ba = true;
                args.remove(i);
            }
            "--online-ba-window" => {
                online_ba_window = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--online-ba-trigger-every" => {
                online_ba_trigger_every = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--online-ba-history" => {
                online_ba_history = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--final-global-ba" => {
                final_global_ba = true;
                args.remove(i);
            }
            "--final-global-ba-iterations" => {
                final_global_ba_iterations = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--loop-closure" => {
                loop_closure = true;
                args.remove(i);
            }
            "--loop-two-view-ba" => {
                loop_two_view_ba = true;
                args.remove(i);
            }
            "--loop-edge-information" => {
                loop_edge_information = true;
                args.remove(i);
            }
            "--global-descriptor-file" => {
                args.remove(i);
                loop_global_descriptor_file = Some(PathBuf::from(args.remove(i)));
            }
            "--loop-matches-dir" => {
                args.remove(i);
                loop_matches_dir = Some(PathBuf::from(args.remove(i)));
            }
            "--loop-pnp-essential-inliers" => {
                loop_pnp_essential_inlier_filter = true;
                args.remove(i);
            }
            "--loop-pnp-confidence-weights" => {
                loop_pnp_confidence_weighted_sampling = true;
                args.remove(i);
            }
            "--loop-min-inlier-ratio" => {
                args.remove(i);
                loop_min_inlier_ratio = Some(args.remove(i).parse()?);
            }
            "--loop-min-inliers" => {
                args.remove(i);
                loop_min_inliers = Some(args.remove(i).parse()?);
            }
            "--loop-min-frame-gap" => {
                loop_min_frame_gap = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--loop-min-path-length" => {
                let v: f64 = args.remove(i + 1).parse()?;
                loop_min_path_length = if v > 0.0 { Some(v) } else { None };
                args.remove(i);
            }
            "--loop-min-similarity" => {
                loop_min_similarity = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--loop-vocab-k" => {
                loop_vocab_k = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--loop-max-candidates-per-frame" => {
                loop_max_candidates_per_frame = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--loop-max-verifications" => {
                let v: usize = args.remove(i + 1).parse()?;
                loop_max_verifications = if v == 0 { None } else { Some(v) };
                args.remove(i);
            }
            "--ba-gravity-prior-weight" => {
                let w: f64 = args.remove(i + 1).parse()?;
                ba_gravity_prior_weight = if w > 0.0 { Some(w) } else { None };
                args.remove(i);
            }
            "--ba-per-pose-gravity-prior-observations" => {
                ba_per_pose_gravity_prior_observations = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--ba-per-pose-gravity-prior-weight" => {
                let w: f64 = args.remove(i + 1).parse()?;
                if !w.is_finite() || w < 0.0 {
                    return Err(
                        "--ba-per-pose-gravity-prior-weight must be finite and non-negative".into(),
                    );
                }
                ba_per_pose_gravity_prior_weight = w;
                args.remove(i);
            }
            "--ba-per-pose-gravity-prior-g-world" => {
                ba_per_pose_gravity_prior_g_world = parse_vec3_csv(&args.remove(i + 1))?;
                args.remove(i);
            }
            "--ba-position-prior-poses" => {
                ba_position_prior_poses = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--ba-position-prior-weights" => {
                ba_position_prior_weights = parse_vec3_csv(&args.remove(i + 1))?;
                args.remove(i);
            }
            "--ba-position-prior-decouple-rotation" => {
                ba_position_prior_couple_rotation = false;
                args.remove(i);
            }
            "--post-ba-position-projection-poses" => {
                post_ba_position_projection_poses = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--post-ba-position-projection-axes" => {
                post_ba_position_projection_axes = parse_vec3_csv(&args.remove(i + 1))?;
                args.remove(i);
            }
            "--post-ba-position-projection-blend" => {
                post_ba_position_projection_blend = parse_vec3_csv(&args.remove(i + 1))?;
                args.remove(i);
            }
            "--post-ba-rotation-projection-poses" => {
                post_ba_rotation_projection_poses = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--imu-windows-dir" => {
                imu_windows_dir = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--kitti-oxts-dir" => {
                kitti_oxts_dir = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--kitti-image-timestamps" => {
                kitti_image_timestamps = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--imu-gravity" => {
                imu_gravity = parse_vec3_csv(&args.remove(i + 1))?;
                args.remove(i);
            }
            "--imu-weight-position" => {
                imu_weight_position = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--imu-weight-velocity" => {
                imu_weight_velocity = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--imu-weight-rotation" => {
                imu_weight_rotation = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--imu-bias-gyro-init" => {
                imu_bias_gyro_init = parse_vec3_csv(&args.remove(i + 1))?;
                args.remove(i);
            }
            "--imu-bias-acc-init" => {
                imu_bias_acc_init = parse_vec3_csv(&args.remove(i + 1))?;
                args.remove(i);
            }
            "--imu-bias-random-walk-weight" => {
                let w: f64 = args.remove(i + 1).parse()?;
                imu_bias_random_walk_weight = if w > 0.0 { Some(w) } else { None };
                args.remove(i);
            }
            "--imu-fix-first-bias" => {
                imu_fix_first_bias = parse_bool_arg(&args.remove(i + 1))?;
                args.remove(i);
            }
            "--imu-fix-first-velocity" => {
                imu_fix_first_velocity = parse_bool_arg(&args.remove(i + 1))?;
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
            "--sfm-colmap-out" => {
                sfm_colmap_export_dir = Some(PathBuf::from(args.remove(i + 1)));
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
            "--online-ba-imu-csv" => {
                online_ba_imu_csv = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    if in_process_onnx {
        if images_dir.is_none() {
            return Err("--in-process-onnx requires --images-dir".into());
        }
        if superpoint_model.is_none() || lightglue_model.is_none() {
            return Err(
                "--in-process-onnx requires --superpoint-model and --lightglue-model".into(),
            );
        }
    }
    let features_dir = if in_process_onnx {
        features_dir.unwrap_or_default()
    } else {
        features_dir.ok_or("--features-dir is required (or use --in-process-onnx)")?
    };

    Ok(CliArgs {
        features_dir,
        in_process_onnx,
        superpoint_model,
        lightglue_model,
        images_dir,
        left_subdir,
        right_subdir,
        onnx_backend_cpu,
        onnx_max_keypoints,
        out_dir,
        calib,
        projection_left,
        projection_right,
        frames: frames.ok_or("--frames is required")?,
        width,
        height,
        fx,
        fy,
        cx,
        cy,
        baseline,
        relative_pose_mode,
        pnp_reprojection_threshold_px,
        min_pnp_inliers,
        stereo_vertical_alignment,
        stereo_vertical_alignment_min_pairs,
        stereo_vertical_alignment_max_correction_m,
        rescue_min_median_translation_m,
        motion_scale_rescue_max_inlier_ratio,
        min_depth_m,
        ba_exclude_rescued_pairs,
        min_stereo_confidence,
        min_temporal_confidence,
        enable_ba,
        ba_min_track_length,
        ba_max_initial_depth_m,
        ba_max_iterations,
        ba_huber_delta_px,
        ba_max_seed_row_fraction,
        ba_max_init_residual_px,
        ba_min_temporal_confidence,
        ba_min_track_count,
        ba_landmark_init,
        ba_window_size,
        online_ba,
        online_ba_window,
        online_ba_trigger_every,
        online_ba_history,
        final_global_ba,
        final_global_ba_iterations,
        loop_closure,
        loop_two_view_ba,
        loop_edge_information,
        loop_global_descriptor_file,
        loop_matches_dir,
        loop_pnp_essential_inlier_filter,
        loop_pnp_confidence_weighted_sampling,
        loop_min_inlier_ratio,
        loop_min_inliers,
        loop_min_frame_gap,
        loop_min_path_length,
        loop_min_similarity,
        loop_vocab_k,
        loop_max_candidates_per_frame,
        loop_max_verifications,
        ba_gravity_prior_weight,
        ba_per_pose_gravity_prior_observations,
        ba_per_pose_gravity_prior_weight,
        ba_per_pose_gravity_prior_g_world,
        ba_position_prior_poses,
        ba_position_prior_weights,
        ba_position_prior_couple_rotation,
        post_ba_position_projection_poses,
        post_ba_position_projection_axes,
        post_ba_position_projection_blend,
        post_ba_rotation_projection_poses,
        imu_windows_dir,
        kitti_oxts_dir,
        kitti_image_timestamps,
        imu_gravity,
        imu_weight_position,
        imu_weight_velocity,
        imu_weight_rotation,
        imu_bias_gyro_init,
        imu_bias_acc_init,
        imu_bias_random_walk_weight,
        imu_fix_first_bias,
        imu_fix_first_velocity,
        colmap_export_dir,
        colmap_export_binary_dir,
        sfm_colmap_export_dir,
        colmap_image_prefix,
        colmap_image_suffix,
        online_ba_imu_csv,
    })
}

fn parse_vec3_csv(value: &str) -> Result<Vector3<f64>, Box<dyn std::error::Error>> {
    let parts: Vec<f64> = value
        .split(',')
        .map(|t| t.trim().parse::<f64>())
        .collect::<Result<_, _>>()?;
    if parts.len() != 3 {
        return Err(format!(
            "expected 3 comma-separated numbers, got {}: {value}",
            parts.len()
        )
        .into());
    }
    Ok(Vector3::new(parts[0], parts[1], parts[2]))
}

fn parse_bool_arg(value: &str) -> Result<bool, Box<dyn std::error::Error>> {
    match value {
        "1" | "true" | "on" | "yes" => Ok(true),
        "0" | "false" | "off" | "no" => Ok(false),
        other => Err(format!("expected boolean (true/false/on/off/1/0), got {other}").into()),
    }
}

fn print_usage() {
    eprintln!(
        "usage: cargo run --example stereo_vo_external_deep_files -- \
         --features-dir <dir> --frames <n> [--out-dir <dir>] \
         [--calib <calib.txt>] [--projection-left P0] [--projection-right P1] \
         [--width <px>] [--height <px>] [--fx <px>] [--fy <px>] \
         [--cx <px>] [--cy <px>] [--baseline <m>] \
         [--relative-pose-mode pnp|kabsch] \
         [--stereo-vertical-alignment] \
         [--stereo-vertical-alignment-min-pairs <n>] \
         [--stereo-vertical-alignment-max-correction <m>] \
         [--rescue-min-median-translation <m>]  lower the shared sustained-motion \
         gate of the weak-consensus rescue clamps (translation-direction / \
         rotation-spike / rotation-vector); the default 1.5 m/frame only arms \
         them at highway speed, 0.5 also covers low-speed urban driving where \
         a crossing vehicle can capture the PnP consensus \
         [--ba-exclude-rescued-pairs]  drop the temporal matches of rescued \
         pairs from online-BA track building, so the optimiser cannot re-impose \
         the rejected (dynamic-object) motion onto the clamped poses \
         [--motion-scale-rescue-max-inlier-ratio <r>]  only rescue the \
         translation magnitude when the PnP inlier ratio is at most <r>. The \
         default 1.05 treats every pose as weak, which lets a sustained \
         fast-then-decelerating stretch lock the translation to the stale \
         median forever (rescued values feed the history); 0.45 restricts the \
         rescue to genuinely weak consensus like the other rescues \
         [--min-depth <m>]  force the legacy fixed minimum stereo depth \
         (adaptive by default; this flag is for A/B and exact old-run replay) \
         [--min-stereo-confidence <0..1, default 0.5>] \
         [--min-temporal-confidence <0..1, default 0.5>] \
         [--ba-position-prior-poses <kitti_poses.txt>]  one-shot BA absolute \
         camera-centre prior source (requires --enable-ba; use generated OXTS \
         cam0 poses for sensor-prior experiments)\n \
         [--ba-position-prior-weights wx,wy,wz]  per-axis position-prior \
         weights, e.g. 0,100,100 for KITTI y+z grade/height\n \
         [--ba-position-prior-decouple-rotation]  apply the position prior \
         through translation-only Jacobians so it cannot pull pose rotation\n \
         [--post-ba-position-projection-poses <kitti_poses.txt>]  after one-shot \
         BA, replace selected camera-centre axes from this pose file\n \
         [--post-ba-position-projection-axes x,y,z]  non-zero entries select \
         axes to project, e.g. 0,1,1 for KITTI y+z\n \
         [--post-ba-position-projection-blend bx,by,bz]  per-axis blend toward \
         projected centre for selected axes (default 1,1,1; 0 keeps visual, \
         1 fully replaces)\n \
         [--post-ba-rotation-projection-poses <kitti_poses.txt>]  after one-shot \
         BA and optional position projection, replace camera orientation from \
         this pose file while preserving the current camera centre\n \
         [--ba-per-pose-gravity-prior-observations <file>]  text file with one \
         `keyframe_id gx gy gz` per line giving accelerometer-derived gravity-in-\
         camera-frame observations (online-friendly sensor prior; does NOT leak \
         GT poses the way absolute-position priors can)\n \
         [--ba-per-pose-gravity-prior-weight w]  scalar weight applied to each \
         observation (default 1.0)\n \
         [--ba-per-pose-gravity-prior-g-world gx,gy,gz]  world gravity direction \
         (default 0,9.81,0; KITTI y-down)\n \
         \n  IMU (requires --enable-ba or --online-ba; the two are mutually exclusive):\n \
         [--imu-windows-dir <dir>]   one frame_NNNNNN_imu.txt per inter-frame window, \
         each line `dt gx gy gz ax ay az` (gravity NOT pre-subtracted from accel)\n \
         [--kitti-oxts-dir <dir>]    KITTI raw <sequence>/oxts directory \
         (loader reads oxts/data/*.txt + oxts/timestamps.txt, then slices into \
         per-keyframe windows using --kitti-image-timestamps)\n \
         [--kitti-image-timestamps <path>]  KITTI raw image_NN/timestamps.txt file \
         supplying per-frame wall-clock times (first --frames rows used)\n \
         [--imu-gravity gx,gy,gz]    world gravity (default 0,9.81,0; KITTI y-down)\n \
         [--imu-weight-position w]   weight on position residual (default 1.0)\n \
         [--imu-weight-velocity w]   weight on velocity residual (default 1.0)\n \
         [--imu-weight-rotation w]   weight on rotation residual (default 1.0)\n \
         [--imu-bias-gyro-init x,y,z]  initial gyro bias (default 0,0,0)\n \
         [--imu-bias-acc-init x,y,z]   initial accel bias (default 0,0,0)\n \
         [--imu-bias-random-walk-weight w]   inter-keyframe bias tie weight (default off)\n \
         [--imu-fix-first-bias on|off]       gauge-fix first bias (default on)\n \
         [--imu-fix-first-velocity on|off]   pin first velocity (default off)\n \
         \n  Final dense global BA (chains after --online-ba):\n \
         [--final-global-ba]  after the streaming windowed BA sweep, run ONE \
         joint bundle adjustment over every pose+landmark (window_size=None) to \
         close residual long-range drift the per-window passes cannot see; \
         mutually exclusive with --enable-ba (which is already a global one-shot)\n \
         [--final-global-ba-iterations <n>]  max LM iterations for that pass \
         (default 30)\n \
         \n  Loop-closure PGO (after VO + optional BA; needs a loopy sequence):\n \
         [--loop-closure]  detect revisited places by VLAD appearance, verify \
         with PnP into metric loop constraints, and re-distribute accumulated \
         drift with a robust GNC SE(3) pose-graph solve. Unlike BA this couples \
         a revisit to its earlier observation, so it removes loop drift that \
         dense global BA cannot. No-op on a loop-free trajectory.\n \
         [--loop-min-frame-gap <n>]  minimum frame gap between the two frames of \
         a loop candidate (default 50)\n \
         [--loop-min-path-length <m>]  minimum accumulated VO travel (metres) \
         between a loop candidate's two frames; the speed/frame-rate-independent \
         loop gate (drift accrues with distance, not frame index). 0 = disable. \
         (default 5)\n \
         [--loop-min-similarity <x>]  minimum VLAD cosine similarity to propose a \
         candidate (default 0.20)\n \
         [--loop-vocab-k <n>]  VLAD vocabulary size / k-means centroids \
         (default 64)\n \
         [--loop-max-candidates-per-frame <n>]  strongest earlier matches kept \
         per query frame (default 3)\n \
         [--loop-max-verifications <n>]  global cap on candidates sent to PnP \
         verification (descending similarity first); bounds the per-pair \
         brute-force matching cost on long sequences. 0 = verify all \
         (default 400)\n \
         [--loop-matches-dir <dir>]  optional external loop candidate matches \
         named loop_OLDER_NEWER_matches.txt, query=newer/train=older; used for \
         matching-stage A/B before PnP, with missing pairs falling back to BF\n \
         [--loop-pnp-essential-inliers]  before PnP, keep only matches that are \
         essential-matrix inliers under the same candidate-pair match set; \
         opt-in diagnostic/policy A/B for long-baseline loop verification\n \
         [--loop-pnp-confidence-weights]  bias loop PnP RANSAC sampling by \
         descriptor-match confidence when enough non-uniform confidences are \
         available; opt-in A/B for LightGlue candidate-pair matches\n \
         [--loop-two-view-ba]  re-grind each verified loop's relative pose with a \
         local two-view bundle adjustment (older pose fixed, newer pose + shared \
         landmarks free, older stereo a soft metric anchor) before the pose graph; \
         removes the older-depth triangulation bias from each loop edge without \
         touching the rest of the trajectory\n \
         [--loop-edge-information]  give each loop edge an anisotropic 6x6 \
         information matrix from its reprojection geometry (trace-normalised to the \
         same total weight), so the PGO routes each loop correction into the \
         directions the loop actually observes instead of pulling all 6 DOF \
         equally\n \
         \n  3DGS / NeRF export (after VO + optional BA):\n \
         [--colmap-export <dir>]      write COLMAP cameras.txt / images.txt / \
         points3D.txt under <dir>, suitable for gaussian-splatting / nerfstudio \
         ingestion; the 3D points are stereo-triangulated landmarks lifted to \
         world via the (refined) poses\n \
         [--colmap-export-binary <dir>]   binary counterpart of --colmap-export: \
         writes cameras.bin / images.bin / points3D.bin under <dir>. Use this \
         when the downstream trainer prefers the binary form (Inria 3DGS / \
         nerfstudio both accept it). Independent of --colmap-export, so a \
         single VO run can emit both formats.\n \
         [--sfm-colmap-out <dir>]     SfM-grade COLMAP export: chains temporal \
         matches into merged multi-view tracks, runs one global bundle \
         adjustment over all poses+landmarks, and writes a model whose POINT3D \
         TRACK[] tails span every observing frame. Unlike --colmap-export (one \
         single-observation landmark per frame), this is the multi-view \
         structure a 3DGS optimizer needs to converge crisply. Reuses the \
         --ba-* knobs (track length, depth/residual gates, huber, iterations)\n \
         [--colmap-image-prefix <s>]  prefix for image NAME field in images.* \
         (default empty)\n \
         [--colmap-image-suffix <s>]  suffix for image NAME field in images.* \
         (default .png); the NAME becomes `<prefix><6-digit frame_idx><suffix>`\n \
         \n  Streaming BA IMU state dump:\n \
         [--online-ba-imu-csv <path>]  write the per-(trigger, in-window keyframe) refined \
         velocity / gyro-bias / accel-bias from streaming BA to <path>; requires --online-ba"
    );
}

fn parse_relative_pose_mode(
    value: &str,
) -> Result<StereoRelativePoseMode, Box<dyn std::error::Error>> {
    match value {
        "pnp" | "pnp-then-kabsch" => Ok(StereoRelativePoseMode::PnpThenKabsch),
        "kabsch" | "kabsch-then-pnp" => Ok(StereoRelativePoseMode::KabschThenPnp),
        other => Err(format!("--relative-pose-mode must be pnp|kabsch, got {other}").into()),
    }
}

fn left_features_name(frame_id: usize) -> String {
    format!("frame_{frame_id:06}_left_features.txt")
}

fn right_features_name(frame_id: usize) -> String {
    format!("frame_{frame_id:06}_right_features.txt")
}

fn stereo_matches_name(frame_id: usize) -> String {
    format!("frame_{frame_id:06}_stereo_matches.txt")
}

fn temporal_matches_name(frame_id: usize) -> String {
    format!("frame_{frame_id:06}_temporal_matches.txt")
}

fn parse_loop_matches_name(path: &Path) -> Option<(usize, usize)> {
    let stem = path.file_stem()?.to_str()?;
    let rest = stem.strip_prefix("loop_")?;
    let rest = rest.strip_suffix("_matches").unwrap_or(rest);
    let (older_raw, newer_raw) = rest.split_once('_')?;
    Some((older_raw.parse().ok()?, newer_raw.parse().ok()?))
}

fn load_loop_matches_dir(dir: &Path) -> Result<LoopMatchesByPair, Box<dyn std::error::Error>> {
    let mut out = HashMap::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("txt") {
            continue;
        }
        let Some((older, newer)) = parse_loop_matches_name(&path) else {
            continue;
        };
        let matches = read_external_deep_matches_txt(&path)?.into_descriptor_matches();
        out.insert((older, newer), matches);
    }
    Ok(out)
}

fn write_trajectory_csv(
    path: &Path,
    centers: &[nalgebra::Point3<f64>],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut text = String::from("id,x,y,z\n");
    for (i, center) in centers.iter().enumerate() {
        text.push_str(&format!(
            "{i},{:.6},{:.6},{:.6}\n",
            center.x, center.y, center.z
        ));
    }
    fs::write(path, text)?;
    Ok(())
}

fn write_pair_diagnostics_csv(
    path: &Path,
    diagnostics: &[StereoVoPairDiagnostics],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut text = String::from(
        "from_id,to_id,source,temporal_matches,temporal_row_gate_px,temporal_confidence_gate,\
pnp_correspondences,stereo_pair_correspondences,inliers,raw_translation_m,raw_rotation_deg,\
translation_m,rotation_deg,motion_scale_rescued,translation_direction_rescued,\
stereo_vertical_aligned,rotation_spike_rescued,rotation_vector_rescued,\
pnp_mean_reprojection_error_px,kabsch_mean_residual_m\n",
    );
    for row in diagnostics {
        text.push_str(&format!(
            "{},{},{:?},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{},{}\n",
            row.from_frame,
            row.to_frame,
            row.source,
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

fn write_depth_gate_diagnostics_csv(
    path: &Path,
    diagnostics: &[StereoDepthGateDiagnostics],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut text = String::from(
        "frame_id,adaptive,candidates,accepted,effective_min_depth_m,effective_max_depth_m,\
depth_quantile_m,disparity_uncertainty_min_px\n",
    );
    for (frame_id, row) in diagnostics.iter().enumerate() {
        text.push_str(&format!(
            "{},{},{},{},{:.6},{:.6},{},{}\n",
            frame_id,
            row.adaptive,
            row.candidate_count,
            row.accepted_count,
            row.effective_min_depth_m,
            row.effective_max_depth_m,
            optional_f64(row.depth_quantile_m),
            optional_f64(row.disparity_uncertainty_min_px),
        ));
    }
    fs::write(path, text)?;
    Ok(())
}

fn write_loop_candidates_csv(
    path: &Path,
    candidates: &[LoopCandidatePair],
    retrieval_label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut text = String::from("frontend,matched_keyframe_id,query_frame_id,score\n");
    let label = csv_cell(retrieval_label);
    for candidate in candidates {
        text.push_str(&format!(
            "{},{},{},{:.8}\n",
            label, candidate.older, candidate.newer, candidate.similarity
        ));
    }
    fs::write(path, text)?;
    Ok(())
}

fn write_loop_candidate_verifications_csv(
    path: &Path,
    diagnostics: &[LoopCandidateVerificationDiagnostic],
    retrieval_label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut text = String::from(
        "frontend,matched_keyframe_id,query_frame_id,score,attempted,match_count,match_source,\
pnp_correspondence_count,pnp_unfiltered_correspondence_count,pnp_filter,\
pnp_weighted_sampling,pnp_weight_policy,pnp_weight_confidence_count,\
pnp_weight_confidence_spread,\
verified,failure_reason,inlier_count,inlier_ratio,\
mean_sampson_error,mean_reprojection_error_px,verification_score,has_relative_pose,\
essential_correspondence_count,essential_verified,essential_failure_reason,\
essential_inlier_count,essential_inlier_ratio,essential_mean_sampson_error,\
essential_score,essential_has_relative_pose\n",
    );
    let label = csv_cell(retrieval_label);
    for diagnostic in diagnostics {
        let failure_reason = diagnostic.failure_reason.as_deref().unwrap_or_default();
        let essential_failure_reason = diagnostic
            .essential_failure_reason
            .as_deref()
            .unwrap_or_default();
        text.push_str(&format!(
            "{},{},{},{:.8},{},{},{},{},{},{},{},{},{},{:.6},{},{},{},{:.6},{:.6},{},{:.6},{},{},{},{},{},{:.6},{:.6},{:.6},{}\n",
            label,
            diagnostic.older,
            diagnostic.newer,
            diagnostic.similarity,
            diagnostic.attempted,
            diagnostic.match_count,
            diagnostic.match_source,
            diagnostic.pnp_correspondence_count,
            diagnostic.pnp_unfiltered_correspondence_count,
            diagnostic.pnp_filter,
            diagnostic.pnp_weighted_sampling,
            diagnostic.pnp_weight_policy,
            diagnostic.pnp_weight_confidence_count,
            diagnostic.pnp_weight_confidence_spread,
            diagnostic.verified,
            csv_cell(failure_reason),
            diagnostic.inlier_count,
            diagnostic.inlier_ratio,
            diagnostic.mean_sampson_error,
            optional_f64(diagnostic.mean_reprojection_error_px),
            diagnostic.score,
            diagnostic.has_relative_pose,
            diagnostic.essential_correspondence_count,
            diagnostic.essential_verified,
            csv_cell(essential_failure_reason),
            diagnostic.essential_inlier_count,
            diagnostic.essential_inlier_ratio,
            diagnostic.essential_mean_sampson_error,
            diagnostic.essential_score,
            diagnostic.essential_has_relative_pose,
        ));
    }
    fs::write(path, text)?;
    Ok(())
}

fn csv_cell(value: &str) -> String {
    if value.chars().any(|c| matches!(c, ',' | '"' | '\n' | '\r')) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn imu_window_name(frame_id: usize) -> String {
    format!("frame_{frame_id:06}_imu.txt")
}

/// Build `frames - 1` IMU windows from a KITTI raw `oxts/` directory + an
/// image-stream `timestamps.txt`. The OXTS records' body-frame accel/gyro and
/// per-sample wall-clock nanoseconds are sliced into per-keyframe windows by
/// [`slice_imu_samples_for_keyframes`] using the image timestamps' first
/// `frames` rows as keyframe times.
fn load_imu_windows_from_kitti_oxts(
    oxts_dir: &Path,
    image_timestamps: &Path,
    frames: usize,
) -> Result<Vec<Vec<StereoVoBaImuSample>>, Box<dyn std::error::Error>> {
    let records = read_kitti_oxts_dir(oxts_dir)
        .map_err(|e| format!("reading {} (oxts): {e}", oxts_dir.display()))?;
    if records.is_empty() {
        return Err(format!(
            "no OXTS records found under {} (expected oxts/data/*.txt)",
            oxts_dir.display(),
        )
        .into());
    }
    let imu_timestamps_ns: Vec<i128> = records.iter().map(|r| r.timestamp_nanoseconds).collect();
    let imu_gyro: Vec<Vector3<f64>> = records
        .iter()
        .map(|r| r.sample.angular_rate_body_rps)
        .collect();
    let imu_accel: Vec<Vector3<f64>> = records
        .iter()
        .map(|r| r.sample.acceleration_body_mps2)
        .collect();

    let kf_text = fs::read_to_string(image_timestamps).map_err(|e| {
        format!(
            "reading {} (image timestamps): {e}",
            image_timestamps.display()
        )
    })?;
    let all_kf_timestamps = parse_kitti_oxts_timestamps_txt(&kf_text).map_err(|e| {
        format!(
            "parsing {} (image timestamps): {e}",
            image_timestamps.display()
        )
    })?;
    if all_kf_timestamps.len() < frames {
        return Err(format!(
            "{} has only {} timestamps but --frames is {frames}",
            image_timestamps.display(),
            all_kf_timestamps.len(),
        )
        .into());
    }
    let kf_timestamps_ns = &all_kf_timestamps[..frames];

    let windows = slice_imu_samples_for_keyframes(
        &imu_timestamps_ns,
        &imu_gyro,
        &imu_accel,
        kf_timestamps_ns,
    )
    .map_err(|e| format!("slicing KITTI OXTS samples into keyframe windows: {e}"))?;
    println!(
        "KITTI OXTS: {} records, {} keyframes ({} windows) from {} + {}",
        records.len(),
        kf_timestamps_ns.len(),
        windows.len(),
        oxts_dir.display(),
        image_timestamps.display(),
    );
    Ok(windows)
}

/// Load `frames - 1` per-window IMU files. Window index `i` covers the
/// integration interval from keyframe `i` to keyframe `i + 1`; the file
/// on disk is named `frame_{i+1:06}_imu.txt` (mirrors the temporal-
/// matches naming, which is "frame N's matches against frame N-1"). A
/// missing file is treated as an empty window so callers can drop in
/// IMU for only part of the sequence.
fn load_imu_windows(
    dir: &Path,
    frames: usize,
) -> Result<Vec<Vec<StereoVoBaImuSample>>, Box<dyn std::error::Error>> {
    let mut out: Vec<Vec<StereoVoBaImuSample>> = Vec::with_capacity(frames.saturating_sub(1));
    for to_frame in 1..frames {
        let path = dir.join(imu_window_name(to_frame));
        if !path.exists() {
            out.push(Vec::new());
            continue;
        }
        let text =
            fs::read_to_string(&path).map_err(|e| format!("reading {}: {e}", path.display()))?;
        let samples = parse_stereo_vo_imu_samples_txt(&text)
            .map_err(|e| format!("parsing {}: {e}", path.display()))?;
        out.push(samples);
    }
    Ok(out)
}

fn write_imu_state_csv(
    path: &Path,
    velocities: &[Vector3<f64>],
    bias_gyro: &[Vector3<f64>],
    bias_acc: &[Vector3<f64>],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut text = String::from("id,vx,vy,vz,bg_x,bg_y,bg_z,ba_x,ba_y,ba_z\n");
    let n = velocities.len().min(bias_gyro.len()).min(bias_acc.len());
    for i in 0..n {
        let v = &velocities[i];
        let bg = &bias_gyro[i];
        let ba = &bias_acc[i];
        text.push_str(&format!(
            "{i},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            v.x, v.y, v.z, bg.x, bg.y, bg.z, ba.x, ba.y, ba.z,
        ));
    }
    fs::write(path, text)?;
    Ok(())
}

fn optional_f64(value: Option<f64>) -> String {
    value
        .filter(|value| value.is_finite())
        .map(|value| format!("{value:.6}"))
        .unwrap_or_default()
}

fn optional_f32(value: Option<f32>) -> String {
    value
        .filter(|value| value.is_finite())
        .map(|value| format!("{value:.6}"))
        .unwrap_or_default()
}

fn stereo_depth_gate_label(gate: &StereoDepthGate) -> &'static str {
    match gate {
        StereoDepthGate::Fixed => "fixed",
        StereoDepthGate::Adaptive(_) => "adaptive",
    }
}
