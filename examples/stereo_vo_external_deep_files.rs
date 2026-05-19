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

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use nalgebra::Vector3;
use visloc_rs::{
    parse_kitti_calibration_txt, parse_kitti_oxts_timestamps_txt, parse_stereo_vo_imu_samples_txt,
    read_external_deep_features_txt, read_external_deep_matches_txt, read_kitti_oxts_dir,
    refine_stereo_vo_with_ba, slice_imu_samples_for_keyframes, write_colmap_binary_model_for_3dgs,
    write_colmap_text_model_for_3dgs, write_online_ba_imu_state_csv, BaConfig, Camera,
    DescriptorMatch, GravityPrior, KabschRansacConfig, LandmarkInit, LinearSolver,
    OnlineStereoVoBa, OnlineStereoVoBaConfig, PerPoseGravityObservation, PerPoseGravityPrior, Pose,
    PoseTrajectory, PositionPrior, PositionPriorObservation, RobustKernel, StereoRelativePoseMode,
    StereoVoBaConfig, StereoVoBaImuInput, StereoVoBaImuSample, StereoVoFrontend,
    StereoVoFrontendConfig, StereoVoPairDiagnostics, TrackingEvent, TrackingState,
    TrajectorySample,
};

#[derive(Debug)]
struct CliArgs {
    features_dir: PathBuf,
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
    let config = StereoVoFrontendConfig {
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

    for frame_id in 0..args.frames {
        let left_features =
            read_external_deep_features_txt(args.features_dir.join(left_features_name(frame_id)))?
                .into_feature_set()?;
        let right_features =
            read_external_deep_features_txt(args.features_dir.join(right_features_name(frame_id)))?
                .into_feature_set()?;
        let stereo_matches =
            read_external_deep_matches_txt(args.features_dir.join(stereo_matches_name(frame_id)))?
                .into_descriptor_matches();
        let stereo_matches =
            filter_matches_by_confidence(stereo_matches, args.min_stereo_confidence);
        let temporal_matches = if frame_id == 0 {
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
        if args.enable_ba {
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
    fs::write(
        args.out_dir.join("summary.txt"),
        format!(
            "frames={} pairs={} trajectory_length_m={:.6}\n",
            frontend.frame_count(),
            frontend.pair_diagnostics.len(),
            frontend.trajectory_length_m(),
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

fn kitti_stereo_calibration(
    calib_path: &Path,
    projection_left: &str,
    projection_right: &str,
) -> Result<(f64, f64, f64, f64, f64), Box<dyn std::error::Error>> {
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

    Ok(CliArgs {
        features_dir: features_dir.ok_or("--features-dir is required")?,
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
