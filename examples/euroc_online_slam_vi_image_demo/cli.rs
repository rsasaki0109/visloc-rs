//! Command-line options (`CliArgs`) and their parsing.

use super::*;

#[cfg(feature = "image-io")]
#[derive(Debug)]
pub(super) struct CliArgs {
    pub(super) euroc_dir: PathBuf,
    pub(super) out_dir: PathBuf,
    pub(super) max_frames: usize,
    /// When set, the SuperPoint ONNX frontend adapts its per-frame keypoint
    /// cap to hold the rolling per-frame processing time near
    /// [`Self::adaptive_frame_budget_ms`]. Off by default.
    pub(super) adaptive_keypoints: bool,
    /// Target per-frame processing budget (ms) for `adaptive_keypoints`. The
    /// controller lowers the keypoint cap when the rolling mean exceeds this
    /// and raises it (up to the configured cap) when it is comfortably below.
    pub(super) adaptive_frame_budget_ms: f64,
    pub(super) gravity_world: Vector3<f64>,
    pub(super) vi_init_max_wait_seconds: f64,
    pub(super) vi_init_gyro_std_limit: Option<f64>,
    pub(super) vi_init_mean_gyro_magnitude_limit: Option<f64>,
    pub(super) vi_init_accel_std_limit: Option<f64>,
    /// Phase-19 lever. Maps to
    /// `OnlineSlamViInitConfig::try_initialize_on_every_frame`.
    /// `false` (default) keeps the historical "VI-init's try_initialize
    /// only fires on new-keyframe frames" contract. `true` lets the
    /// stage attempt promotion on every frame; on success without a new
    /// KF this frame, the promotion binds to the latest existing
    /// keyframe.
    pub(super) vi_init_try_initialize_on_every_frame: bool,
    /// Fixed metric depth assumed for every first-frame corner during the
    /// landmark-map bootstrap. EuRoC MH indoor scenes hover ~4 m from the
    /// camera, so the default keeps the rigid-ATE error reasonable while
    /// still letting the tracker exercise real corner descriptors.
    pub(super) bootstrap_depth_meters: f64,
    /// Corner extractor knobs. Defaults match the patch sizes
    /// `online_slam_image_vo_loop_demo` settled on for real KITTI imagery —
    /// they generalise well to EuRoC.
    pub(super) corner_max_features: usize,
    pub(super) corner_min_score: f32,
    pub(super) corner_descriptor_radius: usize,
    /// When `true` (the default) every extracted corner is mapped from
    /// its raw distorted pixel position to the "ideal pinhole" pixel
    /// using cam0's published radial-tangential distortion model. Set
    /// to `false` via `--no-undistort` to reproduce the
    /// pre-correction behaviour for A/B comparison.
    pub(super) undistort: bool,
    /// When `true` (the default) the seed frame's cam0 corners are
    /// stereo-matched against cam1 and triangulated via DLT using the
    /// published `T_BS` extrinsics. Each surviving match becomes a
    /// metric-scale landmark. Seed keypoints without stereo depth are dropped
    /// by default; fixed-depth fallback is an explicit diagnostic opt-in.
    pub(super) stereo_bootstrap: bool,
    /// Enable the motion-based VI init stage (VIBA1 / optional VIBA2).
    /// Off by default so the demo stays backwards compatible with the
    /// existing baseline. The stage normally gates on static VI init.
    pub(super) motion_vi_init_enabled: bool,
    /// After static VI init gives up, permit motion-VI to start from the
    /// configured running IMU biases. Intended for sequences that start in
    /// motion and therefore have no valid stationary window.
    pub(super) motion_vi_init_after_static_give_up: bool,
    /// Start motion VI immediately from configured IMU biases. Explicit
    /// moving-start opt-in; does not wait for stationary init to terminate.
    pub(super) motion_vi_init_from_configured_bias: bool,
    /// Minimum keyframes after static success or the opted-in give-up fallback
    /// before the motion-VI trigger is allowed to fire. Mirrors
    /// `MotionBasedViInitializerConfig::min_keyframes`.
    pub(super) motion_vi_init_min_keyframes: usize,
    /// Minimum cumulative camera-centre translation (m) before the
    /// motion-VI trigger fires. Mirrors
    /// `MotionBasedViInitializerConfig::min_translation_meters`.
    pub(super) motion_vi_init_min_translation_meters: f64,
    /// When set, enables VIBA2 (alternating scale-recovery outer loop)
    /// instead of VIBA1-only. Defaults `false` because EuRoC is a
    /// stereo-bootstrapped run (known scale) — flip to `true` for a
    /// monocular sanity check.
    pub(super) motion_vi_init_recover_scale: bool,
    /// Opt into the local VI-BA sliding-window stage so the refined
    /// per-keyframe `(velocity, bias)` produced by motion-VI init has a
    /// downstream consumer. Off by default to preserve the existing
    /// baseline behaviour. Mirrors `OnlineSlamLocalBaConfig::default()`
    /// with `gravity_world` rebased onto `--gravity`.
    pub(super) local_vi_ba_enabled: bool,
    /// Global A/B lever shared by local VI-BA and covisibility local BA.
    /// Experimental and off by default until a multi-sequence tracking-cliff
    /// A/B demonstrates a net win.
    pub(super) observation_confidence_ba_enabled: bool,
    /// Schur-marginalize the outgoing navigation state into a dense FEJ prior
    /// on the next window anchor.
    pub(super) local_vi_ba_marginalization: bool,
    /// Use the square-root (SqrtToSqrt) sliding-window marginalization (Basalt
    /// ICCV'21). Default on. Disable with `--no-sqrt-window-marginalization`.
    pub(super) local_vi_ba_use_sqrt_window_marginalization: bool,
    /// Trailing VI-BA keyframe window depth. Defaults to
    /// `OnlineSlamLocalBaConfig::default().window_size` (5). Basalt EuRoC uses
    /// `vio_max_kfs = 7` via `--basalt-config`.
    pub(super) local_vi_ba_window_size: usize,
    /// Optional path to Basalt `euroc_config.json` (or compatible). When set,
    /// applies faithful VIO knobs (`vio_max_kfs`, `vio_sqrt_marg`, …) and
    /// selects the optical-flow frontend.
    pub(super) basalt_config_path: Option<PathBuf>,
    /// Optical-flow knobs used when `feature_extractor == OpticalFlow`
    /// (from Basalt config or EuRoC defaults).
    pub(super) optical_flow_config: visloc_rs::vision::optical_flow::BasaltOpticalFlowConfig,
    /// Optional finite initialization uncertainty `(velocity, gyro bias,
    /// accel bias)` used by the first marginal prior.
    pub(super) local_vi_ba_initial_prior_std_devs: Option<(f64, f64, f64)>,
    /// When `Some(v)`, runs the local VI-BA conditioning fallback:
    /// after the joint solve, if `final_cost / initial_cost > v`, the
    /// window is re-solved with biases gauge-frozen and the bias
    /// updates are discarded. `None` (the default) preserves the legacy
    /// always-update-biases behaviour.
    pub(super) local_vi_ba_freeze_biases_above: Option<f64>,
    /// When `Some(v)`, rejects the entire local VI-BA writeback when
    /// the selected solve has `final_cost / initial_cost > v`. This is
    /// a stricter safety gate than `local_vi_ba_freeze_biases_above`:
    /// rejected passes return diagnostics but leave map poses,
    /// landmarks, velocities, and biases untouched.
    pub(super) local_vi_ba_reject_writeback_above: Option<f64>,
    pub(super) local_vi_ba_reject_final_imu_nis_per_dof_above: Option<f64>,
    /// When `Some(v)`, rejects the entire local VI-BA writeback when
    /// any refined in-window `||velocity_world|| > v`. This catches
    /// non-physical tight-VIO updates that can still reduce reprojection
    /// cost by injecting bad velocity state.
    pub(super) local_vi_ba_reject_velocity_above_mps: Option<f64>,
    pub(super) local_vi_ba_reject_pose_translation_above_meters: Option<f64>,
    pub(super) local_vi_ba_reject_pose_rotation_above_degrees: Option<f64>,
    /// When `Some(v)`, rejects the entire local VI-BA writeback when any
    /// refined in-window `||bias_gyro|| > v` rad/s. Real-data motivation:
    /// see `docs/motion_based_vi_alignment.md`'s "Local VI-BA
    /// bias-magnitude gate" section — an unfiltered local VI-BA writeback
    /// mirrored an accel bias diverging to `‖·‖ ≈ 8.6 m/s²` on full MH_01.
    pub(super) local_vi_ba_reject_gyro_bias_above_rad_s: Option<f64>,
    /// When `Some(v)`, rejects the entire local VI-BA writeback when any
    /// refined in-window `||bias_acc|| > v` m/s². See
    /// [`Self::local_vi_ba_reject_gyro_bias_above_rad_s`].
    pub(super) local_vi_ba_reject_accel_bias_above_mps2: Option<f64>,
    /// Enable the adaptive local VI-BA refined-velocity writeback gate.
    /// The gate derives a per-trigger velocity threshold from the local
    /// window's existing velocity state, pose-delta / IMU-`dt` finite
    /// differences, and IMU-predicted next-keyframe velocities, avoiding
    /// a raw scene-scale `m/s` threshold as the primary decision.
    pub(super) local_vi_ba_adaptive_velocity_gate: bool,
    /// Adaptive local VI-BA velocity gate robust reference quantile.
    pub(super) local_vi_ba_adaptive_velocity_quantile: f64,
    /// Adaptive local VI-BA velocity gate multiplier.
    pub(super) local_vi_ba_adaptive_velocity_multiplier: f64,
    /// Adaptive local VI-BA velocity gate additive slack in m/s.
    pub(super) local_vi_ba_adaptive_velocity_margin_mps: f64,
    /// Adaptive local VI-BA velocity gate lower bound in m/s.
    pub(super) local_vi_ba_adaptive_velocity_min_mps: f64,
    /// Optional adaptive local VI-BA velocity gate upper bound in m/s.
    pub(super) local_vi_ba_adaptive_velocity_max_mps: Option<f64>,
    /// Minimum finite local velocity references required before the
    /// adaptive local VI-BA velocity gate can reject writeback.
    pub(super) local_vi_ba_adaptive_velocity_min_references: usize,
    /// When `Some(v)`, runs the motion-VI post-solve velocity sanity
    /// gate: if any per-keyframe `||velocity_world|| > v`, the inner LM
    /// result is rejected and the stage stays in `Waiting`. `None`
    /// (the default) preserves legacy behaviour. EuRoC V1-class indoor
    /// sequences run safely at `Some(10.0)`.
    pub(super) motion_vi_init_max_velocity_mps: Option<f64>,
    /// Optional post-solve magnitude bound for every recovered gyro bias.
    pub(super) motion_vi_init_max_gyro_bias_rad_s: Option<f64>,
    /// Optional post-solve magnitude bound for every recovered accel bias.
    pub(super) motion_vi_init_max_accel_bias_mps2: Option<f64>,
    /// Optional upper bound on the final whitened IMU NIS per residual DoF.
    pub(super) motion_vi_init_max_imu_nis_per_dof: Option<f64>,
    pub(super) motion_vi_init_max_rotation_residual_rms_rad: Option<f64>,
    pub(super) motion_vi_init_max_velocity_residual_rms_mps: Option<f64>,
    pub(super) motion_vi_init_max_position_residual_rms_meters: Option<f64>,
    /// Optional staged bias-release schedule
    /// ([`visloc_rs::BiasReleaseSchedule`]): `Some` when either
    /// `--vi-bias-release-min-keyframes` or
    /// `--vi-bias-release-min-translation` is passed. The other knob falls
    /// back to the schedule's documented default (10 keyframes / 2.0 m)
    /// when only one flag is given.
    pub(super) vi_bias_release_min_keyframes: Option<usize>,
    pub(super) vi_bias_release_min_translation_meters: Option<f64>,
    /// Enable gravity-direction recovery
    /// ([`visloc_rs::MotionBasedViInitializerConfig::estimate_gravity`]):
    /// estimate the world-frame gravity vector from this window's IMU
    /// preintegration factors + fixed visual poses instead of trusting
    /// `--gravity` as ground truth. Off by default. See
    /// `docs/motion_based_vi_alignment.md`'s "Gravity-direction recovery"
    /// section for the motivating real-data diagnosis (a ~90-degree
    /// misalignment on a moving-start sequence where the static VI
    /// initializer never fires).
    pub(super) motion_vi_init_estimate_gravity: bool,
    /// Override
    /// [`visloc_rs::MotionBasedViInitializerConfig::max_gravity_norm_deviation_ratio`]
    /// (default `0.3`). Ignored unless `motion_vi_init_estimate_gravity` is
    /// set.
    pub(super) motion_vi_init_max_gravity_norm_deviation: Option<f64>,
    /// Enable gyro-bias recovery
    /// ([`visloc_rs::MotionBasedViInitializerConfig::estimate_gyro_bias`]):
    /// estimate the shared gyro bias from rotation-only alignment against
    /// this window's fixed visual poses, BEFORE gravity/velocity alignment
    /// and before the staged solve. Defaults on whenever `--motion-vi-init`
    /// is set; pass `--no-motion-vi-init-estimate-gyro-bias` to disable.
    pub(super) motion_vi_init_estimate_gyro_bias: bool,
    /// When `Some(n)`, restricts descriptor matching during tracking to
    /// landmarks observed by the reference keyframe and up to `n`
    /// co-visible neighbour keyframes (ranked by shared-landmark count).
    /// `None` (the default) preserves legacy whole-map matching.
    pub(super) covisibility_local_map_max_keyframes: Option<usize>,
    /// Minimum shared-landmark count required for a candidate keyframe
    /// to enter the covisibility-derived local map. Only used when
    /// `covisibility_local_map_max_keyframes` is set.
    pub(super) covisibility_local_map_min_shared: usize,
    /// Opt into visual-only covisibility local BA inside
    /// `OnlineSlamPipeline`. This is distinct from the tracker-side
    /// covisibility local map: it runs a backend BA solve after newly
    /// applied keyframes, using co-visible keyframes and fixed boundary
    /// keyframes from the accumulated `VisualMap`. Off by default so
    /// baseline runs stay unchanged.
    pub(super) covisibility_local_ba_enabled: bool,
    /// Enable sparse DROID-style temporal/proximity/stereo factor lifecycle.
    pub(super) sparse_factor_graph_enabled: bool,
    /// Minimum map keyframe count before online covisibility local BA
    /// can run. Skips startup windows that cannot anchor a useful local
    /// solve.
    pub(super) covisibility_local_ba_min_keyframes: usize,
    pub(super) covisibility_local_ba_max_keyframes: Option<usize>,
    pub(super) covisibility_local_ba_motion_vi_raw_activation: Option<(f64, f64, f64)>,
    pub(super) covisibility_local_ba_max_seed_landmarks_for_activation: Option<usize>,
    /// Run online covisibility local BA after every N newly-applied
    /// keyframes. Values below 1 are rejected during argument parsing.
    pub(super) covisibility_local_ba_trigger_every: usize,
    /// Maximum optimized co-visible neighbor keyframes, excluding the
    /// active keyframe.
    pub(super) covisibility_local_ba_max_neighbor_keyframes: usize,
    /// Minimum shared landmarks for a keyframe to become an optimized
    /// covisibility-BA neighbor.
    pub(super) covisibility_local_ba_min_shared: usize,
    /// Maximum fixed boundary keyframes that observe the selected local
    /// landmarks.
    pub(super) covisibility_local_ba_max_boundary_keyframes: usize,
    /// Minimum local-landmark observations for a keyframe to become a
    /// fixed boundary keyframe.
    pub(super) covisibility_local_ba_min_boundary_observations: usize,
    /// Optional lower fixed-boundary threshold used only when the primary
    /// boundary threshold produces no local BA landmarks.
    pub(super) covisibility_local_ba_fallback_min_boundary_observations: Option<usize>,
    /// Optional cap on local landmarks passed to the BA solve.
    pub(super) covisibility_local_ba_max_landmarks: Option<usize>,
    /// Minimum selected local-landmark observations on the active keyframe
    /// before online covisibility local BA is allowed to run.
    pub(super) covisibility_local_ba_min_active_observations: usize,
    /// Optional post-BA reprojection threshold for outlier diagnostics.
    pub(super) covisibility_local_ba_outlier_threshold_px: Option<f64>,
    /// Remove post-BA observations above
    /// `covisibility_local_ba_outlier_threshold_px`.
    pub(super) covisibility_local_ba_remove_outliers: bool,
    /// Use calibrated non-rectified cam0/cam1 factors for stored stereo
    /// observations in covisibility local BA.
    pub(super) covisibility_local_ba_general_stereo: bool,
    pub(super) covisibility_local_ba_general_stereo_max_right_reprojection_px: Option<f64>,
    /// Reject covisibility BA map write-back when post-BA outlier
    /// observations exceed this fraction of selected observations.
    pub(super) covisibility_local_ba_max_outlier_observation_ratio: Option<f64>,
    /// Optional pre-solve guard for large optimized windows with too few
    /// fixed boundary keyframes.
    pub(super) covisibility_local_ba_boundary_support_min_optimized_keyframes: Option<usize>,
    /// Fixed-boundary keyframe floor used with the boundary-support guard.
    pub(super) covisibility_local_ba_boundary_support_min_fixed_keyframes: usize,
    /// Reject covisibility BA map write-back when the fraction of solved
    /// optimized landmarks projecting behind an observing optimized camera
    /// exceeds this value. Off (`None`) by default.
    pub(super) covisibility_local_ba_max_behind_camera_ratio: Option<f64>,
    /// Reject covisibility BA map write-back unless
    /// `fixed >= ceil(optimized * ratio)`. Off (`None`) by default.
    pub(super) covisibility_local_ba_min_fixed_to_optimized_ratio: Option<f64>,
    /// Reject a solved covisibility BA update above this camera-centre shift.
    pub(super) covisibility_local_ba_max_pose_translation_correction_m: Option<f64>,
    /// Reject a solved covisibility BA update above this rotation change.
    pub(super) covisibility_local_ba_max_pose_rotation_correction_deg: Option<f64>,
    /// Optional gauge/global-anchoring pose-prior weight for optimized
    /// covisibility-BA keyframes. Off (`None`) by default.
    pub(super) covisibility_local_ba_anchor_weight: Option<f64>,
    /// When `true`, replenishes the map beyond the stereo bootstrap by
    /// stereo-matching each frame's cam0 keypoints that tracking did NOT
    /// match to an existing landmark against a freshly loaded/undistorted
    /// cam1 image, triangulating the survivors, and staging them as
    /// `LandmarkCandidate`s for `process_frame`. Defaults on whenever
    /// `--local-vi-ba` and stereo bootstrap are both active (cam1 features
    /// required); pass `--no-stereo-landmark-replenish` to disable.
    pub(super) stereo_landmark_replenish: bool,
    /// Cap on new replenishment candidates built per frame. Only
    /// meaningful when `stereo_landmark_replenish` is set.
    pub(super) stereo_landmark_replenish_max_per_frame: usize,
    /// Radius (px) around the reprojected anchor pixel within which a real
    /// detected anchor-keyframe keypoint must exist. Maps to
    /// [`StereoReplenishConfig::anchor_keypoint_match_radius_px`].
    pub(super) stereo_landmark_replenish_anchor_match_radius_px: Option<f64>,
    /// Optional descriptor-distance gate on the associated anchor keypoint.
    /// Maps to [`StereoReplenishConfig::anchor_keypoint_max_descriptor_distance`].
    pub(super) stereo_landmark_replenish_anchor_max_descriptor_distance: Option<f32>,
    /// Radius (px) for geometric duplicate suppression against the anchor
    /// keyframe's own landmarks. Maps to
    /// [`StereoReplenishConfig::duplicate_suppression_radius_px`].
    pub(super) stereo_landmark_replenish_duplicate_radius_px: Option<f64>,
    /// Minimum anchor↔current parallax (degrees). Maps to
    /// [`StereoReplenishConfig::min_parallax_deg`].
    pub(super) stereo_landmark_replenish_min_parallax_deg: Option<f64>,
    /// Minimum triangulated depth (m). Maps to
    /// [`StereoReplenishConfig::min_depth_meters`].
    pub(super) stereo_landmark_replenish_min_depth_meters: Option<f64>,
    /// Maximum triangulated depth (m). Maps to
    /// [`StereoReplenishConfig::max_depth_meters`].
    pub(super) stereo_landmark_replenish_max_depth_meters: Option<f64>,
    /// When `Some(d)`, rejects a tracked frame whose PnP camera-centre
    /// drifts more than `d` metres from the motion-model pose prior
    /// (`ConstantPoseMotionModel` returns the last successful pose, so
    /// this effectively becomes a per-frame translation cap). Catches
    /// catastrophic PnP outliers that teleport the trajectory to
    /// unphysical positions. `None` (the default) preserves legacy
    /// always-accept behaviour. For EuRoC V1-class indoor sequences
    /// (~0.5 m/s walking pace, 50 ms cam0 cadence) values in the
    /// `0.2`–`1.0` m range are reasonable.
    pub(super) max_pose_jump_meters: Option<f64>,
    /// Allow a bounded pose-prior gate widening for PnP solutions with at
    /// least 100 inliers, 0.6 inlier ratio, and 3 px mean reprojection error.
    pub(super) pose_prior_visual_override: bool,
    /// Accept the motion-model pose prior when visual tracking fails
    /// (IMU coast). See `TrackingConfig::accept_motion_prior_on_failure`.
    pub(super) accept_motion_prior_on_failure: bool,
    /// Cap on consecutive IMU coasts (`None` = unlimited; not recommended).
    pub(super) max_consecutive_motion_prior_coasts: Option<usize>,
    /// Temporal re-association of previous inlier landmarks (image-free KLT analogue).
    pub(super) temporal_landmark_tracking: bool,
    /// How many recent successful inlier sets to union for temporal matching.
    pub(super) temporal_landmark_tracking_history_frames: usize,
    /// When `true`, scale `--max-pose-jump-meters` by the number of frames
    /// elapsed since the last successful track (capped at
    /// `--pose-jump-gap-scaling-max-multiplier`, floored at 1) before
    /// comparing. Addresses a permanent-tracking-loss failure mode
    /// observed on EuRoC MH_01: a single tracking failure freezes the
    /// motion-model pose prior (`ConstantPoseMotionModel` returns the last
    /// successful pose), and a subsequent frame with a genuinely good PnP
    /// solution gets rejected by the fixed-radius gate because the prior
    /// is stale, not because the solution is bad. Scaling the gate by the
    /// gap lets a good post-gap solution through while still catching
    /// same-gap outliers at the original radius. Off by default; no effect
    /// unless `--max-pose-jump-meters` is also set.
    pub(super) pose_jump_gap_scaling: bool,
    /// Companion to `--pose-jump-gap-scaling`: caps the gap-scaling
    /// multiplier so an extended tracking outage doesn't inflate the gate
    /// to an unbounded radius. Defaults to `10`.
    pub(super) pose_jump_gap_scaling_max_multiplier: usize,
    /// Minimum PnP inlier count required after localization. Defaults
    /// to `0`, preserving legacy behaviour. Useful when relaxing
    /// pose-prior gates: frames with only a handful of correspondences
    /// should not refresh the tracker or enter the map.
    pub(super) tracking_min_inliers: usize,
    /// Minimum PnP inlier ratio required after localization. Defaults
    /// to `0.0`, preserving legacy behaviour. Combine with
    /// `tracking_min_inliers` to separate visual-rescue frames from
    /// near-lost outliers.
    pub(super) tracking_min_inlier_ratio: f64,
    /// Optional mean reprojection error ceiling for accepted
    /// localizations. Defaults to `None`.
    pub(super) tracking_max_reprojection_error: Option<f64>,
    /// Override the tracker's PnP RANSAC reprojection-error inlier
    /// threshold (`LocalizationConfig::reprojection_threshold`,
    /// default 4.0 px). Larger values admit more correspondences as
    /// inliers and accept more frames at marginal accuracy cost; the
    /// Phase-26 #3a use case is allowing SuperPoint trajectories on
    /// V-class to track further past the universal cliff (where the
    /// stricter default refuses post-cliff frames that the wider
    /// gate would still trust). `None` preserves the default.
    pub(super) pnp_reprojection_threshold_px: Option<f64>,
    /// When `true`, hand the motion-model pose prior to the PnP RANSAC
    /// as a warm-start hypothesis (ORB-SLAM3-style motion-only BA seed).
    /// Random samples must beat the prior's inlier count to win, so a
    /// well-aligned prior short-circuits RANSAC on hard scenes (e.g.,
    /// faster motion where standard PnP would diverge) while a
    /// misaligned prior gracefully degrades to the standard random
    /// search. Combine with `--max-pose-jump-meters` to gate
    /// post-hoc on the warm-start result. Off by default.
    pub(super) pnp_pose_prior_warm_start: bool,
    /// Enable ORB-SLAM3-style projection-guided tracking: when a pose
    /// prior is available, restrict descriptor matching to a per-landmark
    /// projection window instead of an appearance-global search, with a
    /// widen-retry ladder that falls back to today's appearance-global
    /// path if every projection attempt fails, plus a post-hoc local-map
    /// refinement pass. Off by default; enabling it can only ADD tracking
    /// chances since the widen-retry ladder's last rung IS today's
    /// unmodified appearance-global path.
    pub(super) projection_guided_tracking: bool,
    /// Initial per-landmark projection-window radius, in pixels. Only
    /// meaningful when `--projection-guided-tracking` is set.
    pub(super) projection_search_radius_px: f64,
    /// Multiplier applied to the projection-window radius on each
    /// widen-retry attempt after a stage-1 projection attempt fails.
    pub(super) projection_widen_factor: f64,
    /// Maximum number of widen-retry attempts after the initial
    /// projection-window attempt. Total projection attempts per frame is
    /// `1 + projection_max_widen_retries`.
    pub(super) projection_max_widen_retries: u32,
    /// Reverse descriptor ambiguity ratio across projected landmarks that
    /// compete for one query keypoint. `None` restores the legacy
    /// deterministic first-wins behavior.
    pub(super) projection_query_landmark_distance_ratio: Option<f32>,
    /// Disable stage-3 local-map refinement (on by default when
    /// `--projection-guided-tracking` is set): re-projects the
    /// covisibility local map with the estimated pose, harvests
    /// additional correspondences, and re-optimizes the pose, accepting
    /// the refined result only when its inlier count does not decrease.
    pub(super) projection_no_local_map_refinement: bool,
    /// Per-landmark projection-window radius, in pixels, used by the
    /// local-map refinement stage.
    pub(super) projection_refinement_search_radius_px: f64,
    /// Alternating fresh-correspondence / pose-update rounds in stage 3.
    pub(super) projection_refinement_iterations: u32,
    /// Projection radius multiplier after every refinement round.
    pub(super) projection_refinement_radius_shrink_factor: f64,
    /// Permit fresh rounds to reassign a query keypoint to another landmark.
    pub(super) projection_refinement_reassign_correspondences: bool,
    /// Minimum retained fraction of current inlier query/landmark pairs.
    pub(super) projection_refinement_min_inlier_pair_retention_ratio: f64,
    /// Optional per-round camera-centre correction trust region.
    pub(super) projection_refinement_max_translation_correction_m: Option<f64>,
    /// Optional per-round rotation correction trust region, in degrees.
    pub(super) projection_refinement_max_rotation_correction_deg: Option<f64>,
    /// Motion model fed to the tracker. `pose` (default) returns the
    /// last successful pose as the prior (`ConstantPoseMotionModel`);
    /// `velocity` extrapolates the last 2 successful poses to predict
    /// where the body will be at this frame
    /// (`ConstantVelocityMotionModel`), or **integrate the raw inter-
    /// frame IMU samples through a strapdown predictor**
    /// (`ImuPredictiveMotionModel`). The velocity model is a strict
    /// superset of constant-pose — it falls back to the constant-pose
    /// model before 2 successes have accumulated — and the IMU
    /// predictor is the strictest of the three (uses gyro / accel to
    /// extrapolate the previous pose forward by the inter-frame Δt).
    /// `imu` is the appropriate choice for sequences with non-trivial
    /// linear / angular acceleration (e.g., EuRoC MH_01 takeoff),
    /// where constant-velocity breaks down. All three integrate
    /// cleanly with `--pnp-pose-prior-warm-start`, because the
    /// warm-start is only as good as the prior is predictive.
    pub(super) motion_model: MotionModelKind,
    /// When `true`, pass cam0's published `T_BS` (body-from-sensor)
    /// extrinsic into `ImuPredictiveMotionModel` so the strapdown
    /// integration runs in body frame and the input/output stay
    /// camera-pose; **also** wire the per-frame body-velocity update
    /// (finite-difference of two successive successful poses) so the
    /// integrator's initial velocity tracks the body's motion instead
    /// of being pinned at zero. Off by default because in the absence
    /// of a downstream VI-BA refining velocity, the visual tracker's
    /// per-frame noise leaks into the finite-difference velocity and
    /// the resulting (geometrically correct but noisy) prior leads to
    /// a higher rigid-ATE than the `body == camera` approximation —
    /// the trade-off is a perfect metric-scale recovery on
    /// accelerating sequences (MH_01: scale 1.001 vs 1.112). The flag
    /// has no effect when `--motion-model` is not `imu`.
    pub(super) imu_extrinsic_from_cam0: bool,
    /// When `true`, set `ImuPredictiveMotionModelConfig
    /// ::carry_forward_velocity_world` so the motion model commits the
    /// post-strapdown `v_w` back into its initial-velocity slot during
    /// `observe()`. Without it, every per-frame `predict_pose` re-seeds
    /// from the last VI-BA mirror (KF time) — so on frames between
    /// mirrors the predicted velocity does not advance. Off by default;
    /// no effect when `--motion-model` is not `imu`.
    pub(super) imu_motion_model_carry_forward_velocity: bool,
    /// When `true`, wrap the default `BruteForceMatcher` (Lowe ratio
    /// 0.8) in a `CrossCheckMatcher`, keeping only query↔train pairs
    /// where each side picks the other as its single best match. Off
    /// by default — the existing matcher uses the ratio test only.
    /// Cross-check is a standard ORB-SLAM-style filter that raises the
    /// per-frame inlier ratio at the cost of ~2× matcher CPU. The
    /// motivation is to register more keyframes through the strict
    /// `--max-pose-jump-meters 0.2` gate so the local-VI-BA chain (the
    /// Phase-9 mirror) actually fires on EuRoC.
    pub(super) cross_check_matcher: bool,
    /// When set, route the tracker/localizer matching through
    /// `GpuDescriptorMatcher`, loading the CUDA top-2 kernel from this DLL path
    /// (`scripts/build_descriptor_gemm_kernels.ps1`). Exact, so it replaces
    /// `BruteForceMatcher` semantically; falls back to the CPU path when the
    /// library is missing or a device call fails. Requires the `gpu-matcher`
    /// feature (ignored with a warning otherwise).
    pub(super) gpu_matcher_dll: Option<PathBuf>,
    /// When `true`, replace the default `BruteForceMatcher` (and any
    /// `--cross-check-matcher` wrap) with `MutualSoftmaxMatcher`,
    /// LightGlue-style temperature-scaled mutual-softmax over the
    /// full cosine-similarity matrix. Defaults `temperature = 20.0`,
    /// `min_confidence = 0.2`. Aimed at cliff-region cross-attitude
    /// matching where the cross-check filter alone leaves SuperPoint's
    /// signal under-exploited; mutually exclusive with
    /// `--cross-check-matcher`. Off by default.
    pub(super) mutual_softmax_matcher: bool,
    /// Mutual-softmax inverse temperature (higher = peakier). Default 20.
    pub(super) mutual_softmax_temperature: f32,
    /// Mutual-softmax dual-softmax confidence floor in (0, 1]. Default 0.2.
    pub(super) mutual_softmax_min_confidence: f32,
    /// Selects the feature extractor backing the per-frame descriptor
    /// stream. `corner` (the default) is the existing
    /// `CornerFeatureExtractor` with raw patch descriptors. `hog` is
    /// `HogLikeFeatureExtractor` — a unit-norm 128-D HOG/SIFT-flavored
    /// descriptor that raises descriptor signal-to-noise at the cost
    /// of ~4× CPU per keypoint. Motivation: EuRoC's keyframe-
    /// registration floor at the strict pose-jump gate is dominated by
    /// descriptor match recall, not match-filter strictness, so a
    /// stronger descriptor is the natural next lever after Phase-10's
    /// `--cross-check-matcher` proved precision-only doesn't help.
    pub(super) feature_extractor: FeatureExtractorKind,
    /// Companion to `feature_extractor = hog`: maximum keypoints
    /// retained per frame from the HOG corner-response NMS. Larger
    /// numbers favour recall at the cost of CPU; smaller numbers
    /// favour speed.
    pub(super) hog_max_features: usize,
    /// Companion to `feature_extractor = hog`: minimum corner response
    /// score to admit a keypoint candidate (before NMS). Lower numbers
    /// admit more (often weaker) candidates.
    pub(super) hog_min_corner_score: f32,
    /// Companion to `feature_extractor = hog`: enable SIFT-style
    /// dominant-gradient-orientation alignment of the HOG bins (off by
    /// default — adds variance without clear gain on a forward-driving
    /// camera, but worth trying on rotation-heavy EuRoC drone takeoffs).
    pub(super) hog_orient: bool,
    /// When `Some(m)`, overrides `KeyframePolicyConfig.min_translation`
    /// (the cumulative-from-last-keyframe translation a successful
    /// frame must clear to register as a new keyframe). The library
    /// default is `1.0 m`, calibrated for KITTI-class driving cadence;
    /// on EuRoC's slower indoor / takeoff motion plus the strict
    /// `--max-pose-jump-meters 0.2` gate, `1.0 m` is too coarse to
    /// register more than the seed keyframe in 400 frames. Values
    /// around `0.05–0.2 m` register a useful keyframe stream on EuRoC
    /// without flooding the map.
    pub(super) keyframe_min_translation: Option<f64>,
    /// Override `KeyframePolicyConfig.min_frame_id_gap`. Lower values
    /// allow earlier rescue keyframes when tracking quality drops soon
    /// after a promotion; `None` preserves the library default.
    pub(super) keyframe_min_frame_gap: Option<u64>,
    /// Force a keyframe after this many frames when tracking quality passes,
    /// even if estimated translation remains below the metric threshold.
    pub(super) keyframe_max_frame_gap: Option<u64>,
    /// When `Some(r)`, also promotes a keyframe after the normal
    /// frame-id gap if the current PnP inlier count drops to `r` times
    /// the last keyframe's tracked-landmark count. This mirrors the
    /// ORB-SLAM-style "tracked local-map points dropped" trigger while
    /// keeping the existing metric translation threshold for A/B.
    pub(super) keyframe_tracked_landmark_ratio: Option<f64>,
    /// Reference/current-count floor for `keyframe_tracked_landmark_ratio`.
    /// Prevents sparse startup frames or nearly-lost frames from making
    /// the ratio trigger look artificially severe.
    pub(super) keyframe_min_tracked_landmarks_for_ratio: usize,
    /// When `Some(n)`, rejects promoting a frame to a keyframe (even
    /// after it clears the frame-id-gap and translation gates) if its PnP
    /// inlier count is below `n`. Targets the EuRoC MH_01 frame-1096
    /// failure mode where a garbage localization (4 inliers) that
    /// survives the pose-prior gate gets promoted to a keyframe and
    /// poisons the local map / covisibility graph for subsequent frames.
    /// `None` (the default) preserves legacy always-promote behaviour.
    pub(super) keyframe_min_inliers: Option<usize>,
    /// Companion to `--keyframe-min-inliers`: also requires the PnP
    /// inlier ratio to be at least this value for a keyframe promotion to
    /// go through. `None` (the default) preserves legacy behaviour.
    pub(super) keyframe_min_inlier_ratio: Option<f64>,
    /// Override `VisualInertialInitializerConfig.min_samples`; default
    /// `50`. Lowering accelerates VI-init promotion so more KFs
    /// register POST-init and feed the local-VI-BA chain.
    pub(super) vi_init_min_samples: Option<usize>,
    /// Override `VisualInertialInitializerConfig.min_stationary_window_seconds`;
    /// default `0.5 s`. Pairs with `--vi-init-min-samples` to bring
    /// VI-init forward.
    pub(super) vi_init_min_stationary_window_seconds: Option<f64>,
    /// When `true`, sets `OnlineSlamConfig.keep_pre_promotion_imu_factors
    /// = true` so IMU factors staged on keyframes registered BEFORE the
    /// auto-bootstrap stage promotes still flow downstream (into the
    /// local-VI-BA factor history). The factors carry placeholder bias
    /// linearisations; the BA's Gauss-Newton iterations are expected
    /// to absorb the resulting bias error. Empirically on EuRoC this
    /// is the lever that unblocks local-VI-BA's per-trigger
    /// keyframe-state mirror — the strict (default) gate leaves the
    /// chain at 1 trigger per 400 frames, the relaxed gate lifts it
    /// to ~7. See `docs/motion_based_vi_alignment.md` §Phase-13.
    pub(super) keep_pre_promotion_imu_factors: bool,
    /// When `Some((g, a))`, sets
    /// `OnlineSlamLocalBaConfig.relinearise_imu_factor_bias_thresholds`
    /// so banked IMU factors get re-linearised in-place before each BA
    /// pass if their stored `bias_*_linearisation` drifts more than `g`
    /// rad/s (gyro) or `a` m/s² (accel) from the per-keyframe state
    /// estimate. Pairs with `--keep-pre-promotion-imu-factors`: without
    /// the re-linearisation, banked pre-promotion factors carry their
    /// placeholder zero bias_linearisation through the post-promotion
    /// BA, and the linear bias-correction approximation breaks down on
    /// non-trivial biases — producing unphysical mirror velocities even
    /// when rigid ATE improves. See `docs/motion_based_vi_alignment.md`
    /// §Phase-14.
    pub(super) relinearise_imu_factor_bias_thresholds: Option<(f64, f64)>,
    /// Directory holding pre-exported SuperPoint mono features
    /// (`frame_NNNNNN_features.txt` from
    /// `scripts/export_superpoint_lightglue.py --mono-dir <cam0_dir>`)
    /// to replay during the demo run. Required when
    /// `--feature-extractor superpoint-offline` is selected; otherwise
    /// ignored.
    pub(super) superpoint_features_dir: Option<PathBuf>,
    /// Optional companion directory holding pre-exported SuperPoint
    /// cam1 features. When provided alongside
    /// `--superpoint-features-dir <cam0_dir>` and
    /// `--feature-extractor superpoint-offline`, the offline extractor
    /// loads both feature streams and `--stereo-bootstrap` is
    /// permitted. The stereo bootstrap path triangulates cam0 ↔ cam1
    /// SuperPoint matches via the cam0/cam1 extrinsics, producing
    /// metric-depth landmarks that replace the fixed `bootstrap_depth`
    /// back-projection. Phase-17 lever for the Phase-15 bootstrap-
    /// depth bottleneck.
    pub(super) superpoint_cam1_features_dir: Option<PathBuf>,
    /// Phase-27: path to a SuperPoint ONNX model file. Required when
    /// `--feature-extractor superpoint-onnx` is selected. The model
    /// must follow the LightGlue-ONNX-style I/O contract documented
    /// in `docs/superpoint_onnx_runtime_plan.md`: input `image:
    /// (1, 1, H, W) f32`, outputs `keypoints (N, 2) i64`, `scores
    /// (N,) f32`, `descriptors (256, N) f32` (or `(N, 256)`). The
    /// extractor only runs when `visloc-vision` was built with
    /// `--features onnx-inference`; otherwise the first `extract()`
    /// call fails with `FeatureDisabled`.
    pub(super) superpoint_onnx_model: Option<PathBuf>,
    /// ONNX execution-provider policy. `cuda` is strict and aborts at model
    /// loading when CUDA cannot be registered, preventing an evaluation from
    /// silently reporting CPU inference as a GPU run.
    pub(super) superpoint_onnx_backend: SuperPointOnnxBackendArg,
    /// Export one L2-normalized mean local descriptor per processed cam0
    /// frame to `frame_appearance_descriptors.csv`. This is a diagnostic
    /// input for offline retrieval-candidate recall scripts; it does not
    /// affect tracking, mapping, or recovery.
    pub(super) export_frame_appearance_descriptors: bool,
    /// Phase-16 lever. When `true`, sets
    /// `OnlineSlamLocalBaConfig.run_at_vi_init_promotion = true` so
    /// the local-VI-BA pass fires at the same `process_frame` that
    /// promotes VI-init, consuming the banked pre-promotion factors
    /// without waiting for the next keyframe registration. Useful
    /// when the visual tracker is fragile post-promotion so the next
    /// keyframe arrives late or never. Default `false` preserves
    /// Phase-13/14 cadence.
    pub(super) run_local_vi_ba_at_vi_init_promotion: bool,
    /// Phase-23 #1 lever. When `true`, enables the
    /// relocalization-on-tracker-death stage in
    /// [`OnlineSlamConfig::relocalization`]. On every frame whose
    /// primary tracking fails, the pipeline runs a separate
    /// [`LocalizationPipeline`] against the full map and overrides
    /// the tracker's state when the recovered solution clears
    /// `--relocalization-min-inliers` / `--relocalization-min-inlier-ratio`
    /// / `--relocalization-max-reprojection-error`. Default `false`
    /// preserves the universal-cliff behaviour from Phase-21.
    pub(super) relocalization_enabled: bool,
    /// Acceptance threshold for the relocalization recovery PnP solve.
    pub(super) relocalization_min_inliers: usize,
    /// Acceptance threshold for the relocalization recovery PnP solve.
    pub(super) relocalization_min_inlier_ratio: f64,
    /// Optional acceptance threshold for the relocalization recovery
    /// PnP solve.
    pub(super) relocalization_max_reprojection_error: Option<f64>,
    /// Phase-23 #4 lever. When `--motion-model adaptive-imu-pose` is
    /// selected, this is the number of consecutive failed-tracking
    /// frames under the IMU branch that triggers a switch to the
    /// constant-pose branch. Default `2` reacts fast at the cliff
    /// transition; raise to dampen oscillation on noisy regimes.
    pub(super) adaptive_motion_failures_to_switch_to_pose: usize,
    /// Phase-23 #4 lever. Number of consecutive successful frames
    /// under the constant-pose branch that triggers a switch back to
    /// the IMU branch. Default `5` biases for staying in the
    /// (cliff-survival) pose mode once it has fired.
    pub(super) adaptive_motion_successes_to_switch_to_imu: usize,
    /// Phase-24 / Phase-25 lever. Selects which
    /// [`ImuVelocityRefreshPolicy`] the adaptive wrapper uses at every
    /// switch-back-to-IMU event. `FiniteDifference` (default,
    /// Phase-24 behavior) recomputes the IMU's `velocity_world` from
    /// a single finite-difference of the two most recent successful
    /// visual poses. `ZeroReset` (Phase-25 #1) zeroes the velocity
    /// instead. `ThreePoseSmoother` (Phase-25 #2) averages two
    /// finite-differences over the three most recent successful
    /// poses. `None` recovers the Phase-23 #4 no-refresh behavior for
    /// A/B testing. Configurable via
    /// `--adaptive-motion-refresh-policy`.
    pub(super) adaptive_motion_imu_velocity_refresh_policy: ImuVelocityRefreshPolicy,
    /// Phase-23 #2 lever. When `true`, the bootstrap map drops every
    /// keypoint that did NOT receive a stereo-triangulated cam0↔cam1
    /// depth, refusing to fall back to `--bootstrap-depth`. The map
    /// is smaller (only the 30–43 % of cam0 keypoints that survived
    /// stereo matching) but every landmark has a real metric depth.
    /// Empirically, the Phase-23 #1 sweep showed that cross-attitude
    /// HOG descriptor mismatch dominates the recovery-PnP failure
    /// mode; dropping the stale 4-m-depth fallback landmarks reduces
    /// the false-positive matches that the recovery localizer would
    /// otherwise have to filter out. Default `false` preserves the
    /// legacy mixed-depth bootstrap.
    pub(super) stereo_bootstrap_strict: bool,
    /// Phase-23 #1b lever. When `Some(radius_m)`, the recovery PnP
    /// uses the tracker's motion-model pose prior as a warm-start
    /// hypothesis AND filters the candidate landmark set to those
    /// within `radius_m` of the prior's camera centre. Empirically on
    /// EuRoC the Phase-23 #1 sweep showed full-map BruteForce PnP
    /// without a pose prior accepts < 0.3 % of recovery attempts at
    /// the strict gate; threading the prior in short-circuits the
    /// matcher to the local landmark set + seeds RANSAC, both of
    /// which should lift recovery quality. `None` (default) preserves
    /// the no-prior global path from Phase-23 #1.
    pub(super) relocalization_pose_prior_radius_meters: Option<f64>,
    /// Phase-26 #4a active-frontier submap selection. When
    /// `Some(window)`, the recovery PnP's descriptor store is
    /// restricted to landmarks observed by any of the most-recent
    /// `window` keyframes in the map. Targets the Phase-26 #2
    /// V1_01 false-positive failure mode (full-map recovery
    /// accepts wrong-scale solutions because the candidate set
    /// spans the whole map). `None` preserves Phase-23 #1
    /// full-map behaviour.
    pub(super) relocalization_recent_keyframe_window: Option<usize>,
    /// Phase-26 #4b post-acceptance IMU sanity check. When
    /// `Some(max_translation_m)`, recoveries that pass the
    /// inlier-count / inlier-ratio / reprojection-error gates are
    /// further rejected if the recovered camera centre is more
    /// than `max_translation_m` from the tracker's motion-model
    /// prediction. Filters wrong-scale recoveries identified by
    /// Phase-26 #2 diagnosis. `None` preserves Phase-23 #1
    /// no-IMU-sanity-check behaviour.
    pub(super) relocalization_max_translation_from_imu_prediction_meters: Option<f64>,
    /// Minimum frame-id gap between relocalization attempts. Default
    /// `1` preserves the existing every-failed-frame behaviour.
    pub(super) relocalization_attempt_interval_frames: u64,
    /// Optional cap on consecutive failed relocalization attempts while
    /// the primary tracker remains lost. `None` keeps the legacy
    /// unbounded retry behaviour.
    pub(super) relocalization_max_consecutive_failed_attempts: Option<u64>,
    /// Optional pose-continuity gate on accepted relocalization
    /// candidates. When `Some(max_m_per_frame)`, the recovered camera
    /// centre must stay within this translation-per-frame budget from
    /// the last successful tracker pose. `None` preserves the existing
    /// acceptance behaviour.
    pub(super) relocalization_max_translation_per_frame_from_last_success_meters: Option<f64>,
    /// Optional lower bound for recovery inlier median-depth ratio vs
    /// the last successful pose, measured on the same inlier landmarks.
    pub(super) relocalization_min_inlier_depth_median_ratio_to_last_success: Option<f64>,
    /// Optional upper bound for recovery inlier median-depth ratio vs
    /// the last successful pose.
    pub(super) relocalization_max_inlier_depth_median_ratio_to_last_success: Option<f64>,
    /// Enable covisibility-local recovery descriptor-store selection
    /// and cap the number of neighbor keyframes. `None` keeps the
    /// existing full-map / recent-window recovery store.
    pub(super) relocalization_covisibility_max_keyframes: Option<usize>,
    /// Minimum shared landmarks for a neighbor keyframe to enter the
    /// recovery covisibility store.
    pub(super) relocalization_covisibility_min_shared: usize,
    /// Minimum descriptor count required to use the recovery
    /// covisibility store; otherwise the demo falls back to the
    /// broader recovery descriptor store.
    pub(super) relocalization_covisibility_min_landmarks: usize,
    /// Retry with the broader full-map / recent-window recovery store
    /// when the covisibility-local first pass fails the acceptance
    /// gates.
    pub(super) relocalization_covisibility_broader_fallback: bool,
    /// Minimum frame-id gap between broader descriptor-store retries.
    pub(super) relocalization_covisibility_broader_fallback_interval_frames: u64,
    /// Also run the broader store when the covisibility-local first
    /// pass succeeds, then keep the accepted result with the stronger
    /// inlier / reprojection score.
    pub(super) relocalization_covisibility_compare_broader_store: bool,
    /// Enable appearance-retrieval recovery descriptor-store selection
    /// and cap the number of retrieved keyframes. `None` keeps this
    /// policy disabled.
    pub(super) relocalization_appearance_max_keyframes: Option<usize>,
    /// Optional cap on ranked appearance candidates written to
    /// `relocalization_appearance_candidates.csv`. This can be higher
    /// than `relocalization_appearance_max_keyframes` to evaluate
    /// retrieval recall@K without increasing recovery-PnP cost.
    pub(super) relocalization_appearance_candidate_log_limit: Option<usize>,
    /// Minimum mean-descriptor cosine similarity for a retrieved
    /// keyframe to seed recovery.
    pub(super) relocalization_appearance_min_similarity: f32,
    /// Exclude keyframes within this frame-id gap from appearance
    /// retrieval.
    pub(super) relocalization_appearance_exclude_recent_frame_gap: u64,
    /// Minimum descriptor count required to use the appearance-retrieval
    /// recovery store.
    pub(super) relocalization_appearance_min_landmarks: usize,
    /// Retry with the broader full-map / recent-window recovery store
    /// when the appearance first pass fails the acceptance gates.
    pub(super) relocalization_appearance_broader_fallback: bool,
    /// Minimum frame-id gap between broader descriptor-store retries
    /// after an appearance first pass.
    pub(super) relocalization_appearance_broader_fallback_interval_frames: u64,
    /// Also run the broader store when the appearance first pass
    /// succeeds, then keep the accepted result with the stronger score.
    pub(super) relocalization_appearance_compare_broader_store: bool,
    /// Number of consecutive recovery hypotheses required before the
    /// tracker accepts relocalization. `1` preserves immediate accept.
    pub(super) relocalization_confirmation_required_recoveries: usize,
    /// Optional max translation per frame between consecutive recovery
    /// hypotheses inside the confirmation window.
    pub(super) relocalization_confirmation_max_translation_per_frame_meters: Option<f64>,
    /// When `Some(n)`, re-seed tracking from the last accepted pose and the
    /// current calibrated stereo pair after `n` consecutive lost frames once
    /// relocalization (if enabled) has not recovered. This is causal and
    /// GT-free. By default it continues in the existing gauge;
    /// `rebootstrap_independent_submap` starts a fresh Atlas submap instead.
    pub(super) rebootstrap_after_lost_frames: Option<usize>,
    /// Minimum frame gap between accepted re-bootstraps. Default `60`.
    pub(super) rebootstrap_cooldown_frames: usize,
    /// Start every accepted stereo re-bootstrap in a fresh local gauge and
    /// retain the previous map in a [`MapAtlas`]. Until a verified cross-map
    /// SE(3) bridge is available, local poses are exported only to
    /// `submap_trajectory.csv` and never mixed into the global trajectory.
    pub(super) rebootstrap_independent_submap: bool,
    /// Retained current/boundary source keyframes tried per Atlas-level
    /// broader recovery cycle.
    pub(super) atlas_broader_recovery_max_source_keyframes: usize,
    /// Run the more expensive retained-keyframe Atlas search every N logical
    /// bridge attempts. Other attempts use only the current keyframe.
    pub(super) atlas_broader_recovery_interval_attempts: usize,
    /// Opt into the online loop-closure + pose-graph refinement stage
    /// (`OnlineSlamConfig::pose_graph_refinement`). When `true`, the
    /// pipeline mirrors registered keyframe poses into a running pose
    /// graph, verifies every `detect_loop_closure_candidates` output with
    /// an essential-matrix verifier, and periodically re-solves the graph
    /// and writes the optimised poses back into the map. Off by default so
    /// baseline runs stay byte-identical.
    pub(super) pose_graph_refinement_enabled: bool,
    /// Minimum newly-verified loop-closure constraints that must
    /// accumulate before a fresh pose-graph solve fires. Mirrors
    /// `OnlineSlamLoopClosureRefinementConfig::trigger_every_new_constraints`.
    /// Defaults to `1`. Only meaningful when
    /// `pose_graph_refinement_enabled` is set.
    pub(super) pose_graph_refinement_trigger_every: usize,
    /// Optional fixed isotropic loop-edge weight. `None` (default) preserves
    /// the legacy raw-PnP-inlier-count weight. A value such as `1.0` prevents
    /// correspondence count from being mistaken for calibrated inverse pose
    /// covariance relative to unit-weight sequential edges.
    pub(super) pose_graph_refinement_fixed_loop_edge_weight: Option<f64>,
    /// Use covariance-aware anisotropic 6×6 information for PnP loop edges.
    pub(super) pose_graph_refinement_loop_pose_information: bool,
    /// Spectral cap applied to covariance-derived loop/sequential information.
    pub(super) pose_graph_refinement_loop_pose_information_max_eigenvalue: f64,
    /// Relative strength applied to loop matrices only; sequential PnP edge
    /// information remains unchanged.
    pub(super) pose_graph_refinement_loop_pose_information_loop_edge_scale: f64,
    /// Enable the Graduated Non-Convexity robust back-end solve
    /// (`OnlineSlamLoopClosureRefinementConfig::gnc`, using
    /// `gnc::GncConfig::default()`) instead of the plain iterative
    /// M-estimator solve. Off by default. Only meaningful when
    /// `pose_graph_refinement_enabled` is set.
    pub(super) pose_graph_refinement_gnc: bool,
    /// Enable the Pairwise Consistency Maximization front-end screen
    /// (`OnlineSlamLoopClosureRefinementConfig::pcm`, using
    /// `pcm::PcmConfig::default()`). Off by default. Only meaningful when
    /// `pose_graph_refinement_enabled` is set.
    pub(super) pose_graph_refinement_pcm: bool,
    /// Disable PCM's additional single-loop-vs-odometry pre-filter and rely on
    /// pairwise consistency between verified loop measurements. This matches
    /// the paper's drift-tolerant consistency-set test; the legacy individual
    /// gate remains enabled by default.
    pub(super) pose_graph_refinement_pcm_require_individual: bool,
    /// Optional covariance-based chi-squared gate on verified loop
    /// closures. Mirrors
    /// `OnlineSlamLoopClosureRefinementConfig::covariance_gate`. `None`
    /// (default) applies no metric gate. Only meaningful when
    /// `pose_graph_refinement_enabled` is set.
    pub(super) pose_graph_refinement_covariance_gate: Option<f64>,
    /// Which geometric verifier the refinement stage runs on loop-closure
    /// candidates (`OnlineSlamLoopClosureRefinementConfig::verifier`).
    /// `"essential"` (default) keeps the original scale-free two-view
    /// verifier, whose accepted constraints all carry the same
    /// `default_translation_scale` translation regardless of the true
    /// baseline. `"pnp"` verifies on 2D-3D correspondences against the map's
    /// triangulated landmarks instead, so accepted constraints carry the
    /// metric relative translation. Only meaningful when
    /// `pose_graph_refinement_enabled` is set.
    pub(super) pose_graph_refinement_verifier: LoopRefinementVerifierKind,
    /// Opt into the appearance-based long-range loop candidate source
    /// (`OnlineSlamLoopClosureRefinementConfig::appearance_candidates`).
    /// Diagnoses the short-range hypothesis for why PnP-verified loop
    /// closures don't improve ATE: the default shared-landmark detector can
    /// only propose candidates that still reference the SAME map landmark
    /// ids the current frame's inliers carry, which drift re-triangulation
    /// makes inherently short-range. This stream instead ranks past
    /// keyframes by appearance (independent of shared ids) and PnP-verifies
    /// them via descriptor-matched 2D-3D correspondences. Off by default so
    /// baseline runs stay byte-identical. Only meaningful when
    /// `pose_graph_refinement_enabled` is set.
    pub(super) pose_graph_refinement_appearance_loops: bool,
    /// Minimum keyframe-id gap for an appearance candidate (mirrors
    /// `LoopAppearanceCandidateConfig::min_keyframe_id_gap`). Defaults to
    /// `150` — comfortably above the shared-landmark detector's span, so
    /// this stream only ever proposes genuinely long-range loops. Only
    /// meaningful when `pose_graph_refinement_appearance_loops` is set.
    pub(super) pose_graph_refinement_appearance_min_gap: u64,
    /// Maximum number of appearance-ranked candidate keyframes verified per
    /// frame (mirrors `LoopAppearanceCandidateConfig::max_candidates_per_frame`).
    /// Defaults to `3`. Only meaningful when
    /// `pose_graph_refinement_appearance_loops` is set.
    pub(super) pose_graph_refinement_appearance_max_candidates: usize,
    /// Minimum PnP RANSAC inlier count an appearance candidate must produce
    /// to be admitted (mirrors `LoopAppearanceCandidateConfig::pnp_verifier`'s
    /// `min_inliers`). Defaults to `30` — higher than the shared-landmark
    /// PnP path's default, since a false long-range loop closure is
    /// catastrophic. Only meaningful when
    /// `pose_graph_refinement_appearance_loops` is set.
    pub(super) pose_graph_refinement_appearance_min_inliers: usize,
    /// Separate current keyframes required to confirm one covisible region.
    pub(super) pose_graph_refinement_appearance_confirmation_keyframes: usize,
    /// Consecutive keyframe misses tolerated before a pending region resets.
    pub(super) pose_graph_refinement_appearance_confirmation_max_misses: usize,
    /// Optional post-PnP projection-window rematch radius in pixels.
    pub(super) pose_graph_refinement_appearance_projection_radius_px: Option<f64>,
    /// Minimum projection-guided matches required for refined PnP.
    pub(super) pose_graph_refinement_appearance_projection_min_matches: usize,
    /// Persist accepted appearance-PnP inliers as cross-loop observations.
    pub(super) pose_graph_refinement_fuse_loop_observations: bool,
    /// Run covisibility welding BA after a solved, fused loop.
    pub(super) pose_graph_refinement_loop_welding_ba: bool,
    /// Propagate each solved keyframe's pose correction to its anchored
    /// landmarks and to the tracker's continuation state
    /// (`OnlineSlamLoopClosureRefinementConfig::propagate_corrections`).
    /// Off by default so baseline pose-graph-refinement runs stay
    /// byte-identical (only keyframe poses move on write-back). Only
    /// meaningful when `pose_graph_refinement_enabled` is set.
    pub(super) pose_graph_refinement_propagate: bool,
    /// Which pose-graph solver backs the periodic PGO trigger
    /// (`OnlineSlamLoopClosureRefinementConfig::solver`). `"se3"`
    /// (default) keeps the original rigid solve, byte-identical to
    /// today's behaviour. `"sim3"` opts into the `Sim(3)` pose graph
    /// instead, which can absorb scale drift (e.g. from a learned
    /// projection-tracking front end) a rigid solve smears across the
    /// whole trajectory. Only meaningful when
    /// `pose_graph_refinement_enabled` is set. Combining `"sim3"` with
    /// `--pose-graph-refinement-gnc` prints a warning: `Sim3PoseGraph`
    /// has no GNC variant yet, so the flag is silently ignored on that
    /// path.
    pub(super) pose_graph_refinement_solver: LoopRefinementSolverKind,
}

/// CLI-selectable mirror of [`LoopRefinementVerifier`]'s variants. A
/// separate enum (rather than dispatching straight to the library type) so
/// this demo can carry its own `Display`/parsing without coupling to the
/// library enum's exact shape.
#[cfg(feature = "image-io")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LoopRefinementVerifierKind {
    Essential,
    Pnp,
}

/// CLI-selectable mirror of [`LoopRefinementSolver`]'s variants (minus its
/// `Sim3` payload config, which this demo always constructs from
/// `Sim3PoseGraphConfig::default()` — no CLI knobs for the Sim3 solver's
/// own LM settings yet).
#[cfg(feature = "image-io")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LoopRefinementSolverKind {
    Se3,
    Sim3,
}

#[cfg(feature = "image-io")]
pub(super) fn parse_args() -> Result<CliArgs, Box<dyn std::error::Error>> {
    let mut euroc_dir: Option<PathBuf> = None;
    let mut out_dir = PathBuf::from("target/euroc_online_slam_vi_image_demo");
    let mut max_frames: usize = 400;
    let mut adaptive_keypoints: bool = false;
    let mut adaptive_frame_budget_ms: f64 = 50.0;
    let mut gravity_world = Vector3::new(0.0, 0.0, -9.81);
    let mut vi_init_max_wait_seconds: f64 = 5.0;
    let mut vi_init_try_initialize_on_every_frame: bool = false;
    let mut vi_init_gyro_std_limit: Option<f64> = None;
    let mut vi_init_mean_gyro_magnitude_limit: Option<f64> = None;
    let mut vi_init_accel_std_limit: Option<f64> = None;
    let mut bootstrap_depth_meters: f64 = 4.0;
    let mut corner_max_features: usize = 1500;
    let mut corner_min_score: f32 = 0.02;
    let mut corner_descriptor_radius: usize = 5;
    let mut undistort: bool = true;
    let mut stereo_bootstrap: bool = true;
    let mut stereo_bootstrap_strict: bool = true;
    let mut adaptive_motion_failures_to_switch_to_pose: usize = 2;
    let mut adaptive_motion_successes_to_switch_to_imu: usize = 5;
    let mut adaptive_motion_imu_velocity_refresh_policy: ImuVelocityRefreshPolicy =
        ImuVelocityRefreshPolicy::default();
    let mut motion_vi_init_enabled: bool = false;
    let mut motion_vi_init_after_static_give_up: bool = false;
    let mut motion_vi_init_from_configured_bias: bool = false;
    let mut motion_vi_init_min_keyframes: usize = 3;
    let mut motion_vi_init_min_translation_meters: f64 = 0.15;
    let mut motion_vi_init_recover_scale: bool = false;
    let mut local_vi_ba_enabled: bool = false;
    let mut observation_confidence_ba_enabled: bool = false;
    let mut local_vi_ba_marginalization: bool = false;
    let mut local_vi_ba_use_sqrt_window_marginalization: bool = true;
    let mut local_vi_ba_window_size: usize = OnlineSlamLocalBaConfig::default().window_size;
    let mut basalt_config_path: Option<PathBuf> = None;
    let mut optical_flow_config =
        visloc_rs::vision::optical_flow::BasaltOpticalFlowConfig::default();
    let mut local_vi_ba_initial_prior_std_devs: Option<(f64, f64, f64)> = None;
    let mut local_vi_ba_freeze_biases_above: Option<f64> = None;
    let mut local_vi_ba_reject_writeback_above: Option<f64> = None;
    let mut local_vi_ba_reject_final_imu_nis_per_dof_above: Option<f64> = None;
    let mut local_vi_ba_reject_velocity_above_mps: Option<f64> = None;
    let mut local_vi_ba_reject_pose_translation_above_meters: Option<f64> = None;
    let mut local_vi_ba_reject_pose_rotation_above_degrees: Option<f64> = None;
    let mut local_vi_ba_reject_gyro_bias_above_rad_s: Option<f64> = None;
    let mut local_vi_ba_reject_accel_bias_above_mps2: Option<f64> = None;
    let mut local_vi_ba_adaptive_velocity_gate: bool = false;
    let default_adaptive_velocity_gate = AdaptiveVelocityGateConfig::default();
    let mut local_vi_ba_adaptive_velocity_quantile: f64 =
        default_adaptive_velocity_gate.reference_quantile;
    let mut local_vi_ba_adaptive_velocity_multiplier: f64 =
        default_adaptive_velocity_gate.multiplier;
    let mut local_vi_ba_adaptive_velocity_margin_mps: f64 =
        default_adaptive_velocity_gate.margin_mps;
    let mut local_vi_ba_adaptive_velocity_min_mps: f64 =
        default_adaptive_velocity_gate.min_threshold_mps;
    let mut local_vi_ba_adaptive_velocity_max_mps: Option<f64> =
        default_adaptive_velocity_gate.max_threshold_mps;
    let mut local_vi_ba_adaptive_velocity_min_references: usize =
        default_adaptive_velocity_gate.min_reference_count;
    // Keep motion-VI-init physically plausible by default. Without these
    // gates, a numerically-valid but non-physical initialisation can seed
    // extreme velocity/bias and destabilise downstream VI-BA.
    let mut motion_vi_init_max_velocity_mps: Option<f64> = Some(8.0);
    let mut motion_vi_init_max_gyro_bias_rad_s: Option<f64> = Some(0.5);
    let mut motion_vi_init_max_accel_bias_mps2: Option<f64> = Some(3.0);
    let mut motion_vi_init_max_imu_nis_per_dof: Option<f64> = Some(20_000.0);
    let mut motion_vi_init_max_rotation_residual_rms_rad: Option<f64> = None;
    let mut motion_vi_init_max_velocity_residual_rms_mps: Option<f64> = None;
    let mut motion_vi_init_max_position_residual_rms_meters: Option<f64> = None;
    let mut vi_bias_release_min_keyframes: Option<usize> = None;
    let mut vi_bias_release_min_translation_meters: Option<f64> = None;
    let mut motion_vi_init_estimate_gravity: bool = false;
    let mut motion_vi_init_no_estimate_gravity: bool = false;
    let mut motion_vi_init_max_gravity_norm_deviation: Option<f64> = None;
    let mut motion_vi_init_estimate_gyro_bias: bool = false;
    let mut motion_vi_init_no_estimate_gyro_bias: bool = false;
    let mut covisibility_local_map_max_keyframes: Option<usize> = None;
    let mut covisibility_local_map_min_shared: usize = 15;
    let mut covisibility_local_ba_enabled: bool = false;
    let mut sparse_factor_graph_enabled: bool = false;
    let mut covisibility_local_ba_min_keyframes: usize = 3;
    let mut covisibility_local_ba_max_keyframes: Option<usize> = None;
    let mut covisibility_local_ba_motion_vi_raw_activation: Option<(f64, f64, f64)> = None;
    let mut covisibility_local_ba_max_seed_landmarks_for_activation: Option<usize> = None;
    let mut covisibility_local_ba_trigger_every: usize = 1;
    let mut covisibility_local_ba_max_neighbor_keyframes: usize = 10;
    let mut covisibility_local_ba_min_shared: usize = 15;
    let mut covisibility_local_ba_max_boundary_keyframes: usize = 10;
    let mut covisibility_local_ba_min_boundary_observations: usize = 5;
    let mut covisibility_local_ba_fallback_min_boundary_observations: Option<usize> = None;
    let mut covisibility_local_ba_max_landmarks: Option<usize> = None;
    let mut covisibility_local_ba_min_active_observations: usize = 1;
    let mut covisibility_local_ba_outlier_threshold_px: Option<f64> = Some(5.0);
    let mut covisibility_local_ba_remove_outliers: bool = false;
    let mut covisibility_local_ba_general_stereo: bool = false;
    let mut covisibility_local_ba_general_stereo_max_right_reprojection_px: Option<f64> = Some(5.0);
    let mut covisibility_local_ba_max_outlier_observation_ratio: Option<f64> = None;
    let mut covisibility_local_ba_boundary_support_min_optimized_keyframes: Option<usize> = None;
    let mut covisibility_local_ba_boundary_support_min_fixed_keyframes: usize = 0;
    let mut covisibility_local_ba_max_behind_camera_ratio: Option<f64> = None;
    let mut covisibility_local_ba_min_fixed_to_optimized_ratio: Option<f64> = None;
    let mut covisibility_local_ba_max_pose_translation_correction_m: Option<f64> = None;
    let mut covisibility_local_ba_max_pose_rotation_correction_deg: Option<f64> = None;
    let mut covisibility_local_ba_anchor_weight: Option<f64> = None;
    let mut stereo_landmark_replenish: bool = false;
    let mut stereo_landmark_replenish_no: bool = false;
    let mut stereo_landmark_replenish_max_per_frame: usize = 100;
    let mut stereo_landmark_replenish_max_per_frame_overridden: bool = false;
    let mut stereo_landmark_replenish_anchor_match_radius_px: Option<f64> = None;
    let mut stereo_landmark_replenish_anchor_match_radius_px_overridden: bool = false;
    let mut stereo_landmark_replenish_anchor_max_descriptor_distance: Option<f32> = None;
    let mut stereo_landmark_replenish_duplicate_radius_px: Option<f64> = None;
    let mut stereo_landmark_replenish_duplicate_radius_px_overridden: bool = false;
    let mut stereo_landmark_replenish_min_parallax_deg: Option<f64> = None;
    let mut stereo_landmark_replenish_min_parallax_deg_overridden: bool = false;
    let mut stereo_landmark_replenish_min_depth_meters: Option<f64> = None;
    let mut stereo_landmark_replenish_min_depth_meters_overridden: bool = false;
    let mut stereo_landmark_replenish_max_depth_meters: Option<f64> = None;
    let mut stereo_landmark_replenish_max_depth_meters_overridden: bool = false;
    let mut max_pose_jump_meters: Option<f64> = None;
    let mut pose_prior_visual_override: bool = false;
    let mut accept_motion_prior_on_failure: bool = false;
    let mut max_consecutive_motion_prior_coasts: Option<usize> = Some(5);
    let mut temporal_landmark_tracking: bool = false;
    let mut temporal_landmark_tracking_overridden: bool = false;
    let mut temporal_landmark_tracking_history_frames: usize = 1;
    let mut pose_jump_gap_scaling: bool = false;
    let mut pose_jump_gap_scaling_max_multiplier: usize = 10;
    let mut tracking_min_inliers: usize = 0;
    let mut tracking_min_inliers_overridden: bool = false;
    let mut tracking_min_inlier_ratio: f64 = 0.0;
    let mut tracking_max_reprojection_error: Option<f64> = None;
    let mut pnp_pose_prior_warm_start: bool = false;
    let mut pnp_pose_prior_warm_start_overridden: bool = false;
    let mut projection_guided_tracking: bool = false;
    let mut projection_search_radius_px: f64 = 15.0;
    let mut projection_widen_factor: f64 = 2.0;
    let mut projection_max_widen_retries: u32 = 2;
    let mut projection_query_landmark_distance_ratio: Option<f32> = None;
    let mut projection_no_local_map_refinement: bool = false;
    let mut projection_refinement_search_radius_px: f64 = 8.0;
    let mut projection_refinement_iterations: u32 = 1;
    let mut projection_refinement_radius_shrink_factor: f64 = 1.0;
    let mut projection_refinement_reassign_correspondences: bool = false;
    let mut projection_refinement_min_inlier_pair_retention_ratio: f64 = 0.0;
    let mut projection_refinement_max_translation_correction_m: Option<f64> = None;
    let mut projection_refinement_max_rotation_correction_deg: Option<f64> = None;
    let mut pnp_reprojection_threshold_px: Option<f64> = None;
    let mut motion_model: MotionModelKind = MotionModelKind::Pose;
    let mut motion_model_overridden: bool = false;
    let mut imu_extrinsic_from_cam0: bool = false;
    let mut imu_motion_model_carry_forward_velocity: bool = false;
    let mut cross_check_matcher: bool = false;
    let mut gpu_matcher_dll: Option<PathBuf> = None;
    let mut mutual_softmax_matcher: bool = false;
    let mut mutual_softmax_matcher_overridden: bool = false;
    let mut mutual_softmax_temperature: f32 = 20.0;
    let mut mutual_softmax_min_confidence: f32 = 0.2;
    let mut feature_extractor: FeatureExtractorKind = FeatureExtractorKind::Corner;
    let mut hog_max_features: usize = 1500;
    let mut hog_min_corner_score: f32 = 0.05;
    let mut hog_orient: bool = false;
    let mut keyframe_min_translation: Option<f64> = None;
    let mut keyframe_min_frame_gap: Option<u64> = None;
    let mut keyframe_max_frame_gap: Option<u64> = None;
    let mut keyframe_tracked_landmark_ratio: Option<f64> = None;
    let mut keyframe_min_tracked_landmarks_for_ratio: usize = 20;
    let mut keyframe_min_inliers: Option<usize> = None;
    let mut keyframe_min_inlier_ratio: Option<f64> = None;
    let mut vi_init_min_samples: Option<usize> = None;
    let mut vi_init_min_stationary_window_seconds: Option<f64> = None;
    let mut keep_pre_promotion_imu_factors: bool = false;
    let mut relinearise_imu_factor_bias_thresholds: Option<(f64, f64)> = None;
    let mut superpoint_features_dir: Option<PathBuf> = None;
    let mut superpoint_cam1_features_dir: Option<PathBuf> = None;
    let mut superpoint_onnx_model: Option<PathBuf> = None;
    let mut superpoint_onnx_backend = SuperPointOnnxBackendArg::CudaThenCpu;
    let mut export_frame_appearance_descriptors: bool = false;
    let mut run_local_vi_ba_at_vi_init_promotion: bool = false;
    let mut relocalization_enabled: bool = false;
    let mut relocalization_min_inliers: usize = 20;
    let mut relocalization_min_inlier_ratio: f64 = 0.3;
    let mut relocalization_max_reprojection_error: Option<f64> = Some(8.0);
    let mut relocalization_pose_prior_radius_meters: Option<f64> = None;
    let mut relocalization_recent_keyframe_window: Option<usize> = None;
    let mut relocalization_max_translation_from_imu_prediction_meters: Option<f64> = None;
    let mut relocalization_attempt_interval_frames: u64 = 1;
    let mut relocalization_max_consecutive_failed_attempts: Option<u64> = None;
    let mut relocalization_max_translation_per_frame_from_last_success_meters: Option<f64> = None;
    let mut relocalization_min_inlier_depth_median_ratio_to_last_success: Option<f64> = None;
    let mut relocalization_max_inlier_depth_median_ratio_to_last_success: Option<f64> = None;
    let mut relocalization_covisibility_max_keyframes: Option<usize> = None;
    let mut relocalization_covisibility_min_shared: usize = 15;
    let mut relocalization_covisibility_min_landmarks: usize = 30;
    let mut relocalization_covisibility_broader_fallback: bool = true;
    let mut relocalization_covisibility_broader_fallback_interval_frames: u64 = 10;
    let mut relocalization_covisibility_compare_broader_store: bool = false;
    let mut relocalization_appearance_max_keyframes: Option<usize> = None;
    let mut relocalization_appearance_candidate_log_limit: Option<usize> = None;
    let mut relocalization_appearance_min_similarity: f32 = 0.2;
    let mut relocalization_appearance_exclude_recent_frame_gap: u64 = 30;
    let mut relocalization_appearance_min_landmarks: usize = 30;
    let mut relocalization_appearance_broader_fallback: bool = true;
    let mut relocalization_appearance_broader_fallback_interval_frames: u64 = 10;
    let mut relocalization_appearance_compare_broader_store: bool = false;
    let mut relocalization_confirmation_required_recoveries: usize = 1;
    let mut relocalization_confirmation_max_translation_per_frame_meters: Option<f64> = None;
    let mut rebootstrap_after_lost_frames: Option<usize> = None;
    let mut rebootstrap_cooldown_frames: usize = 60;
    let mut rebootstrap_independent_submap: bool = false;
    let mut atlas_broader_recovery_max_source_keyframes: usize = 1;
    let mut atlas_broader_recovery_interval_attempts: usize = 5;
    let mut pose_graph_refinement_enabled: bool = false;
    let mut pose_graph_refinement_trigger_every: usize = 1;
    let mut pose_graph_refinement_fixed_loop_edge_weight: Option<f64> = None;
    let mut pose_graph_refinement_loop_pose_information: bool = false;
    let mut pose_graph_refinement_loop_pose_information_max_eigenvalue: f64 = 1.0;
    let mut pose_graph_refinement_loop_pose_information_loop_edge_scale: f64 = 1.0;
    let mut pose_graph_refinement_gnc: bool = false;
    let mut pose_graph_refinement_pcm: bool = false;
    let mut pose_graph_refinement_pcm_require_individual: bool = true;
    let mut pose_graph_refinement_covariance_gate: Option<f64> = None;
    let mut pose_graph_refinement_verifier: LoopRefinementVerifierKind =
        LoopRefinementVerifierKind::Essential;
    let mut pose_graph_refinement_solver: LoopRefinementSolverKind = LoopRefinementSolverKind::Se3;
    let mut pose_graph_refinement_appearance_loops: bool = false;
    let mut pose_graph_refinement_appearance_min_gap: u64 = 150;
    let mut pose_graph_refinement_appearance_max_candidates: usize = 3;
    let mut pose_graph_refinement_appearance_min_inliers: usize = 30;
    let mut pose_graph_refinement_appearance_confirmation_keyframes: usize = 3;
    let mut pose_graph_refinement_appearance_confirmation_max_misses: usize = 2;
    let mut pose_graph_refinement_appearance_projection_radius_px: Option<f64> = None;
    let mut pose_graph_refinement_appearance_projection_min_matches: usize = 50;
    let mut pose_graph_refinement_fuse_loop_observations: bool = false;
    let mut pose_graph_refinement_loop_welding_ba: bool = false;
    let mut pose_graph_refinement_propagate: bool = false;

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
            "--adaptive-keypoints" => {
                adaptive_keypoints = true;
                args.remove(i);
            }
            "--adaptive-frame-budget-ms" => {
                adaptive_frame_budget_ms = args.remove(i + 1).parse()?;
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
            "--vi-init-try-initialize-on-every-frame" => {
                vi_init_try_initialize_on_every_frame = true;
                args.remove(i);
            }
            "--vi-init-max-wait-seconds" => {
                vi_init_max_wait_seconds = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--vi-init-gyro-std-limit" => {
                vi_init_gyro_std_limit = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--vi-init-mean-gyro-magnitude-limit" => {
                vi_init_mean_gyro_magnitude_limit = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--vi-init-accel-std-limit" => {
                vi_init_accel_std_limit = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--bootstrap-depth" => {
                bootstrap_depth_meters = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--corner-max-features" => {
                corner_max_features = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--corner-min-score" => {
                corner_min_score = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--corner-descriptor-radius" => {
                corner_descriptor_radius = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--no-undistort" => {
                undistort = false;
                args.remove(i);
            }
            "--no-stereo-bootstrap" => {
                stereo_bootstrap = false;
                args.remove(i);
            }
            "--stereo-bootstrap-strict" => {
                stereo_bootstrap_strict = true;
                args.remove(i);
            }
            "--allow-fixed-depth-bootstrap" => {
                stereo_bootstrap_strict = false;
                args.remove(i);
            }
            "--adaptive-motion-failures-to-switch-to-pose" => {
                adaptive_motion_failures_to_switch_to_pose = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--adaptive-motion-successes-to-switch-to-imu" => {
                adaptive_motion_successes_to_switch_to_imu = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--adaptive-motion-no-refresh-imu-velocity-on-switch" => {
                // Backward-compat alias preserved through the Phase-25
                // refactor: equivalent to `--adaptive-motion-refresh-policy none`.
                adaptive_motion_imu_velocity_refresh_policy = ImuVelocityRefreshPolicy::None;
                args.remove(i);
            }
            "--adaptive-motion-refresh-policy" => {
                let raw = args.remove(i + 1);
                adaptive_motion_imu_velocity_refresh_policy = match raw.as_str() {
                    "none" => ImuVelocityRefreshPolicy::None,
                    "finite-diff" | "finite-difference" => {
                        ImuVelocityRefreshPolicy::FiniteDifference
                    }
                    "zero-reset" => ImuVelocityRefreshPolicy::ZeroReset,
                    "three-pose-smoother" => ImuVelocityRefreshPolicy::ThreePoseSmoother,
                    other => {
                        return Err(format!(
                            "--adaptive-motion-refresh-policy expects one of \
                             none|finite-diff|zero-reset|three-pose-smoother; got {other:?}"
                        )
                        .into());
                    }
                };
                args.remove(i);
            }
            "--motion-vi-init" => {
                motion_vi_init_enabled = true;
                args.remove(i);
            }
            "--motion-vi-init-after-static-give-up" => {
                motion_vi_init_after_static_give_up = true;
                args.remove(i);
            }
            "--motion-vi-init-from-configured-bias" => {
                motion_vi_init_from_configured_bias = true;
                args.remove(i);
            }
            "--motion-vi-init-min-keyframes" => {
                motion_vi_init_min_keyframes = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--motion-vi-init-min-translation" => {
                motion_vi_init_min_translation_meters = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--motion-vi-init-recover-scale" => {
                motion_vi_init_recover_scale = true;
                args.remove(i);
            }
            "--local-vi-ba" => {
                local_vi_ba_enabled = true;
                args.remove(i);
            }
            "--observation-confidence-ba" => {
                observation_confidence_ba_enabled = true;
                args.remove(i);
            }
            "--local-vi-ba-marginalization" => {
                local_vi_ba_marginalization = true;
                args.remove(i);
            }
            "--use-sqrt-window-marginalization" => {
                local_vi_ba_use_sqrt_window_marginalization = true;
                args.remove(i);
            }
            "--no-sqrt-window-marginalization" => {
                local_vi_ba_use_sqrt_window_marginalization = false;
                args.remove(i);
            }
            "--local-vi-ba-window-size" => {
                local_vi_ba_window_size = args.remove(i + 1).parse()?;
                if local_vi_ba_window_size < 2 {
                    return Err("--local-vi-ba-window-size must be >= 2".into());
                }
                args.remove(i);
            }
            "--basalt-config" => {
                basalt_config_path = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--basalt-euroc-profile" => {
                // Checked-in verbatim Basalt `data/euroc_config.json`.
                basalt_config_path = Some(PathBuf::from("configs/basalt/euroc_config.json"));
                args.remove(i);
            }
            "--local-vi-ba-initial-prior-std-devs" => {
                let value = args.remove(i + 1);
                let parts: Vec<&str> = value.split(',').collect();
                if parts.len() != 3 {
                    return Err("--local-vi-ba-initial-prior-std-devs expects '<velocity_mps>,<gyro_bias_rad_s>,<accel_bias_mps2>'".into());
                }
                let parsed = (
                    parts[0].parse::<f64>()?,
                    parts[1].parse::<f64>()?,
                    parts[2].parse::<f64>()?,
                );
                if [parsed.0, parsed.1, parsed.2]
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
                {
                    return Err(
                        "--local-vi-ba-initial-prior-std-devs values must be finite and > 0".into(),
                    );
                }
                local_vi_ba_initial_prior_std_devs = Some(parsed);
                args.remove(i);
            }
            "--local-vi-ba-freeze-biases-above" => {
                local_vi_ba_freeze_biases_above = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--local-vi-ba-reject-writeback-above" => {
                local_vi_ba_reject_writeback_above = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--local-vi-ba-reject-final-imu-nis-per-dof-above" => {
                local_vi_ba_reject_final_imu_nis_per_dof_above = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--local-vi-ba-reject-velocity-above" => {
                let threshold: f64 = args.remove(i + 1).parse()?;
                if threshold < 0.0 {
                    return Err("--local-vi-ba-reject-velocity-above must be >= 0".into());
                }
                local_vi_ba_reject_velocity_above_mps = Some(threshold);
                args.remove(i);
            }
            "--local-vi-ba-reject-pose-translation-above" => {
                let threshold: f64 = args.remove(i + 1).parse()?;
                if !threshold.is_finite() || threshold < 0.0 {
                    return Err(
                        "--local-vi-ba-reject-pose-translation-above must be finite and >= 0"
                            .into(),
                    );
                }
                local_vi_ba_reject_pose_translation_above_meters = Some(threshold);
                args.remove(i);
            }
            "--local-vi-ba-reject-pose-rotation-above-deg" => {
                let threshold: f64 = args.remove(i + 1).parse()?;
                if !threshold.is_finite() || threshold < 0.0 {
                    return Err(
                        "--local-vi-ba-reject-pose-rotation-above-deg must be finite and >= 0"
                            .into(),
                    );
                }
                local_vi_ba_reject_pose_rotation_above_degrees = Some(threshold);
                args.remove(i);
            }
            "--local-vi-ba-reject-gyro-bias-above" => {
                let threshold: f64 = args.remove(i + 1).parse()?;
                if !threshold.is_finite() || threshold < 0.0 {
                    return Err(
                        "--local-vi-ba-reject-gyro-bias-above must be finite and >= 0".into(),
                    );
                }
                local_vi_ba_reject_gyro_bias_above_rad_s = Some(threshold);
                args.remove(i);
            }
            "--local-vi-ba-reject-accel-bias-above" => {
                let threshold: f64 = args.remove(i + 1).parse()?;
                if !threshold.is_finite() || threshold < 0.0 {
                    return Err(
                        "--local-vi-ba-reject-accel-bias-above must be finite and >= 0".into(),
                    );
                }
                local_vi_ba_reject_accel_bias_above_mps2 = Some(threshold);
                args.remove(i);
            }
            "--local-vi-ba-adaptive-velocity-gate" => {
                local_vi_ba_adaptive_velocity_gate = true;
                args.remove(i);
            }
            "--local-vi-ba-adaptive-velocity-quantile" => {
                let value: f64 = args.remove(i + 1).parse()?;
                if !(0.0..=1.0).contains(&value) {
                    return Err("--local-vi-ba-adaptive-velocity-quantile must be in [0, 1]".into());
                }
                local_vi_ba_adaptive_velocity_quantile = value;
                args.remove(i);
            }
            "--local-vi-ba-adaptive-velocity-multiplier" => {
                let value: f64 = args.remove(i + 1).parse()?;
                if value < 0.0 {
                    return Err("--local-vi-ba-adaptive-velocity-multiplier must be >= 0".into());
                }
                local_vi_ba_adaptive_velocity_multiplier = value;
                args.remove(i);
            }
            "--local-vi-ba-adaptive-velocity-margin" => {
                let value: f64 = args.remove(i + 1).parse()?;
                if value < 0.0 {
                    return Err("--local-vi-ba-adaptive-velocity-margin must be >= 0".into());
                }
                local_vi_ba_adaptive_velocity_margin_mps = value;
                args.remove(i);
            }
            "--local-vi-ba-adaptive-velocity-min" => {
                let value: f64 = args.remove(i + 1).parse()?;
                if value < 0.0 {
                    return Err("--local-vi-ba-adaptive-velocity-min must be >= 0".into());
                }
                local_vi_ba_adaptive_velocity_min_mps = value;
                args.remove(i);
            }
            "--local-vi-ba-adaptive-velocity-max" => {
                let value: f64 = args.remove(i + 1).parse()?;
                if value < 0.0 {
                    return Err("--local-vi-ba-adaptive-velocity-max must be >= 0".into());
                }
                local_vi_ba_adaptive_velocity_max_mps = Some(value);
                args.remove(i);
            }
            "--local-vi-ba-adaptive-velocity-min-references" => {
                local_vi_ba_adaptive_velocity_min_references = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--motion-vi-init-max-velocity" => {
                motion_vi_init_max_velocity_mps = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--motion-vi-init-max-gyro-bias" => {
                motion_vi_init_max_gyro_bias_rad_s = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--motion-vi-init-max-accel-bias" => {
                motion_vi_init_max_accel_bias_mps2 = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--motion-vi-init-max-imu-nis-per-dof" => {
                motion_vi_init_max_imu_nis_per_dof = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--motion-vi-init-max-rotation-residual-rms-rad" => {
                motion_vi_init_max_rotation_residual_rms_rad = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--motion-vi-init-max-velocity-residual-rms-mps" => {
                motion_vi_init_max_velocity_residual_rms_mps = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--motion-vi-init-max-position-residual-rms-m" => {
                motion_vi_init_max_position_residual_rms_meters = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--vi-bias-release-min-keyframes" => {
                vi_bias_release_min_keyframes = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--vi-bias-release-min-translation" => {
                vi_bias_release_min_translation_meters = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--motion-vi-init-estimate-gravity" => {
                motion_vi_init_estimate_gravity = true;
                args.remove(i);
            }
            "--no-motion-vi-init-estimate-gravity" => {
                motion_vi_init_no_estimate_gravity = true;
                args.remove(i);
            }
            "--motion-vi-init-max-gravity-norm-deviation" => {
                motion_vi_init_max_gravity_norm_deviation = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--motion-vi-init-estimate-gyro-bias" => {
                motion_vi_init_estimate_gyro_bias = true;
                args.remove(i);
            }
            "--no-motion-vi-init-estimate-gyro-bias" => {
                motion_vi_init_no_estimate_gyro_bias = true;
                args.remove(i);
            }
            "--covisibility-local-map-max-keyframes" => {
                covisibility_local_map_max_keyframes = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--covisibility-local-map-min-shared" => {
                covisibility_local_map_min_shared = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--covisibility-local-ba" => {
                covisibility_local_ba_enabled = true;
                args.remove(i);
            }
            "--sparse-factor-graph" => {
                sparse_factor_graph_enabled = true;
                args.remove(i);
            }
            "--covisibility-local-ba-min-keyframes" => {
                covisibility_local_ba_min_keyframes = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--covisibility-local-ba-max-keyframes" => {
                let raw = args.remove(i + 1);
                covisibility_local_ba_max_keyframes = if raw.eq_ignore_ascii_case("none") {
                    None
                } else {
                    Some(raw.parse()?)
                };
                args.remove(i);
            }
            "--covisibility-local-ba-motion-vi-raw-activation" => {
                let raw = args.remove(i + 1);
                let values = raw
                    .split(',')
                    .map(str::parse::<f64>)
                    .collect::<Result<Vec<_>, _>>()?;
                if values.len() != 3
                    || values
                        .iter()
                        .any(|value| !value.is_finite() || *value <= 0.0)
                {
                    return Err("--covisibility-local-ba-motion-vi-raw-activation expects three positive finite values: '<rotation_rad>,<velocity_mps>,<position_m>'".into());
                }
                covisibility_local_ba_motion_vi_raw_activation =
                    Some((values[0], values[1], values[2]));
                args.remove(i);
            }
            "--covisibility-local-ba-max-seed-landmarks-for-activation" => {
                covisibility_local_ba_max_seed_landmarks_for_activation =
                    Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--covisibility-local-ba-trigger-every" => {
                covisibility_local_ba_trigger_every = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--covisibility-local-ba-max-neighbor-keyframes" => {
                covisibility_local_ba_max_neighbor_keyframes = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--covisibility-local-ba-min-shared" => {
                covisibility_local_ba_min_shared = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--covisibility-local-ba-max-boundary-keyframes" => {
                covisibility_local_ba_max_boundary_keyframes = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--covisibility-local-ba-min-boundary-observations" => {
                covisibility_local_ba_min_boundary_observations = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--covisibility-local-ba-fallback-min-boundary-observations" => {
                let raw = args.remove(i + 1);
                covisibility_local_ba_fallback_min_boundary_observations =
                    if raw.eq_ignore_ascii_case("none") {
                        None
                    } else {
                        Some(raw.parse()?)
                    };
                args.remove(i);
            }
            "--covisibility-local-ba-max-landmarks" => {
                let raw = args.remove(i + 1);
                covisibility_local_ba_max_landmarks = if raw.eq_ignore_ascii_case("none") {
                    None
                } else {
                    Some(raw.parse()?)
                };
                args.remove(i);
            }
            "--covisibility-local-ba-min-active-observations" => {
                covisibility_local_ba_min_active_observations = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--covisibility-local-ba-outlier-threshold-px" => {
                let raw = args.remove(i + 1);
                covisibility_local_ba_outlier_threshold_px = if raw.eq_ignore_ascii_case("none") {
                    None
                } else {
                    Some(raw.parse()?)
                };
                args.remove(i);
            }
            "--covisibility-local-ba-remove-outliers" => {
                covisibility_local_ba_remove_outliers = true;
                args.remove(i);
            }
            "--covisibility-local-ba-general-stereo" => {
                covisibility_local_ba_general_stereo = true;
                args.remove(i);
            }
            "--covisibility-local-ba-general-stereo-max-right-reprojection-px" => {
                let raw = args.remove(i + 1);
                covisibility_local_ba_general_stereo_max_right_reprojection_px =
                    if raw.eq_ignore_ascii_case("none") {
                        None
                    } else {
                        Some(raw.parse()?)
                    };
                args.remove(i);
            }
            "--covisibility-local-ba-max-outlier-observation-ratio" => {
                let raw = args.remove(i + 1);
                covisibility_local_ba_max_outlier_observation_ratio =
                    if raw.eq_ignore_ascii_case("none") {
                        None
                    } else {
                        Some(raw.parse()?)
                    };
                args.remove(i);
            }
            "--covisibility-local-ba-boundary-support-min-optimized-keyframes" => {
                let raw = args.remove(i + 1);
                covisibility_local_ba_boundary_support_min_optimized_keyframes =
                    if raw.eq_ignore_ascii_case("none") {
                        None
                    } else {
                        Some(raw.parse()?)
                    };
                args.remove(i);
            }
            "--covisibility-local-ba-boundary-support-min-fixed-keyframes" => {
                covisibility_local_ba_boundary_support_min_fixed_keyframes =
                    args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--covisibility-local-ba-max-behind-camera-ratio" => {
                let raw = args.remove(i + 1);
                covisibility_local_ba_max_behind_camera_ratio = if raw.eq_ignore_ascii_case("none")
                {
                    None
                } else {
                    Some(raw.parse()?)
                };
                args.remove(i);
            }
            "--covisibility-local-ba-min-fixed-to-optimized-ratio" => {
                let raw = args.remove(i + 1);
                covisibility_local_ba_min_fixed_to_optimized_ratio =
                    if raw.eq_ignore_ascii_case("none") {
                        None
                    } else {
                        Some(raw.parse()?)
                    };
                args.remove(i);
            }
            "--covisibility-local-ba-anchor-weight" => {
                covisibility_local_ba_anchor_weight = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--covisibility-local-ba-max-pose-translation-correction-m" => {
                covisibility_local_ba_max_pose_translation_correction_m =
                    Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--covisibility-local-ba-max-pose-rotation-correction-deg" => {
                covisibility_local_ba_max_pose_rotation_correction_deg =
                    Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--stereo-landmark-replenish" => {
                stereo_landmark_replenish = true;
                args.remove(i);
            }
            "--no-stereo-landmark-replenish" => {
                stereo_landmark_replenish_no = true;
                args.remove(i);
            }
            "--stereo-landmark-replenish-max-per-frame" => {
                stereo_landmark_replenish_max_per_frame = args.remove(i + 1).parse()?;
                stereo_landmark_replenish_max_per_frame_overridden = true;
                args.remove(i);
            }
            "--stereo-landmark-replenish-anchor-match-radius-px" => {
                stereo_landmark_replenish_anchor_match_radius_px =
                    Some(args.remove(i + 1).parse()?);
                stereo_landmark_replenish_anchor_match_radius_px_overridden = true;
                args.remove(i);
            }
            "--stereo-landmark-replenish-anchor-max-descriptor-distance" => {
                stereo_landmark_replenish_anchor_max_descriptor_distance =
                    Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--stereo-landmark-replenish-duplicate-radius-px" => {
                stereo_landmark_replenish_duplicate_radius_px = Some(args.remove(i + 1).parse()?);
                stereo_landmark_replenish_duplicate_radius_px_overridden = true;
                args.remove(i);
            }
            "--stereo-landmark-replenish-min-parallax-deg" => {
                stereo_landmark_replenish_min_parallax_deg = Some(args.remove(i + 1).parse()?);
                stereo_landmark_replenish_min_parallax_deg_overridden = true;
                args.remove(i);
            }
            "--stereo-landmark-replenish-min-depth-meters" => {
                stereo_landmark_replenish_min_depth_meters = Some(args.remove(i + 1).parse()?);
                stereo_landmark_replenish_min_depth_meters_overridden = true;
                args.remove(i);
            }
            "--stereo-landmark-replenish-max-depth-meters" => {
                stereo_landmark_replenish_max_depth_meters = Some(args.remove(i + 1).parse()?);
                stereo_landmark_replenish_max_depth_meters_overridden = true;
                args.remove(i);
            }
            "--max-pose-jump-meters" => {
                max_pose_jump_meters = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--pose-prior-visual-override" => {
                pose_prior_visual_override = true;
                args.remove(i);
            }
            "--no-pose-prior-visual-override" => {
                pose_prior_visual_override = false;
                args.remove(i);
            }
            "--accept-motion-prior-on-failure" => {
                accept_motion_prior_on_failure = true;
                args.remove(i);
            }
            "--no-accept-motion-prior-on-failure" => {
                accept_motion_prior_on_failure = false;
                args.remove(i);
            }
            "--max-consecutive-motion-prior-coasts" => {
                let raw = args.remove(i + 1);
                max_consecutive_motion_prior_coasts = if raw.eq_ignore_ascii_case("none") {
                    None
                } else {
                    Some(raw.parse()?)
                };
                args.remove(i);
            }
            "--temporal-landmark-tracking" => {
                temporal_landmark_tracking = true;
                temporal_landmark_tracking_overridden = true;
                args.remove(i);
            }
            "--no-temporal-landmark-tracking" => {
                temporal_landmark_tracking = false;
                temporal_landmark_tracking_overridden = true;
                args.remove(i);
            }
            "--temporal-landmark-tracking-history-frames" => {
                temporal_landmark_tracking_history_frames = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--pose-jump-gap-scaling" => {
                pose_jump_gap_scaling = true;
                args.remove(i);
            }
            "--no-pose-jump-gap-scaling" => {
                pose_jump_gap_scaling = false;
                args.remove(i);
            }
            "--pose-jump-gap-scaling-max-multiplier" => {
                pose_jump_gap_scaling_max_multiplier = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--tracking-min-inliers" => {
                tracking_min_inliers = args.remove(i + 1).parse()?;
                tracking_min_inliers_overridden = true;
                args.remove(i);
            }
            "--tracking-min-inlier-ratio" => {
                tracking_min_inlier_ratio = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--tracking-max-reprojection-error" => {
                let raw = args.remove(i + 1);
                tracking_max_reprojection_error = if raw.eq_ignore_ascii_case("none") {
                    None
                } else {
                    Some(raw.parse()?)
                };
                args.remove(i);
            }
            "--pnp-pose-prior-warm-start" => {
                pnp_pose_prior_warm_start = true;
                pnp_pose_prior_warm_start_overridden = true;
                args.remove(i);
            }
            "--no-pnp-pose-prior-warm-start" => {
                pnp_pose_prior_warm_start = false;
                pnp_pose_prior_warm_start_overridden = true;
                args.remove(i);
            }
            "--projection-guided-tracking" => {
                projection_guided_tracking = true;
                args.remove(i);
            }
            "--projection-query-landmark-distance-ratio" => {
                let raw = args.remove(i + 1);
                projection_query_landmark_distance_ratio = if raw.eq_ignore_ascii_case("none") {
                    None
                } else {
                    Some(raw.parse()?)
                };
                args.remove(i);
            }
            "--projection-search-radius-px" => {
                projection_search_radius_px = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--projection-widen-factor" => {
                projection_widen_factor = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--projection-max-widen-retries" => {
                projection_max_widen_retries = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--projection-no-local-map-refinement" => {
                projection_no_local_map_refinement = true;
                args.remove(i);
            }
            "--projection-refinement-search-radius-px" => {
                projection_refinement_search_radius_px = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--projection-refinement-iterations" => {
                projection_refinement_iterations = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--projection-refinement-radius-shrink-factor" => {
                projection_refinement_radius_shrink_factor = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--projection-refinement-reassign-correspondences" => {
                projection_refinement_reassign_correspondences = true;
                args.remove(i);
            }
            "--projection-refinement-min-inlier-pair-retention-ratio" => {
                projection_refinement_min_inlier_pair_retention_ratio =
                    args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--projection-refinement-max-translation-correction-m" => {
                projection_refinement_max_translation_correction_m =
                    Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--projection-refinement-max-rotation-correction-deg" => {
                projection_refinement_max_rotation_correction_deg =
                    Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--pnp-reprojection-threshold-px" => {
                pnp_reprojection_threshold_px = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--motion-model" => {
                let kind = args.remove(i + 1);
                motion_model = match kind.as_str() {
                    "pose" => MotionModelKind::Pose,
                    "velocity" => MotionModelKind::Velocity,
                    "imu" => MotionModelKind::ImuPredictive,
                    "adaptive-imu-pose" => MotionModelKind::AdaptiveImuPose,
                    other => {
                        return Err(format!(
                            "--motion-model: expected 'pose', 'velocity', 'imu', or 'adaptive-imu-pose', got {other:?}"
                        )
                        .into());
                    }
                };
                motion_model_overridden = true;
                args.remove(i);
            }
            "--imu-extrinsic-from-cam0" => {
                imu_extrinsic_from_cam0 = true;
                args.remove(i);
            }
            "--imu-motion-model-carry-forward-velocity" => {
                imu_motion_model_carry_forward_velocity = true;
                args.remove(i);
            }
            "--cross-check-matcher" => {
                cross_check_matcher = true;
                args.remove(i);
            }
            "--gpu-matcher-dll" => {
                gpu_matcher_dll = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--mutual-softmax-matcher" => {
                mutual_softmax_matcher = true;
                mutual_softmax_matcher_overridden = true;
                args.remove(i);
            }
            "--no-mutual-softmax-matcher" => {
                mutual_softmax_matcher = false;
                mutual_softmax_matcher_overridden = true;
                args.remove(i);
            }
            "--mutual-softmax-temperature" => {
                mutual_softmax_temperature = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--mutual-softmax-min-confidence" => {
                mutual_softmax_min_confidence = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--feature-extractor" => {
                let kind = args.remove(i + 1);
                feature_extractor = match kind.as_str() {
                    "corner" => FeatureExtractorKind::Corner,
                    "hog" => FeatureExtractorKind::Hog,
                    "superpoint-offline" => FeatureExtractorKind::SuperPointOffline,
                    "superpoint-onnx" => FeatureExtractorKind::SuperPointOnnx,
                    "optical-flow" | "basalt-optical-flow" | "klt" => {
                        FeatureExtractorKind::OpticalFlow
                    }
                    other => {
                        return Err(format!(
                            "--feature-extractor: expected 'corner', 'hog', 'superpoint-offline', 'superpoint-onnx', or 'optical-flow', got {other:?}"
                        )
                        .into());
                    }
                };
                args.remove(i);
            }
            "--superpoint-features-dir" => {
                superpoint_features_dir = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--superpoint-cam1-features-dir" => {
                superpoint_cam1_features_dir = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--superpoint-onnx-model" => {
                superpoint_onnx_model = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--superpoint-onnx-backend" => {
                superpoint_onnx_backend = SuperPointOnnxBackendArg::parse(&args.remove(i + 1))?;
                args.remove(i);
            }
            "--export-frame-appearance-descriptors" => {
                export_frame_appearance_descriptors = true;
                args.remove(i);
            }
            "--run-local-vi-ba-at-vi-init-promotion" => {
                run_local_vi_ba_at_vi_init_promotion = true;
                args.remove(i);
            }
            "--hog-max-features" => {
                hog_max_features = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--hog-min-corner-score" => {
                hog_min_corner_score = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--hog-orient" => {
                hog_orient = true;
                args.remove(i);
            }
            "--keyframe-min-translation" => {
                keyframe_min_translation = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--keyframe-min-frame-gap" => {
                keyframe_min_frame_gap = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--keyframe-max-frame-gap" => {
                keyframe_max_frame_gap = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--keyframe-tracked-landmark-ratio" => {
                keyframe_tracked_landmark_ratio = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--keyframe-min-tracked-landmarks-for-ratio" => {
                keyframe_min_tracked_landmarks_for_ratio = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--keyframe-min-inliers" => {
                keyframe_min_inliers = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--keyframe-min-inlier-ratio" => {
                keyframe_min_inlier_ratio = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--vi-init-min-samples" => {
                vi_init_min_samples = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--vi-init-min-stationary-window-seconds" => {
                vi_init_min_stationary_window_seconds = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--keep-pre-promotion-imu-factors" => {
                keep_pre_promotion_imu_factors = true;
                args.remove(i);
            }
            "--relinearise-imu-factor-bias-thresholds" => {
                let raw = args.remove(i + 1);
                let mut parts = raw.split(',');
                let gyro_str = parts.next().ok_or_else(|| {
                    "--relinearise-imu-factor-bias-thresholds: expected '<gyro_rad_s>,<accel_m_s2>'".to_string()
                })?;
                let accel_str = parts.next().ok_or_else(|| {
                    "--relinearise-imu-factor-bias-thresholds: missing accel component".to_string()
                })?;
                let gyro_thresh: f64 = gyro_str.trim().parse()?;
                let accel_thresh: f64 = accel_str.trim().parse()?;
                if gyro_thresh < 0.0 || accel_thresh < 0.0 {
                    return Err(
                        "--relinearise-imu-factor-bias-thresholds: thresholds must be >= 0".into(),
                    );
                }
                relinearise_imu_factor_bias_thresholds = Some((gyro_thresh, accel_thresh));
                args.remove(i);
            }
            "--relocalization-enabled" => {
                relocalization_enabled = true;
                args.remove(i);
            }
            "--relocalization-min-inliers" => {
                relocalization_min_inliers = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--relocalization-min-inlier-ratio" => {
                relocalization_min_inlier_ratio = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--relocalization-max-reprojection-error" => {
                let raw = args.remove(i + 1);
                relocalization_max_reprojection_error = if raw.eq_ignore_ascii_case("none") {
                    None
                } else {
                    Some(raw.parse()?)
                };
                args.remove(i);
            }
            "--relocalization-pose-prior-radius" => {
                relocalization_pose_prior_radius_meters = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--relocalization-recent-keyframe-window" => {
                relocalization_recent_keyframe_window = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--relocalization-max-translation-from-imu-prediction-meters" => {
                relocalization_max_translation_from_imu_prediction_meters =
                    Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--relocalization-attempt-interval-frames" => {
                relocalization_attempt_interval_frames = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--relocalization-max-consecutive-failed-attempts" => {
                relocalization_max_consecutive_failed_attempts = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--relocalization-max-translation-per-frame-from-last-success-meters" => {
                relocalization_max_translation_per_frame_from_last_success_meters =
                    Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--relocalization-min-inlier-depth-median-ratio-to-last-success" => {
                relocalization_min_inlier_depth_median_ratio_to_last_success =
                    Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--relocalization-max-inlier-depth-median-ratio-to-last-success" => {
                relocalization_max_inlier_depth_median_ratio_to_last_success =
                    Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--relocalization-covisibility-max-keyframes" => {
                relocalization_covisibility_max_keyframes = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--relocalization-covisibility-min-shared" => {
                relocalization_covisibility_min_shared = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--relocalization-covisibility-min-landmarks" => {
                relocalization_covisibility_min_landmarks = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--relocalization-covisibility-no-broader-fallback" => {
                relocalization_covisibility_broader_fallback = false;
                args.remove(i);
            }
            "--relocalization-covisibility-broader-fallback-interval-frames" => {
                relocalization_covisibility_broader_fallback_interval_frames =
                    args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--relocalization-covisibility-compare-broader-store" => {
                relocalization_covisibility_compare_broader_store = true;
                args.remove(i);
            }
            "--relocalization-appearance-max-keyframes" => {
                relocalization_appearance_max_keyframes = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--relocalization-appearance-candidate-log-limit" => {
                relocalization_appearance_candidate_log_limit = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--relocalization-appearance-min-similarity" => {
                relocalization_appearance_min_similarity = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--relocalization-appearance-exclude-recent-frame-gap" => {
                relocalization_appearance_exclude_recent_frame_gap = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--relocalization-appearance-min-landmarks" => {
                relocalization_appearance_min_landmarks = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--relocalization-appearance-no-broader-fallback" => {
                relocalization_appearance_broader_fallback = false;
                args.remove(i);
            }
            "--relocalization-appearance-broader-fallback-interval-frames" => {
                relocalization_appearance_broader_fallback_interval_frames =
                    args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--relocalization-appearance-compare-broader-store" => {
                relocalization_appearance_compare_broader_store = true;
                args.remove(i);
            }
            "--relocalization-confirmation-required-recoveries" => {
                relocalization_confirmation_required_recoveries = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--relocalization-confirmation-max-translation-per-frame-meters" => {
                relocalization_confirmation_max_translation_per_frame_meters =
                    Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--rebootstrap-after-lost-frames" => {
                rebootstrap_after_lost_frames = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--rebootstrap-cooldown-frames" => {
                rebootstrap_cooldown_frames = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--rebootstrap-independent-submap" => {
                rebootstrap_independent_submap = true;
                args.remove(i);
            }
            "--atlas-broader-recovery-max-source-keyframes" => {
                atlas_broader_recovery_max_source_keyframes = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--atlas-broader-recovery-interval-attempts" => {
                atlas_broader_recovery_interval_attempts = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--pose-graph-refinement" => {
                pose_graph_refinement_enabled = true;
                args.remove(i);
            }
            "--pose-graph-refinement-trigger-every" => {
                pose_graph_refinement_trigger_every = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--pose-graph-refinement-fixed-loop-edge-weight" => {
                pose_graph_refinement_fixed_loop_edge_weight = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--pose-graph-refinement-loop-pose-information" => {
                pose_graph_refinement_loop_pose_information = true;
                args.remove(i);
            }
            "--pose-graph-refinement-loop-pose-information-max-eigenvalue" => {
                pose_graph_refinement_loop_pose_information_max_eigenvalue =
                    args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--pose-graph-refinement-loop-pose-information-loop-edge-scale" => {
                pose_graph_refinement_loop_pose_information_loop_edge_scale =
                    args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--pose-graph-refinement-gnc" => {
                pose_graph_refinement_gnc = true;
                args.remove(i);
            }
            "--pose-graph-refinement-pcm" => {
                pose_graph_refinement_pcm = true;
                args.remove(i);
            }
            "--pose-graph-refinement-pcm-pairwise-only" => {
                pose_graph_refinement_pcm_require_individual = false;
                args.remove(i);
            }
            "--pose-graph-refinement-covariance-gate" => {
                pose_graph_refinement_covariance_gate = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--pose-graph-refinement-verifier" => {
                let kind = args.remove(i + 1);
                pose_graph_refinement_verifier = match kind.as_str() {
                    "essential" => LoopRefinementVerifierKind::Essential,
                    "pnp" => LoopRefinementVerifierKind::Pnp,
                    other => {
                        return Err(format!(
                            "--pose-graph-refinement-verifier: expected 'essential' or 'pnp', got {other:?}"
                        )
                        .into());
                    }
                };
                args.remove(i);
            }
            "--pose-graph-refinement-solver" => {
                let kind = args.remove(i + 1);
                pose_graph_refinement_solver = match kind.as_str() {
                    "se3" => LoopRefinementSolverKind::Se3,
                    "sim3" => LoopRefinementSolverKind::Sim3,
                    other => {
                        return Err(format!(
                            "--pose-graph-refinement-solver: expected 'se3' or 'sim3', got {other:?}"
                        )
                        .into());
                    }
                };
                args.remove(i);
            }
            "--pose-graph-refinement-appearance-loops" => {
                pose_graph_refinement_appearance_loops = true;
                args.remove(i);
            }
            "--pose-graph-refinement-appearance-min-gap" => {
                pose_graph_refinement_appearance_min_gap = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--pose-graph-refinement-appearance-max-candidates" => {
                pose_graph_refinement_appearance_max_candidates = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--pose-graph-refinement-appearance-min-inliers" => {
                pose_graph_refinement_appearance_min_inliers = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--pose-graph-refinement-appearance-confirmation-keyframes" => {
                pose_graph_refinement_appearance_confirmation_keyframes =
                    args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--pose-graph-refinement-appearance-confirmation-max-misses" => {
                pose_graph_refinement_appearance_confirmation_max_misses =
                    args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--pose-graph-refinement-appearance-projection-radius" => {
                pose_graph_refinement_appearance_projection_radius_px =
                    Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--pose-graph-refinement-appearance-projection-min-matches" => {
                pose_graph_refinement_appearance_projection_min_matches =
                    args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--pose-graph-refinement-fuse-loop-observations" => {
                pose_graph_refinement_fuse_loop_observations = true;
                args.remove(i);
            }
            "--pose-graph-refinement-loop-welding-ba" => {
                pose_graph_refinement_loop_welding_ba = true;
                args.remove(i);
            }
            "--pose-graph-refinement-propagate" => {
                pose_graph_refinement_propagate = true;
                args.remove(i);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    let euroc_dir = euroc_dir.ok_or("--euroc-dir <path/to/MH_01_easy> is required")?;
    if bootstrap_depth_meters <= 0.0 {
        return Err("--bootstrap-depth must be positive".into());
    }
    if !stereo_bootstrap && stereo_bootstrap_strict {
        return Err(
            "--no-stereo-bootstrap requires --allow-fixed-depth-bootstrap; fixed-depth seeding is diagnostic-only"
                .into(),
        );
    }
    if let Some(ratio) = keyframe_tracked_landmark_ratio {
        if !ratio.is_finite() || !(0.0..=1.0).contains(&ratio) {
            return Err("--keyframe-tracked-landmark-ratio must be in [0, 1]".into());
        }
        if keyframe_min_tracked_landmarks_for_ratio == 0 {
            return Err("--keyframe-min-tracked-landmarks-for-ratio must be >= 1".into());
        }
    }
    if let Some(gap) = keyframe_min_frame_gap {
        if gap == 0 {
            return Err("--keyframe-min-frame-gap must be >= 1".into());
        }
    }
    if let Some(gap) = keyframe_max_frame_gap {
        if gap == 0 {
            return Err("--keyframe-max-frame-gap must be >= 1".into());
        }
        if keyframe_min_frame_gap.is_some_and(|minimum| gap < minimum) {
            return Err("--keyframe-max-frame-gap must be >= --keyframe-min-frame-gap".into());
        }
    }
    if !tracking_min_inlier_ratio.is_finite() || !(0.0..=1.0).contains(&tracking_min_inlier_ratio) {
        return Err("--tracking-min-inlier-ratio must be in [0, 1]".into());
    }
    if max_pose_jump_meters.is_some_and(|value| !value.is_finite() || value <= 0.0) {
        return Err("--max-pose-jump-meters must be finite and positive".into());
    }
    // `--pose-prior-visual-override requires --max-pose-jump-meters` is
    // checked after motion-VI defaults (which may fill the jump gate).
    if let Some(max_reprojection_error) = tracking_max_reprojection_error {
        if !max_reprojection_error.is_finite() || max_reprojection_error <= 0.0 {
            return Err("--tracking-max-reprojection-error must be positive or 'none'".into());
        }
    }
    if let Some(ratio) = projection_query_landmark_distance_ratio {
        if !ratio.is_finite() || ratio <= 0.0 || ratio >= 1.0 {
            return Err(
                "--projection-query-landmark-distance-ratio must be in (0, 1) or 'none'".into(),
            );
        }
    }
    if !projection_search_radius_px.is_finite() || projection_search_radius_px <= 0.0 {
        return Err("--projection-search-radius-px must be positive".into());
    }
    if !projection_widen_factor.is_finite() || projection_widen_factor <= 0.0 {
        return Err("--projection-widen-factor must be positive".into());
    }
    if !projection_refinement_search_radius_px.is_finite()
        || projection_refinement_search_radius_px <= 0.0
    {
        return Err("--projection-refinement-search-radius-px must be positive".into());
    }
    if projection_refinement_iterations == 0 {
        return Err("--projection-refinement-iterations must be >= 1".into());
    }
    if !projection_refinement_radius_shrink_factor.is_finite()
        || projection_refinement_radius_shrink_factor <= 0.0
        || projection_refinement_radius_shrink_factor > 1.0
    {
        return Err("--projection-refinement-radius-shrink-factor must be in (0, 1]".into());
    }
    if relocalization_attempt_interval_frames == 0 {
        return Err("--relocalization-attempt-interval-frames must be >= 1".into());
    }
    if let Some(max_attempts) = relocalization_max_consecutive_failed_attempts {
        if max_attempts == 0 {
            return Err("--relocalization-max-consecutive-failed-attempts must be >= 1".into());
        }
    }
    if let Some(max_per_frame) = relocalization_max_translation_per_frame_from_last_success_meters {
        if !max_per_frame.is_finite() || max_per_frame <= 0.0 {
            return Err("--relocalization-max-translation-per-frame-from-last-success-meters must be positive".into());
        }
    }
    if let Some(min_ratio) = relocalization_min_inlier_depth_median_ratio_to_last_success {
        if !min_ratio.is_finite() || min_ratio <= 0.0 {
            return Err(
                "--relocalization-min-inlier-depth-median-ratio-to-last-success must be positive"
                    .into(),
            );
        }
    }
    if let Some(max_ratio) = relocalization_max_inlier_depth_median_ratio_to_last_success {
        if !max_ratio.is_finite() || max_ratio <= 0.0 {
            return Err(
                "--relocalization-max-inlier-depth-median-ratio-to-last-success must be positive"
                    .into(),
            );
        }
    }
    if let (Some(min_ratio), Some(max_ratio)) = (
        relocalization_min_inlier_depth_median_ratio_to_last_success,
        relocalization_max_inlier_depth_median_ratio_to_last_success,
    ) {
        if min_ratio > max_ratio {
            return Err("--relocalization-min-inlier-depth-median-ratio-to-last-success must be <= --relocalization-max-inlier-depth-median-ratio-to-last-success".into());
        }
    }
    if let Some(max_keyframes) = relocalization_covisibility_max_keyframes {
        if max_keyframes == 0 {
            return Err("--relocalization-covisibility-max-keyframes must be >= 1".into());
        }
    }
    if relocalization_covisibility_min_shared == 0 {
        return Err("--relocalization-covisibility-min-shared must be >= 1".into());
    }
    if relocalization_covisibility_min_landmarks == 0 {
        return Err("--relocalization-covisibility-min-landmarks must be >= 1".into());
    }
    if relocalization_covisibility_broader_fallback_interval_frames == 0 {
        return Err(
            "--relocalization-covisibility-broader-fallback-interval-frames must be >= 1".into(),
        );
    }
    if !projection_refinement_min_inlier_pair_retention_ratio.is_finite()
        || !(0.0..=1.0).contains(&projection_refinement_min_inlier_pair_retention_ratio)
    {
        return Err(
            "--projection-refinement-min-inlier-pair-retention-ratio must be in [0, 1]".into(),
        );
    }
    if projection_refinement_max_translation_correction_m
        .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return Err("--projection-refinement-max-translation-correction-m must be positive".into());
    }
    if projection_refinement_max_rotation_correction_deg
        .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return Err("--projection-refinement-max-rotation-correction-deg must be positive".into());
    }
    if let Some(max_keyframes) = relocalization_appearance_max_keyframes {
        if max_keyframes == 0 {
            return Err("--relocalization-appearance-max-keyframes must be >= 1".into());
        }
    }
    if let Some(candidate_log_limit) = relocalization_appearance_candidate_log_limit {
        if candidate_log_limit == 0 {
            return Err("--relocalization-appearance-candidate-log-limit must be >= 1".into());
        }
    }
    if !relocalization_appearance_min_similarity.is_finite() {
        return Err("--relocalization-appearance-min-similarity must be finite".into());
    }
    if relocalization_appearance_min_landmarks == 0 {
        return Err("--relocalization-appearance-min-landmarks must be >= 1".into());
    }
    if relocalization_appearance_broader_fallback_interval_frames == 0 {
        return Err(
            "--relocalization-appearance-broader-fallback-interval-frames must be >= 1".into(),
        );
    }
    if relocalization_confirmation_required_recoveries == 0 {
        return Err("--relocalization-confirmation-required-recoveries must be >= 1".into());
    }
    if let Some(max_per_frame) = relocalization_confirmation_max_translation_per_frame_meters {
        if !max_per_frame.is_finite() || max_per_frame <= 0.0 {
            return Err(
                "--relocalization-confirmation-max-translation-per-frame-meters must be positive"
                    .into(),
            );
        }
    }
    if let Some(lost_frames) = rebootstrap_after_lost_frames {
        if lost_frames == 0 {
            return Err("--rebootstrap-after-lost-frames must be >= 1".into());
        }
        if !stereo_bootstrap && !stereo_landmark_replenish {
            return Err(
                "--rebootstrap-after-lost-frames requires --stereo-bootstrap and/or --stereo-landmark-replenish (cam1 stereo matching)".into(),
            );
        }
    }
    if rebootstrap_cooldown_frames == 0 {
        return Err("--rebootstrap-cooldown-frames must be >= 1".into());
    }
    if rebootstrap_independent_submap && rebootstrap_after_lost_frames.is_none() {
        return Err(
            "--rebootstrap-independent-submap requires --rebootstrap-after-lost-frames".into(),
        );
    }
    if atlas_broader_recovery_max_source_keyframes == 0 {
        return Err("--atlas-broader-recovery-max-source-keyframes must be >= 1".into());
    }
    if atlas_broader_recovery_interval_attempts == 0 {
        return Err("--atlas-broader-recovery-interval-attempts must be >= 1".into());
    }
    if motion_vi_init_after_static_give_up && !motion_vi_init_enabled {
        return Err("--motion-vi-init-after-static-give-up requires --motion-vi-init".into());
    }
    if motion_vi_init_from_configured_bias && !motion_vi_init_enabled {
        return Err("--motion-vi-init-from-configured-bias requires --motion-vi-init".into());
    }
    if !pose_graph_refinement_pcm_require_individual && !pose_graph_refinement_pcm {
        return Err(
            "--pose-graph-refinement-pcm-pairwise-only requires --pose-graph-refinement-pcm".into(),
        );
    }
    if motion_vi_init_enabled {
        if motion_vi_init_no_estimate_gravity {
            motion_vi_init_estimate_gravity = false;
        } else {
            motion_vi_init_estimate_gravity = true;
        }
        if motion_vi_init_no_estimate_gyro_bias {
            motion_vi_init_estimate_gyro_bias = false;
        } else {
            motion_vi_init_estimate_gyro_bias = true;
        }
    }
    if local_vi_ba_enabled && stereo_bootstrap && !stereo_landmark_replenish_no {
        stereo_landmark_replenish = true;
        if motion_vi_init_enabled && !stereo_landmark_replenish_max_per_frame_overridden {
            // Dense replenishment floods long EuRoC runs with weak points and
            // destabilizes late-window VI-BA. Keep a conservative default cap
            // when motion-VI-init + local VI-BA are active together.
            stereo_landmark_replenish_max_per_frame = 10;
        }
        if motion_vi_init_enabled {
            // Tighten replenishment geometry for motion-VI runs unless users
            // explicitly override each knob.
            if !stereo_landmark_replenish_anchor_match_radius_px_overridden {
                stereo_landmark_replenish_anchor_match_radius_px = Some(2.0);
            }
            if !stereo_landmark_replenish_duplicate_radius_px_overridden {
                stereo_landmark_replenish_duplicate_radius_px = Some(5.0);
            }
            if !stereo_landmark_replenish_min_parallax_deg_overridden {
                stereo_landmark_replenish_min_parallax_deg = Some(1.5);
            }
            if !stereo_landmark_replenish_min_depth_meters_overridden {
                stereo_landmark_replenish_min_depth_meters = Some(0.5);
            }
            if !stereo_landmark_replenish_max_depth_meters_overridden {
                stereo_landmark_replenish_max_depth_meters = Some(15.0);
            }
        }
    }
    if local_vi_ba_enabled && motion_vi_init_enabled {
        if local_vi_ba_reject_final_imu_nis_per_dof_above.is_none() {
            local_vi_ba_reject_final_imu_nis_per_dof_above = motion_vi_init_max_imu_nis_per_dof;
        }
        if local_vi_ba_reject_velocity_above_mps.is_none() {
            // Keep the motion-init velocity ceiling. Softening to 12 m/s
            // with looser pose gates (gate43) let late 0.5–0.9 m writebacks
            // through and regressed rigid ATE vs gate42.
            local_vi_ba_reject_velocity_above_mps = motion_vi_init_max_velocity_mps;
        }
        if local_vi_ba_reject_gyro_bias_above_rad_s.is_none() {
            local_vi_ba_reject_gyro_bias_above_rad_s = motion_vi_init_max_gyro_bias_rad_s;
        }
        if local_vi_ba_reject_accel_bias_above_mps2.is_none() {
            local_vi_ba_reject_accel_bias_above_mps2 = motion_vi_init_max_accel_bias_mps2;
        }
        if local_vi_ba_reject_pose_translation_above_meters.is_none() {
            // Gate42 best: block >0.5 m jumps. Softening to 1.0 m (gate43)
            // accepted frames 425–480 and blew max rigid ATE to ~61 m.
            local_vi_ba_reject_pose_translation_above_meters = Some(0.5);
        }
        if local_vi_ba_reject_pose_rotation_above_degrees.is_none() {
            // Gate42 best: block >5°. Softening to 8° admitted 6–7°
            // corrections that then path-dependently enlarged later steps.
            local_vi_ba_reject_pose_rotation_above_degrees = Some(5.0);
        }
        if local_vi_ba_freeze_biases_above.is_none() {
            local_vi_ba_freeze_biases_above = Some(0.9);
        }
    }
    if motion_vi_init_enabled {
        if !tracking_min_inliers_overridden {
            // Gate45: rejecting <30-inlier hover poses recovered Sim(3)
            // scale 0.013 → 0.97 and rigid ATE 1.33 → 0.11 m on MH_01 800f.
            // Softening to 15 (gate46) path-dependently worsened both.
            tracking_min_inliers = 30;
        }
        if max_pose_jump_meters.is_none() {
            // Constant-pose prior: this is a per-frame teleport cap.
            // EuRoC 20 Hz at 2 m/s is 0.1 m/frame; 0.2 m still allows
            // takeoff while blocking the 0.22–0.73 m hover teleports.
            max_pose_jump_meters = Some(0.2);
        }
        if keyframe_min_inliers.is_none() {
            // Tracking may succeed at 30 inliers; promoting those frames
            // as KFs poisons the map during hover. Keep the local map
            // seeded from well-supported poses only.
            keyframe_min_inliers = Some(80);
        }
        if !mutual_softmax_matcher_overridden && !cross_check_matcher {
            // Gate54: mutual-softmax lifted tracking 0.37 → 0.55 while
            // keeping Sim(3) scale ~0.79 (vs collapsed 0.01 without the
            // inlier/jump locks). Slower than brute-force, but usable.
            mutual_softmax_matcher = true;
        }
        if !motion_model_overridden {
            // Gate62: adaptive-imu-pose + MS → tracking 0.56 / scale 0.93
            // (vs constant-pose gate54 0.55 / 0.79). Pure `--motion-model
            // imu` (gate61) kept scale but dropped tracking to 0.22.
            motion_model = MotionModelKind::AdaptiveImuPose;
            if !pnp_pose_prior_warm_start_overridden {
                pnp_pose_prior_warm_start = true;
            }
        }
        // Do NOT default accept-motion-prior-on-failure: gate67 unlimited
        // coast hit tracking=1.0 but collapsed Sim(3) scale to 0.02
        // (626 coasts). Cap+flag remain opt-in for gated A/B.
        if !temporal_landmark_tracking_overridden {
            // Image-free temporal track of previous PnP inliers — Basalt
            // KLT analogue for the SuperPoint descriptor frontend.
            // Gate70 (history=1): tracking 0.68 / rigid ATE 0.068 m.
            // Gate71 (history=5): tracking collapsed to 0.24 — stale
            // landmarks poison the temporal PnP. Keep history=1 default.
            temporal_landmark_tracking = true;
        }
        // Do NOT default temporal history > 1 (gate71). Keep CLI opt-in.
        // Do NOT default pose-prior-visual-override: gate58/60 raised
        // tracking but collapsed Sim(3) scale. Keep opt-in.
        // Do NOT default pose-jump-gap-scaling (gate50: scale 0.55) or
        // covis local map (gate52). Keep those as opt-in.
    }
    if pose_prior_visual_override && max_pose_jump_meters.is_none() {
        return Err("--pose-prior-visual-override requires --max-pose-jump-meters".into());
    }
    if !mutual_softmax_temperature.is_finite() || mutual_softmax_temperature < 0.0 {
        return Err("--mutual-softmax-temperature must be finite and >= 0".into());
    }
    if !mutual_softmax_min_confidence.is_finite()
        || mutual_softmax_min_confidence <= 0.0
        || mutual_softmax_min_confidence > 1.0
    {
        return Err("--mutual-softmax-min-confidence must be in (0, 1]".into());
    }
    if !motion_vi_init_enabled
        && (motion_vi_init_max_velocity_mps.is_some()
            || motion_vi_init_max_gyro_bias_rad_s.is_some()
            || motion_vi_init_max_accel_bias_mps2.is_some()
            || motion_vi_init_max_imu_nis_per_dof.is_some()
            || motion_vi_init_max_rotation_residual_rms_rad.is_some()
            || motion_vi_init_max_velocity_residual_rms_mps.is_some()
            || motion_vi_init_max_position_residual_rms_meters.is_some())
    {
        return Err("motion-VI sanity limits require --motion-vi-init".into());
    }
    if !motion_vi_init_enabled
        && (vi_bias_release_min_keyframes.is_some()
            || vi_bias_release_min_translation_meters.is_some())
    {
        return Err(
            "--vi-bias-release-min-keyframes / --vi-bias-release-min-translation require --motion-vi-init"
                .into(),
        );
    }
    if !motion_vi_init_enabled
        && (motion_vi_init_estimate_gravity || motion_vi_init_max_gravity_norm_deviation.is_some())
    {
        return Err(
            "--motion-vi-init-estimate-gravity / --motion-vi-init-max-gravity-norm-deviation require --motion-vi-init"
                .into(),
        );
    }
    if !motion_vi_init_estimate_gravity && motion_vi_init_max_gravity_norm_deviation.is_some() {
        return Err(
            "--motion-vi-init-max-gravity-norm-deviation requires --motion-vi-init-estimate-gravity"
                .into(),
        );
    }
    if !motion_vi_init_enabled && motion_vi_init_estimate_gyro_bias {
        return Err("--motion-vi-init-estimate-gyro-bias requires --motion-vi-init".into());
    }
    for (value, flag) in [
        (
            motion_vi_init_max_velocity_mps,
            "--motion-vi-init-max-velocity",
        ),
        (
            motion_vi_init_max_gyro_bias_rad_s,
            "--motion-vi-init-max-gyro-bias",
        ),
        (
            motion_vi_init_max_accel_bias_mps2,
            "--motion-vi-init-max-accel-bias",
        ),
        (
            motion_vi_init_max_imu_nis_per_dof,
            "--motion-vi-init-max-imu-nis-per-dof",
        ),
        (
            motion_vi_init_max_rotation_residual_rms_rad,
            "--motion-vi-init-max-rotation-residual-rms-rad",
        ),
        (
            motion_vi_init_max_velocity_residual_rms_mps,
            "--motion-vi-init-max-velocity-residual-rms-mps",
        ),
        (
            motion_vi_init_max_position_residual_rms_meters,
            "--motion-vi-init-max-position-residual-rms-m",
        ),
    ] {
        if value.is_some_and(|limit| !limit.is_finite() || limit <= 0.0) {
            return Err(format!("{flag} must be finite and > 0").into());
        }
    }
    if local_vi_ba_reject_final_imu_nis_per_dof_above
        .is_some_and(|limit| !limit.is_finite() || limit <= 0.0)
    {
        return Err(
            "--local-vi-ba-reject-final-imu-nis-per-dof-above must be finite and > 0".into(),
        );
    }
    if covisibility_local_ba_motion_vi_raw_activation.is_some()
        && (!covisibility_local_ba_enabled || !motion_vi_init_enabled)
    {
        return Err("--covisibility-local-ba-motion-vi-raw-activation requires --covisibility-local-ba and --motion-vi-init".into());
    }
    if let Some(maximum) = covisibility_local_ba_max_seed_landmarks_for_activation {
        if !covisibility_local_ba_enabled || maximum == 0 {
            return Err("--covisibility-local-ba-max-seed-landmarks-for-activation requires --covisibility-local-ba and a value >= 1".into());
        }
    }
    if covisibility_local_ba_enabled {
        if covisibility_local_ba_min_keyframes == 0 {
            return Err("--covisibility-local-ba-min-keyframes must be >= 1".into());
        }
        if covisibility_local_ba_max_keyframes
            .is_some_and(|maximum| maximum < covisibility_local_ba_min_keyframes)
        {
            return Err(
                "--covisibility-local-ba-max-keyframes must be >= the minimum or 'none'".into(),
            );
        }
        if covisibility_local_ba_trigger_every == 0 {
            return Err("--covisibility-local-ba-trigger-every must be >= 1".into());
        }
        if covisibility_local_ba_min_shared == 0 {
            return Err("--covisibility-local-ba-min-shared must be >= 1".into());
        }
        if covisibility_local_ba_min_active_observations == 0 {
            return Err("--covisibility-local-ba-min-active-observations must be >= 1".into());
        }
        if let Some(fallback_min) = covisibility_local_ba_fallback_min_boundary_observations {
            if fallback_min == 0 {
                return Err(
                    "--covisibility-local-ba-fallback-min-boundary-observations must be >= 1 or 'none'"
                        .into(),
                );
            }
        }
        if let Some(px) = covisibility_local_ba_outlier_threshold_px {
            if !px.is_finite() || px <= 0.0 {
                return Err(
                    "--covisibility-local-ba-outlier-threshold-px must be positive or 'none'"
                        .into(),
                );
            }
        }
        if let Some(px) = covisibility_local_ba_general_stereo_max_right_reprojection_px {
            if !px.is_finite() || px <= 0.0 {
                return Err(
                    "--covisibility-local-ba-general-stereo-max-right-reprojection-px must be positive or 'none'"
                        .into(),
                );
            }
        }
        if covisibility_local_ba_remove_outliers
            && covisibility_local_ba_outlier_threshold_px.is_none()
        {
            return Err(
                "--covisibility-local-ba-remove-outliers requires an outlier threshold".into(),
            );
        }
        if let Some(ratio) = covisibility_local_ba_max_outlier_observation_ratio {
            if !(0.0..=1.0).contains(&ratio) {
                return Err(
                    "--covisibility-local-ba-max-outlier-observation-ratio must be in [0, 1]"
                        .into(),
                );
            }
        }
        if let Some(ratio) = covisibility_local_ba_max_behind_camera_ratio {
            if !(0.0..=1.0).contains(&ratio) {
                return Err(
                    "--covisibility-local-ba-max-behind-camera-ratio must be in [0, 1] or 'none'"
                        .into(),
                );
            }
        }
        if let Some(ratio) = covisibility_local_ba_min_fixed_to_optimized_ratio {
            if !(ratio.is_finite() && ratio > 0.0) {
                return Err(
                    "--covisibility-local-ba-min-fixed-to-optimized-ratio must be > 0 or 'none'"
                        .into(),
                );
            }
        }
        if let Some(limit) = covisibility_local_ba_max_pose_translation_correction_m {
            if !limit.is_finite() || limit < 0.0 {
                return Err("--covisibility-local-ba-max-pose-translation-correction-m must be finite and >= 0".into());
            }
        }
        if let Some(limit) = covisibility_local_ba_max_pose_rotation_correction_deg {
            if !limit.is_finite() || limit < 0.0 {
                return Err("--covisibility-local-ba-max-pose-rotation-correction-deg must be finite and >= 0".into());
            }
        }
        if let Some(min_optimized) = covisibility_local_ba_boundary_support_min_optimized_keyframes
        {
            if min_optimized < 1 {
                return Err(
                    "--covisibility-local-ba-boundary-support-min-optimized-keyframes must be >= 1 or 'none'"
                        .into(),
                );
            }
            if covisibility_local_ba_boundary_support_min_fixed_keyframes < 1 {
                return Err(
                    "--covisibility-local-ba-boundary-support-min-fixed-keyframes must be >= 1 when boundary support gate is enabled"
                        .into(),
                );
            }
        }
    }
    if pose_graph_refinement_enabled {
        if pose_graph_refinement_fuse_loop_observations && !pose_graph_refinement_appearance_loops {
            return Err(
                "--pose-graph-refinement-fuse-loop-observations requires --pose-graph-refinement-appearance-loops"
                    .into(),
            );
        }
        if pose_graph_refinement_loop_welding_ba && !pose_graph_refinement_fuse_loop_observations {
            return Err(
                "--pose-graph-refinement-loop-welding-ba requires --pose-graph-refinement-fuse-loop-observations"
                    .into(),
            );
        }
        if pose_graph_refinement_trigger_every == 0 {
            return Err("--pose-graph-refinement-trigger-every must be >= 1".into());
        }
        if pose_graph_refinement_fixed_loop_edge_weight
            .is_some_and(|weight| !weight.is_finite() || weight <= 0.0)
        {
            return Err(
                "--pose-graph-refinement-fixed-loop-edge-weight must be finite and > 0".into(),
            );
        }
        if pose_graph_refinement_fixed_loop_edge_weight.is_some()
            && pose_graph_refinement_loop_pose_information
        {
            return Err(
                "fixed loop weight and loop pose information are mutually exclusive".into(),
            );
        }
        if !pose_graph_refinement_loop_pose_information_max_eigenvalue.is_finite()
            || pose_graph_refinement_loop_pose_information_max_eigenvalue <= 0.0
        {
            return Err(
                "--pose-graph-refinement-loop-pose-information-max-eigenvalue must be finite and > 0"
                    .into(),
            );
        }
        if !pose_graph_refinement_loop_pose_information_loop_edge_scale.is_finite()
            || pose_graph_refinement_loop_pose_information_loop_edge_scale <= 0.0
        {
            return Err(
                "--pose-graph-refinement-loop-pose-information-loop-edge-scale must be finite and > 0"
                    .into(),
            );
        }
        if let Some(gate) = pose_graph_refinement_covariance_gate {
            if !gate.is_finite() || gate <= 0.0 {
                return Err("--pose-graph-refinement-covariance-gate must be positive".into());
            }
        }
        if pose_graph_refinement_appearance_loops {
            if pose_graph_refinement_appearance_min_gap == 0 {
                return Err("--pose-graph-refinement-appearance-min-gap must be >= 1".into());
            }
            if pose_graph_refinement_appearance_max_candidates == 0 {
                return Err(
                    "--pose-graph-refinement-appearance-max-candidates must be >= 1".into(),
                );
            }
            if pose_graph_refinement_appearance_min_inliers == 0 {
                return Err("--pose-graph-refinement-appearance-min-inliers must be >= 1".into());
            }
            if pose_graph_refinement_appearance_confirmation_keyframes == 0 {
                return Err(
                    "--pose-graph-refinement-appearance-confirmation-keyframes must be >= 1".into(),
                );
            }
            if pose_graph_refinement_appearance_confirmation_max_misses == 0 {
                return Err(
                    "--pose-graph-refinement-appearance-confirmation-max-misses must be >= 1"
                        .into(),
                );
            }
            if pose_graph_refinement_appearance_projection_radius_px
                .is_some_and(|radius| !radius.is_finite() || radius <= 0.0)
            {
                return Err(
                    "--pose-graph-refinement-appearance-projection-radius must be finite and > 0"
                        .into(),
                );
            }
            if pose_graph_refinement_appearance_projection_min_matches == 0 {
                return Err(
                    "--pose-graph-refinement-appearance-projection-min-matches must be >= 1".into(),
                );
            }
        }
        if pose_graph_refinement_solver == LoopRefinementSolverKind::Sim3
            && pose_graph_refinement_gnc
        {
            eprintln!(
                "warning: --pose-graph-refinement-gnc has no effect with \
                 --pose-graph-refinement-solver sim3 — Sim3PoseGraph has no GNC \
                 robust-outlier variant yet, so the flag is silently ignored on \
                 that path (see LoopRefinementSolver::Sim3's doc comment)."
            );
        }
        if pose_graph_refinement_solver == LoopRefinementSolverKind::Sim3
            && (stereo_bootstrap || stereo_landmark_replenish)
        {
            return Err(
                "--pose-graph-refinement-solver sim3 is not valid for metric stereo maps; \
                 ORB-SLAM2/3 fix loop scale to 1 for stereo/RGB-D, so use the default \
                 --pose-graph-refinement-solver se3"
                    .into(),
            );
        }
    }
    if let Some(path) = basalt_config_path.as_ref() {
        let loaded = visloc_rs::vision::optical_flow::BasaltVioConfigFile::from_path(path)
            .map_err(|e| format!("--basalt-config {}: {e}", path.display()))?;
        let cfg = loaded.value0;
        // Faithful Basalt EuRoC profile: enable the sliding-window VI-BA path
        // and map the published knobs 1:1 where we already have plumbing.
        local_vi_ba_enabled = true;
        local_vi_ba_marginalization = true;
        local_vi_ba_use_sqrt_window_marginalization = cfg.vio_sqrt_marg;
        local_vi_ba_window_size = cfg.vio_max_kfs.max(2) as usize;
        optical_flow_config = cfg.optical_flow.clone();
        // Stereo LK keeps Basalt FB²=0.04. Temporal gets a looser gate so
        // velocity-seeded tracks that survive LSSD are not culled by a
        // sub-pixel FB residual under EuRoC rotation.
        optical_flow_config.optical_flow_temporal_max_recovered_dist2 = Some(4.0);
        feature_extractor = FeatureExtractorKind::OpticalFlow;
        // Track-id descriptors are exact matches; mutual-softmax / SuperPoint
        // cliff defaults do not apply to the Basalt OF frontend.
        if !mutual_softmax_matcher_overridden {
            mutual_softmax_matcher = false;
        }
        if !tracking_min_inliers_overridden {
            // Stereo-seeded OF maps start with tens of landmarks; tracks thin
            // out under FB=0.04. 8 keeps early frames alive without the
            // SuperPoint cliff (80) defaults.
            tracking_min_inliers = 8;
        }
        // SuperPoint hover gate (80) is unreachable for a ~50-landmark stereo seed.
        keyframe_min_inliers = Some(8);
        // Track-id maps go dark when KLT drops every ID; stereo re-bootstrap
        // re-seeds metric landmarks under the *current* track ids.
        if rebootstrap_after_lost_frames.is_none() {
            rebootstrap_after_lost_frames = Some(5);
        }
        if rebootstrap_cooldown_frames == 60 {
            rebootstrap_cooldown_frames = 20;
        }
        // Motion-VI path caps replenish at 10/frame; OF track turnover needs
        // a higher mint rate so new stereo LMs replace dying IDs before the
        // map goes dark in the mid-sequence cliff.
        if !stereo_landmark_replenish_max_per_frame_overridden {
            stereo_landmark_replenish_max_per_frame = 40;
        }
        eprintln!(
            "basalt profile loaded from {} \
             (vio_max_kfs={}, vio_sqrt_marg={}, frontend=optical-flow, \
             of_levels={}, of_pattern={}, of_grid={})",
            path.display(),
            cfg.vio_max_kfs,
            cfg.vio_sqrt_marg,
            cfg.optical_flow.optical_flow_levels,
            cfg.optical_flow.optical_flow_pattern,
            cfg.optical_flow.optical_flow_detection_grid_size
        );
        if cfg.optical_flow.optical_flow_type != "frame_to_frame" {
            eprintln!(
                "warning: optical_flow_type={} (only frame_to_frame is implemented)",
                cfg.optical_flow.optical_flow_type
            );
        }
    }
    Ok(CliArgs {
        euroc_dir,
        out_dir,
        max_frames,
        adaptive_keypoints,
        adaptive_frame_budget_ms,
        gravity_world,
        vi_init_max_wait_seconds,
        vi_init_try_initialize_on_every_frame,
        vi_init_gyro_std_limit,
        vi_init_mean_gyro_magnitude_limit,
        vi_init_accel_std_limit,
        bootstrap_depth_meters,
        corner_max_features,
        corner_min_score,
        corner_descriptor_radius,
        undistort,
        stereo_bootstrap,
        stereo_bootstrap_strict,
        adaptive_motion_failures_to_switch_to_pose,
        adaptive_motion_successes_to_switch_to_imu,
        adaptive_motion_imu_velocity_refresh_policy,
        motion_vi_init_enabled,
        motion_vi_init_after_static_give_up,
        motion_vi_init_from_configured_bias,
        motion_vi_init_min_keyframes,
        motion_vi_init_min_translation_meters,
        motion_vi_init_recover_scale,
        local_vi_ba_enabled,
        observation_confidence_ba_enabled,
        local_vi_ba_marginalization,
        local_vi_ba_use_sqrt_window_marginalization,
        local_vi_ba_window_size,
        basalt_config_path,
        optical_flow_config,
        local_vi_ba_initial_prior_std_devs,
        local_vi_ba_freeze_biases_above,
        local_vi_ba_reject_writeback_above,
        local_vi_ba_reject_final_imu_nis_per_dof_above,
        local_vi_ba_reject_velocity_above_mps,
        local_vi_ba_reject_pose_translation_above_meters,
        local_vi_ba_reject_pose_rotation_above_degrees,
        local_vi_ba_reject_gyro_bias_above_rad_s,
        local_vi_ba_reject_accel_bias_above_mps2,
        local_vi_ba_adaptive_velocity_gate,
        local_vi_ba_adaptive_velocity_quantile,
        local_vi_ba_adaptive_velocity_multiplier,
        local_vi_ba_adaptive_velocity_margin_mps,
        local_vi_ba_adaptive_velocity_min_mps,
        local_vi_ba_adaptive_velocity_max_mps,
        local_vi_ba_adaptive_velocity_min_references,
        motion_vi_init_max_velocity_mps,
        motion_vi_init_max_gyro_bias_rad_s,
        motion_vi_init_max_accel_bias_mps2,
        motion_vi_init_max_imu_nis_per_dof,
        motion_vi_init_max_rotation_residual_rms_rad,
        motion_vi_init_max_velocity_residual_rms_mps,
        motion_vi_init_max_position_residual_rms_meters,
        vi_bias_release_min_keyframes,
        vi_bias_release_min_translation_meters,
        motion_vi_init_estimate_gravity,
        motion_vi_init_max_gravity_norm_deviation,
        motion_vi_init_estimate_gyro_bias,
        covisibility_local_map_max_keyframes,
        covisibility_local_map_min_shared,
        covisibility_local_ba_enabled,
        sparse_factor_graph_enabled,
        covisibility_local_ba_min_keyframes,
        covisibility_local_ba_max_keyframes,
        covisibility_local_ba_motion_vi_raw_activation,
        covisibility_local_ba_max_seed_landmarks_for_activation,
        covisibility_local_ba_trigger_every,
        covisibility_local_ba_max_neighbor_keyframes,
        covisibility_local_ba_min_shared,
        covisibility_local_ba_max_boundary_keyframes,
        covisibility_local_ba_min_boundary_observations,
        covisibility_local_ba_fallback_min_boundary_observations,
        covisibility_local_ba_max_landmarks,
        covisibility_local_ba_min_active_observations,
        covisibility_local_ba_outlier_threshold_px,
        covisibility_local_ba_remove_outliers,
        covisibility_local_ba_general_stereo,
        covisibility_local_ba_general_stereo_max_right_reprojection_px,
        covisibility_local_ba_max_outlier_observation_ratio,
        covisibility_local_ba_boundary_support_min_optimized_keyframes,
        covisibility_local_ba_boundary_support_min_fixed_keyframes,
        covisibility_local_ba_max_behind_camera_ratio,
        covisibility_local_ba_min_fixed_to_optimized_ratio,
        covisibility_local_ba_max_pose_translation_correction_m,
        covisibility_local_ba_max_pose_rotation_correction_deg,
        covisibility_local_ba_anchor_weight,
        stereo_landmark_replenish,
        stereo_landmark_replenish_max_per_frame,
        stereo_landmark_replenish_anchor_match_radius_px,
        stereo_landmark_replenish_anchor_max_descriptor_distance,
        stereo_landmark_replenish_duplicate_radius_px,
        stereo_landmark_replenish_min_parallax_deg,
        stereo_landmark_replenish_min_depth_meters,
        stereo_landmark_replenish_max_depth_meters,
        max_pose_jump_meters,
        pose_prior_visual_override,
        accept_motion_prior_on_failure,
        max_consecutive_motion_prior_coasts,
        temporal_landmark_tracking,
        temporal_landmark_tracking_history_frames,
        pose_jump_gap_scaling,
        pose_jump_gap_scaling_max_multiplier,
        tracking_min_inliers,
        tracking_min_inlier_ratio,
        tracking_max_reprojection_error,
        pnp_pose_prior_warm_start,
        projection_guided_tracking,
        projection_search_radius_px,
        projection_widen_factor,
        projection_max_widen_retries,
        projection_query_landmark_distance_ratio,
        projection_no_local_map_refinement,
        projection_refinement_search_radius_px,
        projection_refinement_iterations,
        projection_refinement_radius_shrink_factor,
        projection_refinement_reassign_correspondences,
        projection_refinement_min_inlier_pair_retention_ratio,
        projection_refinement_max_translation_correction_m,
        projection_refinement_max_rotation_correction_deg,
        pnp_reprojection_threshold_px,
        motion_model,
        imu_extrinsic_from_cam0,
        imu_motion_model_carry_forward_velocity,
        cross_check_matcher,
        gpu_matcher_dll,
        mutual_softmax_matcher,
        mutual_softmax_temperature,
        mutual_softmax_min_confidence,
        feature_extractor,
        hog_max_features,
        hog_min_corner_score,
        hog_orient,
        keyframe_min_translation,
        keyframe_min_frame_gap,
        keyframe_max_frame_gap,
        keyframe_tracked_landmark_ratio,
        keyframe_min_tracked_landmarks_for_ratio,
        keyframe_min_inliers,
        keyframe_min_inlier_ratio,
        vi_init_min_samples,
        vi_init_min_stationary_window_seconds,
        keep_pre_promotion_imu_factors,
        relinearise_imu_factor_bias_thresholds,
        superpoint_features_dir,
        superpoint_cam1_features_dir,
        superpoint_onnx_model,
        superpoint_onnx_backend,
        export_frame_appearance_descriptors,
        run_local_vi_ba_at_vi_init_promotion,
        relocalization_enabled,
        relocalization_min_inliers,
        relocalization_min_inlier_ratio,
        relocalization_max_reprojection_error,
        relocalization_pose_prior_radius_meters,
        relocalization_recent_keyframe_window,
        relocalization_max_translation_from_imu_prediction_meters,
        relocalization_attempt_interval_frames,
        relocalization_max_consecutive_failed_attempts,
        relocalization_max_translation_per_frame_from_last_success_meters,
        relocalization_min_inlier_depth_median_ratio_to_last_success,
        relocalization_max_inlier_depth_median_ratio_to_last_success,
        relocalization_covisibility_max_keyframes,
        relocalization_covisibility_min_shared,
        relocalization_covisibility_min_landmarks,
        relocalization_covisibility_broader_fallback,
        relocalization_covisibility_broader_fallback_interval_frames,
        relocalization_covisibility_compare_broader_store,
        relocalization_appearance_max_keyframes,
        relocalization_appearance_candidate_log_limit,
        relocalization_appearance_min_similarity,
        relocalization_appearance_exclude_recent_frame_gap,
        relocalization_appearance_min_landmarks,
        relocalization_appearance_broader_fallback,
        relocalization_appearance_broader_fallback_interval_frames,
        relocalization_appearance_compare_broader_store,
        relocalization_confirmation_required_recoveries,
        relocalization_confirmation_max_translation_per_frame_meters,
        rebootstrap_after_lost_frames,
        rebootstrap_cooldown_frames,
        rebootstrap_independent_submap,
        atlas_broader_recovery_max_source_keyframes,
        atlas_broader_recovery_interval_attempts,
        pose_graph_refinement_enabled,
        pose_graph_refinement_trigger_every,
        pose_graph_refinement_fixed_loop_edge_weight,
        pose_graph_refinement_loop_pose_information,
        pose_graph_refinement_loop_pose_information_max_eigenvalue,
        pose_graph_refinement_loop_pose_information_loop_edge_scale,
        pose_graph_refinement_gnc,
        pose_graph_refinement_pcm,
        pose_graph_refinement_pcm_require_individual,
        pose_graph_refinement_covariance_gate,
        pose_graph_refinement_verifier,
        pose_graph_refinement_appearance_loops,
        pose_graph_refinement_appearance_min_gap,
        pose_graph_refinement_appearance_max_candidates,
        pose_graph_refinement_appearance_min_inliers,
        pose_graph_refinement_appearance_confirmation_keyframes,
        pose_graph_refinement_appearance_confirmation_max_misses,
        pose_graph_refinement_appearance_projection_radius_px,
        pose_graph_refinement_appearance_projection_min_matches,
        pose_graph_refinement_fuse_loop_observations,
        pose_graph_refinement_loop_welding_ba,
        pose_graph_refinement_propagate,
        pose_graph_refinement_solver,
    })
}
