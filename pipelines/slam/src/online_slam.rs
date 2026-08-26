//! Online SLAM orchestration: configuration, relocalization / loop-closure
//! refinement state, IMU coupling, and the [`OnlineSlamPipeline`] itself.

use super::loop_pose_information::estimate_loop_pose_information;
pub use super::loop_pose_information::{
    LoopPoseInformationConfig, LoopPoseInformationDiagnostic, LoopPoseInformationFailure,
    LoopPoseInformationFailureCounts,
};
use super::*;
use crate::imu_preintegration::ImuNoiseModel;
use std::time::Instant;

#[derive(Debug, Clone, PartialEq)]
pub struct OnlineSlamConfig {
    pub apply_map_updates: bool,
    pub loop_closure: LoopClosureConfig,
    /// Optional IMU pre-integration hookup. When `Some`, the pipeline accepts
    /// inter-frame IMU samples via [`OnlineSlamPipeline::push_imu_measurement`]
    /// and emits an [`ImuPreintegrationFactor`] connecting each adjacent
    /// keyframe pair on [`OnlineSlamResult::imu_factor`]. The factor is a
    /// hint for downstream pose-graph / BA consumers; the tracker and local
    /// mapper themselves remain appearance-driven. `None` (the default)
    /// keeps the pipeline IMU-free and side-effect-free for existing callers.
    pub imu: Option<OnlineSlamImuConfig>,
    /// Optional sliding-window local VI-BA refinement triggered every
    /// `local_vi_ba.trigger_every` IMU factors. When `Some`, the
    /// [`crate::OnlineSlamLocalBaState`] table tracks per-keyframe
    /// `(velocity, bias)` slots and refines the trailing
    /// [`crate::OnlineSlamLocalBaConfig::window_size`] keyframes' poses +
    /// landmarks + velocities + biases against the staged
    /// [`ImuPreintegrationFactor`] history. Requires `imu = Some(_)`;
    /// when `imu` is `None` (no IMU factors are ever emitted), the local
    /// VI-BA stage simply never fires. `None` (default) keeps the
    /// pipeline appearance-only on the critical path.
    pub local_vi_ba: Option<OnlineSlamLocalBaConfig>,
    /// Optional visual-only covisibility local BA stage triggered after
    /// a new keyframe has been applied to the map. Unlike
    /// [`OnlineSlamLocalBaConfig`], this stage does not require IMU
    /// factors; it selects high-covisibility neighbor keyframes and
    /// fixed boundary keyframes around the active keyframe, then runs
    /// [`crate::refine_visual_map_with_covisibility_ba`]. `None`
    /// (default) preserves the existing online pipeline behavior.
    pub covisibility_local_ba: Option<OnlineSlamCovisibilityLocalBaConfig>,
    /// Optional DROID-style sparse keyframe factor lifecycle. The stage
    /// proposes temporal, proximity, and stereo edges on each committed
    /// keyframe, retains inactive edges for broader recovery, and exposes
    /// optimizer-facing correction/information/damping records. `None`
    /// preserves the existing pipeline path without graph bookkeeping.
    pub sparse_factor_graph: Option<SparseFactorGraphConfig>,
    /// Optional auto-bootstrap stage that runs a
    /// [`crate::VisualInertialInitializer`] over the pipeline's incoming
    /// IMU stream and atomically promotes the recovered
    /// `(R_w←b, b_g, b_a)` into the running pre-integrator + first
    /// keyframe on the first frame where both (a) `try_initialize` has
    /// succeeded and (b) a new keyframe was just registered. Requires
    /// `imu = Some(_)`; the configuration is rejected on
    /// [`OnlineSlamPipeline::new`] otherwise. `None` (default) keeps the
    /// pipeline's bias / rotation seeds at whatever the caller passed on
    /// [`OnlineSlamImuConfig`].
    pub vi_init: Option<OnlineSlamViInitConfig>,
    /// Optional motion-based VI init stage that normally fires after `vi_init`
    /// succeeds and the body has moved enough to give the IMU translational
    /// excitation. An explicit fallback may instead start from configured IMU
    /// biases after static initialization gives up. Holds the known-scale
    /// visual body-pose trajectory fixed while refining per-keyframe velocity
    /// and a shared `(b_g, b_a)` against IMU pre-integration factors (the
    /// inertial-only initialization stage; see
    /// [`crate::vi_motion_initializer`] for the contract). Requires both
    /// `imu = Some(_)` and `vi_init = Some(_)`; otherwise the config is
    /// rejected on [`OnlineSlamPipeline::new`]. `None` (default) keeps
    /// the pipeline at the static-only flavour.
    pub vi_motion_init: Option<OnlineSlamMotionViInitConfig>,
    /// When `true`, IMU factors staged on newly-registered keyframes
    /// flow downstream (into the local-VI-BA factor history and onto
    /// `OnlineSlamResult.imu_factor`) even while the auto-bootstrap
    /// stage is still active. The factors carry the caller's
    /// placeholder bias linearisation; the BA's own Gauss-Newton
    /// iterations are expected to absorb the resulting initial-cost
    /// bump. The legacy contract (this flag `false`, the default)
    /// silently discards every pre-promotion factor and reports the
    /// count on `discarded_stale_factor_count` for audit, matching
    /// the conservative "never feed BA a known-bad linearisation"
    /// posture. Empirically on real EuRoC the strict path leaves
    /// local-VI-BA with at most one trigger before the visual tracker
    /// dies on the takeoff transient (see
    /// `docs/motion_based_vi_alignment.md` §Phase-12); flipping the
    /// flag recovers the 6–7 pre-promotion factors and lets the
    /// Phase-9 mirror update the IMU motion model more frequently.
    pub keep_pre_promotion_imu_factors: bool,
    /// Optional online loop-closure refinement stage. When `Some`, the
    /// pipeline mirrors registered keyframe poses into a running
    /// [`PoseGraph`], runs an [`EssentialMatrixLoopClosureVerifier`] over
    /// every candidate `detect_loop_closure_candidates` produces, and
    /// fires [`PoseGraph::optimize_se3_iterative`] whenever
    /// `trigger_every_new_constraints` fresh verified loop edges have
    /// accumulated since the last solve. On a successful solve the
    /// optimised keyframe poses are written back into
    /// `self.map.keyframes`. `None` (the default) keeps loop-closure
    /// candidates as the existing diagnostic-only output on
    /// [`OnlineSlamResult::loop_closure_candidates`] and never modifies
    /// the map.
    pub pose_graph_refinement: Option<OnlineSlamLoopClosureRefinementConfig>,
    /// Optional relocalization-on-tracker-death stage. When `Some`,
    /// every `process_frame` call whose primary tracker attempt failed
    /// (`tracking.localization.success == false`) re-runs PnP against
    /// the full visual map via the stage's owned
    /// [`LocalizationPipeline`]; if the recovered solution clears the
    /// stage's acceptance gate, the pipeline overrides the tracker's
    /// state via [`Tracker::accept_relocalization_result`] and treats
    /// the frame as a successful `TrackingEvent::Relocalized` event.
    /// `None` (default) keeps the legacy behaviour where a failed
    /// `track_frame` call leaves the tracker dead until manual reset.
    pub relocalization: Option<OnlineSlamRelocalizationConfig>,
}

impl OnlineSlamConfig {
    /// Validate cross-field invariants the type system cannot express.
    /// Called from [`OnlineSlamPipeline::new`]; surfaces the same error
    /// type to callers that want to validate before constructing a
    /// pipeline (e.g. CLI front-ends).
    pub fn validate(&self) -> Result<(), OnlineSlamConfigError> {
        if let Some(vi_init) = &self.vi_init {
            let Some(imu) = &self.imu else {
                return Err(OnlineSlamConfigError::ViInitRequiresImu);
            };
            if (vi_init.initializer.gravity_world - imu.gravity_world).norm() > 1.0e-12 {
                return Err(OnlineSlamConfigError::GravityMismatch {
                    imu_gravity_world: imu.gravity_world,
                    vi_init_gravity_world: vi_init.initializer.gravity_world,
                });
            }
        }
        if let Some(motion) = &self.vi_motion_init {
            if self.imu.is_none() {
                return Err(OnlineSlamConfigError::MotionViInitRequiresImu);
            }
            if self.vi_init.is_none() {
                return Err(OnlineSlamConfigError::MotionViInitRequiresStaticViInit);
            }
            if (motion.allow_after_static_give_up
                || motion.allow_from_configured_bias_before_static)
                && self.vi_init.as_ref().is_some_and(|static_config| {
                    static_config.on_persistent_rejection != ViInitFallback::KeepExistingSeed
                })
            {
                return Err(OnlineSlamConfigError::MotionViInitAfterGiveUpRequiresKeepExistingSeed);
            }
            if let Some(imu) = &self.imu {
                if (motion.initializer.gravity_world - imu.gravity_world).norm() > 1.0e-12 {
                    return Err(OnlineSlamConfigError::MotionGravityMismatch {
                        imu_gravity_world: imu.gravity_world,
                        motion_gravity_world: motion.initializer.gravity_world,
                    });
                }
            }
            if self.vi_init.as_ref().is_some_and(|static_config| {
                motion.initializer.body_to_camera != static_config.body_to_camera
            }) {
                return Err(OnlineSlamConfigError::MotionExtrinsicMismatch);
            }
        }
        if self.local_vi_ba.as_ref().is_some_and(|local| {
            self.vi_init
                .as_ref()
                .is_some_and(|static_config| local.body_to_camera != static_config.body_to_camera)
        }) {
            return Err(OnlineSlamConfigError::LocalViBaExtrinsicMismatch);
        }
        if self.local_vi_ba.as_ref().is_some_and(|local| {
            let invalid = |(gyro, accel): (f64, f64)| {
                !gyro.is_finite() || gyro <= 0.0 || !accel.is_finite() || accel <= 0.0
            };
            local.bias_random_walk_weights.is_some_and(invalid)
                || local.bias_random_walk_noise_densities.is_some_and(invalid)
        }) {
            return Err(OnlineSlamConfigError::InvalidLocalViBaBiasRandomWalkWeights);
        }
        if self
            .sparse_factor_graph
            .as_ref()
            .is_some_and(|config| !config.is_valid())
        {
            return Err(OnlineSlamConfigError::InvalidSparseFactorGraphConfig);
        }
        Ok(())
    }
}

impl Default for OnlineSlamConfig {
    fn default() -> Self {
        Self {
            apply_map_updates: true,
            loop_closure: LoopClosureConfig::default(),
            imu: None,
            local_vi_ba: None,
            covisibility_local_ba: None,
            sparse_factor_graph: None,
            vi_init: None,
            vi_motion_init: None,
            keep_pre_promotion_imu_factors: false,
            pose_graph_refinement: None,
            relocalization: None,
        }
    }
}

/// Per-session configuration for visual-only covisibility local BA inside
/// [`OnlineSlamPipeline`].
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineSlamCovisibilityLocalBaConfig {
    /// Minimum number of keyframes in the map before the stage may run.
    /// Clamped to at least `1`. The default skips the first keyframe so
    /// startup does not report a predictable "no local landmarks" solve.
    pub min_keyframes: usize,
    /// Optional upper map-size bound for the local BA stage. Useful for
    /// limiting it to early map maturation before global loop refinement
    /// becomes the primary correction mechanism.
    pub max_keyframes: Option<usize>,
    /// Optional sequence-independent activation gate driven by the latest
    /// motion-VI initializer raw residual rejection. When configured, visual
    /// BA runs only while motion initialization is pending and all three raw
    /// residuals are within these coarse conditioning bounds. It stops as
    /// soon as motion-VI promotion succeeds.
    pub motion_vi_raw_residual_activation: Option<MotionViRawResidualActivationConfig>,
    /// Optional bootstrap-support activation gate. The stage runs only when
    /// the earliest keyframe observes no more than this many landmarks.
    /// This is evaluated from map evidence, never a sequence identifier.
    pub max_seed_landmarks_for_activation: Option<usize>,
    /// Run after every N newly-applied keyframes. Clamped to at least
    /// `1`; `1` runs on every eligible new keyframe.
    pub trigger_every_new_keyframes: usize,
    /// Optional post-solve quality gate. When `Some(r)`, BA runs on a
    /// cloned map first; if post-BA outlier observations exceed `r` of the
    /// selected observation count, the clone is discarded and the live map is
    /// left unchanged. `None` preserves the legacy direct write-back path.
    pub max_outlier_observation_ratio: Option<f64>,
    /// Optional post-solve behind-camera degeneracy gate. When `Some(r)`, BA
    /// runs on a cloned map first; if the fraction of solved optimized
    /// landmarks that project behind (or onto) an observing optimized camera
    /// exceeds `r`, the clone is discarded and the live map is left unchanged.
    /// Targets degenerate/under-constrained solves that would otherwise
    /// corrupt the map on write-back. `None` (default) leaves this gate off.
    /// See [`crate::behind_camera_optimized_landmark_ratio`].
    pub max_behind_camera_landmark_ratio: Option<f64>,
    /// Optional post-solve fixed-anchor adequacy gate expressed as a ratio.
    /// When `Some(r)`, the write-back is rejected unless
    /// `fixed_keyframe_count >= ceil(optimized_keyframe_count * r)`. This is
    /// the ratio form of the fixed-boundary requirement (the absolute-floor
    /// form lives on
    /// [`CovisibilityLocalBaConfig::boundary_support_min_fixed_keyframes`]).
    /// Evaluated on the selected window after the solve, so diagnostics still
    /// record what would have been optimized. `None` (default) leaves it off.
    /// See [`crate::fixed_to_optimized_ratio_satisfied`].
    pub min_fixed_to_optimized_ratio: Option<f64>,
    /// Optional maximum camera-centre displacement accepted from one solve.
    /// Activates clone-and-check transactional write-back when set.
    pub max_pose_translation_correction_m: Option<f64>,
    /// Optional maximum rotation change accepted from one solve, in radians.
    /// Activates clone-and-check transactional write-back when set.
    pub max_pose_rotation_correction_rad: Option<f64>,
    /// Window selection, optimizer, robust loss, and outlier handling
    /// settings for the actual covisibility BA solve.
    pub ba: CovisibilityLocalBaConfig,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionViRawResidualActivationConfig {
    pub max_rotation_residual_rms_rad: f64,
    pub max_velocity_residual_rms_mps: f64,
    pub max_position_residual_rms_meters: f64,
}

impl Default for OnlineSlamCovisibilityLocalBaConfig {
    fn default() -> Self {
        Self {
            min_keyframes: 2,
            max_keyframes: None,
            motion_vi_raw_residual_activation: None,
            max_seed_landmarks_for_activation: None,
            trigger_every_new_keyframes: 1,
            max_outlier_observation_ratio: None,
            max_behind_camera_landmark_ratio: None,
            min_fixed_to_optimized_ratio: None,
            max_pose_translation_correction_m: None,
            max_pose_rotation_correction_rad: None,
            ba: CovisibilityLocalBaConfig::default(),
        }
    }
}

/// Per-frame outcome of the visual-only covisibility local BA stage.
/// Exposed on [`OnlineSlamResult::covisibility_local_ba`]; `Some` only
/// when the stage was configured, a new keyframe was applied, and the
/// trigger interval fired.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineSlamCovisibilityLocalBaStats {
    pub active_keyframe_id: u64,
    pub map_keyframe_count: usize,
    /// Active sparse-factor neighbors admitted as variable BA candidates.
    /// `None` means the sparse graph stage was disabled.
    pub factor_graph_neighbor_count: Option<usize>,
    pub elapsed_ms: f64,
    pub success: bool,
    pub error: Option<CovisibilityLocalBaError>,
    pub selection: Option<CovisibilityLocalBaSelection>,
    pub ba_result: Option<BaResult>,
    pub mean_reprojection_before_px: Option<f64>,
    pub mean_reprojection_after_px: Option<f64>,
    pub max_pose_translation_correction_m: Option<f64>,
    pub max_pose_rotation_correction_rad: Option<f64>,
    pub updated_keyframe_count: usize,
    pub updated_landmark_count: usize,
    pub outlier_observation_count: usize,
    pub observation_count: usize,
    pub outlier_observation_ratio: Option<f64>,
    pub quality_gate_rejected: bool,
    pub pose_correction_gate_rejected: bool,
    pub removed_observation_count: usize,
}

/// Per-session configuration for the relocalization-on-tracker-death
/// stage owned by [`OnlineSlamPipeline`]. When attached via
/// [`OnlineSlamConfig::relocalization`], the pipeline owns a
/// [`LocalizationPipeline`] and runs it against the full visual map on
/// every frame whose primary `track_frame` call returned
/// `localization.success == false`. The recovered solution is accepted
/// when it clears every gate below; on acceptance, the tracker's history
/// is overwritten via [`Tracker::accept_relocalization_result`] and the
/// frame proceeds through the rest of `process_frame` as if primary
/// tracking had succeeded.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineSlamRelocalizationConfig {
    /// Minimum inlier count the recovered PnP solution must report. A
    /// higher bar than the regular tracker's `min_inliers` makes sense
    /// here because relocalization runs against the full map with no
    /// motion prior, so false positives are costlier.
    pub min_inliers: usize,
    /// Minimum inlier ratio (inliers / correspondences) for acceptance.
    pub min_inlier_ratio: f64,
    /// Optional maximum mean reprojection error. `None` disables the
    /// gate; `Some(px)` rejects solutions with mean reprojection error
    /// strictly greater than `px`.
    pub max_mean_reprojection_error: Option<f64>,
    /// When `Some(radius_m)`, the recovery PnP is run with the
    /// tracker's per-frame motion prior fed into the localizer's
    /// pose-prior warm-start path
    /// ([`FrameLocalizer::localize_frame_with_pose_prior_warm_start_and_descriptor_store`])
    /// instead of the no-prior global PnP. The radius restricts the
    /// candidate landmark set to those within `radius_m` of the
    /// prior's camera centre. Empirically on EuRoC, the no-prior path
    /// accepts < 0.3 % of recovery attempts because cross-attitude
    /// HOG descriptor mismatch dominates; threading the prior in
    /// short-circuits the matcher to the local landmark set + seeds
    /// RANSAC with the predicted pose, both of which lift recovery
    /// quality. `None` (default) preserves the no-prior global path
    /// for callers who explicitly want full-map relocalization.
    pub pose_prior_candidate_radius_meters: Option<f64>,
    /// Phase-26 #4a active-frontier submap selection. When
    /// `Some(window_keyframes)`, the recovery PnP's descriptor store
    /// is restricted to landmarks observed by any of the most recent
    /// `window_keyframes` keyframes in the map (i.e., the "active
    /// frontier" of the current map). `None` (default) preserves the
    /// Phase-23 #1 full-map behaviour. Empirically on EuRoC (Phase-26
    /// #2), the full-map recovery PnP accepts wrong-scale solutions
    /// because the candidate landmark set spans the whole map and
    /// admits geometrically self-consistent recoveries far from the
    /// true pose; the active-frontier window targets that failure
    /// mode by excluding stale landmarks from consideration.
    pub recent_keyframe_window: Option<usize>,
    /// Optional covisibility-local recovery descriptor store. When
    /// `Some(config)`, recovery PnP restricts descriptor matching to
    /// landmarks observed by the last successful keyframe (or the most
    /// recent keyframe before it) plus its high-covisibility neighbors.
    /// This targets full-map ambiguity without requiring a place-
    /// recognition backend. If the local set cannot be built or is too
    /// small, recovery falls back to the existing full-map / recent-
    /// window policy.
    pub covisibility_local_map: Option<OnlineSlamRelocalizationCovisibilityConfig>,
    /// Optional appearance-retrieval recovery descriptor store. When
    /// `Some(config)`, recovery PnP first ranks existing keyframes by
    /// cosine similarity between the failed frame's mean local descriptor
    /// and each keyframe's mean local descriptor, then restricts matching
    /// to landmarks observed by the top retrieved keyframes. This is a
    /// lightweight place-recognition seed for relocalization; stronger
    /// learned global descriptors can replace the mean-descriptor signal
    /// behind the same policy later. `None` (default) keeps the existing
    /// full-map / recent-window / covisibility policy.
    pub appearance_retrieval_map: Option<OnlineSlamRelocalizationAppearanceConfig>,
    /// Phase-26 #4b post-acceptance IMU sanity check. When
    /// `Some(max_translation_m)`, recoveries that otherwise pass the
    /// inlier-count / inlier-ratio / reprojection-error gates are
    /// further rejected if the recovered camera centre is more than
    /// `max_translation_m` away from the tracker's per-frame motion
    /// prior's camera centre. `None` (default) preserves the
    /// Phase-23 #1 no-IMU-sanity-check behaviour. The motivation
    /// (Phase-26 #2 V1_01 false-positive diagnosis): the recovery
    /// PnP can land at a wrong-scale solution that is geometrically
    /// self-consistent but inconsistent with the IMU's belief about
    /// where the camera is; a coarse translation gate filters out
    /// those drift-incompatible recoveries cheaply.
    pub max_translation_from_imu_prediction_meters: Option<f64>,
    /// Minimum frame-id gap between expensive recovery PnP attempts.
    /// `1` preserves the original behaviour of trying on every failed
    /// frame. Larger values throttle global relocalization on long
    /// sequences where repeated full-map recovery dominates runtime.
    /// Values below `1` are treated as `1`.
    pub attempt_interval_frames: u64,
    /// Optional cap on consecutive failed relocalization attempts while
    /// the primary tracker remains lost. When `Some(max)`, the
    /// relocalizer stops running after `max` failed attempts until
    /// primary tracking or relocalization succeeds again. `None`
    /// preserves the legacy unbounded retry behaviour.
    pub max_consecutive_failed_attempts: Option<u64>,
    /// Optional pose-continuity gate against the last successfully
    /// tracked/relocalized pose. When `Some(max_m_per_frame)`,
    /// recoveries that pass the PnP gates are further rejected if the
    /// camera-centre translation from the last successful pose divided
    /// by the frame-id gap is greater than `max_m_per_frame`. `None`
    /// (default) preserves the legacy behaviour. This timestamp-free
    /// gate is intended for demos / datasets where frame ids are dense
    /// enough to make "meters per frame" a useful continuity proxy.
    pub max_translation_per_frame_from_last_success_meters: Option<f64>,
    /// Optional lower bound on the median-depth ratio between the
    /// recovered pose and the last successful pose, measured on the
    /// recovery PnP inlier landmarks:
    /// `median(depth_recovered(inliers)) / median(depth_last_success(inliers))`.
    /// `None` disables the lower-bound gate.
    pub min_inlier_depth_median_ratio_to_last_success: Option<f64>,
    /// Optional upper bound on the same median-depth ratio. `None`
    /// disables the upper-bound gate. This is a scale-aware recovery
    /// sanity check: a PnP solution can be smooth in translation while
    /// placing the matched map landmarks at an implausibly different
    /// depth scale.
    pub max_inlier_depth_median_ratio_to_last_success: Option<f64>,
    /// Number of consecutive recovery hypotheses that must pass the
    /// gates before the tracker state is overwritten. `1` preserves
    /// the legacy immediate-accept behaviour; larger values turn the
    /// relocalizer into a short confirmation window.
    pub confirmation_required_recoveries: usize,
    /// Optional continuity gate between consecutive recovery
    /// hypotheses in the confirmation window. When `Some(max)`, the
    /// camera-centre translation from the previous pending recovery to
    /// the current recovery, divided by frame-id gap, must be <= `max`
    /// or the confirmation chain restarts at the current recovery.
    pub confirmation_max_translation_per_frame_meters: Option<f64>,
}

impl Default for OnlineSlamRelocalizationConfig {
    fn default() -> Self {
        Self {
            min_inliers: 20,
            min_inlier_ratio: 0.3,
            max_mean_reprojection_error: Some(8.0),
            pose_prior_candidate_radius_meters: None,
            recent_keyframe_window: None,
            covisibility_local_map: None,
            appearance_retrieval_map: None,
            max_translation_from_imu_prediction_meters: None,
            attempt_interval_frames: 1,
            max_consecutive_failed_attempts: None,
            max_translation_per_frame_from_last_success_meters: None,
            min_inlier_depth_median_ratio_to_last_success: None,
            max_inlier_depth_median_ratio_to_last_success: None,
            confirmation_required_recoveries: 1,
            confirmation_max_translation_per_frame_meters: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnlineSlamRelocalizationCovisibilityConfig {
    /// Cap on number of co-visible neighbor keyframes to include. The
    /// reference keyframe itself is always included. `None` means no
    /// cap.
    pub max_neighbor_keyframes: Option<usize>,
    /// Minimum shared landmarks required for a neighbor keyframe to
    /// enter the local descriptor store.
    pub min_shared_landmarks: usize,
    /// If the selected descriptor store has fewer descriptors than
    /// this, fall back to the broader recovery descriptor store.
    pub min_local_map_landmarks: usize,
    /// When the covisibility-local store is available but its recovery
    /// result fails the configured acceptance gates, retry the same
    /// recovery attempt with the broader full-map / recent-window
    /// descriptor store. This keeps the local store as a precision
    /// first pass while preserving recall on frames where the local
    /// neighbourhood is too narrow.
    pub fallback_to_broader_store_on_failure: bool,
    /// Minimum frame-id gap between broader descriptor-store retries
    /// after a covisibility-local first pass. Values below `1` are
    /// treated as `1`. This bounds the cost of local-first recovery on
    /// long dead-tracking stretches.
    pub broader_store_retry_interval_frames: u64,
    /// When true, also run the broader descriptor store even if the
    /// covisibility-local result passed the acceptance gates, then keep
    /// the accepted result with the stronger inlier/reprojection score.
    /// Off by default because it roughly doubles recovery-PnP cost.
    pub compare_broader_store_on_success: bool,
}

impl Default for OnlineSlamRelocalizationCovisibilityConfig {
    fn default() -> Self {
        Self {
            max_neighbor_keyframes: Some(10),
            min_shared_landmarks: 15,
            min_local_map_landmarks: 30,
            fallback_to_broader_store_on_failure: true,
            broader_store_retry_interval_frames: 10,
            compare_broader_store_on_success: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OnlineSlamRelocalizationAppearanceConfig {
    /// Keep at most this many appearance-nearest keyframes as recovery
    /// map seeds. Values below `1` are treated as `1`.
    pub max_keyframes: usize,
    /// Optional cap for ranked appearance candidates reported through
    /// [`OnlineSlamRelocalizationStats::appearance_candidates`].
    /// `None` preserves the recovery-store cap (`max_keyframes`). Set
    /// this higher than `max_keyframes` to evaluate top-K retrieval
    /// recall without increasing recovery-PnP cost.
    pub candidate_log_limit: Option<usize>,
    /// Minimum cosine similarity between the failed frame's mean local
    /// descriptor and a keyframe's mean local descriptor.
    pub min_similarity: f32,
    /// Exclude keyframes whose frame-id gap to the failed frame is
    /// smaller than this. This avoids using only near-temporal frames as
    /// a fake "retrieval" signal. `0` disables the exclusion.
    pub exclude_recent_frame_gap: u64,
    /// If the retrieved keyframes observe fewer descriptors than this,
    /// fall back to the broader recovery descriptor store.
    pub min_local_map_landmarks: usize,
    /// Retry the broader full-map / recent-window descriptor store when
    /// the appearance-retrieval first pass fails the acceptance gates.
    pub fallback_to_broader_store_on_failure: bool,
    /// Minimum frame-id gap between broader descriptor-store retries
    /// after an appearance-retrieval first pass. Values below `1` are
    /// treated as `1`.
    pub broader_store_retry_interval_frames: u64,
    /// When true, also run the broader descriptor store even if the
    /// appearance-retrieval result passed the gates, then keep the
    /// accepted result with the stronger inlier/reprojection score.
    pub compare_broader_store_on_success: bool,
}

impl Default for OnlineSlamRelocalizationAppearanceConfig {
    fn default() -> Self {
        Self {
            max_keyframes: 5,
            candidate_log_limit: None,
            min_similarity: 0.2,
            exclude_recent_frame_gap: 30,
            min_local_map_landmarks: 30,
            fallback_to_broader_store_on_failure: true,
            broader_store_retry_interval_frames: 10,
            compare_broader_store_on_success: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OnlineSlamRelocalizationPendingConfirmation {
    pub frame_id: u64,
    pub pose: Pose,
    pub count: usize,
}

/// Running state for the relocalization-on-tracker-death stage. Lives
/// on [`OnlineSlamPipeline`] when [`OnlineSlamConfig::relocalization`]
/// is `Some`. Owns a dedicated [`LocalizationPipeline`] with default
/// thresholds so the relocalization attempt is independent of the
/// tracker's own (typically much stricter, motion-prior-dependent)
/// localization pipeline.
#[derive(Debug, Clone)]
pub struct OnlineSlamRelocalizationState {
    pub config: OnlineSlamRelocalizationConfig,
    pub localizer: LocalizationPipeline,
    pub trigger_count: u64,
    pub success_count: u64,
    pub consecutive_failed_attempts: u64,
    pub budget_skip_count: u64,
    pub last_attempt_frame_id: Option<u64>,
    pub last_broader_descriptor_store_retry_frame_id: Option<u64>,
    pub last_success_frame_id: Option<u64>,
    pub pending_confirmation: Option<OnlineSlamRelocalizationPendingConfirmation>,
}

impl PartialEq for OnlineSlamRelocalizationState {
    fn eq(&self, other: &Self) -> bool {
        // The owned `LocalizationPipeline` does not implement `PartialEq`
        // (it carries closures + RANSAC RNG state). Compare only the
        // observable counters + config, matching the convention used by
        // [`OnlineSlamImuState`] / [`OnlineSlamLocalBaState`].
        self.config == other.config
            && self.trigger_count == other.trigger_count
            && self.success_count == other.success_count
            && self.consecutive_failed_attempts == other.consecutive_failed_attempts
            && self.budget_skip_count == other.budget_skip_count
            && self.last_attempt_frame_id == other.last_attempt_frame_id
            && self.last_broader_descriptor_store_retry_frame_id
                == other.last_broader_descriptor_store_retry_frame_id
            && self.last_success_frame_id == other.last_success_frame_id
            && self.pending_confirmation == other.pending_confirmation
    }
}

impl OnlineSlamRelocalizationState {
    fn new(config: OnlineSlamRelocalizationConfig) -> Self {
        Self {
            config,
            localizer: LocalizationPipeline::default(),
            trigger_count: 0,
            success_count: 0,
            consecutive_failed_attempts: 0,
            budget_skip_count: 0,
            last_attempt_frame_id: None,
            last_broader_descriptor_store_retry_frame_id: None,
            last_success_frame_id: None,
            pending_confirmation: None,
        }
    }

    fn reset(&mut self) {
        self.trigger_count = 0;
        self.success_count = 0;
        self.consecutive_failed_attempts = 0;
        self.budget_skip_count = 0;
        self.last_attempt_frame_id = None;
        self.last_broader_descriptor_store_retry_frame_id = None;
        self.last_success_frame_id = None;
        self.pending_confirmation = None;
    }
}

/// One appearance-retrieval keyframe candidate considered by the
/// relocalization recovery path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnlineSlamRelocalizationAppearanceCandidate {
    pub keyframe_id: u64,
    pub similarity: f32,
}

/// Per-frame outcome of the relocalization-on-tracker-death stage.
/// Exposed on [`OnlineSlamResult::relocalization`]; `Some` only on
/// frames where the stage actually attempted a recovery (i.e. the
/// primary tracker returned `success == false` AND the stage was
/// enabled).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OnlineSlamRelocalizationStats {
    /// `true` iff the stage ran its localizer this frame.
    pub attempted: bool,
    /// `true` iff the recovery localizer produced a successful PnP
    /// result before the relocalization-specific acceptance gates were
    /// applied.
    pub localization_success: bool,
    /// `true` iff the recovered solution cleared every gate AND the
    /// tracker accepted it via `accept_relocalization_result`.
    pub succeeded: bool,
    /// Inlier count reported by the recovery PnP solve. `0` when the
    /// localizer returned no solution at all (e.g. zero correspondences).
    pub inlier_count: usize,
    /// Inlier ratio reported by the recovery PnP solve.
    pub inlier_ratio: f64,
    /// Total correspondence count the recovery PnP solve consumed.
    pub correspondence_count: usize,
    /// Mean reprojection error reported by the recovery PnP solve.
    pub mean_reprojection_error: Option<f64>,
    /// Camera-centre translation between the last successful tracked
    /// pose and the recovered pose, if both are available.
    pub translation_from_last_success_meters: Option<f64>,
    /// `translation_from_last_success_meters` divided by the frame-id
    /// gap to the last successful frame, if both are available. This is
    /// populated even when the optional continuity gate is disabled so
    /// runners can inspect candidate recoveries before choosing a cap.
    pub translation_per_frame_from_last_success_meters: Option<f64>,
    /// Median positive depth of recovery-PnP inlier landmarks under
    /// the recovered pose.
    pub inlier_depth_median_meters: Option<f64>,
    /// Median positive depth of the same inlier landmarks under the
    /// last successful tracker pose.
    pub last_success_inlier_depth_median_meters: Option<f64>,
    /// `inlier_depth_median_meters / last_success_inlier_depth_median_meters`.
    /// Populated even when the optional depth-ratio gate is disabled.
    pub inlier_depth_median_ratio_to_last_success: Option<f64>,
    /// True when the recovered solution cleared the configured PnP,
    /// IMU, continuity, and depth-ratio gates before the optional
    /// confirmation window was applied.
    pub passed_acceptance_gates: bool,
    /// Current consecutive confirmation count after processing this
    /// recovery candidate. Zero when the acceptance gates failed.
    pub confirmation_count: usize,
    /// Required confirmation count configured for this attempt. Values
    /// below one are reported as one.
    pub confirmation_required_count: usize,
    /// Camera-centre translation per frame between the previous
    /// pending recovery and this candidate, if a previous pending
    /// recovery existed.
    pub confirmation_translation_per_frame_from_previous_meters: Option<f64>,
    /// Number of descriptors available to the recovery localizer after
    /// applying any full-map / active-frontier / covisibility-local
    /// store selection.
    pub descriptor_store_landmark_count: usize,
    /// Number of descriptors in the covisibility-local recovery store
    /// when that store was built.
    pub covisibility_local_descriptor_store_landmark_count: Option<usize>,
    /// Number of descriptors in the appearance-retrieval recovery store
    /// when that store was built and used as the first pass.
    pub appearance_descriptor_store_landmark_count: Option<usize>,
    /// Number of descriptors in the broader full-map / recent-window
    /// recovery store when a broader retry was attempted.
    pub broader_descriptor_store_landmark_count: Option<usize>,
    /// True when the recovery attempt tried the covisibility-local
    /// descriptor store before any optional broader retry.
    pub tried_covisibility_local_descriptor_store: bool,
    /// True when the selected recovery result came from the
    /// covisibility-local descriptor store.
    pub used_covisibility_local_descriptor_store: bool,
    /// True when the recovery attempt tried the appearance-retrieval
    /// descriptor store before any optional broader retry.
    pub tried_appearance_descriptor_store: bool,
    /// True when the selected recovery result came from the
    /// appearance-retrieval descriptor store.
    pub used_appearance_descriptor_store: bool,
    /// True when the broader full-map / recent-window descriptor store
    /// was attempted after a narrow first pass.
    pub tried_broader_descriptor_store_fallback: bool,
    /// True when a broader retry would otherwise have run but was
    /// skipped by `broader_store_retry_interval_frames`.
    pub broader_descriptor_store_retry_skipped_by_interval: bool,
    /// True when the selected recovery result came from a broader retry
    /// rather than the covisibility-local first pass.
    pub used_broader_descriptor_store_fallback: bool,
    /// Reference keyframe used for covisibility-local recovery store
    /// selection.
    pub covisibility_reference_keyframe_id: Option<u64>,
    /// Number of keyframes that passed the appearance similarity /
    /// temporal-gap filters.
    pub appearance_candidate_keyframe_count: usize,
    /// Highest appearance similarity among retrieved keyframes.
    pub appearance_best_similarity: Option<f32>,
    /// Best retrieved keyframe id by appearance similarity.
    pub appearance_best_keyframe_id: Option<u64>,
    /// Top retrieved keyframes retained for the appearance-retrieval
    /// recovery store, sorted by descending similarity. Empty when the
    /// appearance policy was disabled or no candidate passed its filters.
    pub appearance_candidates: Vec<OnlineSlamRelocalizationAppearanceCandidate>,
}

fn median_positive_depth_for_landmarks(
    map: &VisualMap,
    pose: &Pose,
    landmark_ids: &[u64],
) -> Option<f64> {
    let mut depths: Vec<f64> = landmark_ids
        .iter()
        .filter_map(|landmark_id| map.landmarks.get(landmark_id))
        .map(|landmark| pose.transform_world_point(&landmark.position).z)
        .filter(|depth| depth.is_finite() && *depth > 0.0)
        .collect();
    if depths.is_empty() {
        return None;
    }
    depths.sort_by(|left, right| left.total_cmp(right));
    let mid = depths.len() / 2;
    if depths.len() % 2 == 0 {
        Some((depths[mid - 1] + depths[mid]) * 0.5)
    } else {
        Some(depths[mid])
    }
}

fn relocalization_pick_covisibility_reference_keyframe(
    map: &VisualMap,
    last_id: u64,
) -> Option<u64> {
    if map.keyframes.contains_key(&last_id) {
        return Some(last_id);
    }
    let mut best: Option<u64> = None;
    for keyframe_id in map.keyframes.keys() {
        if *keyframe_id > last_id {
            continue;
        }
        match best {
            None => best = Some(*keyframe_id),
            Some(current) if *keyframe_id > current => best = Some(*keyframe_id),
            _ => {}
        }
    }
    best
}

fn relocalization_covisibility_descriptor_store(
    map: &VisualMap,
    reference_keyframe_id: u64,
    config: &OnlineSlamRelocalizationCovisibilityConfig,
) -> Option<visloc_core::types::LandmarkDescriptorStore> {
    let reference_keyframe = map.keyframes.get(&reference_keyframe_id)?;
    let reference_landmarks: std::collections::HashSet<u64> = reference_keyframe
        .observations
        .iter()
        .map(|observation| observation.landmark_id)
        .collect();
    if reference_landmarks.is_empty() {
        return None;
    }

    let mut local_landmarks = reference_landmarks.clone();
    let mut shared_counts: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    for landmark_id in &reference_landmarks {
        let Some(landmark) = map.landmarks.get(landmark_id) else {
            continue;
        };
        for observation in &landmark.observations {
            let keyframe_id = observation.frame_id;
            if keyframe_id == reference_keyframe_id || !map.keyframes.contains_key(&keyframe_id) {
                continue;
            }
            *shared_counts.entry(keyframe_id).or_insert(0) += 1;
        }
    }

    let mut ranked_neighbors: Vec<(u64, usize)> = shared_counts
        .into_iter()
        .filter(|(_, count)| *count >= config.min_shared_landmarks)
        .collect();
    ranked_neighbors.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    if let Some(cap) = config.max_neighbor_keyframes {
        ranked_neighbors.truncate(cap);
    }

    for (keyframe_id, _) in ranked_neighbors {
        if let Some(keyframe) = map.keyframes.get(&keyframe_id) {
            for observation in &keyframe.observations {
                local_landmarks.insert(observation.landmark_id);
            }
        }
    }

    let mut local_landmark_ids: Vec<u64> = local_landmarks.into_iter().collect();
    local_landmark_ids.sort_unstable();
    let mut store = visloc_core::types::LandmarkDescriptorStore::new();
    for landmark_id in local_landmark_ids {
        if let Some(descriptor) = map
            .landmarks
            .get(&landmark_id)
            .and_then(|landmark| landmark.descriptor.as_ref())
        {
            store.insert(landmark_id, descriptor.clone());
        }
    }
    if store.len() < config.min_local_map_landmarks {
        return None;
    }
    Some(store)
}

struct RelocalizationAppearanceDescriptorStore {
    store: visloc_core::types::LandmarkDescriptorStore,
    candidate_keyframe_count: usize,
    best_keyframe_id: Option<u64>,
    best_similarity: Option<f32>,
    candidates: Vec<OnlineSlamRelocalizationAppearanceCandidate>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RelocalizationDescriptorStoreKind {
    Broader,
    CovisibilityLocal,
    AppearanceRetrieval,
}

pub(crate) fn relocalization_mean_descriptor(descriptors: &[Vec<f32>]) -> Option<Vec<f32>> {
    let first = descriptors.first()?;
    if first.is_empty()
        || descriptors
            .iter()
            .any(|descriptor| descriptor.len() != first.len())
    {
        return None;
    }
    let mut mean = vec![0.0_f32; first.len()];
    for descriptor in descriptors {
        for (acc, value) in mean.iter_mut().zip(descriptor) {
            *acc += *value;
        }
    }
    let inv_count = 1.0_f32 / descriptors.len() as f32;
    let mut norm = 0.0_f32;
    for value in &mut mean {
        *value *= inv_count;
        norm += *value * *value;
    }
    let norm = norm.sqrt();
    if norm <= 1.0e-12 {
        return None;
    }
    for value in &mut mean {
        *value /= norm;
    }
    Some(mean)
}

pub(crate) fn relocalization_descriptor_cosine(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let dot: f32 = left.iter().zip(right).map(|(l, r)| l * r).sum();
    let left_norm = left.iter().map(|v| v * v).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|v| v * v).sum::<f32>().sqrt();
    if left_norm <= 1.0e-12 || right_norm <= 1.0e-12 {
        0.0
    } else {
        dot / (left_norm * right_norm)
    }
}

fn relocalization_appearance_descriptor_store(
    map: &VisualMap,
    frame: &Frame,
    config: &OnlineSlamRelocalizationAppearanceConfig,
) -> Option<RelocalizationAppearanceDescriptorStore> {
    let query_descriptor = relocalization_mean_descriptor(&frame.descriptors)?;
    let mut ranked_keyframes: Vec<(u64, f32)> = Vec::new();
    for (keyframe_id, keyframe) in &map.keyframes {
        if *keyframe_id >= frame.id {
            continue;
        }
        if frame.id.saturating_sub(*keyframe_id) < config.exclude_recent_frame_gap {
            continue;
        }
        let Some(keyframe_descriptor) = relocalization_mean_descriptor(&keyframe.frame.descriptors)
        else {
            continue;
        };
        let similarity = relocalization_descriptor_cosine(&query_descriptor, &keyframe_descriptor);
        if similarity >= config.min_similarity {
            ranked_keyframes.push((*keyframe_id, similarity));
        }
    }
    ranked_keyframes.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    let candidate_keyframe_count = ranked_keyframes.len();
    let best_keyframe_id = ranked_keyframes
        .first()
        .map(|(keyframe_id, _)| *keyframe_id);
    let best_similarity = ranked_keyframes.first().map(|(_, similarity)| *similarity);
    let recovery_keyframe_count = config.max_keyframes.max(1);
    let log_keyframe_count = config
        .candidate_log_limit
        .unwrap_or(recovery_keyframe_count)
        .max(1);
    let candidates: Vec<OnlineSlamRelocalizationAppearanceCandidate> = ranked_keyframes
        .iter()
        .take(log_keyframe_count)
        .map(
            |(keyframe_id, similarity)| OnlineSlamRelocalizationAppearanceCandidate {
                keyframe_id: *keyframe_id,
                similarity: *similarity,
            },
        )
        .collect();

    let mut landmark_ids = std::collections::HashSet::new();
    for (keyframe_id, _) in ranked_keyframes.iter().take(recovery_keyframe_count) {
        if let Some(keyframe) = map.keyframes.get(keyframe_id) {
            for observation in &keyframe.observations {
                landmark_ids.insert(observation.landmark_id);
            }
        }
    }
    let mut landmark_ids: Vec<u64> = landmark_ids.into_iter().collect();
    landmark_ids.sort_unstable();
    let mut store = visloc_core::types::LandmarkDescriptorStore::new();
    for landmark_id in landmark_ids {
        if let Some(descriptor) = map
            .landmarks
            .get(&landmark_id)
            .and_then(|landmark| landmark.descriptor.as_ref())
        {
            store.insert(landmark_id, descriptor.clone());
        }
    }
    if store.len() < config.min_local_map_landmarks {
        return None;
    }
    Some(RelocalizationAppearanceDescriptorStore {
        store,
        candidate_keyframe_count,
        best_keyframe_id,
        best_similarity,
        candidates,
    })
}

/// Build a [`visloc_core::types::LandmarkDescriptorStore`] scoped to a
/// single keyframe's own observed landmarks (no covisible-neighbor
/// expansion). Used by [`build_appearance_loop_candidates`] to restrict
/// descriptor-based correspondence building to exactly the candidate
/// keyframe's landmarks, since appearance candidates by construction share
/// no landmark ids with the current frame. Returns `None` when the
/// keyframe has no observations at all.
pub fn appearance_loop_candidate_descriptor_store(
    map: &VisualMap,
    keyframe: &Keyframe,
) -> Option<visloc_core::types::LandmarkDescriptorStore> {
    let mut landmark_ids: Vec<u64> = keyframe
        .observations
        .iter()
        .map(|observation| observation.landmark_id)
        .collect();
    landmark_ids.sort_unstable();
    landmark_ids.dedup();
    if landmark_ids.is_empty() {
        return None;
    }
    let mut store = visloc_core::types::LandmarkDescriptorStore::new();
    for landmark_id in landmark_ids {
        if let Some(descriptor) = map
            .landmarks
            .get(&landmark_id)
            .and_then(|landmark| landmark.descriptor.as_ref())
        {
            store.insert(landmark_id, descriptor.clone());
        }
    }
    Some(store)
}

fn keyframes_ranked_by_covisibility(
    map: &VisualMap,
    seed_keyframe_id: u64,
    min_shared_landmarks: usize,
    max_keyframes: usize,
) -> Vec<u64> {
    let Some(seed) = map.keyframes.get(&seed_keyframe_id) else {
        return Vec::new();
    };
    let seed_landmarks: std::collections::HashSet<_> = seed
        .observations
        .iter()
        .map(|observation| observation.landmark_id)
        .collect();
    let mut ranked: Vec<(u64, usize)> = map
        .keyframes
        .iter()
        .filter(|(keyframe_id, _)| **keyframe_id != seed_keyframe_id)
        .filter_map(|(keyframe_id, keyframe)| {
            let shared = keyframe
                .observations
                .iter()
                .filter(|observation| seed_landmarks.contains(&observation.landmark_id))
                .count();
            (shared >= min_shared_landmarks.max(1)).then_some((*keyframe_id, shared))
        })
        .collect();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    ranked.truncate(max_keyframes);
    ranked
        .into_iter()
        .map(|(keyframe_id, _)| keyframe_id)
        .collect()
}

fn appearance_loop_region_descriptor_store(
    map: &VisualMap,
    keyframe_ids: &[u64],
) -> visloc_core::types::LandmarkDescriptorStore {
    let mut landmark_ids = std::collections::BTreeSet::new();
    for keyframe_id in keyframe_ids {
        if let Some(keyframe) = map.keyframes.get(keyframe_id) {
            landmark_ids.extend(
                keyframe
                    .observations
                    .iter()
                    .map(|observation| observation.landmark_id),
            );
        }
    }
    let mut store = visloc_core::types::LandmarkDescriptorStore::new();
    for landmark_id in landmark_ids {
        if let Some(descriptor) = map
            .landmarks
            .get(&landmark_id)
            .and_then(|landmark| landmark.descriptor.as_ref())
        {
            store.insert(landmark_id, descriptor.clone());
        }
    }
    store
}

/// Rank the cached per-keyframe mean descriptors in `descriptor_cache` by
/// cosine similarity against `query_descriptor`, keeping only entries
/// strictly older than `frame_id` and at least `min_keyframe_id_gap`
/// keyframe-ids older. Ties broken by ascending keyframe id for
/// determinism. Shared between [`build_appearance_loop_candidates`] and its
/// unit tests.
fn rank_appearance_loop_candidate_keyframes(
    descriptor_cache: &HashMap<u64, Vec<f32>>,
    frame_id: u64,
    query_descriptor: &[f32],
    min_similarity: f32,
    min_keyframe_id_gap: u64,
) -> Vec<(u64, f32)> {
    let mut ranked: Vec<(u64, f32)> = descriptor_cache
        .iter()
        .filter(|(&keyframe_id, _)| {
            keyframe_id < frame_id && frame_id.saturating_sub(keyframe_id) >= min_keyframe_id_gap
        })
        .map(|(&keyframe_id, descriptor)| {
            (
                keyframe_id,
                relocalization_descriptor_cosine(query_descriptor, descriptor),
            )
        })
        .filter(|(_, similarity)| *similarity >= min_similarity)
        .collect();
    ranked.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked
}

/// Build and PnP-verify appearance-based long-range loop-closure candidates
/// for `frame` against the per-keyframe mean descriptors cached in
/// `descriptor_cache` (see
/// [`OnlineSlamLoopClosureRefinementState::appearance_descriptor_cache`]).
///
/// Ranks past keyframes at least `config.min_keyframe_id_gap` keyframe-ids
/// older than `frame.id` by cosine similarity of their cached mean
/// descriptor against `frame`'s mean descriptor (reusing the same
/// `relocalization_mean_descriptor` / `relocalization_descriptor_cosine`
/// machinery [`OnlineSlamRelocalizationAppearanceConfig`]'s retrieval store
/// uses), keeps the top `config.max_candidates_per_frame`, and for each one:
///
/// 1. builds a [`visloc_core::types::LandmarkDescriptorStore`] scoped to
///    that keyframe's own observed landmarks
///    ([`appearance_loop_candidate_descriptor_store`]);
/// 2. builds 2D-3D correspondences by matching `frame`'s descriptors
///    against that store via
///    [`visloc_localization::CorrespondenceBuilder`] — descriptor
///    appearance alone, since appearance candidates share no landmark ids
///    with `frame` by construction;
/// 3. runs [`PnPLoopClosureVerifier`] (configured by `config.pnp_verifier`)
///    on the resulting correspondences against the candidate keyframe's
///    stored pose.
///
/// Returns one [`LoopClosureCandidate`] per candidate keyframe that passed
/// the PnP verifier's acceptance gates, with `verification` populated —
/// ready for [`LoopClosureConstraint::from_verified_candidate`], the same
/// as the shared-landmark detector's output. Candidates without a stored
/// pose, without a viable descriptor store (fewer than
/// `config.min_candidate_landmark_count` descriptors), or without a
/// correspondence match are silently skipped.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AppearanceLoopCandidateBuildResult {
    pub candidates: Vec<LoopClosureCandidate>,
    pub connected_region_rejected_count: usize,
    pub pnp_verified_count: usize,
    pub projection_rejected_count: usize,
    pub covisibility_rejected_count: usize,
    pub diagnostics: Vec<AppearanceLoopCandidateDiagnostic>,
}

/// Per-candidate evidence retained after primary PnP succeeds, so threshold
/// changes can be based on measured covisibility support rather than only an
/// aggregate accept/reject count.
#[derive(Debug, Clone, PartialEq)]
pub struct AppearanceLoopCandidateDiagnostic {
    pub query_frame_id: u64,
    pub matched_keyframe_id: u64,
    pub appearance_similarity: f32,
    pub matched_region_keyframe_count: usize,
    pub matched_region_landmark_count: usize,
    pub primary_correspondence_count: usize,
    pub primary_inlier_count: usize,
    pub projection_attempted: bool,
    pub projection_correspondence_count: usize,
    pub projection_inlier_count: usize,
    pub projection_accepted: bool,
    pub current_covisible_keyframe_count: usize,
    pub neighbor_pnp_verified_count: usize,
    pub consistent_keyframe_count: usize,
    pub minimum_translation_disagreement_meters: Option<f64>,
    pub minimum_rotation_disagreement_radians: Option<f64>,
    pub accepted: bool,
}

pub fn build_appearance_loop_candidates(
    map: &VisualMap,
    frame: &Frame,
    descriptor_cache: &HashMap<u64, Vec<f32>>,
    config: &LoopAppearanceCandidateConfig,
    camera: &Camera,
) -> Vec<LoopClosureCandidate> {
    build_appearance_loop_candidates_with_diagnostics(map, frame, descriptor_cache, config, camera)
        .candidates
}

pub fn build_appearance_loop_candidates_with_diagnostics(
    map: &VisualMap,
    frame: &Frame,
    descriptor_cache: &HashMap<u64, Vec<f32>>,
    config: &LoopAppearanceCandidateConfig,
    camera: &Camera,
) -> AppearanceLoopCandidateBuildResult {
    let Some(query_descriptor) = relocalization_mean_descriptor(&frame.descriptors) else {
        return AppearanceLoopCandidateBuildResult::default();
    };
    let mut ranked = rank_appearance_loop_candidate_keyframes(
        descriptor_cache,
        frame.id,
        &query_descriptor,
        config.min_similarity,
        config.min_keyframe_id_gap,
    );
    ranked.truncate(config.max_candidates_per_frame.max(1));

    let query = visloc_core::types::QueryImage::from_frame(frame, camera.clone());
    let matcher = visloc_vision::matching::BruteForceMatcher::default();
    let correspondence_builder = visloc_localization::CorrespondenceBuilder::new(matcher);
    let verifier = PnPLoopClosureVerifier {
        ransac: PnPRansac::default(),
        config: config.pnp_verifier,
    };

    let mut result = AppearanceLoopCandidateBuildResult::default();
    let current_connected_keyframes: std::collections::HashSet<u64> =
        keyframes_ranked_by_covisibility(
            map,
            frame.id,
            config.covisibility_min_shared_landmarks,
            usize::MAX,
        )
        .into_iter()
        .chain(std::iter::once(frame.id))
        .collect();
    for (keyframe_id, similarity) in ranked {
        let Some(keyframe) = map.keyframes.get(&keyframe_id) else {
            continue;
        };
        let Some(keyframe_pose) = keyframe.frame.pose.as_ref() else {
            continue;
        };
        let mut matched_region_keyframe_ids = vec![keyframe_id];
        matched_region_keyframe_ids.extend(keyframes_ranked_by_covisibility(
            map,
            keyframe_id,
            config.covisibility_min_shared_landmarks,
            config.covisibility_max_keyframes,
        ));
        if matched_region_keyframe_ids
            .iter()
            .any(|keyframe_id| current_connected_keyframes.contains(keyframe_id))
        {
            result.connected_region_rejected_count += 1;
            continue;
        }
        let store = appearance_loop_region_descriptor_store(map, &matched_region_keyframe_ids);
        if store.len() < config.min_candidate_landmark_count {
            continue;
        }
        let Ok(mut correspondence_set) = correspondence_builder.build(&query, map, &store) else {
            continue;
        };
        let (mut verification, mut pnp_inlier_indices) = verifier.verify_with_inlier_indices(
            &correspondence_set.correspondences,
            keyframe_pose,
            camera,
        );
        if !verification.verified {
            continue;
        }
        result.pnp_verified_count += 1;
        let primary_correspondence_count = verification.correspondence_count;
        let primary_inlier_count = verification.inlier_count;
        let mut projection_attempted = false;
        let mut projection_correspondence_count = 0usize;
        let mut projection_inlier_count = 0usize;
        let mut projection_accepted = false;
        if let Some(search_radius_px) = config.projection_search_radius_px {
            projection_attempted = true;
            if let Some(primary_relative_pose) = verification.relative_pose.as_ref() {
                let recovered_current_pose = Pose {
                    world_to_camera: primary_relative_pose.compose(&keyframe_pose.world_to_camera),
                };
                let projection_builder = visloc_localization::ProjectionCorrespondenceBuilder::new(
                    visloc_vision::matching::BruteForceMatcher::default(),
                );
                if let Ok(projected_set) = projection_builder.build_with_pose_prior(
                    &query,
                    map,
                    &store,
                    &recovered_current_pose,
                    search_radius_px,
                ) {
                    projection_correspondence_count = projected_set.correspondences.len();
                    if projection_correspondence_count >= config.min_projection_correspondence_count
                    {
                        let (projected_verification, projected_inlier_indices) = verifier
                            .verify_with_inlier_indices(
                                &projected_set.correspondences,
                                keyframe_pose,
                                camera,
                            );
                        projection_inlier_count = projected_verification.inlier_count;
                        if projected_verification.verified {
                            projection_accepted = true;
                            correspondence_set = projected_set;
                            verification = projected_verification;
                            pnp_inlier_indices = projected_inlier_indices;
                        }
                    }
                }
            }
        }
        let required_verifications = config.min_covisible_keyframe_verifications.max(1);
        let mut region_verifications = 1usize;
        let current_covisible_keyframe_ids = keyframes_ranked_by_covisibility(
            map,
            frame.id,
            config.covisibility_min_shared_landmarks,
            config.covisibility_max_keyframes,
        );
        let mut diagnostic = AppearanceLoopCandidateDiagnostic {
            query_frame_id: frame.id,
            matched_keyframe_id: keyframe_id,
            appearance_similarity: similarity,
            matched_region_keyframe_count: matched_region_keyframe_ids.len(),
            matched_region_landmark_count: store.len(),
            primary_correspondence_count,
            primary_inlier_count,
            projection_attempted,
            projection_correspondence_count,
            projection_inlier_count,
            projection_accepted,
            current_covisible_keyframe_count: current_covisible_keyframe_ids.len(),
            neighbor_pnp_verified_count: 0,
            consistent_keyframe_count: 1,
            minimum_translation_disagreement_meters: None,
            minimum_rotation_disagreement_radians: None,
            accepted: false,
        };
        if projection_attempted && !projection_accepted {
            result.projection_rejected_count += 1;
            result.diagnostics.push(diagnostic);
            continue;
        }
        if required_verifications > 1 {
            let Some(primary_relative_pose) = verification.relative_pose.as_ref() else {
                result.covisibility_rejected_count += 1;
                result.diagnostics.push(diagnostic);
                continue;
            };
            let Some(current_pose) = map
                .keyframes
                .get(&frame.id)
                .and_then(|keyframe| keyframe.frame.pose.as_ref())
            else {
                result.covisibility_rejected_count += 1;
                result.diagnostics.push(diagnostic);
                continue;
            };
            for current_neighbor_id in current_covisible_keyframe_ids {
                let Some(current_neighbor) = map.keyframes.get(&current_neighbor_id) else {
                    continue;
                };
                let neighbor_query = visloc_core::types::QueryImage::from_frame(
                    &current_neighbor.frame,
                    camera.clone(),
                );
                let Ok(neighbor_correspondences) =
                    correspondence_builder.build(&neighbor_query, map, &store)
                else {
                    continue;
                };
                let neighbor_verification = verifier.verify(
                    &neighbor_correspondences.correspondences,
                    keyframe_pose,
                    camera,
                );
                if neighbor_verification.verified {
                    diagnostic.neighbor_pnp_verified_count += 1;
                    let Some(neighbor_relative_pose) = neighbor_verification.relative_pose.as_ref()
                    else {
                        continue;
                    };
                    let Some(neighbor_pose) = current_neighbor.frame.pose.as_ref() else {
                        continue;
                    };
                    let current_to_neighbor = relative_world_to_camera(current_pose, neighbor_pose);
                    let expected_neighbor_relative =
                        current_to_neighbor.compose(primary_relative_pose);
                    let disagreement =
                        neighbor_relative_pose.compose(&expected_neighbor_relative.inverse());
                    let translation_disagreement = disagreement.translation.norm();
                    let rotation_disagreement = disagreement.rotation.angle();
                    diagnostic.minimum_translation_disagreement_meters = Some(
                        diagnostic
                            .minimum_translation_disagreement_meters
                            .map_or(translation_disagreement, |value| {
                                value.min(translation_disagreement)
                            }),
                    );
                    diagnostic.minimum_rotation_disagreement_radians = Some(
                        diagnostic
                            .minimum_rotation_disagreement_radians
                            .map_or(rotation_disagreement, |value| {
                                value.min(rotation_disagreement)
                            }),
                    );
                    if translation_disagreement
                        <= config.max_covisibility_translation_disagreement_meters
                        && rotation_disagreement
                            <= config.max_covisibility_rotation_disagreement_radians
                    {
                        region_verifications += 1;
                        diagnostic.consistent_keyframe_count = region_verifications;
                    }
                    if region_verifications >= required_verifications {
                        break;
                    }
                }
            }
        }
        if region_verifications < required_verifications {
            result.covisibility_rejected_count += 1;
            result.diagnostics.push(diagnostic);
            continue;
        }
        diagnostic.accepted = true;
        result.diagnostics.push(diagnostic);
        result.candidates.push(LoopClosureCandidate {
            query_frame_id: frame.id,
            matched_keyframe_id: keyframe_id,
            shared_landmark_count: verification.inlier_count,
            query_inlier_count: verification.inlier_count,
            keyframe_observation_count: correspondence_set.candidate_landmark_count,
            shared_landmark_ratio: verification.inlier_ratio,
            score: f64::from(similarity) * verification.score,
            geometrically_verified: true,
            verification: Some(verification),
            pnp_query_landmark_pairs: pnp_inlier_indices
                .into_iter()
                .filter_map(|index| {
                    Some((
                        *correspondence_set.query_indices.get(index)?,
                        *correspondence_set.landmark_ids.get(index)?,
                    ))
                })
                .collect(),
        });
    }
    result
}

fn appearance_regions_overlap(
    map: &VisualMap,
    left_keyframe_id: u64,
    right_keyframe_id: u64,
    config: &LoopAppearanceCandidateConfig,
) -> bool {
    if left_keyframe_id == right_keyframe_id {
        return true;
    }
    let mut left_region = keyframes_ranked_by_covisibility(
        map,
        left_keyframe_id,
        config.covisibility_min_shared_landmarks,
        config.covisibility_max_keyframes,
    );
    left_region.push(left_keyframe_id);
    let right_region: std::collections::HashSet<u64> = keyframes_ranked_by_covisibility(
        map,
        right_keyframe_id,
        config.covisibility_min_shared_landmarks,
        config.covisibility_max_keyframes,
    )
    .into_iter()
    .chain(std::iter::once(right_keyframe_id))
    .collect();
    left_region
        .iter()
        .any(|keyframe_id| right_region.contains(keyframe_id))
}

#[derive(Debug, Clone, PartialEq, Default)]
struct AppearancePendingProjectionResult {
    attempted: bool,
    search_radius_px: Option<f64>,
    correspondence_count: usize,
    inlier_count: usize,
    candidate: Option<LoopClosureCandidate>,
}

/// Carry an already verified appearance region into the next keyframe by
/// projection matching under the pending region's recovered pose plus the
/// intervening odometry motion. This is the temporal-consistency step: later
/// confirmations do not depend on independently retrieving the same place by
/// a global descriptor again.
fn verify_pending_appearance_region_by_projection(
    map: &VisualMap,
    frame: &Frame,
    camera: &Camera,
    config: &LoopAppearanceCandidateConfig,
    pending: &AppearancePendingRegion,
) -> AppearancePendingProjectionResult {
    let Some(search_radius_px) = config.projection_search_radius_px else {
        return AppearancePendingProjectionResult::default();
    };
    let Some(root_keyframe) = map.keyframes.get(&pending.root_keyframe_id) else {
        return AppearancePendingProjectionResult::default();
    };
    let Some(root_pose) = root_keyframe.frame.pose.as_ref() else {
        return AppearancePendingProjectionResult::default();
    };
    let Some(last_current_pose) = map
        .keyframes
        .get(&pending.last_current_keyframe_id)
        .and_then(|keyframe| keyframe.frame.pose.as_ref())
    else {
        return AppearancePendingProjectionResult::default();
    };
    let Some(current_pose) = map
        .keyframes
        .get(&frame.id)
        .and_then(|keyframe| keyframe.frame.pose.as_ref())
    else {
        return AppearancePendingProjectionResult::default();
    };

    let mut region_keyframe_ids = vec![pending.root_keyframe_id];
    region_keyframe_ids.extend(keyframes_ranked_by_covisibility(
        map,
        pending.root_keyframe_id,
        config.covisibility_min_shared_landmarks,
        config.covisibility_max_keyframes,
    ));
    let store = appearance_loop_region_descriptor_store(map, &region_keyframe_ids);
    if store.len() < config.min_candidate_landmark_count {
        return AppearancePendingProjectionResult::default();
    }

    let last_to_current = relative_world_to_camera(last_current_pose, current_pose);
    let predicted_root_to_current = last_to_current.compose(&pending.root_to_last_current);
    let predicted_current_pose = Pose {
        world_to_camera: predicted_root_to_current.compose(&root_pose.world_to_camera),
    };
    let query = visloc_core::types::QueryImage::from_frame(frame, camera.clone());
    let mut result = AppearancePendingProjectionResult {
        attempted: true,
        ..AppearancePendingProjectionResult::default()
    };
    let verifier = PnPLoopClosureVerifier {
        ransac: PnPRansac::default(),
        config: config.pnp_verifier,
    };
    let mut radii = [
        (search_radius_px / 3.0).max(1.0),
        (search_radius_px * 2.0 / 3.0).max(1.0),
        search_radius_px,
    ];
    radii.sort_by(f64::total_cmp);
    let mut previous_radius: Option<f64> = None;
    let mut best_ratio = -1.0_f64;
    for radius in radii {
        if previous_radius.is_some_and(|previous| (radius - previous).abs() <= f64::EPSILON) {
            continue;
        }
        previous_radius = Some(radius);
        let projection_builder = visloc_localization::ProjectionCorrespondenceBuilder::new(
            visloc_vision::matching::BruteForceMatcher::default(),
        );
        let Ok(correspondence_set) = projection_builder.build_with_pose_prior(
            &query,
            map,
            &store,
            &predicted_current_pose,
            radius,
        ) else {
            continue;
        };
        let correspondence_count = correspondence_set.correspondences.len();
        if correspondence_count < config.min_projection_correspondence_count {
            if result.search_radius_px.is_none() {
                result.search_radius_px = Some(radius);
                result.correspondence_count = correspondence_count;
            }
            continue;
        }
        let (verification, pnp_inlier_indices) = verifier.verify_with_inlier_indices(
            &correspondence_set.correspondences,
            root_pose,
            camera,
        );
        let ratio = if correspondence_count == 0 {
            0.0
        } else {
            verification.inlier_count as f64 / correspondence_count as f64
        };
        if ratio > best_ratio
            || (ratio == best_ratio && verification.inlier_count > result.inlier_count)
        {
            best_ratio = ratio;
            result.search_radius_px = Some(radius);
            result.correspondence_count = correspondence_count;
            result.inlier_count = verification.inlier_count;
        }
        if !verification.verified {
            continue;
        }
        result.search_radius_px = Some(radius);
        result.correspondence_count = correspondence_count;
        result.inlier_count = verification.inlier_count;
        result.candidate = Some(LoopClosureCandidate {
            query_frame_id: frame.id,
            matched_keyframe_id: pending.root_keyframe_id,
            shared_landmark_count: verification.inlier_count,
            query_inlier_count: verification.inlier_count,
            keyframe_observation_count: correspondence_set.candidate_landmark_count,
            shared_landmark_ratio: verification.inlier_ratio,
            score: verification.score,
            geometrically_verified: true,
            verification: Some(verification),
            pnp_query_landmark_pairs: pnp_inlier_indices
                .into_iter()
                .filter_map(|index| {
                    Some((
                        *correspondence_set.query_indices.get(index)?,
                        *correspondence_set.landmark_ids.get(index)?,
                    ))
                })
                .collect(),
        });
        break;
    }
    result
}

fn confirm_appearance_region_across_keyframes(
    map: &VisualMap,
    current_keyframe_id: u64,
    candidates: Vec<LoopClosureCandidate>,
    config: &LoopAppearanceCandidateConfig,
    pending: &mut Option<AppearancePendingRegion>,
) -> (Vec<LoopClosureCandidate>, bool, bool) {
    let required = config.region_confirmation_required_keyframes.max(1);
    if required == 1 {
        *pending = None;
        return (candidates, false, false);
    }

    let Some(current_pose) = map
        .keyframes
        .get(&current_keyframe_id)
        .and_then(|keyframe| keyframe.frame.pose.as_ref())
    else {
        *pending = None;
        return (Vec::new(), false, false);
    };

    if pending.is_none() {
        let Some(candidate) = candidates
            .into_iter()
            .max_by(|left, right| left.score.total_cmp(&right.score))
        else {
            return (Vec::new(), false, false);
        };
        let Some(relative_pose) = candidate
            .verification
            .as_ref()
            .and_then(|verification| verification.relative_pose.clone())
        else {
            return (Vec::new(), false, false);
        };
        *pending = Some(AppearancePendingRegion {
            root_keyframe_id: candidate.matched_keyframe_id,
            last_current_keyframe_id: current_keyframe_id,
            root_to_last_current: relative_pose,
            confirmation_count: 1,
            miss_count: 0,
        });
        return (Vec::new(), true, false);
    }

    let state = pending.as_mut().expect("pending region checked above");
    let matching_candidate = candidates.into_iter().find(|candidate| {
        appearance_regions_overlap(
            map,
            state.root_keyframe_id,
            candidate.matched_keyframe_id,
            config,
        )
    });
    let Some(candidate) = matching_candidate else {
        state.miss_count += 1;
        if state.miss_count >= config.region_confirmation_max_misses.max(1) {
            *pending = None;
        }
        return (Vec::new(), true, false);
    };

    let recovered_candidate_to_current = candidate
        .verification
        .as_ref()
        .and_then(|verification| verification.relative_pose.as_ref());
    let root_pose = map
        .keyframes
        .get(&state.root_keyframe_id)
        .and_then(|keyframe| keyframe.frame.pose.as_ref());
    let candidate_pose = map
        .keyframes
        .get(&candidate.matched_keyframe_id)
        .and_then(|keyframe| keyframe.frame.pose.as_ref());
    let last_current_pose = map
        .keyframes
        .get(&state.last_current_keyframe_id)
        .and_then(|keyframe| keyframe.frame.pose.as_ref());
    let Some((recovered_candidate_to_current, root_pose, candidate_pose, last_current_pose)) =
        recovered_candidate_to_current
            .zip(root_pose)
            .zip(candidate_pose)
            .zip(last_current_pose)
            .map(|(((a, b), c), d)| (a, b, c, d))
    else {
        *pending = None;
        return (Vec::new(), false, false);
    };

    let root_to_candidate = relative_world_to_camera(root_pose, candidate_pose);
    let recovered_root_to_current = recovered_candidate_to_current.compose(&root_to_candidate);
    let last_to_current = relative_world_to_camera(last_current_pose, current_pose);
    let predicted_root_to_current = last_to_current.compose(&state.root_to_last_current);
    let disagreement = recovered_root_to_current.compose(&predicted_root_to_current.inverse());
    let consistent = disagreement.translation.norm()
        <= config.max_covisibility_translation_disagreement_meters
        && disagreement.rotation.angle() <= config.max_covisibility_rotation_disagreement_radians;
    if !consistent {
        state.miss_count += 1;
        if state.miss_count >= config.region_confirmation_max_misses.max(1) {
            *pending = None;
        }
        return (Vec::new(), true, false);
    }

    state.confirmation_count += 1;
    state.miss_count = 0;
    state.last_current_keyframe_id = current_keyframe_id;
    state.root_to_last_current = recovered_root_to_current;
    if state.confirmation_count < required {
        return (Vec::new(), true, false);
    }
    *pending = None;
    (vec![candidate], false, true)
}

fn relocalization_recent_keyframe_descriptor_store(
    map: &VisualMap,
    recent_keyframe_window: Option<usize>,
) -> visloc_core::types::LandmarkDescriptorStore {
    match recent_keyframe_window {
        Some(window) if window > 0 => {
            let mut keyframe_ids: Vec<u64> = map.keyframes.keys().copied().collect();
            keyframe_ids.sort();
            let start = keyframe_ids.len().saturating_sub(window);
            let mut active_landmark_ids: std::collections::HashSet<u64> =
                std::collections::HashSet::new();
            for keyframe_id in &keyframe_ids[start..] {
                if let Some(keyframe) = map.keyframes.get(keyframe_id) {
                    for observation in &keyframe.observations {
                        active_landmark_ids.insert(observation.landmark_id);
                    }
                }
            }
            let mut store = visloc_core::types::LandmarkDescriptorStore::new();
            let mut landmark_ids: Vec<u64> = active_landmark_ids.into_iter().collect();
            landmark_ids.sort_unstable();
            for landmark_id in landmark_ids {
                if let Some(descriptor) = map
                    .landmarks
                    .get(&landmark_id)
                    .and_then(|landmark| landmark.descriptor.as_ref())
                {
                    store.insert(landmark_id, descriptor.clone());
                }
            }
            store
        }
        _ => visloc_core::types::LandmarkDescriptorStore::from_visual_map(map),
    }
}

fn relocalization_result_has_better_score(
    candidate: &visloc_core::types::LocalizationResult,
    incumbent: &visloc_core::types::LocalizationResult,
) -> bool {
    candidate.inlier_count > incumbent.inlier_count
        || (candidate.inlier_count == incumbent.inlier_count
            && match (candidate.reprojection_error, incumbent.reprojection_error) {
                (Some(candidate_error), Some(incumbent_error)) => candidate_error < incumbent_error,
                (Some(_), None) => true,
                _ => false,
            })
}

/// Which geometric verifier [`OnlineSlamLoopClosureRefinementConfig`] runs
/// on loop-closure candidates every frame.
///
/// - `EssentialMatrix` (the default) mirrors the original online-refinement
///   behaviour byte-for-byte: candidates are verified on 2D-2D
///   correspondences via [`EssentialMatrixLoopClosureVerifier`] using
///   `verifier_config`. Two-view essential-matrix geometry leaves
///   translation scale unobservable, so every accepted constraint's
///   translation is pinned to
///   `verifier_config.default_translation_scale` regardless of the true
///   relative translation norm — acceptable for topology-only pose-graph
///   corrections, but it actively corrupts the trajectory's metric scale
///   when the true baseline differs.
/// - `Pnp` verifies each candidate on 2D-3D correspondences instead: the
///   current frame's inlier observations of landmarks the older keyframe
///   already observed (built by
///   [`correspondences_2d3d_for_loop_candidate`], landmark positions read
///   from the live map) are handed to [`PnPLoopClosureVerifier`]. Because
///   the keyframe pose already carries the world scale, the recovered
///   relative translation is metric — no `default_translation_scale` is
///   involved. Candidates with too few 2D-3D correspondences (below the
///   configured `min_inliers`) are rejected outright; there is no fallback
///   to the essential-matrix path.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum LoopRefinementVerifier {
    #[default]
    EssentialMatrix,
    Pnp(PnPLoopClosureVerifierConfig),
}

/// Which pose-graph solver backs
/// [`OnlineSlamLoopClosureRefinementConfig`]'s periodic PGO trigger.
///
/// [`Self::Se3`] (the default) is today's rigid solve
/// ([`PoseGraph::optimize_se3_iterative`], or
/// [`PoseGraph::optimize_se3_gnc`] when `gnc` is set) — byte-identical
/// behaviour for existing callers.
///
/// [`Self::Sim3`] instead mirrors the running graph into a parallel
/// [`Sim3PoseGraph`] (one Sim3 node per keyframe, seeded at scale `1.0`
/// from its current rigid pose; one Sim3 edge per sequential/loop-closure
/// edge, likewise seeded at scale `1.0` since both the odometry chain and
/// the PnP loop-closure verifier are metric) and solves that instead. Its
/// extra per-node scale degree of freedom lets a loop closure absorb
/// **scale drift** — e.g. a learned monocular-style tracker's accumulated
/// `1.4-2.1x` scale error — that a rigid `SE(3)` graph cannot represent
/// and instead smears as rotation/translation error across the whole
/// trajectory (see `pipelines/slam/tests/online_slam.rs`'s
/// `sim3_solver_recovers_scale_drift_se3_cannot` for a measured A/B on a
/// synthetic drifted chain).
///
/// This variant is only appropriate when scale is genuinely unobservable,
/// such as a monocular map or a merge between independently scaled maps.
/// Metric stereo and RGB-D maps must use [`Self::Se3`]: following
/// ORB-SLAM2/3, their loop scale is fixed to `1` rather than re-estimated
/// from noisy per-frame depths. Allowing every pose-graph node to change
/// scale in a metric map discards the sensor's scale observation and can
/// turn depth noise into global scale drift.
///
/// Write-back and correction-propagation conventions for this path
/// (`maybe_run_loop_closure_refinement`'s `Sim3` branch): each solved
/// node's rigid pose for `map.keyframes[*].frame.pose` is
/// `Pose { rotation: R, translation: t / s }` (ORB-SLAM2's
/// `OptimizeEssentialGraph` convention — dividing by scale, not
/// multiplying, is what keeps the corrected pose's *reprojection ray*
/// invariant, since a pinhole projection is invariant to a positive
/// rescale of camera-frame depth); each corrected keyframe's world-frame
/// similarity correction for landmark propagation is
/// `Siw_new⁻¹ ∘ Siw_old` (the direct Sim(3) generalisation of the SE(3)
/// path's `T_cw_new⁻¹ ∘ T_cw_old`); the tracker's `last_successful_pose` is
/// corrected by the same correction's rotation+translation part only,
/// with the motion model's cached world velocity additionally scaled by
/// `correction.scale` (see [`visloc_tracking::Tracker::apply_similarity_pose_correction`]).
///
/// [`OnlineSlamLoopClosureRefinementConfig::gnc`] is **ignored** on this
/// path: [`Sim3PoseGraph`] has no Graduated Non-Convexity variant yet.
/// Callers that set both should treat it as a configuration smell (the
/// demo CLI warns); a wrong loop closure will not be annealed out when
/// `solver` is `Sim3`. [`OnlineSlamLoopClosureRefinementConfig::marginalization_window`]
/// is likewise ignored on this path — [`Sim3PoseGraph`] has no
/// fixed-lag marginalization yet, so the Sim3 mirror grows unbounded for
/// the lifetime of the session. PCM / the covariance gate are unaffected:
/// both screen a candidate before it enters either graph, using only the
/// always-maintained SE(3) mirror.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum LoopRefinementSolver {
    #[default]
    Se3,
    Sim3(Sim3PoseGraphConfig),
}

/// Opt-in appearance-based long-range loop-candidate source for
/// [`OnlineSlamLoopClosureRefinementConfig`]. The shared-landmark detector
/// (`detect_loop_closure_candidates`) can only ever propose candidates that
/// still reference the SAME map landmark ids the current frame's inliers
/// carry; once the tracker drifts around a loop and revisits a place, the
/// mapper has typically re-triangulated fresh landmark ids for the very same
/// walls, so shared-id candidates are inherently short-range (near-in-time
/// keyframes only). This stream instead ranks *past* keyframes by cosine
/// similarity of their cached mean local descriptor against the current
/// frame's, restricted to candidates at least `min_keyframe_id_gap`
/// keyframe-ids older — i.e. genuinely long-range, appearance-only place
/// recognition, mirroring
/// [`OnlineSlamRelocalizationAppearanceConfig`]'s retrieval machinery
/// (`relocalization_mean_descriptor` / cosine ranking) but tuned for
/// long-range loop detection rather than short-range recovery-after-loss.
///
/// Each ranked candidate keyframe is verified by building 2D-3D
/// correspondences via descriptor matching (current frame descriptors
/// against the candidate keyframe's own observed landmarks' descriptors —
/// there is no shared-id assembly to fall back to, by construction) and
/// running them through [`PnPLoopClosureVerifier`] configured by
/// `pnp_verifier`. Accepted candidates flow through the exact same
/// downstream fold as shared-landmark candidates (PCM / covariance gate /
/// PGO / GNC / write-back).
#[derive(Debug, Clone, PartialEq)]
pub struct LoopAppearanceCandidateConfig {
    /// Minimum keyframe-id gap between the newly-registered keyframe and a
    /// candidate older keyframe. This is what makes the stream long-range:
    /// set it comfortably above the span the shared-landmark detector and
    /// local mapping window already cover.
    pub min_keyframe_id_gap: u64,
    /// Maximum number of appearance-ranked candidate keyframes verified per
    /// frame. Bounds the per-frame PnP-verification cost.
    pub max_candidates_per_frame: usize,
    /// Minimum cosine similarity between the current frame's mean local
    /// descriptor and a candidate keyframe's cached mean local descriptor.
    pub min_similarity: f32,
    /// Minimum number of descriptor-bearing landmarks a candidate keyframe
    /// must observe before appearance verification runs against it. A
    /// keyframe with fewer descriptors than `pnp_verifier.min_inliers` can
    /// never pass the PnP gate anyway, so this is a cheap early reject.
    pub min_candidate_landmark_count: usize,
    /// Number of strongest covisible neighbors gathered on each side of a
    /// candidate, matching ORB-SLAM3's ten-keyframe local-region search.
    pub covisibility_max_keyframes: usize,
    /// Shared landmark floor used to define a covisibility edge.
    pub covisibility_min_shared_landmarks: usize,
    /// Number of current-region keyframes that must independently PnP-verify
    /// against the matched candidate region. Includes the current keyframe.
    pub min_covisible_keyframe_verifications: usize,
    /// Maximum translation disagreement between a neighbor's independent PnP
    /// loop pose and the primary loop pose propagated through local odometry.
    pub max_covisibility_translation_disagreement_meters: f64,
    /// Rotation counterpart of `max_covisibility_translation_disagreement_meters`.
    pub max_covisibility_rotation_disagreement_radians: f64,
    /// Number of separate current keyframes that must verify the same
    /// covisible candidate region before it may enter the graph.
    pub region_confirmation_required_keyframes: usize,
    /// Consecutive keyframes allowed to miss a pending region before reset.
    pub region_confirmation_max_misses: usize,
    /// Optional ORB-SLAM3-style post-PnP projection search radius. When set,
    /// the primary appearance PnP pose projects the candidate region into the
    /// current image, rematches only inside each landmark's pixel window, and
    /// requires a second PnP verification before covisibility confirmation.
    pub projection_search_radius_px: Option<f64>,
    /// Minimum one-to-one projection-guided correspondences required before
    /// the refined PnP solve. ORB-SLAM3 uses a 50-match pre-optimization gate;
    /// this remains configurable because visloc-rs uses SuperPoint rather than
    /// ORB descriptors and must measure its own correspondence distribution.
    pub min_projection_correspondence_count: usize,
    /// PnP RANSAC verifier thresholds. `min_inliers` defaults higher than
    /// the shared-landmark PnP path's default because a false long-range
    /// loop closure is catastrophic (it can fold the whole trajectory into
    /// a wrong basin), so this stream demands stronger evidence before
    /// admitting a candidate.
    pub pnp_verifier: PnPLoopClosureVerifierConfig,
}

impl Default for LoopAppearanceCandidateConfig {
    fn default() -> Self {
        Self {
            min_keyframe_id_gap: 150,
            max_candidates_per_frame: 3,
            min_similarity: 0.2,
            min_candidate_landmark_count: 30,
            covisibility_max_keyframes: 10,
            covisibility_min_shared_landmarks: 10,
            min_covisible_keyframe_verifications: 3,
            max_covisibility_translation_disagreement_meters: 0.5,
            max_covisibility_rotation_disagreement_radians: 0.2,
            region_confirmation_required_keyframes: 3,
            region_confirmation_max_misses: 2,
            projection_search_radius_px: None,
            min_projection_correspondence_count: 50,
            pnp_verifier: PnPLoopClosureVerifierConfig {
                min_inliers: 30,
                min_inlier_ratio: 0.5,
                max_mean_reprojection_error_px: 4.0,
            },
        }
    }
}

/// Which stream a folded [`LoopClosureConstraint`] (reported on
/// [`OnlineSlamLoopClosureRefinementStats::admitted_constraints`]) came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopClosureCandidateSource {
    /// From the default shared-landmark detector
    /// (`detect_loop_closure_candidates`), verified by
    /// [`OnlineSlamLoopClosureRefinementConfig::verifier`].
    SharedLandmark,
    /// From the opt-in appearance-based long-range candidate source
    /// (`OnlineSlamLoopClosureRefinementConfig::appearance_candidates`).
    Appearance,
}

/// One loop-closure constraint admitted into the running pose graph this
/// frame, with enough detail for external diagnostics (e.g. a per-frame CSV
/// log distinguishing shared-landmark from appearance-sourced closures).
/// Reported on
/// [`OnlineSlamLoopClosureRefinementStats::admitted_constraints`]; does NOT
/// include constraints admitted via the `pcm_batch_rescreen` promotion path
/// (those remain counted only in
/// [`OnlineSlamLoopClosureRefinementStats::loop_closures_pcm_promoted`]).
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineSlamAdmittedLoopConstraint {
    pub from_keyframe_id: u64,
    pub to_keyframe_id: u64,
    pub inlier_count: usize,
    pub translation_norm_m: f64,
    pub relative_pose: SE3,
    pub source: LoopClosureCandidateSource,
}

/// Front-end gate that rejected an otherwise geometrically verified loop
/// constraint before it entered the pose graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnlineSlamLoopConstraintRejectionReason {
    Pcm,
    Covariance,
    /// The PnP inliers did not yield a full-rank, sufficiently conditioned
    /// covariance-aware 6-DoF pose information matrix.
    PoseInformation,
}

/// A verified loop measurement rejected before graph insertion. Keeping the
/// measurement is essential for deciding offline whether a robust gate caught
/// a false closure or over-rejected a ground-truth-consistent one.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineSlamRejectedLoopConstraint {
    pub from_keyframe_id: u64,
    pub to_keyframe_id: u64,
    pub inlier_count: usize,
    pub translation_norm_m: f64,
    pub relative_pose: SE3,
    pub source: LoopClosureCandidateSource,
    pub reason: OnlineSlamLoopConstraintRejectionReason,
}

/// Per-session configuration for the online loop-closure + pose-graph
/// refinement stage owned by [`OnlineSlamPipeline`]. When attached via
/// [`OnlineSlamConfig::pose_graph_refinement`], the pipeline maintains a
/// running [`PoseGraph`] mirror of `map.keyframes`, runs the configured
/// [`LoopRefinementVerifier`] on every candidate emitted by
/// `detect_loop_closure_candidates`, and folds verified
/// [`LoopClosureConstraint`]s into the graph. When
/// `trigger_every_new_constraints` new verified edges have accumulated
/// since the last solve, [`PoseGraph::optimize_se3_iterative`] runs and
/// the optimised keyframe poses are written back into the map.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineSlamLoopClosureRefinementConfig {
    /// Camera intrinsics passed to the loop-closure verifier when it
    /// builds correspondences. Must match the camera that produced the
    /// keyframes' keypoints. Single-monocular for now — per-frame
    /// intrinsics are out of scope for the first version of this stage.
    pub camera: Camera,
    /// Thresholds (`min_inliers`, `min_inlier_ratio`,
    /// `max_mean_sampson_error`, `default_translation_scale`) handed to
    /// the [`EssentialMatrixLoopClosureVerifier`] every frame. Only
    /// consulted when `verifier` is [`LoopRefinementVerifier::EssentialMatrix`]
    /// (the default).
    pub verifier_config: LoopClosureVerifierConfig,
    /// Which verifier backend runs on candidates this frame. Defaults to
    /// [`LoopRefinementVerifier::EssentialMatrix`] so existing callers keep
    /// today's scale-free behaviour byte-identical; opt into
    /// [`LoopRefinementVerifier::Pnp`] for metric loop-closure translations.
    pub verifier: LoopRefinementVerifier,
    /// SE(3) Gauss-Newton settings consumed by
    /// [`PoseGraph::optimize_se3_iterative`] (or, when `gnc` is `Some`, the
    /// shared SE(3) settings consumed by [`PoseGraph::optimize_se3_gnc`])
    /// when the trigger fires.
    pub pose_graph_config: PoseGraphSe3Config,
    /// Override every admitted loop edge's isotropic scalar weight. `None`
    /// preserves the legacy verifier-inlier-count weight. `Some(1.0)` gives a
    /// loop the same scalar information as a sequential edge and is the
    /// conservative control when no calibrated PnP covariance is available.
    /// The same value is used by the SE(3) mirror and by directly admitted
    /// Sim(3) edges. As documented on `pcm_batch_rescreen`, batch reconciliation
    /// with the Sim(3) solver remains unsupported.
    pub fixed_loop_edge_weight: Option<f64>,
    /// Optional covariance-aware anisotropic information for PnP loop edges.
    /// The estimator uses only the verifier's final inlier correspondences,
    /// propagates each map landmark's multi-view covariance into the query
    /// reprojection residual, and rejects rank-deficient/ill-conditioned
    /// geometry instead of hiding it with a diagonal ridge. The identical
    /// estimator and spectral cap are applied to sequential PnP edges so loop
    /// and odometry factors share one information convention. `None` preserves
    /// the scalar edge path. When set, this takes precedence over
    /// `fixed_loop_edge_weight` for SE(3) loop edges.
    pub loop_pose_information: Option<LoopPoseInformationConfig>,
    /// Optional Graduated Non-Convexity outlier rejection for the back-end
    /// solve. `None` (default) runs the plain
    /// [`PoseGraph::optimize_se3_iterative`] M-estimator. `Some(gnc)` runs
    /// [`PoseGraph::optimize_se3_gnc`] instead, so a *verified-but-wrong*
    /// loop closure — a perceptual-aliasing match that still passed
    /// essential-matrix verification — is annealed down to a vanishing
    /// weight at the back-end before it can drag the whole trajectory into a
    /// corrupted basin. This is the last untouched GNC integration point;
    /// the local M-estimator alone cannot escape a basin a confident wrong
    /// closure captures. The rejected closures are reported on
    /// [`OnlineSlamLoopClosureRefinementStats`]. Use `auto_scale` so the
    /// inlier band tracks the live graph's noise; do NOT use
    /// `auto_scale_readapt` here (it is a BA-only win and over-rejects real
    /// edges on pose graphs).
    pub gnc: Option<gnc::GncConfig>,
    /// Optional Pairwise Consistency Maximization *front-end* screen
    /// ([`crate::pcm`]). `None` (default) admits every verified loop closure;
    /// `Some(cfg)` screens each newly-verified closure for geometric
    /// consistency with the already-admitted set BEFORE it enters the graph,
    /// so a perceptual-aliasing false positive never corrupts the
    /// (non-robust) initializer or the solve. A closure is admitted when it is
    /// individually consistent with the odometry (current graph poses) and
    /// pairwise-consistent with a strict majority of the established
    /// closures. This is complementary to `gnc`: PCM removes the gross
    /// outliers up front, GNC catches the borderline ones the residual gate
    /// admits. Rejections are reported on
    /// [`OnlineSlamLoopClosureRefinementStats::loop_closures_pcm_rejected`].
    pub pcm: Option<pcm::PcmConfig>,
    /// Optional covariance-based *metric* gate (a χ² threshold such as
    /// [`covariance::CHI2_95_6DOF`]). `None` (default) applies no metric gate;
    /// `Some(threshold)` admits a verified closure only when the squared
    /// Mahalanobis distance of its innovation — the measured relative pose
    /// versus the estimate's prediction — under the relative-pose covariance
    /// ([`PoseGraph::relative_pose_covariance`]) is `<= threshold`. This is the
    /// metric counterpart to the combinatorial `pcm` screen: it rejects a
    /// confident-but-wrong closure whose implied correction is statistically
    /// implausible given the trajectory's uncertainty. Recovering the relative
    /// covariance solves two block-columns of `Σ` against the sparse block
    /// factor (`O(nnz(L))` per loop, no dense inverse), cheap enough for the
    /// online path. Rejections are reported on
    /// [`OnlineSlamLoopClosureRefinementStats::loop_closures_covariance_rejected`].
    pub covariance_gate: Option<f64>,
    /// When `pcm` is set, also run a *batch* re-screen each frame that admits
    /// loop closures: over the full history of essential-verified closures (the
    /// ones in the graph plus the ones the incremental screen deferred),
    /// recompute the maximum mutually-consistent set
    /// ([`pcm::maximum_consistent_set`], order-independent) and reconcile the
    /// graph's loop edges to it — *promoting* a closure the incremental,
    /// order-dependent admission wrongly deferred and *evicting* one it wrongly
    /// admitted. This self-heals the cold-start failure of the purely
    /// incremental screen: a perceptual-aliasing closure admitted first (against
    /// the empty set, so unconditionally) then poisons every genuine closure
    /// checked against it. `false` (default) keeps the purely incremental
    /// behavior; ignored when `pcm` is `None`. Promotions/evictions are reported
    /// on [`OnlineSlamLoopClosureRefinementStats`] and bump the trigger counter
    /// so a reconciled graph is re-solved.
    ///
    /// Divergence when `solver` is [`LoopRefinementSolver::Sim3`]:
    /// reconciliation only edits the `Se3` mirror (`graph.edges`); the
    /// `Sim3` mirror is not reconciled to match (promoted/evicted loop
    /// edges are not added to / removed from it), so combining both is
    /// currently unsupported. Not exercised by tonight's Sim3 work.
    pub pcm_batch_rescreen: bool,
    /// Optional fixed-lag / sliding-window bound on the pose graph. `None`
    /// (default) keeps the full graph and re-solves it batch every trigger.
    /// `Some(w)` runs [`PoseGraph::marginalize_oldest`] after each solve to keep
    /// at most `w` poses (the anchor among them), marginalizing the oldest into a
    /// dense Gaussian prior so the per-solve cost stays bounded as keyframes
    /// accumulate — the marginalized keyframes' optimised poses are frozen in the
    /// map at marginalization time, and loop closures to them are no longer
    /// admitted (their node has left the graph). Choose `w` ≥ a few so the
    /// sequential chain and recent loop closures stay in the window;
    /// marginalization runs at the just-solved (converged) estimate, so it is
    /// first-order exact. Marginalized pose ids are reported on
    /// [`OnlineSlamLoopClosureRefinementStats::poses_marginalized`].
    pub marginalization_window: Option<usize>,
    /// When `marginalization_window` is set, **sparsify** each marginalized
    /// blanket prior to its Chow-Liu tree
    /// ([`PoseGraph::marginalize_oldest_sparsified`]) instead of keeping the dense
    /// clique. Default `false` (dense, bit-identical to before). Set it so a
    /// long-running window does not accumulate dense priors as it slides —
    /// trading an exact marginal for a sparse, KL-optimal tree approximation that
    /// preserves every kept pose's marginal.
    pub marginalization_sparsify: bool,
    /// Minimum number of *new* verified loop-closure constraints that
    /// must accumulate before a fresh pose-graph solve runs. Clamped to
    /// at least `1`; `1` runs PGO on every accepted loop edge, higher
    /// values batch.
    pub trigger_every_new_constraints: usize,
    /// Optional appearance-based long-range loop-candidate source. `None`
    /// (the default) keeps the stage's only candidate source as
    /// `detect_loop_closure_candidates`'s shared-landmark detector — byte-
    /// identical to the stage's original behaviour. `Some(config)` also
    /// ranks past keyframes by appearance (independent of shared landmark
    /// ids) and PnP-verifies the top candidates every frame; see
    /// [`LoopAppearanceCandidateConfig`] for the retrieval + verification
    /// details. When set, the stage maintains a per-keyframe mean-
    /// descriptor cache (`OnlineSlamLoopClosureRefinementState::appearance_descriptor_cache`)
    /// as keyframes register, independent of whether
    /// [`OnlineSlamConfig::relocalization`] is configured.
    pub appearance_candidates: Option<LoopAppearanceCandidateConfig>,
    /// Persist accepted appearance-PnP inliers as cross-loop map
    /// observations. This is the `SearchAndFuse`/welding data-association step
    /// used by ORB-SLAM before reprojection BA: without it the loop exists only
    /// as a pose-pose edge and BA cannot see the long-range visual evidence.
    /// When a query keypoint already observes a different (duplicated) local
    /// landmark, that one frame/keypoint relation is reassigned to the older
    /// loop landmark on both mirrored observation indices; the duplicate
    /// landmark and its other observations are retained for a later explicit
    /// merge. `false` preserves the historical pose-edge-only behavior.
    pub fuse_loop_observations: bool,
    /// Optional synchronous covisibility "welding" BA after a PGO solve that
    /// fused at least one cross-loop observation. The solve runs on a cloned
    /// map and commits only when reprojection error does not increase and the
    /// resulting map validates. `None` preserves pose-graph-only behavior.
    pub loop_welding_ba: Option<CovisibilityLocalBaConfig>,
    /// When `true`, propagate each solved keyframe's rigid pose
    /// correction `C_k = T_cw_new⁻¹ ∘ T_cw_old` (world-frame, computed from
    /// the keyframe's pose immediately before this solve versus the
    /// solved [`PoseGraph`] pose) to (a) every landmark whose *anchor*
    /// keyframe — the first (lowest-id) keyframe that observed it — is
    /// among the keyframes this solve updated, moving the landmark
    /// rigidly with its anchor so the map stays internally consistent
    /// after PGO write-back, and (b) the tracker's continuation state
    /// (`last_successful_pose` plus motion-model state such as
    /// [`visloc_tracking::ImuPredictiveMotionModel`]'s `velocity_world`)
    /// via [`Tracker::apply_pose_correction`], using the correction of
    /// the highest-id (most-recently registered) solved keyframe so the
    /// very next `track_frame`'s PnP prior is consistent with the
    /// corrected map instead of re-anchoring to the pre-solve drift.
    /// `false` (default) keeps today's behaviour byte-identical: only
    /// `map.keyframes[*].frame.pose` moves; landmarks and the tracker's
    /// state are untouched, which is the write-back gap that leaves the
    /// tracker immediately re-localizing against the stale landmark
    /// field. Per-solve propagation diagnostics land on
    /// [`OnlineSlamLoopClosureRefinementStats::landmarks_moved`],
    /// `max_landmark_displacement_meters`,
    /// `mean_landmark_displacement_meters`, and
    /// `tracker_correction_applied`.
    pub propagate_corrections: bool,
    /// Which pose-graph solver backs the periodic PGO trigger. Defaults
    /// to [`LoopRefinementSolver::Se3`] — byte-identical to this stage's
    /// original behaviour. See [`LoopRefinementSolver`] for the `Sim3`
    /// opt-in (scale-drift correction) and its write-back /
    /// correction-propagation conventions.
    pub solver: LoopRefinementSolver,
}

/// Running state for the online loop-closure + pose-graph refinement
/// stage. Lives on [`OnlineSlamPipeline`] when
/// [`OnlineSlamConfig::pose_graph_refinement`] is `Some`.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineSlamLoopClosureRefinementState {
    pub config: OnlineSlamLoopClosureRefinementConfig,
    /// Running pose-graph mirror. One node per registered keyframe, one
    /// sequential edge between every consecutive pair of registered
    /// keyframes (in insertion order), and one loop-closure edge per
    /// verified [`LoopClosureConstraint`].
    pub graph: PoseGraph,
    /// `Sim(3)` mirror of `graph`, maintained in lockstep (same node ids,
    /// same edges, seeded at scale `1.0`) exactly when
    /// `config.solver` is [`LoopRefinementSolver::Sim3`]; `None`
    /// otherwise, so the [`LoopRefinementSolver::Se3`] default pays no
    /// extra bookkeeping. Solved by [`Sim3PoseGraph::optimize`] on the
    /// same trigger cadence as `graph`; see [`LoopRefinementSolver::Sim3`]
    /// for the write-back convention.
    pub sim3_graph: Option<Sim3PoseGraph>,
    /// Keyframe ids in the order they were first registered. The first
    /// entry is the [`PoseGraph::anchor`]; subsequent entries form the
    /// sequential edge chain.
    pub keyframe_order: Vec<u64>,
    /// Essential-verified loop closures currently admitted into the graph (one
    /// loop-closure edge each). With the incremental PCM screen this is the
    /// admitted set; with `pcm_batch_rescreen` it is the latest batch
    /// maximum-consistent set.
    pub verified_constraints: Vec<LoopClosureConstraint>,
    /// Covariance-derived information retained across PCM deferral/promotion.
    /// Keyed by `(older_keyframe, query_keyframe)`.
    loop_pose_information: HashMap<(u64, u64), Matrix6<f64>>,
    /// Essential-verified loop closures the incremental PCM screen *deferred*
    /// (not in the graph), retained so the batch re-screen
    /// ([`OnlineSlamLoopClosureRefinementConfig::pcm_batch_rescreen`]) can
    /// promote one later if it joins the consensus. Empty when batch re-screen
    /// is off.
    pub pcm_deferred: Vec<LoopClosureConstraint>,
    /// New verified constraints since the last successful PGO trigger.
    /// Reset to `0` after each fired solve.
    pub pending_since_last_trigger: usize,
    /// Total number of [`PoseGraph::optimize_se3_iterative`] calls fired
    /// by the pipeline since construction (counts both converged and
    /// not-converged solves; mismatches between the two go to
    /// `OnlineSlamLoopClosureRefinementStats::pose_graph_result`).
    pub trigger_count: u64,
    /// Cached per-keyframe mean local descriptor for the appearance-based
    /// long-range loop candidate source (see
    /// [`OnlineSlamLoopClosureRefinementConfig::appearance_candidates`]).
    /// Populated as each keyframe registers with the running graph whenever
    /// `config.appearance_candidates` is `Some`; maintained independently of
    /// [`OnlineSlamConfig::relocalization`]. Empty (and never consulted)
    /// when `appearance_candidates` is `None`.
    pub appearance_descriptor_cache: HashMap<u64, Vec<f32>>,
    appearance_pending_region: Option<AppearancePendingRegion>,
    pending_loop_observation_fusions: Vec<PendingLoopObservationFusion>,
}

#[derive(Debug, Clone, PartialEq)]
struct PendingLoopObservationFusion {
    from_keyframe_id: u64,
    query_keyframe_id: u64,
    relative_pose: SE3,
    pairs: Vec<(usize, u64)>,
}

#[derive(Debug, Clone, PartialEq)]
struct AppearancePendingRegion {
    root_keyframe_id: u64,
    last_current_keyframe_id: u64,
    root_to_last_current: SE3,
    confirmation_count: usize,
    miss_count: usize,
}

impl OnlineSlamLoopClosureRefinementState {
    fn new(config: OnlineSlamLoopClosureRefinementConfig) -> Self {
        let sim3_graph =
            matches!(config.solver, LoopRefinementSolver::Sim3(_)).then(Sim3PoseGraph::new);
        Self {
            config,
            graph: PoseGraph::new(),
            sim3_graph,
            keyframe_order: Vec::new(),
            verified_constraints: Vec::new(),
            loop_pose_information: HashMap::new(),
            pcm_deferred: Vec::new(),
            pending_since_last_trigger: 0,
            trigger_count: 0,
            appearance_descriptor_cache: HashMap::new(),
            appearance_pending_region: None,
            pending_loop_observation_fusions: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.appearance_descriptor_cache.clear();
        self.appearance_pending_region = None;
        self.pending_loop_observation_fusions.clear();
        self.graph = PoseGraph::new();
        self.sim3_graph =
            matches!(self.config.solver, LoopRefinementSolver::Sim3(_)).then(Sim3PoseGraph::new);
        self.keyframe_order.clear();
        self.verified_constraints.clear();
        self.loop_pose_information.clear();
        self.pcm_deferred.clear();
        self.pending_since_last_trigger = 0;
        self.trigger_count = 0;
    }
}

/// Per-frame outcome of the online loop-closure + pose-graph
/// refinement stage. Exposed on
/// [`OnlineSlamResult::pose_graph_refinement`] and used by the demos to
/// audit verifier acceptance + PGO trigger cadence.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OnlineSlamLoopClosureRefinementStats {
    /// Number of [`LoopClosureCandidate`]s fed through the verifier
    /// this `process_frame` call.
    pub verified_candidate_count: usize,
    /// Number of candidates the verifier accepted this frame. Note that
    /// only candidates whose `query_frame_id` matches the keyframe
    /// registered this frame are eligible — a verified candidate for a
    /// frame that did not become a keyframe is silently dropped because
    /// the running graph has no node for it.
    pub accepted_count: usize,
    /// `Some` when [`PoseGraph::optimize_se3_iterative`] fired on this
    /// frame; `None` when no new constraint arrived, the new-constraint
    /// trigger threshold was not yet reached, or the GNC robust solver ran
    /// instead (see `gnc_result`).
    pub pose_graph_result: Option<PoseGraphSe3Result>,
    /// `Some` when the GNC robust solver ([`PoseGraph::optimize_se3_gnc`])
    /// fired this frame instead of the plain iterative one — i.e. when
    /// [`OnlineSlamLoopClosureRefinementConfig::gnc`] is set and the trigger
    /// threshold was met. Its `edge_weights` classify every graph edge as
    /// inlier/outlier; `pose_graph_result` is `None` on this path.
    pub gnc_result: Option<PoseGraphGncResult>,
    /// `Some` when [`Sim3PoseGraph::optimize`] fired on this frame — i.e.
    /// [`OnlineSlamLoopClosureRefinementConfig::solver`] is
    /// [`LoopRefinementSolver::Sim3`] and the trigger threshold was met.
    /// Mutually exclusive with `pose_graph_result` / `gnc_result` (always
    /// `None` on the `Se3` path).
    pub sim3_pose_graph_result: Option<Sim3PoseGraphResult>,
    /// `(min, max)` per-node scale across the just-solved `Sim3PoseGraph`
    /// when `sim3_pose_graph_result` is `Some` — observability for how
    /// much scale drift the solve absorbed (both `1.0` when the graph is
    /// scale-consistent; a wide spread means the solve is actively
    /// redistributing accumulated monocular-style scale error). `None`
    /// on the `Se3` path or when no `Sim3` solve fired this frame.
    pub sim3_scale_spread: Option<(f64, f64)>,
    /// Number of *loop-closure* edges the GNC solver drove below the inlier
    /// threshold (weight `< 0.5`) on the solve fired this frame — the
    /// verified-but-wrong closures caught and rejected at the back-end.
    /// Always `0` on the plain iterative path (`gnc` unset).
    pub loop_closures_rejected: usize,
    /// Number of verified loop closures the PCM front-end screen rejected this
    /// frame *before* they entered the graph (geometrically inconsistent with
    /// the established set). Always `0` when `pcm` is unset.
    pub loop_closures_pcm_rejected: usize,
    /// Number of verified loop closures the covariance gate rejected this frame
    /// *before* they entered the graph (innovation statistically implausible
    /// under the relative-pose covariance). Always `0` when `covariance_gate`
    /// is unset.
    pub loop_closures_covariance_rejected: usize,
    /// Number of verified PnP loops rejected because their covariance-aware
    /// 6×6 information was unavailable, rank-deficient, or ill-conditioned.
    pub loop_closures_pose_information_rejected: usize,
    /// Number of admitted SE(3) loop edges carrying covariance-aware 6×6
    /// information rather than an isotropic scalar.
    pub loop_closures_with_pose_information: usize,
    /// Per-loop numerical evidence for covariance-aware matrices that passed
    /// the rank and condition gates, before PCM/covariance admission screens.
    pub loop_pose_information_diagnostics: Vec<LoopPoseInformationDiagnostic>,
    /// Typed reasons for loop information failures, preserving whether the
    /// problem was configuration, missing state, correspondence support, or
    /// pose geometry.
    pub loop_pose_information_failures: LoopPoseInformationFailureCounts,
    /// Sequential PnP edges carrying the same covariance-aware information
    /// convention as loop edges during this frame.
    pub sequential_edges_with_pose_information: usize,
    /// Sequential edges that fell back to identity information because too few
    /// usable landmark covariances survived. The chain is never dropped.
    pub sequential_pose_information_fallbacks: usize,
    pub sequential_pose_information_diagnostics: Vec<LoopPoseInformationDiagnostic>,
    /// Typed reason for each sequential edge that used identity information.
    pub sequential_pose_information_failures: LoopPoseInformationFailureCounts,
    /// Number of deferred loop closures the batch PCM re-screen *promoted* into
    /// the graph this frame (they joined the maximum-consistent consensus the
    /// incremental, order-dependent screen had wrongly excluded). Always `0`
    /// unless `pcm_batch_rescreen` is set.
    pub loop_closures_pcm_promoted: usize,
    /// Number of admitted loop closures the batch PCM re-screen *evicted* from
    /// the graph this frame (inconsistent with the larger consensus — e.g. a
    /// perceptual-aliasing closure admitted first against the empty set). Always
    /// `0` unless `pcm_batch_rescreen` is set.
    pub loop_closures_pcm_evicted: usize,
    /// Number of `map.keyframes[id].frame.pose` slots overwritten with
    /// the optimised pose after PGO. Zero unless a solve fired this frame
    /// (`pose_graph_result.is_some()` or `gnc_result.is_some()`).
    pub keyframes_updated: usize,
    /// Pose ids the fixed-lag `marginalization_window` marginalized out of the
    /// graph this frame (oldest first), folded into a dense Gaussian prior.
    /// Empty unless `marginalization_window` is set and the graph exceeded it.
    pub poses_marginalized: Vec<u64>,
    /// Number of appearance-ranked candidate keyframes evaluated this frame
    /// (post ranking / gap filter, pre PnP verification). Always `0` when
    /// [`OnlineSlamLoopClosureRefinementConfig::appearance_candidates`] is
    /// unset.
    pub appearance_candidate_count: usize,
    /// Number of keyframes surviving appearance ranking and the frame-gap
    /// filter before descriptor matching / PnP verification.
    pub appearance_ranked_candidate_count: usize,
    /// Appearance regions rejected because they intersect the current
    /// keyframe's connected/covisible local region and are not a loop.
    pub appearance_connected_region_rejected_count: usize,
    /// Candidates passing the primary current-vs-candidate-region PnP test.
    pub appearance_pnp_verified_count: usize,
    /// Primary-PnP successes rejected by the optional projection-rematch and
    /// refined-PnP gate.
    pub appearance_projection_rejected_count: usize,
    /// Primary-PnP successes rejected by current-side covisibility checking.
    pub appearance_covisibility_rejected_count: usize,
    /// Pending-region projection verification attempts on later keyframes.
    pub appearance_pending_projection_attempted_count: usize,
    /// Pending-region projection attempts that passed refined PnP.
    pub appearance_pending_projection_verified_count: usize,
    /// Projection correspondences produced for the pending-region attempt on
    /// this keyframe (zero when not attempted or correspondence build failed).
    pub appearance_pending_projection_correspondence_count: usize,
    /// Refined-PnP inliers from the pending-region attempt on this keyframe.
    pub appearance_pending_projection_inlier_count: usize,
    /// Radius selected by the adaptive pending-region projection search.
    pub appearance_pending_projection_search_radius_px: Option<f64>,
    /// Per-primary-PnP candidate evidence for reproducible gate analysis.
    pub appearance_candidate_diagnostics: Vec<AppearanceLoopCandidateDiagnostic>,
    /// A geometrically verified region is being carried into later keyframes.
    pub appearance_region_confirmation_waiting: bool,
    /// A pending region reached its configured cross-keyframe confirmation count.
    pub appearance_region_confirmed_count: usize,
    /// Appearance candidates whose PnP verification succeeded but whose
    /// matched 3D regions could not produce a robust Sim3 scale observation.
    pub appearance_scale_estimation_failed_count: usize,
    pub appearance_scale_insufficient_points_count: usize,
    pub appearance_scale_insufficient_matches_count: usize,
    pub appearance_scale_no_consensus_count: usize,
    /// Robust scale observations rejected because they were within the
    /// Sim3 arm's five-percent no-op band around unit scale.
    pub appearance_near_unit_scale_count: usize,
    /// Number of appearance candidates admitted into the graph this frame
    /// (subset of `accepted_count`). Always `0` when
    /// `appearance_candidates` is unset.
    pub appearance_accepted_count: usize,
    /// Accepted appearance-PnP inlier pairs considered by the optional
    /// cross-loop observation fusion stage.
    pub loop_fusion_pairs_considered: usize,
    /// New query-keyframe/old-landmark observation relations inserted.
    pub loop_fusion_observations_inserted: usize,
    /// Insertions that replaced a query keypoint's relation to a different
    /// local duplicate landmark.
    pub loop_fusion_observations_reassigned: usize,
    /// Pairs withheld because final GNC classified their loop edge below the
    /// inlier threshold.
    pub loop_fusion_pairs_robust_rejected: usize,
    /// Pairs withheld because the corrected query pose failed the pixel-space
    /// reprojection gate.
    pub loop_fusion_pairs_reprojection_rejected: usize,
    /// Pairs skipped because the landmark/keypoint was missing, invalid, or
    /// the older landmark already had another observation in the query frame.
    pub loop_fusion_pairs_skipped: usize,
    pub loop_welding_ba_attempted: bool,
    pub loop_welding_ba_succeeded: bool,
    pub loop_welding_ba_mean_reprojection_before_px: Option<f64>,
    pub loop_welding_ba_mean_reprojection_after_px: Option<f64>,
    pub loop_welding_initial_translation_meters: Option<f64>,
    pub loop_welding_initial_rotation_radians: Option<f64>,
    pub loop_welding_ba_updated_keyframes: usize,
    pub loop_welding_ba_updated_landmarks: usize,
    pub loop_welding_ba_rejected_or_failed: bool,
    pub loop_welding_post_pgo_attempted: bool,
    pub loop_welding_post_pgo_succeeded: bool,
    pub loop_welding_post_pgo_mean_reprojection_px: Option<f64>,
    pub loop_welding_post_ba_error: Option<CovisibilityLocalBaError>,
    pub loop_welding_post_ba_behind_camera_ratio: Option<f64>,
    /// Loop-closure constraints admitted into the graph this frame — both
    /// the shared-landmark and (when configured) appearance streams — with
    /// enough detail for external diagnostics (e.g.
    /// `examples/euroc_online_slam_vi_image_demo.rs`'s `loop_constraints.csv`).
    /// Does NOT include constraints admitted via the `pcm_batch_rescreen`
    /// promotion path.
    pub admitted_constraints: Vec<OnlineSlamAdmittedLoopConstraint>,
    /// Verified loop measurements rejected by PCM or the covariance gate this
    /// frame. Unlike the aggregate counters, this retains the pose needed for
    /// post-hoc ground-truth correctness and over-rejection analysis.
    pub rejected_constraints: Vec<OnlineSlamRejectedLoopConstraint>,
    /// Number of landmarks moved by
    /// [`OnlineSlamLoopClosureRefinementConfig::propagate_corrections`]'s
    /// landmark-propagation pass this frame. Always `0` when
    /// `propagate_corrections` is `false`, no solve fired this frame, or
    /// the solve updated no keyframe with a previously-known pose.
    pub landmarks_moved: usize,
    /// Maximum per-landmark displacement (metres) applied by the
    /// propagation pass this frame. `None` when `landmarks_moved == 0`.
    pub max_landmark_displacement_meters: Option<f64>,
    /// Mean per-landmark displacement (metres) applied by the
    /// propagation pass this frame. `None` when `landmarks_moved == 0`.
    pub mean_landmark_displacement_meters: Option<f64>,
    /// `true` when the tracker's continuation state
    /// (`last_successful_pose` plus motion-model state) was corrected
    /// this frame via [`Tracker::apply_pose_correction`] with the
    /// most-recently solved keyframe's correction. Always `false` when
    /// `propagate_corrections` is `false` or no solve fired this frame.
    pub tracker_correction_applied: bool,
}

/// Per-session IMU integration parameters consumed by
/// [`OnlineSlamPipeline`] when [`OnlineSlamConfig::imu`] is `Some`. Gravity
/// is in the world frame (KITTI y-down: `(0, 9.81, 0)`); biases are the
/// linearisation point for the pre-integrator's first-order bias-Jacobians;
/// the three `weight_*` fields populate the corresponding
/// [`ImuPreintegrationFactor`] weights so a downstream BA can consume the
/// emitted factor without further configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineSlamImuConfig {
    pub gravity_world: Vector3<f64>,
    pub bias_gyro: Vector3<f64>,
    pub bias_acc: Vector3<f64>,
    pub weight_position: f64,
    pub weight_velocity: f64,
    pub weight_rotation: f64,
    /// Optional continuous-time measurement noise used to propagate the
    /// preintegrated `[rotation, velocity, position]` covariance. `None`
    /// preserves the legacy scalar factor weights.
    pub noise_model: Option<ImuNoiseModel>,
}

impl Default for OnlineSlamImuConfig {
    fn default() -> Self {
        Self {
            gravity_world: Vector3::new(0.0, 9.81, 0.0),
            bias_gyro: Vector3::zeros(),
            bias_acc: Vector3::zeros(),
            weight_position: 1.0,
            weight_velocity: 1.0,
            weight_rotation: 1.0,
            noise_model: None,
        }
    }
}

/// Internal IMU running state attached to an [`OnlineSlamPipeline`] when
/// [`OnlineSlamConfig::imu`] is `Some`. Carries the running pre-integrator,
/// the `keyframe_id_from` of the open integration window (`None` before the
/// first keyframe is registered), and any factor staged by `process_frame`
/// for the caller to take via [`OnlineSlamPipeline::take_pending_imu_factor`].
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineSlamImuState {
    pub config: OnlineSlamImuConfig,
    pub preintegrator: ImuPreintegrator,
    pub last_keyframe_id: Option<u64>,
    pub pending_factor: Option<ImuPreintegrationFactor>,
}

impl OnlineSlamImuState {
    fn new(config: OnlineSlamImuConfig) -> Self {
        let preintegrator = config
            .noise_model
            .and_then(|noise| {
                ImuPreintegrator::new_with_bias_and_noise(config.bias_gyro, config.bias_acc, noise)
            })
            .unwrap_or_else(|| ImuPreintegrator::new_with_bias(config.bias_gyro, config.bias_acc));
        Self {
            config,
            preintegrator,
            last_keyframe_id: None,
            pending_factor: None,
        }
    }

    fn reset(&mut self) {
        self.preintegrator = self
            .config
            .noise_model
            .and_then(|noise| {
                ImuPreintegrator::new_with_bias_and_noise(
                    self.config.bias_gyro,
                    self.config.bias_acc,
                    noise,
                )
            })
            .unwrap_or_else(|| {
                ImuPreintegrator::new_with_bias(self.config.bias_gyro, self.config.bias_acc)
            });
        self.last_keyframe_id = None;
        self.pending_factor = None;
    }

    fn reset_preintegrator_with_bias(&mut self, bias_gyro: Vector3<f64>, bias_acc: Vector3<f64>) {
        self.config.bias_gyro = bias_gyro;
        self.config.bias_acc = bias_acc;
        self.preintegrator = self
            .config
            .noise_model
            .and_then(|noise| ImuPreintegrator::new_with_bias_and_noise(bias_gyro, bias_acc, noise))
            .unwrap_or_else(|| ImuPreintegrator::new_with_bias(bias_gyro, bias_acc));
        self.pending_factor = None;
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OnlineSlamPipeline<T, M> {
    pub map: VisualMap,
    pub tracker: T,
    pub mapper: M,
    pub config: OnlineSlamConfig,
    /// Running IMU pre-integration state. `Some` exactly when
    /// `config.imu.is_some()` (initialised by [`Self::new`]); `None`
    /// otherwise so existing IMU-free flows pay no per-frame overhead.
    pub imu_state: Option<OnlineSlamImuState>,
    /// Running local VI-BA state. `Some` exactly when
    /// `config.local_vi_ba.is_some()` (initialised by [`Self::new`]);
    /// `None` otherwise so non-VI-BA flows do no extra book-keeping.
    pub local_vi_ba_state: Option<OnlineSlamLocalBaState>,
    /// Sparse visual factor lifecycle state. Present exactly when
    /// `config.sparse_factor_graph` is configured.
    pub sparse_factor_graph_state: Option<SparseFactorGraph>,
    /// Running auto-bootstrap state. `Some` exactly when
    /// `config.vi_init.is_some() && config.imu.is_some()` (initialised
    /// by [`Self::new`]); deliberately private because writes to
    /// `completed` cross-cut with `imu_state` / `local_vi_ba_state` /
    /// `map.keyframes`. Inspected via [`Self::vi_initialization_status`].
    vi_init_state: Option<OnlineSlamViInitState>,
    /// Running motion-based VI init state. `Some` exactly when
    /// `config.vi_motion_init.is_some() && config.vi_init.is_some() &&
    /// config.imu.is_some()`. Private because the motion-based fire is
    /// normally gated on the static stage having completed first; an explicit
    /// motion-start fallback may instead begin after that stage gives up. The
    /// pipeline owns the ordering. Inspected via
    /// [`Self::motion_vi_initialization_status`].
    vi_motion_init_state: Option<OnlineSlamMotionViInitState>,
    /// Running loop-closure + pose-graph refinement state. `Some`
    /// exactly when `config.pose_graph_refinement.is_some()`
    /// (initialised by [`Self::new`]); `None` otherwise so pure-
    /// odometry flows do no extra book-keeping.
    pub pose_graph_state: Option<OnlineSlamLoopClosureRefinementState>,
    /// Running relocalization-on-tracker-death state. `Some` exactly
    /// when `config.relocalization.is_some()` (initialised by
    /// [`Self::new`]). Owns its own [`LocalizationPipeline`] instance
    /// so the recovery attempt does not perturb the tracker's primary
    /// localizer state.
    pub relocalization_state: Option<OnlineSlamRelocalizationState>,
}

impl Default
    for OnlineSlamPipeline<
        Tracker<LocalizationPipeline, ConstantPoseMotionModel>,
        LocalMappingPipeline<SimpleKeyframePolicy, LinearTriangulator>,
    >
{
    fn default() -> Self {
        Self {
            map: VisualMap::new(),
            tracker: Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            mapper: LocalMappingPipeline::default(),
            config: OnlineSlamConfig::default(),
            imu_state: None,
            local_vi_ba_state: None,
            sparse_factor_graph_state: None,
            vi_init_state: None,
            vi_motion_init_state: None,
            pose_graph_state: None,
            relocalization_state: None,
        }
    }
}

impl<T, M> OnlineSlamPipeline<T, M> {
    /// Construct a pipeline. Validates cross-field configuration
    /// invariants (currently: `vi_init` requires `imu`, gravity vectors
    /// must agree) via [`OnlineSlamConfig::validate`]; panics with the
    /// surfaced [`OnlineSlamConfigError`] on a mismatch. This is a
    /// developer error — every public caller passes a literal config —
    /// so failing loudly at construction is correct.
    pub fn new(map: VisualMap, tracker: T, mapper: M, config: OnlineSlamConfig) -> Self {
        if let Err(err) = config.validate() {
            panic!("OnlineSlamPipeline::new: invalid config: {err}");
        }
        let imu_state = config.imu.clone().map(OnlineSlamImuState::new);
        let local_vi_ba_state = config.local_vi_ba.clone().map(OnlineSlamLocalBaState::new);
        let mut sparse_factor_graph_state = config
            .sparse_factor_graph
            .clone()
            .map(SparseFactorGraph::new);
        if let Some(graph) = sparse_factor_graph_state.as_mut() {
            let mut existing_keyframe_ids = map.keyframes.keys().copied().collect::<Vec<_>>();
            existing_keyframe_ids.sort_unstable();
            for keyframe_id in existing_keyframe_ids {
                graph.update_from_map(&map, keyframe_id);
            }
        }
        let vi_init_state = config.vi_init.clone().map(OnlineSlamViInitState::new);
        let vi_motion_init_state = config
            .vi_motion_init
            .clone()
            .map(OnlineSlamMotionViInitState::new);
        let pose_graph_state = config
            .pose_graph_refinement
            .clone()
            .map(OnlineSlamLoopClosureRefinementState::new);
        let relocalization_state = config
            .relocalization
            .clone()
            .map(OnlineSlamRelocalizationState::new);
        Self {
            map,
            tracker,
            mapper,
            config,
            imu_state,
            local_vi_ba_state,
            sparse_factor_graph_state,
            vi_init_state,
            vi_motion_init_state,
            pose_graph_state,
            relocalization_state,
        }
    }

    /// Read-only snapshot of the auto-bootstrap stage. Returns
    /// [`ViInitializationStatus::Disabled`] when `vi_init` is `None`;
    /// otherwise reports buffering / initialised / gave-up state.
    pub fn vi_initialization_status(&self) -> ViInitializationStatus {
        match &self.vi_init_state {
            None => ViInitializationStatus::Disabled,
            Some(state) => state.snapshot(),
        }
    }

    /// Read-only snapshot of the motion-based VI init stage. Returns
    /// [`MotionViInitializationStatus::Disabled`] when
    /// `vi_motion_init` is `None`; otherwise reports
    /// waiting / initialised state.
    pub fn motion_vi_initialization_status(&self) -> MotionViInitializationStatus {
        match &self.vi_motion_init_state {
            None => MotionViInitializationStatus::Disabled,
            Some(state) => state.snapshot(),
        }
    }

    /// Fold one body-frame IMU sample into the pipeline's running pre-
    /// integrator. No-op when [`OnlineSlamConfig::imu`] is `None`. `dt`
    /// is seconds since the previous sample; the integrator only accepts
    /// strictly positive `dt` and silently drops non-positive values so
    /// callers can replay raw IMU streams without pre-filtering.
    pub fn push_imu_measurement(&mut self, gyro: Vector3<f64>, accel: Vector3<f64>, dt: f64) {
        if dt <= 0.0 || !dt.is_finite() {
            return;
        }
        if let Some(state) = self.imu_state.as_mut() {
            state.preintegrator.integrate_sample(gyro, accel, dt);
        }
        // While the auto-bootstrap stage is still active, also fan the
        // sample into its initialiser buffer. Once `completed` /
        // `gave_up` is `Some`, stop forwarding so the standalone module
        // is not asked to keep growing memory after it has fired.
        if let Some(state) = self.vi_init_state.as_mut() {
            if state.is_active() {
                state.initializer.push_sample(gyro, accel, dt);
                state.samples_buffered += 1;
                state.buffered_duration_seconds += dt;
            }
        }
    }

    /// Feed a sample only to the stationary visual-inertial initializer,
    /// without integrating it into the first inter-keyframe IMU factor.
    /// Dataset runners use this for IMU measurements that precede the visual
    /// seed keyframe: those samples contain the best stationary bias/gravity
    /// window, but including them in the running preintegrator would attach
    /// motion from before the first graph state to its first factor.
    pub fn push_vi_initialization_measurement(
        &mut self,
        gyro: Vector3<f64>,
        accel: Vector3<f64>,
        dt: f64,
    ) {
        if dt <= 0.0 || !dt.is_finite() {
            return;
        }
        if let Some(state) = self.vi_init_state.as_mut() {
            if state.is_active() {
                state.initializer.push_sample(gyro, accel, dt);
                state.samples_buffered += 1;
                state.buffered_duration_seconds += dt;
            }
        }
    }

    /// Take the IMU pre-integration factor staged by the most recent
    /// [`Self::process_frame`] call, if any. Returns `None` when IMU is
    /// disabled, when the last `process_frame` did not register a new
    /// keyframe to close the running window against, or when the
    /// auto-bootstrap stale-factor gate is still active (factors staged
    /// before VI init success are discarded — they were built with the
    /// caller's placeholder bias linearisation and would feed an
    /// inconsistent point into downstream VI-BA).
    pub fn take_pending_imu_factor(&mut self) -> Option<ImuPreintegrationFactor> {
        self.imu_state
            .as_mut()
            .and_then(|s| s.pending_factor.take())
    }

    pub fn map(&self) -> &VisualMap {
        &self.map
    }

    pub fn map_mut(&mut self) -> &mut VisualMap {
        &mut self.map
    }

    /// Synchronize the sparse factor lifecycle after a caller commits a
    /// keyframe outside [`Self::process_frame`], such as a calibrated-stereo
    /// segment restart. Returns `None` when the graph is disabled or the
    /// keyframe is not present in the map.
    pub fn sync_sparse_factor_graph_keyframe(
        &mut self,
        keyframe_id: u64,
    ) -> Option<SparseFactorGraphUpdateStats> {
        if !self.map.keyframes.contains_key(&keyframe_id) {
            return None;
        }
        self.sparse_factor_graph_state
            .as_mut()
            .map(|graph| graph.update_from_map(&self.map, keyframe_id))
    }

    /// Run the appearance-based pairwise loop scanner over every keyframe
    /// currently in `self.map`. Returns the verifier-accepted pairs as
    /// [`LoopClosureCandidate`]s with `verification` populated and
    /// `geometrically_verified = true` — the same shape `process_frame`'s
    /// shared-landmark detector emits, so callers can mix the two streams.
    /// Unlike [`Self::process_frame`], this runs `O(K²)` over the keyframe
    /// count and is meant for periodic (every-N-frames) or end-of-session
    /// use rather than every frame. Each accepted pair has
    /// `query_frame_id` set to the later keyframe and `matched_keyframe_id`
    /// set to the earlier — same convention as `LoopClosureConstraint`.
    /// `shared_landmark_count` is filled with the verifier inlier count
    /// (no map-side landmark intersection is computed) so candidates are
    /// directly consumable by [`LoopClosureConstraint::from_verified_candidate`].
    pub fn scan_appearance_loops<Mat, V>(
        &self,
        matcher: &Mat,
        verifier: &V,
        camera: &Camera,
        settings: &AppearanceLoopScannerSettings,
    ) -> Vec<LoopClosureCandidate>
    where
        Mat: Matcher,
        V: LoopClosureVerifier,
    {
        // Sort keyframes by id for deterministic pair ordering.
        let mut kfs: Vec<(&u64, &Keyframe)> = self.map.keyframes.iter().collect();
        kfs.sort_by_key(|(id, _)| *id);
        let views: Vec<PairwiseKeyframeView> = kfs
            .iter()
            .map(|(_, kf)| PairwiseKeyframeView {
                frame_id: kf.frame.id,
                keypoints: &kf.frame.keypoints,
                descriptors: &kf.frame.descriptors,
            })
            .collect();
        scan_pairwise_loop_closures(
            &views,
            matcher,
            verifier,
            camera,
            &PairwiseLoopClosureScannerConfig {
                min_keyframe_id_gap: settings.min_keyframe_id_gap,
                min_matches: settings.min_matches,
            },
        )
    }
}

/// Tunables for [`OnlineSlamPipeline::scan_appearance_loops`]. Mirrors
/// [`PairwiseLoopClosureScannerConfig`] but with names that emphasise the
/// pipeline-level use ("how aggressively should the periodic scan run?").
/// Defaults are chosen so a typical local-mapping window doesn't get
/// confused for a loop, and so visually disjoint keyframes are rejected
/// before the verifier ever fires.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AppearanceLoopScannerSettings {
    /// Minimum frame-id gap between two keyframes for the pair to be
    /// considered. Set this larger than the local-mapping window so the
    /// scanner never proposes a "loop" between adjacent keyframes that
    /// share most of their tracks anyway.
    pub min_keyframe_id_gap: u64,
    /// Minimum number of brute-force descriptor matches a pair needs
    /// before its 2D-2D correspondences are even handed to the verifier.
    pub min_matches: usize,
}

impl Default for AppearanceLoopScannerSettings {
    fn default() -> Self {
        Self {
            min_keyframe_id_gap: 30,
            min_matches: 30,
        }
    }
}

impl<P, Motion, K, Tri> OnlineSlamPipeline<Tracker<P, Motion>, LocalMappingPipeline<K, Tri>>
where
    P: FrameLocalizer,
    Motion: MotionModel,
    K: KeyframePolicy,
    Tri: Triangulator,
{
    pub fn process_frame<I>(&mut self, frame: &Frame, candidates: I) -> OnlineSlamResult
    where
        I: IntoIterator<Item = LandmarkCandidate>,
    {
        self.process_frame_with_metric_points(frame, candidates, &HashMap::new())
    }

    /// Process a frame while exposing independently triangulated metric 3D
    /// points in the current camera frame, keyed by query keypoint index.
    /// The lookup is consumed only by appearance-loop Sim3 verification;
    /// tracking and mapping retain the existing `process_frame` behaviour.
    pub fn process_frame_with_metric_points<I>(
        &mut self,
        frame: &Frame,
        candidates: I,
        metric_points_camera: &HashMap<usize, Point3<f64>>,
    ) -> OnlineSlamResult
    where
        I: IntoIterator<Item = LandmarkCandidate>,
    {
        let mut tracking = self.tracker.track_frame(frame, &self.map);
        // Relocalization-on-tracker-death: if the primary attempt
        // failed and the stage is enabled, run a fresh PnP against the
        // full map via the pipeline's owned `LocalizationPipeline`. On
        // acceptance, override the tracker's history with the recovered
        // result so the rest of `process_frame` (loop detection,
        // mapper, IMU staging, etc.) sees a successful frame.
        let relocalization_stats = self.maybe_run_relocalization(frame, &mut tracking);
        let mut mapping = None;
        let mut applied_update = None;
        let mut loop_closure_candidates =
            detect_loop_closure_candidates(frame, &tracking, &self.map, &self.config.loop_closure);

        if tracking.localization.success {
            let keyframe = keyframe_from_tracking_result(frame, &tracking);
            let mapping_result = self
                .mapper
                .process_keyframe(&self.map, &tracking, keyframe, candidates);
            if self.config.apply_map_updates && mapping_result.staged_update_validation.is_valid() {
                if let Ok(applied) = mapping_result.staged_update.clone().apply_to(&mut self.map) {
                    if applied.keyframe_count > 0 {
                        record_tracking_observation_confidences(&mut self.map, frame.id, &tracking);
                    }
                    applied_update = Some(applied);
                }
            }
            mapping = Some(mapping_result);
        }

        // When IMU is configured and the mapper registered a new keyframe
        // at this frame, snapshot the running pre-integration window into
        // an `ImuPreintegrationFactor` connecting the previous keyframe to
        // this one, then reset the integrator so the next window starts
        // fresh. The factor is exposed both inline on the result and via
        // `take_pending_imu_factor()` so callers can pick whichever shape
        // fits their pose-graph / BA glue.
        let imu_factor = self.stage_imu_factor_on_new_keyframe(frame, applied_update.as_ref());
        // Run the VI init step AFTER `stage_imu_factor_on_new_keyframe`
        // so that on the success frame the just-promoted keyframe id is
        // already in `imu_state.last_keyframe_id` and the staged factor
        // (if any) has already been discarded by the stale-factor gate.
        // The step is a no-op when VI init is disabled or terminal.
        let vi_init = self.run_vi_init_step(frame, applied_update.as_ref());
        let just_promoted_vi_init =
            matches!(vi_init, Some(ViInitializationEvent::Succeeded { .. }));
        // Motion-based initialization must run before local VI-BA. In the
        // moving-start fallback, static init has terminally given up but the
        // navigation state is still uninitialized; running BA in that gap
        // would optimize with placeholder zero velocity/bias and corrupt the
        // visual map before VIBA1 gets a chance to recover them.
        let vi_motion_init =
            self.run_motion_vi_init_step(frame, applied_update.as_ref(), imu_factor.as_ref());
        let mut local_vi_ba = self.maybe_run_local_vi_ba(imu_factor.clone());
        // Phase-16: when VI-init promoted *this frame* and the config
        // opted in, force a sliding-window BA pass on the banked
        // factors. Bypasses the new-factor gate inside
        // `maybe_run_local_vi_ba` so the post-promotion refinement
        // doesn't have to wait for the next keyframe registration —
        // critical when the visual tracker is fragile and the next KF
        // arrives late or never.
        let run_at_promotion = self
            .config
            .local_vi_ba
            .as_ref()
            .map(|c| c.run_at_vi_init_promotion)
            .unwrap_or(false);
        if just_promoted_vi_init && run_at_promotion && local_vi_ba.is_none() {
            if let Some(state) = self.local_vi_ba_state.as_mut() {
                local_vi_ba = crate::online_slam_vi_ba::run_local_vi_ba(&mut self.map, state);
            }
        }
        let sparse_factor_graph =
            self.maybe_update_sparse_factor_graph(frame.id, applied_update.as_ref());
        // Visual-only covisibility local BA runs after the per-keyframe
        // visual/VI stages so the map reflects the just-finalised active
        // keyframe, and before pose-graph refinement so PGO mirrors the
        // post-local-BA pose.
        let covisibility_local_ba =
            self.maybe_run_covisibility_local_ba(frame, applied_update.as_ref());

        // Online loop-closure + pose-graph refinement runs LAST so the
        // graph mirrors the just-finalised keyframe pose (post local-VI-
        // BA) before PGO write-back. No-op when the stage is disabled or
        // no keyframe was registered this frame.
        let pose_graph_refinement = self.maybe_run_loop_closure_refinement(
            frame,
            &tracking,
            applied_update.as_ref(),
            &mut loop_closure_candidates,
            metric_points_camera,
        );

        OnlineSlamResult {
            tracking,
            mapping,
            applied_update,
            loop_closure_candidates,
            imu_factor,
            local_vi_ba,
            sparse_factor_graph,
            map_keyframe_count: self.map.keyframes.len(),
            map_landmark_count: self.map.landmarks.len(),
            vi_init,
            vi_motion_init,
            covisibility_local_ba,
            pose_graph_refinement,
            relocalization: relocalization_stats,
        }
    }

    fn maybe_update_sparse_factor_graph(
        &mut self,
        frame_id: u64,
        applied_update: Option<&AppliedMapUpdate>,
    ) -> Option<SparseFactorGraphUpdateStats> {
        applied_update.filter(|update| update.keyframe_count > 0)?;
        self.sync_sparse_factor_graph_keyframe(frame_id)
    }

    /// Relocalization-on-tracker-death. Runs only when (a) the
    /// `relocalization` stage is configured AND (b) the primary
    /// `track_frame` call returned `localization.success == false`.
    /// Re-runs PnP against the full visual map via the stage's owned
    /// [`LocalizationPipeline`], gates the recovered solution against
    /// the configured thresholds, and on acceptance overwrites the
    /// tracker's per-frame history via
    /// [`Tracker::accept_relocalization_result`] so the rest of
    /// `process_frame` sees a successful frame. Returns
    /// `Some(stats)` whenever the stage actually ran; `None` when
    /// disabled OR primary tracking already succeeded.
    fn maybe_run_relocalization(
        &mut self,
        frame: &Frame,
        tracking: &mut TrackingResult,
    ) -> Option<OnlineSlamRelocalizationStats> {
        let state = self.relocalization_state.as_mut()?;
        if tracking.localization.success {
            state.consecutive_failed_attempts = 0;
            state.pending_confirmation = None;
            return None;
        }
        if state
            .config
            .max_consecutive_failed_attempts
            .is_some_and(|max| state.consecutive_failed_attempts >= max)
        {
            state.budget_skip_count = state.budget_skip_count.saturating_add(1);
            return None;
        }
        let attempt_interval_frames = state.config.attempt_interval_frames.max(1);
        if let Some(last_attempt_frame_id) = state.last_attempt_frame_id {
            if frame.id.saturating_sub(last_attempt_frame_id) < attempt_interval_frames {
                return None;
            }
        }
        state.trigger_count += 1;
        state.last_attempt_frame_id = Some(frame.id);
        let (recovered, mut stats, accept, tried_broader_descriptor_store) = {
            let pose_prior = if state.config.pose_prior_candidate_radius_meters.is_some() {
                self.tracker.pose_prior_for_frame(frame)
            } else {
                None
            };
            let candidate_radius = state.config.pose_prior_candidate_radius_meters;
            let localize_with_store =
                |descriptor_store: &visloc_core::types::LandmarkDescriptorStore| {
                    if pose_prior.is_some() {
                        state
                            .localizer
                            .localize_frame_with_pose_prior_warm_start_and_descriptor_store(
                                frame,
                                &self.map,
                                descriptor_store,
                                pose_prior.as_ref(),
                                candidate_radius,
                            )
                    } else {
                        state.localizer.localize_frame_with_descriptor_store(
                            frame,
                            &self.map,
                            descriptor_store,
                        )
                    }
                };
            let build_attempt_stats =
                |recovered: &visloc_core::types::LocalizationResult,
                 descriptor_store_landmark_count: usize,
                 covisibility_local_descriptor_store_landmark_count: Option<usize>,
                 appearance_descriptor_store_landmark_count: Option<usize>,
                 broader_descriptor_store_landmark_count: Option<usize>,
                 tried_covisibility_local_descriptor_store: bool,
                 used_covisibility_local_descriptor_store: bool,
                 tried_appearance_descriptor_store: bool,
                 used_appearance_descriptor_store: bool,
                 tried_broader_descriptor_store_fallback: bool,
                 broader_descriptor_store_retry_skipped_by_interval: bool,
                 used_broader_descriptor_store_fallback: bool,
                 covisibility_reference_keyframe_id: Option<u64>,
                 appearance_candidate_keyframe_count: usize,
                 appearance_best_similarity: Option<f32>,
                 appearance_best_keyframe_id: Option<u64>,
                 appearance_candidates: Vec<OnlineSlamRelocalizationAppearanceCandidate>| {
                    let mut stats = OnlineSlamRelocalizationStats {
                        attempted: true,
                        localization_success: recovered.success,
                        succeeded: false,
                        inlier_count: recovered.inlier_count,
                        inlier_ratio: recovered.inlier_ratio,
                        correspondence_count: recovered.correspondence_count,
                        mean_reprojection_error: recovered.reprojection_error,
                        translation_from_last_success_meters: None,
                        translation_per_frame_from_last_success_meters: None,
                        inlier_depth_median_meters: None,
                        last_success_inlier_depth_median_meters: None,
                        inlier_depth_median_ratio_to_last_success: None,
                        passed_acceptance_gates: false,
                        confirmation_count: 0,
                        confirmation_required_count: state
                            .config
                            .confirmation_required_recoveries
                            .max(1),
                        confirmation_translation_per_frame_from_previous_meters: None,
                        descriptor_store_landmark_count,
                        covisibility_local_descriptor_store_landmark_count,
                        appearance_descriptor_store_landmark_count,
                        broader_descriptor_store_landmark_count,
                        tried_covisibility_local_descriptor_store,
                        used_covisibility_local_descriptor_store,
                        tried_appearance_descriptor_store,
                        used_appearance_descriptor_store,
                        tried_broader_descriptor_store_fallback,
                        broader_descriptor_store_retry_skipped_by_interval,
                        used_broader_descriptor_store_fallback,
                        covisibility_reference_keyframe_id,
                        appearance_candidate_keyframe_count,
                        appearance_best_similarity,
                        appearance_best_keyframe_id,
                        appearance_candidates,
                    };
                    if let (Some(last_pose), Some(recovered_pose)) =
                        (self.tracker.last_successful_pose(), recovered.pose.as_ref())
                    {
                        let translation = (recovered_pose.camera_center_world()
                            - last_pose.camera_center_world())
                        .norm();
                        stats.translation_from_last_success_meters = Some(translation);
                        if let Some(last_frame_id) = self.tracker.last_successful_frame_id() {
                            let frame_gap = frame.id.saturating_sub(last_frame_id).max(1) as f64;
                            stats.translation_per_frame_from_last_success_meters =
                                Some(translation / frame_gap);
                        }
                        stats.inlier_depth_median_meters = median_positive_depth_for_landmarks(
                            &self.map,
                            recovered_pose,
                            &recovered.inlier_landmark_ids,
                        );
                        stats.last_success_inlier_depth_median_meters =
                            median_positive_depth_for_landmarks(
                                &self.map,
                                last_pose,
                                &recovered.inlier_landmark_ids,
                            );
                        if let (Some(recovered_depth), Some(last_depth)) = (
                            stats.inlier_depth_median_meters,
                            stats.last_success_inlier_depth_median_meters,
                        ) {
                            if last_depth > 0.0 {
                                stats.inlier_depth_median_ratio_to_last_success =
                                    Some(recovered_depth / last_depth);
                            }
                        }
                    }
                    let basic_accept = recovered.success
                        && recovered.inlier_count >= state.config.min_inliers
                        && recovered.inlier_ratio >= state.config.min_inlier_ratio
                        && match (
                            state.config.max_mean_reprojection_error,
                            recovered.reprojection_error,
                        ) {
                            (Some(max), Some(actual)) => actual <= max,
                            (Some(_), None) => false,
                            (None, _) => true,
                        };
                    let imu_accept = if let Some(max_dist) =
                        state.config.max_translation_from_imu_prediction_meters
                    {
                        match (
                            self.tracker.pose_prior_for_frame(frame),
                            recovered.pose.as_ref(),
                        ) {
                            (Some(predicted), Some(recovered_pose)) => {
                                let predicted_centre =
                                    predicted.world_to_camera.inverse().translation;
                                let recovered_centre =
                                    recovered_pose.world_to_camera.inverse().translation;
                                (recovered_centre - predicted_centre).norm() <= max_dist
                            }
                            _ => true,
                        }
                    } else {
                        true
                    };
                    let continuity_accept = match (
                        state
                            .config
                            .max_translation_per_frame_from_last_success_meters,
                        stats.translation_per_frame_from_last_success_meters,
                    ) {
                        (Some(max), Some(actual)) => actual <= max,
                        (Some(_), None) => true,
                        (None, _) => true,
                    };
                    let depth_ratio_accept = match stats.inlier_depth_median_ratio_to_last_success {
                        Some(ratio) => {
                            state
                                .config
                                .min_inlier_depth_median_ratio_to_last_success
                                .is_none_or(|min| ratio >= min)
                                && state
                                    .config
                                    .max_inlier_depth_median_ratio_to_last_success
                                    .is_none_or(|max| ratio <= max)
                        }
                        None => true,
                    };
                    let accept =
                        basic_accept && imu_accept && continuity_accept && depth_ratio_accept;
                    stats.passed_acceptance_gates = accept;
                    (stats, accept)
                };

            let mut broader_descriptor_store =
                Some(relocalization_recent_keyframe_descriptor_store(
                    &self.map,
                    state.config.recent_keyframe_window,
                ));
            let appearance_store =
                state
                    .config
                    .appearance_retrieval_map
                    .as_ref()
                    .and_then(|appearance_config| {
                        relocalization_appearance_descriptor_store(
                            &self.map,
                            frame,
                            appearance_config,
                        )
                    });
            let covis_store =
                state
                    .config
                    .covisibility_local_map
                    .as_ref()
                    .and_then(|covis_config| {
                        self.tracker
                            .last_successful_frame_id()
                            .and_then(|last_id| {
                                relocalization_pick_covisibility_reference_keyframe(
                                    &self.map, last_id,
                                )
                            })
                            .and_then(|reference_id| {
                                relocalization_covisibility_descriptor_store(
                                    &self.map,
                                    reference_id,
                                    covis_config,
                                )
                                .map(|store| (reference_id, store))
                            })
                    });
            let (
                initial_descriptor_store,
                initial_store_kind,
                reference_keyframe_id,
                appearance_candidate_keyframe_count,
                appearance_best_similarity,
                appearance_best_keyframe_id,
                appearance_candidates,
            ) = if let Some(appearance) = appearance_store {
                (
                    appearance.store,
                    RelocalizationDescriptorStoreKind::AppearanceRetrieval,
                    None,
                    appearance.candidate_keyframe_count,
                    appearance.best_similarity,
                    appearance.best_keyframe_id,
                    appearance.candidates,
                )
            } else if let Some((reference_keyframe_id, store)) = covis_store {
                (
                    store,
                    RelocalizationDescriptorStoreKind::CovisibilityLocal,
                    Some(reference_keyframe_id),
                    0,
                    None,
                    None,
                    Vec::new(),
                )
            } else {
                (
                    broader_descriptor_store
                        .take()
                        .expect("broader descriptor store was built"),
                    RelocalizationDescriptorStoreKind::Broader,
                    None,
                    0,
                    None,
                    None,
                    Vec::new(),
                )
            };
            let fallback_to_broader_store_on_failure = match initial_store_kind {
                RelocalizationDescriptorStoreKind::AppearanceRetrieval => state
                    .config
                    .appearance_retrieval_map
                    .as_ref()
                    .is_some_and(|config| config.fallback_to_broader_store_on_failure),
                RelocalizationDescriptorStoreKind::CovisibilityLocal => state
                    .config
                    .covisibility_local_map
                    .as_ref()
                    .is_some_and(|config| config.fallback_to_broader_store_on_failure),
                RelocalizationDescriptorStoreKind::Broader => false,
            };
            let compare_broader_store_on_success = match initial_store_kind {
                RelocalizationDescriptorStoreKind::AppearanceRetrieval => state
                    .config
                    .appearance_retrieval_map
                    .as_ref()
                    .is_some_and(|config| config.compare_broader_store_on_success),
                RelocalizationDescriptorStoreKind::CovisibilityLocal => state
                    .config
                    .covisibility_local_map
                    .as_ref()
                    .is_some_and(|config| config.compare_broader_store_on_success),
                RelocalizationDescriptorStoreKind::Broader => false,
            };
            let broader_store_retry_interval_frames = match initial_store_kind {
                RelocalizationDescriptorStoreKind::AppearanceRetrieval => state
                    .config
                    .appearance_retrieval_map
                    .as_ref()
                    .map(|config| config.broader_store_retry_interval_frames.max(1))
                    .unwrap_or(1),
                RelocalizationDescriptorStoreKind::CovisibilityLocal => state
                    .config
                    .covisibility_local_map
                    .as_ref()
                    .map(|config| config.broader_store_retry_interval_frames.max(1))
                    .unwrap_or(1),
                RelocalizationDescriptorStoreKind::Broader => 1,
            };
            let broader_retry_interval_allows = state
                .last_broader_descriptor_store_retry_frame_id
                .is_none_or(|last_frame_id| {
                    frame.id.saturating_sub(last_frame_id) >= broader_store_retry_interval_frames
                });
            let initial_is_covisibility =
                initial_store_kind == RelocalizationDescriptorStoreKind::CovisibilityLocal;
            let initial_is_appearance =
                initial_store_kind == RelocalizationDescriptorStoreKind::AppearanceRetrieval;
            let initial_is_narrow =
                initial_store_kind != RelocalizationDescriptorStoreKind::Broader;
            let covisibility_local_descriptor_store_landmark_count =
                initial_is_covisibility.then_some(initial_descriptor_store.len());
            let appearance_descriptor_store_landmark_count =
                initial_is_appearance.then_some(initial_descriptor_store.len());
            let mut recovered = localize_with_store(&initial_descriptor_store);
            let (mut stats, mut accept) = build_attempt_stats(
                &recovered,
                initial_descriptor_store.len(),
                covisibility_local_descriptor_store_landmark_count,
                appearance_descriptor_store_landmark_count,
                None,
                initial_is_covisibility,
                initial_is_covisibility,
                initial_is_appearance,
                initial_is_appearance,
                false,
                false,
                false,
                reference_keyframe_id,
                appearance_candidate_keyframe_count,
                appearance_best_similarity,
                appearance_best_keyframe_id,
                appearance_candidates.clone(),
            );
            let wants_broader_retry = initial_is_narrow
                && ((!accept && fallback_to_broader_store_on_failure)
                    || (accept && compare_broader_store_on_success));
            let should_try_broader = wants_broader_retry && broader_retry_interval_allows;
            if wants_broader_retry && !broader_retry_interval_allows {
                stats.broader_descriptor_store_retry_skipped_by_interval = true;
            }
            if should_try_broader {
                let broader_store = broader_descriptor_store.take().unwrap_or_else(|| {
                    relocalization_recent_keyframe_descriptor_store(
                        &self.map,
                        state.config.recent_keyframe_window,
                    )
                });
                let broader_store_len = broader_store.len();
                let broader_recovered = localize_with_store(&broader_store);
                let (broader_stats, broader_accept) = build_attempt_stats(
                    &broader_recovered,
                    broader_store_len,
                    covisibility_local_descriptor_store_landmark_count,
                    appearance_descriptor_store_landmark_count,
                    Some(broader_store_len),
                    initial_is_covisibility,
                    false,
                    initial_is_appearance,
                    false,
                    true,
                    false,
                    true,
                    reference_keyframe_id,
                    appearance_candidate_keyframe_count,
                    appearance_best_similarity,
                    appearance_best_keyframe_id,
                    appearance_candidates.clone(),
                );
                stats.tried_broader_descriptor_store_fallback = true;
                stats.broader_descriptor_store_landmark_count = Some(broader_store_len);
                let use_broader = if !accept {
                    true
                } else {
                    broader_accept
                        && relocalization_result_has_better_score(&broader_recovered, &recovered)
                };
                if use_broader {
                    recovered = broader_recovered;
                    stats = broader_stats;
                    accept = broader_accept;
                }
            }
            (recovered, stats, accept, should_try_broader)
        };
        if tried_broader_descriptor_store {
            state.last_broader_descriptor_store_retry_frame_id = Some(frame.id);
        }
        if !accept {
            state.consecutive_failed_attempts = state.consecutive_failed_attempts.saturating_add(1);
            state.pending_confirmation = None;
            return Some(stats);
        }
        let confirmation_required = state.config.confirmation_required_recoveries.max(1);
        stats.confirmation_required_count = confirmation_required;
        if confirmation_required > 1 {
            let Some(recovered_pose) = recovered.pose.as_ref() else {
                state.consecutive_failed_attempts =
                    state.consecutive_failed_attempts.saturating_add(1);
                state.pending_confirmation = None;
                stats.passed_acceptance_gates = false;
                return Some(stats);
            };
            let mut confirmation_count = 1usize;
            if let Some(pending) = state.pending_confirmation.as_ref() {
                let translation = (recovered_pose.camera_center_world()
                    - pending.pose.camera_center_world())
                .norm();
                let frame_gap = frame.id.saturating_sub(pending.frame_id).max(1) as f64;
                let translation_per_frame = translation / frame_gap;
                stats.confirmation_translation_per_frame_from_previous_meters =
                    Some(translation_per_frame);
                let chain_is_consistent = state
                    .config
                    .confirmation_max_translation_per_frame_meters
                    .is_none_or(|max| translation_per_frame <= max);
                if chain_is_consistent {
                    confirmation_count = pending.count.saturating_add(1);
                }
            }
            stats.confirmation_count = confirmation_count;
            if confirmation_count < confirmation_required {
                state.pending_confirmation = Some(OnlineSlamRelocalizationPendingConfirmation {
                    frame_id: frame.id,
                    pose: recovered_pose.clone(),
                    count: confirmation_count,
                });
                return Some(stats);
            }
            state.pending_confirmation = None;
        } else {
            stats.confirmation_count = 1;
            state.pending_confirmation = None;
        }
        let recovered_tracking = TrackingResult {
            frame_id: frame.id,
            state: visloc_tracking::TrackingState::Tracking,
            event: visloc_tracking::TrackingEvent::Relocalized,
            successive_failures: 0,
            pose_prior: tracking.pose_prior.clone(),
            used_pose_prior: tracking.used_pose_prior,
            used_external_localization_prior: tracking.used_external_localization_prior,
            external_localization_prior_radius: tracking.external_localization_prior_radius,
            tracking_failure_reason: None,
            map_landmark_count: tracking.map_landmark_count,
            map_stats: tracking.map_stats,
            localization: recovered,
            covisibility_local_map_size: tracking.covisibility_local_map_size,
        };
        self.tracker
            .accept_relocalization_result(recovered_tracking.clone());
        state.success_count += 1;
        state.consecutive_failed_attempts = 0;
        state.last_success_frame_id = Some(frame.id);
        stats.succeeded = true;
        *tracking = recovered_tracking;
        Some(stats)
    }

    /// Online loop-closure + pose-graph refinement. Runs verifier on the
    /// candidates produced by [`detect_loop_closure_candidates`] this
    /// frame, folds verified constraints into a running [`PoseGraph`]
    /// that mirrors the registered keyframes, and triggers
    /// [`PoseGraph::optimize_se3_iterative`] when the new-constraint
    /// threshold has been reached. On a converged solve the optimised
    /// poses are written back into `self.map.keyframes[id].frame.pose`.
    ///
    /// Returns `None` when the stage is disabled OR when the current
    /// frame did not register a new keyframe (the running graph has no
    /// node to anchor verifier output to). Returns `Some(stats)` whenever
    /// the verifier was actually run this frame; `stats.pose_graph_result`
    /// is `Some` only on the frames where PGO fired.
    fn maybe_run_loop_closure_refinement(
        &mut self,
        frame: &Frame,
        tracking: &TrackingResult,
        applied_update: Option<&AppliedMapUpdate>,
        loop_closure_candidates: &mut [LoopClosureCandidate],
        metric_points_camera: &HashMap<usize, Point3<f64>>,
    ) -> Option<OnlineSlamLoopClosureRefinementStats> {
        let state = self.pose_graph_state.as_mut()?;
        // The pipeline mirrors keyframes into the running graph on
        // every new-keyframe frame. Without a new keyframe this call,
        // the verifier output would have no node to anchor against
        // (loop constraints target the just-finalised keyframe id),
        // so skip the entire stage.
        applied_update?;
        let new_keyframe_id = frame.id;
        // Snapshot the new keyframe's pose from the map (which already
        // reflects local-VI-BA refinement applied earlier this frame).
        let new_pose = self
            .map
            .keyframes
            .get(&new_keyframe_id)
            .and_then(|kf| kf.frame.pose.clone())?;
        let mut stats = OnlineSlamLoopClosureRefinementStats::default();

        // Add the node + the sequential edge from the previous
        // keyframe in registration order. Anchor on the first registered
        // keyframe so the absolute frame stays fixed across PGO solves.
        let prev_keyframe_id = state.keyframe_order.last().copied();
        state.graph.add_pose(new_keyframe_id, new_pose.clone());
        if let Some(sim3_graph) = state.sim3_graph.as_mut() {
            sim3_graph.add_pose(
                new_keyframe_id,
                sim3_at_unit_scale(&new_pose.world_to_camera),
            );
        }
        if state.keyframe_order.is_empty() {
            state.graph.anchor(new_keyframe_id);
            if let Some(sim3_graph) = state.sim3_graph.as_mut() {
                sim3_graph.anchor(new_keyframe_id);
            }
        } else if let Some(prev_id) = prev_keyframe_id {
            if let Some(prev_pose) = state.graph.poses.get(&prev_id).cloned() {
                let relative = relative_world_to_camera(&prev_pose, &new_pose);
                if let Some(sim3_graph) = state.sim3_graph.as_mut() {
                    // Scale-1 measurement: the odometry chain's relative
                    // motion between two consecutive keyframes is derived
                    // from the (already metric) tracked poses, same as the
                    // `Se3` path's sequential edge — no independent scale
                    // source exists between adjacent frames.
                    sim3_graph.add_edge(
                        prev_id,
                        new_keyframe_id,
                        sim3_at_unit_scale(&relative),
                        1.0,
                    );
                }
                let sequential_information =
                    state
                        .config
                        .loop_pose_information
                        .map(|information_config| {
                            if !matches!(state.config.solver, LoopRefinementSolver::Se3) {
                                return Err(LoopPoseInformationFailure::UnsupportedSolver);
                            }
                            let pnp_inliers: Vec<(usize, u64)> = tracking
                                .localization
                                .inlier_query_indices
                                .iter()
                                .copied()
                                .zip(tracking.localization.inlier_landmark_ids.iter().copied())
                                .collect();
                            let measurement = LoopClosureConstraint {
                                from_keyframe_id: prev_id,
                                to_keyframe_id: new_keyframe_id,
                                relative_pose: relative.clone(),
                                inlier_count: pnp_inliers.len(),
                                inlier_ratio: 1.0,
                                mean_sampson_error: 0.0,
                                score: pnp_inliers.len() as f64,
                            };
                            estimate_loop_pose_information(
                                &self.map,
                                frame,
                                &state.config.camera,
                                &measurement,
                                &pnp_inliers,
                                information_config,
                            )
                        });
                if let Some(Ok((information, diagnostic))) = sequential_information {
                    state.graph.add_edge_with_information(
                        prev_id,
                        new_keyframe_id,
                        relative,
                        PoseGraphEdgeKind::Sequential,
                        information,
                    );
                    stats.sequential_edges_with_pose_information += 1;
                    stats
                        .sequential_pose_information_diagnostics
                        .push(diagnostic);
                } else {
                    state
                        .graph
                        .add_sequential_edge(prev_id, new_keyframe_id, relative);
                    if state.config.loop_pose_information.is_some() {
                        stats.sequential_pose_information_fallbacks += 1;
                    }
                    if let Some(Err(reason)) = sequential_information {
                        stats.sequential_pose_information_failures.record(reason);
                    }
                }
            }
        }
        state.keyframe_order.push(new_keyframe_id);

        // Cache this keyframe's mean local descriptor for the appearance-
        // based long-range candidate source. Independent of whether
        // relocalization is configured — this cache lives on the pose-
        // graph refinement state, keyed only on `appearance_candidates`.
        if state.config.appearance_candidates.is_some() {
            if let Some(mean_descriptor) = relocalization_mean_descriptor(&frame.descriptors) {
                state
                    .appearance_descriptor_cache
                    .insert(new_keyframe_id, mean_descriptor);
            }
        }

        // Verify candidates with the configured backend (see
        // `LoopRefinementVerifier`): essential-matrix (default, scale-free —
        // every accepted constraint's translation is pinned to
        // `verifier_config.default_translation_scale`) or PnP (2D-3D,
        // metric translation recovered from the map's triangulated
        // landmarks; candidates with too few 2D-3D correspondences are
        // rejected outright, never falling back to essential-matrix).
        if !loop_closure_candidates.is_empty() {
            stats.verified_candidate_count = loop_closure_candidates.len();
            match &state.config.verifier {
                LoopRefinementVerifier::EssentialMatrix => {
                    let verifier = EssentialMatrixLoopClosureVerifier {
                        config: state.config.verifier_config,
                        ..Default::default()
                    };
                    verify_loop_closure_candidates(
                        loop_closure_candidates,
                        frame,
                        tracking,
                        &self.map,
                        &state.config.camera,
                        &verifier,
                    );
                }
                LoopRefinementVerifier::Pnp(pnp_config) => {
                    let verifier = PnPLoopClosureVerifier {
                        ransac: PnPRansac::default(),
                        config: *pnp_config,
                    };
                    verify_loop_closure_candidates_pnp(
                        loop_closure_candidates,
                        frame,
                        tracking,
                        &self.map,
                        &state.config.camera,
                        &verifier,
                    );
                }
            }
        }

        // Appearance-based long-range candidate stream (opt-in, see
        // `LoopAppearanceCandidateConfig`). Independent of the shared-
        // landmark stream above: built and PnP-verified regardless of
        // whether `detect_loop_closure_candidates` found anything this
        // frame, since the whole point is to catch loops the shared-id
        // detector structurally cannot see.
        let appearance_candidates: Vec<LoopClosureCandidate> = if let Some(appearance_config) =
            state.config.appearance_candidates.clone()
        {
            stats.appearance_ranked_candidate_count =
                relocalization_mean_descriptor(&frame.descriptors)
                    .map(|descriptor| {
                        rank_appearance_loop_candidate_keyframes(
                            &state.appearance_descriptor_cache,
                            frame.id,
                            &descriptor,
                            appearance_config.min_similarity,
                            appearance_config.min_keyframe_id_gap,
                        )
                        .len()
                        .min(appearance_config.max_candidates_per_frame.max(1))
                    })
                    .unwrap_or(0);
            let mut builder_config = appearance_config.clone();
            // ORB-SLAM3 first obtains one strong current-vs-region pose,
            // then carries that region into later keyframes. Do not make
            // the one-frame builder consume the cross-keyframe count.
            builder_config.min_covisible_keyframe_verifications = 1;
            let built = build_appearance_loop_candidates_with_diagnostics(
                &self.map,
                frame,
                &state.appearance_descriptor_cache,
                &builder_config,
                &state.config.camera,
            );
            stats.appearance_connected_region_rejected_count =
                built.connected_region_rejected_count;
            stats.appearance_pnp_verified_count = built.pnp_verified_count;
            stats.appearance_projection_rejected_count = built.projection_rejected_count;
            stats.appearance_covisibility_rejected_count = built.covisibility_rejected_count;
            let mut candidates = built.candidates;
            let pending_needs_projection =
                state
                    .appearance_pending_region
                    .as_ref()
                    .is_some_and(|pending| {
                        !candidates.iter().any(|candidate| {
                            appearance_regions_overlap(
                                &self.map,
                                pending.root_keyframe_id,
                                candidate.matched_keyframe_id,
                                &appearance_config,
                            )
                        })
                    });
            if pending_needs_projection {
                let projected = verify_pending_appearance_region_by_projection(
                    &self.map,
                    frame,
                    &state.config.camera,
                    &appearance_config,
                    state
                        .appearance_pending_region
                        .as_ref()
                        .expect("pending projection was requested"),
                );
                stats.appearance_pending_projection_attempted_count =
                    usize::from(projected.attempted);
                stats.appearance_pending_projection_verified_count =
                    usize::from(projected.candidate.is_some());
                stats.appearance_pending_projection_correspondence_count =
                    projected.correspondence_count;
                stats.appearance_pending_projection_inlier_count = projected.inlier_count;
                stats.appearance_pending_projection_search_radius_px = projected.search_radius_px;
                if let Some(candidate) = projected.candidate {
                    candidates.push(candidate);
                }
            }
            stats.appearance_candidate_count = candidates.len();
            stats.appearance_candidate_diagnostics = built.diagnostics;
            let (confirmed, waiting, just_confirmed) = confirm_appearance_region_across_keyframes(
                &self.map,
                frame.id,
                candidates,
                &appearance_config,
                &mut state.appearance_pending_region,
            );
            stats.appearance_region_confirmation_waiting = waiting;
            stats.appearance_region_confirmed_count = usize::from(just_confirmed);
            confirmed
        } else {
            Vec::new()
        };

        if !loop_closure_candidates.is_empty() || !appearance_candidates.is_empty() {
            // PCM front-end screen (optional): snapshot the current odometry
            // (graph poses) and the already-admitted closures once, then admit
            // a new closure only if it is geometrically consistent with the
            // established set — so a wrong closure never enters the graph.
            let pcm_cfg = state.config.pcm;
            let odometry: BTreeMap<u64, SE3> = if pcm_cfg.is_some() {
                state
                    .graph
                    .poses
                    .iter()
                    .map(|(id, p)| (*id, p.world_to_camera.clone()))
                    .collect()
            } else {
                BTreeMap::new()
            };
            let mut admitted: Vec<pcm::LoopMeasurement> = if pcm_cfg.is_some() {
                state
                    .verified_constraints
                    .iter()
                    .map(loop_measurement_of)
                    .collect()
            } else {
                Vec::new()
            };

            // Both streams flow through the identical accept path (PCM /
            // covariance gate / graph add / write-back) so folding is
            // untouched regardless of which stream a candidate came from.
            let tagged_candidates = loop_closure_candidates
                .iter()
                .map(|candidate| (candidate, LoopClosureCandidateSource::SharedLandmark))
                .chain(
                    appearance_candidates
                        .iter()
                        .map(|candidate| (candidate, LoopClosureCandidateSource::Appearance)),
                );

            for (candidate, source) in tagged_candidates {
                let Some(constraint) = LoopClosureConstraint::from_verified_candidate(candidate)
                else {
                    continue;
                };
                // Loop edges target the keyframe registered this frame
                // (`candidate.query_frame_id == new_keyframe_id`); the
                // matched keyframe must already exist in the graph.
                if !state.graph.poses.contains_key(&constraint.from_keyframe_id) {
                    continue;
                }
                let measured_sim3_scale =
                    if matches!(state.config.solver, LoopRefinementSolver::Sim3(_))
                        && source == LoopClosureCandidateSource::Appearance
                    {
                        let scale = match estimate_loop_sim3_scale_3d3d(
                            &self.map,
                            constraint.from_keyframe_id,
                            constraint.to_keyframe_id,
                            &candidate.pnp_query_landmark_pairs,
                            metric_points_camera,
                        ) {
                            Ok(scale) => scale,
                            Err(reason) => {
                                stats.appearance_scale_estimation_failed_count += 1;
                                match reason {
                                    Sim3ScaleEstimationFailure::InsufficientPoints => {
                                        stats.appearance_scale_insufficient_points_count += 1;
                                    }
                                    Sim3ScaleEstimationFailure::InsufficientMatches => {
                                        stats.appearance_scale_insufficient_matches_count += 1;
                                    }
                                    Sim3ScaleEstimationFailure::NoConsensus => {
                                        stats.appearance_scale_no_consensus_count += 1;
                                    }
                                }
                                // A PnP pose alone contains no scale observation.
                                // Do not let an arbitrary scale-1 edge perturb an
                                // already accurate map on the Sim3-only path.
                                continue;
                            }
                        };
                        if (scale - 1.0).abs() < 0.05 {
                            stats.appearance_near_unit_scale_count += 1;
                            // This solver option is specifically the scale-drift
                            // correction arm. Leave near-unit rigid corrections
                            // to the established SE3 path.
                            continue;
                        }
                        Some(scale)
                    } else {
                        None
                    };
                let loop_information =
                    if let Some(information_config) = state.config.loop_pose_information {
                        // The 6×6 matrix describes an SE(3) tangent. Do not
                        // silently reuse it for the 7-DoF Sim(3) mirror.
                        let estimate = if matches!(state.config.solver, LoopRefinementSolver::Se3) {
                            estimate_loop_pose_information(
                                &self.map,
                                frame,
                                &state.config.camera,
                                &constraint,
                                &candidate.pnp_query_landmark_pairs,
                                information_config,
                            )
                        } else {
                            Err(LoopPoseInformationFailure::UnsupportedSolver)
                        };
                        let (mut information, diagnostic) = match estimate {
                            Ok(estimate) => estimate,
                            Err(reason) => {
                                stats.loop_pose_information_failures.record(reason);
                                stats.loop_closures_pose_information_rejected += 1;
                                stats
                            .rejected_constraints
                            .push(OnlineSlamRejectedLoopConstraint {
                                from_keyframe_id: constraint.from_keyframe_id,
                                to_keyframe_id: constraint.to_keyframe_id,
                                inlier_count: constraint.inlier_count,
                                translation_norm_m: constraint.relative_pose.translation.norm(),
                                relative_pose: constraint.relative_pose.clone(),
                                source,
                                reason: OnlineSlamLoopConstraintRejectionReason::PoseInformation,
                            });
                                continue;
                            }
                        };
                        // Preserve the covariance-derived anisotropy while
                        // controlling loop strength relative to sequential
                        // PnP odometry information.
                        information =
                            apply_loop_edge_scale(information, information_config.loop_edge_scale);
                        stats.loop_pose_information_diagnostics.push(diagnostic);
                        state.loop_pose_information.insert(
                            (constraint.from_keyframe_id, constraint.to_keyframe_id),
                            information,
                        );
                        Some(information)
                    } else {
                        None
                    };
                // Front-end screens (both before the closure enters the graph):
                // PCM combinatorial consistency, then the covariance metric gate.
                let pcm_measurement = pcm_cfg.as_ref().map(|cfg| {
                    let m = loop_measurement_of(&constraint);
                    (pcm_admits_loop(&m, &admitted, &odometry, cfg), m)
                });
                if let Some((false, _)) = pcm_measurement {
                    stats.loop_closures_pcm_rejected += 1;
                    stats
                        .rejected_constraints
                        .push(OnlineSlamRejectedLoopConstraint {
                            from_keyframe_id: constraint.from_keyframe_id,
                            to_keyframe_id: constraint.to_keyframe_id,
                            inlier_count: constraint.inlier_count,
                            translation_norm_m: constraint.relative_pose.translation.norm(),
                            relative_pose: constraint.relative_pose.clone(),
                            source,
                            reason: OnlineSlamLoopConstraintRejectionReason::Pcm,
                        });
                    // Hold a deferred-but-verified closure for the batch
                    // re-screen, which may promote it once a consensus forms.
                    if state.config.pcm_batch_rescreen {
                        state.pcm_deferred.push(constraint);
                    }
                    continue;
                }
                if let Some(threshold) = state.config.covariance_gate {
                    if !covariance_gate_admits(&state.graph, &constraint, threshold) {
                        stats.loop_closures_covariance_rejected += 1;
                        stats
                            .rejected_constraints
                            .push(OnlineSlamRejectedLoopConstraint {
                                from_keyframe_id: constraint.from_keyframe_id,
                                to_keyframe_id: constraint.to_keyframe_id,
                                inlier_count: constraint.inlier_count,
                                translation_norm_m: constraint.relative_pose.translation.norm(),
                                relative_pose: constraint.relative_pose.clone(),
                                source,
                                reason: OnlineSlamLoopConstraintRejectionReason::Covariance,
                            });
                        continue;
                    }
                }
                if let Some((_, m)) = pcm_measurement {
                    admitted.push(m);
                }
                let loop_edge_weight = state
                    .config
                    .fixed_loop_edge_weight
                    .filter(|weight| weight.is_finite() && *weight > 0.0)
                    .unwrap_or_else(|| (constraint.inlier_count as f64).max(1.0));
                if let Some(information) = loop_information {
                    state.graph.add_edge_with_information(
                        constraint.from_keyframe_id,
                        constraint.to_keyframe_id,
                        constraint.relative_pose.clone(),
                        PoseGraphEdgeKind::LoopClosure,
                        information,
                    );
                    stats.loop_closures_with_pose_information += 1;
                } else {
                    state
                        .graph
                        .add_loop_closure_constraint_with_weight(&constraint, loop_edge_weight);
                }
                if let Some(sim3_graph) = state.sim3_graph.as_mut() {
                    sim3_graph.add_edge(
                        constraint.from_keyframe_id,
                        constraint.to_keyframe_id,
                        Sim3::new(
                            constraint.relative_pose.rotation,
                            constraint.relative_pose.translation,
                            measured_sim3_scale.unwrap_or(1.0),
                        ),
                        loop_edge_weight,
                    );
                }
                stats
                    .admitted_constraints
                    .push(OnlineSlamAdmittedLoopConstraint {
                        from_keyframe_id: constraint.from_keyframe_id,
                        to_keyframe_id: constraint.to_keyframe_id,
                        inlier_count: constraint.inlier_count,
                        translation_norm_m: constraint.relative_pose.translation.norm(),
                        relative_pose: constraint.relative_pose.clone(),
                        source,
                    });
                if source == LoopClosureCandidateSource::Appearance {
                    stats.appearance_accepted_count += 1;
                    if state.config.fuse_loop_observations {
                        state
                            .pending_loop_observation_fusions
                            .push(PendingLoopObservationFusion {
                                from_keyframe_id: constraint.from_keyframe_id,
                                query_keyframe_id: constraint.to_keyframe_id,
                                relative_pose: constraint.relative_pose.clone(),
                                pairs: candidate.pnp_query_landmark_pairs.clone(),
                            });
                    }
                }
                state.verified_constraints.push(constraint);
                state.pending_since_last_trigger += 1;
                stats.accepted_count += 1;
            }

            // Batch PCM self-heal (optional): the incremental admission above is
            // order-dependent — a wrong closure admitted first (against the
            // empty set) poisons the genuine ones checked against it. Recompute
            // the order-independent maximum-consistent set over the full history
            // (admitted ∪ deferred) and reconcile the graph's loop edges to it,
            // promoting/evicting as the consensus dictates.
            if state.config.pcm_batch_rescreen {
                if let Some(cfg) = pcm_cfg {
                    let (promoted, evicted) = pcm_batch_reconcile(
                        &mut state.graph,
                        &mut state.verified_constraints,
                        &mut state.pcm_deferred,
                        &odometry,
                        &cfg,
                        state.config.fixed_loop_edge_weight,
                    );
                    stats.loop_closures_pcm_promoted = promoted;
                    stats.loop_closures_pcm_evicted = evicted;
                    // `pcm_batch_reconcile` reconstructs promoted edges via
                    // the legacy scalar API. Restore the covariance-derived
                    // matrix retained when that candidate was first verified.
                    for edge in &mut state.graph.edges {
                        if edge.kind != PoseGraphEdgeKind::LoopClosure || edge.information.is_some()
                        {
                            continue;
                        }
                        if let Some(information) = state
                            .loop_pose_information
                            .get(&(edge.from, edge.to))
                            .copied()
                        {
                            edge.information = Some(information);
                            stats.loop_closures_with_pose_information += 1;
                        }
                    }
                    // A changed loop-edge set must be re-solved.
                    state.pending_since_last_trigger += promoted + evicted;
                }
            }
        }

        // Trigger PGO when the configured number of new constraints has
        // accumulated. A higher threshold batches solves; `1` (the
        // recommended default) runs PGO on every accepted loop edge.
        let trigger_threshold = state.config.trigger_every_new_constraints.max(1);
        if state.pending_since_last_trigger >= trigger_threshold {
            let pgo_config = state.config.pose_graph_config.clone();
            let sim3_solver_config = match &state.config.solver {
                LoopRefinementSolver::Sim3(cfg) => Some(cfg.clone()),
                LoopRefinementSolver::Se3 => None,
            };
            // Preserve the actual pre-solve Sim3 nodes, including their
            // accumulated scales. Re-embedding the rigid write-back poses at
            // unit scale here would make an unchanged non-unit solution look
            // like a fresh correction on every solve, repeatedly moving
            // landmarks and the tracker.
            let pre_solve_sim3_nodes = state.sim3_graph.as_ref().map(|graph| graph.poses.clone());
            state.pending_since_last_trigger = 0;
            state.trigger_count += 1;
            // `Sim3` solves the parallel Sim3 mirror instead of the rigid
            // graph; `gnc` has no Sim3 counterpart yet (see
            // `LoopRefinementSolver::Sim3`'s doc comment) so it is ignored
            // on this path. Otherwise: when GNC is configured, run the
            // robust solver so a verified-but-wrong loop closure is
            // annealed out at the back-end; otherwise the plain
            // M-estimator. All paths write the optimised poses back into
            // the map, so a wrong closure GNC rejected never reaches
            // subsequent tracking / local-VI-BA.
            let solved = if let Some(sim3_config) = sim3_solver_config {
                match state
                    .sim3_graph
                    .get_or_insert_with(Sim3PoseGraph::new)
                    .optimize(&sim3_config)
                {
                    Ok(result) => {
                        stats.sim3_scale_spread =
                            state.sim3_graph.as_ref().and_then(sim3_pose_scale_spread);
                        stats.sim3_pose_graph_result = Some(result);
                        true
                    }
                    Err(_) => false,
                }
            } else if let Some(gnc_config) = state.config.gnc {
                match state.graph.optimize_se3_gnc(&pgo_config, &gnc_config) {
                    Ok(result) => {
                        // Count the loop-closure edges driven below the
                        // inlier band — the rejected wrong closures.
                        // `edge_weights` is in `graph.edges` order.
                        stats.loop_closures_rejected = state
                            .graph
                            .edges
                            .iter()
                            .zip(&result.edge_weights)
                            .filter(|(edge, &w)| {
                                edge.kind == PoseGraphEdgeKind::LoopClosure && w < 0.5
                            })
                            .count();
                        stats.gnc_result = Some(result);
                        true
                    }
                    Err(_) => false,
                }
            } else {
                match state.graph.optimize_se3_iterative(&pgo_config) {
                    Ok(result) => {
                        stats.pose_graph_result = Some(result);
                        true
                    }
                    Err(_) => false,
                }
            };
            if solved && matches!(state.config.solver, LoopRefinementSolver::Sim3(_)) {
                // Write-back convention (see `LoopRefinementSolver::Sim3`'s
                // doc comment for the full derivation): each solved node
                // `Siw_new` becomes the rigid pose `[R | t/s]` for
                // `map.keyframes[*].frame.pose` — dividing the translation
                // by the solved scale is what keeps the corrected pose's
                // reprojection *ray* invariant (pinhole projection ignores
                // a positive rescale of camera-frame depth). The SE(3)
                // mirror (`state.graph.poses`) is also updated so the next
                // keyframe's sequential edge is built from the corrected
                // trajectory, not the stale pre-solve one.
                let mut updated = 0usize;
                let mut corrections: HashMap<u64, Sim3> = HashMap::new();
                let solved_nodes: Vec<(u64, Sim3)> = state
                    .sim3_graph
                    .as_ref()
                    .map(|graph| graph.poses.iter().map(|(id, s)| (*id, s.clone())).collect())
                    .unwrap_or_default();
                for (id, siw_new) in &solved_nodes {
                    let Some(keyframe) = self.map.keyframes.get_mut(id) else {
                        continue;
                    };
                    let new_pose = Pose {
                        world_to_camera: SE3::new(
                            siw_new.rotation,
                            siw_new.translation / siw_new.scale,
                        ),
                    };
                    if state.config.propagate_corrections {
                        if let Some(siw_old) = pre_solve_sim3_nodes
                            .as_ref()
                            .and_then(|nodes| nodes.get(id))
                        {
                            corrections.insert(*id, siw_new.inverse().compose(siw_old));
                        }
                    }
                    keyframe.frame.pose = Some(new_pose.clone());
                    state.graph.poses.insert(*id, new_pose);
                    updated += 1;
                }
                stats.keyframes_updated = updated;

                if state.config.propagate_corrections && !corrections.is_empty() {
                    // Landmark propagation: each landmark moves with its
                    // ANCHOR keyframe's world-frame *similarity*
                    // `Siw_new⁻¹ ∘ Siw_old` (the direct Sim(3)
                    // generalisation of the SE(3) path's
                    // `T_cw_new⁻¹ ∘ T_cw_old`); see
                    // `propagate_pose_graph_corrections_sim3`.
                    let (moved, max_displacement, mean_displacement) =
                        propagate_pose_graph_corrections_sim3(&mut self.map, &corrections);
                    stats.landmarks_moved = moved;
                    stats.max_landmark_displacement_meters = max_displacement;
                    stats.mean_landmark_displacement_meters = mean_displacement;

                    // Tracker/current-pose propagation: rotation+
                    // translation into `last_successful_pose`, plus the
                    // motion model's cached world velocity scaled by
                    // `correction.scale` (see
                    // `Tracker::apply_similarity_pose_correction`).
                    if let Some((_, last_correction)) =
                        corrections.iter().max_by_key(|(id, _)| **id)
                    {
                        self.tracker
                            .apply_similarity_pose_correction(last_correction);
                        stats.tracker_correction_applied = true;
                    }
                }
                // `marginalization_window` has no effect on this path yet
                // (`Sim3PoseGraph` has no fixed-lag marginalization) — see
                // that field's doc comment.
            } else if solved {
                // Write optimised poses back into the map so subsequent
                // tracking / local-VI-BA passes see the refined frame.
                // When `propagate_corrections` is set, snapshot each
                // updated keyframe's PRE-solve `world_to_camera` before
                // overwriting it so the rigid world-frame correction
                // `C_k = T_cw_new⁻¹ ∘ T_cw_old` can be computed (see
                // `propagate_pose_graph_corrections` for the derivation
                // and why this is the correction that keeps a landmark's
                // reprojection into keyframe k invariant).
                let mut updated = 0usize;
                let mut corrections: HashMap<u64, SE3> = HashMap::new();
                for (id, pose) in state.graph.poses.iter() {
                    if let Some(keyframe) = self.map.keyframes.get_mut(id) {
                        if state.config.propagate_corrections {
                            if let Some(old_pose) = keyframe.frame.pose.as_ref() {
                                let correction = pose
                                    .world_to_camera
                                    .inverse()
                                    .compose(&old_pose.world_to_camera);
                                corrections.insert(*id, correction);
                            }
                        }
                        keyframe.frame.pose = Some(pose.clone());
                        updated += 1;
                    }
                }
                stats.keyframes_updated = updated;

                if state.config.propagate_corrections && !corrections.is_empty() {
                    // Landmark propagation: each landmark moves rigidly
                    // with its ANCHOR keyframe (the first, lowest-id,
                    // keyframe that observed it). Landmarks anchored to a
                    // keyframe this solve did not update keep their
                    // position untouched.
                    let (moved, max_displacement, mean_displacement) =
                        propagate_pose_graph_corrections(&mut self.map, &corrections);
                    stats.landmarks_moved = moved;
                    stats.max_landmark_displacement_meters = max_displacement;
                    stats.mean_landmark_displacement_meters = mean_displacement;

                    // Tracker/current-pose propagation: map the
                    // tracker's continuation state (last successful pose
                    // + motion-model state) into the corrected frame
                    // using the most-recently solved keyframe's
                    // correction, so the very next `track_frame`'s PnP
                    // prior starts consistent with the just-corrected
                    // map instead of re-anchoring the trajectory to the
                    // pre-solve drift.
                    if let Some((_, last_correction)) =
                        corrections.iter().max_by_key(|(id, _)| **id)
                    {
                        self.tracker.apply_pose_correction(last_correction);
                        stats.tracker_correction_applied = true;
                    }
                }

                // Fixed-lag bound: marginalize the oldest poses past the window
                // into a dense prior, keeping the graph (and next solve) bounded.
                // Runs at the just-solved estimate, so it is first-order exact;
                // the marginalized keyframes keep their written-back pose.
                if let Some(window) = state.config.marginalization_window {
                    let result = if state.config.marginalization_sparsify {
                        state.graph.marginalize_oldest_sparsified(window)
                    } else {
                        state.graph.marginalize_oldest(window)
                    };
                    if let Ok(removed) = result {
                        stats.poses_marginalized = removed;
                    }
                }
            }

            if solved {
                let max_reprojection_error_px = state
                    .config
                    .appearance_candidates
                    .as_ref()
                    .map(|config| config.pnp_verifier.max_mean_reprojection_error_px)
                    .unwrap_or(4.0);
                for pending in std::mem::take(&mut state.pending_loop_observation_fusions) {
                    let edge_is_still_admitted = state
                        .graph
                        .edges
                        .iter()
                        .enumerate()
                        .find(|(_, edge)| {
                            edge.kind == PoseGraphEdgeKind::LoopClosure
                                && edge.from == pending.from_keyframe_id
                                && edge.to == pending.query_keyframe_id
                        })
                        .is_some_and(|(index, _)| {
                            stats
                                .gnc_result
                                .as_ref()
                                .and_then(|result| result.edge_weights.get(index))
                                .is_none_or(|weight| *weight >= 0.5)
                        });
                    if !edge_is_still_admitted {
                        stats.loop_fusion_pairs_considered += pending.pairs.len();
                        stats.loop_fusion_pairs_skipped += pending.pairs.len();
                        stats.loop_fusion_pairs_robust_rejected += pending.pairs.len();
                        continue;
                    }
                    if let Some(welding_config) = state.config.loop_welding_ba.clone() {
                        // ORB-SLAM-style welding is speculative: first move
                        // the current covisible region rigidly onto the PnP
                        // loop pose, then SearchAndFuse and BA on a clone. A
                        // failed selector, fusion, solve, or quality gate
                        // leaves the live map untouched.
                        stats.loop_welding_ba_attempted = true;
                        let active_pose_before = self
                            .map
                            .keyframes
                            .get(&pending.query_keyframe_id)
                            .and_then(|keyframe| keyframe.frame.pose.clone());
                        let live_map_before_welding = self.map.clone();
                        let graph_before_welding = state.graph.clone();
                        let mut candidate_map = live_map_before_welding.clone();
                        let Some(correction) = correct_loop_welding_region(
                            &mut candidate_map,
                            pending.from_keyframe_id,
                            pending.query_keyframe_id,
                            &pending.relative_pose,
                            &welding_config,
                        ) else {
                            stats.loop_fusion_pairs_considered += pending.pairs.len();
                            stats.loop_fusion_pairs_skipped += pending.pairs.len();
                            stats.loop_welding_ba_rejected_or_failed = true;
                            continue;
                        };
                        stats.loop_welding_initial_translation_meters = Some(
                            stats
                                .loop_welding_initial_translation_meters
                                .map_or(correction.translation_meters, |value| {
                                    value.max(correction.translation_meters)
                                }),
                        );
                        stats.loop_welding_initial_rotation_radians = Some(
                            stats
                                .loop_welding_initial_rotation_radians
                                .map_or(correction.rotation_radians, |value| {
                                    value.max(correction.rotation_radians)
                                }),
                        );
                        let (max_welding_translation, max_welding_rotation) = state
                            .config
                            .appearance_candidates
                            .as_ref()
                            .map(|config| {
                                (
                                    config.max_covisibility_translation_disagreement_meters,
                                    config.max_covisibility_rotation_disagreement_radians,
                                )
                            })
                            .unwrap_or((0.5, 0.2));
                        if correction.translation_meters > max_welding_translation
                            || correction.rotation_radians > max_welding_rotation
                        {
                            stats.loop_fusion_pairs_considered += pending.pairs.len();
                            stats.loop_fusion_pairs_skipped += pending.pairs.len();
                            stats.loop_welding_ba_rejected_or_failed = true;
                            continue;
                        }
                        let mut corrected_ids = correction.keyframe_ids;
                        let fusion = fuse_loop_observations(
                            &mut candidate_map,
                            &state.config.camera,
                            pending.query_keyframe_id,
                            &pending.pairs,
                            max_reprojection_error_px,
                        );
                        stats.loop_fusion_pairs_considered += fusion.pairs_considered;
                        stats.loop_fusion_pairs_skipped += fusion.pairs_skipped;
                        stats.loop_fusion_pairs_reprojection_rejected +=
                            fusion.reprojection_rejected;
                        if fusion.observations_inserted == 0 {
                            stats.loop_welding_ba_rejected_or_failed = true;
                            continue;
                        }
                        let rotations_before_ba: HashMap<u64, UnitQuaternion<f64>> = candidate_map
                            .keyframes
                            .iter()
                            .filter_map(|(id, keyframe)| {
                                keyframe
                                    .frame
                                    .pose
                                    .as_ref()
                                    .map(|pose| (*id, pose.world_to_camera.rotation))
                            })
                            .collect();

                        match refine_visual_map_with_covisibility_ba(
                            &mut candidate_map,
                            pending.query_keyframe_id,
                            &welding_config,
                        ) {
                            Ok(result) => {
                                // Preserve the verified/PnP-initialized local
                                // orientations while retaining BA-refined
                                // camera centers and landmarks. This is an
                                // explicit rotation prior implemented as a
                                // transactional projection, until the generic
                                // BA API grows native SO(3) priors.
                                for id in &result.selection.optimized_keyframe_ids {
                                    let (Some(rotation), Some(pose)) = (
                                        rotations_before_ba.get(id),
                                        candidate_map
                                            .keyframes
                                            .get_mut(id)
                                            .and_then(|keyframe| keyframe.frame.pose.as_mut()),
                                    ) else {
                                        continue;
                                    };
                                    let center = pose.camera_center_world();
                                    pose.world_to_camera.rotation = *rotation;
                                    pose.world_to_camera.translation =
                                        -(rotation.transform_vector(&center.coords));
                                }
                                let anchored_reprojection = mean_selected_reprojection_px(
                                    &candidate_map,
                                    &result.selection,
                                );
                                stats.loop_welding_ba_mean_reprojection_before_px =
                                    Some(result.mean_reprojection_before_px);
                                stats.loop_welding_ba_mean_reprojection_after_px =
                                    anchored_reprojection;
                                if !anchored_reprojection.is_some_and(|error| {
                                    error.is_finite()
                                        && error <= result.mean_reprojection_before_px + 1.0e-9
                                }) || !candidate_map.validate().is_valid()
                                {
                                    stats.loop_welding_ba_rejected_or_failed = true;
                                    continue;
                                }
                                corrected_ids.extend(
                                    result.selection.optimized_keyframe_ids.iter().copied(),
                                );
                                corrected_ids.sort_unstable();
                                corrected_ids.dedup();
                                self.map = candidate_map;
                                for id in corrected_ids {
                                    if let Some(pose) = self
                                        .map
                                        .keyframes
                                        .get(&id)
                                        .and_then(|keyframe| keyframe.frame.pose.clone())
                                    {
                                        if state.graph.poses.contains_key(&id) {
                                            state.graph.poses.insert(id, pose);
                                        }
                                    }
                                }
                                stats.loop_welding_post_pgo_attempted = true;
                                let (
                                    post_pgo_succeeded,
                                    post_pgo_reprojection,
                                    post_ba_error,
                                    post_ba_behind_camera_ratio,
                                ) = if matches!(state.config.solver, LoopRefinementSolver::Se3) {
                                    refine_pose_graph_after_welding(
                                        &mut self.map,
                                        &mut state.graph,
                                        &pgo_config,
                                        state.config.gnc.as_ref(),
                                        pending.query_keyframe_id,
                                        &welding_config,
                                        result.mean_reprojection_before_px,
                                    )
                                } else {
                                    (false, None, None, None)
                                };
                                stats.loop_welding_post_pgo_mean_reprojection_px =
                                    post_pgo_reprojection;
                                stats.loop_welding_post_ba_error = post_ba_error;
                                stats.loop_welding_post_ba_behind_camera_ratio =
                                    post_ba_behind_camera_ratio;
                                if !post_pgo_succeeded {
                                    self.map = live_map_before_welding;
                                    state.graph = graph_before_welding;
                                    stats.loop_welding_ba_rejected_or_failed = true;
                                    continue;
                                }
                                stats.loop_welding_post_pgo_succeeded = true;
                                stats.loop_welding_ba_updated_keyframes +=
                                    result.updated_keyframe_count;
                                stats.loop_welding_ba_updated_landmarks +=
                                    result.updated_landmark_count;
                                if let (Some(old_pose), Some(new_pose)) = (
                                    active_pose_before,
                                    self.map
                                        .keyframes
                                        .get(&pending.query_keyframe_id)
                                        .and_then(|keyframe| keyframe.frame.pose.clone()),
                                ) {
                                    let correction = new_pose
                                        .world_to_camera
                                        .inverse()
                                        .compose(&old_pose.world_to_camera);
                                    self.tracker.apply_pose_correction(&correction);
                                    stats.tracker_correction_applied = true;
                                }
                                stats.loop_fusion_observations_inserted +=
                                    fusion.observations_inserted;
                                stats.loop_fusion_observations_reassigned +=
                                    fusion.observations_reassigned;
                                stats.loop_welding_ba_succeeded = true;
                            }
                            Err(_) => stats.loop_welding_ba_rejected_or_failed = true,
                        }
                    } else {
                        let fusion = fuse_loop_observations(
                            &mut self.map,
                            &state.config.camera,
                            pending.query_keyframe_id,
                            &pending.pairs,
                            max_reprojection_error_px,
                        );
                        stats.loop_fusion_pairs_considered += fusion.pairs_considered;
                        stats.loop_fusion_observations_inserted += fusion.observations_inserted;
                        stats.loop_fusion_observations_reassigned += fusion.observations_reassigned;
                        stats.loop_fusion_pairs_skipped += fusion.pairs_skipped;
                        stats.loop_fusion_pairs_reprojection_rejected +=
                            fusion.reprojection_rejected;
                    }
                }
            }
        }

        Some(stats)
    }

    /// Register the freshly-staged IMU factor with the local VI-BA state
    /// table and, when the trigger threshold has been reached, run a
    /// sliding-window VI-BA pass that refines the trailing window's
    /// poses + landmarks + velocities + biases. No-op when local VI-BA
    /// is disabled OR when no IMU factor was staged this frame.
    /// `true` while local VI-BA must stay gated because no stage has yet
    /// replaced the placeholder-zero bias linearisation with a real
    /// estimate.
    ///
    /// When `keep_pre_promotion_imu_factors` lets factors flow before
    /// VI-init promotes, the factors must still be banked for the
    /// post-promotion BA replay, but the BA itself cannot run yet while
    /// its bias linearisation is the placeholder zero seed — a
    /// pre-promotion solve corrupts the map's keyframe poses
    /// (empirically: tracking-success collapses from 9.8 % to 1.8 %
    /// on MH_01 because the next-frame matcher sees BA-shifted
    /// keyframe descriptors).
    ///
    /// The zero-placeholder rationale stops applying the moment EITHER:
    /// * the static stage completes (`vi_init_state.completed.is_some()`),
    ///   or
    /// * the motion-based stage reaches its terminal state
    ///   (`vi_motion_init_state.completed.is_some()`), or
    /// * the motion-based stage has fired a [`crate::BiasReleaseSchedule`]
    ///   Stage A ("velocity stage") solve
    ///   (`vi_motion_init_state.velocity_stage_fired()`).
    ///
    /// Stage A is a non-terminal success: `bias_released` stays `false`
    /// and `vi_motion_init_state.completed` stays `None` so the stage keeps
    /// registering keyframes/translation toward the Stage B release gate,
    /// but its promotion has already mirrored the refined per-keyframe
    /// `(velocity, bias)` into `local_vi_ba_state.keyframe_state` and the
    /// estimated/refined biases into `imu_state`/configs (see
    /// `promote_motion_vi_init_result`). From that point on local VI-BA's
    /// linearisation point is the estimated seed, not the placeholder
    /// zero — ORB-SLAM3 itself runs local VI-BA after VIBA1, which Stage A
    /// mirrors. A later Stage B firing re-mirrors on top harmlessly (same
    /// sinks, refined values), so there is no double-counting risk in
    /// treating Stage A as "no longer pending" here.
    fn vi_initialization_pending(&self) -> bool {
        self.vi_init_state.as_ref().is_some_and(|static_state| {
            static_state.completed.is_none()
                && self
                    .vi_motion_init_state
                    .as_ref()
                    .is_none_or(|motion_state| {
                        motion_state.completed.is_none() && !motion_state.velocity_stage_fired()
                    })
        })
    }

    fn maybe_run_local_vi_ba(
        &mut self,
        new_factor: Option<ImuPreintegrationFactor>,
    ) -> Option<OnlineSlamLocalBaStats> {
        let vi_initialization_pending = self.vi_initialization_pending();
        let state = self.local_vi_ba_state.as_mut()?;
        let factor = new_factor?;
        let should_trigger = state.register_new_factor(factor);
        if !should_trigger || vi_initialization_pending {
            return None;
        }
        crate::online_slam_vi_ba::run_local_vi_ba(&mut self.map, state)
    }

    /// Run visual-only covisibility local BA when a new keyframe has just
    /// entered the map and the configured trigger interval fires. No-op
    /// when disabled, when the mapper only staged-but-did-not-apply the
    /// keyframe, or while the map is still below the configured minimum
    /// keyframe count.
    fn maybe_run_covisibility_local_ba(
        &mut self,
        frame: &Frame,
        applied_update: Option<&AppliedMapUpdate>,
    ) -> Option<OnlineSlamCovisibilityLocalBaStats> {
        let config = self.config.covisibility_local_ba.clone()?;
        let added_new_keyframe = applied_update
            .map(|a| a.keyframe_count > 0)
            .unwrap_or(false);
        if !added_new_keyframe {
            return None;
        }

        if let Some(activation) = config.motion_vi_raw_residual_activation {
            let motion_state = self.vi_motion_init_state.as_ref()?;
            if motion_state.completed.is_some()
                || !motion_state.last_rejection.as_ref().is_some_and(|reason| {
                    motion_vi_raw_residual_activation_satisfied(activation, reason)
                })
            {
                return None;
            }
        }
        if let Some(maximum) = config.max_seed_landmarks_for_activation {
            let seed_support = self
                .map
                .keyframes
                .values()
                .min_by_key(|keyframe| keyframe.frame.id)
                .map(|keyframe| keyframe.observations.len());
            if seed_support.is_none_or(|support| support > maximum) {
                return None;
            }
        }

        let map_keyframe_count = self.map.keyframes.len();
        if map_keyframe_count < config.min_keyframes.max(1) {
            return None;
        }
        if config
            .max_keyframes
            .is_some_and(|maximum| map_keyframe_count > maximum)
        {
            return None;
        }

        let trigger_every = config.trigger_every_new_keyframes.max(1);
        if map_keyframe_count % trigger_every != 0 {
            return None;
        }

        let started = Instant::now();
        // The clone-and-check write-back path activates when any conditioning
        // gate is configured. It solves on a cloned map, evaluates every
        // configured gate against the solved candidate, and only commits the
        // clone into the live map if none fire. If any gate rejects, the live
        // map is left untouched (safe no-op) and the rejection reason is
        // surfaced on the returned stats' `error`.
        let conditioning_gate_active = config.max_outlier_observation_ratio.is_some()
            || config.max_behind_camera_landmark_ratio.is_some()
            || config.min_fixed_to_optimized_ratio.is_some()
            || config.max_pose_translation_correction_m.is_some()
            || config.max_pose_rotation_correction_rad.is_some();
        let factor_neighbor_allowlist = self
            .sparse_factor_graph_state
            .as_ref()
            .map(|graph| graph.active_neighbor_keyframe_ids(frame.id));
        let result = if conditioning_gate_active {
            let mut candidate_map = self.map.clone();
            let result = refine_visual_map_with_covisibility_ba_and_neighbor_allowlist(
                &mut candidate_map,
                frame.id,
                &config.ba,
                factor_neighbor_allowlist.as_ref(),
            );
            if let Ok(ref result) = result {
                let outlier_ratio = outlier_observation_ratio(
                    result.outlier_observation_count,
                    result.selection.observation_count,
                );
                let make_rejection =
                    |error: CovisibilityLocalBaError, quality_gate_rejected: bool| {
                        let pose_correction_gate_rejected = matches!(
                            error,
                            CovisibilityLocalBaError::PoseCorrectionGateRejected { .. }
                        );
                        OnlineSlamCovisibilityLocalBaStats {
                            active_keyframe_id: frame.id,
                            map_keyframe_count,
                            factor_graph_neighbor_count: factor_neighbor_allowlist
                                .as_ref()
                                .map(|neighbors| neighbors.len()),
                            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                            success: false,
                            error: Some(error),
                            selection: Some(result.selection.clone()),
                            ba_result: Some(result.ba_result.clone()),
                            mean_reprojection_before_px: Some(result.mean_reprojection_before_px),
                            mean_reprojection_after_px: Some(result.mean_reprojection_after_px),
                            max_pose_translation_correction_m: Some(
                                result.max_pose_translation_correction_m,
                            ),
                            max_pose_rotation_correction_rad: Some(
                                result.max_pose_rotation_correction_rad,
                            ),
                            updated_keyframe_count: 0,
                            updated_landmark_count: 0,
                            outlier_observation_count: result.outlier_observation_count,
                            observation_count: result.selection.observation_count,
                            outlier_observation_ratio: outlier_ratio,
                            quality_gate_rejected,
                            pose_correction_gate_rejected,
                            removed_observation_count: 0,
                        }
                    };

                let optimized_keyframe_count = result.selection.optimized_keyframe_ids.len();
                let fixed_keyframe_count = result.selection.fixed_keyframe_ids.len();

                let translation_exceeded = config
                    .max_pose_translation_correction_m
                    .is_some_and(|limit| result.max_pose_translation_correction_m > limit);
                let rotation_exceeded = config
                    .max_pose_rotation_correction_rad
                    .is_some_and(|limit| result.max_pose_rotation_correction_rad > limit);
                if translation_exceeded || rotation_exceeded {
                    return Some(make_rejection(
                        CovisibilityLocalBaError::PoseCorrectionGateRejected {
                            translation_correction_m: result.max_pose_translation_correction_m,
                            rotation_correction_rad: result.max_pose_rotation_correction_rad,
                            max_translation_correction_m: config.max_pose_translation_correction_m,
                            max_rotation_correction_rad: config.max_pose_rotation_correction_rad,
                        },
                        false,
                    ));
                }

                // Fixed-anchor adequacy (ratio form).
                if let Some(min_ratio) = config.min_fixed_to_optimized_ratio {
                    if !fixed_to_optimized_ratio_satisfied(
                        optimized_keyframe_count,
                        fixed_keyframe_count,
                        min_ratio,
                    ) {
                        return Some(make_rejection(
                            CovisibilityLocalBaError::FixedSupportRatioRejected {
                                optimized_keyframe_count,
                                fixed_keyframe_count,
                                required_fixed_keyframes: required_fixed_keyframes(
                                    optimized_keyframe_count,
                                    min_ratio,
                                ),
                                min_fixed_to_optimized_ratio: min_ratio,
                            },
                            false,
                        ));
                    }
                }

                // Behind-camera degeneracy (evaluated on the solved clone).
                if let Some(max_behind) = config.max_behind_camera_landmark_ratio {
                    let behind_ratio =
                        behind_camera_optimized_landmark_ratio(&candidate_map, &result.selection)
                            .unwrap_or(0.0);
                    if behind_ratio > max_behind {
                        return Some(make_rejection(
                            CovisibilityLocalBaError::BehindCameraGateRejected {
                                behind_camera_landmark_ratio: behind_ratio,
                                max_behind_camera_landmark_ratio: max_behind,
                            },
                            false,
                        ));
                    }
                }

                // Outlier-observation ratio (legacy quality gate).
                if let Some(max_ratio) = config.max_outlier_observation_ratio {
                    if outlier_ratio.unwrap_or(0.0) > max_ratio {
                        return Some(make_rejection(
                            CovisibilityLocalBaError::QualityGateRejected {
                                outlier_observation_count: result.outlier_observation_count,
                                observation_count: result.selection.observation_count,
                                max_outlier_observation_ratio: max_ratio,
                            },
                            true,
                        ));
                    }
                }

                self.map = candidate_map;
            }
            result
        } else {
            refine_visual_map_with_covisibility_ba_and_neighbor_allowlist(
                &mut self.map,
                frame.id,
                &config.ba,
                factor_neighbor_allowlist.as_ref(),
            )
        };
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

        match result {
            Ok(result) => {
                let observation_count = result.selection.observation_count;
                Some(OnlineSlamCovisibilityLocalBaStats {
                    active_keyframe_id: frame.id,
                    map_keyframe_count,
                    factor_graph_neighbor_count: factor_neighbor_allowlist
                        .as_ref()
                        .map(|neighbors| neighbors.len()),
                    elapsed_ms,
                    success: true,
                    error: None,
                    selection: Some(result.selection),
                    ba_result: Some(result.ba_result),
                    mean_reprojection_before_px: Some(result.mean_reprojection_before_px),
                    mean_reprojection_after_px: Some(result.mean_reprojection_after_px),
                    max_pose_translation_correction_m: Some(
                        result.max_pose_translation_correction_m,
                    ),
                    max_pose_rotation_correction_rad: Some(result.max_pose_rotation_correction_rad),
                    updated_keyframe_count: result.updated_keyframe_count,
                    updated_landmark_count: result.updated_landmark_count,
                    outlier_observation_count: result.outlier_observation_count,
                    observation_count,
                    outlier_observation_ratio: outlier_observation_ratio(
                        result.outlier_observation_count,
                        observation_count,
                    ),
                    quality_gate_rejected: false,
                    pose_correction_gate_rejected: false,
                    removed_observation_count: result.removed_observation_count,
                })
            }
            Err(error) => Some(OnlineSlamCovisibilityLocalBaStats {
                active_keyframe_id: frame.id,
                map_keyframe_count,
                factor_graph_neighbor_count: factor_neighbor_allowlist
                    .as_ref()
                    .map(|neighbors| neighbors.len()),
                elapsed_ms,
                success: false,
                error: Some(error),
                selection: None,
                ba_result: None,
                mean_reprojection_before_px: None,
                mean_reprojection_after_px: None,
                max_pose_translation_correction_m: None,
                max_pose_rotation_correction_rad: None,
                updated_keyframe_count: 0,
                updated_landmark_count: 0,
                outlier_observation_count: 0,
                observation_count: 0,
                outlier_observation_ratio: None,
                quality_gate_rejected: false,
                pose_correction_gate_rejected: false,
                removed_observation_count: 0,
            }),
        }
    }

    fn stage_imu_factor_on_new_keyframe(
        &mut self,
        frame: &Frame,
        applied_update: Option<&AppliedMapUpdate>,
    ) -> Option<ImuPreintegrationFactor> {
        let state = self.imu_state.as_mut()?;
        let added_new_keyframe = applied_update
            .map(|a| a.keyframe_count > 0)
            .unwrap_or(false);
        if !added_new_keyframe {
            return None;
        }
        let new_keyframe_id = frame.id;
        let factor = match state.last_keyframe_id {
            Some(prev_id) if prev_id != new_keyframe_id => {
                let delta = state.preintegrator.delta();
                let factor = ImuPreintegrationFactor {
                    keyframe_id_from: prev_id,
                    keyframe_id_to: new_keyframe_id,
                    delta,
                    gravity_world: state.config.gravity_world,
                    weight_position: state.config.weight_position,
                    weight_velocity: state.config.weight_velocity,
                    weight_rotation: state.config.weight_rotation,
                };
                state.preintegrator.reset();
                Some(factor)
            }
            _ => None,
        };
        state.last_keyframe_id = Some(new_keyframe_id);

        // Stale-factor gate. Until the auto-bootstrap stage has fired
        // (or has given up with `KeepExistingSeed`), factors carry the
        // caller's placeholder bias linearisation and are discarded
        // rather than exposed downstream. `discarded_stale_factor_count`
        // is reported on the `Succeeded` event so callers can audit.
        // Opt-in `OnlineSlamConfig.keep_pre_promotion_imu_factors`
        // bypasses the discard so factors still flow into the local
        // BA's factor history (carrying placeholder biases; the BA's
        // own iterations are expected to absorb the linearisation
        // error). `discarded_stale_factor_count` is still incremented
        // in that branch so auditors can see how many factors were
        // forwarded with stale biases.
        if let Some(vi) = self.vi_init_state.as_mut() {
            if vi.is_active() {
                if factor.is_some() {
                    vi.discarded_stale_factor_count += 1;
                }
                if !self.config.keep_pre_promotion_imu_factors {
                    state.pending_factor = None;
                    return None;
                }
            }
        }

        state.pending_factor = factor.clone();
        factor
    }

    /// Run a single try-initialise step on the VI init buffer when a new
    /// keyframe has just been registered. Returns the state-transition
    /// event for [`OnlineSlamResult::vi_init`]; `None` when nothing
    /// changed this frame (no new keyframe, already terminal, or VI init
    /// disabled).
    fn run_vi_init_step(
        &mut self,
        frame: &Frame,
        applied_update: Option<&AppliedMapUpdate>,
    ) -> Option<ViInitializationEvent> {
        // Cheap pre-checks that don't need to borrow `vi_init_state`
        // mutably yet.
        if self
            .vi_init_state
            .as_ref()
            .map(|s| !s.is_active())
            .unwrap_or(true)
        {
            return None;
        }
        let added_new_keyframe = applied_update
            .map(|a| a.keyframe_count > 0)
            .unwrap_or(false);
        let try_on_every_frame = self
            .vi_init_state
            .as_ref()
            .map(|s| s.config.try_initialize_on_every_frame)
            .unwrap_or(false);
        if !added_new_keyframe && !try_on_every_frame {
            return None;
        }

        // Try the standalone initialiser. We don't need to mutate
        // `vi_init_state` for this read, just borrow it.
        let try_result = {
            let state = self.vi_init_state.as_ref().unwrap();
            state.initializer.try_initialize()
        };

        match try_result {
            Ok(result) => {
                // Bind promotion to the just-registered keyframe if there
                // is one this frame; otherwise (`try_on_every_frame` path
                // on a non-KF frame) bind to the most recent existing
                // keyframe so the rotation rewrite + local-VI-BA seed
                // still target a valid map entry. If the map has no
                // keyframes yet, promote anchorlessly — Step 3 + 4 in
                // `promote_vi_init_result` will skip when the binding is
                // `None`.
                let binding_keyframe_id = if added_new_keyframe {
                    Some(frame.id)
                } else {
                    self.map.keyframes.keys().copied().max()
                };
                Some(self.promote_vi_init_result(binding_keyframe_id, result))
            }
            Err(reason) => {
                // Record the rejection and check whether we've exceeded
                // a cap; either fall through with `StillBuffering` or
                // trigger the configured fallback.
                let vi = self.vi_init_state.as_mut().unwrap();
                vi.last_rejection = Some(reason.clone());
                if vi.cap_exceeded() {
                    Some(self.apply_vi_init_fallback(reason))
                } else {
                    Some(ViInitializationEvent::StillBuffering { reason })
                }
            }
        }
    }

    /// Atomic promotion of a successful VI init result into the
    /// pipeline's running state. The ordering follows the design
    /// contract (see `docs/vi_initialization_integration.md`):
    ///   1. Reset `imu_state.preintegrator` with the new bias linearisation.
    ///   2. Mirror `imu_state.config.{bias_gyro, bias_acc}`.
    ///   3. Rewrite the just-registered keyframe's `Pose` (rotation
    ///      only, camera centre preserved) if requested.
    ///   4. Seed `local_vi_ba_state.keyframe_state[first_keyframe_id]`
    ///      with `velocity_world = 0, bias = (b_g, b_a)`.
    ///   5. Mark `vi_init_state.completed` so the stale-factor gate is
    ///      lifted from this frame onwards.
    fn promote_vi_init_result(
        &mut self,
        binding_keyframe_id: Option<u64>,
        result: VisualInertialInitializationResult,
    ) -> ViInitializationEvent {
        let bias_gyro = result.bias_gyro;
        let bias_acc = result.bias_acc;
        let seed_rotation = result.initial_rotation_body_to_world;

        // Step 1 + 2: reset preintegrator and mirror config biases.
        if let Some(imu) = self.imu_state.as_mut() {
            imu.reset_preintegrator_with_bias(bias_gyro, bias_acc);
        }
        if let Some(imu_cfg) = self.config.imu.as_mut() {
            imu_cfg.bias_gyro = bias_gyro;
            imu_cfg.bias_acc = bias_acc;
        }

        // Step 3: rewrite first keyframe rotation, preserving the
        // camera centre. `R_w←c = R_w←b · R_b←c`, then
        // `Pose.rotation = R_c←w = R_w←c^T`, and
        // `Pose.translation = -R_c←w · C_w_old`. Skipped when promotion
        // happened on a non-KF frame and the map has no keyframe yet to
        // bind to (Phase-19 `try_initialize_on_every_frame` path).
        let seed_first_keyframe_rotation = self
            .vi_init_state
            .as_ref()
            .map(|s| s.config.seed_first_keyframe_rotation)
            .unwrap_or(false);
        let body_to_camera = self
            .vi_init_state
            .as_ref()
            .map(|s| s.config.body_to_camera.clone())
            .unwrap_or_else(SE3::identity);
        if seed_first_keyframe_rotation {
            if let Some(first_keyframe_id) = binding_keyframe_id {
                if let Some(keyframe) = self.map.keyframes.get_mut(&first_keyframe_id) {
                    if let Some(pose) = keyframe.frame.pose.as_mut() {
                        let r_wb = seed_rotation;
                        let r_bc = body_to_camera.rotation;
                        let r_wc = r_wb * r_bc;
                        let r_cw_new = r_wc.inverse();
                        let camera_center_world = -(pose.world_to_camera.rotation.inverse()
                            * pose.world_to_camera.translation);
                        let t_cw_new = -(r_cw_new * camera_center_world);
                        *pose =
                            visloc_core::geometry::Pose::from_world_to_camera(r_cw_new, t_cw_new);
                    }
                }
            }
        }

        // Step 4: seed first-keyframe velocity slot in local VI-BA.
        if let Some(local) = self.local_vi_ba_state.as_mut() {
            if let Some(first_keyframe_id) = binding_keyframe_id {
                local
                    .keyframe_state
                    .entry(first_keyframe_id)
                    .or_insert_with(|| crate::online_slam_vi_ba::KeyframeImuState {
                        velocity_world: Vector3::zeros(),
                        bias_gyro,
                        bias_acc,
                    });
            }
            // Also mirror the new bias linearisation onto the local
            // VI-BA config so subsequent keyframe slots inherit it.
            local.config.bias_gyro_init = bias_gyro;
            local.config.bias_acc_init = bias_acc;
        }
        if let Some(local_cfg) = self.config.local_vi_ba.as_mut() {
            local_cfg.bias_gyro_init = bias_gyro;
            local_cfg.bias_acc_init = bias_acc;
        }

        // Step 5: mark the stage completed.
        let discarded_stale_factor_count;
        {
            let vi = self.vi_init_state.as_mut().unwrap();
            vi.completed = Some(result.clone());
            discarded_stale_factor_count = vi.discarded_stale_factor_count;
        }

        ViInitializationEvent::Succeeded {
            result,
            first_keyframe_id: binding_keyframe_id,
            discarded_stale_factor_count,
        }
    }

    /// Apply the configured fallback after the duration / sample cap
    /// has been exceeded without a successful `try_initialize`.
    fn apply_vi_init_fallback(
        &mut self,
        last_reason: StationaryRejectionReason,
    ) -> ViInitializationEvent {
        let fallback = self
            .vi_init_state
            .as_ref()
            .map(|s| s.config.on_persistent_rejection)
            .unwrap_or(ViInitFallback::KeepExistingSeed);
        match fallback {
            ViInitFallback::KeepExistingSeed => {
                // Stale gate is lifted by clearing `vi_init_state.gave_up`
                // -> not active anymore.
            }
            ViInitFallback::DisableImuStage => {
                self.imu_state = None;
                self.local_vi_ba_state = None;
                self.config.imu = None;
                self.config.local_vi_ba = None;
            }
        }
        if let Some(vi) = self.vi_init_state.as_mut() {
            vi.gave_up = Some(last_reason.clone());
        }
        ViInitializationEvent::GaveUp {
            last_reason,
            fallback,
        }
    }

    /// Run one motion-based VI init step on the current frame. The
    /// stage:
    ///
    /// 1. Banks the freshly-staged IMU factor (if any) onto
    ///    `vi_motion_init_state.factor_history` so the trigger has a
    ///    body of factors to optimise against.
    /// 2. Registers the new keyframe's world-frame camera centre with
    ///    the inner `MotionBasedViInitializer`.
    /// 3. Calls `try_initialize` once the inner gates allow it. On
    ///    success the refined `(velocity, bias)` are mirrored into
    ///    `local_vi_ba_state.keyframe_state` and `imu_state`'s
    ///    pre-integrator is reset with the new bias linearisation.
    ///
    /// Returns the state-transition event for
    /// [`OnlineSlamResult::vi_motion_init`]. `None` when:
    /// * no new keyframe was registered this frame,
    /// * the motion-based stage is disabled,
    /// * the static VI init stage has neither succeeded nor reached the
    ///   explicitly-enabled post-give-up fallback,
    /// * or the stage has already reached the terminal `Initialised`
    ///   state.
    fn run_motion_vi_init_step(
        &mut self,
        frame: &Frame,
        applied_update: Option<&AppliedMapUpdate>,
        new_imu_factor: Option<&ImuPreintegrationFactor>,
    ) -> Option<MotionViInitializationEvent> {
        let added_new_keyframe = applied_update
            .map(|a| a.keyframe_count > 0)
            .unwrap_or(false);
        if !added_new_keyframe {
            return None;
        }
        if self
            .vi_motion_init_state
            .as_ref()
            .map(|s| !s.is_active())
            .unwrap_or(true)
        {
            return None;
        }
        // Prefer stationary-window bias estimates. For a sequence that starts
        // in motion, an explicit opt-in may continue only after the static
        // stage has terminally given up; use the still-live IMU configuration's
        // calibrated/existing biases rather than fabricating a stationary
        // result from a low-variance moving window.
        let static_bias_seed = self
            .vi_init_state
            .as_ref()
            .and_then(|state| state.completed.as_ref())
            .map(|seed| (seed.bias_gyro, seed.bias_acc));
        let allow_after_give_up = self
            .vi_motion_init_state
            .as_ref()
            .map(|state| state.config.allow_after_static_give_up)
            .unwrap_or(false);
        let allow_from_configured_bias_before_static = self
            .vi_motion_init_state
            .as_ref()
            .map(|state| state.config.allow_from_configured_bias_before_static)
            .unwrap_or(false);
        let static_gave_up = self
            .vi_init_state
            .as_ref()
            .is_some_and(|state| state.gave_up.is_some());
        let (bias_gyro_seed, bias_acc_seed) = if let Some(seed) = static_bias_seed {
            seed
        } else if allow_from_configured_bias_before_static
            || (allow_after_give_up && static_gave_up)
        {
            let imu = self.imu_state.as_ref()?;
            (imu.config.bias_gyro, imu.config.bias_acc)
        } else {
            return None;
        };

        // Bank the new factor + register the keyframe centre.
        let camera_center = self
            .map
            .keyframes
            .get(&frame.id)
            .and_then(|kf| kf.frame.pose.as_ref().map(|p| p.camera_center_world()))?;
        {
            let state = self.vi_motion_init_state.as_mut().unwrap();
            if let Some(factor) = new_imu_factor {
                state.push_factor(factor.clone());
            }
            state.initializer.register_keyframe(frame.id, camera_center);
        }

        // Attempt the solve unconditionally. `try_initialize` itself
        // performs the keyframe-count + translation gate checks and
        // returns a structured rejection on miss without touching the
        // map. Snapshot the factor history first so we don't hold a
        // borrow into `vi_motion_init_state` while passing `&mut self.map`.
        let factors_snapshot: Vec<ImuPreintegrationFactor> = self
            .vi_motion_init_state
            .as_ref()
            .map(|s| s.factor_history.iter().cloned().collect())
            .unwrap_or_default();
        let outcome = {
            let state = self.vi_motion_init_state.as_mut().unwrap();
            match state.initializer.try_initialize_with_bias_seed(
                &mut self.map,
                &factors_snapshot,
                bias_gyro_seed,
                bias_acc_seed,
            ) {
                Ok(r) => Ok(r.clone()),
                Err(reason) => {
                    state.last_rejection = Some(reason.clone());
                    Err(reason)
                }
            }
        };
        match outcome {
            Ok(result) => Some(self.promote_motion_vi_init_result(result)),
            Err(reason) => Some(MotionViInitializationEvent::StillWaiting { reason }),
        }
    }

    /// Mirror a successful motion-VI init result into the pipeline's
    /// running state.
    ///
    /// `result.bias_released` distinguishes a terminal firing (the legacy
    /// single-stage path, or a [`crate::BiasReleaseSchedule`] Stage B) from a
    /// non-terminal [`crate::BiasReleaseSchedule`] Stage A ("velocity
    /// stage") firing: the velocity + IMU-bias mirroring below (steps 1-2)
    /// always runs — Stage A's biases equal the seed, so mirroring them is
    /// value-neutral, while mirroring its refined velocities is the entire
    /// point of running Stage A — but step 3 only marks the stage
    /// `completed` (terminal) when `bias_released` is `true`. On a `false`
    /// result the stage stays active so [`Self::run_motion_vi_init_step`]
    /// keeps registering keyframes / translation and can later fire Stage B.
    fn promote_motion_vi_init_result(
        &mut self,
        result: MotionBasedViInitializationResult,
    ) -> MotionViInitializationEvent {
        let mirror_local = self
            .vi_motion_init_state
            .as_ref()
            .map(|s| s.config.mirror_into_local_vi_ba)
            .unwrap_or(true);
        let mirror_imu = self
            .vi_motion_init_state
            .as_ref()
            .map(|s| s.config.mirror_into_imu_state)
            .unwrap_or(true);

        // Step 1: mirror refined per-keyframe states into local VI-BA.
        if mirror_local {
            if let Some(local) = self.local_vi_ba_state.as_mut() {
                for (kf_id, refined) in &result.keyframe_states {
                    local.keyframe_state.insert(*kf_id, refined.clone());
                }
            }
        }

        // Step 2: mirror refined biases of the latest keyframe onto the
        // running IMU pre-integrator and config. ORB-SLAM3 uses the most
        // recent VIBA1-refined bias as the running linearisation point
        // for the next pre-integration window.
        if mirror_imu {
            if let Some((_, latest)) = result.keyframe_states.iter().next_back() {
                let bg = latest.bias_gyro;
                let ba = latest.bias_acc;
                if let Some(imu) = self.imu_state.as_mut() {
                    imu.reset_preintegrator_with_bias(bg, ba);
                }
                if let Some(imu_cfg) = self.config.imu.as_mut() {
                    imu_cfg.bias_gyro = bg;
                    imu_cfg.bias_acc = ba;
                }
                if let Some(local) = self.local_vi_ba_state.as_mut() {
                    local.config.bias_gyro_init = bg;
                    local.config.bias_acc_init = ba;
                }
                if let Some(local_cfg) = self.config.local_vi_ba.as_mut() {
                    local_cfg.bias_gyro_init = bg;
                    local_cfg.bias_acc_init = ba;
                }
            }
        }

        // Step 3: mirror `estimate_gravity`'s recovered gravity vector (if
        // any) into every sink that reads gravity for FUTURE factor
        // staging / VI solves. The map itself is NEVER rotated to match —
        // only the gravity ASSUMPTION used by future solves moves. See
        // `docs/motion_based_vi_alignment.md`'s "Gravity-direction
        // recovery" section. Mirrored sinks:
        // (1) `imu_state.config.gravity_world` — the running IMU stage
        //     that stamps `gravity_world` onto every newly-staged
        //     `ImuPreintegrationFactor` (`stage_imu_factor_on_new_keyframe`);
        // (2) `config.imu.gravity_world` — the persisted config mirror of (1);
        // (3) `local_vi_ba_state.config.gravity_world` and (4)
        //     `config.local_vi_ba.gravity_world` — the local VI-BA stage's
        //     own gravity copy (seeds bias slot initial values);
        // (5) `vi_motion_init_state.initializer`'s own config, via
        //     [`MotionBasedViInitializer::set_gravity_world`], so a LATER
        //     motion-VI window (a subsequent `BiasReleaseSchedule` Stage B,
        //     or a fresh sequence after `reset_sequence_state`) starts from
        //     the corrected assumption instead of the original config
        //     value;
        // (6) `config.vi_motion_init.initializer.gravity_world`, the
        //     persisted mirror of (5); and
        // (7) `config.vi_init.initializer.gravity_world` — the STATIC
        //     stage's persisted config. [`OnlineSlamConfig::validate`]
        //     requires this to agree with `config.imu.gravity_world`
        //     (`GravityMismatch`), so leaving it stale would make a
        //     later `OnlineSlamPipeline::new(_, _, _, config.clone())`
        //     panic (this is exactly how a real rebootstrap that
        //     reconstructs the pipeline from `slam.config.clone()`
        //     surfaced the bug). The live `vi_init_state.initializer` is
        //     deliberately left untouched: by the time motion-VI can
        //     fire the static stage has already reached a terminal
        //     `completed`/`gave_up` snapshot (see
        //     [`OnlineSlamConfigError::MotionViInitRequiresStaticViInit`]
        //     and its sibling checks), so it never runs `try_initialize`
        //     again in THIS pipeline instance; restarting it with the
        //     estimated gravity on a fresh submap (a NEW pipeline built
        //     from this config) is semantically fine — it will
        //     re-estimate/reject as usual.
        if let Some(g) = result.estimated_gravity_world {
            if let Some(imu) = self.imu_state.as_mut() {
                imu.config.gravity_world = g;
            }
            if let Some(imu_cfg) = self.config.imu.as_mut() {
                imu_cfg.gravity_world = g;
            }
            if let Some(local) = self.local_vi_ba_state.as_mut() {
                local.config.gravity_world = g;
            }
            if let Some(local_cfg) = self.config.local_vi_ba.as_mut() {
                local_cfg.gravity_world = g;
            }
            if let Some(state) = self.vi_motion_init_state.as_mut() {
                state.initializer.set_gravity_world(g);
            }
            if let Some(motion_cfg) = self.config.vi_motion_init.as_mut() {
                motion_cfg.initializer.gravity_world = g;
            }
            if let Some(vi_init_cfg) = self.config.vi_init.as_mut() {
                vi_init_cfg.initializer.gravity_world = g;
            }
        }

        // Step 4: mark the stage completed — but ONLY on a terminal firing
        // (`result.bias_released`). A `BiasReleaseSchedule` Stage A result
        // leaves `state.completed` at `None` so `is_active()` stays `true`
        // and the next `run_motion_vi_init_step` call keeps feeding this
        // stage new keyframes/factors toward the eventual Stage B release.
        // (`initializer.try_initialize` mirrors the same non-terminal
        // behaviour on its own `completed` slot for a Stage A result.)
        if let Some(state) = self.vi_motion_init_state.as_mut() {
            if result.bias_released {
                state.completed = Some(result.clone());
            }
            state.last_rejection = None;
        }

        MotionViInitializationEvent::Succeeded { result }
    }

    pub fn reset_sequence_state(&mut self) {
        self.tracker.reset();
        self.mapper.reset();
        if let Some(state) = self.imu_state.as_mut() {
            state.reset();
        }
        if let Some(state) = self.local_vi_ba_state.as_mut() {
            state.reset();
        }
        if let Some(state) = self.vi_init_state.as_mut() {
            state.reset();
        }
        if let Some(state) = self.vi_motion_init_state.as_mut() {
            state.reset();
        }
        if let Some(state) = self.pose_graph_state.as_mut() {
            state.reset();
        }
        if let Some(state) = self.relocalization_state.as_mut() {
            state.reset();
        }
    }
}

fn motion_vi_raw_residual_activation_satisfied(
    config: MotionViRawResidualActivationConfig,
    reason: &MotionBasedViRejectionReason,
) -> bool {
    let MotionBasedViRejectionReason::ImuRawResidualOutOfRange {
        rotation_residual_rms_rad: Some(rotation),
        velocity_residual_rms_mps: Some(velocity),
        position_residual_rms_meters: Some(position),
        ..
    } = reason
    else {
        return false;
    };
    config.max_rotation_residual_rms_rad.is_finite()
        && config.max_rotation_residual_rms_rad > 0.0
        && config.max_velocity_residual_rms_mps.is_finite()
        && config.max_velocity_residual_rms_mps > 0.0
        && config.max_position_residual_rms_meters.is_finite()
        && config.max_position_residual_rms_meters > 0.0
        && rotation.is_finite()
        && velocity.is_finite()
        && position.is_finite()
        && *rotation <= config.max_rotation_residual_rms_rad
        && *velocity <= config.max_velocity_residual_rms_mps
        && *position <= config.max_position_residual_rms_meters
}

#[cfg(test)]
mod motion_vi_raw_residual_activation_tests {
    use super::*;

    fn raw_rejection(rotation: f64, velocity: f64, position: f64) -> MotionBasedViRejectionReason {
        MotionBasedViRejectionReason::ImuRawResidualOutOfRange {
            rotation_residual_rms_rad: Some(rotation),
            velocity_residual_rms_mps: Some(velocity),
            position_residual_rms_meters: Some(position),
            max_rotation_residual_rms_rad: Some(0.01),
            max_velocity_residual_rms_mps: Some(0.25),
            max_position_residual_rms_meters: Some(0.08),
        }
    }

    #[test]
    fn activates_only_for_complete_near_conditioned_raw_residuals() {
        let config = MotionViRawResidualActivationConfig {
            max_rotation_residual_rms_rad: 0.03,
            max_velocity_residual_rms_mps: 0.6,
            max_position_residual_rms_meters: 0.15,
        };
        assert!(motion_vi_raw_residual_activation_satisfied(
            config,
            &raw_rejection(0.02, 0.46, 0.12)
        ));
        assert!(!motion_vi_raw_residual_activation_satisfied(
            config,
            &raw_rejection(0.038, 0.46, 0.12)
        ));
        assert!(!motion_vi_raw_residual_activation_satisfied(
            config,
            &MotionBasedViRejectionReason::InsufficientKeyframes { have: 5, need: 10 }
        ));
    }
}

/// Unit tests for [`OnlineSlamPipeline::promote_motion_vi_init_result`]'s
/// `bias_released`-conditional Step 3, exercised directly (rather than via
/// `process_frame`) so a synthetic
/// [`crate::MotionBasedViInitializationResult`] can pin the Stage A / Stage B
/// contract without standing up a full tracker + IMU + static-VI fixture.
/// See `docs/motion_based_vi_alignment.md`'s "Staged bias release" section.
#[cfg(test)]
mod bias_release_promotion_tests {
    use super::*;

    fn zero_cost_breakdown() -> crate::bundle::BaCostBreakdown {
        crate::bundle::BaCostBreakdown {
            total: 0.0,
            visual: 0.0,
            imu: 0.0,
            bias_random_walk: 0.0,
            navigation_prior: 0.0,
            other_structural: 0.0,
            imu_normalized_squared_residual_per_dof: None,
            imu_rotation_residual_rms_rad: None,
            imu_velocity_residual_rms_mps: None,
            imu_position_residual_rms_meters: None,
        }
    }

    /// Build a synthetic single-keyframe motion-VI result. Biases are
    /// always zero here (Stage A never moves them off the seed, and this
    /// helper reuses the same shape for both stages) — only `velocity` and
    /// `bias_released` vary between calls.
    fn fake_result(
        kf_id: u64,
        velocity: Vector3<f64>,
        bias_released: bool,
    ) -> MotionBasedViInitializationResult {
        let mut keyframe_states = BTreeMap::new();
        keyframe_states.insert(
            kf_id,
            KeyframeImuState {
                velocity_world: velocity,
                bias_gyro: Vector3::zeros(),
                bias_acc: Vector3::zeros(),
            },
        );
        MotionBasedViInitializationResult {
            keyframe_states,
            keyframe_ids: vec![kf_id],
            imu_factors_used: 1,
            scale: 1.0,
            scale_history: Vec::new(),
            viba2_iterations_run: 0,
            trigger_translation_meters: 2.0,
            ba_result: BaResult {
                initial_cost: 1.0,
                final_cost: 0.5,
                iterations: Vec::new(),
                converged: true,
            },
            initial_cost_breakdown: zero_cost_breakdown(),
            final_cost_breakdown: zero_cost_breakdown(),
            bias_released,
            estimated_gravity_world: None,
            estimated_gyro_bias: None,
        }
    }

    fn minimal_pipeline_with_motion_vi_state(
    ) -> OnlineSlamPipeline<Tracker<LocalizationPipeline>, LocalMappingPipeline> {
        let mut slam = OnlineSlamPipeline::new(
            VisualMap::new(),
            Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            LocalMappingPipeline::default(),
            OnlineSlamConfig::default(),
        );
        slam.local_vi_ba_state = Some(OnlineSlamLocalBaState::new(
            OnlineSlamLocalBaConfig::default(),
        ));
        slam.vi_motion_init_state = Some(OnlineSlamMotionViInitState::new(
            OnlineSlamMotionViInitConfig::default(),
        ));
        slam
    }

    #[test]
    fn stage_a_mirrors_velocity_but_leaves_stage_active_then_stage_b_completes_it() {
        let mut slam = minimal_pipeline_with_motion_vi_state();

        // Stage A ("velocity stage"): `bias_released = false`.
        let stage_a = fake_result(1, Vector3::new(1.0, 0.0, 0.0), false);
        match slam.promote_motion_vi_init_result(stage_a) {
            MotionViInitializationEvent::Succeeded { result } => assert!(!result.bias_released),
            other => panic!("expected Succeeded, got {other:?}"),
        }
        let mirrored_after_stage_a = slam
            .local_vi_ba_state
            .as_ref()
            .expect("configured above")
            .keyframe_state
            .get(&1)
            .expect("Stage A must mirror velocity into local_vi_ba_state")
            .velocity_world;
        assert!((mirrored_after_stage_a - Vector3::new(1.0, 0.0, 0.0)).norm() < 1.0e-12);
        let motion_state = slam
            .vi_motion_init_state
            .as_ref()
            .expect("configured above");
        assert!(
            motion_state.completed.is_none(),
            "Stage A must NOT mark the motion-VI stage completed"
        );
        assert!(
            motion_state.is_active(),
            "Stage A must leave the motion-VI stage active"
        );

        // Stage B ("bias release"): `bias_released = true`, a different
        // velocity so the mirror can be told apart from Stage A's.
        let stage_b = fake_result(1, Vector3::new(2.0, 0.0, 0.0), true);
        match slam.promote_motion_vi_init_result(stage_b) {
            MotionViInitializationEvent::Succeeded { result } => assert!(result.bias_released),
            other => panic!("expected Succeeded, got {other:?}"),
        }
        let mirrored_after_stage_b = slam
            .local_vi_ba_state
            .as_ref()
            .expect("configured above")
            .keyframe_state
            .get(&1)
            .expect("Stage B must mirror velocity into local_vi_ba_state")
            .velocity_world;
        assert!((mirrored_after_stage_b - Vector3::new(2.0, 0.0, 0.0)).norm() < 1.0e-12);
        let motion_state = slam
            .vi_motion_init_state
            .as_ref()
            .expect("configured above");
        assert!(
            motion_state.completed.is_some(),
            "Stage B must mark the motion-VI stage completed (terminal)"
        );
        assert!(
            !motion_state.is_active(),
            "Stage B must leave the motion-VI stage inactive (terminal)"
        );
    }

    /// [`OnlineSlamPipeline::promote_motion_vi_init_result`]'s Step 3
    /// (gravity-sink mirroring, see that method's doc comment for the full
    /// enumerated list). Populates every sink the doc comment names —
    /// `imu_state`, `config.imu`, `local_vi_ba_state`, `config.local_vi_ba`,
    /// the motion initializer's own config, `config.vi_motion_init`, and
    /// `config.vi_init` — then asserts a `Some(estimated_gravity_world)`
    /// result moves every one of them, and the map/keyframe poses are
    /// untouched (gauge is never rotated). See
    /// `docs/motion_based_vi_alignment.md`'s "Gravity-direction recovery"
    /// section.
    #[test]
    fn estimated_gravity_mirrors_into_every_documented_sink() {
        let mut slam = minimal_pipeline_with_motion_vi_state();
        slam.imu_state = Some(OnlineSlamImuState::new(OnlineSlamImuConfig::default()));
        slam.config.imu = Some(OnlineSlamImuConfig::default());
        slam.config.local_vi_ba = Some(OnlineSlamLocalBaConfig::default());
        slam.config.vi_motion_init = Some(OnlineSlamMotionViInitConfig::default());
        slam.config.vi_init = Some(OnlineSlamViInitConfig::default());

        let original_gravity = OnlineSlamImuConfig::default().gravity_world;
        let new_gravity = Vector3::new(0.0, 0.0, 9.81);
        assert!(
            (original_gravity - new_gravity).norm() > 1.0,
            "fixture must actually move the gravity assumption"
        );

        let mut result = fake_result(1, Vector3::zeros(), true);
        result.estimated_gravity_world = Some(new_gravity);
        match slam.promote_motion_vi_init_result(result) {
            MotionViInitializationEvent::Succeeded { result } => {
                assert_eq!(result.estimated_gravity_world, Some(new_gravity));
            }
            other => panic!("expected Succeeded, got {other:?}"),
        }

        assert!(
            (slam.imu_state.as_ref().unwrap().config.gravity_world - new_gravity).norm() < 1.0e-12,
            "sink (1): imu_state.config.gravity_world must mirror the estimate"
        );
        assert!(
            (slam.config.imu.as_ref().unwrap().gravity_world - new_gravity).norm() < 1.0e-12,
            "sink (2): config.imu.gravity_world must mirror the estimate"
        );
        assert!(
            (slam
                .local_vi_ba_state
                .as_ref()
                .unwrap()
                .config
                .gravity_world
                - new_gravity)
                .norm()
                < 1.0e-12,
            "sink (3): local_vi_ba_state.config.gravity_world must mirror the estimate"
        );
        assert!(
            (slam.config.local_vi_ba.as_ref().unwrap().gravity_world - new_gravity).norm()
                < 1.0e-12,
            "sink (4): config.local_vi_ba.gravity_world must mirror the estimate"
        );
        assert!(
            (slam
                .vi_motion_init_state
                .as_ref()
                .unwrap()
                .initializer
                .config()
                .gravity_world
                - new_gravity)
                .norm()
                < 1.0e-12,
            "sink (5): the motion initializer's own config must mirror the estimate"
        );
        assert!(
            (slam
                .config
                .vi_motion_init
                .as_ref()
                .unwrap()
                .initializer
                .gravity_world
                - new_gravity)
                .norm()
                < 1.0e-12,
            "sink (6): config.vi_motion_init.initializer.gravity_world must mirror the estimate"
        );
        assert!(
            (slam
                .config
                .vi_init
                .as_ref()
                .unwrap()
                .initializer
                .gravity_world
                - new_gravity)
                .norm()
                < 1.0e-12,
            "sink (7): config.vi_init.initializer.gravity_world must mirror the estimate"
        );
    }

    /// Real-data regression (see the `sp_full75` moving-start bench run):
    /// the motion-VI initializer promoted a Stage A result with a non-
    /// default `estimated_gravity_world` on frame 4. `promote_motion_vi_init_result`
    /// mirrored the estimate into `config.imu.gravity_world` and
    /// `config.vi_motion_init.initializer.gravity_world` but NOT
    /// `config.vi_init.initializer.gravity_world`. A later rebootstrap that
    /// reconstructs the pipeline via
    /// `OnlineSlamPipeline::new(map, tracker, mapper, slam.config.clone())`
    /// (the independent-submap path in
    /// `examples/euroc_online_slam_vi_image_demo.rs`) then hit
    /// `OnlineSlamConfig::validate`'s `GravityMismatch` panic, because
    /// `config.vi_init.initializer.gravity_world` still held the original
    /// seed while `config.imu.gravity_world` held the estimate. This test
    /// builds a real (non-default) config via the public constructor,
    /// drives a synthetic Stage-B result with a non-default estimated
    /// gravity through `promote_motion_vi_init_result`, then reconstructs
    /// the pipeline from the mutated config's clone — this must NOT panic,
    /// and every gravity-agreement invariant `validate` checks must hold.
    #[test]
    fn gravity_mirror_keeps_config_reconstructible_after_rebootstrap() {
        // `OnlineSlamImuConfig::default().gravity_world` is `(0, 9.81, 0)`,
        // not the `(0, 0, -9.81)` that `VisualInertialInitializerConfig`
        // and `MotionBasedViInitializerConfig` default to — so build a
        // config where every gravity-bearing field starts in agreement
        // (required for `OnlineSlamPipeline::new` to succeed at all).
        let seed_gravity = OnlineSlamImuConfig::default().gravity_world;

        let vi_init_config = OnlineSlamViInitConfig {
            initializer: VisualInertialInitializerConfig {
                gravity_world: seed_gravity,
                ..VisualInertialInitializerConfig::default()
            },
            ..OnlineSlamViInitConfig::default()
        };

        let motion_config = OnlineSlamMotionViInitConfig {
            initializer: MotionBasedViInitializerConfig {
                gravity_world: seed_gravity,
                ..MotionBasedViInitializerConfig::default()
            },
            ..OnlineSlamMotionViInitConfig::default()
        };

        let config = OnlineSlamConfig {
            imu: Some(OnlineSlamImuConfig::default()),
            local_vi_ba: Some(OnlineSlamLocalBaConfig::default()),
            vi_init: Some(vi_init_config),
            vi_motion_init: Some(motion_config),
            ..OnlineSlamConfig::default()
        };

        // Sanity check: the seed config must actually be constructible
        // before we exercise the mutation under test.
        let mut slam = OnlineSlamPipeline::new(
            VisualMap::new(),
            Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            LocalMappingPipeline::default(),
            config,
        );

        let estimated_gravity = Vector3::new(0.1678, 9.3632, 2.9222);
        assert!(
            (estimated_gravity - seed_gravity).norm() > 1.0,
            "fixture must actually move the gravity assumption"
        );
        let mut result = fake_result(1, Vector3::zeros(), true);
        result.estimated_gravity_world = Some(estimated_gravity);
        match slam.promote_motion_vi_init_result(result) {
            MotionViInitializationEvent::Succeeded { result } => {
                assert_eq!(result.estimated_gravity_world, Some(estimated_gravity));
            }
            other => panic!("expected Succeeded, got {other:?}"),
        }

        // Every gravity-agreement invariant `OnlineSlamConfig::validate`
        // checks must hold on the mutated config.
        assert_eq!(
            slam.config.imu.as_ref().unwrap().gravity_world,
            estimated_gravity
        );
        assert_eq!(
            slam.config
                .vi_init
                .as_ref()
                .unwrap()
                .initializer
                .gravity_world,
            estimated_gravity
        );
        assert_eq!(
            slam.config
                .vi_motion_init
                .as_ref()
                .unwrap()
                .initializer
                .gravity_world,
            estimated_gravity
        );
        assert!(slam.config.validate().is_ok());

        // The real crash: a rebootstrap reconstructs the pipeline from
        // `slam.config.clone()`. This must not panic.
        let _rebuilt = OnlineSlamPipeline::new(
            VisualMap::new(),
            Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            LocalMappingPipeline::default(),
            slam.config.clone(),
        );
    }

    /// A result with `estimated_gravity_world: None` (the
    /// `estimate_gravity = false` legacy path) must leave every gravity
    /// sink untouched — the mirroring in Step 3 is entirely gated on the
    /// `Some(...)` case.
    #[test]
    fn no_estimated_gravity_leaves_every_sink_untouched() {
        let mut slam = minimal_pipeline_with_motion_vi_state();
        slam.imu_state = Some(OnlineSlamImuState::new(OnlineSlamImuConfig::default()));
        slam.config.imu = Some(OnlineSlamImuConfig::default());
        let original_gravity = OnlineSlamImuConfig::default().gravity_world;

        let result = fake_result(1, Vector3::zeros(), true);
        assert!(result.estimated_gravity_world.is_none());
        slam.promote_motion_vi_init_result(result);

        assert_eq!(
            slam.imu_state.as_ref().unwrap().config.gravity_world,
            original_gravity
        );
        assert_eq!(
            slam.config.imu.as_ref().unwrap().gravity_world,
            original_gravity
        );
    }

    /// A [`crate::BiasReleaseSchedule`] Stage A ("velocity stage")
    /// promotion (`bias_released = false`) must unblock
    /// [`OnlineSlamPipeline::vi_initialization_pending`] even though the
    /// motion-VI stage stays non-terminal (`vi_motion_init_state.completed`
    /// stays `None`) — see that method's doc comment for the full
    /// rationale. Regression for the bug where local VI-BA never fired
    /// after a Stage A promotion on the real bench (3 Stage A promotions,
    /// `local_vi_ba_triggers = 0`).
    #[test]
    fn stage_a_promotion_unblocks_vi_initialization_pending() {
        let mut slam = minimal_pipeline_with_motion_vi_state();
        slam.vi_init_state = Some(OnlineSlamViInitState::new(OnlineSlamViInitConfig::default()));

        // Static init incomplete + motion-VI stage never fired: pending.
        assert!(
            slam.vi_initialization_pending(),
            "with no Stage A fired and static init incomplete, local VI-BA must stay gated"
        );

        // Drive a synthetic Stage A promotion. `promote_motion_vi_init_result`
        // alone only exercises the pipeline-side mirroring (as the other
        // tests in this module do); the inner initializer's own
        // `velocity_stage` cache — what `velocity_stage_fired()` actually
        // reads — is populated by `MotionBasedViInitializer::try_initialize_with_bias_seed`
        // itself as a side effect of the real numeric solve (see
        // `run_motion_vi_init_step`). Use the test-only seam to inject the
        // same result there without replaying that solve; the solve path
        // is covered separately by `vi_motion_initializer`'s own
        // `bias_release_schedule_stage_a_fires_then_awaits_release_gate`.
        let stage_a = fake_result(1, Vector3::new(1.0, 0.0, 0.0), false);
        slam.vi_motion_init_state
            .as_mut()
            .expect("configured above")
            .initializer
            .set_velocity_stage_result_for_test(stage_a.clone());
        match slam.promote_motion_vi_init_result(stage_a) {
            MotionViInitializationEvent::Succeeded { result } => assert!(!result.bias_released),
            other => panic!("expected Succeeded, got {other:?}"),
        }
        let motion_state = slam
            .vi_motion_init_state
            .as_ref()
            .expect("configured above");
        assert!(
            motion_state.completed.is_none(),
            "Stage A must NOT mark the motion-VI stage completed"
        );
        assert!(
            motion_state.velocity_stage_fired(),
            "Stage A must be visible via velocity_stage_fired()"
        );

        assert!(
            !slam.vi_initialization_pending(),
            "a Stage A promotion must unblock vi_initialization_pending even though the \
             motion-VI stage remains non-terminal"
        );
    }

    /// Regression: with no Stage A fired and static init incomplete,
    /// local VI-BA must remain gated. Guards against a future change
    /// accidentally loosening `vi_initialization_pending` beyond the
    /// documented Stage-A carve-out.
    #[test]
    fn no_stage_a_and_incomplete_static_init_keeps_vi_initialization_pending() {
        let mut slam = minimal_pipeline_with_motion_vi_state();
        slam.vi_init_state = Some(OnlineSlamViInitState::new(OnlineSlamViInitConfig::default()));

        assert!(
            slam.vi_initialization_pending(),
            "static init incomplete + motion-VI stage never fired must stay pending"
        );

        // Even a `StillWaiting` rejection (no promotion at all) must leave
        // the gate engaged.
        assert!(slam
            .vi_motion_init_state
            .as_ref()
            .unwrap()
            .completed
            .is_none());
        assert!(!slam
            .vi_motion_init_state
            .as_ref()
            .unwrap()
            .velocity_stage_fired());
        assert!(slam.vi_initialization_pending());
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OnlineSlamResult {
    pub tracking: TrackingResult,
    pub mapping: Option<LocalMappingResult>,
    pub applied_update: Option<AppliedMapUpdate>,
    pub loop_closure_candidates: Vec<LoopClosureCandidate>,
    /// IMU pre-integration factor closed against the keyframe registered
    /// by this `process_frame` call. `Some` only when [`OnlineSlamConfig::imu`]
    /// is `Some`, this frame produced a new keyframe, and at least one
    /// previous keyframe exists to anchor the integration window's left
    /// endpoint. Downstream pose-graph / BA glue consumes the factor; the
    /// `OnlineSlamPipeline` itself does not optimise against it.
    pub imu_factor: Option<ImuPreintegrationFactor>,
    /// Local VI-BA refinement outcome when
    /// [`OnlineSlamConfig::local_vi_ba`] is enabled and the current
    /// `process_frame` call triggered the sliding-window solve. `None`
    /// otherwise (disabled, no new factor, or the window was too short
    /// to refine).
    pub local_vi_ba: Option<OnlineSlamLocalBaStats>,
    /// Sparse visual factor lifecycle update for the newly committed
    /// keyframe. `None` when disabled or no keyframe was added.
    pub sparse_factor_graph: Option<SparseFactorGraphUpdateStats>,
    /// Visual-only covisibility local BA outcome when
    /// [`OnlineSlamConfig::covisibility_local_ba`] is enabled and the
    /// current `process_frame` call triggered the solve. `None`
    /// otherwise (disabled, no newly-applied keyframe, below the
    /// configured minimum keyframe count, or skipped by the trigger
    /// interval).
    pub covisibility_local_ba: Option<OnlineSlamCovisibilityLocalBaStats>,
    pub map_keyframe_count: usize,
    pub map_landmark_count: usize,
    /// State-transition event from the auto-bootstrap stage. `Some`
    /// only on the frame where the VI initialiser actually changed
    /// state — `Succeeded`, `StillBuffering`, or `GaveUp`. `None` on
    /// every other frame (no new keyframe registered, VI init
    /// disabled, or the stage has already reached a terminal event).
    /// Durable state is exposed via
    /// [`OnlineSlamPipeline::vi_initialization_status`].
    pub vi_init: Option<ViInitializationEvent>,
    /// State-transition event from the motion-based VI init stage.
    /// `Some` only on the frame where the motion-based initialiser
    /// actually changed state (`Succeeded` or `StillWaiting`); `None`
    /// otherwise (no new keyframe, stage disabled, static seed not yet
    /// available, or already in the terminal `Initialised` state).
    /// Durable state is exposed via
    /// [`OnlineSlamPipeline::motion_vi_initialization_status`].
    pub vi_motion_init: Option<MotionViInitializationEvent>,
    /// Per-frame outcome of the online loop-closure + pose-graph
    /// refinement stage. `Some` only when
    /// [`OnlineSlamConfig::pose_graph_refinement`] is enabled AND the
    /// verifier was actually run this frame (i.e. the current frame
    /// produced a keyframe AND there was at least one candidate). `None`
    /// otherwise — the stage being disabled, no candidate this frame,
    /// or no new keyframe to attach the verifier output to.
    pub pose_graph_refinement: Option<OnlineSlamLoopClosureRefinementStats>,
    /// Per-frame outcome of the relocalization-on-tracker-death stage.
    /// `Some` only on frames where primary tracking failed AND the
    /// stage was enabled — `stats.attempted` is always `true` when this
    /// field is `Some`, and `stats.succeeded` reflects whether the
    /// recovery PnP solution cleared the acceptance gates. `None` on
    /// every other frame (stage disabled, or primary tracking already
    /// succeeded so no relocalization was needed).
    pub relocalization: Option<OnlineSlamRelocalizationStats>,
}

impl OnlineSlamResult {
    pub fn tracking_succeeded(&self) -> bool {
        self.tracking.localization.success
    }

    pub fn map_was_updated(&self) -> bool {
        self.applied_update.is_some()
    }

    pub fn has_loop_closure_candidate(&self) -> bool {
        !self.loop_closure_candidates.is_empty()
    }
}

/// [`OnlineSlamLoopClosureRefinementConfig::propagate_corrections`]'s
/// landmark-propagation pass. For every landmark in `map.landmarks`,
/// derives its ANCHOR keyframe — the first (lowest-id) keyframe in
/// `map.keyframes` whose `observations` reference that landmark id — and,
/// when that anchor's id is a key in `corrections`, moves the landmark
/// rigidly by the anchor's correction:
/// `landmark.position = correction.transform_point(&landmark.position)`.
///
/// `corrections` maps a solved keyframe id to its world-frame rigid
/// correction `C_k = T_cw_new⁻¹ ∘ T_cw_old` (new camera-to-world composed
/// with old world-to-camera), i.e. the transform such that for any point
/// `p` observed by keyframe `k`,
/// `T_wc_old.transform_point(&p) == T_wc_new.transform_point(&correction.transform_point(&p))`
/// — the point's projection into keyframe `k` is invariant across the
/// correction. This is exactly the rigid motion a landmark anchored to
/// `k` must undergo to stay consistent with `k`'s corrected pose.
///
/// Landmarks whose anchor keyframe is NOT a key in `corrections` (i.e.
/// the anchor either was not part of this solve or had no previously-
/// known pose to diff against) are left untouched, matching ORB-SLAM3's
/// "only points anchored to a corrected keyframe move" propagation rule.
///
/// The anchor index is rebuilt from scratch every call by scanning every
/// keyframe's `observations` in ascending keyframe-id order — this is
/// `O(total observations across the map)`, which for the maps this stage
/// targets (on the order of 1-3k keyframes / 50-200k landmarks) is cheap
/// relative to the pose-graph solve that triggered the call. Returns
/// `(landmarks_moved, max_displacement_meters, mean_displacement_meters)`;
/// the latter two are `None` when no landmark moved.
fn propagate_pose_graph_corrections(
    map: &mut VisualMap,
    corrections: &HashMap<u64, SE3>,
) -> (usize, Option<f64>, Option<f64>) {
    let mut anchor_keyframe_of_landmark: HashMap<u64, u64> = HashMap::new();
    let mut keyframe_ids: Vec<u64> = map.keyframes.keys().copied().collect();
    keyframe_ids.sort_unstable();
    for keyframe_id in keyframe_ids {
        let Some(keyframe) = map.keyframes.get(&keyframe_id) else {
            continue;
        };
        for observation in &keyframe.observations {
            anchor_keyframe_of_landmark
                .entry(observation.landmark_id)
                .or_insert(keyframe_id);
        }
    }

    let mut moved = 0usize;
    let mut max_displacement = 0.0_f64;
    let mut sum_displacement = 0.0_f64;
    for (landmark_id, anchor_keyframe_id) in &anchor_keyframe_of_landmark {
        let Some(correction) = corrections.get(anchor_keyframe_id) else {
            continue;
        };
        let Some(landmark) = map.landmarks.get_mut(landmark_id) else {
            continue;
        };
        let old_position = landmark.position;
        landmark.position = correction.transform_point(&old_position);
        let displacement = (landmark.position - old_position).norm();
        if let Some(covariance) = map.landmark_position_covariances.get_mut(landmark_id) {
            let rotation = correction.rotation.to_rotation_matrix().into_inner();
            *covariance = rotation * *covariance * rotation.transpose();
        }
        moved += 1;
        sum_displacement += displacement;
        if displacement > max_displacement {
            max_displacement = displacement;
        }
    }

    if moved == 0 {
        (0, None, None)
    } else {
        (
            moved,
            Some(max_displacement),
            Some(sum_displacement / moved as f64),
        )
    }
}

/// Embed a rigid `SE(3)` transform as a `Sim(3)` value at scale `1.0`. Used
/// to seed the [`Sim3PoseGraph`] mirror's nodes/edges from the same
/// (metric) poses and relative measurements the `Se3` path uses, and to
/// re-embed a keyframe's PRE-solve pose when computing its Sim3
/// correction (see [`LoopRefinementSolver::Sim3`]'s doc comment).
fn sim3_at_unit_scale(se3: &SE3) -> Sim3 {
    Sim3::new(se3.rotation, se3.translation, 1.0)
}

/// Estimate the relative scale of two independently triangulated keyframe
/// regions from descriptor-matched 3D landmarks. Pairwise-distance ratios are
/// invariant to the loop rotation and translation, so their median supplies
/// the scale of the `from-camera -> to-camera` Sim3 measurement without
/// coupling it to the PnP translation already used by the rigid graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sim3ScaleEstimationFailure {
    InsufficientPoints,
    InsufficientMatches,
    NoConsensus,
}

fn estimate_loop_sim3_scale_3d3d(
    map: &VisualMap,
    from_keyframe_id: u64,
    to_keyframe_id: u64,
    pnp_query_landmark_pairs: &[(usize, u64)],
    metric_points_camera: &HashMap<usize, Point3<f64>>,
) -> Result<f64, Sim3ScaleEstimationFailure> {
    let from = map
        .keyframes
        .get(&from_keyframe_id)
        .ok_or(Sim3ScaleEstimationFailure::InsufficientPoints)?;
    let to = map
        .keyframes
        .get(&to_keyframe_id)
        .ok_or(Sim3ScaleEstimationFailure::InsufficientPoints)?;
    let from_pose = from
        .frame
        .pose
        .as_ref()
        .ok_or(Sim3ScaleEstimationFailure::InsufficientPoints)?;
    let to_pose = to
        .frame
        .pose
        .as_ref()
        .ok_or(Sim3ScaleEstimationFailure::InsufficientPoints)?;

    let collect = |keyframe: &Keyframe, pose: &Pose, prefer_frame_descriptor: bool| {
        keyframe
            .observations
            .iter()
            .filter_map(|observation| {
                let landmark = map.landmarks.get(&observation.landmark_id)?;
                // Fresh stereo/replenished landmarks do not always have a
                // descriptor promoted into the map slot yet, while their
                // observing keyframe still owns the exact feature descriptor
                // that PnP used. Reuse it instead of discarding an otherwise
                // valid 3D point from the Sim3 correspondence pool.
                let frame_descriptor = || {
                    keyframe
                        .frame
                        .descriptors
                        .get(observation.keypoint_index)
                        .cloned()
                };
                let descriptor = if prefer_frame_descriptor {
                    frame_descriptor().or_else(|| landmark.descriptor.clone())
                } else {
                    landmark.descriptor.clone().or_else(frame_descriptor)
                }?;
                Some((
                    observation.landmark_id,
                    descriptor,
                    pose.transform_world_point(&landmark.position),
                ))
            })
            .collect::<Vec<_>>()
    };
    // Match in the same direction as appearance PnP: current/query frame
    // features against the older keyframe's map-landmark descriptors. The
    // current observation's 3D landmark then supplies the second endpoint.
    let from_points = collect(from, from_pose, false);
    let to_points = collect(to, to_pose, true);
    if from_points.len() < 4 || to_points.len() < 4 {
        return Err(Sim3ScaleEstimationFailure::InsufficientPoints);
    }
    let mut matched = Vec::new();
    if pnp_query_landmark_pairs.is_empty() {
        let from_descriptors: Vec<Vec<f32>> =
            from_points.iter().map(|(_, d, _)| d.clone()).collect();
        let to_descriptors: Vec<Vec<f32>> = to_points.iter().map(|(_, d, _)| d.clone()).collect();
        let matcher = visloc_vision::matching::BruteForceMatcher::default();
        let forward = matcher.match_descriptors(&from_descriptors, &to_descriptors);
        let reverse = matcher.match_descriptors(&to_descriptors, &from_descriptors);
        for m in forward {
            let reciprocal = reverse
                .iter()
                .any(|r| r.query_index == m.train_index && r.train_index == m.query_index);
            if reciprocal && from_points[m.query_index].0 != to_points[m.train_index].0 {
                matched.push((from_points[m.query_index].2, to_points[m.train_index].2));
            }
        }
    } else if !metric_points_camera.is_empty() {
        for &(query_index, old_landmark_id) in pnp_query_landmark_pairs {
            let Some(current_point) = metric_points_camera.get(&query_index) else {
                continue;
            };
            let Some(old_landmark) = map.landmarks.get(&old_landmark_id) else {
                continue;
            };
            matched.push((
                from_pose.transform_world_point(&old_landmark.position),
                *current_point,
            ));
        }
    } else {
        let current_landmark_by_keypoint: HashMap<usize, u64> = to
            .observations
            .iter()
            .map(|observation| (observation.keypoint_index, observation.landmark_id))
            .collect();
        for &(query_index, old_landmark_id) in pnp_query_landmark_pairs {
            let Some(&current_landmark_id) = current_landmark_by_keypoint.get(&query_index) else {
                continue;
            };
            if current_landmark_id == old_landmark_id {
                continue;
            }
            let Some(old_landmark) = map.landmarks.get(&old_landmark_id) else {
                continue;
            };
            let Some(current_landmark) = map.landmarks.get(&current_landmark_id) else {
                continue;
            };
            matched.push((
                from_pose.transform_world_point(&old_landmark.position),
                to_pose.transform_world_point(&current_landmark.position),
            ));
        }
    }
    if matched.len() < 6 {
        return Err(Sim3ScaleEstimationFailure::InsufficientMatches);
    }

    // ORB-SLAM's Sim3Solver uses minimal-set RANSAC and accepts a hypothesis
    // only when many MapPoint correspondences agree geometrically. Mirror
    // that safety property here with deterministic triplet enumeration.
    let mut scene_distances = Vec::new();
    for i in 0..matched.len() {
        for j in (i + 1)..matched.len() {
            scene_distances.push((matched[i].1 - matched[j].1).norm());
        }
    }
    scene_distances.sort_by(f64::total_cmp);
    let scene_scale = scene_distances[scene_distances.len() / 2];
    // Rectified stereo depth uncertainty grows quadratically with range;
    // ORB-SLAM therefore gates Sim3 correspondences in image reprojection
    // space rather than demanding centimetre-level 3D agreement. This
    // dependency-free approximation scales the metric tolerance with scene
    // extent while retaining a finite cap against arbitrary matches.
    let inlier_threshold = (scene_scale * 0.10).clamp(0.05, 1.0);

    let mut best_inliers = Vec::new();
    let mut best_rmse = f64::INFINITY;
    let mut hypotheses = 0usize;
    'triplets: for i in 0..matched.len() - 2 {
        for j in (i + 1)..matched.len() - 1 {
            for k in (j + 1)..matched.len() {
                if hypotheses >= 256 {
                    break 'triplets;
                }
                hypotheses += 1;
                let source = [matched[i].0, matched[j].0, matched[k].0];
                let target = [matched[i].1, matched[j].1, matched[k].1];
                let Some(model) =
                    visloc_tracking::umeyama_similarity_transform(&source, &target, true)
                else {
                    continue;
                };
                if !model.scale.is_finite() || !(0.5..=2.0).contains(&model.scale) {
                    continue;
                }
                let mut inliers = Vec::new();
                let mut squared_error = 0.0;
                for (index, (from_point, to_point)) in matched.iter().enumerate() {
                    let error = (model.apply(from_point) - to_point).norm();
                    if error <= inlier_threshold {
                        inliers.push(index);
                        squared_error += error * error;
                    }
                }
                let rmse = if inliers.is_empty() {
                    f64::INFINITY
                } else {
                    (squared_error / inliers.len() as f64).sqrt()
                };
                if inliers.len() > best_inliers.len()
                    || (inliers.len() == best_inliers.len() && rmse < best_rmse)
                {
                    best_inliers = inliers;
                    best_rmse = rmse;
                }
            }
        }
    }
    let min_inliers = 6usize.max(matched.len().div_ceil(2));
    if best_inliers.len() < min_inliers || best_rmse > inlier_threshold * 0.75 {
        return Err(Sim3ScaleEstimationFailure::NoConsensus);
    }
    let source: Vec<Point3<f64>> = best_inliers.iter().map(|&i| matched[i].0).collect();
    let target: Vec<Point3<f64>> = best_inliers.iter().map(|&i| matched[i].1).collect();
    let refined = visloc_tracking::umeyama_similarity_transform(&source, &target, true)
        .ok_or(Sim3ScaleEstimationFailure::NoConsensus)?;
    (refined.scale.is_finite() && (0.5..=2.0).contains(&refined.scale))
        .then_some(refined.scale)
        .ok_or(Sim3ScaleEstimationFailure::NoConsensus)
}

/// `(min, max)` per-node scale across a solved [`Sim3PoseGraph`]'s nodes.
/// `None` when the graph has no nodes (never observed in practice — the
/// graph always has at least the anchor before a solve can fire).
fn sim3_pose_scale_spread(graph: &Sim3PoseGraph) -> Option<(f64, f64)> {
    let mut scales = graph.poses.values().map(|pose| pose.scale);
    let first = scales.next()?;
    let (min, max) = scales.fold((first, first), |(min, max), scale| {
        (min.min(scale), max.max(scale))
    });
    Some((min, max))
}

/// [`LoopRefinementSolver::Sim3`]'s counterpart to
/// [`propagate_pose_graph_corrections`]: for every landmark, derives its
/// ANCHOR keyframe exactly as the `Se3` path does and, when that anchor's
/// id is a key in `corrections`, moves the landmark by the anchor's
/// **similarity** correction: `landmark.position =
/// correction.transform_point(&landmark.position)`.
///
/// `corrections` maps a solved keyframe id to its world-frame `Sim(3)`
/// correction `C_k = Siw_new⁻¹ ∘ Siw_old` — the direct `Sim(3)`
/// generalisation of the `Se3` path's `C_k = T_cw_new⁻¹ ∘ T_cw_old`, where
/// `Siw_old`/`Siw_new` are the keyframe's PRE-/POST-solve `Sim3` nodes
/// (see [`sim3_at_unit_scale`] and [`LoopRefinementSolver::Sim3`]'s doc
/// comment for the full derivation and why this keeps a landmark's
/// reprojection *ray* into keyframe `k` invariant — a positive rescale of
/// camera-frame depth does not move the pinhole-projected pixel).
///
/// Returns `(landmarks_moved, max_displacement_meters, mean_displacement_meters)`,
/// mirroring [`propagate_pose_graph_corrections`]'s contract exactly.
fn propagate_pose_graph_corrections_sim3(
    map: &mut VisualMap,
    corrections: &HashMap<u64, Sim3>,
) -> (usize, Option<f64>, Option<f64>) {
    let mut anchor_keyframe_of_landmark: HashMap<u64, u64> = HashMap::new();
    let mut keyframe_ids: Vec<u64> = map.keyframes.keys().copied().collect();
    keyframe_ids.sort_unstable();
    for keyframe_id in keyframe_ids {
        let Some(keyframe) = map.keyframes.get(&keyframe_id) else {
            continue;
        };
        for observation in &keyframe.observations {
            anchor_keyframe_of_landmark
                .entry(observation.landmark_id)
                .or_insert(keyframe_id);
        }
    }

    let mut moved = 0usize;
    let mut max_displacement = 0.0_f64;
    let mut sum_displacement = 0.0_f64;
    for (landmark_id, anchor_keyframe_id) in &anchor_keyframe_of_landmark {
        let Some(correction) = corrections.get(anchor_keyframe_id) else {
            continue;
        };
        let Some(landmark) = map.landmarks.get_mut(landmark_id) else {
            continue;
        };
        let old_position = landmark.position;
        landmark.position = correction.transform_point(&old_position);
        let displacement = (landmark.position - old_position).norm();
        if let Some(covariance) = map.landmark_position_covariances.get_mut(landmark_id) {
            let rotation = correction.rotation.to_rotation_matrix().into_inner();
            *covariance = rotation * *covariance * rotation.transpose() * correction.scale.powi(2);
        }
        moved += 1;
        sum_displacement += displacement;
        if displacement > max_displacement {
            max_displacement = displacement;
        }
    }

    if moved == 0 {
        (0, None, None)
    } else {
        (
            moved,
            Some(max_displacement),
            Some(sum_displacement / moved as f64),
        )
    }
}

fn apply_loop_edge_scale(information: Matrix6<f64>, loop_edge_scale: f64) -> Matrix6<f64> {
    information * loop_edge_scale
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod sim3_scale_estimation_tests {
    use super::*;
    use nalgebra::{Point2, Point3};
    use visloc_core::types::Landmark;

    #[test]
    fn bias_promotion_reset_preserves_preintegration_noise_model() {
        let noise = ImuNoiseModel {
            gyroscope_noise_density: 1.0e-3,
            accelerometer_noise_density: 2.0e-2,
        };
        let mut state = OnlineSlamImuState::new(OnlineSlamImuConfig {
            noise_model: Some(noise),
            ..OnlineSlamImuConfig::default()
        });
        let bias_gyro = Vector3::new(0.01, -0.02, 0.03);
        let bias_acc = Vector3::new(0.1, -0.2, 0.3);
        state.reset_preintegrator_with_bias(bias_gyro, bias_acc);
        state
            .preintegrator
            .integrate_sample(Vector3::zeros(), Vector3::new(0.0, 0.0, 9.81), 0.01);
        let delta = state.preintegrator.delta();
        assert_eq!(delta.bias_gyro_linearisation, bias_gyro);
        assert_eq!(delta.bias_acc_linearisation, bias_acc);
        assert!(delta.covariance.trace() > 0.0);
    }

    #[test]
    fn loop_edge_scale_preserves_information_anisotropy() {
        let information = Matrix6::from_diagonal(&nalgebra::SVector::<f64, 6>::new(
            1.0, 2.0, 4.0, 8.0, 16.0, 32.0,
        ));
        let scaled = apply_loop_edge_scale(information, 0.1);
        assert!((scaled - information * 0.1).norm() < 1.0e-15);
        assert!((scaled[(5, 5)] / scaled[(0, 0)] - 32.0).abs() < 1.0e-15);
    }

    #[test]
    fn loop_observation_fusion_reassigns_both_map_indices_idempotently() {
        let mut map = VisualMap::new();
        map.cameras
            .insert(1, Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0));
        let mut old_frame = Frame::new(10, 1);
        old_frame.pose = Some(Pose::identity());
        old_frame.keypoints.push(Point2::new(320.0, 240.0));
        let old_observation = Observation {
            frame_id: 10,
            landmark_id: 1,
            keypoint_index: 0,
            xy: old_frame.keypoints[0],
        };
        map.keyframes.insert(
            10,
            Keyframe {
                frame: old_frame,
                observations: vec![old_observation.clone()],
            },
        );
        let mut frame = Frame::new(20, 1);
        frame.pose = Some(Pose::identity());
        frame.keypoints.push(Point2::new(320.0, 240.0));
        let previous = Observation {
            frame_id: 20,
            landmark_id: 2,
            keypoint_index: 0,
            xy: frame.keypoints[0],
        };
        map.keyframes.insert(
            20,
            Keyframe {
                frame,
                observations: vec![previous.clone()],
            },
        );
        let mut old_landmark = Landmark::new(1, Point3::new(0.0, 0.0, 5.0));
        old_landmark.observations.push(old_observation);
        map.landmarks.insert(1, old_landmark);
        let mut duplicate = Landmark::new(2, Point3::new(0.1, 0.0, 5.0));
        duplicate.observations.push(previous);
        map.landmarks.insert(2, duplicate);

        let camera = map.cameras[&1].clone();
        let first = fuse_loop_observations(&mut map, &camera, 20, &[(0, 1)], 4.0);
        assert_eq!(
            first,
            LoopObservationFusionStats {
                pairs_considered: 1,
                observations_inserted: 1,
                observations_reassigned: 1,
                pairs_skipped: 0,
                reprojection_rejected: 0,
            }
        );
        assert_eq!(map.keyframes[&20].observations[0].landmark_id, 1);
        assert_eq!(map.landmarks[&1].observations.len(), 2);
        assert!(map.landmarks[&2].observations.is_empty());
        assert!(map.validate().is_valid());

        let selection = select_covisibility_local_ba_window(
            &map,
            20,
            &CovisibilityLocalBaConfig {
                max_neighbor_keyframes: 1,
                min_shared_landmarks: 1,
                max_boundary_keyframes: 0,
                min_boundary_observations: 1,
                min_observations_per_landmark: 2,
                min_active_observations: 1,
                ..CovisibilityLocalBaConfig::default()
            },
        )
        .expect("fused loop observation should connect both welding regions");
        assert_eq!(selection.optimized_keyframe_ids, vec![20, 10]);
        assert_eq!(selection.landmark_ids, vec![1]);

        let second = fuse_loop_observations(&mut map, &camera, 20, &[(0, 1)], 4.0);
        assert_eq!(second.pairs_considered, 1);
        assert_eq!(second.observations_inserted, 0);
        assert_eq!(second.observations_reassigned, 0);
        assert_eq!(second.pairs_skipped, 0);
        assert_eq!(map.keyframes[&20].observations.len(), 1);
        assert_eq!(map.landmarks[&1].observations.len(), 2);

        map.keyframes
            .get_mut(&20)
            .unwrap()
            .frame
            .keypoints
            .push(Point2::new(100.0, 100.0));
        map.landmarks
            .insert(3, Landmark::new(3, Point3::new(0.0, 0.0, 5.0)));
        let rejected = fuse_loop_observations(&mut map, &camera, 20, &[(1, 3)], 4.0);
        assert_eq!(rejected.pairs_considered, 1);
        assert_eq!(rejected.observations_inserted, 0);
        assert_eq!(rejected.pairs_skipped, 1);
        assert_eq!(rejected.reprojection_rejected, 1);
        assert!(map.keyframes[&20]
            .observations
            .iter()
            .all(|observation| observation.keypoint_index != 1));
    }

    #[test]
    fn loop_welding_region_correction_preserves_local_reprojection() {
        let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut map = VisualMap::new();
        map.cameras.insert(1, camera.clone());

        let mut matched = Frame::new(10, 1);
        matched.pose = Some(Pose::identity());
        map.keyframes.insert(
            10,
            Keyframe {
                frame: matched,
                observations: Vec::new(),
            },
        );

        let drifted_pose = Pose::from_world_to_camera(
            nalgebra::UnitQuaternion::identity(),
            Vector3::new(1.0, 0.0, 0.0),
        );
        let local_point = Point3::new(0.0, 0.0, 5.0);
        let local_xy = camera
            .project(&drifted_pose.transform_world_point(&local_point))
            .unwrap();
        let mut local_landmark = Landmark::new(2, local_point);
        for id in [20_u64, 21_u64] {
            let mut frame = Frame::new(id, 1);
            frame.pose = Some(drifted_pose.clone());
            frame.keypoints.push(local_xy);
            let observation = Observation {
                frame_id: id,
                landmark_id: 2,
                keypoint_index: 0,
                xy: local_xy,
            };
            map.keyframes.insert(
                id,
                Keyframe {
                    frame,
                    observations: vec![observation.clone()],
                },
            );
            local_landmark.observations.push(observation);
        }
        map.landmarks.insert(2, local_landmark);

        let corrected = correct_loop_welding_region(
            &mut map,
            10,
            20,
            &SE3::identity(),
            &CovisibilityLocalBaConfig {
                max_neighbor_keyframes: 1,
                min_shared_landmarks: 1,
                max_boundary_keyframes: 0,
                min_boundary_observations: 1,
                min_observations_per_landmark: 2,
                min_active_observations: 1,
                ..CovisibilityLocalBaConfig::default()
            },
        )
        .unwrap();

        assert_eq!(corrected.keyframe_ids, vec![20, 21]);
        assert!((corrected.translation_meters - 1.0).abs() < 1.0e-9);
        assert!(corrected.rotation_radians < 1.0e-9);
        for id in corrected.keyframe_ids {
            let pose = map.keyframes[&id].frame.pose.as_ref().unwrap();
            let projected = camera
                .project(&pose.transform_world_point(&map.landmarks[&2].position))
                .unwrap();
            assert!((projected - local_xy).norm() < 1.0e-9);
        }
        assert!(
            map.keyframes[&20]
                .frame
                .pose
                .as_ref()
                .unwrap()
                .world_to_camera
                .translation
                .norm()
                < 1.0e-9
        );
        assert!((map.landmarks[&2].position - Point3::new(1.0, 0.0, 5.0)).norm() < 1.0e-9);
    }

    fn covariance_propagation_map() -> VisualMap {
        let mut map = VisualMap::new();
        map.landmarks
            .insert(1, Landmark::new(1, Point3::new(1.0, 0.0, 2.0)));
        map.landmark_position_covariances.insert(
            1,
            nalgebra::Matrix3::from_diagonal(&Vector3::new(1.0, 4.0, 9.0)),
        );
        let frame = Frame::new(10, 1);
        map.keyframes.insert(
            10,
            Keyframe {
                frame,
                observations: vec![Observation {
                    frame_id: 10,
                    landmark_id: 1,
                    keypoint_index: 0,
                    xy: Point2::new(0.0, 0.0),
                }],
            },
        );
        map
    }

    #[test]
    fn se3_landmark_propagation_rotates_position_covariance() {
        let mut map = covariance_propagation_map();
        let correction = SE3::new(
            nalgebra::UnitQuaternion::from_axis_angle(
                &Vector3::z_axis(),
                std::f64::consts::FRAC_PI_2,
            ),
            Vector3::new(0.5, 0.0, 0.0),
        );
        propagate_pose_graph_corrections(&mut map, &HashMap::from([(10, correction)]));
        let covariance = map.landmark_position_covariances.get(&1).unwrap();
        let expected = nalgebra::Matrix3::from_diagonal(&Vector3::new(4.0, 1.0, 9.0));
        assert!((*covariance - expected).norm() < 1.0e-10);
    }

    #[test]
    fn sim3_landmark_propagation_rotates_and_scales_position_covariance() {
        let mut map = covariance_propagation_map();
        let correction = Sim3::new(nalgebra::UnitQuaternion::identity(), Vector3::zeros(), 2.0);
        propagate_pose_graph_corrections_sim3(&mut map, &HashMap::from([(10, correction)]));
        let covariance = map.landmark_position_covariances.get(&1).unwrap();
        let expected = nalgebra::Matrix3::from_diagonal(&Vector3::new(4.0, 16.0, 36.0));
        assert!((*covariance - expected).norm() < 1.0e-10);
    }

    fn verified_appearance_candidate(query_frame_id: u64) -> LoopClosureCandidate {
        verified_appearance_candidate_with_pose(query_frame_id, 10, SE3::identity())
    }

    fn verified_appearance_candidate_with_pose(
        query_frame_id: u64,
        matched_keyframe_id: u64,
        relative_pose: SE3,
    ) -> LoopClosureCandidate {
        LoopClosureCandidate {
            query_frame_id,
            matched_keyframe_id,
            shared_landmark_count: 40,
            query_inlier_count: 40,
            keyframe_observation_count: 40,
            shared_landmark_ratio: 1.0,
            score: 10.0,
            geometrically_verified: true,
            verification: Some(LoopClosureVerification {
                verified: true,
                correspondence_count: 40,
                inlier_count: 40,
                inlier_ratio: 1.0,
                mean_sampson_error: 0.0,
                score: 10.0,
                failure_reason: None,
                relative_pose: Some(relative_pose),
                mean_reprojection_error_px: Some(0.0),
            }),
            pnp_query_landmark_pairs: Vec::new(),
        }
    }

    #[test]
    fn pending_appearance_region_is_carried_by_projection_without_retrieval() {
        use nalgebra::{UnitQuaternion, Vector3};

        let camera = Camera::pinhole(1, 640, 480, 420.0, 415.0, 320.0, 240.0);
        let mut map = VisualMap::new();
        map.cameras.insert(camera.id, camera.clone());
        let mut root_frame = Frame::new(10, camera.id);
        root_frame.pose = Some(Pose::identity());
        let mut root_observations = Vec::new();
        let mut current_frame = Frame::new(110, camera.id);
        let current_pose =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.05, -0.02, 0.01));
        current_frame.pose = Some(current_pose.clone());

        for index in 0..64usize {
            let row = index / 8;
            let column = index % 8;
            let point = Point3::new(
                (column as f64 - 3.5) * 0.22,
                (row as f64 - 3.5) * 0.18,
                4.0 + (index % 7) as f64 * 0.35,
            );
            let mut descriptor = vec![0.0_f32; 64];
            descriptor[index] = 1.0;
            let landmark_id = index as u64 + 1;
            let mut landmark = Landmark::new(landmark_id, point);
            landmark.descriptor = Some(descriptor.clone());
            let root_xy = camera.project(&point).unwrap();
            let observation = Observation {
                frame_id: root_frame.id,
                landmark_id,
                keypoint_index: index,
                xy: root_xy,
            };
            root_frame.keypoints.push(root_xy);
            root_frame.descriptors.push(descriptor.clone());
            root_observations.push(observation.clone());
            landmark.observations.push(observation);
            current_frame.keypoints.push(
                camera
                    .project(&current_pose.transform_world_point(&point))
                    .unwrap(),
            );
            current_frame.descriptors.push(descriptor);
            map.landmarks.insert(landmark_id, landmark);
        }
        map.keyframes.insert(
            root_frame.id,
            Keyframe {
                frame: root_frame,
                observations: root_observations,
            },
        );
        let mut last_frame = Frame::new(100, camera.id);
        last_frame.pose = Some(Pose::identity());
        map.keyframes.insert(
            last_frame.id,
            Keyframe {
                frame: last_frame,
                observations: Vec::new(),
            },
        );
        map.keyframes.insert(
            current_frame.id,
            Keyframe {
                frame: current_frame.clone(),
                observations: Vec::new(),
            },
        );
        let pending = AppearancePendingRegion {
            root_keyframe_id: 10,
            last_current_keyframe_id: 100,
            root_to_last_current: SE3::identity(),
            confirmation_count: 1,
            miss_count: 0,
        };
        let config = LoopAppearanceCandidateConfig {
            min_candidate_landmark_count: 30,
            min_projection_correspondence_count: 50,
            projection_search_radius_px: Some(15.0),
            ..LoopAppearanceCandidateConfig::default()
        };

        let result = verify_pending_appearance_region_by_projection(
            &map,
            &current_frame,
            &camera,
            &config,
            &pending,
        );

        assert!(result.attempted);
        assert!(result.correspondence_count >= 50);
        assert!(result.inlier_count >= config.pnp_verifier.min_inliers);
        let candidate = result.candidate.expect("projection carry should verify");
        assert_eq!(candidate.matched_keyframe_id, 10);
        assert_eq!(candidate.query_frame_id, 110);
        assert!(candidate.pnp_query_landmark_pairs.len() >= config.pnp_verifier.min_inliers);
    }

    #[test]
    fn confirms_the_same_appearance_region_across_three_keyframes() {
        let mut map = VisualMap::new();
        for frame_id in [10, 100, 110, 120] {
            let mut frame = Frame::new(frame_id, 1);
            frame.pose = Some(Pose::identity());
            map.keyframes.insert(
                frame_id,
                Keyframe {
                    frame,
                    observations: Vec::new(),
                },
            );
        }
        let config = LoopAppearanceCandidateConfig {
            region_confirmation_required_keyframes: 3,
            ..LoopAppearanceCandidateConfig::default()
        };
        let mut pending = None;

        let (first, waiting, confirmed) = confirm_appearance_region_across_keyframes(
            &map,
            100,
            vec![verified_appearance_candidate(100)],
            &config,
            &mut pending,
        );
        assert!(first.is_empty() && waiting && !confirmed);
        let (second, waiting, confirmed) = confirm_appearance_region_across_keyframes(
            &map,
            110,
            vec![verified_appearance_candidate(110)],
            &config,
            &mut pending,
        );
        assert!(second.is_empty() && waiting && !confirmed);
        let (third, waiting, confirmed) = confirm_appearance_region_across_keyframes(
            &map,
            120,
            vec![verified_appearance_candidate(120)],
            &config,
            &mut pending,
        );
        assert_eq!(third.len(), 1);
        assert!(!waiting && confirmed);
        assert!(pending.is_none());
    }

    #[test]
    fn confirms_overlapping_region_with_nontrivial_pose_composition() {
        use nalgebra::{UnitQuaternion, Vector3};
        use visloc_core::types::Observation;

        let pose = |x: f64| {
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(x, 0.0, 0.0))
        };
        let observation = |frame_id| Observation {
            frame_id,
            landmark_id: 7,
            keypoint_index: 0,
            xy: Point2::origin(),
        };
        let mut map = VisualMap::new();
        for (frame_id, x) in [(10, 0.0), (20, 1.0), (100, 10.0), (110, 11.0), (120, 12.0)] {
            let mut frame = Frame::new(frame_id, 1);
            frame.pose = Some(pose(x));
            let observations = if matches!(frame_id, 10 | 20) {
                vec![observation(frame_id)]
            } else {
                Vec::new()
            };
            map.keyframes.insert(
                frame_id,
                Keyframe {
                    frame,
                    observations,
                },
            );
        }
        let config = LoopAppearanceCandidateConfig {
            covisibility_min_shared_landmarks: 1,
            region_confirmation_required_keyframes: 3,
            ..LoopAppearanceCandidateConfig::default()
        };
        let mut pending = None;

        let first_edge = SE3::new(UnitQuaternion::identity(), Vector3::new(10.0, 0.0, 0.0));
        let second_edge = SE3::new(UnitQuaternion::identity(), Vector3::new(10.0, 0.0, 0.0));
        let third_edge = SE3::new(UnitQuaternion::identity(), Vector3::new(12.0, 0.0, 0.0));
        let (first, waiting, confirmed) = confirm_appearance_region_across_keyframes(
            &map,
            100,
            vec![verified_appearance_candidate_with_pose(100, 10, first_edge)],
            &config,
            &mut pending,
        );
        assert!(first.is_empty() && waiting && !confirmed);
        let (second, waiting, confirmed) = confirm_appearance_region_across_keyframes(
            &map,
            110,
            vec![verified_appearance_candidate_with_pose(
                110,
                20,
                second_edge,
            )],
            &config,
            &mut pending,
        );
        assert!(second.is_empty() && waiting && !confirmed);
        let (third, waiting, confirmed) = confirm_appearance_region_across_keyframes(
            &map,
            120,
            vec![verified_appearance_candidate_with_pose(120, 10, third_edge)],
            &config,
            &mut pending,
        );
        assert_eq!(third.len(), 1);
        assert_eq!(third[0].matched_keyframe_id, 10);
        assert!(!waiting && confirmed);
        assert!(pending.is_none());
    }

    #[test]
    fn recovers_scale_from_duplicate_landmark_regions() {
        let mut map = VisualMap::new();
        let mut from_frame = Frame::new(10, 1);
        let mut to_frame = Frame::new(20, 1);
        from_frame.pose = Some(Pose::identity());
        to_frame.pose = Some(Pose::identity());
        let mut from_observations = Vec::new();
        let mut to_observations = Vec::new();
        let points = [
            Point3::new(0.0, 0.0, 4.0),
            Point3::new(1.0, 0.0, 4.2),
            Point3::new(0.0, 1.0, 4.4),
            Point3::new(1.0, 1.0, 4.8),
            Point3::new(-0.5, 0.3, 5.0),
            Point3::new(0.4, -0.7, 5.3),
            Point3::new(1.2, 0.6, 5.6),
            Point3::new(-0.8, -0.4, 4.6),
        ];
        for (index, point) in points.into_iter().enumerate() {
            let descriptor = vec![index as f32, 1.0, -(index as f32)];
            from_frame.descriptors.push(descriptor.clone());
            to_frame.descriptors.push(descriptor.clone());
            let from_id = index as u64 + 1;
            let to_id = index as u64 + 101;
            let mut from_landmark = Landmark::new(from_id, point);
            if index % 2 == 0 {
                from_landmark.descriptor = Some(descriptor.clone());
            }
            let transformed = if index < 6 {
                point.coords * 1.7 + Vector3::new(2.0, -1.0, 0.5)
            } else {
                Vector3::new(20.0 + index as f64, -15.0, 2.0)
            };
            let mut to_landmark = Landmark::new(to_id, Point3::from(transformed));
            // The query-side map descriptor may be stale/aggregated; the
            // observing frame descriptor above must take precedence.
            to_landmark.descriptor = Some(vec![100.0 + index as f32, -50.0, 25.0]);
            map.landmarks.insert(from_id, from_landmark);
            map.landmarks.insert(to_id, to_landmark);
            from_observations.push(Observation {
                frame_id: 10,
                landmark_id: from_id,
                keypoint_index: index,
                xy: Point2::origin(),
            });
            to_observations.push(Observation {
                frame_id: 20,
                landmark_id: to_id,
                keypoint_index: index,
                xy: Point2::origin(),
            });
        }
        map.keyframes.insert(
            10,
            Keyframe {
                frame: from_frame,
                observations: from_observations,
            },
        );
        map.keyframes.insert(
            20,
            Keyframe {
                frame: to_frame,
                observations: to_observations,
            },
        );

        let pnp_pairs: Vec<(usize, u64)> = (0..points.len())
            .map(|index| (index, index as u64 + 1))
            .collect();
        let scale =
            estimate_loop_sim3_scale_3d3d(&map, 10, 20, &pnp_pairs, &HashMap::new()).unwrap();
        assert!((scale - 1.7).abs() < 1.0e-9, "scale={scale}");
    }
}

fn outlier_observation_ratio(outlier_count: usize, observation_count: usize) -> Option<f64> {
    if observation_count == 0 {
        None
    } else {
        Some(outlier_count as f64 / observation_count as f64)
    }
}

/// Rigidly place the current covisible region at the absolute query pose
/// implied by an accepted loop measurement. This is the SE(3), metric-stereo
/// counterpart of ORB-SLAM's corrected-Sim3 welding initialization: all
/// selected current-region poses and landmarks receive one common world-frame
/// transform, preserving their internal reprojections before cross-loop
/// observations are fused and bundle adjustment begins.
#[derive(Debug, Clone, PartialEq)]
struct LoopWeldingRegionCorrection {
    keyframe_ids: Vec<u64>,
    translation_meters: f64,
    rotation_radians: f64,
}

/// Re-run the essential graph after a successful local weld so the rigid
/// current-region correction is distributed across sequential constraints
/// instead of leaving a one-edge rotation/translation seam. The operation is
/// speculative and rolls both graph and map back if the solved map is invalid
/// or loses the welding window's pre-BA reprojection quality.
fn refine_pose_graph_after_welding(
    map: &mut VisualMap,
    graph: &mut PoseGraph,
    config: &PoseGraphSe3Config,
    gnc: Option<&gnc::GncConfig>,
    active_keyframe_id: u64,
    welding_config: &CovisibilityLocalBaConfig,
    max_mean_reprojection_px: f64,
) -> (
    bool,
    Option<f64>,
    Option<CovisibilityLocalBaError>,
    Option<f64>,
) {
    let original_map = map.clone();
    let original_graph = graph.clone();
    let mut post_config = config.clone();
    // Preserve the geometrically welded initialization. A second chordal seed
    // would discard it and reproduce the pre-welding pose-only solution.
    post_config.chordal_init = false;
    let solved = if let Some(gnc) = gnc {
        graph.optimize_se3_gnc(&post_config, gnc).is_ok()
    } else {
        graph.optimize_se3_iterative(&post_config).is_ok()
    };
    if !solved {
        *map = original_map;
        *graph = original_graph;
        return (false, None, None, None);
    }

    let mut corrections = HashMap::new();
    for (id, new_pose) in &graph.poses {
        let Some(old_pose) = map
            .keyframes
            .get(id)
            .and_then(|keyframe| keyframe.frame.pose.as_ref())
        else {
            continue;
        };
        corrections.insert(
            *id,
            new_pose
                .world_to_camera
                .inverse()
                .compose(&old_pose.world_to_camera),
        );
    }
    for (id, new_pose) in &graph.poses {
        if let Some(keyframe) = map.keyframes.get_mut(id) {
            keyframe.frame.pose = Some(new_pose.clone());
        }
    }
    propagate_pose_graph_corrections(map, &corrections);
    // Essential-graph optimization is pose-only and can temporarily pull the
    // newly fused observations far off their pixels. Mirror ORB-SLAM's final
    // BA stage from that globally distributed initialization. Unlike the
    // pre-PGO seam BA, rotations are free here: the graph has already removed
    // the discrete region boundary that required temporary anchoring.
    let mut post_welding_config = welding_config.clone();
    post_welding_config.ba_config.max_iterations =
        post_welding_config.ba_config.max_iterations.max(20);
    // Use the same pixel scale as the welding window's default outlier gate.
    // A 20 px transition kept too much leverage on cross-loop residuals and
    // measurably rotated the short-baseline trajectory after an otherwise
    // successful weld.
    post_welding_config.ba_config.robust_kernel = RobustKernel::Huber { delta: 5.0 };
    let post_ba =
        match refine_visual_map_with_covisibility_ba(map, active_keyframe_id, &post_welding_config)
        {
            Ok(result) => result,
            Err(error) => {
                *map = original_map;
                *graph = original_graph;
                return (false, None, Some(error), None);
            }
        };
    for id in &post_ba.selection.optimized_keyframe_ids {
        if let Some(pose) = map
            .keyframes
            .get(id)
            .and_then(|keyframe| keyframe.frame.pose.clone())
        {
            if graph.poses.contains_key(id) {
                graph.poses.insert(*id, pose);
            }
        }
    }
    let mean_reprojection = mean_selected_reprojection_px(map, &post_ba.selection);
    let behind_camera_ratio = behind_camera_optimized_landmark_ratio(map, &post_ba.selection);
    let reprojection_ok = mean_reprojection
        .is_some_and(|error| error.is_finite() && error <= max_mean_reprojection_px + 1.0e-9);
    let behind_camera_ok = behind_camera_ratio.is_some_and(|ratio| ratio <= 0.01);
    if !reprojection_ok || !behind_camera_ok || !map.validate().is_valid() {
        *map = original_map;
        *graph = original_graph;
        return (false, mean_reprojection, None, behind_camera_ratio);
    }
    (true, mean_reprojection, None, behind_camera_ratio)
}

fn correct_loop_welding_region(
    map: &mut VisualMap,
    matched_keyframe_id: u64,
    query_keyframe_id: u64,
    matched_to_query: &SE3,
    config: &CovisibilityLocalBaConfig,
) -> Option<LoopWeldingRegionCorrection> {
    if matched_keyframe_id == query_keyframe_id {
        return None;
    }
    let matched_pose = map
        .keyframes
        .get(&matched_keyframe_id)?
        .frame
        .pose
        .as_ref()?
        .clone();
    let query_pose = map
        .keyframes
        .get(&query_keyframe_id)?
        .frame
        .pose
        .as_ref()?
        .clone();
    let selection = select_covisibility_local_ba_window(map, query_keyframe_id, config).ok()?;

    let desired_query_world_to_camera = matched_to_query.compose(&matched_pose.world_to_camera);
    let world_correction = desired_query_world_to_camera
        .inverse()
        .compose(&query_pose.world_to_camera);
    if !world_correction
        .translation
        .iter()
        .all(|value| value.is_finite())
    {
        return None;
    }

    let mut corrected_keyframe_ids = selection.optimized_keyframe_ids.clone();
    corrected_keyframe_ids.extend(selection.fixed_keyframe_ids.iter().copied());
    corrected_keyframe_ids.retain(|id| *id != matched_keyframe_id);
    corrected_keyframe_ids.sort_unstable();
    corrected_keyframe_ids.dedup();
    if !corrected_keyframe_ids.contains(&query_keyframe_id) {
        return None;
    }
    let inverse_correction = world_correction.inverse();
    for id in &corrected_keyframe_ids {
        let pose = map.keyframes.get_mut(id)?.frame.pose.as_mut()?;
        pose.world_to_camera = pose.world_to_camera.compose(&inverse_correction);
    }

    let rotation = world_correction.rotation.to_rotation_matrix().into_inner();
    for landmark_id in selection.landmark_ids {
        let landmark = map.landmarks.get_mut(&landmark_id)?;
        landmark.position = world_correction.transform_point(&landmark.position);
        if let Some(covariance) = map.landmark_position_covariances.get_mut(&landmark_id) {
            *covariance = rotation * *covariance * rotation.transpose();
        }
    }
    Some(LoopWeldingRegionCorrection {
        keyframe_ids: corrected_keyframe_ids,
        translation_meters: world_correction.translation.norm(),
        rotation_radians: world_correction.rotation.angle(),
    })
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct LoopObservationFusionStats {
    pairs_considered: usize,
    observations_inserted: usize,
    observations_reassigned: usize,
    pairs_skipped: usize,
    reprojection_rejected: usize,
}

/// Persist geometrically verified appearance-loop correspondences on both
/// mirrored map indices. One query keypoint may describe only one landmark.
/// If local mapping associated it with a duplicated recent landmark, move that
/// single observation relation to the older loop landmark; do not delete or
/// geometrically merge the duplicate's other track here.
fn fuse_loop_observations(
    map: &mut VisualMap,
    camera: &Camera,
    query_keyframe_id: u64,
    pairs: &[(usize, u64)],
    max_reprojection_error_px: f64,
) -> LoopObservationFusionStats {
    let mut stats = LoopObservationFusionStats::default();
    for &(keypoint_index, loop_landmark_id) in pairs {
        stats.pairs_considered += 1;
        let Some(keyframe) = map.keyframes.get(&query_keyframe_id) else {
            stats.pairs_skipped += 1;
            continue;
        };
        let Some(&xy) = keyframe.frame.keypoints.get(keypoint_index) else {
            stats.pairs_skipped += 1;
            continue;
        };
        let Some(loop_landmark) = map.landmarks.get(&loop_landmark_id) else {
            stats.pairs_skipped += 1;
            continue;
        };
        let Some(query_pose) = keyframe.frame.pose.as_ref() else {
            stats.pairs_skipped += 1;
            continue;
        };
        let Some(predicted) =
            camera.project(&query_pose.transform_world_point(&loop_landmark.position))
        else {
            stats.pairs_skipped += 1;
            stats.reprojection_rejected += 1;
            continue;
        };
        if !max_reprojection_error_px.is_finite()
            || max_reprojection_error_px <= 0.0
            || (predicted - xy).norm() > max_reprojection_error_px
        {
            stats.pairs_skipped += 1;
            stats.reprojection_rejected += 1;
            continue;
        }

        let existing = keyframe
            .observations
            .iter()
            .find(|observation| observation.keypoint_index == keypoint_index)
            .cloned();
        if existing
            .as_ref()
            .is_some_and(|observation| observation.landmark_id == loop_landmark_id)
        {
            continue;
        }
        // A single landmark must not acquire two different keypoints in one
        // frame. Such a case needs an explicit duplicate/track merge instead
        // of silently manufacturing a second measurement.
        if map.landmarks[&loop_landmark_id]
            .observations
            .iter()
            .any(|observation| {
                observation.frame_id == query_keyframe_id
                    && observation.keypoint_index != keypoint_index
            })
        {
            stats.pairs_skipped += 1;
            continue;
        }

        let observation = Observation {
            frame_id: query_keyframe_id,
            landmark_id: loop_landmark_id,
            keypoint_index,
            xy,
        };
        let keyframe = map
            .keyframes
            .get_mut(&query_keyframe_id)
            .expect("query keyframe existence checked above");
        keyframe
            .observations
            .retain(|candidate| candidate.keypoint_index != keypoint_index);
        keyframe.observations.push(observation.clone());

        if let Some(previous) = existing {
            if previous.landmark_id != loop_landmark_id {
                // The stored score described the original descriptor match.
                // A loop-verification reassignment is a different geometric
                // edge and must not inherit that frontend confidence.
                map.remove_observation_confidence(&previous);
                if let Some(previous_landmark) = map.landmarks.get_mut(&previous.landmark_id) {
                    previous_landmark.observations.retain(|candidate| {
                        !(candidate.frame_id == query_keyframe_id
                            && candidate.keypoint_index == keypoint_index)
                    });
                }
                let loop_stereo_exists = map.stereo_observations.iter().any(|stereo| {
                    stereo.frame_id == query_keyframe_id && stereo.landmark_id == loop_landmark_id
                });
                if loop_stereo_exists {
                    map.stereo_observations.retain(|stereo| {
                        !(stereo.frame_id == query_keyframe_id
                            && stereo.landmark_id == previous.landmark_id)
                    });
                } else if let Some(stereo) = map.stereo_observations.iter_mut().find(|stereo| {
                    stereo.frame_id == query_keyframe_id
                        && stereo.landmark_id == previous.landmark_id
                }) {
                    stereo.landmark_id = loop_landmark_id;
                }
                stats.observations_reassigned += 1;
            }
        }
        let loop_landmark = map
            .landmarks
            .get_mut(&loop_landmark_id)
            .expect("loop landmark existence checked above");
        loop_landmark.observations.retain(|candidate| {
            !(candidate.frame_id == query_keyframe_id && candidate.keypoint_index == keypoint_index)
        });
        loop_landmark.observations.push(observation);
        stats.observations_inserted += 1;
    }
    stats
}

fn keyframe_from_tracking_result(frame: &Frame, tracking: &TrackingResult) -> Keyframe {
    let mut frame = frame.clone();
    frame.pose = tracking.localization.pose.clone();

    let observations = tracking
        .localization
        .inlier_query_indices
        .iter()
        .zip(tracking.localization.inlier_landmark_ids.iter())
        .filter_map(|(keypoint_index, landmark_id)| {
            frame.keypoints.get(*keypoint_index).map(|xy| Observation {
                frame_id: frame.id,
                landmark_id: *landmark_id,
                keypoint_index: *keypoint_index,
                xy: *xy,
            })
        })
        .collect();

    Keyframe {
        frame,
        observations,
    }
}

/// Preserve learned matcher confidence after the mapper has accepted a
/// tracking-produced keyframe. The geometric observation remains the source of
/// truth; missing/invalid confidence simply leaves the observation uniformly
/// weighted in downstream BA.
fn record_tracking_observation_confidences(
    map: &mut VisualMap,
    frame_id: u64,
    tracking: &TrackingResult,
) -> usize {
    let accepted = tracking
        .localization
        .inlier_query_indices
        .iter()
        .copied()
        .zip(tracking.localization.inlier_landmark_ids.iter().copied())
        .zip(tracking.localization.inlier_confidences.iter().copied())
        .filter_map(|((keypoint_index, landmark_id), confidence)| {
            confidence.map(|confidence| (keypoint_index, landmark_id, confidence))
        })
        .collect::<Vec<_>>();

    let Some(keyframe) = map.keyframes.get(&frame_id) else {
        return 0;
    };
    let observations = keyframe.observations.clone();
    let mut recorded = 0usize;
    for (keypoint_index, landmark_id, confidence) in accepted {
        let Some(observation) = observations.iter().find(|observation| {
            observation.keypoint_index == keypoint_index && observation.landmark_id == landmark_id
        }) else {
            continue;
        };
        if map.set_observation_confidence(observation, confidence) {
            recorded += 1;
        }
    }
    recorded
}

#[cfg(test)]
mod observation_confidence_transfer_tests {
    use super::*;
    use nalgebra::Point2;
    use visloc_core::types::{LocalizationResult, LocalizationSuccess};
    use visloc_localization::MapProviderStats;
    use visloc_tracking::{TrackingEvent, TrackingState};

    #[test]
    fn accepted_keyframe_keeps_frontend_inlier_confidence() {
        let mut frame = Frame::new(7, 1);
        frame.keypoints.push(Point2::new(12.0, 34.0));
        let localization = LocalizationResult::success(LocalizationSuccess {
            pose: Pose::identity(),
            candidate_landmark_count: 1,
            match_count: 1,
            correspondence_count: 1,
            inliers: vec![0],
            inlier_query_indices: vec![0],
            inlier_landmark_ids: vec![42],
            inlier_confidences: vec![Some(0.3)],
            inlier_reprojection_errors: vec![0.0],
            mean_reprojection_error: 0.0,
            median_reprojection_error: 0.0,
            max_reprojection_error: 0.0,
        });
        let tracking = TrackingResult {
            frame_id: frame.id,
            state: TrackingState::Tracking,
            event: TrackingEvent::Tracked,
            successive_failures: 0,
            pose_prior: None,
            used_pose_prior: false,
            used_external_localization_prior: false,
            external_localization_prior_radius: None,
            tracking_failure_reason: None,
            map_landmark_count: 1,
            map_stats: MapProviderStats::default(),
            localization,
            covisibility_local_map_size: None,
        };
        let keyframe = keyframe_from_tracking_result(&frame, &tracking);
        let observation = keyframe.observations[0].clone();
        let mut map = VisualMap::new();
        map.keyframes.insert(frame.id, keyframe);

        assert_eq!(
            record_tracking_observation_confidences(&mut map, frame.id, &tracking),
            1
        );
        assert_eq!(map.observation_confidence(&observation), Some(0.3));
    }
}
