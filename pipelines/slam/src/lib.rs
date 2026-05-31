#![forbid(unsafe_code)]
//! Minimal online SLAM orchestration.
//!
//! This crate wires tracking and local mapping together. It is not a full SLAM
//! system, but it now hosts both a sparse pose-graph optimizer (translation-
//! only and full SE(3), with dense or sparse Cholesky solves) and a Schur-
//! complement bundle adjustment that jointly refines poses and landmarks
//! from 2D reprojection residuals.

pub mod bundle;
pub use bundle::{
    BaConfig, BaError, BaGncResult, BaIterationStats, BaObservation, BaResult, BaStereoObservation,
    BiasRandomWalkFactor, BundleAdjustment, BundleAdjustmentRefiner, GravityPrior,
    PairwisePoseFactor, PerPoseGravityObservation, PerPoseGravityPrior, PositionPrior,
    PositionPriorObservation,
};

pub mod imu_preintegration;
pub use imu_preintegration::{ImuPreintegratedDelta, ImuPreintegrationFactor, ImuPreintegrator};

pub mod g2o;
pub use g2o::{read_g2o, write_g2o, G2oError};

mod block_cholesky;
pub mod covariance;
pub mod gnc;
pub mod pcm;
mod reordering;

pub mod sim3_pose_graph;
pub use sim3_pose_graph::{
    Sim3Edge, Sim3Information, Sim3PoseGraph, Sim3PoseGraphConfig, Sim3PoseGraphIterationStats,
    Sim3PoseGraphResult,
};

pub mod stereo_vo_ba;
pub use stereo_vo_ba::{
    parse_stereo_vo_imu_samples_txt, refine_stereo_vo_with_ba, slice_imu_samples_for_keyframes,
    LandmarkInit, StereoVoBaConfig, StereoVoBaError, StereoVoBaImuInput, StereoVoBaImuRefinement,
    StereoVoBaImuSample, StereoVoBaRefinement,
};

pub mod online_stereo_vo_ba;
pub use online_stereo_vo_ba::{
    online_ba_imu_state_rows, write_online_ba_imu_state_csv, OnlineBaImuStateRow,
    OnlineBaTriggerStats, OnlineStereoVoBa, OnlineStereoVoBaConfig,
};

pub mod online_slam_vi_ba;
pub use online_slam_vi_ba::{
    estimate_scale_from_factors, run_inertial_only_vi_ba, run_local_vi_ba,
    run_viba2_inertial_with_scale, InertialOnlyViBaStats, KeyframeImuState,
    OnlineSlamLocalBaConfig, OnlineSlamLocalBaState, OnlineSlamLocalBaStats, Viba2Config,
    Viba2Stats,
};

pub mod vi_initializer;
pub use vi_initializer::{
    StationaryRejectionReason, VisualInertialInitializationResult, VisualInertialInitializer,
    VisualInertialInitializerConfig,
};

pub mod vi_motion_initializer;
pub use vi_motion_initializer::{
    MotionBasedViInitializationResult, MotionBasedViInitializationStatus, MotionBasedViInitializer,
    MotionBasedViInitializerConfig, MotionBasedViRejectionReason,
};

pub mod online_slam_vi_init;
use online_slam_vi_init::OnlineSlamViInitState;
pub use online_slam_vi_init::{
    OnlineSlamConfigError, OnlineSlamViInitConfig, ViInitFallback, ViInitializationEvent,
    ViInitializationStatus,
};

pub mod online_slam_motion_vi_init;
use online_slam_motion_vi_init::OnlineSlamMotionViInitState;
pub use online_slam_motion_vi_init::{
    MotionViInitializationEvent, MotionViInitializationStatus, OnlineSlamMotionViInitConfig,
};

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::path::Path;

use nalgebra::{
    DMatrix, DVector, Matrix3, Matrix6, Point2, Point3, Quaternion, Rotation3, UnitQuaternion,
    Vector3, Vector6,
};
use visloc_core::geometry::{Pose, SE3};
use visloc_core::types::{Camera, Frame, Keyframe, Observation, VisualMap};
use visloc_localization::LocalizationPipeline;
use visloc_mapping::{
    AppliedMapUpdate, KeyframePolicy, LandmarkCandidate, LinearTriangulator, LocalMappingPipeline,
    LocalMappingResult, SimpleKeyframePolicy, Triangulator,
};
use visloc_tracking::{
    ConstantPoseMotionModel, FrameLocalizer, MotionModel, Tracker, TrackingConfig, TrackingResult,
};
use visloc_vision::features::FeatureSet;
use visloc_vision::matching::Matcher;
use visloc_vision::pnp::Correspondence2D3D;
use visloc_vision::ransac::{PnPRansac, RobustPoseEstimator};
use visloc_vision::two_view::{
    EightPointEssentialMatrixEstimator, RelativePoseEstimator, TwoViewCorrespondence,
};

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
    /// Optional motion-based VI init stage that fires AFTER `vi_init`
    /// has succeeded and the body has moved enough to give the IMU
    /// translational excitation. Refines per-keyframe
    /// `(R_w←b, v_w, b_g, b_a)` against IMU pre-integration factors only
    /// (VIBA1, stereo / known-scale path; see
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
            if let Some(imu) = &self.imu {
                if (motion.initializer.gravity_world - imu.gravity_world).norm() > 1.0e-12 {
                    return Err(OnlineSlamConfigError::MotionGravityMismatch {
                        imu_gravity_world: imu.gravity_world,
                        motion_gravity_world: motion.initializer.gravity_world,
                    });
                }
            }
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
            vi_init: None,
            vi_motion_init: None,
            keep_pre_promotion_imu_factors: false,
            pose_graph_refinement: None,
            relocalization: None,
        }
    }
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
}

impl Default for OnlineSlamRelocalizationConfig {
    fn default() -> Self {
        Self {
            min_inliers: 20,
            min_inlier_ratio: 0.3,
            max_mean_reprojection_error: Some(8.0),
            pose_prior_candidate_radius_meters: None,
            recent_keyframe_window: None,
            max_translation_from_imu_prediction_meters: None,
        }
    }
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
    pub last_attempt_frame_id: Option<u64>,
    pub last_success_frame_id: Option<u64>,
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
            && self.last_attempt_frame_id == other.last_attempt_frame_id
            && self.last_success_frame_id == other.last_success_frame_id
    }
}

impl OnlineSlamRelocalizationState {
    fn new(config: OnlineSlamRelocalizationConfig) -> Self {
        Self {
            config,
            localizer: LocalizationPipeline::default(),
            trigger_count: 0,
            success_count: 0,
            last_attempt_frame_id: None,
            last_success_frame_id: None,
        }
    }

    fn reset(&mut self) {
        self.trigger_count = 0;
        self.success_count = 0;
        self.last_attempt_frame_id = None;
        self.last_success_frame_id = None;
    }
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
}

/// Per-session configuration for the online loop-closure + pose-graph
/// refinement stage owned by [`OnlineSlamPipeline`]. When attached via
/// [`OnlineSlamConfig::pose_graph_refinement`], the pipeline maintains a
/// running [`PoseGraph`] mirror of `map.keyframes`, runs an
/// [`EssentialMatrixLoopClosureVerifier`] on every candidate emitted by
/// `detect_loop_closure_candidates`, and folds verified
/// [`LoopClosureConstraint`]s into the graph. When
/// `trigger_every_new_constraints` new verified edges have accumulated
/// since the last solve, [`PoseGraph::optimize_se3_iterative`] runs and
/// the optimised keyframe poses are written back into the map.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineSlamLoopClosureRefinementConfig {
    /// Camera intrinsics passed to the loop-closure verifier when it
    /// builds essential-matrix correspondences. Must match the camera
    /// that produced the keyframes' keypoints. Single-monocular for
    /// now — per-frame intrinsics are out of scope for the first
    /// version of this stage.
    pub camera: Camera,
    /// Thresholds (`min_inliers`, `min_inlier_ratio`,
    /// `max_mean_sampson_error`, `default_translation_scale`) handed to
    /// the [`EssentialMatrixLoopClosureVerifier`] every frame.
    pub verifier_config: LoopClosureVerifierConfig,
    /// SE(3) Gauss-Newton settings consumed by
    /// [`PoseGraph::optimize_se3_iterative`] (or, when `gnc` is `Some`, the
    /// shared SE(3) settings consumed by [`PoseGraph::optimize_se3_gnc`])
    /// when the trigger fires.
    pub pose_graph_config: PoseGraphSe3Config,
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
    /// Minimum number of *new* verified loop-closure constraints that
    /// must accumulate before a fresh pose-graph solve runs. Clamped to
    /// at least `1`; `1` runs PGO on every accepted loop edge, higher
    /// values batch.
    pub trigger_every_new_constraints: usize,
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
    /// Keyframe ids in the order they were first registered. The first
    /// entry is the [`PoseGraph::anchor`]; subsequent entries form the
    /// sequential edge chain.
    pub keyframe_order: Vec<u64>,
    /// All verified loop-closure constraints accumulated to date.
    pub verified_constraints: Vec<LoopClosureConstraint>,
    /// New verified constraints since the last successful PGO trigger.
    /// Reset to `0` after each fired solve.
    pub pending_since_last_trigger: usize,
    /// Total number of [`PoseGraph::optimize_se3_iterative`] calls fired
    /// by the pipeline since construction (counts both converged and
    /// not-converged solves; mismatches between the two go to
    /// `OnlineSlamLoopClosureRefinementStats::pose_graph_result`).
    pub trigger_count: u64,
}

impl OnlineSlamLoopClosureRefinementState {
    fn new(config: OnlineSlamLoopClosureRefinementConfig) -> Self {
        Self {
            config,
            graph: PoseGraph::new(),
            keyframe_order: Vec::new(),
            verified_constraints: Vec::new(),
            pending_since_last_trigger: 0,
            trigger_count: 0,
        }
    }

    fn reset(&mut self) {
        self.graph = PoseGraph::new();
        self.keyframe_order.clear();
        self.verified_constraints.clear();
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
    /// Number of *loop-closure* edges the GNC solver drove below the inlier
    /// threshold (weight `< 0.5`) on the solve fired this frame — the
    /// verified-but-wrong closures caught and rejected at the back-end.
    /// Always `0` on the plain iterative path (`gnc` unset).
    pub loop_closures_rejected: usize,
    /// Number of verified loop closures the PCM front-end screen rejected this
    /// frame *before* they entered the graph (geometrically inconsistent with
    /// the established set). Always `0` when `pcm` is unset.
    pub loop_closures_pcm_rejected: usize,
    /// Number of `map.keyframes[id].frame.pose` slots overwritten with
    /// the optimised pose after PGO. Zero unless a solve fired this frame
    /// (`pose_graph_result.is_some()` or `gnc_result.is_some()`).
    pub keyframes_updated: usize,
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
        let preintegrator = ImuPreintegrator::new_with_bias(config.bias_gyro, config.bias_acc);
        Self {
            config,
            preintegrator,
            last_keyframe_id: None,
            pending_factor: None,
        }
    }

    fn reset(&mut self) {
        self.preintegrator =
            ImuPreintegrator::new_with_bias(self.config.bias_gyro, self.config.bias_acc);
        self.last_keyframe_id = None;
        self.pending_factor = None;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopClosureConfig {
    pub enabled: bool,
    pub min_frame_id_gap: u64,
    pub min_shared_landmarks: usize,
    pub min_shared_landmark_ratio_percent: u8,
    pub max_candidates: usize,
}

impl Default for LoopClosureConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_frame_id_gap: 5,
            min_shared_landmarks: 12,
            min_shared_landmark_ratio_percent: 40,
            max_candidates: 5,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoopClosureCandidate {
    pub query_frame_id: u64,
    pub matched_keyframe_id: u64,
    pub shared_landmark_count: usize,
    pub query_inlier_count: usize,
    pub keyframe_observation_count: usize,
    pub shared_landmark_ratio: f64,
    pub score: f64,
    /// `true` while the candidate has not been rejected by an explicit
    /// verifier. When [`verify_loop_closure_candidates`] runs, this becomes
    /// `LoopClosureVerification::verified`.
    pub geometrically_verified: bool,
    /// Optional verifier output. `Some` when [`verify_loop_closure_candidates`]
    /// (or another caller) has explicitly run a [`LoopClosureVerifier`] over
    /// the candidate; `None` when only the shared-landmark heuristic has
    /// produced the candidate.
    pub verification: Option<LoopClosureVerification>,
}

/// Configuration thresholds for [`EssentialMatrixLoopClosureVerifier`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopClosureVerifierConfig {
    /// Minimum number of inliers an essential-matrix RANSAC fit must produce
    /// for the candidate to be accepted.
    pub min_inliers: usize,
    /// Minimum inlier ratio (inliers / supplied correspondences) for
    /// acceptance.
    pub min_inlier_ratio: f64,
    /// Maximum allowed mean Sampson distance, in normalized image-plane units
    /// (multiply by focal length to convert to pixels).
    pub max_mean_sampson_error: f64,
    /// Translation scale applied when recovering the relative pose. Two-view
    /// geometry leaves translation up to scale; this default is what
    /// [`LoopClosureVerification::relative_pose`] uses unless callers wrap
    /// the verifier with their own scale source. Defaults to `1.0`.
    pub default_translation_scale: f64,
}

impl Default for LoopClosureVerifierConfig {
    fn default() -> Self {
        Self {
            min_inliers: 8,
            min_inlier_ratio: 0.5,
            max_mean_sampson_error: 5.0e-3,
            default_translation_scale: 1.0,
        }
    }
}

/// Reason a [`LoopClosureVerification`] rejected a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopClosureVerificationFailureReason {
    /// Fewer correspondences than the verifier's minimum requirement.
    InsufficientCorrespondences,
    /// The essential-matrix RANSAC failed to find a consensus.
    EssentialEstimationFailed,
    /// The RANSAC produced fewer inliers than `min_inliers`.
    TooFewInliers,
    /// The inlier ratio fell below `min_inlier_ratio`.
    LowInlierRatio,
    /// Mean Sampson error exceeded `max_mean_sampson_error`.
    HighSampsonError,
    /// The hybrid verifier ran both backends successfully, but the recovered
    /// essential-matrix and PnP relative poses disagreed beyond the configured
    /// translation-direction or rotation tolerances.
    PoseDisagreement,
}

/// Output of running a [`LoopClosureVerifier`] on a candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct LoopClosureVerification {
    pub verified: bool,
    pub correspondence_count: usize,
    pub inlier_count: usize,
    pub inlier_ratio: f64,
    /// Mean Sampson distance reported by an essential-matrix verifier (in
    /// normalized image-plane units). `0.0` and uninformative for
    /// PnP-based verifiers; check [`Self::mean_reprojection_error_px`] in
    /// that case.
    pub mean_sampson_error: f64,
    pub score: f64,
    pub failure_reason: Option<LoopClosureVerificationFailureReason>,
    /// Recovered relative pose (older keyframe → current frame) when the
    /// underlying RANSAC converged. `Some` even for non-`verified` cases as
    /// long as a pose was recovered; consult `verified` and `failure_reason`
    /// before consuming. For essential-matrix verifiers the translation is
    /// scaled by [`LoopClosureVerifierConfig::default_translation_scale`];
    /// for PnP verifiers it is in metric units (the keyframe pose carries
    /// the world scale).
    pub relative_pose: Option<SE3>,
    /// Mean reprojection error (in pixels) reported by a PnP-based verifier.
    /// `None` for essential-matrix verifiers.
    pub mean_reprojection_error_px: Option<f64>,
}

/// Trait for a loop-closure candidate verifier. Concrete implementations
/// receive 2D-2D correspondences in pixel coordinates between the older
/// keyframe (`previous_xy`) and the current/query frame (`current_xy`) plus
/// the shared camera intrinsics.
pub trait LoopClosureVerifier {
    fn verify(
        &self,
        correspondences: &[TwoViewCorrespondence],
        camera: &Camera,
    ) -> LoopClosureVerification;

    /// Confidence-aware variant: when `weights[i]` is high, correspondence
    /// `i` is preferred during RANSAC sampling (PROSAC-style). Default
    /// implementation falls back to the unweighted `verify` so existing
    /// implementors don't need to change. Verifiers backed by RANSAC can
    /// override to thread weights into `EssentialRansac::estimate_with_weights`.
    fn verify_with_weights(
        &self,
        correspondences: &[TwoViewCorrespondence],
        weights: Option<&[f32]>,
        camera: &Camera,
    ) -> LoopClosureVerification {
        let _ = weights;
        self.verify(correspondences, camera)
    }
}

/// Geometric verifier that runs the classical essential-matrix RANSAC from
/// `visloc-vision::two_view` on the supplied correspondences and reports
/// inlier statistics, mean Sampson error, and a combined score.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EssentialMatrixLoopClosureVerifier {
    pub estimator: RelativePoseEstimator<EightPointEssentialMatrixEstimator>,
    pub config: LoopClosureVerifierConfig,
}

impl EssentialMatrixLoopClosureVerifier {
    pub fn new(
        estimator: RelativePoseEstimator<EightPointEssentialMatrixEstimator>,
        config: LoopClosureVerifierConfig,
    ) -> Self {
        Self { estimator, config }
    }
}

impl LoopClosureVerifier for EssentialMatrixLoopClosureVerifier {
    fn verify(
        &self,
        correspondences: &[TwoViewCorrespondence],
        camera: &Camera,
    ) -> LoopClosureVerification {
        self.verify_with_weights(correspondences, None, camera)
    }

    fn verify_with_weights(
        &self,
        correspondences: &[TwoViewCorrespondence],
        weights: Option<&[f32]>,
        camera: &Camera,
    ) -> LoopClosureVerification {
        let correspondence_count = correspondences.len();
        let minimum = self
            .estimator
            .ransac
            .estimator
            .min_correspondences
            .max(self.config.min_inliers);
        if correspondence_count < minimum {
            return LoopClosureVerification {
                verified: false,
                correspondence_count,
                inlier_count: 0,
                inlier_ratio: 0.0,
                mean_sampson_error: f64::INFINITY,
                score: 0.0,
                failure_reason: Some(
                    LoopClosureVerificationFailureReason::InsufficientCorrespondences,
                ),
                relative_pose: None,
                mean_reprojection_error_px: None,
            };
        }

        let relative_pose = match weights {
            Some(w) if w.len() == correspondences.len() => {
                self.estimator.estimate_with_scale_and_weights(
                    correspondences,
                    camera,
                    self.config.default_translation_scale,
                    w,
                )
            }
            _ => self.estimator.estimate_with_scale(
                correspondences,
                camera,
                self.config.default_translation_scale,
            ),
        };
        let Some(relative_pose) = relative_pose else {
            return LoopClosureVerification {
                verified: false,
                correspondence_count,
                inlier_count: 0,
                inlier_ratio: 0.0,
                mean_sampson_error: f64::INFINITY,
                score: 0.0,
                failure_reason: Some(
                    LoopClosureVerificationFailureReason::EssentialEstimationFailed,
                ),
                relative_pose: None,
                mean_reprojection_error_px: None,
            };
        };

        let inlier_count = relative_pose.inliers.len();
        let inlier_ratio = inlier_count as f64 / correspondence_count as f64;
        let mean_sampson = relative_pose.mean_sampson_error;

        let mut failure_reason = None;
        if inlier_count < self.config.min_inliers {
            failure_reason = Some(LoopClosureVerificationFailureReason::TooFewInliers);
        } else if inlier_ratio < self.config.min_inlier_ratio {
            failure_reason = Some(LoopClosureVerificationFailureReason::LowInlierRatio);
        } else if mean_sampson > self.config.max_mean_sampson_error {
            failure_reason = Some(LoopClosureVerificationFailureReason::HighSampsonError);
        }
        let verified = failure_reason.is_none();
        let inlier_volume = inlier_ratio * inlier_count as f64;
        let denominator = mean_sampson.max(1.0e-6);
        let score = if denominator.is_finite() {
            inlier_volume / denominator
        } else {
            inlier_volume
        };
        LoopClosureVerification {
            verified,
            correspondence_count,
            inlier_count,
            inlier_ratio,
            mean_sampson_error: mean_sampson,
            score,
            failure_reason,
            relative_pose: Some(relative_pose.previous_to_current),
            mean_reprojection_error_px: None,
        }
    }
}

/// Configuration thresholds for [`PnPLoopClosureVerifier`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PnPLoopClosureVerifierConfig {
    /// Minimum number of PnP RANSAC inliers an accepted candidate must
    /// produce.
    pub min_inliers: usize,
    /// Minimum inlier ratio (inliers / supplied 2D-3D correspondences).
    pub min_inlier_ratio: f64,
    /// Maximum allowed mean reprojection error (in pixels) for inliers.
    pub max_mean_reprojection_error_px: f64,
}

impl Default for PnPLoopClosureVerifierConfig {
    fn default() -> Self {
        Self {
            min_inliers: 8,
            min_inlier_ratio: 0.5,
            max_mean_reprojection_error_px: 4.0,
        }
    }
}

/// PnP-based loop-closure verifier. Reuses the project's [`PnPRansac`] to
/// re-localize the current frame against landmarks observed by the candidate
/// keyframe; if the recovered absolute pose has enough inliers and a small
/// reprojection error, the candidate is accepted and the relative pose
/// (older keyframe → current frame) is filled into
/// [`LoopClosureVerification::relative_pose`].
///
/// Compared with [`EssentialMatrixLoopClosureVerifier`], this verifier:
///
/// - operates on 2D-3D correspondences instead of 2D-2D, so it checks the
///   candidate against the actual 3D map structure rather than two-view
///   geometry alone;
/// - returns metric translations (the keyframe pose carries the world scale),
///   so callers do not need to plug in a separate `default_translation_scale`;
/// - is preferable when the older keyframe has sufficient triangulated
///   landmarks visible from the current frame, which is the common case for
///   in-map loop closures.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PnPLoopClosureVerifier<R = PnPRansac> {
    pub ransac: R,
    pub config: PnPLoopClosureVerifierConfig,
}

impl<R> PnPLoopClosureVerifier<R>
where
    R: RobustPoseEstimator,
{
    pub fn new(ransac: R, config: PnPLoopClosureVerifierConfig) -> Self {
        Self { ransac, config }
    }

    /// Run PnP RANSAC on `correspondences` and turn the report into a
    /// [`LoopClosureVerification`]. `keyframe_pose` is the older keyframe's
    /// stored `world_to_camera` SE3; the recovered current-frame pose is
    /// composed with its inverse to populate `relative_pose`.
    pub fn verify(
        &self,
        correspondences: &[Correspondence2D3D],
        keyframe_pose: &Pose,
        camera: &Camera,
    ) -> LoopClosureVerification {
        let correspondence_count = correspondences.len();
        if correspondence_count < self.config.min_inliers {
            return LoopClosureVerification {
                verified: false,
                correspondence_count,
                inlier_count: 0,
                inlier_ratio: 0.0,
                mean_sampson_error: 0.0,
                score: 0.0,
                failure_reason: Some(
                    LoopClosureVerificationFailureReason::InsufficientCorrespondences,
                ),
                relative_pose: None,
                mean_reprojection_error_px: None,
            };
        }

        let Some(report) = self.ransac.estimate(correspondences, camera) else {
            return LoopClosureVerification {
                verified: false,
                correspondence_count,
                inlier_count: 0,
                inlier_ratio: 0.0,
                mean_sampson_error: 0.0,
                score: 0.0,
                failure_reason: Some(
                    LoopClosureVerificationFailureReason::EssentialEstimationFailed,
                ),
                relative_pose: None,
                mean_reprojection_error_px: None,
            };
        };

        let inlier_count = report.inliers.len();
        let inlier_ratio = inlier_count as f64 / correspondence_count as f64;
        let mean_reprojection_error_px = report.mean_reprojection_error;

        let mut failure_reason = None;
        if inlier_count < self.config.min_inliers {
            failure_reason = Some(LoopClosureVerificationFailureReason::TooFewInliers);
        } else if inlier_ratio < self.config.min_inlier_ratio {
            failure_reason = Some(LoopClosureVerificationFailureReason::LowInlierRatio);
        } else if mean_reprojection_error_px > self.config.max_mean_reprojection_error_px {
            failure_reason = Some(LoopClosureVerificationFailureReason::HighSampsonError);
        }
        let verified = failure_reason.is_none();
        let inlier_volume = inlier_ratio * inlier_count as f64;
        let denominator = mean_reprojection_error_px.max(1.0e-6);
        let score = if denominator.is_finite() {
            inlier_volume / denominator
        } else {
            inlier_volume
        };
        let relative_pose = report
            .pose
            .world_to_camera
            .compose(&keyframe_pose.world_to_camera.inverse());
        LoopClosureVerification {
            verified,
            correspondence_count,
            inlier_count,
            inlier_ratio,
            mean_sampson_error: 0.0,
            score,
            failure_reason,
            relative_pose: Some(relative_pose),
            mean_reprojection_error_px: Some(mean_reprojection_error_px),
        }
    }
}

/// Build 2D-3D correspondences for a loop-closure candidate by intersecting
/// the current frame's tracking inliers with the older keyframe's observed
/// landmarks. Each shared landmark contributes one entry pairing the current
/// frame's pixel observation with the landmark's world position.
pub fn correspondences_2d3d_for_loop_candidate(
    current_frame: &Frame,
    current_inlier_query_indices: &[usize],
    current_inlier_landmark_ids: &[u64],
    keyframe: &Keyframe,
    map: &VisualMap,
) -> Vec<Correspondence2D3D> {
    let keyframe_landmark_ids: HashSet<u64> = keyframe
        .observations
        .iter()
        .map(|observation| observation.landmark_id)
        .collect();
    let mut correspondences = Vec::new();
    for (query_index, landmark_id) in current_inlier_query_indices
        .iter()
        .zip(current_inlier_landmark_ids.iter())
    {
        if !keyframe_landmark_ids.contains(landmark_id) {
            continue;
        }
        let Some(landmark) = map.landmarks.get(landmark_id) else {
            continue;
        };
        let Some(query_xy) = current_frame.keypoints.get(*query_index) else {
            continue;
        };
        correspondences.push(Correspondence2D3D {
            point2d: *query_xy,
            point3d: landmark.position,
            confidence: None,
        });
    }
    correspondences
}

/// Build pixel-space two-view correspondences for a loop-closure candidate
/// from the current frame's tracking inliers and an older keyframe's
/// observations. Each shared landmark id contributes one correspondence
/// `(keyframe_xy, current_xy)`.
pub fn correspondences_for_loop_candidate(
    current_frame: &Frame,
    current_inlier_query_indices: &[usize],
    current_inlier_landmark_ids: &[u64],
    keyframe: &Keyframe,
) -> Vec<TwoViewCorrespondence> {
    let keyframe_lookup: HashMap<u64, Point2<f64>> = keyframe
        .observations
        .iter()
        .map(|observation| (observation.landmark_id, observation.xy))
        .collect();
    let mut correspondences = Vec::new();
    for (query_index, landmark_id) in current_inlier_query_indices
        .iter()
        .zip(current_inlier_landmark_ids.iter())
    {
        let Some(keyframe_xy) = keyframe_lookup.get(landmark_id) else {
            continue;
        };
        let Some(query_xy) = current_frame.keypoints.get(*query_index) else {
            continue;
        };
        correspondences.push(TwoViewCorrespondence {
            previous_xy: *keyframe_xy,
            current_xy: *query_xy,
        });
    }
    correspondences
}

/// Pose-graph-style constraint between two keyframes derived from a verified
/// loop-closure candidate. This is intentionally a lightweight data type — no
/// solver lives in this crate yet — so downstream optimization layers can
/// adopt it without committing to a specific backend.
///
/// `relative_pose` represents the rigid transform that takes a point in
/// `from_keyframe_id`'s camera frame to `to_keyframe_id`'s camera frame, with
/// the translation scaled by the verifier's
/// [`LoopClosureVerifierConfig::default_translation_scale`] (or whatever
/// scale the caller chose to apply before constructing the constraint).
#[derive(Debug, Clone, PartialEq)]
pub struct LoopClosureConstraint {
    pub from_keyframe_id: u64,
    pub to_keyframe_id: u64,
    pub relative_pose: SE3,
    pub inlier_count: usize,
    pub inlier_ratio: f64,
    pub mean_sampson_error: f64,
    pub score: f64,
}

impl LoopClosureConstraint {
    /// Builds a constraint from a verified candidate. Returns `None` when the
    /// candidate has no verifier output, when the verifier rejected it, or
    /// when no relative pose was recovered.
    pub fn from_verified_candidate(
        candidate: &LoopClosureCandidate,
    ) -> Option<LoopClosureConstraint> {
        let verification = candidate.verification.as_ref()?;
        if !verification.verified {
            return None;
        }
        let relative_pose = verification.relative_pose.clone()?;
        Some(LoopClosureConstraint {
            from_keyframe_id: candidate.matched_keyframe_id,
            to_keyframe_id: candidate.query_frame_id,
            relative_pose,
            inlier_count: verification.inlier_count,
            inlier_ratio: verification.inlier_ratio,
            mean_sampson_error: verification.mean_sampson_error,
            score: verification.score,
        })
    }

    /// Lift this loop-closure edge into a [`PairwisePoseFactor`] so it can be
    /// added to a [`BundleAdjustment`] alongside reprojection residuals.
    ///
    /// The relative pose `T_rel = T_to · T_fromⁱ` (the constraint's
    /// `relative_pose`) is used verbatim as the BA measurement; `weight` is
    /// `1 / σ²` for an isotropic SE(3) measurement noise σ. A robust default is
    /// to scale from the verifier's inlier count, e.g.
    /// `weight = (inlier_count as f64) * base_weight`, but the choice is left
    /// to the caller because verifier output differs by backend (essential
    /// matrix vs PnP vs hybrid).
    pub fn to_pairwise_pose_factor(&self, weight: f64) -> PairwisePoseFactor {
        PairwisePoseFactor {
            keyframe_id_from: self.from_keyframe_id,
            keyframe_id_to: self.to_keyframe_id,
            measurement: Pose {
                world_to_camera: self.relative_pose.clone(),
            },
            weight,
        }
    }
}

/// Convenience helper that builds a constraint per verified candidate. Keeps
/// the same ordering as the input slice and silently drops candidates that
/// were not verified or lack a recovered relative pose.
pub fn loop_closure_constraints_from_candidates(
    candidates: &[LoopClosureCandidate],
) -> Vec<LoopClosureConstraint> {
    candidates
        .iter()
        .filter_map(LoopClosureConstraint::from_verified_candidate)
        .collect()
}

/// Convert a slice of verified [`LoopClosureConstraint`]s into BA-ready
/// [`PairwisePoseFactor`]s, all sharing the same scalar `weight`. Convenience
/// wrapper for the common case where every loop edge is treated as
/// equally-informative. For per-edge weighting (e.g. scaling by
/// `inlier_count`), call [`LoopClosureConstraint::to_pairwise_pose_factor`]
/// directly.
pub fn pairwise_pose_factors_from_loop_closures(
    constraints: &[LoopClosureConstraint],
    weight: f64,
) -> Vec<PairwisePoseFactor> {
    constraints
        .iter()
        .map(|c| c.to_pairwise_pose_factor(weight))
        .collect()
}

/// Run `verify_one` on every candidate whose `matched_keyframe_id` still
/// resolves in `map`. Candidates without a matching keyframe are silently
/// skipped (the matched keyframe may have been deleted between candidate
/// generation and verification). When `verify_one` returns `Some`, the
/// candidate is mutated in place: `geometrically_verified` is set to
/// `LoopClosureVerification::verified` and `verification` is set to the
/// returned struct. `None` skips writing.
///
/// This is the shared iteration shell between
/// [`verify_loop_closure_candidates`],
/// [`verify_loop_closure_candidates_pnp`], and
/// [`verify_loop_closure_candidates_hybrid`]. Each public wrapper supplies a
/// closure that builds correspondences and calls its backend; this helper
/// owns the candidate iteration and write-back so the public API stays
/// uniform when new backends are added.
fn verify_each_candidate<F>(
    candidates: &mut [LoopClosureCandidate],
    map: &VisualMap,
    mut verify_one: F,
) where
    F: FnMut(&Keyframe) -> Option<LoopClosureVerification>,
{
    for candidate in candidates.iter_mut() {
        let Some(keyframe) = map.keyframes.get(&candidate.matched_keyframe_id) else {
            continue;
        };
        let Some(verification) = verify_one(keyframe) else {
            continue;
        };
        candidate.geometrically_verified = verification.verified;
        candidate.verification = Some(verification);
    }
}

/// Run `verifier` on every supplied candidate, mutating each
/// [`LoopClosureCandidate`] in place: `verification` is set to the verifier
/// output and `geometrically_verified` is replaced with
/// `LoopClosureVerification::verified`. Candidates whose matched keyframe is
/// no longer in `map` are left untouched.
pub fn verify_loop_closure_candidates<V>(
    candidates: &mut [LoopClosureCandidate],
    current_frame: &Frame,
    tracking: &TrackingResult,
    map: &VisualMap,
    camera: &Camera,
    verifier: &V,
) where
    V: LoopClosureVerifier,
{
    verify_each_candidate(candidates, map, |keyframe| {
        let correspondences = correspondences_for_loop_candidate(
            current_frame,
            &tracking.localization.inlier_query_indices,
            &tracking.localization.inlier_landmark_ids,
            keyframe,
        );
        Some(verifier.verify(&correspondences, camera))
    });
}

/// Run a PnP-based [`PnPLoopClosureVerifier`] on every supplied candidate.
/// For each candidate this builds 2D-3D correspondences via
/// [`correspondences_2d3d_for_loop_candidate`], runs PnP RANSAC on them, and
/// updates `verification` and `geometrically_verified` in place.
/// Candidates whose matched keyframe is no longer in `map`, or whose stored
/// keyframe pose is missing, are left untouched.
pub fn verify_loop_closure_candidates_pnp<R>(
    candidates: &mut [LoopClosureCandidate],
    current_frame: &Frame,
    tracking: &TrackingResult,
    map: &VisualMap,
    camera: &Camera,
    verifier: &PnPLoopClosureVerifier<R>,
) where
    R: RobustPoseEstimator,
{
    verify_each_candidate(candidates, map, |keyframe| {
        let keyframe_pose = keyframe.frame.pose.as_ref()?;
        let correspondences = correspondences_2d3d_for_loop_candidate(
            current_frame,
            &tracking.localization.inlier_query_indices,
            &tracking.localization.inlier_landmark_ids,
            keyframe,
            map,
        );
        Some(verifier.verify(&correspondences, keyframe_pose, camera))
    });
}

/// Configuration for [`HybridLoopClosureVerifier`]: maximum allowed
/// disagreement between the essential-matrix and PnP recovered poses before
/// the hybrid verifier rejects the candidate as inconsistent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HybridLoopClosureVerifierConfig {
    /// Maximum allowed angle (in radians) between the essential and PnP
    /// translation directions. Compared on unit vectors so essential's
    /// scale-up-to-translation ambiguity does not trigger spurious failures.
    pub max_translation_direction_disagreement_rad: f64,
    /// Maximum allowed rotation angle between the essential and PnP rotation
    /// components.
    pub max_rotation_disagreement_rad: f64,
}

impl Default for HybridLoopClosureVerifierConfig {
    fn default() -> Self {
        Self {
            max_translation_direction_disagreement_rad: 0.20,
            max_rotation_disagreement_rad: 0.20,
        }
    }
}

/// Loop-closure verifier that consults both the essential-matrix and PnP
/// backends and reports a consensus decision: the candidate is accepted iff
/// both verifiers accept it AND their recovered relative poses agree to
/// within the configured rotation / translation-direction tolerances. This
/// catches ambiguity where a 2D-2D essential fit looks plausible but
/// disagrees with the 3D map structure (or vice versa).
///
/// The combined [`LoopClosureVerification`] uses the PnP relative pose
/// (metric, no scale parameter), the minimum of both verifiers' inlier
/// counts (conservative), and reports both `mean_sampson_error` and
/// `mean_reprojection_error_px`. When either backend rejects, the failure
/// reason is propagated; if both pass but the poses disagree the failure
/// reason is [`LoopClosureVerificationFailureReason::PoseDisagreement`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HybridLoopClosureVerifier<R = PnPRansac> {
    pub essential: EssentialMatrixLoopClosureVerifier,
    pub pnp: PnPLoopClosureVerifier<R>,
    pub config: HybridLoopClosureVerifierConfig,
}

impl<R> HybridLoopClosureVerifier<R>
where
    R: RobustPoseEstimator,
{
    pub fn new(
        essential: EssentialMatrixLoopClosureVerifier,
        pnp: PnPLoopClosureVerifier<R>,
        config: HybridLoopClosureVerifierConfig,
    ) -> Self {
        Self {
            essential,
            pnp,
            config,
        }
    }
}

/// Run a [`HybridLoopClosureVerifier`] on every supplied candidate. For each
/// candidate this builds both 2D-2D and 2D-3D correspondences, runs the two
/// backends in turn, combines them into a consensus
/// [`LoopClosureVerification`], and writes the result back into
/// `verification` / `geometrically_verified` in place. Candidates whose
/// matched keyframe is no longer in `map` or whose stored keyframe pose is
/// missing are left untouched.
pub fn verify_loop_closure_candidates_hybrid<R>(
    candidates: &mut [LoopClosureCandidate],
    current_frame: &Frame,
    tracking: &TrackingResult,
    map: &VisualMap,
    camera: &Camera,
    verifier: &HybridLoopClosureVerifier<R>,
) where
    R: RobustPoseEstimator,
{
    verify_each_candidate(candidates, map, |keyframe| {
        let keyframe_pose = keyframe.frame.pose.as_ref()?;
        let two_view = correspondences_for_loop_candidate(
            current_frame,
            &tracking.localization.inlier_query_indices,
            &tracking.localization.inlier_landmark_ids,
            keyframe,
        );
        let pnp_corrs = correspondences_2d3d_for_loop_candidate(
            current_frame,
            &tracking.localization.inlier_query_indices,
            &tracking.localization.inlier_landmark_ids,
            keyframe,
            map,
        );
        let essential_v = verifier.essential.verify(&two_view, camera);
        let pnp_v = verifier.pnp.verify(&pnp_corrs, keyframe_pose, camera);
        Some(combine_hybrid_verifications(
            &essential_v,
            &pnp_v,
            &verifier.config,
        ))
    });
}

/// One keyframe's appearance in a pairwise loop-closure scan: the frame id is
/// what the produced [`LoopClosureCandidate`] uses for `query_frame_id` /
/// `matched_keyframe_id`. The keypoint and descriptor slices are typically
/// borrowed from a [`FeatureSet`] (see [`PairwiseKeyframeView::from_features`]).
#[derive(Debug, Clone, Copy)]
pub struct PairwiseKeyframeView<'a> {
    pub frame_id: u64,
    pub keypoints: &'a [Point2<f64>],
    pub descriptors: &'a [Vec<f32>],
}

impl<'a> PairwiseKeyframeView<'a> {
    pub fn from_features(frame_id: u64, features: &'a FeatureSet) -> Self {
        Self {
            frame_id,
            keypoints: &features.keypoints,
            descriptors: &features.descriptors,
        }
    }
}

/// Configuration for [`scan_pairwise_loop_closures`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairwiseLoopClosureScannerConfig {
    /// Minimum frame-id gap between two keyframes for a pair to be eligible.
    /// Pairs `(i, j)` with `keyframes[j].frame_id - keyframes[i].frame_id <
    /// min_keyframe_id_gap` are skipped, so adjacent keyframes (which share
    /// most of their tracks) are never confused for loops.
    pub min_keyframe_id_gap: u64,
    /// Minimum number of descriptor matches a pair needs before its
    /// correspondences are even handed to the verifier. Cheap reject for
    /// keyframes whose appearance has no overlap.
    pub min_matches: usize,
}

impl Default for PairwiseLoopClosureScannerConfig {
    fn default() -> Self {
        Self {
            min_keyframe_id_gap: 20,
            min_matches: 30,
        }
    }
}

/// Walk every keyframe pair `(i, j)` with `i < j` and a sufficient frame-id
/// gap, brute-force match descriptors, and run the verifier on the resulting
/// 2D-2D correspondences. Returns one [`LoopClosureCandidate`] per accepted
/// pair, with `verification` populated and `geometrically_verified = true`.
/// Rejected pairs are dropped (vs the `verify_loop_closure_candidates_*`
/// helpers which mutate caller-supplied candidates in place); this routine is
/// meant for the "I have a list of keyframes and want loop *detection*, not
/// just verification" case.
///
/// `query_frame_id` is the later keyframe and `matched_keyframe_id` is the
/// earlier one — the same convention `LoopClosureConstraint` builders use.
/// `shared_landmark_count` and `keyframe_observation_count` are filled with
/// inlier and total-match counts respectively (no map is consulted), so the
/// candidate is ready for [`LoopClosureConstraint::from_verified_candidate`].
pub fn scan_pairwise_loop_closures<M, V>(
    keyframes: &[PairwiseKeyframeView],
    matcher: &M,
    verifier: &V,
    camera: &Camera,
    config: &PairwiseLoopClosureScannerConfig,
) -> Vec<LoopClosureCandidate>
where
    M: Matcher,
    V: LoopClosureVerifier,
{
    let mut out = Vec::new();
    for i in 0..keyframes.len() {
        for j in (i + 1)..keyframes.len() {
            let from = &keyframes[i];
            let to = &keyframes[j];
            let gap = to.frame_id.saturating_sub(from.frame_id);
            if gap < config.min_keyframe_id_gap {
                continue;
            }
            let matches = matcher.match_descriptors(from.descriptors, to.descriptors);
            if matches.len() < config.min_matches {
                continue;
            }
            let mut correspondences: Vec<TwoViewCorrespondence> = Vec::with_capacity(matches.len());
            let mut weights: Vec<f32> = Vec::with_capacity(matches.len());
            let mut any_confidence = false;
            for m in &matches {
                let Some(prev) = from.keypoints.get(m.query_index) else {
                    continue;
                };
                let Some(curr) = to.keypoints.get(m.train_index) else {
                    continue;
                };
                correspondences.push(TwoViewCorrespondence {
                    previous_xy: *prev,
                    current_xy: *curr,
                });
                if let Some(c) = m.confidence {
                    any_confidence = true;
                    weights.push(c);
                } else {
                    weights.push(1.0);
                }
            }
            if correspondences.len() < config.min_matches {
                continue;
            }
            let weights_slice = if any_confidence {
                Some(weights.as_slice())
            } else {
                None
            };
            let verification =
                verifier.verify_with_weights(&correspondences, weights_slice, camera);
            if !verification.verified {
                continue;
            }
            out.push(LoopClosureCandidate {
                query_frame_id: to.frame_id,
                matched_keyframe_id: from.frame_id,
                shared_landmark_count: verification.inlier_count,
                query_inlier_count: verification.inlier_count,
                keyframe_observation_count: matches.len(),
                shared_landmark_ratio: verification.inlier_ratio,
                score: verification.score,
                geometrically_verified: true,
                verification: Some(verification),
            });
        }
    }
    out
}

fn combine_hybrid_verifications(
    essential: &LoopClosureVerification,
    pnp: &LoopClosureVerification,
    config: &HybridLoopClosureVerifierConfig,
) -> LoopClosureVerification {
    // Inherit the minimum (conservative) inlier count and ratio so the
    // combined diagnostics never overstate either backend's evidence.
    let inlier_count = essential.inlier_count.min(pnp.inlier_count);
    let inlier_ratio = essential.inlier_ratio.min(pnp.inlier_ratio);
    let correspondence_count = essential.correspondence_count.min(pnp.correspondence_count);
    let score = essential.score.min(pnp.score);

    if !essential.verified {
        return LoopClosureVerification {
            verified: false,
            correspondence_count,
            inlier_count,
            inlier_ratio,
            mean_sampson_error: essential.mean_sampson_error,
            score,
            failure_reason: essential.failure_reason,
            relative_pose: pnp.relative_pose.clone(),
            mean_reprojection_error_px: pnp.mean_reprojection_error_px,
        };
    }
    if !pnp.verified {
        return LoopClosureVerification {
            verified: false,
            correspondence_count,
            inlier_count,
            inlier_ratio,
            mean_sampson_error: essential.mean_sampson_error,
            score,
            failure_reason: pnp.failure_reason,
            relative_pose: pnp.relative_pose.clone(),
            mean_reprojection_error_px: pnp.mean_reprojection_error_px,
        };
    }
    // Both verified — check pose agreement.
    let (Some(ess_pose), Some(pnp_pose)) =
        (essential.relative_pose.as_ref(), pnp.relative_pose.as_ref())
    else {
        return LoopClosureVerification {
            verified: false,
            correspondence_count,
            inlier_count,
            inlier_ratio,
            mean_sampson_error: essential.mean_sampson_error,
            score,
            failure_reason: Some(LoopClosureVerificationFailureReason::PoseDisagreement),
            relative_pose: pnp.relative_pose.clone(),
            mean_reprojection_error_px: pnp.mean_reprojection_error_px,
        };
    };

    let direction_disagreement =
        translation_direction_disagreement_rad(&ess_pose.translation, &pnp_pose.translation);
    let rotation_disagreement = ess_pose.rotation.rotation_to(&pnp_pose.rotation).angle();
    let agreement_ok = direction_disagreement <= config.max_translation_direction_disagreement_rad
        && rotation_disagreement <= config.max_rotation_disagreement_rad;

    let failure_reason = if agreement_ok {
        None
    } else {
        Some(LoopClosureVerificationFailureReason::PoseDisagreement)
    };
    LoopClosureVerification {
        verified: agreement_ok,
        correspondence_count,
        inlier_count,
        inlier_ratio,
        mean_sampson_error: essential.mean_sampson_error,
        score,
        failure_reason,
        relative_pose: Some(pnp_pose.clone()),
        mean_reprojection_error_px: pnp.mean_reprojection_error_px,
    }
}

fn translation_direction_disagreement_rad(
    a: &nalgebra::Vector3<f64>,
    b: &nalgebra::Vector3<f64>,
) -> f64 {
    let na = a.norm();
    let nb = b.norm();
    if na < 1.0e-9 || nb < 1.0e-9 {
        return 0.0;
    }
    let dir_a = a / na;
    let dir_b = b / nb;
    dir_a.dot(&dir_b).clamp(-1.0, 1.0).acos()
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
    /// Running auto-bootstrap state. `Some` exactly when
    /// `config.vi_init.is_some() && config.imu.is_some()` (initialised
    /// by [`Self::new`]); deliberately private because writes to
    /// `completed` cross-cut with `imu_state` / `local_vi_ba_state` /
    /// `map.keyframes`. Inspected via [`Self::vi_initialization_status`].
    vi_init_state: Option<OnlineSlamViInitState>,
    /// Running motion-based VI init state. `Some` exactly when
    /// `config.vi_motion_init.is_some() && config.vi_init.is_some() &&
    /// config.imu.is_some()`. Private because the motion-based fire is
    /// gated on the static stage having completed first; the pipeline
    /// owns the ordering. Inspected via
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
        // The motion-based stage gates on the static stage having
        // completed; runs AFTER local-VI-BA so the refined pose +
        // velocity slot are already in `map.keyframes` /
        // `local_vi_ba_state.keyframe_state` when the trigger fires.
        let vi_motion_init =
            self.run_motion_vi_init_step(frame, applied_update.as_ref(), imu_factor.as_ref());

        // Online loop-closure + pose-graph refinement runs LAST so the
        // graph mirrors the just-finalised keyframe pose (post local-VI-
        // BA) before PGO write-back. No-op when the stage is disabled or
        // no keyframe was registered this frame.
        let pose_graph_refinement = self.maybe_run_loop_closure_refinement(
            frame,
            &tracking,
            applied_update.as_ref(),
            &mut loop_closure_candidates,
        );

        OnlineSlamResult {
            tracking,
            mapping,
            applied_update,
            loop_closure_candidates,
            imu_factor,
            local_vi_ba,
            map_keyframe_count: self.map.keyframes.len(),
            map_landmark_count: self.map.landmarks.len(),
            vi_init,
            vi_motion_init,
            pose_graph_refinement,
            relocalization: relocalization_stats,
        }
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
            return None;
        }
        state.trigger_count += 1;
        state.last_attempt_frame_id = Some(frame.id);
        // Build a one-shot descriptor store from the current map so the
        // recovery localizer matches against every landmark, not just
        // the local-map subset the tracker's covisibility filter may
        // have used. When the Phase-26 #4a active-frontier window is
        // configured, restrict to landmarks observed by the most recent
        // N keyframes — this drops stale landmarks that Phase-26 #2
        // showed admit wrong-scale geometrically-self-consistent
        // recoveries.
        let descriptor_store = match state.config.recent_keyframe_window {
            Some(window) if window > 0 => {
                let mut keyframe_ids: Vec<u64> = self.map.keyframes.keys().copied().collect();
                keyframe_ids.sort();
                let start = keyframe_ids.len().saturating_sub(window);
                let mut active_landmark_ids: std::collections::HashSet<u64> =
                    std::collections::HashSet::new();
                for kf_id in &keyframe_ids[start..] {
                    if let Some(kf) = self.map.keyframes.get(kf_id) {
                        for observation in &kf.observations {
                            active_landmark_ids.insert(observation.landmark_id);
                        }
                    }
                }
                let mut store = visloc_core::types::LandmarkDescriptorStore::new();
                for (lid, landmark) in &self.map.landmarks {
                    if active_landmark_ids.contains(lid) {
                        if let Some(descriptor) = landmark.descriptor.as_ref() {
                            store.insert(*lid, descriptor.clone());
                        }
                    }
                }
                store
            }
            _ => visloc_core::types::LandmarkDescriptorStore::from_visual_map(&self.map),
        };
        // When `pose_prior_candidate_radius_meters` is set, thread the
        // tracker's per-frame motion-model prediction into the recovery
        // PnP — both as the RANSAC warm-start hypothesis and as a
        // candidate-landmark filter (only landmarks within the radius
        // of the prior's camera centre are considered). When `None`,
        // fall back to the no-prior global PnP that Phase-23 #1 shipped.
        let pose_prior = if state.config.pose_prior_candidate_radius_meters.is_some() {
            self.tracker.pose_prior_for_frame(frame)
        } else {
            None
        };
        let candidate_radius = state.config.pose_prior_candidate_radius_meters;
        let recovered = if pose_prior.is_some() {
            state
                .localizer
                .localize_frame_with_pose_prior_warm_start_and_descriptor_store(
                    frame,
                    &self.map,
                    &descriptor_store,
                    pose_prior.as_ref(),
                    candidate_radius,
                )
        } else {
            state.localizer.localize_frame_with_descriptor_store(
                frame,
                &self.map,
                &descriptor_store,
            )
        };
        let mut stats = OnlineSlamRelocalizationStats {
            attempted: true,
            succeeded: false,
            inlier_count: recovered.inlier_count,
            inlier_ratio: recovered.inlier_ratio,
            correspondence_count: recovered.correspondence_count,
            mean_reprojection_error: recovered.reprojection_error,
        };
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
        // Phase-26 #4b post-acceptance IMU sanity check. When the
        // configured `max_translation_from_imu_prediction_meters` is
        // set AND a per-frame IMU prediction is available AND the
        // recovered pose has a camera centre, reject recoveries whose
        // recovered camera centre lies further from the IMU's
        // predicted centre than the threshold. Targets Phase-26 #2's
        // V1_01 false-positive failure mode where the recovery PnP
        // accepted geometrically self-consistent but wrong-scale
        // solutions far from the IMU's belief about where the camera
        // was.
        let imu_accept =
            if let Some(max_dist) = state.config.max_translation_from_imu_prediction_meters {
                match (
                    self.tracker.pose_prior_for_frame(frame),
                    recovered.pose.as_ref(),
                ) {
                    (Some(predicted), Some(recovered_pose)) => {
                        let predicted_centre = predicted.world_to_camera.inverse().translation;
                        let recovered_centre = recovered_pose.world_to_camera.inverse().translation;
                        (recovered_centre - predicted_centre).norm() <= max_dist
                    }
                    // No prediction available (e.g. tracker just bootstrapped)
                    // ⇒ do not gate; preserves recovery on cold-start.
                    _ => true,
                }
            } else {
                true
            };
        let accept = basic_accept && imu_accept;
        if !accept {
            return Some(stats);
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

        // Add the node + the sequential edge from the previous
        // keyframe in registration order. Anchor on the first registered
        // keyframe so the absolute frame stays fixed across PGO solves.
        let prev_keyframe_id = state.keyframe_order.last().copied();
        state.graph.add_pose(new_keyframe_id, new_pose.clone());
        if state.keyframe_order.is_empty() {
            state.graph.anchor(new_keyframe_id);
        } else if let Some(prev_id) = prev_keyframe_id {
            if let Some(prev_pose) = state.graph.poses.get(&prev_id).cloned() {
                state.graph.add_sequential_edge(
                    prev_id,
                    new_keyframe_id,
                    relative_world_to_camera(&prev_pose, &new_pose),
                );
            }
        }
        state.keyframe_order.push(new_keyframe_id);

        // Verify candidates (only essential-matrix verifier for now —
        // the pipeline currently exposes only the monocular two-view
        // backend; PnP/Hybrid wiring is a follow-up).
        let mut stats = OnlineSlamLoopClosureRefinementStats::default();
        if !loop_closure_candidates.is_empty() {
            let verifier = EssentialMatrixLoopClosureVerifier {
                config: state.config.verifier_config,
                ..Default::default()
            };
            stats.verified_candidate_count = loop_closure_candidates.len();
            verify_loop_closure_candidates(
                loop_closure_candidates,
                frame,
                tracking,
                &self.map,
                &state.config.camera,
                &verifier,
            );
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

            for candidate in loop_closure_candidates.iter() {
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
                if let Some(cfg) = &pcm_cfg {
                    let m = loop_measurement_of(&constraint);
                    if !pcm_admits_loop(&m, &admitted, &odometry, cfg) {
                        stats.loop_closures_pcm_rejected += 1;
                        continue;
                    }
                    admitted.push(m);
                }
                state.graph.add_loop_closure_constraint(&constraint);
                state.verified_constraints.push(constraint);
                state.pending_since_last_trigger += 1;
                stats.accepted_count += 1;
            }
        }

        // Trigger PGO when the configured number of new constraints has
        // accumulated. A higher threshold batches solves; `1` (the
        // recommended default) runs PGO on every accepted loop edge.
        let trigger_threshold = state.config.trigger_every_new_constraints.max(1);
        if state.pending_since_last_trigger >= trigger_threshold {
            let pgo_config = state.config.pose_graph_config.clone();
            state.pending_since_last_trigger = 0;
            state.trigger_count += 1;
            // When GNC is configured, run the robust solver so a
            // verified-but-wrong loop closure is annealed out at the
            // back-end; otherwise the plain M-estimator. Both paths write
            // the optimised poses back into the map, so a wrong closure
            // GNC rejected never reaches subsequent tracking / local-VI-BA.
            let solved = if let Some(gnc_config) = state.config.gnc {
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
            if solved {
                // Write optimised poses back into the map so subsequent
                // tracking / local-VI-BA passes see the refined frame.
                let mut updated = 0usize;
                for (id, pose) in state.graph.poses.iter() {
                    if let Some(keyframe) = self.map.keyframes.get_mut(id) {
                        keyframe.frame.pose = Some(pose.clone());
                        updated += 1;
                    }
                }
                stats.keyframes_updated = updated;
            }
        }

        Some(stats)
    }

    /// Register the freshly-staged IMU factor with the local VI-BA state
    /// table and, when the trigger threshold has been reached, run a
    /// sliding-window VI-BA pass that refines the trailing window's
    /// poses + landmarks + velocities + biases. No-op when local VI-BA
    /// is disabled OR when no IMU factor was staged this frame.
    fn maybe_run_local_vi_ba(
        &mut self,
        new_factor: Option<ImuPreintegrationFactor>,
    ) -> Option<OnlineSlamLocalBaStats> {
        // When `keep_pre_promotion_imu_factors` lets factors flow before
        // VI-init promotes, the factors must still be banked for the
        // post-promotion BA replay, but the BA itself cannot run yet —
        // its bias linearisation is the placeholder zero seed and a
        // pre-promotion solve corrupts the map's keyframe poses
        // (empirically: tracking-success collapses from 9.8 % to 1.8 %
        // on MH_01 because the next-frame matcher sees BA-shifted
        // keyframe descriptors). Gate the trigger here so the
        // `factor_history` accumulates but the solver waits.
        let vi_init_still_active = self
            .vi_init_state
            .as_ref()
            .map(|s| s.is_active())
            .unwrap_or(false);
        let state = self.local_vi_ba_state.as_mut()?;
        let factor = new_factor?;
        let should_trigger = state.register_new_factor(factor);
        if !should_trigger || vi_init_still_active {
            return None;
        }
        crate::online_slam_vi_ba::run_local_vi_ba(&mut self.map, state)
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
            imu.preintegrator = ImuPreintegrator::new_with_bias(bias_gyro, bias_acc);
            imu.pending_factor = None;
            imu.config.bias_gyro = bias_gyro;
            imu.config.bias_acc = bias_acc;
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
    /// * the static VI init stage has not yet succeeded (the motion-
    ///   based stage's prerequisite),
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
        // Gate on the static stage having completed successfully. The
        // motion-based stage's `static_seed` is the result we mirror.
        let static_seed = self
            .vi_init_state
            .as_ref()
            .and_then(|s| s.completed.clone())?;
        if self
            .vi_motion_init_state
            .as_ref()
            .map(|s| !s.is_active())
            .unwrap_or(true)
        {
            return None;
        }

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
            match state
                .initializer
                .try_initialize(&mut self.map, &factors_snapshot, &static_seed)
            {
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
                    imu.preintegrator = ImuPreintegrator::new_with_bias(bg, ba);
                    imu.pending_factor = None;
                    imu.config.bias_gyro = bg;
                    imu.config.bias_acc = ba;
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

        // Step 3: mark the stage completed. (`initializer.try_initialize`
        // already cached the result in its `completed` slot; mirror onto
        // the state's own `completed` so `is_active()` flips false.)
        if let Some(state) = self.vi_motion_init_state.as_mut() {
            state.completed = Some(result.clone());
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

pub fn online_slam_results_to_html_report(results: &[OnlineSlamResult]) -> String {
    let samples = slam_report_samples(results);
    let loop_candidates = results
        .iter()
        .flat_map(|result| result.loop_closure_candidates.iter())
        .collect::<Vec<_>>();
    let mut output = String::new();
    output.push_str("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
    output.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
    output.push_str("<title>visloc-rs online SLAM loop report</title>\n");
    output.push_str("<style>");
    output.push_str(
        "body{margin:0;font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#f6f7f9;color:#182026}\
         main{max-width:1120px;margin:0 auto;padding:28px}\
         h1{font-size:24px;margin:0 0 8px}\
         h2{font-size:18px;margin:0 0 10px}\
         .sub{margin:0 0 22px;color:#52616b}\
         .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:18px 0}\
         .metric{background:white;border:1px solid #dde3ea;border-radius:8px;padding:12px}\
         .label{display:block;font-size:12px;color:#65727e}\
         .value{display:block;font-size:22px;font-weight:700;margin-top:4px}\
         .panel{background:white;border:1px solid #dde3ea;border-radius:8px;padding:16px;margin-top:14px}\
         table{width:100%;border-collapse:collapse;font-size:13px}\
         th,td{text-align:right;border-bottom:1px solid #e7ecf0;padding:6px 8px;vertical-align:top}\
         th:first-child,td:first-child{text-align:left}\
         .ok{color:#198754;font-weight:700}.warn{color:#a15c00;font-weight:700}\
         svg{width:100%;height:auto;display:block}",
    );
    output.push_str("</style>\n</head>\n<body>\n<main>\n");
    output.push_str("<h1>visloc-rs online SLAM loop report</h1>\n");
    output.push_str("<p class=\"sub\">Top-down tracked camera centers with diagnostic loop-closure candidate edges. This report does not imply global pose-graph optimization.</p>\n");
    output.push_str("<section class=\"grid\">\n");
    push_metric_card(&mut output, "Frames", &results.len().to_string());
    push_metric_card(&mut output, "Tracked poses", &samples.len().to_string());
    push_metric_card(
        &mut output,
        "Loop candidates",
        &loop_candidates.len().to_string(),
    );
    push_metric_card(
        &mut output,
        "Final keyframes",
        &results
            .last()
            .map(|result| result.map_keyframe_count.to_string())
            .unwrap_or_else(|| "0".to_string()),
    );
    push_metric_card(
        &mut output,
        "Final landmarks",
        &results
            .last()
            .map(|result| result.map_landmark_count.to_string())
            .unwrap_or_else(|| "0".to_string()),
    );
    output.push_str("</section>\n");
    output.push_str("<section class=\"panel\">\n");
    output.push_str(&online_slam_loop_svg(&samples, &loop_candidates));
    output.push_str("</section>\n");
    output.push_str("<section class=\"panel\">\n<h2>Loop Closure Candidates</h2>\n");
    output.push_str("<table><thead><tr><th>query frame</th><th>matched keyframe</th><th>shared landmarks</th><th>ratio</th><th>score</th><th>verified</th><th>verifier inliers</th><th>verifier inlier ratio</th><th>mean error</th><th>verifier score</th><th>failure</th></tr></thead><tbody>\n");
    if loop_candidates.is_empty() {
        output.push_str("<tr><td colspan=\"11\">no loop candidates reported</td></tr>\n");
    }
    for candidate in &loop_candidates {
        let verified_class = if candidate.geometrically_verified {
            "ok"
        } else {
            "warn"
        };
        let verified_text = match candidate.verification.as_ref() {
            Some(verification) if verification.verified => "yes",
            Some(_) => "rejected",
            None => "candidate",
        };
        let (inlier_count_text, inlier_ratio_text, mean_text, verifier_score_text, failure_text) =
            if let Some(verification) = candidate.verification.as_ref() {
                (
                    verification.inlier_count.to_string(),
                    format!("{:.3}", verification.inlier_ratio),
                    if let Some(px) = verification.mean_reprojection_error_px {
                        format!("{px:.4} px")
                    } else if verification.mean_sampson_error.is_finite() {
                        format!("{:.4}", verification.mean_sampson_error)
                    } else {
                        "n/a".to_string()
                    },
                    format!("{:.3}", verification.score),
                    verification
                        .failure_reason
                        .as_ref()
                        .map(format_loop_closure_failure_reason)
                        .unwrap_or_else(|| "&mdash;".to_string()),
                )
            } else {
                (
                    "&mdash;".to_string(),
                    "&mdash;".to_string(),
                    "&mdash;".to_string(),
                    "&mdash;".to_string(),
                    "&mdash;".to_string(),
                )
            };
        let _ = writeln!(
            output,
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.3}</td><td>{:.3}</td><td class=\"{}\">{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            candidate.query_frame_id,
            candidate.matched_keyframe_id,
            candidate.shared_landmark_count,
            candidate.shared_landmark_ratio,
            candidate.score,
            verified_class,
            verified_text,
            inlier_count_text,
            inlier_ratio_text,
            mean_text,
            verifier_score_text,
            failure_text,
        );
    }
    output.push_str("</tbody></table>\n</section>\n");

    let constraints: Vec<LoopClosureConstraint> = loop_candidates
        .iter()
        .filter_map(|candidate| LoopClosureConstraint::from_verified_candidate(candidate))
        .collect();
    output.push_str("<section class=\"panel\">\n<h2>Loop Closure Constraints</h2>\n");
    output.push_str("<p class=\"sub\">Each row is a verified candidate turned into a `LoopClosureConstraint` ready for a future pose-graph layer. No global optimization runs in this report.</p>\n");
    output.push_str("<table><thead><tr><th>from keyframe</th><th>to keyframe</th><th>inliers</th><th>inlier ratio</th><th>mean Sampson</th><th>score</th><th>relative translation</th></tr></thead><tbody>\n");
    if constraints.is_empty() {
        output.push_str("<tr><td colspan=\"7\">no verified loop constraints</td></tr>\n");
    }
    for constraint in &constraints {
        let translation = constraint.relative_pose.translation;
        let _ = writeln!(
            output,
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.3}</td><td>{:.4}</td><td>{:.3}</td><td>[{:.3}, {:.3}, {:.3}]</td></tr>",
            constraint.from_keyframe_id,
            constraint.to_keyframe_id,
            constraint.inlier_count,
            constraint.inlier_ratio,
            constraint.mean_sampson_error,
            constraint.score,
            translation.x,
            translation.y,
            translation.z,
        );
    }
    output.push_str("</tbody></table>\n</section>\n</main>\n</body>\n</html>\n");
    output
}

fn format_loop_closure_failure_reason(reason: &LoopClosureVerificationFailureReason) -> String {
    match reason {
        LoopClosureVerificationFailureReason::InsufficientCorrespondences => {
            "insufficient correspondences".to_string()
        }
        LoopClosureVerificationFailureReason::EssentialEstimationFailed => {
            "essential RANSAC failed".to_string()
        }
        LoopClosureVerificationFailureReason::TooFewInliers => "too few inliers".to_string(),
        LoopClosureVerificationFailureReason::LowInlierRatio => "low inlier ratio".to_string(),
        LoopClosureVerificationFailureReason::HighSampsonError => "high Sampson error".to_string(),
        LoopClosureVerificationFailureReason::PoseDisagreement => {
            "pose disagreement (hybrid)".to_string()
        }
    }
}

pub fn write_online_slam_results_html_report(
    results: &[OnlineSlamResult],
    path: impl AsRef<Path>,
) -> std::io::Result<()> {
    std::fs::write(path, online_slam_results_to_html_report(results))
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

#[derive(Debug, Clone, Copy)]
struct SlamReportSample {
    frame_id: u64,
    x: f64,
    y: f64,
    z: f64,
}

fn slam_report_samples(results: &[OnlineSlamResult]) -> Vec<SlamReportSample> {
    results
        .iter()
        .filter_map(|result| {
            let pose = result.tracking.localization.pose.as_ref()?;
            let center = pose.camera_center_world();
            Some(SlamReportSample {
                frame_id: result.tracking.frame_id,
                x: center.x,
                y: center.y,
                z: center.z,
            })
        })
        .collect()
}

fn online_slam_loop_svg(
    samples: &[SlamReportSample],
    candidates: &[&LoopClosureCandidate],
) -> String {
    let projection = SlamReportProjection::from_samples(samples);
    let by_frame_id = samples
        .iter()
        .map(|sample| (sample.frame_id, *sample))
        .collect::<HashMap<_, _>>();
    let mut output = String::new();
    output.push_str("<svg viewBox=\"0 0 900 520\" role=\"img\" aria-label=\"online SLAM loop candidate plot\">\n");
    output.push_str("<rect x=\"0\" y=\"0\" width=\"900\" height=\"520\" fill=\"#fbfcfd\"/>\n");
    output.push_str("<g stroke=\"#e4e9ef\" stroke-width=\"1\">\n");
    for x in [80, 228, 376, 524, 672, 820] {
        let _ = writeln!(output, "<line x1=\"{x}\" y1=\"54\" x2=\"{x}\" y2=\"450\"/>");
    }
    for y in [54, 133, 212, 291, 370, 450] {
        let _ = writeln!(output, "<line x1=\"80\" y1=\"{y}\" x2=\"820\" y2=\"{y}\"/>");
    }
    output.push_str("</g>\n");
    output.push_str("<g fill=\"none\" stroke-linecap=\"round\" stroke-linejoin=\"round\">\n");
    if !samples.is_empty() {
        let points = samples
            .iter()
            .map(|sample| {
                let (x, y) = projection.project(sample);
                format!("{x:.2},{y:.2}")
            })
            .collect::<Vec<_>>()
            .join(" ");
        let _ = writeln!(
            output,
            "<polyline points=\"{points}\" stroke=\"#2676c9\" stroke-width=\"4\"/>"
        );
    }
    output.push_str("</g>\n");
    output.push_str("<g fill=\"none\" stroke=\"#f0a202\" stroke-width=\"4\" stroke-dasharray=\"10 7\" stroke-linecap=\"round\">\n");
    for candidate in candidates {
        let (Some(query), Some(matched)) = (
            by_frame_id.get(&candidate.query_frame_id),
            by_frame_id.get(&candidate.matched_keyframe_id),
        ) else {
            continue;
        };
        let (qx, qy) = projection.project(query);
        let (mx, my) = projection.project(matched);
        let _ = writeln!(
            output,
            "<line x1=\"{qx:.2}\" y1=\"{qy:.2}\" x2=\"{mx:.2}\" y2=\"{my:.2}\"/>"
        );
    }
    output.push_str("</g>\n<g>\n");
    for sample in samples {
        let (x, y) = projection.project(sample);
        let _ = writeln!(
            output,
            "<circle cx=\"{x:.2}\" cy=\"{y:.2}\" r=\"6\" fill=\"#2676c9\"/>"
        );
        let _ = writeln!(
            output,
            "<text x=\"{:.2}\" y=\"{:.2}\" fill=\"#52616b\" font-size=\"12\" text-anchor=\"middle\">{}</text>",
            x,
            y + 22.0,
            sample.frame_id
        );
    }
    output.push_str("</g>\n");
    output.push_str("<rect x=\"80\" y=\"468\" width=\"14\" height=\"6\" fill=\"#2676c9\"/>\n");
    output.push_str(
        "<text x=\"102\" y=\"476\" fill=\"#52616b\" font-size=\"13\">tracked camera path</text>\n",
    );
    output.push_str("<line x1=\"278\" y1=\"472\" x2=\"320\" y2=\"472\" stroke=\"#f0a202\" stroke-width=\"4\" stroke-dasharray=\"10 7\"/>\n");
    output.push_str(
        "<text x=\"330\" y=\"476\" fill=\"#52616b\" font-size=\"13\">loop candidate edge</text>\n",
    );
    output.push_str("</svg>\n");
    output
}

#[derive(Debug, Clone, Copy)]
struct SlamReportProjection {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    axis_y: usize,
}

impl SlamReportProjection {
    fn from_samples(samples: &[SlamReportSample]) -> Self {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for sample in samples {
            min[0] = min[0].min(sample.x);
            min[1] = min[1].min(sample.y);
            min[2] = min[2].min(sample.z);
            max[0] = max[0].max(sample.x);
            max[1] = max[1].max(sample.y);
            max[2] = max[2].max(sample.z);
        }

        if !min[0].is_finite() {
            return Self {
                min_x: -1.0,
                max_x: 1.0,
                min_y: -1.0,
                max_y: 1.0,
                axis_y: 2,
            };
        }

        let spread_y = max[1] - min[1];
        let spread_z = max[2] - min[2];
        let axis_y = if spread_z >= spread_y { 2 } else { 1 };
        let (mut min_x, mut max_x) = padded_range(min[0], max[0]);
        let (mut min_y, mut max_y) = padded_range(min[axis_y], max[axis_y]);
        let x_span = max_x - min_x;
        let y_span = max_y - min_y;
        if x_span > y_span {
            let delta = (x_span - y_span) * 0.5;
            min_y -= delta;
            max_y += delta;
        } else {
            let delta = (y_span - x_span) * 0.5;
            min_x -= delta;
            max_x += delta;
        }

        Self {
            min_x,
            max_x,
            min_y,
            max_y,
            axis_y,
        }
    }

    fn project(&self, sample: &SlamReportSample) -> (f64, f64) {
        let plot_left = 80.0;
        let plot_top = 54.0;
        let plot_width = 740.0;
        let plot_height = 396.0;
        let horizontal = (sample.x - self.min_x) / (self.max_x - self.min_x);
        let vertical_value = if self.axis_y == 2 { sample.z } else { sample.y };
        let vertical = (vertical_value - self.min_y) / (self.max_y - self.min_y);
        (
            plot_left + horizontal * plot_width,
            plot_top + (1.0 - vertical) * plot_height,
        )
    }
}

fn padded_range(min: f64, max: f64) -> (f64, f64) {
    let span = max - min;
    if span.abs() < 1.0e-12 {
        (min - 1.0, max + 1.0)
    } else {
        let padding = span * 0.08;
        (min - padding, max + padding)
    }
}

fn push_metric_card(output: &mut String, label: &str, value: &str) {
    let _ = writeln!(
        output,
        "<div class=\"metric\"><span class=\"label\">{label}</span><span class=\"value\">{value}</span></div>"
    );
}

fn detect_loop_closure_candidates(
    frame: &Frame,
    tracking: &TrackingResult,
    map: &VisualMap,
    config: &LoopClosureConfig,
) -> Vec<LoopClosureCandidate> {
    if !config.enabled || !tracking.localization.success {
        return Vec::new();
    }

    let query_landmarks = tracking
        .localization
        .inlier_landmark_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    if query_landmarks.is_empty() {
        return Vec::new();
    }

    let mut candidates = map
        .keyframes
        .values()
        .filter_map(|keyframe| {
            if frame.id.abs_diff(keyframe.frame.id) < config.min_frame_id_gap {
                return None;
            }

            let keyframe_landmarks = keyframe
                .observations
                .iter()
                .map(|observation| observation.landmark_id)
                .collect::<HashSet<_>>();
            if keyframe_landmarks.is_empty() {
                return None;
            }

            let shared_landmark_count = query_landmarks.intersection(&keyframe_landmarks).count();
            if shared_landmark_count < config.min_shared_landmarks {
                return None;
            }

            let denominator = query_landmarks.len().min(keyframe_landmarks.len());
            let shared_landmark_ratio = shared_landmark_count as f64 / denominator as f64;
            let required_ratio = f64::from(config.min_shared_landmark_ratio_percent) / 100.0;
            if shared_landmark_ratio < required_ratio {
                return None;
            }

            let score = shared_landmark_ratio * shared_landmark_count as f64;
            Some(LoopClosureCandidate {
                query_frame_id: frame.id,
                matched_keyframe_id: keyframe.frame.id,
                shared_landmark_count,
                query_inlier_count: query_landmarks.len(),
                keyframe_observation_count: keyframe_landmarks.len(),
                shared_landmark_ratio,
                score,
                geometrically_verified: true,
                verification: None,
            })
        })
        .collect::<Vec<_>>();

    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.shared_landmark_count.cmp(&a.shared_landmark_count))
            .then_with(|| a.matched_keyframe_id.cmp(&b.matched_keyframe_id))
    });
    candidates.truncate(config.max_candidates);
    candidates
}

/// Kind of an edge inside a [`PoseGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoseGraphEdgeKind {
    /// Sequential odometry edge between consecutive keyframes.
    Sequential,
    /// Loop-closure edge backed by a verified [`LoopClosureConstraint`].
    LoopClosure,
}

/// Edge in a sparse [`PoseGraph`]. Encodes a measured `previous_to_current`
/// SE3 between two keyframes plus a positive weight used by translation-only
/// least squares.
///
/// `information` optionally carries a full 6×6 information matrix `Ω`, i.e. the
/// inverse measurement covariance, ordered `[ρ; ω]` (translation block first,
/// then rotation) to match [`SE3::log`] and the `.g2o` `EDGE_SE3:QUAT`
/// convention. When `Some`, the SE(3) solver minimizes the anisotropic
/// Mahalanobis cost `rᵀ Ω r` for this edge and the scalar `weight` is ignored;
/// when `None`, the edge falls back to the isotropic `weight · ‖r‖²` behavior.
/// This lets the graph ingest external constraints (e.g. `.g2o`
/// `EDGE_SE3:QUAT`) whose blocks couple rotation and translation, without
/// changing the meaning of the internally-built sequential / loop-closure
/// edges.
#[derive(Debug, Clone, PartialEq)]
pub struct PoseGraphEdge {
    pub from: u64,
    pub to: u64,
    pub measurement: SE3,
    pub kind: PoseGraphEdgeKind,
    pub weight: f64,
    pub information: Option<Matrix6<f64>>,
}

/// Single Gauss-Newton step diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct PoseGraphOptimizationStep {
    pub anchor_id: u64,
    pub edge_count: usize,
    pub variable_count: usize,
    pub cost_before: f64,
    pub cost_after: f64,
    pub mean_translation_correction: f64,
    pub max_translation_correction: f64,
}

/// Robust kernel applied to each pose-graph edge's residual norm-squared.
/// Down-weights edges whose squared residual exceeds the kernel threshold so
/// outlier loop closures cannot dominate the least-squares solve.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RobustKernel {
    /// Standard squared-error cost (`ρ(s) = s`).
    #[default]
    None,
    /// Huber kernel: quadratic for `s ≤ δ²`, linear in `√s` beyond.
    /// `delta` is the threshold on residual norm where the kernel switches
    /// from quadratic to linear.
    Huber { delta: f64 },
    /// Cauchy / Lorentzian kernel: `ρ(s) = c² · log(1 + s / c²)`.
    /// `c` is the soft-saturation scale on residual norm.
    Cauchy { c: f64 },
}

impl RobustKernel {
    /// Applied cost `ρ(s)` for `s = ||r||²`.
    pub fn cost(&self, s: f64) -> f64 {
        match *self {
            RobustKernel::None => s,
            RobustKernel::Huber { delta } => {
                let delta_sq = delta * delta;
                if s <= delta_sq {
                    s
                } else {
                    2.0 * delta * s.sqrt() - delta_sq
                }
            }
            RobustKernel::Cauchy { c } => {
                let c_sq = c * c;
                c_sq * (1.0 + s / c_sq).ln()
            }
        }
    }

    /// Influence weight `ρ'(s)` used as a multiplier on each edge's normal-equation
    /// contribution (a.k.a. IRLS weight).
    pub fn weight(&self, s: f64) -> f64 {
        match *self {
            RobustKernel::None => 1.0,
            RobustKernel::Huber { delta } => {
                let delta_sq = delta * delta;
                if s <= delta_sq {
                    1.0
                } else {
                    delta / s.sqrt()
                }
            }
            RobustKernel::Cauchy { c } => {
                let c_sq = c * c;
                1.0 / (1.0 + s / c_sq)
            }
        }
    }
}

/// Linear solver backend used by pose-graph optimizers when the normal
/// equations `(H + λI) δ = -g` (or the translation-only analogue) are solved.
///
/// `Dense` materializes the full SPD matrix as a [`DMatrix`] and uses
/// nalgebra's dense Cholesky (LU fallback). `Sparse` assembles the same
/// system from edge triplets and solves it with the block Cholesky (the
/// `block_cholesky` module) in a fill-reducing order. The two paths produce
/// numerically equivalent solutions on connected, well-conditioned graphs but
/// the sparse path scales to thousands of keyframes where the dense path
/// becomes infeasible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LinearSolver {
    /// Dense Cholesky / LU on a [`DMatrix`].
    #[default]
    Dense,
    /// Sparse block Cholesky on a triplet-assembled system.
    Sparse,
}

/// Configuration for [`PoseGraph::optimize_se3_iterative`].
#[derive(Debug, Clone, PartialEq)]
pub struct PoseGraphSe3Config {
    /// Hard cap on iterations (including rejected LM steps).
    pub max_iterations: usize,
    /// Convergence threshold on the largest per-node 6-vector update of the
    /// most recent accepted step.
    pub step_tolerance: f64,
    /// Convergence threshold on the absolute cost change between two
    /// successive accepted steps.
    pub cost_tolerance: f64,
    /// Robust kernel applied to each edge's squared residual norm.
    pub robust_kernel: RobustKernel,
    /// Initial Levenberg-Marquardt damping `λ`. `None` runs pure
    /// Gauss-Newton (every step is accepted unconditionally). `Some(λ₀)`
    /// enables LM: solve `(H + λI) δ = -g`, accept if cost decreases (and
    /// scale `λ` down by `lambda_decrease_factor`), otherwise reject and
    /// scale `λ` up by `lambda_increase_factor`.
    pub initial_lambda: Option<f64>,
    /// Multiplier applied to `λ` after a rejected LM step.
    pub lambda_increase_factor: f64,
    /// Multiplier applied to `λ` after an accepted LM step.
    pub lambda_decrease_factor: f64,
    /// Upper bound on `λ`. When a step is rejected and `λ * factor > max_lambda`,
    /// the optimizer gives up and returns `converged: false`.
    pub max_lambda: f64,
    /// Lower bound on `λ`. Decreases stop here so `λ` cannot collapse to zero.
    pub min_lambda: f64,
    /// Linear-solver backend. Defaults to dense Cholesky for parity with the
    /// pre-sparse solver. Switch to [`LinearSolver::Sparse`] when the graph
    /// has more than a few hundred nodes so the optimizer scales linearly in
    /// non-zero edges instead of cubically in node count.
    pub linear_solver: LinearSolver,
    /// Seed the solve with a chordal rotation initialization
    /// ([`PoseGraph::initialize_rotations_chordal`]) before the first
    /// Gauss-Newton step. Defaults to `true`: it is strictly beneficial —
    /// the rotation optimum is a fixed point of the relaxation, so on an
    /// already-consistent graph it leaves the estimate essentially unchanged
    /// (a cheap extra factorization), while on a hard, odometry-initialized 3D
    /// graph it rescues the solve from a poor basin. The seeding is best-effort:
    /// if its relaxed system is singular it is silently skipped and the solve
    /// proceeds from the unmodified estimate, so enabling it can never turn a
    /// previously-successful optimization into a failure.
    pub chordal_init: bool,
}

impl Default for PoseGraphSe3Config {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            step_tolerance: 1e-6,
            cost_tolerance: 1e-9,
            robust_kernel: RobustKernel::None,
            initial_lambda: None,
            lambda_increase_factor: 10.0,
            lambda_decrease_factor: 0.1,
            max_lambda: 1e12,
            min_lambda: 1e-9,
            linear_solver: LinearSolver::Dense,
            chordal_init: true,
        }
    }
}

/// Per-iteration diagnostics for [`PoseGraph::optimize_se3_iterative`].
#[derive(Debug, Clone, PartialEq)]
pub struct PoseGraphSe3IterationStats {
    pub iteration: usize,
    pub cost_before: f64,
    pub cost_after: f64,
    pub max_step_norm: f64,
    /// LM damping `λ` used for this iteration (`0.0` for pure Gauss-Newton).
    pub lambda: f64,
    /// `true` when the trial step was kept; only false for rejected LM steps.
    pub step_accepted: bool,
}

/// Diagnostics from [`PoseGraph::initialize_rotations_chordal`].
#[derive(Debug, Clone, PartialEq)]
pub struct ChordalRotationInit {
    pub anchor_id: u64,
    pub edge_count: usize,
    pub variable_count: usize,
    /// Chordal rotation cost (see [`PoseGraph::chordal_rotation_cost`]) before
    /// the relaxation was solved.
    pub cost_before: f64,
    /// Chordal rotation cost after replacing every node rotation with the
    /// SVD-projected chordal solution.
    pub cost_after: f64,
    /// Largest per-node geodesic rotation change (degrees) applied by the init.
    pub max_rotation_update_deg: f64,
}

/// Result of a full SE(3) Gauss-Newton run.
#[derive(Debug, Clone, PartialEq)]
pub struct PoseGraphSe3Result {
    pub anchor_id: u64,
    pub edge_count: usize,
    pub variable_count: usize,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations: Vec<PoseGraphSe3IterationStats>,
    pub converged: bool,
}

/// Outcome of [`PoseGraph::optimize_se3_gnc`], the outlier-robust SE(3)
/// pose-graph solve driven by Graduated Non-Convexity (see [`crate::gnc`]).
///
/// Beyond the usual cost/convergence summary it reports the **final per-edge
/// GNC weight** (`edge_weights`, indexed by edge position, each in `[0, 1]`):
/// an edge annealed to a weight near zero was rejected as an outlier, so this
/// vector doubles as a loop-closure inlier/outlier classification.
#[derive(Debug, Clone, PartialEq)]
pub struct PoseGraphGncResult {
    pub anchor_id: u64,
    pub edge_count: usize,
    pub variable_count: usize,
    /// Plain (non-robust) least-squares cost at the seeded starting point.
    pub initial_cost: f64,
    /// Plain (non-robust) least-squares cost over all edges at the final
    /// estimate. Outlier edges still contribute their (large) residual here;
    /// use [`Self::inlier_cost`] for the cost over retained edges only.
    pub final_cost: f64,
    /// Plain least-squares cost summed over edges whose final weight is at or
    /// above the inlier `threshold` passed to the solve — the cost GNC actually
    /// drove down once outliers were rejected.
    pub inlier_cost: f64,
    /// The inlier scale `c` the solve actually used: the configured
    /// [`gnc::GncConfig::c`] verbatim, or — under
    /// [`gnc::GncConfig::auto_scale`] — the MAD estimate (floored at the
    /// configured `c`).
    pub inlier_scale: f64,
    /// Number of outer `μ` levels executed.
    pub outer_iterations: usize,
    /// Whether the `μ` schedule reached the true robust cost (terminal `μ`).
    pub converged: bool,
    /// Final GNC weight per edge, indexed by edge position, in `[0, 1]`.
    pub edge_weights: Vec<f64>,
}

impl PoseGraphGncResult {
    /// Number of edges GNC kept as inliers: final weight `≥ threshold`.
    pub fn inlier_count(&self, threshold: f64) -> usize {
        self.edge_weights
            .iter()
            .filter(|&&w| w >= threshold)
            .count()
    }

    /// Number of edges GNC rejected as outliers: final weight `< threshold`.
    pub fn outlier_count(&self, threshold: f64) -> usize {
        self.edge_count - self.inlier_count(threshold)
    }
}

/// Errors returned by [`PoseGraph::optimize_translations_once`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoseGraphError {
    /// No anchor was specified before optimization.
    NoAnchor,
    /// An edge or anchor referenced a node that is missing from the graph.
    MissingNode(u64),
    /// The graph contains no edges, so there is nothing to optimize.
    NoEdges,
    /// The graph contains no non-anchor nodes (all variables are fixed).
    NoVariables,
    /// The Gauss-Newton normal equations were singular, e.g., because the
    /// graph has disconnected components or rank-deficient constraints.
    SingularSystem,
}

impl std::fmt::Display for PoseGraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoseGraphError::NoAnchor => write!(f, "pose graph has no anchor"),
            PoseGraphError::MissingNode(id) => write!(f, "pose graph is missing node {id}"),
            PoseGraphError::NoEdges => write!(f, "pose graph has no edges"),
            PoseGraphError::NoVariables => write!(f, "pose graph has no non-anchor nodes"),
            PoseGraphError::SingularSystem => {
                write!(f, "pose graph translation Gauss-Newton system was singular")
            }
        }
    }
}

impl std::error::Error for PoseGraphError {}

/// Sparse pose graph keyed by keyframe id. Stores per-node poses plus a flat
/// list of sequential and loop-closure edges, and provides a single
/// translation-only Gauss-Newton step that keeps node rotations fixed.
///
/// This is intentionally a skeleton: rotations are not optimized, the solver
/// is a single linear least-squares step rather than an iterative SE3 solver,
/// and there is no incremental incremental map update. Future milestones can
/// extend the same data type with full SE3 Jacobians, robust kernels, or a
/// production solver.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PoseGraph {
    /// Keyframe id → pose. `BTreeMap` keeps the iteration order deterministic
    /// so the variable layout in the linear system is reproducible.
    pub poses: BTreeMap<u64, Pose>,
    /// Edges in insertion order.
    pub edges: Vec<PoseGraphEdge>,
    /// Anchor keyframe id; its pose is held fixed during optimization.
    pub anchor: Option<u64>,
}

impl PoseGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a pose for `keyframe_id`.
    pub fn add_pose(&mut self, keyframe_id: u64, pose: Pose) {
        self.poses.insert(keyframe_id, pose);
    }

    /// Designate `keyframe_id` as the anchor whose pose stays fixed during
    /// translation optimization. Replaces any previously selected anchor.
    pub fn anchor(&mut self, keyframe_id: u64) {
        self.anchor = Some(keyframe_id);
    }

    /// Add a sequential odometry edge with weight `1.0`.
    pub fn add_sequential_edge(&mut self, from: u64, to: u64, measurement: SE3) {
        self.edges.push(PoseGraphEdge {
            from,
            to,
            measurement,
            kind: PoseGraphEdgeKind::Sequential,
            weight: 1.0,
            information: None,
        });
    }

    /// Append a loop-closure constraint as a graph edge. The verifier-derived
    /// inlier count is reused as the edge weight (clamped to a minimum of
    /// `1.0`) so loops with more inliers carry more pull on the solver.
    pub fn add_loop_closure_constraint(&mut self, constraint: &LoopClosureConstraint) {
        let weight = (constraint.inlier_count as f64).max(1.0);
        self.edges.push(PoseGraphEdge {
            from: constraint.from_keyframe_id,
            to: constraint.to_keyframe_id,
            measurement: constraint.relative_pose.clone(),
            kind: PoseGraphEdgeKind::LoopClosure,
            weight,
            information: None,
        });
    }

    /// Add an edge carrying a full 6×6 information matrix `Ω`, ordered `[ρ; ω]`
    /// (translation block first, then rotation — the [`SE3::log`] / `.g2o`
    /// `EDGE_SE3:QUAT` convention). The SE(3) solver minimizes the anisotropic
    /// `rᵀ Ω r` for this edge; the scalar `weight` is left at `1.0` and unused
    /// while `information` is `Some`.
    pub fn add_edge_with_information(
        &mut self,
        from: u64,
        to: u64,
        measurement: SE3,
        kind: PoseGraphEdgeKind,
        information: Matrix6<f64>,
    ) {
        self.edges.push(PoseGraphEdge {
            from,
            to,
            measurement,
            kind,
            weight: 1.0,
            information: Some(information),
        });
    }

    /// Sum of squared edge translation residuals in world coordinates.
    /// Rotation residuals are ignored — this is a translation-only metric
    /// that matches what [`Self::optimize_translations_once`] minimizes.
    pub fn translation_cost(&self) -> f64 {
        let mut total = 0.0;
        for edge in &self.edges {
            let (Some(from), Some(to)) = (self.poses.get(&edge.from), self.poses.get(&edge.to))
            else {
                continue;
            };
            let displacement = expected_world_displacement(to, &edge.measurement);
            let actual = to.camera_center_world() - from.camera_center_world();
            let residual = actual - displacement;
            total += edge.weight * residual.norm_squared();
        }
        total
    }

    /// Solve a single Gauss-Newton step on the translation residuals while
    /// holding rotations fixed. With linear-in-translation residuals the
    /// "single step" is the exact least-squares optimum of the underlying
    /// linear system, not a Newton iteration that needs to be repeated.
    ///
    /// Equivalent to [`Self::optimize_translations_once_with`] called with
    /// [`LinearSolver::Dense`].
    pub fn optimize_translations_once(
        &mut self,
    ) -> Result<PoseGraphOptimizationStep, PoseGraphError> {
        self.optimize_translations_once_with(LinearSolver::Dense)
    }

    /// Variant of [`Self::optimize_translations_once`] that selects the
    /// linear-solver backend. Use [`LinearSolver::Sparse`] for graphs with
    /// hundreds-to-thousands of keyframes — the normal-equations matrix is
    /// block-banded with at most four `3×3` blocks per edge, so the sparse
    /// block Cholesky is much faster than the dense path and uses dramatically
    /// less memory.
    pub fn optimize_translations_once_with(
        &mut self,
        linear_solver: LinearSolver,
    ) -> Result<PoseGraphOptimizationStep, PoseGraphError> {
        let anchor_id = self.anchor.ok_or(PoseGraphError::NoAnchor)?;
        let anchor_pose = self
            .poses
            .get(&anchor_id)
            .ok_or(PoseGraphError::MissingNode(anchor_id))?
            .clone();
        let anchor_center = anchor_pose.camera_center_world();
        if self.edges.is_empty() {
            return Err(PoseGraphError::NoEdges);
        }

        let mut node_index: BTreeMap<u64, usize> = BTreeMap::new();
        for &id in self.poses.keys() {
            if id == anchor_id {
                continue;
            }
            let next = node_index.len();
            node_index.insert(id, next);
        }
        let variable_count = node_index.len();
        if variable_count == 0 {
            return Err(PoseGraphError::NoVariables);
        }

        for edge in &self.edges {
            if !self.poses.contains_key(&edge.from) {
                return Err(PoseGraphError::MissingNode(edge.from));
            }
            if !self.poses.contains_key(&edge.to) {
                return Err(PoseGraphError::MissingNode(edge.to));
            }
        }

        let cost_before = self.translation_cost();

        // Assemble the normal equations `A^T A x = A^T b` directly. Each row
        // of `A` (per edge) has at most two nonzero 3×3 identity-shaped
        // blocks (`+w · I` at `to`, `-w · I` at `from`), so the contribution
        // to `A^T A` per edge is also block-structured: `w² · I` at the
        // `(to, to)` and `(from, from)` diagonal blocks and `-w² · I` at
        // both off-diagonal blocks. Anchor-incident edges only push to one
        // diagonal and shift `A^T b` by the anchor center.
        let dim = variable_count * 3;
        let mut h_dense = match linear_solver {
            LinearSolver::Dense => Some(DMatrix::<f64>::zeros(dim, dim)),
            LinearSolver::Sparse => None,
        };
        let mut triplets: Vec<(usize, usize, f64)> = match linear_solver {
            LinearSolver::Dense => Vec::new(),
            LinearSolver::Sparse => Vec::with_capacity(self.edges.len() * 36),
        };
        let mut atb = DVector::<f64>::zeros(dim);

        for edge in &self.edges {
            let to_pose = &self.poses[&edge.to];
            let displacement = expected_world_displacement(to_pose, &edge.measurement);
            let mut rhs = displacement;
            let w2 = edge.weight;

            let i_to = node_index.get(&edge.to).copied();
            let i_from = node_index.get(&edge.from).copied();
            if i_to.is_none() {
                rhs -= anchor_center.coords;
            }
            if i_from.is_none() {
                rhs += anchor_center.coords;
            }

            if let Some(j) = i_to {
                add_diag_block3(&mut h_dense, &mut triplets, j * 3, w2);
                for k in 0..3 {
                    atb[j * 3 + k] += w2 * rhs[k];
                }
            }
            if let Some(i) = i_from {
                add_diag_block3(&mut h_dense, &mut triplets, i * 3, w2);
                for k in 0..3 {
                    atb[i * 3 + k] -= w2 * rhs[k];
                }
            }
            if let (Some(j), Some(i)) = (i_to, i_from) {
                add_offdiag_block3(&mut h_dense, &mut triplets, j * 3, i * 3, -w2);
                add_offdiag_block3(&mut h_dense, &mut triplets, i * 3, j * 3, -w2);
            }
        }

        let solution = match linear_solver {
            LinearSolver::Dense => {
                let h = h_dense.expect("dense matrix initialized when LinearSolver::Dense");
                solve_normal_equations(&h, &atb)?
            }
            LinearSolver::Sparse => {
                let order = reordering::Reordering::fill_reducing(dim, 3, &triplets);
                // One-shot solve (no LM loop), so the symbolic analysis is not reused.
                solve_normal_equations_sparse(&triplets, dim, 3, &atb, 0.0, &order, &mut None)?
            }
        };

        let mut total_correction = 0.0;
        let mut max_correction: f64 = 0.0;
        for (&id, &i) in &node_index {
            let new_center = Point3::new(solution[i * 3], solution[i * 3 + 1], solution[i * 3 + 2]);
            let pose = self
                .poses
                .get_mut(&id)
                .ok_or(PoseGraphError::MissingNode(id))?;
            let old_center = pose.camera_center_world();
            let correction_norm = (new_center - old_center).norm();
            total_correction += correction_norm;
            if correction_norm > max_correction {
                max_correction = correction_norm;
            }
            let rotation_matrix = pose
                .world_to_camera
                .rotation
                .to_rotation_matrix()
                .into_inner();
            pose.world_to_camera.translation = -(rotation_matrix * new_center.coords);
        }

        let cost_after = self.translation_cost();
        let mean_translation_correction = if variable_count > 0 {
            total_correction / variable_count as f64
        } else {
            0.0
        };

        Ok(PoseGraphOptimizationStep {
            anchor_id,
            edge_count: self.edges.len(),
            variable_count,
            cost_before,
            cost_after,
            mean_translation_correction,
            max_translation_correction: max_correction,
        })
    }

    /// Sum of squared SE(3) residuals: r_e = log(meas_e⁻¹ · T_to · T_from⁻¹),
    /// weighted by `edge.weight`. Unlike [`Self::translation_cost`], this
    /// includes both the translation and rotation components of every edge.
    pub fn se3_cost(&self) -> f64 {
        self.robust_se3_cost(&RobustKernel::None)
    }

    /// Robust SE(3) cost: `Σ_e edge.weight · ρ(||r_e||²)` where `ρ` is the
    /// supplied [`RobustKernel`]. With [`RobustKernel::None`] this matches
    /// [`Self::se3_cost`].
    pub fn robust_se3_cost(&self, kernel: &RobustKernel) -> f64 {
        let mut total = 0.0;
        for edge in &self.edges {
            let (Some(from), Some(to)) = (self.poses.get(&edge.from), self.poses.get(&edge.to))
            else {
                continue;
            };
            let predicted = to.world_to_camera.compose(&from.world_to_camera.inverse());
            let r = edge.measurement.inverse().compose(&predicted).log();
            total += match &edge.information {
                // Anisotropic: robust kernel operates on the Mahalanobis
                // distance rᵀΩr; the scalar weight is folded into Ω.
                Some(omega) => kernel.cost((r.transpose() * omega * r)[(0, 0)]),
                // Isotropic: weight scales the kernel output, kernel sees ‖r‖².
                None => edge.weight * kernel.cost(r.norm_squared()),
            };
        }
        total
    }

    /// Robust SE(3) cost with an optional per-edge multiplier (`gnc_weights`,
    /// indexed by edge position). Identical to [`Self::robust_se3_cost`] when
    /// `gnc_weights` is `None`; with weights it is the *weighted* objective
    /// `Σ wᵢ · ρ(sᵢ)` that the Graduated Non-Convexity driver minimizes at a
    /// fixed `μ` level (the Black-Rangarajan inner problem), where `wᵢ` is the
    /// closed-form GNC weight and the kernel is [`RobustKernel::None`] (GNC
    /// supersedes the M-estimator). See [`crate::gnc`] and
    /// [`Self::optimize_se3_gnc`].
    fn robust_se3_cost_weighted(&self, kernel: &RobustKernel, gnc_weights: Option<&[f64]>) -> f64 {
        let mut total = 0.0;
        for (idx, edge) in self.edges.iter().enumerate() {
            let (Some(from), Some(to)) = (self.poses.get(&edge.from), self.poses.get(&edge.to))
            else {
                continue;
            };
            let predicted = to.world_to_camera.compose(&from.world_to_camera.inverse());
            let r = edge.measurement.inverse().compose(&predicted).log();
            let gw = gnc_weights.map_or(1.0, |w| w[idx]);
            total += gw
                * match &edge.information {
                    Some(omega) => kernel.cost((r.transpose() * omega * r)[(0, 0)]),
                    None => edge.weight * kernel.cost(r.norm_squared()),
                };
        }
        total
    }

    /// Per-edge (whitened) squared residual `sᵢ`, indexed by edge position —
    /// the same quantity the [`RobustKernel`] sees: the Mahalanobis distance
    /// `rᵀΩr` for an edge carrying a full information matrix, else `‖r‖²`.
    /// Edges referencing a missing node contribute `0.0` so the vector stays
    /// aligned with [`Self::edges`]. Used by the GNC driver to reweight edges.
    fn edge_squared_residuals(&self) -> Vec<f64> {
        self.edges
            .iter()
            .map(|edge| {
                let (Some(from), Some(to)) = (self.poses.get(&edge.from), self.poses.get(&edge.to))
                else {
                    return 0.0;
                };
                let predicted = to.world_to_camera.compose(&from.world_to_camera.inverse());
                let r = edge.measurement.inverse().compose(&predicted).log();
                match &edge.information {
                    Some(omega) => (r.transpose() * omega * r)[(0, 0)],
                    None => r.norm_squared(),
                }
            })
            .collect()
    }

    /// Assemble the Gauss-Newton normal equations `(H, g)` for the SE(3) pose
    /// graph: the robust-weighted `H = Σ Jᵀ Ω J` and gradient `g = Σ Jᵀ Ω r`
    /// over all edges, in the `node_index` variable layout (`dim = 6 · #vars`).
    ///
    /// Extracted from [`Self::optimize_se3_iterative`] so both the plain
    /// optimizer and the GNC driver share one assembly. `gnc_weights` is an
    /// optional per-edge multiplier (indexed by edge position): `None`
    /// reproduces the plain assembly bit-for-bit; `Some(w)` scales each edge's
    /// contribution by `wᵢ ∈ [0, 1]` for the GNC inner solve (see
    /// [`crate::gnc`]). The isotropic path keeps the legacy semantics
    /// (`weight` outside the kernel, kernel on `‖r‖²`); the anisotropic path
    /// folds `Ω` into both `JᵀJ` and `Jᵀr` and lets the kernel see the
    /// Mahalanobis distance `rᵀΩr`.
    fn assemble_se3_system(
        &self,
        node_index: &BTreeMap<u64, usize>,
        dim: usize,
        kernel: &RobustKernel,
        gnc_weights: Option<&[f64]>,
        linear_solver: LinearSolver,
    ) -> (NormalEquations6, DVector<f64>) {
        let mut builder = NormalEquations6::new(dim, linear_solver, self.edges.len());
        let mut g = DVector::<f64>::zeros(dim);

        for (idx, edge) in self.edges.iter().enumerate() {
            let t_from = &self.poses[&edge.from].world_to_camera;
            let t_to = &self.poses[&edge.to].world_to_camera;
            let predicted = t_to.compose(&t_from.inverse());
            let r = edge.measurement.inverse().compose(&predicted).log();
            let ad_from = t_from.adjoint();
            let (weight, ata, atr) = match &edge.information {
                Some(omega) => {
                    let robust_weight = kernel.weight((r.transpose() * omega * r)[(0, 0)]);
                    let oa = ad_from.transpose() * omega;
                    (robust_weight, oa * ad_from, oa * r)
                }
                None => {
                    let robust_weight = kernel.weight(r.norm_squared());
                    (
                        edge.weight * robust_weight,
                        ad_from.transpose() * ad_from,
                        ad_from.transpose() * r,
                    )
                }
            };
            // GNC reweighting: a multiplier in [0, 1] (1.0 when not running
            // GNC) that scales the whole edge contribution, rejecting outliers
            // as `wᵢ → 0`.
            let weight = weight * gnc_weights.map_or(1.0, |w| w[idx]);

            let i_from = node_index.get(&edge.from).copied();
            let i_to = node_index.get(&edge.to).copied();

            if let Some(j) = i_to {
                builder.add_block6(j * 6, j * 6, weight, &ata);
                add_segment6(&mut g, j * 6, weight, &atr);
            }
            if let Some(i) = i_from {
                builder.add_block6(i * 6, i * 6, weight, &ata);
                add_segment6(&mut g, i * 6, -weight, &atr);
            }
            if let (Some(j), Some(i)) = (i_to, i_from) {
                let cross = -ata;
                let cross_t = cross.transpose();
                builder.add_block6(j * 6, i * 6, weight, &cross);
                builder.add_block6(i * 6, j * 6, weight, &cross_t);
            }
        }

        (builder, g)
    }

    /// Chordal (Frobenius-relaxed) cost of the current node rotations:
    /// `Σ_e w_e · ‖R_to − R_meas_e · R_from‖_F²`, where `R_*` is the rotation
    /// of each node's `world_to_camera` and `R_meas_e` the rotation of the
    /// edge measurement. This is the objective minimized by
    /// [`Self::initialize_rotations_chordal`]; unlike [`Self::se3_cost`] it
    /// ignores translation and uses the chordal (embedded-Euclidean) metric on
    /// SO(3) rather than the geodesic one, so it is a convex function of the
    /// relaxed (unconstrained 3×3) rotation variables.
    pub fn chordal_rotation_cost(&self) -> f64 {
        let mut total = 0.0;
        for edge in &self.edges {
            let (Some(from), Some(to)) = (self.poses.get(&edge.from), self.poses.get(&edge.to))
            else {
                continue;
            };
            let r_from = from
                .world_to_camera
                .rotation
                .to_rotation_matrix()
                .into_inner();
            let r_to = to
                .world_to_camera
                .rotation
                .to_rotation_matrix()
                .into_inner();
            let r_meas = edge.measurement.rotation.to_rotation_matrix().into_inner();
            total += chordal_rotation_weight(edge) * (r_to - r_meas * r_from).norm_squared();
        }
        total
    }

    /// Initialize node rotations by solving the *chordal relaxation* of the
    /// rotation-only sub-problem (Carlone et al., "Initialization Techniques
    /// for 3D SLAM", ICRA 2015). On hard 3D datasets the SE(3) cost surface is
    /// strongly non-convex in rotation, so a full solve started from raw
    /// odometry stalls in a poor basin; seeding it with the chordal solution
    /// lands it near the global optimum.
    ///
    /// Each edge contributes the residual `R_to − R_meas · R_from` measured in
    /// the embedded-Euclidean (Frobenius) metric. Relaxing every `R_i` from
    /// `SO(3)` to an unconstrained `3×3` matrix makes the objective a single
    /// linear least-squares problem; the per-node `9`-vector splits into three
    /// independent `3`-vector systems (one per rotation column) that share the
    /// *same* `3n × 3n` normal matrix — so this factors once and solves three
    /// right-hand sides. Each relaxed `3×3` block is then projected back onto
    /// `SO(3)` with an SVD (`R = U·diag(1,1,det(UVᵀ))·Vᵀ`).
    ///
    /// The anchor's rotation is held fixed (it fixes the global gauge). Each
    /// node's camera *center* is preserved — only its orientation is replaced —
    /// so this is safe to call standalone, though the intended flow is
    /// chordal-rotation → [`Self::optimize_translations_once_with`] →
    /// [`Self::optimize_se3_iterative`].
    pub fn initialize_rotations_chordal(
        &mut self,
        linear_solver: LinearSolver,
    ) -> Result<ChordalRotationInit, PoseGraphError> {
        let anchor_id = self.anchor.ok_or(PoseGraphError::NoAnchor)?;
        let anchor_pose = self
            .poses
            .get(&anchor_id)
            .ok_or(PoseGraphError::MissingNode(anchor_id))?;
        let r_anchor = anchor_pose
            .world_to_camera
            .rotation
            .to_rotation_matrix()
            .into_inner();
        if self.edges.is_empty() {
            return Err(PoseGraphError::NoEdges);
        }
        for edge in &self.edges {
            if !self.poses.contains_key(&edge.from) {
                return Err(PoseGraphError::MissingNode(edge.from));
            }
            if !self.poses.contains_key(&edge.to) {
                return Err(PoseGraphError::MissingNode(edge.to));
            }
        }

        let mut node_index: BTreeMap<u64, usize> = BTreeMap::new();
        for &id in self.poses.keys() {
            if id == anchor_id {
                continue;
            }
            let next = node_index.len();
            node_index.insert(id, next);
        }
        let variable_count = node_index.len();
        if variable_count == 0 {
            return Err(PoseGraphError::NoVariables);
        }

        let cost_before = self.chordal_rotation_cost();

        // Assemble the shared 3n×3n normal matrix and three right-hand sides
        // (the three columns of the stacked rotation matrices). The matrix is
        // identical across columns because it depends only on the (orthonormal)
        // measured rotations and the edge weights; each anchor-incident edge
        // shifts one column's right-hand side by that column of R_anchor.
        let dim = variable_count * 3;
        let mut h_dense = match linear_solver {
            LinearSolver::Dense => Some(DMatrix::<f64>::zeros(dim, dim)),
            LinearSolver::Sparse => None,
        };
        let mut triplets: Vec<(usize, usize, f64)> = match linear_solver {
            LinearSolver::Dense => Vec::new(),
            LinearSolver::Sparse => Vec::with_capacity(self.edges.len() * 36),
        };
        let mut rhs = DMatrix::<f64>::zeros(dim, 3);

        for edge in &self.edges {
            let r_meas = edge.measurement.rotation.to_rotation_matrix().into_inner();
            let w = chordal_rotation_weight(edge);

            let i_to = node_index.get(&edge.to).copied();
            let i_from = node_index.get(&edge.from).copied();

            if let Some(j) = i_to {
                add_diag_block3(&mut h_dense, &mut triplets, j * 3, w);
            }
            if let Some(i) = i_from {
                add_diag_block3(&mut h_dense, &mut triplets, i * 3, w);
            }
            match (i_to, i_from) {
                (Some(j), Some(i)) => {
                    // Both endpoints free: off-diagonal coupling -w·R_meas.
                    add_dense_block3(&mut h_dense, &mut triplets, j * 3, i * 3, &(-w * r_meas));
                    add_dense_block3(
                        &mut h_dense,
                        &mut triplets,
                        i * 3,
                        j * 3,
                        &(-w * r_meas.transpose()),
                    );
                }
                (Some(j), None) => {
                    // `from` is the anchor: g_to += w · R_meas · col(R_anchor).
                    let contribution = w * r_meas * r_anchor;
                    for c in 0..3 {
                        for k in 0..3 {
                            rhs[(j * 3 + k, c)] += contribution[(k, c)];
                        }
                    }
                }
                (None, Some(i)) => {
                    // `to` is the anchor: g_from += w · R_measᵀ · col(R_anchor).
                    let contribution = w * r_meas.transpose() * r_anchor;
                    for c in 0..3 {
                        for k in 0..3 {
                            rhs[(i * 3 + k, c)] += contribution[(k, c)];
                        }
                    }
                }
                (None, None) => {}
            }
        }

        let solution = match linear_solver {
            LinearSolver::Dense => {
                let h = h_dense.expect("dense matrix initialized when LinearSolver::Dense");
                let chol = h.clone().cholesky().ok_or(PoseGraphError::SingularSystem)?;
                chol.solve(&rhs)
            }
            LinearSolver::Sparse => {
                let order = reordering::Reordering::fill_reducing(dim, 3, &triplets);
                solve_normal_equations_sparse_multi(&triplets, dim, 3, &rhs, &order)?
            }
        };

        // Reshape each node's solved columns into a 3×3, project onto SO(3),
        // and replace the node's orientation while keeping its camera center.
        let mut max_rotation_update_rad: f64 = 0.0;
        for (&id, &i) in &node_index {
            let mut block = Matrix3::<f64>::zeros();
            for c in 0..3 {
                for k in 0..3 {
                    block[(k, c)] = solution[(i * 3 + k, c)];
                }
            }
            let projected = project_to_so3(&block).ok_or(PoseGraphError::SingularSystem)?;
            let new_rotation =
                UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(projected));

            let pose = self
                .poses
                .get_mut(&id)
                .ok_or(PoseGraphError::MissingNode(id))?;
            let center = pose.camera_center_world();
            let delta = new_rotation
                .rotation_to(&pose.world_to_camera.rotation)
                .angle();
            max_rotation_update_rad = max_rotation_update_rad.max(delta);
            pose.world_to_camera.rotation = new_rotation;
            // Re-derive translation from the preserved center: t = -R·center.
            pose.world_to_camera.translation = -(projected * center.coords);
        }

        let cost_after = self.chordal_rotation_cost();
        Ok(ChordalRotationInit {
            anchor_id,
            edge_count: self.edges.len(),
            variable_count,
            cost_before,
            cost_after,
            max_rotation_update_deg: max_rotation_update_rad.to_degrees(),
        })
    }

    /// Run a full SE(3) Gauss-Newton optimization with right-perturbation
    /// updates `T_i ← T_i · Exp(δ_i)`. Uses the first-order BCH approximation
    /// `J_r⁻¹(r) ≈ I`, so each edge contributes:
    ///
    /// - residual: `r_e = log(meas_e⁻¹ · T_to · T_from⁻¹)` (6-vector),
    /// - Jacobians: `∂r/∂δ_to = Ad(T_from)`, `∂r/∂δ_from = -Ad(T_from)`.
    ///
    /// The anchor pose is held fixed; all other poses are updated. Returns the
    /// per-iteration cost trace plus a `converged` flag derived from the
    /// configured tolerances.
    pub fn optimize_se3_iterative(
        &mut self,
        config: &PoseGraphSe3Config,
    ) -> Result<PoseGraphSe3Result, PoseGraphError> {
        let anchor_id = self.anchor.ok_or(PoseGraphError::NoAnchor)?;
        if !self.poses.contains_key(&anchor_id) {
            return Err(PoseGraphError::MissingNode(anchor_id));
        }
        if self.edges.is_empty() {
            return Err(PoseGraphError::NoEdges);
        }
        for edge in &self.edges {
            if !self.poses.contains_key(&edge.from) {
                return Err(PoseGraphError::MissingNode(edge.from));
            }
            if !self.poses.contains_key(&edge.to) {
                return Err(PoseGraphError::MissingNode(edge.to));
            }
        }

        let mut node_index: BTreeMap<u64, usize> = BTreeMap::new();
        for &id in self.poses.keys() {
            if id == anchor_id {
                continue;
            }
            let next = node_index.len();
            node_index.insert(id, next);
        }
        let variable_count = node_index.len();
        if variable_count == 0 {
            return Err(PoseGraphError::NoVariables);
        }

        let kernel = config.robust_kernel;
        // `initial_cost` records the true starting point — measured before any
        // seeding — so the reported reduction reflects the full improvement,
        // including the (often large) drop the chordal step front-loads.
        let initial_cost = self.robust_se3_cost(&kernel);

        // Optional chordal rotation seeding: solve the relaxed rotation
        // sub-problem to a globally-consistent orientation and re-derive
        // translations before the non-convex SE(3) solve. Best-effort — a
        // singular relaxation is skipped, leaving the unmodified estimate, so
        // seeding can never turn a solvable problem into a failure. The
        // rotation re-derivation already restores translations from the
        // preserved camera centers, so the translation LS is a further refine
        // whose failure is also harmless.
        if config.chordal_init
            && self
                .initialize_rotations_chordal(config.linear_solver)
                .is_ok()
        {
            let _ = self.optimize_translations_once_with(config.linear_solver);
        }

        let mut iterations: Vec<PoseGraphSe3IterationStats> =
            Vec::with_capacity(config.max_iterations);
        let mut converged = false;
        // The LM accept test compares against the *seeded* cost, so the loop
        // starts from the post-seed state rather than the pre-seed `initial_cost`.
        let mut current_cost = self.robust_se3_cost(&kernel);
        let mut lambda = config.initial_lambda.unwrap_or(0.0);
        let dim = variable_count * 6;
        // The fill-reducing ordering depends only on the (iteration-invariant)
        // sparsity pattern, so compute it lazily on the first sparse solve and
        // reuse it for the rest of the optimization.
        let mut order_cache: Option<reordering::Reordering> = None;
        // The normal-equations sparsity pattern is iteration-invariant, so the
        // block-Cholesky symbolic factorization is analyzed once and reused
        // alongside the fill-reducing order across all LM iterations.
        let mut symbolic_cache: Option<block_cholesky::BlockSymbolic> = None;

        for iteration in 0..config.max_iterations {
            let (builder, g) =
                self.assemble_se3_system(&node_index, dim, &kernel, None, config.linear_solver);

            let neg_g = -&g;
            let delta = builder.solve(lambda, &neg_g, &mut order_cache, &mut symbolic_cache)?;

            // Tentatively apply the step so we can evaluate the new cost.
            let mut max_step_norm: f64 = 0.0;
            let cost_before = current_cost;
            let saved_poses = if config.initial_lambda.is_some() {
                Some(self.poses.clone())
            } else {
                None
            };
            for (&id, &i) in &node_index {
                let block = i * 6;
                let mut xi = Vector6::<f64>::zeros();
                for k in 0..6 {
                    xi[k] = delta[block + k];
                }
                let step = xi.norm();
                if step > max_step_norm {
                    max_step_norm = step;
                }
                let pose = self
                    .poses
                    .get_mut(&id)
                    .ok_or(PoseGraphError::MissingNode(id))?;
                pose.world_to_camera = pose.world_to_camera.compose(&SE3::exp(&xi));
            }

            let cost_after = self.robust_se3_cost(&kernel);
            let step_accepted = match config.initial_lambda {
                None => true,
                Some(_) => cost_after < cost_before,
            };

            if !step_accepted {
                if let Some(saved) = saved_poses {
                    self.poses = saved;
                }
                lambda = (lambda * config.lambda_increase_factor).min(config.max_lambda);
                iterations.push(PoseGraphSe3IterationStats {
                    iteration,
                    cost_before,
                    cost_after,
                    max_step_norm,
                    lambda,
                    step_accepted: false,
                });
                if lambda >= config.max_lambda {
                    // λ saturated without finding a downhill step → bail.
                    break;
                }
                continue;
            }

            iterations.push(PoseGraphSe3IterationStats {
                iteration,
                cost_before,
                cost_after,
                max_step_norm,
                lambda,
                step_accepted: true,
            });
            current_cost = cost_after;
            if config.initial_lambda.is_some() {
                lambda = (lambda * config.lambda_decrease_factor).max(config.min_lambda);
            }

            if max_step_norm < config.step_tolerance {
                converged = true;
                break;
            }
            if (cost_before - cost_after).abs() < config.cost_tolerance {
                converged = true;
                break;
            }
        }

        Ok(PoseGraphSe3Result {
            anchor_id,
            edge_count: self.edges.len(),
            variable_count,
            initial_cost,
            final_cost: current_cost,
            iterations,
            converged,
        })
    }

    /// Outlier-robust SE(3) pose-graph optimization via Graduated Non-Convexity
    /// (GNC; see [`crate::gnc`]). Use this instead of [`Self::optimize_se3_iterative`]
    /// when loop closures may be **wrong** (perceptual aliasing, place-recognition
    /// false positives): a single bad loop closure pulls a plain least-squares —
    /// or even a Huber/Cauchy IRLS — solve into a corrupted basin, whereas GNC
    /// anneals from a convex surrogate that trusts every edge to the true robust
    /// cost that rejects outliers, recovering the inlier trajectory.
    ///
    /// `config` supplies the same SE(3) LM settings as
    /// [`Self::optimize_se3_iterative`] (linear solver, `λ` schedule,
    /// tolerances, chordal seeding); its `robust_kernel` is ignored — GNC
    /// supersedes the M-estimator and the inner solve runs on the GNC-weighted
    /// least-squares cost. `gnc` selects the surrogate family, the inlier scale
    /// `c`, the `μ` annealing factor, the outer-level cap, and the number of
    /// inner LM iterations per level.
    ///
    /// Each outer level reweights every edge by its closed-form GNC weight at
    /// the current `μ`, runs a bounded weighted-LS solve, then sharpens `μ` one
    /// geometric step. The fill-reducing order and block-Cholesky symbolic
    /// factorization are `μ`-invariant and reused across all levels. The
    /// returned [`PoseGraphGncResult::edge_weights`] is the final per-edge
    /// inlier/outlier classification.
    pub fn optimize_se3_gnc(
        &mut self,
        config: &PoseGraphSe3Config,
        gnc: &gnc::GncConfig,
    ) -> Result<PoseGraphGncResult, PoseGraphError> {
        let anchor_id = self.anchor.ok_or(PoseGraphError::NoAnchor)?;
        if !self.poses.contains_key(&anchor_id) {
            return Err(PoseGraphError::MissingNode(anchor_id));
        }
        if self.edges.is_empty() {
            return Err(PoseGraphError::NoEdges);
        }
        for edge in &self.edges {
            if !self.poses.contains_key(&edge.from) {
                return Err(PoseGraphError::MissingNode(edge.from));
            }
            if !self.poses.contains_key(&edge.to) {
                return Err(PoseGraphError::MissingNode(edge.to));
            }
        }

        let mut node_index: BTreeMap<u64, usize> = BTreeMap::new();
        for &id in self.poses.keys() {
            if id == anchor_id {
                continue;
            }
            let next = node_index.len();
            node_index.insert(id, next);
        }
        let variable_count = node_index.len();
        if variable_count == 0 {
            return Err(PoseGraphError::NoVariables);
        }
        let dim = variable_count * 6;

        // GNC replaces the M-estimator, so the inner solve is plain weighted
        // least squares (kernel `None`); the GNC weights carry the robustness.
        let kernel = RobustKernel::None;
        // Report the pre-seed plain L2 cost so the reduction reflects the full
        // improvement, mirroring `optimize_se3_iterative`.
        let initial_cost = self.robust_se3_cost(&kernel);

        // Optional chordal rotation seeding (best-effort). GNC's first surrogate
        // is convex (trusts every edge), so seeding from the all-edge rotation
        // least squares is consistent with it; as μ sharpens, outliers are
        // down-weighted and the estimate moves off any seed they corrupted.
        if config.chordal_init
            && self
                .initialize_rotations_chordal(config.linear_solver)
                .is_ok()
        {
            let _ = self.optimize_translations_once_with(config.linear_solver);
        }

        // Convex first surrogate: initialize μ from the largest seeded residual.
        // The same seeded residuals optionally drive the MAD auto-estimate of
        // the inlier scale `c` (with `gnc.c` as a floor), so the inlier/outlier
        // boundary tracks this graph's noise level instead of a hand-set value.
        let squared_residuals = self.edge_squared_residuals();
        let s_max = squared_residuals.iter().copied().fold(0.0_f64, f64::max);
        let effective_gnc = match gnc.auto_scale {
            Some(k) => {
                let c = gnc::estimate_scale_mad(&squared_residuals, k)
                    .map_or(gnc.c, |est| est.max(gnc.c));
                gnc::GncConfig { c, ..*gnc }
            }
            None => *gnc,
        };
        let mut inlier_scale = effective_gnc.c;
        let mut state = gnc::GncState::new(&effective_gnc, s_max);
        let mut gnc_weights = vec![1.0_f64; self.edges.len()];

        // The sparsity pattern is μ-invariant, so the fill-reducing order and
        // block-Cholesky symbolic factorization are analyzed once and reused.
        let mut order_cache: Option<reordering::Reordering> = None;
        let mut symbolic_cache: Option<block_cholesky::BlockSymbolic> = None;

        let mut converged = false;
        let mut outer_iterations = 0usize;
        let max_outer = gnc.max_outer.max(1);

        for _ in 0..max_outer {
            outer_iterations += 1;
            // A level entered already-terminal is solving at the true robust
            // cost — run it, then stop (guarantees one solve at terminal μ).
            let terminal_level = state.is_terminal();

            // Black-Rangarajan weight update at the current μ.
            let residuals = self.edge_squared_residuals();
            // Adaptive inlier scale: re-derive `c` from the current residuals
            // each level (configured `c` as a floor). Level 0 reproduces the
            // one-shot estimate; later levels tighten as outliers are
            // suppressed and inlier residuals shrink.
            if gnc.auto_scale_readapt {
                if let Some(k) = gnc.auto_scale {
                    if let Some(est) = gnc::estimate_scale_mad(&residuals, k) {
                        let c = est.max(gnc.c);
                        state.set_inlier_scale(c);
                        inlier_scale = c;
                    }
                }
            }
            for (i, &s) in residuals.iter().enumerate() {
                gnc_weights[i] = state.weight(s);
            }

            // Inner weighted-LS solve at fixed weights (a few LM steps).
            let mut lambda = config.initial_lambda.unwrap_or(0.0);
            let mut current_cost = self.robust_se3_cost_weighted(&kernel, Some(&gnc_weights));
            for _ in 0..gnc.inner_iterations.max(1) {
                let (builder, g) = self.assemble_se3_system(
                    &node_index,
                    dim,
                    &kernel,
                    Some(&gnc_weights),
                    config.linear_solver,
                );
                let neg_g = -&g;
                let delta = builder.solve(lambda, &neg_g, &mut order_cache, &mut symbolic_cache)?;

                let saved_poses = if config.initial_lambda.is_some() {
                    Some(self.poses.clone())
                } else {
                    None
                };
                let mut max_step_norm: f64 = 0.0;
                for (&id, &i) in &node_index {
                    let block = i * 6;
                    let mut xi = Vector6::<f64>::zeros();
                    for k in 0..6 {
                        xi[k] = delta[block + k];
                    }
                    max_step_norm = max_step_norm.max(xi.norm());
                    let pose = self
                        .poses
                        .get_mut(&id)
                        .ok_or(PoseGraphError::MissingNode(id))?;
                    pose.world_to_camera = pose.world_to_camera.compose(&SE3::exp(&xi));
                }

                let cost_after = self.robust_se3_cost_weighted(&kernel, Some(&gnc_weights));
                let accepted = match config.initial_lambda {
                    None => true,
                    Some(_) => cost_after < current_cost,
                };
                if !accepted {
                    if let Some(saved) = saved_poses {
                        self.poses = saved;
                    }
                    lambda = (lambda * config.lambda_increase_factor).min(config.max_lambda);
                    if lambda >= config.max_lambda {
                        break;
                    }
                    continue;
                }
                current_cost = cost_after;
                if config.initial_lambda.is_some() {
                    lambda = (lambda * config.lambda_decrease_factor).max(config.min_lambda);
                }
                if max_step_norm < config.step_tolerance {
                    break;
                }
            }

            if terminal_level {
                converged = true;
                break;
            }
            state.anneal();
        }

        // Final per-edge classification at the converged estimate and μ.
        let residuals = self.edge_squared_residuals();
        for (i, &s) in residuals.iter().enumerate() {
            gnc_weights[i] = state.weight(s);
        }
        let final_cost = self.robust_se3_cost(&kernel);
        // Plain L2 over retained inliers: binarize the weights at the inlier
        // cutoff and reuse the weighted cost.
        const INLIER_THRESHOLD: f64 = 0.5;
        let inlier_mask: Vec<f64> = gnc_weights
            .iter()
            .map(|&w| if w >= INLIER_THRESHOLD { 1.0 } else { 0.0 })
            .collect();
        let inlier_cost = self.robust_se3_cost_weighted(&kernel, Some(&inlier_mask));

        Ok(PoseGraphGncResult {
            anchor_id,
            edge_count: self.edges.len(),
            variable_count,
            initial_cost,
            final_cost,
            inlier_cost,
            inlier_scale,
            outer_iterations,
            converged,
            edge_weights: gnc_weights,
        })
    }

    /// Recover the marginal covariance of every non-anchor pose from the
    /// information matrix `Λ = JᵀΩJ` assembled at the *current* estimate (run a
    /// solve first so this is the covariance at the optimum). Uses the
    /// Takahashi sparse-inverse recursion ([`crate::covariance`]) so the dense
    /// `Λ⁻¹` is never formed. Each `Matrix6` is the covariance of that pose in
    /// its local SE(3) tangent (the `[ω | ρ]` ordering of
    /// [`SE3::log`](visloc_core::geometry::SE3::log)); the gauge-fixed anchor
    /// has no free covariance and is omitted from the result.
    ///
    /// Useful for loop-closure gating (gate a candidate on the relative
    /// uncertainty between its endpoints) and uncertainty-aware fusion. Errors
    /// mirror the solvers: [`PoseGraphError::NoAnchor`] / `NoEdges` /
    /// `NoVariables`, and [`PoseGraphError::SingularSystem`] when `Λ` is not
    /// positive-definite (a rank-deficient / disconnected graph).
    pub fn pose_marginal_covariances(&self) -> Result<BTreeMap<u64, Matrix6<f64>>, PoseGraphError> {
        let anchor_id = self.anchor.ok_or(PoseGraphError::NoAnchor)?;
        if !self.poses.contains_key(&anchor_id) {
            return Err(PoseGraphError::MissingNode(anchor_id));
        }
        if self.edges.is_empty() {
            return Err(PoseGraphError::NoEdges);
        }
        // Free-variable indexing (anchor excluded), identical to the solvers.
        let mut node_index: BTreeMap<u64, usize> = BTreeMap::new();
        for &id in self.poses.keys() {
            if id == anchor_id {
                continue;
            }
            let next = node_index.len();
            node_index.insert(id, next);
        }
        let variable_count = node_index.len();
        if variable_count == 0 {
            return Err(PoseGraphError::NoVariables);
        }
        let dim = variable_count * 6;

        // Assemble the dense information matrix at the current estimate (plain
        // L2 — kernel `None`, no GNC weights). Forcing the dense backend so the
        // 6×6-block recursion reads `Λ` directly.
        let (builder, _g) = self.assemble_se3_system(
            &node_index,
            dim,
            &RobustKernel::None,
            None,
            LinearSolver::Dense,
        );
        let lambda = match builder {
            NormalEquations6::Dense(h) => h,
            NormalEquations6::Sparse { .. } => unreachable!("forced the dense backend above"),
        };

        let blocks = covariance::marginal_block_covariances(&lambda, 6)
            .ok_or(PoseGraphError::SingularSystem)?;
        let mut out = BTreeMap::new();
        for (&id, &idx) in &node_index {
            let block = &blocks[idx];
            out.insert(id, Matrix6::from_fn(|r, c| block[(r, c)]));
        }
        Ok(out)
    }

    /// Covariance of the *relative* pose `a → b` implied by the current estimate
    /// — the joint marginal of the two pose blocks reduced to their difference
    /// (`Σ_aa + Σ_bb − Σ_ab − Σ_abᵀ`, the first-order tangent approximation; see
    /// [`covariance::relative_covariance`]). A gauge-fixed anchor endpoint
    /// contributes a zero block (its frame is certain), so the relative
    /// covariance to the anchor is just the other pose's marginal.
    ///
    /// This is the prediction covariance a loop-closure innovation is gated
    /// against: a candidate asserting a relative pose far outside this
    /// uncertainty (a confident-but-wrong place recognition between two
    /// well-localized frames) is statistically implausible. Recovers the full
    /// `Σ` to read the cross-block, so it is dense/`O(n³)` for now — fine for the
    /// occasional gate, not a per-edge inner loop. Errors mirror
    /// [`Self::pose_marginal_covariances`]; also
    /// [`PoseGraphError::MissingNode`] when `a` or `b` is absent.
    pub fn relative_pose_covariance(&self, a: u64, b: u64) -> Result<Matrix6<f64>, PoseGraphError> {
        let anchor_id = self.anchor.ok_or(PoseGraphError::NoAnchor)?;
        if !self.poses.contains_key(&anchor_id) {
            return Err(PoseGraphError::MissingNode(anchor_id));
        }
        if !self.poses.contains_key(&a) {
            return Err(PoseGraphError::MissingNode(a));
        }
        if !self.poses.contains_key(&b) {
            return Err(PoseGraphError::MissingNode(b));
        }
        if self.edges.is_empty() {
            return Err(PoseGraphError::NoEdges);
        }
        let mut node_index: BTreeMap<u64, usize> = BTreeMap::new();
        for &id in self.poses.keys() {
            if id == anchor_id {
                continue;
            }
            let next = node_index.len();
            node_index.insert(id, next);
        }
        let variable_count = node_index.len();
        if variable_count == 0 {
            return Err(PoseGraphError::NoVariables);
        }
        let dim = variable_count * 6;
        let (builder, _g) = self.assemble_se3_system(
            &node_index,
            dim,
            &RobustKernel::None,
            None,
            LinearSolver::Dense,
        );
        let lambda = match builder {
            NormalEquations6::Dense(h) => h,
            NormalEquations6::Sparse { .. } => unreachable!("forced the dense backend above"),
        };
        // Full covariance (the a↔b cross-block may be off the factor pattern);
        // a fixed-anchor endpoint reads a zero block.
        let sigma =
            covariance::sparse_inverse(&lambda, 0.0).ok_or(PoseGraphError::SingularSystem)?;
        let ia = node_index.get(&a);
        let ib = node_index.get(&b);
        // Assemble the 12×12 joint [[Σaa, Σab], [Σba, Σbb]] (anchor → zeros).
        let mut joint = DMatrix::<f64>::zeros(12, 12);
        let mut copy_block =
            |dst_r: usize, dst_c: usize, ir: Option<&usize>, ic: Option<&usize>| {
                if let (Some(&ri), Some(&ci)) = (ir, ic) {
                    for r in 0..6 {
                        for c in 0..6 {
                            joint[(dst_r + r, dst_c + c)] = sigma[(ri * 6 + r, ci * 6 + c)];
                        }
                    }
                }
            };
        copy_block(0, 0, ia, ia); // Σaa
        copy_block(6, 6, ib, ib); // Σbb
        copy_block(0, 6, ia, ib); // Σab
        copy_block(6, 0, ib, ia); // Σba
        let rel = covariance::relative_covariance(&joint, 6);
        Ok(Matrix6::from_fn(|r, c| rel[(r, c)]))
    }

    /// Serialize this pose graph to a plain-text format. The format is
    /// line-oriented and human-readable so it doubles as a debug dump:
    ///
    /// ```text
    /// # visloc-rs PoseGraph v1
    /// P <id> <qw> <qx> <qy> <qz> <tx> <ty> <tz>
    /// ...
    /// A <id>
    /// E <from> <to> <kind:0|1> <weight> <qw> <qx> <qy> <qz> <tx> <ty> <tz>
    /// ...
    /// ```
    ///
    /// `kind = 0` is `Sequential`, `kind = 1` is `LoopClosure`. Lines
    /// starting with `#` and blank lines are ignored on read. Round-trips
    /// through [`Self::load_text`] without precision loss within `f64`'s
    /// `{:.17e}` representation.
    pub fn save_text(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let mut text = String::from("# visloc-rs PoseGraph v1\n");
        for (id, pose) in self.poses.iter() {
            let q = pose.world_to_camera.rotation.into_inner();
            let t = &pose.world_to_camera.translation;
            text.push_str(&format!(
                "P {id} {qw:.17e} {qx:.17e} {qy:.17e} {qz:.17e} {tx:.17e} {ty:.17e} {tz:.17e}\n",
                qw = q.w,
                qx = q.i,
                qy = q.j,
                qz = q.k,
                tx = t.x,
                ty = t.y,
                tz = t.z,
            ));
        }
        if let Some(anchor) = self.anchor {
            text.push_str(&format!("A {anchor}\n"));
        }
        for edge in &self.edges {
            let kind: u8 = match edge.kind {
                PoseGraphEdgeKind::Sequential => 0,
                PoseGraphEdgeKind::LoopClosure => 1,
            };
            let q = edge.measurement.rotation.into_inner();
            let t = &edge.measurement.translation;
            text.push_str(&format!(
                "E {from} {to} {kind} {weight:.17e} {qw:.17e} {qx:.17e} {qy:.17e} {qz:.17e} {tx:.17e} {ty:.17e} {tz:.17e}\n",
                from = edge.from, to = edge.to, weight = edge.weight,
                qw = q.w, qx = q.i, qy = q.j, qz = q.k, tx = t.x, ty = t.y, tz = t.z,
            ));
        }
        std::fs::write(path, text)
    }

    /// Inverse of [`Self::save_text`]. Returns `PoseGraphParseError`
    /// on syntactic problems (unknown line tag, missing column, bad
    /// number, unrecognised kind tag).
    pub fn load_text(path: impl AsRef<std::path::Path>) -> Result<Self, PoseGraphParseError> {
        let text = std::fs::read_to_string(path).map_err(PoseGraphParseError::Io)?;
        let mut graph = PoseGraph::new();
        for (lineno, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut tok = line.split_ascii_whitespace();
            let tag = tok.next().ok_or_else(|| PoseGraphParseError::Syntax {
                line: lineno + 1,
                reason: "empty line tag".to_string(),
            })?;
            match tag {
                "P" => {
                    let id = parse_field::<u64>(tok.next(), lineno, "id")?;
                    let qw = parse_field::<f64>(tok.next(), lineno, "qw")?;
                    let qx = parse_field::<f64>(tok.next(), lineno, "qx")?;
                    let qy = parse_field::<f64>(tok.next(), lineno, "qy")?;
                    let qz = parse_field::<f64>(tok.next(), lineno, "qz")?;
                    let tx = parse_field::<f64>(tok.next(), lineno, "tx")?;
                    let ty = parse_field::<f64>(tok.next(), lineno, "ty")?;
                    let tz = parse_field::<f64>(tok.next(), lineno, "tz")?;
                    let rot = UnitQuaternion::from_quaternion(Quaternion::new(qw, qx, qy, qz));
                    let pose = Pose::from_world_to_camera(rot, Vector3::new(tx, ty, tz));
                    graph.poses.insert(id, pose);
                }
                "A" => {
                    let id = parse_field::<u64>(tok.next(), lineno, "anchor id")?;
                    graph.anchor = Some(id);
                }
                "E" => {
                    let from = parse_field::<u64>(tok.next(), lineno, "from")?;
                    let to = parse_field::<u64>(tok.next(), lineno, "to")?;
                    let kind_tag = parse_field::<u8>(tok.next(), lineno, "kind")?;
                    let kind = match kind_tag {
                        0 => PoseGraphEdgeKind::Sequential,
                        1 => PoseGraphEdgeKind::LoopClosure,
                        other => {
                            return Err(PoseGraphParseError::Syntax {
                                line: lineno + 1,
                                reason: format!("unrecognised edge kind tag {other}"),
                            });
                        }
                    };
                    let weight = parse_field::<f64>(tok.next(), lineno, "weight")?;
                    let qw = parse_field::<f64>(tok.next(), lineno, "qw")?;
                    let qx = parse_field::<f64>(tok.next(), lineno, "qx")?;
                    let qy = parse_field::<f64>(tok.next(), lineno, "qy")?;
                    let qz = parse_field::<f64>(tok.next(), lineno, "qz")?;
                    let tx = parse_field::<f64>(tok.next(), lineno, "tx")?;
                    let ty = parse_field::<f64>(tok.next(), lineno, "ty")?;
                    let tz = parse_field::<f64>(tok.next(), lineno, "tz")?;
                    let rot = UnitQuaternion::from_quaternion(Quaternion::new(qw, qx, qy, qz));
                    graph.edges.push(PoseGraphEdge {
                        from,
                        to,
                        measurement: SE3::new(rot, Vector3::new(tx, ty, tz)),
                        kind,
                        weight,
                        information: None,
                    });
                }
                other => {
                    return Err(PoseGraphParseError::Syntax {
                        line: lineno + 1,
                        reason: format!("unknown line tag '{other}'"),
                    });
                }
            }
        }
        Ok(graph)
    }
}

#[derive(Debug)]
pub enum PoseGraphParseError {
    Io(std::io::Error),
    Syntax { line: usize, reason: String },
}

impl std::fmt::Display for PoseGraphParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoseGraphParseError::Io(e) => write!(f, "I/O error reading pose graph: {e}"),
            PoseGraphParseError::Syntax { line, reason } => {
                write!(f, "pose graph parse error at line {line}: {reason}")
            }
        }
    }
}

impl std::error::Error for PoseGraphParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PoseGraphParseError::Io(e) => Some(e),
            PoseGraphParseError::Syntax { .. } => None,
        }
    }
}

fn parse_field<T: std::str::FromStr>(
    field: Option<&str>,
    lineno: usize,
    name: &str,
) -> Result<T, PoseGraphParseError>
where
    T::Err: std::fmt::Display,
{
    let s = field.ok_or_else(|| PoseGraphParseError::Syntax {
        line: lineno + 1,
        reason: format!("missing field '{name}'"),
    })?;
    s.parse::<T>().map_err(|e| PoseGraphParseError::Syntax {
        line: lineno + 1,
        reason: format!("bad {name}: {e}"),
    })
}

/// Solve `H · x = b` preferring Cholesky (SPD path) and falling back to LU
/// for ill-conditioned or rank-deficient systems.
pub(crate) fn solve_normal_equations(
    h: &DMatrix<f64>,
    b: &DVector<f64>,
) -> Result<DVector<f64>, PoseGraphError> {
    if let Some(chol) = h.clone().cholesky() {
        return Ok(chol.solve(b));
    }
    h.clone()
        .lu()
        .solve(b)
        .ok_or(PoseGraphError::SingularSystem)
}

/// Sparse Cholesky solve of `(H + λI) · x = b` where `H` is supplied as
/// COO triplets and assumed SPD by construction (it is a sum of `wᵀw` block
/// outer products from edge Jacobians). Triplets may contain duplicates;
/// they are summed during the COO → CSC conversion.
///
/// The system is solved in the fill-reducing variable order carried by `order`
/// (see the `reordering` module), applied as a symmetric permutation. That keeps
/// the Cholesky factor near-banded and prevents the catastrophic fill-in that
/// makes poorly-ordered or intrinsically wide 3D pose graphs (e.g.
/// `torus`/`sphere`) intractable. The permutation is purely structural and
/// deterministic, so the returned solution is unchanged up to floating-point
/// summation order. The ordering depends only on the sparsity pattern, so
/// callers compute it once and reuse it across iterations.
///
/// The factorization itself is the block Cholesky (see [`block_cholesky`]):
/// `block_size` is the variable-block dimension (6 for SE(3) poses), and the
/// permuted system keeps those blocks contiguous, so the factor runs on dense
/// `B×B` kernels instead of scalar columns.
///
/// Returns [`PoseGraphError::SingularSystem`] when the factorization fails
/// (e.g., disconnected graph). The damping term `λ` is added to the diagonal
/// before factoring, matching the dense LM path's `H + λI` formulation.
fn solve_normal_equations_sparse(
    triplets: &[(usize, usize, f64)],
    dim: usize,
    block_size: usize,
    b: &DVector<f64>,
    lambda: f64,
    order: &reordering::Reordering,
    symbolic_cache: &mut Option<block_cholesky::BlockSymbolic>,
) -> Result<DVector<f64>, PoseGraphError> {
    let permuted = order.permute_triplets(triplets);
    let rhs_permuted = order.permute_rhs(b);
    let rhs = DMatrix::from_column_slice(dim, 1, rhs_permuted.as_slice());
    // The permuted sparsity pattern is identical across LM iterations, so the
    // block-Cholesky symbolic analysis (and the COO→block assembly) is cached
    // and only the numeric refactorization runs after the first solve.
    let solution = block_cholesky::solve_spd_block_cached(
        symbolic_cache,
        &permuted,
        dim,
        block_size,
        &rhs,
        lambda,
    )
    .map_err(|_| PoseGraphError::SingularSystem)?;
    let solution_permuted = DVector::from_column_slice(solution.as_slice());
    Ok(order.restore_solution(&solution_permuted))
}

/// Multi-right-hand-side variant of [`solve_normal_equations_sparse`]: factor
/// the SPD matrix once and solve every column of `rhs` against it. The chordal
/// rotation initializer assembles one normal matrix shared by all three
/// rotation columns (`block_size` 3), so a single block factorization amortizes
/// over the three solves. The fill-reducing `order` is applied as a symmetric
/// permutation exactly as in the single-RHS path; each column is permuted in,
/// solved, and restored.
fn solve_normal_equations_sparse_multi(
    triplets: &[(usize, usize, f64)],
    dim: usize,
    block_size: usize,
    rhs: &DMatrix<f64>,
    order: &reordering::Reordering,
) -> Result<DMatrix<f64>, PoseGraphError> {
    let permuted = order.permute_triplets(triplets);

    // Permute every right-hand side into the reordered space, factor once via
    // the block Cholesky, and restore each solved column.
    let cols = rhs.ncols();
    let mut rhs_permuted = DMatrix::<f64>::zeros(dim, cols);
    for c in 0..cols {
        let column = DVector::from_column_slice(rhs.column(c).as_slice());
        rhs_permuted.set_column(c, &order.permute_rhs(&column));
    }
    let solved = block_cholesky::solve_spd_block(&permuted, dim, block_size, &rhs_permuted, 0.0)
        .map_err(|_| PoseGraphError::SingularSystem)?;

    let mut out = DMatrix::<f64>::zeros(dim, cols);
    for c in 0..cols {
        let solved_permuted = DVector::from_column_slice(solved.column(c).as_slice());
        out.set_column(c, &order.restore_solution(&solved_permuted));
    }
    Ok(out)
}

fn add_block6(h: &mut DMatrix<f64>, row: usize, col: usize, weight: f64, block: &Matrix6<f64>) {
    for r in 0..6 {
        for c in 0..6 {
            h[(row + r, col + c)] += weight * block[(r, c)];
        }
    }
}

fn add_segment6(g: &mut DVector<f64>, start: usize, weight: f64, v: &Vector6<f64>) {
    for k in 0..6 {
        g[start + k] += weight * v[k];
    }
}

/// Add `value · I_3` to the `(start, start)` diagonal block of either a
/// dense `H` matrix or a triplet vector. Used by the translation-only
/// optimizer where the per-edge contribution to `A^T A` is a scaled
/// identity on the diagonal blocks.
fn add_diag_block3(
    h_dense: &mut Option<DMatrix<f64>>,
    triplets: &mut Vec<(usize, usize, f64)>,
    start: usize,
    value: f64,
) {
    if let Some(h) = h_dense {
        for k in 0..3 {
            h[(start + k, start + k)] += value;
        }
    } else {
        for k in 0..3 {
            triplets.push((start + k, start + k, value));
        }
    }
}

/// Add `value · I_3` to the `(row_start, col_start)` off-diagonal block
/// (off-diagonal in the block-of-3 sense — also used when row != col).
fn add_offdiag_block3(
    h_dense: &mut Option<DMatrix<f64>>,
    triplets: &mut Vec<(usize, usize, f64)>,
    row_start: usize,
    col_start: usize,
    value: f64,
) {
    if let Some(h) = h_dense {
        for k in 0..3 {
            h[(row_start + k, col_start + k)] += value;
        }
    } else {
        for k in 0..3 {
            triplets.push((row_start + k, col_start + k, value));
        }
    }
}

/// Add a full (possibly dense) 3×3 `block` at the `(row_start, col_start)`
/// position of either a dense `H` or a triplet vector. Used by the chordal
/// rotation initializer, whose off-diagonal coupling `-w·R_meas` is a dense
/// rotation matrix rather than a scaled identity. Zero entries are skipped in
/// the sparse path so the rotation matrices contribute only their nonzeros.
fn add_dense_block3(
    h_dense: &mut Option<DMatrix<f64>>,
    triplets: &mut Vec<(usize, usize, f64)>,
    row_start: usize,
    col_start: usize,
    block: &Matrix3<f64>,
) {
    if let Some(h) = h_dense {
        for r in 0..3 {
            for c in 0..3 {
                h[(row_start + r, col_start + c)] += block[(r, c)];
            }
        }
    } else {
        for r in 0..3 {
            for c in 0..3 {
                let v = block[(r, c)];
                if v != 0.0 {
                    triplets.push((row_start + r, col_start + c, v));
                }
            }
        }
    }
}

/// Scalar edge weight used by the chordal rotation initializer. For an
/// isotropic edge this is just `edge.weight`; for an edge carrying a full 6×6
/// information matrix `Ω` (ordered `[ρ; ω]`) it is the mean of the rotational
/// diagonal `(Ω₃₃ + Ω₄₄ + Ω₅₅)/3`, i.e. the confidence g2o assigned to the
/// rotation block. Negative or non-finite results are clamped to a tiny
/// positive weight so the relaxed normal matrix stays positive definite.
fn chordal_rotation_weight(edge: &PoseGraphEdge) -> f64 {
    let w = match &edge.information {
        Some(omega) => (omega[(3, 3)] + omega[(4, 4)] + omega[(5, 5)]) / 3.0,
        None => edge.weight,
    };
    if w.is_finite() && w > 0.0 {
        w
    } else {
        1e-9
    }
}

/// Project a 3×3 matrix onto the nearest rotation in `SO(3)` (Frobenius sense)
/// via its SVD: `R = U·diag(1, 1, det(UVᵀ))·Vᵀ`. The determinant correction
/// guarantees `det(R) = +1` (a proper rotation, never a reflection). Returns
/// `None` only when the SVD fails to converge.
fn project_to_so3(m: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    let svd = m.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let mut r = u * v_t;
    if r.determinant() < 0.0 {
        // Flip the sign of the smallest singular direction (the last column of
        // U) to turn the reflection into a proper rotation.
        let mut u_fixed = u;
        for k in 0..3 {
            u_fixed[(k, 2)] = -u_fixed[(k, 2)];
        }
        r = u_fixed * v_t;
    }
    Some(r)
}

/// Storage for the SE(3) Gauss-Newton normal-equations matrix `H` that
/// dispatches to either a dense [`DMatrix`] or a COO triplet vector backing
/// a sparse Cholesky solve. The right-hand side `g` is assembled separately
/// (see callers) so this builder stays focused on `H`.
enum NormalEquations6 {
    Dense(DMatrix<f64>),
    Sparse {
        triplets: Vec<(usize, usize, f64)>,
        dim: usize,
    },
}

impl NormalEquations6 {
    fn new(dim: usize, solver: LinearSolver, edge_hint: usize) -> Self {
        match solver {
            LinearSolver::Dense => Self::Dense(DMatrix::zeros(dim, dim)),
            // Each edge contributes up to four 6×6 blocks = 4·36 = 144 entries.
            LinearSolver::Sparse => Self::Sparse {
                triplets: Vec::with_capacity(edge_hint * 144),
                dim,
            },
        }
    }

    fn add_block6(&mut self, row: usize, col: usize, weight: f64, block: &Matrix6<f64>) {
        match self {
            Self::Dense(h) => add_block6(h, row, col, weight, block),
            Self::Sparse { triplets, .. } => {
                for r in 0..6 {
                    for c in 0..6 {
                        triplets.push((row + r, col + c, weight * block[(r, c)]));
                    }
                }
            }
        }
    }

    /// Solve the assembled system. For the sparse backend the fill-reducing
    /// ordering is computed once into `order_cache` (the sparsity pattern is
    /// identical across LM iterations) and reused on subsequent calls.
    fn solve(
        self,
        lambda: f64,
        neg_g: &DVector<f64>,
        order_cache: &mut Option<reordering::Reordering>,
        symbolic_cache: &mut Option<block_cholesky::BlockSymbolic>,
    ) -> Result<DVector<f64>, PoseGraphError> {
        match self {
            Self::Dense(mut h) => {
                if lambda > 0.0 {
                    let dim = h.nrows();
                    for k in 0..dim {
                        h[(k, k)] += lambda;
                    }
                }
                solve_normal_equations(&h, neg_g)
            }
            Self::Sparse { triplets, dim } => {
                let order = order_cache.get_or_insert_with(|| {
                    reordering::Reordering::fill_reducing(dim, 6, &triplets)
                });
                solve_normal_equations_sparse(
                    &triplets,
                    dim,
                    6,
                    neg_g,
                    lambda,
                    order,
                    symbolic_cache,
                )
            }
        }
    }
}

/// Compute the relative SE3 `previous_to_current` such that
/// `to_pose.world_to_camera == relative * from_pose.world_to_camera`. This is
/// the same convention used by [`PoseGraphEdge::measurement`].
pub fn relative_world_to_camera(from_pose: &Pose, to_pose: &Pose) -> SE3 {
    to_pose
        .world_to_camera
        .compose(&from_pose.world_to_camera.inverse())
}

/// The PCM [`pcm::LoopMeasurement`] view of a verified loop-closure constraint.
fn loop_measurement_of(c: &LoopClosureConstraint) -> pcm::LoopMeasurement {
    pcm::LoopMeasurement {
        from: c.from_keyframe_id,
        to: c.to_keyframe_id,
        relative: c.relative_pose.clone(),
    }
}

/// PCM admission test for a single newly-verified loop closure (the online,
/// incremental variant of [`pcm::maximum_consistent_set`]). Admits `new` when it
/// is individually consistent with the odometry (if
/// [`pcm::PcmConfig::require_individual`]) and pairwise-consistent with a strict
/// majority of the already-`admitted` closures — so a perceptual-aliasing false
/// positive, inconsistent with the established consensus, is rejected before it
/// enters the graph. The first closure (empty `admitted`) is admitted on the
/// individual check alone.
fn pcm_admits_loop(
    new: &pcm::LoopMeasurement,
    admitted: &[pcm::LoopMeasurement],
    odometry: &BTreeMap<u64, SE3>,
    cfg: &pcm::PcmConfig,
) -> bool {
    if cfg.require_individual {
        match pcm::individual_residual(new, odometry) {
            Some(r) if r <= cfg.threshold => {}
            _ => return false,
        }
    }
    if admitted.is_empty() {
        return true;
    }
    let consistent = admitted
        .iter()
        .filter(|a| {
            pcm::pairwise_residual(new, a, odometry)
                .map(|r| r <= cfg.threshold)
                .unwrap_or(false)
        })
        .count();
    // Strict majority of the established set agrees with the new closure.
    consistent * 2 > admitted.len()
}

/// Translation-only constraint on camera centers in world coordinates implied
/// by `measurement` together with `to_pose`'s rotation: `c_to - c_from`
/// equals this displacement.
fn expected_world_displacement(to_pose: &Pose, measurement: &SE3) -> nalgebra::Vector3<f64> {
    let rotation_matrix = to_pose
        .world_to_camera
        .rotation
        .to_rotation_matrix()
        .into_inner();
    -(rotation_matrix.transpose() * measurement.translation)
}
