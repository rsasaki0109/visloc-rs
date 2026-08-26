//! Motion-based VI initialiser — first cut (VIBA1, stereo / known-scale
//! path).
//!
//! Companion to the stationary-window [`crate::VisualInertialInitializer`].
//! Where the static seed recovers `(R_w←b, b_g, b_a)` from a stationary
//! IMU window (yaw and monocular scale unobservable, accel-bias
//! observability partial), this stage refines the per-keyframe
//! `(R_w←b, v_w, b_g, b_a)` once the body has moved enough to give the
//! IMU translational excitation. It is the analogue of ORB-SLAM3's
//! `VIBA1` step.
//!
//! Triggering. The state machine accumulates camera centres reported by
//! the pipeline via [`MotionBasedViInitializer::register_keyframe`] and
//! fires when **both** of these are satisfied (defaults from
//! [`MotionBasedViInitializerConfig::default`], drawn from the design
//! note `docs/motion_based_vi_alignment.md`):
//!
//! * `keyframes_observed >= min_keyframes` (default `10`)
//! * `cumulative_translation_meters >= min_translation_meters` (default `2.0`)
//!
//! Solve. [`MotionBasedViInitializer::try_initialize`] invokes
//! [`crate::run_inertial_only_vi_ba`] over the accumulated keyframe set,
//! which holds the vision-only poses fixed and optimises velocities plus a
//! shared short-window bias against IMU preintegration factors (landmarks are
//! not touched). Scale is fixed at `1.0`; stereo / RGB-D sequences and
//! monocular sequences with a known metric anchor are well-served.
//!
//! The optional VIBA2 outer loop handles experimental monocular scale
//! recovery. Full joint visual-inertial BA and consistent marginalization
//! remain separate follow-ups.

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::{DMatrix, DVector, Matrix3, Point3, UnitQuaternion, Vector3};
use visloc_core::geometry::SE3;
use visloc_core::types::VisualMap;

use crate::bundle::{BaConfig, BaCostBreakdown, BaResult};
use crate::imu_preintegration::{ImuPreintegratedDelta, ImuPreintegrationFactor};
use crate::online_slam_vi_ba::{
    run_inertial_only_vi_ba, run_inertial_only_vi_ba_with_options, run_viba2_inertial_with_scale,
    KeyframeImuState, Viba2Config,
};
use crate::vi_initializer::VisualInertialInitializationResult;
use crate::LinearSolver;

/// A second, higher excitation gate gating when [`MotionBasedViInitializer`]
/// is allowed to release the gyro/accel bias slots to the LM solver.
///
/// Motivation: firing the VIBA1 inertial-only solve with bias free from the
/// very first eligible window — a short (as few as 3-keyframe) span just
/// past the base `min_keyframes` / `min_translation_meters` gates — let the
/// recovered accelerometer bias diverge to magnitudes around 12 m/s² on a
/// moving-start EuRoC sequence. The base gates only guarantee *some*
/// translational excitation exists; they say nothing about whether that
/// excitation is enough to make bias jointly observable alongside velocity
/// over such a short window.
///
/// When configured, [`MotionBasedViInitializer::try_initialize_with_bias_seed`]
/// runs a two-stage schedule instead of a single solve:
///
/// * **Stage A ("velocity stage")**: while `keyframes_observed <
///   min_keyframes` or `cumulative_translation_meters < min_translation_meters`
///   (this schedule's own, typically stricter, thresholds — independent of
///   the initializer's base gates), the solve holds every bias slot fixed at
///   the seed and refines only per-keyframe velocities via
///   [`crate::run_inertial_only_vi_ba_with_options`] with `solve_bias =
///   false`. Stage A fires **once**; its result is cached
///   ([`MotionBasedViInitializer::velocity_stage_result`]) but does NOT park
///   the initializer in the terminal `Initialised` state, so the pipeline
///   keeps registering keyframes / translation toward the release gate.
/// * **Stage B ("bias release")**: once both thresholds are met, the normal
///   solve runs with biases free (dispatching to VIBA2 when configured),
///   exactly like the legacy single-stage behaviour. Success here is
///   terminal, as usual.
///
/// `None` (the default) preserves the legacy behaviour: bias is free from
/// the very first solve that clears the initializer's base gates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiasReleaseSchedule {
    /// Minimum keyframe count (mirrors [`MotionBasedViInitializerConfig::min_keyframes`]'s
    /// role, but gates bias release rather than the first solve attempt).
    pub min_keyframes: usize,
    /// Minimum cumulative camera-centre translation in metres before bias
    /// is released to the solver.
    pub min_translation_meters: f64,
}

/// Configuration for [`MotionBasedViInitializer`].
#[derive(Debug, Clone, PartialEq)]
pub struct MotionBasedViInitializerConfig {
    /// Minimum number of keyframes (registered after the static seed)
    /// before the VIBA1 trigger is allowed to fire. Default `10`.
    pub min_keyframes: usize,
    /// Minimum cumulative camera-centre translation in metres before
    /// VIBA1 fires. Default `2.0`. Set to `0.0` to disable the
    /// translation gate.
    pub min_translation_meters: f64,
    /// World-frame gravity vector. Echoed onto the BA result for
    /// downstream diagnostics; the IMU factors fed into the solve carry
    /// their own gravity already.
    pub gravity_world: Vector3<f64>,
    /// Rigid transform from the tracked camera/sensor frame into the IMU body
    /// frame (EuRoC `T_BS` for cam0). The visual map stores world-to-camera
    /// poses while preintegration residuals operate on world-to-body poses.
    /// Identity preserves the co-located body/camera convention.
    pub body_to_camera: SE3,
    /// Inner LM solver config. Default mirrors
    /// [`crate::OnlineSlamLocalBaConfig::default`] (sparse linear
    /// solver, 10 LM iterations).
    pub ba_config: BaConfig,
    /// Optional VIBA2 hand-off. When `Some`, the initialiser runs an
    /// outer scale-recovery loop on top of the VIBA1 inertial-only
    /// solve. Stereo / RGB-D / known-scale paths should leave this
    /// `None` or set `recover_scale = false`; monocular paths set
    /// `recover_scale = true`. See [`crate::Viba2Config`] for the
    /// outer-loop knobs. Default `None`.
    pub viba2: Option<Viba2Config>,
    /// Post-solve sanity gate on the recovered per-keyframe
    /// `velocity_world` magnitudes. When `Some(v)`, the initialiser
    /// rejects the inner LM result if any keyframe's
    /// `||velocity_world|| > v` and parks itself in the `Waiting` state
    /// with [`MotionBasedViRejectionReason::VelocityOutOfRange`].
    /// `None` (default) preserves the legacy behaviour of accepting any
    /// LM-converged result. Indoor V1-class EuRoC sequences run safely
    /// at `Some(10.0)` (~36 km/h ceiling); set higher for outdoor
    /// driving datasets.
    pub max_velocity_magnitude_mps: Option<f64>,
    /// Optional post-solve upper bound on every recovered gyro-bias vector
    /// magnitude (rad/s). `None` preserves legacy behavior.
    pub max_gyro_bias_magnitude_rad_s: Option<f64>,
    /// Optional post-solve upper bound on every recovered accelerometer-bias
    /// vector magnitude (m/s²). `None` preserves legacy behavior.
    pub max_accel_bias_magnitude_mps2: Option<f64>,
    /// Optional statistical consistency gate on the final whitened IMU
    /// residual. The value is NIS divided by 9 residual DoF per factor;
    /// a well-modelled solve is expected near one. `None` preserves the
    /// historical behavior.
    pub max_final_imu_nis_per_dof: Option<f64>,
    /// Coarse physical-unit gate for fixed-pose initialization. Unlike NIS,
    /// these bounds do not assume the visual keyframe poses are already
    /// statistically consistent with the IMU covariance.
    pub max_final_imu_rotation_residual_rms_rad: Option<f64>,
    pub max_final_imu_velocity_residual_rms_mps: Option<f64>,
    pub max_final_imu_position_residual_rms_meters: Option<f64>,
    /// Optional staged bias-release schedule. `None` (default) preserves
    /// the legacy behaviour: bias is free from the first solve that clears
    /// `min_keyframes` / `min_translation_meters`. `Some(schedule)` inserts
    /// a bias-fixed "velocity stage" before bias is released — see
    /// [`BiasReleaseSchedule`] for the full contract and the divergence
    /// this guards against.
    pub bias_release: Option<BiasReleaseSchedule>,
    /// Estimate the world-frame gravity DIRECTION from this window's IMU
    /// preintegration factors plus the fixed visual poses, instead of
    /// trusting `gravity_world` as ground truth. See
    /// [`estimate_gravity_and_velocities`] and
    /// `docs/motion_based_vi_alignment.md`'s "Gravity-direction recovery"
    /// section for the motivating diagnosis: on moving-start sequences
    /// where the static [`crate::VisualInertialInitializer`] never fires,
    /// each identity-seeded (sub)map's world frame has gravity pointing in
    /// an ARBITRARY direction — the legacy behaviour (trusting
    /// `gravity_world` from config as truth) then makes every IMU residual
    /// dominated by that misalignment rather than by real inconsistency,
    /// and every promotion is rejected by the raw-residual gates. When
    /// `true`, [`MotionBasedViInitializer::try_initialize_with_bias_seed`]
    /// runs the linear alignment first, feeds the solver factors carrying
    /// the ESTIMATED gravity (rather than the configured one), and seeds
    /// velocities from the same alignment instead of finite differences.
    /// Default `false` preserves the legacy behaviour.
    pub estimate_gravity: bool,
    /// Observability gate for `estimate_gravity`: the UNCONSTRAINED linear
    /// solve's raw gravity norm (see
    /// [`GravityVelocityAlignment::raw_gravity_norm`]) must fall within
    /// this fraction of `gravity_world.norm()` before the estimate is
    /// trusted — a well-conditioned window recovers a norm near the true
    /// magnitude on its own, with no norm constraint imposed, so a large
    /// deviation signals the window's IMU excitation does not yet make
    /// gravity direction reliably observable. `0.3` (30%) is a
    /// conservative default; tighten once per-dataset noise floors are
    /// characterised. Ignored when `estimate_gravity` is `false`.
    pub max_gravity_norm_deviation_ratio: f64,
    /// Estimate the shared gyro bias from ROTATION-ONLY alignment against
    /// this window's fixed visual poses, BEFORE gravity/velocity alignment
    /// and before the staged solve. See [`estimate_gyro_bias`] and
    /// `docs/motion_based_vi_alignment.md`'s "Gyro-bias recovery" section
    /// for the motivating diagnosis: on a moving-start EuRoC fixture the
    /// final fitted IMU rotation residual RMS sat at 0.014-0.022 rad
    /// against a 0.01 gate, bit-identical with/without gravity estimation
    /// (the rotation residual is gravity-independent) — consistent with
    /// EuRoC's typical gyro bias (~0.08 rad/s norm) accumulating over each
    /// factor's `delta_time`, with the bias seed fixed at the configured
    /// zero the whole time. When `true`,
    /// [`MotionBasedViInitializer::try_initialize_with_bias_seed`] runs this
    /// alignment first (right after the body-pose conversion, before
    /// gravity recovery) and, on success, substitutes the recovered bias
    /// for `bias_gyro_seed` everywhere downstream of that point in the same
    /// attempt: the gravity/velocity alignment's own bias-corrected deltas,
    /// the per-keyframe `initial_states` seeding, and (when a
    /// [`BiasReleaseSchedule`] Stage A fires) the fixed-bias solve, which
    /// then fixes gyro bias at the ESTIMATE rather than the raw seed
    /// (accelerometer bias stays fixed at the seed either way — this stage
    /// has no accelerometer evidence). Default `false` preserves the legacy
    /// behaviour of trusting `bias_gyro_seed` verbatim.
    pub estimate_gyro_bias: bool,
}

impl Default for MotionBasedViInitializerConfig {
    fn default() -> Self {
        Self {
            min_keyframes: 10,
            min_translation_meters: 2.0,
            gravity_world: Vector3::new(0.0, 9.81, 0.0),
            body_to_camera: SE3::identity(),
            ba_config: BaConfig {
                linear_solver: LinearSolver::Sparse,
                max_iterations: 10,
                ..BaConfig::default()
            },
            viba2: None,
            max_velocity_magnitude_mps: None,
            max_gyro_bias_magnitude_rad_s: None,
            max_accel_bias_magnitude_mps2: None,
            max_final_imu_nis_per_dof: None,
            max_final_imu_rotation_residual_rms_rad: None,
            max_final_imu_velocity_residual_rms_mps: None,
            max_final_imu_position_residual_rms_meters: None,
            bias_release: None,
            estimate_gravity: false,
            max_gravity_norm_deviation_ratio: 0.3,
            estimate_gyro_bias: false,
        }
    }
}

/// VIBA1 / VIBA2 outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct MotionBasedViInitializationResult {
    /// Refined per-keyframe `(velocity_world, bias_gyro, bias_acc)`.
    pub keyframe_states: BTreeMap<u64, KeyframeImuState>,
    /// Sorted keyframe ids covered by the solve.
    pub keyframe_ids: Vec<u64>,
    /// Number of IMU factors fed into the solve.
    pub imu_factors_used: usize,
    /// Scale factor recovered by the solve. `1.0` on the VIBA1-only
    /// path (no `viba2` config or `recover_scale = false`); the closed-
    /// form least-squares scale on the VIBA2 monocular path.
    pub scale: f64,
    /// Outer-loop scale history when VIBA2 ran (one entry per outer
    /// iteration, starting with `Viba2Config::initial_scale`). Empty
    /// on the VIBA1-only path.
    pub scale_history: Vec<f64>,
    /// Number of VIBA2 outer iterations executed. `0` on the VIBA1-only
    /// path.
    pub viba2_iterations_run: usize,
    /// Cumulative translation (m) since the static seed at the moment
    /// the trigger fired. Surface for logging / diagnostics.
    pub trigger_translation_meters: f64,
    /// Inner LM solve outcome (from the final VIBA2 inner solve when
    /// applicable, else the single VIBA1 solve).
    pub ba_result: BaResult,
    pub initial_cost_breakdown: BaCostBreakdown,
    pub final_cost_breakdown: BaCostBreakdown,
    /// `true` when this result's biases were solved for (legacy path, or
    /// [`BiasReleaseSchedule`] Stage B); `false` when this is a
    /// [`BiasReleaseSchedule`] Stage A "velocity stage" result — every
    /// keyframe's `(bias_gyro, bias_acc)` above equals the seed passed into
    /// [`MotionBasedViInitializer::try_initialize_with_bias_seed`], and the
    /// caller should NOT treat this as the terminal motion-VI-init outcome.
    /// See [`BiasReleaseSchedule`] for the full contract.
    pub bias_released: bool,
    /// Gravity vector recovered by [`MotionBasedViInitializerConfig::estimate_gravity`],
    /// `None` when that feature is off (the legacy path). When `Some`, the
    /// pipeline mirrors it into every downstream gravity sink — see
    /// `crate::OnlineSlamPipeline::promote_motion_vi_init_result`'s doc
    /// comment for the full mirroring list. The map gauge itself is NEVER
    /// rotated to match; only the gravity ASSUMPTION used by future
    /// factor-staging and VI solves moves. See
    /// `docs/motion_based_vi_alignment.md`'s "Gravity-direction recovery"
    /// section.
    pub estimated_gravity_world: Option<Vector3<f64>>,
    /// Gyro bias recovered by [`MotionBasedViInitializerConfig::estimate_gyro_bias`]'s
    /// rotation-only alignment, `None` when that feature is off (the legacy
    /// path). When `Some`, every `keyframe_states` entry's `bias_gyro`
    /// already reflects this estimate (fixed at it on a
    /// [`BiasReleaseSchedule`] Stage A result, refined from it as the
    /// starting point on Stage B / legacy / VIBA2), so the existing
    /// promotion bias-mirroring picks it up with no extra plumbing. See
    /// `docs/motion_based_vi_alignment.md`'s "Gyro-bias recovery" section.
    pub estimated_gyro_bias: Option<Vector3<f64>>,
}

/// Why a [`MotionBasedViInitializer::try_initialize`] call returned
/// without a fresh result.
#[derive(Debug, Clone, PartialEq)]
pub enum MotionBasedViRejectionReason {
    /// Fewer than `min_keyframes` keyframes have been registered.
    InsufficientKeyframes { have: usize, need: usize },
    /// Accumulated camera-centre translation is below the trigger
    /// threshold.
    InsufficientTranslation { have: f64, need: f64 },
    /// The keyframe set has at least `min_keyframes` entries and meets
    /// the translation gate, but none of the supplied preintegration
    /// factors connect any pair of registered keyframes — the IMU
    /// stream has not yet emitted factors over this window, or the
    /// caller passed an unrelated factor slice.
    NoUsableImuFactors,
    /// One or more registered keyframes are missing their pose, or the
    /// camera lookup failed.
    MissingKeyframeData,
    /// Inner LM solver failed to converge (singular system, etc.).
    SolverFailed,
    /// Inner LM solver returned, but at least one per-keyframe
    /// `||velocity_world||` exceeded
    /// `MotionBasedViInitializerConfig::max_velocity_magnitude_mps`. The
    /// solver's bias / velocity slots are NOT promoted to `self.completed`,
    /// and its speculative map poses are discarded, so the next trigger
    /// re-runs from the same linearisation point. `kf_id` is the worst-offender keyframe;
    /// `magnitude_mps` is its recovered speed; `limit_mps` echoes the
    /// configured gate.
    VelocityOutOfRange {
        kf_id: u64,
        magnitude_mps: f64,
        limit_mps: f64,
    },
    /// A recovered gyro bias exceeded the configured physical sanity bound.
    GyroBiasOutOfRange {
        kf_id: u64,
        magnitude_rad_s: f64,
        limit_rad_s: f64,
    },
    /// A recovered accelerometer bias exceeded the configured physical sanity
    /// bound.
    AccelBiasOutOfRange {
        kf_id: u64,
        magnitude_mps2: f64,
        limit_mps2: f64,
    },
    /// The covariance-whitened IMU residual is statistically inconsistent
    /// with the proposed state, so promotion would hand a bad
    /// linearisation point to local VI-BA.
    ImuNisOutOfRange {
        normalized_nis_per_dof: f64,
        rotation_residual_rms_rad: Option<f64>,
        velocity_residual_rms_mps: Option<f64>,
        position_residual_rms_meters: Option<f64>,
        limit: f64,
    },
    /// The fixed-pose inertial initializer is too far from the IMU in
    /// physical units to seed the subsequent joint visual-inertial solve.
    ImuRawResidualOutOfRange {
        rotation_residual_rms_rad: Option<f64>,
        velocity_residual_rms_mps: Option<f64>,
        position_residual_rms_meters: Option<f64>,
        max_rotation_residual_rms_rad: Option<f64>,
        max_velocity_residual_rms_mps: Option<f64>,
        max_position_residual_rms_meters: Option<f64>,
    },
    /// A [`BiasReleaseSchedule`] Stage A ("velocity stage") solve has
    /// already fired, but the schedule's bias-release gates are not yet
    /// met, so there is nothing new to solve — the caller should keep
    /// registering keyframes / translation and retry later. `have_*`
    /// report the current accumulated metrics; `need_*` echo the
    /// configured [`BiasReleaseSchedule`] thresholds.
    AwaitingBiasReleaseExcitation {
        have_keyframes: usize,
        need_keyframes: usize,
        have_translation_meters: f64,
        need_translation_meters: f64,
    },
    /// [`MotionBasedViInitializerConfig::estimate_gravity`] is `true`, but
    /// the gravity/velocity alignment degenerated: fewer than 2 usable
    /// in-window IMU factors survived the alignment's own windowing, or
    /// the unconstrained linear solve produced a non-finite or near-zero
    /// gravity vector. Distinct from [`Self::NoUsableImuFactors`], which
    /// covers the (looser) "the solver has nothing to solve at all" case;
    /// this variant means the alignment specifically could not condition
    /// well enough to propose a gravity estimate at all.
    GravityEstimateDegenerate,
    /// [`MotionBasedViInitializerConfig::estimate_gravity`] recovered a
    /// gravity vector whose UNCONSTRAINED magnitude (`raw_norm_mps2`) sits
    /// further from the expected magnitude (`expected_mps2`, i.e.
    /// `config.gravity_world.norm()`) than
    /// `max_deviation_ratio` allows. This is the observability gate: a
    /// well-conditioned window recovers the true magnitude on its own with
    /// no norm constraint imposed, so a large deviation means the window's
    /// IMU excitation does not yet make gravity direction reliably
    /// observable.
    GravityEstimateOutOfRange {
        raw_norm_mps2: f64,
        expected_mps2: f64,
        max_deviation_ratio: f64,
    },
    /// [`MotionBasedViInitializerConfig::estimate_gyro_bias`] is `true`, but
    /// the rotation-only alignment degenerated: fewer than 2 usable
    /// in-window IMU factors survived the alignment's own windowing (a
    /// missing keyframe pose counts as unusable too), or the Gauss-Newton
    /// normal equations were singular on the very first iteration.
    GyroBiasEstimateDegenerate,
}

/// Read-only snapshot of [`MotionBasedViInitializer`]'s internal state.
#[derive(Debug, Clone, PartialEq)]
pub enum MotionBasedViInitializationStatus {
    /// Trigger has not yet fired. Reports the accumulated metrics plus
    /// the last `Err(...)` returned by [`MotionBasedViInitializer::try_initialize`]
    /// (if any) so callers can render diagnostic progress.
    Waiting {
        keyframes_observed: usize,
        cumulative_translation_meters: f64,
        last_rejection: Option<MotionBasedViRejectionReason>,
        /// `true` once a [`BiasReleaseSchedule`] Stage A ("velocity stage")
        /// solve has fired and is cached on
        /// [`MotionBasedViInitializer::velocity_stage_result`]. Always
        /// `false` when `bias_release` is not configured (legacy path never
        /// parks in a non-terminal `Waiting` success).
        velocity_stage_completed: bool,
        /// Mirrors [`MotionBasedViInitializer::last_gravity_alignment`]: the
        /// most recent gravity/velocity alignment attempt, kept even when
        /// `last_rejection` reports a later gate rejected it. `None` when
        /// `estimate_gravity` is off or no attempt has run yet.
        last_gravity_alignment: Option<GravityVelocityAlignment>,
        /// Mirrors [`MotionBasedViInitializer::last_gyro_bias_alignment`]:
        /// the most recent gyro-bias rotation-only alignment attempt, kept
        /// even when `last_rejection` reports a later gate rejected it.
        /// `None` when `estimate_gyro_bias` is off or no attempt has run
        /// yet.
        last_gyro_bias_alignment: Option<GyroBiasAlignment>,
    },
    /// VIBA1 has fired and succeeded; the result is the recovered
    /// refinement.
    Initialised {
        result: MotionBasedViInitializationResult,
    },
}

/// State machine that fires a single VIBA1 inertial-only solve once the
/// keyframe count + cumulative-translation thresholds are met. See the
/// module docstring and the design note
/// `docs/motion_based_vi_alignment.md` for the full contract.
#[derive(Debug, Clone, PartialEq)]
pub struct MotionBasedViInitializer {
    config: MotionBasedViInitializerConfig,
    keyframes: Vec<(u64, Point3<f64>)>,
    cumulative_translation: f64,
    completed: Option<MotionBasedViInitializationResult>,
    last_rejection: Option<MotionBasedViRejectionReason>,
    /// Track whether this initializer's `register_keyframe` should
    /// charge the inter-keyframe distance against the
    /// `cumulative_translation` running sum. Always `true` after the
    /// first `register_keyframe` call; the first call only seeds the
    /// chain (zero translation by definition).
    has_seed_center: bool,
    /// Cached [`BiasReleaseSchedule`] Stage A ("velocity stage") result.
    /// `Some` once that non-terminal solve has fired; cleared by
    /// [`Self::reset`]. Never populated when `config.bias_release` is
    /// `None` (the legacy path goes straight to a terminal result).
    velocity_stage: Option<MotionBasedViInitializationResult>,
    /// The most recent [`estimate_gravity_and_velocities`] output, kept
    /// regardless of whether the attempt that produced it went on to be
    /// accepted or rejected by a LATER gate (`GravityEstimateOutOfRange`,
    /// `ImuRawResidualOutOfRange`, `ImuNisOutOfRange`, etc). Diagnostic
    /// visibility only: promotion never reads this field, only the return
    /// value threaded through `try_initialize_with_bias_seed`'s own control
    /// flow. `None` when `config.estimate_gravity` is `false`, or once
    /// `reset` is called. See [`Self::last_gravity_alignment`] and
    /// `docs/motion_based_vi_alignment.md`'s "Gravity-direction recovery"
    /// section — this is the fix for that section's observed gap: on real
    /// data most attempts are rejected by a downstream residual gate before
    /// ever reaching a terminal `Initialised` state, so without this field
    /// the recovered gravity vector was invisible on every run except a
    /// lucky first-attempt success.
    last_gravity_alignment: Option<GravityVelocityAlignment>,
    /// The most recent [`estimate_gyro_bias`] output, kept regardless of
    /// whether the attempt that produced it went on to be accepted or
    /// rejected by a later gate. Diagnostic visibility only, mirroring
    /// [`Self::last_gravity_alignment`]'s rationale exactly. `None` when
    /// `config.estimate_gyro_bias` is `false`, or once `reset` is called.
    last_gyro_bias_alignment: Option<GyroBiasAlignment>,
}

impl MotionBasedViInitializer {
    /// Construct with the given config; carries no keyframes yet.
    pub fn new(config: MotionBasedViInitializerConfig) -> Self {
        Self {
            config,
            keyframes: Vec::new(),
            cumulative_translation: 0.0,
            completed: None,
            last_rejection: None,
            has_seed_center: false,
            velocity_stage: None,
            last_gravity_alignment: None,
            last_gyro_bias_alignment: None,
        }
    }

    /// Borrow the active configuration.
    pub fn config(&self) -> &MotionBasedViInitializerConfig {
        &self.config
    }

    /// Overwrite the running config's `gravity_world`. Used by
    /// `crate::OnlineSlamPipeline::promote_motion_vi_init_result` to mirror
    /// a freshly-`estimate_gravity`-recovered gravity vector back into this
    /// initializer so a LATER window (a subsequent
    /// [`BiasReleaseSchedule`] Stage B, or a fresh sequence after
    /// `reset_sequence_state`) starts from the corrected assumption instead
    /// of the original (possibly wrong) config value. Does not touch any
    /// cached result or registered keyframe/translation state.
    pub fn set_gravity_world(&mut self, gravity_world: Vector3<f64>) {
        self.config.gravity_world = gravity_world;
    }

    /// Drop every accumulated keyframe and the cached VIBA1 result. The
    /// configuration is preserved. Mirrors
    /// [`crate::VisualInertialInitializer::reset`].
    pub fn reset(&mut self) {
        self.keyframes.clear();
        self.cumulative_translation = 0.0;
        self.completed = None;
        self.last_rejection = None;
        self.has_seed_center = false;
        self.velocity_stage = None;
        self.last_gravity_alignment = None;
        self.last_gyro_bias_alignment = None;
    }

    /// Number of keyframes registered since construction / last reset.
    pub fn keyframes_observed(&self) -> usize {
        self.keyframes.len()
    }

    /// Cumulative camera-centre translation (m) along the registered
    /// keyframe chain.
    pub fn cumulative_translation_meters(&self) -> f64 {
        self.cumulative_translation
    }

    /// Note a freshly-registered keyframe's camera centre in world
    /// frame. The first call after construction / reset only seeds the
    /// chain; subsequent calls add the inter-keyframe distance to the
    /// running cumulative-translation total.
    ///
    /// Idempotent against re-registering the same `keyframe_id`: a
    /// second call with the same id silently overwrites the prior
    /// centre without double-counting. (This matters when a pipeline
    /// fires a "new keyframe" event and immediately corrects the pose
    /// via local-BA on the same frame.)
    pub fn register_keyframe(&mut self, keyframe_id: u64, camera_center_world: Point3<f64>) {
        if let Some(slot) = self.keyframes.iter_mut().find(|(id, _)| *id == keyframe_id) {
            slot.1 = camera_center_world;
            return;
        }
        if self.has_seed_center {
            if let Some(prev) = self.keyframes.last() {
                self.cumulative_translation += (camera_center_world - prev.1).norm();
            }
        } else {
            self.has_seed_center = true;
        }
        self.keyframes.push((keyframe_id, camera_center_world));
    }

    /// `true` once both trigger gates (keyframe count + translation)
    /// are satisfied. `try_initialize` will still return Err if no IMU
    /// factor connects any pair, but `is_ready` is the cheap pre-check
    /// the pipeline can poll on every frame.
    pub fn is_ready(&self) -> bool {
        self.completed.is_none()
            && self.keyframes.len() >= self.config.min_keyframes
            && self.cumulative_translation >= self.config.min_translation_meters
    }

    /// Returns the cached result if VIBA1 has already fired.
    pub fn result(&self) -> Option<&MotionBasedViInitializationResult> {
        self.completed.as_ref()
    }

    /// Returns the cached [`BiasReleaseSchedule`] Stage A ("velocity
    /// stage") result, if one has fired. Always `None` when
    /// `config().bias_release` is `None`, or once the initializer has
    /// reached the terminal `Initialised` state (Stage A's cache is not
    /// cleared on Stage B success, but callers should prefer
    /// [`Self::result`] at that point).
    pub fn velocity_stage_result(&self) -> Option<&MotionBasedViInitializationResult> {
        self.velocity_stage.as_ref()
    }

    /// Test-only seam: directly inject a cached [`BiasReleaseSchedule`]
    /// Stage A ("velocity stage") result without replaying the full
    /// numeric solve. Exists so pipeline-level tests (see
    /// `online_slam.rs`'s `vi_initialization_pending` tests) can exercise
    /// [`Self::velocity_stage_result`] / [`crate::OnlineSlamMotionViInitState::velocity_stage_fired`]
    /// without reconstructing an entire synthetic map + IMU-factor solve
    /// just to reach Stage A — that solve path is already covered by this
    /// module's own `bias_release_schedule_stage_a_fires_*` tests.
    #[cfg(test)]
    pub(crate) fn set_velocity_stage_result_for_test(
        &mut self,
        result: MotionBasedViInitializationResult,
    ) {
        self.velocity_stage = Some(result);
    }

    /// Returns the most recent [`estimate_gravity_and_velocities`] outcome,
    /// if `config().estimate_gravity` has ever produced one — regardless of
    /// whether that attempt went on to be accepted or rejected by a later
    /// gate. `None` when `estimate_gravity` is off, when no in-window
    /// attempt has run yet, or after [`Self::reset`]. See the field-level
    /// doc comment on `last_gravity_alignment` for why this exists
    /// separately from the terminal [`Self::result`].
    pub fn last_gravity_alignment(&self) -> Option<&GravityVelocityAlignment> {
        self.last_gravity_alignment.as_ref()
    }

    /// Returns the most recent [`estimate_gyro_bias`] outcome, if
    /// `config().estimate_gyro_bias` has ever produced one — regardless of
    /// whether that attempt went on to be accepted or rejected by a later
    /// gate. `None` when `estimate_gyro_bias` is off, when no in-window
    /// attempt has run yet, or after [`Self::reset`]. See the field-level
    /// doc comment on `last_gyro_bias_alignment` for why this exists
    /// separately from the terminal [`Self::result`].
    pub fn last_gyro_bias_alignment(&self) -> Option<&GyroBiasAlignment> {
        self.last_gyro_bias_alignment.as_ref()
    }

    /// Render a read-only snapshot of the current state.
    pub fn status(&self) -> MotionBasedViInitializationStatus {
        if let Some(result) = &self.completed {
            MotionBasedViInitializationStatus::Initialised {
                result: result.clone(),
            }
        } else {
            MotionBasedViInitializationStatus::Waiting {
                keyframes_observed: self.keyframes.len(),
                cumulative_translation_meters: self.cumulative_translation,
                last_rejection: self.last_rejection.clone(),
                velocity_stage_completed: self.velocity_stage.is_some(),
                last_gravity_alignment: self.last_gravity_alignment.clone(),
                last_gyro_bias_alignment: self.last_gyro_bias_alignment,
            }
        }
    }

    /// Attempt to fire VIBA1 over the registered keyframes.
    ///
    /// On success, the inertial-only solve refines `map.keyframes[*]`'s
    /// poses in place and the cached
    /// [`MotionBasedViInitializationResult`] is returned. The
    /// initialiser is single-shot: a successful firing parks the
    /// initialiser in the `Initialised` state and subsequent calls
    /// short-circuit to the cached result. Call [`Self::reset`] to
    /// re-arm for a new sequence.
    ///
    /// `static_seed` is consumed only as a sanity seed for downstream
    /// `keyframe_states` (the inertial-only solver pulls per-keyframe
    /// initial values from `map.keyframes[*]`'s poses + the static
    /// seed's biases / velocity).
    pub fn try_initialize(
        &mut self,
        map: &mut VisualMap,
        preintegration_factors: &[ImuPreintegrationFactor],
        static_seed: &VisualInertialInitializationResult,
    ) -> Result<&MotionBasedViInitializationResult, MotionBasedViRejectionReason> {
        self.try_initialize_with_bias_seed(
            map,
            preintegration_factors,
            static_seed.bias_gyro,
            static_seed.bias_acc,
        )
    }

    /// Motion-based initialization from explicit bias linearisation values.
    /// This is the safe fallback entry point for sequences that begin in
    /// motion: the caller may retain its calibrated/configured biases after a
    /// stationary initializer gives up, without fabricating a false static
    /// gravity/bias result. The solver still requires its normal keyframe and
    /// translation excitation gates.
    ///
    /// When `config().bias_release` is `Some(schedule)`, this call may
    /// instead run [`BiasReleaseSchedule`] Stage A (the bias-fixed
    /// "velocity stage"): see the type-level docs on
    /// [`BiasReleaseSchedule`] for the full two-stage contract. Check
    /// `result.bias_released` — `false` means this is a non-terminal Stage
    /// A result; `true` means the legacy path or Stage B fired and the
    /// initializer is now `Initialised`.
    pub fn try_initialize_with_bias_seed(
        &mut self,
        map: &mut VisualMap,
        preintegration_factors: &[ImuPreintegrationFactor],
        bias_gyro_seed: Vector3<f64>,
        bias_acc_seed: Vector3<f64>,
    ) -> Result<&MotionBasedViInitializationResult, MotionBasedViRejectionReason> {
        if let Some(_existing) = &self.completed {
            // SAFETY: borrow checker — re-fetch by `Option::as_ref` so
            // the `&mut self` borrow above can release before the
            // `&self` borrow below acquires.
            return Ok(self.completed.as_ref().expect("just checked"));
        }

        let observed = self.keyframes.len();
        if observed < self.config.min_keyframes {
            let err = MotionBasedViRejectionReason::InsufficientKeyframes {
                have: observed,
                need: self.config.min_keyframes,
            };
            self.last_rejection = Some(err.clone());
            return Err(err);
        }
        if self.cumulative_translation < self.config.min_translation_meters {
            let err = MotionBasedViRejectionReason::InsufficientTranslation {
                have: self.cumulative_translation,
                need: self.config.min_translation_meters,
            };
            self.last_rejection = Some(err.clone());
            return Err(err);
        }

        // Staged bias release (`BiasReleaseSchedule`): `bias_free` gates
        // whether THIS solve is allowed to touch bias. `None` (legacy)
        // always reports `bias_free = true` — bias is free the moment the
        // base gates above are cleared, as before.
        let bias_free = self.config.bias_release.is_none_or(|schedule| {
            self.keyframes.len() >= schedule.min_keyframes
                && self.cumulative_translation >= schedule.min_translation_meters
        });
        if !bias_free && self.velocity_stage.is_some() {
            // Stage A has already fired once and is cached; there is
            // nothing new to solve until the release gate itself is met.
            let schedule = self
                .config
                .bias_release
                .expect("bias_release must be Some when velocity_stage is populated");
            let err = MotionBasedViRejectionReason::AwaitingBiasReleaseExcitation {
                have_keyframes: self.keyframes.len(),
                need_keyframes: schedule.min_keyframes,
                have_translation_meters: self.cumulative_translation,
                need_translation_meters: schedule.min_translation_meters,
            };
            self.last_rejection = Some(err.clone());
            return Err(err);
        }

        // Seed per-keyframe states for the solver. The first keyframe
        // (the gauge anchor for the inertial-only solve) inherits the
        // static seed's biases + zero velocity; subsequent keyframes
        // also start from those biases but seed velocity from the
        // inter-keyframe centre displacement and the connecting IMU
        // factor's `delta_time` when available.
        let mut initial_states: BTreeMap<u64, KeyframeImuState> = BTreeMap::new();
        let kf_ids: Vec<u64> = self.keyframes.iter().map(|(id, _)| *id).collect();
        for (idx, &kf_id) in kf_ids.iter().enumerate() {
            let velocity = if idx == 0 {
                Vector3::zeros()
            } else {
                let prev_id = kf_ids[idx - 1];
                let factor = preintegration_factors
                    .iter()
                    .find(|f| f.keyframe_id_from == prev_id && f.keyframe_id_to == kf_id);
                match factor {
                    Some(f) if f.delta.delta_time > 0.0 => {
                        let prev_center = self.keyframes[idx - 1].1;
                        let curr_center = self.keyframes[idx].1;
                        (curr_center - prev_center) / f.delta.delta_time
                    }
                    _ => Vector3::zeros(),
                }
            };
            initial_states.insert(
                kf_id,
                KeyframeImuState {
                    velocity_world: velocity,
                    bias_gyro: bias_gyro_seed,
                    bias_acc: bias_acc_seed,
                },
            );
        }

        // Validate that all registered keyframes have a pose in `map`.
        for kf_id in &kf_ids {
            let Some(kf) = map.keyframes.get(kf_id) else {
                let err = MotionBasedViRejectionReason::MissingKeyframeData;
                self.last_rejection = Some(err.clone());
                return Err(err);
            };
            if kf.frame.pose.is_none() {
                let err = MotionBasedViRejectionReason::MissingKeyframeData;
                self.last_rejection = Some(err.clone());
                return Err(err);
            }
        }

        // Solve transactionally. The inertial BA helpers write refined poses
        // into their map argument before returning, but post-solve velocity
        // gates below may still reject the result. Keep those speculative
        // poses on a clone and publish them only after every gate passes.
        let mut candidate_map = map.clone();
        // The visual map stores T_cw, while the IMU factors are expressed in
        // body coordinates. EuRoC T_BS for cam0 is T_bc (camera/sensor to
        // body), hence T_bw = T_bc * T_cw. This conversion is speculative and
        // exists only on the solver clone; the tracked camera map is never
        // overwritten with body poses.
        for kf_id in &kf_ids {
            let pose = candidate_map
                .keyframes
                .get_mut(kf_id)
                .and_then(|kf| kf.frame.pose.as_mut())
                .expect("keyframe poses were validated above");
            pose.world_to_camera = self.config.body_to_camera.compose(&pose.world_to_camera);
        }

        // Gyro-bias recovery (`estimate_gyro_bias`): run the rotation-only
        // alignment over `candidate_map`'s just-converted body poses FIRST —
        // before gravity/velocity recovery and before the staged solve.
        // This is the classical first step ORB-SLAM3 / VINS-Mono run before
        // gravity alignment: the rotation residual is entirely
        // gravity-independent, so a wrong gyro bias leaves the SAME
        // residual whether or not gravity is estimated (see
        // `docs/motion_based_vi_alignment.md`'s "Gyro-bias recovery"
        // section for the motivating real-data diagnosis). On success, the
        // estimate replaces `bias_gyro_seed` for the REST of this attempt:
        // the gravity/velocity alignment call below, the `initial_states`
        // seeding above, and (via `initial_states`) any
        // `BiasReleaseSchedule` Stage A fixed-bias solve.
        let mut effective_bias_gyro = bias_gyro_seed;
        let mut estimated_gyro_bias: Option<Vector3<f64>> = None;
        if self.config.estimate_gyro_bias {
            let alignment = estimate_gyro_bias(
                &candidate_map,
                &kf_ids,
                preintegration_factors,
                bias_gyro_seed,
            );
            let alignment = match alignment {
                Some(alignment) => alignment,
                None => {
                    self.last_gyro_bias_alignment = None;
                    let err = MotionBasedViRejectionReason::GyroBiasEstimateDegenerate;
                    self.last_rejection = Some(err.clone());
                    return Err(err);
                }
            };
            // Record the attempt BEFORE the magnitude gate below, mirroring
            // `last_gravity_alignment`'s rationale: a rejected estimate is
            // exactly as diagnostically interesting as an accepted one.
            self.last_gyro_bias_alignment = Some(alignment);
            let magnitude_rad_s = alignment.bias_gyro.norm();
            if let Some(limit) = self.config.max_gyro_bias_magnitude_rad_s {
                if magnitude_rad_s > limit {
                    let err = MotionBasedViRejectionReason::GyroBiasOutOfRange {
                        kf_id: kf_ids[0],
                        magnitude_rad_s,
                        limit_rad_s: limit,
                    };
                    self.last_rejection = Some(err.clone());
                    return Err(err);
                }
            }
            effective_bias_gyro = alignment.bias_gyro;
            estimated_gyro_bias = Some(alignment.bias_gyro);
            for state in initial_states.values_mut() {
                state.bias_gyro = effective_bias_gyro;
            }
        }

        // Gravity-direction recovery (`estimate_gravity`): run the linear
        // alignment over `candidate_map`'s just-converted body poses
        // BEFORE dispatching to the inertial-only solver. On success, every
        // in-window factor is fed to the solver with `gravity_world`
        // overwritten to the estimate (instead of the configured value),
        // and the finite-difference velocity seeds above are overwritten
        // with the alignment's own velocities. See
        // `docs/motion_based_vi_alignment.md`'s "Gravity-direction
        // recovery" section for the motivating diagnosis and the full
        // formulation.
        let mut estimated_gravity: Option<Vector3<f64>> = None;
        if self.config.estimate_gravity {
            let expected_magnitude = self.config.gravity_world.norm();
            let alignment = estimate_gravity_and_velocities(
                &candidate_map,
                &kf_ids,
                preintegration_factors,
                effective_bias_gyro,
                bias_acc_seed,
                expected_magnitude,
            );
            let alignment = match alignment {
                Some(alignment) => alignment,
                None => {
                    self.last_gravity_alignment = None;
                    let err = MotionBasedViRejectionReason::GravityEstimateDegenerate;
                    self.last_rejection = Some(err.clone());
                    return Err(err);
                }
            };
            // Record the attempt BEFORE the observability gate below: a
            // rejected estimate is exactly as diagnostically interesting as
            // an accepted one (more so, on real data — see
            // `docs/motion_based_vi_alignment.md`'s "Gravity-direction
            // recovery" section), and every later raw-residual /
            // NIS / velocity / bias gate rejection further down this
            // function must not erase it either.
            self.last_gravity_alignment = Some(alignment.clone());
            let deviation_ratio = if expected_magnitude > 0.0 {
                (alignment.raw_gravity_norm - expected_magnitude).abs() / expected_magnitude
            } else {
                f64::INFINITY
            };
            if !deviation_ratio.is_finite()
                || deviation_ratio > self.config.max_gravity_norm_deviation_ratio
            {
                let err = MotionBasedViRejectionReason::GravityEstimateOutOfRange {
                    raw_norm_mps2: alignment.raw_gravity_norm,
                    expected_mps2: expected_magnitude,
                    max_deviation_ratio: self.config.max_gravity_norm_deviation_ratio,
                };
                self.last_rejection = Some(err.clone());
                return Err(err);
            }
            for (kf_id, velocity) in &alignment.velocities {
                if let Some(state) = initial_states.get_mut(kf_id) {
                    state.velocity_world = *velocity;
                }
            }
            estimated_gravity = Some(alignment.gravity_world);
        }
        // In-window factors carrying the estimated gravity when
        // `estimate_gravity` fired, otherwise the caller's factors
        // unchanged (legacy path). Every downstream dispatch branch below
        // solves against this slice instead of `preintegration_factors`
        // directly.
        let gravity_adjusted_factors: Option<Vec<ImuPreintegrationFactor>> =
            estimated_gravity.map(|g| {
                preintegration_factors
                    .iter()
                    .map(|f| {
                        let mut adjusted = f.clone();
                        adjusted.gravity_world = g;
                        adjusted
                    })
                    .collect()
            });
        let solve_factors: &[ImuPreintegrationFactor] = gravity_adjusted_factors
            .as_deref()
            .unwrap_or(preintegration_factors);

        // Dispatch, in priority order:
        // 1. `!bias_free` (`BiasReleaseSchedule` Stage A / "velocity
        //    stage"): bias is fixed at the seed regardless of `viba2` —
        //    scale recovery needs free bias, so VIBA2 never runs here.
        // 2. `bias_free` + `viba2` configured: the VIBA2 outer
        //    scale-recovery loop (legacy path, or Stage B).
        // 3. `bias_free`, no `viba2`: the standalone VIBA1 inertial-only
        //    path (legacy path, or Stage B).
        // All three return the same `MotionBasedViInitializationResult`
        // shape; only the VIBA2 branch populates `scale_history` /
        // `viba2_iterations_run`.
        let result = if !bias_free {
            let stats = run_inertial_only_vi_ba_with_options(
                &mut candidate_map,
                &kf_ids,
                solve_factors,
                &initial_states,
                &self.config.ba_config,
                false,
            );
            let stats = match stats {
                Some(s) => s,
                None => {
                    let any_in_window = solve_factors.iter().any(|f| {
                        kf_ids.contains(&f.keyframe_id_from) && kf_ids.contains(&f.keyframe_id_to)
                    });
                    let err = if any_in_window {
                        MotionBasedViRejectionReason::SolverFailed
                    } else {
                        MotionBasedViRejectionReason::NoUsableImuFactors
                    };
                    self.last_rejection = Some(err.clone());
                    return Err(err);
                }
            };
            MotionBasedViInitializationResult {
                keyframe_states: stats.keyframe_states,
                keyframe_ids: stats.keyframe_ids,
                imu_factors_used: stats.imu_factor_count,
                scale: 1.0,
                scale_history: Vec::new(),
                viba2_iterations_run: 0,
                trigger_translation_meters: self.cumulative_translation,
                ba_result: stats.ba_result,
                initial_cost_breakdown: stats.initial_cost_breakdown,
                final_cost_breakdown: stats.final_cost_breakdown,
                bias_released: false,
                estimated_gravity_world: estimated_gravity,
                estimated_gyro_bias,
            }
        } else if let Some(viba2_cfg) = self.config.viba2.clone() {
            let stats = run_viba2_inertial_with_scale(
                &mut candidate_map,
                &kf_ids,
                solve_factors,
                &initial_states,
                &viba2_cfg,
            );
            let stats = match stats {
                Some(s) => s,
                None => {
                    let any_in_window = solve_factors.iter().any(|f| {
                        kf_ids.contains(&f.keyframe_id_from) && kf_ids.contains(&f.keyframe_id_to)
                    });
                    let err = if any_in_window {
                        MotionBasedViRejectionReason::SolverFailed
                    } else {
                        MotionBasedViRejectionReason::NoUsableImuFactors
                    };
                    self.last_rejection = Some(err.clone());
                    return Err(err);
                }
            };
            MotionBasedViInitializationResult {
                keyframe_states: stats.keyframe_states,
                keyframe_ids: stats.keyframe_ids,
                imu_factors_used: stats.imu_factor_count,
                scale: stats.scale,
                scale_history: stats.scale_history,
                viba2_iterations_run: stats.outer_iterations_run,
                trigger_translation_meters: self.cumulative_translation,
                ba_result: stats.ba_result,
                initial_cost_breakdown: stats.initial_cost_breakdown,
                final_cost_breakdown: stats.final_cost_breakdown,
                bias_released: true,
                estimated_gravity_world: estimated_gravity,
                estimated_gyro_bias,
            }
        } else {
            let stats = run_inertial_only_vi_ba(
                &mut candidate_map,
                &kf_ids,
                solve_factors,
                &initial_states,
                &self.config.ba_config,
            );
            let stats = match stats {
                Some(s) => s,
                None => {
                    let any_in_window = solve_factors.iter().any(|f| {
                        kf_ids.contains(&f.keyframe_id_from) && kf_ids.contains(&f.keyframe_id_to)
                    });
                    let err = if any_in_window {
                        MotionBasedViRejectionReason::SolverFailed
                    } else {
                        MotionBasedViRejectionReason::NoUsableImuFactors
                    };
                    self.last_rejection = Some(err.clone());
                    return Err(err);
                }
            };
            MotionBasedViInitializationResult {
                keyframe_states: stats.keyframe_states,
                keyframe_ids: stats.keyframe_ids,
                imu_factors_used: stats.imu_factor_count,
                scale: 1.0,
                scale_history: Vec::new(),
                viba2_iterations_run: 0,
                trigger_translation_meters: self.cumulative_translation,
                ba_result: stats.ba_result,
                initial_cost_breakdown: stats.initial_cost_breakdown,
                final_cost_breakdown: stats.final_cost_breakdown,
                bias_released: true,
                estimated_gravity_world: estimated_gravity,
                estimated_gyro_bias,
            }
        };

        let final_cost = &result.final_cost_breakdown;
        let rotation_exceeded = self
            .config
            .max_final_imu_rotation_residual_rms_rad
            .is_some_and(|limit| {
                final_cost
                    .imu_rotation_residual_rms_rad
                    .is_none_or(|value| !value.is_finite() || value > limit)
            });
        let velocity_exceeded = self
            .config
            .max_final_imu_velocity_residual_rms_mps
            .is_some_and(|limit| {
                final_cost
                    .imu_velocity_residual_rms_mps
                    .is_none_or(|value| !value.is_finite() || value > limit)
            });
        let position_exceeded = self
            .config
            .max_final_imu_position_residual_rms_meters
            .is_some_and(|limit| {
                final_cost
                    .imu_position_residual_rms_meters
                    .is_none_or(|value| !value.is_finite() || value > limit)
            });
        if rotation_exceeded || velocity_exceeded || position_exceeded {
            let err = MotionBasedViRejectionReason::ImuRawResidualOutOfRange {
                rotation_residual_rms_rad: final_cost.imu_rotation_residual_rms_rad,
                velocity_residual_rms_mps: final_cost.imu_velocity_residual_rms_mps,
                position_residual_rms_meters: final_cost.imu_position_residual_rms_meters,
                max_rotation_residual_rms_rad: self.config.max_final_imu_rotation_residual_rms_rad,
                max_velocity_residual_rms_mps: self.config.max_final_imu_velocity_residual_rms_mps,
                max_position_residual_rms_meters: self
                    .config
                    .max_final_imu_position_residual_rms_meters,
            };
            self.last_rejection = Some(err.clone());
            return Err(err);
        }

        if let Some(limit) = self.config.max_final_imu_nis_per_dof {
            if let Some(normalized_nis_per_dof) = result
                .final_cost_breakdown
                .imu_normalized_squared_residual_per_dof
                .filter(|value| !value.is_finite() || *value > limit)
            {
                let err = MotionBasedViRejectionReason::ImuNisOutOfRange {
                    normalized_nis_per_dof,
                    rotation_residual_rms_rad: result
                        .final_cost_breakdown
                        .imu_rotation_residual_rms_rad,
                    velocity_residual_rms_mps: result
                        .final_cost_breakdown
                        .imu_velocity_residual_rms_mps,
                    position_residual_rms_meters: result
                        .final_cost_breakdown
                        .imu_position_residual_rms_meters,
                    limit,
                };
                self.last_rejection = Some(err.clone());
                return Err(err);
            }
        }

        if let Some(limit) = self.config.max_velocity_magnitude_mps {
            let mut worst: Option<(u64, f64)> = None;
            for (kf_id, state) in &result.keyframe_states {
                let mag = state.velocity_world.norm();
                if mag > limit && worst.is_none_or(|(_, m)| mag > m) {
                    worst = Some((*kf_id, mag));
                }
            }
            if let Some((kf_id, mag)) = worst {
                let err = MotionBasedViRejectionReason::VelocityOutOfRange {
                    kf_id,
                    magnitude_mps: mag,
                    limit_mps: limit,
                };
                self.last_rejection = Some(err.clone());
                return Err(err);
            }
        }
        if let Some(limit) = self.config.max_gyro_bias_magnitude_rad_s {
            let worst = result
                .keyframe_states
                .iter()
                .map(|(kf_id, state)| (*kf_id, state.bias_gyro.norm()))
                .filter(|(_, magnitude)| *magnitude > limit)
                .max_by(|left, right| left.1.total_cmp(&right.1));
            if let Some((kf_id, magnitude_rad_s)) = worst {
                let err = MotionBasedViRejectionReason::GyroBiasOutOfRange {
                    kf_id,
                    magnitude_rad_s,
                    limit_rad_s: limit,
                };
                self.last_rejection = Some(err.clone());
                return Err(err);
            }
        }
        if let Some(limit) = self.config.max_accel_bias_magnitude_mps2 {
            let worst = result
                .keyframe_states
                .iter()
                .map(|(kf_id, state)| (*kf_id, state.bias_acc.norm()))
                .filter(|(_, magnitude)| *magnitude > limit)
                .max_by(|left, right| left.1.total_cmp(&right.1));
            if let Some((kf_id, magnitude_mps2)) = worst {
                let err = MotionBasedViRejectionReason::AccelBiasOutOfRange {
                    kf_id,
                    magnitude_mps2,
                    limit_mps2: limit,
                };
                self.last_rejection = Some(err.clone());
                return Err(err);
            }
        }

        if result.bias_released {
            // Legacy path or Stage B: terminal success.
            self.completed = Some(result);
            self.last_rejection = None;
            Ok(self.completed.as_ref().expect("just inserted"))
        } else {
            // Stage A: cache the velocity-only result but stay non-terminal
            // so the caller keeps registering keyframes / translation
            // toward the bias-release gate.
            self.velocity_stage = Some(result);
            self.last_rejection = None;
            Ok(self.velocity_stage.as_ref().expect("just inserted"))
        }
    }
}

/// Outcome of [`estimate_gravity_and_velocities`]: the world-frame gravity
/// vector recovered from IMU preintegration factors against FIXED visual
/// poses, plus the per-keyframe velocities that come along for free from
/// the same linear solve.
#[derive(Debug, Clone, PartialEq)]
pub struct GravityVelocityAlignment {
    /// Recovered gravity vector, magnitude-constrained to the caller's
    /// `expected_gravity_magnitude` via the VINS-Mono-style tangent-space
    /// refinement described on [`estimate_gravity_and_velocities`].
    pub gravity_world: Vector3<f64>,
    /// Norm of the UNCONSTRAINED linear solve's gravity estimate, before
    /// the magnitude-refinement pass. This is the observability check: a
    /// well-conditioned window recovers a norm near the true magnitude on
    /// its own, with no norm constraint imposed anywhere in that solve.
    /// Callers gate promotion on how close this sits to the expected
    /// magnitude (see
    /// [`MotionBasedViInitializerConfig::max_gravity_norm_deviation_ratio`]).
    pub raw_gravity_norm: f64,
    /// Per-keyframe world-frame velocities from the final refinement
    /// iteration (or from the unconstrained solve, on the plain
    /// scale-to-norm fallback — see the function doc comment). Keyed by
    /// keyframe id; only keyframes included in the (possibly capped, see
    /// below) alignment window are present.
    pub velocities: BTreeMap<u64, Vector3<f64>>,
    /// RMS of the stacked position + velocity preintegration residual (6
    /// scalar components per in-window factor) evaluated at the final
    /// `(gravity_world, velocities)` estimate. Mixes position (m) and
    /// velocity (m/s) residual components into one scalar, so treat as a
    /// relative diagnostic rather than a physically-calibrated quantity —
    /// the initializer's actual promotion decision continues to rely on
    /// the existing raw-residual gates, evaluated by the solver AFTER this
    /// alignment feeds it the estimated gravity.
    pub mean_residual_after: f64,
    /// Number of keyframes in the (possibly window-capped, see
    /// `MAX_ALIGNMENT_WINDOW` on [`estimate_gravity_and_velocities`])
    /// alignment window this estimate was solved over. Diagnostic only —
    /// lets a caller logging [`MotionBasedViInitializer::last_gravity_alignment`]
    /// distinguish "recovered from 10 well-spaced keyframes" from "recovered
    /// from the bare minimum of 2".
    pub window_keyframes: usize,
}

/// One IMU preintegration factor's contribution to the gravity/velocity
/// alignment's linear system, precomputed once so both the unconstrained
/// solve, the tangent-space refinement, and the final residual check can
/// share it without re-deriving the per-factor geometry each time.
struct AlignmentFactorRow {
    /// Index (into the alignment's local keyframe ordering) of the
    /// factor's `keyframe_id_from`.
    idx_from: usize,
    /// Index of `keyframe_id_to`.
    idx_to: usize,
    delta_time: f64,
    /// `(p_j - p_i) - R_wb_i * Δp_ij` — the position equation's known
    /// right-hand side once `(v_i, g)` are moved to the left. See the
    /// function doc comment for the full derivation.
    rhs_position: Vector3<f64>,
    /// `R_wb_i * Δv_ij` — the velocity equation's known right-hand side.
    rhs_velocity: Vector3<f64>,
}

/// Recover the world-frame gravity vector and per-keyframe velocities from
/// a window of IMU preintegration factors evaluated against FIXED visual
/// body poses (`map`'s poses are assumed already converted to
/// world-to-body, exactly as [`MotionBasedViInitializer::try_initialize_with_bias_seed`]'s
/// `candidate_map` conversion produces).
///
/// This is the missing piece that lets the motion-based initializer
/// recover from an identity-seeded (sub)map whose world frame has gravity
/// pointing in an arbitrary direction: rather than trusting
/// `expected_gravity_magnitude`'s DIRECTION (only the magnitude is used,
/// as the physical constant the recovered vector must match), the world
/// frame's actual gravity direction is estimated from the IMU evidence
/// itself. See `docs/motion_based_vi_alignment.md`'s "Gravity-direction
/// recovery" section for the full motivating diagnosis.
///
/// ## Formulation
///
/// With poses known (fixed), the preintegration relations (Forster et al.
/// 2017; ORB-SLAM3 inertial-only MAP) become LINEAR in the unknowns
/// `x = [v_1 .. v_N, g]` (all in world frame), one keyframe velocity per
/// window keyframe plus the shared gravity vector:
///
/// * Position: `v_i · Δt + ½ · g · Δt² = (p_j − p_i) − R_wb_i · Δp_ij`
/// * Velocity: `−v_i + v_j − g · Δt = R_wb_i · Δv_ij`
///
/// where `R_wb_i` is keyframe `i`'s body-to-world rotation, `Δp_ij` /
/// `Δv_ij` are the bias-corrected preintegrated deltas (first-order
/// corrected at `(bias_gyro_seed, bias_acc_seed)` via
/// [`crate::imu_preintegration::ImuPreintegratedDelta::corrected`] when
/// the delta carries bias Jacobians, a no-op correction otherwise), and
/// `Δt` is the factor's `delta_time`. Each factor contributes 6 rows; the
/// window is capped to the most recent `MAX_ALIGNMENT_WINDOW` keyframes
/// (currently 10) since a dense SVD solve over the full un-capped window
/// is unnecessary — a handful of well-spaced factors already fully
/// constrains 3-DoF gravity, and 10 keeps the dense system small
/// regardless of how long a [`BiasReleaseSchedule`] Stage A window grows
/// while awaiting its release gate.
///
/// ## Norm-constrained refinement
///
/// The unconstrained least-squares solve above (via `nalgebra`'s SVD) is
/// evaluated first; its gravity estimate's norm (`raw_gravity_norm`) is
/// the observability check callers gate on. The vector is then refined to
/// match `expected_gravity_magnitude` exactly using the VINS-Mono
/// tangent-space parameterization (Qin & Shen 2018, eq. 7-9):
/// `g = mag · normalize(ĝ + w1·b1 + w2·b2)` where `(b1, b2)` is an
/// orthonormal basis tangent to the current unit estimate `ĝ`. Each
/// iteration re-solves the linear system with `g`'s 3 free components
/// replaced by the 2 tangent coordinates `(w1, w2)`, updates `ĝ`, and
/// rebuilds the tangent basis; 4 iterations are run unconditionally
/// (empirically sufficient for this magnitude of nonlinearity). If any
/// iteration's solve fails (singular system) or produces a non-finite
/// update, refinement stops early and the FALLBACK — a plain
/// scale-to-norm of the unconstrained estimate (`mag · ĝ_raw`), paired
/// with the unconstrained solve's own velocities — is returned instead.
/// This fallback is deliberately the ONLY plain-rescale path: it is never
/// taken when the tangent iteration converges.
///
/// Returns `None` on degeneracy: fewer than 2 usable in-window factors
/// (after applying the window cap and requiring `delta_time > 0`), any
/// windowed keyframe missing its pose, or the unconstrained solve itself
/// producing a non-finite / near-zero gravity vector.
pub fn estimate_gravity_and_velocities(
    map: &VisualMap,
    keyframe_ids: &[u64],
    factors: &[ImuPreintegrationFactor],
    bias_gyro_seed: Vector3<f64>,
    bias_acc_seed: Vector3<f64>,
    expected_gravity_magnitude: f64,
) -> Option<GravityVelocityAlignment> {
    const MAX_ALIGNMENT_WINDOW: usize = 10;
    const REFINEMENT_ITERATIONS: usize = 4;

    if !expected_gravity_magnitude.is_finite() || expected_gravity_magnitude <= 0.0 {
        return None;
    }

    let mut ids: Vec<u64> = keyframe_ids.to_vec();
    ids.sort_unstable();
    ids.dedup();
    if ids.len() > MAX_ALIGNMENT_WINDOW {
        let start = ids.len() - MAX_ALIGNMENT_WINDOW;
        ids = ids[start..].to_vec();
    }
    if ids.len() < 2 {
        return None;
    }
    let in_window: BTreeSet<u64> = ids.iter().copied().collect();
    let index_of: BTreeMap<u64, usize> =
        ids.iter().enumerate().map(|(idx, id)| (*id, idx)).collect();

    // Body-to-world rotation (`R_wb_i`) and world-frame body position
    // (`p_i`) per windowed keyframe. `map`'s poses are already `T_bw`
    // (world-to-body) at the call sites in this module.
    let mut rotation_body_to_world: BTreeMap<u64, UnitQuaternion<f64>> = BTreeMap::new();
    let mut position_world: BTreeMap<u64, Vector3<f64>> = BTreeMap::new();
    for id in &ids {
        let pose = map.keyframes.get(id)?.frame.pose.as_ref()?;
        let world_from_body = pose.camera_to_world();
        rotation_body_to_world.insert(*id, world_from_body.rotation);
        position_world.insert(*id, world_from_body.translation);
    }

    let mut rows: Vec<AlignmentFactorRow> = Vec::new();
    for factor in factors {
        if !in_window.contains(&factor.keyframe_id_from)
            || !in_window.contains(&factor.keyframe_id_to)
        {
            continue;
        }
        let delta_time = factor.delta.delta_time;
        if delta_time <= 0.0 {
            continue;
        }
        let idx_from = *index_of.get(&factor.keyframe_id_from).expect("windowed id");
        let idx_to = *index_of.get(&factor.keyframe_id_to).expect("windowed id");
        let r_i = *rotation_body_to_world
            .get(&factor.keyframe_id_from)
            .expect("windowed id");
        let p_i = *position_world
            .get(&factor.keyframe_id_from)
            .expect("windowed id");
        let p_j = *position_world
            .get(&factor.keyframe_id_to)
            .expect("windowed id");
        let (_, delta_velocity, delta_position) =
            factor.delta.corrected(&bias_gyro_seed, &bias_acc_seed);
        rows.push(AlignmentFactorRow {
            idx_from,
            idx_to,
            delta_time,
            rhs_position: (p_j - p_i) - r_i.transform_vector(&delta_position),
            rhs_velocity: r_i.transform_vector(&delta_velocity),
        });
    }
    if rows.len() < 2 {
        return None;
    }

    let num_kf = ids.len();
    let num_rows = 6 * rows.len();

    // Unconstrained solve: unknowns `[v_1 .. v_N, g]`.
    let mut a = DMatrix::<f64>::zeros(num_rows, 3 * num_kf + 3);
    let mut b = DVector::<f64>::zeros(num_rows);
    for (r, row) in rows.iter().enumerate() {
        let base = 6 * r;
        for k in 0..3 {
            a[(base + k, 3 * row.idx_from + k)] = row.delta_time;
            a[(base + k, 3 * num_kf + k)] = 0.5 * row.delta_time * row.delta_time;
            b[base + k] = row.rhs_position[k];

            a[(base + 3 + k, 3 * row.idx_from + k)] = -1.0;
            a[(base + 3 + k, 3 * row.idx_to + k)] = 1.0;
            a[(base + 3 + k, 3 * num_kf + k)] = -row.delta_time;
            b[base + 3 + k] = row.rhs_velocity[k];
        }
    }
    let raw_solution = a.svd(true, true).solve(&b, 1.0e-9).ok()?;
    let raw_gravity = Vector3::new(
        raw_solution[3 * num_kf],
        raw_solution[3 * num_kf + 1],
        raw_solution[3 * num_kf + 2],
    );
    let raw_gravity_norm = raw_gravity.norm();
    if !raw_gravity_norm.is_finite() || raw_gravity_norm < 1.0e-6 {
        return None;
    }
    let raw_velocities: Vec<Vector3<f64>> = (0..num_kf)
        .map(|k| {
            Vector3::new(
                raw_solution[3 * k],
                raw_solution[3 * k + 1],
                raw_solution[3 * k + 2],
            )
        })
        .collect();

    // Norm-constrained refinement (VINS-Mono tangent-space iteration).
    // Falls back to a plain scale-to-norm of `raw_gravity` (paired with
    // `raw_velocities`) the moment any iteration fails to solve or
    // produces a non-finite update.
    let mag = expected_gravity_magnitude;
    let mut g_hat = raw_gravity / raw_gravity_norm;
    let mut velocities_final = raw_velocities.clone();
    let mut refined = false;
    for _ in 0..REFINEMENT_ITERATIONS {
        let (b1, b2) = tangent_basis(&g_hat);
        let g0 = mag * g_hat;
        let mut a2 = DMatrix::<f64>::zeros(num_rows, 3 * num_kf + 2);
        let mut b2vec = DVector::<f64>::zeros(num_rows);
        for (r, row) in rows.iter().enumerate() {
            let base = 6 * r;
            for k in 0..3 {
                a2[(base + k, 3 * row.idx_from + k)] = row.delta_time;
                a2[(base + k, 3 * num_kf)] = 0.5 * row.delta_time * row.delta_time * mag * b1[k];
                a2[(base + k, 3 * num_kf + 1)] =
                    0.5 * row.delta_time * row.delta_time * mag * b2[k];
                b2vec[base + k] =
                    row.rhs_position[k] - 0.5 * row.delta_time * row.delta_time * g0[k];

                a2[(base + 3 + k, 3 * row.idx_from + k)] = -1.0;
                a2[(base + 3 + k, 3 * row.idx_to + k)] = 1.0;
                a2[(base + 3 + k, 3 * num_kf)] = -row.delta_time * mag * b1[k];
                a2[(base + 3 + k, 3 * num_kf + 1)] = -row.delta_time * mag * b2[k];
                b2vec[base + 3 + k] = row.rhs_velocity[k] + row.delta_time * g0[k];
            }
        }
        let Ok(solution) = a2.svd(true, true).solve(&b2vec, 1.0e-9) else {
            break;
        };
        let w1 = solution[3 * num_kf];
        let w2 = solution[3 * num_kf + 1];
        if !w1.is_finite() || !w2.is_finite() {
            break;
        }
        let candidate = g_hat + w1 * b1 + w2 * b2;
        let candidate_norm = candidate.norm();
        if !candidate_norm.is_finite() || candidate_norm < 1.0e-9 {
            break;
        }
        g_hat = candidate / candidate_norm;
        velocities_final = (0..num_kf)
            .map(|k| Vector3::new(solution[3 * k], solution[3 * k + 1], solution[3 * k + 2]))
            .collect();
        refined = true;
    }

    let gravity_world = if refined {
        mag * g_hat
    } else {
        mag * (raw_gravity / raw_gravity_norm)
    };

    let mut residual_sum_sq = 0.0;
    for row in &rows {
        let v_i = velocities_final[row.idx_from];
        let v_j = velocities_final[row.idx_to];
        let r_pos = (v_i * row.delta_time + 0.5 * row.delta_time * row.delta_time * gravity_world)
            - row.rhs_position;
        let r_vel = (-v_i + v_j - row.delta_time * gravity_world) - row.rhs_velocity;
        residual_sum_sq += r_pos.norm_squared() + r_vel.norm_squared();
    }
    let mean_residual_after = (residual_sum_sq / (6.0 * rows.len() as f64)).sqrt();

    let velocities: BTreeMap<u64, Vector3<f64>> = ids
        .iter()
        .enumerate()
        .map(|(k, id)| (*id, velocities_final[k]))
        .collect();

    Some(GravityVelocityAlignment {
        gravity_world,
        raw_gravity_norm,
        velocities,
        mean_residual_after,
        window_keyframes: num_kf,
    })
}

/// Orthonormal basis `(b1, b2)` tangent to the unit vector `g_hat`, used
/// by [`estimate_gravity_and_velocities`]'s norm-constrained refinement
/// (VINS-Mono style: any basis spanning the tangent plane works, since the
/// refinement re-derives it fresh from the updated `g_hat` every
/// iteration).
fn tangent_basis(g_hat: &Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let reference = if g_hat.x.abs() < 0.9 {
        Vector3::x()
    } else {
        Vector3::y()
    };
    let b1 = g_hat.cross(&reference).normalize();
    let b2 = g_hat.cross(&b1).normalize();
    (b1, b2)
}

/// Outcome of [`estimate_gyro_bias`]: the shared gyro bias recovered from
/// ROTATION-ONLY alignment against a window's fixed visual poses, run
/// BEFORE gravity/velocity alignment — see `docs/motion_based_vi_alignment.md`'s
/// "Gyro-bias recovery" section for the motivating real-data diagnosis and
/// the full derivation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GyroBiasAlignment {
    /// Recovered gyro bias, in the same units/frame as
    /// [`ImuPreintegrationFactor::delta`]'s bias linearisation (rad/s,
    /// body frame).
    pub bias_gyro: Vector3<f64>,
    /// Number of Gauss-Newton iterations actually run (capped at 5,
    /// possibly fewer on early convergence — see [`estimate_gyro_bias`]).
    pub iterations: usize,
    /// RMS of the per-factor rotation residual `Log(ΔR_corrected(b_g_seed)⁻¹
    /// · R_iᵀ · R_j)` evaluated at the CALLER's `bias_gyro_seed`, before any
    /// Gauss-Newton update. Diagnostic: compare against
    /// `rotation_residual_rms_after` to see how much the alignment moved
    /// the residual.
    pub rotation_residual_rms_before: f64,
    /// RMS of the same rotation residual evaluated at the FINAL
    /// `bias_gyro` estimate.
    pub rotation_residual_rms_after: f64,
}

/// Recover the shared gyro bias from a window of IMU preintegration
/// factors' ROTATION component evaluated against FIXED visual body poses
/// (`map`'s poses are assumed already converted to world-to-body, exactly
/// as [`MotionBasedViInitializer::try_initialize_with_bias_seed`]'s
/// `candidate_map` conversion produces — same precondition as
/// [`estimate_gravity_and_velocities`]).
///
/// ## Formulation
///
/// This is the classical first step of ORB-SLAM3's inertial-only MAP
/// estimator and VINS-Mono's IMU initialization (Qin & Shen 2018, §5.2):
/// with poses known (fixed), the ONLY unknown left in the rotation-only
/// preintegration relation is the gyro bias `b_g`. Per in-window
/// consecutive factor `i→j`, define the residual
///
/// `r_ij(b_g) = Log( ΔR_ij(b_g)⁻¹ · R_iᵀ · R_j )`
///
/// where `ΔR_ij(b_g)` is [`crate::imu_preintegration::ImuPreintegratedDelta::corrected`]'s
/// bias-corrected rotation at `b_g` (accelerometer bias held at the
/// factor's own linearisation point — this residual has no accelerometer
/// dependence) and `R_i`, `R_j` are the windowed keyframes' body-to-world
/// rotations. Gauss-Newton solves `min_{b_g} Σ ||r_ij(b_g)||²` using the
/// preintegrator's own `∂(Log ΔR)/∂b_g` Jacobian
/// ([`crate::imu_preintegration::ImuPreintegratedDelta::j_rotation_bg`]) as
/// the (first-order, and thus approximately step-invariant near the
/// linearisation point) local model per iteration:
///
/// `(Σ Jᵢⱼᵀ Jᵢⱼ) · δ = Σ Jᵢⱼᵀ · r_ij(b_g)`, then `b_g ← b_g + δ`
///
/// re-evaluating the exact (nonlinear) residual `r_ij` at the updated `b_g`
/// each iteration. Up to 5 iterations run, stopping early once `‖δ‖ <
/// 1e-10`. This is the same normal-equation shape used by VINS-Mono's
/// `solveGyroscopeBias` (there run as a single linear solve at `b_g = 0`);
/// iterating a few times here lets the alignment start from a nonzero
/// `bias_gyro_seed` and re-linearise around the improving estimate.
///
/// The window is capped to the most recent 10 keyframes (mirroring
/// [`estimate_gravity_and_velocities`]'s `MAX_ALIGNMENT_WINDOW`). Returns
/// `None` on degeneracy: fewer than 2 usable in-window factors (missing
/// keyframe poses, non-positive `delta_time`, or a factor connecting two
/// keyframes outside the window all count as unusable), or a singular
/// normal-equation system on the very first iteration.
pub fn estimate_gyro_bias(
    map: &VisualMap,
    keyframe_ids: &[u64],
    factors: &[ImuPreintegrationFactor],
    bias_gyro_seed: Vector3<f64>,
) -> Option<GyroBiasAlignment> {
    const MAX_ALIGNMENT_WINDOW: usize = 10;
    const MAX_ITERATIONS: usize = 5;
    const CONVERGENCE_STEP_NORM: f64 = 1.0e-10;

    let mut ids: Vec<u64> = keyframe_ids.to_vec();
    ids.sort_unstable();
    ids.dedup();
    if ids.len() > MAX_ALIGNMENT_WINDOW {
        let start = ids.len() - MAX_ALIGNMENT_WINDOW;
        ids = ids[start..].to_vec();
    }
    if ids.len() < 2 {
        return None;
    }
    let in_window: BTreeSet<u64> = ids.iter().copied().collect();

    let mut rotation_body_to_world: BTreeMap<u64, UnitQuaternion<f64>> = BTreeMap::new();
    for id in &ids {
        let pose = map.keyframes.get(id)?.frame.pose.as_ref()?;
        rotation_body_to_world.insert(*id, pose.camera_to_world().rotation);
    }

    struct GyroBiasRow {
        r_i: UnitQuaternion<f64>,
        r_j: UnitQuaternion<f64>,
        delta_index: usize,
    }
    let mut rows: Vec<GyroBiasRow> = Vec::new();
    let mut deltas: Vec<&ImuPreintegratedDelta> = Vec::new();
    for factor in factors {
        if !in_window.contains(&factor.keyframe_id_from)
            || !in_window.contains(&factor.keyframe_id_to)
        {
            continue;
        }
        if factor.delta.delta_time <= 0.0 {
            continue;
        }
        let Some(&r_i) = rotation_body_to_world.get(&factor.keyframe_id_from) else {
            continue;
        };
        let Some(&r_j) = rotation_body_to_world.get(&factor.keyframe_id_to) else {
            continue;
        };
        deltas.push(&factor.delta);
        rows.push(GyroBiasRow {
            r_i,
            r_j,
            delta_index: deltas.len() - 1,
        });
    }
    if rows.len() < 2 {
        return None;
    }

    // Rotation residual at a given gyro bias: `Log(ΔR_corrected(b_g)⁻¹ ·
    // R_iᵀ · R_j)`, matching `ImuPreintegrationFactor::residual_corrected_internal`'s
    // `r_rot` sign/ordering convention exactly. Accelerometer bias is held
    // at each delta's own linearisation point (`δb_a = 0`) since this
    // residual has no accelerometer dependence.
    let residual_at = |bias_gyro: &Vector3<f64>| -> Vec<Vector3<f64>> {
        rows.iter()
            .map(|row| {
                let delta = deltas[row.delta_index];
                let (delta_rot, _, _) = delta.corrected(bias_gyro, &delta.bias_acc_linearisation);
                let q_rel = delta_rot.quaternion().inverse() * row.r_i.inverse() * row.r_j;
                q_rel.scaled_axis()
            })
            .collect()
    };
    let rms = |residuals: &[Vector3<f64>]| -> f64 {
        let sum_sq: f64 = residuals.iter().map(|r| r.norm_squared()).sum();
        (sum_sq / (3.0 * residuals.len() as f64)).sqrt()
    };

    let residuals_before = residual_at(&bias_gyro_seed);
    let rotation_residual_rms_before = rms(&residuals_before);

    let mut bias_gyro = bias_gyro_seed;
    let mut residuals_current = residuals_before;
    let mut iterations = 0usize;
    for iteration in 0..MAX_ITERATIONS {
        let mut ata = Matrix3::zeros();
        let mut atb = Vector3::zeros();
        for (row, residual) in rows.iter().zip(residuals_current.iter()) {
            let j = deltas[row.delta_index].j_rotation_bg;
            ata += j.transpose() * j;
            atb += j.transpose() * *residual;
        }
        let Some(ata_inv) = ata.try_inverse() else {
            if iteration == 0 {
                return None;
            }
            break;
        };
        let delta_step = ata_inv * atb;
        if !delta_step.iter().all(|v| v.is_finite()) {
            if iteration == 0 {
                return None;
            }
            break;
        }
        bias_gyro += delta_step;
        iterations += 1;
        residuals_current = residual_at(&bias_gyro);
        if delta_step.norm() < CONVERGENCE_STEP_NORM {
            break;
        }
    }
    let rotation_residual_rms_after = rms(&residuals_current);

    Some(GyroBiasAlignment {
        bias_gyro,
        iterations,
        rotation_residual_rms_before,
        rotation_residual_rms_after,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use nalgebra::{Point3, UnitQuaternion};
    use visloc_core::geometry::{Pose, SO3};
    use visloc_core::types::{Camera, Frame, Keyframe, VisualMap};

    use crate::imu_preintegration::ImuPreintegratedDelta;

    fn pinhole_camera() -> Camera {
        Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0)
    }

    fn pose_at_world_center(c: Vector3<f64>) -> Pose {
        // World-to-camera with identity orientation: t = -R·C = -C.
        let r = UnitQuaternion::identity();
        Pose::from_world_to_camera(r, -r.transform_vector(&c))
    }

    fn synthetic_seed() -> VisualInertialInitializationResult {
        VisualInertialInitializationResult {
            gravity_world: Vector3::new(0.0, 9.81, 0.0),
            initial_rotation_body_to_world: UnitQuaternion::identity(),
            initial_velocity_world: Vector3::zeros(),
            bias_gyro: Vector3::zeros(),
            bias_acc: Vector3::zeros(),
            samples_consumed: 2000,
            duration_seconds: 1.0,
            gyro_std: Vector3::zeros(),
            accel_std: Vector3::zeros(),
            mean_accel_magnitude: 9.81,
        }
    }

    fn no_acceleration_factor(
        from_id: u64,
        to_id: u64,
        delta_t: f64,
        gravity: Vector3<f64>,
    ) -> ImuPreintegrationFactor {
        // Synthetic factor for an identity-oriented body with NO
        // proper acceleration (i.e., stationary or constant-velocity
        // in world frame). The IMU still senses gravity as specific
        // force, so the pre-integrated body-frame deltas are:
        //   ΔR = identity
        //   Δv = -g · Δt
        //   Δp = -0.5 · g · Δt²
        // With these, the residual
        //   r_v = R_b←w · (v_j - v_i - g·Δt) - Δv
        //       = (v_j - v_i - g·Δt) + g·Δt = v_j - v_i
        // is exactly satisfied for stationary AND for constant-velocity
        // motion (because the IMU integration's gravity term cancels the
        // residual's gravity term), as required for the "no proper
        // acceleration" synthetic scenarios used in these unit tests.
        let mut delta = ImuPreintegratedDelta::identity();
        delta.delta_time = delta_t;
        delta.delta_velocity = -gravity * delta_t;
        delta.delta_position = -0.5 * gravity * delta_t * delta_t;
        ImuPreintegrationFactor {
            keyframe_id_from: from_id,
            keyframe_id_to: to_id,
            delta,
            gravity_world: gravity,
            weight_position: 1.0,
            weight_velocity: 1.0,
            weight_rotation: 1.0,
        }
    }

    /// Like [`no_acceleration_factor`] but decouples the TRUE gravity baked
    /// into the synthetic `(Δv, Δp)` from the ASSUMED gravity stamped onto
    /// `ImuPreintegrationFactor::gravity_world`. Models the diagnosed
    /// real-data bug directly: the physical IMU evidence is ground truth
    /// (built from `true_gravity`), while the factor's `gravity_world` is
    /// whatever the caller's (possibly wrong) config assumes — exactly what
    /// the legacy (`estimate_gravity: false`) path feeds the solver
    /// verbatim, and what the `estimate_gravity: true` path OVERWRITES with
    /// its own recovered estimate before solving (see
    /// `estimate_gravity_and_velocities`'s call site in
    /// `try_initialize_with_bias_seed`).
    fn no_acceleration_factor_with_assumed_gravity(
        from_id: u64,
        to_id: u64,
        delta_t: f64,
        true_gravity: Vector3<f64>,
        assumed_gravity: Vector3<f64>,
    ) -> ImuPreintegrationFactor {
        let mut delta = ImuPreintegratedDelta::identity();
        delta.delta_time = delta_t;
        delta.delta_velocity = -true_gravity * delta_t;
        delta.delta_position = -0.5 * true_gravity * delta_t * delta_t;
        ImuPreintegrationFactor {
            keyframe_id_from: from_id,
            keyframe_id_to: to_id,
            delta,
            gravity_world: assumed_gravity,
            weight_position: 1.0,
            weight_velocity: 1.0,
            weight_rotation: 1.0,
        }
    }

    /// Build a `num_keyframes`-long map of identity-oriented keyframes on a
    /// straight-line, constant-world-velocity trajectory `p_i = velocity *
    /// i * dt`. Generalises [`build_constant_velocity_map`] (which is
    /// pinned to unit-speed +x motion) to an arbitrary off-axis velocity,
    /// needed so the gravity/velocity alignment tests below can exercise a
    /// trajectory that is not co-linear with the true gravity vector under
    /// test.
    fn build_map_at_velocity(num_keyframes: usize, dt: f64, velocity: Vector3<f64>) -> VisualMap {
        let mut map = VisualMap::new();
        let camera = pinhole_camera();
        map.cameras.insert(camera.id, camera.clone());
        for i in 0..num_keyframes {
            let center = velocity * (i as f64) * dt;
            let pose = pose_at_world_center(center);
            let frame_id = (i as u64) + 1;
            let mut frame = Frame::new(frame_id, camera.id);
            frame.pose = Some(pose);
            let kf = Keyframe {
                frame,
                observations: Vec::new(),
            };
            map.keyframes.insert(frame_id, kf);
        }
        map
    }

    fn build_constant_velocity_map(num_keyframes: usize) -> VisualMap {
        let mut map = VisualMap::new();
        let camera = pinhole_camera();
        map.cameras.insert(camera.id, camera.clone());
        for i in 0..num_keyframes {
            let center = Vector3::new(i as f64, 0.0, 0.0);
            let pose = pose_at_world_center(center);
            let frame_id = (i as u64) + 1;
            let mut frame = Frame::new(frame_id, camera.id);
            frame.pose = Some(pose);
            let kf = Keyframe {
                frame,
                observations: Vec::new(),
            };
            map.keyframes.insert(frame_id, kf);
        }
        map
    }

    #[test]
    fn trigger_blocked_until_keyframe_threshold_is_met() {
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 5,
            min_translation_meters: 0.0,
            ..MotionBasedViInitializerConfig::default()
        });
        for i in 0..4 {
            init.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        assert!(!init.is_ready(), "fewer keyframes than the threshold");
        let mut map = build_constant_velocity_map(4);
        let seed = synthetic_seed();
        let err = init
            .try_initialize(&mut map, &[], &seed)
            .expect_err("threshold not met");
        match err {
            MotionBasedViRejectionReason::InsufficientKeyframes { have, need } => {
                assert_eq!(have, 4);
                assert_eq!(need, 5);
            }
            other => panic!("unexpected rejection: {other:?}"),
        }
    }

    #[test]
    fn trigger_blocked_until_translation_threshold_is_met() {
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 5.0,
            ..MotionBasedViInitializerConfig::default()
        });
        init.register_keyframe(1, Point3::new(0.0, 0.0, 0.0));
        init.register_keyframe(2, Point3::new(1.0, 0.0, 0.0));
        init.register_keyframe(3, Point3::new(2.0, 0.0, 0.0));
        assert_eq!(init.keyframes_observed(), 3);
        assert!(
            (init.cumulative_translation_meters() - 2.0).abs() < 1e-12,
            "cumulative_translation tracks chain length, not endpoint distance"
        );
        assert!(!init.is_ready());
        let mut map = build_constant_velocity_map(3);
        let seed = synthetic_seed();
        let err = init
            .try_initialize(&mut map, &[], &seed)
            .expect_err("translation threshold not met");
        match err {
            MotionBasedViRejectionReason::InsufficientTranslation { have, need } => {
                assert!((have - 2.0).abs() < 1e-12);
                assert!((need - 5.0).abs() < 1e-12);
            }
            other => panic!("unexpected rejection: {other:?}"),
        }
    }

    #[test]
    fn re_registering_same_keyframe_does_not_double_count_translation() {
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig::default());
        init.register_keyframe(1, Point3::new(0.0, 0.0, 0.0));
        init.register_keyframe(2, Point3::new(1.0, 0.0, 0.0));
        init.register_keyframe(2, Point3::new(1.5, 0.0, 0.0));
        assert_eq!(init.keyframes_observed(), 2);
        // Only the inter-frame displacement from the FIRST insertion is
        // counted — the re-registration overwrites the centre but does
        // not append to the chain.
        assert!((init.cumulative_translation_meters() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn no_usable_factor_is_reported_distinctly_from_solver_failure() {
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            ..MotionBasedViInitializerConfig::default()
        });
        for i in 0..3 {
            init.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        assert!(init.is_ready());
        let mut map = build_constant_velocity_map(3);
        let seed = synthetic_seed();
        let err = init
            .try_initialize(&mut map, &[], &seed)
            .expect_err("no factors registered");
        match err {
            MotionBasedViRejectionReason::NoUsableImuFactors => {}
            other => panic!("unexpected rejection: {other:?}"),
        }
    }

    #[test]
    fn successful_solve_is_cached_and_reset_re_arms() {
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            ..MotionBasedViInitializerConfig::default()
        });
        let gravity = init.config().gravity_world;
        for i in 0..3 {
            init.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        let mut map = build_constant_velocity_map(3);
        let map_before = map.clone();
        let seed = synthetic_seed();
        let factors = vec![
            no_acceleration_factor(1, 2, 1.0, gravity),
            no_acceleration_factor(2, 3, 1.0, gravity),
        ];

        let result = init
            .try_initialize(&mut map, &factors, &seed)
            .expect("VIBA1 must succeed on synthetic constant-velocity stream")
            .clone();
        assert_eq!(
            map, map_before,
            "inertial-only initialization must keep the visual trajectory fixed"
        );
        assert_eq!(result.keyframe_ids, vec![1, 2, 3]);
        assert_eq!(result.imu_factors_used, 2);
        assert!((result.scale - 1.0).abs() < 1e-12);
        assert!((result.trigger_translation_meters - 2.0).abs() < 1e-12);
        // Solver populated every registered keyframe with a state slot,
        // preserved the fixed vision-only trajectory, and terminated below
        // the initial cost.
        for kf_id in [1u64, 2, 3] {
            assert!(result.keyframe_states.contains_key(&kf_id));
        }
        assert!(result.ba_result.final_cost <= result.ba_result.initial_cost + 1e-9);
        match init.status() {
            MotionBasedViInitializationStatus::Initialised { .. } => {}
            other => panic!("expected Initialised, got {other:?}"),
        }

        // Re-firing returns the cached result rather than re-running.
        let cached = init.try_initialize(&mut map, &factors, &seed).unwrap();
        assert_eq!(cached.keyframe_ids, result.keyframe_ids);

        // After reset, the state machine returns to Waiting and can
        // re-fire on a fresh sequence.
        init.reset();
        assert_eq!(init.keyframes_observed(), 0);
        assert!((init.cumulative_translation_meters() - 0.0).abs() < 1e-12);
        match init.status() {
            MotionBasedViInitializationStatus::Waiting {
                keyframes_observed,
                cumulative_translation_meters,
                ..
            } => {
                assert_eq!(keyframes_observed, 0);
                assert!(cumulative_translation_meters.abs() < 1e-12);
            }
            other => panic!("expected Waiting after reset, got {other:?}"),
        }
    }

    #[test]
    fn stereo_stationary_replay_is_identity() {
        // VIBA1 on three coincident keyframes (zero motion) reproduces
        // the static seed: zero velocity, zero biases. The trigger is
        // forced by relaxing `min_keyframes` and `min_translation_meters`
        // so the "no motion" case is still solvable for the no-op check.
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 0.0,
            ..MotionBasedViInitializerConfig::default()
        });
        let gravity = init.config().gravity_world;
        for i in 0..3 {
            init.register_keyframe((i as u64) + 1, Point3::new(0.0, 0.0, 0.0));
        }
        let mut map = VisualMap::new();
        let camera = pinhole_camera();
        map.cameras.insert(camera.id, camera.clone());
        for i in 0..3 {
            let pose = pose_at_world_center(Vector3::zeros());
            let frame_id = (i as u64) + 1;
            let mut frame = Frame::new(frame_id, camera.id);
            frame.pose = Some(pose);
            let kf = Keyframe {
                frame,
                observations: Vec::new(),
            };
            map.keyframes.insert(frame_id, kf);
        }
        let seed = synthetic_seed();
        let factors = vec![
            no_acceleration_factor(1, 2, 0.1, gravity),
            no_acceleration_factor(2, 3, 0.1, gravity),
        ];
        let result = init
            .try_initialize(&mut map, &factors, &seed)
            .expect("identity replay should succeed");
        for state in result.keyframe_states.values() {
            assert!(
                state.velocity_world.norm() < 1e-6,
                "vel = {:?}",
                state.velocity_world
            );
            assert!(state.bias_gyro.norm() < 1e-6, "b_g = {:?}", state.bias_gyro);
            assert!(state.bias_acc.norm() < 1e-6, "b_a = {:?}", state.bias_acc);
        }
    }

    #[test]
    fn max_velocity_magnitude_gate_rejects_when_exceeded() {
        // Run a known-converging setup TWICE: once without the gate to
        // record the actual recovered max-velocity magnitude, then once
        // with `max_velocity_magnitude_mps = Some(max / 2.0)` so the
        // post-solve sanity gate must fire. Decoupling the threshold
        // from a hard-coded magnitude keeps the test stable against
        // inner-LM precision drift.
        let mut probe = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            ..MotionBasedViInitializerConfig::default()
        });
        let gravity = probe.config().gravity_world;
        for i in 0..3 {
            probe.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        let mut probe_map = build_constant_velocity_map(3);
        let seed = synthetic_seed();
        let factors = vec![
            no_acceleration_factor(1, 2, 1.0, gravity),
            no_acceleration_factor(2, 3, 1.0, gravity),
        ];
        let probe_result = probe
            .try_initialize(&mut probe_map, &factors, &seed)
            .expect("probe run must converge")
            .clone();
        let max_mag = probe_result
            .keyframe_states
            .values()
            .map(|s| s.velocity_world.norm())
            .fold(0.0_f64, f64::max);
        // The inner LM in this setup converges very near v=0 but not
        // exactly zero (small residual from the gravity term); pick the
        // gate threshold below the actual recovered magnitude so the
        // post-solve check has something to reject.
        assert!(
            max_mag > 1.0e-12,
            "probe max |v| too small to test the gate: {max_mag}"
        );

        let mut gated = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            max_velocity_magnitude_mps: Some(max_mag / 2.0),
            ..MotionBasedViInitializerConfig::default()
        });
        for i in 0..3 {
            gated.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        let mut gated_map = build_constant_velocity_map(3);
        let gated_map_before = gated_map.clone();
        let err = gated
            .try_initialize(&mut gated_map, &factors, &seed)
            .expect_err("velocity gate must reject the LM result");
        assert_eq!(
            gated_map, gated_map_before,
            "a rejected speculative motion-VI solve must not write poses back"
        );
        match err {
            MotionBasedViRejectionReason::VelocityOutOfRange {
                kf_id,
                magnitude_mps,
                limit_mps,
            } => {
                assert!([1u64, 2, 3].contains(&kf_id), "kf_id = {kf_id}");
                assert!(magnitude_mps > limit_mps);
                assert!((limit_mps - max_mag / 2.0).abs() < 1.0e-12);
            }
            other => panic!("expected VelocityOutOfRange, got {other:?}"),
        }
        // The stage stays in `Waiting` after a gate rejection so the
        // caller can re-fire on a future trigger.
        match gated.status() {
            MotionBasedViInitializationStatus::Waiting { last_rejection, .. } => {
                assert!(matches!(
                    last_rejection,
                    Some(MotionBasedViRejectionReason::VelocityOutOfRange { .. })
                ));
            }
            other => panic!("expected Waiting after gate rejection, got {other:?}"),
        }
    }

    #[test]
    fn normalized_imu_nis_gate_rejects_inconsistent_promotion() {
        let mut probe = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            ..MotionBasedViInitializerConfig::default()
        });
        let gravity = probe.config().gravity_world;
        for i in 0..3 {
            probe.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        let factors = vec![
            no_acceleration_factor(1, 2, 1.0, gravity),
            no_acceleration_factor(2, 3, 1.0, gravity),
        ];
        let seed = synthetic_seed();
        let mut probe_map = build_constant_velocity_map(3);
        let observed = probe
            .try_initialize(&mut probe_map, &factors, &seed)
            .expect("probe solve")
            .final_cost_breakdown
            .imu_normalized_squared_residual_per_dof
            .expect("IMU factors produce a normalized residual");
        assert!(observed.is_finite() && observed > 0.0);

        let limit = observed * 0.5;
        let mut gated = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            max_final_imu_nis_per_dof: Some(limit),
            ..MotionBasedViInitializerConfig::default()
        });
        for i in 0..3 {
            gated.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        let mut gated_map = build_constant_velocity_map(3);
        let map_before = gated_map.clone();
        let error = gated
            .try_initialize(&mut gated_map, &factors, &seed)
            .expect_err("NIS gate must reject the inconsistent promotion");
        assert_eq!(gated_map, map_before);
        assert!(matches!(
            error,
            MotionBasedViRejectionReason::ImuNisOutOfRange {
                normalized_nis_per_dof,
                rotation_residual_rms_rad: _,
                velocity_residual_rms_mps: _,
                position_residual_rms_meters: _,
                limit: rejected_limit,
            } if (normalized_nis_per_dof - observed).abs() < 1.0e-9
                && (rejected_limit - limit).abs() < 1.0e-12
        ));
    }

    #[test]
    fn physical_imu_residual_gate_rejects_inconsistent_promotion() {
        let mut probe = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            ..MotionBasedViInitializerConfig::default()
        });
        let gravity = probe.config().gravity_world;
        for i in 0..3 {
            probe.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        let factors = vec![
            no_acceleration_factor(1, 2, 1.0, gravity),
            no_acceleration_factor(2, 3, 1.0, gravity),
        ];
        let seed = synthetic_seed();
        let mut probe_map = build_constant_velocity_map(3);
        let observed = probe
            .try_initialize(&mut probe_map, &factors, &seed)
            .expect("probe solve")
            .final_cost_breakdown
            .imu_velocity_residual_rms_mps
            .expect("velocity residual");
        assert!(observed.is_finite() && observed > 0.0);

        let limit = observed * 0.5;
        let mut gated = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            max_final_imu_velocity_residual_rms_mps: Some(limit),
            ..MotionBasedViInitializerConfig::default()
        });
        for i in 0..3 {
            gated.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        let mut gated_map = build_constant_velocity_map(3);
        let map_before = gated_map.clone();
        let error = gated
            .try_initialize(&mut gated_map, &factors, &seed)
            .expect_err("physical residual gate must reject promotion");
        assert_eq!(gated_map, map_before);
        assert!(matches!(
            error,
            MotionBasedViRejectionReason::ImuRawResidualOutOfRange {
                velocity_residual_rms_mps: Some(value),
                max_velocity_residual_rms_mps: Some(rejected_limit),
                ..
            } if (value - observed).abs() < 1.0e-9
                && (rejected_limit - limit).abs() < 1.0e-12
        ));
    }

    #[test]
    fn max_velocity_magnitude_gate_passes_when_under_limit() {
        // Same setup but with a generous limit; the inner result is
        // promoted to `completed` as usual.
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            max_velocity_magnitude_mps: Some(1.0e6),
            ..MotionBasedViInitializerConfig::default()
        });
        let gravity = init.config().gravity_world;
        for i in 0..3 {
            init.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        let mut map = build_constant_velocity_map(3);
        let seed = synthetic_seed();
        let factors = vec![
            no_acceleration_factor(1, 2, 1.0, gravity),
            no_acceleration_factor(2, 3, 1.0, gravity),
        ];
        let _ = init
            .try_initialize(&mut map, &factors, &seed)
            .expect("generous limit must accept the LM result");
        match init.status() {
            MotionBasedViInitializationStatus::Initialised { .. } => {}
            other => panic!("expected Initialised, got {other:?}"),
        }
    }

    #[test]
    fn gyro_bias_gate_rejects_transactionally() {
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            max_gyro_bias_magnitude_rad_s: Some(0.05),
            ..MotionBasedViInitializerConfig::default()
        });
        let gravity = init.config().gravity_world;
        for i in 0..3 {
            init.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        let mut map = build_constant_velocity_map(3);
        let map_before = map.clone();
        let factors = vec![
            no_acceleration_factor(1, 2, 1.0, gravity),
            no_acceleration_factor(2, 3, 1.0, gravity),
        ];
        let err = init
            .try_initialize_with_bias_seed(
                &mut map,
                &factors,
                Vector3::new(0.1, 0.0, 0.0),
                Vector3::zeros(),
            )
            .expect_err("anchored gyro bias must exceed the configured gate");
        assert_eq!(map, map_before, "rejected solve must not mutate the map");
        assert!(matches!(
            err,
            MotionBasedViRejectionReason::GyroBiasOutOfRange {
                magnitude_rad_s,
                limit_rad_s,
                ..
            } if magnitude_rad_s > limit_rad_s && (limit_rad_s - 0.05).abs() < 1.0e-12
        ));
    }

    #[test]
    fn accel_bias_gate_rejects_transactionally() {
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            max_accel_bias_magnitude_mps2: Some(1.0),
            ..MotionBasedViInitializerConfig::default()
        });
        let gravity = init.config().gravity_world;
        for i in 0..3 {
            init.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        let mut map = build_constant_velocity_map(3);
        let map_before = map.clone();
        let factors = vec![
            no_acceleration_factor(1, 2, 1.0, gravity),
            no_acceleration_factor(2, 3, 1.0, gravity),
        ];
        let err = init
            .try_initialize_with_bias_seed(
                &mut map,
                &factors,
                Vector3::zeros(),
                Vector3::new(2.0, 0.0, 0.0),
            )
            .expect_err("anchored accel bias must exceed the configured gate");
        assert_eq!(map, map_before, "rejected solve must not mutate the map");
        assert!(matches!(
            err,
            MotionBasedViRejectionReason::AccelBiasOutOfRange {
                magnitude_mps2,
                limit_mps2,
                ..
            } if magnitude_mps2 > limit_mps2 && (limit_mps2 - 1.0).abs() < 1.0e-12
        ));
    }

    /// Helper: SO3 type sanity (compile coverage for the test imports).
    #[test]
    fn so3_identity_is_quaternion_identity() {
        let r = SO3::identity();
        let q = r.quaternion();
        assert!((q.w - 1.0).abs() < 1e-12);
        assert!((q.i.abs() + q.j.abs() + q.k.abs()) < 1e-12);
    }

    // ============================================================
    // VIBA2 hand-off tests.
    // ============================================================

    #[test]
    fn viba2_handoff_runs_when_configured_and_reports_scale_history() {
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            gravity_world: Vector3::zeros(),
            viba2: Some(Viba2Config {
                initial_scale: 1.0,
                recover_scale: true,
                max_outer_iterations: 3,
                scale_tolerance: 1.0e-4,
                ..Viba2Config::default()
            }),
            ..MotionBasedViInitializerConfig::default()
        });
        let gravity = init.config().gravity_world;
        for i in 0..3 {
            init.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        let mut map = VisualMap::new();
        let camera = pinhole_camera();
        map.cameras.insert(camera.id, camera.clone());
        for i in 0..3 {
            let center = Vector3::new(i as f64, 0.0, 0.0);
            let pose = pose_at_world_center(center);
            let frame_id = (i as u64) + 1;
            let mut frame = Frame::new(frame_id, camera.id);
            frame.pose = Some(pose);
            let kf = Keyframe {
                frame,
                observations: Vec::new(),
            };
            map.keyframes.insert(frame_id, kf);
        }
        let seed = synthetic_seed();
        let factors = vec![
            no_acceleration_factor(1, 2, 1.0, gravity),
            no_acceleration_factor(2, 3, 1.0, gravity),
        ];
        let result = init
            .try_initialize(&mut map, &factors, &seed)
            .expect("VIBA2 path should succeed");
        // VIBA2 ran at least once; scale history starts with the initial
        // scale and may have one more entry per outer iteration. The
        // exact length depends on whether the kinematic denominator was
        // identifiable, which on this zero-gravity-zero-Δp synthetic is
        // degenerate (so the outer loop bails after one inner solve and
        // freezes scale at 1.0).
        assert!(result.scale.is_finite());
        assert!(result.viba2_iterations_run >= 1);
        assert!(!result.scale_history.is_empty());
        assert!((result.scale_history[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn no_viba2_config_keeps_scale_history_empty() {
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            ..MotionBasedViInitializerConfig::default()
        });
        let gravity = init.config().gravity_world;
        for i in 0..3 {
            init.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        let mut map = VisualMap::new();
        let camera = pinhole_camera();
        map.cameras.insert(camera.id, camera.clone());
        for i in 0..3 {
            let pose = pose_at_world_center(Vector3::new(i as f64, 0.0, 0.0));
            let frame_id = (i as u64) + 1;
            let mut frame = Frame::new(frame_id, camera.id);
            frame.pose = Some(pose);
            let kf = Keyframe {
                frame,
                observations: Vec::new(),
            };
            map.keyframes.insert(frame_id, kf);
        }
        let seed = synthetic_seed();
        let factors = vec![
            no_acceleration_factor(1, 2, 1.0, gravity),
            no_acceleration_factor(2, 3, 1.0, gravity),
        ];
        let result = init
            .try_initialize(&mut map, &factors, &seed)
            .expect("VIBA1 path should succeed");
        assert!((result.scale - 1.0).abs() < 1e-12);
        assert!(result.scale_history.is_empty());
        assert_eq!(result.viba2_iterations_run, 0);
    }

    // ============================================================
    // Staged bias release (`BiasReleaseSchedule`) tests.
    // ============================================================

    #[test]
    fn legacy_path_without_bias_release_schedule_reports_bias_released_true() {
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            ..MotionBasedViInitializerConfig::default()
        });
        assert!(init.config().bias_release.is_none());
        let gravity = init.config().gravity_world;
        for i in 0..3 {
            init.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        let mut map = build_constant_velocity_map(3);
        let seed = synthetic_seed();
        let factors = vec![
            no_acceleration_factor(1, 2, 1.0, gravity),
            no_acceleration_factor(2, 3, 1.0, gravity),
        ];
        let result = init
            .try_initialize(&mut map, &factors, &seed)
            .expect("legacy path must succeed");
        assert!(
            result.bias_released,
            "legacy path (bias_release=None) must report bias_released=true"
        );
        assert!(
            init.velocity_stage_result().is_none(),
            "legacy path never populates the velocity stage"
        );
        match init.status() {
            MotionBasedViInitializationStatus::Initialised { .. } => {}
            other => panic!("expected terminal Initialised, got {other:?}"),
        }
    }

    #[test]
    fn bias_release_schedule_stage_a_fires_then_awaits_release_gate() {
        // Base gates (min_keyframes=3, min_translation_meters=1.0) are
        // satisfied by 3 keyframes spaced 1 m apart, but the schedule's own
        // release gate (min_keyframes=5, min_translation_meters=3.0) is not.
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            bias_release: Some(BiasReleaseSchedule {
                min_keyframes: 5,
                min_translation_meters: 3.0,
            }),
            ..MotionBasedViInitializerConfig::default()
        });
        let gravity = init.config().gravity_world;
        for i in 0..3 {
            init.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        assert!(init.is_ready(), "base gates must be satisfied");
        let mut map = build_constant_velocity_map(3);
        let seed = synthetic_seed();
        let factors = vec![
            no_acceleration_factor(1, 2, 1.0, gravity),
            no_acceleration_factor(2, 3, 1.0, gravity),
        ];

        let stage_a = init
            .try_initialize(&mut map, &factors, &seed)
            .expect("Stage A must fire once the base gates are met")
            .clone();
        assert!(!stage_a.bias_released, "Stage A must not release bias");
        for state in stage_a.keyframe_states.values() {
            assert!(
                (state.bias_gyro - seed.bias_gyro).norm() < 1.0e-12,
                "Stage A must not move gyro bias off the seed"
            );
            assert!(
                (state.bias_acc - seed.bias_acc).norm() < 1.0e-12,
                "Stage A must not move accel bias off the seed"
            );
        }
        assert!(init.velocity_stage_result().is_some());
        match init.status() {
            MotionBasedViInitializationStatus::Waiting {
                velocity_stage_completed,
                ..
            } => assert!(
                velocity_stage_completed,
                "status() must report the completed velocity stage"
            ),
            other => panic!("expected non-terminal Waiting after Stage A, got {other:?}"),
        }

        // A second call before the release gate is met has nothing new to
        // solve: Stage A already fired and is cached.
        let err = init
            .try_initialize(&mut map, &factors, &seed)
            .expect_err("release gate not yet met");
        match err {
            MotionBasedViRejectionReason::AwaitingBiasReleaseExcitation {
                have_keyframes,
                need_keyframes,
                have_translation_meters,
                need_translation_meters,
            } => {
                assert_eq!(have_keyframes, 3);
                assert_eq!(need_keyframes, 5);
                assert!((have_translation_meters - 2.0).abs() < 1.0e-12);
                assert!((need_translation_meters - 3.0).abs() < 1.0e-12);
            }
            other => panic!("expected AwaitingBiasReleaseExcitation, got {other:?}"),
        }
    }

    #[test]
    fn bias_release_schedule_stage_b_fires_once_gate_is_met() {
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            bias_release: Some(BiasReleaseSchedule {
                min_keyframes: 5,
                min_translation_meters: 3.0,
            }),
            ..MotionBasedViInitializerConfig::default()
        });
        let gravity = init.config().gravity_world;
        for i in 0..3 {
            init.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        let mut map = build_constant_velocity_map(5);
        let seed = synthetic_seed();
        let factors = vec![
            no_acceleration_factor(1, 2, 1.0, gravity),
            no_acceleration_factor(2, 3, 1.0, gravity),
            no_acceleration_factor(3, 4, 1.0, gravity),
            no_acceleration_factor(4, 5, 1.0, gravity),
        ];

        let stage_a = init
            .try_initialize(&mut map, &factors, &seed)
            .expect("Stage A must fire")
            .clone();
        assert!(!stage_a.bias_released);

        // Register the remaining keyframes to clear the release gate:
        // 5 keyframes total, 4 m cumulative translation.
        init.register_keyframe(4, Point3::new(3.0, 0.0, 0.0));
        init.register_keyframe(5, Point3::new(4.0, 0.0, 0.0));
        assert_eq!(init.keyframes_observed(), 5);
        assert!((init.cumulative_translation_meters() - 4.0).abs() < 1.0e-9);

        let stage_b = init
            .try_initialize(&mut map, &factors, &seed)
            .expect("Stage B must fire once the release gate is met")
            .clone();
        assert!(
            stage_b.bias_released,
            "Stage B must release bias like the legacy path"
        );
        match init.status() {
            MotionBasedViInitializationStatus::Initialised { .. } => {}
            other => panic!("expected terminal Initialised after Stage B, got {other:?}"),
        }
    }

    #[test]
    fn reset_clears_velocity_stage() {
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            bias_release: Some(BiasReleaseSchedule {
                min_keyframes: 5,
                min_translation_meters: 3.0,
            }),
            ..MotionBasedViInitializerConfig::default()
        });
        let gravity = init.config().gravity_world;
        for i in 0..3 {
            init.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        let mut map = build_constant_velocity_map(3);
        let seed = synthetic_seed();
        let factors = vec![
            no_acceleration_factor(1, 2, 1.0, gravity),
            no_acceleration_factor(2, 3, 1.0, gravity),
        ];
        init.try_initialize(&mut map, &factors, &seed)
            .expect("Stage A must fire");
        assert!(init.velocity_stage_result().is_some());

        init.reset();
        assert!(
            init.velocity_stage_result().is_none(),
            "reset() must clear the cached velocity stage"
        );
        match init.status() {
            MotionBasedViInitializationStatus::Waiting {
                velocity_stage_completed,
                ..
            } => assert!(!velocity_stage_completed),
            other => panic!("expected Waiting after reset, got {other:?}"),
        }
    }

    // ============================================================
    // Gravity-direction recovery (`estimate_gravity`) tests.
    // See `docs/motion_based_vi_alignment.md`'s "Gravity-direction
    // recovery" section for the motivating real-data diagnosis.
    // ============================================================

    /// [`estimate_gravity_and_velocities`] direct unit tests (not routed
    /// through the initializer state machine).
    mod gravity_velocity_alignment {
        use super::*;

        #[test]
        fn recovers_off_axis_gravity_and_velocities_from_a_consistent_window() {
            // True gravity is off every coordinate axis (and off the
            // codebase's Y-axis config default) while the body moves at a
            // constant, also off-axis, world velocity. With visual poses
            // held fixed and zero proper acceleration baked into every
            // factor, the linear system is EXACTLY consistent, so the
            // unconstrained solve must recover both quantities to numerical
            // precision.
            let true_gravity = Vector3::new(6.54, -3.27, 6.54); // |g| = 9.81
            let velocity = Vector3::new(1.2, -0.7, 0.3);
            let map = build_map_at_velocity(5, 1.0, velocity);
            let ids: Vec<u64> = (1..=5).collect();
            let factors = vec![
                no_acceleration_factor(1, 2, 1.0, true_gravity),
                no_acceleration_factor(2, 3, 1.0, true_gravity),
                no_acceleration_factor(3, 4, 1.0, true_gravity),
                no_acceleration_factor(4, 5, 1.0, true_gravity),
            ];
            let alignment = estimate_gravity_and_velocities(
                &map,
                &ids,
                &factors,
                Vector3::zeros(),
                Vector3::zeros(),
                9.81,
            )
            .expect("well-conditioned, exactly-consistent window must align");
            assert!(
                (alignment.gravity_world - true_gravity).norm() < 1.0e-6,
                "g_est = {:?}",
                alignment.gravity_world
            );
            assert!(
                (alignment.raw_gravity_norm - 9.81).abs() < 1.0e-6,
                "raw_gravity_norm = {}",
                alignment.raw_gravity_norm
            );
            for (kf_id, v) in &alignment.velocities {
                assert!(
                    (v - velocity).norm() < 1.0e-6,
                    "kf {kf_id}: v_est = {v:?}, expected {velocity:?}"
                );
            }
        }

        #[test]
        fn degenerate_window_with_a_single_factor_returns_none() {
            // Fewer than 2 usable in-window factors (1 factor = 6 rows <
            // the 6 unknowns already contributed by that one factor's own
            // endpoints, but the function's own documented floor is "< 2
            // factors") is degenerate by construction — distinct from a
            // stationary window, which would instead solve to a near-zero
            // raw gravity norm (also degenerate, via the `< 1e-6` guard).
            let velocity = Vector3::new(1.0, 0.0, 0.0);
            let map = build_map_at_velocity(2, 1.0, velocity);
            let ids: Vec<u64> = vec![1, 2];
            let factors = vec![no_acceleration_factor(
                1,
                2,
                1.0,
                Vector3::new(0.0, 9.81, 0.0),
            )];
            let alignment = estimate_gravity_and_velocities(
                &map,
                &ids,
                &factors,
                Vector3::zeros(),
                Vector3::zeros(),
                9.81,
            );
            assert!(
                alignment.is_none(),
                "a single-factor window must be reported degenerate"
            );
        }

        #[test]
        fn norm_constrained_refinement_pins_the_magnitude_when_raw_solve_drifts() {
            // Start from the exactly-consistent fixture above, then inject
            // a small measurement inconsistency into one factor's velocity
            // delta (as if that window's IMU evidence were noisy). The
            // unconstrained solve is pulled off the true magnitude by the
            // inconsistency, but the VINS-Mono-style tangent refinement
            // must still land the FINAL `gravity_world` exactly on the
            // caller's expected magnitude.
            let true_gravity = Vector3::new(6.54, -3.27, 6.54); // |g| = 9.81
            let velocity = Vector3::new(1.2, -0.7, 0.3);
            let map = build_map_at_velocity(4, 1.0, velocity);
            let ids: Vec<u64> = (1..=4).collect();
            let mut factors = vec![
                no_acceleration_factor(1, 2, 1.0, true_gravity),
                no_acceleration_factor(2, 3, 1.0, true_gravity),
                no_acceleration_factor(3, 4, 1.0, true_gravity),
            ];
            factors[1].delta.delta_velocity += Vector3::new(0.35, -0.2, 0.15);

            let alignment = estimate_gravity_and_velocities(
                &map,
                &ids,
                &factors,
                Vector3::zeros(),
                Vector3::zeros(),
                9.81,
            )
            .expect("a mild inconsistency must still be solvable");
            assert!(
                (alignment.raw_gravity_norm - 9.81).abs() > 1.0e-6,
                "fixture must actually perturb the unconstrained solve off the true magnitude, raw = {}",
                alignment.raw_gravity_norm
            );
            assert!(
                (alignment.gravity_world.norm() - 9.81).abs() < 1.0e-9,
                "the constrained estimate must be pinned to the expected magnitude, got {}",
                alignment.gravity_world.norm()
            );
            assert!(
                alignment.mean_residual_after > 0.0,
                "an inconsistent window must leave a nonzero residual"
            );
        }
    }

    /// Initializer-level (`try_initialize` / `try_initialize_with_bias_seed`)
    /// tests exercising the `estimate_gravity` config knob end to end.
    mod estimate_gravity_initializer {
        use super::*;

        #[test]
        fn recovers_from_rotated_true_gravity_where_the_legacy_path_rejects() {
            // Diagnosed real-data scenario: the specific-force reaction
            // lands on an axis (here Z) 90 degrees away from the config's
            // assumed `gravity_world` (Y), with the body moving along a
            // third axis (X). The legacy path trusts the config axis as
            // truth and the raw-residual gate must reject it; enabling
            // `estimate_gravity` recovers the true direction and clears
            // the SAME gate.
            let true_gravity = Vector3::new(0.0, 0.0, 9.81);
            let assumed_gravity = Vector3::new(0.0, 9.81, 0.0);
            let factors = vec![
                no_acceleration_factor_with_assumed_gravity(
                    1,
                    2,
                    1.0,
                    true_gravity,
                    assumed_gravity,
                ),
                no_acceleration_factor_with_assumed_gravity(
                    2,
                    3,
                    1.0,
                    true_gravity,
                    assumed_gravity,
                ),
                no_acceleration_factor_with_assumed_gravity(
                    3,
                    4,
                    1.0,
                    true_gravity,
                    assumed_gravity,
                ),
                no_acceleration_factor_with_assumed_gravity(
                    4,
                    5,
                    1.0,
                    true_gravity,
                    assumed_gravity,
                ),
            ];
            let seed = synthetic_seed();

            // Probe: legacy solve with NO residual gate, just to read the
            // actual velocity-residual RMS the misaligned assumption
            // produces on this fixture (decouples the gate threshold below
            // from a hard-coded magnitude, same technique as
            // `max_velocity_magnitude_gate_rejects_when_exceeded`).
            let mut probe = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
                min_keyframes: 3,
                min_translation_meters: 1.0,
                gravity_world: assumed_gravity,
                ..MotionBasedViInitializerConfig::default()
            });
            for i in 0..5 {
                probe.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
            }
            let mut probe_map = build_constant_velocity_map(5);
            let probe_residual = probe
                .try_initialize(&mut probe_map, &factors, &seed)
                .expect("legacy solve still converges; only the gate is missing here")
                .final_cost_breakdown
                .imu_velocity_residual_rms_mps
                .expect("IMU factors produce a velocity residual");
            assert!(
                probe_residual > 0.5,
                "a 90-degree gravity misalignment should leave a substantial residual, got {probe_residual}"
            );
            let gate = probe_residual * 0.5;

            // A: legacy path (estimate_gravity = false) must be rejected
            // by the gate.
            let mut legacy = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
                min_keyframes: 3,
                min_translation_meters: 1.0,
                gravity_world: assumed_gravity,
                max_final_imu_velocity_residual_rms_mps: Some(gate),
                ..MotionBasedViInitializerConfig::default()
            });
            for i in 0..5 {
                legacy.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
            }
            let mut legacy_map = build_constant_velocity_map(5);
            let err = legacy
                .try_initialize(&mut legacy_map, &factors, &seed)
                .expect_err(
                    "misaligned gravity assumption must be rejected by the raw-residual gate",
                );
            assert!(
                matches!(
                    err,
                    MotionBasedViRejectionReason::ImuRawResidualOutOfRange { .. }
                ),
                "unexpected rejection: {err:?}"
            );

            // B: estimate_gravity = true recovers the true direction and
            // clears the SAME gate.
            let mut estimated = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
                min_keyframes: 3,
                min_translation_meters: 1.0,
                gravity_world: assumed_gravity,
                estimate_gravity: true,
                max_final_imu_velocity_residual_rms_mps: Some(gate),
                ..MotionBasedViInitializerConfig::default()
            });
            for i in 0..5 {
                estimated.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
            }
            let mut estimated_map = build_constant_velocity_map(5);
            let result = estimated
                .try_initialize(&mut estimated_map, &factors, &seed)
                .expect("gravity estimation must recover enough to clear the same gate");
            let g_est = result
                .estimated_gravity_world
                .expect("estimate_gravity=true must report the recovered vector");
            assert!(
                (g_est - true_gravity).norm() < 0.05,
                "recovered gravity {g_est:?} should match the true {true_gravity:?}"
            );
        }

        #[test]
        fn norm_gate_rejects_an_implausible_recovered_magnitude() {
            // A well-conditioned window recovers a norm near the TRUE
            // magnitude with no constraint imposed anywhere in the
            // unconstrained solve; deliberately feeding it evidence
            // consistent with a magnitude nowhere near 9.81 must trip the
            // observability gate rather than silently rescaling and
            // promoting.
            let true_gravity = Vector3::new(0.0, 0.0, 3.0);
            let assumed_gravity = Vector3::new(0.0, 9.81, 0.0);
            let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
                min_keyframes: 3,
                min_translation_meters: 1.0,
                gravity_world: assumed_gravity,
                estimate_gravity: true,
                ..MotionBasedViInitializerConfig::default()
            });
            for i in 0..3 {
                init.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
            }
            let mut map = build_constant_velocity_map(3);
            let seed = synthetic_seed();
            let factors = vec![
                no_acceleration_factor_with_assumed_gravity(
                    1,
                    2,
                    1.0,
                    true_gravity,
                    assumed_gravity,
                ),
                no_acceleration_factor_with_assumed_gravity(
                    2,
                    3,
                    1.0,
                    true_gravity,
                    assumed_gravity,
                ),
            ];
            let err = init
                .try_initialize(&mut map, &factors, &seed)
                .expect_err("implausible recovered magnitude must be rejected");
            match err {
                MotionBasedViRejectionReason::GravityEstimateOutOfRange {
                    raw_norm_mps2,
                    expected_mps2,
                    max_deviation_ratio,
                } => {
                    assert!(
                        (raw_norm_mps2 - 3.0).abs() < 1.0e-6,
                        "raw_norm_mps2 = {raw_norm_mps2}"
                    );
                    assert!((expected_mps2 - 9.81).abs() < 1.0e-9);
                    assert!((max_deviation_ratio - 0.3).abs() < 1.0e-9);
                }
                other => panic!("expected GravityEstimateOutOfRange, got {other:?}"),
            }
        }

        #[test]
        fn degenerate_when_no_factors_connect_the_window() {
            let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
                min_keyframes: 3,
                min_translation_meters: 1.0,
                estimate_gravity: true,
                ..MotionBasedViInitializerConfig::default()
            });
            for i in 0..3 {
                init.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
            }
            let mut map = build_constant_velocity_map(3);
            let seed = synthetic_seed();
            let err = init
                .try_initialize(&mut map, &[], &seed)
                .expect_err("no factors means the alignment cannot condition at all");
            assert!(matches!(
                err,
                MotionBasedViRejectionReason::GravityEstimateDegenerate
            ));
        }

        #[test]
        fn composes_with_staged_bias_release_stage_a() {
            // `estimate_gravity` and `BiasReleaseSchedule` are independent
            // knobs; a Stage A ("velocity stage") firing must still report
            // + apply the recovered gravity while leaving biases pinned at
            // the seed, exactly as Stage A does for legacy (non-estimated)
            // gravity.
            let true_gravity = Vector3::new(0.0, 0.0, 9.81);
            let assumed_gravity = Vector3::new(0.0, 9.81, 0.0);
            let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
                min_keyframes: 3,
                min_translation_meters: 1.0,
                gravity_world: assumed_gravity,
                estimate_gravity: true,
                bias_release: Some(BiasReleaseSchedule {
                    min_keyframes: 5,
                    min_translation_meters: 3.0,
                }),
                ..MotionBasedViInitializerConfig::default()
            });
            for i in 0..3 {
                init.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
            }
            let mut map = build_constant_velocity_map(3);
            let seed = synthetic_seed();
            let factors = vec![
                no_acceleration_factor_with_assumed_gravity(
                    1,
                    2,
                    1.0,
                    true_gravity,
                    assumed_gravity,
                ),
                no_acceleration_factor_with_assumed_gravity(
                    2,
                    3,
                    1.0,
                    true_gravity,
                    assumed_gravity,
                ),
            ];
            let result = init
                .try_initialize(&mut map, &factors, &seed)
                .expect("Stage A must fire even with gravity estimation enabled")
                .clone();
            assert!(!result.bias_released, "Stage A must not release bias");
            let g_est = result
                .estimated_gravity_world
                .expect("estimate must be reported on a Stage A result too");
            assert!(
                (g_est - true_gravity).norm() < 0.05,
                "recovered gravity {g_est:?} should match the true {true_gravity:?}"
            );
            for state in result.keyframe_states.values() {
                assert!(
                    (state.bias_gyro - seed.bias_gyro).norm() < 1.0e-9,
                    "Stage A must not move gyro bias off the seed even with gravity estimation"
                );
                assert!(
                    (state.bias_acc - seed.bias_acc).norm() < 1.0e-9,
                    "Stage A must not move accel bias off the seed even with gravity estimation"
                );
            }
        }
    }

    // ============================================================
    // Gyro-bias recovery (`estimate_gyro_bias`) tests.
    // See `docs/motion_based_vi_alignment.md`'s "Gyro-bias recovery"
    // section for the motivating real-data diagnosis: the fitted IMU
    // rotation residual RMS sat at 0.014-0.022 rad vs a 0.01 gate,
    // bit-identical with/without gravity estimation (the rotation
    // residual is gravity-independent), consistent with EuRoC-scale gyro
    // bias (~0.08 rad/s) accumulating over each factor's `delta_time`
    // while the bias seed stays fixed at the configured zero.
    // ============================================================

    /// [`estimate_gyro_bias`] direct unit tests (not routed through the
    /// initializer state machine), driving the REAL [`ImuPreintegrator`]
    /// rather than a hand-built delta.
    mod gyro_bias_alignment {
        use super::*;
        use crate::imu_preintegration::ImuPreintegrator;

        /// Build a two-factor window over a translating (`build_constant_velocity_map`),
        /// non-rotating body whose raw gyro stream is a known CONSTANT bias
        /// (no other angular excitation), integrated through the REAL
        /// preintegrator at a zero bias-seed linearisation. The window is
        /// short enough (0.02 s at a ~5e-4 rad/s bias) that the total
        /// rotation angle stays tiny, keeping the first-order bias-Jacobian
        /// correction accurate to well below the tolerance asserted below.
        fn real_integrated_factors_with_constant_gyro_bias(
            bias_gyro_true: Vector3<f64>,
            steps: usize,
            dt: f64,
        ) -> Vec<ImuPreintegrationFactor> {
            [(1u64, 2u64), (2, 3)]
                .into_iter()
                .map(|(from_id, to_id)| {
                    let mut pre =
                        ImuPreintegrator::new_with_bias(Vector3::zeros(), Vector3::zeros());
                    for _ in 0..steps {
                        pre.integrate_sample(bias_gyro_true, Vector3::new(0.0, 9.81, 0.0), dt);
                    }
                    ImuPreintegrationFactor {
                        keyframe_id_from: from_id,
                        keyframe_id_to: to_id,
                        delta: pre.delta(),
                        gravity_world: Vector3::new(0.0, 9.81, 0.0),
                        weight_position: 1.0,
                        weight_velocity: 1.0,
                        weight_rotation: 1.0,
                    }
                })
                .collect()
        }

        #[test]
        fn recovers_a_known_constant_bias_through_the_real_preintegrator() {
            let bias_gyro_true = Vector3::new(0.0005, -0.0003, 0.0002);
            let factors =
                real_integrated_factors_with_constant_gyro_bias(bias_gyro_true, 20, 0.001);
            let map = build_constant_velocity_map(3);
            let ids: Vec<u64> = vec![1, 2, 3];

            let alignment = estimate_gyro_bias(&map, &ids, &factors, Vector3::zeros())
                .expect("well-conditioned window must align");
            assert!(
                (alignment.bias_gyro - bias_gyro_true).norm() < 1.0e-6,
                "b_g_est = {:?}, b_g_true = {:?}",
                alignment.bias_gyro,
                bias_gyro_true
            );
            assert!(
                alignment.rotation_residual_rms_before > 0.0,
                "an uncorrected constant bias must leave a nonzero residual"
            );
            assert!(
                alignment.rotation_residual_rms_after
                    < alignment.rotation_residual_rms_before * 1.0e-3,
                "rms_before={}, rms_after={}",
                alignment.rotation_residual_rms_before,
                alignment.rotation_residual_rms_after
            );
        }

        #[test]
        fn degenerate_window_with_a_single_factor_returns_none() {
            let factors = real_integrated_factors_with_constant_gyro_bias(
                Vector3::new(0.001, 0.0, 0.0),
                20,
                0.001,
            );
            let single_factor = vec![factors[0].clone()];
            let map = build_constant_velocity_map(2);
            let ids: Vec<u64> = vec![1, 2];
            assert!(
                estimate_gyro_bias(&map, &ids, &single_factor, Vector3::zeros()).is_none(),
                "a single-factor window must be reported degenerate"
            );
        }

        #[test]
        fn real_preintegrator_regression_gyro_bias_then_gravity_recovery() {
            // Closes the gap noted in the gravity-alignment tests above (all
            // of which use hand-built factors): drive raw IMU samples
            // carrying BOTH a known constant gyro bias AND an off-axis
            // gravity through the REAL preintegrator, run `estimate_gyro_bias`
            // first, then feed its recovered bias into
            // `estimate_gravity_and_velocities` as the (corrected) bias
            // seed — mirroring the exact order
            // `try_initialize_with_bias_seed` runs them in.
            let true_gravity = Vector3::new(6.54, -3.27, 6.54); // |g| = 9.81
            let velocity = Vector3::new(1.2, -0.7, 0.3);
            let bias_gyro_true = Vector3::new(0.0005, -0.0003, 0.0002);
            let dt_step = 0.001;
            let steps = 20;
            let window = (steps as f64) * dt_step;
            let map = build_map_at_velocity(5, window, velocity);
            let ids: Vec<u64> = (1..=5).collect();
            let mut factors = Vec::new();
            for (from_id, to_id) in [(1u64, 2u64), (2, 3), (3, 4), (4, 5)] {
                let mut pre = ImuPreintegrator::new_with_bias(Vector3::zeros(), Vector3::zeros());
                for _ in 0..steps {
                    pre.integrate_sample(bias_gyro_true, -true_gravity, dt_step);
                }
                factors.push(ImuPreintegrationFactor {
                    keyframe_id_from: from_id,
                    keyframe_id_to: to_id,
                    delta: pre.delta(),
                    gravity_world: true_gravity,
                    weight_position: 1.0,
                    weight_velocity: 1.0,
                    weight_rotation: 1.0,
                });
            }

            let gyro_alignment = estimate_gyro_bias(&map, &ids, &factors, Vector3::zeros())
                .expect("well-conditioned window must align");
            assert!(
                (gyro_alignment.bias_gyro - bias_gyro_true).norm() < 1.0e-6,
                "b_g_est = {:?}",
                gyro_alignment.bias_gyro
            );

            let gravity_alignment = estimate_gravity_and_velocities(
                &map,
                &ids,
                &factors,
                gyro_alignment.bias_gyro,
                Vector3::zeros(),
                9.81,
            )
            .expect("gravity alignment must still condition well after gyro-bias correction");
            assert!(
                (gravity_alignment.gravity_world - true_gravity).norm() < 1.0e-4,
                "g_est = {:?}",
                gravity_alignment.gravity_world
            );
        }
    }

    /// Initializer-level (`try_initialize` / `try_initialize_with_bias_seed`)
    /// tests exercising the `estimate_gyro_bias` config knob end to end.
    mod estimate_gyro_bias_initializer {
        use super::*;

        /// Like [`no_acceleration_factor`] but the STORED delta carries an
        /// uncompensated gyro bias: the naively-integrated (bias-seed-zero)
        /// `ΔR` shows a spurious rotation of `Exp(bias_gyro_true *
        /// delta_t)`, with an EXACT bias Jacobian
        /// `j_rotation_bg = -delta_t·I` (the near-static-window first-order
        /// value — see [`estimate_gyro_bias`]'s doc comment). Since
        /// `Exp(a) ⊗ Exp(-a)` is exactly the identity for any vector `a`,
        /// the rotation-only residual at `bias_gyro = bias_gyro_true` is
        /// EXACT zero regardless of `bias_gyro_true`'s direction, letting
        /// the tests below pin an exactly recovered bias.
        fn no_acceleration_factor_with_gyro_bias(
            from_id: u64,
            to_id: u64,
            delta_t: f64,
            gravity: Vector3<f64>,
            bias_gyro_true: Vector3<f64>,
        ) -> ImuPreintegrationFactor {
            let mut delta = ImuPreintegratedDelta::identity();
            delta.delta_time = delta_t;
            delta.delta_velocity = -gravity * delta_t;
            delta.delta_position = -0.5 * gravity * delta_t * delta_t;
            delta.delta_rotation =
                SO3::from_quaternion(UnitQuaternion::from_scaled_axis(bias_gyro_true * delta_t));
            delta.j_rotation_bg = Matrix3::identity() * (-delta_t);
            ImuPreintegrationFactor {
                keyframe_id_from: from_id,
                keyframe_id_to: to_id,
                delta,
                gravity_world: gravity,
                weight_position: 1.0,
                weight_velocity: 1.0,
                weight_rotation: 1.0,
            }
        }

        /// Combines [`no_acceleration_factor_with_gyro_bias`] with
        /// [`no_acceleration_factor_with_assumed_gravity`]'s TRUE-vs-ASSUMED
        /// gravity decoupling, for the composition test exercising
        /// `estimate_gyro_bias` and `estimate_gravity` together.
        fn no_acceleration_factor_with_assumed_gravity_and_gyro_bias(
            from_id: u64,
            to_id: u64,
            delta_t: f64,
            true_gravity: Vector3<f64>,
            assumed_gravity: Vector3<f64>,
            bias_gyro_true: Vector3<f64>,
        ) -> ImuPreintegrationFactor {
            let mut delta = ImuPreintegratedDelta::identity();
            delta.delta_time = delta_t;
            delta.delta_velocity = -true_gravity * delta_t;
            delta.delta_position = -0.5 * true_gravity * delta_t * delta_t;
            delta.delta_rotation =
                SO3::from_quaternion(UnitQuaternion::from_scaled_axis(bias_gyro_true * delta_t));
            delta.j_rotation_bg = Matrix3::identity() * (-delta_t);
            ImuPreintegrationFactor {
                keyframe_id_from: from_id,
                keyframe_id_to: to_id,
                delta,
                gravity_world: assumed_gravity,
                weight_position: 1.0,
                weight_velocity: 1.0,
                weight_rotation: 1.0,
            }
        }

        #[test]
        fn recovers_from_gyro_bias_where_the_legacy_bias_fixed_path_rejects() {
            // Mirrors the EuRoC diagnosis directly: a `BiasReleaseSchedule`
            // Stage A window (bias fixed at the seed) sees a rotation
            // residual driven ENTIRELY by uncompensated gyro bias —
            // independent of velocity, position, or gravity, since bias is
            // not free to move and rotation residual depends only on the
            // (fixed) visual poses and the (bias-fixed) IMU delta.
            let gravity = Vector3::new(0.0, 9.81, 0.0);
            let bias_gyro_true = Vector3::new(0.05, -0.03, 0.02); // EuRoC-scale, |b| ~ 0.062 rad/s
            let delta_t = 0.15; // matches the diagnosed per-factor Δt
            let factors = vec![
                no_acceleration_factor_with_gyro_bias(1, 2, delta_t, gravity, bias_gyro_true),
                no_acceleration_factor_with_gyro_bias(2, 3, delta_t, gravity, bias_gyro_true),
            ];
            let schedule = BiasReleaseSchedule {
                min_keyframes: 5,
                min_translation_meters: 3.0,
            };
            let seed = synthetic_seed();

            // Probe: bias-fixed Stage A with NO residual gate, just to read
            // the actual rotation-residual RMS this fixture's uncompensated
            // bias leaves (decouples the gate threshold below from a
            // hard-coded magnitude — same technique as
            // `max_velocity_magnitude_gate_rejects_when_exceeded`).
            let mut probe = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
                min_keyframes: 3,
                min_translation_meters: 1.0,
                gravity_world: gravity,
                bias_release: Some(schedule),
                ..MotionBasedViInitializerConfig::default()
            });
            for i in 0..3 {
                probe.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
            }
            let mut probe_map = build_constant_velocity_map(3);
            let probe_residual = probe
                .try_initialize(&mut probe_map, &factors, &seed)
                .expect("bias-fixed Stage A still converges; only the gate is missing here")
                .final_cost_breakdown
                .imu_rotation_residual_rms_rad
                .expect("IMU factors produce a rotation residual");
            assert!(
                probe_residual > 0.0,
                "an uncompensated gyro bias should leave a nonzero rotation residual, got {probe_residual}"
            );
            let gate = probe_residual * 0.5;

            let mut legacy = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
                min_keyframes: 3,
                min_translation_meters: 1.0,
                gravity_world: gravity,
                bias_release: Some(schedule),
                max_final_imu_rotation_residual_rms_rad: Some(gate),
                ..MotionBasedViInitializerConfig::default()
            });
            for i in 0..3 {
                legacy.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
            }
            let mut legacy_map = build_constant_velocity_map(3);
            let err = legacy
                .try_initialize(&mut legacy_map, &factors, &seed)
                .expect_err("bias-fixed Stage A must be rejected by the rotation-residual gate");
            assert!(
                matches!(
                    err,
                    MotionBasedViRejectionReason::ImuRawResidualOutOfRange { .. }
                ),
                "unexpected rejection: {err:?}"
            );

            let mut estimated = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
                min_keyframes: 3,
                min_translation_meters: 1.0,
                gravity_world: gravity,
                bias_release: Some(schedule),
                max_final_imu_rotation_residual_rms_rad: Some(gate),
                estimate_gyro_bias: true,
                ..MotionBasedViInitializerConfig::default()
            });
            for i in 0..3 {
                estimated.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
            }
            let mut estimated_map = build_constant_velocity_map(3);
            let result = estimated
                .try_initialize(&mut estimated_map, &factors, &seed)
                .expect("gyro-bias estimation must clear the same gate")
                .clone();
            assert!(
                !result.bias_released,
                "still Stage A — the schedule's release gate is not met"
            );
            let b_g_est = result
                .estimated_gyro_bias
                .expect("estimate_gyro_bias=true must report the recovered vector");
            assert!(
                (b_g_est - bias_gyro_true).norm() < 1.0e-6,
                "recovered bias {b_g_est:?} should match the true {bias_gyro_true:?}"
            );
            for state in result.keyframe_states.values() {
                assert!(
                    (state.bias_gyro - bias_gyro_true).norm() < 1.0e-9,
                    "Stage A must fix gyro bias at the ESTIMATE, not the raw seed"
                );
                assert!(
                    (state.bias_acc - seed.bias_acc).norm() < 1.0e-9,
                    "Stage A must leave accel bias at the seed"
                );
            }
        }

        #[test]
        fn degenerate_when_no_factors_connect_the_window() {
            let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
                min_keyframes: 3,
                min_translation_meters: 1.0,
                estimate_gyro_bias: true,
                ..MotionBasedViInitializerConfig::default()
            });
            for i in 0..3 {
                init.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
            }
            let mut map = build_constant_velocity_map(3);
            let seed = synthetic_seed();
            let err = init
                .try_initialize(&mut map, &[], &seed)
                .expect_err("no factors means the alignment cannot condition at all");
            assert!(matches!(
                err,
                MotionBasedViRejectionReason::GyroBiasEstimateDegenerate
            ));
        }

        #[test]
        fn composes_with_gravity_estimation_and_staged_bias_release_stage_a() {
            let true_gravity = Vector3::new(0.0, 0.0, 9.81);
            let assumed_gravity = Vector3::new(0.0, 9.81, 0.0);
            let bias_gyro_true = Vector3::new(0.05, -0.03, 0.02);
            let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
                min_keyframes: 3,
                min_translation_meters: 1.0,
                gravity_world: assumed_gravity,
                estimate_gravity: true,
                estimate_gyro_bias: true,
                bias_release: Some(BiasReleaseSchedule {
                    min_keyframes: 5,
                    min_translation_meters: 3.0,
                }),
                ..MotionBasedViInitializerConfig::default()
            });
            for i in 0..3 {
                init.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
            }
            let mut map = build_constant_velocity_map(3);
            let seed = synthetic_seed();
            let factors = vec![
                no_acceleration_factor_with_assumed_gravity_and_gyro_bias(
                    1,
                    2,
                    0.15,
                    true_gravity,
                    assumed_gravity,
                    bias_gyro_true,
                ),
                no_acceleration_factor_with_assumed_gravity_and_gyro_bias(
                    2,
                    3,
                    0.15,
                    true_gravity,
                    assumed_gravity,
                    bias_gyro_true,
                ),
            ];
            let result = init
                .try_initialize(&mut map, &factors, &seed)
                .expect("Stage A must fire with both estimations enabled")
                .clone();
            assert!(!result.bias_released, "Stage A must not release bias");
            let g_est = result
                .estimated_gravity_world
                .expect("gravity estimate must be reported alongside the gyro-bias estimate");
            assert!(
                (g_est - true_gravity).norm() < 0.05,
                "recovered gravity {g_est:?} should match the true {true_gravity:?}"
            );
            let b_g_est = result
                .estimated_gyro_bias
                .expect("gyro-bias estimate must be reported alongside the gravity estimate");
            assert!(
                (b_g_est - bias_gyro_true).norm() < 1.0e-6,
                "recovered bias {b_g_est:?} should match the true {bias_gyro_true:?}"
            );
            for state in result.keyframe_states.values() {
                assert!(
                    (state.bias_gyro - bias_gyro_true).norm() < 1.0e-9,
                    "Stage A must fix gyro bias at the ESTIMATE"
                );
            }
        }
    }
}
