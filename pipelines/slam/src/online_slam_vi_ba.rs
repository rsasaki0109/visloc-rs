//! Sliding-window local VI-BA stage for [`crate::OnlineSlamPipeline`].
//!
//! The `OnlineSlamPipeline` already detects new keyframes (via the local
//! mapper's `AppliedMapUpdate { keyframe_count > 0 }`) and snapshots a
//! Forster `ImuPreintegrationFactor` between every consecutive keyframe
//! pair. The factor used to be a "hint for downstream glue" — this module
//! turns it into an active solver step inside the pipeline.
//!
//! Every `trigger_every` newly-emitted IMU factor (default: every factor),
//! the stage:
//!
//! 1. Slices the trailing `window_size` keyframes out of `map.keyframes`
//!    (ordered by keyframe id).
//! 2. Builds a [`BundleAdjustment`] over those keyframes' poses + the
//!    landmarks they observe at least twice within the window (so isolated
//!    "in only one keyframe" tracks don't contribute Jacobian rows).
//! 3. Attaches every stored [`ImuPreintegrationFactor`] whose `from / to`
//!    keyframe pair both live inside the window, then promotes per-keyframe
//!    `(velocity, bias_gyro, bias_acc)` slots out of the per-pipeline state
//!    table. The first in-window keyframe's pose / velocity / bias are
//!    gauge-fixed; everything else is free.
//! 4. Runs `BundleAdjustment::optimize` with the user-supplied
//!    [`BaConfig`].
//! 5. Optionally rejects writeback when the optimiser fails the configured
//!    cost-ratio or velocity quality gate.
//! 6. Writes the refined poses back into `map.keyframes[*].frame.pose` and
//!    `map.landmarks[*].position`, and writes the refined `(velocity, bias_gyro,
//!    bias_acc)` back into the per-keyframe state table so the next trigger
//!    starts from the new linearisation point.
//!
//! The stage is opt-in. `OnlineSlamConfig.local_vi_ba: Option<…>` is `None`
//! by default, so the IMU-free flows added in earlier tasks pay no
//! per-frame overhead.

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::{DMatrix, DVector, Vector3, Vector6};
use visloc_core::geometry::SE3;
use visloc_core::types::{Camera, VisualMap};

use crate::bundle::{
    relative_observation_confidence_weights, BaConfig, BaCostBreakdown, BaGeneralStereoObservation,
    BaObservation, BaResult, BiasRandomWalkFactor, BundleAdjustment, NavigationStatePrior,
};
use crate::imu_preintegration::ImuPreintegrationFactor;
use crate::marginalization::marginalize;
use crate::LinearSolver;

fn bias_random_walk_information(
    noise_densities: (f64, f64),
    delta_time: f64,
) -> Option<(f64, f64)> {
    let (sigma_gyro, sigma_accel) = noise_densities;
    if !sigma_gyro.is_finite()
        || sigma_gyro <= 0.0
        || !sigma_accel.is_finite()
        || sigma_accel <= 0.0
        || !delta_time.is_finite()
        || delta_time <= 0.0
    {
        return None;
    }
    Some((
        1.0 / (sigma_gyro * sigma_gyro * delta_time),
        1.0 / (sigma_accel * sigma_accel * delta_time),
    ))
}

/// Adaptive refined-velocity writeback gate for [`OnlineSlamLocalBaConfig`].
///
/// The gate builds a per-trigger reference envelope from the current
/// in-window velocity state, pose-delta / IMU-`dt` finite differences,
/// and IMU-predicted next-keyframe velocities, then rejects BA writeback
/// only when the refined max velocity exceeds `quantile(reference) *
/// multiplier + margin_mps`, after lower/upper bounds are applied. This
/// keeps a raw fixed `m/s` threshold available as a safety ceiling while
/// making the main decision relative to the local motion scale.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveVelocityGateConfig {
    /// Robust reference quantile in `[0, 1]`. Values outside the range are
    /// clamped at use-site so ad-hoc experiment configs cannot panic.
    pub reference_quantile: f64,
    /// Multiplier applied to the robust reference velocity.
    pub multiplier: f64,
    /// Additive slack in m/s after the multiplier.
    pub margin_mps: f64,
    /// Minimum threshold in m/s, after multiplier/slack.
    pub min_threshold_mps: f64,
    /// Optional hard upper bound in m/s. Use the legacy fixed velocity
    /// gate for an unconditional ceiling; this field keeps the adaptive
    /// threshold itself bounded for experiments.
    pub max_threshold_mps: Option<f64>,
    /// Minimum number of finite reference velocities required before the
    /// adaptive gate is active. If fewer samples are available, the gate
    /// reports `None` and does not reject.
    pub min_reference_count: usize,
}

impl Default for AdaptiveVelocityGateConfig {
    fn default() -> Self {
        Self {
            reference_quantile: 0.8,
            multiplier: 2.5,
            margin_mps: 1.0,
            min_threshold_mps: 3.0,
            max_threshold_mps: None,
            min_reference_count: 2,
        }
    }
}

/// Configuration for the sliding-window local VI-BA stage.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineSlamLocalBaConfig {
    /// Run the BA every `trigger_every` newly-emitted IMU factors. `1`
    /// triggers on every new keyframe; larger values batch a few factors
    /// per solve. A value of `0` is rejected at construction.
    pub trigger_every: usize,
    /// Trailing window depth (keyframe count). The window holds the most
    /// recent `window_size` keyframes (or all of them, when fewer have
    /// been registered). Should be at least `2`. Common KITTI-rate value
    /// is `5..=10`.
    pub window_size: usize,
    /// Minimum number of in-window observations a landmark must have for
    /// it to contribute Jacobian rows. `2` (default) excludes
    /// "seen once in the window" landmarks that would only add a single
    /// 2-row block per BA iteration without disambiguating geometry.
    pub min_observations_per_landmark: usize,
    /// Use calibrated right-camera measurements when the map carries a
    /// matching stereo observation. This anchors metric structure inside the
    /// same tight pose/landmark/IMU solve rather than treating a stereo map as
    /// monocular after bootstrap.
    pub use_general_stereo_observations: bool,
    /// Optional right-image reprojection gate applied before admitting an
    /// existing stereo observation into the local solve.
    pub general_stereo_max_initial_right_reprojection_error_px: Option<f64>,
    /// Consume matcher confidence preserved on map observations. Set false to
    /// run the identical window with uniform visual weights for an A/B
    /// tracking-cliff ablation.
    pub use_observation_confidence_weights: bool,
    /// Inner [`BaConfig`] passed straight through to
    /// [`BundleAdjustment::optimize`]. Default uses
    /// [`LinearSolver::Sparse`] to keep the solve cheap on larger windows.
    pub ba_config: BaConfig,
    /// World-frame gravity vector used to seed bias slot initial values.
    /// Mirrors the parent `OnlineSlamImuConfig.gravity_world` — kept here
    /// so the stage is self-contained and unit tests can drive it without
    /// constructing the full pipeline.
    pub gravity_world: Vector3<f64>,
    /// Rigid transform from the tracked camera/sensor frame into the IMU body
    /// frame (`T_b<-c`, EuRoC `T_BS`). Identity is valid only for co-located,
    /// axis-aligned camera and IMU frames.
    pub body_to_camera: SE3,
    /// Initial bias-gyro linearisation point for newly-promoted keyframes.
    pub bias_gyro_init: Vector3<f64>,
    /// Initial bias-acc linearisation point for newly-promoted keyframes.
    pub bias_acc_init: Vector3<f64>,
    /// Optional `(gyro, accel)` information weights for adjacent bias
    /// random-walk factors. Use `1 / (sigma_random_walk² * dt)` only when
    /// the other IMU residual blocks use compatible covariance weighting.
    /// `None` preserves the diagnostic legacy behavior; promoted tight VI
    /// fusion must provide a physically calibrated prior.
    pub bias_random_walk_weights: Option<(f64, f64)>,
    /// Optional continuous-time gyro/accelerometer bias random-walk noise
    /// densities. When set, each adjacent factor receives information
    /// `1 / (sigma² * delta_time)`, so irregular keyframe spacing is handled
    /// physically. Takes precedence over `bias_random_walk_weights`.
    pub bias_random_walk_noise_densities: Option<(f64, f64)>,
    /// Conditioning fallback: when the first BA pass has
    /// `final_cost / initial_cost > threshold`, the cost surface is too
    /// far from quadratic for the joint pose + velocity + bias solve to
    /// be trusted. The stage re-runs the BA with **all per-keyframe
    /// biases gauge-frozen** at their pre-BA linearisation points and
    /// uses that result for writeback. `None` (default) disables the
    /// fallback — legacy behaviour. A reasonable starting value is
    /// `Some(0.9)`: the BA must reduce cost by at least 10 % per
    /// trigger or the bias updates are discarded.
    pub freeze_biases_when_cost_ratio_above: Option<f64>,
    /// Writeback quality gate: when the selected BA result has
    /// `final_cost / initial_cost > threshold`, the trigger returns
    /// diagnostics but does **not** write refined poses, landmarks,
    /// velocities, or biases back to the map/state. This is stricter
    /// than [`Self::freeze_biases_when_cost_ratio_above`]: the freeze
    /// fallback still accepts pose/velocity/landmark updates from a
    /// re-solve, while this gate treats the whole local VI-BA result as
    /// untrusted.
    ///
    /// `None` (default) preserves legacy behaviour. Start conservative
    /// in experiments, for example `Some(1.0)` to reject passes that do
    /// not reduce cost at all, or lower values when bad IMU factors are
    /// known to destabilize tracking.
    pub reject_writeback_when_cost_ratio_above: Option<f64>,
    /// Reject writeback when the selected joint solve's covariance-whitened
    /// IMU NIS per residual DoF remains above this limit.
    pub reject_writeback_when_final_imu_nis_per_dof_above: Option<f64>,
    /// Writeback velocity sanity gate: when any refined in-window
    /// `||velocity_world||` exceeds this threshold, the trigger returns
    /// diagnostics but skips all map/state writeback. This catches the
    /// common tight-VIO failure mode where the visual residuals can be
    /// fit only by pushing keyframe velocities to non-physical values.
    ///
    /// `None` (default) preserves legacy behaviour. EuRoC indoor runs
    /// can start with a conservative `Some(10.0)` and then tighten once
    /// dataset-specific velocity envelopes are measured.
    pub reject_writeback_when_velocity_norm_above_mps: Option<f64>,
    /// Reject the whole local solve when any keyframe camera centre moves by
    /// more than this many metres relative to the pre-solve map.
    pub reject_writeback_when_pose_translation_above_meters: Option<f64>,
    /// Reject the whole local solve when any keyframe rotation changes by
    /// more than this many radians relative to the pre-solve map.
    pub reject_writeback_when_pose_rotation_above_radians: Option<f64>,
    /// Physical bias-magnitude writeback gate: reject the whole local solve
    /// when any in-window keyframe's refined `||bias_gyro||` exceeds this
    /// bound (rad/s).
    ///
    /// Real-data motivation (`gt_full1500`, full MH_01, 1500 frames, SP
    /// frontend, local-VI-BA NIS gate 50): local VI-BA ran 440 triggers
    /// with 135 writebacks mirrored into the IMU motion model, and the
    /// mirrored accel bias diverged to `[1.047, -8.401, 1.298]`
    /// (`‖·‖ ≈ 8.6 m/s²` — near gravity magnitude). Every one of those
    /// writebacks passed the NIS / velocity / pose-correction gates; none
    /// of them checks whether the solution is a physically plausible IMU
    /// bias versus the visual residuals being absorbed into the bias
    /// slots. This mirrors the physical gate the motion-VI init stage
    /// already applies during bootstrap
    /// (`MotionBasedViInitializerConfig::max_gyro_bias_magnitude_rad_s`,
    /// bench value `0.2`) — local VI-BA had no equivalent for its own
    /// per-trigger bias updates.
    ///
    /// `None` (default) preserves legacy behaviour (no physical bias cap).
    /// A reasonable starting bench value is `Some(0.2)`, mirroring the
    /// motion-VI init bootstrap gate.
    pub reject_gyro_bias_above_rad_s: Option<f64>,
    /// Physical bias-magnitude writeback gate: reject the whole local solve
    /// when any in-window keyframe's refined `||bias_acc||` exceeds this
    /// bound (m/s²).
    ///
    /// See [`Self::reject_gyro_bias_above_rad_s`] for the `gt_full1500`
    /// evidence motivating this gate (mirrored accel bias diverged to
    /// `‖·‖ ≈ 8.6 m/s²`). Mirrors
    /// `MotionBasedViInitializerConfig::max_accel_bias_magnitude_mps2`,
    /// bench value `1.0`.
    ///
    /// `None` (default) preserves legacy behaviour (no physical bias cap).
    /// A reasonable starting bench value is `Some(1.0)`, mirroring the
    /// motion-VI init bootstrap gate.
    pub reject_accel_bias_above_mps2: Option<f64>,
    /// Adaptive writeback velocity gate. Unlike
    /// [`Self::reject_writeback_when_velocity_norm_above_mps`], this is
    /// not a raw scene-scale `m/s` threshold. It derives a per-trigger
    /// threshold from the current local-window motion envelope and rejects
    /// only when the refined velocity is large relative to that envelope.
    ///
    /// `None` (default) preserves legacy behaviour. Keep the fixed gate
    /// available as a safety ceiling for A/B runs; use this gate for the
    /// primary "is this solve locally plausible?" decision.
    pub adaptive_velocity_gate: Option<AdaptiveVelocityGateConfig>,
    /// Threshold-gated IMU factor re-linearisation. When `Some((g, a))`,
    /// before each BA pass the stage walks `state.factor_history` and
    /// re-bakes any factor whose stored `bias_*_linearisation` differs
    /// from the current per-keyframe bias estimate (of the factor's
    /// `keyframe_id_from`) by more than `g` (rad/s, gyro) or `a` (m/s²,
    /// accel). The first-order bias correction is applied in-place via
    /// [`crate::imu_preintegration::ImuPreintegratedDelta::relinearise_at`],
    /// resetting the linearisation point to the new bias so subsequent
    /// `residual_with_bias_correction` stays in the small-`δb` regime.
    ///
    /// Default `None` preserves Phase-13 behaviour: factors stay at
    /// their construction-time linearisation (typically a placeholder
    /// zero pre-VI-init), and the BA's `residual_with_bias_correction`
    /// pays the full first-order extrapolation cost. Enable
    /// `Some((0.01, 0.1))` or similar when running with
    /// `keep_pre_promotion_imu_factors = true` so banked pre-promotion
    /// factors get refreshed to the post-promotion bias estimate before
    /// they're solved against — Phase-14 fix for the mirror-velocity
    /// divergence that appears when banked stale-bias factors are
    /// optimised at non-trivial post-promotion biases.
    pub relinearise_imu_factor_bias_thresholds: Option<(f64, f64)>,
    /// When `true`, run a sliding-window BA pass immediately at the
    /// moment VI-init promotes — without waiting for the next post-
    /// promotion IMU factor to arrive. Phase-16 lever. Currently the
    /// `maybe_run_local_vi_ba` gate requires a freshly-staged
    /// `ImuPreintegrationFactor` to enter the function, so banked
    /// pre-promotion factors (with `keep_pre_promotion_imu_factors =
    /// true`) sit idle for one keyframe-cadence after promotion before
    /// the next factor unlocks them. Setting this flag inserts a
    /// "promotion-time" BA trigger so the banked factors are consumed
    /// the same frame VI-init transitions out of `is_active()`. Pairs
    /// naturally with `keep_pre_promotion_imu_factors` + the
    /// `relinearise_*` knob: banked factors are re-linearised at the
    /// just-promoted bias point and immediately solved, instead of
    /// waiting for the next keyframe registration to trigger the
    /// chain.
    ///
    /// Default `false` preserves Phase-13 / Phase-14 behaviour. Useful
    /// when the visual tracker is fragile post-promotion (next KF
    /// arrives late or never), so the promotion event itself is the
    /// only reliable trigger for a refinement pass.
    pub run_at_vi_init_promotion: bool,
    /// Carry the inertial Markov blanket across window shifts by
    /// marginalizing the outgoing 15-DoF navigation state into a dense FEJ
    /// prior on its successor. This replaces repeatedly fixing the newest
    /// window anchor with infinite confidence.
    pub marginalize_navigation_state: bool,
    /// Optional `(velocity, gyro-bias, accel-bias)` 1-sigma uncertainty for
    /// the sequence's first navigation state. With marginalization enabled,
    /// this avoids treating static-init velocity/bias estimates as infinitely
    /// certain. The global pose gauge remains fixed exactly.
    pub initial_navigation_prior_std_devs: Option<(f64, f64, f64)>,
}

impl Default for OnlineSlamLocalBaConfig {
    fn default() -> Self {
        Self {
            trigger_every: 1,
            window_size: 5,
            min_observations_per_landmark: 2,
            use_general_stereo_observations: true,
            general_stereo_max_initial_right_reprojection_error_px: Some(5.0),
            use_observation_confidence_weights: false,
            ba_config: BaConfig {
                linear_solver: LinearSolver::Sparse,
                max_iterations: 10,
                ..BaConfig::default()
            },
            gravity_world: Vector3::new(0.0, 9.81, 0.0),
            body_to_camera: SE3::identity(),
            bias_gyro_init: Vector3::zeros(),
            bias_acc_init: Vector3::zeros(),
            bias_random_walk_weights: None,
            bias_random_walk_noise_densities: None,
            freeze_biases_when_cost_ratio_above: None,
            reject_writeback_when_cost_ratio_above: None,
            reject_writeback_when_final_imu_nis_per_dof_above: None,
            reject_writeback_when_velocity_norm_above_mps: None,
            reject_writeback_when_pose_translation_above_meters: None,
            reject_writeback_when_pose_rotation_above_radians: None,
            reject_gyro_bias_above_rad_s: None,
            reject_accel_bias_above_mps2: None,
            adaptive_velocity_gate: None,
            relinearise_imu_factor_bias_thresholds: None,
            run_at_vi_init_promotion: false,
            marginalize_navigation_state: false,
            initial_navigation_prior_std_devs: None,
        }
    }
}

/// Per-keyframe `(velocity, bias)` state carried across BA invocations.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyframeImuState {
    pub velocity_world: Vector3<f64>,
    pub bias_gyro: Vector3<f64>,
    pub bias_acc: Vector3<f64>,
}

/// Running state of the local VI-BA stage. Owns the per-keyframe IMU
/// state table plus a small history of recent IMU factors (last
/// `2 * window_size` so the stage can always rebuild the connecting
/// factors for the trailing window).
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineSlamLocalBaState {
    pub config: OnlineSlamLocalBaConfig,
    /// Per-keyframe `(velocity, bias)` indexed by keyframe id.
    pub keyframe_state: BTreeMap<u64, KeyframeImuState>,
    /// Rolling history of recently emitted IMU factors. Kept long enough
    /// to cover the trailing window even if the trigger cadence skips a
    /// few factors. Capped at `4 * window_size` entries to bound memory.
    pub factor_history: Vec<ImuPreintegrationFactor>,
    /// Number of new factors observed since the last successful trigger;
    /// rolled forward by [`Self::register_new_factor`].
    pub pending_factors_since_last_trigger: usize,
    /// Dense prior on the first state of the next window.
    pub navigation_prior: Option<NavigationStatePrior>,
}

impl OnlineSlamLocalBaState {
    pub fn new(config: OnlineSlamLocalBaConfig) -> Self {
        Self {
            config,
            keyframe_state: BTreeMap::new(),
            factor_history: Vec::new(),
            pending_factors_since_last_trigger: 0,
            navigation_prior: None,
        }
    }

    /// Reset the state for a new sequence (mirrors
    /// [`crate::OnlineSlamPipeline::reset_sequence_state`]).
    pub fn reset(&mut self) {
        self.keyframe_state.clear();
        self.factor_history.clear();
        self.pending_factors_since_last_trigger = 0;
        self.navigation_prior = None;
    }

    /// Append a freshly-staged IMU factor to the rolling history and bump
    /// the trigger counter. Returns `true` when the trigger threshold has
    /// been reached, signalling the caller should run a BA pass.
    pub fn register_new_factor(&mut self, factor: ImuPreintegrationFactor) -> bool {
        let cap = (4 * self.config.window_size).max(self.config.window_size);
        self.factor_history.push(factor);
        if self.factor_history.len() > cap {
            let overflow = self.factor_history.len() - cap;
            self.factor_history.drain(..overflow);
        }
        self.pending_factors_since_last_trigger += 1;
        self.pending_factors_since_last_trigger >= self.config.trigger_every
    }
}

/// Per-trigger BA outcome exposed on [`crate::OnlineSlamResult`].
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineSlamLocalBaStats {
    /// Sorted keyframe ids included in this trigger's window.
    pub window_keyframe_ids: Vec<u64>,
    /// Number of landmarks that contributed observations to the solve
    /// (after the `min_observations_per_landmark` filter).
    pub landmark_count: usize,
    /// Number of reprojection observations fed into the solve.
    pub observation_count: usize,
    /// Subset of `observation_count` represented by calibrated 4-row stereo
    /// residuals instead of 2-row left-only residuals.
    pub stereo_observation_count: usize,
    /// Number of IMU factors whose `from / to` pair fell inside the
    /// window.
    pub imu_factor_count: usize,
    /// Optimiser outcome; carries the LM trace + final / initial cost.
    /// When [`Self::bias_frozen`] is `true` this is the SECOND-pass
    /// result (the one used for writeback); the first-pass trace is
    /// dropped on the floor. When [`Self::quality_gate_rejected`] is
    /// `true`, this result was NOT written back.
    pub ba_result: BaResult,
    /// Residual-family costs at the selected solve's input and output
    /// states. These remain available when transactional writeback is
    /// rejected, which makes visual/inertial conflicts diagnosable.
    pub initial_cost_breakdown: BaCostBreakdown,
    pub final_cost_breakdown: BaCostBreakdown,
    /// `final_cost / initial_cost` for [`Self::ba_result`]. Uses `0.0`
    /// when `initial_cost == 0.0` because there was no residual energy
    /// to reduce.
    pub cost_ratio: f64,
    /// Maximum refined in-window `||velocity_world||` in m/s from the
    /// selected BA result. This is reported even when writeback is
    /// rejected so runners can tune the velocity sanity gate.
    pub max_refined_velocity_norm_mps: f64,
    pub max_pose_translation_correction_meters: f64,
    pub max_pose_rotation_correction_radians: f64,
    /// Maximum refined in-window `||bias_gyro||` in rad/s from the selected
    /// BA result. Reported even when writeback is rejected, mirroring
    /// [`Self::max_refined_velocity_norm_mps`].
    pub max_refined_gyro_bias_norm_rad_s: f64,
    /// Maximum refined in-window `||bias_acc||` in m/s² from the selected
    /// BA result. Reported even when writeback is rejected. See the
    /// `gt_full1500` evidence on
    /// [`OnlineSlamLocalBaConfig::reject_accel_bias_above_mps2`]: the
    /// unfiltered mirrored value reached `‖·‖ ≈ 8.6`.
    pub max_refined_accel_bias_norm_mps2: f64,
    /// Adaptive velocity threshold used for this trigger, if the
    /// adaptive gate was enabled and enough finite local reference
    /// velocities were available.
    pub adaptive_velocity_gate_threshold_mps: Option<f64>,
    /// `true` when the conditioning fallback in
    /// [`OnlineSlamLocalBaConfig::freeze_biases_when_cost_ratio_above`]
    /// fired this trigger. The window was re-solved with biases gauge-
    /// frozen at their pre-BA linearisation points, and the BA's bias
    /// updates were NOT written back into the state table.
    pub bias_frozen: bool,
    /// `true` when
    /// [`OnlineSlamLocalBaConfig::reject_writeback_when_cost_ratio_above`]
    /// rejected the selected BA result. The returned diagnostics describe
    /// the rejected solve, but map poses, landmarks, velocities, and
    /// biases were left unchanged.
    pub quality_gate_rejected: bool,
    /// `true` when the cost-ratio part of the writeback quality gate
    /// rejected this trigger.
    pub cost_ratio_gate_rejected: bool,
    pub imu_nis_gate_rejected: bool,
    /// `true` when the refined-velocity part of the writeback quality
    /// gate rejected this trigger.
    pub velocity_gate_rejected: bool,
    pub pose_correction_gate_rejected: bool,
    /// `true` when either
    /// [`OnlineSlamLocalBaConfig::reject_gyro_bias_above_rad_s`] or
    /// [`OnlineSlamLocalBaConfig::reject_accel_bias_above_mps2`] rejected
    /// this trigger because a refined in-window bias magnitude exceeded
    /// its configured bound.
    pub bias_magnitude_gate_rejected: bool,
    /// `true` when the adaptive refined-velocity gate rejected this
    /// trigger.
    pub adaptive_velocity_gate_rejected: bool,
    /// Number of IMU factors whose stored `bias_*_linearisation` was
    /// refreshed by the threshold-gated re-linearisation pass driven by
    /// [`OnlineSlamLocalBaConfig::relinearise_imu_factor_bias_thresholds`].
    /// Always `0` when that config is `None`. Counts every factor walked
    /// in this trigger, not just the in-window subset — refreshing
    /// `state.factor_history` whole-vector ensures future windows
    /// inherit the up-to-date linearisation point.
    pub relinearised_factor_count: usize,
    /// Whether a carried navigation prior constrained this solve.
    pub marginalization_prior_applied: bool,
    /// Whether this trigger produced the prior for the next shifted window.
    pub marginalization_succeeded: bool,
}

fn ba_cost_ratio(result: &BaResult) -> f64 {
    if result.initial_cost > 0.0 {
        result.final_cost / result.initial_cost
    } else {
        0.0
    }
}

fn compute_max_refined_velocity_norm_mps(ba: &BundleAdjustment, window_ids: &[u64]) -> f64 {
    window_ids
        .iter()
        .filter_map(|kf_id| ba.velocities.get(kf_id))
        .map(|velocity| velocity.norm())
        .fold(0.0_f64, f64::max)
}

/// Maximum in-window refined `(||bias_gyro||, ||bias_acc||)` from the
/// selected BA result. Used both for diagnostics and by
/// [`OnlineSlamLocalBaConfig::reject_gyro_bias_above_rad_s`] /
/// [`OnlineSlamLocalBaConfig::reject_accel_bias_above_mps2`].
fn compute_max_refined_bias_norms(ba: &BundleAdjustment, window_ids: &[u64]) -> (f64, f64) {
    let mut max_gyro = 0.0_f64;
    let mut max_accel = 0.0_f64;
    for kf_id in window_ids {
        let Some(bias) = ba.biases.get(kf_id) else {
            continue;
        };
        let gyro_norm = Vector3::new(bias[0], bias[1], bias[2]).norm();
        let accel_norm = Vector3::new(bias[3], bias[4], bias[5]).norm();
        if gyro_norm.is_finite() {
            max_gyro = max_gyro.max(gyro_norm);
        }
        if accel_norm.is_finite() {
            max_accel = max_accel.max(accel_norm);
        }
    }
    (max_gyro, max_accel)
}

fn max_pose_correction(map: &VisualMap, ba: &BundleAdjustment, window_ids: &[u64]) -> (f64, f64) {
    let mut max_translation = 0.0_f64;
    let mut max_rotation = 0.0_f64;
    for id in window_ids {
        let (Some(before), Some(after)) = (
            map.keyframes
                .get(id)
                .and_then(|keyframe| keyframe.frame.pose.as_ref()),
            ba.poses.get(id),
        ) else {
            continue;
        };
        max_translation = max_translation
            .max((after.camera_center_world() - before.camera_center_world()).norm());
        let rotation_delta =
            before.world_to_camera.rotation.inverse() * after.world_to_camera.rotation;
        max_rotation = max_rotation.max(rotation_delta.angle());
    }
    (max_translation, max_rotation)
}

fn robust_quantile(values: &mut Vec<f64>, quantile: f64) -> Option<f64> {
    values.retain(|value| value.is_finite() && *value >= 0.0);
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let q = quantile.clamp(0.0, 1.0);
    let idx = ((values.len() - 1) as f64 * q).ceil() as usize;
    values.get(idx).copied()
}

fn pose_delta_velocity_norms_for_window(
    map: &VisualMap,
    factors: &[ImuPreintegrationFactor],
    body_to_camera: &SE3,
) -> Vec<f64> {
    let mut norms = Vec::new();
    for factor in factors {
        let dt = factor.delta.delta_time;
        if dt <= 0.0 || !dt.is_finite() {
            continue;
        }
        let Some(prev_center) = map
            .keyframes
            .get(&factor.keyframe_id_from)
            .and_then(|kf| kf.frame.pose.as_ref())
            .map(|pose| {
                body_to_camera
                    .compose(&pose.world_to_camera)
                    .inverse()
                    .translation
            })
        else {
            continue;
        };
        let Some(curr_center) = map
            .keyframes
            .get(&factor.keyframe_id_to)
            .and_then(|kf| kf.frame.pose.as_ref())
            .map(|pose| {
                body_to_camera
                    .compose(&pose.world_to_camera)
                    .inverse()
                    .translation
            })
        else {
            continue;
        };
        let velocity_norm = ((curr_center - prev_center) / dt).norm();
        if velocity_norm.is_finite() {
            norms.push(velocity_norm);
        }
    }
    norms
}

fn imu_predicted_velocity_norms_for_window(
    map: &VisualMap,
    state: &OnlineSlamLocalBaState,
    factors: &[ImuPreintegrationFactor],
) -> Vec<f64> {
    let mut norms = Vec::new();
    for factor in factors {
        let dt = factor.delta.delta_time;
        if dt <= 0.0 || !dt.is_finite() {
            continue;
        }
        let Some(from_state) = state.keyframe_state.get(&factor.keyframe_id_from) else {
            continue;
        };
        let Some(rotation_body_to_world) = map
            .keyframes
            .get(&factor.keyframe_id_from)
            .and_then(|kf| kf.frame.pose.as_ref())
            .map(|pose| {
                state
                    .config
                    .body_to_camera
                    .compose(&pose.world_to_camera)
                    .rotation
                    .inverse()
            })
        else {
            continue;
        };
        let predicted_velocity = from_state.velocity_world
            + factor.gravity_world * dt
            + rotation_body_to_world * factor.delta.delta_velocity;
        let norm = predicted_velocity.norm();
        if norm.is_finite() {
            norms.push(norm);
        }
    }
    norms
}

fn compute_adaptive_velocity_gate_threshold_mps(
    map: &VisualMap,
    state: &OnlineSlamLocalBaState,
    window_ids: &[u64],
    factors: &[ImuPreintegrationFactor],
    config: &AdaptiveVelocityGateConfig,
) -> Option<f64> {
    let mut reference_norms: Vec<f64> = window_ids
        .iter()
        .filter_map(|kf_id| state.keyframe_state.get(kf_id))
        .map(|slot| slot.velocity_world.norm())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .collect();
    reference_norms.extend(pose_delta_velocity_norms_for_window(
        map,
        factors,
        &state.config.body_to_camera,
    ));
    reference_norms.extend(imu_predicted_velocity_norms_for_window(map, state, factors));
    if reference_norms.len() < config.min_reference_count.max(1) {
        return None;
    }

    let reference = robust_quantile(&mut reference_norms, config.reference_quantile)?;
    let multiplier = config.multiplier.max(0.0);
    let margin = config.margin_mps.max(0.0);
    let min_threshold = config.min_threshold_mps.max(0.0);
    let mut threshold = reference * multiplier + margin;
    if !threshold.is_finite() {
        return None;
    }
    threshold = threshold.max(min_threshold);
    if let Some(max_threshold) = config.max_threshold_mps {
        if max_threshold.is_finite() && max_threshold >= 0.0 {
            threshold = threshold.min(max_threshold);
        }
    }
    Some(threshold)
}

fn next_navigation_prior(
    map: &VisualMap,
    state: &OnlineSlamLocalBaState,
    window_ids: &[u64],
    factors: &[ImuPreintegrationFactor],
) -> Option<NavigationStatePrior> {
    let (&old_id, &next_id) = (window_ids.first()?, window_ids.get(1)?);
    let boundary = factors
        .iter()
        .find(|factor| factor.keyframe_id_from == old_id && factor.keyframe_id_to == next_id)?;
    let camera_id = map.keyframes.get(&old_id)?.frame.camera_id;
    let camera = map.cameras.get(&camera_id)?.clone();
    let mut boundary_problem = BundleAdjustment::new(camera);
    boundary_problem.set_imu_body_to_camera(state.config.body_to_camera.clone());

    for id in [old_id, next_id] {
        let pose = map.keyframes.get(&id)?.frame.pose.clone()?;
        let nav = state.keyframe_state.get(&id)?;
        boundary_problem.add_pose(id, pose);
        boundary_problem.add_velocity(id, nav.velocity_world);
        boundary_problem.add_bias(
            id,
            Vector6::new(
                nav.bias_gyro.x,
                nav.bias_gyro.y,
                nav.bias_gyro.z,
                nav.bias_acc.x,
                nav.bias_acc.y,
                nav.bias_acc.z,
            ),
        );
    }

    let carried_prior = state
        .navigation_prior
        .as_ref()
        .filter(|prior| prior.keyframe_ids.as_slice() == [old_id]);
    if let Some(prior) = carried_prior {
        boundary_problem.set_navigation_state_prior(prior.clone());
    } else {
        // The sequence's first state defines the global gauge. The first
        // pose is exact, but velocity/static-init biases should retain their
        // finite initialization uncertainty when calibrated standard
        // deviations are available.
        boundary_problem.fix_pose(old_id);
        if let Some((sigma_velocity, sigma_gyro_bias, sigma_accel_bias)) =
            state.config.initial_navigation_prior_std_devs
        {
            if !sigma_velocity.is_finite()
                || sigma_velocity <= 0.0
                || !sigma_gyro_bias.is_finite()
                || sigma_gyro_bias <= 0.0
                || !sigma_accel_bias.is_finite()
                || sigma_accel_bias <= 0.0
            {
                return None;
            }
            let mut information = DMatrix::zeros(15, 15);
            for component in 6..9 {
                information[(component, component)] = 1.0 / sigma_velocity.powi(2);
            }
            for component in 9..12 {
                information[(component, component)] = 1.0 / sigma_gyro_bias.powi(2);
            }
            for component in 12..15 {
                information[(component, component)] = 1.0 / sigma_accel_bias.powi(2);
            }
            let old_nav = state.keyframe_state.get(&old_id)?;
            let old_pose = map.keyframes.get(&old_id)?.frame.pose.clone()?;
            boundary_problem.set_navigation_state_prior(NavigationStatePrior {
                keyframe_ids: vec![old_id],
                reference_poses: BTreeMap::from([(old_id, old_pose)]),
                reference_velocities: BTreeMap::from([(old_id, old_nav.velocity_world)]),
                reference_biases: BTreeMap::from([(
                    old_id,
                    Vector6::new(
                        old_nav.bias_gyro.x,
                        old_nav.bias_gyro.y,
                        old_nav.bias_gyro.z,
                        old_nav.bias_acc.x,
                        old_nav.bias_acc.y,
                        old_nav.bias_acc.z,
                    ),
                )]),
                information,
                gradient: DVector::zeros(15),
                constant_cost: 0.0,
            });
        } else {
            boundary_problem.fix_velocity(old_id);
            boundary_problem.fix_bias(old_id);
        }
    }
    boundary_problem.add_imu_factor(boundary.clone());
    if state.config.bias_random_walk_noise_densities.is_some()
        || state.config.bias_random_walk_weights.is_some()
    {
        let (weight_gyro, weight_accel) =
            if let Some(noise) = state.config.bias_random_walk_noise_densities {
                bias_random_walk_information(noise, boundary.delta.delta_time)?
            } else {
                state.config.bias_random_walk_weights?
            };
        boundary_problem.add_bias_random_walk_factor(BiasRandomWalkFactor {
            keyframe_id_from: old_id,
            keyframe_id_to: next_id,
            weight_gyro,
            weight_accel,
        });
    } else {
        // Without a bias-evolution model the successor's six bias DoFs have
        // no information and the marginal is necessarily singular.
        return None;
    }

    let linearized = boundary_problem.linearized_navigation_system()?;
    let pose_offset = 0usize;
    let velocity_offset = linearized.pose_ids.len() * 6;
    let bias_offset = velocity_offset + linearized.velocity_ids.len() * 3;
    let pose_slot = linearized.pose_ids.iter().position(|id| *id == next_id)?;
    let velocity_slot = linearized
        .velocity_ids
        .iter()
        .position(|id| *id == next_id)?;
    let bias_slot = linearized.bias_ids.iter().position(|id| *id == next_id)?;
    let mut keep = Vec::with_capacity(15);
    keep.extend((0..6).map(|component| pose_offset + pose_slot * 6 + component));
    keep.extend((0..3).map(|component| velocity_offset + velocity_slot * 3 + component));
    keep.extend((0..6).map(|component| bias_offset + bias_slot * 6 + component));

    // Normal equations use gradient g = J^T r, whereas information form uses
    // eta = -g for a local mean dx = -H^-1 g.
    let eta = -linearized.gradient;
    let (information, eta) = marginalize(&linearized.information, &eta, &keep)?;
    let gradient = -eta;
    let constant_cost = information
        .clone()
        .cholesky()
        .map(|chol| gradient.dot(&chol.solve(&gradient)))
        .unwrap_or(0.0);
    let nav = state.keyframe_state.get(&next_id)?;
    let pose = map.keyframes.get(&next_id)?.frame.pose.clone()?;
    Some(NavigationStatePrior {
        keyframe_ids: vec![next_id],
        reference_poses: BTreeMap::from([(next_id, pose)]),
        reference_velocities: BTreeMap::from([(next_id, nav.velocity_world)]),
        reference_biases: BTreeMap::from([(
            next_id,
            Vector6::new(
                nav.bias_gyro.x,
                nav.bias_gyro.y,
                nav.bias_gyro.z,
                nav.bias_acc.x,
                nav.bias_acc.y,
                nav.bias_acc.z,
            ),
        )]),
        information: 0.5 * (&information + information.transpose()),
        gradient,
        constant_cost,
    })
}

/// Run one local VI-BA trigger over `map`'s trailing window of keyframes.
///
/// Returns `None` when the window is too short to support BA (`< 2`
/// keyframes, or fewer than `2` in-window IMU factors) or when no usable
/// camera could be looked up in the map. On a successful trigger the
/// optimiser's refined poses + landmarks are written back into `map`, and
/// the refined `(velocity, bias)` state is written back into
/// `state.keyframe_state`.
///
/// `state.pending_factors_since_last_trigger` is reset to `0` regardless
/// of success — the caller has already decided to fire the trigger; a
/// short-window skip is not a reason to fire again immediately.
pub fn run_local_vi_ba(
    map: &mut VisualMap,
    state: &mut OnlineSlamLocalBaState,
) -> Option<OnlineSlamLocalBaStats> {
    state.pending_factors_since_last_trigger = 0;

    let window_size = state.config.window_size.max(2);
    let mut keyframe_ids: Vec<u64> = map.keyframes.keys().copied().collect();
    keyframe_ids.sort_unstable();
    if keyframe_ids.len() < 2 {
        return None;
    }
    let window_start = keyframe_ids.len().saturating_sub(window_size);
    let window_ids: Vec<u64> = keyframe_ids[window_start..].to_vec();
    let in_window: BTreeSet<u64> = window_ids.iter().copied().collect();

    // Collect IMU factors that fully sit inside the window.
    let in_window_factors: Vec<ImuPreintegrationFactor> = state
        .factor_history
        .iter()
        .filter(|f| {
            in_window.contains(&f.keyframe_id_from) && in_window.contains(&f.keyframe_id_to)
        })
        .cloned()
        .collect();
    if in_window_factors.is_empty() {
        return None;
    }

    // Pull a representative camera from the window's first keyframe. The
    // BA path uses a single shared camera; multi-camera bundles are out of
    // scope for this stage.
    let first_kf = map.keyframes.get(&window_ids[0])?;
    let camera: Camera = map.cameras.get(&first_kf.frame.camera_id)?.clone();

    // Promote per-keyframe IMU state for any window keyframe that doesn't
    // yet have one. New keyframes seed velocity from the inter-keyframe
    // pose-centre delta divided by the IMU factor's `delta_time` (a clean
    // first guess when GT velocity is unknown); biases start at the
    // configured linearisation point.
    for window_idx in 0..window_ids.len() {
        let kf_id = window_ids[window_idx];
        if state.keyframe_state.contains_key(&kf_id) {
            continue;
        }
        let seed_velocity = seed_velocity_for(
            map,
            &window_ids,
            window_idx,
            &in_window_factors,
            &state.config.body_to_camera,
        )
        .unwrap_or_else(Vector3::zeros);
        state.keyframe_state.insert(
            kf_id,
            KeyframeImuState {
                velocity_world: seed_velocity,
                bias_gyro: state.config.bias_gyro_init,
                bias_acc: state.config.bias_acc_init,
            },
        );
    }

    let anchor_id = window_ids[0];

    // Threshold-gated factor re-linearisation. Walks `state.factor_history`
    // (the source of truth) and re-bakes any factor whose stored
    // `bias_*_linearisation` is far from the current per-keyframe bias
    // estimate of its `keyframe_id_from`. Mutates in-place so future BA
    // windows inherit the refreshed delta. Then re-collects the in-window
    // factor slice from the now-up-to-date history.
    let mut relinearised_factor_count = 0usize;
    if let Some((gyro_thresh, accel_thresh)) = state.config.relinearise_imu_factor_bias_thresholds {
        for factor in state.factor_history.iter_mut() {
            let Some(per_kf) = state.keyframe_state.get(&factor.keyframe_id_from) else {
                continue;
            };
            let gyro_drift = (per_kf.bias_gyro - factor.delta.bias_gyro_linearisation).norm();
            let accel_drift = (per_kf.bias_acc - factor.delta.bias_acc_linearisation).norm();
            if gyro_drift > gyro_thresh || accel_drift > accel_thresh {
                factor
                    .delta
                    .relinearise_at(&per_kf.bias_gyro, &per_kf.bias_acc);
                relinearised_factor_count += 1;
            }
        }
    }
    let in_window_factors: Vec<ImuPreintegrationFactor> = if relinearised_factor_count > 0 {
        state
            .factor_history
            .iter()
            .filter(|f| {
                in_window.contains(&f.keyframe_id_from) && in_window.contains(&f.keyframe_id_to)
            })
            .cloned()
            .collect()
    } else {
        in_window_factors
    };

    // Count in-window observations per landmark; only landmarks that meet
    // the `min_observations_per_landmark` threshold contribute to the
    // solve. This protects the linear system from rank-deficient single-
    // view tracks dragged in by the local mapper.
    let mut landmark_obs_counts: BTreeMap<u64, usize> = BTreeMap::new();
    for kf_id in &window_ids {
        let kf = map.keyframes.get(kf_id)?;
        for obs in &kf.observations {
            *landmark_obs_counts.entry(obs.landmark_id).or_insert(0) += 1;
        }
    }
    let active_landmarks: BTreeSet<u64> = landmark_obs_counts
        .iter()
        .filter(|(_, count)| **count >= state.config.min_observations_per_landmark)
        .map(|(id, _)| *id)
        .collect();

    let marginalization_prior_applied = state.config.marginalize_navigation_state
        && state
            .navigation_prior
            .as_ref()
            .is_some_and(|prior| prior.keyframe_ids.as_slice() == [anchor_id]);

    let build_ba =
        |freeze_all_biases: bool| -> Option<(BundleAdjustment, usize, usize, Option<Vec<f64>>)> {
            let mut ba = BundleAdjustment::new(camera.clone());
            ba.set_imu_body_to_camera(state.config.body_to_camera.clone());
            if marginalization_prior_applied {
                ba.set_navigation_state_prior(state.navigation_prior.clone()?);
            }
            for kf_id in &window_ids {
                let kf = map.keyframes.get(kf_id)?;
                let pose = kf.frame.pose.clone()?;
                ba.add_pose(*kf_id, pose);
                if *kf_id == anchor_id && !marginalization_prior_applied {
                    ba.fix_pose(*kf_id);
                }
            }
            for kf_id in &window_ids {
                if let Some(per_kf) = state.keyframe_state.get(kf_id) {
                    ba.add_velocity(*kf_id, per_kf.velocity_world);
                    let bias = Vector6::new(
                        per_kf.bias_gyro.x,
                        per_kf.bias_gyro.y,
                        per_kf.bias_gyro.z,
                        per_kf.bias_acc.x,
                        per_kf.bias_acc.y,
                        per_kf.bias_acc.z,
                    );
                    ba.add_bias(*kf_id, bias);
                    if *kf_id == anchor_id && !marginalization_prior_applied {
                        ba.fix_velocity(*kf_id);
                        ba.fix_bias(*kf_id);
                    } else if freeze_all_biases {
                        ba.fix_bias(*kf_id);
                    }
                }
            }
            if state.config.bias_random_walk_noise_densities.is_some()
                || state.config.bias_random_walk_weights.is_some()
            {
                for factor in &in_window_factors {
                    let (weight_gyro, weight_accel) = if let Some((sigma_gyro, sigma_accel)) =
                        state.config.bias_random_walk_noise_densities
                    {
                        bias_random_walk_information(
                            (sigma_gyro, sigma_accel),
                            factor.delta.delta_time,
                        )?
                    } else {
                        state
                            .config
                            .bias_random_walk_weights
                            .expect("fixed weights were checked above")
                    };
                    ba.add_bias_random_walk_factor(BiasRandomWalkFactor {
                        keyframe_id_from: factor.keyframe_id_from,
                        keyframe_id_to: factor.keyframe_id_to,
                        weight_gyro,
                        weight_accel,
                    });
                }
            }
            let mut observation_count = 0usize;
            let mut stereo_observation_count = 0usize;
            let mut mono_confidences = Vec::new();
            let mut general_stereo_confidences = Vec::new();
            for landmark_id in &active_landmarks {
                let Some(landmark) = map.landmarks.get(landmark_id) else {
                    continue;
                };
                ba.add_landmark(*landmark_id, landmark.position);
                for kf_id in &window_ids {
                    let kf = map.keyframes.get(kf_id)?;
                    let pose = ba.poses.get(kf_id)?.clone();
                    for obs in &kf.observations {
                        if obs.landmark_id == *landmark_id {
                            let stereo = state
                                .config
                                .use_general_stereo_observations
                                .then(|| {
                                    map.stereo_observations
                                        .iter()
                                        .find(|stereo| {
                                            stereo.frame_id == *kf_id
                                                && stereo.landmark_id == *landmark_id
                                        })
                                        .and_then(|stereo| {
                                            map.cameras
                                                .get(&stereo.right_camera_id)
                                                .map(|right_camera| (stereo, right_camera))
                                        })
                                        .filter(|(stereo, right_camera)| {
                                            let Some(max_error) = state
                                            .config
                                            .general_stereo_max_initial_right_reprojection_error_px
                                        else {
                                            return true;
                                        };
                                            if !max_error.is_finite() || max_error <= 0.0 {
                                                return false;
                                            }
                                            let point_left =
                                                pose.transform_world_point(&landmark.position);
                                            right_camera
                                                .project(
                                                    &stereo
                                                        .left_to_right
                                                        .transform_point(&point_left),
                                                )
                                                .is_some_and(|predicted| {
                                                    (predicted - stereo.xy_right).norm()
                                                        <= max_error
                                                })
                                        })
                                })
                                .flatten();
                            if let Some((stereo, right_camera)) = stereo {
                                ba.add_general_stereo_observation(BaGeneralStereoObservation {
                                    keyframe_id: *kf_id,
                                    landmark_id: *landmark_id,
                                    xy_left: obs.xy,
                                    xy_right: stereo.xy_right,
                                    right_camera: right_camera.clone(),
                                    left_to_right: stereo.left_to_right.clone(),
                                });
                                let confidence = state
                                    .config
                                    .use_observation_confidence_weights
                                    .then(|| map.observation_confidence(obs))
                                    .flatten();
                                general_stereo_confidences.push(confidence);
                                stereo_observation_count += 1;
                            } else {
                                ba.add_observation(BaObservation {
                                    keyframe_id: *kf_id,
                                    landmark_id: *landmark_id,
                                    xy: obs.xy,
                                });
                                let confidence = state
                                    .config
                                    .use_observation_confidence_weights
                                    .then(|| map.observation_confidence(obs))
                                    .flatten();
                                mono_confidences.push(confidence);
                            }
                            observation_count += 1;
                        }
                    }
                }
            }
            for factor in &in_window_factors {
                ba.add_imu_factor(factor.clone());
            }
            let observation_weights = relative_observation_confidence_weights(
                mono_confidences
                    .into_iter()
                    .chain(general_stereo_confidences),
            );
            Some((
                ba,
                observation_count,
                stereo_observation_count,
                observation_weights,
            ))
        };

    let imu_factor_count = in_window_factors.len();
    let (mut ba, observation_count, stereo_observation_count, observation_weights) =
        build_ba(false)?;
    let cost_breakdown = |problem: &BundleAdjustment, weights: Option<&[f64]>| {
        weights.map_or_else(
            || Some(problem.cost_breakdown(&state.config.ba_config.robust_kernel)),
            |weights| {
                problem
                    .cost_breakdown_with_observation_weights(
                        &state.config.ba_config.robust_kernel,
                        weights,
                    )
                    .ok()
            },
        )
    };
    let mut initial_cost_breakdown = cost_breakdown(&ba, observation_weights.as_deref())?;
    let weighted_result = match observation_weights.as_deref() {
        Some(weights) => ba.optimize_with_observation_weights(&state.config.ba_config, weights),
        None => ba.optimize(&state.config.ba_config),
    };
    let mut ba_result = match weighted_result {
        Ok(result) => result,
        Err(_) => return None,
    };
    let mut final_cost_breakdown = cost_breakdown(&ba, observation_weights.as_deref())?;

    // Conditioning fallback: if the joint pose + velocity + bias solve
    // failed to bring the cost down enough, the bias updates are not
    // trustworthy (the BA found a "best fit" against drifted upstream
    // poses by abusing the bias slots). Re-solve with all biases gauge-
    // frozen so the LM can only update poses + velocities + landmarks.
    let mut bias_frozen = false;
    if let Some(threshold) = state.config.freeze_biases_when_cost_ratio_above {
        let ratio = ba_cost_ratio(&ba_result);
        if ratio > threshold {
            let (mut ba_frozen, _, _, frozen_observation_weights) = build_ba(true)?;
            let frozen_initial_cost_breakdown =
                cost_breakdown(&ba_frozen, frozen_observation_weights.as_deref())?;
            let frozen_result = match frozen_observation_weights.as_deref() {
                Some(weights) => {
                    ba_frozen.optimize_with_observation_weights(&state.config.ba_config, weights)
                }
                None => ba_frozen.optimize(&state.config.ba_config),
            };
            match frozen_result {
                Ok(result) => {
                    let frozen_final_cost_breakdown =
                        cost_breakdown(&ba_frozen, frozen_observation_weights.as_deref())?;
                    ba = ba_frozen;
                    ba_result = result;
                    initial_cost_breakdown = frozen_initial_cost_breakdown;
                    final_cost_breakdown = frozen_final_cost_breakdown;
                    bias_frozen = true;
                }
                Err(_) => {
                    // Fall back to the first-pass result if the frozen
                    // re-solve itself fails. The caller still sees
                    // `bias_frozen = false` so they know which trace they're
                    // reading.
                }
            }
        }
    }

    let cost_ratio = ba_cost_ratio(&ba_result);
    let max_refined_velocity_norm_mps = compute_max_refined_velocity_norm_mps(&ba, &window_ids);
    let (max_pose_translation_correction_meters, max_pose_rotation_correction_radians) =
        max_pose_correction(map, &ba, &window_ids);
    let (max_refined_gyro_bias_norm_rad_s, max_refined_accel_bias_norm_mps2) =
        compute_max_refined_bias_norms(&ba, &window_ids);
    let adaptive_velocity_gate_threshold_mps = state
        .config
        .adaptive_velocity_gate
        .as_ref()
        .and_then(|config| {
            compute_adaptive_velocity_gate_threshold_mps(
                map,
                state,
                &window_ids,
                &in_window_factors,
                config,
            )
        });
    let cost_ratio_gate_rejected = state
        .config
        .reject_writeback_when_cost_ratio_above
        .is_some_and(|threshold| cost_ratio > threshold);
    let imu_nis_gate_rejected = state
        .config
        .reject_writeback_when_final_imu_nis_per_dof_above
        .is_some_and(|threshold| {
            final_cost_breakdown
                .imu_normalized_squared_residual_per_dof
                .is_none_or(|value| !value.is_finite() || value > threshold)
        });
    let velocity_gate_rejected = state
        .config
        .reject_writeback_when_velocity_norm_above_mps
        .is_some_and(|threshold| max_refined_velocity_norm_mps > threshold);
    let pose_correction_gate_rejected = state
        .config
        .reject_writeback_when_pose_translation_above_meters
        .is_some_and(|threshold| max_pose_translation_correction_meters > threshold)
        || state
            .config
            .reject_writeback_when_pose_rotation_above_radians
            .is_some_and(|threshold| max_pose_rotation_correction_radians > threshold);
    let bias_magnitude_gate_rejected = state
        .config
        .reject_gyro_bias_above_rad_s
        .is_some_and(|threshold| max_refined_gyro_bias_norm_rad_s > threshold)
        || state
            .config
            .reject_accel_bias_above_mps2
            .is_some_and(|threshold| max_refined_accel_bias_norm_mps2 > threshold);
    let adaptive_velocity_gate_rejected = adaptive_velocity_gate_threshold_mps
        .is_some_and(|threshold| max_refined_velocity_norm_mps > threshold);
    let quality_gate_rejected = cost_ratio_gate_rejected
        || imu_nis_gate_rejected
        || velocity_gate_rejected
        || pose_correction_gate_rejected
        || bias_magnitude_gate_rejected
        || adaptive_velocity_gate_rejected;
    if quality_gate_rejected {
        return Some(OnlineSlamLocalBaStats {
            window_keyframe_ids: window_ids,
            landmark_count: active_landmarks.len(),
            observation_count,
            stereo_observation_count,
            imu_factor_count,
            ba_result,
            initial_cost_breakdown,
            final_cost_breakdown,
            cost_ratio,
            max_refined_velocity_norm_mps,
            max_pose_translation_correction_meters,
            max_pose_rotation_correction_radians,
            max_refined_gyro_bias_norm_rad_s,
            max_refined_accel_bias_norm_mps2,
            adaptive_velocity_gate_threshold_mps,
            bias_frozen,
            quality_gate_rejected,
            cost_ratio_gate_rejected,
            imu_nis_gate_rejected,
            velocity_gate_rejected,
            pose_correction_gate_rejected,
            bias_magnitude_gate_rejected,
            adaptive_velocity_gate_rejected,
            relinearised_factor_count,
            marginalization_prior_applied,
            marginalization_succeeded: false,
        });
    }

    // Write refined poses + landmarks back to the map.
    for kf_id in &window_ids {
        if let Some(refined_pose) = ba.poses.get(kf_id) {
            if let Some(kf) = map.keyframes.get_mut(kf_id) {
                kf.frame.pose = Some(refined_pose.clone());
            }
        }
    }
    for (landmark_id, refined_pos) in &ba.landmarks {
        if let Some(landmark) = map.landmarks.get_mut(landmark_id) {
            landmark.position = *refined_pos;
        }
    }
    // Write refined velocities back. Bias writeback is suppressed when
    // the conditioning fallback fired: the frozen re-solve leaves the
    // bias variables at their input values, so writing them back would
    // be a no-op anyway — but we make the intent explicit.
    for kf_id in &window_ids {
        let Some(slot) = state.keyframe_state.get_mut(kf_id) else {
            continue;
        };
        if let Some(velocity) = ba.velocities.get(kf_id) {
            slot.velocity_world = *velocity;
        }
        if !bias_frozen {
            if let Some(bias) = ba.biases.get(kf_id) {
                slot.bias_gyro = Vector3::new(bias[0], bias[1], bias[2]);
                slot.bias_acc = Vector3::new(bias[3], bias[4], bias[5]);
            }
        }
    }

    let marginalization_succeeded = if state.config.marginalize_navigation_state {
        let next_prior = next_navigation_prior(map, state, &window_ids, &in_window_factors);
        let succeeded = next_prior.is_some();
        state.navigation_prior = next_prior;
        succeeded
    } else {
        state.navigation_prior = None;
        false
    };

    Some(OnlineSlamLocalBaStats {
        window_keyframe_ids: window_ids,
        landmark_count: active_landmarks.len(),
        observation_count,
        stereo_observation_count,
        imu_factor_count,
        ba_result,
        initial_cost_breakdown,
        final_cost_breakdown,
        cost_ratio,
        max_refined_velocity_norm_mps,
        max_pose_translation_correction_meters,
        max_pose_rotation_correction_radians,
        max_refined_gyro_bias_norm_rad_s,
        max_refined_accel_bias_norm_mps2,
        adaptive_velocity_gate_threshold_mps,
        bias_frozen,
        quality_gate_rejected,
        cost_ratio_gate_rejected,
        imu_nis_gate_rejected,
        velocity_gate_rejected,
        pose_correction_gate_rejected,
        bias_magnitude_gate_rejected,
        adaptive_velocity_gate_rejected,
        relinearised_factor_count,
        marginalization_prior_applied,
        marginalization_succeeded,
    })
}

/// Per-trigger inertial-only BA outcome, returned by
/// [`run_inertial_only_vi_ba`] / [`run_inertial_only_vi_ba_with_options`].
/// Mirrors [`OnlineSlamLocalBaStats`] but drops the landmark / observation
/// counters (always zero for the inertial-only path) and replaces them with
/// the keyframe count.
#[derive(Debug, Clone, PartialEq)]
pub struct InertialOnlyViBaStats {
    /// Sorted keyframe ids included in this solve.
    pub keyframe_ids: Vec<u64>,
    /// Number of IMU factors whose `from / to` pair fell inside the
    /// keyframe set.
    pub imu_factor_count: usize,
    /// Refined per-keyframe `(velocity, bias_gyro, bias_acc)` after the
    /// solve. Indexed by keyframe id.
    pub keyframe_states: BTreeMap<u64, KeyframeImuState>,
    /// Optimiser outcome; carries the LM trace + final / initial cost.
    pub ba_result: crate::bundle::BaResult,
    pub initial_cost_breakdown: BaCostBreakdown,
    pub final_cost_breakdown: BaCostBreakdown,
}

/// Run an inertial-only sliding-window MAP solve over `keyframe_ids`'s
/// preintegration factors. Mirrors the VIBA1 stage in ORB-SLAM3:
/// **landmarks are not touched** (no visual residuals contribute), and
/// only inertial states are optimised. The vision-only keyframe trajectory is
/// held fixed, as required by Campos–Montiel–Tardós' disjoint inertial-only
/// MAP formulation; allowing poses to move would remove the visual reference
/// that makes gyro bias observable.
/// Scale is fixed at `1.0` for this entry-point — the monocular scale
/// recovery sits in the future VIBA2 stage. Used by
/// [`crate::MotionBasedViInitializer`] to refine biases + poses once
/// the body has moved enough to give the IMU translational excitation.
///
/// Bias is assumed constant over the short initialization window. The generic
/// BA core represents one bias slot per keyframe, so adjacent slots are tied
/// with a high-information equality factor. Unlike local VI-BA, no bias or
/// velocity slot is pinned: fixed visual poses remove the trajectory gauge and
/// the initializer must be able to estimate the common bias itself.
///
/// Returns `None` when:
/// * `keyframe_ids` has fewer than 2 entries,
/// * no IMU factor fully sits inside the keyframe set,
/// * any keyframe is missing its pose, or
/// * the LM solver itself fails (singular system, etc.).
///
/// Thin wrapper over [`run_inertial_only_vi_ba_with_options`] with
/// `solve_bias = true` — this entry point's public signature/behaviour is
/// unchanged; see the sibling function for the bias-fixed "velocity stage"
/// mode used by [`crate::vi_motion_initializer::BiasReleaseSchedule`].
pub fn run_inertial_only_vi_ba(
    map: &mut VisualMap,
    keyframe_ids: &[u64],
    factors: &[ImuPreintegrationFactor],
    initial_states: &BTreeMap<u64, KeyframeImuState>,
    ba_config: &crate::bundle::BaConfig,
) -> Option<InertialOnlyViBaStats> {
    run_inertial_only_vi_ba_with_options(
        map,
        keyframe_ids,
        factors,
        initial_states,
        ba_config,
        true,
    )
}

/// [`run_inertial_only_vi_ba`] with an explicit `solve_bias` switch.
///
/// `solve_bias = true` reproduces `run_inertial_only_vi_ba` exactly (the
/// public entry point is now a thin wrapper around this function). `solve_bias
/// = false` is the "velocity stage" half of
/// [`crate::vi_motion_initializer::BiasReleaseSchedule`]'s staged bias
/// release: every keyframe's bias slot is added at the caller-supplied seed
/// and then immediately gauge-fixed via [`BundleAdjustment::fix_bias`], so the
/// LM solve can only move per-keyframe velocities (and, indirectly, the
/// residual's rotation/position terms through the fixed poses). The
/// [`BiasRandomWalkFactor`] equality ties that keep adjacent bias slots
/// coherent are skipped in this mode: they exist to prevent per-keyframe bias
/// drift on a *free* bias, which is meaningless when every slot is already
/// pinned to the same fixed value.
///
/// Motivation (see `docs/motion_based_vi_alignment.md`): firing the VIBA1
/// solve with bias free from the very first keyframe, over a short
/// (~3-keyframe) window, on a moving-start EuRoC sequence let the recovered
/// accelerometer bias diverge to magnitudes around 12 m/s² — the window
/// simply does not carry enough independent excitation to identify bias
/// safely yet. Holding bias fixed and only refining velocity is the
/// well-conditioned partial solve for that regime; the caller graduates to
/// `solve_bias = true` once more excitation has accumulated.
///
/// The returned [`InertialOnlyViBaStats::keyframe_states`] carries the seed
/// biases back out **unchanged** (down to solver floating-point noise) when
/// `solve_bias = false`, with refined velocities.
pub fn run_inertial_only_vi_ba_with_options(
    map: &mut VisualMap,
    keyframe_ids: &[u64],
    factors: &[ImuPreintegrationFactor],
    initial_states: &BTreeMap<u64, KeyframeImuState>,
    ba_config: &crate::bundle::BaConfig,
    solve_bias: bool,
) -> Option<InertialOnlyViBaStats> {
    if keyframe_ids.len() < 2 {
        return None;
    }
    let mut sorted_ids: Vec<u64> = keyframe_ids.to_vec();
    sorted_ids.sort_unstable();
    sorted_ids.dedup();
    let in_window: BTreeSet<u64> = sorted_ids.iter().copied().collect();

    let in_window_factors: Vec<ImuPreintegrationFactor> = factors
        .iter()
        .filter(|f| {
            in_window.contains(&f.keyframe_id_from) && in_window.contains(&f.keyframe_id_to)
        })
        .cloned()
        .collect();
    if in_window_factors.is_empty() {
        return None;
    }

    // Pull the camera from the first keyframe; required by
    // BundleAdjustment::new but the inertial-only path never reads its
    // intrinsics (no visual residuals contribute).
    let first_kf = map.keyframes.get(&sorted_ids[0])?;
    let camera: Camera = map.cameras.get(&first_kf.frame.camera_id)?.clone();

    let mut ba = BundleAdjustment::new(camera);

    for kf_id in &sorted_ids {
        let kf = map.keyframes.get(kf_id)?;
        let pose = kf.frame.pose.clone()?;
        ba.add_pose(*kf_id, pose);
        ba.fix_pose(*kf_id);
    }
    // Velocity + bias slots per keyframe. Seed from `initial_states` when
    // available; otherwise zero velocity and zero bias.
    for kf_id in &sorted_ids {
        let (velocity, bias_g, bias_a) = match initial_states.get(kf_id) {
            Some(state) => (state.velocity_world, state.bias_gyro, state.bias_acc),
            None => (Vector3::zeros(), Vector3::zeros(), Vector3::zeros()),
        };
        ba.add_velocity(*kf_id, velocity);
        let bias = Vector6::new(bias_g.x, bias_g.y, bias_g.z, bias_a.x, bias_a.y, bias_a.z);
        ba.add_bias(*kf_id, bias);
        if !solve_bias {
            // Velocity stage: gauge-fix every bias slot at the seed value.
            // The LM solve can then only move velocities (and, through the
            // fixed poses, the IMU residual's rotation/position terms),
            // which is exactly the well-conditioned partial solve this mode
            // exists for.
            ba.fix_bias(*kf_id);
        }
    }
    // The inertial-only initialization paper estimates one shared gyro/accel
    // bias over the short window. The generic BA owns one slot per keyframe,
    // so tie them relative to the strongest IMU information in this problem.
    // A fixed 1e8 penalty was not invariant to covariance whitening: EuRoC
    // factors can carry 1e10..1e12 information, making that supposedly "hard"
    // equality weak enough for 0.1 rad/s inter-keyframe bias drift.
    //
    // Skipped entirely when `solve_bias == false`: every bias slot is already
    // gauge-fixed at the same seed value above, so an equality tie between
    // fixed slots contributes nothing but wasted Jacobian rows.
    if solve_bias {
        let strongest_factor_information = in_window_factors
            .iter()
            .map(|factor| {
                factor
                    .covariance_sqrt_information()
                    .map(|whitener| whitener.norm_squared())
                    .unwrap_or_else(|| {
                        factor
                            .weight_rotation
                            .max(factor.weight_velocity)
                            .max(factor.weight_position)
                    })
            })
            .fold(1.0_f64, f64::max);
        let shared_bias_equality_weight =
            (strongest_factor_information * 1.0e4).clamp(1.0e8, 1.0e20);
        for pair in sorted_ids.windows(2) {
            ba.add_bias_random_walk_factor(BiasRandomWalkFactor {
                keyframe_id_from: pair[0],
                keyframe_id_to: pair[1],
                weight_gyro: shared_bias_equality_weight,
                weight_accel: shared_bias_equality_weight,
            });
        }
    }

    let imu_factor_count = in_window_factors.len();
    for factor in in_window_factors {
        ba.add_imu_factor(factor);
    }

    let initial_cost_breakdown = ba.cost_breakdown(&ba_config.robust_kernel);
    let ba_result = ba.optimize(ba_config).ok()?;
    let final_cost_breakdown = ba.cost_breakdown(&ba_config.robust_kernel);

    // Collect refined per-keyframe states for the caller.
    let mut keyframe_states: BTreeMap<u64, KeyframeImuState> = BTreeMap::new();
    for kf_id in &sorted_ids {
        let velocity = ba
            .velocities
            .get(kf_id)
            .copied()
            .unwrap_or_else(Vector3::zeros);
        let bias_pair = ba.biases.get(kf_id).copied().unwrap_or_else(Vector6::zeros);
        let bias_gyro = Vector3::new(bias_pair[0], bias_pair[1], bias_pair[2]);
        let bias_acc = Vector3::new(bias_pair[3], bias_pair[4], bias_pair[5]);
        keyframe_states.insert(
            *kf_id,
            KeyframeImuState {
                velocity_world: velocity,
                bias_gyro,
                bias_acc,
            },
        );
    }

    Some(InertialOnlyViBaStats {
        keyframe_ids: sorted_ids,
        imu_factor_count,
        keyframe_states,
        ba_result,
        initial_cost_breakdown,
        final_cost_breakdown,
    })
}

/// Configuration for [`run_viba2_inertial_with_scale`] — the outer-loop
/// scale-recovery wrapper around [`run_inertial_only_vi_ba`].
///
/// On stereo / RGB-D sequences set `recover_scale = false`; the wrapper
/// then degenerates to a single inertial-only solve at `scale = 1.0` and
/// behaves identically to the VIBA1 entry point. On monocular sequences
/// set `recover_scale = true`; the wrapper iterates a 1-D least-squares
/// scale estimator alongside the inner solver until the relative change
/// in `s` drops below `scale_tolerance`.
#[derive(Debug, Clone, PartialEq)]
pub struct Viba2Config {
    /// Initial scale estimate. `1.0` is appropriate when the visual front
    /// end already exposes a metric anchor (stereo / RGB-D); for a fresh
    /// monocular bootstrap the initialiser produces a single-iteration
    /// closed-form guess from the IMU integral magnitudes before the
    /// first inner solve.
    pub initial_scale: f64,
    /// When `false`, scale is pinned at `initial_scale` (typical stereo
    /// path). When `true`, the wrapper iterates an alternating-
    /// minimisation outer loop: rescale the IMU factors by `1/s`, run
    /// the inertial-only inner solve, re-estimate `s` from the refined
    /// state, repeat. Default `false` to match the stereo-first stance
    /// of the existing VIBA1 entry point.
    pub recover_scale: bool,
    /// Maximum number of outer scale-refinement iterations. Each
    /// iteration runs one inner inertial-only solve. Default `5`.
    pub max_outer_iterations: usize,
    /// Convergence threshold on the relative scale change between two
    /// consecutive outer iterations. Default `1e-3`.
    pub scale_tolerance: f64,
    /// Inner BA config forwarded to [`run_inertial_only_vi_ba`].
    pub ba_config: crate::bundle::BaConfig,
}

impl Default for Viba2Config {
    fn default() -> Self {
        Self {
            initial_scale: 1.0,
            recover_scale: false,
            max_outer_iterations: 5,
            scale_tolerance: 1.0e-3,
            ba_config: crate::bundle::BaConfig {
                linear_solver: crate::LinearSolver::Sparse,
                max_iterations: 10,
                ..crate::bundle::BaConfig::default()
            },
        }
    }
}

/// Outcome of [`run_viba2_inertial_with_scale`].
#[derive(Debug, Clone, PartialEq)]
pub struct Viba2Stats {
    /// Sorted keyframe ids covered by the solve.
    pub keyframe_ids: Vec<u64>,
    /// Number of IMU factors fed into the inner solves.
    pub imu_factor_count: usize,
    /// Refined per-keyframe state from the final inner solve.
    pub keyframe_states: BTreeMap<u64, KeyframeImuState>,
    /// Scale factor recovered by the outer loop. `1.0` when
    /// `recover_scale = false`.
    pub scale: f64,
    /// History of scale values across the outer loop: index `0` is the
    /// `initial_scale`, subsequent entries are the values after each
    /// re-estimation. Length is `outer_iterations_run + 1`.
    pub scale_history: Vec<f64>,
    /// Number of outer iterations executed (one fewer than
    /// `scale_history.len()`).
    pub outer_iterations_run: usize,
    /// Final inner-solve BA result.
    pub ba_result: crate::bundle::BaResult,
    pub initial_cost_breakdown: BaCostBreakdown,
    pub final_cost_breakdown: BaCostBreakdown,
}

/// Run the VIBA2 inertial-with-scale outer loop.
///
/// Strategy. The inner solver ([`run_inertial_only_vi_ba`]) is invoked at
/// the current scale guess `s_k` with IMU factors pre-scaled by `1/s_k`
/// (positions, velocities, and gravity all get divided by `s_k`). After
/// the inner solve we re-estimate `s_{k+1}` by a closed-form 1D least
/// squares against the refined state (see [`estimate_scale_from_factors`]),
/// using the **original** factors so the estimator stays consistent across
/// iterations. The loop terminates when `|s_{k+1} - s_k| / s_k <
/// scale_tolerance` or after `max_outer_iterations` iterations.
///
/// When `config.recover_scale = false` the loop runs a single inner solve
/// at `config.initial_scale` and skips the re-estimation, matching the
/// stereo / known-scale path. The scale field on the result is
/// `config.initial_scale` in that case (typically `1.0`).
///
/// Returns `None` when [`run_inertial_only_vi_ba`] returns `None` on the
/// first inner solve, or when the initial scale is non-finite / non-positive.
pub fn run_viba2_inertial_with_scale(
    map: &mut VisualMap,
    keyframe_ids: &[u64],
    factors: &[ImuPreintegrationFactor],
    initial_states: &BTreeMap<u64, KeyframeImuState>,
    config: &Viba2Config,
) -> Option<Viba2Stats> {
    if !config.initial_scale.is_finite() || config.initial_scale <= 0.0 {
        return None;
    }

    let mut s_current = config.initial_scale;
    let mut scale_history: Vec<f64> = vec![s_current];
    let mut last_stats: Option<InertialOnlyViBaStats> = None;
    let mut outer_iter_run = 0usize;

    let max_iters = if config.recover_scale {
        config.max_outer_iterations.max(1)
    } else {
        1
    };

    for _ in 0..max_iters {
        let scaled_factors = rescale_imu_factors(factors, s_current);
        let stats = run_inertial_only_vi_ba(
            map,
            keyframe_ids,
            &scaled_factors,
            initial_states,
            &config.ba_config,
        )?;
        last_stats = Some(stats);
        outer_iter_run += 1;

        if !config.recover_scale {
            break;
        }

        // Re-estimate scale from the post-solve state, using the ORIGINAL
        // (un-rescaled) factors. The state has been refined under the
        // assumption that visual quantities are up-to-scale and IMU
        // quantities have been rescaled by 1/s_current; matching them
        // against the original factor delta requires the formula in
        // `estimate_scale_from_factors`.
        let refined_states = &last_stats.as_ref().unwrap().keyframe_states;
        let Some(s_new) = estimate_scale_from_factors(map, refined_states, factors) else {
            // Insufficient signal to update the scale (no factor with a
            // non-degenerate kinematic term); freeze at the current value.
            break;
        };
        let rel_change = (s_new - s_current).abs() / s_current.abs().max(1.0e-9);
        scale_history.push(s_new);
        let converged = rel_change < config.scale_tolerance;
        s_current = s_new;
        if converged {
            break;
        }
    }

    let stats = last_stats?;
    Some(Viba2Stats {
        keyframe_ids: stats.keyframe_ids,
        imu_factor_count: stats.imu_factor_count,
        keyframe_states: stats.keyframe_states,
        scale: s_current,
        scale_history,
        outer_iterations_run: outer_iter_run,
        ba_result: stats.ba_result,
        initial_cost_breakdown: stats.initial_cost_breakdown,
        final_cost_breakdown: stats.final_cost_breakdown,
    })
}

/// Rescale every factor's `(Δp, Δv, gravity_world)` by `1/scale` and
/// return a fresh vector. The rotation delta and bias linearisation are
/// preserved unchanged. Used by [`run_viba2_inertial_with_scale`] to
/// invariantly transform the inner solve into the visual-up-to-scale
/// frame.
fn rescale_imu_factors(
    factors: &[ImuPreintegrationFactor],
    scale: f64,
) -> Vec<ImuPreintegrationFactor> {
    if !scale.is_finite() || scale <= 0.0 {
        return factors.to_vec();
    }
    let inv_s = 1.0 / scale;
    factors
        .iter()
        .map(|f| {
            let mut clone = f.clone();
            clone.delta.delta_velocity = f.delta.delta_velocity * inv_s;
            clone.delta.delta_position = f.delta.delta_position * inv_s;
            clone.gravity_world = f.gravity_world * inv_s;
            clone
        })
        .collect()
}

/// Closed-form 1D least squares for the world scale given current
/// keyframe poses + velocities and original (un-rescaled) IMU factors.
///
/// Derivation. The original Forster position residual is
/// `r_p = R_iᵀ · (p_j_metric - p_i_metric - v_i_metric · Δt - 0.5 · g · Δt²) - Δp`.
/// Substituting `p_metric = s · p_visual`, `v_metric = s · v_visual`:
/// `r_p = R_iᵀ · (s · (p_j - p_i - v_i · Δt) - 0.5 · g · Δt²) - Δp`.
/// Defining `a = R_iᵀ · (p_j - p_i - v_i · Δt)` (3-vec, per factor) and
/// `b = Δp + R_iᵀ · 0.5 · g · Δt²` (3-vec, per factor), the least-squares
/// problem `min_s ‖s · a - b‖²` has the closed form
/// `s = (Σ aᵀ b) / (Σ aᵀ a)` summed across all factors.
///
/// Returns `None` when `Σ aᵀ a` is too small to identify the scale (i.e.
/// the kinematic term is degenerate — happens when every factor has
/// `p_j ≈ p_i + v_i · Δt`, the no-motion case). In that case the caller
/// should freeze the current scale estimate.
pub fn estimate_scale_from_factors(
    map: &VisualMap,
    states: &BTreeMap<u64, KeyframeImuState>,
    factors: &[ImuPreintegrationFactor],
) -> Option<f64> {
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for factor in factors {
        let p_i = map
            .keyframes
            .get(&factor.keyframe_id_from)
            .and_then(|kf| kf.frame.pose.as_ref())
            .map(|pose| pose.camera_center_world())?;
        let p_j = map
            .keyframes
            .get(&factor.keyframe_id_to)
            .and_then(|kf| kf.frame.pose.as_ref())
            .map(|pose| pose.camera_center_world())?;
        let v_i = states
            .get(&factor.keyframe_id_from)
            .map(|s| s.velocity_world)
            .unwrap_or_else(Vector3::zeros);
        let r_w_to_c = map
            .keyframes
            .get(&factor.keyframe_id_from)
            .and_then(|kf| kf.frame.pose.as_ref())
            .map(|pose| pose.world_to_camera.rotation)?;
        let dt = factor.delta.delta_time;
        if dt <= 0.0 || !dt.is_finite() {
            continue;
        }
        // `world_to_camera.rotation` is `R_c←w`. We need `R_b←w` for the
        // factor residual; for the body-aligned-with-camera convention
        // used in the rest of this module, that is the same quaternion.
        let kinematic_world: Vector3<f64> = (p_j - p_i) - v_i * dt;
        let a: Vector3<f64> = r_w_to_c * kinematic_world;
        let b: Vector3<f64> =
            factor.delta.delta_position + r_w_to_c * (0.5 * factor.gravity_world * dt * dt);
        num += a.dot(&b);
        den += a.dot(&a);
    }
    if den < 1.0e-12 {
        return None;
    }
    let s = num / den;
    if !s.is_finite() || s <= 0.0 {
        return None;
    }
    Some(s)
}

fn seed_velocity_for(
    map: &VisualMap,
    window_ids: &[u64],
    window_idx: usize,
    factors: &[ImuPreintegrationFactor],
    body_to_camera: &SE3,
) -> Option<Vector3<f64>> {
    // For the first window keyframe we have no prior estimate; the caller
    // falls back to zero. For everyone else, seed from the calibrated body-
    // centre displacement over the connecting IMU factor's `delta_time`.
    if window_idx == 0 {
        return None;
    }
    let kf_id = window_ids[window_idx];
    let prev_id = window_ids[window_idx - 1];
    let factor = factors
        .iter()
        .find(|f| f.keyframe_id_from == prev_id && f.keyframe_id_to == kf_id)?;
    if factor.delta.delta_time <= 0.0 {
        return None;
    }
    let prev_camera_pose = map.keyframes.get(&prev_id)?.frame.pose.as_ref()?;
    let curr_camera_pose = map.keyframes.get(&kf_id)?.frame.pose.as_ref()?;
    let prev_center = body_to_camera
        .compose(&prev_camera_pose.world_to_camera)
        .inverse()
        .translation;
    let curr_center = body_to_camera
        .compose(&curr_camera_pose.world_to_camera)
        .inverse()
        .translation;
    Some((curr_center - prev_center) / factor.delta.delta_time)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, UnitQuaternion};
    use visloc_core::geometry::{Pose, SO3};
    use visloc_core::types::{
        Camera, Frame, Keyframe, Landmark, Observation, StereoObservation, VisualMap,
    };

    use crate::imu_preintegration::ImuPreintegratedDelta;

    #[test]
    fn bias_random_walk_information_scales_with_factor_duration() {
        let noise = (2.0e-5, 3.0e-3);
        let short = bias_random_walk_information(noise, 0.5).expect("valid short factor");
        let long = bias_random_walk_information(noise, 2.0).expect("valid long factor");
        assert!((short.0 / long.0 - 4.0).abs() < 1.0e-12);
        assert!((short.1 / long.1 - 4.0).abs() < 1.0e-12);
        assert!(bias_random_walk_information(noise, 0.0).is_none());
    }

    #[test]
    fn navigation_prior_propagates_across_two_window_shifts() {
        let map = build_three_keyframe_map();
        let config = OnlineSlamLocalBaConfig {
            bias_random_walk_weights: Some((10.0, 10.0)),
            marginalize_navigation_state: true,
            initial_navigation_prior_std_devs: Some((1.0, 0.01, 0.1)),
            ..OnlineSlamLocalBaConfig::default()
        };
        let mut state = OnlineSlamLocalBaState::new(config);
        for id in [10_u64, 20, 30] {
            state.keyframe_state.insert(
                id,
                KeyframeImuState {
                    velocity_world: Vector3::new(0.0, 0.0, -1.0),
                    bias_gyro: Vector3::zeros(),
                    bias_acc: Vector3::zeros(),
                },
            );
        }
        let f_10_20 = constant_velocity_factor(10, 20, 1.0);
        let f_20_30 = constant_velocity_factor(20, 30, 1.0);
        let first = next_navigation_prior(&map, &state, &[10, 20], std::slice::from_ref(&f_10_20))
            .expect("initial fixed-gauge prior");
        assert_eq!(first.keyframe_ids, vec![20]);
        assert_eq!(first.information.shape(), (15, 15));
        assert!(first.information.clone().cholesky().is_some());

        state.navigation_prior = Some(first);
        let second = next_navigation_prior(&map, &state, &[20, 30], std::slice::from_ref(&f_20_30))
            .expect("Schur-propagated finite prior");
        assert_eq!(second.keyframe_ids, vec![30]);
        assert!(second.information.clone().cholesky().is_some());
        assert!(second.gradient.iter().all(|value| value.is_finite()));

        state.navigation_prior = Some(second);
        state.reset();
        assert!(state.navigation_prior.is_none());
    }

    #[test]
    fn covariance_whitened_initializer_keeps_bias_shared_across_keyframes() {
        let mut map = build_three_keyframe_map();
        let mut first = constant_velocity_factor(10, 20, 0.1);
        let mut second = constant_velocity_factor(20, 30, 0.1);
        first.delta.delta_rotation = SO3::from_quaternion(UnitQuaternion::from_scaled_axis(
            Vector3::new(0.01, 0.0, 0.0),
        ));
        second.delta.delta_rotation = SO3::from_quaternion(UnitQuaternion::from_scaled_axis(
            Vector3::new(0.0, 0.02, 0.0),
        ));
        first.delta.j_rotation_bg = nalgebra::Matrix3::identity() * 0.1;
        second.delta.j_rotation_bg = nalgebra::Matrix3::identity() * 0.1;
        first.delta.covariance = crate::imu_preintegration::Matrix9::identity() * 1.0e-8;
        second.delta.covariance = crate::imu_preintegration::Matrix9::identity() * 1.0e-8;
        let result = run_inertial_only_vi_ba(
            &mut map,
            &[10, 20, 30],
            &[first, second],
            &BTreeMap::new(),
            &BaConfig::default(),
        )
        .expect("inertial initializer");
        let biases: Vec<Vector3<f64>> = result
            .keyframe_states
            .values()
            .map(|state| state.bias_gyro)
            .collect();
        for pair in biases.windows(2) {
            assert!(
                (pair[1] - pair[0]).norm() < 1.0e-6,
                "shared initialization bias drifted: {:?} vs {:?}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn solve_bias_false_fixes_bias_exactly_while_velocity_still_converges() {
        // Same 3-keyframe / 2-factor constant-velocity fixture used by
        // `covariance_whitened_initializer_keeps_bias_shared_across_keyframes`
        // above, but seeded with a nonzero bias and zero velocity, then
        // solved with `solve_bias = false`. The camera centers move by
        // exactly (0, 0, -1) per 1-second hop (see `build_three_keyframe_map`),
        // so the position residual (which depends on the FIXED poses, not on
        // bias) pulls every keyframe's velocity to that same value from the
        // zero seed, while the bias slots — gauge-fixed at the seed — must
        // come back completely unchanged.
        let mut map = build_three_keyframe_map();
        let f1 = constant_velocity_factor(10, 20, 1.0);
        let f2 = constant_velocity_factor(20, 30, 1.0);

        let seed_bias_gyro = Vector3::new(0.01, -0.02, 0.03);
        let seed_bias_acc = Vector3::new(0.05, 0.04, -0.03);
        let mut initial_states = BTreeMap::new();
        for kf_id in [10u64, 20, 30] {
            initial_states.insert(
                kf_id,
                KeyframeImuState {
                    velocity_world: Vector3::zeros(),
                    bias_gyro: seed_bias_gyro,
                    bias_acc: seed_bias_acc,
                },
            );
        }

        let result = run_inertial_only_vi_ba_with_options(
            &mut map,
            &[10, 20, 30],
            &[f1, f2],
            &initial_states,
            &BaConfig::default(),
            false,
        )
        .expect("velocity-only solve must converge");

        let expected_velocity = Vector3::new(0.0, 0.0, -1.0);
        for (kf_id, state) in &result.keyframe_states {
            assert!(
                (state.bias_gyro - seed_bias_gyro).norm() < 1.0e-12,
                "kf {kf_id}: gyro bias moved despite solve_bias=false: {:?}",
                state.bias_gyro
            );
            assert!(
                (state.bias_acc - seed_bias_acc).norm() < 1.0e-12,
                "kf {kf_id}: accel bias moved despite solve_bias=false: {:?}",
                state.bias_acc
            );
            assert!(
                (state.velocity_world - expected_velocity).norm() < 1.0e-6,
                "kf {kf_id}: velocity failed to converge: {:?}",
                state.velocity_world
            );
        }
    }

    #[test]
    fn solve_bias_true_path_matches_public_wrapper() {
        // `run_inertial_only_vi_ba_with_options(.., solve_bias = true)` must
        // reproduce `run_inertial_only_vi_ba` exactly — the public entry
        // point is now a thin wrapper around the options function.
        let mut map_a = build_three_keyframe_map();
        let mut map_b = map_a.clone();
        let f1 = constant_velocity_factor(10, 20, 1.0);
        let f2 = constant_velocity_factor(20, 30, 1.0);

        let via_wrapper = run_inertial_only_vi_ba(
            &mut map_a,
            &[10, 20, 30],
            &[f1.clone(), f2.clone()],
            &BTreeMap::new(),
            &BaConfig::default(),
        )
        .expect("wrapper solve");
        let via_options = run_inertial_only_vi_ba_with_options(
            &mut map_b,
            &[10, 20, 30],
            &[f1, f2],
            &BTreeMap::new(),
            &BaConfig::default(),
            true,
        )
        .expect("explicit solve_bias=true solve");

        assert_eq!(via_wrapper.keyframe_ids, via_options.keyframe_ids);
        assert_eq!(via_wrapper.imu_factor_count, via_options.imu_factor_count);
        for kf_id in &via_wrapper.keyframe_ids {
            let a = &via_wrapper.keyframe_states[kf_id];
            let b = &via_options.keyframe_states[kf_id];
            assert!((a.velocity_world - b.velocity_world).norm() < 1.0e-12);
            assert!((a.bias_gyro - b.bias_gyro).norm() < 1.0e-12);
            assert!((a.bias_acc - b.bias_acc).norm() < 1.0e-12);
        }
    }

    fn make_camera() -> Camera {
        Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0)
    }

    fn build_three_keyframe_map() -> VisualMap {
        let mut map = VisualMap::new();
        let camera = make_camera();
        map.cameras.insert(camera.id, camera.clone());
        // 10 landmarks scattered ahead of the camera path.
        let landmark_positions: Vec<Point3<f64>> = (0..10)
            .map(|i| {
                let t = i as f64;
                Point3::new(
                    (t * 0.7).sin() * 1.5,
                    ((t * 0.4).cos() - 0.5) * 1.2,
                    5.0 + (t * 0.5).sin() * 1.0,
                )
            })
            .collect();
        for (i, pos) in landmark_positions.iter().enumerate() {
            map.landmarks
                .insert((i as u64) + 1, Landmark::new((i as u64) + 1, *pos));
        }
        // Three keyframes at z = 0, 1, 2 (positive z because the camera moves
        // *backwards* in world space → world-to-camera puts t.z = -z).
        let centers = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 0.0, -2.0),
        ];
        for (idx, &center) in centers.iter().enumerate() {
            let rotation = UnitQuaternion::identity();
            let pose = Pose::from_world_to_camera(rotation, -rotation.transform_vector(&center));
            let frame_id = ((idx as u64) + 1) * 10;
            let mut frame = Frame::new(frame_id, camera.id);
            frame.pose = Some(pose.clone());
            let mut observations = Vec::new();
            for (lm_idx, lm_pos) in landmark_positions.iter().enumerate() {
                let cam_point = pose.transform_world_point(lm_pos);
                let Some(uv) = camera.project(&cam_point) else {
                    continue;
                };
                if uv.x < 0.0
                    || uv.x >= camera.width as f64
                    || uv.y < 0.0
                    || uv.y >= camera.height as f64
                {
                    continue;
                }
                let lm_id = (lm_idx as u64) + 1;
                let keypoint_index = frame.keypoints.len();
                frame.keypoints.push(uv);
                frame.descriptors.push(vec![lm_id as f32]);
                observations.push(Observation {
                    frame_id,
                    landmark_id: lm_id,
                    keypoint_index,
                    xy: uv,
                });
            }
            for obs in &observations {
                if let Some(landmark) = map.landmarks.get_mut(&obs.landmark_id) {
                    landmark.observations.push(obs.clone());
                }
            }
            map.keyframes.insert(
                frame_id,
                Keyframe {
                    frame,
                    observations,
                },
            );
        }
        map
    }

    fn constant_velocity_factor(from_id: u64, to_id: u64, delta_t: f64) -> ImuPreintegrationFactor {
        // Zero-gravity, constant-velocity scene: ΔR = I, Δv = 0, Δp = 0
        // (the keyframe-i body frame is at rest relative to the world by
        // construction). Weights are 1.0 so the IMU residual sits at the
        // same magnitude as the reprojection rows; the BA's reprojection
        // residuals already vanish at truth, so this factor's job is just
        // to anchor velocity / bias DoFs.
        let delta = ImuPreintegratedDelta {
            delta_rotation: SO3::identity(),
            delta_velocity: Vector3::zeros(),
            delta_position: Vector3::zeros(),
            delta_time: delta_t,
            bias_gyro_linearisation: Vector3::zeros(),
            bias_acc_linearisation: Vector3::zeros(),
            j_rotation_bg: nalgebra::Matrix3::zeros(),
            j_velocity_ba: nalgebra::Matrix3::zeros(),
            j_velocity_bg: nalgebra::Matrix3::zeros(),
            j_position_ba: nalgebra::Matrix3::zeros(),
            j_position_bg: nalgebra::Matrix3::zeros(),
            covariance: crate::imu_preintegration::Matrix9::zeros(),
        };
        ImuPreintegrationFactor {
            keyframe_id_from: from_id,
            keyframe_id_to: to_id,
            delta,
            gravity_world: Vector3::zeros(),
            weight_position: 1.0,
            weight_velocity: 1.0,
            weight_rotation: 1.0,
        }
    }

    #[test]
    fn local_vi_ba_returns_none_without_factors_in_window() {
        let mut map = build_three_keyframe_map();
        let mut state = OnlineSlamLocalBaState::new(OnlineSlamLocalBaConfig::default());
        // No factors registered → no in-window factors → no solve.
        let result = run_local_vi_ba(&mut map, &mut state);
        assert!(result.is_none());
    }

    #[test]
    fn local_vi_ba_refines_window_when_factors_present() {
        let mut map = build_three_keyframe_map();
        let mut state = OnlineSlamLocalBaState::new(OnlineSlamLocalBaConfig {
            // Use zero gravity / zero-rate scene so the test factor stays
            // consistent with the static-scene synthetic poses.
            gravity_world: Vector3::zeros(),
            ..OnlineSlamLocalBaConfig::default()
        });
        state
            .factor_history
            .push(constant_velocity_factor(10, 20, 0.1));
        state
            .factor_history
            .push(constant_velocity_factor(20, 30, 0.1));
        let result = run_local_vi_ba(&mut map, &mut state).expect("BA should run");
        assert_eq!(result.window_keyframe_ids, vec![10, 20, 30]);
        assert_eq!(result.imu_factor_count, 2);
        assert!(result.observation_count > 0);
        assert!(result.ba_result.final_cost.is_finite());
        // The state table now carries an entry per window keyframe.
        for kf_id in &result.window_keyframe_ids {
            assert!(state.keyframe_state.contains_key(kf_id));
        }
    }

    #[test]
    fn local_vi_ba_uses_calibrated_stereo_observations() {
        let mut map = build_three_keyframe_map();
        let right_camera = Camera::pinhole(2, 640, 480, 500.0, 500.0, 320.0, 240.0);
        map.cameras.insert(right_camera.id, right_camera.clone());
        let left_to_right = SE3::new(UnitQuaternion::identity(), Vector3::new(-0.1, 0.0, 0.0));
        for (keyframe_id, keyframe) in &map.keyframes {
            let pose = keyframe.frame.pose.as_ref().unwrap();
            for observation in &keyframe.observations {
                let point = map.landmarks[&observation.landmark_id].position;
                let point_right =
                    left_to_right.transform_point(&pose.transform_world_point(&point));
                let Some(xy_right) = right_camera.project(&point_right) else {
                    continue;
                };
                map.stereo_observations.push(StereoObservation {
                    frame_id: *keyframe_id,
                    landmark_id: observation.landmark_id,
                    right_camera_id: right_camera.id,
                    xy_right,
                    left_to_right: left_to_right.clone(),
                });
            }
        }
        let mut state = OnlineSlamLocalBaState::new(OnlineSlamLocalBaConfig {
            gravity_world: Vector3::zeros(),
            use_general_stereo_observations: true,
            ..OnlineSlamLocalBaConfig::default()
        });
        state
            .factor_history
            .push(constant_velocity_factor(10, 20, 0.1));
        state
            .factor_history
            .push(constant_velocity_factor(20, 30, 0.1));
        let result = run_local_vi_ba(&mut map, &mut state).expect("stereo VI-BA");
        assert!(result.stereo_observation_count > 0);
        assert_eq!(result.stereo_observation_count, result.observation_count);
    }

    #[test]
    fn local_vi_ba_bias_freeze_does_not_fire_when_threshold_disabled() {
        let mut map = build_three_keyframe_map();
        let mut state = OnlineSlamLocalBaState::new(OnlineSlamLocalBaConfig {
            gravity_world: Vector3::zeros(),
            freeze_biases_when_cost_ratio_above: None,
            ..OnlineSlamLocalBaConfig::default()
        });
        state
            .factor_history
            .push(constant_velocity_factor(10, 20, 0.1));
        state
            .factor_history
            .push(constant_velocity_factor(20, 30, 0.1));
        let result = run_local_vi_ba(&mut map, &mut state).expect("BA should run");
        assert!(!result.bias_frozen);
    }

    #[test]
    fn local_vi_ba_bias_freeze_fires_when_threshold_zero() {
        // A `Some(0.0)` threshold says "any non-zero final cost is too
        // much": the conditioning fallback ALWAYS fires (as long as the
        // initial cost itself was non-zero). The re-solve runs with
        // biases gauge-frozen and the bias writeback is suppressed.
        let mut map = build_three_keyframe_map();
        let seeded_bias_gyro = Vector3::new(0.123, 0.234, 0.345);
        let seeded_bias_acc = Vector3::new(-0.456, -0.567, -0.678);
        let mut state = OnlineSlamLocalBaState::new(OnlineSlamLocalBaConfig {
            gravity_world: Vector3::zeros(),
            bias_gyro_init: seeded_bias_gyro,
            bias_acc_init: seeded_bias_acc,
            freeze_biases_when_cost_ratio_above: Some(0.0),
            ..OnlineSlamLocalBaConfig::default()
        });
        state
            .factor_history
            .push(constant_velocity_factor(10, 20, 0.1));
        state
            .factor_history
            .push(constant_velocity_factor(20, 30, 0.1));
        let result = run_local_vi_ba(&mut map, &mut state).expect("BA should run");
        // The fallback only fires when initial_cost > 0; the scene has
        // non-zero IMU residuals from the seeded biases so this gate
        // does activate.
        if result.ba_result.initial_cost > 0.0 {
            assert!(result.bias_frozen);
            // Biases in the state table are at their pre-BA values — the
            // re-solve fixed them and the writeback suppressed the
            // update. Anchor (kf 10) was already gauge-fixed; check
            // non-anchor keyframes (20, 30).
            for kf_id in [20u64, 30] {
                let slot = state.keyframe_state.get(&kf_id).expect("slot exists");
                let dg = (slot.bias_gyro - seeded_bias_gyro).norm();
                let da = (slot.bias_acc - seeded_bias_acc).norm();
                assert!(dg < 1.0e-9, "bias_gyro drift on kf {kf_id}: {dg}");
                assert!(da < 1.0e-9, "bias_acc drift on kf {kf_id}: {da}");
            }
        }
    }

    #[test]
    fn local_vi_ba_quality_gate_rejects_writeback() {
        let mut map = build_three_keyframe_map();
        let original_pose_20 = map
            .keyframes
            .get(&20)
            .and_then(|kf| kf.frame.pose.clone())
            .expect("kf 20 has pose");
        let original_landmark_1 = map.landmarks.get(&1).expect("landmark exists").position;

        let mut state = OnlineSlamLocalBaState::new(OnlineSlamLocalBaConfig {
            gravity_world: Vector3::zeros(),
            // Force the branch deterministically for the unit test. Normal
            // callers should use a non-negative ratio such as 1.0.
            reject_writeback_when_cost_ratio_above: Some(f64::NEG_INFINITY),
            ..OnlineSlamLocalBaConfig::default()
        });
        for kf_id in [10u64, 20, 30] {
            state.keyframe_state.insert(
                kf_id,
                KeyframeImuState {
                    velocity_world: Vector3::new(kf_id as f64, 0.5, -0.25),
                    bias_gyro: Vector3::new(0.01, -0.02, 0.03),
                    bias_acc: Vector3::new(0.1, -0.2, 0.3),
                },
            );
        }
        let original_keyframe_state = state.keyframe_state.clone();
        state
            .factor_history
            .push(constant_velocity_factor(10, 20, 0.1));
        state
            .factor_history
            .push(constant_velocity_factor(20, 30, 0.1));

        let result = run_local_vi_ba(&mut map, &mut state).expect("BA should run");

        assert!(result.quality_gate_rejected);
        assert!(result.cost_ratio_gate_rejected);
        assert!(!result.velocity_gate_rejected);
        assert!(result.cost_ratio.is_finite());
        assert!(result.max_refined_velocity_norm_mps.is_finite());
        assert_eq!(
            map.keyframes
                .get(&20)
                .and_then(|kf| kf.frame.pose.clone())
                .expect("kf 20 has pose"),
            original_pose_20,
            "rejected VI-BA must not write poses back"
        );
        assert_eq!(
            map.landmarks.get(&1).expect("landmark exists").position,
            original_landmark_1,
            "rejected VI-BA must not write landmarks back"
        );
        assert_eq!(
            state.keyframe_state, original_keyframe_state,
            "rejected VI-BA must not write velocity/bias slots back"
        );
    }

    #[test]
    fn local_vi_ba_imu_nis_gate_rejects_writeback() {
        let mut map = build_three_keyframe_map();
        let original_pose_20 = map
            .keyframes
            .get(&20)
            .and_then(|kf| kf.frame.pose.clone())
            .expect("kf 20 has pose");
        let original_landmark_1 = map.landmarks.get(&1).expect("landmark exists").position;

        let mut state = OnlineSlamLocalBaState::new(OnlineSlamLocalBaConfig {
            gravity_world: Vector3::zeros(),
            // Any finite final NIS is above negative infinity, which makes
            // this a deterministic transactional-gate test.
            reject_writeback_when_final_imu_nis_per_dof_above: Some(f64::NEG_INFINITY),
            ..OnlineSlamLocalBaConfig::default()
        });
        for kf_id in [10u64, 20, 30] {
            state.keyframe_state.insert(
                kf_id,
                KeyframeImuState {
                    velocity_world: Vector3::new(kf_id as f64, 0.5, -0.25),
                    bias_gyro: Vector3::new(0.01, -0.02, 0.03),
                    bias_acc: Vector3::new(0.1, -0.2, 0.3),
                },
            );
        }
        let original_keyframe_state = state.keyframe_state.clone();
        state
            .factor_history
            .push(constant_velocity_factor(10, 20, 0.1));
        state
            .factor_history
            .push(constant_velocity_factor(20, 30, 0.1));

        let result = run_local_vi_ba(&mut map, &mut state).expect("BA should run");

        assert!(result.quality_gate_rejected);
        assert!(!result.cost_ratio_gate_rejected);
        assert!(result.imu_nis_gate_rejected);
        assert!(!result.velocity_gate_rejected);
        assert!(result
            .final_cost_breakdown
            .imu_normalized_squared_residual_per_dof
            .is_some_and(f64::is_finite));
        assert_eq!(
            map.keyframes
                .get(&20)
                .and_then(|kf| kf.frame.pose.clone())
                .expect("kf 20 has pose"),
            original_pose_20,
            "NIS-gated VI-BA must not write poses back"
        );
        assert_eq!(
            map.landmarks.get(&1).expect("landmark exists").position,
            original_landmark_1,
            "NIS-gated VI-BA must not write landmarks back"
        );
        assert_eq!(
            state.keyframe_state, original_keyframe_state,
            "NIS-gated VI-BA must not write velocity/bias slots back"
        );
    }

    #[test]
    fn local_vi_ba_velocity_gate_rejects_writeback() {
        let mut map = build_three_keyframe_map();
        let original_pose_20 = map
            .keyframes
            .get(&20)
            .and_then(|kf| kf.frame.pose.clone())
            .expect("kf 20 has pose");

        let mut state = OnlineSlamLocalBaState::new(OnlineSlamLocalBaConfig {
            gravity_world: Vector3::zeros(),
            // Force the branch deterministically for the unit test. Normal
            // callers should use a non-negative physical velocity cap.
            reject_writeback_when_velocity_norm_above_mps: Some(f64::NEG_INFINITY),
            ..OnlineSlamLocalBaConfig::default()
        });
        state
            .factor_history
            .push(constant_velocity_factor(10, 20, 0.1));
        state
            .factor_history
            .push(constant_velocity_factor(20, 30, 0.1));

        let result = run_local_vi_ba(&mut map, &mut state).expect("BA should run");

        assert!(result.quality_gate_rejected);
        assert!(!result.cost_ratio_gate_rejected);
        assert!(result.velocity_gate_rejected);
        assert!(result.max_refined_velocity_norm_mps.is_finite());
        assert_eq!(
            map.keyframes
                .get(&20)
                .and_then(|kf| kf.frame.pose.clone())
                .expect("kf 20 has pose"),
            original_pose_20,
            "velocity-gated VI-BA must not write poses back"
        );
    }

    #[test]
    fn local_vi_ba_pose_correction_gate_rejects_transactionally() {
        let mut map = build_three_keyframe_map();
        let original_pose_20 = map.keyframes[&20].frame.pose.clone().unwrap();
        let original_landmark = map.landmarks[&1].position;
        let mut state = OnlineSlamLocalBaState::new(OnlineSlamLocalBaConfig {
            gravity_world: Vector3::zeros(),
            reject_writeback_when_pose_translation_above_meters: Some(f64::NEG_INFINITY),
            ..OnlineSlamLocalBaConfig::default()
        });
        state
            .factor_history
            .push(constant_velocity_factor(10, 20, 0.1));
        state
            .factor_history
            .push(constant_velocity_factor(20, 30, 0.1));
        let result = run_local_vi_ba(&mut map, &mut state).expect("BA should run");
        assert!(result.quality_gate_rejected);
        assert!(result.pose_correction_gate_rejected);
        assert!(result.max_pose_translation_correction_meters.is_finite());
        assert_eq!(map.keyframes[&20].frame.pose, Some(original_pose_20));
        assert_eq!(map.landmarks[&1].position, original_landmark);
        assert!(state.navigation_prior.is_none());
    }

    /// Regression for the `gt_full1500` real-data failure mode: local
    /// VI-BA writeback mirrored an accel bias that diverged to
    /// `‖·‖ ≈ 8.6 m/s²` because no gate checked the physical plausibility
    /// of the optimized bias itself. `reject_gyro_bias_above_rad_s` must
    /// reject the whole trigger transactionally — map poses, landmarks,
    /// and the mirrored `state.keyframe_state` all stay untouched — the
    /// same pattern as the other writeback quality gates.
    #[test]
    fn local_vi_ba_gyro_bias_magnitude_gate_rejects_transactionally() {
        let mut map = build_three_keyframe_map();
        let original_pose_20 = map.keyframes[&20].frame.pose.clone().unwrap();
        let original_landmark = map.landmarks[&1].position;
        let mut state = OnlineSlamLocalBaState::new(OnlineSlamLocalBaConfig {
            gravity_world: Vector3::zeros(),
            // Force the branch deterministically for the unit test. Normal
            // callers should use a non-negative physical bias cap (bench
            // value 0.2 rad/s, mirroring the motion-VI init bootstrap gate).
            reject_gyro_bias_above_rad_s: Some(f64::NEG_INFINITY),
            ..OnlineSlamLocalBaConfig::default()
        });
        // Pre-seed the per-keyframe IMU state table so
        // `state.keyframe_state` already has entries for every in-window
        // keyframe. Otherwise `run_local_vi_ba`'s auto-seeding of NEW
        // keyframe state slots (which runs unconditionally, before any
        // quality gate is evaluated) would mutate the table even on a
        // rejected trigger, defeating this test's transactional check —
        // mirrors `local_vi_ba_imu_nis_gate_rejects_writeback`.
        for kf_id in [10u64, 20, 30] {
            state.keyframe_state.insert(
                kf_id,
                KeyframeImuState {
                    velocity_world: Vector3::new(kf_id as f64, 0.5, -0.25),
                    bias_gyro: Vector3::new(0.01, -0.02, 0.03),
                    bias_acc: Vector3::new(0.1, -0.2, 0.3),
                },
            );
        }
        let original_keyframe_state = state.keyframe_state.clone();
        state
            .factor_history
            .push(constant_velocity_factor(10, 20, 0.1));
        state
            .factor_history
            .push(constant_velocity_factor(20, 30, 0.1));

        let result = run_local_vi_ba(&mut map, &mut state).expect("BA should run");

        assert!(result.quality_gate_rejected);
        assert!(result.bias_magnitude_gate_rejected);
        assert!(!result.cost_ratio_gate_rejected);
        assert!(!result.velocity_gate_rejected);
        assert!(!result.pose_correction_gate_rejected);
        assert!(result.max_refined_gyro_bias_norm_rad_s.is_finite());
        assert_eq!(
            map.keyframes[&20].frame.pose,
            Some(original_pose_20),
            "gyro-bias-gated VI-BA must not write poses back"
        );
        assert_eq!(
            map.landmarks[&1].position, original_landmark,
            "gyro-bias-gated VI-BA must not write landmarks back"
        );
        assert_eq!(
            state.keyframe_state, original_keyframe_state,
            "gyro-bias-gated VI-BA must not write velocity/bias slots back"
        );
    }

    /// Same transactional contract as
    /// `local_vi_ba_gyro_bias_magnitude_gate_rejects_transactionally`, but
    /// exercising `reject_accel_bias_above_mps2` — the field whose
    /// `gt_full1500` mirrored value (`‖·‖ ≈ 8.6`) directly motivated this
    /// gate (bench value 1.0, mirroring the motion-VI init bootstrap gate).
    #[test]
    fn local_vi_ba_accel_bias_magnitude_gate_rejects_transactionally() {
        let mut map = build_three_keyframe_map();
        let original_pose_20 = map.keyframes[&20].frame.pose.clone().unwrap();
        let original_landmark = map.landmarks[&1].position;
        let mut state = OnlineSlamLocalBaState::new(OnlineSlamLocalBaConfig {
            gravity_world: Vector3::zeros(),
            reject_accel_bias_above_mps2: Some(f64::NEG_INFINITY),
            ..OnlineSlamLocalBaConfig::default()
        });
        // See the matching comment in
        // `local_vi_ba_gyro_bias_magnitude_gate_rejects_transactionally`:
        // pre-seed every in-window keyframe's state so the auto-seed path
        // cannot mutate `state.keyframe_state` ahead of the gate check.
        for kf_id in [10u64, 20, 30] {
            state.keyframe_state.insert(
                kf_id,
                KeyframeImuState {
                    velocity_world: Vector3::new(kf_id as f64, 0.5, -0.25),
                    bias_gyro: Vector3::new(0.01, -0.02, 0.03),
                    bias_acc: Vector3::new(0.1, -0.2, 0.3),
                },
            );
        }
        let original_keyframe_state = state.keyframe_state.clone();
        state
            .factor_history
            .push(constant_velocity_factor(10, 20, 0.1));
        state
            .factor_history
            .push(constant_velocity_factor(20, 30, 0.1));

        let result = run_local_vi_ba(&mut map, &mut state).expect("BA should run");

        assert!(result.quality_gate_rejected);
        assert!(result.bias_magnitude_gate_rejected);
        assert!(!result.cost_ratio_gate_rejected);
        assert!(!result.velocity_gate_rejected);
        assert!(!result.pose_correction_gate_rejected);
        assert!(result.max_refined_accel_bias_norm_mps2.is_finite());
        assert_eq!(
            map.keyframes[&20].frame.pose,
            Some(original_pose_20),
            "accel-bias-gated VI-BA must not write poses back"
        );
        assert_eq!(
            map.landmarks[&1].position, original_landmark,
            "accel-bias-gated VI-BA must not write landmarks back"
        );
        assert_eq!(
            state.keyframe_state, original_keyframe_state,
            "accel-bias-gated VI-BA must not write velocity/bias slots back"
        );
    }

    /// The bias-magnitude gate must stay dormant (and writeback must
    /// proceed) with the default `None` config, and with a generous
    /// bound that the synthetic solve never approaches. Regression guard
    /// against the new gate accidentally becoming stricter than intended
    /// for existing callers that don't set it.
    #[test]
    fn local_vi_ba_bias_magnitude_gate_permits_writeback_when_within_bound() {
        let mut map = build_three_keyframe_map();
        let original_pose_20 = map.keyframes[&20].frame.pose.clone().unwrap();

        // Case 1: gate left at the legacy default (`None`).
        let mut state_default = OnlineSlamLocalBaState::new(OnlineSlamLocalBaConfig {
            gravity_world: Vector3::zeros(),
            ..OnlineSlamLocalBaConfig::default()
        });
        state_default
            .factor_history
            .push(constant_velocity_factor(10, 20, 0.1));
        state_default
            .factor_history
            .push(constant_velocity_factor(20, 30, 0.1));
        let result_default = run_local_vi_ba(&mut map, &mut state_default).expect("BA should run");
        assert!(!result_default.quality_gate_rejected);
        assert!(!result_default.bias_magnitude_gate_rejected);
        assert_ne!(
            map.keyframes[&20].frame.pose,
            Some(original_pose_20.clone()),
            "default-gate VI-BA must write the refined pose back"
        );

        // Case 2: gate enabled but with a bound far above anything the
        // synthetic window's solve can produce.
        let mut map_generous = build_three_keyframe_map();
        let mut state_generous = OnlineSlamLocalBaState::new(OnlineSlamLocalBaConfig {
            gravity_world: Vector3::zeros(),
            reject_gyro_bias_above_rad_s: Some(1.0e6),
            reject_accel_bias_above_mps2: Some(1.0e6),
            ..OnlineSlamLocalBaConfig::default()
        });
        state_generous
            .factor_history
            .push(constant_velocity_factor(10, 20, 0.1));
        state_generous
            .factor_history
            .push(constant_velocity_factor(20, 30, 0.1));
        let result_generous =
            run_local_vi_ba(&mut map_generous, &mut state_generous).expect("BA should run");
        assert!(!result_generous.quality_gate_rejected);
        assert!(!result_generous.bias_magnitude_gate_rejected);
        assert_ne!(
            map_generous.keyframes[&20].frame.pose,
            Some(original_pose_20),
            "generously-bounded VI-BA must write the refined pose back"
        );
    }

    #[test]
    fn local_vi_ba_adaptive_velocity_gate_rejects_writeback() {
        let mut map = build_three_keyframe_map();
        let original_pose_20 = map
            .keyframes
            .get(&20)
            .and_then(|kf| kf.frame.pose.clone())
            .expect("kf 20 has pose");

        let mut state = OnlineSlamLocalBaState::new(OnlineSlamLocalBaConfig {
            gravity_world: Vector3::zeros(),
            adaptive_velocity_gate: Some(AdaptiveVelocityGateConfig {
                // Force the branch deterministically: the synthetic test
                // window has finite reference velocities, so a zero threshold
                // must reject any non-zero refined velocity.
                reference_quantile: 0.5,
                multiplier: 0.0,
                margin_mps: 0.0,
                min_threshold_mps: 0.0,
                max_threshold_mps: None,
                min_reference_count: 1,
            }),
            ..OnlineSlamLocalBaConfig::default()
        });
        state
            .factor_history
            .push(constant_velocity_factor(10, 20, 0.1));
        state
            .factor_history
            .push(constant_velocity_factor(20, 30, 0.1));

        let result = run_local_vi_ba(&mut map, &mut state).expect("BA should run");

        assert!(result.quality_gate_rejected);
        assert!(!result.cost_ratio_gate_rejected);
        assert!(!result.velocity_gate_rejected);
        assert!(result.adaptive_velocity_gate_rejected);
        assert_eq!(result.adaptive_velocity_gate_threshold_mps, Some(0.0));
        assert_eq!(
            map.keyframes
                .get(&20)
                .and_then(|kf| kf.frame.pose.clone())
                .expect("kf 20 has pose"),
            original_pose_20,
            "adaptive velocity-gated VI-BA must not write poses back"
        );
    }

    #[test]
    fn run_at_vi_init_promotion_default_is_false() {
        // Phase-16 lever defaults to off so existing callers preserve
        // their Phase-13 / Phase-14 cadence (BA fires only on a new
        // post-promotion IMU factor). Flip the flag explicitly to opt
        // into the "promotion-time" trigger.
        let cfg = OnlineSlamLocalBaConfig::default();
        assert!(!cfg.run_at_vi_init_promotion);
    }

    #[test]
    fn relinearise_threshold_off_leaves_factor_linearisation_at_construction_value() {
        // Default config: `relinearise_imu_factor_bias_thresholds = None`.
        // Pre-seed factor history at b_lin = 0 and per-keyframe state at a
        // non-trivial bias. After a BA pass the factor's stored
        // `bias_*_linearisation` must STILL be zero (no re-linearisation),
        // and the stats counter must be zero.
        let mut map = build_three_keyframe_map();
        let mut state = OnlineSlamLocalBaState::new(OnlineSlamLocalBaConfig {
            gravity_world: Vector3::zeros(),
            relinearise_imu_factor_bias_thresholds: None,
            ..OnlineSlamLocalBaConfig::default()
        });
        // Seed per-keyframe state with a non-zero bias for the from-kf of
        // every banked factor.
        for kf_id in [10u64, 20, 30] {
            state.keyframe_state.insert(
                kf_id,
                KeyframeImuState {
                    velocity_world: Vector3::zeros(),
                    bias_gyro: Vector3::new(0.05, -0.03, 0.02),
                    bias_acc: Vector3::new(0.2, -0.15, 0.10),
                },
            );
        }
        state
            .factor_history
            .push(constant_velocity_factor(10, 20, 0.1));
        state
            .factor_history
            .push(constant_velocity_factor(20, 30, 0.1));
        let result = run_local_vi_ba(&mut map, &mut state).expect("BA should run");
        assert_eq!(
            result.relinearised_factor_count, 0,
            "no factors should re-linearise when the threshold is None"
        );
        for factor in &state.factor_history {
            assert_eq!(
                factor.delta.bias_gyro_linearisation,
                Vector3::zeros(),
                "linearisation point must stay at construction value with threshold=None"
            );
        }
    }

    #[test]
    fn relinearise_threshold_refreshes_factors_above_drift() {
        // With `relinearise_imu_factor_bias_thresholds = Some((g, a))` and
        // a per-keyframe bias far from the factor's stored `b_*_lin = 0`,
        // both banked factors must be re-linearised: stats counter = 2 and
        // `bias_*_linearisation` updated to match the per-keyframe state of
        // the factor's from-keyframe.
        let mut map = build_three_keyframe_map();
        let mut state = OnlineSlamLocalBaState::new(OnlineSlamLocalBaConfig {
            gravity_world: Vector3::zeros(),
            relinearise_imu_factor_bias_thresholds: Some((0.01, 0.1)),
            ..OnlineSlamLocalBaConfig::default()
        });
        let kf10_bias_g = Vector3::new(0.05, -0.03, 0.02);
        let kf10_bias_a = Vector3::new(0.2, -0.15, 0.10);
        let kf20_bias_g = Vector3::new(0.06, -0.04, 0.025);
        let kf20_bias_a = Vector3::new(0.25, -0.18, 0.12);
        state.keyframe_state.insert(
            10,
            KeyframeImuState {
                velocity_world: Vector3::zeros(),
                bias_gyro: kf10_bias_g,
                bias_acc: kf10_bias_a,
            },
        );
        state.keyframe_state.insert(
            20,
            KeyframeImuState {
                velocity_world: Vector3::zeros(),
                bias_gyro: kf20_bias_g,
                bias_acc: kf20_bias_a,
            },
        );
        state.keyframe_state.insert(
            30,
            KeyframeImuState {
                velocity_world: Vector3::zeros(),
                bias_gyro: Vector3::zeros(),
                bias_acc: Vector3::zeros(),
            },
        );
        state
            .factor_history
            .push(constant_velocity_factor(10, 20, 0.1));
        state
            .factor_history
            .push(constant_velocity_factor(20, 30, 0.1));
        let result = run_local_vi_ba(&mut map, &mut state).expect("BA should run");
        assert_eq!(result.relinearised_factor_count, 2);
        let f0 = &state.factor_history[0];
        let f1 = &state.factor_history[1];
        assert!((f0.delta.bias_gyro_linearisation - kf10_bias_g).norm() < 1.0e-15);
        assert!((f0.delta.bias_acc_linearisation - kf10_bias_a).norm() < 1.0e-15);
        assert!((f1.delta.bias_gyro_linearisation - kf20_bias_g).norm() < 1.0e-15);
        assert!((f1.delta.bias_acc_linearisation - kf20_bias_a).norm() < 1.0e-15);
    }

    #[test]
    fn relinearise_threshold_skips_factors_within_drift() {
        // With a generous threshold (`(1.0, 10.0)`) no factor's bias drift
        // exceeds it. The counter stays at zero and the linearisation
        // point is unchanged.
        let mut map = build_three_keyframe_map();
        let mut state = OnlineSlamLocalBaState::new(OnlineSlamLocalBaConfig {
            gravity_world: Vector3::zeros(),
            relinearise_imu_factor_bias_thresholds: Some((1.0, 10.0)),
            ..OnlineSlamLocalBaConfig::default()
        });
        for kf_id in [10u64, 20, 30] {
            state.keyframe_state.insert(
                kf_id,
                KeyframeImuState {
                    velocity_world: Vector3::zeros(),
                    bias_gyro: Vector3::new(0.05, -0.03, 0.02),
                    bias_acc: Vector3::new(0.2, -0.15, 0.10),
                },
            );
        }
        state
            .factor_history
            .push(constant_velocity_factor(10, 20, 0.1));
        state
            .factor_history
            .push(constant_velocity_factor(20, 30, 0.1));
        let result = run_local_vi_ba(&mut map, &mut state).expect("BA should run");
        assert_eq!(result.relinearised_factor_count, 0);
        for factor in &state.factor_history {
            assert_eq!(factor.delta.bias_gyro_linearisation, Vector3::zeros());
            assert_eq!(factor.delta.bias_acc_linearisation, Vector3::zeros());
        }
    }

    #[test]
    fn local_vi_ba_state_reset_clears_history() {
        let mut state = OnlineSlamLocalBaState::new(OnlineSlamLocalBaConfig::default());
        let triggered = state.register_new_factor(constant_velocity_factor(10, 20, 0.1));
        assert!(triggered);
        assert_eq!(state.factor_history.len(), 1);
        state.reset();
        assert!(state.factor_history.is_empty());
        assert!(state.keyframe_state.is_empty());
        assert_eq!(state.pending_factors_since_last_trigger, 0);
    }

    #[test]
    fn register_new_factor_respects_trigger_every() {
        let mut state = OnlineSlamLocalBaState::new(OnlineSlamLocalBaConfig {
            trigger_every: 3,
            ..OnlineSlamLocalBaConfig::default()
        });
        assert!(!state.register_new_factor(constant_velocity_factor(10, 20, 0.1)));
        assert!(!state.register_new_factor(constant_velocity_factor(20, 30, 0.1)));
        assert!(state.register_new_factor(constant_velocity_factor(30, 40, 0.1)));
        // Counter not yet reset (caller does that via run_local_vi_ba).
        assert_eq!(state.pending_factors_since_last_trigger, 3);
        // Simulate the caller running the BA (no map / so it returns None
        // but resets the counter as a side effect of being invoked).
        state.pending_factors_since_last_trigger = 0;
        assert!(!state.register_new_factor(constant_velocity_factor(40, 50, 0.1)));
    }

    // ============================================================
    // VIBA2 (inertial-with-scale) unit tests.
    // ============================================================

    /// Build a synthetic constant-velocity trajectory + IMU factors. The
    /// metric ground truth places the body at `x = i * step_metric` for
    /// `i = 0..num_keyframes`, with constant velocity `step_metric / dt`
    /// along `+x` and identity orientation. Returns the map (poses at
    /// `metric_step * scale_factor` to simulate the visual-up-to-scale
    /// frame), the factors (metric, NOT rescaled), and the ground-truth
    /// scale that maps `visual → metric`.
    fn build_monocular_constant_velocity(
        num_keyframes: usize,
        step_metric: f64,
        dt: f64,
        scale_factor: f64,
    ) -> (VisualMap, Vec<ImuPreintegrationFactor>, f64) {
        let mut map = VisualMap::new();
        let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
        map.cameras.insert(camera.id, camera.clone());
        for i in 0..num_keyframes {
            // Visual-frame position: `x = (i * step_metric) / scale_factor`.
            // That way `metric_position = scale_factor * visual_position`.
            let visual_x = (i as f64) * step_metric / scale_factor;
            let center = Vector3::new(visual_x, 0.0, 0.0);
            let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), -center);
            let frame_id = (i as u64) + 1;
            let mut frame = Frame::new(frame_id, camera.id);
            frame.pose = Some(pose);
            let kf = Keyframe {
                frame,
                observations: Vec::new(),
            };
            map.keyframes.insert(frame_id, kf);
        }
        // IMU factors: identity-oriented body, zero gravity, constant
        // velocity along +x of `step_metric / dt` m/s, so
        // Δp = R^T · (p_j - p_i - v_i·Δt - 0) = (step_metric - step_metric) = 0,
        // Δv = R^T · (v_j - v_i - 0) = 0. (The IMU sees no proper
        // acceleration in zero-gravity constant-velocity flight.)
        let mut factors = Vec::new();
        for i in 0..(num_keyframes - 1) {
            let delta = ImuPreintegratedDelta {
                delta_rotation: SO3::identity(),
                delta_velocity: Vector3::zeros(),
                delta_position: Vector3::zeros(),
                delta_time: dt,
                bias_gyro_linearisation: Vector3::zeros(),
                bias_acc_linearisation: Vector3::zeros(),
                j_rotation_bg: nalgebra::Matrix3::zeros(),
                j_velocity_ba: nalgebra::Matrix3::zeros(),
                j_velocity_bg: nalgebra::Matrix3::zeros(),
                j_position_ba: nalgebra::Matrix3::zeros(),
                j_position_bg: nalgebra::Matrix3::zeros(),
                covariance: crate::imu_preintegration::Matrix9::zeros(),
            };
            factors.push(ImuPreintegrationFactor {
                keyframe_id_from: (i as u64) + 1,
                keyframe_id_to: (i as u64) + 2,
                delta,
                gravity_world: Vector3::zeros(),
                weight_position: 1.0,
                weight_velocity: 1.0,
                weight_rotation: 1.0,
            });
        }
        (map, factors, scale_factor)
    }

    #[test]
    fn viba2_with_recover_scale_false_reduces_to_viba1() {
        // Stereo / known-scale path: the wrapper must run a single inner
        // solve at `initial_scale = 1.0` and report `scale_history = [1.0]`.
        let (mut map, factors, _gt_scale) = build_monocular_constant_velocity(3, 0.5, 0.1, 1.0);
        let mut initial_states: BTreeMap<u64, KeyframeImuState> = BTreeMap::new();
        let v_metric_per_visual_unit = 0.5 / 0.1;
        for kf_id in [1u64, 2, 3] {
            initial_states.insert(
                kf_id,
                KeyframeImuState {
                    velocity_world: Vector3::new(v_metric_per_visual_unit, 0.0, 0.0),
                    bias_gyro: Vector3::zeros(),
                    bias_acc: Vector3::zeros(),
                },
            );
        }
        let result = run_viba2_inertial_with_scale(
            &mut map,
            &[1, 2, 3],
            &factors,
            &initial_states,
            &Viba2Config {
                initial_scale: 1.0,
                recover_scale: false,
                ..Viba2Config::default()
            },
        )
        .expect("inner solve must succeed");
        assert_eq!(result.outer_iterations_run, 1);
        assert_eq!(result.scale_history.len(), 1);
        assert!((result.scale - 1.0).abs() < 1e-12);
    }

    #[test]
    fn viba2_unit_scale_synthetic_stream_recovers_near_one() {
        // Monocular ground truth scale = 1.0 (visual frame == metric).
        // Starting from `initial_scale = 1.0` the outer loop should keep
        // the scale near 1.0 across iterations.
        let (mut map, factors, _gt_scale) = build_monocular_constant_velocity(4, 0.5, 0.1, 1.0);
        let mut initial_states: BTreeMap<u64, KeyframeImuState> = BTreeMap::new();
        let v_visual_per_unit = 0.5 / 0.1;
        for kf_id in [1u64, 2, 3, 4] {
            initial_states.insert(
                kf_id,
                KeyframeImuState {
                    velocity_world: Vector3::new(v_visual_per_unit, 0.0, 0.0),
                    bias_gyro: Vector3::zeros(),
                    bias_acc: Vector3::zeros(),
                },
            );
        }
        let result = run_viba2_inertial_with_scale(
            &mut map,
            &[1, 2, 3, 4],
            &factors,
            &initial_states,
            &Viba2Config {
                initial_scale: 1.0,
                recover_scale: true,
                max_outer_iterations: 5,
                scale_tolerance: 1.0e-4,
                ..Viba2Config::default()
            },
        )
        .expect("inner solve must succeed");
        // Even with zero-gravity / zero-Δp factors the kinematic
        // denominator may degenerate, in which case the wrapper preserves
        // the initial scale and reports `outer_iterations_run = 1`. Both
        // outcomes are valid here — the assertion is the absence of
        // catastrophic scale drift.
        assert!(result.scale.is_finite());
        assert!(
            (result.scale - 1.0).abs() < 1.0,
            "scale must remain bounded near 1.0; got {}",
            result.scale
        );
    }

    #[test]
    fn estimate_scale_from_factors_recovers_known_scale() {
        // Synthetic scene with a closed-form expected scale: build a
        // body in identity-orientation, +y-down gravity 9.81, moving from
        // origin to (1.0 metric, 0, 0) over Δt = 1.0 s with v_i = 0.
        // Then metric `p_j = (1.0, 0, 0)`, `Δp_metric = R^T · (1 - 0 - 0
        //  · 1 - 0.5 · g · 1²) = (1, -4.905, 0)`. With visual scale `s_gt`,
        // visual `p_j = (1/s_gt, 0, 0)`. The estimator's least-squares
        // recovers `s_gt`.
        let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut map = VisualMap::new();
        map.cameras.insert(camera.id, camera.clone());
        let s_gt = 2.5;
        for (id, center) in [
            (1u64, Vector3::new(0.0, 0.0, 0.0)),
            (2u64, Vector3::new(1.0 / s_gt, 0.0, 0.0)),
        ] {
            let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), -center);
            let mut frame = Frame::new(id, camera.id);
            frame.pose = Some(pose);
            map.keyframes.insert(
                id,
                Keyframe {
                    frame,
                    observations: Vec::new(),
                },
            );
        }
        let gravity = Vector3::new(0.0, 9.81, 0.0);
        // The metric Δp under v_i = 0, p_j = (1, 0, 0): Δp = (1, -4.905, 0).
        let delta = ImuPreintegratedDelta {
            delta_rotation: SO3::identity(),
            delta_velocity: -gravity,
            delta_position: Vector3::new(1.0, 0.0, 0.0) - 0.5 * gravity,
            delta_time: 1.0,
            bias_gyro_linearisation: Vector3::zeros(),
            bias_acc_linearisation: Vector3::zeros(),
            j_rotation_bg: nalgebra::Matrix3::zeros(),
            j_velocity_ba: nalgebra::Matrix3::zeros(),
            j_velocity_bg: nalgebra::Matrix3::zeros(),
            j_position_ba: nalgebra::Matrix3::zeros(),
            j_position_bg: nalgebra::Matrix3::zeros(),
            covariance: crate::imu_preintegration::Matrix9::zeros(),
        };
        let factor = ImuPreintegrationFactor {
            keyframe_id_from: 1,
            keyframe_id_to: 2,
            delta,
            gravity_world: gravity,
            weight_position: 1.0,
            weight_velocity: 1.0,
            weight_rotation: 1.0,
        };
        let mut states: BTreeMap<u64, KeyframeImuState> = BTreeMap::new();
        for id in [1u64, 2] {
            states.insert(
                id,
                KeyframeImuState {
                    velocity_world: Vector3::zeros(),
                    bias_gyro: Vector3::zeros(),
                    bias_acc: Vector3::zeros(),
                },
            );
        }
        let s_recovered =
            estimate_scale_from_factors(&map, &states, &[factor]).expect("denom not degenerate");
        // The closed-form least-squares minimises `‖s·a - b‖²` over both
        // x and y axes. With a = R · (p_j - 0 - 0) = (1/s_gt, 0, 0) and
        // b = Δp + R · 0.5 · g · dt² = (1, 0, 0), the LS gives:
        //   s = aᵀb / aᵀa = (1/s_gt · 1) / (1/s_gt)² = s_gt.
        assert!(
            (s_recovered - s_gt).abs() / s_gt < 0.05,
            "expected {} ± 5 %, got {}",
            s_gt,
            s_recovered
        );
    }

    #[test]
    fn estimate_scale_returns_none_on_degenerate_kinematics() {
        // Coincident keyframes + zero velocity → `a = 0` for every
        // factor → denominator below tolerance → `None`.
        let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut map = VisualMap::new();
        map.cameras.insert(camera.id, camera.clone());
        for id in [1u64, 2] {
            let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
            let mut frame = Frame::new(id, camera.id);
            frame.pose = Some(pose);
            map.keyframes.insert(
                id,
                Keyframe {
                    frame,
                    observations: Vec::new(),
                },
            );
        }
        let delta = ImuPreintegratedDelta::identity();
        let factor = ImuPreintegrationFactor {
            keyframe_id_from: 1,
            keyframe_id_to: 2,
            delta: ImuPreintegratedDelta {
                delta_time: 1.0,
                ..delta
            },
            gravity_world: Vector3::zeros(),
            weight_position: 1.0,
            weight_velocity: 1.0,
            weight_rotation: 1.0,
        };
        let mut states: BTreeMap<u64, KeyframeImuState> = BTreeMap::new();
        for id in [1u64, 2] {
            states.insert(
                id,
                KeyframeImuState {
                    velocity_world: Vector3::zeros(),
                    bias_gyro: Vector3::zeros(),
                    bias_acc: Vector3::zeros(),
                },
            );
        }
        let outcome = estimate_scale_from_factors(&map, &states, &[factor]);
        assert!(outcome.is_none(), "expected None on degenerate kinematics");
    }
}
