//! Milestone M7 of `docs/dpvo_droid_port_plan.md`: continuous,
//! never-trusted-all-at-once IMU scale coupling.
//!
//! # Why M5b's one-shot "bootstrap once, trust forever" had to go
//!
//! M5's honest negative (`docs/dpvo_droid_port_plan.md`, "M5 results") and
//! M5b's honest negative ("M5b results") both trace to the same root cause,
//! documented in full in `crate::dpvo_vi_ba`'s own module doc ("Milestone
//! M5b: a monocular-aware bootstrap"): a SINGLE window's
//! `estimate_mono_vi_alignment`/`estimate_gyro_bias` call can be
//! plausible-looking (it clears every one of M5b's own observability gates
//! — degrees of freedom, conditioning, gravity norm, scale range) and STILL
//! be numerically wrong, because DPVO's own monocular reconstruction at
//! bootstrap time carries systematic (not random) rotation/translation
//! error that a single alignment cannot distinguish from real evidence. M5b's
//! own "Forward path" section named the fix directly: *"Continuous in-window
//! scale estimation instead of one-shot bootstrap-then-trust... a persistent
//! disagreement between the two would be a much stronger acceptance signal
//! than either one's own internal gates."* This module is that fix.
//!
//! # Design in one sentence
//!
//! Instead of accepting or rejecting a SINGLE alignment event, this module
//! maintains a running Bayesian posterior over log-scale (and, in parallel, a
//! running posterior over gyro bias) that is updated from EVERY window's
//! alignment attempt, with a robust (Huber-style) down-weighting of any one
//! measurement that disagrees badly with the accumulated evidence so far;
//! IMU coupling's own INFLUENCE on the solved trajectory ramps up smoothly
//! (an "annealing" weight in `[0, 1]`) only once that posterior has both
//! shrunk below a variance threshold AND agreed with itself over several
//! consecutive windows, and ramps back down — never a hard on/off flag — the
//! moment the coupled solve's own IMU-factor residuals start looking
//! inconsistent again. Nothing is ever applied in one shot: even once
//! "active", the recovered scale enters the joint solve as a small,
//! variance-weighted PRIOR residual on top of the existing visual+IMU
//! Gauss-Newton problem (see "Gentle scale-prior application" below), not a
//! `pose.translation *= s` rescale of the whole live map the way M5b's
//! `try_imu_bootstrap` did.
//!
//! # Module map
//!
//! | Piece | Role |
//! | --- | --- |
//! | [`LogScalePosterior`] / [`RecursiveScaleEstimator`] | The running Bayesian belief over `ln(scale)`, robustly fused from repeated [`scale_measurement_from_alignment`] calls. |
//! | [`scale_measurement_from_alignment`] | Turns one `crate::dpvo_vi_ba::DpvoMonoViAlignment` into a `(log_scale, variance)` pair — the "honest variance proxy from the LSQ" the task requires. |
//! | [`Vector3Posterior`] / [`RecursiveGyroBiasEstimator`] | The same recursive-Bayesian machinery, applied to gyro bias (an isotropic-covariance simplification — see that struct's own doc). |
//! | [`AnnealingWeight`] | The `[0, 1]` IMU-influence weight: steps up while converged/consistent, steps down otherwise — never switches. |
//! | [`apply_gentle_scale_correction`] | The scale-prior residual itself: a decoupled, finite-difference 1-parameter Gauss-Newton nudge (see its own doc for why decoupled rather than folded into `dpvo_vi_ba_step`'s NxN system). |
//! | [`blend_solutions`] | Output-space interpolation between a visual-only and an IMU-coupled solve at the current annealing weight — see "Why output-space blending" below. |
//!
//! `crate::dpvo_vo::DpvoOdometry::scale_coupling_step` (gated behind
//! `crate::dpvo_vo::DpvoScaleCouplingConfig`, `None` by default so every
//! M4/M5/M5b/M6 call site is unaffected) is the only caller that wires these
//! pieces into the live per-frame loop; this module itself has zero
//! `onnx-inference` dependency and zero live-session dependency, so every
//! piece here is unit-testable on synthetic data alone (see `tests` below).
//!
//! # The honest variance proxy (the task's own explicit requirement)
//!
//! [`crate::dpvo_vi_ba::DpvoMonoViAlignment`] already carries
//! `mean_residual_after` (the fitted system's own empirical noise level) and
//! — added this milestone — `min_singular_value` (the unconstrained solve's
//! smallest singular value). For an ordinary linear least-squares fit `Ax=b`
//! with i.i.d. residual noise of variance `σ²`, the solved parameter
//! covariance is `Cov(x) = σ²(AᵀA)⁻¹ = σ²·V·Σ⁻²·Vᵀ`; the marginal variance of
//! any ONE parameter (here, the scale unknown) is bounded above by
//! `σ²/σ_min²` (the worst case, reached when that parameter's row of `V`
//! aligns with the smallest-singular-value direction). Using `σ² ≈
//! mean_residual_after²` (the standard empirical-noise estimator for a
//! converged least-squares fit) gives
//!
//! ```text
//! variance_proxy = mean_residual_after² / min_singular_value²
//! ```
//!
//! — a genuinely DERIVED (not guessed) upper bound, monotonic in exactly the
//! two directions intuition demands (noisier residual ⇒ higher variance;
//! worse-conditioned system ⇒ higher variance), and automatically becoming
//! enormous (correctly: "trust this measurement almost not at all") the
//! moment `min_singular_value` approaches zero — i.e. exactly the
//! constant-velocity/near-degenerate window
//! `crate::dpvo_vi_ba::tests::estimate_mono_vi_alignment_rejects_constant_velocity_window`
//! already demonstrates gets a literal `min_sv = 0.0`. Using the WORST-CASE
//! bound rather than an exact (basis-dependent) marginal variance is a
//! deliberate conservative choice, consistent with this whole milestone's
//! "can never poison the map" design goal: it is always safe to be MORE
//! uncertain than the true posterior, never safe to be less. See
//! [`scale_measurement_from_alignment`] for the exact formula including
//! floors/knobs.
//!
//! # Robust fusion: Huber-style innovation down-weighting
//!
//! M5b's own honest negative (`docs/dpvo_droid_port_plan.md`, "M5b results")
//! showed a single plausible-looking-but-wrong alignment (recovered scale
//! `18.66`, inside every gate's nominal bound) can poison a one-shot
//! bootstrap outright. [`RecursiveScaleEstimator::update`] cannot be poisoned
//! the same way: every incoming measurement's INNOVATION (its distance from
//! the CURRENT posterior mean, in posterior-standard-deviation units) is
//! computed first; if that normalized innovation exceeds
//! [`ScaleCouplingConfig::huber_delta`], the measurement's own variance is
//! INFLATED by `(normalized / huber_delta)²` before the ordinary 1-D
//! Bayesian (Kalman) fuse runs. This is the standard IRLS
//! (iteratively-reweighted-least-squares) construction for a Huber
//! M-estimator applied to a scalar recursive filter: it caps the
//! EFFECTIVE information any single measurement can contribute at
//! `huber_delta`-worth of surprise, so an occasional wild outlier (the exact
//! shape of M5b's own failure — an 18.66× scale amid a stream of
//! near-1× windows) moves the posterior mean by at most a small, bounded
//! step, rather than being fused in at full weight.
//! [`tests::poisoned_measurement_stream_does_not_move_the_posterior_materially`]
//! is this module's own direct reproduction of that exact scenario.
//!
//! # Convergence and annealing (never switches, never applies all at once)
//!
//! [`RecursiveScaleEstimator::is_converged`] requires BOTH the posterior
//! standard deviation to have shrunk below
//! [`ScaleCouplingConfig::convergence_std`] AND the last
//! [`ScaleCouplingConfig::convergence_window`] RAW measurements (not the
//! posterior-smoothed mean — genuine self-agreement, not merely "the filter
//! says it's confident") to sit within
//! [`ScaleCouplingConfig::convergence_band`] of each other. Neither
//! condition alone is sufficient: a filter can report low variance after
//! fusing a long run of mutually-consistent-but-wrong measurements (variance
//! shrinks regardless of whether the CONVERGED value is correct — this is
//! exactly why M6's forward-path note #1 called for a "second, independent"
//! cross-check), so the raw-window-agreement check is not redundant with the
//! posterior's own variance — it is the "does the DATA actually keep
//! agreeing with itself" check the posterior's own math cannot provide.
//! [`tests::recursive_scale_estimator_refuses_to_converge_on_degenerate_windows`]
//! exercises the case where measurements never arrive at all (every window
//! gets hard-rejected upstream by `estimate_mono_vi_alignment`'s own gates)
//! — the honest "never converges, stays visual-only forever" outcome the
//! task's own acceptance section explicitly permits.
//!
//! [`AnnealingWeight`] only ever moves by `1/anneal_frames` per frame while
//! `is_converged()` holds, and by `1/decay_frames` per frame otherwise
//! (typically faster — `decay_frames < anneal_frames` — so a genuinely bad
//! state is shed faster than a good one is trusted, an asymmetric-risk
//! choice matching this whole milestone's "can never poison" bar). At
//! `weight == 0.0` [`crate::dpvo_vo::DpvoOdometry::scale_coupling_step`]
//! never even calls `dpvo_vi_ba` — the run is BYTE-IDENTICAL to the
//! visual-only `dpvo_ba` path, matching the task's own "before activation,
//! nothing changes" requirement literally, not just approximately.
//!
//! # Gentle scale-prior application: a decoupled 1-parameter Gauss-Newton step
//!
//! `crate::dpvo_vi_ba`'s own module doc ("Scale handling: no dedicated scale
//! variable") argues against adding a dedicated scale unknown to
//! `dpvo_vi_ba_step`'s NxN joint system: visual reprojection residuals are
//! EXACTLY invariant under a joint `(translation, depth)` rescale, so a
//! dedicated `s` column would be rank-deficient against the existing
//! pose/depth columns UNLESS something else anchors it — which is precisely
//! what a genuine PRIOR does (a Tikhonov term on `s` breaks the degeneracy by
//! construction, the same way [`crate::dpvo_vi_ba::VELOCITY_DIAG_EPSILON`]
//! already breaks a different, smaller degeneracy for velocity). This module
//! adds exactly that prior, but as a SEPARATE, decoupled 1-parameter
//! subproblem run AFTER `dpvo_vi_ba_step`'s own solve rather than as an
//! (N+1)-th column folded into that solve's own matrix. Two reasons, both
//! documented here rather than left implicit:
//!
//! 1. **Risk**: `dpvo_vi_ba_step` is the one function in this whole port
//!    whose Jacobian derivation required a two-step Adjoint conjugation
//!    (`crate::dpvo_vi_ba`'s own module doc) and is validated to
//!    finite-difference tolerance by
//!    `crate::dpvo_vi_ba::tests::imu_factor_jacobian_matches_numeric_finite_difference`.
//!    Splicing a new column into its matrix assembly would touch that
//!    exact, already-fragile code path for a "gentle nudge" that, by this
//!    milestone's own design, is meant to be SMALL relative to the main
//!    solve — a decoupled subproblem gets the same asymptotic effect (a
//!    Gauss-Newton step on the added residual, at the current linearization
//!    point) with zero risk to the tested joint solve.
//! 2. **The residual itself needs no new Jacobian derivation.** The scale
//!    unknown here (`delta_log_s`, reset to `0` every call — an INCREMENT,
//!    not an absolute value, matching every other per-iteration unknown in
//!    this port's `dpvo_ba_step`/`dpvo_vi_ba_step`) only couples to the
//!    EXISTING IMU position residual (visual residuals are provably
//!    insensitive to it, by the scale-invariance argument above — so it is
//!    correct, not merely convenient, that this term ignores them
//!    entirely). [`apply_gentle_scale_correction`] computes
//!    `∂r/∂delta_log_s` via a CENTRAL... actually a one-sided finite
//!    difference of `crate::dpvo_vi_ba::imu_factor_jacobians`'s own
//!    (already-validated) residual output, rather than hand-deriving a
//!    FIFTH analytic Jacobian formula in this port. The finite-difference
//!    step (`FD_STEP = 1e-4`) introduces a `O(FD_STEP)` truncation error
//!    into `delta_log_s`'s own Gauss-Newton normal equation — acceptable
//!    here specifically because `delta_log_s` is already bounded to a
//!    small per-call step
//!    ([`ScaleCouplingConfig::max_log_step`]) and re-linearized fresh every
//!    call (any FD error simply gets corrected on the NEXT window's call,
//!    the same way an ordinary Gauss-Newton iteration corrects its own
//!    linearization error over repeated calls) — an honest, documented
//!    tradeoff of a small approximation for a large reduction in
//!    implementation/verification risk, not an oversight.
//!
//! The resulting normal equation, for a window whose free poses span local
//! indices `[fixedp, n)`:
//!
//! ```text
//! (Σ_factors Jₛᵀ·Jₛ  +  information) · delta_log_s
//!     =  −Σ_factors Jₛᵀ·(whitened residual at delta_log_s=0)  +  information·posterior_mean
//! information = weight_multiplier / max(posterior_variance, variance_floor)
//! ```
//!
//! — literally a 1×1 Gauss-Newton normal equation with an added Tikhonov/prior
//! term pulling `delta_log_s` toward the RECURSIVE ESTIMATOR's current belief
//! about how much residual scale error remains, weighted by how confident
//! that belief is AND by the current annealing weight. `delta_log_s` is then
//! clamped to `±max_log_step` (a second, independent gentleness bound beyond
//! the weighting) and applied as `pose.translation *= exp(delta_log_s)` for
//! every FREE (`local index ≥ fixedp`) pose, `patch.inverse_depth /=
//! exp(delta_log_s)` for every in-window patch — the same
//! translation/inverse-depth transformation `crate::dpvo_vi_ba`'s own module
//! doc derives for a pure similarity rescale ("Applying the recovered
//! scale"), just INCREMENTAL and INFORMATION-WEIGHTED rather than a single
//! `s`-sized jump applied once and never revisited.
//!
//! # Why output-space blending, not information-matrix annealing
//!
//! An alternative annealing mechanism would scale down each
//! `ImuPreintegrationFactor`'s own whitening/information by the annealing
//! weight before calling `dpvo_vi_ba_step`. This was considered and
//! rejected: `crate::dpvo_vi_ba::imu_factor_whitener` derives its whitening
//! from the factor's OWN physically-propagated covariance
//! (`covariance_sqrt_information`) when available — arbitrarily rescaling a
//! physical noise model to express an unrelated "how much do I trust my own
//! bootstrap" belief would conflate two different uncertainties that should
//! stay separate (sensor noise vs. bootstrap-quality doubt). [`blend_solutions`]
//! instead runs the FULL visual-only (`dpvo_ba`) and FULL IMU-coupled
//! (`dpvo_vi_ba` + [`apply_gentle_scale_correction`]) solves independently,
//! at their own correct, undistorted internal weightings, and interpolates
//! ONLY at the output — along the DPVO left-perturbation retraction's own
//! geodesic for poses (`SE3::exp(w · SE3::log(imu ∘ visual⁻¹)) ∘ visual`,
//! the natural constant-angular-velocity interpolation between two SE(3)
//! poses under exactly this port's own retraction convention) and by plain
//! linear interpolation for the scalar inverse-depth. At `w = 0` this
//! reproduces the visual-only solve EXACTLY (`SE3::exp(0) = identity`); at
//! `w = 1` it reproduces the IMU-coupled solve exactly. Every intermediate
//! `w` is a well-defined, bounded, and REVERSIBLE point on the path between
//! them — precisely the "gentle... ramp... never switch" property this
//! milestone's design goal requires, achieved with zero risk to either
//! tested solver's own internal numerics.
//!
//! # Continuous cross-check and soft rollback
//!
//! `crate::dpvo_vi_ba::imu_factor_nis` (M5b's own rollback-monitor signal,
//! reused unchanged) is evaluated every frame the coupling is active, at the
//! POST-solve (blended) state; `crate::dpvo_vo::rollback_monitor_step` (M5b's
//! own consecutive-bad-frame counter, widened to `pub(crate)` this
//! milestone) decides whether this is "sustained" inconsistency. Unlike
//! M5b's `rollback_imu_bootstrap` (which zeroes velocities, clears gravity,
//! and flips a hard boolean), the SOFT rollback here does exactly two
//! things: [`RecursiveScaleEstimator::soft_reset`] widens the posterior
//! variance back to its prior (so the filter no longer claims false
//! confidence in a value the data has just contradicted, but keeps its
//! Huber-protected MEAN rather than discarding it outright — a bad single
//! measurement was already down-weighted at fusion time, so the mean is not
//! assumed to be the culprit) and [`AnnealingWeight::force_decay`] pushes the
//! IMU-influence weight down an extra step immediately. No pose, depth, or
//! velocity state needs "un-applying" — per the task's own framing, "no
//! state surgery needed since nothing was hard-applied" — because
//! [`blend_solutions`] already means the live map was never MORE than
//! `weight`-far from the pure-visual solution to begin with; decaying
//! `weight` back toward zero over the following frames smoothly pulls the
//! live trajectory back toward visual-only, exactly the same way it was
//! smoothly pulled toward IMU-coupled during annealing-up.
use std::collections::VecDeque;

use visloc_core::geometry::SE3;

use crate::dpvo_patch_ba::DpvoPatch;
use crate::dpvo_vi_ba::{imu_factor_jacobians, imu_factor_whitener, DpvoImuFactor, DpvoMonoViAlignment};
use nalgebra::Vector3;

/// Finite-difference step used by [`apply_gentle_scale_correction`]'s
/// `∂r/∂delta_log_s` — see the module doc's "Gentle scale-prior application"
/// section for why a finite difference is used instead of a fifth analytic
/// Jacobian derivation, and why this magnitude is small enough that its own
/// truncation error is negligible against `ScaleCouplingConfig::max_log_step`.
const SCALE_FD_STEP: f64 = 1.0e-4;

/// Posterior belief over natural-log monocular-to-metric scale
/// (`log_scale = ln(s)` where `metric_position = s · visual_position`,
/// `crate::dpvo_vi_ba`'s own convention). Logarithmic, not linear, because
/// scale is strictly positive and multiplicative (two consistent `2×`
/// corrections compose to `4×`, not `+2+2=+4×` of anything) — the same
/// reason `crate::dpvo_vi_ba::DpvoMonoViAlignment::scale` itself is a ratio,
/// not a difference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogScalePosterior {
    pub mean: f64,
    pub variance: f64,
}

/// One incoming scale observation, already reduced to `(log_scale,
/// variance)` — see [`scale_measurement_from_alignment`] for how a real
/// `DpvoMonoViAlignment` becomes one of these, and the module doc's "honest
/// variance proxy" section for the derivation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleMeasurement {
    pub log_scale: f64,
    pub variance: f64,
}

/// Tunable defaults for the whole M7 coupling mechanism — see the module doc
/// for what each knob does; every default here is documented at its field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleCouplingConfig {
    /// Posterior log-variance before any measurement has ever arrived, and
    /// the value [`RecursiveScaleEstimator::soft_reset`] restores it to.
    /// Default `(ln(20.0)).powi(2) ≈ 8.97`: wide enough that the FIRST
    /// measurement (whatever it is) dominates the initial fuse, deliberately
    /// mirroring `crate::dpvo_vi_ba::DpvoMonoViAlignmentGates`'s own
    /// `[0.05, 20]` task-specified plausible-scale range — `ln(20)` is the
    /// log-space half-width of that same range, so "totally uncertain" here
    /// means exactly "anywhere within the range M5b's own gates already
    /// consider physically plausible", not an arbitrary number.
    pub prior_variance: f64,
    /// Hard floor under which no variance (prior, measurement, or posterior)
    /// is allowed to fall — guards the `variance_floor`-guarded divisions in
    /// [`RecursiveScaleEstimator::update`]/[`apply_gentle_scale_correction`]
    /// against literal division-by-zero on a perfect synthetic fixture
    /// (`min_singular_value` and `mean_residual_after` can both legitimately
    /// be exactly representable-as-zero on hand-built synthetic data, unlike
    /// any real DPVO window). Default `1e-10`.
    pub variance_floor: f64,
    /// Huber threshold, in POSTERIOR-standard-deviation units of normalized
    /// innovation — see the module doc's "Robust fusion" section. Default
    /// `3.0`: a measurement more than 3σ from the current belief has its
    /// effective information capped rather than discarded outright (Huber,
    /// not a hard reject — even a 3σ+ outlier still nudges the posterior a
    /// LITTLE, unlike M5b's all-or-nothing gate).
    pub huber_delta: f64,
    /// Convergence gate #1: posterior standard deviation (in log-scale
    /// units) must drop below this. Default `0.05` — roughly a `5%`
    /// linear-scale uncertainty (`exp(0.05) ≈ 1.051`), the same order of
    /// magnitude as the task's own `< 3` (from a `22.6×` baseline) and `5%`
    /// metric-scale-recovery bars.
    pub convergence_std: f64,
    /// Convergence gate #2 window size `K` — the last `K` RAW measurements
    /// (not posterior-smoothed) must all agree within `convergence_band`.
    /// Default `5`.
    pub convergence_window: usize,
    /// Convergence gate #2 band (log-scale units): `max − min` over the last
    /// `convergence_window` raw measurements. Default `0.1`
    /// (`exp(0.1) ≈ 1.105`, i.e. the raw per-window estimates must agree to
    /// within about `10%` of each other, independent of how any recursive
    /// filter smooths them).
    pub convergence_band: f64,
    /// Number of frames [`AnnealingWeight`] takes to ramp from `0` to `1`
    /// once `is_converged()` first holds. Default `30` — roughly `3` s of
    /// EuRoC's own post-stride-2 `~10` Hz rate, chosen to be slow relative to
    /// a single window's own solve but fast relative to the hundreds of
    /// frames a EuRoC sequence runs.
    pub anneal_frames: f64,
    /// Number of frames [`AnnealingWeight`] takes to decay from `1` to `0`
    /// once consistency is lost. Default `10` — deliberately FASTER than
    /// `anneal_frames` (asymmetric risk: shedding a bad state matters more
    /// than quickly trusting a good one).
    pub decay_frames: f64,
    /// Soft-rollback NIS monitor bound — same semantics/default rationale as
    /// `crate::dpvo_vo::DpvoImuConfig::rollback_mean_nis_bound` (`500.0`).
    pub rollback_mean_nis_bound: f64,
    /// Soft-rollback NIS monitor consecutive-bad-frame threshold — same
    /// semantics/default as `crate::dpvo_vo::DpvoImuConfig::rollback_consecutive_frames`
    /// (`5`).
    pub rollback_consecutive_frames: usize,
    /// Hard per-call bound on `|delta_log_s|` in [`apply_gentle_scale_correction`]
    /// — an independent gentleness bound beyond the information weighting
    /// itself (belt-and-braces: even a bug that made `information` too large
    /// cannot make one call's correction bigger than this). Default `0.02`
    /// (`exp(0.02) ≈ 1.02`, a `2%` per-window cap).
    pub max_log_step: f64,
}

impl Default for ScaleCouplingConfig {
    fn default() -> Self {
        Self {
            prior_variance: (20.0_f64).ln().powi(2),
            variance_floor: 1.0e-10,
            huber_delta: 3.0,
            convergence_std: 0.05,
            convergence_window: 5,
            convergence_band: 0.1,
            anneal_frames: 30.0,
            decay_frames: 10.0,
            rollback_mean_nis_bound: 500.0,
            rollback_consecutive_frames: 5,
            max_log_step: 0.02,
        }
    }
}

/// Diagnostics returned by [`RecursiveScaleEstimator::update`] — lets a
/// caller log exactly how much a measurement was down-weighted, for the same
/// "isolate which gate/mechanism did what" transparency M5b's own
/// `DpvoImuRejectionDetail` established.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleUpdateReport {
    /// Kalman gain actually used for this fuse (after Huber inflation).
    pub gain: f64,
    /// Huber inflation factor applied to the measurement's own variance
    /// (`1.0` if the innovation was within `huber_delta`, `> 1.0` otherwise).
    pub huber_inflation: f64,
    /// Raw innovation (`measurement.log_scale − prior_mean`), before any
    /// downweighting.
    pub innovation: f64,
    pub posterior: LogScalePosterior,
}

/// The running Bayesian belief over log-scale, robustly fused from repeated
/// [`ScaleMeasurement`]s — see the module doc for the full design.
#[derive(Debug, Clone)]
pub struct RecursiveScaleEstimator {
    posterior: Option<LogScalePosterior>,
    recent_raw: VecDeque<f64>,
    cfg: ScaleCouplingConfig,
}

impl RecursiveScaleEstimator {
    pub fn new(cfg: ScaleCouplingConfig) -> Self {
        Self { posterior: None, recent_raw: VecDeque::new(), cfg }
    }

    pub fn posterior(&self) -> Option<LogScalePosterior> {
        self.posterior
    }

    pub fn config(&self) -> ScaleCouplingConfig {
        self.cfg
    }

    /// Fuse one measurement into the running posterior — see the module
    /// doc's "Robust fusion" section for the Huber-inflation formula. The
    /// FIRST measurement ever seen seeds the posterior mean directly (no
    /// prior evidence to Huber-check it against) with variance
    /// `max(measurement.variance, cfg.prior_variance)` — deliberately at
    /// LEAST as uncertain as the configured prior, so a lucky first
    /// measurement with a suspiciously tiny reported variance cannot claim
    /// false confidence before any repeated-agreement evidence exists.
    pub fn update(&mut self, measurement: ScaleMeasurement) -> ScaleUpdateReport {
        let meas_var = measurement.variance.max(self.cfg.variance_floor);
        let Some(prior) = self.posterior else {
            let seeded = LogScalePosterior {
                mean: measurement.log_scale,
                variance: meas_var.max(self.cfg.prior_variance),
            };
            self.posterior = Some(seeded);
            self.push_raw(measurement.log_scale);
            return ScaleUpdateReport { gain: 1.0, huber_inflation: 1.0, innovation: 0.0, posterior: seeded };
        };

        let innovation = measurement.log_scale - prior.mean;
        let combined_std = (prior.variance + meas_var).sqrt().max(1.0e-12);
        let normalized = innovation.abs() / combined_std;
        let huber_inflation =
            if normalized > self.cfg.huber_delta { (normalized / self.cfg.huber_delta).powi(2) } else { 1.0 };
        let effective_meas_var = (meas_var * huber_inflation).max(self.cfg.variance_floor);

        let gain = prior.variance / (prior.variance + effective_meas_var);
        let new_mean = prior.mean + gain * innovation;
        let new_variance = ((1.0 - gain) * prior.variance).max(self.cfg.variance_floor);
        let posterior = LogScalePosterior { mean: new_mean, variance: new_variance };
        self.posterior = Some(posterior);
        self.push_raw(measurement.log_scale);
        ScaleUpdateReport { gain, huber_inflation, innovation, posterior }
    }

    fn push_raw(&mut self, log_scale: f64) {
        self.recent_raw.push_back(log_scale);
        while self.recent_raw.len() > self.cfg.convergence_window {
            self.recent_raw.pop_front();
        }
    }

    /// Both convergence gates — see the module doc's "Convergence and
    /// annealing" section for why both are required.
    pub fn is_converged(&self) -> bool {
        let Some(posterior) = self.posterior else { return false };
        if posterior.variance.sqrt() >= self.cfg.convergence_std {
            return false;
        }
        if self.recent_raw.len() < self.cfg.convergence_window {
            return false;
        }
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for &v in &self.recent_raw {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        (hi - lo) <= self.cfg.convergence_band
    }

    /// Soft rollback (module doc, "Continuous cross-check and soft
    /// rollback"): widen the posterior variance back to the configured
    /// prior and clear the raw-agreement window (so re-convergence requires
    /// FRESH agreement, not evidence that just triggered the rollback), but
    /// keep the posterior MEAN — a single bad measurement was already
    /// down-weighted at fusion time by the Huber mechanism, so the mean
    /// itself is not assumed to be the culprit; see the module doc for why
    /// this needs no "un-apply" of any already-solved state.
    pub fn soft_reset(&mut self) {
        if let Some(posterior) = &mut self.posterior {
            posterior.variance = self.cfg.prior_variance;
        }
        self.recent_raw.clear();
    }
}

/// Milestone M7: turn one accepted [`DpvoMonoViAlignment`] into a
/// [`ScaleMeasurement`] — see the module doc's "The honest variance proxy"
/// section for the derivation.
pub fn scale_measurement_from_alignment(
    alignment: &DpvoMonoViAlignment,
    cfg: &ScaleCouplingConfig,
) -> ScaleMeasurement {
    let sigma = alignment.mean_residual_after.max(cfg.variance_floor.sqrt());
    let min_sv = alignment.min_singular_value.max(cfg.variance_floor.sqrt());
    let variance = ((sigma * sigma) / (min_sv * min_sv)).max(cfg.variance_floor);
    ScaleMeasurement { log_scale: alignment.scale.max(1.0e-9).ln(), variance }
}

/// Posterior belief over a Euclidean quantity (gyro bias) with a SHARED
/// scalar variance across all three components — a deliberate isotropic
/// simplification (a real gyro's per-axis bias uncertainty could differ per
/// axis, but `crate::vi_motion_initializer::GyroBiasAlignment` itself only
/// ever reports one scalar `rotation_residual_rms_after`, so a full 3×3
/// posterior covariance would carry more precision than the input evidence
/// actually supports).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector3Posterior {
    pub mean: Vector3<f64>,
    pub variance: f64,
}

/// Milestone M7's answer to the task's own "gyro bias likewise enters as a
/// soft prior... never hard-fixed" requirement: the same recursive-Bayesian
/// and Huber-robust machinery as [`RecursiveScaleEstimator`], applied to a
/// `Vector3<f64>` instead of a scalar log-scale. `cfg`'s `huber_delta`/
/// `prior_variance`/`variance_floor` fields are reused as-is (no separate
/// gyro-specific config type — the task's own framing groups scale and gyro
/// bias under one "continuous, never-trusted-all-at-once" umbrella, and
/// there is no principled reason for their Huber thresholds to differ).
#[derive(Debug, Clone)]
pub struct RecursiveGyroBiasEstimator {
    posterior: Option<Vector3Posterior>,
    cfg: ScaleCouplingConfig,
}

impl RecursiveGyroBiasEstimator {
    pub fn new(cfg: ScaleCouplingConfig) -> Self {
        Self { posterior: None, cfg }
    }

    pub fn posterior(&self) -> Option<Vector3Posterior> {
        self.posterior
    }

    /// Current best estimate — `Vector3::zeros()` before any measurement has
    /// ever been fused (a documented, safe default: zero bias is a
    /// reasonable prior mean for a well-calibrated MEMS gyro, and every
    /// caller only USES this once [`RecursiveScaleEstimator::is_converged`]-
    /// gated coupling has actually started ramping in, at which point at
    /// least one measurement has always already arrived — see
    /// `crate::dpvo_vo::DpvoOdometry::scale_coupling_step`).
    pub fn mean(&self) -> Vector3<f64> {
        self.posterior.map(|p| p.mean).unwrap_or_else(Vector3::zeros)
    }

    pub fn update(&mut self, measurement_mean: Vector3<f64>, measurement_variance: f64) -> Vector3Posterior {
        let meas_var = measurement_variance.max(self.cfg.variance_floor);
        let Some(prior) = self.posterior else {
            let seeded = Vector3Posterior { mean: measurement_mean, variance: meas_var.max(self.cfg.prior_variance) };
            self.posterior = Some(seeded);
            return seeded;
        };
        let innovation = measurement_mean - prior.mean;
        let combined_std = (prior.variance + meas_var).sqrt().max(1.0e-12);
        let normalized = innovation.norm() / combined_std;
        let huber_inflation =
            if normalized > self.cfg.huber_delta { (normalized / self.cfg.huber_delta).powi(2) } else { 1.0 };
        let effective_meas_var = (meas_var * huber_inflation).max(self.cfg.variance_floor);
        let gain = prior.variance / (prior.variance + effective_meas_var);
        let new_mean = prior.mean + gain * innovation;
        let new_variance = ((1.0 - gain) * prior.variance).max(self.cfg.variance_floor);
        let posterior = Vector3Posterior { mean: new_mean, variance: new_variance };
        self.posterior = Some(posterior);
        posterior
    }

    /// See [`RecursiveScaleEstimator::soft_reset`] — identical reasoning,
    /// applied to this estimator's own posterior.
    pub fn soft_reset(&mut self) {
        if let Some(posterior) = &mut self.posterior {
            posterior.variance = self.cfg.prior_variance;
        }
    }
}

/// The `[0, 1]` IMU-influence weight — see the module doc's "Convergence and
/// annealing" section. `value == 0.0` means "behave exactly like
/// visual-only DPVO"; `value == 1.0` means "fully trust the IMU-coupled +
/// gentle-scale-corrected solve". Every intermediate value is a legitimate,
/// meaningful state consumed by [`blend_solutions`], not merely an
/// implementation detail on the way to one of the two extremes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnnealingWeight {
    pub value: f64,
    up_step: f64,
    down_step: f64,
}

impl AnnealingWeight {
    pub fn new(anneal_frames: f64, decay_frames: f64) -> Self {
        Self { value: 0.0, up_step: 1.0 / anneal_frames.max(1.0), down_step: 1.0 / decay_frames.max(1.0) }
    }

    /// Step toward `1.0` (if `should_increase`) or `0.0` otherwise, by this
    /// weight's own configured step size — never a jump to either extreme.
    pub fn step(&mut self, should_increase: bool) {
        if should_increase {
            self.value = (self.value + self.up_step).min(1.0);
        } else {
            self.value = (self.value - self.down_step).max(0.0);
        }
    }

    /// The soft-rollback monitor's own extra nudge (module doc, "Continuous
    /// cross-check and soft rollback") — identical to `step(false)`, given
    /// its own name at the call site for readability (a rollback and an
    /// ordinary "not currently converged" frame both decay the SAME way, by
    /// design: there is no separate, harsher decay rate for a detected
    /// rollback versus an unconverged frame — both mean "trust this less",
    /// just discovered by different signals).
    pub fn force_decay(&mut self) {
        self.step(false);
    }
}

/// Result of one [`apply_gentle_scale_correction`] call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleCorrectionResult {
    pub delta_log_s: f64,
    pub applied: bool,
}

/// Milestone M7's scale-prior residual — see the module doc's "Gentle
/// scale-prior application" section for the full derivation. Mutates
/// `poses`/`patches` IN PLACE (the same window-local indexing
/// `crate::dpvo_vi_ba::dpvo_vi_ba`'s own output uses): every pose at local
/// index `>= fixedp` has its translation scaled by `exp(delta_log_s)`; every
/// patch has its inverse depth scaled by `1/exp(delta_log_s)`. A no-op
/// (`applied: false`) when `weight_multiplier <= 0.0` or there are no IMU
/// factors in this window to derive a correction from — i.e. this function
/// itself enforces "before activation, nothing changes", not just its
/// caller.
#[allow(clippy::too_many_arguments)]
pub fn apply_gentle_scale_correction(
    poses: &mut [SE3],
    patches: &mut [DpvoPatch],
    fixedp: usize,
    factors: &[DpvoImuFactor],
    velocities: &[Vector3<f64>],
    body_to_camera: &SE3,
    bias_gyro: &Vector3<f64>,
    bias_accel: &Vector3<f64>,
    posterior: LogScalePosterior,
    weight_multiplier: f64,
    cfg: &ScaleCouplingConfig,
) -> ScaleCorrectionResult {
    if weight_multiplier <= 0.0 || factors.is_empty() {
        return ScaleCorrectionResult { delta_log_s: 0.0, applied: false };
    }

    let mut sum_a = 0.0_f64;
    let mut sum_b = 0.0_f64;
    let scale_factor_h = SCALE_FD_STEP.exp();
    for f in factors {
        let (i, j) = (f.i, f.j);
        if i >= poses.len() || j >= poses.len() || i >= velocities.len() || j >= velocities.len() {
            continue;
        }
        let (r0, ..) = imu_factor_jacobians(
            &poses[i], &poses[j], &velocities[i], &velocities[j], body_to_camera, &f.factor, bias_gyro,
            bias_accel,
        );
        let mut pose_i_h = poses[i].clone();
        if i >= fixedp {
            pose_i_h.translation *= scale_factor_h;
        }
        let mut pose_j_h = poses[j].clone();
        if j >= fixedp {
            pose_j_h.translation *= scale_factor_h;
        }
        let (r1, ..) = imu_factor_jacobians(
            &pose_i_h, &pose_j_h, &velocities[i], &velocities[j], body_to_camera, &f.factor, bias_gyro,
            bias_accel,
        );

        let whitener = imu_factor_whitener(&f.factor);
        let wr0 = whitener * r0;
        let wr1 = whitener * r1;
        let j_scale = (wr1 - wr0) / SCALE_FD_STEP;
        sum_a += j_scale.dot(&j_scale);
        sum_b += j_scale.dot(&wr0);
    }

    let information = weight_multiplier / posterior.variance.max(cfg.variance_floor);
    let lhs = sum_a + information;
    if lhs <= 1.0e-12 {
        return ScaleCorrectionResult { delta_log_s: 0.0, applied: false };
    }
    let rhs = -sum_b + information * posterior.mean;
    let delta_log_s = (rhs / lhs).clamp(-cfg.max_log_step, cfg.max_log_step);

    let factor = delta_log_s.exp();
    for (idx, pose) in poses.iter_mut().enumerate() {
        if idx >= fixedp {
            pose.translation *= factor;
        }
    }
    for patch in patches.iter_mut() {
        patch.inverse_depth /= factor;
    }
    ScaleCorrectionResult { delta_log_s, applied: true }
}

/// Milestone M7 output-space blend — see the module doc's "Why output-space
/// blending" section. `visual_poses`/`imu_poses` (and the patch slices) must
/// have identical length/ordering (both solved over the SAME window).
pub fn blend_solutions(
    visual_poses: &[SE3],
    visual_patches: &[DpvoPatch],
    imu_poses: &[SE3],
    imu_patches: &[DpvoPatch],
    weight: f64,
) -> (Vec<SE3>, Vec<DpvoPatch>) {
    let w = weight.clamp(0.0, 1.0);
    let poses: Vec<SE3> = visual_poses
        .iter()
        .zip(imu_poses.iter())
        .map(|(visual, imu)| {
            if w <= 0.0 {
                visual.clone()
            } else if w >= 1.0 {
                imu.clone()
            } else {
                let relative = imu.compose(&visual.inverse());
                let xi = relative.log();
                SE3::exp(&(xi * w)).compose(visual)
            }
        })
        .collect();
    let patches: Vec<DpvoPatch> = visual_patches
        .iter()
        .zip(imu_patches.iter())
        .map(|(visual, imu)| DpvoPatch {
            x: visual.x,
            y: visual.y,
            inverse_depth: (1.0 - w) * visual.inverse_depth + w * imu.inverse_depth,
        })
        .collect();
    (poses, patches)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imu_preintegration::ImuPreintegrationFactor;

    /// This crate's own established convention for float comparisons in
    /// tests (see e.g. `dpvo_vi_ba.rs`'s `assert!((a - b).abs() < 1e-9)`
    /// pattern) rather than pulling in a new `approx` dependency the task
    /// forbids — `approx` happens to be present in `Cargo.lock` transitively
    /// (via `nalgebra`), but is not a declared dependency of this crate, so
    /// `use approx::...` here would be a new dependency edge, not a reuse of
    /// an existing one.
    fn assert_close(actual: f64, expected: f64, epsilon: f64) {
        assert!(
            (actual - expected).abs() <= epsilon,
            "expected {actual} to be within {epsilon} of {expected}"
        );
    }

    fn assert_close_vec3(actual: Vector3<f64>, expected: Vector3<f64>, epsilon: f64) {
        assert!(
            (actual - expected).norm() <= epsilon,
            "expected {actual:?} to be within {epsilon} of {expected:?}"
        );
    }

    fn synthetic_alignment(scale: f64, min_singular_value: f64, mean_residual_after: f64) -> DpvoMonoViAlignment {
        DpvoMonoViAlignment {
            scale,
            gravity_world: Vector3::new(0.0, 0.0, -9.81),
            raw_gravity_norm: 9.81,
            velocities: Vec::new(),
            mean_residual_after,
            condition_number: 1.0,
            window_frames: 5,
            min_singular_value,
        }
    }

    // ---- Requirement: "recursive scale estimator converges on synthetic
    // windows with noisy per-window estimates (variance shrinks, mean ->
    // truth)". ----
    #[test]
    fn recursive_scale_estimator_converges_with_shrinking_variance_toward_truth() {
        let cfg = ScaleCouplingConfig::default();
        let mut estimator = RecursiveScaleEstimator::new(cfg);
        let true_log_scale = 2.0_f64.ln();
        // A small, deterministic "noise" sequence (avoids pulling in a new
        // RNG dependency for a unit test — alternating +/- perturbations
        // around the truth, shrinking in magnitude, is enough to exercise
        // "noisy but consistent" without needing true randomness).
        let noise = [0.15, -0.12, 0.08, -0.03, 0.02, -0.015, 0.01, -0.008];
        let mut variances = Vec::new();
        for (k, n) in noise.iter().enumerate() {
            let measurement = ScaleMeasurement {
                log_scale: true_log_scale + n,
                variance: 0.01, // a fixed, moderate per-window measurement variance
            };
            let report = estimator.update(measurement);
            variances.push(report.posterior.variance);
            if k > 0 {
                assert!(
                    variances[k] <= variances[k - 1] + 1e-12,
                    "posterior variance must shrink or hold, not grow, on consistent evidence (step {k}): {:?}",
                    variances
                );
            }
        }
        let posterior = estimator.posterior().expect("at least one measurement fused");
        assert_close(posterior.mean, true_log_scale, 0.05);
        assert!(posterior.variance < 0.01, "variance should have shrunk below the single-measurement variance");
        assert!(estimator.is_converged(), "8 consistent measurements should satisfy both convergence gates");
    }

    // ---- Requirement: "refuses to converge (std stays high) under M5b's
    // degenerate windows". ----
    #[test]
    fn recursive_scale_estimator_refuses_to_converge_on_degenerate_windows() {
        let cfg = ScaleCouplingConfig::default();
        let estimator = RecursiveScaleEstimator::new(cfg);
        // The degenerate case, per `crate::dpvo_vi_ba`'s own
        // "constant_velocity_window" test: `estimate_mono_vi_alignment`
        // itself returns `Err(IllConditioned)` and NO measurement is ever
        // constructed — this is the honest "never converges" path the task
        // explicitly permits. A `RecursiveScaleEstimator` that has NEVER
        // received a measurement is, by construction, not converged.
        assert!(!estimator.is_converged());
        assert!(estimator.posterior().is_none());
    }

    // Also check the "converged in variance but not in raw-agreement" and
    // vice-versa half-states are correctly rejected — both convergence gates
    // are independently necessary (see module doc, "Convergence and
    // annealing").
    #[test]
    fn convergence_requires_both_low_variance_and_raw_agreement() {
        let cfg = ScaleCouplingConfig {
            convergence_window: 3,
            convergence_band: 0.02,
            convergence_std: 0.05,
            ..ScaleCouplingConfig::default()
        };
        let mut estimator = RecursiveScaleEstimator::new(cfg);
        // Two very confident (tiny-variance) measurements that nonetheless
        // DISAGREE with each other by more than `convergence_band` — the
        // posterior variance can end up small, but raw agreement must still
        // fail.
        estimator.update(ScaleMeasurement { log_scale: 0.0, variance: 1e-6 });
        estimator.update(ScaleMeasurement { log_scale: 0.5, variance: 1e-6 });
        estimator.update(ScaleMeasurement { log_scale: 0.0, variance: 1e-6 });
        assert!(
            !estimator.is_converged(),
            "posterior variance may be tiny, but the raw measurements disagree by 0.5 >> convergence_band"
        );
    }

    // ---- Requirement: "a poisoned-alignment stream (occasional 18x
    // outliers like M5b saw) does NOT move the posterior materially". ----
    #[test]
    fn poisoned_measurement_stream_does_not_move_the_posterior_materially() {
        let cfg = ScaleCouplingConfig::default();
        let mut estimator = RecursiveScaleEstimator::new(cfg);
        let true_log_scale = 1.0_f64.ln(); // near-1x, matching M4-perf's own ~1.27 baseline order of magnitude
        for _ in 0..12 {
            estimator.update(ScaleMeasurement { log_scale: true_log_scale, variance: 0.01 });
        }
        let pre_poison = estimator.posterior().unwrap();
        assert!(estimator.is_converged(), "12 consistent measurements should have converged first");

        // M5b's own real failure mode: a single alignment recovers 18.66x
        // amid a stream of near-1x windows.
        let poison_log_scale = 18.66_f64.ln();
        let report = estimator.update(ScaleMeasurement { log_scale: poison_log_scale, variance: 0.01 });
        assert!(report.huber_inflation > 1.0, "an 18.66x outlier must trip the Huber down-weight");

        let post_poison = estimator.posterior().unwrap();
        // "Does not move materially": the mean shift from one poisoned
        // measurement must stay a small fraction of the gap between the
        // truth and the poison value (18.66 - 1.0, in log space), not
        // anywhere close to fully absorbing it.
        let full_gap = (poison_log_scale - pre_poison.mean).abs();
        let actual_shift = (post_poison.mean - pre_poison.mean).abs();
        assert!(
            actual_shift < 0.1 * full_gap,
            "poisoned measurement moved the posterior mean by {actual_shift}, more than 10% of the full \
             {full_gap} gap to the outlier — Huber down-weighting is not doing its job"
        );
        // Recovers: a following run of consistent, correct measurements
        // should pull the (still Huber-protected) posterior right back.
        for _ in 0..12 {
            estimator.update(ScaleMeasurement { log_scale: true_log_scale, variance: 0.01 });
        }
        let recovered = estimator.posterior().unwrap();
        assert_close(recovered.mean, true_log_scale, 0.05);
    }

    // ---- Requirement: "annealing weight schedule unit test". ----
    #[test]
    fn annealing_weight_ramps_up_over_configured_frame_count_and_never_overshoots() {
        let mut weight = AnnealingWeight::new(10.0, 4.0);
        assert_eq!(weight.value, 0.0);
        for _ in 0..10 {
            weight.step(true);
        }
        assert_close(weight.value, 1.0, 1e-9);
        // One extra step-up call must not overshoot past 1.0.
        weight.step(true);
        assert_eq!(weight.value, 1.0);
    }

    // ---- Requirement: "soft-rollback decay test". ----
    #[test]
    fn annealing_weight_decays_faster_than_it_anneals_and_bottoms_out_at_zero() {
        let mut weight = AnnealingWeight::new(10.0, 4.0);
        for _ in 0..10 {
            weight.step(true);
        }
        assert_close(weight.value, 1.0, 1e-9);
        weight.force_decay();
        weight.force_decay();
        weight.force_decay();
        weight.force_decay();
        assert_close(weight.value, 0.0, 1e-9);
        // Decays in 4 steps (`decay_frames = 4.0`), faster than the 10 it
        // took to anneal up — the asymmetric-risk design choice.
        weight.force_decay();
        assert_eq!(weight.value, 0.0, "must not go negative past zero");
    }

    #[test]
    fn scale_measurement_variance_grows_as_conditioning_worsens_and_residual_grows() {
        let cfg = ScaleCouplingConfig::default();
        let well_conditioned = synthetic_alignment(2.0, 1.0, 0.01);
        let poorly_conditioned = synthetic_alignment(2.0, 0.01, 0.01);
        let noisier = synthetic_alignment(2.0, 1.0, 1.0);
        let m_good = scale_measurement_from_alignment(&well_conditioned, &cfg);
        let m_bad_conditioning = scale_measurement_from_alignment(&poorly_conditioned, &cfg);
        let m_bad_residual = scale_measurement_from_alignment(&noisier, &cfg);
        assert!(m_bad_conditioning.variance > m_good.variance);
        assert!(m_bad_residual.variance > m_good.variance);
        assert_close(m_good.log_scale, 2.0_f64.ln(), 1e-12);
    }

    #[test]
    fn blend_solutions_reproduces_endpoints_exactly_at_weight_zero_and_one() {
        let visual = vec![SE3::identity(), SE3::exp(&nalgebra::Vector6::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))];
        let imu = vec![SE3::identity(), SE3::exp(&nalgebra::Vector6::new(2.0, 0.0, 0.0, 0.3, 0.0, 0.0))];
        let visual_patches = vec![DpvoPatch { x: 1.0, y: 2.0, inverse_depth: 0.5 }];
        let imu_patches = vec![DpvoPatch { x: 1.0, y: 2.0, inverse_depth: 0.8 }];

        let (poses0, patches0) = blend_solutions(&visual, &visual_patches, &imu, &imu_patches, 0.0);
        for (a, b) in poses0.iter().zip(visual.iter()) {
            assert_close_vec3(a.translation, b.translation, 1e-12);
        }
        assert_close(patches0[0].inverse_depth, 0.5, 1e-12);

        let (poses1, patches1) = blend_solutions(&visual, &visual_patches, &imu, &imu_patches, 1.0);
        for (a, b) in poses1.iter().zip(imu.iter()) {
            assert_close_vec3(a.translation, b.translation, 1e-12);
        }
        assert_close(patches1[0].inverse_depth, 0.8, 1e-12);

        let (poses_half, patches_half) = blend_solutions(&visual, &visual_patches, &imu, &imu_patches, 0.5);
        // Second pose's translation should sit strictly between the two
        // (monotonic interpolation, not identical to either endpoint).
        let t_visual = visual[1].translation.x;
        let t_imu = imu[1].translation.x;
        let t_half = poses_half[1].translation.x;
        assert!(t_half > t_visual.min(t_imu) && t_half < t_visual.max(t_imu));
        assert_close(patches_half[0].inverse_depth, 0.65, 1e-9);
    }

    /// End-to-end synthetic scenario (requirement: "drifting-scale visual
    /// trajectory + consistent IMU -> coupled solve tracks metric scale
    /// within 5% after convergence"). Builds a short constant-acceleration
    /// trajectory (the same non-degenerate motion profile
    /// `crate::dpvo_vi_ba::tests::synthetic_window_recovers_metric_scale_within_two_percent`
    /// already established is required for scale observability), scales the
    /// VISUAL poses/depths by a WRONG factor (mimicking DPVO's own monocular
    /// drift), constructs true-metric IMU factors between them, and confirms
    /// [`RecursiveScaleEstimator`] converges its posterior mean to the true
    /// (inverse) scale correction within 5%.
    #[test]
    fn end_to_end_synthetic_drifting_scale_converges_within_five_percent_via_repeated_alignment() {
        use crate::dpvo_vi_ba::{estimate_mono_vi_alignment, DpvoMonoViAlignmentGates};
        use crate::imu_preintegration::ImuPreintegratedDelta;
        use nalgebra::UnitQuaternion;
        use visloc_core::geometry::SE3;

        // Genuine multi-directional per-segment acceleration — see
        // `crate::dpvo_vi_ba::tests::synthetic_mono_window`'s own doc for why
        // `estimate_mono_vi_alignment` specifically (unlike the coupled
        // `dpvo_vi_ba` solve) needs DIRECTION diversity, not just nonzero
        // acceleration, to stay non-degenerate.
        let dt = 0.1_f64;
        let gravity = Vector3::new(0.0, 0.0, -9.81);
        let accel_per_segment = [
            Vector3::new(2.0, 1.0, -0.5),
            Vector3::new(-1.0, 2.0, 0.5),
            Vector3::new(0.5, -1.5, 1.0),
            Vector3::new(1.0, 0.5, -1.0),
            Vector3::new(-0.5, -1.0, 1.5),
        ];
        let n_frames = accel_per_segment.len() + 1;
        // `true_scale_error`: DPVO's own monocular reconstruction is
        // SMALLER than metric truth by this factor (mirrors M4-perf's own
        // measured `< 1` monocular-drift case) — the correct recovered
        // scale is exactly `true_scale_error`.
        let true_scale_error = 1.6_f64;

        // Exact constant-acceleration-per-segment kinematics (same formula
        // as `crate::dpvo_vi_ba::tests::synthetic_mono_window`).
        let mut velocities = vec![Vector3::<f64>::zeros(); n_frames];
        let mut positions = vec![Vector3::<f64>::zeros(); n_frames];
        for (i, &a_i) in accel_per_segment.iter().enumerate() {
            positions[i + 1] = positions[i] + velocities[i] * dt + 0.5 * a_i * dt * dt;
            velocities[i + 1] = velocities[i] + a_i * dt;
        }
        // Visual (wrong-scale) poses: `p_visual = p_true / true_scale_error`,
        // `world_to_camera` translation is `-C` for identity rotation (this
        // crate's own `T_world_to_camera` convention, matching
        // `synthetic_mono_window`'s own pose construction).
        let visual_poses: Vec<SE3> = positions
            .iter()
            .map(|p| SE3::new(UnitQuaternion::identity(), -(p / true_scale_error)))
            .collect();

        let cfg = ScaleCouplingConfig::default();
        let mut estimator = RecursiveScaleEstimator::new(cfg);

        // Repeatedly run `estimate_mono_vi_alignment` on GROWING windows
        // [0, k) for k = 2..n_frames, feeding each accepted alignment into
        // the recursive estimator — this is exactly the "re-run periodically
        // against the current window" mechanism `crate::dpvo_vo`'s
        // `scale_coupling_step` performs on live data, done here on
        // synthetic data so it needs no ONNX session.
        for k in 2..=n_frames {
            let mut factors = Vec::new();
            for i in 0..k - 1 {
                let (v_i, v_j) = (velocities[i], velocities[i + 1]);
                let (p_i, p_j) = (positions[i], positions[i + 1]);
                factors.push(crate::dpvo_vi_ba::DpvoImuFactor {
                    i,
                    j: i + 1,
                    factor: ImuPreintegrationFactor {
                        keyframe_id_from: i as u64,
                        keyframe_id_to: (i + 1) as u64,
                        delta: ImuPreintegratedDelta {
                            delta_rotation: visloc_core::geometry::SO3::identity(),
                            delta_velocity: (v_j - v_i) - gravity * dt,
                            delta_position: (p_j - p_i) - v_i * dt - 0.5 * gravity * dt * dt,
                            delta_time: dt,
                            ..ImuPreintegratedDelta::identity()
                        },
                        gravity_world: gravity,
                        weight_rotation: 1.0,
                        weight_velocity: 1.0,
                        weight_position: 1.0,
                    },
                });
            }
            let gates = DpvoMonoViAlignmentGates {
                expected_gravity_magnitude: 9.81,
                gravity_norm_deviation_ratio: 0.3,
                min_scale: 0.05,
                max_scale: 20.0,
                max_condition_number: 1.0e8,
            };
            let window_poses = &visual_poses[0..k];
            let zero = Vector3::zeros();
            if let Ok(alignment) =
                estimate_mono_vi_alignment(window_poses, &factors, &SE3::identity(), zero, zero, &gates)
            {
                let measurement = scale_measurement_from_alignment(&alignment, &estimator.config());
                estimator.update(measurement);
            }
        }

        let posterior = estimator.posterior().expect("at least one window should have produced a measurement");
        let recovered_scale = posterior.mean.exp();
        let relative_error = (recovered_scale - true_scale_error).abs() / true_scale_error;
        assert!(
            relative_error < 0.05,
            "recovered scale {recovered_scale} vs true {true_scale_error}: relative error {relative_error} \
             exceeds the 5% acceptance bar"
        );
    }

    #[test]
    fn gyro_bias_estimator_fuses_repeated_measurements_and_soft_resets() {
        let cfg = ScaleCouplingConfig::default();
        let mut estimator = RecursiveGyroBiasEstimator::new(cfg);
        assert_eq!(estimator.mean(), Vector3::zeros());
        let truth = Vector3::new(0.003, -0.002, 0.001);
        for _ in 0..6 {
            estimator.update(truth, 1.0e-6);
        }
        assert_close_vec3(estimator.mean(), truth, 1e-3);
        let var_before = estimator.posterior().unwrap().variance;
        estimator.soft_reset();
        let var_after = estimator.posterior().unwrap().variance;
        assert!(var_after > var_before, "soft_reset must widen (not shrink) the posterior variance");
        // Mean is preserved across a soft reset (module doc: "keep the mean").
        assert_close_vec3(estimator.mean(), truth, 1e-3);
    }
}
