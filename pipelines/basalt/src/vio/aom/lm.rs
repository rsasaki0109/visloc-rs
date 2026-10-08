//! Levenberg-Marquardt configuration, problem trait, trial binding and solver loops.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LmConfig {
    pub lambda_initial: f64,
    pub lambda_min: f64,
    pub lambda_max: f64,
    pub max_iterations: usize,
    pub convergence_step: f64,
}
impl Default for LmConfig {
    fn default() -> Self {
        Self {
            lambda_initial: 1e-4,
            lambda_min: 1e-6,
            lambda_max: 1e2,
            max_iterations: 7,
            convergence_step: 1e-4,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LmDecision {
    Accepted,
    Rejected,
    Converged,
    Failed,
}
#[derive(Debug, Clone, PartialEq)]
pub struct LmTraceEntry {
    pub iteration: usize,
    pub lambda_before: f64,
    pub lambda_after: f64,
    pub cost_before: f64,
    pub model_cost: f64,
    pub actual_cost: f64,
    pub step_norm: f64,
    pub decision: LmDecision,
}
#[derive(Debug, Clone, PartialEq)]
pub struct LmLinearization {
    pub factors: Vec<WhitenedFactorRowStack>,
    pub cost: f64,
}

/// Observation-only callback payload for the pinned per-iteration oracle.
///
/// The default implementation on [`LmProblem`] is empty, so production runs
/// do not allocate or write diagnostic state. The active-window problem uses
/// this payload only when `VISLOC_BASALT_DETAIL_ITERATIONS=1` is set.
pub struct LmDiagnosticEvent<'a> {
    pub iteration: usize,
    pub trial: usize,
    pub phase: &'static str,
    pub lambda: f64,
    pub lambda_after: f64,
    pub cost_before: f64,
    pub model_cost: Option<f64>,
    /// Unrounded model decrease, observed directly before cost subtraction.
    pub model_decrease: Option<f64>,
    pub actual_cost: Option<f64>,
    pub step_norm: Option<f64>,
    pub decision: &'static str,
    /// State at the beginning of the iteration. For an after-decision event
    /// this remains the pre-trial state so landmark back-substitution can be
    /// reproduced exactly.
    pub base_state: &'a DVector<f64>,
    /// State represented by this snapshot (pre-trial, trial, or accepted
    /// state, depending on `phase`).
    pub state: &'a DVector<f64>,
    pub trial_state: Option<&'a DVector<f64>>,
    pub step: Option<&'a DVector<f64>>,
    pub damping_diag: &'a DVector<f64>,
    /// The solver's binary32 damped normal matrix, when the active
    /// `UpstreamF32` path owns one.  This is an observation-only checkpoint:
    /// callers must not rebuild it from the reduced f64 mirror and damping
    /// diagonal because that would introduce a different cast/add schedule.
    pub damped_h: Option<&'a DMatrix<f32>>,
    pub linearization: &'a LmLinearization,
    pub reduced: &'a ReducedNormalSystem,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LmFailure {
    NonFinite,
    RankDeficient,
    LinearSolve,
}
pub trait LmProblem {
    fn linearize(&self, state: &DVector<f64>) -> Result<LmLinearization, LmFailure>;
    fn cost(&self, state: &DVector<f64>) -> Result<f64, LmFailure>;

    /// Optional frame identity for the diagnostic-only IMU reduction audit.
    /// Generic/synthetic problems remain unlabelled and incur no work.
    fn diagnostic_frame_id(&self) -> Option<u64> {
        None
    }

    /// Numeric owner for the active solver. The default keeps synthetic and
    /// legacy callers on the historical f64 path; WindowProblem overrides it
    /// for the pinned `SqrtKeypointVioEstimator<float>` compatibility mode.
    fn scalar_mode(&self) -> ScalarMode {
        ScalarMode::ExtendedF64
    }

    /// Optional observation-only hook used by the full upstream iteration
    /// oracle. Keeping this a default no-op preserves the normal solver path
    /// and avoids imposing a trace implementation on synthetic test problems.
    fn diagnostic_lm_event(&mut self, _event: LmDiagnosticEvent<'_>) {}

    /// Optional, explicitly opt-in normal-system patch used by numeric
    /// counterfactuals. The default is a no-op, so ordinary solver behavior
    /// and all production callers remain unchanged. WindowProblem uses this
    /// only when a diagnostic IMU H/b delta file is named in the environment.
    fn diagnostic_patch_reduced_f32(
        &self,
        _iteration: usize,
        _h: &mut DMatrix<f32>,
        _b: &mut DVector<f32>,
    ) {
    }

    /// Apply a solver increment to a state. Generic problems retain the
    /// historical additive contract; manifold problems override this so the
    /// trial cost uses the same update rule as their Jacobians.
    fn apply_step(&self, state: &DVector<f64>, step: &DVector<f64>) -> DVector<f64> {
        state + step
    }

    /// Evaluate a trial state with any eliminated variables recovered from
    /// the same linearization.  Basalt back-substitutes landmarks before
    /// `computeError`; state-only problems keep the ordinary cost path.
    fn trial_cost(
        &self,
        _state: &DVector<f64>,
        _step: &DVector<f64>,
        trial: &DVector<f64>,
    ) -> Result<f64, LmFailure> {
        self.cost(trial)
    }

    /// Timed wrapper for a trial objective.  Keeping the historical
    /// `trial_cost` signature preserves generic callers; WindowProblem
    /// overrides this hook to split full-manifold trial construction from
    /// objective evaluation without changing its immutable borrow contract.
    fn trial_cost_timed(
        &self,
        state: &DVector<f64>,
        step: &DVector<f64>,
        trial: &DVector<f64>,
        timing: &mut TimingBreakdown,
    ) -> Result<f64, LmFailure> {
        timing.measure(TimingBucket::LmTrialCost, || {
            self.trial_cost(state, step, trial)
        })
    }

    /// Evaluate a trial and optionally retain exact eliminated-variable work
    /// for the immediately following accepted step.  The default hook keeps
    /// existing `LmProblem` implementations source-compatible and carries no
    /// payload; concrete manifold problems may override it when trial work
    /// already computed a safe, one-shot commit token.
    fn trial_cost_timed_with_token(
        &self,
        state: &DVector<f64>,
        step: &DVector<f64>,
        trial: &DVector<f64>,
        timing: &mut TimingBreakdown,
    ) -> Result<(f64, LmTrialToken), LmFailure> {
        self.trial_cost_timed(state, step, trial, timing)
            .map(|cost| (cost, LmTrialToken::default()))
    }

    /// Evaluate a trial with an optional clean-path landmark preparation.
    /// Generic and retained implementations deliberately discard this opaque
    /// payload and continue through their existing token hook; only a future
    /// concrete window override may consume the prepared increments.
    fn trial_cost_timed_with_preparation(
        &self,
        state: &DVector<f64>,
        step: &DVector<f64>,
        trial: &DVector<f64>,
        preparation: Option<LmTrialPreparation>,
        timing: &mut TimingBreakdown,
    ) -> Result<(f64, LmTrialToken), LmFailure> {
        drop(preparation);
        self.trial_cost_timed_with_token(state, step, trial, timing)
    }

    /// Commit eliminated-variable increments after a successful trial.
    fn accept_step(
        &mut self,
        _state: &DVector<f64>,
        _step: &DVector<f64>,
    ) -> Result<(), LmFailure> {
        Ok(())
    }

    /// Commit a trial's one-shot token.  The default discards the token and
    /// preserves the historical accepted-step hook for generic and retained
    /// diagnostic implementations.
    fn accept_step_with_token(
        &mut self,
        state: &DVector<f64>,
        step: &DVector<f64>,
        token: LmTrialToken,
    ) -> Result<(), LmFailure> {
        let _ = token;
        self.accept_step(state, step)
    }
}

#[derive(Debug)]
pub(super) struct LmPreparedLandmarkStep {
    pub(super) landmark_index: usize,
    pub(super) track_id: u64,
    pub(super) step: Option<Vector3<f64>>,
}

/// Opaque one-shot preparation produced by the clean UpstreamF32 reducer
/// after a state step is solved.  The vector is intentionally private: a
/// concrete consumer must validate its own landmark topology before using it;
/// generic `LmProblem` implementations simply drop it via the default hook.
#[derive(Debug)]
pub struct LmTrialPreparation {
    pub(super) landmark_steps: Vec<LmPreparedLandmarkStep>,
    pub(super) tolerance_bits: u64,
    pub(super) state_fingerprint: u64,
    pub(super) step_fingerprint: u64,
}

impl LmTrialPreparation {
    /// Consume this one-shot preparation and expose only the mapped values a
    /// concrete window consumer needs for validation.  The compact f32
    /// payload itself never crosses the public trait boundary.
    pub(crate) fn take_landmark_steps(
        self,
    ) -> (u64, u64, u64, Vec<(usize, u64, Option<Vector3<f64>>)>) {
        (
            self.tolerance_bits,
            self.state_fingerprint,
            self.step_fingerprint,
            self.landmark_steps
                .into_iter()
                .map(|entry| (entry.landmark_index, entry.track_id, entry.step))
                .collect(),
        )
    }

    #[cfg(test)]
    pub(crate) fn from_test_entries(
        tolerance: f64,
        entries: Vec<(usize, u64, Option<Vector3<f64>>)>,
    ) -> Self {
        Self {
            landmark_steps: entries
                .into_iter()
                .map(|(landmark_index, track_id, step)| LmPreparedLandmarkStep {
                    landmark_index,
                    track_id,
                    step,
                })
                .collect(),
            tolerance_bits: tolerance.to_bits(),
            state_fingerprint: 0,
            step_fingerprint: 0,
        }
    }

    #[cfg(test)]
    pub(crate) fn from_test_entries_for_state_step(
        state: &DVector<f64>,
        step: &DVector<f64>,
        tolerance: f64,
        entries: Vec<(usize, u64, Option<Vector3<f64>>)>,
    ) -> Self {
        Self {
            landmark_steps: entries
                .into_iter()
                .map(|(landmark_index, track_id, step)| LmPreparedLandmarkStep {
                    landmark_index,
                    track_id,
                    step,
                })
                .collect(),
            tolerance_bits: tolerance.to_bits(),
            state_fingerprint: lm_trial_vector_fingerprint(state),
            step_fingerprint: lm_trial_vector_fingerprint(step),
        }
    }
}

/// Opaque, one-shot work product passed from an LM trial to its immediate
/// accepted-step commit.  The default token is empty; `WindowProblem` stores
/// only the exact landmark increments it already computed while building the
/// eager trial view.  Keeping this type public is required because
/// [`LmProblem`] is public, while its payload remains private to this module.
#[derive(Debug, Default)]
pub struct LmTrialToken {
    pub(super) landmark_steps: Option<Vec<Option<Vector3<f64>>>>,
    binding: Option<LmTrialBinding>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LmTrialBinding {
    window_identity: usize,
    window_generation: u64,
    scalar_mode: ScalarMode,
    state_fingerprint: u64,
    step_fingerprint: u64,
}

impl LmTrialBinding {
    pub(crate) const fn new(
        window_identity: usize,
        window_generation: u64,
        scalar_mode: ScalarMode,
        state_fingerprint: u64,
        step_fingerprint: u64,
    ) -> Self {
        Self {
            window_identity,
            window_generation,
            scalar_mode,
            state_fingerprint,
            step_fingerprint,
        }
    }
}

impl LmTrialToken {
    pub(crate) const fn with_landmark_steps(landmark_steps: Vec<Option<Vector3<f64>>>) -> Self {
        Self {
            landmark_steps: Some(landmark_steps),
            binding: None,
        }
    }

    pub(crate) const fn with_bound_landmark_steps(
        landmark_steps: Vec<Option<Vector3<f64>>>,
        binding: LmTrialBinding,
    ) -> Self {
        Self {
            landmark_steps: Some(landmark_steps),
            binding: Some(binding),
        }
    }

    pub(crate) fn take_landmark_steps(self) -> Option<Vec<Option<Vector3<f64>>>> {
        self.landmark_steps
    }

    pub(crate) fn take_landmark_steps_for(
        self,
        expected_binding: LmTrialBinding,
    ) -> Result<Option<Vec<Option<Vector3<f64>>>>, ()> {
        let Self {
            landmark_steps,
            binding,
        } = self;
        match landmark_steps {
            None => Ok(None),
            Some(landmark_steps) if binding == Some(expected_binding) => Ok(Some(landmark_steps)),
            Some(_) => Err(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LmResult {
    pub state: DVector<f64>,
    pub cost: f64,
    pub lambda: f64,
    pub iterations: usize,
    pub trace: Vec<LmTraceEntry>,
}

/// Whether the lean LM should evaluate the model decrease from the compact
/// Q1/Q2 payload retained by the same landmark reduction (instead of
/// re-factoring every landmark in `model_cost_decrease_f32`).
///
/// The payload evaluator is bit-identical to the full evaluator on the
/// audited factor mixes (see the `m7_q2_model_reuse_*` tests) and returns
/// `None` on any topology/rank/dimension mismatch, in which case the caller
/// falls back to the full evaluator.
///
/// The variable is deliberately *not* `VISLOC_BASALT_*`: any such key opts the
/// estimator back into the fully diagnostic path (`diagnostic_env_active`), so
/// a `VISLOC_BASALT_`-prefixed switch could not isolate this evaluator.  Set
/// `VISLOC_RS_PAYLOAD_MODEL_DECREASE=0` to force the historical full evaluator
/// for A/B replay.
fn payload_model_decrease_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("VISLOC_RS_PAYLOAD_MODEL_DECREASE").as_deref(),
            Ok("0") | Ok("false")
        )
    })
}

/// Lean UpstreamF32 LM loop used only by the explicit no-diagnostics window
/// path.  The retained/diagnostic implementation below deliberately keeps its
/// historical f64 mirror and event payloads.  In this path the active solve
/// owns only the f32 reduced H/b objects: no diagnostic ReducedNormalSystem,
/// f64 H/b mirror, damping payload, or event snapshot is materialized.
fn solve_lm_without_diagnostics_f32<P: LmProblem>(
    problem: &mut P,
    initial: DVector<f64>,
    config: LmConfig,
    retain_trace: bool,
    timing: &mut TimingBreakdown,
) -> Result<LmResult, LmFailure> {
    let _diagnostic_run_guard = begin_diagnostic_lm_run();
    set_active_diagnostic_lm_iteration(None);
    set_active_diagnostic_lm_frame(None);
    if !config.lambda_initial.is_finite()
        || config.lambda_initial <= 0.0
        || config.lambda_min <= 0.0
        || config.lambda_max < config.lambda_min
    {
        return Err(LmFailure::NonFinite);
    }

    let mut state = initial;
    let mut lambda = (config.lambda_initial as f32)
        .clamp(config.lambda_min as f32, config.lambda_max as f32) as f64;
    let mut cost = problem.cost(&state)?;
    if !cost.is_finite() {
        return Err(LmFailure::NonFinite);
    }
    let mut trace = retain_trace.then(Vec::new);
    // Upstream uses `it <= vio_max_iterations`, i.e. a configured value of
    // seven permits eight accepted/rejected attempts in total.
    let mut lambda_vee = 2.0;
    // The linearization point only changes on acceptance, and rejection leaves
    // `state` untouched.  Relinearizing and re-reducing on a rejected damping
    // trial therefore recomputes bit-identical `lin`/`reduced` from identical
    // inputs.  Cache them and re-run only the damping, the small reduced-system
    // solve, the model evaluation, and the trial cost for each lambda attempt.
    // This mirrors upstream Basalt's inner damping backtracking loop without
    // changing the attempt budget or the arithmetic of any evaluation.
    let mut cached: Option<(LmLinearization, ReducedNormalSystemF32)> = None;
    for iteration in 0..=config.max_iterations {
        set_active_diagnostic_lm_iteration(Some(iteration));
        set_active_diagnostic_lm_frame(problem.diagnostic_frame_id());
        if cached.is_none() {
            let lin = timing.measure(TimingBucket::LmLinearize, || problem.linearize(&state))?;
            if !lin.cost.is_finite() {
                return Err(LmFailure::NonFinite);
            }
            let reduced = timing.measure(TimingBucket::LmLandmarkReduction, || {
                reduce_landmark_factors_f32_checked_with_compact_back_substitution(
                    &lin.factors,
                    state.len(),
                    1e-10,
                )
                .map_err(|_| LmFailure::LinearSolve)
            })?;
            cached = Some((lin, reduced));
        }
        let (lin, reduced) = cached
            .as_ref()
            .expect("linearization cache populated above");

        // Native refreshes error_total from linearizeProblem each iteration;
        // the previous trial's expression schedule can yield different bits.
        // With the cache, `lin.cost` is the same value the fresh linearization
        // would return for this unchanged state.
        cost = lin.cost;
        let mut h32 = reduced.h.clone();
        if !h32.iter().all(|value| value.is_finite())
            || !reduced.b.iter().all(|value| value.is_finite())
        {
            return Err(LmFailure::NonFinite);
        }
        for i in 0..h32.nrows() {
            // Literal `vio_lm_pose_damping_variant == 1`: use the undamped
            // f32 normal diagonal and the configured f32 lambda floor.
            let damping = (h32[(i, i)] * lambda as f32).max(config.lambda_min as f32);
            h32[(i, i)] += damping;
        }

        let step = timing.measure(
            TimingBucket::LmLinearSystemSolve,
            || match eigen_ldlt_solve_f32(&h32, &reduced.b) {
                Some(value) => Ok(value.map(|entry| f64::from(-entry))),
                None => Err(LmFailure::LinearSolve),
            },
        )?;
        if !step.iter().all(|value| value.is_finite()) {
            return Err(LmFailure::NonFinite);
        }

        // Prefer the Q1/Q2 payload retained by the reduction that produced
        // this same reduced system: it is bit-identical to the full evaluator
        // on the audited mixes and avoids re-factoring every landmark.  Fall
        // back to the complete transformed row-stack evaluator whenever the
        // payload is absent or structurally ineligible.
        let model_decrease = timing.measure(TimingBucket::LmModelDecrease, || {
            let payload = payload_model_decrease_enabled()
                .then(|| reduced.model_cost_decrease_from_payload(&lin.factors, &step, 1e-10))
                .flatten();
            payload
                .or_else(|| model_cost_decrease_f32(&lin.factors, &step, 1e-10))
                .ok_or(LmFailure::RankDeficient)
        })?;
        let model = (cost as f32 - model_decrease as f32) as f64;
        let step_norm = step
            .iter()
            .map(|value| (*value as f32).abs())
            .fold(0.0_f32, f32::max) as f64;

        // The clean f32 reducer has already retained the Q1/R rows.  Recover
        // each mapped landmark exactly once after the state solve and pass the
        // opaque result through the default-compatible trial hook.  The
        // concrete WindowProblem consumer may use it directly; generic and
        // retained implementations discard it and keep their legacy recovery
        // behavior.
        let preparation = timing.measure(TimingBucket::LmCompactBackSubstitution, || {
            reduced.trial_preparation_ref(&state, &step, 1e-10)
        });

        let trial = timing.measure(TimingBucket::LmCompactApplyStep, || {
            problem.apply_step(&state, &step)
        });
        let (actual, trial_token) = problem.trial_cost_timed_with_preparation(
            &state,
            &step,
            &trial,
            preparation,
            timing,
        )?;
        if !actual.is_finite() {
            return Err(LmFailure::NonFinite);
        }

        let convergence_step = config.convergence_step as f32 as f64;
        let function_decrease = (cost as f32 - actual as f32) as f64;

        let before = lambda;
        let cost_before = cost;
        let (_predicted_decrease, relative_decrease) =
            timing.measure(TimingBucket::LmDecisionStateBookkeeping, || {
                // Native uses backSubstitute's l_diff directly. Recovering it
                // from the rounded model cost can erase a small decrease.
                let predicted_decrease = model_decrease as f32 as f64;
                let relative_decrease = if predicted_decrease > 0.0 {
                    ((cost as f32 - actual as f32) / predicted_decrease as f32) as f64
                } else {
                    f64::NEG_INFINITY
                };
                (predicted_decrease, relative_decrease)
            });

        let (decision, after) = if actual < cost && relative_decrease > 0.0 {
            timing.measure(TimingBucket::LmAccept, || {
                problem.accept_step_with_token(&state, &step, trial_token)
            })?;
            // The linearization point moved, so the next attempt must
            // relinearize and re-reduce rather than reuse the cached pair.
            cached = None;
            timing.measure(TimingBucket::LmDecisionStateBookkeeping, || {
                state = trial.clone();
                cost = actual;
                let scale = (1.0_f32 - (2.0_f32 * relative_decrease as f32 - 1.0_f32).powi(3))
                    .max(1.0_f32 / 3.0_f32) as f64;
                lambda_vee = 2.0;
                (
                    LmDecision::Accepted,
                    ((lambda as f32 * scale as f32).max(config.lambda_min as f32)) as f64,
                )
            })
        } else {
            timing.measure(TimingBucket::LmDecisionStateBookkeeping, || {
                let next = (lambda as f32 * lambda_vee as f32) as f64;
                lambda_vee = (lambda_vee as f32 * 2.0_f32) as f64;
                (LmDecision::Rejected, next)
            })
        };
        lambda = after;
        if let Some(trace) = trace.as_mut() {
            trace.push(LmTraceEntry {
                iteration,
                lambda_before: before,
                lambda_after: after,
                cost_before,
                model_cost: model,
                actual_cost: actual,
                step_norm,
                decision,
            });
        }
        // Basalt does not clamp at the maximum: the rejected attempt is
        // recorded, then optimization terminates once damping crosses it.
        if decision == LmDecision::Rejected && lambda > config.lambda_max {
            break;
        }
        if decision == LmDecision::Accepted
            && (step_norm < convergence_step
                || (function_decrease > 0.0 && function_decrease < 1e-6_f32 as f64))
        {
            return Ok(LmResult {
                state,
                cost,
                lambda,
                iterations: iteration + 1,
                trace: trace.unwrap_or_default(),
            });
        }
    }
    Ok(LmResult {
        state,
        cost,
        lambda,
        iterations: config.max_iterations + 1,
        trace: trace.unwrap_or_default(),
    })
}

pub fn solve_lm<P: LmProblem>(
    problem: &mut P,
    initial: DVector<f64>,
    config: LmConfig,
) -> Result<LmResult, LmFailure> {
    let mut timing = TimingBreakdown::from_env();
    solve_lm_with_timing(problem, initial, config, true, true, &mut timing)
}

pub(crate) fn solve_lm_without_trace<P: LmProblem>(
    problem: &mut P,
    initial: DVector<f64>,
    config: LmConfig,
) -> Result<LmResult, LmFailure> {
    let mut timing = TimingBreakdown::from_env();
    // Retaining the result trace is independent from retaining the diagnostic
    // event payload.  Keep the historical no-trace helper's event behavior;
    // the explicit lean path is selected only by WindowProblem's
    // no-diagnostics entry point below.
    solve_lm_with_timing(problem, initial, config, false, true, &mut timing)
}

pub(crate) fn solve_lm_with_timing<P: LmProblem>(
    problem: &mut P,
    initial: DVector<f64>,
    config: LmConfig,
    retain_trace: bool,
    retain_diagnostics: bool,
    timing: &mut TimingBreakdown,
) -> Result<LmResult, LmFailure> {
    let scalar_mode = problem.scalar_mode();
    if !retain_diagnostics && scalar_mode == ScalarMode::UpstreamF32 {
        return solve_lm_without_diagnostics_f32(problem, initial, config, retain_trace, timing);
    }
    let _diagnostic_run_guard = begin_diagnostic_lm_run();
    // Keep direct `cost()`/`linearize()` calls outside the LM loop
    // unselected by the optional per-iteration IMU diagnostic.
    set_active_diagnostic_lm_iteration(None);
    set_active_diagnostic_lm_frame(None);
    if !config.lambda_initial.is_finite()
        || config.lambda_initial <= 0.0
        || config.lambda_min <= 0.0
        || config.lambda_max < config.lambda_min
    {
        return Err(LmFailure::NonFinite);
    }
    let mut state = initial;
    let mut lambda = if scalar_mode == ScalarMode::UpstreamF32 {
        (config.lambda_initial as f32).clamp(config.lambda_min as f32, config.lambda_max as f32)
            as f64
    } else {
        config
            .lambda_initial
            .clamp(config.lambda_min, config.lambda_max)
    };
    let mut cost = problem.cost(&state)?;
    if !cost.is_finite() {
        return Err(LmFailure::NonFinite);
    }
    let mut trace = retain_trace.then(Vec::new);
    // Upstream uses `it <= vio_max_iterations`, i.e. a configured value of
    // seven permits eight accepted/rejected attempts in total.
    let mut lambda_vee = 2.0;
    for iteration in 0..=config.max_iterations {
        set_active_diagnostic_lm_iteration(Some(iteration));
        set_active_diagnostic_lm_frame(problem.diagnostic_frame_id());
        let lin = timing.measure(TimingBucket::LmLinearize, || problem.linearize(&state))?;
        if !lin.cost.is_finite() {
            return Err(LmFailure::NonFinite);
        }
        if scalar_mode == ScalarMode::UpstreamF32 {
            cost = lin.cost;
        }
        let reduced_f32 = timing.measure(TimingBucket::LmLandmarkReduction, || {
            if scalar_mode == ScalarMode::UpstreamF32 {
                reduce_landmark_factors_f32_checked(&lin.factors, state.len(), 1e-10)
                    .map(Some)
                    .map_err(|_| LmFailure::LinearSolve)
            } else {
                Ok(None)
            }
        })?;
        let reduced_f64 = if scalar_mode == ScalarMode::ExtendedF64 {
            Some(timing.measure(TimingBucket::LmLandmarkReduction, || {
                reduce_landmark_factors(&lin.factors, state.len(), 1e-10)
            }))
        } else {
            None
        };
        let (reduced, h, b, h32, b32, damping_diag) =
            timing.measure(TimingBucket::LmNormalSystemPrep, || {
                if let Some(reduced_f32) = reduced_f32.as_ref() {
                    emit_imu_reduction_diagnostic(
                        problem.diagnostic_frame_id(),
                        iteration,
                        reduced_f32,
                    );
                }
                let reduced = reduced_f32
                    .as_ref()
                    .map(ReducedNormalSystemF32::as_f64)
                    .or_else(|| reduced_f64)
                    .expect("one scalar-mode normal reduction");
                let mut h = reduced.h.clone();
                let mut b = reduced.b.clone();
                let mut h32 = reduced_f32.as_ref().map(|reduced| reduced.h.clone());
                let mut b32 = reduced_f32.as_ref().map(|reduced| reduced.b.clone());
                if let (Some(h32), Some(b32)) = (h32.as_mut(), b32.as_mut()) {
                    problem.diagnostic_patch_reduced_f32(iteration, h32, b32);
                    h = DMatrix::from_fn(h32.nrows(), h32.ncols(), |row, col| {
                        h32[(row, col)] as f64
                    });
                    // Keep the f64 mirror used by diagnostic payloads and the
                    // non-f32 fallback coherent with the counterfactual solve.
                    // The active f32 path still solves the patched f32 objects.
                    b = DVector::from_iterator(b32.len(), b32.iter().copied().map(f64::from));
                }
                if !h.iter().all(|x| x.is_finite()) || !b.iter().all(|x| x.is_finite()) {
                    return Err(LmFailure::NonFinite);
                }
                let mut damping_diag = DVector::zeros(h.nrows());
                for i in 0..h.nrows() {
                    // Literal `vio_lm_pose_damping_variant == 1`: damp by the
                    // undamped normal diagonal, with the configured minimum lambda
                    // as an absolute floor.
                    if let Some(h32) = h32.as_mut() {
                        let damping = (h32[(i, i)] * lambda as f32).max(config.lambda_min as f32);
                        damping_diag[i] = damping as f64;
                        h[(i, i)] = h32[(i, i)] as f64 + damping as f64;
                        h32[(i, i)] += damping;
                    } else {
                        damping_diag[i] = (reduced.h[(i, i)] * lambda).max(config.lambda_min);
                        h[(i, i)] += damping_diag[i];
                    }
                }
                Ok((reduced, h, b, h32, b32, damping_diag))
            })?;
        problem.diagnostic_lm_event(LmDiagnosticEvent {
            iteration,
            trial: 0,
            phase: "iteration_start",
            lambda,
            lambda_after: lambda,
            cost_before: cost,
            model_cost: None,
            model_decrease: None,
            actual_cost: None,
            step_norm: None,
            decision: "pending",
            base_state: &state,
            state: &state,
            trial_state: None,
            step: None,
            damping_diag: &damping_diag,
            damped_h: h32.as_ref(),
            linearization: &lin,
            reduced: &reduced,
        });
        let step = timing.measure(TimingBucket::LmLinearSystemSolve, || {
            if let (Some(h32), Some(b32)) = (h32.as_ref(), b32.as_ref()) {
                // Upstream's `MatrixXf::ldlt().solve(b)` is a diagonal-pivoted
                // lower LDLT solve.  Keep the native sign convention: Eigen
                // returns `H^-1 b`, then ABS_QR applies the negative increment.
                match eigen_ldlt_solve_f32(h32, b32) {
                    Some(value) => Ok(value.map(|entry| f64::from(-entry))),
                    None => Err(LmFailure::LinearSolve),
                }
            } else {
                match h.cholesky() {
                    Some(factor) => Ok(factor.solve(&(-&b))),
                    None => Err(LmFailure::LinearSolve),
                }
            }
        })?;
        if !step.iter().all(|x| x.is_finite()) {
            return Err(LmFailure::NonFinite);
        }
        // ABS_QR's backSubstitute evaluates the complete transformed row
        // stack, including each landmark block's Q1 rows.  The reduced
        // camera quadratic alone therefore underestimates the model
        // decrease and changes the LM accept/reject schedule.
        let model_decrease = timing.measure(TimingBucket::LmModelDecrease, || {
            if scalar_mode == ScalarMode::UpstreamF32 {
                model_cost_decrease_f32(&lin.factors, &step, 1e-10)
            } else {
                model_cost_decrease(&lin.factors, &step, 1e-10)
            }
            .ok_or(LmFailure::RankDeficient)
        })?;
        // Basalt's linearized marginal-prior error deliberately drops the
        // FEJ-independent residual constant and may therefore be negative.
        // Its LM loop compares that reduced cost directly; clamping the model
        // to zero changes `l_diff`/relative-decrease at the first window
        // shift and can alter the accept/reject schedule.
        let model = if scalar_mode == ScalarMode::UpstreamF32 {
            (cost as f32 - model_decrease as f32) as f64
        } else {
            cost - model_decrease
        };
        let step_norm = if scalar_mode == ScalarMode::UpstreamF32 {
            step.iter()
                .map(|value| (*value as f32).abs())
                .fold(0.0_f32, f32::max) as f64
        } else {
            step.iter().map(|value| value.abs()).fold(0.0, f64::max)
        };
        let trial = timing.measure(TimingBucket::LmCompactApplyStep, || {
            problem.apply_step(&state, &step)
        });
        let actual = problem.trial_cost_timed(&state, &step, &trial, timing)?;
        if !actual.is_finite() {
            return Err(LmFailure::NonFinite);
        }
        let before = lambda;
        let cost_before = cost;
        let state_before =
            timing.measure(TimingBucket::LmDecisionStateBookkeeping, || state.clone());
        problem.diagnostic_lm_event(LmDiagnosticEvent {
            iteration,
            trial: 0,
            phase: "trial",
            lambda,
            lambda_after: lambda,
            cost_before,
            model_cost: Some(model),
            model_decrease: Some(model_decrease),
            actual_cost: Some(actual),
            step_norm: Some(step_norm),
            decision: "pending",
            base_state: &state_before,
            state: &state,
            trial_state: Some(&trial),
            step: Some(&step),
            damping_diag: &damping_diag,
            damped_h: h32.as_ref(),
            linearization: &lin,
            reduced: &reduced,
        });
        let convergence_step = if scalar_mode == ScalarMode::UpstreamF32 {
            config.convergence_step as f32 as f64
        } else {
            config.convergence_step
        };
        let function_decrease = if scalar_mode == ScalarMode::UpstreamF32 {
            (cost as f32 - actual as f32) as f64
        } else {
            cost - actual
        };
        let (_predicted_decrease, relative_decrease) =
            timing.measure(TimingBucket::LmDecisionStateBookkeeping, || {
                let predicted_decrease = if scalar_mode == ScalarMode::UpstreamF32 {
                    // Keep l_diff independent of the diagnostic model cost.
                    model_decrease as f32 as f64
                } else {
                    cost - model
                };
                let relative_decrease = if predicted_decrease > 0.0 {
                    if scalar_mode == ScalarMode::UpstreamF32 {
                        ((cost as f32 - actual as f32) / predicted_decrease as f32) as f64
                    } else {
                        (cost - actual) / predicted_decrease
                    }
                } else {
                    f64::NEG_INFINITY
                };
                (predicted_decrease, relative_decrease)
            });
        let (decision, after) = if actual < cost && relative_decrease > 0.0 {
            timing.measure(TimingBucket::LmAccept, || {
                problem.accept_step(&state, &step)
            })?;
            timing.measure(TimingBucket::LmDecisionStateBookkeeping, || {
                state = trial.clone();
                cost = actual;
                let scale = if scalar_mode == ScalarMode::UpstreamF32 {
                    (1.0_f32 - (2.0_f32 * relative_decrease as f32 - 1.0_f32).powi(3))
                        .max(1.0_f32 / 3.0_f32) as f64
                } else {
                    (1.0 - (2.0 * relative_decrease - 1.0).powi(3)).max(1.0 / 3.0)
                };
                lambda_vee = 2.0;
                (
                    LmDecision::Accepted,
                    if scalar_mode == ScalarMode::UpstreamF32 {
                        ((lambda as f32 * scale as f32).max(config.lambda_min as f32)) as f64
                    } else {
                        (lambda * scale).max(config.lambda_min)
                    },
                )
            })
        } else {
            timing.measure(TimingBucket::LmDecisionStateBookkeeping, || {
                let next = if scalar_mode == ScalarMode::UpstreamF32 {
                    (lambda as f32 * lambda_vee as f32) as f64
                } else {
                    lambda * lambda_vee
                };
                lambda_vee = if scalar_mode == ScalarMode::UpstreamF32 {
                    (lambda_vee as f32 * 2.0_f32) as f64
                } else {
                    lambda_vee * 2.0
                };
                (LmDecision::Rejected, next)
            })
        };
        lambda = after;
        let phase = if decision == LmDecision::Accepted {
            "accepted"
        } else {
            "rejected"
        };
        problem.diagnostic_lm_event(LmDiagnosticEvent {
            iteration,
            trial: 0,
            phase,
            lambda: before,
            lambda_after: after,
            cost_before,
            model_cost: Some(model),
            model_decrease: Some(model_decrease),
            actual_cost: Some(actual),
            step_norm: Some(step_norm),
            decision: if decision == LmDecision::Accepted {
                "accepted"
            } else {
                "rejected"
            },
            base_state: &state_before,
            state: &state,
            trial_state: Some(&trial),
            step: Some(&step),
            damping_diag: &damping_diag,
            damped_h: h32.as_ref(),
            linearization: &lin,
            reduced: &reduced,
        });
        if let Some(trace) = trace.as_mut() {
            trace.push(LmTraceEntry {
                iteration,
                lambda_before: before,
                lambda_after: after,
                cost_before,
                model_cost: model,
                actual_cost: actual,
                step_norm,
                decision,
            });
        }
        // Basalt does not clamp at the maximum: the rejected attempt is
        // recorded, then optimization terminates once damping crosses it.
        if decision == LmDecision::Rejected && lambda > config.lambda_max {
            break;
        }
        let convergence_function = if scalar_mode == ScalarMode::UpstreamF32 {
            1e-6_f32 as f64
        } else {
            1e-6_f64
        };
        if decision == LmDecision::Accepted
            && (step_norm < convergence_step
                || (function_decrease > 0.0 && function_decrease < convergence_function))
        {
            return Ok(LmResult {
                state,
                cost,
                lambda,
                iterations: iteration + 1,
                trace: trace.unwrap_or_default(),
            });
        }
    }
    Ok(LmResult {
        state,
        cost,
        lambda,
        iterations: config.max_iterations + 1,
        trace: trace.unwrap_or_default(),
    })
}
