//! Process-lifetime diagnostic environment snapshot, the IMU f64 capture
//! scopes and the diagnostic prior handoff.

use super::*;

/// These variables are harness metadata or fixture paths.  They are not read
/// by the estimator's numeric/diagnostic hooks, so they do not disable the
/// lean no-output path.  Keep this list explicit: an unknown future
/// `VISLOC_BASALT_*` key must fail closed into the diagnostic path.
const NON_DIAGNOSTIC_ENV_SUFFIXES: &[&str] = &[
    "GT_FREE",
    "PROFILE",
    "SEQUENCE",
    "INPUT_ROOT",
    "OUTPUT_ROOT",
    "SEED",
    "TEMPORAL_SEED",
    "THREADS",
    // Pure timing instrumentation; it does not enable a diagnostic payload.
    "TIMING_BREAKDOWN",
    // Test-only fixture/output paths confirmed by the source-side harnesses.
    "TRACE",
    "M8D_SETUP_INPUT",
    "M8D_SETUP_ORACLE",
    "MH01_ROOT",
    "CALIBRATION",
    "CONFIG",
    // The identity sidecar reads already-owned estimator records at a frame
    // boundary and does not require the retained diagnostic solver path.
    "VISUAL_FACTOR_IDENTITY_TRACE",
    "VISUAL_FACTOR_IDENTITY_TRACKS",
    // The lifecycle sidecar reads store counts/IDs only at a frame boundary;
    // it deliberately remains on the lean solver path.
    "LIFECYCLE_TRACE",
];

/// Diagnostic values which are consulted from the estimator/AOM hot paths.
///
/// The process environment is captured once and converted to owned values so
/// the solver never performs an environment lookup, path allocation, or filter
/// parse while reducing a factor.  The policy is intentionally process scoped:
/// callers must set all diagnostic variables before the first estimator call.
#[derive(Debug, Default)]
pub(crate) struct DiagnosticEnvSnapshot {
    pub(crate) active: bool,
    pub(crate) detail_iterations: bool,
    pub(crate) detail_trace: Option<PathBuf>,
    pub(crate) detail_frame: Option<u64>,
    pub(crate) solver_frontier_trace: Option<PathBuf>,
    pub(crate) diagnostic_imu_rows: Option<PathBuf>,
    pub(crate) diagnostic_imu_factor_inputs: Option<PathBuf>,
    /// JSONL path for the opt-in initial f64 IMU unwhitened-residual capture.
    /// This records only the already-computed quaternion product and raw
    /// rotation residual; it never participates in factor construction.
    pub(crate) diagnostic_imu_unwhitened_f64: Option<PathBuf>,
    pub(crate) diagnostic_frame: Option<u64>,
    pub(crate) diagnostic_imu_link: usize,
    pub(crate) diagnostic_imu_iteration: Option<usize>,
    pub(crate) diagnostic_imu_iterations: Option<Vec<usize>>,
    pub(crate) diagnostic_imu_hb: Option<PathBuf>,
    pub(crate) diagnostic_iteration: Option<usize>,
    pub(crate) diagnostic_normal_stages: bool,
    pub(crate) state_update_trace_frame: Option<u64>,
    pub(crate) marg_trace: Option<PathBuf>,
    pub(crate) marg_trace_event: Option<usize>,
    pub(crate) sqrt_trace: bool,
    pub(crate) sqrt_focus_k12: bool,
    pub(crate) sqrt_dump_workspace_all: bool,
    pub(crate) sqrt_dump_tail: bool,
    pub(crate) sqrt_dump_qj: Option<PathBuf>,
    pub(crate) visual_prefix_trace: Option<PathBuf>,
    /// JSONL visual-factor identity sidecar.  This is intentionally separate
    /// from the reduced-row visual-prefix trace: it records the estimator's
    /// causal observation/landmark lifecycle before solve/marginalization.
    pub(crate) visual_factor_identity_trace: Option<PathBuf>,
    pub(crate) visual_factor_identity_tracks: Option<Vec<u64>>,
    /// Selector values are parsed while the process snapshot is built.  The
    /// inner error is retained so a malformed configured selector fails closed
    /// without making AOM decode the raw environment value again.
    pub(crate) visual_prefix_trace_frame: Option<Result<u64, ()>>,
    pub(crate) visual_prefix_trace_iteration: Option<Result<usize, ()>>,
    pub(crate) visual_prefix_trace_trial: Option<Result<usize, ()>>,
    pub(crate) visual_chain_trace: bool,
    pub(crate) landmark_backsub_probe: Option<PathBuf>,
    pub(crate) landmark_backsub_probe_tracks: Option<Vec<u64>>,
    pub(crate) diagnostic_prior_input: Option<PathBuf>,
    pub(crate) q2_abs_hb: Option<PathBuf>,
    /// JSONL path for the MargData-boundary prior/FEJ sidecar.  This is
    /// intentionally separate from `marg_trace`: the latter records the
    /// reduced Q2 transition, while this path joins pre/post PriorData with
    /// the exact pre-shift schema-4 frame tables.
    pub(crate) margdata_sidecar: Option<PathBuf>,
    /// JSONL path for a frame-boundary store-count/eviction sidecar.  This is
    /// an observation-only hook and must not force retained solver payloads.
    pub(crate) lifecycle_trace: Option<PathBuf>,
    /// JSONL path for the exact pre-marginalization Q2 input boundary.  This
    /// is diagnostic-only and records the source observation slots that are
    /// retained as native null/zero rows after the AOM suffix is truncated.
    pub(crate) q2_boundary_trace: Option<PathBuf>,
    /// JSONL path for the opt-in f32 rig-triangulation attempt trace.  This
    /// key is deliberately diagnostic (rather than a lean metadata suffix),
    /// because it captures DLT/SVD intermediates and candidate history.
    pub(crate) triangulation_trace: Option<PathBuf>,
    pub(crate) triangulation_trace_tracks: Option<Vec<u64>>,
    /// Opt-in lossless ordered-factor oracle.  This is intentionally separate
    /// from the human-readable detail trace: it captures the immutable factor
    /// vector and reducer parity at one selected event.
    pub(crate) full70_factor_oracle: Option<PathBuf>,
    pub(crate) full70_factor_oracle_frame: Option<Result<u64, ()>>,
    pub(crate) full70_factor_oracle_iteration: Option<Result<usize, ()>>,
    pub(crate) full70_factor_oracle_trial: Option<Result<usize, ()>>,
    pub(crate) full70_factor_oracle_phase: Option<String>,
}

const DIAGNOSTIC_VALUE_KEYS: &[&str] = &[
    "VISLOC_BASALT_DETAIL_ITERATIONS",
    "VISLOC_BASALT_DETAIL_TRACE",
    "VISLOC_BASALT_DETAIL_FRAME",
    "VISLOC_BASALT_SOLVER_FRONTIER_TRACE",
    "VISLOC_BASALT_DIAGNOSTIC_IMU_ROWS",
    "VISLOC_BASALT_DIAGNOSTIC_IMU_FACTOR_INPUTS",
    "VISLOC_BASALT_DIAGNOSTIC_IMU_UNWHITENED_F64",
    "VISLOC_BASALT_DIAGNOSTIC_FRAME",
    "VISLOC_BASALT_DIAGNOSTIC_IMU_LINK",
    "VISLOC_BASALT_DIAGNOSTIC_IMU_ITERATION",
    "VISLOC_BASALT_DIAGNOSTIC_IMU_ITERATIONS",
    "VISLOC_BASALT_DIAGNOSTIC_IMU_HB",
    "VISLOC_BASALT_DIAGNOSTIC_ITERATION",
    "VISLOC_BASALT_DIAGNOSTIC_NORMAL_STAGES",
    "VISLOC_BASALT_STATE_UPDATE_TRACE_FRAME",
    "VISLOC_BASALT_MARG_TRACE",
    "VISLOC_BASALT_MARG_TRACE_EVENT",
    "VISLOC_BASALT_SQRT_TRACE",
    "VISLOC_BASALT_SQRT_FOCUS_K12",
    "VISLOC_BASALT_SQRT_DUMP_WORKSPACE_ALL",
    "VISLOC_BASALT_SQRT_DUMP_TAIL",
    "VISLOC_BASALT_SQRT_DUMP_QJ",
    "VISLOC_BASALT_SQRT_TRACE_INPUT",
    "VISLOC_BASALT_VISUAL_PREFIX_TRACE",
    "VISLOC_BASALT_VISUAL_PREFIX_FRAME_ID",
    "VISLOC_BASALT_VISUAL_PREFIX_ITERATION",
    "VISLOC_BASALT_VISUAL_PREFIX_TRIAL",
    "VISLOC_BASALT_VISUAL_FACTOR_IDENTITY_TRACE",
    "VISLOC_BASALT_VISUAL_FACTOR_IDENTITY_TRACKS",
    "VISLOC_BASALT_VISUAL_CHAIN_TRACE",
    "VISLOC_BASALT_LANDMARK_BACKSUB_PROBE",
    "VISLOC_BASALT_LANDMARK_BACKSUB_PROBE_TRACKS",
    "VISLOC_BASALT_DIAGNOSTIC_PRIOR_INPUT",
    "VISLOC_BASALT_Q2_ABS_HB",
    "VISLOC_BASALT_MARGDATA_SIDECAR",
    "VISLOC_BASALT_LIFECYCLE_TRACE",
    "VISLOC_BASALT_Q2_BOUNDARY_TRACE",
    "VISLOC_BASALT_TRIANGULATION_TRACE",
    "VISLOC_BASALT_TRIANGULATION_TRACE_TRACKS",
    "VISLOC_BASALT_FULL70_FACTOR_ORACLE",
    "VISLOC_BASALT_FULL70_FRAME_ID",
    "VISLOC_BASALT_FULL70_ITERATION",
    "VISLOC_BASALT_FULL70_TRIAL",
    "VISLOC_BASALT_FULL70_PHASE",
    "VISLOC_BASALT_FULL70_EXECUTABLE_SHA256",
    "VISLOC_BASALT_FULL70_SOURCE_SHA256",
    "VISLOC_BASALT_FULL70_CONFIG_SHA256",
    "VISLOC_BASALT_FULL70_CALIBRATION_SHA256",
    "VISLOC_BASALT_FULL70_INPUT_SHA256",
    "VISLOC_BASALT_FULL70_FEATURES",
    "VISLOC_BASALT_FULL70_TARGET",
    "VISLOC_BASALT_FULL70_DIRTY_SCOPE_SHA256",
];

fn cached_env_key(key: &OsStr) -> Option<&'static str> {
    let key = key.to_string_lossy();
    DIAGNOSTIC_VALUE_KEYS
        .iter()
        .copied()
        .find(|known| key.eq_ignore_ascii_case(known))
}

fn cached_path(values: &HashMap<&'static str, OsString>, key: &'static str) -> Option<PathBuf> {
    values.get(key).map(PathBuf::from)
}

fn cached_u64(values: &HashMap<&'static str, OsString>, key: &'static str) -> Option<u64> {
    values
        .get(key)
        .and_then(|value| value.to_str())
        .and_then(|value| value.parse::<u64>().ok())
}

pub(super) fn cached_selector_u64(
    values: &HashMap<&'static str, OsString>,
    key: &'static str,
) -> Option<Result<u64, ()>> {
    values.get(key).map(|value| {
        let value = value.to_str().ok_or(())?;
        if value.is_empty() {
            return Err(());
        }
        value.parse::<u64>().map_err(|_| ())
    })
}

pub(super) fn cached_selector_usize(
    values: &HashMap<&'static str, OsString>,
    key: &'static str,
) -> Option<Result<usize, ()>> {
    cached_selector_u64(values, key)
        .map(|value| value.and_then(|value| usize::try_from(value).map_err(|_| ())))
}

fn cached_usize(values: &HashMap<&'static str, OsString>, key: &'static str) -> Option<usize> {
    values
        .get(key)
        .and_then(|value| value.to_str())
        .and_then(|value| value.parse::<usize>().ok())
}

fn cached_usize_filter(
    values: &HashMap<&'static str, OsString>,
    key: &'static str,
) -> Option<Vec<usize>> {
    values.get(key).map(|value| {
        value
            .to_string_lossy()
            .split(',')
            .filter_map(|entry| entry.trim().parse::<usize>().ok())
            .collect()
    })
}

fn cached_u64_filter(
    values: &HashMap<&'static str, OsString>,
    key: &'static str,
) -> Option<Vec<u64>> {
    values.get(key).map(|value| {
        value
            .to_string_lossy()
            .split(',')
            .filter_map(|entry| entry.trim().parse::<u64>().ok())
            .collect()
    })
}

/// The process environment, or none where there is none:
/// `std::env::vars_os` panics on wasm32-unknown-unknown (the browser build).
fn environment() -> impl Iterator<Item = (OsString, OsString)> {
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    return std::env::vars_os();
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    return std::iter::empty();
}

fn diagnostic_env_snapshot_from_current() -> DiagnosticEnvSnapshot {
    let mut values = HashMap::new();
    let mut active = false;
    for (key, value) in environment() {
        active |= diagnostic_env_key_requires_fallback(&key);
        if let Some(canonical) = cached_env_key(&key) {
            // Windows environment names are case-insensitive.  If an unusual
            // environment contains the same key twice, the first snapshot
            // value is stable and no hot-path lookup is reintroduced.
            values.entry(canonical).or_insert(value);
        }
    }
    DiagnosticEnvSnapshot {
        active,
        detail_iterations: values.contains_key("VISLOC_BASALT_DETAIL_ITERATIONS"),
        detail_trace: cached_path(&values, "VISLOC_BASALT_DETAIL_TRACE"),
        detail_frame: cached_u64(&values, "VISLOC_BASALT_DETAIL_FRAME"),
        solver_frontier_trace: cached_path(&values, "VISLOC_BASALT_SOLVER_FRONTIER_TRACE"),
        diagnostic_imu_rows: cached_path(&values, "VISLOC_BASALT_DIAGNOSTIC_IMU_ROWS"),
        diagnostic_imu_factor_inputs: cached_path(
            &values,
            "VISLOC_BASALT_DIAGNOSTIC_IMU_FACTOR_INPUTS",
        ),
        diagnostic_imu_unwhitened_f64: cached_path(
            &values,
            "VISLOC_BASALT_DIAGNOSTIC_IMU_UNWHITENED_F64",
        ),
        diagnostic_frame: cached_u64(&values, "VISLOC_BASALT_DIAGNOSTIC_FRAME"),
        diagnostic_imu_link: cached_usize(&values, "VISLOC_BASALT_DIAGNOSTIC_IMU_LINK")
            .unwrap_or(0),
        diagnostic_imu_iteration: cached_usize(&values, "VISLOC_BASALT_DIAGNOSTIC_IMU_ITERATION"),
        diagnostic_imu_iterations: cached_usize_filter(
            &values,
            "VISLOC_BASALT_DIAGNOSTIC_IMU_ITERATIONS",
        ),
        diagnostic_imu_hb: cached_path(&values, "VISLOC_BASALT_DIAGNOSTIC_IMU_HB"),
        diagnostic_iteration: cached_usize(&values, "VISLOC_BASALT_DIAGNOSTIC_ITERATION"),
        diagnostic_normal_stages: values.contains_key("VISLOC_BASALT_DIAGNOSTIC_NORMAL_STAGES"),
        state_update_trace_frame: cached_u64(&values, "VISLOC_BASALT_STATE_UPDATE_TRACE_FRAME"),
        marg_trace: cached_path(&values, "VISLOC_BASALT_MARG_TRACE"),
        marg_trace_event: cached_usize(&values, "VISLOC_BASALT_MARG_TRACE_EVENT"),
        sqrt_trace: values.contains_key("VISLOC_BASALT_SQRT_TRACE"),
        sqrt_focus_k12: values.contains_key("VISLOC_BASALT_SQRT_FOCUS_K12"),
        sqrt_dump_workspace_all: values.contains_key("VISLOC_BASALT_SQRT_DUMP_WORKSPACE_ALL"),
        sqrt_dump_tail: values.contains_key("VISLOC_BASALT_SQRT_DUMP_TAIL"),
        sqrt_dump_qj: cached_path(&values, "VISLOC_BASALT_SQRT_DUMP_QJ"),
        visual_prefix_trace: cached_path(&values, "VISLOC_BASALT_VISUAL_PREFIX_TRACE"),
        visual_factor_identity_trace: cached_path(
            &values,
            "VISLOC_BASALT_VISUAL_FACTOR_IDENTITY_TRACE",
        ),
        visual_factor_identity_tracks: cached_u64_filter(
            &values,
            "VISLOC_BASALT_VISUAL_FACTOR_IDENTITY_TRACKS",
        ),
        visual_prefix_trace_frame: cached_selector_u64(
            &values,
            "VISLOC_BASALT_VISUAL_PREFIX_FRAME_ID",
        ),
        visual_prefix_trace_iteration: cached_selector_usize(
            &values,
            "VISLOC_BASALT_VISUAL_PREFIX_ITERATION",
        ),
        visual_prefix_trace_trial: cached_selector_usize(
            &values,
            "VISLOC_BASALT_VISUAL_PREFIX_TRIAL",
        ),
        visual_chain_trace: values.contains_key("VISLOC_BASALT_VISUAL_CHAIN_TRACE"),
        landmark_backsub_probe: cached_path(&values, "VISLOC_BASALT_LANDMARK_BACKSUB_PROBE"),
        landmark_backsub_probe_tracks: cached_u64_filter(
            &values,
            "VISLOC_BASALT_LANDMARK_BACKSUB_PROBE_TRACKS",
        ),
        diagnostic_prior_input: cached_path(&values, "VISLOC_BASALT_DIAGNOSTIC_PRIOR_INPUT"),
        q2_abs_hb: cached_path(&values, "VISLOC_BASALT_Q2_ABS_HB"),
        margdata_sidecar: cached_path(&values, "VISLOC_BASALT_MARGDATA_SIDECAR"),
        lifecycle_trace: cached_path(&values, "VISLOC_BASALT_LIFECYCLE_TRACE"),
        q2_boundary_trace: cached_path(&values, "VISLOC_BASALT_Q2_BOUNDARY_TRACE"),
        triangulation_trace: cached_path(&values, "VISLOC_BASALT_TRIANGULATION_TRACE"),
        triangulation_trace_tracks: cached_u64_filter(
            &values,
            "VISLOC_BASALT_TRIANGULATION_TRACE_TRACKS",
        ),
        full70_factor_oracle: cached_path(&values, "VISLOC_BASALT_FULL70_FACTOR_ORACLE"),
        full70_factor_oracle_frame: cached_selector_u64(&values, "VISLOC_BASALT_FULL70_FRAME_ID"),
        full70_factor_oracle_iteration: cached_selector_usize(
            &values,
            "VISLOC_BASALT_FULL70_ITERATION",
        ),
        full70_factor_oracle_trial: cached_selector_usize(&values, "VISLOC_BASALT_FULL70_TRIAL"),
        full70_factor_oracle_phase: values
            .get("VISLOC_BASALT_FULL70_PHASE")
            .and_then(|value| value.to_str())
            .map(str::to_owned),
    }
}

// The diagnostic mode is a process-lifetime policy.  Snapshot the environment
// at first use so hot LM/estimator calls do not repeatedly enumerate the
// process environment or parse selector values.  Callers must set diagnostic
// variables before the first window/estimator operation; changing them later
// cannot change the selected path.  The pure key predicate below remains
// separate so tests can exercise the fail-closed allowlist without mutating
// process environment.
static DIAGNOSTIC_ENV_SNAPSHOT: OnceLock<DiagnosticEnvSnapshot> = OnceLock::new();

// Scope the f64 IMU observer to the explicit link walk in
// `initial_window_diagnostics`.  The diagnostic prepass first calls
// `WindowProblem::linearize`, which also reaches `imu_factor_with_trace`;
// that call must not be mistaken for the authoritative one-link capture.
// A thread-local flag keeps this boundary local even if an estimator is
// driven from more than one worker thread.
thread_local! {
    static INITIAL_IMU_F64_CAPTURE_SCOPE: Cell<bool> = const { Cell::new(false) };
    static INITIAL_IMU_F64_CAPTURE_LINK_INDEX: Cell<Option<usize>> = const { Cell::new(None) };
}

pub(super) struct InitialImuF64CaptureScope {
    previous: bool,
    previous_link_index: Option<usize>,
}

impl InitialImuF64CaptureScope {
    pub(super) fn enter() -> Self {
        let previous = INITIAL_IMU_F64_CAPTURE_SCOPE.with(|active| {
            let previous = active.get();
            active.set(true);
            previous
        });
        let previous_link_index = INITIAL_IMU_F64_CAPTURE_LINK_INDEX.with(|link_index| {
            let previous = link_index.get();
            link_index.set(None);
            previous
        });
        Self {
            previous,
            previous_link_index,
        }
    }
}

impl Drop for InitialImuF64CaptureScope {
    fn drop(&mut self) {
        INITIAL_IMU_F64_CAPTURE_SCOPE.with(|active| active.set(self.previous));
        INITIAL_IMU_F64_CAPTURE_LINK_INDEX
            .with(|link_index| link_index.set(self.previous_link_index));
    }
}

pub(super) fn initial_imu_f64_capture_scope_active() -> bool {
    INITIAL_IMU_F64_CAPTURE_SCOPE.with(Cell::get)
}

pub(super) struct InitialImuF64CaptureLink {
    previous: Option<usize>,
}

impl InitialImuF64CaptureLink {
    pub(super) fn set(link_index: usize) -> Self {
        let previous = INITIAL_IMU_F64_CAPTURE_LINK_INDEX.with(|current| {
            let previous = current.get();
            current.set(Some(link_index));
            previous
        });
        Self { previous }
    }
}

impl Drop for InitialImuF64CaptureLink {
    fn drop(&mut self) {
        INITIAL_IMU_F64_CAPTURE_LINK_INDEX.with(|current| current.set(self.previous));
    }
}

pub(super) fn initial_imu_f64_capture_link_index() -> Option<usize> {
    INITIAL_IMU_F64_CAPTURE_LINK_INDEX.with(Cell::get)
}

// The native prior transition captures the newly reduced square-root prior
// immediately before its f32 RHS shift.  Keep that pre-shift value in a
// process-local diagnostic handoff so the estimator can join it with the
// post-shift WindowPrior without adding fields to the solver state or packet.
static DIAGNOSTIC_PRIOR_PRE: OnceLock<Mutex<Option<WindowPrior>>> = OnceLock::new();

pub(super) fn remember_diagnostic_prior_pre(prior: &WindowPrior) {
    if diagnostic_env_snapshot().margdata_sidecar.is_none() {
        return;
    }
    if let Ok(mut slot) = DIAGNOSTIC_PRIOR_PRE.get_or_init(|| Mutex::new(None)).lock() {
        *slot = Some(prior.clone());
    }
}

pub(crate) fn take_diagnostic_prior_pre() -> Option<WindowPrior> {
    DIAGNOSTIC_PRIOR_PRE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .ok()
        .and_then(|mut slot| slot.take())
}

fn diagnostic_env_key_requires_fallback(key: &std::ffi::OsStr) -> bool {
    let key = key.to_string_lossy();
    let prefix = "VISLOC_BASALT_";
    let Some(candidate_prefix) = key.get(..prefix.len()) else {
        return false;
    };
    if !candidate_prefix.eq_ignore_ascii_case(prefix) {
        return false;
    }
    let Some(suffix) = key.get(prefix.len()..) else {
        return false;
    };
    !NON_DIAGNOSTIC_ENV_SUFFIXES
        .iter()
        .any(|allowed| suffix.eq_ignore_ascii_case(allowed))
}

pub(super) fn diagnostic_env_active_for_keys(
    keys: impl IntoIterator<Item = std::ffi::OsString>,
) -> bool {
    keys.into_iter()
        .any(|key| diagnostic_env_key_requires_fallback(&key))
}

/// Returns the process-lifetime diagnostic/probe environment snapshot.
///
/// Canonical harness metadata and source-confirmed fixture paths are
/// deliberately ignored; every other `VISLOC_BASALT_*` key opts back into the
/// diagnostic path so a newly added probe cannot silently lose its boundary.
/// The snapshot is taken on first use, before the first window/estimator
/// operation, and is intentionally not refreshed after that point.
pub(crate) fn diagnostic_env_active() -> bool {
    diagnostic_env_snapshot().active
}

pub(crate) fn diagnostic_env_snapshot() -> &'static DiagnosticEnvSnapshot {
    DIAGNOSTIC_ENV_SNAPSHOT.get_or_init(diagnostic_env_snapshot_from_current)
}
