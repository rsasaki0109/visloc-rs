//! LM trial fingerprints and the thread-local diagnostic LM / projection context.

use super::*;

// Keep the state/step provenance check shared by the producer and the
// concrete WindowProblem consumer.  The preparation is one-shot, so a
// deterministic bit fingerprint is preferable to a borrowed state reference:
// it rejects a stale solve before any trial-side values are materialized.
pub(super) const LM_VECTOR_FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
pub(super) const LM_VECTOR_FNV_PRIME: u64 = 1_099_511_628_211;

#[inline]
pub(crate) fn lm_trial_vector_fingerprint(value: &DVector<f64>) -> u64 {
    let mut hash = LM_VECTOR_FNV_OFFSET;
    hash ^= value.len() as u64;
    hash = hash.wrapping_mul(LM_VECTOR_FNV_PRIME);
    for component in value.iter() {
        hash ^= component.to_bits();
        hash = hash.wrapping_mul(LM_VECTOR_FNV_PRIME);
    }
    hash
}

// The full solver's `linearize` trait method intentionally has no iteration
// argument because legacy/synthetic callers use it directly.  Keep diagnostic
// identity in a thread-local slot so concurrent windows cannot relabel one
// another's sidecars.  `None` remains the inactive state for pre-loop costs
// and direct callers.
#[derive(Clone, Copy, Debug, Default)]
struct DiagnosticLmContext {
    run_id: Option<u128>,
    frame_id: Option<u64>,
    iteration: Option<usize>,
}

thread_local! {
    static ACTIVE_DIAGNOSTIC_LM_CONTEXT: Cell<Option<DiagnosticLmContext>> = const { Cell::new(None) };
    // Marginalization re-linearizes the truncated AOM after the LM guard has
    // ended, so its diagnostic projection records need an explicit event
    // identity instead of borrowing the (already cleared) LM frame slot.
    static ACTIVE_DIAGNOSTIC_PROJECTION_EVENT: Cell<Option<DiagnosticProjectionEvent>> =
        const { Cell::new(None) };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DiagnosticProjectionEvent {
    pub(crate) frame_id: u64,
    pub(crate) timestamp_ns: i64,
}

pub(crate) fn set_active_diagnostic_projection_event(event: Option<DiagnosticProjectionEvent>) {
    ACTIVE_DIAGNOSTIC_PROJECTION_EVENT.with(|context| context.set(event));
}

pub(super) fn active_diagnostic_projection_event() -> Option<DiagnosticProjectionEvent> {
    ACTIVE_DIAGNOSTIC_PROJECTION_EVENT.with(Cell::get)
}

static NEXT_DIAGNOSTIC_LM_RUN_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) struct DiagnosticLmRunGuard {
    previous: Option<DiagnosticLmContext>,
}

impl Drop for DiagnosticLmRunGuard {
    fn drop(&mut self) {
        ACTIVE_DIAGNOSTIC_LM_CONTEXT.with(|context| context.set(self.previous));
    }
}

pub(crate) fn begin_diagnostic_lm_run() -> DiagnosticLmRunGuard {
    let run_id = next_diagnostic_lm_run_id();
    let previous = ACTIVE_DIAGNOSTIC_LM_CONTEXT.with(|context| {
        let previous = context.get();
        context.set(Some(DiagnosticLmContext {
            run_id: Some(run_id),
            ..DiagnosticLmContext::default()
        }));
        previous
    });
    DiagnosticLmRunGuard { previous }
}

pub(super) fn next_diagnostic_lm_run_id() -> u128 {
    let sequence = NEXT_DIAGNOSTIC_LM_RUN_ID.fetch_add(1, Ordering::Relaxed);
    (u128::from(std::process::id()) << 64) | u128::from(sequence)
}

fn active_diagnostic_lm_context() -> Option<DiagnosticLmContext> {
    ACTIVE_DIAGNOSTIC_LM_CONTEXT.with(Cell::get)
}

pub(crate) fn active_diagnostic_lm_run_id() -> Option<u128> {
    active_diagnostic_lm_context().and_then(|context| context.run_id)
}

pub(crate) fn set_active_diagnostic_lm_iteration(iteration: Option<usize>) {
    ACTIVE_DIAGNOSTIC_LM_CONTEXT.with(|context| {
        let mut value = context.get().unwrap_or_default();
        value.iteration = iteration;
        if value.run_id.is_none() && value.frame_id.is_none() && value.iteration.is_none() {
            context.set(None);
        } else {
            context.set(Some(value));
        }
    });
}

pub(crate) fn active_diagnostic_lm_iteration() -> Option<usize> {
    active_diagnostic_lm_context().and_then(|context| context.iteration)
}

pub(crate) fn set_active_diagnostic_lm_frame(frame_id: Option<u64>) {
    ACTIVE_DIAGNOSTIC_LM_CONTEXT.with(|context| {
        let mut value = context.get().unwrap_or_default();
        value.frame_id = frame_id;
        if value.run_id.is_none() && value.frame_id.is_none() && value.iteration.is_none() {
            context.set(None);
        } else {
            context.set(Some(value));
        }
    });
}

pub(crate) fn active_diagnostic_lm_frame() -> Option<u64> {
    active_diagnostic_lm_context().and_then(|context| context.frame_id)
}
