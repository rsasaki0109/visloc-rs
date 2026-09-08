# BA retry phase diagnostic

Status: instrumentation, no solve-policy or cache change.

Atlas profiling found the solver consumed87.06s of a164.04s main replay.
Normal equations are built at each LM iteration, including after rejected
steps. The existing solver consumes its camera Hessian, so blindly retaining
that system is unsafe; duplicating a dense Hessian is not an acceptable memory
tradeoff without a separate bound and measurement.

`VISLOC_BA_TRACE_PHASE_TIMING=1` emits scalar stderr records from the shared
weighted optimizer: normal-equation assembly, rollback snapshot, linear solve,
and tentative update/cost. Each record includes iteration, variable pose and
landmark counts, and whether the previous iteration was rejected. No arrays or
normal-system copies are added. No clocks start when disabled. These timings
are nested in the atlas solver timer, not additional work to add to its sum.

An iteration after rejection identifies potential repeated linearization work,
not proof that arbitrary backend state can be reused. QR has no normal assembly
and its reported normal-equation phase is only setup overhead. Linear failures
have no tentative-update phase; infer outcomes from existing solver reports,
not missing timing lines alone. Early errors also produce partial traces.
Debug/shadow modes must remain off in the measurement; trace I/O and other loop
bookkeeping are outside the reported intervals.

Verification: enabled native fixed-state/rollback tests, then unchanged-input
atlas components with all pre-BA/final model bytes compared to retained models.
Measure rejected-iteration assembly share before choosing an optimization.
Any implementation must preserve numerical traces, damping and acceptance,
explicitly handle the consumed Hessian, invalidate after accepted state changes,
and avoid a full dense copy. This diagnostic does not improve accuracy or close
native end-to-end, final-tier, restart or100k I/O gates.

First tail replay: all six pre-BA/final files match the retained historical
model. There are240 normal-equation events not following rejection (1.2082s)
and22 following rejection (0.14784s). The latter is the measured assembly work
a perfect rejected-state cache could avoid in this replay, before cache costs.
It is not enough evidence to add a large retained system. Main is running.
Commands, timing and model hashes:
`benchmarks/electro/m8-openloris-ba-retry-timing-v1.json`.
