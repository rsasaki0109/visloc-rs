# Implicit landmark QR — staged implementation

## Shared-state diagnostic under development

Branch `diag/m8-qr-shared-state` adds an opt-in
`VISLOC_SFM_DEBUG_QR_SHARED_STATE=1` shadow solve on the Legacy path. At each
eligible pre-step state it computes QR with the same observations, robust
weights and lambda, reports pose/point delta differences and PCG status, then
uses only the original Legacy delta. It does not change the acceptance policy.
The comparison is restricted to pure rig/no-GNC states with at most 60 variable
poses, 64 total poses, 1,024 total landmarks and 20,000 rig observations.
Other states are not compared. Duplicate diagnostic storage is bounded by
these caps; this mode must never be used for timing or RSS performance claims.
Native model-byte parity and diagnostic coverage remain to be checked before
interpreting its output. PR #97's separate validation-speedup CI is pending.

## Follow-up: bounded validation work

On `perf/m8-qr-validation-scans`, full pose-vector finite checks are moved out
of each landmark's action/scatter into the global operator boundary. Each
track still checks its computed rows and updated output coordinates. This
removes O(landmarks × poses) validation scans, preserving arithmetic order;
the retained memory layout and solver/LM policy are unchanged. Tests cover
NaN in a fixed-rotation coordinate, overflow in the global damping action,
and nonfinite values in a touched long-track coordinate. Seven QR tests and
test-inclusive clippy pass. Native byte parity and timing are not yet measured;
this does not repair the rejected trajectory result by itself.

Measured follow-up: release `c7c3d8b` (rebased unchanged patch `4072e05`)
produces exact prior Legacy/QR model bytes. QR mapper is 13.117345 / 12.589078 s,
versus an additional old-QR-binary control at 19.352382 s; paired Legacy is
4.597046 s. QR retains 51 failures and 166 accepted steps. This is a limited
serial diagnostic improvement, not an accepted three-repeat performance gate:
QR remains slower than Legacy and retains the failed trajectory. No larger
tier or README claim follows. [Evidence](../benchmarks/electro/m8-openloris-qr-validation-v1.json).

Parent PR #96 passed all nine CI jobs (`34192298439`) and merged as `bcbe2d5`;
its local/remote branch was removed. This follow-up branch remains unmerged.

## Native 1k result — rejected (2026-09-08)

Release `f62f09b` completed in 56.62 s. Same-binary Legacy reproduces all three
historical champion model files exactly. The two QR models are byte-identical;
all 1,000 images / 500 rig frames are supported in one component, calibration
and ordered input xy are preserved, and all 27,815 observations have positive
depth (2,683 points). Independent geometry and GT scoring use repeat A.

| Measure | Legacy / unchanged limit | QR A | QR B |
| --- | ---: | ---: | ---: |
| Mapper seconds | 4.064347 | 19.876707 | 18.796062 |
| Process wall seconds | 4.61 | 20.38 | 19.26 |
| Process peak RSS KiB | 82,060 | 81,620 | 81,836 |
| GT RMSE m (308 scored images) | ≤0.022695321 | 0.044781448 | identical model |
| GT p95 m | ≤0.037778776 | 0.064069244 | identical model |
| Raw mean reprojection px | ≤0.671637382 | 0.660774274 | identical model |

Both QR runs have 86 BA calls / 688 iterations, 166 accepted steps, 51 failed
linear steps and 12 calls with no accepted step. This reduces observed linear
failures relative to historical strict PCG (589), but does not establish the
same-linear-system comparison: native trajectories and track memberships differ.
Reprojection passes; trajectory and speed fail. Small-window RSS is not a 10k
memory result, and saved-feature/snapshot replay is not native end-to-end.
No default change, larger-tier promotion, tolerance or damping sweep follows.

[Commands, hashes, logs and independent audits](../benchmarks/electro/m8-openloris-native-qr-v1.json)
retain the rejected result. Next diagnosis must distinguish remaining numerical
failures from nonlinear path differences and account for QR operator cost before
another native candidate; reducing failure counts alone is not the objective.
PR/CI/merge for this branch remains pending.

### Unchanged-binary rejection diagnosis

Both context-only and detailed-residual debug replays reproduce all normal QR
model bytes. The 51 linear failures comprise 45 `MaxIterations` and six
`ResidualCheckFailed`. Of 471 rejected candidate steps, 326 reduce cost but
fail feasibility, 126 fail both gates, and 19 fail cost only. Together with
166 accepted steps and 51 linear failures this accounts for all 688 iterations.
Thus 452 candidate steps fail feasibility: better linear convergence alone is
not sufficient. These are observed rejection mechanisms, not proof of a unique
cause of trajectory regression. Debug timing is not performance evidence.
[Exact commands and diagnostic audit](../benchmarks/electro/m8-openloris-native-qr-diagnostic-v1.json).

Legacy control qualification: its unchanged debug model is champion-byte-exact,
yet 462 of its 476 rejected candidates also fail feasibility (353 with cost
decrease), versus QR's 452 of 471. The first three rejected-step log lines
are identical; the next differs only in printed cost precision. Feasibility
rejection is therefore not QR-specific and does not by itself explain the
trajectory regression. The next comparison needs a shared pre-step BA state,
not aggregate counts from already-divergent nonlinear paths. No threshold or
observation change is justified by this audit.
[Legacy/QR rejection evidence](../benchmarks/electro/m8-openloris-legacy-qr-rejections-v1.json).

Implementation-stage status (superseded by the measured result above):
default-off native `--ba-backend matrix-free-qr` implemented. Kernel,
shared LM and native fixed-state tests pass; real-data quality/performance
is not yet measured. M8–M10 remains incomplete.

## Native integration status

The former test-only kernel and generic PCG recurrence now compile in normal
builds. The existing Schur PCG is unchanged; its oracle tests reuse the QR
recurrence to check parity. `optimize_rig_qr` validates the usual matrix-free
contract plus pure, initially projectable rig observations before mutation.
Positive damping remains required. Per-iteration unprojectable rows or failed
factorizations become explicit failed linear steps, not dropped observations
or a legacy fallback. PCG remains 128 iterations, 1e-12 relative/absolute,
no restarts. Native all-pose-fixed windows retain the existing explicit
landmark-only/no-variable dispatch, not a pose Schur fallback.

The native selector is default-off. Tests exercise the production entry,
native fixed rotations/points/observations and rollback, strict CLI parsing,
and rejection of monocular input without mutation. The shared LM QR arm does
not assemble a duplicate normal system. No native dataset run, timing or RSS
claim follows from these tests. Next: certify the release binary and run the
unchanged 1k Legacy control and two QR repeats, then independently audit the
existing identity/geometry/trajectory gates before any larger-tier promotion.

The staged descriptions below document the implementation sequence; their
test-only status is superseded by this native integration section.

## Reason and references

Native strict, cluster8 and cluster8+restart1 all fail the unchanged 1k
quality gate. Further tolerance/cluster/restart sweeps are not selected.
[Demmel et al., CVPR 2021](https://arxiv.org/abs/2103.01843) eliminate landmark
variables by QR/nullspace operations, algebraically equivalent to Schur
elimination, and report stability benefits as well as dense-problem memory
costs. Their results do not establish a gain on our calibrated rig.

The inspected [RootBA dynamic landmark storage](https://github.com/nikolausdemmel/rootba/blob/master/src/rootba/qr/landmark_block_dynamic.hpp)
has rows proportional to observations and columns proportional to observed
poses. Do not adopt that dense per-track layout: our prior 1k audit already
estimated 1.731 GiB for a naive camera-Jacobian layout alone. These upstream
references were checked 2026-09-08; no upstream code was copied or dependency
added. Householder storage below is our bounded implementation choice.

## Kernel and intended operator

For each variable landmark, start from its weighted three-column Jacobian
with three `sqrt(lambda) * I` landmark-damping rows appended. Factor it with
three compact Householder reflectors. Retain only those reflectors and the
3×3 triangular factor, not full Q or an explicit nullspace/projector.
For m augmented rows, reflector storage is exactly 3m−3 scalar entries;
factorization additionally owns a temporary 3m-scalar input. Work is O(m).
The 20,003-row test checks logical storage counts, not process peak RSS.

The reduced action is implemented as `A*x = tail(Q^T * (J_pose*x))`, and
its adjoint as `A^T*y = J_pose^T * Q * [0; y]`. Keep original sparse per-row
pose Jacobians (six columns per residual row) and reuse per-track vector
scratch. This avoids materializing dense transformed camera blocks and the
subtraction `Hpp - Hpl Hll^-1 Hlp` during landmark elimination. An `A^T A`
iterative solve would still have normal-equation conditioning; this does not
claim to solve every conditioning or PCG convergence problem.

The kernel currently lives behind `cfg(test)` in `landmark_qr.rs`; it has no
production selector or effect on existing BA. Five kernel tests
pass: elimination/reconstruction, adjoint and back-substitution identities;
projected cost versus an independent damped least-squares calculation; and
linear retained storage plus invalid-input rejection. Clippy including tests
passes. The sparse-row tests additionally compare action, adjoint, reduced
normal action/RHS and the recovered full pose/point step to a small damped
normal system at three damping values. Repeated pose slots, fixed-pose rows,
fixed-landmark bypass and preweighted rows are included. A 10,003-row track
with 10,000 possible poses retains one sparse row per residual, three compact
reflectors and one longest-track scratch buffer, not a pose-pair graph.
The normal action reuses that buffer without copying a dense reduced block;
global pose damping must be added once by its future caller.
Zero columns and nonfinite operations reject explicitly. This is not
a rank-revealing undamped pseudoinverse implementation.

## Required next steps before native comparison

The test-only `RigQrLinearization` adapter now reuses the actual
`rig_residual_jacobians` function and `RobustKernel::weight`. It preserves two
rows per rig observation and applies square-root weights before elimination.
Fixed pose rows remain in point elimination; fixed points bypass it. Fixed
rotation columns are zeroed and get the same identity diagonal as existing
assembly, in addition to shared pose damping applied once globally.

A new adapter test compares multi-landmark action, RHS and recovered pose /
point step against existing weighted normal assembly and direct Schur solve.
It covers None and Huber6, lambda 0.5 and 100, repeated sensors with rotated
extrinsics, fixed anchor/rotation/point and unchanged input state. Missing
references, nonpositive damping and invalid action dimensions reject. The
adapter rejects unprojectable rows rather than silently dropping them; this
eligibility restriction must remain explicit during nonlinear integration.
All six kernel/adapter tests and test-inclusive clippy pass.

The adapter now connects to the existing test PCG recurrence with the unchanged
128-iteration budget and relative/absolute tolerances of 1e-12. The rig fixture
checks independently recomputed true residuals, agreement with the direct step,
exact repeated results and failure with a zero iteration budget; no direct
fallback is installed. This remains a test-only solver.

The shared LM loop now has a test-only QR dispatch. It skips normal-equation
assembly entirely for that arm, uses the caller's complete landmark layout
when recovering point updates, and then uses unchanged step acceptance,
rollback and lambda updates. Unobserved variable landmarks receive zero
updates. The fixture runs three nonlinear iterations twice and checks cost
decrease, exact repeatability, fixed pose/rotation/point, an unobserved point
and observation preservation. A one-iteration PCG budget with a deliberately
unattainable residual target checks unchanged state and three recorded LM
failures with increasing damping. This exercises the actual LM loop, but
does not provide a production/native selector or real-data quality evidence.

Its block-Jacobi preconditioner retains one inverse 6×6 block per variable pose
(36P scalars). Construction accumulates pose diagonals and one landmark's
pose/point cross blocks at a time, using the QR triangular factor for point
elimination. Scratch is O(unique poses in the current track), with BTreeMap
lookup overhead; no pose-pair enumeration or duplicate full normal system is
needed. Unlike the QR operator action, this diagonal construction still uses
normal-form subtraction and can suffer cancellation; nonfinite/non-SPD blocks
reject. The fixture checks agreement with the existing Schur preconditioner.
These are logical storage bounds, not measured native peak RSS.

1. Promote the tested multi-landmark action/PCG connection out of test scope.
   Preserve total PCG budget and honest residual stopping; do not add another
   public policy before the bounded linear solve is validated.
2. Preserve the bounded preconditioner and matched iterative/full-step tests. Audit
   O(observations + poses + landmarks) retained storage and longest-track
   scratch; prohibit all-pose pair enumeration, global Q, and retaining a
   duplicate full normal system just to obtain a preconditioner.
3. Integrate only after the linear kernel checks, keeping existing damping,
   loss and native acceptance gates. Define eligibility for positive damping
   and rank-deficient cases explicitly; no silent legacy fallback.
4. Certify one default-off native arm and unchanged same-binary controls;
   require independent geometry, identity, exact repeat and all native 1k
   quality limits before larger tiers or performance claims. No README
   performance change follows from kernel tests.
