# Implicit landmark QR — staged implementation

Status: default-off native `--ba-backend matrix-free-qr` implemented. Kernel,
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
