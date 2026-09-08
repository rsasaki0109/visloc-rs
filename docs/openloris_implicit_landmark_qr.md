# Implicit landmark QR — staged implementation

Status: compact elimination kernel tested; not a native BA backend or a
quality/performance improvement. M8–M10 remains incomplete.

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

Next implement the reduced action as `A*x = tail(Q^T * (J_pose*x))`, and
its adjoint as `A^T*y = J_pose^T * Q * [0; y]`. Keep original sparse per-row
pose Jacobians (six columns per residual row) and reuse per-track vector
scratch. This avoids materializing dense transformed camera blocks and the
subtraction `Hpp - Hpl Hll^-1 Hlp` during landmark elimination. An `A^T A`
iterative solve would still have normal-equation conditioning; this does not
claim to solve every conditioning or PCG convergence problem.

The kernel currently lives behind `cfg(test)` in `landmark_qr.rs`; it has no
production selector, solver connection or effect on existing BA. Three tests
pass: elimination/reconstruction, adjoint and back-substitution identities;
projected cost versus an independent damped least-squares calculation; and
linear retained storage plus invalid-input rejection. Clippy including tests
passes. Zero columns and nonfinite operations reject explicitly. This is not
a rank-revealing undamped pseudoinverse implementation.

## Required next steps before native comparison

1. Add sparse pose-row reduced action/adjoint and RHS, preserving observation
   order, repeated rig sensors, fixed poses/rotations and robust square-root
   weights. Fixed landmarks bypass elimination; calibration stays fixed.
2. Compare action, adjoint, RHS and recovered full step on small damped
   problems. Audit O(observations + poses + landmarks) retained storage and
   longest-track scratch; prohibit all-pose pair enumeration and global Q.
3. Integrate only after the linear kernel checks, keeping existing damping,
   loss and native acceptance gates. Define eligibility for positive damping
   and rank-deficient cases explicitly; no silent legacy fallback.
4. Certify one default-off native arm and unchanged same-binary controls;
   require independent geometry, identity, exact repeat and all native 1k
   quality limits before larger tiers or performance claims. No README
   performance change follows from kernel tests.
