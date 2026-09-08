# Bounded native BA policy — unmeasured candidate

The objective remains native quality, speed and bounded 10k memory, not forcing
every small BA window through an iterative solver. Same-state evidence shows
117 unavailable QR steps on the Legacy path; paired small-window measurements
also favor Legacy. An explicit size policy is therefore the next candidate.

`--ba-backend bounded-direct64-qr` selects before the first solve: at most 64
variable poses use the existing caller-configured direct backend, larger
systems use compact QR. The selection is logged. There is no failure-triggered
retry on another backend, no default change and no tolerance/LM change. Both
arms validate the shared fixed-calibration pure-visual eligibility contract.
Fixed-state checks remain enabled. The 64 boundary is a fixed resource cap,
not a GT-tuned accuracy threshold; it bounds a six-DoF pose matrix to 384×384
(1,179,648 bytes for one dense f64 matrix, not total process memory). Pose-point
storage can still scale with observations/landmarks, so a measured RSS gate
remains mandatory. No global all-pose pair graph is authorized by this policy.

[Ceres upstream solver documentation](https://raw.githubusercontent.com/ceres-solver/ceres-solver/master/docs/source/nnls_solving.rst)
distinguishes factorization and iterative solvers and describes quadratic
Schur storage as limiting direct dense methods to smaller camera problems.
This supports size-aware selection in general, not our exact threshold or
claims of improved quality. The source was checked 2026-09-08; the rendered
documentation timed out, so its upstream source was inspected instead.

Required validation:

1. Exact boundary tests (64 direct, 65 QR), fixed state and invalid-input
   rollback, strict CLI parsing; no selector should depend on solver failure.
2. Same-binary 1k Legacy control and policy repeats: model parity, backend
   selection coverage, independent quality gates and timing. Passing only
   the direct arm does not validate QR at larger sizes.
3. Existing 2.5k/5k/10k native tier gates with selection coverage and 2 GiB cap.
   The large QR arm is still unproven; reject on quality or RSS failure.
4. Full native end-to-end and repeat/restart/100k I/O requirements remain;
   a saved-feature replay cannot close them. README stays unchanged until
   measured COLMAP comparison supports a claim.

This candidate does not establish faster-than-Legacy small-window BA or solve
the previously rejected large-scale quality result. PR #98's shadow diagnostic
is a separate pending change; this branch is stacked on it.
