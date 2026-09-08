# Native rig matrix-free integration

Status: implemented default-off; native 1k A/B fails quality, not promoted.

## Outcome and actual target

Connect the existing unscaled matrix-free BA API to the **native calibrated-rig
mapper**, not a substitute monocular pipeline or another text-model driver.
The [M8–M10 objective](openloris_m8_m10_plan.md) still requires COLMAP accuracy,
faster mapper/native E2E and lower RSS at 10k, plus tier/restart/100k checks.
This integration alone proves none of those outcome gates.

`rig_sfm::run_rig_bundle_adjustment` currently dispatches every local/final BA
to `BundleAdjustment::optimize`. Its legacy Schur construction expands each
landmark's cross blocks into pose-pair blocks. The existing implicit backend
avoids that expansion, but retains observation cross blocks, preconditioner
caches and rollback state. It is not constant-memory BA.

Ordinary rig mapping uses 40/60-frame windows; PCG may be slower there.
A full-map rig call also exists in `refine_rig_sfm_with_fixed_frame_rotations`,
but that rotation policy is not approved by this integration. Do not enlarge
windows, change rotation policy, or claim full-map savings from a small-window
test. The atlas integrator's separate capped BA is not changed in this PR.

## One default-off implementation

- Add `RigBaBackend::{Legacy, MatrixFreeStrict}` and a `RigSfmConfig` selector,
  default `Legacy`. The name Legacy deliberately preserves callers that set
  `BaConfig.linear_solver` themselves, rather than falsely labeling them Sparse.
- Expose `generalized_rig_sfm --ba-backend legacy|matrix-free`; option absent
  preserves the old output/log path. Keep the selector in the common native
  rig BA dispatch so local, final-window and full-map rig calls use it.
- Preserve caller robust loss, calibration, damping, iteration count, anchors,
  fixed rotations, observation construction, filtering and write-back policy.
  This does not promote the rejected adaptive, column-scaled or Huber-3 arms.
- Use `MatrixFreeBaOptions::default()`: PCG maximum 128, relative and absolute
  tolerances 1e-12, no restarts, no scaling or adaptive damping. No GT tuning.
- Matrix-free errors must not fall back to a large legacy Schur solve. Report
  requested/used backend and eligibility outcome explicitly, only when opted in.
- All poses fixed with variable landmarks needs an explicit, bounded
  landmark-only path: zero variable-pose Schur dimension, existing loss and
  observation semantics, no silently skipped BA. Restrict it to supported pure
  visual/fixed-calibration configuration; invalid configuration fails closed.
  No-variable problems are explicit no-ops, not invented successful updates.
- Validate and solve in the already-local BA problem. On error do not write
  partial solver state into mapper poses/tracks. No extra full-model clone,
  synthetic anchor, dense fallback, public `LinearSolver` variant or `BaConfig`
  change. Existing public rig APIs are re-exported consistently.
- Reuse the matrix-free validator for the zero-pose path through a minimal
  crate-private entry: only the requirement for a variable pose differs.
  Do not duplicate or weaken camera, loss, LM or factor eligibility checks.
  Successful eligibility/backend logs follow validation; a rejected request
  must not first be logged as an eligible successful dispatch.

## Correctness and first native comparison

Scoped tests cover selector defaults/CLI conflicts, unchanged legacy dispatch,
small synthetic rig matrix-free vs direct, fixed anchor/rotations/calibration,
observation retention, all-fixed landmark-only behavior, ineligible input and
no write-back on failure. Reuse solver tests rather than another oracle ladder.

Before candidate execution, certify source/binary and the actual 1k native
feature/snapshot/rig inputs from
[the path-specific champion](../benchmarks/electro/m8-openloris-visloc-rig-1k-champion.json).
Reproduce an option-absent current-main control first. If historical defaults
or inputs no longer reproduce it, resolve the discrepancy without changing
the reference or weakening its quality limits.

Then the candidate changes only `--ba-backend matrix-free`. Validate full
registration/support, image/keypoint identity, fixed calibration and anchor,
positive depth and each output's track consistency. Backend-dependent mapper
decisions can change track membership; record that difference explicitly rather
than pretending this is a frozen-observation BA comparison. Require deterministic
repeat and no loss of registered/supported images or frames.

Post-only quality limits: trajectory RMSE <= **0.02269532080131782 m**, p95 <=
**0.03777877644562742 m**, and independently evaluated raw mean reprojection
no worse than the reproduced champion (historically logged **0.671637382 px**).
Use the exact scorer/GT/calibration convention. No scale or tolerance sweep.
If quality passes, measure three unloaded runs against the same native control.
Report mapper and process timing separately: verified-snapshot replay still
excludes feature extraction/matching and is not full native E2E.

Failure stops promotion and larger performance claims. Passing only authorizes
the next same-profile tier assessment. All final COLMAP/10k/native-E2E/restart/
100k gates remain open, and README claims remain unchanged until verified.

## Reproduced native main control

Before implementation, root built the unmodified native code at main
`ece2b71`, binary SHA
`7a7446459817b492724f7b784894f826c1bbd1888637590a310d646a1ee39bcd`.
The actual champion rig manifest, features256 directory and verified snapshot
reproduce **all three historical model files byte-for-byte**. Independent
audits confirm 1,000 supported images, 500 supported frames, 2,658 landmarks,
27,716 observations, one component, fixed calibration and positive depth.
Post-only RMSE/p95 also exactly reproduce 0.02269532080131782 /
0.03777877644562742 m. Independent raw mean is 0.6716373819326764 px.

The historical feature-tree hash did not specify its algorithm; it is not
claimed equal to the new explicitly specified tree digest. Freeze the actual
1,000 files / 302,873,132 bytes with `run_official_baselines.directory_identity`
for candidate/control runs. Historical model and quality references stay
unchanged. Their exact reproduction establishes the required mapper control,
not historical equivalence of any ignored descriptor bytes.

The first correctness run is 4.42 s process / 3.864511 s mapper, external
peak RSS 81,916 KiB. This single run is not a speedup measurement. The writer's
historical zero point-ERROR placeholders remain unchanged; independently
recomputed residuals, not stored ERROR, define the quality gate.
[Certificate, inputs, commands and audits](../benchmarks/electro/m8-openloris-native-rig-matrix-free-preflight-v1.json)
were saved before candidate execution.

## Native candidate result — not promoted

Implementation `0298e0c` adds the strict selector without changing Legacy
defaults. Root independently passed 29 rig tests and 15 matrix-free API tests;
the example's 15 tests, scoped clippy and formatting checks also pass.
The release CLI rejects unknown and duplicate backend options.

The same candidate binary's option-absent output reproduces the main and
historical champion model bytes exactly. Both matrix-free runs also match
each other exactly, but fail the predeclared native quality limits:

| Metric | Legacy control | Matrix-free |
| --- | ---: | ---: |
| Supported images / rig frames | 1,000 / 500 | 1,000 / 500 |
| Trajectory RMSE (m) | 0.0226953 | 0.1402194 |
| Trajectory p95 (m) | 0.0377788 | 0.2345797 |
| Independent raw mean (px) | 0.6716374 | 0.7250106 |
| Landmarks / observations | 2,658 / 27,716 | 2,796 / 28,039 |
| Mapper time (s) | 3.345699 | 5.001141 / 4.850592 |
| Process time (s) | 3.81 | 5.45 / 5.28 |
| External peak RSS (KiB) | 81,876 | 82,260 / 82,396 |

Calibration, image identities, all 256,000 ordered keypoint coordinates,
positive depths, bidirectional references and one-component support pass.
Native track memberships can change; this is not frozen-observation BA.
The feature-tree digest remains unchanged after the runs.

Each candidate logs 86 backend calls, 688 iterations, 589 PCG failures,
71 accepted steps and 52 calls with no accepted step. These diagnostics
cover a different subset from the mapper's aggregate BA line. Failed PCG
steps are rolled back; eligibility does not mean numerical convergence.
This identifies a practical failure of the strict default policy, not proof
of a unique underlying cause or permission to relax the quality gate.

The correctness runs show no speed or memory improvement; the planned
three-run performance gate is not entered after quality failure. No 10k
promotion, tolerance sweep, hidden direct fallback or README performance
claim follows. Keep Legacy default. The next work must diagnose native
linear-solve failures before proposing a bounded, separately predeclared
policy; full native E2E and remaining M8–M10 gates stay open.

[Candidate certificate, commands, logs and independent audits](../benchmarks/electro/m8-openloris-native-rig-matrix-free-v1.json).

## Fixed-profile linear failure classification

After PR #93 merged (`3136b20`), one replay of the same certified binary and
inputs enabled existing BA/step/LM-quality debug output only. Solver settings
and thread count stayed unchanged. All three model files exactly match the
non-debug candidate; debug timing is not a performance measurement.

All 589 reported failed linear steps classify as **296 ResidualCheckFailed**
and **293 MaxIterations**. Residual-check failures used 46–128 iterations;
their true residual / target ratios range from 1.003329 to 1130.530454.
All iteration-limit failures used 128 iterations, with true residual / target
from 1.262194 to 20,774,208,042.164658. The two variants name the true residual
field differently (`true_norm` versus `residual_norm`); do not omit the latter
when reporting all-failure ranges. There were no
reported non-SPD, curvature, nonfinite or back-substitution failures.
The aggregate field named `pcg_failures` can include other linear-step
failures in general; this diagnostic establishes its actual contents here.

This narrows the immediate issue to stopping-residual reliability and bounded
convergence. It does not prove an incorrect operator or that raising iteration
limits / relaxing tolerances would recover quality. Compare existing restart
evidence before selecting one new native policy; no sweep or gate change.
[Classification, every failure record and read-only audit](../benchmarks/electro/m8-openloris-native-linear-failure-diagnostic-v1.json).

### Memory constraint on the next preconditioner

Do not construct the entire pose-pair Schur sparsity graph for IC(0): one
landmark visible in P poses can induce P(P-1)/2 blocks. Keeping only the
lower triangle or calling the pattern sparse does not bound it below O(P²).
The current implicit operator retains cross blocks, not an already-built
pose-pair graph; constructing that graph would undo the central memory saving.

[Ceres' preconditioner documentation](https://github.com/ceres-solver/ceres-solver/blob/master/docs/source/nnls_solving.rst)
describes Schur block-Jacobi and visibility-cluster alternatives, including
the cost of clustering. A possible next arm is **fixed-size cluster Jacobi**:
at most eight consecutive variable-pose slots per cluster, no global
visibility graph, cross-cluster Schur blocks omitted only from the
preconditioner, exact implicit action/RHS retained. This is a bounded
temporal approximation, not Ceres' visibility clustering implementation.
For fixed cluster size K, factor storage is O(PK), with local matrices at
most 6K × 6K. Group repeated rig-sensor cross blocks before forming local
Schur blocks. Do not enumerate all pairs of a long track before filtering.

Before implementation, verify the construction's O(observations × K) bound,
SPD handling and exact action/anchor tests. Keep PCG128, both tolerances
1e-12, loss, damping and all native quality gates unchanged. A failed factor
must fail explicitly, never fall back to a global direct solve. No improvement
is claimed until the native 1k comparison passes.

The default-off `matrix-free-cluster8` native selector is now implemented.
The operator action/RHS and scalar PCG/LM settings are unchanged. Construction
uses per-pose scratch reset only for touched slots, aggregates repeated sensor
rows, and loops over pairs only inside an at-most-eight-pose cluster. Local
Cholesky inversion uses at most 48 × 48 workspace; no full-model clone or
global pose-pair graph is added. Non-SPD clusters return a linear-step error.

Five scoped tests pass: principal-Schur equivalence with repeated sensors,
17-pose long-track/partial-cluster storage bounds, indefinite-cluster rejection,
API anchor/observation/repeat checks, and native fixed-rotation/landmark-only/
rollback behavior. Related rig/API/example tests and scoped clippy pass.
Release `8ba580c` and the same-input native 1k comparison are complete.
Both same-binary controls reproduce PR #93 model files exactly; the two
cluster8 output models also match each other byte-for-byte.

| Metric | Legacy | Strict matrix-free | Cluster8 |
| --- | ---: | ---: | ---: |
| Supported images / frames | 1000 / 500 | 1000 / 500 | 1000 / 500 |
| RMSE (m) | 0.0226953 | 0.1402194 | 0.2237376 |
| p95 (m) | 0.0377788 | 0.2345797 | 0.4324001 |
| Raw mean reprojection (px) | 0.6716374 | 0.7250106 | 0.7601143 |
| Mapper seconds | 4.302785 | 6.047962 | 6.283087 / 6.454014 |
| Process seconds | 4.87 | 6.59 | 6.85 / 7.01 |
| External RSS (KiB) | 81948 | 82448 | 82476 / 82492 |

Cluster8 fails all native quality gates despite preserving calibration,
positive depth, ordered image/keypoint identity and connected support.
It yields 2880 landmarks / 28825 observations. Linear-step failures decrease
only from 589 to 587, accepted steps fall from 71 to 59, and zero-accepted
calls increase from 52 to 54. Each backend changes subsequent mapper state;
these aggregate counts are not a controlled same-linear-system comparison.

The bounded-memory construction remains a tested implementation property,
not evidence of lower process RSS or improved native accuracy/speed.
No three-run performance gate, cluster-size sweep or 10k promotion follows.
Keep Legacy default and README performance claims unchanged.
[Certificate, all commands and independent audits](../benchmarks/electro/m8-openloris-native-cluster8-v1.json).
