# Native rig matrix-free integration

Status: predeclared; implementation and native A/B not yet completed.

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
