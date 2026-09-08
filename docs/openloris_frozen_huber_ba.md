# Frozen-observation Huber BA experiment

Status: predeclared, not measured. One candidate; no loss-scale sweep.

## Hypothesis and source boundary

The same-input adaptive and Ceres solves lowered squared reprojection cost
but worsened trajectory. Fixing 39 weak-angle XYZ values also worsened it
([fixed-point evidence](../benchmarks/electro/m8-openloris-weak-angle-fixed-landmarks-v1.json)).
Test whether reducing large-residual influence, without removing observations
or fixing landmarks, improves the pose/structure solution.

This is not a repeat of the earlier
[robust-triangulation experiment](../benchmarks/electro/m8-openloris-robust-triangulation-ab.json):
that changed tracks and observations (its 1k control has 270,204 observations,
versus 279,489 in the Huber-3 arm). It also tested Huber-1 BA and landmark-only
refinement on different populations; those failures remain valid, but do not
answer this exact frozen 130,900-observation question.

Use **Huber delta 3 px**, the existing engineering default in
`IncrementalSfmConfig::default`, not a parameter estimated from this GT.
[Ceres's primary modeling documentation](https://ceres-solver.readthedocs.io/latest/nnls_modeling.html#lossfunction)
defines robust scale in residual-norm units and describes the Huber loss.
It supports the loss semantics, not the claim that 3 px is statistically
optimal for OpenLORIS. This experiment uses the existing Rust IRLS solver,
not Ceres or COLMAP's potentially different local-BA loss/default settings.
Improvement is unproven; already-filtered residuals may give limited leverage.

## Frozen contract

- Same original 1k model, rig, frame-zero anchor and source SHA as PR #91:
  `ecb825c37e2aed47adef6b20e0d362544c56e37238861d5bcf80ed80cb915829`.
- Preserve all 1,000 images, 500 supported frames, 4,716 variable XYZ points,
  130,900 observations, 361,170 keypoints, identities/order and calibration.
  No point freezing, filtering, triangulation, pose prior or GT-based selection.
- Candidate changes only `RobustKernel::None` to `Huber { delta: 3.0 }` in
  adaptive column-scaled matrix-free BA. Twenty LM iterations, lambda 1e-4,
  PCG 512 / relative 1e-8 / absolute 1e-12 / restart zero, one thread.
  Existing scaling clamps, adaptive rho rule and depth feasibility unchanged.
- Driver-only opt-in `--huber-loss-3px`, restricted to adaptive column-scaled
  matrix-free mode; reject fixture export and fixed-landmark combinations.
  Option absent must preserve previous outputs and numerical traces.

## Cost accounting and memory

Separate full raw squared cost `C_raw = sum(s)` and optimized robust cost
`C_H = sum(rho(s))`, where `s` is each 2D residual's squared norm and
`rho(s) = s` for `s <= 9`, otherwise `6*sqrt(s)-9`. Rust reports sums without
Ceres's one-half factor. The IRLS weight is one below the threshold and
`3/sqrt(s)` above it. Raw decrease is diagnostic, not the LM acceptance rule.

The candidate's existing LM cost/acceptance/actual-reduction traces must be
explicitly labeled as robust; its local quadratic prediction is the weighted
IRLS surrogate. Do not label it as an unweighted raw squared objective.
Record initial/final raw and robust cost, valid/nonprojectable counts, raw
mean/RMSE/max and weight count/min/sum through scalar streaming accumulators.
Cross-check the robust sum against `BundleAdjustment::robust_cost` and solver
initial/final cost. No second model/normal clone or per-observation weight
array is needed. Existing per-iteration traces provide robust candidate
costs, not raw candidate costs; adding a state hook solely for that diagnostic
is outside this driver-only experiment.

Root independently recomputes final raw/robust costs, residual quantiles and
weights after timed solves. Exact p95 may use a bounded 130,900-scalar audit
array outside the timed solver; do not claim constant-memory exact quantiles.

## Runs, gates and next decision

Certify one reviewed/tested release binary and input hashes. Run four serial
processes with 180-second bounds: legacy None, adaptive None, adaptive Huber-3,
and exact Huber-3 repeat. Record process wall/RSS, exit status and numerical
traces. Controls must reproduce PR #91 models/traces; candidate repeat must
produce byte-identical model files and matching numerical/policy traces.

Before post-only GT scoring, verify complete identity/support, one component,
fixed calibration/anchor, positive depth for every observation, finite costs,
robust objective nonincrease and at least one accepted update. No observation
may disappear through robustification. Score the same 308 images with the
unchanged scorer/GT/transform files. Predeclared quality limits remain legacy
RMSE **0.026608055174816774 m**, trajectory p95 **0.04109998478546261 m** and
raw mean reprojection **0.691588658326868 px**, all no worse. Raw reprojection
RMSE/p95 are additional diagnostics, not newly invented trajectory gates.

Failure means no atlas/default promotion, no scale sweep. Success permits
the same candidate's bounded atlas assessment, not automatic native rollout.
Local BA timing excludes frontend, window mapping and atlas construction;
it cannot establish COLMAP mapper/native-E2E superiority. Full M8–M10 tier,
restart, 100k I/O and README comparison gates remain open.
