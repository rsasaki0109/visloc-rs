# Frozen-observation Huber BA experiment

Status: measured; failed the predeclared quality gate. No atlas/default
promotion and no loss-scale sweep.

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

## Measured result (2026-09-08)

Implementation `c026f40` changes only the comparison driver. Luna Max's
28 example tests and example-only clippy passed; root independently repeated
both, plus the existing library Huber derivative and matrix-free rig tests.
The certified release binary SHA is
`cdbdf63466e9cca7940014d20605ec9485c566f184a86ac7c11dd5476fea23f3`.
All four processes exit zero after 20 iterations (`converged=false`).

| Arm | Wall s | Peak RSS KiB | Accepted LM | Raw squared cost | Huber-3 cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy None | 19.84 | 84,652 | 3 | 118070.554369 | 117471.290968 |
| Adaptive None | 22.26 | 84,856 | 15 | 113550.219339 | 112906.086062 |
| Adaptive Huber-3 | 22.00 | 85,156 | 13 | 113816.998930 | 112892.143080 |
| Huber-3 repeat | 22.43 | 85,108 | 13 | same | same |

The two controls reproduce PR #91's three model files and numerical traces
exactly (40/82 rows). Huber repeat reproduces its three model files and 85
numerical/policy rows exactly. The control Huber costs above are independent
evaluations of their byte-identical prior outputs, not their optimized loss.
No timing difference between these two adaptive arms is claimed significant.

Independent audits retain all 1,000 supported images, 500 supported frames,
4,716 points, 130,900 observations and 361,170 keypoints in one component.
Image/point/track/POINTS2D identities and order, cameras and fixed rig/anchor
are unchanged. All observations have positive depth. Independent final raw
and robust cost differ from the driver by only 6.55e-10 and 5.68e-10.
Initial/final downweighted counts are 1,984/1,617 and minimum weights
0.750081/0.543071. Downweighting never removes a residual.

Post-only GT scoring uses the unchanged 308-image set and agrees on repeat:

| Quality metric | Legacy limit | Huber-3 | Gate |
| --- | ---: | ---: | --- |
| Trajectory RMSE m | 0.026608055 | 0.028822200 | fail |
| Trajectory p95 m | 0.041099985 | 0.044018261 | fail |
| Raw mean reprojection px | 0.691588658 | 0.673262240 | pass |

The candidate slightly improves trajectory over adaptive None
(0.029190/0.044521 m), but still fails both legacy trajectory limits.
Raw reprojection RMSE/p95/max are 0.932468/1.991369/5.524140 px;
its raw squared cost rises relative to adaptive None even though mean and
p95 decrease. Lower robust loss is not evidence of lower raw squared loss
or a passing trajectory. Maximum point motion is 221.120 m, and maximum
camera-centre motion is 0.024035 m; neither substitutes for GT accuracy.

Reject the candidate without another scale or atlas run. Library/default
solver and README performance claims remain unchanged. Native mapper/E2E,
10k COLMAP parity, tier/restart and 100k I/O gates are still open.

[Preflight and independent input accounting](../benchmarks/electro/m8-openloris-frozen-huber-ba-preflight-v1.json)
and [all four runs, commands, hashes and audits](../benchmarks/electro/m8-openloris-frozen-huber-ba-v1.json)
record the completed experiment. Full logs remain under the certified local
artifact root named there; the evidence stores their hashes and a reproducible
compact trace auditor rather than duplicating every control trace.
