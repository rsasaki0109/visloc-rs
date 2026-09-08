# Fixed boundary observations for native BA — implementation contract

Component anchoring improved 5k RMSE but worsened p95/reprojection. The
remaining singleton 1616 explains only 2.134% of squared trajectory error;
do not tune isolated-pose repairs or anchors against GT.

Current `run_rig_bundle_adjustment` discards observations outside the active
frame set, while updating the selected landmarks globally. The next single
factor is retaining their outside observations with fixed poses. Keep the
registration-order window, solver, thresholds, iterations and track policy
unchanged. Start from Legacy with component anchoring OFF to isolate this
factor, not a combined parameter sweep.

## Source and limits

COLMAP's [AddPointToProblem](https://raw.githubusercontent.com/colmap/colmap/main/src/colmap/estimators/bundle_adjustment_ceres.cc)
adds remaining observations of explicitly selected points using constant-pose
residuals. Its [local mapper](https://raw.githubusercontent.com/colmap/colmap/main/src/colmap/sfm/incremental_mapper.cc)
also uses covisibility to select local images. We borrow the fixed-boundary
principle only, not claim identical point selection or change covisibility
simultaneously. These main-branch sources are design references, not the
version certificate for the retained COLMAP benchmark.

## Implementation and validation

- Add a default-off explicit option. Select only positioned tracks having
  usable active-frame observations; never include boundary-only tracks.
- Keep the existing projectability/reprojection eligibility predicate.
  Include eligible registered outside observations on those tracks with
  fixed body poses and unchanged sensor calibration. Do not duplicate rows.
- Count pose support only for observations actually admitted to BA. Preserve
  legacy behavior when disabled; do not silently broaden the variable pose set.
- State must be O(selected observations + selected points + boundary poses).
  No global pose-pair graph, dense N-by-N state or full model clone. Fixed
  boundary poses must not enlarge the variable Schur dimension.
- Tests: one active observation plus fixed-boundary support; no eligible
  boundary; fully internal tracks; duplicate avoidance; fixed-state and
  rollback; variable dimension; disabled-path parity; long sparse tracks.
- Run same-binary control and two 5k candidate repeats. Audit full registration,
  support, positive depth, fixed rig, raw reprojection, whole-model GT and
  exact repeatability. Record actual variable/fixed counts and resources.
- Require evidence of useful quality improvement before larger-tier rollout.
  No GT-driven threshold or boundary-cap sweep. 10k and native E2E remain open.
