# Progress

This file tracks the project milestone completion used in development updates.

## Current Development Completion

**Deep VO / Loop Close completion: 100%**

This score means the file-backed two-view match VO path drives a short tracking
sequence end-to-end, a classical essential-matrix RANSAC frontend recovers
metric relative pose from the same external correspondences, loop-closure
candidates are geometrically verified through that classical frontend,
verified candidates lift into a `LoopClosureConstraint`, a sparse
`PoseGraph` consumes sequential and loop-closure edges and now exposes both
a translation-only Gauss-Newton step (fast linear baseline) and a full
SE(3) iterative Gauss-Newton solver (right-perturbation + first-order BCH)
that corrects rotations alongside translations, and the end-to-end
six-keyframe loop demo (`online_slam_pose_graph_loop_demo`) drives the
whole tracking + verifier + pose-graph stack on a single self-contained
synthetic sequence with measured drift correction for both pure-translation
and combined translation + rotation drift. The MVP scope is feature-complete;
remaining stretch work (public-data loop demos, learned frontends,
Levenberg-Marquardt damping, robust kernels, sparse Cholesky solvers) is
tracked as growth opportunities rather than gaps in the milestone.

Completed pieces:

- `VisualOdometryFrontend` and `VisualOdometryPriorProvider` boundaries exist.
- VO-derived pose priors can narrow tracking candidates.
- Externally generated two-view match files can be parsed.
- External two-view correspondences can produce a lightweight translation-only
  VO prior through `TwoViewMatchVisualOdometryFrontend`.
- File-backed two-view match files can drive `TwoViewMatchVisualOdometryFrontend`
  through `read_two_view_matches_txt`, and the resulting VO priors feed the
  external-prior tracking path on a short multi-frame sequence in the
  `track_sequence_with_two_view_match_vo_prior` example.
- A classical essential-matrix two-view geometry pipeline now lives in
  `visloc-vision::two_view` with a Hartley-normalized 8-point estimator,
  Sampson-distance scored RANSAC, and 4-fold cheirality disambiguation.
- `EssentialMatrixVisualOdometryFrontend` exposes that pipeline as a
  `VisualOdometryFrontend`, returning a full SE3 relative pose with
  caller-supplied translation scale; `two_view_vo_compare` runs the new
  frontend alongside the flow-only adapter on the same synthetic three-frame
  sequence to make the difference visible.
- A classical-geometry `EssentialMatrixLoopClosureVerifier` consumes the
  same essential-matrix RANSAC and reports `LoopClosureVerification` with
  inlier count, inlier ratio, mean Sampson error, score, recovered relative
  pose, and an enumerated failure reason. `verify_loop_closure_candidates`
  plus `correspondences_for_loop_candidate` plumb shared landmarks from the
  current frame's tracking inliers and an older keyframe's observations into
  the verifier without requiring `OnlineSlamPipeline` callers to change.
- `LoopClosureConstraint` (with `from_verified_candidate` /
  `loop_closure_constraints_from_candidates`) lifts each verified candidate
  into a stand-alone constraint (`from_keyframe_id`, `to_keyframe_id`,
  `relative_pose`, `inlier_count`, `inlier_ratio`, `mean_sampson_error`,
  `score`) that a future pose-graph backend can consume.
- `PoseGraph` skeleton (nodes = `BTreeMap<u64, Pose>`, edges =
  `PoseGraphEdge { from, to, measurement, kind, weight }`,
  anchor = `Option<u64>`) plus `PoseGraphEdgeKind::{Sequential, LoopClosure}`,
  builders (`add_pose`, `add_sequential_edge`, `add_loop_closure_constraint`,
  `anchor`, `relative_world_to_camera`), `translation_cost`, and a single
  translation-only Gauss-Newton step `optimize_translations_once` that holds
  rotations fixed and returns a `PoseGraphOptimizationStep` diagnostic. The
  step is exact for translation-only residuals.
- `online_slam_loop_candidate_with_verifier_dummy` example now also builds
  per-frame `LoopClosureConstraint`s and prints the recovered relative
  translation; the loop HTML/SVG report surfaces a separate Loop Closure
  Constraints table alongside the candidate diagnostics. The example also
  injects a small drift into the most recent keyframe, builds a `PoseGraph`,
  runs `optimize_translations_once`, and prints the cost / mean-correction /
  max-correction diagnostics so the loop drift correction is visible.
- `online_slam_pose_graph_loop_demo` example exercises the full pipeline on
  a six-keyframe synthetic loop: classical-tracker localization, verifier
  validation of the closed loop with the matching translation scale,
  `PoseGraph` construction with five sequential edges plus the verified
  loop-closure constraint, a translation-only Gauss-Newton step that
  pulls a `[0.06, 0.03, -0.05]` injected drift back to the loop-closed truth
  in one solve (`cost_before=0.105 → cost_after=0.000`, all six keyframes at
  `err=0.0`), and a follow-up full SE(3) iterative Gauss-Newton run that
  recovers from a combined `[0.04, 0, -0.03]` translation drift plus a
  `0.18 rad` yaw drift on the most recent keyframe in 2 iterations
  (`se3_cost_before=0.557 → 0.000`).
- `PoseGraph::optimize_se3_iterative` (with `PoseGraphSe3Config`,
  `PoseGraphSe3IterationStats`, and `PoseGraphSe3Result`) runs full SE(3)
  Gauss-Newton with right-perturbation updates `T_i ← T_i · Exp(δ_i)`,
  per-edge residual `r = log(meas⁻¹ · T_to · T_from⁻¹)`, and
  Jacobians `Ad(T_from)` (to-node) and `-Ad(T_from)` (from-node) under a
  first-order BCH approximation. `PoseGraph::se3_cost` reports the matching
  cost; `optimize_translations_once` remains the fast linear baseline.
- SE(3) Lie-group helpers (`SE3::log`, `SE3::exp`, `SE3::adjoint`,
  `so3_left_jacobian`, `so3_left_jacobian_inverse`) live in
  `visloc-core::geometry::se3` with Taylor fallbacks for small angles and
  `exp ∘ log` round-trip + adjoint-conjugation tests.
- A second loop-closure demo, `online_slam_public_loop_demo`, ingests a
  COLMAP-text-format sparse reconstruction from disk (defaulting to a
  synthesized 12-keyframe / 60-landmark orbit fixture written via
  `write_colmap_text_model`) and drives the full SLAM pipeline on the
  loaded data. With `--colmap-path <dir>` it loads any user-supplied
  reconstruction, reporting `se3_cost_before ≈ 8.3 → ≈ 1e-4` in 3
  iterations on a combined `[0.05, 0, -0.04]` translation + `0.18 rad`
  yaw drift. Synthetic per-landmark descriptors are generated when no
  `landmark_descriptors.txt` is supplied so the demo stays runnable on
  any registered COLMAP model.
- Online SLAM composition exists over tracking and local mapping.
- Loop-closure candidates can be detected from shared verified landmarks.
- Loop-candidate HTML/SVG reporting exists for synthetic sequence demos.

The milestone is feature-complete for its MVP scope. Stretch tasks tracked
beyond 100% (in `PLAN.md`) include:

- A public-data loop demo on real imagery (KITTI / COLMAP South Building) to
  replace the synthetic six-keyframe sequence.
- Levenberg-Marquardt damping plus robust kernels (Huber / Cauchy) on top of
  `optimize_se3_iterative`, and a sparse Cholesky / Schur-complement solver
  path so the optimizer scales beyond a handful of keyframes.
- Verifier reuse from PnP / tracking inliers via `PnPRansac` so candidates can
  be checked against the 3D map structure as well as essential-matrix
  two-view geometry.

## Reporting Rule

Development updates should report this value as:

```text
Deep VO / Loop Close completion: 100%
```

Increase the number only when a runnable example, test, or documented API
materially advances the Deep VO or loop-closure path.

## Rubric

- **0-20%:** map-based localization only; no sequence, VO, or loop direction.
- **20-40%:** tracking, local mapping, and SLAM composition boundaries exist.
- **40-60%:** VO frontend boundaries, external match IO, loop-candidate
  detection, and visible loop reports exist.
- **60-80%:** real external classical or learned frontend drives sequence
  tracking, and public demos show correspondences, pose continuity, and loop
  candidates clearly.
- **80-100%:** loop constraints, pose-graph hooks, regression datasets, and
  stable interfaces exist, while heavy runtimes remain optional integrations.
