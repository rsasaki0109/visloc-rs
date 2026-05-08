# Progress

This file tracks the project milestone completion used in development updates.

## Current Development Completion

**Deep VO / Loop Close completion: 90%**

This score means the file-backed two-view match VO path drives a short tracking
sequence end-to-end, a classical essential-matrix RANSAC frontend recovers
metric relative pose from the same external correspondences, loop-closure
candidates are geometrically verified through that classical frontend,
verified candidates lift into a `LoopClosureConstraint`, a sparse
`PoseGraph` skeleton consumes sequential and loop-closure edges and runs a
translation-only Gauss-Newton step that pulls drifted nodes back along the
verified loop, and an end-to-end six-keyframe loop demo
(`online_slam_pose_graph_loop_demo`) drives the whole tracking + verifier +
pose-graph stack on a single self-contained synthetic sequence with measured
drift correction. The project has not yet reached a real learned-frontend
sequence demo on public-data imagery or full SE3 pose-graph optimization with
rotation updates.

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
  loop-closure constraint, and a translation-only Gauss-Newton step that
  pulls a `[0.06, 0.03, -0.05]` injected drift back to the loop-closed truth
  in one solve (`cost_before=0.105 → cost_after=0.000`, all six keyframes at
  `err=0.0`).
- Online SLAM composition exists over tracking and local mapping.
- Loop-closure candidates can be detected from shared verified landmarks.
- Loop-candidate HTML/SVG reporting exists for synthetic sequence demos.

Remaining pieces before this milestone is considered complete:

- Feed a real classical or learned two-view frontend into the tracking demo.
- Run a public sequence demo that shows denser correspondences and smoother
  frame-to-frame motion.
- Add stronger geometric verification and diagnostics for loop candidates.
- Add pose-graph constraint hooks while keeping global optimization optional.
- Keep full pose-graph optimization out of the core until the candidate layer is
  proven by demos and tests.

## Reporting Rule

Development updates should report this value as:

```text
Deep VO / Loop Close completion: 90%
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
