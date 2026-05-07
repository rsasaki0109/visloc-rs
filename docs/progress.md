# Progress

This file tracks the project milestone completion used in development updates.

## Current Development Completion

**Deep VO / Loop Close completion: 70%**

This score means the file-backed two-view match VO path drives a short tracking
sequence end-to-end, a classical essential-matrix RANSAC frontend recovers
metric relative pose from the same external correspondences, loop-closure
candidates are geometrically verified through that classical frontend, and
verified candidates now lift into a `LoopClosureConstraint` type ready for a
future pose-graph layer. The project has not yet reached a real
learned-frontend sequence demo or pose-graph optimization.

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
  `score`) that a future pose-graph backend can consume. No solver lives
  in this crate yet.
- `online_slam_loop_candidate_with_verifier_dummy` example now also builds
  per-frame `LoopClosureConstraint`s and prints the recovered relative
  translation; the loop HTML/SVG report surfaces a separate Loop Closure
  Constraints table alongside the candidate diagnostics.
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
Deep VO / Loop Close completion: 70%
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
