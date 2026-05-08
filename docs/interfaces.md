# Interfaces

## Core Types

Use `visloc_rs::prelude::*` for the common application-facing imports. The prelude includes the core map/query/pose types, the default localization pipeline, provider traits, default matching/PnP/RANSAC pieces, tracking/mapping/SLAM/fusion handles, and small IO helpers. Explicit crate modules remain available when applications want narrower imports.

- `Camera`: intrinsic model and projection helpers
- `Pose`: world-to-camera pose backed by `SE3`
- `Frame`: query or temporal image data
- `Keyframe`: frame with map observations
- `Landmark`: 3D visual map point with optional descriptor
- `LandmarkDescriptorStore`: descriptor storage keyed by landmark id, usable with descriptors loaded outside the visual map
- `Observation`: 2D measurement of a landmark in a frame
- `VisualMap`: cameras, keyframes, and landmarks, with validation helpers for structural references and localization descriptors
- `VisualMapValidationReport`: non-panicking validation diagnostics for loaded or generated maps
- `QueryImage`: query keypoints and descriptors
- `LocalizationResult`: pose, failure reason, candidate landmark count, match count, correspondence count, inlier/outlier counts, inlier ratio, inlier correspondence indices, inlier query indices, inlier landmark ids, reprojection error statistics, and optional pose-estimator diagnostics
- `PoseEstimatorDiagnostics`: records whether non-linear refinement ran and the mean reprojection error before/after refinement when available
- `PoseEstimationFailureDiagnostics`: records pose-estimation failure context such as minimum correspondence count, RANSAC iterations, threshold, best inlier count, and failure category

## Pipeline

```rust
fn localize(query: QueryImage, map: VisualMap) -> LocalizationResult
fn localize_frame(frame: Frame, map: VisualMap) -> LocalizationResult
fn localize_frames(frames: Vec<Frame>, map: VisualMap) -> Vec<FrameLocalizationResult>
```

`QueryImage` is the direct image-query input. `Frame` can also be localized when its `camera_id` exists in the `VisualMap`; batch frame localization is a stateless sequence entry point for tracking-oriented callers.

`MapProvider` and `DescriptorProvider` describe how a map and optional descriptor store are supplied to localization/tracking. `map_provider_stats` exposes lightweight map diagnostics (`MapProviderStats`) for cameras, landmarks, keyframes, and descriptors. `InMemoryMapProvider` is the initial implementation for tests, examples, COLMAP-loaded maps, and small experiments. It can also materialize landmark-id or radius-filtered submaps from another provider. Future providers can wrap databases, retrieval-selected submaps, or GNSS-prior local map windows.

`SubmapSelector` separates local-map selection from localization. `AllMapSelector`, `FixedLandmarkSubmapSelector`, and `RadiusSubmapSelector` can be used with `SelectableMapProvider` to expose a selected submap through the normal provider traits.

`LocalizationPrior` represents optional prior pose or world position plus a radius. It can be converted into a `RadiusSubmapSelector` or used through `PriorSubmapSelector`, giving GNSS/VIO/last-pose hints a common input shape without implementing sensor fusion yet.

`visloc_io::colmap::ColmapMapProvider` loads a COLMAP text or binary model and optional landmark descriptor text file, then exposes them through the same provider traits. It also provides `validate_map` for structural checks and `validate_for_localization` for descriptor-aware checks; the `*_validated` constructors return an error when those checks fail. `write_colmap_text_model` saves a `VisualMap` back to COLMAP text files so maps updated by online pipelines can be reused by later localization runs.

`visloc_io::calibration` reads KITTI-style calibration text files. `parse_kitti_calibration_txt` extracts projection rows such as `P0` and `P2`, and `read_kitti_pinhole_camera` converts a selected row into a `Camera::pinhole` when the image dimensions are supplied by the caller. This is intentionally a small automotive-dataset bridge, not a full calibration database.

With the `image-io` feature enabled, `visloc_io::kitti` can load a KITTI-style image directory together with optional timestamp text and calibration. `KittiImageSequence` returns the selected `Camera`, loaded frames, sequence summary, dimension issues, and timestamp issues so automotive demos can validate the dataset before starting localization or tracking.

`FeatureSet` stores externally supplied keypoints and descriptors and can validate shape/dimension consistency with `FeatureSet::new`. `FeatureExtractor` can be connected through `ImageLocalizer` or the lower-level `LocalizationPipeline::localize_image_with_extractor` methods. `ProvidedFeatureExtractor` wraps precomputed features, and `FnFeatureExtractor` wraps a closure so OpenCV, learned features, or custom image pipelines can be connected without making those dependencies mandatory. `GrayscaleImage` and `CornerFeatureExtractor` provide a small dependency-free image-input extractor for smoke tests and examples; production feature pipelines should still plug in through the same trait boundary.

`visloc_io::images` provides dependency-free PGM grayscale IO through `read_pgm`, `parse_pgm`, `write_pgm_ascii`, and `to_pgm_ascii`. This is intended for fixtures, smoke tests, and minimal examples rather than broad image-format coverage.

With the `image-io` feature enabled, `visloc_io::images` also provides optional PNG/JPEG-backed grayscale loading through `read_common_image` and `decode_common_image`, plus `write_png_gray` for examples and fixtures. `read_common_image_sequence`, `read_common_image_sequence_with_timestamps`, `read_common_image_sequence_dir`, and `read_common_image_sequence_dir_with_timestamp_file` load ordered image frames for `ImageTracker` sequence experiments. `parse_timestamp_nanoseconds_txt` / `read_timestamp_nanoseconds_txt` read simple timestamp files where each non-comment line starts with a nanosecond timestamp. `common_image_sequence_summary`, `validate_common_image_sequence_dimensions`, and `validate_common_image_sequence_timestamps` report frame counts, image size, timestamp coverage/order, and mixed-dimension inputs before tracking. This keeps the default crate dependency-light while allowing real camera-image files to enter the same `GrayscaleImage -> FeatureExtractor -> ImageLocalizer/ImageTracker` path.

`CorrespondenceBuilder` converts `QueryImage + VisualMap + LandmarkDescriptorStore` into 2D-3D correspondences before pose estimation. The localization pipeline uses this builder internally.

`Matcher` is the descriptor matching boundary. `BruteForceMatcher` provides L2 nearest-neighbor matching with optional ratio test, and `CrossCheckMatcher` wraps another matcher to keep only mutual nearest matches. `DescriptorMatch` records query/train indices, best distance, second-best distance, and ratio diagnostics when available. `CorrespondenceSet` preserves those descriptor matches alongside 2D-3D correspondences for debugging.

`CandidateSelector` controls which landmark ids are eligible before descriptor matching. The default selector is `AllLandmarksSelector`; `FixedLandmarkSelector` is useful for tests and early experiments. `RadiusLandmarkSelector` selects landmarks near a world-space prior and is the first hook for GNSS/prior-pose style narrowing. Future selectors can use covisibility, retrieval, prior pose, GNSS, or inertial hints.

`IntersectCandidateSelector` combines two selectors by landmark-id intersection. Tracking uses this to preserve the pipeline's configured selector while adding a last-pose radius prior for a single localization call.

`PoseEstimator` is the minimal PnP boundary and `PoseRefiner` is the optional non-linear refinement boundary. The default robust estimator uses `DltPnP` inside `PnPRansac`, then applies `GaussNewtonPoseRefiner` to the final inlier set. Future implementations can wrap EPnP, AP3P, OpenCV, motion-prior estimators, or test doubles.

`RobustPoseEstimator` is the RANSAC-style pose-estimation boundary used by `LocalizationPipeline`. The default implementation is `PnPRansac`.

`LocalizationConfig` includes a quality gate. A pose can be estimated but rejected with `QualityGateFailed` if it does not meet inlier count, inlier ratio, or reprojection error thresholds. Rejected results retain pose and diagnostics for logging and debugging.

The default pipeline:

1. Collects landmark descriptors from a `LandmarkDescriptorStore`
2. Matches query descriptors to map descriptors
3. Builds 2D-3D correspondences
4. Runs PnP RANSAC
5. Refines the final inlier pose with Gauss-Newton reprojection minimization
6. Returns pose, inliers, inlier ratio, and reprojection error statistics

The default `localize(query, map)` path builds a descriptor store from `Landmark.descriptor` for convenience. For COLMAP-style maps where descriptors are stored separately, use `localize_with_descriptor_store`.

## Tracking Skeleton

`visloc-tracking` provides a localization-based tracking scaffold:

- `TrackingState`: `Uninitialized`, `Tracking`, or `Lost`
- `TrackingEvent`: `Initialized`, `Tracked`, `TrackingFailed`, `Lost`, or `Relocalized`
- `TrackingResult`: localization result plus state transition, motion pose prior, external localization-prior diagnostics, map provider stats, map landmark count convenience field, and optional tracking failure reason
- `TrackingConfig`: controls `min_successive_failures_to_lost`, optional `last_pose_candidate_radius`, and tracking quality gates for pose-prior translation error, minimum inliers, minimum inlier ratio, and maximum mean reprojection error
- `TrackingFailureReason`: records tracking-layer quality gate failures such as insufficient inliers, low inlier ratio, high reprojection error, or pose jumps that exceed the motion-prior translation threshold
- `TrackingStats`: sequence diagnostics including first/last frame id, success/failure counts, lost/relocalization counts, motion-pose-prior and external-localization-prior usage, tracking quality-gate failures, aggregate inlier/correspondence totals with rate helpers, and JSON summary export
- `TrackingEvaluationConfig`, `TrackingEvaluationResult`, and `TrackingEvaluationFailure`: turn tracking summaries into pass/fail checks for success rate, failure rate, lost count, tracking quality-gate failures, external prior usage, inlier ratio, and mean inlier count
- `TrackingStats::from_results`: rebuilds sequence-level diagnostics from stored `TrackingResult` values
- `tracking_results_to_csv`: exports frame-by-frame tracking state, localization counts, failures, priors, and map stats for downstream plotting or regression checks
- `tracking_results_to_html_report`: creates a self-contained HTML tracking report with per-frame state, inlier diagnostics, failures, and a timeline
- `PoseTrajectory` and `TrajectorySample`: convert successful `TrackingResult` values into frame ids, poses, camera centers, path length, reprojection-error summaries, CSV output, KITTI-style 3x4 pose rows, and TUM-style trajectory rows for sequence demos and evaluation export
- `PoseTrajectory::from_kitti_poses_str` / `read_kitti_poses`: read KITTI-style 3x4 pose rows back into a trajectory, using row index as frame id
- `PoseTrajectory::from_tum_poses_str` / `read_tum_poses`: read TUM-style `frame_id tx ty tz qx qy qz qw` rows back into a trajectory for simple file-based evaluation
- `TrajectoryErrorSummary` and `TrajectoryTranslationError`: compare an estimated trajectory against a reference trajectory by matching frame ids and reporting translation ATE-style mean, RMSE, max, missing-pose counts, CSV rows, and JSON summaries
- `TrajectoryEvaluationConfig`, `TrajectoryEvaluationResult`, and `TrajectoryEvaluationFailure`: turn trajectory-error summaries into benchmark-style pass/fail results using optional mean/RMSE/max translation-error thresholds, minimum matched-pose counts, and minimum match ratio
- `TrajectoryAlignment::FirstMatchedTranslation`: optionally removes the translation offset at the first matched frame before computing trajectory translation errors
- `PoseTrajectory::to_html_report`: creates a self-contained HTML / SVG trajectory report for sequence-localization demos and quick inspection
- `PoseTrajectory::to_html_report_against`: creates a self-contained HTML / SVG trajectory-evaluation report with reference-pose comparison
- `MotionModel`: predicts an optional pose prior for the next frame
- `ConstantPoseMotionModel`: default motion model that reuses the last successful pose
- `ConstantVelocityMotionModel`: extrapolates the next camera center from the two latest successful poses
- `VisualOdometryFrontend`: optional two-frame frontend boundary for classical or learned VO integrations
- `VisualOdometryEstimate`: relative frame motion, match/inlier diagnostics, and helper for turning a previous pose into a current-frame pose prior
- `VisualOdometryPriorProvider`: turns a frontend's relative pose estimate plus the previous absolute pose into a current-frame pose prior
- `VisualOdometryPosePrior`: bundles the generated pose prior with the VO estimate diagnostics
- `NoopVisualOdometryFrontend`: default no-estimate implementation for callers that want to wire the interface before adding a real frontend
- `TwoViewMatchVisualOdometryFrontend`: converts externally supplied two-view correspondences into a lightweight translation-only VO prior without adding a model runtime dependency
- `EssentialMatrixVisualOdometryFrontend`: classical-geometry VO frontend that runs the `visloc-vision::two_view` essential-matrix RANSAC + cheirality recovery on externally supplied correspondences, returning a full `SE3` relative pose with caller-supplied translation scale
- `Tracker`: feeds frames through a localization pipeline and updates state from success/failure
- `ImageTracker`: extracts features from image inputs and feeds generated frames into `Tracker`

`Tracker` also exposes `last_result`, `last_successful_frame_id`, `last_successful_pose`, and next-frame `LocalizationPrior` helpers for caller-side diagnostics or lightweight temporal consumers. `track_frames`, `track_frames_with_provider`, and `track_frames_with_descriptor_store` process a slice of frames while preserving tracker state across the sequence. `reset` clears sequence state, history, stats, and motion-model history so a caller can start a new sequence explicitly. `ImageTracker` mirrors those sequence and reset APIs for image inputs by extracting features per image before updating tracker state. `track_frame_with_prior_submap_provider` and the matching `ImageTracker` method can use the motion prior to create a temporary radius submap before localization. `VisualOdometryPriorProvider` is the bridge for learned or classical VO frontends: callers can estimate a two-frame motion prior, then pass that prior through the existing external-prior path. The `two_view_match_vo_prior_dummy` example demonstrates the first file-backed path from external correspondences to a VO pose prior; `track_sequence_with_visual_odometry_prior` demonstrates using VO-derived priors to reduce candidate landmarks before localization. `track_sequence_with_two_view_match_vo_prior` extends that to a short three-frame sequence by reading per-pair two-view match files with `read_two_view_matches_txt`, populating `TwoViewMatchVisualOdometryFrontend`, and feeding each prior through `track_frame_with_localization_prior_submap_provider` so the file-backed external matcher path is exercised end-to-end. `track_frame_with_localization_prior_submap_provider` accepts an external `LocalizationPrior`, such as a GNSS or odometry prior from `visloc-fusion`, to create a temporary radius submap even before visual tracking has a last pose. When `last_pose_candidate_radius` is set, successful localization stores the last pose and later frames use its camera center as a temporary radius prior for landmark selection. Tracking quality gates can reject a pose estimate when inlier count, inlier ratio, mean reprojection error, or pose-prior translation error is outside the configured thresholds while retaining the localization diagnostics.

This is not SLAM: it does not create keyframes, update maps, run bundle adjustment, or estimate map structure.

## Local Mapping Skeleton

`visloc-mapping` starts the local-mapping layer with explicit keyframe decisions and staged map edits:

- `KeyframePolicy`: evaluates a `TrackingResult` and returns a keyframe-selection decision
- `SimpleKeyframePolicy`: selects the first successful frame, optional relocalized frames, and later frames that pass frame-id gap and camera-translation thresholds
- `KeyframePolicyConfig`: controls minimum frame-id gap, minimum translation, and relocalized-frame selection
- `KeyframeDecision`: reports whether the frame was selected, why it was selected or rejected, and the current keyframe-policy counters
- `StagedMapUpdate`: collects keyframes, landmarks, and observations before mutating a `VisualMap`
- `MapUpdateValidationReport`: reports duplicate staged entities, existing-map conflicts, missing references, and keypoint bounds errors
- `AppliedMapUpdate`: counts how many staged entities were applied after validation
- `LocalMapWindow`: selects a recent keyframe window and the landmarks observed inside it
- `LocalMapWindowConfig`: controls local window size
- `LandmarkCandidate`: represents an untriangulated landmark hypothesis from multiple keyframe observations
- `LandmarkCandidateValidationReport`: checks observation count, keyframe references, keypoint bounds, duplicate observations, and optional local-window membership
- `Triangulator`: converts a valid landmark candidate into a triangulated landmark through a swappable backend
- `LinearTriangulator`: minimal normalized DLT triangulator for bootstrapping local mapping experiments
- `LocalMappingPipeline`: composes keyframe selection, local window construction, candidate validation, triangulation, and staged map updates without applying mutations itself
- `LocalRefiner`: optional local optimization hook that can adjust staged keyframes/landmarks before validation; `NoopLocalRefiner` is the default

This is the first v0.3 extension point. Future landmark candidates, triangulation, and local refinement should consume these decisions and staged updates instead of being baked into tracking.

## Online SLAM MVP

`visloc-slam` composes tracking and local mapping with lightweight loop-closure candidate diagnostics, but without global pose-graph optimization:

- `OnlineSlamPipeline`: owns a `VisualMap`, a `Tracker`, and a `LocalMappingPipeline`
- `OnlineSlamConfig`: controls whether valid staged updates are applied immediately and configures loop-closure candidate thresholds
- `LoopClosureConfig`: controls candidate detection using frame-id gap, shared-landmark count, shared-landmark ratio, and max returned candidates
- `LoopClosureCandidate`: reports the current frame, matched older keyframe, shared landmark count, overlap ratio, score, and geometric verification flag
- `OnlineSlamResult`: returns tracking output, optional mapping output, optional applied update counts, loop-closure candidates, and current map sizes
- `online_slam_results_to_html_report`: creates a self-contained HTML/SVG report showing tracked camera centers and loop-candidate edges, including verifier inlier/Sampson columns when `LoopClosureCandidate::verification` is populated
- `LoopClosureVerifier`: trait for swappable loop-closure verifiers operating on pixel-space `TwoViewCorrespondence`s plus camera intrinsics
- `EssentialMatrixLoopClosureVerifier`: classical-geometry verifier built on `visloc-vision::two_view`'s essential-matrix RANSAC, with `LoopClosureVerifierConfig` thresholds for `min_inliers`, `min_inlier_ratio`, `max_mean_sampson_error`, and a `default_translation_scale` applied when recovering the relative pose
- `LoopClosureVerification` and `LoopClosureVerificationFailureReason`: verifier outputs covering inlier count, inlier ratio, mean Sampson error, combined score, recovered relative pose (`Option<SE3>`), and an enumerated failure reason
- `LoopClosureConstraint`: pose-graph-style constraint built from a verified candidate (`from_keyframe_id`, `to_keyframe_id`, `relative_pose`, `inlier_count`, `inlier_ratio`, `mean_sampson_error`, `score`); intentionally a data type without a solver so downstream optimization layers can adopt it incrementally
- `PoseGraph`, `PoseGraphEdge`, `PoseGraphEdgeKind`, `PoseGraphOptimizationStep`, `PoseGraphError`: sparse pose-graph keyed by keyframe id with sequential and loop-closure edges. `optimize_translations_once` is a single translation-only Gauss-Newton step that holds rotations fixed and is exact for translation-only residuals (kept as a fast linear baseline); `optimize_se3_iterative` is the full SE(3) iterative solver
- `PoseGraphSe3Config`, `PoseGraphSe3IterationStats`, `PoseGraphSe3Result`: configuration (max iterations, step / cost tolerances, robust kernel selection, optional Levenberg-Marquardt damping schedule), per-iteration cost / step / `λ` / accept-reject diagnostics, and run summary for `PoseGraph::optimize_se3_iterative`. The solver uses right-perturbation updates `T_i ← T_i · Exp(δ_i)` with a first-order BCH approximation (`J_r⁻¹ ≈ I`); per edge, residual is `r = log(meas⁻¹ · T_to · T_from⁻¹)` with Jacobians `Ad(T_from)` (to-node) and `-Ad(T_from)` (from-node). `PoseGraph::se3_cost` reports the matching squared-error cost; `PoseGraph::robust_se3_cost(&kernel)` reports the kernel-shaped cost when a `RobustKernel` is in use. The inner normal-equations solve prefers Cholesky and falls back to LU on ill-conditioned systems
- `RobustKernel`: per-edge IRLS cost selector (`None` / `Huber { delta }` / `Cauchy { c }`) used by `optimize_se3_iterative`. `cost(s)` returns `ρ(s)` and `weight(s)` returns the influence multiplier `ρ'(s)` applied to each edge's normal-equations contribution
- `PnPLoopClosureVerifier`, `PnPLoopClosureVerifierConfig`: PnP-based loop-closure verifier built on `visloc-vision::ransac::PnPRansac`. Operates on 2D-3D correspondences (current frame pixel ↔ landmark world point) and re-localizes the current frame against landmarks observed by the candidate keyframe. Returns a metric relative SE(3) directly (no scale parameter needed) plus inlier count / ratio / mean reprojection error in pixels. The companion `correspondences_2d3d_for_loop_candidate` helper intersects the current frame's tracking inliers with the older keyframe's observed landmarks; `verify_loop_closure_candidates_pnp` runs the verifier over a slice of candidates and updates each `verification` and `geometrically_verified` field in place
- `SE3::log` / `SE3::exp` / `SE3::adjoint` plus public `so3_left_jacobian` and `so3_left_jacobian_inverse`: SE(3) Lie-group helpers in `visloc-core::geometry::se3`. The tangent layout is `[ρ; ω]` (translation first, then rotation); both helpers fall back to Taylor series for small angles
- `relative_world_to_camera`: helper that derives the `previous_to_current` SE3 measurement between two `Pose`s for use as a `PoseGraphEdge::measurement` value
- `correspondences_for_loop_candidate`: helper that builds two-view correspondences for a candidate from the current frame's tracking inliers and the older keyframe's observations
- `verify_loop_closure_candidates`: convenience helper that runs a `LoopClosureVerifier` over a slice of `LoopClosureCandidate`s in place, updating each `verification` and `geometrically_verified` field
- `loop_closure_constraints_from_candidates`: builds `LoopClosureConstraint`s from a slice of candidates, silently dropping unverified candidates and those without a recovered relative pose

Each frame is localized/tracked first. If tracking succeeds, the pipeline creates a keyframe from the tracked frame, runs local mapping with caller-supplied landmark candidates, and optionally applies the validated staged update to the growing map. If tracking fails, mapping is skipped and the caller still receives the tracking diagnostics.

Loop-closure candidates are deliberately diagnostic at this layer. They identify likely returns to older keyframes using shared verified landmarks, but they do not yet add pose-graph constraints or correct the map globally.

The `online_slam_loop_candidate_dummy` example shows this diagnostic path on a tiny synthetic sequence: an older keyframe is inserted first, a later frame observes the same landmarks, and the pipeline reports a loop candidate with overlap score and verification status. With `--out-dir`, it writes `loop_report.html` so the candidate edge is visible in a browser.

## Sensor Fusion Foundation

`visloc-fusion` provides loose-coupling inputs for automotive and UAV localization without implementing a full GNSS/INS backend:

- `Timestamp`, `TimeDelta`, and `Timed<T>` represent timestamped measurements and frame metadata.
- `TimedFrame`, `TimedPose`, and `FrameTimestampIndex` attach sensor time to existing core frames and poses without changing the core `Frame` or `Pose` structs.
- `MeasurementBuffer<T>` keeps timestamped measurements ordered, finds the latest or nearest measurement, and can resolve a frame's nearest external prior through `FrameTimestampIndex`.
- `FramePriorSource<T>` packages frame timestamps, measurement buffers, sync tolerance, and prior radius configuration for per-frame external prior lookup.
- `FramePriorSyncSummary` reports frame/measurement counts, matched frames, missing measurements, and sync ratio for debugging real automotive/UAV sensor logs.
- `FramePriorSyncEvaluationConfig` turns sync summaries into pass/fail results with minimum matched-frame count or ratio thresholds. `FramePriorSyncEvaluationResult` can be exported as JSON for demo smoke checks.
- `PositionCovariance`, `PoseCovariance`, and `PoseCovarianceMatrix` preserve uncertainty for future fusion backends while still supporting simple radius-based localization priors.
- `GnssMeasurement` stores a world-position prior plus optional horizontal/vertical accuracy or 3D position covariance.
- `PosePriorMeasurement` stores an external pose prior, such as odometry, VIO, or a previous fused estimate, plus optional translation sigma or 6D pose covariance.
- `ImuMeasurement` stores angular velocity, linear acceleration, and optional orientation.
- `LocalizationPriorProvider` converts GNSS or pose-prior measurements into `LocalizationPrior`; with `MeasurementBuffer`, the nearest external prior can drive radius submap selection in localization/tracking.
- `PriorConfig` controls default radius, minimum radius, and confidence multiplier.

`visloc_io::sensors::read_gnss_measurements_txt` loads simple whitespace- or comma-separated GNSS prior logs in the form `timestamp_ns x y z [horizontal_accuracy] [vertical_accuracy]`. This keeps public automotive/UAV demos file-backed without choosing a full GNSS/INS format yet.

This is intentionally loose coupling: visual-only users do not need fusion types, and robotics users can plug in their own GNSS/INS/VIO stack while still guiding visual localization.

## Descriptor Store Text Format

```text
# LANDMARK_ID D0 D1 D2 ...
1000 0.1 0.2 0.3
1001 1.0 0.0 0.5
```

Use `visloc_io::descriptors::read_landmark_descriptors_txt` for the first experimental path. This format is not intended to be the final large-scale descriptor backend.

## Query Feature Text Format

```text
# X Y D0 D1 D2 ...
195.0 115.0 0.0 1.0 0.5
431.1 128.9 1.0 1.0 0.5
```

Use `visloc_io::query_features::read_query_features_txt` for file-based experiments. `examples/localize_from_files.rs` combines this query format with a COLMAP text model and landmark descriptor file.

## Two-View Geometry

`visloc-vision::two_view` ships the classical two-view geometry pieces used by
`EssentialMatrixVisualOdometryFrontend`:

- `TwoViewCorrespondence`: pixel-space correspondence between the previous and
  current frame.
- `EssentialMatrixEstimator` / `EightPointEssentialMatrixEstimator`:
  Hartley-normalized 8-point essential-matrix estimator. Pixels are first
  normalized with the camera intrinsics, then Hartley-normalized for
  conditioning before solving the linear system on `A^T A` so the smallest
  right singular vector is always available.
- `EssentialRansac`: Sampson-distance scored RANSAC that returns the best
  essential matrix, inlier indices, and mean Sampson error.
- `recover_relative_pose`: SVD-based decomposition of the essential matrix
  into the four (R, t) candidates, with cheirality scoring to pick the
  candidate that puts the most inliers in front of both cameras.
- `RelativePoseEstimator`: composes the RANSAC and recovery steps and applies a
  caller-supplied translation scale (`default_translation_scale` or per-pair
  scale through `EssentialMatrixVisualOdometryFrontend::insert_matches_with_scale`)
  because translation is recovered up to a scalar.

This module deliberately avoids OpenCV / OpenGV dependencies. Heavier or more
specialized two-view solvers can be added as separate frontends without
touching this baseline.

## Two-View Match Text Format

```text
# PREV_IDX CURR_IDX PREV_X PREV_Y CURR_X CURR_Y [SCORE]
0 3 120.0 140.0 124.5 141.0 0.99
1 9 260.0 180.0 263.0 183.5 0.94
```

Use `visloc_io::two_view_matches::read_two_view_matches_txt` for early external-frontend experiments. This format is intended for outputs from learned or classical matchers, such as SuperPoint/LightGlue-style pipelines, without adding a model runtime dependency to `visloc-rs`.
