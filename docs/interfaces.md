# Interfaces

## Core Types

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

`visloc_io::colmap::ColmapMapProvider` loads a COLMAP text or binary model and optional landmark descriptor text file, then exposes them through the same provider traits. It also provides `validate_map` for structural checks and `validate_for_localization` for descriptor-aware checks; the `*_validated` constructors return an error when those checks fail.

`FeatureSet` stores externally supplied keypoints and descriptors and can validate shape/dimension consistency with `FeatureSet::new`. `FeatureExtractor` can be connected through `ImageLocalizer` or the lower-level `LocalizationPipeline::localize_image_with_extractor` methods. `ProvidedFeatureExtractor` wraps precomputed features, and `FnFeatureExtractor` wraps a closure so OpenCV, learned features, or custom image pipelines can be connected without making those dependencies mandatory.

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
- `TrackingResult`: localization result plus state transition, pose prior, map provider stats, map landmark count convenience field, prior-use diagnostics, and optional tracking failure reason
- `TrackingConfig`: controls `min_successive_failures_to_lost`, optional `last_pose_candidate_radius`, and tracking quality gates for pose-prior translation error, minimum inliers, minimum inlier ratio, and maximum mean reprojection error
- `TrackingFailureReason`: records tracking-layer quality gate failures such as insufficient inliers, low inlier ratio, high reprojection error, or pose jumps that exceed the motion-prior translation threshold
- `TrackingStats`: sequence diagnostics including first/last frame id, success/failure counts, lost/relocalization counts, pose-prior usage, tracking quality-gate failures, and aggregate inlier/correspondence totals with rate helpers
- `MotionModel`: predicts an optional pose prior for the next frame
- `ConstantPoseMotionModel`: default motion model that reuses the last successful pose
- `ConstantVelocityMotionModel`: extrapolates the next camera center from the two latest successful poses
- `Tracker`: feeds frames through a localization pipeline and updates state from success/failure
- `ImageTracker`: extracts features from image inputs and feeds generated frames into `Tracker`

`Tracker` also exposes `last_result`, `last_successful_frame_id`, `last_successful_pose`, and next-frame `LocalizationPrior` helpers for caller-side diagnostics or lightweight temporal consumers. `track_frames`, `track_frames_with_provider`, and `track_frames_with_descriptor_store` process a slice of frames while preserving tracker state across the sequence. `reset` clears sequence state, history, stats, and motion-model history so a caller can start a new sequence explicitly. `ImageTracker` mirrors those sequence and reset APIs for image inputs by extracting features per image before updating tracker state. `track_frame_with_prior_submap_provider` and the matching `ImageTracker` method can use the motion prior to create a temporary radius submap before localization. When `last_pose_candidate_radius` is set, successful localization stores the last pose and later frames use its camera center as a temporary radius prior for landmark selection. Tracking quality gates can reject a pose estimate when inlier count, inlier ratio, mean reprojection error, or pose-prior translation error is outside the configured thresholds while retaining the localization diagnostics.

This is not SLAM: it does not create keyframes, update maps, run bundle adjustment, or estimate map structure.

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
