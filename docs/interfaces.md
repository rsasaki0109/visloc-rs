# Interfaces

## Canonical Imports

Application code should usually start with:

```rust
use visloc_rs::prelude::*;
```

For narrower imports, prefer the crate modules re-exported by the root crate:
`visloc_rs::core`, `visloc_rs::localization`, `visloc_rs::tracking`,
`visloc_rs::mapping`, `visloc_rs::slam`, `visloc_rs::fusion`,
`visloc_rs::vision`, and `visloc_rs::io`. Root-level item re-exports such as
`visloc_rs::Camera` are a convenience facade, not the canonical ownership path
for every public type. See [api_stability.md](api_stability.md) for the
stable-intent allowlist and feature support tiers.

## Stability Scope

This document describes public interfaces that applications and examples can use
today. It does not mean every listed type has the same stability promise.
Stable-intent items and replaceable trait boundaries are enumerated in
[api_stability.md](api_stability.md) and compiled by `tests/api_stability.rs`.

The localization core, map/pose types, provider traits, feature/matcher/PnP
traits, and trajectory summaries should change conservatively. Online SLAM,
covisibility BA, visual-inertial integration, GNSS/prior-assisted tracking, and
large reconstruction/export paths are composition layers: they are documented so
callers can experiment against explicit contracts, but they may still evolve
before 1.0. The root facade and `prelude` are convenience surfaces; canonical
ownership remains with the crate modules listed above.

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

`visloc_vision::stereo_vo` provides the rectified-stereo VO frontend. `StereoFeatureConfig::depth_gate` defaults to `StereoDepthGate::Adaptive`, which derives the effective lower-depth gate from each frame's stereo-depth distribution, a disparity-uncertainty floor, bounded min/max limits, and optional hysteresis through `StereoDepthGateState`. `StereoDepthGate::Fixed` preserves the historical `min_depth_m` behavior for A/B replay; the public KITTI stereo demos expose that path with `--min-depth <m>`. `StereoDepthGateDiagnostics` records candidate/accepted counts, effective min/max depth, the depth quantile, and the disparity-uncertainty floor for registry artifacts and per-frame CSV inspection.

`visloc_io::images` provides dependency-free PGM grayscale IO through `read_pgm`, `parse_pgm`, `write_pgm_ascii`, and `to_pgm_ascii`. This is intended for fixtures, smoke tests, and minimal examples rather than broad image-format coverage.

With the `image-io` feature enabled, `visloc_io::images` also provides optional PNG/JPEG-backed grayscale loading through `read_common_image` and `decode_common_image`, plus `write_png_gray` for examples and fixtures. `read_common_image_sequence`, `read_common_image_sequence_with_timestamps`, `read_common_image_sequence_dir`, and `read_common_image_sequence_dir_with_timestamp_file` load ordered image frames for `ImageTracker` sequence experiments. `parse_timestamp_nanoseconds_txt` / `read_timestamp_nanoseconds_txt` read simple timestamp files where each non-comment line starts with a nanosecond timestamp. `common_image_sequence_summary`, `validate_common_image_sequence_dimensions`, and `validate_common_image_sequence_timestamps` report frame counts, image size, timestamp coverage/order, and mixed-dimension inputs before tracking. This keeps the default crate dependency-light while allowing real camera-image files to enter the same `GrayscaleImage -> FeatureExtractor -> ImageLocalizer/ImageTracker` path.

`CorrespondenceBuilder` converts `QueryImage + VisualMap + LandmarkDescriptorStore` into 2D-3D correspondences before pose estimation. The localization pipeline uses this builder internally. When the matcher emits `DescriptorMatch::confidence`, the builder stores that value on each `Correspondence2D3D` and `LocalizationPipeline` forwards the resulting weights into `RobustPoseEstimator::estimate_with_weights`; matchers without confidence keep the original unweighted path.

`Matcher` is the descriptor matching boundary. `BruteForceMatcher` provides L2 nearest-neighbor matching with optional ratio test, and `CrossCheckMatcher` wraps another matcher to keep only mutual nearest matches. `DescriptorMatch` records query/train indices, best distance, second-best distance, ratio diagnostics, and optional confidence when available. `CorrespondenceSet` preserves those descriptor matches alongside 2D-3D correspondences for debugging.

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
- `TrackingConfig`: controls `min_successive_failures_to_lost`, optional `last_pose_candidate_radius`, tracking quality gates for pose-prior translation error, minimum inliers, minimum inlier ratio, and maximum mean reprojection error, and an optional `projection_guided_tracking: Option<ProjectionGuidedTrackingConfig>` (default `None`, matching today's appearance-global behaviour bit-for-bit)
- `ProjectionGuidedTrackingConfig`: ORB-SLAM3-style projection-guided tracking — when a pose prior is available, restricts descriptor matching to a per-landmark projection window (`search_radius_px`) instead of an appearance-global search, widens the window and retries (`widen_factor`, `max_widen_retries`) on failure before falling back to today's appearance-global path, and optionally re-projects the covisibility local map with the newly estimated pose to harvest additional correspondences and re-optimize the pose (`local_map_refinement`, `refinement_search_radius_px`), accepting the refined pose only when its inlier count does not decrease
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
- `SimpleKeyframePolicy`: selects the first successful frame, optional relocalized frames, and later frames that pass the frame-id gap plus either the camera-translation threshold or the opt-in tracked-landmark-drop ratio
- `KeyframePolicyConfig`: controls minimum frame-id gap, minimum translation, relocalized-frame selection, and the optional tracked-landmark-drop keyframe trigger (`None` by default for legacy A/B)
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

## Online SLAM And Optimization Building Blocks

`visloc-slam` composes tracking and local mapping, then exposes opt-in
loop-closure verification, pose-graph refinement, local BA, and VI building
blocks. The default `OnlineSlamPipeline` keeps loop-candidate diagnostics,
covisibility local BA, pose-graph refinement, relocalization, and IMU stages
explicitly configurable so applications can adopt them one at a time. This is a
foundation-layer API, not a claim of production full SLAM behavior.

- `OnlineSlamPipeline`: owns a `VisualMap`, a `Tracker`, and a `LocalMappingPipeline`
- `OnlineSlamConfig`: controls whether valid staged updates are applied immediately, loop-closure candidate thresholds, optional covisibility local BA, optional pose-graph refinement, relocalization, and optional IMU / VI stages
- `LoopClosureConfig`: controls candidate detection using frame-id gap, shared-landmark count, shared-landmark ratio, and max returned candidates
- `LoopClosureCandidate`: reports the current frame, matched older keyframe, shared landmark count, overlap ratio, score, and geometric verification flag
- `OnlineSlamResult`: returns tracking output, optional mapping output, optional applied update counts, loop-closure candidates, optional VI / covisibility BA / pose-graph / relocalization stats, and current map sizes
- `online_slam_results_to_html_report`: creates a self-contained HTML/SVG report showing tracked camera centers and loop-candidate edges, including verifier inlier/Sampson columns when `LoopClosureCandidate::verification` is populated
- `OnlineSlamRelocalizationConfig`: configures the opt-in recovery-PnP path used when primary tracking fails. The default still tries every failed frame; `attempt_interval_frames` can throttle expensive global recovery attempts on long runs while preserving the original behaviour at `1`. `covisibility_local_map` can restrict the recovery descriptor store to landmarks observed by the last successful keyframe plus high-covisibility neighbours. `appearance_retrieval_map` can instead rank older keyframes by mean-descriptor cosine similarity to the failed frame and build a recovery store from the top retrieved keyframes; this is a lightweight place-recognition seed that can later be backed by learned global descriptors. `OnlineSlamRelocalizationAppearanceConfig::candidate_log_limit` can be higher than `max_keyframes` so runs can log recall@K candidates without expanding the recovery-PnP descriptor store. `OnlineSlamRelocalizationCovisibilityConfig` and `OnlineSlamRelocalizationAppearanceConfig` can retry the broader full-map / recent-window store on local-store failure, with `broader_store_retry_interval_frames` bounding the retry cost. `max_translation_per_frame_from_last_success_meters` is an opt-in pose-continuity gate. `min_inlier_depth_median_ratio_to_last_success` / `max_inlier_depth_median_ratio_to_last_success` are opt-in scale sanity gates comparing recovery-inlier median depth against the last successful pose. `confirmation_required_recoveries` plus `confirmation_max_translation_per_frame_meters` can require a short sequence of mutually consistent recovery hypotheses before tracker state is overwritten. `OnlineSlamRelocalizationStats` reports descriptor-store size, covisibility-store usage, appearance-retrieval usage, ranked appearance candidates, broader retry usage/skips, translation-per-frame, depth-ratio, and confirmation diagnostics even when the gates are disabled. The EuRoC image demo exports `relocalization_appearance_candidates.csv` and `frame_groundtruth.csv` so candidate-recall evaluation can keep failed recovery-query frames in the denominator instead of depending on success-only `slam_errors.csv`.
- `LoopClosureVerifier`: trait for swappable loop-closure verifiers operating on pixel-space `TwoViewCorrespondence`s plus camera intrinsics
- `EssentialMatrixLoopClosureVerifier`: classical-geometry verifier built on `visloc-vision::two_view`'s essential-matrix RANSAC, with `LoopClosureVerifierConfig` thresholds for `min_inliers`, `min_inlier_ratio`, `max_mean_sampson_error`, and a `default_translation_scale` applied when recovering the relative pose
- `LoopClosureVerification` and `LoopClosureVerificationFailureReason`: verifier outputs covering inlier count, inlier ratio, mean Sampson error, combined score, recovered relative pose (`Option<SE3>`), and an enumerated failure reason
- `LoopClosureConstraint`: pose-graph-style constraint built from a verified candidate (`from_keyframe_id`, `to_keyframe_id`, `relative_pose`, `inlier_count`, `inlier_ratio`, `mean_sampson_error`, `score`) and consumed by `PoseGraph::add_loop_closure_constraint`, pairwise factor conversion, and downstream BA / PGO tools
- `PoseGraph`, `PoseGraphEdge`, `PoseGraphEdgeKind`, `PoseGraphOptimizationStep`, `PoseGraphError`: sparse pose-graph keyed by keyframe id with sequential and loop-closure edges. `optimize_translations_once` is a single translation-only Gauss-Newton step that holds rotations fixed and is exact for translation-only residuals (kept as a fast linear baseline); `optimize_se3_iterative` is the full SE(3) iterative solver
- `PoseGraphSe3Config`, `PoseGraphSe3IterationStats`, `PoseGraphSe3Result`: configuration (max iterations, step / cost tolerances, robust kernel selection, optional Levenberg-Marquardt damping schedule), per-iteration cost / step / `λ` / accept-reject diagnostics, and run summary for `PoseGraph::optimize_se3_iterative`. The solver uses right-perturbation updates `T_i ← T_i · Exp(δ_i)` with a first-order BCH approximation (`J_r⁻¹ ≈ I`); per edge, residual is `r = log(meas⁻¹ · T_to · T_from⁻¹)` with Jacobians `Ad(T_from)` (to-node) and `-Ad(T_from)` (from-node). `PoseGraph::se3_cost` reports the matching squared-error cost; `PoseGraph::robust_se3_cost(&kernel)` reports the kernel-shaped cost when a `RobustKernel` is in use. The inner normal-equations solve prefers Cholesky and falls back to LU on ill-conditioned systems
- `RobustKernel`: per-edge IRLS cost selector (`None` / `Huber { delta }` / `Cauchy { c }`) used by `optimize_se3_iterative`. `cost(s)` returns `ρ(s)` and `weight(s)` returns the influence multiplier `ρ'(s)` applied to each edge's normal-equations contribution
- `select_covisibility_local_ba_window` and `refine_visual_map_with_covisibility_ba`: explicit A/B entry points for covisibility-selected local BA over an existing `VisualMap`. Selection starts from an active keyframe, ranks high-covisibility neighbor keyframes by shared landmark count, adds fixed boundary keyframes that observe the local landmarks, caps the local landmark set, and can require a minimum number of selected local-landmark observations on the active keyframe through `CovisibilityLocalBaConfig::min_active_observations`. `CovisibilityLocalBaConfig::boundary_support_min_optimized_keyframes` plus `boundary_support_min_fixed_keyframes` can reject large optimized windows before solving when too few fixed boundary keyframes would anchor them; the error reports the optimized/fixed counts so runners can log the rejected window shape. When strict boundary selection produces no eligible local landmarks, `CovisibilityLocalBaConfig::fallback_min_boundary_observations` can opt into one lower-threshold boundary retry; `CovisibilityLocalBaSelection::boundary_fallback_used` reports whether that retry supplied the final window. It then runs `BundleAdjustment` with `CovisibilityLocalBaConfig::ba_config`. `CovisibilityLocalBaSelection` reports optimized/fixed keyframes, local landmarks, observation count, and ranked candidate diagnostics. `CovisibilityLocalBaResult` reports the BA trace, mean reprojection before/after, map update counts, and optional post-BA outlier observation removal. `OnlineSlamConfig::covisibility_local_ba` exposes the same solver as an opt-in visual-only `OnlineSlamPipeline` stage after newly-applied keyframes and before pose-graph refinement; `OnlineSlamCovisibilityLocalBaConfig::max_outlier_observation_ratio` can run the solve on a cloned map and reject write-back when post-BA outlier observations exceed a bounded fraction of the selected observations. Two further opt-in write-back gates share that same clone-and-check path: `OnlineSlamCovisibilityLocalBaConfig::max_behind_camera_landmark_ratio` rejects write-back when too large a fraction of the solved optimized landmarks project behind an observing optimized camera (a direct degenerate-solve detector, exposed by the `behind_camera_optimized_landmark_ratio` helper), and `min_fixed_to_optimized_ratio` rejects write-back unless the fixed boundary keyframes anchoring the window meet `ceil(optimized_keyframe_count * ratio)` (the ratio form of the fixed-anchor requirement, via the `fixed_to_optimized_ratio_satisfied` / `required_fixed_keyframes` helpers). Both default to disabled. `OnlineSlamResult::covisibility_local_ba` reports per-trigger stats, failures, `elapsed_ms`, observation/outlier counts, outlier ratio, and which conditioning gate (quality/behind-camera/fixed-ratio) rejected the write-back. The default remains disabled so existing online pipelines keep their prior behavior.
- `HybridLoopClosureVerifier`, `HybridLoopClosureVerifierConfig`: consensus loop-closure verifier that runs both `EssentialMatrixLoopClosureVerifier` and `PnPLoopClosureVerifier` on the same candidate and accepts only when both verify AND their recovered relative poses agree within configurable rotation / translation-direction tolerances. The companion runner is `verify_loop_closure_candidates_hybrid`. When both backends accept but disagree, the failure reason is `LoopClosureVerificationFailureReason::PoseDisagreement`. The combined `LoopClosureVerification` carries the PnP relative pose (metric) and the conservative minimum of both backends' inlier counts / ratios; both `mean_sampson_error` and `mean_reprojection_error_px` are populated for downstream display
- `PnPLoopClosureVerifier`, `PnPLoopClosureVerifierConfig`: PnP-based loop-closure verifier built on `visloc-vision::ransac::PnPRansac`. Operates on 2D-3D correspondences (current frame pixel ↔ landmark world point) and re-localizes the current frame against landmarks observed by the candidate keyframe. Returns a metric relative SE(3) directly (no scale parameter needed) plus inlier count / ratio / mean reprojection error in pixels. The companion `correspondences_2d3d_for_loop_candidate` helper intersects the current frame's tracking inliers with the older keyframe's observed landmarks; `verify_loop_closure_candidates_pnp` runs the verifier over a slice of candidates and updates each `verification` and `geometrically_verified` field in place
- `SE3::log` / `SE3::exp` / `SE3::adjoint` plus public `so3_left_jacobian` and `so3_left_jacobian_inverse`: SE(3) Lie-group helpers in `visloc-core::geometry::se3`. The tangent layout is `[ρ; ω]` (translation first, then rotation); both helpers fall back to Taylor series for small angles
- `relative_world_to_camera`: helper that derives the `previous_to_current` SE3 measurement between two `Pose`s for use as a `PoseGraphEdge::measurement` value
- `correspondences_for_loop_candidate`: helper that builds two-view correspondences for a candidate from the current frame's tracking inliers and the older keyframe's observations
- `verify_loop_closure_candidates`: convenience helper that runs a `LoopClosureVerifier` over a slice of `LoopClosureCandidate`s in place, updating each `verification` and `geometrically_verified` field
- `loop_closure_constraints_from_candidates`: builds `LoopClosureConstraint`s from a slice of candidates, silently dropping unverified candidates and those without a recovered relative pose

Each frame is localized/tracked first. If tracking succeeds, the pipeline creates a keyframe from the tracked frame, runs local mapping with caller-supplied landmark candidates, and optionally applies the validated staged update to the growing map. If tracking fails, mapping is skipped and the caller still receives the tracking diagnostics.

Loop-closure candidates are diagnostic unless the caller explicitly consumes
them, or enables `OnlineSlamConfig::pose_graph_refinement`. With pose-graph
refinement enabled, verified constraints can update keyframe poses in the
pipeline's map; without it, candidates remain reportable evidence for an
external optimizer or offline A/B run.

The `online_slam_loop_candidate_dummy` example shows the diagnostic path on a tiny synthetic sequence: an older keyframe is inserted first, a later frame observes the same landmarks, and the pipeline reports a loop candidate with overlap score and verification status. With `--out-dir`, it writes `loop_report.html` so the candidate edge is visible in a browser. Pose-graph and loop-refinement examples exercise the opt-in optimization path separately from the default pipeline behavior.

## Visual-Inertial Estimation

`visloc-slam` exposes the IMU stack as separable layers so visual-only callers pay no per-frame cost while VIO-style callers can opt-in piece by piece. The conventions are common to every layer: angular velocity and linear acceleration are body-frame, gravity is NOT pre-subtracted from `accel`, world gravity is supplied per session (KITTI y-down `(0, 9.81, 0)`; EuRoC z-up `(0, 0, -9.81)`), and bias is held at a linearisation point with first-order Jacobians so bias updates do not require re-integration.

- `ImuPreintegrator` and `ImuPreintegratedDelta`: on-manifold pre-integration (Forster et al., T-RO 2017). `ImuPreintegrator::new` / `new_with_bias` build the accumulator at a chosen bias linearisation point; `integrate_sample(gyro, accel, dt)` folds one body-frame sample (positive `dt` required); `delta()` snapshots the running `ImuPreintegratedDelta` without resetting state; `reset()` returns to the identity delta while preserving the linearisation point.
- `ImuPreintegratedDelta`: gravity-free `(ΔR, Δv, Δp)` in keyframe-`i`'s body frame plus `Δt` and the bias-Jacobians `∂(Log ΔR)/∂b_g`, `∂Δv/∂b_{a,g}`, `∂Δp/∂b_{a,g}` for first-order bias correction. `corrected(bias_gyro, bias_acc)` evaluates the bias-corrected `(ΔR, Δv, Δp)` triplet at a new bias point without re-running integration.
- `ImuPreintegrationFactor`: BA-side residual connecting `keyframe_id_from → keyframe_id_to` with `delta`, world `gravity`, and `weight_{position, velocity, rotation}`. `residual_with_bias_correction` applies the stored bias-Jacobians at residual-evaluation time.
- `OnlineSlamConfig::imu: Option<OnlineSlamImuConfig>` enables per-pipeline IMU ingestion. With `Some`, `OnlineSlamPipeline::push_imu_measurement` accumulates inter-frame samples and `OnlineSlamResult::imu_factor` carries the staged factor connecting every adjacent keyframe pair. `take_pending_imu_factor` exposes the same factor explicitly for downstream pose-graph / BA consumers. `None` (default) keeps the pipeline appearance-only.
- `OnlineSlamImuConfig`: per-session `gravity_world`, bias linearisation `(bias_gyro, bias_acc)`, and factor weights `(weight_position, weight_velocity, weight_rotation)`. Mirrored into `OnlineSlamImuState` (preintegrator + last-keyframe id + pending factor) when the pipeline owns IMU state.
- `OnlineSlamConfig::local_vi_ba: Option<OnlineSlamLocalBaConfig>` enables a sliding-window VI-BA refinement that fires every `trigger_every` newly-emitted IMU factors. The window holds the trailing `window_size` keyframes; reprojection residuals from landmarks observed at least `min_observations_per_landmark` times inside the window are stacked with every in-window `ImuPreintegrationFactor` and solved through `BundleAdjustment` under the supplied `BaConfig`. Refined poses, landmarks, velocities, and biases write back into `map` and `OnlineSlamLocalBaState.keyframe_state` unless the cost-ratio, fixed refined-velocity, or adaptive refined-velocity writeback gate rejects the solve. Requires `imu = Some(_)`; without IMU factors the stage never fires.
- `OnlineSlamLocalBaConfig`, `AdaptiveVelocityGateConfig`, `OnlineSlamLocalBaState`, `KeyframeImuState`, `OnlineSlamLocalBaStats`, and the free function `run_local_vi_ba`: per-trigger configuration, running per-keyframe `(velocity, bias)` table, rolling factor history (capped at `4 × window_size`), and per-trigger BA outcome (window keyframe ids, observation / landmark / IMU-factor counts, optimiser `BaResult`, `cost_ratio`, `max_refined_velocity_norm_mps`, optional `adaptive_velocity_gate_threshold_mps`, `bias_frozen`, `quality_gate_rejected`, and gate-specific rejection booleans). The adaptive velocity gate derives its per-trigger threshold from in-window velocity state, pose-delta / IMU-`dt` finite differences, and IMU-predicted next-keyframe velocities; raw fixed `m/s` caps remain available as safety ceilings and A/B controls. The stage is exposed as a function so unit tests can drive it directly without constructing the full pipeline.
- `VisualInertialInitializer` and `VisualInertialInitializerConfig`: stationary-window VI bootstrap. `new(config)` constructs the buffer; `push_sample(gyro, accel, dt)` appends a body-frame IMU sample (non-positive `dt` silently dropped, matching `ImuPreintegrator::integrate_sample`); `samples_seen` / `buffered_duration_seconds` / `config` are read-only diagnostics; `reset` clears the buffer; `try_initialize` returns `Ok(VisualInertialInitializationResult)` or `Err(StationaryRejectionReason)`. The recovery is three closed-form readouts: `b_g = ω̄` (gyro mean over the stationary window), `R_w←b = rotation_between(ā_b, -g_world / ‖g_world‖)` (shortest rotation that lifts the mean specific-force into world-up; yaw is unobservable from gravity alone), and `b_a = ā_b - R_w←b^T · (-g_world)`. `VisualInertialInitializerConfig` defaults match the EuRoC stationary holding period (`gravity_world = (0, 0, -9.81)`, `min_stationary_window_seconds = 0.5`, `max_gyro_std = 0.05 rad/s`, `max_accel_std = 0.5 m/s²`, `max_accel_magnitude_error = 0.5 m/s²`, `min_samples = 50`); KITTI / TUM users override `gravity_world` and tighten or loosen the noise thresholds to match their dataset's leading window.
- `VisualInertialInitializationResult`: `gravity_world` (echoed from config), `initial_rotation_body_to_world`, `initial_velocity_world` (zero by construction — the stationary window means the body is at rest), `bias_gyro`, `bias_acc`, plus diagnostics `samples_consumed`, `duration_seconds`, `gyro_std`, `accel_std`, `mean_accel_magnitude`. Callers typically use it once per sequence to seed `OnlineSlamImuConfig.{bias_gyro, bias_acc}` and the first-keyframe pose before `OnlineSlamPipeline.local_vi_ba` takes over. **Accel-bias caveat**: `bias_acc` is essentially the *magnitude residual* between the mean specific force and `‖g_world‖`. Because `R_w←b` is constructed from `ā_b` itself, lateral components of the true accelerometer bias are absorbed into the recovered rotation (they appear as roll/pitch offsets); only the gravity-axis component survives into `bias_acc`. Full 3-axis accelerometer bias observability requires translation in the visual frontend and belongs to the sliding-window VI-BA stage downstream, not to the static bootstrap.
- `StationaryRejectionReason`: `InsufficientSamples { have, need }`, `InsufficientDuration { have, need }`, `GyroNoiseTooHigh { observed, limit }`, `AccelNoiseTooHigh { observed, limit }`, `AccelMagnitudeMismatch { observed, expected, tolerance }`. The failing predicate's measured value is always populated so callers can either widen the threshold, wait for a longer stationary window, or fall back to ground-truth seeding without re-running the buffer.
- `OnlineSlamConfig::vi_init: Option<OnlineSlamViInitConfig>` opts the pipeline into an auto-bootstrap stage that runs a `VisualInertialInitializer` over the running IMU stream and atomically promotes the recovered `(R_w←b, b_g, b_a)` into `OnlineSlamImuState`, the first keyframe's `Pose`, and `OnlineSlamLocalBaState` on the first frame where (a) `try_initialize` succeeds AND (b) a new keyframe was just registered. Requires `imu = Some(_)`; the constructor rejects `vi_init: Some, imu: None` and `imu.gravity_world != vi_init.initializer.gravity_world` via `OnlineSlamConfig::validate() -> Result<(), OnlineSlamConfigError>`. While the stage is active, `take_pending_imu_factor` / inline `OnlineSlamResult::imu_factor` are gated (factors built with the placeholder bias linearisation are dropped to keep the downstream VI-BA point consistent); the discarded count surfaces on the `Succeeded` event for auditing.
- `OnlineSlamViInitConfig`: inner `VisualInertialInitializerConfig`, body-to-camera `SE3` extrinsic, `seed_first_keyframe_rotation: bool`, `on_persistent_rejection: ViInitFallback`, twin caps `max_wait_duration_seconds: f64` (default `5.0`) and `max_buffered_samples: usize` (default `2000`; either set to `0` / `0.0` disables that cap). Defaults assume IMU and camera share orientation (`SE3::identity()`).
- `ViInitFallback`: `KeepExistingSeed` leaves `imu_state` intact (the caller's `OnlineSlamImuConfig` defaults stay in effect; the stale-factor gate is lifted so new factors flow); `DisableImuStage` clears `imu_state` / `local_vi_ba_state` and the corresponding `config` slots, dropping the IMU stage for the rest of the sequence.
- `OnlineSlamResult::vi_init: Option<ViInitializationEvent>` carries the state-transition event for the auto-bootstrap stage on the frame it fires: `Succeeded { result, first_keyframe_id, discarded_stale_factor_count }` (emitted at most once per sequence, atomically promotes the recovered seed; the keyframe Pose is rewritten as `R_c←w = (R_w←b · R_b←c)^T` with the camera centre preserved via `t_c←w_new = -R_c←w_new · C_w_old`), `StillBuffering { reason }` (non-fatal `try_initialize` rejection; buffer preserved), `GaveUp { last_reason, fallback }` (cap exceeded; fallback already applied). `None` on every other frame; durable state goes through `OnlineSlamPipeline::vi_initialization_status() -> ViInitializationStatus { Disabled, Buffering { samples_buffered, buffered_duration_seconds, last_rejection }, Initialised { result }, GaveUp { last_reason, fallback } }`.
- `OnlineSlamConfigError`: `ViInitRequiresImu` and `GravityMismatch { imu_gravity_world, vi_init_gravity_world }`. Surfaced by `OnlineSlamConfig::validate`; `OnlineSlamPipeline::new` panics with the same message on a mismatch (developer error, not runtime — every caller passes a literal config).

The end-to-end validation path for VI initialisation is the `examples/euroc_imu_dead_reckon_demo` example: `--run-vi-init` runs the initialiser against the leading IMU window and logs the recovered `(bias_gyro, bias_acc, rotation)` plus three residuals against ground truth (full-quaternion `rotation_residual_vs_gt_deg`, yaw-invariant `gravity_alignment_residual_deg`, and the L² norms of the gyro / accel bias residuals); `--seed-from-vi-init` swaps the propagator's GT seed for the VI-init bootstrap so downstream ATE numbers reflect the honest no-GT-cheating recovery; `--vi-init-window-seconds <s>` / `--vi-init-gyro-std-limit <r>` / `--vi-init-accel-std-limit <r>` tune the stationary window for datasets without a clean leading hold. Visual-inertial integration into the `OnlineSlamPipeline` pre-tracking stage is described in [vi_initialization_integration.md](vi_initialization_integration.md) and is now shipped — callers using `OnlineSlamConfig.vi_init` no longer need to buffer the leading IMU window themselves.

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
