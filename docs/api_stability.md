# API Stability

`visloc-rs` is pre-1.0, but the public API should already move toward stable boundaries. The goal is to keep the localization-first core useful while allowing tracking, local mapping, online SLAM, and fusion layers to evolve.

## Canonical Import Paths

The root crate exports a broad facade for examples and quick experiments, but it
is not the canonical stability boundary for every item. Public documentation and
new examples should use these paths:

| Use case | Canonical path |
| --- | --- |
| Common application code | `visloc_rs::prelude::*` |
| Core poses, maps, cameras, ids, and validation reports | `visloc_rs::core::{geometry, types}` |
| Map-based localization pipeline and provider traits | `visloc_rs::localization` |
| Sequence tracking APIs and trajectory evaluation | `visloc_rs::tracking` |
| Local mapping, online SLAM, BA, pose graph, and loop-closure building blocks | `visloc_rs::{mapping, slam}` |
| Loose prior/fusion inputs | `visloc_rs::fusion` |
| Feature extraction, matching, PnP, RANSAC, two-view, and stereo VO algorithms | `visloc_rs::vision` |
| COLMAP, image, KITTI, sensor, descriptor, and two-view-match IO | `visloc_rs::io` |

Root-level re-exports such as `visloc_rs::Camera` and `visloc_rs::Pose` remain a
convenience layer, especially for examples. Adding an item to the root facade
does not by itself make its exact module ownership stable; the canonical module
paths above are the review target for public API compatibility.

The canonical paths and stable-intent items in this document are exercised by
the compile-time integration test in `tests/api_stability.rs`. When adding,
moving, or retiring a public item listed below, update this document and the
test in the same change so documentation drift is caught by `cargo test
--test api_stability` and by the full `scripts/check.sh` gate.

## Stability Tiers

### Core Stable Candidates

These APIs should change conservatively because many layers depend on them:

| Surface | Canonical stable-intent path |
| --- | --- |
| Camera identity and models | `visloc_rs::core::types::{Camera, CameraId, CameraModel}` |
| Image/map entity ids and observations | `visloc_rs::core::types::{Frame, FrameId, Keyframe, Observation, QueryImage}` |
| Sparse visual map entities | `visloc_rs::core::types::{Landmark, LandmarkId, LandmarkDescriptorStore, VisualMap}` |
| Map validation diagnostics | `visloc_rs::core::types::{VisualMapValidationIssue, VisualMapValidationReport}` |
| Pose and Lie-group geometry | `visloc_rs::core::geometry::{Pose, SE3, SO3, Sim3, reproject}` |
| Localization result contract | `visloc_rs::core::types::{LocalizationResult, LocalizationSuccess, LocalizationFailureReason}` |
| Pose-estimation diagnostics | `visloc_rs::core::types::{PoseEstimatorDiagnostics, PoseEstimationFailureDiagnostics, PoseEstimationFailureReason}` |
| Tracking pipeline handles | `visloc_rs::tracking::{Tracker, ConstantPoseMotionModel}` |
| Tracking summaries and trajectory evaluation | `visloc_rs::tracking::{TrackingStats, TrackingEvaluationConfig, TrackingEvaluationResult, PoseTrajectory, TrajectoryErrorSummary}` |

`visloc_rs::prelude` is the convenience import surface for common application
code and should remain additive where practical. The prelude is broader than the
stable-intent allowlist; adding an experimental item to the prelude does not
promote that item to a stable candidate unless it is listed here.

Changes here should be additive when practical. Field removals, coordinate-frame semantic changes, and id-type changes should wait for a major release.

### Replaceable Algorithm Boundaries

These are intended extension points:

| Boundary | Canonical stable-intent path |
| --- | --- |
| Feature extraction | `visloc_rs::vision::features::FeatureExtractor` |
| Descriptor matching | `visloc_rs::vision::matching::Matcher` |
| PnP estimation | `visloc_rs::vision::pnp::PoseEstimator` |
| Pose refinement | `visloc_rs::vision::pnp::PoseRefiner` |
| Robust pose estimation | `visloc_rs::vision::ransac::RobustPoseEstimator` |
| Landmark candidate selection | `visloc_rs::localization::CandidateSelector` |
| Map access | `visloc_rs::localization::MapProvider` |
| Descriptor access | `visloc_rs::localization::DescriptorProvider` |
| Submap selection | `visloc_rs::localization::SubmapSelector` |
| Tracking motion prediction | `visloc_rs::tracking::MotionModel` |
| Local-map triangulation | `visloc_rs::mapping::Triangulator` |
| Local refinement | `visloc_rs::mapping::LocalRefiner` |

The trait names and responsibilities should remain stable, but new methods may still be added before 1.0 if an example or test demonstrates the need.

Public error enums and config structs attached to these boundaries should stay
structured. Prefer adding enum variants or optional fields over replacing typed
failures with strings. Public error types should implement `Display` and
`std::error::Error` where they can cross crate or application boundaries.

### Experimental Composition Layers

These APIs are useful now but still expected to evolve:

| Surface | Canonical documented path |
| --- | --- |
| Local mapping composition | `visloc_rs::mapping::LocalMappingPipeline` |
| Online SLAM composition | `visloc_rs::slam::{OnlineSlamConfig, OnlineSlamPipeline}` |
| Covisibility local BA | `visloc_rs::slam::{CovisibilityLocalBaConfig, OnlineSlamCovisibilityLocalBaConfig, OnlineSlamCovisibilityLocalBaStats, CovisibilityLocalBaSelection, select_covisibility_local_ba_window, refine_visual_map_with_covisibility_ba}` |
| Stereo VO configuration and depth gating | `visloc_rs::vision::stereo_vo::{StereoVoFrontendConfig, StereoFeatureConfig, StereoDepthGate, StereoAdaptiveDepthGateConfig, StereoDepthGateState, StereoDepthGateDiagnostics}` |
| External frame priors | `visloc_rs::fusion::{FramePriorSource, MeasurementBuffer}` |
| Sensor-prior measurements | `visloc_rs::fusion::{GnssMeasurement, PosePriorMeasurement, ImuMeasurement}` |
| Fusion timestamps and covariance helpers | `visloc_rs::fusion::{Timestamp, TimeDelta, Timed, TimedFrame, TimedPose, PoseCovariance, PoseCovarianceMatrix, PositionCovariance}` |

Changes should still be additive where possible, but these layers may need refinement as public automotive/UAV sequence examples become more realistic.

### Feature Support Tiers

Feature support is defined in [feature_matrix.md](feature_matrix.md).

- Tier 1: `--no-default-features`, default, and `image-io`. These are checked
  on Linux and Windows, and the `image-io` path is part of the Rust 1.82 MSRV
  check.
- Tier 2 opt-in: `onnx-inference`. This path may download ONNX Runtime binaries
  and tracks current stable Rust rather than the Rust 1.82 MSRV.
- Tier 2 hardware-gated: `onnx-cuda`. This path requires CUDA-capable runners
  and is not part of the default CI gate.

## Crate Boundaries

- `visloc-core`: data types, poses, cameras, maps, and geometry primitives.
- `visloc-vision`: feature abstractions, matching, PnP, RANSAC, and local pose refinement.
- `visloc-io`: map and descriptor IO, currently centered on COLMAP-compatible sparse maps.
- `visloc-localization`: stateless map-based localization pipeline.
- `visloc-tracking`: sequence state around repeated localization calls.
- `visloc-mapping`: local mapping skeleton and staged map updates.
- `visloc-slam`: minimal online orchestration over tracking and local mapping.
- `visloc-fusion`: loose-coupling sensor priors that can guide localization/tracking without forcing a GNSS/INS backend.

Pipeline crates should compose lower-level crates. They should not move algorithmic primitives out of `visloc-vision` or map semantics out of `visloc-core`.

## Compatibility Rules Toward v1.0

- Prefer adding methods or new structs over changing existing function signatures.
- Prefer canonical module paths in docs and examples; keep root-facade changes additive.
- Keep visual-only workflows free from fusion dependencies in their call sites.
- Keep full SLAM-specific state out of `visloc-core` unless localization also needs it.
- Keep examples deterministic unless they intentionally demonstrate public dataset IO.
- Keep benchmark claims generated from the registry rather than hand-edited tables.
- Keep optional native runtimes behind Cargo features and out of the default dependency surface.
- Document new public types in `docs/interfaces.md`.
- Keep `tests/api_stability.rs` aligned with the stable-intent allowlist and canonical import paths in this document.
- Add each public behavior to `CHANGELOG.md` under `Unreleased`.

## v1.0 Exit Bar

Before a 1.0 release:

- `README.md`, `docs/interfaces.md`, and this document should match exported public APIs.
- `docs/feature_matrix.md` should match Cargo features and CI jobs.
- `tests/api_stability.rs` should compile against the documented canonical paths.
- Public examples should cover map-based localization, sequence tracking, COLMAP IO, and external-prior tracking.
- `scripts/check.sh` should pass on a clean checkout.
- `scripts/check_feature_matrix.sh` should pass for Tier 1 features, and any release claim using ONNX/CUDA should record the opt-in feature check that was run.
- Any known breaking API changes should be collected and resolved before tagging.
