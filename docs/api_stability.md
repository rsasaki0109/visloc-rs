# API Stability

`visloc-rs` is pre-1.0, but the public API should already move toward stable boundaries. The goal is to keep the localization-first core useful while allowing tracking, local mapping, online SLAM, and fusion layers to evolve.

## Stability Tiers

### Core Stable Candidates

These APIs should change conservatively because many layers depend on them:

- `Camera`, `Pose`, `Frame`, `Keyframe`, `Landmark`, `Observation`, `VisualMap`
- `LocalizationResult` and localization diagnostics
- `SE3`, `SO3`, projection and reprojection helpers
- `LandmarkDescriptorStore`
- `visloc_rs::prelude` as the convenience import surface for common application code

Changes here should be additive when practical. Field removals, coordinate-frame semantic changes, and id-type changes should wait for a major release.

### Replaceable Algorithm Boundaries

These are intended extension points:

- `FeatureExtractor`
- `Matcher`
- `PoseEstimator`
- `PoseRefiner`
- `RobustPoseEstimator`
- `CandidateSelector`
- `MapProvider`
- `DescriptorProvider`
- `SubmapSelector`
- `MotionModel`
- `Triangulator`
- `LocalRefiner`

The trait names and responsibilities should remain stable, but new methods may still be added before 1.0 if an example or test demonstrates the need.

### Experimental Composition Layers

These APIs are useful now but still expected to evolve:

- `LocalMappingPipeline`
- `OnlineSlamPipeline`
- `FramePriorSource`
- `MeasurementBuffer`
- `GnssMeasurement`, `PosePriorMeasurement`, and `ImuMeasurement`
- Covariance and timestamp helper types

Changes should still be additive where possible, but these layers may need refinement as public automotive/UAV sequence examples become more realistic.

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
- Keep visual-only workflows free from fusion dependencies in their call sites.
- Keep full SLAM-specific state out of `visloc-core` unless localization also needs it.
- Keep examples deterministic unless they intentionally demonstrate public dataset IO.
- Document new public types in `docs/interfaces.md`.
- Add each public behavior to `CHANGELOG.md` under `Unreleased`.

## v1.0 Exit Bar

Before a 1.0 release:

- `README.md`, `docs/interfaces.md`, and this document should match exported public APIs.
- Public examples should cover map-based localization, sequence tracking, COLMAP IO, and external-prior tracking.
- `scripts/check.sh` should pass on a clean checkout.
- Any known breaking API changes should be collected and resolved before tagging.
