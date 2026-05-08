# Roadmap

`visloc-rs` is intended to grow from a practical Visual Localization core into a foundation for robotics Visual SLAM, SfM map reuse, and sensor-fusion based localization.

The main showcase direction is automotive / robotics sequence localization, with UAV localization as a closely related target once GNSS/altitude priors are available. The project should keep one rule throughout the roadmap: make each stage useful on its own, then use it as the base for the next stage. Online SLAM is planned, but it should not be added before the map-based localization and tracking layers are reliable.

## Direction

The long-term shape is:

1. Load or reuse an existing visual map.
2. Localize a query image or robotics image sequence against that map.
3. Track poses across frames with priors and relocalization.
4. Add local mapping and lightweight keyframe policies.
5. Grow into online Visual SLAM with incremental map updates.
6. Add a deep visual odometry frontend as an optional tracking path.
7. Add loop-closure candidate detection and later pose-graph optimization hooks.
8. Add visual-inertial and GNSS priors/fusion.

## Near-Term Technical Bets

Two goals should drive the next public demos:

- **Deep Visual Odometry frontend.** The project should support learned feature
  extraction and learned matching as replaceable frontends for frame-to-frame
  motion estimation. This should stay optional: model runtimes, weights, and
  accelerator-specific code should live behind traits or integration crates, not
  inside `visloc-core`. The first `VisualOdometryFrontend` and
  `VisualOdometryPriorProvider` boundaries exist in `visloc-tracking`; the next
  step is replacing the fixed demo frontend with a real classical or learned
  implementation in sequence demos.
- **Loop-closure candidate detection.** The first loop-closure milestone should
  detect and report candidates, then geometrically verify them. Lightweight
  shared-landmark candidate reporting and HTML/SVG candidate-edge reporting now
  exist in `visloc-slam`; full global pose graph optimization can come later.
  The demo value is showing that the system recognizes a previously visited
  place and exposes the candidate clearly.

## v0.1: Map-Based Visual Localization

Goal: make the vertical slice solid.

Focus:

- COLMAP text and binary model loading
- `VisualMap`, `Landmark`, `Observation`, `Frame`, `Keyframe`, `Camera`, and `Pose` types
- 2D-3D correspondence construction
- Descriptor matching with diagnostics
- PnP + RANSAC pose estimation
- Optional pose refinement
- Localization result quality and failure diagnostics
- Public-data demo and examples, with automotive sequence demos prioritized

Out of scope:

- Online map updates
- Full bundle adjustment
- Loop closure
- Dense mapping

Exit criteria:

- A COLMAP/SfM map can be loaded and validated.
- A query frame or short image sequence can be localized against the map.
- Failures are explainable through diagnostics.
- Public examples are reproducible without hidden assets.

## v0.2: Tracking Layer

Goal: turn independent localizations into a usable sequence-localization pipeline.

Focus:

- Frame-to-frame tracking state
- Motion priors
- Pose-prior based landmark selection
- Lost/relocalized states
- Tracking quality gates
- Sequence-level examples and tests
- Better public sequence demos and benchmarks

Out of scope:

- Creating new landmarks online
- Persistent map mutation
- Loop closure

Exit criteria:

- A sequence can be processed with explicit tracking state.
- Tracking can recover through relocalization against the existing map.
- Pose priors reduce matching/search work without hiding failures.
- Automotive-style sequence demos can show pose continuity and failure recovery clearly.

## v0.3: Local Mapping Skeleton

Goal: introduce map mutation carefully without committing to full SLAM complexity.

Focus:

- Lightweight keyframe selection
- Local map windows
- New landmark candidate representation
- Observation insertion and validation
- Map update transactions or staged map edits
- Interfaces for triangulation and local refinement

Out of scope:

- Global optimization
- Loop closure
- Large-scale map management
- Production-grade bundle adjustment

Exit criteria:

- The library can represent and validate staged local map updates.
- Keyframes and observations can be added without breaking the existing localization API.
- Mapping pieces remain separable from the stateless geometry and matching core.

## v0.4: Online Visual SLAM MVP

Goal: combine tracking and local mapping into a minimal online SLAM pipeline.

Focus:

- Tracking + local mapping orchestration
- Incremental map updates
- Local consistency checks
- Relocalization into the growing map
- Map save/load after online updates
- Simple local optimization hooks

Out of scope:

- Full loop closure
- Global pose graph optimization
- Dense or neural mapping

Exit criteria:

- A short image sequence can start from an initial map or bootstrap state and update a sparse map online.
- Lost tracking can relocalize.
- Updated maps can be saved and reused for localization.

## v0.5: Deep Visual Odometry Frontend

Goal: make frame-to-frame motion estimation stronger while keeping the Rust core replaceable and inspectable.

Focus:

- Visual-odometry frontend traits for two-view motion and frame-to-frame pose priors
- Learned feature/matcher integration points for SuperPoint/LightGlue-style pipelines
- Descriptor/keypoint adapters that can consume external model outputs
- Sequence demos that compare classical/localization-only tracking with deep-frontend priors
- Failure diagnostics for low-texture, motion blur, and weak-match cases

Out of scope:

- Bundling large neural-network weights into the core crate
- Requiring a specific inference runtime
- Dense neural reconstruction

Exit criteria:

- A deep or externally supplied VO frontend can provide pose priors without changing `VisualMap`, `Frame`, or localization APIs.
- The same tracking pipeline can run with either classical features or learned frontend outputs.
- Public demos make feature density, correspondence quality, and pose continuity visible.

## v0.6: Loop Closure Candidate Layer

Goal: make the SLAM direction visible by detecting likely returns to previously seen places.

Focus:

- Loop-closure candidate types and diagnostics
- Keyframe/image similarity interfaces
- Geometric verification through reusable matching and pose-estimation components
- Demo visualization for candidate links in a sequence trajectory
- Hooks for future pose-graph constraints

Out of scope:

- Full pose-graph optimization
- Large-scale image retrieval infrastructure
- Production-grade place-recognition databases

Exit criteria:

- A sequence can report likely loop candidates with scores and verification status.
- Demos can show a loop candidate edge without claiming global optimization.
- The API leaves room for later pose-graph optimization and map correction.

## v0.7: Sensor Fusion Foundation

Goal: prepare Visual Localization and Visual SLAM for robotics use cases.

Focus:

- Timestamped frames and poses
- IMU/GNSS measurement traits
- Pose prior interfaces
- Loose-coupling fusion hooks
- Covariance/uncertainty representation
- Time synchronization utilities
- Automotive odometry/GNSS and UAV GNSS/altitude prior compatibility

Out of scope:

- Full tightly-coupled visual-inertial optimization
- Production GNSS/INS stack

Exit criteria:

- External IMU/GNSS priors can guide localization and tracking.
- Fusion interfaces do not force a specific backend.
- Visual-only users can ignore fusion types without API friction.

## v1.0: Stable API

Goal: freeze the public API around proven abstractions.

Focus:

- Stable core type semantics
- Stable map-provider and feature/matcher/pose-estimator traits
- Clear crate boundaries
- API stability tiers for core types, replaceable algorithm traits, and experimental composition layers
- Public benchmark scripts
- Deep VO and loop-closure extension points documented as optional layers
- Compatibility notes for COLMAP/SfM map formats
- Migration guides for earlier versions

Exit criteria:

- The public API is documented and intentionally stable.
- Common Visual Localization and sequence-localization workflows are covered by examples.
- SLAM and fusion extension points are documented, even if advanced backends remain optional.

## Non-Goals

These are not near-term goals:

- Dense mapping
- Neural rendering
- Full SfM replacement
- Production-grade global bundle adjustment
- Large-scale place recognition

Those may become integrations later, but they should not distract from the localization-first core.

## Design Principles

- Keep geometry, matching, PnP, RANSAC, and map validation reusable.
- Keep pipeline crates as composition layers, not global state containers.
- Prefer trait boundaries for feature extraction, matching, pose estimation, map providers, and future fusion backends.
- Add complexity only when a runnable example or test needs it.
- Preserve the path from Visual Localization to Online Visual SLAM without pretending the first release is already SLAM.
