# visloc-rs Development Plan and Handoff

This document is the current handoff plan for continuing `visloc-rs` on another
machine or with another coding agent.

## Current Status

Repository:

- GitHub: `rsasaki0109/visloc-rs`
- Main branch is the active development branch.
- Latest functional milestone before this handoff: `2d6fa30 add two-view match vo prior adapter`
- Rust MSRV: 1.82
- Unsafe code is forbidden.
- Main math dependency: `nalgebra`

Current milestone completion:

```text
Deep VO / Loop Close completion: 55%
```

Report this value at the end of development updates until it changes. Increase
it only when a runnable example, test, or documented API materially advances the
Deep VO or loop-closure path.

## Project Goal

`visloc-rs` is a Rust foundation library for map-based visual localization and
future Visual SLAM.

The short-term goal is not full SLAM. The short-term goal is a solid vertical
slice:

1. Load or reuse an existing COLMAP/SfM visual map.
2. Accept query features, descriptors, or images.
3. Build 2D-3D correspondences.
4. Estimate camera pose through PnP + RANSAC.
5. Track image sequences with priors and diagnostics.
6. Grow toward Deep VO and loop-closure demos without forcing heavy runtimes
   into the core crates.

The design must keep Visual Localization as the core and leave room for:

- Visual SLAM
- Visual map based localization
- SfM / SLAM map reuse
- Optional Deep VO frontends
- Visual-inertial and GNSS fusion
- Loop-closure candidate detection and future pose-graph optimization

## Non-Goals Right Now

Do not claim or implement these as completed production features yet:

- Full Visual SLAM
- Full SfM
- Full loop closure with global pose-graph optimization
- Dense mapping
- Full bundle adjustment
- Tightly coupled VIO or GNSS/INS
- Bundled neural-network weights or mandatory model runtimes

The repository may expose hooks for these, but the public wording should stay
honest.

## Architecture Overview

Workspace layout:

```text
crates/core          Core geometry and map/query types
crates/vision        Features, matching, PnP, RANSAC
crates/io            COLMAP, image, calibration, sensor, and match-file IO
pipelines/localization
pipelines/tracking
pipelines/mapping
pipelines/slam
pipelines/fusion
examples
tests
docs
```

Important top-level re-exports live in:

- `src/lib.rs`
- `src/two_view_vo.rs`

Core rules:

- Keep geometry, PnP, matching, and RANSAC reusable and mostly stateless.
- Keep pipelines as composition layers.
- Prefer trait boundaries for feature extraction, matching, pose estimation,
  motion priors, map providers, and future fusion/VO backends.
- Do not introduce mandatory OpenCV, ONNX, PyTorch, TensorRT, or GPU runtime
  dependencies into core/default crates.
- Use optional integration crates or file-backed adapters for heavy learned
  pipelines.

## Implemented Capabilities

### Map-Based Localization

Implemented:

- `VisualMap`, `Landmark`, `Observation`, `Frame`, `Keyframe`, `Camera`,
  `Pose`, `LocalizationResult`
- `SE3`, `SO3`, and reprojection helpers
- COLMAP text and binary model loading
- Descriptor store support for landmark descriptors outside COLMAP maps
- Brute-force descriptor matching with L2 distance and ratio test
- Cross-check matcher wrapper
- 2D-3D correspondence builder
- DLT PnP
- PnP RANSAC
- Optional Gauss-Newton pose refinement
- Localization quality gates and diagnostics

Representative examples:

```bash
cargo run --example localize_dummy
cargo run --example localize_colmap_provider
cargo run --example localize_from_files
```

### Image and Public Data Demos

Implemented:

- Dependency-free PGM image IO
- Optional `image-io` feature for common image formats
- Common image sequence loading
- KITTI-style calibration and image-sequence loading
- README public-data demo from COLMAP South Building imagery

Representative examples:

```bash
cargo run --features image-io --example track_image_sequence_from_common_images
cargo run --features image-io --example load_kitti_image_sequence
```

### Tracking

Implemented:

- `Tracker`
- `ImageTracker`
- Tracking states: uninitialized, tracking, lost
- Tracking events: initialized, tracked, tracking failed, lost, relocalized
- Motion models:
  - `ConstantPoseMotionModel`
  - `ConstantVelocityMotionModel`
- Last-pose candidate radius narrowing
- External localization prior narrowing
- Tracking stats, CSV, JSON, and HTML reports
- Pose trajectory export in CSV, KITTI, and TUM-like formats
- Trajectory evaluation summaries and HTML reports

Representative examples:

```bash
cargo run --example track_sequence_dummy
cargo run --example track_sequence_with_gnss_prior
```

### Local Mapping Skeleton

Implemented:

- `KeyframePolicy`
- `SimpleKeyframePolicy`
- `LocalMapWindow`
- `StagedMapUpdate`
- Landmark candidate representation
- Candidate validation
- Linear triangulation
- Local refinement hook with `NoopLocalRefiner`

This is intentionally a skeleton. It is enough to stage map updates and keep the
future SLAM path open, but it is not production mapping.

### Online SLAM MVP

Implemented:

- `OnlineSlamPipeline`
- Tracking + local mapping orchestration
- Optional validated staged update application
- Map-size diagnostics
- Lightweight loop-closure candidate reporting
- Loop candidate HTML/SVG report

Important limitation:

- Loop closure is candidate detection and visualization only.
- There is no global pose graph optimization yet.

Representative example:

```bash
cargo run --example online_slam_loop_candidate_dummy
cargo run --example online_slam_loop_candidate_dummy -- --out-dir target/visloc_loop_demo
```

### Fusion Foundation

Implemented:

- Timestamp types
- Timed frames and poses
- GNSS measurement type
- Pose prior measurement type
- IMU measurement type
- Measurement buffers
- Frame-prior sync utilities
- Loose-coupling `LocalizationPrior` path for GNSS/odometry/VIO style inputs

Representative example:

```bash
cargo run --features image-io --example track_timestamped_image_sequence_with_gnss_prior
```

### Deep VO / External Matcher Path

Implemented:

- `VisualOdometryFrontend`
- `VisualOdometryEstimate`
- `VisualOdometryPriorProvider`
- `VisualOdometryPosePrior`
- `NoopVisualOdometryFrontend`
- `TwoViewFeatureMatch`
- `TwoViewMatchSet`
- `parse_two_view_matches_txt`
- `read_two_view_matches_txt`
- `TwoViewMatchVisualOdometryFrontend`
- `TwoViewMatchVisualOdometryConfig`

Current external two-view match format:

```text
# PREV_IDX CURR_IDX PREV_X PREV_Y CURR_X CURR_Y [SCORE]
0 3 120.0 140.0 124.5 141.0 0.99
1 9 260.0 180.0 263.0 183.5 0.94
```

The current `TwoViewMatchVisualOdometryFrontend` is deliberately minimal:

- It consumes externally supplied two-view correspondences.
- It estimates a robust median-centered 2D flow.
- It rejects outlier flows by pixel residual.
- It returns a lightweight translation-only `VisualOdometryEstimate`.
- It is a bridge for demos and integration tests, not a real metric Deep VO
  solver.

Representative examples:

```bash
cargo run --example read_two_view_matches_dummy
cargo run --example two_view_match_vo_prior_dummy
cargo run --example visual_odometry_prior_dummy
cargo run --example track_sequence_with_visual_odometry_prior
```

## Key Files to Read First

Read these files before making changes:

```text
README.md
docs/progress.md
docs/roadmap.md
docs/interfaces.md
docs/decisions.md
src/lib.rs
src/two_view_vo.rs
crates/io/src/two_view_matches.rs
pipelines/tracking/src/lib.rs
pipelines/slam/src/lib.rs
examples/two_view_match_vo_prior_dummy.rs
examples/track_sequence_with_visual_odometry_prior.rs
examples/online_slam_loop_candidate_dummy.rs
tests/two_view_vo.rs
tests/tracking.rs
```

## Quality Gates

Use these commands before committing:

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets --all-features
cargo test --test two_view_vo
sh scripts/check_docs_links.sh
scripts/check.sh
```

`scripts/check.sh` is the main local gate. It runs formatting, clippy/checks,
tests, examples, docs, package checks, and selected demo-output checks.

After pushing, watch GitHub Actions:

```bash
gh run list --repo rsasaki0109/visloc-rs --limit 5
gh run watch <RUN_ID> --repo rsasaki0109/visloc-rs --exit-status
```

Existing GitHub Actions may warn about Node.js 20 deprecation for
`actions/checkout@v4` or `actions/upload-artifact@v4`. That warning has not
blocked CI.

## Current Deep VO / Loop Close Score

Current score:

```text
Deep VO / Loop Close completion: 55%
```

Why 55%:

- Tracking and local mapping scaffolds exist.
- Online SLAM composition exists.
- Lightweight loop-candidate diagnostics and reports exist.
- VO frontend trait boundary exists.
- VO estimates can become tracking pose priors.
- External two-view match files can be read.
- External two-view matches can now produce a lightweight VO prior.
- File-backed two-view match files now drive a short tracking sequence
  end-to-end through `read_two_view_matches_txt`, `TwoViewMatchVisualOdometryFrontend`,
  `VisualOdometryPriorProvider`, and `track_frame_with_localization_prior_submap_provider`,
  exercised by the `track_sequence_with_two_view_match_vo_prior` example and a
  matching integration test.

Why not higher:

- No real learned frontend is running in Rust yet.
- No real classical two-view geometry backend is producing metric VO yet.
- No public sequence demo shows learned correspondences driving tracking.
- Loop closure is candidate reporting only.
- No pose-graph constraints or global correction exist.

## Next Milestone: 55% to 60%

Goal: make the Deep VO path less synthetic while still avoiding mandatory model
runtimes.

Recommended next task:

### 1. File-Backed Two-View VO Sequence Example (Done at 55%)

`examples/track_sequence_with_two_view_match_vo_prior.rs` now exists. It builds
a small visual map and three synthetic frames, generates per-pair two-view
match text files for frame pairs `100 -> 101` and `101 -> 102`, reads them back
with `read_two_view_matches_txt`, populates a
`TwoViewMatchVisualOdometryFrontend`, and feeds each VO prior through
`track_frame_with_localization_prior_submap_provider`. The example prints
per-frame diagnostics (frame id, whether a VO prior was used, match count,
inlier count, mean flow residual, candidate landmark count, localization
success, used external prior, and estimated camera center) and writes a text
report plus the generated input match files when `--out-dir` is supplied.
`tests/two_view_vo.rs` covers the file-backed read → frontend → prior chain
across consecutive frame pairs.

### 2. Make the VO Adapter Diagnostics More Explicit

Current `VisualOdometryEstimate.mean_reprojection_error` is reused as
`mean_flow_residual_px` in the example. That is acceptable for now, but a clearer
diagnostic path would help:

- Keep `VisualOdometryEstimate` stable if possible.
- Consider adding optional metadata only if there is a clear local pattern.
- Avoid over-designing a generic metadata map.
- Better short-term option: document that for two-view-match VO, the
  `mean_reprojection_error` field represents mean inlier flow residual in
  pixels.

### 3. Add Real Classical Two-View Geometry Next

After the file-backed sequence example, implement a classical two-view geometry
path before adding a neural runtime:

- Normalize keypoints with camera intrinsics.
- Estimate essential matrix or fundamental matrix with RANSAC.
- Recover relative rotation and translation direction.
- Keep scale optional or supplied by:
  - previous pose scale
  - GNSS/odometry prior
  - configured default translation scale
- Return `VisualOdometryEstimate`.

Suggested module location:

```text
crates/vision/src/two_view/
```

Possible public types:

```rust
TwoViewCorrespondence
EssentialMatrixEstimator
EssentialRansac
RelativePoseEstimator
```

Do not bundle OpenCV as a required dependency. If OpenCV is used, make it an
optional integration later.

This would justify moving toward 60%.

## Next Milestone: 60% to 70%

Goal: make the demo visibly feel like VO and loop closure.

Recommended tasks:

### 1. Public Sequence Demo with Visible Correspondences

Use a small public sequence, preferably automotive or robotics:

- KITTI odometry sequence subset, if licensing and file size are practical.
- A small self-contained public image sequence fixture.
- Or generated demo assets only if they remain honest and clearly labeled.

The demo should show:

- Previous/current frame pair.
- Dense or semi-dense correspondences.
- Inliers vs outliers.
- Estimated camera path.
- Map landmarks or sparse visual map.
- Whether tracking used localization only or VO prior.

Do not make fake visual claims. If the images are synthetic, label them as
synthetic. For public data, document the source.

### 2. Optional Learned Frontend Bridge

Do not add a heavy runtime to default crates. Prefer one of these:

- File-backed output from a Python SuperPoint/LightGlue pipeline.
- Optional `visloc-deep` integration crate later.
- CLI-generated match files consumed by current Rust IO.

The best next step is probably a `tools/` or `scripts/` helper that documents
how to generate two-view match text from an external Python pipeline, but avoid
making Python required for `cargo test`.

### 3. Improve README Demo

README currently has a public-data localization GIF. It should eventually show:

- Sequence frames.
- Feature/match tracks.
- VO prior arrows.
- Loop-candidate edge.
- Clear labels that this is localization + tracking + candidate reporting, not
  full globally optimized SLAM.

## Loop Closure Plan

Current state:

- Candidate detection is based on shared verified landmarks.
- HTML/SVG report can show candidate edges.
- Geometric verification is lightweight and diagnostic.

Next loop closure tasks:

### 1. Stronger Candidate Verification

Add a verification layer that can reuse existing pose-estimation and matching
components:

- Candidate pair input:
  - current frame id
  - older keyframe id
  - shared landmarks
  - optional 2D-3D correspondences
- Verification output:
  - verified boolean
  - inlier count
  - inlier ratio
  - mean reprojection error
  - score
  - failure reason

Keep this as candidate verification, not global correction.

### 2. Pose-Graph Constraint Hook

Add a type to represent future pose-graph constraints, but do not implement full
optimization yet.

Possible type:

```rust
pub struct LoopClosureConstraint {
    pub from_keyframe_id: FrameId,
    pub to_keyframe_id: FrameId,
    pub relative_pose: SE3,
    pub inlier_count: usize,
    pub score: f64,
}
```

This type should live in the SLAM pipeline or a future optimization module, not
in the core geometry layer unless it becomes broadly reusable.

### 3. Demo Report Update

Update loop report to show:

- candidate edge
- verification status
- inlier count
- score
- no global correction yet

This would justify raising Deep VO / Loop Close completion above 60% when paired
with a runnable example and test.

## API Stability Notes

Stable-ish:

- Core types: `Camera`, `Pose`, `SE3`, `Frame`, `VisualMap`, `Landmark`
- Localization entry points
- Map provider and descriptor provider traits
- Matching and pose-estimation traits

Experimental:

- Local mapping skeleton
- Online SLAM pipeline
- Loop-candidate diagnostics
- Deep VO / two-view VO adapters
- Fusion measurement helpers

Keep experimental APIs documented as such.

## Documentation Rules

When changing behavior, update the relevant docs:

- `README.md` for user-visible examples and badges.
- `docs/progress.md` for milestone completion.
- `docs/roadmap.md` for staged plan.
- `docs/interfaces.md` for public API shape.
- `docs/decisions.md` for design decisions.
- `CHANGELOG.md` for notable changes.

Run:

```bash
sh scripts/check_docs_links.sh
```

## Release and Publish Notes

The crate is currently versioned `0.1.0`.

Do not publish casually. Before publishing:

1. Read `docs/publishing.md`.
2. Read `docs/release_checklist.md`.
3. Run `scripts/check.sh`.
4. Confirm package contents.
5. Confirm README claims match actual features.
6. Confirm CI is green on GitHub.

## Suggested Immediate Next Prompt for Claude

If handing off to another agent, use this:

```text
You are continuing the Rust project visloc-rs.
Read PLAN.md, docs/progress.md, docs/roadmap.md, docs/interfaces.md, and src/two_view_vo.rs first.
Current Deep VO / Loop Close completion is 55%.
The file-backed two-view VO sequence example already exists as track_sequence_with_two_view_match_vo_prior with covering tests in tests/two_view_vo.rs. Implement the next milestone: a real classical two-view geometry path (essential or fundamental matrix RANSAC, relative-pose recovery, optional scale from priors) under crates/vision/src/two_view/, expose it through VisualOdometryFrontend, drive a short sequence demo that compares it with the existing flow-only adapter, add tests, update README/docs/CHANGELOG, run scripts/check.sh, commit, push, and watch CI.
Do not add mandatory deep-learning runtime dependencies and do not claim full SLAM or full loop closure.
End every status/final message with: Deep VO / Loop Close completion: <percent>.
```

## Final Handoff Checklist

- Pull latest `main`.
- Confirm `git status --short` is clean.
- Read this `PLAN.md`.
- Run `cargo check --workspace --all-targets --all-features`.
- Start with the classical two-view geometry path (essential/fundamental matrix
  RANSAC and relative-pose recovery) once the file-backed two-view VO sequence
  milestone is in.
- Keep completion at 55% until the next runnable example/test/docs milestone is
  complete.
