# Changelog

All notable changes to `visloc-rs` will be documented here.

## Unreleased

### Fixed

- **Bundle adjustment uses the camera's full lens model.** Pose/structure BA
  (serial, parallel, dense-QR assembly and the GNC residual vector) now
  linearises the same lens that `Camera::project` measures, with analytic
  Jacobians for OPENCV (k1, k2, p1, p2), FULL_OPENCV, OPENCV_FISHEYE,
  SIMPLE_RADIAL_FISHEYE, RADIAL_FISHEYE, FOV and Double Sphere. These cameras
  were previously rejected with `UnsupportedCameraModel`; SIMPLE_RADIAL and
  RADIAL are now accepted too. Pipelines that used to skip BA silently on
  such a camera (they swallow the error) now run it. Unknown models stay
  rejected, and stereo observations with any camera other than PINHOLE /
  SIMPLE_PINHOLE now return `UnsupportedCameraModel` instead of being
  modelled as a pinhole. Distortion-free and radial-only cameras are
  bit-for-bit unchanged (pinned by tests). Not yet covered: rig observations
  still project with a pinhole, and the matrix-free / GPU backends still
  decline distorted cameras and fall back to the regular solver.

### Added

- **ROS 2 nodes (`ros2/visloc-ros2`)** over pure-Rust DDS (`ros2-client` /
  RustDDS; no ROS install needed to build). `visloc_vio_node` runs the Basalt
  stereo-inertial VIO on two `sensor_msgs/Image` topics plus
  `sensor_msgs/Imu` and publishes `nav_msgs/Odometry`,
  `geometry_msgs/PoseStamped`, `nav_msgs/Path` and `/tf` (VIO only; the
  online mapper is not attached). `visloc_localize_node` localizes
  `sensor_msgs/Image` frames against a COLMAP map plus landmark descriptors
  and publishes `PoseWithCovarianceStamped`, an inlier count and
  diagnostics. The crate is excluded from the root workspace, has its own
  lockfile and a dedicated CI job, and is tested end to end over RTPS
  (RustDDS to RustDDS), not yet against a ROS 2 install.
- **`Camera::project_with_point_jacobian`** (visloc-core): the pixel from
  `Camera::project` plus its analytic 2×3 Jacobian with respect to the
  camera-frame point, for every lens model.
- **`BaConfig::refine_tangential_distortion` / `--refine-tangential-distortion`**
  (unordered SfM demo): opt-in self-calibration of p1/p2 on top of
  `refine_distortion`. A PINHOLE camera is promoted to OPENCV
  `[fx, fy, cx, cy, k1, k2, p1, p2]` starting from p1 = p2 = 0.
  `refine_intrinsics` now also refines OPENCV cameras (their lens terms stay
  fixed unless `refine_distortion` is set). Default off.

## 0.2.1 - 2026-10-08

Patch release: camera-distortion correctness fixes. Each fix below corrected
results that were silently wrong, not just errors that were reported.

### Fixed

- **COLMAP camera export with self-calibrated distortion.** A `Pinhole`
  camera carrying the refined radial `[k1, k2]` tail
  (`--refine-distortion`) was written as `PINHOLE` with 6 params, which is
  not valid COLMAP (`PINHOLE` has 4); COLMAP rejects it and other readers
  silently drop the distortion. Every COLMAP writer (text and binary) now
  exports it as `OPENCV` with `p1 = p2 = 0` (the identical projection), and
  fallible writers reject any camera whose parameter count does not match its
  COLMAP model. New public helper `visloc_io::colmap::colmap_camera_record`.
- **OPENCV tangential distortion.** `Camera::project` / `normalize_pixel` /
  `unit_ray_from_pixel` ignored `p1, p2` of an `OpenCv` camera, so a COLMAP
  `OPENCV` model with non-zero tangential terms was silently treated as
  radial-only. Such cameras now use the full Brown-Conrady model (shared with
  `FULL_OPENCV`, k3..k6 = 0); results with `p1 = p2 = 0` are bit-identical to
  before. New `Camera::tangential_distortion`.
- **Pose/structure BA ignored self-calibrated radial distortion.** With
  `refine_intrinsics = false` (e.g. the SfM polish / warm-start passes after
  `--refine-distortion`), `optimize_weighted` built its residuals and
  Jacobians with an undistorted pinhole while the cost it accepts steps
  against used the distorted `Camera::project`, so LM optimised a different
  model than it measured and could stall. The monocular residual and Jacobian
  (serial, parallel, and dense-QR assembly, plus the GNC residual vector) now
  apply `1 + k1·r² + k2·r⁴` with an analytic Jacobian; distortion-free cameras
  keep the historical closed form bit-for-bit.

### Added

- **`BaConfig::shared_focal` / `--shared-focal`** (unordered and sequential
  SfM demos): constrain intrinsics self-calibration to `fx == fy`. Previously
  `fx` and `fy` were always refined independently, so weakly observable
  motion (near-pure forward translation) could settle on a spurious aspect
  ratio. Default off; the joint solve is unchanged when it is not set.

## 0.2.0 - 2026-09-15

### Highlights

- **Real-data SfM at scale.** All ten ETH3D low-resolution many-view scenes:
  **9,996/10,008 cameras registered (99.88%)**, with no mapper run above
  3.32 GiB. Same-input ETH3D Electro 1,200 runs **3.46× faster than COLMAP**
  with **25.2% lower** camera-centre RMSE, and the 38-image courtyard control
  reaches **0.5379 cm vs COLMAP's 1.6166 cm**.
- **Visual-Inertial SLAM (faithful Basalt Rust port).** A tightly-coupled
  stereo-inertial VIO estimator plus offline mapper that matches native
  Basalt's ATE within **0.1%** on all 11 EuRoC sequences and, with EuRoC's
  official calibration, beats measured ORB-SLAM3 on **8 of 11** sequences.
- **COLMAP-compatible unordered SfM** with VLAD retrieval, essential-matrix
  verification, incremental mapping, connected-component models, and COLMAP
  text export for downstream 3DGS / MVS.
- **Online stereo SLAM** with loop closure, SE(3)/Sim(3) pose-graph
  optimization, GNC outlier rejection, and bundle adjustment, benchmarked on
  KITTI, EuRoC, and TUM RGB-D.
- **Opt-in deep frontend** (SuperPoint / LightGlue via in-Rust ONNX Runtime,
  CUDA-accelerated), plus public-data map-reuse localization and GNSS-prior
  tracking demos.

### Full change log

The itemized 0.2.0 change log (every Added / Changed / Fixed entry from
2026-05-07 to 2026-09-15, about 4,800 lines) is archived in
[`docs/archive/changelog_0.2.0_full.md`](docs/archive/changelog_0.2.0_full.md).

## 0.1.0 - 2026-05-07

### Added

- Workspace split into core, vision, IO, localization pipeline, and tracking pipeline crates.
- Core visual localization types: `Frame`, `Keyframe`, `VisualMap`, `Landmark`, `Observation`, `Camera`, `Pose`, and `LocalizationResult`.
- `SO3` / `SE3` pose wrappers and reprojection utilities built on `nalgebra`.
- Brute-force descriptor matching with L2 distance, ratio test, optional cross-checking, and match diagnostics.
- Minimal DLT PnP estimator, PnP RANSAC, pose-estimation diagnostics, and optional Gauss-Newton pose refinement.
- COLMAP text and binary map parsers for `cameras`, `images`, and `points3D`.
- Text parsers for landmark descriptors and query features.
- Localization pipeline over query descriptors and visual-map landmarks.
- Map providers, submap selectors, priors, localization quality gates, and map validation reports.
- Tracking skeleton with motion models, state transitions, and sequence examples.
- Local mapping skeleton with keyframe policy, local map windows, landmark candidates, linear triangulation, staged map updates, and local refinement hooks.
- Online SLAM MVP composition that combines tracking and local mapping without loop closure or global optimization.
- COLMAP text model writer for saving reusable sparse maps.
- Sensor-fusion foundation crate with timestamped frames/poses, GNSS/pose/IMU measurements, covariance types, measurement buffers, frame prior sources, and external localization-prior tracking hooks.
- GNSS-prior tracking example showing radius-submap narrowing before localization.
- COLMAP compatibility notes covering supported sparse model inputs, descriptor handling, writer behavior, and current limitations.
- Root crate prelude and top-level re-exports for common application-facing localization APIs.
- Pre-1.0 to v1.0 migration guide covering recommended imports, localization boundaries, COLMAP descriptor handling, tracking priors, and experimental layers.
- Package metadata and crate-content checks in the local quality gate and CI.
- crates.io package metadata now includes project homepage and repository URLs.
- Workspace member crates now use crate-specific descriptions and docs.rs URLs.
- Publishing guide documenting workspace crate publish order and package-check workflow.
- Examples, integration tests, design docs, local check script, and GitHub Actions CI.

### Not Yet Implemented

- Full Visual SLAM.
- Full SfM.
- Loop closure.
- Dense mapping.
- Full bundle adjustment.
- Full tightly-coupled visual-inertial or GNSS/INS fusion.
