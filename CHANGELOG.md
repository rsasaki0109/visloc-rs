# Changelog

All notable changes to `visloc-rs` will be documented here.

## Unreleased

### Fixed

- **`gsplat_photos` panicked on large photo sets** (wgpu: descriptor bank
  over `max_storage_buffer_binding_size`, e.g. 1,940 photos x 4,000 SIFT);
  it now matches in blocks. The trainer's logged L1 no longer rounds to 0
  on large images (2.8 MP), gradients were unaffected.
- **README hero splat filter read the wrong PLY columns.** Its
  large-and-transparent rule used SH coefficients as opacity and scale; the
  columns are now looked up by name.
- **`config.optical_flow_imu_seed_rotation` can be set from a Basalt config
  JSON.** The adapter read the key, but `BasaltConfig::from_json` rejected it
  as unknown, so only code could enable it.
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

- **README hero on a city block: Hierarchical 3DGS SmallCity.** 5,822
  photos, 5,609 registered with 960,849 points, held-out PSNR 21.79, 5.0M
  gaussians and a 5.7M-triangle mesh from one `gsplat_photos` command on a
  Google Colab A100 (earlier in this cycle: Tanks and Temples Courthouse,
  553 frames). `scripts/colab/hero_pipeline.py` runs the whole job on a
  Colab GPU (Tanks and Temples, Mill-19 and Hierarchical 3DGS scenes; from
  the terminal with the Colab CLI or from
  `scripts/colab/readme_hero_courthouse.ipynb`), adding the NVIDIA Vulkan
  user-space driver when the VM lacks it. `scripts/make_readme_hero.py` is
  scene-agnostic (intrinsics from `cameras.txt`, margins relative to the
  camera ring, nadir-aware up axis, orbit / filter flags, Linux + EGL).
- **`gsplat_photos` for large photo sets.** `--retrieval K` (VLAD top-K pairs
  past `--exhaustive-max`), `--exhaustive-max`, `--window`,
  `--max-keypoints`, `--sift-opt key=value` (wide-baseline `affine` /
  `domain_size_pooling` run on the CPU) and `--refine-intrinsics-max`;
  GPU matching splits the descriptor bank into blocks past the ~2 GiB
  binding limit.
- **Faster mapper and BA on large models.** The COLMAP-port mapper filters
  points and completes / merges tracks in parallel (byte-identical
  output), builds BA problems with less overhead, and factors reduced
  camera systems of 64+ frames in a fill-reducing order (nested
  dissection / RCM). The joint pose + intrinsics BA gained a block-sparse,
  parallel solve (`LinearSolver::Sparse`, used by `gsplat_photos`).

- **Multi-camera Basalt VIO for divergent rigs (opt-in, aimed at LaMAria /
  Project Aria).** The VIO used only cam0 for new keypoints, landmark hosting
  and keyframe decisions, and seeded the cam0→cam1 stereo KLT at the same
  pixel, which on Aria's ~75°-divergent cameras found almost no matches. New
  optional keys: `optical_flow_matching_guess_type = "REPROJ_FIX_DEPTH"` with
  `optical_flow_matching_default_depth` (reprojection seed plus a predicted
  patch warp), `optical_flow_detect_all_cameras` (cam1 keypoint
  replenishment), `vio_landmarks_all_cameras` (cam1-hosted landmarks for
  tracks cam0 never sees) and `vio_kf_connectivity_all_cameras`. LaMAria
  variant `configs/basalt/variants/lamaria/euroc_config_big_window_multicam.json`.
  Synthetic Aria-like rig: RMS error 7.1 → 3.3 mm on a textured room, and
  23.5 → 5.1 mm when cam0 faces a blank wall for 3 s. Not yet measured on
  LaMAria; see `docs/lamaria_multicam.md`. Defaults are unchanged.
- **Joint GNSS + visual-odometry fusion** (`visloc_slam::gnss_fusion`,
  `visloc_slam::gnss_pose_graph`). VO relative-pose factors and GNSS
  position factors are optimised together (full batch or a causal sliding
  window) with an estimated GNSS-ENU-to-map alignment (yaw-only or full
  rotation plus translation; scale and per-frame scale drift for monocular
  VO), antenna lever arm, per-fix covariance, GNC outlier rejection with a
  χ² gate, and dropout bridging. Analytic Jacobians with finite-difference
  tests; RANSAC alignment bootstrap. GNSS fixes are matched to frame times by
  `visloc_fusion::interpolate_gnss_fix` / `gnss_fix_covariance` (plus
  `MeasurementBuffer::as_slice`). New `gnss_vo_fusion_demo` example and
  `scripts/check_gnss_fusion_demo_outputs.sh`. Synthetic ~1 km loop: ATE
  5.25 m (VO only) → 0.90 m fused (metric); 3.22 m (Sim(3)-aligned VO) →
  0.37 m (monocular). Loosely coupled: receiver ENU positions only, no raw
  pseudoranges and no IMU. See `docs/gnss_fusion.md`. `visloc-slam` now
  depends directly on `visloc-fusion` (it already did through `visloc-io`).
- **Monocular-inertial VIO on the Basalt port (experimental).**
  `basalt_euroc_vio_demo --mono` replays cam0 + IMU with the stereo
  calibration reduced to camera 0. The estimator needed no change: it already
  triangulates new keyframe landmarks across frames, initialises from IMU
  gravity alignment, and its KLT frontend treats cam1 as optional; the gap was
  the EuRoC reader and rig construction. New `EurocSensorDataset::open_monocular`,
  `BasaltCalibration::retain_cameras` and
  `visloc_basalt::vio_estimator_from_calibration`. The stereo-inertial path is
  unchanged. Synthetic tests (including rendered images through the real KLT
  frontend) recover metric scale within 0.3% at 5–9 mm RMS; EuRoC is still to
  be measured, and position drifts with IMU bias while the platform is static
  before the first motion. See `docs/mono_inertial_vio.md`.
- **Python bindings (`bindings/python`, package `visloc`).** pyo3/maturin
  extension with NumPy interop: `Camera` (COLMAP models + Double Sphere),
  `Pose`/`SE3`, COLMAP text/binary `Reconstruction` read/write, `localize` and
  `estimate_pose_pnp_ransac`, and `evaluate_ate` / `evaluate_rpe` /
  `umeyama_alignment`. Ships `.pyi` stubs and pytest tests; built and tested
  by a new `python-bindings` CI job. The crate is excluded from the root
  workspace (own `Cargo.lock`), so pyo3/numpy do not enter the Rust gates.
- **`visloc_io::colmap::write_colmap_binary_model`** for a `VisualMap`.
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

### Changed

- **Large source files split into submodules (no behavior change).**
  `pipelines/slam/src/incremental_sfm.rs` (18.5k lines), `bundle.rs` (17k),
  `examples/unordered_sfm_demo.rs` (22.5k, now
  `examples/unordered_sfm_demo/main.rs` plus modules) and
  `pipelines/basalt/src/vio/aom.rs` (21.5k) are now directories of files
  mostly under 3k lines. Pure code moves: public paths, test names and test
  counts are unchanged, and the Basalt parity arithmetic is untouched.
- **Planning docs archived.** The long `PLAN.md` log and the itemized 0.2.0
  change log moved to `docs/archive/`; `PLAN.md` is now a one-page handoff.

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
