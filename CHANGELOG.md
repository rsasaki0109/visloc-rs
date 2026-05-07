# Changelog

All notable changes to `visloc-rs` will be documented here.

## Unreleased

### Added

- `PoseTrajectory` and `TrajectorySample` helpers for extracting successful tracking poses, camera centers, path length, mean reprojection error, CSV output, KITTI-style 3x4 pose rows, and TUM-style trajectory rows from sequence-localization results.
- KITTI- and TUM-style trajectory parsers and file readers for reading pose rows back into `PoseTrajectory`.
- `TrajectorySummary` helper and JSON summary export for sequence-localization demos and downstream visualization scripts.
- `TrajectoryErrorSummary` and per-frame translation-error helpers for comparing estimated trajectories against reference poses.
- Optional first-matched-frame translation alignment for trajectory-error reports.
- Self-contained HTML / SVG trajectory-evaluation reports for quick visual inspection.
- Self-contained HTML / SVG single-trajectory reports for sequence-localization demos.
- Self-contained HTML tracking reports for frame-by-frame state, failures, priors, and inlier diagnostics.
- CSV export for frame-by-frame tracking state, localization counts, failures, priors, and map stats.
- `TrackingStats` JSON export for aggregate tracking summaries.
- Tracking diagnostics now distinguish motion pose priors from external localization priors such as GNSS-derived submap narrowing.
- Tracking HTML reports now summarize motion-prior and external-localization-prior usage in the top-level metrics.
- `TrackingStats::from_results` helper for rebuilding sequence diagnostics from stored tracking outputs.
- Trajectory-evaluation example showing frame-id matched translation errors, CSV output, and JSON summary output.
- File-based TUM trajectory evaluation example with optional CSV / JSON / HTML output directory.
- File-based KITTI trajectory evaluation example with optional CSV / JSON / HTML output directory.
- File-based sequence localization example that tracks query feature files and prints or writes CSV / KITTI / TUM trajectory exports plus `summary.json`, `tracking.csv`, `tracking_summary.json`, `trajectory_report.html`, and `tracking_report.html`.
- Tracking sequence example with optional `tracking.csv`, `tracking_summary.json`, `tracking_report.html`, and `trajectory_report.html` output directory.
- Moving-camera GNSS-prior tracking example with optional tracking diagnostics plus `trajectory.csv`, `poses.txt`, `trajectory_tum.txt`, `trajectory_summary.json`, and `trajectory_report.html` output directory.
- GNSS-prior tracking demo output now includes an `index.html` dashboard linking the tracking report, trajectory report, CSVs, KITTI/TUM poses, and JSON summaries.
- GNSS-prior tracking demo now exports a synthetic reference trajectory plus translation-error CSV, JSON summary, and trajectory-comparison HTML report.
- GNSS-prior tracking demo output now includes `manifest.json` with generated file names and top-level tracking / trajectory / error metrics.
- Local quality checks now include a GNSS demo output smoke test for the dashboard, manifest, trajectory exports, and error reports.
- CI now runs the GNSS demo output smoke test in addition to the regular example suite.
- CI now uploads the checked GNSS demo dashboard and export directory as a `gnss-demo-outputs` artifact.
- Documentation now includes a GNSS-prior tracking demo guide with dashboard, report, export, and expected-metric notes.
- GitHub issue templates, a pull request template, contribution guide, and security policy now document the project scope and local quality gate.
- Dependabot now checks Rust crate and GitHub Actions dependencies weekly.
- CI now verifies the declared Rust 1.82 MSRV with `cargo check --workspace --all-targets`.
- Trajectory evaluation now has reusable pass/fail threshold types, evaluator CLI threshold flags, `evaluation_result.json` export, and a local trajectory-evaluation smoke check.
- Tracking statistics now have reusable pass/fail threshold types, and the GNSS-prior demo exports `tracking_evaluation.json` for tracking smoke checks.
- `GrayscaleImage` and `CornerFeatureExtractor` provide a dependency-free image feature extraction smoke path, with a new `localize_with_corner_extractor` example.
- PGM grayscale image IO now supports dependency-free image fixtures and the `localize_from_pgm` example.
- Optional `image-io` feature support for PNG/JPEG grayscale loading and the `localize_from_common_image` example.
- Optional common-image sequence loading and the `track_image_sequence_from_common_images` example.
- Common-image sequence summaries and dimension validation for image-sequence tracking inputs.
- Optional nanosecond timestamps and timestamp validation for common-image sequence inputs.
- Timestamped common-image sequence tracking example with GNSS-derived localization priors.
- Timestamp text parsing for image-sequence datasets with separate image folders and timestamp files.
- GNSS text/CSV parsing for timestamped world-position priors used by sequence localization demos.
- KITTI-style camera calibration parser for turning projection rows such as `P2` into `Camera::pinhole` inputs for automotive sequence demos.
- KITTI-style image sequence loader that combines image frames, optional timestamp files, calibration, and validation summaries.
- KITTI-style image sequence loader example that writes a small automotive-like image folder, timestamps, and calibration before reading them back.
- Local and CI smoke checks now verify KITTI-style image sequence demo outputs and upload them as a CI artifact.
- Documentation now includes a KITTI-style image sequence demo guide covering generated images, timestamps, calibration, output logs, and CI artifacts.
- Local and CI checks now verify local README/docs markdown links and anchors.
- Local and CI MSRV checks now cover all workspace targets and all features through `scripts/check_msrv.sh`.
- docs.rs metadata now builds every publishable crate with all features enabled so optional APIs are included in hosted documentation, and `scripts/package_check.sh` verifies the metadata is present.
- Local and CI checks now verify release metadata consistency across Cargo manifests, docs.rs settings, publish docs, and documented CI demo artifacts.
- README first-view copy and imagery now highlight the real public-data localization demo, robotics use case, current inputs/outputs, working demos, and explicit non-goals for readers evaluating the project quickly.
- README public-data demo assets now include a feature-rich variant with many detected image features and highlighted pose-link overlays for a clearer visual-localization first impression.
- Demo guidance now calls out feature-rich visualization and the future path for learned feature/matcher integrations without implying bundled deep models.
- Roadmap and demo strategy now make deep visual odometry and loop-closure candidate detection explicit next technical targets.
- `visloc-slam` now reports lightweight loop-closure candidates from shared verified landmarks between the current frame and older keyframes.
- `VisualOdometryFrontend`, `VisualOdometryEstimate`, and `NoopVisualOdometryFrontend` now provide the first tracking-level boundary for optional classical or learned two-frame VO integrations.
- `VisualOdometryPriorProvider` and `VisualOdometryPosePrior` convert two-frame VO estimates into current-frame pose priors.
- Two-view match text parsing supports external learned/classical matcher outputs for VO frontend experiments.
- `read_two_view_matches_dummy` demonstrates the external two-view match text bridge.
- `visual_odometry_prior_dummy` demonstrates the VO-prior adapter path without bundling a model runtime.
- `track_sequence_with_visual_odometry_prior` demonstrates using a VO-derived external prior to narrow localization candidates during tracking.
- `online_slam_loop_candidate_dummy` demonstrates loop-candidate reporting on a tiny synthetic sequence.
- Online SLAM results can now be exported to a self-contained HTML/SVG loop-candidate report.
- Large README animation GIFs are excluded from the root crates.io package while remaining available on GitHub.
- `FramePriorSyncSummary` diagnostics for checking external measurement coverage against frame timestamps.
- `FramePriorSyncEvaluationConfig` and pass/fail sync evaluation for CI-checkable external sensor coverage.
- JSON export for frame-prior sync evaluation results and the timestamped image GNSS-prior demo.
- Local and CI smoke checks now verify timestamped image GNSS-prior demo outputs and sync evaluation JSON.
- CI now uploads timestamped image GNSS-prior demo outputs as a separate artifact.
- Documentation now includes a timestamped image GNSS-prior demo guide covering generated images, timestamp/GNSS files, sync evaluation JSON, and CI artifacts.

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
