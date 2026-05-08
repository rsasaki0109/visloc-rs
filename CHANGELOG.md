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
- `TrackingStats::from_results` helper for rebuilding sequence diagnostics from stored tracking outputs.
- Trajectory-evaluation example showing frame-id matched translation errors, CSV output, and JSON summary output.
- File-based TUM trajectory evaluation example with optional CSV / JSON / HTML output directory.
- File-based KITTI trajectory evaluation example with optional CSV / JSON / HTML output directory.
- File-based sequence localization example that tracks query feature files and prints or writes CSV / KITTI / TUM trajectory exports plus `summary.json`, `tracking.csv`, `tracking_summary.json`, `trajectory_report.html`, and `tracking_report.html`.
- Tracking sequence example with optional `tracking.csv`, `tracking_summary.json`, `tracking_report.html`, and `trajectory_report.html` output directory.

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
