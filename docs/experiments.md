# Experiments

Initial experiments should stay focused on map-based localization:

- Synthetic 2D-3D correspondences with known pose
- COLMAP text model loading smoke tests
- Descriptor matching quality with ratio-test thresholds
- PnP RANSAC sensitivity to reprojection threshold and outlier ratio
- IO-backed localization with `cargo run --example localize_colmap_text`
- Dependency-free grayscale corner extraction with `cargo run --example localize_with_corner_extractor`
- PGM-backed grayscale image localization with `cargo run --example localize_from_pgm`
- Optional PNG/JPEG-backed grayscale image localization with `cargo run --features image-io --example localize_from_common_image`
- Optional PNG/JPEG-backed image sequence tracking with `cargo run --features image-io --example track_image_sequence_from_common_images`
- Timestamped PNG/JPEG-backed image sequence tracking with GNSS priors using `cargo run --features image-io --example track_timestamped_image_sequence_with_gnss_prior`
- `scripts/check_timestamped_gnss_image_demo_outputs.sh` verifies the timestamped image GNSS-prior demo images, timestamp file, GNSS log, and sync evaluation JSON.
- File-based sequence localization and tracking report export with `cargo run --example localize_sequence_from_files -- --out-dir target/visloc_sequence_demo`
- Localization-based tracking state transitions with `cargo run --example track_sequence_dummy`
- Tracking report export with `cargo run --example track_sequence_dummy -- --out-dir target/visloc_tracking_demo`
- Trajectory evaluation with `cargo run --example evaluate_trajectory_dummy`
- KITTI file-based trajectory evaluation with `cargo run --example evaluate_trajectory_from_kitti_files -- --out-dir target/visloc_eval_kitti`
- TUM file-based trajectory evaluation with `cargo run --example evaluate_trajectory_from_tum_files -- --out-dir target/visloc_eval`
- KITTI / TUM trajectory evaluators support pass/fail thresholds with `--max-mean`, `--max-rmse`, `--max-max`, `--min-matched`, and `--min-match-ratio`; `scripts/check_trajectory_evaluation.sh` runs both fixture checks.
- Browser-viewable reports are written as `trajectory_report.html` / `tracking_report.html`, with frame-level tracking diagnostics in `tracking.csv` and aggregate tracking metrics in `tracking_summary.json`
- Moving-camera GNSS-prior submap narrowing with `cargo run --example track_sequence_with_gnss_prior -- --out-dir target/visloc_gnss_tracking_demo`, including an `index.html` dashboard, `manifest.json`, tracking diagnostics, `tracking_evaluation.json`, KITTI/TUM poses, synthetic-reference translation errors, and trajectory CSV / JSON / HTML exports
- The GNSS-prior demo output guide is in [gnss_demo.md](gnss_demo.md).
- CI checks both the moving-camera GNSS dashboard demo and the timestamped image GNSS-prior demo; it uploads the checked output directories as `gnss-demo-outputs` and `timestamped-gnss-image-demo-outputs` artifacts.

Future experiments can add image feature extraction, online Visual SLAM, inertial priors, and public automotive or UAV sequence data after the visual localization slice is stable.
