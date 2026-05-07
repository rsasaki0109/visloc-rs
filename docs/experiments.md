# Experiments

Initial experiments should stay focused on map-based localization:

- Synthetic 2D-3D correspondences with known pose
- COLMAP text model loading smoke tests
- Descriptor matching quality with ratio-test thresholds
- PnP RANSAC sensitivity to reprojection threshold and outlier ratio
- IO-backed localization with `cargo run --example localize_colmap_text`
- File-based sequence localization and tracking report export with `cargo run --example localize_sequence_from_files -- --out-dir target/visloc_sequence_demo`
- Localization-based tracking state transitions with `cargo run --example track_sequence_dummy`
- Tracking report export with `cargo run --example track_sequence_dummy -- --out-dir target/visloc_tracking_demo`
- Trajectory evaluation with `cargo run --example evaluate_trajectory_dummy`
- KITTI file-based trajectory evaluation with `cargo run --example evaluate_trajectory_from_kitti_files -- --out-dir target/visloc_eval_kitti`
- TUM file-based trajectory evaluation with `cargo run --example evaluate_trajectory_from_tum_files -- --out-dir target/visloc_eval`
- Browser-viewable reports are written as `trajectory_report.html` / `tracking_report.html`, with frame-level tracking diagnostics in `tracking.csv`
- GNSS-prior submap narrowing with `cargo run --example track_sequence_with_gnss_prior`

Future experiments can add image feature extraction, online Visual SLAM, inertial priors, and public automotive or UAV sequence data after the visual localization slice is stable.
