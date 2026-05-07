# Experiments

Initial experiments should stay focused on map-based localization:

- Synthetic 2D-3D correspondences with known pose
- COLMAP text model loading smoke tests
- Descriptor matching quality with ratio-test thresholds
- PnP RANSAC sensitivity to reprojection threshold and outlier ratio
- IO-backed localization with `cargo run --example localize_colmap_text`
- Localization-based tracking state transitions with `cargo run --example track_sequence_dummy`
- Trajectory evaluation with `cargo run --example evaluate_trajectory_dummy`
- GNSS-prior submap narrowing with `cargo run --example track_sequence_with_gnss_prior`

Future experiments can add image feature extraction, online Visual SLAM, inertial priors, and public automotive or UAV sequence data after the visual localization slice is stable.
