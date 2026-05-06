# Experiments

Initial experiments should stay focused on map-based localization:

- Synthetic 2D-3D correspondences with known pose
- COLMAP text model loading smoke tests
- Descriptor matching quality with ratio-test thresholds
- PnP RANSAC sensitivity to reprojection threshold and outlier ratio
- IO-backed localization with `cargo run --example localize_colmap_text`
- Localization-based tracking state transitions with `cargo run --example track_sequence_dummy`

Future experiments can add image feature extraction, tracking, inertial priors, and GNSS priors after the visual localization slice is stable.
