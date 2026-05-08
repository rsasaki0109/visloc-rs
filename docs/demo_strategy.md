# Demo Strategy

The public demo path should prioritize clarity over breadth.

## Primary Demo: Automotive / Robotics Sequence Localization

The best near-term showcase is automotive-style visual localization:

- A moving camera sequence.
- A reusable sparse visual map.
- Per-frame localization against the map.
- A visible pose trajectory.
- Match/inlier diagnostics.
- Clear failure and relocalization states.
- Enough visual feature points, inlier links, sparse map points, and trajectory motion to read as localization at a glance.

This is easy for non-specialists to understand because the camera motion and map reuse are visible. It also exercises the same core pieces needed later for online SLAM.

Learned feature pipelines such as SuperPoint-style keypoints or LightGlue-style matching should plug in through the existing feature-extractor and matcher traits. README assets should not imply deep weights are bundled unless the repository actually ships that integration.

Good dataset candidates:

- KITTI odometry-style sequences.
- Oxford RobotCar-style street sequences.
- nuScenes / autonomous-driving camera sequences.
- Small self-contained public visual localization benchmarks with permissive redistribution rules.

The current KITTI-style loader smoke path is documented in
[kitti_image_sequence_demo.md](kitti_image_sequence_demo.md). It covers the
image-directory, timestamp, and calibration-file plumbing needed before a larger
public automotive localization demo is added.

The first demo does not need full SLAM. It should show map-based sequence localization, tracking state, and pose continuity.

## SLAM-Feeling Demo Targets

To make the project feel closer to practical SLAM while staying honest about the current implementation, the next demos should target two visible behaviors:

- **Deep Visual Odometry frontend:** show denser and more stable frame-to-frame correspondences from an optional learned-feature pipeline, then feed the resulting motion as a tracking prior.
- **Loop-closure candidate:** show a sequence revisiting a place, draw a candidate loop edge, and report the candidate score and geometric verification status.

The demo should label these as targets until the corresponding Rust APIs and examples exist. It should not claim full loop closure or global pose-graph correction before those pieces are implemented.

## Secondary Demo: UAV Localization

UAV localization is still a strong target use case, especially for inspection, mapping reuse, and aerial robotics. It should follow after the automotive demo because the UAV story benefits from additional priors:

- GNSS position.
- Altitude.
- Timestamped frames.
- IMU attitude or gravity prior.
- Larger map search areas.

The UAV demo should show an aerial query sequence localized against an SfM / photogrammetry map and should report how GNSS/altitude priors narrow the map search.

## Rule

Automotive and UAV demos must use the same core library interfaces. Dataset-specific code belongs in scripts, examples, or docs, not in core geometry, matching, localization, tracking, mapping, or SLAM crates.
