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

This is easy for non-specialists to understand because the camera motion and map reuse are visible. It also exercises the same core pieces needed later for online SLAM.

Good dataset candidates:

- KITTI odometry-style sequences.
- Oxford RobotCar-style street sequences.
- nuScenes / autonomous-driving camera sequences.
- Small self-contained public visual localization benchmarks with permissive redistribution rules.

The first demo does not need full SLAM. It should show map-based sequence localization, tracking state, and pose continuity.

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
