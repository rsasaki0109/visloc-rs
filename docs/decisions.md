# Decisions

## Start with Visual Localization

The first implementation uses an existing visual map and estimates query camera pose through 2D-3D matching and PnP RANSAC.

Full SLAM is deliberately out of scope because it would add keyframe policy, loop closure, local mapping, global optimization, and failure recovery before the map-based localization core has been validated.

Online Visual SLAM remains a planned direction. The project keeps the localization core, map types, frame/keyframe types, and tracking pipeline boundaries available so that online mapping can be added incrementally after the map-based localization path is reliable.

## Stateless Core

Geometry, matching, PnP, and RANSAC are implemented as reusable components. Pipeline crates compose these pieces but do not own global state.

## Trait Boundaries

Feature extraction, matching, and pose estimation are trait-oriented so future implementations can replace the initial brute-force matcher and DLT PnP solver without changing map types.

## Public Demo Direction

The near-term public demo should emphasize automotive / robotics sequence localization. That demo shape is easy to understand: a camera moves through a mapped environment, frames are localized against a reusable sparse map, and the estimated pose trajectory is visible.

UAV localization remains an important target, but it benefits more from GNSS, altitude, timestamps, and sensor-fusion hooks. Those should be added as optional priors after the visual localization and tracking demos are strong.

The core library should remain domain-neutral. Automotive and UAV demos should exercise the same map, feature, matching, PnP, tracking, mapping, and fusion interfaces instead of creating domain-specific forks.

## Visual-Inertial Initialisation Direction

VI initialisation lands in stages, not in one large drop. The first stage is the **stationary-window flavour** (`VisualInertialInitializer`): detect a quiet leading IMU segment, then read out `(R_w←b, b_g, b_a)` in closed form, leaving yaw at zero because gravity alone cannot observe it. The first stage ships as a standalone module so callers can validate it against ground truth before the `OnlineSlamPipeline` integration lands. Pipeline integration is a separate, smaller change tracked in [vi_initialization_integration.md](vi_initialization_integration.md).

The second stage — **motion-based initialisation** (ORB-SLAM3's "wait for translation, then run a motion-only optimisation") — recovers yaw and, on monocular pipelines, scale. It is intentionally scheduled after the stationary flavour because it depends on a hot visual frontend and on the keyframe / pose-graph plumbing already in place. Splitting the two stages keeps the first usable on every IMU-carrying dataset, even when no visual frontend is wired up yet, and lets the second be added without churning the public API around the first.

## Deep VO and Loop Closure Direction

Visual odometry should be able to use deep frontends, but the core crates should not depend on one model runtime or ship large weights. Learned keypoints, descriptors, and matchers should enter through the existing feature-extractor and matcher traits, or through a future VO frontend trait that returns frame-to-frame pose priors.

Loop closure is a roadmap goal, but the first milestone should be candidate detection and geometric verification rather than full pose-graph optimization. This keeps the demo honest: the system can show that it recognizes a previously visited place before claiming globally optimized SLAM.
