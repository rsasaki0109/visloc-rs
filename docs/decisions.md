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
