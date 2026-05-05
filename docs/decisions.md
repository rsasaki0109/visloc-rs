# Decisions

## Start with Visual Localization

The first implementation uses an existing visual map and estimates query camera pose through 2D-3D matching and PnP RANSAC.

Full SLAM is deliberately out of scope because it would add keyframe policy, loop closure, local mapping, global optimization, and failure recovery before the map-based localization core has been validated.

## Stateless Core

Geometry, matching, PnP, and RANSAC are implemented as reusable components. Pipeline crates compose these pieces but do not own global state.

## Trait Boundaries

Feature extraction, matching, and pose estimation are trait-oriented so future implementations can replace the initial brute-force matcher and DLT PnP solver without changing map types.

