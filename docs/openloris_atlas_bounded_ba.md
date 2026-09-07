# Bounded observation-based atlas refinement

Status: implementation in progress; no BA quality or performance result yet.
The [recovery foundation](openloris_atlas_landmark_integration.md) is merged as
[PR #71](https://github.com/rsasaki0109/visloc-rs/pull/71), after all eight CI
checks passed. Its RMSE/p95 remain above the frozen COLMAP acceptance gates.

## First controlled policy

- Default-off, one forward sweep, 60-frame windows with stride 30.
- Joint rig-pose/landmark BA using existing `BundleAdjustment`, rig observation
  factors and the sparse solver; at most 20 iterations per window.
- Intrinsics and sensor extrinsics remain fixed. Referenced poses outside the
  active window are fixed anchors; an explicit pose anchor handles windows
  without external anchors.
- Select every landmark touching an active frame and include all its
  observations, including those outside the window. Never silently truncate
  observations to satisfy a resource cap.
- Caps: 60 free poses, 16,384 landmarks, 262,144 observations and 1,024
  referenced poses. Over-cap windows are explicitly reported, not hidden.
- Validate finite, non-increasing full selected-observation cost and unchanged
  positive-depth / min-two-observation / mean-2-px / maximum-4-px track gates.
  Preserve supported images and rig frames. Apply a window transactionally;
  failed validation leaves the model unchanged.
- Keep one candidate problem at a time. No all-image dense system or copies of
  the complete model per window. GT is post-map evaluation only.

## Measured input footprint

A read-only scan of `recovery-v1/component-000/points3D.txt` grouped landmarks
by every 60-frame / stride-30 window containing one of their observations.
Every selected track's full observation list and referenced frame set counted
toward the footprint, not just its in-window observations. Frame IDs came from
the frozen rig manifest. Across 150 windows:

| Quantity | Maximum | Window start |
|---|---:|---:|
| Selected landmarks | 13,213 | 360 |
| All selected-track observations | 185,173 | 180 |
| Referenced rig poses | 593 | 420 |

This demonstrates input-size feasibility for the proposed caps, not measured
BA peak RSS or convergence. Sparse factorization, total process RSS, exact
repeatability and unchanged default output still require execution checks.
If strict validation rejects every window, retain that evidence and diagnose
the failure; do not relax the frozen COLMAP acceptance gates.
