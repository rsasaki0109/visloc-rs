# Bounded observation-based atlas refinement

Status: first strict BA policy measured and rejected for trajectory regression.
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

## First measured result

The implementation passes 23 example tests. In the 10k control, 8 of 150 main
windows were accepted and none of 17 tail windows were accepted. Main-component
support files and all six tail output files remain byte-identical to recovery.
The main component took 120.50 s at 520,228 KiB peak RSS; the tail took 11.34 s.
These are integration/recovery/BA-only times, not mapper or native E2E results.

Aggregate mean reprojection improves to 0.631996 px, but RMSE/p95 worsen from
0.391465/0.643175 m to 0.391797/0.643537 m. The policy is **not promoted**.
Early rejection diagnostics show that lower total cost can still move an
individual track beyond its mean-2-px or maximum-4-px gate. Complete logs and
exact hashes are in
[first BA evidence](../benchmarks/electro/m8-openloris-atlas-joint-ba-v1.json).
The next decision must examine rejected observations and existing post-BA
filter/retriangulation behavior, not treat lower residual cost as trajectory
success or loosen the frozen COLMAP target.
