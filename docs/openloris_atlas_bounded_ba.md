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

## Connectivity changes the next diagnostic

An independent rig-frame/landmark graph audit finds two disconnected groups
inside the atlas main file: frames 0–1999 and 2000–4493. The tail is one group.
COLMAP's explicit frame assignments and tracks yield one 4,494-frame main
group and one 505-frame tail group. Thus all 4,999 frames having observations
and being published in two files is not enough to establish connectivity parity.
The first BA policy preserves this split. Before changing BA filtering, trace
which source tracks cross frame 1999/2000 and where integration loses them.
See [connectivity evidence](../benchmarks/electro/m8-openloris-atlas-connectivity-v1.json).

The source trace finds 31 cross-boundary point rows, all in node 25 (window
start 1950). All pass ownership merging (19 new / 12 extended, no collision
rejection), but no crossing track survives final atlas-pose triangulation.
An independent row-level replay attributes failures to reprojection or depth,
not absent input matches. The next bounded audit compares source/atlas pose
ownership and left/right-only triangulations, deduplicated by final union track
identity.

That follow-up confirms 31 distinct merged tracks. Left-only triangulation
passes for 14, right-only for 16, and both sides for 7, without changing any
geometry gate. The left anchors provide at least six distinct 3D-track
correspondences in 14 right-side frames (maximum 11 per frame). These are
resection candidates, not yet verified GeneralizedPnP recoveries.

Frame 1999 uses node 25 and frame 2000 uses node 8. However, the 0.261 m jump
at 1997/1998 is internal to node 25, and the 0.900 m jump at 2022/2023 is
internal to node 8. A single rigid shift of the entire later component cannot
explain all three discontinuities. Independent source-pose inspection confirms
the same 0.261 m and 0.900 m jumps already exist in their respective owners.
In contrast, source node 25 moves only 0.033 m across 1999/2000 and 0.027 m
across 2022/2023; the boundary jump is not intrinsic to every available source.
The next default-off diagnostic therefore
uses existing GeneralizedPnP on bounded per-frame candidates, independently
checks cross-track recovery and loss of existing support, and does not publish
new poses automatically. GT remains evaluation-only.

### Resection diagnostic acceptance contract

This is a candidate audit, not a replacement mapper or an accepted recovery.
Use the existing generalized-rig RANSAC with seed 7, 4,096 iterations and
4 px inlier threshold. Count distinct landmark tracks, not camera observations:
the same point seen by both sensors does not count twice toward the six-point
minimum. Construct anchors only from the left-side observations, excluding
every target-side pose from anchor triangulation.

For every candidate, report separately:

- Crossing tracks whose **entire original observation set** passes unchanged
  triangulation gates with the candidate pose.
- Tracks passing only on the left-plus-target subset. This is partial support,
  not evidence of full-track recovery.
- Existing retained tracks/observations that would violate geometry or lose
  support, including support loss in other frames sharing those tracks.

Keep fixed sensor extrinsics, explicit resource caps, deterministic ordering
and one candidate pose at a time. Do not update model poses or silently prune
observations. Synthetic tests must cover duplicate-sensor inlier counting,
left-only anchors, failed geometry and unchanged diagnostic input state.

### Measured resection result

The default-off diagnostic passes 27 example tests and Clippy. Of 142 candidate
frames, 10 yield at least six distinct inlier tracks. Across these independent
candidates, left-plus-target triangulation succeeds in 79 of 80 attempts, but
full original crossing tracks succeed in only 2 of 80. Both successes occur
for frame 2000. Its candidate also causes 5 of 66 existing touching tracks to
fail full-observation re-triangulation (all 66 fail if their XYZ remains fixed).
No pose is adopted. These are per-candidate counts, not a simultaneous model.

The next audit must check whether adding those 2 full tracks and excluding the
5 failed tracks actually joins the main graph or merely moves the cut to
2000/2001. It must count image/frame support losses, including other frames
sharing the failed tracks, without writing a modified model.

That hypothetical graph audit now shows frame 2000 joins the two supported
groups into one 4,493-frame group, even after removing the 5 failed tracks
(39 observations) and adding the 2 full crossing tracks (30 observations).
No previously supported image or frame loses all support. Frame 4493 remains
unsupported because this diagnostic precedes the separate recovery step.
Including that isolated frame, the all-frame graph has two components, not
one. Other candidates do not improve connectivity; candidates 2007, 2008 and
2011 split the supported graph into three groups.

This justifies a default-off transactional repair experiment, not immediate
quality promotion. Require strictly fewer supported graph components and
preservation of every supported image/frame; report all removed observations.
Apply at most one candidate chosen deterministically without GT, combine it
with unsupported-frame recovery in a real output run, then independently audit
serialization, full graph connectivity, trajectory and reprojection. See the
[hypothetical graph evidence](../benchmarks/electro/m8-openloris-atlas-cross-boundary-pnp-dsu-v1.json).

The archived pilot took 8.20 s at 304,536 KiB peak RSS. Independent repetition
produces byte-identical diagnostic rows and no model directory. With the flag
off, all six model/support/summary files remain byte-identical to fixed-pose
integration. These are diagnostic-only shared-machine runs, not COLMAP speed
results. Local full-workspace testing hit disk capacity; it is not recorded as
a pass. Only regenerable dev build artifacts were cleaned afterward.
See [resection evidence](../benchmarks/electro/m8-openloris-atlas-cross-boundary-pnp-v1.json).

Reproduce without GT or a model write:

```bash
cargo run --release --example integrate_rig_atlas_landmarks -- \
  --rig-manifest "$RIG_MANIFEST" --nodes-tsv "$ATLAS_INPUT/nodes.tsv" \
  --atlas-dir "$ATLAS_INPUT/atlas-l-newest/component-000" \
  --out-dir "$ATLAS_INPUT/diagnostic-unused-output" \
  --diagnose-cross-boundary-pnp --diagnostic-left-max-frame 1999
```

Here `ATLAS_INPUT` is the artifact parent recorded in the evidence, and
`RIG_MANIFEST` is the frozen 10k S/F manifest. The current CLI still requires
`--out-dir`, but diagnostic mode returns before creating or writing it.

For subsequent policy design, the upstream
[COLMAP local BA implementation](https://github.com/colmap/colmap/blob/main/src/colmap/sfm/incremental_mapper.cc)
refines a local bundle, completes/merges tracks, then filters observations.
That is different from rejecting a whole window when any track exceeds a
gate. The existing visloc `filter_positioned_track_observations` likewise
filters observations after BA. These are references for a future controlled
experiment, not evidence that filtering alone will recover missing connectivity.

## Real-model boundary repair

The default-off transactional implementation now passes 31 example tests and
Clippy. With unsupported-frame recovery followed by
`--repair-cross-boundary --repair-left-max-frame 1999`, the first candidate
(frame 2000) passes the geometry/support/connectivity policy. Only its two
sensor poses and affected landmarks are changed; existing calibration stays
fixed. Removed tracks and observations remain explicit: 5 tracks / 39
observations removed, 2 complete tracks / 30 observations added.

Independent auditing of the serialized output confirms all 4,999 rig frames
are supported and the main/tail track graphs are connected (4,494 + 505),
matching the frozen COLMAP component sizes. The frame 4493 recovery remains
intact. All bidirectional references, positive depths and fixed-rig checks
pass. An independent repeat yields byte-identical main model/support/summary
files.

| Quality measure | Recovery baseline | Boundary repair | Frozen COLMAP |
| --- | ---: | ---: | ---: |
| Supported rig frames | 4,999 | 4,999 | 4,999 |
| Main connected groups | 2 | 1 | 1 |
| Supported images | 9,997 | 9,997 | 9,996 |
| Trajectory RMSE | 0.391465 m | 0.391075 m | 0.384307 m |
| Trajectory p95 | 0.643175 m | 0.643107 m | 0.638669 m |

Aggregate mean reprojection is 0.633112 px with 353,794 landmarks and
1,453,320 observations. The main integration/recovery/repair pilot takes
11.99 s at 509,284 KiB peak RSS; this is not mapper/E2E timing.
Both trajectory gates still fail, so this is a connected refinement
foundation, **not** the M8 quality champion. Continue observation-based
refinement on this connected real model, preserving the original scoring
alignment and reporting any observation filtering. Full hashes and controls:
[boundary-repair evidence](../benchmarks/electro/m8-openloris-atlas-boundary-repair-v1.json).

## Next controlled experiment: strict BA on the connected model

PR #72 is merged at `bd07922798ac54d21d14a5bfc5ec544e947faffe`; its final
head passed all eight CI checks. The next experiment must separate the effect
of the repaired connectivity from any change to BA filtering or loss function.

Use one process: source integration → unsupported-frame recovery → boundary
repair → existing strict bounded BA. Do not feed repaired poses back through
source integration as an apparent BA-only input: doing so rebuilds tracks and
may change the observation set before optimization.

An optional pre-BA checkpoint must serialize the repaired in-memory state
through the existing writer and match all six frozen `boundary-repair-v1`
model/support/summary files byte-for-byte. Serialization order must not change
BA's variable or observation order. Keep only a small ordered reference/index
view, not a complete candidate model, and release writer buffers before BA.
Reject overlapping input/checkpoint/final destinations and do not overwrite
an existing nonempty checkpoint.

First retain the exact strict BA policy: 60-frame windows, stride 30, 20
iterations, fixed calibration, full selected tracks and outside-window anchors,
the existing resource caps, independently recomputed non-increasing cost,
and unchanged positive-depth/min-two/mean-2-px/max-4-px gates. No filtering or
threshold sweep belongs in this control. Check observation identities,
support, component membership and fixed rig geometry after serialization;
score only afterward with the unchanged original model alignment.

If strict BA still rejects useful windows, a separately named filtering arm
may follow the existing SfM filter/re-triangulation ordering. It must retain
the non-increasing **full pre-filter observation** cost gate, explicitly count
every removed observation/track, preserve supported image/frame sets and
prevent component splits. That arm is not implemented or accepted by this
plan alone. Both trajectory gates and the full M8–M10 performance/nonregression
requirements remain open.
