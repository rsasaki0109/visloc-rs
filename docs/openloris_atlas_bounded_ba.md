# Bounded observation-based atlas refinement

Status: strict BA was rejected for trajectory regression; the separately named
filtering arm improves RMSE/p95 to 0.388993/0.638173 m, passing only the frozen
p95 gate. RMSE and the full M8–M10 objective remain unmet. See the latest result
at the end of this document; earlier sections preserve the experiment history.
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

### Connected strict control result

Both pre-BA checkpoints match the frozen repair output across all six files.
The run without checkpoint output also produces identical final main files,
so checkpoint serialization does not change the BA result. The default-off
control remains byte-identical to fixed-pose integration; the tail is unchanged.

Strict BA again accepts 8/150 main windows and 0/17 tail windows. Of the 142
main rejections, the first failing track exceeds the max-4-px gate in 117
windows and the mean-2-px gate in 25. Windows starting at 1950 and 1980 remain
rejected despite the repaired graph. Main/tail connectivity and all support
are preserved, but RMSE/p95 worsen to 0.391408/0.643460 m. The strict result
is not promoted. Its 132.07 s / 525,412 KiB main measurement includes the
checkpoint and is only a shared-machine refinement pilot.
See [connected strict evidence](../benchmarks/electro/m8-openloris-atlas-connected-strict-ba-v1.json).

### Separate post-BA filtering arm

The next explicitly named mode keeps the solver, window schedule, calibration
and resource caps unchanged. It first checks finite, non-increasing cost on
the **entire original selected observation set**, before any removal. Only
then may it discard observations failing projection/depth/max-4-px checks.
Tracks with fewer than two observations or failing the unchanged DLT/mean-2-px/
max-4-px/depth checks are removed with complete accounting.

On the final retained keys, require independently recomputed post-filter cost
to be no larger than the **pre-BA cost on those same keys**. Log full pre/post
BA costs and retained pre-BA/post-filter costs separately. Require the same
supported image/frame sets and no component split. Stage only sparse candidate
updates and removal IDs, apply in place only after all gates pass, and validate
the entire model before publishing. No GT-based selection or threshold sweep.

Rebuilding connectivity scans the model for each candidate. This has bounded
extra state, but is not a claim of linear total runtime: window count and
global graph scans can multiply. Measure the validation cost before deciding
how to optimize it; do not weaken the connectivity check to hide that cost.

This isolates filtering, inspired by the upstream local-BA/filter ordering and
existing visloc rig filtering. It does not reproduce COLMAP's complete
merge/completion policy or establish a quality improvement before measurement.

### Independent deletion accounting

Run the transition auditor on each original component, followed by the existing
standalone geometry/rig/connectivity auditor on the actual output:

```bash
python3 scripts/audit_colmap_observation_filter.py \
  BASELINE/component-000/images.txt FILTERED/component-000/images.txt
python3 scripts/audit_colmap_pinhole_model.py \
  FILTERED/component-000 FILTERED/component-001 --rig-manifest RIG_MANIFEST
```

The transition auditor streams image rows and checks unchanged image identity,
camera assignment and every keypoint's serialized coordinates/order. It allows
point-ID renumbering and deletion only, rejecting added observations, track
merges/splits and lost image support. Its removed-observation and removed-point
counts must match the accepted-window log totals. It uses O(points) ID mappings;
its memory is audit overhead, not included in mapper RSS measurements.
It does not replace the second auditor's points3D references, projection,
fixed-rig and frame-connectivity checks.

The transition auditor's seven tests and the geometry auditor's thirteen tests
pass. Applied to the frozen connected strict control, it independently confirms
8,988 main image rows, 319,144 points and 1,327,687 observations unchanged, with
840 changed pose rows and zero removed observations or points.

### First connected filtering result

Implementation `be09d1a` passes 44 example tests and 20 independent Python auditor
tests. Both pre-BA checkpoints match the frozen repair baseline across all six
files; filter-OFF main output also matches the frozen connected strict BA output.

All 150 main and 17 tail windows are accepted, including the previously rejected
windows starting at 1950 and 1980. Main filtering removes 922 points / 13,788
observations; tail filtering removes 35 points / 652 observations. The independent
transition auditor reproduces these exact counts, with no added observations,
track merge/split, keypoint identity change or loss of supported images.
The independent geometry auditor confirms all 4,999 supported rig frames, one
4,494-frame main graph and one 505-frame tail graph, fixed calibration,
bidirectional references and positive depth. The final model has 352,837 points
and 1,438,880 observations, with observation-weighted mean reprojection error
0.581744 px. The earlier 0.580644 px display incorrectly weighted the component
observation means by point counts; it is superseded, without changing the
frozen COLMAP observation-weighted gate or any trajectory result.

With unchanged scoring, RMSE/p95 improve from 0.391075/0.643107 m to
**0.388993/0.638173 m**. Only p95 meets the frozen COLMAP gates
(0.384307/0.638669 m). This is an improved experimental refinement candidate,
not a production promotion or proof of full COLMAP parity.

Main/tail pilots take 201.84/15.60 s with peak RSS 525,236/77,684 KiB. They include
integration/recovery/repair/checkpoint/refinement, run on a shared machine with
overlapping control/repeat jobs, and are **not** mapper/native-E2E comparisons.
Both components repeat byte-for-byte across all six output files; the tail
repeat omits checkpoint writing and still matches. All eight CI checks pass
for implementation `be09d1a` (run 34094029224). Hashes, exact counts and scope are in
[filtered BA evidence](../benchmarks/electro/m8-openloris-atlas-connected-filtered-ba-v1.json).

### Next controlled arm: preserve valid optimized points

PR #73 is merged as `07a4104`, after all eight final-head CI checks passed
(run 34094661047). Its filtering arm remains the measured experimental candidate,
not a COLMAP-parity claim. The next branch is
`feat/m8-preserve-optimized-atlas-points`.

A code audit finds that every retained selected track currently replaces the
raw BA point with a DLT estimate, even when no observation was removed. The
main pilot has 120/150 nonconverged windows at the iteration cap. Its cumulative
full-observation raw BA cost falls 6.11%, whereas cumulative retained-key cost
after DLT falls 2.78%. **These percentages use different key sets and overlapping
windows; they do not prove DLT is the cause of the remaining trajectory gap.**
They motivate a controlled comparison, not more iteration/threshold searching.

The [upstream local BA implementation](https://github.com/colmap/colmap/blob/main/src/colmap/sfm/incremental_mapper.cc)
solves BA, merges/completes relevant tracks, then filters observations. It does
not unconditionally replace every retained optimized point with a DLT estimate
in that local-BA sequence. This motivates preserving an already valid solution;
the proposed arm still does not implement COLMAP's full merge/completion policy.

Use an explicitly named, default-off
`--joint-rig-ba-preserve-optimized-points`, requiring the filtering flag:

- If the post-classification observation keys are exactly unchanged, validate
  the raw BA XYZ against the same finite, bounded-sample parallax, positive-depth,
  min-two, mean-2-px and max-4-px gates. Retain it only if all gates pass, with
  freshly evaluated output metrics.
- Otherwise use the existing DLT path unchanged. In particular, any observation
  deletion still forces DLT. Do not add a new nonlinear solver or robust loss.
- Keep one 60-frame/stride-30 sweep, 20 BA iterations, calibration, anchors,
  resource caps and full-original/same-retained-key cost gates unchanged.
- Preserve the original support sets, connectivity and transactional deletion
  ledger. Stage only the bounded window's updates.
- Log raw-retained and DLT-attempted track counts plus fallback reasons by
  window and cumulatively; these count window events, not unique model points.

Freeze the same source/recovery/repair checkpoint before either arm. First
prove flag-OFF equality to `connected-filtered-ba-v1`, then run the new arm,
independently audit the actual output and score only after mapping. Require
repeat byte identity and retain negative results. Both COLMAP trajectory gates
and the entire M8–M10 speed/memory/nonregression objective remain unchanged.

The previous photometric quadrilateral experiment is already complete on a
different mapper input; do not repeat it as if it were new. Neither optimized
point retention, fixed-pose nonlinear point refinement nor a second sweep has
yet been measured on this connected filtering input. Test point retention first
to isolate this implementation difference before introducing other changes.

### Optimized point retention result: not promoted

Implementation `0e5a9bf` passes 47 example tests, 21 independent auditor tests,
clippy and release build. Both pre-BA checkpoints match the same repaired model;
retention-OFF main output matches the frozen filtering control across six files.
Both retention-ON component outputs repeat byte-for-byte, including when the
repeat omits checkpoint writing. All eight implementation CI checks pass.

Main/tail accept 150/150 and 17/17 windows. Raw points are retained in
846,957/82,888 selected-track events, with only 736/46 DLT attempts. These are
overlapping window events, not unique point counts. Final removal accounting
is 384 points / 4,408 observations, independently reproduced from serialized
keypoint identities. All supported images, 4,999 rig frames, the 4,494/505-frame
connected graphs, calibration and reference/depth/track-mean/max-error gates pass.

Observation-weighted reprojection improves slightly from 0.581744 to 0.580628 px,
but RMSE/p95 **worsen from 0.388993/0.638173 to 0.389420/0.639357 m**. Both
COLMAP trajectory gates now fail. Do not promote retention or select it merely
because it avoids DLT or preserves more landmarks. Retain the previous filtering
model as the improved experimental candidate.

Main/tail pilot wall times are 235.31/25.93 s at 525,676/77,896 KiB; they run
with overlapping jobs on a shared machine. Neither these numbers nor fewer DLT
calls establish mapper/native-E2E speed. Full hashes and scope are in
[point retention evidence](../benchmarks/electro/m8-openloris-atlas-connected-preserved-ba-v1.json).

### Next controlled arm: exactly two filtering sweeps

Point-retention PR #74 is merged as `f92fa3c` after eight final-head CI checks
passed (run 34097238801). Keep retention disabled. The next comparison changes
only the number of original filtered/DLT sweeps from one to exactly two; it is
not an arbitrary pass-count search or a combined retention/refinement policy.
Previous native-mapper refinement experiments used different inputs/tracks and
do not substitute for this connected-atlas comparison.

- Add `--joint-rig-ba-filter-sweeps 1|2`, default 1. Explicit use requires the
  filtering flag. Reject zero, values above two and duplicate arguments.
- Two sweeps require `--post-pass1-out-dir PATH` and reject optimized-point
  retention. The checkpoint must be new or empty, with the same input/output/
  ancestor/symlink collision checks as the existing pre-BA checkpoint.
- The one-sweep path must preserve existing numerical order, logs and output.
- Run the existing filtered/DLT function once, validate, update output counts
  and write a canonical reference-view checkpoint. Do not sort/mutate solver
  input, reparse the checkpoint or clone the complete model. Release writer
  buffers before applying the same function again to the in-memory state.
- Keep window length/stride, 20-iteration solver cap, point/observation/pose
  caps, anchors, calibration and every full-cost/retained-cost/geometry/support/
  connectivity/transactional guard unchanged in each pass.
- Mark pass boundaries and keep each pass's summaries separate. Costs sum
  overlapping window events; they are not a single global objective value.
  Count deletion reasons without retaining a pass-wide duplicate key map.
  Since accepted transitions only delete observations, a deleted key cannot
  reappear in another window; audit this independently from serialized outputs.

Before evaluating pass two, require both pass-one checkpoints to match all six
`connected-filtered-ba-v1` files. The pre-BA checkpoints must still match
`boundary-repair-v1`. Run transition audits for repair→pass one and pass one→
pass two, plus the independent final geometry/rig/connectivity audit and fixed
post-map scorer. Require two-run output equality and a one-sweep control. No GT
or score may decide mapping acceptance or termination. Keep all original M8–M10
gates, including mapper/native-E2E timing and total peak RSS, unchanged.

### Two-sweep result: not promoted

Implementation `c99c381` passes 50 example tests and 23 independent auditor
tests. Both pass-one checkpoints match the existing filtered model, both pre-BA
checkpoints match boundary repair, and the default one-sweep main control is
unchanged, all across six files. Pass two accepts 145/150 main windows and
17/17 tail windows, removing another 135/5 points and 2,120/73 observations.
Independent transition audits reproduce these counts without additions,
splits, merges or keypoint identity changes. Final geometry audits preserve
9,998 poses, 9,997 supported images, 4,999 supported rig frames and connected
4,494/505-frame graphs. Fixed extrinsics, bidirectional references, positive
depth, track-mean <=2 px and observation-max <=4 px all pass.

Final output has 352,697 landmarks and 1,436,687 observations. Observation-
weighted mean reprojection improves to 0.574285 px and p95 to 0.633716 m,
but **RMSE worsens from 0.388993 to 0.390165 m**, still above COLMAP's
0.384307 m gate. Do not promote two sweeps or launch a pass-count search.
Both final outputs and pass-one checkpoints repeat across all six files per
component, even with optional pre-BA checkpoint writing omitted on repeat.
See [two-sweep evidence](../benchmarks/electro/m8-openloris-atlas-connected-two-sweep-ba-v1.json).
Main/tail pilot times are 366.03/23.23 s
at 525,168/77,932 KiB with overlapping shared-machine jobs, not mapper/native
E2E timing or a speed comparison. The one-sweep candidate stays in README's
COLMAP comparison table; do not replace it with a cherry-picked p95 result.

### Subsequent output-preserving performance candidate

Keep this separate from the two-sweep quality experiment. Code inspection of
the normal filtering-window path finds three calls to
`selected_landmarks_for_frames`: window-count diagnostics, filtering candidate
construction and raw BA construction each repeat the full landmark scan.
Each scan tests observation image/frame membership and returns the same ordered
index vector before any candidate is applied.

After freezing the quality result, test computing that vector once per window
and borrowing it for the three consumers. Keep the exact index/observation order,
selection definition, caps, rejection behavior and independent geometry checks.
Discard it before the next window mutates/compacts the landmark vector; never
cache these indices across accepted updates. This needs no global adjacency
index or extra whole-model copy. Verify complete output equality and measure
time/RSS before claiming a gain. Removing duplicate scans does not eliminate
the remaining per-window global scan or prove linear total runtime.
