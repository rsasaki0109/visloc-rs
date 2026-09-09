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

Two-sweep PR #75 merged as `0989877` after all eight final CI checks passed
(run 34100723331); its old local/remote branch is removed. Work proceeds on
`perf/m8-reuse-atlas-window-selection`. The archived baseline executable is
`selected-scan-reuse-v1/baseline-integrate_rig_atlas_landmarks` under the frozen
atlas artifact root (same binary SHA as the two-sweep evidence). Default
one-sweep main run `baseline-main-1` takes 172.69 s / 525,052 KiB and matches
all six frozen filtering files. No other task benchmark overlapped this run,
but the machine is shared. Do not treat this single baseline as a speed gain.
Compare multiple serial runs before/after, verify outputs and per-window logs,
and retain mapper/native-E2E scope limitations.

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

The selection-reuse implementation `1eacc08` passes 51 example tests and 23
independent auditor tests. In six serial, warm-input, shared-machine main-
component runs, baseline wall times are 172.69/172.80/169.48 s and candidate
times are 154.39/156.84/152.09 s. Medians are **172.69 → 154.39 s (10.6% less
time)**, with essentially unchanged RSS medians of 525,232/525,036 KiB.
All six model files and complete logs match across every run. No task benchmark
or build overlaps these measurements. Other-mode regression checks now pass:
strict main, preserved-point main/tail, two-sweep main/tail and filtered tail
each match all six frozen output files. Both two-sweep pass-one checkpoints
also match their frozen filtering models. This measures
integration/recovery/repair/filtering/publication only,
not the source-window mapper, atlas construction, frontend or native E2E.
See [selection reuse evidence](../benchmarks/electro/m8-openloris-atlas-selected-scan-reuse-v1.json).

### Solver-memory audit (production path unchanged)

The current pure-visual sparse path in `pipelines/slam/src/bundle.rs`,
`solve_step_pose_blocks`, avoids a dense camera Hessian but still materializes
the reduced Schur matrix as block-column maps. Its nested pose-pair loop for
each landmark adds shared-track couplings, and it calls the direct cached
block-Cholesky solver. `LandmarkBlock.cross` retains pose/landmark cross blocks.
PR #77 initially validated only a private test prototype. The additive runtime
entry point described below keeps the legacy direct path unchanged.

A matrix-free Schur operator could avoid storing the reduced matrix and its
factor fill, but would still retain observation/cross-block state unless that
is explicitly streamed. It must not silently reuse the current global-BA
memory or quality claims: the earlier >2 GiB global experiment used a different
native-mapper input, not this connected atlas. No such method has been measured
on the connected-atlas input and no accuracy or memory win is established.

Before selecting this larger change, require small-system operator/step checks
against explicit Schur with identical damping, bounded iteration and residual
criteria, singular/low-parallax handling, deterministic LM acceptance, fixed
rig calibration and correct component gauges. Whole-process peak RSS and time
still need measurement. This audit identifies a memory option; it does not
authorize an unbounded global solve or replace the outstanding quality gate.

Primary references checked for this option: [Ceres' iterative Schur
documentation](https://ceres-solver.readthedocs.io/latest/nnls_solving.html#iterative-schur)
describes CG applied to the reduced camera system through implicit matrix-vector
products; `SCHUR_JACOBI` uses its block diagonal as a preconditioner.
[Agarwal et al., Bundle Adjustment in the Large, Eq. 12](https://homes.cs.washington.edu/~sagarwal/bal.pdf)
derives the implicit product and analyzes the diagonal preconditioner's storage.
These sources motivate an operator-level comparison, not a prediction that
global refinement will improve this dataset's trajectory. A future experiment
must retain identical observations, calibration, damping and GT-free acceptance
before attributing any change to the solver.

#### Private operator gate

PR #76 merged as `56bea96` after all eight final CI checks passed
(run 34104286558); its old branch is removed. The selected next bounded task
on `feat/m8-implicit-schur-prototype` starts with a private, test-only operator/PCG
prototype, not a new mapper flag or global solve. Reuse `NormalEquationsBa`,
`LandmarkBlock` and the already constrained `CameraHessian::PoseDiagonal`;
explicitly reject `Dense` input. Do not add a variant to the shared public
`LinearSolver` enum or a required field to public `BaConfig` struct literals.
A later production entry point must reject unsupported priors/factors before
normal-equation assembly: rejecting an already allocated `Dense` system cannot
undo its quadratic allocation. The private prototype alone does not prove this
pipeline-level memory gate.

For each landmark, compute the sum of all cross-block transposed products
before applying its damped 3-by-3 inverse, then scatter through every original
cross block. Multiple sensor observations may share the same rig-pose slot:
their cross terms must interact, not be treated as separate poses. A Schur
block-diagonal preconditioner must likewise aggregate cross blocks by pose
before forming each diagonal contribution. Keep deterministic accumulation
order and avoid pose-pair blocks, triplets or factor-fill state.

Match the current pose/landmark identity damping and singular-landmark inverse
skip/back-substitution behavior in the operator oracle. Check operator action,
reduced RHS, preconditioner diagonal and complete pose/landmark step against
small explicit systems, including repeated sensor slots, zero/nonzero damping,
fixed rotations and empty pose systems. The prototype rejects an empty pose
system; this is not a test of the production landmark-only/all-fixed solve.
That separate branch must be validated during integration. Test dimension/nonfinite errors,
residual criteria, iteration limits and repeated-run determinism. Numerical
agreement with a direct solver is tolerance-based; default-path compatibility
and same-prototype repeat determinism are separate checks.

Linear residual convergence, including zero RHS or a system regularized by
positive damping, is not evidence of a physically valid gauge. Component
anchoring and fixed-rig calibration remain model-level prerequisites before
any later integration. This gate proves a solver primitive only; global memory,
runtime, trajectory quality and all original M8–M10 outcomes remain unproven.

The implementation (`469ee9a`, numerical supplement `8a1485a`) passes all ten
private prototype tests and all 21 tests in the BA namespace, independently
rerun by the reviewer. A hand-computable two-pose fixture checks zero/nonzero
damping against the full 15-by-15 normal system, including same-pose sensor
cross terms. The rig matvec tolerance uses the magnitudes of the unreduced and
eliminated terms so cancellation does not hide the floating-point scale.
Clippy, formatting and 46 relevant Python tests pass. PR #77 passed all eight
final-head CI checks (run 34108930129), merged as `0460c9c`, and its old branch
was removed; [numerical evidence](../benchmarks/electro/m8-openloris-implicit-schur-prototype-v1.json)
records scope and limitations. No production or 10k performance claim follows.

#### Integration gate (runtime API merged in PR #78; 1k numerical agreement failed)

After the private numerical gate and CI, expose an additive, opt-in pure-visual
entry point with separate iterative options. Keep existing public enum variants,
configuration struct literals and default solver behavior unchanged. Before
assembling normal equations, reject velocity/bias states and unsupported
priors or nonvisual factors; never fall back to a dense camera matrix. Check
component anchors and fixed physical rig calibration at the model boundary.

First compare the same anchored small model against the direct solver, then a
frozen 1k input with identical observations, calibration, damping and robust
loss. Record true linear residuals, iteration counts, nonlinear cost acceptance,
whole-process peak RSS, time and output geometry. Explicitly test landmark-only
systems and rejected steps without partial pose/point mutation. A failed PCG
solve must be visible and handled through a bounded rejection/damping policy,
not an unbounded direct-solver fallback. Preserve deterministic repeated runs.

Only after these gates pass, select a separately recorded connected-atlas 10k
experiment with a 2 GiB stop limit and fixed observation/support/geometry
checks. Keep GT out of optimization and acceptance; score the unchanged two
components afterward. Lower linear residual or nonlinear reprojection cost
alone does not meet the trajectory gate. Mapper-only and native-E2E accounting
must include their respective upstream stages; this prototype supplies neither
a pipeline memory bound nor an end-to-end speed comparison.

#### Runtime entry-point contract

`BundleAdjustment::optimize_matrix_free(&BaConfig, MatrixFreeBaOptions)` is the
explicit selection point. `BaConfig::linear_solver` remains relevant to the
existing entry points; this method always requests pose-diagonal assembly and
implicit Schur PCG. Separate options, result diagnostics and errors avoid
changing the fields or variants of existing public configuration/result/error
types. The operator implementation is shared with its numerical oracle tests;
LM acceptance, rollback and damping are shared with the existing optimizer.

The initial scope is fixed, undistorted PINHOLE/SIMPLE_PINHOLE calibration with
monocular, rectified-stereo, general-stereo or rig visual observations. Reject
intrinsics/distortion refinement and nonvisual states/priors before assembly.
Require finite model state and observations, valid referenced IDs, finite
positive LM damping with ordered bounds, and bounded PCG options. The new entry
point rejects an all-fixed-pose problem rather than silently selecting a
different solver; the existing landmark-only optimizer remains available.

At least one existing pose must be fixed. This is only a minimal eligibility
check: the caller still must anchor every component and remove monocular scale
freedom. Do not infer physical gauge validity from damping or convergence. A
calibrated multi-sensor test must use a nonzero physical baseline and perturbed
state; a zero-cost identity-sensor fixture does not verify an actual rig update.

Per-iteration diagnostics distinguish failed solves from successful true-
residual checks. Unknown values are optional rather than reported as zero.
The linear residual is a normal-equation residual, not pixel reprojection or
trajectory error. PCG failure rejects the step without applying its delta,
records the reason, and enters the shared bounded LM damping retry. No dense
or direct fallback is permitted. The limit is at most one bounded PCG solve
per LM iteration, with termination at the LM iteration/damping limits.

The [frozen 1k input](../benchmarks/electro/m8-openloris-matrix-free-1k-input-v1.json)
has been independently hash-checked, geometry-audited and trajectory-rescored.
The single-arm driver is implemented in `c810be9`; ten example tests, targeted
clippy, formatting and the release build pass. Initial real-model results are
below; input preparation and API unit tests alone are not the comparison gate.

#### Frozen 1k post-map experiment contract

The comparison driver on `feat/m8-matrix-free-rig-ba-comparison` runs one solver
per process on the same immutable model. It resolves image names through the
rig manifest, derives body poses from sensor 0 and checks the other sensor
poses against the fixed calibration. The source has 500 rig frames, 1,000
supported images, 4,716 landmarks and 130,900 observations in one connected
frame component. Its 174 cross-sensor tracks include 136 with same-frame
stereo; this is evidence of metric observations, not a numerical-rank proof.

Both arms fix frame 0 only, retain all landmarks and observations, and use
20 LM iterations, initial damping `1e-4`, no robust loss, fixed intrinsics and
distortion, and `BaConfig::parallel=false`. The candidate starts with 128 PCG iterations
and relative/absolute tolerances `1e-12`. Any later iteration-budget arm must
be separately labeled and justified by the recorded true linear residuals;
do not silently loosen tolerances or fall back to a direct solve.

Preserve image IDs/names/order, full POINTS2D coordinates/index/point IDs,
point IDs/RGB and track pair order. Only poses, XYZ and recomputed per-track
arithmetic mean ERROR may change; copy cameras byte-for-byte. The historical
source ERROR is not reliable, so audit reprojection directly. Its maximum
track mean exceeds the later atlas-specific 2 px threshold: do not import that
filter into this all-observations comparison. Publish through a staging
directory into a new or empty non-overlapping output, and stream output rows
without constructing and reparsing another whole model.

Require identical initial nonlinear cost between arms. Independently audit
the output identities, bidirectional tracks, positive depths, rig calibration,
image/frame support and connected component. Score the same 308 GT-associated
images afterward using the frozen scorer and calibration; GT is never passed
to optimization. Numerical agreement with direct BA, improvement over the
input, and the COLMAP trajectory gate are separate results.

The first runs exposed an independent thread control: the existing block
Cholesky backend uses the Rayon pool even with `BaConfig::parallel=false`.
Retain those runs as automatic-thread diagnostics (nine live process threads
were observed for direct BA); use explicit `RAYON_NUM_THREADS=1` for separately
labeled single-thread controls. Do not call the initial direct runs single-threaded.

Record pure solve time separately from whole-process elapsed time and peak
RSS, including loading, validation, problem assembly and publication in the
latter. Run arms serially with no concurrent project build or benchmark;
record the shared-machine limitation. These are **post-mapping BA** costs,
not mapper-only or native end-to-end costs. Repeated runs must reproduce the
model and numerical trace, excluding timing/path fields. No 10k experiment or
performance promotion follows until the actual 1k results have been reviewed.

The experimental command uses a complete matching rig manifest and requires
every image to have support in one connected rig-frame component:

```bash
cargo build --release --example compare_rig_bundle_adjustment
# Use separate new/empty output directories; run these processes serially.
RAYON_NUM_THREADS=1 /usr/bin/time -v target/release/examples/compare_rig_bundle_adjustment \
  --model /path/to/frozen/model --rig-manifest /path/to/rig-manifest.txt \
  --solver direct --out-dir /path/to/results/direct
RAYON_NUM_THREADS=1 /usr/bin/time -v target/release/examples/compare_rig_bundle_adjustment \
  --model /path/to/frozen/model --rig-manifest /path/to/rig-manifest.txt \
  --solver matrix-free --out-dir /path/to/results/matrix-free
```

Initial 1k runs have identical initial cost `126510.3987573094`. Direct BA ends
at `117496.0317308232`, whereas default 128-iteration PCG ends at
`126419.0824771856`; both exhaust 20 LM iterations without convergence. PCG
reports both iteration-limit failures and true-residual verification failures,
so its shorter run is **not equivalent-work acceleration**. Recomputed mean
reprojection is 0.689260 px for direct and 0.720314 px for PCG, versus 0.720679 px
for the input. Direct increases maximum observation error from 3.999569 to
4.937790 px; retaining all observations is not a guarantee of maximum-error
nonregression.

Independent output audits retain all 1,000 images, 500 supported frames,
4,716 landmarks and 130,900 observations in one component, with positive
depths and fixed calibration. Every original POINTS2D token and point
ID/RGB/track token/order is preserved. Post-map RMSE/p95 is
0.026519/0.040989 m for direct and 0.026703/0.041347 m for PCG. These scores do
not establish numerical agreement between solvers.

The [complete 1k comparison record](../benchmarks/electro/m8-openloris-matrix-free-rig-ba-comparison-v1.json)
contains ten serially executed processes: two automatic-thread runs per solver,
then two explicit single-thread runs each for direct, PCG-128 and PCG-512.
Every same-condition repeat has byte-identical model files and numerical
traces. All PCG variants produce the same final model. PCG-512 changes the
failure counts from 12 iteration-limit / 5 true-residual failures to 10 / 7,
but still accepts only three steps; increasing the budget alone is not promoted.
Single-thread direct costs differ slightly from automatic-thread direct
(`117496.0330739301` versus `117496.0317308232`), so cross-thread bit identity is
not claimed. Compared with single-thread direct, PCG retains maximum camera
centre/landmark differences of 0.003442/0.042606 m: the solver-agreement gate fails.

Single-thread whole-process times are 45.54/57.34 s for direct, 6.43/8.08 s for
PCG-128 and 17.53/20.90 s for PCG-512. Corresponding peak RSS ranges are
161,736–161,780 / 84,644–84,692 / 84,436–84,516 KiB. These shared-machine
measurements describe different optimization progress, not an equivalent-
quality speedup or a pipeline memory win. The README comparison is not promoted.

Next use the exact same initial normal equations at solve damping `1e-4` and
`1e10` to compare explicit Schur/direct and implicit action/RHS/steps, true
residuals, predicted decrease and feasibility. PCG makes no state updates before
LM14, so the initial model is sufficient for both diagnostic damping values.
Existing LM trace `lambda` is the increased value after rejection, but the
solve value after acceptance: LM0 solves at `1e-4`, not its logged rejection
value `1e-3`. Separate operator cancellation/residual drift, conditioning and
convergence hypotheses before changing preconditioning or residual updates;
the current measurements do not prove which is the sole cause. Preserve the
true-residual gate and keep this next diagnostic in a separate PR.

The next oracle must remain library `#[cfg(test)]` code, with an explicitly
ignored real-input test and ordinary small synthetic CI tests. A hidden or
nondefault-feature public method still expands the public API and is not the
selected approach. If private test code needs an interchange input, prefer an
explicit example-only export of bounded pose/point/observation records linked
to the frozen source hashes, preserving every observation and its order; do
not export a normal matrix or duplicate the complete source model in memory.
The exporter/fixture was subsequently implemented in `3e50dc5`; measured results
are recorded below.

Before any explicit Schur allocation, cap variable poses at 512, points at
8,192, observations at 262,144, scalar dimension at 3,072 and pre-count the
cross-pair work against a 64,000,000-pair bound with checked arithmetic.
The frozen 1k input has 130,677 variable-pose cross blocks and
`sum(cross_length²) = 28,556,575` (maximum track cross length 891), counting
multiple sensor blocks for the same frame separately. A real-input test must fail clearly
when its requested fixture is missing or over cap, not report an empty passing
measurement. Reuse one initial normal system across the two damping values;
keep only the small pose-diagonal copy needed by the mutating direct path.
Record arithmetic scales and check the existing gradient/RHS sign and the
objective's factor of two before reporting predicted reduction. No claim that
the direct step itself meets the PCG residual target is made before measuring it.

The direct implementation factors its lower-triangle Schur storage. Report
raw explicit Schur asymmetry and distinguish its full matvec from the
lower-mirrored symmetric matrix actually used by factorization. For the
current unrobust cost `sum(||r||²)`, assembly stores `b = Jᵀr` and `H = JᵀJ`:
the undamped linearized cost reduction is `-2 bᵀδ - δᵀHδ`. A half-cost damped
quadratic prediction is a different quantity and must be labeled accordingly.
Test these signs and factors independently on small systems.

Primary-source context for the next diagnosis (checked 2026-09-07):

- [Ceres nonlinear least-squares documentation](https://ceres-solver.readthedocs.io/latest/nnls_solving.html)
  describes inexact LM for iterative solves and enables Jacobian-column scaling
  by default. Its `eta` stopping rule uses quadratic-model progress, not this
  prototype's fixed `1e-12` true-residual rule. These are not interchangeable
  tolerances. Scaling or an inexact policy could be a later controlled arm,
  not a silent reinterpretation of the current failed gate. Distinguish an
  equivalent change of linear coordinates from changing the physical-coordinate
  LM damping metric; neither permits scaling fixed rig extrinsics.
- [PETSc KSPPIPECGRR](https://petsc.org/main/manualpages/KSP/KSPPIPECGRR/)
  uses residual replacement to improve robustness of pipelined CG. That is a
  specific recurrence, not the classical PCG implemented here. It motivates
  measuring recursive/true residual gaps, but does not establish that residual
  replacement alone fixes this input or justify adopting its update rule
  without a separate numerical test.

### Frozen 1k real normal-system oracle (2026-09-07)

[Machine-readable evidence](../benchmarks/electro/m8-openloris-real-normal-system-oracle-v1.json)
records the bounded private oracle implemented in `3e50dc5`. The standalone
example export is additive; normal solver CLI behavior and the public library
API are unchanged. Example usage (the output file must not already exist):

```bash
cargo build --release --example compare_rig_bundle_adjustment
target/release/examples/compare_rig_bundle_adjustment \
  --model "$MODEL" --rig-manifest "$RIG_MANIFEST" \
  --export-oracle-fixture "$EXISTING_OUTPUT_DIR/input.fixture"
cargo test --release -p visloc-slam --lib matrix_free_real_oracle_tests --no-run
# Set both variables to the exported fixture and its printed combined source hash.
RAYON_NUM_THREADS=1 \
VISLOC_MATRIX_FREE_ORACLE_FIXTURE="$EXISTING_OUTPUT_DIR/input.fixture" \
VISLOC_MATRIX_FREE_ORACLE_EXPECTED_SOURCE_SHA256="$SOURCE_SHA256" \
cargo test --release -p visloc-slam --lib \
  bundle::matrix_free_real_oracle_tests::ignored_real_fixture_runs_bounded_damping_oracle \
  -- --exact --ignored --nocapture
```

The 30,587,925-byte fixture preserves all 130,900 observations in source-track
order, all 4,716 landmarks, name-to-rig mapping, fixed sensor extrinsics and
fixed pose 0. Independent checks verified source/combined hashes, observation
identity and coordinates, point coordinates and initial-cost bits
`4683430141614839853` (126,510.39875730938). No observation was removed.
The same initial normal system has 499 free pose blocks / scalar dimension 2,994.

| Initial solve damping | Matrix-free PCG 128 / 512 | Direct step residual: lower / implicit | Trial geometry |
|---|---|---|---|
| `1e-4` | both fail; true residual 41,263.7 / 318.346 against `7.35e-7` | `2.75e-6` / `0.01848` | 880 nonpositive-depth observations |
| `1e10` | both succeed at 22 iterations; `5.64e-7` against `7.38e-7` | `3.02e-9` / `4.49e-7` | all 130,900 valid |

At low damping, sparse direct and explicit lower-mirrored Schur agree to
`1.02e-10` in pose-delta norm and `1.78e-8` in landmark-delta norm, but **the
direct solution itself does not meet the implicit residual target**. The raw
Schur asymmetry and different accumulation orders matter to further diagnosis;
this does not establish one sole cause of PCG failure. The deterministic probe's
raw/lower action errors are 0.18767 / 0.13604. Their ratios of about `1e-13`
use the **unreduced** arithmetic scale (`9.94e11`), not final `Sx` or solution
accuracy. PCG recursive and true residuals are close at these low-damping
iteration limits, so residual replacement alone is not established as a fix.

At high damping, matrix-free/explicit pose-delta difference is `2.09e-17` and
all trial geometry costs are 126,477.42813448103. Low-damping trial geometry cost/RMS
exclude 880 invalid observations and must not be called full-objective
improvements. Reported feasibility means finite and valid geometry, not that
the PCG residual target passed. Prediction fields distinguish half-cost damped,
squared-cost damped and squared-cost undamped reductions.

Two serial release runs produced identical numerical reports (16.84 / 16.62 s;
346,084 / 346,020 KiB peak RSS). These are whole **diagnostic** processes with
explicit matrices, not production solver speed/memory results. Both existing
direct and matrix-free CLI regression runs preserved all three model files
byte-for-byte and the numerical trace against PR #79. No README performance
claim or 10k promotion follows from this diagnostic.

Next compare test-only PCG on the lower-mirrored explicit Schur against the
implicit action, with the same input, damping, block-Jacobi preconditioner,
128/512 limits and true-residual rule. This isolates accumulation from
convergence/preconditioning before selecting scaling or a different
preconditioner. Preserve the baseline and do not silently relax its tolerance.

PR #80 passed all eight final-head CI checks (run `34122318818`, head
`b294bab`) and merged as `2b4ef99`; its local and remote topic branches were
removed. The fixture also re-exported byte-identically to a second filename.
The arithmetic-scale denominator above is specifically
`max(1, ||base_action|| + ||accumulated_eliminated_action||)`, not a sum of
individual landmark-action norms.

### Explicit-PCG isolation contract

The next test-only arm borrows the existing lower-mirrored Schur matrix and
calls the **same existing implicit operator's preconditioner application**.
It must not substitute an independently recomputed block diagonal: that would
change two numerical components at once. Preserve RHS, zero initial iterate,
PCG recurrence, true-residual recheck, damping and iteration caps. A test-only
callback recurrence must reproduce the production PCG when given the implicit
action, including failure diagnostics, before interpreting its explicit-action
comparison. This leaves production APIs and defaults untouched.

For explicit PCG, report both its own lower-Schur residual and a re-evaluation
with the implicit action; convergence for one finite-precision representation
does not establish convergence for the other. Report successful complete steps
and trial geometry separately from failed numerical trials. Retain both damping
values and the previous oracle report as controls. Reuse the existing bounded
matrix and fixture, not an additional model clone or a new production dense path.

Include `lambda=1e5` in the new isolation report in addition to the two endpoint
controls. The frozen PR #79 `direct-serial-1.log` rejects LM0 through LM8 and
first accepts LM9 at solve damping `1e5`, with trial cost
118,084.4047208355. Thus this third damping also uses the unchanged initial
model and tests a step that actually improved the baseline optimizer, rather
than only a geometrically invalid low-damping step and a tiny high-damping
step. Preserve the original four-case oracle report for direct comparison;
the isolation report has six cases (three damping values, two iteration caps).

If both representations stall, investigate preconditioning and conditioning;
if only the implicit representation stalls, prioritize accumulation and
symmetry. These are hypotheses to refine with the measured residuals, not a
binary proof of a unique cause. Do not turn a shorter failed solve into a speed
claim, relax the stopping threshold, or promote to 10k from this experiment.

[Ceres' official solver FAQ](https://ceres-solver.readthedocs.io/latest/solving_faqs.html)
(checked 2026-09-07) discusses explicit Schur for smaller problems and stronger
cluster preconditioners when Schur-Jacobi is insufficient. This supports testing
the representations/preconditioner separately; it does not establish that a
cluster implementation will fit this project's memory budget or solve its
current numerical failure. Any later cluster arm needs an explicit bounded
storage/work design and an unchanged-quality comparison.

A separate numerical-stability candidate, not selected for this isolation PR,
is [Demmel et al., CVPR 2021, Square Root Bundle Adjustment](https://openaccess.thecvf.com/content/CVPR2021/html/Demmel_Square_Root_Bundle_Adjustment_for_Large-Scale_Reconstruction_CVPR_2021_paper.html).
It eliminates landmark variables with QR/nullspace operations and reports
better numerical stability than normal-equation Schur elimination, but also
larger memory requirements on dense problems. The authors' [RootBA OSS](https://github.com/nikolausdemmel/rootba)
is a reference, not a dependency added here. Our inference is that such a
representation may merit a bounded experiment if accumulation is limiting;
its reported BAL results do not prove a win for this calibrated rig or justify
replacing the current linear-storage design without a track-length memory audit.

An independent frozen-fixture count illustrates that constraint: storing a
naive dense `2 * observation_count` by `6 * unique_free_pose_count` camera
Jacobian for every landmark would require 232,363,908 f64 entries,
1,858,911,264 bytes (1.731 GiB), already at 1k and before damping, RHS or
workspace. Same-frame sensor observations share pose columns; fixed pose 0
is excluded from columns but its observation rows remain. Maximum track length
is 893 observations. This is a hypothetical storage warning, **not** measured
RootBA memory or a lower bound for implicit/streamed QR. Do not select that
naive layout for the 10k implementation.

### Explicit-PCG isolation results (2026-09-07)

[Evidence](../benchmarks/electro/m8-openloris-explicit-pcg-isolation-v1.json)
records implementation `cb14bdc`. Both arms use the same normal system, RHS,
existing block-Jacobi preconditioner and stopping rule. The original four-case
oracle is assembled separately first and reproduces all PR #80 numerical lines
exactly. The new isolation system is then assembled once for all six cases.

| Damping | Cap | Explicit lower-PCG true residual | Implicit-PCG true residual | Outcome |
|---|---:|---:|---:|---|
| `1e-4` | 128 | 5,614.69 | 41,263.7 | both hit cap |
| `1e-4` | 512 | 0.0355111 | 318.346 | both hit cap |
| `1e5` | 128 | 1,411.68 | 1,673.76 | both hit cap |
| `1e5` | 512 | `8.947e-7` | `9.079e-4` | explicit recheck fails at 316; implicit hits cap |
| `1e10` | 128 / 512 | `3.336e-8` | `5.639e-7` | explicit 23 / implicit 22 iterations, both pass |

Targets are approximately `7.35e-7` to `7.38e-7`. At the useful `1e5` damping,
explicit recursive residual reaches `7.537e-8`, but its true residual exceeds
the target, so rejection is correct under this baseline contract. The dense
reference itself has lower/implicit residuals `2.143e-6` / `0.002572` there.
This does not prove the target is mathematically unattainable. It does show
that replacing the operator with an explicit matrix is not sufficient to pass
the current gate, and that both failures cannot be attributed solely to the
preconditioner. Failed iterates are discarded, so their cross-operator
residuals, deltas and trial geometry are unavailable, not zero.

At high damping, the explicit solution also passes the implicit-action recheck
(`1.275e-7`); its pose/landmark delta differences from the implicit solution are
`2.09e-17` / `9.92e-18`. All 130,900 trial observations remain valid and trial
geometry cost is unchanged from PR #80. Two serial release executions reproduce
all eleven numerical lines exactly. Whole diagnostic process wall times are
45.02 / 44.77 s and peak RSS 346,196 / 346,148 KiB; these include reference
factorizations and repeated arms, are from a shared host, and are not solver
speed or production memory results. No new model or README claim is promoted.

The next bounded improvement candidate is a test-only damped-landmark-block
Cholesky construction versus the current general 3x3 inverse, used consistently
by the implicit action, explicit reference and preconditioner. First measure
the input-block and inverse asymmetry; do not assume it is nonzero. Keep
physical coordinates, damping, observations, caps and stopping rule unchanged,
then compare actual residuals, convergence and valid-step quality. A win here
must still pass a full nonlinear 1k comparison before production promotion.

[Ceres' Schur implementation](https://ceres-solver.googlesource.com/ceres-solver/+/master/internal/ceres/schur_eliminator.h)
discusses Cholesky inversion for SPD landmark blocks but uses a general inverse
in its small fixed-size specialization for speed. Thus it supports considering
the stability/speed tradeoff, not claiming that Cholesky is always faster or
that the existing inverse is inherently incorrect. Separately labeled
coordinate scaling or an inexact LM policy remain options after this A/B;
the fixed `1e-12` rule is a diagnostic baseline, not the user's ultimate goal.
Any changed policy must retain honest true-residual reporting and demonstrate
the requested trajectory/reprojection/resource quality, not merely report more
linear solves as successful.

PR #81 passed all eight final-head CI checks (run `34125303050`, head
`7b4b6cb`) and merged as `06ca2e1`; its local/remote topic branches were removed.
The Cholesky A/B must audit original `Hll` symmetry and inverse residuals before
attributing any improvement to symmetry. Use each arm's block construction
consistently for RHS, implicit action, explicit reference, preconditioner and
landmark back-substitution. A failed Cholesky factorization is a reported
failure, not permission to fall back silently. The new test helper's general
inverse control must match the existing production operator on small fixtures;
preserve the real-data baseline reports as well. Do not retain a full second
normal system/model or scale physical rig calibration.

The locked `nalgebra 0.33.3` implementation (`src/linalg/inverse.rs`, 3x3
branch, inspected locally) uses explicit cofactors and a determinant for
`try_inverse`. Its formula does not imply asymmetric output for a symmetric
input. Therefore the A/B concerns inverse accuracy as well as symmetry, not
an assumed asymmetric-inverse defect. Record `Hll * inverse - I` residuals,
original and inverse off-diagonal differences, and scale metrics before
interpreting a change in PCG convergence. Cholesky can change rounding even
when both inverse representations are symmetric.

### Cholesky landmark elimination result (2026-09-07)

[Frozen evidence](../benchmarks/electro/m8-openloris-cholesky-landmark-elimination-v1.json)
records two release runs of implementation `05500bf`. All seventeen numerical
report lines repeat exactly; the eleven existing lines also match PR #81.
Original damped blocks and general inverses have zero measured asymmetry at
all three damping values. The proposed asymmetric-inverse explanation is thus
unsupported on this input. Cholesky changes rounding, but is not uniformly
better: at `1e-4`, the maximum block inverse-identity residual increases from
`2.000e-8` to `2.634e-8`.

| Damping | Cap | General true residual | Cholesky true residual | Result |
|---|---:|---:|---:|---|
| `1e-4` | 512 | 318.346 | 5.38483 | both hit cap |
| `1e5` | 512 | `9.079e-4` | `1.215e-5` | general hits cap; Cholesky recheck fails at 322 |
| `1e10` | 128 / 512 | `5.639e-7` | `1.539e-8` | both pass, 22 / 23 iterations |

At useful damping `1e5`, Cholesky recursive residual is `5.936e-7`, but true
residual exceeds its `7.363e-7` target. Its dense reference's own-action
residual improves from `0.002572` to `1.076e-5`, and original pose-normal
equation residual from `0.001232` to `8.359e-6`. The latter is a different
equation norm, not subject automatically to the Schur stopping threshold.
No failed PCG iterate is applied or assigned hypothetical geometry metrics.
At `1e10`, all 130,900 observations remain valid with identical trial cost
126,477.42813448103; cross-method pose/landmark delta differences are only
`2.54e-17` / `1.54e-17`. This does not establish nonlinear quality improvement.

The test-only operator borrows existing cross blocks and uses the same
Cholesky factors consistently in RHS, action, preconditioner and back-substitution.
It audits symmetry before mirroring the lower triangle and never substitutes a
diagonal or falls back. General setup failures still abort this diagnostic
runner; Cholesky setup and PCG failures are reported. The frozen fixture has
no setup failures. Two combined diagnostic processes take 80.44 / 90.64 s and
349,472 / 349,540 KiB peak RSS on a shared host. These include all reference
arms and are not production performance measurements. Cholesky remains
test-only and no README claim is promoted.

The next experiment exposes an example-only, explicitly logged
`--pcg-relative-tolerance` option. Compare production general-inverse PCG512
with relative `1e-8`, absolute `1e-12`, against the unchanged strict `1e-12`
baseline and direct solver on the same full nonlinear 1k input. This is a
separate stopping-policy arm, not Ceres' quadratic-progress `eta`, and not a
retroactive pass of the strict oracle. Keep LM settings, observations and
physical calibration fixed; audit full support, depth, identity, recomputed
reprojection and post-only GT trajectory, plus repeated wall/RSS, before any
10k promotion. Local BA timing alone cannot support a COLMAP end-to-end claim.

### Explicit relative-tolerance nonlinear 1k result (2026-09-07)

[Evidence](../benchmarks/electro/m8-openloris-relative-pcg-tolerance-v1.json)
records implementation `7b6056a`, the same release binary for six serial
processes, and independent audits of the three unique models. The example
accepts `--pcg-relative-tolerance 1e-8` only for matrix-free optimization and
logs actual relative/absolute values both before solving and in its summary.
The initial exploratory summary mislabeled its tolerance; its artifacts are
retained separately, and all comparisons were rerun after the logging fix.
Default summary and numerical behavior remain unchanged. Cholesky is not
used by any of these nonlinear runs.

| Fixed-input 1k BA arm | Final squared cost | Mean reprojection px | GT RMSE / p95 m | Wall s (two runs) | Peak KiB (two runs) |
|---|---:|---:|---:|---|---|
| Direct, serial | 117,496.033 | 0.689260 | 0.026519 / 0.040989 | 45.49 / 45.37 | 161,692 / 161,932 |
| MF512, relative `1e-12` | 126,419.082 | 0.720314 | 0.026703 / 0.041347 | 17.81 / 18.30 | 84,568 / 84,400 |
| MF512, relative `1e-8` | 118,070.554 | 0.691589 | 0.026608 / 0.041100 | 20.19 / 20.29 | 84,564 / 84,564 |

Both PCG arms retain absolute `1e-12` and total cap 512; all arms retain the
same 20 LM iterations and initial cost 126,510.3987573094. Relative `1e-8`
first accepts LM9 at solve damping `1e5`, versus strict LM14 at `1e10`, but
still accepts only three steps (direct five). None reports nonlinear
convergence. Relative `1e-8` improves the listed quality measures over strict,
but does not match direct. Its maximum observation error 4.840089 px also
exceeds input 3.999569 px and strict 3.997840 px (direct 4.937791 px). Do not
describe this as universal reprojection non-regression.

All models retain 1,000 supported images, 500 supported rig frames, 4,716
landmarks, 130,900 observations and 361,170 full keypoints. Camera bytes,
image identity/order/POINTS2D and point IDs/RGB/track order remain exact;
fixed calibration, one connected frame component and positive depths pass.
GT uses the same 308 associated images, only after optimization. The new
model's stored ERROR differs from recomputed track means by at most
`5.91e-9` px. Its maximum centre/landmark differences from direct remain
0.001098 / 0.019274 m, not numerical equivalence.

Each repeated arm has identical model hashes and numerical traces. New
direct and strict controls also match PR #79 model bytes and traces; input
hashes are unchanged. These warm-cache shared-host times include loading,
validation, solve and publication, not frontend/mapping/atlas construction.
The local time and memory advantage is measured at different final quality;
it is not an equivalent-quality speedup or a COLMAP end-to-end result. The
frozen COLMAP 1k RMSE/p95 is 0.027969 / 0.042266 m, but being below that local
reference does not establish the outstanding 10k gate. README is unchanged.

#### Next bounded candidate: true-residual restart

The new arm still has ten `MaxIterations` and seven `ResidualCheckFailed`
attempts. For example, LM12 has recursive residual `0.001371` but true
residual `0.116403`; LM13 has `0.001045` versus `0.103147`. The existing
classical PCG immediately rejects after such a failed final check. A
separately disabled-by-default candidate can reuse that true residual,
restart with `r=b-Ax`, `z=M^-1 r`, `rho=r^T z`, `p=z`, and continue only
within the original total 512-iteration budget. Start with at most one
restart; never reset the iteration count, accept an unchecked iterate,
change physical damping, add dense fallback, or retain an iteration history.
Reuse O(N) vectors and measure actual allocations/RSS. Preserve existing
public struct shapes and default behavior.

[Greenbaum's analysis](https://epubs.siam.org/doi/10.1137/S0895479895284944)
supports the finite-precision residual-gap diagnosis.
[Van der Vorst and Ye](https://epubs.siam.org/doi/10.1137/S1064827599353865)
study error-bound-guided replacement that limits perturbation of Krylov
recurrences. The proposed single event-triggered restart is an engineering
hypothesis, not a reproduction or guarantee of their scheme. Restart drops
accumulated conjugacy and may slow or stagnate; it does not directly address
the ten iteration-limit failures. [PETSc's manual](https://petsc.org/release/manual/ksp/)
also distinguishes estimated residual monitoring from explicit `b-Ax`
monitoring and warns that the latter adds work.

First test disabled-path exactness, fixed total iteration/restart bounds,
finite/curvature failures, zero RHS and a controlled residual-gap fixture.
Then compare frozen real systems and full nonlinear 1k quality/resources
with explicit restart counters. A reduction in recheck failures alone is
not sufficient for promotion; retain direct/strict controls and all geometry,
identity and GT gates. No restart implementation is included in this result.

### Actual atlas driver-boundary pilot (2026-09-07)

[Evidence](../benchmarks/electro/m8-openloris-matrix-free-atlas-pilot-v1.json)
records a PR #82 binary trial on the retained filtered main component under
`RLIMIT_AS=2 GiB`, CPU 900 s and core dumps disabled. Its derived rig manifest
keeps original sensor/header lines and selects F records by exact source image
name, without relabeling IDs or changing calibration. The model contains
8,988 poses, 318,222 points, 1,313,899 observations and 4,672,106 full keypoints.

The process rejects during source preflight: image 8987 (`cam1_008986.png`,
sensor 0, frame 4493) has no landmark support. This is the known retained
source state, not a newly lost image. Sensor 1 of the same calibrated frame,
image 8988, has ten observations, and all 4,494 rig frames are supported.
No BA starts and no model is published. The 0.94 s / 407,032 KiB process
measurement is parsing/preflight only, not a solver memory or 10k pass.
Virtual-address-space limiting is stricter than an RSS limit and cannot turn
an aborted solve into a resource success.

Before full atlas comparison, the driver needs a separately explicit mode
that retains observation-free sensor images only when their shared rig frame
is supported by another sensor. Preserve the original supported and unsupported
image sets; still reject unsupported entire frames and disconnected frame
graphs. Do not delete a sensor row to bypass validation. The tail has 505
frames with IDs 4495..4999, so also expose an explicit existing supported
fixed-frame ID (default 0 unchanged), anchoring the tail at 4495 without
renumbering. This is a driver boundary extension, not a change to physical
calibration, observation selection, PCG or LM acceptance.

Keep the two original components and score them together with the existing
post-only alignment. Direct-BA numerical agreement remains a diagnostic;
the ultimate advancement gate is measured COLMAP quality and resources,
not perfect agreement with one internal backend. The unsupported-image
boundary and anchor extension are not yet implemented in this pilot.

An independent streaming point/track scan also finds main-component positive
depths from `1.8218e-6` to 14,900.75 m (tail 0.01513 to 39.07486 m). This is
a conditioning clue, not a measured Hessian condition number or permission
to remove points. Main/tail have 281,079 / 30,995 tracks with same-frame
cross-sensor observations; metric observations are present, without proving
full rank. Excluding the selected anchor's observations gives 1,313,303 /
124,955 cross entries and sums of squared per-point cross counts 40,517,191 /
3,124,173. These are work/storage inventory, not a dense allocation plan.

### Bounded PCG residual restart: 1k result (2026-09-08)

[Nine-run evidence](../benchmarks/electro/m8-openloris-bounded-pcg-restart-v1.json)
uses implementation `ca67e80`, explicit restart limit 0/1 and the unchanged
512-total-iteration budget. True residual failure can restart once, but never
extends the budget or bypasses the nonlinear acceptance/depth checks. The
default remains restart-off; existing API shapes and the LM loop are retained.

At relative tolerance 1e-8, restart increases successful linear solves from
3 to 5 and accepted LM steps from 3 to 4. Seven restarts produce cost
118070.232073 versus 118070.554369 without restart, and GT RMSE
0.02660802754 versus 0.02660805517 m: the trajectory improvement is negligible.
The recovered linear solve at LM8 is still rejected by LM; the trace alone
does not establish the exact rejection reason. Maximum individual reprojection
also rises slightly, from 4.840089 to 4.840263 px. Direct remains better in
cost and trajectory RMSE, so this is not an equivalent-quality speed result.

Restart-on repeats have identical model files and all 60 numerical/diagnostic
trace rows. Legacy relative/direct repeats are also exact; explicit restart 0
matches the legacy model and LM/PCG trace. Under strict 1e-12 tolerance,
seven restarts leave the final model byte-identical to strict restart-off.
All 1,000 supported images, 500 supported frames, 4,716 points, 130,900
observations and 361,170 full keypoints are retained, with one connected
frame graph, fixed calibration and no nonpositive depths.

Restart-on wall times are 24.00/20.82 s, relative-off 23.71/20.21 s and
direct 56.01/44.67 s on the shared host; these are local single-thread BA
processes, not mapper/native end-to-end measurements. This implementation
does not promote the 10k or README performance claims. The next experiment
is the actual two-component atlas with the explicit boundary/anchor policy,
not further tuning solely to reproduce the internal direct backend.

### Actual two-component matrix-free BA (2026-09-08)

[Six-run evidence and independent audits](../benchmarks/electro/m8-openloris-matrix-free-atlas-policy-v1.json).
Driver `c2eeb70` preserves the original main/tail anchors 0/4495 and retains
the known unsupported main sensor image only through explicit shared-rig
support policy. The 17 example tests and a fresh default-policy 1k run pass;
the latter's three model files and 40 LM/PCG rows match the previous driver.
Both actual components now complete under the 2 GiB virtual-address-space
cap, CPU 900 s cap, disabled core dumps and one Rayon thread.

| Same 10k evaluation | Registered images | Pooled GT RMSE / p95 (m) | Observation-weighted reprojection (px) |
|---|---:|---:|---:|
| Frozen COLMAP | 9,998 | 0.384307 / 0.638669 | 0.903003 |
| Retained filtered atlas input | 9,998 | 0.388993 / 0.638173 | 0.581744 |
| Matrix-free post-map BA | 9,998 | 0.388720 / 0.638174 | 0.579509 |

GT is used only for post-mapping scoring with one Sim(3) per original
component and pooled errors from 9,306 scored images. All 9,997 supported
images, 4,999 supported frames, 352,837 points and 1,438,880 observations
remain. Full keypoint, point/track identity, calibration and original fixed
anchors pass independent checks; no nonpositive depths appear. The tail's
maximum individual reprojection rises to 4.097846 px and maximum track mean
to 2.239294 px, so this is not universal per-observation nonregression.

Main restart-off takes 88.75 s and 1,079,880 KiB peak RSS; tail takes
41.40 s and 109,036 KiB. These are separate serial local BA processes, not
the full mapper or native pipeline, and cannot be compared to COLMAP's
mapper time as a speedup. Main accepts only 1 of 18 LM trials and stops at
the existing maximum-damping boundary; tail accepts 8 of 20. Main's first
nine trials reject `NonSpdPreconditioner(191)`, followed by two curvature
failures. The number 191 is a variable-pose slot, not a source frame ID.

Enabling one bounded restart performs four restarts on main and zero on tail.
Both final models are byte-identical to restart-off: no 10k quality benefit
from restart. Restart-on repeats match all six model files and both components'
LM/PCG/restart traces exactly. Main repeats take 93.85/93.64 s and tail
41.81/41.12 s; main peak RSS is 1,079,764/1,079,724 KiB. The slightly
improved atlas still fails COLMAP's RMSE gate.
Keep the production defaults and README claims unchanged. Next isolate the
failed main preconditioner block using bounded diagnostics (slot/frame mapping,
actual solve damping, 6x6 block spectrum and local subtraction norms), without
dropping observations, adding a dense global matrix or claiming a cause from
support/depth range alone.

### Local Schur-block diagnostic contract (2026-09-08)

PR #83 merged as `a71f40a` after eight final-head CI checks and independent
artifact review. Continue on `feat/m8-local-schur-block-diagnostic` with an
explicit, default-off diagnostic of variable-pose slot 191 (source frame 192).
Keep the actual solver arithmetic and acceptance unchanged. Record only this
6x6 block, its damped pose diagonal, local elimination norms, finite/asymmetry
checks and the largest contributing landmark's 3x3 block/inverse and 6x3 cross
block. No full normal-system clone, global dense matrix or history bank.
Label actual solve damping and the chosen triangular interpretation used for
spectral checks; never repair a block in the diagnostic path.

Independent source-image geometry inspection finds 417 incident tracks and
614 observations at frame 192. Point 13921 has two observations (image 386,
keypoint 80; image 390, keypoint 89), with target depth about 1.82e-6 m versus
0.431 m for the next closest point. Its two camera centers are 0.319 mm apart.
These source-image calculations are not bit-identical to production rig-normal
assembly; the actual block dump must establish the numerical effect. Do not
remove the point solely because it is near a camera.

Choose the next opt-in numerical experiment from the measured failure, not
from another tolerance sweep. Distinguish preconditioning, damping and coordinate
scaling: Ceres documents camera-block `JACOBI` separately from `SCHUR_JACOBI`,
uses a diagonal metric for its LM trust region, and supports Jacobian-column
scaling. These are different mechanisms, not interchangeable settings.
[Ceres solver reference](https://ceres-solver.readthedocs.io/latest/nnls_solving.html).
Changing the damping metric would be a separately labeled behavioral policy,
not an exact-output coordinate rewrite. Any candidate must earn 1k/actual-atlas
quality, true-residual, resource and repeatability gates; the user's final
COLMAP mapper/native-E2E and scale requirements stay unchanged.

### Local Schur-block measurements (2026-09-08)

[Measured evidence](../benchmarks/electro/m8-openloris-local-schur-block-diagnostic-v1.json)
uses `bd3291a`. Two diagnostic main-component runs have identical 18-line
block dumps, final model files and LM/PCG/restart traces. The main OFF control
and the fresh 1k OFF control also match PR #83's corresponding outputs and
numerical traces. Main ON wall times are 74.11/73.77 s, OFF 74.50 s; peak RSS
is 1,080,176/1,080,324/1,080,000 KiB. Shared-host timings are not evidence of
a speed improvement from diagnostics. All runs retain the previous model.

Point 13921 is the largest local elimination contributor at every solve
damping. At initial damping, the damped pose block norm is 4.8575e16 and its
Schur block norm 1.4762e12. The block's asymmetry is 4.01e6, large compared
with its weakest eigenvalues despite being small compared with its norm.

| Actual solve damping | Selected block lower-triangle minimum eigenvalue | Full solve result |
|---:|---:|---|
| 1e-4 | -1.037e7 | Non-SPD preconditioner |
| 1e4 | -7.868e5 | Non-SPD preconditioner |
| 1e5 | 9.302e6 | Nonpositive PCG curvature |
| 1e7 | 1.723e7 | PCG iteration cap |
| 1e11 (LM15) | 1.000e11 | Only accepted LM update |

An independent 80-digit Decimal calculation on the round-trip binary64 dump
confirms a negative lower-triangle LDL pivot at the initial damping. Correcting
only the largest contributor's stored inverse coefficient error gives a local
correction proxy of norm 3.34495e7 and positive proxy pivots there. However,
the proxy remains indefinite at damping 1, 100 and 1000. This is not a full
high-precision Schur reconstruction: assembly and product/subtraction rounding
remain. Neither inverse replacement alone nor selected-block SPD is sufficient
evidence that the global solver will converge. Raw coefficients and model
coordinates remain in the external audit artifact, with its digest recorded.

The next candidate is explicit column equilibration plus identity LM damping
in scaled coordinates: choose positive, bounded `d_j` from the original
normal diagonal, let `T = diag(1/sqrt(d_j))`, solve
`(T H T + lambda I) delta_hat = -T b`, and return `delta = T delta_hat`.
This changes physical damping to `lambda diag(d)`, so it is a behavioral
policy, not an exact-output rewrite of the old `lambda I` solve. Transform
pose, point, cross and RHS blocks consistently in the iteration-local normal
system; retain only linear-size scale vectors and restore physical step units
before LM acceptance/convergence checks. Specify finite clamping bounds and
label scaled PCG residual units. Small full-normal tests precede fixed-budget
1k and both-component atlas A/B; observations, calibration, depth gates and
the final COLMAP/mapper/E2E requirements are unchanged. This policy is not
implemented or promoted by the diagnostic result.

### Column-scaled LM implementation and measurement contract (2026-09-08)

PR #84 merged as `c49e542` after all eight final-head checks on `fe4a710`
(run `34140417320`). Its evidence JSON preserves the pre-CI snapshot; this
closure records the later CI result. Implementation now proceeds on
`feat/m8-column-scaled-lm`; no scaled-policy result is available yet.

Use private fixed bounds `d_j = clamp(H_jj, 1e-6, 1e32)` after fixed-rotation
constraints and before damping. Reject nonfinite or negative original
diagonals; a zero diagonal uses the lower bound. Transform every pose, point,
cross and RHS block consistently in-place. Fixed rotation unit rows have
scale one, and fixed poses/points remain absent from the variable layout.
Keep only iteration-local O(P+L) scale vectors and scalar diagnostic history,
with no additional normal-system or model clone. Check transformed values
and unscaled deltas for finiteness before applying any update. Failed solves
discard the local system and rebuild it on the next LM iteration.

The additive API and explicit `--matrix-free-column-scaling` flag must leave
legacy options, results, arithmetic and logs unchanged when absent. Reject
direct-solver, standalone oracle-export and nonzero PCG-restart combinations.
Label the new residual and target values as scaled coordinates, including
any enabled local Schur diagnostic. Existing LM step norms and acceptance
checks use physical deltas after unscaling. This policy changes physical
damping to `lambda diag(d)`; it is not an exact-output transformation or a
complete reproduction of Ceres.

First compare against a small full-normal direct solution of
`(H + lambda diag(d)) delta = -b`, including calibrated nonzero rig baselines,
fixed variables, same-pose multi-sensor cross terms, clamp boundaries and
failure rollback. Then run serial, same-binary frozen 1k legacy matrix-free,
direct and scaled controls. Fix 20 LM iterations, initial lambda 1e-4 and
the existing lambda schedule; matrix-free arms use 512 PCG iterations,
relative tolerance 1e-8, absolute tolerance 1e-12 and zero restarts.

Next run each original retained atlas component twice with the same policy,
under 2 GiB address-space, 900-second CPU and zero core-dump limits, with
`RAYON_NUM_THREADS=1`. Keep source frame anchors 0 and 4495 respectively;
retain the explicitly allowed unsupported main-component sensor image.
Do not use already optimized outputs as the new input or delete near points.
Audit all image/keypoint/point/track identities, calibration, support and
connectivity, finite positive depths, reprojection, post-only GT scores,
RSS, wall time and model/numerical-trace repeatability. Source and binary
hashes must accompany measurements. Do not tune bounds using GT.

COLMAP's 10k RMSE gate remains 0.3843065335 m; the previous unscaled policy
scores 0.3887199838 m. A local BA improvement alone does not establish
mapper-only/native-E2E speed or the remaining tier, restart and 100k I/O
gates. Keep README comparison claims unchanged until their scope is verified.

### Initial column-scaled 1k result (2026-09-08)

The opt-in policy is implemented in `99d899b`. Final release binary SHA256 is
`a8d7330cf4233081c50c4e5b0e8df26735dd3e713dabaae33681f026e9cd80ef`.
Same-binary legacy and direct controls preserve their previous model files
and numerical traces. Two scaled runs preserve all image/keypoint/point/track
identities, fixed anchor and calibration, with no nonpositive depths; their
three model files and LM/PCG/scaling traces match exactly.

| 1k local BA arm | Wall seconds | Peak RSS KiB | Final squared cost | Accepted LM steps |
|---|---:|---:|---:|---:|
| Legacy matrix-free | 24.21 | 84,656 | 118,070.554369 | 3/20 |
| Legacy direct | 55.76 | 161,776 | 117,496.033074 | 5/20 |
| Column-scaled, run 1 | 28.39 | 84,592 | 113,754.040510 | 10/20 |
| Column-scaled, run 2 | 28.33 | 84,764 | 113,754.040510 | 10/20 |

Scaled PCG succeeds in all 20 solves, but this does not imply better trajectory
accuracy. Post-only GT RMSE/p95 is **0.028550/0.043791 m**, worse than legacy
matrix-free 0.026608/0.041100 m and COLMAP 0.027969/0.042266 m. Mean
observation reprojection improves to 0.675819 px, while maximum reprojection
is 5.098235 px and maximum track mean is 3.170287 px. All 1,000 supported
images, 500 supported frames, 4,716 points and 130,900 observations remain.

The policy therefore fails the 1k quality gate and is **not promoted**. The
fixed-policy actual-atlas runs remain diagnostic measurements, not promotion
or permission to tune against GT. They test whether the main component's
numerical failure and the 2 GiB resource constraint are addressed. Shared-host
local-stage times do not establish mapper-only or native-E2E speed.

### Initial actual-atlas column-scaled result (2026-09-08)

The same final binary completes both original components under the fixed
2 GiB address-space cap. Main wall time is 342.46 s with peak RSS 1,090,964
KiB; tail is 37.69 s / 110,512 KiB. Main accepts 10/20 LM steps, with 10
PCG successes and 10 iteration-cap failures, and no non-SPD-preconditioner,
curvature or true-residual-check failures. Tail accepts 11/20, with seven
iteration-cap and two true-residual-check failures. Neither arm reports LM
convergence. At every main iteration the original diagonal is inside the
fixed bounds, so neither clamp is used.

Independent audits preserve all 9,998 images, 5,085,072 full keypoints,
352,837 points and 1,438,880 observations, all original identities, camera
calibration and anchors 0/4495. All 9,997 supported images and 4,999 supported
frames remain in the original 4,494/505-frame components, with zero
nonpositive-depth observations. The mean observation reprojection is
0.562992 px; main/tail maximum errors are 4.427287/4.091211 px and maximum
track means are 2.395166/2.204463 px. Report these tails separately from the
improved mean; the original input's filtering limits are not a newly invented
universal post-BA gate.

Post-only pooled RMSE/p95 is **0.387518/0.635967 m**, improved from the
previous unscaled 0.388720/0.638174 m, but still above COLMAP's RMSE
0.384307 m. Main and tail component RMSE are 0.409907 and 0.059373 m;
the tail regresses from 0.058738 m. Combined with the 1k regression, this
does not qualify for default or mapper promotion. Repeated atlas runs and
diagnostic-output compatibility checks are still in progress at this checkpoint.

The [completed measurements](../benchmarks/electro/m8-openloris-column-scaled-lm-v1.json)
also include the second runs: main 343.31 s / 1,090,912 KiB and tail 38.36 s /
110,488 KiB. Both have byte-exact model files and LM/PCG/scaling traces versus
their first run. A same-binary legacy main diagnostic control (88.04 s /
1,081,380 KiB) preserves PR #84's model, numerical/restart traces and all 18
raw diagnostic lines. The scaled 1k debug control (28.17 s / 84,636 KiB)
preserves its OFF model/numerical/scaling traces and labels all 20 local dumps
as scaled coefficients. Debug timings are separate from OFF-arm comparisons.
The three pre-final pilot runs are retained externally and explicitly excluded
from final-binary comparisons because per-launch binary identity was not
independently certified during the rebuild. Raw coefficients remain external.

The main trace alternates failed solves at actual lambda 1e-4 (512 PCG
iterations) with accepted solves at 1e-3 (237–272 iterations). Rejected LM
rows report the post-increase lambda, so their printed 1e-3 must not be
mistaken for actual solve damping. This repeated work is a measured remaining
cost, not justification for a GT-selected damping sweep.

Before selecting another policy, a bounded diagnostic can record actual
solve damping, each policy's normalized physical-equation residual/backward
error, and undamped predicted versus actual objective decrease. Raw norms
in scaled and physical coordinates are not directly comparable. Ceres uses
step quality to adapt the trust region and supports inexact iterative LM;
its implementation supplies a quadratic-progress tolerance rather than the
same fixed residual tolerance used here. See the
[Ceres solving reference](https://ceres-solver.readthedocs.io/latest/nnls_solving.html)
and [LM strategy source](https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/levenberg_marquardt_strategy.cc)
(reviewed 2026-09-08). These motivate a diagnostic, not a claim that adaptive
LM or looser solves will cure the observed 1k trajectory regression. Keep
GT post-only, all observations, bounded memory and defaults unchanged.

### Step-quality diagnostic design checkpoint (2026-09-08)

PR #85 merged as `d751f6e` after eight final-head checks on `11f36b8`
(run `34145940542`); its evidence file is the earlier pre-CI snapshot.
Continue on `feat/m8-lm-step-quality-diagnostic`. This adds diagnostics only,
not another tolerance, damping or acceptance policy.

For the squared-error convention, the undamped predicted decrease is
`-2 b^T delta - delta^T H delta`; compare it with the actual cost decrease
only for a finite, successfully computed candidate. Keep actual solve lambda
distinct from the post-rejection lambda printed by legacy LM traces. Failed
linear solves have no candidate decrease or fabricated residual.

The scaled normal is already transformed in-place. Any residual mapped back
from it describes the physical-equivalent system represented by those rounded
scaled coefficients, not an independently preserved original normal. Label
that distinction. Normalize residuals rather than comparing raw norms across
coordinate systems. For a componentwise backward-error diagnostic, repeated
cross contributions to the same pose/landmark coefficient must be summed
before taking absolute values; summing absolute contributions would produce
a different denominator. Handle zero denominators and nonfinite arithmetic
explicitly without modifying the solve's result.

Keep only bounded scalar records, linear-size scratch and at most one
landmark's cross aggregation; no extra model/normal clone, global dense
matrix, coefficient dump or per-point history. Existing APIs, default logs,
numerical steps and LM/depth/rollback gates remain unchanged. Small explicit
full-normal tests and debug ON/OFF model/trace comparisons must precede using
the diagnostic to select a new policy. The 1k regression, unmet 10k COLMAP
RMSE and full mapper/E2E/scale requirements remain unresolved.

Further read-only inspection of PR #85's certified scaled 1k debug log finds
nine of ten rejected candidates lower the reported cost but increase
nonprojectable observations from zero to 209, 211 or 14. Their actual solve
lambda is 1e-5. The other rejection (iteration 6) increases cost without
increasing nonprojectable observations. The existing cost routine omits
nonprojectable observations, so the nine decreases are not comparable
same-observation objective improvements. New diagnostics must report both
counts and both acceptance gates; define rho only when both counts are zero,
cost values are finite and predicted decrease is positive. Otherwise record
an explicit undefined reason without changing the existing LM decisions.

The implementation should fold each landmark's three residual/denominator
entries immediately after its cross terms, retaining only pose accumulators
and one landmark's aggregation. This reduces new diagnostic scratch to
O(P + maximum track length), rather than storing another residual/denominator
pair for every landmark. Use nonquadratic cross aggregation and preserve each
coefficient's contribution order. If physical-equivalent metrics are evaluated
by reconstructing coefficients and recomputing the residual, label that extra
rounding explicitly; it is not bit-identical to simply scaling the already
computed residual by `T^-1`.

### Step-quality 1k measurement checkpoint (2026-09-08)

The diagnostic implementation is `97d9fb0`; the measured release binary was
built at `9016b8f` (scratch-bound comment correction), SHA256
`fbc7445ac295702f6c98571420bef601bc9532735b52de01d20e86af8bb949f5`.
Subsequent `d998fc0` changes only tests: perturbed dense solutions compare
the same damped system in scaled and physical-equivalent coordinates, with
strictly nonzero, nonsaturated componentwise eta. The legacy `H+lambda I`
must not be mistaken for the physical equivalent of scaled `Hhat+lambda I`.

Legacy and scaled 1k ON/OFF controls preserve every model file and existing
LM/PCG/scaling trace from PR #85. A second scaled ON run also preserves all
20 quality rows. Scaled candidate componentwise eta is 8.62e-11–1.33e-9,
including the ten rejected candidates. Nine have nonprojectable observations
and undefined rho; the sole comparable cost-increase rejection has
rho -4.18755. Two accepted steps have low rho (0.144615 and 0.286570), while
the existing policy still reduces lambda by ten after each acceptance.

These observations separate nonlinear feasibility/model mismatch from the
measured linear-equation error. They do not establish a forward-error bound
for this ill-conditioned problem or explain all trajectory regression.
Consider a separately opt-in, predeclared step-quality-based damping policy
after completing actual-atlas diagnostics. Keep observations, PCG budget and
tolerance, calibration and GT post-only scoring fixed for that first A/B;
do not combine it with another tolerance sweep or claim COLMAP parity.

### Completed diagnostic measurements (2026-09-08)

All seven certified-binary runs are recorded in the
[step-quality evidence](../benchmarks/electro/m8-openloris-lm-step-quality-v1.json).
Main completes in 346.91 s / 1,090,824 KiB; tail in 38.62 s / 110,400 KiB,
both under a 2 GiB address-space cap. These are diagnostic-ON post-map BA
times, not mapper/native-E2E performance claims. Every output model file and
existing numerical trace is byte-exact versus PR #85. Its independent
geometry/identity/GT audits therefore carry over; no fresh GT scoring is
claimed for unchanged bytes. The scaled 1k ON repeat also matches every new
quality row and all ten existing rejected-step detail rows. Actual-atlas ON
was measured once per component, not repeated in this diagnostic experiment.

Main has ten capped solves at actual lambda 1e-4, alternating with ten
accepted candidates at 1e-3. Their rho is 0.984477–1.000015 and componentwise
eta is 3.83e-10–1.81e-9. Tail has eleven accepted candidates, seven capped
solves and two true-residual-check failures; accepted rho is 0.909699–1.000608.
No diagnostic arithmetic failure occurred. All atlas candidates that reached
nonlinear evaluation had zero nonprojectable observations before and after.
The scaled 1k trajectory regression and pooled 10k COLMAP RMSE gap remain:
the diagnostic does not alter either result or promote this solver policy.

### Predeclared adaptive-damping A/B contract (2026-09-08)

PR #86 merged as `d339093` after eight final-head checks on `b5ffa66`
(run `34149951882`). Continue on `feat/m8-adaptive-scaled-lm-damping`.
This is a production-capable opt-in policy experiment, not a default change.

Change only the accepted-step multiplier to
`u(rho) = max(1/3, 1 - (2*rho - 1)^3)` and
`lambda_next = clamp(lambda_solve * u(rho), min_lambda, max_lambda)`.
For finite `rho >= 1`, evaluate the saturated factor directly to avoid cubic
overflow. Keep the existing rejection/linear-failure increase factor (10 in
the experiment). The accepted-step formula follows the
[Ceres LM strategy source](https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/levenberg_marquardt_strategy.cc)
(reviewed 2026-09-08), but Ceres also adapts its rejection multiplier and uses
different linear-solve stopping rules. This isolated policy is not a Ceres
implementation or a claim of equivalent behavior.

Use the current scaled system's undamped squared-cost prediction and the
same zero-nonprojectable rho contract as PR #86. The new opt-in entry must
reject initially nonprojectable input before mutation. A finite candidate
with invalid/nonpositive prediction or unavailable rho is an explicit
candidate rejection with rollback, not a fabricated linear failure. Existing
cost/feasibility gates remain necessary. Require strictly positive finite rho
for acceptance: division of a positive finite decrease by a large prediction
can underflow to zero and must reject, not reach an accepted-update assertion.
For linear failures, uncomputed candidate cost/feasibility gates and the
post-candidate nonprojectable count are unavailable, not fabricated values.
Log actual solve lambda separately
from the next lambda; retain finite bounds and bounded termination.

Compute only the prediction scalar from borrowed blocks and deltas, after
back-substitution and before physical unscaling. Do not enable full residual/
backward-error scans merely to use adaptive damping. No additional model or
normal clone, dense global matrix, full residual scratch or per-point history.
Keep any same-pose cross aggregation bounded by one track and preserve its
coefficient contribution order. Old APIs, struct shapes, defaults and logs
remain unchanged; new API/results and CLI selection are additive. The CLI
requires explicit column scaling and rejects direct/oracle/export/nonzero
restart combinations.

The frozen 1k comparison uses PCG 512 iterations, relative tolerance 1e-8,
absolute tolerance 1e-12, restart zero, initial lambda 1e-4, bounds 1e-9–1e12,
20 LM attempts and one Rayon thread. Keep every input observation and fixed
rig calibration/anchor. Freeze source and binary hashes before timed runs,
with no concurrent local compilation. Legacy MF and direct controls must
match their earlier outputs; fixed-scaled and adaptive arms each run twice.
An adaptive debug-ON control must preserve OFF models and numerical traces.

The fixed-scaled arm isolates the policy change but is **not** a passing
quality reference. Before any adaptive atlas run, require 1k RMSE no worse
than legacy MF's 0.026608055174816774 m and p95 no worse than
0.04109998478546261 m; report direct and COLMAP alongside it. Require identical
image/keypoint/track identities and counts, all 1,000 supported images and 500
supported rig frames in one component, fixed cameras/rig/anchor, finite state,
zero nonpositive depths, and mean observation reprojection no worse than
legacy MF's 0.691588658326868 px. Report maxima and point movement separately;
the input filter thresholds are not invented post-BA maximum-error gates.
Also require same-observation final objective no worse than legacy MF and
repeatability of outputs/traces. GT stays post-only: no damping/tolerance
sweep selected on trajectory score. Report timing and RSS without claiming
mapper/native-E2E acceleration. Failure of this 1k gate stops atlas progression
for this candidate and requires a new evidence-based decision, not a relaxed
threshold or README promotion.

### Adaptive-damping result: 1k gate failed (2026-09-08)

The [seven-run evidence](../benchmarks/electro/m8-openloris-adaptive-scaled-lm-v1.json)
uses the `005dbcc` production build, SHA256
`58dfc58ff1ce821f92547b00dc2627efa6a82978d1dad85444313858972ec5b6`.
Later `75df6e5` and `7e25dee` only strengthen tests; no release rebuild occurred.
The input, PCG policy, initial damping, 20-attempt budget and single thread
were held fixed. All runs completed serially without overlapping compilation.

| 1k arm | Wall s | Peak RSS KiB | Accepted / attempts | RMSE / p95 m | Mean observation error px |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy MF | 24.11 | 84,424 | 3 / 20 | 0.026608 / 0.041100 | 0.691589 |
| Direct | 55.57 | 161,916 | 5 / 20 | 0.026519 / 0.040989 | 0.689260 |
| Fixed-scaled | 27.93 / 27.67 | 84,628 / 84,840 | 10 / 20 | 0.028550 / 0.043791 | 0.675819 |
| Adaptive-scaled | 25.83 / 25.54 | 84,860 / 84,852 | 15 / 20 | 0.029190 / 0.044521 | 0.675153 |
| Existing COLMAP mapper reference | Not a same-input BA timing | — | — | 0.027969 / 0.042266 | 0.810207 |

Adaptive debug ON takes 26.97 s / 84,864 KiB. Its three model files and all
LM/PCG/scaling/adaptive traces match OFF and the OFF repeat exactly. Legacy,
direct and fixed-scaled models and prior numerical traces match PR #85.
The reference quality numbers carry over those exact bytes; adaptive GT,
geometry and full identity checks were independently recomputed. Times here
include model loading/export but exclude frontend, mapping and atlas creation.
They are not mapper/native-E2E claims or a speed comparison with COLMAP.

All 20 adaptive PCG solves succeed. Five candidates (iterations 2, 8, 10, 13,
16) increase nonprojectable counts and are rejected with unavailable rho;
the remaining 15 pass both gates. Prediction-only scalars agree with the
separate full-quality diagnostic, and all accepted/rejected damping factors
and actual-to-next lambda chains are independently verified. The raw CLI
uses `unknown` for absent optional values; audit parser v2 normalizes this to
JSON null and decodes quoted reasons without changing any solver output.

Final cost is 113550.2193390116, but the trajectory gate fails both RMSE and
p95. All 1,000 supported images, 500 supported rig frames in one component,
4,716 points, 130,900 observations and 361,170 keypoints remain. Camera bytes,
full image/keypoint/track identity and order are preserved; fixed sensor
extrinsics and anchor frame 0 pass the serialized geometry tolerance, and no
nonpositive depth remains. Maximum observation error is 5.101116 px and
maximum landmark movement is 230.771019 m; these are reported independently,
not hidden by mean error or reinterpreted as invented post-BA filter gates.

**Do not run adaptive atlas or promote the default/README.** Fewer rejected
steps and lower reprojection cost do not establish better trajectory accuracy.
The result motivates distinguishing the observation objective and geometry
from the optimization path before another solver policy is proposed. It does
not prove the cause of the trajectory regression or that robust loss,
calibration refinement or point removal would fix it. GT stays post-only.

### Frozen Ceres reference contract (2026-09-08)

The existing Rust-exported `VISLOC_BA_ORACLE_FIXTURE 1` contains the exact
camera/rig transforms, fixed pose and complete ordered observation set needed
by this diagnostic. Reuse the frozen 1k fixture, SHA256
`a71ef3401ea6d75a06f18ded0d475ad48fd929202a64fecaeaa790ee964da1ea`,
instead of introducing a COLMAP rig/frame reconstruction conversion first.
This is a **standalone Ceres reference**, not COLMAP native BA or mapper
performance. The existing mapper reference remains unchanged.

Before the one fixed-configuration solve, independently verify all 500 poses,
4,716 XYZ points, 130,900 observations, two PINHOLE cameras, fixed frame 0 and
sensor extrinsics against the original model/manifest, including identities
and ordering. No pixel offset is added. Emit every initial residual and depth
and compare against an independent projection: per residual coordinate
`abs(error) <= 1e-6 px + 1e-12 * abs(reference residual)`; depth tolerance
`1e-9 m + 1e-12 * abs(reference depth)`; initial squared-cost tolerance
`1e-6 + 1e-10 * abs(reference cost)`. These are arithmetic-parity tolerances,
not relaxed trajectory gates. A mismatch invalidates the reference run.

Use Ceres 2.2.0, AutoDiff PINHOLE rig factors, wxyz QuaternionManifold plus
translation blocks, all XYZ variable, fixed camera/sensor transforms and
all anchor pose coordinates constant. The reference may pack wxyz quaternion
and translation into one seven-scalar block with
`ProductManifold<QuaternionManifold, EuclideanManifold<3>>`; group 1 then
contains these pose blocks and group 0 the three-scalar XYZ blocks. This
packing does not free either part of the anchor. No robust loss or filtering.
Nonfinite/behind-camera candidates fail residual evaluation instead of
dropping observations. Ceres minimizes one-half the squared cost; report the
common full squared cost separately. The [Ceres tutorial](https://ceres-solver.readthedocs.io/latest/nnls_tutorial.html)
and versioned [manifold](https://github.com/ceres-solver/ceres-solver/blob/2.2.0/include/ceres/manifold.h)
and [solver options](https://github.com/ceres-solver/ceres-solver/blob/2.2.0/include/ceres/solver.h)
define the reference conventions.

Fix one thread, LM, initial trust-region radius 1e4, SPARSE_SCHUR with points
in elimination group 0 and pose blocks in group 1, and max_num_iterations 20.
Record all other versioned Ceres stopping/damping defaults and actual
iterations; do not call them equivalent to visloc's PCG/LM stopping policy.
The installed 2.2.0 header gives function/gradient/parameter tolerances
1e-6/1e-10/1e-8, min/max radius 1e-32/1e16, min relative decrease 1e-3,
LM diagonal bounds 1e-6/1e32, at most five consecutive invalid steps, Jacobi
scaling on and nonmonotonic/inner iterations off. These defaults are not
COLMAP's overridden BA settings.
No GT-selected lambda/tolerance sweep. Keep the diagnostic under a dedicated
2 GiB container memory/swap limit, one CPU and a 900 s solve timeout; no
explicit dense global normal, and no enlargement of the fixture to atlas.
Development dependencies stay in this disposable, task-specific container.

Publish only a validated model preserving all original image/keypoint/track
tokens and camera bytes; independently audit fixed calibration/anchor,
finite positive-depth state, full support and connectivity before scoring.
Report identity failures or solver termination honestly, never repair the
output silently. A single nonconvex reference is evidence about this
objective, not proof of the cause or of general COLMAP parity. The prior
adaptive 1k failure still prohibits adaptive atlas/default promotion.

#### Publication and derivative acceptance gates

Initial residual agreement alone does not validate optimization. Before the
real solve, test ambient AutoDiff derivatives against independent central
differences, and tangent derivatives against finite differences through the
actual ProductManifold `Plus` operation. Use nonidentity rig and sensor
rotations, a nonzero baseline, variable poses and XYZ, and positive depths
away from the rejection boundary. Also run a synthetic Ceres Problem with
the same factor/block/manifold setup and check cost decrease, unchanged
anchor parameters and unchanged calibration. Check derivative tolerances on
the synthetic case, not against GT or by tuning real-data solver settings.

The solved state must contain every original pose and point ID exactly once,
the fixed anchor ID and all original source digests. Model publication must
bind that state to the original fixture/model/rig manifest; frame and sensor
membership comes from image names in the manifest, never image-ID arithmetic.
For each output image, compose `T_sensor<-rig * T_rig<-world`. Preserve camera
file bytes, image ID/camera ID/name and order, every POINTS2D token and order,
and each point's ID/RGB/track tokens and order. Only pose coordinates, XYZ
and recomputed mean Euclidean reprojection ERROR may change.

Reject missing/extra/duplicate state IDs, nonfinite values, invalid quaternions,
source digest mismatch, unsupported output observations or anchor movement.
Recompute every final observation's positive depth and full squared cost
from serialized output before GT scoring. Audit both directions of tracks,
all 1,000 supported images/500 supported frames/4,716 points/130,900 observations,
the original 361,170 keypoints and one connected rig component. Preserve
the existing fixed-anchor comparison tolerance rather than loosening it
to accommodate a candidate. Stage output in an owned private directory;
reject any existing destination, symlink or source overlap, and never remove
an unowned path when publication fails. This is a separate post-solve tool,
not part of the measured Ceres optimization time; report phase timing clearly.

### Frozen Ceres result: lower cost, worse trajectory (2026-09-08)

The [reference evidence](../benchmarks/electro/m8-openloris-ceres-reference-solve-v1.json)
records one certified Ceres solve at `1f31335`, followed by the strict model
publisher at `90672d4`. Root rebuilt and checked derivatives independently;
all initial residual/depth rows byte-match the earlier audited checkpoint.
The process took 77.45 s and 305,072 KiB peak RSS, including fixture evaluation
and state output but excluding model publication, scoring and mapping. Ceres
reports 76.776 s internally and stops at the 20-iteration limit with a usable
state (`NO_CONVERGENCE`), not a demonstrated converged optimum. Its successful
count of 14 includes iteration zero: 13 actual updates were accepted, seven
rejected. No second real solve or parameter sweep was run.

The following arms use the same frozen initial state and observations, but
different documented solver policies; historical Rust figures come from the
linked PR #85/#87 evidence, not new benchmark runs:

| Frozen 1k arm | Full squared cost | RMSE (m) | p95 (m) | Mean observation error (px) |
| --- | ---: | ---: | ---: | ---: |
| Legacy matrix-free | 118070.554369 | 0.026608 | 0.041100 | 0.691589 |
| Rust direct | 117496.033074 | 0.026519 | 0.040989 | 0.689260 |
| Fixed column-scaled | 113754.040510 | 0.028550 | 0.043791 | 0.675819 |
| Adaptive column-scaled | 113550.219339 | 0.029190 | 0.044521 | 0.675153 |
| Standalone Ceres | 113473.879982 | 0.029571 | 0.044943 | 0.674902 |

Both publications of the same Ceres state are byte-identical. Independent
audits retain all 1,000 supported images, 500 supported rig frames in one
component, 4,716 points, 130,900 observations and 361,170 keypoints, full
identity/order, exact camera bytes and fixed calibration/anchor. Every final
observation has positive depth. Maximum landmark motion is 231.001 m and
maximum camera-centre motion 0.029428 m; maximum observation error is
5.103062 px. These are reported separately from the mean. GT was consumed
only after publication audits; the same 308 images were scored and the
repeat score is identical.

An independent optimizer also lowers this objective while worsening the
trajectory. That supports investigating the observation objective and
geometric observability, but neither identifies a unique cause nor proves
that a different objective would improve accuracy. The existing COLMAP mapper
has a different point/observation set and is not this frozen objective control.
No atlas/default/README promotion follows. The next step is a bounded,
GT-free diagnosis of which input geometry and track populations carry cost
reduction and state motion, with its metric and work cap fixed before use.

### Pose-coupling diagnostic contract after connected mapping v2

Observation-identity motion and top-motion geometry audits are recorded in
`m8-atlas-observation-key-motion-v1.json` and
`m8-atlas-top-motion-geometry-v1.json`. Numeric point IDs are not stable across
publication. The five largest main-component motions are all two-observation
tracks with pre-BA ray angles below 0.113 degrees. The largest landmark motion
is 1004.613 m while its observing camera centres move at most 0.004214 m.
This identifies weak range observability, not a demonstrated trajectory cause.

The next diagnostic must use the actual `rig_residual_jacobians` in
`pipelines/slam/src/bundle.rs`, not an independently chosen camera-pose
parameterization. Its translation columns are the world-to-rig rotation;
rotation columns include `-R * skew(point_world)`, followed by the sensor
rotation and pinhole projection derivative. Preserve the actual fixed-pose
set, sensor transforms and fixed calibration. The atlas builder uses plain
squared residuals (`RobustKernel::None`) and sparse LM with initial lambda
1e-4; do not silently introduce robust weights into the diagnostic.

Implement an opt-in report at the first production window only, reusing its
selected landmarks and existing normal-equation blocks. Keep all production
window/landmark/observation caps, do not assemble a global dense pose matrix,
and never change acceptance, damping or landmark selection. Record observation
count, ray-angle range, residual squared cost and separate translation/rotation
gradient statistics. For pose coupling, attribute each landmark's Schur
contribution using the **same damped point block and fixed-pose handling as the
solver**. Do not interpret a raw Jacobian norm as influence: units, gauge and
point elimination matter. Near-singular/rejected blocks must be counted with
the solver's existing policy, not inverted through a new fallback.

Before applying this to real data, require synthetic fixed-boundary rig tests
and equality of aggregated diagnostic contributions with the production
system to a declared numerical tolerance. Require diagnostic-OFF model equality
and diagnostic-ON unchanged model bytes. No threshold tuning or intervention
is authorized by this diagnostic contract. In particular, the previously
rejected weak-angle-freezing arm is not a new experiment. Camera rotation,
coupling attribution and causality remain unmeasured at this checkpoint.

Implementation inspection: `collect_schur_block_debug_counts` already groups
all cross entries for one variable pose before forming its diagonal landmark
elimination block, and accepts the solver's inverse cache. Its existing rig
test covers two cross entries on the same pose and fixed-pose slot mapping.
Reuse that helper; summing separate observation elimination norms would miss
cross terms. Its current emission is in the matrix-free preconditioner path,
not the production atlas `solve_step_pose_blocks` path. The latter constructs
the actual damped inverse cache, skips singular point inverses and accumulates
lower-triangle pose blocks before factorization. Connect diagnostics after
that cache is constructed and before factorization, passing the cache rather
than recomputing inverses. Stable frame/landmark IDs must be passed from the
existing index maps; numeric exported point IDs cannot supply that mapping.
This is the identified implementation boundary, not a completed diagnostic.

### Feasibility v2 result and next experimental contract (2026-09-09)

The sparse diagnostic connection and before-state probe have now completed.
`m8-schur-feasibility-v2.json` records 14 unchanged reference hashes and 115
sampled rejected observations, all projectable before the tentative update.
Five internal tracks cross behind their sensors. Their sampled before-depth
range is 0.000068–0.038669 m. These are not the far-range top-motion tracks;
neither diagnosis proves a cause of trajectory error.

Next candidate, not implemented or promoted: bounded joint-step backtracking
on a feasibility-rejected legacy sparse rig LM update. The motivation is to
avoid another factorization when a smaller already-computed step may work.
[Ceres documents smaller-trust-region retries for invalid steps](https://ceres-solver.readthedocs.io/latest/nnls_solving.html#_CPPv4N5ceres6Solver7Options33max_num_consecutive_invalid_stepsE);
[Ipopt documents backtracking and fraction-to-boundary machinery](https://coin-or.github.io/Ipopt/classIpopt_1_1BacktrackingLineSearch.html).
These are methodological references, not evidence that either uses this exact
rig-BA heuristic or that it will improve visloc quality.

Fix the experimental policy before scoring: default OFF; legacy sparse rig LM
only, no velocity/bias/adaptive-damping path; try alpha=1/2,1/4,1/8,1/16
after a full-step feasibility rejection. Recompute every candidate from the
existing rollback state using the production SE3 update convention, scaling
pose and landmark increments together. Require finite strictly reduced cost,
the existing feasibility gate, and no newly nonprojectable previously valid
observation. Do not remove tracks, weaken depth checks, or select internal IDs.
If no candidate passes, restore exactly and take the existing LM rejection
path. Do not reuse full-step gain predictions or convergence-step norms for a
scaled update. Reuse rollback buffers; no extra full-state snapshots or dense
global matrices. Four evaluations are a work cap, not a tuned quality threshold.

Before any real replay, test joint pose/point scaling, fixed-pose handling,
nonfinite rejection, exact rollback, trial cap, cost-increasing feasible steps,
and OFF-path equivalence. Then freeze one arm and run a same-input A/B with
factorization/evaluation counts, wall time, peak RSS, registration/support,
reprojection and post-only GT trajectory scoring. Reject if quality or measured
runtime regresses; lower BA cost alone is not success. No threshold sweep using
GT. Current disk free space is only 1.1 GiB: no new full pipeline or repeated
atlas outputs until the storage budget is resolved without losing evidence.

Implementation checkpoint: `VISLOC_SFM_BA_FEASIBLE_BACKTRACK=1` enables the
experimental bounded policy; absent/other values retain the full-step path.
Separate release test processes with the flag absent/present each pass all
four generalized rig tests. The production point-only fixture rejects alpha
0.5 on increased cost and accepts alpha 0.25; a second fixture exhausts exactly
four trials and restores the complete problem. Tests assert fixed-pose equality
and accepted landmark-step norm consistency. This is not yet joint-variable
pose/point or fixed-rotation coverage, nor real-atlas OFF parity or an A/B result.
Clippy for the release slam library passes. No default or quality claim changes.

The production fixture now also runs with variable pose translation and variable
landmark position while holding rotation fixed. ON/OFF processes both pass:
joint alpha 0.25 is accepted after alpha 0.5 increases cost; both state increments
are nonzero and match the reported step norms; rotation remains exactly fixed.
Both point-only and joint exhaustion restore the entire problem exactly. CI now
runs the flag-ON rig tests separately after the normal flag-OFF workspace tests.
This extends synthetic coverage, not real-model parity or performance evidence.

### Real bounded-backtracking arm: rejected

`m8-feasible-backtrack-off-v1.json` proves real OFF parity for all 14 reference
files. `m8-feasible-backtrack-on-v1.json` records the same binary's completed ON
trial and eight independent pre-BA/fixed-camera hash matches. Scorer and scoring
input hashes, aliases and interpolation gap match the historical control.
RMSE worsens from 0.3889930047 m to 0.3891842840 m (COLMAP gate 0.3843065335 m).
P95 changes from 0.6381734851 m to 0.6380699612 m; registered images remain 9998.
Comparable integration-stage wall totals are OFF 153.3184523 s and ON
233.9327779 s. This is a single sequential suffix comparison, not a cold E2E
speed measurement. ON sampled aggregate RSS is 560852 KiB.

Reject this fixed arm: it fails trajectory nonregression and gives no evidence
of runtime benefit. Keep the flag OFF; do not repeat unchanged or tune the
alpha schedule against GT. Preserve the experiment and logs for accounting.
Before another mechanism, analyze the existing OFF/ON phase timings to check
where time was actually spent; no further geometry replay is justified by this
result alone. Free disk is now 386 MiB, requiring storage work before new runs.

Post-run phase accounting (`m8-feasible-backtrack-phase-accounting-v1.json`):
main normal-equation and linear-solve timer events both remain 2804. Linear
solve time is 27.038570 s OFF versus 26.998891 s ON, whereas tentative update
and cost time grows from 29.963184 s to 109.656562 s. Main emits 3430 scaled
candidates with 291 accepted. Timer counts are not an independent count of
numerical factorizations. The evidence does not support the intended reduction
of repeated solves; do not pursue this arm by tuning the backtracking schedule.
