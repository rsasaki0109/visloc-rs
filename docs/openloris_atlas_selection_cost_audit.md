# Atlas selection cost audit (2026-09-08)

This is a source/log audit, not a performance measurement or quality pass.
The closest retained scaled atlas still fails COLMAP RMSE and its1k gate.
Do not revive it as a promoted end-to-end pipeline.

## Observed work

`selected_landmarks_for_frames` in `integrate_rig_atlas_landmarks.rs` visits
every landmark per window and checks its observations for an active frame.
The filtering runner calls this once per window. Saved accepted-window logs
from `corridor1-1-m8-atlas-landmarks-v1/connected-filtered-ba-v1` contain:

| Component | Accepted windows | Selected landmark events | Selected observation events |
|---|---:|---:|---:|
| main | 150 | 846728 | 5504692 |
| tail | 17 | 82904 | 392637 |

These are events, not unique points. Final retained landmark counts are318222
and34615. Since this filtering pass only removes landmarks, selection inspects
at least47733300 and588455 landmark entries respectively. These bounds exclude
observation visits, validation, connectivity and solver work. They do not prove
that selection dominates wall time; there is no phase timer in this runner.

## Safe next optimization boundary

Before replacing selection, measure its elapsed time separately from solving,
validation/connectivity and application/compaction on an unchanged replay.
An index must preserve exactly the ascending current landmark-index selection,
not select only the top covisible tracks. Every active observation remains in
scope. Persistent IDs must survive accepted-window landmark compaction and
observation deletion; stale-index reuse could silently omit constraints.

Potential state is frame-to-track incidence O(observations), not frame-pair
adjacency O(N²). Its additional RSS must be measured. Updating/rebuilding it
after every compaction may erase the speed benefit. Retain the full-scan oracle
for differential tests, including deleted tracks, multi-camera duplicate frame
incidence and repeated windows. Do not remove the independent selection or
connectivity validation just to improve a timing number.

This work can reduce unchanged-objective overhead. It cannot itself fix the
remaining trajectory error or establish native end-to-end parity. Those gates
remain explicit in the M8–M10 plan.

## Diagnostic implementation

First retained-tail replay completes in10.78s with peak77828KiB and all six
pre-BA/final model files byte-identical to the old run. Selection totals0.0643s
(about0.6% of wall), versus6.4076s for combined build/solve/validation.
Main also completes with all six files byte-identical:185.59s wall,
525248KiB peak RSS,8.8369s selection (4.76% of wall),153.8248s combined
build/solve/validation,5.2928s application and0.6831s final validation.
Thus selection is not dominant in either measured component. Even removing
selection entirely would save under5% of this main replay. Prioritize separating
the combined phase into solver, retriangulation and connectivity costs before
adding an incidence index. Do not remove validation or change numerical policy
on this timing evidence. Evidence:
`benchmarks/electro/m8-openloris-atlas-timing-v1.json`.

`VISLOC_ATLAS_TRACE_WINDOW_TIMING=1` enables stderr-only timers in the filtering
runner for selection, counts, combined build/solve/validation, application,
baseline connectivity and final validation. No timer is started when disabled.
The combined phase deliberately does not claim solver-only timing. Trace I/O
is outside reported phase intervals; their sum is not end-to-end wall time.
Benchmark both actual components with identical inputs and compare model bytes
before treating this instrumentation as behavior-preserving evidence.

Build preflight found the filesystem full. Only regenerable workspace
`target/debug/incremental` cache (about4.9GiB) was moved to
`/dev/shm/visloc-atlas-build-cache.RcnQnI/incremental`; source and experiment
artifacts are unchanged. This tmpfs backup is not durable across reboot and
is not experiment evidence. Continue diagnostic builds with
`CARGO_INCREMENTAL=0` to avoid immediately refilling the disk.

## Inner-phase diagnostic

The same environment flag additionally emits `atlas-inner-timing` for
`problem_build`, `solver`, `output_conversion_and_costs`,
`filter_retriangulate_costs`, `candidate_connectivity` and
`candidate_pose_validation`. These are nested inside the previously measured
combined phase; do not add them to that outer duration when summing work.
The final phase reports on scope exit, including early rejection, so a timing
line alone does not certify acceptance. Pair with existing window status logs.
The active-frame start identifies windows; repeated passes must be separated
using log order and the outer pass field. Numerical settings and support/cost
acceptance remain unchanged. New full-input model parity is still required.
