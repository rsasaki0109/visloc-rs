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

The inner-timing tail replay preserves all six old model files. Wall14.46s,
peak77848KiB; solver6.9542s, build0.1957s, conversion/cost0.3180s,
filter/retriangulation0.8739s, connectivity0.3147s. The nested outer phase is
8.6698s. Main completes in164.04s with peak525228KiB and all six model files
byte-identical. Main solver87.0614s (53.1% wall), connectivity27.3257s (16.7%),
filter/retriangulation10.4504s, build3.4788s and conversion/cost4.2847s.
Both solver and connectivity merit investigation; keep the connectivity gate
intact. These measurements do not establish a speed improvement. Single-run
wall differs from the earlier10.78s trace; neither regression nor speedup is
established without a controlled repeat. Evidence:
`benchmarks/electro/m8-openloris-atlas-inner-timing-v1.json`.

## Solver ownership check before further optimization

Source inspection of `BundleAdjustment::optimize_weighted_backend` shows normal
equations rebuilt at each LM iteration, including after an exactly rolled-back
rejection. Reusing an unchanged linearization is a candidate for investigation,
not a measured win. `solve_step` currently moves `system.h_pp` out with
`mem::replace` for both pose-diagonal and dense branches. Retaining that consumed
system without restoring the original Hessian is incorrect. Copying a dense
Hessian to enable caching would reverse an existing memory optimization.
Column equilibration also mutates the system and needs a separate contract.
Any reuse experiment must first measure rejected-step assembly cost, preserve
the LM iteration/acceptance trace exactly, limit additional state to an explicit
bounded representation, and invalidate after accepted updates. Do not implement
a blanket clone cache or alter damping to obtain more accepted steps.

## Compact connectivity candidate

The candidate replaces per-observation supported-set insertion with image/frame
boolean flags, plus one image-to-compact-slot lookup. It unions each track's
frames as observations are visited, removing per-track temporary tree sets.
Duplicate observations merely repeat an idempotent union. Internal DSU roots
can differ, but component IDs are assigned by ascending supported frame order,
so the public support sets and component mapping must remain identical.
All candidate observations are still visited; support-loss/split gates remain.
Temporary extra state is O(images + frames), never frame-pair or landmark-pair
state. The old implementation remains a test oracle.64 variants of deletions,
replacement tracks, reversed observations and duplicate keys compare exactly;
unknown-image errors also agree. All52 example tests pass. No measured speedup
is claimed yet: repeat both retained components and verify model bytes/RSS.

Compact candidate tail: all six model files equal the historical output;
connectivity0.10246s versus preceding control0.31470s, wall12.15s versus14.46s,
RSS77916 versus77848KiB. These are single-run observations, not a certified
overall speedup or memory reduction. Main also preserves all six model files:
connectivity9.55577s versus27.32571s, wall151.46s versus164.04s,
RSS524744 versus525228KiB. Candidate main repeat is running, followed by the
retained control repeat. Do not infer improved trajectory: model bytes match
the still-RMSE-failing atlas baseline exactly.
Evidence: `benchmarks/electro/m8-openloris-compact-connectivity-v1.json`.

Serial repeats are complete and all six model files in every run match the
historical models. Main connectivity: control27.33/31.70s, compact9.56/10.47s;
main wall: control164.04/185.19s, compact151.46/158.84s. Tail connectivity:
control0.315/0.240s, compact0.102/0.106s. Tail wall variation is larger than
the isolated saving (control repeat10.69s is faster overall than compact's
first12.15s), so do not claim uniform whole-pipeline speed improvement.
The repeated phase reduction supports this equivalent-result optimization;
it does not establish lower trajectory error, generalization, a memory-saving
claim, or native end-to-end COLMAP superiority.

## Upstream cost is not zero

Read-only provenance audit matches21 of23 retained source-window models to
the old shifted-bridge manifest, including all three model-file hashes per
node. Their logged mapper durations sum to628.665215s. Nodes27 and28 are not
in that manifest and remain unresolved. This is a historical mapper subtotal,
not serial wall or native E2E: frontend, the two missing sources, alignment,
integration/refinement, publication and possible overlap accounting remain.
Evidence: `benchmarks/electro/m8-openloris-atlas-source-cost-v1.json`.
Do not compare the compact integration's151–159s directly against COLMAP's
complete mapper as though saved source windows were free.

Follow-up resolves nodes27/28 through
`/tmp/visloc-m8-seam650-nodes-20260907.tsv`. All23 retained source models now
match their original three files exactly. Added mapper times22.431005s and
27.669512s bring the raw per-model sum to678.765732s. This closes
source-model provenance, not native E2E accounting: alignment, frontend and
other excluded costs still require a reproducible phase ledger and replay.

Accounting correction: nodes12/13 share one `window-3000/mapper.log`, and
nodes18/19 share `window-4250/mapper.log`. These are different output components
of the same producing executions, not separate mapping runs. Deduplicating
by producing log yields21 executions and627.123191s, not678.765732s.
The evidence preserves the raw node sum and adds explicit execution groups so
the double count is auditable. Neither subtotal is a native E2E measurement.

## Source command coverage audit

A bounded follow-up rechecks all21 producing log hashes successfully and
looks for sibling time records containing a recorded command. Only5 of21
executions have such records. These are candidate provenance, not certified
replay commands: each must still be bound to the producing model, binary and
environment. The other16 lack command records in this checked location;
this does not prove records are absent everywhere. See
`benchmarks/electro/m8-openloris-source-command-coverage-v1.json` for the exact
search scope, files, hashes and raw command lines.

Do not call the historical627.123191s subtotal reproducible E2E. A future
source replay must explicitly save argv, environment policy, immutable binary
and input hashes, output hashes, exit status, wall and RSS for every unique
execution, then include frontend and atlas alignment/integration phases.
Do not silently guess historical flags to make a parity claim.

First concrete source replay uses the recorded650-start/500-frame command,
the saved0466499 binary and a new output directory. It completes in36.60s
with340100KiB peak RSS, and all three output model files exactly match the
historical source. This demonstrates one reproducible source output under the
new recorded execution, not the historical binary/environment or full E2E.
See `benchmarks/electro/m8-openloris-source-replay-650-v1.json` for argv,
resource report, full log and old/new output hashes. The other20 executions
and frontend/alignment/integration accounting remain outside this replay.
