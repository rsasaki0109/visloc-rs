# Bounded covisibility local BA experiment

Status: opt-in implementation fails1k quality gates; not promoted.
Same-binary default-off control matches Legacy exactly. Two candidate runs
produce identical model files and register1000 images, but RMSE0.025216m
exceeds0.022695m and reprojection0.703792px exceeds0.671637px.
See [repeat evidence](../benchmarks/electro/m8-openloris-1000-covisibility-v1.json).
Do not advance this unchanged policy to larger tiers or tune thresholds on GT.
`--local-ba-covisibility` selects the same helper in legacy and dynamic paths.
Initial selector tests cover stereo deduplication, ties, input order, caps,
unusable/unpositioned tracks and a100k-frame long track. CLI check passes.
The native fixed-state fixtures now obtain their active set through the selector
and check oldest selected anchor, unregistered observations, metric-only filtering
and default-off selection. All39 rig_sfm tests pass, including fixed-state and
rollback tests. Default parity and repeatability pass; quality fails. Larger-tier
and native E2E gates remain open, not implied by these tests.

## Evidence and hypothesis

The repeated strong/deferred boundary arm regresses trajectory and mapper time
despite retaining9998 image poses; see
[result](../benchmarks/electro/m8-openloris-strong-boundary-result-v1.json).
Do not combine that rejected arm with this experiment.

Both native registration paths currently choose the last40 registrations for
local BA. Registration order need not express useful geometric overlap.
Hypothesis: selecting already registered neighbors with shared usable landmarks
can improve the constraints in the same bounded pose budget. This does not
establish that selection explains the remaining10k trajectory error.

[COLMAP AdjustLocalBundle](https://github.com/colmap/colmap/blob/main/src/colmap/sfm/incremental_mapper.cc)
selects images using shared3D points, includes their rig frames, and separately
controls gauges and variable landmarks. This is design evidence inspected on
2026-09-08, not a pinned version of the benchmark executable. The policy below
is our proposed adaptation, not an exact reproduction of COLMAP.

## Single-factor contract

- Add an opt-in selector; default registration-order selection remains exact.
- Keep window cap40, BA frequency, iterations, solver, calibration, feature and
  pair inputs, registration thresholds and final BA unchanged.
- Boundary observations and disconnected-component anchoring stay disabled.
- Select the newest registered frame plus at most39 registered neighbors.
- Count distinct shared positioned tracks, not camera observations: stereo
  observations cannot give two votes for the same track/frame.
- Require a usable observation in the newest frame and in each counted neighbor:
  positive projection and the existing BA gate of2×max_reprojection_error_px.
  Honor the existing metric-only filter when that option is enabled.
- Rank by descending shared-track count, then ascending frame ID. Do not pad
  with zero-overlap frames. With no neighbor, skip local BA explicitly.
- Anchor the earliest registered selected frame. This preserves the existing
  oldest-in-window gauge rule, but its identity may change with selection.
  Do not fix the newest PnP pose or choose anchors using GT errors.
- Both legacy and dynamic paths call the same selector. Final BA is unchanged.

## Resource and ownership contract

Reuse `image_tracks` from each registration path to enumerate only tracks
incident to the newest frame. Deduplicate track IDs before visiting their
observations. Maintain temporary neighbor counts and a per-track frame set;
do not retain a global frame-pair graph or scan every track to find neighbors.
Use a bounded top-K heap with deterministic ties, not a full candidate sort.
Temporary storage is O(incident tracks + distinct neighbors + longest incident
track + K); neighbor count can be O(N), but no N² state is introduced.
Selection time depends on incident observations, not just K: long tracks must
be represented in synthetic tests and measured, not described as constant cost.
Existing BA still scans tracks when assembling its problem; this proposal does
not claim to remove that independent cost.

## Verification order and rejection gates

1. Unit tests: recent-vs-shared selection, duplicate stereo votes, unusable and
   unpositioned tracks, unregistered neighbors, deterministic ties and reversed
   input order, cap0/1/40, no-neighbor skip, oldest selected anchor, metric-only
   filter and long-track/100k-frame sparse state. Check both call sites.
2. Same-binary default-off1k control must match retained model files exactly.
   Run candidate1k twice; audit registration, support, fixed rig, positive depth,
   reprojection and GT with frozen scorer. Reject quality regression; do not
   tune on individual GT frame IDs or relax gates to rescue it.
3. If1k passes, run independent2.5k, derived5k with its scope disclosed, then
   the complete10k control/candidate repeats. Record selection cost, mapper wall
   and peak RSS, not merely BA iterations. Recheck unsupported rig frames.
4.10k must meet the frozen COLMAP registration, support, RMSE, p95 and
   reprojection gates, plus RSS≤2097152KiB. A small-tier win is not promotion.
5. Only an eligible candidate proceeds to independent final-tier validation,
   native end-to-end accounting including saved-prior generation, restart and
   100k I/O stress. No README superiority claim before those gates pass.

Disk preflight is mandatory before build and each run: current filesystem has
under100MiB available. Preserve input datasets and prior evidence. Reclaim only
verified duplicate experiment output storage or obtain additional capacity;
never overwrite existing hardlinked model outputs.
The strong-boundary candidate-b six model files were rechecked with `cmp` and
hardlinked to candidate-a, recovering about100MiB while preserving paths/bytes.
