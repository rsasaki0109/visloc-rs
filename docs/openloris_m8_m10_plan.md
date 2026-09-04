# OpenLORIS M8–M10 execution plan

**Status:** active  
**Scope:** OpenLORIS `corridor1-1`, frozen 1k / 2.5k / 5k / 10k tiers  
**Outcome:** visloc-rs must match or beat official COLMAP on registration,
post-map trajectory accuracy, and reprojection quality, while using less wall
time and peak RSS in both mapper-only and native end-to-end comparisons.

This plan is outcome-gated. ANN retrieval, bridge discovery, track repair, and
BA changes are possible means, not milestone success by themselves.

## Frozen facts and current deficits

| Evidence | Current result | Consequence |
|---|---:|---|
| 10k candidate manifest | 70,000 pairs (`7N`) | Candidate state must remain `O(NK)`, never all-pairs |
| M7 verified graph | 58,879 pairs, 313 components | Missing verified connectivity is one measured limiter |
| M7 component mapper | 6,791 / 10,000, 0.928–1.293 px | Registration and local reprojection are partial passes |
| M7 mapper resource | 644.45 s, 1,343,432 KiB | Already below the 2 GiB ceiling; preserve it |
| M7 official-GT score | 6,099 scored, 5.459 m weighted RMSE | Low reprojection error does not prevent long-range drift |
| M7 1k official-GT score | 1,000 registered, 308 scored, 0.358 m RMSE | Trajectory error starts before the 10k component split |

The GT score uses the official `base_link` trajectory, timestamp interpolation,
and official `base_link -> T265 fisheye` transforms. GT and transforms are
post-mapping inputs only. Candidate generation, matching, model selection, and
mapping must not read them.

## Comparison contracts

### Contract A — same-candidate control

This isolates frontend/mapping behavior while fixing the view budget.

- Same frozen tier images and two calibrated physical cameras.
- Same exact candidate pair list and pair count.
- CPU only, eight threads, immutable COLMAP container image digest.
- Fixed feature cap and matching thresholds recorded in `plan.json`.
- OpenCV principal points are shifted by `+0.5 px` when passed to COLMAP, as
  required by COLMAP's pixel-centre convention.
- Phase wall time, process `VmHWM`, whole-container cgroup peak, database
  counts, model hashes, registration, reprojection, and official-GT score are
  all retained.

This control is run at 1k before 10k. A failed or partial COLMAP model is still
evidence; it is not silently replaced by a weaker target.

### Contract B — native end to end

This measures the tool a user would actually run.

- visloc-rs: feature extraction, streamed global descriptor / ANN candidate
  generation, matching and verification, mapper, and output publication.
- COLMAP: feature extraction, its bounded non-exhaustive matching route, mapper,
  and output publication.
- Both receive the same images, official calibration, CPU8 budget, and no GT.
- Report feature capacity, produced/verified pair count, and output quality so a
  speed win cannot be purchased by silently doing less useful work.

Mapper-only means the mapper consumes each engine's already verified database
or snapshot. Native end-to-end includes every non-download phase. Cache state
and warm/cold labels are recorded; the decision uses three completed runs and
their median unless a run is a deterministic DNF.

## M8 — establish the quality target and close it

### Frozen 1k COLMAP acceptance baseline (2026-09-02)

The calibrated-rig v3 control completed on the frozen 1k tier. It registered
all `500/500` synchronized frames (`1000/1000` images) in one model, with
`0.810207 px` observation-weighted mean reprojection error. The official-GT
post-map score covers 308 interpolable images and reports `0.027969 m` ATE
RMSE and `0.042266 m` p95. Mapper wall time was `2073.400 s` and mapper
process `VmHWM` was `629664 KiB`; the measured CPU8 feature + rig setup +
matching + mapper total was `2405.538 s`.

The retained independent-pose diagnostic also registered `1000/1000`, but
needed `5676.497 s` in the mapper, used `1191612 KiB` mapper `VmHWM`, and
scored `0.029894 m` RMSE / `0.044628 m` p95. Thus the calibrated rig is the
official 1k target: it is 2.738x faster in the mapper, modestly more accurate,
and preserves reprojection quality. These are COLMAP-vs-COLMAP control results,
not a visloc performance claim and not the M8 10k exit gate.

The exact input, container, runner, database, model, score, timing, memory, and
warning hashes/counters are frozen in
[`benchmarks/electro/m8-openloris-colmap-1k-control.json`](../benchmarks/electro/m8-openloris-colmap-1k-control.json).

### Frozen 10k COLMAP acceptance baseline (2026-09-02)

The calibrated-rig v3 same-candidate control completed on the frozen 10k tier.
It registered `4999/5000` synchronized frames (`9998/10000` images) in two
independent models: 4494 frames in the largest model and 505 in the second.
The observation-weighted mean reprojection error is `0.903003 px`. Under one
Sim(3) alignment per model, the official-GT post-map score covers 9306 images
and reports `0.384307 m` ATE RMSE, `0.358549 m` median, and `0.638669 m` p95.

The mapper took `16663.880 s` (4 h 37 min 44 s); all measured CPU8 phases took
`19256.599 s` (5 h 20 min 57 s). Mapper process `VmHWM` was `2177632 KiB` and
the pipeline-wide container peak was `2225220 KiB`, both above 2 GiB. The
mapper completed successfully despite 962 retained linear-solver-failure
warnings. Its verified image graph was one 10,000-image component, so the two
output gauges and missing frame are mapper outcomes rather than a missing-edge
component bound.

This is now the authoritative M8 10k target. visloc-rs must reach at least
99.98% unique registration, no more than two gauges, a largest model of at
least 4494 frames, no worse trajectory and reprojection values, faster mapper
and native end-to-end wall time, and at most 2 GiB peak RSS. Exact inputs,
artifacts, hashes, model breakdown, scores, phase timings, memory, and warning
counters are frozen in
[`m8-openloris-colmap-10k-control.json`](../benchmarks/electro/m8-openloris-colmap-10k-control.json).

### Frozen 1k visloc-rs rig champion (2026-09-02)

The generalized-rig mapper now passes every 1k COLMAP quality and mapper
control gate on the same frozen images and calibrated physical rig. Three
independent runs produced byte-identical models and registered all `500/500`
frames (`1000/1000` images). The official-GT score is `0.022695 m` RMSE and
`0.037779 m` p95, with `0.671637 px` observation-weighted mean reprojection
error. All three are better than the calibrated COLMAP control (`0.027969 m`,
`0.042266 m`, and `0.810207 px`).

Median mapper wall time is `5.724 s` (362.22x faster than COLMAP's
`2073.400 s`) and median process peak RSS is `80924 KiB` (7.78x lower than
COLMAP's `629664 KiB`). The promoted bounded refinement uses registration-order
local BA every ten frames over at most 40 frames, two endpoint-anchored final
passes over at most 60 frames, and independent fixed-pose landmark refinement.
Frontier track IDs are canonicalized before triangulation, removing the
process-randomized cache/RANSAC ordering found during A/B testing.

This is the frozen M8 **1k control pass**, not the M8 exit claim. The same
quality and resource gates still have to pass at 10k. Exact inputs, defaults,
commands, three-run measurements, scores, and hashes are retained in
[`m8-openloris-visloc-rig-1k-champion.json`](../benchmarks/electro/m8-openloris-visloc-rig-1k-champion.json).

### Dense same-candidate 1k pilot (2026-09-03)

The COLMAP-compatible 256-keypoint / two-orientation SIFT arm is now validated
on the first 1k images before committing its full 10k frontend cost. It keeps
the frozen 7,000 candidate pairs, yields 6,935 verified pairs and 1,775,213
accepted correspondences, and runs through the provenance-bound streamed
matcher and normal atomic merge path. With legacy conflict handling and a
256-correspondence mapper cap it registers 1,000/1,000 images at 0.720679 px,
0.026703 m ATE RMSE, and 0.041346 m p95. All four quality values beat the
official COLMAP 1k control.

The cap-aware compact reader validates the complete snapshot and every raw
index relation, but releases excess accepted and mapper-unused essential
correspondences at each pair boundary. It preserves byte-identical
`cameras.txt`, `images.txt`, and `points3D.txt` while reducing the pilot replay
peak from 205,832 KiB to 140,792 KiB. Cap 96 and pair-confidence ordering were
rejected because they worsened trajectory accuracy. Full measurements and
hashes are in
[`m8-openloris-dense256x2-1k-control.json`](../benchmarks/electro/m8-openloris-dense256x2-1k-control.json).

This pilot proves the configuration and 1k gate only. The dense 10k
extraction/match/map/score remains the M8 decision run.

### Dense ANN 2.5k promotion gate (2026-09-03)

The first dense 2.5k control registered all 2,500 images but had no verified
pair at a rig-frame gap of 128 or more; its late trajectory drift reached
0.382746 m RMSE and 0.885519 m p95. A streamed VLAD-LSH schedule now retains
the full offset-1/2/4/8/16/32 temporal pyramid and same-frame rig edges, then
spends the remainder of an 8N budget only on appearance pairs at least 128
rig frames apart. It produced 2,086 verified long-range pairs without GT.

The promoted 2.5k arm still registers 2,500/2,500 images in one model. Its
0.788549 px reprojection mean is within 0.006 px of the control, while official
GT scoring improves to 0.133927 m RMSE and 0.237364 m p95 (65.01% and 73.19%
reductions). Candidate generation, streamed matching, bounded merge, and mapper
peak RSS were 108,796, 121,084, 7,588, and 361,732 KiB respectively. The exact
policy, phase measurements, scores, and hashes are frozen in
[`m8-openloris-dense256x2-2500-ann-gap128.json`](../benchmarks/electro/m8-openloris-dense256x2-2500-ann-gap128.json).

This is a promotion gate, not the M8 exit claim. The same policy must now pass
the frozen official 10k registration, trajectory, reprojection, and resource
controls.

### Dense ANN 10k first-decision diagnostics (2026-09-03)

The promoted bounded frontend completed on the frozen 10k manifest without
using ground truth for candidate selection, matching, or mapping. It produced
80,000 unique candidates, 64,862 verified pairs, and 7,698,149 accepted
correspondences. Candidate preparation took 500.80 s at 389,820 KiB peak RSS;
the persistent streamed matcher took 564.98 s at 451,532 KiB; and the bounded
atomic merge took 12.64 s at 18,156 KiB. The merged snapshot is 517,363,024
bytes with SHA-256
`02cd6475098453cb6cf93761ff990ea31acba86217b715dfaf43660f3af0f05c`.

The first legacy-union-find / 256-match-cap mapper registered 4,968/5,000
frames (9,936/10,000 images) in two models, including 4,492 frames in the
largest model. It completed in 96.947 s at 1,121,496 KiB peak RSS and achieved
0.811235 px mean reprojection error, but official-GT scoring was only
0.878064 m RMSE / 1.485270 m p95. This passes the reprojection, mapper-time,
and memory controls but fails registration and trajectory, so it is not the
M8 champion.

A GT-free two-pass diagnostic then used a local-track first-pass model only as
a rotation-consistency prior. Of 5,056 verified pairs at a rig-frame gap of at
least 128, a 5 degree gate admitted 179 into structure construction and kept
4,877 deferred. It retained the same 9,936-image registration, improved mean
reprojection to 0.798247 px, and improved trajectory to 0.639967 m RMSE /
1.174449 m p95 in 97.139 s at 1,124,644 KiB peak RSS. This is a real quality
gain but still fails the COLMAP trajectory and registration gates. Controlled
4 and 10 degree variants regressed to 0.924542 m and 0.914736 m RMSE; requiring
35 essential inliers at 5 degrees regressed to 0.945990 m. Threshold searching
therefore stops here: the next implementation must apply verified long-range
rotations directly to bounded frame-pose refinement rather than relying on a
discontinuous track-topology side effect.

The direct-rotation follow-up was also measured rather than assumed. A sparse
chordal pass admitted 13,394 deduplicated frame constraints, but its one-degree
per-component safety gate allowed updates to only 12 frames; 50 components had
proposed updates above the bound, with a 156.34 degree maximum. The subsequent
fixed-rotation rig BA retained 9,936 registered images and improved the score
only marginally to 0.638240 m RMSE / 1.173289 m p95, while mapper time rose to
1,307.540 s and peak RSS to 1,701,412 KiB. It is therefore rejected: it still
fails registration and trajectory, and the 0.00173 m RMSE change does not
justify a 13.5x mapper slowdown. The next quality iteration must first diagnose
orientation conventions and inconsistent cycles, pass synthetic and small-tier
gates, and apply only bounded local/incremental refinement before another 10k
run. Exact counters, hashes, and A/B scores are frozen in
[`m8-openloris-dense256x2-10k-rotation-gate-ab.json`](../benchmarks/electro/m8-openloris-dense256x2-10k-rotation-gate-ab.json).

The next first-divergence experiment imported only COLMAP's frontend boundary:
3,476,053 keypoints and 9,352,301 verified correspondences across 69,621 pairs,
with no poses or points. Feeding those matches to the legacy all-pairs track
union registered only 3,886 images, proving that the lower visloc verified-pair
count is not the sole cause: COLMAP's correspondence graph exposes destructive
track-conflict chains in the current builder. A separate diagnostic transferred
only final track membership as chain edges. It recovered a 4,493-frame largest
component (COLMAP: 4,494), but not COLMAP's initialization/BA trajectory; its
RMSE was 7.630 m and it could not seed the remaining component. Track membership
therefore diagnoses registration topology but is not a quality shortcut or a
candidate production path.

A BA boundary A/B on the GT-free five-degree arm further localizes the drift.
Removing all BA reduced registration to 6,486 images and scored 1.315 m RMSE.
Keeping local BA but removing final BA restored all 9,936 images and scored
0.649 m; final BA reached 0.640 m. Relative rotation increments already agree
with COLMAP at 0.085 degree median / 0.300 degree p95, while median translation
step scale grows from 1.016 in the first 500 frames to 1.217 around frames
3,000–3,499. Hard-fixing both local-window endpoints was rejected at 1.927 m
RMSE because it freezes the newest PnP error. The next implementation target is
therefore a soft, bounded metric-scale constraint inside incremental local BA,
not another global rotation solve or threshold search. Full bridge hashes and
counters are frozen in
[`m8-openloris-10k-first-divergence.json`](../benchmarks/electro/m8-openloris-10k-first-divergence.json).

The first soft-scale candidate used a five-centimetre translation trust region
around each pre-BA generalized-PnP centre. It was stopped at the frozen 1k
gate: registration remained 1,000/1,000, but RMSE regressed from 0.026703 m to
0.031071 m and p95 from 0.041346 m to 0.047075 m, exceeding the COLMAP 1k
quality control. The implementation was removed and was not run at 10k. The
next change must instead retain more non-conflicting observations while
preserving metric stereo anchors; only then should scale be reassessed.

Whole-builder replacement also fails the scale test. Conflict-preserving
tracks doubled 1k observations but regressed RMSE/p95 to 0.029775/0.045297 m,
so that arm stopped before 10k. Metric-temporal-cycle tracks passed 1k at
0.026650/0.041043 m, but at 10k they increased retained observations to
1,565,365 while regressing reprojection to 0.935693 px and trajectory to
1.558200 m RMSE / 3.178082 m p95. Repeated corridor views create false cycles
that a short-tier gate cannot expose. The next track repair must keep the
stable local legacy topology and admit only pose-consistent verified long-range
observations into already-triangulated tracks; switching the complete graph to
a more permissive builder is rejected.

The registration cliff is now closed at the frozen COLMAP gate. Inspection of
the 32-frame gap showed 542 candidate edges already present, but the historical
`min_matches=30` verifier retained only 51 incident edges and three right-side
boundary edges. Replaying the same 80,000 candidates and same feature bank with
COLMAP's eight-inlier minimum retained 70,567 verified pairs and 7,818,765
correspondences; the gap rose to 242 incident and 33 right-boundary edges. The
streamed worker remained bounded at 451,544 KiB and took 572.97 s versus
564.98 s for the 30-match control.

Putting all newly accepted pairs into legacy union-find was rejected: track
conflicts reduced registration to 8,986 images. The promoted boundary instead
keeps the previous 59,985 strong/rotation-consistent pairs and 7,249,960
correspondences byte-for-byte in structure construction, while moving 5,705
low-support pairs plus rejected long pairs into the existing deferred suffix.
Deferred pairs never merge tracks or move established structure. An opt-in
two-frame direct bridge may use them only to create temporary source-stereo 3-D
points for robust PnP; its one-sensor exception is separate from normal PnP.
This reaches the exact COLMAP registration gate of 9,998/10,000 images (4,494
and 505 frames), with 0.730311 px reprojection, 136.54 s mapper time, and
1,480,916 KiB peak RSS. GT scoring improves slightly over the previous visloc
arm to 0.637290 m RMSE / 1.170177 m p95, but still fails COLMAP's
0.384307/0.638669 m trajectory gate. The snapshot SHA-256 is
`01137f6ca189c0d54eb59940a7eadccd9cde775b4e472db303b2934de4ff408a`.

Restricting every BA row to metric-anchored tracks was also stopped at 1k: it
reduced mapper time from 20.36 s to 12.77 s but regressed RMSE from 0.02672 m
to 0.17914 m. One-sided long-range observation completion was then tested and
removed: a permissive form produced a 64.37 m pose outlier, while a metric,
one-observation-per-track, one-pixel gate admitted only 84 observations and
slightly regressed RMSE to 0.637291 m. Running COLMAP global BA on the unchanged
small visloc component improved its RMSE from 0.1902 m to 0.1404 m, still far
from the native COLMAP component's 0.0557 m. The remaining quality work must
therefore combine bounded track completion/retriangulation with refinement;
neither BA row filtering, BA alone, nor one-sided observation attachment is
sufficient. Exact A/B counters and hashes are recorded in
[`m8-openloris-10k-first-divergence.json`](../benchmarks/electro/m8-openloris-10k-first-divergence.json).

A bounded whole-graph completion pass is now retained as a reprojection-only
improvement. After registration it scans the 70,567-pair sparse graph at most
twice, never merges two positioned tracks, and uses a private owner index so
the PnP cache is unchanged. At 10k it attached 10,348 observations while
holding registration and trajectory exactly fixed, improving mean reprojection
from 0.730311 to 0.727707 px at 1,481,008 KiB peak RSS. The final dense
per-keypoint owner implementation reproduced every 1k and 10k COLMAP text
model byte-for-byte against the earlier sparse-owner implementation; its
measured 10k mapper time was 139.15 s and it remains below the 2 GiB gate.
Moving completion before
the first BA admitted nothing at 1--2 px; relaxing it to 4 px regressed the 1k
trajectory and reprojection. Re-running BA after completion nearly doubled 1k
mapper time and also regressed both metrics, so neither variant was retained.

Uniformly strengthening metric-track BA rows was also rejected. A weight of two
improved the 1k RMSE by about 5%, but regressed the dense 10k arm to 0.675458 m
and the previously best adaptive/pair-confidence arm to 0.865597 m. Requiring
stereo support in two or more rig frames still regressed 10k to 0.839489 m:
long repeated-corridor false tracks can satisfy a support-count gate. Restricting
the same test to usable stereo observations inside each BA window improved all
1k metrics but still regressed 10k to 0.863692 m. The next scale-drift
experiment must therefore test local stereo triangulations for geometric
self-consistency before a track may influence a pose-moving scale anchor; raw
metric/support/window labels are insufficient.

That self-consistency test is now implemented as a GT-free post-mapping
diagnostic and rejects the direct BA-weighting hypothesis. In the large 10k
component, visloc's median maximum deviation between independently
triangulated stereo points is 0.0862 m, apparently better than COLMAP's
0.1153 m despite visloc's substantially worse ATE. COLMAP's 1,010-image
component has no track observed stereoscopically in two frames at all, yet its
trajectory remains much better. Reprojection-derived self-consistency is
therefore endogenous to the current optimized solution and cannot establish
absolute trajectory correctness. The next pose-quality experiment must use
independent metric inter-frame motion from synchronized stereo 3D-to-3D
correspondences, not another per-track BA multiplier. Full distributions and
the bounded diagnostic contract are recorded in
[`m8-openloris-stereo-track-consistency.json`](../benchmarks/electro/m8-openloris-stereo-track-consistency.json).

Independent metric inter-frame motion was then tested rather than merely
planned. Calibrated stereo points in each rig frame feed deterministic
3D-to-3D Kabsch RANSAC, with no GT in estimation. One-to-two-frame motion is
unobservable (median metric/mapper translation ratio 22.30 and median direction
cosine 0.16), while exact 128-frame motion is stable at 1k (ratio 1.094,
direction cosine 0.993). It does not cover the 10k failure region: the
16--32-frame screen has only six accepted pairs in each of frames 2000--2499
and 2500--2999. Worse, its frame-3000--3499 median asks for a 1.056 scale
increase, whereas the independent post-mapping GT diagnostic shows mapper
steps are already 1.217 times too large there. This sparse signal is therefore
rejected as a pose constraint. The bounded implementation, synthetic metric
test, and complete per-bin counts are recorded in
[`m8-openloris-metric-motion-observability.json`](../benchmarks/electro/m8-openloris-metric-motion-observability.json).

The near-gate adaptive pair-confidence model was also passed through COLMAP's
100-iteration global bundle adjuster to measure the fixed-track solver ceiling.
The small component improves from 0.1751 to 0.1396 m RMSE, but the dominant
component regresses from 0.4358 to 0.4477 m. Aggregate RMSE therefore worsens
from 0.4155 to 0.4252 m and p95 from 0.8322 to 0.8373 m. Its usable
observation-weighted reprojection does fall from 1.0073 to 0.8972 px, below
the COLMAP control's 0.9030 px, while 146 observations become nonprojectable.
This isolates topology/outlier handling rather than the nonlinear solver as
the next boundary: fixed-track global BA can close reprojection but not the
trajectory gate.

Running the bounded completion pass on that adaptive model attached 19,660
observations and improved mean reprojection from 1.00727 to 0.99046 px while
leaving its 9,998-image registration and 0.41550 m RMSE trajectory unchanged.
COLMAP BA after completion reached 0.88457 px but regressed trajectory to
0.42776 m. Applying COLMAP's two-view/4 px/1.5 degree point filter first
reached 0.87122 px but regressed further to 0.42983 m. Completion is therefore
retained as a bounded reprojection improvement; neither more fixed-track BA nor
post-hoc point filtering is the missing trajectory mechanism.

A separate GT-free continuity diagnostic found metre-scale, one-frame pose
detours in the adaptive trajectory. An opt-in `O(frames * passes)` repair now
requires a 0.25 m midpoint error and an 8x detour, replaces the rig pose by
translation midpoint/rotation SLERP, and reruns fixed-pose structure
refinement. The 1k control selected zero poses and reproduced all three model
files byte-for-byte. A diagnostic 10k run retained 9,998 registrations and
about 954 MiB peak RSS, improving RMSE from 0.41550 to 0.40150 m and maximum
error from 4.16 to 1.48 m, with a negligible reprojection change from 0.99046
to 0.99051 px. It still misses COLMAP's 0.38431 m gate and remains disabled by
default. Applying the same repair before final BA is better: several outliers
have fewer than the configured 32 pose observations and were otherwise frozen
at their bad PnP estimates. The pre/post arm selected 26 largest-component
poses and reached 0.39445 m RMSE / 0.79608 m p95 at 0.99055 px, 9,998 images,
976,916 KiB peak RSS, and 150.20 s mapper time. It is still 0.01014 m above the
COLMAP RMSE gate. The CLI also disables continuity repair on compacted later-model
inputs, whose adjacent local indices are not guaranteed to be adjacent source
timestamps.

The bounded paired-jump follow-up was then implemented and measured rather
than promoted from its offline score. It pairs mutually cancelling translation
steps at most 16 frames apart and subtracts one common segment offset while
preserving internal motion and rotation. The detector is `O(frames * 16)`, is
disabled by default, and selected zero candidates at 1k; that control reproduced
all three model files byte-for-byte. On the adaptive 10k arm it repaired 14
segments / 31 poses before final BA and retained 9,998 registrations at 976,532
KiB peak RSS. Reprojection improved slightly from 0.99055 to 0.98969 px, but
RMSE regressed from 0.39445 to 0.39467 m and p95 from 0.79608 to 0.80719 m.
The real mapper A/B therefore rejects this arm despite its 0.39301 m offline
screen: it neither beats isolated-only nor reaches the COLMAP trajectory or
reprojection gates. Exact diagnostics and the pre-safety-fix run are frozen in
[`m8-openloris-pose-continuity-repair-ab.json`](../benchmarks/electro/m8-openloris-pose-continuity-repair-ab.json).

The segment-scale follow-up was implemented as a bounded, default-off
translation averager and stopped at its 1k promotion gate. It fixes the
incremental rig rotations, re-estimates same-sensor translation axes from at
most 64 actual correspondences per pair, rejects non-isolated epipolar
nullspaces and direction outliers, and uses the existing independent-edge-scale
solver. A unit-weight consecutive-frame backbone made all 500 registered
frames solvable without fixing their current step magnitudes. The 727 accepted
verified directions plus 499 backbone edges proposed updates up to 1.5776 m.

That connected proposal is geometrically invalid: fixed-rotation structure
refinement retained only 364/4,809 tracks and 63,732/172,752 observations,
while mean reprojection worsened from 0.67503 to 1.46679 px. Both independent
publication guards fired, and the transaction restored byte-identical control
models. The run took 184.59 s mapper time and 261,196 KiB peak RSS, so this is
also not a viable performance trade. No 2.5k or 10k threshold search is
allowed. Exact configuration, candidate-before-rollback measurements, hashes,
and raw artifacts are frozen in
[`m8-openloris-rig-translation-average-1k.json`](../benchmarks/electro/m8-openloris-rig-translation-average-1k.json).

The next quality step returns to the first-divergence evidence: localize track
support, triangulation, and robust-BA residual changes across the first 10k
drift segment before proposing another pose solver. The independent-scale
failure shows that sparse corridor pair directions are not themselves a safe
segment-scale constraint, even with fixed rotations and explicit graph
connectivity. Any replacement must first demonstrate a GT-free residual signal
on the existing snapshot, remain linear in observations plus admitted sparse
edges, and pass the 1k gate before a larger run.

A new GT-free temporal support diagnostic now supplies that signal. It parses
the published COLMAP text models directly, recomputes every usable
reprojection residual, and bins track length, temporal span, and endpoint
triangulation angle without a track Cartesian product. In the equal 8,988-image
largest components, visloc retains 40,203 points / 589,009 observations versus
COLMAP's 103,611 / 1,800,787. In frames 3000--3499, independently identified
above as a translation-scale drift interval, visloc has 3,052 anchored tracks
versus 10,236; median track support is 4 versus 6 observations, median span is
9 versus 17 frames, and median endpoint angle is 2.59 versus 6.26 degrees.
Mean reprojection in the same bin is 1.032 versus 0.876 px.

This changes the next implementation boundary: trace the verified
correspondences lost between snapshot track construction, triangulation, and
final map support in frames 3000--3499, then repair only the measured
long/wide-angle topology deficit under explicit per-feature alternative caps.
Another pose solver is not justified until that support gap is addressed. The
reproducible per-component/per-bin measurements are in
[`m8-openloris-temporal-support-residuals.json`](../benchmarks/electro/m8-openloris-temporal-support-residuals.json),
generated by `scripts/diagnose_sfm_temporal_support.py`.

The source snapshot rules out simple retrieval/verifier starvation in that
interval. Frames 3000--3499 still have 6,315 verified pairs, 280,717 accepted
correspondences, and 46,546 accepted correspondences on pairs spanning at
least 16 frames; several other bins have less verified input but more final
support. The deficit therefore occurs after verification. The next bounded
instrumentation must classify track-construction retention and triangulation
rejections by 500-frame bin and frame-gap class, without retaining per-pair
diagnostic state. The snapshot-versus-map counts and their deliberately
non-ratio population definitions are frozen in
[`m8-openloris-track-support-loss.json`](../benchmarks/electro/m8-openloris-track-support-loss.json).

The first bounded repair now prunes inconsistent *registered* observations
during an already scheduled triangulation attempt instead of discarding the
whole track. It requires at least two inliers and a configurable consensus
fraction, keeps unregistered observations available for later growth, and
restores the complete track state after a failed seed. The feature is disabled
by default and adds no pairwise state. At 1k, a 75% consensus with a 3 px Huber
BA kernel registers all images, raises support from 6,266 tracks / 270,204
observations to 6,530 / 279,489, and improves RMSE from 0.02751 to 0.02681 m
and p95 from 0.04235 to 0.04152 m. Its 0.80350 px reprojection also beats the
frozen COLMAP 1k control's 0.81021 px.

At 2.5k the same arm improves the legacy control trajectory from 0.13393 to
0.07185 m RMSE and 0.23736 to 0.13545 m p95, but reprojection rises from
0.78855 to 0.91968 px. Bounded two-pass completion reaches only 0.91669 px.
The predeclared Huber-1 check reaches 0.91873 px but regresses RMSE/p95 to
0.07352/0.14237 m, so threshold search is closed. A GT-free residual audit
shows that the restored support is not a single bad temporal tail: residuals
rise in every span class, while 32--127-frame observations grow from 58,296 to
187,414 and 128+ observations from 60,187 to 225,818. All counters, resource
measurements, scored variants, and the rejected fixed-pose Huber probe are
frozen in
[`m8-openloris-robust-triangulation-ab.json`](../benchmarks/electro/m8-openloris-robust-triangulation-ab.json).

COLMAP's official mapper source performs global BA, correspondence completion,
observation filtering, then repeats BA while the model still changes. A
bounded, default-off rig adaptation now exposes
`--final-filter-refinement-passes`: each pass scans retained observations once,
removes registered observations outside the existing 4 px mapper gate, and
reruns the existing windowed BA plus fixed-pose structure refinement. It keeps
unregistered observations and introduces no dense/global matrix or pairwise
state. With one pass, 1k remains above every frozen COLMAP quality gate while
improving reprojection from 0.80350 to 0.77112 px. At 2.5k it improves
reprojection from 0.91968 to 0.87528 px, RMSE from 0.07185 to 0.07116 m, and
p95 from 0.13545 to 0.13458 m at 508,884 KiB peak RSS. Default-off output is
byte-identical. The complete A/B and upstream source links are frozen in
[`m8-openloris-final-filter-refinement-ab.json`](../benchmarks/electro/m8-openloris-final-filter-refinement-ab.json).

At 10k, robust triangulation does not generalize: combined with one filter
pass it falls to 9,994 registered images and 0.91568 m RMSE, despite reaching
0.89650 px. With robust admission disabled, one and two filter passes preserve
9,998 images and the 4,494-frame largest model. Two passes reach 0.87117 px,
0.39147 m RMSE, and 0.77205 m p95 in 226.57 mapper seconds at 977,380 KiB.
Thus registration, reprojection, time, and memory pass, while RMSE and p95
remain above COLMAP's 0.38431/0.63867 m gates. A third pass is not eligible:
its 1k p95 is 0.042455 m versus COLMAP's 0.042266 m. The refinement cap is
therefore frozen at two; the next quality work must address the measured
segment-5/6 topology/scale deficit rather than continue a BA-pass search.

The earlier adaptive-feature ANN80k failure was not an equal-base comparison:
it omitted 5,281 pairs present in the later targeted7 snapshot. Reusing its
verified output, a deterministic sparse union added 10,264 pairs and 606,602
correspondences to targeted7 without rerunning retrieval or matching. Under
the same two-pass filter mapper it preserved 9,998 images and improved
reprojection from 0.87117 to 0.86536 px, but regressed RMSE/p95 from
0.39147/0.77205 to 0.39283/0.78604 m at 1,061,400 KiB. The initial
triangulation count remained byte-for-byte identical because all ANN pairs
were appended after the frozen 59,961-pair structure prefix. This rejects the
union as a post-registration completion fix, not ANN retrieval as a structure
constraint. The next ANN arm must use a GT-free topology gate and place only
the admitted support inside the structure prefix while retaining targeted7 as
the deferred tail. Exact provenance and hashes are frozen in
[`m8-openloris-ann80k-union-10k-ab.json`](../benchmarks/electro/m8-openloris-ann80k-union-10k-ab.json).

A stricter follow-up classified matches against the frozen structure graph and
admitted only one-sided extensions: one endpoint had to belong to an existing
track, the other had to be a singleton, established tracks could never merge,
duplicate-image observations were forbidden, and each track was capped at one
extension. Only 9,109 of 43,947 candidate matches survived. Once inserted into
the structure prefix, however, they reduced registration to 9,994 images and
failed catastrophically at 1.28612 m RMSE / 1.66458 m p95. The mode is retained
only in the offline admission utility to reproduce this diagnosis; it is not a
mapper option or promoted pipeline. In a repeated corridor, graph membership
or cycle shape alone is not independent evidence against aliasing. Future
structural ANN admission must additionally prove multi-path relative-rotation
consistency and wide-baseline triangulation quality before it can influence
poses.

A disjoint dense-feature overlay then isolated ordering from feature volume.
The frozen adaptive prefix remains the only registration input; dense SIFT
indices are shifted into a separate per-image range, and a default-off
post-registration pass builds and triangulates only previously unowned tracks.
Putting dense pairs into the structure stage failed catastrophically. Deferring
them preserved the exact 9,998-image frontier. The bounded 32--128-frame metric
arm retained 10,971 dense pairs / 1,099,253 correspondences, reached 360,968
tracks and 1,528,640 observations, reduced reprojection to 0.68200 px and RMSE
to 0.38754 m, and used 1,397,980 KiB peak RSS in 333.37 mapper seconds. It
still missed p95 at 0.76836 m. Allowing every long candidate regressed p95 to
0.78403 m, and widening final BA from 60 to 120 frames regressed RMSE/p95 to
0.39899/0.80671 m. A full 0--128-frame overlay exceeded the memory gate at
2,327,836 KiB. These results close feature-volume, long-candidate, and wider
local-window searches; the remaining segment-5/6 error requires inter-submap
scale consistency. Exact A/B data are frozen in
[`m8-openloris-disjoint-dense-deferred-ab.json`](../benchmarks/electro/m8-openloris-disjoint-dense-deferred-ab.json).

A counter audit found that deferred-track pruning initialized its retain index
at the new-track boundary instead of zero. It preserved every positioned base
track but misreported those tracks as newly recovered and removed unused,
unpositioned base tracks before completion. After correcting the index and
adding a regression test, the compact 32--128 arm reproduced both component
`images.txt` files byte-for-byte and exactly reproduced every published quality
metric. The corrected counters are 317,350 deferred tracks / 942,977
observations; mapper time and peak RSS are 339.42 s and 1,397,916 KiB.

Post-map diagnosis now separates local trajectory shape from inter-segment
gauge error. Independently Sim(3)-aligning ten temporal pieces of the dominant
component reduces its worst drift piece (segment 6) from 0.870 m RMSE in the
component-wide gauge to 0.078 m RMSE and 0.140 m p95. This use of GT is
diagnostic only and occurs after mapping. Freeing the second final-BA window
anchor does not solve the problem: it regresses aggregate RMSE/p95 to
0.4363/0.9239 m. Restricting only the post-registration refinement to
metric-labelled tracks while freeing that anchor regresses further to
0.5319/1.1483 m, so its temporary options were removed.

The GT-free cause is observable in the dense final map. It contains 352,185
independently triangulated local stereo points and 28,092 tracks with such
points in two frames, but robust 3D-to-3D motion at gaps 32--128 is extremely
skewed: 12,780 accepted frame pairs start in frames 0--499, 172 in 500--999,
76 in 1000--1499, and none in 1500--4493. Same-frame metric feature volume
therefore does not connect the drifting middle submaps, and a free BA gauge is
driven by repeated-corridor aliases. The next bounded quality implementation
must create and geometrically verify multi-frame metric tracks in that missing
interval before any submap scale solve; another BA-window or feature-count
search is not justified.

The bounded follow-up restored the omitted gap-1/2/4/8/16 dense chains without
changing the frozen registration prefix. A separate overlay cap retained at
most 64 matches per added pair while the base remained at 256, and deferred
track construction used metric-first temporal cycle ordering. Only tracks with
synchronized stereo support in at least two distinct rig frames could survive.
This recovered 10,706 tracks / 358,462 observations from the full 0--128-frame
stream, retained 9,998 registrations, and reached 0.78210 px in 295.94 s.
However, trajectory regressed to 0.39167 m RMSE / 0.78045 m p95 and peak RSS
was 2,034,548 KiB, only about 62 MiB below the hard limit.

Allowing a single anchor only in the post-registration metric-track BA cycles
also failed: mapper time fell to 263.08 s and reprojection remained below
COLMAP at 0.81667 px, but RMSE/p95 regressed to 0.40148/0.92538 m. The
temporary post-BA controls were removed. The separate overlay cap and bounded
multi-frame metric-track diagnostic remain default-off and tested. This closes
BA gauge adjustment even after explicit metric-cycle recovery. The next
quality boundary moves before track construction: repeated-corridor aliases
must be rejected by an independent sequential/retrieval consistency signal;
post-hoc metric labels cannot make the admitted false cycles safe.

Restricting the dense graph further to synchronized stereo plus verified
adjacent-frame edges recovers 27,976 multi-frame metric tracks with a much
safer 1,342,928 KiB peak. Used only after registration it reaches 0.72842 px
but leaves trajectory at 0.38999 m RMSE / 0.76725 m p95. Moving those disjoint
tracks into incremental triangulation is not safe without a bootstrap gate.
It improves the 1,010-image component from 0.151 m to 0.065 m RMSE, but the
dominant component regresses to 0.716 m. Freezing the original base seed
frames (2943 and 4622) still leaves the dominant component at 0.696 m, proving
that seed selection is not the cause; the adjacent SIFT cycles themselves are
biased in the repeated corridor. Both incremental variants were removed.

A frozen-base two-pass screen was implemented and rejected. It used base-only
poses, robust triangulation at the mapper's 4 px gate, and required temporal
cycle tracks to contain stereo support in two frames before replay. It still
accepted 25,145 tracks / 337,256 observations in the large component and
3,089 / 38,185 in the small component. Registration fell to 9,994 images;
although reprojection was 0.71458 px, mapper time doubled to 625.10 s and
RMSE/p95 failed catastrophically at 1.2479/2.1591 m. The experimental API and
CLI were removed. Reprojection against the trajectory that is being corrected
is a circular gate: self-consistent corridor aliases satisfy the same erroneous
geometry and then strengthen it during replay.

The next implementation boundary is therefore independent correspondence
identity, not another pose, BA, or reprojection gate. First require an exact
four-edge stereo-temporal quadrilateral: left/right stereo matches at both
timestamps and temporal matches on both sensors must close on the same four
features. Build only these four-observation tracks, deduplicate them
deterministically, cap work per adjacent frame pair, and publish no union-chain
extensions. This is a stricter version of the already bounded cycle machinery
and remains linear in verified sparse observations. If its coverage or quality
is insufficient, add a forward-backward photometric image-tracking check; that
signal is independent of both SIFT descriptor identity and the reconstructed
trajectory. COLMAP's robust triangulation remains the publication gate after,
not a substitute for, this independent correspondence gate. GT remains
unavailable until mapping completes.

The first exact-quadrilateral 10k replay recovered 38,740 independent tracks,
each with exactly four observations. It preserved 9,998 registrations, ran in
244.15 s at 1,276,728 KiB peak RSS, and reached 0.77529 px. Post-mapping score
was 0.39072 m RMSE / 0.77367 m p95: bounded and much safer than unvalidated
union chains, but below the existing dense champion and still outside COLMAP's
trajectory gate. Post-registration use is therefore rejected. The remaining
structural A/B is to insert only these quadrilaterals after the original seed
has been fixed, so they can affect growth without selecting the seed. If that
arm still introduces trajectory bias, descriptor-cycle identity has reached
its evidence limit and the next implementation is forward-backward
photometric tracking on the original images.

The final structural arm inserted exact quadrilaterals only after the frozen
base seed. It preserved 9,998 registrations and stayed bounded at 1,277,280
KiB, but reproduced the earlier failure pattern: the 1,010-image component
improved to 0.0705 m RMSE while the dominant component regressed to 0.7579 m.
Aggregate RMSE/p95 were 0.71594/1.60465 m. Its temporary seed-admission API was
removed. This is decisive evidence that SIFT descriptor agreement, even over
all four edges of a stereo-temporal quadrilateral, is not an independent
identity test in this repeated corridor. The next gate must track the original
pixel patches forward and backward between adjacent frames and intersect that
photometric result with the exact quadrilateral endpoints before incremental
publication. No reconstructed pose may enter that acceptance test.

That independent photometric A/B is now complete. A pose-free pyramidal LK
screen checked both temporal edges of all 65,747 exact quadrilaterals against
their descriptor endpoints, with forward/backward consistency. It accepted
59,948 tracks (91.18%) in 107.21 s at 148,400 KiB; every 500-frame bin retained
at least 82.3% coverage. Filtering the deferred overlay by those full
four-observation identities reduced it from 967,670 to 239,792 matches and ran
the mapper in 237.26 s at 1,263,008 KiB. The model reached 0.77286 px and
0.39043/0.76448 m RMSE/p95, but registration fell from 9,998 to 9,980 images.
That first mapper arm filtered the same suffix used by deferred registration,
so its 18-image loss is now classified as a methodological confound rather
than track-quality evidence. Registration and retriangulation tails are now
separate: original pairs remain available to PnP, while only a disjoint
filtered copy enters the track builder.

The corrected 32–128-frame test found 81,045 exact wide quadrilaterals and
accepted 36,981 (45.63%) with direct pose-free LK, including 6,078 in frames
2,500–3,499. It preserved 9,998 registrations and stayed bounded at 1,312,132
KiB, but 30,355 retriangulated tracks scored 0.39004/0.77557 m RMSE/p95, behind
the existing 0.38754/0.76836 m visloc arm and COLMAP. Independent wide support
therefore exists, but adding it as ordinary reprojection tracks does not
correct inter-segment gauge. The next use must be an explicit robust relative
metric factor or scale factor, not another track-topology variation. Frozen details are in
[`m8-openloris-photometric-quadrilateral-ab.json`](../benchmarks/electro/m8-openloris-photometric-quadrilateral-ab.json).

An independent metric-motion follow-up also rules out simply turning those
adjacent tracks into odometry factors. Local stereo 3-D points yielded 3,767
adjacent-frame rigid fits, but median translation-direction cosine against the
reconstructed motion was only 0.345. Just 32 fits agreed within 1 degree and
0.99 direction cosine, with none in frames 3,000–3,499. The physical stereo
baseline is too small relative to adjacent motion/depth noise. A viable metric
factor must accumulate photometric tracks to a longer temporal baseline before
fitting 3-D–3-D motion.

That longer-baseline 32–128-frame factor experiment is also complete. With a
loose 0.5-degree stereo triangulation-angle gate it produced 468 factors, but
translation had a 0.773 m p95 and 4.332 m maximum: the late factors are not
metrically observable. Tightening the independent calibration-only gate to 2
degrees reduced the median fit residual to 0.0136 m and retained 200 factors,
but every factor began before frame 971 (195 in frames 0–499 and five in
500–999). It therefore supplies no constraint in the 2,500–3,499 drift region.
No factor is injected into mapping: clean local fits without spatial coverage
cannot close the trajectory gap, while lowering the observability gate admits
unstable direction and scale.

A one-sided stereo-PnP formulation was checked before closing that path. It
triangulates metric points at only one endpoint, solves the opposite rig pose
independently through both target sensors, repeats in reverse, and requires the
two sensor poses to agree within 0.5 degrees / 2 cm plus forward/reverse closure
within 2 degrees / 10 cm. This avoids the two-depth noise of 3-D–3-D fitting.
Of 2,029 candidate frame pairs, 398 passed every gate in 7.44 s at 172,984 KiB.
However, 325 were in frames 0–499, 64 in 500–999, eight in 1,000–1,499, and
one in 2,000–2,499; none covered frames 2,500–3,499. The formulation is
geometrically valid and synthetic-tested, but it cannot constrain the measured
drift interval, so no pose factor is injected.

The corresponding final-track audit changes the implementation target. On the
same 8,988-image dominant model, COLMAP publishes 103,611 points with 1,800,787
observations, while the current visloc champion publishes 327,036 points with
1,409,420 observations. In every 500-frame bin COLMAP's median track length is
5–14 observations. Visloc's is only 2–3; its median temporal span is zero in
six bins because most new landmarks contain only the synchronized stereo pair.
COLMAP does not obtain its late accuracy from a dense set of explicit metric
factors: its stereo-supported long-range fits are concentrated in the first
1,000 frames. The actionable difference is therefore persistent landmark
identity. The next controlled arm is a bounded post-registration equivalent of
COLMAP `MergeTracks`: only disjoint positioned tracks connected by a verified
match may merge, with symmetric cross-reprojection, 3-D distance, duplicate-
image, and deterministic pass gates before the existing refinement cycle.

That arm is now closed as a negative result. It was byte-identical to the
control at 1k because no applicable fragments existed. At 10k, a 0.05 m gate
merged only 34 dominant-component fragments and changed RMSE/p95/reprojection
from 0.38754 m / 0.76836 m / 0.681998 px to 0.38767 m / 0.77511 m /
0.682012 px. Expanding the preselected geometry gate to 0.25 m produced only
53 merges and still scored 0.38765 m / 0.77508 m / 0.682109 px. Both arms
kept 9,998 registrations and about 1.33 GiB peak RSS, but both failed the
quality gate. The temporary mapper option was removed. The small increase from
34 to 53 merges shows that 3-D distance is not the main bottleneck: existing
verified edges mostly meet already-owned or same-image-conflicting topology.
The next implementation must preserve supported alternatives before conflict
resolution, or form persistent landmarks during incremental triangulation,
instead of merging final map points. Frozen results are in
[`m8-openloris-positioned-track-merge-ab.json`](../benchmarks/electro/m8-openloris-positioned-track-merge-ab.json).

The remaining difference from COLMAP point creation was then tested directly.
The existing `robust-triangulation` path still initialized from only the
widest ray pair; a bounded replacement scored at most 120 hypotheses from 16
deterministically sampled rays, matching COLMAP's consensus-first create
semantics without image-quadratic state. It improved the 1k robust arm to
0.02688/0.04079 m RMSE/p95 at 0.75677 px. At 10k it preserved 9,998
registrations and reached 0.73311 px at 1,424,996 KiB, but the dominant
component failed catastrophically at 1.38782/2.68913 m RMSE/p95. The small
component simultaneously improved to 0.06277/0.13274 m, confirming that the
estimator works where correspondence identity is unambiguous. In the repeated
corridor, however, maximizing reprojection consensus against the trajectory
being corrected is circular and strengthens the wrong alias. The experimental
candidate selection was removed; frozen data are in
[`m8-openloris-robust-triangulation-hypothesis-ab.json`](../benchmarks/electro/m8-openloris-robust-triangulation-hypothesis-ab.json).

The next boundary now has a measured Phase 1 foundation. A rig-local,
read-only CSR flattens every feature observation to a `u32` id and stores
sorted, duplicate-free neighbours behind `u64` row offsets. Unlike the
existing general COLMAP-compatibility graph, construction never allocates a
`Vec` per keypoint. On the champion's exact 10k mapping prefix it represents
5,085,131 observations and 3,337,520 undirected verified edges in 67,461,224
persistent bytes (64.34 MiB); two independent builds produced digest
`4c81124f200f3a8c`. CSR construction took 0.311/0.300 s, and the complete
snapshot/feature-load/preview process peaked at 589,520 KiB. The explicit
`--preview-rig-correspondence-csr` path is diagnostic-only, so the default
mapper does not build or consume the graph yet. This proves the sparse state
bound, not trajectory quality. Phase 2 must use the CSR during registration
to create seed stereo points and continue/create tracks only when frames
become registered; pre-unioning the entire graph would preserve the failure
being addressed. Frozen details are in
[`m8-openloris-rig-correspondence-csr-preview.json`](../benchmarks/electro/m8-openloris-rig-correspondence-csr-preview.json).

Phase 2 now exercises that CSR behind the default-off
`--dynamic-correspondence-tracking` mapper path. The first direct-stereo-only
bootstrap exposed only three seed frames and selected frame 26; a bounded
bootstrap fix now inspects legacy component membership only long enough to
recover the historical support-descending metric seed order, copies the
selected frame's 96 stereo pairs, and drops the global tracks before dynamic
growth. This restores the frozen frame-8 seed and registers 1,000/1,000 images
in one model. Every one of the 361,170 observation rows is activated once and
3,210,348 directed CSR edges are visited without an image-pair matrix.

The quality gate nevertheless rejects the arm. Its best result is 0.02910 m
RMSE / 0.04368 m p95 / 0.74328 px at 171,732 KiB and 36.54 mapper seconds,
versus the frozen visloc control's 0.02670 m / 0.04135 m / 0.72068 px and the
official COLMAP control's 0.02797 m / 0.04227 m / 0.81021 px. Registration and
resource bounds pass, but trajectory parity does not. A single-edge fragment
merge reduced topology fragmentation but worsened RMSE to 0.02941 m and was
removed. Metric-only BA, disabled local BA, and twice-frequent local BA also
failed. Requiring two currently owned neighbours rejected 51,197 continuation
attempts and slightly improved p95 to 0.04348 m, but regressed RMSE to 0.02919
m; that temporary option was removed as well. Requiring a closed
correspondence triangle rejected 69,393 continuations and improved runtime,
RSS, and reprojection, but still regressed RMSE/p95 to 0.02913/0.04372 m and
was removed. The dynamic path therefore remains a diagnostic and is not
promoted to 2.5k or 10k. Raw multiplicity and appearance-cycle topology are
both closed as correction sources. The next diagnostic must locate the first
PnP/local-BA pose transition that diverges from the frozen legacy mapper,
without consulting GT during mapping. Exact counters, hashes, and A/Bs are frozen in
[`m8-openloris-dynamic-correspondence-1k.json`](../benchmarks/electro/m8-openloris-dynamic-correspondence-1k.json).

That registration transition is now traced without GT. Both paths select frame
8 with 88 landmarks, but their first PnP choice already differs: legacy chooses
frame 58 with 158/158 support/inliers, while dynamic chooses frame 12 with
130/130. Across the 500-frame run, 347 frames occupy a different registration
order; same-frame camera centres first differ by more than 1 mm at dynamic
order 12 and by more than 10 mm at order 100. The mean and maximum final
same-frame deltas are 0.00765 m and 0.05337 m. Registration scheduling therefore
diverges before substantial pose drift. The bounded next A/B will expose the
full legacy components of the selected seed tracks only as owner-neutral initial
PnP proposals, then discard that bootstrap state before CSR-driven growth. It
must not import those observations into dynamic ownership or track topology.
Frozen trace hashes and comparisons are in
[`m8-openloris-dynamic-registration-trace-1k.json`](../benchmarks/electro/m8-openloris-dynamic-registration-trace-1k.json).

The first owner-neutral proposal A/B confirms but does not close that diagnosis.
Exposing all 35,824 observations from the selected 96 seed components restores
the exact legacy first PnP choice (frame 58, 158/158 support/inliers, identical
pose). The paths diverge again at registration order two because tracks newly
positioned after frame 58 still expose direct CSR neighbours only. The final
arm registers 1,000/1,000 at 172,608 KiB in 34.47 s, but regresses to 0.02930 m
RMSE / 0.04408 m p95 / 0.74408 px, so it is rejected. The next A/B generalizes
the proven visibility mechanism: retain a compact observation-to-clean-component
index, and publish each component once when its first dynamic track becomes
positioned. These are still PnP proposals only—no transitive owner or track
topology—and total component scans and storage must remain O(observations).

That streaming extension is also rejected. It publishes 4,760 clean components
and scans 142,125 component observations exactly once, remaining bounded at
175,548 KiB, but order two still diverges and the final result regresses to
0.02940 m RMSE / 0.04400 m p95 / 0.74609 px in 37.51 s. A component can contain
two newly registered views connected only through unregistered CSR nodes;
legacy pre-union membership triangulates it, while no positioned dynamic
fragment exists to source the streaming proposal. Thus visibility alone is not
enough: a further dynamic experiment would require an owner-neutral shadow
landmark per clean component, triangulated only after it has sufficient
registered rays. It must remain separate from dynamic ownership and pass 1k
before promotion. The failed streaming option and compact index are removed.

That final shadow-landmark A/B is now closed and removed as well. The bounded
prototype stored one compact component state and retried triangulation only at
power-of-two registered support (2, 4, 8, ...), so persistent state remained
`O(observations + components + CSR)` with no image-pair matrix. It registered
1,000/1,000 images and owner-neutral shadow inliers did bridge unregistered CSR
nodes, but the 3,627 published shadows required 15,909 triangulation attempts.
Mapper time rose from the frozen visloc control's 16.50 s to 35.04 s and peak
RSS rose from 142,044 to 187,852 KiB. More importantly, RMSE/p95 regressed to
0.03183/0.04750 m, worse than both visloc (0.02670/0.04135 m) and COLMAP
(0.02797/0.04227 m), despite 0.72586 px reprojection remaining better than
COLMAP. Seed-time shadow publication also changed the first PnP from legacy
frame 58 to frame 74, so it destroyed the scheduling equivalence restored by
the narrower seed-component arm. Dynamic component widening is therefore
exhausted as the next correction source. The next bounded experiment is the
remaining conflict-resolution boundary: preserve only a small fixed number of
geometrically supported track alternatives before same-image ownership becomes
irreversible, and require the frozen 1k trajectory and runtime gates before a
larger tier. The GT-independent paired-pose-jump path is not reopened here: its
real 10k A/B already regressed RMSE/p95 versus isolated-only despite a positive
offline screen. Exact counters, hashes, and decisions remain in
[`m8-openloris-dynamic-correspondence-1k.json`](../benchmarks/electro/m8-openloris-dynamic-correspondence-1k.json).

The frozen-1k PairConfidence conflict-region recovery A/B reached BA with
3,778,059 observations and cost 6,320,355.944471336 -> 3,883,862.718358550
after 864 iterations (`converged=false`), using 88.13 s user / 0.19 s system /
88.34 s wall and 233,740 KiB peak RSS, but export failed with
`InvalidExportInput("keypoint (0, 3) belongs to multiple exported landmarks")`.
Successful components had already been emitted, so additive recovery violated
the observation-ownership invariant; exporter rejection is correct and the
experiment is rejected/reverted. The only next arm is a bounded,
mutually-exclusive partition or hypothesis selection before ownership, never
duplicate observation assignment. Details are frozen in
[`m8-openloris-conflict-region-recovery-ab.json`](../benchmarks/electro/m8-openloris-conflict-region-recovery-ab.json).

The diagnostic-only PairConfidence topology preview is bounded and leaves the
mapper untouched: at 1k it processed 1,605,174 correspondences in 0.535056 s
(1.05 s wall, 233,428 KiB) and found 1,075 regions, while the 10k mapping
prefix processed 3,337,520 correspondences in 2.770818 s (8.03 s wall,
832,196 KiB) and found 4,341 regions. The 10k histogram includes regions of
9,044, 7,799, 3,894, and 2,666 successful components, so a region-K=2 recovery
policy is rejected. The default PairConfidence mapper remains byte-identical
to the f70d77f control (1,000 images, 6,654 tracks, 268,586 observations,
0.755513995 px reprojection); timing is recorded but is not a decision gate.
The next arm must resolve ownership before publication: for each reference
row, batch registered unowned neighbors and follow COLMAP's
`IncrementalTriangulator::Find`/`Create` pattern with robust multi-view
triangulation, claim only inlier observations exclusively, cap row degree and
hypotheses, and skip point-point merges. Full stats, histogram, commands, and
the official COLMAP references are in
[`m8-openloris-pair-confidence-conflict-topology.json`](../benchmarks/electro/m8-openloris-pair-confidence-conflict-topology.json).

The owner-before-Create follow-up used the bounded, COLMAP
`IncrementalTriangulator::Find`/`Create`-inspired arm: each registered CSR
reference row scored at most 32 neighbours and 128 ray-pair hypotheses, then
claimed only one mutually-exclusive inlier set. It retained all 1,000 images
at 1k, but regressed reprojection/RMSE/p95 from 0.743283945 / 0.029099636 /
0.043678871 to 0.762919353 / 0.029501087 / 0.044415400, so the arm is
rejected and not promoted to 2.5k. The uncommitted source and CLI changes
were reverted; future work must preserve mutually-exclusive hypotheses before
ownership, never duplicate an observation across output tracks. Exact
counters, commands, and model hashes are frozen in
[`m8-openloris-dynamic-batched-create-ab.json`](../benchmarks/electro/m8-openloris-dynamic-batched-create-ab.json).

The current pure-visual sparse Schur solver also reopened the historical
global-BA memory question. A full 500-frame solve passed the frozen 1k gate at
269,052 KiB, with 0.02703/0.04217 m RMSE/p95. On 10k, however, repeating a
global solve after deferred track growth crossed the hard 2 GiB limit and was
terminated at 2,320,792 KiB. Running global BA only on the established base and
returning post-deferred refinement to 60-frame windows completed at 1,716,224
KiB. It preserved 9,998 registrations and improved reprojection to 0.66658 px,
but regressed trajectory to 0.39826/0.81975 m RMSE/p95. The temporary schedule
control was removed. This closes both local and global reprojection-only BA as
the correction source; details are frozen in
[`m8-openloris-sparse-global-ba-ab.json`](../benchmarks/electro/m8-openloris-sparse-global-ba-ab.json).

A final COLMAP-style growth-schedule check ran full solves when registration
reached 500, 1,000, 2,000, and 4,000 frames. The frozen 1k result was identical
to its control, but 10k registered only 9,992 images and regressed sharply to
0.81762/1.35665 m RMSE/p95, including 12.74 m outliers. It remained within the
memory target at 1,621,132 KiB and finished mapping in 446.20 s, so resource
use is not the rejection cause. Moving the active gauge during growth changes
subsequent PnP outcomes and amplifies repeated-corridor errors. The opt-in
growth scheduling API and CLI were removed.

### M8.1 Official control and scoring infrastructure

1. Finish the official COLMAP 1k pilot and preserve its immutable inputs,
   commands, phase logs, database statistics, model hashes, and failure/warning
   state.
2. Run the corrected, calibrated stereo-rig same-candidate v3 control at 1k,
   then 10k. The v2 no-rig result is retained only as a diagnostic because it
   assigns an independent pose to each synchronized camera exposure.
3. Score every visloc and COLMAP model with the same official-GT scorer. For
   disconnected outputs, align each independent gauge separately and publish
   both per-component and image-weighted results.
4. Establish the native COLMAP control without exhaustive matching and freeze
   its command before viewing the 10k result.

### M8.2 Diagnose by first divergence

For each tier, compare in this order:

1. candidate graph connectivity and cross-time/cross-camera coverage;
2. raw matches, verified pairs, inliers, and component sizes;
3. track conflicts, track length, observations, and triangulation angle;
4. registration order and failed PnP attempts;
5. local/global BA residuals and official-GT drift by time segment.

The first divergence selects the implementation. A bridge implementation is
promoted only if verified cross-component edges and final quality improve. A BA
change is promoted only if it reduces trajectory error without losing
registration or exceeding reprojection/resource gates.

### M8.3 Bounded quality implementations

Implement and A/B in this order, stopping once the outcome gate is met:

1. **Calibrated rig frames.** Group the synchronized T265 images into one
   two-camera rig, optimize one frame pose per timestamp, and hold the official
   camera-from-rig transform fixed. Run the same calibrated-rig arm in COLMAP.
   The no-rig arm remains a diagnostic, not the accuracy target: M7's recovered
   same-time baseline is about 20 times the physical baseline and cannot be
   repaired by one global Sim(3). Frame growth must pool both sensors in a
   generalized absolute-pose RANSAC/refinement and then continue tracks for the
   entire frame; running two independent central-camera PnPs and selecting one
   is not the promoted implementation. This follows COLMAP's
   `RegisterNextGeneralFrame` contract and the non-central absolute-pose model
   implemented by OpenGV.
2. **Drift repair.** Revisit registration/BA after the rig scale is coherent.
   A/B global refinement frequency, stable scale gauge, retriangulation, and
   long-baseline constraints. GT poses may never enter mapping.
3. **Component-aware ANN retrieval.** Stream one global descriptor per image,
   query a bounded ANN index, retain mutual or bidirectionally supported
   cross-component candidates, and apply per-image/per-component quotas. The
   total candidate envelope is `O(NK)` and explicitly capped.
4. **Verified bridge admission.** Match only the bounded additions, run the same
   geometric verifier, and merge only verified edges. Do not materialize the
   Cartesian product of components and do not use post-hoc Sim(3) to merge
   models with no common registered images.
5. **Track/triangulation repair.** Preserve geometrically supported alternatives
   when a same-image conflict would otherwise discard an entire chain. Bound
   alternatives per feature/component and retain deterministic ordering.

### M8 exit gate

At 10k, visloc-rs must be no worse than the stronger official COLMAP control on:

- unique registered fraction;
- largest-model registered fraction and number of independent gauges;
- official-GT ATE RMSE and p95 under the same alignment convention;
- observation-weighted mean reprojection error.

The result must also keep peak RSS at or below 2 GiB and show no `N^2` state.
Passing registration alone or reprojection alone does not complete M8.

### Rig-BATA quality experiment rejected at 1k (2026-09-04)

A bounded, fixed-rotation rig-BATA prototype tested GLOMAP's `ONLY_POINTS`
bearing residual after the final visual refinement. The detached solver fixed
the seed centre, used positive inverse depths and Cauchy IRLS, and committed
only when finite state, cheirality, published support, and pixel reprojection
all remained non-regressing. Its state was linear in positioned observations,
tracks, and frames; it formed no frame-pair matrix.

The first arm used only the 33 metric-anchored tracks and underconstrained 378
movable frame centres: one/two rounds worsened pixel reprojection, while
four/eight rounds created new negative-depth observations. The corrected arm
used all 2,753 positioned tracks (29,751 observations) in the seed-connected
component while requiring a real metric anchor. It improved the all-positioned
mean from 5.224 px to as low as 3.011 px and created no negative-depth
observations, but every one/two/four/eight-round candidate reduced the set of
published inlier observations or tracks. The transaction therefore rolled
back every arm, preserving the frozen 1k model byte-for-byte. The 10k run was
skipped by the 1k promotion gate, and the prototype was removed rather than
retained as an ineffective option. Full commands, hashes, gates, and variants
are frozen in
[`m8-openloris-rig-bata-ab.json`](../benchmarks/electro/m8-openloris-rig-bata-ab.json).

### Prefix-stable metric seed proxy rejected at 10k (2026-09-04)

COLMAP's successful two-view initialization stays near the same temporal
prefix at both tiers (frames 445/477 at 1k and 446/462 at 10k), while the
visloc metric-support seed moves from frame 26 at the frozen 1k control to
frame 2943 in the 10k champion. A default-off proxy therefore restricted the
existing support-sorted seed candidates to the first 500 local synchronized
frames. It was byte-identical to the frozen 1k champion, including all three
model hashes, so the small-tier gate passed exactly.

On the otherwise frozen 10k champion command, however, the proxy selected
frame 38, registered only 9,996/10,000 images, and reduced the largest model
from 4,494 to 4,493 frames. Reprojection remained low at 0.685862 px and peak
RSS stayed bounded at 1,393,748 KiB, but post-mapping ATE regressed from
0.38754/0.76836 m RMSE/p95 to 0.94517/1.47600 m. The option and tests were
removed. This closes tier-prefix restriction as a seed policy; it does not
test COLMAP's actual two-view initialization, which uses temporal parallax
rather than a one-frame stereo-support maximum. Exact inputs, hashes, and
component scores are frozen in
[`m8-openloris-prefix-stable-seed-ab.json`](../benchmarks/electro/m8-openloris-prefix-stable-seed-ab.json).

### Bounded two-frame metric seed rejected at frozen 1k (2026-09-04)

A default-off bounded two-frame seed was tested as a closer proxy for COLMAP's
temporal-parallax initialization. Verified edges were scanned once into sparse
canonical temporal frame pairs, with deterministic match-count ranking and a
32-probe cap. Each source endpoint reused the existing same-frame stereo
triangulation and each target was solved with the existing
`GeneralizedPnPRansac`; no new solver, dense frame-pair table, or trial-count
track clones were introduced. The state bound was `O(E)` for candidate
aggregation plus `O(F + observations)` mapper state. The dynamic-correspondence
mode is incompatible with this experiment because it cannot provide the legacy
metric source tracks required by the probe.

The frozen 1k control registered `500/500` frames (`1000/1000` images), with
2,658 tracks, 27,716 observations, `0.671637382 px` reprojection error,
`3.254479 s` mapper time, `3.70 s` wall time, and `81,928 KiB` peak RSS. Its
seed was frame 26 and all three model files remained byte-identical to the
champion (`cameras.txt` `65e29cd8...`, `images.txt` `937098610...`, and
`points3D.txt` `11eb71b6...`). The candidate considered 2,234 pairs and probed
32; source frame 42 to target frame 26 was the sole accepted probe, with 17/17
inliers, `0.062470°` median temporal parallax, and a `0.002086812 m` metric
baseline. It retained `500/500` frames and `1000/1000` images, but changed the
map to 2,806 tracks / 28,421 observations, `0.770560867 px`, `4.021271 s`,
`4.53 s`, and `82,384 KiB`. Its score was `0.11912569450660591 m` RMSE,
`0.2160373033773416 m` p95, `0.3669530777398771 m` max,
`0.08574868160134173 m` median, and Sim(3) scale `1.605404874928131`
(`score_sha256=c6b9140a491a05fbd7887469304f067a575a2dc4dccd727f142eef0c1caf3780`).

This is a clear quality failure against the frozen 1k champion
(`0.02269532080131782 m` RMSE, `0.03777877644562742 m` p95, and
`0.671637382 px`): registration passed, but all three quality checks failed.
The implementation was removed, and the required test evidence remains
`620 passed, 0 failed, 7 ignored` with zero prototype symbols after revert.
The 10k run was skipped by the failed 1k promotion gate. This experiment is a
stereo-PnP proxy and does not refute COLMAP GR6P/GR8P itself; this route is
closed (`stereo-PnP proxyでありCOLMAP GR6P/GR8P自体を否定しないが、この経路は終了`).
The complete record, including the score digest and exact default-off hashes,
is frozen in
[`m8-openloris-two-frame-metric-seed-ab.json`](../benchmarks/electro/m8-openloris-two-frame-metric-seed-ab.json).

## M9 — make the quality champion faster than COLMAP

Freeze the M8 quality champion before performance edits.

### Exact-output performance work accepted during M8 diagnosis (2026-09-03)

- The persistent match worker now has an opt-in descriptor-streaming mode with
  a deterministic 65,536-row LRU. On the frozen 1k / 7,000-pair / 219-shard
  control, all 219 verified snapshots are byte-identical to the fully resident
  worker. The single-run wall/RSS result was 118.20 s / 57,600 KiB versus
  125.55 s / 158,056 KiB resident. These runs shared ongoing extraction load,
  so they accept exactness and the state bound; final speed claims still
  require unloaded three-run medians at the promoted 10k configuration.
- `benchmark_electro.py` passes this mode through the normal plan, atomic shard
  publication, restart recovery, index-hash validation, and merge path. Dense
  features must generate a new feature manifest and worker plan; a plan bound
  to the old feature bank is rejected rather than reused.
- Snapshot merging now decodes one shard at a time, spools only accepted edge
  keys, and publishes the twice-validated result atomically. On the dense 2.5k
  control (547 shards), its output SHA-256 is byte-identical to the former
  owned merge while peak RSS fell from 429,460 KiB to 7,740 KiB. Wall time was
  6.76 s versus 5.65 s under the concurrent 10k extraction load; final speed
  reporting remains reserved for the unloaded 10k decision run.
- Candidate shards now use a source- and image-order-hash-bound compact v2
  envelope containing pair indices only. The canonical image list lives once
  in the source/index/worker plan instead of once per shard, removing the
  previous `O(images × shards)` disk and parse cost. Legacy v1 shards remain
  readable, while the next prepare migrates them to v2. A synthetic 100,000
  image test, corruption checks, image-count/hash mismatch checks, and both
  v1/v2 worker-plan parser tests pass; the full M10 100k replay is still due.
- Generalized-rig frontier triangulation now scans each track linearly but caps
  widest-baseline pair search to a deterministic 1,024-ray sample. On the 10k
  champion, both component `cameras.txt`, `images.txt`, and `points3D.txt`
  outputs are byte-identical to the uncapped result: 9,998/10,000 registered,
  1.007267671 px mean reprojection, and 838,964 KiB peak RSS. The only metadata
  difference is the newly reported deferred-interpolation counter.

The raw 1k streamed-worker A/B is frozen in
[`m9-streamed-match-worker-1k-ab.json`](../benchmarks/electro/m9-streamed-match-worker-1k-ab.json).
The dense 2.5k merge A/B is frozen in
[`m9-bounded-snapshot-merge-2500-ab.json`](../benchmarks/electro/m9-bounded-snapshot-merge-2500-ab.json).
Neither result closes M8: the current 10k trajectory and reprojection values
remain worse than COLMAP, so the dense same-candidate quality arm is still in
progress.

1. Capture phase-level CPU, wall, allocations/RSS, pair counts, and artifact
   hashes. Optimize the measured top two phases first.
2. Prefer exact-output changes: persistent feature/match workers, bounded
   queues, descriptor lifetime reduction, mmap/streamed reads, top-k selection
   without full sorting, sparse BA assembly reuse, and deterministic component
   scheduling.
3. Every behavioral ANN or numeric change gets recall/quality A/B against the
   frozen champion. Hash equality is required only for declared exact-output
   changes; behavioral changes must pass all M8 quality gates.
4. Record live counters for resident descriptors, queued pairs, observations,
   tracks, and BA nonzeros. Assert caps derived from `N`, `K`, and shard size;
   a hidden dense `N x N` allocation is a test failure.

### M9 exit gate

- median mapper-only wall time is lower than official COLMAP;
- median native end-to-end wall time is lower than official COLMAP;
- peak RSS is lower than official COLMAP and at most 2 GiB at 10k;
- M8 registration, trajectory, and reprojection gates remain green.

## M10 — scale, recovery, evidence, and release closure

1. Run the frozen champion at 1k, 2.5k, 5k, and 10k. No tier may hide a
   registration, trajectory, reprojection, time, or memory regression.
2. Repeat the decision run three times. Candidate/snapshot/model hashes must be
   deterministic where promised; otherwise the variance and cause must be
   explicit and quality-equivalent.
3. Inject interruption after candidate, match-shard, merged-snapshot, and mapper
   checkpoints. Resume without recomputing valid artifacts and reproduce the
   uninterrupted hashes.
4. Re-run deterministic 10k and 100k I/O stress, corruption fail-closed checks,
   and bounded-state assertions.
5. Freeze machine/container identity, commands, hashes, phase ledgers, scores,
   and raw logs in benchmark evidence plus a human-readable report.
6. Put the COLMAP head-to-head near the top of README with a real-model GIF or
   image and a compact comparison table. Do not add a memory before/after
   optimization table; report only the final visloc-vs-COLMAP resource result.
7. Land as reviewable PRs: control/scoring, quality, performance, then closure
   docs. Run full CI, squash-merge each PR, delete merged remote/local topic
   branches, and finish on clean `main`.

## Stop conditions

- Do not launch a larger tier after an input/hash/calibration mismatch.
- Stop a run before swap thrash if RSS exceeds its declared cap; retain the DNF
  evidence and fix the state bound at a smaller tier.
- Do not claim trajectory parity from reprojection error or visual appearance.
- Do not call disconnected models one reconstruction unless they share
  registered images and are validly merged/refined.
- Do not tune against GT inside candidate, match, or map execution.

## Primary references

- [COLMAP FAQ: incremental/global/hierarchical SfM, known intrinsics, model
  merging, and BA scaling](https://colmap.github.io/faq.html)
- [COLMAP command-line interface](https://colmap.github.io/cli.html)
- [COLMAP incremental mapper rig registration source](https://github.com/colmap/colmap/blob/main/src/colmap/sfm/incremental_mapper.cc)
- [COLMAP incremental triangulator and robust track creation](https://github.com/colmap/colmap/blob/main/src/colmap/sfm/incremental_triangulator.cc)
- [ORB-SLAM3 paper: local covisibility and multi-map SLAM](https://arxiv.org/abs/2007.11898)
- [OpenLORIS-Scene dataset and GT interpolation
  guidance](https://lifelong-robotic-vision.github.io/dataset/scene.html)
- [OpenLORIS-Scene tools](https://github.com/lifelong-robotic-vision/openloris-scene-tools)
- [OpenGV non-central absolute pose and RANSAC implementations](https://github.com/laurentkneip/opengv)
- [Faiss research foundations: inverted files, PQ, and
  HNSW](https://github.com/facebookresearch/faiss/wiki/)
