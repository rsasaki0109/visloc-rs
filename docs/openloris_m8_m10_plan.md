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
   repaired by one global Sim(3).
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

## M9 — make the quality champion faster than COLMAP

Freeze the M8 quality champion before performance edits.

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
- [OpenLORIS-Scene dataset and GT interpolation
  guidance](https://lifelong-robotic-vision.github.io/dataset/scene.html)
- [OpenLORIS-Scene tools](https://github.com/lifelong-robotic-vision/openloris-scene-tools)
- [Faiss research foundations: inverted files, PQ, and
  HNSW](https://github.com/facebookresearch/faiss/wiki/)
