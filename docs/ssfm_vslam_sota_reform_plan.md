# Sequential SfM + Visual SLAM SOTA reform plan

Status: active proposal, 2026-07-22

This plan turns the current COLMAP-beating and DPVO diagnostic work into a
claimable state-of-the-art program. It supplements
`visual_slam_sequential_sfm_plan.md`; completed experiments and detailed
historical verdicts remain there.

The headline remains **calibrated monocular RGB, one fixed configuration per
dataset family**. Stereo, IMU, metric-depth, and uncalibrated modes get separate
tables and may not be used to rescue the monocular headline.

## 1. What “SOTA” will mean

There are three deliberately separate levels:

1. **Internal frontier:** better than every reproduced baseline on the same
   images, hardware, sensor assumptions, evaluator, and resource budget.
2. **Benchmark frontier:** best or Pareto-best result on a recognized public
   protocol, including failures, runtime, RAM, and VRAM rather than ATE alone.
3. **Claimable SOTA:** the benchmark-frontier result survives a frozen-config
   held-out suite, at least three runs where nondeterminism matters, independent
   reproduction from archived commands, and release of code, model hashes,
   trajectories/reconstructions, and failure cases.

“SOTA” is never inferred from one EuRoC crop or one COLMAP comparison. A result
may instead be named precisely: best accuracy, best runtime, best robustness,
or Pareto SOTA.

### 1.1 Visual SLAM target

The primary class is online, calibrated, monocular RGB SLAM. The scorecard is:

- all 11 EuRoC sequences, Sim(3) ATE, success rate, RPE, tracking coverage,
  scale profile, Hz, p95 latency, peak RAM/VRAM;
- ETH3D SLAM training for development and an official test submission for the
  robustness claim, using its AUC-style evaluation rather than cherry-picking
  one sequence;
- KITTI 00-10 as the long-motion and scale-drift guard;
- three fixed seeds where output is nondeterministic, reporting median and
  worst, including catastrophic failures.

The first research target is **all-sequence EuRoC mean ATE <= 0.020 m with
11/11 successful runs**, while sustaining the dataset input rate and avoiding
any scale-cliff event. This is intentionally below the roughly 0.022-0.024 m
calibrated-monocular averages reported for DROID-SLAM, DPV-SLAM(++), and
GO-SLAM in the MASt3R-SLAM evaluation. It is a target floor, not a standing
claim; the baseline table must be refreshed immediately before publication.

### 1.2 Sequential SfM target

The primary class is calibrated monocular ordered video SfM. The scorecard is:

- EuRoC full sequences and ETH3D ordered videos for independent-GT pose,
  registration/completeness, reprojection, tracks, runtime, and memory;
- ORBIT's 100 difficult real-world clips for the in-the-wild camera-pose claim;
- unordered ETH3D/South Building/Gerrard Hall only as a regression guard;
- same-machine current baselines: COLMAP incremental and global mapper,
  GLUEMAP, and InstantSfM where their supported platform permits a fair run.
  Published-only results are labelled separately from reproduced results.

Claimable SOTA requires one frozen policy to sit on the Pareto frontier against
the strongest valid baseline on at least three held-out ordered sequences, then
remain frontier-competitive on ORBIT. Winning only against COLMAP is now an
engineering milestone, because current competitors also include global,
feed-forward/global-hybrid, and GPU-native SfM systems.

## 2. Current position and binding gaps

### Sequential SfM

- Fresh held-out MH_04 300f is a genuine three-axis win over COLMAP 4.1:
  300/300, 1.0648 vs 6.5102 cm common-frame Sim(3) ATE, and 1104 vs 2316 s.
- That is still one short crop. The attempted MH_03 2700f run produced no valid
  final model and must be rerun cleanly.
- The current mapper performs too much repeated global work. Full-sequence
  scaling, not local accuracy, is now the first architectural blocker.
- Completeness recovery is still sequence-sensitive on hard ETH3D scenes.

### Visual SLAM

- Mean-pool retrieval plus per-frame queries reaches about 0.99 labelled-loop
  recall on the MH_01 800f microscope.
- Essential-matrix rotation is independently correct on true pre/post-hover
  revisits; long-span DPVO rotation is wrong by roughly 51-108 degrees.
- Both existing scale paths are closed negatives: retained DPVO patch 3D-3D
  alignment and old-patch-3D/current-2D PnP provide no trustworthy bridge.
- Consequently, the loop frontend can find the right place but cannot yet
  observe an independent scale. Accepted scale-bearing loops remain zero.

The common missing primitive is therefore a **fresh, independently optimized
local submap** with persistent multi-view tracks and uncertainty. SSfM needs it
for hierarchical scaling; VSLAM needs two such submaps to measure a loop Sim(3)
without recycling corrupted DPVO depths.

## 3. Architectural reform

### R0 — Stop optimizing demos; establish product boundaries

- Move accepted algorithms out of example binaries into tested library APIs.
- Treat retrieval proposal, geometric verification, metric/scale measurement,
  graph optimization, and state write-back as separate typed stages.
- Every estimate carries provenance, frame convention, covariance or a robust
  uncertainty proxy, and degeneracy classification.
- Keep experimental flags default-off. A mechanism graduates only through its
  full gate, then redundant negative branches are removed or retained solely as
  diagnostics.

Gate: both headline runners are thin orchestration layers, and synthetic
convention/degeneracy tests cover every transform crossing a module boundary.

### R1 — Shared local-submap kernel (highest priority)

Build a reusable `LocalSubmapBuilder` over an ordered keyframe interval:

1. form bounded temporal edges plus a small number of appearance-retrieval
   edges;
2. estimate E/F/H models and retain only geometrically consistent 2D tracks;
3. initialize rotations and translation directions independently of DPVO
   depths;
4. triangulate only tracks with sufficient parallax, cheirality, coverage, and
   leave-one-view-out reprojection support;
5. run gauge-fixed local BA and emit poses, landmarks, observations, quality
   statistics, and uncertainty;
6. support overlapping windows so adjacent submaps share independently
   re-observed landmarks.

This kernel must accept either SSfM's persistent features or VSLAM's retained
SuperPoint observations. It must never read GT and must be able to build the
pre-hover and post-hover sides from pixels even when the live DPVO map is bad.

Gate: on synthetic Sim(3) fixtures and EuRoC non-degenerate windows, recover
rotation within 2 degrees, translation direction within 5 degrees, scale within
5%, and reject pure-rotation/low-parallax windows without emitting scale.

### R2 — Typed submap alignment and transactional merge

Add two legal loop products:

- `RotationOnlyConstraint` for E-supported, scale-unobservable geometry;
- `SubmapSim3Constraint` only when independently built submaps provide robust
  cross-submap 3D support.

Estimate Sim(3) with cross-submap track correspondences, robust sampling, E
rotation consensus, forward/backward reprojection, scale dispersion, and
leave-one-view-out stability. A homography-dominant or low-parallax pair can
never emit scale. Apply the result to a clone first and commit only if residual,
connectivity, tracking/registration, and correction-magnitude gates all pass.

Gate: zero false accepted scale edges on the labelled development suite; every
rejection has a single recorded reason; injected synthetic scale cliffs are
recovered without changing already-correct prefixes.

R1/R2 foundation slice (2026-07-22): the shared library boundary now exists in
`pipelines/slam/src/local_submap.rs` and `submap_alignment.rs`. The
`LocalSubmapBuilder` accepts only camera calibration, source-frame identity,
raw per-frame features, and geometrically verified pairwise matches. It calls
the existing independent incremental-SfM track/triangulation/BA path and emits
source-remapped poses, multi-view landmarks, observations, BA diagnostics, and
an auditable quality record (registration, track length, parallax, camera
spread, reprojection). It validates all image/keypoint indices before the
mapper can index them and reports exactly one deterministic rejection reason.

The alignment half introduces distinct `RotationOnlyConstraint` and
`SubmapSim3Constraint` types. A full scale edge can be constructed only from
one-to-one cross-submap 3D matches that pass deterministic RANSAC, non-collinear
geometry on both sides, refit/reclassification, inlier count/ratio, normalized
residual, independent E-rotation agreement, and leave-one-out log-scale
stability. It neither accepts nor reads a live DPVO pose/depth. Synthetic tests
recover a known 3.5x rotated/translated scale in the presence of four outliers,
reject a 69-degree independent-rotation disagreement, reject collinear point
sets, and prove that rotation-only evidence cannot acquire a scale field.

Verification: full `visloc-slam` library suite with `onnx-inference` passes
448 tests (7 ignored). This is an architectural positive, not an R1/R2 gate
pass; no constraint is wired to the backend and all defaults remain unchanged.

V1a real-data result (2026-07-22):
`examples/dpvo_independent_submap_probe.rs` now reconstructs two fresh local
maps directly from the 800-frame MH_01 SuperPoint dump, with no DPVO pose,
depth, backend, or GT input. The known old/new anchors are arrivals 38/462.
At radius 16, both maps passed the fixed conditioning gates (27/33 and 33/33
registered, 390/467 landmarks, 5.57/6.61-degree median parallax, and
0.75/0.78 patch-pixel mean reprojection). Essential verification returned 35
anchor inliers and 33 were independently triangulated in both maps, but only
7/33 fit one Sim3, so the typed estimator correctly rejected `TooFewInliers`.
Radius 24 improved to 13/31 but still failed the fixed 0.60 consensus gate at
0.419. Radius 32 failed earlier because the new map registered only 38/65
(0.585). A ratio-0.90 control, multi-frame bridge voting, and COLMAP-style
local/global refinement also rejected; none was admitted or written back.
Logs are under
`E:/visloc_archive/dpvo_a3_20260721/independent_submap_probe_*_20260722`.

Verdict: V1a is a useful negative, not a scale-bridge pass. Appearance and
two-view geometry find the revisit, but current sparse feature tracks plus the
incremental initializer do not produce two sufficiently Sim3-rigid local maps.
Per the V1 stop rule, the next R1b slice must test a stronger learned
two-view/multi-view geometry initializer or dense correspondence prior while
retaining the same independent-map and typed acceptance gates.

R1b learned-prior result (2026-07-22): the probe now accepts externally
generated match graphs and indexed anchor point clouds without weakening any
R2 gate. `scripts/export_dpvo_submap_lightglue_matches.py` replayed official
SuperPoint-LightGlue on all 310 temporal edges plus the loop anchor. It raised
anchor Essential support from 35 to 49, but sparse independent-SfM rigidity
fell to 5/32 = 0.156; matcher recall was not the limiting factor.

`scripts/export_vggt_submap_anchor_points.py` then ran the official VGGT-1B
implementation (`facebookresearch/vggt` revision `a288dd0`) independently on
five views per side at offsets -16/-8/0/8/16. On the available GTX 1660 Ti,
fp16 produced all-NaN tokens/depth and was rejected; fp32 was finite and fit at
5.64 GiB peak allocated VRAM. Depth-unprojected anchor geometry improved the
same 49 LightGlue+E correspondences to 14/49 = 0.286. The dedicated point-map
head reached 13/49 = 0.265. Both failed the frozen 0.60 consensus gate, so no
scale was emitted. Artifacts are under
`E:/visloc_archive/dpvo_a3_20260721/{lightglue_submap_r16_20260722,vggt_submap_5view_fp32_20260722,vggt_submap_5view_point_fp32_20260722,independent_submap_probe_r16_vggt5*_20260722}`.

Verdict: R1b is another informative negative. Learned matching increases
two-view support and VGGT roughly doubles the rigid-consensus fraction over the
LightGlue+sparse-SfM arm, but neither makes this severe pre/post-hover view pair
scale-observable at the required reliability. The next initializer comparison
should be a correspondence-grounded 3D model such as MASt3R/MASt3R-SLAM, not a
looser R2 threshold or another sparse matcher sweep.

R1c MASt3R result (2026-07-22):
`scripts/export_mast3r_submap_anchor_points.py` runs the official
`naver/mast3r` implementation (revision `f5209af`, DUSt3R `3cc8c88`) as two
separate pair-conditioned reconstructions. Each inference keeps the relevant
loop anchor as view 1 and uses arrival -8 or +8 only from the same temporal
side; the exported anchor point map is therefore independent of the opposite
side and all live trajectory state. Official ViT-L metric weights produced
finite 512x320 geometry at 3.11 GiB peak VRAM. With the same 49
LightGlue+Essential correspondences, partner -8 yielded 10/49 = 0.204 Sim3
inliers and partner +8 yielded 11/49 = 0.224. Both failed the unchanged 0.60
gate and were rejected. Artifacts are under
`E:/visloc_archive/dpvo_a3_20260721/{mast3r_submap_pair_m8_20260722,mast3r_submap_pair_p8_retry_20260722,independent_submap_probe_r16_mast3r_*_20260722}`.

Verdict: independently inferred metric point maps are not mutually rigid enough
on this extreme foreground/occlusion change. A next learned-scale experiment
must align a metric prior to each same-side multi-view local map separately and
transfer the two independently estimated gauges; directly joint-inferring the
loop pair would put both sides in one model-created gauge and is not by itself
evidence that the tracker/submap scale cliff was measured.

R1d same-side metric scale-transfer result (2026-07-22): the probe now aligns
each sparse local map to its own learned metric geometry before any cross-loop
comparison. A scale is printed only if both same-side Sim3 estimates pass the
unchanged typed gate. MASt3R -8 point maps produced 39/171 = 0.228 old-side
and 20/153 = 0.131 new-side consensus. VGGT depth produced 28/171 = 0.164 and
29/153 = 0.190; VGGT point maps produced 34/171 = 0.199 and 27/153 = 0.176.
All were rejected. A depth-independent VGGT camera-head arm inferred 13 views
per side in fp32 at 5.22 GiB peak VRAM and expressed both trajectories in their
own anchor-camera frames. The old target centres were geometrically degenerate;
the new sparse map registered only 10 of the selected views, below the 12-point
gate. No gauge ratio was emitted. Artifacts are under
`E:/visloc_archive/dpvo_a3_20260721/{same_side_scale_transfer_*,vggt_submap_13view_camera_fp32_20260722,camera_center_scale_transfer_vggt13_20260722}`.

Verdict: direct sparse-map-to-metric-prior transfer is closed for these three
tested geometry products. The sparse local reconstructions are not rigidly
compatible with either learned depth/point geometry, while the short camera
trajectory is not a full-Sim3-observable point set. Do not combine their
minority consensuses or relax the 0.60 gate. Further VSLAM scale work requires
a genuinely jointly optimized dense submap representation (MASt3R-SLAM-style)
whose poses and geometry share one objective, followed by the same independent
cross-submap test.

### R3 — Sparse hierarchical optimization

- Replace dense whole-history Sim(3) solving with a sparse submap graph.
- Optimize poses/points locally; optimize submap transforms globally; expand to
  global BA only on growth, loop, or residual triggers.
- Reuse Schur sparsity and linearization structure, parallelize independent
  pair/submap work deterministically, and cap online work queues.
- Preserve exact transactional fallback when a solve fails or worsens quality.

Gate: near-linear empirical growth from 300 to full EuRoC length, bounded VSLAM
memory, and no accuracy/completeness regression relative to the dense control.

## 4. Sequential SfM campaign

### S1 — Freeze and measure full-sequence scaling

- Cleanly rerun MH_03 2700f with the current frozen winner and strongest
  available COLMAP 4.1 incremental/global baselines.
- Capture CPU/GPU utilization, peak memory, stage timings, refinement count,
  graph size, and residual change per refinement.
- Use the run only to identify the dominant growth term; do not tune accuracy
  thresholds from it.

Exit: a reproducible full-sequence baseline and a measured complexity budget.

### S2 — Hierarchical mapper

- Partition the sequence into overlapping, motion-adaptive submaps rather than
  fixed frame-count chunks alone.
- Build submaps with R1, align neighbors and verified loops with R2, solve their
  global graph with R3, then perform selective cross-boundary retriangulation.
- Keep the existing trusted/wide transactional policy inside each submap until
  the hierarchy proves a better fixed policy.

Gate: on MH_03 full, equal or better registration and ATE than the current full
control at <=50% wall time and <=50% peak RAM; no submap seam exceeds the RPE
threshold.

### S3 — Hard-video robustness

- Add motion/blur/dynamic-region quality scores to edge selection, not to the
  geometry acceptance threshold.
- Use multiple local hypotheses only where the quality model predicts
  degeneracy; rank them by independent track and reprojection evidence.
- Run ORBIT development clips and ETH3D failure classes; maintain a failure
  taxonomy for low overlap, pure rotation, dynamics, blur, repetition, and
  calibration mismatch.
- Evaluate an optional feed-forward geometry proposal behind the same R2 gates;
  it may improve proposals but may not bypass classical verification or BA.

Gate: material ORBIT success/accuracy improvement with no EuRoC or Tier-C
catastrophe and a bounded runtime/memory cost.

### S4 — Frozen benchmark and release

Freeze once, then run at least three untouched full ordered sequences, ORBIT,
and the unordered guards. Compare against current COLMAP global/incremental,
GLUEMAP, and InstantSfM on supported equal hardware. Publish the full registry,
not just wins.

## 5. Visual SLAM campaign

### V1 — Acceptance-neutral dual-submap scale probe

- Around each true pre/post-hover loop, build independent old-side and new-side
  R1 windows (initially 16/24/32 keyframes with overlap chosen without GT).
- Match persistent tracks across the two local maps and measure R2 Sim(3), but
  do not feed it to the backend.
- Log scale error, rotation vs E, inlier spatial coverage, leave-one-view-out
  stability, and uncertainty separately for true and negative candidates.

Gate: at least 90% of labelled observable bridges yield a stable Sim(3), zero
labelled false bridges pass, and MH_01's expected correction is recovered in
the diagnostic. If this fails, stop tuning DPVO patches; evaluate a learned
two-view/multi-view geometry prior only as an alternative R1 initializer.

### V2 — Sound loop factors and scale-cliff repair

- Replace the invalid E-vs-long-span-DPVO rotation gate with independent
  E/submap consensus; retain the old comparison as a drift diagnostic.
- Send rotation-only factors and full Sim(3) factors through distinct backend
  paths with observability-aware information matrices.
- Apply correction transactionally to retained poses, active DPVO poses,
  points/depths, motion state, and exported trajectory; immediately relinearize
  the active window.
- Add a rollback monitor for post-loop reprojection, flow residual, scale jump,
  and tracking loss.

Gate: MH_01 800f returns to scale 1.0-1.3, Sim(3) ATE <1.5 m, tracking >=0.98,
zero false loops, and no material 400f regression.

### V3 — Full-sequence robustness and relocalization

- Run full MH_01 and MH_03 development sequences; require ATE improvement on
  both with zero false loops before touching held-out data.
- Add submap-level relocalization and map restart/welding for true tracking
  loss, reusing R1/R2 rather than inventing a second map format.
- Make low-parallax state explicit: suppress scale updates while retaining
  rotation/appearance evidence, then reinitialize scale from a fresh parallax
  window.

Gate: all five EuRoC Machine Hall sequences complete without catastrophic scale
events and with a single configuration.

### V4 — SOTA matrix, latency, and submission

- Freeze the EuRoC policy and run all 11 sequences against reproduced
  DROID-SLAM, DPV-SLAM(++), MASt3R-SLAM, VGGT-SLAM, and SLAM-MER where runnable.
- Profile and optimize only measured p95 hotspots; target sustained real-time
  input rate with bounded queues and memory.
- Freeze the ETH3D policy on training, submit test trajectories once, then run
  KITTI guards. Report accuracy/robustness and speed frontiers separately.

Gate: meet the target in section 1.1, place on a public benchmark frontier, and
release all artifacts required for the claim.

## 6. Execution order and resource split

The order is designed to make each expensive experiment answer one question:

1. **Week 0-1:** S1 clean full-sequence baseline; baseline/container manifests;
   no new tuning.
2. **Week 1-3:** R0 interfaces and R1 synthetic/offline local-submap kernel.
3. **Week 3-5:** V1 diagnostic on MH_01 800f and S2 hierarchy on MH_03 300f.
4. **Week 5-8:** R2/R3, V2 scale repair, and MH_03 full hierarchical SfM.
5. **Week 8-12:** V3 full EuRoC development and S3 ORBIT/ETH3D robustness.
6. **Week 12-16:** frozen held-out matrices, public submission, release package.

Until V1 proves a scale-bearing bridge, allocate compute roughly 60% to R1/SfM
and 40% to VSLAM diagnostics. After V1 passes, reverse that split until V2's
MH_01 gate is decided. Do not run more than one timing-sensitive benchmark at a
time.

## 7. Immediate next three slices

1. **S1:** rerun the dead MH_03 2700f head-to-head in a fresh archive directory,
   with process-safe monitoring and complete manifests.
2. **R1a:** specify and test the local-submap data model, track builder,
   triangulation quality record, and gauge conventions; reuse existing
   two-view, track, BA, and Sim(3) primitives before adding dependencies.
3. **V1a (measured negative):** the 16/24/32-radius MH_01 probe rejected every
   scale bridge without backend write-back. Proceed to **R1b**: compare a
   stronger learned geometry initializer against the frozen sparse-SfM result;
   do not weaken R2's consensus gate.

## 8. Stop rules

- No more threshold sweeps on retained DPVO 3D-3D or patch-PnP scale bridges;
  both are experimentally closed.
- No loop may use GT, retrieval similarity, or the drifted live trajectory as
  its correctness proof.
- No feed-forward prior is a measurement by itself; it must pass R2 geometry
  and uncertainty gates.
- No full global refinement cadence is optimized without a stage profile.
- No per-sequence configuration enters a headline table.
- A failed gate ends in an honest negative or an explicitly named missing
  measurement, never a weakened acceptance criterion.

## 9. External frontier references (checked 2026-07-22)

- ETH3D SLAM protocol and ranking:
  https://www.eth3d.net/slam_overview and
  https://www.eth3d.net/slam_benchmark
- MASt3R-SLAM (CVPR 2025):
  https://openaccess.thecvf.com/content/CVPR2025/html/Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.html
- VGGT-SLAM (NeurIPS 2025): https://arxiv.org/abs/2505.12549
- SLAM-MER (CVPR 2026):
  https://openaccess.thecvf.com/content/CVPR2026/html/Piedade_Revisiting_Monocular_SLAM_with_Spatio-Temporal_Scene_Modeling_CVPR_2026_paper.html
- ORBIT SfM benchmark (CVPR 2026):
  https://openaccess.thecvf.com/content/CVPR2026/html/Sabour_ORBIT_Benchmarking_SfM_in_the_Wild_with_360deg_Video_CVPR_2026_paper.html
- COLMAP/GLOMAP: https://github.com/colmap/glomap
- GLUEMAP (CVPR 2026): https://github.com/colmap/gluemap
- InstantSfM: https://github.com/cre185/InstantSfM
- VGGT and current successors: https://github.com/facebookresearch/vggt
