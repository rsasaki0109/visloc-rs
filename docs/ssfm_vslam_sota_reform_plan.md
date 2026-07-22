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

R1e MASt3R-SLAM preparation (2026-07-22): the official `windows` branch is
pinned at revision `6717231a`, as recommended upstream for WSL. The host has
WSL2 Ubuntu 22.04 and a GTX 1660 Ti with 6 GiB VRAM. The official default
512-keyframe CUDA buffer alone is approximately 4.4 GiB at a 512x384 input, so
it cannot coexist safely with the ViT-L model on this GPU. The reproducible
probe therefore fixes two independent 33-frame processes, a 40-keyframe buffer,
and retention of every accepted frame as a keyframe; this is a scale-source
probe, not a runtime baseline. `scripts/export_mast3r_slam_submap_anchor_points.py`
prepares the old/new windows separately, invokes the two processes, extracts
only each central anchor's jointly maintained canonical pointmap, and records
source/config/output SHA-256 hashes. The exact minimal upstream patch is
`scripts/patches/mast3r_slam_windows_6717231_submap_export.patch`; the frozen
configuration is `scripts/configs/mast3r_slam_v1_calib.yaml`. Apply the
zero-context patch with `git apply --unidiff-zero <patch>`. Patch reversal,
Python static checks, and synthetic pointmap-coordinate extraction pass.

The real R1e gate has not run yet. Ubuntu currently has neither the requested
Python 3.11 environment nor CUDA `nvcc`; installing and compiling the official
backend is deferred until the already-running frozen S1/S2 timing controls are
finished, so their resource measurements are not contaminated. No MASt3R-SLAM
measurement, R2 edge, or backend writeback is claimed from this preparation.

### R3 — Sparse hierarchical optimization

- Replace dense whole-history Sim(3) solving with a sparse submap graph.
- Optimize poses/points locally; optimize submap transforms globally; expand to
  global BA only on growth, loop, or residual triggers.
- Reuse Schur sparsity and linearization structure, parallelize independent
  pair/submap work deterministically, and cap online work queues.
- Preserve exact transactional fallback when a solve fails or worsens quality.

Gate: near-linear empirical growth from 300 to full EuRoC length, bounded VSLAM
memory, and no accuracy/completeness regression relative to the dense control.

R3a typed hierarchy foundation (2026-07-22):
`pipelines/slam/src/hierarchical_submap_graph.rs` now owns independent
`LocalSubmap` nodes and stores node state as `local_from_atlas`. Under that
convention R2's measured `target_from_source` composes directly as a graph
edge, avoiding an easy inverse/order bug. Initialization propagates only over
verified scale-bearing constraints from a fixed root and rejects every
disconnected node. `RotationOnlyConstraint` values remain in a separate typed
collection and cannot enter the scale solver. Tests prove shared points from
two gauges materialize to the same atlas point, a connected three-node scale
chain optimizes, and disconnected/rotation-only hierarchies fail explicitly.

This is an architecture milestone, not the R3 gate: the current submap graph
delegates to the existing deterministic Sim3 optimizer, whose normal equations
are dense. It provides the stable boundary for a block-sparse submap solve and
S2 mapper, but no near-linear/full-sequence claim is made yet.

R3b block-sparse Sim(3) solve (2026-07-22):
the shared block Cholesky backend now supports 7-DOF blocks, and
`Sim3PoseGraph` switches at a configurable node threshold from its dense
reference solve to direct scalar-COO assembly. The sparse path never allocates
the `7N x 7N` dense Hessian and reuses one symbolic factorization across LM
iterations. A forced sparse-vs-dense regression reaches matching final cost
within `1e-12` and every optimized node within `1e-8` tangent norm; the complete
SLAM library result is 422 passed / 6 ignored. The release one-iteration chain
plus loop probe measured 300 nodes in 0.006 s, 900 in 0.016 s, and 2700 in
0.050 s: 3x node growth cost 2.89x then 3.16x. The reproducible ignored test is
`bench_sparse_growth_to_euroc_length`.

This closes the dense-solver bottleneck and supplies near-linear solver evidence
to full MH_03 frame count, but it still does not close the whole R3 gate. The S2
mapper must route real independent submaps through this graph, then demonstrate
bounded end-to-end VSLAM memory and no full-sequence accuracy/completeness
regression against the dense control.

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

S2a partition boundary (2026-07-22):
`submap_partition` now deterministically divides an ordered verified view graph
into overlapping windows with hard minimum/maximum sizes and no short tail.
Within a bounded radius of the target size it selects the seam with the most
verified cross-boundary correspondences; an optional non-negative per-cut hint
lets the frontend downweight blur, dynamics, or unsafe motion without weakening
any geometric acceptance gate. Pair indices are filtered and remapped into each
independent local window. Five tests cover exact overlap/coverage, supported
seam selection, motion-quality override, local-index remapping, and invalid
input rejection. This is the deterministic input boundary for the S2 mapper,
not yet an end-to-end S2 gate result.

S2b typed assembly path (2026-07-22):
`submap_overlap` converts shared `(source frame, keypoint)` identities into
mutual-best one-to-one landmark matches and separately transforms verified
essential pair rotations into the two independent local gauges. A weighted
angular consensus rejects inconsistent rotation evidence before R2. The new
`hierarchical_sfm` composition then runs S2a partitioning, rebuilds every local
window independently, verifies every adjacent 3D seam through the unchanged R2
gates, and commits node transforms only after the R3 graph succeeds. Synthetic
tests recover an exact scale-2.5 seam through the complete R2/R3 transaction,
reject absent independent rotation evidence, and handle a one-submap sequence
without inventing a scale edge. Real-data frontend/export wiring and the MH_03
S2 gate remain open.

S2c real-data frontend smoke (2026-07-22):
`sequential_sfm_demo --hierarchical` now retains each essential-verified pair
rotation from the existing matching pass, exposes bounded submap partition/build
controls, runs independent local maps in a capped Rayon pool, and exports a
unique-image COLMAP atlas plus per-seam diagnostics. On the frozen first 120
MH_03 frames, a 35-frame window correctly failed `NoSeedPair`; 64-frame windows
then reconstructed but the unchanged R2 gate rejected 845/1471 = 0.5744. The
accepted configuration used windows 0..88 and 16..120 (72-frame overlap), at
least four shared observations per point match, and produced 120/120 registered,
3658 points, 39058 observations. Its seam passed with 676/946 = 0.7146 inliers,
mean normalized residual 0.014526, and 864/864 essential-rotation consensus.

Two concurrent local builds reduced the same hierarchical run from 294.2 s to
140.8 s total versus 336.5 s for the frozen dense control: 41.8% of dense wall
time, satisfying the S2 speed ratio on this smoke. Accuracy does not pass yet:
Sim(3)-aligned ATE is 1.432 cm versus dense 0.468 cm (both 120/120 registered;
73 timestamp-associated GT poses). A guarded fixed-rotation camera-centre
scale/translation proposal was added behind explicit opt-in; it must re-pass
the original landmark gates, camera residual, scale-change, and leave-one-out
checks. The real seam rejected it as `HighCameraResidual`, so no correction was
committed. Artifacts are under
`E:/visloc_archive/sota_s2_hierarchical_smoke_mh03_120_20260722`.

Verdict: S2 now has a real, transactionally safe frontend and meets its smoke
speed target, but the accuracy gate remains open. S2c must jointly refine the
overlap poses/points (and selectively retriangulate/weld seam tracks) before a
full MH_03 claim; the camera-only degeneracy may not be bypassed or relaxed.

S2d selective seam BA (2026-07-22):
the accepted R2 inlier landmark identities now form explicit cross-submap
unions for an opt-in transactional BA. Root-only prefix poses remain fixed to
anchor gauge and preserve trusted history; poses repeated in the overlap and
all target-side poses/points are jointly optimized with sparse Schur BA. A
candidate writes back to both local gauges only when its robust reprojection
cost does not worsen. Structure observed exclusively by fixed poses is omitted
because its Schur blocks are exactly disconnected from every pose update.

On the same 120-frame development smoke, fixing the 16-frame root-only prefix
and running five seam-BA iterations reduced 56,363-observation cost from
1,182,171 to 53,763. It retained 120/120 registration and achieved Sim(3) ATE
0.4641 cm versus dense-control 0.4681 cm. The deterministic rerun took 163.57 s
versus dense 336.47 s (48.61%) and produced a byte-identical `images.txt`
SHA-256 `5de383041e322e262ddecf91e0282036de5fd118621c9775ae0a69bd4592c29e`.
Four BA iterations were faster (160.14 s) but missed accuracy at 0.4742 cm, and
an 80-frame root window was independently rejected by the unchanged R2 mean
residual gate (0.01658 > 0.015). Thus the frozen smoke choice is the verified
88/104 windows, 72-frame overlap, four shared observations, two local workers,
and five selective BA iterations.

This satisfies the S2 gate only on the 120-frame development smoke. Full MH_03
must still confirm peak RAM, every seam RPE, registration/ATE, and <=50% wall
time without further tuning. Export now also welds only R2-inlier local landmark
groups into one deduplicated multi-view COLMAP track; all unwelded points remain
submap-owned, so unverified cross-seam observations cannot be merged.

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

Held-out freeze (2026-07-22): before downloading or evaluating any remaining
EuRoC Vicon sequence, `V1_02_medium`, `V1_03_difficult`, and `V2_02_medium`
were fixed as the three full ordered held-out sequences in
`benchmarks/protocols/ssfm_heldout_euroc_v1.json`. Exact-name searches found
no prior repository or `E:/visloc_archive` experiment artifacts for these
sequences. All five Machine Hall sequences plus previously used V1_01, V2_01,
and V2_03 are explicitly excluded as development data. The manifest binds the
already-frozen S2 policy/source revision, forbids post-result tuning, requires
failure/DNF cells, and requires per-sequence plus median/worst reporting.
The official Room 1/2 archive URLs, byte sizes, and publisher-provided MD5
checksums are bound in that manifest. `download_ssfm_heldout_euroc.py` performs
resumable downloads and refuses promotion from `.partial` until size and MD5
both pass; it is queued behind the clean S2 launcher. Input preparation is
isolated in `prepare_ssfm_heldout_euroc_inputs.py`, which never opens GT and
records calibration/index, frontend source, SuperPoint weight, protocol, and
feature-set hashes. `run_colmap_ssfm_frozen.py` runs COLMAP 4.1 incremental and
global mappers from one same-input SIFT/sequential-match database, preserves
explicit DNF cells, and monitors wall/RAM/global-VRAM. Held-out execution uses
`run_ssfm_heldout_sequence.py`: neither the hierarchical nor COLMAP runner is
given a GT path. The initial safe extractor omits every
`state_groundtruth_estimate0` member rather than reading or hashing labels.
Only after both timed engine manifests prove `ground_truth_read=false`,
`materialize_ssfm_heldout_ground_truth.py` extracts that sequence's exact GT
CSV and binds it to both exit-manifest hashes; the finalizer verifies this
chain before evaluation. Synthetic ZIP tests prove that initial extraction
emits no GT bytes and that materialization rejects missing isolation evidence.
These runners are prepared; no held-out result has been observed yet. Frontend
RAM/global-VRAM is now sampled alongside mapping RAM. Expected runner failures
also write immutable top-level manifests, and `summarize_ssfm_heldout_suite.py`
turns a pre-finalization failure into three explicit DNF cells instead of
omitting the sequence. It requires all three frozen sequences, reports
median/worst plus the per-sequence reproduced Pareto frontier, and deliberately
leaves the claimable SOTA gate false while GLUEMAP, InstantSfM, ORBIT, or
release evidence is absent. `run_ssfm_heldout_suite.py` is the frozen serial
entry point: it verifies the GT-free extraction and all executable/script
hashes before the first run, executes the three sequences once in protocol
order, preserves a synthetic failure manifest if a child crashes before
writing one, and always attempts the failure-inclusive aggregate. Its
end-to-end failure-path test confirms that three preparation failures remain
three DNF rows per engine rather than disappearing from the report.
`verify_ssfm_heldout_suite.py` is the release audit: it rehashes every frozen
input, rejects overlapping sequence intervals, verifies the deferred-GT
engine-exit hash/timestamp chain for successful runs, and regenerates the suite
summary to detect edited aggregate results. Both explicit-failure and
successful deferred-GT evidence chains have regression coverage.

External-baseline pre-freeze (2026-07-22):
`benchmarks/protocols/ssfm_external_baselines_v1.json` binds GLUEMAP official
HEAD `adc9e4bb5f41014d3f7c157a879edc278588c829`, its full default Pi3 pipeline,
the same 6 GiB GPU/full-resolution/full-sequence policy, and mandatory
GLUEMAP/InstantSfM cells. The published InstantSfM URL currently redirects to
`flqcsvqqvw/InstantSfM`, while both original and redirected Git transports
return repository-not-found and the redirected codeload archive returns 404.
This is recorded as a source outage, not yet a DNF; it must be rechecked just
before setup. Critically, external baselines cannot be run later after labels
are visible: their success/DNF manifests must join the per-sequence transaction
before GT materialization. `run_external_ssfm_baselines_frozen.py` now executes
hash-bound official adapters or emits command/return-code/resource-backed DNF
cells, and the sequence transaction is fixed as prepare, hierarchical, COLMAP,
external pair, GT materialization, then finalization. The materializer,
finalizer, five-engine aggregator, and release verifier all reject a missing,
unattempted, or GT-reading external cell. Success/DNF adapters and the complete
five-engine failure path have synthetic regression coverage. The frozen
held-out suite has not started. The pre-GT GLUEMAP adapter is now implemented
in `run_gluemap_ssfm_adapter.py`: because the official CLI requires a COLMAP
reconstruction at `gt_intrinsics_path`, it converts only `rect/calib.txt` and
ordered image names into a calibration-only reconstruction with explicit
identity dummy poses, never pose GT. It runs Pi3, sequential sampling 1,
SHARED known intrinsics, Doppelgangers++, and full refinement, then converts
the final `gluemap_aba` model to TUM with the frozen input timestamps. Real
GLUEMAP installation/setup evidence and an InstantSfM success or fresh
source-outage DNF remain required after the clean S2 timing run.
`prepare_external_ssfm_setup.py` performs that one-time audit: exact main and
recursive-submodule revisions, a clean tracked tree, all four official
checkpoint hashes, `gluemap`/`pygluemap` imports, CUDA availability, and the
official CLI must pass before it emits a hash-bound ready adapter. It also
repeats both published/redirected InstantSfM `ls-remote` attempts; any newly
available source stops setup for revision binding and adapter implementation
instead of being mislabeled as an outage DNF.

ORBIT release audit (2026-07-22): the official CVPR paper/supplement defines
the 100-clip protocol and success thresholds, but its Code/Dataset links remain
the literal placeholder `anon`, and the official supplemental ZIP contains no
dataset, trajectories, or evaluator. See
[`orbit_benchmark_protocol.md`](orbit_benchmark_protocol.md). Do not replace
the missing official release with sample videos or an inferred metric; rerun
the audit before S4 freeze and preserve the exact released evaluator revision.

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

V2a backend-safety slice (2026-07-22): the opt-in DPVO Sim(3) backend no
longer writes a solved proposal directly into live state. It applies every
pose/depth correction to a cloned patch graph, scores the already-materialized
learned target reprojections before and after, checks valid-edge retention and
the maximum absolute log-scale jump, and swaps the clone into the live graph
only when every gate passes. A rejection reports exactly one of
`NonFiniteCorrection`, `ScaleJump`, `ActiveReprojectionValidityLoss`, or
`ActiveReprojectionWorsened`; diagnostics and the EuRoC runner distinguish a
solved proposal from an actually committed correction. Folded-frame patch
depths now receive the same scale as their retained pose, closing a state
consistency hole in exported/full-history geometry. Seven focused backend
tests, the complete 446-test no-default `visloc-slam` library suite, and the
ONNX EuRoC example check pass.

This is a rollback/write-back positive, not the V2 data gate. It does not make
the rejected V1/R1d scale sources valid, does not admit a new scale edge, and
does not change the conclusion that a jointly optimized independent dense
submap measurement is still required before MH_01 correction may be enabled.

V2b typed-factor slice (2026-07-22): `VerifiedDpvoLoopFactor` and
`run_verified_submap_backend` form a second backend entry point that cannot
accept legacy `Sim3LoopMeasurement`. Its anchors carry both R1 submap identity
and an independent local world-to-camera pose; source/target ID mismatches are
rejected before solve or write-back. `RotationOnlyConstraint` produces a
factor whose information matrix has exactly the three rotation diagonals
nonzero. `SubmapSim3Constraint` preserves the complete independently measured
similarity and derives bounded translation, rotation, and log-scale
information from R2 residual, rotation-consensus, leave-one-out scale, scene
scale, and inlier support. Both use the same sparse Sim3 solve and V2a
transaction gates after their deliberately distinct measurement paths.
Synthetic tests prove that rotation-only execution cannot alter scale, a full
typed factor activates scale correction, full Sim3 gauge conversion is
preserved, and bad submap provenance leaves the graph byte-identical. The 12
focused backend tests pass. This path remains unwired by default until an R1
dense-submap source clears the frozen V1 acceptance gate.

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

Implementation status (2026-07-23):
`benchmarks/protocols/vslam_euroc_sota_v4.json` now freezes the exact 11-sequence,
three-repetition, failure-inclusive matrix before any V4 result. It requires
full camera streams, at least 90% tracking coverage, all 33 successful runs,
mean per-sequence Sim(3) ATE <= 0.020 m, sustained 20 Hz input, bounded CPU/GPU
memory and queues, and no committed correction beyond the Sim(3) backend's
frozen `4.0` absolute-log-scale transaction gate. It also requires
`--onnx-cuda`: CUDA execution-provider registration is strict, CPU fallback is
an error, and the requested backend is recorded in and revalidated from each
run summary. The runner summary records
both that configured threshold and the cumulative committed scale maximum;
rejected scale-jump proposals are counted separately. The evaluator
`scripts/evaluate_vslam_sota_v4.py` reports mean, median, and worst sequence
ATE and deliberately leaves `claimable_sota=false` unless separate verified
public ETH3D SLAM or ORBIT frontier evidence and a released-artifact SHA-256
are supplied. No EuRoC result or public-frontier result has been claimed by
this protocol yet. `scripts/run_vslam_euroc_sota_v4.py` is the corresponding
serial runner: it audits official cam0 row counts before launch, owns the
full-sequence/stride/seed arguments, rechecks executable, configuration,
DPVO/SuperPoint model-bundle, protocol, and ONNX Runtime hashes, samples
working-set and per-process GPU memory, and preserves every failed run. The
algorithm configuration remains a separate mandatory hashed input until V3
development selects the single policy; this avoids freezing an unfinished
configuration while preventing mixed-policy V4 matrices.

The first strict-CUDA MH_01 probe used the locally installed CUDA-enabled ORT
1.24.2 plus its cuDNN 9 dependencies (30 frames, stride 2, 48 patches/frame).
It completed 30/30 frames and proved that the CUDA provider can load, but took
844.13 ms/frame: encoder 90.89 ms, correlation 388.59 ms, update 273.98 ms,
and BA 9.94 ms. Sim(3) ATE over this short diagnostic prefix was 0.1073 m.
Thus CUDA ONNX inference alone is not close to V4's 50 ms/frame gate; the
native CPU correlation and update paths are the measured primary bottlenecks.
The previous ORT 1.23.2 CPU distribution now fails honestly under the same
strict flag instead of silently producing CPU-labelled-as-CUDA evidence.
Reducing the diagnostic graph to 16 patches/frame and smaller 8/4/7 windows
still took 287.32 ms/frame and produced a short-prefix Sim(3) scale of 2.46,
so graph starvation is not an acceptable route to the gate. Running the
independent fnet/inet CUDA sessions concurrently preserved the exact same
trajectory and reduced that matched probe to 243.56 ms/frame (encoder 42.60
ms), a useful but insufficient 15.2% end-to-end improvement. The next runtime
architecture step must keep correlation/update tensors on GPU rather than
relying on further queue shrinkage or encoder-only tuning.

The update half of that architecture step is now implemented. The exporter
uses host-computed compact group IDs and an E-row scratch tensor to fuse both
SoftAgg reductions between the pre/post update networks into
`dpvo_update_full.onnx`; model bundles without it retain the legacy split
path. CPU and strict-CUDA fixture parity pass at <=3.624e-5 max absolute
error. On the matched 48-patch probe, fused update time fell from 273.98 to
65.56 ms/frame (76.1%) and total time from 844.13 to 679.88 ms/frame. The
remaining dominant stage is CPU correlation at 469.48 ms/frame, so the next
GPU-residency slice is the batched two-level correlation volume, not further
update-cell or encoder tuning.

The first grouped correlation graph is a measured negative, retained only
behind `--onnx-correlation`. It exports the exact two-level 882-value volume,
passes the frozen level-0 fixture under strict CUDA, and caches channel-first
maps, but one ORT call and host upload per destination-frame group increased
the matched 16-patch correlation stage from 127.14 to 303.17 ms/frame (total
250.03 to 443.57 ms/frame). It is therefore off by default and forbidden as a
V4 speed path. The next implementation must batch across target frames while
keeping the feature-map pyramid resident on device (or call a native indexed
CUDA kernel); repeating target-by-target ORT calls is experimentally closed.

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
