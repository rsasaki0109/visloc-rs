# Visual SLAM + Sequential SfM development plan

Status: proposed, 2026-07-19

User priority (2026-07-19): **monocular first and monocular by default.** All
headline SLAM and Sequential SfM work is mono-to-mono. IMU/stereo work is
deferred and requires an explicit decision before it becomes an active
milestone.

This document is the forward plan for advancing both visual SLAM and
**ordered/sequential** structure-from-motion. It does not replace the detailed
historical records in `dpvo_droid_port_plan.md` and `colmap_port_plan.md`.

## 1. Mission

Build two products on a shared Rust geometry and retrieval foundation:

1. **Visual SLAM:** robust online monocular tracking first, with stereo and IMU
   as separately scored modes. The immediate wall is DPVO's scale failure after
   low-parallax motion on EuRoC MH_01.
2. **Sequential SfM:** an offline, ordered-image reconstruction pipeline that
   beats current COLMAP under the same sensor input, data, hardware, and scoring
   protocol. Sequence order is allowed and must be used explicitly; unordered
   photo SfM remains a generalization guard, not the primary product.

“Beat COLMAP” means a reproducible Pareto result, not winning one cherry-picked
column: on the declared benchmark suite, a single fixed visloc configuration
must match or exceed COLMAP's registration/completeness and independent-GT
accuracy, while being faster or materially more resource-efficient. A result
against a COLMAP-authored reference model does not count as an independent-GT
accuracy win.

## 2. Current evidence

### Visual SLAM

- DPVO 400-frame MH_01 is strong: Sim(3) ATE 0.152 m, scale 1.234, tracked 1.0.
- The 800-frame run fails after the real ~24 s near-still hover: scale 20.633,
  Sim(3) ATE 2.875 m.
- M13 localized the failure; M14's freeze response and M15's depth damping were
  honest negatives. M15 preserved full tracking and proved that the abrupt
  hover-exit transition, not frame admission itself, is the remaining local
  failure.
- Long-loop infrastructure exists, but no rotation-consistent long-range loop
  has survived verification. Correspondence soundness and retrieval recall are
  still open.

### Sequential SfM

- The historical 300-frame monocular EuRoC MH_03 result reached 299/300 at
  1.64 cm Sim(3) camera-centre RMSE; its historical COLMAP row reached 300/300
  at 0.37 cm. P0 refreshed this comparison below; historical numbers remain
  context, not the active baseline.
- The full 2700-frame result that beats an older COLMAP run in time uses visloc
  stereo against COLMAP monocular, so it is useful engineering evidence but not
  the primary head-to-head claim.
- The COLMAP-port M1-M6 work delivered multi-model verification, a persistent
  correspondence graph, vocab-tree retrieval, mapper stall recovery, rescue
  bridging, and LightGlue ONNX.
- LightGlue improves ETH3D terrace (1.38 -> 0.74 cm) and office (18/26 -> 22/26,
  0.50 -> 0.44 cm), but courtyard stays at 14/38 even after the view graph is
  connected. The remaining courtyard failure is concentrated 2D-to-existing-3D
  support for image registration, not candidate-pair connectivity.
- Existing COLMAP speed claims must be refreshed. COLMAP 4.1 adds optional
  Caspar GPU BA, and its official changelog reports large incremental-mapper
  speedups. However, the official Windows CUDA binary `fa8e3b3` installed on the
  benchmark machine was probed on 2026-07-19 and rejects Caspar because it was
  built without `CASPAR_ENABLED`. The required baseline is therefore the
  strongest backend actually available in that official binary (Ceres), with a
  separately-labelled Caspar row only if a Caspar-enabled build is provisioned.

#### P0 fresh mono-to-mono baseline (2026-07-19)

Both engines consumed the same first 300 rectified MH_03 cam0 frames on the same
machine. Ground truth was read only after both engines exited. Shared
rectification and the Rust build are excluded from engine wall time.

| Engine | Frontend / BA | Registered | Common-frame Sim(3) ATE | Mean reprojection | Wall time |
|---|---|---:|---:|---:|---:|
| visloc HEAD `37d9897` | SuperPoint / Rust BA | 260/300 | **0.427 cm** | 0.604 px | **17.0 min** |
| COLMAP 4.1 `fa8e3b3` | CUDA SIFT / Ceres | **300/300** | 0.670 cm | **0.510 px** | 38.2 min |

This is an accuracy-and-runtime win but not a product win: visloc is 2.24x
faster and has 36% lower trajectory RMSE on the exact 213-pose common timestamp
subset, while
losing 40 images. Registration completeness is now the binding first gate.
The GT stream begins 2.35 s after the camera stream, explaining the 213/260 and
253/300 associated-pose counts; it is not a timestamp-conversion loss.
On each engine's native registered set, COLMAP scores 0.616 cm and visloc remains
0.427 cm; the common-frame row is the fair head-to-head accuracy comparison.

Artifacts:

- `E:/visloc_archive/p0_mono_sfm_mh03_300_sift_ceres_b_20260719/summary.json`
- registry manifests `mono-sfm-{visloc,colmap}-MH_03_medium-20260718T225823Z.json`

P0.1 confirmed a changed completeness/quality trade-off after `dac1400`: that
commit registers 300/300 on the exact same fresh feature cache in 800.1 s, but
scores 8.35 cm Sim(3) ATE (8.92 cm on HEAD's common timestamps) due to a
catastrophic pose, versus HEAD's 260/300, 872.2 s, and 0.427 cm. The fix must
combine old reach with current geometric safety; simply reverting image
selection would restore completeness by reintroducing a bad reconstruction.

A controlled HEAD policy A/B then separated selection from every later mapper
change. The legacy correspondence-count ranking, now exposed as an experimental
policy, reached 284/300 at 0.399 cm native Sim(3) ATE (0.413 cm on its exact
common subset with the visibility run), 0.598 px reprojection, and 881.8 s SfM.
It improves both completeness and accuracy over visibility's 260/300 and
0.427 cm, without reviving `dac1400`'s frame-191 failure, but is about 10%
slower and still leaves 16 images. Those images exhaust three PnP trials despite
517-665 available 2D-3D correspondences because only 3-10 survive PnP. This
directly prioritizes ordered relative-pose/structure-less registration in B2;
raw correspondence supply alone is not the remaining bottleneck.

## 3. Benchmark contract

### 3.1 Rules shared by both tracks

- Same images, calibration assumptions, sensor mode, machine, and evaluation
  script for both engines.
- Record the exact commit/binary version, command, seed, hardware, runtime,
  peak RAM/VRAM, and output artifact in the benchmark registry.
- Run at least three seeds when nondeterminism changes the verdict. Report
  median plus worst run; do not select the best seed.
- Tune on a declared development subset. A held-out sequence decides whether a
  mechanism ships.
- Compare one fixed configuration per sensor/dataset family. Per-sequence
  tuning may be reported as an ablation, never as the headline.
- All new mechanisms default off until they pass their full acceptance gate.

### 3.2 Visual SLAM scorecard

Primary monocular development set:

- EuRoC MH_01, MH_03, MH_05: 400/550/800-frame microscopes plus full sequences.
- ETH3D SLAM training sequences: at least one loop, one non-loop, and one
  low-texture/low-parallax case.
- KITTI 00/02/05/06/07/09 as long-motion and scale-drift guards.

Metrics:

- tracking fraction and longest tracking outage;
- ATE RMSE (Sim(3) for monocular visual, SE(3) for metric modes);
- windowed scale, drift slope, and post-degeneracy flatness;
- relative pose error, loop precision/recall, and catastrophic-run count;
- end-to-end Hz, p95 frame latency, RAM, and VRAM.

Near-term MH_01 gate: 800-frame scale below 10 or Sim(3) ATE below 1.5 m, with
tracked fraction >= 0.98 and no material 400-frame regression. Final gate:
post-hover scale must return to the pre-hover regime (roughly 1.0-1.3) and stay
flat, then generalize to MH_03 and a held-out ETH3D sequence.

### 3.3 Sequential SfM scorecard

Tier A — ordered metric sequences:

- EuRoC MH_01/MH_03/MH_05 and V1_01/V2_03, monocular-to-monocular first;
- full-sequence runs plus the existing MH_03 300-frame microscope;
- optional stereo-to-stereo table kept separate.

Tier B — independent-GT video generalization:

- ETH3D SLAM training sequences, including loop and non-loop motion;
- KITTI 00-10 for long-baseline trajectory error where appropriate.

Tier C — unordered regression guard:

- ETH3D courtyard/terrace/office and South Building/Gerrard Hall.

Metrics:

- registered images and largest registered component;
- independent-GT camera-centre Sim(3)/SE(3) RMSE and relative pose error;
- mean/median/p95 reprojection error;
- valid 3D points, observations, median track length, and track conflicts;
- end-to-end wall time split into extraction, retrieval, matching,
  verification, registration, local BA, and global BA;
- peak RAM/VRAM and deterministic output hash where applicable.

First win gate: on MH_03 300 mono, reach 300/300, <= the fresh COLMAP 4.1
Sim(3) RMSE (currently 0.616 cm), and beat its fresh wall time. Program-level win gate: meet all
three of accuracy, registration, and runtime on at least three held-out ordered
sequences with one configuration, with no catastrophic Tier-C regression.

COLMAP baseline matrix:

- COLMAP 4.1 incremental mapper with sequential matching and built-in periodic
  loop detection;
- Ceres and Caspar BA where actually compiled/supported, taking COLMAP's better
  valid result and recording unavailable backends rather than assuming them;
- official SIFT and ALIKED frontends where available, taking the strongest
  declared pipeline rather than an intentionally weak default;
- global mapper as a secondary reference, not silently mixed into the
  incremental result.

## 4. Shared architecture

The products share evidence and infrastructure, but must not be forced into one
state estimator.

Shared:

- image ingestion, camera models, feature extraction, LightGlue matching;
- temporal and appearance retrieval, multi-model geometric verification;
- persistent correspondence/view graph, tracks, SE3/Sim3 math;
- robust BA/PGO primitives, COLMAP IO, benchmark registry and scorers.

SLAM-specific:

- DPVO patch graph, recurrent update, fixed-lag BA, online marginalization;
- strict latency and bounded-memory policies;
- online loop closure and optional IMU scale/bias state.

Sequential-SfM-specific:

- persistent image/track database and retryable image registration;
- batch or hierarchical submap construction;
- scheduled local/global refinement and offline self-calibration;
- completeness may trade latency, but not geometric validity.

This boundary prevents an SfM improvement from destabilizing online tracking,
while allowing verified retrieval, loops, graph, and optimizer improvements to
flow both ways.

## 5. Work plan

### P0 — Reset and lock the battlefield (1-2 weeks)

1. Add one-command benchmark manifests for the Visual SLAM and Sequential SfM
   suites above.
2. Re-run current visloc HEAD and COLMAP 4.1, including Caspar, on the same
   machine. Preserve older results as historical rows.
3. Add windowed-scale and stage-timing fields to the registry; fail reports that
   omit sensor mode or alignment type.
4. Archive all outputs under `E:/visloc_archive/`; never write benchmark output
   to C:.

Gate: a generated comparison table can be reproduced from raw images without
manual editing, and every claimed win/behind verdict points to registered runs.

### A1 — Close the pure-monocular hover transition (M16, 1-2 weeks)

Implement gradual per-patch release from hover damping to normal weight. Release
must be spread across the observed unflag interval, capped per frame, and logged
as a histogram so a mass-release cliff cannot hide in an average.

Experiment order:

1. unit/fixture tests for monotonic release and bounded release count;
2. MH_01 550-frame A/B to detect early overshoot cheaply;
3. same-binary 400/800 controls and mechanism-on runs;
4. windowed-scale profile, not only final ATE.

Gate: the near-term MH_01 gate in section 3.2. If gradual release still leaves
scale >= 10 or introduces another exit cliff, stop adding pure-mono hover
heuristics. Record the negative and move to A2.

Status (2026-07-19): implemented and evaluated. M16's duration-32/cap-4
geometric release removed the mass-release cliff and kept tracking at 1.0000,
but failed the 800-frame accuracy gate: similarity scale `26.593` and Sim(3)
ATE `3.086 m`, versus control `20.633` and `2.875 m`. The windowed profile
shows a delayed crossover after frame 639 rather than recovery. A1 is therefore
a bounded negative and detector-specific mono hover tuning stops here. Because
the current product directive is monocular-first, A2 remains deferred; active
development moves directly to B1 and B2.

### B1 — Make sequence order a first-class view graph (2 weeks)

Replace the demo-only fixed temporal window with a reusable ordered pair
generator:

- short temporal edges with motion/overlap-adaptive window size;
- skip edges for stronger baseline and track survival;
- periodic appearance retrieval for loop/non-local pairs;
- transitive candidates from the persistent correspondence graph;
- the existing multi-model verifier on every admitted edge.

Keep nearest-neighbour matching as the fast first pass and invoke LightGlue on
ambiguous/high-value pairs. Add a small session pool or CUDA session path so
LightGlue is not serialized behind one mutex.

Gate: higher verified track length and no lower registration on MH_03 300;
LightGlue throughput improves by at least 4x from M6's serialized CPU path, with
identical accepted matches within numeric tolerance.

First-slice status (2026-07-19): `ordered_view_graph` now provides a reusable,
deterministic generator that merges adaptive temporal windows, fixed skip
offsets, appearance pairs, and transitive pairs while retaining every proposal
source on one deduplicated edge. The sequential demo uses this generator and
adds `--skip-offsets`; its default remains exactly the legacy fixed window.
Four policy tests and the release example build pass.

On the same MH_03 300 feature cache, window 5 plus skip offsets 8/12 produced
2065/2065 verified pairs and 2,317,249 inlier correspondences. Against the P0
visibility baseline it improved registration `260 -> 286`, native Sim(3) ATE
`0.427 -> 0.347 cm`, common-frame Sim(3) ATE `0.427 -> 0.370 cm`, and mean
track length `13.80 -> 15.03`. It remains ahead of COLMAP's common-frame ATE
(`0.667 cm`) and mapper runtime (`1092.6 s` vs `2260.6 s`), but total landmarks
and observations fell (`9062 -> 7465`, `125100 -> 112166`) and SfM runtime rose
36% versus P0. Registration/track-length parts of the gate pass; B1 is not
complete because adaptive candidate budgeting, appearance/transitive feeding,
and the LightGlue throughput gate remain. The next slice must rank/budget skip
edges rather than admitting every fixed-offset pair, then B2 targets the final
14 unregistered images.

Second-slice status (2026-07-19): deterministic skip-source budgeting was added
as `skip_source_stride` / `--skip-stride` (default 1 preserves the dense policy).
With stride 2, the graph retained 290 skip edges instead of 580 and improved the
Pareto point substantially: `297/300` registered, native Sim(3) ATE `0.351 cm`,
mean track length `15.82`, 130,584 observations, and SfM `960.7 s`. A controlled
count-policy cross-check on the identical graph registered only `261/300`; the
visibility pyramid remains the selected policy.

The three missing frames (251, 284, 297) each had 526-532 nominal 2D-3D
correspondences but only three PnP inliers during growth. A new opt-in,
one-shot `post_refinement_registration` completion pass retries each missing
image exactly once after final filtering/re-triangulation. It registered all
three with 385-387/385-387 inliers, proving that contaminated pre-refinement
tracks—not missing two-view support—caused the gap. One final BA then produced:

| MH_03 first 300 | visloc B1/B2 precursor | COLMAP 4.1 |
|---|---:|---:|
| registered | **300/300** | 300/300 |
| Sim(3) ATE | **0.349 cm** | 0.616 cm |
| mapper runtime | **1185.7 s** | 2260.6 s |
| mean reprojection | 0.570 px | 0.510 px |

This is the first controlled same-image result that beats COLMAP on accuracy,
registration, and mapper runtime simultaneously. The pass is bounded and off by
default pending held-out evaluation. It is not a substitute for full B2:
relative-pose/submap recovery is still required for held-out sequences where
final filtering does not expose a clean 2D-3D set.

Held-out status (2026-07-19): the MH_03-selected configuration was frozen and
run without tuning on the first 300 frames of MH_01. The ordered graph verified
1759/1775 pairs with 1,285,154 inlier correspondences. visloc registered
`300/300` during normal growth (`post-refinement +0`), with 19,035 points,
153,967 observations, and 0.811 px mean reprojection error. This confirms that
the B1 graph and visibility policy generalize to complete registration without
depending on the bounded B2 precursor.

The full held-out COLMAP-beating gate did not pass. On the common 278 GT-aligned
timestamps, visloc Sim(3) ATE was `0.9255 cm` versus COLMAP's `0.9126 cm` (1.4%
worse); both exported `300/300` registered images. visloc's reported pure SfM
work was `604.1 s`, slightly below COLMAP mapper's `615.7 s`, but pair
verification made the comparable visloc mapping stage `680.4 s`. Including
each engine's feature/matching front end, visloc took `814.6 s` versus COLMAP's
`637.4 s`. Therefore MH_01 is registration parity, near accuracy parity, and a
runtime loss—not a held-out COLMAP win. Do not tune the fixed offsets against
this single held-out result. The next runtime slice should profile and batch the
SuperPoint/LightGlue front end and avoid unnecessary verified edges while
holding `300/300` and common-subset ATE; full B2 remains reserved for sequences
that actually expose a structure-less registration gap.

B4 first runtime slice (2026-07-19): opt-in debug timing now separates track
build, seed growth, next-image selection, PnP, triangulation, local BA, growth
global refinement, final refinement, and output assembly without changing the
normal output path. On the frozen MH_01 cache, the control's `554.8 s` SfM time
was dominated by growth global refinement: `387.5 s / 40 calls` (89% of the
`432.3 s` growth phase). PnP, triangulation, and local BA together took only
`43.8 s`.

The first bounded optimization skips stall-recovery refinement when every image
is already registered **and image filtering is disabled**; no missing image
exists for that recovery to unlock, while filtering-enabled runs retain the
round so they can still demote weak poses. The mandatory final iterative
refinement still runs. The cached-feature A/B
kept `300/300`, 19,035 points, 153,967 observations, 0.811 px reprojection, and
Sim(3) ATE `0.9255 cm`, while reducing global-refinement calls `40 -> 39`, growth
`432.3 -> 337.0 s`, and total SfM `554.8 -> 454.2 s` (18.1%). This is a runtime
ablation, not yet a fresh end-to-end head-to-head; its manifest is
`mono-sfm-visloc-b4-skip-complete-stall-MH_01_easy-20260719T033505Z.json`.

Fresh end-to-end confirmation used newly rectified images and newly extracted
features for both engines. visloc again produced `300/300`, 19,035 points,
153,967 observations, 0.811 px reprojection, and Sim(3) ATE `0.9255 cm`.
Its mapping stage fell from the pre-change `680.4 -> 505.3 s`, and total engine
wall time from `814.6 -> 637.1 s`. Fresh COLMAP also registered `300/300`, with
ATE `0.9116 cm` and total wall `700.0 s`. Thus the first B4 slice converts the
held-out runtime loss into a 9.0% runtime win with registration parity and no
visloc accuracy regression, but the 1.5% ATE deficit means this is still not a
held-out all-axis COLMAP win. Registry manifests end in `20260719T040219Z`.

### A2 — Targeted visual-inertial hover-exit scale recovery (2-4 weeks)

Use the proven M14 low-parallax detector only as a trigger. During the stationary
interval, estimate gravity and biases using the existing stationary-VI machinery.
At exit, estimate one bounded scale correction from visual translation and IMU
excitation, then inject it once into the active visual state with covariance and
sanity checks.

Safety conditions:

- require sufficient accelerometer excitation and estimator observability;
- reject corrections outside a conservative scale interval;
- log proposed/applied scale and residual change;
- never apply repeatedly or when the visual-only trajectory is already healthy.

Gate: MH_01 post-hover scale returns near 1.0-1.3, tracked >= 0.98, and MH_03 plus
one non-hover sequence show no material regression. Score this as mono+IMU, never
as a monocular result.

### B2 — Structure-less registration and submap recovery (3-5 weeks)

Attack the measured courtyard/ordered-gap bottleneck: images that have good
two-view support but too few correspondences to already-triangulated 3D points.

1. Add relative-pose-based structure-less image registration against multiple
   registered neighbours.
2. Grow a local submap when direct PnP repeatedly fails despite a verified local
   component.
3. Align/merge the submap through shared tracks using robust SE3/Sim3, then run a
   guarded joint BA.
4. Preserve COLMAP-style retry accounting and reject merges that worsen robust
   residuals or violate rotation consistency.

Gate: improve MH_03 300 beyond 299/300 and increase ETH3D courtyard registration
above 14/38 without worsening common-subset RMSE. A connected view graph alone is
not success.

### B3 — Multi-view track quality before more BA (2-4 weeks)

- Enforce one observation per image per track and diagnose one-to-many conflicts.
- Score tracks by baseline, descriptor consistency, and multi-view reprojection,
  not only pairwise inlier membership.
- Add geometry-guided rematching only after a trustworthy model exists.
- Adapt feature density to uncovered image regions and weak 3D support, avoiding
  the already-measured failure of mixing arbitrary per-image densities.
- Retriangulate from the widest supported baseline and retain competing
  hypotheses until multi-view evidence resolves them.

First diagnostic slice (2026-07-19): track construction now reports verified
input correspondences, connected components, same-image conflicts, observations
lost with conflicted components, and retained track/observation counts. On the
frozen MH_03 300 graph, legacy union-find consumed 2,033,967 verified
correspondences but found 2,352 conflicted components containing 255,974
observations; only 24,038 tracks / 248,633 observations survived. Thus roughly
half of the observations entering component extraction are swallowed by
same-image-conflicted components before triangulation.

Two controlled attempts to rescue those components by edge ordering were
negative. Ranking image pairs by verified support and rejecting only unions
that introduced a duplicate image retained 31,878 tracks / 502,758 observations
after rejecting 47,060 edges, but reconstruction fell to `287/300`, reprojection
rose `0.570 -> 0.718 px`, SfM time rose `1185.7 -> 1193.7 s`, and common-pose
Sim(3) ATE worsened `0.3568 -> 0.4219 cm` (18.2%). This proves that pair support
does not disambiguate erroneous transitive chains. The experimental builder is
not retained; the reusable conflict diagnostics remain.

Ordering individual edges by mean squared descriptor distance before pair
support was worse still. It retained 31,778 tracks / 503,648 observations after
rejecting 39,434 conflict-forming edges, but growth stopped at `288/300` and the
bounded post-refinement pass reached only `297/300`. Mean reprojection rose to
`0.698 px` and SfM took `2684.2 s` (2.26x the frozen legacy run). On the common
250 GT-associated poses, Sim(3) ATE was `15.6735 cm` versus legacy `0.3505 cm`,
a 44.7x catastrophic regression with a 0.00782 recovered scale. The negative
manifests are
`mono-sfm-visloc-b3-{pair-support,descriptor}-conflict-MH_03_medium-*.json`.

The next B3 mechanism must start from the clean legacy model, revisit dropped
components only after poses are trustworthy, and select at most one observation
per image through multi-view reprojection hypotheses. Require at least three
views, sufficient baseline, and a residual-improvement gate; unresolved
components stay dropped. Local descriptor or pair-support ordering alone is no
longer an authorized promotion path.

Geometry-guided recovery slice (implementation, 2026-07-19): the default path
still builds and refines only clean union-find tracks. An opt-in bounded pass
then revisits dropped components against those trusted poses. It ranks at most
32 verified anchor edges per component by posed-view parallax, triangulates a
hypothesis under cheirality and the strict baseline gate, selects at most one
observation per image, and requires at least three registered views, <=2 px per
observation, <=1 px mean error, plus an independent verified-edge cycle. At most
one hypothesis per conflicted component enters recovery. Incomplete models may
run one global BA; its added tracks, points, and changed poses are restored
byte-for-byte if the clean tracks' mean reprojection rises by more than 0.1% or
recovered residuals miss their gate. Complete models never re-solve poses.
Synthetic tests cover correct split/admission, rejection without a cycle,
complete-model pose immutability, and byte-identical rollback.

The frozen MH_03 300 A/B admitted 992 tracks / 65,522 observations from the
2,352 dropped components. The guarded solve moved clean-track mean reprojection
only `0.571941 -> 0.572492 px` (0.096%, within the 0.1% cap), while recovered
tracks measured `0.595910 px`. Final registration stayed `300/300`; structure
grew 8,254 -> 9,251 points and 131,742 -> 197,433 observations. On all 253
common GT-associated poses, Sim(3) ATE improved `0.3487 -> 0.3364 cm` (3.55%).
SfM time rose `1185.7 -> 1534.1 s`, still below COLMAP's `2260.6 s`, whose ATE
was `0.6163 cm`. Thus the development gate passes on accuracy, registration,
and mapper runtime, unlike both local edge-order attempts. The manifest is
`mono-sfm-visloc-b3-geometry-conflict-MH_03_medium-20260719T060907Z.json`.

The identical frozen configuration failed held-out MH_01 despite passing every
internal residual gate. It admitted 2,280 tracks / 66,438 observations and kept
`300/300`, but Sim(3) ATE worsened `0.9255 -> 0.9398 cm` (1.55%) and SfM rose
`433.6 -> 542.3 s`; COLMAP remained better at `0.9116 cm`. This proves that a
slightly better or nearly unchanged reprojection objective does not authorize
another pose solve on an already-complete trajectory. The negative manifest is
`mono-sfm-visloc-b3-geometry-conflict-MH_01_easy-20260719T062240Z.json`.

The resulting dataset-independent safety rule is now explicit: if every image
is registered before conflict recovery, recovered landmarks are structure-only
and poses remain byte-identical; only an incomplete model may run the one
guarded pose BA needed to unlock missing-image PnP. Synthetic tests pin both
branches. The revised MH_01 regression confirms the contract in the real
pipeline: it recovered the same 2,280 tracks / 66,438 observations, reported
`pose-ba=false`, retained `300/300`, and its exported trajectory SHA-256 is
byte-identical to the frozen legacy trajectory. Common-278 Sim(3) ATE is
therefore exactly preserved at `0.9255 cm`; SfM took `484.0 s`, including only
`2.042 s` for geometry recovery. The supporting manifest is
`mono-sfm-visloc-b3-geometry-structure-only-MH_01_easy-20260719T063841Z.json`.
Because MH_01 informed this revision, it is a regression sequence, not fresh
held-out evidence; the revised policy must validate on MH_05 or a declared
ETH3D sequence before promotion. The feature remains off by default.

The frozen mechanism then failed the fresh MH_05-difficult 300 head-to-head.
It registered `276/300` (including 95 images from the bounded post-refinement
pass), recovered 425 tracks / 10,789 observations with `pose-ba=true`, and
measured `21.96 cm` common-252 Sim(3) ATE. COLMAP registered `300/300` at
`1.263 cm`. Visloc remained much faster (`425.9 s` including feature extraction
versus COLMAP's `1820.7 s`) but lost both completeness and accuracy, so the B3
mechanism is not promoted and remains off by default. The negative manifest is
`mono-sfm-visloc-MH_05_difficult-20260719T072220Z.json`; the paired COLMAP
manifest is `mono-sfm-colmap-MH_05_difficult-20260719T072220Z.json`.

The same-cache no-recovery isolation proves B3 is actively harmful here, not
merely insufficient. With recovery disabled, the identical mapper registered
`294/300` rather than `276/300`, the post-refinement pass added `113` rather
than 95 images, mean reprojection improved `0.639 -> 0.595 px`, and common-251
Sim(3) ATE improved `21.980 -> 21.786 cm`. Recovery was faster in SfM time
(`242.3` versus `275.5 s`) only because it finished with 18 fewer cameras.
The no-recovery supporting manifest is
`mono-sfm-visloc-b3-no-geometry-MH_05_difficult-20260719T073034Z.json`.

Without B3, the remaining six missing frames are `79,86,186,240,242,244`.
This is a measured structure-less registration gap suitable for B2 recovery
work, not a reason to lower PnP thresholds. The `21.786 cm` trajectory error
also shows that completing those six poses alone will not close the COLMAP
accuracy gap; relative-pose/submap consistency must be scored independently.

B2 first-slice status (2026-07-19): an opt-in, one-sweep multi-neighbour
structure-less registration pass is now implemented. It independently
re-estimates essential geometry against registered neighbours, requires a
three-edge rotation and translation consensus, recovers metric-in-model scale
from the intersection of neighbour camera-centre direction lines, and rejects
single-pair arbitrary scale, parallel lines, reversed directions, rotation
outliers, and translation outliers. Tentative insertion, triangulation, and BA
are transactional. Admission requires at least 20 supported tracks, at most
2 px image mean reprojection, no more than 0.1% increase on points that existed
before insertion, and a final relative-geometry recheck. The local BA varies
only the recovered camera and its incident landmarks; every already-registered
camera is a fixed submap boundary.

On the frozen MH_05 cache this safely recovered frame 240:

| MH_05 first 300 | no B2 recovery | B2 first slice | COLMAP 4.1 |
|---|---:|---:|---:|
| registered | 294/300 | **295/300** | 300/300 |
| points / observations | 6085 / 59444 | **6109 / 59969** | 11958 / n/a |
| mean reprojection | **0.595 px** | 0.597 px | 0.413 px |
| SfM / mapper runtime | **275.5 s** | 293.8 s | 1798.8 s |
| common-265 Sim(3) ATE | 21.5235033 cm | **21.5235033 cm** | 1.2469148 cm |

The pre-existing 294 camera poses are numerically identical to the disabled
arm; the newly added frame 240 has 7.94 cm GT error under the Sim(3) fitted only
to those common poses, so the completion is not merely a registration-count
win. Six deterministic geometry tests plus a fixed-boundary BA test pass.

Frame 86 exposed the next architectural boundary. Its unrefined consensus pose
had only 40 supported tracks at 2.549 px. Reprojection BA reached 167 tracks at
0.664 px and preserved the clean residual, but left the independent rotation
consensus at about 3.90 degrees, outside the 3-degree gate. Lifting the robust
edges into the existing isotropic SE(3) BA at 1x and 10x inlier weight changed
that result by only 0.003 degrees; this factor formulation is a controlled
negative and is not retained. The gate was not loosened.

B2 bounded-local-submap status (2026-07-19): verified consensus edges can now
synthesize independent local landmarks without duplicating observations owned
by the existing 3D map. Two-view local landmarks are allowed because the camera
pose itself still requires an independent three-neighbour relative-pose
consensus. Previously registered cameras and landmarks form a fixed boundary,
and rejected candidates restore the complete track/pose/point state. When the
ordinary new-camera BA crosses the relative-geometry boundary, a deterministic
trust-region projection searches camera rotation and centre independently.
It first uses a bounded 5% grid, then refines only the best near-admissible cell
at 1% resolution. Every candidate must still pass the original 3-degree,
20-track, 2 px, direction/line, and clean-map +0.1% gates.

This recovers frame 86 as well as frame 240 on the same frozen cache:

| MH_05 first 300 | no B2 recovery | B2 local submap | COLMAP 4.1 |
|---|---:|---:|---:|
| registered | 294/300 | **296/300** | 300/300 |
| points / observations | 6085 / 59444 | **6237 / 60761** | 11958 / n/a |
| mean reprojection | **0.595 px** | 0.640 px | 0.413 px |
| SfM / mapper runtime | **275.5 s** | 301.4 s | 1798.8 s |
| common-265 Sim(3) ATE | 21.5235033 cm | **21.5235033 cm** | 1.2469148 cm |

All 294 pre-existing pose values remain byte-identical. Frame 86 is admitted at
rotation-alpha 0.02 / centre-alpha 0.40 with 22 supported tracks, 1.990 px mean
reprojection, 2.852-degree worst consensus rotation error, and no measurable
change in the clean-map mean (`0.594518 -> 0.594518`). Under the Sim(3) fitted
only to common baseline poses, frame 86 has 12.879 cm GT error and frame 240 has
7.945 cm. Eleven focused structure-less/local-submap tests and the complete
workspace test suite pass; the three Python evaluation suites pass 8/8. The
supporting manifest is
`mono-sfm-visloc-b2-adaptive-local-submap-MH_05_difficult-20260719T110329Z.json`.
The remaining frames are `79,186,242,244`. Completion and the large COLMAP
accuracy gap remain open; the next slice must generalize recovery beyond this
single missing-frame basin and then improve the shared trajectory, not relax
admission thresholds.

B2 full-registration status (2026-07-19): the centre-line estimator now treats
its robust fit monotonically. After each weighted least-squares refit it drops
only edges that no longer satisfy the unchanged forward-direction and line
residual gates, refits the surviving set, and rejects the candidate if fewer
than three independent neighbours remain. This fixes a real estimator bug:
previously a valid RANSAC set could be rejected wholesale because its refit
moved one marginal edge across the sign boundary. A focused regression test
pins this case; no admission threshold was loosened.

On the same frozen MH_05 cache, the corrected estimator recovers all four
remaining frames and reaches `300/300`:

| MH_05 first 300 | no B2 recovery | B2 corrected refit | COLMAP 4.1 |
|---|---:|---:|---:|
| registered | 294/300 | **300/300** | **300/300** |
| points / observations | 6085 / 59444 | **6641 / 63532** | 11958 / n/a |
| mean reprojection | **0.595 px** | 0.644 px | 0.413 px |
| matching + SfM / mapper runtime | 311.7 s | **308.2 s** | 1798.8 s |
| all-300 Sim(3) ATE | n/a | 21.657 cm | **1.248 cm** |

The 294 baseline poses are byte-identical. Under a Sim(3) fitted only to the
baseline-common poses, recovered-frame translation errors are 16.14, 12.88,
60.79, 7.94, 8.64, and 9.56 cm for frames 79, 86, 186, 240, 242, and 244.
The common-265 ATE remains exactly 21.5235033 cm. Registration completeness is
therefore solved without damaging the trusted submap, while the localized
trajectory bulge near frames 150--190 remains an accuracy problem. The
supporting manifest is
`mono-sfm-visloc-b2-refit-reclassification-MH_05_difficult-20260719T121723Z.json`.

A development-only wide-constraint experiment demonstrates that the remaining
accuracy gap is not intrinsic to the feature measurements. Adding fixed
offsets 20 and 30 to the existing 8/12 graph (stride 2) registers all 300
images through ordinary growth and obtains 1.1606 cm all-300 Sim(3) ATE,
beating COLMAP's 1.2475 cm by about 7%. It also remains faster at 792.9 s for
matching plus SfM versus COLMAP's 1798.8 s mapper time. Because MH_05 selected
these offsets, this is a development upper-bound result, not held-out evidence.
Its supporting manifest is
`mono-sfm-visloc-b4-wide-skip-dev-MH_05_difficult-20260719T121724Z.json`.

The exact frozen configuration fails to generalize to MH_03: both systems
register `300/300`, but visloc degrades to 4.5017 cm ATE versus COLMAP's
0.6163 cm, with 1066.5 s matching plus SfM. This controlled held-out negative
shows that placing all wide correspondences into the initial union-find tracks
can corrupt track topology even when every pair passes two-view verification.
Fixed wide skip edges are therefore not promoted. The negative manifest is
`mono-sfm-visloc-b4-wide-skip-heldout-negative-MH_03_medium-20260719T121725Z.json`.

An additional pose-only experiment keeps wide matches out of the track graph,
admits 205 independently gated relative-pose edges, and reduces pose-graph cost
from 53.259 to 13.208. Landmark-only BA then loses valid observations and raises
mean reprojection from 0.644 to 80.060 px, so the transactional quality gate
restores the complete base reconstruction. This proves that relative-pose
factors alone do not carry the multi-view information that produced the MH_05
gain. The next slice is geometry-guided wide-track integration after a trusted
base reconstruction: validate correspondences with existing poses, merge only
compatible track components, retriangulate multi-view landmarks, and commit
joint refinement only when clean-track, completeness, and held-out gates pass.

The first post-trusted-map implementation now covers three increasingly strong
cases while keeping that transaction boundary: (1) attach an unowned wide
keypoint only when an existing landmark reprojects within 2 px; (2) merge two
disjoint landmark fragments only when both 3D estimates reproject within 2 px
over every observation of the other fragment; and (3) create a new landmark
from an unowned wide component only with at least three distinct images, at
least 2 degrees parallax, valid cheirality, and at most 2 px error in every
view. Greedy ownership prevents duplicate image/keypoint assignments, two
poses fix the monocular BA gauge, and commit additionally requires preservation
of every clean observation, no more than 0.1% clean-error regression, every new
observation valid, and maximum pose correction below 10% of scene diameter.

On MH_05, the strongest arm safely committed 18 fragment merges, 798 new
multi-view tracks / 4,901 observations, and 846 further observations. It kept
`300/300`, increased structure from 6,641 / 63,532 to 7,421 points / 69,279
observations, and finished at 0.630 px mean reprojection. Nevertheless all-300
Sim(3) ATE remained effectively unchanged (`21.6569 -> 21.6570 cm`); maximum
pose correction was only 0.0114 in a 17.876 reconstruction diameter. The
earlier attachment-only and attachment+merge arms were equally ATE-neutral.
This is a controlled neutral result, not a promotion: geometry-safe structure
completion after convergence cannot move the mapper out of the bad trajectory
basin. The next implementation must rebuild complete base+wide components
against the trusted map and use them to form an alternative initialization or
growth hypothesis, then select between reconstructions transactionally. It
must not put unfiltered wide edges back into initial union-find, which already
failed MH_03 held-out generalization.

B4 transactional hypothesis status (2026-07-19): the mapper now grows two
independent monocular reconstructions when `--wide-hypothesis` is enabled. The
trusted arm uses offsets 8/12; the candidate uses the merged 8/12/20/30 graph.
No poses, tracks, or BA state are shared. The candidate replaces the trusted
result only if it does not reduce registration, retains at least 90% of final
landmarks, adds at least 25% valid observations, and strictly lowers mean
reprojection. Mapper failure or any failed conjunct leaves the trusted result
byte-identical.

This turns the earlier MH_05 upper bound into an end-to-end selectable result:

| MH_05 first 300 | trusted arm | selected wide arm | COLMAP 4.1 |
|---|---:|---:|---:|
| registered | 300/300 | **300/300** | **300/300** |
| points / observations | 6641 / 63532 | 6513 / **101178** | 11958 / n/a |
| mean reprojection | 0.644 px | 0.572 px | **0.413 px** |
| matching + all mapper work | 308.2 s | **861.6 s** | 1798.8 s |
| all-300 Sim(3) ATE | 21.657 cm | **1.1606 cm** | 1.2475 cm |

The GT-free selector observes 98.1% point retention, 159.3% observation count,
and lower residual, so it accepts. The selected `images.txt` is SHA-256
identical before and after the later preflight optimization. Thus, on this
development sequence visloc now simultaneously beats COLMAP accuracy,
registration, and mapper runtime. This is still development evidence because
MH_05 selected the wide offsets and hypothesis policy.

The policy was then transferred unchanged to MH_03. The completed wide arm
showed the harmful signature (31.5% point retention, 72.6% observations, and
`0.570 -> 0.708 px`), was rejected, and left the trusted output at `300/300`
and 0.3487 cm ATE versus COLMAP's 0.6163 cm. However, fully computing a rejected
arm raised SfM time to 3091.7 s, failing the runtime gate despite correct
selection.

A public track-build preview now exposes candidate topology before seed search,
triangulation, PnP, or BA. A preflight requires at least 82% retained tracks and
85% retained observations relative to the trusted graph. MH_05 passes at
85.6% / 90.5%; MH_03 rejects at 77.6% / 81.9%. The optimized MH_03 run writes an
`images.txt` byte-identical to the full-selector trusted result, retains the
same 0.3487 cm ATE, and reduces matching + SfM to 1330.3 s, again faster than
COLMAP's 2260.6 s mapper. Because the MH_03 negative informed these preflight
thresholds, MH_03 is now a regression sequence, not fresh held-out evidence.
The complete policy (preflight plus final selector) must be frozen and tested on
a new sequence such as MH_04 or a declared ETH3D sequence before promotion.

Fresh held-out MH_04 result (2026-07-20): the complete configuration was frozen
before any MH_04 feature, mapper, or GT output was inspected. The preflight
passed at 83.9% retained tracks / 87.4% retained observations, so both arms ran.
The final selector then correctly rejected the candidate: it retained 95.9% of
points but only 97.3% of observations (below the required 125%) and worsened
reprojection from 0.584 to 0.588 px. The trusted reconstruction was committed.

| MH_04 first 300, fresh held-out | selected visloc | COLMAP 4.1 Ceres |
|---|---:|---:|
| registered | **300/300** | **300/300** |
| points | 3634 | **6748** |
| mean reprojection | **0.584 px** | 0.738 px |
| engine wall incl. own features | **1104.4 s** | 2315.7 s |
| common-267 Sim(3) ATE | **1.0648 cm** | 6.5102 cm |
| common-267 maximum error | **2.0232 cm** | 16.6803 cm |

This is the first untouched-sequence evidence that the frozen transactional
policy beats COLMAP simultaneously in accuracy, registration, and speed while
rejecting an unnecessary wide arm. The original requested COLMAP 4.1 global
Caspar arm is a DNF on the installed official binary: incremental mapper exits
before reconstruction with the explicit invariant
`ba_global_backend != BundleAdjustmentBackend::CASPAR`. The same already-built
SIFT database and sequential matches were therefore mapped with Ceres; GT was
still unopened until both engines exited. The head-to-head runner now records
the requested/effective backend and performs this fallback only for that exact
unsupported-Caspar failure; unrelated mapper failures remain fatal.

Gate: MH_03 300 reaches the first accuracy/registration gate or a controlled
ablation proves the remaining error is optimizer-, not track-limited. Tier-C
terrace must not regain the bent-shape failure.

### A3 — Sound long-range loop closure (3-5 weeks)

Split the problem into two independently scored stages:

1. retrieval recall: the known MH_01 revisit and labelled ETH3D/KITTI loops must
   appear in top-K;
2. geometric precision: every accepted edge must pass 2D geometry, trusted-
   rotation consistency, spatial coverage, and robust residual-improvement gates.

Use stored image keypoints/descriptors to build loop geometry; do not treat
drifted DPVO patch depths as ground truth. Only after a sound relative pose is
established should a 3D-3D Sim3 scale be estimated. Keep all M11/M12 catastrophic
guards.

Gate: >= 90% labelled-loop recall at the chosen K, zero accepted false loops on
the development suite, and improved full-sequence ATE on at least two sequences.

Stage-1 baseline (2026-07-21): `scripts/eval_dpvo_long_loop_recall.py` labels
GT revisit pairs (position radius, camera optical-axis angle < 30 degrees,
arrival gap >= the index's own `min_temporal_gap`) and scores the demo's
`long_loop_candidates.csv` against them; 28 pytest tests pin the metric.
Scoring the archived M12 MH_01 runs (stride 2, defaults `min_temporal_gap=150`,
`top_k=3`, `query_frequency=40`, verified against source and empirically
against all three CSVs) revises the M12 interpretation: conditioned on a query
being issued, recall@K is 1.0000 at every K in {1,3,5,10} for both 1.0 m and
0.5 m radii on all three runs. The failure is opportunity coverage: only 13
(800f) / 3 (400f) distinct query arrivals appear at all, so 98.3% of
labelled-revisit query arrivals (353/359 at r=1.0) were never queried, and no
query ever landed within +/-5 arrivals of the tightest GT revisit (42,456).
The M11/M12 "retrieval never surfaces the best revisit" gap is therefore a
query-cadence/logging gap, not a vocabulary-ranking gap, on this evidence.
Stage-1 next slice: densify query cadence (query_frequency ~1-5, plus logging
of zero-candidate queries so issued-but-empty queries become visible), then
re-measure; only if issued-query recall drops materially at dense cadence does
vocabulary/aggregation work re-enter scope. Baseline JSONs:
`E:/visloc_archive/dpvo_a3_20260721/recall_baseline/`.

Stage-1b densified-cadence A/B (2026-07-22): `query_frequency` is now the demo
flag `--ll-query-frequency` (default 40 unchanged), issued-but-empty queries
are counted (`queries_issued_total`, `queries_with_zero_candidates`) and
dumped as `rank=-1` CSV rows, and the recall harness counts them as issued
misses (23 Rust + 33 Python tests). Same-binary MH_01 800f A/B, M12
spanchored-final mechanism flags, seed 0:

- qf40 control-compat arm: byte-identical ATE/scale/tracked to the M12
  baseline (4.0866 / 3.3224 / 16.019 / 1.0), 20 issued queries (7 empty —
  previously invisible), corrected recall@K 6/9 = 0.667 (the earlier 1.0000
  was conditioned on non-empty queries only).
- qf5 dense arm: 159 issued queries, opportunity-coverage miss falls
  97.5% -> 79.9% (5x more GT-revisit arrivals actually queried), and queries
  finally land at arrivals {453, 458}, within +/-5 of the tightest GT revisit
  (42,456) — but candidate 42 never enters top-3 (best-ranked candidates are
  arrivals 199/196 at similarity 0.39/0.30). Issued-query recall drops to
  41/72 = 0.569. Per the stage-1 trigger above, vocabulary/ranking work is
  now formally in scope: the tightest revisit is a proven ranking miss, not a
  cadence miss.
- Un-tuned side effect: density surfaces 5 newly accepted long loops
  (220->388 ... 282->443, similarity 0.78-0.84, rotation disagreement
  15.7-19.8 degrees, all just under the 20-degree gate) which drive
  `ate_similarity_scale` 16.0 -> 0.278 and rigid ATE 4.09 -> 9.71 (sim ATE
  improves 3.32 -> 2.01). This reproduces M12's "passes every gate, still
  physically wrong" failure mode at higher density and is direct evidence for
  stage 2's prescription: loop geometry must come from stored 2D
  keypoints/descriptors (2D-2D relative pose first), not from drifted patch
  depths, before any Sim3 scale is applied. No gates were tuned or loosened.

Runs: `E:/visloc_archive/dpvo_a3_20260721/on_800_qf{40,5}`, recall JSONs in
`.../recall_qf_ab/`. Note: both arms ran concurrently with the long-running
MH_03 full-sequence SfM benchmark's mapper (~1 core); ATE/scale are
deterministic and unaffected, per-frame timing from these runs is diagnostic
only.

### B4 — Win the runtime without weakening geometry (3-5 weeks)

Profile first, then optimize the largest measured stages:

- pooled ONNX sessions and batched/pipelined pair inference;
- bounded temporal matching plus periodic retrieval instead of all-pairs work;
- local BA over a covisibility window;
- landmark/observation selection for global BA;
- block-sparse solve reuse and deterministic parallel scheduling;
- growth-triggered global refinement, with final full solve only when it changes
  the solution materially.

Gate: beat the fresh best COLMAP 4.1 wall time on MH_03 300 while retaining its
accuracy/registration gate; then demonstrate scaling on a full EuRoC sequence.

### A4/B5 — Generalization and release candidate (2-4 weeks)

- Freeze one configuration per sensor/dataset family before held-out runs.
- Run full EuRoC, ETH3D SLAM, KITTI guards, and unordered Tier C.
- Repeat critical comparisons three times and record median/worst.
- Produce failure galleries for every lost frame, rejected registration, and
  rejected/accepted loop.
- Promote mechanisms to defaults only after the full matrix passes.

Gate: the program-level win conditions in sections 3.2 and 3.3. If the SfM result
wins accuracy/registration but not runtime, call it accuracy parity/a win on
quality, not “beats COLMAP” overall.

## 6. Execution order and resource allocation

Recommended monocular order: P0 -> A1 -> B1 -> B2 -> B3 -> A3 -> B4 -> A4/B5.
A2 (visual-inertial recovery) remains a documented fallback, not an authorized
next step; starting it requires a new explicit decision.

This alternates short monocular SLAM and SfM milestones so neither product becomes a long
unvalidated branch. Until the MH_01 scale cliff is closed, use roughly 40% of
experiment time for SLAM and 60% for Sequential SfM, because the explicit
COLMAP-beating goal requires a new baseline and several mapper changes. After A2
passes, rebalance based on the largest measured gap.

Each milestone ends with exactly one of: pass and integrate; honest negative and
stop that mechanism; or inconclusive with a named missing measurement. Do not
carry an inconclusive experiment into the next architectural layer.

## 7. Immediate next actions

1. Stop B3 geometry-conflict recovery after the controlled MH_05 negative; keep
   it off by default and retain only its diagnostics and safety tests.
2. Extend the safe B2 first slice into bounded local-submap recovery for MH_05's
   remaining five frames `79,86,186,242,244` (then ETH3D), scoring both
   completion and relative-pose/Sim3 consistency; do not weaken PnP or
   structure-less admission thresholds.
3. Resume B4 global-refinement cadence work only after the recovered-track
   mechanism preserves `300/300` and the frozen common-subset ATE; density that
   merely makes BA slower is not progress.

## 8. Stop conditions

- Stop adding detector-specific hover heuristics after A1 if scale remains >= 10.
  Continue the monocular track through sound long-range retrieval, relocalization,
  loop geometry, and submap Sim3 recovery; do not silently switch to IMU.
- Stop lowering matching/PnP thresholds if registration rises but independent-GT
  error or robust residuals worsen. The terrace and M5 rescue failures already
  demonstrate this trap.
- Stop optimizing BA until profiling attributes the dominant error/runtime to BA;
  M1-M6 showed that view/track support can dominate even with a capable mapper.
- Never loosen loop rotation/geometric gates to manufacture loop recall.
- Do not claim a COLMAP speed win until the COLMAP 4.1 Caspar baseline is in the
  registry.

## 9. Primary references

- Existing project evidence: `docs/dpvo_droid_port_plan.md`,
  `docs/colmap_port_plan.md`, `docs/sfm_vs_colmap_benchmark.md`, and
  `docs/euroc_sfm_benchmark.md`.
- COLMAP 4.1 changelog and current sequential-matching documentation:
  https://colmap.github.io/changelog.html and
  https://colmap.github.io/tutorial.html
- EuRoC dataset: https://projects.asl.ethz.ch/datasets/euroc-mav/
- ETH3D SLAM benchmark: https://www.eth3d.net/slam_overview
- KITTI odometry benchmark: https://www.cvlibs.net/datasets/kitti/eval_odometry.php
