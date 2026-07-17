# COLMAP → Rust Port Plan

Scope: which parts of COLMAP are worth porting into `visloc-rs`, in what order,
and why — driven by the project's own measured benchmark gap, not by COLMAP's
pipeline order. This is a design document; no Rust was changed to produce it.

## TL;DR

- **The measured gap is not the incremental mapper.** `docs/sfm_vs_colmap_benchmark.md`
  already shows visloc-rs at COLMAP grade or faster on ordered metric video and
  on dense unordered orbits (South Building, Gerrard Hall), using its own
  from-scratch Rust incremental SfM (`pipelines/slam/src/incremental_sfm.rs`),
  which has already absorbed COLMAP's `IncrementalMapper` schedule: P3P
  registration, multi-seed robust init, visibility-pyramid next-best-view,
  local BA + growth-triggered global refinement + registration retries,
  re-triangulation, image filtering, and joint intrinsics/distortion
  self-calibration. **On independent laser-scan ground truth (ETH3D)**, however,
  COLMAP is ahead — 0.15–0.16 cm vs 0.36–0.37 cm on well-behaved scenes, and
  categorically more conservative on `terrace` (COLMAP under-registers 8/23 but
  stays tight; visloc-rs over-commits to 23/23 with a globally bent shape). The
  repo's own diagnosis, after ruling out track density and intrinsics
  refinement by ablation, is: **"a frontend detection-coverage and view-graph
  problem, not a mapper-schedule problem."**
- **License: clean.** COLMAP core (`src/colmap/*`) is BSD-3-Clause
  (ETH Zurich / UNC Chapel Hill). A Rust port of the core algorithms is legally
  unencumbered. Two third-party subdirectories must **not** be ported/depended
  on: `src/thirdparty/LSD` (AGPLv3 — copyleft) and `src/thirdparty/SiftGPU`
  (UNC's own non-commercial "educational, research and non-profit" clause, not
  OSI-approved). COLMAP's CPU SIFT path already uses `src/thirdparty/VLFeat`
  (BSD, standard 2-clause style) instead of SiftGPU, so **VLFeat is the
  legally-clean reference for SIFT algorithm behavior** if a SIFT port is ever
  undertaken. `PoissonRecon` (MIT) and `Symforce-Caspar` (Apache-2.0) are also
  clean but are dense-reconstruction/GPU-BA-codegen concerns, out of scope here.
- **Recommended first milestone: M1 — two-view geometric-verification parity +
  a real correspondence graph.** This attacks the diagnosed gap directly (view
  graph quality) rather than re-porting the mapper visloc-rs already has.
  Acceptance criterion: on ETH3D `courtyard`/`terrace`/`office`, the fraction of
  VLAD-proposed candidate pairs COLMAP-style multi-model verification
  reclassifies as `DEGENERATE`/`PANORAMIC`/`WATERMARK` (today silently accepted
  or rejected by a single essential-matrix RANSAC) is measured, and the `terrace`
  bent-shape failure (1.5 px reprojection, tens-of-cm camera error) is reduced
  by rejecting the degenerate pairs that cause it — target: `terrace` common-subset
  Sim(3) RMSE materially closer to COLMAP's 0.41 cm than today's 17.43 cm,
  without regressing South Building / Gerrard Hall / EuRoC (still ≤ current cm
  figures within noise).

---

## 1. COLMAP architecture inventory (from `github.com/colmap/colmap`, `main`)

COLMAP's `main` branch has absorbed GLOMAP (global SfM) as a first-class mapper
and added ONNX/ALIKED learned-feature support since the 4.x release line; the
inventory below is against that current tree, cross-checked against the pinned
`4.1.0` release used in the repo's own benchmarks (`fa8e3b3`).

### 1.1 Feature extraction (`src/colmap/feature/`)

- `sift.h` / `sift.cc` — `SIFTExtractionOptions`: `max_num_features` (default
  8192), pyramid (`first_octave=-1`, `num_octaves=4`, `octave_resolution=3`),
  peak/edge thresholds, `Normalization::{L1_ROOT, L2}`.
  - **Affine-covariant detection**: `estimate_affine_shape` — oriented-ellipse
    (covariant) keypoints instead of oriented disks, via VLFeat's covariant
    detector (`covdet.c`).
  - **Domain-size pooling (DSP-SIFT)**: `domain_size_pooling` +
    `dsp_{min,max}_scale`, `dsp_num_scales` — averages the SIFT descriptor over
    a range of scales around the detected one (Dong & Soatto, CVPR'15); COLMAP
    auto-switches to the VLFeat covariant extractor when either affine or DSP
    is enabled.
  - `CreateSiftFeatureExtractor` — GPU (SiftGPU, OpenGL context) or CPU
    (VLFeat) backend selection.
  - `aliked.h/.cc`, `onnx_matchers.h/.cc`, `onnx_utils.h/.cc` — **COLMAP itself
    now runs learned ONNX features (ALIKED) and ONNX matchers** as an
    alternative to SIFT. This is the direct precedent for visloc-rs's
    SuperPoint/LightGlue frontend being a legitimate "COLMAP-style" pipeline,
    not a deviation from it.
- `extractor.h/.cc`, `matcher.h/.cc`, `index.h/.cc` — extractor/matcher
  interfaces and a nearest-neighbor feature index (FLANN-backed in the C++
  build).

### 1.2 Matching modes (`src/colmap/controllers/pairing.h/.cc`)

Pair-generator classes, each producing a candidate-pair stream fed to
geometric verification:

| Class | Strategy |
|---|---|
| `ExhaustivePairGenerator` | all `N·(N−1)/2` pairs, block-tiled |
| `SequentialPairGenerator` | consecutive-frame window + rig/loop-detection support |
| `VocabTreePairGenerator` | hierarchical vocab-tree retrieval (see §1.7) |
| `SpatialPairGenerator` | GPS/location-prior nearest neighbors |
| `TransitivePairGenerator` | transitive closure over already-verified pairs |
| `ImportedPairGenerator` | pre-computed pair list from file |
| `ExistingMatchedPairGenerator` | pairs already scored in the database |

### 1.3 Two-view geometric verification (`src/colmap/scene/two_view_geometry.h`, `src/colmap/estimators/two_view_geometry.cc`, `fundamental_matrix_degensac.cc`)

`TwoViewGeometry` carries `E`/`F`/`H` (whichever model won), `cam2_from_cam1`,
inlier matches, and a **`ConfigurationType`**: `UNDEFINED`, `DEGENERATE` (no
overlap / too few inliers), `CALIBRATED` (essential-matrix, calibrated),
`CALIBRATED_RIG`, `UNCALIBRATED` (fundamental-matrix), `PLANAR` (homography,
real baseline), `PANORAMIC` (homography, pure rotation), `PLANAR_OR_PANORAMIC`,
`WATERMARK` (pure 2D translation confined to image borders — the classic
stock-photo watermark artifact), `MULTIPLE` (inliers explained by more than one
non-degenerate model — e.g. a foreground plane plus background parallax).
Verification therefore **estimates multiple models (E, F, H) and a degeneracy
test**, not a single RANSAC. `fundamental_matrix_degensac.cc` is COLMAP's
DEGENSAC-style degeneracy-aware fundamental-matrix estimator (detects
dominant-plane/pure-rotation configurations that would make a plain F-matrix
solve unstable).

### 1.4 Scene / correspondence graph (`src/colmap/scene/correspondence_graph.h/.cc`)

`CorrespondenceGraph`: per image, the number of observations and a flattened
`(other_image, other_point2D)` correspondence list per own point2D; per image
pair, inlier match count + stored `TwoViewGeometry` (without the raw matches,
which live in the flattened per-point structure). Key methods:
`AddImage`/`AddTwoViewGeometry` (ingest), `FindCorrespondences`/
`ExtractCorrespondences` (matches for one point2D), `ExtractTransitiveCorrespondences`
(BFS-style multi-hop closure — the same operation `TransitivePairGenerator`
uses at the pairing level, but here at the point level, for track building),
`NumObservationsForImage`/`NumMatchesBetweenImages` (connectivity stats),
`Finalize` (flatten + drop unobserved images for memory). This is COLMAP's
persistent "view graph" object — the incremental mapper, the pair generators,
and track/point registration all query it rather than re-deriving connectivity
ad hoc.

### 1.5 Incremental mapper (`src/colmap/sfm/incremental_mapper.h/.cc`, `incremental_mapper_impl.h/.cc`, `incremental_triangulator.h/.cc`, `observation_manager.h/.cc`, `scene/visibility_pyramid.h/.cc`)

`IncrementalMapper`: `FindInitialImagePair` (seed selection, skipping
previously-failed pairs), `RegisterInitialImagePair`, `FindNextImages`
(next-best-view ranking), `RegisterNextImage` / `RegisterNextStructureLessImage`,
`TriangulateImage`, `AdjustLocalBundle` / `AdjustGlobalBundle` (Ceres),
`IterativeLocalRefinement` / `IterativeGlobalRefinement` (the growth-triggered
schedule), `FilterFrames` / `FilterPoints`. Next-best-view scoring uses
`scene/visibility_pyramid.h` — multi-resolution (`2×2…64×64`) occupancy grids
rewarding correspondence spread over raw count, exactly as described in the
Schönberger & Frahm 2016 paper.

### 1.6 Robust estimators (`src/colmap/optim/`)

`loransac.h` — Lo-RANSAC (local optimization: a refinement step on the current
best model before continuing sampling), `ransac.h`, `sprt.h` (Sequential
Probability Ratio Test early termination), `random_sampler.h` /
`progressive_sampler.h` / `combination_sampler.h` (PROSAC-style progressive and
exhaustive-combination samplers), `sparse_cholesky.h` (BA linear solve),
`least_absolute_deviations.h`. Paired with `fundamental_matrix_degensac.cc`
above for degeneracy-aware two-view robust fitting.

### 1.7 Retrieval / vocab-tree (`src/colmap/retrieval/`)

`visual_index.h/.cc` — hierarchical-k-means vocabulary tree +
`inverted_file.h`/`inverted_index.h` (TF-IDF-weighted inverted index over
visual words) for `O(log N)`-ish retrieval at thousands-of-images scale;
`vote_and_verify.h/.cc` — geometric spatial verification of retrieved
candidates using keypoint-position voting (RANSAC-free, cheap pre-filter
before full two-view verification); `geometry.h/.cc` — retrieval-specific
geometric utilities.

### 1.8 Global SfM / GLOMAP (`src/colmap/controllers/global_pipeline.cc`, `src/colmap/sfm/global_mapper.cc`, `src/colmap/estimators/{rotation_averaging,global_positioning,view_graph_calibration}.cc`)

**GLOMAP was a separate repository (`colmap/glomap`, BSD-3-Clause, ETH Zurich
2024) implementing a global SfM pipeline: rotation averaging over the whole
view graph, then a joint camera+point "global positioning" step (skipping
separate translation averaging + triangulation), claimed to match incremental
SfM's accuracy while being much faster because it avoids COLMAP's repeated
global BA.** As of the current COLMAP `main`/4.x line, **GLOMAP has been
merged into COLMAP itself as a first-class alternative mapper** (`--mapper
GLOBAL` / `global_mapper`, `automatic_reconstructor`); the standalone
`colmap/glomap` repo is now marked `[DEPRECATED]` and redirects here. License
stays BSD-3-Clause throughout — no separate-license complication.

This is architecturally the closest COLMAP component to visloc-rs's own thesis
(§"Why COLMAP loses on its own metric" in `docs/sfm_vs_colmap_benchmark.md`):
COLMAP's incremental mapper's repeated global BA is *why* it takes 11.7 h on
2700 frames. GLOMAP-style rotation-averaging + global positioning is COLMAP's
own answer to that cost, independently converging on the same "avoid repeated
global BA" idea visloc-rs already exploits via VO + one-shot pose-graph.

### 1.9 Reconstruction / IO (`src/colmap/scene/{reconstruction,track,rig,camera,pose_graph}.h/.cc`, `scene/reconstruction_io*.h/.cc`)

Standard model container + COLMAP text/binary IO — visloc-rs already has an
independent implementation of the on-disk format (see §2).

---

## 2. What visloc-rs already has — component-by-component verdict

| COLMAP component | visloc-rs equivalent | Verdict |
|---|---|---|
| SIFT / feature extraction | SuperPoint ONNX (`crates/vision/src/features/superpoint_onnx.rs`), LightGlue ONNX matcher (`lightglue_onnx.rs`) | **PARTIAL** — a modern learned frontend replaces SIFT entirely (COLMAP itself now supports this pattern via ALIKED/ONNX, §1.1); no DSP/affine-covariant detector, no classical SIFT path at all |
| Matching modes (exhaustive/sequential/spatial/transitive) | Sequential: temporal frame→frame (ordered SfM path). Unordered: VLAD top-K only (`crates/vision/src/place_recognition/mod.rs`) | **PARTIAL** — no exhaustive, spatial, or transitive-closure pairing generators; no vocab-tree (flat VLAD cosine over all images, not a hierarchical inverted index) |
| Two-view geometric verification | `EssentialRansac` / `RelativePoseEstimator` (`crates/vision/src/two_view/mod.rs`) — essential-matrix RANSAC only | **PARTIAL → gap.** No homography estimation, no `ConfigurationType` classification (`DEGENERATE`/`PANORAMIC`/`WATERMARK`/`MULTIPLE`), no DEGENSAC-style degeneracy-aware fitting. This is the concrete missing piece behind "view-graph quality" |
| Scene / correspondence graph | Ad hoc union-find over `PairwiseMatches` inside `incremental_sfm()`, rebuilt per call | **PARTIAL → gap.** No persistent, queryable graph object (no `FindCorrespondences`/`ExtractTransitiveCorrespondences`/connectivity stats reusable across ordered + unordered paths) |
| Incremental mapper: init pair selection | `seed_trials` multi-seed robust init, parallax-gated seed acceptance (`pipelines/slam/src/incremental_sfm.rs`) | **EXISTS** — COLMAP-equivalent robust initialization, unit-tested |
| Next-best-view / visibility scoring | `RankNextImages`-equivalent visibility pyramid (2×2…64×64), ported per `docs/sfm_vs_colmap_benchmark.md` | **EXISTS** |
| PnP minimal solvers | DLT + Grunert P3P (`crates/vision/src/pnp/{mod.rs,p3p.rs}`) | **EXISTS** (P3P added specifically to fix DLT's coplanar degeneracy) |
| Local BA / global BA scheduling | `colmap_style_mapper`: local BA + growth-triggered iterative global refinement + registration retries (`IncrementalSfmConfig`) | **EXISTS** — explicit port of COLMAP `Mapper.ba_local_num_images` / `ba_global_images_ratio` / `ba_global_max_refinements` / `max_reg_trials` |
| Re-triangulation / completion | `retriangulate` config flag, guarded re-seed | **EXISTS** (off by default; measured density-only, not accuracy, on this data) |
| Image / point filtering | `filter_images` (COLMAP `Reconstruction::FilterImages` port), track_filter_iterations | **EXISTS** |
| Joint intrinsics/distortion self-calibration | `refine_intrinsics` / `refine_distortion` inside Schur-complement BA (`pipelines/slam/src/bundle.rs`) | **EXISTS** — includes radial distortion `(k1,k2)` self-calibration, not just in COLMAP's controllers but as a first-class BA mode |
| Robust estimator: RANSAC + local refinement | `PnPRansac` with `GaussNewtonPoseRefiner` (LO-RANSAC-shaped: refine-inside-loop) | **PARTIAL** — has the local-optimization step but not COLMAP's SPRT early termination or progressive/combination samplers |
| Robust estimator: degeneracy-aware | GNC (graduated non-convexity, `pipelines/slam/src/gnc.rs`) | **PARTIAL, different paradigm** — GNC is a genuine robust-estimation alternative to LO-RANSAC/DEGENSAC (arguably more principled), but it is not degeneracy-*classifying* (H-vs-E model selection); it doesn't replace `ConfigurationType` |
| Global SfM (GLOMAP-style) | Loop-closure pose-graph optimization + one-shot solve (`pipelines/slam/src/pose_graph.rs`, `sim3_pose_graph.rs`) is architecturally the same "avoid repeated global BA" idea, but for VO trajectories, not photo collections | **MISSING** as a general unordered-photo-collection global SfM (rotation averaging + global positioning over an arbitrary view graph) |
| Vocab-tree / large-scale retrieval | Flat VLAD (`Vocabulary`/`vlad`/`retrieve_mutual`) | **PARTIAL** — no hierarchical clustering, no inverted-file TF-IDF, no `vote_and_verify` spatial pre-filter; scales `O(N)` per query, fine at hundreds of images, not built for thousands |
| COLMAP model IO (text/binary) | `crates/io/src/colmap/mod.rs`, full read/write, 3DGS export | **EXISTS** — independently implemented, already benchmarked (`crates/io/tests/colmap*.rs`) |
| Reconstruction container / covisibility | `pipelines/mapping`, `covisibility_ba.rs`, `map_atlas.rs` | **EXISTS** (map_atlas.rs is concurrently under active development by another agent for DPVO multi-session support — not touched here) |
| Dense MVS / Poisson surface reconstruction | none | **MISSING**, explicitly out of scope (`docs/colmap_compatibility.md` "Current Non-Goals": dense outputs) |

**Summary counts:** 8 EXISTS, 8 PARTIAL (of which 3 are the concrete gap: two-view
verification, correspondence graph, vocab-tree/retrieval), 2 MISSING (general
global SfM over unordered photo collections; dense MVS, out of scope).

---

## 3. Gap-ordered milestones

Ordered by measured deficit (ETH3D laser-GT gap), not COLMAP's pipeline order.
C2/C3 as originally scoped ("port next-best-view", "port BA scheduling") are
listed for completeness but are **already done** — reflected in reduced
effort/acceptance below rather than re-porting.

| # | Milestone | What to port (COLMAP files/classes) | What to reuse from visloc-rs | Acceptance criteria | Effort |
|---|---|---|---|---|---|
| **M1** | **Two-view verification parity: multi-model estimation + `ConfigurationType`** | `scene/two_view_geometry.h` (`ConfigurationType` enum, `TwoViewGeometry` struct), `estimators/two_view_geometry.cc` (E/F/H estimation + model selection + degeneracy test), `estimators/fundamental_matrix_degensac.cc` | `crates/vision/src/two_view/mod.rs`'s `EightPointEssentialMatrixEstimator`/`EssentialRansac` stay as the `CALIBRATED` path; add a homography estimator (4-point DLT, already have the linear-algebra machinery pattern from `pnp/mod.rs`'s DLT) and a degeneracy/model-selection function that picks E vs H vs DEGENERATE by inlier-count + GRIC-style comparison (COLMAP's actual test) | On ETH3D `courtyard`/`terrace`/`office`: (a) report the count of VLAD-proposed pairs reclassified away from a naive essential-matrix accept (i.e. now `DEGENERATE`/`PANORAMIC`/`WATERMARK`); (b) `terrace` common-subset Sim(3) RMSE improves from today's 17.43 cm materially toward COLMAP's 0.41 cm without regressing South Building (1.09 cm) / Gerrard Hall (0.68 cm) / EuRoC MH_03 (1.64 cm) outside noise | ~2–3 weeks (estimator math is well-understood; the model-selection threshold needs tuning against the same ETH3D scenes) |
| **M2** (was "C1" scene-graph half) | **Persistent `CorrespondenceGraph`** | `scene/correspondence_graph.h/.cc` | Replace the ad hoc union-find in `incremental_sfm()` with a queryable graph type; reuse it for both the unordered path and (optionally) the ordered stereo-VO track builder in `stereo_vo_ba.rs`, so both paths share one connectivity structure | New unit tests: `ExtractTransitiveCorrespondences`-equivalent returns the same track membership the current union-find produces on the existing `pipelines/slam/tests/*` fixtures (byte-identical tracks — a refactor gate, not an accuracy claim); enables M1's degeneracy stats and M4's transitive pairing to be queried in one place | ~1 week (mechanical refactor once M1 exists; do after M1 so the graph stores `ConfigurationType`, not just inlier matches) |
| **M3** (was "C2/C3") | **Harden, don't re-port, the mapper schedule** | (already ported: `IncrementalMapper::FindNextImages`/visibility pyramid, `IterativeLocalRefinement`/`IterativeGlobalRefinement`, `FilterFrames`/`FilterPoints`) | `colmap_style_mapper`, `filter_images`, `retriangulate` in `pipelines/slam/src/incremental_sfm.rs` — already exist and are unit-tested | Verify `colmap_style_mapper=true` is the *default* recommendation for ETH3D-class sparse wide-baseline sets (currently defaults to `false`); re-run the ETH3D battle with it on + the M1 verification and confirm no accuracy regression vs the existing 0.44 cm South-Building / 0.91 cm intrinsics-recovery results | ~2–3 days (config-default + re-benchmark, no new algorithm) |
| **M4** | **Vocab-tree-style retrieval (hierarchical + inverted file)** | `retrieval/visual_index.h/.cc` (HKM tree), `inverted_file.h`/`inverted_index.h` (TF-IDF), `vote_and_verify.h/.cc` | Keep `Vocabulary`/`vlad` as the local-to-global descriptor step; add hierarchical clustering (recursive k-means, reusing the existing deterministic-LCG k-means++ in `place_recognition/mod.rs`) and an inverted-file index on top, so retrieval is sub-linear at thousands-of-images scale; add a `TransitivePairGenerator`-equivalent using M2's graph | ETH3D scenes are only 23–38 images (VLAD is already adequate there — this milestone doesn't move the ETH3D number). Acceptance is a **new** benchmark: synthesize or source an unordered set of ≥2,000 images (e.g. concatenate multiple COLMAP example sets or an internet photo collection) and show retrieval wall-clock scales sub-linearly vs flat VLAD at equal or better registered-image count | ~3–4 weeks (largest single item; defer until M1/M2 land since verification quality matters more at any retrieval scale) |
| **M5** | **Benchmark harness + ETH3D laser-GT rematch** | n/a (infrastructure) | Extend `scripts/run_colmap_sfm_benchmark.sh` and the ETH3D battle scripts referenced in `docs/sfm_vs_colmap_benchmark.md` (`benchmarks/registry/runs/eth3d/`) to run automatically after M1–M3 land; add the M1 degeneracy-classification counts and M3's `colmap_style_mapper` A/B as first-class reported columns | A rerun of the exact ETH3D table in `docs/sfm_vs_colmap_benchmark.md` §"Independent ground truth" with all three scenes, both full-registered-set and common-subset RMSE, gated on **no regression** vs the numbers already in that doc and a **measured improvement** on `terrace`/`courtyard` | ~3–5 days engineering + reruns (COLMAP binary + ETH3D data must be re-provisioned first, see Risks) |

### Why this order deviates from the naive C1→C5 reading

The task brief's expected ordering (C1 view-graph, C2 next-best-view, C3
retriangulation/BA scheduling, C4 vocab-tree, C5 benchmark harness) assumed
next-best-view and BA scheduling were still open. Reading
`pipelines/slam/src/incremental_sfm.rs` and `docs/sfm_vs_colmap_benchmark.md`
shows they were **ported and measured already** (`colmap_style_mapper`,
visibility-pyramid `RankNextImages`, `filter_images`, `retriangulate`). Spending
a milestone re-implementing them would not move the ETH3D number, which the
repo's own ablations (density, intrinsics refinement) already show are *not*
the lever. The two real open items are the **two-view verification model
richness** (M1) and the **persistent graph structure** (M2) that COLMAP's view
graph provides and visloc-rs's ad hoc union-find does not — directly addressing
"a frontend detection-coverage and view-graph problem, not a mapper-schedule
problem." Vocab-tree (M4) is real but doesn't move the *current* ETH3D scenes
(too few images); it's the right lever for a future thousands-of-images
benchmark, so it's sequenced after the verification fix, not before.

---

## 4. Risks

- **SIFT quality parity on CPU.** COLMAP's SIFT (VLFeat covariant + DSP
  pooling) is heavily tuned and, per the repo's own ETH3D sweep, **SuperPoint
  keypoint density is a real but saturating lever** (2048→4096 keypoints: 3.5×
  ATE improvement on `courtyard`, no change on `office`, worse on `terrace`).
  Two paths: (a) keep the learned-feature bet and tune density/thresholds
  per-scene (cheap, already measured to help); (b) port a classical SIFT
  (VLFeat's BSD-licensed `sift.c`/`covdet.c` as the legally-clean reference,
  **not** SiftGPU) as an alternative frontend for wide-baseline sparse sets
  where learned features may under-detect. (b) is high effort for an uncertain
  marginal gain beyond (a) and is **not** included as a milestone above; revisit
  only if M1–M3 fail to close the `courtyard`/`office` gap.
- **Vocab-tree assets.** COLMAP ships a pre-trained vocabulary tree binary for
  its vocab-tree matcher; a from-scratch Rust port (M4) needs either a
  from-scratch trained tree (needs a large, diverse training image/descriptor
  corpus not currently in the repo) or a converter for COLMAP's existing tree
  format. Low priority given M4 is already sequenced last.
- **Scale of the port.** COLMAP's C++ source (excluding thirdparty) is on the
  order of ~150 files across `controllers/estimators/feature/scene/sfm/optim/retrieval`;
  this plan intentionally ports **3 of those areas** (two-view verification,
  correspondence graph, retrieval) rather than the whole tree, because the
  other areas (mapper core, PnP, BA, filtering) are verified already-equivalent
  or already-ported in visloc-rs.
- **Benchmark reproducibility.** COLMAP is **not installed locally**
  (confirmed: no `colmap` binary on `PATH`); the prior ETH3D battle
  (`benchmarks/registry/runs/eth3d/*.json`, dated 2026-07-06) is archived
  evidence only — the ETH3D `dslr_undistorted` scene data itself is not
  present under `E:\visloc_archive` today (only unrelated SLAM/tracking run
  directories are there). M5 requires re-provisioning both COLMAP 4.1.0(+
  CUDA, per the existing benchmark's machine) and the three ETH3D scenes
  (`courtyard`, `terrace`, `office`, from eth3d.net) before any rematch can run.
  This is a hard prerequisite for M1/M3's acceptance criteria, not just M5's.
- **GLOMAP / global-SfM scope creep.** §1.8 shows GLOMAP is now literally part
  of COLMAP's `main` branch, and it is architecturally the COLMAP component
  closest to visloc-rs's own "avoid repeated global BA" thesis. It is
  deliberately **not** one of M1–M5: it targets unordered photo collections at
  a scale/structure visloc-rs's pose-graph global-SfM analogue doesn't yet
  generalize to, and pursuing it now would be exactly the "blindly reimplement
  everything" failure mode the task brief warned against. Flag as a candidate
  **M6 (future)** only after M1/M2 land and only if M4's thousands-of-image
  benchmark shows the incremental mapper (even hardened) becoming the new
  bottleneck.
- **DPVO/DROID port in progress.** `pipelines/slam/src/dpvo_*.rs`,
  `crates/vision/src/dpvo/`, `pipelines/slam/src/map_atlas.rs`, and
  `pipelines/slam/src/sparse_factor_graph.rs` are under active concurrent
  development for the DPVO/DROID port and were read-only inspected here for
  context, not modified. Any M1–M5 implementation work should coordinate on
  `pipelines/slam/src/lib.rs` module-list conflicts before landing.

---

## 5. Sources

- [colmap/colmap](https://github.com/colmap/colmap) — `main` branch source tree
  (`src/colmap/{controllers,estimators,feature,scene,sfm,optim,retrieval,exe}`),
  `COPYING.txt` (BSD-3-Clause), `src/thirdparty/{LSD,SiftGPU,VLFeat,PoissonRecon,Symforce-Caspar}/LICENSE`.
- [colmap/glomap](https://github.com/colmap/glomap) — now `[DEPRECATED]`,
  BSD-3-Clause (`LICENSE`, ETH Zurich 2024), merged into `colmap/colmap` as the
  `GLOBAL` mapper.
- [Releases · colmap/colmap](https://github.com/colmap/colmap/releases) —
  current release `4.1.0`, matching the version already used in this repo's own
  benchmark (`docs/sfm_vs_colmap_benchmark.md`, "COLMAP 4.1.0 (fa8e3b3)").
- [What is the difference between incremental and global SfM in COLMAP?](https://colmap.org/what-is-the-difference-between-incremental-and-global-sfm-in-colmap/)
- [GLOMAP — Visual-SLAM Roadmap](https://www.cv-learn.com/visual-slam-roadmap/level-03-monocular-slam/glomap/)
- In-repo: `docs/sfm_vs_colmap_benchmark.md`, `docs/euroc_sfm_benchmark.md`,
  `docs/unordered_sfm_benchmark.md`, `docs/colmap_compatibility.md`,
  `pipelines/slam/src/incremental_sfm.rs`, `pipelines/slam/src/bundle.rs`,
  `pipelines/slam/src/covisibility_ba.rs`, `pipelines/slam/src/gnc.rs`,
  `crates/vision/src/two_view/mod.rs`, `crates/vision/src/pnp/{mod.rs,p3p.rs}`,
  `crates/vision/src/ransac/mod.rs`, `crates/vision/src/place_recognition/mod.rs`,
  `crates/vision/src/matching/mod.rs`, `crates/io/src/colmap/mod.rs`,
  `scripts/run_colmap_sfm_benchmark.sh`, `benchmarks/registry/runs/eth3d/*.json`.
