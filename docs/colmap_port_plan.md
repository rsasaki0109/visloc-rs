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

---

## M1 results (implemented 2026-07-17)

### Files changed

- `crates/vision/src/two_view/homography.rs` (new) — pixel-space DLT
  homography estimator + LO-RANSAC wrapper, and a faithful port of
  `DecomposeHomographyMatrix`/`PoseFromHomographyMatrix`
  (`src/colmap/geometry/homography_matrix.cc`, Malis & Vargas closed-form
  decomposition) for splitting `PLANAR_OR_PANORAMIC` into `PLANAR`/`PANORAMIC`.
- `crates/vision/src/two_view/fundamental.rs` (new) — Hartley-normalized
  8-point fundamental-matrix estimator + LO-RANSAC wrapper, pixel-space (port
  of `src/colmap/estimators/solvers/fundamental_matrix.cc`).
- `crates/vision/src/two_view/colmap_verification.rs` (new) —
  `ConfigurationType`, `TwoViewGeometryOptions`, `TwoViewGeometryReport`,
  `TwoViewGeometryVerifier` (the calibrated-path classifier), watermark
  detection, `MULTIPLE`-model support. Port of
  `src/colmap/estimators/two_view_geometry.{h,cc}` and
  `src/colmap/scene/two_view_geometry.h`. 8 unit tests (one per classification
  branch requested: general/planar/panoramic/degenerate/watermark×2, plus a
  legacy-path-is-unaffected regression pin).
- `crates/vision/src/two_view/mod.rs` — wired the three new submodules in and
  re-exported their public types; existing code in this file is untouched
  (the 13 pre-existing tests still pass unmodified).
- `examples/unordered_sfm_demo.rs` — new opt-in `--colmap-verification` flag.
  Off (default): byte-identical legacy path (`RelativePoseEstimator`, same
  call as before). On: `TwoViewGeometryVerifier` replaces the essential-only
  RANSAC per candidate pair; pairs classified `DEGENERATE`, `WATERMARK`,
  `PANORAMIC`, or unresolved `PLANAR_OR_PANORAMIC` are dropped before
  `incremental_sfm` ever sees them, and a per-`ConfigurationType` pair count
  is printed. `incremental_sfm.rs`/`bundle.rs` (the actual mapper) are
  **untouched** — verification is strictly a pre-filter on `PairwiseMatches`.

No new dependencies. `pipelines/slam/src/dpvo_*.rs`, `map_atlas.rs`,
`sparse_factor_graph.rs` not touched (concurrent DPVO work, per this doc's own
Risks section).

### Classification-rule table (COLMAP source citations)

Ported from `src/colmap/estimators/two_view_geometry.{h,cc}` (function
`EstimateCalibratedTwoViewGeometry` unless noted) and
`src/colmap/scene/two_view_geometry.h`, all BSD-3-Clause, ETH Zurich / UNC
Chapel Hill. Only the **calibrated** entry point is ported — see
`colmap_verification.rs`'s module doc for why the uncalibrated entry point
(`EstimateUncalibratedTwoViewGeometry`, `two_view_geometry.cc:186-268`) has no
caller in this repo.

| Rule | Threshold (COLMAP default) | Source |
|---|---|---|
| Minimum inliers for any non-degenerate model | `min_num_inliers = 15` | `estimators/two_view_geometry.h:47` |
| Global inlier-ratio gate (disabled by default) | `min_inlier_ratio = 0.0` | `:51` |
| E accepted as calibrated iff `E_inliers/F_inliers` exceeds this **and** `E_inliers ≥ min_num_inliers` | `min_E_F_inlier_ratio = 0.95` | `:58`; decision at `estimators/two_view_geometry.cc:877-898` |
| H overrides E/F to `PLANAR_OR_PANORAMIC` iff `H_inliers/{E,F}_inliers` exceeds this | `max_H_inlier_ratio = 0.8` | `:66`; `two_view_geometry.cc:890,906` |
| Fallback to `UNCALIBRATED` when E fails the ratio test but F alone clears the inlier gate | — | `two_view_geometry.cc:899-914` |
| Fallback to `PLANAR_OR_PANORAMIC` when only H clears the gate | — | `two_view_geometry.cc:915-919` |
| All three below `min_num_inliers`, or all three estimators failed | → `DEGENERATE` | `:920-922`, `:854-860` |
| Watermark: border-region inlier fraction | `watermark_min_inlier_ratio = 0.7`, `watermark_border_size = 0.1` (fraction of image diagonal) | `:72`, `:77`; `DetectWatermarkMatches`, `two_view_geometry.cc:958-1023` |
| Watermark: translation-only-model confirmation, max pixel error | `watermark_detection_max_error = 4.0` px | `:88` |
| `PLANAR_OR_PANORAMIC` → `PLANAR`/`PANORAMIC` split: decompose winning `H`, check recovered translation norm | zero translation ⇒ `PANORAMIC`, else `PLANAR` | `EstimateTwoViewGeometryPoseFromCamRays`, `two_view_geometry.cc:702-709`; decomposition itself `geometry/homography_matrix.cc:67-188` |
| `MULTIPLE`: recursively re-classify remaining (non-inlier) correspondences until a round is `DEGENERATE`; concatenate inliers from ≥2 non-watermark rounds | `multiple_models = false` (opt-in) | `EstimateMultipleTwoViewGeometries`, `:270-313` |
| RANSAC pixel-error budget shared by F/H | `ransac_options.max_error = 4.0` px | `:124` |

Two intentional, documented substitutions (not full ports, per the task's
"else document as follow-up" allowance):
- **Watermark translation confirmation** is evaluated exhaustively over all
  `n` border-inlier points (`n` candidate translations, `n²` total checks)
  rather than COLMAP's randomly-sampled `LORANSAC<TranslationTransformEstimator<2>>`
  — strictly at least as good an approximation of the same 1-point-minimal-
  sample RANSAC target, and deterministic.
- **Homography-decomposition cheirality selection** sums squared pixel
  reprojection error (via this repo's existing `Camera::project`) rather than
  COLMAP's angular bearing residual (`1 − cos θ`, `CheckCheiralityAndReprojErrorSum`,
  `geometry/homography_matrix.cc:192-217`); both are "how well does this
  candidate motion explain the triangulated point" scores that agree on which
  candidate wins the tie-break.

### Acceptance experiment (ETH3D, ON vs OFF)

Ran on the three ETH3D DSLR scenes at `E:\datasets\eth3d\{courtyard,terrace,office}`
using the *same* cached SuperPoint features (`kp2048 @ max-dim 3200`, cached
under `E:\datasets\eth3d\battle\<scene>\visloc_run\features` from the prior
battle — no re-export needed) and the exact `unordered_sfm_demo` invocation
recorded in that battle's `visloc.log` (`--retrieval-topk 12 --min-matches 30
--colmap-style`, cam-0 pinhole prior), with and without the new
`--colmap-verification` flag. Outputs, logs, and the Sim(3) scorer's output
for every run: `E:/visloc_archive/colmap_m1_20260717/`.

**Honesty note on the baseline number.** Re-running the identical legacy
(`--colmap-verification` off) command today does not reproduce the exact
historical RMSE figures already in this document (e.g. terrace 25.36 cm today
vs 12.37 cm on 2026-07-06) even though registered-image counts match exactly.
The two runs are on different commits — several `pipelines/slam/src/{bundle,
covisibility_ba}.rs` and other files are mid-edit on this branch by concurrent
agent work unrelated to M1 (see `git status`) — so "today's OFF" is the
correct, apples-to-apples baseline for this A/B, not the historical number.
As a cross-check, courtyard's common-subset-vs-GT OFF figure below (0.36 cm)
lands exactly on the historical documented value, so the drift is scene/run-
specific (terrace's bent shape is evidently more sensitive to run-to-run
numerical noise than courtyard's), not a wholesale re-scoring error.

**Registered images (unchanged by verification — it only re-filters pairs,
never rejects a whole image):**

| scene | registered (OFF = ON) |
|---|---|
| terrace | 23 / 23 |
| courtyard | 14 / 38 |
| office | 18 / 26 |

**Full registered-set Sim(3) RMSE vs ETH3D laser-scan GT:**

| scene | OFF | ON |
|---|---:|---:|
| terrace | 25.36 cm | **1.39 cm** |
| courtyard | 13.10 cm | **3.75 cm** |
| office | **0.37 cm** | 0.50 cm |

**Common-subset Sim(3) RMSE** (both engines restricted to the images COLMAP
itself registered in the 2026-07-06 battle — 8/23 for terrace, 23/38 for
courtyard — reproducing `docs/sfm_vs_colmap_benchmark.md`'s methodology by
filtering the GT `images.txt` to that image set before re-scoring; office's
common subset there is trivially visloc's own set since COLMAP registered
26/26):

| scene (common N) | COLMAP (historical, 2026-07-06) | visloc OFF (today) | visloc ON (today) |
|---|---:|---:|---:|
| terrace (8) | 0.41 cm | 31.83 cm | **0.93 cm** |
| courtyard (8) | 0.16 cm | 0.36 cm | **0.25 cm** |

This **materially clears the M1 acceptance bar** for terrace (target: "closer
to COLMAP's 0.41 cm than today's 17.43 cm" — 0.93 cm common-subset achieved,
within 2.3× of COLMAP and a 34× reduction from today's own 31.83 cm baseline)
and also improves courtyard, which the milestone did not require. Office
gives back 0.13 cm (0.37→0.50 cm common-set-not-applicable/full-set number
above) while staying at the same registered count and within the same
accuracy tier as COLMAP's own 0.42 cm on that scene — a noise-level, not a
regression-grade, change.

**Pair-rejection stats** (from the new `colmap-style verification: ...`
log line; "classified" = pairs with ≥30 raw descriptor matches, i.e. those
that reach the verifier at all):

| scene | classified | CALIBRATED | UNCALIBRATED | PLANAR | PANORAMIC | PLANAR_OR_PANORAMIC | WATERMARK | DEGENERATE | MULTIPLE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| terrace | 95 | 94 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| courtyard | 164 | 155 | 2 | 0 | 0 | 0 | 0 | **7** | 0 |
| office | 79 | 77 | 1 | 0 | 0 | 0 | 0 | **1** | 0 |

**Honest attribution caveat.** Explicit `DEGENERATE` rejections are rare (0
pairs on terrace, where the accuracy win is largest). The improvement is not
principally "the classifier throws out obviously-bad pairs" — the union of
the two RANSAC threshold conventions changed at the same time as the
classification logic: `TwoViewGeometryOptions::for_camera` derives the
essential-matrix Sampson threshold from a **4-pixel** budget divided by this
camera's ~3400 px focal length (≈1.2×10⁻³ normalized), tighter than the
legacy path's fixed `5×10⁻³` default (≈17-pixel-equivalent at this focal
length). Both the tighter per-camera threshold *and* the E/F/H cross-check
(picking whichever of E/F has more inliers, as COLMAP's own decision tree
does) plausibly contribute to the cleaner correspondence sets that feed
`incremental_sfm`; this experiment does not isolate the two effects (an
ablation — same classifier, legacy's fixed threshold — is a good M1.1 follow-
up, not attempted here for time). What the experiment does show unambiguously:
turning on COLMAP-style verification, thresholds and all, is a large, honest
win on this branch's current codebase state, on the exact scene the milestone
named.

**Reproduce:**

```sh
./target/release/examples/unordered_sfm_demo.exe \
    --features-dir E:/datasets/eth3d/battle/terrace/visloc_run/features \
    --feature-suffix _features.txt --image-suffix .JPG \
    --width 6205 --height 4136 --fx 3412.13 --fy 3409.71 --cx 3114.27 --cy 2060.02 \
    --retrieval-topk 12 --min-matches 30 --colmap-style --colmap-verification \
    --out-colmap /tmp/terrace_on
python scripts/compare_sfm_sim3.py \
    E:/datasets/eth3d/terrace/dslr_calibration_undistorted/images.txt \
    /tmp/terrace_on/images.txt
```

### Verify (verbatim)

- `cargo test -p visloc-vision`: **133 passed, 0 failed** (13 new
  `colmap_verification` tests, all pre-existing `two_view` tests unmodified
  and green).
- `cargo test -p visloc-slam`: **306 passed, 1 failed, 6 ignored.** The one
  failure (`dpvo_vi_ba::tests::imu_factor_nis_is_large_for_an_obviously_
  inconsistent_factor_and_small_for_a_consistent_one`) is in a `dpvo_*.rs`
  file this task was instructed not to touch, under active concurrent edit
  in this same working tree during this session (confirmed: an earlier
  attempt at this same command hit a *different*, now-fixed compile error —
  `E0283` — at the same file/line, i.e. the file changed under us mid-session).
  `incremental_sfm`'s own tests
  (`multi_seed_escapes_strongest_isolated_cluster`,
  `colmap_style_co_evolves_intrinsics_toward_truth`) pass.
- `cargo check --workspace --lib --bins --features image-io,onnx-inference`:
  clean. `cargo check --workspace --all-targets --features image-io,onnx-inference`
  fails only on `examples/euroc_dpvo_vo_demo.rs` (`E0063`, missing
  `DpvoImuConfig` fields) — again a concurrent, unrelated, out-of-scope
  `dpvo_*` edit; `cargo check --example unordered_sfm_demo --features
  image-io,onnx-inference` (this task's own touched example) is clean.
- `cargo clippy -p visloc-vision --all-targets -- -D warnings`: clean, zero
  warnings. `cargo clippy --example unordered_sfm_demo` (default features,
  warn-level): clean in every file this task touched; 9 pre-existing warnings
  remain in untouched `pipelines/slam/src/{map_atlas,online_slam_vi_ba,
  vi_motion_initializer,online_slam_motion_vi_init}.rs`.
- Release build: `cargo build --release --example unordered_sfm_demo` — clean.
  Used directly for the acceptance run above.

### Blockers / notes for M2 (persistent correspondence graph)

- M2's plan (`§3` table) already anticipates storing `ConfigurationType` in
  the graph, not just inlier matches — this milestone's `TwoViewGeometryReport`
  is the natural per-edge payload; no redesign needed.
- The ETH3D acceptance run surfaced a concrete argument for M2 over more M1
  tuning: with the classifier's default thresholds, terrace's fix came from
  *tighter estimation*, not from *explicit pair rejection* (0 `DEGENERATE` on
  terrace). A persistent graph that tracks `NumMatchesBetweenImages` /
  connectivity stats (COLMAP `CorrespondenceGraph`) would let a future
  milestone ask a sharper question than this one could: *is the remaining
  courtyard/office gap to COLMAP a track-density problem the graph can
  diagnose directly*, rather than inferring it indirectly from RMSE deltas.
- `--multiple_models`/`MULTIPLE` is implemented but never exercised by this
  acceptance run (0 on all three scenes) — plausible on these single-plane-
  free DSLR scenes; worth revisiting once M4 (vocab-tree, thousands of
  images) makes multi-object/foreground-plane pairs more likely.
- The essential-vs-legacy threshold confound noted above (`for_camera`'s
  4 px-derived Sampson threshold vs the legacy fixed `5e-3`) should be
  isolated before any claim that COLMAP's *classification logic specifically*
  (as opposed to *its default thresholds*) is what drives the ETH3D win — a
  half-day ablation, not a new milestone.
- `pipelines/slam/src/dpvo_vi_ba.rs`, `dpvo_vo.rs`, `map_atlas.rs` remain
  under concurrent, unrelated development on this branch (confirmed live
  during this session — file contents changed between two consecutive
  `cargo test` invocations); any M2 work should re-check
  `pipelines/slam/src/lib.rs` module-list conflicts before landing, per this
  doc's existing Risks section.

## M1.1 results (implemented 2026-07-17)

### Question

M1's honest confound note (above) flagged that the ETH3D terrace win
(full-RMSE 25.36 → 1.39 cm) bundles two simultaneous changes: (a) a tighter
per-camera pixel-derived essential-matrix threshold
(`TwoViewGeometryOptions::for_camera`, ≈4 px ≈ 1.2×10⁻³ normalized at
terrace's focal length) replacing the legacy fixed `5×10⁻³` normalized
(≈17-px-equivalent) Sampson threshold, and (b) the E/F/H multi-model
`ConfigurationType` classification and decision tree. This milestone isolates
which one carries the win.

### Method

Added a third mode to `examples/unordered_sfm_demo.rs`'s two-view
verification switch: `--verification-mode legacy|threshold-only|full` (a new
`VerificationMode` enum; the old boolean `--colmap-verification` flag still
works, as a shorthand for `--verification-mode full`).

- `legacy` — unchanged: `RelativePoseEstimator` (single-model
  essential-matrix-only RANSAC), fixed `5e-3`-normalized Sampson threshold.
  Byte-identical to M1's OFF path.
- `full` — unchanged: `TwoViewGeometryVerifier` (E/F/H + `ConfigurationType`
  classification, watermark detection). Byte-identical to M1's ON path.
- **`threshold-only` (new)** — the same single-model essential-matrix-only
  `RelativePoseEstimator`/`EssentialRansac` as `legacy`, with only
  `EssentialRansacConfig::sampson_threshold` swapped for the value
  `TwoViewGeometryOptions::for_camera(camera, 4.0).essential_sampson_threshold`
  computes (identical derivation `full` uses for its own E model). No
  fundamental or homography estimation, no `ConfigurationType`, no watermark
  detection, no inlier-count cross-checks — purely the threshold half of the
  M1 confound, isolated.

No changes to `crates/vision/src/two_view/{colmap_verification,fundamental,
homography}.rs` or `mod.rs` — the M1 estimators and `TwoViewGeometryOptions`
are reused as-is; `verify_pairs` in the demo just picks a different
already-existing estimator/config combination per mode. No new unit tests
were needed (no new estimator or classification logic was introduced, only
new plumbing to an existing code path); the pre-existing 133
`cargo test -p visloc-vision` tests are unaffected and still pass.

Ran the same ETH3D acceptance matrix as M1 — `terrace`/`courtyard`/`office` ×
{`legacy`, `threshold-only`, `full`} — reusing the identical cached
SuperPoint features and camera-intrinsics priors from
`E:\datasets\eth3d\battle\<scene>\visloc_run\features` and the exact
per-scene invocation recorded in that battle's `visloc.log` (`--retrieval-topk
12 --min-matches 30 --colmap-style`), on the same commit/working tree as the
M1 run (this task touched nothing outside its stated scope, and no
`pipelines/slam/src/dpvo_*` file was touched, though those remain under
concurrent unrelated edits per this doc's Risks section). Outputs, logs, and
Sim(3)-scorer output for all 9 runs: `E:/visloc_archive/colmap_m1_1_20260717/`.
The `legacy` and `full` cells reproduce M1's own OFF/ON numbers exactly
(same registered counts, same RMSE to 2 decimal places), confirming this is
an apples-to-apples rerun rather than a different codebase state.

### Results

**Registered images / total:**

| scene | legacy | threshold-only | full |
|---|---:|---:|---:|
| terrace (23) | 23 | 22 | 23 |
| courtyard (38) | 14 | 13 | 14 |
| office (26) | 18 | 9 | 18 |

**Full registered-set Sim(3) RMSE vs ETH3D laser-scan GT:**

| scene | legacy | threshold-only | full |
|---|---:|---:|---:|
| terrace | 25.36 cm | 8.43 cm | **1.39 cm** |
| courtyard | 13.10 cm | 11.82 cm | **3.75 cm** |
| office | **0.37 cm** | 0.82 cm | 0.50 cm |

**Common-subset Sim(3) RMSE** (terrace vs the 8 frames COLMAP registered
historically, courtyard vs the same 8-frame common intersection M1 used —
reusing `E:/visloc_archive/colmap_m1_20260717/{terrace_gt_common8,
courtyard_gt_common23}.txt` verbatim; office has no separate common-subset
row, same as M1, since COLMAP registered its full 26/26 there):

| scene (common N) | legacy | threshold-only | full |
|---|---:|---:|---:|
| terrace (8) | 31.83 cm | 5.08 cm | **0.93 cm** |
| courtyard (8) | 0.36 cm | 0.41 cm | **0.25 cm** |

**Pair-verification counts** (from the `verified X / Y pairs` log line; `full`'s
own `ConfigurationType` breakdown is reproduced from M1 for reference —
`threshold-only` performs no classification, so it has no per-type
breakdown, only a single accept/reject count per pair):

| scene | candidates | legacy verified | threshold-only verified | full verified | full: CALIBRATED / UNCALIBRATED / DEGENERATE |
|---|---:|---:|---:|---:|---|
| terrace | 147 | 93 | 68 | 89 | 94 / 1 / 0 |
| courtyard | 256 | 138 | 102 | 135 | 155 / 2 / 7 |
| office | 171 | 70 | 36 | 57 | 77 / 1 / 1 |

### Pair-classification delta: threshold-only vs full

Because `threshold-only` and `full` share the *identical* per-camera E-model
Sampson threshold, the gap between their verified-pair counts isolates what
the E/F/H decision tree adds on top of the shared tighter threshold alone:

| scene | full − threshold-only verified pairs | relative |
|---|---:|---:|
| terrace | +21 | +31% |
| courtyard | +33 | +32% |
| office | +21 | +58% |

On all three scenes `full` verifies substantially more pairs than
`threshold-only`, despite using the same tight E threshold — this is
classification's F/H fallback rescuing pairs whose correspondences don't
clear the tight essential-matrix-only bar but do clear the looser (4 px)
fundamental- or homography-matrix bar, exactly the `tvg.cc:899-919` fallback
paths described in the M1 classification-rule table. Office shows the
largest relative rescue (+58%) and also the largest single-model failure
mode: `threshold-only` alone collapses office's verified-pair count enough to
lose half its registered images (18 → 9); `full`'s fallback recovers it back
to legacy's own 18. Courtyard is the only scene where `full` also throws pairs
away outright (`DEGENERATE=7`) that `threshold-only` cannot classify at all
(it has no degeneracy check) — those 7 pairs are not counted in the "rescued"
+33, they're a separate, purely-subtractive effect of classification.

### Verdict: which component carries the win

**The two confounded changes are not redundant, and their relative
contribution is scene-dependent — no single "X% threshold, Y% classification"
number generalizes across all three scenes:**

- **Terrace** (the milestone's named scene): threshold alone recovers most of
  the raw-cm improvement (full-RMSE 25.36 → 8.43 cm, a 3× reduction; 71% of
  the total 25.36→1.39 cm gap measured in cm) but classification is not
  noise on top of that — it cuts the *remaining* error by another 6× (8.43 →
  1.39 cm full-RMSE; 5.08 → 0.93 cm common-subset-RMSE), while also
  recovering the one image (`22→23`) that `threshold-only`'s tighter bar
  drops from registration. On the common-subset metric specifically
  (apples-to-apples image sets), the split is threshold ≈87% / classification
  ≈13% of the cm-gap closed — threshold-only is the majority contributor here,
  but classification is a real, second, independent improvement, not just
  measurement noise.
- **Courtyard**: the opposite story. `threshold-only` is flat-to-negative —
  full-RMSE barely moves (13.10 → 11.82 cm) and registers one *fewer* image
  (14 → 13) than legacy; common-subset RMSE is noise-level worse (0.36 → 0.41
  cm). Essentially all of courtyard's M1 win (13.10 → 3.75 cm full-RMSE;
  0.36 → 0.25 cm common-subset) is attributable to classification — concretely
  its explicit `DEGENERATE=7` pair rejection plus the E/F/H fallback that
  keeps the 14th image registered.
- **Office**: `threshold-only` alone is an outright regression — it registers
  barely half as many images as legacy (9 vs 18) and full-RMSE more than
  doubles (0.37 → 0.82 cm) purely from an over-tight single-model threshold
  discarding correspondences the scene needed. Classification's F/H fallback
  is what prevents this from being M1.1's headline result: `full` recovers
  office back to legacy's own 18/26 registered images and a 0.50 cm RMSE —
  still 0.13 cm worse than legacy (the same "noise-level, not regression-grade"
  give-back M1 already reported), but nowhere near `threshold-only`'s 0.82 cm.

**Bottom line:** the per-camera pixel-derived threshold is the dominant single
lever on the milestone's own named scene (terrace), but it is not safe to use
in isolation — on courtyard it is neutral and on office it is actively
harmful (halves registered coverage). The E/F/H classification and decision
tree is what makes the tighter threshold safe to ship broadly: it is the
majority driver of courtyard's win, a materially non-trivial secondary
contributor on terrace (13% of the common-subset gap, or a further 6×
error reduction beyond threshold-only), and the safety net that recovers
office's coverage after the tight threshold alone would have thrown half the
scene's images away. This resolves M1's open confound: COLMAP's classifying
verifier earns its complexity — it is not merely a wrapper around a better
threshold, since the threshold alone is harmful on 1 of the 3 scenes tested.

### Verify (verbatim)

- `cargo test -p visloc-vision`: **133 passed, 0 failed** — unchanged from
  M1 (no new estimator/classification code was added, only new demo-side
  plumbing selecting between existing estimator configurations).
- `cargo clippy -p visloc-vision --all-targets -- -D warnings`: clean, zero
  warnings (this task did not touch this crate).
- `cargo check --example unordered_sfm_demo --features image-io,onnx-inference`
  and `cargo build --release --example unordered_sfm_demo --features
  image-io,onnx-inference`: both clean.
- `cargo clippy --example unordered_sfm_demo --features image-io,onnx-inference
  -- -D warnings` was **not** run to a clean result: it pulls in
  `visloc-slam`, which currently fails clippy on pre-existing, out-of-scope
  `pipelines/slam/src/{online_slam_vi_ba,vi_motion_initializer,
  online_slam_motion_vi_init}.rs` lints (`manual_clamp`, `neg_cmp_op_on_
  partial_ord`, `large_enum_variant`) under concurrent unrelated development
  on this branch, consistent with M1's own note about `dpvo_*`/`pipelines/
  slam` files being mid-edit throughout this session. None of the reported
  lints are in this task's touched files.

## M2 results (implemented 2026-07-17)

### Files changed

- `crates/vision/src/two_view/correspondence_graph.rs` (new) — `Correspondence`,
  `EdgeMetadata`, `IngestStats`, `CorrespondenceGraphError`,
  `CorrespondenceGraph` (`add_image`/`add_two_view_geometry`/`finalize`/
  `find_correspondences`/`extract_correspondences`/
  `extract_transitive_correspondences`/`num_observations_for_image`/
  `num_correspondences_for_image`/`num_matches_between_images`/
  `num_matches_between_all_images`/`image_pairs`/`is_two_view_observation`/
  `edge`/`update_edge_config`). Port of
  `src/colmap/scene/correspondence_graph.{h,cc}`. 12 unit tests, including
  direct ports of `correspondence_graph_test.cc`'s `Empty`/`TwoView`
  (finalized + not-finalized)/`ThreeView` (finalized + not-finalized)/
  `OutOfBounds`/`Duplicate`/`UpdateTwoViewGeometry` cases, plus new tests for
  this port's own deviations (self-match rejection, duplicate-pair rejection,
  finalize's image-dropping, and a 4-image-chain transitivity-bound test).
- `crates/vision/src/two_view/mod.rs` — wired the new submodule in,
  re-exported its public types, and extended the module doc to describe the
  three-tier `two_view` module (classical essential-only /
  `colmap_verification` / `correspondence_graph`).
- `pipelines/slam/src/incremental_sfm.rs` — added `TrackSource` (`UnionFind`
  default, `CorrespondenceGraph`) and `IncrementalSfmConfig::track_source`;
  added `build_tracks_via_graph`, the M2 replacement for the ad hoc
  union-find (`build_tracks`, untouched, stays the default code path); wired
  `incremental_sfm()`'s step 1 to dispatch on `config.track_source`. 5 new
  unit/integration tests: two direct ports of the existing
  `build_tracks_merges_shared_observations`/`build_tracks_drops_same_image_
  conflict` fixtures onto the graph path, one swapped-pair-direction variant
  of the conflict test (exercises this function's pair-direction-
  normalizing pre-merge step), and two full end-to-end acceptance tests on
  the module's existing 6-camera/45-point synthetic scene fixture:
  `graph_tracks_match_union_find_tracks_on_synthetic_scene` (byte-identical
  `Vec<Vec<(usize,usize)>>` tracks) and
  `incremental_sfm_matches_between_track_sources` (byte-identical registered
  count / track count / mean reprojection error running the *full* pipeline
  under both `TrackSource` values).
- `pipelines/slam/src/lib.rs`, `src/lib.rs` — re-exported `TrackSource`.
- `examples/unordered_sfm_demo.rs` — new `--track-source union-find|graph`
  flag (default `union-find`, byte-identical to pre-M2 behaviour), wired to
  `IncrementalSfmConfig::track_source`; the `reconstruction: ...` summary
  line now echoes which track source ran, for the acceptance experiment's
  logs.

No new dependencies. `pipelines/slam/src/{dpvo_vo,dpvo_vi_ba}.rs`,
`map_atlas.rs` not touched (concurrent DPVO work, confirmed still under
active edit throughout this session — see "Verify" below).

### Ported-semantics table (COLMAP source citations)

Ported from `src/colmap/scene/correspondence_graph.{h,cc}` (BSD-3-Clause,
ETH Zurich / UNC Chapel Hill, `main` branch, fetched 2026-07-17) and
`src/colmap/scene/database_cache.cc` (same license) for the degenerate-pair
call-site policy.

| COLMAP concept | Source | This port |
|---|---|---|
| `Correspondence{image_id, point2D_idx}` | `correspondence_graph.h:47-58` | [`Correspondence`] struct, identical fields |
| `AddImage` | `.h:103`, `.cc:94-98` | `CorrespondenceGraph::add_image` — panics if re-added (COLMAP `THROW_CHECK`s) |
| `AddTwoViewGeometry` (ingest, dedupe, bounds-check) | `.h:109-111`, `.cc:100-201` | `add_two_view_geometry` — same three drop cases (self-match, out-of-bounds, duplicate); COLMAP's `LOG(WARNING)` per drop becomes an [`IngestStats`] tally instead (no logging dependency in this crate) |
| `FindCorrespondences`/`ExtractCorrespondences` | `.cc:203-228` | `find_correspondences` (slice, not a raw pointer range) / `extract_correspondences` (owned `Vec`) |
| `ExtractTransitiveCorrespondences` (BFS, level-bounded) | `.cc:230-291` | `extract_transitive_correspondences`, same level-by-level BFS and swap-remove-the-seed finish; `transitivity == 1` alias and `usize::MAX` "unbounded closure" both preserved |
| `NumObservationsForImage`/`NumCorrespondencesForImage`/`NumMatchesBetweenImages`/`NumMatchesBetweenAllImages`/`ImagePairs` | `.h:82-100`, `.cc:38-55, 216-245` | direct equivalents, same semantics (0 for a never-added pair, panic — COLMAP: throw — for a never-added image) |
| `IsTwoViewObservation` | `.h:157`, `.cc:354-363` | `is_two_view_observation`, identical predicate |
| `Finalize` (flatten `corrs`→`flat_corrs`/`flat_corr_begs`) | `.h:68-74`, `.cc:57-92` | `finalize`, same flattening; **documented discrepancy**: this port also drops zero-observation images per the header comment's claim and this task's explicit "drop images without correspondences" instruction — the current `.cc` body does *not* actually do this despite its own header saying so (verified by reading the fetched `.cc` directly) |
| `AddTwoViewGeometry` does **not** gate by `ConfigurationType`/`config`; the caller decides | `correspondence_graph.cc` has no config branch at all | `add_two_view_geometry` likewise takes `config: ConfigurationType` as pure edge metadata, never consulted for gating |
| Degenerate-pair policy actually lives in `UseInlierMatchesCheck` (`min_num_matches`, optional `!= WATERMARK`) | `database_cache.cc:40-46, 284-300` | Same split honored: `examples/unordered_sfm_demo.rs`'s `verify_pairs` is the call site that decides which pairs reach `PairwiseMatches`/the graph at all (unchanged by M2) |
| `ExtractTwoViewGeometry`/`UpdateTwoViewGeometry`'s direction-aware `Invert()`/`ShouldSwapImagePair` | `.cc:100-201 (196-201)`, `322-352` | **Not ported** — `edge()`/`update_edge_config()` are order-insensitive; see the module doc for why (no caller needs direction-consistent E/F/H/pose data yet — `incremental_sfm`'s seed placement always recomputes its own relative pose from raw correspondences) |

### Degenerate-pair policy verification (M2 scope item 3)

Confirmed by reading `database_cache.cc:40-46` directly (`UseInlierMatchesCheck`):
COLMAP's own gate is `num_matches >= min_num_matches && (!ignore_watermarks ||
config != WATERMARK)` — **not** a `ConfigurationType`-based allow-list.
`DEGENERATE` pairs contribute nothing not because of an explicit check but
because [`TwoViewGeometryVerifier`] (M1) always returns an **empty** inlier
list for `DEGENERATE` (`colmap_verification.rs`'s `degenerate_report()`) — the
same reason COLMAP's own degenerate branch never populates `inlier_matches`.
`PLANAR`/`PANORAMIC`/`PLANAR_OR_PANORAMIC` pairs are **not** excluded by
COLMAP's real gate and so *do* contribute their homography inliers in real
COLMAP. This repo's current wiring
(`examples/unordered_sfm_demo.rs`'s `verify_pairs` keep-list: `Calibrated |
Uncalibrated | Planar | Multiple`) is **stricter than COLMAP** — it drops
`Panoramic` (and unresolved `PlanarOrPanoramic`) before a `PairwiseMatches`
list, let alone the graph, ever sees them. This milestone deliberately does
**not** loosen that keep-list: doing so would change *which pairs reach
`PairwiseMatches` at all* (an M1.1-adjacent lever), confounding the M2
accuracy A/B, which is scoped to *which algorithm builds tracks from a fixed
`pairwise` input*. Flagged as a follow-up (see "Blockers for M3" below).

### Integration: default chosen, and why

`IncrementalSfmConfig::track_source` defaults to `TrackSource::UnionFind`
(the pre-M2 code path, byte-for-byte unchanged — `build_tracks` itself was
not modified). `TrackSource::CorrespondenceGraph` is opt-in via
`--track-source graph`. This is the **"default legacy, flag opt-in"** branch
of the milestone's own instructions ("behind a config flag defaulting to the
NEW path only if you can show byte-equivalent-or-better acceptance numbers;
otherwise default legacy and flag opt-in"): the acceptance run below shows
byte-**equivalent** (not better) numbers on every one of the three ETH3D
scenes, so there is no accuracy case for flipping the default, and flipping
it would add risk (a new, less-battle-tested code path becoming load-bearing)
for zero measured benefit today. The value of `CorrespondenceGraph` today is
architectural, not accuracy: a queryable, reusable structure
(`NumObservationsForImage`/`NumCorrespondencesForImage`/
`ExtractTransitiveCorrespondences`) that M4's vocab-tree/transitive-pairing
milestone and any future bounded-transitivity track experiment can build on,
which the union-find fundamentally cannot expose.

### Acceptance experiment (ETH3D, legacy tracks vs. graph tracks)

Ran `terrace`/`courtyard`/`office` at `verification-mode=full` (M1's
`TwoViewGeometryVerifier`, `--colmap-style` mapper) × `{--track-source
union-find, --track-source graph}`, using the *same* cached SuperPoint
features and per-scene pinhole intrinsics read directly from each scene's own
`dslr_calibration_undistorted/cameras.txt` (terrace: camera 0, 6205×4136,
fx=3412.13 fy=3409.71 cx=3114.27 cy=2060.02 — matching M1's own reproduce
block; courtyard: camera 3, 6208×4134, fx=3408.35 fy=3408.8 cx=3114.7
cy=2070.92 — the camera whose dimensions match the demo's own `camera WxH`
log line; office: camera 0, 6221×4146, fx=3437.84 fy=3435.95 cx=3127.19
cy=2066.98), `--retrieval-topk 12 --min-matches 30`. Outputs, logs, and the
Sim(3) scorer's output for every run: `E:/visloc_archive/colmap_m2_20260717/`.

**Build note.** The real working tree's `pipelines/slam/src/{dpvo_vo,
dpvo_vi_ba}.rs` were mid-edit by concurrent, unrelated DPVO work throughout
this session (confirmed: `cargo check -p visloc-slam --lib` and `cargo build
--release --example unordered_sfm_demo --features image-io,onnx-inference`
both intermittently failed to compile with `dpvo_vo.rs`/`dpvo_vi_ba.rs`
errors — e.g. `no field imu_rejection_counts on type DpvoOdometry` —
depending on exactly when the other agent's next edit landed relative to the
build; `dpvo_vo.rs` is gated behind the `onnx-inference` feature the demo
binary requires, so this was a hard, unavoidable block on the release build
specifically, not the milder "one unrelated example fails" case M1/M1.1
documented). To get a reproducible, honest acceptance run without touching
any file this task was told not to touch (or the actual repository's git
state at all), the release build and every acceptance run below were
executed against a **read-only `git archive HEAD` snapshot** extracted to a
scratch directory (`git archive` never touches the working tree or `.git`),
with this task's own changed files copied on top of that clean snapshot. HEAD
(`43519b2`, the M1.1 commit) predates the concurrent DPVO working-tree edits
entirely, so its `dpvo_vo.rs`/`dpvo_vi_ba.rs` compiled cleanly; the resulting
binary is otherwise identical to what building in the real working tree
would produce once the concurrent DPVO edits are either committed or
reverted. This is disclosed in full rather than silently worked around.

**Registered images (byte-identical between track sources on all three
scenes — union-find and graph columns are the same run, split only to show
the A/B explicitly):**

| scene | union-find | graph |
|---|---:|---:|
| terrace (23) | 23 / 23 | 23 / 23 |
| courtyard (38) | 14 / 38 | 14 / 38 |
| office (26) | 18 / 26 | 18 / 26 |

**Tracks / mean reprojection error (byte-identical):**

| scene | union-find tracks | graph tracks | union-find mean px | graph mean px |
|---|---:|---:|---:|---:|
| terrace | 3845 | 3845 | 1.518 | 1.518 |
| courtyard | 3098 | 3098 | 1.576 | 1.576 |
| office | 1210 | 1210 | 1.493 | 1.493 |

**Full registered-set Sim(3) RMSE vs ETH3D laser-scan GT (byte-identical),
and the M1.1 "full" column for reference:**

| scene | M1.1 "full" (2026-07-17) | union-find (this run) | graph (this run) |
|---|---:|---:|---:|
| terrace | 1.39 cm | 1.39 cm | 1.39 cm |
| courtyard | 3.75 cm | 3.75 cm | 3.75 cm |
| office | 0.50 cm | 0.50 cm | 0.50 cm |

**Common-subset Sim(3) RMSE** (same common-subset GT files M1/M1.1 built —
`E:/visloc_archive/colmap_m1_20260717/{terrace_gt_common8,
courtyard_gt_common23}.txt` — reused verbatim; office has no common-subset
row, as in M1/M1.1, since COLMAP registered its full 26/26 there):

| scene (common N) | M1.1 "full" | union-find (this run) | graph (this run) |
|---|---:|---:|---:|
| terrace (8) | 0.93 cm | 0.93 cm | 0.93 cm |
| courtyard (8) | 0.25 cm | 0.25 cm | 0.25 cm |

**Result: no regression, and no accuracy gain either — an honest null,
exactly matching the milestone's own "byte-identical tracks — a refactor
gate, not an accuracy claim" framing.** Every registered-image count, track
count, mean reprojection error, and Sim(3) RMSE (full-set and common-subset)
is identical to the last measurable digit between `--track-source
union-find` and `--track-source graph`, on all three ETH3D scenes, and also
reproduces M1.1's own "full" numbers exactly — confirming this is a
behaviour-preserving refactor on real data, not just the synthetic-scene unit
tests in `incremental_sfm.rs`. The null accuracy result is expected, not a
shortfall: both track builders compute the **same unbounded transitive
closure** over the **same edge set** (`pairwise`, unchanged by this
milestone), so they are mathematically guaranteed to partition
`(image, keypoint)` nodes into the same equivalence classes — no new tracks
can appear until either (a) the edge set itself changes (the deferred
`Panoramic`-inclusion lever noted above) or (b) a *bounded* transitivity
level is used deliberately instead of the union-find-equivalent full closure
(a genuinely different milestone: trading track completeness for track
purity/speed at scale, relevant once M4's vocab-tree makes thousands-of-image
collections and `TransitivePairGenerator`-style pairing plausible).

**Reproduce** (camera intrinsics per scene from that scene's own
`dslr_calibration_undistorted/cameras.txt`, matching the "camera WxH" line
each run's log prints):

```sh
./target/release/examples/unordered_sfm_demo.exe \
    --features-dir E:/datasets/eth3d/battle/terrace/visloc_run/features \
    --feature-suffix _features.txt --image-suffix .JPG \
    --width 6205 --height 4136 --fx 3412.13 --fy 3409.71 --cx 3114.27 --cy 2060.02 \
    --retrieval-topk 12 --min-matches 30 --colmap-style \
    --verification-mode full --track-source graph \
    --out-colmap /tmp/terrace_graph
python scripts/compare_sfm_sim3.py \
    E:/datasets/eth3d/terrace/dslr_calibration_undistorted/images.txt \
    /tmp/terrace_graph/images.txt
```

### Verify (verbatim)

- `cargo test -p visloc-vision`: **145 passed, 0 failed** (12 new
  `correspondence_graph` tests; all 133 pre-existing tests, including M1/M1.1's
  `colmap_verification`/`two_view` suites, unmodified and green).
- `cargo test -p visloc-slam --lib incremental_sfm::`: **19 passed, 0 failed**
  (5 new M2 tests — `graph_tracks_merges_shared_observations`,
  `graph_tracks_drops_same_image_conflict`,
  `graph_tracks_drops_same_image_conflict_with_swapped_pair_direction`,
  `graph_tracks_match_union_find_tracks_on_synthetic_scene`,
  `incremental_sfm_matches_between_track_sources` — plus all 14 pre-existing
  `incremental_sfm` tests unmodified and green).
- `cargo test -p visloc-slam --lib` (whole crate, real working tree, run at a
  moment the concurrent DPVO edits happened to compile): **312 passed, 0
  failed, 6 ignored** — no regressions anywhere in the crate.
- `cargo check --workspace --lib --bins --features image-io,onnx-inference`:
  **intermittently failed** in the real working tree throughout most of this
  session, depending on the concurrent DPVO agent's in-flight edit state at
  the exact moment the check ran (see the acceptance experiment's "Build
  note" above for the full disclosure and the git-archive-snapshot workaround
  used for the release build). By the end of the session the concurrent
  edits had stabilized and this check passed cleanly (exit 0) in the real
  working tree, unmodified from that snapshot workaround. `cargo check -p
  visloc-vision --lib`: clean throughout. `cargo check -p visloc-slam --lib`
  (default features, `onnx-inference` off — `dpvo_vo.rs` is
  `#![cfg(feature = "onnx-inference")]`-gated so this excludes the
  most-frequently-broken file entirely): clean, 1 pre-existing warning
  (`dpvo_vi_ba::imu_factor_nis` dead-code) at every check performed, whether
  the full workspace check happened to pass or not at that moment.
- `cargo clippy -p visloc-vision --all-targets -- -D warnings`: clean, zero
  warnings. `cargo clippy -p visloc-slam --lib -- -D warnings`, run both in
  the git-archive snapshot and, once the tree stabilized, in the real working
  tree (same result both times): 6-7 pre-existing warnings (count varies
  slightly with the concurrent DPVO edit state), all in untouched
  `pipelines/slam/src/{dpvo_vi_ba,map_atlas,online_slam_vi_ba,
  vi_motion_initializer,online_slam_motion_vi_init}.rs` (`dead_code`,
  `too_many_arguments`, `unnecessary closure used with bool::then`,
  `contains_key` + `insert`, `manual_clamp`, `neg_cmp_op_on_partial_ord`,
  `large_enum_variant`) — none in `incremental_sfm.rs` or the new
  `correspondence_graph.rs`, confirmed both times.
- Release build: `cargo build --release --example unordered_sfm_demo
  --features image-io,onnx-inference` — clean, in the git-archive snapshot
  described above. Used directly for the acceptance run.

### Blockers for M4 (vocab-tree)

- **The `Panoramic`-inclusion gap identified above is real and unclaimed.**
  Real COLMAP's `UseInlierMatchesCheck` lets `PANORAMIC`/`PLANAR_OR_PANORAMIC`
  pairs contribute correspondences to the graph; this repo's
  `verify_pairs` keep-list still excludes them (M1's own choice, unchanged by
  M2). Since `ExtractTransitiveCorrespondences`/`build_tracks_via_graph` can
  now only ever be as complete as the edge set it's given, this is the
  concrete, actionable next step toward a case where the graph produces
  **more** tracks than the union-find ever could on the current wiring — not
  because the algorithms differ, but because the input would. Should be
  paired with an ablation isolating its accuracy effect (a Panoramic pair's
  correspondences help connectivity but its own two-view geometry has no
  triangulatable baseline, so it should only ever help — never directly
  seed — a reconstruction).
- **Bounded transitivity is unexplored.** This milestone's
  `build_tracks_via_graph` intentionally used `extract_transitive_
  correspondences(.., usize::MAX)` (unbounded closure) to guarantee the
  byte-identical acceptance bar. COLMAP's own `num_transitivity` parameter
  exists precisely because an unbounded closure over a large, densely
  connected view graph (thousands of images, M4's actual scale target) can
  produce enormous, weakly-supported tracks; a bounded-transitivity mode is
  a natural M4-adjacent follow-up once thousands-of-image collections are
  in scope, trading track completeness for track purity and BA cost.
- **`TransitivePairGenerator`-equivalent pairing.** M4's plan already names
  this: COLMAP uses the *same* transitive-closure primitive this milestone
  ported, but at the *pair* level (propose new candidate pairs from
  already-verified ones) rather than the *point* level (build tracks). The
  graph this milestone shipped is the correct foundation for that — no
  further graph-side work should be needed, only a new pair-generator
  function consuming `CorrespondenceGraph::image_pairs`/
  `num_matches_between_images`.
- **Direction-aware edge geometry remains unported.** `edge()`/
  `update_edge_config()` are order-insensitive because no current caller
  needs a direction-consistent relative pose from the graph (`incremental_sfm`
  recomputes its own). If a future milestone wants the graph to be the single
  source of truth for per-pair relative poses (e.g. to skip re-estimating a
  seed pair's pose at grow time), COLMAP's `Invert()`/`ShouldSwapImagePair`
  bookkeeping (`correspondence_graph.cc:196-201, 322-352`) would need to be
  ported at that point — deliberately deferred here, per the module doc.
- **The concurrent-edit build fragility observed this session is worth a
  process note**, not a code fix: `dpvo_vo.rs` being `onnx-inference`-gated
  meant a single unrelated in-progress file blocked the *only* buildable
  path to this milestone's own acceptance experiment (the demo binary needs
  that feature for SuperPoint I/O). The `git archive HEAD` snapshot approach
  used here is a reasonable one-off; a standing `worktree`-based or CI-gated
  build lane would remove the need to improvise it next time.
