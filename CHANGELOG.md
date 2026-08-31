# Changelog

All notable changes to `visloc-rs` will be documented here.

## Unreleased

### Added

- **Memory-bounded Electro snapshot replay (2026-09-01).** Added the explicit,
  default-off `--snapshot-keypoints-only` path for file-backed verified-pair
  snapshot replay with the plain incremental mapper. It retains keypoints and
  row shape, drops descriptor payloads image-by-image, and re-reads one file at
  a time after calibration to reproduce the original descriptor-bound manifest
  hash. Unsafe descriptor-dependent modes fail closed. Two complete 1,200-image
  runs reproduced the accepted cameras/images/points hashes exactly at
  1200/1200 and 0.03501 m RMSE. Median peak RSS fell **63.6%**, from 4,011,160
  KiB to **1,459,194 KiB (1.39 GiB)**, passing the 2 GiB M3 gate; conservative
  snapshot-to-model wall was 336.90 s, still **14.63x faster** than the
  same-pair COLMAP mapper.

- **Quality-gated Electro BA schedule (2026-09-01).** Added the explicit
  `--global-ba-max-refinements` control without changing its five-round
  default. The frozen 1,200-image champion passes at zero follow-up rounds and
  eight LM iterations: two byte-identical runs register 1200/1200 at 0.03501 m
  RMSE, with 212.93 s median mapper core and 268.49 s median wall. That is
  18.36x faster than the same-pair COLMAP mapper and 80.4% below the prior
  visloc wall. The five-iteration control was rejected at 0.06788 m RMSE.
  Peak RSS remains 3.83 GiB and is still open work.

- **Pure-visual block Schur BA and symbolic reuse (2026-09-01).** Sparse
  pure-visual BA now assembles deterministic 6x6 pose blocks directly, retains
  only the lower Schur triangle, moves those blocks into the Cholesky numeric
  phase without a scalar COO copy, and reuses symbolic sparsity across stable
  LM iterations. Visual-inertial, prior-bearing, dense, and parallel BA retain
  their existing general path. On the frozen Electro 1,200-image champion the
  three COLMAP text files remain SHA-256 identical; median 20-iteration BA fell
  **15.0%** (153.764 s to 130.681 s), mapper core fell 8.1% (1425.985 s to
  1310.111 s), and external mapper wall fell 8.3% (1490.07 s to 1366.75 s),
  lifting the same-pair COLMAP mapper advantage from 3.31x to **3.61x**. Peak
  RSS remains 3.83 GiB and is explicitly still open work.

- **Electro 1,200-image quality parity and decision trace (2026-09-01).** On
  the frozen 12,000-pair snapshot, an explicit mapper cap of 96 plus the
  existing bounded post-refinement registration pass reaches **1200/1200** at
  **0.03224 m** centre RMSE, versus COLMAP's 1200/1200 at 0.04679 m. The
  visloc mapper remains **3.31x faster** (1490.07 s vs 4929.56 s) with a
  3.83 GiB peak RSS. Attribution separated early one-shot PnP exhaustion from
  the cap64 accuracy loss; cap128 and uncapped growth were consecutive
  regressions. Added a compact deterministic debug/timing trace summarizer,
  machine-readable attribution ledger, report, and refreshed README PNG/GIF.
  Fixed cap96 is deliberately not a default: its courtyard negative regressed
  accuracy, while the unchanged default reproduced courtyard 38/38 at
  0.005379 m and South Building 128/128 at 0.73 cm with exact model hashes.

- **Milestone 4 large-scale unordered-SfM plan (2026-08-31).** Added a
  planning-only, primary-source-linked runbook for scaling from the frozen
  courtyard control to ETH3D `electro` (300-image probe, then 1,200 images),
  10k+ scene-sharded validation, and optional 100k+/million-image datasets.
  It defines licensing/availability caveats, O(NK) retrieval, bounded
  candidate manifests, atomic resumable feature/match shards, component
  recovery, resource ceilings, metrics, stop/go gates, and preservation of the
  exhaustive 38/38 / 0.005379 m courtyard gate. No dataset was downloaded or
  large run launched.

- **Bounded pre-match candidate schedules (2026-08-31).** Milestone 3 adds
  deterministic `vlad-mutual` and bounded `vlad-union` candidate sources plus
  the image-name-bound, atomic `visloc_candidate_manifest_v1` export/import
  path. Selection uses only numeric-stem local overlap and pre-match VLAD
  retrieval; frozen raw matches are consumed only after selection for
  verification replay, with no GT or extrinsic input. On the immutable
  high-resolution courtyard artifact, exhaustive control was **703** candidates,
  **366** verified / **261,724** inliers, **38/38**, **43,852 tracks / 152,432
  observations**, 0.579 px and **0.5379 cm**. The local-window≤3 + VLAD top-8
  union capped at **200** candidates (**172** verified / **199,871** inliers)
  and retained **38/38**, **45,016 / 148,192**, 0.537 px and **0.66 cm**
  (71.6% fewer proposed pairs; manifest SHA-256
  `2d654b9e…5ed32c1`). Non-mutual VLAD top-8 used 188 candidates and reached
  38/38 but 14.17 cm; mutual top-8 used 116 and reached 13/38 (33.97 cm on
  its partial subset); sequential stem≤3 used 108 and reached 23/38 (1.08 cm
  on its partial subset); vocab-tree top-4 used 104 and reached 38/38 but was
  slower and not a ≤1 cm result. The Python benchmark now exposes named
  schedules, manifest validation, `--allow-incomplete` negative A/B JSON,
  candidate/verification/mapping counters and elapsed time; exhaustive remains
  the default/control and the reduced 200-pair result is not the README
  champion.

- **Office milestone-2 evidence and Auto post quality guard (2026-08-31).** On
  the frozen `office-authoritative-venv` feature cache, exhaustive `.8` +
  cross-check raised the candidate graph to **325 pairs**, but the safe
  verified graph still had a 23-image component and isolated `DSC_0238`–
  `DSC_0240`; the exact current Auto run remained **17/26** before post and
  **18/26** after one post-registration addition (`1,082` tracks / `3,037`
  observations, `1.512 px`).  The missing no-flag images were
  `DSC_0236`–`DSC_0241`, `DSC_0253`, and `DSC_0254`; `.8` cross-check had zero
  verified support for `DSC_0238`–`DSC_0240`.  A bounded `.9` cross-check
  control reached **21/26** but still left `DSC_0237`–`DSC_0241` incomplete;
  no-cross-check rescue connected the graph but was rejected for unsafe
  conflicts.  Auto post candidates now require both a strict registration
  increase and finite, non-increasing mean reprojection error, so extra
  registrations cannot silently replace a cleaner model.  The change is
  general/default-safe and does not lower PnP thresholds or admit unverified
  bridges; focused Auto tests, release example check, and the frozen Office
  rerun passed.  Under safe `.8` verification, the graph gives a hard
  connectivity ceiling below 26/26; no unsupported pose was fabricated.

- **Office high-density frontend rescue diagnostic (2026-08-31).** A genuine
  `colmap/colmap:latest` CPU run (COLMAP `4.2.0.dev0`) on the full-resolution
  26-image Office set, with supplied PINHOLE intrinsics only, produced
  **167,891** SIFT rows, **325/325** raw pairs and **173** non-empty verified
  pairs (**83,438** inliers).  Official COLMAP mapping reached **26/26** with
  13,425 points / 44,979 observations and 0.50 cm calibration-reference
  centre RMSE.  The same exported keypoints/descriptors through visloc's
  existing NN/full-verifier path reached **26/26**, 17,772 tracks / 47,503
  observations, 0.380 px mean reprojection, and 0.34 cm centre RMSE.
  Bounded SuperPoint/LightGlue rematching of the frozen cache reached only
  **21/26** (88/325 verified, 7,641 inliers), with `DSC_0238`–`DSC_0240`
  still lacking raw support.  The existing explicit supplement/import path
  is the reusable deterministic artifact-merge mechanism; no automatic
  cross-frontend trigger is added because the successful official SIFT arm is
  a separate diagnostic frontend and the learned rescue does not provide a
  safe complete candidate.  Durable models, DB, feature export, hashes and
  logs are under
  `runs/office-colmap-sift8192-20260831`; exact commands and per-image
  support are recorded in the current handover.

- **Courtyard benchmark automation (2026-08-31).** Added the
  hash-pinned `scripts/benchmark_courtyard.sh` / JSON manifest entry point.
  Its dependency-free verify-only smoke validates the durable 38-image,
  439,481-feature, exhaustive 703-pair / 306,324-raw-match artifact, both
  38/38 text models, and the README PNG/GIF/table claims; the stored visloc
  control is **0.005379 m** centre RMSE (under the **0.01 m** gate). `--full`
  reruns the recorded per-image-calibration mapping and emits deterministic
  JSON/logs, with optional COLMAP validation/run and visual regeneration.
  Large dataset-derived files remain external; hosted CI runs only parser,
  hash/threshold tests and shell syntax, with self-hosted instructions in
  [the benchmark guide](docs/courtyard_benchmark.md).

- **Auto demo-default closure and snapshot compatibility (2026-08-31).** The
  two SfM demos now omit `--next-image-policy` as `Auto`, while
  `IncrementalSfmConfig::default()` remains historical `CorrespondenceCount`.
  Auto compares Count whenever the Visibility candidate is incomplete, then
  runs post-refinement only from a clean selected candidate and adopts it only
  for a strict registered-image increase.  On the frozen no-flag caches,
  South was **127/128 vs 123/128 → post 128/128**, **20,554/93,647**, 1.406 px,
  0.73 cm; terrace was **12/23 vs 23/23**, selected Count with post skipped,
  **3,595/10,119**, 1.574 px, 2.56 cm (avoiding the earlier 78.54 cm
  recovery+post path); office was **17/26 vs 17/26 → post 18/26**,
  **1,082/3,037**, 1.512 px, 0.45 cm; and courtyard remained **38/38** at
  **0.5379 cm** with byte-identical champion camera/image/point hashes
  (`76fc7583…`, `a14ac6b9…`, `d7b680e6…`).  Snapshot import without an explicit
  policy still forces Count: the low-resolution replay matched its Count
  control byte-for-byte (**20,649/68,514**, 0.342 px); explicit Auto remains
  available and reproduces the visibility model (**20,086/66,894**, 0.281 px).
  Terrace/office same-cache detached comparisons are now separated from their
  unavailable historical feature-cache arms; EuRoC same-cache classification
  uses one predeclared `max(5%, 5 mm)` metric delta plus exact pose/update
  coverage.  Exact commands, run paths, hashes, and pass/inconclusive labels
  are in [the final non-regression record](docs/nonregression_20260830.md) and
  [the current handover](docs/codex_handover.md).

- **Final closure audit (historical pre-final-Auto patch, 2026-08-31).** The
  durable exhaustive courtyard
  control is confirmed at **38/38, 0.5379 cm** (366/703 verified,
  43,852/152,432 tracks/observations, 0.579 px) with identical first/repeat
  model hashes; its 64-entry `SHA256SUMS` digest is
  `12e91cd3a2e595625ef167d8cd8a2af6310d3ea3cd1e3b1a0c2f8264004fa96b` under
  `/media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/artifacts/colmap_highres_exhaustive_allpairs_20260830`.
  The closure classification is explicit: South default Count is **fail**
  against strict 128/128 (127/128; opt-in post reaches 128/128), terrace and
  office are **inconclusive** against their historical 12.37/0.37 cm arms
  because the archived feature bytes are unavailable, and EuRoC open/loop
  pass the recorded controls while full/full2v/full2vh/full2vhi remain
  **inconclusive** numerically because no same-cache tolerance was specified
  (all retain 2,700/2,700 poses). EuRoC Auto is N/A because the loop runner
  has no next-image policy. Linux `scripts/check.sh`, fmt, and diff checks
  pass; the complete table, hashes, exact commands, and remaining caveats are
  in [the current handover](docs/codex_handover.md).  This entry preserves the
  pre-final-policy record; the first entry above is authoritative for the
  current omitted-policy behavior.

- **GT-free automatic next-image fallback (historical pre-final-Auto A/B,
  2026-08-31).** Added the explicit
  `--next-image-policy auto` mode to both SfM demos and to the incremental
  library API. It runs the visibility-pyramid policy first, retries from the
  same immutable inputs with correspondence-count ranking only when its
  registered fraction is below the fixed 90% completeness threshold, and
  selects lexicographically by registered cameras, valid observations, track
  count, then lower finite mean reprojection (exact ties retain visibility).
  Only small config values are cloned; feature and match storage remains
  shared. On the cache-fixed A/B, Auto selected count for terrace
  (**12/23 -> 23/23, 3,595 tracks / 10,119 observations, 1.574 px, 2.56 cm**)
  and office (**17/26 for both; visibility 931/2,612 at 1.533 px versus
  count 1,024/2,904 at 1.531 px, selected count, 0.43 cm**), and retained
  visibility for South (**127/128, 20,313 / 92,682, 1.409 px, 0.74 cm**) and
  exhaustive courtyard (**38/38, 43,852 / 152,432, 0.579 px, 0.5379 cm**).
  The low-resolution verified-pair snapshot also selected visibility
  (**38/38, 20,086 / 66,894, 0.281 px, 3.42 cm**) and reproduced its model
  byte-for-byte; explicit count was **38/38, 20,649 / 68,514, 0.342 px,
  8.78 cm** with different model hashes. Thus Auto is deterministic and
  useful, but it is intentionally not the new default: changing the
  snapshot's current Count-default bytes would violate compatibility, so
  `NextImagePolicy::default()` and omitted CLI options remain Count. The 90%
  guard can deliberately accept a complete-but-poor visibility candidate
  without testing Count; the snapshot result (3.42 versus 8.78 cm) supports
  visibility-first but does not establish global optimality. Observed Auto
  overhead was negligible when Count was skipped (courtyard 1:28, 1.07 GB
  peak RSS) and a full second mapper when it fell back (terrace 1:11, 252 MB;
  office 0:49, 189 MB; South 3:28, 960 MB). Focused unit/CLI tests, release
  check, formatting, and diff checks pass. Exact commands, logs, artifact
  paths, hashes, and the deterministic snapshot repeat are in
  [the cache-fixed A/B record](docs/nonregression_20260830.md).  Its omitted
  policy was subsequently changed to Auto only after the final closure A/B
  above; the library Count default and snapshot Count override remain.

- **Unified Auto/recovery/post validation (historical explicit-stack A/B,
  2026-08-31).** With
  `--next-image-policy auto --geometry-guided-conflict-recovery
  --post-refinement-registration --final-iterative-refinement
  --pnp-max-iterations 100000 --min-pnp-inliers 8`, the frozen Legacy caches
  produced terrace **23/23, 3,522 tracks / 9,790 observations, 1.559 px,
  78.54 cm**, office **20/26, 1,210 / 3,325, 1.502 px, 0.53 cm**, and South
  **128/128, 21,427 / 98,981, 1.379 px, 0.92 cm**. Courtyard's imported
  full-verifier/per-image-calibration run retained **38/38, 43,852 / 152,432,
  0.579 px, 0.54 cm** and matched the champion model hashes; terrace is a
  clear accuracy non-regression failure, so Auto remains explicit and omitted
  defaults are unchanged. Terrace/office/South and courtyard repeats were
  byte-identical. The EuRoC exporter was subsequently completed with the
  hardened helper: legacy PID 3686761 was stopped gracefully after frame 1848,
  and disjoint atomic workers repaired `[1849,1966)`, `[1966,2083)`,
  `[2083,2200)`, `[2200,2450)`, and `[2450,2700)`. The canonical manifest has
  2,700 left/right feature files, 2,700 stereo files, and 2,699 temporal files
  (10,799 total), with JSON SHA-256
  `6c6f9f64551882bd5dafbe98719348879511c10cfeed280dcee25630db97ed38`.
  Exact range hashes, validation, commands, and paths are in [the
  non-regression record](docs/nonregression_20260830.md). The final
  helper-hardening `scripts/check.sh` rerun exited 0 (Python 243 tests, eight
  skipped); the focused exporter safety suite is 5/5.

- **EuRoC MH_03 same-cache baseline/current ATE-RPE (2026-08-31).** The
  detached `2a36d44` baseline and current worktree were run against the same
  frozen 2,700-frame rectified SuperPoint/LightGlue cache. All six variants
  retained 2,700/2,700 poses. ATE translation RMSE (SE3 / Sim3, metres) for
  baseline → current was: open `2.384514/2.153919 → 2.401288/2.174040`, loop
  `0.498364/0.487327 → 0.445577/0.439303`, full
  `0.084135/0.083063 → 0.087830/0.084345`, full2v
  `0.058594/0.050140 → 0.058698/0.054061`, full2vh
  `0.063683/0.053853 → 0.063884/0.056473`, and full2vhi
  `0.059021/0.050955 → 0.059811/0.053683`. Consecutive one-frame translation
  RPE (SE3 / Sim3) was respectively `0.068109/0.061289 → 0.068052/0.061246`,
  `0.069934/0.070975 → 0.070392/0.071137`,
  `0.073060/0.072916 → 0.073224/0.072961`,
  `0.071469/0.071164 → 0.071357/0.071129`,
  `0.071463/0.071122 → 0.071362/0.071064`, and
  `0.071377/0.071079 → 0.071241/0.070979`.  The benchmark logs/models are
  retained under `/home/sasaki/euroc_mh03_official_20260830/runs/`; the
  baseline/current binary hashes and exact command are documented in
  `docs/nonregression_20260830.md`. This EuRoC loop-closure example has no
  `NextImagePolicy`/`--next-image-policy` selector, so a separate current
  “Auto” ATE/RPE run is not applicable; Auto remains evaluated by the ETH3D
  incremental/SfM A/B. No registration regression was observed, but current
  full2vhi Sim3 is 2.728 mm worse than baseline, so this is an evidence-only
  same-cache non-regression record, not a claim of universal improvement. A
  correctly configured current full2vhi repeat reproduced the `vo_poses.txt`
  and `est.tum` hashes and all four metrics byte-for-byte; the repeat evidence
  is retained in the non-regression record.

- **Historical next-image ranking restoration A/B (2026-08-31).** The
  default `NextImagePolicy` (including both demos when the option is omitted)
  now uses the historical raw correspondence-count ranking restored
  from the pre-`9c35f72` semantics; `--next-image-policy visibility` remains an
  explicit opt-in. Targeted tests, release build, `cargo fmt --check`, and
  `git diff --check` pass. On identical feature caches, terrace recovered
  **23/23, 3,595 tracks / 10,119 observations, 1.574 px, 2.56 cm** and office
  remained **17/26, 1,024 / 2,904, 1.531 px, 0.43 cm**. The same default count
  policy produced South **123/128, 19,658 / 88,114, 1.411 px, 1.15 cm** versus
  the current visibility result **127/128, 0.74 cm**, and the exhaustive
  courtyard control fell from explicit-visibility **38/38, 43,852 / 152,432,
  0.5379 cm** to **38/38, 42,352 / 142,379, 5.03 cm**. Thus the terrace
  regression is fixed, but the requested cross-suite/accuracy non-regression
  gates are not met by this unconditional default switch; no completion claim
  is made. Full commands and artifacts are recorded in
  [the cache-fixed A/B record](docs/nonregression_20260830.md).

- **Cross-dataset non-regression follow-up (2026-08-30).** A durable external
  venv at `/media/sasaki/aiueo1/visloc-rs/envs/nonregression_20260830` was
  provisioned without changing system or repository manifests: Python 3.12.3,
  CPU `torch 2.3.1+cpu`/`torchvision 0.18.1+cpu`, LightGlue commit
  `eb42fee2d71449efb0aa5c10549752b5d75384d8`, `evo 1.31.1`, and current
  `pip-freeze.txt` SHA-256
  `dae41bf42ceedf9a214cd040d002f941d88f8dcc036d5ecd1dc637808dad8f9f`.
  South Building's documented default command was rerun twice on the verified
  128-image archive: both runs produced **127/128** (the same missing
  `P1180163.JPG`), 750/872 verified pairs, 277,354 inliers, 20,313 tracks /
  92,682 observations, 1.409 px mean reprojection, and **0.74 cm**
  COLMAP-reference Sim(3) RMSE (median 0.60, max 1.59 cm). The model files
  were byte-identical across repeats; this is one missing registration versus
  the recorded 128/128, 1.09 cm default control, so it is a registration
  regression candidate rather than a pass claim. The explicitly separate
  `--colmap-style` A/B recovered **128/128**, 20,570/93,844 tracks/observations,
  1.405 px and **0.40 cm**, without changing the default.
  Terrace/office archives are SHA-256 verified, but their historical cached
  feature files are unavailable. An explicitly labelled external LightGlue
  SuperPoint max-dimension-3200 frontend reproduction (CPU, area resize and
  coordinate unscale) yielded terrace **23/23, 1.528 px, 132.07 cm** and
  office **18/26, 1.519 px, 1.28 cm**; these are not comparable exact
  replays of the cached-feature 12.37/0.37 cm controls. Current full-resolution
  helper A/Bs yielded terrace **23/23, 1.511 px, 130.62 cm** and office
  **17/26, 1.522 px, 0.35 cm**, confirming that the missing cached frontend
  prevents a valid regression verdict. Logs/models are under
  `/media/sasaki/aiueo1/visloc-rs/eth3d/nonregression_20260830/runs/` and the
  command/config caveat is in
  [the non-regression record](docs/nonregression_20260830.md). For EuRoC,
  `evo 1.31.1` and the rectifier were provisioned, but both official legacy
  MH_03 URLs timed out (`curl` 28), the official Research Collection endpoint
  returned 403, its metadata reports a 12,096.15 MB Machine Hall archive, and
  only 4.7 GB durable space remained; the exact runner still lacks
  `mav0/cam0/sensor.yaml`, so no substitute dataset or ATE/RPE result is
  claimed. This follow-up adds evidence and identifies the South schedule
  difference, but does not silently alter default behavior.

- **South regression provenance closure (2026-08-30).** The historical
  128/128 plain default is the `scripts/run_colmap_sfm_benchmark.sh` result
  corrected by commit `2a36d44` (the earlier 0.58 cm report was superseded by
  1.09 cm); the separate registry 128/128, 0.40 cm run used the scratchpad
  `--colmap-style` command on dirty Windows commit
  `fd7d06901c15869133f6d72e7b5dc14f4ef24d41`. Neither record contains a
  feature-cache byte hash, and no historical South feature cache exists in
  the mounted workspace. The current CPU/LightGlue-v2.3.1 environment and
  dirty tree therefore do not prove a default-code regression: repeated
  plain runs have the same graph (872 candidates, 750 verified, 277,354
  inliers) but stop at **127/128** when `P1180163.JPG` has only **443**
  track-backed correspondences and PnP gets **3** inliers (635 pair
  correspondences are available). The existing opt-in
  `--post-refinement-registration` is the smallest isolated scheduling
  difference: after final refinement it retries that image with **615**
  correspondences and **508** inliers, yielding **128/128, 20,554 tracks /
  93,647 observations, 1.406 px and 0.73 cm**; its cameras/images/points
  files match the debug repeat byte-for-byte. No default behavior was
  changed. Terrace/office likewise have only historical scratchpad commands,
  dirty Windows frontend revisions and dataset tree hashes, not feature bytes;
  their current 3200-pixel CPU reproductions (132.07/1.28 cm) cannot establish
  a regression against 12.37/0.37 cm. For EuRoC, the official Machine Hall
 archive metadata reports **12,096.15 MB**; the durable data disk has only
 4.7 GB free (the root disk has about 120 GB), old official sequence URLs
 timed out and the Research Collection endpoint returned 403, so no valid
 `MH_03_medium/mav0` or ATE/RPE result is claimed. Full logs and provenance
 are in [the non-regression record](docs/nonregression_20260830.md).

- **Cache-fixed mapper source A/B (2026-08-30).** Feeding the same durable
  terrace feature cache and CLI to detached builds localizes the 12/23 default
  result to the visibility-pyramid next-image ranking introduced by `9c35f72`:
  `2a36d44`, `c48750a`, `dac1400`, and `e18dea1` each produce 23/23, while
  `9c35f72` and current `101e5cc` produce 12/23 from the identical
  147/92/18,561 candidate/verified/inlier graph. A temporary raw-count policy
  recovers terrace 23/23 and office 17/26, but reduces South from current
  127/128 to 123/128; it was removed, so no default behavior changed. The
  cache-fixed comparison, output hashes, and exact detached worktree paths are
  recorded in [the non-regression record](docs/nonregression_20260830.md).

- **EuRoC official-input closure (2026-08-31).** The exact ETH Research
  Collection Machine Hall bitstream was retrieved to NVMe via its public DSpace
  API (`7b2419c1-62b5-4714-b7f8-485e5fe3e5fe`), verified at 12,683,729,426
  bytes with SHA-256
  `5ed7d07903f8d19b6c8808e2ae8a0872b281f6e34ef5497023b8ac58c3de0f6f`, and
  its nested official `MH_03_medium.zip` passed `zipinfo -t` (5,420 files;
  SHA-256
  `0f1707dfd6c9cda2c38302f4f7a47abb9a01a622a515dcbd6863730f0990f442`).
  The frozen OpenCV rectification of 2,700 stereo pairs completed in 34.73 s
  (`752x480`, `fx=436.2443`, baseline `0.110078 m`). Detached baseline
  `2a36d44` and current `101e5cc` builds both passed a ten-frame smoke on the
  same frozen inputs with matching temporal/stereo/PnP count ranges; the full
  2,700-frame CPU SuperPoint/LightGlue cache is still running resumably, so no
  full EuRoC ATE/RPE pass claim is made yet. Archive, rectification, build,
  smoke, and export logs are under
  `/home/sasaki/euroc_mh03_official_20260830/`; details are in
  [the non-regression record](docs/nonregression_20260830.md).

- **Exhaustive high-resolution COLMAP SIFT graph audit (2026-08-30).** Removing
  the overlap-3 candidate restriction and matching all 703 pairs with official
  COLMAP 4.2 CPU SIFT produced 306,324 raw matches and 428 calibrated geometry
  rows / 433,279 inliers; the 37 exclusive cross-component images form one
  connected, rank-4 relative-pose graph (59 strong F-to-E edges, condition
  about 8.19).  Official COLMAP single and multiple mapper controls reached
  38/38 at **1.6166 cm** calibration-proxy RMSE (38,422 points / 169,590
  observations).  The same external features/raw matches in visloc with
  `--exhaustive` reached reproducible **38/38, 43,852 tracks / 152,432
  observations, 0.579 px, 0.5379 cm**; a repeated run was byte-identical.
  The old two-submap direction-constrained merge remained worse (5.37345 cm
  robust, 9.8599 cm unrobust), so no merge implementation was added.  These are
  calibration-proxy controls, not a replacement for the supplied 2.842 cm
  champion or an independent laser-GT result; full commands, hashes, graph
  inventory, and score caveats are in
  [the exhaustive high-resolution audit](docs/colmap_highres_exhaustive_audit_20260830.md).

- **Courtyard reproducibility and regression-closure record (2026-08-30).** The
  38/38 high-resolution all-pairs control is now preserved outside the repo at
  `/media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/artifacts/colmap_highres_exhaustive_allpairs_20260830`
  with a verified 64-entry manifest (`SHA256SUMS` digest
  `12e91cd3a2e595625ef167d8cd8a2af6310d3ea3cd1e3b1a0c2f8264004fa96b`), both
  model/log copies, feature/DB provenance, and the exact repeat hashes.  The
  documented extraction, exhaustive 703-pair matching/import, mapping, and
  evaluation commands reproduce **366/703 verified pairs, 43,852/152,432
  tracks/observations, 0.579 px, and 0.5379 cm RMSE**; aligned centre residuals
  are p25/p50/p75/p90/p95/p99/max = **0.2235/0.3042/0.5017/0.9629/1.0336/
  1.2526/1.2737 cm**.  The repeat is byte-identical; this remains the local
  calibration/GT proxy rather than an independent laser-GT score.  South,
  terrace, office, and EuRoC inputs/attempts are archived, but their exact
  authoritative runs remain unmeasured (missing `torch`/`lightglue` or valid
  EuRoC `mav0`); no substitute pass/regression is claimed.  The Linux
  format/clippy (default and `image-io`)/workspace-test, Python, registry,
  docs, examples, MSRV, package, and `scripts/check.sh` gates all pass;
  Windows-only gates were not run.  Exact paths, hashes, blockers, and gate
  logs are in [the closure record](docs/reproducibility_ci_closure_20260830.md).

- **High-resolution COLMAP submap-merge feasibility audit (2026-08-30).** The official
  high-resolution multi-model output was checked without using official extrinsics or
  GT in fitting: model 0 contains `DSC_0286..0308` (23 images) and model 1
  `DSC_0308..0323` (16 images), with only `DSC_0308` shared.  Among the 330 pairs
  between the 22+15 exclusive images, only four had raw rows and all four were
  `rows=0/config=0` (no verified E/F/H), so there were **0 valid exclusive cross-E
  edges**.  Three valid E edges from the shared camera to the right component
  confirmed rotations within 0.147--2.952 degrees and translation directions within
  1.931--11.621 degrees, but a shared-anchor direction system cannot observe relative
  scale; its apparent 9x4 numerical solve collapsed to the trivial zero-scale solution.
  A non-GT shared-camera merge scored **128.349 cm** (s=1) or **60.623 cm** using the
  boundary median-step ratio `s=2.760958293`, versus independent component proxies
  of **2.964 cm** (23/38) and **2.466 cm** (16/38).  Naively concatenating points also
  leaves eight same-camera feature conflicts.  No production submap-merge path or BA
  wiring was added; the next valid experiment is to obtain multiple non-shared,
  verified cross-component bridges and gate rank/conditioning, cheirality, and
  direction residual before solving scale.  Full membership, point statistics,
  cross-edge diagnostics, and the proxy/independent-model caveat are in
  [the submap merge audit](docs/colmap_highres_submap_merge_audit_20260830.md).

- **Courtyard evaluator/frame audit (2026-08-30).** Read-only independent
  reimplementation confirmed the scorer's COLMAP convention `C=-Rᵀt` and
  proper positive-scale Umeyama (synthetic scale 2.75 recovered to `3e-15`
  max centre error; q/−q is identical).  The local `gt` symlink is byte-identical
  to `dslr_calibration_undistorted/images.txt`, so official-vs-`gt` is the
  **0 cm identity proxy**, not an independent laser-camera-pose check; the
  separate scan-evaluation archive is not present locally.  Recomputed controls
  are full COLMAP **38/38, 1.709 cm RMSE / 1.170 cm median / 4.132 cm max**,
  low COLMAP **24/38, 3.590 / 2.488 / 11.487 cm**, and the current
  COLMAP-feature champion **38/38, 2.842 / 2.243 / 7.091 cm**.  The low model's
  common set is cam0 `0323` plus cam1's 23 images; missing cameras are not
  imputed by the scorer.  Full COLMAP's official→model reverse fit is
  **1.355 cm RMSE**, confirming the expected direction-dependent Sim(3) fit.
  Four-camera residual groups and all matched-subset tables are in
  [the evaluator audit](docs/evaluator_audit_20260830.md).  No rig/lever-arm
  metadata or evaluator bug supports a production correction; sub-cm remains
  an aspirational target until the independent scan-evaluation pose/GT is
  provisioned.  The highest-leverage work remains bridge matching,
  track/PnP basin selection, and mapper/BA stability rather than more
  completeness-only heuristics.

- **Official high-resolution COLMAP SIFT A/B (2026-08-30).** Docker
  `colmap/colmap:latest` (`4.2.0.dev0`, image
  `sha256:b809882552887b6471094dcadd2f2eb01656b010663564c43a5e7f04c0a08f2f`)
  extracted the 38 original-resolution courtyard images with CPU SIFT,
  `first_octave=-1`, 4 octaves, `max_num_features=8192`, peak `0.00667`,
  edge `10`, and max orientations `2`: **439,481 rows** (DB
  `b676d6fc...6a7acb1`, exact six-column conversion manifest
  `030b0298...ffd0d2a`).  With per-image calibration, N=3, ratio 0.8,
  implicit cross-check, guided/full verification, plain PnP(100,000)/min8,
  recovery/post/final and sequence fallback off, visloc retained **100/108
  verified pairs / 137,171 inliers**, reached **26/38** (`0298..0323`),
  **29,639 tracks / 80,632 observations / 0.451 px**, and scored **3.612 cm
  RMSE** on the audited calibration `gt` proxy.  Focused visloc rows for
  0307–0308, 0308–0309, 0309–0310 were respectively raw/accepted/E-F-H
  `357/284/91-284-6`, `699/617/391-617-130`, and
  `2,012/1,926/1,544-1,926-35`; growth stopped at 0297 (`65→3` PnP
  inliers).  Official CPU8 COLMAP sequential matching (14.799 min) produced
  **107 raw/geometry rows** (raw sum 142,214; geometry sum 189,112), with
  0308–0309 `694/1,276` and 0309–0310 `1,996/2,445`; its single mapper model
  reached **23/38**, 16,714 points/68,797 observations, 0.668 px and
  **2.964 cm** proxy RMSE.  A multi-model run covered 23+16 (0308 overlap)
  but had separate gauges.  This is a genuine source-exact high-res control,
  not an accuracy-champion update: it does not beat the existing
  COLMAP-feature **38/38, 2.842 cm** or supplied-model **1.709 cm** controls.
  Full commands, hashes, logs and the independent-laser-GT caveat are in
  [the high-resolution feature audit](docs/colmap_highres_feature_audit_20260830.md).

- **Opt-in global E-bearing position initializer audit (2026-08-30).** The
  existing global mapper was first audited on the high-resolution prefix
  artifact: its per-edge unit-displacement rows impose one baseline length on
  every pair, and the legacy run reached **25/38** at **255.89 cm** on its
  registered subset (versus the incremental control's **23/38, 4.51 cm**).
  Added default-off `--global-independent-edge-scales`, which selects the
  verified essential edge stream whenever its minimum support is available
  (the run used **49** E edges; the remaining retained edges use the existing
  verified fallback) and eliminates each unknown edge scale via
  the perpendicular constraint `(I-ddᵀ)(Cj-Ci)=0`; one highest-support edge
  supplies the global scale row.  Connected cyclic graphs use a deterministic
  SVD minimum-norm solve for small geometric nullspaces, while sparse trees
  retain the legacy fallback; CLI parsing and synthetic variable-baseline/tree
  tests pass.  On prefix8192 with the established per-image-calibration,
  N=3, ratio-0.8/cross-check/full-verifier, recovery/post/final stack, the
  corrected path kept **82/101** pairs (**65,784** inliers), reached **38/38**,
  and produced **1,611 tracks / 3,225 observations / 2.162 px** reprojection,
  but scored **468.40 cm** full-scene laser-GT Sim(3) RMSE (rank **104/111**;
  4:15.63, **817,456 kB** peak RSS).  A repeat produced byte-identical
  cameras/images/points hashes.  Adding the pre-existing joint-track
  positioning option was also negative (**38/38, 349 tracks / 723 obs,
  2.508 px, 467.18 cm**).  Thus the correction improves completeness and
  local reprojection but does not beat the incremental/COLMAP controls; the
  full 518k-feature run was not launched.  Models/logs are under
  `/tmp/eth3d_courtyard_highres_cap8192_global_edge_scales_{v4,v5,joint}_20260830`.

- **COLMAP top-scale high-resolution cap audit (2026-08-30).** Audited
  [`ExtractTopScaleFeatures`](https://github.com/colmap/colmap/blob/main/src/colmap/feature/utils.cc#L65-L95)
  and [`FeatureKeypoint::ComputeScale`](https://github.com/colmap/colmap/blob/main/src/colmap/feature/types.cc#L127-L141):
  CPU SIFT keeps the `std::partial_sort` top rows in descending computed
  scale, copying keypoints and descriptors together.  From the immutable
  518,015-row high-resolution artifact
  `/media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/highres_stream38_20260830`,
  a non-destructive derivative selected exactly 8,192 rows per image with
  the same float32 descending partial-sort semantics (manifest
  `/tmp/eth3d_courtyard_highres_topscale8192_manifest_20260830.sha256`,
  digest `2b74b86c04860df612894a74f3088ae236a67c303f3050c24eade215d9c11acb`).
  The sidecar has isotropic sigma rather than COLMAP affine columns, so this
  is a scale-metadata proxy for `ComputeScale`, not an affine reproduction;
  all feature/locus rows remain aligned.  The derivative has 311,296 rows,
  40.0037% endpoint-row overlap with the prefix-8,192 derivative, and all
  38 per-image selections/manifests validated.  With per-image calibration,
  N=3, ratio 0.8/cross-check/full verification, plain PnP (100,000; min 8),
  recovery, post-registration and final refinement, sequence fallback off
  retained 98/255 candidate pairs and verified 90/98 (89,888 inliers), then
  stopped at 14/38 (12,597 tracks / 41,917 observations / 0.511 px; 16.10 cm
  registered-subset laser-GT RMSE; runtime 4:06.80, peak RSS 818,920 kB).
  The after-post historical-median fallback control promoted 13 edges but
  stopped at 22/38: 99/256 candidates, 91/99 verified (91,227 inliers),
  18,639 tracks / 57,748 observations / 0.481 px and 1.58 cm
  registered-subset RMSE.  Its next consecutive pair had 120 F-supported
  matches but only 15 selected essential matches, below the 30-match fallback
  gate, so no provisional pose was admitted; no full-scene score exists.
  Log/models: `/tmp/eth3d_courtyard_highres_topscale8192_window3_control_20260830.log`
  and `/tmp/eth3d_courtyard_topscale8192_window3_after_post_median_20260830.log`.
  This does not reproduce the prefix-8,192 control (23/38, 4.51 cm subset)
  or full-artifact control (23/38, 5.02 cm subset), and introduced no code or
  heuristic.

- **Full high-resolution feature reconstruction control (2026-08-30).** The
  immutable streaming artifact
  `/media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/highres_stream38_20260830`
  contains **518,015 feature rows** across 38 images (76 files); its checked
  manifest is `/tmp/eth3d_courtyard_highres_stream38_manifest_20260830.sha256`
  with digest `e63931fef8f9ac8257d552ff97e0bd1b5a1ecd73c56bd114a3046d85c85e42af`.
  The per-image-calibration, N=3 pair-window control used ratio **0.8**,
  cross-check/full verification, plain incremental PnP (`100000` iterations,
  minimum 8 inliers), conflict recovery, post-registration and final
  refinement, with sequence fallback off.  It retained **99/257 candidate
  pairs**, verified **92/99** with **141,758 inlier correspondences**, and
  grew to **23/38** cameras: `DSC_0306` **86→29**, `DSC_0307` **83→64**,
  `DSC_0308` **20→17** PnP correspondences/inliers; `DSC_0309` and all later
  images had **0 usable 2D–3D correspondences**.  The model contains
  **28,014 tracks / 86,304 observations / 0.431 px** mean reprojection; its
  registered-subset laser-GT score is **5.02 cm RMSE** (23/38, so no full-scene
  score).  Runtime was **7:05.65** with peak RSS **2,791,864 kB**; log/model:
  `/tmp/eth3d_courtyard_highres_full_window3_control_20260830[.log]`.
  A same-input after-post relaxed fallback run reached the fallback gate for
  `DSC_0309`, but its projected scale was **-1.525542** and was rejected by
  the required finite-positive policy; later images consequently remained
  unregistered.  It produced byte-identical cameras/images/points hashes and
  the same **23/38, 28,014 / 86,304 / 0.431 px / 5.02 cm** result (runtime
  **7:24.30**, peak RSS **2,791,824 kB**).  This closes the full-artifact
  control without adding a new heuristic; logs/models:
  `/tmp/eth3d_courtyard_highres_full_window3_{control_20260830,after_post_relaxed_20260830}`.

- **Full high-resolution median-scale sequence fallback control
  (2026-08-30).** Reused the immutable 518,015-row artifact and the full
  per-image-calibration/N=3/ratio-0.8/cross-check/full-verifier/plain-PnP
  stack above, enabling only `--sequence-relative-pose-fallback
  --sequence-fallback-after-post` (the historical median-magnitude scale;
  no constant-velocity or carry flags).  The automatic sequence F→E pass
  promoted **9** stable edges, including **2** high-support spread overrides;
  the graph remained **99/257 candidates**, **92/99 verified**, and **141,758
  inliers**.  Ordinary growth and post-PnP again reached **23/38**:
  `DSC_0306` **86→29**, `DSC_0307` **83→64**, `DSC_0308` **20→17**, while
  `DSC_0309` onward had no ordinary usable 2D–3D support.
  After-post fallback examined `DSC_0309` through stored essential pair 57
  with **118** selected matches and **118/118** cheirality, but obtained
  **0** valid triangulations (`valid_ratio=0.000000`), so the ordinary
  standard gate (`30` minimum and **50%**) rejected it.  The 30% admission
  exception was not applicable because this edge was not one of the marked
  high-support override entries; later predecessors consequently remained
  unregistered.  The output was **23/38**, **28,014 tracks / 86,304
  observations / 0.431 px**, and registered-subset laser-GT **5.02 cm RMSE**
  (no full-scene score), with runtime **7:21.31** and peak RSS **2,791,952
  kB**.  Cameras/images/points hashes matched the sequence-off control
  exactly.  Log/model:
  `/tmp/eth3d_courtyard_highres_full_window3_after_post_median_20260830[.log]`.
  This is a negative structural control; no new heuristic or code change was
  made.

- **Consecutive provisional scale carry (2026-08-30).** Added the separate
  default-off `--sequence-fallback-carry-scale` policy for
  `--sequence-fallback-after-post`.  The first provisional registration still
  uses the relaxed constant-velocity projection; when no ordinary post/PnP
  image is inserted, the next consecutive fallback reuses the previous
  accepted baseline magnitude, subject only to the finite/positive
  **0.25x–4x** recent-median sanity bound.  Any ordinary post/PnP insertion
  clears the carry chain, and an invalid/stale carried value falls back to
  the freshly projected proposal.  CLI validation, deterministic pose-rescale
  and carry-state tests cover default-off behavior, first projection, carry,
  reset, invalid fallback, and rotation/convention preservation.
  On the exact high-resolution cap-8,192/N=3 after-post stack
  (`--match-ratio 0.8 --verification-mode full --pnp-max-iterations 100000
  --min-pnp-inliers 8 --geometry-guided-conflict-recovery
  --post-refinement-registration --final-iterative-refinement
  --sequence-relative-pose-fallback --sequence-fallback-after-post
  --sequence-relaxed-constant-velocity-scale
  --sequence-fallback-carry-scale`), `DSC_0309` kept its first-fallback
  projection **1.146905** (recent median **1.898480**, MAD **0.110514**), and
  `DSC_0310` reused that carried **1.146905** instead of its fresh projection
  **1.525837** (recent median **1.581204**, MAD **0.427790**).  The first
  fallback resumed **0** PnP images; the second resumed **13**, which cleared
  the carry state and completed **38/38**.  The output had **27,906 tracks /
  74,962 observations / 0.291 px** mean reprojection and full-scene laser-GT
  Sim(3) **31.15 cm RMSE** (median **27.66 cm**, max **49.47 cm** at
  `DSC_0310`, fitted scale **0.694170**), compared with the no-carry
  after-post result **29.21 cm**.  This is an honest negative accuracy A/B;
  the flag remains opt-in and no scale blend/sweep was run.  Runtime was
  **4:41.74**, peak RSS **817,412 kB**; log/model:
  `/tmp/eth3d_courtyard_highres_cap8192_window3_seqfallback_after_post_relaxed_carry_20260830[.log]`.
  The low-resolution lossless snapshot identity run with the carry flag
  remained byte-identical to control at **38/38**, **20,777 tracks /
  72,431 observations / 0.282 px**, and **3.422 cm** laser-GT RMSE
  (`/tmp/seq_override_snapshot_after_post_relaxed_carry_20260830`; camera,
  image, and point hashes unchanged).  No further experiment was performed.

- **Sequence-fallback after-post scheduling (2026-08-30).** Added separate
  default-off `--sequence-fallback-after-post`.  With sequence fallback
  enabled, ordinary growth no longer admits provisional relative poses; the
  existing recovery and post-refinement PnP/BA stage runs first, then one
  validated consecutive fallback is admitted at a time and ordinary PnP is
  resumed before the next fallback.  The eager sequence mode remains
  unchanged when this flag is absent, and CLI validation requires both the
  sequence fallback and post-refinement flags.  Tests cover default/eager
  scheduling identity, deferred growth gating, CLI defaults, missing
  dependency, and validation.  On the exact high-resolution cap-8,192/N=3
  stack with high-support F→E override, 30% admission, relaxed projection,
  recovery, post-registration and final refinement, ordinary growth stopped
  at **20/38**; the ordinary post pass then registered `DSC_0306`–`DSC_0308`
  (**28→25**, **15→15**, **13→12** PnP inliers), reaching **23/38** before
  sequence fallback.  The corresponding no-fallback 23-camera control was
  **4.51 cm** RMSE on its registered subset (diagnostic only).  After the
  post-stage, fallback admitted `DSC_0309` at projected scale **1.146905**
  (recent median **1.898480**, MAD **0.110514**) and `DSC_0310` at **1.525837**
  (median **1.581204**, MAD **0.427790**); the first resumed **0** ordinary
  PnP images and the second resumed **13**, completing **38/38**.  The final
  model had **27,907 tracks / 74,967 observations / 0.291 px** mean
  reprojection and full-scene laser-GT Sim(3) **29.21 cm RMSE** (median
  **25.92 cm**, max **46.56 cm** at `DSC_0310`, fitted scale `0.693343`).
  This improves the eager relaxed result (**60.42 cm**, also 38/38), but is
  not an accuracy champion.  Runtime was **4:27.62**, peak RSS **817,492 kB**;
  log:
  `/tmp/eth3d_courtyard_highres_cap8192_window3_seqfallback_after_post_relaxed_20260830.log`;
  model:
  `/tmp/eth3d_courtyard_highres_cap8192_window3_seqfallback_after_post_relaxed_20260830`.
  The low-resolution verified snapshot remained byte-identical to control at
  **38/38**, **20,777 tracks / 72,431 observations / 0.282 px**, and **3.422
  cm** laser-GT RMSE, including identical camera/image/point hashes.  No
  additional scale sweep was performed.

- **Sequence-fallback relaxed constant-velocity scale (2026-08-30).** Added
  separate default-off `--sequence-relaxed-constant-velocity-scale`; it keeps
  the positive finite constant-velocity projection but replaces the strict
  local 3-MAD fence with only a broad **0.25x–4x** recent-median bound.  The
  strict `--sequence-constant-velocity-scale` policy remains reproducible and
  the historical median-magnitude policy remains the default; the two
  projected policies are mutually exclusive.  Tests cover a positive turn
  admitted by the relaxed policy, broad-bound rejection, finite/positive
  validation, CLI default-off behavior, missing fallback dependency, and
  strict/relaxed mutual exclusion.  On the exact high-resolution cap-8,192 /
  N=3 run with high-support F→E override and 30% admission, relaxed projection
  admitted `DSC_0306` with projected scale **1.647286** (recent median
  `1.701457`, MAD `0.009787`), `DSC_0309` with **0.972742** (median
  `1.623472`, MAD `0.122260`), and `DSC_0310` with **1.304092** (median
  `1.353605`, MAD `0.380864`).  The projected values were within the broad
  bounds; the laser-GT step lengths (**1.067104 m**, **0.792290 m**, and
  **0.693120 m**, consulted only after the run) were not used for admission.
  The run completed **38/38**, with **27,857 tracks / 74,853 observations /
  0.295 px** mean reprojection and full-scene laser-GT Sim(3) **60.42 cm RMSE**
  (median **54.92 cm**, max per-image **92.79 cm**, at `DSC_0298`; fitted
  scale `0.525782`).  This is materially better than the prior median-scale
  fallback (**38/38, 143.05 cm**) and reaches full registration, but is not an
  accuracy champion.  Runtime was **4:21.15**, peak RSS **817,808 kB**; log:
  `/tmp/eth3d_courtyard_highres_cap8192_window3_seqfallback_projected_relaxed_20260830.log`;
  model:
  `/tmp/eth3d_courtyard_highres_cap8192_window3_seqfallback_projected_relaxed_20260830`.
  An exact repeat produced the same **38/38**, **27,857 / 74,853 / 0.295 px**,
  and **60.42 cm** result with byte-identical `cameras.txt`, `images.txt`, and
  `points3D.txt` hashes (`76fc7583…`, `d279aab7…`, `10eb1fee…`); repeat log/model:
  `/tmp/eth3d_courtyard_highres_cap8192_window3_seqfallback_projected_relaxed_20260830_repeat`.
  The low-resolution verified-snapshot identity remained byte-identical to
  control at **38/38**, **20,777 tracks / 72,431 observations / 0.282 px**, and
  **3.422 cm** laser-GT RMSE.  No further scale-blend sweep was performed.

- **Sequence-fallback constant-velocity projected scale (2026-08-30).** Added
  default-off `--sequence-constant-velocity-scale`.  For the opt-in sequence
  fallback only, it predicts a world-frame velocity from the latest one to
  three registered consecutive steps using component-wise medians, projects
  it onto the candidate relative-translation direction, and requires a
  positive finite result inside the existing recent median/3-MAD fence.  The
  historical median step-magnitude estimator remains the default.  Tests cover
  straight motion, a bounded turn, negative/near-zero directions, MAD
  outlier handling, zero-MAD bounding, insufficient history, and CLI
  default/validation.  On the exact high-resolution cap-8,192/N=3 command,
  0306 had median `1.706350`, anchored median `1.701457`, MAD `0.009787`, and
  projected `1.647286` (rejected outside the 3-MAD fence); 0309 had median
  `2.021362`, anchored median `1.999496`, MAD `0.043732`, and projected
  `1.042500` (rejected).  No projected fallback pose was admitted, so 0310
  was not reached; ordinary post-refinement registered 0306–0308 and the run
  stopped at **23/38**, **16,192 tracks / 46,393 observations / 0.324 px**.
  Registered-image laser-GT RMSE was **4.51 cm** (not a full-scene score), with
  runtime **4:29.95** and peak RSS **817,696 kB**.  Log:
  `/tmp/eth3d_courtyard_highres_cap8192_window3_seqfallback_projected_20260830_v2.log`;
  model:
  `/tmp/eth3d_courtyard_highres_cap8192_window3_seqfallback_projected_20260830_v2`.
  The low-resolution verified snapshot identity run remained byte-identical
  to control: **38/38**, **20,777 tracks / 72,431 observations / 0.282 px**,
  **3.422 cm** laser-GT RMSE, with identical camera/image/point hashes under
  `/tmp/seq_override_snapshot_control_20260830_v4` and
  `/tmp/seq_override_snapshot_projected_20260830_v2`.
  This negative result rejects the projected scale as a replacement for the
  current high-support median fallback on this turning trajectory; no further
  heuristic or threshold relaxation was added.

- **Sequence-fallback high-support triangulation admission (2026-08-30).**
  The opt-in sequence fallback now relaxes its final valid-triangulation
  fraction from the historical `>=50%` to `>=30%` only for pair entries
  explicitly marked by the conservative high-support F→E translation-spread
  override; it still requires at least **100** valid triangulations and the
  configured seed floor.  Ordinary/strict sequence pairs and all default
  behavior retain the `>=50%` gate.  Boundary tests cover 100/30% acceptance,
  99-point and sub-30% rejection, the ordinary 50% path, and the unchanged
  prior cheirality/reprojection gates.  On the exact high-resolution
  cap-8,192/N=3 command from the preceding trace, the fallback admitted
  `DSC_0306` **156/180** (`0.866667`, scale **1.706350**), `DSC_0309`
  **233/234** (`0.995726`, scale **1.623455**), and the high-support-only
  `DSC_0310` **221/555** (`0.398198`, scale **1.623455**); the latter was
  previously rejected at 221 < 278 (half-support).  Subsequent ordinary PnP
  completed all cameras, yielding **38/38**, **27,854 tracks / 74,848
  observations / 0.295 px**, but the full-scene laser-GT score was **143.05
  cm RMSE** (the newly admitted bridge is therefore structural, not an
  accuracy-champion result).  Runtime was **4:19.20**, peak RSS **817,544 kB**;
  log: `/tmp/eth3d_courtyard_highres_cap8192_window3_seqfallback_f2e_relaxed30_20260830.log`;
  model: `/tmp/eth3d_courtyard_highres_cap8192_window3_seqfallback_f2e_relaxed30_20260830`.
  The low-resolution snapshot identity rerun kept control and sequence-enabled
  outputs byte-identical: **38/38**, **20,777 tracks / 72,431 observations /
  0.282 px**, **3.422 cm** laser-GT RMSE, with matching camera/image/point
  SHA-256 files under `/tmp/seq_override_snapshot_{control,enabled}_20260830_v4`.

- **Per-image COLMAP PINHOLE calibration (2026-08-30).** Added the
  default-off `--input-colmap-calibration MODEL_DIR` path to
  `unordered_sfm_demo`.  It reads and validates `cameras.txt` and
  `images.txt`, maps loaded names by exact name/basename/unique stem, checks
  feature bounds and (when `--images-dir` is supplied) decoded image
  dimensions, and supports finite four-parameter `PINHOLE` cameras only.
  `visloc-slam` now exposes `PerImageCameras` plus incremental/global
  convenience wrappers: differing intrinsics are converted losslessly to a
  reference pixel convention for the existing ray/PnP/triangulation/BA
  implementation, while the multi-camera exporter restores native camera IDs
  and pixels.  Shared geometry delegates as an exact feature no-op; per-image
  intrinsics remain fixed and snapshot import/export plus intrinsics
  refinement are rejected until their manifests/parameter blocks are extended.
  Focused tests passed (`4` camera-rig tests, `42` two-view tests, `23` CLI
  tests, and the multi-camera export test); the official high-resolution
  two-image feature smoke produced **4** CSV rows from the `DSC_0305/0306`
  cache and, with `--images-dir`, validated the decoded JPEG dimensions
  against the calibration model.  A full high-resolution reconstruction is
  not claimed: the bounded pair has insufficient parallax for the default
  seed gate.

- **Streaming SIFT feature export and no-default target restoration
  (2026-08-30).** Moved the module-only verified-pair snapshot codec from
  `examples/verified_pair_snapshot.rs` to `src/verified_pair_snapshot.rs` and
  imported it from the unordered demo, so
  `cargo check --workspace --all-targets --no-default-features` no longer
  discovers a fake standalone example.  Added default-off
  `--sift-stream-export`, which requires
  `--feature-extractor sift --export-features-dir DIR --export-features-only`,
  enumerates supported images lexically, decodes/extracts one image at a time,
  validates optional per-image COLMAP calibration dimensions, and atomically
  renames each feature and `_loci.txt` result.  The normal batch path and all
  defaults are unchanged.  Loader-order, one-at-a-time, default-validation,
  byte-identity, and extraction-failure/no-partial-file tests pass.  On a
  Workspace-wide no-default check and clippy also pass; the module-only
  snapshot is now library support rather than an auto-discovered target.  On
  1600×1066 `DSC_0305/0306` subset (`max-keypoints=256`, compatible detector /
  descriptor, bilinear orientations), batch and stream outputs had identical
  SHA-256 files and **793** total rows; stream time was **9.70 s** with
  **698,916 kB** peak RSS.  On official-resolution `DSC_0305` (camera 1),
  `DSC_0308` and `DSC_0309` (camera 3), the stream produced **35,614** rows
  in **4:04.00** with **10,386,164 kB** peak RSS (about 9.9 GiB), remaining
  below the 12 GiB single-process budget and validating both camera dimensions.
  The full 38-image run completed successfully in **52:11.47** with
  **518,015** keypoints and **10,381,448 kB** peak RSS (about 9.9 GiB).
  It produced 38 feature/loci pairs (76 files, 664 MiB); all rows had the
  expected 131 feature fields and four locus fields, and no temporary files
  remained.  The per-image calibration mapping covered all 38 images and all
  decoded JPEG dimensions matched their referenced PINHOLE camera.  The
  reproducible command and log were:
  `RAYON_NUM_THREADS=1 /usr/bin/time -v target/release/examples/unordered_sfm_demo
  --feature-extractor sift --images-dir
  /home/sasaki/datasets/eth3d/courtyard/images/dslr_images_undistorted
  --input-colmap-calibration
  /home/sasaki/datasets/eth3d/courtyard/dslr_calibration_undistorted
  --sift-max-keypoints 8192 --sift-max-orientations 2
  --sift-vlfeat-compatible-detector --sift-vlfeat-compatible-descriptor
  --sift-vlfeat-bilinear-orientations --sift-vlfeat-compatible-output-order
  --export-features-dir
  /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/highres_stream38_20260830
  --export-features-only --sift-stream-export --out-colmap
  /tmp/eth3d_courtyard_highres_stream38_model_20260830 >
  /tmp/eth3d_courtyard_highres_stream38_run_20260830.log 2>&1`; the
  76-file SHA-256 manifest is
  `/tmp/eth3d_courtyard_highres_stream38_manifest_20260830.sha256` (manifest
  digest `e63931fef8f9ac8257d552ff97e0bd1b5a1ecd73c56bd114a3046d85c85e42af`).
  Progress was emitted once per completed image.

- **High-resolution capped reconstruction probe (2026-08-30).** Preserved the
  38-image streaming artifact and created the non-destructive
  `highres_stream38_cap4096_20260830` derivative by copying the first 4,096
  feature/locus rows per image.  All 38 feature/locus pairs, 155,648 rows,
  bounds, coordinates, descriptor/locus ordering, and the 76-file manifest
  were validated; the manifest digest is
  `42bf6e26a1ccb3828d772bdfce09d0f840343fbbfa86d0a20cd794becef87805`.
  With per-image PINHOLE calibration, full verification, ratio `0.8`,
  `--pair-stem-window 3`, `pnp-max-iterations=100000`, `min-pnp-inliers=8`,
  conflict recovery, post-registration, and final iterative refinement, the
  exact capped run retained **108** candidate pairs, verified **57** pairs
  with **29,644** inliers, and registered **11/38** images (5,302 tracks,
  17,450 observations, mean reprojection **0.277 px**) in **58.93 s** with
  **344,104 kB** peak RSS.  The partial 11-camera laser-GT score was
  **1.33 cm RMSE** (not a full-scene result).  A bounded `window=4`
  diagnostic added 34 candidate edges; at ratio `0.8` + cross-check, 11 of
  those edges were verified (2,071 inliers), but the only new edges adjacent
  to the first stalled transition (0296→0297) had zero accepted inliers
  (`0293–0297` and `0297–0301`).  Therefore no `window=4` reconstruction or
  exhaustive run was launched; the immediate bottleneck remains the
  0296→0297 transition rather than an untested wider window.  The capped
  reconstruction log is `/tmp/eth3d_courtyard_highres_cap4096_window3_20260830.log`.

- **High-resolution 8,192-feature reconstruction probe (2026-08-30).** A
  second non-destructive derivative,
  `/media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/highres_stream38_cap8192_20260830`,
  retained the first 8,192 feature/locus rows from each immutable full
  artifact.  Prefix bytes, 38 feature/locus pairs, 311,296 rows, 131/4
  fields, coordinates, bounds, and the 76-file manifest were validated; its
  manifest digest is
  `39985ad86d14ea2ac73cc9e7970dda87ebf60827dc7e90e73eabd8db7c42ce54`.
  The per-image-calibrated N=3 run with ratio `0.8` + cross-check, full
  verification, `pnp-max-iterations=100000`, `min-pnp-inliers=8`, conflict
  recovery, post-registration, and final iterative refinement retained 108
  candidate pairs, verified **85** pairs with **66,084** inliers, and selected
  seed `(2,3)` (`DSC_0288`–`DSC_0289`).  Growth registered `DSC_0290` through
  `DSC_0305`; the transition diagnostics were `DSC_0296`: **129→108** PnP
  inliers and `DSC_0297`: **52→40**.  Post-refinement then added
  `DSC_0306` (**28→25**), `DSC_0307` (**15→15**), and `DSC_0308`
  (**13→12**).  The final model reached **23/38** images with **16,192**
  tracks and **0.324 px** mean reprojection error, in **4:24.27** with
  **813,192 kB** peak RSS; its partial 23-camera laser-GT score was
  **4.51 cm RMSE** (not a full-scene result).  The exact debug log is
  `/tmp/eth3d_courtyard_highres_cap8192_window3_20260830.log`.  The focused
  ratio-`0.9` diagnostic was not run because this cap did not stall at
  `DSC_0297`; it stalled after the 23-camera post-refinement result instead.

- **High-resolution N=4 window control (2026-08-30).** Re-running the
  cap-8,192 artifact with the same per-image calibration, ratio `0.8` +
  cross-check, full verifier, `pnp-max-iterations=100000`, recovery,
  post-registration, and final refinement retained **142** candidate pairs,
  verified **102** pairs with **70,369** inliers, and again registered only
  through `DSC_0308` (**23/38** images).  The model contained **16,241**
  tracks with **0.333 px** mean reprojection error; runtime was **5:38.68**
  and peak RSS **812,908 kB**.  Its partial 23-camera laser-GT score was
  **6.54 cm RMSE**, worse than the N=3 cap-8,192 control's **4.51 cm**.
  Debug growth selected the same `(2,3)` seed; at the final stage image index
  23 (`DSC_0309`) had **0** usable 2D–3D correspondences, despite the direct
  N=4 pair `DSC_0308–DSC_0309` having **310 raw / 236 accepted** matches.
  Newly admitted gap-4 edges around this boundary included
  `DSC_0305–DSC_0309` (**43/0**), `DSC_0306–DSC_0310` (**36/0**),
  `DSC_0307–DSC_0311` (**50/0**), `DSC_0308–DSC_0312` (**42/0**), and
  `DSC_0309–DSC_0313` (**121/79**, raw/accepted), so the wider window did not
  provide a reliable new registration bridge.  The focused ratio-`0.9`
  diagnostic over the 22 pairs internal to `DSC_0306`–`DSC_0313` found
  **14,578 raw / 9,443 accepted** matches across **14/22** verified pairs;
  `DSC_0308–DSC_0309` rose to **646/333** and `DSC_0309–DSC_0313` to
  **336/126**, but several adjacent pairs remained degenerate and the
  existing ratio-`0.8` edge already had substantial pair support while
  yielding zero 2D–3D tracks.  No full ratio-`0.9` reconstruction was
  launched; the evidence points to track/PnP admission rather than raw pair
  scarcity.  Logs are under `/tmp/eth3d_courtyard_highres_cap8192_window4_20260830.log`
  and `/tmp/eth3d_courtyard_highres_cap8192_window4_0308_0313_pairs_diagnose_20260830.log`.

- **High-resolution orientation-locus canonicalization control
  (2026-08-30).** On the cap-8,192 N=3 stack, `DSC_0308` contained 8,192
  rows but only **7,168** physical `(x,y,scale)` loci (**1,024** collapsed
  orientation rows; 1,022 loci had multiplicity >1).  The canonicalized run
  retained the same **85/108** verified pairs and **66,062** inliers versus
  66,084 in the no-canonicalization control, then reduced the accepted stream
  from 66,062 to **59,039** (**7,023** deterministic locus-pair
  deduplications across 82 pairs).  Around the stalled boundary, accepted
  counts changed from the control's `0306–0308=37` to **34**,
  `0307–0308=69` to **67**, `0308–0309=236` to **233**, and
  `0308–0310=41` to **40**; these pair-level reductions are the measurable
  orientation-alternative collisions (the snapshot codec cannot be combined
  with per-image calibration, so raw pre-canonical row indices were not
  exportable without changing the established path).  The resulting model
  stayed at **23/38** registered images, with **14,193** tracks and **0.331
  px** mean reprojection error; its partial 23-camera laser-GT score improved
  from the control's **4.51 cm** to **3.53 cm RMSE**.  Track topology did not
  gain 3-view support: global length-3-or-longer tracks changed **6,028→5,365**
  (length-3 exactly **2,715→2,370**), and `DSC_0308` had **20→17** such tracks;
  `DSC_0309` remained unregistered.  The canonicalization log is
  `/tmp/eth3d_courtyard_highres_cap8192_window3_locus_20260830.log`.

- **High-resolution `DSC_0308` transition endpoint audit (2026-08-30).** A
  one-shot `VISLOC_SFM_DEBUG_DUMP_MATCH_INDICES=1` replay of the unchanged
  cap-8,192/N=3 control captured accepted feature indices for the transition:
  `0305–0308=0`, `0306–0308=37`, `0307–0308=69`, and `0308–0309=236`.  The
  registered-side `DSC_0308` set therefore had **89** unique rows (17
  duplicate rows across the two incident pairs), while the `0309`-side set had
  **236** rows; their exact row-index intersection was **0**.  Deterministic
  nearest-neighbor matching between these same-image sets had a minimum xy
  separation of **724.84 px** (A→B median **2,363 px**; B→A median
  **2,869 px**), and the one-to-one candidate count was **0** at both 0.5 px
  and 1.0 px with scale ratio ≤1.25.  As an independent high-resolution
  COLMAP-model diagnostic, nearest observation distances were median **83.42
  px** for the registered-side set (2/89 ≤0.5 px) versus **0.301 px** for the
  `0309` set (143/236 ≤0.5 px); among positive-3D observations within 1 px,
  the two sets shared **0** COLMAP point IDs.  The endpoint populations are
  spatially distinct rather than orientation rows of one locus, so no
  default-off spatial-locus stitching was added; this transition remains a
  descriptor/match-track recall gap.  The raw diagnostic log is
  `/tmp/eth3d_courtyard_highres_cap8192_window3_endpoint_control_20260830.log`.

- **Opt-in sequence relative-pose registration fallback (2026-08-30).** Added
  `--sequence-relative-pose-fallback`, default off, to plain incremental.  It
  validates unique numeric stems, requires an immediately consecutive posed
  predecessor, uses a hardened E-derived pose over normalized-Sampson-supported
  rows with cheirality/parallax gates, and obtains scale from the MAD-filtered
  median of up to three recent consecutive steps; accepted poses retriangulate
  before later BA.  The flag also appends missing consecutive retrieval
  candidates only when enabled.  In the cap-8,192 high-resolution N=3 run all
  37 consecutive edges were already among the 101 retrieval candidates, no
  fallback pose passed the gates, and the result remained **23/38**, **16,192**
  tracks / **46,393** observations / **0.324 px**, partial laser-GT **4.51 cm
  RMSE**, **4:24.21** runtime, and **817,284 kB** peak RSS; the log is
  `/tmp/eth3d_courtyard_highres_cap8192_window3_seqfallback4_20260830.log`.
  In the low-resolution lossless-snapshot A/B, control and enabled runs were
  both **38/38**, **20,777/72,431/0.282 px**, **3.422 cm RMSE**, and produced
  identical camera/image/point SHA-256 outputs.  This is an opt-in structural
  completion diagnostic, not an accuracy-champion update; no fallback stem
  was accepted.

- **Sequence fallback F→E rejection diagnosis (2026-08-30).** The high-resolution
  cap-8,192/N=3 pair probe showed why the direct essential estimate is unsafe at
  the first missing bridge: `DSC_0308–0309` had **310 raw / 236 accepted**
  matches and direct-E **60** inliers with p10/p25/median triangulation angles
  **6.911/6.974/6.984°**, yet its GT-only relative rotation/translation-direction
  error was **11.70°/160.52°**. Reprojecting the same F winner through the
  known per-image calibration gave **236 E_F inliers**, **1.000** F-overlap,
  calibrated-F residual **0.597878 px**, E_F-on-F normalized residual
  **0.00018501**, cheirality **235/236** (second solution 1), p10/p25/median
  angles **2.475/2.531/2.574°**, three stable refits, projection distortion
  **0.000331**, singular-value mismatch **0.000661**, and pose spread
  **0.571° rotation / 7.952° translation**; the GT-only E_F error was
  **0.05°/0.98°**. Thus only the existing conservative **5°** global
  translation-spread gate rejected this otherwise geometrically sound F→E
  candidate. The neighboring controls were not safe to loosen: `0306–0307`
  had **444/319** raw/accepted, direct E **158** inliers and F→E **317**
  inliers but **16.06°** translation spread (GT direct/F→E
  **2.03°/40.74°** vs **0.15°/1.16°**), while `0307–0308` had only **14**
  direct-E inliers, cheirality **8/14** and p25 angle **0.023°**, and F→E
  retained **5/69** matches (invalid quality; GT direct **10.40°/69.40°**).
  `0309–0310` had F→E translation spread **49.03°** and remains rejected.
  An opt-in sequence-only promotion therefore retains all other strict gates,
  uses the sequential-SfM **10°** translation-spread bound, and leaves the
  ordinary **5°** gate/default graph unchanged. On the exact cap-8,192/N=3
  run it promoted **5** stable F→E edges and accepted sequence poses for
  `DSC_0306` (**180** inliers, **156** triangulated, scale **1.706350**) and
  `DSC_0309` (**236**, **233**, scale **1.623455**), reaching **24/38** vs
  the no-promotion **23/38** control. The output had **16,427 tracks / 46,865
  observations / 0.325 px**, partial 24-camera laser-GT **4.39 cm RMSE**,
  **4:27.66** runtime, and **816,376 kB** peak RSS. This remains an opt-in
  structural diagnostic rather than an accuracy-champion update; later growth
  still stopped at `DSC_0310` (image 24, **2 < 6** usable PnP correspondences).
  Log/model: `/tmp/eth3d_courtyard_highres_cap8192_window3_seqfallback_f2e_20260830.log`.

- **Sequence fallback non-immediate-neighbor audit (2026-08-30).** After the
  opt-in run reached `DSC_0309`, the three registered-neighbor candidates for
  `DSC_0310` were checked with the same raw/F/E/F→E and sequence stability
  diagnostics.  `DSC_0307–0310` (21,24) had only **37** raw matches at the
  production ratio-0.8/cross-check setting and was `DEGENERATE` (**0** E/F/H
  inliers); ratio **0.9** and **0.95** cross-check remained degenerate as well.
  The exploratory ratio-0.95/no-cross-check profile had **3,387** raw,
  **142 F / 132 E** inliers, but is not the production candidate and was not
  promoted.  `DSC_0308–0310` (22,24) had **88 raw / 41 F** accepted inliers
  (**0 E / 0 H**); its F→E probe retained **38 F / 0 E** rows, with
  projection distortion **0.043776**, singular-value mismatch **0.087636**,
  F-to-E normalized-residual ratio **43.744628**, no cheirality solution, and
  **1.108° rotation / 19.177° translation** refit spread.  It therefore has
  no valid composed pose or scale estimate.  The immediate
  `DSC_0309–0310` (23,24) edge had **624 raw / 556 F / 318 E / 0 H**
  inliers; F→E retained **555/556** F rows (**0.998201** overlap), with
  **555/555** cheirality, direct-E p10/p25/median parallax
  **0.150/0.246/0.944°**, F→E **1.047/1.528/1.906°**, but
  **2.955° rotation / 49.094° translation** refit spread.  Although its
  GT-only direct/F→E relative errors were **2.583°/22.754°** and
  **0.136°/1.384°** (rotation/translation direction), the 49° spread fails
  the sequence-only **10°** gate, so it is not a safe fallback edge.  The
  already accepted fallback scales remain **1.706350** (`DSC_0306`) and
  **1.623455** (`DSC_0309`), from the recent consecutive-step history; no
  composed scale/pose was produced for `DSC_0310`.  Thus no non-immediate
  neighbor passed the existing stability gates, and no neighbor-selection or
  stem-gap scaling generalization was justified.  The exhaustive quality log
  is `/tmp/diag_seq_neighbors_exhaustive_quality_20260830.log`; this is a
  negative structural diagnostic, not an accuracy-champion update.

- **Sequence-only high-support F→E spread override (2026-08-30).** Added a
  conservative exception inside the existing default-off
  `--sequence-relative-pose-fallback` path.  It can ignore only the
  sequence translation-spread bound when all other F→E gates still pass and
  the candidate has at least **100 F and E inliers**, F/E overlap **≥0.95**,
  positive-depth ratio **≥0.95**, cheirality winner margin **≥0.75**, p25
  triangulation angle **≥1°**, finite essential-manifold/residual diagnostics,
  at least two stable refits, and the existing **5°** rotation-spread limit.
  The ordinary global 5° gate and sequence 10° gate are unchanged; the
  override is reported per edge under `VISLOC_SFM_DEBUG`.  On the exact
  cap-8,192/N=3 high-resolution run it accepted **7** F→E promotions: **5**
  ordinary stable edges and two spread overrides,
  `DSC_0306–0307` (**319 F / 317 E**, overlap **0.990596**, cheirality
  **0.996845**, p25 angle **3.840761°**, spread **1.939°/16.058°**) and
  `DSC_0309–0310` (**556/555**, overlap **0.998201**, cheirality **1.0**,
  p25 **1.527653°**, spread **2.955°/49.094°**, rotation/translation).
  The mapper still accepted sequence poses only for `DSC_0306` and
  `DSC_0309` (scales **1.706350** and **1.623455**); `DSC_0310` remained at
  **2 < 6** usable PnP correspondences.  The final result was unchanged at
  **24/38**, **16,427 tracks / 46,865 observations / 0.325 px**, partial
  laser-GT **4.39 cm RMSE**, **4:20.16** runtime, and **817,576 kB** peak
  RSS.  Thus the strong-support override did not unlock the next image and
  is retained as an opt-in structural diagnostic only.  The low-resolution
  lossless snapshot control and sequence-enabled A/B remained byte-identical
  (**38/38**, **20,086 tracks / 66,894 observations / 0.281 px**, **3.422 cm**
  laser-GT RMSE; identical camera/image/point hashes).  Focused override tests,
  release check, rustfmt check, and `git diff --check` passed.  Logs/models:
  `/tmp/eth3d_courtyard_highres_cap8192_window3_seqfallback_f2e_override_20260830`
  and `/tmp/seq_override_snapshot_{control,enabled}_20260830_v2`.

- **Sequence fallback gate-trace for promoted `DSC_0309–0310` (2026-08-30).**
  Added `VISLOC_SFM_DEBUG`-only rejection records around the opt-in sequence
  fallback's pair lookup/model availability, predecessor registration, scale
  history/MAD, stored-E recovery, cheirality, pose composition, and
  triangulation/admission gates.  The promotion is not diagnostic-only: the
  trace consumed the same mutable `PairwiseMatches` entry after promotion,
  showing `image=24 stem=310 pair=53 model=stored_essential direction=23-24`
  with **555** selected E_F-supported matches.  Both fallback attempts had a
  valid scale history and stored finite E_F: before final refinement the
  scale was **1.623455**, hardened cheirality was **504/555**, and
  triangulation admitted **221/555**; after refinement the scale was
  **1.571497**, with the same **504/555** and **221/555**.  The candidate was
  rejected only at the provisional triangulation support gate because 221 is
  below the required half-support **278** (and not because of lookup,
  missing-E, scale, pose-finiteness, or cheirality failure).  Consequently
  `DSC_0310` remained at **2 < 6** PnP correspondences and the fresh
  cap-8,192/N=3 run remained **24/38**, **16,427 tracks / 46,865 observations /
  0.325 px**, partial laser-GT **4.39 cm RMSE**, **4:17.08**, peak RSS
  **817,152 kB**.  The low-resolution snapshot control/enabled rerun stayed
  byte-identical at **38/38**, **20,086 / 66,894 / 0.281 px**, and **3.422 cm**
  RMSE (camera/image/point hashes unchanged).  No fallback threshold or
  plumbing change was made; detailed trace:
  `/tmp/eth3d_courtyard_highres_cap8192_window3_seqfallback_f2e_trace_20260830.log`.

- **ETH3D courtyard original-resolution source audit (2026-08-30).** The
  filesystem cache already contains the official-style extracted
  `courtyard_dslr_undistorted` subset at
  `/home/sasaki/datasets/eth3d/courtyard`: all **38** files
  `DSC_0286.JPG`–`DSC_0323.JPG`, **500,762,181** image bytes, with dimensions
  **6205×4135 (23)**, **6208×4134 (12)**, **6200×4134 (2)**, and **6198×4129
  (1)**.  Every JPEG dimension agrees with the camera ID referenced by the
  local `dslr_calibration_undistorted/images.txt`; `cameras.txt` contains the
  four corresponding undistorted `PINHOLE` blocks (SHA-256
  `0cf9d1f1615b89eed8e48a92ef0ee44352f0fbadbb1be0c00b12f15d01c1f83c`).  The
  official datasets page lists the same courtyard as 38 images and its
  `courtyard_dslr_undistorted.7z` archive as 0.5 GB; a read-only HEAD to
  `https://www.eth3d.net/data/courtyard_dslr_undistorted.7z` returned
  `Content-Length: 500990569` and the duplicate download was skipped.  The
  official evaluation archive is `courtyard_dslr_scan_eval.7z` (listed as
  0.2 GB; a read-only HEAD returned `Content-Length: 182151847`); it was not
  duplicated or extracted because no calibration-faithful full-resolution
  run could be scored.  The local `images.txt`/`points3D.txt` are the
  COLMAP-derived calibration/triangulation package, not a replacement for
  the separate laser-evaluation archive.
  The official [documentation](https://www.eth3d.net/documentation) identifies
  these as pre-undistorted JPEGs with COLMAP-format calibration, while the
  [dataset page](https://www.eth3d.net/datasets) supplies the archive and
  ground-truth links; the [ETH3D license](https://www.eth3d.net/) is CC
  BY-NC-SA 4.0.  The true COLMAP binary is unavailable in this environment.
  Scaling each official block anisotropically to the existing 1600×1066
  working size gives, for the dominant camera-1 block,
  `(fx,fy,cx,cy)=(879.182595,878.951158,803.264464,532.285896)` (the other
  blocks give `(878.440722,878.998742,802.757732,534.010818)`,
  `(879.331613,878.813082,803.310968,532.638994)`, and
  `(880.650532,880.378135,804.574379,532.488816)`), explaining but not
  replacing the current approximate shared camera.
  The demo accepts one global `Camera`, whereas the source calibration has
  four image-size/intrinsic blocks, so no exact full-resolution reconstruction
  was claimed.  The exact bounded compatible-SIFT probe command was
  `RAYON_NUM_THREADS=1 target/release/examples/unordered_sfm_demo
  --feature-extractor sift --images-dir
  /tmp/eth3d_courtyard_highres_probe_single2/images --width 6205 --height
  4135 --fx 3409.58 --fy 3409.44 --cx 3115.16 --cy 2064.73
  --sift-max-keypoints 8192 --sift-vlfeat-compatible-detector
  --sift-vlfeat-compatible-descriptor --sift-vlfeat-bilinear-orientations
  --sift-vlfeat-compatible-output-order --export-features-dir
  /tmp/eth3d_courtyard_highres_probe_single2/features --export-features-only
  --out-colmap /tmp/eth3d_courtyard_highres_probe_single2/out`.
  A bounded probe using the dominant block
  (`6205×4135`, `fx=3409.58`, `fy=3409.44`, `cx=3115.16`, `cy=2064.73`) and
  `RAYON_NUM_THREADS=1` took **80.6 s / 76.8 s** for `DSC_0305/0306`, emitted
  **13,038 / 12,576** rows, and used about **10 GB RSS** per image; the
  parallel two-image probe reached **19 GB RSS**.  Thus a 38-image
  8192-keypoint run (and its exhaustive NN matching) is not a bounded,
  calibration-faithful experiment on this 30-GiB host.  Probe artifacts and
  manifest are under `/tmp/eth3d_courtyard_highres_probe_single2` and
  `/tmp/eth3d_courtyard_highres_manifest_20260830`; the current 1600×1066
  data was not modified and no high-resolution Sim(3) result is asserted.

- **Exact-descriptor length-2 NN audit (2026-08-30).** Recomputed forward
  and reverse best/second-best L2 distances directly from the immutable
  COLMAP feature files and mapped them to the split/merge model's **10,989**
  length-2 tracks.  **10,920** had a direct accepted snapshot edge
  (**1,093** oracle-positive / **9,827** negative); the remaining 69
  transitive pairs included **40** oracle positives and were kept separate.
  On the direct subset, forward Lowe-ratio AUC/AP were only **0.518 / 0.104**,
  reverse ratio **0.507 / 0.102**, mutual-margin **0.502 / 0.100**, and
  pair-distribution-normalized absolute distance **0.489 / 0.095**.  The
  strict matcher-like cut (`ratio <= 0.8`, mutual best, and reverse ratio)
  retained **10,865 / 1,089** pairs (**10.02% precision / 99.63% recall**);
  at ratio cutoffs 0.6/0.7/0.8/0.9 its precision/recall were respectively
  **10.05/50.32%, 10.03/72.46%, 10.02/99.63%, 10.00/99.91%**.  Adding the
  only useful graph signal, `cycle >= 1`, raised the 0.8-cut subset to
  **78.72% precision / 24.70% recall**, so it discards most true candidates.
  Exact ratios/mutual margins were absent from snapshot v1 and are not
  propagated into tracks; no schema change or final length-2 gate was added.
  The descriptor-level result confirms that a safe source-standard rule is
  not supported by the available data.

- **Length-2 track quality audit (2026-08-30).** An oracle-only audit of
  `/tmp/pose_merge_2px_gate4_selective2_8_9_20260830` labeled its **10,989**
  length-2 tracks against the valid COLMAP point-track membership: **1,133**
  pairs were true matches and **9,856** were false, while the COLMAP model
  contains **1,564** length-2 tracks.  Cycle support was the only useful
  discriminator (AUC **0.631**); the standard `cycle >= 1` rule retained 407
  candidates at **75.9% precision / 27.3% recall**.  Other available signals
  were effectively non-separating (angle AUC **0.531**, descriptor L2 **0.482**,
  accepted pair support **0.410**, final reprojection **0.464**); for example,
  angle >=2 degrees retained essentially every candidate at only **10.3%**
  precision, and reprojection <=1 px gave **10.1% / 94.4%** precision/recall.
  The lossless snapshot/features do not retain exact Lowe ratios, mutual-NN
  margins, scale, or orientation, and 69 transitive pairs without a direct
  snapshot edge include 40 oracle positives.  No default-off final length-2
  gate was added: available source-standard thresholds cannot retain most
  true candidates without leaving the false population, so further topology
  or metadata evidence is required.

- **Pose-guided post-split track merging (2026-08-30).** Added the
  default-off `--pose-guided-track-merging` pass, available only with
  `--pose-guided-track-splitting`.  It groups only verified cross-track edges,
  rejects any same-image overlap, refits the complete union at the fixed
  poses, and requires finite cheirality-valid reprojections within the
  split-only gate; accepted unions are ranked by support, independent image
  pairs, parallax, robust reprojection, and stable observation order, with
  geometry recomputed after every union.  Unit coverage includes complementary
  fragments, geometric false-edge rejection, same-image conflict rejection,
  transitive recomputation, permutation invariance, and default-off identity.
  On the immutable COLMAP snapshot with seed `(8,9)`, the 2 px no-bridge
  split/recovery stack tested **857** active cross-track groups and accepted
  **0** merges.  It therefore retained **21,819 tracks / 76,211 observations**
  with length histogram `{2: 11,027, 3: 4,426, 4: 1,970, 5: 1,202, 6: 804,
  7: 571, 8: 524, 9: 447, 10: 283, 11: 188, 12: 169, 13: 86, 14: 36,
  15: 27, 16: 19, 17: 24, 18: 16}`, **0.256477 px** mean reprojection,
  38/38 registration, and **2.71 cm** laser-GT RMSE.  Against the valid
  COLMAP point-track partition (12,983 tracks / 145,204 observation pairs),
  pair precision/recall remained **84.415% / 91.854%**, identical to the
  no-merge 2 px control.  The known-pose oracle diagnostic likewise tested
  **1,086** groups, accepted **0**, and retained **23,409 / 81,474** tracks /
  observations, **0.256250 px**, **1.20 cm** (38/38), and **84.823% /
  97.860%** partition precision/recall.  Since the one-edge mode produced no
  geometrically valid candidate, no two-edge variant was added; reducing the
  length-2 population requires a new source of valid complementary edges.
  Commands/models/logs are `/tmp/pose_merge_2px_8_9_20260830_v2` and
  `/tmp/pose_merge_2px_oracle_20260830`.

- **Separate post-split merge reprojection gate (2026-08-30).** Added the
  default-off `--pose-guided-merge-max-reproj PX` option.  An omitted value
  inherits the 2 px split gate; an explicit value gates only candidate-union
  fitting.  After BA, only merged tracks are checked against the ordinary
  4 px hard threshold; failing unions restore their exact source fragments
  and trigger one deterministic retry BA.  A retry failure or an overall
  support/objective failure still rolls back the complete candidate.  With
  split=2 px and merge=4 px on the immutable snapshot, the current-pose seed
  `(8,9)` produced **866** candidates, **31** proposed, **31** good, and
  **0** restored merges; the accepted output has **21,788 tracks / 76,211
  observations / 0.259 px / 2.70 cm** at 38/38, with length histogram
  `{2: 10,989, 3: 4,418, 4: 1,975, 5: 1,202, 6: 805, 7: 574, 8: 528,
  9: 448, 10: 284, 11: 187, 12: 169, 13: 87, 14: 36, 15: 27, 16: 19,
  17: 24, 18: 16}` and partition precision/recall **84.396% / 91.986%**.
  The known-pose oracle tested **1,097** candidates, accepted **38** good
  unions, restored **0**, and yielded **23,371 / 81,474 / 0.259 px / 1.20
  cm** at 38/38 with partition precision/recall **84.811% / 98.076%**.
  No merge required the retry in these runs, so no current union was
  restored; the selective path avoids rejecting healthy merges because of an
  unrelated split-track residual.  No additional threshold sweep was run.
  Models/dumps/logs are under
  `/tmp/pose_merge_2px_gate4_selective2_{8_9_20260830,oracle_20260830}`.

- **Final minimum track-length gate (2026-08-30).** Added the default-off
  `--final-min-track-length 3` gate.  It runs only after registration and all
  configured recovery/splitting passes, removes length-2 landmarks while
  leaving growth/PnP history untouched, re-triangulates the remaining tracks,
  and runs a guarded BA; solver errors, non-finite state, registration/support
  loss, or a non-improving BA objective restore the complete pre-gate state.
  The initial track-length audit on the immutable COLMAP snapshot showed
  length-2 fractions of **10,583/20,777 = 50.936%** for legacy,
  **11,089/21,501 = 51.574%** for pose-split 1 px,
  **11,062/21,849 = 50.629%** for pose-split 2 px with bridge cuts, versus
  **1,564/13,379 = 11.690%** in the actual COLMAP membership model.  With
  seed `(8,9)`, the legacy gate retained 38/38 and produced **10,194
  tracks / 51,265 observations / 0.289 px / 3.40 cm** laser-GT RMSE versus
  the no-gate **20,777 / 72,431 / 0.282 px / 3.42 cm** control.  The 2 px
  no-bridge pose split retained 38/38 with **10,792 / 54,157 / 0.270 px /
  2.76 cm**, versus **21,819 / 76,211 / 0.256 px / 2.707 cm**; bridge-cut
  pose split gave **10,787 / 54,117 / 0.269 px / 2.757 cm**, versus
  **21,849 / 76,241 / 0.256 px / 2.704 cm**.  An actual-COLMAP-pose seeded
  diagnostic gave **9,827 / 47,332 / 0.296 px / 1.30 cm**; this is an oracle
  control, not a production result.  The legacy improvement is marginal and
  both pose-split variants regress, so the gate remains opt-in and is not an
  accuracy champion.  CLI validation accepts only the first source-motivated
  value `3`; unit tests cover default identity, short-track removal,
  length-3 preservation, support validation, determinism, and incompatible
  `--no-final-ba` usage.  Commands/logs/models are under
  `/tmp/seed_final_min3_{legacy_8_9,pose2_nobridge_8_9,pose2_bridge_8_9,oracle}_20260830`.

- **Pose-split bridge-cut refinement (2026-08-30).** Added the default-off
  `--pose-guided-track-splitting-bridge-cuts` subflag.  Before the existing
  pose-guided splitter, an iterative deterministic Tarjan traversal considers
  only graph bridges whose two sides each contain at least two distinct posed
  images and independently fit one finite point under the existing split
  reprojection gate; the bridge is cut only when the combined observations do
  not fit one point under that same gate.  Singleton/invalid sides and genuine
  sparse chains are retained.  The resulting sides then use the unchanged
  pose-guided split/rollback path.  Synthetic tests cover a false bridge,
  valid sparse chain, singleton side, permutation invariance, and default-off
  behavior.  On the immutable COLMAP snapshot with seed `(8,9)`, the
  recovery→split pre-entry was **72,431 observations / 0.281787 px**.  At 1.0
  px, **153 bridges in 153 components** were cut (side sizes 2–17), yielding
  **21,501 tracks / 74,240 observations / 0.209641 px / 2.719 cm** laser-GT
  RMSE and **84.466% precision / 88.228% recall**; the no-cut composition was
  **21,461 / 74,207 / 0.209750 px / 2.722 cm**, **84.459% / 88.325%**.  At
  2.0 px, **133 bridges in 133 components** were cut (side sizes 2–17),
  yielding **21,849 / 76,241 / 0.256369 px / 2.704 cm** and **84.407% /
  91.788%**, versus no-cut **21,819 / 76,211 / 0.256477 px / 2.707 cm** and
  **84.415% / 91.854%**.  Both retained 38/38; the precision/laser gains are
  only marginal and recall decreases, so this remains an opt-in diagnostic,
  not an accuracy-champion update.  Logs/models are under
  `/tmp/seed_bridge_cut_8_9_{1px,2px}_20260830`.

- **Recovery→pose-split composition (2026-08-30).** The default-off
  `--geometry-guided-conflict-recovery` and
  `--pose-guided-track-splitting` flags may now be composed on the plain
  incremental/union-find path.  The mapper snapshots clean and conflicting
  components before recovery, lets recovery/post/final refinement finish, and
  then rebuilds the split candidate from that immutable original snapshot
  (never recursively from recovered tracks), followed by one guarded BA.  The
  CLI label and parser test cover the composition; a unit test verifies source
  ownership, imported-membership exclusion, default-off behavior, and
  deterministic rollback semantics.  On snapshot
  `/tmp/snapshot_colmap_verified_20260830.vps` with seed `(8,9)`,
  `--min-pnp-inliers 8 --pnp-max-iterations 100000
  --geometry-guided-conflict-recovery --post-refinement-registration
  --final-iterative-refinement`, the recovery-only control was **38/38**,
  **20,777 tracks / 72,431 observations / 0.282 px / 3.42 cm** laser-GT
  RMSE.  The post-recovery split entry was **72,431 observations / 0.281787
  px**; the 1.0 px composition accepted **21,461 tracks / 74,207
  observations / 0.210 px / 2.72 cm**, with partition **84.459% precision /
  88.325% recall** and guarded objective **0.281787→0.217769→0.209750 px**.
  The 2.0 px composition accepted **21,819 / 76,211 / 0.256 px / 2.71 cm**,
  with **84.415% / 91.854%** partition precision/recall and objective
  **0.281787→0.262582→0.256477 px**; both retained 38/38.  These are opt-in
  diagnostic results, not a default or accuracy-champion change.  Exact
  outputs/logs are `/tmp/seed_recovery_8_9_20260830`,
  `/tmp/seed_comp_8_9_{1px,2px}_20260830`; the 1.0/2.0 px effective-config
  hashes were `7ca6c9d82954842b`/`dabca888c9b6c891`.  The common invocation
  was `target/release/examples/unordered_sfm_demo --feature-extractor files
  --features-dir /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_features_export
  --feature-suffix _features.txt --image-suffix .png --width 1600 --height
  1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6
  --import-verified-pairs-snapshot /tmp/snapshot_colmap_verified_20260830.vps
  --min-matches 20 --mapper incremental --pnp-max-iterations 100000
  --min-pnp-inliers 8 --geometry-guided-conflict-recovery
  --post-refinement-registration --final-iterative-refinement
  --pose-guided-track-splitting --pose-guided-split-max-reproj {1.0,2.0}
  --pose-guided-track-splitting-iterations 1 --seed-pair 8,9 --out-colmap
  /tmp/seed_comp_8_9_{1px,2px}_20260830` (with separate runs/thresholds).

- **Opt-in pose-guided track splitting (2026-08-30).** Added
  `--pose-guided-track-splitting` for a complete, posed model: it revisits
  legacy union components (including the **803** same-image-conflict
  components), proposes deterministic wide-baseline 3-D hypotheses, keeps at
  most one observation per image under cheirality/reprojection gates, locally
  refines fixed-pose points, and validates one rebuilt partition with a
  rollback guard.  The default union-find path is unchanged; the flag rejects
  alternate track strategies and leaves incomplete pose models unchanged.  A synthetic
  two-point/conflicting-edge fixture verifies splitting, outlier pruning,
  permutation determinism, and default-off behavior.  On the immutable COLMAP
  snapshot (`/tmp/snapshot_colmap_verified_20260830.vps`, 380 pairs / 140,445
  accepted edges), the ordinary control was **38/38**, **20,086 tracks /
  66,894 observations**, **0.281 px**, **3.42 cm** laser-GT RMSE.  The
  current-pose split accepted **21,954 tracks / 76,885 observations**, **0.289
  px**, and **3.17 cm** (`/tmp/snapshot_colmap_pose_split_current_20260830e`),
  with split diagnostics `components=25,306, preserved=20,085, split=1,081,
  hypotheses=27,223, discarded_obs=7,633`; this is a modest opt-in result,
  not an accuracy-champion update.  The candidate observation partition had
  **84.19% pair precision / 92.65% recall** against valid COLMAP point tracks
  (145,204 reference pairs), so substantial false merges remain.  With all
  COLMAP camera poses supplied only as an oracle diagnostic (intrinsics matched
  in a temporary `/tmp` camera header), the split was **38/38**, **23,580 /
  82,274**, **0.291 px**, **1.45 cm**, versus **1.44 cm** for the no-split
  oracle-pose control; hence the partition does not independently improve the
  known-pose basin.  The exact oracle command used
  `VISLOC_SFM_DEBUG=1 VISLOC_SFM_DEBUG_POSE_SPLIT_DUMP=/tmp/pose_split_oracle_20260830e.tsv target/release/examples/unordered_sfm_demo --feature-extractor files --features-dir /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_features_export --feature-suffix _features.txt --image-suffix .png --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 --import-verified-pairs-snapshot /tmp/snapshot_colmap_verified_20260830.vps --initial-poses /tmp/colmap_oracle_pose_seed_cam8794_20260830/images.txt --no-final-ba --periodic-ba-min-registered-images 999 --min-matches 20 --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8 --pose-guided-track-splitting --out-colmap /tmp/snapshot_colmap_pose_split_oracle_20260830e`.

- **Pose-guided graph-support admission (2026-08-30).** Added the separate
  default-off `--pose-guided-track-splitting-graph-support` subflag.  After a
  two-view anchor, each added observation must have direct verified edges to
  two distinct images already in the hypothesis; multi-view emissions also
  require two independent cross-image supports, while genuine length-2
  hypotheses remain valid.  The admission order and support histogram are
  deterministic, and the existing pose-guided result is unchanged when the
  subflag is omitted.  Synthetic tests cover a single false bridge versus two
  independent supports, length-2 fallback, permutation invariance, and CLI
  dependency validation.  On the same immutable snapshot and mapper stack,
  the prior pose-split result was **21,954 tracks / 76,885 observations / 0.289
  px / 3.17 cm**; graph support produced **22,347 / 76,299 / 0.280 px / 3.21
  cm**, with `components=25,306, preserved=20,085, split=1,081,
  hypotheses=30,038, discarded_obs=8,219, graph_tracks=1,246,
  graph_len2=1,016, graph_hist=[0,0,1913,1207,723,455,269,317]`.  Candidate
  partition pair precision/recall against valid COLMAP point tracks changed
  from **84.19% / 92.65%** to **85.22% / 89.64%** (reference pair set
  145,204): precision improved only about one percentage point while recall
  and laser-GT Sim(3) regressed slightly, so this remains an opt-in negative
  experiment rather than an accuracy-champion change.  The exact graph run
  used `VISLOC_SFM_DEBUG=1 VISLOC_SFM_DEBUG_POSE_SPLIT_DUMP=/tmp/pose_split_graph_20260830.tsv target/release/examples/unordered_sfm_demo --feature-extractor files --features-dir /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_features_export --feature-suffix _features.txt --image-suffix .png --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 --import-verified-pairs-snapshot /tmp/snapshot_colmap_verified_20260830.vps --min-matches 20 --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8 --post-refinement-registration --final-iterative-refinement --pose-guided-track-splitting --pose-guided-track-splitting-graph-support --out-colmap /tmp/snapshot_colmap_pose_split_graph_20260830`.  The oracle-pose diagnostic was **24,035 tracks / 81,547 observations / 0.282 px / 1.47 cm**, versus the prior oracle-pose split **23,580 / 82,274 / 0.291 px / 1.45 cm**; its graph histogram was `[0,0,2293,1379,830,517,313,365]`, so the small precision gain did not survive the known-pose accuracy check.

- **Pose-guided split-only reprojection gate (2026-08-30).** Added the
  default-off `--pose-guided-split-max-reproj PX` option. `None` reuses the
  ordinary `--max-reproj` value (4.0 px in this experiment), and the gate is
  applied only while selecting/refining pose-guided split observations; the
  ordinary mapper, PnP, and final BA thresholds are unchanged. CLI validation
  rejects non-positive/non-finite values and use without
  `--pose-guided-track-splitting`; the default config remains `None`. On the
  immutable 380-pair COLMAP snapshot with the base pose-split stack, the
  current/4.0 px control was **21,954 tracks / 76,885 observations / 0.289 px /
  3.17 cm**, with candidate partition pair precision/recall **84.187% /
  92.650%**. The only two bounded tests were 1.0 px (**21,461 / 74,207 /
  0.210 px / 2.72 cm**, **84.459% / 88.325%**, `discarded_obs=10,311`) and
  2.0 px (**21,819 / 76,211 / 0.256 px / 2.71 cm**, **84.415% / 91.854%**,
  `discarded_obs=8,307`); all remained 38/38. With the temporary known-pose
  diagnostic, the corresponding results were 1.0 px **22,998 / 79,239 /
  0.208 px / 1.10 cm** and 2.0 px **23,409 / 81,474 / 0.256 px / 1.20 cm**,
  versus the prior 4.0 px oracle split **23,580 / 82,274 / 0.291 px / 1.45
  cm**; these oracle numbers are not a deployable accuracy claim. The
  candidate partition reference was 12,983 valid COLMAP points and 145,204
  observation pairs. Neither threshold loses registration, so no length-2
  fallback was added; the modest score gain remains an opt-in diagnostic, not
  an accuracy-champion update. The exact current-pose 1.0 px command was
  `VISLOC_SFM_DEBUG=1 VISLOC_SFM_DEBUG_POSE_SPLIT_DUMP=/tmp/pose_split_threshold_1px_20260830.tsv target/release/examples/unordered_sfm_demo --feature-extractor files --features-dir /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_features_export --feature-suffix _features.txt --image-suffix .png --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 --import-verified-pairs-snapshot /tmp/snapshot_colmap_verified_20260830.vps --min-matches 20 --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8 --post-refinement-registration --final-iterative-refinement --pose-guided-track-splitting --pose-guided-split-max-reproj 1.0 --out-colmap /tmp/pose_split_threshold_1px_20260830` (the 2.0 px A/B changed only the threshold and output paths).

- **Bounded pose-guided split iterations (2026-08-30).** Added
  `--pose-guided-track-splitting-iterations N` (CLI `1..=8`; omitted means
  one pass when splitting is enabled). Every pass rebuilds from the original
  clean/conflicting components with the latest complete poses, never from the
  previous split output. The first-pass guard is unchanged; later passes also
  require finite reprojection, unchanged registration/support, and strict
  improvement over the already accepted model, otherwise that pass is rolled
  back and iteration stops. Synthetic guards cover N=1 identity, second-pass
  improvement, support/registration rollback, non-finite input, early stop,
  determinism, and CLI validation. On the immutable snapshot with the base
  pose-split stack, N=2 + 1.0 px accepted iteration 1 at **21,461 / 74,207 /
  0.210 px / 2.72 cm** (candidate P/R **84.459% / 88.325%**); iteration 2
  proposed **21,440 / 74,027** and **84.448% / 88.080%**, but was rejected
  because support fell below 74,207 and its candidate mean rose to 0.276 px
  from 0.210 px. N=2 + 2.0 px accepted iteration 1 at **21,819 / 76,211 /
  0.256 px / 2.71 cm** (P/R **84.415% / 91.854%**); iteration 2 proposed
  **21,814 / 76,163** and **84.413% / 91.803%**, and was likewise rejected
  (support below 76,211; candidate mean 0.407 px versus 0.256 px). Both
  remained 38/38, so no N=3 run was justified. The explicit N=1 run and the
  omitted-iterations run produced byte-identical split TSV SHA-256
  `901d6cf035d2176d7f9697f8cae06a06892409489510f69308bca2d0eef4d081`.
  Oracle-pose ceiling controls also stopped after iteration 1: 1.0 px
  **22,998 / 79,239 / 0.208 px / 1.10 cm** and 2.0 px **23,409 / 81,474 /
  0.256 px / 1.20 cm**. These remain opt-in diagnostic results, not an
  accuracy-champion update.

- **Pose-split seed replay (2026-08-30).** Replayed only the four existing
  high-quality seeds `(8,9)`, `(7,8)`, `(7,9)`, and `(10,11)` with the
  immutable COLMAP verified snapshot
  `/tmp/snapshot_colmap_verified_20260830.vps` (SHA-256
  `6511181ac3b099cb9a9c8d7525b1746d28b7d5c7459df27e8460fef27f71f82a`, 380
  pairs / 140,445 accepted edges), `--min-pnp-inliers 8`,
  `--post-refinement-registration`, `--final-iterative-refinement`,
  `--pose-guided-track-splitting --pose-guided-split-max-reproj 1.0
  --pose-guided-track-splitting-iterations 1`.  The common command was the
  snapshot-import command with `--seed-pair I,J` and output paths
  `/tmp/seed_split_{8_9,7_8,7_9,10_11}_1px_20260830`.  All runs reached
  **38/38**.  Before splitting, the complete-pose models had respectively
  **66,894 / 66,609 / 67,462 / 66,722** valid observations and split-entry
  mean reprojection **0.281 / 0.392 / 0.354 / 0.379 px**.  The 1.0 px split
  results in seed order were: `(8,9)` **21,461 tracks / 74,207 observations /
  0.210 px / 2.72 cm**, partition **84.459% precision / 88.325% recall**;
  `(7,8)` **21,490 / 72,369 / 0.248 px / 11.55 cm**, **84.359% / 82.096%**;
  `(7,9)` **21,749 / 73,618 / 0.232 px / 10.37 cm**, **84.339% / 84.184%**;
  `(10,11)` **21,391 / 72,526 / 0.265 px / 16.38 cm**, **84.498% /
  83.462%**.  Final mean reprojection/support ranks `(8,9)` first, then
  `(7,9)`, `(7,8)`, `(10,11)`, matching the laser-GT ranking on this bounded
  set; no seed was near/sub-cm, so no duplicate reproducibility run was
  warranted.  Because `(8,9)` was best but its 1.0 px partition recall was
  only 88.325%, one bounded 2.0 px A/B was run for that seed only: **21,819 /
  76,211 / 0.256 px / 2.71 cm**, **84.415% / 91.854%**, still 38/38.  This is
  a small opt-in diagnostic result, not a default or accuracy-champion
  update.

- **Fixed-rotation pose-split probe (2026-08-30).** A temporary
  `--pose-guided-split-fix-rotations` composition reused the existing
  fixed-rotation BA constraints to hold every registered pose at its
  pre-split champion rotation while validating the split candidate; only
  translations and landmarks were optimized.  On seed `(8,9)` and the same
  snapshot/min8/post/final stack, 1.0 px produced **21,461 tracks / 74,207
  observations / 0.215 px / 3.36 cm**, with **84.459% / 88.325%** partition
  precision/recall, versus free-rotation **0.210 px / 2.72 cm**; 2.0 px
  produced **21,819 / 76,211 / 0.261 px / 3.30 cm**, **84.415% / 91.854%**,
  versus free-rotation **0.256 px / 2.71 cm**.  Both remained 38/38 and the
  fixed candidate BA reported maximum rotation change **3.205e-15 degrees**;
  support and partition membership were unchanged.  Existing actual-COLMAP-
  pose ceiling artifacts re-score **1.10 cm / 1.20 cm** for their 1.0/2.0 px
  oracle split supports (**79,239 / 81,474 observations**), and are not a
  deployable or current-rotation result.  Current-rotation fixing therefore
  fails the quality gate and the temporary CLI/config code was removed; no
  persistent flag or default behavior change is shipped.  Logs/models were
  retained under `/tmp/seed_split_8_9_{1px,2px}_fixrot_20260830`.

- **Lossless verified-pair snapshots (2026-08-30).** Added the explicit,
  default-off `--export-verified-pairs-snapshot PATH` /
  `--import-verified-pairs-snapshot PATH` path in
  `examples/unordered_sfm_demo.rs`.  Schema 1 stores image/feature manifests,
  intrinsics, effective/verifier configuration hashes and text, exact pair and
  raw/accepted/essential correspondence order, inlier indices/counts,
  CALIBRATED flag/configuration, E/F/H matrices, relative pose, ordered and
  unordered edge hashes, and an FNV-1a payload checksum.  Import validates all
  manifests, camera values, checksums, pair/index relationships, and hashes,
  then bypasses matcher/verifier and preserves the stored stream; legacy text
  import is unchanged.  Codec round-trip, exact float-bit/order, checksum,
  schema-version, CLI default, and conflict tests pass.  The authoritative
  COLMAP export command was
  `SSD=/media/sasaki/aiueo1/visloc-rs/eth3d/courtyard; target/release/examples/unordered_sfm_demo --feature-extractor files --features-dir "$SSD/colmap_features_export" --import-matches-file "$SSD/colmap_matches_import.txt" --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 --exhaustive --min-matches 20 --verification-mode full --mapper incremental --out-colmap /tmp/snapshot_colmap_export_model_20260830 --pnp-max-iterations 100000 --final-iterative-refinement --export-verified-pairs-snapshot /tmp/snapshot_colmap_verified_20260830.vps`; the separate import replaced the raw-import/export options with `--import-verified-pairs-snapshot /tmp/snapshot_colmap_verified_20260830.vps` and wrote `/tmp/snapshot_colmap_import_model_20260830b`.
  COLMAP-feature command (raw `colmap_matches_import.txt`, full verification,
  plain incremental, PnP 100k, final iterative) exported
  `/tmp/snapshot_colmap_verified_20260830.vps` (SHA-256
  `6511181ac3b099cb9a9c8d7525b1746d28b7d5c7459df27e8460fef27f71f82a`,
  380 pairs / 140,445 accepted edges, ordered hash
  `d8f83d1c42305fcd`, unordered hash `ee3768f65c0c53b4`).  A separate import
  process reproduced 38/38, 20,086 tracks / 66,894 observations, 0.281 px,
  laser-GT 3.422 cm, and byte-identical `cameras.txt`, `images.txt`, and
  `points3D.txt` (SHA-256 respectively
  `a2132068b1a4dbe21f1ad68a23ff05461026c5a84e0b0de14f06311533e5b958`,
  `d8cd03627eb77459ed8cfe0cffc5e909f54fa81665f9f0fe0eb7726de7e06d61`,
  `97f0abfb05ec706e7d11a4939dc034533db2161d3abeddecfab896f1d271d1fe`).
  The same fresh replay on own-SIFT features used
  `target/release/examples/unordered_sfm_demo --feature-extractor files --features-dir /tmp/vlfeat_source_latest_floor_20260829 --feature-suffix _features.txt --image-suffix .png --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 --exhaustive --min-matches 15 --match-ratio 0.9 --guided-matching --verification-mode full --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8 --geometry-guided-conflict-recovery --post-refinement-registration --final-iterative-refinement`, with export to `/tmp/snapshot_own_verified_20260830.vps` and `/tmp/snapshot_own_export_model_20260830`; the separate import wrote `/tmp/snapshot_own_import_model_20260830b`.
  `/tmp/vlfeat_source_latest_floor_20260829` exported
  `/tmp/snapshot_own_verified_20260830.vps` (SHA-256
  `8293224402c7cd0ddb49f43f9a8f19033194ce2acf09185071df3440b6bdf72b`,
  639 pairs / 154,857 edges, ordered hash `02cd21957f1636ed`, unordered hash
  `a67e3c7680522a7c`) and imported to byte-identical 38/38,
  15,638-track / 52,404-observation, 0.275 px output (laser-GT 61.679 cm).
  The snapshot path is a reproducibility/diagnostic mechanism; it does not
  change default reconstruction behavior or claim an accuracy-champion update.

- **Validated-snapshot coordinate override (2026-08-30).** Added the
  default-off diagnostic `--snapshot-coordinate-override-dir DIR`.  It is only
  accepted with `--import-verified-pairs-snapshot` (and file features), first
  validates the base feature manifest through the snapshot, then requires the
  replacement directory to have identical image-name order, row counts, and
  descriptor f32 bit patterns before copying only keypoint `x,y`; pair order,
  indices, models, and ordered/unordered edge hashes are unchanged.  Focused
  tests cover default-off/dependency validation, coordinate-only replacement,
  exact descriptor-bit checks, and name-order rejection.  The exact common
  mapper command was
  `target/release/examples/unordered_sfm_demo --feature-extractor files --features-dir /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_features_export --import-verified-pairs-snapshot /tmp/snapshot_colmap_verified_20260830.vps [--snapshot-coordinate-override-dir DIR] --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 --exhaustive --min-matches 20 --verification-mode full --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8 --geometry-guided-conflict-recovery --post-refinement-registration --final-iterative-refinement --out-colmap MODEL`.
  The immutable snapshot is SHA-256
  `6511181ac3b099cb9a9c8d7525b1746d28b7d5c7459df27e8460fef27f71f82a` with
  380 pairs / 140,445 accepted correspondences and ordered/unordered hashes
  `d8f83d1c42305fcd` / `ee3768f65c0c53b4`.  A (base coordinates) produced
  38/38, 20,777 tracks / 72,431 observations, 0.282 px, and 3.42 cm laser-GT
  RMSE; B (`/tmp/colmap_features_subpixel3b_20260829`) preserved both hashes
  and 38/38 but produced 20,023 / 70,196, 0.451 px, and 22.24 cm.  The
  existing Rust coordinate artifact
  `/tmp/rust_subpixel3_exactcv3_20260829` passed the descriptor-bit contract
  and produced 38/38, 21,060 / 73,538, 0.337 px, and 4.52 cm.  Model text
  SHA-256 (cameras/images/points3D) was A
  `a2132068b1a4dbe21f1ad68a23ff05461026c5a84e0b0de14f06311533e5b958` /
  `c50fb73ce0199b765c99f1431846c752cc1c7943c7f86b22a2119fddf682c030` /
  `ff6bd0246279744a121c31336d183816ac84866e6b23968cc3ebf928213d9c01`, B
  `a2132068b1a4dbe21f1ad68a23ff05461026c5a84e0b0de14f06311533e5b958` /
  `f200164084bfc2099c75c030fa105e91901da04233ad73daf5f0420b6e54374c` /
  `e6fa8351a24053e02103982bbd98ed209df331124a45d667a7075790c46013b`, and
  Rust `a2132068b1a4dbe21f1ad68a23ff05461026c5a84e0b0de14f06311533e5b958` /
  `1ce78635a010fb69406e485b097ef771e8260e111a0270d6e743d84f801c5032` /
  `996f1ff7fe29366b819d84c88a242006c5a343c0d936d0989e7318940be72df3`.
  Thus neither coordinate variant justifies a production post-snapshot
  refinement; the path remains diagnostic-only and does not alter defaults.

- **Opt-in incremental correspondence triangulation (2026-08-30).** Audited
  the lossless COLMAP 380-pair snapshot before changing the mapper: the
  pre-unioned closure contained **24,503** conflict-free tracks / **80,698**
  observations and discarded **803** same-image-conflict components / **9,896**
  observations, while the actual COLMAP sparse model had **13,379** points /
  **62,448** observations.  Only **10,018** pre-unioned tracks were exact
  observation-set matches to COLMAP points (11,459 had at least two shared
  observations); coverage intersection was **52,946** observations.  The
  actual model's median/p90 track reprojection was **0.432/0.800 px** and
  median/p10/p90 triangulation angle was **15.48/4.86/39.91 deg**.
  Added default-off `--incremental-correspondence-triangulation`: verified
  edges are physically sorted and added through an explicit
  observation-to-track map, same-image-conflicting edges are rejected in
  isolation, and every live point is revisited after registration with a
  guarded widest-baseline re-triangulation.  The plain seed/growth/PnP
  schedule is retained and the CLI rejects `--colmap-style` with this mode;
  create/continue/merge/conflict/retriangulation, permutation, and default
  no-op tests were added.  On the same snapshot
  (`/tmp/snapshot_colmap_verified_20260830.vps`, SHA-256
  `6511181ac3b099cb9a9c8d7525b1746d28b7d5c7459df27e8460fef27f71f82a`), the
  mode retained **25,965** pre-filter tracks / **90,213** observations with
  **2,819** edge-level conflicts, reached **38/38** (growth **37/38**, post
  **+1**), and ended with **20,279 tracks / 68,629 observations / 0.347 px**;
  laser-GT Sim(3) was **13.22 cm**, versus the plain legacy snapshot control's
  **3.42 cm / 20,777 tracks / 72,431 observations / 0.282 px**.  This confirms
  the mode can preserve completeness but its changed topology is a quality
  regression, so it remains experimental and does not update the accuracy
  champion; no own-SIFT run was justified after the COLMAP control failed the
  quality gate.  The actual COLMAP model remains the **1.71 cm** oracle.  The
  reproducible mode command was `target/release/examples/unordered_sfm_demo
  --feature-extractor files --features-dir
  /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_features_export
  --feature-suffix _features.txt --image-suffix .png --width 1600 --height 1066
  --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6
  --import-verified-pairs-snapshot /tmp/snapshot_colmap_verified_20260830.vps
  --min-matches 20 --mapper incremental --pnp-max-iterations 100000
  --min-pnp-inliers 8 --geometry-guided-conflict-recovery
  --post-refinement-registration --final-iterative-refinement
  --incremental-correspondence-triangulation --out-colmap MODEL`; the rebuilt
  binary reproduced the same 13.22 cm result.

- **COLMAP track-membership oracle replay (2026-08-30).** Added the
  default-off `--diagnose-colmap-track-membership MODEL/points3D.txt` path.
  Its sibling `images.txt` maps `IMAGE_ID`/`POINT2D_IDX` to the loaded image
  and feature-row manifest; only observation membership is imported, while
  COLMAP XYZ/color/error and camera poses are ignored.  The parser validates
  image names, `POINTS2D` row counts, bounds, and one observation per image;
  the actual model's 396 source tracks containing same-image duplicates
  (3,871 observations) are skipped explicitly, leaving 12,983 valid tracks /
  58,577 observations from 13,379 points / 62,448 observations.  On the
  immutable verified snapshot `/tmp/snapshot_colmap_verified_20260830.vps`
  (380 pairs / 140,445 accepted correspondences), the oracle membership
  replay used the same plain incremental, PnP-100k, recovery, post-registration,
  and final-iterative stack and produced 38/38, 12,827 tracks / 58,184
  observations, 0.307 px mean reprojection, and 2.01 cm laser-GT Sim(3)
  RMSE (median 1.32 cm).  The same stack with legacy union tracks produced
  38/38, 20,777 / 72,431, 0.282 px, and 3.42 cm; the actual COLMAP sparse
  model is a 1.71 cm oracle.  An exact-overlap control retaining only the 933
  COLMAP points whose complete observation sets matched legacy tracks (3,179
  observations) reached only 11/38, 816 tracks / 2,863 observations, 0.215 px,
  and 3.30 cm on its 11-image subset, showing that the full COLMAP partition,
  not merely a small intersection, is needed for connectivity.  Thus the
  observation partition explains most of the gap, while the remaining error
  is in mapper initialization/BA and support details; default reconstruction
  behavior is unchanged.  Exact
  oracle command: `target/release/examples/unordered_sfm_demo
  --feature-extractor files --features-dir
  /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_features_export
  --feature-suffix _features.txt --image-suffix .png --width 1600 --height 1066
  --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6
  --import-verified-pairs-snapshot /tmp/snapshot_colmap_verified_20260830.vps
  --diagnose-colmap-track-membership
  /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_oracle_full/sparse_txt/points3D.txt
  --min-matches 20 --mapper incremental --pnp-max-iterations 100000
  --min-pnp-inliers 8 --geometry-guided-conflict-recovery
  --post-refinement-registration --final-iterative-refinement
  --out-colmap /tmp/snapshot_colmap_oracle_tracks_20260830`.

- **Post-verification subpixel fixed-stream replay (2026-08-30).** Replayed
  the exact available imported COLMAP verified stream
  (`/media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_verified_import.txt`:
  401/703 pairs, 141,698 correspondences; unordered edge FNV-1a64
  `55009052f64f8520`) while changing only feature coordinates after import;
  all three runs preserved the pair/match stream and hash.  Original
  coordinates produced 19,754 tracks / 68,577 observations, 0.408 px mean
  reprojection, and 23.05 cm laser-GT Sim(3) RMSE; the OpenCV 3x3 artifact
  produced 20,342 / 69,532, 0.405 px, and 24.43 cm; the existing Rust
  cornerSubPix-equivalent artifact produced 19,821 / 67,924, 0.427 px, and
  19.63 cm.  Both refiners moved about 10.8k/208,785 rows by more than 1e-6
  px (mean 0.0329 px); Rust and OpenCV differed by >0.01 px on 778 rows,
  so the Rust coordinates do not track the OpenCV artifact exactly.  With
  actual COLMAP rotations fixed, the same three variants scored 1.95, 2.02,
  and 2.02 cm respectively (all 38/38), so no variant reached a sub-cm
  production-quality result.  The artifacts are COLMAP-feature rows
  (208,785), not the own-SIFT rows (208,746), and therefore cannot be safely
  applied to the own-SIFT index space.  No `--post-verification-subpixel`
  path is shipped; the next bottleneck remains order-sensitive track
  topology and an exact fixed replay of the transient fresh 380-pair stream.

- **Opt-in COLMAP-style guided matching audit (2026-08-30).** Added
  `--colmap-guided-matching` (requires `--guided-matching`) as an append-only
  diagnostic path.  It follows COLMAP's `FindGuidedMatches` model choice
  (calibrated `E`, uncalibrated `F`, planar/panoramic `H`), uses the
  corresponding pixel/normalized Sampson or homography transfer gate, true
  descriptor L2 distance with the `.7·512` distance cap, mutual nearest
  matching, deterministic deduplication, and preserves the pre-guided
  verified matches.  The implementation is compared with the authoritative
  [COLMAP SIFT matcher](https://github.com/colmap/colmap/blob/main/src/colmap/feature/sift.cc);
  that source computes a dense geometry-masked distance matrix rather than a
  separate spatial index, so the compatibility path likewise uses a bounded
  deterministic full candidate scan.  On focused DSC_0305–0307 features at
  ratio 0.9, the added matches were 139/125/167 for pairs 0305–0306,
  0305–0307, and 0306–0307, respectively, but only 11/139 (7.91%),
  6/125 (4.80%), and 14/167 (8.38%) overlapped the mapped COLMAP raw set
  (the corresponding verified overlap was 11/139, 7/125, and 14/167).
  The full reproducible own-SIFT run verified 639/703 pairs and registered
  38/38, but retained 8,633 tracks / 26,680 observations at 0.327 px mean
  reprojection and scored 185.26 cm laser-GT Sim(3) centre RMSE, versus
  31.84 cm for the existing compatible-bilinear control.  Thus guided
  additions did not improve the accuracy champion; the flag remains
  default-off and no normal reconstruction behavior changes.

- **GMS local-neighborhood probe (2026-08-30).** A read-only probe
  (`python3 /tmp/gms_probe.py`) evaluated the own compatible-SIFT feature
  artifact `/tmp/locus_own_features_reverse_20260829` (208,746 rows with
  x/y/scale/orientation sidecars) before any production filter was added.
  It follows the [GMS CVPR 2017 paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Bian_GMS_Grid-based_Motion_CVPR_2017_paper.html)
  and its [reference implementation](https://github.com/JiawangBian/GMS-Feature-Matcher/blob/master/include/gms_matcher.h):
  20x20 normalized grids, four half-cell offsets, `THRESH_FACTOR=6`, the
  prescribed five scale ratios and eight rotation patterns.  The sidecar
  scale/orientation metadata was also measured per retained match; the
  reference GMS scale/rotation switches are global grid hypotheses rather
  than a per-keypoint affine gate.  COLMAP oracle labels used a nearest
  physical endpoint map with a 0.75 px cutoff, and recall denominators count
  only mapped endpoints.  On the four critical pairs, own NN ratio-0.9
  cross-check versus GMS changed raw precision/recall as follows:
  0305-0306 **94.5/72.9% -> 97.1/43.5%**, 0305-0307
  **93.2/62.7% -> 100.0/25.4%**, 0306-0307 **94.5/75.9% ->
  96.7/65.2%**, and 0296-0300 **95.8/75.6% -> 100.0/19.5%**.
  Across the deterministic 12-pair temporal sample, weighted raw
  precision/recall changed **94.3/74.3% -> 95.8/54.5%** (verified recall
  **75.1% -> 59.6%**).  The critical bridge pose was not preserved: for
  0305-0306 the E rotation/translation error changed **0.119/1.420° ->
  1.010/6.229°**, and for 0305-0307 **0.577/3.724° -> 16.951/24.461°**;
  0306-0307 changed **0.448/3.067° -> 0.417/3.260°**.  F-derived diagnostics
  likewise worsened on 0305-0307 and 0306-0307.  Since the small precision
  gain costs substantial bridge recall and harms E/F pose quality, no GMS or
  local-neighborhood pre-verifier is shipped; default matching is unchanged.

- **Numeric stem pair window (2026-08-30).** Added the default-off
  `--pair-stem-window N` candidate restriction. It validates a unique trailing
  decimal stem for every loaded image and rejects missing, non-numeric,
  duplicate, or zero-width values; stable source order is preserved. The
  window applies consistently to candidate generation, transitive/rescue
  expansion, imported verified pairs, and optional prior rematching, while
  omission preserves the historical all-candidate path. Unit coverage
  includes parser validation, deterministic candidate filtering, default
  identity, and imported-pair filtering. On the reproducible own compatible-
  SIFT artifact (208,746 features), N=2 retained **73/703** candidates; all
  73 verified with **68,358** accepted inliers and a connected 38-image graph,
  but the safe incremental stack stopped at **21/38** (**13,466 tracks /
  42,188 observations**, 0.279 px mean reprojection) and scored **24.46 cm**
  laser-GT Sim(3) centre RMSE (median 19.52, max 44.97). N=3 retained
  **108/703** candidates; **107** verified with **82,528** accepted inliers
  and a connected graph, reached **22/38** (**12,540 / 40,343**
  tracks/observations, 0.283 px), but scored **158.36 cm** (median 122.39,
  max 370.30). The corresponding COLMAP raw graph retained **73/703** and
  **108/703** pairs with **67,001** and **81,204** raw matches (verified
  **72/103**, **64,567/77,589** inliers). These connected restricted graphs
  do not beat the exhaustive 38/38 controls (31.84 cm historical own-SIFT;
  14.00 cm reverse-order artifact), and no post-registration long-range
  observation expansion was enabled. The measured bottleneck is mapper growth
  after DSC_0304: N=2 gave DSC_0305 **224→4** PnP inliers and DSC_0306
  **12→4**, while N=3 gave DSC_0306 **20→8** and DSC_0307 **8→4**. The flag
  remains an opt-in sequence diagnostic; default unordered reconstruction is
  unchanged.

- **Numeric stem window COLMAP control (2026-08-30).** The same full
  incremental stack was replayed on the COLMAP feature export plus
  `/media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_matches_import.txt`
  (`pnp-max-iterations=100000`, `min-pnp-inliers=8`, recovery, post, and final
  iterative refinement). N=2 retained **73/703** candidates, verified
  **72/73** with **64,394** accepted inliers, and stopped at **22/38** after
  DSC_0306 **67→22** and DSC_0307 **54→39** PnP inliers; the final model had
  **13,418 tracks / 42,358 observations**, 0.238 px mean reprojection, and
  **10.81 cm** laser-GT centre RMSE (median 6.38, max 24.58). N=3 retained
  **108/703**, verified **103/108** with **77,330** inliers, reached **23/38**
  after DSC_0306 **192→138**, DSC_0307 **164→127**, and DSC_0308 **10→8**;
  its final **13,447 / 43,247** tracks/observations had 0.239 px reprojection
  and **15.28 cm** RMSE (median 7.72, max 57.16). Missing images then had no
  usable initial support (N=2 image 22 had 2 correspondences; N=3 image 23
  had 0). Thus COLMAP also fails to produce a full skeleton under either
  restricted window, despite lower error on its registered prefix. No
  staged local-skeleton→exhaustive admission or post-registration expansion
  was justified; the sequence-window strategy is closed as an accuracy path,
  and the flag remains diagnostic-only with unordered defaults unchanged.

- **CI health audit and feature-gated diagnostics (2026-08-30).** The two
  image-only diagnostic examples now declare `required-features = ["image-io"]`,
  preventing no-default workspace targets from compiling feature-gated code.
  Linux runs of the CI workflow's MSRV, Tier-1 feature matrix, clippy
  (`-D warnings`), workspace tests, Python/registry/docs/examples/output/docs/package
  checks passed. `scripts/check.sh` now invokes registry `check-generated` with
  the same `--readme docs/readme_details.md` argument as CI. The authorized
  mechanical `cargo fmt --all -- --check` pass now succeeds on the existing
  dirty Rust implementation; the formatter changed no non-Rust files.

- **Cross-dataset non-regression preflight (2026-08-30).** The exact
  authoritative commands and controls for South Building, ETH3D terrace/office,
  and EuRoC MH_03_medium were audited and recorded in
  `docs/nonregression_20260830.md`. No substitute run was claimed: the local
  checkout contains only courtyard data, the South/terrace/office/EuRoC inputs
  and prepared feature artifacts are absent, and the required `torch`,
  `lightglue`, `evo`, and `colmap` tools are unavailable. Consequently no new
  registration/ATE/RPE/reprojection number or regression verdict is reported;
  the archived controls remain the acceptance targets.

- **Opt-in staged/resume incremental poses (2026-08-30).** Added
  `--initial-poses MODEL/images.txt` for the incremental mapper. It validates
  image stems, the sibling `PINHOLE` `cameras.txt` dimensions/intrinsics, and
  at least two finite seed poses; supplied poses are held fixed while the
  full verified track graph is triangulated and missing images are grown by
  PnP, then the existing final BA runs with its ordinary gauge handling.
  `None` remains the legacy path, and focused parser/mapper tests cover stem
  mapping, invalid inputs, fixed growth, missing-image PnP, final-BA release,
  and deterministic output. Using the established recovery/post/final stack,
  COLMAP features plus raw matches seeded from
  `/tmp/pair_window_colmap_raw_n2_20260830` reached **38/38**, **21,207 tracks
  / 74,202 observations**, 0.331 px mean reprojection, and **5.94 cm**
  laser-GT Sim(3) centre RMSE; the 22 seeded cameras were released for final
  BA and moved by 15.25 cm RMSE relative to their partial-model gauge. Own-SIFT
  verified pairs seeded from `/tmp/pair_window_own_n2_20260830` also reached
  **38/38**, but retained **17,459 / 58,072** tracks/observations at 0.558 px
  and **120.79 cm** RMSE; its 21 seeded cameras shifted by **221.40 cm** RMSE
  after final-BA release (aligned over their common names). Exact logs/models are
  `/tmp/staged_colmap_raw_full_20260830[.log]` and
  `/tmp/staged_own_full_20260830[.log]`. These controls show that seeding
  recovers completeness but does not preserve the partial basin after release
  or improve the recorded exhaustive controls (COLMAP 2.84 cm; own historical
  31.84 cm and current locus-path 24.95 cm); the flag remains default-off.

- **DSP-SIFT domain-size pooling, corrected VLFeat base (2026-08-30).** The
  default-off `--sift-dsp` path now pools the corrected VLFeat/COLMAP descriptor
  over the published uniform domain-size preset `λ₁=1/6`, `λ₂=4/3`,
  `N_σ̂=15`, averages the unnormalized histograms, then applies the existing
  clamp/renormalize/512-equivalent quantization once. A one-sample request is
  an exact descriptor identity; repeated extraction is deterministic. The
  preset follows [Dong & Soatto, CVPR 2015](https://openaccess.thecvf.com/content_cvpr_2015/html/Dong_Domain-Size_Pooling_in_2015_CVPR_paper.html)
  and the uniform sampling loop in [COLMAP's SIFT extractor](https://github.com/colmap/colmap/blob/main/src/colmap/feature/sift.cc).
  The full command was `target/release/examples/unordered_sfm_demo
  --feature-extractor sift --images-dir
  /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/images_1600x1066 --width 1600
  --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6
  --sift-max-keypoints 4096 --sift-vlfeat-compatible-detector
  --sift-vlfeat-compatible-descriptor --sift-vlfeat-bilinear-orientations
  --sift-dsp --exhaustive --min-matches 15 --match-ratio 0.9
  --guided-matching --verification-mode full --mapper incremental
  --pnp-max-iterations 100000 --min-pnp-inliers 8
  --geometry-guided-conflict-recovery --post-refinement-registration
  --final-iterative-refinement --out-colmap /tmp/dsp_vlfeat_full_20260830`;
  the focused CSV is `/tmp/dsp_focus_file_diag_20260830.csv` and the full log
  is `/tmp/dsp_vlfeat_full_20260830.log`.
  Focused physical-locus replay on DSC_0305–0307 at ratio 0.8 + cross-check
  gave own/imported raw and direct overlap `354/338/196`, `149/143/65`,
  `630/598/418` (precision/recall `55.37/57.99%`, `43.62/45.45%`,
  `66.35/69.90%`); ratio 0.9 gave `583/338/228`, `381/143/82`,
  `839/598/448` (`39.11/67.46%`, `21.52/57.34%`, `53.40/74.92%`).
  The fresh full 38-image NN + guided/full-verification + recovery/post/final
  run used 208,746 keypoints, verified **621/703** pairs and 158,163 inliers,
  registered **38/38**, retained **17,508 tracks / 57,863 observations** at
  **0.300 px** mean reprojection, but scored **344.41 cm** laser-GT Sim(3)
  centre RMSE (273.37 cm versus the COLMAP sparse model). This is an honest
  negative against the existing compatible-bilinear control (109.74 cm), the
  historical own-feature control (31.84 cm), and the COLMAP-feature champion
  (2.84 cm); no default or accuracy champion changed, and descriptor-ensemble
  follow-up is not justified by the focused overlap.

- **Camera-frame lever-arm oracle diagnostic (2026-08-30).** A read-only
  NumPy/SciPy fit tested the documented camera-center convention
  `C=-Rᵀt` and jointly fitted `Yᵢ=sQ(Cᵢ+Rᵢᵀd)+t` (Sim(3) plus one constant
  camera-frame offset `d`) against the ETH3D laser centers.  The actual
  COLMAP model (`colmap_oracle_full/sparse_txt/images.txt`) changed from
  **1.709/1.170/4.132 cm** (RMSE/median/max) to **1.341/0.885/3.785 cm**
  with `d=(0.67,-8.95,10.39) mm`; the COLMAP-feature champion
  (`/tmp/ba_champion_control_20260829/images.txt`) changed from
  **2.842/2.243/7.091 cm** to **2.518/2.178/6.512 cm** with
  `d=(11.83,-87.04,-0.02) mm`; and the best fixed-rotation subpixel control
  (`/tmp/colmap_subpixel3_fixedrot_20260829/images.txt`) changed from
  **1.513/0.883/4.137 cm** to **1.435/0.837/4.168 cm** with
  `d=(0.89,-22.66,11.05) mm`.  These are oracle fits, not production
  accuracy claims.  Leave-one-camera-out RMSEs were respectively
  **1.488/2.806/1.602 cm** with the fitted arm, versus **1.815/3.046/1.619
  cm** without it; the apparent gains are small relative to model spread.
  More importantly, half-trajectory fits were unstable: the first/last-half
  offsets were `(1.63,9.97,13.43)/(−0.02,−27.01,15.54) mm` for COLMAP,
  `(38.96,43.37,15.87)/(4.72,−111.80,0.15) mm` for the champion, and
  `(0.45,24.31,20.23)/(0.93,−57.65,19.72) mm` for fixed-rotation best.
  The dataset contains one camera sensor per COLMAP rig (`rigs.txt` has no
  sensor pose), no constant-rig calibration file, and no exposure/velocity
  metadata; GT cameras are independent PINHOLE models.  A world-fixed offset
  is already absorbed by Sim(3), while the unstable fitted `d` has no
  documented calibration counterpart.  The alternative `Rᵢd` convention was
  not consistently better.  Therefore no lever-arm correction or trajectory
  prior is shipped; the next work should remain focused on mapper/measurement
  errors rather than fitting this oracle offset.

- **Explicit COLMAP CPU SIFT source-row order (2026-08-30).** Added the
  default-off `SiftConfig::vlfeat_compatible_output_order` and
  `--sift-vlfeat-compatible-output-order` (requires
  `--sift-vlfeat-compatible-detector`).  The opt-in groups retained rows by
  ascending `(octave, level)` and preserves VLFeat's scan/orientation sequence
  within each group; it does not introduce a response/scale/descriptor sort.
  This follows COLMAP `sift.cc`'s complete-DoG-level suffix cap (the cap counts
  unoriented extrema) and the CPU wrapper's joint keypoint/descriptor
  permutation, with the source contracts documented in [COLMAP `sift.cc`](https://github.com/colmap/colmap/blob/main/src/colmap/feature/sift.cc),
  [COLMAP `sift.h`](https://github.com/colmap/colmap/blob/main/src/colmap/feature/sift.h),
  and [VLFeat `sift.c`](https://raw.githubusercontent.com/vlfeat/vlfeat/master/vl/sift.c).
  Synthetic tied-group/reference-order and default-identity tests pass.  On
  the rounded-gray, corrected-descriptor, bilinear 4096 export command
  (`--sift-vlfeat-compatible-detector --sift-vlfeat-compatible-descriptor
  --sift-vlfeat-bilinear-orientations --sift-colmap-compatible-grayscale
  --sift-vlfeat-compatible-output-order --export-features-only`), **208,819**
  rows were emitted; the feature-content SHA
  `3feab3a0a5707ee491a2f4db0b69db2a1caf935300a23a798d7305474205dbed` was
  byte-identical to the existing rounded-gray export.  A deterministic DB
  locus join matched **208,590/208,785 (99.9066%)** rows; only **1,422** full
  rank inversions remained (weighted **0.000244%**; critical `0305/0306/0307`
  had **0/0/0**), so the previously reported ~0.327% discrepancy is matching
  ambiguity rather than a missing response/scale key.  Focused ratio-.8
  cross-check raw/accepted rows were `0305--0306=309/258`,
  `0305--0307=152/102`, and `0306--0307=585/539`; imported physical-locus
  remapping accepted `317/122/590` respectively.  With the common ratio-.9,
  guided/full, recovery+post/final stack, fresh NN verification gave
  **638/703, 154,858 inliers, 38/38, 14,827 tracks, 0.347 px, 401.95 cm**
  laser-GT Sim(3) RMSE.  The diagnostic physical-locus COLMAP raw replay
  retained **161,083/161,301** correspondence rows from **685** pairs and
  gave **405/703, 140,718 inliers, 38/38, 19,893 tracks, 0.392 px, 17.18 cm**;
  it is not an accuracy claim because 218 endpoints could not be remapped.
  The output-order flag therefore preserves the default and does not displace
  the existing fixed-COLMAP 1.76 cm or rounded-gray controls.

- **Rounded-gray detector parity and physical-locus remap audit
  (2026-08-30).** A read-only all-image join of the source-order export and
  the actual COLMAP six-column keypoint rows reproduced **208,819 own / 208,785
  DB rows / 208,590 row matches**.  The exact unmatched row indices were
  classified by nearest physical locus: own-only **229 = 186 orientation
  multiplicity + 3 localization (≤3 px) + 40 no-DB-locus rows**; DB-only
  **195 = 173 orientation multiplicity + 2 max=4096 level-cap rows + 2
  localization (≤1.5 px) + 18 no-own-candidate rows**.  The two cap rows are
  `DSC_0294` DB indices **434,435** (own pre-candidate `(octave=-1,level=1)`,
  response `-0.033509814`, edge score `6.2913`, present after orientation but
  removed by the complete-level cap).  No row difference was attributable to
  duplicate/tie traversal ordering; the 1,422 rank inversions are order-only
  ambiguity among already matched physical loci.  The only candidate-boundary
  own-only rows were `DSC_0304[3080]` and `DSC_0319[282]` near the
  `0.02/3` contrast threshold (absolute responses `0.0066729/0.0066849`),
  plus 15 rows within 0.1 of the edge limit `12.1`; these are input-sensitive
  candidates, not a proven source comparison bug.  The per-image exact row
  lists and category counts were generated with the deterministic spatial
  join used for this audit.

  The callable upstream VLFeat probe agreed with the compatible detector's
  pre-orientation candidate set for `DSC_0305` (**4598/4598**) and `DSC_0306`
  (**3113/3113**) at reciprocal spatial tolerance 0.1 px; `DSC_0307` differed
  by only two own candidates (**3066** vs callable **3064**), with own responses
  `-0.023385716` (edge `9.2569`) and `+0.023396332` (edge `4.5857`), far from
  the contrast/edge rejection boundaries.  Source inspection confirms the
  exact strict neighbor tests, five-step localization, six histogram smoothing
  passes, first-two orientation cap, and complete `(octave,level)` suffix cap
  from COLMAP/VLFeat; no compatible-mode code change was justified.  A direct
  one-image cap control further showed **9,727 rows at max=8192** versus
  **5,347 at max=4096**, while the DB row was **5,342**, so the DB's effective
  extraction is consistent with a 4096-level cap despite the `sift.h` header
  default of 8192; the exact runtime option is not recoverable from SQLite.

  A physical-locus many-to-one raw remap (same deterministic nearest own row
  may represent orientation alternatives) retained **161,290/161,301** raw
  correspondences under distance ≤1 px and `|log σ|≤0.2`; the remaining 11
  rows require a nonphysical assignment.  Its mapper result was **404/703,
  140,890 inliers, 38/38, 17,121 tracks, 0.324 px, 12.67 cm** laser-GT
  Sim(3).  A forced nearest-row file retained all **161,301/161,301** rows but
  injected those 11 nonphysical assignments and regressed to **404/703,
  140,890 inliers, 38/38, 17,310 tracks, 0.396 px, 16.03 cm**.  Neither raw
  file is byte-identical to the original COLMAP index stream (raw SHA differs
  from `7022ffa2...`), and neither reproduces the fixed-COLMAP raw-import
  control (**38/38, 1.76 cm**); the experiment is diagnostic only.  No broad
  threshold/order/cap change was added.

- **Opt-in orientation-locus canonicalization (2026-08-29).** SIFT exports
  now retain `(x, y, sigma, orientation)` in an optional `_loci.txt` sidecar
  (and six-column affine feature rows are accepted); legacy feature files
  without metadata are unchanged.  With
  `--orientation-locus-canonicalization`, orientation rows sharing one
  quantized `(x, y, sigma)` locus are reduced to one deterministic
  representative before track construction, and duplicate accepted
  locus-pairs keep the best descriptor-distance candidate with geometric and
  stable-key tie breaks.  On the reproducible all-floor SIFT input this
  collapsed **30,385** orientation rows from **208,746** rows into **178,361**
  loci; the accepted stream changed from **154,559** to **136,509** rows
  (**18,050** locus-pair duplicates across **553** pairs).  The original and
  reversed metadata-bearing feature orders then produced identical
  **38/38**, **10,274-track**, **0.405 px** reconstructions and identical
  laser-GT Sim(3) centre RMSE **223.01 cm** (median **145.07 cm**).  A
  metadata-free fixed-COLMAP-keypoint raw-import control was unchanged at
  **38/38**, **21,011 tracks**, **0.289 px**, and **3.01 cm**.  An initial
  metadata-free trial appeared to improve to **6.02 cm**, but was found to
  sort an identity-mapped match stream and thereby violate the required
  legacy no-op; the corrected behavior and proper baseline are recorded
  below.  This remains a deterministic topology/order diagnostic, not an
  accuracy champion; the flag remains default-off.  Focused tests cover
  representative selection, distinct scales, permutation invariance, and the
  metadata-free no-op behavior.

- **Metadata-free locus no-op and own-SIFT lock-down (2026-08-29).** The
  canonicalizer now returns before any match sorting when all feature files
  lack locus metadata, so enabling the diagnostic flag cannot alter legacy
  traversal.  The exact command used the 38 metadata-free files in
  `/tmp/vlfeat_source_latest_floor_20260829` (**208,746 rows, 261,602,022
  bytes, no `_loci.txt` files; name+content SHA-256
  `a810da800ffb0eae14d034729b85fae8212a8113a538cf57fbbcab5e187f2964`), the
  verified import `/tmp/repro_31p84_reverse_remapped_verified_e_20260829.txt`
  (**648 pairs / 154,559 inliers, raw SHA-256
  `1f6fefdf7d94bc3d25b5b8ce0a9d2775d9fef584d963945accf44c4517dd5c5b`), and
  the established exhaustive ratio-0.9/guided/full, plain incremental,
  PnP-100k/min-8, recovery+post+final stack.  The no-flag command was
  `target/release/examples/unordered_sfm_demo --feature-extractor files
  --features-dir /tmp/vlfeat_source_latest_floor_20260829 --feature-suffix
  _features.txt --image-suffix .png --width 1600 --height 1066 --fx 879.4
  --fy 879.4 --cx 803.4 --cy 532.6 --import-verified-pairs-file
  /tmp/repro_31p84_reverse_remapped_verified_e_20260829.txt --exhaustive
  --min-matches 15 --match-ratio 0.9 --guided-matching --verification-mode
  full --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8
  --geometry-guided-conflict-recovery --post-refinement-registration
  --final-iterative-refinement --out-colmap
  /tmp/locus_own_locked2_run1_20260829`.  The no-flag effective-config
  snapshot hash was **`ab855aa8f91d4a94`**; both runs had edge FNV
  **`f38f80a2ef9c2c96`**, track build **25,945 components / 1,479 conflicts /
  24,466 retained tracks / 78,980 observations**, **36 PnP attempts** (sum
  **24,607→19,268** correspondences/inliers), growth **38/38**, and final
  **12,658 tracks / 43,753 observations / 0.569 px**.  Two runs were
  byte-identical: `cameras.txt` SHA
  `a2132068b1a4dbe21f1ad68a23ff05461026c5a84e0b0de14f06311533e5b958`,
  `images.txt` SHA
  `06f84b547914e97229d76ae95c50005929e027b3ac4526c465409ae3a0a00fcf`, and
  `points3D.txt` SHA
  `059d0881a54440e2db0ede91ea0595c3bd4c62385ad211188cf8372333820fe7`.
  Baseline artifacts are `/tmp/locus_own_locked2_run1_artifact_20260829` and
  `/tmp/locus_own_locked2_run1_20260829`; the explicit-flag control is
  `/tmp/locus_own_locked2_flag_20260829`.
  Explicit `--orientation-locus-canonicalization` (snapshot
  `f3463e6c0beb3865`) produced the same three model hashes, confirming the
  metadata-free no-op.  The baseline laser-GT Sim(3) score is **184.01 cm
  RMSE / 158.03 cm median / 339.83 cm max** (scale **0.385887**), and its
  COLMAP-model-relative score is **145.51 cm**; the actual COLMAP sparse model
  is a separate **1.71 cm / 1.17 cm / 4.13 cm** laser-GT control.  Baseline
  per-image error is largest at `DSC_0323` **339.8 cm**, `DSC_0307` **321.0
  cm**, `DSC_0316` **290.8 cm**, and `DSC_0317` **253.5 cm**; the complete
  table is `/tmp/locus_own_locked2_per_image_20260829.txt`.  The exact
  historical **31.84 cm** run is not the same experiment: it re-extracted
  the same 208,746-row compatible SIFT configuration and performed fresh NN
  verification (**639/703, 154,854 inliers; config hash
  `ce75f7453d787fce`), whereas this lock-down imports the distinct
  **648/703, 154,559** verified set.  No feature-row or default-SIFT change
  explains the score difference.

  The one requested intrinsics A/B added only `--refine-intrinsics` (snapshot
  `b621d11a503b112c`): it retained **38/38**, **12,560 tracks / 43,397
  observations**, **0.373 px**, and improved the laser-GT score to **168.64
  cm RMSE / 122.55 cm median**, but worsened the maximum to **371.08 cm**;
  final `fx/fy/cx/cy=878.72/880.45/802.51/532.03`.  The repeated run was
  byte-identical (`cameras.txt`
  `4f9446dbbd7925ee791e2d5912d70076bd7fd527b6d124e25716dabec620118e`,
  `images.txt`
  `a11d3d8e38415fe0b4cc57aefc9f8be7118e8ebd35f7faacae3ba3b2ed27124d`,
  `points3D.txt`
  `c74f9fe97729ea94c16b9aa37954f49959e644e19f2337f4072d5c0aa87c2073`), so
  artifacts are `/tmp/locus_own_locked2_refine_run1_artifact_20260829` and
  `/tmp/locus_own_locked2_refine_20260829`; this is reproducible but not an
  accuracy champion or a justification to change defaults.

- **Offline affine-normalized multi-view LK subpixel probe (2026-08-29).**
  A read-only probe used the six-column affine keypoint metadata from
  `colmap_oracle_full/database.db` to render fixed **9×9 canonical** patches
  on the normalized images, then ran 10 translation Gauss–Newton iterations
  in both directions.  A proposal required finite/in-bounds sampling, image
  shift ≤`1.0 px`, forward/backward endpoint consistency ≤`0.2 px`, and
  bidirectional ZNCC ≥`0.9`; endpoint proposals were combined by weighted
  geometric median and retained only with at least two proposals and MAD
  ≤`0.2 px`.  Across **142,720** COLMAP verified inlier rows, **49,211**
  proposals passed (**34.48%**), covering **44,556** endpoints; **15,890**
  endpoints were updated, while **23,020** had fewer than two proposals and
  **5,646** failed the MAD gate.  Accepted proposal forward/backward-error
  median/p90/p99 was **0.070/0.165/0.196 px**, and the conservative shift
  median was **0.227 px** per endpoint.  Focused verified-row acceptance was
  **90/320**, **19/123**, **228/591**, and **3/22** on
  `0305--0306`, `0305--0307`, `0306--0307`, and `0306--0308` respectively.
  Reusing the exact bridge-supplement/champion mapper stack on
  `/tmp/colmap_features_lk9_20260829` changed verification from **380/703**
  to **375/703**, produced **38/38, 20,021 tracks, 63,345 observations,
  0.432 px**, and scored **12.53 cm** laser-GT centre RMSE (median **11.00
  cm**, max **31.54 cm**, scale **0.56597**) versus the original **2.84 cm**
  champion and offline cornerSubPix3's **2.01 cm**.  The fixed-COLMAP-rotation
  oracle reduced reprojection **9.321→0.444 px** but still scored **2.55 cm**
  (not deployable).  The artifact and logs are
  `/tmp/colmap_features_lk9_20260829` and
  `/tmp/{subpixel_lk_full,colmap_lk9_champion,colmap_lk9_fixedrot}_20260829*`.
  This strict affine/LK update is therefore a reproducible negative result;
  no Rust implementation or default behavior change is justified.

- **Courtyard camera-model and radial-residual audit (2026-08-29).** The
  authoritative `colmap_oracle_full/database.db` contains one shared
  `PINHOLE` camera (1600×1066, prior `fx/fy/cx/cy =
  879.4/879.4/803.4/532.6`); its optimized sparse model is still `PINHOLE`
  with `880.1955302/879.3997538/803.4/532.6` and has no distortion slots.
  The ETH3D `gt` symlink is explicitly
  `dslr_calibration_undistorted` and its four original-resolution cameras are
  also `PINHOLE`.  The normalized RGB PNGs are 1600×1066 and carry no EXIF or
  color-profile metadata; the mixed 1065/1067-height source set is not the
  authoritative normalized input.  A read-only projection audit over
  `74,091` observations in `/tmp/ba_champion_control2_20260829` found median /
  p90 / p99 reprojection `0.170/0.618/1.970 px` (RMSE `0.461 px`), radial
  residual-versus-normalized-radius correlation `-0.0044`, and the physically
  constrained `e_r = b0 + b3 r^3` fit `b3=-0.00149±0.00510 px` (`k1=-1.7e-6
  ±5.8e-6`).  The `/tmp/subpixel_ab_refineintr_delta3_20260829` model was
  consistent: `72,981` observations, median / p90 / p99 `0.176/0.641/1.954
  px`, radial correlation `-0.0082`, `b3=-0.00434±0.00522 px` (`k1=-4.9e-6
  ±5.9e-6`).  Eight 45° angular bins had mean radial and tangential residuals
  below `0.011 px` in magnitude; the actual COLMAP sparse model showed the
  same no-outward-growth pattern (radial correlation `-0.0192`).  Therefore
  the data do not support a new shared/global `k1`/`k2` camera model or a
  distortion safeguard; the existing opt-in `--refine-distortion` remains
  untouched and the default pinhole path is unchanged.

- **Offline COLMAP-keypoint subpixel probe (2026-08-29).** A read-only probe
  preserved feature rows, descriptors, and match indices while applying
  bounded OpenCV/libpng `cornerSubPix` x/y refinement.  The 3x3 variant kept
  **380/703 verified pairs, 38/38 registration**, and improved the champion's
  laser-GT Sim(3) centre RMSE from **2.84 cm to 2.01 cm** (mean reprojection
  **0.294 px**); 5x5 degraded it to **20.98 cm** (0.447 px).  With COLMAP
  rotations fixed as an oracle, the 3x3 result was **1.51 cm** (0.290 px),
  but this is not a deployable solution.  A Rust in-process reimplementation
  (including the libpng integer gray coefficients and source-style f32 patch
  sampling) matched the critical pair raw/accepted counts and was numerically
  within roughly 1e-5 px for most rows, yet its legacy traversal produced
  **38/38, 20,743 tracks, 0.329 px, 11.68 cm**.  The few unstable corner
  updates were enough to change order-sensitive track construction, so the
  experimental CLI/helper was removed; no default behavior changed.  The
  exact OpenCV-generated feature artifacts remain under
  `/tmp/colmap_features_subpixel3b_20260829` and
  `/tmp/colmap_features_subpixel5b_20260829` for reproducibility.

- **Offline subpixel 2x2 BA A/B (2026-08-29).** Reusing the exact
  `/tmp/colmap_features_subpixel3b_20260829` rows and bridge supplement with
  the common exhaustive/full-verifier, plain-incremental,
  `pnp100k + min8 + recovery + post + final-iterative` stack, all four
  variants retained **380/703 verified pairs, 140,663 inliers, and 38/38
  registration**.  With fixed intrinsics, δ=`3.0` gave **21,149 tracks,
  0.294 px, 2.008/1.379/5.237 cm** (RMSE/median/max laser-GT), while
  δ=`0.5` gave **21,153, 0.291 px, 5.459/3.054/22.727 cm**.  With
  `--refine-intrinsics`, δ=`3.0` was best at **21,120 tracks, 0.290 px,
  1.831/1.442/4.011 cm**, ending at
  `fx/fy/cx/cy=880.4941/879.8661/803.8111/532.2838`; δ=`0.5` gave
  **21,111, 0.286 px, 4.901/2.429/24.039 cm** and
  `881.8698/880.1322/803.4380/532.8686`.  The δ=`3.0` + intrinsics result
  reproduced twice byte-identically (`21,120` tracks and the same model
  text).  A fixed-COLMAP-rotation oracle run on that best configuration
  reduced reprojection **1.370850→0.297496 px** and scored **1.643 cm**
  (median **0.952**, max **4.130 cm**), but is not deployable.  The modest
  1.831 cm offline result does not justify restoring a Rust subpixel path:
  the exact in-process reimplementation previously changed order-sensitive
  tracks and scored 11.68 cm; no default behavior or production feature path
  is enabled.

- **BA Huber-delta diagnostic (2026-08-29).** Added the default-off
  `--ba-huber-delta PX` CLI override for the shared periodic/final/global BA
  Huber threshold; omission remains the historical `delta=3 px` and requires
  no behavior change.  On the COLMAP-feature recovery+post champion, the
  final 74,091-observation residual distribution at the control was
  `median=0.170 px, p90=0.618, p95=0.948, p99=1.970, p99.9=2.973`, with only
  **0.09%** above 3 px.  The bounded `delta={0.5,1.0,2.0,3.0}` runs kept
  38/38 registration and 20-iteration caps, but the existing iterative
  filter/re-triangulation naturally changed support: respectively
  **21,341/21,405/21,129/21,338 tracks**, **0.281/0.289/0.291/0.283 px**,
  and **3.60/2.92/3.31/2.34 cm** against the actual COLMAP model (the
  control is **2.84 cm** against laser GT).  With actual COLMAP rotations
  fixed on the same diagnostic path, the laser-GT Sim(3) centre RMSE was
  **1.57/1.89/1.64/1.62 cm** (δ={0.5,1.0,2.0,3.0}; median
  **1.06/1.33/1.04/1.06 cm**, max **3.52/4.66/3.98/4.04 cm**, scales
  **0.5742/0.5766/0.5740/0.5760**).  The corresponding scores against the
  actual COLMAP model were **0.74/1.07/0.62/0.65 cm** with medians
  **0.57/0.75/0.52/0.51 cm** and maxima **2.45/4.24/1.64/1.56 cm**;
  the actual COLMAP sparse model itself is **1.71 cm** versus laser GT.
  Thus no fixed-rotation delta is sub-cm against the authoritative GT;
  this remains an oracle control, not a production improvement.  Since the residuals are
  overwhelmingly below 3 px and no delta consistently improves the
  GT-independent champion while preserving support, the default remains
  3 px and no new production policy is enabled.  Artifacts:
  `/tmp/ba_huber_{0p5,1p0,2p0}_20260829` and
  `/tmp/fixed_rotation_oracle_huber_{0p5,1p0,2p0}_20260829`.

- **Incremental rotation-vs-translation BA decomposition (2026-08-29).**
  Added the default-off `--diagnose-fixed-rotation-ba SOURCE` diagnostic and
  `BundleAdjustment::fix_pose_rotation`: `SOURCE=current` freezes the
  incremental champion rotations, while a COLMAP `images.txt` source is
  right-aligned to the champion gauge and freezes only those rotations;
  translations, landmarks, support, and the ordinary gauge anchors remain
  free.  The fixed-rotation constraint is identity-row projected in the
  existing six-DoF Schur system, with tests for rotation immutability,
  translation optimization, and empty-set/default identity.  On the
  COLMAP-feature champion configuration (`--import-matches-supplement-file
  colmap_bridge_matches_import.txt --exhaustive --min-matches 20
  --match-ratio 0.8 --verification-mode full --pnp-max-iterations 100000
  --min-pnp-inliers 8 --geometry-guided-conflict-recovery
  --post-refinement-registration --final-iterative-refinement`), B=`current`
  preserved **38/38, 21,338 tracks, 0.283 px, 2.84 cm**.  A using
  `colmap_oracle_full/sparse_txt/images.txt` reduced fixed-support reprojection
  **2.459→0.278 px**, kept rotations within **3.0e-15°**, and scored
  **1.62 cm** versus laser GT (**0.65 cm** versus the actual COLMAP model).
  C using `/tmp/global_colmap_raw_ba_diag_20260829/images.txt` reduced
  reprojection **78.139→4.666 px** but scored **85.58 cm**; its source
  rotations have **6.74° median / 8.31° p90** error against COLMAP.  Thus the
  oracle control exposes a rotation-independent translation/measurement floor,
  but no GT-independent rotation-averaging-to-incremental path is shipped:
  the existing global rotation average is not accurate enough.  Artifacts:
  `/tmp/fixed_rotation_{current,oracle,global}_20260829`; the exact controls
  were run with the command above plus `--diagnose-fixed-rotation-ba
  {current|/media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_oracle_full/sparse_txt/images.txt|/tmp/global_colmap_raw_ba_diag_20260829/images.txt}`.

- **Sequence-aware trajectory-regularization probe (2026-08-29).** A
  read-only NumPy/SciPy probe ordered the four complete 38-camera models by
  numeric filename stem (`DSC_0286`--`DSC_0323`; every observed stem gap was
  1), and fit only the estimated poses—laser ground truth was consulted only
  after fitting. The inputs were the actual COLMAP model
  (`colmap_oracle_full/sparse_txt/images.txt`, **1.709 cm** versus laser GT),
  the COLMAP-feature champion (`/tmp/ba_champion_control_20260829`,
  **2.842 cm**), the fixed physical-hash traversal candidate
  (`/tmp/union_hash_fixed_20260829_seed11_reverse`, **1.995 cm** against the
  actual COLMAP model and **3.011 cm** against laser GT), and the best
  complete own-SIFT traversal (`/tmp/order_own_reverse_full_20260829`,
  **9.650 cm**). The apparent historical fixed-traversal **1.99 cm** number
  is therefore the COLMAP-model comparison, not the laser-GT score.

  Center first differences (median/p90/max, m per stem) were
  **0.648/1.955/7.069** (COLMAP), **1.402/4.276/15.422** (champion),
  **1.394/4.292/15.393** (fixed), and **1.506/4.351/15.446** (own). The
  second-difference norms (median/p90/max, m per stem²) were respectively
  **0.285/2.832/7.741**, **0.630/6.217/16.901**,
  **0.632/6.201/16.874**, and **0.769/6.337/16.939**; robust MAD scales were
  **0.265/0.591/0.594/0.820 m**. Orientation increments were already large
  and consistent across the models (median/p90/max **24.78/38.94/74.63°**
  for COLMAP and **24.76/39.11/74.56°** for the champion), with orientation
  second-difference medians **11.53°** and **11.51°**. In particular, the
  real path turn at `DSC_0310`/`DSC_0311` produced center second differences
  **16.48/16.90 m** in the champion (COLMAP itself has **7.55/7.74 m**), so
  it is not an observation-noise-like high-frequency fluctuation.

  Tested standard local quadratic Savitzky--Golay fits with windows 5 and 7
  (actual stem coordinates, endpoint samples held fixed) and a
  second-difference Tikhonov fit
  `min ||C-C_hat||² + λ||D²C_hat||²`. The latter used fixed Huber weights from
  the pre-fit acceleration to avoid suppressing large turns and selected λ
  by GCV. GCV selected the lower tested bound **1e-6** for every robust fit,
  i.e. a numerically identity result; unweighted GCV selected **0.0562** for
  COLMAP/champion/fixed and **0.0316** for own, but those fits also over-smoothed
  the real turn. Laser-GT Sim(3) RMSE (cm) was:

  | model | raw | SG-5 | SG-7 | robust Tikhonov |
  |---|---:|---:|---:|---:|
  | COLMAP | 1.709 | 64.479 | 91.687 | 1.709 |
  | champion | 2.842 | 64.530 | 91.719 | 2.842 |
  | fixed traversal | 3.011 | 64.578 | 91.732 | 3.011 |
  | own traversal | 9.650 | 64.683 | 91.749 | 9.650 |

  The champion's SG-5 did improve isolated images (`DSC_0305` **2.90→1.12
  cm**, `DSC_0307` **1.90→1.40 cm**, `DSC_0308` **5.07→4.00 cm**) but moved
  `DSC_0310` **1.70→241.45 cm**, `DSC_0311` **4.95→255.86 cm**, and
  `DSC_0312` **3.64→92.18 cm**; SG-7 sent the first two to **306.07/336.60
  cm**. Holding original rotations, landmarks, and observations fixed, the
  same SG-5 center replacement raised model reprojection RMS from
  **0.461→36.934 px** (champion; COLMAP **0.482→34.768 px**), confirming that
  an export-only smoothing result is not a valid BA initialization. Since
  robust GCV correctly collapsed to no-op and both SG windows catastrophically
  damaged even the actual COLMAP trajectory, no trajectory-prior/refinement
  CLI or BA path was added; unordered/default behavior is unchanged. The next
  useful direction remains mapper/observation geometry rather than filename
  order regularization.

- **Global-SfM independent-initialization audit (2026-08-29).** Re-ran the
  current `--mapper global` path with `VISLOC_GLOBAL_DEBUG=1` and
  `VISLOC_SFM_DEBUG_BA=1` on the normalized courtyard. The COLMAP-feature
  graph used imported COLMAP raw matches plus our full verifier (380/703
  pairs, **140,445** inliers); the fixed-COLMAP-keypoint corrected-descriptor
  graph used ordinary NN+ratio-0.8 matching (408/703, **122,211** inliers).
  Both rotation trees reached **38/38**, but systematic translation-direction
  error remained: COLMAP graph post-average rotation error **2.5149°**
  (230/304 kept), translation-sign repair **7/230**, and final position
  residual **127.36°**; fixed graph **2.7925°** (240/291), **19/240**, and
  **109.01°** respectively. Representative GT-only bearing diagnostics
  (not used by mapping) were COLMAP `0305--0306` E=261/R=1.3°/bearing=29.0°,
  `0306--0307` 477/2.4°/13.5°, and `0305--0307` 0/8.0°/62.5°; fixed was
  209/0.7°/6.2°, 398/3.4°/28.0°, and 67/4.9°/13.2°. The COLMAP graph's
  global BA moved only **3/17 accepted steps** (Huber cost
  `1.512701139e7→1.155881365e7`) and finished **38/38, 2,662 tracks,
  262.697 px, 440.80 cm** Sim(3) RMSE; fixed BA accepted **6/20** steps
  (`7.947595948e6→7.537940475e5`) but moved cameras up to **1.053 m** and
  finished **38/38, 2,120 tracks, 26.919 px, 413.94 cm**. A fixed 401-pair
  COLMAP verified-import control was also negative (**141.03°** position
  residual, **506.803 px**, **464.07 cm**). These results reproduce the
  established global failure against the incremental COLMAP-feature
  champion (**2.842 cm**) and actual COLMAP (**1.709 cm**): rotation residuals
  are locally small while translation bearings are globally inconsistent, so
  no new generic rotation/translation weighting or cycle threshold was
  shipped; existing default-off robust options remain unchanged. Artifacts:
  `/tmp/global_colmap_raw_ba_diag_20260829`,
  `/tmp/global_fixed_vlfeat_ba_diag_20260829`, and
  `/tmp/global_colmap_verified_20260829`.

- **Seeded physical-edge union traversal replay (2026-08-29).** Extended
  `--union-traversal-order` with `physical-hash:SEED` and
  `physical-hash-reverse:SEED`. These modes sort already-verified matches by
  a seeded FNV-1a key over canonical image IDs and quantized (1e-6 px)
  endpoint coordinates, then sort pairs by their first physical edge; the
  descending form reverses both streams. Matching, verification, feature rows,
  and correspondence values are untouched. The focused CLI test covers
  decimal/hex seeds, invalid values, default identity, deterministic replay,
  and the unchanged unordered-edge multiset.

  The replay command template (with `ORDER` set to each listed value and a
  fresh `OUT` directory) was
  `target/release/examples/unordered_sfm_demo --feature-extractor files --features-dir FEATURES --feature-suffix _features.txt --image-suffix .png --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 --import-verified-pairs-file VERIFIED --exhaustive --min-matches 15 --match-ratio 0.9 --guided-matching --verification-mode full --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8 --geometry-guided-conflict-recovery --post-refinement-registration --final-iterative-refinement --union-traversal-order ORDER --out-colmap OUT`.
  It fixed features and bypassed matching/verifier:
  own `/tmp/vlfeat_source_latest_floor_20260829` (**208,746** rows) with
  `/tmp/repro_31p84_reverse_remapped_verified_e_20260829.txt` (**648** pairs,
  **154,559** edges, **49** calibrated-E references), and fixed COLMAP
  `/tmp/colmap_fixed_vlfeat_all_l2_20260829` (**208,785** rows) with
  `colmap_verified_import.txt` (**401** pairs, **141,698** edges, **76**
  calibrated-E references). For the fixed family the same template used
  `--match-ratio 0.8` and omitted `--guided-matching`; all other mapper flags
  were unchanged. Original plus 12 forward and 12 reverse seeds
  all registered **38/38** and preserved one edge hash per family:
  own=`f38f80a2ef9c2c96`, fixed=`55009052f64f8520`. Own runtime was
  **11.31--23.61 s** (tracks **12,314--16,402**, reprojection
  **0.294--0.631 px**); fixed runtime was **13.41--21.57 s** (tracks
  **19,256--21,011**, reprojection **0.279--0.413 px**).
  The same-replay `original` controls were own **12,658 tracks / 0.569 px /
  145.51 cm** (`12.057710°`) and fixed **19,754 tracks / 0.408 px /
  17.92 cm** (`15.542829°`).

  For exact candidate traceability, seeds are listed in order `0..11`.
  Own forward rotation-score degrees were
  `[14.482310,10.590897,10.058644,10.632620,14.261293,12.095812,9.681233,12.721597,10.615958,13.527809,9.870056,12.048548]`
  with GT-evaluation Sim(3) cm
  `[159.48,108.01,187.65,172.11,204.61,187.26,85.71,135.47,134.10,182.40,50.18,140.03]`;
  reverse was
  `[10.645807,10.025560,10.092637,12.025894,10.215592,10.388985,9.922394,13.038395,10.045891,10.377913,11.026348,10.511796]`
  and `[117.85,88.86,20.96,146.67,58.39,143.35,53.60,190.57,108.90,77.62,140.73,181.01]`.
  Fixed forward was
  `[15.328568,15.423492,15.325871,15.539681,15.503927,15.529937,15.423235,15.289910,15.505591,15.529937,15.487982,15.542829]`
  and `[4.91,10.59,2.65,21.19,14.76,17.09,10.58,19.17,15.71,17.09,12.66,17.92]`;
  reverse was
  `[15.295781,15.529937,15.480235,15.523372,15.332489,15.542829,15.542829,15.529937,15.505591,15.529937,15.529937,15.321756]`
  and `[19.00,17.09,15.30,14.44,10.29,17.92,17.92,17.09,15.71,17.09,17.09,1.99]`.
  The lowest rotation score therefore selected own forward seed 6 (**9.681233°,
  85.71 cm**) and fixed forward seed 7 (**15.289910°, 19.17 cm**), while the
  GT-best runs were own reverse seed 2 (**10.092637°, 20.96 cm**) and fixed
  reverse seed 11 (**15.321756°, 1.99 cm**). Pearson/Spearman score-vs-GT
  correlations were only **0.638/0.704** (own) and **0.589/0.557** (fixed),
  so no multi-traversal selector was added. The historical 31.84/14.00 cm
  and 1.76 cm controls use different internally generated/imported traversal
  streams and are not silently conflated with this replay. Targeted
  `cargo check`, 10 `diagnose_cli_tests`, and `git diff --check` pass;
  `cargo fmt --all -- --check` still reports pre-existing formatting in the
  shared dirty tree and was not used to rewrite unrelated files.

- **GT-independent multi-model pose-consensus probe (2026-08-29).** Offline
  replayed the existing 24 complete 38-camera traversal models in each family
  (12 forward + 12 reverse) without using GT during construction. A medoid was
  selected by pairwise Sim(3)-normalized centre-shape distance; each model was
  aligned to the current consensus, camera centres were robustly averaged by
  weighted geometric median, rotations by deterministic SO(3) quaternion
  averaging, and the arbitrary zero-scale solution was removed by fixing the
  medoid centroid/radius. The alternating solve converged in 8 (own) and 5
  (fixed) iterations. Medoids were own `physical-hash-reverse:6` and fixed
  `physical-hash-reverse:3`; pair-shape dispersion was median/p90
  `0.5175/0.7499` (own) and `0.03344/0.06350` (fixed). Centre uncertainty after
  per-model alignment was median/p90 **156.31/409.47 cm** (own) and
  **8.82/23.49 cm** (fixed), with largest fixed uncertainty at `DSC_0316`
  (**52.17/87.78 cm**) and `DSC_0307` (**12.30/91.36 cm**); own was dominated
  by `DSC_0307` (**355.46/697.81 cm**) and `DSC_0316` (**339.44/669.37 cm**).
  The deterministic raw consensus scored **80.31 cm** (own) and **16.50 cm**
  (fixed) Sim(3) centre RMSE. For comparison, the current 24-model individual
  best/median were **26.83/173.76 cm** (own) and **3.01/21.03 cm** (fixed).
  Consensus pose files are `/tmp/consensus_20260829/{own,fixed}_gauge_images.txt`
  (SHA-256 `5413cd27…`/`9b17758e…`).

  As a fixed-support BA control, the existing opt-in
  `--diagnose-ba-oracle-poses CONSENSUS/images.txt` path was run with each
  family's medoid traversal and the common verified set; it retained 38/38
  cameras and ran the unchanged BA, but deliberately reuses/transforms the
  existing medoid track support rather than changing track membership. The
  resulting models had own **14,644 tracks / 0.322 px / 89.05 cm** and fixed
  **20,487 tracks / 0.377 px / 17.06 cm**, both worse than their raw consensus
  and individual best (fixed BA reprojection **3.3798→0.3766 px**, own
  **29.0020→0.3221 px**). Since consensus does not approach the existing
  accuracy champions and BA moves it away, no default-off consensus importer,
  selector, or retriangulation path was added. Artifacts are under
  `/tmp/consensus_{20260829,ba_*_gauge_medoid_20260829}`; construction and
  scoring used no GT, which was consulted only for these final evaluations.

- **Calibrated-E relative-direction model cross-validation (2026-08-29).**
  Extended the default-off `--diagnose-model-score MODEL/images.txt` report
  with a GT-independent relative translation-direction check. For every
  registered pair carrying a calibrated imported E, the diagnostic uses the
  accepted imported correspondences, hardened cheirality sign selection
  (positive-depth ratio >=0.5, >=15% winner margin, p25 triangulation angle
  >=1 degree), and deterministic prefix/suffix/evenly-spaced 8-point refits.
  References with fewer than two valid refits or >20 degree translation spread
  are excluded from the stable subset. Pair rows now expose stable rotation and
  signed translation disagreement, cheirality/parallax/refit counts and
  rotation/translation refit spreads; summary rows expose pair-balanced
  translation mean/median/p90/Huber values and image-balanced Huber/coverage.
  The exact replay command was the existing model-score template with
  `--feature-extractor files --features-dir FEATURES --feature-suffix
  _features.txt --image-suffix .png --width 1600 --height 1066 --fx 879.4
  --fy 879.4 --cx 803.4 --cy 532.6 --import-verified-pairs-file VERIFIED
  --diagnose-model-score MODEL/images.txt --out-colmap
  /tmp/diagnose_model_xval_scratch`; no reconstruction was rerun.
  Across the 24 own-SIFT traversal candidates (12 forward + 12 reverse), 50
  calibrated-config rows were available (the legacy rotation decomposition
  was finite for 49) and 28 passed the stability gate (56.0% pair coverage;
  59.1% image coverage). Stable-rotation alone selected
  `seed5` at **0.836339 degrees** (GT **187.26 cm**), translation Huber alone
  selected `seed6_reverse` at **4.071559 degrees** (GT **53.60 cm**), and the
  equal-unit stable-rotation + translation-Huber score also selected
  `seed6_reverse`; the GT-best `seed2_reverse` (**20.96 cm**) ranked 15th.
  Across the 24 fixed-COLMAP-feature candidates, 76 calibrated-config rows and
  53 stable references yielded 69.7% pair (63.8% image) coverage. The same
  equal-unit score selected `seed2` (**2.65 cm** GT), while the GT-best
  `seed11_reverse` was **1.99 cm** (rank 4); translation Huber alone also
  selected `seed2`. Thus no selector was added: the new translation scores are
  near-uncorrelated with own-SIFT basin quality and the minima are not
  consistent across the two families. Existing default behavior and the
  reporting-only rotation selector remain unchanged. Logs are under
  `/tmp/model_xval_translation_{own,fixed}_20260829_*.log`; targeted
  `diagnose_cli_tests` (11), release check, and `git diff --check` pass.

- **Completed-model correspondence cross-validation diagnostic (2026-08-29).**
  Added the default-off `--diagnose-model-score MODEL/images.txt` probe. It
  requires `--import-verified-pairs-file`, exits before matching/reconstruction,
  and scores the complete imported verified correspondence multiset rather
  than only observations retained by tracks. Each registered pair reports
  pose-induced calibrated normalized Sampson residuals, verifier-threshold
  fraction, positive-depth and ≥1° triangulation-angle feasibility, with
  overall, pair-balanced, image-balanced, and deterministic 20%-hash-held-out
  aggregates. When the import contains calibrated E references, the output
  also reports a GT-independent rotation-only disagreement and a guarded
  `model_cross_validation_selection_score` (lower is better, ≥3 shared
  calibrated references; reporting only, no automatic selection).
  Captured controls used the same camera and complete verified set within each
  feature family: own-SIFT (648 pairs / 154,559 correspondences, 49 calibrated
  E references) scored original **31.84 cm** at pair residual/under-depth-angle
  `(0.125241, 0.601859, 0.572281, 0.564049)`, reverse-feature **14.00 cm** at
  `(0.124193, 0.601031, 0.571043, 0.563648)`, coordinate **63.59 cm** at
  `(0.125913, 0.602516, 0.562146, 0.554023)`, and reverse-matches
  **157.04 cm** at `(0.159302, 0.549580, 0.558341, 0.551448)`; their rotation
  selection scores were respectively **10.125477°, 10.068295°, 10.259104°,
  10.569519°**, correctly ordering this small same-set A/B. Fixed COLMAP
  features (401 pairs / 141,698 correspondences, 76 calibrated E references)
  scored legacy **1.76 cm** at **15.323840°**, stable **4.52 cm** at
  **15.333647°**, and cycle **67.07 cm** at **16.776908°**. Raw residual,
  depth/angle, and held-out fractions were nearly indistinguishable within
  the fixed family (e.g. pair-under `0.908607/0.908591/0.908606`), so they do
  not provide a reliable universal basin selector; rotation disagreement is
  retained only as a candidate score when the correspondence multiset and
  calibrated references are shared. Logs are `/tmp/model_xval_*_final_20260829.log`.

- **Post-verification union traversal decomposition (2026-08-29).** Added the
  default-off diagnostic `--union-traversal-order
  original|reverse-pairs|reverse-matches|reverse-both` to reverse pair and/or
  accepted-match iteration without changing verified contents or feature
  indices. The focused unit test confirms default identity, equal
  correspondence counts, and an identical unordered edge multiset hash. On
  the reproducible in-process max-orientations=2 input (**208,746** rows), all
  four runs had **639/703** verified pairs, **154,854** inliers, the same
  unordered-edge FNV-1a hash **`f1073f360f8e8f1a`**, and the same track-build
  summary (**25,798** components, **1,500** conflicts, **24,298** retained
  tracks / **78,425** observations). `original` and `reverse-pairs` were
  byte-equivalent downstream: growth **37/38**, post-registration completed
  the set, **15,992/54,303** final tracks/observations, **0.298 px**, and
  **31.84 cm** Sim(3) RMSE. Reversing accepted matches, with or without pair
  reversal, produced the same other basin: growth **37/38**, post registered
  the remaining camera, **15,169/51,485**, **0.318 px**, and **157.04 cm**;
  the pair-only reversal had no measurable effect. Thus the edge-set hash is
  invariant, but legacy downstream traversal is strongly match-order
  sensitive. Logs/artifacts are under
  `/tmp/repro_31p84_traversal_sift_{original,reverse_pairs,reverse_matches,reverse_both}_20260829.log`
  and the corresponding `_artifact_20260829` model directories.
  The historical external reverse-feature **14.00 cm** accepted set was then
  dumped with its verifier E matrices (**648** pairs / **154,559** matches),
  and every feature file was verified to be the exact row reversal of the
  source file; the physical coordinate-level match multiset was identical
  after `rev_index → count-1-rev_index` remapping. On the fixed reverse-row
  input, imported `original` reproduced **38/38, 16,057 tracks, 0.331 px,
  14.00 cm**, while `reverse-matches` gave **38/38, 14,543 tracks, 0.306 px,
  128.54 cm**. On source-row order after remapping, the same set scored
  **184.01 cm** (original) versus **237.23 cm** (reverse-matches), exposing
  feature/observation order interaction in addition to match traversal.
  These are diagnostic A/B results only; no default or accuracy champion was
  changed. The E-matrix dump is emitted only with the existing pair-outcome
  debug gate so the verified-set replay remains self-contained.

- **Current-source own-SIFT traversal replay (2026-08-29).** Re-extracted
  from the courtyard images (rather than reusing the stale 154,884-row dump)
  with floor grayscale, `--sift-max-keypoints 4096`,
  `--sift-max-orientations 1`, the compatible four-octave/source-level
  detector, corrected VLFeat-compatible descriptor, and bilinear
  orientations. The current source emitted **178,599** feature rows; this
  dump is byte-identical across all 38 files to the prior
  `/tmp/vlfeat_source_latest_floor_o1_20260829` extraction. Its
  **178,361** unique `(x,y)` loci leave exactly **238** duplicate rows
  (**236** duplicate loci, maximum multiplicity 3), and every duplicate row
  is full-row/location/descriptor-identical. Thus the previously reported
  178,361 figure is a deduplicated-locus count, not a missing feature dump.
  The feature export has no octave/level/orientation columns; the compatible
  detector diagnostic associates the duplicate loci with repeated
  `before_orientation` candidates, primarily octaves -1/0 and levels 0--2.
  The critical stems 0299--0308 contain respectively 14, 13, 5, 11, 11,
  10, 6, 9, 4, and 6 duplicate rows. This accounts for the 238-row
  difference and does not implicate grayscale, cap nondeterminism, or input
  ordering. The archived **31.84 cm** command also omitted
  `--sift-max-orientations`; its runtime log records **208,746** loaded rows
  (the compatible-mode effective cap is 2 when the raw value is 0), not the
  178,599 max-1 rows. The max-2 dump has the same 178,361 unique loci and
  adds 30,147 orientation-cap rows. Replaying that
  exact command at `/tmp/repro_31p84_exact_run1_20260829` and
  `/tmp/repro_31p84_exact_run2_20260829` produced the same **208,746**,
  **639/703** verified pairs, **15992** tracks, **0.298 px**, and **31.84 cm**
  score. The latter log records the complete parsed configuration and its
  deterministic raw-Args hash `ce75f7453d787fce`; the extraction-only max-1
  replay records hash `4386796feeb6547c`. The 178,599 max-1 command and the
  208,746 exact command therefore are not effective-config-equivalent; no
  compatible-mode default was changed to force the missing 178,361 count.
  The mapper command used
  `--exhaustive --min-matches 15 --match-ratio 0.9 --guided-matching
  --verification-mode full --mapper incremental --pnp-max-iterations 100000
  --min-pnp-inliers 8 --geometry-guided-conflict-recovery
  --post-refinement-registration --final-iterative-refinement`; original row
  order produced **591/703 verified pairs, 128,124
  inliers; 21,927 components, 1,281 conflicts, 20,646 initial tracks/66,118
  observations; growth 38/38; 11,565 final tracks/39,538 observations,
  0.355 px, 310.44 cm Sim(3) RMSE** (max **569.44 cm**, DSC_0307).
  Reversing every feature-file row (same 178,599 rows) produced
  **571/703, 127,970; 22,003 components, 1,286 conflicts, 20,717 initial
  tracks/66,053 observations; growth 38/38; 13,902/46,114, 0.359 px,
  177.12 cm** (max **394.49 cm**, DSC_0307). Image 20/21 PnP changed from
  **205→85 / 149→74** to **176→73 / 129→65** correspondences/inliers.
  The reverse result is not cm-class, so no `--refine-intrinsics` A/B was
  justified. For the fixed COLMAP-keypoint + corrected-descriptor normal
  matching path (ratio-0.8, no guided matching), the same row reversal gave
  **403/703, 122,234 inliers; 38/38; 17,935 tracks/60,117 observations,
  0.382 px, 86.37 cm** (max **321.18 cm**, DSC_0307), versus the historical
  unpermuted **2.92 cm** control. Thus this is a negative traversal-order
  result; no selector, default, or accuracy champion changed. Artifacts:
  `/tmp/vlfeat_source_latest_floor_o1_20260829`,
  `/tmp/source_latest_floor_o1_original_20260829`,
  `/tmp/source_latest_floor_o1_reverse_20260829`, and
  `/tmp/order_fixed_reverse_full_20260829`.

- **Historical max-orientations=2 feature-order A/B replay (2026-08-29).**
  Starting from the exact **208,746-row** max-2 feature extraction
  `/tmp/vlfeat_source_latest_floor_20260829`, generated only two temporary
  row orders: per-image `(x,y)` lexicographic order and complete reverse
  order. Both runs used the same staging path and the same legacy
  `UnionFind` track builder; all experimental ordering/track flags remained
  off. Because an external permutation is required, the replay uses
  `--feature-extractor files --features-dir /tmp/repro_31p84_order_input_20260829`
  while retaining the historical camera, ratio/guided/full-verification,
  PnP-100k/min-8, recovery+post+final settings. The effective-config snapshot
  hash is **`cda9f5bb8ebf6815` for both orders**; only the staged feature-row
  contents differ (the original in-process SIFT command's raw-Args hash is
  `ce75f7453d787fce`). Coordinate order produced **644/703 verified pairs,
  154,123 inliers; 25,883 components, 1,477 conflicts, 24,406 initial
  tracks/78,860 observations; growth 36/38; post registered 0306/0307 as
  225→113 / 187→65; final 16,664 tracks/55,524 observations, 0.307 px,
  63.59 cm** Sim(3) RMSE (max **290.15 cm**, DSC_0307). Reverse order
  produced **648/703, 154,559; 25,945 components, 1,479 conflicts, 24,466
  tracks/78,980 observations; growth 38/38 without post completion; image
  20/21 PnP 238→90 / 177→76; final 16,057/53,668, 0.331 px, 14.00 cm**
  (max **68.45 cm**, DSC_0316). The reverse result is a material improvement
  over the 31.84 cm historical order but remains above the 10 cm threshold,
  so no intrinsics-refinement A/B was run. A second reverse replay had the
  same **648/703, 154,559, 38/38, 16,057 tracks, 0.331 px, 14.00 cm**
  result and byte-identical `cameras.txt`, `images.txt`, and `points3D.txt`
  (artifacts `/tmp/repro_31p84_reverse_artifact_20260829` and
  `/tmp/repro_31p84_reverse_repeat_artifact_20260829`; hashes
  `a2132068…e5b958`, `b8d4afa…346b0`, `60e0ec…5cf8105`). Coordinate and
  reverse artifacts are `/tmp/repro_31p84_coordinate_artifact_20260829` and
  the logs are `/tmp/repro_31p84_coordinate_20260829.log`,
  `/tmp/repro_31p84_reverse_20260829.log`, and
  `/tmp/repro_31p84_reverse_repeat_20260829.log`.

- **Legacy traversal-order decomposition on all-floor 4-octave bilinear SIFT (2026-08-29).**
  Reused `/tmp/vlfeat_detector_features_20260829` (154,884 fixed
  all-floor/VLFeat-compatible-detector + bilinear-orientation rows) and the
  same exhaustive ratio-0.9/guided/full-verification, plain incremental,
  PnP-100k/min-8, recovery+post+final stack for the original, coordinate,
  reverse, and three fixed Blake2b hash row orders. The current-tree control
  was **603/703 verified, 125,125 inliers, 18,005 retained tracks/59,840
  observations after 1,175 conflicts**, growth **35/38**, then post completed
  **38/38**, **11,533 final tracks/38,957 observations, 0.371 px, 142.99 cm**
  Sim(3). The other orders were:

  | order | verified/inliers | components/conflicts/retained tracks/obs | growth→post | final tracks/obs | reproj | Sim(3) RMSE |
  |---|---:|---:|---:|---:|---:|---:|
  | coordinate | 623/125,140 | 19,043/1,173/17,870/59,548 | 36→38 | 12,142/41,675 | 0.366 px | 46.17 cm |
  | reverse | 613/125,334 | 19,172/1,135/18,037/60,051 | 35→38 | 12,817/43,877 | 0.353 px | 9.65 cm |
  | hash0 | 602/125,238 | 19,158/1,177/17,981/59,820 | 35→38 | 9,026/33,422 | 0.497 px | 393.76 cm |
  | hash1 | 603/125,036 | 19,179/1,160/18,019/60,098 | 38→38 | 11,017/38,191 | 0.969 px | 101.87 cm |
  | hash2 | 607/125,279 | 19,029/1,180/17,849/59,259 | 36→38 | 11,026/38,515 | 0.373 px | 118.49 cm |

  Each full run remained 38/38, but took **210.29 s** for the control and
  **374.72--551.93 s** for the permutations (peak RSS **1.10--1.19 GB**).
  Final track count/reprojection roughly ranked the 9.65 cm reverse run, but
  the pre-growth metrics did not: reverse had fewer conflicts and the most
  retained observations, while hash1 had the most retained observations and
  still scored 101.87 cm; coordinate had the most verified pairs but scored
  46.17 cm. The fixed-COLMAP raw-import control is stronger negative evidence:
  the exact same **161,301** raw correspondences and identical verifier counts
  produced legacy **1.76/3.43/5.03/4.86 cm** across original/coordinate/
  reverse/hash orders, while stable ordering made all permutations **4.52 cm**.
  Thus a pre-growth support/reprojection selector cannot be justified across
  both feature sets; selecting the best candidate would require replaying the
  order-sensitive growth/triangulation/PnP path (or using GT), contrary to the
  bounded selector goal. No multi-order selector, default change, or new
  champion was added. Existing historical all-floor 4-octave bilinear
  **31.84 cm** and fixed raw **1.76 cm** controls remain documented; the
  current-tree rerun is reported separately rather than silently replacing
  either control.

- **Opt-in cycle-supported track construction (2026-08-29).** Added
  `IncrementalSfmConfig::cycle_supported_tracks` and the demo flag
  `--cycle-supported-tracks`. For each accepted match edge, the strategy
  counts exact third-view feature cycles and distinct supporting images, then
  resolves same-image conflicts in that order with calibrated-E residual and
  pair-support/stable-physical tie-breaks. Legacy, stable, and canonical modes
  remain unchanged and the new strategy is default-off. Synthetic tests cover
  a supported three-view edge beating an unsupported conflict, feature/pair
  permutation invariance, and deterministic no-cycle fallback.
  On the fixed COLMAP-keypoint/corrected-descriptor raw-import control
  (404/703 verified pairs, **140,842** inliers), original/coordinate/reverse/
  hash feature orders were identical under this strategy: **38/38**, **19,147**
  final tracks, **63,402** observations, **0.379 px**, and **67.07 cm** Sim(3)
  center RMSE. Initial cycle construction retained **25,900** tracks and
  **90,350** observations from **26,342** components with zero same-image
  conflicts. Each run took **20.12--20.75 s** with peak RSS
  **265,088--265,408 KB**. This is permutation-invariant but regresses the
  raw-import controls (legacy **21,487 tracks/1.76 cm**, stable
  **19,929/4.52 cm**), so no default or accuracy champion changed; ordinary
  matching and full split/all-floor sweeps were not repeated after this
  negative isolation result. Existing split and 4-octave bilinear controls
  remain as previously recorded.

- **Feature-index/order sensitivity and opt-in physical ordering (2026-08-29).**
  A fixed corrected-descriptor/COLMAP-keypoint input (208,785 rows) was
  permuted per image by coordinate order, reverse order, and a fixed hash
  shuffle. Ordinary NN matching remained 38/38 but produced **398/703,
  403/703, and 411/703** verified pairs, with **7.29/86.37/15.90 cm**
  Sim(3) center RMSE and **18,982/17,935/18,149** tracks; the historical
  unpermuted control was **2.92 cm**. Exact raw-import correspondence files
  (all **161,301** remapped rows validated) held verification at **404/703,
  140,842** inliers for every order, yet final tracks/scores still varied:
  unpermuted **21,487/1.76 cm**, coordinate **20,782/3.43 cm**, reverse
  **19,702/5.03 cm**, and hash **19,927/4.86 cm**. This isolates a
  downstream index/traversal dependency in track construction, triangulation,
  and PnP rather than only matcher tie-breaking.
  Added default-off `IncrementalSfmConfig::stable_track_order` and the demo's
  `--canonical-feature-order`; the latter canonically reorders feature rows
  by quantized physical coordinates plus descriptor contents and remaps
  imported raw/verified indices, while stable track ordering canonically sorts
  observations/tracks/conflicts. With `--stable-track-order`, all four exact
  raw-import permutations became identical (**404/703, 140,842 inliers,
  38/38, 19,929 tracks, 0.318 px, 4.52 cm**); canonical feature order gave
  the same result. Canonical ordinary matching on the original/coordinate/
  reverse inputs likewise matched exactly at **398/703, 122,132 inliers,
  18,717 tracks, 0.288 px, 6.37 cm**. Deterministic permutation tests pass,
  but the opt-in path is not an accuracy champion and defaults remain
  unchanged; no broad canonical full sweep was accepted as a new result.
  Separately, the detector/DB metadata join matched **196,677/208,746
  (94.22%)** own rows, with median per-image rank correlation about **0.998**
  and weighted inversion fraction **0.327%** (critical 0305/0306/0307:
  **0.9971/0.9973/0.9968**), so row ranking is largely aligned and does not
  explain the large reconstruction variance by itself. Existing split and
  all-floor/bilinear controls are unchanged. Targeted release checks and
  permutation tests pass; no default reconstruction behavior or champion was
  changed.

- **Split COLMAP-detector / legacy-descriptor SIFT diagnostic (2026-08-29).**
  Added the default-off `--sift-split-colmap-detector-grayscale` path. It
  detects/localizes/orients once on COLMAP's rounded grayscale conversion, then
  describes those exact keypoints from the legacy floor grayscale; it requires
  both compatible SIFT modes, rejects the all-rounded flag, preserves keypoint
  order/counts, and leaves the ordinary path unchanged. The focused
  `.8+cross-check` raw/accepted counts for DSC_0305--0306, 0305--0307, and
  0306--0307 were **314/264, 150/100, 580/539**; coordinate-mapped overlap
  with COLMAP raw matches was **248/353 (79.23% precision, 70.25% recall),
  100/161 (66.67%, 62.11%), and 523/630 (90.17%, 83.02%)**. At identical
  rounded detector loci, split descriptors versus all-rounded descriptors had
  aggregate cosine **0.999891** (median **0.999949**); against COLMAP rows the
  split descriptor cosine was **0.942506** (12,498 mapped rows; median
  **0.944755**), so the grayscale choice changes descriptors only marginally.
  The established full safe stack (compatible detector/descriptor, bilinear
  orientations, exhaustive `--match-ratio 0.9 --guided-matching`, full
  verification, `--pnp-max-iterations 100000`, `--min-pnp-inliers 8`, geometry
  recovery, post registration, final iterative refinement) remained **38/38,
  15,472 tracks, 0.342 px**, but scored **234.13 cm** Sim(3) center RMSE;
  this is better than all-rounded grayscale (**382.21 cm**) but worse than the
  legacy-floor control (**31.84 cm**). Thus split preprocessing is a useful
  controlled decomposition, not an accuracy champion or default change.
  Regression coverage includes detector metadata/descriptors equality and CLI
  validation; release example tests passed (**15/15**).

- **Opt-in COLMAP input grayscale parity probe (2026-08-29).** Audited
  `colmap_oracle_full/database.db`: its 38 image names and 1600×1066 camera
  dimensions match the RGB 8-bit PNGs in `images_1600x1066` (those files have
  no EXIF/profile metadata; the DB itself does not record the source path).
  COLMAP's `Bitmap::CloneAsGrey` evaluates float32
  `.2126f*R+.7152f*G+.0722f*B+.5f` then truncates to `uint8_t`, whereas the
  existing `image` 0.25.5 path uses the same integer coefficients and floors
  the result. On **64,812,800** pixels, COLMAP-round minus legacy-floor was
  **0 or +1** (max **1**; +1 in **31,109,482 / 64,812,800 = 47.99898%**;
  mean absolute **0.47999**, RMS **0.69281** gray levels). Added the
  default-off `--sift-colmap-compatible-grayscale` loader and exact RGB/RGBA
  rounding/alpha regression coverage; ordinary image loading is unchanged.
  A callable upstream VLFeat PGM probe and the compatible detector agreed on
  the rounded grayscale candidates (0305/0306/0307: **4598/3113/3064** vs
  **4598/3113/3066** Rust candidates), and the locus join to COLMAP reached
  **99.87%/99.84%/99.89%** of own detections. However, this is not an
  accuracy improvement: focused ratio-0.8+cross-check raw/accepted became
  **309/258, 152/102, 585/539** (0305--0306, 0305--0307, 0306--0307), with
  mapped COLMAP-raw overlap **246/353, 102/161, 527/630**. The full compatible
  detector+descriptor+bilinear safe stack
  (`--exhaustive --match-ratio 0.9 --guided-matching --verification-mode full
  --pnp-max-iterations 100000 --min-pnp-inliers 8
  --geometry-guided-conflict-recovery --post-refinement-registration
  --final-iterative-refinement`) remained **38/38**, but yielded **14,934
  tracks, 0.406 px** and **382.21 cm** Sim(3) center RMSE versus the legacy
  grayscale control's **38/38, 15,992 tracks, 0.298 px, 31.84 cm**. The
  conversion is source-correct but rejected as a default or champion path;
  the remaining discrepancy is downstream descriptor/matching sensitivity.
  Sources: [COLMAP Bitmap](https://raw.githubusercontent.com/colmap/colmap/main/src/colmap/sensor/bitmap.cc),
  [COLMAP SIFT options](https://raw.githubusercontent.com/colmap/colmap/main/src/colmap/feature/sift.h),
  and the [image crate conversion](https://github.com/image-rs/image/blob/0.25.5/src/color.rs).

- **VLFeat/COLMAP detector-metadata decomposition and bilinear orientation mode (2026-08-29).** A deterministic one-to-one oracle join of the VLFeat-compatible detected features to the six-column COLMAP keypoints used `d(x,y) <= 1.0 px` and `|log(sigma_own/sigma_colmap)| <= log(1.25)`. It matched **4,871/5,483 (88.84%)** DSC_0305, **3,342/3,782 (88.37%)** DSC_0306, and **3,188/3,626 (87.92%)** DSC_0307 detections; matched spatial-error medians were **0.0292/0.0311/0.0323 px** and p95 values **0.1561/0.1949/0.1863 px**, while scale-log medians were **0.00381/0.00371/0.00368**. Re-described focused variants (A own metadata, B own xy+COLMAP scale/orientation, C COLMAP xy+own scale/orientation, D own xy/scale+COLMAP orientation, E own xy/orientation+COLMAP scale, F all-COLMAP metadata through our descriptor) showed that xy/scale substitutions were effectively neutral; orientation substitution raised mapped descriptor cosine from **0.8883** (A/C/E aggregate) to **0.9428/0.9429** (D/F), but spatial-only ties can cross co-located orientation rows, so this is not a shippable oracle result. At ratio **0.8 + cross-check**, mapped own-vs-COLMAP raw-match overlap was A **649/946 = 68.6% precision, 56.7% recall**, B **791/936 = 84.5%, 69.1%**, D **786/943 = 83.4%, 68.7%**, and F **871/1,040 = 83.8%, 76.1%**; at ratio **0.9**, A was **753/1,501 = 50.2%, 65.8%**, B **927/1,484 = 62.5%, 81.0%**, and F **1,028/1,683 = 61.1%, 89.9%**. These are diagnostic joins, not accuracy claims or oracle dependencies.

  The authoritative source audit found COLMAP's vendored `src/thirdparty/VLFeat/sift.c` defines `VL_SIFT_BILINEAR_ORIENTATIONS 1` (the vendored define at source line **669**, with the half-bin accumulation around **1636–1645**), whereas upstream VLFeat defaults to nearest-bin orientation accumulation ([COLMAP source](https://github.com/colmap/colmap/blob/main/src/thirdparty/VLFeat/sift.c), [upstream source](https://raw.githubusercontent.com/vlfeat/vlfeat/master/vl/sift.c)). Added default-off `--sift-vlfeat-bilinear-orientations` (requires `--sift-vlfeat-compatible-detector`), with source-equivalent half-bin/circular interpolation, mass-conservation coverage, and default-identity tests. On focused DSC_0305–0307 pairs, bilinear mode produced `.8` raw/accepted **349/294, 156/98, 626/570** and coordinate-mapped oracle overlap **788/964 = 81.7% precision, 68.9% recall**, versus own-metadata A **649/946 = 68.6%, 56.7%**. The full safe stack (`--sift-vlfeat-compatible-detector --sift-vlfeat-compatible-descriptor --sift-vlfeat-bilinear-orientations --exhaustive --match-ratio 0.9 --guided-matching --verification-mode full --pnp-max-iterations 100000 --min-pnp-inliers 8 --geometry-guided-conflict-recovery --post-refinement-registration --final-iterative-refinement`) reached **38/38, 15,230 tracks, 0.351 px, 109.74 cm Sim(3) RMSE**; fair current source-cap A (`/tmp/vlfeat_detector_corrected_4096_features_20260829`) reached **38/38, 14,988 tracks, 0.387 px, 276.42 cm**, so bilinear is a material opt-in improvement but neither displaces the COLMAP-feature champion (**38/38, 2.84 cm**) nor proves descriptor parity. The critical-only metadata substitution control reached **38/38, 14,137 tracks, 0.375 px, 290.59 cm**, a negative result despite its focused overlap gain. Artifacts: `/tmp/vlfeat_metadata_variants_20260829`, `/tmp/vlfeat_bilinear_focus.csv`, `/tmp/vlfeat_bilinear_full_20260829.log`, `/tmp/vlfeat_current_A_full_20260829.log`.

- **VLFeat/COLMAP orientation-set parity audit (2026-08-29).** A follow-up
  unordered circular-angle assignment at strictly matched
  detector loci found that bilinear mode agrees with COLMAP's orientation
  multiplicity at **9,702 loci / 11,303 orientation rows** across
  DSC_0305–0307 (own rows: **5,467/3,780/3,631**; COLMAP rows:
  **5,354/3,647/3,508**). Bilinear equal-cardinality matches were
  **4,138/2,809/2,755** loci (single-orientation: **3,454/2,309/2,340**;
  two-orientation: **684/500/414**; four-orientation: **0/0/1**), with only
  **2/3/1** two-orientation source-order swaps. Minimum-cost circular angle
  errors were median **0.185/0.206/0.202°**, aggregate p90 **0.663°**, p95
  **1.002°**; the remaining **16** loci with >10° errors are localized
  peak/multiplicity mismatches, not a systematic angle sign or row-order
  error. The nearest-bin compatible dump had aggregate median **0.543°** and
  p95 **2.006°**, confirming the COLMAP bilinear switch as the material
  orientation discrepancy. Source audit matches the vendored implementation:
  `winf=1.5`, `W=floor(3*sigmaw)`, gradients from the current DoG Gaussian
  layer, six circular three-tap smoothing passes, strict `>0.8*maxh` local
  maxima, bounded quadratic interpolation, bin-ascending return order with
  at most four peaks, then COLMAP's first-two cap and whole-DoG-level global
  cap (vendored `sift.c` lines **1446–1529**, **1563–1691**; COLMAP wrapper
  lines **242–258**, **294–331**). The independent reference-vector,
  circular-boundary, peak-order, cap, and default-identity tests pass; no
  further orientation mode or oracle dependency was added.

- **VLFeat detector numeric and unmatched-locus audit (2026-08-29).** The
  vendored implementation stores `vl_sift_pix` as `float` and uses float
  separable convolution, upsampling, and DoG subtraction, so a focused f64
  versus f32 probe was run against that contract before considering a numeric
  compatibility mode. Across the 17 critical stems, strict detector-locus
  matches were **72,760** (f64) versus **72,759** (f32), own-only loci were
  **6,890** versus **6,886**, and response/edge-score distributions were
  unchanged. On DSC_0305--0307, ratio **0.8 + cross-check** produced the same
  **349/294, 156/98, 626/570** raw/accepted counts; ratio **0.9** changed only
  0305--0306 by **606/415** (f64) versus **605/417** (f32), with the other two
  pairs unchanged. Thus f32 arithmetic does not explain the remaining
  detector gap, and no numeric-compatibility flag was shipped.

  The pre-fix all-image metadata join (the image-size-derived eight-octave
  detector with a 4096 source-level cap) found **166,001** strict matched
  loci, **13,896** own-only loci,
  and **12,359** COLMAP-only loci. Re-running the join before the global cap
  showed that **3,014 (24.4%)** of the COLMAP-only loci had an own candidate
  removed by capping/ranking, while **9,345 (75.6%)** had no own candidate even
  in the unbounded extrema set. Critical bridge images 0305/0306/0307 had
  essentially no cap loss (detector candidate mismatch dominated). Own-only
  points were not border artifacts (border-distance median **203 px** versus
  **203 px** for matched points), had comparable orientation multiplicity,
  but were closer to the edge-test limit (edge-score median **6.65** versus
  **5.67**) and had larger nearest-locus displacement (median **7.04 px**).
  The audit TSV now includes deterministic `edge_score` metadata for these
  classifications. The source arithmetic and extrema audit follows COLMAP's
  vendored `sift.c` (`vl_sift_pix`, `_vl_sift_smooth`, upsample, DoG, and
  strict extrema comparisons); the next bottleneck is source-level detector
  candidate/localization parity, not a blanket precision change.

- **VLFeat/COLMAP octave-count parity and pyramid probe (2026-08-29).** The
  source audit found a larger semantic mismatch than arithmetic precision:
  COLMAP's `SiftExtractionOptions` defaults to `first_octave=-1` and
  `num_octaves=4`, while the compatible Rust path interpreted `octaves=0` as
  the image-size-derived eight-octave limit on the 1600×1066 courtyard. The
  extra coarse octaves changed the whole-DoG-level cap and created own-only
  detections at octaves **3--6**. A deterministic 64×48 impulse+ramp/noise
  probe compared the current f64 pyramid formulas with a callable VLFeat
  (`vl_sift_process_first_octave/process_next_octave`) reference: Gaussian
  levels differed by at most **1.91e-7** (RMS at most **7.33e-8**), DoG levels
  by at most **1.90e-7** (RMS at most **4.02e-8**), and the refined extrema
  sets were identical at every tested level (maximum x/y/σ disagreement
  **7.5e-5** from the reference's float storage). Thus upsampling, boundary
  extension, kernel support, subtraction order, and strict 3×3×3 extrema
  semantics are numerically consistent; no f32 mode was justified.

  The compatible detector now uses the source-compatible default of four
  octaves when `SiftConfig::octaves=0`; explicit octave counts remain honored,
  and the legacy detector is untouched. On the all-image metadata join, strict
  matched loci improved **166,001→168,285**, own-only fell
  **13,896→10,076**, COLMAP-only fell **12,359→10,075**, and cap-attributed
  COLMAP-only fell **3,014→730**; the remaining **9,345** detector-only loci
  are unchanged and all COLMAP-only rows infer to octaves **-1..2**. Per-level
  detector-only counts are largest at **(-1,2)=2,378**, **(0,0)=1,825**, and
  **(-1,0)=1,211**; own-only points have response median **0.0127** and
  edge-score median **7.59** versus matched **0.0204** and **5.66**, so they
  are not simply low-response border artifacts. Focused ratio-0.8/cross-check
  raw/accepted rows became **318/265**, **145/82**, and **582/526** for
  0305--0306, 0305--0307, and 0306--0307 respectively (a matching-density
  tradeoff despite better locus parity). The full bilinear compatible safe
  stack reached **38/38, 15,992 tracks, 0.298 px, 31.84 cm Sim(3) RMSE**,
  improving the prior eight-octave result (**109.74 cm**) but remaining well
  above the COLMAP-feature champion (**2.84 cm**). Source references are
  COLMAP's vendored `sift.c` process/detect routines (process-first
  **977--1059**, process-next **1083--1134**, DoG/extrema **1153--1232**)
  and `src/colmap/feature/sift.h` defaults (**41--55**); the new octave-count
  regression test and existing detector tests pass.

- **BA numerical Jacobian and linear-system audit (2026-08-29).** Added the
  default-off `VISLOC_SFM_DEBUG_BA_JACOBIANS=1` diagnostic (with
  `VISLOC_SFM_DEBUG_BA=1`) and focused tests for the right-perturbation pose,
  world-landmark, and pinhole-intrinsics Jacobians. Central differences on
  synthetic normal, far-depth, low-parallax, and high-residual cases stayed
  below **4.9e-8 px absolute** (the far-depth landmark relative error was
  **4.32e-7**); the real 27-camera state audited 64 observations with no
  invalid samples. Its largest absolute/relative errors were pose
  **1.63e-7 / 1.32e-10**, landmark **2.48e-7 / 2.48e-5**, and intrinsics
  **1.41e-10 / 9.78e-11** (the larger relative landmark values occur on
  near-zero far/low-parallax columns). A Schur-complement solve was compared
  with the full 9×9 normal system (**<1e-10** difference), and a Huber
  derivative test confirms `weight(s)=rho'(s)` (**<1e-8** finite-difference
  error). The audit confirms the residual sign (`prediction-measurement`),
  right-pose convention, Schur RHS signs, caller-owned gauge anchors, and
  identity LM damping; no corrected solver or diagonal-scaling option was
  justified by these checks. The new `VISLOC_SFM_DEBUG_BA_STEPS=1` detail
  line explains the scoped 27-camera point-only warm-start stall: all five
  trials failed the objective gate (feasibility passed), with
  **6.540293608e6→4.069881100e7–4.075306232e7** candidate cost and
  nonprojectable observations **37→35**. This is a nonlinear/weak-geometry
  step rejection, not an analytic-Jacobian defect; default reconstruction
  behavior is unchanged. In a captured full-raw-import 27-camera state, the
  separated max absolute/relative errors (bucket order normal/far/low/high)
  were translation **1.11e-7/1.16e-9, 1.11e-7/1.56e-8, 1.50e-7/2.59e-5,
  1.14e-7/2.76e-5**, rotation **1.24e-7/8.36e-11, 9.67e-8/7.77e-11,
  1.10e-7/8.63e-11, 1.02e-7/7.64e-11**, landmark **1.73e-7/1.35e-9,
  9.86e-8/2.54e-8, 1.08e-7/2.16e-5, 1.01e-7/2.69e-5**, and intrinsics
  **8.30e-11/7.08e-11, 7.94e-11/6.24e-11, 1.34e-10/1.00e-10,
  9.59e-11/8.06e-11** (64 samples, zero invalid). Reproducible traces:
  `/tmp/ba_step_detail_20260829.log` and
  `/tmp/ba_jacobian_27_import_20260829.log`.

- **Point-only landmark BA warm-start A/B (2026-08-29).** Added the
  default-off `--landmark-ba-warm-start-iterations N` path, with optional
  `--landmark-ba-warm-start-min-registered-images N` scoping. Before each
  global/periodic joint BA it builds the same robust reprojection problem with
  all registered poses and intrinsics fixed, optimizes only landmarks, and
  copies points back only after finite/non-increasing-cost checks; support and
  camera states are otherwise untouched. The synthetic test covers camera
  immutability, cost decrease, deterministic output, and the `0` no-op.
  On the COLMAP-feature champion (`--landmark-ba-warm-start-iterations 5`),
  applying the warm start to every global/periodic solve changed the early
  growth path immediately: the first 7-camera point-only solve accepted all 5
  steps and moved a point by **1.675e5 m**. The 27-camera joint BA then had
  oracle centre RMSE **12.7896 cm** (historical schedule **12.2241 cm**), and
  the full run ended at **38/38, 20,936 tracks, 0.373 px, 37.47 cm** versus
  the champion's **38/38, 21,338 tracks, 0.283 px, 2.842 cm**. An evidence-
  scoped run with `--landmark-ba-warm-start-min-registered-images 27` left
  the target 27-camera solve unchanged (point-only cost
  **6.540293608e6→6.540293608e6**, zero accepted steps), but later warm starts
  still yielded **21,325 tracks, 0.283 px, 2.85 cm**; its oracle-pose control
  scored **1.98 cm** versus the existing **1.95 cm** control, with probe
  reprojection **1.425801→0.275717 px**. Thus the point-only solve does not
  safely absorb the ill-conditioned movement and remains opt-in; the default
  schedule and accuracy champion are unchanged. Trace/score artifacts are
  `/tmp/landmark_warm5_trace_20260829.log`,
  `/tmp/landmark_warm5_score_20260829.log`,
  `/tmp/landmark_warm5_min27_trace_20260829.log`, and
  `/tmp/landmark_warm5_min27_score_20260829.log`.

- **BA landmark-conditioning diagnosis and opt-in safeguard (2026-08-29).**
  Added `VISLOC_SFM_DEBUG_BA_LANDMARKS=1` diagnostics for the fixed pre-BA
  landmark geometry: track length, baseline/depth ratio, widest
  triangulation angle, robust reprojection residual, a weighted 3x3 point
  Hessian condition proxy, and post-solve displacement. In the COLMAP-feature
  champion's 27-camera solve, only **118/15,799** landmarks had widest angle
  below the existing **2°** triangulation gate, but they accounted for
  **84.5885%** of total landmark displacement; the **76** points with proxy
  condition above **1e8** accounted for **40.6845%**. The largest movement was
  **9.574e3 m** (track 2194, angle **0.0195°**, baseline/depth
  **6.57e-4**, median residual **227.5 px**), confirming that a small,
  numerically weak and already-misfitting subset drives the basin jump.
  Added default-off `--freeze-ill-conditioned-landmarks`; despite its name,
  the guarded implementation omits only ill-conditioned landmarks whose
  fixed pre-BA residual is already outside the ordinary reprojection gate,
  so their bad residual rows cannot pull camera variables. Retaining those
  rows while freezing the points worsened the 27-camera oracle RMSE to
  **37.28 cm**. The guarded exclusion reduced the traced 27-camera RMSE to
  **9.1224 cm** and excluded **213 landmarks / 1,321 observations** there,
  but its full non-oracle run changed the final support and quality to
  **38/38, 21,541 tracks, 0.348 px, 7.39 cm**, versus the champion's
  **38/38, 21,338 tracks, 0.283 px, 2.842 cm**. The flag therefore remains an
  opt-in diagnostic, with the historical default and accuracy champion
  unchanged. Focused deterministic conditioning tests, CLI parsing tests,
  and release example compilation pass; trace artifacts are
  `/tmp/ba_landmark_condition_trace3_20260829.log`,
  `/tmp/ba_landmark_conditional_trace_20260829.log`, and
  `/tmp/ba_landmark_conditional_score_20260829.log`.

- **Periodic BA basin-jump diagnosis and deferred-schedule A/B (2026-08-29).**
  Added default-off `--periodic-ba-min-registered-images N` for the plain
  incremental schedule (`0` preserves the historical `ba_every` behavior; the
  knob does not suppress the configured final BA). With
  `VISLOC_SFM_DEBUG_BA=1`, the mapper now
  reports pre/post support, robust and L2 costs, LM trial costs/damping and
  acceptance, gauge anchors, camera/landmark displacement, and explicit
  filter/re-triangulation support transitions; `VISLOC_SFM_DEBUG_BA_STEPS=1`
  enables the per-trial stream. On the COLMAP-feature champion seed `(8,9)`,
  the 27-camera solve was confirmed as the basin jump **after** BA and before
  any pruning: support stayed **15,799 tracks / 57,794 observations**, Huber
  cost was **6.540293608e6→4.697525563e6**, with **5 accepted / 15 rejected**
  LM trials, final `lambda=1e6`, max camera displacement **0.916 m**, max
  landmark displacement **9.57e3 m**, and anchors `(0,30)`; oracle-aligned
  centre RMSE changed **3.7776→12.2241 cm**. The evidence-derived threshold
  `N=32` defers that 27-camera solve until the next five-camera connectivity
  boundary, but its first 32-camera solve enters a different basin (oracle
  RMSE **14.8959 cm** at that point). The controlled A/B used the same
  exhaustive/full-verification, bridge-supplement, `pnp-max-iterations=100000`,
  recovery/post/final-refinement configuration: historical schedule produced
  **38/38, 21,338 tracks, 0.283 px, 2.842 cm**, while `N=32` produced
  **38/38, 20,435 tracks, 0.301 px, 5.787 cm**. Thus deferral is a useful
  reproducible diagnostic but a negative accuracy result; no fixed-keypoint
  follow-up was run, and the option remains opt-in. Trace logs/models are
  `/tmp/ba_schedule_off_trace2_20260829(.log)` and
  `/tmp/ba_schedule_defer32_trace_20260829(.log)`; score runs are
  `/tmp/ba_schedule_off_score_20260829` and
  `/tmp/ba_schedule_defer32_score_20260829`.

- **Registration-time pose-drift diagnostic (2026-08-29).** Added the
  default-off `sfm-debug-oracle` transition log, enabled by the existing
  `--diagnose-ba-oracle-poses` input together with `VISLOC_SFM_DEBUG=1`, and
  `sfm-debug-pnp-geometry` for selected images (`VISLOC_SFM_DEBUG_IMAGES`).
  On the COLMAP-feature champion seed `(8,9)`, the first persistent drift was
  not PnP: after image **11**'s **434→262** PnP (track-length median **5→4**,
  parallax **20.649°→17.548°**, conditioning **2.84→3.31**, reprojection
  median **2.528→1.562 px**), the following periodic BA changed aligned
  centre RMSE **3.7776→12.2241 cm** (**+8.4465 cm**), with worst images
  **36=31.08 cm, 23=28.69 cm, 11=23.76 cm**.  The failed growth attempts
  were image **20: 22→7** inliers (median parallax **2.695°→2.482°**,
  conditioning **13.34→23.09**, reprojection **12.199→2.606 px**) and image
  **21: 6→3** inliers (median parallax **6.464°→7.107°**, conditioning
  **8.08→8.08**).  An offline, deterministic high-information subset pose
  refinement was mixed: image 11 worsened **7.683→7.965 cm**, image 12
  **12.121→13.936 cm**, image 13 improved **42.031→39.543 cm**, image 17
  **40.095→39.707 cm**, and image 22 **39.428→36.605 cm**; post-refinement
  image 20 improved only **3.967→3.883 cm**, while image 21 improved
  **169.647→132.466 cm** but remained grossly wrong.  Therefore no
  `geometry_weighted_pnp` policy was enabled: the evidence does not provide
  a safe GT-independent pose refinement, and the default mapper is unchanged.
  Exact focused command (output `/tmp/pnp_oracle_trace_focus_20260829`) was:
  `env VISLOC_SFM_DEBUG=1 VISLOC_SFM_DEBUG_BA=1 VISLOC_SFM_DEBUG_IMAGES=11,12,13,17,20,21,22 target/release/examples/unordered_sfm_demo --feature-extractor files --features-dir /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_features_export --feature-suffix _features.txt --image-suffix .png --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 --import-matches-supplement-file /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_bridge_matches_import.txt --exhaustive --min-matches 20 --match-ratio 0.8 --verification-mode full --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8 --geometry-guided-conflict-recovery --post-refinement-registration --final-iterative-refinement --seed-pair 8,9 --diagnose-ba-oracle-poses /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_oracle_full/sparse_txt/images.txt --out-colmap /tmp/pnp_oracle_trace_focus_20260829`.

- **SIFT-scale heteroscedastic BA diagnostic (2026-08-29).** Read the
  six-column COLMAP affine keypoints from
  `/media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_oracle_full/database.db`
  (`sigma = sqrt(det(A))`) and joined them to the observation indices in the
  existing COLMAP-feature champion and oracle-pose-injected controls. The
  champion's scale quartiles (Q1–Q4, 18,523 observations each) had median
  scales **1.578, 2.200, 2.707, 4.792** and median reprojection errors
  **0.133, 0.150, 0.160, 0.294 px** (Q4 p90 **1.076 px** versus Q1 **0.439
  px**); the oracle-pose-injected control was nearly identical (**0.126,
  0.141, 0.157, 0.284 px**). Q4 also had longer tracks (mean **5.55** versus
  **3.80** in Q1) and higher parallax (median **16.71°** versus **8.82°**),
  so the residual elevation is not an isolated localization-noise measure.
  Observation-level `corr(log(sigma), residual)` was only **0.297** (oracle
  **0.300**) and, more importantly, image-level median scale was
  anti-correlated with aligned camera-centre error: **−0.377** for the
  champion versus GT (38/38, **2.842 cm**) and **−0.497** for the oracle-BA
  control versus the actual COLMAP model (**1.658 cm**). The actual COLMAP
  model versus GT was also **−0.754** (**1.709 cm**), with the largest centre
  errors on low-scale bridge images (e.g. DSC_0308 scale **1.401**, error
  **4.132 cm**). Thus scale predicts some high-residual observations but not
  the camera displacement that matters here; no scale metadata/BA-weighting
  mode was added, and the default path is unchanged. The next safe weighting
  probe is track-level residual/geometry quality (with robust whitening), not
  a uniform `1/sigma` or `1/sigma²` sweep.

- **Track-level parallax/information BA diagnostic (2026-08-29).** A
  read-only join of the COLMAP-feature champion's **21,338** triangulated
  tracks / **74,091** observations measured length, widest-baseline
  `sin²(parallax)`, depth-to-baseline conditioning, and reprojection residuals
  before any new weighting. Short tracks are the majority (**10,975** length-2
  tracks; median widest angle **4.956°**, condition **7.95**, information
  **0.007**, median residual **0.109 px**), while length-≥5 tracks were **4,189**
  (median **30.911°**, **1.35**, **1.534**, **0.204 px**). The ill-conditioned
  condition-≥10 bucket contained **4,124** tracks (median angle **3.492°**,
  information **0.004**, median residual **0.095 px**); low parallax is not
  simply a high-pixel-residual bucket. Nevertheless, camera-centre drift in
  the oracle-pose BA control correlated strongly with track geometry: centre
  error versus track count **−0.740**, widest angle **−0.660**, condition
  **+0.624**, and information **−0.473** (actual COLMAP-vs-GT gave **−0.598**,
  **+0.580** for angle/condition). The champion-vs-GT correlations were weaker
  (**−0.317** for information, **+0.338** for median residual), indicating
  mapper-basin effects in addition to geometry.

  Added default-off `--geometry-weighted-ba`, which leaves registration and
  support untouched and runs one final fixed-support BA using pre-solve,
  median-normalized `sin²(widest parallax)` observation weights clamped to
  **[0.25, 4]**; tracks with unavailable geometry get weight 1 and track length
  is not multiplied a second time. On the COLMAP-feature recovery+post stack,
  off (`/tmp/track_geom_colmap_off_20260829`) was **38/38, 21,338 tracks,
  74,091 observations, 0.283 px, 2.842 cm**; on
  (`/tmp/track_geom_colmap_weighted_20260829`) stayed **38/38** with identical
  support but **0.285 px, 2.993 cm**, so it is not an accuracy-champion update.
  The oracle-pose control remained **38/38** and improved from the existing
  ordinary-BA **1.954 cm** (`/tmp/oracle_ba_champion_probe2_20260829`) to
  **1.774 cm** (`/tmp/track_geom_oracle_weighted_20260829`), while mean
  reprojection was **1.397976→0.275904 px** versus **1.539402→0.276868 px**;
  this supports reduced COLMAP-basin drift but does not justify enabling it by
  default. The fixed-keypoint follow-up was skipped because the champion
  quality gate regressed; logs are `/tmp/track_geom_colmap_off_20260829.log`,
  `/tmp/track_geom_colmap_weighted_20260829.log`, and
  `/tmp/track_geom_oracle_weighted_20260829.log`.

- **COLMAP-basin control and seed replay diagnostic (2026-08-29).** Added
  default-off `--diagnose-ba-oracle-poses PATH` and `--seed-pair I,J` controls.
  The former parses the actual COLMAP sparse `images.txt`, Sim(3)-aligns the
  mapper's existing support into that pose frame, and runs one ordinary
  fixed-support BA solve; it never changes the normal mapper path. The actual
  COLMAP sparse model scores **1.71 cm** centre RMSE (38/38), while the
  COLMAP-feature recovery+post champion scores **2.84 cm** (38/38,
  **21,338** tracks, **0.283 px**). Injecting the COLMAP poses with the
  champion's support starts at **1.398 px** mean reprojection and reaches
  **0.276 px** (**2.160243226e5→1.533969354e4** BA cost, 20 iterations), but
  scores **1.95 cm** after BA: lowering reprojection does not preserve the
  1.71 cm COLMAP basin, so the solver/support is a contributor but not the
  sole mapper gap. The control command used
  `env VISLOC_SFM_DEBUG=1 VISLOC_SFM_DEBUG_BA=1 VISLOC_SFM_DEBUG_IMAGES=20,21 target/release/examples/unordered_sfm_demo --feature-extractor files --features-dir /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_features_export --feature-suffix _features.txt --image-suffix .png --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 --import-matches-supplement-file /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_bridge_matches_import.txt --exhaustive --min-matches 20 --match-ratio 0.8 --verification-mode full --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8 --geometry-guided-conflict-recovery --post-refinement-registration --final-iterative-refinement --diagnose-ba-oracle-poses /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_oracle_full/sparse_txt/images.txt --out-colmap /tmp/oracle_ba_champion_probe2_20260829`.
  The source-derived seed replays used the same command/configuration with
  `--diagnose-ba-oracle-poses` omitted and `--seed-pair 7,8`, `7,9`, or
  `10,11`, writing to `/tmp/seed_replay_7_8_20260829`,
  `/tmp/seed_replay_7_9_20260829`, and `/tmp/seed_replay_10_11_20260829`.
  The normal strongest seed `(8,9)` has **2,507** accepted matches, reaches
  **36** before the existing post/refinement completion, and ends at
  **38/38, 21,338 tracks, 0.283 px, 2.84 cm**. The alternatives all reached
  38/38 during growth but were worse: `(7,8)` (**1,364** matches,
  **21,198** tracks, **0.298 px, 4.88 cm**), `(7,9)` (**1,107**, **20,819**,
  **0.406 px, 13.99 cm**), and `(10,11)` (**902**, **20,824**, **0.387 px,
  14.25 cm**). Since the normal seed also wins the GT-independent support /
  reprojection criteria, no automatic multi-seed selector was enabled.

- **Final fixed-support L2 BA polish diagnostic (2026-08-29).** Added the
  default-off `--final-ba-polish-iterations N` path (`0` remains a no-op). It
  runs after registration, filtering/re-triangulation, recovery, and post-
  refinement, rebuilding the same gauge anchors and exact existing support,
  with fixed intrinsics and `RobustKernel::None`; tracks, observations, and
  registration states are never edited. A solve is committed only when all
  state/cost values are finite and its pure-L2 SSE is non-increasing; otherwise
  pose/landmark snapshots are restored. `VISLOC_SFM_DEBUG_BA=1` reports support
  identity, SSE, accepted/rejected steps, convergence, and final damping.
  The COLMAP-feature champion command was
  `env VISLOC_SFM_DEBUG=1 VISLOC_SFM_DEBUG_BA=1 VISLOC_SFM_DEBUG_IMAGES=20,21 target/release/examples/unordered_sfm_demo --feature-extractor files --features-dir /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_features_export --feature-suffix _features.txt --image-suffix .png --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 --import-matches-supplement-file /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_bridge_matches_import.txt --exhaustive --min-matches 20 --match-ratio 0.8 --verification-mode full --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8 --geometry-guided-conflict-recovery --post-refinement-registration --final-iterative-refinement --final-ba-polish-iterations 10 --out-colmap /tmp/ba_champion_polish10_20260829`. It preserved **38/38**, **21,338 tracks**, and **0.283 px**; SSE changed **1.575328576e4→1.574786401e4** in **5/5** accepted steps and converged, but Sim(3) RMSE was **2.85 cm** versus the control's **2.84 cm**. The fixed-keypoint command was
  `env VISLOC_SFM_DEBUG=1 VISLOC_SFM_DEBUG_BA=1 VISLOC_SFM_DEBUG_IMAGES=20,21 target/release/examples/unordered_sfm_demo --feature-extractor files --features-dir /tmp/colmap_fixed_vlfeat_all_l2_20260829 --feature-suffix _features.txt --image-suffix .png --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 --exhaustive --min-matches 15 --match-ratio 0.8 --verification-mode full --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8 --geometry-guided-conflict-recovery --post-refinement-registration --final-iterative-refinement --final-ba-polish-iterations 10 --out-colmap /tmp/ba_fixed_vlfeat_polish10_20260829`. Its polish had **10/10** rejected steps and SSE **2.443825662e4→2.443825662e4**; the contemporaneous no-polish control was byte-identical (**38/38**, **18,506 tracks**, **0.399 px**, **21.67 cm** Sim(3) RMSE). The historical fixed-keypoint reference remains **2.92 cm** in its prior log, but is not reproduced by this current shared worktree; therefore no accuracy champion/default changed. Focused support/determinism/default-no-op tests pass.

- **Bundle-adjustment iteration-cap diagnostic (2026-08-29).** Added the
  default-off `--ba-max-iterations <n>` CLI override (the library/default
  remains **20**) and `VISLOC_SFM_DEBUG_BA=1` solve diagnostics. The logs show
  the existing Huber-`δ=3` LM/Dense-Schur solver, `λ=1e-4` with accepted-step
  decay to `1e-9`, and most solves reaching the cap rather than a convergence
  tolerance. A narrow 40-iteration A/B was negative: the exact COLMAP-feature
  champion command was
  `env VISLOC_SFM_DEBUG=1 VISLOC_SFM_DEBUG_BA=1 VISLOC_SFM_DEBUG_IMAGES=20,21 target/release/examples/unordered_sfm_demo --feature-extractor files --features-dir /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_features_export --feature-suffix _features.txt --image-suffix .png --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 --import-matches-supplement-file /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_bridge_matches_import.txt --exhaustive --min-matches 20 --match-ratio 0.8 --verification-mode full --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8 --geometry-guided-conflict-recovery --post-refinement-registration --final-iterative-refinement --ba-max-iterations 40 --out-colmap /tmp/ba_champion_ba40_20260829`, which kept **38/38** but changed **21,338→20,193 tracks**, mean reprojection **0.283→0.354 px**, and Sim(3) center RMSE **2.84→12.12 cm**. The fixed-keypoint compatible-descriptor command was
  `env VISLOC_SFM_DEBUG=1 VISLOC_SFM_DEBUG_BA=1 VISLOC_SFM_DEBUG_IMAGES=20,21 target/release/examples/unordered_sfm_demo --feature-extractor files --features-dir /tmp/colmap_fixed_vlfeat_all_l2_20260829 --feature-suffix _features.txt --image-suffix .png --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 --exhaustive --min-matches 15 --match-ratio 0.8 --verification-mode full --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8 --geometry-guided-conflict-recovery --post-refinement-registration --final-iterative-refinement --ba-max-iterations 40 --out-colmap /tmp/ba_fixed_vlfeat_ba40_20260829`, which kept **38/38** but changed the known control's **19,898 tracks/0.271 px/2.92 cm** to **18,652 tracks/0.303 px/19.06 cm**. Since 40 iterations degraded both quality paths despite lower robust costs, the override remains an opt-in diagnostic and no accuracy champion/default was changed. CLI tests cover default `None`, positive `40`, and rejection of `0`.

- **Calibrated-essential primary policy (2026-08-29).** Added the separate,
  default-off `--calibrated-essential-primary` strategy for full verification.
  On known-intrinsics `UNCALIBRATED` F-winning pairs it refits the direct E
  RANSAC support with the deterministic eight-point estimator, rescored all
  candidates at the calibrated threshold, requires COLMAP's minimum support,
  at least **50%** of F support, and the source-derived hardened cheirality
  gates (≥**1°** triangulation angle, ≤**0.85** ambiguity, ≥**50%** positive
  depth). A passing E becomes the primary track/translation model and is
  labelled `CALIBRATED`; F/H counts remain in diagnostics. Weak, planar, or
  pure-rotation candidates fall back to the historical F winner. The existing
  strict F→E option can still be combined to use its stability-gated fallback
  only when direct E is not admitted. Default behavior is unchanged; focused
  tests cover F having more support while healthy E is selected and degenerate
  E rejection.

  Verification-only diagnostics kept the VLFeat graph connected (**650/703**
  pairs; all **38** images) and promoted **57** F-winning pairs (effective
  calibrated-primary edges: **91** classified `CALIBRATED` + **57** promoted).
  The exact
  full command was:
  `env VISLOC_SFM_DEBUG=1 VISLOC_SFM_DEBUG_IMAGES=20,21 target/release/examples/unordered_sfm_demo --feature-extractor files --features-dir /tmp/vlfeat_detector_corrected_4096_features_20260829 --feature-suffix _features.txt --image-suffix .png --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 --exhaustive --min-matches 15 --match-ratio 0.9 --guided-matching --verification-mode full --calibrated-essential-primary --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8 --geometry-guided-conflict-recovery --post-refinement-registration --final-iterative-refinement --out-colmap /tmp/ce_vlfeat_full_20260829`.
  It reached **38/38**, but changed the model to **13,894 tracks**, **0.368
  px** mean reprojection, and Sim(3) RMSE **239.67 cm** (median **156.63 cm**)
  versus the established F→E fallback run's **117.48 cm**. This is a negative
  accuracy result despite preserved registration.

  The COLMAP-feature graph likewise stayed connected (**380/703** pairs;
  **38/38** images) and promoted **41** F-winning pairs (effective
  calibrated-primary edges: **167** classified `CALIBRATED` + **41** promoted).
  The exact command was:
  `env VISLOC_SFM_DEBUG=1 VISLOC_SFM_DEBUG_IMAGES=20,21 target/release/examples/unordered_sfm_demo --feature-extractor files --features-dir /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_features_export --feature-suffix _features.txt --image-suffix .png --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 --import-matches-supplement-file /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_bridge_matches_import.txt --exhaustive --min-matches 20 --match-ratio 0.8 --verification-mode full --calibrated-essential-primary --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8 --geometry-guided-conflict-recovery --post-refinement-registration --final-iterative-refinement --out-colmap /tmp/ce_colmap_full_20260829`.
  It preserved **38/38**, with **20,722 tracks**, **0.353 px**, and Sim(3)
  RMSE **11.38 cm** (median **9.44 cm**) versus the COLMAP-feature champion's
  **2.84 cm**. Since neither graph disconnected, no post-registration-only
  fallback tier was added; the policy remains an opt-in diagnostic rather than
  an accuracy champion. Logs/models:
  `/tmp/ce_vlfeat_verify_20260829.log`, `/tmp/ce_vlfeat_full_20260829.log`,
  `/tmp/ce_colmap_full_20260829.log`, `/tmp/ce_vlfeat_full_20260829`, and
  `/tmp/ce_colmap_full_20260829`.

- **Strict uncalibrated F→E exclusion strategy (2026-08-29).** Added the
  separate, default-off `--strict-uncalibrated-f-to-essential` strategy. For a
  known-intrinsics `UNCALIBRATED` F-winning pair it uses the existing strict
  manifold/support/residual/refit-stability gate: a passing pair is retained
  with its refined `E_F`, while a failing pair is omitted entirely from
  translation/track construction (this implementation has no rotation-only
  edge representation). Pairs without usable intrinsics and non-UNCALIBRATED
  reports remain on the historical path; the existing
  `--refine-uncalibrated-f-to-essential` fallback behavior is unchanged. The
  CLI predicate has focused tests for default-off, strict-pass, strict-fail,
  calibrated, no-calibration, and deterministic behavior.

  Before mapping, the strict verification probe removed **574 pairs / 75,285
  F inliers** from the corrected-VLFeat graph; **76 pairs / 80,286 inliers**
  remained and the graph split into a 26-image component, a 6-image
  component (`indices 15--20`), and six isolated images (`0307` included),
  so its critical graph was not connected. The exact full command was:
  `env VISLOC_SFM_DEBUG=1 VISLOC_SFM_DEBUG_IMAGES=20,21 target/release/examples/unordered_sfm_demo --feature-extractor files --features-dir /tmp/vlfeat_detector_corrected_4096_features_20260829 --feature-suffix _features.txt --image-suffix .png --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 --exhaustive --min-matches 15 --match-ratio 0.9 --guided-matching --verification-mode full --strict-uncalibrated-f-to-essential --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8 --geometry-guided-conflict-recovery --post-refinement-registration --final-iterative-refinement --out-colmap /tmp/f2e_vlfeat_strict_exclude_20260829`.
  It reached **26/38** images, with **14,647 tracks**, **0.298 px** mean
  reprojection, and Sim(3) centre RMSE **81.96 cm** (median **44.02 cm**);
  only `0309` was added by post-refinement, while `0306`/`0307` had no
  initial 2D--3D support. This is a registration and accuracy negative versus
  the connected fallback run, and no post-registration reuse of rejected F
  edges was added.

  On the COLMAP-feature graph, strict verification removed **200 pairs /
  35,258 F inliers**, leaving **180 pairs / 105,451 inliers**; the retained
  graph stayed connected across all **38/38** images (including `0305`,
  `0306`, and `0307`). The exact full command was:
  `env VISLOC_SFM_DEBUG=1 VISLOC_SFM_DEBUG_IMAGES=20,21 target/release/examples/unordered_sfm_demo --feature-extractor files --features-dir /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_features_export --feature-suffix _features.txt --image-suffix .png --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 --import-matches-supplement-file /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_bridge_matches_import.txt --exhaustive --min-matches 20 --match-ratio 0.8 --verification-mode full --strict-uncalibrated-f-to-essential --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8 --geometry-guided-conflict-recovery --post-refinement-registration --final-iterative-refinement --out-colmap /tmp/f2e_colmap_strict_exclude_20260829`.
  It preserved **38/38**, with **20,054 tracks**, **0.273 px**, and Sim(3)
  RMSE **4.90 cm** (median **3.72 cm**), versus the existing COLMAP champion
  **2.84 cm**. The opt-in strategy is therefore retained as a principled
  graph-ablation diagnostic, not promoted as an accuracy champion.

- **Strict F→E candidate-stability gate (2026-08-29).** The opt-in
  `--refine-uncalibrated-f-to-essential` path now keeps the existing
  support/cheirality checks and additionally requires a calibrated
  `Kᵀ F K` projection within **1%** relative Frobenius distance of the
  essential manifold (`s1/s2` mismatch ≤**2%**, `s3/s2` ≤**5%**), at least
  **90%** overlap with the F inliers, normalized E/F residual ratio ≤**3**,
  at least **2/3** deterministic prefix/suffix/evenly-spaced F-refits, and
  ≤**5°** rotation and translation-direction spread across those refits.
  `VISLOC_SFM_DEBUG_DUMP_F2E_DIAGNOSTICS=1` logs these singular values,
  projection/residual fields, cheirality margin, and pose spreads for every
  `UNCALIBRATED` candidate; no GT is used by the gate. Offline GT
  analysis found the VLFeat `0306--0307` candidate superficially essential
  (`projection_distortion=0.001503`, F-overlap **0.988006**) but unstable at
  **22.055565°** translation spread, while `0305--0306` was stable at
  **0.689680°**. The broad support-only gate admitted **67/586** VLFeat and
  **70/218** COLMAP candidates; the strict gate admits **12** and **18**.
  Across the broad sets, F→E improved/worsened GT translation direction on
  **54/13** VLFeat and **56/14** COLMAP pairs (diagnostic labels only), so
  pairwise pose improvement alone is not treated as a mapper-quality gate.

  Focused corrected-VLFeat A/B (`0305--0307`, `0306--0307`) retained
  `0305--0306` as **424 E_F** and rejected `0306--0307` back to **667 F**;
  the 0305--0307 projection also fell back to **136 F**, so accepted support
  stayed **1227** rather than the broad-gate **1219**. Exact full-run command:
  `VISLOC_SFM_DEBUG=1 VISLOC_SFM_DEBUG_IMAGES=20,21 target/release/examples/unordered_sfm_demo --feature-extractor files --features-dir /tmp/vlfeat_detector_corrected_4096_features_20260829 --feature-suffix _features.txt --image-suffix .png --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 --exhaustive --min-matches 15 --match-ratio 0.9 --guided-matching --verification-mode full --refine-uncalibrated-f-to-essential --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8 --geometry-guided-conflict-recovery --post-refinement-registration --final-iterative-refinement --out-colmap /tmp/f2e_vlfeat_strict_20260829`.
  It accepted **650/703** pairs and **155,571** inlier correspondences,
  reached **35/38** in growth, then recovery/post-refinement registered the
  remaining three (**38/38** total). Post-PnP registered `0306` at
  **189→78** and `0307` at **205→90** inliers. The final model has **14,864
  tracks**, **0.350 px** mean reprojection, and Sim(3) RMSE **117.48 cm**
  (median **69.59 cm**), improving both the no-refinement source-cap control
  (**323.70 cm**) and the preceding broad-gate run (**281.61 cm**).

  COLMAP-feature champion command (same downstream stack, bridge supplement,
  ratio 0.8):
  `VISLOC_SFM_DEBUG=1 VISLOC_SFM_DEBUG_IMAGES=20,21 target/release/examples/unordered_sfm_demo --feature-extractor files --features-dir /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_features_export --feature-suffix _features.txt --image-suffix .png --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 --import-matches-supplement-file /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_bridge_matches_import.txt --exhaustive --min-matches 20 --match-ratio 0.8 --verification-mode full --refine-uncalibrated-f-to-essential --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8 --geometry-guided-conflict-recovery --post-refinement-registration --final-iterative-refinement --out-colmap /tmp/f2e_colmap_strict_20260829`.
  It produced **380/703** pairs, **140,709** inlier correspondences,
  and **18** strict refinements. It preserved **38/38**, **21,342 tracks**,
  **0.284 px**, and Sim(3) RMSE **3.38 cm** (median **2.67 cm**) versus the
  champion **21,338 / 0.283 px / 2.84 cm**; this is a small, acceptable A/B
  delta and avoids the broad-gate regression to **21.43 cm**. The option
  remains explicitly opt-in. Logs/models:
  `/tmp/f2e_vlfeat_diagnostics_20260829.log`,
  `/tmp/f2e_colmap_diagnostics_20260829.log`,
  `/tmp/f2e_vlfeat_strict_20260829.log`,
  `/tmp/f2e_colmap_strict_20260829.log`,
  `/tmp/f2e_vlfeat_strict_20260829`, and
  `/tmp/f2e_colmap_strict_20260829`.

- **Guarded calibrated F→E refinement (2026-08-29).** Added the opt-in
  `--refine-uncalibrated-f-to-essential` path to the full verifier. It applies
  only to an `UNCALIBRATED` (F-winning) report: `E_F = K_jᵀ F K_i` is projected
  to the essential manifold, every candidate is rescored at the calibrated
  normalized 4 px threshold, and the pair is replaced only when both support
  is at least `max(8,min_matches)` and at least 50% of the F support, with
  ≥75% positive-depth cheirality and second/best cheirality ≤0.25. Invalid or
  weak projections fall back to the original winning set; calibrated E
  winners and the default path are unchanged. The resulting `E_F` is exposed
  through `PairwiseMatches::essential_matrix` and its accepted inlier set is
  used for tracks only after the guards pass. Synthetic tests cover recovery,
  invalid/weak fallback, calibrated-input no-op, and CLI default-off parsing.

  Focused corrected VLFeat-feature control (three-image subset,
  `--match-ratio 0.9 --guided-matching --verification-mode full`) changed
  `0305--0306` **424 F → 424 E_F** (360/424 positive-depth),
  `0306--0307` **667 F → 659 E_F** (615/659), and rejected the weak
  `0305--0307` projection (remained **136 F**); total accepted matches were
  **1227 → 1219**, with **2/3** pairs refined. The full source-cap run used
  `/tmp/vlfeat_detector_corrected_4096_features_20260829`, exhaustive ratio
  0.9 + guided matching, full verification, `pnp-max-iterations=100000`,
  `min-pnp-inliers=8`, geometry recovery, post-refinement, and final
  iterative refinement. Exact command:
  `VISLOC_SFM_DEBUG=1 VISLOC_SFM_DEBUG_IMAGES=20,21 target/release/examples/unordered_sfm_demo --feature-extractor files --features-dir /tmp/vlfeat_detector_corrected_4096_features_20260829 --feature-suffix _features.txt --image-suffix .png --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 --exhaustive --min-matches 15 --match-ratio 0.9 --guided-matching --verification-mode full --refine-uncalibrated-f-to-essential --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8 --geometry-guided-conflict-recovery --post-refinement-registration --final-iterative-refinement --out-colmap /tmp/f2e_vlfeat_full_20260829`.
  It accepted **650/703** pairs, **152,685** inlier correspondences, and
  **67** guarded refinements; 0306/0307 growth had **203→147 / 170→125**
  PnP inliers. Recovery completed the initial **37/38** growth to **38/38**;
  the final model has **12,561 tracks**, **0.366 px** mean reprojection, and
  Sim(3) centre RMSE **281.61 cm** (median **248.84 cm**), improving the
  no-refinement source-cap control (**12,057 tracks, 0.491 px, 323.70 cm**).

  The COLMAP-feature champion A/B used the bridge supplement, exhaustive
  ratio 0.8, full verification, and the same mapper stack. Exact command:
  `VISLOC_SFM_DEBUG=1 VISLOC_SFM_DEBUG_IMAGES=20,21 target/release/examples/unordered_sfm_demo --feature-extractor files --features-dir /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_features_export --feature-suffix _features.txt --image-suffix .png --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 --import-matches-supplement-file /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_bridge_matches_import.txt --exhaustive --min-matches 20 --match-ratio 0.8 --verification-mode full --refine-uncalibrated-f-to-essential --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8 --geometry-guided-conflict-recovery --post-refinement-registration --final-iterative-refinement --out-colmap /tmp/f2e_colmap_full_20260829`.
  Refinement accepted **70** pairs: registration stayed **38/38**, but the
  model changed
  from the champion **21,338 tracks / 0.283 px / 2.84 cm** to **20,216 tracks /
  0.313 px / 21.43 cm** (median **9.74 cm**). This is a clear accuracy
  regression, so the option remains default-off and is not an accuracy
  champion; use it only for the guarded F→E diagnostic/ablation. Models and
  logs: `/tmp/f2e_vlfeat_full_20260829` and
  `/tmp/f2e_colmap_full_20260829` (`/tmp/f2e_colmap_full_20260829.log`).

- **GT-independent essential-pair parallax/cheirality audit (2026-08-29).**
  Added the default-off `VISLOC_SFM_DEBUG_DUMP_ESSENTIAL_QUALITY=1` probe in
  `unordered_sfm_demo`. For every attempted full-verifier pair it reports
  E/F/H support, the four-hypothesis positive-depth count, the second/best
  cheirality score, p10/p25/median triangulation angle, and the scale-free
  positive-depth ratio `min(z1,z2)/max(z1,z2)` from a deterministic maximum
  256-sample E-inlier subset; it exits before mapper/track construction, so
  ordinary behavior is unchanged. On the VLFeat-compatible 4096 feature
  graph (`/tmp/vlfeat_detector_features_20260829`, exhaustive, ratio 0.9,
  guided, full, min-matches 15), **603/703** pairs were admitted and **577**
  had usable E quality. Pair-level angle quantiles were **3.981°/8.923°/
  18.715°** (p10/p25/median), cheirality fraction quantiles
  **0.513/0.667/0.889**, and per-pair depth-ratio p10 quantiles
  **0.026/0.118/0.435**. The known wrong-translation edge
  `DSC_0306--DSC_0307` remained apparently healthy (**E/F/H=590/642/201**,
  cheirality **585/590**, second/best **0.009**, angle
  **1.265°/2.091°/3.724°**, depth-ratio median **0.933**); other wrong edges
  were likewise not uniformly weak: `0296--0300` had **E=38**, angle
  **4.308°/8.464°/9.889°**, and depth-ratio median **0.964**, while
  `0301--0303` had **E=405**, cheirality **403/405**, and angle median
  **3.381°**. Across the **502** E-bearing edges that joined the existing
  oracle bearing diagnostic, median angle correlated only **r=0.226** with
  translation-bearing error (rotation error **r=0.604**); depth p10 was the
  strongest aggregate signal (**r=-0.361/-0.690** for translation/rotation),
  but standard floors did not separate the incidents (median-angle `<2°`
  rejected only **4/85** edges with translation error ≥45°, while the known
  80.1° edge passed). The COLMAP-feature bridge-supplement graph used for the
  **2.84 cm** champion had **380** admitted pairs with **376** usable E rows,
  angle quantiles **3.242°/6.507°/12.554°**, cheirality
  **0.857/0.967/0.994**, and per-pair depth-ratio p10
  **0.408/0.492/0.743**. The
  compatible full safe run remains **38/38, 0.371 px, 142.99 cm** versus
  the COLMAP-feature recovery+post champion **38/38, 0.283 px, 2.84 cm**.
  Therefore no default-off pair eligibility gate was added: low-parallax or
  weak-cheirality filtering cannot remove the demonstrated false E directions
  without also removing valid edges. The next exact issue is E/F model
  selection/accepted-index and track topology (not a generic parallax gate).
  A current-binary safe-stack rerun of the COLMAP-feature champion preserved
  **38/38, 21,338 tracks, 0.283 px, and 2.84 cm** (log:
  `/tmp/essential_quality_colmap_full_20260829.log`; score/model:
  `/tmp/essential_quality_colmap_full_20260829`).
  Logs: `/tmp/essential_quality_vlfeat_run_20260829.log`,
  `/tmp/essential_quality_vlfeat_gt_20260829.log`, and
  `/tmp/essential_quality_colmap_champion_20260829.log`.

- **Calibrated F→E relative-pose audit (2026-08-29).** Extended the
  default-off `VISLOC_SFM_DEBUG_DUMP_ESSENTIAL_QUALITY=1` report with the
  refined-F inlier set, pixel/normalized Sampson residuals, and
  `E_F = K_jᵀ F K_i` projected to the essential manifold by equalizing its
  first two singular values. The emitted pose fields are diagnostic only;
  neither verification nor tracks consume `E_F`. The projection convention
  is covered by a synthetic `Kᵀ F K` unit test. Direct-E audit found no
  normalization or cheirality bug: the estimator first calibrates pixels,
  Hartley-normalizes each side, solves the 8-point `AᵀA` system, projects to
  equal nonzero singular values, and RANSAC uses 256 iterations/seed 7 with
  the normalized `4/f = 0.004548` threshold and inlier refit/rescore;
  decomposition evaluates all four positive-depth hypotheses. Fundamental
  RANSAC independently uses the pixel 4 px threshold.

  In the VLFeat-compatible graph, **34** rows were `CALIBRATED` and **29**
  had an 8-point F→E pose. F→E E-inlier count was lower than direct E on
  **20/29** rows (median delta **−101**, mean **−192**); on the common F
  inlier set its normalized Sampson residual was lower on **17/29** rows and
  higher on **12/29**, while cheirality fraction was higher on **22/29**.
  GT is used only for the audit: paired rotation error mean changed
  **2.06° → 0.69°** (better **19/29**), but translation changed
  **13.75° → 9.81°** with only **12/29** improvements and **16/29**
  regressions (median delta **+0.06°**). In the COLMAP-feature graph,
  **126** rows were `CALIBRATED` and **105** had an F→E pose; F→E lost E
  inliers on **72/105** rows (median delta **−112**, mean **−155**), and
  was lower on the common-F residual for **49/105** versus **56/105**
  regressions. Rotation mean improved **6.55° → 4.42°**, but translation
  worsened **17.90° → 21.71°** (better **36/105**, worse **67/105**).
  Therefore F→E is not a consistent GT-independent selector and no
  default-off mapper/model-selection switch was added.

  The requested incident edges are `UNCALIBRATED` in both graphs, so they
  are not eligible for the calibrated branch; their diagnostic direct/F→E
  E counts and GT pose errors (R/T degrees) were VLFeat
  `0305--0306: 363/391, .66/.08, 2.03/.90`,
  `0305--0307: 45/119, 18.22/.59, 154.83/2.44`,
  `0306--0307: 590/634, .63/.62, 80.12/5.96`, and
  `0296--0300: 38/0, 91.34/NA, 93.81/NA`; COLMAP
  `0305--0306: 261/316, 1.33/.04, 28.97/1.50`,
  `0305--0307: 64/73, 7.98/.92, 62.49/8.29`,
  `0306--0307: 477/591, 2.37/1.81, 13.52/13.91`, and
  `0296--0300: 0/0, NA/NA, NA/NA`. These rows explain why F→E is a useful
  forensic comparison, not a safe global replacement. A no-diagnostic
  control after the change preserved the established outputs: VLFeat
  **38/38, 11,533 tracks, 0.371 px** (known Sim(3) RMSE **142.99 cm**) and
  COLMAP **38/38, 21,338 tracks, 0.283 px, 2.84 cm**. Logs:
  `/tmp/essential_quality_vlfeat_f2e_pose_20260829.log`,
  `/tmp/essential_quality_colmap_f2e_pose_20260829.log`,
  `/tmp/essential_quality_vlfeat_full_f2e_control_20260829.log`, and
  `/tmp/essential_quality_colmap_full_f2e_control_20260829.log`.

- **VLFeat/COLMAP detector metadata and feature-cap audit (2026-08-29).**
  Added the explicit `diagnose_sift_vlfeat_detector` API and
  `examples/vlfeat_detector_diagnostic.rs`, which export x/y/σ/orientation,
  refined DoG response, octave/level, and orientation index before expansion,
  after expansion, after the per-locus orientation cap, and after the global
  cap. The audit used all **38** images in
  `colmap_oracle_full/database.db`: **208,785** COLMAP six-column rows,
  **3,508--6,970** rows/image (median **5,366**), all below the header's
  `max_num_features=8192` default.  The SQLite database stores keypoint blobs,
  not the SiftExtraction options, so that bound alone does **not** prove that
  the recorded run used an uncapped 8192-feature setting.  The affine rows are scaled rotations in this
  database, so `σ=sqrt(det(A))` and `atan2(a21,a11)` are lossless for the
  isotropic comparison. The unbounded compatible detector produced **334,560**
  extrema, **393,764** oriented rows, and **392,099** rows with the source
  default two-orientation cap. Nearest spatial matching found
  **199,395/208,785=95.50%** of COLMAP rows within 1 px (median/p95
  **0.0315/0.5918 px**); the matched signed ours-minus-COLMAP offset was only
  **(+0.00019,+0.00013) px** at the median, with no ±0.5-pixel convention
  bias. Matched `|log(σ_ours/σ_colmap)|` was **0.00237/0.0252** median/p95
  overall (critical 0305/0306/0307: **0.00385/0.00375/0.00372** median),
  while the nearest compatible orientation error was **0.478°/1.85°**
  median/p95 (critical medians **0.544°/0.562°/0.535°**). Orientation
  multiplicity was comparable to COLMAP: **29.83%** versus **29.10%** of
  rows in rounded multi-orientation loci (max four), so duplication is not
  the missing-feature explanation. A true response-rank comparison is not
  possible because the COLMAP database does not retain DoG responses; the
  diagnostic retains our refined response for an explicitly labelled proxy,
  and reports missing rows by that proxy rather than pretending it is a
  COLMAP rank. With the explicit proxy definition “per-image top N extrema by
  `|response|`, then COLMAP-row nearest distance ≤1 px”, all-image hit/missing
  counts were **52,568/156,217** (N=2048), **104,373/104,412** (4096), and
  **175,926/32,859** (8192). At N=4096 the corresponding critical counts were
  0305 **4,401/953**, 0306 **3,392/255**, 0307 **3,228/280**; the late
  0320/0321/0322 rows were **1,622/3,718**, **1,763/3,115**, and
  **1,753/3,057**. These are spatial coverage proxies, not source response
  rank overlap.

  The cap implementation now follows COLMAP `sift.cc`: count unoriented
  extrema per `(octave, level)`, walk groups from coarse to fine in reverse
  source order, and retain the complete suffix that crosses the cap; it does
  not sort/truncate individual orientation copies. This is the source
  behavior in [COLMAP `sift.cc`](https://github.com/colmap/colmap/blob/main/src/colmap/feature/sift.cc)
  (group cap) together with [COLMAP `sift.h`](https://github.com/colmap/colmap/blob/main/src/colmap/feature/sift.h)
  (`8192`, first octave `-1`, two orientations), while the detector stages
  follow [VLFeat `sift.c`](https://raw.githubusercontent.com/vlfeat/vlfeat/master/vl/sift.c)
  (refined response, octave/level order, and orientation peaks). At the
  justified **4096** cap, source-level suffix retention yielded **210,864**
  emitted rows and a **93.90%** weighted COLMAP spatial recall proxy versus
  **154,864** rows and **69.45%** from the previous arbitrary oriented-row
  truncation. The resulting real-feature focused bridge rows at ratio .9 +
  cross-check were `0305--0306=611/423`, `0305--0307=362/136`, and
  `0306--0307=825/667` (raw/accepted); the full compatible safe stack reached
  **38/38**, **12,057** tracks, **0.491 px**, but **323.70 cm** Sim(3) RMSE.
  Thus source-correct capping fixes feature-count semantics and preserves
  registration, but is not an accuracy-champion update (the fixed-endpoint
  control remains **2.92 cm**). An 8192 source-cap export (**364,373** rows)
  was generated as the only other source-justified cap; its mapper run was
  stopped at roughly **4.7 GB RSS** before producing a result, so no score is
  claimed. Targeted VLFeat detector tests (including stage alignment,
  determinism, and whole-level cap behavior) pass.

- **VLFeat descriptor support-radius audit and fixed-keypoint A/B
  (2026-08-29).** The earlier `ceil(3σ)` wording was corrected: that radius
  belongs to VLFeat's orientation histogram (`sift.c` lines 1579--1580), not
  the descriptor. The authoritative `vl_sift_calc_keypoint_descriptor` loop
  sets `sigma=k->sigma/xper`, `SBP=magnif*sigma+VL_EPSILON_D`, and
  `W=floor(sqrt(2.0)*SBP*(NBP+1)/2.0+0.5)` (`sift.c` lines 1953--1965), then
  clips samples to `max(-W,1-xi)..min(W,w-xi-2)` and the analogous y range
  (lines 2006--2010), applies `exp(-(nx²+ny²)/(2*(NBP/2)²))` (lines
  2027--2033), and uses the `nx-0.5`/`ny-0.5`/`nt` trilinear bins (lines
  2037--2063). The compatible Rust path now has an explicit, tested
  `vlfeat_descriptor_window_radius` for that exact support expression,
  source-equivalent boundary finite differences from `update_gradient`
  (lines 1447--1529), and the source keypoint scale-to-octave selection
  (lines 2172--2180); COLMAP's UBC orientation transpose remains its `q={0,7,
  6,5,4,3,2,1}` assignment (sift.cc lines 118--135). A fixed-keypoint probe
  (`examples/fixed_sift_descriptor_probe.rs` plus
  `scripts/compare_sift_descriptors.py`) on identical COLMAP rows reported
  cosine means **0.942752/0.943360/0.942735** for `0305/0306/0307`, nonzero
  layout Jaccard **0.820688/0.809097/0.807644**, and **0** exact quantized
  rows; the best fixed spatial/orientation/sign transform on 2,048 rows was
  the identity (`cosine=0.945285`). Ratio-0.8 + cross-check focus rows after
  the correction were **310/261, 150/101, 580/535** (raw/accepted), with
  COLMAP raw-index overlap **248/353, 101/161, 522/630**; before correction
  the overlaps were **245/353, 94/161, 516/630**. The full fixed-endpoint
  command used `--exhaustive --min-matches 15 --match-ratio 0.8
  --verification-mode full --mapper incremental --pnp-max-iterations 100000
  --min-pnp-inliers 8 --geometry-guided-conflict-recovery
  --post-refinement-registration --final-iterative-refinement` and reached
  **409/703** verified pairs, **38/38** images, **19,898** tracks, **0.271
  px**, and **2.92 cm** Sim(3) centre RMSE (output:
  `/tmp/visloc_colmap_fixed_vlfeat_after_20260829_pty2_cLpTOv`; prior fixed
  compatible output was **21.67 cm**). The required detected-compatible
  follow-up used the same stack with `--feature-extractor sift --sift-max-
  keypoints 4096 --sift-vlfeat-compatible-detector
  --sift-vlfeat-compatible-descriptor --match-ratio 0.9 --guided-matching`
  and reached **37/38**, **9,588 tracks**, **0.363 px**, **192.03 cm**; it is
  a negative detected-feature result versus the prior **38/38, 142.99 cm**
  run, so no accuracy champion changed. The exact source references are
  [VLFeat `sift.c`](https://github.com/vlfeat/vlfeat/blob/master/vl/sift.c)
  and [COLMAP `sift.cc`](https://github.com/colmap/colmap/blob/main/src/colmap/feature/sift.cc).

- **VLFeat-compatible match-level discriminant audit (2026-08-29).** A
  one-off NumPy probe reimplemented the Rust NN+Lowe-0.9+mutual matcher on
  the persisted 4096-keypoint files, labelled matches by GT Sampson residual
  (`<1 px`) and by nearest same-image COLMAP-keypoint mapping (`≤0.5 px`) to
  COLMAP's verified match set, and examined four representative edges. The
  aggregate of **1,897** matches gave descriptor-distance / Lowe-ratio /
  mutual-margin means of **132.3/170.7**, **0.569/0.740**, and
  **0.381/0.215** for GT-positive/negative matches; the corresponding
  COLMAP-verified-positive/negative values were **127.6/166.1**,
  **0.541/0.724**, and **0.410/0.229**. Per-edge positive counts and
  diagnostics were `0305--0306: 309/243 GT+/- (192/360 COLMAP+/-),
  distance 135.2/174.1, ratio .606/.747, margin .332/.214, Hough
  support 6,913`, `0305--0307: 90/226 (59/257), 158.0/184.3, .669/.760,
  .285/.191, support 1,519`, `0306--0307: 521/298 (432/387), 126.0/152.0,
  .531/.680, .427/.269, support 18,918`, and `0296--0300: 0/210
  (11/199), 178.6, .795, .167, support 264`. Support is the match-pair
  count around the dominant deterministic 4-D similarity Hough bin; the
  corresponding dominant-bin counts were **1,529/296/5,287/49**, and
  normalized 8×8 source/target grid entropies were respectively
  **.920/.874, .901/.873, .860/.863, .800/.758**. Crucially, the known
  wrong-translation edge `0306--0307` has the *best* descriptor statistics
  and strongest Hough support, despite its two-view translation-bearing
  error of **80.1°**; a descriptor-distance, ratio, spatial, or Hough gate
  would retain it. A ratio `<0.7` gives only **69.6% precision / 74.2%
  recall** on GT labels (**57.5%/81.4%** on COLMAP labels), while the
  stronger `margin>0.3` gives **75.7%/62.2%** (**64.9%/70.7%**); these are
  not safe source-derived admission thresholds, and the poor `0296--0300`
  edge alone does not justify dataset-specific pair pruning. No new
  pre-verification gate or A/B was therefore added. The exported feature
  format contains only `(x,y,score,descriptor)` and `FeatureSet` has already
  discarded `SiftKeypoint.sigma/orientation`, so log-scale and orientation
  deltas cannot be measured honestly from this run. The same-keypoint parity
  audit remains the actionable bottleneck: compatible-descriptor raw-index
  overlap with actual COLMAP descriptors at ratio .8 + cross-check was only
  **50.4/37.5/71.5%** on `0305--0306/0305--0307/0306--0307` (mag-3 improved
  this to **59.4/39.2/80.2%**, mean descriptor cosine **0.942**), far below
  the COLMAP control. Inputs/logs: `/tmp/vlfeat_detector_features_20260829`,
  `/tmp/diagnose_pair_match_discriminants_20260829.log`, and the prior fixed
  endpoint runs under `/tmp/colmap_fixed_oursift_mag8_20260828` and
  `/tmp/colmap_fixed_oursift_mag3_20260828`. This is diagnostic evidence,
  not an accuracy-champion update.

- **VLFeat-compatible 38/38 distortion diagnosis and cycle audit
  (2026-08-29).** The 4096-keypoint VLFeat-compatible detector+descriptor
  model is registered on all **38/38** images with **11,533** tracks and
  **0.371 px** mean reprojection, but its aligned centre RMSE is **142.99 cm**
  (median **89.31 cm**, Sim(3) scale **0.431753**, pair-distance correlation
  **0.90420**), versus the COLMAP-feature recovery+post model's **2.842 cm**,
  **0.283 px**, scale **0.577188**, and correlation **0.999965**. Exact
  aligned centre errors in `images.txt` order (`DSC_0286` through
  `DSC_0323`, cm) are
  `19.82,11.85,17.18,28.19,42.34,54.15,65.25,84.14,97.47,110.35,115.15,`
  `131.71,113.13,90.57,79.71,74.62,80.83,82.42,86.69,93.13,88.06,118.37,`
  `120.24,34.18,25.28,208.10,189.72,195.93,139.00,158.29,521.67,235.21,`
  `142.08,100.14,71.56,67.05,66.47,269.68`. This is a distributed/bent
  pose basin with localized catastrophic cameras (`0311--0317`, especially
  `0316=521.67 cm`), not a single far-orbit island or a uniform scale-only
  drift; the low reprojection therefore does not certify the metric pose.
  The same-run growth log (`/tmp/sift_vlfeat_detector_cycles_20260829.log`)
  seeded pair `(8,9)` and grew through `0293,0292,0291,0290,0289,0288,0287,`
  `0286,0310,0296,0309,0318,0297,0312,0313,0317,0323,0311,0314,0315,`
  `0319,0320,0321,0322,0316,0298,0299,0300,0302,0303,0301,0304,0305,`
  `0308`; growth stopped at **35/38** and post-refinement added `0304`
  (**633→490** PnP inliers), `0306` (**188→105**), and `0307`
  (**172→94**). On the representative bridge edges, focused ratio-0.9
  NN+cross-check rows were `0305--0306=552 raw/392 accepted, E=363,
  UNCALIBRATED`, `0305--0307=316/94, E=50, UNCALIBRATED`, and
  `0306--0307=819/658, E=636, CALIBRATED`; the corresponding same-feature
  pose probe measured rotation/translation-bearing errors of
  `0305--0306=(0.7°,2.0°)`, `0305--0307=(18.2°,25.2°)`, and
  `0306--0307=(0.6°,80.1°)`. The last edge is a high-inlier but wrong
  translation constraint, while `0296--0300` (`E=38`, `R=91.3°`, bearing
  `86.2°`), `0301--0303` (`E=405`, `R=7.6°`, bearing `84.1°`), and
  `0321--0323` (`E=123`, `R=9.5°`, bearing `87.8°`) show that wrong direction
  hypotheses are distributed through the graph. Descriptor overlap against
  COLMAP raw matches on the three bridge pairs was only **48.3/45.0%**,
  **43.2/33.5%**, and **63.5/61.1%** (precision/recall proxy).
  The full-graph reproduction used
  `VISLOC_SFM_DEBUG=1 VISLOC_SFM_DEBUG_DUMP_ROTATION_CYCLES=1
  target/release/examples/unordered_sfm_demo --feature-extractor files
  --features-dir /tmp/vlfeat_detector_features_20260829 --width 1600
  --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 --exhaustive
  --min-matches 15 --match-ratio 0.9 --guided-matching --verification-mode
  full --mapper incremental --pnp-max-iterations 100000
  --min-pnp-inliers 8 --geometry-guided-conflict-recovery
  --post-refinement-registration --final-iterative-refinement
  --diagnose-bearing-gt /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/gt/images.txt
  --out-colmap /tmp/sift_vlfeat_detector_cycles_20260829`; the focused
  nine-image edge reprobe is `/tmp/sift_vlfeat_detector_edges_focus2_20260829.log`.
  A GT-independent rotation-cycle dump (`VISLOC_SFM_DEBUG_DUMP_ROTATION_CYCLES=1`)
  found **502** essential rotations, **3,413** triangles, and globally
  multimodal contradictions: `0305--0306` had cycle median/p90
  **17.3°/58.7°**, `0305--0307` **17.3°/111.0°**, and `0306--0307`
  **93.2°/134.1°**, while the full graph's worst medians reached
  **174.0°**. Thus no single deterministic cycle threshold is safely
  supported; the existing post-PnP two-view gate was already a no-op on the
  38/38 COLMAP-feature basin, so no new mapper rejection policy was enabled.
  The only code change in this slice is the default-off diagnostic output
  (including relative-rotation error and essential fallback reporting) in
  `unordered_sfm_demo`; the next safe intervention is edge-level
  multi-hypothesis/accepted-index quality analysis rather than arbitrary
  pruning. This is a diagnostic result, not an accuracy-champion update.

- **Fixed-COLMAP-keypoint descriptor decomposition (2026-08-29).** A
  descriptor-only probe read all **208,785** six-column COLMAP keypoints and
  re-described the identical `(x, y)` endpoints on the normalized images with
  the existing SIFT API at magnification **8.0** and **3.0**; no detector,
  orientation re-detection, or endpoint reordering was involved. The affine
  rows map as `sigma = sqrt(det(A))`, `orientation = atan2(a21, a11)` while
  dropping `A`'s anisotropic shape. In this database every determinant was
  positive and every matrix was a scaled rotation (`a11=a22`, `a12=-a21` at
  float32 precision; singular-value ratio median/p90/max **1.0/1.0/1.0**), so
  an affine-aware sampling extension was not justified. With identical
  ratio-0.8 + cross-check + full-verifier settings, focused bridge raw/accepted
  counts for `0305--0306`, `0305--0307`, `0306--0307` were COLMAP
  **356/321, 163/126, 631/585**, fixed-endpoint our-SIFT@8
  **224/206, 32/20, 593/563**, and fixed-endpoint our-SIFT@3
  **207/183, 51/30, 565/526**; raw-index overlap with COLMAP was respectively
  **50.4/37.5/71.5%** (@8) and **59.4/39.2/80.2%** (@3). The safe full stack
  (exhaustive, ratio 0.8, full verification, plain incremental,
  `pnp100k`, min-PnP 8, conflict recovery, post-refinement, final iterative)
  gave the known COLMAP-descriptor control **382/703, 38/38, 20,392 tracks,
  0.376 px, 18.62 cm Sim(3) RMSE**, versus fixed-endpoint our-SIFT@8
  **302/703, 29/38, 9,303, 0.259 px, 387.50 cm** and @3 **285/703, 35/38,
  16,211, 0.309 px, 117.62 cm**. Thus detector localization is held out of
  the A/B/C gap; descriptor formulation/sampling (including any remaining
  scale/orientation convention or quantization difference) is the measured
  bottleneck, with @3 a material but insufficient improvement. The temporary
  probe was removed after the run; no default CLI or normal-path behavior was
  changed. Logs/models are under `/tmp/visloc_colmap_fixed_colmapdesc_20260829`,
  `/tmp/visloc_colmap_fixed_oursift_mag8_20260828`, and
  `/tmp/visloc_colmap_fixed_oursift_mag3_20260828` (each has a matching
  `.log`). Next experiment is COLMAP/VLFeat descriptor parity, not an
  unsupported affine-shape extension.

- **COLMAP frame/sampling convention audit (2026-08-29).** The authoritative
  COLMAP `FeatureKeypoint` implementation defines
  `ComputeScale=(||col_1(A)||+||col_2(A)||)/2` and
  `ComputeOrientation=atan2(a21,a11)`; its VLFeat extractor adds **+0.5** to
  exported frame centers. The database/FAQ convention therefore maps a stored
  COLMAP `(x,y)` to this crate's integer-centered image sampler as
  `(x-0.5,y-0.5)` while keeping the exported geometry unchanged. This is from
  COLMAP `main` as fetched 2026-08-29:
  [types.cc](https://github.com/colmap/colmap/blob/main/src/colmap/feature/types.cc),
  [sift.cc](https://github.com/colmap/colmap/blob/main/src/colmap/feature/sift.cc),
  [database.rst](https://github.com/colmap/colmap/blob/main/doc/database.rst),
  and [FAQ pixel convention](https://github.com/colmap/colmap/blob/main/doc/faq.rst).
  A bounded mag-3 probe on the three bridge pairs compared no offset,
  `(-0.5,-0.5)` offset, and offset plus reversed orientation. The first two
  produced raw/accepted `207/183, 51/30, 565/526` versus
  `223/199, 48/32, 543/511`; COLMAP-raw index overlap precision/recall changed
  `58.9/34.6, 41.2/13.0, 80.2/71.9%` to
  `59.2/37.4, 52.1/15.5, 79.9/68.9%` (pair order `0305--0306`,
  `0305--0307`, `0306--0307`). Reversing the orientation gave only
  `116/84, 33/0, 379/339`, so the source orientation sign is confirmed; the
  center offset is semantically correct but not a consistent pair-level gain,
  and no full reconstruction or default CLI was justified. The mag-8 offset
  result was likewise mixed (`224/206,32/20,593/563` →
  `207/192,32/23,590/567`). This audit motivated the cohesive descriptor mode
  recorded above: VLFeat requires the gradient to be evaluated on the
  keypoint-scale-smoothed image and its raw implementation uses integer
  gradient samples plus fractional displacement, while the historical crate
  path uses direct ±1-pixel central differences on the unsmoothed source. See the
  [VLFeat descriptor API](https://www.vlfeat.org/api/sift_8c.html) and
  [reference implementation](https://raw.githubusercontent.com/vlfeat/vlfeat/master/vl/sift.c).
  The partial scale-adaptive experiment below was retained for comparison, but
  affine-shape sampling remains unsupported by evidence because this
  database's matrices are isotropic.

- **Opt-in cohesive VLFeat/COLMAP SIFT descriptor (2026-08-29).** Added
  `SiftConfig::vlfeat_compatible_descriptor` and the default-off
  `--sift-vlfeat-compatible-descriptor` path. The mode keeps the legacy
  detector and all prior experimental flags intact, but selects the complete
  descriptor contract as one unit: a σ₀/2=0.8 full-resolution base pyramid
  (octaves downsampled by two), integer central-difference gradients from the
  rounded VLFeat scale level (finite `ceil(4σ)` Gaussian support with
  continuity padding, as used by `_vl_sift_smooth`), m=3 support (`SBP=3σ`
  and VLFeat's finite `W`),
  `(x-0.5,y-0.5)` COLMAP-center mapping, rotated sample coordinates,
  2-cell Gaussian window, 4×4×8 trilinear accumulation with circular
  orientation bins, L2→0.2 clamp→L2 or COLMAP L1-root normalization, and
  `round(512·d)` uint8-equivalent values followed by COLMAP's
  VLFeat→UBC orientation permutation. CLI validation rejects the complete
  mode with the partial scale-adaptive, affine, alternate-bank, or conflicting
  magnification flags; default behavior remains byte-identical. Focused tests
  cover bin conservation/orientation wrap, normalization/quantization/layout,
  pyramid level selection, deterministic output, and default identity.
  Fixed external COLMAP keypoints on normalized `images_1600x1066` improved
  ratio-0.8 + cross-check raw/accepted counts for `0305--0306`,
  `0305--0307`, `0306--0307` to **310/263, 143/95, 580/534**, versus the
  prior fixed mag-3 **207/183, 51/30, 565/526** and actual COLMAP
  **356/321, 163/126, 631/585**; descriptor cosine to the corresponding
  COLMAP rows was **0.942 mean** over all 3-image fixed sets. The full fixed
  run (`--exhaustive --verification-mode full --min-matches 15
  --min-pnp-inliers 8 --pnp-max-iterations 100000
  --geometry-guided-conflict-recovery --post-refinement-registration
  --final-iterative-refinement`) reached **405/703** verified pairs,
  **38/38** images, **18,873** tracks, **0.385 px** mean reprojection, and
  **21.67 cm** Sim(3) center RMSE (output:
  `/media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/runs/fixed_vlfeat_compatible_l2`,
  log: `/tmp/fixed_vlfeat_compatible_l2.log`; the COLMAP-descriptor
  control is 38/38, 0.376 px, 18.62 cm). A detected our-SIFT run with 4096 keypoints,
  ratio 0.9 + guided matching and the same safe mapper stack reached
  **542/703**, **24/38**, **5,925** tracks, **0.448 px**, **415.43 cm**;
  this is a descriptor-parity success on fixed endpoints but not a detected
  SIFT accuracy champion (output:
  `/media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/runs/sift_vlfeat_compatible_l2`,
  log: `/tmp/sift_vlfeat_compatible_l2.log`; the prior strongest
  extra-keypoint run was 26/38).
  The implementation follows the [VLFeat SIFT reference
  implementation](https://raw.githubusercontent.com/vlfeat/vlfeat/master/vl/sift.c),
  [VLFeat image convolution](https://raw.githubusercontent.com/vlfeat/vlfeat/master/vl/imopv.c),
  [VLFeat descriptor API](https://www.vlfeat.org/api/sift_8c.html), and
  [COLMAP's CPU SIFT conversion/quantization](https://github.com/colmap/colmap/blob/main/src/colmap/feature/sift.cc).

- **Opt-in cohesive VLFeat/COLMAP DoG detector (2026-08-29).** Added
  `SiftConfig::vlfeat_compatible_detector` and
  `--sift-vlfeat-compatible-detector`. The default-off path mirrors the
  source detector's first octave `-1` with bilinear 2× upsampling, SIFT's
  four-sigma separable smoothing, `0.02/S` peak threshold interpretation,
  strict 3×3×3 extrema, five-iteration 3-D quadratic localization, normalized
  spatial Hessian edge rejection, Gaussian-layer orientation assignment, and
  deterministic large-scale-first capping. It exports the refined centers in
  COLMAP's `+0.5` pixel convention so it composes with the compatible
  descriptor; affine and the partial orientation-selector flag are rejected
  for this isotropic mode. Unit tests cover bilinear first-octave upsampling,
  subpixel/subscale quadratic recovery, edge rejection, cap ordering,
  determinism, default identity, and CLI validation.
  At 4096 features/image, the detector produced **154,884** keypoints versus
  **208,785** COLMAP rows. Nearest-neighbor spatial repeatability (ours→COLMAP)
  was **92.6% at 0.5 px / 93.3% at 1 px** overall and **91.6% / 92.2%** on
  `DSC_0297--0308`; among the spatial matches, scale ratio ≤1.25 was **92.6%**
  overall and **91.5%** on the critical subset. The reverse COLMAP→ours rate
  was **68.4%** overall and **67.2%** critical at 0.5 px, primarily reflecting
  the 4096 cap rather than a missing detector locus. Focused ratio-0.8 +
  cross-check + full-verifier bridge counts were `0305--0306=329/283`,
  `0305--0307=125/88`, and `0306--0307=606/557` (raw/accepted).
  Pairing these features with the compatible descriptor and the safe stack
  (ratio 0.9 + guided/full verification, plain incremental, `pnp100k`, min-PnP
  8, conflict recovery, post-refinement, final iterative) produced
  **603/703** verified pairs, **38/38** registered images, **11,533** tracks,
  **0.371 px** mean reprojection, and **142.99 cm** Sim(3) center RMSE. The
  same compatible descriptor without the detector mode was **24/38** and
  **415.43 cm**, so registration count improved materially, but the result is
  still far behind the fixed-COLMAP-descriptor control (**18.62 cm**) and is
  not an accuracy-champion update. Features were generated by the detector
  and saved under `/tmp/vlfeat_detector_features_20260829`; the mapper model
  and log are `/media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/runs/sift_vlfeat_detector_descriptor_l2`
  and `/tmp/sift_vlfeat_detector_descriptor_l2.log`. The implementation follows
  [VLFeat's SIFT detector](https://raw.githubusercontent.com/vlfeat/vlfeat/master/vl/sift.c),
  [VLFeat image convolution](https://raw.githubusercontent.com/vlfeat/vlfeat/master/vl/imopv.c),
  and [COLMAP's SIFT options](https://github.com/colmap/colmap/blob/main/src/colmap/feature/sift.h).

- **VLFeat/COLMAP detector cap A/B (2026-08-29).** A controlled `4096` vs
  `8192` compatible-detector/descriptor run found that the larger cap is a
  negative result, not an accuracy-champion update: features increased from
  **154,884** to **292,639**, with the 4096 output an exact per-image prefix of
  the 8192 output, but ours→COLMAP spatial repeatability at 0.5 px fell from
  **92.57%** to **68.24%** overall (critical `0297--0308`: **91.59%** to
  **77.50%**), while the reverse rate rose **68.44%** to **94.77%**. On the
  focused `.8` cross-check bridges, raw/accepted counts changed only
  `0305--0306: 329/283→350/304`, `0305--0307: 125/88→137/91`, and
  `0306--0307: 606/557→606/557`. With the identical safe mapper stack, the
  verified graph changed **603/703→516/703** pairs and **125,125→205,372**
  inlier correspondences; registration fell **38/38→36/38**, despite tracks
  rising **11,533→17,884**, and Sim(3) RMSE worsened **142.99→223.79 cm**
  (mean reprojection **0.371→0.356 px**, center-distance correlation
  **0.904→0.793**). The current global sigma/contrast ordering is directionally
  consistent with VLFeat's coarse-level-first source order, but exact COLMAP
  CPU capping retains complete DoG levels and requires level metadata not yet
  carried by `SiftKeypoint`; no unvalidated edge-pruning gate was added.
  Outputs: `/tmp/vlfeat_detector_features_8192_20260829` and
  `/tmp/sift_vlfeat_detector_descriptor_l2_8192.log`. Next experiment is an
  exact level-group cap/source-order diagnostic (with per-edge quality data)
  before any verified-edge quality gate.

- **Opt-in scale-adaptive SIFT gradients (2026-08-29).** Added
  `SiftConfig::scale_adaptive_gradients` and the default-off
  `--sift-scale-adaptive-gradients` path. It builds a deterministic
  original-image Gaussian pyramid from `sigma_input`, computes central
  differences on each blurred level, maps fixed keypoints into the selected
  octave (`x,y` divided by `2^octave`), and interpolates adjacent levels in
  log scale; the legacy direct-source descriptor path is left untouched when
  the flag is off. Tests cover constant/linear gradients, level selection,
  octave coordinate round-trip, determinism, descriptor distinction, and
  default identity; `diagnose_cli_tests` covers the default-off/parseable CLI.
  A bounded fixed-COLMAP-keypoint mag-3 probe used the existing normalized
  `images_1600x1066` and COLMAP raw index set on `0305--0306`, `0305--0307`,
  `0306--0307`. Legacy raw/accepted/overlap(P/R) was
  **207/183/122 (58.94/34.56%)**, **51/30/21 (41.18/13.04%)**, and
  **565/526/453 (80.18/71.90%)**; scale-adaptive became
  **202/156/118 (58.42/33.43%)**, **92/0/20 (21.74/12.42%)**, and
  **512/453/415 (81.05/65.87%)**. The extra raw matches on `0305--0307`
  were geometrically degenerate, so no full reconstruction or accuracy
  champion was claimed. The source contract is the VLFeat
  [descriptor API](https://www.vlfeat.org/api/sift_8c.html) and its
  [reference implementation](https://raw.githubusercontent.com/vlfeat/vlfeat/master/vl/sift.c);
  the next parity step is matching its exact gradient rasterization and
  descriptor histogram/quantization, not loosening verification. Validation
  passed with `cargo test --release -p visloc-vision sift --lib`,
  `cargo test --release --example unordered_sfm_demo diagnose_cli_tests
  --features image-io`, `cargo check --release --example unordered_sfm_demo
  --features image-io`, and `git diff --check`; no full reconstruction was
  run because the focused A/B was not an improvement.

- **Standards-aligned SIFT orientation peaks (2026-08-28).** Added the
  default-off `SiftConfig::standard_orientation_peaks` and
  `--sift-standard-orientations` path: six circular three-tap smoothing
  passes, strict local maxima at least 80% of the dominant peak, and bounded
  finite-safe parabolic angle interpolation. Focused ratio-0.8 NN+cross-check
  diagnostics changed `DSC_0305--0306` from **83/64** raw/accepted to
  **82/65**, `0305--0307` from **13/DEGENERATE-0** to **17/DEGENERATE-0**,
  and `0306--0307` from **237/201** to **216/192**; at ratio 0.9 the
  `0305--0307` edge became **73/24 Uncalibrated** (legacy **68/0
  Degenerate**). The full replacement run used SIFT@4096, exhaustive ratio
  0.9 + guided/full verification, plain incremental + `pnp100k`, min-PnP 8,
  geometry recovery + post-refinement + final iterative refinement, and
  produced **330/703** verified pairs, **42,890** inliers, **28/38** images,
  **5,368 tracks**, **0.620 px** mean reprojection, and **507.77 cm** Sim(3)
  centre RMSE (log/model:
  `/tmp/visloc_oursift_standard_orientation_20260828.log` and
  `/tmp/visloc_oursift_standard_orientation_20260828`). Legacy no-extra
  reference was **340/703**, **26/38**, **5,195 tracks**, **0.569 px**, and
  **499.89 cm**; the standard mode left `DSC_0299--0308` missing and is not
  an accuracy-champion update. Extracted coordinate-duplicate rows, a proxy
  for secondary orientations because the feature export omits orientation,
  fell from **28.91%** to **17.58%** overall and from **24.41%** to **17.24%**
  on the three bridge images. An append-only corrected-orientation bank was
  not added: `FeatureSet` has no orientation/scale identity, and duplicating
  endpoints would alter track topology; a safe bank needs explicit mapped
  feature identities. Added unit coverage for adjacent-bin collapse, separated
  peaks, circular boundaries, interpolation/invalid inputs, determinism, and
  default identity. This remains an opt-in diagnostic result; next candidate
  is an affine-shape-aware feature bank rather than more orientation copies.

- **SIFT orientation-multiplicity audit (2026-08-28).** The existing
  `assign_orientations` path already emits every 36-bin histogram bin at or
  above 80% of the maximum (`max_orientations=0` is the unchanged unlimited
  default), so no secondary-orientation implementation was added. On the
  normalized 1600×1066 courtyard set, our SIFT@4096 had **145,188** keypoints
  over **104,512** same-(x,y) loci, including **40,676** same-scale duplicate
  orientation rows (**28.02%**; 1.389 orientations/locus). The full COLMAP
  six-column affine-keypoint database had **208,785 / 178,359** and **30,426**
  duplicate rows (**14.57%**; 1.171/locus); on critical `DSC_0297–0308`, our
  duplicate rate was **27.42%** (10,611/38,692) versus COLMAP **14.89%**
  (9,778/65,668). Our same-scale duplicate angle gaps were **6,408/19,408
  (33.0%) ≤20°** (critical images 25.1–35.4%), whereas COLMAP affine
  orientations (`atan2(a21,a11)`) were **283/30,551 (0.93%) ≤20°** with
  median gap **158.4°**. Thus the current detector over-generates adjacent
  histogram-bin copies rather than lacking secondary peaks. On three critical
  raw-oracle bridges, after ≤3 px coordinate transfer, ratio-0.8 mutual-NN
  oracle hits conditioned on a multi-orientation locus were
  `0305–0306=14/98` versus no-secondary `13/81`, `0305–0307=3/38` versus
  `1/31`, and `0306–0307=64/162` versus `57/144`—no consistent benefit.
  This is a diagnostic negative (no A/B or accuracy-champion update); the
  next detector candidate is a separately gated local-maximum/smoothing
  correction or affine-shape path, not more secondary orientations.

- **Per-correspondence geometric confidence track ordering (2026-08-28).**
  Added default-off `--geometric-confidence-tracks` and
  `IncrementalSfmConfig::geometric_confidence_tracks`. The opt-in builder sorts
  E-supported correspondences from `CALIBRATED` pairs by finite normalized
  Sampson residual, then verified/essential pair support and deterministic
  endpoint indices; `UNCALIBRATED` (F-won), homography, watermark, multiple,
  degenerate, missing, invalid, and out-of-range entries use the existing
  pair-level fallback, so incomparable pixel/model residuals are never mixed.
  Added stable NaN/degenerate handling, residual-priority conflict and
  permutation tests, and kept the default track output byte-identical. The
  strongest our-SIFT stack (SIFT@4096 + threshold-0.01 2048 extras on
  `DSC_0299--0308, DSC_0320--0321`, append-only extras + descriptor ensemble
  3.0, exhaustive ratio-0.9 guided/full verification, plain incremental,
  `pnp100k`, min-PnP 8, recovery + post + final iterative refinement) reached
  **27/38** (growth 26 then post +1), **6,553 tracks**, **5.866 px** mean
  reprojection, and **498.60 cm** Sim(3) centre RMSE; `DSC_0306` remained
  **6→3** PnP inliers and `DSC_0307` had **1** candidate correspondence.
  Exact our-SIFT invocation: `target/release/examples/unordered_sfm_demo
  --feature-extractor sift --images-dir
  /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/images_1600x1066 --width 1600
  --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 --exhaustive
  --min-matches 20 --sift-max-keypoints 4096
  --sift-extra-keypoints-stems DSC_0299,DSC_0300,DSC_0301,DSC_0302,DSC_0303,DSC_0304,DSC_0305,DSC_0306,DSC_0307,DSC_0308,DSC_0320,DSC_0321
  --sift-extra-keypoints 2048 --sift-extra-contrast-threshold 0.01
  --sift-extra-matches-append-only --sift-append-descriptor-magnification 3.0
  --match-ratio 0.9 --guided-matching --verification-mode full --mapper
  incremental --pnp-max-iterations 100000 --min-pnp-inliers 8
  --geometry-guided-conflict-recovery --post-refinement-registration
  --final-iterative-refinement --geometric-confidence-tracks
  --out-colmap /tmp/visloc_oursift_geometric_confidence_20260828` (with
  `VISLOC_SFM_DEBUG=1 VISLOC_SFM_DEBUG_IMAGES=20,21`). The matching-input
  comparison used `target/release/examples/unordered_sfm_demo
  --feature-extractor files --features-dir
  /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_features_export
  --feature-suffix _features.txt --image-suffix .png --width 1600 --height 1066
  --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6
  --import-matches-supplement-file
  /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_bridge_matches_import.txt
  --exhaustive --min-matches 20 --match-ratio 0.8 --verification-mode full
  --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8
  --geometry-guided-conflict-recovery --post-refinement-registration
  --final-iterative-refinement --geometric-confidence-tracks
  --out-colmap /tmp/visloc_colmap_geometric_confidence_20260828`.
  On the COLMAP-feature + 454-pair bridge-supplement recovery+post champion,
  the mode preserved **38/38** and produced **21,137 tracks / 0.290 px** but
  scored **4.06 cm**, versus the same-input default legacy builder's
  **2.84 cm / 0.283 px / 21,338 tracks**. Registration count therefore passes
  but the quality gate fails; neither result updates the accuracy champion and
  the strategy remains explicitly opt-in. Logs/models:
  `/tmp/visloc_colmap_geometric_confidence_20260828.log` and
  `/tmp/visloc_colmap_geometric_confidence_20260828` (our-SIFT model:
  `/tmp/visloc_oursift_geometric_confidence_20260828`). Verification:
  `cargo test --release -p visloc-slam --lib incremental_sfm` (**48 passed**),
  `cargo test --release --example unordered_sfm_demo --features image-io diagnose_cli_tests`
  (**4 passed**), `cargo check --release --example unordered_sfm_demo
  --features image-io`, and `git diff --check`.

- **SIFT descriptor-domain audit and magnification A/B (2026-08-28).** Our
  descriptors are L2-normalized (norm ≈ **1**) while the imported COLMAP
  descriptors are quantized uint8 (norm ≈ **512**); the global scale cancels
  in NN/ratio distances, so quantization is not the primary discrepancy. Our
  descriptor cell uses **8σ** versus the roughly **3σ** COLMAP/VLFeat domain;
  however, ratio-0.8 mutual-NN overlap with the transferred 8-px oracle stayed
  low on temporal bridges (for example `0305--0306`: **21/222** recall,
  `0306--0307`: **88/366**). Added default-off
  `--sift-descriptor-magnification` (default **8.0**) with a byte-identical
  explicit-8 path and deterministic tests, then ran the established
  normalized-our-SIFT stack at **3.0** with threshold-0.01 append-only extras,
  ratio **0.9** + guided/full verification, recovery + post, min-PnP **8**,
  and final iterative refinement (log:
  `/tmp/visloc_diag_oursift_descmag3_appendonly_20260828.log`). The run had
  **274/703** verified pairs, **36,567** inlier correspondences, **6,268**
  tracks, and finished **26/38** at **0.598 px** (append-only 8σ reference:
  **342/703**, **50,707**, **5,195**, **25/38**, **0.388 px**). It produced
  nondegenerate accepted edges `19--20=116`, `19--21=20`, and `20--21=354`,
  but growth still had **0** 2D--3D correspondences for both `DSC_0306/0307`;
  no verified registered bridge was recovered. This is a diagnostic negative,
  not an accuracy-champion update. The next code-level frontend candidate is
  affine-shape-aware descriptor sampling: the COLMAP database keypoints retain
  six-column affine metadata, whereas this SIFT feature path retains only
  `(x,y)` for exported matching.

- **Primary-preserving SIFT descriptor ensemble (2026-08-28).** Added the
  default-off `--sift-append-descriptor-magnification X` NN-only path (it
  rejects LightGlue): one detector pass produces the established primary
  descriptors, a second bank is described at `X` for the exact same keypoint
  indices, and deterministic NN+ratio/cross-check matches from that bank are
  appended only when neither endpoint is already used by the primary result.
  Bank lengths are asserted against the feature indices, so no keypoints or
  tracks are duplicated; the mode composes with
  `--sift-extra-matches-append-only`. Focused `--diagnose-pair` runs at
  magnification `8.0 + 3.0`, ratio `0.8`, cross-check, gave primary → ensemble
  raw/accepted results `0305--0306: 83/64 → 158/118`,
  `0305--0307: 13/DEGENERATE-0 → 50/21`, and
  `0306--0307: 237/201 → 342/283` (same **145,188** keypoints); the
  `0305--0307` bridge is therefore geometrically nondegenerate under the
  alternate bank, not merely a raw-match increase. The full normalized
  courtyard command used SIFT@4096, exhaustive/full verification, ratio `0.9`
  + guided matching, plain incremental + `pnp100k`, min-PnP `8`, geometry
  recovery + post-refinement + final iterative refinement; the exact no-extra
  invocation was
  `unordered_sfm_demo --feature-extractor sift --images-dir .../images_1600x1066 --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 --sift-max-keypoints 4096 --sift-append-descriptor-magnification 3.0 --exhaustive --min-matches 20 --match-ratio 0.9 --guided-matching --verification-mode full --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8 --geometry-guided-conflict-recovery --post-refinement-registration --final-iterative-refinement`
  (log/model: `/tmp/visloc_diag_oursift_descriptor_ensemble_20260828.log`).
  Without extra keypoints, ensemble verification was **382/703**, growth/post ended at
  **24/38** (no target post registration), **5,401 tracks**, **0.954 px**,
  Sim(3) centre RMSE **279.53 cm**; the primary-only comparison was
  **340/703**, **26/38**, **5,195 tracks**, **0.569 px**, **499.89 cm**.
  With the established threshold-`0.01` 2048-extra append-only setup on
  `DSC_0299--0308, DSC_0320--0321`, ensemble verification was **403/703**,
  growth **26/38**, post added images **11/12** (`70→9` / `79→9` PnP
  correspondences/inliers), and final **28/38**, **5,956 tracks**, **0.413
  px**, Sim(3) RMSE **375.53 cm** (log/model:
  `/tmp/visloc_diag_oursift_descriptor_ensemble_extra_20260828.log`; exact
  extra flags: `--sift-extra-keypoints-stems DSC_0299,DSC_0300,DSC_0301,DSC_0302,DSC_0303,DSC_0304,DSC_0305,DSC_0306,DSC_0307,DSC_0308,DSC_0320,DSC_0321 --sift-extra-keypoints 2048 --sift-extra-contrast-threshold 0.01 --sift-extra-matches-append-only`; primary-preserving extra-only reference:
  **342/703**, **25/38**, **5,195 tracks**, **0.388 px**). The target far
  images still had only **6** track correspondences for `DSC_0306` and none
  sufficient for `DSC_0307`; descriptor-ensemble gains do not survive track
  construction/growth and are not an accuracy-champion update.

- **Confidence-ordered conflict-aware track builder (2026-08-28).** Added the
  default-off `--confidence-ordered-tracks` / `IncrementalSfmConfig::confidence_ordered_tracks`
  policy. It orders verified correspondence edges by retained pair inlier
  count, then essential-inlier count and image/keypoint indices, and rejects a
  merge when the two components already contain the same image. This uses only
  confidence metadata retained by `PairwiseMatches`; no synthetic per-match
  score is introduced. Unit tests cover preservation of a strong 3-view chain
  against a weak conflicting edge, permutation determinism, and the unchanged
  default union-find result. On the strongest normalized our-SIFT stack
  (`--sift-max-keypoints 4096`, threshold-`0.01` 2048 extras on
  `DSC_0299--0308, DSC_0320--0321`, append-only extras, descriptor ensemble
  `3.0`, exhaustive ratio `0.9` + guided/full verification, plain incremental,
  `pnp100k`, min-PnP `8`, recovery + post + final iterative refinement), the
  flag rejected **6,078** conflicting edge merges and increased pre-filter
  tracks/observations to **21,413/66,570** (baseline
  **17,253/45,437**). Growth reached **27/38** (baseline **26/38**), post
  refinement added images **10/12**, and final registration reached **29/38**
  with registered stems `DSC_0286--0299, DSC_0309--0319, DSC_0320--0323`;
  only `DSC_0300--0308` remained missing. `DSC_0306` still had only
  **6** PnP correspondences and `DSC_0307` remained below the candidate floor.
  Final output was **6,001 tracks**, **4.203 px** mean reprojection, and
  Sim(3) centre RMSE **458.59 cm**, versus the confidence-off ensemble's
  **28/38**, **5,956 tracks**, **0.413 px**, **375.53 cm**. The extra
  registration is therefore a pose/quality regression, not a champion update;
  the new policy remains explicitly opt-in and the next track change needs a
  stronger per-match geometric/confidence signal than pair-level inlier count.

- **Far-orbit verifier and track-conflict follow-up (2026-08-28).** The
  append-only our-SIFT run was rebuilt with
  `VISLOC_SFM_DEBUG_DUMP_PAIR_OUTCOMES=1` and
  `VISLOC_SFM_DEBUG_DUMP_MATCH_INDICES=1` (log:
  `/tmp/visloc_diag_oursift_extra2048_t001_appendonly_outcomes2_20260828.log`).
  For `DSC_0306` (image 20), every edge to an image in the registered
  component was rejected as `DEGENERATE` (raw **26--50**) or `TOO_FEW` (raw
  **15/19**); for `DSC_0307` (image 21), the same held (raw **22--51**),
  except `10--21` was `UNCALIBRATED`, **15** accepted inliers, and rejected
  below the **8**-inlier floor. Their useful edges were confined to the
  unregistered island (`20--21`: **523→379**, `19--20`: **303→185**), so a
  stem-only verifier relaxation has no geometrically supported registered
  bridge to admit. For `DSC_0320/0321/0322`, our target endpoint duplicate
  rates were **33.5/29.8/35.2%**, with **250/205/212** conflict components;
  COLMAP features were **56.6/53.4/52.7%** duplicate endpoints but only
  **142/110/104** conflict components. Pair-specific conflicts reached
  **49--53%** on several near-neighbour edges for our SIFT versus roughly
  **6--16%** for COLMAP. However, conflict-component co-located target
  indices within **0.5 px** were only **10/146 (6.8%)**, **9/131 (6.9%)**,
  and **4/131 (3.1%)** (images 34/35/36), comparable to COLMAP's
  **6/62 (9.7%)**, **3/52 (5.8%)**, and **3/52 (5.8%)**. Thus the evidence
  rejects same-image orientation-duplicate canonicalization as the smallest
  safe fix: the dominant loss is inconsistent descriptor correspondence
  chains, not co-located-index selection. A low-inlier retry would face only
  **6/59--62** post-PnP inliers (a **9.7--10.5%** ratio); `0306/0307` have no
  verified registered-neighbour support, while `0320--0322` have support but
  no measured pose-consistency margin. No new relaxation or track mutation
  was therefore added.
  Existing oracle bridge runs remain the ranked follow-up: derive a
  production matcher change from the **28/38** 8-px transferred-match upper
  bound, then test a bounded recovery→PnP retry with reprojection,
  cheirality, and verified-neighbour consistency gates.

- **Append-only our-SIFT image-level diagnosis (2026-08-28).** Re-ran the
  primary-preserving extra-keypoint stack with the default-off raw/verified
  pair dump (`VISLOC_SFM_DEBUG_DUMP_MATCH_STATS=1
  VISLOC_SFM_DEBUG_DUMP_PAIRS=1`):
  `--feature-extractor sift --images-dir /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/images_1600x1066 --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 --exhaustive --min-matches 20 --sift-max-keypoints 4096 --sift-extra-keypoints-stems DSC_0299,DSC_0300,DSC_0301,DSC_0302,DSC_0303,DSC_0304,DSC_0305,DSC_0306,DSC_0307,DSC_0308,DSC_0320,DSC_0321 --sift-extra-keypoints 2048 --sift-extra-contrast-threshold 0.01 --sift-extra-matches-append-only --match-ratio 0.9 --guided-matching --verification-mode full --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8 --geometry-guided-conflict-recovery --post-refinement-registration --final-iterative-refinement`. It produced **342/703** verified pairs and **50,707** inlier correspondences; growth stopped at **24/38**, recovery accepted **485 tracks / 2,681 observations**, and the final model remained **25/38**, **5,195 tracks**, **0.388 px**. For the final 25-image component, verified edges touching `DSC_0299–0305/0308` numbered **9/5/2/3/4/3/3/2** of 25 possible registered neighbours; `DSC_0306/0307` had **0/0** verified registered-neighbour edges (raw edges ≥20 existed for **35/37** and **34/37** pairs), and the stop diagnostic reported **0 2D–3D tracks** for both. `DSC_0320/0321/0322` had **14/12/12** verified registered-neighbour edges with **1,129/1,011/1,024** inlier matches, but only **107/102/106** track correspondences and **7/5/7** growth inliers; post-refinement was **59/62/57→6/6/6**. By comparison, the fixed-COLMAP-feature run had **647/697/1,251→529/595/1,132** for `0320/0321/0322` and post placed `0306/0307` at **200→151 / 191→107**, reaching **38/38**. This separates a registered-component verification/connectivity gap for `0306/0307` from mixed sparse verification/topology loss for `0299–0305/0308`, and severe track-topology compression for `0320–0322`; scheduling alone is insufficient. No new low-inlier threshold was enabled: accepting the observed **6/59–62** post inliers without a pose-quality gate is unsafe. The existing append-only switch remains default-off and showed no completeness gain; next ranked work is targeted accepted-index/co-located-keypoint comparison, then bridge augmentation for `0306/0307`, followed by a guarded recovery→PnP retry. Diagnostic result only; **no accuracy champion update**.

- **P0 COLMAP-feature pair/track provenance diagnosis (2026-08-28).** On the fixed COLMAP-feature set, ratio **0.8 + cross-check** raw-match precision/recall was **98.91%/98.98%**; the ratio **0.9** raw increment was **99.43% oracle-external**. `DSC_0306`/`DSC_0307` had **zero verified-pair loss**, yet full-raw PnP was **204→144 / 176→133** versus supplement+NN **22→7 / 6→3**. Provenance support from `DSC_0304`/`DSC_0305` fell **182/188→20/20** (0306) and **114/153→3/4** (0307). Oracle ablation on the `DSC_0315–0317` triangle reached **38/38** with any two of its three added edges, while every single edge remained **36/38**; accepted-set differences were only **~10–12**, exposing track-topology sensitivity. Diagnostic/oracle result only; **no accuracy champion update**. Next: verify co-located orientation-duplicate indices.

- **Graph-track and geometry-conflict-recovery A/B (2026-08-28).** On COLMAP features + the 454-pair bridge supplement + NN fallback + plain incremental + `pnp100k`, `--track-source graph` matched union-find at **36/38**, **22→7** PnP inliers for `DSC_0306`, **6→3** for `DSC_0307`, **18,405 tracks**, and **0.282 px** mean reprojection; only the graph diagnostic's exposed conflict accounting changed (`803`/`9,928` dropped conflict components/observations → `0`/`0`), with the retained track topology unchanged. The new default-off `--geometry-guided-conflict-recovery` proposed **676 tracks / 5,515 observations** and passed its guarded BA (`0.282451→0.281792` clean mean; recovered mean **0.285697**), but remained **36/38** because recovery runs after plain growth/refinement and does not re-run PnP; image 20/21 still had **22→7 / 6→3**. The target provenance contained **43/49** and **23/24** conflicted components/observations. This is a diagnostic result, not a champion update; the next minimal change is a bounded recovery→PnP pass during/after growth, not another track-source swap.

- **Plain post-refinement registration A/B (2026-08-28).** Exposed the existing bounded completion pass as default-off `--post-refinement-registration` and allowed it after plain final refinement as well as COLMAP-style refinement. On the same bridge-supplement B stack, post-only made no registration progress (`DSC_0306` **19→9**; `DSC_0307` stayed below the candidate floor), remaining **36/38**, **18,405 tracks**, **0.282 px**. Geometry recovery + post-refinement tried **DSC_0306 27→9** and **DSC_0307 7→3**, still **36/38**, **19,081 tracks**, **0.282 px**; no final iterative score was warranted. Recovery-added tracks improve structure density but not the missing-image PnP inliers, so the next change must alter/augment correspondence support or use a bounded registration method beyond this single PnP attempt.

- **Post-refinement PnP threshold A/B (2026-08-28).** With `--min-pnp-inliers 8 --post-refinement-registration --pnp-max-iterations 100000` on the same B stack, post-only registered `DSC_0306` (**19→9**) then `DSC_0307` (**69→33**) and reached **38/38**; geometry recovery + post registered **27→9** then **73→27** and also reached **38/38**. Adding `--final-iterative-refinement` and scoring all 38 centres gave **2.92 cm RMSE** (post-only; median **2.33 cm**) and **2.84 cm RMSE** (recovery+post; median **2.24 cm**); final mean reprojection was **0.284/0.283 px**. Sim(3)-aligned per-camera errors for `DSC_0306/0307` were **1.28/1.06 cm** (post-only) and **1.61/1.90 cm** (recovery+post); the worst remained `DSC_0316` at **6.95/7.09 cm**. This is an A/B diagnostic result, not a benchmark champion update; the gain comes from accepting low-inlier PnP, so a later safety gate should validate pose quality before promotion.

- **Our-SIFT low-inlier post-refinement A/B (2026-08-28).** On normalized `images_1600x1066`, SIFT@4096 (145,188 keypoints), exhaustive ratio **0.9** + guided matching, full verification (**340/703**), plain incremental + `pnp100k`, `min-pnp-inliers=8`, and final iterative refinement: post-only reached **25/38** (growth stopped at 25; no post image registered — `DSC_0298`/image 12 was **67→7**), while geometry recovery + post reached **26/38** (recovery **472 tracks / 2,528 observations**, post registered only `DSC_0298`/image 12 at **80→12**). Final reprojection was **3.123/0.569 px**; registered-subset Sim(3) scores were **493.90/499.89 cm RMSE** (25/26 images, not a full-scene score), versus the existing **23/38** pnp100k+final-polish baseline. A gained `DSC_0322` in both runs and `DSC_0298` only with recovery account for the **+2/+3** registrations; missing sets were post-only `DSC_0298–0308, DSC_0320–0321` and recovery+post `DSC_0299–0308, DSC_0320–0321`. The recovery flag adds one image but does not approach 38/38; the low-inlier threshold's extra registrations remain an honest negative for our-SIFT accuracy and require a pose-quality gate.

- **Our-SIFT low-threshold extra-keypoint comparison (2026-08-28).** Added default-off `--sift-extra-contrast-threshold` (`None` preserves the primary extraction and the existing 0.5 px spatial-novel append path). On normalized `images_1600x1066`, ratio **0.9** + guided matching, full verification, plain incremental + `pnp100k`, `min-pnp-inliers=8`, geometry recovery + post-refinement + final iterative refinement, adding **2048** extra keypoints at threshold **0.01** on the 12 missing-stem targets raised total features from **145,188** to **166,044** and verified pairs from **340/703** to **349/703**, but recovery tracks/observations changed **472/2,528→414/1,940** and final registration fell **26/38→25/38** (mean reprojection **0.569→2.857 px**). No post-refinement candidate registered in the extra run; no full-scene score is reported and this is an honest diagnostic negative, not an accuracy-champion update. Lowering the extra-only threshold increases raw/verified support without recovering the missing cameras, so the next test remains co-located orientation-duplicate index validation.

- **Primary-preserving append-only extra matching (2026-08-28).** Added default-off `--sift-extra-matches-append-only`: each image records its actual primary SIFT count before extras; NN+ratio/cross-check matches on those prefixes are preserved exactly, and only deterministic, non-conflicting full-set candidates involving an extra descriptor are appended. A synthetic distractor test demonstrates that ordinary full matching replaces a primary Lowe match while append-only retains it. On the same normalized our-SIFT stack with **2048** extras at threshold **0.01** for `DSC_0299–0308, DSC_0320–0321`, append-only produced **342/703** verified pairs, growth **24/38**, recovery **485 tracks / 2,681 observations**, post registered `DSC_0297` (**82→14**), and final **25/38**, **5,195 tracks**, **0.388 px** mean reprojection. The normal-extra run was **349/703**, **25/38**, **414/1,940**, and **2.857 px**; the prior no-extra B was **340/703**, **26/38**, **472/2,528**, and **0.569 px**. Missing remained `DSC_0299–0308, DSC_0320–0322`; no full-scene score is reported. Preserving the baseline match set avoids the normal-extra replacement effect but does not unlock the far-orbit cameras; the next bottleneck remains sparse/incorrect far-stem support and track topology.

- **GT-gated rematch rejector (`--rematch-max-gt-bearing-deg`, `--rematch-gt-bearing-path`) (2026-08-28).** Oracle ceiling: reject free↔prior rematch E-gains whose essential bearing vs ETH3D GT exceeds threshold. Default off.
  - Champion baseline (no GT gate): **38/38 @ ~244 cm** (this session rebase).
  - `--rematch-max-gt-bearing-deg 60` + GT path: rejects **17** E-gains (including load-bearing `0315–0320` @63°, worst bridges `0316–0321` @89°) → **~433 cm** — **honest negative**. Even oracle bearing gate cannot surgically drop only the ~90° wrong-E bridges without breaking graph connectivity; confirms wrong inlier **sets** are load-bearing in the current basin.

- **GT-free rematch quality gates (`--rematch-guided-max-error-px`, `--rematch-require-calibrated`, `--rematch-max-mean-sampson`) (2026-08-28).** Production-facing admission filters on free↔prior rematch E-gains; `pair_essential_mean_sampson_error` in `global_sfm`. Default off.
  - Champion baseline: **38/38 @ ~244 cm**.
  - `--rematch-guided-max-error-px 1.0` (default guided gate is 2.0 px): rejects **6** E-gains → **~247 cm** — **near-neutral** (tighter epipolar densify does not straighten wrong-E basin).
  - `--rematch-require-calibrated`: rejects **47/47** rematch E-gains — **all** far-orbit bridges classify `Uncalibrated` despite non-zero E inliers → **~297 cm** — **honest negative** (Calibrated-only admission drops the entire rematch unlock path).
  - `--rematch-max-mean-sampson 0.003`: rejects **42** E-gains → **~330 cm** — **honest negative** (wrong-but-self-consistent E inliers still pass low mean Sampson on survivors; threshold at RANSAC floor is too blunt).

- **Rematch-only E/F model selection (`--rematch-min-e-f-inlier-ratio`, `--rematch-calibrated-prefer-essential`) (2026-08-28).** Lower the Calibrated gate and prefer E inliers on free↔prior rematch only (baseline rematch: **47/47** `Uncalibrated` because `E/F ≤ 0.95`). Default off.
  - `--rematch-min-e-f-inlier-ratio 0.7 --rematch-calibrated-prefer-essential`: **6** rematch gains `Calibrated`, **19** still `Uncalibrated` → **~261 cm** — **honest negative** vs ~244 cm baseline. Promoting F-suppressed pairs to Calibrated/E does not straighten the far-orbit basin; wrong E inlier sets survive the looser gate.

- **Rematch guided Lowe ratio (`--rematch-guided-lowe-ratio`) (2026-08-28).** Tighten the Lowe gate on COLMAP-style guided epipolar densify during free↔prior rematch (default **0.8** when unset). Default off (`None` = 0.8).
  - `--rematch-guided-lowe-ratio 0.65` on champion stack → **~285 cm** — **honest negative** (fewer guided false matches but still metres-scale; mean reproj **31.5 px** vs ~14.6 baseline).

- **COLMAP-style SIFT knobs on champion stack (2026-08-28).** `--sift-prefer-larger-scale --sift-full-pyramid --sift-max-orientations 2` @4096 kp (coarser-scale-first truncation, full octave walk, 2 ori cap): **38/38 @ ~381 cm** — **honest negative** vs ~244 cm legacy ranking. Detector policy change alone does not unlock courtyard; still need correct far-orbit correspondence set.

- **Incremental prior-ray guided rematch (`--rematch-prior-ray-guided`) (2026-08-28).** Triangulate free centres from incremental prior↔free essentials (`estimate_free_poses_from_prior_rays`), build approximate free poses, then pose-guide epipolar densify on those rematch pairs before standard verify. Default off.
  - Default `min_rays=2`, `min_e=25`: **4** free poses, **20/42** pairs pose-guided → **38/38 @ ~238 cm** — **near-neutral / slight** vs ~244 cm baseline (mean reproj **15.1 px**). Incremental metric frame yields few free pose estimates; directional but still metres-scale.
  - `min_e=20` + `--rematch-calibrated-prefer-essential`: still **4** poses → **~237 cm** — **near-neutral**.
  - `min_rays=1`: still **4** poses → **~237 cm** — **near-neutral** (ray-count not binding; `min_e=25` anchor edges are).
  - Fix: pose-guided failures now per-pair fallback to standard verify (previously dropped ~22/42 guided attempts with no fallback).
  - F-won ray expansion (accept `essential_matrix` + F inliers when `essential_matches` empty) @`min_e=15`: **7** free poses, **47/84** pose-guided → **~288 cm** — **honest negative** (reverted; F-won bearings poison incremental pose guide).

- **RootSIFT (`--sift-l1-root`) on champion stack (2026-08-28).** COLMAP-style L1-root descriptor normalization @4096 kp: **38/38 @ ~412 cm** — **honest negative** vs ~244 cm L2 champion (fewer tracks 5447 vs ~6290).

- **Rematch essential-only verify (`--rematch-verification-mode threshold-only`) (2026-08-28).** Skip F/H model selection on free↔prior rematch only (main pass stays `full`). Default off.
  - Champion stack: **38/38 @ ~393 cm** — **honest negative** vs ~244 cm (E-only rematch drops load-bearing F-supported bridges / wrong E without F cross-check).

- **COLMAP match import for courtyard oracle A/B (`--import-matches-file`, `--import-matches-supplement-file`, `--import-verified-pairs-file`) (2026-08-28).** `export_colmap_matches.py` / `export_colmap_bridge_matches.py` / `export_colmap_verified_pairs.py` on SSD; demo skips NN (`--import-matches-file`), supplements listed pairs with imported raw matches and NN-fallback elsewhere (`--import-matches-supplement-file`; indices must match loaded features — COLMAP export only), or skips verification (`--import-verified-pairs-file`, TVG inliers + config + E). Default off. SSD: `/media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/`.
  External SSD mounted at `/media/sasaki/aiueo1`; courtyard work dir `visloc-rs/eth3d/courtyard/` (images copied/normalized, GT symlinked). `colmap/colmap:latest` Docker image for true COLMAP baseline.
  - **Bridge-only match supplement on COLMAP features (2026-08-28).** `--feature-extractor files` + our NN + `--import-matches-supplement-file` (454 far-orbit bridge raw-match pairs from COLMAP): **380/703 verified**, plain incremental + `pnp100k + final-iterative-refinement` → **36/38 @ ~2.85 cm** (reproj **0.28 px**). Missing **`DSC_0306`/`DSC_0307`** (low-texture far stems). Same verified-pair count as full COLMAP-match oracle (**380**) but **2 images short of 38/38** — confirms bridge matching is load-bearing for far-orbit **connectivity**, and incremental still needs non-bridge NN quality for the sparsest stems. Full COLMAP-match import remains **38/38 @ ~3.4 cm**.
  - **Pitfall:** 14/38 PNGs are **1600×1065 or 1600×1067** (not 1066). COLMAP `--ImageReader.single_camera 1` silently skips them (`CAMERA_SINGLE_DIM_ERROR`); visloc-rs forces `--height 1066` and still loads them. First oracle run (mixed dims) registered only **24/38 @ ~3.6 cm** — misleading.
  - **Normalized images** (`images_1600x1066/`, Lanczos resize): COLMAP 4.1.1 SIFT@4096 + exhaustive match + incremental mapper → **38/38 @ ~1.71 cm** Sim(3) centre RMSE vs ETH3D laser GT (median **1.17 cm**). Confirms sub-cm ceiling is reachable on this scene with true COLMAP.
  - **COLMAP SIFT features → visloc hybrid champion** (`export_colmap_sift_features.py` from SQLite → `--feature-extractor files`, same hybrid/rematch flags): **38/38 @ ~137 cm** (median **96 cm**), mean reproj **2.35 px**, **382/703** verified pairs, **0** rematch E-gains. **Decisive split:** COLMAP features improve ~**2×** vs our SIFT champion (**~249 cm**) but remain **~80×** worse than true COLMAP mapper — binding gap is **both** frontend correspondence quality **and** global/hybrid positioning (good local reproj does not imply sub-cm Sim(3) centres). Rematch stack is inert on COLMAP features (already strong E bridges).
  - **COLMAP SIFT + COLMAP raw matches → visloc hybrid** (`--import-matches-file`; skips NN, still runs full verifier): **38/38 @ ~49 cm** (median **42 cm**), **380/703** verified, mean reproj **2.41 px**. **~2.8× better than COLMAP-SIFT+NN (137 cm)** with nearly identical verified-pair count — NN correspondence quality is a real unlock, not just pair coverage.
  - **COLMAP SIFT + COLMAP TVG inliers → visloc hybrid** (`--import-verified-pairs-file`; bypasses verify): **38/38 @ ~118 cm** (median **27 cm**), **401** pairs, mean reproj **2.99 px** — **honest negative vs raw+verify (49 cm)**. COLMAP's own two-view inlier sets are **not** the right oracle for our global/hybrid mapper; our verifier on raw COLMAP matches filters to a better edge set for bearing-graph positioning. Still **~69×** vs true COLMAP (**1.71 cm**) → **global/hybrid positioning** is the binding residual gap at the best-known correspondence input (**49 cm** floor).
  - **Mapper split on COLMAP-SIFT + COLMAP-raw-matches + our verify** (same 380 verified pairs, no rematch/champion extras): **incremental 38/38 @ ~66 cm** (mean reproj **0.44 px**, `--colmap-style`), **global 38/38 @ ~440 cm**, **hybrid 38/38 @ ~49 cm**. Global-alone collapses on COLMAP-grade input.
  - **Incremental BA schedule on COLMAP matches** (2026-08-28): **`--colmap-style` regresses courtyard** on oracle input. **plain incremental** (no `--colmap-style`): **38/38 @ ~8.7 cm** (reproj **0.28 px**) — **~7.5× better than `--colmap-style` (66 cm)** and **~5.6× better than hybrid (49 cm)** on same matches. `--colmap-style --structureless-registration`: still **66 cm**; handover (`+seed-trials 8 +pnp-max-iterations 100000`): **36/38 @ ~35 cm**. Simple one-shot final BA beats COLMAP IncrementalMapper schedule when correspondences are already strong.
    - **plain incremental knob sweep** (COLMAP matches, no `--colmap-style`): `seed_trials` 8/12/26/36/48/64 → **8.7 cm** (all neutral, identical 20120-track basin); `pnp-max-iterations 100000` → **38/38 @ ~5.6 cm** (reproj 0.28 px, 19506 tracks); `--track-source graph` → **8.7 cm** (neutral). **`--pnp-max-iterations 100000` best oracle so far @ ~5.6 cm** — still **~3.3×** vs true COLMAP (**1.7 cm**).
    - **`--final-iterative-refinement` (plain growth + COLMAP final polish only) (2026-08-28).** New flag: keeps simple growth schedule but swaps the one-shot final BA for `iterative_global_refinement` (multi-round BA + filter + re-triangulate). Default off. On COLMAP matches + plain incremental + `pnp-max-iterations 100000`: **38/38 @ ~3.4 cm** (reproj 0.28 px) — **~2×** vs true COLMAP (**1.7 cm**), **~1.6× better than pnp100k alone (5.6 cm)**. Confirms colmap-style **growth** regresses courtyard but colmap-style **final polish** is load-bearing.
    - **Our SIFT + new incremental stack (honest negative, 2026-08-28).** Same normalized images, `--verification-mode full`, `--mapper incremental` (no hybrid): plain **22/38 @ ~54 cm** (211/703 verified); **`pnp100k + final-iterative-refinement` → 21/38 @ ~421 cm**; **`--colmap-style` → 22/38 @ ~300 cm**. Oracle polish stack **requires COLMAP-grade correspondence density** (380 verified pairs); our SIFT still stops at 211/703 — **hybrid champion (38/38 @ ~249 cm) remains the completeness baseline** until matching unlocks far-orbit bridges.
    - **Our SIFT matching knobs (2026-08-28).** `--match-ratio 0.9` (default 0.8): **340/703 verified** (+62% vs 211), but plain incremental still **22/38** — same 16 missing far-orbit stems (`0297–0309`, `0320–0322`); registered subset Sim(3) **~40 cm** (vs ~54 cm @0.8). `--guided-matching` alone: **211/703**, **22/38 @ ~233 cm** — **neutral on pair count**. Ratio+guided: same 340/703 as ratio alone. **Diagnosis:** COLMAP verifies **183 bridge pairs** incident to far-orbit stems; our extra 129 verified pairs are mostly within the near component — incremental completeness still blocked on far-orbit bridges, not raw pair count. COLMAP extracts **~5200–5700 kp/image** vs our **4096 cap** (several far stems fall to **1500–2700 kp** at contrast=0.02). Hybrid+guided on default ratio: **38/38 @ ~357 cm** (211 verified + rematch).
    - **Our SIFT ratio0.9 + oracle incremental stack (2026-08-28).** Same 340/703 verified, `--match-ratio 0.9 --guided-matching`: **`pnp-max-iterations 100000` → 23/38 @ ~152 cm** (adds **`DSC_0309`** only vs 22/38 baseline); **`+ --final-iterative-refinement` → 23/38 @ ~185 cm** — **honest negative** (final polish regresses when graph is incomplete; reproj 6.4 px vs 0.35 px). **`--rescue-bridging` + final polish → 23/38** — **neutral** (verify graph already **1 connected component / 38**; rescue has nothing to bridge — failure is incremental registration/PnP on far orbit, not verify-graph fragmentation). **ratio0.9 + hybrid champion + final polish → 38/38 @ ~318 cm** — **honest negative** vs ~249 cm champion @ratio0.8 (more verified pairs do not straighten hybrid basin). **`--sift-max-keypoints 8192` + ratio0.9 + guided:** **391/703 verified** (+51 vs 4096 cap) but plain incremental still **22/38 @ ~148 cm** — **honest negative** (more kp does not unlock far-orbit bridge verification; still blocked on wrong/missing bridge pair set, not total density).

- **Rematch E-gain admission gates (`--rematch-min-chirality-margin`, `--rematch-prior-anchor`) (2026-08-28).** Before accepting free↔prior rematch E-gains, optionally require essential chirality margin ≥ threshold and/or that primary chirality points at a triangulation anchor from two other prior↔free essentials (`rematch_essential_admission_ok` in `global_sfm`). Default off.
  - Champion stack baseline (no new flags): **38/38 @ ~245 cm** (rebase variance vs historical ~230–241).
  - `--rematch-min-chirality-margin 0.15`: rejects **7** E-gains → **~254 cm** — **honest negative** (drops load-bearing bridges including `0311–0322`, `0312–0320`, `0317–0321`).
  - `--rematch-prior-anchor`: rejects **17** E-gains → **~256 cm** — **honest negative** (anchor rays share the same wrong basin).
  - Both gates default off; champion rematch admission unchanged.

- **Metric-prior chirality at global edge construction (`--metric-prior-chirality-edges`) (2026-08-28).** Triangulate free-camera centres from prior↔free essentials in the incremental metric frame (`estimate_free_centres_from_prior_rays`), then flip prior↔free edge chirality when the alternate bearing aligns better with that anchor (requires `--multi-hypothesis-edges`). Default off.
  - `min_rays=3` (5 anchors, 9 flips): **38/38 @ ~250 cm** — **honest negative** vs ~245 cm baseline.
  - `min_rays=2` (6 anchors, 11 flips): **~248 cm** — **honest negative**. Incremental metric frame cannot break the self-consistent far-orbit basin when anchor rays share the same flipped chirality.

- **GT bearing diagnostic + chirality oracle (`--diagnose-bearing-gt`, `--gt-chirality-oracle`) (2026-08-28).** Compare essential bearings vs ETH3D GT centres on far-orbit stems; oracle flips primary/alternate at edge build when alternate aligns better with GT.
  - **Post-rematch (40 E pairs):** mean primary error **25.1°**; **`alt_would_help=0/40`** — wrong bearings are **not** chirality-ambiguous (primary ≈ alternate error; e.g. `0316–0321` **89°** both hypotheses).
  - **GT chirality oracle:** **0** edge flips → **~248 cm** — same as baseline variance. **Decisive finding:** metres-scale RMSE is **not** unlocked by perfect chirality pick; rematch bridges carry **~90°** essential bearing error from **wrong E / wrong inliers**, not antipodal flip.
  - Worst post-rematch: `0316–0321` 89°, `0310–0322` 82°, `0316–0322` 78°. Best far-orbit: `0320–0321` 1.4°, `0319–0322` 1.2°. Next unlock: **match quality / verifier inlier set** on rematch pairs, not global chirality surgery.

- **Stored verifier E on `PairwiseMatches` + prior-guided free chirality (2026-08-28).** `PairwiseMatches::essential_matrix` records the full-verifier essential matrix; rematch paths populate it. Global edge construction was tested decomposing this matrix directly for prefer-E edges (instead of re-running E RANSAC on the E-inlier subset) — **38/38 @ ~267 cm** on the champion stack — **honest negative** (reverted to legacy `RelativePoseEstimator` path; field kept for future gated experiments). New default-off `--prior-guided-free-chirality`: before rotation averaging, flip prior↔free edges to the chirality alternate when multi-view prior rays to the same free camera agree better (requires `--multi-hypothesis-edges`). On champion stack: **40** candidate edges, **0** flips (wrong basin is self-consistent across prior rays) — **near-neutral** (~217–243 cm run variance vs ~239 cm baseline). Anchor-triangulation scoring variant (14 flips) → **~243 cm** — **honest negative** (not shipped).

- **Pre-global PnP seed priors (`--repnp-seed-free-as-priors`) (2026-08-28).** Triangulate prior-only structure, PnP free cams, promote successes to hard pose priors before averaging. Dropped hub stems (`--hybrid-drop-prior-stems`) are excluded from re-pinning. Default off.
  - `min_corrs=6` (no exclude): seeds `0296`+`0320` → **~309 cm** — **honest negative** (re-pins dropped hub; weak 0320 @6 inliers).
  - `min_corrs=25` (no exclude): seeds only `0296` → **~346 cm** — **honest negative**.
  - Exclude drop-stems + `min_corrs=6`: seeds `0320` @6 inliers → **~273 cm** — **honest negative**.
  - Exclude + `min_corrs=12`/`25`: **0** seeds → **~242 / ~244 cm** — **near-neutral** (champion variance). Far free cams have ~40–50 prior-anchored corrs but PnP rejects — prior structure still cannot place the far orbit.
  - `--essential-edge-weight-boost 3`/`5` on champion stack: **~375 / ~393 cm** — **honest negatives** (over-weighting rematch E bearings fights the prior metric frame).
  - `--calibrated-view-edges-only` (strict): **32/38 @ ~318 cm** — **honest negative** (drops F-won `10-11` and rematch bridges).
  - Same flag after allowing Uncalibrated pairs that still carry strong `essential_matches`: keeps `10-11` (138 E) but **37/38 @ ~319 cm** — **honest negative** vs champion ~230–241; F-only drop alone does not straighten the far orbit.

- **Pass-2 free-edge surgery from pass-1 poses (2026-08-28).** Diagnosis: far-orbit pairs (`0319–0322`, `0315–0322`, …) show **~90°** est-vs-GT relative bearings (chirality/antipodal contamination). New default-off flags: `--repair-free-edges-from-solved`, `--repair-free-edges-only-flipped`, `--repair-free-edges-stems`, `--drop-free-edges-antipodal`. On champion stack:
  - Rewrite all free-incident (71 edges, 26 flipped): **~318 cm** — **honest negative** (pass-1 basin wrong).
  - Rewrite flipped-only (24): **~349 cm**; far stems only (8): **~361 cm** — **honest negatives**.
  - Drop antipodal far stems (8): **~345 cm** — **honest negative**. Pass-1 geometry cannot fix flipped bearings; unlock remains at **two-view edge admission** before averaging.

- **Pose-guided post-global rematch + detector oracles (2026-08-28).** `--rematch-pose-guided-after-global` re-matches free↔prior under E from estimated absolute poses after the first hybrid global, accepts E-gains, re-runs global. `--rematch-pose-guided-gt PATH` substitutes COLMAP `images.txt` poses for guidance only (GT/oracle). Default off.
  - Est-pose guide on ~230–241 champion stack: **0** E-gain pairs → **~256 cm** (run variance vs rebase **~241 cm**) — no-op.
  - **GT-pose guide** (perfect E for Sampson): improved **1** pair (`0311–0321` E 0→41) → **~240 cm** — **near-neutral; does not unlock**. Correct epipolar guidance barely densifies calibrated bridges beyond the pre-global rematch; descriptor/admission quality remains binding.
  - GT guide + rematch **all** free (empty stems) pre-global: **~402 cm** — **honest negative**.
  - Drop prior `DSC_0323` (only far rematch-stem that is still an incremental prior) + rematch as free: **~336 cm** — **honest negative**.
  - `--require-essential-stems DSC_0320,DSC_0321,DSC_0322` @min_e=25 on champion stack: **~240 cm** — **near-neutral** (drops 6 F-only incident edges; does not straighten far free).
  - **OpenCV SIFT oracle** (files extractor, same hybrid stack): **~246 cm** @38/38 — still metres-scale; Rust SIFT champion slightly better. OpenCV + drop far priors `0296,0320–0323`: **~260 cm** — **honest negative**. Detector swap alone does not clear sub-cm.

- **Far-orbit free↔prior rematch (2026-08-28).** Extending `--rematch-free-vs-priors` stems to `DSC_0297,DSC_0320,DSC_0321,DSC_0322,DSC_0323` @0.9 guided on the stem-E base (with re-PnP optional):
  - All E-gains prefer-E + re-PnP: **38/38**, Sim(3) **~244 cm** (`0320` 924→315); worst `0321` ~642.
  - `--rematch-prefer-min-e-inliers 25` + re-PnP: **~230 cm** — **new hybrid champion** (drops weak E=23 bridge; keeps `0297` bridges at E=25–39). Hub `0296` ~139 / `0297` ~340; worst `0320` ~506 / `0315` ~490.
  - Same far stems, no re-PnP, `min_e=0`: **~231 cm** — ties champion; re-PnP not required for this basin.
  - `min_e=50` / `80`: **~383 / ~358 cm** — **honest negatives** (filters out load-bearing `0297` E=25–39 bridges).
  - `0297+0320` only / `0320` only: **~308 / ~336 cm** — need the full far free set. Still metres-scale; `0320–0319` E·GT≈0.996 confirms true E was F-suppressed.
  - Drop prior `0315` / `0315+0319` / `0315+0316+0317` on the ~230 stack: **~292 / ~392 / ~386 cm** — **honest negatives** (0315 prior is load-bearing despite high Sim3 residual).
  - Rematch all free cams: **~309 cm** — **honest negative** (0314→76 cm locally but poisons `0298` cluster). Far+`0314` prefer-all: **~393 cm**. Far+`0314` with `--rematch-prefer-strong-stems DSC_0314` @50/@80: **~255 / ~311 cm** — still behind ~230; 0314 improves locally without unlocking sub-cm.
  - Far @0.95: **~289 cm** — **honest negative**.
  - Drop `0315` prior + rematch it as free (± strong prefer): **~353–379 cm**; drop `0315+0319` + rematch: **~350 cm** — **honest negatives** (unpinning 0315 without a better free placement loses the metric frame).
  - `--rematch-tracks-use-essential` on the ~230 stack: **~274 cm** — **honest negative** (E-only tracks thin prior-anchored support; `0297` re-PnP still fails @8 corrs).

- **Free↔prior rematch + re-PnP stack (2026-08-28).** Combining `--rematch-free-vs-priors --rematch-guided` @0.9 on `DSC_0297` with `--repnp-free-from-priors --repnp-free-min-corrs 6` on the stem-E champion base: **38/38**, Sim(3) **~251 cm** — prior champion. Re-PnP accepts `0297` (16→8 inliers) and one other free cam; hub **`0297` ~205 cm / `0296` ~132 cm**, but **`DSC_0320` ~924 cm** still absorbs the bend. Prefer-E subsets / `min_e` filters / forcing `11-29,11-30` only all regress (**~262–462 cm**). Still metres-scale.
  - Same stack @0.85 + re-PnP: **~258 cm** (0297 PnP still fails @10 corrs) — near prior champion.
  - `--sift-contrast-threshold 0.0067` (COLMAP `0.02/3`) on the ~251 stack: **14/38** @ ~125 cm over registered — **honest negative** (completeness). @0.01: **36/38** @ ~455 cm — **honest negative**. Looser peak gate displaces the load-bearing 4096 set / stem-E graph.

- **Post-prior free↔prior rematch (`--rematch-free-vs-priors`) (2026-08-27).** After hybrid incremental priors (and stem drops) are known, rematch `--rematch-stems` (or all non-priors) only against prior cameras; accept only when **E inliers increase**; auto-`prefer-essential-pairs` on those gains. Skips pre-incremental stem rematch when this flag is on (avoids free–free densify). Guided seed prefers E inliers when ≥8. Default off. **Courtyard A/B** on stem-E champion stack:
  - Diagnosis: early `--rematch-guided` `0297` @0.85 only improved free–free `0297–0298` (E 82→123) → ~276–282 cm; not a prior bridge.
  - F-only free↔prior accept (pre-E-gate): 11 pairs, all E=0 → **~312–370 cm** — **honest negative**.
  - E-gated + auto prefer-E, `DSC_0297` @0.85 guided: improved `0297–0315`/`0316` (E 0→31/27); **38/38**, Sim(3) **~258 cm** (ties historical champion); `0297` **~555 cm** (was ~720–746). Pairwise median rel-err **~34%**.
  - Same @0.9: E gains on `0313/0315/0317`; **~259 cm**; pairwise median **~25%** — **ties champion**, best pairwise among near-champion runs. Per-cam hub **`0297` ~164 cm / `0296` ~155 cm** (huge local win) but error moves to **`DSC_0320` ~976 cm** — redistributes the bend rather than clearing it.
  - `--rematch-prefer-min-e-inliers 30` @0.85 (prefer only `11-29`/`0297–0315`): **~261 cm** but `0297` **~775 cm** — **near-neutral overall / hub regresses**; both E bridges were load-bearing for the hub win.
  - `--rematch-prefer-min-e-inliers 35` @0.9 (prefer only `0315`): **~372 cm** — **honest negative**.
  - Tracks-only (`min_e=999`, no prefer-E): **~289 cm** — **near-neutral**; denser F/E matches without edge prefer-E do not move GT shape.
  - Stems `0297,0298,0299` @0.85: 6 prefer-E pairs → **~338 cm** — **honest negative** (over-bridge).
  - Still metres-scale: prior↔hub E bridges help the hub locally but do not unlock sub-cm until far-camera contamination is controlled.

- **Stem-local SIFT extras (`--sift-extra-keypoints-stems` / `--sift-extra-keypoints`) (2026-08-27).** Raise the DoG budget only on listed stems. Implementation appends spatially novel keypoints from a denser extract onto an intact primary `--sift-max-keypoints` prefix (so non-stem images and the stem primary set stay byte-identical to champion). Default off. **Courtyard A/B** on stem-E champion stack:
  - Budget-raise (pre-append; displaces prefix): `0297`+2048 → **~320 cm**; `0296,0297`+2048 → **~433 cm** — **honest negatives** (also thinned stem-E to 4 pairs / contaminated `10-11`).
  - Append-only: `0297`+2048 (`5437` kp, stem-E still **5** pairs, `10-11` E=153) → **~381 cm**; both stems → **~359 cm** — **honest negatives**. Extra fine/coarse peaks on the free hub densify matches but do not straighten GT shape; unlock is still true prior↔hub bridges, not more hub-local texture.
  - Rematch without cross-check (`--rematch-stems DSC_0297 --rematch-no-cross-check`): @0.85 improved **0** pairs → **~304 cm** (no-op / champion variance); @0.9 improved **1** → **~452 cm** — **honest negative**. Diagnose-only raw densification without CC does not convert into better verified E on the hub.
  - Stem-local guided rematch (`--rematch-guided`, main pass unguided): `DSC_0297` @0.85 improved **1** pair → **~282 cm** (**near-neutral / slight** vs today’s champion ~303; historical best ~258). Per-cam `0297` **746→584 cm** — directional hub help, still metres-scale. @0.9 improved **3** → **~403 cm**; stems `0297,0298,0299` @0.85 improved **9** → **~425 cm** — **honest negatives** (over-densify contaminates). Keep as optional probe, not default.

- **Re-PnP free cameras from prior structure (`--repnp-free-from-priors`) (2026-08-27).** After hybrid BA, optionally re-estimate free (non-prior) poses via PnP on tracks also observed by an anchor camera. Anchors = hard pose priors ∪ soft-anchors from `--prefer-essential-stems` (so a dropped hub like `DSC_0296` can still support neighbours). `--repnp-free-min-corrs N` (0 = mapper `min_pnp_inliers`). Default off. **Courtyard A/B** on stem-E champion stack (drop-`0296` + prefer-E `0296` + repair + metric):
  - Soft-anchor + `min_corrs=6`: **38/38**, Sim(3) **~272 cm** — only `image 28` replaced (accepts worse prior-anchored reproj); hub `0297` still fails PnP on ~8 corrs; most free cams have **0** prior-anchored tracks. **Near-neutral / no unlock** vs champion variance (~258–303 cm).
  - Soft-anchor + `min_corrs=8`: **38/38**, Sim(3) **~394 cm** — **honest negative** (same lone replace of `image 28` poisons the gauge).
  - Diagnosis unchanged: bent free hub is structurally isolated from prior-anchored triangulations; positioning polish cannot invent the missing bridges.

- **Pair-local essential edges by stem / free endpoints (2026-08-27).** Extends edge-only E preference beyond `--prefer-essential-inliers`:
  - `--prefer-essential-free-endpoints` → `prefer_essential_edge_matches_free_endpoints` (E only when ≥1 endpoint lacks a pose prior).
  - `--prefer-essential-stems STEM,…` → `prefer_essential_edge_image_indices` (E only on edges incident to those images).
  - `--prefer-essential-stem-clique` / `--prefer-essential-pairs I-J,…` / `--require-essential-selected-edges` / `--require-essential-stems` / `--require-essential-min-e-inliers` / `--essential-edge-weight-boost` / `--rematch-stems`+`--rematch-ratio` for tighter hub surgery.
  - Precedence: all → **union of** explicit pairs **and** stems → free-endpoints (pairs no longer replace stems).
  **Courtyard A/B** on drop-`DSC_0296` + repair + metric stack:
  - Free-endpoints (E on **37** edges): **~314 cm** — **honest negative**.
  - Stems `DSC_0296` only (E on **5** edges: `8-10,9-10,10-11,10-13,10-26`): **38/38**, Sim(3) **~258 cm** (repeat) — **current hybrid champion**. `DSC_0297` still ~720 cm.
  - Stem `0296` ∪ pairs `11-12` / `11-12,11-13` / `+12-13`: **~279–298 cm** — **honest negatives**; additive 0297-neighbour E does not unlock the free hub.
  - `--require-essential-stems DSC_0297` (drop F-only 0297 edges; prefer-E still on 0296): **~370 cm** / with global E≥50 store **~285 cm** (0296 per-cam **~164 cm** but overall regresses) / require-min-E 50–100: **~291–418 cm** — **honest negatives**. Isolating 0297 helps the hub locally sometimes but not global GT shape.
  - `--rematch-stems DSC_0297` @0.9 (3 pairs densified; `10-11` E 138→218): **~399 cm** — **honest negative** (looser matches contaminate E). @0.85: **~259 cm** — **near-neutral** vs champion ~258.
  - Clique / curated pairs / drop weak `10-26` / weight-boost / max-ori=2 / guided / DSP: **~279–442 cm** — **honest negatives**.

- **SIFT `max_orientations` + orientation-order stability (2026-08-27).** `SiftConfig::max_orientations` / `--sift-max-orientations N` (COLMAP default 2; `0` = unlimited legacy). Cap keeps strongest peaks then restores ascending-bin order. **Bug:** an intermediate sort-all-peaks path reshuffled multi-orientation keypoints at the 4096 truncate boundary and collapsed the stem-E champion to **~414 cm**. Restoring bin order when uncapped recovers **~258 cm**.

- **SIFT COLMAP-style keypoint budget (`--sift-prefer-larger-scale`, `--sift-full-pyramid`) (2026-08-27).** `prefer_larger_scale` prunes by σ↓ (covdet-like); `full_pyramid` disables the fine-octave early break before truncation. Unit test `prefer_larger_scale_keeps_coarser_keypoints_when_capped`. **Courtyard** on champion stack:
  - `--sift-prefer-larger-scale`: **37/38**, Sim(3) **~338 cm** — **honest negative** (completeness + RMSE; loses strong `10-11` E).
  - `--sift-full-pyramid`: **38/38**, denser (211 pairs / 28.7k inliers / reproj **14 px**) but Sim(3) **~428 cm** — **honest negative**; also loses `10-11`. Fine-octave contrast set that feeds hub E is load-bearing.

- **GLOMAP-style joint track positioning (`--joint-global-positioning`) (2026-08-27).** After pairwise bearing `average_positions`, optionally refine camera centres jointly with track midpoints via Huber IRLS on ray residuals `||û × (X−c)||` (`refine_centers_joint_tracks`); pose priors stay pinned. Demo flag default off. **Courtyard A/B** (SIFT@4096 / multi-hyp / `--refine-global-translations` / `--chirality-harden` / 8 seeds):
  - `--mapper global`: **38/38**, Sim(3) **~399 cm** (plain global ~437 cm) — small global-only win, still metres-scale.
  - `--mapper hybrid`: **38/38**, Sim(3) **~469 cm** (hybrid champion **~352 cm**) — **honest negative**: joint rays fight the incremental metric frame / scale gauge (hybrid mean reproj ~22 px; seed basin still 38/38). Positioning-only polish does not unlock courtyard sub-cm; edge/matching quality remains the binding constraint.

- **Prefer essential inliers (`--prefer-essential-inliers`) (2026-08-27).** `TwoViewGeometryReport` exposes `essential_inliers` + E/F/H counts; `PairwiseMatches::essential_matches` stores the E subset alongside the winning F/H set. Flag drives `GlobalReconstructionTuning::prefer_essential_edge_matches` so global/hybrid *edges* re-estimate relative pose from E inliers while tracks/incremental keep the denser winning set.
  - Replace-all matches with E: **38/38**, Sim(3) **~425 cm**, priors **20/38** — **honest negative** vs champion ~356 cm.
  - Edge-only (142/210 pairs used E; priors **22/38**; verified inliers stay **27 694**): **38/38**, Sim(3) **~352 cm** — ties champion within variance; does **not** move GT shape off metres-scale (`DSC_0296` hub remains).

- **Prior–prior edge repair (`--repair-prior-edges`) (2026-08-27).** For hybrid edges whose both endpoints have incremental pose priors, rewrite `R_ij` / `direction_ij` from the prior metric frame and 2×-boost weight (`repair_edges_from_pose_priors`). Unit test `prior_edge_repair_unflips_prior_pair_direction`. **Courtyard** hybrid multi-hyp: rewrote **80** edges (flipped **5**); **38/38**, Sim(3) **~352 cm** (prior champion ~356 cm) — pairwise median rel-err 44%→**38%**, but **DSC_0296** still ~803 cm. **Near-neutral alone**.

- **Metric scale from prior–prior baseline (`--metric-prior-scale`) (2026-08-27).** `average_positions_with_priors` sets the position-averaging displacement row from the highest-weight prior–prior edge's true metric length (instead of a unit seed edge). Demo flag default off.
  - Alone: **38/38**, Sim(3) **~411 cm** — **honest negative** vs champion ~356 cm (wrong gauge without prior-consistent bearings).
  - With `--repair-prior-edges` (single highest-weight prior–prior row, len≈1.41): **38/38**, Sim(3) **~311 cm** (**≈−45 cm** vs ~356 cm) — prior hybrid best; still metres-scale. `DSC_0296` 825→**685 cm**.
  - All prior–prior metric rows (80 edges, median len≈7.52) + repair: **38/38**, Sim(3) **~353 cm** — **honest negative** vs single-row ~311 cm; conflicting incremental lengths over-constrain the free cameras. Kept single-row.

- **Surgical hybrid prior drop (`--hybrid-drop-prior-stems`) (2026-08-27).** Clear incremental pose priors whose image stem matches a comma-separated list (default empty). Unlike `--hybrid-filter-priors` (mass drop by track quality → ~428 cm), this isolates bent hubs. **Courtyard** repair+metric champion stack:
  - Drop `DSC_0296` only (21/38 priors): **38/38**, Sim(3) **~270 cm** (**≈−41 cm** vs ~311 cm) — new hybrid best; free placement of the worst pinned hub helps GT shape without discarding the rest of the incremental frame. Per-cam still metres-scale (`DSC_0296`/`DSC_0297` ~667 cm).
  - Drop `DSC_0296,DSC_0322,DSC_0297`: **38/38**, Sim(3) **~279 cm** — **near-neutral / slight regression** vs 0296-only (~270); `0322`/`0297` were not incremental priors (still 21 pinned), so this is mostly run variance.
  - Drop `DSC_0296,DSC_0323,DSC_0319,DSC_0317` (18/38 priors): **38/38**, Sim(3) **~336 cm** — **honest negative** vs 0296-only; over-thinning the prior frame loses the metric gauge that made ~270 work.

- **GT-free inconsistent-prior drop (`--hybrid-drop-inconsistent-priors`) (2026-08-27).** Tries to automate the `DSC_0296` unpin win without GT stems.
  - Edge-bearing disagreement (`filter_pose_priors_by_edge_disagreement`): dropped **DSC_0316** (flip frac ~0.65) not 0296 → **38/38**, Sim(3) **~395 cm** — **honest negative** vs manual 0296-only ~270 / repair+metric ~311. Bent hub is self-consistent with wrong essentials, so translation antipodes hit the wrong camera.
  - Leave-one-out free-centre Sim(3) residual vs rotation-pinned probe (`filter_pose_priors_by_free_centre_residual`): dropped **DSC_0315** not 0296 → **38/38**, Sim(3) **~360 cm** — **honest negative**. Manual `--hybrid-drop-prior-stems DSC_0296` (~270 cm) remains the champion; GT-free prior ranking does not yet recover that stem.

- **Post-PnP two-view registration gate (`--verify-registration-two-view`) (2026-08-27).** After successful PnP, require the absolute pose to agree (same translation hemisphere) with independent essentials vs already-registered neighbours (`pose_agrees_with_two_view_neighbors`). Targets chirality-flipped incremental hubs that still have low local reproj. **Diagnosis:** incremental `DSC_0296` alone is already **~864 cm** after Sim(3) on the 22-prior set (local mean reproj ~0.3 px) — wrong absolute pose, locally consistent. **Courtyard A/B** (repair+metric stack, no stem drop): **0 rejects**, priors still **22/38**, **38/38**, Sim(3) **~352 cm** — **honest negative** / no-op vs ~311 cm; the bent hub agrees with its two-view neighbours (self-consistent wrong basin), so hemisphere checks cannot unmask it. Unlock remains matching/detection quality that breaks the false consensus, not more pose-consistency gates on the same essentials.

- **COLMAP `multiple_models` (`--multiple-models`) (2026-08-27).** Wires `TwoViewGeometryOptions::multiple_models` through the full verifier. When ≥2 non-watermark sub-models peel out, keep the strongest **Calibrated** (else largest) sub-model's inliers/pose and label `Multiple` — do **not** concatenate incompatible inlier sets (that poisoned later essential RANSAC). Unit tests for single-model path unchanged. **Courtyard A/B** on champion stack (`--repair-prior-edges --metric-prior-scale --hybrid-drop-prior-stems DSC_0296`): MULTIPLE=**47** / CALIBRATED=39 / UNCALIBRATED=173; priors **20/38**; **38/38**; Sim(3) **~425 cm** — **honest negative** vs champion ~270 cm (stricter multi-model peel thins good bridges and does not straighten the free hub).

- **Stricter Lowe ratio on champion stack (`--match-ratio 0.7`) (2026-08-27).** Same drop-0296 + repair + metric stack with tighter NN ratio. Verified **115/703** (vs ~210); priors **11/38**; registration **23/38**; Sim(3) **~208 cm** over the registered 23 only — **honest negative** on completeness (fails the 38/38 gate); thinner matches do not unlock a full courtyard model.

- **CorrespondenceGraph tracks on champion (`--track-source graph`) (2026-08-27).** Same drop-0296 + repair + metric stack with M2 `CorrespondenceGraph` track builder. Priors **21/38**; **38/38**; Sim(3) **~270 cm** — **near-neutral** vs union-find champion ~270 cm (within run variance); track source alone does not move GT shape.

- **Essential-inlier pushes on champion stack (2026-08-27).** Hub diagnosis: `DSC_0296`–neighbour pairs at ratio 0.8 all classify **Uncalibrated** (F-dominant; E/F ratio fails COLMAP's 0.95 gate).
  - `--prefer-essential-inliers` (edges only): **38/38**, Sim(3) **~305 cm** — **honest negative** vs ~270 cm.
  - `--force-essential-matches` (tracks+edges use E when available): priors **19/38**; **38/38**; Sim(3) **~388 cm** — **honest negative**; forcing E thins support without straightening the free hub.

- **Hub E-vs-GT diagnosis + selective force-E (`--force-essential-min-ef-ratio`) (2026-08-27).** Diagnose now prints E/F/H counts and E-recovered translation direction. On courtyard @ratio 0.8:
  - `DSC_0296`–`0297`: E=138 F=190 (ratio 0.73), **E·GT = 0.997** (near-perfect).
  - `DSC_0296`–`0295`: E=110 F=116 (ratio 0.95), **E·GT = 0.968**.
  - `DSC_0296`–`0300`: E=12 F=23, E·GT ≈ 0 (weak-E / wrong).
  Gate: use E matches only when E/F ≥ threshold (default 0.7). **Courtyard A/B**:
  - Selective E @0.7 + drop `0296`: **38/38**, Sim(3) **~379 cm** — **honest negative** vs ~270.
  - Selective E @0.7, no stem drop: **38/38**, Sim(3) **~384 cm**; `DSC_0296` err **~376 cm** (was ~685–864 when F-pinned) — E helps the hub locally but overall shape regresses; worst cameras move to `DSC_0323`/`0322`.
  - Selective E @0.7 + drop `0323,0322,0316`: only **1** prior dropped (`0323`; others not incremental) → **~399 cm** — **honest negative**.
  - Selective E @0.7 + drop `0296,0323`: **~402 cm** — **honest negative**.
  - RootSIFT (`--sift-l1-root`) + drop `0296`: **~344 cm** — **honest negative**.
  - SIFT@8192 + drop `0296`: **~375 cm** — **honest negative** (0296 often not in the denser incremental prior set).
  Unlock: E is the right model for the close hub, but naïvely promoting it globally fights other pairs; need pair-local or detector-side changes that raise E without poisoning the rest of the graph.

- **Stronger pair-local E gates (2026-08-27).** Added `--force-essential-min-e-inliers`, `--force-essential-uncalibrated-only`, `--min-e-f-inlier-ratio`, `--calibrated-prefer-essential` (Calibrated branch keeps E even when F has more inliers).
  - Uncalibrated-only force-E (E/F≥0.7, E≥100): swapped **18** pairs; **~414 cm** (no drop) / **~422 cm** (+drop `0296`) — **honest negative**.
  - Strong-E for **edges only** (`--prefer-essential-inliers` + E≥100 + E/F≥0.7) + drop `0296`: used E on **58** edges; **38/38**, Sim(3) **~333 cm** — still **honest negative** vs ~270 (closer than track swaps).
  - `--min-e-f-inlier-ratio 0.7 --calibrated-prefer-essential` + drop `0296`: CALIBRATED **50→119**; **37/38**, Sim(3) **~448 cm** — **honest negative** (completeness regression). Raising Calibrated admissions alone does not unlock sub-cm.

- **Calibrated-only view-graph edges (`--calibrated-view-edges-only`) (2026-08-27).** `PairwiseMatches::two_view_config` records COLMAP `ConfigurationType` from full verification; global SfM drops edges that are not `Calibrated`/`Multiple` (orphan rescue respects the same gate). Demo flag default off. Unit test `calibrated_view_edges_only_skips_uncalibrated_pairs`. **Courtyard A/B** (hybrid multi-hyp stack): pre-fix labels were inverted (see Changed §ConfigurationType swap) — first run **37/38 @ ~421 cm** kept the wrong 209 F-dominant edges. **After fix** (CALIBRATED=50 / UNCALIBRATED=209): **31/38**, Sim(3) **~432 cm** — **honest negative**: the 50 E/F-agreement edges alone do not span the courtyard graph; champion still **38/38 @ ~356 cm** with all 210 verified pairs.

- **SIFT affine + multi-anisotropy on hybrid (`--sift-affine --sift-multi-anisotropy`) (2026-08-27).** Re-ran hybrid multi-hyp stack with covdet-ordered affine descriptors + budgeted multi-anisotropy proposals. Verified **160/703** (vs plain **210/703**); incremental priors **13/38** (was **22/38**); **24/38** registered; Sim(3) **~189 cm** over the registered 24 only — **honest negative**: affine detection thins the view graph on this façade scene (same diagnosis as prior global-only `--sift-affine` **22/38** run); completeness and GT shape both regress vs SIFT hybrid champion **38/38 @ ~352 cm**.

- **SuperPoint+NN courtyard A/B vs SIFT hybrid champion (2026-08-27).** Exported SuperPoint@4096 (`scripts/export_superpoint_lightglue.py --mono-dir`, CPU) for ETH3D courtyard; ran `--feature-extractor files --matcher nn` through the hybrid multi-hyp stack. **38/38**, verified **195/703** (48k inliers), incremental priors **24/38**, mean reproj **4.37 px** (SIFT hybrid ~22–27 px) — denser tracks — but Sim(3) centre RMSE **~394 cm** vs SIFT champion **~352 cm**. **Honest negative**: SuperPoint+NN alone does not beat SIFT on courtyard GT shape.

- **SuperPoint+LightGlue ONNX courtyard A/B (2026-08-27).** Exported `models/lightglue_courtyard.onnx` (1600×1066). Demo gains `--onnx-backend auto|cuda|cpu` (use `cpu` when no NVIDIA driver — `auto` hangs in CUDA EP registration) and `--lightglue-max-keypoints N` (score-sorted prefix; CPU 4096×4096 is multi-minute/pair). LightGlue pair verification runs **sequentially** (rayon+ORT threadpool deadlocks). Requires `ORT_DYLIB_PATH` → matching ORT (`onnxruntime-linux-x64-1.23.2` verified). **Courtyard** hybrid multi-hyp:
  - `@512`: verified **313/703** (22k inliers); priors **9/38**; **38/38**; Sim(3) **~458 cm**.
  - `@1024`: verified **463/703** (70k inliers); priors **12/38**; **37/38** (missing **DSC_0301**); Sim(3) **~458 cm**.
  Both worse than SIFT hybrid champion **~352 cm** / SP+NN **~394 cm** — **honest negative** (more pairs ≠ better GT shape; thin incremental seed).

- **Hybrid rotation-only priors (`--hybrid-rotation-priors-only`) (2026-08-27).** Pins incremental orientations during global rotation averaging but keeps globally bearing-averaged centres for prior cameras (not incremental centres). Gauge: median inter-prior distance scale (≥3 prior pairs) + seed-prior translation anchor. Default off (= full pose priors). Unit test `rotation_only_priors_keep_global_centres_not_incremental` on a 6-camera ring. **Courtyard A/B** (same hybrid multi-hyp stack): **38/38**, Sim(3) **~471 cm** (full-prior champion **~352 cm**) — **honest negative**: incremental inter-prior distances live in a bent layout; median scale gauge collapses (~0) and freeing bad centres (e.g. **DSC_0296**) without better edges does not beat pinning the partial incremental metric frame.

- **Translation-refine flip count on `GlobalSfmPoses` + multi-seed debug (2026-08-27).** `GlobalSfmPoses::translation_refine_flips` records how many edge directions [`refine_edge_directions_under_rotations`] flipped; multi-seed debug logs include `trans_flips`. **Courtyard A/B** (same SIFT@4096 / multi-hyp / `--refine-global-translations` stack): global-only seed ranking that *preferred* fewer flips chose seed 3 (`trans_flips=2`) over the prior seed 26 basin (`flip_alts=true`) → **38/38**, Sim(3) **~474 cm** (prior global ~437 cm) — **honest negative**: low flip count tracks self-consistent chirality repair, not GT shape. Hybrid unchanged in basin choice (seed 0 / `flip_alts=false` / 2 flips) at Sim(3) **~357 cm** (prior ~352 cm, within run variance). Flip count is diagnostic-only; not used in seed ranking.

- **Hybrid incremental→global mapper (`--mapper hybrid`) + pose-prior gauge (2026-08-27).** Runs incremental SfM first, then `reconstruct_global_sfm_with_priors` pins those cameras' absolute orientations/centres while global averaging places the remaining images. `solve_global_sfm` accepts optional pose priors; MST rotation seeding grows from the prior set; Sim(3) centre alignment onto priors after position averaging. Demo: `--mapper hybrid`. **Courtyard** (same global multi-hyp stack as champion): incremental **22/38** priors → hybrid **38/38**; Sim(3) centre RMSE **~352 cm** (global-only ~437 cm) — partial GT win, still metres-scale / not sub-cm. **Diagnosis:** incremental priors are not uniformly good — **DSC_0296** is pinned at **864 cm** GT error (1.92 px / 13 obs locally); 16 hybrid-only images still mean **~426 cm**. Honest negatives: `--hybrid-filter-priors` (13/22 kept @0.45 px) **~428 cm**; pin+Sim(3) centre averaging **~360 cm**; `--guided-matching` on hybrid **~416 cm**; global+guided **~467 cm**; RMS-normalized Sim(3) fallback **~472 cm**.

- **Opt-in hybrid prior quality gate (`--hybrid-filter-priors`, `--hybrid-prior-min-obs`, `--hybrid-prior-max-reproj`) (2026-08-27).** `filter_pose_priors_by_track_quality` drops incremental priors with too few track observations or high local mean reprojection before global placement. Default off. Courtyard @min_obs=50,max_reproj=0.45: keeps **13/22** priors (drops **DSC_0296** among others) → Sim(3) **~428 cm** (worse than unfiltered ~352 cm). Looser gate @max_reproj=1.0 keeps **17/22** → **~374 cm** — still regresses; unfiltered hybrid priors remain the courtyard champion among hybrid variants.

- **Global mapper CLI: `--min-edge-parallax-deg` (2026-08-27).** Wires `GlobalReconstructionTuning::min_edge_parallax_deg` (default 2.0°). **Courtyard honest negative** at `--min-edge-parallax-deg 5`: **38/38** but Sim(3) **~470 cm** (default ~437 cm); view graph thins to 78 edges / median bearing residual explodes — stricter parallax alone does not straighten the bent basin here.

- **Guided matching on global mapper (`--guided-matching --mapper global`) (2026-08-27).** Same COLMAP-style epipolar rematch as incremental (210 verified pairs, 27k inliers vs plain 193/703). **Courtyard honest negative**: **38/38** but Sim(3) **~467 cm** (plain global ~437 cm) — denser correspondences alone do not escape the bent global basin on this façade set.

- **COLMAP L1-root / RootSIFT descriptor normalization (`--sift-l1-root`) (2026-08-27).** `SiftNormalization::{L2,L1Root}` — L1-normalize then √ per bin (Arandjelović & Zisserman / COLMAP `descriptor_normalization=l1_root`), then unit L2. Default remains L2 (byte-identical). Demo: `--sift-l1-root`. Unit test `l1_root_normalization_is_unit_and_differs_from_l2`. **Courtyard** (`--sift-l1-root` + global multi-hyp stack): verified 193/703; **38/38**; Sim(3) **~439 cm** (prior plain-L2 ~396 cm) — **honest negative**: COLMAP's default norm alone does not straighten the façade basin here.

- **DSP-SIFT domain-size pooling (`--sift-dsp`) (2026-08-27).** `SiftConfig::{domain_size_pooling,dsp_min_scale,dsp_max_scale,dsp_num_scales}` averages unnormalized SIFT histograms over a geometric scale range then L2-normalizes once (Dong & Soatto / COLMAP). Default range follows the paper (`1/6…4/3`); sampling stride keeps cost ~O(num_scales). Demo: `--sift-dsp`, `--sift-dsp-num-scales N` (default 10). Parallel image extraction with progress logs. Unit test `dsp_sift_keeps_dimension_and_differs_from_plain`. **Courtyard** (`--sift-dsp --sift-dsp-num-scales 5` + global multi-hyp stack): **38/38**; Sim(3) **~438 cm** (prior plain ~396 cm); mean reproj 122 px — **honest negative**: DSP alone does not straighten the bent basin on this façade set (and slightly worsens GT).

- **Denser multi-seed ranking + opt-in chirality-margin edge weights (2026-08-27).** Multi-seed selection now keeps only candidates with ≥50% of the densest pre-BA observation count (same registration), then ranks by `reproj*(1+bearing)` — rejects collapsed self-consistent basins that win on tiny track sets. `RelativePose::chirality_margin` + `--weight-by-chirality-margin` scales edge weights by `(0.1+margin)`. **Courtyard** (orphan rescue + multi-hyp + dense ranking): **38/38**, Sim(3) **~396 cm**. Honest negative: margin weighting alone preferred a 13 px / 574-track basin at Sim3 **~438 cm** before the density floor.

- **Orphan-edge rescue for degree-0 images + `--min-edge-inliers` + gate diagnostics (2026-08-27).** After strict relative-pose gates, images with zero view-graph edges retry their verified pairs at `max(8, min_edge_inliers/2)` (parallax gate skipped on the rescue pass). Debug logs report per-image fail estimate/inliers/parallax counts. Demo: `--min-edge-inliers N` (default 15). **Courtyard** (`--mapper global --chirality-harden --rotation-seed-trials 8 --refine-global-translations --multi-hypothesis-edges`): rescue adds 1 edge for **DSC_0308** (was fail-inliers-only); registration **38/38** (first complete courtyard registration); Sim(3) centre RMSE **~395 cm** (prior 37/38 ~404 cm) — completeness cleared; GT shape still metres-scale / pairwise dist rel-err ~78%.

- **Multi-hypothesis view-graph edges for chirality-ambiguous essentials (`--multi-hypothesis-edges`) (2026-08-27).** Essential decomposition now returns an optional runner-up `(R, t)` on `RelativePoseRecovery` / `RelativePose`. `GlobalSfmEdge` carries `rotation_alt` / `direction_alt`; rotation averaging, IRLS trim, triplet sanitation (min over hyp combinations), and position bearings pick primary vs alternate by agreement with the emerging global solution. `CheiralityOptions::hardened_keep_ambiguous()` keeps angle/depth gates without the ambiguity *rejection* ratio. Demo: `--multi-hypothesis-edges`. Unit tests: `multi_hypothesis_rotation_averaging_picks_alternate`, `hardened_keep_ambiguous_exposes_alternate_on_clean_pair`. **Courtyard** (`--mapper global --chirality-harden --rotation-seed-trials 8 --refine-global-translations --multi-hypothesis-edges`): 163/182 edges carry alternates (13 pairs still fail angle/depth); seed 26 wins at **78 px** pre-BA reproj (was ~97); **37/38**; Sim(3) centre RMSE **~404 cm** (prior refine-only ~457–466 cm) — honest partial: shape basin softens but still metres-scale / missing one image; not COLMAP sub-cm.

- **Multi-hypothesis hyp-flip basin search + residual PnP after global BA (2026-08-27).** With `--multi-hypothesis-edges`, each rotation seed also tries an all-alternates-as-primary graph (MST locks onto stored primaries). Courtyard: flip_alts rarely wins (seed 26 primary still preferred); best pre-BA ~43–75 px; **37/38**; Sim(3) **~404–407 cm**. Residual PnP (`post_refinement_registration_pass` after global BA) does **not** pick up missing **DSC_0308** — no triangulated 2D–3D track support into that image (matching/connectivity gap, not averaging). Honest negative on seed-time alternate *promotion*: reinforces the wrong MST basin (Sim3 ~446 cm); reverted.

- **Global-R translation refine after rotation averaging (`--refine-global-translations`) (2026-08-27).** Edges retain an inlier pixel sample; after consensus rotations are fixed, each edge's `direction_ij` is re-scored under `R_j R_i⁻¹` (±pairwise dir + fixed-R epipolar nullspace) and chirality-flipped bearings are corrected. Unit test `global_r_translation_refine_unflips_wrong_chirality` pins the synthetic flip case. Multi-seed selection now ranks by **pre-BA track reprojection** (not bearing residual), which prefers image-consistent basins over self-consistent wrong ones. **Courtyard** (`--mapper global --chirality-harden --rotation-seed-trials 8 --refine-global-translations`): flips ~4–7 edges/trial; seed 26 wins at ~97 px pre-BA reproj; **37/38**; Sim(3) RMSE **~457–466 cm** (prior ~470 cm) — small movement, GT shape still metres-scale.

- **Hessian-Laplace detector + multi-anisotropy proposals (2026-08-27).** `SiftDetector::{Dog, HessianLaplace}` and `SiftConfig::multi_anisotropy` (default off). Hessian-Laplace finds spatial peaks of `|det H|` on the Gaussian pyramid with Laplacian scale selection (Mikolajczyk / VLFeat). Multi-anisotropy (requires `affine`) detects on a few det-1 x-stretches, maps survivors back under strict NMS + budget. Demo: `--sift-detector dog|hessian-laplace`, `--sift-multi-anisotropy`. Unit tests cover blob detection and budgeted extras. Stretch harness with hess+affine+multi still below the ≥4 bar (ignored). **Courtyard** (`--sift-detector hessian-laplace --mapper global --chirality-harden --rotation-seed-trials 8`): **37/38**, Sim(3) RMSE **~470 cm** — parity with DoG; detector swap alone does not straighten the bent shape.

### Changed

- **`TwoViewGeometryVerifier` ConfigurationType labels swapped (2026-08-27).** In `classify_single`, the E/F-agreement branch (`tvg.cc:877-898`) was labelling pairs `Uncalibrated` and the F-only branch (`:899-914`) `Calibrated` — inverted vs COLMAP. Fixed; unit test `general_scene_classifies_calibrated_or_uncalibrated` now expects `Calibrated` on a general 3D scene. **Courtyard** verification counts flip to **CALIBRATED=50 / UNCALIBRATED=209** (was 209/50); hybrid champion unchanged **38/38**, Sim(3) **~356 cm** (prior mislabelled run ~352 cm, within variance) — pair *admission* is unchanged (both configs stay in the keep-list), but config-aware filters (e.g. `--calibrated-view-edges-only`) now target the intended edges.

- **SIFT affine path: VLFeat covdet ordering + location refine (2026-08-27).** When `SiftConfig::affine` / `--sift-affine` is on: (1) estimate Baumberg shape first, (2) refine the detection locus inside the affine-normalized patch via peak squared-gradient search, (3) assign orientation on canonical-axis gradients, (4) describe through `A`. Shape adaptation gains VLFeat-style min-singular-value hold, anisotropy cap (6×), and convergence gate. Cross-stretch harness improves plain=1 → **affine=3** mutual matches (still ignored; ≥4 bar not met). **Courtyard honest negative** (`--mapper global --sift-affine --chirality-harden --rotation-seed-trials 8`): verified pairs drop (158/703), registration **22/38** (was 37/38 without affine), Sim(3) RMSE still metres-scale — descriptor-side+ordering affine alone thins the view graph on this façade scene; fuller multi-anisotropy detection remains open.

### Added

- **Chirality-hardened relative-pose recovery + multi-seed rotation averaging + MST translation-sign repair (2026-08-27) — courtyard GT parity; edge-quality levers wired.**
  - `CheiralityOptions` / `recover_relative_pose_with_options` / `RelativePoseRecovery` in `visloc_vision::two_view`: min triangulation angle, ambiguity rejection (`second/best` ratio), and minimum positive-depth fraction. `CheiralityOptions::hardened()` = 1° / 0.85 / 0.5. Default options keep the legacy positive-depth-only selector byte-identical.
  - `GlobalReconstructionTuning::{chirality_harden_edges, rotation_seed_trials}` (defaults `false` / `1`) plus demo flags `--chirality-harden` / `--rotation-seed-trials N`. Multi-seed tries the component's highest-degree nodes and keeps the solve with most cameras / lowest mean bearing residual.
  - `average_positions` now runs an MST-guided translation-sign repair before graduated Huber: off-tree bearings anti-aligned with the tree placement are flipped.
  - Demo also exposes `--sift-affine` (see Changed entry above for the 2026-08-27 covdet-ordering upgrade).
  - **COLMAP CLI shim** `examples/colmap.rs` (required-features `image-io`): drop-in `feature_extractor` / `exhaustive_matcher` / `mapper` / `model_converter` driving the same pipeline library; state under `<db>.d/`, sparse model via `write_colmap_reconstruction_for_3dgs`. Smoke-tested on courtyard (1024 kp → 12/38 registered).
  - **Measured on ETH3D courtyard (SIFT@4096, exhaustive, `--mapper global --chirality-harden --rotation-seed-trials 8`)**: chirality-harden rejected 20 pairs (175 edges kept); MST sign-repair flips ~8–9 bearings/trial; multi-seed picks vary by residual; registration **37/38** unchanged; Sim(3) centre RMSE vs GT **~469–473 cm** (prior ~471 cm) — **honest negative: GT shape parity does not move**. Self-consistent wrong basins survive harden + multi-seed + sign repair; the remaining unlock is detector-side affine-covariant sampling (VLFeat covdet; the ignored cross-stretch SIFT test) and/or learned matching.

- **GLOMAP-style parallax gate for view-graph edges + COLMAP-style guided matching (`--guided-matching`) (2026-08-26).**
  - `GlobalReconstructionTuning::min_edge_parallax_deg` (default 2.0°): each candidate edge's median two-view triangulation angle over a sample of its inliers is measured under the pair pose (`triangulate_two_view_left_frame`); edges below the threshold are dropped before averaging. Tiny-baseline pairs produce well-fit essential matrices whose translation direction is pure noise — courtyard drops 8 such edges and median bearing residual improves 21.3° → **8.33°** (mean reprojection 59 px → 35.8 px in the global mapper). Set to 0 to disable.
  - Guided matching ports COLMAP's post-verification rematch: descriptors missed by NN+ratio get one more chance under the verified epipolar geometry, gated by a conservative Lowe ratio (< 0.8) AND Sampson distance (< 2 px), conflicts resolved greedily by distance, then the pair is re-classified so config/inliers describe the final set. Off by default; on courtyard incremental it holds registration at 21/38 while tightening geometry: mean reproj 0.477 px → **0.403 px**, tracks 4958 → 5141. Looser gates (ratio 0.9 / 4 px) measurably *degrade* the solve (19/38, 0.763 px) — recorded as the honest boundary.

### Added

- **COLMAP `TransitivePairGenerator` port (`--pair-source transitive`) — pairing module 60% → 70% (2026-08-26).**
  - Faithful port of `src/colmap/pairing.cc`: after a vocab-tree base pass is verified, images that share a common matched partner but have no direct pair yet are proposed (`expand_transitive`), for up to two expansion rounds (`TRANSITIVE_ROUNDS`), never re-proposing a pair.
  - `VerificationStats::merge` accumulates classifier counts across passes.
  - Measured on ETH3D courtyard: base pass 703 candidates → 210 verified pairs after expansion (+cross-component bridge proposals); incremental registration unchanged at **21/38** — decisive negative finding: pair *supply* is not the binding constraint for the unregistered images; their cross-component structure does not chain into tracks under NN+ratio matching (the M5/M6 LightGlue motivation, model not present locally so the joint-matcher A/B remains open).

### Changed

- **Global-SfM position stage: trust-hierarchy graduated Huber IRLS (spanning-tree tiers) — internal consistency ×3, GT parity; basin-flip diagnosis recorded (2026-08-26).** `average_positions` now builds a maximum-weight spanning tree (Kruskal) over the bearing graph and runs 8 graduated Huber rounds (~2.9°→~22.9°) where tree-tier edges tolerate **double** the angular error of off-tree edges each round, so systematically wrong off-tree bearings are demoted first while the trusted skeleton anchors the topology. Measured on ETH3D courtyard (SIFT, global mapper): median bearing residual **69.5°→21.3°** and mean reprojection **509 px→59 px** (×3 internal-consistency gain) at identical GT outcome (37/38, Sim(3) RMSE 471 cm vs 477 cm). Clean-data tests are bit-identical to plain LS because nothing exceeds even the tight threshold. Decisive negative finding: courtyard's bearing errors are NOT outlier-distributed — they are systematic (chirality-ambiguous essential estimates on repetitive façades survive both trimming and soft weighting), so robust positioning alone cannot straighten the shape. The binding lever remains upstream edge quality: chirality-hardened relative-pose estimation and/or detector-side affine-covariant sampling. Kept because the graduated form is strictly more principled (no information deleted) and is the substrate the eventual edge-quality fix needs.


### Added

- **COLMAP-port continuation slice 4: adaptive PnP RANSAC budget (`--pnp-max-iterations`) — courtyard registration 12/38 → 21/38 (2026-08-26, same day follow-up 3).**
  - **Diagnosis.** `VISLOC_SFM_DEBUG` logs on ETH3D courtyard (SIFT frontend) showed the incremental mapper's binding failure is NOT missing correspondences: one unregistered image had **516 2D-3D correspondences but only 3 PnP RANSAC inliers** at the fixed 128-sample budget. With an inlier ratio that low, 128 samples almost never draw an all-inlier minimal set — the search is under-budgeted, not starved.
  - **Faithful port.** `PnPRansac` gains COLMAP's dynamic iteration termination (`confidence: Option<f64>`): `iterations` becomes a fail-safe cap and the search exits once the best model's inlier ratio implies 99.9% registration confidence (`needed = log(1-conf)/log(1-w^s)`), mirroring COLMAP's `UpdateNumIterations`. The mapper opts in when `IncrementalSfmConfig::pnp_max_iterations > 128` (new knob, default 128 = byte-identical legacy behaviour) and only applies the expanded budget to correspondence sets ≥64 — small clean sets keep the historical trajectory exactly (the synthetic-ring intrinsics co-evolution test pins this).
  - **Measured effect (courtyard, SIFT@4096, exhaustive pairs):** incremental registration **12 → 21 / 38** with mean reprojection 0.477 px; Sim(3) centre RMSE vs ETH3D GT over the registered set improves to **152.9 cm (median 89.7 cm)** from the pre-slice state. Still not COLMAP-equivalent (COLMAP registers 38/38 at laser-GT sub-cm accuracy); the remaining gap is per-image correspondence quality crossing the second component, not search budget. New demo flag: `unordered_sfm_demo --pnp-max-iterations N`.

- **COLMAP-port continuation slice 3: `--mapper global` end-to-end on real data + SIFT affine shape adaptation (2026-08-26, same day follow-up 2).**
  1. **`unordered_sfm_demo --mapper global`** wires `reconstruct_global_sfm` into the unordered demo as an alternative to the incremental grow-from-seed path, sharing feature loading, candidate generation, verification, and the COLMAP export. Three robustness fixes landed in `global_sfm.rs` en route, each caught by real-data diagnostics on ETH3D courtyard (SIFT frontend): (a) seed-component selection now solves the LARGEST connected component of the view graph instead of a fixed image index (the caller's seed sat in a 4-image side component while 37 images formed the main one); (b) rotation seeding replaced Kruskal with Prim-style maximum-weight frontier growth — Kruskal consumed tree edges inside the unreached mass before the frontier arrived, stranding the seed at posed=4; (c) triplet loop-closure sanitisation drops edges whose median triangle rotation-loop error exceeds twice the IRLS trim threshold, plus IRLS trimming inside rotation averaging itself (post-average relative-rotation error: median 0.78°, max 6.8° over 130 kept edges) and bearing-trimming IRLS rounds in position averaging.
  2. **Measured courtyard verdict (honest)**: the global mapper registers **37 / 38** images where the incremental mapper registers **12 / 38** on identical SIFT features — a decisive completeness win — but Sim(3) centre RMSE against ETH3D GT poses is poor (477.6 cm mean / 439.3 cm median over all 37): locally-consistent rotations with globally bent positions, because per-edge monocular bearings from noisy NN+ratio essentials are satisfied exactly by the least squares while their errors accumulate as curvature. Named missing mechanism (future work): robust positioning (L1/graduated-norm or per-bearing covariance weights) plus chirality-hardened edge estimation; both stages remain opt-in via `--mapper global`, default unchanged.
  3. **SIFT affine shape adaptation** (`SiftConfig::affine`, default off): Baumberg/covdet-style structure-tensor iteration (`estimate_affine_shape`) estimates a det-1 shape per keypoint by resampling through the current warp and updating `A ← A·μ^{-1/2}`; the descriptor sampling grid and canonical-axis gradients are warped by `A` accordingly. Unit tests pin shape sanity (finite, det=1, bounded anisotropy ≤100 sv ratio) and full descriptor self-consistency (105/105 mutual self-matches); the cross-anisotropic-stretch matching test is present but `#[ignore]`d with its diagnosis recorded — the isotropic DoG detector does not repeat across affine warps (detections land on non-corresponding structures), so closing it needs detector-side covariant sampling rather than descriptor work.

- **COLMAP-port continuation slice 2: global-SfM end-to-end pipeline, SIFT demo wiring, retrieval-scaling registry evidence, and a first CPU-only ETH3D courtyard run (2026-08-26, same day follow-up).**
  1. **`visloc_slam::global_sfm::reconstruct_global_sfm`** — the global geometry stage is now an end-to-end reconstruction entry point: verified pairwise matches → per-pair essential estimation (`RelativePoseEstimator`) → `solve_global_sfm` (rotation averaging + position averaging) → track triangulation against the averaged cameras → one joint Schur BA. A synthetic full-ring E2E test recovers all 6 cameras with gauge-aligned rotations (<1e-2 rad), scale-invariant centres (<5e-2 m after alignment), dense BA-refined tracks, and <0.5 px mean reprojection; a disconnected-input test pins island images staying unposed. Incremental-mapper internals reused via `pub(crate)` promotions only.
  2. **`unordered_sfm_demo --feature-extractor sift --images-dir <dir>`** — the pure-Rust SIFT frontend now runs in-process from common-format images (requires `--features image-io`; compiles to a runtime error without it), alongside `--sift-max-keypoints` (default 2048) and `--structureless-registration` exposure on this demo. The default `files` path is byte-identical to before.
  3. **Retrieval-scaling registry evidence** — new manifest `benchmarks/registry/runs/retrieval/retrieval-scale-synthetic-20260826.json` (validated) plus `docs/generated/retrieval_scaling.md`: full 250→2000-image sweep on this machine shows the vocab tree 4.2x faster at 2000 images (196.9 s vs 828.3 s wall clock) with near-parity recall@10 (0.957 vs 1.000), entries-visited growing linearly (~4.0x per corpus doubling) while the flat scan is quadratic; crossover at ≥1000 images with the fixed 4096-word vocabulary.
  4. **First real-data run of the classical path (ETH3D courtyard, CPU-only)** — dataset downloaded and extracted locally; SIFT@1600 px + NN+ratio matching registers **12 / 38** (4096 kp, exhaustive pairs, min-matches 20) vs the recorded SuperPoint run's 14 / 38 on the same image cluster, in ~4 minutes end-to-end with zero Python/GPU. Sim(3) centre RMSE against ETH3D GT poses over the registered set is poor (median ~1.8 m, monotone drift = the documented bent-shape failure). Honest negative: detection density alone moved registration 11→12; matching quality (no LightGlue model on this machine) remains the binding constraint for the classical frontend. Recorded in `docs/colmap_port_plan.md`'s 2026-08-26 continuation section.

- **COLMAP-port continuation slice: multi-round structure-less registration, deterministic retrieval-scaling evidence, GLOMAP-style global SfM back-end, and a pure-Rust SIFT frontend (2026-08-26).** Four independent additions advancing `docs/colmap_port_plan.md`'s open items, all unit-tested and default-off or additive:
  1. **Iterative structure-less registration rounds** (`IncrementalSfmConfig::structureless_max_rounds`, default 4): the unordered incremental mapper's experimental structure-less completion pass previously ran exactly one ascending scan, so an island image whose bridge has a HIGHER index was left unregistered forever (the courtyard second-component failure shape). The pass now repeats until a round registers nothing (budget `--structureless-max-rounds` on `sequential_sfm_demo`); one round reproduces the historical behaviour byte-for-byte because rounds only re-run when at least one image registered. A synthetic two-image island fixture (dependent index < bridge index; disjoint strided keypoint bands keep every island-touching union-find component below the track-length floor) proves a single scan recovers only the bridge while multiple rounds chain the dependent through it with sub-0.01 rad rotation error.
  2. **Retrieval-scaling work counters + benchmark** (closes `colmap_port_plan.md` M4's deferred thousands-of-images acceptance): `VocabTree::query_with_work` returns `QueryWorkStats { leaf_distance_computations, entries_visited }`, making retrieval-cost scaling assertable without wall-clock noise — a new unit test pins entries-visited growth to near-linear (<10x) across an 8x corpus growth where flat-VLAD pairwise scanning is analytically 64x. New `examples/retrieval_scale_benchmark.rs` measures both arms on a fixed-vocabulary synthetic places corpus: at 2000 images the tree proposes pairs ~32x faster than the flat scan (32 s vs 1037 s wall clock on this machine) with stable recall@10 (0.71 tree vs 1.00 flat), and its measured work grows linearly (4.0x per corpus doubling).
  3. **GLOMAP-style global SfM geometry stage** (`visloc_slam::global_sfm::solve_global_sfm`): from verified pairwise relative poses, maximum-spanning-tree rotation seeding plus iterative geodesic-consensus sweeps (weighted SO(3) quaternion outer-product power iteration — chordal L2 evaluated lazily, no dense system) produce global orientations; position averaging then solves camera centres by conjugate gradient on perpendicular-bearing least-squares rows with one scale-fixing unit-displacement row on the seed's strongest edge (the homogeneous trivial solution otherwise collapses everything to the origin). Synthetic ring/tetrahedron fixtures verify rotations to <1e-3 rad after gauge alignment and centre geometry up to the monocular scale; small edge noise stays bounded and disconnected images report `None`.
  4. **Pure-Rust SIFT frontend** (`visloc_vision::features::sift::extract_sift`): DoG scale space over doubled-input octaves (σ²-composition incremental blurring), 3×3×3 extrema detection with contrast gate and Lowe edge test, 36-bin orientation assignment with parabolic peak interpolation, and trilinearly binned 4×4×8 descriptors with clamp-and-renormalize — algorithm semantics per Lowe IJCV 2004 with VLFeat as the BSD-clean behavioural reference (no SiftGPU/LSD code or dependency). Deterministic; blob-scene tests verify detection locality, 128-dim descriptor shape, byte-identical repeatability, and FeatureSet wrapper consistency. Not yet wired into any extractor trait path or benchmark; that is follow-up work.

- **PGO loop-closure correction propagation to landmarks + tracker (2026-07-10).** New opt-in `OnlineSlamLoopClosureRefinementConfig::propagate_corrections: bool` (default `false`, byte-identical to today's write-back-only behaviour) fixes the online loop-closure refinement stage's write-back gap: previously a converged pose-graph solve moved only `map.keyframes[*].frame.pose`, leaving landmarks at their pre-solve positions and the tracker's continuation state anchored to the old drift, so the very next `track_frame` immediately re-localized against the stale landmark field and undid the correction. With `propagate_corrections: true`, `maybe_run_loop_closure_refinement` now (1) computes each solved keyframe's world-frame rigid correction `C_k = T_cw_new ∘ T_wc_old` (new camera-to-world composed with the pre-solve world-to-camera) and moves every landmark rigidly with its ANCHOR keyframe — the first (lowest-id) keyframe in `map.keyframes[*].observations` that observed it, rebuilt fresh each solve — landmarks anchored outside this solve's updated keyframes are left untouched; and (2) applies the most-recently-solved keyframe's correction to the tracker's continuation state via a new `Tracker::apply_pose_correction(&SE3)` (and a new `MotionModel::apply_pose_correction` default-no-op trait method, implemented for `ImuPredictiveMotionModel`, `ConstantVelocityMotionModel`, and `AdaptiveImuPoseMotionModel`), so the next frame's PnP prior starts consistent with the corrected map. New `OnlineSlamLoopClosureRefinementStats` fields (`landmarks_moved`, `max_landmark_displacement_meters`, `mean_landmark_displacement_meters`, `tracker_correction_applied`) report per-solve propagation diagnostics, surfaced in `examples/euroc_online_slam_vi_image_demo`'s `summary.txt` alongside the new `--pose-graph-refinement-propagate` CLI flag. Unit tests cover exact landmark displacement matching a solved keyframe's correction, landmarks anchored to keyframes outside the solved graph staying untouched, `propagate_corrections: false` leaving landmarks byte-identical, and a reprojection round-trip sanity check (a landmark's pixel in its anchor keyframe is preserved after both are corrected together) that would fail immediately on a backwards correction. EuRoC benchmark evidence is pending a follow-up run.

- **ORB-SLAM3-style projection-guided tracking (2026-07-10).** New opt-in `TrackingConfig::projection_guided_tracking: Option<ProjectionGuidedTrackingConfig>` (default `None`, bitwise-identical to today's behaviour) replaces the tracker's single-shot appearance-global descriptor search with a three-stage flow when a pose prior is available: (1) `ProjectionCorrespondenceBuilder` (new, `visloc-localization`) projects each candidate landmark into the frame with the prior pose and `Camera::project`, matching its descriptor only against query keypoints within `search_radius_px` (default `15.0`) of the projection, reusing the configured `Matcher`'s ratio-test/cross-check semantics; (2) a widen-retry ladder multiplies the radius by `widen_factor` (default `2.0`) and retries up to `max_widen_retries` times (default `2`) when a rung fails the existing PnP/quality-gate path, falling back to today's unmodified appearance-global matching if every projection attempt fails, so enabling the feature can only add tracking chances, never remove them; (3) optional `local_map_refinement` (default `true`, requires `covisibility_local_map` also configured) re-projects the covisibility local map with the newly ESTIMATED pose at `refinement_search_radius_px` (default `8.0`), re-optimizes over the union of harvested and existing inlier correspondences, and accepts the refined pose only if its inlier count does not decrease. New `FrameLocalizer` trait methods (`localize_frame_with_projection_window_and_descriptor_store`, `refine_pose_with_local_map_and_descriptor_store`) default to the pre-existing appearance-global path, so third-party `FrameLocalizer` implementors are unaffected. New `TrackingStats` counters (`projection_guided_attempt_count`, `projection_guided_widen_retry_count`, `projection_guided_success_count`, `projection_guided_fallback_success_count`, `local_map_refinement_correspondence_gain_total`, `local_map_refinement_accepted_count`, `local_map_refinement_rejected_count`) are populated live by `Tracker` and surfaced in `examples/euroc_online_slam_vi_image_demo`'s `summary.txt`; the demo also exposes `--projection-guided-tracking`, `--projection-search-radius-px`, `--projection-widen-factor`, `--projection-max-widen-retries`, `--projection-no-local-map-refinement`, and `--projection-refinement-search-radius-px`. Unit tests cover projection matching reproducing appearance matching on a clean frame, projection matching disambiguating identical descriptors at different image locations (appearance-only matching cannot), the widen-retry ladder triggering and recovering from a slightly-off prior, the ladder falling back to appearance-global matching when every projection attempt fails, refinement's accept/reject inlier-count comparison in both directions, and the `None` default leaving legacy tracker behaviour and stats untouched. EuRoC benchmark evidence is pending a follow-up run.

- **Covisibility BA gauge-anchoring gate: ATE-primary win over disabled (2026-07-03).** Added `scripts/summarize_euroc_covisibility_anchor_gate.py` and `docs/generated/euroc_covisibility_anchor_gate.md`, plus six registry manifests under `benchmarks/registry/runs/euroc/`, evaluating the opt-in `--covisibility-local-ba-anchor-weight <w>` gauge/global-anchoring prior (`CovisibilityLocalBaConfig::pose_anchor_prior_weight`) that pins each optimized keyframe's camera centre towards its pre-BA estimate. At anchor weight 10, the same 400-frame MH_01/MH_03/MH_05 A/B used throughout this feature's evidence trail shows covisibility local BA beating the disabled baseline on the primary `ate_rigid_rmse_m` metric on all three sequences simultaneously for the first time: MH_01 `0.0561` vs `0.0642` disabled (-12.6%), MH_03 `0.0634` vs `0.0648` (-2.2%), MH_05 `0.0884` vs `0.1139` (-22.4%). The MH_05 result is a direct reversal of the previously-documented no-anchor regression (`0.1683`, worse than disabled), confirming the diagnosed failure mode was locally-consistent-but-globally-drifting solves rather than behind-camera degeneracy, and that anchoring each keyframe's camera centre to its pre-BA estimate fixes it. Both disabled and anchored arms reproduced bit-identically across repeat runs on MH_03 and MH_05. Caveat kept prominent: `tracking_success_rate` is not a simultaneous win (MH_01 improves `0.380` -> `0.672`, but MH_03 dips `0.865` -> `0.840` and MH_05 dips `0.565` -> `0.420`, even though MH_05 recovers hugely from the `0.220` no-anchor collapse), so this makes covisibility local BA ATE-safe but not yet tracking-coverage-safe, and the feature remains an explicit opt-in, not a new default. The weight (10) was chosen from a `{1,10,100,1000,10000}` sweep as the best ATE balance; higher weights over-constrain the solve, recovering some tracking but worsening ATE.

- **Single-binary deep stereo SLAM wall-clock: registry-backed evidence (2026-07-03).** Phase 3 of `docs/next_development_plan.md` formalizes the existing single-binary in-process deep-stereo-SLAM end-to-end wall-clock result documented in `docs/inprocess_slam_benchmark.md` (EuRoC MH_03_medium, full 2700-frame) as machine-readable registry evidence: two run manifests (`benchmarks/registry/runs/euroc/inprocess-deep-slam-onnx-MH_03_medium-20260703T130000Z.json`, `inprocess-deep-slam-filebased-MH_03_medium-20260703T130000Z.json`), a new `scripts/summarize_inprocess_deep_slam.py` summarizer wired into `scripts/benchmark_registry.py check-generated`, and the generated table `docs/generated/inprocess_deep_slam_wallclock.md` (front-end, dependency, wall-clock, verified loops, ATE SE(3), ATE Sim(3)). Both arms run the same loop-closure/BA config (`--online-ba --online-ba-window 10 --online-ba-history 20 --loop-closure --loop-min-frame-gap 200 --loop-two-view-ba --loop-edge-information`) through the same `stereo_vo_external_deep_files` binary, differing only in whether SuperPoint + LightGlue come from `--in-process-onnx` (ONNX Runtime CUDA, no Python) or a pre-exported `--features-dir` (Python + PyTorch, ~30 GB feature dump). No numbers were re-measured: this GPU run needs Windows CUDA ONNX Runtime provider DLLs, cuDNN 9, and a PyTorch SuperPoint/LightGlue ONNX export, all impractical to set up for this evidence-formalization pass, so both manifests are explicitly a documented prior GPU run, not reproduced this session. The headline is that the single-binary in-process ONNX front-end is 1.45x faster end-to-end (199 s vs 289 s) AND at least as accurate (0.051 m vs 0.066 m ATE SE(3); 0.047 m vs 0.057 m ATE Sim(3)) while dropping the Python export stage and its ~30 GB dump; supporting throughput context (135 fps CUDA SuperPoint extraction, 34 fps full learned front-end, 23.9 fps V2_03 single-binary VO) is cited from the sibling `docs/superpoint_onnx_cuda_benchmark.md` / `docs/lightglue_onnx_benchmark.md` registry evidence. Honest caveat kept: the two arms are not bit-identical (the file-based keypoint *sets* differ from the separate Python SuperPoint export pass; given the same features the ONNX LightGlue matches are bit-identical to the Python reference), and both land within ~2.4x of ORB-SLAM3 on this flight. README's single-binary deep stereo pipeline row now links to the generated doc.

- **SfM-vs-COLMAP head-to-head: registry-backed evidence (2026-07-03).** Phase 2 of `docs/next_development_plan.md` formalizes the existing EuRoC MH_03_medium (full 2700-frame) sequential SfM vs COLMAP head-to-head documented in `docs/sfm_vs_colmap_benchmark.md` as machine-readable registry evidence: two run manifests (`benchmarks/registry/runs/euroc/sfm-vs-colmap-visloc-MH_03_medium-20260703T120000Z.json`, `sfm-vs-colmap-colmap-MH_03_medium-20260703T120000Z.json`), a new `scripts/summarize_sfm_vs_colmap.py` summarizer wired into `scripts/benchmark_registry.py check-generated`, and the generated 5-metric table `docs/generated/sfm_vs_colmap_headtohead.md` (wall-clock, registration rate, ATE-vs-GT, mean reprojection, downstream 3DGS quality). No numbers were re-measured: COLMAP is not installed on this machine and its documented 11.7 h wall-clock cost (COLMAP 4.0.3, no CUDA, single CPU) makes a local re-run impractical, so the COLMAP manifest is explicitly marked `result_kind=external_rerun` / a prior-run reference, not reproduced this session. The generated table keeps the source doc's honest framing: visloc stereo VO + loop SfM wins metric-video speed (~117x, 6 min vs 11.7 h), accuracy (~17-33x, 0.13 m Sim3 / 0.066 m SE3 metric vs COLMAP's 2.18 m Sim3), and metric scale recovery (COLMAP's monocular reconstruction is scale-free), but the stereo-vs-monocular asymmetry is the thesis, not a thumb on the scale: on COLMAP's home turf (the 300-frame monocular subset, both engines scale-free) COLMAP still wins, 0.37 cm vs visloc 1.64 cm, and the downstream 3DGS blur on this forward-flight sequence is capture-geometry-limited for both engines, not a pose defect. README's SfM-vs-COLMAP pillar line now links to the generated doc, and the readme-claims registry entry now cites both the source doc and the two new run IDs as evidence.

- **Covisibility BA write-back conditioning gates (2026-07-03).** Two new optional, `None`-default write-back gates on `OnlineSlamCovisibilityLocalBaConfig`, integrated into the existing clone-and-check path (no second solve) alongside `max_outlier_observation_ratio`. `max_behind_camera_landmark_ratio` runs the solve on a cloned map and rejects write-back when the fraction of selected optimized landmarks that project to non-positive depth on any observing optimized camera exceeds the bound — a direct detector for the degenerate/under-constrained solves behind the MH_05 regression (e.g. an optimized=7/fixed=1 window collapsing 83% of observations behind the cameras), which the reprojection-ratio gate only caught incidentally. `min_fixed_to_optimized_ratio` rejects write-back unless `fixed_keyframe_count >= ceil(optimized_keyframe_count * ratio)`, the ratio form of the fixed-anchor requirement (the absolute-floor form on `CovisibilityLocalBaConfig::boundary_support_min_fixed_keyframes` was already swept and insufficient). Both are surfaced as `CovisibilityLocalBaError::BehindCameraGateRejected` / `FixedSupportRatioRejected` with reason strings mirroring `QualityGateRejected`, and as `covisibility_local_ba_behind_camera_gate_failures` / `covisibility_local_ba_fixed_ratio_gate_failures` counters plus the per-trigger CSV `error` column. `examples/euroc_online_slam_vi_image_demo` exposes `--covisibility-local-ba-max-behind-camera-ratio <r|none>` (validated `[0,1]`) and `--covisibility-local-ba-min-fixed-to-optimized-ratio <r|none>` (validated `> 0`). New public helpers `behind_camera_optimized_landmark_ratio`, `fixed_to_optimized_ratio_satisfied`, and `required_fixed_keyframes` are re-exported from `visloc-slam` and covered by unit tests (degenerate vs. healthy windows, ratio semantics) plus an online map-unchanged-on-rejection integration test. The default remains fully disabled (safe no-op: a rejection discards the clone and leaves the live map bit-identical). End-to-end MH_01/MH_03/MH_05 A/B validation is pending an EuRoC-equipped run; no tracking/ATE improvement is claimed here.

- **Covisibility BA write-back gate verification: verified-negative evidence (2026-07-03).** Added `scripts/summarize_euroc_covisibility_mh05_writeback_gate.py` and `docs/generated/euroc_covisibility_mh05_writeback_gate.md`, plus nine registry manifests under `benchmarks/registry/runs/euroc/`, completing the end-to-end MH_01/MH_03/MH_05 A/B/C validation left pending by the write-back conditioning gates above. Disabled/enabled-no-gate/enabled+gate (`max_behind_camera_ratio=0.3`, `min_fixed_to_optimized_ratio=0.34`) tracking success rates: MH_01 `0.380 / 0.585 / 0.705` (win), MH_03 `0.865 / 0.973 / 0.882` (marginal win), MH_05 `0.565 / 0.220 / 0.258` (still far below the `0.565` disabled baseline). The gates do not fix MH_05: the behind-camera gate never fires on any sequence at this ratio (the MH_05-corrupting solves keep low post-BA reprojection error, `0.2`-`0.5` px, so the collapse is global drift from locally-consistent solves, not behind-camera degeneracy), and MH_05 only reaches parity with disabled when the fixed-ratio gate is strict enough to reject 100% of solves (`fixed_ratio=2.0`, a true no-op); `fixed_ratio=1.0` gets MH_05 to `0.448` but drops MH_03 to about `0.860`, below its own disabled baseline. Run-to-run nondeterminism was also observed (the same MH_01 + `fixed_ratio=1.0` config produced `0.458` in one run and `0.642` in another), so gated numbers are documented as single-run, not as a stable measurement. Covisibility local BA therefore remains an explicit opt-in feature; the write-back gates do not make it safe to enable by default, and the MH_05 regression stays documented rather than swept away.

- **Public release boundary and CI gate hardening (2026-06-20).** README first-view and demo wording now remove bulky local showcase images and keep public benchmark language scoped to the registry and claim matrix. The cleanup also deletes unreferenced docs assets left behind by older README/demo presentations (`demo-preview.svg`, `visloc-hero.svg`, `localization-flow.svg`, `scanner_loop_closure_demo.png`, `euroc_v101_splat_compare.png`, and `euroc_v203_visloc_splat_compare.png`), with `tests/test_docs_assets.py` now guarding against new unreferenced `docs/assets` files. `docs/interfaces.md` now separates documented public interfaces from the stable-intent allowlist in `docs/api_stability.md`, and `docs/release_change_sets.md` groups the large release branch into reviewable P0/P1 change sets with validation commands and residual risks. CI now runs the API stability allowlist, Python registry/docs tests, release metadata gate, and trajectory-evaluation gate explicitly, while `tests/test_ci_release_gate.py` guards against CI release-gate drift. `scripts/check_release_metadata.sh`, `scripts/package_check.sh`, and `scripts/check_trajectory_evaluation.sh` are robust to Windows bash PATH and CRLF behavior.

- **Covisibility BA write-back quality gate (2026-06-20).** `OnlineSlamCovisibilityLocalBaConfig` now has optional `max_outlier_observation_ratio`: when set, online covisibility local BA runs on a cloned map first and only writes the optimized poses/landmarks back if the post-BA outlier-observation fraction stays within the configured bound. Rejected solves leave the newly-applied keyframe in the live map but report `CovisibilityLocalBaError::QualityGateRejected`, zero updated keyframes/landmarks, `observation_count`, `outlier_observation_ratio`, and `quality_gate_rejected=true` on `OnlineSlamCovisibilityLocalBaStats`. `examples/euroc_online_slam_vi_image_demo` exposes this as `--covisibility-local-ba-max-outlier-observation-ratio <r|none>`, adds `outlier_observation_ratio` and `quality_gate_rejected` to `covisibility_ba_log.txt`, records `covisibility_local_ba_quality_gate_failures` in `summary.txt`, and the EuRoC covisibility BA registry/summarizer path keeps gated and `--remove-outliers` runs separated from the default evidence. The generated `docs/generated/euroc_covisibility_mh05_quality_gate_0p3.md` table shows the ratio-`0.3` gate only fires on the unstable early cadence: `min3/every1` rejects 4/25 triggers and improves tracking versus ungated (`0.220 -> 0.265`), while `min6/every3` and `min10/every5` reject `0` triggers and reproduce their ungated trajectory metrics. The gate is therefore a write-back safety net, not a substitute for cadence/selection policy; even gated `min3/every1` still lags the disabled baseline (`0.565`) and cadence-throttled `min10/every5` (`0.525`).

- **Covisibility BA active-observation and fallback-boundary gates (2026-06-19).** `CovisibilityLocalBaConfig` now has `min_active_observations`, and `select_covisibility_local_ba_window` rejects a local BA problem when the active keyframe has too little support from the selected local landmarks. It also has opt-in `fallback_min_boundary_observations`, used only when the primary fixed-boundary threshold produces no eligible local landmarks; `CovisibilityLocalBaSelection::boundary_fallback_used` records the retry, and `InsufficientActiveObservations` records whether that retry supplied the rejected window. This keeps tracked-landmark-drop keyframe experiments from immediately feeding weak active keyframes into BA while giving `NoLocalLandmarks` windows an explicit A/B path. `examples/euroc_online_slam_vi_image_demo` exposes the gates as `--covisibility-local-ba-min-active-observations <n>` and `--covisibility-local-ba-fallback-min-boundary-observations <n|none>`, records both in `summary.txt`, adds `boundary_fallback_used` to `covisibility_ba_log.txt`, and splits covisibility BA failure counters by active-observation gate, fallback-then-active-gate, no-local-landmarks, no-observations, solver, and other failures. `scripts/run_euroc_covisibility_local_ba_ab.py`, `scripts/run_euroc_keyframe_policy_ab.py`, and the covisibility BA registry template capture the same evidence for A/B runs.

- **Covisibility BA runtime evidence (2026-06-20).** `OnlineSlamCovisibilityLocalBaStats` now records wall-clock `elapsed_ms` for each triggered covisibility local BA attempt, including selection failures. `examples/euroc_online_slam_vi_image_demo` writes that per-trigger value to `covisibility_ba_log.txt` and summarizes `covisibility_local_ba_elapsed_ms_total`, `mean`, and `max` in `summary.txt`; the EuRoC covisibility BA and keyframe-policy registry runners capture those metrics so max-landmark and window-cap A/B runs can be judged against runtime as well as trajectory error.

- **EuRoC covisibility BA runtime sweep (2026-06-20).** Added `scripts/run_euroc_covisibility_runtime_sweep.py` and `scripts/summarize_euroc_covisibility_runtime_sweep.py` to run registry-backed enabled-only landmark-cap sweeps and render `docs/generated/euroc_covisibility_runtime_sweep.md`. The initial MH_03 80-frame smoke sweep with `--covisibility-local-ba-min-active-observations 20` and fallback disabled records caps 100/200/400: cap 100 is fastest (`350.970 ms` mean) but weaker (`0.0714 m` rigid ATE, tracking `0.900`), cap 200 gives the best rigid ATE (`0.0544 m`, tracking `0.938`, `685.996 ms` mean), and cap 400 improves tracking (`0.975`) but increases the max BA spike (`1309.790 ms`) without improving rigid ATE (`0.0588 m`). This supports keeping `200` as the current MH smoke-run opt-in cap while broader KITTI/EuRoC sweeps remain pending.

- **EuRoC covisibility BA window sweep (2026-06-20).** Added `scripts/run_euroc_covisibility_window_sweep.py` and `scripts/summarize_euroc_covisibility_window_sweep.py` to keep the landmark cap fixed while varying neighbor/boundary keyframe caps, with `scripts/benchmark_registry.py check-generated` now checking both the generated 80-frame sweep and the longer 400-frame validation plus missing window manifests. The MH_01/MH_03/MH_05 80-frame smoke sweep at landmark cap `200`, active-observation floor `20`, and fallback disabled records 5/5, 10/10, and 15/15 windows; it made 5/5 look attractive on runtime, especially on MH_03. The follow-up 400-frame validation narrows the decision to 5/5 vs 10/10 and reverses that provisional reading: 10/10 improves MH_01 tracking from `0.357` to `0.585` and rigid ATE from `0.0785 m` to `0.0607 m`, improves MH_03 tracking from `0.920` to `0.973` and rigid ATE from `0.0639 m` to `0.0394 m`, and slightly improves MH_05 ATE (`0.1725 m` to `0.1683 m` rigid / `0.0971 m` to `0.0888 m` Sim(3)) while reducing MH_05 tracking (`0.343` to `0.220`). This supports keeping the existing 10/10 window budget for now; 5/5 remains a diagnostic low-cost mode, not a default candidate.

- **EuRoC covisibility BA disabled/enabled A/B summary (2026-06-20).** Added `scripts/summarize_euroc_covisibility_ab.py` and `docs/generated/euroc_covisibility_ab_400.md` so the selected 10/10, landmark-cap-200, active-floor-20 configuration is compared against a disabled baseline rather than only against alternate enabled windows. The 400-frame MH-class A/B shows real wins on MH_01 (`tracking 0.380 -> 0.585`, rigid ATE `0.0642 m -> 0.0607 m`) and MH_03 (`0.865 -> 0.973`, `0.0648 m -> 0.0394 m`), but a clear MH_05 regression (`0.565 -> 0.220`, `0.1139 m -> 0.1683 m` rigid, despite Sim(3) improving from `0.1118 m` to `0.0888 m`). `scripts/benchmark_registry.py check-generated` now also checks this A/B table and fails when either disabled or enabled manifests are missing. This keeps covisibility local BA as an evidence-backed opt-in stage rather than a default-path claim.

- **MH_05 covisibility BA mitigation sweep (2026-06-20).** Added `scripts/summarize_euroc_covisibility_mh05_mitigation.py` and `docs/generated/euroc_covisibility_mh05_mitigation.md` after the 400-frame A/B exposed the MH_05 regression. The original enabled cadence (`min_keyframes=3`, `trigger_every=1`) fired early and often, produced only `19` map keyframes, hit `11` no-local-landmark BA failures, and tracked only `0.220` of frames. Delaying and throttling BA improves stability: `min6/every3` restores `42` map keyframes, reduces no-local-landmark failures to `1`, and raises tracking to `0.445` with rigid ATE back near baseline (`0.1142 m` vs disabled `0.1139 m`); `min10/every5` eliminates BA selection failures and raises tracking to `0.525`, but rigid ATE remains worse than disabled (`0.1255 m`). This confirms the immediate failure mode is early/frequent local BA on MH_05, but also shows that cadence throttling alone is not enough to beat the baseline.

- **EuRoC active-observation sweep summary (2026-06-19).** Added `scripts/run_euroc_active_observation_sweep.py`, which runs the fixed/tracked-drop keyframe-policy A/B across requested EuRoC sequences and active-observation floors, captures registry manifests, and regenerates the sweep report. Added `scripts/summarize_euroc_active_observation_sweep.py`, which reads benchmark-registry manifests and renders `docs/generated/euroc_active_observation_sweep.md`. The generated MH_01/MH_03/MH_05 400-frame table compares `--covisibility-local-ba-min-active-observations 20` versus `50` for fixed and tracked-drop keyframe policies, keeping fallback boundary selection disabled. The recorded sweep supports using `20` as the MH smoke-run opt-in value while leaving the library default unchanged. `scripts/benchmark_registry.py check-generated` now also fails when an expected active-observation floor/sequence/variant registry manifest is missing, even if the generated Markdown contains a `missing` row.

- **Tracked-landmark keyframe promotion (2026-06-19).** `KeyframePolicyConfig` now has an opt-in tracked-landmark-drop trigger: after the normal frame-id gap, `SimpleKeyframePolicy` can promote a frame when its localization inlier count falls below a configured ratio of the last keyframe's tracked-landmark count, guarded by both last-keyframe and current-frame tracked-landmark count floors so nearly-lost frames are not promoted just because the ratio is severe. The default remains disabled for legacy A/B. `examples/euroc_online_slam_vi_image_demo` exposes the lever as `--keyframe-tracked-landmark-ratio <r>` plus `--keyframe-min-tracked-landmarks-for-ratio <n>` and records both in `summary.txt`, giving V1/V2 slow-hover EuRoC runs an ORB-SLAM-style keyframe-count lever without relying only on hand-tuned `--keyframe-min-translation`. The demo also writes `keyframe_decisions.csv`, logging every mapper decision with `KeyframeDecisionReason`, inlier count, frame-id gap, translation threshold fields, and tracked-landmark-drop diagnostics. Added `scripts/run_euroc_keyframe_policy_ab.py` and `benchmarks/registry/templates/euroc_keyframe_policy_ab_v1.json` so fixed-vs-tracked-drop runs capture paired registry manifests before any public metric claim is made; manifests include decision-derived metrics such as `keyframe_selected_count` and `keyframe_tracked_landmark_drop_count`.

- **KITTI loop retrieval recall evaluator (2026-06-18).** Added `scripts/eval_loop_retrieval_recall.py`, a dependency-free evaluator for loop-candidate recall@K/MRR against pose-derived true revisits. It reads raw retrieval or post-geometry candidate CSVs (`matched_keyframe_id`, `query_frame_id`, `score`, optional `frontend`), labels eligible true revisits from KITTI `poses.txt` or simple pose CSV files with configurable distance, temporal-gap, and path-length gates, and can emit Markdown/JSON plus optional recall gates. By default, the recall denominator is every pose query with a true revisit, so a positive query with zero emitted candidates counts as a miss; MRR likewise averages misses as zero. `VoLoopClosureResult` now retains `candidate_pairs`, plus per-candidate `verification_diagnostics` for the geometry stage, and `examples/stereo_vo_external_deep_files` writes raw gated appearance proposals to `loop_candidates.csv` plus PnP verifier outcomes to `loop_candidate_verifications.csv` whenever `--loop-closure` runs. New `scripts/capture_kitti_loop_retrieval_recall.py` runs the evaluator and captures recall artifacts/metrics into the benchmark registry; `scripts/run_kitti_multiseq_benchmark.sh --capture-retrieval-recall` and `scripts/run_kitti_loop_closure_benchmark.sh --capture-retrieval-recall` now wire that capture into KITTI runs as an opt-in artifact. `benchmarks/registry/templates/kitti_loop_retrieval_recall_v1.json` documents the manifest shape. `docs/kitti_multiseq_benchmark.md` now points seq02 work at this evaluator so stronger retrieval backends must first prove candidate-stage recall before changing verifier or optimizer settings.

- **KITTI seq02 loop verifier matching A/B tools (2026-06-19).** Added `scripts/export_lightglue_loop_candidate_matches.py`, which replays Python LightGlue over cached SuperPoint feature files for non-adjacent loop candidates and writes `loop_lightglue_pair_diagnostics.csv` plus optional `loop_OLDER_NEWER_matches.txt` files (`query=newer`, `train=older`). `examples/stereo_vo_external_deep_files` now accepts `--loop-matches-dir` and feeds those external candidate-pair matches into PnP verification when present, falling back to the built-in brute-force matcher for missing pairs; `loop_candidate_verifications.csv` records the `match_source` plus a parallel essential-matrix verifier diagnostic over the same match set. `scripts/run_kitti_multiseq_benchmark.sh --loop-matches-dir` wires the same opt-in path into full-stack KITTI runs. Empirical seq02 CPU/512 diagnostics with 400 attempted external LightGlue candidates show the external path changes the single accepted loop edge but does not increase `verified_loops` beyond 1; 383/400 candidates pass 2D-2D essential verification while PnP accepts only 1/400, so the next seq02 work should focus on 2D-3D lifting / PnP verification policy rather than retrieval or pair matching alone. A follow-up `--loop-pnp-essential-inliers` A/B kept only essential-matrix inlier matches before PnP; it still accepted 1/400 candidates, changed the accepted edge from `1924->4509` to `1974->4554`, and regressed seq02 CPU/512 ATE from `12.6603m` to `12.6906m`, so the filter stays opt-in diagnostics rather than default policy. A second follow-up `--loop-pnp-confidence-weights` A/B uses existing descriptor-match confidence to bias PnP RANSAC sampling; on the same seq02 CPU/512 run it accepts 2/400 candidates (`1015->4261`, `1944->4528`) and improves ATE to `5.8106m` SE(3) / `5.6977m` Sim(3). The confidence-weighted policy is now guarded: even when the flag is on, PnP only uses weights when at least the verifier's minimum inlier count has finite positive confidences and the confidence spread is non-uniform; otherwise it falls back to historical uniform sampling and records `pnp_weight_policy`, confidence count, and spread in `loop_candidate_verifications.csv`. Re-running seq02 CPU/512 with the guarded policy preserves the improvement exactly (`5.8106m` SE(3) / `5.6977m` Sim(3), `verified_loops=2`), with 400/400 attempted external candidates using `pnp_weight_policy=enabled` and the same accepted edges (`1015->4261`, `1944->4528`). Seq00 CPU/512 smoke non-regression passes keep ATE unchanged at 600 frames (`0.4083m` SE(3) / `0.4031m` Sim(3), 7/7 attempted candidates), 900 frames (`0.4491m` SE(3) / `0.4422m` Sim(3), 33/33 attempted candidates), 1200 frames (`0.5722m` SE(3) / `0.5511m` Sim(3), 26/26 attempted candidates), and 1500 frames (`0.9418m` SE(3) / `0.9312m` Sim(3), 36/36 attempted candidates), with `verified_loops=0` in both baseline and confidence-weighted runs. At 2000 frames, where seq00 starts producing real accepted loops, external loop matches regress against the built-in matcher: baseline BF verifies 117 loops at `1.2064m` SE(3) / `1.1299m` Sim(3), external LightGlue + uniform PnP verifies 139 loops at `1.2416m` / `1.1747m`, and external LightGlue + guarded confidence weights verifies 111 loops at `1.2419m` / `1.1711m`. The representative seq02 positive and seq00 negative/supporting runs are now captured in `benchmarks/registry/runs/kitti/`. This makes the seq02 confidence-weighted path a useful opt-in diagnostic/per-sequence lever, not a default policy.

- **Covisibility local BA MVP (2026-06-18).** Added `visloc_slam::refine_visual_map_with_covisibility_ba` and `select_covisibility_local_ba_window` as explicit A/B entry points for ORB-SLAM-style local bundle adjustment over an existing `VisualMap`. The selector starts from an active keyframe, ranks high-covisibility neighbor keyframes by shared landmark count, adds fixed boundary keyframes that observe the same local landmarks, builds a capped local landmark set, and runs the existing Schur-complement BA with robust loss. Results report the selected keyframes/landmarks, observation count, BA trace, mean reprojection before/after, map update counts, and optional post-BA outlier observation removal. `OnlineSlamConfig::covisibility_local_ba` now wires the same solver into `OnlineSlamPipeline` as an opt-in visual-only stage after new keyframe application and before pose-graph refinement; the default remains disabled. `examples/euroc_online_slam_vi_image_demo` exposes the stage through `--covisibility-local-ba` plus window/outlier knobs, writes `covisibility_ba_log.txt`, and records trigger/success/failure/reprojection totals in `summary.txt` for registry-backed A/B runs. New `scripts/run_euroc_covisibility_local_ba_ab.py` runs disabled/enabled EuRoC pairs and captures benchmark-registry manifests from the resulting artifacts.

- **Public release hardening pass (2026-06-18).** Documented canonical public import paths and the stable-intent API allowlist in `docs/api_stability.md`, clarifying that the broad root facade remains a convenience layer while `visloc_rs::prelude::*` and crate modules are the canonical surfaces for docs and examples. Added `docs/feature_matrix.md` with Tier 1 (`--no-default-features`, default, `image-io`) vs Tier 2 opt-in (`onnx-inference`, `onnx-cuda`) support boundaries, MSRV expectations, and local validation commands. Added `scripts/check_feature_matrix.sh`, wired it into `scripts/check.sh`, extended release checklist items, and added Linux/Windows CI coverage for Tier 1 feature combinations plus benchmark-registry validation.

- **Public API stability allowlist test (2026-06-19).** Added `tests/api_stability.rs`, a compile-time integration test that imports and lightly exercises the documented canonical paths for core types, replaceable algorithm traits, experimental composition layers, and the root `prelude`. `docs/api_stability.md` and `docs/release_checklist.md` now require this test to stay aligned with the stable-intent allowlist before release.

- **Public interface claim cleanup (2026-06-19).** Updated `docs/interfaces.md`, `docs/migration.md`, GNSS demo docs, roadmap/progress notes, and the loop-candidate HTML report text so public documentation no longer says pose-graph optimization or loop closure are absent. The revised wording separates the shipped opt-in loop-closure / pose-graph / BA building blocks from any production full-SLAM claim.

- **Feature matrix drift test (2026-06-19).** Added `tests/test_feature_matrix.py` to keep `docs/feature_matrix.md` aligned with the root Cargo features, Rust 1.82 MSRV declaration, `scripts/check_feature_matrix.sh`, and the Linux/Windows Tier 1 CI matrix.

- **Benchmark claim matrix (2026-06-19).** Added `benchmarks/registry/claim_matrix_v1.json` plus `docs/generated/benchmark_claim_matrix.md` so ORB-SLAM / OV2SLAM / VINS-style comparisons are scoped by dataset, sequence, sensor mode, metric, protocol, and verdict. The initial matrix records the documented KITTI ORB-SLAM2 wins, EuRoC VINS-Fusion stereo wins, OV2SLAM-near rows, and ORB-SLAM3 `behind` rows rather than implying a broad ORB-SLAM3 win. `scripts/benchmark_registry.py check-generated`, `scripts/check.sh`, and CI now validate the matrix alongside README benchmark claims and registered run evidence.

- **Benchmark-result registry v1 (2026-06-18).** Added `benchmarks/registry/` as the machine-readable evidence layer for benchmark claims. New `schema_v1.json` defines run manifests with commit SHA, `Cargo.lock` hash, feature flags, command, dataset identity/checksum fields, model hashes, hardware, config/seed, metrics, artifacts, status, and DNF/failure reason. New dependency-free helper `scripts/benchmark_registry.py` can `capture` a run manifest, `validate` registry JSON, render registered-run tables, and regenerate the README benchmark table from `benchmarks/registry/readme_claims_v1.json`. The existing README benchmark rows are now represented in that claim registry and rendered into `README.md` plus `docs/generated/benchmark_snapshot.md`; rows are explicitly labelled as `documented_historical`, `external_published`, `mixed`, or future `registered_run` instead of silently mixing evidence classes. `render-readme --with-heading` now makes the standalone `docs/generated/benchmark_snapshot.md` identify itself as the public headline snapshot, while README links to `docs/generated/registered_runs.md` for machine-readable supporting, exploratory, and negative evidence. `render-runs --with-heading` emits that registered-run evidence table with claim scope, secondary metrics, and notes. New `check-generated` fails CI when `README.md`, `docs/generated/benchmark_snapshot.md`, or `docs/generated/registered_runs.md` drift from the registry inputs. `scripts/run_kitti_multiseq_benchmark.sh` now writes `full/evaluation.json` and can capture the full ATE/loop run through `--capture-run-registry`; runs that use external loop-match or confidence-weight diagnostic flags are captured as exploratory. `scripts/check.sh` now validates registry JSON and generated benchmark docs, and `docs/release_checklist.md` requires registry-backed README rows and preservation of DNF/failure manifests.

- **Adaptive stereo depth gate v1 (2026-06-18).** `StereoFeatureConfig` now has a `StereoDepthGate` policy: `Fixed` preserves the historical `min_depth_m` gate for A/B and exact old-run replay, while the new default adaptive policy derives the effective lower-depth gate from a robust per-frame depth quantile, bounded by absolute floor/ceiling limits and smoothed by frontend-level hysteresis. The adaptive path also applies a disparity-derived relative-depth-uncertainty floor before accepting stereo candidates. `StereoVoFrontend` records per-frame `StereoDepthGateDiagnostics` (`candidate_count`, `accepted_count`, effective min/max depth, depth quantile, disparity uncertainty floor), and the KITTI / external-deep VO examples write `frontend_depth_gate_diagnostics.csv`. Existing `--min-depth` on `stereo_vo_external_deep_files` now explicitly forces the fixed gate so previous EuRoC/KITTI benchmark recipes remain reproducible. Added focused unit coverage for legacy fixed behavior, room-scale lowering, far-scene near-outlier rejection, and hysteresis stability. Benchmark reruns are still required before updating headline README numbers.

- **README KITTI revisit loop asset - replaced toy loop visual with real public-data scanner output (2026-05-22).** The README loop-candidate panel now uses `docs/assets/kitti_revisit_loop_candidate.jpg`, generated from the real KITTI 00 start/revisit scanner report instead of the synthetic 9-keyframe scanner image. `examples/kitti_revisit_scanner_demo.rs` now supports `--out-dir` report export (`summary.txt`, `candidates.csv`, `index.html`, copied strongest-pair images, strongest-pair verified-inlier overlay SVG), configurable scan/verifier knobs (`--max-features`, `--min-matches`, `--min-inliers`, `--min-inlier-ratio`, `--max-mean-sampson-error`), and reproduction-ready HTML. New cross-platform runner `scripts/run_kitti_deep_vo_revisit_smoke.py` and POSIX runner `scripts/run_kitti_deep_vo_revisit_smoke.sh` default to the quick strict public-data run (`50x30`, deep frontend, 200 features/frame), forward those knobs, can render the README JPEG directly with `--readme-asset-out`, and expose README-headline regression checks via `--readme-headline-gate` plus lower-level expectation flags (`--expect-min-candidates`, `--expect-strongest-from`, `--expect-strongest-to`, `--expect-min-strongest-inliers`, `--expect-min-strongest-ratio`). New renderer `scripts/render_kitti_revisit_report_asset.py` composes the report's strongest verified-inlier overlay into the README JPEG. Empirical quick run: KITTI 00 start frames 0..49 vs revisit frames 4500..4529 finds 41 verified cross-segment candidates; strongest pair `49 -> 4501` has 57/95 inliers, inlier ratio 0.600, score 16083.07. The generated report lives at `target/kitti_revisit_report_50x30_deep200_strict/index.html`; the README asset is reproducible from that report with the renderer script. `cargo check --features image-io --example kitti_revisit_scanner_demo` passes on the pinned Rust toolchain.

- **README COLMAP visual asset — replaced misleading Python overlay with real deep-pipeline output (2026-05-19).** The previous README header GIF (`south-building-localization-rich.gif`) was generated by `scripts/build_rich_readme_demo.py`, which drew `cv2.goodFeaturesToTrack` (classical Shi-Tomasi) overlays on top of the South Building images for visual flair — it did **not** exercise the `HogLikeFeatureExtractor` / `MutualSoftmaxMatcher` that the README's Benchmark Snapshot attributed `+37–98 %` inlier gains to. A careful viewer would notice the small visible feature count and call this out. Fixed by (1) adding a `--out-dir` correspondence-export path to `examples/deep_localization_demo.rs` that writes `correspondences.json` with per-frontend inlier (query_xy, map_xy) pairs from the actual Rust pipeline, and (2) shipping a new asset-generation script `scripts/render_deep_localization_matches.py` that consumes that JSON plus the dataset images and renders a side-by-side classical-vs-deep match overlay where each line is exactly one Rust-pipeline inlier and the title bar's match/inlier counts are read straight from the demo's output. The new asset `docs/assets/south-building-deep-vs-classical-matches.jpg` (597 KB JPEG, P1180141 → P1180144 map/query pair) shows Classical Corner+BF at 257 matches / 132 inliers vs Deep HogLike+MutualSoftmax at 473 matches / 289 inliers (+119 % inliers) — the same numbers the README's Benchmark Snapshot quotes. Removed `docs/assets/south-building-localization-rich.gif`, `docs/assets/south-building-localization-rich.png`, and `scripts/build_rich_readme_demo.py`. `docs/public_data_demo.md` updated with the new reproduction recipe.

- **Phase-27 SuperPoint ONNX runtime — skeleton activated behind `onnx-inference` feature (2026-05-19).** `crates/vision/Cargo.toml` gains an opt-in `onnx-inference` feature that pulls in `ort = "2.0.0-rc.12"` (with `download-binaries`, `ndarray`, `std`, `tls-rustls`) and `ndarray = "0.17"`. The root `Cargo.toml` exposes a pass-through `onnx-inference` feature (`= ["visloc-vision/onnx-inference"]`) so the EuRoC demo can opt-in via `cargo run --features image-io,onnx-inference`. `crates/vision/src/features/superpoint_onnx.rs` was upgraded from a `FeatureDisabled`-only stub to an actual in-Rust SuperPoint ONNX extractor: behind the feature flag, `SuperPointOnnxExtractor::load_from_path` builds an `ort::session::Session` at optimization Level 3 (wrapped in `Arc<Mutex<_>>` so the extractor remains `Clone` for stereo cam0/cam1 plumbing) and `DeepFeatureExtractor::extract_deep` runs the model per-frame with preprocess (grayscale `(1, 1, H, W) f32` in `[0, 1]`) + postprocess (auto-detected output shapes `(N, 2)/(1, N, 2)` keypoints, `(N,)/(1, N)` scores, `(N, 256)/(256, N)/(1, N, 256)/(1, 256, N)` descriptors; min-score filter; descending-score sort; top-`max_keypoints` truncation; defensive L2-normalisation). Outputs are read by name from the LightGlue-ONNX-style export convention (`keypoints`, `scores`, `descriptors`); models with different output names fail loudly with `OutputShapeMismatch`. Without the feature flag the stub remains in place, returning `SuperPointOnnxError::FeatureDisabled` so consumers fail fast with a clear pointer to `docs/superpoint_onnx_runtime_plan.md`. EuRoC demo wiring (`examples/euroc_online_slam_vi_image_demo.rs`): new `FeatureExtractorKind::SuperPointOnnx` variant, new `DemoExtractor::SuperPointOnnx(SuperPointOnnxExtractor)` variant, new CLI flag `--superpoint-onnx-model <path>`, audit-log line `superpoint_onnx_model={…}`, kind-match string `superpoint-onnx`. Stereo cam1 extraction works because `extract_deep` runs on whatever image is passed — `set_camera`/`set_frame_idx` are no-ops for this extractor; the demo can call `extract` directly on the cam1 image at seed-frame. **What still requires validation before claiming Phase-27 complete**: (a) downloading a SuperPoint ONNX model file (~10 MB) per `docs/superpoint_onnx_runtime_plan.md`'s sourcing notes (`magic-leap-research/SuperPoint`, `fabio-sim/LightGlue-ONNX` releases); (b) bit-identical descriptor regression vs Phase-26 #1 Python pre-export; (c) EuRoC V1_01 strict empirical re-run reproducing Phase-26 #1's 0.0029 m rigid ATE; (d) per-frame inference latency benchmark. The implementation is contract-correct against the LightGlue-ONNX-style export per the plan doc; validation against an actual model file is the next contributor step. New tests: `postprocess_filters_by_min_score_and_truncates_to_max_keypoints`, `postprocess_normalises_descriptors_to_unit_norm`, `postprocess_rejects_inconsistent_lengths`, `postprocess_skips_nonfinite_and_below_threshold_scores`, `normalise_descriptors_handles_all_supported_layouts` (5 new tests behind the feature; the existing 2 default-feature tests remain). Workspace builds and tests pass in both default (no onnx) and `--features onnx-inference` configurations.

- **Binary determinism — SuperPoint variant verified on the pinned toolchain, V2_01 headline corrected (2026-05-19).** `scripts/verify_binary_determinism.sh` extended to accept `VARIANT=baseline|superpoint`. SP variant uses the Phase-26 #1 strict-stereo recipe (cross-check, pose-prior warm start, local-VI-BA, stereo-bootstrap-strict, f=2/s=5) and consumes the existing pre-export at `target/euroc_phase26_superpoint/<seq>/cam{0,1}/`. Ran the three-step protocol on V2_01_easy and V1_01_easy SP+strict-stereo. **Both sequences produce bit-identical results across all three runs** (within-binary + cross-rebuild after `touch crates/vision/src/ransac/mod.rs`). Combined with the earlier baseline-corner row, the toolchain pin is confirmed as the complete fix for cross-rebuild determinism across every configuration tested. The Phase-26 #4 cross-rebuild variance was caused by `rustup update` between build moments shifting LLVM codegen; with the channel pinned to `1.94.0` that class of variance is gone. **The Kahan summation / P3P closed-form / `-Cllvm-args=-fp-contract=off` conditional levers are no longer warranted** for cross-rebuild reproducibility (still documented as fallback if a future toolchain bump re-introduces variance). **V2_01 strict SP headline corrected**: Phase-26 #1's reported 0.0107 m rigid ATE (sim_scale 1.095) was a one-time-only result on a pre-pin binary that is now gone; the pinned binary stably and reproducibly produces 0.2013 m (sim_scale 1.955) at the same configuration. V1_01 strict SP 0.0029 m (sim_scale 1.026) **reproduces exactly** on the pinned binary. The V-class breakthrough framing is preserved for V1_01 only on the current binary; V2_01 strict SP is "deterministic but in the wrong-scale regime" — a separate empirical question from determinism that may warrant a bootstrap-FP-sensitivity follow-up. `docs/binary_determinism_findings.md` empirical-results table updated with all three rows; `docs/phase_20_to_27_closeout.md` V2_01 strict table appended with the post-pin reproducible row and the Binary determinism section rewritten to reflect the resolved-by-pin outcome.

- **Binary determinism mitigation — `rust-toolchain.toml` pin + verification script + findings doc (2026-05-19).** Phase-26 #4 closeout identified that cross-rebuild variance (V2_01 strict rigid ATE shifting O(10⁻³ m) between two `cargo build --release` runs of the same source) is the most likely binary-level non-determinism remaining after the HashMap iteration fixes. New `rust-toolchain.toml` pins channel `1.94.0` with `minimal` profile + `rustfmt` / `clippy` components, eliminating the "different rustc minor version between builds" contributor. New `scripts/verify_binary_determinism.sh` runs a three-step protocol (clean build → V2_01 strict; same binary second run; touch a source file + rebuild + run) and writes `target/binary_determinism_verify/COMPARE.md` with side-by-side ATE numbers. New `docs/binary_determinism_findings.md` documents (a) the problem statement with measured magnitudes, (b) the four-hypothesis ranking (rustc codegen variation > HashMap iteration leakage > toolchain drift > parallel reduction), (c) the mitigations shipped (toolchain pin, defensive HashMap sorts from Phase-26 #4, verification script), (d) the empirical-results ledger table updated as toolchain bumps happen, (e) why deterministic-estimator swaps (Kahan summation, P3P closed-form, `-fp-contract=off`) are *not* yet shipped — they're conditional on the toolchain pin proving insufficient on real-world data, (f) cross-references to the closeout doc and the verification script. **First empirical result post-pin (2026-05-19)**: V2_01_easy strict baseline (corner extractor, adaptive-imu-pose, f=3/s=10) is **bit-identical across all three runs** (within-binary and cross-rebuild after `touch crates/vision/src/ransac/mod.rs`); `ate_rigid_rmse_m=4.8783` reproduces exactly at the printed precision. This suggests the toolchain pin alone may be sufficient for the corner-extractor baseline; the Phase-26 #1 SuperPoint+strict-stereo configuration that originally exposed the variance is not yet covered by the verification script and remains an open characterisation question (the script gates on pre-export feature files; extending it with a SuperPoint variant is the documented next step). The pin **does not** retroactively fix the Phase-26 #4 observation (those numbers were taken on pre-pin binaries); going forward, contributors who bump the channel must re-run the protocol and update the ledger.

- **Phase-{20..27} consolidation pass — unified closeout doc + defensive determinism fixes (2026-05-19).** New `docs/phase_20_to_27_closeout.md` provides a single-source-of-truth synthesis of the entire EuRoC tracker-cliff arc (Phase-20 baseline through Phase-27 ONNX skeleton): TL;DR recommended-config table (cross-class accuracy / V-class indoor accuracy opt-in / survival priority); phase-by-phase ship-and-outcome summary table (16 rows: Phase-21 cliff doc through Phase-27 plan); concrete-artifact inventory (library code, recommended vs experimental CLI flags audited, scripts, docs, tests); empirical-headline tables for the V1_01 and V2_01 strict rigid-ATE evolution across the arc (with the **V1_01 strict 0.0272 → 0.0029 m / V2_01 strict 0.0040 m strict-stereo + 0.0107 m SuperPoint** headlines); known-issues section (binary determinism, structurally-unsalvageable recovery PnP on cliffs, MH-class accuracy/continuity trade-off); next-direction list; standard-reproduction recipe for the V-class headline; reading-order pointer for a fresh contributor. **CLI knob audit** confirms all 53 flags on `examples/euroc_online_slam_vi_image_demo.rs` are justified — no removals needed; the audit categorizes each as recommended (8 flags / 4 flag-groups) vs experimental-diagnostic (6 flags from Phase-23 #1b, Phase-24 alias, Phase-26 #3a / #3b / #4) and clarifies that experimental knobs ship for reproducibility of empirical investigations even when the investigation produced a negative result. **Defensive binary-determinism fixes**: two `HashMap` / `HashSet` iteration sites (`pipelines/tracking/src/lib.rs:4126` in `Tracker::build_covisibility_local_map_store`, `pipelines/localization/src/lib.rs:409` in `descriptor_store_for_submap`) now sort landmark ids before iterating into `LandmarkDescriptorStore`. The fixes do *not* stabilize the V2_01 strict / MH_01 strict cross-build variance (Phase-26 #4 showed this rebuilds the symptom — most likely root cause is rustc-level codegen variation cascading through PnP RANSAC FP comparisons, not visible HashMap iteration), but they ship as cheap defense-in-depth and document the root-cause hypothesis (`Kahan summation` / pinned-toolchain / deterministic-robust-estimator) as a known issue beyond the arc scope. 576 workspace tests still passing.

- **Phase-27 in-Rust SuperPoint ONNX runtime — plan doc + activation skeleton (2026-05-19).** No `ort` dependency added; no model bundled; no inference code shipped. New module `crates/vision/src/features/superpoint_onnx.rs` ships a feature-gated skeleton (`SuperPointOnnxExtractor`, `SuperPointOnnxConfig`, `SuperPointOnnxError`) that implements `DeepFeatureExtractor` but returns `SuperPointOnnxError::FeatureDisabled` from every call until the `onnx-inference` feature is wired up. Re-exported from `visloc-rs` as `visloc_rs::superpoint_onnx`. Two new unit tests (`skeleton_extractor_load_returns_feature_disabled`, `default_config_matches_phase26_pre_export_settings` — verifies the in-Rust defaults match the Python pre-export `--max-keypoints 1500` setting used in Phase-26 #1 for apples-to-apples once activated). New plan doc `docs/superpoint_onnx_runtime_plan.md` covers: (a) the trade-off analysis (in-Rust ONNX vs Phase-26 #1 Python pre-export — empirically equivalent at descriptor level so this is a deployment/latency concern not research), (b) the exact Cargo dependency additions (`ort = "2.0"` with `download-binaries` feature, `ndarray`), (c) the model-file distribution strategy (do not bundle — download from `magic-leap-research/SuperPoint` or `fabio-sim/LightGlue-ONNX`), (d) the extractor implementation sketch with preprocessing / postprocessing parity requirements against `scripts/export_superpoint_lightglue.py`, (e) the EuRoC demo wiring (mirrors the existing `SuperPointOfflineExtractor` path with new `--feature-extractor superpoint-onnx` + `--superpoint-onnx-model <path>` flags), (f) the validation plan (bit-identical descriptor regression test + EuRoC V1_01 strict reproduce + latency benchmark), (g) out-of-scope items (LightGlue ONNX, multi-resolution, mobile quantisation, TensorRT/OpenVINO/CoreML EPs), and (h) reasonable wall-time estimate (4-8 hours for an experienced Rust + ML engineer with `ort` familiarity). The plan-doc-only approach respects the Phase-27 framing ("deployment win, not research win") and avoids pulling a 50 MB ONNX Runtime download into the workspace until a concrete consumer needs it. 576 workspace tests passing (574 baseline + 2 new skeleton tests).

- **Phase-26 #4 structural recovery rework (active-frontier submap + IMU sanity check) — honest negative, Phase-26 thread closed (2026-05-19).** `OnlineSlamRelocalizationConfig` (`pipelines/slam/src/lib.rs`) gained `recent_keyframe_window: Option<usize>` + `max_translation_from_imu_prediction_meters: Option<f64>` fields (both `None` defaults preserve Phase-23 #1 semantics). `maybe_run_relocalization` (`pipelines/slam/src/lib.rs:~1882`) now builds the recovery descriptor store either as full-map (default) or restricted to landmarks observed by the most recent N keyframes; post-acceptance evaluates an IMU sanity check that rejects recoveries whose camera centre is more than M meters from the tracker's per-frame motion-model prediction. New CLI flags on `examples/euroc_online_slam_vi_image_demo.rs`: `--relocalization-recent-keyframe-window <N>` and `--relocalization-max-translation-from-imu-prediction-meters <M>`. 3 existing relocalization tests updated to populate the new fields with defaults. New sweep driver `scripts/run_euroc_phase26_4_structural_recovery_sweep.sh` (6 parallel runs = 3 seqs × 2 thresholds × 1 variant with `window=5` + `max_translation=2.0`). **Empirical sweep**: 4 of 6 cases (MH_01 both, V2_01 both) bit-identical to baseline (0 recoveries — strict gate still impossible). V1_01 strict still accepts 3 false positives, V1_01 imuFavor accepts **5** (vs Phase-26 #2's 2 — an *increase*); all V1_01 recoveries land at sim_scale 0.26-0.28, rigid ATE 0.38 m, V-class breakthrough destroyed. **Diagnosis refined**: the Phase-26 #2 framing ("full-map candidate set admits wrong-scale solutions") was a symptom; the root cause is that the cliff-region landmarks themselves support wrong-scale solutions (regardless of candidate-set trimming) AND the IMU prediction post-cliff drifts into the same wrong-scale neighborhood (so a 2.0 m IMU sanity ball admits drifted recoveries). On V1_01 imuFavor the smaller candidate set actually *raised* the wrong-scale-solution inlier ratio (fewer competing correspondences) → MORE false positives accepted. Recovery PnP on EuRoC cliffs is structurally unsalvageable with the tracker-friendly intervention space tested across Phase-26 #2 / #2b / #4. **Binary-determinism caveat surfaced**: V2_01 strict / MH_01 strict baseline numbers shifted between binary builds despite per-build determinism (two consecutive runs of the current binary are bit-identical, verified). Likely root cause: `std::collections::HashMap` per-process SipHash seed leaking into RANSAC iteration order via matching stages. V1_01 strict 0.0029 m reproduces across both binaries; V2_01 strict 0.0107 m from Phase-26 #1 writeup should be re-verified on a fresh binary. Addressing the determinism root cause (deterministic iteration via `BTreeMap` or sorted keys at match-input time) is a Phase-{20..26} consolidation-pass task. **Recommendation**: do not ship the new knobs as part of any recommended config; both ship as diagnostic / experimental (`None` defaults preserve Phase-23 #1 semantics). Phase-26 #1 V-class accuracy opt-in remains unchanged. **The Phase-26 thread is empirically closed.** Remaining options: Phase-27 (in-Rust ONNX runtime, deployment win not research win); Phase-{20..26} consolidation pass (release tag + clean-up + binary-determinism fix). 574 workspace tests still passing. Full per-seq breakdown + reproduction at `target/euroc_phase26_4_structural_recovery_ab/SUMMARY.md`.

- **Phase-26 #3c MH_01 ATE regression decomposition — analysis-only closeout, MH-class trade-off precisely characterized (2026-05-19).** No Rust-side code changes. Analysis-only writeup using the existing Phase-25 HOG and Phase-26 #1 SuperPoint `slam_errors.csv` artifacts. **Method**: same-window truncation (SP first N tracked frames where N = HOG's total tracked-frame count; recompute raw RMSE), common-frame analysis (frames both methods succeed on), bin-by-frame-range (early <300 / mid 300-599 / late >=600). **Result for MH_01 strict**: raw position RMSE on the first 99 SP-tracked frames is **0.251 m vs HOG's 0.238 m — SP is only +5.2 % worse on the HOG-covered window**; the full Phase-26 #1 rigid ATE regression (0.121 → 0.198 m, +64 %) is dominated by drift accumulated over the extra 77 mid/late frames SP survives at moderate per-frame drift cost. Per-frame bins: mid 300-599 shows 2.6× SP density at +32 % per-frame error; late ≥600 shows 10× SP density at +10 % per-frame error. **Hypothesis (a) "drift in extra frames" is ~93 % of the regression; hypothesis (b) "SuperPoint outdoor descriptor penalty" is a minor ~7 % contributor (the +9 % SP-vs-HOG difference on shared 58-frame common window).** **Result for MH_01 imuFavor (different pattern)**: SP has *fewer* total tracked frames (124) than HOG (177) but extends *further* (last_frame 1069 vs 909); raw RMSE favours SP (0.366 vs 0.471 m) but rigid-aligned ATE favours HOG (0.193 vs 0.296 m, +53 % SP regression). Sparser per-region density under SP imuFavor (f=3/s=10 keeps SP in Pose mode longer) penalises Umeyama alignment against the longer-reaching but sparser trajectory; cannot cleanly attribute to (a) or (b) alone. **MH-class caveat refined**: from "SP regresses MH-class ATE" to "SP regresses *aggregate* MH-class ATE by extending tracking into harder frames; same-window per-frame accuracy is within ~5 % of HOG; use SP for trajectory continuity, HOG for accuracy." No recommendation change (HOG remains the MH-class accuracy default; SP+strict-stereo remains the V-class accuracy opt-in; both characterizations sharpened by this analysis). **The Phase-26 #3 thread (3a/3b/3c) is fully closed**: tracker-side intervention space for V-class cliff extension is empirically exhausted (#3a gate-loosening refuted, #3b matcher-swap refuted), MH-class trade-off precisely characterized (#3c). The only remaining EuRoC arc direction is Phase-26 #4 (structural map-side rework: per-keyframe submap selection + post-acceptance IMU-covariance sanity check), or pausing the Phase-{20..26} arc with a consolidation pass. Full per-seq breakdown + reproduction (just `awk` on the existing CSV files) at `target/euroc_phase26_3c_mh01_decomposition/SUMMARY.md`. 574 workspace tests still passing.

- **Phase-26 #3b V-class SuperPoint + MutualSoftmaxMatcher — honest negative, same failure mode as #3a (2026-05-19).** New CLI flag `--mutual-softmax-matcher` on `examples/euroc_online_slam_vi_image_demo.rs`. Mutually exclusive with `--cross-check-matcher` (parser enforces this with an explicit error message). `DemoMatcher` enum gained a `MutualSoftmax` variant alongside the existing `BruteForce` / `CrossCheck`; matches the same `Matcher` trait so the existing `LocalizationPipeline` construction is unchanged. The wrapped `MutualSoftmaxMatcher` uses `MutualSoftmaxConfig::default()` (`temperature = 20.0`, `min_confidence = 0.2`) — the LightGlue-style filter from `crates/vision/src/matching/mutual_softmax.rs`. New audit line `mutual_softmax_matcher=<bool>`. New sweep driver `scripts/run_euroc_phase26_3b_mutual_softmax_sweep.sh` (V-class only by default — 4 parallel runs = 2 seqs × 2 thresholds × 1 variant). **Empirical V-class sweep on top of Phase-26 #1 (SuperPoint+strict-stereo), cross-check vs mutual-softmax**: mutual-softmax extends V-class trajectories spectacularly (V1_01 strict last_frame 113 → **1452**, basically running the full `--max-frames 1500`; V2_01 strict 113 → 994) but **the extended frames are scale-wrong** in the same way Phase-26 #3a's loose PnP threshold was, and more extreme: V1_01 sim_scale 1.026 → **22.85** (order of magnitude beyond Phase-26 #3a's 6-9), V2_01 strict 1.095 → 2.32, V2_01 imuFavor 1.579 → 4.68. Rigid ATE explodes by 30-475× vs Phase-26 #1. **Refines the Phase-26 #3a diagnosis to a definitive conclusion**: the V-class cliff at frame ~113 is *not* a tracker-side gate or matcher problem. Whether one loosens the PnP threshold (#3a) or swaps to a more permissive matcher (#3b), trajectories extend by accepting marginal post-cliff correspondences that drive PnP into geometrically self-consistent but metric-incorrect solutions. **Phase-26 #1's combination of cross-check matcher + 4 px PnP gate is empirically the Pareto-optimal tracker-side filter pair** for SuperPoint V-class accuracy. The cliff problem at this stage is correspondence-set-limited (post-cliff viewpoint diverges far enough from bootstrap landmarks that the few genuinely co-visible ones cannot dominate the inlier consensus against false candidates from the full map), not tracker-filter-limited; the only remaining intervention is map-side (Phase-26 #4 submap selection + post-acceptance IMU-covariance sanity check). **The Phase-26 #3 thread is empirically closed.** Recommendation: Phase-26 #1 V-class accuracy config stays unchanged (`--cross-check-matcher` + default 4 px gate); `--mutual-softmax-matcher` ships as an experimental knob for users who want trajectory extension at any accuracy cost or for future descriptor-pipeline regression tests. Full per-seq breakdown + reproduction at `target/euroc_phase26_3b_mutual_softmax_ab/SUMMARY.md`. 574 workspace tests still passing (no behavioural change for callers that don't pass the new flag).

- **Phase-26 #3a V-class PnP-threshold sweep — honest mixed result, 4 px default confirmed optimal (2026-05-19).** New CLI flag `--pnp-reprojection-threshold-px <px>` on `examples/euroc_online_slam_vi_image_demo.rs` overrides `LocalizationConfig::reprojection_threshold` (default `4.0`); `None` preserves the default. New audit line `pnp_reprojection_threshold_px=<value>`. New sweep driver `scripts/run_euroc_phase26_3a_pnp_threshold_sweep.sh` (V-class only — 8 parallel runs = 2 seqs × 2 thresholds × 2 loosened gates {8, 12 px}). **Empirical V-class A/B vs Phase-26 #1 (4 px default)**: loosening the PnP threshold extends trajectories dramatically (V1_01 strict last_frame 113 → 530 / 602, V2_01 strict 113 → 258 / 266) but the extended frames are **scale-wrong** — V1_01 sim_scale collapses from 1.026 to **6.17 / 9.81** at 8 / 12 px (gross over-scaling, trajectory inflated 6-10×). Only borderline case is V2_01 strict @ 8 px (rigid ATE 0.0107 → 0.1085 — 10× worse than Phase-26 #1 but still **45 % below the Phase-25 HOG baseline 0.1984**), with sim_scale 1.288 (no collapse) — a real Pareto point for cliff-extension-priority use cases but not for accuracy. **Phase-26 #1's 4 px default was already optimal for V-class accuracy.** The trajectory shortening at 4 px is the *price* of metric correctness, not a gate bug. Refines the Phase-26 #2 diagnosis: the working hypothesis "stricter gate refuses HOG-accepted frames at accuracy cost" is wrong — the stricter gate is *correctly* rejecting cliff-region correspondences that cannot support a metric-correct pose under any threshold. The cliff problem at this stage is correspondence-quality limited, not gate-tuning limited; the correct interventions are either better matchers (Phase-26 #3b MutualSoftmaxMatcher) or smaller geometrically-constrained candidate sets (Phase-26 #4 structural recovery rework). The `--pnp-reprojection-threshold-px` CLI flag ships as an experimental knob; default unchanged. Phase-26 #1 V-class accuracy config and the Phase-25 cross-class defaults remain unchanged. Full per-seq breakdown + reproduction at `target/euroc_phase26_3a_pnp_threshold_ab/SUMMARY.md`. 574 workspace tests still passing (no behavioural change for callers that don't pass the new flag).

- **Phase-26 #2 / #2b relocalization re-evaluation with SuperPoint — honest negative for cliff extension (2026-05-19).** No Rust-side code changes; this round runs the Phase-23 #1 recovery PnP stage + Phase-23 #1b pose-prior-radius infrastructure on top of the Phase-26 #1 SuperPoint+strict-stereo config. Two sweep drivers shipped: `scripts/run_euroc_phase26b_superpoint_relocalization.sh` (variant #2 = `--relocalization-enabled` with strict default gates `min_inliers=20 / min_inlier_ratio=0.3 / max_reprojection_error=8.0`) and `scripts/run_euroc_phase26b2_superpoint_reloc_poseprior.sh` (variant #2b = #2 + `--relocalization-pose-prior-radius 5.0`). **Empirical 3-seq × 2-threshold × 2-variant sweep**: 4 of 6 sequence × threshold pairs (MH_01 both, V2_01 both) accept **0 recoveries out of 1300+ attempts** under both variants → bit-identical to the Phase-26 #1 baseline (Phase-23 #1's side-effect-free property reaffirmed; the strict gate is still unreachable even with SuperPoint and a pose-prior). The remaining 2 of 6 cases (V1_01 strict / V1_01 imuFavor) accept 2-4 false-positive recoveries: cliff extends +54-60 % (frame 113 → 174-180) and keyframes grow 5× (2 → 10-11), **but rigid ATE explodes from 0.0029 m to ~0.38 m (factor 130×)** with sim_scale collapsing 1.026 → ~0.27 (factor 3.7× shrink — Phase-26 #1's V-class breakthrough destroyed). The pose-prior at radius=5 m made V1_01 strict *worse* (3 → **4 false positives accepted**) because the IMU prediction itself is drifted post-cliff, so a "nearby" wrong-scale landmark still passes the radius check. **Diagnosis refines Phase-23 #1**: SuperPoint descriptors **can** reach the inlier-ratio gate on the easiest cliff regime (V1_01) but produce wrong-scale solutions because the full-map candidate landmark set admits geometrically self-consistent recoveries far from the true pose. Pose-prior radius=5 m does not fix this because the IMU prediction at recovery time is itself drifted. The MH_01 / V2_01 strict-gate impossibility is *separate* from the V1_01 false-positive problem and is harder to attack — the cliff-region viewpoint likely diverges far enough from the bootstrap that even SuperPoint cannot find 20 inliers at ratio 0.3 against the full pre-cliff landmark set. **What this rules out**: naive recovery PnP enable + pose-prior is not a viable cliff extension on EuRoC; the Phase-23 #1 recovery PnP path likely needs structural changes (per-keyframe submap selection that constrains candidate landmarks to "recently visible from the active map frontier" rather than full-map; OR a post-acceptance geometric sanity check that rejects recoveries whose pose is inconsistent with the IMU's covariance ellipsoid, not just a metric ball around the prediction). **Recommended config update — none.** Phase-26 #1 remains the V-class accuracy opt-in; do not enable `--relocalization-enabled` on top of it without further work. Phase-26 #3 follow-up candidates (decreasing payoff): #3a loosen tracker PnP RANSAC reprojection threshold for SuperPoint to extend V-class trajectories (cheapest, directly addresses the Phase-26 #1 caveat); #3b wire `MutualSoftmaxMatcher` into bootstrap and tracker (shipped but unused by #1/#2/#2b); #3c decompose MH_01 ATE regression (drift over longer window vs SuperPoint outdoor characteristic); Phase-26 #4+ structural recovery PnP rework. Full per-seq breakdown + reproduction at `target/euroc_phase26_relocalization_ab/SUMMARY.md`. 574 workspace tests still passing (no Rust changes this round).

- **Phase-26 #1 SuperPoint + strict-stereo bootstrap on Phase-25 stack — V-class breakthrough (2026-05-19).** No Rust-side code changes; this round re-runs the existing `SuperPointOfflineExtractor` (Phase-15) on top of the Phase-25 default stack (strict-stereo bootstrap + ThreePoseSmoother refresh) using the existing `--feature-extractor superpoint-offline --superpoint-features-dir <cam0_dir> --superpoint-cam1-features-dir <cam1_dir>` flag set. The only prerequisite is running the existing `scripts/export_superpoint_lightglue.py --mono-dir` once per camera per sequence to produce `frame_NNNNNN_features.txt` files. New sweep driver `scripts/run_euroc_phase26_superpoint_strict_stereo.sh` parallel-runs the Phase-25 default config × {strict, imuFavor} thresholds with the SuperPoint extractor swapped in for HOG. **Empirical 3-seq EuRoC sweep**: V1_01 strict rigid ATE 0.0272 → **0.0029 m (-89 %)** with sim_scale 1.031 → 1.026 (near-metric); V1_01 imuFavor 0.0227 → 0.0029 (-87 %); **V2_01 strict 0.1984 → 0.0107 m (-95 %)** with sim_scale 2.27 → **1.095** (gross over-scaling fixed to near-metric); V2_01 imuFavor 0.1954 → 0.1554 (-20 %). **The cleanest EuRoC result in the entire Phase-{20..26} thread is V1_01 strict SuperPoint at 0.0029 m rigid ATE — an order of magnitude better than the next-best Phase-25 number.** Trade-off on V-class is a slightly shorter trajectory before the universal cliff fires (V1_01 113 vs 158, V2_01 strict 113 vs 215) — SuperPoint's stricter PnP gate refuses marginal post-cliff frames HOG accepts at accuracy cost. MH_01 is the inverse trade: tracking density nearly doubles on strict (n_tracked 99 → 176), trajectory extends +18 % on imuFavor (last_frame 909 → 1069), but rigid ATE regresses +53 % (imuFavor) / +64 % (strict) from longer drift accumulation. **The Phase-15 negative SuperPoint finding is empirically reversed on the Phase-25 stack.** Phase-15 saw SuperPoint regress on all 3 seqs because the bootstrap landmarks were wrong-depth fallbacks at 4 m; Phase-23 #2 strict-stereo dropped those landmarks, and with real metric depths in the map the SuperPoint descriptor's quality finally shows through. The Phase-15 conclusion ("descriptor strength is NOT the binding constraint") was correct *for the Phase-15 stack* but not for the Phase-25 stack — strict-stereo moved the bottleneck *to* the descriptor on V-class sequences. **Recommended as opt-in for V-class indoor accuracy** (add `--feature-extractor superpoint-offline --superpoint-features-dir <cam0> --superpoint-cam1-features-dir <cam1>` to the Phase-25 default config). The default extractor remains `--feature-extractor hog --cross-check-matcher` to preserve the no-external-dependency story and avoid the MH-class ATE regression. Pre-export prerequisites: Python + PyTorch + LightGlue + (optional) CUDA — V2_01 cam0+cam1 1500 frames ≈ 6 min wall on RTX-class GPU. Full per-seq breakdown + reproduction at `target/euroc_phase26_superpoint_strict_stereo/SUMMARY.md`. Phase-26 #1 follow-ups: wire `MutualSoftmaxMatcher` into bootstrap (LightGlue-style matcher already shipped at `crates/vision/src/matching/mutual_softmax.rs` but unused by this round); investigate why V-class SuperPoint trajectories die at frame 113 (hypothesis: stricter PnP gate); enable Phase-23 #1 relocalization with SuperPoint descriptors to test post-cliff recovery; in-Rust ONNX runtime via `ort` crate (Phase-27 candidate, deployment concern not research).

- **Phase-25 refresh-policy A/B (zero-reset + 3-pose smoother) — V2_01 strict WIN, new default `ThreePoseSmoother` (2026-05-19).** The Phase-24 `AdaptiveImuPoseMotionModelConfig::refresh_imu_velocity_on_switch_to_imu: bool` field is replaced with `imu_velocity_refresh_policy: ImuVelocityRefreshPolicy`. The new enum has four variants — `None` (Phase-23 #4 behavior; opt-out preserves the no-refresh A/B baseline), `FiniteDifference` (Phase-24 behavior; single FD between the two most recent successful visual poses), `ZeroReset` (overwrites `velocity_world` with zeros at every switch), `ThreePoseSmoother` (averages two FDs computed across the three most recent successful poses; falls back to single FD when fewer than three are recorded). `AdaptiveImuPoseMotionModel` now tracks a 3-deep pose history (`oldest_successful_pose` + `dt_between_previous_two_observations` alongside the Phase-24 previous / latest fields). New non-mutating helper `ImuPredictiveMotionModel::body_velocity_from_camera_pose_difference` lets the smoother combine two FD velocities without intermediate state writes. EuRoC demo gained `--adaptive-motion-refresh-policy {none|finite-diff|zero-reset|three-pose-smoother}` (the Phase-24 `--adaptive-motion-no-refresh-imu-velocity-on-switch` flag is preserved as a backward-compat alias for `none`); summary audit line `adaptive_motion_imu_velocity_refresh_policy=<name>` replaces the Phase-24 bool. Three new unit tests cover the zero-reset write + counter increment, the 3-pose smoother arithmetic (4 m/s + 8 m/s ⇒ 6 m/s — verified numerically), and the smoother's degradation to single-FD when only two poses are recorded. **Empirical 3-seq EuRoC sweep (Phase-20 config + `--stereo-bootstrap-strict`, 12 runs = 3 seqs × 2 threshold sets × 2 new policies, joined against Phase-23 #4 and Phase-24 baselines)**: `ThreePoseSmoother` is the new shipping default — strictly improves on or matches `FiniteDifference` on every case tested. **V2_01 strict rigid ATE 0.2629 → 0.1984 m (-25 % vs Phase-23 #4 baseline, -36 % vs Phase-24)** — the largest motion-model-layer win since Phase-23 #2 strict-stereo; MH_01 strict 0.1219 → 0.1210 (-1 % vs Phase-23 #4 baseline, was +53 % under Phase-24); MH_01 imuFavor / V1_01 (both thresholds) / V2_01 imuFavor: bit-identical to FiniteDifference (the hook never fires or the next localization washes out the seed). `ZeroReset` is *not* viable as a default — catastrophic on MH_01 strict (rigid ATE 0.1219 → 0.6902, **+466 %**) despite a modest -9 % V2_01 strict win; the asymmetry reflects that discarding the visual motion estimate degrades the seed on body trajectories where the FD *was* informative. The Phase-23 production-recommended config (`--motion-model adaptive-imu-pose --adaptive-motion-failures-to-switch-to-pose 3 --adaptive-motion-successes-to-switch-to-imu 10 --stereo-bootstrap-strict`) is *bit-identical* under the new default because no switch-back fires at those thresholds — the default change is risk-free. Users wanting Phase-24 behavior back can pass `--adaptive-motion-refresh-policy finite-diff`; Phase-23 #4 behavior is `--adaptive-motion-refresh-policy none`. The Phase-24 noise hypothesis is partially confirmed: averaging two independent FDs halves the variance enough to flip Phase-24's V2_01 regression into a strong win, but the residual cliff oscillation on MH_01 strict (only -1 % vs baseline) is upstream of the motion-model layer — the next-thread direction remains learned (SuperPoint+LightGlue) descriptors that raise the cliff-region inlier count. Full per-seq breakdown + reproduction at `target/euroc_phase25_refresh_policy_ab/SUMMARY.md`; sweep driver at `scripts/run_euroc_phase25_refresh_policy_ab.sh`; comparison helper at `scripts/tabulate_phase25_results.sh`. 574 workspace tests passing (571 baseline + 3 new).

- **Phase-24 IMU-velocity-refresh-on-switch-to-IMU (2026-05-19, infra shipped + honest negative empirical result).** New `AdaptiveImuPoseMotionModelConfig::refresh_imu_velocity_on_switch_to_imu` field (`bool`, default `true`). When enabled, `AdaptiveImuPoseMotionModel.observe()` captures the IMU's `pending_samples_total_dt()` before forwarding (i.e. the wall-clock dt between the previous and current successful observation) and tracks `previous_successful_pose` / `latest_successful_pose`; at every Pose → IMU switch event the wrapper recomputes the inner IMU model's `velocity_world` via `update_velocity_from_camera_pose_difference(prev, latest, dt)`. New public APIs: `ImuPredictiveMotionModel::pending_samples_total_dt()` and `AdaptiveImuPoseMotionModel::velocity_refreshes_on_switch_to_imu()` (telemetry counter, ≤ `switches_to_imu()`). New CLI flag `--adaptive-motion-no-refresh-imu-velocity-on-switch` on `examples/euroc_online_slam_vi_image_demo.rs` plus two new summary audit lines (`adaptive_motion_refresh_imu_velocity_on_switch` and `adaptive_motion_velocity_refreshes_on_switch_to_imu`). Three new unit tests in `pipelines/tracking/src/lib.rs::umeyama_alignment_tests`: refresh-enabled overwrites stale velocity with visual finite-difference; refresh-disabled leaves velocity untouched (Phase-23 #4 behavior preserved for A/B); `reset()` clears the Phase-24 recent-pose state and refresh counter. **Empirical 3-seq EuRoC A/B (Phase-20 config + `--stereo-bootstrap-strict`, refresh-on vs Phase-23 #4 refresh-off)**: the architectural hypothesis (stale IMU `velocity_world` is the dominant MH_01 / V2_01 oscillation noise source) is **empirically disproved**. The refresh hook fires correctly (counter matches `switches_to_imu`) but: V1_01 f=2/s=5 rigid ATE matches Phase-23 #4 to 4 decimal places (0.0272 m / 6 KF / 2 refreshes — slow hover keeps the visual finite-difference and the stale velocity close enough that the cliff-gate outcome doesn't change); V1_01 f=3/s=10 is bit-identical (0 switches back ⇒ refresh never fires); MH_01 f=2/s=5 ATE regresses +53 % (0.122 → 0.187 m / 24 → 30 KF / 4 refreshes); MH_01 f=3/s=10 improves modestly -9 % (0.213 → 0.193 m / 35 → 37 KF / 4 refreshes); V2_01 f=2/s=5 ATE regresses +19 % (0.263 → 0.312 m / 13 → 11 KF / 3 refreshes, one extra oscillation cycle) and V2_01 f=3/s=10 is bit-identical (0 refreshes). **The most likely diagnosis: the pose-mode visual poses ARE themselves the noise** — the constant-pose branch's successive successes are dominated by PnP reprojection noise at the cliff-region landmark count, so the finite-difference velocity computed from them injects PnP noise instead of resetting to a clean state. The Phase-23 #2 strict-stereo bootstrap remains the only universal EuRoC win in the recent thread; the recommended defaults from Phase-23 are unchanged (accuracy: `--motion-model imu --stereo-bootstrap-strict`; survival on V-class: `--motion-model adaptive-imu-pose --adaptive-motion-failures-to-switch-to-pose 3 --adaptive-motion-successes-to-switch-to-imu 10 --stereo-bootstrap-strict`, for which Phase-24's refresh-on default is a no-op because no switch-back fires at those thresholds). Full per-seq breakdown + reproduction at `target/euroc_phase24_adaptive_refresh/SUMMARY.md`; sweep driver at `scripts/run_euroc_phase24_adaptive_refresh.sh`. 571 workspace tests passing (568 baseline + 3 new). The clean follow-up directions, given this round's diagnosis: (a) refresh from a 3-pose smoother instead of a single finite-difference; (b) reset-to-zero policy A/B (information-free but might dominate if visual finite-difference IS the noise source); (c) defer cliff-extension to a learned SuperPoint+LightGlue descriptor pipeline that lets relocalization recover post-cliff frames directly.

- **Phase-23 #4 adaptive IMU↔Pose motion model (2026-05-18, shipped + sequence-specific empirical result).** New `AdaptiveImuPoseMotionModel` in `pipelines/tracking/src/lib.rs` wraps `ImuPredictiveMotionModel` + `ConstantPoseMotionModel` and dispatches `predict_pose` through the currently-selected inner model. The wrapper switches modes based on per-frame counters: `failures_to_switch_to_pose` consecutive failures under IMU → switch to Pose; `successes_to_switch_to_imu` consecutive successes under Pose → switch back to IMU. Both inner models are kept fed by `observe()` regardless of which dispatches predictions, so the switch is instantaneous (the IMU's pending-sample buffer and last-successful-pose anchor stay current). Public `imu_mut()` accessor lets callers forward raw IMU samples + VI-BA-refined state into the IMU branch. `MotionModelKind::AdaptiveImuPose` + `--motion-model adaptive-imu-pose` CLI flag on `examples/euroc_online_slam_vi_image_demo.rs`, plus `--adaptive-motion-failures-to-switch-to-pose` / `--adaptive-motion-successes-to-switch-to-imu` threshold knobs. Five new unit tests covering start-mode-is-imu, switch-to-pose-after-failures, switch-back-after-successes, intermittent-failure-resets-recovery-counter, and reset. **Empirical EuRoC 3-seq A/B (Phase-20 config + `--stereo-bootstrap-strict`)**: the wrapper produces results that interpolate between the IMU and Pose endpoints, with one clean win and two oscillation-degraded results. **V1_01 with imu-favouring thresholds (f=3, s=10) is the cleanest case** — adaptive lands `rigid_ATE = 0.0227 m / 5 keyframes` vs pose-only `0.0220 m / 5 KF` and IMU-only `0.0128 m / 2 KF`, i.e. matches pose survival at near-pose ATE with one switch firing. **MH_01 / V2_01 show survival extension but ATE oscillation** because the IMU's `velocity_world` goes stale during pose-mode intervals (5 / 3 switches to pose, 4 / 2 back, repeated re-entry produces wrong first-frame predictions that re-trigger the switch). The honest follow-up is **IMU-state-refresh-on-switch** (either zero `velocity_world` at switch-back or run `update_velocity_from_camera_pose_difference` continuously during pose-mode) — out of scope for this round. The recommended EuRoC default remains `--motion-model imu --stereo-bootstrap-strict` (Phase-23 #2). The adaptive wrapper is a documented opt-in for survival-priority use cases on slow-hover sequences. Full per-seq breakdown + reproduction at `target/euroc_phase23_adaptive_motion/SUMMARY.md`.

- **Phase-23 follow-up: strict-stereo × motion-model trade-off characterised (2026-05-18, accuracy ↔ survival).** 3-seq sweep of `--stereo-bootstrap-strict` × `--motion-model {imu, velocity, pose}` shows the IMU motion model's predictive aggressiveness is double-edged: it produces the tight pre-cliff trajectory accuracy (MH_01 `0.0206 m`, V1_01 `0.0128 m`, V2_01 `0.0021 m` rigid ATE under IMU) AND the cliff itself. Switching to `--motion-model pose` extends the keyframe survival window 25–313 % (MH_01 specifically jumps 7 → 29 keyframes — a real cliff extension) but degrades rigid ATE 4–100× and collapses or over-scales the similarity scale (MH_01 `1.011 → 0.311`, V2_01 `1.000 → 2.368`). The pattern is consistent across all three seqs. **The accuracy-oriented EuRoC default remains `--motion-model imu --stereo-bootstrap-strict`** (the Phase-23 #2 headline). The architectural fix that would close the trade-off is an adaptive motion model (IMU while tracker is healthy, switch to constant-pose when IMU prediction diverges from visual PnP consensus) — out of scope for this round but the obvious follow-up for any future cliff-extension work. Sweep artifacts at `target/euroc_phase23_*_strict_mm_{velocity,pose}/`; combined writeup at `target/euroc_phase23_strict_stereo/SUMMARY.md` (updated).

- **Phase-23 #2 `--stereo-bootstrap-strict` — first measured EuRoC ATE win in the Phase-23 thread (2026-05-18).** New `strict_stereo: bool` parameter on `bootstrap_map_from_first_frame` (`examples/euroc_online_slam_vi_image_demo.rs`) plus matching CLI flag `--stereo-bootstrap-strict` (default `false`, preserves legacy mixed-depth bootstrap). When set, every cam0 keypoint that did NOT receive a stereo-triangulated cam0↔cam1 depth is dropped from the bootstrap map instead of falling back to the fixed `--bootstrap-depth 4.0` back-projection. **The original Phase-23 #2 framing was a misdiagnosis**: HOG-with-stereo-bootstrap had been the default since the demo gained HOG (`--feature-extractor hog --stereo-bootstrap`), so "wire HOG into the stereo path" was already shipped. The actual lever this phase adds is dropping the 60-70 % of bootstrap landmarks that received the wrong 4-m fallback depth — those landmarks were the dominant ATE noise source because the tracker's PnP solve included their false-depth correspondences in the inlier consensus. **Empirical 3-seq EuRoC A/B (Phase-20 config + 1500-frame cap, strict bootstrap)**: MH_01 rigid ATE `0.0265 → 0.0206 m (-22 %)`, V1_01 `0.0154 → 0.0128 m (-17 %)` with similarity scale `1.060 → 1.007 (near-metric)`, **V2_01 `0.0040 → 0.0021 m (-48 %)` with similarity scale `1.0000 (perfect metric to 5e-5 fractional error)`**. Map landmark count drops to 486/638/519 (the 30–43 % of cam0 keypoints that survived stereo matching) but the trajectory quality improves dramatically because every remaining landmark has a real metric depth. `tracking_success_rate` is unchanged or marginally up (0.021→0.023 / 0.063→0.066 / 0.051→0.051) — **this is a trajectory-quality win, not a cliff-extension win**; the Phase-21 universal cliff at f60-115 still fires and Phase-23 #3 (loop-closure on revisit) remains the unaddressed lever. New unit test `bootstrap_map_strict_stereo_drops_keypoints_without_triangulated_depth`. Full per-seq breakdown + reproduction at `target/euroc_phase23_strict_stereo/SUMMARY.md`.

- **Phase-23 #1b pose-prior-guided recovery PnP — honest negative result (2026-05-18).** New `pose_prior_candidate_radius_meters: Option<f64>` field on `OnlineSlamRelocalizationConfig` (default `None` preserves Phase-23 #1's no-prior global PnP). When set, `OnlineSlamPipeline::maybe_run_relocalization` queries the tracker's per-frame motion-model prediction via `Tracker::pose_prior_for_frame` and threads it through the localizer's `localize_frame_with_pose_prior_warm_start_and_descriptor_store` path (RANSAC warm-start + candidate-landmark radius filter). EuRoC demo gained the matching `--relocalization-pose-prior-radius <meters>` CLI flag. **Empirical 3-seq EuRoC sweep at radius={2 m, 10 m} produces 0 successful recoveries on every seq (tight radius excludes visible landmarks because the post-cliff IMU motion model prediction is `|g·Δt|`-scale off; loose radius (10 m) admits the same false positives the no-prior Phase-23 #1 path had — MH_01 collapsed to scale `0.000233` with `2101 m` rigid ATE on the 1 admitted recovery).** Side-effect-free invariant validated again (V2_01 strict + prior: bit-for-bit identical to baseline). The Phase-23 #1b extension does not rescue the Phase-23 #1 stage on EuRoC because the underlying problem is the cross-attitude HOG descriptor mismatch upstream of the recovery PnP, not the absence of a pose-prior in the localizer. New unit test `pose_prior_guided_recovery_uses_motion_prior_radius`. Full results at `target/euroc_phase23_relocalization_ab/SUMMARY.md` (updated).

- **Phase-23 #1 EuRoC A/B sweep — relocalization-on-tracker-death is empirically NOT effective at the current Phase-20 config (2026-05-18, honest negative result).** New `scripts/run_euroc_phase23_relocalization_ab.sh` drives a 3-seq × 2-variant sweep (MH_01_easy / V1_01_easy / V2_01_easy × {Phase-20 baseline, +relocalization-enabled}, 1500 frames each). **Side-effect-free invariant validated empirically**: V2_01 strict accepted 0 recoveries across 1423 attempts → ATE bit-for-bit identical to baseline (`0.0040 m` rigid, `sim_scale = 1.093072` to six decimal places). **Strict-gate recoveries regress ATE**: MH_01 4/1469 accepted → +27 % rigid ATE; V1_01 2/1395 accepted → +248 % rigid ATE. **Looser thresholds (`min_inlier_ratio=0.15, max_rep_err=16`) make it worse**: MH_01 5 accepted → similarity scale collapses to `0.030` (trajectory shrinks 33×), rigid ATE blows up to `0.960 m`. Root cause: post-cliff body attitude diverges from bootstrap-time landmarks, full-map BruteForce PnP without a pose prior cannot reach the inlier-ratio bar without admitting cheirality-flipped false positives. The infrastructure remains shipped and ready for the surviving Phase-23 candidates (#2 HOG-with-stereo-bootstrap to fix landmark depth quality, #3 pose-prior-guided recovery PnP as the smallest extension), but the stage as currently configured does not lift the universal cliff on EuRoC. Full per-seq breakdown + reproduction at `target/euroc_phase23_relocalization_ab/SUMMARY.md`.

- **Relocalization-on-tracker-death stage inside `OnlineSlamPipeline` (2026-05-18, Phase-23 candidate #1 from the EuRoC tracker-cliff thread).** New `OnlineSlamConfig::relocalization: Option<OnlineSlamRelocalizationConfig>` (min_inliers / min_inlier_ratio / max_mean_reprojection_error) opt-in attaches a running `OnlineSlamRelocalizationState` that owns a dedicated `LocalizationPipeline` instance. Inside `process_frame`, AFTER the primary `tracker.track_frame(...)` call, the stage checks `tracking.localization.success`: on failure it re-runs PnP against the full visual map via the owned localizer, gates the recovered solution against the config thresholds, and on acceptance overwrites the tracker's history via the new public `Tracker::accept_relocalization_result(...)` method (which restores `state = Tracking`, mirrors the recovered pose into `motion_model.observe(...)`, resets `successive_failures`, and bumps `stats.relocalization_count`). The recovered TrackingResult is substituted in place for the rest of `process_frame` so loop detection, mapper, IMU staging, and downstream stages all see the frame as `TrackingEvent::Relocalized`. Per-frame outcome surfaces on the new `OnlineSlamResult::relocalization: Option<OnlineSlamRelocalizationStats>` (attempted / succeeded / inlier_count / inlier_ratio / correspondence_count / mean_reprojection_error). Five new integration tests in `pipelines/slam/tests/online_slam.rs::relocalization_on_tracker_death` cover the contract: default-off, no-op on primary success, recovery on bad-camera-id-induced failure, rejected recovery leaves the tracker dead with `attempted=true,succeeded=false`, and `reset_sequence_state` clears the stage's counters. **Empirical scope and known limitations**: the recovery localizer is `LocalizationPipeline::default()` (BruteForceMatcher + AllLandmarksSelector + PnPRansac) — a future revision can let users supply a custom one for descriptor-bank / submap-radius / vocabulary-tree variants. The stage attempts recovery on every failed-tracking frame; on a sequence where the tracker dies and stays dead (e.g. EuRoC's universal cliff, see `docs/motion_based_vi_alignment.md` §Phase-21), the per-frame attempt cost is one localizer invocation regardless of whether recovery is geometrically plausible — callers who want backoff can add it via a future config knob. This stage is the prerequisite for the Phase-23 candidate #3 backlog item (loop-closure-on-dead-tracker, which uses the previously-shipped `OnlineSlamConfig::pose_graph_refinement` infrastructure once the tracker survives long enough to record return-to-scene landmarks).

- **Online loop-closure + pose-graph refinement inside `OnlineSlamPipeline` (2026-05-18).** The `LoopClosureVerifier` / `PoseGraph` / `PoseGraph::optimize_se3_iterative` machinery has existed for many releases but only as end-of-sequence standalone-demo glue. This change moves the verifier + a running pose-graph into the pipeline itself: new `OnlineSlamConfig::pose_graph_refinement: Option<OnlineSlamLoopClosureRefinementConfig>` (camera + `LoopClosureVerifierConfig` + `PoseGraphSe3Config` + `trigger_every_new_constraints`) opt-in toggle attaches a running `OnlineSlamLoopClosureRefinementState` (graph + keyframe registration order + verified-constraint accumulator + trigger counter) to the pipeline. Inside `process_frame`, after `applied_update` lands a new keyframe in the map, the stage (a) adds a node + sequential edge (relative to the previous keyframe in registration order) and anchors on the first keyframe, (b) runs `verify_loop_closure_candidates` over the candidates `detect_loop_closure_candidates` emits this frame using an `EssentialMatrixLoopClosureVerifier` built from the config, (c) folds each accepted `LoopClosureConstraint` into the graph and increments the pending counter, (d) when the pending count crosses `trigger_every_new_constraints`, fires `PoseGraph::optimize_se3_iterative` and writes the refined poses back into `self.map.keyframes[id].frame.pose`. Per-frame outcome is exposed via the new `OnlineSlamResult::pose_graph_refinement: Option<OnlineSlamLoopClosureRefinementStats>` (verified candidate count, accepted count, PGO result if it fired, keyframes updated). Six new integration tests in `pipelines/slam/tests/online_slam.rs::online_loop_closure_refinement` cover the contract: default-off, anchor-on-first-keyframe, sequential-edge accumulation, `reset_sequence_state` clearing the running state, no-op when no keyframe was registered this frame, and end-to-end PGO trigger + map write-back on a 3-keyframe return-to-origin synthetic fixture. A new reference example `examples/online_slam_pipeline_loop_closure_demo.rs` demonstrates the new API on a 6-keyframe orbit and replaces the manual `verify_loop_closure_candidates` / `loop_closure_constraints_from_candidates` / `PoseGraph` / `optimize_se3_iterative` boilerplate that `examples/online_slam_pose_graph_loop_demo.rs` shows for comparison. The pre-existing `loop_closure_candidates` field on `OnlineSlamResult` remains the diagnostic-only output for callers who want to drive their own verifier; the new field is strictly additive. `OnlineSlamConfig::default()` returns `pose_graph_refinement: None`, so the pre-existing behaviour of every callsite is preserved.

- **Phase-22 EuRoC documentation: carry-forward velocity inside the IMU motion model (2026-05-18).** New `§Phase-22` section in `docs/motion_based_vi_alignment.md` documenting the shipped `ImuPredictiveMotionModelConfig::carry_forward_velocity_world` toggle (in `pipelines/tracking/src/lib.rs`), the matching `--imu-motion-model-carry-forward-velocity` CLI flag on `examples/euroc_online_slam_vi_image_demo.rs`, the three unit tests covering the contract (`carry_forward_default_off_leaves_velocity_frozen`, `carry_forward_on_advances_velocity_per_frame`, `carry_forward_reset_clears_last_successful_pose`), the Phase-22 empirical baseline run on V1_01_easy (`target/euroc_phase22_V1_01_easy_extrinsic`) that refuted the cliff-extension hypothesis as a sufficient explanation, and the narrowed Phase-23 backlog (relocalization-on-tracker-death, HOG-with-stereo-bootstrap). The carry-forward path is shipped off-by-default as a clean architectural fix for the inter-mirror velocity desync in `ImuPredictiveMotionModel`, independent of the universal-cliff problem documented in `§Phase-21`.

- **Forster 2017 IMU pre-integration KITTI sweep (2026-05-18, Phase-5 of long-term sensor-prior integration).** No new BA / factor code — the `ImuPreintegrationFactor` infrastructure (in `pipelines/slam/src/imu_preintegration.rs` with per-keyframe velocity + bias state in `BundleAdjustment::imu_factors`) and the `--kitti-oxts-dir` / `--kitti-image-timestamps` CLI loader on `examples/stereo_vo_external_deep_files.rs` were already in place for the VI-init thread. Phase 5 ships `scripts/run_kitti_imu_preintegration_benchmark.sh` which mirrors `run_kitti_sensor_prior_only_benchmark.sh` but plugs the OXTS samples into the IMU PI factor on every BA-active seq (with strict bias gauge: `--imu-fix-first-bias on --imu-fix-first-velocity on --imu-bias-random-walk-weight 1000`). Weight sweep at `(p, v, r, brww) ∈ {(10,1,1,10), (10,1,100,10), (1000,100,100,1000), (10000,1000,1000,10000)}` converges to a `0.006 pp` band around **`mean_t_rel = 1.3011 %` (`+0.030 pp` regression vs rank70-v1)** — four weight scales spanning four orders of magnitude all land at essentially the same aggregate, and a `100×` rotation-residual emphasis is exactly identical to the gentle row; per-keyframe velocity and bias are free BA variables so the optimiser satisfies every IMU residual by adjusting `(v_i, b_i)` rather than the pose. Per-seq breakdown: seq08 improves marginally (`4.290 → 4.278`, `-0.012 pp` vs `-0.913 pp` from the per-pose gravity prior), seq03 regresses (`0.903 → 1.169`, `+0.266 pp`), all others ±0.01 pp. **The Phase 3 selective-seq08 per-pose gravity prior (`mean_t_rel = 1.1884 %`) remains the honest sensor-prior-only headline.** Phase 5 is the empirical confirmation that "just bolt on Forster 2017" is not the silver bullet for the KITTI slope-ambiguity problem at this benchmark setup (260-frame windows, no loop closure, no pose-graph layer above BA). Sweep artifacts in `target/kitti_sp_lg_vo_train_benchmark_rank70_v1_imu_preint_p*_v*_r*_brww*/`.

- **Per-pose gravity prior motion-accel correction + per-obs weight infra (2026-05-18, Phase-4 of long-term sensor-prior integration).** Three new pieces. (1) `scripts/convert_kitti_raw_oxts_to_per_pose_gravity.py` gains `--velocity-correction` (subtract central-difference of OXTS body-frame velocity `vf,vl,vu` from raw accel to remove vehicle-frame linear acceleration), `--motion-accel-soft-gate-sigma σ` (emit per-obs weight `1/(1+(|a_motion|/σ)²)`), and `--motion-accel-hard-gate τ` (mute obs where `|a_motion| > τ`). (2) `PerPoseGravityObservation` gains a `weight: f64` field; cost contribution and Jacobian assembly in `pipelines/slam/src/bundle.rs` multiply per-obs weight on top of the global `prior.weight`, with a zero per-obs weight short-circuiting the observation entirely. The CLI text reader in `examples/stereo_vo_external_deep_files.rs` accepts a 5-column `keyframe_id gx gy gz weight` line in addition to the legacy 4-column form (default weight `1.0`). (3) A new unit test `per_pose_gravity_prior_per_obs_weight_scales_cost` verifies per-obs `w=4.0` matches global `w=4.0` exactly and per-obs `w=0.0` mutes the observation under any global scale. **Empirical 11-seq result**: velocity-correction regresses at every fixed weight tested vs the uncorrected raw prior (e.g. `mean_t_rel` w=30: raw `1.842%`, vcorr `1.924%`); it helps highway-accel seqs (`seq01` w=30: `7.02 → 6.23`) but hurts slope seqs (`seq03 2.22 → 3.85`, `seq08 3.38 → 3.91`) because `vu` is world-vertical not body-vertical and OXTS Kalman-filtered velocity has ~50-100 ms lag that numerical-diff amplifies. Soft-gate at `σ=1.0` consistently moderates the high-weight regression vs the un-gated vcorr (w=10: `1.573 → 1.491`, w=30: `1.924 → 1.853`, w=50: `2.290 → 2.186`) but does not lift any setting below the visual-only baseline; the best gated aggregate (`w=10`) is still `+0.22 pp` above rank70-v1. **The selective seq08-only-at-`w=30` headline from Phase 3 (`mean_t_rel = 1.1884 %`) remains the best honest single-policy sensor-prior result on KITTI.** Phase 4 lands the infrastructure for future motion-aware priors but the empirical finding is that simple accelerometer-only / OXTS-velocity-only signals cannot uniformly beat visual-only on KITTI — the textbook IMU pre-integration approach is the actual fix.

- **11-seq sensor-prior-only KITTI aggregate, NO GT leak (2026-05-18, Phase-3 of long-term sensor-prior integration).** New `scripts/run_kitti_sensor_prior_only_benchmark.sh` extends the per-pose gravity prior from a single-seq diagnostic to the full 11 KITTI training sequences, reusing rank70-v1's cached SP/LG features and BA recipe (`01:win=30,01:tracks=200,01:huber=1.5`, `02:resid=8`, `03:win=50,03:tracks=200,03:huber=1.5`, `04:resid=5`, `06:resid=8`, `07:resid=8`, `09:resid=5`, `10:resid=8`, `00,05:skip`) per seq, and adding `--ba-per-pose-gravity-prior-{observations,weight}` on top. Fixed-weight sweeps at `w ∈ {0.5, 1, 3, 5, 10, 30}` all **regress** the 11-seq aggregate vs rank70-v1 because the seq08 win is more than offset by motion-acceleration leakage on seq01 (highway) and sliding-window-BA re-injection of noisy obs (seq03/04). Best fixed weight is `w=3` with `+0.037 pp` regression on mean_t. However, applied **selectively** to seq08 only (off elsewhere), the 11-seq aggregate improves: rank70-v1 `mean_t_rel=1.2715%, max_t_rel=2.9785%` → selective seq08-prior `mean_t_rel=1.1884% (-6.5%), max_t_rel=2.6337% (-11.6%)`, rotation neutral (`+0.0004 deg/m`). This is the **first 11-seq KITTI sub-`1.2 %` aggregate that does not rely on GT-leaking diagnostics** — it consumes only raw OXTS accelerometer + raw body→cam0 extrinsic. The rank2-ish/rank3-ish aggregates (`mean_t_rel=0.620 %`) still leak GT via post-BA xyz-projection of OXTS GT-derived camera centers, so the new `1.1884 %` row is a strictly different category. Full per-seq breakdown and weight sweep at `target/kitti_sp_lg_vo_train_benchmark_rank70_v1_per_pose_gravity_w5_weight30/SUMMARY.md`. PLAN.md's recommended-next-move #5 is now in Phase 3.

- **OXTS-derived per-keyframe gravity prior measured win on seq08 (2026-05-18, Phase-2 of long-term sensor-prior integration).** New `scripts/convert_kitti_raw_oxts_to_per_pose_gravity.py` reads raw OXTS accelerometer (body-frame `(ax, ay, az)` at fields 11-13) plus the raw body→cam0 extrinsic (`calib_imu_to_velo.txt × calib_velo_to_cam.txt × R_rect_00`), inverts the specific-force convention (`g_body = -a_body`), rotates into the rectified-cam0 frame, and rescales each observation to `|g| = 9.81 m/s²` so motion-acceleration contamination does not change the prior's residual scale. Optional `--window-half-size N` boxcar-smooths the accelerometer to suppress motion noise. The output is the `# keyframe_id gx gy gz` text file the new Phase-1 CLI flag consumes. **Empirical seq08 result**: visual-only baseline `mean_t_rel=4.290%, max_t_rel=14.310%` → with `--ba-per-pose-gravity-prior-observations <oxts-derived> --ba-per-pose-gravity-prior-weight 30` → `mean_t_rel=3.376% (-21.3 %), max_t_rel=10.516% (-26.5 %)`, with rotation error rising slightly (`0.010 → 0.014 deg/m`). This is the **first measured seq08 improvement from a real online sensor signal** — the prior consumes only raw OXTS accelerometer + raw extrinsic calibration, no GT poses; it partially closes the gap from "needs GT to fix" (post-BA OXTS-projection diagnostic, `1.084 %` mean) to "fixable from onboard sensors only" (`3.376 %` mean). Weight sweep documents a broad optimum around `weight = 20-30`; below `10` the prior under-constrains, above `~50` it over-fights visual evidence. Detailed reproduction and weight sweep at `target/kitti_seq08_per_pose_gravity_w5_weight30/SUMMARY.md`. PLAN.md's recommended-next-move #5 is now in Phase 2 (was Phase 1 earlier in the same session).

- **Per-keyframe gravity prior infrastructure (2026-05-18, Phase-1 of long-term sensor-prior integration).** New `PerPoseGravityObservation` / `PerPoseGravityPrior` types in `pipelines/slam/src/bundle.rs` accept per-keyframe `R_wc · g_world ≈ g_camera_observed` observations, wired through `BundleAdjustment` (cost contribution, Jacobian assembly modeled on the existing global `GravityPrior`, `set_per_pose_gravity_prior` setter) and re-exported at the workspace top level. `StereoVoBaConfig` gains a `per_pose_gravity_prior` field with sliding-window slicing (`slice_per_pose_gravity_prior_for_window`). `examples/stereo_vo_external_deep_files.rs` gains three new CLI flags (`--ba-per-pose-gravity-prior-observations <file>`, `--ba-per-pose-gravity-prior-weight <w>`, `--ba-per-pose-gravity-prior-g-world <gx,gy,gz>`) plus a text-format loader for `# keyframe_id gx gy gz` observation files; the audit log prints the active observation count and parameters when the flag is set. Five new unit tests cover the contract: `per_pose_gravity_prior_zero_cost_on_consistent_trajectory`, `per_pose_gravity_prior_recovers_per_keyframe_pitch`, `per_pose_gravity_prior_respects_per_keyframe_observation_independence`, `stereo_vo_ba::tests::ba_with_per_pose_gravity_prior_wires_through_config`, `stereo_vo_ba::tests::slice_per_pose_gravity_prior_remaps_local_window_ids`. This is the BA-side infrastructure for an online, sensor-only gravity prior; the OXTS-accelerometer → `g_camera_observed` derivation (which would close the GT-leak in the post-BA position-projection diagnostic) is a separate task. End-to-end CLI smoke on seq02 with a deliberately-wrong level prior changes the trajectory in the expected direction, confirming the wiring is live. PLAN.md's recommended-next-move #5 has its first-stage delivery shipped.

- **seq01 OXTS-projection dual-summary (2026-05-18).** PLAN.md's OXTS-Assisted KITTI Handoff Snapshot now documents two parallel aggregates that differ only in seq01: rank2-ish (mean-best, seq01=`xyz=0.85,0.125,0.875`, `mean_t_rel=0.620487%`, `mean_max_t_rel=1.507846%`) and rank3-ish (max-safer, seq01=`xz=0.875,0.85`, `mean_t_rel=0.621857%`, `mean_max_t_rel=1.481664%`). Tradeoff is `+0.00137 pp` aggregate mean for `-0.02618 pp` aggregate worst-window. The seq01 row in the handoff table is split into `01 (mean-best)` / `01 (max-safer)`, the README's OXTS diagnostic table shows both rows, and PLAN.md's recommended-next-move #3 is now resolved.

- **README OXTS sensor-prior diagnostic note (2026-05-18).** README §SP/LG VO + Multi-frame BA gains a "Local sensor-prior diagnostics (NOT visual-only, NOT a leaderboard claim)" subsection that compares the visual-only rank70-v1 headline (`mean_t_rel = 1.2715 %`) with the current best OXTS-assisted local aggregate (`mean_t_rel = 0.6205 %`). The subsection explicitly states that OXTS components are GNSS/IMU-derived and would not be available to a vision-only system at deployment time, so the `0.6205 %` figure is a diagnostic engineering target, not a project VO accuracy claim. PLAN.md's recommended-next-move #2 is now resolved.

- **seq02 OXTS-prior reproducibility resolved (2026-05-18).** The adopted `target/kitti_seq02_post_ba_oxts_xz_projection` artifact (`mean_t_rel=0.363934%, max_t_rel=0.795503%`) is now reproducible from current code to within machine epsilon (max `vo_poses.txt` element diff `6.94e-18`). The exact recipe is rank70-v1 BA defaults (no sliding window, `--ba-min-track-count 2000`, default `--ba-huber-delta 3`) with override `02:resid=8` and the default `0.5/0.5` confidence floor — NOT the seq01/03-style recipe (`win=30, resid=3, tracks=200, huber=1.5`) with `0.7/0.7` confidence that the earlier `_resid3_check` rerun used. The exact command lives at `target/kitti_seq02_post_ba_oxts_xz_projection_reprod_conf05_resid8.run.log`. PLAN.md's OXTS-Assisted KITTI Handoff Snapshot section and the seq02 paragraph under §27 are updated accordingly; recommended-next-move #1 is now resolved.

- **Phase-21: universal tracker-cliff diagnostic (negative-result documentation).** No code change. Investigation of Phase-20's leftover V1_01 rigid-ATE gap (0.0061 m residual vs Phase-18 baseline) discovered that the tracker dies cleanly within the first ~50–100 frames on EVERY EuRoC sequence on this bench: MH_01_easy at f62 (41 tracked frames, ~2.05 s), V1_01_easy at f115 (95 frames, ~4.75 s), V2_01_easy at f98 (77 frames, ~3.85 s). **Every Phase-13 through Phase-20 rigid-ATE figure on this bench is computed over at most the first ~5 seconds of the sequence**; the recommended configs are not invalidated but their scope is now correctly bounded — they tune the static-VI + first-few-KF behaviour, and the post-cliff regime is genuinely unhandled by the current pipeline. V1_01-specific `--max-pose-jump-meters` sweep (`0.2` default → `0.5` relaxed → omitted): with the default gate, last_tracked=f115, rigid ATE 0.0137 m, sim. scale 0.714, 2 KFs; with the relaxed gate, last_tracked=f124 (+9 frames), rigid ATE 0.0473 m (3.5× worse), 4 KFs, motion-VI fires at f123; with the gate disabled, last_tracked=f1183 (+1068 frames), rigid ATE **724 m** (sim. scale 2.9 × 10⁻⁵ — 30 000× collapse), 20 KFs, BA triggers 19. **The pose-jump gate is correctly catching the cliff** — loosening it postpones the diagnosis without recovering accuracy. `--bootstrap-depth 2.0` sweep on V1_01 (matched against the scene's true ~2 m feature distance) lands the same failure pattern (last_tracked f122, rigid ATE 0.0468 m, sim. scale 0.51) — the cliff is not a bootstrap-scale tuning artefact. Per-frame error growth at V1_01's cliff: position error 4.5 cm at f110 → 25 cm at f115 over 5 frames (0.25 s) — a 4 cm/frame drift on a body whose GT moves ~5 cm/s, dominated by a wildly wrong z-velocity in the motion-model/PnP combination. The drift direction (est_pz drops while gt_pz climbs) suggests the motion model is over-predicting downward velocity while the PnP solve under-corrects, likely because the bootstrap landmarks at ~4 m depth produce poor z-axis observability for a slow vertical hover. **Path forward (post-Phase-21, no longer about VI-init knobs)**: (1) periodic relocalization on tracker death via the existing `crates/visloc-localization::FrameLocalizer`; (2) velocity hand-off from local-VI-BA's `velocity_world` slot into the IMU motion model on every BA trigger (Phase-9 mirror-chain validation); (3) HOG-with-stereo-bootstrap variant — uses the stereo-bootstrap triangulation path with HOG features instead of SuperPoint, should produce more cam0↔cam1 matches than Phase-17's SuperPoint variant on V1/V2's low-texture indoor scenes; (4) loop-closure radius expansion, gated on (1) being in place. Phase-21 ships no code; it ships the **universal-cliff caveat** that recontextualises every prior ATE number in `docs/motion_based_vi_alignment.md` §Phase-21.

- **Phase-20: stationary-window floor lands the first universal config delivering both Phase-18 (motion-VI-init activation) and Phase-19 (mirror velocity elimination) wins simultaneously.** No code change — combines existing `--vi-init-try-initialize-on-every-frame` (Phase-19) with the existing `--vi-init-min-stationary-window-seconds 1.5` floor (`VisualInertialInitializerConfig.min_stationary_window_seconds`, default 0.5 s, previously only paired with the KF-gated path). Three-sequence A/B vs Phase-18 and Phase-19 baselines (full stack + both flags): **MH_01_easy** VI-init promoted at f52 (back to Phase-18 timing, restoring the 9-KF cadence that Phase-19's f51 fire perturbed down to 6), motion-VI-init **fires at f62 with scale=1.0** (Phase-18 success criterion preserved), `local_vi_ba_triggers` now **3** (vs 1 in P18/P19) — Phase-16's `run_at_vi_init_promotion = true` empirically additive, not redundant; rigid ATE matches Phase-18's 0.0266 m. **V1_01_easy** VI-init promoted at f51 (middle ground vs P18 f111, P19 f31), mirror velocity y component `−40.31 m/s → 0.55 m/s` (**−99 %** vs Phase-18, even tighter than Phase-19's −0.73), rigid ATE **0.0261 → 0.0137 m** (47 % of the Phase-19 regression recovered; remaining 0.0061 m to P18 baseline is the 1.5 s buffer being 3× shorter than P18's 4.5 s rotation-seed budget), similarity scale 0.518 → **0.714**. **V2_01_easy** VI-init promoted at f55 (middle ground vs P18 f98, P19 f35), mirror velocity y component 34.85 → 20.27 m/s (−42 % vs Phase-18), rigid ATE unchanged at 0.0034 m (no penalty), similarity scale unchanged at 1.118. The arithmetic: at floor=1.5 s the staged pre-promotion IMU factor's `Δt` shrinks from Phase-18's ~5 s to ~1.5 s, giving `|g · Δt| = 15 m/s` (5× smaller than Phase-18's 54 m/s baseline) — that quantity is the dominant term in the mirror velocity's converged minimum (Phase-14 / Phase-19 finding). **Recommended Phase-20 config is now universal across MH-class and V-class sequences**; see `docs/motion_based_vi_alignment.md` §Phase-20. Phase-19's documented mitigation is empirically validated as a clean win. Next levers move upstream of VI-init/motion-VI to tracker survivability and KF count (V-class sequences still emit only 2 KFs / 1500 frames at tracking_success_rate ~5 %).

- **Phase-19: decouple VI-init from KF gating.** New `OnlineSlamViInitConfig.try_initialize_on_every_frame: bool` (default `false`). When `true`, `OnlineSlamPipeline::run_vi_init_step` calls the inner `try_initialize` on every frame instead of only on frames that registered a new keyframe. On promotion without a new KF this frame, `promote_vi_init_result` binds to the latest existing keyframe (or `None` if the map is empty — the rotation rewrite and the `local_vi_ba_state.keyframe_state` seed are then skipped while the IMU pre-integrator bias reset still applies). Demo gains `--vi-init-try-initialize-on-every-frame` plus a `summary.txt::vi_init_try_initialize_on_every_frame` audit field. Two unit tests in `pipelines/slam/src/online_slam_vi_init.rs::tests` (`default_try_initialize_on_every_frame_is_false`, `config_round_trips_try_initialize_on_every_frame_override`). **Empirical finding: Phase-14's `\|g · Δt\|` mirror velocity diagnosis is empirically validated.** Three-sequence A/B vs Phase-18 (full Phase-18 stack + `--vi-init-try-initialize-on-every-frame --run-local-vi-ba-at-vi-init-promotion`, 1500 frames each, HOG): MH_01_easy VI-init promoted f52 → f51 (−1), rigid ATE 0.0266 → **0.0242 m (−9 %)** ✓. V1_01_easy VI-init promoted f111 → **f31 (−80 frames)**, mirror velocity y collapsed from `−40.31 m/s` → **`−0.73 m/s` (−98 %)** ✓✓ — confirming the gravity-integration baseline (`|g · Δt|`) shrank with the shorter pre-promotion window, but rigid ATE regressed 0.0076 → 0.0261 m (+243 %) and similarity scale dropped from 0.94 → **0.52** because at f31 only ~0.55 s of stationary samples are buffered vs ~4.5 s at f111, giving a noisier seed `R_w←b` that propagates through the bootstrap landmarks. V2_01_easy VI-init promoted f98 → f35 (−63), mirror velocity unchanged, rigid ATE unchanged at 0.0034 m. **Phase-16's promotion-time BA trigger (`run_at_vi_init_promotion=true`) is now empirically reachable.** Pre-Phase-19, that branch was structurally dead because VI-init promotion always coincided with a new-KF frame and the standard `maybe_run_local_vi_ba` already fired; with `--vi-init-try-initialize-on-every-frame` set, promotion happens on non-KF frames (V1: f31, V2: f35) where `imu_factor.is_none()` and the previously-dead Phase-16 branch now drives the single `local_vi_ba_triggers=1` event. **Trade-off characterized**: the every-frame gate trades rotation-seed precision (shorter stationary buffer → noisier `R_w←b`) for promotion latency (shorter `Δt` → smaller mirror velocity). For MH-class sequences with stronger motion and richer KF cadence this is net-win; for V-class indoor-hover sequences where rotation precision dominates, the regression is mitigable by setting a stationary-window floor (`--vi-init-min-stationary-window-seconds 1.5`). The recommended Phase-19 config in `docs/motion_based_vi_alignment.md` §Phase-19 documents both the MH-class and V-class envelopes.

- **Phase-18: motion-VI-init validation on real EuRoC.** First recorded successful firing of `OnlineSlamMotionViInitState` on a real EuRoC sequence — the previously-plumbed motion-based VIBA1 stage that Phase-3 onwards had been wired but never observed activating outside synthetic phantom datasets. Three-sequence A/B on MH_01 / V1_01 / V2_01 (1500 frames each, HOG + cross-check matcher + `--motion-vi-init --motion-vi-init-min-keyframes 3 --motion-vi-init-min-translation 0.1 --motion-vi-init-max-velocity 10.0` + `--vi-init-accel-std-limit 2.0` to admit MH_01's takeoff transient through the static-VI gate). **MH_01_easy fires `motion_vi_init_succeeded_frame=Some(62)`** with scale=1.0 (no VIBA2 outer loop), 3 keyframes / 2 IMU factors, BA converged in 5 iterations from initial cost 24.96 → 2.8e-13. Recovered KF 62 velocity `(3.19, 3.55, -3.52) m/s` (‖v‖=5.93 m/s) inside the 10 m/s sanity gate, biases `(b_g, b_a) = ((-0.011, 0.011, 0.081) rad/s, (-0.268, 0.002, 0.101) m/s²)` mirrored into `local_vi_ba_state.keyframe_state[62]` and `imu_state.config` without panic. **V1_01_easy / V2_01_easy still cannot fire** at 1500-frame bench windows: `KeyframePolicy` emits only 2 KFs (slow indoor drone hover saturates parallax after the first inter-KF baseline) and motion-VI sees `keyframes_observed=1` post-promotion, below the minimum-3 gate. **ATE-neutral milestone**: the MH_01 mirror writes refined `(velocity, bias)` into local-VI-BA state, but tracking_success_rate=2.7% means ~40/1500 frames make it through the visual tracker, none in the post-mirror frame range — rigid ATE 0.0266 m is identical whether `--motion-vi-init` is on or off. **Honest diagnosis**: the inner VIBA1 LM solve at 3 KFs / 2 factors (18 DOF / 18 constraints, just-determined) burns slack onto the unconstrained intermediate-KF biases — recovered KF 57 `bias_gyro=(−0.86, 0.22, 1.37) rad/s` is 10× the physical value because the BA cost surface is degenerate along the bias-difference null-space when the seed-bias values at the first and last KFs are anchored only by the static-VI seed (no Tikhonov-style random-walk prior between consecutive KFs, which ORB-SLAM3 uses at σ≈1e-4 rad/s/√Δt for gyro). **What this delivers**: (1) the stage's design contract — trigger fires, inner solve runs, refined values reach the mirror site — is now verifiable from a single CLI invocation against vanilla EuRoC MH_01, no synthetic data; (2) the activation envelope is mapped (accel-std gate ≥ 2.0 m/s² for MH-class motion, ≥ 3 KFs post-promotion for the inner BA to be just-determined, velocity sanity gate ≥ 6 m/s for indoor drone speeds). **Remaining gaps**: (a) the 3-KF / 2-factor solve's intermediate-KF biases are not usable as a refined linearisation point — needs a bias-drift prior to lift; (b) V1/V2 motion-VI activation is gated by `KeyframePolicy`'s 2-KF emission per 1500-frame bench, which is also the binding constraint on V1/V2's tracking_success_rate ≤ 6 %; (c) mirror-to-tracker coupling is invisible because tracking_success ~3 % on MH_01 leaves no tracked frame in the post-mirror window — validating the mirror's effect on tracking needs either a sequence where the tracker survives the takeoff transient or an isolated unit-test scope. The recommended config replicates the MH_01 fire as documented in `docs/motion_based_vi_alignment.md` §Phase-18.

- `examples/euroc_online_slam_vi_image_demo` `SuperPointOfflineExtractor` gains a stereo (cam0+cam1) loading path (`load_with_cam1`) plus per-call `set_camera(Cam0|Cam1)` switching. Demo gains `--superpoint-cam1-features-dir <path>` and lifts the previous "superpoint-offline implies --no-stereo-bootstrap" restriction whenever both feature directories are supplied. A new `summary.txt::superpoint_cam1_features_dir` audit field lands. The stereo-bootstrap path's cam1 `extract()` call now switches the extractor to cam1 + the cam1 seed frame index, then restores cam0 for the loop. Phase-17 directly addresses Phase-15's finding that the fixed 4 m back-projection (forced by `--no-stereo-bootstrap` under mono SuperPoint) was the binding constraint: with cam0+cam1 SuperPoint exports, the existing stereo-bootstrap path triangulates per-keypoint metric-depth landmarks, replacing the fixed-depth assumption that previously starved SuperPoint's per-frame PnP. **Empirical finding (Phase-17 lands the first descriptor-side win over the Phase-13 HOG baseline on rigid ATE, but the magnitude depends on the seed-frame stereo overlap).** Three-sequence A/B vs Phase-13 (full Phase-13 stack + `--feature-extractor superpoint-offline --superpoint-features-dir <cam0> --superpoint-cam1-features-dir <cam1>` — `--stereo-bootstrap` is the default, so no extra flag is required to enable it): MH_01_easy tracking_success 9.2 % → 8.5 %, map_keyframes 6 → **7**, VI-init promoted f58 → **f55 (earlier)**, mirror velocity sane in both, rigid ATE 0.0281 → **0.0265 m (−6 %)**, similarity scale 1.008 → 0.944 (slightly under metric). 668 stereo-triangulated landmarks replace the 1500 fixed-depth back-projections at the seed frame. V1_01_easy tracking_success ↑ to 0.250, KFs unchanged at 2, VI-init promoted later (f120) so the gravity-integration `Δt` actually grew, rigid ATE 0.0154 → 0.0251 m (+63 % ✗) and similarity scale 1.060 → 1.412 (worse). 514 stereo matches. V2_01_easy essentially unchanged: rigid ATE 0.0040 → 0.0042 m, similarity scale 1.093 → 1.123. 367 stereo matches. The mechanism is now traceable end-to-end: stereo-bootstrap quality is gated on the seed scene's cam0↔cam1 overlap; MH_01's well-textured warehouse scene yields 668 reliable triangulations, V1's takeoff transient's hovering motion against close-range walls yields 514 but with a less stable depth distribution, V2 yields only 367. The depth distribution at bootstrap then sets the achievable PnP accuracy on subsequent frames. **Ships behind opt-in cam1 features dir; the Phase-13 HOG stack remains the recommended config for V1/V2** until the seed-scene's stereo overlap can be guaranteed (the V1/V2 sequences' regression isn't a Phase-17 bug, it's a scene-content limitation that any stereo-triangulation bootstrap shares). MH_01 has a new candidate "best-recommended" config when the operator can guarantee the seed-frame stereo overlap matches MH_01's profile.

- `OnlineSlamLocalBaConfig.run_at_vi_init_promotion: bool` (default `false`) — Phase-16 lever. When set, `OnlineSlamPipeline::process_frame` calls `run_local_vi_ba` directly at the same frame VI-init promotes, bypassing the "new factor required" gate of `maybe_run_local_vi_ba`. The intent: when the visual tracker is fragile post-promotion (next KF arrives late or never), the promotion event itself becomes a reliable BA trigger that consumes the banked pre-promotion factors immediately. Demo gains `--run-local-vi-ba-at-vi-init-promotion` CLI flag plus a `summary.txt::run_local_vi_ba_at_vi_init_promotion` audit field. One unit test in `pipelines/slam/src/online_slam_vi_ba.rs::tests::run_at_vi_init_promotion_default_is_false` locks the default. **Empirical finding: Phase-16 is bit-identical to Phase-14 on the current 400-frame EuRoC bench across all three sequences (MH_01 / V1_01 / V2_01) — the `local_vi_ba_triggers` counter stays at `1` in every run, matching Phase-14 exactly.** The reason is structural: `run_vi_init_step` (`pipelines/slam/src/lib.rs:1644`) early-returns when `applied_update.keyframe_count == 0`, so VI-init's `try_initialize` is *only* called on frames that registered a new keyframe. VI-init promotion therefore always coincides with a new-KF event → `imu_factor` is `Some` on that frame → `maybe_run_local_vi_ba` already fires its standard BA pass → Phase-16's `if local_vi_ba.is_none()` guard fails. The promotion-time trigger path is structurally unreachable until the KF gating in `run_vi_init_step` is relaxed (substantive design change: the current contract is "VI-init's result attaches to the just-registered keyframe"; non-KF-frame promotion needs a different binding strategy). Phase-16 ships the infrastructure so when that gating is lifted, the trigger activates with one CLI arg. **Honest cross-phase takeaway**: Phase-14 (re-linearisation) lands an empirically no-op intervention because the converged BA minimum for a 1-factor system is algebraically `v_j* = R · Δv + g · Δt` regardless of linearisation point; Phase-15 (SuperPoint offline) regressed tracking_success because the bench bottleneck is the bootstrap-depth assumption, not descriptor signal-to-noise; Phase-16 (promotion-time trigger) is unreachable until VI-init's KF gating relaxes. All three ship behind opt-in defaults so the existing Phase-13 config remains the recommended one. The most productive next-iteration levers are upstream of all three: (1) stereo SuperPoint bootstrap with properly-triangulated metric-depth landmarks, (2) motion-VI-init triangulation on real EuRoC, or (3) decoupling VI-init's `try_initialize` from KF gating.

- `examples/euroc_online_slam_vi_image_demo` gains a `SuperPointOfflineExtractor` (a third `DemoExtractor` variant alongside `Corner` / `Hog`) that replays pre-computed SuperPoint mono features from a directory of `frame_NNNNNN_features.txt` files (the format already consumed by `crates/io/src/external_deep.rs` and by `examples/stereo_vo_external_deep_files`). The companion `scripts/export_superpoint_lightglue.py` script gains a `--mono-dir` mode that writes the single-camera per-frame feature files (no stereo / temporal matches — just `frame_NNNNNN_features.txt` per cam0 frame), so the offline replay can be produced from any monocular EuRoC cam0 stream with the existing Python LightGlue / torch stack. Demo gains `--feature-extractor superpoint-offline` plus `--superpoint-features-dir <path>` CLI flags; the path is required when the extractor is selected. A `summary.txt::superpoint_features_dir` audit field lands. Cross-cuts: the offline extractor requires `--no-stereo-bootstrap` (the mono pre-export covers cam0 only — wiring cam1 too is deferred); the `set_frame_idx(idx)` setter on the extractor is called by the demo before every `extract()` so the replay tracks the cam0 frame stream even when the seed offset is non-zero. This is the **infrastructure** for Phase-15 — the ONNX Runtime path (`ort` crate, online inference inside Rust) is deferred to a follow-up; for empirical evaluation purposes, the offline Python pre-export produces bit-identical SuperPoint descriptors at the BA-side, so the empirical signal does not depend on the runtime path. **Empirical finding (descriptor strength alone is NOT the binding constraint on this bench, and the Phase-15 offline path produces WORSE tracking on EuRoC's first 400 frames than the Phase-13 HOG baseline).** Three-sequence A/B vs Phase-13 (`--feature-extractor hog --cross-check-matcher` + the Phase-13 stack) under the same flags except `--feature-extractor superpoint-offline --superpoint-features-dir <path> --no-stereo-bootstrap`: MH_01_easy tracking_success 9.2 % → 1.5 %, map_keyframes 6 → 2, VI-init promoted (frame 58) → never promoted, rigid ATE 0.0281 → 0.0282 m, similarity scale 1.008 → **1.526** (much worse). V1_01_easy tracking_success ↓, map_keyframes 2 → 3, **VI-init promotes earlier (frame 33 vs HOG's pattern)** so the Phase-13 mirror velocity `(-7.3, -41.8, -37.1) m/s` collapses to a sane `(0.08, -5.6, -4.9) m/s` — the underlying gravity-integration over `Δt` shrinks because the promotion window is shorter — but rigid ATE 0.0154 → 0.071 m (worse) and similarity scale 1.060 → 0.006 (degenerate). V2_01_easy tracking_success ↓, map_keyframes 2 → 1, VI-init never promoted, rigid ATE 0.0040 → 0.031 m, similarity scale 1.093 → 0.038. The mechanism is upstream of the descriptor: the demo's seed-frame bootstrap back-projects all 1500 features at a fixed 4 m depth (when `--no-stereo-bootstrap` is on, mandatory for the mono SuperPoint pre-export). HOG's corner-style detector tends to re-detect at the same image pixel across consecutive frames, so the per-frame reprojection error from the wrong-depth landmark is roughly constant and PnP finds inliers among "similar-enough" 2D-3D pairs. SuperPoint detects at finer-grained patches whose re-detection drifts a few pixels frame-to-frame; combined with the wrong landmark depth, the per-frame PnP residuals exceed the inlier threshold and the tracker dies. **The right unlock isn't a better descriptor — it's better landmark depth at bootstrap (stereo triangulation of cam1 SuperPoint features against cam0, or letting motion-VI-init triangulate landmarks from the translation excitation it observes during takeoff).** Phase-15 ships the descriptor-replay path so subsequent iterations can A/B against any pre-exported deep descriptor; the empirical conclusion is that the binding constraint at this stack is the bootstrap-depth assumption, not descriptor signal-to-noise. (One incidental upside: V1_01's early VI-init promotion under SuperPoint reduces the mirror velocity divergence Phase-13 flagged — confirming the Phase-14 / Phase-15 derivation that the mirror velocity bound is `|g · Δt|` over the pre-promotion window. The cost was the rest of the trajectory, but the mechanism is now verifiable.)

- `ImuPreintegratedDelta::relinearise_at(&mut self, bias_gyro, bias_acc)` — in-place ORB-SLAM3-style re-linearisation. Bakes the first-order bias correction `δb = new_bias − old_linearisation` into `delta_rotation / delta_velocity / delta_position` via the existing `corrected()` Jacobians, then resets `bias_*_linearisation = new_bias` so subsequent `residual_with_bias_correction` evaluations at biases near `new_bias` stay in the small-`δb` regime. The Jacobians (`j_rotation_bg`, `j_velocity_ba`, `j_velocity_bg`, `j_position_ba`, `j_position_bg`) are preserved unchanged — this is the standard trick that avoids the O(N) re-integration cost by accepting that the bias-Jacobians don't move much for small bias shifts (full re-integration would be more accurate for very large jumps but requires holding raw IMU samples, which the factor does not). Two unit tests in `pipelines/slam/src/imu_preintegration.rs::tests` cover the new method: `relinearise_at_updates_linearisation_point_and_bakes_correction` (build a delta at `b_lin = 0`, re-linearise at a non-trivial `b_new`, assert the stored linearisation point is now `b_new`, the baked deltas equal what `corrected()` produced from the old point, and `corrected(b_new, b_new)` reproduces the baked deltas exactly — the trick is identity at the re-linearisation boundary by construction); `relinearise_at_zero_delta_is_no_op_on_identity` (re-linearising at the same point is a perfect no-op).

- `OnlineSlamLocalBaConfig.relinearise_imu_factor_bias_thresholds: Option<(f64, f64)>` (default `None`) — threshold-gated IMU factor re-linearisation. When `Some((gyro_rad_s, accel_m_s2))`, `run_local_vi_ba` walks `state.factor_history` before every BA pass and re-bakes any factor whose stored `bias_*_linearisation` differs from the current per-keyframe bias estimate of its `keyframe_id_from` by more than the threshold. Mutates in-place so future BA windows inherit the refreshed delta; counts the refresh in the new `OnlineSlamLocalBaStats.relinearised_factor_count` field. **Companion** to `keep_pre_promotion_imu_factors`: with both opt-ins on, banked pre-promotion factors (placeholder `b_lin = 0`) get refreshed to the post-promotion bias estimate before the BA solves them, so the linear bias-correction stays inside its small-`δb` validity radius. Demo gains a `--relinearise-imu-factor-bias-thresholds <gyro_rad_s>,<accel_m_s2>` CLI flag plus two `summary.txt` audit fields (`relinearise_imu_factor_bias_thresholds`, `local_vi_ba_relinearised_factor_total`). Three integration tests in `pipelines/slam/src/online_slam_vi_ba.rs::tests`: `relinearise_threshold_off_leaves_factor_linearisation_at_construction_value` (default `None` → counter stays 0, linearisation point unchanged), `relinearise_threshold_refreshes_factors_above_drift` (per-kf bias far from `b_lin = 0` → counter = factor_count, each factor's `bias_*_linearisation` matches its from-keyframe's state), `relinearise_threshold_skips_factors_within_drift` (generous threshold `(1.0, 10.0)` → no factor exceeds → counter stays 0). **Empirical finding (Phase-14 is the right mechanism but the current bench doesn't have the conditions to exercise it).** Three-sequence A/B at the recommended Phase-13 stack + `--relinearise-imu-factor-bias-thresholds 0.01,0.1` (first 400 frames): the `relinearised_factor_total` counter rises from `0` to factor-count on every BA trigger across MH_01 (4-of-5), V1_01 (1-of-1), V2_01 (1-of-1), confirming the threshold fires and the in-place baking executes; but rigid ATE, similarity scale, AND the mirrored velocity components are bit-identical to Phase-13. Re-deriving Forster eq. 45-47 for the single-factor case explains the no-op: `r_v = R_iᵀ · (v_j − v_i − g · Δt) − Δv` is just-determined (3 residual rows × 3 `v_j` DOFs), so the BA's converged `v_j* = R_i · Δv + g · Δt`. The mirror velocity magnitude on V1/V2 (~40 m/s y-component) is the gravity-integration baseline `g · Δt ≈ -39 m/s` over the ~4 s pre-promotion window — **not** bias-extrapolation residual. Re-linearisation bakes the bias correction into `Δv` and shifts `b_lin`, but `Δv_baked + J · (b − b_new) = Δv_raw + J · b` is an algebraic identity at any `b`, so the converged minimum doesn't move. The Phase-13 "bias-extrapolation" framing was a partial diagnosis; the actual mirror velocity bound is `|g · Δt|`, and shrinking that requires more keyframes per BA window (Phase-15 visual descriptor unblock) or smaller per-factor `Δt` (Phase-16 per-keyframe BA cadence). The infrastructure ships opt-in so it pays no per-frame cost; it'll surface its value once Phase-15 / Phase-16 grow the factor count enough that bias-correction accuracy matters.

- `OnlineSlamConfig.keep_pre_promotion_imu_factors: bool` (default `false`) — when set, the stale-factor gate in `OnlineSlamPipeline::stage_imu_factor_on_new_keyframe` no longer discards IMU factors built on keyframes that registered while the auto-bootstrap stage was still active. The factors retain their placeholder bias linearisation but bank into `result.imu_factor` and the local-VI-BA's `factor_history`. **Companion change** to `maybe_run_local_vi_ba`: when the stage is still active, the BA accumulates factor banks but does NOT execute the solver — running BA with placeholder biases corrupts the map's keyframe poses and empirically collapses tracking-success on real EuRoC. Once VI-init promotes, the next post-promotion factor unlocks a single BA pass that consumes the full banked history at the correct bias linearisation. Three pre-existing test files (`pipelines/slam/tests/online_slam.rs`, `examples/euroc_online_slam_vi_demo.rs`, `examples/euroc_online_slam_vi_image_demo.rs`) needed `keep_pre_promotion_imu_factors: false` added to their explicit `OnlineSlamConfig` literals to compile under the new field. Demo gains a `--keep-pre-promotion-imu-factors` opt-in flag + a `summary.txt::keep_pre_promotion_imu_factors` audit field.

- **Phase-13 empirical breakthrough: this is the first config on the bench that beats the Phase-7 baseline on rigid ATE AND lands near-metric similarity scale across all three EuRoC sequences (first 400 frames).** With the full Phase-11 stack (`--feature-extractor hog --cross-check-matcher --keyframe-min-translation 0.1 --max-pose-jump-meters 0.2 --motion-model imu --pnp-pose-prior-warm-start --vi-init-gyro-std-limit 0.5 --vi-init-accel-std-limit 5.0 --local-vi-ba`) plus `--keep-pre-promotion-imu-factors`: MH_01_easy rigid ATE 0.041 → **0.027 m (-34 %)**, similarity scale 1.112 → 1.016 (near-metric); V1_01_easy rigid ATE 0.022 → **0.015 m (-32 %)**, scale 0.246 → **1.060 (Procrustes shrinkage gone)**; V2_01_easy rigid ATE 0.025 → **0.004 m (-84 %)**, scale 1.216 → 1.093 (closer to metric from the over-scaled side). The new diagnostic `imu_factors_staged` confirms the wiring: it rises from `1` (Phase-12 baseline, single post-promotion factor) to `6` on MH_01, meaning 5 pre-promotion factors are now banked AND consumed by the post-promotion BA pass. The single local-VI-BA trigger still occurs once per 400 frames (the visual tracker still dies on the takeoff transient), but the BA pass it runs is now informed by 6× more IMU constraints — the resulting pose refinement on the trailing 5 keyframes is what drives the ATE collapse. The mirrored velocity values diverge to unphysical magnitudes (e.g. V1: `-41.8 m/s y-component`) because the velocity slot absorbs the bias error from placeholder linearisation, but this doesn't affect rigid ATE — the IMU motion model's velocity prior matters only on the next IMU-predicted frame, and tracking dies before any such frame arrives. The recommended config moves to the above flags-stack.

- `examples/euroc_online_slam_vi_image_demo` gains two static-VI-init tuning CLI knobs (`--vi-init-min-samples <n>` and `--vi-init-min-stationary-window-seconds <s>`, overrides for `VisualInertialInitializerConfig.min_samples` / `min_stationary_window_seconds`) plus a new `summary.txt::imu_factors_staged` diagnostic counter that reports how many `OnlineSlamPipeline::process_frame` calls actually returned `result.imu_factor.is_some()` (post stale-factor gate). **Empirical finding from Phase-12 (the trigger-frequency knob is not the bottleneck on real EuRoC):** with the Phase-11 winning config on MH_01_easy, `imu_factors_staged = 1` despite `map_keyframes = 8` — 7-of-8 keyframes register BEFORE VI-init promotes at frame 54, and the stale-factor gate (the documented policy that discards IMU factors carrying placeholder bias linearisations) drops their factors; only the single keyframe after VI-init promotion emits a usable factor, which fires exactly one local-VI-BA trigger / one mirror into the IMU motion model. The pipeline already runs `OnlineSlamLocalBaConfig.trigger_every = 1` (its default), so the BA fires on the very first usable factor and there's no room to "tighten" further. Lowering `--vi-init-min-samples 10 --vi-init-min-stationary-window-seconds 0.05` is bit-identical to the default (the actual gate is `max_gyro_std` / `max_accel_std`, not sample count — the buffer fills 200×/sec and the std-gate fails until the drone hovers momentarily at frame 54). Enabling motion-VI-init (`--motion-vi-init --motion-vi-init-min-keyframes 3 --motion-vi-init-min-translation 0.3`) also stays in `Waiting` — the post-promotion keyframe count never accumulates because the visual tracker dies on the takeoff transient at frame ~60. The single trigger's mirrored `velocity_world ≈ (0.60, 0.38, -3.47) m/s` is plausible but the prior cannot survive the visual-quality cliff at takeoff onset. **Conclusion: the chain bottleneck is upstream visual tracking quality across the takeoff transient — descriptor / matcher / motion-prior all on the visual side — not the BA's trigger frequency.** The new diagnostic counter ships so the next iteration can verify the chain depth at a glance; the empirical numbers are documented as a baseline.

- `examples/euroc_online_slam_vi_image_demo` gains `--feature-extractor {corner, hog}` (default `corner`, the existing `CornerFeatureExtractor` with raw patch descriptors; `hog` selects `HogLikeFeatureExtractor`, the deep-shaped 128-D HOG/SIFT-flavored unit-norm descriptor already shipped in `crates/vision/src/features/deep.rs`), three HOG tuning knobs `--hog-max-features <N>` / `--hog-min-corner-score <s>` / `--hog-orient`, and `--keyframe-min-translation <m>` (overrides `KeyframePolicyConfig.min_translation`, library default `1.0 m`). A new `DemoExtractor` local enum unifies the two backing extractors' divergent error types (`String`-based) so a single `Tracker<..., LocalizationPipeline<DemoMatcher, _, _>>` type carries through the demo. Two new `summary.txt` fields land: `feature_extractor`, `keyframe_min_translation`. **Empirical finding (this is the first opt-in config that beats the Phase-7 baseline on rigid ATE on real EuRoC):** the three-knob combination `--feature-extractor hog --cross-check-matcher --keyframe-min-translation 0.1` plus the existing `--max-pose-jump-meters 0.2 --motion-model imu --pnp-pose-prior-warm-start --vi-init-gyro-std-limit 0.5 --vi-init-accel-std-limit 5.0 --local-vi-ba` flags unblocks the keyframe-registration loop that previously stalled at 1 keyframe per 400 frames on MH_01. The HOG descriptor raises the per-frame match recall (corner+patch's 1 % tracking-success → HOG-alone's 2 %); cross-check filters HOG's higher-recall but lower-precision matches (→ HOG+cc 35.5 % tracking-success); the relaxed keyframe-min-translation lets the EuRoC takeoff cadence register useful keyframes (→ 8 KFs at gate=0.2 m). Once keyframes accumulate, VI-init promotes (frame 54 on MH_01), local-VI-BA fires, the Phase-9 mirror activates, and the IMU motion model's refined `(velocity, biases)` feeds back into the tracker. **Three-sequence A/B vs the Phase-7 baseline (first 400 frames):** MH_01_easy rigid ATE 0.041 → **0.033 m (-19 %)**, similarity scale 1.112 → **1.001 (essentially perfect metric)**, orientation RMSE 0.72 → 0.66 deg. V2_01_easy rigid ATE 0.025 → **0.016 m (-37 %)**, similarity scale 1.216 → 0.939, orientation RMSE 0.46 → 0.30 deg. V1_01_easy is mixed: rigid ATE 0.022 → 0.034 m (+55 %) but the V1 baseline carried an outlier-shrunk similarity-scale of `0.246` (the trajectory was geometrically broken even at low rigid-ATE); Phase-11 fixes the scale (0.246 → 0.803, ~3× more metric) at the cost of a small rigid-ATE regression. Two-of-three sequences improve unambiguously; V1 trades rigid for metric correctness. An ablation isolates the contribution: `--keyframe-min-translation 0.1` alone (without HOG / cross-check) on MH_01 stays at 1 keyframe and rigid ATE 0.041 m — the kf relaxation only matters when HOG+cc supplies the tracking-success the kf threshold needs to register against. **The new recommended config** for stronger metric-scale recovery is documented as `--feature-extractor hog --cross-check-matcher --keyframe-min-translation 0.1` on top of the Phase-8b stack.

- `examples/euroc_online_slam_vi_image_demo` gains the `--cross-check-matcher` CLI flag (default off) that wraps the default `BruteForceMatcher` (Lowe ratio 0.8) in `CrossCheckMatcher`, keeping only query↔train descriptor pairs where each side picks the other as its single best match. A new `DemoMatcher` local enum dispatches between the two at runtime (mirroring the `DemoMotionModel` pattern) so a single concrete `LocalizationPipeline<DemoMatcher, _, _>` type carries through. One new `summary.txt` field lands: `cross_check_matcher`. **Empirical finding (the wiring lands a real win on the loose-gate VI-BA path, but does NOT unblock the Phase-7 tight-gate keyframe-registration floor):** at `--max-pose-jump-meters 0.2 --motion-model imu --pnp-pose-prior-warm-start`, cross-check actually worsens MH_01 (rigid ATE 0.041 → 0.080 m, tracking-success 1.0 % → 3.0 %, gate failures 165 → 222) — the per-frame match recall drops and PnP starves on a thin candidate set. At the loose-gate VI-BA configuration (`--max-pose-jump-meters 1.0 --vi-init-gyro-std-limit 0.5 --vi-init-accel-std-limit 5.0 --local-vi-ba`), cross-check **does** improve rigid ATE (0.261 → 0.163 m, 1.6× tighter), similarity scale (0.187 → 0.366), and tracking-success rate (21 % → 31 %), but in turn reduces the keyframe count enough that the local-VI-BA trigger no longer fires (1 → 0 triggers, the mirror never activates this run). Neither configuration beats the Phase-7 baseline on rigid ATE. The diagnostic conclusion: cross-check trades precision for recall and recall is the dominant constraint on real EuRoC's noisy corner+patch descriptor — the deeper bottleneck is descriptor signal-to-noise (deeper descriptor / better local image representation), not match-filter strictness. The capability ships behind the flag so it's one CLI arg away whenever a downstream config wants the precision-for-recall trade.

- `examples/euroc_online_slam_vi_image_demo` wires the local-VI-BA's refined per-keyframe `(velocity_world, bias_gyro, bias_acc)` into `ImuPredictiveMotionModel::set_velocity_world` / `set_biases` (the documented hooks the model exposes for downstream solvers to refresh the strapdown integrator's state). When `--local-vi-ba` is on AND `--motion-model imu` is selected, every `OnlineSlamResult.local_vi_ba` carrying a successful (non-bias-frozen) trigger pulls the trigger's window's most recent keyframe's `KeyframeImuState` and mirrors it into the tracker's motion model via the new `DemoMotionModel::mirror_vi_ba_state`. When `--local-vi-ba` is off (default), `result.local_vi_ba` is always `None` and the mirror is a no-op — Phase-7/8 behaviour preserved bit-for-bit. Five new `summary.txt` fields land: `local_vi_ba_triggers` (cumulative count), `local_vi_ba_mirrors_into_imu_motion_model` (subset that actually pushed state into the IMU model), and `last_mirrored_{velocity_world,bias_gyro,bias_acc}` (the most recent refined values). **Empirical finding (the wiring is functional but blocked upstream): on MH_01_easy with the recommended Phase-7 config (`--max-pose-jump-meters 0.2`), only 1 keyframe registers in 400 frames (tracking-success 1 %), VI-init never promotes, local-VI-BA never triggers, mirror never fires. Loosening the gate enables the chain — gate-sweep at `--max-pose-jump-meters ∈ {0.5, 1.0, 1.5, 2.0}` (with `--vi-init-gyro-std-limit 0.5 --vi-init-accel-std-limit 5.0 --local-vi-ba`) registers 2 → 4 → 7 → 8 keyframes, VI-init promotes at frame 64-90 across all four, local-VI-BA triggers 0 / 1 / 3 / 1 times. At `gate=1.5 m` (the sweet spot for trigger count) the mirror writes physically plausible drone-takeoff velocities like `(0.57, 0.08, 0.49) m/s`. BUT the loose-gate visual quality regresses rigid ATE catastrophically (Phase-7 baseline 0.041 m → 0.321 m at `gate=1.5 m`), so the mirror's empirical benefit on real EuRoC is currently negative — the upstream blocker is the visual front-end's PnP noise at the gate widths that VI-BA needs.** The mirror wiring is ready for a future iteration that improves the tracker's per-frame inlier ratio (deeper descriptors, finer correspondence filtering, motion-VI-init seed, …); shipping the plumbing now means the moment the visual quality clears the threshold, the (v, b) refresh activates with one config flag (`--local-vi-ba`) instead of a new code change.

- `docs/motion_based_vi_alignment.md` gains a "Phase-9 follow-up validation — local-VI-BA → IMU motion model state mirror" section recording the diagnostic motivation (Phase-8 left two prerequisite hooks unwired: the model has `set_velocity_world` / `set_biases` for downstream BA refresh, but nothing called them on real EuRoC), the implementation (a single demo-side fan-out from `OnlineSlamResult.local_vi_ba` into `slam.local_vi_ba_state.keyframe_state[latest_kf]`, then `slam.tracker.motion_model_mut().mirror_vi_ba_state(v, b_g, b_a)`), the gate-sweep table (`gate ∈ {0.2, 0.5, 1.0, 1.5, 2.0}` × keyframe / trigger / mirror / velocity / ATE columns), the diagnosed structural tension (the gate width VI-BA needs to trigger is the same gate width that lets enough PnP outliers through to dominate the trajectory), the path forward (improve the visual front-end so the gate can stay at 0.2 m with enough keyframes registering to support VI-init / VI-BA), and the explicit non-goal (this iteration is not the place to chase ATE improvement — the wiring is the deliverable, the visual-quality unblock is upstream work).

- `ImuPredictiveMotionModelConfig.body_to_sensor: SE3` (default `SE3::identity()`) — plumbs the cam0 ↔ IMU `T_BS` (body-from-sensor) rigid extrinsic into the strapdown predictor. `predict_pose` now (1) converts the input camera pose to a body pose `T_bw = T_bs · T_cw`, (2) integrates gyro/accel in the body frame, (3) converts the integrated body pose back to a camera pose `T_cw_new = T_cb · T_bw_new`. With `body_to_sensor = identity` the math reduces bit-for-bit to the original `body == camera` approximation, so existing wire-ups don't regress. Companion `ImuPredictiveMotionModel::update_velocity_from_camera_pose_difference(prev, curr, dt_seconds)` method that finite-differences the body centres of two successive camera poses to refresh `velocity_world` — the documented hook for callers that don't run a downstream VI-BA (which would otherwise leave the integrator's initial velocity pinned at zero forever, systematically under-predicting motion). Five new unit tests in `pipelines/tracking/src/lib.rs::imu_predictive_motion_tests` cover the new code paths: `imu_predictive_motion_t_bs_offset_preserves_extrinsic_under_translation` (body translates 0.5 m, camera 0.1 m ahead in body-x, predicted camera centre = 0.6 m), `imu_predictive_motion_t_bs_offset_propagates_rotation_with_lever_arm` (body rotates +90° about world-z with a 0.1 m body-x lever arm, predicted camera centre traces from (0.1, 0, 0) to (0, 0.1, 0)), `imu_predictive_motion_update_velocity_from_camera_pose_diff_recovers_body_velocity` (identity-extrinsic finite-diff recovers (1, 0, 0) from a 0.5 m / 0.5 s pose pair), `imu_predictive_motion_update_velocity_with_lever_arm_uses_body_centre` (the velocity-update strips the lever-arm and reports the body velocity, not the camera's), `imu_predictive_motion_update_velocity_rejects_nonpositive_dt` (zero / negative / NaN `dt` leaves `velocity_world` unchanged).

- `examples/euroc_online_slam_vi_image_demo` gains the `--imu-extrinsic-from-cam0` CLI flag (default off) that atomically (1) wires cam0's published `T_BS` into `ImuPredictiveMotionModelConfig.body_to_sensor` instead of identity, AND (2) enables the per-frame body-velocity finite-difference refresh of the IMU motion model from two successive successful poses (via the new `DemoMotionModel::update_velocity_from_pose_diff` dispatch). The flag is opt-in because of an empirically observed three-sequence rigid-ATE regression on EuRoC (first 400 frames, `--motion-model imu --pnp-pose-prior-warm-start --max-pose-jump-meters 0.2`): MH_01 rigid 0.041 → 0.068 m, V1 rigid 0.022 → 0.048 m, V2 rigid 0.025 → 0.087 m. The mechanism: with the geometrically correct extrinsic and a non-zero velocity prior, more IMU predictions land inside the 0.2 m gate (MH_01 gate failures 165 → 162; V1 119 → 78; V2 144 → 86), but the finite-difference velocity carries the visual tracker's per-frame noise (~1–2 cm jitter / 50 ms ≈ 20–40 cm/s of velocity noise), and that noisy prior feeds the PnP warm-start to a worse local minimum. The trade-off is a **dramatic metric-scale recovery: MH_01 similarity scale 1.112 → 1.000723 (essentially perfect — first time on the bench)**; V2 scale 1.216 → 0.947 (closer to metric from the over-scaled side); V1 scale 0.246 → 0.827 (much closer to metric from the under-scaled outlier-shrunk side). Default (`imu_extrinsic_from_cam0=false`) preserves Phase-7 bit-identical behaviour — verified by re-running MH_01 with the flag off (rigid 0.0411 m, scale 1.112503, gate failures 165, all matching the pre-flag Phase-7 baseline to the last decimal). One new `summary.txt` field lands: `imu_extrinsic_from_cam0`.

- `docs/motion_based_vi_alignment.md` gains a "Phase-8 follow-up validation — T_BS extrinsic plumbing + finite-difference velocity update" section recording the diagnostic motivation (Phase-7 left two open assumptions: (a) `body == camera` and (b) `velocity_world = 0` forever, both made invisible by the body-as-camera approximation), the two-part implementation (`body_to_sensor` plumbed through `predict_pose`'s frame conversions + a finite-difference velocity refresh hook), the three-sequence A/B grid (Phase-7 baseline vs Phase-8a "T_BS only, no velocity update" vs Phase-8b "T_BS + velocity update"), the per-mode findings (Phase-8a — geometrically correct but `v=0` damps motion → fewer gate rejections but noisy PnP slips through; Phase-8b — perfect metric scale recovery on MH_01 (1.001) at the cost of per-frame rigid ATE because finite-diff velocity amplifies visual jitter), the empirically validated recommendation (keep the flag off by default until a downstream VI-BA can refine the velocity instead — finite-difference of two noisy poses is not a noise-free substitute), and the remaining gaps (wire `ImuPredictiveMotionModel::set_velocity_world` to local-VI-BA's refined per-keyframe velocity once VI-BA is exercised on real EuRoC, mirror per-keyframe bias updates into the motion model's `set_biases`, and propagate the cam0 `T_BS` change-of-frame correctly inside `OnlineSlamPipeline::push_imu_measurement` if/when the pre-integrator gets refactored to consume body-frame samples).

- `Tracker::motion_model()` / `Tracker::motion_model_mut()` accessors — read / mut access to the configured motion model. The mut accessor is the documented hook for feeding out-of-band inputs the per-frame `track_frame*` path does not surface, primarily raw IMU samples into [`ImuPredictiveMotionModel::push_imu_measurement`]. The read accessor is informational. Both are zero-cost (return `&self.motion_model` / `&mut self.motion_model`) so no existing caller pays the abstraction.

- `examples/euroc_online_slam_vi_image_demo` extends `--motion-model` with a new `imu` choice that selects `ImuPredictiveMotionModel` (the IMU-strapdown predictor that integrates the inter-frame body-frame gyro/accel samples through Forster's strapdown step to predict the next camera pose from the previous one). The demo's existing per-frame IMU sample loop now also forwards each `(gyro, accel, dt)` sample into the tracker's motion model via `slam.tracker.motion_model_mut().push_imu_measurement(...)`, so the IMU stream feeds both the pipeline's pre-integrator AND the motion model's prediction state. The `DemoMotionModel` enum gains an `ImuPredictive` variant whose `predict_pose` / `observe` / `reset` dispatch into the wrapped model, and an internal `push_imu_measurement` method that no-ops for the non-IMU variants. The IMU motion model is constructed from `--gravity` (the existing gravity-world flag, defaulting to `(0, 0, -9.81)` for EuRoC) with zero gyro/accel biases — the cam0 ↔ IMU `T_BS` extrinsic is approximated as identity for this wire-up (cam0 sits ~0.1 m from the IMU on EuRoC, so the body-vs-camera frame mismatch is bounded over a single ~50 ms inter-frame integration window; a follow-up will plumb `T_BS` through for tighter integration). The summary.txt `motion_model` field grows the `"imu"` value. Empirically validated on three EuRoC sequences (first 400 frames, `--pnp-pose-prior-warm-start --max-pose-jump-meters 0.2`, comparing `velocity` vs. `imu` motion models): **MH_01_easy rigid ATE 0.149 → 0.041 m (3.6× improvement), similarity scale 0.390 → 1.112 (the drone-takeoff sequence now recovers metric scale within ~11 % vs. ~60 % under-scaled with constant-velocity), orientation RMSE 1.91 → 0.72 deg (2.7× tighter)** — the velocity-model breakdown documented in Phase-6 (constant-velocity does not predict the accelerating-takeoff motion) is resolved by an IMU-priored prior. V1_01_easy rigid ATE 0.034 → 0.022 m (1.5× tighter, slow indoor walk where constant-velocity is already close to truth); V2_01_easy rigid ATE 0.034 → 0.025 m (1.4× tighter). On the slow V1/V2 sequences the IMU prior modestly degrades the metric scale recovery (V1: 0.611 → 0.246; V2: 1.020 → 1.216) — the gyro/accel noise integrated over many tiny inter-frame windows adds a small drift that pulls the trajectory slightly off the true scale; constant-velocity remains the right choice on quasi-static motion. **Recommended config: `--motion-model imu --pnp-pose-prior-warm-start --max-pose-jump-meters 0.2` on EuRoC sequences with non-trivial linear / angular acceleration (MH_*), `--motion-model velocity` everywhere else.**

- `docs/motion_based_vi_alignment.md` gains a "Phase-7 follow-up validation — IMU-priored motion model wire-up" section recording the diagnostic motivation (Phase-6 left MH_01 as the sole outlier among the three EuRoC sequences validated, because the drone-takeoff acceleration violated the constant-velocity assumption used by the warm-start prior), the implementation strategy (the existing `ImuPredictiveMotionModel` had no demo wire-up because the demo's IMU stream only fed the pipeline's pre-integrator — Phase-7 adds the `Tracker::motion_model_mut` accessor and a per-IMU-sample fan-out in the demo so the motion model gets the same stream), the full A/B grid (`velocity` vs `imu` × 3 sequences, all with warm-start + gate=0.2), the per-scene findings (MH_01 rigid 3.6× tighter, V1/V2 rigid 1.4–1.5× tighter), the explanation of the V1/V2 metric-scale trade-off (IMU noise over many quasi-static frames pulls scale off slightly, where constant-velocity has no such drift), the new recommended config split by motion regime (IMU for accelerating sequences, velocity for quasi-static), and the remaining gaps (cam0 ↔ IMU `T_BS` extrinsic plumbing — currently approximated as identity; bias estimation hook from local VI-BA's per-keyframe state once that path is exercised on real EuRoC).

- `PnPRansac` gains the `estimate_with_pose_prior_and_weights(correspondences, camera, pose_prior, weights)` path on the existing `RobustPoseEstimator` trait (default impl ignores the prior and delegates to the weighted / unweighted variant, so existing implementors don't need to change). `PnPRansac` overrides this: before iterating, it scores the supplied pose prior against the correspondence set and seeds `best_pose` / `best_inliers` / `best_error` with that score; random samples must beat the prior's inlier count to win. This is the ORB-SLAM3 motion-only BA warm-start pattern — a well-aligned prior short-circuits RANSAC on hard scenes (faster motion where standard PnP would diverge under a strict pose-jump gate) while a misaligned prior gracefully degrades to the standard random search (verified by `pnp_ransac_pose_prior_with_zero_inliers_falls_back_to_random_search`). Two new tests in `crates/vision/src/ransac/mod.rs::tests`: `pnp_ransac_pose_prior_warm_start_recovers_when_random_search_fails` pins the warm-start contract with a 1-iteration budget against a 7-inliers-and-5-outliers scene where the no-prior estimate returns `None` at the fixed seed but the identity-prior estimate converges to all 7 inliers; the fall-back test confirms that a 1000 m off prior leaves the random-search result bit-identical.

- `TrackingConfig.pnp_pose_prior_warm_start: bool` (default `false`) — opt-in flag that flips the tracker's localization path from "consume the motion-model pose prior only as a candidate-radius filter" to "ALSO feed it into PnP RANSAC as a warm-start hypothesis." New `FrameLocalizer::localize_frame_with_pose_prior_warm_start_and_descriptor_store` method threads the prior through `LocalizationPipeline::localize_with_candidate_selector_and_descriptor_store_and_pose_prior` → `LocalizationPipeline::run_localization` (the latter takes a new `pose_prior: Option<&Pose>` parameter) → `PnPRansac::estimate_with_pose_prior_and_weights`. The existing prior-as-radius-filter path is preserved bit-for-bit when the flag is `false`.

- `examples/euroc_online_slam_vi_image_demo` gains two new CLI flags: `--pnp-pose-prior-warm-start` (wires `TrackingConfig.pnp_pose_prior_warm_start = true`, off by default) and `--motion-model {pose, velocity}` (selects between `ConstantPoseMotionModel` (the existing default, preserves Phase-5 reproducibility) and `ConstantVelocityMotionModel` (extrapolates the last 2 successful poses to predict where the body will be this frame)). The flags compose: the warm-start is only as good as the prior is predictive — `pose + warm-start` on a moving body just freezes the trajectory at the last pose (verified on V1_01_easy: rigid ATE 0.046 → 0.119 m, orientation max 147°), `velocity + warm-start` is the right combination. A local `DemoMotionModel` enum wrapper inside the demo dispatches between the two stock models at runtime so a single concrete `Tracker<P, DemoMotionModel>` type carries through the rest of the demo. Two new `summary.txt` fields land: `pnp_pose_prior_warm_start` and `motion_model`. Empirically validated on three EuRoC sequences (first 400 frames) starting from the Phase-4+5 best config (`--motion-vi-init --local-vi-ba --covisibility-local-map-max-keyframes 10 --covisibility-local-map-min-shared 15 --max-pose-jump-meters 0.2`): `velocity + warm-start` collapses **V1_01_easy similarity ATE 0.040 → 0.012 m (3.3×)** with rigid ATE 0.046 → 0.034 m (-26 %); **V2_01_easy similarity scale 0.831 → 1.020 — the first configuration on the bench where the trajectory comes out in true metric units (≤ 2 % error from the metric truth)** with rigid ATE 0.036 → 0.034 m; **MH_01_easy similarity ATE 0.117 → 0.057 m (halved)** but rigid ATE 0.126 → 0.149 m (+18 %) and tracking-success collapses to 1.8 %, because the drone-takeoff motion violates the constant-velocity prediction (the body is accelerating, not coasting) — MH_01 needs `ImuPredictiveMotionModel` for an even tighter prior, which is the documented next iteration. Loosening MH_01's gate to 0.5 m makes things worse (rigid 0.149 → 0.468 m), confirming the threshold is right and the gap is in the motion model. The Phase-5 hypothesis that "residual sim-ATE gap on V2 was metric-scale recovery work" turns out to be wrong: a sharper PnP prior was sufficient to pin the scale tightly; the metric-scale recovery (Viba2's monocular path) is not the next lever for stereo-bootstrapped sequences.

- `docs/motion_based_vi_alignment.md` gains a "Phase-6 follow-up validation — motion-priored PnP warm start" section recording the diagnostic motivation (Phase-5 left two coupled gaps — sparse trajectories and sub-metric similarity scale), the two-part implementation (the `PnPRansac::estimate_with_pose_prior_and_weights` path and the demo's `--pnp-pose-prior-warm-start` / `--motion-model` flags), the full 10-cell A/B grid showing the four-knob interaction (`pose` vs `velocity` × `warm-start` on / off), the per-scene findings (V1_01 sim-ATE 3.3× tighter, V2_01 metric-scale recovered, MH_01 needs an IMU-priored motion model), the explanation of why both knobs are required (`pose + warm-start` freezes the trajectory, `velocity` alone collapses keyframe coverage), the new recommended best config (`--motion-model velocity --pnp-pose-prior-warm-start --max-pose-jump-meters 0.2`), and the remaining gaps (MH_01 IMU-priored PnP, V1 metric-scale closure via richer local-VI-BA or stereo re-triangulation).

- `docs/motion_based_vi_alignment.md` gains a "Phase-5 generalization — pose-jump gate on other EuRoC sequences" section recording the cross-sequence A/B grid (V1_01_easy / MH_01_easy / V2_01_easy, first 400 frames, gate=0.2 m vs. baseline). The recommended `--max-pose-jump-meters 0.2` setting generalises: rigid ATE drops 23× on MH_01 (2.934 → 0.126 m) and 17 000× on V2_01 (608.333 → 0.036 m). The outlier-shrunk-similarity-scale signature (`ate_similarity_scale = 0.0099` on MH_01, `0.000051` on V2_01) confirms the V1_01 Procrustes-shrinkage pattern is universal, not V1-specific. V2_01 with `gate=0.2` is now the best ATE result on the bench (rigid 0.036 m, sim 0.034 m, scale 0.831 — even sharper than V1_01's 0.046 m). MH_01 is the worst case (rigid 0.126 m) because the drone takeoff pushes valid inter-frame translations toward the 0.2 m threshold; loosening to `gate=0.5 m` does NOT recover quality (rigid 0.192 m, scale 0.229 — outliers leak back in), so MH_01 needs the companion fix (motion-priored PnP warm start), not a looser gate. `vi_init_succeeded_frame=None` on every sequence — static-VI init trips `AccelNoiseTooHigh` on MH_01 / V2_01 and motion-VI init never accumulates enough gated keyframes; both pathways remain blocked on the keyframe-registration-loop fix.

- `examples/euroc_online_slam_vi_image_demo` gains the `--max-pose-jump-meters <m>` CLI flag wiring the existing `TrackingConfig.max_pose_prior_translation_error` knob (which has been in the tracker since the original quality-gate work but was never opted into by the EuRoC image demo) end-to-end on the EuRoC image path. With the default `ConstantPoseMotionModel`, the motion-model pose prior IS the last successful pose, so the gate becomes a hard per-frame camera-centre translation cap — any PnP result whose camera centre exceeds `<m>` away from the previously accepted pose is rejected as a catastrophic outlier (`tracking_failure_reason = PosePriorTranslationErrorExceeded`). The demo also fixes a side-channel where rejected frames still leaked their bad PnP pose into the trajectory CSV and the ATE summation: the tracker's `rejected_by_quality_gate()` only flips `localization.success = false` but leaves `localization.pose` populated. The demo now treats `success == false` as "no estimate this frame" so the rejected pose appears as empty CSV columns and is excluded from `estimated_positions`. Two new `summary.txt` fields land: `max_pose_jump_meters` (the configured threshold) and `tracking_quality_gate_failures` (the cumulative `TrackingStats::tracking_quality_gate_failure_count` so a Phase-5 A/B is one `diff summary.txt` away). Empirically validated on V1_01_easy (first 400 frames) starting from the Phase-4 best config (`--motion-vi-init --local-vi-ba --covisibility-local-map-max-keyframes 10 --covisibility-local-map-min-shared 15`): sweep over `--max-pose-jump-meters ∈ {5.0, 2.0, 1.0, 0.5, 0.2}` collapses rigid ATE 7.421 m → 0.046 m (160× improvement) and similarity-aligned ATE 0.535 m → 0.040 m (13× improvement) at `0.2 m`; the previously documented "residual sim-ATE 0.53 m needs metric-scale recovery" diagnosis was wrong — the `ate_similarity_scale = 0.007` was an outlier-shrinkage artefact of Procrustes alignment, not a metric-scale problem (the trajectory carries metric scale from the stereo bootstrap; sporadic catastrophic PnP teleports >9 m per 50 ms cam0 step were dragging the rigid ATE up, and Procrustes was compensating by scaling the estimate down 140×). Trade-off: at the 0.2 m best-config the 80 % rejection rate means a sparse trajectory and the VI / motion-VI init paths never trigger (insufficient keyframes register through the gate); for downstream stages to remain useful the threshold needs to scale with `(expected_velocity_mps × cam0_period_s) + margin`, and on faster sequences the Phase-4 covisibility filter becomes load-bearing again. With the gate at 0.2 m the covisibility filter falls back to the full-map descriptor pool on every frame (`covisibility_local_map_mean_size = 1500.00` vs. `630.89` at Phase-4 best) and contributes nothing to the headline number; running the same config with the cov flags omitted produces bit-identical ATE. The covisibility filter is now defence in depth; the pose-jump gate is the dominant intervention.

- `docs/motion_based_vi_alignment.md` gains a "Phase-5 follow-up validation — pose-jump rejection gate" section documenting the root-cause diagnostic walk-through (frame 31 → 32 → 33 teleport on V1_01_easy and the 84-of-158 frames with >2 m frame-to-frame translation under the no-gate baseline), the two-part fix (CLI flag + the demo's rejected-frame side-channel), the full `--max-pose-jump-meters` sweep result table (5 thresholds × 6 metrics), the recommended best config, the trade-off and threshold-tuning guidance, the interaction with the Phase-4 covisibility filter (it becomes load-irrelevant once the gate is strict), and the remaining gaps that future iterations might pick up (motion-priored PnP warm start, tighter keyframe-registration loop so the VI-refinement layers can carry weight under a strict gate).

- `TrackingConfig.covisibility_local_map: Option<CovisibilityLocalMapConfig>` (default `None`) — tracker-side covisibility-graph guided landmark selection. When set, `Tracker::track_frame_*` only matches the current frame's descriptors against landmarks observed by the **reference keyframe** (`last_successful_frame_id`, falling back to the nearest past keyframe in the map) plus its top-`max_keyframes` co-visible neighbours (ranked by descending shared-landmark count, ≥ `min_shared_landmarks`). The filter is skipped when the tracker is not in `TrackingState::Tracking`, when no reference keyframe is available, or when the resulting local-map descriptor count drops below `min_local_map_landmarks`, so it is safe to leave enabled across uninitialized / lost windows. `TrackingResult` gains a `covisibility_local_map_size: Option<usize>` field exposing the per-frame restricted-store size (`None` when the filter fell back) and `TrackingStats` gains a `covisibility_local_map_used_count` field. Empirically validated on V1_01_easy (first 400 frames), enabling the covisibility filter alone collapses rigid ATE **159.21 m → 7.42 m (21.5× improvement)**, with the mean restricted-store size of 631 landmarks vs. the 1500-landmark full map (a 58 % reduction in match candidates). The covisibility filter is now the single strongest tracker-side intervention on this prefix and stacks favourably with the static + motion VI initialisers; counter-intuitively it stacks **worse** with the Phase-3 bias-freeze fallback (cov-only 7.42 m vs. cov + Phase-3 10.90 m) because the freeze trigger is calibrated for the high-drift regime — recommended default is to leave the covisibility filter on and turn the Phase-3 freeze threshold off. Six new unit tests in `pipelines/tracking/src/lib.rs::covisibility_local_map_tests` cover reference-keyframe resolution (exact-match, nearest-prior, no-prior-available), the shared-count filter, the cap pruning path, and the union-of-co-visible-keyframe-observations construction. The new `--covisibility-local-map-max-keyframes <N>` / `--covisibility-local-map-min-shared <M>` CLI flags on `examples/euroc_online_slam_vi_image_demo` wire the knob to the demo, with `covisibility_local_map_max_keyframes` / `covisibility_local_map_min_shared` / `covisibility_local_map_used_frames` / `covisibility_local_map_mean_size` fields landing on `summary.txt`. `visloc_rs::CovisibilityLocalMapConfig` is added to the facade re-export.

- `docs/motion_based_vi_alignment.md` gains a "Phase-4 follow-up validation — tracker-side drift suppression" section recording the covisibility-filter A/B grid (Phase-2 baseline 159.21 m / Phase-3 best 25.17 m / cov-only 7.42 m / cov + Phase-3 10.90 m) plus the parameter-sensitivity sweep (six `--covisibility-local-map-max-keyframes` × `--covisibility-local-map-min-shared` combinations all produce bit-identical ATE on this prefix — the reference keyframe alone already contributes the full 660-landmark local map, so the covisibility ranking is essentially a no-op on short indoor sequences where most landmarks are visible from any nearby keyframe). The validated best opt-in CLI configuration is `--motion-vi-init --local-vi-ba --covisibility-local-map-max-keyframes 10 --covisibility-local-map-min-shared 15`. Documents the remaining sim-ATE 0.53 m residual as being dominated by missing metric-scale recovery (scale factor 0.007 indicates the trajectory is in arbitrary units ~140× smaller than ground truth) — two-view triangulation refinement on the stereo seed and `Viba2Config::recover_scale` monocular-scale recovery are flagged as the natural next levers.

- `MotionBasedViInitializerConfig.max_velocity_magnitude_mps: Option<f64>` (default `None`) — post-solve sanity gate on the per-keyframe `||velocity_world||` recovered by the motion-VI inner LM solve. When `Some(v)`, the initialiser rejects any LM-converged result with at least one keyframe whose speed exceeds `v`, surfaces a new `MotionBasedViRejectionReason::VelocityOutOfRange { kf_id, magnitude_mps, limit_mps }` variant, and parks itself in `Waiting { last_rejection: Some(VelocityOutOfRange { .. }) }` so the next trigger re-runs from the same linearisation point. `None` preserves the legacy "accept any LM-converged result" behaviour. Empirically validated on V1_01_easy (first 400 frames): with the real EuRoC tracker producing ~24 m of cumulative drift, the inner motion-VI solve previously promoted unphysical per-keyframe states (velocities ±45 m/s, bias-acc up to -280 m/s²) into `local_vi_ba_state` and `imu_state`; with `--motion-vi-init-max-velocity 10` the gate catches and rejects the divergent solve. Two new unit tests in `pipelines/slam/src/vi_motion_initializer.rs::tests`: `max_velocity_magnitude_gate_rejects_when_exceeded` runs the inner solver twice (once to discover the recovered velocity, once with `Some(max/2.0)` to assert the gate fires and returns `VelocityOutOfRange`) and `max_velocity_magnitude_gate_passes_when_under_limit` confirms a generous `Some(1.0e6)` cap still admits the LM result. The new `--motion-vi-init-max-velocity <m/s>` CLI flag on `examples/euroc_online_slam_vi_image_demo` wires the knob to the demo and adds a `motion_vi_init_max_velocity_mps` field to `summary.txt`.

- `OnlineSlamLocalBaConfig.freeze_biases_when_cost_ratio_above: Option<f64>` (default `None`) — conditioning fallback on the sliding-window VI-BA. After the first BA pass returns, if `final_cost / initial_cost > threshold` the stage re-solves the window with **all per-keyframe biases gauge-frozen** at their pre-BA linearisation points and uses the second-pass result for writeback; the bias writeback into `state.keyframe_state` is suppressed regardless of what the BA arithmetic produces. `OnlineSlamLocalBaStats` gains a `bias_frozen: bool` field surfacing whether the fallback fired this trigger. `None` preserves the legacy always-update-biases behaviour. Empirical sweep on V1_01_easy (first 400 frames, with `--motion-vi-init-max-velocity 10` to gate the motion stage): `threshold ∈ {0.9, 0.5, 0.3, 0.1}` leaves rigid ATE at ~158 m (the cost-ratio sits well below those caps even when the BA's bias updates are pathological); `threshold = 0.01` triggers the fallback on essentially every trigger and collapses rigid ATE 158.24 m → **25.17 m**, within ~1 m of the local-VI-BA-disabled Phase-2 baseline of 23.72 m. The flag is opt-in because the right calibration is dataset-dependent (a stricter cap is safer when upstream tracker drift dominates). Internal refactor: `run_local_vi_ba`'s 100-line procedural BA construction is consolidated into a single `build_ba(freeze_all_biases: bool)` closure so the two passes share the same buildup and only differ in the `ba.fix_bias(kf_id)` calls for non-anchor keyframes. Two new unit tests in `pipelines/slam/src/online_slam_vi_ba.rs::tests` pin the wiring: `local_vi_ba_bias_freeze_does_not_fire_when_threshold_disabled` confirms `None` keeps `bias_frozen = false`, and `local_vi_ba_bias_freeze_fires_when_threshold_zero` confirms `Some(0.0)` always fires (when `initial_cost > 0`), seeds non-zero biases via `bias_gyro_init` / `bias_acc_init`, and asserts the post-trigger state-table values remain at their pre-BA values within `1e-9`. The new `--local-vi-ba-freeze-biases-above <ratio>` CLI flag on `examples/euroc_online_slam_vi_image_demo` wires the knob to the demo and adds a `local_vi_ba_freeze_biases_above` field to `summary.txt`.

- `docs/motion_based_vi_alignment.md` gains a "Phase-3 follow-up validation" section recording the per-knob A/B grid + the bias-freeze threshold sweep, plus the validated best opt-in CLI configuration (`--motion-vi-init --local-vi-ba --motion-vi-init-max-velocity 10 --local-vi-ba-freeze-biases-above 0.01` → rigid ATE 25.17 m). Documents the remaining gap to ORB-SLAM3 territory as being upstream of every BA stage (visual tracker drift suppression) and proposes a more sensitive trigger for the bias-freeze fallback (bias-update magnitude rather than cost ratio) as the next iteration.

- `examples/euroc_online_slam_vi_image_demo` gains a `--local-vi-ba` flag opting the demo into the existing `OnlineSlamLocalBaConfig` sliding-window stage. Off by default to preserve the baseline. The gravity gauge from `--gravity` is mirrored onto `OnlineSlamLocalBaConfig.gravity_world` so the tri-gravity validator accepts the build. A new `local_vi_ba_enabled` field lands on `summary.txt` so an A/B grid can be assembled mechanically. The flag exists to give the motion-based VI init stage's refined `(velocity, bias)` slots a downstream consumer — without local-VI-BA enabled, motion-VI's atomic mirror into `local_vi_ba_state.keyframe_state` is a no-op (state is `None`) and the stage cannot change the trajectory. `visloc_rs::OnlineSlamLocalBaConfig` is added to the facade re-export so the demo can construct the type without reaching into `visloc_slam::` directly.

- `docs/motion_based_vi_alignment.md` gains a "Real-data validation (V1_01_easy, first 400 frames)" section recording the four-cell A/B grid (`baseline` / `motion-VI only` / `local-VI-BA only` / `motion-VI + local-VI-BA`) and the diagnosis. Key findings: (a) the motion-VI wiring is exercised end-to-end on real EuRoC data — static stage fires at frame 46, motion stage's trigger fires once `min_keyframes >= 10 ∧ cumulative_translation >= 2.0 m`, inner VIBA1 LM solver runs and converges, refined slots are written into `OnlineSlamMotionViInitState.completed`; (b) motion-VI is ATE-neutral when local-VI-BA is off — refined values have no downstream consumer; (c) `--local-vi-ba` alone takes rigid ATE 23.72 m → 158.72 m because the sliding-window VI-BA's joint Gauss-Newton update can't recover from a visual pose stream that already carries ~24 m of cumulative drift over the 400-frame prefix; (d) motion-VI on top of local-VI-BA adds a small +0.5 m marginal degradation (158.72 → 159.21 m), well below the local-VI-BA-vs-baseline gap; (e) concrete numerical evidence of upstream drift: motion-VI's recovered `trigger_translation_meters` reports `3553 m` for a sequence whose ground-truth length is ~50 m. Conclusion: the motion-VI stage's design contract holds; ATE improvement is gated on upstream tracker drift suppression + local-VI-BA conditioning + post-solve sanity gates, not on shipping more VIBA variants. Three follow-up items are enumerated at the bottom of the doc for the next session to pick up.

- Three regression tests in `pipelines/slam/tests/online_slam.rs` pinning the "tracker pose vs stored map keyframe pose" invariant after `OnlineSlamPipeline::process_frame` accepts a new keyframe: `keyframe_pose_storage_matches_tracker_pose` covers the base no-IMU / no-VI-init path, `keyframe_pose_storage_matches_tracker_with_imu` covers the IMU-enabled path, and `vi_motion_init_integration::keyframe_pose_storage_matches_tracker_under_full_vi_config` covers the full IMU + static-VI-init + motion-VI-init configuration with `seed_first_keyframe_rotation: true`. The trio formalises the audit conclusion that `keyframe_from_tracking_result` + `StagedMapUpdate::apply_to` do not mutate `frame.pose` between tracker output and map storage — the only pose-rewrite site in the production pipeline is the static stage's `seed_first_keyframe_rotation` branch in `OnlineSlamPipeline::promote_vi_init_result`, which only touches the first keyframe and only rewrites rotation while preserving the camera centre. This closes the synthetic-frame pose-storage discrepancy flagged as a follow-up from the previous motion-VI work session (the observation was not reproducible under any of the three audited configurations; the original report appears to have been a misreading of debug output). Workspace test suite now at 534 passing / 0 failing.

- `examples/euroc_online_slam_vi_demo` (synthetic-feature variant) gains the same motion-based VI init wiring as the image demo, so the two demos stay symmetric — no `image-io` feature required for parity. Same four CLI flags (`--motion-vi-init`, `--motion-vi-init-min-keyframes <N>`, `--motion-vi-init-min-translation <m>`, `--motion-vi-init-recover-scale`), same defaults (off / `10` / `2.0 m` / off), same `motion_vi_init_log.txt` file + six `summary.txt` fields. Gravity is propagated from `--gravity` to all three gates (`OnlineSlamImuConfig.gravity_world`, `VisualInertialInitializerConfig.gravity_world`, `MotionBasedViInitializerConfig.gravity_world`). One additional integration test in `pipelines/slam/tests/online_slam.rs::vi_motion_init_integration::motion_vi_init_stays_waiting_until_static_seed_fires` pins the end-to-end gate contract: with motion-VI enabled but a deliberately noisy IMU stream (jitter on the dynamic-window thresholds via `max_gyro_std = max_accel_std = 1.0e-6`) so the static stage never succeeds, the motion-based stage MUST stay in `Waiting` across multiple `process_frame` calls and never surface `Succeeded`, validating that the static seed is the prerequisite the motion path mirrors. Workspace test suite now at 531 passing / 0 failing.

- `examples/euroc_online_slam_vi_image_demo` now wires the motion-based VI init stage end-to-end. Four new CLI flags surface the knobs without changing the default behaviour: `--motion-vi-init` opts the demo into the stage (off by default for backwards compatibility), `--motion-vi-init-min-keyframes <N>` and `--motion-vi-init-min-translation <m>` mirror `MotionBasedViInitializerConfig::min_keyframes` / `min_translation_meters` (defaults `10` / `2.0 m` from the design note), and `--motion-vi-init-recover-scale` flips on `Viba2Config { recover_scale: true, .. }` for a monocular sanity check (off by default since EuRoC is stereo-bootstrapped). Every `MotionViInitializationEvent` returned on `OnlineSlamResult.vi_motion_init` is appended to a new `motion_vi_init_log.txt` next to `vi_init_log.txt` and echoed to stdout (`vi_motion_init Succeeded …` / `vi_motion_init StillWaiting reason=…`). On the first successful event the demo records the recovered `scale` and `viba2_iterations_run`; these plus the first/succeeded frame indices and the final `MotionViInitializationStatus` land in `summary.txt` as five new fields (`motion_vi_init_enabled`, `motion_vi_init_first_event_frame`, `motion_vi_init_succeeded_frame`, `motion_vi_init_recovered_scale`, `motion_vi_init_viba2_iterations`, `motion_vi_init_status_final`) so an A/B comparison against the static-only baseline is one `diff summary.txt` away. Closes the last documented follow-up for the motion-based stage: the integration tests in `pipelines/slam/tests/online_slam.rs::vi_motion_init_integration` validate the contract on synthetic data, and the demo now exercises the same wiring against real EuRoC imagery. The gravity gauge is propagated from `--gravity` into both `OnlineSlamImuConfig.gravity_world` and `MotionBasedViInitializerConfig.gravity_world` so the pipeline's tri-gravity validator (`MotionGravityMismatch`) accepts the build.

- `visloc_slam::run_viba2_inertial_with_scale` + `Viba2Config` + `Viba2Stats` + `estimate_scale_from_factors` — the VIBA2 outer-loop wrapper that ships the monocular scale-recovery flavour on top of the existing VIBA1 inertial-only solver. Strategy is alternating minimisation: at iteration `k`, the wrapper builds a rescaled copy of every IMU factor (`Δp ← Δp/s_k`, `Δv ← Δv/s_k`, `gravity_world ← gravity/s_k`) so the visual-up-to-scale state and the rescaled metric integrals share a frame, runs the existing `run_inertial_only_vi_ba` against the rescaled factors, and re-estimates `s_{k+1}` via a closed-form 1-D least squares against the post-solve state and the **original** (un-rescaled) factors. The estimator is `s = (Σ aᵀb) / (Σ aᵀa)` with `a = R_iᵀ · (p_j - p_i - v_i·Δt)` and `b = Δp + R_iᵀ · 0.5·g·Δt²` summed across in-window factors — derivable from the Forster position residual under `p_metric = s·p_visual`. `Viba2Config { initial_scale, recover_scale, max_outer_iterations, scale_tolerance, ba_config }` exposes the knobs; the default `recover_scale = false` reduces the wrapper to a single VIBA1 call at `initial_scale = 1.0` and preserves stereo / RGB-D behaviour. The outer loop terminates on `|s_{k+1} - s_k| / s_k < scale_tolerance` or `max_outer_iterations`. `estimate_scale_from_factors` returns `None` when the kinematic denominator `Σ aᵀa < 1e-12` (i.e. the body has not moved enough to identify scale); the wrapper freezes the current `s` and exits in that case rather than dividing by near-zero. `MotionBasedViInitializerConfig.viba2: Option<Viba2Config>` (default `None`) flips the inner solver inside `MotionBasedViInitializer::try_initialize` from VIBA1 → VIBA2 — when `None`, the existing VIBA1 path runs unchanged and the result reports `scale = 1.0` with `scale_history = Vec::new()` and `viba2_iterations_run = 0`; when `Some`, the VIBA2 wrapper drives the solve and the result carries the recovered scale plus the full per-iteration scale history for diagnostics. Four new unit tests in `pipelines/slam/src/online_slam_vi_ba.rs::tests` pin the VIBA2 contract: (a) `viba2_with_recover_scale_false_reduces_to_viba1` — `recover_scale = false` runs exactly one inner solve, `scale_history.len() = 1`, `outer_iterations_run = 1`, `scale = 1.0`; (b) `viba2_unit_scale_synthetic_stream_recovers_near_one` — a four-keyframe constant-velocity stream at unit ground-truth scale starting from `initial_scale = 1.0` keeps `scale` finite and bounded near `1.0` across the outer loop; (c) `estimate_scale_from_factors_recovers_known_scale` — synthetic two-keyframe scene with ground-truth scale `2.5` (visual `p_j = (1/2.5, 0, 0)`, metric `Δp = (1, -4.905, 0)` under `y-down` gravity) recovers `s ≈ 2.5` within 5 %; (d) `estimate_scale_returns_none_on_degenerate_kinematics` — coincident keyframes + zero velocity produce `a = 0` for every factor and the estimator returns `None` instead of dividing by zero. Two new tests in `pipelines/slam/src/vi_motion_initializer.rs::tests` pin the handoff: `viba2_handoff_runs_when_configured_and_reports_scale_history` confirms the VIBA2 path runs and surfaces `viba2_iterations_run >= 1` + non-empty `scale_history`, and `no_viba2_config_keeps_scale_history_empty` confirms the VIBA1-only path is preserved bit-for-bit. The design note's status banner is updated from "minimal first-cut shipped" → "shipped" and the OSS-parity table in `docs/vi_initialization_integration.md` flips the "Dynamic VI alignment" column for `visloc-rs` from `◐ partial` to `○ shipped`.

- `visloc_slam::OnlineSlamMotionViInitConfig` + pipeline glue + `MotionViInitializationEvent` + `MotionViInitializationStatus` — wires `MotionBasedViInitializer` into `OnlineSlamPipeline`'s frame loop, mirroring the existing `OnlineSlamViInitConfig` pattern for the static stage. The new `OnlineSlamConfig.vi_motion_init: Option<OnlineSlamMotionViInitConfig>` (default `None`) opts the pipeline into the motion-based stage; requires `imu = Some(_)` AND `vi_init = Some(_)` (rejected at `OnlineSlamPipeline::new` via three new variants on `OnlineSlamConfigError`: `MotionViInitRequiresImu`, `MotionViInitRequiresStaticViInit`, `MotionGravityMismatch`). The config carries the inner `MotionBasedViInitializerConfig`, two atomic-promotion toggles (`mirror_into_local_vi_ba: bool` default `true`, `mirror_into_imu_state: bool` default `true`), and `max_buffered_factors: usize` (default `64`, `0` disables eviction) to bound the rolling-history memory. On every new keyframe the pipeline registers the keyframe's world-frame camera centre via `MotionBasedViInitializer::register_keyframe` and banks the freshly-staged IMU factor into the new `OnlineSlamMotionViInitState::factor_history` ring buffer; once both trigger gates fire the pipeline calls `try_initialize` against the banked factors. The motion-based stage is **gated on the static stage having completed** — the inner `try_initialize` consumes the static seed `VisualInertialInitializationResult` as its linearisation point — so until `vi_init_state.completed = Some(...)` the motion-based step is a no-op. On success the pipeline atomically (a) mirrors the refined per-keyframe `(velocity_world, bias_gyro, bias_acc)` into `local_vi_ba_state.keyframe_state` so the local-VI-BA trigger restarts from the refined linearisation point, and (b) mirrors the latest refined biases onto `imu_state.config.{bias_*}` + resets the running pre-integrator's bias linearisation + mirrors `local_vi_ba_state.config.bias_*_init` (so subsequent keyframe slots inherit the new linearisation), then marks `vi_motion_init_state.completed = Some(...)`. `OnlineSlamResult.vi_motion_init: Option<MotionViInitializationEvent>` exposes the state-transition (`Succeeded { result }` / `StillWaiting { reason }`); durable state is on the new `OnlineSlamPipeline::motion_vi_initialization_status() -> MotionViInitializationStatus { Disabled, Waiting { keyframes_observed, cumulative_translation_meters, buffered_factor_count, last_rejection }, Initialised { result } }` accessor. The `vi_motion_init_state` field on `OnlineSlamPipeline` is deliberately private — same invariant as `vi_init_state` (writes cross-cut `local_vi_ba_state` / `imu_state` / `map.keyframes`). `reset_sequence_state` now also resets `vi_motion_init_state` so the next sequence re-arms the motion-based stage. Six new tests in `pipelines/slam/tests/online_slam.rs::vi_motion_init_integration` pin the integration contract: (a) `validate_rejects_motion_init_without_imu`, (b) `validate_rejects_motion_init_without_static_vi_init`, (c) `validate_rejects_motion_init_gravity_mismatch`, (d) `snapshot_is_disabled_when_motion_init_is_none` — the disabled-by-default invariant, (e) `snapshot_reports_waiting_when_motion_init_enabled_but_no_keyframes_yet` — the empty-waiting snapshot reports `keyframes_observed = 0, buffered_factor_count = 0, last_rejection = None`, (f) `reset_clears_motion_init_state` — after `reset_sequence_state` the snapshot flips back to `Waiting { keyframes_observed: 0, buffered_factor_count: 0, .. }`. Four pre-existing test files (`pipelines/slam/tests/online_slam.rs` ×3, `examples/euroc_online_slam_vi_demo.rs`, `examples/euroc_online_slam_vi_image_demo.rs`) needed `vi_motion_init: None,` added to their explicit `OnlineSlamConfig` literals to compile under the new field; the test sites using `..OnlineSlamConfig::default()` are unaffected.

- `visloc_slam::MotionBasedViInitializer` + `run_inertial_only_vi_ba` — first cut (VIBA1, stereo / known-scale path) of the motion-based VI initialiser scoped in `docs/motion_based_vi_alignment.md`. `MotionBasedViInitializer` is a standalone state machine that accumulates camera-centre snapshots via `register_keyframe(keyframe_id, world_center)` and fires VIBA1 once `keyframes_observed >= min_keyframes` (default `10`) AND `cumulative_translation_meters >= min_translation_meters` (default `2.0`) — exactly the design note's recommended defaults. The new `MotionBasedViInitializerConfig { min_keyframes, min_translation_meters, gravity_world, ba_config }`, plus the `MotionBasedViInitializationResult { keyframe_states, keyframe_ids, imu_factors_used, scale, trigger_translation_meters, ba_result }`, `MotionBasedViRejectionReason { InsufficientKeyframes, InsufficientTranslation, NoUsableImuFactors, MissingKeyframeData, SolverFailed }`, and `MotionBasedViInitializationStatus { Waiting, Initialised }` types are surfaced through the `visloc_rs` facade. The solver itself is a thin wrapper around the existing `BundleAdjustment::optimize` path: poses + velocities + biases are slotted, the first keyframe in the sorted set is gauge-fixed (pose + velocity + bias), the supplied `ImuPreintegrationFactor`s are added, and **no landmarks / visual observations are registered** so the solve is pure IMU-only. To make this possible, `BundleAdjustment::optimize`'s early-bail checks were relaxed: `NoLandmarks` now only fires when visual observations are present but no landmarks are registered, and `NoObservations` becomes the joint "neither visual observations nor IMU factors" guard. The relaxation is behaviour-preserving for every existing caller (visual + IMU code paths still take the same shape; the existing `bundle_optimize_with_no_data_returns_appropriate_errors` test in `pipelines/slam/tests/bundle_adjustment.rs` is updated to add a visual observation before asserting `NoLandmarks`). Seven new unit tests in `pipelines/slam/src/vi_motion_initializer.rs::tests` pin the state machine and the inertial-only solver contract: (a) `trigger_blocked_until_keyframe_threshold_is_met` — registering 4 of `min_keyframes = 5` keyframes returns `InsufficientKeyframes { have: 4, need: 5 }`; (b) `trigger_blocked_until_translation_threshold_is_met` — 3 keyframes at positions `(0, 1, 2)` along x with `min_translation = 5.0 m` returns `InsufficientTranslation { have: 2.0, need: 5.0 }`, confirming cumulative-translation tracks the **chain length** (sum of inter-frame distances), not the endpoint distance; (c) `re_registering_same_keyframe_does_not_double_count_translation` — calling `register_keyframe(2, …)` twice silently overwrites without appending to the chain; (d) `no_usable_factor_is_reported_distinctly_from_solver_failure` — calling `try_initialize` with an empty factor slice returns `NoUsableImuFactors` rather than `SolverFailed`; (e) `successful_solve_is_cached_and_reset_re_arms` — on a synthetic three-keyframe constant-velocity stream (with the correct `Δv = -g·Δt`, `Δp = -0.5·g·Δt²` factor encoding for an identity-oriented body sensing gravity), VIBA1 fires, populates a `keyframe_states` slot per keyframe id, returns `scale = 1.0` and `trigger_translation_meters = 2.0`, the LM result has `final_cost <= initial_cost`, the state machine transitions to `Initialised`, a second call returns the cached result, and `reset` flips back to `Waiting`; (f) `stereo_stationary_replay_is_identity` — three coincident keyframes (zero motion) recover zero velocity and zero biases for every keyframe within `1e-6`, confirming the no-op path is identity; (g) `so3_identity_is_quaternion_identity` — type-import sanity. The full workspace test suite remains green (514 passing, 0 failing). What is intentionally NOT yet shipped (per the design note's "What's missing" section and the minimal-first-cut path): (1) monocular scale recovery — the `scale` field is wired but pinned at `1.0`, since the joint visual-inertial-scale BA (VIBA2) is the natural host and is a separate follow-up; (2) pipeline-level glue mirroring `OnlineSlamViInitState` — the initialiser is callable standalone (matching the static `VisualInertialInitializer`'s pattern) but is not yet wired into `OnlineSlamPipeline`'s `vi_init` path; (3) the absolute-velocity recovery that requires either visual observations or a velocity prior (the constant-velocity unit test is deliberately scoped down to "solver runs and converges" rather than "recovers ground-truth velocity"; the structural correctness check is the stationary-replay test). The design note's status banner has been updated from "planned" to "minimal first-cut shipped" with an explicit pointer to the two shipped surfaces (`run_inertial_only_vi_ba`, `MotionBasedViInitializer`) and the two follow-up surfaces (VIBA2 + pipeline glue).

- `docs/motion_based_vi_alignment.md` — design note that scopes the last "planned" entry in the OSS-parity comparison table on `docs/vi_initialization_integration.md` (motion-based / dynamic VI alignment; ORB-SLAM3's `VIBA1` / `VIBA2`). The note explains why the static-bootstrap flavour cannot recover yaw or monocular scale on its own, identifies the three building blocks `visloc-rs` already has (bias-corrected IMU preintegration, sliding-window VI-BA, static seed) and the three gaps that separate them from a landable implementation (inertial-only solver variant, shared scale state on `KeyframeImuState`, trigger / state machine), sketches a minimal first-cut path that ships only VIBA1 with `monocular = false` first, lists seven named tests the future implementation must pass, and explains why this is a multi-session deliverable rather than a single-PR change. The integration doc and the OSS comparison table both link to the new file so the planned column is now first-class documented rather than just a row in a table. The design note's status banner was updated post-implementation to mark VIBA1 as shipped (see entry above); the remaining VIBA2 work (joint visual-inertial-scale BA with monocular scale recovery) plus the pipeline-level glue remain the documented next milestones.

- `visloc_vision::stereo_bootstrap` — closes the last documented scope cut on `examples/euroc_online_slam_vi_image_demo` (limitation (ii): "the initial map is scale-anchored at a single depth"). The new module ships a general-baseline two-view DLT triangulator that does NOT assume rectification — necessary because EuRoC's cam0/cam1 are mounted with a small relative rotation in addition to the baseline translation, so the existing `visloc_vision::stereo::triangulate_stereo_pixel` (which assumes shared intrinsics + identity relative rotation + pure-translation `+x` baseline) is too narrow. The public API is `bootstrap_stereo_landmarks(left_camera, right_camera, left_to_right_se3, left_features, right_features, &StereoBootstrapConfig) -> Vec<StereoBootstrapLandmark>` plus the lower-level `triangulate_two_view_left_frame(...)` helper. Matching uses the existing `CrossCheckMatcher<BruteForceMatcher>` so the cam0 ↔ cam1 descriptor distance reuses the same code path the tracker's per-frame matcher does. Each surviving match is gated on three predicates: (a) in-front-of-camera with `Z ∈ [min_depth_meters, max_depth_meters]` in the left frame, (b) in-front-of-camera with `Z ≥ min_depth_meters` in the right frame (after transforming via `left_to_right`), and (c) reprojection error `≤ max_reprojection_error_pixels` in **both** cameras. Defaults (`matcher_ratio = 0.8`, `min_depth = 0.1 m`, `max_depth = 50 m`, `max_reprojection_error = 2.0 px`) admit any realistic indoor / outdoor scene while clamping the obvious outliers. The DLT itself is a 4×4 SVD on the standard `[I|0]` vs `[R|t]` projection setup, with the smallest right-singular vector dehomogenised; the `1e-12` scale guard rejects degenerate near-zero `w` solutions instead of producing infinity. Seven unit tests pin the contract: (a) `dlt_triangulation_round_trips_pure_translation_baseline` — an EuRoC-style `(-0.11, 0, 0)` left-to-right baseline with identity rotation recovers a known 3D point within `1e-9`, (b) `dlt_triangulation_round_trips_with_rotated_right_camera` — a `0.05 rad` yaw on the right camera (modelling EuRoC's unrectified geometry) still recovers the same point within `1e-8`, (c) `bootstrap_recovers_metric_landmarks_under_known_correspondence` — three one-hot-descriptor landmarks all triangulate to within `1e-6` of ground truth with sub-`1e-6 px` reprojection errors, (d) `bootstrap_drops_points_behind_either_camera` — a deliberately negative-disparity correspondence is dropped while the valid match passes through, (e) `bootstrap_returns_empty_on_empty_input` — both empty feature sets short-circuit, (f) `bootstrap_drops_matches_with_reprojection_above_threshold` — a 5 px perturbation on a right keypoint blows past the default 2 px gate, and (g) `bootstrap_results_are_sorted_by_left_keypoint_index` — survivors come out in ascending left-keypoint order so the caller can use the index directly as a `Vec<Option<...>>` lookup. The three new types (`StereoBootstrapConfig`, `StereoBootstrapLandmark`) plus the two public functions are re-exported through the `visloc_rs` facade.

- `examples/euroc_online_slam_vi_image_demo` now seeds the initial visual map from stereo triangulation by default. On the seed frame the demo locates the cam1 image whose timestamp matches the cam0 seed timestamp, extracts corners on cam1, undistorts them with the published cam1 distortion coefficients, computes the stereo extrinsic `cam0 → cam1 = T_BS_cam1⁻¹ ∘ T_BS_cam0` from the per-camera `T_BS` matrices, and feeds the two feature sets into `bootstrap_stereo_landmarks`. Each surviving match's 3D point is transformed from the cam0 frame into the EuRoC world frame via the GT-derived seed pose and overrides the fixed `--bootstrap-depth` back-projection for that cam0 keypoint; corners that fail to match a cam1 descriptor (or whose triangulation falls outside the configured depth / reprojection-error gates) still get the depth-only seed so the matcher input keypoint count is unchanged from the cam0 extractor. The new `--no-stereo-bootstrap` CLI flag (default `false`, i.e. stereo bootstrap is on) reproduces the depth-only seeding for A/B comparison. The `summary.txt` gains four new lines (`stereo_bootstrap`, `stereo_bootstrap_cam1_features`, `stereo_bootstrap_cam1_features_after_undistort`, `stereo_bootstrap_matches`) so downstream ATE / `slam_errors.csv` diffs can be attributed to the seeding strategy that produced them. One new test pins the wiring: `bootstrap_map_overrides_landmark_position_when_stereo_point_provided` — given `stereo_world_points = [Some(override_point), None]` against two keypoints, the first landmark sits at `override_point` and the second at the depth-back-projection of the second pixel within `1e-9`. The file header is updated to record that two scope cuts (per-keypoint distortion correction, stereo-triangulation seed map) are now closed; the remaining "planned" entry in the OSS comparison table is motion-based / dynamic VI alignment.

- `visloc_vision::distortion::RadialTangential` — OpenCV / Bouguet plumb-bob `(k₁, k₂, p₁, p₂)` distortion model. Closed-form forward `distort_normalized(undistorted)` and fixed-point inverse `undistort_normalized(distorted)` (20-iteration cap; converges to `<1e-12` residual in ~6 steps for EuRoC magnitudes), with `is_identity()` short-circuit on the zero-coefficient path and a `from_euroc_coefficients(&[f64]) -> Option<Self>` constructor that strictly accepts the 4-coefficient form so a calibration with a different shape surfaces as `None` instead of a silent mis-parse. `undistort_pixel(&Camera, distorted_pixel) -> Option<Point2<f64>>` is the convenience wrapper for the common pipeline path (normalize via `Camera::intrinsics`, undistort, project back); returns `None` only when the camera model lacks intrinsics. Seven unit tests pin the contract: (a) `identity_distortion_is_a_no_op` — zero coefficients leave both forward and inverse maps as the identity, (b) `from_euroc_coefficients_parses_four_element_vector` — EuRoC `MH_01_easy` cam0's published `[-0.28340811, 0.07395907, 0.00019359, 0.0000176187114]` round-trips through the parser, (c) `from_euroc_coefficients_rejects_other_lengths` — lengths 0 / 1 / 3 / 5 return `None`, (d) `distort_then_undistort_returns_input` — forward then inverse round-trip residual `<1e-10` across six normalized coordinates spanning `[-0.6, 0.6] × [-0.6, 0.6]` (covering EuRoC's ~80° FOV), (e) `undistort_then_distort_returns_input` — reverse direction round-trip residual `<1e-10` pins both halves of the model agree, (f) `euroc_cam0_edge_pixel_shifts_meaningfully` — under EuRoC distortion the principal-point pixel shifts `<0.01 px` while a `(10, 10)` corner pixel shifts `>5 px` (and `<200 px`), guarding against the silent bug where `undistort_pixel` collapses to the identity, and (g) `undistort_pixel_returns_none_for_unknown_camera_model` — propagates the `intrinsics()` `None` instead of panicking. The type is re-exported through the `visloc_rs` facade as `RadialTangential` for parity with `Camera`, `Frame`, etc.

- `examples/euroc_online_slam_vi_image_demo` now applies cam0's radial-tangential distortion correction by default. Each cam0 frame's extracted corners are mapped from their raw distorted pixel positions to the "ideal pinhole" pixel positions implied by the published `distortion_coefficients` on the cam0 calibration *before* being back-projected (during bootstrap) or fed into `process_frame` (per-frame), so the pipeline's pinhole-only geometry sees pixels that actually match its model. The corner *descriptors* are still extracted from the raw distorted image patches — this is the standard VIO simplification (the patch signal-to-noise is unchanged; only the position is corrected) and keeps the matcher's descriptor index in lock-step with the corrected keypoint index. The new `--no-undistort` CLI flag (default `false`, i.e. correction is applied) reproduces the pre-correction behaviour for A/B comparison. Two new tests in the example pin the wiring: (a) `undistort_feature_keypoints_is_noop_under_identity_distortion` — under `RadialTangential::IDENTITY` the helper returns a `FeatureSet` whose keypoints and descriptors match the input bit-for-bit, (b) `undistort_feature_keypoints_shifts_edge_pixels_under_euroc_distortion` — under EuRoC cam0 distortion the principal-point pixel shifts `<0.01 px` while a `(10, 10)` corner pixel shifts `>5 px`, with descriptors preserved in order. The bootstrap-feature count is reported separately as `extracted_features=… after_undistort=…` so a calibration whose `undistort_pixel` legitimately drops a keypoint (e.g. an `Unknown` camera model — which the demo never sees with EuRoC `Pinhole`) is auditable from the console. Closes documented limitation (i) in the demo's file header — only the single-depth scale anchor limitation remains, and the multi-view-stereo bootstrap on the leading static window remains the documented supersede path for a publishable EuRoC VIO ATE number. The `undistort` choice is also surfaced on `summary.txt` so downstream `slam_errors.csv` diffs can be filtered against the calibration setting that produced them.

- `VisualInertialInitializationResult::gravity_alignment_residual_deg(&UnitQuaternion<f64>) -> f64` — yaw-gauge-aware rotation residual exposed as a public method so the metric carries the same contract callers were already computing inline. The metric projects the world "up" direction (`-gravity_world / ‖gravity_world‖`) into body frame under both the recovered and a caller-supplied ground-truth rotation, then returns the angle between the two body-frame vectors. This isolates the roll / pitch component of the recovered rotation — the only part the gravity-only stationary bootstrap can observe — instead of charging the unobservable yaw mismatch against the initialiser. Degenerate `gravity_world ≈ 0` returns `0.0` rather than NaN-poisoning a diagnostic log. `examples/euroc_imu_dead_reckon_demo` is refactored to call this method instead of recomputing the metric inline, so the ablation harness and the unit tests now share the exact same formula. Two new tests close out the last deferred entry in `docs/vi_initialization_integration.md`'s test-strategy section: (a) `gravity_alignment_residual_is_invariant_under_world_yaw` is **test #12 from the design document**: two stationary streams whose ground-truth body→world rotations differ only by a 30° rotation about the world up axis both produce `gravity_alignment_residual_deg < 1e-6`, while the full quaternion residual against the yaw-rotated stream is `~30°` within `1e-4` — confirming the metric correctly identifies yaw as gauge instead of charging it against the initialiser (the precise mistake the metric is designed to avoid); (b) `gravity_alignment_residual_returns_zero_for_zero_gravity_config` pins the degenerate-config guard and cross-checks that a 90° tilt against an identity GT produces the expected 90° residual. All 14 design tests in `docs/vi_initialization_integration.md` are now shipped (test #12 here on the standalone module, test #13 already shipped via the sliding-window detector, the remaining 12 already shipped through `pipelines/slam/tests/online_slam.rs::vi_init_integration`).

- `VisualInertialInitializer` gains a **sliding detector window** + **time-weighted statistics** — the two improvements that `docs/vi_initialization_integration.md`'s "Detector windowing" section had planned and that deferred test #13 (`sliding_window_non_stationary→stationary`) was blocked on. The new `VisualInertialInitializerConfig.detector_window_seconds: f64` field (default `f64::INFINITY`) controls a trailing slice of buffered samples: when the buffer exceeds the configured width, only the trailing slice — walked from the END of the buffer backwards until cumulative `dt` first reaches `detector_window_seconds` — is evaluated for every stationary-window predicate AND the gyro / accel mean / std read-outs. `f64::INFINITY` preserves the historical "evaluate on the whole buffer" behaviour exactly (the default), so existing callers see zero behavioural change. The mean / variance formula is also switched from sample-count weighted (`Σ x_i / N`) to **time-weighted** (`μ = Σ x_i · dt_i / Σ dt_i`, `σ² = Σ (x_i - μ)² · dt_i / Σ dt_i`); for uniform `dt` the two formulas are numerically identical (all 11 pre-existing tests pass unchanged), while for irregular `dt` from a real IMU stream that drops or duplicates samples the time-weighted form is the unbiased statistic. `samples_consumed` and `duration_seconds` on `VisualInertialInitializationResult` now reflect the SLICE, not the entire buffer — so a caller running a pipeline with `detector_window_seconds = 1.0` and a 2 s buffer correctly sees `samples_consumed = ~200` not `~400`. Six new tests pin the contract: (a) `default_detector_window_seconds_is_infinity` — backward-compat anchor; (b) `time_weighted_statistics_match_unweighted_for_uniform_dt` — under uniform `dt` (e.g. clean 200 Hz EuRoC) the new formula matches the old to within `1e-12` (the property that lets the 11 existing tests pass); (c) `time_weighted_statistics_shift_mean_under_nonuniform_dt` — under non-uniform `dt` (sample A: `dt = 0.001 s`, `gyro.x = 0`; sample B: `dt = 0.999 s`, `gyro.x = 1.0`) the weighted mean is `0.999` not the sample-count midpoint `0.5`, confirming the weighting is correct; (d) `sliding_window_non_stationary_then_stationary_succeeds` — **test #13 from the design document, now unblocked** — 0.5 s of `±1 rad/s` gyro noise followed by 1.5 s of clean stationary samples with `detector_window_seconds = 1.0` succeeds with `samples_consumed = 200` (entirely from the stationary phase) and `gyro_std = 0`; `samples_seen()` still reports the full buffer (400) — the sliding window is an evaluation detail, not a buffer prune; (e) `sliding_window_rejects_when_recent_samples_are_noisy` — reverse path: 1.5 s stationary followed by 0.5 s noisy gyro with `detector_window_seconds = 0.5` rejects with `GyroNoiseTooHigh`, confirming the slice is taken from the END of the buffer (not the START); (f) `sliding_window_insufficient_when_trailing_slice_too_short` — predicates apply to the SLICE: a 400-sample (2 s) buffer with a 0.1 s window and `min_stationary_window_seconds = 0.5` is rejected as `InsufficientDuration` because the slice's `~0.1 s` is below the required `0.5 s`, even though the total buffer comfortably exceeds it. The change pattern matches OpenVINS's `StaticInitializer` (it watches the most recent N samples and triggers on a detected motion-onset that's preceded by a clean window) and unblocks the pipeline-integration use case where the pipeline starts during motion and the body becomes stationary only later in the sequence.

- `examples/euroc_online_slam_vi_image_demo` — companion to the synthetic-landmark `examples/euroc_online_slam_vi_demo` that drives `OnlineSlamPipeline` with **real cam0 pixels**. Each cam0 frame is decoded via `visloc-io`'s `read_common_image` (gated on the `image-io` feature), passed through `CornerFeatureExtractor` (defaults `max_features = 1500 / min_score = 0.02 / descriptor_radius = 5`, matching the patch sizes `online_slam_image_vo_loop_demo` settled on for real KITTI imagery), and the resulting `FeatureSet` is adapted into a `Frame { keypoints, descriptors }` fed straight into `process_frame`. The map is bootstrapped from the first cam0 frame at or after the IMU/GT-overlap seed timestamp: every extracted corner is back-projected through `Camera::normalize_pixel` at a fixed `--bootstrap-depth` (default 4.0 m, tuned for EuRoC MH indoor scenes) and the inverse of the GT-derived first world-to-camera pose lifts each point into the metric EuRoC world frame; the corner's patch descriptor is attached as the seeded landmark's descriptor so the tracker's default `BruteForceMatcher` + `PnPRansac` can match against it from frame zero. The IMU stream is consumed exactly as the synthetic variant — same `--vi-init-*` knobs, same `--gravity (0, 0, -9.81)` z-up convention, same `push_imu_measurement` drain loop interleaved on every cam0 frame — and the auto-bootstrap stage (`vi_init: Some(_)`) is enabled with the same `OnlineSlamViInitConfig { initializer, body_to_camera, seed_first_keyframe_rotation, on_persistent_rejection, max_wait_duration_seconds, max_buffered_samples }` shape. Output mirrors the synthetic variant (`slam_trajectory.csv` per-cam0-frame estimated `(px, py, pz, qw, qx, qy, qz)`, `slam_errors.csv` per-frame residual against GT, `vi_init_log.txt` every `ViInitializationEvent`, and `summary.txt` with `frames_recorded`, `tracking_success_rate`, the per-frame feature-count `(mean, min, max)`, `bootstrap_depth_meters`, `bootstrap_landmarks`, `vi_init_first_event_frame`, `vi_init_succeeded_frame`, the final `vi_initialization_status()` snapshot, and ATE rigid / similarity RMSE + max) so the synthetic and real-pixel runs can be diffed apples-to-apples on the same EuRoC sequence. Two intentional, scope-controlled limitations are documented in the file header: (i) cam0's radial-tangential distortion coefficients are **not** undistorted — pixel coordinates flow straight into the pinhole intrinsics, which inflates rigid ATE on the edge regions but keeps the scope contained; (ii) the initial map is scale-anchored at a single depth — the pipeline's local mapper still grows the map with real triangulations as new keyframes arrive, but the first few frames carry the `--bootstrap-depth` choice as the only metric anchor, so a multi-view-stereo bootstrap on the leading static window is the recommended supersede path for a publishable EuRoC VIO ATE number. Six `[cfg(test)]` unit tests pin the new helpers: (a) `se3_from_t_bs_identity_round_trip` and (b) `world_to_camera_pose_round_trip_identity_rig` carry over from the synthetic variant, (c) `back_project_pixel_to_world_at_identity_camera_returns_camera_frame_point` confirms the principal-point pixel back-projects to `(0, 0, depth)` under an identity world-to-camera pose, (d) `back_project_then_project_round_trip_identity_camera` confirms the projection / unprojection are exact inverses to `1e-9` for an arbitrary pixel at a known depth, (e) `bootstrap_map_seeds_one_landmark_per_keypoint` confirms each first-frame corner produces exactly one seeded landmark and the patch descriptor is preserved on the landmark, and (f) `frame_from_features_preserves_keypoint_and_descriptor_order` confirms the `FeatureSet → Frame` adapter does not reorder keypoints or descriptors (load-bearing because the tracker's `BruteForceMatcher` indexes both vectors in lockstep). CLI flags add `--bootstrap-depth <m>`, `--corner-max-features <N>`, `--corner-min-score <f32>`, `--corner-descriptor-radius <pixels>` on top of the synthetic variant's flags. The demo gates the entire main on `#[cfg(feature = "image-io")]`, falling back to a clear error message when the feature isn't enabled — matching `online_slam_image_vo_loop_demo`'s pattern.

- `examples/euroc_online_slam_vi_demo` — end-to-end smoke run that drives `OnlineSlamPipeline` with the real EuRoC MAV IMU stream and cam0 frame cadence, with the new auto-bootstrap stage (`vi_init: Some(_)`) enabled. The demo seeds a deterministic 5×5 landmark grid in front of the first GT camera pose and projects it into each cam0 frame under the GT-derived `R_c←w = (R_w←b · R_b←c)^T` (with cam0's `T_BS` honoured) so the tracker's visual side is anchored — what is being validated is the integration path: that the pipeline accepts the real ~200 Hz IMU rate without buffer pathologies, that the auto-bootstrap fires on a real EuRoC leading stationary window (cf. `MH_01_easy` requires `--vi-init-accel-std-limit 1.0` or similar from the dead-reckon harness's findings), and that the atomic five-step promotion preserves the camera centre across the rotation rewrite (no world-frame inversion). CLI flags mirror the dead-reckon demo where they overlap (`--euroc-dir`, `--out-dir`, `--max-frames`, `--gravity`, `--vi-init-max-wait-seconds`, `--vi-init-gyro-std-limit`, `--vi-init-accel-std-limit`); writes `slam_trajectory.csv` (per-cam0-frame estimated `(px, py, pz, qw, qx, qy, qz)`), `slam_errors.csv` (per-frame position + orientation residual against GT), `vi_init_log.txt` (every `ViInitializationEvent` with frame index and timestamp), and `summary.txt` with `frames_recorded`, `tracking_success_rate`, `vi_init_first_event_frame`, `vi_init_succeeded_frame`, the final `vi_initialization_status()` snapshot, and ATE rigid / similarity RMSE + max in the same shape the dead-reckon demo emits so the two outputs can be diffed directly. Four `[cfg(test)]` unit tests pin the SE(3) helper and frame-rendering correctness: `se3_from_t_bs_identity_round_trip` and `se3_from_t_bs_recovers_translation` cover the `T_BS` → `SE3` decomposition, `world_to_camera_pose_round_trip_identity_rig` confirms the camera centre matches the body position under identity rig, and `frame_renders_seeded_landmarks_for_identity_pose` confirms the 3×3 grid renders all 9 landmarks at a synthetic 752×480 EuRoC resolution. Pixel images are intentionally **not** decoded — the integration question this demo answers is "does the pipeline + VI init promotion work on a real EuRoC IMU + cam0 timestamp stream"; a future demo that decodes pixels and runs a real feature extractor on top of this scaffolding is the next step toward a publishable EuRoC VIO ATE number.
- `OnlineSlamPipeline` gains an optional auto-bootstrap stage that wires the standalone `VisualInertialInitializer` into the pipeline's incoming IMU stream and atomically promotes the recovered `(R_w←b, b_g, b_a)` into the running pre-integrator + first keyframe — closing the integration gap previously documented in `docs/vi_initialization_integration.md`. `OnlineSlamConfig.vi_init: Option<OnlineSlamViInitConfig>` (default `None`) opts the pipeline into the stage; requires `imu = Some(_)` (rejected at `OnlineSlamPipeline::new` via the new `OnlineSlamConfig::validate -> Result<(), OnlineSlamConfigError>` cross-field validator that also enforces `imu.gravity_world == vi_init.initializer.gravity_world`). The new `OnlineSlamViInitConfig { initializer, body_to_camera, seed_first_keyframe_rotation, on_persistent_rejection, max_wait_duration_seconds, max_buffered_samples }` carries an inner `VisualInertialInitializerConfig`, a body-to-camera SE(3) extrinsic so the IMU/camera convention difference is supplied explicitly rather than assumed identity, and twin duration / sample caps (default `5.0 s` / `2000` samples) gated on `ViInitFallback::{KeepExistingSeed, DisableImuStage}`. `push_imu_measurement` now also fans samples into the auto-bootstrap buffer while the stage is active; on the first frame where (a) `try_initialize` succeeds AND (b) a new keyframe was just registered, the pipeline atomically performs the five-step promotion: (1) reset `imu_state.preintegrator` with the new bias linearisation via `ImuPreintegrator::new_with_bias`, (2) mirror `imu_state.config.{bias_gyro, bias_acc}` (and `config.imu`) to the recovered values, (3) if `seed_first_keyframe_rotation = true`, rewrite the just-registered keyframe's `Pose.rotation` to `R_c←w = (R_w←b · R_b←c)^T` and recompute `t_c←w = -R_c←w · C_w_old` so the camera centre is preserved (the rotation-direction promotion is the single highest-risk part of the integration — writing `R_w←b` directly into world-to-camera would invert the world frame and inflate ATE by hundreds of metres before tracking failed and masked the bug), (4) seed `local_vi_ba_state.keyframe_state[first_keyframe_id]` with `velocity_world = 0, bias = (b_g, b_a)` (mirror `local_vi_ba.bias_*_init` too so subsequent keyframe slots inherit the new linearisation), and (5) mark `vi_init_state.completed = Some(result)` so the stale-factor gate is lifted from this frame onwards. The stale-factor gate is critical: until step 5, `take_pending_imu_factor()` returns `None` and inline `OnlineSlamResult.imu_factor` is suppressed even when a keyframe transition would normally stage one — those factors were built with the caller's placeholder bias linearisation and would feed an inconsistent point into downstream VI-BA; the gated factors are dropped (NOT re-integrated) and the count surfaces on `ViInitializationEvent::Succeeded { discarded_stale_factor_count, .. }` so callers can audit. `OnlineSlamResult.vi_init: Option<ViInitializationEvent>` carries the state-transition event (`Succeeded` / `StillBuffering { reason }` / `GaveUp { last_reason, fallback }`); the durable snapshot is exposed via the new `OnlineSlamPipeline::vi_initialization_status() -> ViInitializationStatus { Disabled, Buffering { samples_buffered, buffered_duration_seconds, last_rejection }, Initialised { result }, GaveUp { last_reason, fallback } }` accessor. The `vi_init_state` field on `OnlineSlamPipeline` is deliberately **private** (unlike the existing `pub imu_state` / `pub local_vi_ba_state`) because writes to `completed` cross-cut with `imu_state` / `local_vi_ba_state` / `map.keyframes` invariants — exposing it as `pub` would let callers leave the pipeline in a half-initialised state. `reset_sequence_state` now also resets `vi_init_state` so the next sequence re-arms with a fresh buffer and a fresh stale-factor gate. The new types and the error / event / status enums are re-exported through `pipelines/slam` (and downstream through the `visloc_rs` prelude). Thirteen new tests in `pipelines/slam/tests/online_slam.rs::vi_init_integration` pin the design contract, including the three load-bearing `[pose-conv]` tests: (a) `config_validate_rejects_vi_init_without_imu` — `vi_init: Some, imu: None` → `Err(ViInitRequiresImu)`, (b) `new_panics_when_vi_init_lacks_imu` — same condition at `OnlineSlamPipeline::new` panics with `"vi_init is Some"`, (c) `config_validate_rejects_gravity_mismatch` — EuRoC z-up vs KITTI y-down on the two gravity fields → `Err(GravityMismatch { .. })`, (d) `stationary_stream_success_emits_succeeded_event` — 200-sample level-body window produces `Succeeded` with zero biases, zero rotation angle, `discarded_stale_factor_count = 0`, status flips to `Initialised`, (e) `succeeded_event_does_not_re_fire_on_subsequent_frames` — the second `process_frame` carries `vi_init: None` and the status snapshot is unchanged after pushing 50 more IMU samples (the standalone module is no longer fanned-into), (f) `stale_factors_before_success_are_counted_and_discarded` — three keyframes with `min_stationary_window_seconds = 4.0 / min_samples = 800` causes the first two keyframes to gate-discard their factor and the third to emit `Succeeded { discarded_stale_factor_count: 2 }`; `take_pending_imu_factor()` returns `None` on the gated frames, (g) `[pose-conv] pose_conv_rotation_direction_uses_transpose_of_body_to_world` — body tilted 30° about world-x, body_to_camera identity → `Pose.world_to_camera.rotation * world_up == accel_body` within `1e-9`, i.e. `R_c←w == R_w←b^T` (NOT `R_w←b`), (h) `[pose-conv] pose_conv_camera_center_is_preserved_across_promotion` — pre-set camera centre `(0.4, -0.7, 0.2)` survives the rotation rewrite within `1e-9`, (i) `[pose-conv] pose_conv_respects_body_to_camera_extrinsic` — 30° tilt about world-x AND non-identity `body_to_camera = SE3(R_y(π/2))` produces `Pose.rotation == (R_w←b · R_b←c)^T` within `1e-9` angle, (j) `noisy_gyro_emits_still_buffering_with_reason` — alternating ±1 rad/s gyro emits `StillBuffering { GyroNoiseTooHigh { observed.x > 0.05 limit } }`, (k) `keep_existing_seed_fallback_leaves_imu_state_in_place` — `max_wait_duration_seconds = 0.5` triggered with noisy gyro emits `GaveUp { KeepExistingSeed }`, `imu_state` survives, status is `GaveUp`, (l) `disable_imu_stage_fallback_clears_imu_state` — same trigger with `DisableImuStage` clears `imu_state`, `local_vi_ba_state`, `config.imu`, `config.local_vi_ba`, and (m) `reset_sequence_state_rearms_vi_init` — after success, `reset_sequence_state` flips status back to `Buffering { samples_buffered: 0, .. }`. Two of the fourteen design tests are deliberately deferred: #12 constant-yaw invariance belongs to the standalone validator (not the pipeline integration), and #13 sliding-window non-stationary→stationary is gated on the planned `detector_window_seconds` knob on `VisualInertialInitializer` which is a follow-up. This closes the most actionable of the OSS-parity gaps tracked in `docs/vi_initialization_integration.md` — the pipeline now self-bootstraps the stationary-window flavour without callers having to buffer the leading IMU window out-of-band and translate the result into `OnlineSlamImuConfig` themselves. The motion-based / dynamic VI alignment flavour (ORB-SLAM3's `VIBA1` / `VIBA2`) remains a separate design note and is the "planned" column in the OSS comparison table.

- `examples/euroc_imu_dead_reckon_demo` gains a per-component seed-source ablation harness so the dead-reckon ATE drift on each EuRoC sequence can be attributed cleanly to its source instead of relying on the existing "all-from-VI vs all-from-GT" binary. Four new CLI flags — `--seed-rotation-source {gt|vi|vi_rollpitch_gt_yaw|gt_rollpitch_vi_yaw}`, `--seed-velocity-source {gt|vi|zero}`, `--seed-bias-gyro-source {gt|vi|zero}`, `--seed-bias-acc-source {gt|vi|zero}` — let the propagator be seeded with any mix of VI-init and ground-truth components on every axis. The two hybrid rotation sources decompose each input rotation `R` into a yaw `R_yaw` about world-up and a roll/pitch tilt `R_tilt` such that `R = R_tilt · R_yaw` (yaw extracted via `tilt^-1 · R` — the only order that makes `R_yaw · world_up = world_up`; the swapped order is a silent gauge-invariance bug that GPT-flagged during the design review). `vi_rollpitch_gt_yaw` takes roll/pitch from VI init but yaw from GT, isolating "did the gravity-only init's missing yaw cause the drift?"; `gt_rollpitch_vi_yaw` takes roll/pitch from GT but yaw from VI init, isolating the gravity-alignment residual. A fifth flag `--imu-cam-time-offset-ms <ms>` shifts IMU sample timestamps relative to cam0 before the per-frame ATE comparison so IMU/cam timing offsets can be swept independently. `--seed-from-vi-init` becomes a backward-compatible shorthand that sets all four sources to `vi` simultaneously; any non-`gt` source on any axis implicitly turns on `--run-vi-init`. Six new unit tests in `examples/euroc_imu_dead_reckon_demo.rs::tests` pin the rotation decomposition: (a) `split_yaw_tilt_identity` returns identity for both parts on the identity rotation, (b) `split_yaw_tilt_pure_yaw` recovers `yaw.angle() = 0.7` and `tilt.angle() = 0` from `R_z(0.7)`, (c) `split_yaw_tilt_pure_pitch` recovers `yaw.angle() = 0` and `tilt.angle() = 0.3` from `R_y(0.3)` plus a recombination check that `tilt · yaw == R_y(0.3)`, (d) `compose_yaw_tilt_mixed` picks `R_z(0.5)` yaw from one source and `R_y(0.3)` tilt from another and confirms the decomposition of the mixed result matches both, (e) `compose_yaw_tilt_roundtrip_preserves_full_rotation` mixes a non-trivial Euler rotation `(0.2, -0.4, 0.7)` with itself and confirms `compose_yaw_tilt(R, R) == R`, and (f) `vi_rollpitch_gt_yaw_keeps_yaw_only_from_gt` proves the most subtle case: when `R_gt = R_z(0.7) · R_y(0.1)` is mixed with `R_vi = R_y(0.3)`, the result's yaw component matches `0.7` and the tilt component matches `0.3` — the ablation is correctly attributing each axis to its source. This harness is the recommended first step for resolving the previously-conjectured "`14°` yaw error explains MH_03's 4× ATE penalty" claim from the prior VI-init validation: the new flags decouple rotation, yaw, roll/pitch, gyro bias drift, accel bias, initial velocity, and IMU/cam time offset, so the actual contribution of each can be measured rather than guessed.
- `docs/vi_initialization_integration.md` revised after external design review. The biggest behavioural changes are (a) the pose-rotation promotion path now spells out the world→camera Pose inversion explicitly (`R_c←w = (R_w←b · R_b←c)^T`, NOT `R_w←b`) and the matching translation update `t_c←w_new = -R_c←w_new · C_w_old` that preserves the camera centre across the rotation rewrite; the original draft would have silently inverted world frame on every promotion, (b) the stale-IMU-factor gate that suppresses `take_pending_imu_factor()` and inlined `OnlineSlamResult.imu_factor` until `vi_init.completed.is_some()` — once init succeeds, pre-buffered factors are discarded and the count is reported via `Succeeded { discarded_stale_factor_count, .. }` so callers can audit the drop, (c) the `max_samples_before_giving_up` cap is replaced by twin `max_wait_duration_seconds` (semantic) + `max_buffered_samples` (memory guard), (d) the previously-public `vi_init_state` is made private and a `vi_initialization_status() -> ViInitializationStatus` accessor is added so callers cannot leave the pipeline half-initialised by writing through the field, (e) the `Succeeded` / `GaveUp` events are now contracted to fire at most once per sequence with `None` between, and (f) a body-to-camera SE(3) extrinsic field is added to `OnlineSlamViInitConfig` so the IMU/camera convention difference is supplied explicitly rather than assumed identity. Three new sections: "Accel-bias caveat" notes that the static bootstrap's `bias_acc` is essentially the gravity-magnitude residual and the lateral components are absorbed into `R_w←b` (the standalone `interfaces.md` description carries the same caveat); "Detector windowing" records the planned sliding-window + time-weighted statistics changes to the standalone `VisualInertialInitializer` so it can handle "non-stationary then stationary" sequences; "OSS comparison and feature-matrix wording" splits the previously-flat "VI initialization ○" comparison into a three-column breakdown (static bootstrap / dynamic alignment / periodic scale refinement) so the parity claim against ORB-SLAM3 / VINS-Mono / OKVIS / Kimera-VIO is honest about which column is `visloc-rs`'s next milestone. The test strategy section is expanded from three test layers to fourteen named tests, with two marked `[pose-conv]` as load-bearing for the rotation/translation conversion that can otherwise produce silent runtime bugs. The progress.md MH_03 attribution is softened from "the ~14° rotation residual that the gravity-only initialiser cannot fix" to an explicit acknowledgement that `0.0019 rad/s × 132.6 s ≈ 14.4°` also explains the residual as accumulated gyro-bias drift, and points to the new ablation harness as the resolution path; the `decisions.md` "Visual-Inertial Initialisation Direction" stays as-is.

- `visloc_slam::VisualInertialInitializer` — closes the last of the OSS-parity gaps that the IMU-only EuRoC baseline highlighted. ORB-SLAM3, VINS-Mono, OKVIS, and Kimera-VIO all gate the joint estimator on a "VI initialisation" stage that turns the raw IMU stream into `(R_w←b, v_w, b_g, b_a)` before any image constraint is consumed; without that stage every state has to be seeded from ground truth (cheating) or accumulate hundreds of metres of dead-reckoning drift before the first keyframe. The new module ships the stationary-window flavour: ingest body-frame IMU samples via `push_sample(gyro, accel, dt)` (non-positive `dt` silently dropped, matching the existing `ImuPreintegrator` convention), then `try_initialize()` either returns `Ok(VisualInertialInitializationResult { gravity_world, initial_rotation_body_to_world, initial_velocity_world, bias_gyro, bias_acc, samples_consumed, duration_seconds, gyro_std, accel_std, mean_accel_magnitude })` or `Err(StationaryRejectionReason::{InsufficientSamples, InsufficientDuration, GyroNoiseTooHigh, AccelNoiseTooHigh, AccelMagnitudeMismatch})` with the failing predicate's observed value vs the configured limit. The recovery uses three closed-form readouts: (a) `b_g = ω̄` — the gyro signal mean over the stationary window, (b) `R_w←b` = shortest rotation that lifts the mean specific-force direction into the world "up" direction `-g_w / ‖g_w‖` (`UnitQuaternion::rotation_between(&a_b_mean, &world_up)`; yaw is unobservable from gravity alone and is left at zero), and (c) `b_a` = `a_b_mean - R_w←b^T · (-g_w)` — the magnitude residual after the rotation absorbs the direction. Stationary detection guards the read-out with three thresholds — per-axis gyro standard deviation, per-axis accel standard deviation, and the magnitude error between the mean specific-force and `‖g_w‖` — plus a minimum sample count and minimum window duration. The defaults match the EuRoC stationary holding period (`gravity_world = (0, 0, -9.81)`, `min_stationary_window_seconds = 0.5`, `max_gyro_std = 0.05 rad/s`, `max_accel_std = 0.5 m/s²`, `max_accel_magnitude_error = 0.5 m/s²`, `min_samples = 50`). Twelve new tests in `pipelines/slam/src/vi_initializer.rs::tests` pin the contract: (a) `default_config_uses_z_up_gravity` — Default ships EuRoC z-up gravity, (b) `stationary_level_window_recovers_zero_rotation_and_zero_biases` — 200 samples × 5 ms of `gyro = 0`, `accel = (0, 0, 9.81)` recovers identity rotation, zero gyro/accel bias, zero velocity, zero std-devs, and mean magnitude exactly `9.81 m/s²`, (c) `tilted_stationary_window_recovers_rotation` — body tilted 30° about world-x produces `accel_body = R_w←b^T · (0,0,9.81)`; the recovered rotation lifts that accel back to world-up within `1e-9` and matches the synthetic rotation exactly (no yaw, so the no-yaw assumption is unambiguous), (d) `nonzero_gyro_mean_becomes_bias` — `gyro_bias = (0.01, -0.02, 0.005)` recovered within `1e-12`, (e) `rejects_noisy_gyro` — alternating ±1 rad/s gyro on x-axis triggers `GyroNoiseTooHigh` with `observed.x > 0.05 limit`, (f) `rejects_noisy_accel` — alternating ±2 m/s² accel triggers `AccelNoiseTooHigh`, (g) `rejects_accel_magnitude_mismatch` — `accel = (0, 0, 2.0)` (way off 9.81) triggers `AccelMagnitudeMismatch` with the actual `observed / expected / tolerance` populated, (h) `rejects_insufficient_samples` — 10 samples vs default min 50 triggers `InsufficientSamples { have: 10, need: 50 }`, (i) `rejects_insufficient_duration` — 50 samples × 5 ms = 0.25 s under a 1 s threshold triggers `InsufficientDuration`, (j) `reset_clears_buffered_samples` — `reset` zeros the buffer and the buffered duration, (k) `push_sample_drops_nonpositive_dt` — zero / negative / NaN `dt` all leave `samples_seen` at zero, and (l) `accel_magnitude_mismatch_absorbed_by_bias_when_within_tolerance` — `accel = (0, 0, 9.95)` (within tolerance) recovers a non-zero `bias_acc` such that `R_w←b · (accel - bias_acc) = -gravity_world` within `1e-9`, pinning the bias-absorption invariant. The initialiser ships outside the SLAM critical path; callers typically use it once per sequence to seed `OnlineSlamImuConfig.bias_gyro_linearisation` / `bias_acc_linearisation` and the first-keyframe pose before `OnlineSlamPipeline.local_vi_ba` takes over. All three types and the rejection-reason enum re-exported through the `visloc_rs` facade.
- `examples/euroc_imu_dead_reckon_demo` gains `--run-vi-init` / `--seed-from-vi-init` / `--vi-init-window-seconds <s>` / `--vi-init-gyro-std-limit <r>` / `--vi-init-accel-std-limit <r>` CLI flags so the same EuRoC harness can validate the VI initialiser against ground truth and run the propagator with the honest "no GT cheating" bootstrap. With `--run-vi-init`, the demo collects the leading IMU window (default 1 s = 200 samples at 200 Hz on EuRoC), runs `try_initialize()`, and logs the recovered `(bias_gyro, bias_acc, rotation)` plus three residuals against ground truth: `rotation_residual_vs_gt_deg` (full quaternion distance — dominated by unobservable yaw), `gravity_alignment_residual_deg` (angle between the body-frame "up" directions implied by the recovered and GT orientations — the *fair* metric, since gravity does not pin yaw), and the L² norms of the gyro / accel bias residuals. With `--seed-from-vi-init`, the propagator uses the VI-init bootstrap (`rotation`, `velocity = 0`, `bias_gyro`, `bias_acc`) instead of the GT-cheating seed; position is still anchored at the GT location since IMU cannot observe absolute position. Empirical validation on the five EuRoC sequences that ship in the existing baseline table: under loose-threshold settings (`--vi-init-gyro-std-limit 0.3 --vi-init-accel-std-limit 1.0`), `MH_02_easy / MH_03_medium / V2_01_easy` all initialise successfully with gravity-alignment residual `< 2°` and gyro-bias L² residual `< 0.12 rad/s` (MH_03 and V2_01 reach `< 0.002 rad/s`, i.e. within the EuRoC IMU's gyro random-walk floor); `MH_01_easy / V1_01_easy` are still rejected (accel-std-too-high) because the leading IMU window in those recordings contains real motion before the camera-stationary period, not pure handling noise. Honest dead-reckon ATE comparison (V2_01_easy): `rigid ATE RMSE = 274 m` with VI-init seed vs `258 m` with GT seed — *nearly indistinguishable*, i.e. the VI-init recovery on this sequence captures essentially the same state as the GT Vicon-derived seed; on MH_03_medium the same comparison reads `929 m` vs `209 m` because the recovered orientation carries a `~14°` yaw-equivalent error that compounds into position drift through gravity coupling (the dominant failure mode of yaw-unobservable VI init under monocular-only motion). The CLI flags + the in-example residual logging are the validation path that lets a downstream caller A/B test the initialiser against any future VI-init upgrade (e.g. the ORB-SLAM3 motion-only optimisation that also recovers scale + velocity once the visual frontend is hot) without re-running ground truth comparisons by hand.

- `umeyama_similarity_transform(source, target, with_scale) -> Option<TrajectorySimilarityTransform>` plus two new variants on the existing `TrajectoryAlignment` enum (`Umeyama` for rigid SE(3) Procrustes and `UmeyamaWithScale` for Sim(3) including scale recovery) so `PoseTrajectory::translation_error_summary_against_with_alignment` can produce the standard ATE numbers that ORB-SLAM3 / VINS-Mono / OKVIS / Kimera-VIO report on EuRoC. The closed-form solver follows Umeyama 1991 — centre both point clouds, build the `(1/n) Σ (b_i − b̄)(a_i − ā)^T` cross-covariance matrix, take its SVD, repair the reflection case via `S = diag(1, 1, sign(det(U V^T)))`, recover `R = U S V^T`, and (when `with_scale = true`) set `s = trace(D S) / σ_a²` with `σ_a² = (1/n) Σ ‖a_i − ā‖²`; the rigid variant fixes `s = 1`. The returned `TrajectorySimilarityTransform { scale, rotation, translation }` exposes an `apply(point) -> Point3` helper so callers can re-project an estimated trajectory into the reference frame without re-deriving the transform. Internally `trajectory_alignment_transform` replaces the previous `translation_alignment_offset` helper (still private), so the `None` / `FirstMatchedTranslation` variants keep their bit-identical behaviour via `TrajectorySimilarityTransform::identity()` / `TrajectorySimilarityTransform::pure_translation(t)` shims — every existing report-generation path (`translation_errors_against_with_alignment`, `translation_errors_csv_against_with_alignment`, `to_html_report_against_with_alignment`, `trajectory_comparison_svg`) automatically picks up the new variants without further plumbing. Seven new tests in `pipelines/tracking/src/lib.rs::umeyama_alignment_tests` cover (a) `umeyama_recovers_pure_translation` — pure-translation problem with zero rotation and unit scale recovers the translation within `1e-10`, (b) `umeyama_recovers_rotation_and_translation` — non-trivial rotation `(0.3, -0.4, 0.5) rad` + translation `(-1, 2, -3)` round-trips through the solver within `1e-8` and the per-pose residual after applying the transform is `< 1e-8`, (c) `umeyama_recovers_similarity_with_scale` — `s=3.4`, non-trivial rotation, non-zero translation recovers all three within `1e-7..1e-8`, (d) `umeyama_returns_none_for_insufficient_points` — single correspondence returns `None`, (e) `umeyama_returns_none_for_zero_variance_source` — coincident source points return `None` instead of producing an ill-conditioned NaN transform, (f) `trajectory_alignment_umeyama_drives_error_to_zero_for_similarity_perturbation` — a 6-pose reference trajectory perturbed by `Sim(3)` with `s=2`, non-trivial rotation and translation produces sub-`1e-7` ATE RMSE under `UmeyamaWithScale` while the raw `None` alignment leaves an error `> 1m`, and (g) `trajectory_alignment_umeyama_rigid_does_not_remove_scale` — a half-scale trajectory leaves a non-trivial residual under `Umeyama` but collapses to `< 1e-8` under `UmeyamaWithScale`, pinning the rigid-vs-similarity behavioural split. Both new types are re-exported through the `visloc_rs` facade. The `examples/euroc_imu_dead_reckon_demo` example now consumes both variants to emit `ate_rigid_rmse_m` / `ate_rigid_max_m` / `ate_similarity_rmse_m` / `ate_similarity_max_m` / `ate_similarity_scale` lines in its `summary.txt` so the IMU-only baseline produces a directly comparable ATE number against published EuRoC results, in addition to the previous raw-drift columns.
- Measured per-sequence IMU-only baseline ATE on five EuRoC MAV sequences (MH_01_easy, MH_02_easy, MH_03_medium, V1_01_easy, V2_01_easy) using the new `euroc_imu_dead_reckon_demo` harness with Umeyama-aligned ATE. The raw position RMSE ranges from 297 m (MH_03_medium) up to 1905 m (MH_01_easy), as expected for a forward-Euler strapdown propagator that consumes ~1.5–3 minutes of accelerometer + gyro samples with no visual aiding; rigid SE(3) Umeyama alignment compresses this to 209–1356 m, and Sim(3) alignment with scale produces 1.79–4.50 m at scale factors `0.000544..0.004832` — a vivid demonstration that pure-IMU integration does not preserve trajectory shape (the recovered scale is `~10⁻³`, i.e. the IMU-integrated trajectory has expanded by a factor of `~10³` against the bounded GT, so the similarity ATE absorbs the explosion into the scale parameter rather than reflecting a real "good shape" match). Numbers serve as the apples-to-apples lower-bound that any visual-inertial pipeline added on top of `OnlineSlamPipeline.local_vi_ba` + `ImuPredictiveMotionModel` is expected to drive down by orders of magnitude — published EuRoC ATE numbers for the reference OSS systems are `0.035 m` (ORB-SLAM3) / `0.23 m` (VINS-Mono / OKVIS) / `0.14 m` (Kimera-VIO), so the headroom is between three and four orders of magnitude. The measurement methodology, the per-sequence summaries, and the comparison with published OSS numbers are now in `docs/progress.md`; the per-sequence `summary.txt` / `imu_dead_reckon.csv` / `imu_dead_reckon_errors.csv` outputs are reproducible from `cargo run --release --example euroc_imu_dead_reckon_demo -- --euroc-dir <MH_xx_easy> --out-dir <out>` given a local EuRoC clone.
- `visloc_tracking::ImuPredictiveMotionModel` — a loosely-coupled inertial pose predictor that drops into [`Tracker`]'s `MotionModel` slot to replace the rotation-static [`ConstantPoseMotionModel`] / position-only [`ConstantVelocityMotionModel`]. The model carries an `ImuPredictiveMotionModelConfig { gravity_world, bias_gyro, bias_acc }` (KITTI y-down default `(0, 9.81, 0)`), an internal `pending_samples` buffer (mutated via `push_imu_measurement(gyro, accel, dt)`; non-positive `dt` silently dropped to keep raw IMU replays robust), and a world-frame `velocity_world` (mutated via `set_velocity_world(v)`; the model does NOT auto-update this on `observe` — downstream code is expected to feed in the refined velocity from a local VI-BA, finite-difference, or simply leave it at zero for a "rotation prior, constant-translation" predictor). `set_biases(bg, ba)` mirrors the same shape for the gyro/accel bias linearisation point so a downstream solver can rebase the predictor as biases are estimated. The `MotionModel::predict_pose` impl converts the previous successful pose to body-to-world form (`R_bw = R_wc⁻¹`, `p_bw = camera_center_world`), forward-Euler integrates every pending sample using the strapdown equations `R_{k+1} = R_k · Exp((ω − b_g)·Δt)`, `v_{k+1} = v_k + (R_k·(a − b_a) + g_world)·Δt`, `p_{k+1} = p_k + v_k·Δt + 0.5·(R_k·(a − b_a) + g_world)·Δt²`, then converts the result back into `world_to_camera` form for the tracker. `observe` drains the pending buffer on a successful frame so the next inter-frame window starts fresh; `reset` zeros both the buffer and the velocity. The predictor is opt-in — `Tracker::with_motion_model(localization, ImuPredictiveMotionModel::new(...), config)` swaps it in without touching any other tracker config. The new type is re-exported through the `visloc_rs` facade. Seven new tests in `pipelines/tracking/src/lib.rs::imu_predictive_motion_tests` cover (a) `returns_last_pose_when_no_samples_pushed` — empty buffer → identity prediction, (b) `returns_none_when_no_previous_pose` — `predict_pose(.., None)` short-circuits to `None`, (c) `stationary_under_gravity_holds_pose` — body at world origin with accel exactly cancelling gravity (`(0,0,9.81)` reading under `gravity_world = (0,0,-9.81)`) integrates 10 samples × 0.05 s without any pose drift, (d) `pure_yaw_rotation_propagates_rotation` — zero-gravity scene with 1 s of `(0,0,π/2 rad/s)` gyro produces the expected ±90° world-frame rotation of the world-to-camera quaternion within `0.1°`, (e) `constant_velocity_translates_position` — zero-gravity / zero-accel / zero-gyro but `velocity_world = (1, 0, 0)` and 100 × 0.01 s samples translates the predicted center by exactly `1 m` along world-x, (f) `observe_drains_pending_window_on_success` — three buffered samples are wiped by a successful `TrackingResult`, and (g) `reset_clears_velocity_and_samples` — `reset` zeros both the buffer and the velocity. Wiring `ImuPredictiveMotionModel` end-to-end through `OnlineSlamPipeline` (sharing the same IMU stream pushed via `push_imu_measurement` with the pre-integrator state) is the natural next step — the predictor lives in `pipelines/tracking` (no `pipelines/slam` dependency) so callers can compose either pipeline independently. Closes the last of the three OSS-parity gaps identified after task #55; combined with the EuRoC dead-reckoning baseline + the local VI-BA stage that landed in this release, every level of inertial integration (rotation prior → joint VI-BA → measurement harness) is now reachable from a single workspace.
- `OnlineSlamPipeline` gains an optional sliding-window local visual-inertial BA stage that closes the loop between the previously-staged `ImuPreintegrationFactor` and the appearance-only critical path. `OnlineSlamConfig.local_vi_ba: Option<OnlineSlamLocalBaConfig>` (default `None`) opts the pipeline into per-keyframe `(velocity, bias_gyro, bias_acc)` book-keeping plus a trailing-window BA solve every `trigger_every` IMU factors (default `1`, i.e. every new keyframe), refining the last `window_size` keyframes (default `5`) over their `min_observations_per_landmark`-filtered (default `2`) landmark observations + every stored IMU factor whose `from / to` keyframe pair both sit inside the window. The first in-window keyframe's pose / velocity / bias are gauge-fixed; everything else is free. The new `crate::OnlineSlamLocalBaState { config, keyframe_state, factor_history, pending_factors_since_last_trigger }` carries the rolling state (capped at `4 * window_size` recent factors so memory is bounded), and a new free function `run_local_vi_ba(map, state)` does the work; the pipeline's `process_frame` calls a private `maybe_run_local_vi_ba` wrapper after `stage_imu_factor_on_new_keyframe` so the BA fires exactly when a new factor was just emitted. Initial velocity for a newly-promoted keyframe is seeded from the inter-keyframe camera-centre displacement divided by the connecting factor's `delta_time`; biases start at the config's linearisation point. On a successful trigger the refined poses + landmarks are written back into `map.keyframes[*].frame.pose` / `map.landmarks[*].position`, and the refined `(velocity, bias)` state is written back into `state.keyframe_state` so the next trigger starts from the new linearisation point. The new `OnlineSlamResult.local_vi_ba: Option<OnlineSlamLocalBaStats>` carries the per-trigger window keyframe ids, landmark / observation / IMU-factor counts, and the inner `BaResult` (LM trace + initial / final cost). `reset_sequence_state` now also clears the VI-BA state. To preserve every existing caller, the new `local_vi_ba` field is added with `Default = None`; the four `OnlineSlamConfig { ... }` struct literals that didn't already spread the default (in `pipelines/slam/tests/online_slam.rs`) gained a trailing `local_vi_ba: None` so the IMU-free behaviour is bit-identical on those paths. Six new tests cover (a) `local_vi_ba_returns_none_without_factors_in_window` — no in-window factors → no solve, (b) `local_vi_ba_refines_window_when_factors_present` — 3-keyframe scene with 2 IMU factors produces `Some(stats)` with the expected window ids / factor count / non-zero observation count plus per-keyframe state, (c) `local_vi_ba_state_reset_clears_history` — `reset` clears factor history + keyframe state + pending counter, (d) `register_new_factor_respects_trigger_every` — counter rolls forward and the trigger boundary fires at `trigger_every`, (e) `online_slam_runs_local_vi_ba_when_factor_emitted` — full `OnlineSlamPipeline` wiring on a 2-keyframe scene with `local_vi_ba: Some(_)` produces `result.local_vi_ba.is_some()` with the expected window keyframe ids + factor count, and the state table populates per-keyframe entries that get cleared by `reset_sequence_state`, and (f) `online_slam_skips_local_vi_ba_when_disabled` — with `local_vi_ba: None`, the factor is still emitted but `result.local_vi_ba` stays `None` and `local_vi_ba_state` is `None`. Closes the most expensive of the three OSS-parity gaps (visual-inertial joint optimisation on the critical path); full benchmark numbers against ORB-SLAM3 / VINS-Mono on EuRoC sequences are a follow-up that needs the dataset downloaded locally.
- EuRoC MAV stereo-inertial dataset loader (`visloc_io::euroc`) plus an `examples/euroc_imu_dead_reckon_demo` IMU-only baseline harness that produces an ATE number on real EuRoC data. The loader parses `mav0/cam0|cam1/data.csv` (timestamp + filename image manifests), `mav0/imu0/data.csv` (200 Hz body-frame gyro + accel rows), `mav0/state_groundtruth_estimate0/data.csv` in all three published column layouts (8-column pose-only, 13-column pose + velocity, 17-column pose + velocity + gyro/accel biases — the EuRoC MH-sequence carries the full 17 columns including Vicon-derived biases), and the matching `cam*/sensor.yaml` + `imu0/sensor.yaml` calibration blobs (`T_BS` 4×4 body-to-sensor extrinsics, `rate_hz`, `resolution`, pinhole `intrinsics (fu, fv, cu, cv)`, radial-tangential `distortion_coefficients`, and the IMU noise / random-walk densities). The YAML parser is a small line-based extractor specialised for the `key: value` + `key: [a, b, c, ...]` + indented `T_BS:` block grammar EuRoC ships — no new YAML dependency is pulled into the workspace. A composite `read_euroc_dataset_dir(dir)` returns an aggregate `EurocDataset { root, cam0_image_dir, cam1_image_dir, cam0_images, cam1_images, cam0_calibration, cam1_calibration, imu_samples, imu_calibration, ground_truth }`; ground truth is optional so the loader still succeeds on EuRoC subsets without the Vicon CSV. Image pixels are not decoded here — pair the returned `EurocImageEntry.filename` with `visloc_io::images::read_common_image` when running the harness, so the loader stays outside the `image-io` feature gate. Five new unit tests in `crates/io/src/euroc.rs::tests` cover (a) `parses_camera_sensor_yaml` — a trimmed-down replica of the canonical EuRoC cam0 sensor.yaml round-trips through the parser with intrinsics / `T_BS` / distortion matching bit-for-bit, (b) `parses_imu_sensor_yaml` — IMU sensor.yaml parser reads rate_hz / noise densities / random walks correctly, (c) `parses_imu_csv` — 7-column IMU CSV with `#` headers and arbitrary whitespace produces ordered `EurocImuSample` rows, (d) `parses_image_manifest_and_ground_truth_variants` — image manifest + GT CSV parsed in 17/13/8-column variants with the optional velocity / bias fields populated only when present, and the Hamilton `(qw, qx, qy, qz)` order correctly lifted into nalgebra's `Quaternion::new(w, x, y, z)`, and (e) `reads_complete_dataset_dir` — composite reader walks a synthetic `mav0/` layout and returns the aggregate handle with empty ground truth tolerated. The `euroc_imu_dead_reckon_demo` example seeds a forward-Euler strapdown propagator from the first GT row that lies inside the IMU window (position, velocity, orientation, gyro + accel biases), then propagates `(R, v, p)` through every IMU sample with `R_{k+1} = R_k · Exp((ω − b_g)·Δt)`, `v_{k+1} = v_k + (R_k·(a − b_a) + g_world)·Δt`, `p_{k+1} = p_k + v_k·Δt + 0.5·(R_k·(a − b_a) + g_world)·Δt²` (EuRoC convention `g_world = (0, 0, −9.81)`), and at every cam0 frame timestamp records the per-frame nearest-neighbour position error (m) and orientation error (deg) versus GT. Writes `imu_dead_reckon.csv` (per-IMU-step trajectory: timestamp + position + quaternion + velocity), `imu_dead_reckon_errors.csv` (per-cam0-frame ATE: timestamp + GT position + estimated position + position_error_m + orientation_error_deg), and `summary.txt` (RMSE + max for both metrics, frames recorded, IMU samples consumed, duration). The example is intentionally the "lower bound" reference: no visual aiding, no bias re-estimation, no zero-velocity updates — any visual-inertial pipeline added on top should drive the numbers down, and having the IMU-only baseline on disk makes that improvement quantifiable. Smoke-tested end-to-end on a synthetic hovering EuRoC fixture (10 cam0 frames @ 20 Hz, 100 IMU samples @ 200 Hz, accel exactly cancelling gravity) with ATE_RMSE_position = 0 m and ATE_RMSE_orientation = 0 deg, confirming the propagator + nearest-neighbour ATE wiring is correct before any real-data run.
- `OnlineSlamPipeline` now optionally wires the Forster IMU pre-integrator into its keyframe stream. `OnlineSlamConfig` gains `imu: Option<OnlineSlamImuConfig>` (gravity-world, gyro/accel bias linearisation, position/velocity/rotation weights) and the pipeline carries a matching `imu_state: Option<OnlineSlamImuState>` (running `ImuPreintegrator`, `last_keyframe_id` window-left anchor, and a `pending_factor` slot). Two new methods on `OnlineSlamPipeline`: `push_imu_measurement(gyro, accel, dt)` folds a body-frame sample into the running window (no-op when IMU is unconfigured; non-positive `dt` silently dropped to keep raw IMU replays robust), and `take_pending_imu_factor() -> Option<ImuPreintegrationFactor>` retrieves the factor most recently staged by `process_frame`. Each `process_frame` call now checks whether the local mapper registered a new keyframe at the incoming `frame.id` (i.e. `applied_update.keyframe_count > 0`); if so AND there is a previous keyframe to anchor against, the pipeline snapshots the running pre-integration delta into an `ImuPreintegrationFactor { keyframe_id_from = prev_kf, keyframe_id_to = frame.id, delta, gravity_world, weight_* }`, resets the integrator so the next inter-keyframe window starts fresh, and exposes the factor on both `OnlineSlamResult.imu_factor` and `take_pending_imu_factor()`. The pipeline itself does not solve against the factor — it stays appearance-driven on the critical path; the factor is a hint for downstream pose-graph / BA glue (existing `BundleAdjustment::add_imu_factor`, `ImuPreintegrationFactor::residual`). `reset_sequence_state` now also resets the integrator and clears `last_keyframe_id` / `pending_factor` so a per-sequence reset is symmetric with the existing tracker / mapper reset. To preserve every existing caller, the new `imu` field is added with `Default = None`; the six `OnlineSlamConfig { ... }` struct literals in the repo (examples + tests) gain a trailing `..OnlineSlamConfig::default()` so the IMU-free behaviour is bit-identical on those paths. `ImuPreintegrator` now derives `PartialEq` (was `Debug, Clone` only) so `OnlineSlamPipeline`'s `PartialEq` derive still compiles with the new state. Seven new tests in `pipelines/slam/tests/online_slam.rs` cover (a) `online_slam_emits_imu_factor_between_consecutive_keyframes` — IMU-enabled pipeline + two distinct camera positions (1.5 m apart, above the `SimpleKeyframePolicy::min_translation = 1.0` default so the mapper actually registers a second keyframe) + 10 synthetic samples between them produces `imu_factor.is_some()` with `keyframe_id_from = 10`, `keyframe_id_to = 30`, `delta.delta_time = 1.0 s ± 1e-9`, then `take_pending_imu_factor()` returns the factor exactly once, (b) `online_slam_imu_factor_is_none_without_imu_config` — default config keeps `imu_state` as `None`, `push_imu_measurement` is a no-op, and every `process_frame` result carries `imu_factor: None`, (c) `online_slam_reset_clears_imu_window_state` — `reset_sequence_state` clears `last_keyframe_id` / `pending_factor` while preserving the map, (d) `online_slam_imu_integrator_resets_after_factor_emission` — after a factor closes a window, the running `ImuPreintegrator.delta().delta_time` reads only the post-reset samples (not the cumulative trajectory; a 3-sample post-emit window reads `0.3 s`, not `1.3 s`), (e) `online_slam_imu_factor_propagates_gravity_weights_and_bias_linearisation` — a non-default `OnlineSlamImuConfig` (gravity `(0.1, 9.7, -0.2)`, non-zero gyro / accel biases, `weight_position=3 / velocity=5 / rotation=7`) produces a factor whose `gravity_world`, all three weights, AND `delta.bias_gyro_linearisation` / `bias_acc_linearisation` match the config bit-for-bit — pins the wiring against a copy-paste regression that would silently degrade downstream BA conditioning, (f) `online_slam_imu_window_persists_across_non_keyframe_frames` — when an intermediate frame tracks successfully but the mapper rejects it as a keyframe (0.1 m camera shift, below `min_translation = 1.0`), the IMU window stays open against the prior keyframe and the eventual factor at the next genuine keyframe carries ALL samples from both sub-windows (4 + 6 samples → `delta_time = 1.0 s`), and (g) `online_slam_imu_emits_factors_across_three_consecutive_keyframes` — three keyframes in a row produce two chained factors `KF1→KF2` (`0.5 s`) and `KF2→KF3` (`0.7 s`) with the second factor's `delta_time` proving the post-reset accumulator is genuinely starting from zero (not carrying over the first window's `0.5 s`); the third call's `take_pending_imu_factor()` returns the `KF2→KF3` factor, with the next call returning `None`. Closes the deferred IMU-integration follow-up; full BA against the emitted factor remains a downstream caller responsibility.
- `scripts/run_kitti_3dgs_smoke.sh` now gates on `landmarks > 0` after the cross-format parity check. The inspect examples already exit non-zero on empty cameras / keyframes, but a run where every stereo feature was rejected upstream (e.g. disparity gate dropped everything) would still produce a writer-symmetric pair of COLMAP triples whose `landmarks=0` agreed and trivially passed the parity step. The 3DGS bootstrap directory is useless without 3D structure, so the smoke now refuses to call such a run "OK" and aborts with a clear message.
- `docs/colmap_compatibility.md` now documents the 3DGS / NeRF bootstrap exporter (`write_colmap_text_model_for_3dgs` + `write_colmap_binary_model_for_3dgs`) — symmetric writer pair, shared image-name + camera-model validation, cross-format parity test + smoke harness pointers. The "Current Non-Goals" list was updated accordingly: "Binary COLMAP writing" is no longer a blanket non-goal (the 3DGS-shaped binary writer ships), narrowed to "generic binary COLMAP writing from a `VisualMap`" which is still deferred (no caller needs it).
- `scripts/run_kitti_3dgs_smoke.sh` now uses portable `cmd > log` redirection (with `cat log` afterwards) instead of `cmd | tee log` for the inspect example and `ns-train splatfacto` invocations. POSIX sh / dash do not have `set -o pipefail`, so `tee`'s zero exit would silently mask a failed `cargo run --example inspect_colmap_*` or a failed `ns-train splatfacto` — `set -e` now catches their exit codes directly. The `--run-ns-train` case in particular previously exited zero even when the trainer crashed; the smoke now fails loudly in that case. The summary file also gets an `ns_train_log=<path>` line when `--run-ns-train` actually ran the trainer, so the trainer log is locatable from the summary.
- COLMAP 3DGS text and binary writers now share a single image-NAME validator (`validate_colmap_image_name`) and both gate on `colmap_id_from_camera_model`, so any input accepted by one writer is also accepted by the other. Previously `write_colmap_text_model_for_3dgs` would happily write `CameraModel::Unknown("ANY_STRING")` into `cameras.txt` while the binary counterpart rejected unknown model names, and the text writer would happily write `image_name` strings containing ASCII whitespace, tabs, LF, or CR even though those characters break the text format's space-separated tokens (a `\n` would inject a spurious image record, since `images.txt` alternates a header line with a 2D-points line per image). The binary writer's previous NUL-only check is subsumed by the new shared validator (which rejects NUL + space + tab + LF + CR), tightening the binary side too so the writer pair stays symmetric. Two new tests in `crates/io/tests/colmap_export.rs` pin the symmetry: (a) `write_colmap_text_model_for_3dgs_rejects_unknown_camera_model` mirrors the existing binary-side test on the text writer (`CameraModel::Unknown("BOGUS")` → `InvalidExportInput`), and (b) `write_colmap_models_for_3dgs_reject_image_name_with_format_breaking_characters` parameterises over NUL / space / tab / LF / CR and asserts BOTH writers reject every variant with `InvalidExportInput`. This closes a gap in the smoke harness where a `--colmap-image-prefix`/`--colmap-image-suffix` containing format-breaking characters would have produced an inconsistent half-export — binary errors out, text silently writes a corrupted file that `inspect_colmap_text_model` would then misparse.
- `examples/inspect_colmap_text_model` mirrors `inspect_colmap_binary_model` on the text side: calls `visloc_rs::io::colmap::read_colmap_text_model(<dir>)` and prints `cameras / keyframes / landmarks / observations` counts, exiting non-zero if cameras or keyframes are empty or if the reader rejects the model. `scripts/run_kitti_3dgs_smoke.sh` now runs BOTH inspect tools on the live VO output (text dir AND binary dir), captures their stdout into `colmap_text_inspect.txt` / `colmap_binary_inspect.txt`, then grep/awks the four count fields and aborts (`exit 1`) if any field disagrees across formats OR is missing on either side (an empty value on one side would otherwise silently match another empty value if an inspect tool's output schema drifted). This closes a gap where the text dir was previously only file-existence-checked — `read_colmap_text_model` is now actually exercised on real driving data — and adds a real-data cross-format parity safeguard on top of the synthetic-input parity test from `crates/io/tests/colmap_export.rs::write_colmap_text_and_binary_models_for_3dgs_emit_equivalent_maps`. The smoke summary file's `## COLMAP binary inspect` section also now actually contains the binary inspect log (the prior wiring referenced an undefined `$inspect_log` shell variable that would have tripped `set -u` at runtime — caught before any real run).
- COLMAP 3DGS writer text-vs-binary parity now pinned by `write_colmap_text_and_binary_models_for_3dgs_emit_equivalent_maps` in `crates/io/tests/colmap_export.rs`. Drives both writers against the same synthetic 3-frame stereo scene, reads them back through `read_colmap_text_model` / `read_colmap_binary_model`, and asserts (a) summary counts agree (frame / landmark / observation), (b) the shared camera's `width` / `height` / `params` are identical, (c) every keyframe id resolves to a keyframe whose translation differs by `< 1e-9` and whose rotation distance (via `rotation_to(&q).angle()`) is `< 1e-9`, and (d) every landmark id has the same world-frame position within `1e-9`. This is the in-repo invariant that justifies the smoke harness emitting both formats from a single VO run and trusting they encode the same map.
- `scripts/run_kitti_3dgs_smoke.sh` + `examples/inspect_colmap_binary_model` — KITTI → COLMAP → 3DGS bootstrap smoke harness. The script fetches a small stride-4 KITTI subset (default sequence 00, 60 frames; reuses `scripts/fetch_kitti_seq00_images.py` with `--cameras image_0,image_1 --also-fetch-poses --skip-existing`), runs `online_slam_stereo_vo_kitti_demo --frontend classical --colmap-export <out>/colmap_text --colmap-export-binary <out>/colmap_binary` against it, asserts both writer surfaces produced their three files (`cameras.{txt,bin}` / `images.{txt,bin}` / `points3D.{txt,bin}`) under `<out>`, then loads the binary directory back through the new `inspect_colmap_binary_model` example so writer ↔ reader divergence cannot slip through the smoke unnoticed. `inspect_colmap_binary_model` is a tiny example that calls `visloc_rs::io::colmap::read_colmap_binary_model(<dir>)` and prints `cameras / keyframes / landmarks / observations` counts (exits non-zero if cameras or keyframes are empty, or if the reader rejects the model). Optional `--run-ns-train` invokes `ns-train splatfacto --data <out>/colmap_text` if `ns-train` is on `PATH`; the trainer is otherwise skipped so the smoke stays tractable without a CUDA Python environment. All knobs (`KITTI_3DGS_DATA_DIR` / `_OUT_DIR` / `_SEQUENCE` / `_FETCH_STRIDE` / `_FETCH_MAX_FRAMES` / `_MAX_FRAMES` / `_START_FRAME` / `_WORKERS` / `_PROGRESS_EVERY` / `_COLMAP_IMAGE_PREFIX` / `_COLMAP_IMAGE_SUFFIX`) also exposed as long flags following the existing `run_kitti_deep_vo_smoke.sh` conventions; `--skip-fetch` reuses an already-fetched subset. Writes `kitti_3dgs_smoke_summary.txt` alongside the raw `summary.txt` and the `colmap_binary_inspect.txt` reader output for inclusion in CI summaries.
- `examples/online_slam_stereo_vo_kitti_demo` gains `--colmap-export <dir>`, `--colmap-export-binary <dir>`, `--colmap-image-prefix <s>`, `--colmap-image-suffix <s>` CLI flags so the existing real-data KITTI stereo VO demo can produce a 3D Gaussian Splatting / NeRF bootstrap directory in one step (mirrors the surface already on `examples/stereo_vo_external_deep_files`). The export uses the BA-refined poses when `--no-stereo-ba` was not passed, the raw VO poses otherwise; PGO-refined poses are intentionally NOT used because PGO is gated behind a synthetic-loop-closure edge that isn't always available — callers who want a PGO-aware export can re-run from `vo_poses.txt`. The image NAME field uses `<prefix><6-digit frame_idx><suffix>` (suffix defaults to `.png` to match the KITTI image filenames), so a typical KITTI invocation is `--colmap-image-prefix "" --colmap-image-suffix .png`. Drop the resulting `<dir>` straight into `nerfstudio ns-train splatfacto --data <dir>` or Inria gaussian-splatting's `convert.py --skip-matching` flow.
- COLMAP binary writer validation edge cases now pinned by two new tests in `crates/io/tests/colmap_export.rs`: (a) `write_colmap_binary_model_for_3dgs_rejects_image_name_with_nul_byte` — an `image_name` closure that returns a string containing an embedded NUL byte returns `ColmapError::InvalidExportInput` instead of writing a `images.bin` whose NAME field is silently truncated at the first NUL (the COLMAP binary NAME is NUL-terminated). (b) `write_colmap_binary_model_for_3dgs_rejects_unknown_camera_model` — a `Camera` whose model is `CameraModel::Unknown(<unrecognised name>)` (i.e. not one of the COLMAP camera-model aliases the `colmap_id_from_camera_model` helper recognises) returns `ColmapError::InvalidExportInput` rather than picking an arbitrary model id. These were reachable but untested validation paths in the newly-landed binary writer.
- `visloc_io::colmap::write_colmap_binary_model_for_3dgs` — binary counterpart of `write_colmap_text_model_for_3dgs`. Writes `cameras.bin`, `images.bin`, `points3D.bin` under `out_dir` in the little-endian layout that COLMAP's reference reader expects (`u64 count` framing per file; cameras as `u32 id, i32 model_id, u64 width, u64 height, f64 params[n]`; images as `u32 frame_id, f64 qw qx qy qz, f64 tx ty tz, u32 camera_id, NUL-terminated NAME, u64 points2d_count, [f64 x, f64 y, i64 point3d_id]*`; points3D as `u64 id, f64 x y z, u8 r g b, f64 error, u64 track_length, [u32 image_id, u32 point2d_idx]*`). All other semantics — single shared camera, per-frame stereo landmarks lifted through `pose.camera_to_world()`, `frame_idx` as the COLMAP image id, `image_name(frame_idx)` for the NAME field — match the text writer exactly. A new private `colmap_id_from_camera_model` helper inverts the existing `camera_model_from_colmap_id` so the writer encodes `CameraModel` variants (Pinhole / SimplePinhole / SimpleRadial / Radial / OpenCv plus the `Unknown(name)` aliases the binary reader already recognises) to their COLMAP model id; unknown / unencodable names return `ColmapError::InvalidExportInput`. CLI surface: `examples/stereo_vo_external_deep_files` gains `--colmap-export-binary <dir>` alongside the existing `--colmap-export` flag (the two are independent so a single VO run can emit both formats), and `--colmap-image-prefix` / `--colmap-image-suffix` flags now apply to both writers. Two new tests in `crates/io/tests/colmap_export.rs` cover (a) `write_colmap_binary_model_for_3dgs_round_trips_through_binary_reader` — synthetic 3-frame +z trajectory + 6 stereo landmarks → `read_colmap_binary_model` reconstructs the same camera intrinsics, keyframe poses (KF0 at origin, KF2 at world z = -2), and landmark world positions (lm1 = (0.5, 0, 5.0), lm3 lifted from KF1 = z 5.0), and (b) `write_colmap_binary_model_for_3dgs_rejects_length_mismatch` — mismatched `(poses, left_features, stereo_per_frame)` lengths return `ColmapError::InvalidExportInput` without writing partial files. Drop the resulting `<dir>` into `nerfstudio ns-train splatfacto --data <dir>` or Inria gaussian-splatting workflows that prefer the binary form.
- Gyro-bias observability test (`ba_with_imu_input_recovers_gyro_bias_under_rotation` in `pipelines/slam/src/stereo_vo_ba.rs::tests`) — 4-keyframe rotation-only synthetic scene (camera yaws around its own y-axis at `omega_yaw = 0.06 rad/s` while the camera centre stays at the world origin) with IMU samples that encode the true body-frame angular rate plus a known +y gyro bias `b_truth = (0, 0.015, 0) rad/s`. The test feeds BA the windows with `bias_gyro_init = 0` and `fix_first_bias = false` so all four bias slots are free; with `bias_random_walk_weight = Some(10.0)` tying the slots together, the visual factor pins each pose's rotation absolutely, the IMU rotation residual is paid entirely by the bias slot, and every keyframe's refined gyro bias must land within `5e-3` of `b_truth`. The test also asserts (a) refined accel bias stays within `5e-3` of zero (no signal to push it away from the linearisation point under a zero-gravity / zero-accel scene), and (b) the refined poses retain the input rotation within `2e-3 rad`. This pins down the gyro-bias observability claim the prior IMU-input wiring test (`ba_with_imu_input_wires_through_config`) deferred — that test only exercised the constant-velocity / zero-rotation case where gyro bias is fundamentally unobservable.
- `visloc_slam::write_online_ba_imu_state_csv` / `online_ba_imu_state_rows` / `OnlineBaImuStateRow` (also re-exported from the `visloc_rs` facade) flatten a streaming `OnlineStereoVoBa::trigger_history` into a per-(trigger, in-window keyframe) IMU-state stream. The CSV writer takes `(path, &[OnlineBaTriggerStats])` and writes one header line plus `(trigger_idx, window_start, window_end, window_kf_offset, vx, vy, vz, bg_x, bg_y, bg_z, ba_x, ba_y, ba_z)` per (trigger, keyframe), with numeric columns formatted via Rust's default `f64` Display (full round-trip precision). Triggers without an `imu_refinement` (visual-only, or refiner `Err`) emit no rows; the absolute frame id for any row is `window_start + window_kf_offset`. `OnlineBaImuStateRow` exposes the same fields as a typed value so callers that want to post-process programmatically don't have to re-parse the CSV. The CLI gains `--online-ba-imu-csv <path>` on `examples/stereo_vo_external_deep_files` (requires `--online-ba`; the validator rejects the flag without it), and the writer is invoked after the `online BA summary` log so the captured trigger history reflects the full run. Two new tests in `pipelines/slam/src/online_stereo_vo_ba.rs::tests` cover (a) `write_online_ba_imu_state_csv_emits_one_row_per_keyframe` — 15-frame +z constant-velocity scene with one zero-accel / zero-gyro IMU sample per inter-frame window emits header + (N rows = number of successful trigger × in-window keyframes), with the first data line matching the `online_ba_imu_state_rows` helper output verbatim, and (b) `write_online_ba_imu_state_csv_skips_triggers_without_imu_refinement` — a visual-only `OnlineStereoVoBa` run produces a header-only file (zero data rows) and the writer's row-count return value is `0`.
- `visloc_io::colmap::write_colmap_text_model_for_3dgs` and the matching `--colmap-export <dir>` / `--colmap-image-prefix <s>` / `--colmap-image-suffix <s>` CLI flags on `examples/stereo_vo_external_deep_files` provide a 3D Gaussian Splatting / NeRF-friendly export path off the existing stereo VO output. The writer takes `(camera, &[Pose], &[FeatureSet], &[Vec<StereoFeature>], image_name: Fn(usize) -> String)` and emits the three COLMAP text files (`cameras.txt`, `images.txt`, `points3D.txt`) under `<dir>`: one shared PINHOLE camera, one image entry per pose (with `world_to_camera` quaternion + translation matching COLMAP's convention, `NAME = <prefix><6-digit-frame-idx><suffix>` so the downstream trainer can resolve `<dataset>/images/<NAME>`), and a sparse `points3D.txt` built by lifting every stereo feature's left-camera `point_cam` through the (refined) `pose.camera_to_world()`. Each landmark gets one `(frame_id, left_keypoint_index)` track entry, and the per-image second line emits `X Y POINT3D_ID` triples (with `-1` for unpaired left keypoints) so COLMAP-aware trainers see a valid 2D↔3D mapping. A new `ColmapExportSummary { frame_count, landmark_count, observation_count }` is returned for logging; `ColmapError` gains an `InvalidExportInput` variant covering input length mismatches. Two new tests in `crates/io/tests/colmap_export.rs` cover (a) `write_colmap_text_model_for_3dgs_round_trips_through_reader` — synthetic 3-frame +z trajectory + 6 stereo landmarks; the existing `read_colmap_text_model` reconstructs the same camera / keyframe / landmark counts and the world-frame landmark positions land at the expected lifted positions, and (b) `write_colmap_text_model_for_3dgs_rejects_length_mismatch` — mismatched `(poses, left_features, stereo_per_frame)` lengths return `ColmapError::InvalidExportInput` instead of writing partial files. The CLI invokes the writer after VO + (optional) BA so refined poses + stereo points are the ones exported. Drop the resulting `<dir>` straight into `nerfstudio ns-train splatfacto --data <dir>` or Inria gaussian-splatting's `convert.py --skip-matching` workflow.
- `OnlineStereoVoBaConfig.imu_input: Option<StereoVoBaImuInput>` lifts the Forster IMU factor stack into the sliding-window online BA path. The wrapper holds a global IMU input spanning the full trajectory (`windows[i]` covers `(kf[i], kf[i+1]]`); on each trigger `run_ba_window(start, end)` slices `windows[start..end - 1]` and rebuilds a per-trigger `StereoVoBaImuInput` (gravity / bias linearisation / weights / fix-first flags are passed through verbatim) before handing it to `refine_stereo_vo_with_ba`. `fix_first_bias` / `fix_first_velocity` therefore pin the *first keyframe of each trailing window* — mirroring the post-process BA semantics where pose 0 of the window is the gauge anchor. The inner `ba_config.imu_input` must be `None`; setting both at once returns a structured `StereoVoBaError::InvalidImuInput` on the next trigger so the wrapper never silently mis-attributes IMU samples. Likewise, an `imu_input.windows.len()` that does not cover the trailing window also returns `StereoVoBaError::InvalidImuInput` instead of panicking. Two new tests in `pipelines/slam/src/online_stereo_vo_ba.rs::tests` cover (a) `online_ba_with_imu_input_refines_velocities` — 15-frame +z constant-velocity scene with one zero-accel / zero-gyro IMU sample per inter-frame window (gravity = 0), at least one trigger fires successfully and the refined per-keyframe velocity inside the trailing window stays within `0.05 m/s` of the truth (0, 0, step/Δt) and (b) `online_ba_imu_input_too_short_returns_invalid_imu_input` — when the wrapper-level `windows` is too short to cover the trailing 5-frame window, the trigger surfaces an `InvalidImuInput` error in the trigger history instead of panicking. `OnlineStereoVoBa::default()` still sets `imu_input: None`, preserving the visual-only sliding BA default.
- `examples/stereo_vo_external_deep_files` extends `--imu-windows-dir` / `--kitti-oxts-dir` to work with `--online-ba` in addition to `--enable-ba`. The CLI now loads IMU windows once at startup (regardless of which BA path is taken), passes them straight into `OnlineStereoVoBaConfig.imu_input` when streaming BA is requested, and into the post-process refiner when one-shot BA is requested. `--online-ba` and `--enable-ba` remain mutually exclusive (one is interleaved with VO, the other runs once after the VO loop finishes). Updated CLI help text accordingly.
- `examples/stereo_vo_external_deep_files` gains `--kitti-oxts-dir <dir>` and `--kitti-image-timestamps <path>` CLI flags wiring the KITTI raw OXTS / IMU loader + slicer directly into the post-process BA path. The new helper `load_imu_windows_from_kitti_oxts(oxts_dir, image_timestamps, frames)` reads every synchronised `oxts/data/*.txt` + `oxts/timestamps.txt` row via `read_kitti_oxts_dir`, lifts each record's `acceleration_body_mps2` / `angular_rate_body_rps` triplets plus wall-clock nanoseconds into parallel `Vec<i128>` / `Vec<Vector3<f64>>` arrays, parses the image-stream `timestamps.txt` (first `--frames` rows used as keyframe times via `parse_kitti_oxts_timestamps_txt`), and calls `slice_imu_samples_for_keyframes` to produce the per-keyframe `Vec<Vec<StereoVoBaImuSample>>` consumed by `StereoVoBaImuInput.windows`. The flag requires `--enable-ba`, conflicts with `--online-ba` (streaming IMU slicing remains future work), and is mutually exclusive with the explicit per-window `--imu-windows-dir` source. A user can therefore drive a KITTI raw VO+IMU run end-to-end with `--features-dir <sp_lg_export> --calib <calib_cam_to_cam.txt> --kitti-oxts-dir <sequence>/oxts --kitti-image-timestamps <sequence>/image_02/timestamps.txt --enable-ba` (plus the existing IMU weight / gravity / bias flags). Two new integration tests in `tests/kitti_oxts_imu_pipeline.rs` cover (a) `kitti_oxts_dir_records_slice_into_keyframe_windows` — a 4-sample OXTS layout sliced over 3 keyframes (0/100/300 ms) into a 1-sample + 2-sample window pair with total integrated `Δt = 0.3 s`, and (b) `kitti_oxts_slice_emits_trailing_zoh_when_last_sample_short` — a 2-sample OXTS layout against a single 200 ms keyframe interval ending past the last sample, verifying the trailing zero-order-hold step extends the last sample's gyro/accel to close the integration interval.
- `visloc_rs` re-exports `parse_kitti_oxts_sample`, `parse_kitti_oxts_timestamp_line`, `parse_kitti_oxts_timestamps_txt`, `read_kitti_oxts_dir`, `KittiOxtsError`, `KittiOxtsRecord`, `KittiOxtsSample`, and `KITTI_OXTS_FIELD_COUNT` from `visloc_io::kitti_imu` so the facade-only callers can build OXTS+timestamp pipelines without reaching into the `io::` namespace.
- `visloc_io::kitti_imu` — KITTI raw OXTS / IMU log loader. `read_kitti_oxts_dir(<sequence>/oxts)` reads every synchronised `oxts/data/<10-digit>.txt` (sorted lexicographically) together with the matching `oxts/timestamps.txt` rows and returns `Vec<KittiOxtsRecord>` carrying wall-clock nanoseconds (parsed from the textual `YYYY-MM-DD HH:MM:SS.fffffffff` format via a private Howard Hinnant `days_from_civil` helper, so the loader is timezone-free and chrono-free) plus a typed `KittiOxtsSample` with the full 30-field row (`lat/lon/alt`, `roll/pitch/yaw`, `vn/ve/vf/vl/vu`, `acceleration_body_mps2 (ax,ay,az)`, `acceleration_nav_mps2 (af,al,au)`, `angular_rate_body_rps (wx,wy,wz)`, `angular_rate_nav_rps (wf,wl,wu)`, `position_accuracy_m`, `velocity_accuracy_mps`, `navigation_status`, `number_of_satellites`, `position_mode`, `velocity_mode`, `orientation_mode`). The body-frame gyro / accel triplets are directly consumable by `ImuPreintegrator::integrate_sample`. Errors (`KittiOxtsError`) distinguish I/O failures, missing `data/` directory or `timestamps.txt`, data/timestamp count mismatch, invalid 30-field rows, and invalid timestamp lines. Free helpers `parse_kitti_oxts_sample(text)`, `parse_kitti_oxts_timestamps_txt(text)`, and `parse_kitti_oxts_timestamp_line(line)` are exposed for in-memory parsing. The loader sits outside the `image-io` feature gate (pure-text) so it is always available. Nine new tests in `crates/io/tests/kitti_imu.rs` cover (a) full 30-field sample parse, (b) comment / blank-line skipping, (c) too-few fields, (d) non-numeric field, (e) timestamp with 9-digit fractional seconds, (f) timestamp without fraction, (g) timestamps.txt with blanks / `#` comments, (h) full `read_kitti_oxts_dir` round-trip on a temp directory, and (i) data/timestamp count-mismatch detection.
- `visloc_slam::slice_imu_samples_for_keyframes` — bucket a globally-timestamped IMU stream (matching `imu_timestamps_ns` / `imu_gyro` / `imu_accel` slices, ns absolute) into the per-keyframe `Vec<Vec<StereoVoBaImuSample>>` layout expected by `StereoVoBaImuInput::windows`. For each keyframe pair `(kf[i], kf[i+1])` the window collects every IMU sample whose timestamp `t` satisfies `kf[i] < t <= kf[i+1]` with `dt` anchored at the preceding timestamp (first sample anchored at `kf[i]`); if the last sample stops short of `kf[i+1]`, a trailing zero-order-hold step extends the last sample's gyro/accel to close the interval, so the integrated `Δt` always matches the inter-keyframe duration when there is at least one sample in the window. Windows without IMU coverage are returned empty, signalling `refine_stereo_vo_with_ba` to silently skip wiring an IMU factor on that segment. Together with `visloc_io::kitti_imu` this lets a KITTI raw recording feed `StereoVoBaImuInput.windows` directly: read the OXTS records and image timestamps, lift `acceleration_body_mps2` / `angular_rate_body_rps` into `Vec<Vector3<f64>>`, then call this slicer with the chosen keyframe timestamps. Three new tests in `pipelines/slam/src/stereo_vo_ba.rs::tests` cover (a) `slice_imu_samples_buckets_by_keyframe_intervals` — mixed-coverage 4-sample / 3-window scene with the total integrated `Δt` matching the keyframe span, (b) `slice_imu_samples_emits_empty_window_when_no_coverage` — a window without IMU samples lands empty (no synthetic ZOH from prior windows), and (c) `slice_imu_samples_validates_lengths_and_monotonicity` — gyro/accel length mismatch, single keyframe, and non-monotonic keyframes each return descriptive errors instead of panicking. Streaming IMU through `OnlineStereoVoBa` and a KITTI raw VO+IMU end-to-end CLI smoke remain follow-up extensions; this slicer is the building block for both.
- `StereoVoBaConfig.imu_input: Option<StereoVoBaImuInput>` and `examples/stereo_vo_external_deep_files` CLI surface for the Forster IMU factor stack. The new `StereoVoBaImuSample { dt, gyro, accel }` and `StereoVoBaImuInput { windows, gravity_world, bias_*_init, weight_*, bias_random_walk_weight, fix_first_bias, fix_first_velocity }` types let the post-process refiner consume per-edge IMU windows: `refine_stereo_vo_with_ba` pre-integrates each window with `ImuPreintegrator::new_with_bias`, registers a velocity + bias slot per IMU-active keyframe (velocity seeded from the inter-keyframe pose-centre delta scaled by the integrated `Δt`, biases at the supplied linearisation point), pushes an `ImuPreintegrationFactor` per non-empty window, and optionally ties consecutive biases with a `BiasRandomWalkFactor` of weight `bias_random_walk_weight`. The first IMU-active keyframe's bias is gauge-fixed by default (`fix_first_bias = true`); the first velocity is left free unless `fix_first_velocity = true`. Sliding-window BA (`config.window_size`) is rejected with a structured `StereoVoBaError::InvalidImuInput` because the window slicer doesn't yet slice IMU samples. The refinement now returns `imu_refinement: Option<StereoVoBaImuRefinement { refined_velocities, refined_bias_gyro, refined_bias_acc }>`, with non-IMU keyframes carrying the linearisation bias and zero velocity defaults. A new free function `parse_stereo_vo_imu_samples_txt` parses one whitespace-separated `dt gyro_x gyro_y gyro_z accel_x accel_y accel_z` per line (gravity NOT pre-subtracted from accel; `#` comments and blank lines tolerated). The CLI surface adds `--imu-windows-dir <dir>` (loads `frame_NNNNNN_imu.txt`, where `NNNNNN` is the "to" keyframe of the window — mirroring the existing `frame_NNNNNN_temporal_matches.txt` convention; missing files are treated as empty windows so partial coverage is fine), `--imu-gravity gx,gy,gz` (default `0,9.81,0` KITTI y-down), `--imu-weight-{position,velocity,rotation}` (default `1.0`), `--imu-bias-{gyro,acc}-init x,y,z`, `--imu-bias-random-walk-weight`, `--imu-fix-first-bias on|off` (default on), and `--imu-fix-first-velocity on|off` (default off). The flag requires `--enable-ba` and conflicts with `--online-ba` (streaming IMU slicing is a future extension); on success the CLI writes `<out>/ba_imu_state.csv` with `(id, vx, vy, vz, bg_x, bg_y, bg_z, ba_x, ba_y, ba_z)` per keyframe and prints the last-keyframe refined velocity. Four new tests in `pipelines/slam/src/stereo_vo_ba.rs::tests` cover (a) `ba_with_imu_input_wires_through_config` — full wiring on a 3-frame +2 m/s constant-velocity scene recovers refined per-keyframe velocity within `0.05 m/s` of truth and leaves poses within `5 mm` of input, (b) `ba_with_imu_input_validates_window_count_and_sliding_window` — wrong window count and `window_size` + IMU each return `StereoVoBaError::InvalidImuInput { reason }` instead of panicking, (c) `imu_samples_txt_round_trips_through_pre_integration` — parser accepts comments / blank lines and preserves order, and (d) `imu_samples_txt_rejects_bad_lines` — wrong column count, non-positive `dt`, and unparseable tokens each produce descriptive errors. Strengthening gyro-bias observability scenarios and wiring the IMU factor stack through `OnlineStereoVoBa` remain follow-up extensions.
- `visloc_slam::BiasRandomWalkFactor` — per-edge bias random-walk prior between two keyframes' 6-vector IMU biases. The factor adds the cost `weight · ‖b_j − b_i‖²` with the linear Jacobian `J_i = −I`, `J_j = +I` against the bias slots; a typical weight is `1 / (σ_bw² · Δt_{ij})` where `σ_bw` is the gyro / accel bias random-walk noise density and `Δt_{ij}` is the inter-keyframe time. `BundleAdjustment` gains `bias_random_walk_factors: Vec<BiasRandomWalkFactor>` and `add_bias_random_walk_factor` to wire it; the bias-slot index now harvests both endpoints of every random-walk factor in addition to the "from" side of every IMU factor, so a keyframe whose bias is unobservable through its own IMU window (e.g., gyro bias on a straight trajectory) can still get a slot via the random-walk tie. Three new tests in `pipelines/slam/tests/bundle_adjustment.rs` cover (a) zero cost at truth on a 10↔20 random-walk tie sitting on top of an IMU factor, (b) the random-walk pull alone bringing two non-fixed bias slots together (with no IMU factor), and (c) the random-walk factor propagating an observable accel bias from KF10 (anchored by its IMU factor) to KF20 (which has no factor of its own) so both bias estimates converge within `1e-2` of the hidden truth.
- Forster 2017 first-order bias-Jacobian propagation (eq. 35-39) and BA-side per-keyframe bias state, completing the Forster ImuFactor stack. `ImuPreintegratedDelta` now carries `bias_gyro_linearisation` / `bias_acc_linearisation` plus the five 3×3 Jacobians `j_rotation_bg`, `j_velocity_ba`, `j_velocity_bg`, `j_position_ba`, `j_position_bg`, all propagated through `ImuPreintegrator::integrate_sample` with a private `right_jacobian_so3(φ) = I − (1−cos θ)/θ² · [φ]× + (θ−sin θ)/θ³ · [φ]×²` helper (Taylor-fallback `I − ½[φ]× + (1/6)[φ]×²` near `θ = 0`). `ImuPreintegratedDelta::corrected(b_g, b_a)` returns the first-order corrected `(ΔR, Δv, Δp)` and `ImuPreintegrationFactor::residual_with_bias_correction` lifts that into the Forster residual without re-integrating. On the BA side, `BundleAdjustment` grows `biases: BTreeMap<u64, Vector6<f64>>` (gyro 0..3, accel 3..6) and `fixed_biases: BTreeSet<u64>` with `add_bias` / `fix_bias` hooks; when a non-fixed bias is registered on the "from" keyframe of an IMU factor, the Schur-reduced linear system layout extends to `(6P + 3V + 6B) × (6P + 3V + 6B)` and the optimiser adds a 9×6 bias Jacobian column via `∂r_R/∂δb_g = −Jr⁻¹(r_R) · Exp(−r_R) · J_R_bg` (Forster eq. 159, simplified by dropping the `Jr(J_R·δb)` factor that is ≈I in the small-bias regime), `∂r_v/∂δb_g = −J_v_bg`, `∂r_v/∂δb_a = −J_v_ba`, `∂r_p/∂δb_g = −J_p_bg`, `∂r_p/∂δb_a = −J_p_ba`. All 5 cross-blocks (pose_i / pose_j / vel_i / vel_j with bias_i, plus the bias_i diagonal) are accumulated. When `B = 0` the matrix layout falls back bit-identical to the post-task-#26 pose+velocity system, so every reprojection / pairwise / IMU-only test continues to pass. Five new tests: (a) `bias_jacobians_match_finite_difference` numerically verifies the corrected delta matches a re-integrated delta to `< 1e-4` for `|δb| ~ 1e-3` motion, (b) `residual_with_bias_correction_matches_residual_at_linearisation_point` confirms the bias-correction path is a no-op at the linearisation point, (c) `imu_bias_zero_cost_at_truth` confirms `cost()` stays zero with a registered zero bias, (d) `imu_bias_recovers_hidden_accel_bias` recovers a hidden `+0.5 m/s²` accel bias from an inconsistent pre-integrated delta with poses + velocities fixed, and (e) `imu_bias_fixed_acts_as_correction_only` shows that a fixed bias still feeds the residual correction without adding a DoF. Bias random-walk priors between consecutive keyframes and gyro-bias observability scenarios are left as follow-up extensions.
- `BundleAdjustment` now jointly optimises per-keyframe world-frame velocity together with pose, lifting the IMU pre-integration scaffold into a full Forster ImuFactor inside the BA solver. `velocities: BTreeMap<u64, Vector3<f64>>` / `fixed_velocities: BTreeSet<u64>` / `imu_factors: Vec<ImuPreintegrationFactor>` ship the new state, and `add_velocity` / `fix_velocity` / `add_imu_factor` are the construction-side hooks. The Schur-reduced linear system grows from `(6P) × (6P)` to `((6P + 3V)) × ((6P + 3V))` when at least one IMU factor binds two velocities (so when `V = 0` the legacy reprojection-only layout is bit-identical, preserving every existing test). The Forster 2017 `[r_R; r_v; r_p]` 9-vector residual is linearised with all six analytical right-perturbation Jacobian blocks: `∂r_R/∂ω = ±Jr⁻¹(r_R)·R_wcⱼ`, `∂r_v/∂ω_i = −R_wcᵢ·[v_j−v_i−g·Δt]×`, `∂r_v/∂v_i / ∂v_j = ∓R_wcᵢ`, `∂r_p/∂ρ_i / ∂ρ_j = ±R_wcᵢ`, `∂r_p/∂ω_i = −R_wcᵢ·[C_j−v_i·Δt−½g·Δt²]×`, `∂r_p/∂ω_j = R_wcᵢ·[C_j]×`, `∂r_p/∂v_i = −Δt·R_wcᵢ`. A private `right_jacobian_inverse_so3(φ) = I + ½[φ]× + c·[φ]×²` helper (with the same `c = (1/θ²)(1 − (θ/2)cot(θ/2))` formula as `visloc_core::geometry::se3::so3_left_jacobian_inverse`) provides Forster eq. 8's `Jr⁻¹` so the rotation residual stays linearly accurate past the small-angle regime. Three new integration tests in `pipelines/slam/tests/bundle_adjustment.rs` cover (a) zero IMU cost on a constant-accel truth configuration, (b) a single IMU factor pulling a 1 m/s `v_0` drift back to truth with both poses + `v_1` fixed, and (c) BA + two IMU factors pulling a 0.2 m lateral pose drift on the middle of a 3-keyframe constant-velocity scene back to truth. Bias state and Forster's first-order bias-Jacobian propagation (eq. 40-44) remain deferred — `ImuPreintegrator::new_with_bias` still treats bias as a fixed input.
- `visloc_slam::imu_preintegration` — Forster 2017 on-manifold IMU pre-integration scaffold. `ImuPreintegrator` accumulates body-frame `(gyro, accel)` samples into an `ImuPreintegratedDelta { delta_rotation, delta_velocity, delta_position, delta_time }` in keyframe-`i`'s body frame, gravity-free. `ImuPreintegrationFactor` ships the gravity-compensated 9-vector residual `[r_R; r_v; r_p]` for the BA-side hook-up (Forster eq. 45-47): `r_R = log(ΔR.T · R_iᵀ · R_j)`, `r_v = R_iᵀ(v_j − v_i − g·Δt) − Δv`, `r_p = R_iᵀ(p_j − p_i − v_i·Δt − 0.5·g·Δt²) − Δp`. Seven new unit tests cover (a) zero motion → identity delta, (b) constant 2 m/s² body accel for 1 s → analytical Δv / Δp, (c) constant π/2 rad/s yaw rate for 1 s → analytical ΔR, (d) constant bias cancelled by `new_with_bias`, (e) rotated-then-accelerated body frame producing a world-z velocity, (f) residual vanishing at a consistent gravity-free state, and (g) residual vanishing under pure-gravity motion (so the compensation terms are wired correctly). BA-side velocity / bias state and the residual Jacobians (full Forster ImuFactor) are deferred — this scaffold lets a follow-up task hand the BA solver a verified pre-integration primitive.
- `StereoVoBaConfig.gravity_prior: Option<GravityPrior>` wires the rotation-alignment gravity prior into the stereo VO BA refiner; on the file-backed VO example a new `--ba-gravity-prior-weight <w>` CLI flag (weight `0` disables, positive `w` enables a KITTI-style level-world prior `g_world = g_camera_observed = (0, 9.81, 0)`). The prior applies on every window of the sliding-window path too, so it stacks with `OnlineStereoVoBa`. A new wiring test (`ba_with_gravity_prior_wires_through_config`) confirms the prior is accepted by config and the refinement still converges at parity with the no-prior baseline on a well-conditioned synthetic stereo scene. Deeper "prior actually corrects rotation drift" coverage continues to live in `bundle_adjustment.rs::gravity_prior_recovers_pitched_pose` on under-constrained mono bundles, plus the new stacked-factor integration test below.
- Stacked-factor integration test (`stacked_factors_correct_independent_drift_modes`) drives the full BA factor stack — reprojection + `GravityPrior` + `PositionPrior` (altitude-only) + `PairwisePoseFactor` (loop-closure shape) — on a single 4-keyframe KITTI-style bundle. Each non-anchor pose is injected with a DIFFERENT failure mode that only one of the three priors can fix (pitch / vertical translation / yaw + lateral translation), and BA converges with all priors stacked. This is the foundational check before the task #24 end-to-end real-data smoke: it pins down that the four factor types do not interfere with each other inside the same LM solve.
- `LoopClosureConstraint::to_pairwise_pose_factor(weight)` and `pairwise_pose_factors_from_loop_closures(&[…], weight)` lift verified loop-closure edges into BA-ready `PairwisePoseFactor`s. The constraint's `relative_pose` is reused verbatim (already in `T_to · T_fromⁱ` convention), so a verified loop now feeds straight into a unified BA solve alongside reprojection + `GravityPrior` + `PositionPrior` — no separate post-VO pose-graph stage required when the BA backend is preferred. Two new tests cover (a) the adapter preserving ids / SE(3) measurement / weight and dropping verifier metadata, and (b) end-to-end on a 3-keyframe bundle: KF40 is dragged in translation, the truth relative pose `KF10→KF40` is fed as a verified `LoopClosureConstraint`, and BA pulls the drift to <10 % of its input magnitude.
- `visloc_slam::PairwisePoseFactor` — Kimera-VIO / GTSAM-style relative-pose factor inside `BundleAdjustment`. Takes a `measurement: Pose` between two BA keyframes and a scalar `weight = 1/σ²`; cost is `weight · ‖log(meas⁻¹ · T_to · T_fromⁱ)‖²`. Right-perturbation Jacobians follow the existing `PoseGraph::optimize_se3_iterative` template (`∂r/∂δ_to = Ad(T_from)`, `∂r/∂δ_from = −Ad(T_from)`) and write both diagonal `±AdᵀAd` Hessian blocks plus the cross-pose `−AdᵀAd / −(AdᵀAd)ᵀ` off-diagonals. This is the v1 entry point for lifting IMU pre-integration deltas, wheel-odometry ticks, or verified loop-closure edges into the unified BA solve alongside reprojection + `GravityPrior` + `PositionPrior`. Three new unit tests cover (a) zero cost at truth, (b) the factor pulling a translation-drifted keyframe back on its own, and (c) the rotation half of the Jacobian recovering a yaw drift. Full IMU pre-integration with velocity / bias states is deferred — non-GPL OSS (Kimera-VIO BSD-2, Forster 2017 paper) provide the template for the velocity / bias factor extension.
- Real-data loop-closure detection validated on KITTI 00 long-revisit segments. `examples/kitti_revisit_scanner_demo --frontend classical` over the 30-frame start subset (frames 0-29) + the 30-frame revisit subset (frames 4500-4529) detects **8 cross-segment loop candidates** with the geometric verifier accepting all of them; strongest pair is `(KF 12, KF 4527)` with `25 inliers`, `inlier_ratio = 0.500`, `mean_sampson_error = 0.00057`, `score = 21997`. The candidate cluster `(12-18, 4511-4527)` corresponds to the actual KITTI 00 trajectory overlap where the vehicle returns near its start position. Confirms the pairwise scanner + essential-matrix verifier path works end-to-end on real driving data and is not limited to synthetic fixtures.
- `visloc_slam::PositionPrior` / `PositionPriorObservation` — per-keyframe absolute camera-centre prior on `BundleAdjustment`. Each observation contributes an axis-weighted residual `(C_w − target)` with Jacobian `J = [−I | [C_w]_×]` under the right-perturbation xi-order `[ρ; ω]`. The per-axis `axis_weights: Vector3<f64>` collapses to zero on any axis with zero weight, so an altitude-only GNSS / GT prior is expressed as `axis_weights = (0, w, 0)`. Three new unit tests cover (a) truth-trajectory zero cost, (b) BA + altitude-only prior recovering pure-vertical drift (the seq08-shaped failure mode that `GravityPrior` documented as out of scope), and (c) zero-weight axes adding no cost or Jacobian rows. Together with `GravityPrior`, `BundleAdjustment` now spans both rotation and translation domain priors.
- `--online-ba`, `--online-ba-window`, `--online-ba-trigger-every` CLI flags on `examples/stereo_vo_external_deep_files` wire the `OnlineStereoVoBa` wrapper into the file-backed deep VO pipeline. On the local 900-frame KITTI 00 stride-1 subset, online BA (`window=30, trigger_every=10`) reaches **`t_rel = 1.4590 %`**, **`max_t_rel = 3.7561 %`** — a -41.8 % / -38.4 % reduction vs no-BA (`2.5074 %` / `6.0994 %`) and -28.6 % / -25.1 % vs the single-shot post-process BA (`2.0432 %` / `5.0180 %`). The interleaved sliding-window approach beats global post-process BA on this scale because each trigger sees a fresh boundary anchor and avoids the local minimum that the full-trajectory drift drags the global solver into. 88 triggers fired across the run, refining 50,518 aggregate tracks / 263,434 observations.
- `visloc_slam::OnlineStereoVoBa` — sliding-window stereo VO + BA composition. Wraps `StereoVoFrontend` and triggers `refine_stereo_vo_with_ba` every `trigger_every_frames` processed pairs over the trailing `window_size` frames, writing refined poses back into the frontend. `StereoVoFrontend` gained a `temporal_matches_per_pair: Vec<Vec<DescriptorMatch>>` field that captures the filtered temporal matches actually used, so downstream BA refiners do not need to re-run the matcher. Manual `run_ba_now()` available for one-shot post-process. Two new unit tests cover (a) clean-trajectory stability across triggers and (b) the `trigger_every_frames = 0` disabled-path with manual override. Keeps BA outside `visloc-vision` to avoid a vision → slam dependency cycle.
- `visloc_slam::GravityPrior` — rotation-alignment gravity prior for `BundleAdjustment`. Adds a 3-vector residual `R_wc · g_world − g_camera_observed` per non-fixed pose with a configurable `weight`, implemented as a non-robust L2 contribution to `cost()` / `robust_cost()` and a `J^T J / J^T r` block in `build_normal_equations` (right-perturbation Jacobian `J = [0_3×3 | −R · [g_w]_×]`, xi-order `[ρ; ω]`). Three new unit tests cover (a) zero cost on consistent trajectory, (b) BA + prior recovering a pitched pose injected on a 3-frame mono bundle, and (c) the documented limitation that pure-translation drift is unaffected by a rotation-only prior. The seq08 vertical-translation bias (per-pair `Δy ≈ +0.176 m`, with rotation already matching GT per the pre-BA diagnostic) is therefore NOT addressable by this prior alone — a translation/altitude prior fed from IMU velocity or GNSS is required and is left for future work.
- Long-sequence KITTI 00 stereo VO benchmark on the local 900-frame stride-1 stereo subset. With the same SP/LG export and the v0.5 BA refiner (`--ba-max-init-residual 3 --ba-min-track-count 2000 --ba-huber-delta 3`), aggregate `t_rel` drops from `2.5074 %` (no BA) to **`2.0432 %`** (-18.5 %) and `max_t_rel` from `6.0994 %` to **`5.0180 %`** (-17.7 %) across all 900 frames. BA refined 12,400 tracks / 54,608 observations in 11 LM iterations on this slice. Drift is substantially larger than on the 260-frame triage subset, as expected for VO without loop closure; the BA refiner remains beneficial at this scale.
- `visloc_slam::refine_stereo_vo_with_ba` — multi-frame Schur BA refiner for stereo VO trajectories. The new module (`pipelines/slam/src/stereo_vo_ba.rs`) builds forward feature tracks by chaining per-pair temporal matches, initialises each landmark from its first stereo observation, and runs sparse-Cholesky LM BA over all poses (pose 0 fixed) with a Huber kernel. `StereoVoBaConfig` exposes the full filter set: `min_track_length`, `max_initial_depth_m`, `max_seed_row_fraction`, `max_init_residual_px` (per-track pre-BA reprojection gate that rejects pseudo-tracks), `min_temporal_confidence`, `min_track_count` (auto-skip threshold), `landmark_init` (`StereoSingleFrame` / `MultiViewDlt`), and `window_size` (sliding-window BA — per-window joint optimisation instead of global). `StereoVoBaError` cleanly distinguishes `TooFewFrames`, `InputLengthMismatch`, `NoLongTracks`, `InsufficientTracks { count, required }`, and `Ba(BaError)`. Three new unit tests cover round-trip stability, 5 cm drift correction, and multi-view DLT exact recovery on synthetic data. On the local 00-10 / 260-frame SP/LG benchmark this drops the aggregate `mean_t_rel` from 1.4685 % (tuned SP/LG, no BA) to **1.3403 %** (-8.7 %) and `mean_max_t_rel` from 3.4228 % to **3.1354 %** (-8.4 %); 10 of 11 sequences now beat the HOG/MutualSoftmax reference and `seq03` is within 0.011 pp.
- `examples/stereo_vo_external_deep_files` gains 11 `--ba-*` CLI flags wiring the new refiner: `--enable-ba`, `--ba-min-track-length`, `--ba-max-initial-depth`, `--ba-max-iterations`, `--ba-huber-delta`, `--ba-max-seed-row-fraction`, `--ba-max-init-residual`, `--ba-min-temporal-confidence`, `--ba-min-track-count`, `--ba-landmark-init`, and `--ba-window-size`.
- `scripts/run_kitti_superpoint_lightglue_vo_train_benchmark.sh` gains BA support via `--enable-ba`, `--ba-max-init-residual`, `--ba-min-track-count`, `--ba-huber-delta`, and per-seq `--ba-overrides <SEQ:directive,...>` where directives are `resid=<float>`, `tracks=<int>`, `win=<int>`, `huber=<float>`, or `skip`.
- KITTI odometry leaderboard-style relative-motion evaluation is now available through `PoseTrajectory::kitti_odometry_benchmark_against` and `examples/evaluate_kitti_odometry_benchmark.rs`. The evaluator matches KITTI pose rows by frame id, builds reference path-length windows, and reports mean/max `t_rel` (%) plus mean/max `r_rel` (deg/m) over configurable segment lengths. The default now follows KITTI's current public benchmark lengths (`100,200,...,800` m), and `--lengths` can override it for short development windows. The stereo KITTI demo now also exports `vo_poses.txt`, `ba_poses.txt`, `pgo_poses.txt`, `gt_poses.txt`, and `relative_pose_errors.csv` so those trajectories can be evaluated directly and per-frame translation/rotation drift can be inspected. On the README 50-frame KITTI 00 slice (`gt_length = 45.70 m`), short development windows (`--lengths 5,10,20,40`) report raw deep VO `t_rel = 3.41%`, `r_rel = 0.0447 deg/m`. A longer 260-frame seq00 subset (`gt_length = 183.11 m`) reports raw deep VO mean/max ATE 2.49 / 4.18 m and current-public-length relative errors `t_rel = 0.675%`, `r_rel = 0.0146 deg/m` over 97 windows with the leaderboard-oriented 3.32 px PnP reprojection gate, guarded PnP refinement, high-consensus PnP early-stop, and lazy Kabsch fallback; the older short-window development set (`--lengths 5,10,50,100,150,200,250,300,350,400`) reports `t_rel = 1.97%`, `r_rel = 0.0438 deg/m` over 816 windows on the same run.
- `scripts/run_kitti_deep_vo_smoke.sh` now automates the long KITTI deep stereo VO smoke path: fetch/reuse the stride-1 seq00 stereo subset, run `online_slam_stereo_vo_kitti_demo --frontend deep` with the reproducible 260-frame settings, evaluate public KITTI odometry lengths plus 100 m-only windows, and write `deep_vo_smoke_summary.txt` alongside the raw `summary.txt` and evaluator JSON/CSV exports. It also runs the new `scripts/visual_slam_debug_report.py` helper, which combines `frontend_pair_diagnostics.csv`, `relative_pose_errors.csv`, and KITTI segment errors into `slam_debug_report.md`, `slam_debug_report.html`, `slam_debug_summary.json`, and `slam_debug_worst_pairs.csv` for fast VO/SLAM triage. The helper accepts `--compare <baseline-run-dir>` to emit `slam_debug_compare.{md,html,json}` plus `slam_debug_compare_metrics.csv`, including ATE/KITTI/source-count deltas and worst KITTI segment deltas between runs.
- `scripts/run_kitti_deep_vo_train_benchmark.sh` runs that deep stereo VO smoke path across KITTI odometry training sequences 00-10 and writes a consolidated `summary.csv` / `summary.md` with ATE, public-length `t_rel` / `r_rel`, fallback counts, worst relative-pose pairs, worst KITTI segments, and per-sequence visual-SLAM debug report links. It also accepts `--compare-root <old-benchmark-root>` to emit per-sequence `slam_debug_compare` artifacts and link them from the benchmark summary. The fetch helper and smoke script now accept `--sequence`, so sequence-level triage no longer requires seq00-specific path rewrites.
- `scripts/run_kitti_deep_vo_revisit_smoke.sh` automates the KITTI 00 revisit scanner smoke path: fetch/reuse the 50-frame start slice and 30-frame revisit slice around frame 4500, run `kitti_revisit_scanner_demo --frontend <classical|deep|deep-ms|both>`, require that a strongest loop pair is reported, and write `deep_revisit_smoke_summary.txt` next to the scanner's raw `summary.txt`.
- `scripts/run_kitti_deep_stack_smoke.sh` is the single-command deep KITTI stack gate: it runs the long deep stereo VO smoke and the revisit scanner smoke into `target/kitti_deep_stack_smoke/{vo,revisit}`, checks that VO ATE, relative-pose diagnostics, KITTI segment counts, and strongest revisit loop ids are present, and emits both `deep_stack_smoke_summary.txt` and a machine-readable `deep_stack_smoke_summary.json` with ATE, per-frame relative-pose diagnostics, mean/max KITTI relative-motion, and strongest revisit-loop metrics.
- Stereo VO now prefers confidence-weighted 2D-3D PnP RANSAC for consecutive-frame motion and keeps 3D-3D Kabsch RANSAC as a fallback. The PnP path reuses the previous frame's stereo-triangulated 3D points and the current frame's 2D keypoints, so the deep frontend no longer compounds stereo-depth noise from both frames in the primary motion estimate. On the README KITTI 00 50-frame deep run this improves raw VO from mean/max ATE 2.79 / 5.97 m to 1.32 / 2.01 m.
- PnP RANSAC now guards the Gauss-Newton refinement step: the refined pose is kept only when it preserves or improves the reprojection consensus, otherwise the least-squares PnP pose estimated from the best inlier set is retained. On the 260-frame KITTI 00 deep VO smoke this improves current-public-length relative errors from `t_rel = 0.986%`, `r_rel = 0.0142 deg/m` to `t_rel = 0.896%`, `r_rel = 0.0125 deg/m`.
- Stereo VO now enables high-consensus PnP early-stop and runs the Kabsch fallback lazily. The pathological KITTI 00 pair around original frames 271→272 now completes with the full 1500-feature / 1000-iteration setting in about 0.5 s, and a 300-frame smoke completes with all 299 pairs sourced from PnP (`t_rel = 1.271%`, `r_rel = 0.0140 deg/m` over current public lengths).
- Stereo VO now has an adaptive 60 m PnP depth rescue path. The frontend keeps the normal uncapped PnP estimate as the primary candidate, but when its inlier ratio falls below `0.65` or primary PnP fails, it also evaluates a 60 m depth-limited candidate and selects by full-correspondence reprojection consensus. This preserves the 260-frame README result unchanged (`t_rel = 0.675%`, `r_rel = 0.0146 deg/m`) while improving the 900-frame local seq00 smoke from `t_rel = 2.4255%`, `r_rel = 0.01666 deg/m`, 4 Kabsch fallbacks to `t_rel = 2.3232%`, `r_rel = 0.01584 deg/m`, 0 Kabsch fallbacks.
- Stereo VO now includes conservative motion-scale band rescue, p75 scale-target rescue, translation-direction rescue, rotation-spike rescue, automatic stereo translation refinement, and rotation-vector rescue for fast motion. Motion-scale rescue detects per-frame translation outside a recent-median band only after a fast-motion history is established, keeps the estimated direction and rotation, and rescales the translation magnitude to the recent p75 target to reduce acceleration lag; its collapse gate now catches moderate collapses below `0.97x` of the recent median. Translation-direction rescue keeps the estimated translation magnitude but clamps weak-consensus lateral outliers back to the recent average direction when they deviate by more than `10 deg`. Rotation-spike rescue clamps weak-consensus rotation angles that exceed the recent median by a large ratio; automatic stereo translation refinement stays gated by fast-motion history (`recent median >= 1.5 m`) but is eligible across PnP consensus levels in those highway windows. Rotation-vector rescue then clamps weak-consensus rotation-vector outliers back to the recent fast-motion trend, targeting the residual highway yaw drift that remains after translation rescue. The seq00 260-frame guardrail remains unchanged (`t_rel = 0.675%`, `r_rel = 0.0146 deg/m`), while the seq01 260-frame highway slice improves from `t_rel = 20.230%`, `r_rel = 0.0313 deg/m` to `t_rel = 3.845%`, `r_rel = 0.0124 deg/m`.
- Motion-scale rescue is tuned for the seq01 highway collapse regime with a stricter `0.97x` translation-ratio gate, and the KITTI smoke/benchmark scripts expose `--motion-scale-rescue-min-translation-ratio`, `--rotation-vector-rescue-min-history`, and `--rotation-vector-rescue-max-delta-deg` for future sweeps. On the 260-frame seq01 subset this improves current-public-length `t_rel` from `4.637%` at the previous `0.94x` gate to `4.552%`; fast-motion auto stereo refinement brings it to `4.314%` with max `t_rel = 9.844%`, the first rotation-vector rescue pass brings it to `4.090%` with max `t_rel = 6.689%`, and the tightened `0.40 deg` rotation-vector trigger brings it to `3.845%` with max `t_rel = 6.015%`. The 100 m-only mean improves from `4.101%` to `3.506%`, and the prior worst `178->216` highway segment drops from `9.844%` to about `4.829%`; seq00 remains unchanged at `t_rel = 0.675%`, `r_rel = 0.0146 deg/m`.
- Motion-scale rescue now allows high-consensus PnP pairs through the fast-motion scale guard (`motion_scale_rescue_max_pnp_inlier_ratio = 1.05`) while still requiring a recent median translation of at least `1.5 m`. This targets seq01's remaining highway failure mode: a clean-looking run of PnP pairs around frames `110->154` that was consistently short rather than a single weak-consensus outlier. On the 260-frame seq01 subset, current-public-length `t_rel` improves from `3.845%` to `3.776%`, max `t_rel` from `6.015%` to `5.934%`, and 100 m-only mean/max from `3.506% / 6.015%` to `3.428% / 5.614%`; `r_rel` is unchanged at `0.0124 deg/m`. Guard runs confirm seq00 remains unchanged at `0.675% / 0.0146 deg/m`, and seq02 remains unchanged at `1.156% / 0.00759 deg/m`.
- The local 260-frame KITTI training sweep now covers sequences 00-10 with the current default deep stereo VO settings and reports 11/11 successful runs in `target/kitti_deep_vo_train_benchmark_fast_scale_guard/summary.md`. The full local sweep changes the current limiter from seq01 to seq08: seq08 reports public-length `t_rel = 4.576%`, `r_rel = 0.0124 deg/m`, ATE mean/RMSE/max `12.14 / 12.85 / 17.66 m`, and a worst 100 m segment at `0->143` with `t_rel = 14.933%`. Seq01 remains improved at `3.776% / 0.0124`, while seq10 is the best translational sequence at `0.440% / 0.00867`. The benchmark and smoke scripts also normalize zero-padded sequence ids safely, so `08` and `09` no longer trip POSIX `printf`'s octal parsing.
- Local KITTI training triage now covers the repaired 260-frame seq02 subset in addition to seq00/seq01. The seq02 subset had one zero-byte right-camera frame (`image_1/000071.png`); removing it and rerunning the fetch helper repaired the dataset. With the current deep stereo VO defaults, seq02 reports `t_rel = 1.156%`, `r_rel = 0.00759 deg/m`, max `t_rel = 2.357%`, ATE mean/RMSE/max `2.87 / 3.21 / 5.21 m`, and no Kabsch fallbacks. Across the currently local 00/01/02 smoke set, seq01 remains the limiting sequence by translational error.
- README's KITTI VO GIF/static asset now uses the current 260-frame deep-frontend metric stereo VO smoke output (`vo.csv`, `gt.csv`) instead of the older 50-frame BA/PGO hero. `scripts/build_kitti_loop_asset.py --mode stereo-vo --frontend-label "deep stereo VO"` renders a two-panel trajectory + ATE animation: raw deep stereo VO mean/RMSE/max ATE 2.49 / 2.63 / 4.18 m, with KITTI-style 100 m-window `t_rel = 0.675%` and `r_rel = 0.0146 deg/m` on the local public seq00 subset. The older loop-closure asset path remains available through `--mode stereo` for BA/PGO visualization.
- Query-side localization now preserves matcher confidence through the full PnP path: `Correspondence2D3D` carries `confidence: Option<f32>`, `CorrespondenceBuilder` copies `DescriptorMatch.confidence` into each 2D-3D correspondence, and `LocalizationPipeline` calls `RobustPoseEstimator::estimate_with_weights` whenever at least one valid confidence is present. Classical matchers still leave confidence as `None`, so they keep the existing unweighted PnP RANSAC behaviour.
- `examples/deep_localization_demo.rs` now includes a `deep-ms` frontend that anchors `MultiScaleDeepExtractor<HogLikeFeatureExtractor>` descriptors at each COLMAP keypoint across three octaves. Since `VisualMap` stores one descriptor per landmark, the demo expands each map landmark into per-octave descriptor copies at the same 3D position and duplicates query keypoints per octave, letting `MutualSoftmaxMatcher` choose octave-compatible 2D-3D correspondences while preserving the same COLMAP keypoint detector for the single-scale and multi-scale comparisons.
- `examples/deep_localization_demo.rs` extended with a `--sweep` mode that runs the classical, deep, and deep-ms pipelines over a fixed 5×5 grid of (map, query) pairs from COLMAP South Building (map ∈ {P1180141..P1180145}, file gap ∈ {1..5} → 25 pairs × 3 frontends = 75 pipeline runs), records `match_count`, `inlier_count`, `inlier_ratio`, `translation_error_m`, `rotation_error_rad`, `reprojection_error_px` per run, and writes `pairs.csv` + `summary_by_gap.txt` to `--out-dir`. The updated sweep is runnable end-to-end on a freshly downloaded South Building dataset:

  ```bash
  cargo run --release --features image-io --example deep_localization_demo -- \
      --root ~/datasets/south-building/south-building --sweep \
      --out-dir target/deep_localization_sweep
  ```

  Aggregated by viewpoint distance (n=5 pairs per row, success_rate=1.00 in every cell — the pipeline returns a pose even on near-degenerate 2-3 inlier consensus, so per-gap accuracy must be read together with `mean_inliers`):

  | gap (file delta) | classical mean inliers | deep mean inliers | Δ inliers | classical mean transl err (m) | deep mean transl err (m) |
  | ---: | ---: | ---: | ---: | ---: | ---: |
  | 1 | 426.4 | **584.2** | **+37 %** | 0.0115 | 0.0115 |
  | 2 | 179.4 | **278.6** | **+55 %** | 0.0111 | 0.0110 |
  | 3 | 77.8 | **143.8** | **+85 %** | 0.0654 | **0.0103** |
  | 4 | 57.8 | **107.0** | **+85 %** | 0.4854 | 0.8234 |
  | 5 | 53.8 | **106.4** | **+98 %** | 0.9892 | 1.1300 |

  The headline finding: **the deep frontend's verified-inlier advantage grows monotonically with viewpoint distance** (+37 % → +98 % as gap goes 1 → 5), because dual-softmax + 16×16 HOG is more discriminating across descriptor-space drift than 19×19 Corner patches against a brute-force ratio matcher. Pose accuracy is comparable at gap 1-2 (~11 mm for both), but at gap 3 deep keeps the translation error at **1.0 cm** while classical degrades to 6.5 cm because one of its five gap-3 pairs (`P1180143 → P1180146`) collapses to 5 inliers and a 0.28 m error — deep retains 48 inliers on the same pair. At gap 4-5 both pipelines hit pairs where the descriptor type itself runs out (e.g. `P1180141 → P1180146`, `P1180145 → P1180150` collapse to 2-3 inliers on both sides), producing pose-estimator outputs that are technically valid but geometrically meaningless; the inflated mean translation errors reflect those handful of catastrophic pairs rather than a steady degradation. The inlier-count column is a more honest progress signal than success rate at this baseline regime, and confirms the deep advantage is largest exactly where it matters: cross-viewpoint queries that classical struggles with.
- Deep frontend reaches the query-side localization path: `RobustPoseEstimator::estimate_with_weights(correspondences, camera, weights)` (default impl falls back to unweighted `estimate`), and `PnPRansac` overrides it with a PROSAC-style sampler — sort indices by descending weight, expand the sampling subset linearly from `sample_size` to `n` over the iteration budget, so high-confidence correspondences anchor early iterations. Two new tests cover (a) `weighted_pnp_ransac_recovers_pose_with_outlier_heavy_input`: 12 inliers (weight 0.9) + 24 random outliers (weight 0.05), recovers ≥11/12 geometric inliers; (b) `weighted_pnp_ransac_falls_back_to_uniform_when_weight_length_mismatches`: degenerate weights yield bit-identical results to the uniform path.
- New `CornerFeatureExtractor::describe_at(image, x, y) -> Option<Vec<f32>>` and `HogLikeFeatureExtractor::describe_at(image, cx, cy) -> Option<Vec<f32>>` accessors expose each extractor's per-keypoint descriptor at a caller-specified location, skipping the corner detector. Both return `None` when the centre is too close to the image border for a full descriptor patch, so the call is fail-soft. This is the missing surface needed for "anchor descriptors at externally-detected keypoint locations" workflows — the deep localization demo uses it to build map and query descriptors at COLMAP-supplied SIFT keypoint positions.
- `examples/deep_localization_demo.rs` rebuilt against the COLMAP **South Building** public dataset (128 real photos, 61,514 sparse landmarks). The previous synthetic 4×4-bit-pattern synthetic scene was discarded; the new demo loads `sparse/{cameras,images,points3D}.txt`, picks a *map source* photo + a *query* photo by COLMAP image name, anchors each landmark's descriptor at its COLMAP-detected SIFT keypoint location on the map photo (via the new `*::describe_at` accessors), then describes the query image's COLMAP SIFT keypoints with the same extractor and runs `LocalizationPipeline::localize`. Both sides share the COLMAP-detected keypoint locations so the only moving variable in the comparison is the descriptor + matcher pair. Truth pose comes from the COLMAP reconstruction. Sample on `P1180141.JPG → P1180144.JPG` (file gap 3, moderate viewpoint change):

  | Metric | Classical (Corner + BF) | Deep (HogLike + MutualSoftmax) |
  | --- | ---: | ---: |
  | Matches | 257 | **473** (+84 %) |
  | Inliers | 132 | **289** (+119 %) |
  | Inlier ratio | 0.51 | **0.61** |
  | Translation error | 0.0112 m | **0.0106 m** |
  | Rotation error | 0.06° | 0.06° |
  | Mean reprojection | 1.88 px | **1.64 px** |

  Deep produces **2.2× more verified inliers** at the localization layer on a real cross-viewpoint pair while both pipelines recover the COLMAP-truth pose to ~11 mm. The advantage compounds with viewpoint distance: at file gap 4 (`P1180141 → P1180145`) classical retains only 29 inliers vs deep's 81 (**2.8×**); at file gap 5 (`P1180141 → P1180146`) both pipelines fail because COLMAP-supplied SIFT keypoints lie on image structure that neither Corner-patch nor 16×16 HOG can describe at that baseline (the failure boundary is a property of the descriptor type, not of the demo). PnP RANSAC is bumped to `iterations = 65 536, reprojection_threshold = 12.0` for the real-image regime; matcher confidence now flows through `CorrespondenceBuilder` into `Correspondence2D3D.confidence`, and `LocalizationPipeline` routes valid weights into `PnPRansac::estimate_with_weights`. Fetch + run instructions:

  ```bash
  mkdir -p ~/datasets/south-building && cd ~/datasets/south-building && \
      curl -L -o south-building.zip \
          https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip && \
      unzip south-building.zip
  cargo run --release --features image-io --example deep_localization_demo -- \
      --root ~/datasets/south-building/south-building \
      --map-image P1180141.JPG --query-image P1180144.JPG
  ```
- Confidence pipeline now flows end-to-end from matcher → scanner → verifier → RANSAC. New methods plumbed through:
  - `LoopClosureVerifier::verify_with_weights(correspondences, Option<&[f32]>, camera)` (default impl falls back to `verify` so existing implementors are unaffected). `EssentialMatrixLoopClosureVerifier` overrides it and forwards weights.
  - `RelativePoseEstimator::estimate_with_scale_and_weights(correspondences, camera, scale, weights)` and the internal `estimate_with_scale_and_optional_weights` dispatch routing weighted requests to `EssentialRansac::estimate_with_weights` (the PROSAC variant added earlier).
  - `scan_pairwise_loop_closures` now collects `DescriptorMatch.confidence` per match and passes it as `weights` to `verify_with_weights`. When the matcher is `BruteForceMatcher` (no confidence signal), the slice is `None` and behaviour is bit-identical to before; when the matcher is `MutualSoftmaxMatcher`, every match carries a dual-softmax confidence and the scanner's RANSAC samples high-confidence correspondences first.
  - All paths are fail-soft: weight slices that are the wrong length or all-non-finite silently fall back to uniform RANSAC sampling.

  Real-data ablation on the existing 50 + 30 frame KITTI 00 sandwich (`min_keyframe_id_gap = 50, min_matches = 30, --frontend deep`):

  | Metric | Deep (uniform RANSAC) | Deep + confidence-weighted | Δ |
  | --- | ---: | ---: | --- |
  | Cross-segment candidates | 34 | **62** | **+82 %** |
  | Strongest pair inliers | 144 | **152** | +6 % |
  | Strongest pair ratio | 0.731 | **0.749** | +2 % |
  | Total wall time | 31.0 s | 35.4 s | +14 % |

  PROSAC ordering with `MutualSoftmaxMatcher.confidence` lets the scanner find **+82 %** more verified cross-segment loop pairs without lowering verifier thresholds — pairs whose inlier consensus is strong but spread thinly across noisy matches now land their essential matrix early enough in the RANSAC iteration budget to pass verification, instead of timing out at the iteration cap. The wall-time overhead (+14 %) is the cost of sorting indices by weight per `EssentialRansac::estimate_with_weights` call across 62 candidate pairs. Existing classical (`BruteForceMatcher`) callers see no behavioural change because the `confidence: None` slice triggers the uniform-shuffle fallback.
- Generic `StereoVoFrontend<E, M>` over the feature extractor and matcher types so any `FeatureExtractor<Image = GrayscaleImage>` + `Matcher` pair can drive the stereo VO pipeline. The previous `StereoVoFrontend::new` signature is preserved (defaults to `<CornerFeatureExtractor, BruteForceMatcher>`) so existing demos and tests continue to compile unchanged. New `StereoVoFrontend::new_with(camera, baseline, config, extractor, matcher)` constructor accepts arbitrary types — the only constraint is `E::Error: std::error::Error + Send + Sync + 'static` so it round-trips through `StereoVoError::Feature(Box<dyn Error>)` (renamed: the variant now stores a type-erased boxed error rather than a concrete `CornerFeatureError`, keeping the public enum closed but supporting any extractor). The internals (stereo triangulation, Kabsch RANSAC, world-frame composition) are unchanged. `online_slam_stereo_vo_kitti_demo` extends `--frontend classical|deep` (default `classical` keeps the original Corner + BF path; `deep` swaps in `HogLikeFeatureExtractor (max_features=1500, orient=false)` + `MutualSoftmaxMatcher (temperature=25, min_confidence=0.15)`). Real-data ablation on KITTI 00 stride-1 / 50 frames (`gt_length=45.70 m`):

  | Metric | Classical | Deep |
  | --- | ---: | ---: |
  | Kabsch inliers (pair 0→1) | 262 | **454** (+73%) |
  | Kabsch inliers (mean) | 284 | **442** (+56%) |
  | Stereo triangulations (median) | 977 | **1033** |
  | Long tracks (≥3 frames) | 6,848 | **7,233** |
  | Track length (mean / max) | 4.3 / 17 | **4.7 / 22** |
  | BA cost (initial → final) | 1.81 M → 1.11 M (-38%) | 1.38 M → **493 K (-64%)** |
  | VO ATE (mean / max) | **2.55 / 5.46 m** | 2.79 / 5.97 m |
  | BA ATE (mean / max) | 1.59 / 2.89 m | 1.60 / **2.77 m** |

  The deep frontend produces dramatically more Kabsch inliers per pair (+73% on the first pair, +56% on average), longer projection-guided tracks (+29% max length), and BA's residual cost shrinks 64% (vs classical's 38%) — indicating the structure-from-motion problem is materially cleaner. The raw VO mean ATE is slightly worse than classical (2.79 vs 2.55 m) because Kabsch RANSAC's higher inlier consensus on noisier matches can shift slightly off; after BA refinement the two frontends tie on mean ATE while deep wins by 4% on max ATE. Track-length leverage suggests the deep frontend's advantage compounds on longer windows where BA's per-track constraint count grows more than linearly with sequence length. Implementation in `examples/online_slam_stereo_vo_kitti_demo.rs`: extracted a `FrontendState` struct + `run_stereo_vo_frontend(choice, …)` dispatch that constructs the appropriate generic frontend, runs the per-pair loop, and returns the per-frame state the rest of the demo (BA, PGO, ATE evaluation) consumes — so swapping frontends is a single CLI flag without surrounding orchestration changes.
- Match-level confidence plumbing across the deep frontend → RANSAC chain. New `confidence: Option<f32>` field on `DescriptorMatch` (positioned after `ratio`); `MutualSoftmaxMatcher` now populates it with the dual-softmax confidence it had previously been computing internally and discarding, and `BruteForceMatcher` leaves it `None` (no probabilistic interpretation). Two new matcher tests cover (a) self-matches saturate confidence at ~1.0, (b) a distinctive query yields strictly higher confidence than an ambiguous one against the same train set. The downstream consumer is `EssentialRansac::estimate_with_weights(correspondences, camera, weights)` — a PROSAC-flavoured variant that sorts correspondences by descending weight and, for iteration `k`, expands the sampling subset linearly from `sample_size` to `n` over the configured iteration budget. High-confidence matches anchor the early iterations; outliers only get evaluated late. The dispatch is fail-soft: when `weights` is the wrong length, all-zero, or all-non-finite the function silently falls back to the existing uniform shuffle, so external callers cannot accidentally regress through bad data. The previous `EssentialRansac::estimate` is now a thin wrapper that calls the same internal path with no weights, so existing callers see no behavioural change. Two new tests cover the weighted path: (a) `weighted_ransac_recovers_pose_with_correctly_ordered_confidence_weights` injects 12 inliers (weight 0.9) + 18 random outliers (weight 0.05), runs the weighted RANSAC, and asserts that ≥11/12 geometric inliers are recovered with `mean_sampson < 5e-3`; (b) `weighted_ransac_falls_back_to_uniform_when_weights_are_all_zero` verifies bit-identical behaviour to the unweighted path on clean inputs when weights are degenerate. The companion stereo-vo test that previously hand-rolled `DescriptorMatch { ... }` literals now spells out `confidence: None` so the new field is visible at all call sites.
- Optional rotation invariance for `HogLikeFeatureExtractor` (`HogLikeFeatureConfig::orient: bool`). When enabled, a 36-bin gradient orientation histogram is computed in a circular window of radius `patch_radius` around each keypoint, the peak is parabola-interpolated for sub-bin precision, and the HOG descriptor patch is then sampled in that rotated frame using bilinear interpolation — so each cell's gradient histogram is referenced to the keypoint's intrinsic frame rather than the image axis. The output descriptor is therefore invariant to in-plane camera rotation, the SIFT-style behaviour the canonical comparison expects. New helpers `bilinear_sample` (edge-clamped fractional pixel sampler) and `dominant_orientation` (window-radius dependent peak picker) are added inside the module. Two new tests cover (a) `dominant_orientation_aligns_with_strong_gradient` — synthetic vertical-edge image's dominant orientation lands within 0.3 rad of `0` or `π`, the gradient axis; (b) `oriented_descriptors_are_more_rotation_stable_than_axis_aligned` — render the same multi-scale checker + landmark blob texture at `0°` and `30°` rotation, run both `orient: true` and `orient: false` extractions, compare best-cosine-similarity from each kf-A descriptor to its closest kf-B descriptor — oriented wins by a strict margin. The default is `false`: for forward-driving / static-up camera setups (KITTI etc.) the per-keypoint dominant orientation estimate adds variance without providing real invariance and harms matching (KITTI 00 sandwich `--frontend deep` regresses from `34 candidates / strongest 144 inliers` to `1 candidate / 73 inliers` when forced on), so it is opt-in for handheld / rotation-heavy use cases (UAV, cellphone, etc.). The tradeoff is documented on the config field. Companion change: every existing call site (`scanner_loop_closure_demo`, `kitti_revisit_scanner_demo` × 2, `deep_frontend_two_view_demo`, `OnlineSlamPipeline` integration test) now spells out `orient: false` explicitly so the field's tradeoff is visible at the call site rather than hidden behind a default.
- Multi-scale image-pyramid wrapper for the deep frontend: new `build_pyramid(image, levels)` utility (2× downsampling by 2×2 averaging — coarse low-pass without a separable Gaussian dependency) plus `MultiScaleDeepExtractor<E>` that wraps any `DeepFeatureExtractor<Image = GrayscaleImage>` and runs it on each pyramid level, rescaling per-octave keypoint coordinates back to the original image's pixel frame so the merged `DeepFeatureSet` is drop-in compatible with any single-scale consumer (essential RANSAC, scanner, PnP). Per-octave extraction is "fail-soft" — if a coarser octave shrinks below the inner extractor's required margin (e.g. HogLike's 9-pixel patch radius) the wrapper skips that octave instead of aborting, so callers don't have to pre-compute the maximum safe pyramid depth. 4 unit tests cover (a) pyramid shape (each level halves the previous), (b) graceful stop when dimensions reach zero (8×8 → 4×4 → 2×2 → 1×1, 4 levels total), (c) recovery of finer keypoints than a single-octave run, (d) keypoint coordinates remain inside the original image bounds. `kitti_revisit_scanner_demo` extends `--frontend` with a new `deep-ms` (aliases `multiscale` / `deep-multi-scale`) variant that wraps `HogLikeFeatureExtractor` with a 3-octave pyramid (per-octave cap 200 → ~600 features per keyframe, comparable to the 400-feature single-scale baseline). Real-data ablation on the existing 50 + 30 frame KITTI 00 sandwich (`min_keyframe_id_gap = 50, min_matches = 30`): multi-scale deep reports **37 candidates / strongest `(kf49, kf4500)` 216 inliers / ratio 0.742 / 66.2 s wall**, vs single-scale deep's **34 / 144 / 0.731 / 31.0 s**. The strongest-pair inlier count jumps **+50 %** because the pyramid recovers correspondences whose effective image-space size doesn't match the 16×16 patch radius at the original resolution. The runtime roughly doubles because the matcher cost is `O(N²)` and the merged feature count rises from 400 to 600 per keyframe — a fair tradeoff for production-style deep VO where missed correspondences across viewpoint-induced scale changes are the dominant failure mode. README's "Real KITTI Loop Detection" comparison table now lists all three frontends; the multi-scale wrapper composes with any future learned descriptor head (`SuperPoint`, `DISK`) by satisfying the same `DeepFeatureExtractor` trait the wrapper consumes.
- `scanner_loop_closure_demo` now accepts `--frontend classical|deep|both` and runs the full appearance-scanner → SE(3) PGO chain through either frontend (or both side-by-side) on the existing 9-keyframe arc trajectory. `classical` (default) keeps the previous analytical projection path — each visible 3D point becomes a keypoint with descriptor `[idx, 1.0]`, matched by `BruteForceMatcher`. `deep` renders every keyframe to a 320×240 procedurally-textured grayscale image (multi-scale checker plane at world depth `z = 14` plus per-landmark bright blobs), runs `HogLikeFeatureExtractor` on each rendering (`max_features = 256, descriptor_clip = 0.2`), and feeds the resulting `FeatureSet`s to the scanner with `MutualSoftmaxMatcher (temperature 25, min_confidence 0.15)`. `both` runs each back-to-back and prints `candidates / loop pair / pgo max err / drift reduction` side-by-side. Selection logic switched to "longest-baseline gap among accepted candidates, score tie-break" — for the synthetic clean classical path all candidates saturate `mean_sampson ≈ 0` so this reduces to the previous score+gap behaviour, but for the deep frontend it correctly prefers the chain-anchoring loop over short-baseline high-overlap pairs (which the dual-softmax matcher would otherwise rank higher because of better visual co-visibility). With `--out-dir` and `--frontend both` the demo writes per-frontend CSV bundles (`truth.classical.csv`, `pgo.classical.csv`, `loop_edges.classical.csv`, and the `.deep.csv` counterparts). Sample run: classical gets `15 candidates / (kf0, kf8) / 0.402 m drift → 0.016 m PGO max (25.6× reduction)`; deep gets `5 candidates / (kf0, kf8) / 0.402 m → 0.015 m (27.6× reduction)` — i.e., the deep frontend ties the classical analytical-projection path on drift recovery despite running the full HOG → dual-softmax → essential RANSAC chain on rendered images, with no synthetic-correspondence shortcuts. This is the missing end-to-end visible artifact for the deep VO frontend: the matching real-camera deep VO pipeline would behave identically as soon as a learned descriptor head is plugged in for the rendered-pixels stand-in. README's "Synthetic Scanner Loop-Closure Demo" section now embeds the side-by-side comparison table; `pose graph plot via plot_stereo_vo_trajectories.py` works against either frontend's CSV bundle.
- End-to-end deep-frontend `OnlineSlamPipeline` integration test (`online_slam_pipeline_scan_appearance_loops_with_deep_frontend_finds_loop_pair`). Renders three synthetic 320×240 textured grayscale views from independent camera poses (kf10 / kf100 / kf200) — kf10 and kf200 sit at slightly different camera centres + 0.05 rad yaw observing the same 45-landmark cloud over a procedurally textured plane, kf100 is parked 25 m away seeing a fully disjoint cloud + a phase-shifted background. Each rendered view is run through `HogLikeFeatureExtractor` (`max_features = 256, min_corner_score = 0.05, descriptor_clip = 0.2`) and the resulting (keypoints, descriptors) get parked directly on `Keyframe.frame.{keypoints, descriptors}` — the same path real-camera frames would follow. The pipeline is then asked for `scan_appearance_loops(MutualSoftmaxMatcher { temperature 25, min_confidence 0.15 }, EssentialMatrixLoopClosureVerifier { min_inliers 12, min_inlier_ratio 0.4, max_mean_sampson 5e-3 }, AppearanceLoopScannerSettings { min_keyframe_id_gap 50, min_matches 12 })` and we verify that (a) the (10, 200) pair is in the candidate set, (b) the essential-matrix verifier accepts it (`verified == true, inlier_count >= 12`), (c) no candidate involves kf100 — i.e., no false positive from the disjoint scene. This is the missing real proof that the deep frontend (`HogLikeFeatureExtractor` + `MutualSoftmaxMatcher`) and the production pipeline (`OnlineSlamPipeline::scan_appearance_loops`) compose end-to-end with no orchestration changes: the pipeline has always cached `keypoints` / `descriptors` on `Keyframe.frame`, so swapping the upstream extractor is the *only* change needed to get a deep-VO loop closure path. README's "Deep-Style Frontend" section now references the integration test alongside the synthetic two-view + KITTI sandwich ablations.
- `kitti_revisit_scanner_demo` now accepts `--frontend classical|deep|both` and runs the appearance scanner with the chosen frontend(s). `classical` (default) keeps the previous `CornerFeatureExtractor` (`max_features = 400, radius 9`) + `BruteForceMatcher (ratio 0.85)` path. `deep` swaps in `HogLikeFeatureExtractor` (same 400-feature cap, 128-D L2-normalized HOG descriptors) + `MutualSoftmaxMatcher (temperature 25, min_confidence 0.15)` — the same dual pipeline the synthetic two-view demo introduced. `both` runs each frontend back-to-back, prints a side-by-side comparison block (`candidates / best score / best inliers / total time`), and writes both reports into `summary.txt`. Real-data ablation on the fixed 50 + 30 frame `start_50 + revisit_4500` sandwich (`min_keyframe_id_gap = 50, min_matches = 30`): classical reports **25 candidates / strongest `(kf48, kf4500)` 57 inliers / ratio 0.640 / score 24,570 / 76.0 s wall**; deep reports **62 candidates / strongest `(kf49, kf4500)` 152 inliers / ratio 0.749 / score 76,575 / 30.4 s wall**. The deep frontend pulls **2.7× more verified inliers** on the strongest pair, **3.1× higher score**, +37 cross-segment candidates, and runs **~2.5× faster** end-to-end — the L2-normalized descriptors let the matcher cut to inner-product cosine + dual-softmax, skipping the brute-force matcher's per-pair sqrt + ratio bookkeeping. Both frontends pinpoint KITTI 00's actual revisit (seg-A end ↔ frame 4500), and the deep frontend tightens the high-score cluster around it. README's "Deep-Style Frontend" section now embeds the comparison table; companion `summary.txt` keeps the per-frontend report blocks for downstream tooling.
- Deep-style frontend for the vision pipeline: `DeepFeatureExtractor` trait, `HogLikeFeatureExtractor` (SIFT/HOG-flavoured proxy), `MutualSoftmaxMatcher` (LightGlue-style), and `deep_frontend_two_view_demo` example. The new trait at `visloc_vision::features::deep::DeepFeatureExtractor` returns a `DeepFeatureSet { keypoints, scores, descriptors }` so an ONNX/Candle SuperPoint or DISK backend can drop in later by satisfying the same trait — `visloc-core` stays runtime-free. `HogLikeFeatureExtractor` implements both `DeepFeatureExtractor` and the existing `FeatureExtractor` trait: it detects corners with the same response function as `CornerFeatureExtractor`, builds a 4×4 cell × 8 orientation-bin oriented gradient histogram over a 16×16 patch (128-D), and L2-normalizes / Lowe-clips at 0.2 / re-normalizes so the output descriptor is unit-norm and downstream cosine-similarity matchers (`MutualSoftmaxMatcher`) can treat the inner product as the similarity. Per-keypoint score is the corner response normalized into `[0, 1]`, mirroring SuperPoint's per-pixel score map. `MutualSoftmaxMatcher` (in `visloc_vision::matching::mutual_softmax`) takes any two descriptor sets, computes the cosine similarity matrix, applies temperature-scaled softmax across rows *and* columns, and keeps the mutual-NN pair when `confidence = sqrt(row_softmax × col_softmax)` exceeds `min_confidence` (defaults `temperature = 20.0`, `min_confidence = 0.2`). 7 unit tests cover identity matching, ambiguity rejection in one and both directions, mutual-NN enforcement, empty-input handling, dimension mismatch, and zero-temperature uniform fallback. New companion adapter `CornerDeepAdapter` exposes the existing `CornerFeatureExtractor` through the deep interface for A/B testing. `examples/deep_frontend_two_view_demo.rs` renders a 320×240 textured synthetic scene from two camera poses (0.30 m baseline + 0.06 rad yaw, 45 landmark blobs over a multi-scale checker plane) and runs both `Corner + BruteForceMatcher (ratio 0.8)` and `HogLike + MutualSoftmaxMatcher` through the same `RelativePoseEstimator` (essential RANSAC + chirality recovery), so the comparison isolates the frontend itself: classical reports `225 putative / 44 inliers (0.196) / rot_err 3.12° / t_dir_err 73.31°`, deep-style reports `242 putative / 164 inliers (0.678) / rot_err 0.11° / t_dir_err 2.96°` — ~30× rotation/translation-direction accuracy and inlier ratio ~20 % → ~68 %, matching the qualitative gain real SuperPoint+LightGlue offers over classical FAST/ORB pairings on viewpoint-changed inputs. `--out-dir` writes `summary.txt` for downstream comparisons. README's "Deep-Style Frontend (HOG-like + Mutual Softmax)" section walks through the demo end-to-end and refreshes the "Next Technical Targets" list — the Deep VO frontend goal is now realized in skeleton, the next steps shift to swapping in a learned descriptor head behind the trait and rerunning the same ablation on real KITTI data.
- `kitti_revisit_scanner_demo` example — first end-to-end demonstration of `scan_pairwise_loop_closures` on real KITTI imagery. Loads two image directories (e.g., the start subset `frames 0–49` and a revisit subset around `frames 4500–4529`), parses the original KITTI frame indices straight from each filename stem (`004500.png` → 4500) so the combined keyframe stream has globally meaningful ids, extracts `CornerFeatureExtractor` features per frame (FAST corners + intensity-patch descriptors at radius 9, capped at `max_features = 400` for tractable `O(K²)` brute-force matching), and runs the appearance scanner over the combined views with `min_keyframe_id_gap` set to `max(span(A), span(B))` — so intra-segment pairs are filtered before the verifier ever fires and only cross-segment candidates surface. Result on the fixed `kitti_seq00_start_50` (frames 0–49) + `kitti_seq00_revisit_4500` (frames 4500–4529) layout: **25 cross-segment loop candidates, strongest `(kf48, kf4500)` with 57 inliers at ratio 0.640** — i.e., the appearance pipeline correctly detects KITTI 00's major revisit at the start area programmatically, with no GT, no manual seeding, no PnP path. Companion `--out-dir` writes a `summary.txt` listing every accepted pair plus `strongest_from / strongest_to / strongest_score` for downstream consumption. This is the missing real-data validation the README's "Next Technical Targets" called out — alongside the synthetic `scanner_loop_closure_demo` it covers both ends of the appearance-scanner robustness story.
- `PoseGraph::save_text(path)` and `PoseGraph::load_text(path)` for plain-text persistence — no `serde` dependency added. Format is line-oriented and human-readable: `# visloc-rs PoseGraph v1` header, one `P <id> <qw> <qx> <qy> <qz> <tx> <ty> <tz>` per pose, optional `A <id>` for the anchor, and one `E <from> <to> <kind:0|1> <weight> <qw> <qx> <qy> <qz> <tx> <ty> <tz>` per edge (`kind 0 = Sequential`, `1 = LoopClosure`). Numbers are written with `{:.17e}` so the round-trip is bit-identical within `f64` precision (round-trip test asserts `< 1.0e-12` per quaternion / translation component). Companion error type `PoseGraphParseError { Io, Syntax { line, reason } }` reports the offending line number for malformed inputs (verified by negative-path test: parsing `P 0 not-a-number 0 0 0 0 0 0` returns `Syntax { line: 1, ... }`). Unblocks (a) saving a long-session pose graph and resuming offline analysis, (b) bundling a fixture pose graph with a regression test, and (c) bringing maps in/out of demos without standing up a serde feature.
- `OnlineSlamPipeline::scan_appearance_loops(matcher, verifier, camera, settings)` brings the appearance-based pairwise loop scanner into the production pipeline. Iterates the pipeline's stored `map.keyframes` (sorted by frame id for deterministic pair ordering), builds `PairwiseKeyframeView`s straight from each keyframe's `frame.keypoints` / `frame.descriptors` (which are already cached on every keyframe — no FeatureSet sidecar needed), and runs `scan_pairwise_loop_closures`. Returns `LoopClosureCandidate`s with `verification` populated and `geometrically_verified = true`, in the same shape as `process_frame`'s shared-landmark detection — so callers can mix the two streams. Companion `AppearanceLoopScannerSettings { min_keyframe_id_gap, min_matches }` (defaults `30` / `30`) is the pipeline-level wrapper around `PairwiseLoopClosureScannerConfig`. Unlike `process_frame` the new method is `O(K²)` over the keyframe count, intended for periodic or end-of-session use rather than every-frame online detection. The shared-landmark path stays the right call for typical online use; the appearance scanner adds a complementary "find loops the local-mapping window has aged out" path that the previous design couldn't reach. New integration test (`online_slam_pipeline_scan_appearance_loops_finds_revisited_pair`) drops three keyframes (ids 10 / 100 / 200) into a `VisualMap` — only ids 10 and 200 share descriptor space — and verifies the scanner emits exactly the (10, 200) pair with `min_keyframe_id_gap = 50`.
- README "Synthetic Scanner Loop-Closure Demo" subsection between the stereo VO + BA + Loop-closure showcase and "Try It". Documents the no-external-data demo (`scanner_loop_closure_demo`), shows the runnable command, lists the recovered ATE-style numbers (drifted max `0.402 m → PGO 0.016 m`, `25.6×` reduction in 4 LM iter), and embeds `docs/assets/scanner_loop_closure_demo.png` so a reader sees the recovered trajectory at a glance — drifted blue chain → dashed pink scanner-detected loop edge `kf0 ↔ kf8` → purple PGO trace overlapping the green ground-truth circle. Refreshed the stale "Next Technical Targets" section: most of the previously-listed gaps (loop verifier → constraint → SE(3) PGO → bundle adjustment → stereo VO → appearance scanner → loop-edge visualization) are now done, so the targets shift to wiring the appearance scanner into `OnlineSlamPipeline`, demonstrating a real KITTI 00 long-revisit run on the existing pipeline, and the long-standing Deep VO frontend goal.
- `scanner_loop_closure_demo` example — a self-contained synthetic SLAM pipeline that exercises the new `scan_pairwise_loop_closures` end-to-end without needing any external dataset. Builds a 9-keyframe arc trajectory (`R = 3 m`, identity rotation, open arc so kf0 and kf8 keep a meaningful translation baseline — essential-matrix RANSAC degenerates at zero parallax), projects a 30-landmark cloud at depth ~12 m into each keyframe to produce one `FeatureSet` per keyframe, runs the appearance scanner over all pairs `(i, j)` with `min_keyframe_id_gap = 4`, and reports every accepted candidate. The demo's selection picks the highest-`score` candidate with a longest-baseline tie-break (so on synthetic-clean data where multiple pairs saturate the score the loop with the largest frame-id gap wins, since it constrains the largest arc of the drifted chain). The selected pair becomes the loop edge of a `PoseGraph` whose sequential edges are the *drifted* relative poses (per-edge `0.018 rad` yaw + `0.01 m` translation perturbation, ~`1°` per edge), and `optimize_se3_iterative` (Sparse Cholesky + LM + Huber `δ = 0.1`) recovers the trajectory. Sample run: scanner emits 15 candidates and selects `(0, 8)` (gap = 8); drifted max keyframe-center error `0.402 m → PGO 0.016 m` (`25.6×` reduction) in 4 LM iterations (`SE(3) cost 2.18 → 0.003`). With `--out-dir <dir>` the demo writes `truth.csv`, `drifted.csv`, `pgo.csv`, and `loop_edges.csv` so `scripts/plot_stereo_vo_trajectories.py` renders the comparison directly.
- Loop-edge overlay on the stereo VO trajectory plot. The KITTI demo now writes `loop_edges.csv` (`from_id,to_id,source,from_x,from_y,from_z,to_x,to_y,to_z`) for every PGO loop-closure constraint it adds, with `source` distinguishing appearance-detected (`scanner`) from GT-fallback (`synthetic-gt`) edges. `scripts/plot_stereo_vo_trajectories.py` picks the file up automatically (or via `--loop-edges`) and overlays each entry as a dashed line on the top-down `X–Z` plot — pink for `scanner`, olive for `synthetic-gt` — with the connected keyframe ids labelled. On the current 50-frame stride-1 window the scanner finds nothing and the plot shows the single `kf0 ↔ kf49` GT-fallback edge spanning the trajectory's full Z extent, making the geometric pull of the loop constraint visible alongside the VO/BA/PGO chains. Source-color separation makes it easy to scan a multi-loop run (longer KITTI sequence with real revisits) and tell at a glance which edges came from the appearance pipeline vs the reproducibility fallback.
- `online_slam_stereo_vo_kitti_demo` now drives loop *detection* through the new `scan_pairwise_loop_closures` instead of probing only the hard-coded `(0, n-1)` pair. The demo builds `PairwiseKeyframeView`s over every left-camera `FeatureSet`, asks the scanner for accepted pairs (`min_keyframe_id_gap = min(20, n/3)`, `min_matches = 30`), picks the highest-`score` candidate (when any), and re-matches descriptors for that pair so cross-frame stereo Kabsch RANSAC can recover the metric loop edge from stereo correspondences. When no candidate exists the demo falls back to the GT-derived edge as before, additionally running an essential-matrix probe on `(0, n-1)` purely so its `inlier_count` becomes the PGO edge weight (without that the GT-fallback weight collapses to `1.0` and the loop edge stops pulling the chain — observed regression: stride-1/50-frame PGO mean ATE `0.20 m → 0.29 m`). On the current 50-frame stride-1 window the scanner correctly reports `candidates=0` (no physical loop in the data), the `(0, 49)` probe verifies as `false (9/57 inliers, ratio 0.158)`, the GT fallback produces a pose-graph edge with weight 9, and the end-to-end `VO 2.55 m → BA 1.59 m → PGO 0.20 m` ATE numbers are bit-identical to the pre-refactor run. On a longer sequence with a real loop the scanner picks up the visual loop automatically and routes the metric edge through the verifier rather than the GT fallback.
- Pairwise loop-closure *scanner* in `visloc-slam`. New `scan_pairwise_loop_closures(keyframes, matcher, verifier, camera, config) -> Vec<LoopClosureCandidate>` walks every keyframe pair `(i, j)` with `i < j` and a sufficient frame-id gap, brute-force matches descriptors, and runs any `LoopClosureVerifier` (defaults wire `EssentialMatrixLoopClosureVerifier`) on the resulting `TwoViewCorrespondence` set, returning one candidate per accepted pair with the verifier's relative pose populated. Companion types `PairwiseKeyframeView { frame_id, keypoints, descriptors }` (with a `from_features(frame_id, &FeatureSet)` convenience constructor) and `PairwiseLoopClosureScannerConfig { min_keyframe_id_gap, min_matches }` keep the input/threshold ergonomics consistent with the existing `verify_loop_closure_candidates_*` helpers. Two integration tests cover the happy path (loop-pairing geometry on synthetic 12-landmark keyframes at frame ids 10 / 100 / 200 — the disjoint id=100 keyframe is correctly excluded, only `(10, 200)` is emitted) and the gap-rejection path (raising `min_keyframe_id_gap` past the actual gap drops the pair). The result is a real loop *detector* surface (vs only a verifier on caller-supplied candidates), so demos that don't run the shared-landmark candidate pipeline (e.g. `online_slam_stereo_vo_kitti_demo`) can now ask "find any loop in this set of keyframes" with one call.
- README "Metric Stereo VO + BA + Loop-closure Demo" section between the hero asset and "Try It". Documents the pipeline (rectified-stereo triangulation → 3D-3D Kabsch RANSAC → projection-guided multi-frame stereo BA → real `EssentialMatrixLoopClosureVerifier` candidate check → SE(3) PGO with verifier-recovered or GT-derived loop edge), the stride-1 KITTI 00 fetch + demo + plot commands, and a stage-by-stage ATE table on the 50-frame stride-1 window (`VO 2.55 m → BA 1.59 m → PGO 0.20 m mean`). Also extends the "Implemented now" Scope list with four bullets covering `StereoVoFrontend`, `BundleAdjustment` with `BaStereoObservation`, `PoseGraph::optimize_se3_iterative`, and the demo's KITTI-GT ATE evaluation + plotting pipeline so the documentation aligns with the current implementation surface.
- Real `EssentialMatrixLoopClosureVerifier` integration in `online_slam_stereo_vo_kitti_demo`. Before falling back to the GT-derived synthetic edge, the demo now (a) brute-force matches `left_features[0] ↔ left_features[n-1]`, (b) feeds the matches as `TwoViewCorrespondence`s to `EssentialMatrixLoopClosureVerifier::verify`, and (c) prints the verdict (`verified`, inlier count, ratio, mean Sampson). When the verifier accepts the candidate, a cross-frame stereo Kabsch RANSAC over the same descriptor-match inlier subset (filtered to keypoints with stereo at both ends) supplies the metric translation magnitude that essential's pose is missing, and that becomes the loop edge for PGO. On the current 50-frame stride-1 / 30-frame stride-8 windows the verifier correctly rejects the (non-loop) candidate (`9 inliers / 57 matches, ratio 0.158`), and PGO falls back to the GT-derived edge with the loop-edge source label printed alongside it. The end-to-end stride-1 ATE is `VO 2.55 m mean → BA 1.59 m → PGO 0.20 m`. Honest demonstration: integration is in place, the verifier exhibits correct rejection behaviour on data without a physical loop, and a real visual loop would route the metric edge through the verifier rather than the GT fallback.
- Synthetic loop-closure + SE(3) pose-graph optimization wired into `online_slam_stereo_vo_kitti_demo`. New `--synthetic-loop-closure` flag (requires `--gt-poses`) builds a `PoseGraph` over the BA-refined keyframes (or raw VO when BA is disabled), seeds sequential edges from the recovered relative poses, derives a single loop-closure edge between frame 0 and frame n-1 from the GT-subsampled trajectory (so the same data the eval block consumes is also what the loop edge references), then runs `optimize_se3_iterative` (sparse Cholesky + LM + Huber `δ=0.1`). Demonstrates the full SLAM stack (VO → BA → loop closure → PGO) end-to-end without needing a physical visual loop in the data window. Also extends `scripts/plot_stereo_vo_trajectories.py` to overlay `pgo.csv` on the top-down + per-frame-ATE plots. Quantitative impact against KITTI ground truth: stride-1 / 50 frames `VO ATE mean=2.55 m → BA mean=1.59 m → PGO mean=0.29 m (−89 %), max 5.46 m → 0.60 m (−89 %)`; stride-8 / 30 frames `VO mean=33.43 m → BA mean=29.98 m → PGO mean=18.55 m (−45 %), max 61.62 m → 32.54 m (−47 %)`. SE(3) cost collapses from `0.54 → 0.15` (stride-1) and `43.92 → 0.32` (stride-8) in 20 LM iterations.
- Reusable `StereoVoFrontend` struct in `visloc_vision::stereo_vo`. Wraps the per-frame orchestration the KITTI demo previously open-coded: each `process_pair(&mut self, left, right) -> Result<Pose, StereoVoError>` extracts FAST corners + ORB-style patch descriptors, triangulates left/right via the row-restricted matcher, runs the `left_i ↔ left_(i+1)` brute-force matcher and 3D-3D Kabsch RANSAC against the previous frame, and composes the metric relative pose into the running world-frame trajectory. Per-frame state (`poses`, `left_features`, `right_features`, `stereo_per_frame`, `per_pair_translation_m`, `kabsch_inlier_counts`) is exposed as public fields so a downstream BA orchestrator can build multi-frame tracks without re-running the front-end. New `StereoVoFrontendConfig` packs `CornerFeatureConfig` + `StereoFeatureConfig` + `KabschRansacConfig` + temporal Lowe ratio with KITTI-tuned defaults; `frontend.adaptive_track_search_radius_px()` returns the empirical `clamp(12 + 2.0·median_t, 20, 35)` radius used by the projection-guided track extender. Synthetic 3-frame round-trip test verifies metric pose recovery (sub-µm error) and per-call state collections stay in lockstep. The stereo VO demo's per-frame loop now drops from ~120 lines of orchestration to ~25 — the demo itself shrinks accordingly while ATE numbers stay bit-identical to the pre-refactor run (stride-1 BA mean 1.59 m, stride-8 BA mean 29.98 m).
- Adaptive projection-track search radius and trajectory plotting in the stereo VO demo. The KITTI demo's track-extension search radius now scales with the median per-pair Kabsch translation as `radius_px = clamp(12 + 2.0 · t_m, 20, 35)`, so stride-1 (per-pair ~0.81 m → 20 px floor) and stride-8 (~5.47 m → 22.9 px) both stay inside KITTI's discriminative descriptor band — the previously hardcoded 25 px coupling broke under stride change. Sample numbers vs ground truth: stride-1 / 50 frames `BA mean 1.59 m` (was 2.55 m raw VO), stride-8 / 30 frames `BA mean 29.98 m` (was 33.43 m raw VO). Removed the now-dead `pair_inlier_indices` accumulator the projection-guided pass replaced. New `scripts/plot_stereo_vo_trajectories.py` renders a side-by-side top-down (`X–Z`) plot of `vo.csv` / `ba.csv` / `gt.csv` plus a per-frame ATE curve so the demo's CSV outputs become visually checkable; the stride-8 plot makes the KITTI 00 right-turn rotation drift around `Z = 85 m` immediately legible.
- Projection-guided stereo track extension. New `visloc_vision::stereo_vo::extend_stereo_tracks_via_projection(poses, left_features, stereo_per_frame, camera, &TrackExtensionConfig)` projects each frame's stereo-triangulated landmarks forward through subsequent VO poses, searches for a descriptor-matching stereo feature within `search_radius_px` of each predicted pixel (Lowe's ratio-tested against the second-best within-radius candidate), and extends the track until the landmark leaves the FOV / no in-radius match passes the ratio. New types `StereoTrack { landmark_world, observations: Vec<StereoTrackObservation> }` and `TrackExtensionConfig`. The synthetic 4-frame / 5-landmark unit test confirms each landmark produces exactly one length-4 track with deduplication. The KITTI demo replaces its earlier pair-chain track-linking pass with this routine. **Stride-8 KITTI 00 (per-pair motion ~6.9 m, where the brute-force pair-chain peaked at `max_len=4`):** `total tracks 919 → 5299, long(≥3) 68 → 1345 (20× more), max_len 4 → 8`, BA cost `447309 → 263113 px²`, **VO ATE mean 33.43 m → BA mean 30.84 m (−8 %), max 61.62 m → 57.09 m (−7 %) — first time BA reduces ATE on stride-8 data.** Stride-1 KITTI 00 (per-pair motion ~0.86 m): `tracks long 2861 → 6784`, BA still beats VO substantially (BA mean 1.49 m vs VO 2.55 m).
- Multi-frame stereo BA in `online_slam_stereo_vo_kitti_demo`. The demo now chains per-pair Kabsch-RANSAC inlier `(a_kp, b_kp)` entries into multi-frame tracks: if pair `i` reports `(a, b)` and pair `i+1` reports `(b, c)`, `(i, a) → (i+1, b) → (i+2, c)` becomes one BA landmark observed across three keyframes, vs the earlier setup that gave each Kabsch inlier its own 2-frame landmark. Tracks shorter than 3 frames are dropped (no extra constraint over per-pair Kabsch); long tracks feed `BundleAdjustment::optimize` (sparse Cholesky, Huber `δ=4 px`, 15 LM iter) with the chained `BaStereoObservation`s. Track length is dataset-driven: stride-8 KITTI 00 only yields max-length-4 tracks (per-pair motion ~6.9 m so features leave the FOV in ~4 frames), but **stride-1 KITTI 00 (per-pair motion ~0.86 m) yields max-length-22 tracks** and BA actually moves the trajectory toward truth. Quantitative comparison at 50 frames against KITTI ground truth: `VO ATE mean=2.55 m / max=5.46 m → BA ATE mean=1.38 m (−46 %) / max=2.14 m (−61 %)` on stride-1 (`gt_length=45.70 m`); on stride-8 the ATE stays flat because `max_len=4` tracks can only constrain four adjacent pose pairs at a time. The takeaway is structural: the BA infrastructure is correct, the per-pair Kabsch front-end is metric-accurate, and the dominant remaining drift is rotation accumulated across the per-pair window — fixable when track length covers more of the trajectory, requiring either smaller per-pair motion or projection-guided track extension.
- KITTI ground-truth ATE evaluation in `online_slam_stereo_vo_kitti_demo`. New `--gt-poses <path>` and `--gt-original-stride <n>` flags load `dataset/poses/<seq>.txt`, subsample the GT trajectory by `frame_stride · gt_original_stride` so estimated and reference frame indices line up, and report mean / RMSE / max translation error of both the raw stereo VO and the BA-refined trajectory against truth (no Procrustes alignment — both start at identity). Sample run on KITTI 00 stride-8: at 10 frames the stereo VO has `gt_length=67.43 m, vo_length=51.82 m (77 %), ATE mean=10.93 m max=25.08 m`; at 20 frames, `gt=111.42 m, vo=98.98 m (89 %), ATE mean=22.04 m max=41.00 m`; at 50 frames, `gt=286.75 m, vo=271.43 m (95 %), ATE mean=70.99 m max=189.63 m` — length-ratio improves with longer windows but per-frame translation error grows roughly linearly because each pair's small rotation residual accumulates without loop closure or multi-view BA. The 2-frame-only stereo BA in the demo cannot reduce that drift further (each landmark is observed in exactly 2 consecutive frames), confirming that long-track landmarks are needed before BA helps. Demo also writes `gt.csv` alongside `vo.csv` / `ba.csv` for plotting.
- Metric stereo VO frontend on KITTI. New `visloc_vision::stereo_vo` module exposes (a) `triangulate_stereo_features(left, right, camera, baseline, &StereoFeatureConfig)` which row-restricts the L↔R descriptor search to the rectified-stereo epipolar band (`|v_l − v_r| ≤ max_row_residual_px`) with positive disparity and Lowe's ratio filtering, then triangulates each accepted pair into a metric `Point3` in the left-camera frame; (b) `stereo_pair_correspondences(a_features, b_features, temporal_matches)` which joins per-frame triangulations on a temporal descriptor matcher's output; (c) `estimate_relative_pose_kabsch_ransac(correspondences, &KabschRansacConfig)` which runs 3-point Kabsch + RANSAC on 3D-3D correspondences to recover `T_a_to_b` directly in metric units, sidestepping the forward-motion degeneracy of essential-matrix VO. 6 unit tests cover the round-trip recovery, row-band rejection, far-field rejection, the temporal-correspondence join, an end-to-end PnP round-trip, and Kabsch outlier rejection.
- `online_slam_stereo_vo_kitti_demo` example (`image-io` feature gated) loads KITTI rectified stereo (`image_0` + `image_1`) plus `calib.txt`, recovers the baseline `b = −tx_P1 / fx ≈ 0.537 m`, extracts left/right corner features (descriptor radius 9 for L↔R discrimination), triangulates ~1000 stereo features per frame with row-restricted matching + min-depth `3 m` / max-depth `80 m` gating, runs Kabsch + RANSAC on the 3D-3D temporal pairs (4000 iterations, 2 m inlier threshold tuned for KITTI's `Z²·σ_disp/(fx·b)` triangulation noise at 25 m) for each consecutive pair, composes the metric trajectory, and optionally refines the result with a stereo-BA pass over the Kabsch-inlier landmarks (sparse Cholesky on Schur-reduced poses, Huber kernel `δ=4 px`, 5 LM iterations). Writes `vo.csv` and `ba.csv`. Sample run on KITTI 00 stride-8 (50 frames, 49 pairs): VO trajectory `271 m`, BA refines cost `407359 → 268947 px²` to `280 m` against ground-truth `~337 m`. The fetch script `scripts/fetch_kitti_seq00_images.py` now accepts `--cameras image_0,image_1` to pull both grayscale rectified streams in one pass with shared frame indices.
- Stereo observation model for bundle adjustment. `BundleAdjustment` gains a parallel `stereo_observations: Vec<BaStereoObservation>` collection plus `stereo_baseline: Option<f64>`; each `BaStereoObservation` adds the right-eye horizontal coordinate `u_right` to the left pixel and contributes a 3-vector residual `(u_l_pred − u_l_meas, v_l_pred − v_l_meas, u_r_pred − u_r_meas)` with `u_r_pred = u_l_pred − fx·b/Z`. The Schur-complement structure is unchanged: per-observation `J_π_st` (3×3) feeds 3×6 / 3×3 pose / landmark Jacobians whose Hessian / cross-block accumulations have the same 6×6 / 6×3 / 3×3 shapes as the monocular path. Single fixed pose suffices to remove gauge (no scale freedom — the baseline anchors metric scale). New `BaError::MissingStereoBaseline` rejects stereo observations without a positive baseline. 5 stereo BA tests cover residual-zero at truth, missing-baseline rejection, single-anchor metric pose recovery, drifted-landmark recovery, dense=sparse parity, and mixed mono+stereo joint optimization.
- `verify_loop_closure_candidates_{essential,pnp,hybrid}` now share a private `verify_each_candidate(candidates, map, |keyframe| -> Option<LoopClosureVerification>)` helper that owns the candidate iteration, the keyframe lookup against `map`, and the write-back of `verification` / `geometrically_verified`. Each public wrapper supplies a closure that builds correspondences and calls its backend; the iteration shell stops being repeated.
- Rectified-stereo triangulation primitives. New `visloc_vision::stereo::triangulate_stereo_pixel(camera, baseline, left_xy, right_xy, min_disparity_px)` recovers a metric 3D point in the left-camera frame from a rectified stereo pair, returning `None` for non-positive baselines, sub-threshold disparities, or unsupported camera models. New `KittiProjection::stereo_baseline_from(reference)` recovers the baseline magnitude from two KITTI projections (e.g., `P1.stereo_baseline_from(P0)` returns ≈ 0.537 m on KITTI 00). 6 stereo unit tests cover round-trip projection, off-axis points, and the rejection paths; 3 calibration tests cover the baseline helper.
- `online_slam_image_vo_loop_demo` now runs a real-image bundle adjustment pass after PGO. The demo links each consecutive pair's RANSAC-inlier matches into multi-frame feature tracks (≥3 frames per track), triangulates each track with a SVD-based DLT using the post-PGO poses, filters out tracks with negative depth or > 32 px reprojection error on any view, and feeds the remaining landmarks + observations to `BundleAdjustment::optimize` (sparse Cholesky on the Schur-reduced camera system, Huber kernel δ=4 px, 50 LM iterations). When the loop edge has been verified the demo also fixes the last keyframe to anchor both loop endpoints; without a loop edge only the first keyframe is fixed and BA may shift the trajectory endpoint. Sample run on KITTI 00 (60 frames at stride 1, loop closure verified): 11869 raw tracks → 3078 long → 903 accepted landmarks / 2868 observations; BA cost `301143 → 26244 px²`; average per-observation residual ≈ 2.14 px. The refined trajectory is written to `ba.csv` alongside `vo.csv` and `corrected.csv`.
- Robust kernels for bundle adjustment. New `BaConfig::robust_kernel` accepts `RobustKernel::{None, Huber, Cauchy}` and applies the IRLS weight `ρ'(||r||²)` per observation to every contribution into `H_PP`, `H_LL`, `H_PL`, and the gradients. New `BundleAdjustment::robust_cost` reports the matching minimization objective. Tests cover (a) sanity parity that explicit `RobustKernel::None` matches the pre-robust default, (b) cost-clipping semantics: `Huber(δ=10)` and `Cauchy(c=10)` both report strictly smaller cost than the squared cost when an outlier observation is present, and (c) optimization semantics: a `Huber`-driven optimizer reaches a strictly lower Huber-cost than the unweighted optimizer on the same fixture (i.e., the unweighted optimizer is biased by the outlier into a state the Huber objective considers worse).
- `BundleAdjustmentRefiner` implements `LocalRefiner` from `visloc-mapping`, which means `LocalMappingPipeline<K, T, BundleAdjustmentRefiner>` will run windowed BA on every staged map update before it is applied to the visual map. Existing keyframes and landmarks in the local window stay fixed (gauge anchor); newly-staged keyframe poses and landmarks become BA variables, with observations gathered from both the working map and the staged update. Refined values are written back into the staged update so subsequent `apply_to(&mut map)` lands BA-corrected poses / landmarks. Tests cover the drifted-staged-keyframe recovery path (5 cm translation drift → < 1 µm recovery on a 6-landmark fixture) and the empty-staging skip path.
- `online_slam_public_loop_demo` now follows up the SE(3) pose-graph step with a windowed bundle-adjustment pass. After PGO converges, the demo snapshots the COLMAP-loaded landmark positions, drifts every third landmark by ~5 cm, runs `BundleAdjustment::optimize` (sparse Cholesky on the Schur-reduced camera system) with the first two keyframes pinned, and reports per-landmark recovery vs. truth. Sample run on the synthetic 12-keyframe / 60-landmark / 720-observation fixture: cost `27065 → 80 px²` in 12 LM iterations, max landmark error `8.3 mm`, RMS `6.7 mm` — bounded by the residual gauge offset of the fixed second keyframe rather than by BA itself.
- `bundle_adjustment_demo` example: builds a 5-keyframe / 30-landmark / 150-observation synthetic pinhole scene, perturbs the last 3 keyframes (translation + yaw) and a third of the landmarks (3D shifts), runs `BundleAdjustment::optimize` with sparse Cholesky on the Schur-reduced camera system, and prints the iteration trace plus per-keyframe pose error and per-landmark recovery against truth. Sample run: cost `8220 → 1.5e-25` in 4 LM iterations, max pose translation error `2.3e-15 m`, max landmark error `2.9e-15 m` (machine precision recovery).
- Bundle adjustment with Schur-complement landmark elimination in `visloc-slam`. New `BundleAdjustment` data type owns poses, landmarks, observations, a shared pinhole `Camera`, and `fixed_poses` / `fixed_landmarks` gauge-fixing sets; new `BaConfig` exposes Levenberg-Marquardt damping, step / cost tolerances, and a `linear_solver` knob. Each iteration assembles per-observation `2×6` / `2×3` Jacobians (right perturbation `T ← T · Exp(ξ)` with tangent layout `[ρ; ω]`), accumulates `H_PP`, per-landmark `H_LL` `3×3` blocks, and per-(pose,landmark) `H_PL` `6×3` cross blocks, then forms the reduced camera system `S = H_PP - Σ_l H_PL_l · H_LL_l^{-1} · H_PL_l^T` and back-substitutes for landmark deltas. The full `H_PL` matrix is never materialized — Schur reduction is done landmark-by-landmark. Either dense (`Cholesky` / `LU` fallback) or sparse (`CscCholesky`) backs the reduced solve. Tests cover residual-zero at truth, pure pose recovery, pure landmark recovery (all-poses-fixed landmark-only path), joint pose+landmark recovery with yaw drift, and dense=sparse parity.
- Sparse Cholesky pose-graph solver in `visloc-slam`. New `LinearSolver::{Dense, Sparse}` enum, `PoseGraphSe3Config::linear_solver`, and `PoseGraph::optimize_translations_once_with(LinearSolver)` let callers swap the inner linear solve. `Sparse` assembles `(H + λI) δ = -g` (and the translation-only normal equations) as a `nalgebra_sparse::CscMatrix` from per-edge `6×6` / `3×3` block triplets and factors with `CscCholesky`; the dense path is unchanged and remains the default for parity. New parity tests assert the two paths produce numerically identical poses on the existing drifted-loop fixtures (translation-only, SE(3) GN, and SE(3) LM). Measured speedup on a synthetic circular loop with sequential odometry edges plus one loop closure: at 1000 keyframes translation PGO is `2.0 s → 1.3 ms` (~1.5k×) and SE(3) LM is `186.8 s → 55.2 ms` (~3.4k×); at 2000 keyframes translation PGO is `18.6 s → 2.7 ms` (~6.8k×). The real-image VO + loop-closure demo (`online_slam_image_vo_loop_demo`) now uses `LinearSolver::Sparse` so the 1112-keyframe KITTI 00 run scales with non-zero edges instead of cubically with node count.
- README hero asset (`docs/assets/kitti_loop_closure.{png,gif}`) is now generated from the real-image `online_slam_image_vo_loop_demo` pipeline on KITTI 00 (1112 keyframes from `image_0`), not the GT-pose-based fallback. Monocular essential-matrix VO drifts ~548 m by the end of the loop; a single loop closure edge plus translation-only pose-graph followed by SE(3) Gauss-Newton brings the start-anchored endpoint error down to ~5 m. The companion `scripts/build_kitti_loop_asset.py` gains a `--mode real-vo` path that takes `vo.csv` + `corrected.csv` plus a KITTI poses file, runs start-anchored similarity Procrustes alignment of VO and corrected to the GT subsample, and renders the comparison panels and animation.
- `scripts/fetch_kitti_seq00_images.py` streams a stride-subsampled subset of KITTI odometry seq 00 `image_0` directly from the public S3 archive via `remotezip` HTTP byte-ranges, with a parallel-worker fetch loop, so the README asset can be regenerated without downloading the whole 23 GB grayscale archive.
- `online_slam_image_vo_loop_demo` now runs translation-only PGO first (one exact linear solve, scales the chain to satisfy the loop edge) and follows with a small SE(3) refine, giving cleaner convergence on long monocular trajectories than full SE(3) iterative GN alone.
- `online_slam_image_vo_loop_demo` example (`image-io` feature gated) runs end-to-end real-image visual odometry plus loop closure in Rust: it reads a KITTI-format grayscale image sequence + `calib.txt`, extracts `CornerFeatureExtractor` features per frame, matches them with `CrossCheckMatcher<BruteForceMatcher>` between consecutive frames, recovers each pair's relative SE(3) via 8-point essential-matrix RANSAC + cheirality, integrates the trajectory, attempts a single loop closure between the first and last frames through the same pipeline, and finally runs `PoseGraph::optimize_se3_iterative` (LM + Cholesky). Outputs `vo.csv` and `corrected.csv` so the existing asset-generation tooling can render it. No simulated drift, no GT poses; the drifted trajectory is what monocular essential-matrix VO actually produces from the pixel data.
- README KITTI 00 loop-closure asset (`docs/assets/kitti_loop_closure.{png,gif}`) generated from the real KITTI odometry ground-truth trajectory, not synthetic data. The new `online_slam_kitti_loop_demo` Rust example loads `<KITTI>/poses/<seq>.txt`, subsamples to ~150 keyframes, injects a realistic per-edge yaw drift to simulate accumulated odometry error, adds a single truth-relative loop-closure constraint between the first and last keyframes, runs `PoseGraph::optimize_se3_iterative` (Levenberg-Marquardt + Cholesky), and writes truth / drifted / corrected trajectories as CSV. The companion `scripts/build_kitti_loop_asset.py` renders the three-panel comparison PNG and the drifted → corrected morph GIF. Sample run: 152 keyframes, endpoint error collapses from 160.9 m drifted to 0.007 m after 12 LM iterations.
- Hybrid loop-closure verifier `HybridLoopClosureVerifier` (`HybridLoopClosureVerifierConfig` with `max_translation_direction_disagreement_rad` and `max_rotation_disagreement_rad`, `verify_loop_closure_candidates_hybrid` runner) that runs both the essential-matrix and PnP backends on the same candidate, accepts only when both verify AND their recovered relative poses agree to within the configured rotation / translation-direction tolerances, and propagates whichever backend's failure reason is responsible. Adds `LoopClosureVerificationFailureReason::PoseDisagreement` for the consensus-disagreement case. The combined verification reports the PnP relative pose (metric), `min(inlier_count_essential, inlier_count_pnp)` for conservative diagnostics, plus both `mean_sampson_error` and `mean_reprojection_error_px`.
- PnP-based loop-closure verifier `PnPLoopClosureVerifier` (with `PnPLoopClosureVerifierConfig`, `correspondences_2d3d_for_loop_candidate`, and `verify_loop_closure_candidates_pnp` runner) that re-localizes the current frame against the candidate keyframe's landmarks via `PnPRansac`. Returns metric relative poses directly (the keyframe pose carries world scale, no `default_translation_scale` parameter needed) and reports inlier count / inlier ratio / mean reprojection error in pixels. `LoopClosureVerification` now also carries an optional `mean_reprojection_error_px` field so essential-matrix and PnP verifications can be told apart at a glance; the HTML report renders whichever metric is populated.
- `online_slam_pnp_loop_demo` example runs both verifiers side-by-side on the same loop candidate, printing the essential-matrix path (needs an externally supplied translation scale) alongside the PnP path (metric pose recovered directly) and the truth relative SE(3) for comparison.
- Levenberg-Marquardt damping plus Huber / Cauchy robust kernels on top of `PoseGraph::optimize_se3_iterative`. New `RobustKernel` enum (`None` / `Huber { delta }` / `Cauchy { c }`) selects the per-edge IRLS cost; `PoseGraphSe3Config` now exposes `robust_kernel`, `initial_lambda`, `lambda_increase_factor`, `lambda_decrease_factor`, `max_lambda`, and `min_lambda`. With `initial_lambda: None` the solver runs pure Gauss-Newton (every step accepted); with `Some(λ₀)` it runs LM (`(H + λI) δ = -g`, accept on cost decrease, otherwise revert and grow `λ`). `PoseGraphSe3IterationStats` records `lambda` and `step_accepted` per attempt. `PoseGraph::robust_se3_cost` reports the matching minimization objective.
- The pose-graph SE(3) inner solve now prefers Cholesky and falls back to LU on ill-conditioned systems, which is faster for SPD normal equations and survives `H + λI` damping reliably.
- `pose_graph_robust_demo` example runs the same outlier-prone three-keyframe loop through pure Gauss-Newton (KF30 drifts ~0.20 m off truth) and LM + Huber (`delta=0.05`, `λ₀=1e-4`); LM + Huber recovers KF30 to within ~2 mm of truth and prints the full per-iteration `λ` / accept-reject trace.
- `online_slam_public_loop_demo` example ingests a COLMAP-text-format reconstruction from disk and drives the full tracking + verifier + pose-graph SE(3) Gauss-Newton stack on it, exercising the same I/O path that real public-data reconstructions (e.g., COLMAP South Building or KITTI-derived sparse models) would take. Without flags it synthesizes a 12-keyframe / 60-landmark orbit fixture, writes it via `write_colmap_text_model` plus a paired `landmark_descriptors.txt`, reads it back, and reports `se3_cost_before ≈ 8.31 → ≈ 0.0001` (3 iterations) on a combined `[0.05, 0, -0.04]` translation + `0.18 rad` yaw drift injected on the loop-closing keyframe. With `--colmap-path <dir>` it loads a user-supplied sparse reconstruction instead, and `--descriptors-path <file>` lets callers pin landmark descriptors (otherwise synthetic per-landmark descriptors are generated so the demo stays runnable on any registered reconstruction).
- Full SE(3) Gauss-Newton pose-graph optimizer in `visloc-slam`: `PoseGraph::se3_cost`, `PoseGraph::optimize_se3_iterative`, plus `PoseGraphSe3Config` (max iterations, step / cost tolerances), per-iteration `PoseGraphSe3IterationStats`, and a `PoseGraphSe3Result` summary. Uses right-perturbation updates `T_i ← T_i · Exp(δ_i)` with a first-order BCH approximation (`J_r⁻¹ ≈ I`) so each edge contributes `r = log(meas⁻¹ · T_to · T_from⁻¹)` together with `∂r/∂δ_to = Ad(T_from)` and `∂r/∂δ_from = -Ad(T_from)`. Anchors stay fixed; rotations are now corrected alongside translations.
- SE(3) Lie-group helpers in `visloc-core::geometry::se3`: `SE3::log` / `SE3::exp` (right-perturbation `[ρ; ω]` tangent layout), `SE3::adjoint`, plus public `so3_left_jacobian` and `so3_left_jacobian_inverse` (with Taylor fallbacks for small angles). Exercised by `exp ∘ log` round-trip tests and an `Ad(T) · ξ` ↔ conjugation consistency test.
- `online_slam_pose_graph_loop_demo` example now also injects a combined `[0.04, 0, -0.03]` translation drift plus a `0.18 rad` yaw drift on the most recent keyframe and runs `optimize_se3_iterative`, taking `se3_cost_before=0.557` down to `0.000` in 2 iterations and printing per-iteration `cost_before / cost_after / max_step` together with each keyframe's post-optimization translation and rotation error.
- Deep VO / loop-close milestone completion increased to 100% to reflect the full SE(3) Gauss-Newton solver and rotation-aware demo.
- `online_slam_pose_graph_loop_demo` example exercises the full tracking + verifier + pose-graph stack on a six-keyframe synthetic loop: classical localization, verified loop-closure constraint with the matching translation scale, sparse `PoseGraph` with five sequential edges plus the loop edge, a `[0.06, 0.03, -0.05]` injected drift on the last keyframe, and a single translation-only Gauss-Newton step that takes `cost_before=0.105` down to `cost_after=0.000` and reports each keyframe's post-optimization error against the truth path. With `--out-dir` it writes `loop_demo_report.html`.
- Deep VO / loop-close milestone completion increased to 90% to reflect the end-to-end loop demo.
- Sparse `PoseGraph` skeleton with `PoseGraphEdge`, `PoseGraphEdgeKind::{Sequential, LoopClosure}`, builders (`add_pose`, `add_sequential_edge`, `add_loop_closure_constraint`, `anchor`), `translation_cost`, and a single translation-only `optimize_translations_once` Gauss-Newton step that holds rotations fixed and returns `PoseGraphOptimizationStep` diagnostics.
- `relative_world_to_camera` helper turns two `Pose`s into a `previous_to_current` SE3 measurement for `PoseGraphEdge`.
- `online_slam_loop_candidate_with_verifier_dummy` example now also injects a small drift into the most recent keyframe, builds a `PoseGraph`, and runs `optimize_translations_once` so the loop drift correction is visible: cost goes from 0.0585 to 0.0 with mean translation correction ~0.034 m.
- Deep VO / loop-close milestone completion increased to 80% to reflect the pose-graph skeleton and translation-only solver.
- `LoopClosureConstraint` plus `LoopClosureConstraint::from_verified_candidate` and `loop_closure_constraints_from_candidates` lift a verified `LoopClosureCandidate` into a stand-alone constraint (`from_keyframe_id`, `to_keyframe_id`, `relative_pose`, `inlier_count`, `inlier_ratio`, `mean_sampson_error`, `score`) ready for a future pose-graph backend; no solver lives in the crate yet.
- `LoopClosureVerification` now carries the recovered `relative_pose: Option<SE3>` so callers can build constraints (or apply their own scale) without re-running the essential-matrix RANSAC, and `LoopClosureVerifierConfig` adds `default_translation_scale` for caller-controlled translation scale.
- `online_slam_loop_candidate_with_verifier_dummy` example now also builds and prints `LoopClosureConstraint`s; the loop HTML/SVG report renders a separate Loop Closure Constraints table next to the candidate diagnostics.
- Deep VO / loop-close milestone completion increased to 70% to reflect the constraint type and verifier-output enrichment.
- `LoopClosureVerifier` trait, `EssentialMatrixLoopClosureVerifier`, `LoopClosureVerifierConfig`, `LoopClosureVerification`, and `LoopClosureVerificationFailureReason` give loop-closure candidates a classical-geometry verifier built on `visloc-vision::two_view`'s essential-matrix RANSAC, with explicit inlier count, inlier ratio, mean Sampson error, score, and enumerated failure reasons.
- `correspondences_for_loop_candidate` and `verify_loop_closure_candidates` plumb the current frame's tracking inliers and an older keyframe's observations into the verifier without forcing `OnlineSlamPipeline` callers to change their constructors.
- `LoopClosureCandidate.verification` now optionally carries the verifier's output; `geometrically_verified` is updated in place when the verifier rejects a candidate.
- `online_slam_loop_candidate_with_verifier_dummy` example demonstrates the candidate-detection plus geometric-verification path on a 12-landmark synthetic sequence; the loop HTML/SVG report adds verifier inlier counts, inlier ratio, mean Sampson error, score, and failure-reason columns.
- Deep VO / loop-close milestone completion increased to 65% to reflect the loop-closure verifier and verifier-aware demo.
- `visloc-vision::two_view` module with `TwoViewCorrespondence`, a Hartley-normalized 8-point `EightPointEssentialMatrixEstimator`, Sampson-distance-scored `EssentialRansac`, 4-fold `recover_relative_pose` cheirality decomposition, and a composing `RelativePoseEstimator` that applies a caller-supplied translation scale.
- `EssentialMatrixVisualOdometryFrontend` and `EssentialMatrixVisualOdometryConfig` expose the classical-geometry pipeline as a `VisualOdometryFrontend`, returning a full SE3 relative pose plus inlier/Sampson diagnostics and supporting per-pair translation-scale overrides.
- `two_view_vo_compare` example runs the classical essential-matrix frontend alongside the flow-only `TwoViewMatchVisualOdometryFrontend` on the same synthetic three-frame sequence to make the structural difference visible; with `--out-dir` it writes a per-frame text report.
- Deep VO / loop-close milestone completion increased to 60% to reflect the classical two-view geometry pipeline and demo.
- `track_sequence_with_two_view_match_vo_prior` example reads per-pair two-view match text files with `read_two_view_matches_txt`, populates `TwoViewMatchVisualOdometryFrontend`, and feeds the resulting VO priors through `track_frame_with_localization_prior_submap_provider` for a short three-frame sequence; with `--out-dir` it writes the generated input match files plus a per-frame text report.
- `tests/two_view_vo.rs` now covers the file-backed two-view match VO path across consecutive frame pairs to guard the `read_two_view_matches_txt` → `TwoViewMatchVisualOdometryFrontend` → `VisualOdometryPriorProvider` chain.
- Documentation now clarifies that `VisualOdometryEstimate::mean_reprojection_error` stores the mean inlier two-view flow residual in pixels when produced by `TwoViewMatchVisualOdometryFrontend`, and recommends labeling the field as `mean_flow_residual_px` in user-facing logs/reports for that case.
- Deep VO / loop-close milestone completion increased to 55% to reflect the file-backed two-view VO sequence path.
- `PoseTrajectory` and `TrajectorySample` helpers for extracting successful tracking poses, camera centers, path length, mean reprojection error, CSV output, KITTI-style 3x4 pose rows, and TUM-style trajectory rows from sequence-localization results.
- KITTI- and TUM-style trajectory parsers and file readers for reading pose rows back into `PoseTrajectory`.
- `TrajectorySummary` helper and JSON summary export for sequence-localization demos and downstream visualization scripts.
- `TrajectoryErrorSummary` and per-frame translation-error helpers for comparing estimated trajectories against reference poses.
- Optional first-matched-frame translation alignment for trajectory-error reports.
- Self-contained HTML / SVG trajectory-evaluation reports for quick visual inspection.
- Self-contained HTML / SVG single-trajectory reports for sequence-localization demos.
- Self-contained HTML tracking reports for frame-by-frame state, failures, priors, and inlier diagnostics.
- CSV export for frame-by-frame tracking state, localization counts, failures, priors, and map stats.
- `TrackingStats` JSON export for aggregate tracking summaries.
- Tracking diagnostics now distinguish motion pose priors from external localization priors such as GNSS-derived submap narrowing.
- Tracking HTML reports now summarize motion-prior and external-localization-prior usage in the top-level metrics.
- `TrackingStats::from_results` helper for rebuilding sequence diagnostics from stored tracking outputs.
- Trajectory-evaluation example showing frame-id matched translation errors, CSV output, and JSON summary output.
- File-based TUM trajectory evaluation example with optional CSV / JSON / HTML output directory.
- File-based KITTI trajectory evaluation example with optional CSV / JSON / HTML output directory.
- File-based sequence localization example that tracks query feature files and prints or writes CSV / KITTI / TUM trajectory exports plus `summary.json`, `tracking.csv`, `tracking_summary.json`, `trajectory_report.html`, and `tracking_report.html`.
- Tracking sequence example with optional `tracking.csv`, `tracking_summary.json`, `tracking_report.html`, and `trajectory_report.html` output directory.
- Moving-camera GNSS-prior tracking example with optional tracking diagnostics plus `trajectory.csv`, `poses.txt`, `trajectory_tum.txt`, `trajectory_summary.json`, and `trajectory_report.html` output directory.
- GNSS-prior tracking demo output now includes an `index.html` dashboard linking the tracking report, trajectory report, CSVs, KITTI/TUM poses, and JSON summaries.
- GNSS-prior tracking demo now exports a synthetic reference trajectory plus translation-error CSV, JSON summary, and trajectory-comparison HTML report.
- GNSS-prior tracking demo output now includes `manifest.json` with generated file names and top-level tracking / trajectory / error metrics.
- Local quality checks now include a GNSS demo output smoke test for the dashboard, manifest, trajectory exports, and error reports.
- CI now runs the GNSS demo output smoke test in addition to the regular example suite.
- CI now uploads the checked GNSS demo dashboard and export directory as a `gnss-demo-outputs` artifact.
- Documentation now includes a GNSS-prior tracking demo guide with dashboard, report, export, and expected-metric notes.
- GitHub issue templates, a pull request template, contribution guide, and security policy now document the project scope and local quality gate.
- Dependabot now checks Rust crate and GitHub Actions dependencies weekly.
- CI now verifies the declared Rust 1.82 MSRV with `cargo check --workspace --all-targets`.
- Trajectory evaluation now has reusable pass/fail threshold types, evaluator CLI threshold flags, `evaluation_result.json` export, and a local trajectory-evaluation smoke check.
- Tracking statistics now have reusable pass/fail threshold types, and the GNSS-prior demo exports `tracking_evaluation.json` for tracking smoke checks.
- `GrayscaleImage` and `CornerFeatureExtractor` provide a dependency-free image feature extraction smoke path, with a new `localize_with_corner_extractor` example.
- PGM grayscale image IO now supports dependency-free image fixtures and the `localize_from_pgm` example.
- Optional `image-io` feature support for PNG/JPEG grayscale loading and the `localize_from_common_image` example.
- Optional common-image sequence loading and the `track_image_sequence_from_common_images` example.
- Common-image sequence summaries and dimension validation for image-sequence tracking inputs.
- Optional nanosecond timestamps and timestamp validation for common-image sequence inputs.
- Timestamped common-image sequence tracking example with GNSS-derived localization priors.
- Timestamp text parsing for image-sequence datasets with separate image folders and timestamp files.
- GNSS text/CSV parsing for timestamped world-position priors used by sequence localization demos.
- KITTI-style camera calibration parser for turning projection rows such as `P2` into `Camera::pinhole` inputs for automotive sequence demos.
- KITTI-style image sequence loader that combines image frames, optional timestamp files, calibration, and validation summaries.
- KITTI-style image sequence loader example that writes a small automotive-like image folder, timestamps, and calibration before reading them back.
- Local and CI smoke checks now verify KITTI-style image sequence demo outputs and upload them as a CI artifact.
- Documentation now includes a KITTI-style image sequence demo guide covering generated images, timestamps, calibration, output logs, and CI artifacts.
- Local and CI checks now verify local README/docs markdown links and anchors.
- Local and CI MSRV checks now cover all workspace targets and all features through `scripts/check_msrv.sh`.
- docs.rs metadata now builds every publishable crate with all features enabled so optional APIs are included in hosted documentation, and `scripts/package_check.sh` verifies the metadata is present.
- Local and CI checks now verify release metadata consistency across Cargo manifests, docs.rs settings, publish docs, and documented CI demo artifacts.
- README first-view copy and imagery now highlight the real public-data localization demo, robotics use case, current inputs/outputs, working demos, and explicit non-goals for readers evaluating the project quickly.
- README public-data demo assets now include a feature-rich variant with many detected image features and highlighted pose-link overlays for a clearer visual-localization first impression.
- Demo guidance now calls out feature-rich visualization and the future path for learned feature/matcher integrations without implying bundled deep models.
- Roadmap and demo strategy now make deep visual odometry and loop-closure candidate detection explicit next technical targets.
- `visloc-slam` now reports lightweight loop-closure candidates from shared verified landmarks between the current frame and older keyframes.
- `VisualOdometryFrontend`, `VisualOdometryEstimate`, and `NoopVisualOdometryFrontend` now provide the first tracking-level boundary for optional classical or learned two-frame VO integrations.
- `VisualOdometryPriorProvider` and `VisualOdometryPosePrior` convert two-frame VO estimates into current-frame pose priors.
- Two-view match text parsing supports external learned/classical matcher outputs for VO frontend experiments.
- `read_two_view_matches_dummy` demonstrates the external two-view match text bridge.
- Deep VO / loop-close milestone completion is now tracked in docs and surfaced in the README.
- `TwoViewMatchVisualOdometryFrontend` converts external two-view correspondences into a lightweight translation-only VO prior.
- `two_view_match_vo_prior_dummy` demonstrates the first file-backed bridge from external matches to `VisualOdometryPriorProvider`.
- `PLAN.md` now captures a detailed development handoff for the next Deep VO / loop-closure milestones.
- `visual_odometry_prior_dummy` demonstrates the VO-prior adapter path without bundling a model runtime.
- `track_sequence_with_visual_odometry_prior` demonstrates using a VO-derived external prior to narrow localization candidates during tracking.
- `online_slam_loop_candidate_dummy` demonstrates loop-candidate reporting on a tiny synthetic sequence.
- Online SLAM results can now be exported to a self-contained HTML/SVG loop-candidate report.
- Large README animation GIFs are excluded from the root crates.io package while remaining available on GitHub.
- `FramePriorSyncSummary` diagnostics for checking external measurement coverage against frame timestamps.
- `FramePriorSyncEvaluationConfig` and pass/fail sync evaluation for CI-checkable external sensor coverage.
- JSON export for frame-prior sync evaluation results and the timestamped image GNSS-prior demo.
- Local and CI smoke checks now verify timestamped image GNSS-prior demo outputs and sync evaluation JSON.
- CI now uploads timestamped image GNSS-prior demo outputs as a separate artifact.
- Documentation now includes a timestamped image GNSS-prior demo guide covering generated images, timestamp/GNSS files, sync evaluation JSON, and CI artifacts.

## 0.1.0 - 2026-05-07

### Added

- Workspace split into core, vision, IO, localization pipeline, and tracking pipeline crates.
- Core visual localization types: `Frame`, `Keyframe`, `VisualMap`, `Landmark`, `Observation`, `Camera`, `Pose`, and `LocalizationResult`.
- `SO3` / `SE3` pose wrappers and reprojection utilities built on `nalgebra`.
- Brute-force descriptor matching with L2 distance, ratio test, optional cross-checking, and match diagnostics.
- Minimal DLT PnP estimator, PnP RANSAC, pose-estimation diagnostics, and optional Gauss-Newton pose refinement.
- COLMAP text and binary map parsers for `cameras`, `images`, and `points3D`.
- Text parsers for landmark descriptors and query features.
- Localization pipeline over query descriptors and visual-map landmarks.
- Map providers, submap selectors, priors, localization quality gates, and map validation reports.
- Tracking skeleton with motion models, state transitions, and sequence examples.
- Local mapping skeleton with keyframe policy, local map windows, landmark candidates, linear triangulation, staged map updates, and local refinement hooks.
- Online SLAM MVP composition that combines tracking and local mapping without loop closure or global optimization.
- COLMAP text model writer for saving reusable sparse maps.
- Sensor-fusion foundation crate with timestamped frames/poses, GNSS/pose/IMU measurements, covariance types, measurement buffers, frame prior sources, and external localization-prior tracking hooks.
- GNSS-prior tracking example showing radius-submap narrowing before localization.
- COLMAP compatibility notes covering supported sparse model inputs, descriptor handling, writer behavior, and current limitations.
- Root crate prelude and top-level re-exports for common application-facing localization APIs.
- Pre-1.0 to v1.0 migration guide covering recommended imports, localization boundaries, COLMAP descriptor handling, tracking priors, and experimental layers.
- Package metadata and crate-content checks in the local quality gate and CI.
- crates.io package metadata now includes project homepage and repository URLs.
- Workspace member crates now use crate-specific descriptions and docs.rs URLs.
- Publishing guide documenting workspace crate publish order and package-check workflow.
- Examples, integration tests, design docs, local check script, and GitHub Actions CI.

### Not Yet Implemented

- Full Visual SLAM.
- Full SfM.
- Loop closure.
- Dense mapping.
- Full bundle adjustment.
- Full tightly-coupled visual-inertial or GNSS/INS fusion.
