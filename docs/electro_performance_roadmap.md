# Electro performance and scale roadmap

**Status:** execution plan, 2026-08-31  
**Scope:** unordered SfM after the courtyard closure, using ETH3D `electro`
as the first large calibrated control.  
**Primary objective:** become faster than COLMAP end to end on the same frozen
input and candidate policy without trading away registration, camera-centre
accuracy, determinism, or bounded memory.

The operational dataset/sharding protocol remains in
[`large_scale_unordered_sfm_plan.md`](large_scale_unordered_sfm_plan.md). This
document controls implementation order and stop/go decisions. Every
optimization is one-variable, default-safe, and compared against a frozen
artifact. Ground truth is used only by the post-mapping scorer.

## 1. Frozen baseline and final targets

The authoritative first comparison is the 1,200-image, four-camera `electro`
run with the same deterministic 12,000-pair `temporal-pyramid-v1` candidate
manifest.

The merged verified-pair snapshot contains 11,625 verified pairs and
7,475,384 correspondences; its SHA-256 is
`93bf51451866cc066718b827d0ecec8f498ffc23842bd2c1e3902fd7a2697a86`.

| phase / metric | visloc-rs current | COLMAP 3.9.1 CPU | gap |
| --- | ---: | ---: | ---: |
| feature extraction | not yet comparable | 304.12 s | measurement missing |
| exact-pair matching | 2,085.89 s | 471.37 s | visloc 4.43x slower |
| mapper wall | 513.05 s | 4,929.56 s | visloc 9.61x faster |
| mapper peak RSS | 3,977,652 KiB | 1,255,996 KiB | visloc 3.17x higher |
| registered images | 1193 / 1200 | 1200 / 1200 | visloc misses 7 |
| centre RMSE after Sim(3) | 0.1194 m | 0.0468 m | visloc 2.55x worse |
| median / p95 | 0.1099 / 0.1436 m | 0.0316 / 0.0968 m | accuracy gap |

The current visloc model and its pre-memory-optimization control are
byte-identical. The memory work reduced peak RSS from 8,485,216 KiB to
3,977,652 KiB without changing the official score.

The roadmap has three target levels:

| level | registration and quality | mapper | matching | memory |
| --- | --- | --- | --- | --- |
| **Preservation** | `>=1193`, RMSE `<=0.1194 m`; exact hash when behavior is not meant to change | `<=550 s` | no regression above 2,085.89 s | `<=4.0 GiB` |
| **Competitive** | `>=1198`, RMSE `<=0.060 m`, p95 `<=0.110 m` | median `<=300 s` | median `<=600 s` | mapper `<=2.0 GiB` |
| **Parity / release gate** | 1200/1200, RMSE `<=0.051 m` (within 10% of control), p95 `<=0.106 m` | median `< COLMAP` | median `<=471.37 s` on the same pairs | `<=1.5x` COLMAP mapper RSS |

End-to-end victory may be claimed only after both implementations include
feature extraction, candidate generation, exact-pair matching/verification,
mapping, and model writing. Cached-feature and mapper-only results remain
separate claims.

## 2. Experiment contract

Every experiment must declare before it runs:

1. one hypothesis and one intended behavioral variable;
2. the source revision, binary hash, input/candidate/snapshot hashes, command,
   thread count, and cold/warm cache classification;
3. expected invariant model hashes, or the predeclared numeric quality gate
   when a behavioral change is intentional;
4. phase wall/CPU time, peak RSS and memory checkpoints, registration/model
   counters, official post-map score, and durable output hashes;
5. a pass, fail, or inconclusive decision and the next action.

Performance decisions use the median of three warm repetitions after one
untimed warm-up. A single long run may reject a clear regression, but it may
not establish a small speedup. Quality-changing runs are scored once only
after the configuration grid is frozen; failed variants are retained as
negative evidence. No threshold may be selected from the `electro` GT score
alone: the chosen variant must also pass the courtyard control and the
applicable South/terrace/office/EuRoC non-regression suite.

Immediate stop conditions are an unplanned model-hash change, GT entering
candidate/matching/mapping decisions, an `N^2` allocation, RSS above the
declared tier ceiling, non-resumable partial artifacts, or a quality loss
outside the current level's gate.

## 3. Milestone 0 — stabilize the current branch

**Purpose:** turn the present large uncommitted result into a reviewable,
reproducible baseline before adding another optimization.

### Work

- Inventory every tracked/untracked file and classify it as artifact protocol,
  candidate/matching control, mapper/timing, memory ownership, or evidence.
- Record the complete current diff and create a recoverable local integration
  safety point before splitting it. Do not submit that compound safety commit.
- Rebuild and merge six dependent, individually gated PRs in this order:
  1. verified-pair snapshot format plus merge helper;
  2. electro runner, official COLMAP control, scorer, and Python tests;
  3. mapper timing, targeted growth, and no-final-BA control;
  4. BA memory instrumentation and ownership reductions;
  5. the single-file demo integration for rig-aware candidates, resumable
     feature export, snapshot workers, and mapper/resource controls;
  6. benchmark manifests, README visuals, and evidence documentation.
  The demo integration remains one reviewable CLI boundary because its parser,
  validation, dispatch, and tests live in one example file; mapper and solver
  behavior remain isolated in PRs 3 and 4.
- Add a repository-sized visloc result manifest containing the exact external
  artifact paths only as descriptive provenance, never dataset contents. It
  records phase timings, RSS, model hashes, score, candidate/snapshot hashes,
  binary hash, and commands for the frozen run.
- Re-run formatting, Python runner tests, example CLI tests, the complete
  `visloc-slam` library suite, and the frozen courtyard verification gate.
- After merge, delete only branches proven merged; retain external benchmark
  artifacts and immutable model controls.

### Exit gate and deliverables

- Six green dependent PRs with no unrelated worktree changes.
- Exact electro model hashes and score reproduced from the recorded command.
- Courtyard remains 38/38 and at or below its 0.01 m gate.
- The next experiment starts from clean `main`, not from the current compound
  diff.

Failure to reproduce any hash stops all performance work and reopens the
baseline commit that introduced the discrepancy.

## 4. Milestone 1 — calibrate measurement and restart behavior

**Purpose:** prove the runner and resource accounting on the 300-image probe
before expensive quality or performance searches on 1,200 images.

### Work

- Measure feature extraction, candidate generation, each match shard, merge,
  mapping, scoring, disk high-water, and process/stage RSS separately.
- Run the complete probe twice and compare candidate, feature, snapshot, and
  model hashes.
- Inject a kill during feature extraction and matching, resume, and verify that
  only complete hash-valid shards are reused.
- Replace one artifact with same-size corrupt content and verify fail-closed
  behavior.
- Run an exhaustive or highest-feasible diagnostic control to measure bounded
  candidate verified-pair recall; this control remains score-only evidence and
  may not influence pair selection.

### Exit gate and deliverables

- Two identical completed runs, verified-pair recall `>=99%`, registration
  loss `<=1` percentage point, RAM `<=16 GiB`, and disk `<=50 GiB`.
- Corrupt or partial artifacts are rejected and the resumed final hashes match
  an uninterrupted run.
- Deliver `electro-300-phase-ledger.json`, failure-injection logs, and runner
  regression tests. Any silent reuse or hash mismatch blocks 1,200-image work.

## 5. Milestone 2 — attribute the quality gap

**Purpose:** determine why the same 12,000 candidate pairs yield seven fewer
cameras and 2.55x higher centre RMSE. Faster BA is not useful if it freezes the
wrong model.

### 2A. Input and graph ledger

- Compare per-image feature rows, verified degree, correspondence count,
  connected components, track-length distribution, conflicts, triangulation
  angle, and registration order between visloc and COLMAP.
- Produce a seven-image missing-registration ledger. For each image classify
  the first divergence as candidate absent, raw match absent, verification
  rejected, track unavailable/conflicted, PnP rejected, filtered after BA, or
  never selected.
- Compare cap32/64/96/128 and uncapped mapping using the same complete
  snapshot. The cap is an explicit mapper resource variable; matching and
  verification remain unchanged. Compare deterministic prefix, descriptor
  distance, spatial-bin, and camera/time-diverse sampling only after the cap
  ceiling is understood.

**Gate:** first reach the diagnostic target `<=0.0702 m` (1.5x COLMAP), then
select the smallest safe cap that reaches at least 1198 images and `<=0.060 m`,
or document that the cap is not the limiting cause. Stop a cap increase if RSS
exceeds 4 GiB or quality worsens in two consecutive levels.

### 2B. Mapper-decision replay

- Add a compact deterministic decision trace for seed ranking, next-image
  support, PnP inliers/residuals, triangulation additions, image/observation
  filtering, and each refinement round.
- Normalize equivalent COLMAP log/database counters into the same comparison
  report. Do not copy COLMAP poses or point identities into visloc decisions.
- Isolate track construction, registration order, triangulation/completion,
  filtering, and BA schedule one at a time.

**Gate:** every proposed behavior change must explain a named divergence,
register no fewer images, and improve either RMSE by at least 10% or the missing
image count. Reprojection-only improvements do not pass.

### 2C. Quality closure

- Predeclare a small configuration matrix from the attribution report rather
  than broad parameter search.
- Re-run the winner on courtyard and at least one held-out unordered suite.
- Freeze a new quality champion only if it reaches the Competitive level on
  electro and passes every preservation control.

**Deliverable:** `quality-attribution.json` plus a short report identifying the
dominant registration and RMSE causes. If no safe variant reaches Competitive,
the report becomes the input to a scoped algorithm milestone; performance
tuning continues only on exact-behavior paths.

## 6. Milestone 3 — mapper speed and BA memory

**Purpose:** reduce the 452.90 s final-refinement cost while retaining the
Milestone-2 quality champion.

Work proceeds in this order, one PR and A/B per item:

1. Assemble the pure-visual Schur pose system directly as deterministic 6x6
   blocks; remove the initial dense matrix and scalar dense scan.
2. Reuse symbolic sparsity/factor analysis across LM iterations when the
   observation graph is unchanged.
3. Reuse normal equations for rejected LM damping retries where the state did
   not change.
4. Replace whole-model rollback clones with a bounded update journal.
5. Audit why three global BAs are required. Remove or localize a refinement
   round only when the model hash is unchanged, or when the quality champion's
   numeric gate passes.
6. Consider parallel block assembly/factorization only after deterministic
   serial block assembly is proven.

### Gates

- Exact-behavior changes require byte-identical electro and courtyard models.
- Each accepted PR must reduce median BA time or peak RSS by at least 10%; two
  consecutive sub-5% results stop that line of work.
- Intermediate target: final refinement `<=250 s`, mapper RSS `<=2.0 GiB`.
- Release target: mapper median `<=300 s` at Competitive quality. The existing
  `<COLMAP` mapper result is retained as a hard non-regression ceiling.

## 7. Milestone 4 — matching and end-to-end speed

**Purpose:** remove the current 24-fold feature-bank reload pattern and beat
COLMAP on the exact same pair workload.

### Work

1. Add a persistent match worker that loads/validates the feature bank once and
   processes multiple candidate shards in stable order.
2. Bound worker queues and shard result buffers; preserve atomic snapshot
   publication and exact ordered/unordered edge hashes.
3. Introduce a versioned memory-mapped feature store only after the persistent
   worker A/B. Keep the text feature reader as the compatibility oracle.
4. Avoid retaining descriptors after matching when a mapper consumes a
   validated snapshot; retain only keypoint coordinates and row identity.
5. Profile matcher kernels before changing descriptor representation or SIMD.
   A u8/quantized/SIMD path is a behavioral change unless it reproduces the
   exact match snapshot.
6. Measure visloc SIFT extraction under the same 2048-feature, CPU8, calibration
   and image-size contract as COLMAP. Only then publish an end-to-end total.

### Gates

- First gate: identical merged snapshot hash with matching sum `<=600 s` and
  peak RSS `<=2.0 GiB`.
- Parity gate: median matching `<=471.37 s` on the same 12,000 pairs.
- End-to-end gate: comparable extraction + candidate + matching + mapper wall
  is lower than COLMAP while Competitive quality and reproducibility pass.
- Any shard-order or snapshot-hash change is a failure until explained and
  independently quality-gated.

## 8. Milestone 5 — larger environments

Scale validation begins only after Milestones 0–4 satisfy their preservation
gates. Image count and physical scene scale are tested separately.

1. Re-run the 300-image electro probe with injected interruption/restart and
   confirm identical artifact hashes.
2. Run every ETH3D low-res many-view scene independently and aggregate the
   approximately 10,008 images as a batch report. Never connect unrelated
   scenes into one graph.
3. Select one genuinely larger connected environment (for example a frozen
   Tanks and Temples video-derived scene) at 1k, 2.5k, 5k, then 10k frames.
   Freeze license, sampling, calibration assumptions, and manifests before the
   first run.
4. Run an I/O-only synthetic manifest stress at 10k/100k to validate sharding,
   hash checking, failure recovery, and absence of `N^2` state without making
   a geometry-quality claim.

### Scale gates

- Candidate count remains `O(NK)` and is bounded by the manifest.
- No single worker exceeds 4 GiB at 10k; exceeding it stops the tier and
  triggers smaller shards or streaming, not silent pair/image removal.
- A killed worker resumes without recomputing valid shards and produces the
  same final hashes as an uninterrupted run.
- Registered fraction loses at most one percentage point against that scene's
  smaller/exhaustive control, and no new unexplained component appears.
- Report per-scene quality and resource distributions; an aggregate average
  may not hide a failed scene.

## 9. Dependencies and risk register

The default dependency chain is `M0 -> M1 -> M2 -> (M3, M4) -> M5`.
Exact-output M3/M4 prototypes may start after M1, but no behavioral optimizer
may be promoted until M2 freezes its quality champion. M5 requires both mapper
and matching preservation gates, not necessarily their final stretch targets.

| risk | detection | mitigation / rollback |
| --- | --- | --- |
| A change is lost or misassigned while splitting the dirty branch | path/hunk inventory differs from the safety diff | retain the local safety point until all six PR trees reproduce the full intended diff |
| Cache or thermal state creates a false speedup | cold/warm labels or three-run spread disagree | report all samples, decide on warm median, classify small differences as inconclusive |
| Electro GT is overfit | a threshold was chosen after viewing score, or held-out suite regresses | predeclare the grid, use GT post-map only, require courtyard plus one held-out unordered suite |
| Sparse/parallel arithmetic changes the solution | dense-oracle test or model hash changes | keep dense backend, revert the PR, or treat it as a quality-changing experiment with full gates |
| An allocation is removed but the peak moves elsewhere | `/usr/bin/time` and stage HWM disagree | retain both measurements and optimize the true global peak, not one checkpoint |
| Persistent workers leak state across shards | fresh-process and persistent snapshot hashes differ | reset per-shard state, fail closed, retain the one-shard process compatibility path |
| Large runs exceed storage, RAM, or dataset terms | preflight manifest/quota/license check fails | stop before download/run; reduce tier or shard size without dropping data silently |

## 10. Immediate execution queue

The next four work packages are deliberately narrow:

1. **Baseline closure:** create the result manifest, split the current diff into
   six dependent PRs, run the full gates, and merge them in order.
2. **300 probe closure:** freeze the phase ledger and prove restart/corruption
   behavior twice before another 1,200-image experiment.
3. **Quality attribution:** build the seven-image ledger and cap32/64/96/128/
   uncapped controls; choose no mapper change until the first divergence is
   named.
4. **First performance pair:** direct block-sparse BA assembly and persistent
   matching worker as separate PRs, both exact-output A/Bs.

After each package, update this roadmap's baseline table and decision log.
Do not start the 10k geometry run merely because the 1,200-image mapper is
fast; quality, end-to-end matching, memory, and restart gates must all be green.
