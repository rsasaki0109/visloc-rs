# Large-scale unordered-SfM validation plan

**Milestone 4 protocol; Milestone 5 measured update (2026-09-02).** This
document defines the validation stages and restartable runner now exercised
through independent ETH3D 10,008-image aggregation, synthetic 100k I/O, and a
connected OpenLORIS 10k tier. It does not change the production default. The
38-image courtyard result and M3 reduced-candidate result remain acceptance
controls.

Implementation order, performance/quality targets, stop conditions, and PR
boundaries are maintained separately in
[`electro_performance_roadmap.md`](electro_performance_roadmap.md). This
document remains the dataset, artifact, and execution protocol.

## 1. Objective and rules

The objective is to establish whether unordered SfM remains reproducible as
the image count grows from hundreds to 10k+ images, while keeping candidate
generation, matching, and mapping bounded. Every run must be independently
restartable and must record the exact input/config/tool hashes.

The following rules are non-negotiable:

- Candidate selection before local matching may use only image metadata/order,
  cheap global retrieval descriptors, and calibration metadata. It must not use
  raw matches, verified inliers, ground truth, or supplied extrinsics.
- Reference poses/geometry are used only after mapping for scoring. A COLMAP
  run is a control, not an input to visloc.
- A scene/sequence is a unit of reconstruction. Do not connect unrelated
  scenes merely to increase the image count.
- No `N × N` pair list, descriptor matrix, or all-image feature resident set is
  allowed beyond the small courtyard control. Pair generation is `O(NK)` and
  pair/feature work is streamed or sharded.
- An interrupted run may resume only from hash-validated complete artifacts;
  file size alone is never sufficient. Temporary outputs are atomically
  renamed into the manifest.

## 2. Dataset selection and authoritative candidates

Selection criteria are: an official download/terms page, a stable scene or
sequence identity, calibration and a reference trajectory/structure when
available, a declared license or access agreement, and a manifest from which
image counts and checksums can be frozen. Current availability and terms must
be rechecked immediately before any download.

| Scale | Official candidate and source | What is known before download | Use and caveat |
| --- | --- | --- | --- |
| Control / hundreds | [ETH3D high-res multi-view datasets](https://www.eth3d.net/datasets) | The official table lists 13 training scenes totaling 454 DSLR images; all undistorted training images are listed as 5.5 GB and each scene has calibration/scan evaluation data. | First cross-scene-capable validation after the 38-image control. Run one scene at a time; never treat the aggregate as one connected map. |
| Recommended first large target | [ETH3D low-res many-view, official table](https://www.eth3d.net/datasets) | `electro` is listed as **4×300 images** and its undistorted image archive as about **0.3 GB**, with calibration and scan evaluation data. | Start with a deterministic 300-image shard for resource calibration, then the complete 1,200-image `electro` scene. It is the closest scalable control with explicit calibration/GT and manageable storage. |
| Thousands | [EuRoC MAV](https://projects.asl.ethz.ch/datasets/euroc-mav/) and [KITTI raw data](https://www.cvlibs.net/datasets/kitti/raw_data.php) | EuRoC provides stereo, synchronized IMU, calibration, and motion/structure ground truth. KITTI provides rectified/ unrectified stereo, timestamps, calibration, and GPS/IMU; raw downloads require account access. | Use EuRoC for a known-trajectory sequential sanity check, not as the sole unordered benchmark. Use one KITTI sequence at a time and derive counts from the official files rather than assuming a frame count. KITTI data is CC BY-NC-SA 3.0 per its [official license notice](https://www.cvlibs.net/datasets/kitti/). |
| 10k+ | ETH3D low-res many-view aggregate, still sourced from the [official dataset table](https://www.eth3d.net/datasets) | The listed training scenes sum to 4,796 images (`4×237 + 4×300 + 4×257 + 4×240 + 4×165`); the listed test scenes sum to 5,212 (`4×266 + 4×278 + 4×199 + 4×208 + 4×352`), or 10,008 across separate scenes. | This is a batch of independently scored scenes, not a single 10,008-image graph. It is the preferred 10k validation after `electro`; process one scene/rig shard at a time and aggregate resource/quality statistics. |
| 10k–100k | [Tanks and Temples download page](https://tanksandtemples.org/download/) | The official site supplies per-scene high-resolution videos and sampled image sets, recommends its MD5-checking downloader, and provides training geometry for selected scenes. Intrinsics are not explicitly supplied; the site gives an initialization heuristic. | Useful for real large-scene stress tests after ETH3D. Freeze the selected scene/image manifest and accept the [non-commercial terms](https://tanksandtemples.org/license/). Do not infer a total image count from the site; record the downloaded manifest. |
| 100k+ (static, very large) | [DTU Point Feature Data Set](https://roboimagedata.compute.dtu.dk/?page_id=24) | The official page states 60 scenes × 119 positions × 19 illuminations = **135,660 images**, 1200×1600, with calibration and structured-light surfaces; full-resolution data is listed as about **730 GB**. | Strong controlled feature/repeatability stress test, but illumination repeats and fixed robot views are not representative of a driving sequence. Start with one 6-scene archive; do not reserve 730 GB until the shard protocol is proven. |
| Optional 100k+ (long-term/dynamic) | [Oxford RobotCar datasets](https://robotcar-dataset.robots.ox.ac.uk/datasets/) and [downloads/terms](https://robotcar-dataset.robots.ox.ac.uk/downloads/) | The official site describes more than 100 repetitions of an Oxford route, supplies calibration/ground truth tooling, and lists individual downloads from roughly 10 GB to 244 GB. Download registration is required and the license is CC BY-NC-SA 4.0. | A realistic long-term retrieval/mapping stress test after the controlled tiers. Weather, traffic, construction, and very large I/O make it unsuitable as the first large run. |
| Optional 1M+ (RGB-D/synthetic alternatives) | [ScanNet official repository](https://github.com/ScanNet/ScanNet), [TartanAir documentation](https://tartanair.org/), and [CO3Dv2 official tools](https://github.com/facebookresearch/co3d) | ScanNet reports 2.5M views in more than 1,500 scans but requires an institutional Terms of Use agreement. TartanAir V2 documents 65 environments and CC BY 4.0; its V1 paper reports over 1M simulated frames, which must not be conflated with V2 counts. CO3Dv2 lists 5.5 TB for all archives and 8.9 GB for its single-sequence subset, under CC BY-NC 4.0. | Use only when the required terms and modality conversion are explicitly frozen. These are supplementary scale tests, not substitutes for calibrated real-photo SfM. |

The recommended order is therefore: current courtyard → one ETH3D
high-res scene → ETH3D `electro` (300 probe, then 1,200 full) → all ETH3D
low-res many-view scenes (10,008 images in separate maps) → selected DTU or
Oxford shards. No data is downloaded as part of this milestone.

## 3. Staged runbook and resource budgets

Budgets below are safety ceilings for planning, not measured performance
claims. Tier A must calibrate the ceilings before Tier B is started.

| Stage | Images per mapping job | Candidate ceiling | Working RAM ceiling | Working disk ceiling | Go condition |
| --- | ---: | ---: | ---: | ---: | --- |
| Courtyard control | 38 | 703 exhaustive or 200 reduced | existing control | existing external artifact | Exact durable control remains valid. |
| Tier A | 100–454 | `min(N(N-1)/2, 32N)` | 8 GB | 20 GB | Two identical runs, complete manifests, and no loss against an exhaustive scene control. |
| Tier B probe | 300 | `min(N(N-1)/2, 32N)` | 16 GB | 50 GB | One-at-a-time extraction and bounded matching stay within ceilings; all files validate after restart. |
| Tier B full | 1,200 | `min(N(N-1)/2, 32N)` | 32 GB | 150 GB | Full registration or a documented observability ceiling, with candidate recall and COLMAP comparison recorded. |
| Tier C | 10k aggregate, scene-sharded | `≤32N` per scene/job | 32 GB per worker | 250 GB per worker | No process materializes all pairs; shard merge is hash-validated and component results are reproducible. |
| Tier D | 100k+ shards | `≤64N` per shard, unless a measured recall audit justifies another bound | 64 GB per worker | 2 TB per shard host | Only after Tier C passes; network/disk high-water and retry behavior are recorded. |

The initial retrieval profile is a fixed, documented top-`K` policy (start with
`K=32` for new tiers) plus a small local/sequence reserve where timestamps or
official frame order is part of the input. The effective budget is a manifest
field, not an implicit loop limit. `K` and any reserve may be changed only in
a new A/B manifest; they are not selected from verified-match counts.

### Per-run procedure

1. Freeze the source revision, tool/container versions, dataset terms, image
   manifest, calibration manifest, and output root outside the repository.
2. Decode a small image sample and run a feature smoke test. Record dimensions,
   feature rows, descriptor schema, peak RSS, and elapsed time.
3. Stream feature extraction one image at a time. Write per-image feature and
   locus shards to temporary files, validate row counts/hashes, then rename
   atomically. Pass `--sift-stream-resume` to make the Rust SIFT exporter
   publish a per-image completion sidecar only after both files are complete;
   a later invocation skips an image only after its extractor/configuration,
   source-image, row-count, and output hashes validate. A manifest records
   config hash, source-image hash, row count, and schema version.
4. Build cheap global descriptors/retrieval index in `O(NK)` query work. Emit
   an image-name-bound candidate manifest before local matching. Validate pair
   bounds, duplicates, image order, budget, config hash, and checksum.
5. Match and verify candidate shards with bounded workers. Store exact pair
   order, raw/accepted counts, verifier configuration/outcome, and checksum.
   A retry must replace only an invalid temporary shard.
6. Map from streamed/sharded correspondences. Keep connected-component state
   bounded; run adaptive expansion only for an unregistered/low-degree image or
   component after the initial schedule, with its additions recorded as a new
   manifest revision. Never use GT to decide an expansion.
7. Run the same frozen visloc and official COLMAP controls where feasible. Do
   not compare a reduced visloc graph to an exhaustive COLMAP graph without
   labelling the difference.
8. Score after mapping, archive logs/JSON/hashes, and repeat the complete run
   or a deterministic shard subset. A restart must reuse only complete,
   hash-valid artifacts.

## 4. Required measurements and acceptance rules

Every result JSON must include:

- image count, registered count/fraction, connected-component count and sizes;
- proposed, matched, verified, and accepted-inlier pair counts; per-image
  degree/support; candidate recall against an exhaustive reference when that
  reference is available (diagnostic only, never a selection input);
- track count, observation count, track-length histogram, same-image conflicts,
  triangulation/parallax summaries, mean/median/p95 reprojection, and rejected
  observations;
- extraction, retrieval, matching, verification, mapping, BA, and total wall
  time; peak RSS; feature/match/model bytes; disk high-water; network bytes and
  retry/cache-hit counts when applicable;
- source/config/tool/container hashes, ordered and unordered manifest hashes,
  restart outcome, and byte hashes for each durable model artifact;
- centre/pose accuracy only after alignment when a reference exists (RMSE,
  median, p95, max, scale, and per-image outliers), plus ATE/RPE for trajectory
  datasets. Reprojection settings and “tracks” versus “points” terminology
  must remain explicit.

Use the following predeclared decisions:

- **Courtyard safety gate:** before and after any scheduler change,
  `scripts/benchmark_courtyard.sh --verify-only` must pass; the exhaustive
  model must remain 38/38 and ≤0.01 m centre RMSE. The current 703-pair
  champion is the accuracy control; the 200-pair union is a successful bounded
  A/B, not a replacement or a new champion.
- **Structural gate:** on a scene with a complete reference/control, a reduced
  candidate run must not lose more than one percentage point of registered
  fraction or introduce an extra disconnected component. A failure stops the
  reduction and triggers candidate-recall/component diagnostics.
- **Accuracy gate:** when a same-input COLMAP/control score exists, require
  centre/pose RMSE no worse than the frozen control tolerance declared in that
  dataset manifest, and reprojection no worse than 1.10× the control unless
  the metric definition differs. If no reference exists, report structure and
  reprojection but classify accuracy as inconclusive.
- **Candidate gate:** Tier A/B must retain at least 99% of the exhaustive
  verified-pair recall on the bounded calibration scene and preserve every
  component bridge needed by the control. For larger tiers, report recall per
  shard/component; a low global average cannot hide an isolated component.
- **Reproducibility gate:** a repeated run must have identical input,
  candidate, feature/match-shard, and model hashes, or a documented
  nondeterministic dependency with a bounded numeric tolerance. Partial,
  corrupt, or silently skipped artifacts are a stop condition.
- **Resource gate:** exceed any RAM/disk/network ceiling, or materialize an
  `N²` object, and stop before increasing the tier. First reduce worker count or
  shard size; do not silently drop images/pairs.

## 5. Scale requirements for the implementation

The current M3 flat manifest is sufficient for the courtyard A/B but is not a
claim that the mapper is already 100k-image ready. The next implementation
work, only after a Tier A/B resource probe, should preserve these contracts:

### Retrieval and candidate manifests

- Keep global-descriptor retrieval at `O(NK)` with deterministic total ordering,
  stable image IDs, and a bounded local/bridge reserve.
- Keep `visloc_candidate_manifest_v1` image-name-bound and replayable. Generated
  manifests may carry a canonical `metadata KEY VALUE` block recording the
  pair source and local grouping policy; the runner preserves and validates it
  across every shard. A future `v2` should add dataset/config hash,
  retrieval-model hash, shard ID, pair range, source/score metadata, row count,
  and checksum while retaining a clear v1 reader.
- Shard candidate manifests by stable image ranges or connected retrieval
  buckets. A top-level index must list every shard and its checksum; merging
  shards must reject overlap, omission, reversed duplicates, and mismatched
  image order.

### Feature, match, and track storage

- Feature extraction and local matching must be streaming, resumable, and
  atomic. Existing files are skipped only after schema/config/source hashes
  validate.
- Pair shards must preserve exact input order and verifier metadata so a fresh
  reconstruction can replay without recomputing descriptors or matches.
- Track construction must process connected components or bounded windows from
  disk-backed correspondence state. It must not retain every image's
  descriptors, every pair's matches, and every landmark simultaneously.
- Component recovery may add a bounded, explicitly logged shard after a
  registration stall. It must not use a reference model, GT, or hidden
  full-match statistics.

### Observability and failure handling

- Log the first missing bridge/component, candidate degree, shard state, and
  reason for every registration rejection. Distinguish “not proposed”, “raw
  match absent”, “verification rejected”, “track unavailable”, and “PnP/BA
  rejected”.
- Keep a deterministic state machine (`planned → running → complete` or
  `failed`) for every shard. A failed shard is retried into a new temporary
  path; a partial final file is never treated as complete.
- Keep mapping defaults and snapshot replay semantics unchanged. New streaming
  or adaptive behavior must be opt-in until the courtyard exhaustive and 200
  pair gates plus the applicable cross-suite controls pass.

## 6. Future artifact/runbook shape

Each external run root should contain at least:

```text
run.json                    # schema, source/config/tool hashes, terms
images.manifest             # ordered names, dimensions, source hashes
calibration.manifest        # camera IDs/params and checksum
features/index.json         # shard paths, rows, schema/config/source hashes
features/shards/*.part      # atomically completed per-image files
candidates/index.json       # v1/v2 manifests and budget/retrieval metadata
candidates/shards/*.part   # deterministic pair ranges
matches/index.json          # verifier config/outcome and pair-shard hashes
mapping/summary.json        # counters, timings, RSS, components, metrics
mapping/logs/*
models/*                    # cameras/images/points + hashes
```

The top-level run manifest must identify the exact source revision and must
never embed licensed images, databases, descriptors, or generated models in
the repository.

The initial candidate/match sharding implementation is
[`scripts/benchmark_electro.py`](../scripts/benchmark_electro.py) (or its
`benchmark_electro.sh` wrapper). It preserves the Rust candidate manifest
metadata, splits its ordered pair stream into image-name-bound shards, and
records each shard's SHA-256 in `candidates/index.json`. A match worker invokes
`unordered_sfm_demo --export-verified-pairs-only`, so it writes the complete
verifier stream without running a partial mapper. `matches/index.json` changes
to `complete` only after the snapshot hash validates. The
`merge_verified_pair_snapshots` helper rejects image/configuration mismatches
and overlapping pairs, recomputes stream hashes, and writes the merged snapshot
atomically. The final mapper is the only process that consumes all merged
correspondences and may apply the explicit `--max-mapper-matches-per-pair N`
resource guard.

For an already extracted feature bank:

```sh
scripts/benchmark_electro.sh --prepare \
  --features-dir /external/electro/features \
  --calibration-dir /external/electro/calibration \
  --artifact-root /external/electro/run \
  --pairs-per-shard 256 \
  --local-stem-window 3 \
  --rig-local-grouping \
  --candidate-budget 12000
scripts/benchmark_electro.sh --match --resume \
  --features-dir /external/electro/features \
  --calibration-dir /external/electro/calibration \
  --artifact-root /external/electro/run
scripts/benchmark_electro.sh --map \
  --features-dir /external/electro/features \
  --calibration-dir /external/electro/calibration \
  --artifact-root /external/electro/run \
  --max-mapper-matches-per-pair 128 \
  --no-final-ba
```

`--no-final-ba` is useful when recording the incremental-mapper phase in
isolation; omit it for a production reconstruction that includes final bundle
adjustment.

`--verify-only` performs only hash/schema checks and does not require the Rust
executables. `--run` combines the three stages. Existing feature files,
candidate shards, match snapshots, and merged snapshots are reused only after
full content-hash validation; a same-size or partial file is never accepted.

For mapper profiling, `VISLOC_SFM_TIMING=1` adds compact seed/growth checkpoints
and one `sfm-timing-ba` record per bundle-adjustment invocation. The seed
summary reports rejected/accepted trial counts, successful trials include their
elapsed time, and growth checkpoints report bounded registration intervals;
none of these enable the per-image provenance stream. The BA records separate
point/pose assembly, solver, and writeback time. The plain, non-COLMAP-style count-ranked
incremental path also maintains its 2D--3D candidate counts incrementally as
tracks become points, and triangulates only tracks observed by the newly
registered image; after a BA, the next successful registration performs one
full pending-track refresh before returning to the targeted path. COLMAP-style
growth retains its historical full scan because local/global completion can
change support outside the newly registered image. These are support-preserving
changes: the count tie order,
triangulation gates, and periodic-BA schedule are unchanged. Use the timing
environment variable together with `/usr/bin/time -v` when recording wall time
and peak RSS, and report the number of full versus targeted scans from the
`sfm-timing` growth line.

`--no-final-ba` now also suppresses the post-BA filter/retriangulation rounds in
the plain mapper. Those rounds could otherwise launch additional global solves
after a caller explicitly requested a growth-only timing/control run. The
default production path (final BA enabled) retains the existing support
refinement. For solver experiments, `--ba-linear-solver sparse` selects the
existing block-sparse Schur backend; omission retains the dense reproducibility
baseline. Solver changes must pass the same registered/track/reprojection and
official-reference accuracy gates before a large run.

For a resumable Rust SIFT extraction (the feature bank itself is outside the
repository), use the stream flags explicitly:

```sh
cargo run --release --features image-io --example unordered_sfm_demo -- \
  --feature-extractor sift --images-dir /external/electro/images \
  --input-colmap-calibration /external/electro/calibration \
  --sift-stream-export --sift-stream-resume \
  --export-features-dir /external/electro/features --export-features-only \
  --out-colmap /external/electro/unused-model
```

Each image gets a `{stem}_sift_stream_manifest.txt` sidecar. Removing or
changing an image, extractor option, calibration assignment, feature file, or
locus file invalidates that sidecar and causes only that image to be rebuilt.

For a multi-camera rig whose flat names repeat timestamps (for example
`cam4_1474975187520882738.png` through `cam7_...`), pair generation must opt
into `--rig-local-grouping`. It connects nearby timestamps only within each
camera and adds the bounded same-timestamp cross-camera edges; the default
unique-global-stem schedule intentionally rejects those duplicate timestamps.
For the fixed-budget electro comparison, `--pair-source temporal-pyramid`
selects deterministic within-camera offsets `1,2,4,8,16,32`, adds the
same-timestamp cross-camera edges, and fills the remaining budget by VLAD
score. The generated manifest records this policy and its maximum offset, so
the COLMAP control and visloc mapper consume the identical candidate list.

### Official COLMAP same-candidate control

The COLMAP control is described by
[`benchmarks/electro/colmap_1200_v1.json`](../benchmarks/electro/colmap_1200_v1.json)
and prepared with
[`scripts/benchmark_electro_colmap.py`](../scripts/benchmark_electro_colmap.py).
It reads the validated visloc `candidates/index.json`, checks every shard and
the source-manifest hash, and emits one deterministic `candidate_pairs.txt`
whose names are the same flat `camN_timestamp.png` names in the feature bank.
COLMAP's `matches_importer --match_type pairs` then computes SIFT matches and
geometric verification only for those 12,000 pairs; it is not an exhaustive
control and it must not be compared to an exhaustive graph without saying so.

The four camera folders are passed to four separate `feature_extractor`
invocations. This preserves the official per-camera PINHOLE intrinsics while
keeping the stored image names identical to the visloc manifest. The mapper
reads the flat image root. `--prepare` only validates and writes the external
plan; `--run` is an explicit CPU-heavy opt-in:

```sh
python3 scripts/benchmark_electro_colmap.py --prepare \
  --candidate-index /external/electro/run/candidates/index.json \
  --output-root /external/electro/run/colmap_candidate12000 \
  --image-root /external/electro/staging1200/images \
  --camera-root /external/electro/staging1200/camera_shards \
  --calibration-dir /external/electro/staging1200/calibration \
  --colmap-binary /opt/colmap/bin/colmap \
  --colmap-library-path /opt/colmap/lib
python3 scripts/benchmark_electro_colmap.py --run \
  --plan /external/electro/run/colmap_candidate12000/plan.json
```

Each feature, matching, and mapping phase has a separate log and GNU
`time -v` resource file. Score only a completed model, after mapping, with:

```sh
python3 scripts/score_electro_model.py \
  /external/electro/electro/rig_calibration_undistorted/images.txt \
  /external/electro/run/colmap_candidate12000/models/0/images.txt \
  --output-json /external/electro/run/colmap_candidate12000/score.json
```

The scorer joins flat and official names by `(camera number, numeric
timestamp)`, performs a Sim(3) camera-centre alignment, and reports registered
counts plus RMSE/median/p95/max and per-camera errors. The reference model is
not included in any candidate, matching, or mapper command.

### 2026-08-31 electro-1200 temporal-pyramid result

The first complete same-candidate comparison used the frozen
`temporal-pyramid-v1` 12,000-pair manifest (within-camera offsets
`1,2,4,8,16,32`, same-timestamp cross-camera pairs, then VLAD fill). Ground
truth was used only by the post-mapping scorer.

| implementation | registered | mapper wall | peak RSS | RMSE | median | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| visloc, cap64 + sparse final BA, memory optimized | 1193 / 1200 | 513.05 s | 3,977,652 KiB | 0.1194 m | 0.1099 m | 0.1436 m |
| visloc, cap64 + sparse final BA, original memory path | 1193 / 1200 | 506.36 s | 8,485,216 KiB | 0.1194 m | 0.1099 m | 0.1436 m |
| COLMAP 3.9.1 CPU control | 1200 / 1200 | 4,929.56 s | 1,255,996 KiB | 0.0468 m | 0.0316 m | 0.0968 m |

The memory-optimized visloc mapper process was 9.61x faster wall-to-wall. Its
instrumented SFM core took 455.51 s (10.82x below the COLMAP mapper wall), of
which 452.90 s was the three sparse final-BA solves. Growth itself took 2.38 s.
The original and optimized models are byte-identical; the 1.3% wall-time
difference is treated as run-to-run variation, not a speed improvement. This
establishes a real speed advantage, but it is not yet a quality-equivalent
win: COLMAP registered seven more images and its aligned camera-centre RMSE
was 2.55x lower. The acceptance gate for the next optimization therefore
requires both the current visloc speed class and a substantial reduction of
the accuracy gap; reprojection error alone is not an adequate gate.

The phase comparison also exposes the opposite bottleneck in matching.
COLMAP's four CPU feature phases summed to 304.12 s and its same-pair matcher
took 471.37 s. The current 24-shard visloc matcher summed to 2,085.89 s because
each shard reloads the full feature bank. A persistent feature-bank worker (or
memory-mapped descriptor store) is required before claiming an end-to-end
speed win.

Memory is now a first-class acceptance gate. The original visloc peak was
6.76x the COLMAP mapper peak; the first three safe ownership/lifetime changes
reduced it by 4,507,564 KiB (53.1%) to 3,977,652 KiB, or 3.17x the COLMAP
peak, with a byte-identical model and unchanged official score. Set
`VISLOC_SFM_MEMORY=1` to emit `/proc/self/status`
RSS/HWM checkpoints at feature loading/canonicalization, snapshot import and
materialization, track construction, seed growth, BA assembly/Schur/factor,
final refinement, and output assembly. Reduce memory in this order, with the
electro score and courtyard controls guarding every behavioral change:

1. **Completed first pass:** move/drop the verified snapshot and
   materialization metadata as soon as hashes and mapper inputs are
   established. On the 1,200-image cap64 growth-only A/B, peak RSS fell from
   8,461,428 KiB to 7,291,408 KiB (1,170,020 KiB, 13.8%) while
   `cameras.txt`, `images.txt`, and `points3D.txt` remained byte-identical and
   the official score remained unchanged (1193 registered, 0.2414 m RMSE).
2. **Completed:** remove simultaneous native/canonical descriptor banks;
   retain only native keypoint coordinates for export and canonicalize the
   single descriptor bank in place. The same 1,200-image A/B reduced peak RSS
   again from 7,291,408 KiB to 3,951,368 KiB (3,340,040 KiB, 45.8%). Combined
   with step 1 this is a 4,510,060 KiB (53.3%) reduction from the original
   8,461,428 KiB peak. The larger-than-one-bank saving also removes the
   transient descriptor clone during canonicalization. All three COLMAP model
   files remained byte-identical to the original control.
3. **Completed first pass:** move the dense pose normal matrix into the Schur
   system instead of cloning it, then release the reduced dense matrix after
   sparse triplet generation and release solver temporaries before landmark
   back-substitution. For 1193 poses each avoided dense overlap is about
   390 MiB. Direct block-sparse assembly remains the next step because the
   initial dense matrix and scalar triplet scan still exist.
4. Replace whole-model LM rollback clones with a bounded change journal.
5. Stream or memory-map feature/match shards so mapping does not require the
   complete descriptor and correspondence banks to remain resident.

## 7. Milestone-4 stop/go checklist

Before starting the first download, confirm:

- [ ] official URL, license/terms, archive size, and current availability were
      rechecked and recorded;
- [ ] external storage and network quotas meet the selected tier;
- [ ] a 300-image ETH3D `electro` probe has a complete atomic manifest and a
      second-run hash check;
- [ ] courtyard exhaustive verify-only and the 200-pair manifest replay pass;
- [ ] the proposed candidate budget and shard layout are fixed before matching;
- [ ] the run can stop cleanly at a component/shard boundary and resume;
- [ ] no GT/extrinsics/reference model is available to candidate selection or
      mapping decisions.

The electro-1200 run above completes the first 1k-scale execution of this
plan. The checklist remains the required preflight for larger 10k+/100k+
tiers and for any fresh external dataset root.

## 8. Milestone 5 measured outcome

All ten ETH3D low-res scenes were mapped independently: 9,996/10,008 supplied
cameras register (99.88%), with no mapper above 3.32 GiB. The separate
synthetic restart gate emits about `32N` pairs and replays the 100k tier at
33.6 MiB without an `N²` matrix.

The connected environment is commit-pinned OpenLORIS `corridor1-1`: 5,000
frames from each T265 fisheye, globally timestamp sorted and rectified from the
official Kannala-Brandt calibration. One external feature bank feeds four
prefix tiers under the same temporal-pyramid + VLAD-fill `7N` policy.

| Tier | Candidate / verified pairs | Registered | Total wall | Peak phase RSS |
| --- | ---: | ---: | ---: | ---: |
| 1k | 7,000 / 6,869 | 989/1,000 | 2:25 | 280,916 KiB |
| 2.5k | 17,500 / 16,321 | 1,223/2,500 | 7:10 | 401,308 KiB |
| 5k | 35,000 / 31,521 | 1,212/5,000 | 20:33 | 691,840 KiB |
| 10k | 70,000 / 58,879 | 199/10,000 | 1:03:45 | 1,869,412 KiB |

The resource gate passes below 2 GiB, but the connected quality gate fails.
The manifest is sparse, yet exact VLAD ranking still computes all-image
similarities and loads the file feature bank; the 10k candidate phase alone is
49:49 at 1.14 GiB. UnionFind track conflicts also collapse usable support as
the graph grows. Do not call this a successful connected 10k reconstruction.
The frozen evidence and negative controls are in
[`m5-openloris-connected-scale-validation.json`](../benchmarks/electro/m5-openloris-connected-scale-validation.json).

M6 keeps that failed baseline immutable. Streamed VLAD plus scale-aware LSH
reduces 10k candidate generation to 8:51. Replaying the same 58,879 verified
pairs with confidence-ordered conflict rejection and one follow-up refinement
registers 3,664/10,000 at 1.140 px mean reprojection. The 18.4× registration
recovery confirms track conflicts were a binding failure, but 36.64% remains
below the connected-scale gate. See
[`m6-ann-streaming.json`](../benchmarks/electro/m6-ann-streaming.json) and
[`m6-conflict-aware-tracks.json`](../benchmarks/electro/m6-conflict-aware-tracks.json).
