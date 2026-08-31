# Large-scale unordered-SfM validation plan

**Milestone 4 — planning only (2026-08-31).** This document defines the next
validation stages; it does not download data, run a large reconstruction, or
change the production default. The current 38-image courtyard result and the
M3 reduced-candidate result remain the acceptance controls.

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
   atomically. A manifest records config hash, source-image hash, row count,
   and schema version.
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
- Keep `visloc_candidate_manifest_v1` image-name-bound and replayable. A future
  `v2` should add dataset/config hash, retrieval-model hash, shard ID, pair
  range, source/score metadata, row count, and checksum while retaining a clear
  v1 reader.
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
the repository. A later benchmark helper can accept `--artifact-root`,
`--verify-only`, `--resume`, and `--shard` without changing the current
courtyard command.

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

This plan intentionally leaves the first large download and all 10k+/100k+
measurements for a subsequent execution request.
