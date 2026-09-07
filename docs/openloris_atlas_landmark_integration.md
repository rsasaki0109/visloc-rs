# OpenLORIS atlas landmark integration

Status: M8 quality gate **not passed** (2026-09-07). This extends the
[trajectory diagnostic](openloris_rig_atlas_diagnostic.md) with real landmarks;
it is not a promoted mapper or end-to-end performance result.

## Fixed-pose baseline

`integrate_rig_atlas_landmarks` reads the 23 audited source windows one at a
time. Global observations use image identity plus original keypoint index,
never source-local point IDs. Compatible multi-owner tracks are unioned
transactionally; same-image/different-keypoint conflicts are rejected.
Each track is triangulated with at most 64 sampled observations, then checked
against every observation. Positive depth, mean error at most 2 px, and
individual error at most 4 px are required. Independent atlas gauges remain
separate. Camera calibration and full keypoint order are validated.

| Metric | Fixed-pose atlas | Frozen calibrated COLMAP |
|---|---:|---:|
| Pose rows | 9,998 | 9,998 |
| Observation-supported images | 9,996 | 9,998 |
| Observation-supported rig frames | 4,998 | 4,999 |
| ATE RMSE (m) | 0.391778 | 0.384307 |
| ATE p95 (m) | 0.643698 | 0.638669 |
| Observation-weighted mean reprojection (px) | 0.633124 | 0.903003 |
| Landmarks | 353,787 | 113,710 |
| Observations | 1,453,205 | 1,947,815 |

The atlas retains 83.03% of unique source observations. Low reprojection error
does not compensate for lost registration or worse trajectory accuracy.
GT is used only for post-map scoring; this repeatedly evaluated development
sequence is not a held-out generalization result.

The two integration processes took 10.36 s and 3.60 s, with maximum peak RSS
508,296 KiB. This single-run, integration-only measurement excludes source
window mapping and frontend work. It cannot establish a speed win over COLMAP.
Independent output projection checks found valid bidirectional references and
no nonpositive-depth observations. All 13 example tests pass.

## Why frame 4493 loses support

All 14 source observations at frame 4493 pass track ownership merging but
their merged tracks fail triangulation at the stitched pose: nine fail
reprojection and five have nonpositive depth. No collision-policy change or
reprojection-threshold relaxation is justified.

The frame 4492-to-4493 centre step is 0.986510 m in the atlas, versus
0.043069 m in source node 27; rotation steps are 7.983677 and 3.472590 degrees.
This identifies a publication-boundary pose discontinuity. These diagnostics
use camera geometry, not GT. The next bounded experiment is unsupported-frame
pose recovery from other frames' observation-supported geometry, followed by
retriangulation with unchanged gates. It must not triangulate PnP anchors using
the target pose itself. Successful recovery alone would not close the remaining
trajectory-accuracy gate; bounded observation-based refinement is still needed.

## Reproduce

```bash
cargo build --release --example integrate_rig_atlas_landmarks
atlas_root=/home/sasaki/datasets/openloris/corridor1-1-m8-atlas-landmarks-v1
target/release/examples/integrate_rig_atlas_landmarks \
  --rig-manifest /home/sasaki/datasets/openloris/corridor1-1-m8-visloc-rig/tier-10000-champion/rig-manifest.txt \
  --nodes-tsv "$atlas_root/nodes.tsv" \
  --atlas-dir "$atlas_root/atlas-l-newest/component-000" \
  --out-dir /tmp/visloc-atlas-landmark-replay-component-000
```

Run component 001 separately with a distinct output directory. Input hashes,
source audits and durable locations are in
[source evidence](../benchmarks/electro/m8-openloris-atlas-source-audit.json).
Exact output hashes, archived executable hash, resource measurements and scores
are in [fixed-pose evidence](../benchmarks/electro/m8-openloris-atlas-fixed-pose-v1.json).
