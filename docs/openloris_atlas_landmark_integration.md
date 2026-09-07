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
| Observation-supported images | 9,996 | 9,996 |
| Observation-supported rig frames | 4,998 | 4,999 |
| ATE RMSE (m) | 0.391778 | 0.384307 |
| ATE p95 (m) | 0.643698 | 0.638669 |
| Observation-weighted mean reprojection (px) | 0.633124 | 0.903003 |
| Landmarks | 353,787 | 113,710 |
| Observations | 1,453,205 | 1,947,815 |

The atlas retains 83.03% of unique source observations. Low reprojection error
does not compensate for lost registration or worse trajectory accuracy.
COLMAP also has two unsupported individual camera images, but in distinct rig
frames whose other camera is supported. In the atlas both unsupported images
belong to frame 4493, leaving one fewer supported rig frame. Thus equal counts
of supported camera images do not imply equal rig registration.
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

### Unsupported-frame recovery pilot

The default-off `--recover-zero-support-frames` pilot recovered frame 4493
without relaxing triangulation gates. Excluding the target frame leaves 12
valid anchor points; deterministic calibrated PnP accepts 10 inliers. Final
retriangulation retains 10 target landmarks and adds 124 total observations.
Only the target frame's two sensor poses change. The boundary centre step
becomes 0.057975 m, and all 4,494 main-component rig frames regain support.

Using the unchanged tail component gives 4,999 supported rig frames and 9,997
supported individual images across 9,998 pose rows. RMSE/p95 improve only to
0.391465/0.643175 m, still above COLMAP's frozen limits. Main-component mean
reprojection is 0.643293 px, with valid bidirectional references and no
nonpositive-depth observations. Integration/recovery alone took 10.69 s and
561,888 KiB peak RSS in this pilot; this is not a complete mapper speed claim.
See [recovery evidence](../benchmarks/electro/m8-openloris-atlas-recovery-v1.json).

The final recovery implementation passes 17 example tests, including full
synthetic acceptance and rollback when six PnP inliers leave only five target
landmarks after triangulation. It uses sparse pose overrides and target-image
observation-index lookups, without cloning all image/keypoint data. A replay
produces all six main-component files byte-for-byte identically to the pilot
at 10.80 s / 508,896 KiB peak RSS. Default-off main output and enabled tail
output also match their fixed-pose baselines exactly. All 319,137 pre-existing
main landmarks retain identical observations, XYZ, RGB and stored error.

### Baseline command

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

For an independent read-only check of either engine's serialized PINHOLE model:

```bash
python3 scripts/audit_colmap_pinhole_model.py /path/to/component-000 /path/to/component-001
python3 -m unittest discover -s tests -p test_audit_colmap_pinhole_model.py
```

The auditor rejects unsupported camera models and malformed references. It
reports nonpositive-depth counts and stored-versus-recomputed mean error;
callers must inspect those values, not treat process success as a quality pass.
The frozen COLMAP models contain 14,593 / 1,325 tracks with multiple keypoints
in the same image (51,707 / 4,562 excess observations), but no duplicate exact
observation references. The auditor reports this distinction instead of
imposing the atlas's stricter merge policy on the comparison engine.
