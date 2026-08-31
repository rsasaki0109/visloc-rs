# Reproducible courtyard benchmark

`benchmarks/courtyard/exhaustive_control.json` pins the high-resolution ETH3D
courtyard control used by the README. The 38 source images, six-column
COLMAP-SIFT files, all-pairs raw matches, calibration, and completed model
outputs remain in the external artifact directory; none are checked into this
repository.

The one-command entry point is:

```bash
scripts/benchmark_courtyard.sh --verify-only
```

The default artifact root is the path recorded in the manifest. On another
machine, pass the durable copy explicitly (or set
`COURTYARD_ARTIFACT_ROOT`). The source images and calibration model are
licensed external inputs and are pinned as absolute paths in this checkout;
when they live elsewhere, pass both overrides as well. Their files are still
checked against the hashes in the manifest:

```bash
scripts/benchmark_courtyard.sh \
  --artifact-root /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/artifacts/colmap_highres_exhaustive_allpairs_20260830 \
  --images-dir /path/to/dslr_images_undistorted \
  --calibration-model /path/to/dslr_calibration_undistorted \
  --verify-only
```

## Modes and gates

`--verify-only` is the fast, read-only mode and is the default when neither
mode flag is supplied. It checks:

- the artifact checksum manifest, all 38 feature-file hashes and row counts
  (439,481 rows total), all 38 source-image hashes, and the complete 703-pair
  / 306,324-correspondence raw-match stream;
- the three calibration files and the durable visloc and official COLMAP text
  models, including their exact hashes and registered/track/observation
  counts; and
- the tracked README PNG/GIF hashes, dimensions, frame count, and central
  claim/table references.

It also validates the stored visloc score: 38/38 cameras and centre RMSE
`0.005379 m`, below the default `0.01 m` gate. `--max-rmse-m` can make the
threshold stricter, but a looser threshold should be used only for a clearly
labelled exploratory run.

`--full` first repeats those input checks, builds the image-IO example, runs
the normal mapping command, scores the newly written model against the
calibration model, and writes logs plus a deterministic `summary.json` under
`--output-dir`. A fresh output directory is required; `--force` removes only
the explicitly named output directory. The mapping command receives features,
raw matches, source images, and per-image calibration only. The calibration
model is opened again for centre scoring after mapping and is never supplied
as mapping ground truth.

```bash
scripts/benchmark_courtyard.sh --full \
  --output-dir /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/runs/benchmark_courtyard_$(date +%Y%m%d) \
  --colmap-control validate --visuals check
```

The effective visloc configuration is pinned in the JSON manifest: exhaustive
703-pair import, ratio 0.8, full verification, per-image PINHOLE calibration,
plain incremental mapping, 100,000 PnP iterations, eight minimum PnP inliers,
geometry-guided recovery, post-registration refinement, final iterative
refinement, and the recorded `visibility` next-image policy. Output paths are
not part of the input hashes, so a rerun can use any fresh external output
directory. Use `--no-build` only when the release example was built with
`--features image-io` from the same source revision.

The final stdout document is sorted, indented JSON. By default it is also
atomically written to `target/benchmark_courtyard/verify-summary.json` in
verify-only mode or `summary.json` in the full output directory. Use
`--summary-json PATH`, or `--summary-json -` to make stdout the only summary
channel. Child output is captured in named log files; a failed command prints
the command and its last log lines with a remediation hint.

## Pre-match candidate schedules

The exhaustive control is still the accuracy control: it proposes all
`38 * 37 / 2 = 703` unordered image pairs. Optional `--candidate-strategy`
schedules select image pairs before local matching and verification, using only
image names/order and cheap VLAD retrieval. In this frozen-artifact benchmark,
the raw match stream is replayed after selection, so no descriptor or local
match recomputation is claimed by the reduced-run timing.

The bounded schedule that currently meets the full gate is:

```bash
scripts/benchmark_courtyard.sh --full --no-build \
  --candidate-strategy local-vlad-union-3-8-200 \
  --write-candidate-manifest /path/to/runs/courtyard-union/candidates.txt \
  --output-dir /path/to/runs/courtyard-union \
  --colmap-control validate --visuals skip
```

The manifest is image-name bound and versioned (`visloc_candidate_manifest_v1`)
with atomic write, duplicate/order checks, and a SHA-256 summary. A later run
can skip candidate generation and replay it directly with
`--candidate-manifest PATH`. `--allow-incomplete` returns negative A/B
registration and score results as JSON instead of turning incompleteness into
a process failure. Summaries include candidate count, verified pairs/inliers,
mapping elapsed time, tracks/observations, reprojection, score, and a
machine-readable `gate_passed` field.

Measured on the frozen high-resolution artifact (ratio 0.8, cross-check/full
verifier, imported raw matches, same recovery/post/final mapper):

| pre-match schedule | candidates | verified / inliers | registered | tracks / observations | reproj. | centre RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exhaustive control | 703 | 366 / 261,724 | 38/38 | 43,852 / 152,432 | 0.579 px | 0.5379 cm |
| local stem ≤3 + VLAD top-8, budget 200 | 200 | 172 / 199,871 | 38/38 | 45,016 / 148,192 | 0.537 px | 0.66 cm |
| VLAD top-8 (non-mutual) | 188 | 158 / 187,191 | 38/38 | 44,800 / 135,485 | 0.502 px | 14.17 cm |
| VLAD top-8 (mutual) | 116 | 112 / 156,140 | 13/38 | 16,646 / 50,278 | 0.478 px | 33.97 cm* |
| VLAD top-4 (non-mutual) | 99 | 93 / 138,421 | 13/38 | 16,725 / 47,915 | 0.456 px | 25.20 cm* |
| sequential stem ≤3 | 108 | 100 / 136,553 | 23/38 | 25,315 / 77,273 | 0.499 px | 1.08 cm* |
| vocab-tree base, top-4 | 104 | 89 / 129,766 | 38/38 | 44,002 / 121,742 | 0.425 px | exploratory |

An asterisk marks a partial-reconstruction score over its registered subset.
The 200-pair schedule removes 503 of 703 candidates (71.6%) while retaining
38/38 and the ≤1 cm gate. Its reproducibility-run manifest hash is
`2d654b9ed124ec1732d65f4f3829dfadba5e3e978b085844b59c1ee5e5ed32c1`.
Candidate selection contains no ground-truth, official-extrinsic, raw-match,
or inlier-count input; imported raw matches are used only after selection for
verification replay. The exhaustive model remains the durable README/champion
artifact.

## COLMAP and README controls

`--colmap-control validate` (the default) checks the durable official
COLMAP CPU-SIFT model: 38/38, 38,422 points, 169,590 observations, and the
recorded 1.6166 cm calibration-proxy centre RMSE. `--colmap-control skip`
omits this control when only the visloc artifact is available. In a full run,
`--colmap-control run` additionally requires the `colmap` executable and runs
it against the durable calibrated database into a new output subdirectory;
its result is scored for comparison but is not subjected to the visloc 1 cm
gate.

`--visuals check` verifies the committed PNG/GIF and README references without
plotting dependencies. In a full run, `--visuals regenerate` invokes
[`generate_courtyard_readme_visuals.py`](../scripts/generate_courtyard_readme_visuals.py)
in a fresh output directory and requires byte-identical hashes and dimensions
for the committed assets. This makes plotting-library drift an explicit,
actionable failure rather than silently rewriting tracked visuals.

## Artifact provenance and CI policy

The default durable artifact root is approximately 465 MiB and contains
licensed dataset-derived data and runtime outputs. The manifest records exact
SHA-256 values for those inputs and the source-image files; copy it outside
the repository and pass `--artifact-root` on a different host. Do not commit
the datasets, databases, feature files, logs, or generated models.
The checked `SHA256SUMS` file itself is pinned as
`12e91cd3a2e595625ef167d8cd8a2af6310d3ea3cd1e3b1a0c2f8264004fa96e`.

Hosted CI intentionally does not run the full benchmark: the dataset size,
external artifact storage, and licensing/provenance requirements make a
public runner invocation misleading. The normal CI suite runs the
dependency-free parser/hash/threshold tests and shell-syntax smoke check.
For a self-hosted or scheduled run, provision the durable artifact and use:

```bash
COURTYARD_ARTIFACT_ROOT=/path/to/colmap_highres_exhaustive_allpairs_20260830 \
  scripts/benchmark_courtyard.sh \
    --images-dir /path/to/dslr_images_undistorted \
    --calibration-model /path/to/dslr_calibration_undistorted \
    --verify-only
COURTYARD_ARTIFACT_ROOT=/path/to/colmap_highres_exhaustive_allpairs_20260830 \
  scripts/benchmark_courtyard.sh --full \
    --images-dir /path/to/dslr_images_undistorted \
    --calibration-model /path/to/dslr_calibration_undistorted \
    --output-dir /path/to/benchmark-runs/courtyard-$(date +%Y%m%d) \
    --colmap-control validate --visuals regenerate
```

The JSON summary is suitable for a self-hosted nightly wrapper to archive and
compare. There is no hosted workflow claiming a result without the external
artifact.

The manifest retains `future_scale.feature_shards` for a future large-scale
runner and records the versioned candidate-manifest format used by the
optional schedules above. Feature sharding remains reserved; candidate
reduction is explicit and does not replace the exhaustive control.

## Large-scale follow-up plan

The next validation milestone is documented in
[`docs/large_scale_unordered_sfm_plan.md`](large_scale_unordered_sfm_plan.md).
It starts with a 300-image ETH3D low-res `electro` probe, then a 1,200-image
run, and defines the resource, candidate-recall, shard-integrity, and
reproducibility gates before any 10k+ experiment. This M4 document is
planning-only; it does not download data or alter this courtyard control.
