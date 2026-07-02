# Benchmark Registry

This directory is the evidence ledger for benchmark claims.

There are two record types:

- `runs/**/*.json`: one command execution, with commit, `Cargo.lock` hash, feature flags, dataset/model identity, command, hardware, metrics, artifacts, and status.
- `readme_claims_v1.json`: public README benchmark rows in machine-readable form. A row can cite registered run IDs, historical docs, and external published numbers, but its `claim_kind` must say which kind of evidence it uses.
- `claim_matrix_v1.json`: per-system comparison rows scoped by dataset,
  sequence, sensor mode, metric, protocol, and verdict. This is where
  ORB-SLAM / COLMAP / VINS-style comparisons stay narrow enough to audit.

Run manifests use [`schema_v1.json`](schema_v1.json). The Python helper validates the subset of the schema that matters for release hygiene and renders public tables.

## Capture A Run

```sh
python scripts/benchmark_registry.py capture \
  --out benchmarks/registry/runs/kitti/seq00_full_20260618.json \
  --benchmark-id kitti-multiseq \
  --benchmark-name "KITTI multi-sequence full-stack stereo SLAM" \
  --script scripts/run_kitti_multiseq_benchmark.sh \
  --docs docs/kitti_multiseq_benchmark.md \
  --dataset-name "KITTI odometry" \
  --dataset-sequence 00 \
  --dataset-version "odometry grayscale" \
  --dataset-path "$HOME/datasets/kitti_seq00_full" \
  --result-kind visloc_run \
  --claim-scope exploratory \
  --status success \
  --command "scripts/run_kitti_multiseq_benchmark.sh --sequence 00 --data-root ..." \
  --feature image-io \
  --profile release \
  --config min_stereo_confidence=0.5 \
  --config min_temporal_confidence=0.5 \
  --metric ate_rmse_se3_m=1.23:m \
  --primary-metric ate_rmse_se3_m \
  --artifact trajectory=target/kitti_multiseq/seq00/full/vo_poses.txt \
  --artifact summary=target/kitti_multiseq/seq00/full/summary.txt
```

For DNF or failure runs, keep the manifest:

```sh
python scripts/benchmark_registry.py capture \
  --out benchmarks/registry/runs/euroc/v203_dnf_20260618.json \
  --benchmark-id euroc-loop-closure \
  --dataset-name "EuRoC MAV" \
  --dataset-sequence V2_03 \
  --status dnf \
  --failure-reason "tracker starved during sensor blackout"
```

The KITTI adaptive/fixed stereo depth-gate smoke is rendered from two
exploratory manifests into
[`docs/generated/kitti_adaptive_depth_gate_smoke.md`](../../docs/generated/kitti_adaptive_depth_gate_smoke.md).
It is a gate-diagnostics artifact, not a trajectory benchmark claim:

```sh
python scripts/summarize_kitti_adaptive_depth_gate_smoke.py \
  --registry-dir benchmarks/registry/runs/kitti \
  --out docs/generated/kitti_adaptive_depth_gate_smoke.md
```

For the covisibility local-BA A/B, start from
[`templates/euroc_covisibility_local_ba_v1.json`](templates/euroc_covisibility_local_ba_v1.json)
and keep separate disabled-baseline and enabled-run manifests under
`runs/euroc/`.

The helper below runs both sides of that A/B and captures manifests from each
`summary.txt`:

```sh
python scripts/run_euroc_covisibility_local_ba_ab.py \
  --euroc-root /datasets/euroc \
  --sequence MH_01_easy --sequence V1_01_easy --sequence V2_01_easy \
  --max-frames 1500 \
  --min-active-observations 1 \
  --fallback-min-boundary-observations none \
  --demo-args "--gravity 0,0,-9.81 --feature-extractor hog --cross-check-matcher --keyframe-min-translation 0.1 --max-pose-jump-meters 0.2 --stereo-bootstrap-strict"
```

Use `--min-active-observations <n>` to A/B the active-keyframe local
observation gate. Values above the default `1` skip BA solves whose active
keyframe has too little support from the selected local landmark set.
Use `--fallback-min-boundary-observations <n|none>` to A/B a lower fixed
boundary-keyframe threshold only when strict selection would otherwise produce
no local landmarks.
Registered covisibility BA runs also capture
`covisibility_local_ba_elapsed_ms_total`, `covisibility_local_ba_elapsed_ms_mean`,
and `covisibility_local_ba_elapsed_ms_max`, measured over every trigger
including selection failures, so window-size and landmark-cap sweeps can be
judged against runtime as well as ATE/RPE.

The MH-class active-observation sweep is rendered from the registry into
[`docs/generated/euroc_active_observation_sweep.md`](../../docs/generated/euroc_active_observation_sweep.md):

```sh
python scripts/run_euroc_active_observation_sweep.py \
  --euroc-root /datasets/euroc \
  --sequence MH_01_easy --sequence MH_03_medium --sequence MH_05_difficult \
  --active-floor 20 --active-floor 50 \
  --max-frames 400
```

To regenerate only the Markdown from already-captured manifests:

```sh
python scripts/summarize_euroc_active_observation_sweep.py \
  --registry-dir benchmarks/registry/runs/euroc \
  --out docs/generated/euroc_active_observation_sweep.md
```

`scripts/benchmark_registry.py check-generated` also treats a missing
floor/sequence/variant cell as a failure, even if the Markdown already contains
a `missing` row. Re-run the sweep wrapper when the check reports a missing
active-observation manifest.

The tight-VIO local-BA writeback gate smoke is an exploratory diagnostics table
for cost-ratio and refined-velocity writeback gates, rendered from
`euroc-tight-vio-local-ba-gates` manifests into
[`docs/generated/euroc_tight_vio_gate_smoke.md`](../../docs/generated/euroc_tight_vio_gate_smoke.md):

```sh
python scripts/summarize_euroc_tight_vio_gate_smoke.py \
  --registry-dir benchmarks/registry/runs/euroc \
  --out docs/generated/euroc_tight_vio_gate_smoke.md
```

Use the `adaptive_velocity` rows as the primary policy comparison; raw
`gated_*mps` rows are fixed-cap safety-ceiling A/B runs.

The longer MH_03-only A/B renders
[`docs/generated/euroc_tight_vio_gate_mh03_1500.md`](../../docs/generated/euroc_tight_vio_gate_mh03_1500.md)
from the same manifests with `--max-frames 1500` and only
`baseline` / `adaptive_velocity` variants.

Unlike the required sweeps above, optional gate caps that have not been run yet
stay visible as `missing` rows so exploratory evidence is not mistaken for a
complete release benchmark.

The MH_03 covisibility-BA runtime smoke sweep varies the local landmark cap and
renders `docs/generated/euroc_covisibility_runtime_sweep.md`:

```sh
python scripts/run_euroc_covisibility_runtime_sweep.py \
  --euroc-root /datasets/euroc \
  --sequence MH_03_medium \
  --landmark-cap 100 --landmark-cap 200 --landmark-cap 400 \
  --max-frames 80 \
  --max-neighbor-keyframes 10 \
  --max-boundary-keyframes 10 \
  --min-active-observations 20
```

Regenerate only the runtime Markdown from existing manifests:

```sh
python scripts/summarize_euroc_covisibility_runtime_sweep.py \
  --registry-dir benchmarks/registry/runs/euroc \
  --out docs/generated/euroc_covisibility_runtime_sweep.md \
  --neighbor-keyframes 10 \
  --boundary-keyframes 10
```

The matching MH-class window-cap smoke sweep keeps the landmark cap fixed and
varies neighbor/boundary keyframe caps, rendering
`docs/generated/euroc_covisibility_window_sweep.md`:

```sh
python scripts/run_euroc_covisibility_window_sweep.py \
  --euroc-root /datasets/euroc \
  --sequence MH_01_easy --sequence MH_03_medium --sequence MH_05_difficult \
  --window-cap 5:5 --window-cap 10:10 --window-cap 15:15 \
  --max-frames 80 \
  --landmark-cap 200 \
  --min-active-observations 20
```

Regenerate only the window-cap Markdown from existing manifests:

```sh
python scripts/summarize_euroc_covisibility_window_sweep.py \
  --registry-dir benchmarks/registry/runs/euroc \
  --out docs/generated/euroc_covisibility_window_sweep.md \
  --sequence MH_01_easy --sequence MH_03_medium --sequence MH_05_difficult \
  --window-cap 5:5 --window-cap 10:10 --window-cap 15:15 \
  --min-keyframes 3 \
  --trigger-every 1
```

The longer validation table compares only the current lightweight candidate
against the existing 10/10 budget at 400 frames and renders
[`docs/generated/euroc_covisibility_window_validation.md`](../../docs/generated/euroc_covisibility_window_validation.md):

```sh
python scripts/run_euroc_covisibility_window_sweep.py \
  --euroc-root /datasets/euroc \
  --sequence MH_01_easy --sequence MH_03_medium --sequence MH_05_difficult \
  --window-cap 5:5 --window-cap 10:10 \
  --max-frames 400 \
  --landmark-cap 200 \
  --min-active-observations 20 \
  --summary-out docs/generated/euroc_covisibility_window_validation.md
```

Regenerate only the validation Markdown from existing manifests:

```sh
python scripts/summarize_euroc_covisibility_window_sweep.py \
  --registry-dir benchmarks/registry/runs/euroc \
  --out docs/generated/euroc_covisibility_window_validation.md \
  --max-frames 400 \
  --sequence MH_01_easy --sequence MH_03_medium --sequence MH_05_difficult \
  --window-cap 5:5 --window-cap 10:10 \
  --min-keyframes 3 \
  --trigger-every 1
```

The 400-frame disabled/enabled A/B keeps the selected 10/10 window budget and
renders [`docs/generated/euroc_covisibility_ab_400.md`](../../docs/generated/euroc_covisibility_ab_400.md):

```sh
python scripts/run_euroc_covisibility_local_ba_ab.py \
  --euroc-root /datasets/euroc \
  --sequence MH_01_easy --sequence MH_03_medium --sequence MH_05_difficult \
  --max-frames 400 \
  --profile dev \
  --max-neighbor-keyframes 10 \
  --max-boundary-keyframes 10 \
  --max-landmarks 200 \
  --min-active-observations 20 \
  --fallback-min-boundary-observations none \
  --demo-args "--gravity 0,0,-9.81 --feature-extractor hog --cross-check-matcher --keyframe-min-translation 0.1 --max-pose-jump-meters 0.2 --stereo-bootstrap-strict"
```

Regenerate only the A/B Markdown from existing manifests:

```sh
python scripts/summarize_euroc_covisibility_ab.py \
  --registry-dir benchmarks/registry/runs/euroc \
  --out docs/generated/euroc_covisibility_ab_400.md \
  --max-frames 400 \
  --sequence MH_01_easy --sequence MH_03_medium --sequence MH_05_difficult \
  --enabled-neighbor-keyframes 10 \
  --enabled-boundary-keyframes 10 \
  --enabled-min-keyframes 3 \
  --enabled-trigger-every 1 \
  --enabled-landmark-cap 200 \
  --enabled-min-active-observations 20 \
  --enabled-fallback none
```

The MH_05 mitigation table records the follow-up cadence sweep after the
default 10/10 enabled row regressed that sequence:

```sh
python scripts/run_euroc_covisibility_local_ba_ab.py \
  --euroc-root /datasets/euroc \
  --sequence MH_05_difficult \
  --max-frames 400 \
  --profile dev \
  --only enabled \
  --max-neighbor-keyframes 10 \
  --max-boundary-keyframes 10 \
  --max-landmarks 200 \
  --min-active-observations 20 \
  --min-keyframes 10 \
  --trigger-every 5 \
  --fallback-min-boundary-observations none \
  --demo-args "--gravity 0,0,-9.81 --feature-extractor hog --cross-check-matcher --keyframe-min-translation 0.1 --max-pose-jump-meters 0.2 --stereo-bootstrap-strict"
```

Regenerate the mitigation Markdown from existing manifests:

```sh
python scripts/summarize_euroc_covisibility_mh05_mitigation.py \
  --registry-dir benchmarks/registry/runs/euroc \
  --out docs/generated/euroc_covisibility_mh05_mitigation.md
```

For the tracked-landmark keyframe-policy A/B, start from
[`templates/euroc_keyframe_policy_ab_v1.json`](templates/euroc_keyframe_policy_ab_v1.json).
The helper keeps the fixed baseline and tracked-drop run separate and captures
`keyframe_decisions.csv`, which records whether each mapper-evaluated frame was
selected and which `KeyframeDecisionReason` fired. The registry manifest also
records decision-derived metrics such as `keyframe_selected_count` and
`keyframe_tracked_landmark_drop_count`:

```sh
python scripts/run_euroc_keyframe_policy_ab.py \
  --euroc-root /datasets/euroc \
  --sequence MH_01_easy --sequence V1_01_easy --sequence V2_01_easy \
  --max-frames 1500 \
  --tracked-landmark-ratio 0.9 \
  --demo-args "--gravity 0,0,-9.81 --feature-extractor hog --cross-check-matcher --keyframe-min-translation 0.1 --max-pose-jump-meters 0.2 --stereo-bootstrap-strict"
```

For KITTI loop retrieval work, keep the candidate-stage evidence separate from
end-to-end ATE. `stereo_vo_external_deep_files --loop-closure` writes raw
gated appearance proposals to `loop_candidates.csv`; register recall@K from
that CSV with:

```sh
python scripts/capture_kitti_loop_retrieval_recall.py \
  --sequence 02 \
  --candidates target/kitti_seq02_full/loop_candidates.csv \
  --poses /datasets/kitti_seq02_full/poses_02.txt \
  --distance-threshold-m 10 \
  --min-temporal-gap 50 \
  --min-path-length-m 5 \
  --ks 1 5 20 \
  --dnf-if-recall-at 20=0.01
```

Template: [`templates/kitti_loop_retrieval_recall_v1.json`](templates/kitti_loop_retrieval_recall_v1.json).

## Validate And Render

```sh
python scripts/benchmark_registry.py validate benchmarks/registry/readme_claims_v1.json benchmarks/registry/claim_matrix_v1.json benchmarks/registry/runs
python scripts/benchmark_registry.py render-readme \
  --claims benchmarks/registry/readme_claims_v1.json \
  --out docs/generated/benchmark_snapshot.md \
  --readme README.md
python scripts/benchmark_registry.py render-runs \
  --registry-dir benchmarks/registry/runs \
  --out docs/generated/registered_runs.md
python scripts/benchmark_registry.py render-claim-matrix \
  --matrix benchmarks/registry/claim_matrix_v1.json \
  --with-heading \
  --out docs/generated/benchmark_claim_matrix.md
```

## Evidence Policy

- Do not delete failed or DNF manifests. Mark `status` and `failure_reason`.
- Keep `external_published`, `external_rerun`, `visloc_run`, and `exploratory` distinct.
- `visloc_run` requires `git.commit` and `build.cargo_lock_sha256`.
- ONNX models, exported feature files, trajectories, summaries, and metric JSON should be listed as artifacts or models with hashes when present.
- README headline rows should eventually cite registered run IDs. Until historical results are migrated, keep their `claim_kind` as `documented_historical` or `mixed`.
- A comparison verdict of `behind` is a retained non-win. Do not restate it as
  "beats" in README or release notes unless the matrix row and evidence are
  updated first.
