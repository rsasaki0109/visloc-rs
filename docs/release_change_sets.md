# Release Change Sets

This page groups the current pre-0.1 release hardening work into reviewable
change sets. It is a review aid: it does not replace `docs/release_checklist.md`
or the benchmark registry.

## P0 Public Release Hardening

Purpose:

- Narrow public claims to evidence-backed benchmark language.
- Keep the README first view free of bulky local showcase images.
- Declare stable-intent API paths and keep broad facades as convenience layers.
- Make CI run the release gates that are already expected locally.

Primary paths:

- `.github/workflows/ci.yml`
- `README.md`
- `CHANGELOG.md`
- `docs/api_stability.md`
- `docs/assets/`
- `docs/gnss_demo.md`
- `docs/interfaces.md`
- `docs/migration.md`
- `docs/progress.md`
- `docs/roadmap.md`
- `docs/timestamped_gnss_image_demo.md`
- `docs/feature_matrix.md`
- `docs/release_checklist.md`
- `docs/generated/benchmark_snapshot.md`
- `benchmarks/registry/readme_claims_v1.json`
- `src/lib.rs`
- `tests/api_stability.rs`
- `tests/test_ci_release_gate.py`
- `tests/test_docs_assets.py`
- `tests/test_feature_matrix.py`
- `scripts/check.sh`
- `scripts/check_docs_links.sh`
- `scripts/check_feature_matrix.sh`
- `scripts/check_msrv.sh`
- `scripts/check_release_metadata.sh`
- `scripts/check_trajectory_evaluation.sh`
- `scripts/package_check.sh`

Validation:

```sh
python -m unittest tests.test_ci_release_gate tests.test_feature_matrix tests.test_docs_assets
python scripts/benchmark_registry.py check-generated
bash scripts/check_msrv.sh
bash scripts/check_feature_matrix.sh
bash scripts/check_docs_links.sh
bash scripts/check_release_metadata.sh
cargo test --test api_stability -- --nocapture
bash scripts/package_check.sh
```

Residual risk:

- The README now avoids broad full-SLAM framing, but any new benchmark row still
  needs registry or claim-matrix coverage before it becomes public copy.
- The root facade remains available for convenience; reviewers should check that
  new examples use canonical paths when teaching public API.

## P0 Adaptive Stereo Depth Gate

Purpose:

- Replace the scene-scale-dependent fixed `--min-depth` default with an internal
  bounded adaptive policy.
- Preserve the fixed gate for A/B and replay of old benchmark recipes.
- Emit per-frame diagnostics so KITTI/EuRoC changes can be inspected rather than
  tuned silently.

Primary paths:

- `crates/vision/src/stereo_vo.rs`
- `examples/online_slam_stereo_vo_kitti_demo.rs`
- `examples/stereo_vo_external_deep_files.rs`
- `docs/interfaces.md`
- `docs/api_stability.md`
- `tests/api_stability.rs`
- `scripts/fetch_kitti_seq00_images.py`
- `benchmarks/registry/runs/kitti/kitti-adaptive-depth-gate-smoke-seq00-*.json`
- `docs/generated/kitti_adaptive_depth_gate_smoke.md`
- `scripts/summarize_kitti_adaptive_depth_gate_smoke.py`
- `tests/test_summarize_kitti_adaptive_depth_gate_smoke.py`

Validation:

```sh
cargo test -p visloc-vision depth_gate -- --nocapture
cargo check --workspace --all-targets
python -m unittest tests.test_summarize_kitti_adaptive_depth_gate_smoke
python scripts/benchmark_registry.py check-generated
```

Residual risk:

- The retained KITTI seq00 smoke is intentionally small. Full KITTI/EuRoC
  sequence evidence is still required before changing headline performance
  claims.
- The fixed gate remains important for replaying existing recipes and should not
  be removed in this cycle.

## P0 Benchmark Registry And Evidence

Purpose:

- Keep README benchmark rows generated from machine-readable evidence.
- Preserve failed or DNF runs with `status` and `failure_reason`.
- Separate external published numbers, documented historical results,
  exploratory runs, and registered reruns.

Primary paths:

- `benchmarks/registry/`
- `scripts/benchmark_registry.py`
- `scripts/capture_kitti_multiseq_run.py`
- `scripts/capture_kitti_loop_retrieval_recall.py`
- `docs/generated/benchmark_snapshot.md`
- `docs/generated/registered_runs.md`
- `docs/generated/benchmark_claim_matrix.md`
- `docs/kitti_multiseq_benchmark.md`
- `docs/kitti_loop_closure_benchmark.md`
- `tests/test_benchmark_registry.py`
- `tests/test_capture_kitti_multiseq_run.py`
- `tests/test_capture_kitti_loop_retrieval_recall.py`

Validation:

```sh
python scripts/benchmark_registry.py validate benchmarks/registry/readme_claims_v1.json benchmarks/registry/claim_matrix_v1.json benchmarks/registry/runs
python scripts/benchmark_registry.py check-generated
python -m unittest tests.test_benchmark_registry tests.test_capture_kitti_multiseq_run tests.test_capture_kitti_loop_retrieval_recall
```

Residual risk:

- Existing historical numbers are now labelled, but not all of them are
  re-executed registered runs yet.
- Artifact size and retention policy should be revisited before adding large
  trajectory or image artifacts directly under the registry.

## P1 Covisibility Local BA And Keyframe Policy

Purpose:

- Add ORB-SLAM-style covisibility local BA as an opt-in online stage.
- Capture runtime, active-observation, fallback-boundary, quality-gate, and
  window-size evidence before considering defaults.
- Keep MH_05 regressions visible instead of hiding them behind averaged wins.

Primary paths:

- `pipelines/slam/src/covisibility_ba.rs`
- `pipelines/slam/src/online_slam.rs`
- `pipelines/slam/src/lib.rs`
- `pipelines/slam/tests/online_slam.rs`
- `pipelines/mapping/src/lib.rs`
- `pipelines/mapping/tests/keyframe_policy.rs`
- `examples/euroc_online_slam_vi_demo.rs`
- `examples/euroc_online_slam_vi_image_demo.rs`
- `docs/motion_based_vi_alignment.md`
- `scripts/run_euroc_covisibility_local_ba_ab.py`
- `scripts/run_euroc_covisibility_runtime_sweep.py`
- `scripts/run_euroc_covisibility_window_sweep.py`
- `scripts/run_euroc_keyframe_policy_ab.py`
- `scripts/summarize_euroc_covisibility_*.py`
- `scripts/summarize_euroc_active_observation_sweep.py`
- `docs/generated/euroc_covisibility_*.md`
- `docs/generated/euroc_active_observation_sweep.md`
- `benchmarks/registry/runs/euroc/`

Validation:

```sh
cargo test -p visloc-slam covisibility -- --nocapture
cargo test -p visloc-mapping keyframe -- --nocapture
python -m unittest \
  tests.test_run_euroc_active_observation_sweep \
  tests.test_run_euroc_covisibility_runtime_sweep \
  tests.test_run_euroc_covisibility_window_sweep \
  tests.test_summarize_euroc_active_observation_sweep \
  tests.test_summarize_euroc_covisibility_ab \
  tests.test_summarize_euroc_covisibility_mh05_boundary_support_gate \
  tests.test_summarize_euroc_covisibility_mh05_mitigation \
  tests.test_summarize_euroc_covisibility_runtime_sweep \
  tests.test_summarize_euroc_covisibility_window_sweep
python scripts/benchmark_registry.py check-generated
```

Residual risk:

- This remains opt-in. MH_01/MH_03 improve in the captured evidence, but MH_05
  still has a tracking regression under the main enabled configuration.
- Runtime spikes are now measured, but wider datasets and longer windows should
  be benchmarked before considering a default.

## P1 KITTI Seq02 Loop Retrieval And Verification Diagnostics

Purpose:

- Measure candidate-stage recall and PnP verification behavior before swapping
  retrieval backends.
- Keep confidence-weighted PnP sampling as an exploratory per-sequence lever,
  not a default policy.

Primary paths:

- `pipelines/slam/src/vo_loop_closure.rs`
- `pipelines/slam/src/loop_closure.rs`
- `pipelines/slam/src/report.rs`
- `scripts/eval_loop_retrieval_recall.py`
- `scripts/export_lightglue_loop_candidate_matches.py`
- `scripts/export_superpoint_lightglue.py`
- `scripts/capture_kitti_loop_retrieval_recall.py`
- `scripts/capture_kitti_multiseq_run.py`
- `scripts/run_kitti_multiseq_benchmark.sh`
- `scripts/run_kitti_loop_closure_benchmark.sh`
- `docs/kitti_multiseq_benchmark.md`
- `docs/kitti_loop_closure_benchmark.md`
- `benchmarks/registry/runs/kitti/`
- `tests/test_loop_retrieval_recall.py`
- `tests/test_export_lightglue_loop_candidate_matches.py`
- `tests/test_export_superpoint_lightglue.py`
- `tests/test_capture_kitti_loop_retrieval_recall.py`
- `tests/test_capture_kitti_multiseq_run.py`

Validation:

```sh
python -m unittest tests.test_loop_retrieval_recall tests.test_export_lightglue_loop_candidate_matches tests.test_export_superpoint_lightglue tests.test_capture_kitti_loop_retrieval_recall tests.test_capture_kitti_multiseq_run
python scripts/benchmark_registry.py check-generated
cargo test -p visloc-slam loop -- --nocapture
```

Residual risk:

- Seq02 confidence-weighted PnP improved the captured exploratory run, but seq00
  2000-frame support runs show external LightGlue paths can regress. Keep it
  opt-in until cross-sequence precision and trajectory effects are stronger.

## Suggested Review Split

Review in this order so mechanical evidence work does not obscure algorithm
changes:

1. **Public release hardening.**
   Include README claim cleanup, API stability docs/tests, feature matrix,
   release CI gates, docs asset cleanup, release checklist, root re-exports, and
   small shell-script portability fixes. This commit should not include new
   trajectory evidence except generated benchmark headline docs that are directly
   tied to README claim wording.

2. **Adaptive stereo depth gate.**
   Include the stereo VO gate policy, CLI replay knobs, per-frame diagnostics,
   KITTI seq00 smoke manifests, generated adaptive-gate summary, and the small
   KITTI fetch/summarizer helpers. Keep the fixed gate in this commit so old
   recipes remain replayable.

3. **Benchmark registry and claim evidence.**
   Include registry schema/templates, README claim registry, claim matrix,
   registered-run renderer, generated benchmark tables, capture helpers, and
   registry tests. This can be reviewed mostly as evidence infrastructure.

4. **Covisibility local BA and keyframe policy.**
   Include the covisibility BA selector/solver wiring, online stats, quality and
   boundary gates, tracked-landmark-drop keyframe trigger, EuRoC runners,
   generated EuRoC evidence tables, and EuRoC registry manifests. Keep the
   feature opt-in and preserve the MH_05 regression notes.

5. **KITTI seq02 loop diagnostics.**
   Include loop retrieval recall tooling, LightGlue loop-candidate export,
   confidence-weighted PnP diagnostics, KITTI benchmark runner updates, seq02
   and seq00 exploratory manifests, and loop-specific tests. Keep all new
   policies opt-in.

After each review pass, the focused validation block for that change set should
pass. Do not include `target/` logs or locally downloaded benchmark data in any
commit.

## Bisectable Commit Order

The review split above is not the same as a safe path-only staging order. Several
files intentionally cross change-set boundaries:

- `src/lib.rs` re-exports adaptive depth-gate, covisibility BA, and loop
  diagnostics APIs together.
- `tests/api_stability.rs` imports adaptive depth-gate and covisibility BA
  symbols, so it must not land before those implementations.
- `docs/interfaces.md`, `CHANGELOG.md`, and `README.md` describe multiple
  change sets in one public surface.
- `.github/workflows/ci.yml` calls registry and API-stability gates, so staging
  it before the registry helper and API-stability test would make that commit's
  CI fail.

If every intermediate commit must compile and pass its own focused checks, use a
coarser order:

1. **Benchmark registry foundation.**
   Land the registry schema, renderer, claim inputs, generated benchmark docs,
   capture helpers, and registry tests before CI depends on them.

2. **Algorithm/API implementations.**
   Land adaptive depth gate, covisibility local BA/keyframe policy, loop
   diagnostic APIs, root re-exports, and their Rust/Python tests. Keep new
   default-changing claims out of this commit.

3. **Public release hardening and CI.**
   Land README/docs claim cleanup, API stability allowlist, feature matrix,
   release checklist, asset cleanup, CI release gates, and portability fixes once
   the symbols and registry commands they reference already exist.

4. **Evidence manifests.**
   Land large batches of EuRoC/KITTI registry run manifests and generated
   evidence tables in a final evidence commit if review size matters. If the
   generated README/claim tables depend on these manifests, keep the matching
   generated files in the same commit.

Use partial staging only when a mixed file must be split across the order above.
Before tagging or presenting the whole branch as release-ready, run the full gate
below.

## Full Gate

Run before tagging or before presenting the whole branch as release-ready:

```sh
bash scripts/check.sh
```

The full gate is intentionally broader than any single change set. It covers
formatting, MSRV, feature matrix, clippy, Rust tests, Python tests, registry
validation, generated docs, release metadata, examples, trajectory evaluation,
demo output artifacts, rustdoc, and package checks.
