# Release Checklist

`visloc-rs` is still pre-1.0. A release should preserve the current goal: a small working visual-localization vertical slice, without pretending to be a full SLAM system.

## Before Tagging

- Run `scripts/check.sh`, including the Rust 1.82 MSRV `image-io` check, Tier 1 feature matrix, Python registry/CI drift tests, benchmark-registry validation, package metadata, crate-content checks, documentation link checks, GNSS demo output smoke checks, timestamped image GNSS sync-output checks, and KITTI image sequence demo output checks.
- Use `docs/release_change_sets.md` to review the branch by change-set scope instead of treating the full diff as one release change.
- Confirm `scripts/check_feature_matrix.sh` passes for Tier 1 features (`--no-default-features`, default, and `image-io`) and that CI runs those checks on Linux and Windows.
- Confirm `python -m unittest tests.test_ci_release_gate tests.test_feature_matrix tests.test_docs_assets` passes after changing CI, feature support, release gates, generated benchmark evidence paths, or docs showcase assets.
- Confirm CI uploads the public demo artifacts: `gnss-demo-outputs`, `timestamped-gnss-image-demo-outputs`, and `kitti-image-sequence-demo-outputs`.
- Confirm trajectory evaluation thresholds still pass through `scripts/check_trajectory_evaluation.sh`.
- Confirm the CI MSRV job passes with Rust 1.82.0 through `scripts/check_msrv.sh`.
- Run every example that is expected to stay user-facing with `scripts/run_examples.sh` (also covered by `scripts/check.sh`).
- Confirm `README.md` describes the current public API and does not imply full SLAM support.
- Confirm README benchmark rows are generated from `benchmarks/registry/readme_claims_v1.json` via `scripts/benchmark_registry.py render-readme`, and that any headline metric change has a registered run manifest or an explicit `documented_historical` / `external_published` evidence label.
- Regenerate `docs/generated/benchmark_snapshot.md` with `scripts/benchmark_registry.py render-readme --claims benchmarks/registry/readme_claims_v1.json --out docs/generated/benchmark_snapshot.md --with-heading` so the standalone headline snapshot explains that exploratory and negative run evidence lives elsewhere.
- Regenerate `docs/generated/registered_runs.md` with `scripts/benchmark_registry.py render-runs --registry-dir benchmarks/registry/runs --with-heading --out docs/generated/registered_runs.md` so supporting, exploratory, and negative run evidence stays visible outside the README headline table.
- Regenerate `docs/generated/benchmark_claim_matrix.md` with `scripts/benchmark_registry.py render-claim-matrix --matrix benchmarks/registry/claim_matrix_v1.json --with-heading --out docs/generated/benchmark_claim_matrix.md` so ORB-SLAM / COLMAP / VINS comparisons stay scoped by sequence, sensor mode, metric, and verdict.
- Confirm `scripts/benchmark_registry.py check-generated` passes locally and in CI, proving `README.md`, `docs/generated/benchmark_snapshot.md`, `docs/generated/registered_runs.md`, and `docs/generated/benchmark_claim_matrix.md` match the registry inputs.
- Confirm any benchmark number produced with opt-in diagnostic flags (for example `--loop-matches-dir`, `--loop-pnp-essential-inliers`, `--loop-pnp-confidence-weights`, or the covisibility BA boundary-support gate flags) is separated from the default claim matrix, labelled `exploratory` or per-sequence opt-in, and paired with at least one retained non-regression or negative artifact before it is mentioned publicly.
- Confirm failed or DNF benchmark runs that informed a claim are kept in `benchmarks/registry/runs` with `status` and `failure_reason`, not deleted from the record.
- Confirm local README/docs links and anchors pass with `scripts/check_docs_links.sh`.
- Confirm release metadata consistency passes with `scripts/check_release_metadata.sh`.
- Confirm docs.rs metadata is present for every publishable crate so optional APIs are documented with all features enabled.
- Confirm `docs/interfaces.md` matches the exported traits and structs.
- Confirm `docs/api_stability.md` reflects the intended stable and experimental API tiers.
- Confirm `cargo test --test api_stability` passes so the documented canonical paths and stable-intent allowlist still compile.
- Confirm `docs/feature_matrix.md` matches root Cargo features, crate feature pass-throughs, and CI coverage.
- Confirm `docs/colmap_compatibility.md` matches current COLMAP reader/writer behavior.
- Confirm `docs/migration.md` records any known pre-1.0 API migrations.
- Confirm `docs/decisions.md` explains any new module boundary or algorithm choice.
- Update `CHANGELOG.md`.
- Confirm `docs/publishing.md` matches workspace dependencies and publish order.
- Check package metadata, docs.rs metadata, and crate contents with `scripts/package_check.sh` when package details are changed independently of the full local gate.
- For crates that depend on other `visloc-*` crates, run `VISLOC_PACKAGE_ALL=1 scripts/package_check.sh` in publish order after their internal dependencies are available from the target registry.

## API Stability

- Keep `core` types conservative and broadly useful.
- Keep `vision` algorithms reusable outside the localization pipeline.
- Keep pipeline APIs trait-based so feature extraction, matching, and pose estimation remain replaceable.
- Prefer additive API changes while the library is still early.
- Avoid adding SLAM-specific state to `core` unless localization also benefits from it.

## Quality Bar

- New public behavior should have an integration test or a focused unit test.
- New parsers should include a minimal fixture and at least one malformed-input case.
- New estimators should report enough diagnostics to explain failure.
- Examples should use deterministic synthetic data unless they are explicitly IO examples.
