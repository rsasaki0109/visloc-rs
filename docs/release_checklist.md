# Release Checklist

`visloc-rs` is still pre-1.0. A release should preserve the current goal: a small working visual-localization vertical slice, without pretending to be a full SLAM system.

## Before Tagging

- Run `scripts/check.sh`, including MSRV all-features checks, package metadata, crate-content checks, documentation link checks, GNSS demo output smoke checks, timestamped image GNSS sync-output checks, and KITTI image sequence demo output checks.
- Confirm CI uploads the public demo artifacts: `gnss-demo-outputs`, `timestamped-gnss-image-demo-outputs`, and `kitti-image-sequence-demo-outputs`.
- Confirm trajectory evaluation thresholds still pass through `scripts/check_trajectory_evaluation.sh`.
- Confirm the CI MSRV job passes with Rust 1.82.0 through `scripts/check_msrv.sh`.
- Run every example that is expected to stay user-facing with `scripts/run_examples.sh` (also covered by `scripts/check.sh`).
- Confirm `README.md` describes the current public API and does not imply full SLAM support.
- Confirm local README/docs links and anchors pass with `scripts/check_docs_links.sh`.
- Confirm release metadata consistency passes with `scripts/check_release_metadata.sh`.
- Confirm docs.rs metadata is present for every publishable crate so optional APIs are documented with all features enabled.
- Confirm `docs/interfaces.md` matches the exported traits and structs.
- Confirm `docs/api_stability.md` reflects the intended stable and experimental API tiers.
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
