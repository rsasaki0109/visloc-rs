# Release Checklist

`visloc-rs` is still pre-1.0. A release should preserve the current goal: a small working visual-localization vertical slice, without pretending to be a full SLAM system.

## Before Tagging

- Run `scripts/check.sh`.
- Run every example that is expected to stay user-facing.
- Confirm `README.md` describes the current public API and does not imply full SLAM support.
- Confirm `docs/interfaces.md` matches the exported traits and structs.
- Confirm `docs/api_stability.md` reflects the intended stable and experimental API tiers.
- Confirm `docs/decisions.md` explains any new module boundary or algorithm choice.
- Update `CHANGELOG.md`.
- Check package metadata with `cargo package -p visloc-core --allow-dirty --no-verify` first.
- For crates that depend on other `visloc-*` crates, run `cargo package` in publish order after their internal dependencies are available from the target registry.

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
