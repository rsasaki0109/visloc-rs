# Contributing

`visloc-rs` is a localization-first Rust project. Contributions should keep the core useful for map-based visual localization while leaving room for tracking, local mapping, online SLAM, and sensor-fusion extensions.

## Good First Contributions

- Improve diagnostics for localization, tracking, parsers, and demos.
- Add focused tests for map validation, matching, PnP, RANSAC, trajectory export, or prior handling.
- Improve COLMAP/SfM compatibility notes with reproducible fixtures.
- Add small deterministic examples that exercise one workflow end to end.
- Improve public-data demo documentation without adding hidden assets.

## Design Expectations

- Keep geometry, matching, PnP, RANSAC, and IO reusable and mostly stateless.
- Keep full SLAM, loop closure, dense mapping, and production-grade fusion out of early core APIs.
- Prefer trait boundaries for replaceable feature extraction, matching, pose estimation, map providers, tracking, mapping, and fusion inputs.
- Add complexity only when a runnable example or test needs it.
- Use `nalgebra` for geometry and avoid `unsafe`.

## Local Checks

Run the full local gate before opening a PR:

```bash
scripts/check.sh
```

For a narrower iteration loop, run:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
scripts/run_examples.sh
sh scripts/check_gnss_demo_outputs.sh
```

## Documentation

When adding public behavior, update the matching docs:

- `README.md` for common user workflows.
- `docs/interfaces.md` for public types and traits.
- `docs/roadmap.md` for stage-level planning changes.
- `docs/colmap_compatibility.md` for COLMAP/SfM IO behavior.
- `docs/gnss_demo.md` for GNSS-prior sequence demo output changes.
- `CHANGELOG.md` under `Unreleased`.
