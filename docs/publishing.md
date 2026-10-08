# Publishing

`visloc-rs` is a workspace made of several crates. Publish workspace members in dependency order so crates that depend on internal `visloc-*` crates can resolve them from the target registry.

## Publish Order

`visloc-rs` depends on every crate below (the GPU crates as optional
dependencies, which crates.io still requires to exist), so all sixteen
workspace crates must be published. With cargo 1.90 or newer the simplest
route is to let cargo order them:

```sh
cargo publish --workspace
```

To publish crate by crate instead, use this order:

1. `visloc-core`
2. `visloc-dpvo-cuda-runtime`
3. `visloc-vision`
4. `visloc-localization`
5. `visloc-fusion`
6. `visloc-io`
7. `visloc-tracking`
8. `visloc-mapping`
9. `visloc-slam`
10. `visloc-basalt`
11. `visloc-gsplat-core`
12. `visloc-gsplat-render`
13. `visloc-sift-gpu`
14. `visloc-ba-gpu`
15. `visloc-gsplat-train`
16. `visloc-rs`

The order follows internal dependencies (optional ones marked `?`):

- `visloc-vision` depends on `visloc-core` and `visloc-dpvo-cuda-runtime?`.
- `visloc-localization` depends on `visloc-core` and `visloc-vision`.
- `visloc-fusion` depends on `visloc-core` and `visloc-localization`.
- `visloc-io` depends on `visloc-core`, `visloc-vision`, `visloc-localization`, and `visloc-fusion`.
- `visloc-tracking` depends on `visloc-core`, `visloc-vision`, and `visloc-localization`.
- `visloc-mapping` depends on `visloc-core`, `visloc-vision`, and `visloc-tracking`.
- `visloc-slam` depends on `visloc-core`, `visloc-vision`, `visloc-io`, `visloc-localization`, `visloc-tracking`, and `visloc-mapping`.
- `visloc-basalt` depends on `visloc-core`.
- `visloc-gsplat-core` depends on `visloc-core` and `visloc-io?`.
- `visloc-gsplat-render` depends on `visloc-gsplat-core`.
- `visloc-sift-gpu` depends on `visloc-vision` and `visloc-gsplat-render`.
- `visloc-ba-gpu` depends on `visloc-core`, `visloc-slam`, and `visloc-gsplat-render`.
- `visloc-gsplat-train` depends on `visloc-gsplat-core`, `visloc-gsplat-render`, and optionally `visloc-core`, `visloc-vision`, `visloc-io`, `visloc-slam`, `visloc-sift-gpu`, and `visloc-ba-gpu`.
- `visloc-rs` re-exports the workspace crates (`visloc-sift-gpu` and `visloc-ba-gpu` optionally).

## Local Checks

Run the normal quality gate before publishing:

```sh
scripts/check.sh
```

This runs formatting, clippy, tests, examples, release metadata checks, the GNSS demo output smoke check, docs, and package checks.

Each crate sets `package.metadata.docs.rs.all-features = true`, so docs.rs builds include optional public APIs such as image IO.

The root `visloc-rs` package uses an `include` allowlist (`src/`, `examples/`, `configs/`, README, licenses). Benchmark evidence, docs media, scripts, and work logs stay in the GitHub repository only; without the allowlist the root package would be about 80 MB, far over the crates.io limit. Packaged with the allowlist, the largest crate (`visloc-slam`) is about 1.5 MB compressed.

When only package metadata changed, this narrower check is useful:

```sh
scripts/package_check.sh
```

By default, `scripts/package_check.sh` verifies docs.rs metadata, lists package contents for all crates, and packages the first independently publishable crate, `visloc-core`.

To package and verify-build every crate before anything is published, run:

```sh
VISLOC_PACKAGE_ALL=1 scripts/package_check.sh
```

This uses `cargo package --workspace` (cargo 1.90 or newer), which resolves
internal `visloc-*` dependencies from the freshly packaged local crates, so it
works before the first crates.io release.

## Release Steps

1. Confirm `CHANGELOG.md` is up to date.
2. Confirm `README.md`, `docs/interfaces.md`, `docs/api_stability.md`, `docs/colmap_compatibility.md`, and `docs/migration.md` describe the current public API.
3. Run `scripts/check.sh`.
4. Tag the release only after the main branch CI passes.
5. Publish with `cargo publish --workspace`, or crate by crate in the order listed above.
6. When publishing crate by crate, wait until each internal crate is available from the registry before publishing the next dependent crate.

## Notes

- Do not publish `visloc-rs` before all internal crates it re-exports are available from crates.io.
- Keep version numbers aligned across workspace crates unless there is a deliberate release-management reason to diverge later.
- The default package check uses `--no-verify` to stay fast. `VISLOC_PACKAGE_ALL=1` verify-builds every crate from its packaged sources.
