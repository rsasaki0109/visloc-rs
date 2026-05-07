# Publishing

`visloc-rs` is a workspace made of several crates. Publish workspace members in dependency order so crates that depend on internal `visloc-*` crates can resolve them from the target registry.

## Publish Order

Use this order for crates.io releases:

1. `visloc-core`
2. `visloc-vision`
3. `visloc-localization`
4. `visloc-io`
5. `visloc-tracking`
6. `visloc-mapping`
7. `visloc-slam`
8. `visloc-fusion`
9. `visloc-rs`

The order follows internal dependencies:

- `visloc-vision` depends on `visloc-core`.
- `visloc-localization` depends on `visloc-core` and `visloc-vision`.
- `visloc-io` depends on `visloc-core`, `visloc-vision`, and `visloc-localization`.
- `visloc-tracking` depends on `visloc-core`, `visloc-vision`, and `visloc-localization`.
- `visloc-mapping` depends on `visloc-core` and `visloc-tracking`.
- `visloc-slam` depends on `visloc-core`, `visloc-localization`, `visloc-tracking`, and `visloc-mapping`.
- `visloc-fusion` depends on `visloc-core` and `visloc-localization`.
- `visloc-rs` re-exports all workspace crates.

## Local Checks

Run the normal quality gate before publishing:

```sh
scripts/check.sh
```

This runs formatting, clippy, tests, examples, docs, and package checks.

When only package metadata changed, this narrower check is useful:

```sh
scripts/package_check.sh
```

By default, `scripts/package_check.sh` lists package contents for all crates and packages the first independently publishable crate, `visloc-core`.

After internal dependencies are published to the target registry, run:

```sh
VISLOC_PACKAGE_ALL=1 scripts/package_check.sh
```

This verifies that every crate can be packaged in publish order.

## Release Steps

1. Confirm `CHANGELOG.md` is up to date.
2. Confirm `README.md`, `docs/interfaces.md`, `docs/api_stability.md`, `docs/colmap_compatibility.md`, and `docs/migration.md` describe the current public API.
3. Run `scripts/check.sh`.
4. Tag the release only after the main branch CI passes.
5. Publish crates in the order listed above.
6. After each internal crate is available from the registry, continue to the next dependent crate.

## Notes

- Do not publish `visloc-rs` before all internal crates it re-exports are available from crates.io.
- Keep version numbers aligned across workspace crates unless there is a deliberate release-management reason to diverge later.
- The current package check uses `--no-verify` to avoid requiring already-published internal dependencies during early workspace packaging. Use `VISLOC_PACKAGE_ALL=1` after dependencies are available from the registry.
