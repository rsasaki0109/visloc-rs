# Publishing

`visloc-rs` is a workspace made of several crates. Publish workspace members in dependency order so crates that depend on internal `visloc-*` crates can resolve them from the target registry.

## Publish Order

`visloc-rs` depends on every crate below (the GPU crates as optional
dependencies, which crates.io still requires to exist), so all seventeen
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
11. `visloc-objects`
12. `visloc-gsplat-core`
13. `visloc-gsplat-render`
14. `visloc-sift-gpu`
15. `visloc-ba-gpu`
16. `visloc-mvs`
17. `visloc-gsplat-train`
18. `visloc-rs`

The order follows internal dependencies (optional ones marked `?`):

- `visloc-vision` depends on `visloc-core` and `visloc-dpvo-cuda-runtime?`.
- `visloc-localization` depends on `visloc-core` and `visloc-vision`.
- `visloc-fusion` depends on `visloc-core` and `visloc-localization`.
- `visloc-io` depends on `visloc-core`, `visloc-vision`, `visloc-localization`, and `visloc-fusion`.
- `visloc-tracking` depends on `visloc-core`, `visloc-vision`, and `visloc-localization`.
- `visloc-mapping` depends on `visloc-core`, `visloc-vision`, and `visloc-tracking`.
- `visloc-slam` depends on `visloc-core`, `visloc-vision`, `visloc-io`, `visloc-fusion`, `visloc-localization`, `visloc-tracking`, and `visloc-mapping`.
- `visloc-basalt` depends on `visloc-core`.
- `visloc-objects` has no internal dependencies.
- `visloc-gsplat-core` depends on `visloc-core` and `visloc-io?`.
- `visloc-gsplat-render` depends on `visloc-gsplat-core`.
- `visloc-sift-gpu` depends on `visloc-vision` and `visloc-gsplat-render`.
- `visloc-ba-gpu` depends on `visloc-core`, `visloc-slam`, and `visloc-gsplat-render`.
- `visloc-mvs` depends on `visloc-gsplat-render?`.
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

1. Bump `version` in the root `Cargo.toml` (`[package]`, `[workspace.package]`
   and every internal `visloc-*` dependency requirement) and in
   `bindings/python/Cargo.toml`; `scripts/check_release_metadata.sh` fails
   while they disagree.
2. Confirm `CHANGELOG.md` is up to date, with a `## <version> - <date>` section.
3. Confirm `README.md`, `docs/interfaces.md`, `docs/api_stability.md`, `docs/colmap_compatibility.md`, and `docs/migration.md` describe the current public API.
4. Run `scripts/check.sh`.
5. Tag the release only after the main branch CI passes, and push the tag
   (`git tag v<version> && git push origin v<version>`). The `release`
   workflow does the rest (next section).

## Automated Release

A `v*` tag push runs `.github/workflows/release.yml`:

| Job | What it does |
| --- | --- |
| `release` | Creates the GitHub release from the tag's `CHANGELOG.md` section. |
| `binaries` | Attaches prebuilt Linux, Windows and macOS command-line binaries. |
| `crates` | Checks that the tag matches the workspace version, verify-builds every crate (`cargo package --workspace`), then uploads them in publish order with `scripts/publish_crates.sh`. |
| `wheels`, `sdist` | Builds the `visloc` Python wheels (CPython 3.10–3.14 on Linux x86_64 / aarch64 manylinux, Windows x64 and macOS arm64) and the sdist with maturin, and smoke-tests a wheel with the bindings' pytest suite. |
| `pypi` | Uploads the wheels and the sdist to PyPI and attaches them to the GitHub release. |

`scripts/publish_crates.sh` skips every crate whose version is already on
crates.io, so a failed or cancelled run is resumed by running the workflow by
hand (Actions → release → Run workflow, with the tag and **publish** ticked).
`scripts/publish_crates.sh --dry-run` lists what it would upload.

**The first crates.io release takes about two hours.** crates.io lets an
account create only a few new crates in a burst and then one every few
minutes; the script waits for the time crates.io names after each HTTP 429.
Later releases only update existing crates and finish in minutes.

### One-time setup

Until this is done the `crates` job skips itself with a notice, and the
`pypi` job fails, so the GitHub release and binaries are unaffected.

**crates.io.**

1. Create an API token at <https://crates.io/settings/tokens> with the
   `publish-new` and `publish-update` scopes. (crates.io's trusted
   publishing cannot create a crate, so the first release needs a token.)
2. Add it to the repository as the Actions secret `CARGO_REGISTRY_TOKEN`
   (Settings → Secrets and variables → Actions).
3. Optional, after the first release: configure trusted publishing on each
   crate's crates.io settings page (repository `rsasaki0109/visloc-rs`,
   workflow `release.yml`), set the repository variable
   `CRATES_IO_TRUSTED_PUBLISHING` to `true`, and delete the
   `CARGO_REGISTRY_TOKEN` secret. The workflow then exchanges a short-lived
   token over OIDC (`rust-lang/crates-io-auth-action`) instead.

**PyPI.** No token is stored; the `pypi` job uses trusted publishing.

1. At <https://pypi.org/manage/account/publishing/> add a pending publisher:
   project `visloc`, owner `rsasaki0109`, repository `visloc-rs`, workflow
   `release.yml`, environment `pypi`.
2. Optional: in the repository's Settings → Environments, give the `pypi`
   environment required reviewers so every upload waits for an approval.

### Publishing by hand

`cargo publish --workspace` (cargo 1.90 or newer) publishes everything in
dependency order, or publish crate by crate in the order listed above,
waiting until each internal crate is available from the registry before
publishing the next dependent crate. For the wheels, see
`bindings/python/README.md` (`maturin build --release`).

## Notes

- Do not publish `visloc-rs` before all internal crates it re-exports are available from crates.io.
- Keep version numbers aligned across workspace crates unless there is a deliberate release-management reason to diverge later.
- The default package check uses `--no-verify` to stay fast. `VISLOC_PACKAGE_ALL=1` verify-builds every crate from its packaged sources.
