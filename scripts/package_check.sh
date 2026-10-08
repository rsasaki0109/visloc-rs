#!/usr/bin/env sh
set -eu

if ! command -v cargo >/dev/null 2>&1 && [ -d "$HOME/.cargo/bin" ]; then
    PATH="$HOME/.cargo/bin:$PATH"
    export PATH
fi

grep_normalized() {
    pattern="$1"
    file="$2"
    tr -d '\r' < "$file" | grep -q "$pattern"
}

# Publish order (see docs/publishing.md).
packages="
visloc-core
visloc-dpvo-cuda-runtime
visloc-vision
visloc-localization
visloc-fusion
visloc-io
visloc-tracking
visloc-mapping
visloc-slam
visloc-basalt
visloc-gsplat-core
visloc-gsplat-render
visloc-sift-gpu
visloc-ba-gpu
visloc-gsplat-train
visloc-rs
"

manifests="
Cargo.toml
crates/core/Cargo.toml
crates/dpvo-cuda-runtime/Cargo.toml
crates/vision/Cargo.toml
crates/io/Cargo.toml
crates/gsplat-core/Cargo.toml
crates/gsplat-render/Cargo.toml
crates/gsplat-train/Cargo.toml
crates/sift-gpu/Cargo.toml
crates/ba-gpu/Cargo.toml
pipelines/localization/Cargo.toml
pipelines/tracking/Cargo.toml
pipelines/mapping/Cargo.toml
pipelines/slam/Cargo.toml
pipelines/fusion/Cargo.toml
pipelines/basalt/Cargo.toml
"

for manifest in $manifests; do
    echo "Checking docs.rs metadata: $manifest"
    grep_normalized '^\[package.metadata.docs.rs\]$' "$manifest"
    grep_normalized '^all-features = true$' "$manifest"
done

for package in $packages; do
    echo "Listing package contents: $package"
    cargo package -p "$package" --allow-dirty --no-verify --list >/dev/null
done

if [ "${VISLOC_PACKAGE_ALL:-0}" = "1" ]; then
    # `cargo package --workspace` (cargo 1.90+) resolves internal visloc-*
    # dependencies from the freshly packaged local crates, so this works
    # before anything is published.
    echo "Packaging and verify-building every crate"
    cargo package --workspace --allow-dirty
else
    echo "Packaging first publishable crate: visloc-core"
    cargo package -p visloc-core --allow-dirty --no-verify
    echo "Set VISLOC_PACKAGE_ALL=1 to package and verify-build every crate."
fi
