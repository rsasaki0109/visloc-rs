#!/usr/bin/env sh
set -eu

grep_normalized() {
    pattern="$1"
    file="$2"
    tr -d '\r' < "$file" | grep -q "$pattern"
}

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

echo "Checking workspace release metadata"
grep_normalized '^version = "0.2.1"$' Cargo.toml
grep_normalized '^rust-version = "1.88"$' Cargo.toml
grep_normalized '^license = "MIT OR Apache-2.0"$' Cargo.toml
grep_normalized '^repository = "https://github.com/rsasaki0109/visloc-rs"$' Cargo.toml

for manifest in $manifests; do
    echo "Checking manifest metadata: $manifest"
    grep_normalized '^\[package.metadata.docs.rs\]$' "$manifest"
    grep_normalized '^all-features = true$' "$manifest"
done

for artifact in \
    gnss-demo-outputs \
    timestamped-gnss-image-demo-outputs \
    kitti-image-sequence-demo-outputs
do
    echo "Checking CI artifact documentation: $artifact"
    grep_normalized "name: $artifact" .github/workflows/ci.yml
    grep_normalized "$artifact" docs/release_checklist.md
    grep_normalized "$artifact" docs/experiments.md
done

for package in $packages; do
    echo "Checking publish/package references: $package"
    grep_normalized "$package" docs/publishing.md
    grep_normalized "$package" scripts/package_check.sh
done

echo "Release metadata checks passed"
