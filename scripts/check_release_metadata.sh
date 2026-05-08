#!/usr/bin/env sh
set -eu

packages="
visloc-core
visloc-vision
visloc-localization
visloc-io
visloc-tracking
visloc-mapping
visloc-slam
visloc-fusion
visloc-rs
"

manifests="
Cargo.toml
crates/core/Cargo.toml
crates/vision/Cargo.toml
crates/io/Cargo.toml
pipelines/localization/Cargo.toml
pipelines/tracking/Cargo.toml
pipelines/mapping/Cargo.toml
pipelines/slam/Cargo.toml
pipelines/fusion/Cargo.toml
"

echo "Checking workspace release metadata"
grep -q '^version = "0.1.0"$' Cargo.toml
grep -q '^rust-version = "1.82"$' Cargo.toml
grep -q '^license = "MIT OR Apache-2.0"$' Cargo.toml
grep -q '^repository = "https://github.com/rsasaki0109/visloc-rs"$' Cargo.toml

for manifest in $manifests; do
    echo "Checking manifest metadata: $manifest"
    grep -q '^\[package.metadata.docs.rs\]$' "$manifest"
    grep -q '^all-features = true$' "$manifest"
done

for artifact in \
    gnss-demo-outputs \
    timestamped-gnss-image-demo-outputs \
    kitti-image-sequence-demo-outputs
do
    echo "Checking CI artifact documentation: $artifact"
    grep -q "name: $artifact" .github/workflows/ci.yml
    grep -q "$artifact" README.md
    grep -q "$artifact" docs/release_checklist.md
    grep -q "$artifact" docs/experiments.md
done

for package in $packages; do
    echo "Checking publish/package references: $package"
    grep -q "$package" docs/publishing.md
    grep -q "$package" scripts/package_check.sh
done

echo "Release metadata checks passed"
