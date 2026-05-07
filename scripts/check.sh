#!/usr/bin/env sh
set -eu

cargo fmt --all -- --check
sh scripts/check_msrv.sh
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
sh scripts/check_docs_links.sh
sh scripts/check_release_metadata.sh
scripts/run_examples.sh
sh scripts/check_trajectory_evaluation.sh
sh scripts/check_gnss_demo_outputs.sh
sh scripts/check_timestamped_gnss_image_demo_outputs.sh
sh scripts/check_kitti_image_sequence_demo_outputs.sh
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
scripts/package_check.sh
