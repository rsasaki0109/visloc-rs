#!/usr/bin/env sh
set -eu

cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
scripts/run_examples.sh
sh scripts/check_gnss_demo_outputs.sh
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
scripts/package_check.sh
