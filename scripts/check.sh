#!/usr/bin/env sh
set -eu

cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
scripts/run_examples.sh
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
