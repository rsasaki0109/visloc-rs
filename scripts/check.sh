#!/usr/bin/env sh
set -eu

if ! command -v cargo >/dev/null 2>&1 && [ -d "$HOME/.cargo/bin" ]; then
    PATH="$HOME/.cargo/bin:$PATH"
    export PATH
fi
if [ -z "${PYTHON:-}" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON=python3
    elif command -v python >/dev/null 2>&1; then
        PYTHON=python
    else
        echo "python3 or python is required" >&2
        exit 1
    fi
fi

cargo fmt --all -- --check
sh scripts/check_msrv.sh
sh scripts/check_feature_matrix.sh
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
"$PYTHON" -m unittest discover -s tests -p 'test_*.py'
"$PYTHON" scripts/benchmark_registry.py validate benchmarks/registry/readme_claims_v1.json benchmarks/registry/claim_matrix_v1.json benchmarks/registry/runs
"$PYTHON" scripts/benchmark_registry.py check-generated --readme docs/readme_details.md
sh scripts/check_docs_links.sh
sh scripts/check_release_metadata.sh
scripts/run_examples.sh
sh scripts/check_trajectory_evaluation.sh
sh scripts/check_gnss_demo_outputs.sh
sh scripts/check_timestamped_gnss_image_demo_outputs.sh
sh scripts/check_kitti_image_sequence_demo_outputs.sh
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
scripts/package_check.sh
