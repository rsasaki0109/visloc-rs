#!/usr/bin/env bash
# Thin reproducible entry point for the explicit official COLMAP control.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$script_dir/benchmark_electro_colmap.py" "$@"
