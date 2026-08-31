#!/usr/bin/env sh
# Thin shell entry point for the dependency-free electro benchmark runner.
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$script_dir/benchmark_electro.py" "$@"
