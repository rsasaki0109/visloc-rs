#!/usr/bin/env sh
# One-command entry point for the externally stored courtyard control.
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$script_dir/benchmark_courtyard.py" "$@"
