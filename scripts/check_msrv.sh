#!/usr/bin/env sh
set -eu

cargo +1.82.0 check --workspace --all-targets --all-features
