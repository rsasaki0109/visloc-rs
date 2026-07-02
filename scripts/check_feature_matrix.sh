#!/usr/bin/env sh
set -eu

if ! command -v cargo >/dev/null 2>&1 && [ -d "$HOME/.cargo/bin" ]; then
    PATH="$HOME/.cargo/bin:$PATH"
    export PATH
fi

# Tier-1 feature checks. These must stay free of mandatory native runtimes and
# should pass on Linux and Windows CI.
cargo check --workspace --all-targets --no-default-features
cargo check --workspace --all-targets
cargo check --workspace --all-targets --features image-io

# Tier-2 opt-in checks. Enable locally or in a dedicated CI job when validating
# ONNX runtime changes; they may download native ONNX Runtime binaries.
if [ "${VISLOC_CHECK_ONNX:-0}" = "1" ]; then
    cargo check --workspace --all-targets --features image-io,onnx-inference
fi

# Hardware-gated CUDA path. This intentionally stays out of the default gate.
if [ "${VISLOC_CHECK_ONNX_CUDA:-0}" = "1" ]; then
    cargo check --workspace --all-targets --features image-io,onnx-cuda
fi
