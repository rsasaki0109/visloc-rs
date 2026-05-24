#!/usr/bin/env sh
set -eu

# The 1.82 MSRV guarantee covers the core library and the `image-io` demo path.
#
# It deliberately does NOT cover the opt-in `onnx-inference` feature: that pulls
# the `ort` ONNX runtime, whose build dependency `ureq`/`ureq-proto` requires
# edition2024 (Rust >= 1.85) and cannot even be parsed by Cargo 1.82. The ONNX
# runtime tracks its own toolchain floor; keeping it out of this check confines
# that requirement to the feature boundary instead of leaking it into the core
# MSRV. Build the ONNX path on a current stable toolchain instead.
cargo +1.82.0 check --workspace --all-targets --features image-io
