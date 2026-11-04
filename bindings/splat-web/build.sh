#!/usr/bin/env bash
# Builds the browser splat viewer into web/pkg: the release wasm for
# wasm32-unknown-unknown and its wasm-bindgen glue (ES module), then
# wasm-opt if it is on PATH.
#
# Needs: the wasm32-unknown-unknown target and wasm-bindgen-cli matching
# Cargo.lock's wasm-bindgen (`cargo install wasm-bindgen-cli --version <v>`).
set -euo pipefail
cd "$(dirname "$0")"
TARGET=wasm32-unknown-unknown
cargo build --release --lib --target "$TARGET"
wasm-bindgen --target web --no-typescript --out-dir web/pkg \
  "target/$TARGET/release/visloc_splat_web.wasm"
if command -v wasm-opt > /dev/null; then
  wasm-opt -O3 --enable-bulk-memory --enable-nontrapping-float-to-int \
    -o web/pkg/visloc_splat_web_bg.wasm web/pkg/visloc_splat_web_bg.wasm
fi
ls -l web/pkg
