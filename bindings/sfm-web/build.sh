#!/usr/bin/env bash
# Builds the browser SfM into web/pkg: the release wasm for
# wasm32-unknown-unknown with SIMD (as the browser VIO), its wasm-bindgen glue
# (ES module), then wasm-opt if it is on PATH; and copies the browser VIO's
# three.js bundle into web/vendor for the 3D view.
#
# Needs: the wasm32-unknown-unknown target and wasm-bindgen-cli matching
# Cargo.lock's wasm-bindgen (`cargo install wasm-bindgen-cli --version <v>`).
set -euo pipefail
cd "$(dirname "$0")"
TARGET=wasm32-unknown-unknown
RUSTFLAGS="-C target-feature=+simd128" cargo build --release --lib --target "$TARGET"
wasm-bindgen --target web --no-typescript --out-dir web/pkg \
  "target/$TARGET/release/visloc_sfm_web.wasm"
if command -v wasm-opt > /dev/null; then
  wasm-opt -O3 --enable-simd --enable-bulk-memory --enable-nontrapping-float-to-int \
    -o web/pkg/visloc_sfm_web_bg.wasm web/pkg/visloc_sfm_web_bg.wasm
fi
mkdir -p web/vendor
cp ../wasm/web/vendor/three.js ../wasm/web/vendor/three.LICENSE web/vendor/
ls -l web/pkg
