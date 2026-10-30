#!/usr/bin/env bash
# Builds the demo's two WebAssembly variants into web/pkg (single-threaded,
# the repository's pinned toolchain) and web/pkg-mt (rayon on Web Workers:
# nightly with atomics and a rebuilt std, used when the page is cross-origin
# isolated). Both enable wasm SIMD (Safari 16.4+, Chrome 91+).
#
# Needs: rustup with the wasm32-unknown-unknown target on the pinned
# toolchain and on $NIGHTLY (with rust-src), wasm-bindgen-cli matching
# Cargo.lock's wasm-bindgen, and optionally wasm-opt (binaryen) on PATH or
# in $WASM_OPT.
set -euo pipefail
cd "$(dirname "$0")"

NIGHTLY="${NIGHTLY:-nightly-2026-04-10}"
WASM_OPT="${WASM_OPT:-$(command -v wasm-opt || true)}"
TARGET=wasm32-unknown-unknown

# optimize <file.wasm> [extra wasm-opt flags]
optimize() {
  local file="$1"
  shift
  if [[ -n "$WASM_OPT" ]]; then
    "$WASM_OPT" -O3 --enable-simd --enable-bulk-memory --enable-nontrapping-float-to-int \
      --enable-sign-ext --enable-mutable-globals "$@" -o "$file.opt" "$file"
    mv "$file.opt" "$file"
  fi
}

echo "== single-threaded"
RUSTFLAGS="-C target-feature=+simd128" \
  cargo build --release --lib --target "$TARGET"
wasm-bindgen --target web --no-typescript --out-dir web/pkg \
  "target/$TARGET/release/visloc_wasm.wasm"
optimize web/pkg/visloc_wasm_bg.wasm

echo "== threaded ($NIGHTLY)"
# Shared, imported memory and the TLS exports wasm-bindgen's thread support
# needs (rustc no longer passes these for +atomics). 1 GiB maximum: iOS
# Safari fails to reserve 4 GiB shared memories.
MT_LINK="-C link-arg=--shared-memory -C link-arg=--import-memory -C link-arg=--max-memory=1073741824"
MT_LINK="$MT_LINK -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size"
MT_LINK="$MT_LINK -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base"
RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128 $MT_LINK" \
  CARGO_TARGET_DIR=target/mt \
  cargo "+$NIGHTLY" build --release --lib --target "$TARGET" --features threads \
  -Z build-std=panic_abort,std
wasm-bindgen --target web --no-typescript --out-dir web/pkg-mt \
  "target/mt/$TARGET/release/visloc_wasm.wasm"
optimize web/pkg-mt/visloc_wasm_bg.wasm --enable-threads
# wasm-bindgen-rayon's pool workers import the package as '../../..', which
# only a bundler resolves; served as plain files it must name the module.
sed -i "s#import('../../..')#import('../../../visloc_wasm.js')#" web/pkg-mt/snippets/wasm-bindgen-rayon-*/src/workerHelpers.js

ls -la web/pkg/*.wasm web/pkg-mt/*.wasm
