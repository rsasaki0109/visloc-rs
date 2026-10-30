#!/usr/bin/env bash
# Build the optional CUDA PatchMatch MVS (native/mvs_cuda) into the shared
# library that crates/mvs (feature `cuda`) loads at runtime. Cargo never
# compiles it, so ordinary builds need no CUDA toolkit.
#
#   scripts/build_mvs_cuda.sh OUT_DIR [ARCH]
#
# ARCH defaults to `native` (the GPU of this machine); pass e.g. sm_89 to
# build for another one. NVCC overrides the compiler (default: nvcc on PATH,
# then /usr/local/cuda/bin/nvcc).
set -euo pipefail
out=${1:?usage: build_mvs_cuda.sh OUT_DIR [ARCH]}
arch=${2:-native}
nvcc=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
repo=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$out"
lib="$out/libvisloc_mvs_cuda.so"
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -arch="$arch" -lineinfo \
    -o "$lib" "$repo/native/mvs_cuda/mvs_cuda.cu"
sha256sum "$lib"
