#!/usr/bin/env bash
# Build the optional native CUDA 3DGS rasterizer (native/gsplat_cuda) into the
# shared library that crates/gsplat-cuda loads at runtime. Cargo never
# compiles it, so ordinary builds need no CUDA toolkit.
#
#   scripts/build_gsplat_cuda.sh OUT_DIR [ARCH]
#
# ARCH defaults to `native` (the GPU of this machine); pass e.g. sm_89 to
# build for another one. NVCC overrides the compiler (default: nvcc on PATH,
# then /usr/local/cuda/bin/nvcc).
set -euo pipefail
out=${1:?usage: build_gsplat_cuda.sh OUT_DIR [ARCH]}
arch=${2:-native}
nvcc=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
repo=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$out"
lib="$out/libvisloc_gsplat_cuda.so"
"$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC -arch="$arch" -lineinfo \
    -o "$lib" "$repo/native/gsplat_cuda/gsplat_cuda.cu"
sha256sum "$lib"
