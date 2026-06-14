#!/usr/bin/env bash
# In-process SuperPoint ONNX throughput benchmark (CPU vs CUDA).
#
# Builds the `superpoint_onnx_throughput` example with the CUDA execution
# provider, wires up the two runtime dependencies the CUDA path needs, and runs
# the CPU-vs-CUDA comparison. This measures the in-Rust deep front-end's
# latency / FPS — the basis of the "real-time deep-frontend stereo SLAM, pure
# Rust, no Python feature-export step" claim.
#
# Two runtime wrinkles the CUDA execution provider has, both handled here:
#   1. ONNX Runtime's provider bridge dlopen()s `libonnxruntime_providers_shared.so`
#      (and `..._cuda.so`) from the *executable's directory*. ort downloads
#      them into ~/.cache/ort.pyke.io/...; we symlink them next to the binary.
#   2. The CUDA provider needs libcudnn.so.9 (and cuBLAS/cuFFT) at run time.
#      We add the pip `nvidia-*` lib dirs to LD_LIBRARY_PATH; cuBLAS/cuFFT are
#      usually also on the system path.
#
#   scripts/run_superpoint_onnx_throughput.sh \
#       --model models/superpoint_1500.onnx \
#       --images-dir /tmp/MH_03_rect/image_0 --frames 300 --backend both
#
# Generate the model first with:
#   scripts/export_superpoint_onnx.py --out models/superpoint_1500.onnx
set -eu

model="models/superpoint_1500.onnx"
images_dir="/tmp/MH_03_rect/image_0"
frames=300
backend="both"
while [ $# -gt 0 ]; do
  case "$1" in
    --model) model="$2"; shift 2 ;;
    --images-dir) images_dir="$2"; shift 2 ;;
    --frames) frames="$2"; shift 2 ;;
    --backend) backend="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(dirname -- "$script_dir")
cd "$repo_root"

feature="image-io onnx-inference"
if [ "$backend" != "cpu" ]; then
  feature="image-io onnx-cuda"
fi

echo "# building example (features: $feature)"
cargo build --release --example superpoint_onnx_throughput --features "$feature" >/dev/null

exe_dir="target/release/examples"
if [ "$backend" != "cpu" ]; then
  # (1) Symlink ort's downloaded provider shared libs next to the binary.
  ort_provider=$(find "$HOME/.cache/ort.pyke.io" -name libonnxruntime_providers_shared.so 2>/dev/null | head -1)
  if [ -n "$ort_provider" ]; then
    ort_dir=$(dirname "$ort_provider")
    for so in libonnxruntime_providers_shared.so libonnxruntime_providers_cuda.so; do
      [ -f "$ort_dir/$so" ] && ln -sf "$ort_dir/$so" "$exe_dir/$so"
    done
    echo "# linked ORT CUDA provider libs from $ort_dir"
  else
    echo "# WARNING: ORT provider libs not found under ~/.cache/ort.pyke.io" >&2
  fi

  # (2) Put cuDNN 9 (+ cuBLAS/cuFFT) on LD_LIBRARY_PATH. Prefer the pip
  #     nvidia-* wheels; the system path usually also has cuBLAS/cuFFT.
  nv_root=$(python3 -c "import os,glob;\
    c=glob.glob(os.path.expanduser('~/.local/lib/python*/site-packages/nvidia'))+\
      glob.glob(os.path.expanduser('~/.local/lib/python*/site-packages/nvidia'));\
    print(c[0] if c else '')" 2>/dev/null || true)
  if [ -n "$nv_root" ]; then
    for sub in cudnn cublas cufft curand cuda_runtime; do
      [ -d "$nv_root/$sub/lib" ] && LD_LIBRARY_PATH="$nv_root/$sub/lib:${LD_LIBRARY_PATH:-}"
    done
    export LD_LIBRARY_PATH
    echo "# LD_LIBRARY_PATH includes pip nvidia-* libs under $nv_root"
  fi
fi

echo "# running"
"$exe_dir/superpoint_onnx_throughput" \
  --model "$model" --images-dir "$images_dir" --frames "$frames" --backend "$backend"
