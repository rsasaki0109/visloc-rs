#!/usr/bin/env bash
# Full in-process deep front-end (SuperPoint + LightGlue, ONNX) — CPU vs CUDA.
#
# Runs the entire learned front-end (extraction AND matching) inside the
# process via ONNX Runtime, no Python and no pre-exported feature dump. Handles
# the two CUDA-provider runtime wrinkles automatically (see
# run_superpoint_onnx_throughput.sh for the explanation):
#   1. symlink ort's downloaded provider shared libs next to the binary;
#   2. add the pip nvidia-* cuDNN/cuBLAS/cuFFT lib dirs to LD_LIBRARY_PATH.
#
# Export the two models first:
#   scripts/export_superpoint_onnx.py --out models/superpoint_1500.onnx
#   scripts/export_lightglue_onnx.py  --out models/lightglue.onnx
#
#   scripts/run_deep_frontend_onnx_demo.sh \
#       --images-dir /tmp/MH_03_rect/image_0 --pairs 150 --backend both
set -eu

superpoint_model="models/superpoint_1500.onnx"
lightglue_model="models/lightglue.onnx"
images_dir="/tmp/MH_03_rect/image_0"
pairs=150
backend="both"
while [ $# -gt 0 ]; do
  case "$1" in
    --superpoint-model) superpoint_model="$2"; shift 2 ;;
    --lightglue-model) lightglue_model="$2"; shift 2 ;;
    --images-dir) images_dir="$2"; shift 2 ;;
    --pairs) pairs="$2"; shift 2 ;;
    --backend) backend="$2"; shift 2 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(dirname -- "$script_dir")
cd "$repo_root"

feature="image-io onnx-inference"
[ "$backend" != "cpu" ] && feature="image-io onnx-cuda"

echo "# building example (features: $feature)"
cargo build --release --example deep_frontend_onnx_demo --features "$feature" >/dev/null

exe_dir="target/release/examples"
if [ "$backend" != "cpu" ]; then
  ort_provider=$(find "$HOME/.cache/ort.pyke.io" -name libonnxruntime_providers_shared.so 2>/dev/null | head -1)
  if [ -n "$ort_provider" ]; then
    ort_dir=$(dirname "$ort_provider")
    for so in libonnxruntime_providers_shared.so libonnxruntime_providers_cuda.so; do
      [ -f "$ort_dir/$so" ] && ln -sf "$ort_dir/$so" "$exe_dir/$so"
    done
  fi
  nv_root=$(python3 -c "import glob,os;\
    c=glob.glob(os.path.expanduser('~/.local/lib/python*/site-packages/nvidia'));\
    print(c[0] if c else '')" 2>/dev/null || true)
  if [ -n "$nv_root" ]; then
    for sub in cudnn cublas cufft curand cuda_runtime; do
      [ -d "$nv_root/$sub/lib" ] && LD_LIBRARY_PATH="$nv_root/$sub/lib:${LD_LIBRARY_PATH:-}"
    done
    export LD_LIBRARY_PATH
  fi
fi

"$exe_dir/deep_frontend_onnx_demo" \
  --superpoint-model "$superpoint_model" --lightglue-model "$lightglue_model" \
  --images-dir "$images_dir" --pairs "$pairs" --backend "$backend"
