#!/usr/bin/env bash
# Single-binary deep stereo SLAM — SuperPoint + LightGlue (ONNX/GPU) front-end,
# online BA, and VLAD->PnP->GNC loop closure, all in one Rust binary. No Python,
# no pre-exported feature dump. Handles the two CUDA-provider runtime wrinkles
# automatically (see run_superpoint_onnx_throughput.sh for the explanation):
#   1. symlink ort's downloaded provider shared libs next to the binary;
#   2. add the pip nvidia-* cuDNN/cuBLAS/cuFFT lib dirs to LD_LIBRARY_PATH.
#
# Export the two models first (for a non-752x480 resolution, export LightGlue at
# that size: --width/--height):
#   scripts/export_superpoint_onnx.py --out models/superpoint_1500.onnx
#   scripts/export_lightglue_onnx.py  --out models/lightglue.onnx
#
# EuRoC MH_03 (aerial — wants --loop-min-frame-gap 200):
#   scripts/run_deep_stereo_slam.sh \
#       --images-dir /tmp/MH_03_rect --calib /tmp/MH_03_rect/calib.txt \
#       --width 752 --height 480 --frames 2700 --loop-min-frame-gap 200 \
#       --out-dir target/deep_slam_mh03
#
# KITTI seq00 (driving — default frame gap 50; export LightGlue at 1241x376):
#   scripts/run_deep_stereo_slam.sh \
#       --images-dir ~/datasets/kitti_seq00_full \
#       --calib ~/datasets/kitti_seq00_full/calib.txt \
#       --lightglue-model models/lightglue_kitti.onnx \
#       --width 1241 --height 376 --frames 4541 --out-dir target/deep_slam_kitti00
set -eu

backend="cuda"
args=()
out_dir="target/deep_stereo_slam"
while [ $# -gt 0 ]; do
  case "$1" in
    --onnx-cpu) backend="cpu"; args+=("--onnx-cpu"); shift ;;
    --out-dir) out_dir="$2"; args+=("--out-dir" "$2"); shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) args+=("$1"); shift ;;
  esac
done

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(dirname -- "$script_dir")
cd "$repo_root"

feature="image-io onnx-inference"
[ "$backend" != "cpu" ] && feature="image-io onnx-cuda"

echo "# building example (features: $feature)"
cargo build --release --example deep_stereo_slam --features "$feature" >/dev/null

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

"$exe_dir/deep_stereo_slam" "${args[@]}"
