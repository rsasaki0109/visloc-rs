#!/usr/bin/env bash
# Compute learned global descriptors (EigenPlaces / CosPlace VPR) for an image
# sequence with the in-process ONNX runtime, and write them to a text file for
# use as the loop-closure retrieval front-end (consumed by
# stereo_vo_external_deep_files --global-descriptor-file). No Python at run time.
#
# Handles the two CUDA-provider runtime wrinkles automatically (same as
# run_deep_stereo_slam.sh): symlink ort's provider .so next to the binary, and
# add the pip nvidia-* cuDNN/cuBLAS lib dirs to LD_LIBRARY_PATH.
#
# Export the model once:
#   scripts/export_vpr_onnx.py --out models/eigenplaces_r50_2048.onnx
#
# KITTI seq02:
#   scripts/run_vpr_global_descriptor.sh \
#       --images-dir ~/datasets/kitti_seq02_full --subdir image_0 \
#       --model models/eigenplaces_r50_2048.onnx \
#       --out /tmp/seq02_eigenplaces.txt
set -eu

backend="cuda"
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --onnx-cpu) backend="cpu"; args+=("--onnx-cpu"); shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) args+=("$1"); shift ;;
  esac
done

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(dirname -- "$script_dir")
cd "$repo_root"

feature="image-io onnx-inference"
[ "$backend" != "cpu" ] && feature="image-io onnx-cuda"

echo "# building example (features: $feature)"
cargo build --release --example vpr_global_descriptor_demo --features "$feature" >/dev/null

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

"$exe_dir/vpr_global_descriptor_demo" "${args[@]}"
