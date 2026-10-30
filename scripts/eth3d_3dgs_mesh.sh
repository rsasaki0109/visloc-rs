#!/usr/bin/env bash
# 3DGS geometry on an ETH3D high-res training scene, with the CUDA trainer:
# the scene's undistorted DSLR images and ground-truth poses scaled to
# 1600x1067, 30k training steps on every view, a mesh, and the official
# evaluation of that mesh (accuracy / completeness / F1 at 1 cm to 50 cm).
#
#   scripts/eth3d_3dgs_mesh.sh SCENE_DIR WORK_DIR [gsplat_mesh_cuda options...]
#
# SCENE_DIR holds the extracted <scene>_dslr_undistorted.7z and
# <scene>_dslr_scan_eval.7z (dslr_calibration_undistorted/, images/,
# dslr_scan_eval/). Needs VISLOC_GSPLAT_CUDA_LIB (scripts/build_gsplat_cuda.sh)
# and ETH3D_EVAL, the ETH3DMultiViewEvaluation binary
# (scripts/build_eth3d_evaluator.sh). The mesh options default to the
# ETH3D-oriented ones of docs/gsplat_cuda.md: 1 cm voxels, no SfM support
# filter, depths up to 20 m, voxels seen once kept, planar depth.
set -euo pipefail
scene_dir=${1:?usage: eth3d_3dgs_mesh.sh SCENE_DIR WORK_DIR [mesh options...]}
work=${2:?usage: eth3d_3dgs_mesh.sh SCENE_DIR WORK_DIR [mesh options...]}
shift 2
mesh_opts=("$@")
if [ ${#mesh_opts[@]} -eq 0 ]; then
  mesh_opts=(--voxel 0.01 --support-voxels 0 --max-depth 20 --min-weight 1 --planar-depth)
fi
: "${VISLOC_GSPLAT_CUDA_LIB:?set VISLOC_GSPLAT_CUDA_LIB}"
: "${ETH3D_EVAL:?set ETH3D_EVAL to the ETH3DMultiViewEvaluation binary}"
repo=$(cd "$(dirname "$0")/.." && pwd)
(cd "$repo" &&
  cargo build -q --release -p visloc-gsplat-train --example colmap_resize &&
  cargo build -q --release -p visloc-gsplat-cuda --example gsplat_train_cuda --example gsplat_mesh_cuda)
bin=${CARGO_TARGET_DIR:-$repo/target}/release/examples
mkdir -p "$work"
if [ ! -f "$work/data/sparse/0/images.txt" ]; then
  "$bin/colmap_resize" --model "$scene_dir/dslr_calibration_undistorted" \
    --images "$scene_dir/images" --out "$work/data" --size 1600x1067 | tail -1
fi
cd "$work"
if [ ! -f scene.ply ]; then
  "$bin/gsplat_train_cuda" --data data --steps 30000 --eval-every 5000 --holdout 0 \
    --max-gaussians 2500000 --out scene.ply > train.log 2>&1
fi
grep -E "^trained" train.log
"$bin/gsplat_mesh_cuda" --ply scene.ply --data data --out mesh.ply "${mesh_opts[@]}" | grep -E "^mesh"
"$ETH3D_EVAL" --reconstruction_ply_path mesh.ply \
  --ground_truth_mlp_path "$scene_dir/dslr_scan_eval/scan_alignment.mlp" \
  --tolerances 0.01,0.02,0.05,0.1,0.2,0.5 2>&1 | grep -E "^(Tolerances|Complete|Accurac|F1)"
