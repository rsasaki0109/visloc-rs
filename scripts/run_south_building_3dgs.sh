#!/usr/bin/env sh
# Photorealistic 3D Gaussian Splatting from the pure-Rust unordered SfM model of
# a real building (COLMAP's South-Building, 128 unordered photos).
#
# This closes the loop on the unordered-SfM benchmark: the same VLAD -> essential
# -> incremental SfM that scores ~1 cm against COLMAP also produces a COLMAP model
# good enough to train a crisp 3DGS — a completely independent, pure-Rust frontend
# feeding a standard gsplat trainer. Pipeline:
#
#   1. export SuperPoint features (the same 2048-keypoint export the SfM benchmark
#      uses — this reconstructs the *identical* model scored at ~1 cm vs COLMAP);
#   2. reconstruct with unordered_sfm_demo (128/128 images, ~1.4 px mean reproj);
#   3. prepare_3dgs_from_colmap.py: undistort + half-scale the photos to the ideal
#      pinhole, recolour the points, lay out <work>/undistorted/{images,sparse};
#   4. gsplat_sfm_3dgs_train.py: gsplat DefaultStrategy (gradient-driven adaptive
#      densification) + degree-3 SH -> a crisp .splat + GT|render comparison + a
#      fly-through GIF.
#
# The trainer choice is the *entire* crisp lever: the minimal MCMC trainer
# (gsplat_mcmc_train.py) renders this model SOFT, and tightening the SfM
# reprojection from 1.4 px to 0.66 px does NOT sharpen it — DefaultStrategy, which
# clones/splits Gaussians where image gradients demand detail and prunes
# transparent floaters, is what makes it crisp (ablation in the docs).
#
# Requires: a Python env with torch + gsplat + lightglue + opencv (the export and
# train stages); pass --python /path/to/python (default python3) and a CUDA GPU.
set -eu

sb_root="${SB_ROOT:-$HOME/datasets/south-building/south-building}"
work="${SB_3DGS_WORK:-target/south_building_3dgs}"
python_bin="${SB_3DGS_PYTHON:-python3}"
device="${SB_3DGS_DEVICE:-cuda}"
max_keypoints="${SB_3DGS_MAX_KEYPOINTS:-2048}"
iters="${SB_3DGS_ITERS:-30000}"
scale="${SB_3DGS_SCALE:-0.5}"

while [ $# -gt 0 ]; do
    case "$1" in
        --sb-root) sb_root="$2"; shift 2 ;;
        --work) work="$2"; shift 2 ;;
        --python) python_bin="$2"; shift 2 ;;
        --device) device="$2"; shift 2 ;;
        --max-keypoints) max_keypoints="$2"; shift 2 ;;
        --iters) iters="$2"; shift 2 ;;
        --scale) scale="$2"; shift 2 ;;
        -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(dirname -- "$script_dir")
images_dir="$sb_root/images"
cameras_txt="$sb_root/sparse/cameras.txt"
[ -d "$images_dir" ] || { echo "error: $images_dir not found (set --sb-root to the South-Building dir)" >&2; exit 1; }
[ -f "$cameras_txt" ] || { echo "error: $cameras_txt not found" >&2; exit 1; }
mkdir -p "$work"

feat_dir="$work/features"
n_img=$(find "$images_dir" -maxdepth 1 -type f | wc -l | tr -d ' ')
n_feat=$(find "$feat_dir" -maxdepth 1 -name '*_features.txt' 2>/dev/null | wc -l | tr -d ' ')
if [ "$n_feat" != "$n_img" ]; then
    echo "# 1. exporting SuperPoint features ($max_keypoints keypoints, device=$device)"
    "$python_bin" "$script_dir/export_superpoint_undistorted.py" \
        --images-dir "$images_dir" --cameras-txt "$cameras_txt" \
        --out-dir "$feat_dir" --max-keypoints "$max_keypoints" --device "$device"
else
    echo "# 1. reusing $n_feat cached feature files"
fi

# ideal pinhole the demo needs (mirror cameras.txt; South Building = SIMPLE_RADIAL)
pinhole=$(awk '/^#/{next} NF>=7{m=$2;w=$3;h=$4;
    if(m=="SIMPLE_PINHOLE"||m=="SIMPLE_RADIAL"||m=="RADIAL") printf "--width %s --height %s --fx %s --fy %s --cx %s --cy %s",w,h,$5,$5,$6,$7;
    else if(m=="PINHOLE"||m=="OPENCV"||m=="FULL_OPENCV") printf "--width %s --height %s --fx %s --fy %s --cx %s --cy %s",w,h,$5,$6,$7,$8; exit}' "$cameras_txt")

colmap_out="$work/colmap"
echo "# 2. unordered SfM (same settings as the SfM benchmark)"
# shellcheck disable=SC2086
( cd "$repo_root" && cargo run --release --example unordered_sfm_demo -- \
    --features-dir "$feat_dir" --feature-suffix _features.txt --image-suffix .JPG \
    $pinhole --retrieval-topk 12 --min-matches 30 \
    --out-colmap "$colmap_out" )

echo "# 3. undistort + lay out the gsplat model"
"$python_bin" "$script_dir/prepare_3dgs_from_colmap.py" \
    --source-images "$images_dir" --source-cameras "$cameras_txt" \
    --colmap-model "$colmap_out" --out "$work" --scale "$scale"

echo "# 4. gsplat DefaultStrategy + SH training ($iters iters)"
"$python_bin" "$script_dir/gsplat_sfm_3dgs_train.py" \
    "$work" "$work/south_building.splat" "$iters" \
    "$work/south_building_compare.png" 30,64,100

echo
echo "splat:       $work/south_building.splat"
echo "comparison:  $work/south_building_compare.png"
echo "fly-through: $work/south_building_compare_flythrough.gif"
