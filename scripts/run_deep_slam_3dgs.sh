#!/usr/bin/env bash
# End-to-end: single-binary in-process deep stereo SLAM -> metric COLMAP model
# -> 3D Gaussian Splatting. Demonstrates that deep_stereo_slam's --sfm-colmap-out
# output (raw images -> SuperPoint+LightGlue/CUDA -> stereo VO + online BA + loop
# closure -> merged multi-view COLMAP, no Python, no COLMAP mapper) is directly
# 3DGS-trainable.
#
# EuRoC V2_03 Vicon-room orbit (already-rectified, so no undistortion needed):
#   scripts/run_deep_slam_3dgs.sh \
#       --images-dir /tmp/V2_03_rect --calib /tmp/V2_03_rect/calib.txt \
#       --width 752 --height 480 --frames 150 --out /tmp/deep_v203_3dgs
#
# Validated (V2_03, 150 frames): deep_stereo_slam VO 23.9 fps, SfM reproj
# 0.98 -> 0.60 px, 520 tracks; gsplat MCMC 7000 iters -> 153k gaussians, L1
# 0.0204; the splat renders the room (curtains, table, chair) recognizably from
# the SLAM poses. Needs gsplat (pip) + the EuRoC rectified frames.
set -eu

iters=7000
cap=200000
out=/tmp/deep_slam_3dgs
slam_args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --out) out="$2"; shift 2 ;;
    --iters) iters="$2"; shift 2 ;;
    --cap) cap="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) slam_args+=("$1"); shift ;;
  esac
done

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(dirname -- "$script_dir")
cd "$repo_root"

colmap="$out/colmap/sparse/0"
mkdir -p "$colmap"

echo "# [1/3] in-process deep stereo SLAM -> COLMAP model"
scripts/run_deep_stereo_slam.sh "${slam_args[@]}" \
  --sfm-colmap-out "$colmap" --sfm-ba-iterations 15 --out-dir "$out/slam"

echo "# [2/3] lay out gsplat dir (rectified frames are already undistorted)"
base="$out/undistorted"
mkdir -p "$base/sparse" "$base/images"
cp "$colmap"/{cameras,images,points3D}.txt "$base/sparse/"
# Symlink only the frames the COLMAP model references (NAME column of images.txt).
src_left=$(grep -oE '\-\-images-dir [^ ]+' <<<"${slam_args[*]}" | awk '{print $2}')
left_sub=$(grep -oE '\-\-left-subdir [^ ]+' <<<"${slam_args[*]}" | awk '{print $2}'); left_sub=${left_sub:-image_0}
awk 'NR%2==1 && $1!="#"{print $NF}' "$base/sparse/images.txt" | while read -r name; do
  ln -sf "$src_left/$left_sub/$name" "$base/images/$name"
done
echo "    linked $(ls "$base/images" | wc -l) frames"

echo "# [3/3] gsplat MCMC training ($iters iters, cap $cap)"
python3 scripts/gsplat_mcmc_train.py "$out" "$out/splat.ply" "$iters" "$cap"
echo "# done -> $out/splat.ply"
echo "#   fly-through: python3 scripts/render_splat_flythrough.py $out $out/splat.ply $out/flythrough.gif 3 15"
