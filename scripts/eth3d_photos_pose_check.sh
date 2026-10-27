#!/usr/bin/env bash
# gsplat_photos SfM on the ETH3D DSLR training scenes, scored against the
# ground-truth poses (scripts/eth3d_pose_errors.py).
#
#   scripts/eth3d_photos_pose_check.sh WORK_DIR TAG [scene ...]
#
# Per scene: download and extract the undistorted DSLR images (0.2-1.3 GB),
# resample them onto the scene's most used camera
# (scripts/eth3d_one_camera.py), run gsplat_photos' SfM at 1,600 px with
# 4,096 SIFT keypoints and the intrinsics refined from a focal length 5% off
# the ground truth, append the scores to WORK_DIR/results.txt as
# "<scene> <TAG> | <registered> | <scores>", then delete the images (the
# resampled copies stay unless ETH3D_DELETE_IMAGES=1).
#
# Compare two builds or settings by running the script twice with different
# TAGs: extra gsplat_photos flags come from PHOTOS_ARGS, extra environment
# variables are simply inherited, and PHOTOS points at another binary
# (default target/release/examples/gsplat_photos, built with
# `cargo build --release -p visloc-gsplat-train --features "gpu euroc"
# --example gsplat_photos`). Needs curl, python3 with numpy, Pillow and py7zr.
# Without a GPU, add `--cpu --cpu-sift` to PHOTOS_ARGS.
set -euo pipefail
W=$1
TAG=$2
shift 2
scenes=("$@")
[ ${#scenes[@]} -gt 0 ] || scenes=(courtyard delivery_area electro facade kicker meadow office
  pipes playground relief relief_2 terrace terrains)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PHOTOS=${PHOTOS:-$ROOT/target/release/examples/gsplat_photos}
mkdir -p "$W"
for s in "${scenes[@]}"; do
  D=$W/$s
  mkdir -p "$D"
  if [ ! -d "$D/one/images/dslr_images_undistorted" ]; then
    curl -fsSL -C - --retry 8 --retry-all-errors --retry-delay 10 -o "$W/$s.7z" \
      "https://www.eth3d.net/data/${s}_dslr_undistorted.7z"
    python3 -c "import py7zr, sys; py7zr.SevenZipFile(sys.argv[1]).extractall(sys.argv[2])" \
      "$W/$s.7z" "$D"
    rm -f "$W/$s.7z"
    python3 -I "$ROOT/scripts/eth3d_one_camera.py" "$D/$s/dslr_calibration_undistorted" \
      "$D/$s/images" "$D/one" >"$D/one.log" 2>&1
    rm -rf "${D:?}/$s"
  fi
  # The ground-truth focal length scaled to 1,600 px, then 5% off.
  read -r width height focal <<<"$(awk '!/^#/{print $3, $4, $5; exit}' \
    "$D/one/calibration/cameras.txt")"
  f=$(python3 -c "print($focal * 1600 / max($width, $height) * 1.05)")
  O=$D/$TAG
  rm -rf "$O"
  # shellcheck disable=SC2086 # PHOTOS_ARGS is a list of flags.
  "$PHOTOS" --images "$D/one/images/dslr_images_undistorted" --out "$O" --max-size 1600 \
    --steps 0 --no-mesh --focal "$f" --max-keypoints 4096 ${PHOTOS_ARGS:-} >"$D/$TAG.log" 2>&1 ||
    { echo "$s $TAG | gsplat_photos failed, see $D/$TAG.log" >>"$W/results.txt"; continue; }
  rm -rf "$O/images" "$O/port"
  echo "$s $TAG | $(grep -oE '[0-9]+/[0-9]+ registered' "$D/$TAG.log" | tail -1) | $(
    python3 -I "$ROOT/scripts/eth3d_pose_errors.py" "$O/sparse/0/images.txt" \
      "$D/one/calibration/images.txt" "$O/image_names.txt"
  )" >>"$W/results.txt"
  [ "${ETH3D_DELETE_IMAGES:-0}" = 1 ] && rm -rf "$D/one/images"
done
tail -n "${#scenes[@]}" "$W/results.txt"
