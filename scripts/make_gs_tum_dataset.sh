#!/usr/bin/env bash
# Build the TUM RGB-D Pioneer-hall 3DGS training set of docs/navigation.md
# ("Real to sim") from the public data: download, undistort, SfM on the CPU,
# metric alignment to the motion capture, crop.
#
#   scripts/make_gs_tum_dataset.sh WORK_DIR
#
# Writes WORK_DIR/train (a COLMAP text model in metres, z up, and 1,3xx
# cropped 600 x 440 images) for
#
#   gsplat_train --data WORK_DIR/train --steps 7000 --max-gaussians 400000
#
# (2.3 h on 4 CPU cores with lavapipe, held-out PSNR 25.0 dB), the set the
# 3DGS-training speed work measures on. Needs curl, about 3 GB of disk (the
# RGB frames only; the tarballs are streamed), python3 with numpy and OpenCV
# (PYTHON=... picks the interpreter), and the release `gsplat_photos` example
# (built here if missing). SfM takes about 10 minutes on 4 cores. Steps whose
# output exists are skipped.
set -euo pipefail
W=$(mkdir -p "$1" && cd "$1" && pwd)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
GS=$ROOT/ros2/visloc-ros2/gs
PY=${PYTHON:-python3}
mkdir -p "$W/data"
for s in 360 slam slam3; do
  d=$W/data/rgbd_dataset_freiburg2_pioneer_$s
  [ -d "$d/rgb" ] && continue
  echo "download pioneer_$s"
  curl -fsSL "https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_pioneer_$s.tgz" \
    | tar -xz -C "$W/data" --exclude='*/depth/*'
done
[ -d "$W/images" ] || "$PY" "$GS/prep_tum_images.py" undistort --data "$W/data" --out "$W/images"
PHOTOS=$ROOT/target/release/examples/gsplat_photos
[ -x "$PHOTOS" ] || (cd "$ROOT" && cargo build --release -p visloc-gsplat-train --features gpu,euroc --example gsplat_photos)
[ -f "$W/sfm/sparse/0/images.txt" ] || "$PHOTOS" --images "$W/images" --out "$W/sfm" --steps 0 --cpu \
  --camera "PINHOLE 640 480 520.9 521.0 325.1 249.7" --no-refine-intrinsics \
  --window 20 --retrieval 30 --max-keypoints 4096 --mask-black 12
[ -f "$W/metric/sparse/0/images.txt" ] || "$PY" "$GS/align_sfm_tum.py" --model "$W/sfm/sparse/0" \
  --images "$W/sfm/images" --data "$W/data" --out "$W/metric"
[ -f "$W/train/sparse/0/images.txt" ] || "$PY" "$GS/prep_tum_images.py" crop \
  --model "$W/metric/sparse/0" --images "$W/sfm/images" --out "$W/train"
echo "training set: $W/train"
