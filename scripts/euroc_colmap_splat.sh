#!/usr/bin/env bash
# Lean EuRoC -> COLMAP SfM -> 3DGS pipeline.
#
# The visloc-pose / GT-pose splats render as fog: gsplat derives each gaussian's
# initial scale from the kNN distance of the init points, and random volumetric
# init points (surfaces fill <1% of the volume) give a ~0.2 m init scale ->
# giant gaussians -> fog. The fix is real on-surface SfM points. COLMAP 4.0.3's
# rig/frame model rejects hand-built pose models (point_triangulator aborts), so
# we run full incremental SfM. 590 frames was pathologically slow on CPU-only
# COLMAP (>1.5 h, no model), so we subsample to ~200 frames: enough parallax for
# a clean reconstruction, fast enough to actually finish.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

EUROC=${1:-old_~2026/tier2/simple_visual_slam/datasets/euroc/MH_01_easy/mav0}
OUT=${2:-target/euroc_mh01_fast}
STRIDE=${3:-16}
SKIP=${4:-200}
ITERS=${5:-15000}
GSMAPPER_SRC="old_~2026/tier1/nerf-gs-playground/src"
PARAMS="458.654,457.296,367.215,248.375,-0.28340811,0.07395907,0.00019359,1.76187114e-05"

IMG=$OUT/images
DB=$OUT/database.db
rm -rf "$OUT"; mkdir -p "$IMG"

echo "[0/5] copy frames (stride $STRIDE, skip $SKIP)"
python3 - "$EUROC" "$IMG" "$STRIDE" "$SKIP" <<'PY'
import sys, shutil, pathlib
euroc, img, stride, skip = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
frames = sorted((pathlib.Path(euroc)/"cam0"/"data").glob("*.png"))[skip::stride]
for f in frames:
    shutil.copy(f, img/f.name)
print(f"    copied {len(frames)} frames")
PY

echo "[1/5] feature_extractor"
colmap feature_extractor --database_path "$DB" --image_path "$IMG" \
  --ImageReader.camera_model OPENCV --ImageReader.single_camera 1 \
  --ImageReader.camera_params "$PARAMS" \
  --FeatureExtraction.use_gpu 0 --SiftExtraction.max_num_features 8192 >/dev/null 2>&1

echo "[2/5] sequential_matcher"
colmap sequential_matcher --database_path "$DB" \
  --FeatureMatching.use_gpu 0 --SequentialMatching.overlap 20 \
  --SequentialMatching.loop_detection 1 >/dev/null 2>&1

echo "[3/5] mapper"
mkdir -p "$OUT/sparse"
colmap mapper --database_path "$DB" --image_path "$IMG" --output_path "$OUT/sparse" >/dev/null 2>&1
M=$OUT/sparse/0
colmap model_converter --input_path "$M" --output_path "$M" --output_type TXT >/dev/null 2>&1
echo "    registered: $(grep -c '\.png' $M/images.txt 2>/dev/null||echo 0) imgs, $(grep -vc '^#' $M/points3D.txt 2>/dev/null||echo 0) points"

echo "[4/5] image_undistorter"
colmap image_undistorter --image_path "$IMG" --input_path "$M" \
  --output_path "$OUT/undistorted" --output_type COLMAP >/dev/null 2>&1

echo "[5/5] gsplat train ($ITERS iters)"
python3 - "$OUT" "$GSMAPPER_SRC" "$ITERS" <<'PY'
import sys, os
out, src, iters = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path.insert(0, src)
from gs_sim2real.train.gsplat_trainer import train_gsplat
from gs_sim2real.viewer.web_export import ply_to_splat
ply = train_gsplat(data_dir=out, output_dir=os.path.join(out,"gsplat"), num_iterations=iters)
splat = ply_to_splat(str(ply), os.path.join(out,"euroc_mh01.splat"), min_opacity=0.02)
print("SPLAT", splat, os.path.getsize(splat)//32, "gaussians")
PY
echo "DONE"
