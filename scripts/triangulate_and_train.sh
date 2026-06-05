#!/usr/bin/env bash
# Real-SfM init for the GT-posed EuRoC splat: random volumetric points give the
# gsplat trainer a huge kNN init-scale (surfaces fill <1% of the volume) and it
# renders fog. Instead we keep the ground-truth poses fixed and triangulate REAL
# surface points with COLMAP (feature extract -> sequential match ->
# point_triangulator), then undistort + retrain. Surface points -> small init
# scale -> actual reconstruction.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

OUT=target/euroc_mh01_gt_colmap
IMG=$OUT/images
DB=$OUT/database.db
MODEL_IN=$OUT/sparse/0
MODEL_TRI=$OUT/sparse_tri
GSMAPPER_SRC="/media/sasaki/aiueo/ai_coding_ws/old_~2026/tier1/nerf-gs-playground/src"
PARAMS="458.654,457.296,367.215,248.375,-0.28340811,0.07395907,0.00019359,1.76187114e-05"

rm -f "$DB"
echo "[1/5] feature_extractor"
colmap feature_extractor --database_path "$DB" --image_path "$IMG" \
  --ImageReader.camera_model OPENCV --ImageReader.single_camera 1 \
  --ImageReader.camera_params "$PARAMS" \
  --FeatureExtraction.use_gpu 0 --SiftExtraction.max_num_features 4096 >/dev/null

echo "[2/5] sequential_matcher"
colmap sequential_matcher --database_path "$DB" \
  --FeatureMatching.use_gpu 0 --SequentialMatching.overlap 15 >/dev/null

echo "[3/5] point_triangulator (GT poses fixed)"
# point_triangulator needs an empty points3D.txt in the input model
: > "$MODEL_IN/points3D.txt"
rm -rf "$MODEL_TRI"; mkdir -p "$MODEL_TRI"
colmap point_triangulator --database_path "$DB" --image_path "$IMG" \
  --input_path "$MODEL_IN" --output_path "$MODEL_TRI" \
  --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 \
  --Mapper.ba_refine_extra_params 0 >/dev/null
NPTS=$(grep -vc '^#' "$MODEL_TRI/points3D.txt" || true)
echo "    triangulated points: $NPTS"

echo "[4/5] image_undistorter (with triangulated model)"
rm -rf "$OUT/undistorted"
colmap image_undistorter --image_path "$IMG" --input_path "$MODEL_TRI" \
  --output_path "$OUT/undistorted" --output_type COLMAP >/dev/null

echo "[5/5] gsplat train (15000 iters)"
python3 - "$OUT" "$GSMAPPER_SRC" <<'PY'
import sys, os
out, src = sys.argv[1], sys.argv[2]
sys.path.insert(0, src)
from gs_sim2real.train.gsplat_trainer import train_gsplat
from gs_sim2real.viewer.web_export import ply_to_splat
# trainer reads <out>/sparse + <out>/images; point it at the triangulated model
ply = train_gsplat(data_dir=out, output_dir=os.path.join(out, "gsplat_sfm"), num_iterations=15000)
splat = ply_to_splat(str(ply), os.path.join(out, "euroc_mh01_gt_sfm.splat"), min_opacity=0.02)
print("SPLAT", splat, os.path.getsize(splat)//32, "gaussians")
PY
echo "DONE"
