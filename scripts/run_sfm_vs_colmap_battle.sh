#!/usr/bin/env bash
# Head-to-head metric-video SfM: visloc-rs vs COLMAP on a EuRoC sequence.
#
# The other SfM benchmarks measure visloc against itself or against COLMAP-as-
# ground-truth. This one is an honest *competitor* head-to-head on the turf the
# PLAN identifies as visloc's: ordered metric video. Both reconstruct the SAME
# rectified frames; both are scored against the timestamped EuRoC GT with the
# same evo tooling (the DROID/DPVO same-tool battle pattern).
#
#   * visloc: linear-time stereo VO + online windowed BA -> one global BA ->
#     COLMAP model (`stereo_vo_external_deep_files --sfm-colmap-out`). METRIC
#     scale by construction (rectified stereo baseline).
#   * COLMAP: monocular sequential_matcher + incremental mapper (its SIFT
#     frontend). Scale-free, so its ATE needs a Sim(3) (scale-fitted) alignment.
#
# Reported per engine: wall-clock, registered / total frames, mean reprojection,
# ATE rmse (SE3 = metric for visloc; Sim3 = scale-absorbed for both).
#
# The decisive axis is scale: COLMAP's incremental mapper interleaves a growing
# global bundle adjustment, so its cost is super-linear in frame count and a
# full 2700-frame flight is hours; visloc's VO frontend is linear and finishes
# in minutes, with metric scale COLMAP's monocular path cannot recover.
#
#   scripts/run_sfm_vs_colmap_battle.sh \
#       --rect-dir /tmp/MH_03_rect --feat-dir /tmp/sp_MH_03 \
#       --gt /tmp/MH_03_gt.tum --frames 2700 --out-dir target/sfm_vs_colmap
#
# Requires: COLMAP (>=4.x), evo, python3, and the rectified images + exported
# SuperPoint/LightGlue stereo+temporal features (reuse the loop-closure /
# euroc-sfm benchmark artifacts; see run_euroc_loop_closure_benchmark.sh).
set -eu

rect_dir="${RECT_DIR:-/tmp/MH_03_rect}"
feat_dir="${FEAT_DIR:-/tmp/sp_MH_03}"
gt_tum="${GT_TUM:-/tmp/MH_03_gt.tum}"
frames="${FRAMES:-2700}"
out_dir="${OUT_DIR:-target/sfm_vs_colmap}"
online_ba_window="${ONLINE_BA_WINDOW:-10}"
online_ba_history="${ONLINE_BA_HISTORY:-20}"
ba_iterations="${BA_ITERATIONS:-30}"

while [ $# -gt 0 ]; do
  case "$1" in
    --rect-dir) rect_dir="$2"; shift 2 ;;
    --feat-dir) feat_dir="$2"; shift 2 ;;
    --gt) gt_tum="$2"; shift 2 ;;
    --frames) frames="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --online-ba-window) online_ba_window="$2"; shift 2 ;;
    --online-ba-history) online_ba_history="$2"; shift 2 ;;
    --ba-iterations) ba_iterations="$2"; shift 2 ;;
    -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(dirname -- "$script_dir")
cd "$repo_root"

ts_txt="$rect_dir/timestamps.txt"
calib="$rect_dir/calib.txt"
images="$rect_dir/image_0"
[ -f "$ts_txt" ] || { echo "error: $ts_txt missing" >&2; exit 1; }
[ -f "$calib" ] || { echo "error: $calib missing" >&2; exit 1; }

# Read the rectified left pinhole from calib P0 (KITTI 3x4 "P0: fx 0 cx ... fy cy ...").
read -r fx cx fy cy <<EOF
$(awk '/^P0:/{print $2, $4, $7, $8}' "$calib")
EOF
size=$(identify "$images/000000.png" 2>/dev/null | awk '{print $3}')
img_w="${size%x*}"; img_h="${size#*x}"
img_w="${img_w:-752}"; img_h="${img_h:-480}"
echo "pinhole fx=$fx fy=$fy cx=$cx cy=$cy  size=${img_w}x${img_h}  frames=$frames"

mkdir -p "$out_dir"

# ---- visloc: stereo VO + online BA -> SfM -> COLMAP model -------------------
echo "# visloc-rs stereo VO + online BA SfM"
cargo build --release --example stereo_vo_external_deep_files >/dev/null 2>&1
mkdir -p "$out_dir/visloc_vo"
t0=$(date +%s)
./target/release/examples/stereo_vo_external_deep_files \
  --features-dir "$feat_dir" --frames "$frames" --out-dir "$out_dir/visloc_vo" \
  --relative-pose-mode pnp --calib "$calib" --projection-left P0 --projection-right P1 \
  --width "$img_w" --height "$img_h" \
  --min-stereo-confidence 0.5 --min-temporal-confidence 0.5 \
  --online-ba --online-ba-window "$online_ba_window" --online-ba-history "$online_ba_history" \
  --loop-closure --loop-min-frame-gap 200 --loop-two-view-ba --loop-edge-information \
  --final-global-ba-iterations "$ba_iterations" \
  --sfm-colmap-out "$out_dir/visloc_colmap" 2>&1 | grep -iE "SfM reconstruction|SfM COLMAP|LOOP-CLOSURE" || true
t1=$(date +%s); visloc_wall=$((t1-t0))

# Score the SfM-export model poses (apples-to-apples with COLMAP's model).
python3 "$script_dir/colmap_images_to_tum.py" "$out_dir/visloc_colmap/images.txt" "$ts_txt" "$out_dir/visloc.tum"
visloc_se3=$(evo_ape tum "$gt_tum" "$out_dir/visloc.tum" -a  2>/dev/null | awk '/rmse/{print $2}')
visloc_sim3=$(evo_ape tum "$gt_tum" "$out_dir/visloc.tum" -as 2>/dev/null | awk '/rmse/{print $2}')
visloc_reg=$(grep -vcE '^#' "$out_dir/visloc_colmap/images.txt" 2>/dev/null); visloc_reg=$((visloc_reg/2))
# Also score the loop-closed VO trajectory (the SLAM-grade poses, before the
# reproj-minimising export BA deforms them).
if [ -f "$out_dir/visloc_vo/vo_poses.txt" ]; then
  python3 "$script_dir/kitti_poses_to_tum.py" "$out_dir/visloc_vo/vo_poses.txt" "$ts_txt" "$out_dir/visloc_traj.tum" >/dev/null 2>&1 || true
  visloc_traj_se3=$(evo_ape tum "$gt_tum" "$out_dir/visloc_traj.tum" -a 2>/dev/null | awk '/rmse/{print $2}')
fi

# ---- COLMAP: monocular sequential matcher + incremental mapper -------------
echo "# COLMAP monocular incremental SfM (this is the slow one)"
cdb="$out_dir/colmap/database.db"; csparse="$out_dir/colmap/sparse"
mkdir -p "$out_dir/colmap" "$csparse"; rm -f "$cdb"; rm -rf "$csparse"/*
mkdir -p "$out_dir/colmap/images"
# COLMAP needs a flat image dir of the frames it should use.
n=0; for f in $(ls "$images"/*.png | head -n "$frames"); do ln -sf "$f" "$out_dir/colmap/images/$(basename "$f")"; n=$((n+1)); done
t0=$(date +%s)
colmap feature_extractor --database_path "$cdb" --image_path "$out_dir/colmap/images" \
  --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE \
  --ImageReader.camera_params "$fx,$fy,$cx,$cy" --FeatureExtraction.use_gpu 0 >/dev/null 2>&1
colmap sequential_matcher --database_path "$cdb" --FeatureMatching.use_gpu 0 >/dev/null 2>&1
colmap mapper --database_path "$cdb" --image_path "$out_dir/colmap/images" --output_path "$csparse" >/dev/null 2>&1
t1=$(date +%s); colmap_wall=$((t1-t0))

cmodel=$(ls -d "$csparse"/*/ 2>/dev/null | head -1)
colmap_reg="?"; colmap_sim3="?"
if [ -n "${cmodel:-}" ]; then
  [ -f "$cmodel/images.bin" ] && colmap model_converter --input_path "$cmodel" --output_path "$cmodel" --output_type TXT >/dev/null 2>&1 || true
  python3 "$script_dir/colmap_images_to_tum.py" "$cmodel/images.txt" "$ts_txt" "$out_dir/colmap.tum"
  colmap_sim3=$(evo_ape tum "$gt_tum" "$out_dir/colmap.tum" -as 2>/dev/null | awk '/rmse/{print $2}')
  colmap_reg=$(grep -vcE '^#' "$cmodel/images.txt" 2>/dev/null); colmap_reg=$((colmap_reg/2))
fi

# ---- Table -----------------------------------------------------------------
echo
echo "================ SfM vs COLMAP — $frames frames ================"
printf "%-26s %-10s %-12s %-12s %-10s\n" "engine" "wall(s)" "registered" "ATE_SE3(m)" "ATE_Sim3(m)"
printf "%-26s %-10s %-12s %-12s %-10s\n" "visloc stereo VO+loop SfM" "$visloc_wall" "$visloc_reg/$frames" "${visloc_se3:-?}" "${visloc_sim3:-?}"
printf "%-26s %-10s %-12s %-12s %-10s\n" "COLMAP mono incremental" "$colmap_wall" "$colmap_reg/$frames" "scale-free" "${colmap_sim3:-?}"
echo "================================================================"
[ -n "${visloc_traj_se3:-}" ] && echo "visloc loop-closed VO trajectory (SLAM-grade) ATE_SE3 = ${visloc_traj_se3}m"
