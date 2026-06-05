#!/usr/bin/env sh
# Structure-from-Motion reconstruction benchmark on a EuRoC MAV sequence.
#
# The loop-closure benchmarks (run_{kitti,euroc}_loop_closure_benchmark.sh)
# measure trajectory accuracy. This one measures the *structure* visloc-rs's
# SfM pillar produces: it turns a stereo VO run into a COLMAP-grade sparse
# reconstruction and reports how much a single global bundle adjustment tightens
# the multi-view reprojection error.
#
# The streaming VO lifts one fresh landmark per frame (every 3D point seen once),
# so there is no multi-view constraint for a bundle adjustment to exploit. The
# `--sfm-colmap-out` path instead chains the temporal matches into merged
# multi-view tracks, runs ONE global BA over all poses + landmarks (pose 0
# fixed, metric scale anchored by the rectified stereo baseline), and writes a
# COLMAP model whose POINT3D TRACK[] tails span every observing frame — the form
# a downstream 3D Gaussian Splatting / MVS pipeline needs.
#
#   scripts/run_euroc_sfm_benchmark.sh --mav0 /path/to/MH_03_medium/mav0 --frames 2700
#
# Requires: python with torch+lightglue+opencv (rectify+export stages) and the
# built stereo_vo_external_deep_files example. Reuses the same rectified images
# and SuperPoint/LightGlue features as the loop-closure benchmark, so pass
# --rect-dir / --feat-dir to skip re-deriving them.
#
# Measured on MH_03_medium (2700 frames): 178973 merged tracks, 2.03 M
# observations (~11.3 views/track), mean reprojection 4.08 px -> 1.04 px (3.9x).
set -eu

mav0="${EUROC_SFM_MAV0:-}"
out_dir="${EUROC_SFM_OUT_DIR:-target/euroc_sfm_benchmark}"
frames="${EUROC_SFM_FRAMES:-2700}"
device="${EUROC_SFM_DEVICE:-cuda}"
max_keypoints="${EUROC_SFM_MAX_KEYPOINTS:-2048}"
min_stereo_confidence="${EUROC_SFM_MIN_STEREO_CONFIDENCE:-0.5}"
min_temporal_confidence="${EUROC_SFM_MIN_TEMPORAL_CONFIDENCE:-0.5}"
ba_iterations="${EUROC_SFM_BA_ITERATIONS:-30}"
img_width="${EUROC_SFM_WIDTH:-752}"
img_height="${EUROC_SFM_HEIGHT:-480}"
python_bin="${EUROC_SFM_PYTHON:-python3}"
rect_dir="${EUROC_SFM_RECT_DIR:-}"
feat_dir="${EUROC_SFM_FEAT_DIR:-}"
skip_rectify=0
skip_export=0

usage() {
  cat <<'EOF'
usage: scripts/run_euroc_sfm_benchmark.sh --mav0 <dir> [options]

Build a COLMAP-grade SfM reconstruction from a EuRoC stereo VO run and report
the multi-view reprojection error a single global bundle adjustment achieves.

Options:
  --mav0 <dir>               EuRoC mav0/ dir (cam0, cam1)
  --out-dir <dir>            benchmark output root (default target/euroc_sfm_benchmark)
  --frames <n>               frames to process (default 2700)
  --rect-dir <dir>           reuse an already-rectified dir (implies --skip-rectify)
  --feat-dir <dir>           reuse already-exported features (implies --skip-export)
  --ba-iterations <n>        global BA iterations (default 30)
  --width <px> --height <px> rectified image size for cameras.txt (default 752x480)
  --device <auto|cpu|cuda>   torch device for the export (default cuda)
  --python <bin>             python with torch+lightglue+opencv (default python3)
  --skip-rectify             reuse $out_dir/rect (or --rect-dir)
  --skip-export              reuse $out_dir/external_deep (or --feat-dir)
  -h, --help                 show this help

The COLMAP model is written to $out_dir/colmap (cameras/images/points3D.txt).
Point it at gsplat / nerfstudio via $out_dir/colmap as the sparse model.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --mav0) mav0="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --frames) frames="$2"; shift 2 ;;
    --rect-dir) rect_dir="$2"; skip_rectify=1; shift 2 ;;
    --feat-dir) feat_dir="$2"; skip_export=1; shift 2 ;;
    --ba-iterations) ba_iterations="$2"; shift 2 ;;
    --width) img_width="$2"; shift 2 ;;
    --height) img_height="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --python) python_bin="$2"; shift 2 ;;
    --skip-rectify) skip_rectify=1; shift ;;
    --skip-export) skip_export=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$mav0" ] && [ "$skip_rectify" -ne 1 ]; then
  echo "error: --mav0 is required (or pass --rect-dir to reuse a rectified dir)" >&2
  usage >&2
  exit 2
fi

mkdir -p "$out_dir"
[ -n "$rect_dir" ] || rect_dir="$out_dir/rect"
[ -n "$feat_dir" ] || feat_dir="$out_dir/external_deep"
colmap_dir="$out_dir/colmap"

echo "# Building examples"
cargo build --release --example stereo_vo_external_deep_files >/dev/null 2>&1

if [ "$skip_rectify" -ne 1 ]; then
  echo "# Rectifying EuRoC stereo (OpenCV)"
  "$python_bin" scripts/rectify_euroc_stereo.py --mav0 "$mav0" --out-dir "$rect_dir"
fi

if [ "$skip_export" -ne 1 ]; then
  echo "# Exporting SuperPoint/LightGlue features ($frames frames, device=$device)"
  "$python_bin" scripts/export_superpoint_lightglue.py \
    --left-dir "$rect_dir/image_0" --right-dir "$rect_dir/image_1" \
    --out-dir "$feat_dir" --frames "$frames" \
    --device "$device" --max-keypoints "$max_keypoints"
fi

mkdir -p "$out_dir/vo"
echo "# Stereo VO + SfM reconstruction -> COLMAP ($colmap_dir)"
./target/release/examples/stereo_vo_external_deep_files \
  --features-dir "$feat_dir" --frames "$frames" --out-dir "$out_dir/vo" \
  --relative-pose-mode pnp --calib "$rect_dir/calib.txt" \
  --projection-left P0 --projection-right P1 \
  --width "$img_width" --height "$img_height" \
  --min-stereo-confidence "$min_stereo_confidence" \
  --min-temporal-confidence "$min_temporal_confidence" \
  --final-global-ba-iterations "$ba_iterations" \
  --sfm-colmap-out "$colmap_dir" \
  2>&1 | tee "$out_dir/vo/sfm.log" | grep -E "SfM" || true

echo
echo "# Reconstruction summary"
grep -E "SfM reconstruction:" "$out_dir/vo/sfm.log" || true
echo "# COLMAP model: $colmap_dir/{cameras,images,points3D}.txt"
echo "# Feed it to 3DGS: place images under <base>/undistorted/images and"
echo "#   $colmap_dir/*.txt under <base>/undistorted/sparse, then run"
echo "#   scripts/gsplat_mcmc_train.py <base> <out.splat> 7000 400000"
