#!/usr/bin/env sh
# Metric loop-closure benchmark on a EuRoC MAV sequence (UAV, 6-DOF flight).
#
# The KITTI loop-closure benchmark (scripts/run_kitti_loop_closure_benchmark.sh)
# shows loop closure on a ground vehicle. This is the aerial counterpart: a
# 6-DOF MAV flight where the same open-VO-vs-loop-closure comparison holds, on a
# GPS-denied indoor scene with Vicon/Leica ground truth.
#
# EuRoC's cam0/cam1 are radtan-distorted (unlike KITTI's pre-rectified images),
# so this script first undistorts+rectifies with OpenCV
# (scripts/rectify_euroc_stereo.py), then runs the same file-backed
# SuperPoint/LightGlue stereo VO twice over identical features (open vs
# `--loop-closure`). EuRoC ground truth is timestamped, so VO poses are
# converted to TUM via the rectifier's timestamps.txt and scored with evo_ape
# (SE(3) and Sim(3) alignment).
#
#   scripts/run_euroc_loop_closure_benchmark.sh \
#     --mav0 /path/to/MH_03_medium/mav0 --frames 2700
#
# Requires: python with torch+lightglue+opencv (rectify+export stages), evo
# (`pip install evo`), and the built stereo_vo_external_deep_files example.
#
# NOTE on --loop-min-frame-gap: at EuRoC's 20 Hz a small gap matches only
# slow-motion temporal neighbours that are already odometry-consistent and yield
# no drift correction. Measured on MH_03: gap=30 left ATE unchanged (2.46 m),
# gap=200 (10 s) cut it to 0.46 m. Default is 200.
set -eu

mav0="${EUROC_LC_MAV0:-}"
out_dir="${EUROC_LC_OUT_DIR:-target/euroc_loop_closure_benchmark}"
frames="${EUROC_LC_FRAMES:-2700}"
device="${EUROC_LC_DEVICE:-cuda}"
max_keypoints="${EUROC_LC_MAX_KEYPOINTS:-2048}"
min_stereo_confidence="${EUROC_LC_MIN_STEREO_CONFIDENCE:-0.5}"
min_temporal_confidence="${EUROC_LC_MIN_TEMPORAL_CONFIDENCE:-0.5}"
loop_min_frame_gap="${EUROC_LC_LOOP_MIN_FRAME_GAP:-200}"
loop_min_path_length="${EUROC_LC_LOOP_MIN_PATH_LENGTH:-5}"
loop_min_similarity="${EUROC_LC_LOOP_MIN_SIMILARITY:-0.2}"
python_bin="${EUROC_LC_PYTHON:-python3}"
rect_dir="${EUROC_LC_RECT_DIR:-}"
feat_dir="${EUROC_LC_FEAT_DIR:-}"
gt_tum="${EUROC_LC_GT_TUM:-}"
skip_rectify=0
skip_export=0

usage() {
  cat <<'EOF'
usage: scripts/run_euroc_loop_closure_benchmark.sh --mav0 <dir> [options]

Measure ATE before/after metric loop closure on a EuRoC MAV sequence.

Options:
  --mav0 <dir>               EuRoC mav0/ dir (cam0, cam1, state_groundtruth_estimate0)
  --out-dir <dir>            benchmark output root
  --frames <n>               frames to process (default 2700)
  --rect-dir <dir>           reuse an already-rectified dir (implies --skip-rectify)
  --feat-dir <dir>           reuse already-exported features (implies --skip-export)
  --gt-tum <file>            reuse a GT TUM file instead of deriving it from mav0
  --device <auto|cpu|cuda>   torch device for the export (default cuda)
  --loop-min-frame-gap <n>   min frame gap of a loop pair (default 200)
  --loop-min-path-length <m> min accumulated travel between a loop pair (default 5)
  --loop-min-similarity <x>  min VLAD cosine similarity (default 0.2)
  --python <bin>             python with torch+lightglue+opencv (default python3)
  --skip-rectify             reuse $out_dir/rect (or --rect-dir)
  --skip-export              reuse $out_dir/external_deep (or --feat-dir)
  -h, --help                 show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --mav0) mav0="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --frames) frames="$2"; shift 2 ;;
    --rect-dir) rect_dir="$2"; skip_rectify=1; shift 2 ;;
    --feat-dir) feat_dir="$2"; skip_export=1; shift 2 ;;
    --gt-tum) gt_tum="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --loop-min-frame-gap) loop_min_frame_gap="$2"; shift 2 ;;
    --loop-min-path-length) loop_min_path_length="$2"; shift 2 ;;
    --loop-min-similarity) loop_min_similarity="$2"; shift 2 ;;
    --python) python_bin="$2"; shift 2 ;;
    --skip-rectify) skip_rectify=1; shift ;;
    --skip-export) skip_export=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$out_dir"
[ -n "$rect_dir" ] || rect_dir="$out_dir/rect"
[ -n "$feat_dir" ] || feat_dir="$out_dir/external_deep"
[ -n "$gt_tum" ] || gt_tum="$out_dir/gt.tum"

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

# Ground truth -> TUM (timestamp tx ty tz qx qy qz qw); EuRoC GT is body-frame
# p_RS_R + q_RS (w-first); evo_ape's SE(3) alignment absorbs the small constant
# body<-camera offset.
if [ ! -f "$gt_tum" ]; then
  echo "# Deriving GT TUM from mav0/state_groundtruth_estimate0"
  "$python_bin" - "$mav0/state_groundtruth_estimate0/data.csv" "$gt_tum" <<'PY'
import sys
src, out = sys.argv[1], sys.argv[2]
with open(out, "w") as f:
    for ln in open(src):
        if ln.startswith("#") or not ln.strip():
            continue
        v = ln.split(",")
        ts = float(v[0]) / 1e9
        px, py, pz = v[1], v[2], v[3]
        qw, qx, qy, qz = v[4], v[5], v[6], v[7]
        f.write(f"{ts:.9f} {px} {py} {pz} {qx} {qy} {qz} {qw}\n")
PY
fi

run_vo() {
  # run_vo <subdir> [extra VO args...]
  sub="$1"; shift
  mkdir -p "$out_dir/$sub"
  ./target/release/examples/stereo_vo_external_deep_files \
    --features-dir "$feat_dir" --frames "$frames" --out-dir "$out_dir/$sub" \
    --relative-pose-mode pnp --calib "$rect_dir/calib.txt" \
    --projection-left P0 --projection-right P1 \
    --min-stereo-confidence "$min_stereo_confidence" \
    --min-temporal-confidence "$min_temporal_confidence" \
    "$@" > "$out_dir/$sub/vo.log" 2>&1
  "$python_bin" scripts/kitti_poses_to_tum.py \
    "$out_dir/$sub/vo_poses.txt" "$rect_dir/timestamps.txt" "$out_dir/$sub/est.tum"
}

echo "# Open VO (no loop closure)"
run_vo open

echo "# Loop-closure VO (VLAD -> PnP -> GNC SE(3) PGO)"
run_vo loop \
  --loop-closure \
  --loop-min-frame-gap "$loop_min_frame_gap" \
  --loop-min-path-length "$loop_min_path_length" \
  --loop-min-similarity "$loop_min_similarity"

ate() {
  # ate <est.tum> <align-flag>  -> rmse
  evo_ape tum "$gt_tum" "$1" $2 2>/dev/null | awk '/rmse/{print $2; exit}'
}

o_se3=$(ate "$out_dir/open/est.tum" -a)
o_sim3=$(ate "$out_dir/open/est.tum" -as)
l_se3=$(ate "$out_dir/loop/est.tum" -a)
l_sim3=$(ate "$out_dir/loop/est.tum" -as)
verified=$(grep "LOOP-CLOSURE PGO: candidates" "$out_dir/loop/vo.log" | head -1)

{
  echo ""
  echo "# EuRoC loop-closure benchmark (MAV, 6-DOF)"
  echo ""
  echo "ATE rmse via evo_ape, timestamp-associated to the Vicon/Leica ground truth."
  echo ""
  echo "| trajectory     | ATE rmse SE(3) | ATE rmse Sim(3) |"
  echo "| -------------- | -------------: | --------------: |"
  printf "| open VO        | %14s | %15s |\n" "$o_se3" "$o_sim3"
  printf "| + loop closure | %14s | %15s |\n" "$l_se3" "$l_sim3"
  echo ""
  echo "$verified" | sed 's/^/  /'
} | tee "$out_dir/summary.md"
