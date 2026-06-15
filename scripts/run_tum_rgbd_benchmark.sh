#!/usr/bin/env sh
# Metric RGB-D VO/SLAM benchmark on a TUM RGB-D sequence (indoor, handheld).
#
# The KITTI / EuRoC loop-closure benchmarks show ground-vehicle and aerial
# stereo SLAM. This is the indoor handheld counterpart on the classic TUM
# RGB-D benchmark, where ORB-SLAM2 / ElasticFusion publish ATE.
#
# TUM RGB-D has no stereo pair — only one RGB image plus a registered depth
# map. We feed the same stereo VO backend by *virtual stereo*: each depth-valid
# SuperPoint keypoint becomes a synthetic right keypoint shifted by the
# disparity it would have at a chosen virtual baseline
# (disparity = baseline * fx / depth). The Rust triangulator inverts this
# exactly, so the recovered depth — and hence the metric scale — is the true
# TUM depth, independent of the baseline, as long as the SAME --baseline is
# passed to the binary. This reuses the entire stereo VO / online-BA / loop
# backend (examples/stereo_vo_external_deep_files.rs) with no Rust changes.
#
#   scripts/run_tum_rgbd_benchmark.sh \
#     --seq-dir /path/to/rgbd_dataset_freiburg1_xyz
#
# Requires: python with torch+lightglue+pillow (export stage), evo
# (`pip install evo`), and the built stereo_vo_external_deep_files example.
#
# Best measured config is `full_tv` (window BA + loop closure + two-view loop
# BA). On fr1_xyz it reaches ATE RMSE ~0.014 m, on fr1_desk ~0.026 m
# (~1.3-1.6x ORB-SLAM2 RGB-D), vision-only.
set -eu

seq_dir="${TUM_SEQ_DIR:-}"
out_dir="${TUM_OUT_DIR:-target/tum_rgbd_benchmark}"
device="${TUM_DEVICE:-cuda}"
max_keypoints="${TUM_MAX_KEYPOINTS:-2048}"
baseline="${TUM_BASELINE:-0.1}"
# Freiburg1 intrinsics by default; override for fr2/fr3.
fx="${TUM_FX:-517.3}"; fy="${TUM_FY:-516.5}"; cx="${TUM_CX:-318.6}"; cy="${TUM_CY:-255.3}"
min_depth="${TUM_MIN_DEPTH:-0.3}"
loop_min_frame_gap="${TUM_LOOP_MIN_FRAME_GAP:-50}"
loop_min_path_length="${TUM_LOOP_MIN_PATH_LENGTH:-2}"
loop_min_similarity="${TUM_LOOP_MIN_SIMILARITY:-0.2}"
python_bin="${TUM_PYTHON:-python3}"
feat_dir="${TUM_FEAT_DIR:-}"
skip_export=0

usage() {
  cat <<'EOF'
usage: scripts/run_tum_rgbd_benchmark.sh --seq-dir <dir> [options]

Measure ATE on a TUM RGB-D sequence via the virtual-stereo VO backend.
Runs three conditions: open VO, full (window BA + loop), full_tv (+ two-view BA).

Options:
  --seq-dir <dir>            TUM sequence dir (rgb.txt, depth.txt, rgb/, depth/, groundtruth.txt)
  --out-dir <dir>            benchmark output root
  --feat-dir <dir>          reuse already-exported virtual-stereo features (implies --skip-export)
  --device <auto|cpu|cuda>  torch device for the export (default cuda)
  --baseline <m>            virtual stereo baseline (default 0.1)
  --fx/--fy/--cx/--cy <v>   intrinsics (default Freiburg1)
  --loop-min-frame-gap <n>  min frame gap of a loop pair (default 50)
  --python <bin>            python with torch+lightglue+pillow (default python3)
  --skip-export             reuse $out_dir/external_deep (or --feat-dir)
  -h, --help                show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --seq-dir) seq_dir="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --feat-dir) feat_dir="$2"; skip_export=1; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --baseline) baseline="$2"; shift 2 ;;
    --fx) fx="$2"; shift 2 ;;
    --fy) fy="$2"; shift 2 ;;
    --cx) cx="$2"; shift 2 ;;
    --cy) cy="$2"; shift 2 ;;
    --loop-min-frame-gap) loop_min_frame_gap="$2"; shift 2 ;;
    --loop-min-path-length) loop_min_path_length="$2"; shift 2 ;;
    --loop-min-similarity) loop_min_similarity="$2"; shift 2 ;;
    --python) python_bin="$2"; shift 2 ;;
    --skip-export) skip_export=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$seq_dir" ]; then
  echo "error: --seq-dir is required" >&2
  usage >&2
  exit 2
fi

mkdir -p "$out_dir"
[ -n "$feat_dir" ] || feat_dir="$out_dir/external_deep"
gt="$seq_dir/groundtruth.txt"

echo "# Building example"
cargo build --release --example stereo_vo_external_deep_files >/dev/null 2>&1

if [ "$skip_export" -ne 1 ]; then
  echo "# Exporting virtual-stereo SuperPoint/LightGlue features"
  "$python_bin" scripts/export_tum_rgbd_virtual_stereo.py \
    --seq-dir "$seq_dir" --out-dir "$feat_dir" --device "$device" \
    --max-keypoints "$max_keypoints" --baseline "$baseline" \
    --fx "$fx" --fy "$fy" --cx "$cx" --cy "$cy" --min-depth "$min_depth"
fi

frames=$(find "$feat_dir" -name 'frame_*_left_features.txt' | wc -l | tr -d ' ')
echo "# $frames frames"

cal="--fx $fx --fy $fy --cx $cx --cy $cy --width 640 --height 480 --baseline $baseline --min-depth $min_depth --relative-pose-mode pnp"

run_vo() {
  sub="$1"; shift
  mkdir -p "$out_dir/$sub"
  ./target/release/examples/stereo_vo_external_deep_files \
    --features-dir "$feat_dir" --frames "$frames" --out-dir "$out_dir/$sub" \
    $cal "$@" > "$out_dir/$sub/vo.log" 2>&1
  "$python_bin" scripts/kitti_poses_to_tum.py \
    "$out_dir/$sub/vo_poses.txt" "$feat_dir/frame_timestamps.txt" "$out_dir/$sub/est.tum" >/dev/null
  se3=$(evo_ape tum "$gt" "$out_dir/$sub/est.tum" -a 2>/dev/null | awk '/rmse/{print $2}')
  sim3=$(evo_ape tum "$gt" "$out_dir/$sub/est.tum" -as 2>/dev/null | awk '/rmse/{print $2}')
  loops=$(grep -oE 'verified_loops=[0-9]+' "$out_dir/$sub/vo.log" | head -1)
  printf '%-10s ATE(SE3)=%-9s ATE(Sim3)=%-9s %s\n' "$sub" "$se3" "$sim3" "$loops"
}

echo "# Open VO (no BA, no loop)"
run_vo open

echo "# Full pipeline (window BA + loop closure)"
run_vo full \
  --online-ba --online-ba-window 30 --online-ba-trigger-every 10 \
  --loop-closure --loop-min-frame-gap "$loop_min_frame_gap" \
  --loop-min-path-length "$loop_min_path_length" --loop-min-similarity "$loop_min_similarity"

echo "# Full pipeline + two-view loop BA (best)"
run_vo full_tv \
  --online-ba --online-ba-window 30 --online-ba-trigger-every 10 \
  --loop-closure --loop-min-frame-gap "$loop_min_frame_gap" \
  --loop-min-path-length "$loop_min_path_length" --loop-min-similarity "$loop_min_similarity" \
  --loop-two-view-ba
