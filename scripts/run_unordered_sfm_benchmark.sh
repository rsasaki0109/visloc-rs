#!/usr/bin/env sh
# Unordered structure-from-motion benchmark (the COLMAP-style SfM pillar).
#
# The ordered SfM benchmark (run_euroc_sfm_benchmark.sh) reconstructs from an
# *ordered* stereo video — temporal matches give tracks, stereo gives scale.
# This one reconstructs from an *unordered* monocular photo set: no temporal
# order, no overlap graph, no metric scale. The pipeline discovers the view
# graph (VLAD retrieval), verifies each pair (essential-matrix RANSAC), and
# grows one reconstruction incrementally (seed -> PnP register -> triangulate ->
# bundle-adjust) — `unordered_sfm_demo` / `visloc_slam::incremental_sfm`.
#
# It reuses the V2_03 Vicon-Room orbit's LEFT-camera SuperPoint features (the
# same capture whose ordered stereo SfM produced the crisp 3DGS splat): take a
# strided subset, shuffle away the order, and reconstruct. When --ordered-colmap
# is given, the recovered camera centres are Sim(3)-aligned to the ordered
# reconstruction to report a camera-centre RMSE.
#
#   scripts/run_unordered_sfm_benchmark.sh \
#       --feat-dir /path/to/V2_03/left-features \
#       --ordered-colmap /path/to/v203_sfm_colmap
#
# Measured (31 left images, frames 0-150 stride 5): 27/31 registered, 608 tracks,
# mean reprojection 0.63 px; Sim(3) vs ordered reconstruction 1.01 cm RMSE.
set -eu

feat_dir="${UNORDERED_SFM_FEAT_DIR:-}"
out_dir="${UNORDERED_SFM_OUT_DIR:-target/unordered_sfm_benchmark}"
ordered_colmap="${UNORDERED_SFM_ORDERED_COLMAP:-}"
feature_suffix="${UNORDERED_SFM_FEATURE_SUFFIX:-_left_features.txt}"
frame_start="${UNORDERED_SFM_FRAME_START:-0}"
frame_end="${UNORDERED_SFM_FRAME_END:-150}"
frame_step="${UNORDERED_SFM_FRAME_STEP:-5}"
img_width="${UNORDERED_SFM_WIDTH:-752}"
img_height="${UNORDERED_SFM_HEIGHT:-480}"
fx="${UNORDERED_SFM_FX:-436.2442956471}"
fy="${UNORDERED_SFM_FY:-436.2442956471}"
cx="${UNORDERED_SFM_CX:-364.4412345886}"
cy="${UNORDERED_SFM_CY:-256.951675415}"
retrieval_topk="${UNORDERED_SFM_TOPK:-10}"
min_matches="${UNORDERED_SFM_MIN_MATCHES:-30}"
python_bin="${UNORDERED_SFM_PYTHON:-python3}"

usage() {
    sed -n '2,24p' "$0"
    echo
    echo "Flags: --feat-dir DIR  --ordered-colmap DIR  --out-dir DIR"
    echo "       --feature-suffix S  --frames START:END:STEP  --topk N  --min-matches N"
    echo "       --width W --height H --fx FX --fy FY --cx CX --cy CY"
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --feat-dir) feat_dir="$2"; shift 2 ;;
        --ordered-colmap) ordered_colmap="$2"; shift 2 ;;
        --out-dir) out_dir="$2"; shift 2 ;;
        --feature-suffix) feature_suffix="$2"; shift 2 ;;
        --frames) frame_start="${2%%:*}"; rest="${2#*:}"; frame_end="${rest%%:*}"; frame_step="${rest#*:}"; shift 2 ;;
        --topk) retrieval_topk="$2"; shift 2 ;;
        --min-matches) min_matches="$2"; shift 2 ;;
        --width) img_width="$2"; shift 2 ;;
        --height) img_height="$2"; shift 2 ;;
        --fx) fx="$2"; shift 2 ;;
        --fy) fy="$2"; shift 2 ;;
        --cx) cx="$2"; shift 2 ;;
        --cy) cy="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

[ -n "$feat_dir" ] || { echo "error: --feat-dir is required (left-camera feature files)" >&2; usage 1; }
[ -d "$feat_dir" ] || { echo "error: feat-dir not found: $feat_dir" >&2; exit 1; }

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(dirname -- "$script_dir")
unordered_dir="$out_dir/unordered_features"
colmap_dir="$out_dir/colmap"
mkdir -p "$unordered_dir" "$colmap_dir"
rm -f "$unordered_dir"/*"$feature_suffix"

# Build the unordered set: strided left-camera feature files, order shuffled
# away (the pipeline rediscovers overlap; lexical filename order is irrelevant).
linked=0
n="$frame_start"
while [ "$n" -le "$frame_end" ]; do
    name=$(printf "frame_%06d%s" "$n" "$feature_suffix")
    src="$feat_dir/$name"
    if [ -f "$src" ]; then
        ln -sf "$(CDPATH= cd -- "$(dirname -- "$src")" && pwd)/$name" "$unordered_dir/$name"
        linked=$((linked + 1))
    fi
    n=$((n + frame_step))
done
echo "unordered set: $linked left-camera images (frames $frame_start..$frame_end step $frame_step)"
[ "$linked" -ge 2 ] || { echo "error: need >=2 feature files; found $linked" >&2; exit 1; }

echo "building reconstruction with unordered_sfm_demo ..."
( cd "$repo_root" && cargo run --release --example unordered_sfm_demo -- \
    --features-dir "$unordered_dir" \
    --feature-suffix "$feature_suffix" --image-suffix .png \
    --width "$img_width" --height "$img_height" \
    --fx "$fx" --fy "$fy" --cx "$cx" --cy "$cy" \
    --retrieval-topk "$retrieval_topk" --min-matches "$min_matches" \
    --out-colmap "$colmap_dir" )

if [ -n "$ordered_colmap" ] && [ -f "$ordered_colmap/images.txt" ]; then
    echo
    echo "Sim(3) comparison vs ordered reconstruction:"
    "$python_bin" "$script_dir/compare_sfm_sim3.py" \
        "$ordered_colmap/images.txt" "$colmap_dir/images.txt"
fi

echo
echo "COLMAP model: $colmap_dir"
