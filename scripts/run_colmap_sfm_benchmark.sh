#!/usr/bin/env sh
# Unordered structure-from-motion on a real COLMAP photo collection, scored
# against COLMAP's own reconstruction.
#
# The ordered SfM benchmark (run_euroc_sfm_benchmark.sh) reconstructs from an
# *ordered* stereo video; run_unordered_sfm_benchmark.sh reconstructs from a
# strided, order-shuffled subset of one orbit. This one closes the loop on the
# README's headline claim: take a genuine, unordered internet-style photo set —
# one of COLMAP's own example datasets — reconstruct it from scratch with a
# completely independent SuperPoint frontend, and align the recovered camera
# centres to COLMAP's reference model with a Sim(3) transform (the right gauge
# for a monocular reconstruction whose absolute scale is free).
#
# Pipeline:
#   1. download + unzip the dataset (images/ + COLMAP sparse/ text model);
#   2. export undistorted SuperPoint features with the repo's helper, reading the
#      intrinsics from sparse/cameras.txt (SIMPLE_PINHOLE / PINHOLE /
#      SIMPLE_RADIAL / OPENCV) and undistorting keypoints to an ideal pinhole;
#   3. reconstruct with unordered_sfm_demo (VLAD view graph -> essential-matrix
#      verification -> robust multi-seed incremental SfM -> COLMAP export);
#   4. Sim(3)-align to sparse/images.txt and report the camera-centre RMSE.
#
#   scripts/run_colmap_sfm_benchmark.sh --dataset gerrard-hall
#
# Measured (default VLAD top-k=12 retrieval, no hand-tuning):
#   South Building (128 photos)  128 / 128 registered, Sim(3) 0.58 cm RMSE
#   Gerrard Hall   (100 photos)   98 / 100 registered, Sim(3) 0.68 cm RMSE
# both ~0.1 % of the trajectory extent vs COLMAP's own model.
#
# Requires: a Python env with torch + lightglue (the SuperPoint export stage) and
# curl/wget + unzip. The export GPU step dominates the runtime; pass
# --python /usr/bin/python3 (or wherever lightglue lives) and --device cpu/cuda.
set -eu

dataset="${COLMAP_SFM_DATASET:-south-building}"
data_root="${COLMAP_SFM_DATA_ROOT:-$HOME/datasets/colmap_sfm}"
out_dir="${COLMAP_SFM_OUT_DIR:-target/colmap_sfm_benchmark}"
python_bin="${COLMAP_SFM_PYTHON:-python3}"
device="${COLMAP_SFM_DEVICE:-cuda}"
max_keypoints="${COLMAP_SFM_MAX_KEYPOINTS:-2048}"
retrieval_topk="${COLMAP_SFM_TOPK:-12}"
min_matches="${COLMAP_SFM_MIN_MATCHES:-30}"
base_url="${COLMAP_SFM_BASE_URL:-https://demuc.de/colmap/datasets}"

usage() {
    sed -n '2,37p' "$0"
    echo
    echo "Flags: --dataset south-building|gerrard-hall  --data-root DIR  --out-dir DIR"
    echo "       --python BIN  --device cuda|cpu  --topk N  --min-matches N  --max-keypoints N"
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dataset) dataset="$2"; shift 2 ;;
        --data-root) data_root="$2"; shift 2 ;;
        --out-dir) out_dir="$2"; shift 2 ;;
        --python) python_bin="$2"; shift 2 ;;
        --device) device="$2"; shift 2 ;;
        --topk) retrieval_topk="$2"; shift 2 ;;
        --min-matches) min_matches="$2"; shift 2 ;;
        --max-keypoints) max_keypoints="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

case "$dataset" in
    south-building) zip_name="South-Building.zip" ;;
    gerrard-hall)   zip_name="gerrard-hall.zip" ;;
    *) echo "error: unknown --dataset '$dataset' (south-building | gerrard-hall)" >&2; exit 1 ;;
esac

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(dirname -- "$script_dir")
mkdir -p "$data_root" "$out_dir"

# ---- 1. Download + unzip the dataset (idempotent) ------------------------------
sparse_cam=$(find "$data_root" -path '*/sparse/cameras.txt' 2>/dev/null | head -1 || true)
if [ -z "$sparse_cam" ]; then
    zip_path="$data_root/$zip_name"
    if [ ! -f "$zip_path" ]; then
        echo "downloading $zip_name ..."
        if command -v curl >/dev/null 2>&1; then
            curl -fL "$base_url/$zip_name" -o "$zip_path"
        elif command -v wget >/dev/null 2>&1; then
            wget -O "$zip_path" "$base_url/$zip_name"
        else
            echo "error: need curl or wget to download $zip_name" >&2
            exit 1
        fi
    fi
    echo "unzipping $zip_name ..."
    unzip -q "$zip_path" -d "$data_root"
    sparse_cam=$(find "$data_root" -path '*/sparse/cameras.txt' 2>/dev/null | head -1 || true)
fi
[ -n "$sparse_cam" ] || { echo "error: sparse/cameras.txt not found under $data_root" >&2; exit 1; }

sparse_dir=$(dirname -- "$sparse_cam")
ds_dir=$(dirname -- "$sparse_dir")
images_dir="$ds_dir/images"
[ -d "$images_dir" ] || { echo "error: images/ not found next to $sparse_dir" >&2; exit 1; }
n_images=$(find "$images_dir" -maxdepth 1 -type f | wc -l | tr -d ' ')
echo "dataset $dataset: $n_images images, GT model $sparse_dir"

# ---- 2. Derive the ideal-pinhole the demo needs from sparse/cameras.txt --------
# The export undistorts keypoints to this pinhole; mirror its per-model convention
# (SIMPLE_PINHOLE/SIMPLE_RADIAL/RADIAL share one focal; PINHOLE/OPENCV have fx,fy).
pinhole=$(awk '
    /^#/ { next }
    NF >= 7 {
        model=$2; w=$3; h=$4; p1=$5; p2=$6; p3=$7; p4=$8
        if (model=="SIMPLE_PINHOLE" || model=="SIMPLE_RADIAL" || model=="RADIAL")
            printf "--width %s --height %s --fx %s --fy %s --cx %s --cy %s", w, h, p1, p1, p2, p3
        else if (model=="PINHOLE" || model=="OPENCV" || model=="FULL_OPENCV")
            printf "--width %s --height %s --fx %s --fy %s --cx %s --cy %s", w, h, p1, p2, p3, p4
        exit
    }' "$sparse_cam")
[ -n "$pinhole" ] || { echo "error: unsupported camera model in $sparse_cam" >&2; exit 1; }
echo "demo pinhole: $pinhole"

# ---- 3. Export undistorted SuperPoint features (cached) ------------------------
feat_dir="$out_dir/$dataset/features"
mkdir -p "$feat_dir"
n_feat=$(find "$feat_dir" -maxdepth 1 -name '*_features.txt' 2>/dev/null | wc -l | tr -d ' ')
if [ "$n_feat" != "$n_images" ]; then
    echo "exporting SuperPoint features ($python_bin, device=$device) ..."
    "$python_bin" "$script_dir/export_superpoint_undistorted.py" \
        --images-dir "$images_dir" \
        --cameras-txt "$sparse_cam" \
        --out-dir "$feat_dir" \
        --max-keypoints "$max_keypoints" \
        --device "$device"
else
    echo "reusing $n_feat cached feature files in $feat_dir"
fi

# Match the COLMAP NAME extension so the export's image names line up with the GT
# (compare_sfm_sim3.py also tolerates a mismatch — it keys on the first integer).
img_suffix=$(find "$images_dir" -maxdepth 1 -type f | head -1 | sed 's/.*\(\.[^.\/]*\)$/\1/')

# ---- 4. Reconstruct with the unordered SfM demo --------------------------------
colmap_out="$out_dir/$dataset/colmap"
mkdir -p "$colmap_out"
echo "reconstructing with unordered_sfm_demo (top-k=$retrieval_topk, min-matches=$min_matches) ..."
# shellcheck disable=SC2086
( cd "$repo_root" && cargo run --release --example unordered_sfm_demo -- \
    --features-dir "$feat_dir" \
    --feature-suffix _features.txt --image-suffix "$img_suffix" \
    $pinhole \
    --retrieval-topk "$retrieval_topk" --min-matches "$min_matches" \
    --out-colmap "$colmap_out" )

# ---- 5. Sim(3) comparison vs COLMAP's own model --------------------------------
echo
echo "Sim(3) camera-centre comparison vs COLMAP's reference model:"
"$python_bin" "$script_dir/compare_sfm_sim3.py" \
    "$sparse_dir/images.txt" "$colmap_out/images.txt"

echo
echo "COLMAP model: $colmap_out"
