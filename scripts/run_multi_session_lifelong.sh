#!/usr/bin/env bash
# Multi-session lifelong mapping A/B on 7-Scenes chess: build a map from the
# first session (GT poses), grow it across later sessions by relocalizing each
# keyframe against the map so far (no GT), then localize the held-out test
# sessions against the bootstrap-only and full grown maps. Runs the learned
# EigenPlaces retrieval gate and the bag-of-features baseline so their
# cross-session integration can be compared. See
# docs/multi_session_lifelong_benchmark.md.
#
# Pre-export SuperPoint features and EigenPlaces globals for all six sequences
# (the globals are read as text, so no ONNX/Python at run time):
#   scripts/export_vpr_onnx.py --out models/eigenplaces_r50_2048.onnx
#   scripts/export_superpoint_7scenes.py  --dataset $DS --seqs 1,2,3,4,5,6 --stride 20 --out-dir /tmp/sp_7scenes_chess
#   scripts/export_vpr_globals_7scenes.py --dataset $DS --seqs 1,2,3,4,5,6 --stride 20 --out-dir /tmp/vpr_7scenes_chess
#
#   scripts/run_multi_session_lifelong.sh --dataset ~/datasets/7scenes/chess
set -eu

dataset=""
sp_dir=/tmp/sp_7scenes_chess
vpr_dir=/tmp/vpr_7scenes_chess
extra=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dataset) dataset="$2"; shift 2 ;;
    --sp-features-dir) sp_dir="$2"; shift 2 ;;
    --global-descriptor-dir) vpr_dir="$2"; shift 2 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) extra+=("$1"); shift ;;
  esac
done
[ -n "$dataset" ] || { echo "missing --dataset <path to 7scenes/chess>" >&2; exit 1; }

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(dirname -- "$script_dir")
cd "$repo_root"

echo "# building example (features: image-io)"
cargo build --release --example multi_session_lifelong_demo --features image-io >/dev/null
EX=target/release/examples/multi_session_lifelong_demo

common=(--dataset "$dataset" --sp-features-dir "$sp_dir"
        --sessions 1,2,4,6 --test-seqs 3,5
        --session-stride 20 --test-stride 20
        --retrieve-topk 15 --ratio 0.9 --reproj 6 "${extra[@]}")

echo "############ LEARNED (EigenPlaces) ############"
"$EX" "${common[@]}" --global-descriptor-dir "$vpr_dir"

echo
echo "############ BASELINE (bag-of-features) ############"
"$EX" "${common[@]}"
