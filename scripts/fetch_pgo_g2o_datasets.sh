#!/usr/bin/env sh
# Fetch the canonical SE(3) pose-graph benchmark datasets (`EDGE_SE3:QUAT`
# format) used by the `pgo_g2o_benchmark` example. Mirrors the datasets the
# SLAM back-end literature (g2o, GTSAM, SE-Sync) reports on.
#
# Usage:
#   scripts/fetch_pgo_g2o_datasets.sh [out-dir]
#
# Then, e.g.:
#   cargo run --release --example pgo_g2o_benchmark -- <out-dir>/sphere2500.g2o
set -eu

out_dir="${1:-datasets/pgo_g2o}"
base="https://raw.githubusercontent.com/david-m-rosen/SE-Sync/master/data"
datasets="sphere2500 torus3D parking-garage cubicle grid3D rim"

mkdir -p "$out_dir"
for name in $datasets; do
    dest="$out_dir/$name.g2o"
    echo "Fetching $name -> $dest"
    curl -fSL -o "$dest" "$base/$name.g2o"
done

echo "Done. Run e.g.:"
echo "  cargo run --release --example pgo_g2o_benchmark -- $out_dir/sphere2500.g2o"
