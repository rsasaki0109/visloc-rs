#!/bin/sh
# Default-path parity check for Basalt VIO changes: runs basalt_euroc_vio_demo stereo and
# --mono on MH_01/MH_04 and prints the sha256 of each trajectory.tum. Compare against a
# baseline recorded before the change.  usage: scripts/bench/basalt_vio_parity.sh <out_dir>
set -eu
OUT="$1"; EUROC_DIR="${EUROC_DIR:-$HOME/datasets/euroc}"
cd "$(dirname "$0")/../.."
cargo build -q --release --example basalt_euroc_vio_demo
for s in MH_01_easy MH_04_difficult; do
  for m in stereo mono; do
    d="$OUT/$m/$s"; mkdir -p "$d"; f=""; [ $m = mono ] && f=--mono
    # shellcheck disable=SC2086
    target/release/examples/basalt_euroc_vio_demo --euroc-dir "$EUROC_DIR/$s" \
      --calibration benchmarks/basalt/release_inputs/euroc_ds_calib.json \
      --config configs/basalt/euroc_config.json --out-dir "$d" $f --no-trace > "$d.log" 2>&1
    echo "$m $s $(shasum -a 256 "$d/trajectory.tum" | cut -d' ' -f1)"
  done
done
