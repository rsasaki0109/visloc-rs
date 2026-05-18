#!/usr/bin/env bash
# Phase-23 #1 empirical: 3-seq EuRoC × {baseline, +relocalization} A/B sweep.
#
# Baseline = the Phase-20 recommended config from
# docs/motion_based_vi_alignment.md §Phase-20 — exactly the universal-cliff
# baseline. Variant = same config + --relocalization-enabled with the demo's
# defaults (min_inliers=20, min_inlier_ratio=0.3, max_rep_err=8.0 px).
#
# Each run caps at 1500 frames (Phase-20 bench length) so the per-frame
# CSVs span the cliff region. With relocalization on, the tracker should
# survive past the cliff and write more rows.
#
# Usage:
#   bash scripts/run_euroc_phase23_relocalization_ab.sh
#
# Outputs under target/euroc_phase23_${seq}_{baseline,reloc}/

set -euo pipefail

EUROC="${EUROC:-old_~2026/simple_visual_slam/datasets/euroc}"
MAX_FRAMES="${MAX_FRAMES:-1500}"
DEMO=./target/release/examples/euroc_online_slam_vi_image_demo

if [[ ! -x "$DEMO" ]]; then
  echo "building demo..."
  cargo build --release --example euroc_online_slam_vi_image_demo --features image-io
fi

SEQS=("${SEQS:-MH_01_easy V1_01_easy V2_01_easy}")

run_one() {
  local seq="$1"
  local variant="$2"  # baseline | reloc
  local out="target/euroc_phase23_${seq}_${variant}"
  local extra=""
  if [[ "$variant" == "reloc" ]]; then
    extra="--relocalization-enabled"
  fi
  rm -rf "$out"
  echo "=== $seq | $variant ==="
  "$DEMO" \
    --euroc-dir "$EUROC/$seq" \
    --out-dir "$out" \
    --max-frames "$MAX_FRAMES" \
    --gravity 0,0,-9.81 \
    --feature-extractor hog --cross-check-matcher \
    --keyframe-min-translation 0.1 --max-pose-jump-meters 0.2 \
    --motion-model imu --pnp-pose-prior-warm-start \
    --vi-init-gyro-std-limit 0.5 --vi-init-accel-std-limit 5.0 \
    --vi-init-try-initialize-on-every-frame \
    --vi-init-min-stationary-window-seconds 1.5 \
    --local-vi-ba --run-local-vi-ba-at-vi-init-promotion \
    --keep-pre-promotion-imu-factors \
    $extra
}

for seq in $SEQS; do
  for variant in baseline reloc; do
    run_one "$seq" "$variant"
  done
done

# Aggregate
echo ""
echo "=== AGGREGATE Phase-23 #1 relocalization A/B ==="
printf "%-14s %-10s %-12s %-10s %-10s %-9s %-9s\n" \
  "seq" "variant" "success_rate" "rigid_ATE" "sim_scale" "reloc_at" "reloc_ok"
for seq in $SEQS; do
  for variant in baseline reloc; do
    f="target/euroc_phase23_${seq}_${variant}/summary.txt"
    if [[ ! -f "$f" ]]; then continue; fi
    rate=$(grep '^tracking_success_rate=' "$f" | cut -d= -f2)
    ate=$(grep '^ate_rigid_rmse_m=' "$f" | cut -d= -f2)
    scale=$(grep '^ate_similarity_scale=' "$f" | cut -d= -f2)
    reloc_at=$(grep '^relocalization_attempts=' "$f" | cut -d= -f2 || echo "0")
    reloc_ok=$(grep '^relocalization_successes=' "$f" | cut -d= -f2 || echo "0")
    printf "%-14s %-10s %-12s %-10s %-10s %-9s %-9s\n" \
      "$seq" "$variant" "$rate" "$ate" "$scale" "$reloc_at" "$reloc_ok"
  done
done
