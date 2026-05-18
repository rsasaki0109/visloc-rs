#!/usr/bin/env bash
# Phase-26 #2 — SuperPoint + Phase-23 #1 relocalization re-evaluation.
#
# Phase-23 #1 (relocalization-on-tracker-death) shipped the recovery
# PnP infrastructure with strict default gates (min_inliers=20 /
# min_inlier_ratio=0.3 / max_reprojection_error=8.0). Under HOG those
# gates accepted very few recoveries on EuRoC and the few accepted ones
# REGRESSED ATE (cross-attitude HOG descriptors could not match cleanly
# enough). Phase-26 #1 then showed that SuperPoint+strict-stereo
# produces dramatically higher V-class accuracy on the pre-cliff
# window. The natural next test: do SuperPoint descriptors now lift
# recovery PnP above the strict gate so post-cliff frames become
# recoverable — i.e. can we extend the universal cliff at frame ~113
# (V-class) / 891-1069 (MH-class)?
#
# This sweep adds `--relocalization-enabled` (with the strict default
# gates) on top of the Phase-26 #1 SuperPoint+strict-stereo config.
# 6 parallel runs = 3 seqs × 2 thresholds × 1 variant.
#
# Prerequisites: cam0+cam1 SuperPoint pre-export at
# `target/euroc_phase26_superpoint/<seq>/cam{0,1}/`. Produced by
# `scripts/run_euroc_phase26_superpoint_strict_stereo.sh` prerequisites.
#
# Usage:
#   scripts/run_euroc_phase26b_superpoint_relocalization.sh [SEQ ...]
#
# Defaults to all 3 EuRoC seqs.

set -euo pipefail

EUROC="${EUROC:-old_~2026/simple_visual_slam/datasets/euroc}"
BIN="./target/release/examples/euroc_online_slam_vi_image_demo"
SP_DIR="target/euroc_phase26_superpoint"

if [ "$#" -eq 0 ]; then
  declare -a SEQS=("MH_01_easy" "V1_01_easy" "V2_01_easy")
else
  declare -a SEQS=("$@")
fi

declare -a THRESHOLDS=(
  "strict|2|5"
  "strict_imuFavor|3|10"
)

shared_flags=(
  --max-frames 1500
  --gravity 0,0,-9.81
  --cross-check-matcher
  --keyframe-min-translation 0.1 --max-pose-jump-meters 0.2
  --motion-model adaptive-imu-pose --pnp-pose-prior-warm-start
  --vi-init-gyro-std-limit 0.5 --vi-init-accel-std-limit 5.0
  --vi-init-try-initialize-on-every-frame
  --vi-init-min-stationary-window-seconds 1.5
  --local-vi-ba --run-local-vi-ba-at-vi-init-promotion
  --keep-pre-promotion-imu-factors
  --stereo-bootstrap-strict
  --relocalization-enabled
  --relocalization-min-inliers 20
  --relocalization-min-inlier-ratio 0.3
  --relocalization-max-reprojection-error 8.0
)

for seq in "${SEQS[@]}"; do
  cam0_dir="${SP_DIR}/${seq}/cam0"
  cam1_dir="${SP_DIR}/${seq}/cam1"
  if [ ! -d "$cam0_dir" ] || [ ! -d "$cam1_dir" ]; then
    echo "ERROR: missing pre-exported SuperPoint dirs for ${seq}: ${cam0_dir} or ${cam1_dir}" >&2
    exit 2
  fi
  for thr_spec in "${THRESHOLDS[@]}"; do
    IFS='|' read -r thr_suffix fail_thr succ_thr <<< "$thr_spec"
    out_dir="target/euroc_phase26b_${seq}_${thr_suffix}_superpoint_reloc"
    log_path="${out_dir}.log"
    echo "=== launching ${seq} ${thr_suffix} SuperPoint+reloc (f=${fail_thr}, s=${succ_thr}) → ${out_dir} ==="
    "$BIN" \
      --euroc-dir "${EUROC}/${seq}" \
      --out-dir "${out_dir}" \
      "${shared_flags[@]}" \
      --feature-extractor superpoint-offline \
      --superpoint-features-dir "${cam0_dir}" \
      --superpoint-cam1-features-dir "${cam1_dir}" \
      --adaptive-motion-failures-to-switch-to-pose "${fail_thr}" \
      --adaptive-motion-successes-to-switch-to-imu "${succ_thr}" \
      > "${log_path}" 2>&1 &
  done
done

wait
echo "=== all SuperPoint+reloc runs complete ==="
