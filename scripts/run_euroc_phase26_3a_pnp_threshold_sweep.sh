#!/usr/bin/env bash
# Phase-26 #3a — V-class SuperPoint PnP-threshold sweep.
#
# Phase-26 #1 (SuperPoint+strict-stereo) lifted V-class accuracy by
# an order of magnitude (V1_01 strict rigid ATE 0.0272 → 0.0029 m,
# V2_01 strict 0.1984 → 0.0107 m) at the cost of a slightly shorter
# pre-cliff trajectory window (V1_01 last_frame 158 → 113, V2_01
# strict 215 → 113). The trajectory shortening is the only
# unambiguous regression of Phase-26 #1; the hypothesis is that
# SuperPoint's stricter inlier gate refuses marginal post-cliff
# frames that HOG accepts at accuracy cost.
#
# This sweep tests that hypothesis by loosening the tracker's PnP
# RANSAC reprojection-error threshold from the default 4.0 px to
# 8.0 / 12.0 px. If the hypothesis is right, the looser thresholds
# should extend V-class trajectories without exploding ATE (because
# the underlying SuperPoint matches are good — they just couldn't
# squeeze enough inliers under the tight gate).
#
# Prerequisites: cam0+cam1 SuperPoint pre-exports at
# `target/euroc_phase26_superpoint/<seq>/cam{0,1}/`. Produced by the
# `scripts/run_euroc_phase26_superpoint_strict_stereo.sh` setup.
#
# Scope: V-class only (V1_01_easy, V2_01_easy). MH_01 already has
# long trajectories under Phase-26 #1 so loosening the gate there
# would only inflate ATE further; pass MH_01_easy as an arg if you
# want to test it anyway.
#
# Usage:
#   scripts/run_euroc_phase26_3a_pnp_threshold_sweep.sh [SEQ ...]

set -euo pipefail

EUROC="${EUROC:-old_~2026/simple_visual_slam/datasets/euroc}"
BIN="./target/release/examples/euroc_online_slam_vi_image_demo"
SP_DIR="target/euroc_phase26_superpoint"

if [ "$#" -eq 0 ]; then
  declare -a SEQS=("V1_01_easy" "V2_01_easy")
else
  declare -a SEQS=("$@")
fi

declare -a THRESHOLDS=(
  "strict|2|5"
  "strict_imuFavor|3|10"
)

# 4.0 px is the LocalizationConfig default (omitted from CLI ⇒
# baseline matches Phase-26 #1). 8 / 12 px are the loosened
# variants.
declare -a PNP_THRESHOLDS=("8.0" "12.0")

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
    for pnp_px in "${PNP_THRESHOLDS[@]}"; do
      label_px="${pnp_px//./}"
      out_dir="target/euroc_phase26_3a_${seq}_${thr_suffix}_superpoint_pnp${label_px}px"
      log_path="${out_dir}.log"
      echo "=== launching ${seq} ${thr_suffix} SuperPoint pnp=${pnp_px}px (f=${fail_thr}, s=${succ_thr}) → ${out_dir} ==="
      "$BIN" \
        --euroc-dir "${EUROC}/${seq}" \
        --out-dir "${out_dir}" \
        "${shared_flags[@]}" \
        --feature-extractor superpoint-offline \
        --superpoint-features-dir "${cam0_dir}" \
        --superpoint-cam1-features-dir "${cam1_dir}" \
        --pnp-reprojection-threshold-px "${pnp_px}" \
        --adaptive-motion-failures-to-switch-to-pose "${fail_thr}" \
        --adaptive-motion-successes-to-switch-to-imu "${succ_thr}" \
        > "${log_path}" 2>&1 &
    done
  done
done

wait
echo "=== all SuperPoint pnp-threshold runs complete ==="
