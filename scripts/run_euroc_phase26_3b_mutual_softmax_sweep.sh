#!/usr/bin/env bash
# Phase-26 #3b — SuperPoint + MutualSoftmaxMatcher sweep.
#
# Phase-26 #1 (SuperPoint + cross-check matcher) lifted V-class
# accuracy by an order of magnitude but the trajectory died early at
# the universal cliff (V1_01 frame 113). Phase-26 #3a empirically
# refuted the "loosen the PnP gate" intervention: looser gate
# extends trajectories but at scale-wrong solutions.
#
# This sweep tests the alternative: replace cross-check with
# `MutualSoftmaxMatcher` (LightGlue-style temperature-scaled
# mutual-softmax over the full cosine-similarity matrix). The
# hypothesis: mutual-softmax admits *correct* additional
# cross-attitude correspondences that cross-check rejects, lifting
# the cliff-region inlier honestly (instead of by loosening the
# gate on noisy matches).
#
# Prerequisites: cam0+cam1 SuperPoint pre-exports at
# `target/euroc_phase26_superpoint/<seq>/cam{0,1}/`.
#
# Scope: V-class only by default (V1_01, V2_01). Pass MH_01_easy as
# extra arg to extend.
#
# Usage:
#   scripts/run_euroc_phase26_3b_mutual_softmax_sweep.sh [SEQ ...]

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

# Drop --cross-check-matcher (incompatible with --mutual-softmax-matcher);
# everything else matches the Phase-26 #1 V-class accuracy config.
shared_flags=(
  --max-frames 1500
  --gravity 0,0,-9.81
  --keyframe-min-translation 0.1 --max-pose-jump-meters 0.2
  --motion-model adaptive-imu-pose --pnp-pose-prior-warm-start
  --vi-init-gyro-std-limit 0.5 --vi-init-accel-std-limit 5.0
  --vi-init-try-initialize-on-every-frame
  --vi-init-min-stationary-window-seconds 1.5
  --local-vi-ba --run-local-vi-ba-at-vi-init-promotion
  --keep-pre-promotion-imu-factors
  --stereo-bootstrap-strict
  --mutual-softmax-matcher
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
    out_dir="target/euroc_phase26_3b_${seq}_${thr_suffix}_superpoint_mutsoft"
    log_path="${out_dir}.log"
    echo "=== launching ${seq} ${thr_suffix} SuperPoint+mutual-softmax (f=${fail_thr}, s=${succ_thr}) → ${out_dir} ==="
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
echo "=== all SuperPoint+mutual-softmax runs complete ==="
