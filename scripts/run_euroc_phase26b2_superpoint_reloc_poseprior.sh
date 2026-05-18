#!/usr/bin/env bash
# Phase-26 #2b — SuperPoint + relocalization + pose-prior-radius.
#
# Phase-26 #2 showed that turning Phase-23 #1 relocalization on top of
# SuperPoint+strict-stereo produced an honest mixed result: MH_01 and
# V2_01 stayed bit-identical to Phase-26 #1 (the strict gate rejected
# all ~1300+ recovery attempts), but V1_01 accepted 2-3 false-positive
# recoveries that collapsed sim_scale to ~0.27 and exploded rigid ATE
# from 0.0029 m to 0.378 m. Diagnosis: SuperPoint descriptors lift
# match quality enough for cliff-region cross-attitude solutions to
# pass the inlier-ratio gate, but the recovery PnP's full-map
# candidate landmark set admits geometrically-consistent solutions at
# the wrong global scale.
#
# This sweep adds Phase-23 #1b's `--relocalization-pose-prior-radius`
# to filter candidate landmarks to a metric neighborhood of the IMU's
# motion-model prediction. The hypothesis: with SuperPoint descriptors
# AND a pose-prior radius, the recovery PnP can finally find
# *correct* recoveries — V1_01 false positives get filtered out, and
# MH_01 / V2_01 may finally land their first valid recoveries because
# the candidate set is smaller and the matching task easier.
#
# Radius = 5 m is the natural middle ground between Phase-23 #1b's
# 2 m (excluded all landmarks because IMU prediction was off by
# |g·Δt|) and 10 m (admitted MH_01 false positive that collapsed
# scale to 0.000233).
#
# Prerequisites: cam0+cam1 SuperPoint pre-export at
# `target/euroc_phase26_superpoint/<seq>/cam{0,1}/`.
#
# Usage:
#   scripts/run_euroc_phase26b2_superpoint_reloc_poseprior.sh [SEQ ...]

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
  --relocalization-pose-prior-radius 5.0
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
    out_dir="target/euroc_phase26b2_${seq}_${thr_suffix}_superpoint_reloc_pp5m"
    log_path="${out_dir}.log"
    echo "=== launching ${seq} ${thr_suffix} SuperPoint+reloc+pp5m (f=${fail_thr}, s=${succ_thr}) → ${out_dir} ==="
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
echo "=== all SuperPoint+reloc+pp5m runs complete ==="
