#!/usr/bin/env bash
# Phase-24 IMU-velocity-refresh-on-switch sweep.
#
# Re-runs the Phase-23 #4 adaptive 3-seq × 2-threshold sweep with the
# Phase-24 refresh-on-switch hook ENABLED (the new default). All six
# runs go to target/euroc_phase24_*/. Compare against Phase-23 #4
# artifacts at target/euroc_phase23_{seq}_adaptive_strict{,_imuFavor}/.
#
# Goal: validate that the V1_01 win generalizes — MH_01 and V2_01
# previously oscillated because the IMU's `velocity_world` went
# stale during pose-mode intervals. Phase-24 refreshes it from a
# finite-difference of the last two successful visual poses at every
# switch-back; this should turn the V1_01-shape Pareto win into a
# 3-seq universal win.
#
# Usage:
#     scripts/run_euroc_phase24_adaptive_refresh.sh
#
# Override EUROC env var to point at a different copy of the dataset.

set -euo pipefail

EUROC="${EUROC:-old_~2026/simple_visual_slam/datasets/euroc}"
BIN="./target/release/examples/euroc_online_slam_vi_image_demo"

declare -a SEQS=("MH_01_easy" "V1_01_easy" "V2_01_easy")

# (label_suffix, failures_to_switch_to_pose, successes_to_switch_to_imu)
declare -a VARIANTS=(
  "adaptive_refresh_strict|2|5"
  "adaptive_refresh_strict_imuFavor|3|10"
)

shared_flags=(
  --max-frames 1500
  --gravity 0,0,-9.81
  --feature-extractor hog --cross-check-matcher
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
  for variant_spec in "${VARIANTS[@]}"; do
    IFS='|' read -r suffix fail_thr succ_thr <<< "$variant_spec"
    out_dir="target/euroc_phase24_${seq}_${suffix}"
    log_path="${out_dir}.log"
    echo "=== launching ${seq} ${suffix} (f=${fail_thr}, s=${succ_thr}) → ${out_dir} ==="
    "$BIN" \
      --euroc-dir "${EUROC}/${seq}" \
      --out-dir "${out_dir}" \
      "${shared_flags[@]}" \
      --adaptive-motion-failures-to-switch-to-pose "${fail_thr}" \
      --adaptive-motion-successes-to-switch-to-imu "${succ_thr}" \
      > "${log_path}" 2>&1 &
  done
done

wait
echo "=== all 6 runs complete ==="
