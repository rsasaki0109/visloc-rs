#!/usr/bin/env bash
# Phase-25 IMU-velocity-refresh-policy A/B sweep.
#
# Phase-24 introduced an `ImuVelocityRefreshPolicy::FiniteDifference`
# hook that refreshes the wrapped IMU's `velocity_world` at every
# Pose → IMU switch using a single finite-difference of the two most
# recent successful visual poses. That hook produced a Pareto win on
# V1_01_easy but degraded MH_01_easy / V2_01_easy. The diagnosis was
# that the constant-pose branch's successive poses are themselves
# PnP-noise-dominated at cliff-region landmark counts (4-8 inliers),
# so the finite-difference reset injects PnP noise into
# velocity_world rather than producing a clean reset.
#
# Phase-25 ships two alternative refresh policies for A/B comparison:
#
#   * `zero-reset`          — overwrite velocity_world with zeros at
#                             every switch. Removes the PnP-noise
#                             injection entirely at the cost of
#                             discarding any genuine motion estimate.
#   * `three-pose-smoother` — average two finite-differences across
#                             the three most recent successful poses.
#                             Halves the PnP-noise variance compared
#                             with single finite-difference; falls
#                             back to single finite-difference when
#                             fewer than 3 poses are available.
#
# This sweep runs both new policies × 3 sequences × 2 threshold sets
# = 12 runs, matched against the Phase-24 baseline at
# `target/euroc_phase24_*/`.
#
# Goal: confirm or refute the Phase-24 diagnosis. If `zero-reset` /
# `three-pose-smoother` also fail to clear the MH_01 / V2_01
# regressions, the cliff problem is upstream of the motion-model
# layer and the next-thread direction is SuperPoint+LightGlue
# descriptors (Phase-23 #1 follow-up).
#
# Usage:
#     scripts/run_euroc_phase25_refresh_policy_ab.sh
#
# Override EUROC env var to point at a different copy of the dataset.

set -euo pipefail

EUROC="${EUROC:-/media/sasaki/aiueo/ai_coding_ws/old_~2026/simple_visual_slam/datasets/euroc}"
BIN="./target/release/examples/euroc_online_slam_vi_image_demo"

declare -a SEQS=("MH_01_easy" "V1_01_easy" "V2_01_easy")

# (label_suffix, failures_to_switch_to_pose, successes_to_switch_to_imu)
declare -a THRESHOLDS=(
  "strict|2|5"
  "strict_imuFavor|3|10"
)

declare -a POLICIES=(
  "zero-reset"
  "three-pose-smoother"
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
  for thr_spec in "${THRESHOLDS[@]}"; do
    IFS='|' read -r thr_suffix fail_thr succ_thr <<< "$thr_spec"
    for policy in "${POLICIES[@]}"; do
      out_dir="target/euroc_phase25_${seq}_${thr_suffix}_${policy//-/_}"
      log_path="${out_dir}.log"
      echo "=== launching ${seq} ${thr_suffix} policy=${policy} (f=${fail_thr}, s=${succ_thr}) → ${out_dir} ==="
      "$BIN" \
        --euroc-dir "${EUROC}/${seq}" \
        --out-dir "${out_dir}" \
        "${shared_flags[@]}" \
        --adaptive-motion-failures-to-switch-to-pose "${fail_thr}" \
        --adaptive-motion-successes-to-switch-to-imu "${succ_thr}" \
        --adaptive-motion-refresh-policy "${policy}" \
        > "${log_path}" 2>&1 &
    done
  done
done

wait
echo "=== all 12 runs complete ==="
