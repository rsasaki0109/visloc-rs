#!/usr/bin/env bash
# Phase-26 #4 — structural recovery PnP rework sweep.
#
# Phase-26 #2 / #2b showed that enabling Phase-23 #1 recovery PnP on
# top of the Phase-26 #1 SuperPoint+strict-stereo config produces:
#   - 4 of 6 cases: 0 recoveries accepted out of 1300+ attempts
#     (strict gate impossible even with SuperPoint)
#   - 2 of 6 cases (V1_01 both thresholds): 2-4 false-positive
#     recoveries accepted, scale collapses 1.026 → 0.27, rigid ATE
#     explodes 0.0029 → 0.38 m (Phase-26 #1's V-class win destroyed)
#
# The diagnosis: SuperPoint descriptors can reach the inlier gate on
# the easiest cliff regime (V1_01), but the recovered solution is at
# the wrong global scale because the full-map candidate landmark set
# admits geometrically self-consistent recoveries far from the true
# pose. Pose-prior radius (Phase-26 #2b) did not help because the IMU
# prediction at recovery time is itself drifted.
#
# Phase-26 #4 ships two structural fixes:
#   #4a — active-frontier submap selection: restrict the recovery
#         descriptor store to landmarks observed by the most recent
#         N keyframes. Targets the "full-map admits wrong-scale" mode
#         by excluding stale landmarks.
#   #4b — post-acceptance IMU sanity check: reject recoveries whose
#         recovered camera centre is more than M meters from the
#         tracker's per-frame motion-model prediction. Targets the
#         "geometrically self-consistent but drift-incompatible"
#         false positives.
#
# This sweep enables both fixes on top of Phase-26 #1 SuperPoint and
# runs 6 = 3 seqs × 2 thresholds.
#
# Parameter rationale:
#   recent_keyframe_window=5 — modest window covering the active
#     map frontier (the cliff lives at frame ~113 in V-class, and
#     the few keyframes immediately before the cliff are the
#     genuinely-co-visible candidates).
#   max_translation_from_imu_prediction_meters=2.0 — V-class hover
#     at ~0.5 m/s × cliff-recovery delay of a few seconds = 1-2.5 m
#     expected IMU drift. 2.0 m comfortably admits correct recoveries
#     (within ~10 cm of prediction) while rejecting scale-wrong
#     recoveries (Phase-26 #2 V1_01 false positives landed at scale
#     0.27, i.e. trajectory shrunk 3.7× → recovered positions
#     differ from prediction by metres-to-tens-of-metres).
#
# Prerequisites: cam0+cam1 SuperPoint pre-exports at
# `target/euroc_phase26_superpoint/<seq>/cam{0,1}/`.
#
# Usage:
#   scripts/run_euroc_phase26_4_structural_recovery_sweep.sh [SEQ ...]

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
  --relocalization-recent-keyframe-window 5
  --relocalization-max-translation-from-imu-prediction-meters 2.0
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
    out_dir="target/euroc_phase26_4_${seq}_${thr_suffix}_superpoint_reloc_structural"
    log_path="${out_dir}.log"
    echo "=== launching ${seq} ${thr_suffix} SuperPoint+reloc+structural (f=${fail_thr}, s=${succ_thr}) → ${out_dir} ==="
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
echo "=== all SuperPoint+reloc+structural runs complete ==="
