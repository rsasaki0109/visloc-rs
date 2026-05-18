#!/usr/bin/env bash
# Phase-26 #1 — SuperPoint+strict-stereo bootstrap re-test on top of
# the Phase-25 default stack.
#
# Phase-15 (2026-04-xx) shipped a `SuperPointOfflineExtractor` that
# replays pre-exported `frame_NNNNNN_features.txt` mono-feature files
# (one per cam0 frame), and concluded "descriptor strength is NOT the
# binding constraint at this stack — the 4 m fixed bootstrap depth
# was." That finding was correct for the Phase-13 mixed-depth
# bootstrap. After Phase-23 #2 (strict-stereo) dropped the wrong
# 4 m fallback landmarks, the SuperPoint replay needs to be re-tested
# on top of the Phase-25 default config to see whether the
# cliff-region descriptor mismatch is now the binding constraint.
#
# This sweep runs SuperPoint cam0+cam1 strict-stereo bootstrap on the
# Phase-25 recommended config, compared against the Phase-25 HOG
# baseline. Per-sequence per-threshold variants are run in parallel.
#
# Prerequisites (per sequence, e.g. V2_01_easy):
#   target/euroc_phase26_superpoint/V2_01_easy/cam0/frame_*_features.txt
#   target/euroc_phase26_superpoint/V2_01_easy/cam1/frame_*_features.txt
# Produced by:
#   python3 scripts/export_superpoint_lightglue.py --mono-dir \
#     /…/V2_01_easy/mav0/cam{0,1}/data --out-dir \
#     target/euroc_phase26_superpoint/V2_01_easy/cam{0,1} \
#     --frames 1500 --max-keypoints 1500 --device cuda
#
# Usage:
#   scripts/run_euroc_phase26_superpoint_strict_stereo.sh [SEQ...]
#
# Defaults to the V2_01_easy fast-signal target (the seq Phase-25's
# ThreePoseSmoother won -25 % on, so the easiest place to detect a
# further descriptor-layer lift). Pass additional sequence labels as
# args to extend.
#
# Override EUROC env var to point at a different copy of the dataset.

set -euo pipefail

EUROC="${EUROC:-old_~2026/simple_visual_slam/datasets/euroc}"
BIN="./target/release/examples/euroc_online_slam_vi_image_demo"
SP_DIR="target/euroc_phase26_superpoint"

if [ "$#" -eq 0 ]; then
  declare -a SEQS=("V2_01_easy")
else
  declare -a SEQS=("$@")
fi

# (label_suffix, failures_to_switch_to_pose, successes_to_switch_to_imu)
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
)

for seq in "${SEQS[@]}"; do
  cam0_dir="${SP_DIR}/${seq}/cam0"
  cam1_dir="${SP_DIR}/${seq}/cam1"
  if [ ! -d "$cam0_dir" ] || [ ! -d "$cam1_dir" ]; then
    echo "ERROR: missing pre-exported SuperPoint dirs for ${seq}: ${cam0_dir} or ${cam1_dir}" >&2
    echo "Run scripts/export_superpoint_lightglue.py --mono-dir first (see header)." >&2
    exit 2
  fi
  for thr_spec in "${THRESHOLDS[@]}"; do
    IFS='|' read -r thr_suffix fail_thr succ_thr <<< "$thr_spec"
    out_dir="target/euroc_phase26_${seq}_${thr_suffix}_superpoint"
    log_path="${out_dir}.log"
    echo "=== launching ${seq} ${thr_suffix} SuperPoint (f=${fail_thr}, s=${succ_thr}) → ${out_dir} ==="
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
echo "=== all SuperPoint runs complete ==="
