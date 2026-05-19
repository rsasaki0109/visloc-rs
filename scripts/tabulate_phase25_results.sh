#!/usr/bin/env bash
# Read all Phase-25 sweep summaries plus the Phase-24 baseline and emit
# a single comparable markdown table. Standalone — invoke after
# scripts/run_euroc_phase25_refresh_policy_ab.sh has finished.
set -euo pipefail

# Order: seq | thresholds | policy | switches_pose | switches_imu
#        | refreshes | ate_rigid_m | ate_sim_m | sim_scale
header="| seq | thresh | policy | switches_pose | switches_imu | refr | ate_rigid_m | ate_sim_m | sim_scale |"
sep="|-----|--------|--------|---:|---:|---:|---:|---:|---:|"

extract_row() {
  local s="$1" seq_label="$2" thr_label="$3" policy_label="$4"
  local sp si refr rigid sim scale
  sp=$(grep -oP 'switches_to_pose=\K-?\d+' "$s" || echo "?")
  si=$(grep -oP 'switches_to_imu=\K-?\d+' "$s" || echo "?")
  refr=$(grep -oP 'velocity_refreshes_on_switch_to_imu=\K-?\d+' "$s" || echo "?")
  rigid=$(grep -oP 'ate_rigid_rmse_m=\K[0-9.]+' "$s" || echo "?")
  sim=$(grep -oP 'ate_similarity_rmse_m=\K[0-9.]+' "$s" || echo "?")
  scale=$(grep -oP 'ate_similarity_scale=\K[0-9.]+' "$s" || echo "?")
  printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" \
    "$seq_label" "$thr_label" "$policy_label" "$sp" "$si" "$refr" "$rigid" "$sim" "$scale"
}

echo "$header"
echo "$sep"

for seq in MH_01_easy V1_01_easy V2_01_easy; do
  for thr in strict strict_imuFavor; do
    # Phase-23 #4 (no refresh — baseline before Phase-24)
    s="target/euroc_phase23_${seq}_adaptive_${thr}/summary.txt"
    [ -f "$s" ] && extract_row "$s" "$seq" "$thr" "none (P-23 #4)"
    # Phase-24 (finite-diff — baseline before Phase-25)
    s="target/euroc_phase24_${seq}_adaptive_refresh_${thr}/summary.txt"
    [ -f "$s" ] && extract_row "$s" "$seq" "$thr" "finite-diff (P-24)"
    # Phase-25 new variants
    for policy in zero_reset three_pose_smoother; do
      s="target/euroc_phase25_${seq}_${thr}_${policy}/summary.txt"
      [ -f "$s" ] && extract_row "$s" "$seq" "$thr" "${policy//_/-}"
    done
  done
done
