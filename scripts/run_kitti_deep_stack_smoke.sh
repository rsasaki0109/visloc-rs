#!/usr/bin/env sh
set -eu

out_dir="${KITTI_DEEP_STACK_OUT_DIR:-target/kitti_deep_stack_smoke}"
skip_fetch=0
revisit_frontend="${KITTI_DEEP_STACK_REVISIT_FRONTEND:-deep}"

usage() {
  cat <<'EOF'
usage: scripts/run_kitti_deep_stack_smoke.sh [options]

Run the deep KITTI stack smoke: metric stereo VO evaluation plus revisit
appearance-loop scanner validation. Outputs are grouped under one directory.

Options:
  --out-dir <dir>             Output root, default target/kitti_deep_stack_smoke
  --revisit-frontend <name>   classical|deep|deep-ms|both, default deep
  --skip-fetch                Reuse already-fetched KITTI subsets
  -h, --help                  Show this help

Environment:
  KITTI_DEEP_STACK_OUT_DIR
  KITTI_DEEP_STACK_REVISIT_FRONTEND

The underlying smoke scripts also honor KITTI_DEEP_VO_* and KITTI_REVISIT_*
environment variables for dataset locations and tuning knobs.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --out-dir)
      out_dir="$2"
      shift 2
      ;;
    --revisit-frontend)
      revisit_frontend="$2"
      shift 2
      ;;
    --skip-fetch)
      skip_fetch=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$out_dir"

vo_out="$out_dir/vo"
revisit_out="$out_dir/revisit"
fetch_arg=""
if [ "$skip_fetch" -eq 1 ]; then
  fetch_arg="--skip-fetch"
fi

scripts/run_kitti_deep_vo_smoke.sh \
  $fetch_arg \
  --out-dir "$vo_out"

scripts/run_kitti_deep_vo_revisit_smoke.sh \
  $fetch_arg \
  --frontend "$revisit_frontend" \
  --out-dir "$revisit_out"

test -s "$vo_out/deep_vo_smoke_summary.txt"
test -s "$revisit_out/deep_revisit_smoke_summary.txt"

grep -q "vo_ate_mean_m=" "$vo_out/deep_vo_smoke_summary.txt"
grep -q "relative_pose_mean_t_mag_err_m=" "$vo_out/deep_vo_smoke_summary.txt"
grep -q '"segment_count"' "$vo_out/deep_vo_smoke_summary.txt"
grep -q "strongest_from=" "$revisit_out/deep_revisit_smoke_summary.txt"
grep -q "strongest_to=" "$revisit_out/deep_revisit_smoke_summary.txt"

summary_file="$out_dir/deep_stack_smoke_summary.txt"
summary_json="$out_dir/deep_stack_smoke_summary.json"

metric_value() {
  key="$1"
  file="$2"
  sed -n "s/.*$key=\([^ ]*\).*/\1/p" "$file" | head -n 1
}

json_number() {
  key="$1"
  file="$2"
  sed -n "s/.*\"$key\": \([^,}]*\).*/\1/p" "$file" | head -n 1
}

vo_summary="$vo_out/summary.txt"
public_json="$vo_out/kitti_eval_public_lengths/kitti_odometry_summary.json"
kitti_100m_json="$vo_out/kitti_eval_100m/kitti_odometry_summary.json"
revisit_summary="$revisit_out/deep_revisit_smoke_summary.txt"

vo_ate_mean_m="$(metric_value "vo_ate_mean_m" "$vo_summary")"
vo_ate_rmse_m="$(metric_value "vo_ate_rmse_m" "$vo_summary")"
vo_ate_max_m="$(metric_value "vo_ate_max_m" "$vo_summary")"
relative_pose_pairs="$(metric_value "relative_pose_pairs" "$vo_summary")"
relative_pose_mean_t_mag_err_m="$(metric_value "relative_pose_mean_t_mag_err_m" "$vo_summary")"
relative_pose_max_t_mag_err_m="$(metric_value "relative_pose_max_t_mag_err_m" "$vo_summary")"
relative_pose_mean_rot_err_deg="$(metric_value "relative_pose_mean_rot_err_deg" "$vo_summary")"
relative_pose_max_rot_err_deg="$(metric_value "relative_pose_max_rot_err_deg" "$vo_summary")"
kitti_public_segment_count="$(json_number "segment_count" "$public_json")"
kitti_public_t_rel_percent="$(json_number "mean_translational_error_percent" "$public_json")"
kitti_public_r_rel_deg_per_m="$(json_number "mean_rotational_error_deg_per_m" "$public_json")"
kitti_public_max_t_rel_percent="$(json_number "max_translational_error_percent" "$public_json")"
kitti_public_max_r_rel_deg_per_m="$(json_number "max_rotational_error_deg_per_m" "$public_json")"
kitti_100m_segment_count="$(json_number "segment_count" "$kitti_100m_json")"
kitti_100m_t_rel_percent="$(json_number "mean_translational_error_percent" "$kitti_100m_json")"
kitti_100m_r_rel_deg_per_m="$(json_number "mean_rotational_error_deg_per_m" "$kitti_100m_json")"
kitti_100m_max_t_rel_percent="$(json_number "max_translational_error_percent" "$kitti_100m_json")"
kitti_100m_max_r_rel_deg_per_m="$(json_number "max_rotational_error_deg_per_m" "$kitti_100m_json")"
revisit_candidates="$(metric_value "candidates" "$revisit_summary")"
revisit_strongest_from="$(metric_value "strongest_from" "$revisit_summary")"
revisit_strongest_to="$(metric_value "strongest_to" "$revisit_summary")"
revisit_strongest_score="$(metric_value "strongest_score" "$revisit_summary")"

{
  echo "# KITTI deep stack smoke summary"
  echo "out_dir=$out_dir"
  echo "revisit_frontend=$revisit_frontend"
  echo
  echo "## Deep Stereo VO"
  cat "$vo_out/deep_vo_smoke_summary.txt"
  echo
  echo "## Revisit Loop Scanner"
  cat "$revisit_out/deep_revisit_smoke_summary.txt"
} > "$summary_file"

cat > "$summary_json" <<EOF
{
  "vo_ate_mean_m": $vo_ate_mean_m,
  "vo_ate_rmse_m": $vo_ate_rmse_m,
  "vo_ate_max_m": $vo_ate_max_m,
  "relative_pose_pairs": $relative_pose_pairs,
  "relative_pose_mean_t_mag_err_m": $relative_pose_mean_t_mag_err_m,
  "relative_pose_max_t_mag_err_m": $relative_pose_max_t_mag_err_m,
  "relative_pose_mean_rot_err_deg": $relative_pose_mean_rot_err_deg,
  "relative_pose_max_rot_err_deg": $relative_pose_max_rot_err_deg,
  "kitti_public_segment_count": $kitti_public_segment_count,
  "kitti_public_t_rel_percent": $kitti_public_t_rel_percent,
  "kitti_public_r_rel_deg_per_m": $kitti_public_r_rel_deg_per_m,
  "kitti_public_max_t_rel_percent": $kitti_public_max_t_rel_percent,
  "kitti_public_max_r_rel_deg_per_m": $kitti_public_max_r_rel_deg_per_m,
  "kitti_100m_segment_count": $kitti_100m_segment_count,
  "kitti_100m_t_rel_percent": $kitti_100m_t_rel_percent,
  "kitti_100m_r_rel_deg_per_m": $kitti_100m_r_rel_deg_per_m,
  "kitti_100m_max_t_rel_percent": $kitti_100m_max_t_rel_percent,
  "kitti_100m_max_r_rel_deg_per_m": $kitti_100m_max_r_rel_deg_per_m,
  "revisit_frontend": "$revisit_frontend",
  "revisit_candidates": $revisit_candidates,
  "revisit_strongest_from": $revisit_strongest_from,
  "revisit_strongest_to": $revisit_strongest_to,
  "revisit_strongest_score": $revisit_strongest_score
}
EOF

grep -q '"vo_ate_mean_m"' "$summary_json"
grep -q '"relative_pose_mean_t_mag_err_m"' "$summary_json"
grep -q '"kitti_public_segment_count"' "$summary_json"
grep -q '"kitti_public_max_t_rel_percent"' "$summary_json"
grep -q '"revisit_strongest_from"' "$summary_json"

echo "# Wrote $summary_file"
echo "# Wrote $summary_json"
sed -n '1,120p' "$summary_file"
