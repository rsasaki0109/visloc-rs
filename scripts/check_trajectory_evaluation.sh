#!/usr/bin/env sh
set -eu

if ! command -v cargo >/dev/null 2>&1 && [ -d "$HOME/.cargo/bin" ]; then
  PATH="$HOME/.cargo/bin:$PATH"
  export PATH
fi

output_root="target/visloc_trajectory_eval_check"
rm -rf "$output_root"
mkdir -p "$output_root"

cargo run --example evaluate_trajectory_from_kitti_files -- \
  --out-dir "$output_root/kitti" \
  --max-mean 0.4 \
  --max-rmse 0.7 \
  --max-max 1.4 \
  --min-matched 4 \
  --min-match-ratio 1.0

cargo run --example evaluate_trajectory_from_tum_files -- \
  --out-dir "$output_root/tum" \
  --max-mean 0.06 \
  --max-rmse 0.08 \
  --max-max 0.11 \
  --min-matched 3 \
  --min-match-ratio 0.75

cargo run --example evaluate_kitti_odometry_benchmark -- \
  --lengths 100 \
  --out-dir "$output_root/kitti_odometry"

for dataset in kitti tum; do
  test -s "$output_root/$dataset/translation_errors.csv"
  test -s "$output_root/$dataset/error_summary.json"
  test -s "$output_root/$dataset/evaluation_result.json"
  test -s "$output_root/$dataset/trajectory_report.html"
  grep -q '"passed": true' "$output_root/$dataset/evaluation_result.json"
  grep -q '"failures": \[\]' "$output_root/$dataset/evaluation_result.json"
done

test -s "$output_root/kitti_odometry/kitti_odometry_segments.csv"
test -s "$output_root/kitti_odometry/kitti_odometry_summary.json"
grep -q '"segment_count": 3' "$output_root/kitti_odometry/kitti_odometry_summary.json"
grep -q '"mean_translational_error_percent": 1' "$output_root/kitti_odometry/kitti_odometry_summary.json"
