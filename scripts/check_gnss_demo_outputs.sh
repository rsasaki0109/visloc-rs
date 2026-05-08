#!/usr/bin/env sh
set -eu

output_dir="target/visloc_gnss_tracking_demo_check"
rm -rf "$output_dir"

cargo run --example track_sequence_with_gnss_prior -- --out-dir "$output_dir"

expected_files="
index.html
manifest.json
tracking.csv
tracking_summary.json
tracking_evaluation.json
tracking_report.html
trajectory.csv
poses.txt
trajectory_tum.txt
reference_poses.txt
reference_tum.txt
translation_errors.csv
error_summary.json
trajectory_summary.json
trajectory_report.html
trajectory_evaluation.html
"

for file in $expected_files; do
    test -s "$output_dir/$file"
done

grep -q '"demo": "track_sequence_with_gnss_prior"' "$output_dir/manifest.json"
grep -q '"frame_count": 3' "$output_dir/manifest.json"
grep -q '"external_localization_prior_used_count": 3' "$output_dir/manifest.json"
grep -q '"matched_reference_pose_count": 3' "$output_dir/manifest.json"
grep -q 'trajectory_evaluation.html' "$output_dir/index.html"
grep -q 'tracking_evaluation.json' "$output_dir/index.html"
grep -q 'manifest.json' "$output_dir/index.html"
grep -q 'Mean translation error' "$output_dir/index.html"
grep -q 'frame_id,translation_error' "$output_dir/translation_errors.csv"
grep -q '"matched_pose_count": 3' "$output_dir/error_summary.json"
grep -q '"passed": true' "$output_dir/tracking_evaluation.json"
grep -q '"min_success_rate": 1' "$output_dir/tracking_evaluation.json"
grep -q '"failures": \[\]' "$output_dir/tracking_evaluation.json"
