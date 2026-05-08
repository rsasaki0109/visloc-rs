#!/usr/bin/env sh
set -eu

output_dir="target/visloc_timestamped_image_sequence_demo"
rm -rf "$output_dir"

cargo run --features image-io --example track_timestamped_image_sequence_with_gnss_prior

expected_files="
0000.png
0001.png
0002.png
timestamps_ns.txt
gnss_world.txt
gnss_sync_evaluation.json
"

for file in $expected_files; do
    test -s "$output_dir/$file"
done

grep -q '^0$' "$output_dir/timestamps_ns.txt"
grep -q '^100000000$' "$output_dir/timestamps_ns.txt"
grep -q '^200000000$' "$output_dir/timestamps_ns.txt"
grep -q '100000000 0.0 0.0 0.0 10.0 10.0' "$output_dir/gnss_world.txt"
grep -q '"passed": true' "$output_dir/gnss_sync_evaluation.json"
grep -q '"frame_count": 3' "$output_dir/gnss_sync_evaluation.json"
grep -q '"measurement_count": 3' "$output_dir/gnss_sync_evaluation.json"
grep -q '"matched_frame_count": 3' "$output_dir/gnss_sync_evaluation.json"
grep -q '"missing_measurement_count": 0' "$output_dir/gnss_sync_evaluation.json"
grep -q '"matched_frame_ratio": 1' "$output_dir/gnss_sync_evaluation.json"
grep -q '"failures": \[\]' "$output_dir/gnss_sync_evaluation.json"
