#!/usr/bin/env sh
set -eu

output_dir="target/visloc_kitti_image_sequence_demo"
rm -rf "$output_dir"
mkdir -p "$output_dir"

cargo run --features image-io --example load_kitti_image_sequence > "$output_dir/output.log"

expected_files="
image_2/000000.png
image_2/000001.png
image_2/000002.png
times_ns.txt
calib.txt
output.log
"

for file in $expected_files; do
    test -s "$output_dir/$file"
done

grep -q '^0$' "$output_dir/times_ns.txt"
grep -q '^100000000$' "$output_dir/times_ns.txt"
grep -q '^200000000$' "$output_dir/times_ns.txt"
grep -q '^P2: 710 0 32 0 0 705 24 0 0 0 1 0$' "$output_dir/calib.txt"

grep -q 'camera id=2 size=64x48 intrinsics=Some((710.0, 705.0, 32.0, 24.0))' "$output_dir/output.log"
grep -q 'frames=3 timestamps=3 timestamp_valid=true dimension_issues=0 timestamp_issues=0' "$output_dir/output.log"
grep -q 'frame=0 timestamp_ns=Some(0)' "$output_dir/output.log"
grep -q 'frame=1 timestamp_ns=Some(100000000)' "$output_dir/output.log"
grep -q 'frame=2 timestamp_ns=Some(200000000)' "$output_dir/output.log"
