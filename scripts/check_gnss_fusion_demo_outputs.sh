#!/usr/bin/env sh
# Smoke check for the joint GNSS + visual-odometry fusion demo
# (docs/gnss_fusion.md): runs the synthetic metric scenario and asserts the
# fused trajectory is well below the VO-only error and that the injected
# multipath outliers were rejected.
set -eu

output_dir="target/visloc_gnss_fusion_demo_check"
rm -rf "$output_dir"

cargo run --example gnss_vo_fusion_demo -- --out-dir "$output_dir"

for file in summary.json trajectory.csv; do
    test -s "$output_dir/$file"
done

grep -q '"demo": "gnss_vo_fusion_demo"' "$output_dir/summary.json"
grep -q '"frame_count": 600' "$output_dir/summary.json"
grep -q 'timestamp_ns,truth_x' "$output_dir/trajectory.csv"
grep -q ',outlier$' "$output_dir/trajectory.csv"
grep -q ',no_fix$' "$output_dir/trajectory.csv"

json_number() {
    sed -n "s/.*\"$1\": \([-0-9.eE+]*\).*/\1/p" "$output_dir/summary.json"
}

fused=$(json_number ate_fused_m)
vo_bootstrap=$(json_number ate_vo_only_bootstrap_aligned_m)
vo_oracle=$(json_number ate_vo_only_oracle_aligned_m)
outlier_frames=$(json_number frames_with_outlier_fix)
corrupted=$(json_number corrupted_fix_count)

awk -v fused="$fused" -v vo="$vo_bootstrap" -v oracle="$vo_oracle" \
    -v flagged="$outlier_frames" -v corrupted="$corrupted" 'BEGIN {
    if (!(fused < 1.5)) { print "fused ATE too large: " fused; exit 1 }
    if (!(fused < 0.3 * vo)) { print "fused ATE " fused " not well below VO " vo; exit 1 }
    if (!(fused < oracle)) { print "fused ATE " fused " not below oracle-aligned VO " oracle; exit 1 }
    if (!(flagged >= corrupted)) { print "only " flagged " outlier frames for " corrupted " corrupted fixes"; exit 1 }
}'

echo "GNSS fusion demo OK: fused ATE ${fused} m vs VO ${vo_bootstrap} m (oracle-aligned ${vo_oracle} m)"
