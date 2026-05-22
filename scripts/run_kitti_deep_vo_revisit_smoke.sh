#!/usr/bin/env sh
set -eu

start_dir="${KITTI_REVISIT_START_DIR:-$HOME/datasets/kitti_seq00_start_50}"
revisit_dir="${KITTI_REVISIT_DIR:-$HOME/datasets/kitti_seq00_revisit_4500}"
out_dir="${KITTI_REVISIT_OUT_DIR:-target/kitti_revisit_deep_smoke}"
start_frames="${KITTI_REVISIT_START_FRAMES:-50}"
revisit_start_frame="${KITTI_REVISIT_START_FRAME:-4500}"
revisit_frames="${KITTI_REVISIT_FRAMES:-30}"
workers="${KITTI_REVISIT_WORKERS:-8}"
frontend="${KITTI_REVISIT_FRONTEND:-deep}"
max_features="${KITTI_REVISIT_MAX_FEATURES:-200}"
min_matches="${KITTI_REVISIT_MIN_MATCHES:-30}"
min_inliers="${KITTI_REVISIT_MIN_INLIERS:-12}"
min_inlier_ratio="${KITTI_REVISIT_MIN_INLIER_RATIO:-0.4}"
max_mean_sampson_error="${KITTI_REVISIT_MAX_MEAN_SAMPSON_ERROR:-0.005}"
readme_asset_out="${KITTI_REVISIT_README_ASSET_OUT:-}"
expect_min_candidates="${KITTI_REVISIT_EXPECT_MIN_CANDIDATES:-}"
expect_strongest_from="${KITTI_REVISIT_EXPECT_STRONGEST_FROM:-}"
expect_strongest_to="${KITTI_REVISIT_EXPECT_STRONGEST_TO:-}"
expect_min_strongest_inliers="${KITTI_REVISIT_EXPECT_MIN_STRONGEST_INLIERS:-}"
expect_min_strongest_ratio="${KITTI_REVISIT_EXPECT_MIN_STRONGEST_RATIO:-}"
readme_headline_gate=0
skip_fetch=0

usage() {
  cat <<'EOF'
usage: scripts/run_kitti_deep_vo_revisit_smoke.sh [options]

Fetch/reuse KITTI 00 start and revisit image_0 subsets, run the appearance
loop scanner, and write a compact revisit-smoke summary.

Options:
  --start-dir <dir>          Start segment directory
  --revisit-dir <dir>        Revisit segment directory
  --out-dir <dir>            Output directory
  --start-frames <n>         Number of start frames, default 50
  --revisit-start-frame <n>  First revisit KITTI frame id, default 4500
  --revisit-frames <n>       Number of revisit frames, default 30
  --workers <n>              Parallel download workers
  --frontend <name>          classical|deep|deep-ms|both, default deep
  --max-features <n>         Feature cap per frame, default 200
  --min-matches <n>          Scanner raw-match threshold, default 30
  --min-inliers <n>          Verifier inlier threshold, default 12
  --min-inlier-ratio <x>     Verifier inlier-ratio threshold, default 0.4
  --max-mean-sampson-error <x>
                             Verifier Sampson threshold, default 0.005
  --readme-asset-out <jpg>   Optional README JPEG to render from the report
  --readme-headline-gate     Apply README headline expectations for quick run
  --expect-min-candidates <n>
                             Require at least n accepted candidates
  --expect-strongest-from <n>
                             Require strongest pair source frame
  --expect-strongest-to <n>  Require strongest pair target frame
  --expect-min-strongest-inliers <n>
                             Require strongest pair inlier floor
  --expect-min-strongest-ratio <x>
                             Require strongest pair ratio floor
  --skip-fetch               Reuse already-fetched subsets
  -h, --help                 Show this help

Environment variables with the KITTI_REVISIT_* prefix mirror the options.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --start-dir)
      start_dir="$2"
      shift 2
      ;;
    --revisit-dir)
      revisit_dir="$2"
      shift 2
      ;;
    --out-dir)
      out_dir="$2"
      shift 2
      ;;
    --start-frames)
      start_frames="$2"
      shift 2
      ;;
    --revisit-start-frame)
      revisit_start_frame="$2"
      shift 2
      ;;
    --revisit-frames)
      revisit_frames="$2"
      shift 2
      ;;
    --workers)
      workers="$2"
      shift 2
      ;;
    --frontend)
      frontend="$2"
      shift 2
      ;;
    --max-features)
      max_features="$2"
      shift 2
      ;;
    --min-matches)
      min_matches="$2"
      shift 2
      ;;
    --min-inliers)
      min_inliers="$2"
      shift 2
      ;;
    --min-inlier-ratio)
      min_inlier_ratio="$2"
      shift 2
      ;;
    --max-mean-sampson-error)
      max_mean_sampson_error="$2"
      shift 2
      ;;
    --readme-asset-out)
      readme_asset_out="$2"
      shift 2
      ;;
    --readme-headline-gate)
      readme_headline_gate=1
      shift
      ;;
    --expect-min-candidates)
      expect_min_candidates="$2"
      shift 2
      ;;
    --expect-strongest-from)
      expect_strongest_from="$2"
      shift 2
      ;;
    --expect-strongest-to)
      expect_strongest_to="$2"
      shift 2
      ;;
    --expect-min-strongest-inliers)
      expect_min_strongest_inliers="$2"
      shift 2
      ;;
    --expect-min-strongest-ratio)
      expect_min_strongest_ratio="$2"
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

if [ "$readme_headline_gate" -eq 1 ]; then
  expect_min_candidates="${expect_min_candidates:-41}"
  expect_strongest_from="${expect_strongest_from:-49}"
  expect_strongest_to="${expect_strongest_to:-4501}"
  expect_min_strongest_inliers="${expect_min_strongest_inliers:-57}"
  expect_min_strongest_ratio="${expect_min_strongest_ratio:-0.6}"
fi

if [ "$skip_fetch" -eq 0 ]; then
  python3 scripts/fetch_kitti_seq00_images.py \
    --stride 1 \
    --max-frames "$start_frames" \
    --workers "$workers" \
    --skip-existing \
    --cameras image_0 \
    --out-dir "$start_dir"

  python3 scripts/fetch_kitti_seq00_images.py \
    --stride 1 \
    --start-frame "$revisit_start_frame" \
    --max-frames "$revisit_frames" \
    --workers "$workers" \
    --skip-existing \
    --cameras image_0 \
    --out-dir "$revisit_dir"
fi

test -d "$start_dir/image_0"
test -d "$revisit_dir/image_0"
test -s "$start_dir/calib.txt"
test -s "$revisit_dir/calib.txt"

rm -rf "$out_dir"
mkdir -p "$out_dir"

cargo run --release --features image-io \
  --example kitti_revisit_scanner_demo -- \
  --segment-a "$start_dir/image_0" \
  --calib-a "$start_dir/calib.txt" \
  --segment-b "$revisit_dir/image_0" \
  --calib-b "$revisit_dir/calib.txt" \
  --projection P0 \
  --frontend "$frontend" \
  --max-features "$max_features" \
  --min-matches "$min_matches" \
  --min-inliers "$min_inliers" \
  --min-inlier-ratio "$min_inlier_ratio" \
  --max-mean-sampson-error "$max_mean_sampson_error" \
  --out-dir "$out_dir"

test -s "$out_dir/summary.txt"
test -s "$out_dir/candidates.csv"
test -s "$out_dir/index.html"

if ! grep -q "strongest_from=" "$out_dir/summary.txt"; then
  echo "no strongest loop pair found in $out_dir/summary.txt" >&2
  exit 1
fi

python3 - "$out_dir/candidates.csv" \
  "$expect_min_candidates" \
  "$expect_strongest_from" \
  "$expect_strongest_to" \
  "$expect_min_strongest_inliers" \
  "$expect_min_strongest_ratio" <<'PY'
import csv
import sys

csv_path, min_candidates, strongest_from, strongest_to, min_inliers, min_ratio = sys.argv[1:]
with open(csv_path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise SystemExit("candidates.csv has no accepted candidates")
strongest = max(rows, key=lambda row: float(row["score"]))
failures = []
if min_candidates and len(rows) < int(min_candidates):
    failures.append(f"expected at least {min_candidates} candidates, got {len(rows)}")
if strongest_from and int(strongest["matched_keyframe_id"]) != int(strongest_from):
    failures.append(
        f"expected strongest_from={strongest_from}, got {strongest['matched_keyframe_id']}"
    )
if strongest_to and int(strongest["query_frame_id"]) != int(strongest_to):
    failures.append(f"expected strongest_to={strongest_to}, got {strongest['query_frame_id']}")
if min_inliers and int(strongest["inliers"]) < int(min_inliers):
    failures.append(f"expected strongest inliers >= {min_inliers}, got {strongest['inliers']}")
if min_ratio and float(strongest["inlier_ratio"]) < float(min_ratio):
    failures.append(
        f"expected strongest ratio >= {min_ratio}, got {float(strongest['inlier_ratio']):.6f}"
    )
if failures:
    raise SystemExit("KITTI revisit expectation check failed:\n  - " + "\n  - ".join(failures))
PY

if [ -n "$readme_asset_out" ]; then
  python3 scripts/render_kitti_revisit_report_asset.py "$out_dir" --out "$readme_asset_out"
  test -s "$readme_asset_out"
fi

summary_file="$out_dir/deep_revisit_smoke_summary.txt"
{
  echo "# KITTI deep revisit scanner smoke summary"
  echo "start_dir=$start_dir"
  echo "revisit_dir=$revisit_dir"
  echo "out_dir=$out_dir"
  echo "start_frames=$start_frames"
  echo "revisit_start_frame=$revisit_start_frame"
  echo "revisit_frames=$revisit_frames"
  echo "frontend=$frontend"
  echo "max_features=$max_features"
  echo "min_matches=$min_matches"
  echo "min_inliers=$min_inliers"
  echo "min_inlier_ratio=$min_inlier_ratio"
  echo "max_mean_sampson_error=$max_mean_sampson_error"
  echo "report=$out_dir/index.html"
  echo "candidates_csv=$out_dir/candidates.csv"
  if [ -n "$readme_asset_out" ]; then
    echo "readme_asset=$readme_asset_out"
  fi
  if [ "$readme_headline_gate" -eq 1 ]; then
    echo "readme_headline_gate=true"
  fi
  if [ -n "$expect_min_candidates" ]; then
    echo "expect_min_candidates=$expect_min_candidates"
  fi
  if [ -n "$expect_strongest_from" ]; then
    echo "expect_strongest_from=$expect_strongest_from"
  fi
  if [ -n "$expect_strongest_to" ]; then
    echo "expect_strongest_to=$expect_strongest_to"
  fi
  if [ -n "$expect_min_strongest_inliers" ]; then
    echo "expect_min_strongest_inliers=$expect_min_strongest_inliers"
  fi
  if [ -n "$expect_min_strongest_ratio" ]; then
    echo "expect_min_strongest_ratio=$expect_min_strongest_ratio"
  fi
  echo
  cat "$out_dir/summary.txt"
} > "$summary_file"

echo "# Wrote $summary_file"
sed -n '1,120p' "$summary_file"
