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
  --out-dir "$out_dir"

test -s "$out_dir/summary.txt"

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
  echo
  cat "$out_dir/summary.txt"
} > "$summary_file"

if ! grep -q "strongest_from=" "$out_dir/summary.txt"; then
  echo "no strongest loop pair found in $out_dir/summary.txt" >&2
  exit 1
fi

echo "# Wrote $summary_file"
sed -n '1,120p' "$summary_file"
