#!/usr/bin/env sh
set -eu
# This script deliberately AVOIDS `cmd | tee log` for any command whose
# exit code matters, because POSIX sh / dash do not have pipefail and
# `tee`'s zero exit would silently mask an upstream failure (a failed
# inspect example, or a failed `ns-train splatfacto`). Inspect runs and
# ns-train below redirect to a file with `>` and then `cat` the file,
# so `set -e` catches their exit codes directly.

# KITTI -> COLMAP -> 3DGS bootstrap smoke harness.
#
# Fetches a small stride-4 KITTI subset, runs the classical stereo VO demo
# with --colmap-export and --colmap-export-binary, then loads the binary
# model back through inspect_colmap_binary_model so the round-trip writer
# <-> reader is exercised on real data, not just on synthetic unit-test
# fixtures.
#
# If `ns-train` is on PATH and --run-ns-train is passed, invokes
# `ns-train splatfacto --data <colmap_text_dir>` against the text export
# directory (nerfstudio convention). The trainer step is otherwise
# skipped — the smoke is intentionally tractable without a CUDA Python
# environment.

sequence="${KITTI_3DGS_SEQUENCE:-00}"
if [ "${KITTI_3DGS_DATA_DIR+x}" ]; then
  data_dir="$KITTI_3DGS_DATA_DIR"
  data_dir_is_default=0
else
  data_dir=""
  data_dir_is_default=1
fi
out_dir="${KITTI_3DGS_OUT_DIR:-target/kitti_3dgs_smoke}"
fetch_stride="${KITTI_3DGS_FETCH_STRIDE:-4}"
fetch_max_frames="${KITTI_3DGS_FETCH_MAX_FRAMES:-60}"
max_frames="${KITTI_3DGS_MAX_FRAMES:-60}"
start_frame="${KITTI_3DGS_START_FRAME:-0}"
workers="${KITTI_3DGS_WORKERS:-8}"
progress_every="${KITTI_3DGS_PROGRESS_EVERY:-10}"
colmap_image_prefix="${KITTI_3DGS_COLMAP_IMAGE_PREFIX:-}"
colmap_image_suffix="${KITTI_3DGS_COLMAP_IMAGE_SUFFIX:-.png}"
skip_fetch=0
run_ns_train=0

usage() {
  cat <<'EOF'
usage: scripts/run_kitti_3dgs_smoke.sh [options]

Fetch a small stride-4 KITTI stereo subset, run the classical metric
stereo VO demo with COLMAP 3DGS export enabled (both text and binary),
and verify both writer surfaces round-trip through their in-repo
readers (read_colmap_text_model / read_colmap_binary_model) with
matching per-format counts.

Options:
  --data-dir <dir>              KITTI subset directory
  --sequence <00..10>           KITTI odometry training sequence id
  --out-dir <dir>               Output directory
  --fetch-stride <n>            KITTI fetch stride (default 4)
  --fetch-max-frames <n>        Frames to pull (default 60)
  --max-frames <n>              Frames to consume in the demo
  --start-frame <n>             Start offset inside the fetched subset
  --workers <n>                 Parallel download workers
  --progress-every <n>          Progress print interval, 0 disables
  --colmap-image-prefix <s>     Prefix for COLMAP image NAME field
  --colmap-image-suffix <s>     Suffix for COLMAP image NAME field (default .png)
  --skip-fetch                  Reuse an already-fetched dataset
  --run-ns-train                Also invoke `ns-train splatfacto` against the text export
  -h, --help                    Show this help

Environment variables with the KITTI_3DGS_* prefix mirror the options.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --data-dir)
      data_dir="$2"
      data_dir_is_default=0
      shift 2
      ;;
    --sequence)
      sequence_num=$(printf "%s" "$2" | sed 's/^0*//')
      if [ -z "$sequence_num" ]; then
        sequence_num=0
      fi
      sequence=$(printf "%02d" "$sequence_num")
      shift 2
      ;;
    --out-dir)
      out_dir="$2"
      shift 2
      ;;
    --fetch-stride)
      fetch_stride="$2"
      shift 2
      ;;
    --fetch-max-frames)
      fetch_max_frames="$2"
      shift 2
      ;;
    --max-frames)
      max_frames="$2"
      shift 2
      ;;
    --start-frame)
      start_frame="$2"
      shift 2
      ;;
    --workers)
      workers="$2"
      shift 2
      ;;
    --progress-every)
      progress_every="$2"
      shift 2
      ;;
    --colmap-image-prefix)
      colmap_image_prefix="$2"
      shift 2
      ;;
    --colmap-image-suffix)
      colmap_image_suffix="$2"
      shift 2
      ;;
    --skip-fetch)
      skip_fetch=1
      shift
      ;;
    --run-ns-train)
      run_ns_train=1
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

if [ "$data_dir_is_default" -eq 1 ]; then
  data_dir="$HOME/datasets/kitti_seq${sequence}_3dgs_subset"
fi

if [ "$skip_fetch" -eq 0 ]; then
  python3 scripts/fetch_kitti_seq00_images.py \
    --sequence "$sequence" \
    --stride "$fetch_stride" \
    --max-frames "$fetch_max_frames" \
    --workers "$workers" \
    --skip-existing \
    --also-fetch-poses \
    --cameras image_0,image_1 \
    --out-dir "$data_dir"
fi

test -d "$data_dir/image_0"
test -d "$data_dir/image_1"
test -s "$data_dir/calib.txt"
test -s "$data_dir/poses_${sequence}.txt"

rm -rf "$out_dir"
mkdir -p "$out_dir"

colmap_text_dir="$out_dir/colmap_text"
colmap_binary_dir="$out_dir/colmap_binary"

cargo run --release --features image-io \
  --example online_slam_stereo_vo_kitti_demo -- \
  --image-left "$data_dir/image_0" \
  --image-right "$data_dir/image_1" \
  --calib "$data_dir/calib.txt" \
  --gt-poses "$data_dir/poses_${sequence}.txt" \
  --gt-original-stride "$fetch_stride" \
  --start-frame "$start_frame" \
  --max-frames "$max_frames" \
  --frame-stride 1 \
  --frontend classical \
  --progress-every "$progress_every" \
  --colmap-export "$colmap_text_dir" \
  --colmap-export-binary "$colmap_binary_dir" \
  --colmap-image-prefix "$colmap_image_prefix" \
  --colmap-image-suffix "$colmap_image_suffix" \
  --out-dir "$out_dir"

# Both writer surfaces must be present after the demo run.
test -s "$colmap_text_dir/cameras.txt"
test -s "$colmap_text_dir/images.txt"
test -s "$colmap_text_dir/points3D.txt"
test -s "$colmap_binary_dir/cameras.bin"
test -s "$colmap_binary_dir/images.bin"
test -s "$colmap_binary_dir/points3D.bin"

# Round-trip both writer surfaces through their in-repo readers so
# writer/reader divergence cannot slip through the smoke unnoticed. Then
# diff the per-format counts so a real-data cross-format parity failure
# also fails the smoke (the unit-test parity check in
# crates/io/tests/colmap_export.rs covers only synthetic input).
text_inspect_log="$out_dir/colmap_text_inspect.txt"
binary_inspect_log="$out_dir/colmap_binary_inspect.txt"
# Redirect to file (no pipe) so a failed `cargo run` is caught by set -e
# instead of being swallowed by tee's zero exit. cargo's own compile /
# Running... progress still goes to stderr and reaches the terminal live.
cargo run --release --example inspect_colmap_text_model -- "$colmap_text_dir" \
  > "$text_inspect_log"
cat "$text_inspect_log"
cargo run --release --example inspect_colmap_binary_model -- "$colmap_binary_dir" \
  > "$binary_inspect_log"
cat "$binary_inspect_log"

extract_count() {
  grep -E "^${2}=" "$1" | head -n 1 | sed -e "s/^${2}=//"
}
parity_ok=1
for field in cameras keyframes landmarks observations; do
  text_value=$(extract_count "$text_inspect_log" "$field")
  binary_value=$(extract_count "$binary_inspect_log" "$field")
  # Treat a missing field as a hard failure on either side: an empty
  # value would silently match another empty value and let an output
  # schema drift slip past the parity check.
  if [ -z "$text_value" ] || [ -z "$binary_value" ]; then
    echo "# parity check failed: $field is missing from one or both inspect logs (text='$text_value' binary='$binary_value')" >&2
    parity_ok=0
    continue
  fi
  if [ "$text_value" != "$binary_value" ]; then
    echo "# parity check failed: $field text=$text_value binary=$binary_value" >&2
    parity_ok=0
  fi
done
if [ "$parity_ok" -ne 1 ]; then
  echo "# text-vs-binary inspect counts disagree on real data; aborting" >&2
  exit 1
fi
echo "# text-vs-binary inspect counts agree"

# The inspect examples already gate on cameras > 0 and keyframes > 0; smoke
# additionally enforces landmarks > 0 so a run that silently dropped every
# stereo feature (e.g. disparity gate rejected everything upstream) does
# not pass as "smoke OK" — the resulting COLMAP triple would be useless
# for 3DGS bootstrap. Text and binary already agree by the parity check
# above, so checking the text side is sufficient.
landmark_count=$(extract_count "$text_inspect_log" landmarks)
if [ "$landmark_count" -le 0 ] 2>/dev/null; then
  echo "# smoke failed: landmarks=$landmark_count after VO + COLMAP export; the 3DGS bootstrap directory has no 3D structure" >&2
  exit 1
fi
echo "# landmarks=$landmark_count (>0 OK)"

ns_train_log=""
if [ "$run_ns_train" -eq 1 ]; then
  if command -v ns-train >/dev/null 2>&1; then
    echo "# launching ns-train splatfacto on $colmap_text_dir"
    ns_train_log="$out_dir/ns_train.log"
    # Redirect (no tee pipe) so set -e propagates a failed trainer
    # directly. Trainer progress is reachable via `tail -f $ns_train_log`
    # from another terminal; the smoke prints the path on completion.
    ns-train splatfacto --data "$colmap_text_dir" > "$ns_train_log" 2>&1
    echo "# ns-train splatfacto completed; full log at $ns_train_log"
  else
    echo "# --run-ns-train requested but ns-train was not found on PATH; skipping" >&2
  fi
fi

summary_file="$out_dir/kitti_3dgs_smoke_summary.txt"
{
  echo "# KITTI -> COLMAP -> 3DGS smoke summary"
  echo "sequence=$sequence"
  echo "data_dir=$data_dir"
  echo "out_dir=$out_dir"
  echo "fetch_stride=$fetch_stride"
  echo "fetch_max_frames=$fetch_max_frames"
  echo "max_frames=$max_frames"
  echo "start_frame=$start_frame"
  echo "colmap_text_dir=$colmap_text_dir"
  echo "colmap_binary_dir=$colmap_binary_dir"
  echo "colmap_image_prefix=$colmap_image_prefix"
  echo "colmap_image_suffix=$colmap_image_suffix"
  echo "run_ns_train=$run_ns_train"
  if [ -n "$ns_train_log" ]; then
    echo "ns_train_log=$ns_train_log"
  fi
  echo
  echo "## ATE"
  cat "$out_dir/summary.txt"
  echo
  echo "## COLMAP text inspect"
  cat "$text_inspect_log"
  echo
  echo "## COLMAP binary inspect"
  cat "$binary_inspect_log"
} > "$summary_file"

echo "# Wrote $summary_file"
sed -n '1,40p' "$summary_file"
