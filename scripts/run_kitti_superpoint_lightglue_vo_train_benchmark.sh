#!/usr/bin/env sh
set -eu

sequences="${KITTI_SP_LG_TRAIN_SEQUENCES:-00,01,02,03,04,05,06,07,08,09,10}"
data_root="${KITTI_SP_LG_TRAIN_DATA_ROOT:-$HOME/datasets/kitti_odometry_training_subsets}"
out_dir="${KITTI_SP_LG_TRAIN_OUT_DIR:-target/kitti_superpoint_lightglue_vo_train_benchmark}"
hog_compare_root="${KITTI_SP_LG_HOG_COMPARE_ROOT:-target/kitti_deep_vo_train_benchmark_auto_conf145}"
max_frames="${KITTI_SP_LG_MAX_FRAMES:-260}"
start_frame="${KITTI_SP_LG_START_FRAME:-0}"
frame_stride="${KITTI_SP_LG_FRAME_STRIDE:-1}"
device="${KITTI_SP_LG_DEVICE:-cuda}"
max_keypoints="${KITTI_SP_LG_MAX_KEYPOINTS:-2048}"
relative_pose_mode="${KITTI_SP_LG_RELATIVE_POSE_MODE:-pnp}"
min_stereo_confidence="${KITTI_SP_LG_MIN_STEREO_CONFIDENCE:-0.5}"
min_temporal_confidence="${KITTI_SP_LG_MIN_TEMPORAL_CONFIDENCE:-0.5}"
confidence_overrides="${KITTI_SP_LG_CONFIDENCE_OVERRIDES:-}"
projection_left="${KITTI_SP_LG_PROJECTION_LEFT:-P0}"
projection_right="${KITTI_SP_LG_PROJECTION_RIGHT:-P1}"
skip_export=0
skip_vo=0
keep_going=0
enable_ba=0
ba_max_init_residual="${KITTI_SP_LG_BA_MAX_INIT_RESIDUAL:-3}"
ba_min_track_count="${KITTI_SP_LG_BA_MIN_TRACK_COUNT:-2000}"
ba_huber_delta="${KITTI_SP_LG_BA_HUBER_DELTA:-3}"
ba_overrides="${KITTI_SP_LG_BA_OVERRIDES:-}"

usage() {
  cat <<'EOF'
usage: scripts/run_kitti_superpoint_lightglue_vo_train_benchmark.sh [options]

Run file-backed SuperPoint/LightGlue stereo VO across KITTI odometry training
subsets and collect sequence-level KITTI metrics.

Options:
  --sequences <ids>                 Comma-separated ids, default 00..10
  --data-root <dir>                 Root containing seqXX/image_0, image_1, calib.txt, poses_XX.txt
  --out-dir <dir>                   Benchmark output root
  --hog-compare-root <dir>          Optional HOG benchmark root for delta columns
  --max-frames <n>                  Frames per sequence, default 260
  --start-frame <n>                 Start offset per sequence, default 0
  --frame-stride <n>                Frame stride, default 1
  --device <auto|cpu|cuda>          PyTorch device, default cuda
  --max-keypoints <n>               SuperPoint max keypoints, default 2048
  --relative-pose-mode <pnp|kabsch> VO pose mode, default pnp
  --min-stereo-confidence <x>       LightGlue stereo confidence floor, default 0.5
  --min-temporal-confidence <x>     LightGlue temporal confidence floor, default 0.5
  --confidence-overrides <spec>      Per-sequence floors, e.g. 01:0.7:0.7,03:0.7:0.7
  --projection-left <label>         KITTI left projection label, default P0
  --projection-right <label>        KITTI right projection label, default P1
  --skip-export                     Reuse already exported feature/match files
  --skip-vo                         Reuse already generated vo_poses.txt
  --keep-going                      Continue after a sequence fails
  --enable-ba                       Run multi-frame BA refinement after VO
  --ba-max-init-residual <px>       Per-track init residual gate, default 3
  --ba-min-track-count <n>          Below this, BA auto-skips, default 2000
  --ba-huber-delta <px>             Huber kernel delta, default 3
  --ba-overrides <spec>             Per-seq BA tuning, e.g.
                                    "05:resid=2.5,02:skip,03:win=50,03:huber=1.5"
                                    (resid=X overrides --ba-max-init-residual,
                                     tracks=X overrides --ba-min-track-count,
                                     win=X enables sliding-window BA,
                                     huber=X overrides --ba-huber-delta,
                                     skip disables BA for that sequence)
  -h, --help                        Show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --sequences)
      sequences="$2"
      shift 2
      ;;
    --data-root)
      data_root="$2"
      shift 2
      ;;
    --out-dir)
      out_dir="$2"
      shift 2
      ;;
    --hog-compare-root)
      hog_compare_root="$2"
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
    --frame-stride)
      frame_stride="$2"
      shift 2
      ;;
    --device)
      device="$2"
      shift 2
      ;;
    --max-keypoints)
      max_keypoints="$2"
      shift 2
      ;;
    --relative-pose-mode)
      relative_pose_mode="$2"
      shift 2
      ;;
    --min-stereo-confidence)
      min_stereo_confidence="$2"
      shift 2
      ;;
    --min-temporal-confidence)
      min_temporal_confidence="$2"
      shift 2
      ;;
    --confidence-overrides)
      confidence_overrides="$2"
      shift 2
      ;;
    --projection-left)
      projection_left="$2"
      shift 2
      ;;
    --projection-right)
      projection_right="$2"
      shift 2
      ;;
    --skip-export)
      skip_export=1
      shift
      ;;
    --skip-vo)
      skip_vo=1
      shift
      ;;
    --keep-going)
      keep_going=1
      shift
      ;;
    --enable-ba)
      enable_ba=1
      shift
      ;;
    --ba-max-init-residual)
      ba_max_init_residual="$2"
      shift 2
      ;;
    --ba-min-track-count)
      ba_min_track_count="$2"
      shift 2
      ;;
    --ba-huber-delta)
      ba_huber_delta="$2"
      shift 2
      ;;
    --ba-overrides)
      ba_overrides="$2"
      shift 2
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
summary_csv="$out_dir/summary.csv"

python3 - "$summary_csv" <<'PY'
from pathlib import Path
import csv
import sys

path = Path(sys.argv[1])
with path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "sequence",
        "status",
        "frames",
        "min_stereo_confidence",
        "min_temporal_confidence",
        "vo_length_m",
        "ate_mean_m",
        "ate_rmse_m",
        "ate_max_m",
        "kitti_segment_count",
        "t_rel_percent",
        "r_rel_deg_per_m",
        "max_t_rel_percent",
        "max_r_rel_deg_per_m",
        "hog_t_rel_percent",
        "hog_r_rel_deg_per_m",
        "hog_max_t_rel_percent",
        "delta_t_rel_percent",
        "delta_r_rel_deg_per_m",
        "delta_max_t_rel_percent",
        "mean_temporal_matches",
        "mean_stereo_pairs",
        "mean_inliers",
        "out_dir",
    ])
PY

slice_gt_poses() {
  seq="$1"
  seq_data="$2"
  seq_out="$3"
  python3 - "$seq" "$seq_data" "$seq_out/gt_poses.txt" "$start_frame" "$frame_stride" "$max_frames" <<'PY'
from pathlib import Path
import sys

seq, data_dir, out_path, start, stride, frames = sys.argv[1:]
data_dir = Path(data_dir)
pose_path = data_dir / f"poses_{seq}.txt"
if not pose_path.exists():
    pose_path = data_dir / "poses.txt"
lines = pose_path.read_text().splitlines()
start = int(start)
stride = int(stride)
frames = int(frames)
selected = []
for index in range(start, len(lines), stride):
    selected.append(lines[index])
    if len(selected) >= frames:
        break
if len(selected) < frames:
    raise SystemExit(f"not enough GT poses in {pose_path}: need {frames}, got {len(selected)}")
Path(out_path).write_text("\n".join(selected) + "\n")
PY
}

append_summary_row() {
  seq="$1"
  seq_out="$2"
  status="$3"
  seq_min_stereo_confidence="$4"
  seq_min_temporal_confidence="$5"
  python3 - "$summary_csv" "$seq" "$seq_out" "$status" "$hog_compare_root" \
    "$seq_min_stereo_confidence" "$seq_min_temporal_confidence" <<'PY'
from pathlib import Path
import csv
import json
import math
import sys

summary_csv = Path(sys.argv[1])
seq = sys.argv[2]
seq_out = Path(sys.argv[3])
status = sys.argv[4]
hog_root = Path(sys.argv[5]) if sys.argv[5] else None
seq_min_stereo_confidence = sys.argv[6]
seq_min_temporal_confidence = sys.argv[7]

def read_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())

def read_fields(path):
    fields = {}
    if path.exists():
        for token in path.read_text().replace("\n", " ").split():
            if "=" in token:
                key, value = token.split("=", 1)
                fields[key] = value
    return fields

def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")

fields = read_fields(seq_out / "summary.txt")
ate = read_json(seq_out / "ate_eval" / "error_summary.json")
kitti = read_json(seq_out / "kitti_eval" / "kitti_odometry_summary.json")

hog_kitti = {}
if hog_root:
    hog_candidates = [
        hog_root / f"seq{seq}" / "kitti_eval_public_lengths" / "kitti_odometry_summary.json",
        hog_root / f"seq{seq}" / "kitti_eval" / "kitti_odometry_summary.json",
    ]
    for candidate in hog_candidates:
        if candidate.exists():
            hog_kitti = read_json(candidate)
            break

diag_path = seq_out / "frontend_pair_diagnostics.csv"
mean_temporal = ""
mean_stereo = ""
mean_inliers = ""
if diag_path.exists():
    with diag_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if rows:
        def mean(key):
            vals = [num(row.get(key)) for row in rows]
            vals = [v for v in vals if math.isfinite(v)]
            return sum(vals) / len(vals) if vals else ""
        mean_temporal = mean("temporal_matches")
        mean_stereo = mean("stereo_pair_correspondences")
        mean_inliers = mean("inliers")

t_rel = kitti.get("mean_translational_error_percent", "")
r_rel = kitti.get("mean_rotational_error_deg_per_m", "")
max_t = kitti.get("max_translational_error_percent", "")
max_r = kitti.get("max_rotational_error_deg_per_m", "")
hog_t = hog_kitti.get("mean_translational_error_percent", "")
hog_r = hog_kitti.get("mean_rotational_error_deg_per_m", "")
hog_max_t = hog_kitti.get("max_translational_error_percent", "")

def delta(current, base):
    c = num(current)
    b = num(base)
    if math.isfinite(c) and math.isfinite(b):
        return c - b
    return ""

row = [
    seq,
    status,
    fields.get("frames", ""),
    seq_min_stereo_confidence,
    seq_min_temporal_confidence,
    fields.get("trajectory_length_m", ""),
    ate.get("mean_translation_error", ""),
    ate.get("rmse_translation_error", ""),
    ate.get("max_translation_error", ""),
    kitti.get("segment_count", ""),
    t_rel,
    r_rel,
    max_t,
    max_r,
    hog_t,
    hog_r,
    hog_max_t,
    delta(t_rel, hog_t),
    delta(r_rel, hog_r),
    delta(max_t, hog_max_t),
    mean_temporal,
    mean_stereo,
    mean_inliers,
    str(seq_out),
]

with summary_csv.open("a", newline="") as f:
    csv.writer(f).writerow(row)
PY
}

resolve_confidence_for_sequence() {
  seq="$1"
  python3 - "$seq" "$min_stereo_confidence" "$min_temporal_confidence" "$confidence_overrides" <<'PY'
import sys

seq, stereo_default, temporal_default, overrides = sys.argv[1:]
stereo = stereo_default
temporal = temporal_default
seq_int = str(int(seq))
for item in [part.strip() for part in overrides.split(",") if part.strip()]:
    fields = item.replace("=", ":").replace("/", ":").split(":")
    if len(fields) != 3:
        raise SystemExit(f"bad confidence override {item!r}; expected SEQ:STEREO:TEMPORAL")
    item_seq, item_stereo, item_temporal = fields
    try:
        normalized = f"{int(item_seq):02d}"
    except ValueError as exc:
        raise SystemExit(f"bad sequence in confidence override {item!r}") from exc
    if normalized == seq or str(int(normalized)) == seq_int:
        stereo = item_stereo
        temporal = item_temporal
print(stereo, temporal)
PY
}

# Per-sequence BA tuning. Spec syntax:
#   <seq>:resid=<float>          override --ba-max-init-residual for that seq
#   <seq>:tracks=<int>           override --ba-min-track-count for that seq
#   <seq>:win=<int>              enable sliding-window BA at this window size
#   <seq>:huber=<float>          override --ba-huber-delta for that seq
#   <seq>:skip                   disable BA entirely for that seq
# Multiple overrides can be comma-joined,
# e.g. "03:win=50,03:huber=1.5,03:tracks=200,10:resid=8,02:skip,05:skip".
resolve_ba_for_sequence() {
  seq="$1"
  python3 - "$seq" "$enable_ba" "$ba_max_init_residual" "$ba_min_track_count" "$ba_huber_delta" "$ba_overrides" <<'PY'
import sys

seq, enable_ba, resid_default, tracks_default, huber_default, overrides = sys.argv[1:]
enable = int(enable_ba)
resid = resid_default
tracks = tracks_default
huber = huber_default
win = ""
seq_int = str(int(seq))
for item in [part.strip() for part in overrides.split(",") if part.strip()]:
    if ":" not in item:
        raise SystemExit(f"bad BA override {item!r}; expected SEQ:directive")
    item_seq, directive = item.split(":", 1)
    try:
        normalized = f"{int(item_seq):02d}"
    except ValueError as exc:
        raise SystemExit(f"bad sequence in BA override {item!r}") from exc
    if normalized != seq and str(int(normalized)) != seq_int:
        continue
    directive = directive.strip()
    if directive == "skip":
        enable = 0
    elif directive.startswith("resid="):
        resid = directive[len("resid="):]
    elif directive.startswith("tracks="):
        tracks = directive[len("tracks="):]
    elif directive.startswith("win="):
        win = directive[len("win="):]
    elif directive.startswith("huber="):
        huber = directive[len("huber="):]
    else:
        raise SystemExit(f"unknown BA directive {directive!r} in override {item!r}")
print(enable, resid, tracks, win if win else "-", huber)
PY
}

old_ifs="$IFS"
IFS=","
set -- $sequences
IFS="$old_ifs"

for raw_seq in "$@"; do
  seq_num=$(printf "%s" "$raw_seq" | sed 's/^0*//')
  if [ -z "$seq_num" ]; then
    seq_num=0
  fi
  seq=$(printf "%02d" "$seq_num")
  seq_data="$data_root/seq${seq}"
  seq_out="$out_dir/seq${seq}"
  export_dir="$seq_out/external_deep"
  set -- $(resolve_confidence_for_sequence "$seq")
  seq_min_stereo_confidence="$1"
  seq_min_temporal_confidence="$2"
  set -- $(resolve_ba_for_sequence "$seq")
  seq_enable_ba="$1"
  seq_ba_resid="$2"
  seq_ba_tracks="$3"
  seq_ba_win="$4"
  seq_ba_huber="$5"
  echo "# KITTI SuperPoint/LightGlue VO benchmark: sequence $seq"
  echo "# confidence floors: stereo=$seq_min_stereo_confidence temporal=$seq_min_temporal_confidence"
  if [ "$seq_enable_ba" -eq 1 ]; then
    if [ "$seq_ba_win" != "-" ]; then
      echo "# BA: resid=$seq_ba_resid min_tracks=$seq_ba_tracks window_size=$seq_ba_win huber=$seq_ba_huber"
    else
      echo "# BA: resid=$seq_ba_resid min_tracks=$seq_ba_tracks huber=$seq_ba_huber"
    fi
  else
    echo "# BA: disabled"
  fi
  mkdir -p "$seq_out"

  run_status=ok
  if [ "$skip_export" -ne 1 ]; then
    if ! scripts/export_superpoint_lightglue.py \
      --left-dir "$seq_data/image_0" \
      --right-dir "$seq_data/image_1" \
      --out-dir "$export_dir" \
      --start-frame "$start_frame" \
      --frame-stride "$frame_stride" \
      --frames "$max_frames" \
      --device "$device" \
      --max-keypoints "$max_keypoints"; then
      run_status=failed_export
    fi
  fi

  if [ "$run_status" = "ok" ]; then
    if ! slice_gt_poses "$seq" "$seq_data" "$seq_out"; then
      run_status=failed_gt
    fi
  fi

  if [ "$run_status" = "ok" ] && [ "$skip_vo" -ne 1 ]; then
    ba_args=""
    if [ "$seq_enable_ba" -eq 1 ]; then
      ba_args="--enable-ba --ba-max-init-residual $seq_ba_resid --ba-min-track-count $seq_ba_tracks --ba-huber-delta $seq_ba_huber"
      if [ "$seq_ba_win" != "-" ]; then
        ba_args="$ba_args --ba-window-size $seq_ba_win"
      fi
    fi
    # shellcheck disable=SC2086
    if ! cargo run --release --example stereo_vo_external_deep_files -- \
      --features-dir "$export_dir" \
      --frames "$max_frames" \
      --out-dir "$seq_out" \
      --relative-pose-mode "$relative_pose_mode" \
      --calib "$seq_data/calib.txt" \
      --projection-left "$projection_left" \
      --projection-right "$projection_right" \
      --min-stereo-confidence "$seq_min_stereo_confidence" \
      --min-temporal-confidence "$seq_min_temporal_confidence" \
      $ba_args > "$seq_out/vo.log" 2>&1; then
      run_status=failed_vo
    fi
  fi

  if [ "$run_status" = "ok" ]; then
    if ! cargo run --example evaluate_trajectory_from_kitti_files -- \
      --out-dir "$seq_out/ate_eval" \
      --align-origin \
      "$seq_out/vo_poses.txt" \
      "$seq_out/gt_poses.txt" > "$seq_out/ate_eval.log" 2>&1; then
      run_status=failed_ate
    fi
  fi

  if [ "$run_status" = "ok" ]; then
    if ! cargo run --example evaluate_kitti_odometry_benchmark -- \
      --out-dir "$seq_out/kitti_eval" \
      "$seq_out/vo_poses.txt" \
      "$seq_out/gt_poses.txt" > "$seq_out/kitti_eval.log" 2>&1; then
      run_status=failed_kitti
    fi
  fi

  append_summary_row "$seq" "$seq_out" "$run_status" \
    "$seq_min_stereo_confidence" "$seq_min_temporal_confidence"
  if [ "$run_status" != "ok" ]; then
    echo "# Sequence $seq failed: $run_status" >&2
    if [ "$keep_going" -ne 1 ]; then
      exit 1
    fi
  fi
done

python3 - "$summary_csv" "$out_dir/summary.md" <<'PY'
from pathlib import Path
import csv
import math
import sys

csv_path = Path(sys.argv[1])
md_path = Path(sys.argv[2])
rows = list(csv.DictReader(csv_path.open()))
ok_rows = [row for row in rows if row["status"] == "ok"]

def as_float(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")

def mean(rows, key):
    vals = [as_float(row, key) for row in rows]
    vals = [value for value in vals if math.isfinite(value)]
    return sum(vals) / len(vals) if vals else float("nan")

lines = ["# KITTI SuperPoint/LightGlue VO Training Benchmark", ""]
lines.append(f"rows={len(rows)} ok={len(ok_rows)}")
if ok_rows:
    lines.append(
        "mean_t_rel={:.6f} mean_r_rel={:.6f} mean_max_t_rel={:.6f}".format(
            mean(ok_rows, "t_rel_percent"),
            mean(ok_rows, "r_rel_deg_per_m"),
            mean(ok_rows, "max_t_rel_percent"),
        )
    )
    if any(row.get("hog_t_rel_percent") for row in ok_rows):
        lines.append(
            "mean_delta_vs_hog_t_rel={:.6f} mean_delta_vs_hog_r_rel={:.6f} "
            "mean_delta_vs_hog_max_t_rel={:.6f}".format(
                mean(ok_rows, "delta_t_rel_percent"),
                mean(ok_rows, "delta_r_rel_deg_per_m"),
                mean(ok_rows, "delta_max_t_rel_percent"),
            )
        )
lines.append("")
lines.append("| seq | status | conf s/t | t_rel | r_rel | max_t | HOG t_rel | delta t | delta r | mean temporal | mean stereo pairs | mean inliers |")
lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for row in rows:
    lines.append(
        f"| {row['sequence']} | {row['status']} | "
        f"{row['min_stereo_confidence']}/{row['min_temporal_confidence']} | "
        f"{row['t_rel_percent']} | "
        f"{row['r_rel_deg_per_m']} | {row['max_t_rel_percent']} | "
        f"{row['hog_t_rel_percent']} | {row['delta_t_rel_percent']} | "
        f"{row['delta_r_rel_deg_per_m']} | {row['mean_temporal_matches']} | "
        f"{row['mean_stereo_pairs']} | {row['mean_inliers']} |"
    )

md_path.write_text("\n".join(lines) + "\n")
print(f"# Wrote {csv_path}")
print(f"# Wrote {md_path}")
print(md_path.read_text())
PY
