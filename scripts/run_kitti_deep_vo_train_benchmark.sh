#!/usr/bin/env sh
set -eu

sequences="${KITTI_DEEP_VO_TRAIN_SEQUENCES:-00,01,02,03,04,05,06,07,08,09,10}"
data_root="${KITTI_DEEP_VO_TRAIN_DATA_ROOT:-$HOME/datasets/kitti_odometry_training_subsets}"
out_dir="${KITTI_DEEP_VO_TRAIN_OUT_DIR:-target/kitti_deep_vo_train_benchmark}"
compare_root="${KITTI_DEEP_VO_TRAIN_COMPARE_ROOT:-}"
max_frames="${KITTI_DEEP_VO_MAX_FRAMES:-260}"
start_frame="${KITTI_DEEP_VO_START_FRAME:-0}"
workers="${KITTI_DEEP_VO_WORKERS:-8}"
deep_max_features="${KITTI_DEEP_VO_MAX_FEATURES:-1500}"
deep_descriptor_clip="${KITTI_DEEP_VO_DESCRIPTOR_CLIP:-0.2}"
deep_min_confidence="${KITTI_DEEP_VO_MIN_CONFIDENCE:-0.15}"
deep_temperature="${KITTI_DEEP_VO_TEMPERATURE:-25.0}"
relative_pose_iterations="${KITTI_DEEP_VO_RELATIVE_POSE_ITERATIONS:-1000}"
min_pnp_inliers="${KITTI_DEEP_VO_MIN_PNP_INLIERS:-12}"
pnp_reprojection_threshold="${KITTI_DEEP_VO_PNP_REPROJECTION_THRESHOLD:-3.32}"
pnp_max_depth="${KITTI_DEEP_VO_PNP_MAX_DEPTH:-}"
pnp_depth_hypotheses="${KITTI_DEEP_VO_PNP_DEPTH_HYPOTHESES:-}"
relative_pose_mode="${KITTI_DEEP_VO_RELATIVE_POSE_MODE:-pnp}"
motion_scale_rescue_min_translation_ratio="${KITTI_DEEP_VO_MOTION_SCALE_RESCUE_MIN_TRANSLATION_RATIO:-}"
rotation_vector_rescue_min_history="${KITTI_DEEP_VO_ROTATION_VECTOR_RESCUE_MIN_HISTORY:-}"
rotation_vector_rescue_max_delta_deg="${KITTI_DEEP_VO_ROTATION_VECTOR_RESCUE_MAX_DELTA_DEG:-}"
temporal_max_row_delta="${KITTI_DEEP_VO_TEMPORAL_MAX_ROW_DELTA:-}"
progress_every="${KITTI_DEEP_VO_PROGRESS_EVERY:-100}"
stereo_pose_refinement=0
stereo_vertical_alignment="${KITTI_DEEP_VO_STEREO_VERTICAL_ALIGNMENT:-0}"
skip_fetch=0
keep_going=0

usage() {
  cat <<'EOF'
usage: scripts/run_kitti_deep_vo_train_benchmark.sh [options]

Run the deep stereo VO smoke pipeline across KITTI odometry training
sequences 00-10 and collect a single summary.csv for sequence-level triage.

Options:
  --sequences <ids>                 Comma-separated ids, default 00..10
  --data-root <dir>                 Root for fetched sequence subsets
  --out-dir <dir>                   Benchmark output root
  --compare-root <dir>              Optional previous benchmark root for per-sequence debug diffs
  --max-frames <n>                  Frames per sequence, default 260
  --start-frame <n>                 Start offset per sequence
  --workers <n>                     Parallel download workers
  --deep-max-features <n>           HogLike feature cap per image
  --deep-descriptor-clip <x>        HogLike descriptor clipping threshold
  --deep-min-confidence <x>         Mutual-softmax confidence floor
  --deep-temperature <x>            Mutual-softmax inverse temperature
  --relative-pose-iterations <n>    Relative-pose RANSAC iterations
  --min-pnp-inliers <n>             Minimum accepted PnP inlier count
  --pnp-reprojection-threshold <px> PnP RANSAC reprojection threshold
  --pnp-max-depth <m>               Optional PnP 3D-point max depth
  --pnp-depth-hypotheses <m,...>    Extra guarded PnP max-depth candidates
  --stereo-pose-refinement          Refine relative pose with current stereo reprojection
  --stereo-vertical-alignment       Align only relative-pose vertical translation with stereo 3D pairs
  --motion-scale-rescue-min-translation-ratio <x>
                                    Override motion-scale rescue collapse threshold
  --rotation-vector-rescue-min-history <n>
                                    Override rotation-vector rescue history length
  --rotation-vector-rescue-max-delta-deg <deg>
                                    Override rotation-vector rescue trigger delta
  --relative-pose-mode <pnp|kabsch> Relative-pose source
  --temporal-max-row-delta <px>     Optional vertical temporal-match gate
  --progress-every <n>              Progress print interval, 0 disables
  --skip-fetch                      Reuse already-fetched sequence subsets
  --keep-going                      Continue after a sequence fails
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
    --compare-root)
      compare_root="$2"
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
    --deep-max-features)
      deep_max_features="$2"
      shift 2
      ;;
    --deep-descriptor-clip)
      deep_descriptor_clip="$2"
      shift 2
      ;;
    --deep-min-confidence)
      deep_min_confidence="$2"
      shift 2
      ;;
    --deep-temperature)
      deep_temperature="$2"
      shift 2
      ;;
    --relative-pose-iterations)
      relative_pose_iterations="$2"
      shift 2
      ;;
    --min-pnp-inliers)
      min_pnp_inliers="$2"
      shift 2
      ;;
    --pnp-reprojection-threshold)
      pnp_reprojection_threshold="$2"
      shift 2
      ;;
    --pnp-max-depth)
      pnp_max_depth="$2"
      shift 2
      ;;
    --pnp-depth-hypotheses)
      pnp_depth_hypotheses="$2"
      shift 2
      ;;
    --stereo-pose-refinement)
      stereo_pose_refinement=1
      shift
      ;;
    --stereo-vertical-alignment)
      stereo_vertical_alignment=1
      shift
      ;;
    --motion-scale-rescue-min-translation-ratio)
      motion_scale_rescue_min_translation_ratio="$2"
      shift 2
      ;;
    --rotation-vector-rescue-min-history)
      rotation_vector_rescue_min_history="$2"
      shift 2
      ;;
    --rotation-vector-rescue-max-delta-deg)
      rotation_vector_rescue_max_delta_deg="$2"
      shift 2
      ;;
    --relative-pose-mode)
      relative_pose_mode="$2"
      shift 2
      ;;
    --temporal-max-row-delta)
      temporal_max_row_delta="$2"
      shift 2
      ;;
    --progress-every)
      progress_every="$2"
      shift 2
      ;;
    --skip-fetch)
      skip_fetch=1
      shift
      ;;
    --keep-going)
      keep_going=1
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
        "n_frames",
        "gt_length_m",
        "vo_length_m",
        "ate_mean_m",
        "ate_rmse_m",
        "ate_max_m",
        "public_segment_count",
        "public_t_rel_percent",
        "public_r_rel_deg_per_m",
        "public_max_t_rel_percent",
        "public_max_r_rel_deg_per_m",
        "rel_mean_t_mag_err_m",
        "rel_max_t_mag_err_m",
        "rel_mean_rot_err_deg",
        "rel_max_rot_err_deg",
        "source_pnp",
        "source_pnp_fallback",
        "source_kabsch",
        "source_kabsch_fallback",
        "worst_t_pair",
        "worst_t_mag_err_m",
        "worst_rot_pair",
        "worst_rot_err_deg",
        "worst_segment",
        "worst_segment_t_rel_percent",
        "worst_segment_r_rel_deg_per_m",
        "slam_debug_summary",
        "slam_debug_report",
        "slam_debug_html",
        "slam_debug_worst_pairs",
        "slam_debug_compare",
        "slam_debug_compare_report",
        "out_dir",
    ])
PY

append_summary_row() {
  seq="$1"
  seq_out="$2"
  status="$3"
  python3 - "$summary_csv" "$seq" "$seq_out" "$status" <<'PY'
from pathlib import Path
import csv
import json
import math
import sys

summary_csv = Path(sys.argv[1])
seq = sys.argv[2]
out_dir = Path(sys.argv[3])
status = sys.argv[4]

fields = {}
summary_path = out_dir / "summary.txt"
if summary_path.exists():
    for token in summary_path.read_text().replace("\n", " ").split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value

public = {}
public_path = out_dir / "kitti_eval_public_lengths" / "kitti_odometry_summary.json"
if public_path.exists():
    public = json.loads(public_path.read_text())

debug = {}
debug_path = out_dir / "slam_debug" / "slam_debug_summary.json"
if debug_path.exists():
    debug = json.loads(debug_path.read_text())

worst_t_pair = ""
worst_t = ""
worst_rot_pair = ""
worst_rot = ""
worst_segment = ""
worst_segment_t = ""
worst_segment_r = ""
rel_path = out_dir / "relative_pose_errors.csv"
if rel_path.exists():
    with rel_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if rows:
        def as_float(row, key):
            try:
                return float(row[key])
            except (KeyError, TypeError, ValueError):
                return -math.inf
        t_row = max(rows, key=lambda row: as_float(row, "translation_magnitude_error_m"))
        r_row = max(rows, key=lambda row: as_float(row, "rotation_error_deg"))
        worst_t_pair = f"{t_row.get('from_id', '')}->{t_row.get('to_id', '')}"
        worst_t = t_row.get("translation_magnitude_error_m", "")
        worst_rot_pair = f"{r_row.get('from_id', '')}->{r_row.get('to_id', '')}"
        worst_rot = r_row.get("rotation_error_deg", "")

segments = debug.get("worst_kitti_segments") or []
if segments:
    seg = segments[0]
    worst_segment = f"{seg.get('first_frame_id', '')}->{seg.get('last_frame_id', '')}@{seg.get('length_m', '')}m"
    worst_segment_t = seg.get("translational_error_percent", "")
    worst_segment_r = seg.get("rotational_error_deg_per_m", "")

def field(name):
    return fields.get(name, "")

def public_field(name):
    value = public.get(name, "")
    return "" if value is None else value

row = [
    seq,
    status,
    field("n_frames"),
    field("gt_length_m"),
    field("vo_length_m"),
    field("vo_ate_mean_m"),
    field("vo_ate_rmse_m"),
    field("vo_ate_max_m"),
    public_field("segment_count"),
    public_field("mean_translational_error_percent"),
    public_field("mean_rotational_error_deg_per_m"),
    public_field("max_translational_error_percent"),
    public_field("max_rotational_error_deg_per_m"),
    field("relative_pose_mean_t_mag_err_m"),
    field("relative_pose_max_t_mag_err_m"),
    field("relative_pose_mean_rot_err_deg"),
    field("relative_pose_max_rot_err_deg"),
    field("relative_pose_source_pnp"),
    field("relative_pose_source_pnp_fallback"),
    field("relative_pose_source_kabsch"),
    field("relative_pose_source_kabsch_fallback"),
    worst_t_pair,
    worst_t,
    worst_rot_pair,
    worst_rot,
    worst_segment,
    worst_segment_t,
    worst_segment_r,
    str(out_dir / "slam_debug" / "slam_debug_summary.json") if debug_path.exists() else "",
    str(out_dir / "slam_debug" / "slam_debug_report.md") if (out_dir / "slam_debug" / "slam_debug_report.md").exists() else "",
    str(out_dir / "slam_debug" / "slam_debug_report.html") if (out_dir / "slam_debug" / "slam_debug_report.html").exists() else "",
    str(out_dir / "slam_debug" / "slam_debug_worst_pairs.csv") if (out_dir / "slam_debug" / "slam_debug_worst_pairs.csv").exists() else "",
    str(out_dir / "slam_debug" / "slam_debug_compare.json") if (out_dir / "slam_debug" / "slam_debug_compare.json").exists() else "",
    str(out_dir / "slam_debug" / "slam_debug_compare.md") if (out_dir / "slam_debug" / "slam_debug_compare.md").exists() else "",
    str(out_dir),
]

with summary_csv.open("a", newline="") as f:
    csv.writer(f).writerow(row)
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
  echo "# KITTI deep VO training benchmark: sequence $seq"

  smoke_args="--sequence $seq --data-dir $seq_data --out-dir $seq_out --max-frames $max_frames --start-frame $start_frame --workers $workers --deep-max-features $deep_max_features --deep-descriptor-clip $deep_descriptor_clip --deep-min-confidence $deep_min_confidence --deep-temperature $deep_temperature --relative-pose-iterations $relative_pose_iterations --min-pnp-inliers $min_pnp_inliers --pnp-reprojection-threshold $pnp_reprojection_threshold --relative-pose-mode $relative_pose_mode --progress-every $progress_every"
  if [ -n "$pnp_max_depth" ]; then
    smoke_args="$smoke_args --pnp-max-depth $pnp_max_depth"
  fi
  if [ -n "$pnp_depth_hypotheses" ]; then
    smoke_args="$smoke_args --pnp-depth-hypotheses $pnp_depth_hypotheses"
  fi
  if [ "$stereo_pose_refinement" -eq 1 ]; then
    smoke_args="$smoke_args --stereo-pose-refinement"
  fi
  if [ "$stereo_vertical_alignment" -eq 1 ]; then
    smoke_args="$smoke_args --stereo-vertical-alignment"
  fi
  if [ -n "$motion_scale_rescue_min_translation_ratio" ]; then
    smoke_args="$smoke_args --motion-scale-rescue-min-translation-ratio $motion_scale_rescue_min_translation_ratio"
  fi
  if [ -n "$rotation_vector_rescue_min_history" ]; then
    smoke_args="$smoke_args --rotation-vector-rescue-min-history $rotation_vector_rescue_min_history"
  fi
  if [ -n "$rotation_vector_rescue_max_delta_deg" ]; then
    smoke_args="$smoke_args --rotation-vector-rescue-max-delta-deg $rotation_vector_rescue_max_delta_deg"
  fi
  if [ -n "$temporal_max_row_delta" ]; then
    smoke_args="$smoke_args --temporal-max-row-delta $temporal_max_row_delta"
  fi
  if [ "$skip_fetch" -eq 1 ]; then
    smoke_args="$smoke_args --skip-fetch"
  fi

  if scripts/run_kitti_deep_vo_smoke.sh $smoke_args; then
    if [ -n "$compare_root" ]; then
      compare_seq_dir="$compare_root/seq${seq}"
      if [ -d "$compare_seq_dir" ]; then
        python3 scripts/visual_slam_debug_report.py \
          "$seq_out" \
          --compare "$compare_seq_dir" \
          --out-dir "$seq_out/slam_debug"
      else
        echo "# compare-root sequence dir not found, skipping compare: $compare_seq_dir" >&2
      fi
    fi
    append_summary_row "$seq" "$seq_out" "ok"
  else
    append_summary_row "$seq" "$seq_out" "failed"
    if [ "$keep_going" -ne 1 ]; then
      echo "# Sequence $seq failed; rerun with --keep-going to continue." >&2
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

def max_finite(rows, key):
    finite_rows = [row for row in rows if math.isfinite(as_float(row, key))]
    if not finite_rows:
        return None
    return max(finite_rows, key=lambda row: as_float(row, key))

def link(label, path):
    if not path:
        return ""
    return f"[{label}]({path})"

lines = ["# KITTI Deep VO Training Benchmark", ""]
lines.append(f"rows={len(rows)} ok={len(ok_rows)}")
lines.append("")
lines.append("| seq | status | frames | ATE mean/RMSE/max | t_rel | r_rel | fallbacks | worst pair | worst segment | debug |")
lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |")
for row in rows:
    fallbacks = 0
    for key in ("source_pnp_fallback", "source_kabsch", "source_kabsch_fallback"):
        try:
            fallbacks += int(row[key])
        except (KeyError, ValueError):
            pass
    ate = f"{row['ate_mean_m']} / {row['ate_rmse_m']} / {row['ate_max_m']}"
    worst = f"{row['worst_t_pair']} ({row['worst_t_mag_err_m']} m)"
    segment = row.get("worst_segment", "")
    if row.get("worst_segment_t_rel_percent"):
        segment = (
            f"{segment} ({row['worst_segment_t_rel_percent']}%, "
            f"{row.get('worst_segment_r_rel_deg_per_m', '')} deg/m)"
        )
    debug_links = " ".join(
        part for part in [
            link("report", row.get("slam_debug_report", "")),
            link("html", row.get("slam_debug_html", "")),
            link("compare", row.get("slam_debug_compare_report", "")),
        ] if part
    )
    lines.append(
        f"| {row['sequence']} | {row['status']} | {row['n_frames']} | {ate} | "
        f"{row['public_t_rel_percent']} | {row['public_r_rel_deg_per_m']} | "
        f"{fallbacks} | {worst} | {segment} | {debug_links} |"
    )

if ok_rows:
    worst_t = max_finite(ok_rows, "public_t_rel_percent")
    worst_r = max_finite(ok_rows, "public_r_rel_deg_per_m")
    worst_segment = max_finite(ok_rows, "worst_segment_t_rel_percent")
    lines.append("")
    if worst_t:
        lines.append(
            f"worst_t_rel_sequence={worst_t['sequence']} "
            f"value={worst_t['public_t_rel_percent']}"
        )
    if worst_r:
        lines.append(
            f"worst_r_rel_sequence={worst_r['sequence']} "
            f"value={worst_r['public_r_rel_deg_per_m']}"
        )
    if worst_segment:
        lines.append(
            f"worst_segment_sequence={worst_segment['sequence']} "
            f"segment={worst_segment.get('worst_segment', '')} "
            f"t_rel={worst_segment.get('worst_segment_t_rel_percent', '')} "
            f"r_rel={worst_segment.get('worst_segment_r_rel_deg_per_m', '')}"
        )

md_path.write_text("\n".join(lines) + "\n")
print(f"# Wrote {csv_path}")
print(f"# Wrote {md_path}")
print(md_path.read_text())
PY
