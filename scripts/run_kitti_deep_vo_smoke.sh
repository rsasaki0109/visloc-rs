#!/usr/bin/env sh
set -eu

sequence="${KITTI_DEEP_VO_SEQUENCE:-00}"
if [ "${KITTI_DEEP_VO_DATA_DIR+x}" ]; then
  data_dir="$KITTI_DEEP_VO_DATA_DIR"
  data_dir_is_default=0
else
  data_dir=""
  data_dir_is_default=1
fi
out_dir="${KITTI_DEEP_VO_OUT_DIR:-target/kitti_stereo_vo_deep_260}"
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
stereo_pose_refinement="${KITTI_DEEP_VO_STEREO_POSE_REFINEMENT:-0}"
stereo_vertical_alignment="${KITTI_DEEP_VO_STEREO_VERTICAL_ALIGNMENT:-0}"
motion_scale_rescue_min_translation_ratio="${KITTI_DEEP_VO_MOTION_SCALE_RESCUE_MIN_TRANSLATION_RATIO:-}"
rotation_vector_rescue_min_history="${KITTI_DEEP_VO_ROTATION_VECTOR_RESCUE_MIN_HISTORY:-}"
rotation_vector_rescue_max_delta_deg="${KITTI_DEEP_VO_ROTATION_VECTOR_RESCUE_MAX_DELTA_DEG:-}"
relative_pose_mode="${KITTI_DEEP_VO_RELATIVE_POSE_MODE:-pnp}"
temporal_max_row_delta="${KITTI_DEEP_VO_TEMPORAL_MAX_ROW_DELTA:-}"
progress_every="${KITTI_DEEP_VO_PROGRESS_EVERY:-25}"
skip_fetch=0

usage() {
  cat <<'EOF'
usage: scripts/run_kitti_deep_vo_smoke.sh [options]

Fetch a stride-1 KITTI stereo subset, run the deep-style metric stereo VO
demo, and export current-public and explicit-100 m KITTI odometry summaries.

Options:
  --data-dir <dir>                  KITTI subset directory
  --sequence <00..10>               KITTI odometry training sequence id
  --out-dir <dir>                   Output directory
  --max-frames <n>                  Number of stride-1 frames to fetch/run
  --start-frame <n>                 Start offset inside the fetched subset
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
  --skip-fetch                      Reuse an already-fetched dataset
  -h, --help                        Show this help

Environment variables with the KITTI_DEEP_VO_* prefix mirror the options.
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
  data_dir="$HOME/datasets/kitti_seq${sequence}_stride1_subset"
fi

if [ "$skip_fetch" -eq 0 ]; then
  python3 scripts/fetch_kitti_seq00_images.py \
    --sequence "$sequence" \
    --stride 1 \
    --max-frames "$max_frames" \
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

frame_end=$((start_frame + max_frames - 1))
for cam in image_0 image_1; do
  frame="$start_frame"
  while [ "$frame" -le "$frame_end" ]; do
    frame_name=$(printf "%06d.png" "$frame")
    frame_path="$data_dir/$cam/$frame_name"
    if [ ! -s "$frame_path" ]; then
      echo "missing or empty KITTI frame: $frame_path" >&2
      echo "re-run without --skip-fetch, or repair the subset before running VO" >&2
      exit 1
    fi
    frame=$((frame + 1))
  done
done

rm -rf "$out_dir"
mkdir -p "$out_dir"

if [ -n "$pnp_max_depth" ]; then
  pnp_depth_args="--pnp-max-depth $pnp_max_depth"
else
  pnp_depth_args=""
fi
if [ -n "$pnp_depth_hypotheses" ]; then
  pnp_depth_hypothesis_args="--pnp-depth-hypotheses $pnp_depth_hypotheses"
else
  pnp_depth_hypothesis_args=""
fi
if [ "$stereo_pose_refinement" = "1" ]; then
  stereo_refine_args="--stereo-pose-refinement"
else
  stereo_refine_args=""
fi
if [ "$stereo_vertical_alignment" = "1" ]; then
  stereo_vertical_args="--stereo-vertical-alignment"
else
  stereo_vertical_args=""
fi
if [ -n "$motion_scale_rescue_min_translation_ratio" ]; then
  motion_scale_rescue_args="--motion-scale-rescue-min-translation-ratio $motion_scale_rescue_min_translation_ratio"
else
  motion_scale_rescue_args=""
fi
if [ -n "$rotation_vector_rescue_min_history" ]; then
  rotation_vector_rescue_history_args="--rotation-vector-rescue-min-history $rotation_vector_rescue_min_history"
else
  rotation_vector_rescue_history_args=""
fi
if [ -n "$rotation_vector_rescue_max_delta_deg" ]; then
  rotation_vector_rescue_delta_args="--rotation-vector-rescue-max-delta-deg $rotation_vector_rescue_max_delta_deg"
else
  rotation_vector_rescue_delta_args=""
fi
if [ -n "$temporal_max_row_delta" ]; then
  temporal_row_args="--temporal-max-row-delta $temporal_max_row_delta"
else
  temporal_row_args=""
fi

cargo run --release --features image-io \
  --example online_slam_stereo_vo_kitti_demo -- \
  --image-left "$data_dir/image_0" \
  --image-right "$data_dir/image_1" \
  --calib "$data_dir/calib.txt" \
  --gt-poses "$data_dir/poses_${sequence}.txt" \
  --gt-original-stride 1 \
  --start-frame "$start_frame" \
  --max-frames "$max_frames" \
  --frame-stride 1 \
  --frontend deep \
  --deep-max-features "$deep_max_features" \
  --deep-descriptor-clip "$deep_descriptor_clip" \
  --deep-min-confidence "$deep_min_confidence" \
  --deep-temperature "$deep_temperature" \
  --relative-pose-iterations "$relative_pose_iterations" \
  --min-pnp-inliers "$min_pnp_inliers" \
  --pnp-reprojection-threshold "$pnp_reprojection_threshold" \
  $pnp_depth_args \
  $pnp_depth_hypothesis_args \
  $stereo_refine_args \
  $stereo_vertical_args \
  $motion_scale_rescue_args \
  $rotation_vector_rescue_history_args \
  $rotation_vector_rescue_delta_args \
  --relative-pose-mode "$relative_pose_mode" \
  $temporal_row_args \
  --progress-every "$progress_every" \
  --no-stereo-ba \
  --out-dir "$out_dir"

cargo run --example evaluate_kitti_odometry_benchmark -- \
  --out-dir "$out_dir/kitti_eval_public_lengths" \
  "$out_dir/vo_poses.txt" \
  "$out_dir/gt_poses.txt"

cargo run --example evaluate_kitti_odometry_benchmark -- \
  --lengths 100 \
  --out-dir "$out_dir/kitti_eval_100m" \
  "$out_dir/vo_poses.txt" \
  "$out_dir/gt_poses.txt"

python3 scripts/visual_slam_debug_report.py "$out_dir" --out-dir "$out_dir/slam_debug"

summary_file="$out_dir/deep_vo_smoke_summary.txt"
{
  echo "# KITTI deep stereo VO smoke summary"
  echo "sequence=$sequence"
  echo "data_dir=$data_dir"
  echo "out_dir=$out_dir"
  echo "start_frame=$start_frame"
  echo "max_frames=$max_frames"
  echo "deep_max_features=$deep_max_features"
  echo "deep_descriptor_clip=$deep_descriptor_clip"
  echo "deep_min_confidence=$deep_min_confidence"
  echo "deep_temperature=$deep_temperature"
  echo "relative_pose_iterations=$relative_pose_iterations"
  echo "min_pnp_inliers=$min_pnp_inliers"
  echo "pnp_reprojection_threshold=$pnp_reprojection_threshold"
  echo "pnp_max_depth=${pnp_max_depth:-none}"
  echo "pnp_depth_hypotheses=${pnp_depth_hypotheses:-none}"
  echo "stereo_pose_refinement=$stereo_pose_refinement"
  echo "stereo_vertical_alignment=$stereo_vertical_alignment"
  echo "motion_scale_rescue_min_translation_ratio=${motion_scale_rescue_min_translation_ratio:-default}"
  echo "rotation_vector_rescue_min_history=${rotation_vector_rescue_min_history:-default}"
  echo "rotation_vector_rescue_max_delta_deg=${rotation_vector_rescue_max_delta_deg:-default}"
  echo "relative_pose_mode=$relative_pose_mode"
  echo "temporal_max_row_delta=${temporal_max_row_delta:-none}"
  echo
  echo "## ATE"
  cat "$out_dir/summary.txt"
  echo
  echo "## KITTI public lengths"
  cat "$out_dir/kitti_eval_public_lengths/kitti_odometry_summary.json"
  echo
  echo "## KITTI 100m only"
  cat "$out_dir/kitti_eval_100m/kitti_odometry_summary.json"
  echo
  echo "## Visual SLAM debug report"
  echo "slam_debug_summary=$out_dir/slam_debug/slam_debug_summary.json"
  echo "slam_debug_report=$out_dir/slam_debug/slam_debug_report.md"
  echo "slam_debug_html=$out_dir/slam_debug/slam_debug_report.html"
  echo "slam_debug_worst_pairs=$out_dir/slam_debug/slam_debug_worst_pairs.csv"
} > "$summary_file"

echo "# Wrote $summary_file"
sed -n '1,80p' "$summary_file"
