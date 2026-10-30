#!/bin/sh
# Phase 4 of docs/object_metric_slam_plan.md: metric objects on KITTI raw
# against the tracklets, for mono + IMU, stereo + IMU and the OXTS poses.
#
#   scripts/run_kitti_objects_eval.sh WORK DRIVE [START_FRAME]
#
# DRIVE is the drive number of 2011_09_26 (e.g. 0084). The VIO starts at
# rest: pick START_FRAME at a stop (default 0). Needs release builds of
# basalt_euroc_vio_demo, objects_metric_offline and yolox_track_demo
# (features image-io,onnx-inference), YOLOX_MODEL (yolox_s.onnx) and
# ORT_DYLIB_PATH. Writes WORK/<drive>/{euroc,run_*,objects_*}.
set -eu
WORK=$1; DRIVE=$2; START=${3:-0}
REPO=$(cd "$(dirname "$0")/.." && pwd)
BIN=${BIN:-$REPO/target/release/examples}
NAME=2011_09_26_drive_$DRIVE
D=$WORK/$DRIVE
mkdir -p "$D"
python3 "$REPO/scripts/fetch_kitti_raw_drive.py" --drive "$NAME" --out "$WORK/raw"
python3 "$REPO/scripts/kitti_raw_to_euroc.py" --raw "$WORK/raw/2011_09_26" --drive "$NAME" \
  --out "$D/euroc" --start-frame "$START"
python3 "$REPO/scripts/kitti_tracklets_to_csv.py" --raw "$WORK/raw/2011_09_26" --drive "$NAME" \
  --euroc "$D/euroc" --out "$D/euroc/gt_objects.csv"
"$BIN/yolox_track_demo" --model "$YOLOX_MODEL" --images "$D/euroc/mav0/cam0/data" \
  --out-dir "$D/yolox" --boxes-csv "$D/yolox/boxes.csv" --all-classes > "$D/yolox.log" 2>&1
for mode in mono stereo; do
  flag=""; [ $mode = mono ] && flag=--mono
  mkdir -p "$D/run_$mode"
  "$BIN/basalt_euroc_vio_demo" --euroc-dir "$D/euroc" --calibration "$D/euroc/calibration.json" \
    --config "$REPO/configs/basalt/kitti/kitti_config.json" --out-dir "$D/run_$mode" --threads 4 \
    --no-trace --no-marg-data $flag --dump-tracks "$D/run_$mode/tracks.csv" > "$D/run_$mode.log" 2>&1 || true
done
for run in mono stereo gt; do
  if [ $run = gt ]; then traj=$D/euroc/gt_imu.tum; tracks=$D/run_stereo/tracks.csv
  else traj=$D/run_$run/trajectory.tum; tracks=$D/run_$run/tracks.csv; fi
  [ -f "$traj" ] || { echo "== $DRIVE $run: no trajectory"; continue; }
  mkdir -p "$D/objects_$run"
  cp "$D/yolox/boxes.csv" "$D/objects_$run/boxes.csv"
  "$BIN/objects_metric_offline" --tracks "$tracks" --trajectory "$traj" \
    --calibration "$D/euroc/calibration.json" --boxes "$D/objects_$run/boxes.csv" > "$D/objects_$run/run.txt" 2>&1
  echo "== $DRIVE $run"
  python3 "$REPO/scripts/evaluate_kitti_objects.py" --gt "$D/euroc/gt_objects.csv" \
    --boxes "$D/objects_$run/boxes.csv" --ground "$D/objects_$run/ground_objects.csv" \
    --out "$D/objects_$run/matches.csv"
done
