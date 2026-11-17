#!/usr/bin/env sh
# VO + GNSS joint fusion on one KITTI raw drive, scored against the OXTS
# GPS/INS ground truth. See docs/gnss_fusion.md "Results on KITTI raw".
#
#   python3 scripts/fetch_kitti_raw_drive.py --drive 2011_09_30_drive_0027 \
#       --out ~/data/kitti_raw/2011_09_30_drive_0027 --skip-tracklets --skip-extract-oxts
#   scripts/run_kitti_raw_gnss_fusion_eval.sh --drive 2011_09_30_drive_0027 \
#       --data-root ~/data/kitti_raw/2011_09_30_drive_0027
#
# Steps: (1) rectified P0/P1 calib.txt from calib_cam_to_cam.txt; (2) the
# pure-Rust deep-frontend stereo VO (`online_slam_stereo_vo_kitti_demo
# --frontend deep`, same flags as scripts/run_kitti_deep_vo_smoke.sh) on
# image_00/image_01, skipped when <out>/vo/vo_poses.txt exists; (3) fusion
# with degraded GNSS (seeded) and with the native OXTS positions.
set -eu

drive=""
data_root=""
out_dir=""
vo_stride=1
window=300
seed=1
extra=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --drive) drive="$2"; shift 2 ;;
    --data-root) data_root="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --vo-stride) vo_stride="$2"; shift 2 ;;
    --window) window="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --) shift; extra="$*"; break ;;
    -h|--help)
      echo "usage: $0 --drive <date_drive_NNNN> --data-root <dir> [--out-dir <dir>] [--vo-stride 1] [--window 300] [--seed 1] [-- <extra fusion flags>]"
      exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[ -n "$drive" ] && [ -n "$data_root" ] || { echo "--drive and --data-root are required" >&2; exit 2; }
date=$(printf '%s' "$drive" | cut -c1-10)
date_dir="$data_root/$date"
sync_dir="$date_dir/${drive}_sync"
out_dir="${out_dir:-target/kitti_raw_gnss/$drive}"
window_frames=$(( window / vo_stride ))
mkdir -p "$out_dir/vo"

cargo build --release --features image-io \
  --example online_slam_stereo_vo_kitti_demo --example kitti_raw_gnss_vo_fusion
bin=target/release/examples

"$bin/kitti_raw_gnss_vo_fusion" --date-dir "$date_dir" --write-vo-calib "$out_dir/calib.txt"

if [ ! -f "$out_dir/vo/vo_poses.txt" ]; then
  "$bin/online_slam_stereo_vo_kitti_demo" \
    --image-left "$sync_dir/image_00/data" \
    --image-right "$sync_dir/image_01/data" \
    --calib "$out_dir/calib.txt" \
    --max-frames 99999 --frame-stride "$vo_stride" \
    --frontend deep --no-stereo-ba --progress-every 100 \
    --motion-scale-rescue-min-translation-ratio 0 \
    --out-dir "$out_dir/vo" > "$out_dir/vo.log" 2>&1
fi

for mode in degraded native; do
  # shellcheck disable=SC2086
  "$bin/kitti_raw_gnss_vo_fusion" --date-dir "$date_dir" --drive "$drive" \
    --vo-poses "$out_dir/vo/vo_poses.txt" --vo-stride "$vo_stride" \
    --gnss "$mode" --seed "$seed" --window "$window_frames" $extra \
    --out-dir "$out_dir/fusion_$mode" | tee "$out_dir/fusion_$mode.log"
done
