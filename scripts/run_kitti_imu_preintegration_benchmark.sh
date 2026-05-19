#!/usr/bin/env bash
# 11-seq KITTI sweep with full IMU pre-integration (Forster 2017).
# Mirrors run_kitti_sensor_prior_only_benchmark.sh structure but swaps the
# per-pose gravity prior factor for `--kitti-oxts-dir` + IMU weights.
# Reuses cached rank70-v1 SP/LG features and per-seq BA recipe.
set -euo pipefail

RANK70_ROOT="target/kitti_sp_lg_vo_train_benchmark_rank70_v1"
DATA_ROOT="$HOME/datasets/kitti_odometry_training_subsets"
WP="${IMU_WEIGHT_POSITION:-1000}"
WV="${IMU_WEIGHT_VELOCITY:-100}"
WR="${IMU_WEIGHT_ROTATION:-100}"
BRWW="${IMU_BIAS_RANDOM_WALK_WEIGHT:-1000}"
OUT_ROOT="${IMU_OUT_ROOT:-target/kitti_sp_lg_vo_train_benchmark_rank70_v1_imu_preint_p${WP}_v${WV}_r${WR}_brww${BRWW}}"
SEQUENCES="${IMU_SEQUENCES:-00 01 02 03 04 05 06 07 08 09 10}"

mkdir -p "$OUT_ROOT"

declare -A CONF_S CONF_T BA_MODE BA_RESID BA_TRACKS BA_WIN BA_HUBER
for s in 00 02 05 06 07 08 09; do CONF_S[$s]=0.5; CONF_T[$s]=0.5; done
CONF_S[01]=0.7; CONF_T[01]=0.7
CONF_S[03]=0.7; CONF_T[03]=0.7
CONF_S[04]=0.5; CONF_T[04]=0.7
CONF_S[10]=0.9; CONF_T[10]=0.7

BA_MODE[00]=skip; BA_MODE[05]=skip
for s in 01 02 03 04 06 07 08 09 10; do BA_MODE[$s]=full; done

BA_RESID[01]=3; BA_TRACKS[01]=200; BA_WIN[01]=30; BA_HUBER[01]=1.5
BA_RESID[02]=8; BA_TRACKS[02]=2000; BA_WIN[02]=; BA_HUBER[02]=3
BA_RESID[03]=3; BA_TRACKS[03]=200; BA_WIN[03]=50; BA_HUBER[03]=1.5
BA_RESID[04]=5; BA_TRACKS[04]=2000; BA_WIN[04]=; BA_HUBER[04]=3
BA_RESID[06]=8; BA_TRACKS[06]=2000; BA_WIN[06]=; BA_HUBER[06]=3
BA_RESID[07]=8; BA_TRACKS[07]=2000; BA_WIN[07]=; BA_HUBER[07]=3
BA_RESID[08]=3; BA_TRACKS[08]=2000; BA_WIN[08]=; BA_HUBER[08]=3
BA_RESID[09]=5; BA_TRACKS[09]=2000; BA_WIN[09]=; BA_HUBER[09]=3
BA_RESID[10]=8; BA_TRACKS[10]=2000; BA_WIN[10]=; BA_HUBER[10]=3

for s in $SEQUENCES; do
  seq_out="$OUT_ROOT/seq${s}"
  mkdir -p "$seq_out"
  features="$RANK70_ROOT/seq${s}/external_deep"
  gt="$RANK70_ROOT/seq${s}/gt_poses.txt"
  oxts="target/kitti_raw_oxts_seq${s}_260_odom_aligned/oxts"
  ts="target/kitti_raw_oxts_seq${s}_260_odom_aligned/image_00/timestamps.txt"
  calib="$DATA_ROOT/seq${s}/calib.txt"

  cp "$gt" "$seq_out/gt_poses.txt"

  if [ "${BA_MODE[$s]}" = "skip" ]; then
    echo "=== seq${s}: BA skipped — copying rank70-v1 vo_poses (no IMU effect) ==="
    cp "$RANK70_ROOT/seq${s}/vo_poses.txt" "$seq_out/vo_poses.txt"
  else
    ba_args=(
      --enable-ba
      --ba-max-init-residual "${BA_RESID[$s]}"
      --ba-min-track-count "${BA_TRACKS[$s]}"
      --ba-huber-delta "${BA_HUBER[$s]}"
      --kitti-oxts-dir "$oxts"
      --kitti-image-timestamps "$ts"
      --imu-gravity "0,9.81,0"
      --imu-weight-position "$WP"
      --imu-weight-velocity "$WV"
      --imu-weight-rotation "$WR"
      --imu-bias-random-walk-weight "$BRWW"
      --imu-fix-first-bias on
      --imu-fix-first-velocity on
    )
    if [ -n "${BA_WIN[$s]}" ]; then
      ba_args+=(--ba-window-size "${BA_WIN[$s]}")
    fi
    echo "=== seq${s}: BA full + IMU PI (p=$WP v=$WV r=$WR brww=$BRWW) ==="
    target/release/examples/stereo_vo_external_deep_files \
      --features-dir "$features" \
      --frames 260 \
      --out-dir "$seq_out" \
      --relative-pose-mode pnp \
      --calib "$calib" \
      --projection-left P0 --projection-right P1 \
      --min-stereo-confidence "${CONF_S[$s]}" \
      --min-temporal-confidence "${CONF_T[$s]}" \
      "${ba_args[@]}" \
      > "$seq_out/vo.log" 2>&1
  fi

  target/release/examples/evaluate_kitti_odometry_benchmark \
    --out-dir "$seq_out/kitti_eval" \
    "$seq_out/vo_poses.txt" \
    "$seq_out/gt_poses.txt" \
    > "$seq_out/kitti_eval.log" 2>&1
done

python3 - "$OUT_ROOT" "$RANK70_ROOT" $SEQUENCES <<'PY'
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
rank70_root = Path(sys.argv[2])
sequences = sys.argv[3:]
rows = []
for seq in sequences:
    cur = json.loads((out_root / f"seq{seq}" / "kitti_eval" / "kitti_odometry_summary.json").read_text())
    base = json.loads((rank70_root / f"seq{seq}" / "kitti_eval" / "kitti_odometry_summary.json").read_text())
    rows.append({
        "seq": seq,
        "t_rel": cur["mean_translational_error_percent"],
        "max_t": cur["max_translational_error_percent"],
        "r_rel": cur["mean_rotational_error_deg_per_m"],
        "base_t": base["mean_translational_error_percent"],
        "base_max_t": base["max_translational_error_percent"],
        "base_r": base["mean_rotational_error_deg_per_m"],
    })

def avg(key):
    return sum(r[key] for r in rows) / len(rows)

lines = ["# seq | mean_t_rel% | max_t_rel% | mean_r_deg/m | (vs rank70-v1)"]
for r in rows:
    lines.append(
        f"  {r['seq']} | {r['t_rel']:.4f} | {r['max_t']:.4f} | {r['r_rel']:.6f}  "
        f"(base: {r['base_t']:.4f}, {r['base_max_t']:.4f}, {r['base_r']:.6f})"
    )
lines.append("")
lines.append(
    f"AGGREGATE 00-10 (n={len(rows)}):  mean_t={avg('t_rel'):.6f}%  mean_max_t={avg('max_t'):.6f}%  "
    f"mean_r={avg('r_rel'):.6f} deg/m"
)
lines.append(
    f"RANK70-V1 BASELINE          :  mean_t={avg('base_t'):.6f}%  mean_max_t={avg('base_max_t'):.6f}%  "
    f"mean_r={avg('base_r'):.6f} deg/m"
)
lines.append(
    f"DELTA (IMU PI - rank70):       mean_t={avg('t_rel')-avg('base_t'):+.6f}pp  "
    f"mean_max_t={avg('max_t')-avg('base_max_t'):+.6f}pp  "
    f"mean_r={avg('r_rel')-avg('base_r'):+.6f}"
)
output = "\n".join(lines) + "\n"
(out_root / "aggregate.txt").write_text(output)
print(output, end="")
PY
