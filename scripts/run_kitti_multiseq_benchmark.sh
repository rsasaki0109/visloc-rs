#!/usr/bin/env sh
# Multi-sequence KITTI odometry benchmark for the full visloc-rs stereo stack.
#
# Runs the file-backed SuperPoint/LightGlue stereo VO with the complete
# accuracy stack — online window BA + fixed-prefix local-map history, the
# dynamic-object rescue chain (low-speed weak-consensus rescue arming +
# rescued-pair BA exclusion + motion-scale weak-consensus bound), and
# VLAD->PnP->GNC SE(3) loop closure with two-view loop BA + anisotropic
# loop-edge information — over one full stride-1 KITTI odometry sequence,
# then reports Umeyama SE(3)/Sim(3) ATE RMSE against the official poses.
#
# This is the per-sequence worker behind docs/kitti_multiseq_benchmark.md
# (sequences 00/02/05/06/07/09 — the loop-closure training split with
# published stereo-SLAM ATE to compare against, ORB-SLAM2 Table I and
# OV2SLAM Table V).
#
# Data layout (produced by scripts/fetch_kitti_seq00_images.py):
#   <data-root>/image_0/*.png  <data-root>/image_1/*.png
#   <data-root>/calib.txt      <data-root>/poses_<seq>.txt
# e.g.:
#   python3 scripts/fetch_kitti_seq00_images.py --sequence 05 --stride 1 \
#       --max-frames 99999 --cameras image_0,image_1 --also-fetch-poses \
#       --out-dir ~/datasets/kitti_seq05_full
#
#   scripts/run_kitti_multiseq_benchmark.sh --sequence 05 \
#       --data-root ~/datasets/kitti_seq05_full
#
# Requires: a Python env with torch + lightglue (export stage; skipped with
# --skip-export when features already exist) and numpy (evaluation).
set -eu

sequence="${KITTI_MS_SEQUENCE:-05}"
data_root=""
features_dir=""
out_dir=""
frames=""
device="${KITTI_MS_DEVICE:-cuda}"
max_keypoints="${KITTI_MS_MAX_KEYPOINTS:-2048}"
python_bin="${KITTI_MS_PYTHON:-python3}"
ba_max_init_residual="${KITTI_MS_BA_MAX_INIT_RESIDUAL:-}"
skip_export=0

usage() {
  cat <<'EOF'
usage: scripts/run_kitti_multiseq_benchmark.sh [options]

Full-stack stereo VO + loop closure on one stride-1 KITTI sequence,
evaluated as Umeyama SE(3)/Sim(3) ATE RMSE vs the official poses.

Options:
  --sequence <id>        KITTI odometry sequence (default 05)
  --data-root <dir>      dir with image_0/ image_1/ calib.txt poses_<seq>.txt
                         (default ~/datasets/kitti_seq<seq>_full)
  --features-dir <dir>   SuperPoint/LightGlue export dir
                         (default <out-dir>/external_deep)
  --out-dir <dir>        output root (default target/kitti_multiseq/seq<seq>)
  --frames <n>           frames to process (default: poses line count)
  --device <dev>         torch device for the export (default cuda)
  --ba-max-init-residual <px>
                         optional BA track init-residual gate (px); filters
                         dynamic-object tracks that survive the frontend
                         (helps 00/05/07 + EuRoC, hurts 06/09 — see the doc)
  --python <bin>         python with torch+lightglue (default python3)
  --skip-export          reuse already-exported features
  -h, --help             show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --sequence) sequence="$2"; shift 2 ;;
    --data-root) data_root="$2"; shift 2 ;;
    --features-dir) features_dir="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --frames) frames="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --ba-max-init-residual) ba_max_init_residual="$2"; shift 2 ;;
    --python) python_bin="$2"; shift 2 ;;
    --skip-export) skip_export=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

data_root="${data_root:-$HOME/datasets/kitti_seq${sequence}_full}"
out_dir="${out_dir:-target/kitti_multiseq/seq${sequence}}"
features_dir="${features_dir:-$out_dir/external_deep}"
gt_poses="$data_root/poses_${sequence}.txt"
[ -f "$gt_poses" ] || { echo "missing GT poses: $gt_poses" >&2; exit 2; }
frames="${frames:-$(wc -l < "$gt_poses" | tr -d ' ')}"

mkdir -p "$out_dir"

echo "# Building example"
cargo build --release --features image-io \
  --example stereo_vo_external_deep_files >/dev/null 2>&1

if [ "$skip_export" -ne 1 ]; then
  echo "# Exporting SuperPoint/LightGlue features (seq$sequence, $frames frames, device=$device)"
  "$python_bin" scripts/export_superpoint_lightglue.py \
    --left-dir "$data_root/image_0" \
    --right-dir "$data_root/image_1" \
    --out-dir "$features_dir" \
    --frames "$frames" \
    --device "$device" \
    --max-keypoints "$max_keypoints"
fi

echo "# Full-stack stereo VO (online BA + rescue chain + loop closure)"
# The rescue flags harden the frontend against dynamic objects:
#   --rescue-min-median-translation 0.5  arms the weak-consensus rescue
#       clamps at low urban speed (default 1.5 m/frame = highway-only);
#   --ba-exclude-rescued-pairs           keeps online BA from re-imposing a
#       rescued (rejected) motion through the same contaminated matches;
#   --motion-scale-rescue-max-inlier-ratio 0.45  restricts the motion-scale
#       rescue to genuinely weak consensus (the 1.05 default lets a
#       fast-then-decelerating stretch freeze the translation magnitude).
# All three are measured bit-identical on sequences where the gates never
# arm (KITTI seq00, EuRoC MH_03).
set -- \
  --features-dir "$features_dir" --frames "$frames" --out-dir "$out_dir/full" \
  --relative-pose-mode pnp --calib "$data_root/calib.txt" \
  --projection-left P0 --projection-right P1 \
  --min-stereo-confidence 0.5 --min-temporal-confidence 0.5 \
  --online-ba --online-ba-window 30 --online-ba-trigger-every 10 \
  --online-ba-history 20 \
  --rescue-min-median-translation 0.5 --ba-exclude-rescued-pairs \
  --motion-scale-rescue-max-inlier-ratio 0.45 \
  --loop-closure --loop-two-view-ba --loop-edge-information \
  --loop-min-frame-gap 50 --loop-min-path-length 5 --loop-min-similarity 0.2 \
  --loop-vocab-k 64 --loop-max-candidates-per-frame 3 --loop-max-verifications 400
if [ -n "$ba_max_init_residual" ]; then
  set -- "$@" --ba-max-init-residual "$ba_max_init_residual"
fi
mkdir -p "$out_dir/full"
./target/release/examples/stereo_vo_external_deep_files "$@" \
  > "$out_dir/full/vo.log" 2>&1

"$python_bin" - "$out_dir/full/vo_poses.txt" "$gt_poses" "$sequence" <<'PY'
import sys
import numpy as np


def centres(path):
    rows = []
    for ln in open(path):
        v = ln.split()
        if len(v) >= 12:
            rows.append([float(v[3]), float(v[7]), float(v[11])])
    return np.array(rows)


def umeyama_rmse(src, dst, with_scale):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    h = sc.T @ dc / len(src)
    u, d, vt = np.linalg.svd(h)
    m = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        m[2, 2] = -1
    r = vt.T @ m @ u.T
    s = (np.trace(np.diag(d) @ m) / (sc ** 2).sum() * len(src)) if with_scale else 1.0
    t = mu_d - s * r @ mu_s
    aligned = (s * (r @ src.T).T) + t
    return float(np.sqrt(((aligned - dst) ** 2).sum(1).mean()))


est, gt = centres(sys.argv[1]), centres(sys.argv[2])
n = min(len(est), len(gt))
est, gt = est[:n], gt[:n]
se3 = umeyama_rmse(est, gt, False)
sim3 = umeyama_rmse(est, gt, True)
print(f"seq{sys.argv[3]}  frames={n}  ATE rmse SE3={se3:.4f} m  Sim3={sim3:.4f} m")
PY

grep -o 'verified_loops=[0-9]*' "$out_dir/full/vo.log" | head -1 || true
