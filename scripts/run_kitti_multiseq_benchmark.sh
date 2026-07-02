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
capture_retrieval_recall="${KITTI_MS_CAPTURE_RETRIEVAL_RECALL:-0}"
retrieval_distance_threshold="${KITTI_MS_RETRIEVAL_DISTANCE_THRESHOLD:-10}"
retrieval_ks="${KITTI_MS_RETRIEVAL_KS:-1 5 20}"
retrieval_registry_dir="${KITTI_MS_RETRIEVAL_REGISTRY_DIR:-benchmarks/registry/runs/kitti}"
retrieval_dnf_if_recall_at="${KITTI_MS_RETRIEVAL_DNF_IF_RECALL_AT:-}"
capture_run_registry="${KITTI_MS_CAPTURE_RUN_REGISTRY:-0}"
run_registry_dir="${KITTI_MS_RUN_REGISTRY_DIR:-benchmarks/registry/runs/kitti}"
loop_matches_dir="${KITTI_MS_LOOP_MATCHES_DIR:-}"
loop_pnp_essential_inliers="${KITTI_MS_LOOP_PNP_ESSENTIAL_INLIERS:-0}"
loop_pnp_confidence_weights="${KITTI_MS_LOOP_PNP_CONFIDENCE_WEIGHTS:-0}"
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
  --capture-retrieval-recall
                         evaluate full/loop_candidates.csv and capture a
                         benchmark-registry manifest for seq02-style recall
                         diagnosis (default off)
  --retrieval-distance-threshold <m>
                         true-revisit pose distance for recall capture
                         (default 10)
  --retrieval-ks "<k...>"
                         recall@K values for recall capture (default "1 5 20")
  --retrieval-registry-dir <dir>
                         registry output dir for recall capture
                         (default benchmarks/registry/runs/kitti)
  --retrieval-dnf-if-recall-at <K=VALUE>
                         optional recall gate, e.g. 20=0.01
  --capture-run-registry
                         capture ATE/loop metrics and trajectory artifacts as
                         a full KITTI run benchmark-registry manifest
                         (default off)
  --run-registry-dir <dir>
                         registry output dir for full run capture
                         (default benchmarks/registry/runs/kitti)
  --loop-matches-dir <dir>
                         optional external loop_OLDER_NEWER_matches.txt files
                         for loop-verifier matching A/B
  --loop-pnp-essential-inliers
                         send only essential-matrix inlier matches to PnP
                         during loop verification (default off)
  --loop-pnp-confidence-weights
                         bias loop PnP RANSAC sampling by descriptor-match
                         confidence when enough non-uniform confidences are
                         available (default off)
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
    --capture-retrieval-recall) capture_retrieval_recall=1; shift ;;
    --retrieval-distance-threshold) retrieval_distance_threshold="$2"; shift 2 ;;
    --retrieval-ks) retrieval_ks="$2"; shift 2 ;;
    --retrieval-registry-dir) retrieval_registry_dir="$2"; shift 2 ;;
    --retrieval-dnf-if-recall-at) retrieval_dnf_if_recall_at="$2"; shift 2 ;;
    --capture-run-registry) capture_run_registry=1; shift ;;
    --run-registry-dir) run_registry_dir="$2"; shift 2 ;;
    --loop-matches-dir) loop_matches_dir="$2"; shift 2 ;;
    --loop-pnp-essential-inliers) loop_pnp_essential_inliers=1; shift ;;
    --loop-pnp-confidence-weights) loop_pnp_confidence_weights=1; shift ;;
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
if [ -n "$loop_matches_dir" ]; then
  set -- "$@" --loop-matches-dir "$loop_matches_dir"
fi
if [ "$loop_pnp_essential_inliers" -eq 1 ]; then
  set -- "$@" --loop-pnp-essential-inliers
fi
if [ "$loop_pnp_confidence_weights" -eq 1 ]; then
  set -- "$@" --loop-pnp-confidence-weights
fi
mkdir -p "$out_dir/full"
./target/release/examples/stereo_vo_external_deep_files "$@" \
  > "$out_dir/full/vo.log" 2>&1

evaluation_json="$out_dir/full/evaluation.json"
"$python_bin" - "$out_dir/full/vo_poses.txt" "$gt_poses" "$sequence" "$evaluation_json" <<'PY'
import json
import sys
from pathlib import Path

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
Path(sys.argv[4]).write_text(
    json.dumps(
        {
            "sequence": sys.argv[3],
            "frames": int(n),
            "ate_rmse_se3_m": se3,
            "ate_rmse_sim3_m": sim3,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

grep -o 'verified_loops=[0-9]*' "$out_dir/full/vo.log" | head -1 || true

if [ "$capture_run_registry" -eq 1 ]; then
  echo "# Capturing KITTI full-run registry manifest"
  run_claim_scope="supporting"
  if [ -n "$loop_matches_dir" ] || [ "$loop_pnp_essential_inliers" -eq 1 ] || [ "$loop_pnp_confidence_weights" -eq 1 ]; then
    run_claim_scope="exploratory"
  fi
  run_registry_command="scripts/run_kitti_multiseq_benchmark.sh --sequence $sequence --data-root $data_root --out-dir $out_dir --features-dir $features_dir --frames $frames --device $device"
  if [ "$skip_export" -eq 1 ]; then
    run_registry_command="$run_registry_command --skip-export"
  fi
  if [ -n "$ba_max_init_residual" ]; then
    run_registry_command="$run_registry_command --ba-max-init-residual $ba_max_init_residual"
  fi
  if [ -n "$loop_matches_dir" ]; then
    run_registry_command="$run_registry_command --loop-matches-dir $loop_matches_dir"
  fi
  if [ "$loop_pnp_essential_inliers" -eq 1 ]; then
    run_registry_command="$run_registry_command --loop-pnp-essential-inliers"
  fi
  if [ "$loop_pnp_confidence_weights" -eq 1 ]; then
    run_registry_command="$run_registry_command --loop-pnp-confidence-weights"
  fi
  "$python_bin" scripts/capture_kitti_multiseq_run.py \
    --sequence "$sequence" \
    --evaluation-json "$evaluation_json" \
    --vo-log "$out_dir/full/vo.log" \
    --vo-poses "$out_dir/full/vo_poses.txt" \
    --poses "$gt_poses" \
    --dataset-path "$data_root" \
    --features-dir "$features_dir" \
    --out-dir "$out_dir/full" \
    --registry-dir "$run_registry_dir" \
    --claim-scope "$run_claim_scope" \
    --command "$run_registry_command" \
    --config "device=$device" \
    --config "max_keypoints=$max_keypoints" \
    --config "skip_export=$skip_export" \
    --config "ba_max_init_residual=${ba_max_init_residual:-null}" \
    --config "loop_matches_dir=${loop_matches_dir:-null}" \
    --config "loop_pnp_essential_inliers=$loop_pnp_essential_inliers" \
    --config "loop_pnp_confidence_weights=$loop_pnp_confidence_weights"
fi

if [ "$capture_retrieval_recall" -eq 1 ]; then
  candidates_csv="$out_dir/full/loop_candidates.csv"
  [ -s "$candidates_csv" ] || {
    echo "missing loop candidate CSV for retrieval recall capture: $candidates_csv" >&2
    exit 2
  }

  echo "# Capturing KITTI loop retrieval recall"
  if [ -n "$retrieval_dnf_if_recall_at" ]; then
    # shellcheck disable=SC2086
    "$python_bin" scripts/capture_kitti_loop_retrieval_recall.py \
      --sequence "$sequence" \
      --candidates "$candidates_csv" \
      --poses "$gt_poses" \
      --dataset-path "$data_root" \
      --distance-threshold-m "$retrieval_distance_threshold" \
      --min-temporal-gap 50 \
      --min-path-length-m 5 \
      --ks $retrieval_ks \
      --out-dir "$out_dir/full/retrieval_recall" \
      --registry-dir "$retrieval_registry_dir" \
      --dnf-if-recall-at "$retrieval_dnf_if_recall_at"
  else
    # shellcheck disable=SC2086
    "$python_bin" scripts/capture_kitti_loop_retrieval_recall.py \
      --sequence "$sequence" \
      --candidates "$candidates_csv" \
      --poses "$gt_poses" \
      --dataset-path "$data_root" \
      --distance-threshold-m "$retrieval_distance_threshold" \
      --min-temporal-gap 50 \
      --min-path-length-m 5 \
      --ks $retrieval_ks \
      --out-dir "$out_dir/full/retrieval_recall" \
      --registry-dir "$retrieval_registry_dir"
  fi
fi
