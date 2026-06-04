#!/usr/bin/env sh
# Metric loop-closure benchmark on a loopy KITTI odometry sequence.
#
# A streaming stereo-VO frontend produces an *open* trajectory: only the first
# pose is gauge-fixed, so drift accumulates unbounded and dense global bundle
# adjustment cannot remove it (a loop-free reprojection minimum just deforms the
# path). A revisited place is the one constraint that pulls the drift back. This
# script measures exactly that: it runs the file-backed SuperPoint/LightGlue
# stereo VO twice over the SAME exported features — once open, once with VLAD
# appearance loop detection + PnP verification + robust GNC SE(3) pose-graph
# optimization (`--loop-closure`) — and reports ATE before vs after.
#
# KITTI seq00 is the canonical loopy sequence (revisits its start near the end).
# Default data is the stride-2 subset (2271 frames) prepared by
# scripts/fetch_kitti_seq00_images.py; point --data-root at a full stride-1
# sequence for the dense version.
#
#   scripts/run_kitti_loop_closure_benchmark.sh \
#     --data-root ~/datasets/kitti_seq00_full2 \
#     --gt-poses  ~/datasets/kitti_seq00_full2/poses_00_stride2.txt \
#     --frames 2271
#
# Requires: a Python env with torch + lightglue (export stage), and the built
# `stereo_vo_external_deep_files` / `evaluate_trajectory_from_kitti_files`
# examples (the script builds them).
set -eu

data_root="${KITTI_LC_DATA_ROOT:-$HOME/datasets/kitti_seq00_full2}"
gt_poses="${KITTI_LC_GT_POSES:-$data_root/poses_00_stride2.txt}"
out_dir="${KITTI_LC_OUT_DIR:-target/kitti_loop_closure_benchmark}"
frames="${KITTI_LC_FRAMES:-2271}"
start_frame="${KITTI_LC_START_FRAME:-0}"
frame_stride="${KITTI_LC_FRAME_STRIDE:-1}"
device="${KITTI_LC_DEVICE:-cuda}"
max_keypoints="${KITTI_LC_MAX_KEYPOINTS:-2048}"
projection_left="${KITTI_LC_PROJECTION_LEFT:-P0}"
projection_right="${KITTI_LC_PROJECTION_RIGHT:-P1}"
min_stereo_confidence="${KITTI_LC_MIN_STEREO_CONFIDENCE:-0.5}"
min_temporal_confidence="${KITTI_LC_MIN_TEMPORAL_CONFIDENCE:-0.5}"
loop_min_frame_gap="${KITTI_LC_LOOP_MIN_FRAME_GAP:-50}"
loop_min_path_length="${KITTI_LC_LOOP_MIN_PATH_LENGTH:-5}"
loop_min_similarity="${KITTI_LC_LOOP_MIN_SIMILARITY:-0.2}"
loop_vocab_k="${KITTI_LC_LOOP_VOCAB_K:-64}"
loop_max_candidates="${KITTI_LC_LOOP_MAX_CANDIDATES:-3}"
loop_max_verifications="${KITTI_LC_LOOP_MAX_VERIFICATIONS:-400}"
python_bin="${KITTI_LC_PYTHON:-python3}"
skip_export=0

usage() {
  cat <<'EOF'
usage: scripts/run_kitti_loop_closure_benchmark.sh [options]

Measure ATE before/after metric loop closure on a loopy KITTI sequence.

Options:
  --data-root <dir>          seqXX dir with image_0, image_1, calib.txt
  --gt-poses <file>          KITTI 3x4 GT poses matching the exported frames
  --out-dir <dir>            benchmark output root
  --frames <n>               frames to process (default 2271)
  --start-frame <n>          start offset (default 0)
  --frame-stride <n>         frame stride into the image dirs (default 1)
  --device <auto|cpu|cuda>   torch device for the export (default cuda)
  --max-keypoints <n>        SuperPoint cap (default 2048)
  --projection-left <label>  KITTI left projection (default P0)
  --projection-right <label> KITTI right projection (default P1)
  --loop-min-frame-gap <n>   min frame gap of a loop pair (default 50)
  --loop-min-path-length <m> min accumulated travel between a loop pair, the
                             frame-rate-independent gate (default 5)
  --loop-min-similarity <x>  min VLAD cosine similarity (default 0.2)
  --python <bin>             python with torch+lightglue (default python3)
  --skip-export              reuse already-exported features
  -h, --help                 show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --data-root) data_root="$2"; shift 2 ;;
    --gt-poses) gt_poses="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --frames) frames="$2"; shift 2 ;;
    --start-frame) start_frame="$2"; shift 2 ;;
    --frame-stride) frame_stride="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --max-keypoints) max_keypoints="$2"; shift 2 ;;
    --projection-left) projection_left="$2"; shift 2 ;;
    --projection-right) projection_right="$2"; shift 2 ;;
    --loop-min-frame-gap) loop_min_frame_gap="$2"; shift 2 ;;
    --loop-min-path-length) loop_min_path_length="$2"; shift 2 ;;
    --loop-min-similarity) loop_min_similarity="$2"; shift 2 ;;
    --python) python_bin="$2"; shift 2 ;;
    --skip-export) skip_export=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

export_dir="$out_dir/external_deep"
mkdir -p "$out_dir"

echo "# Building examples"
cargo build --release \
  --example stereo_vo_external_deep_files \
  --example evaluate_trajectory_from_kitti_files >/dev/null 2>&1

if [ "$skip_export" -ne 1 ]; then
  echo "# Exporting SuperPoint/LightGlue features ($frames frames, device=$device)"
  "$python_bin" scripts/export_superpoint_lightglue.py \
    --left-dir "$data_root/image_0" \
    --right-dir "$data_root/image_1" \
    --out-dir "$export_dir" \
    --start-frame "$start_frame" \
    --frame-stride "$frame_stride" \
    --frames "$frames" \
    --device "$device" \
    --max-keypoints "$max_keypoints"
fi

# Slice the GT poses to the processed frame window.
head -n "$frames" "$gt_poses" > "$out_dir/gt_poses.txt"

run_vo() {
  # run_vo <subdir> [extra VO args...]
  sub="$1"; shift
  mkdir -p "$out_dir/$sub"
  ./target/release/examples/stereo_vo_external_deep_files \
    --features-dir "$export_dir" \
    --frames "$frames" \
    --out-dir "$out_dir/$sub" \
    --relative-pose-mode pnp \
    --calib "$data_root/calib.txt" \
    --projection-left "$projection_left" \
    --projection-right "$projection_right" \
    --min-stereo-confidence "$min_stereo_confidence" \
    --min-temporal-confidence "$min_temporal_confidence" \
    "$@" > "$out_dir/$sub/vo.log" 2>&1
  ./target/release/examples/evaluate_trajectory_from_kitti_files \
    --out-dir "$out_dir/$sub/ate" --align-origin \
    "$out_dir/$sub/vo_poses.txt" "$out_dir/gt_poses.txt" > "$out_dir/$sub/ate.log" 2>&1
}

echo "# Open VO (no loop closure)"
run_vo open

echo "# Loop-closure VO (VLAD -> PnP -> GNC SE(3) PGO)"
run_vo loop \
  --loop-closure \
  --loop-min-frame-gap "$loop_min_frame_gap" \
  --loop-min-path-length "$loop_min_path_length" \
  --loop-min-similarity "$loop_min_similarity" \
  --loop-vocab-k "$loop_vocab_k" \
  --loop-max-candidates-per-frame "$loop_max_candidates" \
  --loop-max-verifications "$loop_max_verifications"

"$python_bin" - "$out_dir" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])

# GT camera centres from the KITTI 3x4 poses sliced to the frame window.
gt = []
for ln in (root / "gt_poses.txt").read_text().splitlines():
    v = list(map(float, ln.split()))
    if len(v) == 12:
        gt.append((v[3], v[7], v[11]))


def read_centres(path):
    xyz = []
    for ln in path.read_text().splitlines():
        if ln.startswith("id") or not ln.strip():
            continue
        v = ln.split(",")
        if len(v) >= 4:
            xyz.append((float(v[1]), float(v[2]), float(v[3])))
    return xyz


def umeyama_rmse(src, dst):
    # Sim(3) alignment src->dst (rotation, uniform scale, translation), then RMSE.
    # Isolates trajectory *shape* consistency by absorbing a single global
    # scale/rotation/offset, so a metric frontend's residual scale drift does
    # not dominate the number. We also report origin-only alignment below.
    n = min(len(src), len(dst))
    import math

    sx = [src[i] for i in range(n)]
    dx = [dst[i] for i in range(n)]
    mus = [sum(c[k] for c in sx) / n for k in range(3)]
    mud = [sum(c[k] for c in dx) / n for k in range(3)]
    S = [[c[k] - mus[k] for k in range(3)] for c in sx]
    D = [[c[k] - mud[k] for k in range(3)] for c in dx]
    cov = [[sum(D[i][r] * S[i][c] for i in range(n)) / n for c in range(3)] for r in range(3)]
    # 3x3 SVD via numpy if available, else a small power-free fallback is overkill;
    # numpy is a hard dep of the export stack, so use it.
    import numpy as np

    U, d, Vt = np.linalg.svd(np.array(cov))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    var = sum(sum(v * v for v in row) for row in S) / n
    s = float(np.trace(np.diag(d)) / var) if var > 0 else 1.0
    Sm = np.array(S)
    Dm = np.array(D)
    al = (s * (R @ Sm.T)).T
    return float(np.sqrt(((al - Dm) ** 2).sum(1).mean()))


def origin_ate(sub):
    d = json.loads((root / sub / "ate" / "error_summary.json").read_text())
    return d["rmse_translation_error"], d["mean_translation_error"], d["max_translation_error"]


o_o = origin_ate("open")
l_o = origin_ate("loop")
o_sim3 = umeyama_rmse(read_centres(root / "open" / "vo.csv"), gt)
l_sim3 = umeyama_rmse(read_centres(root / "loop" / "vo.csv"), gt)

loop_log = (root / "loop" / "vo.log").read_text()
verified = next((ln for ln in loop_log.splitlines() if "LOOP-CLOSURE PGO: candidates" in ln), "")

lines = [
    "",
    "# KITTI loop-closure benchmark",
    "",
    "Same exported SuperPoint/LightGlue features, run twice: open VO vs",
    "+ VLAD->PnP->GNC SE(3) pose-graph loop closure. ATE under two alignments:",
    "Sim(3) (absorbs global scale/rotation/offset, isolates shape) and",
    "origin-only (gauge-fixed at frame 0, the metric-frame number).",
    "",
    "| trajectory     | ATE rmse Sim(3) | ATE rmse origin | ATE mean origin | ATE max origin |",
    "| -------------- | --------------: | --------------: | --------------: | -------------: |",
    f"| open VO        | {o_sim3:15.3f} | {o_o[0]:15.3f} | {o_o[1]:15.3f} | {o_o[2]:14.3f} |",
    f"| + loop closure | {l_sim3:15.3f} | {l_o[0]:15.3f} | {l_o[1]:15.3f} | {l_o[2]:14.3f} |",
    "",
    f"Sim(3) ATE rmse improvement: {o_sim3 / l_sim3:.2f}x  ({o_sim3:.2f} m -> {l_sim3:.2f} m)",
    verified.strip(),
]
text = "\n".join(lines) + "\n"
(root / "summary.md").write_text(text)
print(text)
PY
