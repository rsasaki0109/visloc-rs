#!/usr/bin/env python3
"""Rectify an EuRoC MAV stereo sequence into pinhole rectified images for visloc-rs.

EuRoC's `cam0`/`cam1` images are radial-tangential distorted (unlike the
pre-rectified KITTI odometry images), so the file-backed SuperPoint/LightGlue
stereo-VO path (`examples/stereo_vo_external_deep_files.rs`, which assumes a
pinhole model) needs an undistort+rectify step first. This helper reads the two
`mav0/camN/sensor.yaml` calibrations, computes the stereo rectification with
OpenCV, and writes:

    <out>/image_0/000000.png ...   rectified left  (pinhole)
    <out>/image_1/000000.png ...   rectified right (pinhole)
    <out>/calib.txt                KITTI-format P0/P1 (read by --calib)
    <out>/timestamps.txt           "<frame_idx> <cam0_timestamp_ns>" per line

Then run the existing pipeline on `<out>`:

    python scripts/export_superpoint_lightglue.py \\
        --left-dir <out>/image_0 --right-dir <out>/image_1 \\
        --out-dir <feat> --device cuda --max-keypoints 2048
    cargo run --release --example stereo_vo_external_deep_files -- \\
        --features-dir <feat> --calib <out>/calib.txt \\
        --baseline <B from calib.txt> --relative-pose-mode pnp \\
        --loop-closure --loop-min-frame-gap 200 --out-dir <vo>

The `timestamps.txt` lets you convert the frame-indexed KITTI poses back to a
timestamped TUM trajectory for `evo_ape` association against the EuRoC ground
truth (`state_groundtruth_estimate0/data.csv`).

NOTE on `--loop-min-frame-gap`: it must be large enough that meaningful drift
accumulates between a loop's two frames. At EuRoC's 20 Hz a small gap (e.g. 30 =
1.5 s) only matches slow-motion temporal neighbours that are already
odometry-consistent and contribute no drift correction; 200 (10 s) catches
genuine revisits. Measured on MH_03_medium: gap=30 left ATE unchanged (2.46 m),
gap=200 cut it to 0.46 m.

Requires `opencv-python` and `numpy`.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import cv2
import numpy as np


def _load_cam(path: str):
    """Parse intrinsics, radtan distortion, T_BS and resolution from a
    `mav0/camN/sensor.yaml` (no YAML dep — the EuRoC files are flat enough to
    regex)."""
    text = open(path).read()

    def inline_array(key: str):
        m = re.search(key + r"\s*:\s*\[([^\]]+)\]", text)
        if not m:
            raise ValueError(f"{key} not found in {path}")
        return [float(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]

    md = re.search(r"T_BS:.*?data:\s*\[(.*?)\]", text, re.S)
    t_bs = np.array(
        [float(x) for x in md.group(1).replace("\n", " ").split(",")]
    ).reshape(4, 4)
    fu, fv, cu, cv = inline_array("intrinsics")
    k = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]])
    d = np.array(inline_array("distortion_coefficients"))[:4]
    res = inline_array("resolution")
    return k, d, t_bs, (int(res[0]), int(res[1]))


def _kitti_p_line(p: np.ndarray) -> str:
    return " ".join(f"{v:.12e}" for v in p[:3, :4].reshape(-1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mav0", required=True, help="path to the EuRoC mav0/ directory")
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="stereoRectify free-scaling: 0=crop to valid pixels "
                         "(no black border, recommended for VO), 1=keep all")
    args = ap.parse_args()

    m = args.mav0
    out = args.out_dir
    os.makedirs(f"{out}/image_0", exist_ok=True)
    os.makedirs(f"{out}/image_1", exist_ok=True)

    k0, d0, t0, size = _load_cam(f"{m}/cam0/sensor.yaml")
    k1, d1, t1, _ = _load_cam(f"{m}/cam1/sensor.yaml")

    # T_BS is sensor->body; point_cam1 = T_S1_S0 @ point_cam0.
    t_s1_s0 = np.linalg.inv(t1) @ t0
    rot = t_s1_s0[:3, :3]
    trans = t_s1_s0[:3, 3]

    r0, r1, p0, p1, _q, _, _ = cv2.stereoRectify(
        k0, d0, k1, d1, size, rot, trans,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=args.alpha,
    )
    m0x, m0y = cv2.initUndistortRectifyMap(k0, d0, r0, p0, size, cv2.CV_16SC2)
    m1x, m1y = cv2.initUndistortRectifyMap(k1, d1, r1, p1, size, cv2.CV_16SC2)
    fx = p0[0, 0]
    baseline = -p1[0, 3] / fx
    print(f"rectified fx={fx:.4f} cx={p0[0,2]:.4f} cy={p0[1,2]:.4f} "
          f"baseline={baseline:.6f}m size={size}", flush=True)

    with open(f"{out}/calib.txt", "w") as f:
        f.write(f"P0: {_kitti_p_line(p0)}\nP1: {_kitti_p_line(p1)}\n")

    left = sorted(os.listdir(f"{m}/cam0/data"))
    right = sorted(os.listdir(f"{m}/cam1/data"))
    if len(left) != len(right):
        print(f"ERROR cam0/cam1 frame count mismatch: {len(left)} vs {len(right)}",
              file=sys.stderr)
        return 1

    with open(f"{out}/timestamps.txt", "w") as ts:
        for i, (a, b) in enumerate(zip(left, right)):
            ia = cv2.imread(f"{m}/cam0/data/{a}", cv2.IMREAD_GRAYSCALE)
            ib = cv2.imread(f"{m}/cam1/data/{b}", cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(f"{out}/image_0/{i:06d}.png",
                        cv2.remap(ia, m0x, m0y, cv2.INTER_LINEAR))
            cv2.imwrite(f"{out}/image_1/{i:06d}.png",
                        cv2.remap(ib, m1x, m1y, cv2.INTER_LINEAR))
            ts.write(f"{i:06d} {os.path.splitext(a)[0]}\n")
            if (i + 1) % 500 == 0:
                print(f"  rectified {i + 1}/{len(left)}", flush=True)

    print(f"DONE rectified {len(left)} stereo pairs -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
