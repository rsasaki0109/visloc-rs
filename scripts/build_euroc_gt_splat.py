#!/usr/bin/env python3
"""Higher-quality EuRoC 3D Gaussian Splat: dense ground-truth-posed frames.

The visloc-rs-pose splat (export_euroc_colmap.py) is a sparse, low-parallax
proof-of-concept. This builds the quality ceiling for the same scene: a dense
subsample of cam0 frames posed with the EuRoC ground truth (Vicon/Leica),
covering the actual flight (real parallax), then COLMAP -> undistort -> gsplat.

GT poses give T_WB (body-in-world); the camera pose is T_WC = T_WB · T_BS
(T_BS = cam0->body extrinsic from sensor.yaml). COLMAP wants world-to-camera:
R_CW = R_WC^T, t_CW = -R_CW · t_WC. Camera is written OPENCV (radtan) so the
loader undistorts.

Runs the whole pipeline (export -> colmap image_undistorter -> gsplat train ->
.splat) and prints the final L1 to compare against the SLAM-pose baseline.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

FX, FY, CX, CY = 458.654, 457.296, 367.215, 248.375
K1, K2, P1, P2 = -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05
WIDTH, HEIGHT = 752, 480
GSMAPPER_SRC = "/media/sasaki/aiueo/ai_coding_ws/old_~2026/tier1/nerf-gs-playground/src"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--euroc", type=Path, required=True, help="EuRoC <seq>/mav0 dir")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stride", type=int, default=6, help="keep every Nth cam0 frame")
    p.add_argument("--skip-head", type=int, default=120, help="skip the first N frames (stationary takeoff)")
    p.add_argument("--iters", type=int, default=18000)
    p.add_argument("--init-points", type=int, default=20000)
    p.add_argument("--cap", type=int, default=600000, help="gsplat MCMC gaussian budget")
    return p.parse_args()


def quat_to_R(w, x, y, z):
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def R_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = (t + 1.0) ** 0.5 * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5 * 2
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5 * 2
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5 * 2
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return w, x, y, z


def main():
    args = parse_args()
    import yaml

    T_BS = np.array(yaml.safe_load((args.euroc / "cam0" / "sensor.yaml").open())["T_BS"]["data"]).reshape(4, 4)
    R_BS, t_BS = T_BS[:3, :3], T_BS[:3, 3]

    # GT: timestamp, p_RS_R_x/y/z, q_RS_w/x/y/z
    gt_ts, gt_p, gt_q = [], [], []
    with (args.euroc / "state_groundtruth_estimate0" / "data.csv").open() as fh:
        for r in csv.reader(fh):
            if r[0].startswith("#"):
                continue
            gt_ts.append(int(r[0]))
            gt_p.append([float(r[1]), float(r[2]), float(r[3])])
            gt_q.append([float(r[4]), float(r[5]), float(r[6]), float(r[7])])
    gt_p = np.array(gt_p); gt_q = np.array(gt_q)

    imgs = sorted((args.euroc / "cam0" / "data").glob("*.png"))
    imgs = imgs[args.skip_head :: args.stride]

    sparse = args.out / "sparse" / "0"
    img_out = args.out / "images"
    sparse.mkdir(parents=True, exist_ok=True)
    img_out.mkdir(parents=True, exist_ok=True)
    (sparse / "cameras.txt").write_text(
        "# Camera list\n1 OPENCV {} {} {} {} {} {} {} {} {} {}\n".format(
            WIDTH, HEIGHT, FX, FY, CX, CY, K1, K2, P1, P2))

    lines = ["# Image list\n"]
    xs = ys = zs = None
    cx_, cy_, cz_ = [], [], []
    kept = 0
    for img in imgs:
        ts = int(img.stem)
        j = bisect.bisect_left(gt_ts, ts)
        cand = [k for k in (j - 1, j) if 0 <= k < len(gt_ts)]
        if not cand:
            continue
        k = min(cand, key=lambda kk: abs(gt_ts[kk] - ts))
        if abs(gt_ts[k] - ts) > 10_000_000:  # 10 ms tolerance
            continue
        R_WB = quat_to_R(*gt_q[k])
        t_WB = gt_p[k]
        R_WC = R_WB @ R_BS
        t_WC = R_WB @ t_BS + t_WB
        R_CW = R_WC.T
        t_CW = -R_CW @ t_WC
        qw, qx, qy, qz = R_to_quat(R_CW)
        kept += 1
        cx_.append(t_WC[0]); cy_.append(t_WC[1]); cz_.append(t_WC[2])
        dst = img_out / img.name
        if not dst.exists():
            os.symlink(img.resolve(), dst)
        lines.append(f"{kept} {qw} {qx} {qy} {qz} {t_CW[0]} {t_CW[1]} {t_CW[2]} 1 {img.name}\n\n")
    (sparse / "images.txt").write_text("".join(lines))

    # random init points in the camera-centre bounding box
    import random
    random.seed(0)
    lo = [min(cx_), min(cy_), min(cz_)]
    hi = [max(cx_), max(cy_), max(cz_)]
    pad = [max((hi[i] - lo[i]) * 0.4, 0.5) for i in range(3)]
    pts = ["# 3D point list\n"]
    for pid in range(1, args.init_points + 1):
        pts.append("{} {:.4f} {:.4f} {:.4f} 128 128 128 1.0\n".format(
            pid,
            random.uniform(lo[0] - pad[0], hi[0] + pad[0]),
            random.uniform(lo[1] - pad[1], hi[1] + pad[1]),
            random.uniform(lo[2] - pad[2], hi[2] + pad[2])))
    (sparse / "points3D.txt").write_text("".join(pts))
    print(f"[export] {kept} GT-posed frames, bbox X[{lo[0]:.1f},{hi[0]:.1f}] "
          f"Y[{lo[1]:.1f},{hi[1]:.1f}] Z[{lo[2]:.1f},{hi[2]:.1f}]", flush=True)

    # undistort
    colmap_env = {**os.environ, "PATH": os.environ["HOME"] + "/.local/bin:" + os.environ["PATH"]}
    print("[undistort] running colmap image_undistorter ...", flush=True)
    subprocess.run([
        "colmap", "image_undistorter",
        "--image_path", str(img_out),
        "--input_path", str(sparse),
        "--output_path", str(args.out / "undistorted"),
        "--output_type", "COLMAP",
    ], check=True, env=colmap_env)
    # gsplat_mcmc_train.py reads a TXT model
    subprocess.run([
        "colmap", "model_converter",
        "--input_path", str(args.out / "undistorted" / "sparse"),
        "--output_path", str(args.out / "undistorted" / "sparse"),
        "--output_type", "TXT",
    ], check=True, env=colmap_env)

    # train with the official gsplat MCMC strategy (the gs-mapper hand-rolled
    # trainer fogs — see scripts/gsplat_mcmc_train.py for the why)
    splat = str(args.out / f"{args.out.name}.splat")
    print(f"[train] gsplat MCMC {args.iters} iters -> {splat}", flush=True)
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([
        sys.executable, os.path.join(here, "gsplat_mcmc_train.py"),
        str(args.out), splat, str(args.iters), str(args.cap),
    ], check=True)
    n = os.path.getsize(splat) // 32
    print(f"[done] splat={splat} ({n} gaussians, {os.path.getsize(splat)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
