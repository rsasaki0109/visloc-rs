#!/usr/bin/env python3
"""Run DROID-SLAM in STEREO mode on a KITTI odometry sequence and save the
estimated trajectory (TUM). KITTI image_0/image_1 are already rectified, so
(unlike the EuRoC eval) no cv2.remap is needed — just resize + per-axis
intrinsic scaling. Saves RAW poses (no EuRoC 1.10 scale fudge); ATE is scored
separately with evo (SE3 for the metric reading, Sim3 for scale-corrected).

Usage:
  python droid_kitti_stereo.py --left <image_0> --right <image_1> \
      --weights droid.pth --n 900 --out traj.txt [--stride 2]
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch

# KITTI seq00 rectified intrinsics (P0): fx fy cx cy, full-res 1241x376
KITTI_FX, KITTI_FY, KITTI_CX, KITTI_CY = 718.856, 718.856, 607.1928, 185.2157
WD0, HT0 = 1241, 376
IMAGE_SIZE = [192, 640]  # H, W — exact KITTI 3.33 aspect, < EuRoC mem footprint


def image_stream(left_dir, right_dir, n, stride):
    lefts = sorted(glob.glob(os.path.join(left_dir, "*.png")))[:n][::stride]
    out = []
    for t, lp in enumerate(lefts):
        rp = os.path.join(right_dir, os.path.basename(lp))
        imL = cv2.imread(lp)
        imR = cv2.imread(rp)
        imgs = [cv2.resize(im, (IMAGE_SIZE[1], IMAGE_SIZE[0])) for im in (imL, imR)]
        imgs = torch.from_numpy(np.stack(imgs, 0)).permute(0, 3, 1, 2).to(torch.float32)
        intr = torch.as_tensor([KITTI_FX, KITTI_FY, KITTI_CX, KITTI_CY])
        intr[0] *= IMAGE_SIZE[1] / WD0
        intr[1] *= IMAGE_SIZE[0] / HT0
        intr[2] *= IMAGE_SIZE[1] / WD0
        intr[3] *= IMAGE_SIZE[0] / HT0
        out.append((stride * t, imgs, intr))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="droid.pth")
    ap.add_argument("--buffer", type=int, default=512)
    ap.add_argument("--image_size", default=IMAGE_SIZE)
    ap.add_argument("--disable_vis", action="store_true", default=True)
    ap.add_argument("--stereo", action="store_true", default=True)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--filter_thresh", type=float, default=2.4)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--keyframe_thresh", type=float, default=3.0)
    ap.add_argument("--frontend_thresh", type=float, default=17.5)
    ap.add_argument("--frontend_window", type=int, default=20)
    ap.add_argument("--frontend_radius", type=int, default=2)
    ap.add_argument("--frontend_nms", type=int, default=1)
    ap.add_argument("--backend_thresh", type=float, default=24.0)
    ap.add_argument("--backend_radius", type=int, default=2)
    ap.add_argument("--backend_nms", type=int, default=2)
    ap.add_argument("--upsample", action="store_true", default=False)
    ap.add_argument("--asynchronous", action="store_true", default=False)
    ap.add_argument("--frontend_device", type=str, default="cuda")
    ap.add_argument("--backend_device", type=str, default="cuda")
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    from droid import Droid
    from tqdm import tqdm

    torch.multiprocessing.set_start_method("spawn")
    droid = Droid(args)

    images = image_stream(args.left, args.right, args.n, args.stride)
    for (t, image, intrinsics) in tqdm(images, desc="DROID-KITTI-stereo"):
        droid.track(t, image, intrinsics=intrinsics)
    traj_est = droid.terminate(images)  # (N,7) tx ty tz qx qy qz qw

    # timestamps = the strided frame indices we actually tracked
    tstamps = [stride_t for (stride_t, _, _) in images]
    with open(args.out, "w") as f:
        for ts, p in zip(tstamps, traj_est):
            f.write(f"{ts} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]}\n")
    print(f"wrote {len(traj_est)} poses -> {args.out}")


if __name__ == "__main__":
    main()
