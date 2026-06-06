#!/usr/bin/env python3
"""Export undistorted SuperPoint features for a flat image directory, using a
COLMAP cameras.txt for the intrinsics. Writes one `<stem>_features.txt` per
image in the `X Y SCORE D0 D1 ...` format visloc-rs's
`read_external_deep_features_txt` consumes, with keypoints undistorted to the
ideal pinhole so the demo can treat the camera as pinhole.

Supported camera models: SIMPLE_PINHOLE / PINHOLE (no distortion),
SIMPLE_RADIAL (one-parameter radial, inverted analytically) and OPENCV
(k1,k2,p1,p2 radial+tangential, inverted with cv2.undistortPoints). The
pinhole the demo should use is printed as `--fx --fy --cx --cy`."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from lightglue import SuperPoint
from lightglue.utils import load_image


def read_camera(cameras_txt):
    for ln in open(cameras_txt):
        if ln.startswith("#") or not ln.strip():
            continue
        v = ln.split()
        model = v[1]
        w, h = int(v[2]), int(v[3])
        p = list(map(float, v[4:]))
        return model, w, h, p
    raise RuntimeError("no camera in " + str(cameras_txt))


def undistort_simple_radial(uv, f, cx, cy, k1):
    """SIMPLE_RADIAL: distorted = undist*(1+k1*r_u^2). Invert iteratively."""
    xd = (uv[:, 0] - cx) / f
    yd = (uv[:, 1] - cy) / f
    xu, yu = xd.copy(), yd.copy()
    for _ in range(10):
        r2 = xu * xu + yu * yu
        d = 1.0 + k1 * r2
        xu = xd / d
        yu = yd / d
    return np.stack([f * xu + cx, f * yu + cy], axis=1)


def undistort_opencv(uv, fx, fy, cx, cy, k1, k2, p1, p2):
    """OPENCV (k1,k2,p1,p2): invert with cv2.undistortPoints, mapping back into
    the same pinhole K so the output keypoints are ideal-pinhole pixels."""
    import cv2

    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    dist = np.array([k1, k2, p1, p2])
    pts = uv.reshape(-1, 1, 2).astype(np.float64)
    out = cv2.undistortPoints(pts, k, dist, P=k)
    return out.reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--cameras-txt", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-keypoints", type=int, default=2048)
    ap.add_argument("--suffix", default="_features.txt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    extractor = SuperPoint(max_num_keypoints=args.max_keypoints).eval().to(dev)

    model = None
    fx = fy = cx = cy = None
    opencv_dist = None  # (k1,k2,p1,p2) when model == OPENCV
    k1 = 0.0
    if args.cameras_txt:
        model, w, h, p = read_camera(args.cameras_txt)
        if model == "SIMPLE_RADIAL":
            fx = fy = p[0]
            cx, cy, k1 = p[1], p[2], p[3]
        elif model == "PINHOLE":
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        elif model == "SIMPLE_PINHOLE":
            fx = fy = p[0]
            cx, cy = p[1], p[2]
        elif model == "OPENCV":
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
            opencv_dist = (p[4], p[5], p[6], p[7])
        else:
            print(f"warning: camera model {model} not handled; no undistortion", file=sys.stderr)
        print(f"camera {model} {w}x{h} fx={fx} fy={fy} cx={cx} cy={cy} "
              f"k1={k1} opencv_dist={opencv_dist}")
        print(f"  -> demo pinhole: --width {w} --height {h} --fx {fx} --fy {fy} "
              f"--cx {cx} --cy {cy}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted([p for p in args.images_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    print(f"{len(imgs)} images -> {args.out_dir}")
    for i, ip in enumerate(imgs):
        image = load_image(ip).to(dev)
        with torch.no_grad():
            feats = extractor.extract(image)
        kpts = feats["keypoints"][0].cpu().numpy()           # (N,2) pixel
        desc = feats["descriptors"][0].cpu().numpy()         # (N,256)
        scores = feats["keypoint_scores"][0].cpu().numpy()   # (N,)
        if model == "OPENCV" and opencv_dist is not None:
            kpts = undistort_opencv(kpts, fx, fy, cx, cy, *opencv_dist)
        elif model == "SIMPLE_RADIAL" and k1 != 0.0:
            kpts = undistort_simple_radial(kpts, fx, cx, cy, k1)
        out = args.out_dir / (ip.stem + args.suffix)
        with open(out, "w") as fo:
            fo.write("# X Y SCORE D0 D1 ...\n")
            for n in range(len(kpts)):
                row = [f"{kpts[n,0]:.4f}", f"{kpts[n,1]:.4f}", f"{scores[n]:.6f}"]
                row += [f"{d:.6f}" for d in desc[n]]
                fo.write(" ".join(row) + "\n")
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(imgs)} ({len(kpts)} kpts)")
    print("done")


if __name__ == "__main__":
    main()
