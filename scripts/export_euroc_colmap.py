#!/usr/bin/env python3
"""Export a visloc-rs EuRoC VI-SLAM run to a COLMAP sparse model, so the
estimated camera trajectory can bootstrap 3D Gaussian Splatting / NeRF training
(gsplat, nerfstudio) — the "3DGS bootstrap export" from the roadmap.

Input is the demo's `slam_trajectory.csv` (one row per tracking-success frame:
`timestamp_ns, frame_idx, px,py,pz, qw,qx,qy,qz, tracking_success`). visloc-rs
stores each pose as **camera-to-world**: (px,py,pz) is the camera centre C in
world, and (qw..qz) is the camera-to-world rotation R_cw. COLMAP's images.txt
wants **world-to-camera**: qvec = R_wc = R_cw^{-1} (quaternion conjugate),
tvec = -R_wc · C. The cam0 intrinsics are written as an OPENCV camera (radial-
tangential k1,k2,p1,p2) so the gsplat/nerfstudio COLMAP loader undistorts.

Outputs a COLMAP text model (cameras.txt / images.txt / points3D.txt) plus an
`images/` dir of symlinks to the cam0 frames, ready for e.g.
`gsplat` simple_trainer with `--data_dir`.

Usage:
    python3 scripts/export_euroc_colmap.py \\
        --traj   target/euroc_improve_MH_01_nonstrict_superpoint/slam_trajectory.csv \\
        --images <euroc>/MH_01_easy/mav0/cam0/data \\
        --out    target/euroc_mh01_colmap

Asset/bootstrap tool, not part of CI.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

# EuRoC cam0 (MH / V sequences share these) — pinhole + radial-tangential.
FX, FY, CX, CY = 458.654, 457.296, 367.215, 248.375
K1, K2, P1, P2 = -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05
WIDTH, HEIGHT = 752, 480


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traj", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True, help="EuRoC cam0/data dir (<timestamp>.png)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stride", type=int, default=1, help="keep every Nth tracked frame")
    p.add_argument("--init-points", type=int, default=8000, help="random init points written to points3D.txt")
    return p.parse_args()


def quat_to_R(w, x, y, z):
    """Rotation matrix from a (w,x,y,z) unit quaternion (here R_cw)."""
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def matvec(R, v):
    return [sum(R[i][j] * v[j] for j in range(3)) for i in range(3)]


def transpose(R):
    return [[R[j][i] for j in range(3)] for i in range(3)]


def main() -> int:
    args = parse_args()
    rows = [r for r in csv.DictReader(args.traj.open()) if r["px"] != ""]
    rows = rows[:: args.stride]
    if not rows:
        print("no posed rows in trajectory")
        return 1

    out = args.out
    sparse = out / "sparse" / "0"
    img_out = out / "images"
    sparse.mkdir(parents=True, exist_ok=True)
    img_out.mkdir(parents=True, exist_ok=True)

    # cameras.txt — single OPENCV camera (lets the loader undistort EuRoC radtan)
    (sparse / "cameras.txt").write_text(
        "# Camera list\n# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 OPENCV {WIDTH} {HEIGHT} {FX} {FY} {CX} {CY} {K1} {K2} {P1} {P2}\n"
    )

    # images.txt — world-to-camera per frame
    lines = ["# Image list\n# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"]
    xs, ys, zs = [], [], []
    linked = 0
    for i, r in enumerate(rows, start=1):
        C = [float(r["px"]), float(r["py"]), float(r["pz"])]
        xs.append(C[0]); ys.append(C[1]); zs.append(C[2])
        w, x, y, z = float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])
        R_cw = quat_to_R(w, x, y, z)
        R_wc = transpose(R_cw)
        t = matvec(R_wc, C)
        tvec = [-t[0], -t[1], -t[2]]
        # qvec of R_wc = conjugate of the camera-to-world quaternion
        qw, qx, qy, qz = w, -x, -y, -z
        name = f"{r['timestamp_ns']}.png"
        src = args.images / name
        dst = img_out / name
        if src.exists() and not dst.exists():
            os.symlink(src.resolve(), dst)
            linked += 1
        lines.append(f"{i} {qw} {qx} {qy} {qz} {tvec[0]} {tvec[1]} {tvec[2]} 1 {name}\n")
        lines.append("\n")  # empty POINTS2D line
    (sparse / "images.txt").write_text("".join(lines))

    # points3D.txt — random init in the trajectory's bounding box (gsplat refines)
    import random
    random.seed(0)
    lo = [min(xs), min(ys), min(zs)]
    hi = [max(xs), max(ys), max(zs)]
    pad = [max((hi[k] - lo[k]) * 0.5, 0.5) for k in range(3)]
    pts = ["# 3D point list\n# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n"]
    for pid in range(1, args.init_points + 1):
        px = random.uniform(lo[0] - pad[0], hi[0] + pad[0])
        py = random.uniform(lo[1] - pad[1], hi[1] + pad[1])
        pz = random.uniform(lo[2] - pad[2], hi[2] + pad[2])
        pts.append(f"{pid} {px:.4f} {py:.4f} {pz:.4f} 128 128 128 1.0\n")
    (sparse / "points3D.txt").write_text("".join(pts))

    print(f"wrote COLMAP model to {sparse} : {len(rows)} images ({linked} symlinked), "
          f"{args.init_points} init points")
    print(f"scene bbox X[{lo[0]:.2f},{hi[0]:.2f}] Y[{lo[1]:.2f},{hi[1]:.2f}] Z[{lo[2]:.2f},{hi[2]:.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
