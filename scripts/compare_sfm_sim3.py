#!/usr/bin/env python3
"""Sim(3)-align two COLMAP reconstructions by camera centre and report the RMSE.

Usage:  compare_sfm_sim3.py REFERENCE/images.txt QUERY/images.txt

Both files are COLMAP `images.txt`. Images are matched by the first integer in
their NAME field (so `000005.png` and `frame_000005.png` pair up). The QUERY
camera centres are aligned to the REFERENCE with an Umeyama similarity transform
(rotation + uniform scale + translation), which is the right gauge for a
monocular reconstruction whose absolute scale is free. Used by
run_unordered_sfm_benchmark.sh to check the unordered reconstruction reproduces
the trusted ordered one.
"""
import re
import sys

import numpy as np


def quat_to_rot(qw, qx, qy, qz):
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])


def load_centers(path):
    """frame_index -> camera centre (world frame), from a COLMAP images.txt."""
    centers = {}
    lines = [ln for ln in open(path) if not ln.startswith("#") and ln.strip()]
    # images.txt alternates a pose line and a (possibly empty) POINTS2D line.
    i = 0
    while i < len(lines):
        v = lines[i].split()
        i += 2  # skip the POINTS2D line
        if len(v) < 10:
            continue
        qw, qx, qy, qz = map(float, v[1:5])
        t = np.array(list(map(float, v[5:8])))
        m = re.search(r"(\d+)", v[9])
        if m is None:
            continue
        frame = int(m.group(1))
        R = quat_to_rot(qw, qx, qy, qz)
        centers[frame] = -R.T @ t  # camera centre = -R^T t
    return centers


def umeyama(src, dst):
    """Similarity (scale s, rotation R, translation t) mapping src -> dst."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (sc ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    ref = load_centers(sys.argv[1])
    qry = load_centers(sys.argv[2])
    common = sorted(set(ref) & set(qry))
    print(f"  reference frames: {len(ref)}, query registered: {len(qry)}, common: {len(common)}")
    if len(common) < 3:
        print("  too few common frames to align")
        sys.exit(1)
    src = np.array([qry[f] for f in common])
    dst = np.array([ref[f] for f in common])
    s, R, t = umeyama(src, dst)
    aligned = (s * (R @ src.T).T) + t
    err = np.linalg.norm(aligned - dst, axis=1)
    extent = np.linalg.norm(dst.max(0) - dst.min(0))
    print(f"  Sim(3) scale = {s:.4f}")
    print(f"  camera-centre RMSE = {np.sqrt((err ** 2).mean()) * 100:.2f} cm "
          f"(median {np.median(err) * 100:.2f} cm, max {err.max() * 100:.2f} cm)")
    print(f"  reference trajectory extent = {extent:.3f} m "
          f"({100 * np.sqrt((err ** 2).mean()) / extent:.1f}% of extent)")


if __name__ == "__main__":
    main()
