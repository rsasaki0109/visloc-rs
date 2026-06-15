#!/usr/bin/env python3
"""Plot a TUM-format estimated trajectory against ground truth (top-down XY).

Associates est<->gt by nearest timestamp, Umeyama-aligns (SE(3), no scale) the
estimate to ground truth, and writes a top-down XY plot. Standalone matplotlib
so it does not depend on evo's (version-fragile) plotting.

  scripts/plot_tum_trajectory.py --gt groundtruth.txt --est est.tum --out fig.png \
      --title "TUM fr1_xyz: virtual-stereo VO + loop closure"
"""
from __future__ import annotations

import argparse

import numpy as np


def read_tum(path: str) -> np.ndarray:
    rows = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 4:
            rows.append([float(p[0]), float(p[1]), float(p[2]), float(p[3])])
    return np.array(rows)  # [t, x, y, z]


def associate(est: np.ndarray, gt: np.ndarray, max_diff: float = 0.02):
    gt_t = gt[:, 0]
    e_idx, g_idx = [], []
    for i, t in enumerate(est[:, 0]):
        j = int(np.argmin(np.abs(gt_t - t)))
        if abs(gt_t[j] - t) <= max_diff:
            e_idx.append(i)
            g_idx.append(j)
    return est[e_idx, 1:4].T, gt[g_idx, 1:4].T  # (3,N), (3,N)


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """SE(3) (rotation+translation, no scale) mapping src->dst, both (3,N)."""
    mu_s = src.mean(axis=1, keepdims=True)
    mu_d = dst.mean(axis=1, keepdims=True)
    s = src - mu_s
    d = dst - mu_d
    cov = d @ s.T / src.shape[1]
    u, _, vt = np.linalg.svd(cov)
    e = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        e[2, 2] = -1
    r = u @ e @ vt
    t = mu_d - r @ mu_s
    return r @ src + t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--est", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    est = read_tum(args.est)
    gt = read_tum(args.gt)
    e_xyz, g_xyz = associate(est, gt)
    e_aligned = umeyama(e_xyz, g_xyz)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(g_xyz[0], g_xyz[1], "-", color="0.4", lw=2.0, label="ground truth")
    ax.plot(e_aligned[0], e_aligned[1], "-", color="C3", lw=1.4, label="estimate (SE(3)-aligned)")
    ax.scatter([g_xyz[0, 0]], [g_xyz[1, 0]], c="g", s=40, zorder=5, label="start")
    ax.set_aspect("equal", "datalim")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if args.title:
        ax.set_title(args.title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    err = np.linalg.norm(e_aligned - g_xyz, axis=0)
    ax.text(0.02, 0.02,
            f"ATE rmse {np.sqrt((err ** 2).mean()) * 1000:.1f} mm  (N={err.size})",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
