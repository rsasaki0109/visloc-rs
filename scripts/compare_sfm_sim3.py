#!/usr/bin/env python3
"""Sim(3)-align two COLMAP reconstructions by camera centre and report the RMSE.

Usage:  compare_sfm_sim3.py REFERENCE/images.txt QUERY/images.txt [--plot OUT.png]

Both files are COLMAP `images.txt`. Images are matched by the first integer in
their NAME field (so `000005.png` and `frame_000005.png` pair up). The QUERY
camera centres are aligned to the REFERENCE with an Umeyama similarity transform
(rotation + uniform scale + translation), which is the right gauge for a
monocular reconstruction whose absolute scale is free. Used by
run_unordered_sfm_benchmark.sh to check the unordered reconstruction reproduces
the trusted ordered one.

With `--plot OUT.png` it also renders a figure: the Sim(3)-aligned estimated
camera centres overlaid on COLMAP's reference centres (top-down PCA view, with
the estimate's sparse point cloud as faint background) plus a sorted per-camera
error panel. Pass `--points QUERY/points3D.txt` to draw the cloud.
"""
import argparse
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


def load_points(path):
    """Nx3 array of XYZ from a COLMAP points3D.txt (ignores colour/track)."""
    pts = []
    for ln in open(path):
        if ln.startswith("#") or not ln.strip():
            continue
        v = ln.split()
        if len(v) >= 4:
            pts.append([float(v[1]), float(v[2]), float(v[3])])
    return np.array(pts) if pts else np.empty((0, 3))


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


def plot(out_path, dst, aligned, err, points, title):
    """Top-down camera-centre overlay + sorted per-camera error, to OUT.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A building's tallest spread is vertical; PCA on the reference camera
    # centres picks the two widest axes so the overlay reads as a top-down map
    # regardless of how the reconstruction happens to be oriented.
    mu = dst.mean(0)
    _, _, basis = np.linalg.svd(dst - mu)
    e0, e1 = basis[0], basis[1]
    proj = lambda P: np.column_stack([(P - mu) @ e0, (P - mu) @ e1])
    d2, a2 = proj(dst), proj(aligned)
    rmse = np.sqrt((err ** 2).mean()) * 100

    fig, (ax, axe) = plt.subplots(1, 2, figsize=(13, 5.2),
                                  gridspec_kw={"width_ratios": [1.7, 1]})

    if points is not None and len(points):
        p2 = proj(points)
        # clip the faint cloud to the camera span so a few far points don't
        # blow out the axes.
        lo, hi = a2.min(0) - 1.0, a2.max(0) + 1.0
        keep = np.all((p2 >= lo) & (p2 <= hi), axis=1)
        ax.scatter(p2[keep, 0], p2[keep, 1], s=1.2, c="0.78",
                   alpha=0.5, linewidths=0, label=f"sparse cloud ({keep.sum()} pts)")

    # residual whiskers (mostly invisible at ~cm scale, but honest)
    for r, a in zip(d2, a2):
        ax.plot([r[0], a[0]], [r[1], a[1]], "-", c="0.6", lw=0.6, zorder=2)
    ax.scatter(d2[:, 0], d2[:, 1], s=70, facecolors="none", edgecolors="#1f77b4",
               linewidths=1.4, label=f"COLMAP reference ({len(dst)})", zorder=3)
    ax.scatter(a2[:, 0], a2[:, 1], s=14, c="#d62728",
               label="visloc-rs (Sim(3)-aligned)", zorder=4)
    ax.set_aspect("equal")
    ax.set_xlabel("PCA axis 1 (m)")
    ax.set_ylabel("PCA axis 2 (m)")
    ax.set_title("Camera-centre overlay (top-down)")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, ls=":", alpha=0.4)

    order = np.argsort(err)
    ecm = err[order] * 100
    axe.bar(np.arange(len(ecm)), ecm, width=1.0, color="#d62728", alpha=0.85)
    axe.axhline(rmse, ls="--", c="k", lw=1.0, label=f"RMSE {rmse:.2f} cm")
    axe.axhline(np.median(err) * 100, ls=":", c="0.35", lw=1.0,
                label=f"median {np.median(err) * 100:.2f} cm")
    axe.set_xlabel("camera (sorted by error)")
    axe.set_ylabel("camera-centre error (cm)")
    axe.set_title("Per-camera Sim(3) residual")
    axe.legend(loc="upper left", fontsize=8)
    axe.grid(True, axis="y", ls=":", alpha=0.4)

    if title:
        fig.suptitle(title, fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96 if title else 1.0))
    fig.savefig(out_path, dpi=130)
    print(f"  wrote figure {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", help="COLMAP REFERENCE/images.txt")
    ap.add_argument("query", help="COLMAP QUERY/images.txt")
    ap.add_argument("--plot", metavar="OUT.png", help="render an overlay figure")
    ap.add_argument("--points", metavar="QUERY/points3D.txt",
                    help="estimate's sparse cloud, drawn faint under the cameras")
    ap.add_argument("--title", default="", help="figure suptitle")
    args = ap.parse_args()

    ref = load_centers(args.reference)
    qry = load_centers(args.query)
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

    if args.plot:
        points = None
        if args.points:
            # the cloud lives in the QUERY frame; carry it through the same Sim(3).
            raw = load_points(args.points)
            points = (s * (R @ raw.T).T) + t if len(raw) else raw
        plot(args.plot, dst, aligned, err, points, args.title)


if __name__ == "__main__":
    main()
