#!/usr/bin/env python3
"""Animate an EuRoC online-SLAM run into a GIF: a UAV flying a trajectory
while the estimate tracks ground truth in real time.

Consumes the ``slam_errors.csv`` written by the EuRoC online-SLAM demo
(``examples/euroc_online_slam_vi_image_demo``) — one row per
tracking-success frame, holding the ground-truth and estimated camera
position. The estimate is rigidly aligned to ground truth (Umeyama,
no scale — the same convention as the reported rigid ATE) so the two
trajectories share a world frame; then both are grown frame by frame.

Layout (pure 2D, no 3D backend needed):
  * left  — top-down (X, Y) route: ground truth solid, estimate dashed,
            a marker at the current estimated pose.
  * right-top    — altitude (Z) over time: ground truth vs estimate
                   (the third dimension of the flight).
  * right-bottom — live rigidly-aligned position error.
The title carries the running RMSE.

Usage:
    python3 scripts/animate_euroc_slam.py \\
        --errors-csv target/euroc_phase26_1_rebase_MH_01_easy_strict/slam_errors.csv \\
        --output     docs/assets/euroc_mh01_slam.gif \\
        --title      'EuRoC MH_01 (UAV / Machine Hall) — visloc-rs online SLAM' \\
        --fps 15

Asset-generation tool, not part of CI.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--errors-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--title", type=str, default="EuRoC UAV — visloc-rs online SLAM")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--stride", type=int, default=1, help="keep every Nth frame (thins long runs)")
    p.add_argument("--align", choices=["rigid", "sim", "none"], default="rigid")
    p.add_argument(
        "--trim-tail-error",
        type=float,
        default=0.0,
        help="drop trailing frames whose aligned error exceeds this many metres "
        "(strips the tracking-loss tail the demo logs as its last success frame)",
    )
    return p.parse_args()


def load(path: Path):
    import numpy as np

    gt, est = [], []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            gt.append([float(row["gt_px"]), float(row["gt_py"]), float(row["gt_pz"])])
            est.append([float(row["est_px"]), float(row["est_py"]), float(row["est_pz"])])
    return np.array(gt), np.array(est)


def umeyama(src, dst, with_scale: bool):
    """Least-squares similarity/rigid transform mapping src onto dst."""
    import numpy as np

    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = s_c.T @ d_c / len(src)
    u, sig, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    diag = np.diag([1.0, 1.0, d])
    rot = vt.T @ diag @ u.T
    scale = ((sig * np.array([1.0, 1.0, d])).sum() / ((s_c ** 2).sum() / len(src))) if with_scale else 1.0
    t = mu_d - scale * rot @ mu_s
    return (scale * (rot @ src.T)).T + t


def main() -> int:
    args = parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.gridspec import GridSpec
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        print(f"missing dependency: {exc} (need matplotlib + numpy + pillow)", file=sys.stderr)
        return 2

    gt, est = load(args.errors_csv)
    if len(gt) < 10:
        print(f"too few rows in {args.errors_csv}", file=sys.stderr)
        return 1

    if args.align == "rigid":
        est = umeyama(est, gt, with_scale=False)
    elif args.align == "sim":
        est = umeyama(est, gt, with_scale=True)

    if args.stride > 1:
        gt, est = gt[:: args.stride], est[:: args.stride]

    # Strip the tracking-loss tail: the demo logs the frame where the tracker
    # is about to give up as its last "success" row, which spikes the error.
    if args.trim_tail_error > 0.0:
        tail_err = np.linalg.norm(est - gt, axis=1)
        keep = len(gt)
        while keep > 1 and tail_err[keep - 1] > args.trim_tail_error:
            keep -= 1
        if keep < len(gt):
            print(f"trimmed {len(gt) - keep} tracking-loss tail frame(s)")
        gt, est = gt[:keep], est[:keep]

    n = len(gt)

    err = np.linalg.norm(est - gt, axis=1)
    rmse = float(np.sqrt((err ** 2).mean()))

    gt_color, est_color = "#3a3a3a", "#d23737"

    fig = plt.figure(figsize=(10, 4.7), dpi=92)
    gs = GridSpec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1, 1], figure=fig)
    axxy = fig.add_subplot(gs[:, 0])
    axz = fig.add_subplot(gs[0, 1])
    axerr = fig.add_subplot(gs[1, 1])

    # --- top-down route ---
    # Use a SQUARE data window (equal x/y span, centred on the route) so that
    # equal-aspect + adjustable="box" simply squares the box without clipping
    # the dominant axis — keeping the map metric-correct and every sample in view.
    allxy = np.vstack([gt[:, :2], est[:, :2]])
    lo, hi = allxy.min(0), allxy.max(0)
    center = (lo + hi) / 2
    half = (hi - lo).max() / 2 * 1.12 + 0.05
    axxy.set_xlim(center[0] - half, center[0] + half)
    axxy.set_ylim(center[1] - half, center[1] + half)
    axxy.set_aspect("equal", adjustable="box")
    axxy.set_xlabel("X [m]", fontsize=9)
    axxy.set_ylabel("Y [m]", fontsize=9)
    axxy.grid(True, alpha=0.3)
    axxy.set_title("Top-down route (UAV)", fontsize=10)
    # faint full ground-truth route for context
    axxy.plot(gt[:, 0], gt[:, 1], color=gt_color, lw=1.0, alpha=0.15)
    (gt_xy,) = axxy.plot([], [], color=gt_color, lw=2.2, label="ground truth")
    (est_xy,) = axxy.plot([], [], color=est_color, lw=1.5, ls="--", label="visloc-rs estimate")
    (drone_xy,) = axxy.plot([], [], marker="o", color=est_color, ms=9, mfc=est_color, mec="white", mew=1.2)
    axxy.scatter([gt[0, 0]], [gt[0, 1]], color=gt_color, s=35, zorder=4)
    axxy.legend(loc="best", fontsize=9, framealpha=0.85)

    # --- altitude (Z) over time ---
    axz.set_xlim(0, n)
    zlo = min(gt[:, 2].min(), est[:, 2].min())
    zhi = max(gt[:, 2].max(), est[:, 2].max())
    zpad = (zhi - zlo) * 0.15 + 0.02
    axz.set_ylim(zlo - zpad, zhi + zpad)
    axz.set_ylabel("altitude Z [m]", fontsize=9)
    axz.grid(True, alpha=0.3)
    axz.set_title("Altitude — truth vs estimate", fontsize=10)
    (gt_z,) = axz.plot([], [], color=gt_color, lw=1.8)
    (est_z,) = axz.plot([], [], color=est_color, lw=1.2, ls="--")
    axz.tick_params(labelbottom=False)

    # --- live error ---
    axerr.set_xlim(0, n)
    axerr.set_ylim(0, max(err.max() * 1.12, 0.05))
    axerr.set_xlabel("tracking-success frame", fontsize=9)
    axerr.set_ylabel("aligned error [m]", fontsize=9)
    axerr.grid(True, alpha=0.3)
    axerr.axhline(rmse, color=gt_color, lw=1.0, ls=":", label=f"final RMSE {rmse * 100:.1f} cm")
    axerr.legend(loc="upper left", fontsize=8)
    axerr.set_title("Live localization error", fontsize=10)
    (err_line,) = axerr.plot([], [], color=est_color, lw=1.3)
    (err_dot,) = axerr.plot([], [], marker="o", color=est_color, ms=5)

    suptitle = fig.suptitle(args.title, fontsize=11)
    frames = np.arange(n)

    def init():
        for ln in (gt_xy, est_xy, drone_xy, gt_z, est_z, err_line, err_dot):
            ln.set_data([], [])
        return gt_xy, est_xy, drone_xy, gt_z, est_z, err_line, err_dot

    def update(i):
        k = i + 1
        gt_xy.set_data(gt[:k, 0], gt[:k, 1])
        est_xy.set_data(est[:k, 0], est[:k, 1])
        drone_xy.set_data([est[i, 0]], [est[i, 1]])

        gt_z.set_data(frames[:k], gt[:k, 2])
        est_z.set_data(frames[:k], est[:k, 2])

        err_line.set_data(frames[:k], err[:k])
        err_dot.set_data([i], [err[i]])

        running = float(np.sqrt((err[:k] ** 2).mean()))
        suptitle.set_text(f"{args.title}\nframe {k}/{n}    running RMSE {running * 100:.1f} cm")
        return gt_xy, est_xy, drone_xy, gt_z, est_z, err_line, err_dot

    anim = FuncAnimation(fig, update, init_func=init, frames=n, blit=False, interval=1000 / args.fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    anim.save(str(args.output), writer=PillowWriter(fps=args.fps))

    # Shrink the GIF: quantise to a 128-colour palette and re-save optimised.
    try:
        from PIL import Image, ImageSequence

        src = Image.open(args.output)
        out_frames = [
            f.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=128)
            for f in ImageSequence.Iterator(src)
        ]
        out_frames[0].save(
            args.output,
            save_all=True,
            append_images=out_frames[1:],
            loop=0,
            duration=int(1000 / args.fps),
            optimize=True,
        )
    except Exception as exc:  # pragma: no cover - optimisation is best-effort
        print(f"(gif optimisation skipped: {exc})", file=sys.stderr)

    size_mb = args.output.stat().st_size / 1e6
    print(f"wrote {args.output} ({n} frames, final rigid RMSE {rmse * 100:.1f} cm, {size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
