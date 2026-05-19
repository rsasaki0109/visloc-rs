#!/usr/bin/env python3
"""Render EuRoC ground-truth vs estimated trajectory from slam_errors.csv.

Consumes the `slam_errors.csv` written by the EuRoC online-SLAM demo
(`examples/euroc_online_slam_vi_image_demo`). Each row holds the
ground-truth and estimated camera position at one tracking-success
frame, plus per-frame position / orientation error.

Produces a single side-by-side PNG: top-down XY trajectory overlay
(left) and per-frame position error vs frame index (right). EuRoC's
gravity is along +Z, so the (X, Y) projection is the top-down view.

Usage:
    python3 scripts/plot_euroc_trajectory.py \\
        --errors-csv target/binary_determinism_verify_superpoint_v1/run1/slam_errors.csv \\
        --output    docs/assets/euroc_v1_01_sp_strict.png \\
        --title     'EuRoC V1_01_easy — SuperPoint + strict-stereo (Phase-26 #1)'

Asset-generation tool, not part of CI.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--errors-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", type=str, default="EuRoC trajectory overlay")
    parser.add_argument(
        "--annotate-stats",
        action="store_true",
        default=True,
        help="overlay rigid ATE and frame-count text on the plot (default true)",
    )
    return parser.parse_args()


def load_errors(path: Path):
    frames, gt_x, gt_y, gt_z, est_x, est_y, est_z, perr = [], [], [], [], [], [], [], []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frames.append(int(row["frame_idx"]))
            gt_x.append(float(row["gt_px"]))
            gt_y.append(float(row["gt_py"]))
            gt_z.append(float(row["gt_pz"]))
            est_x.append(float(row["est_px"]))
            est_y.append(float(row["est_py"]))
            est_z.append(float(row["est_pz"]))
            perr.append(float(row["position_error_m"]))
    return frames, gt_x, gt_y, gt_z, est_x, est_y, est_z, perr


def main() -> int:
    args = parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; install it first (pip install matplotlib).", file=sys.stderr)
        return 2

    frames, gt_x, gt_y, gt_z, est_x, est_y, est_z, perr = load_errors(args.errors_csv)
    if not frames:
        print(f"no rows in {args.errors_csv}", file=sys.stderr)
        return 1

    rmse = (sum(e * e for e in perr) / len(perr)) ** 0.5
    max_err = max(perr)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))

    left.plot(gt_x, gt_y, color="#3a3a3a", lw=2.0, label="ground truth")
    left.plot(est_x, est_y, color="#d23737", lw=1.2, ls="--", label="estimated")
    left.scatter([gt_x[0]], [gt_y[0]], color="#3a3a3a", s=40, zorder=5, label="start")
    left.scatter([gt_x[-1]], [gt_y[-1]], color="#3a3a3a", marker="x", s=60, zorder=5, label="end")
    left.set_xlabel("X [m]")
    left.set_ylabel("Y [m]")
    left.set_aspect("equal", adjustable="datalim")
    left.grid(True, alpha=0.3)
    left.legend(loc="best", fontsize=9)
    left.set_title("Top-down (X, Y) overlay")

    right.plot(frames, perr, color="#d23737", lw=1.0)
    right.axhline(rmse, color="#3a3a3a", lw=1.0, ls=":", label=f"RMSE {rmse * 1000:.1f} mm")
    right.set_xlabel("frame index")
    right.set_ylabel("position error [m]")
    right.grid(True, alpha=0.3)
    right.legend(loc="best", fontsize=9)
    right.set_title("Per-frame position error")

    if args.annotate_stats:
        stats = f"frames={len(frames)}    position RMSE={rmse * 1000:.1f} mm    max={max_err * 1000:.1f} mm"
        fig.suptitle(f"{args.title}\n{stats}", fontsize=11)
    else:
        fig.suptitle(args.title, fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140)
    print(f"wrote {args.output} (RMSE {rmse * 1000:.1f} mm, max {max_err * 1000:.1f} mm, {len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
