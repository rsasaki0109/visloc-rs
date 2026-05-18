#!/usr/bin/env python3
"""Render side-by-side stereo VO / BA / ground-truth trajectories.

Reads the CSVs the `online_slam_stereo_vo_kitti_demo` writes
(`vo.csv`, optionally `ba.csv` and `gt.csv`) and produces a
top-down (X–Z) plot plus per-frame ATE curves so the demo run is
visually checkable. KITTI's gravity is along `+Y`, so projecting
to `(X, Z)` is the standard top-down view.

Asset-generation tool, not part of CI. Output: a single
`<out-dir>/trajectories.png`.

Usage:

    python3 scripts/plot_stereo_vo_trajectories.py \
        --out-dir target/kitti_stereo_vo_demo

Or pass an explicit set of files:

    python3 scripts/plot_stereo_vo_trajectories.py \
        --vo target/.../vo.csv --ba target/.../ba.csv \
        --gt target/.../gt.csv --output target/.../trajectories.png
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Optional


def load_trajectory(path: Path) -> list[tuple[float, float, float]]:
    pts: list[tuple[float, float, float]] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pts.append((float(row["x"]), float(row["y"]), float(row["z"])))
    return pts


def load_loop_edges(
    path: Path,
) -> list[tuple[int, int, str, tuple[float, float, float], tuple[float, float, float]]]:
    """Load each row of the demo's loop_edges.csv as
    (from_id, to_id, source, from_xyz, to_xyz). Rendered as a dashed
    overlay on the top-down plot to show which keyframe pair the PGO
    loop edge connects (and where it came from: the appearance-based
    `scanner` or the `synthetic-gt` fallback)."""
    edges = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            edges.append((
                int(row["from_id"]),
                int(row["to_id"]),
                row["source"],
                (float(row["from_x"]), float(row["from_y"]), float(row["from_z"])),
                (float(row["to_x"]), float(row["to_y"]), float(row["to_z"])),
            ))
    return edges


def per_frame_ate(estimated, reference) -> list[float]:
    n = min(len(estimated), len(reference))
    out: list[float] = []
    for i in range(n):
        ex, ey, ez = estimated[i]
        rx, ry, rz = reference[i]
        out.append(math.sqrt((ex - rx) ** 2 + (ey - ry) ** 2 + (ez - rz) ** 2))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Demo output dir (looks for vo.csv / ba.csv / gt.csv).")
    parser.add_argument("--vo", type=Path, default=None)
    parser.add_argument("--ba", type=Path, default=None)
    parser.add_argument("--pgo", type=Path, default=None)
    parser.add_argument("--gt", type=Path, default=None)
    parser.add_argument("--loop-edges", type=Path, default=None,
                        help="loop_edges.csv (rendered as dashed overlays).")
    parser.add_argument("--output", type=Path, default=None,
                        help="PNG output path (defaults to <out-dir>/trajectories.png).")
    args = parser.parse_args()

    vo_path: Optional[Path] = args.vo
    ba_path: Optional[Path] = args.ba
    pgo_path: Optional[Path] = args.pgo
    gt_path: Optional[Path] = args.gt
    loop_edges_path: Optional[Path] = args.loop_edges
    output: Optional[Path] = args.output
    if args.out_dir is not None:
        if vo_path is None and (args.out_dir / "vo.csv").exists():
            vo_path = args.out_dir / "vo.csv"
        if ba_path is None and (args.out_dir / "ba.csv").exists():
            ba_path = args.out_dir / "ba.csv"
        if pgo_path is None and (args.out_dir / "pgo.csv").exists():
            pgo_path = args.out_dir / "pgo.csv"
        if gt_path is None and (args.out_dir / "gt.csv").exists():
            gt_path = args.out_dir / "gt.csv"
        if loop_edges_path is None and (args.out_dir / "loop_edges.csv").exists():
            loop_edges_path = args.out_dir / "loop_edges.csv"
        if output is None:
            output = args.out_dir / "trajectories.png"
    if vo_path is None or output is None:
        parser.error("either --out-dir or both --vo and --output must be supplied")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot", flush=True)
        return 0

    vo = load_trajectory(vo_path)
    ba = load_trajectory(ba_path) if ba_path else None
    pgo = load_trajectory(pgo_path) if pgo_path else None
    gt = load_trajectory(gt_path) if gt_path else None
    loop_edges = load_loop_edges(loop_edges_path) if loop_edges_path else []

    fig, axes = plt.subplots(1, 2 if gt else 1, figsize=(14 if gt else 8, 6))
    ax_top = axes[0] if gt else axes
    ax_top.plot([p[0] for p in vo], [p[2] for p in vo], "-o",
                color="#1f77b4", label="VO", markersize=3, linewidth=1)
    if ba:
        ax_top.plot([p[0] for p in ba], [p[2] for p in ba], "-^",
                    color="#d62728", label="BA-refined", markersize=3, linewidth=1)
    if pgo:
        ax_top.plot([p[0] for p in pgo], [p[2] for p in pgo], "-s",
                    color="#9467bd", label="PGO (loop-closed)", markersize=3, linewidth=1)
    if gt:
        ax_top.plot([p[0] for p in gt], [p[2] for p in gt], "-",
                    color="#2ca02c", label="GT", linewidth=2)
    # Loop-closure overlays. One dashed line per accepted loop edge,
    # colored by source so the appearance-detected ones (`scanner`) are
    # distinguishable from the GT-derived `synthetic-gt` fallback.
    seen_sources: set[str] = set()
    source_colors = {"scanner": "#e377c2", "synthetic-gt": "#bcbd22"}
    for from_id, to_id, source, from_xyz, to_xyz in loop_edges:
        color = source_colors.get(source, "#7f7f7f")
        label = f"loop ({source})" if source not in seen_sources else None
        seen_sources.add(source)
        ax_top.plot(
            [from_xyz[0], to_xyz[0]],
            [from_xyz[2], to_xyz[2]],
            "--",
            color=color,
            linewidth=1.5,
            label=label,
        )
        ax_top.scatter(
            [from_xyz[0], to_xyz[0]],
            [from_xyz[2], to_xyz[2]],
            color=color,
            s=30,
            zorder=5,
        )
        ax_top.annotate(
            f"kf{from_id}",
            xy=(from_xyz[0], from_xyz[2]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color=color,
        )
        ax_top.annotate(
            f"kf{to_id}",
            xy=(to_xyz[0], to_xyz[2]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color=color,
        )
    ax_top.set_xlabel("X (m, lateral)")
    ax_top.set_ylabel("Z (m, forward)")
    ax_top.set_title("Top-down (X–Z) trajectory")
    ax_top.set_aspect("equal", adjustable="datalim")
    ax_top.legend()
    ax_top.grid(True, alpha=0.3)

    if gt:
        ax_err = axes[1]
        vo_ate = per_frame_ate(vo, gt)
        ax_err.plot(range(len(vo_ate)), vo_ate, "-o",
                    color="#1f77b4", label="VO", markersize=3, linewidth=1)
        if ba:
            ba_ate = per_frame_ate(ba, gt)
            ax_err.plot(range(len(ba_ate)), ba_ate, "-^",
                        color="#d62728", label="BA-refined", markersize=3, linewidth=1)
        if pgo:
            pgo_ate = per_frame_ate(pgo, gt)
            ax_err.plot(range(len(pgo_ate)), pgo_ate, "-s",
                        color="#9467bd", label="PGO (loop-closed)", markersize=3, linewidth=1)
        ax_err.set_xlabel("frame index")
        ax_err.set_ylabel("|estimate − GT| (m)")
        ax_err.set_title("Per-frame translation error vs ground truth")
        ax_err.legend()
        ax_err.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=120)
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
