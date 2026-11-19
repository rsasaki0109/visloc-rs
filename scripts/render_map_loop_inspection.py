#!/usr/bin/env python3
"""Plot cross-session PnP support along the query trajectories."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="basalt_map_merge loop_inspection.json")
    parser.add_argument("--out", type=Path, required=True, help="output plot (PNG, SVG, or PDF)")
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    if report.get("schema") != "visloc.basalt_loop_inspection.v1":
        parser.error("expected a visloc.basalt_loop_inspection.v1 report")
    n = len(report["pairs"])
    if n == 0:
        parser.error("report contains no session pairs")
    sessions = report["sessions"]

    columns = min(n, 3)
    rows = (n + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, squeeze=False, figsize=(4.5 * columns, 4 * rows))
    legend = [
        Line2D([], [], color="#888888", label="Query trajectory (camera 0)"),
        Line2D([], [], linestyle="", marker="o", color="#1f77b4", label="Agrees with registration"),
        Line2D([], [], linestyle="", marker="x", color="#c44e24", label="Other PnP candidate"),
    ]
    for ax, pair in zip(axes.flat, report["pairs"]):
        base = sessions[pair["base_session"]]
        query = sessions[pair["other_session"]]
        path = np.asarray(query["camera0_centers_m"])
        if path.size:
            ax.plot(path[:, 0], path[:, 1], color="#888888", linewidth=1)
        for agreeing, marker, color in [(False, "x", "#c44e24"), (True, "o", "#1f77b4")]:
            centers = np.asarray([
                candidate["other_camera_center_m"]
                for candidate in pair["candidates"]
                if candidate["agrees_with_registration"] == agreeing
            ])
            if centers.size:
                ax.scatter(centers[:, 0], centers[:, 1], marker=marker, color=color, s=10, alpha=0.65)
        ax.set_title(f"{Path(base['source']).name} ← {Path(query['source']).name}", fontsize=10)
        ax.set(xlabel="Query x (m)", ylabel="Query y (m)", aspect="equal")
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.legend(handles=legend, loc="lower center", ncol=min(columns, len(legend)))
    fig.tight_layout(rect=(0, 0.15 if columns == 1 else 0.11, 1, 1))
    fig.savefig(args.out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
