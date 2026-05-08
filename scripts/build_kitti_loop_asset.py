#!/usr/bin/env python3
"""Build the README KITTI loop-closure asset.

Reads the truth / drifted / corrected trajectories produced by
`cargo run --example online_slam_kitti_loop_demo` and emits two
visualization artifacts:

- `kitti_loop_closure.png`  — three-panel (truth / drifted / corrected)
                              top-down view plus an overlay panel showing
                              the loop-closure correction.
- `kitti_loop_closure.gif`  — short animation that fades from the drifted
                              trajectory toward the corrected one so the
                              viewer can see the pose-graph SE(3) GN
                              pulling the chain back along the loop edge.

Asset-generation tool, not part of CI. Requires Python with matplotlib,
numpy, and Pillow (already available on the development machine).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def load_trajectory_xz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    zs: list[float] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            zs.append(float(row["z"]))
    return np.array(xs), np.array(zs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("target/kitti_loop_demo"),
        help="Directory with truth.csv / drifted.csv / corrected.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/assets"),
        help="Where to write kitti_loop_closure.{png,gif}",
    )
    parser.add_argument(
        "--gif-frames",
        type=int,
        default=40,
        help="Number of frames in the morph animation (drifted -> corrected).",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=15,
        help="Frames per second in the output GIF.",
    )
    args = parser.parse_args()

    truth_x, truth_z = load_trajectory_xz(args.input_dir / "truth.csv")
    drift_x, drift_z = load_trajectory_xz(args.input_dir / "drifted.csv")
    corr_x, corr_z = load_trajectory_xz(args.input_dir / "corrected.csv")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Static three-panel PNG ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    truth_color = "#22c55e"  # green
    drift_color = "#ef4444"  # red
    corr_color = "#3b82f6"   # blue
    truth_kw = dict(color=truth_color, linewidth=2.0, label="ground truth")
    drift_kw = dict(color=drift_color, linewidth=2.0, label="drifted odometry")
    corr_kw = dict(color=corr_color, linewidth=2.0, label="corrected (SE(3) GN)")

    axes[0].plot(truth_x, truth_z, **truth_kw)
    axes[0].scatter(truth_x[0], truth_z[0], color="black", s=30, zorder=5, label="start")
    axes[0].set_title("Ground truth (KITTI 00)")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("z [m]")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(truth_x, truth_z, alpha=0.4, **truth_kw)
    axes[1].plot(drift_x, drift_z, **drift_kw)
    axes[1].scatter(drift_x[-1], drift_z[-1], color=drift_color, s=40, zorder=6, marker="x")
    axes[1].set_title("With simulated yaw drift\n(no loop closure applied)")
    axes[1].set_xlabel("x [m]")
    axes[1].axis("equal")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(truth_x, truth_z, alpha=0.4, **truth_kw)
    axes[2].plot(corr_x, corr_z, **corr_kw)
    axes[2].scatter(corr_x[-1], corr_z[-1], color=corr_color, s=40, zorder=6, marker="*")
    axes[2].set_title("After SE(3) Gauss-Newton + loop closure")
    axes[2].set_xlabel("x [m]")
    axes[2].axis("equal")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right", fontsize=8)

    end_drift_err = float(np.hypot(drift_x[-1] - truth_x[-1], drift_z[-1] - truth_z[-1]))
    end_corr_err = float(np.hypot(corr_x[-1] - truth_x[-1], corr_z[-1] - truth_z[-1]))
    fig.suptitle(
        f"KITTI 00 loop closure — {len(truth_x)} keyframes, "
        f"endpoint error: drifted={end_drift_err:.1f} m → corrected={end_corr_err:.3f} m",
        fontsize=12,
        y=1.04,
    )
    png_path = args.out_dir / "kitti_loop_closure.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")

    # ---------- Animated GIF: morph drifted → corrected ----------
    n_frames = max(args.gif_frames, 4)
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)

    drift_xy = np.stack([drift_x, drift_z], axis=1)
    corr_xy = np.stack([corr_x, corr_z], axis=1)

    truth_line, = ax.plot(truth_x, truth_z, color=truth_color, linewidth=2.0,
                          alpha=0.5, label="ground truth")
    morph_line, = ax.plot([], [], color=drift_color, linewidth=2.5,
                          label="estimate")
    start_marker = ax.scatter(truth_x[0], truth_z[0], color="black",
                              s=40, zorder=5)
    endpoint_marker = ax.scatter([], [], s=80, marker="o", facecolor="none",
                                 edgecolor=drift_color, linewidth=2.0, zorder=6)
    title = ax.set_title("")
    ax.legend(loc="upper right")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")

    pad_x = (max(np.r_[truth_x, drift_x, corr_x]) -
             min(np.r_[truth_x, drift_x, corr_x])) * 0.05
    pad_z = (max(np.r_[truth_z, drift_z, corr_z]) -
             min(np.r_[truth_z, drift_z, corr_z])) * 0.05
    ax.set_xlim(min(np.r_[truth_x, drift_x, corr_x]) - pad_x,
                max(np.r_[truth_x, drift_x, corr_x]) + pad_x)
    ax.set_ylim(min(np.r_[truth_z, drift_z, corr_z]) - pad_z,
                max(np.r_[truth_z, drift_z, corr_z]) + pad_z)

    def lerp(t: float) -> np.ndarray:
        return (1.0 - t) * drift_xy + t * corr_xy

    def color_for(t: float) -> str:
        # Lerp between drift_color (red) and corr_color (blue) in RGB.
        c0 = np.array([0xef, 0x44, 0x44]) / 255.0
        c1 = np.array([0x3b, 0x82, 0xf6]) / 255.0
        rgb = (1.0 - t) * c0 + t * c1
        return matplotlib.colors.to_hex(rgb)

    def update(frame: int):
        # Hold a few frames at the start (drifted) and end (corrected).
        if frame < 3:
            t = 0.0
        elif frame >= n_frames - 3:
            t = 1.0
        else:
            phase = (frame - 3) / max(1, (n_frames - 6))
            # Smoothstep so the morph eases in/out.
            t = phase * phase * (3.0 - 2.0 * phase)
        xy = lerp(t)
        morph_line.set_data(xy[:, 0], xy[:, 1])
        morph_line.set_color(color_for(t))
        endpoint_marker.set_offsets(xy[-1:].reshape(1, 2))
        endpoint_marker.set_edgecolor(color_for(t))
        end_err = float(np.linalg.norm(xy[-1] - np.array([truth_x[-1], truth_z[-1]])))
        title.set_text(
            f"KITTI 00 SE(3) loop closure  "
            f"({len(truth_x)} keyframes, endpoint err = {end_err:.2f} m)"
        )
        return morph_line, endpoint_marker, title

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    gif_path = args.out_dir / "kitti_loop_closure.gif"
    anim.save(gif_path, writer=PillowWriter(fps=args.gif_fps))
    plt.close(fig)
    print(f"wrote {gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
