#!/usr/bin/env python3
"""Build the README KITTI loop-closure asset.

Reads the truth / drifted / corrected trajectories produced by
`cargo run --example online_slam_kitti_loop_demo` and emits two
visualization artifacts that depict the actual SLAM pipeline:

1. `kitti_loop_closure.png`  — three static panels: ground truth, drifted
   odometry, corrected trajectory after pose-graph SE(3) optimization.

2. `kitti_loop_closure.gif`  — a pipeline animation with three phases that
   match what an online visual SLAM system would do:

       Phase A (VO drifting):     the drifted odometry polyline is built up
                                  segment by segment, so the viewer can see
                                  the per-edge drift accumulating as the
                                  vehicle drives around the loop.

       Phase B (Loop detection):  once VO finishes, a yellow loop-closure
                                  edge is drawn between the current keyframe
                                  and the matched older keyframe. The title
                                  bar names the matched pair.

       Phase C (Graph optim.):    the trajectory smoothly morphs from the
                                  drifted state to the post-PGO corrected
                                  state while the loop edge shrinks and
                                  fades. The title shows the running
                                  endpoint error so the convergence is
                                  legible at a glance.

Asset-generation tool, not part of CI. Requires Python with matplotlib,
numpy, and Pillow (already on the development machine).
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
from matplotlib.colors import to_rgba

TRUTH_COLOR = "#22c55e"   # green
DRIFT_COLOR = "#ef4444"   # red
CORR_COLOR = "#3b82f6"    # blue
LOOP_COLOR = "#facc15"    # yellow


def load_trajectory_xz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    zs: list[float] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            zs.append(float(row["z"]))
    return np.array(xs), np.array(zs)


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


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
        "--vo-frames",
        type=int,
        default=70,
        help="Number of GIF frames spent building up the VO drifted polyline.",
    )
    parser.add_argument(
        "--detect-frames",
        type=int,
        default=12,
        help="Number of GIF frames spent on the loop-detection pulse.",
    )
    parser.add_argument(
        "--optim-frames",
        type=int,
        default=40,
        help="Number of GIF frames spent morphing drifted -> corrected.",
    )
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=12,
        help="Number of GIF frames holding the final corrected state.",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=18,
        help="Frames per second in the output GIF.",
    )
    args = parser.parse_args()

    truth_x, truth_z = load_trajectory_xz(args.input_dir / "truth.csv")
    drift_x, drift_z = load_trajectory_xz(args.input_dir / "drifted.csv")
    corr_x, corr_z = load_trajectory_xz(args.input_dir / "corrected.csv")

    n = len(truth_x)
    assert len(drift_x) == n and len(corr_x) == n, "trajectory lengths must match"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    end_drift_err = float(
        np.hypot(drift_x[-1] - truth_x[-1], drift_z[-1] - truth_z[-1])
    )
    end_corr_err = float(
        np.hypot(corr_x[-1] - truth_x[-1], corr_z[-1] - truth_z[-1])
    )

    # ---------- Static three-panel PNG ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    truth_kw = dict(color=TRUTH_COLOR, linewidth=2.0, label="ground truth")
    drift_kw = dict(color=DRIFT_COLOR, linewidth=2.0, label="drifted odometry")
    corr_kw = dict(color=CORR_COLOR, linewidth=2.0, label="corrected (SE(3) GN)")

    axes[0].plot(truth_x, truth_z, **truth_kw)
    axes[0].scatter(
        truth_x[0], truth_z[0], color="black", s=30, zorder=5, label="start"
    )
    axes[0].set_title("Ground truth (KITTI 00)")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("z [m]")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(truth_x, truth_z, alpha=0.4, **truth_kw)
    axes[1].plot(drift_x, drift_z, **drift_kw)
    axes[1].scatter(
        drift_x[-1], drift_z[-1], color=DRIFT_COLOR, s=40, zorder=6, marker="x"
    )
    axes[1].set_title(
        "VO drifts as the loop closes\n(no loop closure applied yet)"
    )
    axes[1].set_xlabel("x [m]")
    axes[1].axis("equal")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(truth_x, truth_z, alpha=0.4, **truth_kw)
    axes[2].plot(corr_x, corr_z, **corr_kw)
    axes[2].scatter(
        corr_x[-1], corr_z[-1], color=CORR_COLOR, s=40, zorder=6, marker="*"
    )
    axes[2].set_title("After loop detection + SE(3) Gauss-Newton")
    axes[2].set_xlabel("x [m]")
    axes[2].axis("equal")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"KITTI 00 loop closure — {n} keyframes, "
        f"endpoint error: drifted={end_drift_err:.1f} m → "
        f"corrected={end_corr_err:.3f} m",
        fontsize=12,
        y=1.04,
    )
    png_path = args.out_dir / "kitti_loop_closure.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")

    # ---------- Animated GIF: VO build → loop detect → PGO ----------
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)

    # Limits chosen to fit the full union of all three trajectories so the
    # axes never rescale during the animation.
    all_x = np.r_[truth_x, drift_x, corr_x]
    all_z = np.r_[truth_z, drift_z, corr_z]
    pad_x = (all_x.max() - all_x.min()) * 0.08
    pad_z = (all_z.max() - all_z.min()) * 0.08
    ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax.set_ylim(all_z.min() - pad_z, all_z.max() + pad_z)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")

    # Faint truth guide stays on screen the whole time.
    ax.plot(
        truth_x,
        truth_z,
        color=TRUTH_COLOR,
        linewidth=1.5,
        alpha=0.35,
        label="ground truth",
        zorder=1,
    )
    ax.scatter(
        truth_x[0], truth_z[0], color="black", s=40, zorder=5, label="start"
    )

    # The estimated trajectory line and its current-frame marker.
    estimate_line, = ax.plot(
        [], [], color=DRIFT_COLOR, linewidth=2.5, label="estimate", zorder=4
    )
    current_marker = ax.scatter(
        [], [], color=DRIFT_COLOR, s=70, marker="o", zorder=6
    )

    # Loop-closure edge between matched KFs (drawn during phases B + C).
    loop_line, = ax.plot(
        [], [], color=LOOP_COLOR, linewidth=2.0, alpha=0.0, zorder=3
    )
    matched_marker = ax.scatter(
        [], [], color=LOOP_COLOR, s=120, marker="o",
        facecolor="none", edgecolor=LOOP_COLOR, linewidth=2.0, alpha=0.0, zorder=5,
    )

    title = ax.set_title("")
    phase_label = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#0f172a", ec="#1e293b", alpha=0.85),
        color="white",
    )
    ax.legend(loc="upper right")

    drifted_xy = np.stack([drift_x, drift_z], axis=1)
    corrected_xy = np.stack([corr_x, corr_z], axis=1)
    truth_endpoint = np.array([truth_x[-1], truth_z[-1]])

    vo_frames = max(args.vo_frames, 4)
    detect_frames = max(args.detect_frames, 1)
    optim_frames = max(args.optim_frames, 4)
    hold_frames = max(args.hold_frames, 1)
    total_frames = vo_frames + detect_frames + optim_frames + hold_frames

    def lerp_color(a: str, b: str, t: float) -> tuple[float, float, float, float]:
        ca = np.array(to_rgba(a))
        cb = np.array(to_rgba(b))
        return tuple((1.0 - t) * ca + t * cb)

    def update(frame: int):
        if frame < vo_frames:
            # Phase A: VO drifted polyline grows segment by segment.
            progress = (frame + 1) / vo_frames
            kf_count = max(2, int(round(progress * n)))
            xy = drifted_xy[:kf_count]
            estimate_line.set_data(xy[:, 0], xy[:, 1])
            estimate_line.set_color(DRIFT_COLOR)
            estimate_line.set_alpha(1.0)
            current_marker.set_offsets(xy[-1:].reshape(1, 2))
            current_marker.set_color(DRIFT_COLOR)
            loop_line.set_alpha(0.0)
            matched_marker.set_alpha(0.0)
            running_err = float(np.linalg.norm(xy[-1] - truth_endpoint)) if kf_count == n else 0.0
            phase_label.set_text("Phase 1/3 — visual odometry")
            if kf_count < n:
                title.set_text(
                    f"KITTI 00  •  building VO trajectory  •  "
                    f"keyframe {kf_count}/{n}"
                )
            else:
                title.set_text(
                    f"KITTI 00  •  VO complete  •  "
                    f"endpoint drift = {running_err:.1f} m"
                )
            return estimate_line, current_marker, loop_line, matched_marker, title, phase_label

        if frame < vo_frames + detect_frames:
            # Phase B: Loop detection — pulse the loop edge.
            local = frame - vo_frames
            pulse = 0.4 + 0.6 * np.sin(local * np.pi / max(1, detect_frames - 1)) ** 2
            estimate_line.set_data(drifted_xy[:, 0], drifted_xy[:, 1])
            estimate_line.set_color(DRIFT_COLOR)
            estimate_line.set_alpha(1.0)
            current_marker.set_offsets(drifted_xy[-1:].reshape(1, 2))
            current_marker.set_color(DRIFT_COLOR)
            loop_xy = np.stack([drifted_xy[0], drifted_xy[-1]], axis=0)
            loop_line.set_data(loop_xy[:, 0], loop_xy[:, 1])
            loop_line.set_alpha(pulse)
            loop_line.set_color(LOOP_COLOR)
            matched_marker.set_offsets(loop_xy)
            matched_marker.set_alpha(pulse)
            phase_label.set_text("Phase 2/3 — loop closure detected")
            title.set_text(
                f"KITTI 00  •  loop closure: KF 0 ↔ KF {n - 1}  •  "
                f"endpoint drift = {end_drift_err:.1f} m"
            )
            return estimate_line, current_marker, loop_line, matched_marker, title, phase_label

        if frame < vo_frames + detect_frames + optim_frames:
            # Phase C: Graph optimization — morph drifted -> corrected.
            local = frame - (vo_frames + detect_frames)
            t = smoothstep(local / max(1, optim_frames - 1))
            xy = (1.0 - t) * drifted_xy + t * corrected_xy
            estimate_line.set_data(xy[:, 0], xy[:, 1])
            color = lerp_color(DRIFT_COLOR, CORR_COLOR, t)
            estimate_line.set_color(color)
            current_marker.set_offsets(xy[-1:].reshape(1, 2))
            current_marker.set_color(color)
            # Loop edge collapses as the endpoints come together.
            loop_xy = np.stack([xy[0], xy[-1]], axis=0)
            loop_line.set_data(loop_xy[:, 0], loop_xy[:, 1])
            loop_line.set_alpha(max(0.0, 1.0 - t))
            matched_marker.set_offsets(loop_xy)
            matched_marker.set_alpha(max(0.0, 1.0 - t))
            running_err = float(np.linalg.norm(xy[-1] - truth_endpoint))
            phase_label.set_text("Phase 3/3 — pose-graph SE(3) Gauss-Newton")
            title.set_text(
                f"KITTI 00  •  optimizing  •  "
                f"endpoint drift = {running_err:.2f} m"
            )
            return estimate_line, current_marker, loop_line, matched_marker, title, phase_label

        # Hold corrected state.
        estimate_line.set_data(corrected_xy[:, 0], corrected_xy[:, 1])
        estimate_line.set_color(CORR_COLOR)
        current_marker.set_offsets(corrected_xy[-1:].reshape(1, 2))
        current_marker.set_color(CORR_COLOR)
        loop_line.set_alpha(0.0)
        matched_marker.set_alpha(0.0)
        phase_label.set_text("Done — corrected trajectory")
        title.set_text(
            f"KITTI 00  •  done  •  "
            f"drifted {end_drift_err:.1f} m → corrected {end_corr_err:.3f} m"
        )
        return estimate_line, current_marker, loop_line, matched_marker, title, phase_label

    anim = FuncAnimation(fig, update, frames=total_frames, blit=False)
    gif_path = args.out_dir / "kitti_loop_closure.gif"
    anim.save(gif_path, writer=PillowWriter(fps=args.gif_fps))
    plt.close(fig)
    print(f"wrote {gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
