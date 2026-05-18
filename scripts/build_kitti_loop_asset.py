#!/usr/bin/env python3
"""Build the README KITTI loop-closure asset.

Two modes are supported:

1. **gt-drift mode** (default): reads `truth.csv`, `drifted.csv`, and
   `corrected.csv` produced by the GT-pose-based example
   `online_slam_kitti_loop_demo`. The drifted trajectory is integrated
   from yaw-perturbed truth edges; the corrected trajectory is the
   pose-graph SE(3) GN output.

2. **real-vo mode**: reads `vo.csv` + `corrected.csv` produced by the
   real-image example `online_slam_image_vo_loop_demo` plus a separate
   KITTI ground-truth pose file (e.g., `<KITTI>/poses/00.txt`). The
   monocular essential-matrix VO is metric-ambiguous (unit scale per
   pair), so VO and corrected are Procrustes-aligned (rotation + scale +
   translation) to the GT subsample before plotting. This is the same
   alignment used by ATE evaluation in the visual SLAM literature.

3. **stereo mode**: reads `vo.csv`, `ba.csv`, `pgo.csv`, `gt.csv`, and
   optionally `loop_edges.csv` from `online_slam_stereo_vo_kitti_demo`.
   This is the preferred README asset because the rectified stereo
   baseline gives metric scale and the GIF can show the full improvement
   chain: raw stereo VO → bundle adjustment → pose-graph correction.

4. **stereo-vo mode**: reads `vo.csv` and `gt.csv` from the deep stereo VO
   smoke run. It renders the raw metric odometry trajectory and ATE curve
   without requiring BA/PGO outputs, which is the clearest asset for
   leaderboard-style VO tuning.

The output artifacts are `kitti_loop_closure.png` and
`kitti_loop_closure.gif`. In stereo mode they show a two-panel
trajectory + ATE comparison across raw VO, BA, and PGO. The legacy modes
keep the original truth / drifted / corrected visualization.

Asset-generation tool, not part of CI. Requires Python with matplotlib,
numpy, and Pillow.
"""
from __future__ import annotations

import argparse
import csv
import math
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
BA_COLOR = "#f97316"      # orange
PGO_COLOR = "#2563eb"     # blue
GRID_COLOR = "#94a3b8"


def load_trajectory_xyz(path: Path) -> np.ndarray:
    pts: list[tuple[float, float, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            pts.append((float(row["x"]), float(row["y"]), float(row["z"])))
    return np.array(pts)


def load_trajectory_xz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    zs: list[float] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            zs.append(float(row["z"]))
    return np.array(xs), np.array(zs)


def load_loop_edges(path: Path) -> list[dict]:
    if not path.exists():
        return []
    edges: list[dict] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append(
                {
                    "from_id": int(row["from_id"]),
                    "to_id": int(row["to_id"]),
                    "source": row.get("source", "loop"),
                    "from": np.array(
                        [
                            float(row["from_x"]),
                            float(row["from_y"]),
                            float(row["from_z"]),
                        ]
                    ),
                    "to": np.array(
                        [
                            float(row["to_x"]),
                            float(row["to_y"]),
                            float(row["to_z"]),
                        ]
                    ),
                }
            )
    return edges


def ate(estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    n = min(len(estimate), len(truth))
    return np.linalg.norm(estimate[:n] - truth[:n], axis=1)


def metric_summary(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    errors = ate(estimate, truth)
    return {
        "mean": float(errors.mean()),
        "rmse": float(math.sqrt(float(np.mean(errors * errors)))),
        "max": float(errors.max()),
    }


def xz(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return points[:, 0], points[:, 2]


def load_kitti_truth_xz(
    path: Path, stride: int, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Read camera-to-world poses (3x4 row-major), pick stride-subsampled
    centers, return X / Z columns of length `count`."""
    rows = np.loadtxt(path)
    centers = rows[:, [3, 7, 11]]  # tx, ty, tz columns of 3x4 row-major
    sampled = centers[::stride][:count]
    return sampled[:, 0], sampled[:, 2]


def procrustes_2d(
    src: np.ndarray, dst: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Start-anchored similarity alignment (rotation + uniform scale) of
    `src` to `dst`. Both arrays are (N, 2). The output is forced to share
    the same start point as `dst` (i.e., `aligned[0] == dst[0]`) so the
    visualization compares trajectory shape relative to a common origin
    instead of letting an unconstrained Procrustes alignment drift the
    centroid of a loop trajectory away from its start."""
    src_anchor = src[0]
    dst_anchor = dst[0]
    src_c = src - src_anchor
    dst_c = dst - dst_anchor
    # Optimal rotation via SVD of cross-covariance (origin-anchored).
    h = src_c.T @ dst_c
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    src_norm = np.linalg.norm(src_c)
    dst_norm = np.linalg.norm(dst_c)
    scale = dst_norm / max(src_norm, 1e-12)
    aligned = (scale * (src_c @ r.T)) + dst_anchor
    return aligned, {"rotation": r, "scale": scale,
                     "src_anchor": src_anchor, "dst_anchor": dst_anchor}


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def render_stereo_asset(args: argparse.Namespace) -> int:
    frontend_label = args.frontend_label
    frontend_title = frontend_label[:1].upper() + frontend_label[1:]
    truth = load_trajectory_xyz(args.input_dir / "gt.csv")
    vo = load_trajectory_xyz(args.input_dir / "vo.csv")
    ba = load_trajectory_xyz(args.input_dir / "ba.csv")
    pgo = load_trajectory_xyz(args.input_dir / "pgo.csv")
    edges = load_loop_edges(args.input_dir / "loop_edges.csv")
    n = min(len(truth), len(vo), len(ba), len(pgo))
    truth = truth[:n]
    vo = vo[:n]
    ba = ba[:n]
    pgo = pgo[:n]

    vo_m = metric_summary(vo, truth)
    ba_m = metric_summary(ba, truth)
    pgo_m = metric_summary(pgo, truth)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    truth_x, truth_z = xz(truth)
    vo_x, vo_z = xz(vo)
    ba_x, ba_z = xz(ba)
    pgo_x, pgo_z = xz(pgo)

    def set_stereo_limits(ax):
        all_x = np.r_[truth_x, vo_x, ba_x, pgo_x]
        all_z = np.r_[truth_z, vo_z, ba_z, pgo_z]
        x_span = max(float(all_x.max() - all_x.min()), 6.0)
        z_span = max(float(all_z.max() - all_z.min()), 6.0)
        x_mid = float((all_x.max() + all_x.min()) * 0.5)
        ax.set_xlim(x_mid - x_span * 0.62, x_mid + x_span * 0.62)
        ax.set_ylim(float(all_z.min()) - z_span * 0.04, float(all_z.max()) + z_span * 0.06)

    # ---------- Static README PNG ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    ax = axes[0]
    ax.plot(truth_x, truth_z, color=TRUTH_COLOR, linewidth=2.8, label="ground truth")
    ax.plot(vo_x, vo_z, color=DRIFT_COLOR, linewidth=1.8, label=frontend_label)
    ax.plot(ba_x, ba_z, color=BA_COLOR, linewidth=2.0, label="after BA")
    ax.plot(pgo_x, pgo_z, color=PGO_COLOR, linewidth=2.4, label="after PGO")
    ax.scatter([truth_x[0]], [truth_z[0]], color="#111827", s=42, zorder=5, label="start")
    for edge in edges:
        frm = edge["from"]
        to = edge["to"]
        ax.plot([frm[0], to[0]], [frm[2], to[2]], "--", color=LOOP_COLOR, linewidth=1.8)
        ax.scatter([frm[0], to[0]], [frm[2], to[2]], color=LOOP_COLOR, s=32, zorder=6)
    ax.set_title(f"{frontend_title} KITTI 00 trajectory")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    set_stereo_limits(ax)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper left", fontsize=9)

    ax_err = axes[1]
    ax_err.plot(
        ate(vo, truth),
        color=DRIFT_COLOR,
        linewidth=1.9,
        label=f"{frontend_label} mean {vo_m['mean']:.2f} m",
    )
    ax_err.plot(ate(ba, truth), color=BA_COLOR, linewidth=2.0, label=f"BA mean {ba_m['mean']:.2f} m")
    ax_err.plot(ate(pgo, truth), color=PGO_COLOR, linewidth=2.4, label=f"PGO mean {pgo_m['mean']:.2f} m")
    ax_err.set_title("Per-frame translation error")
    ax_err.set_xlabel("keyframe")
    ax_err.set_ylabel("ATE [m]")
    ax_err.grid(True, alpha=0.28)
    ax_err.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        f"{frontend_title} on KITTI 00: strong raw odometry plus BA and PGO "
        f"(max ATE {vo_m['max']:.2f} m → {pgo_m['max']:.2f} m; BA {ba_m['max']:.2f} m)",
        fontsize=13,
    )
    png_path = args.out_dir / "kitti_loop_closure.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")

    # ---------- Animated README GIF ----------
    fig, (ax, ax_err_anim) = plt.subplots(1, 2, figsize=(11.2, 5.8), constrained_layout=True)
    set_stereo_limits(ax)
    ax.grid(True, alpha=0.28)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.plot(truth_x, truth_z, color=TRUTH_COLOR, linewidth=2.0, alpha=0.38, label="ground truth")
    ax.scatter([truth_x[0]], [truth_z[0]], color="#111827", s=45, zorder=5, label="start")
    estimate_line, = ax.plot([], [], color=DRIFT_COLOR, linewidth=2.8, label="estimate", zorder=4)
    current_marker = ax.scatter([], [], color=DRIFT_COLOR, s=70, zorder=6)
    loop_line, = ax.plot([], [], color=LOOP_COLOR, linewidth=2.0, alpha=0.0, zorder=3)
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
        bbox=dict(boxstyle="round,pad=0.3", fc="#0f172a", ec="#1e293b", alpha=0.88),
        color="white",
    )
    metric_label = ax.text(
        0.02,
        0.08,
        "",
        transform=ax.transAxes,
        verticalalignment="bottom",
        horizontalalignment="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cbd5e1", alpha=0.88),
        color="#0f172a",
    )
    ax.legend(loc="upper right", fontsize=8)

    vo_errors = ate(vo, truth)
    ba_errors = ate(ba, truth)
    pgo_errors = ate(pgo, truth)
    x_frames = np.arange(n)
    ax_err_anim.plot(x_frames, vo_errors, color=DRIFT_COLOR, alpha=0.22, linewidth=1.4)
    ax_err_anim.plot(x_frames, ba_errors, color=BA_COLOR, alpha=0.22, linewidth=1.4)
    ax_err_anim.plot(x_frames, pgo_errors, color=PGO_COLOR, alpha=0.22, linewidth=1.4)
    active_err_line, = ax_err_anim.plot([], [], color=DRIFT_COLOR, linewidth=2.6)
    progress_line = ax_err_anim.axvline(0, color="#0f172a", alpha=0.35, linewidth=1.2)
    ax_err_anim.set_xlim(0, max(1, n - 1))
    ax_err_anim.set_ylim(0, max(float(vo_errors.max()), float(ba_errors.max()), float(pgo_errors.max())) * 1.08)
    ax_err_anim.set_title("ATE vs ground truth")
    ax_err_anim.set_xlabel("keyframe")
    ax_err_anim.set_ylabel("error [m]")
    ax_err_anim.grid(True, alpha=0.28)

    vo_frames = max(args.vo_frames, 4)
    ba_frames = max(args.detect_frames + args.optim_frames // 2, 10)
    pgo_frames = max(args.optim_frames // 2, 10)
    hold_frames = max(args.hold_frames, 1)
    total_frames = vo_frames + ba_frames + pgo_frames + hold_frames

    def lerp_color(a: str, b: str, t: float) -> tuple[float, float, float, float]:
        ca = np.array(to_rgba(a))
        cb = np.array(to_rgba(b))
        return tuple((1.0 - t) * ca + t * cb)

    def update(frame: int):
        if frame < vo_frames:
            progress = (frame + 1) / vo_frames
            kf_count = max(2, int(round(progress * n)))
            stage = vo[:kf_count]
            sx, sz = xz(stage)
            estimate_line.set_data(sx, sz)
            estimate_line.set_color(DRIFT_COLOR)
            current_marker.set_offsets(stage[-1:, [0, 2]])
            current_marker.set_color(DRIFT_COLOR)
            loop_line.set_alpha(0.0)
            active_err_line.set_data(x_frames[:kf_count], vo_errors[:kf_count])
            active_err_line.set_color(DRIFT_COLOR)
            progress_line.set_xdata([kf_count - 1, kf_count - 1])
            phase_label.set_text(f"Phase 1/3 — {frontend_label}")
            title.set_text(f"KITTI 00 {frontend_label} • keyframe {kf_count}/{n}")
            metric_label.set_text(
                f"raw {frontend_label}: mean ATE {vo_m['mean']:.2f} m, max {vo_m['max']:.2f} m"
            )
            return (
                estimate_line,
                current_marker,
                loop_line,
                title,
                phase_label,
                metric_label,
                active_err_line,
                progress_line,
            )

        if frame < vo_frames + ba_frames:
            local = frame - vo_frames
            t = smoothstep(local / max(1, ba_frames - 1))
            interp = (1.0 - t) * vo + t * ba
            ix, iz = xz(interp)
            estimate_line.set_data(ix, iz)
            estimate_line.set_color(lerp_color(DRIFT_COLOR, BA_COLOR, t))
            current_marker.set_offsets(interp[-1:, [0, 2]])
            current_marker.set_color(lerp_color(DRIFT_COLOR, BA_COLOR, t))
            loop_line.set_alpha(0.0)
            running = metric_summary(interp, truth)
            active_err_line.set_data(x_frames, ate(interp, truth))
            active_err_line.set_color(lerp_color(DRIFT_COLOR, BA_COLOR, t))
            progress_line.set_xdata([n - 1, n - 1])
            phase_label.set_text("Phase 2/3 — sparse stereo BA")
            title.set_text("Bundle adjustment refines the stereo track graph")
            metric_label.set_text(
                f"mean ATE {vo_m['mean']:.2f} → {ba_m['mean']:.2f} m  "
                f"(now {running['mean']:.2f} m)"
            )
            return (
                estimate_line,
                current_marker,
                loop_line,
                title,
                phase_label,
                metric_label,
                active_err_line,
                progress_line,
            )

        if frame < vo_frames + ba_frames + pgo_frames:
            local = frame - (vo_frames + ba_frames)
            t = smoothstep(local / max(1, pgo_frames - 1))
            interp = (1.0 - t) * ba + t * pgo
            ix, iz = xz(interp)
            estimate_line.set_data(ix, iz)
            estimate_line.set_color(lerp_color(BA_COLOR, PGO_COLOR, t))
            current_marker.set_offsets(interp[-1:, [0, 2]])
            current_marker.set_color(lerp_color(BA_COLOR, PGO_COLOR, t))
            if edges:
                edge = edges[0]
                frm = edge["from"]
                to = edge["to"]
                loop_line.set_data([frm[0], to[0]], [frm[2], to[2]])
                loop_line.set_alpha(0.85 * (1.0 - t))
                loop_line.set_color(LOOP_COLOR)
            running = metric_summary(interp, truth)
            active_err_line.set_data(x_frames, ate(interp, truth))
            active_err_line.set_color(lerp_color(BA_COLOR, PGO_COLOR, t))
            progress_line.set_xdata([n - 1, n - 1])
            phase_label.set_text("Phase 3/3 — pose graph correction")
            title.set_text("Loop edge + SE(3) PGO pulls the path onto ground truth")
            metric_label.set_text(
                f"mean ATE {ba_m['mean']:.2f} → {pgo_m['mean']:.2f} m  "
                f"(now {running['mean']:.2f} m)"
            )
            return (
                estimate_line,
                current_marker,
                loop_line,
                title,
                phase_label,
                metric_label,
                active_err_line,
                progress_line,
            )

        estimate_line.set_data(pgo_x, pgo_z)
        estimate_line.set_color(PGO_COLOR)
        current_marker.set_offsets(pgo[-1:, [0, 2]])
        current_marker.set_color(PGO_COLOR)
        loop_line.set_alpha(0.0)
        active_err_line.set_data(x_frames, pgo_errors)
        active_err_line.set_color(PGO_COLOR)
        progress_line.set_xdata([n - 1, n - 1])
        phase_label.set_text("Done — BA + PGO trajectory")
        title.set_text(f"KITTI 00 {frontend_label} recovered")
        metric_label.set_text(
            f"mean ATE {vo_m['mean']:.2f} → {ba_m['mean']:.2f} → {pgo_m['mean']:.2f} m; "
            f"max {vo_m['max']:.2f} → {pgo_m['max']:.2f} m"
        )
        return (
            estimate_line,
            current_marker,
            loop_line,
            title,
            phase_label,
            metric_label,
            active_err_line,
            progress_line,
        )

    anim = FuncAnimation(fig, update, frames=total_frames, blit=False)
    gif_path = args.out_dir / "kitti_loop_closure.gif"
    anim.save(gif_path, writer=PillowWriter(fps=args.gif_fps))
    plt.close(fig)
    print(f"wrote {gif_path}")
    return 0


def render_stereo_vo_asset(args: argparse.Namespace) -> int:
    frontend_label = args.frontend_label
    frontend_title = frontend_label[:1].upper() + frontend_label[1:]
    truth = load_trajectory_xyz(args.input_dir / "gt.csv")
    vo = load_trajectory_xyz(args.input_dir / "vo.csv")
    n = min(len(truth), len(vo))
    truth = truth[:n]
    vo = vo[:n]
    vo_m = metric_summary(vo, truth)
    errors = ate(vo, truth)
    truth_x, truth_z = xz(truth)
    vo_x, vo_z = xz(vo)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def set_vo_limits(ax):
        all_x = np.r_[truth_x, vo_x]
        all_z = np.r_[truth_z, vo_z]
        x_span = max(float(all_x.max() - all_x.min()), 8.0)
        z_span = max(float(all_z.max() - all_z.min()), 8.0)
        x_mid = float((all_x.max() + all_x.min()) * 0.5)
        z_mid = float((all_z.max() + all_z.min()) * 0.5)
        span = max(x_span, z_span)
        ax.set_xlim(x_mid - span * 0.56, x_mid + span * 0.56)
        ax.set_ylim(z_mid - span * 0.56, z_mid + span * 0.56)
        ax.set_aspect("equal", adjustable="box")

    # ---------- Static README PNG ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.9), constrained_layout=True)
    ax = axes[0]
    ax.plot(truth_x, truth_z, color=TRUTH_COLOR, linewidth=3.0, label="ground truth")
    ax.plot(vo_x, vo_z, color=PGO_COLOR, linewidth=2.5, label=frontend_label)
    ax.scatter([truth_x[0]], [truth_z[0]], color="#111827", s=44, zorder=5, label="start")
    ax.scatter([truth_x[-1]], [truth_z[-1]], color=TRUTH_COLOR, s=60, marker="s", zorder=6, label="GT end")
    ax.scatter([vo_x[-1]], [vo_z[-1]], color=PGO_COLOR, s=60, marker="s", zorder=6, label="VO end")
    ax.set_title(f"{frontend_title} KITTI 00 trajectory")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    set_vo_limits(ax)
    ax.grid(True, alpha=0.28, color=GRID_COLOR)
    ax.legend(loc="upper left", fontsize=9)

    ax_err = axes[1]
    ax_err.fill_between(np.arange(n), errors, color=PGO_COLOR, alpha=0.14)
    ax_err.plot(errors, color=PGO_COLOR, linewidth=2.5, label="ATE")
    ax_err.axhline(vo_m["mean"], color="#0f172a", linewidth=1.5, linestyle="--", label=f"mean {vo_m['mean']:.2f} m")
    ax_err.set_title("Per-frame translation error")
    ax_err.set_xlabel("frame")
    ax_err.set_ylabel("ATE [m]")
    ax_err.grid(True, alpha=0.28, color=GRID_COLOR)
    ax_err.legend(loc="upper left", fontsize=9)
    fig.suptitle(
        f"{frontend_title} on KITTI 00 seq00 subset: mean/RMSE/max ATE "
        f"{vo_m['mean']:.2f}/{vo_m['rmse']:.2f}/{vo_m['max']:.2f} m",
        fontsize=13,
    )
    png_path = args.out_dir / "kitti_deep_vo.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")

    # ---------- Animated README GIF ----------
    fig, (ax, ax_err_anim) = plt.subplots(1, 2, figsize=(11.2, 5.8), constrained_layout=True)
    set_vo_limits(ax)
    ax.grid(True, alpha=0.28, color=GRID_COLOR)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.plot(truth_x, truth_z, color=TRUTH_COLOR, linewidth=2.3, alpha=0.34, label="ground truth")
    ax.scatter([truth_x[0]], [truth_z[0]], color="#111827", s=46, zorder=5, label="start")
    vo_line, = ax.plot([], [], color=PGO_COLOR, linewidth=3.0, label=frontend_label, zorder=4)
    gt_progress, = ax.plot([], [], color=TRUTH_COLOR, linewidth=2.0, alpha=0.75, zorder=3)
    current_marker = ax.scatter([], [], color=PGO_COLOR, s=74, zorder=6)
    title = ax.set_title("")
    metric_label = ax.text(
        0.02,
        0.08,
        "",
        transform=ax.transAxes,
        verticalalignment="bottom",
        horizontalalignment="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cbd5e1", alpha=0.90),
        color="#0f172a",
    )
    phase_label = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#0f172a", ec="#1e293b", alpha=0.88),
        color="white",
    )
    ax.legend(loc="upper right", fontsize=8)

    x_frames = np.arange(n)
    ax_err_anim.fill_between(x_frames, errors, color=PGO_COLOR, alpha=0.10)
    ax_err_anim.plot(x_frames, errors, color=PGO_COLOR, alpha=0.22, linewidth=1.4)
    active_err_line, = ax_err_anim.plot([], [], color=PGO_COLOR, linewidth=2.7)
    progress_line = ax_err_anim.axvline(0, color="#0f172a", alpha=0.35, linewidth=1.2)
    ax_err_anim.axhline(vo_m["mean"], color="#0f172a", linewidth=1.1, linestyle="--", alpha=0.65)
    ax_err_anim.set_xlim(0, max(1, n - 1))
    ax_err_anim.set_ylim(0, max(float(errors.max()) * 1.12, 1.0))
    ax_err_anim.set_title("ATE vs ground truth")
    ax_err_anim.set_xlabel("frame")
    ax_err_anim.set_ylabel("error [m]")
    ax_err_anim.grid(True, alpha=0.28, color=GRID_COLOR)

    draw_frames = max(args.vo_frames, 8)
    hold_frames = max(args.hold_frames, 1)
    total_frames = draw_frames + hold_frames

    def update(frame: int):
        if frame < draw_frames:
            progress = (frame + 1) / draw_frames
            kf_count = max(2, int(round(progress * n)))
        else:
            kf_count = n
        vo_stage = vo[:kf_count]
        truth_stage = truth[:kf_count]
        sx, sz = xz(vo_stage)
        gx, gz = xz(truth_stage)
        vo_line.set_data(sx, sz)
        gt_progress.set_data(gx, gz)
        current_marker.set_offsets(vo_stage[-1:, [0, 2]])
        active_err_line.set_data(x_frames[:kf_count], errors[:kf_count])
        progress_line.set_xdata([kf_count - 1, kf_count - 1])
        current_mean = float(errors[:kf_count].mean())
        current_max = float(errors[:kf_count].max())
        phase_label.set_text(f"Deep stereo VO • frame {kf_count}/{n}")
        title.set_text("KITTI 00 raw metric visual odometry")
        metric_label.set_text(
            f"ATE mean {current_mean:.2f} m, max {current_max:.2f} m\n"
            f"final mean/RMSE/max {vo_m['mean']:.2f}/{vo_m['rmse']:.2f}/{vo_m['max']:.2f} m"
        )
        return (
            vo_line,
            gt_progress,
            current_marker,
            active_err_line,
            progress_line,
            title,
            metric_label,
            phase_label,
        )

    anim = FuncAnimation(fig, update, frames=total_frames, blit=False)
    gif_path = args.out_dir / "kitti_deep_vo.gif"
    anim.save(gif_path, writer=PillowWriter(fps=args.gif_fps))
    plt.close(fig)
    print(f"wrote {gif_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["gt-drift", "real-vo", "stereo", "stereo-vo"],
        default="gt-drift",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("target/kitti_loop_demo"),
        help="Directory with truth.csv / drifted.csv / corrected.csv (gt-drift) "
             "or vo.csv / corrected.csv (real-vo), "
             "vo.csv / ba.csv / pgo.csv / gt.csv (stereo), or "
             "vo.csv / gt.csv (stereo-vo).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/assets"),
        help="Where to write kitti_loop_closure.{png,gif}",
    )
    parser.add_argument(
        "--truth-kitti-poses",
        type=Path,
        help="(real-vo only) KITTI poses/SS.txt for the underlying sequence.",
    )
    parser.add_argument(
        "--gt-stride",
        type=int,
        default=1,
        help="(real-vo only) Stride used to subsample the KITTI poses to match "
             "the VO frame count. The Rust example uses --frame-stride which "
             "should match this value.",
    )
    parser.add_argument(
        "--vo-frames", type=int, default=70,
        help="Number of GIF frames spent building up the VO polyline.",
    )
    parser.add_argument(
        "--detect-frames", type=int, default=12,
        help="Number of GIF frames spent on the loop-detection pulse.",
    )
    parser.add_argument(
        "--optim-frames", type=int, default=40,
        help="Number of GIF frames spent morphing drifted -> corrected.",
    )
    parser.add_argument(
        "--hold-frames", type=int, default=12,
        help="Number of GIF frames holding the final corrected state.",
    )
    parser.add_argument("--gif-fps", type=int, default=18)
    parser.add_argument(
        "--frontend-label",
        default="metric stereo VO",
        help="(stereo only) Label shown in the rendered asset title.",
    )
    args = parser.parse_args()

    if args.mode == "stereo":
        return render_stereo_asset(args)
    if args.mode == "stereo-vo":
        return render_stereo_vo_asset(args)

    if args.mode == "gt-drift":
        truth_x, truth_z = load_trajectory_xz(args.input_dir / "truth.csv")
        drift_x, drift_z = load_trajectory_xz(args.input_dir / "drifted.csv")
        corr_x, corr_z = load_trajectory_xz(args.input_dir / "corrected.csv")
        mode_label = "GT-pose drift"
    else:
        if args.truth_kitti_poses is None:
            parser.error("--truth-kitti-poses is required in real-vo mode")
        vo_x, vo_z = load_trajectory_xz(args.input_dir / "vo.csv")
        co_x, co_z = load_trajectory_xz(args.input_dir / "corrected.csv")
        truth_x, truth_z = load_kitti_truth_xz(
            args.truth_kitti_poses, args.gt_stride, len(vo_x)
        )
        # Align unit-scale VO and corrected to GT in the XZ plane using
        # similarity Procrustes (rotation + scale + translation).
        truth_xy = np.column_stack([truth_x, truth_z])
        vo_aligned, vo_xform = procrustes_2d(np.column_stack([vo_x, vo_z]), truth_xy)
        co_aligned, co_xform = procrustes_2d(np.column_stack([co_x, co_z]), truth_xy)
        drift_x, drift_z = vo_aligned[:, 0], vo_aligned[:, 1]
        corr_x, corr_z = co_aligned[:, 0], co_aligned[:, 1]
        print(f"# Procrustes alignment (real-vo):")
        print(f"  vo:  scale={vo_xform['scale']:.3f}")
        print(f"  cor: scale={co_xform['scale']:.3f}")
        mode_label = "real-image VO"

    n = len(truth_x)
    assert len(drift_x) == n and len(corr_x) == n, "trajectory lengths must match"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    end_drift_err = float(
        np.hypot(drift_x[-1] - truth_x[-1], drift_z[-1] - truth_z[-1])
    )
    end_corr_err = float(
        np.hypot(corr_x[-1] - truth_x[-1], corr_z[-1] - truth_z[-1])
    )

    drift_label = "drifted odometry" if args.mode == "gt-drift" else "monocular VO (Procrustes-aligned)"
    corr_label = "corrected (SE(3) GN)" if args.mode == "gt-drift" else "after loop closure + SE(3) GN"

    # ---------- Static three-panel PNG ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    truth_kw = dict(color=TRUTH_COLOR, linewidth=2.0, label="ground truth")
    drift_kw = dict(color=DRIFT_COLOR, linewidth=2.0, label=drift_label)
    corr_kw = dict(color=CORR_COLOR, linewidth=2.0, label=corr_label)

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
        f"VO drifts as the loop closes\n({mode_label}, no loop closure applied)"
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
        f"KITTI 00 loop closure ({mode_label}) — {n} keyframes, "
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

    ax.plot(
        truth_x, truth_z, color=TRUTH_COLOR, linewidth=1.5, alpha=0.35,
        label="ground truth", zorder=1,
    )
    ax.scatter(
        truth_x[0], truth_z[0], color="black", s=40, zorder=5, label="start"
    )

    estimate_line, = ax.plot(
        [], [], color=DRIFT_COLOR, linewidth=2.5,
        label="estimate", zorder=4,
    )
    current_marker = ax.scatter(
        [], [], color=DRIFT_COLOR, s=70, marker="o", zorder=6
    )
    loop_line, = ax.plot(
        [], [], color=LOOP_COLOR, linewidth=2.0, alpha=0.0, zorder=3
    )
    matched_marker = ax.scatter(
        [], [], color=LOOP_COLOR, s=120, marker="o",
        facecolor="none", edgecolor=LOOP_COLOR, linewidth=2.0, alpha=0.0, zorder=5,
    )
    title = ax.set_title("")
    phase_label = ax.text(
        0.02, 0.98, "",
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

    drift_phase_label = "Phase 1/3 — visual odometry"
    detect_phase_label = "Phase 2/3 — loop closure detected"
    optim_phase_label = "Phase 3/3 — pose-graph SE(3) Gauss-Newton"

    def update(frame: int):
        if frame < vo_frames:
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
            phase_label.set_text(drift_phase_label)
            if kf_count < n:
                title.set_text(
                    f"KITTI 00 ({mode_label})  •  building VO trajectory  •  "
                    f"keyframe {kf_count}/{n}"
                )
            else:
                running_err = float(np.linalg.norm(xy[-1] - truth_endpoint))
                title.set_text(
                    f"KITTI 00 ({mode_label})  •  VO complete  •  "
                    f"endpoint drift = {running_err:.1f} m"
                )
            return estimate_line, current_marker, loop_line, matched_marker, title, phase_label

        if frame < vo_frames + detect_frames:
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
            phase_label.set_text(detect_phase_label)
            title.set_text(
                f"KITTI 00 ({mode_label})  •  loop closure: KF 0 ↔ KF {n - 1}  •  "
                f"endpoint drift = {end_drift_err:.1f} m"
            )
            return estimate_line, current_marker, loop_line, matched_marker, title, phase_label

        if frame < vo_frames + detect_frames + optim_frames:
            local = frame - (vo_frames + detect_frames)
            t = smoothstep(local / max(1, optim_frames - 1))
            xy = (1.0 - t) * drifted_xy + t * corrected_xy
            estimate_line.set_data(xy[:, 0], xy[:, 1])
            color = lerp_color(DRIFT_COLOR, CORR_COLOR, t)
            estimate_line.set_color(color)
            current_marker.set_offsets(xy[-1:].reshape(1, 2))
            current_marker.set_color(color)
            loop_xy = np.stack([xy[0], xy[-1]], axis=0)
            loop_line.set_data(loop_xy[:, 0], loop_xy[:, 1])
            loop_line.set_alpha(max(0.0, 1.0 - t))
            matched_marker.set_offsets(loop_xy)
            matched_marker.set_alpha(max(0.0, 1.0 - t))
            running_err = float(np.linalg.norm(xy[-1] - truth_endpoint))
            phase_label.set_text(optim_phase_label)
            title.set_text(
                f"KITTI 00 ({mode_label})  •  optimizing  •  "
                f"endpoint drift = {running_err:.2f} m"
            )
            return estimate_line, current_marker, loop_line, matched_marker, title, phase_label

        estimate_line.set_data(corrected_xy[:, 0], corrected_xy[:, 1])
        estimate_line.set_color(CORR_COLOR)
        current_marker.set_offsets(corrected_xy[-1:].reshape(1, 2))
        current_marker.set_color(CORR_COLOR)
        loop_line.set_alpha(0.0)
        matched_marker.set_alpha(0.0)
        phase_label.set_text("Done — corrected trajectory")
        title.set_text(
            f"KITTI 00 ({mode_label})  •  done  •  "
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
