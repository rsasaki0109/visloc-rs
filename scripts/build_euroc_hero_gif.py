#!/usr/bin/env python3
"""Build the README hero GIF for EuRoC online stereo SLAM.

Two-panel layout (modelled on the committed ``animate_euroc_*`` tools):
  * left  — onboard cam0 grayscale image for the current tracked frame
  * right — top-down live map: full ground-truth flight path (faint context),
            rigidly-aligned estimated trajectory grown in sync (Umeyama, no
            scale), current pose marker, and landmark map points revealed as
            the camera first comes within a radius of each point.

False-relocalization spikes (``position_error_m`` above a threshold) are omitted
from the estimated trajectory so gaps stay honest instead of drawing straight
bridges across bad recoveries. Frame-index dropouts get the same treatment.

Usage:
    python3 scripts/build_euroc_hero_gif.py \\
        --errors-csv  E:/visloc_archive/.../reloc_fix/slam_errors.csv \\
        --landmarks-csv E:/visloc_archive/.../reloc_fix/slam_landmarks.csv \\
        --image-dir   <euroc>/MH_01_easy/mav0/cam0/data \\
        --gt-csv      <euroc>/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv \\
        --output      docs/assets/hero_euroc_mh01_slam.gif

Asset-generation tool, not part of CI.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


# README hero palette (matched to committed docs/assets/hero_euroc_mh01_slam.gif)
BG = "#1a1a1a"
PANEL = "#222222"
GT_COLOR = "#8a8a8a"
LM_COLOR = "#1fb883"
EST_COLOR = "#5887bc"
TEXT = "#e8e8e8"
MUTED = "#9aa7b4"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--errors-csv", type=Path, required=True)
    p.add_argument("--landmarks-csv", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--gt-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--stride", type=int, default=17, help="keep every Nth tracked row")
    p.add_argument("--align", choices=["rigid", "sim", "none"], default="rigid")
    p.add_argument(
        "--max-position-error",
        type=float,
        default=0.5,
        help="exclude rows above this aligned error from the estimate polyline",
    )
    p.add_argument(
        "--landmark-reveal-radius",
        type=float,
        default=2.5,
        help="reveal a map point once the aligned camera is within this many metres",
    )
    p.add_argument("--dpi", type=int, default=92)
    p.add_argument("--fig-width", type=float, default=8.27)
    p.add_argument("--fig-height", type=float, default=3.52)
    return p.parse_args()


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
    return (scale * (rot @ src.T)).T + t, scale, rot, t


def apply_umeyama(pts, scale, rot, t):
    import numpy as np

    return (scale * (rot @ pts.T)).T + t


def load_errors(path: Path):
    rows = list(csv.DictReader(path.open()))
    ts = [r["timestamp_ns"] for r in rows]
    fidx = [int(r["frame_idx"]) for r in rows]
    gt = [[float(r["gt_px"]), float(r["gt_py"]), float(r["gt_pz"])] for r in rows]
    est = [[float(r["est_px"]), float(r["est_py"]), float(r["est_pz"])] for r in rows]
    raw_err = [float(r["position_error_m"]) for r in rows]
    return rows, ts, fidx, gt, est, raw_err


def load_landmarks(path: Path):
    import numpy as np

    pts = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            pts.append([float(row["x"]), float(row["y"]), float(row["z"])])
    return np.array(pts) if pts else np.zeros((0, 3))


def load_full_gt(path: Path):
    import numpy as np

    pts = []
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.split(",")
            pts.append([float(cols[1]), float(cols[2]), float(cols[3])])
    return np.array(pts)


def optimise_gif(path: Path, fps: int) -> None:
    from PIL import Image, ImageSequence

    src = Image.open(path)
    rgb = [f.convert("RGB") for f in ImageSequence.Iterator(src)]
    sample = rgb[len(rgb) // 2]
    pal = sample.quantize(colors=128, method=Image.FASTOCTREE)
    quant = [f.quantize(palette=pal, dither=Image.NONE) for f in rgb]
    quant[0].save(
        path,
        save_all=True,
        append_images=quant[1:],
        loop=0,
        duration=int(1000 / fps),
        optimize=True,
        disposal=1,
    )


def main() -> int:
    args = parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        print(f"missing dependency: {exc} (need matplotlib + numpy + pillow)", file=sys.stderr)
        return 2

    rows, ts, fidx, gt, est, raw_err = load_errors(args.errors_csv)
    if len(rows) < 10:
        print(f"too few rows in {args.errors_csv}", file=sys.stderr)
        return 1

    gt = np.array(gt)
    est = np.array(est)
    fidx = np.array(fidx)
    raw_err = np.array(raw_err)

    # Align on trustworthy rows so false-reloc spikes do not skew the rigid fit.
    good_for_align = raw_err <= args.max_position_error
    if good_for_align.sum() < 10:
        good_for_align = np.ones(len(gt), dtype=bool)

    if args.align == "rigid":
        est_aligned, scale, rot, t = umeyama(est[good_for_align], gt[good_for_align], False)
        est = apply_umeyama(est, scale, rot, t)
    elif args.align == "sim":
        est_aligned, scale, rot, t = umeyama(est[good_for_align], gt[good_for_align], True)
        est = apply_umeyama(est, scale, rot, t)
    else:
        scale, rot, t = 1.0, np.eye(3), np.zeros(3)

    aligned_err = np.linalg.norm(est - gt, axis=1)
    good = aligned_err <= args.max_position_error
    n_bad = int((~good).sum())
    if n_bad:
        print(f"excluding {n_bad} frame(s) with aligned error > {args.max_position_error} m from estimate polyline")

    rmse = float(np.sqrt((aligned_err[good] ** 2).mean()))

    landmarks = load_landmarks(args.landmarks_csv)
    if len(landmarks):
        landmarks = apply_umeyama(landmarks, scale, rot, t)

    full_gt = load_full_gt(args.gt_csv)

    n_tracked = len(rows)
    span = int(fidx[-1] - fidx[0] + 1)
    coverage = n_tracked / span if span else 1.0
    est_full = est.copy()
    fidx_full = fidx.copy()

    # Landmark reveal frame: first sequence frame where camera is close enough.
    reveal_idx = np.full(len(landmarks), 10**9, dtype=int)
    if len(landmarks):
        dists = np.linalg.norm(est_full[None, :, :] - landmarks[:, None, :], axis=2)
        within = dists <= args.landmark_reveal_radius
        has = within.any(axis=1)
        first_i = within.argmax(axis=1)
        reveal_idx[has] = fidx_full[first_i[has]]

    # Segment membership on the full (unstrided) series so striding does not
    # look like frame-index gaps. Break segments on tracking dropouts or spikes.
    segment_id = np.zeros(n_tracked, dtype=int)
    for i in range(1, n_tracked):
        if fidx[i] - fidx[i - 1] > 1 or not good[i]:
            segment_id[i] = segment_id[i - 1] + 1
        else:
            segment_id[i] = segment_id[i - 1]

    if args.stride > 1:
        sel = np.arange(0, n_tracked, args.stride)
        ts = [ts[i] for i in sel]
        fidx = fidx[sel]
        gt = gt[sel]
        est = est[sel]
        good = good[sel]
        aligned_err = aligned_err[sel]
        segment_id = segment_id[sel]

    n = len(ts)

    def load_img(i: int):
        return mpimg.imread(str(args.image_dir / f"{ts[i]}.png"))

    img0 = load_img(0)
    h, w = img0.shape[:2]
    vmax = 255.0 if img0.max() > 1.5 else 1.0

    fig = plt.figure(figsize=(args.fig_width, args.fig_height), dpi=args.dpi, facecolor=BG)
    fig_w_in, fig_h_in = fig.get_size_inches()
    canvas_w = fig_w_in * args.dpi
    canvas_h = fig_h_in * args.dpi

    # Camera panel: ~60 % of canvas width, nearly full height (minimal margins).
    panel_h_px = canvas_h * 0.94
    panel_w_px = panel_h_px * (w / h)
    cam_x = 0.006
    cam_y = 0.012
    cam_w = panel_w_px / canvas_w
    cam_h = panel_h_px / canvas_h
    map_x = cam_x + cam_w + 0.008
    map_w = 0.992 - map_x
    map_h = cam_h

    ax_cam = fig.add_axes([cam_x, cam_y, cam_w, cam_h], facecolor=PANEL)
    ax_map = fig.add_axes([map_x, cam_y, map_w, map_h], facecolor=PANEL)

    fig.text(0.008, 0.985, "visloc-rs", color=TEXT, fontsize=11, fontweight="bold", va="top")
    fig.text(
        0.118,
        0.985,
        "online stereo SLAM · EuRoC MH_01 · pure Rust",
        color=MUTED,
        fontsize=8,
        va="top",
    )
    fig.text(
        0.992,
        0.985,
        f"ATE RMSE {rmse * 100:.1f} cm",
        color=EST_COLOR,
        fontsize=10,
        fontweight="bold",
        ha="right",
        va="top",
    )

    imart = ax_cam.imshow(img0, cmap="gray", vmin=0, vmax=vmax)
    ax_cam.set_xlim(0, w)
    ax_cam.set_ylim(h, 0)
    ax_cam.set_xticks([])
    ax_cam.set_yticks([])
    for spine in ax_cam.spines.values():
        spine.set_color("#333333")
    cam_label = ax_cam.text(
        0.02, 0.97, "onboard camera", transform=ax_cam.transAxes, color=MUTED, fontsize=8, va="top"
    )

    allxy = np.vstack([full_gt[:, :2], gt[:, :2], est[good, :2] if good.any() else est[:, :2]])
    lo, hi = allxy.min(0), allxy.max(0)
    center = (lo + hi) / 2
    half = (hi - lo).max() / 2 * 1.10 + 0.05
    ax_map.set_xlim(center[0] - half, center[0] + half)
    ax_map.set_ylim(center[1] - half, center[1] + half)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_color("#333333")
    ax_map.grid(True, color="#333333", alpha=0.55, linewidth=0.6)

    ax_map.plot(full_gt[:, 0], full_gt[:, 1], color=GT_COLOR, lw=1.0, ls="--", alpha=0.45, zorder=1)
    lm_scatter = ax_map.scatter(
        [], [], s=3, c=LM_COLOR, alpha=0.55, linewidths=0, edgecolors="none", zorder=2
    )
    (est_xy,) = ax_map.plot(
        [], [], color=EST_COLOR, lw=2.2, alpha=0.95, solid_capstyle="round", zorder=4
    )
    (drone,) = ax_map.plot(
        [],
        [],
        marker="o",
        color=EST_COLOR,
        ms=7,
        mfc=EST_COLOR,
        mec="white",
        mew=1.2,
        zorder=5,
    )

    ax_map.text(0.03, 0.97, "live map", transform=ax_map.transAxes, color=TEXT, fontsize=10, fontweight="bold", va="top")
    map_sub = ax_map.text(0.03, 0.88, "", transform=ax_map.transAxes, color=LM_COLOR, fontsize=8.5, va="top")
    ax_map.text(0.03, 0.08, "ground truth", transform=ax_map.transAxes, color=GT_COLOR, fontsize=7.5, va="bottom")
    ax_map.text(0.03, 0.02, "estimated", transform=ax_map.transAxes, color=EST_COLOR, fontsize=7.5, va="bottom")

    # Precompute segment endpoint pairs for prefixes.
    def est_path_upto(k: int):
        xs, ys = [], []
        for i in range(1, k + 1):
            if not (good[i - 1] and good[i]):
                continue
            if segment_id[i] != segment_id[i - 1]:
                continue
            xs.extend([est[i - 1, 0], est[i, 0], np.nan])
            ys.extend([est[i - 1, 1], est[i, 1], np.nan])
        return xs, ys

    last_good_pose = est[0] if good[0] else None

    def update(i: int):
        nonlocal last_good_pose
        imart.set_data(load_img(i))

        est_xy.set_data(*est_path_upto(i))

        if good[i]:
            last_good_pose = est[i]
        if last_good_pose is not None:
            drone.set_data([last_good_pose[0]], [last_good_pose[1]])
        else:
            drone.set_data([], [])

        revealed = reveal_idx <= int(fidx[i])
        if revealed.any():
            pts = landmarks[revealed]
            lm_scatter.set_offsets(pts[:, :2])
            n_lm = int(revealed.sum())
        else:
            lm_scatter.set_offsets(np.zeros((0, 2)))
            n_lm = 0

        map_sub.set_text(f"{n_lm:,} map points")
        cam_label.set_text("onboard camera")
        return imart, est_xy, drone, lm_scatter, map_sub, cam_label

    anim = FuncAnimation(fig, update, frames=n, blit=False, interval=1000 / args.fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(args.output), writer=PillowWriter(fps=args.fps))

    try:
        optimise_gif(args.output, args.fps)
    except Exception as exc:  # pragma: no cover
        print(f"(gif optimisation skipped: {exc})", file=sys.stderr)

    size_mb = args.output.stat().st_size / 1e6
    print(
        f"wrote {args.output} ({n} gif frames from {n_tracked} tracked rows, "
        f"coverage {coverage * 100:.1f} %, rigid RMSE {rmse * 100:.1f} cm on good frames, "
        f"{size_mb:.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
