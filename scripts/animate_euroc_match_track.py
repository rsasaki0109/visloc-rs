#!/usr/bin/env python3
"""Animate a UAV SLAM run as a two-panel GIF: SuperPoint feature matches on the
camera image (left) next to the growing 2D estimated trajectory (right).

Left  - the cam0 image at each tracked frame with the SuperPoint matches to the
        previous tracked frame drawn as motion vectors (mutual-NN + Lowe ratio
        on the exported 256-D descriptors).
Right - the online VI-SLAM estimate (red) tracking ground truth (black) in the
        top-down (X, Y) plane, grown frame by frame. Rigidly aligned (the
        reported rigid-ATE convention).

Inputs are what the EuRoC stack already has on disk:
  * --errors-csv    the demo's slam_errors.csv (timestamp, frame_idx, gt/est xyz)
  * --image-dir     EuRoC mav0/cam0/data (images named <timestamp_ns>.png)
  * --features-dir  exported SuperPoint per-frame features
                    (frame_<frame_idx:06d>_features.txt, "# X Y SCORE D0 D1 ...")

Usage:
    python3 scripts/animate_euroc_match_track.py \\
        --errors-csv   target/euroc_phase26_MH_01_easy_strict_superpoint/slam_errors.csv \\
        --image-dir    <euroc>/MH_01_easy/mav0/cam0/data \\
        --features-dir target/euroc_phase26_superpoint/MH_01_easy/cam0 \\
        --output       docs/assets/euroc_mh01_match_track.gif \\
        --title 'EuRoC MH_01 (UAV) - SuperPoint matches + online VI-SLAM trajectory'

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
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--title", type=str, default="EuRoC UAV - SuperPoint matches + trajectory")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--align", choices=["rigid", "sim", "none"], default="rigid")
    p.add_argument("--trim-tail-error", type=float, default=0.0)
    p.add_argument("--top-features", type=int, default=600, help="keep the N highest-score features per frame before matching")
    p.add_argument("--max-draw", type=int, default=220, help="cap drawn match lines for legibility")
    p.add_argument("--ratio", type=float, default=0.85, help="Lowe ratio threshold")
    return p.parse_args()


def umeyama(src, dst, with_scale: bool):
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


def load_features(path: Path, top_n: int):
    """Return (xy[N,2], desc[N,256] L2-normalised), keeping the top_n by score."""
    import numpy as np

    arr = np.loadtxt(path, comments="#", ndmin=2)
    if arr.size == 0:
        return np.zeros((0, 2)), np.zeros((0, 1))
    xy = arr[:, :2]
    score = arr[:, 2]
    desc = arr[:, 3:]
    if top_n and len(arr) > top_n:
        keep = np.argpartition(-score, top_n)[:top_n]
        xy, desc = xy[keep], desc[keep]
    desc = desc / (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-9)
    return xy, desc


def match(xy_a, da, xy_b, db, ratio: float):
    """Mutual-NN + Lowe ratio. a = current, b = previous. Returns (cur_xy, prev_xy)."""
    import numpy as np

    if len(da) < 2 or len(db) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2))
    sim = da @ db.T  # cosine similarity, higher = closer
    # a -> b best two
    nn = np.argpartition(-sim, 1, axis=1)[:, :2]
    rows = np.arange(len(da))[:, None]
    top2 = sim[rows, nn]
    order = np.argsort(-top2, axis=1)
    best_b = nn[rows, order][:, 0]
    s1 = top2[rows, order][:, 0]
    s2 = top2[rows, order][:, 1]
    # cosine -> distance d = sqrt(2-2s); Lowe ratio d1/d2 < ratio  <=>  (2-2s1) < ratio^2 (2-2s2)
    d1 = np.sqrt(np.clip(2 - 2 * s1, 0, None))
    d2 = np.sqrt(np.clip(2 - 2 * s2, 0, None))
    good = d1 < ratio * (d2 + 1e-9)
    # mutual: b's best back to a
    best_a = np.argmax(sim, axis=0)  # for each b, best a
    mutual = best_a[best_b] == np.arange(len(da))
    keep = good & mutual
    ia = np.arange(len(da))[keep]
    ib = best_b[keep]
    return xy_a[ia], xy_b[ib]


def main() -> int:
    args = parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.collections import LineCollection
        from matplotlib.gridspec import GridSpec
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        print(f"missing dependency: {exc} (need matplotlib + numpy + pillow)", file=sys.stderr)
        return 2

    rows = []
    with args.errors_csv.open() as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    ts = [r["timestamp_ns"] for r in rows]
    fidx = [int(r["frame_idx"]) for r in rows]
    gt = np.array([[float(r["gt_px"]), float(r["gt_py"]), float(r["gt_pz"])] for r in rows])
    est = np.array([[float(r["est_px"]), float(r["est_py"]), float(r["est_pz"])] for r in rows])

    if args.align == "rigid":
        est = umeyama(est, gt, False)
    elif args.align == "sim":
        est = umeyama(est, gt, True)

    if args.trim_tail_error > 0.0:
        e = np.linalg.norm(est - gt, axis=1)
        keep = len(gt)
        while keep > 1 and e[keep - 1] > args.trim_tail_error:
            keep -= 1
        ts, fidx, gt, est = ts[:keep], fidx[:keep], gt[:keep], est[:keep]

    n = len(gt)
    err = np.linalg.norm(est - gt, axis=1)
    rmse = float(np.sqrt((err ** 2).mean()))

    # Pre-compute matches between consecutive tracked frames (cur vs previous).
    print(f"matching {n} frames ...", file=sys.stderr)
    feat = [load_features(args.features_dir / f"frame_{f:06d}_features.txt", args.top_features) for f in fidx]
    matches = [None]
    for i in range(1, n):
        ca, cb = match(feat[i][0], feat[i][1], feat[i - 1][0], feat[i - 1][1], args.ratio)
        matches.append((ca, cb))

    import matplotlib.image as mpimg

    def load_img(i):
        return mpimg.imread(str(args.image_dir / f"{ts[i]}.png"))

    img0 = load_img(0)
    h, w = img0.shape[:2]

    # --- figure ---
    fig = plt.figure(figsize=(9.0, 3.7), dpi=80)
    gs = GridSpec(1, 2, width_ratios=[w / h * 0.95, 1.0], figure=fig)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    imart = axL.imshow(img0, cmap="gray", vmin=0, vmax=255 if img0.max() > 1.5 else 1.0)
    axL.set_xlim(0, w)
    axL.set_ylim(h, 0)
    axL.set_xticks([])
    axL.set_yticks([])
    axL.set_title("SuperPoint matches (cam0)", fontsize=10)
    lc = LineCollection([], colors="#16c79a", linewidths=0.8, alpha=0.9)
    axL.add_collection(lc)
    scat = axL.scatter([], [], s=7, c="#ffd166", edgecolors="none")

    # --- trajectory ---
    gt_color, est_color = "#3a3a3a", "#d23737"
    allxy = np.vstack([gt[:, :2], est[:, :2]])
    lo, hi = allxy.min(0), allxy.max(0)
    center = (lo + hi) / 2
    half = (hi - lo).max() / 2 * 1.12 + 0.05
    axR.set_xlim(center[0] - half, center[0] + half)
    axR.set_ylim(center[1] - half, center[1] + half)
    axR.set_aspect("equal", adjustable="box")
    axR.set_xlabel("X [m]", fontsize=9)
    axR.set_ylabel("Y [m]", fontsize=9)
    axR.grid(True, alpha=0.3)
    axR.set_title("Online VI-SLAM estimate (2D)", fontsize=10)
    axR.plot(gt[:, 0], gt[:, 1], color=gt_color, lw=1.0, alpha=0.15)
    (gt_xy,) = axR.plot([], [], color=gt_color, lw=2.0, label="ground truth")
    (est_xy,) = axR.plot([], [], color=est_color, lw=1.5, ls="--", label="estimate")
    (drone,) = axR.plot([], [], marker="o", color=est_color, ms=9, mfc=est_color, mec="white", mew=1.2)
    axR.legend(loc="best", fontsize=9, framealpha=0.85)

    suptitle = fig.suptitle(args.title, fontsize=11)

    def update(i):
        imart.set_data(load_img(i))
        m = matches[i]
        if m is not None and len(m[0]):
            cur, prev = m
            ndraw = min(len(cur), args.max_draw)
            sel = np.linspace(0, len(cur) - 1, ndraw).astype(int) if len(cur) > ndraw else np.arange(len(cur))
            segs = np.stack([prev[sel], cur[sel]], axis=1)
            lc.set_segments(segs)
            scat.set_offsets(cur[sel])
            nmatch = len(cur)
        else:
            lc.set_segments([])
            scat.set_offsets(np.zeros((0, 2)))
            nmatch = 0
        axL.set_title(f"SuperPoint matches (cam0): {nmatch}", fontsize=10)

        k = i + 1
        gt_xy.set_data(gt[:k, 0], gt[:k, 1])
        est_xy.set_data(est[:k, 0], est[:k, 1])
        drone.set_data([est[i, 0]], [est[i, 1]])
        running = float(np.sqrt((err[:k] ** 2).mean()))
        suptitle.set_text(f"{args.title}\nframe {k}/{n}    {nmatch} matches    running rigid RMSE {running * 100:.1f} cm")
        return imart, lc, scat, gt_xy, est_xy, drone

    anim = FuncAnimation(fig, update, frames=n, blit=False, interval=1000 / args.fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    anim.save(str(args.output), writer=PillowWriter(fps=args.fps))

    try:
        from PIL import Image, ImageSequence

        src = Image.open(args.output)
        rgb = [f.convert("RGB") for f in ImageSequence.Iterator(src)]
        # One shared (global) palette for every frame: matching palettes let the
        # GIF encoder store only inter-frame pixel diffs (the static trajectory
        # panel and title barely change), which a per-frame adaptive palette
        # defeats - the difference between a ~25 MB and a few-MB photo GIF.
        sample = rgb[len(rgb) // 2]
        pal = sample.quantize(colors=64, method=Image.FASTOCTREE)
        quant = [f.quantize(palette=pal, dither=Image.NONE) for f in rgb]
        quant[0].save(
            args.output,
            save_all=True,
            append_images=quant[1:],
            loop=0,
            duration=int(1000 / args.fps),
            optimize=True,
            disposal=1,
        )
    except Exception as exc:  # pragma: no cover
        print(f"(gif optimisation skipped: {exc})", file=sys.stderr)

    size_mb = args.output.stat().st_size / 1e6
    print(f"wrote {args.output} ({n} frames, final rigid RMSE {rmse * 100:.1f} cm, {size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
