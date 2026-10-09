#!/usr/bin/env python3
"""Render a VI-SLAM comparison GIF: tracking images (left) + trajectories (right).

Left: the two Aria SLAM cameras (cam0 above cam1) with the keypoints the VIO
is tracking at that instant and a short tail of their recent positions
(cyan = cam0, magenta = cam1). Right: top-down trajectories growing in time,
each Sim(3)-aligned to the ground truth over the whole run (as the LaMAria
evaluator does), with the official Score in the legend.

Inputs are files produced elsewhere in this repo:

  --images DIR        EuRoC/ASL sequence root (mav0/cam0, mav0/cam1)
  --tracks FILE       `basalt_euroc_vio_demo --dump-tracks` CSV
  --gt FILE           reference trajectory, TUM with ns timestamps
  --traj LABEL=FILE   one or more estimates (TUM ns); the first is drawn as
                      the highlighted one. Append `@SCORE` to show a score.

Example:

  python scripts/render_vislam_gif.py --images seq/sequence_1_19 \\
      --tracks gif_tracks.csv --gt sequence_1_19_pgt.txt \\
      --traj "visloc-rs=ours.txt@50.0" --traj "OpenVINS=ovins.txt@49.9" \\
      --out docs/assets/hero_vislam_lamaria.gif
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#2f7ed8", "#f28f43", "#d62728", "#8bbc21", "#910000"]


def load_tum(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    return data[np.argsort(data[:, 0])]


def associate(est: np.ndarray, ref: np.ndarray, tol_ns: float = 2e7):
    idx = np.clip(np.searchsorted(ref[:, 0], est[:, 0]), 1, len(ref) - 1)
    prev = idx - 1
    use = np.where(np.abs(ref[prev, 0] - est[:, 0]) < np.abs(ref[idx, 0] - est[:, 0]), prev, idx)
    ok = np.abs(ref[use, 0] - est[:, 0]) < tol_ns
    return est[ok], ref[use[ok]]


def umeyama_sim3(src: np.ndarray, dst: np.ndarray):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    u, d, vt = np.linalg.svd(d_c.T @ s_c / len(src))
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    rot = u @ sign @ vt
    scale = (d * np.diag(sign)).sum() / s_c.var(0).sum()
    return scale, rot, mu_d - scale * rot @ mu_s


def aligned_positions(est: np.ndarray, gt: np.ndarray) -> np.ndarray:
    e, g = associate(est, gt)
    scale, rot, trans = umeyama_sim3(e[:, 1:4], g[:, 1:4])
    out = est.copy()
    out[:, 1:4] = (scale * (rot @ est[:, 1:4].T)).T + trans
    return out


def read_image_index(cam_dir: Path):
    rows = []
    with open(cam_dir / "data.csv") as handle:
        for row in csv.reader(handle):
            if not row or row[0].startswith("#"):
                continue
            rows.append((int(row[0]), cam_dir / "data" / row[1].strip()))
    rows.sort()
    return np.array([r[0] for r in rows]), [r[1] for r in rows]


def load_tracks(path: Path, wanted: set[int]):
    """Observations for the wanted timestamps: {ts: [(cam, track, x, y)]}."""
    out: dict[int, list] = defaultdict(list)
    with open(path) as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            ts = int(row[1])
            if ts in wanted:
                out[ts].append((int(row[2]), int(row[3]), float(row[4]), float(row[5])))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--traj", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=100, help="GIF frames")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--tail", type=int, default=6, help="camera frames of track tail")
    parser.add_argument("--title", default="")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--colors", type=int, default=64)
    args = parser.parse_args()

    gt = load_tum(args.gt)
    trajs = []
    for spec in args.traj:
        label, rest = spec.split("=", 1)
        path, _, score = rest.partition("@")
        trajs.append((label, aligned_positions(load_tum(Path(path)), gt), score))

    cam_ts, cam0_files = read_image_index(args.images / "mav0" / "cam0")
    cam1_ts, cam1_files = read_image_index(args.images / "mav0" / "cam1")
    t0, t1 = max(cam_ts[0], gt[0, 0]), min(cam_ts[-1], gt[-1, 0])
    sample_ts = np.linspace(t0, t1, args.frames)
    picks = np.clip(np.searchsorted(cam_ts, sample_ts), 0, len(cam_ts) - 1)
    wanted = set()
    for p in picks:
        for k in range(max(0, p - args.tail), p + 1):
            wanted.add(int(cam_ts[k]))
    tracks = load_tracks(args.tracks, wanted)

    # Fixed map extent from everything that will be drawn.
    xy = np.vstack([gt[:, 1:3]] + [t[:, 1:3] for _, t, _ in trajs])
    lo, hi = xy.min(0), xy.max(0)
    pad = 0.06 * (hi - lo).max()
    centre, half = (lo + hi) / 2, (hi - lo).max() / 2 + pad

    height = int(args.width * 0.56)
    img_w = int(height * 0.66)
    map_w = args.width - img_w
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    # Fixed palette: the camera images are greyscale, so grey levels plus the
    # exact overlay/line colours keep every curve its legend colour.
    accents = ["#000000", "#ffffff", "#cccccc", "#00dcff", "#ff3cdc"] + COLORS
    levels = max(8, args.colors - len(accents))
    entries = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in accents]
    entries += [(v, v, v) for v in np.linspace(0, 255, levels).astype(int)]
    flat = [c for rgb in entries for c in rgb]
    palette = Image.new("P", (1, 1))
    palette.putpalette(flat + flat[-3:] * (256 - len(entries)))

    frames = []
    for gi, (ts, pick) in enumerate(zip(sample_ts, picks)):
        # Left: two camera images with tracks.
        cam_h = height // 2
        left = Image.new("RGB", (img_w, height), "black")
        for cam, files, cts, row in ((0, cam0_files, cam_ts, 0), (1, cam1_files, cam1_ts, 1)):
            j = int(np.clip(np.searchsorted(cts, cam_ts[pick]), 0, len(files) - 1))
            image = Image.open(files[j]).convert("RGB")
            sx, sy = img_w / image.width, cam_h / image.height
            image = image.resize((img_w, cam_h))
            draw = ImageDraw.Draw(image)
            history = defaultdict(list)
            for k in range(max(0, pick - args.tail), pick + 1):
                for c, tid, x, y in tracks.get(int(cam_ts[k]), []):
                    if c == cam:
                        history[tid].append((k, x * sx, y * sy))
            colour = (0, 220, 255) if cam == 0 else (255, 60, 220)
            live = 0
            for tid, pts in history.items():
                if pts[-1][0] != pick:
                    continue  # track ended before this frame
                live += 1
                xy_pts = [(x, y) for _, x, y in pts]
                if len(xy_pts) > 1:
                    draw.line(xy_pts, fill=colour, width=1)
                x, y = xy_pts[-1]
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=colour)
            label = f"cam{cam}: {live} tracked points"
            box = draw.textbbox((6, 4), label, font=font)
            draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(0, 0, 0))
            draw.text((6, 4), label, fill="white", font=font)
            left.paste(image, (0, row * cam_h))

        # Right: trajectories up to this time.
        fig = plt.figure(figsize=(map_w / 100, height / 100), dpi=100)
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.88])
        ax.plot(gt[:, 1], gt[:, 2], color="#cccccc", lw=1.0, zorder=1)
        m = gt[:, 0] <= ts
        ax.plot(gt[m, 1], gt[m, 2], color="black", lw=1.6, label="ground truth (pseudo-GT)", zorder=2)
        for k, (label, traj, score) in enumerate(trajs):
            m = traj[:, 0] <= ts
            text = f"{label} — Score {score}" if score else label
            lw = 2.4 if k == 0 else 1.6
            ax.plot(traj[m, 1], traj[m, 2], color=COLORS[k % len(COLORS)], lw=lw,
                    label=text, zorder=5 - k)
            if m.any():
                ax.plot(traj[m][-1, 1], traj[m][-1, 2], "o", color=COLORS[k % len(COLORS)],
                        ms=5, zorder=6)
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=8, frameon=True)
        elapsed = (ts - t0) / 1e9
        fig.text(0.02, 0.95, f"{args.title}   t = {elapsed:5.0f} s", fontsize=9)
        fig.canvas.draw()
        right = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                                 fig.canvas.buffer_rgba()).convert("RGB")
        plt.close(fig)

        frame = Image.new("RGB", (args.width, height), "white")
        frame.paste(left, (0, 0))
        frame.paste(right.resize((map_w, height)), (img_w, 0))
        frames.append(frame.quantize(palette=palette, dither=Image.Dither.NONE))
        if gi % 25 == 0:
            print(f"frame {gi}/{len(sample_ts)}")

    hold = [frames[-1]] * int(args.fps * 2)
    frames[0].save(args.out, save_all=True, append_images=frames[1:] + hold,
                   duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
