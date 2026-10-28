#!/usr/bin/env python3
"""Render a map-based localization GIF: the map, with the camera image inset.

The saved map's landmarks in 3D, the localized camera positions so far
(orange), and the current camera (red) with rays to the landmarks its image
matched (green). Inset: the query camera image with those matched keypoints
(PnP inliers, green).

With `--trajectory`, the map-frame trajectory of a `--vio` run (the VIO
anchored to the map), the orange path and the red camera are that
trajectory, which has a pose for every frame; rays and keypoints are drawn
for the images that also matched the map, and the inset says how long the
VIO has carried the pose since the last match on the others.

Inputs are files produced elsewhere in this repo:

  --map DIR            map from `basalt_euroc_online_slam_demo --save-map`
  --images DIR         EuRoC/ASL query sequence root (mav0/cam0)
  --localization FILE  `basalt_euroc_localize_demo` localization.tum (IMU
                       poses in the map frame, seconds)
  --inliers FILE       `basalt_euroc_localize_demo --dump-inliers` CSV
  --trajectory FILE    optional: trajectory_map.tum of a `--vio` run

Example:

  basalt_euroc_localize_demo --map runs/mh01_map --euroc-dir MH_02_easy \\
      --out-dir runs/mh02_on_mh01 --dump-inliers runs/mh02_inliers.csv
  python scripts/render_map_localization_gif.py --map runs/mh01_map \\
      --images MH_02_easy --localization runs/mh02_on_mh01/localization.tum \\
      --inliers runs/mh02_inliers.csv --out docs/assets/hero_map_localization.gif

  # The VIO anchored to the map: a pose for every frame.
  basalt_euroc_localize_demo --map runs/mh01_map --euroc-dir MH_02_easy \\
      --out-dir runs/mh02_vio --vio --calibration CALIB --config CONFIG \\
      --projection-radius-px 20 --dump-inliers runs/mh02_vio_inliers.csv
  python scripts/render_map_localization_gif.py --map runs/mh01_map \\
      --images MH_02_easy --localization runs/mh02_vio/localization.tum \\
      --trajectory runs/mh02_vio/trajectory_map.tum \\
      --inliers runs/mh02_vio_inliers.csv --out hero.gif
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GREEN, ORANGE, RED, MAP_BLUE = "#2ecc40", "#ff851b", "#ff2e2e", "#7fa7c9"


def quaternion_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def load_landmarks(map_dir: Path) -> dict[int, np.ndarray]:
    landmarks = {}
    for line in (map_dir / "sparse" / "points3D.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split()
        landmarks[int(fields[0])] = np.array([float(v) for v in fields[1:4]])
    return landmarks


def camera_from_imu(map_dir: Path):
    """Rotation and translation of T_imu_cam for camera 0 (`map.json`)."""
    camera = json.loads((map_dir / "map.json").read_text())["cameras"][0]["t_imu_cam"]
    w, x, y, z = camera["quaternion_wxyz"]
    return quaternion_matrix(x, y, z, w), np.array(camera["translation"])


def load_localization(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    poses = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        t, p, q = fields[0], np.array([float(v) for v in fields[1:4]]), [float(v) for v in fields[4:8]]
        poses[int(round(float(t) * 1e9))] = (p, quaternion_matrix(*q))
    return poses


def load_inliers(path: Path) -> dict[int, list[tuple[float, float, int]]]:
    inliers = defaultdict(list)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            inliers[int(row["timestamp_ns"])].append(
                (float(row["x"]), float(row["y"]), int(row["landmark_id"])))
    return inliers


def read_image_index(cam_dir: Path):
    stamps, files = [], []
    with (cam_dir / "data.csv").open() as handle:
        for row in csv.reader(handle):
            if row and not row[0].startswith("#"):
                stamps.append(int(row[0]))
                files.append(cam_dir / "data" / row[1].strip())
    return np.array(stamps), files


def nearest(keys: np.ndarray, t: int, tolerance_ns: int = 2_000_000):
    i = int(np.clip(np.searchsorted(keys, t), 1, len(keys) - 1))
    j = i - 1 if abs(keys[i - 1] - t) < abs(keys[i] - t) else i
    return int(keys[j]) if abs(keys[j] - t) <= tolerance_ns else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--localization", type=Path, required=True)
    parser.add_argument("--inliers", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path,
                        help="trajectory_map.tum of a --vio run: a pose for every frame")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0.0, help="seconds into the sequence")
    parser.add_argument("--duration", type=float, default=60.0, help="seconds of the sequence")
    parser.add_argument("--frames", type=int, default=150, help="GIF frames")
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--title", default="")
    parser.add_argument("--subtitle", default="", help="e.g. the run's localized share and error")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--inset", type=int, default=300, help="camera inset width (px)")
    parser.add_argument("--elev", type=float, default=32.0)
    parser.add_argument("--azim", type=float, default=-60.0)
    parser.add_argument("--zoom", type=float, default=1.0, help="> 1 crops the map view")
    parser.add_argument("--map-points", type=int, default=9000, help="landmarks drawn")
    parser.add_argument("--rays", type=int, default=40, help="most rays drawn per frame")
    parser.add_argument("--colors", type=int, default=64)
    args = parser.parse_args()

    landmarks = load_landmarks(args.map)
    r_ic, t_ic = camera_from_imu(args.map)
    poses = load_localization(args.localization)
    pose_keys = np.array(sorted(poses))
    # The poses drawn: the trajectory's when given, else the localizations'.
    track = load_localization(args.trajectory) if args.trajectory else poses
    track_keys = np.array(sorted(track))
    inliers = load_inliers(args.inliers)
    cam_ts, cam_files = read_image_index(args.images / "mav0" / "cam0")

    rng = np.random.default_rng(0)
    ids = np.array(sorted(landmarks))
    points = np.array([landmarks[i] for i in ids])
    # Drop the far tail so the room fills the view.
    centre = np.median(points, axis=0)
    keep = np.linalg.norm(points - centre, axis=1) < np.percentile(
        np.linalg.norm(points - centre, axis=1), 97)
    points = points[keep]
    shown = points[rng.choice(len(points), min(args.map_points, len(points)), replace=False)]
    lo, hi = np.percentile(points, 1, axis=0), np.percentile(points, 99, axis=0)

    t0 = cam_ts[0] + int(args.start * 1e9)
    sample_ts = np.linspace(t0, t0 + args.duration * 1e9, args.frames).astype(np.int64)
    picks = np.clip(np.searchsorted(cam_ts, sample_ts), 0, len(cam_ts) - 1)

    height = int(args.width * 0.5625)
    inset_w = args.inset
    inset_h = int(inset_w * 480 / 752)
    half = (hi - lo) / 2 / args.zoom
    mid = (hi + lo) / 2
    lo, hi = mid - half, mid + half
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    # Fixed palette: grey levels for the greyscale image and the background,
    # plus the exact overlay colours.
    accents = ["#000000", "#ffffff", GREEN, ORANGE, RED, MAP_BLUE, "#2b2b2b", "#555555"]
    blues = [f"#{int(r):02x}{int(g):02x}{int(b):02x}" for r, g, b in
             np.linspace([60, 90, 120], [180, 210, 235], 12)]
    entries = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in accents + blues]
    levels = max(8, args.colors - len(entries))
    entries += [(v, v, v) for v in np.linspace(0, 255, levels).astype(int)]
    flat = [c for rgb in entries for c in rgb]
    palette = Image.new("P", (1, 1))
    palette.putpalette(flat + flat[-3:] * (256 - len(entries)))

    # Camera centres of every drawn pose, for the trail.
    centres = np.array([track[k][0] + track[k][1] @ t_ic for k in track_keys])
    localized = 0
    frames = []
    for gi, pick in enumerate(picks):
        stamp = int(cam_ts[pick])
        # The localization nearest this image, if it was localized, and the
        # drawn pose of this image.
        key = nearest(pose_keys, stamp)
        matches = inliers.get(key, []) if key is not None else []
        track_key = nearest(track_keys, stamp)
        camera_centre = None
        if track_key is not None:
            p_imu, r_imu = track[track_key]
            camera_centre = p_imu + r_imu @ t_ic
            localized += 1
        earlier = pose_keys[pose_keys <= stamp]
        since_fix = (stamp - earlier[-1]) / 1e9 if len(earlier) else None

        # The map, the trail, the camera and its rays.
        fig = plt.figure(figsize=(args.width / 100, height / 100), dpi=100,
                         facecolor="#2b2b2b")
        ax = fig.add_axes([-0.05, -0.12, 1.1, 1.24], projection="3d", facecolor="#2b2b2b")
        ax.scatter(shown[:, 0], shown[:, 1], shown[:, 2], s=0.8, c=shown[:, 2],
                   cmap="Blues", vmin=lo[2] - 2.0, vmax=hi[2], depthshade=False,
                   linewidths=0)
        path = centres[track_keys <= stamp]
        if len(path) > 1:
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color=ORANGE, lw=1.6)
        if camera_centre is not None:
            hits = [landmarks[i] for _, _, i in matches if i in landmarks]
            for target in hits[:: max(1, len(hits) // args.rays)]:
                ax.plot(*zip(camera_centre, target), color=GREEN, lw=0.4)
            ax.scatter(*camera_centre, s=50, color=RED, depthshade=False, zorder=10)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_axis_off()
        elapsed = (stamp - cam_ts[0]) / 1e9
        fig.text(0.40, 0.93, args.title, fontsize=10, color="white")
        fig.text(0.40, 0.885, args.subtitle, fontsize=8.5, color="#dddddd")
        fig.text(0.40, 0.845, f"t = {elapsed:3.0f} s", fontsize=8, color="#999999")
        fig.text(0.02, 0.03, "map landmarks", fontsize=8, color=MAP_BLUE)
        fig.text(0.16, 0.03, "map-frame trajectory" if args.trajectory
                 else "localized trajectory", fontsize=8, color=ORANGE)
        fig.text(0.33, 0.03, "camera", fontsize=8, color=RED)
        fig.text(0.41, 0.03, "rays to the landmarks its image matched", fontsize=8,
                 color=GREEN)
        fig.canvas.draw()
        frame = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                                 fig.canvas.buffer_rgba()).convert("RGB")
        plt.close(fig)

        # Inset: the image with its matched keypoints.
        image = Image.open(cam_files[pick]).convert("RGB")
        sx, sy = inset_w / image.width, inset_h / image.height
        image = image.resize((inset_w, inset_h))
        draw = ImageDraw.Draw(image)
        for x, y, _ in matches:
            x, y = x * sx, y * sy
            draw.ellipse((x - 2.5, y - 2.5, x + 2.5, y + 2.5), outline=GREEN, width=2)
        if matches:
            label, colour = f"{len(matches)} keypoints matched to the map", "white"
        elif camera_centre is not None and args.trajectory:
            label = ("no map match: VIO holds the pose" if since_fix is None
                     else f"no map match: VIO holds the pose ({since_fix:.1f} s)")
            colour = ORANGE
        else:
            label, colour = "not localized", RED
        box = draw.textbbox((6, 4), label, font=font)
        draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(0, 0, 0))
        draw.text((6, 4), label, fill=colour, font=font)
        frame.paste((255, 255, 255), (8, 8, 12 + inset_w, 12 + inset_h))
        frame.paste(image, (10, 10))
        frames.append(frame.quantize(palette=palette, dither=Image.Dither.NONE))
        if gi % 25 == 0:
            print(f"frame {gi}/{len(picks)}")

    hold = [frames[-1]] * int(args.fps * 1.5)
    frames[0].save(args.out, save_all=True, append_images=frames[1:] + hold,
                   duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB, {len(frames)} frames,"
          f" {localized} localized)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
