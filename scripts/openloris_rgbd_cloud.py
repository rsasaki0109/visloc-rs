#!/usr/bin/env python3
"""Colour point cloud of an OpenLORIS-Scene session, placed by its SLAM map.

For `render_map_merge_gif.py --cloud`: the D435i's colour images and the
depth aligned to them, back-projected from the poses of a map's keyframes.

    python scripts/openloris_rgbd_cloud.py --map runs/o11_map \\
        --rgbd data/rgbd/office1-1 --out runs/o11_cloud.npz

`--map` is a map saved by `basalt_euroc_online_slam_demo --save-map` from
the sequence converted by `openloris_to_euroc.py`; its camera 0 is the T265's
fisheye1. `--rgbd` is the sequence's extracted `color/`, `aligned_depth/`,
`color.txt`, `aligned_depth.txt` and `trans_matrix.yaml`, `sensors.yaml`.
For each keyframe (every `--every`-th camera-0 image) the colour image and
depth taken within `--max-dt-ms` of it are back-projected (every
`--stride`-th pixel, depths within `--min-depth`..`--max-depth`) into the
map's world: world <- fisheye1 (the keyframe's pose) <- base_link <- colour
camera (`trans_matrix.yaml`). Points are thinned to one per `--voxel` cube,
keeping the first keyframe's colour.

Writes points (N x 3, the map's world), colours (N x 3, 0..1) and first_seen
(N, the time in seconds from the map's first keyframe at which the voxel was
first seen). Dataset images stay outside the repository and remain subject
to the dataset's CC BY-ND 4.0 terms.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image


def quaternion_matrix(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def yaml_matrices(path: Path) -> dict[tuple[str, str], np.ndarray]:
    """(parent, child) -> 4x4 from an OpenCV `trans_matrix.yaml`."""
    text = path.read_text()
    out = {}
    pattern = re.compile(r"parent_frame:\s*(\S+)\s*child_frame:\s*(\S+).*?data:\s*\[([^\]]*)\]", re.S)
    for parent, child, data in pattern.findall(text):
        out[(parent, child)] = np.array([float(v) for v in data.replace("\n", " ").split(",")]).reshape(4, 4)
    return out


def colour_intrinsics(path: Path) -> tuple[float, float, float, float]:
    """fx, fy, cx, cy of d400_color_optical_frame (`sensors.yaml` lists fx, cx, fy, cy)."""
    text = path.read_text()
    block = text[text.index("d400_color_optical_frame:"):]
    data = re.search(r"intrinsics:.*?data:\s*\[([^\]]*)\]", block, re.S).group(1)
    fx, cx, fy, cy = (float(v) for v in data.replace("\n", " ").split(","))
    return fx, fy, cx, cy


def read_stamped(path: Path) -> tuple[np.ndarray, list[str]]:
    stamps, files = [], []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 2 and not line.startswith("#"):
            stamps.append(float(fields[0]))
            files.append(fields[1])
    return np.asarray(stamps), files


def nearest(stamps: np.ndarray, t: float) -> int:
    i = int(np.clip(np.searchsorted(stamps, t), 1, len(stamps) - 1))
    return i - 1 if abs(stamps[i - 1] - t) < abs(stamps[i] - t) else i


def keyframe_poses(map_dir: Path):
    """(time in s, world <- camera 0 as 4x4) for each camera-0 keyframe, in time order."""
    cams = {}
    for line in (map_dir / "images.csv").read_text().splitlines()[1:]:
        f = line.split(",")
        if len(f) >= 4 and f[3]:
            cams[int(f[0])] = (int(f[2]), int(f[3]) * 1e-9)
    lines = [l for l in (map_dir / "sparse" / "images.txt").read_text().splitlines()
             if l and not l.startswith("#")]
    out = []
    for line in lines[::2]:
        f = line.split()
        image_id = int(f[0])
        if cams.get(image_id, (1,))[0] != 0:
            continue
        qw, qx, qy, qz, tx, ty, tz = (float(v) for v in f[1:8])
        r = quaternion_matrix(qw, qx, qy, qz)  # camera from world
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = r.T, -r.T @ np.array([tx, ty, tz])
        out.append((cams[image_id][1], pose))
    out.sort(key=lambda p: p[0])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--rgbd", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--min-depth", type=float, default=0.3)
    parser.add_argument("--max-depth", type=float, default=4.0)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--voxel", type=float, default=0.02)
    parser.add_argument("--max-dt-ms", type=float, default=20.0)
    args = parser.parse_args()

    frames = yaml_matrices(args.rgbd / "trans_matrix.yaml")
    base_f1 = frames[("base_link", "t265_fisheye1_optical_frame")]
    base_colour = frames[("base_link", "d400_color_optical_frame")]
    f1_colour = np.linalg.inv(base_f1) @ base_colour
    fx, fy, cx, cy = colour_intrinsics(args.rgbd / "sensors.yaml")
    colour_t, colour_files = read_stamped(args.rgbd / "color.txt")
    depth_t, depth_files = read_stamped(args.rgbd / "aligned_depth.txt")

    poses = keyframe_poses(args.map)
    t0 = poses[0][0]
    points, colours, seen = [], [], []
    used = 0
    for time_s, world_f1 in poses[::args.every]:
        ci, di = nearest(colour_t, time_s), nearest(depth_t, time_s)
        if (abs(colour_t[ci] - time_s) > args.max_dt_ms * 1e-3
                or abs(depth_t[di] - time_s) > args.max_dt_ms * 1e-3):
            continue
        rgb = np.asarray(Image.open(args.rgbd / colour_files[ci]).convert("RGB"))
        depth = np.asarray(Image.open(args.rgbd / depth_files[di])).astype(np.float64) * args.depth_scale
        v, u = np.mgrid[0:depth.shape[0]:args.stride, 0:depth.shape[1]:args.stride]
        d = depth[v, u]
        ok = (d > args.min_depth) & (d < args.max_depth)
        u, v, d = u[ok], v[ok], d[ok]
        local = np.stack([(u - cx) / fx * d, (v - cy) / fy * d, d, np.ones_like(d)])
        world = (world_f1 @ f1_colour @ local)[:3].T
        points.append(world)
        colours.append(rgb[v, u] / 255.0)
        seen.append(np.full(len(d), time_s - t0))
        used += 1
    points = np.concatenate(points)
    colours = np.concatenate(colours)
    seen = np.concatenate(seen)
    # One point per voxel: the first keyframe's (the arrays are in time order).
    keys = np.floor(points / args.voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    first.sort()
    np.savez_compressed(args.out, points=points[first].astype(np.float32),
                        colours=colours[first].astype(np.float32),
                        first_seen=seen[first].astype(np.float32))
    print(f"{args.out}: {used} keyframes, {len(first):,} points "
          f"(from {len(points):,}, {args.voxel * 100:.0f} cm voxels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
