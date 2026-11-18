#!/usr/bin/env python3
"""Colour point cloud of an RGB-D session, placed by its map's keyframes.

For `render_map_merge_gif.py --cloud`: the session's colour images and the
depth registered to them, back-projected from the poses of a map's keyframes.

    # OpenLORIS-Scene: a map of the T265 sequence (openloris_to_euroc.py)
    python scripts/rgbd_map_cloud.py --map runs/o11_map \\
        --openloris data/rgbd/office1-1 --out runs/o11_cloud.npz
    # TUM RGB-D: a map from tum_rgbd_save_map
    python scripts/rgbd_map_cloud.py --map runs/fr1_room_map \\
        --tum rgbd_dataset_freiburg1_room --out runs/fr1_room_cloud.npz

- `--openloris DIR`: the sequence's extracted `color/`, `aligned_depth/`,
  `color.txt`, `aligned_depth.txt`, `trans_matrix.yaml` and `sensors.yaml`.
  The map's camera 0 is the T265's fisheye1, so a point goes world <-
  fisheye1 (the keyframe's pose) <- base_link <- colour camera
  (`trans_matrix.yaml`). Depth is in millimetres.
- `--kitti DIR --kitti-frames FILE`: a KITTI raw drive's `_sync` directory
  with the colour cameras' rectified image_02 and image_03 of the keyframes
  (`fetch_kitti_raw_drive.py --colour-frames`), and the `frames.csv` of
  `kitti_raw_to_euroc.py`. The map's camera 0 is image_00; depth comes from
  the colour pair by OpenCV's SGBM, and image_02 gives the colours.
- `--tum DIR`: the sequence (`rgb.txt`, `depth.txt`, `rgb/`, `depth/`). The
  map's camera 0 is the RGB camera itself, with `--tum-camera`'s calibration
  (Freiburg 1's by default; undistorted with OpenCV). Depth is in 1/5000 m.

For each keyframe (every `--every`-th camera-0 image) the colour image and
depth taken within `--max-dt-ms` of it are back-projected (every
`--stride`-th pixel, depths within `--min-depth`..`--max-depth`) into the
map's world. Points are thinned to one per `--voxel` cube, keeping the first
keyframe's colour.

Writes points (N x 3, the map's world), colours (N x 3, 0..1) and first_seen
(N, the time in seconds from the map's first keyframe at which the voxel was
first seen). Dataset images stay outside the repository and remain subject
to their dataset's terms (OpenLORIS-Scene: CC BY-ND 4.0; TUM RGB-D: CC BY
4.0).
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


def stereo_depth(left: np.ndarray, right: np.ndarray, fx: float, baseline: float) -> np.ndarray:
    """Metric depth of the left image from a rectified pair (OpenCV SGBM),
    0 where the disparity is invalid or fails the left-right check."""
    import cv2

    grey = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in (left, right)]
    matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=5,
                                    P1=8 * 25, P2=32 * 25, uniquenessRatio=10,
                                    speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1,
                                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disparity = matcher.compute(grey[0], grey[1]).astype(np.float64) / 16.0
    depth = np.zeros_like(disparity)
    valid = disparity > 1.0
    depth[valid] = fx * baseline / disparity[valid]
    return depth


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--openloris", type=Path, help="extracted OpenLORIS-Scene sequence")
    source.add_argument("--tum", type=Path, help="TUM RGB-D sequence")
    source.add_argument("--kitti", type=Path,
                        help="KITTI raw drive's _sync directory with image_02/ and image_03/ "
                             "(fetch_kitti_raw_drive.py --colour-frames)")
    parser.add_argument("--kitti-frames", type=Path,
                        help="with --kitti: the frames.csv of kitti_raw_to_euroc.py, which maps "
                             "the map's timestamps to sync frame indices")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--min-depth", type=float, default=0.3)
    parser.add_argument("--max-depth", type=float, default=4.0)
    parser.add_argument("--depth-scale", type=float,
                        help="metres per depth unit (default 0.001 OpenLORIS, 0.0002 TUM)")
    parser.add_argument("--voxel", type=float, default=0.02)
    parser.add_argument("--max-dt-ms", type=float, default=20.0)
    parser.add_argument("--drop-overexposed", type=float, metavar="LEVEL",
                        help="drop pixels whose red, green and blue are all above LEVEL (0..1): "
                             "blown-out walls and sky that carry no colour")
    parser.add_argument("--tum-camera", nargs=9, type=float, metavar="V",
                        default=[517.3, 516.5, 318.6, 255.3, 0.2624, -0.9531, -0.0054, 0.0026, 1.1633],
                        help="TUM RGB camera: fx fy cx cy k1 k2 p1 p2 k3 (default Freiburg 1; "
                             "Freiburg 2: 520.9 521.0 325.1 249.7 0.2312 -0.7849 -0.0033 -0.0001 0.9172)")
    args = parser.parse_args()

    if args.openloris is not None:
        root = args.openloris
        frames = yaml_matrices(root / "trans_matrix.yaml")
        base_f1 = frames[("base_link", "t265_fisheye1_optical_frame")]
        base_colour = frames[("base_link", "d400_color_optical_frame")]
        cam0_colour = np.linalg.inv(base_f1) @ base_colour
        fx, fy, cx, cy = colour_intrinsics(root / "sensors.yaml")
        distortion = None
        colour_t, colour_files = read_stamped(root / "color.txt")
        depth_t, depth_files = read_stamped(root / "aligned_depth.txt")
        depth_scale = args.depth_scale or 0.001
    elif args.kitti is not None:
        root = args.kitti
        calib = {}
        for line in (root.parent / "calib_cam_to_cam.txt").read_text().splitlines():
            key, _, value = line.partition(":")
            try:
                calib[key] = np.array([float(v) for v in value.split()])
            except ValueError:
                pass
        p2 = calib["P_rect_02"].reshape(3, 4)
        p3 = calib["P_rect_03"].reshape(3, 4)
        fx, fy, cx, cy = p2[0, 0], p2[1, 1], p2[0, 2], p2[1, 2]
        distortion = None
        # Rectified cameras share their rotation; camera 2 sits P[0,3]/fx
        # along x from camera 0 (X2 = X0 + t), camera 3 likewise.
        cam0_colour = np.eye(4)
        cam0_colour[0, 3] = -p2[0, 3] / fx
        stereo_baseline = (p2[0, 3] - p3[0, 3]) / fx
        frame_of = {}
        for line in args.kitti_frames.read_text().splitlines()[1:]:
            frame, stamp = line.split(",")
            frame_of[int(stamp)] = int(frame)
        stamps_ns = sorted(frame_of)
        colour_t = depth_t = np.array(stamps_ns) * 1e-9
        colour_files = depth_files = [f"{frame_of[stamp]:010d}.png" for stamp in stamps_ns]
        depth_scale = 1.0
    else:
        root = args.tum
        cam0_colour = np.eye(4)
        # The RGB camera (TUM RGB-D calibration page).
        fx, fy, cx, cy = args.tum_camera[:4]
        distortion = np.array(args.tum_camera[4:])
        colour_t, colour_files = read_stamped(root / "rgb.txt")
        depth_t, depth_files = read_stamped(root / "depth.txt")
        depth_scale = args.depth_scale or 1 / 5000

    poses = keyframe_poses(args.map)
    t0 = poses[0][0]
    points, colours, seen = [], [], []
    used = 0
    for time_s, world_cam0 in poses[::args.every]:
        ci, di = nearest(colour_t, time_s), nearest(depth_t, time_s)
        if (abs(colour_t[ci] - time_s) > args.max_dt_ms * 1e-3
                or abs(depth_t[di] - time_s) > args.max_dt_ms * 1e-3):
            continue
        if args.kitti is not None:
            left_path = root / "image_02" / "data" / colour_files[ci]
            right_path = root / "image_03" / "data" / colour_files[ci]
            if not (left_path.exists() and right_path.exists()):
                continue
            rgb = np.asarray(Image.open(left_path).convert("RGB"))
            depth = stereo_depth(rgb, np.asarray(Image.open(right_path).convert("RGB")),
                                 fx, stereo_baseline)
        else:
            rgb = np.asarray(Image.open(root / colour_files[ci]).convert("RGB"))
            depth = np.asarray(Image.open(root / depth_files[di])).astype(np.float64) * depth_scale
        v, u = np.mgrid[0:depth.shape[0]:args.stride, 0:depth.shape[1]:args.stride]
        d = depth[v, u]
        ok = (d > args.min_depth) & (d < args.max_depth)
        u, v, d = u[ok], v[ok], d[ok]
        if args.drop_overexposed is not None:
            keep = ~np.all(rgb[v, u] > args.drop_overexposed * 255, axis=1)
            u, v, d = u[keep], v[keep], d[keep]
        if distortion is None:
            x, y = (u - cx) / fx, (v - cy) / fy
        else:
            import cv2
            pixels = np.stack([u, v], axis=1).astype(np.float64).reshape(-1, 1, 2)
            camera = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            x, y = cv2.undistortPoints(pixels, camera, distortion).reshape(-1, 2).T
        local = np.stack([x * d, y * d, d, np.ones_like(d)])
        world = (world_cam0 @ cam0_colour @ local)[:3].T
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
