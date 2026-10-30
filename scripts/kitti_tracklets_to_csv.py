#!/usr/bin/env python3
"""KITTI raw tracklets as per-frame ground-truth objects, in the world of
`kitti_raw_to_euroc.py` and in camera 0's image.

    python3 scripts/kitti_tracklets_to_csv.py --raw data/kitti_raw/2011_09_26 \
        --drive 2011_09_26_drive_0091 --euroc data/kitti_euroc/0091 --out gt_objects.csv

One row per object and frame:
`frame,timestamp_ns,gt_id,type,h,w,l,x,y,z,yaw,distance,near_distance,u_min,v_min,u_max,v_max,occlusion,truncation`.
- `x,y,z`: the box's bottom centre in the world of `gt_imu.tum` (z up), and
  `yaw` its heading there.
- `distance`: horizontal distance from camera 0 to the bottom centre;
  `near_distance` to the nearest point of the footprint (what ground contact
  sees: the near edge).
- `u_*`, `v_*`: the box's 8 corners projected into rectified camera 0, empty
  if a corner is behind the camera.
- `occlusion` and `truncation` as KITTI labels them (0 = none).
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kitti_raw_to_euroc import read_calib, rigid  # noqa: E402


def quat_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--drive", required=True)
    parser.add_argument("--euroc", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    frames = {}
    for line in (args.euroc / "frames.csv").read_text().splitlines()[1:]:
        index, ns = line.split(",")
        frames[int(index)] = int(ns)
    poses = {}
    for line in (args.euroc / "gt_imu.tum").read_text().splitlines():
        v = line.split()
        seconds, _, fraction = v[0].partition(".")
        poses[int(seconds) * 1_000_000_000 + int(fraction.ljust(9, "0")[:9])] = rigid(quat_matrix(*map(float, v[4:8])), np.array([float(x) for x in v[1:4]]))

    cam_to_cam = read_calib(args.raw / "calib_cam_to_cam.txt")
    velo_to_cam = read_calib(args.raw / "calib_velo_to_cam.txt")
    imu_to_velo = read_calib(args.raw / "calib_imu_to_velo.txt")
    t_velo_imu = rigid(imu_to_velo["R"], imu_to_velo["T"])
    t_imu_velo = np.linalg.inv(t_velo_imu)
    t_cam_velo = rigid(cam_to_cam["R_rect_00"], np.zeros(3)) @ rigid(velo_to_cam["R"], velo_to_cam["T"])
    p = cam_to_cam["P_rect_00"].reshape(3, 4)
    fx, fy, cx, cy = p[0, 0], p[1, 1], p[0, 2], p[1, 2]

    root = ET.parse(args.raw / f"{args.drive}_sync" / "tracklet_labels.xml").getroot()
    rows = []
    gt_id = 0
    for item in root.iter("item"):
        kind = item.find("objectType")
        if kind is None:
            continue
        gt_id += 1
        h, w, l = (float(item.find(k).text) for k in ("h", "w", "l"))
        first = int(item.find("first_frame").text)
        footprint = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
        corners = np.array([[x, y, z] for z in (0.0, h) for x, y in footprint])
        for k, pose in enumerate(item.find("poses").iter("item")):
            frame = first + k
            if frame not in frames or frames[frame] not in poses:
                continue
            t = np.array([float(pose.find(n).text) for n in ("tx", "ty", "tz")])
            yaw = float(pose.find("rz").text)
            r = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
            box_velo = corners @ r.T + t
            world = poses[frames[frame]] @ t_imu_velo
            camera = t_cam_velo
            bottom = world @ np.append(t, 1.0)
            heading = world[:3, :3] @ r[:, 0]
            cam_center = (world @ np.linalg.inv(camera))[:3, 3]
            foot_world = (world @ np.c_[box_velo[:4], np.ones(4)].T).T[:, :2]
            distance = math.hypot(*(bottom[:2] - cam_center[:2]))
            near = nearest_distance(foot_world, cam_center[:2])
            in_cam = (camera @ np.c_[box_velo, np.ones(8)].T)[:3].T
            if (in_cam[:, 2] > 0.5).all():
                u = fx * in_cam[:, 0] / in_cam[:, 2] + cx
                v = fy * in_cam[:, 1] / in_cam[:, 2] + cy
                box = f"{u.min():.1f},{v.min():.1f},{u.max():.1f},{v.max():.1f}"
            else:
                box = ",,,"
            occlusion = int(pose.find("occlusion").text)
            truncation = int(pose.find("truncation").text)
            rows.append(f"{frame},{frames[frame]},{gt_id},{kind.text.replace(' ', '_')},{h:.3f},{w:.3f},{l:.3f},"
                        f"{bottom[0]:.3f},{bottom[1]:.3f},{bottom[2]:.3f},{math.atan2(heading[1], heading[0]):.4f},"
                        f"{distance:.3f},{near:.3f},{box},{occlusion},{truncation}")
    header = ("frame,timestamp_ns,gt_id,type,h,w,l,x,y,z,yaw,distance,near_distance,"
              "u_min,v_min,u_max,v_max,occlusion,truncation\n")
    args.out.write_text(header + "\n".join(rows) + "\n")
    print(f"{gt_id} objects, {len(rows)} rows")
    return 0


def nearest_distance(polygon: np.ndarray, point: np.ndarray) -> float:
    """Distance from `point` to a convex polygon (0 inside)."""
    best, sides = math.inf, set()
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge, rel = b - a, point - a
        sides.add(edge[0] * rel[1] - edge[1] * rel[0] > 0)
        s = np.clip(rel @ edge / (edge @ edge), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(rel - s * edge)))
    return 0.0 if len(sides) == 1 else best


if __name__ == "__main__":
    raise SystemExit(main())
