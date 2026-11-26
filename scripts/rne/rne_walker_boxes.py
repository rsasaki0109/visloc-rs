#!/usr/bin/env python3
"""True camera-0 boxes of RNE example 113's walkers, for the VIO's moving-object filter.

From an export with walkers (`RNE_WALKERS`): `people.csv` (each walker's
position at each camera stamp), `gt.tum` (the robot base = IMU in RNE's
y-up world, ns) and `visloc_calib.json` (cam0's `T_imu_cam` and pinhole
intrinsics). Each walker is a vertical cylinder (`--radius`, `--height`,
times `--scale`), projected into cam0; its box is the bounding box of the
projected points in front of the camera, clipped to the image. Scene
occlusion is ignored: a walker behind a wall still gets a box, as a
detector would not, so this is the filter's best case.

Writes `timestamp_ns,object_id,class_id,score,x_min,y_min,x_max,y_max`
(`visloc_basalt::dynamic_objects::read_object_boxes_csv`).

  python3 scripts/rne/rne_walker_boxes.py --run target/drjohnson_people --out boxes.csv
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def quat_to_rot(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=0.3)
    parser.add_argument("--height", type=float, default=1.8)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    calib = json.loads((args.run / "visloc_calib.json").read_text())["value0"]
    t = calib["T_imu_cam"][0]
    imu_cam = np.eye(4)
    imu_cam[:3, :3] = quat_to_rot(t["qx"], t["qy"], t["qz"], t["qw"])
    imu_cam[:3, 3] = [t["px"], t["py"], t["pz"]]
    k = calib["intrinsics"][0]["intrinsics"]
    width, height = calib["resolution"][0]

    base = {}
    for line in (args.run / "gt.tum").read_text().splitlines():
        v = line.split()
        if len(v) == 8:
            pose = np.eye(4)
            pose[:3, :3] = quat_to_rot(*map(float, v[4:8]))
            pose[:3, 3] = list(map(float, v[1:4]))
            base[int(v[0])] = pose

    r, h = args.radius * args.scale, args.height * args.scale
    ring = [(r * math.cos(a), r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
    rows = []
    for row in csv.DictReader((args.run / "people.csv").open()):
        stamp = int(row["timestamp_ns"])
        world_base = base.get(stamp)
        if world_base is None:
            continue
        cam_world = np.linalg.inv(world_base @ imu_cam)
        x, z = float(row["x"]), float(row["z"])
        points = np.array([[x + dx, y, z + dz, 1.0] for dx, dz in ring for y in (0.0, h / 2, h)])
        cam = (cam_world @ points.T).T[:, :3]
        cam = cam[cam[:, 2] > 0.05]
        if len(cam) < 3:
            continue
        u = k["fx"] * cam[:, 0] / cam[:, 2] + k["cx"]
        v = k["fy"] * cam[:, 1] / cam[:, 2] + k["cy"]
        x0, x1 = max(0.0, u.min()), min(width - 1.0, u.max())
        y0, y1 = max(0.0, v.min()), min(height - 1.0, v.max())
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        rows.append(f"{stamp},{row['walker']},0,1.0,{x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}")
    args.out.write_text("timestamp_ns,object_id,class_id,score,x_min,y_min,x_max,y_max\n" + "\n".join(rows) + "\n")
    print(f"{len(rows)} boxes -> {args.out}")


if __name__ == "__main__":
    main()
