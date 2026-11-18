#!/usr/bin/env python3
"""Turn a rectified stereo EuRoC-layout sequence and its SLAM trajectory into
the TUM RGB-D layout that `tum_rgbd_save_map` reads.

    python3 scripts/euroc_stereo_to_rgbd.py --euroc data/kitti_euroc/0027 \\
        --trajectory runs/0027_slam/trajectory_online.tum --out runs/0027_rgbd

For sequences whose cameras are already rectified pinholes (KITTI raw through
`kitti_raw_to_euroc.py`: Double Sphere with xi = alpha = 0), so that a map
with depth at every corner can be built where the SLAM's own map holds only
its few tracked landmarks:

- `rgb.txt`, `depth.txt`: every `--every`-th camera-0 image (linked, not
  copied) and its depth, from OpenCV's SGBM on the camera-0/1 pair
  (`depth/<stamp>.png`, millimetres, 16 bit; pass `--depth-scale 0.001` to
  `tum_rgbd_save_map`).
- `trajectory.tum`: the trajectory's IMU poses moved to camera 0
  (world <- IMU <- camera 0 from `calibration.json`), as `tum_rgbd_save_map`
  wants the camera's poses.
- `camera.txt`: `fx fy cx cy` of camera 0, for `tum_rgbd_save_map --fx ...`
  (distortion 0).

No `accelerometer.txt` is written: `tum_rgbd_save_map` then levels the map
by its largest horizontal plane (on KITTI, the road).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def quaternion_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_quaternion(r):
    w = np.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def pose(entry):
    t = np.eye(4)
    t[:3, :3] = quaternion_matrix(entry["qx"], entry["qy"], entry["qz"], entry["qw"])
    t[:3, 3] = [entry["px"], entry["py"], entry["pz"]]
    return t


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--euroc", type=Path, required=True, help="sequence with mav0/cam0, mav0/cam1, calibration.json")
    parser.add_argument("--trajectory", type=Path, required=True, help="IMU poses, TUM format")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument("--max-disparity", type=int, default=128)
    args = parser.parse_args()

    import cv2

    calibration = json.loads((args.euroc / "calibration.json").read_text())
    calibration = calibration.get("value0", calibration)
    cam = [c["intrinsics"] for c in calibration["intrinsics"]]
    for c in cam:
        if c.get("xi", 0) != 0 or c.get("alpha", 0) != 0:
            raise SystemExit("cameras must be pinholes (Double Sphere with xi = alpha = 0)")
    imu_cam0, imu_cam1 = (pose(t) for t in calibration["T_imu_cam"][:2])
    baseline = np.linalg.norm((np.linalg.inv(imu_cam0) @ imu_cam1)[:3, 3])
    fx, fy, cx, cy = cam[0]["fx"], cam[0]["fy"], cam[0]["cx"], cam[0]["cy"]

    out = args.out
    (out / "depth").mkdir(parents=True, exist_ok=True)
    rgb_link = out / "rgb"
    if not rgb_link.exists():
        os.symlink((args.euroc / "mav0" / "cam0" / "data").resolve(), rgb_link)
    left = sorted((args.euroc / "mav0" / "cam0" / "data").glob("*.png"))
    right_dir = args.euroc / "mav0" / "cam1" / "data"
    matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=args.max_disparity, blockSize=5,
                                    P1=8 * 25, P2=32 * 25, uniquenessRatio=10,
                                    speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1,
                                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    rgb_rows, depth_rows = [], []
    for path in left[::args.every]:
        stamp = int(path.stem)
        seconds = f"{stamp // 1_000_000_000}.{stamp % 1_000_000_000:09d}"
        depth_path = out / "depth" / path.name
        if not depth_path.exists():
            grey = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in (path, right_dir / path.name)]
            disparity = matcher.compute(grey[0], grey[1]).astype(np.float64) / 16.0
            depth = np.zeros_like(disparity)
            valid = disparity > 1.0
            depth[valid] = fx * baseline / disparity[valid]
            cv2.imwrite(str(depth_path), np.clip(depth * 1000.0, 0, 65535).astype(np.uint16))
        rgb_rows.append(f"{seconds} rgb/{path.name}")
        depth_rows.append(f"{seconds} depth/{path.name}")
    (out / "rgb.txt").write_text("# timestamp filename\n" + "\n".join(rgb_rows) + "\n")
    (out / "depth.txt").write_text("# timestamp filename\n" + "\n".join(depth_rows) + "\n")

    rows = []
    for line in args.trajectory.read_text().splitlines():
        fields = line.split()
        if not fields or line.startswith("#"):
            continue
        t, p, q = fields[0], [float(v) for v in fields[1:4]], [float(v) for v in fields[4:8]]
        world_imu = np.eye(4)
        world_imu[:3, :3] = quaternion_matrix(*q)
        world_imu[:3, 3] = p
        world_cam = world_imu @ imu_cam0
        qx, qy, qz, qw = matrix_quaternion(world_cam[:3, :3])
        x, y, z = world_cam[:3, 3]
        rows.append(f"{float(t):.9f} {x:.6f} {y:.6f} {z:.6f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}")
    (out / "trajectory.tum").write_text("\n".join(rows) + "\n")
    (out / "camera.txt").write_text(f"{fx} {fy} {cx} {cy}\n")
    print(f"{out}: {len(rgb_rows)} frames, baseline {baseline:.3f} m, fx {fx:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
