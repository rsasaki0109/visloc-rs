#!/usr/bin/env python3
"""Turn a KITTI raw drive into an EuRoC-style sequence for basalt_euroc_vio_demo.

    python3 scripts/kitti_raw_to_euroc.py --raw data/kitti_raw/2011_09_26 \
        --drive 2011_09_26_drive_0091 --out data/kitti_euroc/0091

Input is what `fetch_kitti_raw_drive.py` writes. Output:
- `mav0/cam0`, `mav0/cam1`: the rectified grey cameras image_00 and image_01
  (10 Hz), named by their timestamps.
- `mav0/imu0/data.csv`: the 100 Hz OXTS of the `_extract` zip as the IMU:
  the body rates (wx, wy, wz) and specific force (ax, ay, az, which include
  gravity) in the OXTS frame (x forward, y left, z up).
- `calibration.json`: Basalt calibration. Pinhole cameras (double sphere with
  xi = alpha = 0) from `P_rect_00/01`; `T_imu_cam` from the IMU-to-velodyne,
  velodyne-to-camera and rectifying rotations, and the rectified baseline.
- `gt_imu.tum`: the OXTS pose of every image frame (IMU frame, local
  Mercator east-north-up, metres, from the first frame's position; z stays
  up), TUM (seconds, with all nine decimals).
- `frames.csv`: `frame,timestamp_ns`, the sync frame index of every image
  (the tracklets count frames this way).

KITTI timestamps are local clock times; they are read as UTC, which only
shifts every stream by the same amount. OXTS samples that do not advance in
time are dropped. `--start-frame` drops the frames before it (and the IMU
until half a second before it), for a VIO that must start at rest;
`frames.csv` keeps the original indices.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np

EARTH_RADIUS_M = 6378137.0


def stamp_ns(text: str) -> int:
    day, clock = text.strip().split(" ")
    whole, fraction = clock.split(".")
    seconds = dt.datetime.strptime(f"{day} {whole}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return int(seconds.timestamp()) * 1_000_000_000 + int(fraction.ljust(9, "0")[:9])


def read_stamps(path: Path) -> list[int]:
    return [stamp_ns(line) for line in path.read_text().splitlines() if line.strip()]


def read_calib(path: Path) -> dict[str, np.ndarray]:
    values = {}
    for line in path.read_text().splitlines():
        key, _, rest = line.partition(":")
        try:
            values[key.strip()] = np.array([float(v) for v in rest.split()])
        except ValueError:
            pass
    return values


def rigid(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = r.reshape(3, 3)
    m[:3, 3] = t
    return m


def quaternion(r: np.ndarray) -> tuple[float, float, float, float]:
    """(x, y, z, w) of a rotation matrix."""
    w = math.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2.0
    x = math.copysign(math.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) / 2.0, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2])) / 2.0, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2])) / 2.0, r[1, 0] - r[0, 1])
    return x, y, z, w


def transform_json(m: np.ndarray) -> dict[str, float]:
    x, y, z, w = quaternion(m[:3, :3])
    return {"px": m[0, 3], "py": m[1, 3], "pz": m[2, 3], "qx": x, "qy": y, "qz": z, "qw": w}


def oxts_pose(packet: np.ndarray, scale: float) -> np.ndarray:
    """IMU-to-world of an OXTS packet: Mercator east/north, altitude up."""
    lat, lon, alt, roll, pitch, yaw = packet[:6]
    mx = scale * lon * math.pi * EARTH_RADIUS_M / 180.0
    my = scale * EARTH_RADIUS_M * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rigid(rz @ ry @ rx, np.array([mx, my, alt]))


def read_oxts(directory: Path) -> tuple[list[int], list[np.ndarray]]:
    stamps = read_stamps(directory / "timestamps.txt")
    packets = [np.array([float(v) for v in (directory / "data" / f"{i:010d}.txt").read_text().split()])
               for i in range(len(stamps))]
    return stamps, packets


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw", required=True, type=Path, help="the date directory, e.g. .../2011_09_26")
    parser.add_argument("--drive", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--accel-noise", type=float, default=0.02, help="accel_noise_std (m/s^2/sqrt(Hz))")
    parser.add_argument("--gyro-noise", type=float, default=0.0005, help="gyro_noise_std (rad/s/sqrt(Hz))")
    parser.add_argument("--start-frame", type=int, default=0, help="first image frame to keep")
    args = parser.parse_args()
    sync = args.raw / f"{args.drive}_sync"
    mav = args.out / "mav0"

    # Cameras. Both are named by image_00's times: the pair is triggered
    # together, and their logged times differ by a few microseconds.
    frames: list[tuple[int, int]] = []
    stamps = read_stamps(sync / "image_00" / "timestamps.txt")
    for cam in (0, 1):
        data = mav / f"cam{cam}" / "data"
        data.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, ns in list(enumerate(stamps))[args.start_frame:]:
            link_or_copy(sync / f"image_0{cam}" / "data" / f"{index:010d}.png", data / f"{ns}.png")
            rows.append(f"{ns},{ns}.png")
            if cam == 0:
                frames.append((index, ns))
        (mav / f"cam{cam}" / "data.csv").write_text("#timestamp [ns],filename\n" + "\n".join(rows) + "\n")
    (args.out / "frames.csv").write_text("frame,timestamp_ns\n" + "".join(f"{i},{ns}\n" for i, ns in frames))

    # IMU: the 100 Hz OXTS.
    stamps, packets = read_oxts(args.raw / f"{args.drive}_extract" / "oxts")
    (mav / "imu0").mkdir(parents=True, exist_ok=True)
    rows, last, dropped = [], -1, 0
    first = frames[0][1] - 500_000_000
    for ns, p in sorted(zip(stamps, packets), key=lambda item: item[0]):
        if ns < first:
            continue
        if ns <= last:
            dropped += 1
            continue
        last = ns
        ax, ay, az, wx, wy, wz = p[11], p[12], p[13], p[17], p[18], p[19]
        rows.append(f"{ns},{wx:.9f},{wy:.9f},{wz:.9f},{ax:.9f},{ay:.9f},{az:.9f}")
    header = "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
    (mav / "imu0" / "data.csv").write_text(header + "\n".join(rows) + "\n")
    rate = (len(rows) - 1) / ((last - int(rows[0].split(",")[0])) * 1e-9)

    # Calibration.
    cam_to_cam = read_calib(args.raw / "calib_cam_to_cam.txt")
    velo_to_cam = read_calib(args.raw / "calib_velo_to_cam.txt")
    imu_to_velo = read_calib(args.raw / "calib_imu_to_velo.txt")
    t_velo_imu = rigid(imu_to_velo["R"], imu_to_velo["T"])
    t_cam_velo = rigid(velo_to_cam["R"], velo_to_cam["T"])
    t_rect_cam = rigid(cam_to_cam["R_rect_00"], np.zeros(3))
    t_cam0_imu = t_rect_cam @ t_cam_velo @ t_velo_imu
    t_imu_cam0 = np.linalg.inv(t_cam0_imu)
    intrinsics, transforms, resolutions = [], [], []
    for cam in (0, 1):
        p = cam_to_cam[f"P_rect_0{cam}"].reshape(3, 4)
        size = cam_to_cam[f"S_rect_0{cam}"]
        t_cam_cam0 = rigid(np.eye(3), np.array([p[0, 3] / p[0, 0], 0.0, 0.0]))
        transforms.append(transform_json(t_imu_cam0 @ np.linalg.inv(t_cam_cam0)))
        intrinsics.append({"camera_type": "ds", "intrinsics": {"fx": p[0, 0], "fy": p[1, 1], "cx": p[0, 2],
                                                                "cy": p[1, 2], "xi": 0.0, "alpha": 0.0}})
        resolutions.append([int(round(size[0])), int(round(size[1]))])
    identity = {"px": 0.0, "py": 0.0, "pz": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
    calibration = {"value0": {
        "T_imu_cam": transforms, "intrinsics": intrinsics, "resolution": resolutions,
        "calib_accel_bias": [0.0] * 9, "calib_gyro_bias": [0.0] * 12, "imu_update_rate": round(rate, 1),
        "accel_noise_std": [args.accel_noise] * 3, "gyro_noise_std": [args.gyro_noise] * 3,
        "accel_bias_std": [0.001] * 3, "gyro_bias_std": [0.0001] * 3,
        "T_mocap_world": identity, "T_imu_marker": identity,
        "mocap_time_offset_ns": 0, "mocap_to_imu_offset_ns": 0, "cam_time_offset_ns": 0,
    }}
    (args.out / "calibration.json").write_text(json.dumps(calibration, indent=4))

    # Ground truth: the 10 Hz OXTS of the image frames.
    gt_stamps, gt_packets = read_oxts(sync / "oxts")
    scale = math.cos(gt_packets[0][0] * math.pi / 180.0)
    origin = oxts_pose(gt_packets[args.start_frame], scale)[:3, 3]
    lines = []
    for (_, ns), packet in zip(frames, gt_packets[args.start_frame:]):
        pose = oxts_pose(packet, scale)
        pose[:3, 3] -= origin
        x, y, z, w = quaternion(pose[:3, :3])
        lines.append(f"{ns // 1_000_000_000}.{ns % 1_000_000_000:09d} {pose[0, 3]:.6f} {pose[1, 3]:.6f} {pose[2, 3]:.6f} {x:.9f} {y:.9f} {z:.9f} {w:.9f}")
    (args.out / "gt_imu.tum").write_text("\n".join(lines) + "\n")
    print(f"{len(frames)} frames, IMU {len(rows)} samples at {rate:.1f} Hz ({dropped} dropped), "
          f"baseline {-cam_to_cam['P_rect_01'][3] / cam_to_cam['P_rect_01'][0]:.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
