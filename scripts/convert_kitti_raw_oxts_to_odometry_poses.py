#!/usr/bin/env python3
"""Convert KITTI raw OXTS packets to KITTI odometry-format camera poses.

The output is one 3x4 camera-to-world matrix per line, in the same text format
used by the KITTI odometry benchmark.  The first selected camera frame is used
as the world origin, so the resulting file can be compared directly with an
odometry sequence pose file when the raw frame range is aligned.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math


EARTH_RADIUS_M = 6378137.0


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows = len(a)
    cols = len(b[0])
    inner = len(b)
    return [[sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)] for i in range(rows)]


def transpose3(r: list[list[float]]) -> list[list[float]]:
    return [[r[j][i] for j in range(3)] for i in range(3)]


def transform_from_rt(r: list[list[float]], t: list[float]) -> list[list[float]]:
    return [
        [r[0][0], r[0][1], r[0][2], t[0]],
        [r[1][0], r[1][1], r[1][2], t[1]],
        [r[2][0], r[2][1], r[2][2], t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def invert_transform(t: list[list[float]]) -> list[list[float]]:
    r = [row[:3] for row in t[:3]]
    rt = transpose3(r)
    p = [t[0][3], t[1][3], t[2][3]]
    inv_p = [-sum(rt[i][j] * p[j] for j in range(3)) for i in range(3)]
    return transform_from_rt(rt, inv_p)


def rotx(theta: float) -> list[list[float]]:
    c = math.cos(theta)
    s = math.sin(theta)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def roty(theta: float) -> list[list[float]]:
    c = math.cos(theta)
    s = math.sin(theta)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def rotz(theta: float) -> list[list[float]]:
    c = math.cos(theta)
    s = math.sin(theta)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def parse_calib_values(path: Path) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        tokens = rest.split()
        if not tokens:
            continue
        try:
            values[key] = [float(token) for token in tokens]
        except ValueError:
            continue
    return values


def require(values: dict[str, list[float]], key: str, count: int, path: Path) -> list[float]:
    row = values.get(key)
    if row is None:
        raise SystemExit(f"{path}: missing {key}")
    if len(row) != count:
        raise SystemExit(f"{path}: {key} expected {count} values, got {len(row)}")
    return row


def matrix3(values: list[float]) -> list[list[float]]:
    return [values[0:3], values[3:6], values[6:9]]


def load_cam0_rect_from_imu(calib_dir: Path) -> list[list[float]]:
    imu_to_velo_path = calib_dir / "calib_imu_to_velo.txt"
    velo_to_cam_path = calib_dir / "calib_velo_to_cam.txt"
    cam_to_cam_path = calib_dir / "calib_cam_to_cam.txt"

    imu_to_velo = parse_calib_values(imu_to_velo_path)
    velo_to_cam = parse_calib_values(velo_to_cam_path)
    cam_to_cam = parse_calib_values(cam_to_cam_path)

    t_velo_imu = transform_from_rt(
        matrix3(require(imu_to_velo, "R", 9, imu_to_velo_path)),
        require(imu_to_velo, "T", 3, imu_to_velo_path),
    )
    t_cam_velo = transform_from_rt(
        matrix3(require(velo_to_cam, "R", 9, velo_to_cam_path)),
        require(velo_to_cam, "T", 3, velo_to_cam_path),
    )
    r_rect = transform_from_rt(
        matrix3(require(cam_to_cam, "R_rect_00", 9, cam_to_cam_path)),
        [0.0, 0.0, 0.0],
    )
    return matmul(matmul(r_rect, t_cam_velo), t_velo_imu)


def oxts_packet_to_world_from_imu(packet: list[float], scale: float) -> list[list[float]]:
    lat, lon, alt, roll, pitch, yaw = packet[:6]
    mx = scale * lon * math.pi * EARTH_RADIUS_M / 180.0
    my = scale * EARTH_RADIUS_M * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    r = matmul(matmul(rotz(yaw), roty(pitch)), rotx(roll))
    return transform_from_rt(r, [mx, my, alt])


def read_oxts_packets(oxts_dir: Path, frames: int | None) -> list[list[float]]:
    data_dir = oxts_dir / "data"
    paths = sorted(data_dir.glob("*.txt"))
    if frames is not None:
        paths = paths[:frames]
    if not paths:
        raise SystemExit(f"{data_dir}: no OXTS data files found")
    packets = []
    for path in paths:
        rows = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            raise SystemExit(f"{path}: empty OXTS file")
        packet = [float(token) for token in rows[0].split()]
        if len(packet) < 6:
            raise SystemExit(f"{path}: expected at least 6 OXTS fields")
        packets.append(packet)
    return packets


def format_kitti_pose(t: list[list[float]]) -> str:
    return " ".join(f"{t[row][col]:.12e}" for row in range(3) for col in range(4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oxts-dir", required=True, type=Path, help="KITTI raw oxts directory")
    parser.add_argument(
        "--calib-dir",
        required=True,
        type=Path,
        help="directory containing calib_imu_to_velo.txt, calib_velo_to_cam.txt, calib_cam_to_cam.txt",
    )
    parser.add_argument("--out", required=True, type=Path, help="output KITTI pose text file")
    parser.add_argument("--frames", type=int, help="limit to the first N OXTS packets")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames is not None and args.frames <= 0:
        raise SystemExit("--frames must be positive")

    packets = read_oxts_packets(args.oxts_dir, args.frames)
    scale = math.cos(packets[0][0] * math.pi / 180.0)
    t_cam_imu = load_cam0_rect_from_imu(args.calib_dir)
    t_imu_cam = invert_transform(t_cam_imu)

    world_from_camera = [
        matmul(oxts_packet_to_world_from_imu(packet, scale), t_imu_cam) for packet in packets
    ]
    origin_inv = invert_transform(world_from_camera[0])
    odometry_poses = [matmul(origin_inv, pose) for pose in world_from_camera]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(format_kitti_pose(pose) for pose in odometry_poses) + "\n")
    print(f"wrote {len(odometry_poses)} poses to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
