#!/usr/bin/env python3
"""Calibrate where a wheel odometry's reference point sits relative to the IMU, from a VIO run.

`basalt_euroc_localize_demo --odometry FILE --imu-from-body T` (and
`visloc_map_localize_node` with `extrinsics tf`) turn the odometry's body
pose into the IMU pose with `T_imu_body`. If that extrinsic puts the body a
few centimetres away from the point the wheel odometry actually reports
(the midpoint of the drive axle, the robot's turning centre), every turn
moves the computed IMU pose by up to twice that offset, which looks like
odometry drift. On OpenLORIS-Scene the dataset's base_link sits about 7 cm
from the wheel odometry's point: corrected, the wheel odometry follows the
ground truth to about 1 cm instead of 4-9 cm.

This fits that offset from a run with both sensors: the VIO's IMU poses
(gravity-aligned, `trajectory_vio.tum` of `basalt_euroc_localize_demo
--vio`) and the wheel odometry (planar TUM poses of the body, in seconds).
In the plane,

    p_vio(t) = s * R(a) * (p_wheel(t) + R(yaw_wheel(t)) * r) + t0

where p_vio is the body point the current `T_imu_body` places from the VIO
pose, r the planar offset (in the body frame) from the wheel odometry's
point to that body point, and s, a, t0 a similarity between the two
trajectories (the VIO's scale may be a few percent off). r is best observed
while the robot turns in place. The corrected extrinsic puts the body at
the wheel odometry's point: `T_imu_body' = T_imu_body * (I, -r)`.

It also levels the body frame. Planar wheel odometry keeps its body level,
so the extrinsic's rotation must be right about the level too: with the
camera high above the wheels, a tilt of a degree or two places the IMU
centimetres off horizontally from where it is. On a level floor, the
VIO's gravity says which way is up in the body frame (its mean over the
run); the corrected rotation turns the body by the smallest rotation that
makes that its z axis. `--keep-tilt` leaves the rotation as it is (a robot
whose floor is not level). On OpenLORIS-Scene the dataset's base_link is
tilted by 1.9 deg against the VIO's gravity, with the T265 0.9 m above it.

    python3 scripts/calibrate_wheel_offset.py --vio runs/o13_vio/trajectory_vio.tum \\
        --odometry data/office1-3/odom.tum \\
        --imu-from-body=0.0306,-0.8937,-0.2452,-0.5030,-0.4927,-0.5074,0.4967

(Write `--imu-from-body=VALUE`: a value starting with a minus sign would
otherwise read as an option.)
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np


def quat_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def read_tum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stamps (ns), positions (N x 3) and rotations (N x 3 x 3) of a TUM file in seconds."""
    stamps, positions, rotations = [], [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        # Decimal: exact for "123.456789012" and for "1.23456789012e9" alike.
        stamps.append(int(Decimal(fields[0]) * 1_000_000_000))
        values = [float(v) for v in fields[1:8]]
        positions.append(values[:3])
        rotations.append(quat_matrix(*values[3:7]))
    return np.array(stamps, dtype=np.int64), np.array(positions), np.array(rotations)


def parse_pose(text: str) -> tuple[np.ndarray, np.ndarray]:
    """`tx,ty,tz,qx,qy,qz,qw` -> (rotation, translation); the quaternion is
    normalized, as one written to four decimals is not quite unit length."""
    values = [float(v) for v in text.split(",")]
    if len(values) != 7:
        raise argparse.ArgumentTypeError("expected tx,ty,tz,qx,qy,qz,qw")
    q = np.array(values[3:7])
    if not np.linalg.norm(q) > 0:
        raise argparse.ArgumentTypeError("the quaternion must not be zero")
    return quat_matrix(*(q / np.linalg.norm(q))), np.array(values[:3])


def rotation_quat(rotation: np.ndarray) -> np.ndarray:
    """(x, y, z, w) with w >= 0."""
    m = rotation
    w = np.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def fit_offset(
    reference: np.ndarray, wheel: np.ndarray, wheel_yaw: np.ndarray, iterations: int = 30
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Planar offset r, scale s, rotation a and RMS residual of
    reference ~ s R(a) (wheel + R(wheel_yaw) r) + t0 (all N x 2)."""
    c, s_ = np.cos(wheel_yaw), np.sin(wheel_yaw)
    r = np.zeros(2)
    scale = 1.0
    for _ in range(iterations):
        q = wheel + np.stack([c * r[0] - s_ * r[1], s_ * r[0] + c * r[1]], 1)
        mq, mg = q.mean(0), reference.mean(0)
        h = (q - mq).T @ (reference - mg)
        a = np.arctan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        scale = float(np.sum((reference - mg) * ((q - mq) @ rot.T)) / np.sum((q - mq) ** 2))
        t0 = mg - scale * rot @ mq
        # reference - s R(a) wheel - t0 = s R(a) R(yaw) r, linear in r.
        blocks = scale * np.einsum("ij,njk->nik", rot, np.stack([np.stack([c, -s_], 1), np.stack([s_, c], 1)], 1))
        rhs = reference - scale * wheel @ rot.T - t0
        r = np.linalg.lstsq(blocks.reshape(-1, 2), rhs.reshape(-1), rcond=None)[0]
    q = wheel + np.stack([c * r[0] - s_ * r[1], s_ * r[0] + c * r[1]], 1)
    residual = reference - (scale * q @ rot.T + t0)
    return r, scale, float(a), np.array([np.sqrt((residual**2).sum(1).mean())])


def level_rotation(up: np.ndarray) -> np.ndarray:
    """The smallest rotation taking the z axis to the unit vector `up`."""
    z = np.array([0.0, 0.0, 1.0])
    axis, cosine = np.cross(z, up), float(z @ up)
    sine = np.linalg.norm(axis)
    if sine < 1e-12:
        return np.eye(3)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + k + k @ k * ((1 - cosine) / sine**2)


def calibrate(
    vio_path: Path,
    odometry_path: Path,
    imu_from_body: tuple[np.ndarray, np.ndarray],
    max_gap_ns: int = 20_000_000,
    level: bool = True,
) -> dict:
    vio_t, vio_p, vio_r = read_tum(vio_path)
    od_t, od_p, od_r = read_tum(odometry_path)
    r_ib, t_ib = imu_from_body
    # The body point the current extrinsic places from each VIO IMU pose.
    body = vio_p + np.einsum("nij,j->ni", vio_r, t_ib)
    nearest = np.clip(np.searchsorted(vio_t, od_t), 1, len(vio_t) - 1)
    nearest = np.where(np.abs(vio_t[nearest - 1] - od_t) < np.abs(vio_t[nearest] - od_t), nearest - 1, nearest)
    ok = np.abs(vio_t[nearest] - od_t) <= max_gap_ns
    if ok.sum() < 10:
        raise SystemExit(f"only {int(ok.sum())} odometry poses within {max_gap_ns / 1e6:.0f} ms of a VIO pose")
    wheel_yaw = np.arctan2(od_r[ok, 1, 0], od_r[ok, 0, 0])
    r, scale, _, rms = fit_offset(body[nearest[ok], :2], od_p[ok, :2], wheel_yaw)
    zero_r_rms = _rms_without_offset(body[nearest[ok], :2], od_p[ok, :2])
    # Gravity's up in the body frame, averaged over the run.
    up = np.einsum("nji,j->ni", np.einsum("nij,jk->nik", vio_r, r_ib), np.array([0.0, 0.0, 1.0])).mean(0)
    up /= np.linalg.norm(up)
    tilt = float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
    corrected_r = r_ib @ level_rotation(up) if level else r_ib
    offset = np.array([r[0], r[1], 0.0])
    corrected_t = t_ib - corrected_r @ offset
    turn = float(np.degrees(np.abs(np.diff(np.unwrap(wheel_yaw))).sum()))
    return dict(
        pairs=int(ok.sum()),
        offset_m=offset[:2],
        scale=scale,
        rms_m=float(rms[0]),
        rms_without_offset_m=zero_r_rms,
        turned_deg=turn,
        tilt_deg=tilt,
        imu_from_body=(corrected_r, corrected_t),
    )


def _rms_without_offset(reference: np.ndarray, wheel: np.ndarray) -> float:
    mq, mg = wheel.mean(0), reference.mean(0)
    h = (wheel - mq).T @ (reference - mg)
    a = np.arctan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    scale = float(np.sum((reference - mg) * ((wheel - mq) @ rot.T)) / np.sum((wheel - mq) ** 2))
    residual = reference - (scale * (wheel - mq) @ rot.T + mg)
    return float(np.sqrt((residual**2).sum(1).mean()))


def format_pose(rotation: np.ndarray, translation: np.ndarray) -> str:
    q = rotation_quat(rotation)
    return ",".join(f"{v:.4f}" for v in [*translation, *q])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vio", type=Path, required=True, help="VIO IMU poses (TUM, seconds), gravity-aligned")
    parser.add_argument("--odometry", type=Path, required=True, help="wheel odometry body poses (TUM, seconds)")
    parser.add_argument("--imu-from-body", type=parse_pose, required=True, help="current T_imu_body: tx,ty,tz,qx,qy,qz,qw")
    parser.add_argument("--keep-tilt", action="store_true", help="do not level the body frame (floor not level)")
    args = parser.parse_args(argv)
    result = calibrate(args.vio, args.odometry, args.imu_from_body, level=not args.keep_tilt)
    print(f"pairs: {result['pairs']}, turned {result['turned_deg']:.0f} deg in total")
    print(f"offset of the current body point from the wheel odometry's: "
          f"({result['offset_m'][0] * 100:.1f}, {result['offset_m'][1] * 100:.1f}) cm in the body frame")
    print(f"VIO scale vs wheel odometry: {result['scale']:.4f}")
    print(f"residual: {result['rms_m'] * 100:.2f} cm RMS (without the offset: {result['rms_without_offset_m'] * 100:.2f} cm)")
    print(f"tilt of the body frame against the VIO's gravity: {result['tilt_deg']:.2f} deg"
          + (" (kept)" if args.keep_tilt else " (leveled)"))
    # With "=": the value may start with a minus sign.
    print(f"--imu-from-body={format_pose(*result['imu_from_body'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
