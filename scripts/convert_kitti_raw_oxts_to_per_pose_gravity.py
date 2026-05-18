#!/usr/bin/env python3
"""Convert KITTI raw OXTS packets to per-keyframe `g_camera_observed`
observations consumed by `stereo_vo_external_deep_files --ba-per-pose-gravity-
prior-observations`.

For each OXTS packet, the accelerometer specific-force vector (fields 11-13,
body-frame `(ax, ay, az)` in m/s²) is converted into a gravity-direction
observation expressed in the rectified-cam0 frame using the same body→cam0
extrinsic as `convert_kitti_raw_oxts_to_odometry_poses.py`.

Specific force convention: an IMU at rest reads `+g` along its body-up axis
(i.e. opposite to gravity, because the sensor measures the reaction force
keeping it stationary). Body-frame gravity is therefore `g_body = -a_body`.
Vehicle motion acceleration is typically << 9.81 m/s², so the raw single-sample
reading is dominated by gravity; an optional `--window-half-size N` argument
averages over a boxcar window to further suppress motion-acceleration noise.

The output magnitude is rescaled to `--g-magnitude` (default 9.81) so the prior
residual sits in the same order of magnitude as a pixel reprojection residual,
independently of motion-acceleration contamination.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math


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


def matvec3(r: list[list[float]], v: list[float]) -> list[float]:
    return [sum(r[i][j] * v[j] for j in range(3)) for i in range(3)]


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


def load_r_cam_imu(calib_dir: Path) -> list[list[float]]:
    """Return the 3×3 rotation that maps a body-frame (IMU) direction vector
    into the rectified-cam0 frame: `v_cam = R_cam_imu @ v_imu`.

    The full SE(3) chain is `T_cam_imu = R_rect · T_velo_imu^{-1}^{-1} ·
    T_cam_velo · T_velo_imu = R_rect · T_cam_velo · T_velo_imu`, and we
    extract its rotation block.
    """
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
    t_cam_imu = matmul(matmul(r_rect, t_cam_velo), t_velo_imu)
    return [t_cam_imu[i][:3] for i in range(3)]


def read_oxts_acceleration_body(oxts_dir: Path, frames: int | None) -> list[list[float]]:
    """Return one body-frame accelerometer triplet (ax, ay, az) per OXTS
    packet, in lexicographic order of data file name."""
    data_dir = oxts_dir / "data"
    paths = sorted(data_dir.glob("*.txt"))
    if frames is not None:
        paths = paths[:frames]
    if not paths:
        raise SystemExit(f"{data_dir}: no OXTS data files found")
    accels: list[list[float]] = []
    for path in paths:
        rows = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            raise SystemExit(f"{path}: empty OXTS file")
        tokens = rows[0].split()
        if len(tokens) < 14:
            raise SystemExit(
                f"{path}: expected >= 14 OXTS fields (need ax,ay,az at idx 11-13), got {len(tokens)}"
            )
        accels.append([float(tokens[11]), float(tokens[12]), float(tokens[13])])
    return accels


def read_oxts_velocity_body(oxts_dir: Path, frames: int | None) -> list[list[float]]:
    """Return one body-frame velocity triplet (vf, vl, vu) per OXTS packet,
    in lexicographic order of data file name. vf/vl are body-axis horizontal
    components (parallel to earth surface), vu is world-frame vertical
    velocity; for slowly-pitching ground vehicles this is a close enough
    proxy for body-frame velocity to differentiate into motion accel."""
    data_dir = oxts_dir / "data"
    paths = sorted(data_dir.glob("*.txt"))
    if frames is not None:
        paths = paths[:frames]
    if not paths:
        raise SystemExit(f"{data_dir}: no OXTS data files found")
    velocities: list[list[float]] = []
    for path in paths:
        rows = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            raise SystemExit(f"{path}: empty OXTS file")
        tokens = rows[0].split()
        if len(tokens) < 11:
            raise SystemExit(
                f"{path}: expected >= 11 OXTS fields (need vf,vl,vu at idx 8-10), got {len(tokens)}"
            )
        velocities.append([float(tokens[8]), float(tokens[9]), float(tokens[10])])
    return velocities


def central_difference_accel(velocities: list[list[float]], dt: float) -> list[list[float]]:
    """Estimate per-sample motion accel by central-difference of the velocity
    sequence. Endpoints use forward/backward difference so the array length
    matches the input. dt is the sample period in seconds."""
    n = len(velocities)
    out: list[list[float]] = []
    for i in range(n):
        lo = max(0, i - 1)
        hi = min(n - 1, i + 1)
        span = (hi - lo) * dt
        if span <= 0.0:
            out.append([0.0, 0.0, 0.0])
            continue
        out.append(
            [
                (velocities[hi][axis] - velocities[lo][axis]) / span
                for axis in range(3)
            ]
        )
    return out


def boxcar_window(values: list[list[float]], idx: int, half_size: int) -> list[float]:
    """Compute the per-axis mean of `values[idx-half_size : idx+half_size+1]`,
    clipped at the array bounds. Returns a length-3 vector."""
    n = len(values)
    lo = max(0, idx - half_size)
    hi = min(n, idx + half_size + 1)
    cnt = hi - lo
    out = [0.0, 0.0, 0.0]
    for k in range(lo, hi):
        for axis in range(3):
            out[axis] += values[k][axis]
    return [out[axis] / cnt for axis in range(3)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oxts-dir", required=True, type=Path, help="KITTI raw oxts directory")
    parser.add_argument(
        "--calib-dir",
        required=True,
        type=Path,
        help="directory containing calib_imu_to_velo.txt, calib_velo_to_cam.txt, calib_cam_to_cam.txt",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="output observation file (consumed by --ba-per-pose-gravity-prior-observations)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        help="limit to the first N OXTS packets (matches the BA --frames count)",
    )
    parser.add_argument(
        "--window-half-size",
        type=int,
        default=0,
        help="boxcar half-window for low-pass smoothing of the body-frame "
        "accelerometer (default 0 = single-sample; suggested 5-10 for "
        "10 Hz OXTS to suppress motion-accel noise)",
    )
    parser.add_argument(
        "--g-magnitude",
        type=float,
        default=9.81,
        help="rescale every observation so |g_camera| equals this value "
        "(default 9.81 m/s²); preserves the prior's natural residual scale "
        "regardless of motion-accel contamination",
    )
    parser.add_argument(
        "--velocity-correction",
        action="store_true",
        help="subtract motion accel (central-difference of OXTS body-frame "
        "velocity vf,vl,vu at fields 8-10) from the raw accel reading "
        "before computing the gravity direction. Removes vehicle-frame "
        "linear acceleration (the dominant motion contamination on highway "
        "and accel/decel events) at the cost of a small slope-induced "
        "error in the vu component",
    )
    parser.add_argument(
        "--sample-dt",
        type=float,
        default=0.1,
        help="OXTS sample period in seconds for --velocity-correction "
        "(default 0.1 = 10 Hz)",
    )
    parser.add_argument(
        "--motion-accel-soft-gate-sigma",
        type=float,
        default=None,
        help="emit a per-observation weight in the 5th column equal to "
        "1 / (1 + (|a_motion|/SIGMA)^2). Lower SIGMA = sharper "
        "downweighting of motion-contaminated frames. Requires "
        "--velocity-correction to compute |a_motion|. The BA reader "
        "treats the 5th column as an inverse-variance multiplier on "
        "top of --ba-per-pose-gravity-prior-weight",
    )
    parser.add_argument(
        "--motion-accel-hard-gate",
        type=float,
        default=None,
        help="emit per-obs weight = 0 for frames where |a_motion| > "
        "this threshold (m/s²), else 1. Combine with "
        "--motion-accel-soft-gate-sigma to clamp the soft weight to 0 "
        "for catastrophic frames",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames is not None and args.frames <= 0:
        raise SystemExit("--frames must be positive")
    if args.window_half_size < 0:
        raise SystemExit("--window-half-size must be non-negative")
    if not (args.g_magnitude > 0 and math.isfinite(args.g_magnitude)):
        raise SystemExit("--g-magnitude must be positive and finite")

    accels_body = read_oxts_acceleration_body(args.oxts_dir, args.frames)
    motion_accel: list[list[float]] | None = None
    if args.velocity_correction:
        if args.sample_dt <= 0.0 or not math.isfinite(args.sample_dt):
            raise SystemExit("--sample-dt must be positive and finite")
        vel_body = read_oxts_velocity_body(args.oxts_dir, args.frames)
        if len(vel_body) != len(accels_body):
            raise SystemExit(
                f"velocity rows ({len(vel_body)}) != accel rows ({len(accels_body)})"
            )
        motion_accel = central_difference_accel(vel_body, args.sample_dt)
        accels_body = [
            [accels_body[i][axis] - motion_accel[i][axis] for axis in range(3)]
            for i in range(len(accels_body))
        ]

    emit_per_obs_weight = (
        args.motion_accel_soft_gate_sigma is not None
        or args.motion_accel_hard_gate is not None
    )
    if emit_per_obs_weight and motion_accel is None:
        raise SystemExit(
            "--motion-accel-soft-gate-sigma / --motion-accel-hard-gate require "
            "--velocity-correction (need motion-accel estimate)"
        )
    if args.motion_accel_soft_gate_sigma is not None and not (
        args.motion_accel_soft_gate_sigma > 0
        and math.isfinite(args.motion_accel_soft_gate_sigma)
    ):
        raise SystemExit("--motion-accel-soft-gate-sigma must be positive and finite")
    if args.motion_accel_hard_gate is not None and not (
        args.motion_accel_hard_gate >= 0
        and math.isfinite(args.motion_accel_hard_gate)
    ):
        raise SystemExit("--motion-accel-hard-gate must be non-negative and finite")

    r_cam_imu = load_r_cam_imu(args.calib_dir)

    # Header captures all knobs so the produced file is self-describing.
    header_lines = [
        "# Per-keyframe gravity-in-camera-frame observations derived from KITTI",
        "# raw OXTS accelerometer + raw body->cam0 extrinsic.",
        f"# oxts_dir            = {args.oxts_dir}",
        f"# calib_dir           = {args.calib_dir}",
        f"# frames              = {len(accels_body)}",
        f"# window_half_size    = {args.window_half_size}",
        f"# g_magnitude         = {args.g_magnitude}",
        f"# velocity_correction = {args.velocity_correction}",
        f"# sample_dt           = {args.sample_dt}",
        f"# motion_accel_soft_gate_sigma = {args.motion_accel_soft_gate_sigma}",
        f"# motion_accel_hard_gate       = {args.motion_accel_hard_gate}",
        ("# keyframe_id gx gy gz weight" if emit_per_obs_weight else "# keyframe_id gx gy gz"),
    ]

    body_count = 0
    observation_lines: list[str] = []
    for idx in range(len(accels_body)):
        a_body = boxcar_window(accels_body, idx, args.window_half_size)
        # Specific force = -g_body. Body-frame gravity points opposite the
        # accelerometer reading because the IMU's body-up direction reads
        # +9.81 m/s² when stationary.
        g_body = [-a_body[axis] for axis in range(3)]
        g_cam = matvec3(r_cam_imu, g_body)
        norm = math.sqrt(g_cam[0] ** 2 + g_cam[1] ** 2 + g_cam[2] ** 2)
        if not math.isfinite(norm) or norm <= 0.0:
            raise SystemExit(
                f"frame {idx}: derived |g_camera| = {norm}; cannot rescale "
                "(check OXTS data integrity)"
            )
        scale = args.g_magnitude / norm
        g_cam = [g_cam[axis] * scale for axis in range(3)]
        if emit_per_obs_weight:
            assert motion_accel is not None  # narrowed by the gating check above
            am = motion_accel[idx]
            am_mag = math.sqrt(am[0] ** 2 + am[1] ** 2 + am[2] ** 2)
            obs_weight = 1.0
            if args.motion_accel_soft_gate_sigma is not None:
                obs_weight = 1.0 / (
                    1.0 + (am_mag / args.motion_accel_soft_gate_sigma) ** 2
                )
            if args.motion_accel_hard_gate is not None and am_mag > args.motion_accel_hard_gate:
                obs_weight = 0.0
            observation_lines.append(
                f"{idx} {g_cam[0]:.6f} {g_cam[1]:.6f} {g_cam[2]:.6f} {obs_weight:.6f}"
            )
        else:
            observation_lines.append(
                f"{idx} {g_cam[0]:.6f} {g_cam[1]:.6f} {g_cam[2]:.6f}"
            )
        body_count += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(header_lines + observation_lines) + "\n"
    )
    print(
        f"wrote {body_count} per-pose gravity observations to {args.out} "
        f"(window_half_size={args.window_half_size}, |g|={args.g_magnitude})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
