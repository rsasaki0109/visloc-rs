#!/usr/bin/env python3
"""Synthesize a legged robot's kinematic odometry from a ground-truth trajectory.

    python3 scripts/synthesize_leg_odometry.py --ground-truth GT.csv --out leg.tum [--seed 0]

No public dataset pairs a legged robot's odometry with a map-localization
benchmark, so this makes one from ground truth with the error a leg
odometry has (kinematics + foot contact, attitude from the IMU):

- a scale error in the distance travelled (`--scale`, foot slip and leg
  length), applied to every step, horizontal and vertical alike;
- a heading drift proportional to distance (`--yaw-deg-per-m`) plus a yaw
  random walk (`--yaw-walk-deg`), as foot slip in turns gives;
- a vertical drift per metre travelled (`--z-per-m`), as uneven contact
  heights give;
- per-step position noise (`--step-noise-m`) and a gait bounce: a vertical
  oscillation (`--bounce-m`) and a roll/pitch sway (`--sway-deg`) at the
  step frequency (`--gait-hz`). Roll and pitch otherwise follow the ground
  truth, as an IMU attitude filter keeps them.

The output is the body's pose in the odometry's own frame (starting at the
identity yaw and origin), TUM (seconds), at `--rate-hz`. The body is the
ground truth's (EuRoC: the IMU), so no `--imu-from-body` is needed.
"""
import argparse
import math

import numpy as np


def quat_to_matrix(x, y, z, w):
    q = np.array([x, y, z, w]) / np.linalg.norm([x, y, z, w])
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(r):
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = math.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2
    y = math.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2
    z = math.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2
    x = math.copysign(x, r[2, 1] - r[1, 2])
    y = math.copysign(y, r[0, 2] - r[2, 0])
    z = math.copysign(z, r[1, 0] - r[0, 1])
    return x, y, z, w


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def read_ground_truth(path):
    """EuRoC state_groundtruth_estimate0/data.csv or TUM: (t_ns, p, R)."""
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = [f for f in line.replace(",", " ").split()]
            if path.endswith(".csv"):
                t = int(fields[0])
                p = np.array([float(v) for v in fields[1:4]])
                w, x, y, z = (float(v) for v in fields[4:8])
            else:
                t = int(round(float(fields[0]) * 1e9))
                p = np.array([float(v) for v in fields[1:4]])
                x, y, z, w = (float(v) for v in fields[4:8])
            rows.append((t, p, quat_to_matrix(x, y, z, w)))
    rows.sort(key=lambda row: row[0])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--scale", type=float, default=1.03)
    parser.add_argument("--yaw-deg-per-m", type=float, default=0.5)
    parser.add_argument("--yaw-walk-deg", type=float, default=0.05, help="per sqrt(m)")
    parser.add_argument("--z-per-m", type=float, default=0.01)
    parser.add_argument("--step-noise-m", type=float, default=0.002)
    parser.add_argument("--bounce-m", type=float, default=0.01)
    parser.add_argument("--sway-deg", type=float, default=0.5)
    parser.add_argument("--gait-hz", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    truth = read_ground_truth(args.ground_truth)
    step_ns = int(1e9 / args.rate_hz)
    samples = [truth[0]]
    for row in truth[1:]:
        if row[0] - samples[-1][0] >= step_ns:
            samples.append(row)

    position = np.zeros(3)
    heading = 0.0  # odometry yaw error accumulated, on top of the true yaw
    _, p0, r0 = samples[0]
    yaw0 = math.atan2(r0[1, 0], r0[0, 0])
    lines = ["# timestamp tx ty tz qx qy qz qw  (synthetic legged odometry)"]
    previous_p, previous_r = p0, r0
    walk_scale = math.radians(args.yaw_walk_deg)
    for t, p, r in samples:
        delta_world = p - previous_p
        distance = float(np.linalg.norm(delta_world))
        # The true step in the previous yaw frame, then into the odometry's
        # drifted heading.
        previous_yaw = math.atan2(previous_r[1, 0], previous_r[0, 0])
        step_local = rz(previous_yaw).T @ delta_world
        heading += math.radians(args.yaw_deg_per_m) * distance
        heading += walk_scale * math.sqrt(distance) * rng.standard_normal()
        odometry_yaw = previous_yaw - yaw0 + heading
        step = args.scale * step_local
        step[2] += args.z_per_m * distance
        step += args.step_noise_m * rng.standard_normal(3) * min(1.0, distance / 0.01)
        position = position + rz(odometry_yaw) @ step
        previous_p, previous_r = p, r

        phase = 2 * math.pi * args.gait_hz * (t - samples[0][0]) * 1e-9
        bounce = args.bounce_m * math.sin(phase)
        yaw = math.atan2(r[1, 0], r[0, 0])
        # Roll/pitch of the truth, re-headed into the odometry's yaw, plus sway.
        tilt = rz(yaw).T @ r
        sway = rx(math.radians(args.sway_deg) * math.sin(phase)) @ ry(
            math.radians(args.sway_deg) * math.cos(phase)
        )
        rotation = rz(yaw - yaw0 + heading) @ sway @ tilt
        x, y, z, w = matrix_to_quat(rotation)
        out = position + np.array([0.0, 0.0, bounce])
        lines.append(
            f"{t // 1_000_000_000}.{t % 1_000_000_000:09d} {out[0]:.6f} {out[1]:.6f} {out[2]:.6f} "
            f"{x:.9f} {y:.9f} {z:.9f} {w:.9f}"
        )
    with open(args.out, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"{len(samples)} poses -> {args.out}")


if __name__ == "__main__":
    main()
