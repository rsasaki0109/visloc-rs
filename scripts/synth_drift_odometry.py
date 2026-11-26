#!/usr/bin/env python3
"""Make a drifting odometry from a ground-truth trajectory (TUM, seconds).

For datasets without an IMU or wheels (TUM RGB-D Freiburg 3), this stands in
for the odometry that `basalt_euroc_localize_demo --odometry` corrects into
the map frame. The ground truth's world must be gravity-aligned (z up), as
the odometry frame must be.

Between consecutive ground-truth poses, the relative motion is kept in
roll and pitch (an IMU keeps those) and corrupted in what odometry gets
wrong:

- translation scaled by `1 + --scale-error` (2%),
- a yaw error of `--yaw-deg-per-m` (3 deg) per metre travelled, plus
- white noise on each step (`--noise-m`, `--noise-deg`).

The result starts at the ground truth's first pose, in a frame of its own
that drifts away from the map's. Deterministic for a given `--seed`.

  python3 scripts/synth_drift_odometry.py --gt groundtruth.txt --out odom.tum --seed 1
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_conj(q):
    return (-q[0], -q[1], -q[2], q[3])


def quat_rotate(q, v):
    x, y, z = quat_mul(quat_mul(q, (v[0], v[1], v[2], 0.0)), quat_conj(q))[:3]
    return (x, y, z)


def yaw_quat(angle):
    return (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))


def normalize(q):
    n = math.sqrt(sum(c * c for c in q))
    return tuple(c / n for c in q)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scale-error", type=float, default=0.02)
    parser.add_argument("--yaw-deg-per-m", type=float, default=3.0)
    parser.add_argument("--noise-m", type=float, default=0.0005)
    parser.add_argument("--noise-deg", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    rows = []
    for line in args.gt.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        v = [float(x) for x in line.split()]
        rows.append((v[0], tuple(v[1:4]), normalize(tuple(v[4:8]))))
    rows.sort()

    # The odometry world from the true one: a yaw about z plus a position,
    # updated step by step.
    drift_yaw = 0.0
    position = rows[0][1]
    out = ["# timestamp tx ty tz qx qy qz qw"]
    previous = rows[0]
    out.append("{:.6f} {:.6f} {:.6f} {:.6f} {:.7f} {:.7f} {:.7f} {:.7f}".format(previous[0], *position, *previous[2]))
    for row in rows[1:]:
        stamp, p, q = row
        delta = tuple(p[k] - previous[1][k] for k in range(3))
        step = math.sqrt(sum(d * d for d in delta))
        drift_yaw += math.radians(args.yaw_deg_per_m) * step + math.radians(args.noise_deg) * rng.gauss(0, 1) * (step > 0)
        rotated = quat_rotate(yaw_quat(drift_yaw), delta)
        position = tuple(
            position[k] + (1 + args.scale_error) * rotated[k] + args.noise_m * rng.gauss(0, 1) * (step > 0)
            for k in range(3)
        )
        orientation = normalize(quat_mul(yaw_quat(drift_yaw), q))
        out.append("{:.6f} {:.6f} {:.6f} {:.6f} {:.7f} {:.7f} {:.7f} {:.7f}".format(stamp, *position, *orientation))
        previous = row
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n")
    print(f"{len(rows)} poses -> {args.out}")


if __name__ == "__main__":
    main()
