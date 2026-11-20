#!/usr/bin/env python3
"""The planar transform from a SLAM map's frame to the Gazebo world.

    python3 ros2/visloc-ros2/gz/align_map.py --trajectory runs/gz_nav/slam/trajectory.tum \
        --ground-truth runs/gz_nav/mapping/mav0/state_groundtruth_estimate0/data.csv \
        --out runs/gz_nav/map/world_from_map.json

The map is in the SLAM's own gravity-aligned frame (origin and heading where
mapping started); the goals and the scoring are in the world's. This fits a
yaw and a translation (least squares, no scale) between the SLAM's IMU
trajectory and the ground truth at the same stamps, and writes it with the
fit's residuals. Evaluation only: navigation itself never sees it.
"""

import argparse
import bisect
import json
import math


def read_tum(path):
    rows = []
    for line in open(path, encoding='utf-8'):
        if line.startswith('#') or not line.strip():
            continue
        v = line.split()
        t = float(v[0])
        rows.append((t if t > 1e12 else t * 1e9, float(v[1]), float(v[2])))
    return rows


def read_gt(path):
    rows = []
    for line in open(path, encoding='utf-8'):
        if line.startswith('#') or not line.strip():
            continue
        v = line.split(',')
        rows.append((float(v[0]), float(v[1]), float(v[2])))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--trajectory', required=True)
    parser.add_argument('--ground-truth', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    est = read_tum(args.trajectory)
    gt = read_gt(args.ground_truth)
    stamps = [g[0] for g in gt]
    pairs = []
    for t, x, y in est:
        i = bisect.bisect_left(stamps, t)
        best = min((j for j in (i - 1, i) if 0 <= j < len(gt)), key=lambda j: abs(stamps[j] - t))
        if abs(stamps[best] - t) < 30e6:
            pairs.append(((x, y), gt[best][1:]))
    n = len(pairs)
    ma = [sum(p[0][k] for p in pairs) / n for k in (0, 1)]
    mb = [sum(p[1][k] for p in pairs) / n for k in (0, 1)]
    sxx = sum((a[0] - ma[0]) * (b[0] - mb[0]) + (a[1] - ma[1]) * (b[1] - mb[1]) for a, b in pairs)
    sxy = sum((a[0] - ma[0]) * (b[1] - mb[1]) - (a[1] - ma[1]) * (b[0] - mb[0]) for a, b in pairs)
    yaw = math.atan2(sxy, sxx)
    c, s = math.cos(yaw), math.sin(yaw)
    tx, ty = mb[0] - (c * ma[0] - s * ma[1]), mb[1] - (s * ma[0] + c * ma[1])
    errors = sorted(math.hypot(c * a[0] - s * a[1] + tx - b[0], s * a[0] + c * a[1] + ty - b[1])
                    for a, b in pairs)
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    result = {'yaw': yaw, 'x': tx, 'y': ty, 'pairs': n, 'rmse_m': rmse, 'max_m': errors[-1]}
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
