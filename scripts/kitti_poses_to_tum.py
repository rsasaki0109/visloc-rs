#!/usr/bin/env python3
"""Convert frame-indexed KITTI 3x4 poses + a EuRoC-style timestamps.txt into a
TUM trajectory (timestamp tx ty tz qx qy qz qw) for evo_ape association against
the timestamped EuRoC ground truth.

  kitti_to_tum.py <vo_poses.txt> <timestamps.txt> <out.tum>

timestamps.txt rows: "<frame_idx> <timestamp_ns>" (rectify_euroc_stereo.py output).
KITTI poses are camera-to-world T_w_c (row-major 3x4), so position is the last
column and orientation is the 3x3 rotation block.
"""
import sys
import numpy as np


def R_to_q(R):
    # returns (qx, qy, qz, qw)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return qx, qy, qz, qw


poses_path, ts_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

ts = []
for ln in open(ts_path):
    p = ln.split()
    if len(p) >= 2:
        ts.append(float(p[1]) / 1e9)  # ns -> s

poses = []
for ln in open(poses_path):
    v = list(map(float, ln.split()))
    if len(v) == 12:
        poses.append(v)

n = min(len(ts), len(poses))
with open(out_path, "w") as f:
    for i in range(n):
        v = poses[i]
        R = np.array([[v[0], v[1], v[2]], [v[4], v[5], v[6]], [v[8], v[9], v[10]]])
        tx, ty, tz = v[3], v[7], v[11]
        qx, qy, qz, qw = R_to_q(R)
        f.write(f"{ts[i]:.9f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
print(f"wrote {out_path}: {n} poses")
