#!/usr/bin/env python3
"""Convert a COLMAP `images.txt` (world-to-camera poses) into a TUM trajectory
(timestamp tx ty tz qx qy qz qw, camera-to-world) for evo_ape association
against timestamped EuRoC ground truth.

  colmap_images_to_tum.py <images.txt> <timestamps.txt> <out.tum>

COLMAP images.txt pose lines are:
  IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
encoding T_cam_world (rotates a world point into the camera). The camera centre
in the world is C = -R^T t and the camera-to-world rotation is R^T.

The frame index is the first integer found in NAME (so both `000123.png` and
`frame_000123.png` work); it indexes into timestamps.txt rows
`<frame_idx> <timestamp_ns>`.
"""
import os
import re
import sys

import numpy as np


def quat_wxyz_to_R(qw, qx, qy, qz):
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def R_to_quat_xyzw(R):
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


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    images_txt, ts_txt, out_tum = sys.argv[1:4]

    timestamps = {}
    with open(ts_txt) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                timestamps[int(parts[0])] = int(parts[1])

    rows = []
    with open(images_txt) as f:
        lines = f.readlines()
    # COLMAP images.txt: after the '#' comment header, each image is TWO lines —
    # a pose line then a points2D line — strictly alternating. Walk the
    # non-comment lines and take only the even-indexed (pose) ones; the odd
    # (points2D) lines also parse as floats, so a content-based filter would
    # double-count them.
    data_lines = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
    for idx in range(0, len(data_lines), 2):
        parts = data_lines[idx].split()
        if len(parts) < 10:
            continue
        try:
            qw, qx, qy, qz, tx, ty, tz = map(float, parts[1:8])
        except ValueError:
            continue
        # NAME may be a full path (COLMAP resolves symlinks), so take the
        # basename's numeric stem — NOT the first integer in the whole path,
        # which could match a digit in a parent directory (e.g. "MH_03").
        name = os.path.basename(parts[9])
        m = re.search(r"\d+", name)
        if m is None:
            continue
        frame = int(m.group(0))
        if frame not in timestamps:
            continue
        R_cw = quat_wxyz_to_R(qw, qx, qy, qz)
        t = np.array([tx, ty, tz])
        C = -R_cw.T @ t                 # camera centre in world
        R_wc = R_cw.T                   # camera-to-world rotation
        ox, oy, oz, ow = R_to_quat_xyzw(R_wc)
        ts = timestamps[frame] * 1e-9   # ns -> s (TUM seconds)
        rows.append((ts, C[0], C[1], C[2], ox, oy, oz, ow))

    rows.sort(key=lambda r: r[0])
    with open(out_tum, "w") as f:
        for r in rows:
            f.write("%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n" % r)
    print(f"wrote {len(rows)} poses to {out_tum}")


if __name__ == "__main__":
    main()
