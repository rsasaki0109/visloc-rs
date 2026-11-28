#!/usr/bin/env python3
"""Re-estimate a stereo pair's relative rotation from the sequence's images.

A Basalt-format calibration.json (Double Sphere cameras) gives the right
camera's pose in the IMU frame. A small error in the left-right rotation,
mostly about the cameras' vertical axis, shifts every stereo disparity by a
near-constant amount, and stereo VIO then comes out with a wrong metric
scale that the IMU does not fully correct. On UZH-FPV, 0.4 deg of yaw made the
trajectory 18% too short (indoor_forward_9: ATE 1.33 m -> 0.30 m after this
fix). FoMo had the same problem, a published pre-rectification rotation
(see docs/fomo_forest.md).

This keeps the baseline vector, matches ORB features between simultaneous
left and right images, and finds the rotation correction that minimises the
epipolar error of the unit-ray matches (robust Gauss-Newton). Only the
images are used, never the ground truth.

  python3 scripts/stereo_rotation_selfcal.py --sequence /data/fpv/euroc/indoor_forward_9 \\
      [--out calibration_selfcal.json] [--step 40]

Writes the corrected calibration (default: <sequence>/calibration_selfcal.json)
and prints the correction in degrees about the left camera's x, y, z axes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def quat_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_quat(r):
    w = np.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return x, y, z, w


def to_matrix(d):
    t = np.eye(4)
    t[:3, :3] = quat_matrix(d["qx"], d["qy"], d["qz"], d["qw"])
    t[:3, 3] = d["px"], d["py"], d["pz"]
    return t


def rotvec_matrix(w):
    angle = np.linalg.norm(w)
    if angle < 1e-15:
        return np.eye(3)
    k = w / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def ds_unproject(pixels, k):
    fx, fy, cx, cy, xi, alpha = (k[n] for n in ("fx", "fy", "cx", "cy", "xi", "alpha"))
    mx = (pixels[:, 0] - cx) / fx
    my = (pixels[:, 1] - cy) / fy
    r2 = mx * mx + my * my
    mz = (1 - alpha * alpha * r2) / (alpha * np.sqrt(np.maximum(1 - (2 * alpha - 1) * r2, 0)) + 1 - alpha)
    s = (mz * xi + np.sqrt(mz * mz + (1 - xi * xi) * r2)) / (mz * mz + r2)
    rays = np.stack([s * mx, s * my, s * mz - xi], axis=1)
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def match_rays(sequence: Path, calib: dict, step: int, max_features: int):
    left_names = sorted((sequence / "mav0/cam0/data").iterdir())[::step]
    orb = cv2.ORB_create(max_features)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    left, right = [], []
    for path in left_names:
        other = sequence / "mav0/cam1/data" / path.name
        if not other.exists():
            continue
        a, b = cv2.imread(str(path), 0), cv2.imread(str(other), 0)
        ka, da = orb.detectAndCompute(a, None)
        kb, db = orb.detectAndCompute(b, None)
        if da is None or db is None:
            continue
        for m in matcher.match(da, db):
            left.append(ka[m.queryIdx].pt)
            right.append(kb[m.trainIdx].pt)
    k0 = calib["intrinsics"][0]["intrinsics"]
    k1 = calib["intrinsics"][1]["intrinsics"]
    return ds_unproject(np.array(left), k0), ds_unproject(np.array(right), k1), len(left_names)


def epipolar(left, right, rotation, baseline):
    t = baseline / np.linalg.norm(baseline)
    essential = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]) @ rotation
    return np.einsum("ij,ij->i", left, right @ essential.T)


def refine(left, right, rotation, baseline, iterations=30):
    """Correction w (left frame) with rotvec(w) @ rotation minimising a
    Huber-weighted epipolar error; outliers dropped between rounds."""
    w = np.zeros(3)
    keep = np.ones(len(left), dtype=bool)
    for _round in range(3):
        for _ in range(iterations):
            r = epipolar(left[keep], right[keep], rotvec_matrix(w) @ rotation, baseline)
            jac = np.empty((r.size, 3))
            for i in range(3):
                dw = np.zeros(3)
                dw[i] = 1e-7
                jac[:, i] = (epipolar(left[keep], right[keep], rotvec_matrix(w + dw) @ rotation,
                                      baseline) - r) / 1e-7
            scale = 1.4826 * np.median(np.abs(r))
            weight = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(r), 1e-12))
            h = jac.T @ (jac * weight[:, None])
            delta = np.linalg.solve(h + 1e-9 * np.eye(3), -jac.T @ (weight * r))
            w += delta
            if np.abs(delta).max() < 1e-9:
                break
        residual = np.abs(epipolar(left, right, rotvec_matrix(w) @ rotation, baseline))
        keep = residual < 3 * np.median(residual)
    return w


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, help="default: <sequence>/calibration.json")
    parser.add_argument("--out", type=Path, help="default: <sequence>/calibration_selfcal.json")
    parser.add_argument("--step", type=int, default=40, help="use every n-th stereo pair")
    parser.add_argument("--max-features", type=int, default=1500)
    args = parser.parse_args()

    calib_path = args.calibration or args.sequence / "calibration.json"
    full = json.loads(calib_path.read_text())
    calib = full["value0"]
    imu_left = to_matrix(calib["T_imu_cam"][0])
    imu_right = to_matrix(calib["T_imu_cam"][1])
    left_right = np.linalg.inv(imu_left) @ imu_right
    rotation, baseline = left_right[:3, :3], left_right[:3, 3]

    left, right, pairs = match_rays(args.sequence, calib, args.step, args.max_features)
    before = np.abs(epipolar(left, right, rotation, baseline))
    w = refine(left, right, rotation, baseline)
    corrected = rotvec_matrix(w) @ rotation
    after = np.abs(epipolar(left, right, corrected, baseline))

    new_left_right = np.eye(4)
    new_left_right[:3, :3] = corrected
    new_left_right[:3, 3] = baseline
    imu_right_new = imu_left @ new_left_right
    qx, qy, qz, qw = matrix_quat(imu_right_new[:3, :3])
    calib["T_imu_cam"][1] = {"px": float(imu_right_new[0, 3]), "py": float(imu_right_new[1, 3]),
                             "pz": float(imu_right_new[2, 3]), "qx": float(qx), "qy": float(qy),
                             "qz": float(qz), "qw": float(qw)}
    out = args.out or args.sequence / "calibration_selfcal.json"
    out.write_text(json.dumps(full, indent=2) + "\n")
    print(json.dumps({
        "sequence": str(args.sequence), "stereo_pairs": pairs, "matches": int(len(left)),
        "correction_deg_xyz": np.degrees(w).round(4).tolist(),
        "correction_deg": round(float(np.degrees(np.linalg.norm(w))), 4),
        "median_epipolar_mdeg_before": round(float(np.degrees(np.median(before)) * 1000), 1),
        "median_epipolar_mdeg_after": round(float(np.degrees(np.median(after)) * 1000), 1),
        "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
