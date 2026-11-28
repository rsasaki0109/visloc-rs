#!/usr/bin/env python3
"""Score a Basalt trajectory on a FoMo sequence converted by fomo_to_euroc.py.

FoMo's ground truth is PPK-GNSS positions (no orientation), so the estimate's
IMU poses are moved to base_link (T_base_imu from extrinsics.json), matched
to the ground truth by time (linear interpolation of the 10 Hz positions),
and aligned with a rigid SE(3) (and, for reference, Sim(3)) Umeyama fit.
Reports ATE RMSE, the final-position error after the alignment, and the
fraction of the ground-truth time the estimate covers.

  python3 scripts/fomo_eval.py --sequence /data/fomo/summer_blue \\
      --estimate target/fomo_summer/trajectory.tum [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def quat_to_matrix(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def umeyama(src, dst, with_scale):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    u, d, vt = np.linalg.svd(b.T @ a / len(src))
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1
    r = u @ s @ vt
    c = (d * np.diag(s)).sum() / a.var(0).sum() if with_scale else 1.0
    return c, r, mu_d - c * r @ mu_s


def load_tum(path: Path):
    rows = [l.split() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    return np.array(rows, dtype=float)


def evaluate(sequence: Path, estimate: Path) -> dict:
    extr = json.loads((sequence / "extrinsics.json").read_text())
    base_imu = np.array(extr["T_base_imu"])
    imu_base = np.linalg.inv(base_imu)
    gt = load_tum(sequence / "gt_base.tum")
    est = load_tum(estimate)
    est = est[(est[:, 0] >= gt[0, 0]) & (est[:, 0] <= gt[-1, 0])]
    # Base-link origin in the estimate's world frame.
    p_est = np.array([quat_to_matrix(r[4:8]) @ imu_base[:3, 3] + r[1:4] for r in est])
    p_gt = np.stack([np.interp(est[:, 0], gt[:, 0], gt[:, i]) for i in (1, 2, 3)], axis=1)
    out = {"poses": len(est),
           "coverage": float((est[-1, 0] - est[0, 0]) / (gt[-1, 0] - gt[0, 0])) if len(est) else 0.0,
           "gt_length_m": float(np.linalg.norm(np.diff(gt[:, 1:4], axis=0), axis=1).sum())}
    for name, with_scale in (("se3", False), ("sim3", True)):
        c, r, t = umeyama(p_est, p_gt, with_scale)
        err = np.linalg.norm((c * (r @ p_est.T)).T + t - p_gt, axis=1)
        out[f"ate_{name}_m"] = float(np.sqrt((err ** 2).mean()))
        out[f"final_{name}_m"] = float(err[-1])
        if with_scale:
            out["sim3_scale"] = float(c)
    # Drift without a whole-trajectory fit: align on the first 10% only.
    n = max(10, len(p_est) // 10)
    c, r, t = umeyama(p_est[:n], p_gt[:n], False)
    err = np.linalg.norm((r @ p_est.T).T + t - p_gt, axis=1)
    out["final_first10pct_aligned_m"] = float(err[-1])
    out["final_drift_pct"] = 100.0 * float(err[-1]) / out["gt_length_m"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--estimate", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = evaluate(args.sequence, args.estimate)
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
