#!/usr/bin/env python3
"""Evaluate EuRoC-format body trajectories without exposing GT to the engine.

Estimated files may use the ORB-SLAM3/TUM convention
``timestamp_ns tx ty tz qx qy qz qw`` or visloc-rs `slam_trajectory.csv`
(``timestamp_ns,...,px,py,pz,qw,qx,qy,qz,tracking_success``). EuRoC ground truth uses
``timestamp_ns,px,py,pz,qw,qx,qy,qz,...``. Association is nearest-timestamp,
then a metric SE(3) Umeyama alignment is applied before translation ATE and
consecutive-pose RPE are measured. Sim(3) ATE is diagnostic only.
"""

import argparse
import bisect
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np


def timestamp_ns(token):
    """Parse integer-like timestamps without losing precision through float."""
    return int(token.strip().split(".", 1)[0])


def quaternion_matrix(x, y, z, w):
    q = np.asarray([x, y, z, w], dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid zero/non-finite quaternion")
    x, y, z, w = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def load_ground_truth(path):
    poses = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        for row in csv.reader(stream):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 8:
                raise ValueError("EuRoC ground-truth row has fewer than 8 fields")
            values = [float(value) for value in row[1:8]]
            px, py, pz, qw, qx, qy, qz = values
            poses.append(
                (timestamp_ns(row[0]), np.asarray([px, py, pz]), quaternion_matrix(qx, qy, qz, qw))
            )
    if not poses:
        raise ValueError("ground-truth trajectory is empty")
    poses.sort(key=lambda pose: pose[0])
    return poses


def trajectory_timestamp_ns(token, unit):
    if unit == "ns":
        return timestamp_ns(token)
    if unit == "s":
        return int(round(float(token) * 1e9))
    raise ValueError(f"unsupported trajectory timestamp unit: {unit}")


def load_estimate(path, tum_time_unit="ns"):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        first = next(
            (line.strip() for line in stream if line.strip() and not line.lstrip().startswith("#")),
            "",
        )
    if "," in first:
        return load_visloc_estimate(path)

    poses = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) != 8:
                raise ValueError("estimated trajectory row must have 8 fields")
            values = [float(value) for value in fields[1:]]
            tx, ty, tz, qx, qy, qz, qw = values
            poses.append(
                (
                    trajectory_timestamp_ns(fields[0], tum_time_unit),
                    np.asarray([tx, ty, tz]),
                    quaternion_matrix(qx, qy, qz, qw),
                )
            )
    if not poses:
        raise ValueError("estimated trajectory is empty")
    poses.sort(key=lambda pose: pose[0])
    return poses


def load_visloc_estimate(path):
    poses = []
    required = {"timestamp_ns", "px", "py", "pz", "qw", "qx", "qy", "qz"}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise ValueError(
                "visloc trajectory is missing columns: {}".format(", ".join(sorted(missing)))
            )
        for row in reader:
            success = row.get("tracking_success")
            if success is not None and success.strip().lower() not in {"1", "true"}:
                continue
            px, py, pz = (float(row[name]) for name in ("px", "py", "pz"))
            qw, qx, qy, qz = (float(row[name]) for name in ("qw", "qx", "qy", "qz"))
            poses.append(
                (
                    timestamp_ns(row["timestamp_ns"]),
                    np.asarray([px, py, pz]),
                    quaternion_matrix(qx, qy, qz, qw),
                )
            )
    if not poses:
        raise ValueError("estimated trajectory is empty")
    poses.sort(key=lambda pose: pose[0])
    return poses


def restrict_to_common_timestamps(estimates):
    """Keep only estimate timestamps present in every compared trajectory."""
    if not estimates:
        return estimates
    common = set(pose[0] for pose in estimates[0])
    for estimate in estimates[1:]:
        common.intersection_update(pose[0] for pose in estimate)
    if len(common) < 3:
        raise ValueError("fewer than 3 common estimate timestamps")
    return [[pose for pose in estimate if pose[0] in common] for estimate in estimates]


def associate(ground_truth, estimate, max_diff_ns):
    gt_stamps = [pose[0] for pose in ground_truth]
    pairs = []
    used_gt = set()
    for est in estimate:
        index = bisect.bisect_left(gt_stamps, est[0])
        candidates = [candidate for candidate in (index - 1, index) if 0 <= candidate < len(ground_truth)]
        if not candidates:
            continue
        best = min(candidates, key=lambda candidate: abs(gt_stamps[candidate] - est[0]))
        delta = abs(gt_stamps[best] - est[0])
        if delta <= max_diff_ns and best not in used_gt:
            used_gt.add(best)
            pairs.append((ground_truth[best], est, delta))
    return pairs


def umeyama(src, dst, with_scale):
    if len(src) < 3:
        raise ValueError("at least 3 associated poses are required")
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_centered, dst_centered = src - src_mean, dst - dst_mean
    covariance = src_centered.T @ dst_centered / len(src)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, sign])
    rotation = vt.T @ correction @ u.T
    variance = float(np.square(src_centered).sum() / len(src))
    scale = float((singular * np.asarray([1.0, 1.0, sign])).sum() / variance) if with_scale else 1.0
    translation = dst_mean - scale * rotation @ src_mean
    return scale, rotation, translation


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def rotation_angle(rotation):
    cosine = max(-1.0, min(1.0, (float(np.trace(rotation)) - 1.0) * 0.5))
    return math.acos(cosine)


def evaluate(ground_truth, estimate, max_diff_ns):
    pairs = associate(ground_truth, estimate, max_diff_ns)
    if len(pairs) < 3:
        raise ValueError("fewer than 3 trajectory poses associated to ground truth")
    gt_positions = np.asarray([pair[0][1] for pair in pairs])
    est_positions = np.asarray([pair[1][1] for pair in pairs])
    _, align_rotation, align_translation = umeyama(est_positions, gt_positions, False)
    sim_scale, sim_rotation, sim_translation = umeyama(est_positions, gt_positions, True)
    aligned_positions = (align_rotation @ est_positions.T).T + align_translation
    sim_positions = (sim_scale * (sim_rotation @ est_positions.T)).T + sim_translation
    aligned_rotations = [align_rotation @ pair[1][2] for pair in pairs]
    gt_rotations = [pair[0][2] for pair in pairs]

    rpe_translation = []
    rpe_rotation_degrees = []
    for index in range(len(pairs) - 1):
        gt_delta_rotation = gt_rotations[index].T @ gt_rotations[index + 1]
        est_delta_rotation = aligned_rotations[index].T @ aligned_rotations[index + 1]
        gt_delta_translation = gt_rotations[index].T @ (
            gt_positions[index + 1] - gt_positions[index]
        )
        est_delta_translation = aligned_rotations[index].T @ (
            aligned_positions[index + 1] - aligned_positions[index]
        )
        error_rotation = gt_delta_rotation.T @ est_delta_rotation
        error_translation = gt_delta_rotation.T @ (
            est_delta_translation - gt_delta_translation
        )
        rpe_translation.append(float(np.linalg.norm(error_translation)))
        rpe_rotation_degrees.append(math.degrees(rotation_angle(error_rotation)))

    return {
        "estimate_poses": len(estimate),
        "associated_poses": len(pairs),
        "association_ratio": len(pairs) / len(estimate),
        "max_association_delta_ns": max(pair[2] for pair in pairs),
        "ate_translation_se3_m": distribution(np.linalg.norm(aligned_positions - gt_positions, axis=1)),
        "ate_translation_sim3_m": distribution(np.linalg.norm(sim_positions - gt_positions, axis=1)),
        "sim3_scale": sim_scale,
        "rpe_translation_consecutive_m": distribution(rpe_translation),
        "rpe_rotation_consecutive_deg": distribution(rpe_rotation_degrees),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-csv", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--max-diff-ns", type=int, default=10_000_000)
    parser.add_argument(
        "--tum-time-unit",
        choices=("ns", "s"),
        default="ns",
        help="timestamp unit for whitespace/TUM trajectories (default: ns)",
    )
    parser.add_argument(
        "--common-estimate-timestamps",
        action="store_true",
        help="score every trajectory on their exact common timestamp subset",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    if args.max_diff_ns < 0:
        parser.error("--max-diff-ns must be non-negative")
    ground_truth = load_ground_truth(args.ground_truth_csv)
    estimates = [
        load_estimate(trajectory, tum_time_unit=args.tum_time_unit)
        for trajectory in args.trajectory
    ]
    if args.common_estimate_timestamps:
        estimates = restrict_to_common_timestamps(estimates)
    results = []
    for trajectory, estimate in zip(args.trajectory, estimates):
        result = evaluate(
            ground_truth,
            estimate,
            args.max_diff_ns,
        )
        result["trajectory"] = str(trajectory.resolve())
        results.append(result)
    payload = {
        "schema_version": 1,
        "protocol": {
            "ground_truth_used_after_engine_exit": True,
            "association": "nearest_timestamp_one_to_one",
            "max_diff_ns": args.max_diff_ns,
            "tum_time_unit": args.tum_time_unit,
            "common_estimate_timestamps": args.common_estimate_timestamps,
            "alignment": "SE(3) Umeyama; Sim(3) diagnostic only",
            "rpe_delta": "consecutive associated poses",
        },
        "ground_truth": str(args.ground_truth_csv.resolve()),
        "runs": results,
        "median": {
            "ate_translation_se3_rmse_m": statistics.median(
                result["ate_translation_se3_m"]["rmse"] for result in results
            ),
            "rpe_translation_rmse_m": statistics.median(
                result["rpe_translation_consecutive_m"]["rmse"] for result in results
            ),
            "rpe_rotation_rmse_deg": statistics.median(
                result["rpe_rotation_consecutive_deg"]["rmse"] for result in results
            ),
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
