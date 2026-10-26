#!/usr/bin/env python3
"""Score map-based localization (`basalt_euroc_localize_demo`) on EuRoC.

Every pose is an IMU/body pose. Estimates are TUM files with timestamps in
seconds (`localization.tum`, the SLAM demo's `trajectory_*.tum`); ground truth
is EuRoC's `state_groundtruth_estimate0/data.csv`. Poses are associated by
nearest timestamp within `--max-diff-ns`.

Modes:

  direct  estimate vs a reference trajectory in the same (map) frame, no
          alignment; e.g. localization vs the SLAM trajectory that built the map.
  gt      estimate vs ground truth after an SE(3) Umeyama fit on the estimate
          itself (the usual ATE).
  fixed   estimate vs ground truth through a map->GT transform fitted once on
          another trajectory (`--align-est` vs `--align-gt`), typically the SLAM
          run that built the map. Nothing is fitted to the scored poses, so this
          is the honest cross-session number.
"""

import argparse
import bisect
import json
from pathlib import Path

import numpy as np


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


def load_tum_seconds(path):
    """`timestamp_s tx ty tz qx qy qz qw` -> {timestamp_ns: (p, R)}."""
    poses = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) != 8:
            raise ValueError(f"{path}: TUM row must have 8 fields")
        values = [float(value) for value in fields[1:]]
        poses[int(round(float(fields[0]) * 1e9))] = (
            np.asarray(values[:3]),
            quaternion_matrix(*values[3:]),
        )
    if not poses:
        raise ValueError(f"{path}: empty trajectory")
    return poses


def load_euroc_ground_truth(path):
    """EuRoC `timestamp_ns,px,py,pz,qw,qx,qy,qz,...` -> {timestamp_ns: (p, R)}."""
    poses = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split(",")
        qw, qx, qy, qz = (float(value) for value in fields[4:8])
        poses[int(fields[0])] = (
            np.asarray([float(value) for value in fields[1:4]]),
            quaternion_matrix(qx, qy, qz, qw),
        )
    if not poses:
        raise ValueError(f"{path}: empty ground truth")
    return poses


def associate(estimate, reference, max_diff_ns):
    """Pairs (estimate_ns, reference_ns) of nearest timestamps."""
    keys = sorted(reference)
    pairs = []
    for timestamp in sorted(estimate):
        index = bisect.bisect_left(keys, timestamp)
        candidates = keys[max(index - 1, 0) : index + 1]
        nearest = min(candidates, key=lambda key: abs(key - timestamp))
        if abs(nearest - timestamp) <= max_diff_ns:
            pairs.append((timestamp, nearest))
    return pairs


def umeyama_se3(source, target):
    """R, t minimizing sum |target - (R source + t)|^2."""
    source_mean, target_mean = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    return rotation, target_mean - rotation @ source_mean


def fit_to_ground_truth(estimate, ground_truth, max_diff_ns):
    pairs = associate(estimate, ground_truth, max_diff_ns)
    if len(pairs) < 3:
        raise ValueError("fewer than three associated poses to fit an alignment")
    return umeyama_se3(
        np.asarray([estimate[a][0] for a, _ in pairs]),
        np.asarray([ground_truth[b][0] for _, b in pairs]),
    )


def pose_errors(estimate, reference, pairs, rotation=np.eye(3), translation=np.zeros(3)):
    """Translation (m) and rotation (deg) errors after mapping the estimate."""
    translation_errors, rotation_errors = [], []
    for a, b in pairs:
        position, orientation = estimate[a]
        reference_position, reference_orientation = reference[b]
        translation_errors.append(
            np.linalg.norm(rotation @ position + translation - reference_position)
        )
        delta = (rotation @ orientation).T @ reference_orientation
        cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
        rotation_errors.append(np.degrees(np.arccos(cosine)))
    return np.asarray(translation_errors), np.asarray(rotation_errors)


def summary(values):
    if len(values) == 0:
        return {}
    return {
        "rmse": float(np.sqrt(np.mean(values**2))),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def evaluate(mode, estimate, reference, max_diff_ns, alignment=None):
    rotation, translation = np.eye(3), np.zeros(3)
    if mode == "gt":
        rotation, translation = fit_to_ground_truth(estimate, reference, max_diff_ns)
    elif mode == "fixed":
        rotation, translation = alignment
    pairs = associate(estimate, reference, max_diff_ns)
    translation_errors, rotation_errors = pose_errors(
        estimate, reference, pairs, rotation, translation
    )
    return {
        "mode": mode,
        "estimated_poses": len(estimate),
        "associated": len(pairs),
        "translation_m": summary(translation_errors),
        "rotation_deg": summary(rotation_errors),
        "within_5cm": float(np.mean(translation_errors < 0.05)) if len(pairs) else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("direct", "gt", "fixed"), required=True)
    parser.add_argument("--estimate", type=Path, required=True, help="TUM, seconds")
    parser.add_argument(
        "--reference", type=Path, required=True, help="TUM (direct) or EuRoC GT csv (gt, fixed)"
    )
    parser.add_argument("--align-estimate", type=Path, help="fixed: TUM defining map->GT")
    parser.add_argument("--align-ground-truth", type=Path, help="fixed: GT csv for --align-estimate")
    parser.add_argument("--queries", type=int, help="query image count, adds localized_rate")
    parser.add_argument("--max-diff-ns", type=int, default=10_000_000)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()
    if args.mode == "fixed" and not (args.align_estimate and args.align_ground_truth):
        parser.error("--mode fixed needs --align-estimate and --align-ground-truth")

    estimate = load_tum_seconds(args.estimate)
    if args.mode == "direct":
        reference = load_tum_seconds(args.reference)
    else:
        reference = load_euroc_ground_truth(args.reference)
    alignment = None
    if args.mode == "fixed":
        alignment = fit_to_ground_truth(
            load_tum_seconds(args.align_estimate),
            load_euroc_ground_truth(args.align_ground_truth),
            args.max_diff_ns,
        )
    result = evaluate(args.mode, estimate, reference, args.max_diff_ns, alignment)
    if args.queries:
        result["localized_rate"] = len(estimate) / args.queries
    text = json.dumps(result, indent=2)
    if args.out_json:
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
