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
          another trajectory (`--align-estimate` vs `--align-ground-truth`), the SLAM
          run that built the map. Nothing is fitted to the scored poses, so this
          is the honest cross-session number.
  local   like fixed, but the map->GT transform is fitted per scored pose on the
          stretch of --align-estimate within --align-radius-m of it (in the map
          frame) and, by the ground truth, facing within 90 deg of it. On a map
          whose SLAM drifted over a long loop this scores the localization
          against the map around it, not the map's own drift; the facing keeps
          a pass the other way, whose landmarks it did not see, out.

Without `--prior-radius-m`, `basalt_euroc_localize_demo` localizes every image
independently with a fixed RANSAC seed, so one run at a low `--min-inliers`
contains the result of every higher threshold: `--per-frame` with
`--min-inliers N` keeps the poses whose logged inlier count is at least N.
"""

import argparse
import bisect
import csv
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


def filter_by_inliers(estimate, per_frame_path, min_inliers, max_diff_ns=1_000_000):
    """Keep the poses whose `per_frame.csv` row succeeded with >= min_inliers."""
    keep = set()
    keys = sorted(estimate)
    with Path(per_frame_path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["success"] != "true" or int(row["inliers"]) < min_inliers:
                continue
            timestamp = int(row["timestamp_ns"])
            index = bisect.bisect_left(keys, timestamp)
            candidates = keys[max(index - 1, 0) : index + 1]
            if candidates:
                nearest = min(candidates, key=lambda key: abs(key - timestamp))
                if abs(nearest - timestamp) <= max_diff_ns:
                    keep.add(nearest)
    return {timestamp: pose for timestamp, pose in estimate.items() if timestamp in keep}


def umeyama_se3(source, target):
    """R, t minimizing sum |target - (R source + t)|^2."""
    source_mean, target_mean = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    return rotation, target_mean - rotation @ source_mean


def fit_yaw(source, target):
    """R (about z only), t minimizing sum |target - (R source + t)|^2."""
    source_mean, target_mean = source.mean(0), target.mean(0)
    a, b = source - source_mean, target - target_mean
    yaw = np.arctan2(
        np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]),
        np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]),
    )
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
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


def local_errors(estimate, reference, pairs, sources, targets, radius_m, rotations=None):
    """pose_errors with a map->GT fit on the map stretch around each pose.

    sources and targets are the map and GT positions of --align-estimate and
    --align-ground-truth, in time order, and rotations their GT orientations.
    The stretch is the run of map positions within radius_m of the pose, and
    facing within 90 deg of it by the ground truth, through the nearest one: a
    place the map passed twice is fitted on the pass whose view the pose
    shares, however far off the pose's own heading is. The fit
    turns about the vertical only, as both frames are level, which a straight
    stretch would not pin down otherwise. Poses with fewer than three map
    positions around them are not scored.
    """
    translation_errors, rotation_errors = [], []
    for pair in pairs:
        distances = np.linalg.norm(sources - estimate[pair[0]][0], axis=1)
        near = distances <= radius_m
        if rotations is not None:
            # Less than 90 deg between the two true orientations:
            # trace(R_map^T R) > 1.
            near &= np.einsum("nij,ij->n", rotations, reference[pair[1]][1]) > 1.0
        distances = np.where(near, distances, np.inf)
        if not np.isfinite(distances).any():
            continue
        first = last = int(np.argmin(distances))
        while first > 0 and near[first - 1]:
            first -= 1
        while last + 1 < len(near) and near[last + 1]:
            last += 1
        if last - first < 2:
            continue
        rotation, translation = fit_yaw(sources[first : last + 1], targets[first : last + 1])
        translation_error, rotation_error = pose_errors(
            estimate, reference, [pair], rotation, translation
        )
        translation_errors.append(translation_error[0])
        rotation_errors.append(rotation_error[0])
    return np.asarray(translation_errors), np.asarray(rotation_errors)


def evaluate(mode, estimate, reference, max_diff_ns, alignment=None):
    rotation, translation = np.eye(3), np.zeros(3)
    if mode == "gt":
        rotation, translation = fit_to_ground_truth(estimate, reference, max_diff_ns)
    elif mode == "fixed":
        rotation, translation = alignment
    pairs = associate(estimate, reference, max_diff_ns)
    if mode == "local":
        translation_errors, rotation_errors = local_errors(estimate, reference, pairs, *alignment)
    else:
        translation_errors, rotation_errors = pose_errors(
            estimate, reference, pairs, rotation, translation
        )
    return {
        "mode": mode,
        "estimated_poses": len(estimate),
        "associated": len(pairs),
        "scored": len(translation_errors),
        "translation_m": summary(translation_errors),
        "rotation_deg": summary(rotation_errors),
        "within_5cm": float(np.mean(translation_errors < 0.05)) if len(translation_errors) else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("direct", "gt", "fixed", "local"), required=True)
    parser.add_argument("--estimate", type=Path, required=True, help="TUM, seconds")
    parser.add_argument(
        "--reference", type=Path, required=True, help="TUM (direct) or EuRoC GT csv (gt, fixed, local)"
    )
    parser.add_argument("--align-estimate", type=Path, help="fixed, local: TUM defining map->GT")
    parser.add_argument(
        "--align-ground-truth", type=Path, help="fixed, local: GT csv for --align-estimate"
    )
    parser.add_argument(
        "--align-radius-m", type=float, default=5.0, help="local: map poses used per scored pose"
    )
    parser.add_argument("--queries", type=int, help="query image count, adds localized_rate")
    parser.add_argument("--per-frame", type=Path, help="the run's per_frame.csv, for --min-inliers")
    parser.add_argument(
        "--min-inliers", type=int, help="score only poses accepted with at least this many inliers"
    )
    parser.add_argument("--max-diff-ns", type=int, default=10_000_000)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()
    if args.mode in ("fixed", "local") and not (args.align_estimate and args.align_ground_truth):
        parser.error(f"--mode {args.mode} needs --align-estimate and --align-ground-truth")
    if (args.min_inliers is None) != (args.per_frame is None):
        parser.error("--min-inliers and --per-frame go together")

    estimate = load_tum_seconds(args.estimate)
    if args.min_inliers is not None:
        estimate = filter_by_inliers(estimate, args.per_frame, args.min_inliers)
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
    elif args.mode == "local":
        map_poses = load_tum_seconds(args.align_estimate)
        map_ground_truth = load_euroc_ground_truth(args.align_ground_truth)
        map_pairs = associate(map_poses, map_ground_truth, args.max_diff_ns)
        alignment = (
            np.asarray([map_poses[a][0] for a, _ in map_pairs]),
            np.asarray([map_ground_truth[b][0] for _, b in map_pairs]),
            args.align_radius_m,
            np.asarray([map_ground_truth[b][1] for _, b in map_pairs]),
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
