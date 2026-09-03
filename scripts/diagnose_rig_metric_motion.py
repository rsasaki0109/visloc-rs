#!/usr/bin/env python3
"""Compare GT-free stereo 3D-to-3D motion with reconstructed rig motion.

Each synchronized stereo observation is triangulated in its own calibrated
rig frame, without using reconstructed world points or trajectory ground
truth. Tracks shared by nearby frames then provide deterministic RANSAC/Kabsch
metric relative poses. Comparing their translation magnitudes with the mapper
trajectory exposes accumulated scale drift that reprojection-only diagnostics
cannot observe.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import diagnose_rig_stereo_track_consistency as stereo  # noqa: E402


@dataclass(frozen=True)
class SensorPose:
    sensor_from_rig_rotation: np.ndarray
    sensor_from_rig_translation: np.ndarray


def load_sensor_poses(path: Path) -> dict[int, SensorPose]:
    sensors = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if not fields or fields[0].startswith("#") or fields[0] == "F":
            continue
        if fields[0] != "S" or len(fields) != 16:
            raise stereo.DiagnosticError(f"malformed sensor row {path}:{line_number}")
        sensor_index = int(fields[1])
        rotation = stereo.rotation(*(float(value) for value in fields[9:13]))
        translation = np.array([float(value) for value in fields[13:16]], dtype=float)
        if sensor_index in sensors or not np.all(np.isfinite(translation)):
            raise stereo.DiagnosticError(f"invalid or duplicate sensor {sensor_index} in {path}")
        sensors[sensor_index] = SensorPose(rotation, translation)
    if len(sensors) < 2:
        raise stereo.DiagnosticError(f"fewer than two calibrated sensors in {path}")
    return sensors


def local_ray(
    image: stereo.Image,
    camera: stereo.Camera,
    sensor_pose: SensorPose,
    point_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if point_index < 0 or point_index >= len(image.points2d):
        raise stereo.DiagnosticError(
            f"POINT2D index {point_index} is outside image {image.name!r}"
        )
    x, y, _ = image.points2d[point_index]
    camera_ray = np.array(
        [(x - camera.cx) / camera.fx, (y - camera.cy) / camera.fy, 1.0]
    )
    rig_from_sensor = sensor_pose.sensor_from_rig_rotation.T
    center = -(rig_from_sensor @ sensor_pose.sensor_from_rig_translation)
    direction = rig_from_sensor @ camera_ray
    direction /= np.linalg.norm(direction)
    return center, direction


def widest_stereo_point(frame_observations, min_angle_deg: float):
    best = None
    for left_index, (left_sensor, left_ray) in enumerate(frame_observations):
        for right_sensor, right_ray in frame_observations[left_index + 1 :]:
            if left_sensor == right_sensor:
                continue
            cosine = float(np.clip(np.dot(left_ray[1], right_ray[1]), -1.0, 1.0))
            angle = math.degrees(math.acos(cosine))
            if best is None or angle > best[0]:
                best = (angle, left_ray, right_ray)
    if best is None:
        return None
    return stereo.triangulate(best[1], best[2], min_angle_deg)


def fit_rigid(source: np.ndarray, target: np.ndarray):
    if len(source) < 3 or source.shape != target.shape:
        return None
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = source_zero.T @ target_zero
    try:
        left, singular, right_t = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return None
    if singular[1] <= 1e-10:
        return None
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1] *= -1
        rotation = right_t.T @ left.T
    translation = target_center - rotation @ source_center
    if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
        return None
    return rotation, translation


def robust_rigid(source: np.ndarray, target: np.ndarray, threshold_m: float, iterations: int, seed: int):
    if len(source) < 3:
        return None
    rng = np.random.default_rng(seed)
    samples = [np.arange(len(source))]
    samples.extend(rng.choice(len(source), 3, replace=False) for _ in range(iterations))
    best = None
    for sample in samples:
        fit = fit_rigid(source[sample], target[sample])
        if fit is None:
            continue
        rotation, translation = fit
        residuals = np.linalg.norm((rotation @ source.T).T + translation - target, axis=1)
        inliers = residuals <= threshold_m
        count = int(np.count_nonzero(inliers))
        if count < 3:
            continue
        median = float(np.median(residuals[inliers]))
        if best is None or count > best[0] or (count == best[0] and median < best[1]):
            best = (count, median, inliers)
    if best is None:
        return None
    fit = fit_rigid(source[best[2]], target[best[2]])
    if fit is None:
        return None
    rotation, translation = fit
    residuals = np.linalg.norm((rotation @ source.T).T + translation - target, axis=1)
    inliers = residuals <= threshold_m
    return rotation, translation, inliers, residuals


def reconstructed_rig_poses(images, assignments, aliases, sensors):
    poses = {}
    for image in images.values():
        flat_name = aliases.get(image.name, image.name)
        if flat_name not in assignments:
            raise stereo.DiagnosticError(f"manifest has no assignment for image {image.name!r}")
        frame, sensor_index = assignments[flat_name]
        sensor = sensors[sensor_index]
        world_to_sensor_rotation = image.camera_to_world.T
        world_to_sensor_translation = -(world_to_sensor_rotation @ image.center)
        rig_from_sensor = sensor.sensor_from_rig_rotation.T
        world_to_rig_rotation = rig_from_sensor @ world_to_sensor_rotation
        world_to_rig_translation = rig_from_sensor @ (
            world_to_sensor_translation - sensor.sensor_from_rig_translation
        )
        candidate = (world_to_rig_rotation, world_to_rig_translation)
        previous = poses.get(frame)
        if previous is not None:
            rotation_error = previous[0] @ world_to_rig_rotation.T
            angle = math.degrees(
                math.acos(float(np.clip((np.trace(rotation_error) - 1.0) * 0.5, -1, 1)))
            )
            if angle > 1e-3 or np.linalg.norm(previous[1] - world_to_rig_translation) > 1e-5:
                raise stereo.DiagnosticError(
                    f"registered sensors disagree on reconstructed rig pose for frame {frame}"
                )
        else:
            poses[frame] = candidate
    return poses


def percentile(values: list[float], fraction: float):
    return stereo.percentile(values, fraction)


def summarize(values: list[float]) -> dict:
    return {
        "count": len(values),
        "median": percentile(values, 0.5),
        "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
        "min": min(values, default=None),
        "max": max(values, default=None),
    }


def summarize_pose_consistent(
    rows: list[dict], max_rotation_error_deg: float, min_direction_cosine: float
) -> dict:
    selected = [
        row
        for row in rows
        if row["rotation_error_deg"] <= max_rotation_error_deg
        and row["translation_direction_cosine"] >= min_direction_cosine
    ]
    return {
        "pair_count": len(selected),
        "metric_to_mapper_translation_ratio": summarize(
            [row["metric_to_mapper_translation_ratio"] for row in selected]
        ),
        "inlier_fraction": summarize([row["inlier_fraction"] for row in selected]),
        "inlier_residual_median_m": summarize(
            [row["inlier_residual_median_m"] for row in selected]
        ),
    }


def diagnose(
    model: Path,
    manifest_path: Path,
    image_aliases_path: Path | None,
    min_frame_gap: int,
    max_frame_gap: int,
    min_correspondences: int,
    min_inliers: int,
    min_angle_deg: float,
    ransac_threshold_m: float,
    ransac_iterations: int,
    bin_size: int,
    max_rotation_error_deg: float = 1.0,
    min_direction_cosine: float = 0.99,
) -> dict:
    cameras = stereo.load_cameras(model / "cameras.txt")
    images = stereo.load_images(model / "images.txt")
    assignments = stereo.load_manifest(manifest_path)
    aliases = stereo.load_image_aliases(image_aliases_path) if image_aliases_path else {}
    sensors = load_sensor_poses(manifest_path)
    tracks = stereo.load_tracks(model / "points3D.txt")
    rig_poses = reconstructed_rig_poses(images, assignments, aliases, sensors)

    pair_correspondences = defaultdict(list)
    stereo_track_count = 0
    stereo_point_count = 0
    for _, observations in tracks.values():
        by_frame = defaultdict(list)
        for image_id, point_index in observations:
            image = images[image_id]
            flat_name = aliases.get(image.name, image.name)
            frame, sensor_index = assignments[flat_name]
            by_frame[frame].append(
                (
                    sensor_index,
                    local_ray(image, cameras[image.camera_id], sensors[sensor_index], point_index),
                )
            )
        local_points = []
        for frame, frame_observations in by_frame.items():
            point = widest_stereo_point(frame_observations, min_angle_deg)
            if point is not None:
                local_points.append((frame, point))
        local_points.sort(key=lambda item: item[0])
        if len(local_points) >= 2:
            stereo_track_count += 1
        stereo_point_count += len(local_points)
        for left_index, (left_frame, left_point) in enumerate(local_points):
            for right_frame, right_point in local_points[left_index + 1 :]:
                gap = right_frame - left_frame
                if gap > max_frame_gap:
                    break
                if gap < min_frame_gap:
                    continue
                pair_correspondences[(left_frame, right_frame)].append((left_point, right_point))

    results = []
    rejected_low_support = 0
    rejected_ransac = 0
    for (left_frame, right_frame), correspondences in sorted(pair_correspondences.items()):
        if len(correspondences) < min_correspondences:
            rejected_low_support += 1
            continue
        if left_frame not in rig_poses or right_frame not in rig_poses:
            continue
        source = np.stack([item[0] for item in correspondences])
        target = np.stack([item[1] for item in correspondences])
        robust = robust_rigid(
            source,
            target,
            ransac_threshold_m,
            ransac_iterations,
            seed=(left_frame * 1_000_003 + right_frame),
        )
        if robust is None or int(np.count_nonzero(robust[2])) < min_inliers:
            rejected_ransac += 1
            continue
        metric_rotation, metric_translation, inliers, residuals = robust
        left_rotation, left_translation = rig_poses[left_frame]
        right_rotation, right_translation = rig_poses[right_frame]
        mapper_rotation = right_rotation @ left_rotation.T
        mapper_translation = right_translation - mapper_rotation @ left_translation
        mapper_norm = float(np.linalg.norm(mapper_translation))
        metric_norm = float(np.linalg.norm(metric_translation))
        if mapper_norm <= 1e-9 or metric_norm <= 1e-9:
            continue
        rotation_delta = metric_rotation @ mapper_rotation.T
        rotation_error_deg = math.degrees(
            math.acos(float(np.clip((np.trace(rotation_delta) - 1.0) * 0.5, -1, 1)))
        )
        direction_cosine = float(
            np.dot(metric_translation, mapper_translation) / (metric_norm * mapper_norm)
        )
        results.append(
            {
                "left_frame": left_frame,
                "right_frame": right_frame,
                "gap": right_frame - left_frame,
                "correspondences": len(correspondences),
                "inliers": int(np.count_nonzero(inliers)),
                "inlier_fraction": float(np.mean(inliers)),
                "inlier_residual_median_m": float(np.median(residuals[inliers])),
                "metric_translation_m": metric_norm,
                "mapper_translation_m": mapper_norm,
                "metric_to_mapper_translation_ratio": metric_norm / mapper_norm,
                "rotation_error_deg": rotation_error_deg,
                "translation_direction_cosine": direction_cosine,
            }
        )

    bins = {}
    for result in results:
        start = (result["left_frame"] // bin_size) * bin_size
        bins.setdefault(start, []).append(result)
    bin_summary = {}
    pose_consistent_bin_summary = {}
    for start, rows in sorted(bins.items()):
        label = f"{start}-{start + bin_size - 1}"
        bin_summary[label] = {
            "pair_count": len(rows),
            "metric_to_mapper_translation_ratio": summarize(
                [row["metric_to_mapper_translation_ratio"] for row in rows]
            ),
            "rotation_error_deg": summarize([row["rotation_error_deg"] for row in rows]),
            "translation_direction_cosine": summarize(
                [row["translation_direction_cosine"] for row in rows]
            ),
        }
        pose_consistent_bin_summary[label] = summarize_pose_consistent(
            rows, max_rotation_error_deg, min_direction_cosine
        )
    return {
        "schema": "visloc_rig_metric_interframe_motion_v1",
        "ground_truth_used": False,
        "model": str(model),
        "manifest": str(manifest_path),
        "image_aliases": str(image_aliases_path) if image_aliases_path else None,
        "registered_images": len(images),
        "registered_frames": len(rig_poses),
        "tracks": len(tracks),
        "tracks_with_two_local_stereo_points": stereo_track_count,
        "local_stereo_points": stereo_point_count,
        "candidate_frame_pairs": len(pair_correspondences),
        "accepted_frame_pairs": len(results),
        "rejected_low_support_pairs": rejected_low_support,
        "rejected_ransac_pairs": rejected_ransac,
        "config": {
            "max_frame_gap": max_frame_gap,
            "min_frame_gap": min_frame_gap,
            "min_correspondences": min_correspondences,
            "min_inliers": min_inliers,
            "min_triangulation_angle_deg": min_angle_deg,
            "ransac_threshold_m": ransac_threshold_m,
            "ransac_iterations": ransac_iterations,
            "frame_bin_size": bin_size,
            "pose_consistent_max_rotation_error_deg": max_rotation_error_deg,
            "pose_consistent_min_direction_cosine": min_direction_cosine,
        },
        "all_pairs": {
            "metric_translation_m": summarize(
                [row["metric_translation_m"] for row in results]
            ),
            "mapper_translation_m": summarize(
                [row["mapper_translation_m"] for row in results]
            ),
            "metric_to_mapper_translation_ratio": summarize(
                [row["metric_to_mapper_translation_ratio"] for row in results]
            ),
            "rotation_error_deg": summarize([row["rotation_error_deg"] for row in results]),
            "translation_direction_cosine": summarize(
                [row["translation_direction_cosine"] for row in results]
            ),
            "inlier_fraction": summarize([row["inlier_fraction"] for row in results]),
            "inlier_residual_median_m": summarize(
                [row["inlier_residual_median_m"] for row in results]
            ),
        },
        "frame_bins": bin_summary,
        "pose_consistent_pairs": summarize_pose_consistent(
            results, max_rotation_error_deg, min_direction_cosine
        ),
        "pose_consistent_frame_bins": pose_consistent_bin_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-aliases", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-frame-gap", type=int, default=2)
    parser.add_argument("--min-frame-gap", type=int, default=1)
    parser.add_argument("--min-correspondences", type=int, default=12)
    parser.add_argument("--min-inliers", type=int, default=8)
    parser.add_argument("--min-triangulation-angle-deg", type=float, default=1.0)
    parser.add_argument("--ransac-threshold-m", type=float, default=0.15)
    parser.add_argument("--ransac-iterations", type=int, default=128)
    parser.add_argument("--frame-bin-size", type=int, default=500)
    parser.add_argument("--pose-consistent-max-rotation-error-deg", type=float, default=1.0)
    parser.add_argument("--pose-consistent-min-direction-cosine", type=float, default=0.99)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.min_frame_gap <= 0
        or args.max_frame_gap < args.min_frame_gap
        or args.min_correspondences < 3
        or args.min_inliers < 3
    ):
        raise stereo.DiagnosticError("frame gap must be positive and support gates at least three")
    if args.min_inliers > args.min_correspondences:
        raise stereo.DiagnosticError("--min-inliers cannot exceed --min-correspondences")
    if args.ransac_iterations <= 0 or args.frame_bin_size <= 0:
        raise stereo.DiagnosticError("RANSAC iterations and frame bin size must be positive")
    for label, value in (
        ("minimum triangulation angle", args.min_triangulation_angle_deg),
        ("RANSAC threshold", args.ransac_threshold_m),
        ("pose-consistent rotation error", args.pose_consistent_max_rotation_error_deg),
    ):
        if not math.isfinite(value) or value <= 0:
            raise stereo.DiagnosticError(f"{label} must be finite and positive")
    if not math.isfinite(args.pose_consistent_min_direction_cosine) or not (
        -1.0 <= args.pose_consistent_min_direction_cosine <= 1.0
    ):
        raise stereo.DiagnosticError(
            "pose-consistent direction cosine must be finite and in [-1, 1]"
        )
    result = diagnose(
        args.model,
        args.manifest,
        args.image_aliases,
        args.min_frame_gap,
        args.max_frame_gap,
        args.min_correspondences,
        args.min_inliers,
        args.min_triangulation_angle_deg,
        args.ransac_threshold_m,
        args.ransac_iterations,
        args.frame_bin_size,
        args.pose_consistent_max_rotation_error_deg,
        args.pose_consistent_min_direction_cosine,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
