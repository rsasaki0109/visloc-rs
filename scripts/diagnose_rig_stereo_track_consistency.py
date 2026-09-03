#!/usr/bin/env python3
"""Measure independent multi-frame stereo consistency in a COLMAP text model.

The diagnostic uses only calibrated cameras, reconstructed poses, and track
observations.  It never reads trajectory ground truth.  Every synchronized
multi-sensor observation independently triangulates a metric world point; a
track is self-consistent when those points agree across frames.  This is the
bounded geometric screen used to design the next rig-BA experiment without
mistaking repeated-scene support count for trustworthy metric structure.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class DiagnosticError(RuntimeError):
    """An input model or rig manifest is malformed."""


@dataclass(frozen=True)
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class Image:
    camera_id: int
    name: str
    center: np.ndarray
    camera_to_world: np.ndarray
    points2d: tuple[tuple[float, float, int], ...]


def rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if not math.isfinite(norm) or norm <= 0.0:
        raise DiagnosticError("COLMAP pose has a zero or non-finite quaternion")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def load_cameras(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) < 8:
            raise DiagnosticError(f"malformed camera row {path}:{line_number}")
        camera_id = int(fields[0])
        model = fields[1]
        parameters = [float(value) for value in fields[4:]]
        if model == "PINHOLE" and len(parameters) == 4:
            fx, fy, cx, cy = parameters
        elif model == "SIMPLE_PINHOLE" and len(parameters) == 3:
            fx, cx, cy = parameters
            fy = fx
        else:
            raise DiagnosticError(f"unsupported camera model {model!r} at {path}:{line_number}")
        if camera_id in cameras or not all(math.isfinite(value) and value > 0 for value in (fx, fy)):
            raise DiagnosticError(f"invalid or duplicate camera {camera_id} at {path}:{line_number}")
        cameras[camera_id] = Camera(fx, fy, cx, cy)
    if not cameras:
        raise DiagnosticError(f"no cameras in {path}")
    return cameras


def load_images(path: Path) -> dict[int, Image]:
    lines = path.read_text(encoding="utf-8").splitlines()
    images: dict[int, Image] = {}
    offset = 0
    while offset < len(lines):
        raw = lines[offset].strip()
        offset += 1
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split()
        if len(fields) < 10:
            raise DiagnosticError(f"malformed image pose row {path}:{offset}")
        if offset >= len(lines):
            raise DiagnosticError(f"missing POINTS2D row after {path}:{offset}")
        points_line = lines[offset].strip()
        offset += 1
        while points_line.startswith("#") and offset < len(lines):
            points_line = lines[offset].strip()
            offset += 1
        point_fields = points_line.split()
        if len(point_fields) % 3 != 0:
            raise DiagnosticError(f"malformed POINTS2D row after {path}:{offset - 1}")
        image_id = int(fields[0])
        world_to_camera = rotation(*(float(value) for value in fields[1:5]))
        translation = np.array([float(value) for value in fields[5:8]], dtype=float)
        camera_to_world = world_to_camera.T
        center = -(camera_to_world @ translation)
        points2d = tuple(
            (float(point_fields[index]), float(point_fields[index + 1]), int(point_fields[index + 2]))
            for index in range(0, len(point_fields), 3)
        )
        if image_id in images:
            raise DiagnosticError(f"duplicate image id {image_id} in {path}")
        images[image_id] = Image(int(fields[8]), fields[9], center, camera_to_world, points2d)
    if not images:
        raise DiagnosticError(f"no images in {path}")
    return images


def load_manifest(path: Path) -> dict[str, tuple[int, int]]:
    assignments: dict[str, tuple[int, int]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if not fields or fields[0].startswith("#") or fields[0] == "S":
            continue
        if fields[0] != "F" or len(fields) != 4:
            raise DiagnosticError(f"malformed manifest row {path}:{line_number}")
        name = fields[2]
        if name in assignments:
            raise DiagnosticError(f"duplicate image name {name!r} in {path}")
        assignments[name] = (int(fields[1]), int(fields[3]))
    if not assignments:
        raise DiagnosticError(f"no frame assignments in {path}")
    return assignments


def load_image_aliases(path: Path) -> dict[str, str]:
    """Load the frozen OpenLORIS ``flat_name -> COLMAP name`` TSV."""
    rows = path.read_text(encoding="utf-8").splitlines()
    if not rows or rows[0].split("\t") != ["flat_name", "colmap_name"]:
        raise DiagnosticError(f"unsupported image-alias header in {path}")
    aliases = {}
    for line_number, raw in enumerate(rows[1:], 2):
        fields = raw.split("\t")
        if len(fields) != 2 or not all(fields):
            raise DiagnosticError(f"malformed image alias row {path}:{line_number}")
        flat_name, colmap_name = fields
        if colmap_name in aliases:
            raise DiagnosticError(f"duplicate COLMAP image alias {colmap_name!r} in {path}")
        aliases[colmap_name] = flat_name
    return aliases


def load_tracks(path: Path) -> dict[int, tuple[np.ndarray, tuple[tuple[int, int], ...]]]:
    tracks = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) < 8 or (len(fields) - 8) % 2 != 0:
            raise DiagnosticError(f"malformed point row {path}:{line_number}")
        point_id = int(fields[0])
        observations = tuple(
            (int(fields[index]), int(fields[index + 1]))
            for index in range(8, len(fields), 2)
        )
        if point_id in tracks:
            raise DiagnosticError(f"duplicate point id {point_id} in {path}")
        tracks[point_id] = (np.array([float(value) for value in fields[1:4]]), observations)
    return tracks


def observation_ray(image: Image, camera: Camera, point_index: int) -> tuple[np.ndarray, np.ndarray]:
    if point_index < 0 or point_index >= len(image.points2d):
        raise DiagnosticError(f"POINT2D index {point_index} is outside image {image.name!r}")
    x, y, _ = image.points2d[point_index]
    camera_ray = np.array([(x - camera.cx) / camera.fx, (y - camera.cy) / camera.fy, 1.0])
    world_ray = image.camera_to_world @ camera_ray
    norm = float(np.linalg.norm(world_ray))
    if not math.isfinite(norm) or norm <= 0:
        raise DiagnosticError(f"invalid bearing in image {image.name!r}")
    return image.center, world_ray / norm


def triangulate(left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray], min_angle_deg: float):
    left_center, left_ray = left
    right_center, right_ray = right
    dot = float(np.clip(np.dot(left_ray, right_ray), -1.0, 1.0))
    angle = math.degrees(math.acos(dot))
    denominator = 1.0 - dot * dot
    if angle < min_angle_deg or denominator <= 1e-12:
        return None
    delta = left_center - right_center
    left_offset = float(np.dot(left_ray, delta))
    right_offset = float(np.dot(right_ray, delta))
    left_depth = (dot * right_offset - left_offset) / denominator
    right_depth = (right_offset - dot * left_offset) / denominator
    if left_depth <= 0.0 or right_depth <= 0.0:
        return None
    point = 0.5 * (
        left_center + left_depth * left_ray + right_center + right_depth * right_ray
    )
    return point if np.all(np.isfinite(point)) else None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def diagnose(
    model: Path,
    manifest_path: Path,
    thresholds: list[float],
    min_angle_deg: float,
    image_aliases_path: Path | None = None,
) -> dict:
    cameras = load_cameras(model / "cameras.txt")
    images = load_images(model / "images.txt")
    assignments = load_manifest(manifest_path)
    image_aliases = load_image_aliases(image_aliases_path) if image_aliases_path else {}
    tracks = load_tracks(model / "points3D.txt")
    per_threshold = {
        str(threshold): {"tracks": 0, "observations": 0} for threshold in thresholds
    }
    eligible = 0
    eligible_observations = 0
    stereo_frame_counts: list[float] = []
    max_deviations: list[float] = []
    landmark_medians: list[float] = []
    rejected_stereo_frames = 0

    for landmark, observations in tracks.values():
        by_frame: dict[int, list[tuple[int, tuple[np.ndarray, np.ndarray]]]] = defaultdict(list)
        for image_id, point_index in observations:
            image = images.get(image_id)
            if image is None:
                raise DiagnosticError(f"track references missing image id {image_id}")
            assignment_name = image_aliases.get(image.name, image.name)
            assignment = assignments.get(assignment_name)
            if assignment is None:
                raise DiagnosticError(
                    f"manifest has no assignment for image {image.name!r} (alias {assignment_name!r})"
                )
            camera = cameras.get(image.camera_id)
            if camera is None:
                raise DiagnosticError(f"image {image.name!r} references missing camera {image.camera_id}")
            frame, sensor = assignment
            by_frame[frame].append((sensor, observation_ray(image, camera, point_index)))

        stereo_points = []
        for frame_observations in by_frame.values():
            best = None
            for left_index, (left_sensor, left) in enumerate(frame_observations):
                for right_sensor, right in frame_observations[left_index + 1 :]:
                    if left_sensor == right_sensor:
                        continue
                    angle = math.degrees(math.acos(float(np.clip(np.dot(left[1], right[1]), -1, 1))))
                    if best is None or angle > best[0]:
                        best = (angle, left, right)
            if best is None:
                continue
            point = triangulate(best[1], best[2], min_angle_deg)
            if point is None:
                rejected_stereo_frames += 1
            else:
                stereo_points.append(point)
        if len(stereo_points) < 2:
            continue

        eligible += 1
        eligible_observations += len(observations)
        stereo_frame_counts.append(float(len(stereo_points)))
        robust_center = np.median(np.stack(stereo_points), axis=0)
        deviations = [float(np.linalg.norm(point - robust_center)) for point in stereo_points]
        landmark_errors = [float(np.linalg.norm(point - landmark)) for point in stereo_points]
        max_deviation = max(deviations)
        max_deviations.append(max_deviation)
        landmark_medians.append(float(np.median(landmark_errors)))
        for threshold in thresholds:
            if max_deviation <= threshold:
                bucket = per_threshold[str(threshold)]
                bucket["tracks"] += 1
                bucket["observations"] += len(observations)

    for bucket in per_threshold.values():
        bucket["track_fraction"] = bucket["tracks"] / eligible if eligible else 0.0
        bucket["observation_fraction"] = (
            bucket["observations"] / eligible_observations if eligible_observations else 0.0
        )
    return {
        "schema": "visloc_rig_stereo_track_consistency_v1",
        "ground_truth_used": False,
        "model": str(model),
        "manifest": str(manifest_path),
        "image_aliases": str(image_aliases_path) if image_aliases_path else None,
        "registered_images": len(images),
        "tracks": len(tracks),
        "tracks_with_two_stereo_frames": eligible,
        "observations_on_eligible_tracks": eligible_observations,
        "rejected_stereo_frames": rejected_stereo_frames,
        "min_triangulation_angle_deg": min_angle_deg,
        "stereo_frames_per_eligible_track": {
            "median": percentile(stereo_frame_counts, 0.5),
            "p95": percentile(stereo_frame_counts, 0.95),
            "max": max(stereo_frame_counts, default=None),
        },
        "max_cross_frame_deviation_m": {
            "median": percentile(max_deviations, 0.5),
            "p95": percentile(max_deviations, 0.95),
            "max": max(max_deviations, default=None),
        },
        "median_stereo_to_final_landmark_error_m": {
            "median": percentile(landmark_medians, 0.5),
            "p95": percentile(landmark_medians, 0.95),
            "max": max(landmark_medians, default=None),
        },
        "thresholds_by_max_cross_frame_deviation_m": per_threshold,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-aliases", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-triangulation-angle-deg", type=float, default=1.0)
    parser.add_argument("--threshold-m", type=float, action="append")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = args.threshold_m or [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
    if any(not math.isfinite(value) or value <= 0 for value in thresholds):
        raise DiagnosticError("--threshold-m values must be finite and positive")
    if not math.isfinite(args.min_triangulation_angle_deg) or args.min_triangulation_angle_deg <= 0:
        raise DiagnosticError("--min-triangulation-angle-deg must be finite and positive")
    result = diagnose(
        args.model,
        args.manifest,
        sorted(set(thresholds)),
        args.min_triangulation_angle_deg,
        args.image_aliases,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
