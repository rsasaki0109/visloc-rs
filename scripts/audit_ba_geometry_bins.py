#!/usr/bin/env python3
"""Bounded, read-only observation-geometry comparison for two COLMAP models.

The first model defines the track populations.  The second model is one
already-produced candidate; this program never runs an optimizer and never
writes a model.  It intentionally reports descriptive residual and motion
associations, not causal attribution.

Inputs are required to be stable regular files.  The bounded preflight and the
post-traversal hash check reject observed mutation, but this tool does not try
to provide a race-safe filesystem snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import stat
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MAX_CAMERAS = 16
MAX_IMAGES = 1024
MAX_POINTS = 8192
MAX_OBSERVATIONS = 262_144
MAX_TOTAL_KEYPOINTS = 1_000_000
MAX_TRACK = 1024
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_MODEL_BYTES = 64 * 1024 * 1024
MAX_LINE_BYTES = 2 * 1024 * 1024
MAX_CROSS_WORK = 64_000_000

# These are the frozen 1k input counts.  The command-line entry point requires
# both models to have these counts; the library-style ``diagnose`` helper keeps
# the check optional so bounded synthetic unit fixtures remain possible.
EXPECTED_FROZEN_COUNTS = {
    "observations": 130_900,
    "w2": 28_705_634,
    "pairs": 14_287_367,
}

K_BINS = (
    ("k_2_3", 2, 3),
    ("k_4_8", 4, 8),
    ("k_9_16", 9, 16),
    ("k_17_plus", 17, MAX_TRACK),
)
ANGLE_BINS = (
    ("angle_0_0.1_deg", 0.0, 0.1),
    ("angle_0.1_1_deg", 0.1, 1.0),
    ("angle_1_5_deg", 1.0, 5.0),
    ("angle_5_90_deg", 5.0, 90.0),
)


class DiagnosticError(ValueError):
    """A fail-closed input, identity, geometry, or bounded-resource error."""


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ROOT = Path(__file__).resolve().parents[1]
_PUBLISH = _load_module(
    "publish_ceres_rig_reference_for_geometry_audit",
    _ROOT / "tools" / "ceres_rig_reference" / "publish_model.py",
)
_AUDIT = _load_module(
    "audit_colmap_pinhole_for_geometry_audit",
    _ROOT / "scripts" / "audit_colmap_pinhole_model.py",
)


def _float_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def _finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise DiagnosticError(f"{label} is non-finite")
    return value


def _close(left: float, right: float) -> bool:
    return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= (
        1.0e-7 + 1.0e-12 * max(abs(left), abs(right))
    )


def _regular(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise DiagnosticError(f"{label} is unavailable: {path}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise DiagnosticError(f"{label} must be a regular non-symlink file: {path}")
    return metadata


def _regular_dir(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise DiagnosticError(f"{label} is unavailable: {path}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise DiagnosticError(f"{label} must be a regular non-symlink directory: {path}")


def _preflight_file(path: Path, label: str) -> tuple[str, int]:
    metadata = _regular(path, label)
    if metadata.st_size > MAX_FILE_BYTES:
        raise DiagnosticError(f"{label} exceeds {MAX_FILE_BYTES} bytes")
    digest = hashlib.sha256()
    actual_size = 0
    try:
        with path.open("rb") as stream:
            line_number = 0
            while True:
                line = stream.readline(MAX_LINE_BYTES + 1)
                if not line:
                    break
                line_number += 1
                if len(line) > MAX_LINE_BYTES:
                    raise DiagnosticError(
                        f"{label} line {line_number} exceeds {MAX_LINE_BYTES} bytes"
                    )
                actual_size += len(line)
                if actual_size > MAX_FILE_BYTES:
                    raise DiagnosticError(f"{label} exceeds {MAX_FILE_BYTES} bytes")
                digest.update(line)
    except DiagnosticError:
        raise
    except OSError as error:
        raise DiagnosticError(f"cannot read {label}: {error}") from error
    if actual_size != metadata.st_size:
        raise DiagnosticError(f"{label} changed during preflight")
    return digest.hexdigest(), metadata.st_size


def _bounded_readline(stream, label: str, line_number: int) -> bytes:
    line = stream.readline(MAX_LINE_BYTES + 1)
    if line and len(line) > MAX_LINE_BYTES:
        raise DiagnosticError(f"{label} line {line_number} exceeds {MAX_LINE_BYTES} bytes")
    return line


def _parse_int(token: bytes, label: str) -> int:
    try:
        value = int(token, 10)
    except ValueError as error:
        raise DiagnosticError(f"{label} is not an integer") from error
    if value < 0:
        raise DiagnosticError(f"{label} is negative")
    return value


def _scan_cameras(path: Path) -> int:
    count = 0
    ids: set[int] = set()
    bytes_read = 0
    with path.open("rb") as stream:
        line_number = 0
        while True:
            line_number += 1
            raw = _bounded_readline(stream, "cameras.txt", line_number)
            if not raw:
                break
            bytes_read += len(raw)
            if bytes_read > MAX_FILE_BYTES:
                raise DiagnosticError("cameras.txt exceeds bounded size during scan")
            if not raw.strip() or raw.lstrip().startswith(b"#"):
                continue
            fields = raw.split()
            if len(fields) != 8 or fields[1] != b"PINHOLE":
                raise DiagnosticError(f"cameras.txt line {line_number} is not PINHOLE")
            camera_id = _parse_int(fields[0], "camera id")
            if camera_id in ids:
                raise DiagnosticError("duplicate camera id")
            ids.add(camera_id)
            count += 1
            if count > MAX_CAMERAS:
                raise DiagnosticError("camera cap exceeded before parser reuse")
    if count == 0:
        raise DiagnosticError("cameras.txt is empty")
    return count


def _scan_images(path: Path) -> tuple[int, int]:
    image_count = 0
    keypoint_count = 0
    names: set[bytes] = set()
    ids: set[int] = set()
    bytes_read = 0
    with path.open("rb") as stream:
        line_number = 0
        while True:
            line_number += 1
            raw = _bounded_readline(stream, "images.txt", line_number)
            if not raw:
                break
            bytes_read += len(raw)
            if bytes_read > MAX_FILE_BYTES:
                raise DiagnosticError("images.txt exceeds bounded size during scan")
            if not raw.strip() or raw.lstrip().startswith(b"#"):
                continue
            fields = raw.split()
            if len(fields) != 10:
                raise DiagnosticError("malformed image header before parser reuse")
            image_id = _parse_int(fields[0], "image id")
            if image_id in ids or fields[9] in names:
                raise DiagnosticError("duplicate image id or name")
            ids.add(image_id)
            names.add(fields[9])
            line_number += 1
            points_line = _bounded_readline(stream, "images.txt", line_number)
            if not points_line:
                raise DiagnosticError("image header has no POINTS2D line")
            bytes_read += len(points_line)
            if bytes_read > MAX_FILE_BYTES:
                raise DiagnosticError("images.txt exceeds bounded size during scan")
            if points_line.lstrip().startswith(b"#"):
                raise DiagnosticError("POINTS2D line is a comment")
            point_fields = points_line.split()
            if len(point_fields) % 3:
                raise DiagnosticError("malformed POINTS2D line")
            keypoints = len(point_fields) // 3
            keypoint_count += keypoints
            if keypoint_count > MAX_TOTAL_KEYPOINTS:
                raise DiagnosticError("total keypoint cap exceeded")
            image_count += 1
            if image_count > MAX_IMAGES:
                raise DiagnosticError("image cap exceeded before parser reuse")
    if image_count == 0:
        raise DiagnosticError("images.txt is empty")
    return image_count, keypoint_count


def _scan_points(path: Path) -> tuple[int, int, int, int, int]:
    point_count = 0
    observations = 0
    sum_k_squared = 0
    pair_count = 0
    max_track = 0
    ids: set[int] = set()
    bytes_read = 0
    with path.open("rb") as stream:
        line_number = 0
        while True:
            line_number += 1
            raw = _bounded_readline(stream, "points3D.txt", line_number)
            if not raw:
                break
            bytes_read += len(raw)
            if bytes_read > MAX_FILE_BYTES:
                raise DiagnosticError("points3D.txt exceeds bounded size during scan")
            if not raw.strip() or raw.lstrip().startswith(b"#"):
                continue
            fields = raw.split()
            if len(fields) < 12 or (len(fields) - 8) % 2:
                raise DiagnosticError(f"malformed points3D line {line_number}")
            point_id = _parse_int(fields[0], "point id")
            if point_id in ids:
                raise DiagnosticError("duplicate point id")
            ids.add(point_id)
            k = (len(fields) - 8) // 2
            if k < 2 or k > MAX_TRACK:
                raise DiagnosticError(f"track length {k} is outside 2..{MAX_TRACK}")
            point_count += 1
            observations += k
            if point_count > MAX_POINTS:
                raise DiagnosticError("landmark cap exceeded before parser reuse")
            if observations > MAX_OBSERVATIONS:
                raise DiagnosticError("observation cap exceeded before parser reuse")
            term = k * k
            sum_k_squared += term
            pair_count += k * (k - 1) // 2
            if sum_k_squared > MAX_CROSS_WORK:
                raise DiagnosticError("W2 cross-pair cap exceeded before pair traversal")
            max_track = max(max_track, k)
    if point_count == 0:
        raise DiagnosticError("points3D.txt is empty")
    return point_count, observations, sum_k_squared, pair_count, max_track


@dataclass(frozen=True)
class Preflight:
    file_hashes: dict[str, str]
    file_sizes: dict[str, int]
    images: int
    keypoints: int
    points: int
    observations: int
    sum_k_squared: int
    pair_count: int
    max_track: int


def _preflight_model(path: Path) -> Preflight:
    _regular_dir(path, "model")
    required = {
        "cameras.txt": path / "cameras.txt",
        "images.txt": path / "images.txt",
        "points3D.txt": path / "points3D.txt",
    }
    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for name, file_path in required.items():
        hashes[name], sizes[name] = _preflight_file(file_path, name)
    if sum(sizes.values()) > MAX_MODEL_BYTES:
        raise DiagnosticError(f"model exceeds {MAX_MODEL_BYTES} bytes")
    cameras = _scan_cameras(required["cameras.txt"])
    images, keypoints = _scan_images(required["images.txt"])
    points, observations, work, pairs, max_track = _scan_points(required["points3D.txt"])
    if cameras > MAX_CAMERAS:
        raise DiagnosticError("camera cap exceeded")
    return Preflight(
        hashes,
        sizes,
        images,
        keypoints,
        points,
        observations,
        work,
        pairs,
        max_track,
    )


def _verify_preflight(path: Path, preflight: Preflight) -> None:
    current = _preflight_model(path)
    if current != preflight:
        raise DiagnosticError(f"model changed during audit: {path}")


def _matrix_from_image(image) -> tuple[tuple[float, float, float], ...]:
    try:
        rows = _AUDIT.rotation(image.q)
    except (TypeError, ValueError) as error:
        raise DiagnosticError(f"image {image.image_id} has invalid quaternion") from error
    return tuple(tuple(float(value) for value in row) for row in rows)


def _mat_vec(matrix, vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(
        sum(matrix[row][column] * vector[column] for column in range(3))
        for row in range(3)
    )


def _camera_center(image, matrix) -> tuple[float, float, float]:
    return tuple(
        -sum(matrix[row][column] * image.t[row] for row in range(3))
        for column in range(3)
    )


def _project(image, camera, point, matrix) -> tuple[float, float, float]:
    if camera.model != "PINHOLE" or len(camera.params) != 4:
        raise DiagnosticError("only four-parameter PINHOLE cameras are supported")
    camera_point = _mat_vec(matrix, point)
    camera_point = tuple(camera_point[i] + image.t[i] for i in range(3))
    if not all(math.isfinite(value) for value in camera_point) or camera_point[2] <= 0.0:
        raise DiagnosticError(
            f"image {image.image_id} has nonpositive or nonfinite candidate depth"
        )
    fx, fy, cx, cy = camera.params
    x = fx * camera_point[0] / camera_point[2] + cx
    y = fy * camera_point[1] / camera_point[2] + cy
    if not math.isfinite(x) or not math.isfinite(y):
        raise DiagnosticError(f"image {image.image_id} has nonfinite reprojection")
    return x, y, camera_point[2]


def _identity_check(initial, candidate) -> None:
    if initial.cameras_bytes != candidate.cameras_bytes:
        raise DiagnosticError("camera bytes changed")
    if len(initial.images) != len(candidate.images) or len(initial.points) != len(candidate.points):
        raise DiagnosticError("model record counts changed")
    for left, right in zip(initial.images, candidate.images):
        if (left.image_id, left.camera_id, left.name) != (
            right.image_id,
            right.camera_id,
            right.name,
        ):
            raise DiagnosticError("image ID/order/name/camera assignment changed")
        if len(left.keypoints) != len(right.keypoints):
            raise DiagnosticError(f"image {left.image_id} keypoint count changed")
        for left_keypoint, right_keypoint in zip(left.keypoints, right.keypoints):
            if (
                _float_bits(left_keypoint.x) != _float_bits(right_keypoint.x)
                or _float_bits(left_keypoint.y) != _float_bits(right_keypoint.y)
                or left_keypoint.point_id != right_keypoint.point_id
            ):
                raise DiagnosticError(f"image {left.image_id} keypoint association changed")
    for left, right in zip(initial.points, candidate.points):
        if (left.point_id, left.rgb, left.track) != (right.point_id, right.rgb, right.track):
            raise DiagnosticError(f"point {left.point_id} RGB/track identity changed")


def _track_angle(point, centers) -> float:
    rays = []
    for center in centers:
        ray = tuple(point[index] - center[index] for index in range(3))
        norm = math.sqrt(sum(value * value for value in ray))
        if not math.isfinite(norm) or norm == 0.0:
            raise DiagnosticError("zero-length or nonfinite initial geometry ray")
        rays.append(tuple(value / norm for value in ray))
    maximum = 0.0
    for index in range(len(rays)):
        for other in range(index):
            cosine = sum(rays[index][axis] * rays[other][axis] for axis in range(3))
            cosine = max(-1.0, min(1.0, cosine))
            theta = math.acos(cosine)
            acute = min(theta, math.pi - theta)
            if not math.isfinite(acute):
                raise DiagnosticError("nonfinite initial pair angle")
            maximum = max(maximum, acute)
    return math.degrees(maximum)


def _bin_index(k: int, angle_degrees: float) -> tuple[int, int]:
    for index, (_, lower, upper) in enumerate(K_BINS):
        if lower <= k <= upper:
            k_index = index
            break
    else:
        raise DiagnosticError(f"track length {k} has no fixed stratum")
    for index, (_, lower, upper) in enumerate(ANGLE_BINS):
        if (lower <= angle_degrees < upper) or (
            index == len(ANGLE_BINS) - 1 and lower <= angle_degrees <= upper
        ):
            return k_index, index
    raise DiagnosticError(f"initial angle {angle_degrees} is outside [0,90]")


def _new_cell(k_index: int, angle_index: int) -> dict:
    return {
        "k_stratum": K_BINS[k_index][0],
        "angle_bin_deg": ANGLE_BINS[angle_index][0],
        "points": 0,
        "observations": 0,
        "initial_squared_cost": 0.0,
        "candidate_squared_cost": 0.0,
        "cost_reduction": 0.0,
        "cost_increase": 0.0,
        "initial_reprojection_sum_px": 0.0,
        "candidate_reprojection_sum_px": 0.0,
        "initial_reprojection_max_px": 0.0,
        "candidate_reprojection_max_px": 0.0,
        "landmark_displacement_sum_m": 0.0,
        "landmark_displacement_max_m": 0.0,
        "camera_center_motion_observation_sum_m": 0.0,
        "camera_center_motion_observation_max_m": 0.0,
    }


_AGGREGATE_FLOAT_FIELDS = (
    "initial_squared_cost",
    "candidate_squared_cost",
    "cost_reduction",
    "cost_increase",
    "initial_reprojection_sum_px",
    "candidate_reprojection_sum_px",
    "initial_reprojection_max_px",
    "candidate_reprojection_max_px",
    "landmark_displacement_sum_m",
    "landmark_displacement_max_m",
    "camera_center_motion_observation_sum_m",
    "camera_center_motion_observation_max_m",
)


def _new_totals() -> dict[str, float | int]:
    return {
        "points": 0,
        "observations": 0,
        **{field: 0.0 for field in _AGGREGATE_FLOAT_FIELDS},
    }


def _empty_cells() -> dict[tuple[int, int], dict]:
    return {
        (k_index, angle_index): _new_cell(k_index, angle_index)
        for k_index in range(len(K_BINS))
        for angle_index in range(len(ANGLE_BINS))
    }


def _mean(total: float, count: int) -> float | None:
    return total / count if count else None


def _safe_fsum(values: Iterable[float], label: str) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as error:
        raise DiagnosticError(f"{label} overflowed") from error
    return _finite(result, label)


def _cell_report(cell: dict) -> dict:
    result = dict(cell)
    result.update(
        {
            "initial_reprojection_mean_px": _mean(
                cell["initial_reprojection_sum_px"], cell["observations"]
            ),
            "candidate_reprojection_mean_px": _mean(
                cell["candidate_reprojection_sum_px"], cell["observations"]
            ),
            "landmark_displacement_mean_m": _mean(
                cell["landmark_displacement_sum_m"], cell["points"]
            ),
            "camera_center_motion_observation_mean_m": _mean(
                cell["camera_center_motion_observation_sum_m"], cell["observations"]
            ),
        }
    )
    return result


def _add_cost(cell: dict, initial_sq: float, candidate_sq: float) -> None:
    delta = initial_sq - candidate_sq
    if not math.isfinite(delta):
        raise DiagnosticError("nonfinite observation cost difference")
    cell["initial_squared_cost"] += initial_sq
    cell["candidate_squared_cost"] += candidate_sq
    if delta >= 0.0:
        cell["cost_reduction"] += delta
    else:
        cell["cost_increase"] -= delta


def _aggregate_totals(cells: Iterable[dict]) -> dict[str, float | int]:
    cells = list(cells)
    result: dict[str, float | int] = {
        "points": sum(cell["points"] for cell in cells),
        "observations": sum(cell["observations"] for cell in cells),
        "initial_squared_cost": _safe_fsum(
            (cell["initial_squared_cost"] for cell in cells),
            "aggregate initial squared cost",
        ),
        "candidate_squared_cost": _safe_fsum(
            (cell["candidate_squared_cost"] for cell in cells),
            "aggregate candidate squared cost",
        ),
        "cost_reduction": _safe_fsum(
            (cell["cost_reduction"] for cell in cells),
            "aggregate cost reduction",
        ),
        "cost_increase": _safe_fsum(
            (cell["cost_increase"] for cell in cells),
            "aggregate cost increase",
        ),
        "initial_reprojection_sum_px": _safe_fsum(
            (cell["initial_reprojection_sum_px"] for cell in cells),
            "aggregate initial reprojection",
        ),
        "candidate_reprojection_sum_px": _safe_fsum(
            (cell["candidate_reprojection_sum_px"] for cell in cells),
            "aggregate candidate reprojection",
        ),
        "initial_reprojection_max_px": max(
            (cell["initial_reprojection_max_px"] for cell in cells), default=0.0
        ),
        "candidate_reprojection_max_px": max(
            (cell["candidate_reprojection_max_px"] for cell in cells), default=0.0
        ),
        "landmark_displacement_sum_m": _safe_fsum(
            (cell["landmark_displacement_sum_m"] for cell in cells),
            "aggregate landmark displacement",
        ),
        "landmark_displacement_max_m": max(
            (cell["landmark_displacement_max_m"] for cell in cells), default=0.0
        ),
        "camera_center_motion_observation_sum_m": _safe_fsum(
            (cell["camera_center_motion_observation_sum_m"] for cell in cells),
            "aggregate camera-centre motion",
        ),
        "camera_center_motion_observation_max_m": max(
            (cell["camera_center_motion_observation_max_m"] for cell in cells),
            default=0.0,
        ),
    }
    for key, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise DiagnosticError(f"aggregate {key} is non-finite")
    return result


def _reconcile(
    direct: dict[str, float | int],
    cells: dict[tuple[int, int], dict],
    expected_points: int,
    expected_observations: int,
) -> dict[str, float | int]:
    """Check an independently accumulated traversal against all 16 cells."""
    totals = _aggregate_totals(cells.values())
    if direct["points"] != expected_points or direct["observations"] != expected_observations:
        raise DiagnosticError("traversal counts do not match bounded preflight")
    for field in ("points", "observations"):
        if totals[field] != direct[field]:
            raise DiagnosticError(f"{field} bins do not reconstruct")
    for field in _AGGREGATE_FLOAT_FIELDS:
        if not _close(float(totals[field]), float(direct[field])):
            raise DiagnosticError(f"{field} bins do not reconstruct")
    if not _close(
        float(direct["initial_squared_cost"]) - float(direct["candidate_squared_cost"]),
        float(direct["cost_reduction"]) - float(direct["cost_increase"]),
    ):
        raise DiagnosticError("global cost increase/reduction does not reconcile")
    for cell in cells.values():
        if not _close(
            cell["initial_squared_cost"] - cell["candidate_squared_cost"],
            cell["cost_reduction"] - cell["cost_increase"],
        ):
            raise DiagnosticError("cell cost increase/reduction does not reconcile")
        for value in cell.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise DiagnosticError("cell aggregate is non-finite")
    for value in direct.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise DiagnosticError("global aggregate is non-finite")
    if totals["cost_reduction"] < 0.0 or totals["cost_increase"] < 0.0:
        raise DiagnosticError("cost increase/reduction is negative")
    return totals


def _diagnose(initial, candidate, candidate_label: str, initial_preflight: Preflight, candidate_preflight: Preflight) -> dict:
    _identity_check(initial, candidate)
    if initial_preflight.observations != candidate_preflight.observations:
        raise DiagnosticError("initial/candidate observation counts differ")
    if initial_preflight.points != candidate_preflight.points:
        raise DiagnosticError("initial/candidate landmark counts differ")

    initial_images = {image.image_id: image for image in initial.images}
    candidate_images = {image.image_id: image for image in candidate.images}
    candidate_points = {point.point_id: point for point in candidate.points}
    initial_matrices = {image_id: _matrix_from_image(image) for image_id, image in initial_images.items()}
    candidate_matrices = {image_id: _matrix_from_image(image) for image_id, image in candidate_images.items()}
    initial_centers = {
        image_id: _camera_center(image, initial_matrices[image_id])
        for image_id, image in initial_images.items()
    }
    candidate_centers = {
        image_id: _camera_center(image, candidate_matrices[image_id])
        for image_id, image in candidate_images.items()
    }
    image_motion = {}
    for image_id in initial_images:
        motion = math.dist(initial_centers[image_id], candidate_centers[image_id])
        if not math.isfinite(motion):
            raise DiagnosticError("nonfinite camera-centre displacement")
        image_motion[image_id] = motion

    cells = _empty_cells()
    direct = _new_totals()
    for point in initial.points:
        candidate_point = candidate_points[point.point_id]
        centers = [initial_centers[image_id] for image_id, _ in point.track]
        angle = _track_angle(point.xyz, centers)
        k_index, angle_index = _bin_index(len(point.track), angle)
        cell = cells[(k_index, angle_index)]
        cell["points"] += 1
        direct["points"] += 1
        displacement = math.dist(point.xyz, candidate_point.xyz)
        if not math.isfinite(displacement):
            raise DiagnosticError(f"point {point.point_id} displacement is nonfinite")
        cell["landmark_displacement_sum_m"] += displacement
        cell["landmark_displacement_max_m"] = max(
            cell["landmark_displacement_max_m"], displacement
        )
        direct["landmark_displacement_sum_m"] += displacement
        direct["landmark_displacement_max_m"] = max(
            direct["landmark_displacement_max_m"], displacement
        )
        for image_id, keypoint_index in point.track:
            initial_image = initial_images[image_id]
            candidate_image = candidate_images[image_id]
            initial_keypoint = initial_image.keypoints[keypoint_index]
            candidate_keypoint = candidate_image.keypoints[keypoint_index]
            if (
                _float_bits(initial_keypoint.x) != _float_bits(candidate_keypoint.x)
                or _float_bits(initial_keypoint.y) != _float_bits(candidate_keypoint.y)
            ):
                raise DiagnosticError("candidate keypoint coordinate changed")
            initial_camera = initial.cameras[initial_image.camera_id]
            candidate_camera = candidate.cameras[candidate_image.camera_id]
            initial_x, initial_y, _ = _project(
                initial_image, initial_camera, point.xyz, initial_matrices[image_id]
            )
            candidate_x, candidate_y, _ = _project(
                candidate_image,
                candidate_camera,
                candidate_point.xyz,
                candidate_matrices[image_id],
            )
            initial_error = math.hypot(initial_x - initial_keypoint.x, initial_y - initial_keypoint.y)
            candidate_error = math.hypot(candidate_x - candidate_keypoint.x, candidate_y - candidate_keypoint.y)
            initial_sq = (initial_x - initial_keypoint.x) ** 2 + (initial_y - initial_keypoint.y) ** 2
            candidate_sq = (candidate_x - candidate_keypoint.x) ** 2 + (candidate_y - candidate_keypoint.y) ** 2
            for value in (initial_error, candidate_error, initial_sq, candidate_sq):
                _finite(value, "reprojection metric")
            cell["observations"] += 1
            _add_cost(cell, initial_sq, candidate_sq)
            cell["initial_reprojection_sum_px"] += initial_error
            cell["candidate_reprojection_sum_px"] += candidate_error
            cell["initial_reprojection_max_px"] = max(cell["initial_reprojection_max_px"], initial_error)
            cell["candidate_reprojection_max_px"] = max(cell["candidate_reprojection_max_px"], candidate_error)
            cell["camera_center_motion_observation_sum_m"] += image_motion[image_id]
            cell["camera_center_motion_observation_max_m"] = max(
                cell["camera_center_motion_observation_max_m"], image_motion[image_id]
            )
            direct["observations"] += 1
            direct["initial_squared_cost"] += initial_sq
            direct["candidate_squared_cost"] += candidate_sq
            delta = initial_sq - candidate_sq
            if delta >= 0.0:
                direct["cost_reduction"] += delta
            else:
                direct["cost_increase"] -= delta
            direct["initial_reprojection_sum_px"] += initial_error
            direct["candidate_reprojection_sum_px"] += candidate_error
            direct["initial_reprojection_max_px"] = max(
                direct["initial_reprojection_max_px"], initial_error
            )
            direct["candidate_reprojection_max_px"] = max(
                direct["candidate_reprojection_max_px"], candidate_error
            )
            direct["camera_center_motion_observation_sum_m"] += image_motion[image_id]
            direct["camera_center_motion_observation_max_m"] = max(
                direct["camera_center_motion_observation_max_m"], image_motion[image_id]
            )

    totals = _reconcile(
        direct,
        cells,
        initial_preflight.points,
        initial_preflight.observations,
    )

    image_motion_sum = _safe_fsum(
        image_motion.values(), "global image-weighted camera-centre motion"
    )
    result = {
        "schema": "m8_observation_geometry_diagnostic_v1",
        "candidate_label": candidate_label,
        "membership": {
            "initial_only": True,
            "all_anchor_observations": True,
            "angle_definition": "max_pair_min(theta, pi-theta)",
            "angle_bins_deg": [name for name, _, _ in ANGLE_BINS],
            "track_length_bins": [name for name, _, _ in K_BINS],
        },
        "caps": {
            "cameras": MAX_CAMERAS,
            "images": MAX_IMAGES,
            "points": MAX_POINTS,
            "observations": MAX_OBSERVATIONS,
            "total_keypoints": MAX_TOTAL_KEYPOINTS,
            "track_length": MAX_TRACK,
            "file_bytes": MAX_FILE_BYTES,
            "model_bytes": MAX_MODEL_BYTES,
            "line_bytes": MAX_LINE_BYTES,
            "w2": MAX_CROSS_WORK,
        },
        "initial_file_hashes": initial_preflight.file_hashes,
        "candidate_file_hashes": candidate_preflight.file_hashes,
        "initial_counts": {
            "images": initial_preflight.images,
            "keypoints": initial_preflight.keypoints,
            "points": initial_preflight.points,
            "observations": initial_preflight.observations,
            "sum_track_lengths": initial_preflight.observations,
            "w2": initial_preflight.sum_k_squared,
            "pairs": initial_preflight.pair_count,
            "max_track": initial_preflight.max_track,
        },
        "candidate_counts": {
            "images": candidate_preflight.images,
            "keypoints": candidate_preflight.keypoints,
            "points": candidate_preflight.points,
            "observations": candidate_preflight.observations,
            "sum_track_lengths": candidate_preflight.observations,
            "w2": candidate_preflight.sum_k_squared,
            "pairs": candidate_preflight.pair_count,
            "max_track": candidate_preflight.max_track,
        },
        "cells": [
            _cell_report(cells[(k_index, angle_index)])
            for k_index in range(len(K_BINS))
            for angle_index in range(len(ANGLE_BINS))
        ],
        "global": {
            **totals,
            "initial_reprojection_mean_px": _mean(
                totals["initial_reprojection_sum_px"], totals["observations"]
            ),
            "candidate_reprojection_mean_px": _mean(
                totals["candidate_reprojection_sum_px"], totals["observations"]
            ),
            "landmark_displacement_mean_m": _mean(
                totals["landmark_displacement_sum_m"], totals["points"]
            ),
            "camera_center_motion_observation_mean_m": _mean(
                totals["camera_center_motion_observation_sum_m"], totals["observations"]
            ),
            "image_weighted_camera_center_motion": {
                "images": len(image_motion),
                "sum_m": image_motion_sum,
                "mean_m": _mean(image_motion_sum, len(image_motion)),
                "max_m": max(image_motion.values(), default=0.0),
            },
            "causal_attribution": False,
        },
        "input_integrity": {
            "stable_regular_files_required": True,
            "preflight_and_post_hash_checked": True,
            "race_safe_snapshot": False,
            "rig_and_calibration_recertified": False,
            "camera_motion_is_serialized_camera_centre": True,
        },
    }
    return result


def _runtime_metadata() -> dict[str, int | float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = int(usage.ru_maxrss)
    if sys.platform == "darwin":
        rss //= 1024
    return {"peak_rss_kib": rss, "peak_rss_target_kib": 524_288}


def _require_frozen_counts(preflight: Preflight, label: str) -> None:
    actual = {
        "observations": preflight.observations,
        "w2": preflight.sum_k_squared,
        "pairs": preflight.pair_count,
    }
    if actual != EXPECTED_FROZEN_COUNTS:
        raise DiagnosticError(
            f"{label} does not match the frozen input counts: "
            f"expected {EXPECTED_FROZEN_COUNTS}, got {actual}"
        )


def diagnose(
    initial_path: Path,
    candidate_path: Path,
    candidate_label: str,
    include_runtime: bool = False,
    require_frozen: bool = False,
) -> dict:
    started = time.monotonic()
    initial_preflight = _preflight_model(initial_path)
    candidate_preflight = _preflight_model(candidate_path)
    if require_frozen:
        _require_frozen_counts(initial_preflight, "initial model")
        _require_frozen_counts(candidate_preflight, "candidate model")
    try:
        # The diagnostic's input contract is stable regular files.  Recheck
        # immediately before parser reuse, and again after traversal; the
        # latter catches any mutation without permitting an output report.
        _verify_preflight(initial_path, initial_preflight)
        _verify_preflight(candidate_path, candidate_preflight)
        initial = _PUBLISH.parse_model(initial_path)
        candidate = _PUBLISH.parse_model(candidate_path)
        report = _diagnose(
            initial,
            candidate,
            candidate_label,
            initial_preflight,
            candidate_preflight,
        )
    except (_PUBLISH.PublisherError, OSError, ValueError, IndexError) as error:
        raise DiagnosticError(str(error)) from error
    _verify_preflight(initial_path, initial_preflight)
    _verify_preflight(candidate_path, candidate_preflight)
    report["frozen_input_counts"] = {
        "expected": EXPECTED_FROZEN_COUNTS,
        "verified": require_frozen,
    }
    if include_runtime:
        report["runtime"] = {
            **_runtime_metadata(),
            "wall_seconds": time.monotonic() - started,
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--candidate-label", required=True)
    args = parser.parse_args(argv)
    try:
        report = diagnose(
            args.initial_model,
            args.candidate_model,
            args.candidate_label,
            include_runtime=True,
            require_frozen=True,
        )
    except (DiagnosticError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
