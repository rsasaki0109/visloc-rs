#!/usr/bin/env python3
"""Estimate GT-free metric rig-motion factors from accepted quadrilaterals.

The input is the four-observation TSV emitted by
``diagnose_rig_photometric_quadrilaterals.py``.  For every frame pair, each
quadrilateral is triangulated independently in the source rig frame using its
two calibrated sensors.  The resulting metric points are then sent to
OpenCV's bounded ``solvePnPRansac`` independently for every sensor in the
target frame.  A sensor PnP pose is converted back to the rig frame using the
fixed manifest extrinsics.  The reverse direction repeats the whole
triangulation/PnP procedure, so no reconstructed pose or ground truth enters
the estimate.

The JSON is deliberately a diagnostic rather than a mapper input.  It keeps
per-direction and per-sensor support/residuals, sensor-to-sensor disagreement,
and forward versus inverse-reverse consistency.  PnP hypotheses use a stable
OpenCV RNG seed derived from frame, direction, and sensor, and the number of
iterations and tracks per pair are bounded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - exercised only without OpenCV
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import diagnose_rig_stereo_track_consistency as stereo  # noqa: E402


class DiagnosticError(ValueError):
    """Raised when the accepted-track TSV or rig manifest is malformed."""


@dataclass(frozen=True)
class Sensor:
    camera: stereo.Camera
    sensor_from_rig_rotation: np.ndarray
    sensor_from_rig_translation: np.ndarray


@dataclass(frozen=True)
class Observation:
    image: int
    keypoint: int
    frame: int
    sensor: int
    name: str
    point: np.ndarray


@dataclass(frozen=True)
class Quadrilateral:
    track: int
    frame_i: int
    frame_j: int
    observations: dict[tuple[int, int], Observation]


def _finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise DiagnosticError(f"{label} is not finite")
    return value


def load_sensors(path: Path) -> dict[int, Sensor]:
    """Read ``S`` rows from the generalized-rig manifest.

    The manifest stores ``sensor_from_rig`` as a quaternion and translation,
    with camera parameters in the four fields immediately before it.  ``F``
    rows are intentionally ignored here; they are optional for this TSV
    diagnostic and are checked separately when present.
    """

    sensors: dict[int, Sensor] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DiagnosticError(f"cannot read rig manifest {path}: {exc}") from exc
    for line_number, raw in enumerate(lines, 1):
        fields = raw.split()
        if not fields or fields[0].startswith("#") or fields[0] == "F":
            continue
        if fields[0] != "S" or len(fields) != 16:
            raise DiagnosticError(f"malformed sensor row {path}:{line_number}")
        try:
            index = int(fields[1])
            fx, fy, cx, cy = (float(value) for value in fields[5:9])
            rotation = stereo.rotation(*(float(value) for value in fields[9:13]))
            translation = np.asarray(
                [float(value) for value in fields[13:16]], dtype=np.float64
            )
        except (TypeError, ValueError, stereo.DiagnosticError) as exc:
            raise DiagnosticError(f"invalid sensor row {path}:{line_number}") from exc
        if index in sensors:
            raise DiagnosticError(f"duplicate sensor {index} at {path}:{line_number}")
        if not all(math.isfinite(value) and value > 0.0 for value in (fx, fy)):
            raise DiagnosticError(f"invalid focal length for sensor {index}")
        if not all(math.isfinite(value) for value in (cx, cy)):
            raise DiagnosticError(f"invalid principal point for sensor {index}")
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise DiagnosticError(f"invalid translation for sensor {index}")
        sensors[index] = Sensor(
            stereo.Camera(fx, fy, cx, cy), rotation, translation
        )
    if len(sensors) < 2:
        raise DiagnosticError("stereo PnP factors require at least two calibrated sensors")
    return sensors


def load_manifest_assignments(path: Path) -> dict[str, tuple[int, int]]:
    """Read optional ``F frame image sensor`` rows for name consistency checks."""

    assignments: dict[str, tuple[int, int]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if not fields or fields[0].startswith("#") or fields[0] == "S":
            continue
        if fields[0] != "F" or len(fields) != 4:
            raise DiagnosticError(f"malformed frame row {path}:{line_number}")
        try:
            frame, name, sensor = int(fields[1]), fields[2], int(fields[3])
        except ValueError as exc:
            raise DiagnosticError(f"invalid frame row {path}:{line_number}") from exc
        if not name or name in assignments:
            raise DiagnosticError(f"duplicate or empty image name {name!r} at {path}:{line_number}")
        assignments[name] = (frame, sensor)
    return assignments


def local_ray(sensor: Sensor, point: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Return the ray center and direction in the rig coordinate frame."""

    camera_ray = np.asarray(
        [
            (point[0] - sensor.camera.cx) / sensor.camera.fx,
            (point[1] - sensor.camera.cy) / sensor.camera.fy,
            1.0,
        ],
        dtype=np.float64,
    )
    rig_from_sensor = sensor.sensor_from_rig_rotation.T
    center = -(rig_from_sensor @ sensor.sensor_from_rig_translation)
    direction = rig_from_sensor @ camera_ray
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 0.0:
        raise DiagnosticError("invalid calibrated bearing")
    return center, direction / norm


def _required_columns(fieldnames: Iterable[str]) -> None:
    columns = set(fieldnames)
    required = {"track"}
    required.update(
        f"{prefix}_{index}"
        for index in range(4)
        for prefix in ("image", "keypoint", "frame", "sensor", "name", "x", "y")
    )
    missing = sorted(required - columns)
    if missing:
        raise DiagnosticError(f"accepted TSV is missing columns: {', '.join(missing)}")


def parse_track(
    row: dict[str, str],
    sensors: dict[int, Sensor],
    assignments: dict[str, tuple[int, int]] | None = None,
) -> Quadrilateral:
    """Parse and canonically arrange one exact two-sensor/two-frame row."""

    try:
        track = int(row["track"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DiagnosticError("quadrilateral has an invalid track id") from exc
    if track < 0:
        raise DiagnosticError(f"track {track} has a negative id")

    observations: dict[tuple[int, int], Observation] = {}
    seen_observations: set[tuple[int, int]] = set()
    for index in range(4):
        try:
            image = int(row[f"image_{index}"])
            keypoint = int(row[f"keypoint_{index}"])
            frame = int(row[f"frame_{index}"])
            sensor_index = int(row[f"sensor_{index}"])
            name = row[f"name_{index}"]
            x, y = float(row[f"x_{index}"]), float(row[f"y_{index}"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DiagnosticError(f"track {track} has an invalid observation {index}") from exc
        if image < 0 or keypoint < 0 or not name or any(char.isspace() for char in name):
            raise DiagnosticError(f"track {track} has invalid observation identity")
        if not (math.isfinite(x) and math.isfinite(y)):
            raise DiagnosticError(f"track {track} has non-finite pixel coordinates")
        if sensor_index not in sensors:
            raise DiagnosticError(f"track {track} names unknown sensor {sensor_index}")
        if assignments:
            expected = assignments.get(name)
            if expected is None:
                raise DiagnosticError(f"manifest has no assignment for image {name!r}")
            if expected != (frame, sensor_index):
                raise DiagnosticError(
                    f"track {track} disagrees with manifest for image {name!r}: "
                    f"TSV={(frame, sensor_index)} manifest={expected}"
                )
        identity = (image, keypoint)
        if identity in seen_observations:
            raise DiagnosticError(f"track {track} repeats image/keypoint {identity}")
        seen_observations.add(identity)
        slot = (frame, sensor_index)
        if slot in observations:
            raise DiagnosticError(f"track {track} repeats frame/sensor slot {slot}")
        observations[slot] = Observation(
            image, keypoint, frame, sensor_index, name, np.asarray([x, y], dtype=np.float64)
        )

    frames = sorted({frame for frame, _ in observations})
    sensor_indices = sorted({sensor_index for _, sensor_index in observations})
    if len(frames) != 2 or len(sensor_indices) != 2:
        raise DiagnosticError(f"track {track} is not a two-frame/two-sensor quadrilateral")
    if any(sum(frame == candidate for frame, _ in observations) != 2 for candidate in frames):
        raise DiagnosticError(f"track {track} does not have two observations per frame")
    if any(sum(sensor == candidate for _, sensor in observations) != 2 for candidate in sensor_indices):
        raise DiagnosticError(f"track {track} does not have two observations per sensor")
    if set(observations) != {(frame, sensor) for frame in frames for sensor in sensor_indices}:
        raise DiagnosticError(f"track {track} lacks one observation for a frame/sensor slot")
    return Quadrilateral(track, frames[0], frames[1], observations)


def read_quadrilaterals(
    tracks_path: Path,
    sensors: dict[int, Sensor],
    assignments: dict[str, tuple[int, int]] | None,
    min_frame_gap: int,
    max_frame_gap: int | None,
    max_tracks_per_pair: int | None,
) -> tuple[dict[tuple[int, int], list[Quadrilateral]], dict[str, int]]:
    """Read rows and deterministically cap each temporal pair."""

    grouped: dict[tuple[int, int], list[Quadrilateral]] = defaultdict(list)
    seen_tracks: set[int] = set()
    input_tracks = 0
    skipped_frame_gap = 0
    with tracks_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None:
            raise DiagnosticError("accepted TSV has no header")
        _required_columns(reader.fieldnames)
        for row in reader:
            if not any(value not in (None, "") for value in row.values()):
                continue
            input_tracks += 1
            quadrilateral = parse_track(row, sensors, assignments)
            if quadrilateral.track in seen_tracks:
                raise DiagnosticError(f"duplicate track id {quadrilateral.track}")
            seen_tracks.add(quadrilateral.track)
            gap = quadrilateral.frame_j - quadrilateral.frame_i
            if gap < min_frame_gap or (max_frame_gap is not None and gap > max_frame_gap):
                skipped_frame_gap += 1
                continue
            grouped[(quadrilateral.frame_i, quadrilateral.frame_j)].append(quadrilateral)

    capped_tracks = 0
    if max_tracks_per_pair is not None:
        for pair, rows in list(grouped.items()):
            rows.sort(key=lambda row: row.track)
            if len(rows) > max_tracks_per_pair:
                capped_tracks += len(rows) - max_tracks_per_pair
                grouped[pair] = rows[:max_tracks_per_pair]
    return grouped, {
        "input_tracks": input_tracks,
        "skipped_frame_gap": skipped_frame_gap,
        "capped_tracks": capped_tracks,
    }


def _triangulated_points(
    row: Quadrilateral, sensors: dict[int, Sensor], min_angle_deg: float
) -> dict[int, np.ndarray]:
    points: dict[int, np.ndarray] = {}
    for frame in (row.frame_i, row.frame_j):
        frame_observations = [
            row.observations[(frame, sensor_index)]
            for sensor_index in sorted({sensor for _, sensor in row.observations})
        ]
        rays = [
            local_ray(sensors[observation.sensor], tuple(observation.point))
            for observation in frame_observations
        ]
        point = stereo.triangulate(rays[0], rays[1], min_angle_deg)
        if point is not None:
            points[frame] = point
    return points


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _seed(frame_i: int, frame_j: int, direction: str, sensor_index: int) -> int:
    direction_code = 0 if direction == "forward" else 1
    # Keep the value in OpenCV's signed 32-bit seed range while avoiding Python
    # hash randomisation, which would make repeated diagnostics differ.
    value = (
        frame_i * 1_000_003
        + frame_j * 9176
        + direction_code * 65_537
        + sensor_index * 257
    ) & 0x7FFFFFFF
    return int(value or 1)


def _camera_matrix(camera: stereo.Camera) -> np.ndarray:
    return np.asarray(
        [[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def solve_sensor_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    sensor: Sensor,
    *,
    threshold_px: float,
    iterations: int,
    confidence: float,
    seed: int,
) -> dict[str, object]:
    """Run one deterministic bounded central-camera PnP estimate."""

    if cv2 is None:  # pragma: no cover - depends on environment
        raise DiagnosticError(f"OpenCV is required for solvePnPRansac: {_CV2_IMPORT_ERROR}")
    support = int(len(object_points))
    result: dict[str, object] = {
        "support": support,
        "inliers": 0,
        "inlier_fraction": 0.0,
        "median_inlier_residual_px": None,
        "p95_inlier_residual_px": None,
        "max_inlier_residual_px": None,
        "success": False,
        "seed": seed,
    }
    if support < 4:
        result["failure"] = "insufficient_correspondences"
        return result

    object_points = np.ascontiguousarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
    image_points = np.ascontiguousarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    try:
        cv2.setRNGSeed(seed)
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            _camera_matrix(sensor.camera),
            np.zeros((4, 1), dtype=np.float64),
            iterationsCount=int(iterations),
            reprojectionError=float(threshold_px),
            confidence=float(confidence),
            flags=int(cv2.SOLVEPNP_EPNP),
        )
    except cv2.error as exc:
        result["failure"] = f"opencv:{exc}"
        return result
    if not retval or rvec is None or tvec is None or inliers is None:
        result["failure"] = "ransac_no_pose"
        return result
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
        result["failure"] = "nonfinite_pose"
        return result
    try:
        rotation, _ = cv2.Rodrigues(rvec)
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, _camera_matrix(sensor.camera), None
        )
    except cv2.error as exc:
        result["failure"] = f"opencv_projection:{exc}"
        return result
    projected = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
    residuals = np.linalg.norm(projected - image_points.reshape(-1, 2), axis=1)
    indices = np.asarray(inliers, dtype=np.int64).reshape(-1)
    indices = indices[(indices >= 0) & (indices < support)]
    inlier_mask = np.zeros(support, dtype=bool)
    inlier_mask[np.unique(indices)] = True
    finite_mask = np.isfinite(residuals)
    inlier_mask &= finite_mask
    inlier_residuals = residuals[inlier_mask]
    count = int(len(inlier_residuals))
    result.update(
        {
            "success": True,
            "inliers": count,
            "inlier_fraction": count / support if support else 0.0,
            "median_inlier_residual_px": float(np.median(inlier_residuals))
            if count
            else None,
            "p95_inlier_residual_px": float(np.percentile(inlier_residuals, 95))
            if count
            else None,
            "max_inlier_residual_px": float(np.max(inlier_residuals)) if count else None,
            "rotation_sensor": rotation,
            "translation_sensor": tvec.reshape(3),
            "residuals_px": residuals,
            "inlier_mask": inlier_mask,
        }
    )
    return result


def _to_rig_pose(
    estimate: dict[str, object], sensor: Sensor
) -> tuple[np.ndarray, np.ndarray]:
    rotation_sensor = np.asarray(estimate["rotation_sensor"], dtype=np.float64)
    translation_sensor = np.asarray(estimate["translation_sensor"], dtype=np.float64)
    rig_from_sensor = sensor.sensor_from_rig_rotation.T
    return (
        rig_from_sensor @ rotation_sensor,
        rig_from_sensor @ (translation_sensor - sensor.sensor_from_rig_translation),
    )


def _build_direction(
    rows: list[Quadrilateral],
    source_frame: int,
    target_frame: int,
    direction: str,
    sensors: dict[int, Sensor],
    min_angle_deg: float,
    min_correspondences: int,
    min_inliers: int,
    threshold_px: float,
    iterations: int,
    confidence: float,
    max_sensor_rotation_error_deg: float,
    max_sensor_translation_error_m: float,
) -> dict[str, object]:
    sensor_indices = sorted({sensor for row in rows for _, sensor in row.observations})
    usable: dict[int, list[tuple[np.ndarray, np.ndarray, int]]] = {
        sensor_index: [] for sensor_index in sensor_indices
    }
    triangulated = 0
    for row in rows:
        points = _triangulated_points(row, sensors, min_angle_deg)
        point = points.get(source_frame)
        if point is None:
            continue
        triangulated += 1
        for sensor_index in sensor_indices:
            observation = row.observations[(target_frame, sensor_index)]
            usable[sensor_index].append((point, observation.point, row.track))

    result: dict[str, object] = {
        "direction": direction,
        "source_frame": source_frame,
        "target_frame": target_frame,
        "candidate_tracks": len(rows),
        "triangulated_source_tracks": triangulated,
        "support": 0,
        "accepted": False,
        "sensors": {},
    }
    if triangulated < min_correspondences:
        result["failure"] = "insufficient_triangulated_support"
        return result

    sensor_results: dict[int, dict[str, object]] = {}
    for sensor_index in sensor_indices:
        entries = usable[sensor_index]
        object_points = np.asarray([entry[0] for entry in entries], dtype=np.float64)
        image_points = np.asarray([entry[1] for entry in entries], dtype=np.float64)
        estimate = solve_sensor_pnp(
            object_points,
            image_points,
            sensors[sensor_index],
            threshold_px=threshold_px,
            iterations=iterations,
            confidence=confidence,
            seed=_seed(source_frame, target_frame, direction, sensor_index),
        )
        estimate["sensor"] = sensor_index
        estimate["track_ids"] = [entry[2] for entry in entries]
        accepted = bool(estimate["success"]) and int(estimate["inliers"]) >= min_inliers
        estimate["accepted"] = accepted
        if accepted:
            rig_rotation, rig_translation = _to_rig_pose(estimate, sensors[sensor_index])
            estimate["rotation_rig"] = rig_rotation
            estimate["translation_rig"] = rig_translation
            estimate["translation_m"] = float(np.linalg.norm(rig_translation))
            estimate["rotation_deg"] = _rotation_angle_deg(rig_rotation)
        sensor_results[sensor_index] = estimate
    result["sensors"] = sensor_results
    result["support"] = min(
        (int(value["support"]) for value in sensor_results.values()), default=0
    )
    accepted_sensor_results = [
        value for value in sensor_results.values() if bool(value.get("accepted"))
    ]
    if len(accepted_sensor_results) != len(sensor_indices):
        result["failure"] = "sensor_pnp_rejected"
        return result

    # Both sensor estimates are already expressed as the same rig-to-rig
    # transform.  Keep the best-supported estimate as the canonical factor;
    # the other estimate remains in the diagnostic for consistency checking.
    canonical = min(
        accepted_sensor_results,
        key=lambda value: (
            -int(value["inliers"]),
            float(value["median_inlier_residual_px"]),
            int(value["sensor"]),
        ),
    )
    rotation = np.asarray(canonical["rotation_rig"], dtype=np.float64)
    translation = np.asarray(canonical["translation_rig"], dtype=np.float64)
    result.update(
        {
            "pose_sensor": int(canonical["sensor"]),
            "rotation": rotation,
            "translation": translation,
            "inliers": int(canonical["inliers"]),
            "inlier_fraction": float(canonical["inlier_fraction"]),
            "median_inlier_residual_px": float(canonical["median_inlier_residual_px"]),
            "p95_inlier_residual_px": float(canonical["p95_inlier_residual_px"]),
            "translation_m": float(np.linalg.norm(translation)),
            "rotation_deg": _rotation_angle_deg(rotation),
        }
    )
    if len(accepted_sensor_results) >= 2:
        first = accepted_sensor_results[0]
        second = accepted_sensor_results[1]
        first_rotation = np.asarray(first["rotation_rig"])
        second_rotation = np.asarray(second["rotation_rig"])
        first_translation = np.asarray(first["translation_rig"])
        second_translation = np.asarray(second["translation_rig"])
        result["sensor_rotation_error_deg"] = _rotation_angle_deg(
            first_rotation @ second_rotation.T
        )
        result["sensor_translation_error_m"] = float(
            np.linalg.norm(first_translation - second_translation)
        )
    else:
        result["sensor_rotation_error_deg"] = None
        result["sensor_translation_error_m"] = None
    result["sensor_consistent"] = bool(
        result["sensor_rotation_error_deg"] is not None
        and float(result["sensor_rotation_error_deg"])
        <= max_sensor_rotation_error_deg
        and float(result["sensor_translation_error_m"])
        <= max_sensor_translation_error_m
    )
    result["accepted"] = result["sensor_consistent"]
    if not result["accepted"]:
        result["failure"] = "sensor_pose_disagreement"
    return result


def _serializable_sensor(value: dict[str, object]) -> dict[str, object]:
    keys = (
        "sensor",
        "support",
        "inliers",
        "inlier_fraction",
        "median_inlier_residual_px",
        "p95_inlier_residual_px",
        "max_inlier_residual_px",
        "success",
        "accepted",
        "failure",
        "seed",
        "translation_m",
        "rotation_deg",
    )
    return {key: value[key] for key in keys if key in value}


def _serializable_direction(value: dict[str, object]) -> dict[str, object]:
    output = {
        key: value[key]
        for key in (
            "direction",
            "source_frame",
            "target_frame",
            "candidate_tracks",
            "triangulated_source_tracks",
            "support",
            "inliers",
            "inlier_fraction",
            "median_inlier_residual_px",
            "p95_inlier_residual_px",
            "translation_m",
            "rotation_deg",
            "pose_sensor",
            "accepted",
            "failure",
            "sensor_rotation_error_deg",
            "sensor_translation_error_m",
            "sensor_consistent",
        )
        if key in value
    }
    output["sensors"] = {
        str(sensor): _serializable_sensor(sensor_result)
        for sensor, sensor_result in sorted(value.get("sensors", {}).items())
    }
    if value.get("accepted"):
        output["rotation"] = np.asarray(value["rotation"]).tolist()
        output["translation"] = np.asarray(value["translation"]).tolist()
    return output


def _inverse_pose(rotation: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inverse_rotation = rotation.T
    return inverse_rotation, -(inverse_rotation @ translation)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), fraction * 100.0))


def summary(values: list[float]) -> dict[str, object]:
    return {
        "count": len(values),
        "median": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "max": max(values, default=None),
    }


def estimate(
    tracks_path: Path,
    manifest_path: Path,
    *,
    min_angle_deg: float = 0.5,
    min_correspondences: int = 8,
    min_inliers: int = 6,
    ransac_threshold_px: float = 3.0,
    ransac_iterations: int = 128,
    ransac_confidence: float = 0.999,
    min_frame_gap: int = 1,
    max_frame_gap: int | None = None,
    max_tracks_per_pair: int | None = 2048,
    max_sensor_rotation_error_deg: float = 0.5,
    max_sensor_translation_error_m: float = 0.02,
    max_forward_reverse_rotation_error_deg: float = 5.0,
    max_forward_reverse_translation_error_m: float = 0.5,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Estimate accepted directional rig factors and return JSON diagnostics."""

    if cv2 is None:
        raise DiagnosticError(f"OpenCV is required: {_CV2_IMPORT_ERROR}")
    if not math.isfinite(min_angle_deg) or min_angle_deg <= 0.0:
        raise DiagnosticError("minimum triangulation angle must be finite and positive")
    if min_correspondences < 4 or min_inliers < 4 or min_inliers > min_correspondences:
        raise DiagnosticError("PnP support gates must satisfy 4 <= min_inliers <= min_correspondences")
    if not math.isfinite(ransac_threshold_px) or ransac_threshold_px <= 0.0:
        raise DiagnosticError("PnP reprojection threshold must be finite and positive")
    if ransac_iterations < 1:
        raise DiagnosticError("PnP RANSAC iterations must be positive")
    if not math.isfinite(ransac_confidence) or not 0.0 < ransac_confidence < 1.0:
        raise DiagnosticError("PnP confidence must be in (0, 1)")
    if min_frame_gap < 1 or (max_frame_gap is not None and max_frame_gap < min_frame_gap):
        raise DiagnosticError("frame-gap range is invalid")
    if max_tracks_per_pair is not None and max_tracks_per_pair < min_correspondences:
        raise DiagnosticError("max tracks per pair cannot be below min correspondences")
    for label, value in (
        ("sensor rotation gate", max_sensor_rotation_error_deg),
        ("sensor translation gate", max_sensor_translation_error_m),
        ("forward/reverse rotation gate", max_forward_reverse_rotation_error_deg),
        ("forward/reverse translation gate", max_forward_reverse_translation_error_m),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise DiagnosticError(f"{label} must be finite and non-negative")

    sensors = load_sensors(manifest_path)
    assignments = load_manifest_assignments(manifest_path)
    grouped, read_stats = read_quadrilaterals(
        tracks_path,
        sensors,
        assignments or None,
        min_frame_gap,
        max_frame_gap,
        max_tracks_per_pair,
    )
    source_sha256 = hashlib.sha256(tracks_path.read_bytes()).hexdigest()
    direction_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    factors: list[dict[str, object]] = []
    forward_reverse_rotation_errors: list[float] = []
    forward_reverse_translation_errors: list[float] = []
    sensor_rotation_errors: list[float] = []
    sensor_translation_errors: list[float] = []
    rejected_support = 0
    rejected_pnp = 0
    frame_pairs_with_any_source_triangulation = 0
    triangulated_direction_tracks = 0

    for (frame_i, frame_j), rows in sorted(grouped.items()):
        forward = _build_direction(
            rows,
            frame_i,
            frame_j,
            "forward",
            sensors,
            min_angle_deg,
            min_correspondences,
            min_inliers,
            ransac_threshold_px,
            ransac_iterations,
            ransac_confidence,
            max_sensor_rotation_error_deg,
            max_sensor_translation_error_m,
        )
        reverse = _build_direction(
            rows,
            frame_j,
            frame_i,
            "reverse",
            sensors,
            min_angle_deg,
            min_correspondences,
            min_inliers,
            ransac_threshold_px,
            ransac_iterations,
            ransac_confidence,
            max_sensor_rotation_error_deg,
            max_sensor_translation_error_m,
        )
        triangulated_direction_tracks += int(forward["triangulated_source_tracks"])
        triangulated_direction_tracks += int(reverse["triangulated_source_tracks"])
        frame_pairs_with_any_source_triangulation += int(
            bool(forward["triangulated_source_tracks"])
            or bool(reverse["triangulated_source_tracks"])
        )
        if int(forward["triangulated_source_tracks"]) < min_correspondences or int(
            reverse["triangulated_source_tracks"]
        ) < min_correspondences:
            rejected_support += int(
                int(forward["triangulated_source_tracks"]) < min_correspondences
            )
            rejected_support += int(
                int(reverse["triangulated_source_tracks"]) < min_correspondences
            )
        for direction in (forward, reverse):
            direction_rows.append(direction)
            if direction.get("accepted"):
                if direction.get("sensor_rotation_error_deg") is not None:
                    sensor_rotation_errors.append(float(direction["sensor_rotation_error_deg"]))
                    sensor_translation_errors.append(float(direction["sensor_translation_error_m"]))
            elif int(direction["triangulated_source_tracks"]) >= min_correspondences:
                rejected_pnp += 1

        pair_result: dict[str, object] = {
            "frame_i": frame_i,
            "frame_j": frame_j,
            "frame_gap": frame_j - frame_i,
            "forward": forward,
            "reverse": reverse,
            "bidirectional_consistent": False,
            "forward_reverse_rotation_error_deg": None,
            "forward_reverse_translation_error_m": None,
        }
        if forward.get("accepted") and reverse.get("accepted"):
            forward_rotation = np.asarray(forward["rotation"])
            forward_translation = np.asarray(forward["translation"])
            reverse_rotation = np.asarray(reverse["rotation"])
            reverse_translation = np.asarray(reverse["translation"])
            inverse_rotation, inverse_translation = _inverse_pose(
                reverse_rotation, reverse_translation
            )
            rotation_error = _rotation_angle_deg(forward_rotation @ inverse_rotation.T)
            translation_error = float(np.linalg.norm(forward_translation - inverse_translation))
            pair_result.update(
                {
                    "forward_reverse_rotation_error_deg": rotation_error,
                    "forward_reverse_translation_error_m": translation_error,
                    "bidirectional_consistent": (
                        rotation_error <= max_forward_reverse_rotation_error_deg
                        and translation_error <= max_forward_reverse_translation_error_m
                    ),
                }
            )
            forward_reverse_rotation_errors.append(rotation_error)
            forward_reverse_translation_errors.append(translation_error)
            for direction in (forward, reverse):
                direction["forward_reverse_rotation_error_deg"] = rotation_error
                direction["forward_reverse_translation_error_m"] = translation_error
                direction["forward_reverse_consistent"] = bool(
                    pair_result["bidirectional_consistent"]
                )
            if pair_result["bidirectional_consistent"]:
                factors.extend((forward, reverse))
        pair_rows.append(pair_result)

    consistent_pairs = sum(bool(row["bidirectional_consistent"]) for row in pair_rows)
    factors_payload = [_serializable_direction(factor) for factor in factors]
    pair_payload = []
    for pair in pair_rows:
        pair_payload.append(
            {
                "frame_i": pair["frame_i"],
                "frame_j": pair["frame_j"],
                "frame_gap": pair["frame_gap"],
                "bidirectional_consistent": pair["bidirectional_consistent"],
                "forward_reverse_rotation_error_deg": pair[
                    "forward_reverse_rotation_error_deg"
                ],
                "forward_reverse_translation_error_m": pair[
                    "forward_reverse_translation_error_m"
                ],
                "forward": _serializable_direction(pair["forward"]),
                "reverse": _serializable_direction(pair["reverse"]),
            }
        )
    result: dict[str, object] = {
        "schema": "visloc_rig_stereo_pnp_factors_v1",
        "ground_truth_used": False,
        "descriptor_values_used": False,
        "reconstructed_poses_used": False,
        "tracks_tsv": str(tracks_path),
        "tracks_tsv_sha256": source_sha256,
        "rig_manifest": str(manifest_path),
        "manifest_frame_assignments": len(assignments),
        "input_tracks": read_stats["input_tracks"],
        "candidate_frame_pairs": len(grouped),
        "frame_pairs_with_any_source_triangulation": frame_pairs_with_any_source_triangulation,
        "triangulated_direction_tracks": triangulated_direction_tracks,
        "accepted_factors": len(factors),
        "accepted_frame_pairs": sum(
            bool(row["forward"].get("accepted")) and bool(row["reverse"].get("accepted"))
            for row in pair_rows
        ),
        "forward_reverse_consistent_pairs": consistent_pairs,
        "rejected_support_directions": rejected_support,
        "rejected_pnp_directions": rejected_pnp,
        "skipped_frame_gap_tracks": read_stats["skipped_frame_gap"],
        "capped_tracks": read_stats["capped_tracks"],
        "sensors": sorted(sensors),
        "gates": {
            "min_triangulation_angle_deg": min_angle_deg,
            "min_correspondences": min_correspondences,
            "min_inliers": min_inliers,
            "ransac_threshold_px": ransac_threshold_px,
            "ransac_iterations": ransac_iterations,
            "ransac_confidence": ransac_confidence,
            "min_frame_gap": min_frame_gap,
            "max_frame_gap": max_frame_gap,
            "max_tracks_per_pair": max_tracks_per_pair,
            "max_sensor_rotation_error_deg": max_sensor_rotation_error_deg,
            "max_sensor_translation_error_m": max_sensor_translation_error_m,
            "max_forward_reverse_rotation_error_deg": max_forward_reverse_rotation_error_deg,
            "max_forward_reverse_translation_error_m": max_forward_reverse_translation_error_m,
        },
        "sensor_consistency": {
            "rotation_error_deg": summary(sensor_rotation_errors),
            "translation_error_m": summary(sensor_translation_errors),
        },
        "forward_reverse_consistency": {
            "rotation_error_deg": summary(forward_reverse_rotation_errors),
            "translation_error_m": summary(forward_reverse_translation_errors),
            "consistent_pair_count": consistent_pairs,
            "candidate_pair_count": len(pair_rows),
        },
        "factor_summary": {
            "translation_m": summary([float(value["translation_m"]) for value in factors]),
            "rotation_deg": summary([float(value["rotation_deg"]) for value in factors]),
            "inlier_fraction": summary([float(value["inlier_fraction"]) for value in factors]),
            "median_inlier_residual_px": summary(
                [float(value["median_inlier_residual_px"]) for value in factors]
            ),
        },
        "factors": factors_payload,
        "frame_pairs": pair_payload,
    }
    return factors, result


def _write_factors(path: Path, factors: list[dict[str, object]]) -> None:
    fields = [
        "frame_i",
        "frame_j",
        "direction",
        "source_frame",
        "target_frame",
        "pose_sensor",
        "accepted",
        "support",
        "inliers",
        "inlier_fraction",
        "median_inlier_residual_px",
        "p95_inlier_residual_px",
        "sensor_rotation_error_deg",
        "sensor_translation_error_m",
        "forward_reverse_rotation_error_deg",
        "forward_reverse_translation_error_m",
        "forward_reverse_consistent",
        "r00",
        "r01",
        "r02",
        "r10",
        "r11",
        "r12",
        "r20",
        "r21",
        "r22",
        "tx",
        "ty",
        "tz",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t")
        writer.writeheader()
        for factor in factors:
            rotation = np.asarray(factor["rotation"], dtype=np.float64)
            translation = np.asarray(factor["translation"], dtype=np.float64)
            values = {
                "frame_i": factor["source_frame"]
                if factor["direction"] == "forward"
                else factor["target_frame"],
                "frame_j": factor["target_frame"]
                if factor["direction"] == "forward"
                else factor["source_frame"],
                "direction": factor["direction"],
                "source_frame": factor["source_frame"],
                "target_frame": factor["target_frame"],
                "pose_sensor": factor["pose_sensor"],
                "accepted": int(bool(factor["accepted"])),
                "support": factor["support"],
                "inliers": factor["inliers"],
                "inlier_fraction": f"{factor['inlier_fraction']:.17g}",
                "median_inlier_residual_px": f"{factor['median_inlier_residual_px']:.17g}",
                "p95_inlier_residual_px": f"{factor['p95_inlier_residual_px']:.17g}",
                "sensor_rotation_error_deg": f"{factor['sensor_rotation_error_deg']:.17g}",
                "sensor_translation_error_m": f"{factor['sensor_translation_error_m']:.17g}",
                "forward_reverse_rotation_error_deg": f"{factor.get('forward_reverse_rotation_error_deg', float('nan')):.17g}",
                "forward_reverse_translation_error_m": f"{factor.get('forward_reverse_translation_error_m', float('nan')):.17g}",
                "forward_reverse_consistent": int(bool(factor.get('forward_reverse_consistent', False))),
                **{
                    f"r{row}{column}": f"{rotation[row, column]:.17g}"
                    for row in range(3)
                    for column in range(3)
                },
                "tx": f"{translation[0]:.17g}",
                "ty": f"{translation[1]:.17g}",
                "tz": f"{translation[2]:.17g}",
            }
            writer.writerow(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-tsv", "--tracks-tsv", dest="tracks_tsv", type=Path, required=True)
    parser.add_argument("--rig-manifest", type=Path, required=True)
    parser.add_argument("--output-factors-tsv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--min-angle-deg", "--min-triangulation-angle-deg", dest="min_angle_deg", type=float, default=0.5)
    parser.add_argument("--min-correspondences", type=int, default=8)
    parser.add_argument("--min-inliers", type=int, default=6)
    parser.add_argument("--ransac-threshold-px", "--pnp-reprojection-threshold-px", dest="ransac_threshold_px", type=float, default=3.0)
    parser.add_argument("--ransac-iterations", "--pnp-ransac-iterations", dest="ransac_iterations", type=int, default=128)
    parser.add_argument("--ransac-confidence", type=float, default=0.999)
    parser.add_argument("--min-frame-gap", type=int, default=1)
    parser.add_argument("--max-frame-gap", type=int)
    parser.add_argument("--max-tracks-per-pair", type=int, default=2048)
    parser.add_argument("--max-sensor-rotation-error-deg", type=float, default=0.5)
    parser.add_argument("--max-sensor-translation-error-m", type=float, default=0.02)
    parser.add_argument("--max-forward-reverse-rotation-error-deg", type=float, default=5.0)
    parser.add_argument("--max-forward-reverse-translation-error-m", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_tracks_per_pair == 0:
        raise DiagnosticError("--max-tracks-per-pair must be positive or negative for no cap")
    max_tracks_per_pair = args.max_tracks_per_pair if args.max_tracks_per_pair > 0 else None
    factors, result = estimate(
        args.tracks_tsv,
        args.rig_manifest,
        min_angle_deg=args.min_angle_deg,
        min_correspondences=args.min_correspondences,
        min_inliers=args.min_inliers,
        ransac_threshold_px=args.ransac_threshold_px,
        ransac_iterations=args.ransac_iterations,
        ransac_confidence=args.ransac_confidence,
        min_frame_gap=args.min_frame_gap,
        max_frame_gap=args.max_frame_gap,
        max_tracks_per_pair=max_tracks_per_pair,
        max_sensor_rotation_error_deg=args.max_sensor_rotation_error_deg,
        max_sensor_translation_error_m=args.max_sensor_translation_error_m,
        max_forward_reverse_rotation_error_deg=args.max_forward_reverse_rotation_error_deg,
        max_forward_reverse_translation_error_m=args.max_forward_reverse_translation_error_m,
    )
    _write_factors(args.output_factors_tsv, factors)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"factors", "frame_pairs"}},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
