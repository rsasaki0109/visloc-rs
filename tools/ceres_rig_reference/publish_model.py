#!/usr/bin/env python3
"""Publish a frozen Ceres rig solve as an identity-preserving COLMAP model.

This is deliberately a small, strict bridge for the bounded Ceres reference.
It is not a mapper and it never changes observations, camera calibration, or
track membership.  Only the final pose and landmark coordinates from the
state report are consumed; point ``ERROR`` values are recomputed from every
serialized observation.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import math
import os
import shutil
import struct
import sys
import tempfile
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


MAX_POSES = 512
MAX_POINTS = 8192
MAX_OBSERVATIONS = 262144
SHA_FIELDS = (
    "SOURCE_SHA256",
    "SOURCE_SHA256_CAMERAS",
    "SOURCE_SHA256_IMAGES",
    "SOURCE_SHA256_POINTS",
    "SOURCE_SHA256_MANIFEST",
)
EPS = 1.0e-12
QUAT_TOL = 1.0e-6
COST_TOL_REL = 1.0e-10
COST_TOL_ABS = 1.0e-6


class PublisherError(ValueError):
    """A source/state/publication contract violation."""


Vec = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # w, x, y, z


def _finite(value: str, label: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise PublisherError(f"{label} is not a number: {value!r}") from error
    if not math.isfinite(result):
        raise PublisherError(f"{label} is non-finite")
    return result


def _uint(value: str, label: str) -> int:
    try:
        result = int(value, 10)
    except ValueError as error:
        raise PublisherError(f"{label} is not an integer: {value!r}") from error
    if result < 0:
        raise PublisherError(f"{label} is negative")
    return result


def _bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PublisherError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _combined_sha(hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for label, key in (
        ("cameras.txt", "SOURCE_SHA256_CAMERAS"),
        ("images.txt", "SOURCE_SHA256_IMAGES"),
        ("points3D.txt", "SOURCE_SHA256_POINTS"),
        ("rig-manifest", "SOURCE_SHA256_MANIFEST"),
    ):
        digest.update(label.encode("ascii"))
        digest.update(b"\0")
        digest.update(hashes[key].encode("ascii"))
        digest.update(b"\xff")
    return digest.hexdigest()


def _norm(q: Quat) -> float:
    return math.sqrt(sum(component * component for component in q))


def _unit_quat(values: Sequence[str], label: str) -> Quat:
    if len(values) != 4:
        raise PublisherError(f"{label} requires four components")
    q = tuple(_finite(value, f"{label}[{index}]") for index, value in enumerate(values))
    norm = _norm(q)  # type: ignore[arg-type]
    if not math.isfinite(norm) or norm <= 1.0e-12 or abs(norm - 1.0) > QUAT_TOL:
        raise PublisherError(f"{label} is not a unit quaternion")
    return q  # Preserve serialized components; do not silently renormalize.


def _vec(values: Sequence[str], label: str) -> Vec:
    if len(values) != 3:
        raise PublisherError(f"{label} requires three components")
    return tuple(_finite(value, f"{label}[{index}]") for index, value in enumerate(values))  # type: ignore[return-value]


def _qmul(left: Quat, right: Quat) -> Quat:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _qconj(q: Quat) -> Quat:
    return (q[0], -q[1], -q[2], -q[3])


def _qrotate(q: Quat, value: Vec) -> Vec:
    pure = (0.0, value[0], value[1], value[2])
    rotated = _qmul(_qmul(q, pure), _qconj(q))
    return rotated[1], rotated[2], rotated[3]


def _vadd(left: Vec, right: Vec) -> Vec:
    return left[0] + right[0], left[1] + right[1], left[2] + right[2]


def _vsub(left: Vec, right: Vec) -> Vec:
    return left[0] - right[0], left[1] - right[1], left[2] - right[2]


def _qnormalize(q: Quat) -> Quat:
    norm = _norm(q)
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise PublisherError("composed image quaternion is invalid")
    return tuple(component / norm for component in q)  # type: ignore[return-value]


def _close(a: float, b: float, absolute: float = 1.0e-9, relative: float = 1.0e-9) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= absolute + relative * max(abs(a), abs(b))


def _vec_close(a: Vec, b: Vec, absolute: float = 1.0e-9, relative: float = 1.0e-9) -> bool:
    return all(_close(x, y, absolute, relative) for x, y in zip(a, b))


def _quat_close(a: Quat, b: Quat, absolute: float = 1.0e-9, relative: float = 1.0e-9) -> bool:
    direct = max(abs(x - y) for x, y in zip(a, b))
    negated = max(abs(x + y) for x, y in zip(a, b))
    return min(direct, negated) <= absolute + relative * max(max(abs(x) for x in a), max(abs(y) for y in b))


def _project(camera: "Camera", q: Quat, t: Vec, point: Vec, sensor_q: Quat, sensor_t: Vec) -> tuple[float, float, float]:
    rig_point = _vadd(_qrotate(q, point), t)
    sensor_point = _vadd(_qrotate(sensor_q, rig_point), sensor_t)
    if not all(math.isfinite(value) for value in sensor_point):
        raise PublisherError("projection produced non-finite sensor coordinates")
    if sensor_point[2] <= 0.0:
        raise PublisherError("projection has nonpositive depth")
    fx, fy, cx, cy = camera.params
    x = fx * sensor_point[0] / sensor_point[2] + cx
    y = fy * sensor_point[1] / sensor_point[2] + cy
    if not math.isfinite(x) or not math.isfinite(y):
        raise PublisherError("projection produced non-finite pixel coordinates")
    return x, y, sensor_point[2]


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, float, float, float]


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    point_id: int


@dataclass
class Image:
    image_id: int
    q: Quat
    t: Vec
    camera_id: int
    name: str
    keypoints: list[Keypoint]
    header_index: int
    points_index: int


@dataclass
class Point:
    point_id: int
    xyz: Vec
    rgb: tuple[str, str, str]
    error: float
    track: list[tuple[int, int]]


@dataclass
class Model:
    path: Path
    cameras_bytes: bytes
    camera_lines: list[str]
    image_lines: list[str]
    point_lines: list[str]
    cameras: dict[int, Camera]
    images: list[Image]
    points: list[Point]


@dataclass(frozen=True)
class Sensor:
    index: int
    camera_id: int
    width: int
    height: int
    params: tuple[float, float, float, float]
    q: Quat
    t: Vec


@dataclass
class Manifest:
    sensors: dict[int, Sensor]
    assignments: dict[str, tuple[int, int]]


@dataclass(frozen=True)
class FixtureCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, float, float, float]


@dataclass
class Fixture:
    hashes: dict[str, str]
    initial_cost: float
    initial_cost_bits: int
    cameras: dict[int, FixtureCamera]
    poses: dict[int, tuple[Quat, Vec]]
    fixed_pose: int
    landmarks: dict[int, Vec]
    observations: list[tuple[int, int, float, float, int, Quat, Vec]]


@dataclass
class State:
    hashes: dict[str, str]
    fixed_pose: int
    counts: dict[str, int]
    declared_initial_cost: float
    declared_initial_bits: int
    initial_cost: float
    initial_bits: int
    final_cost: float
    final_bits: int
    summary_initial_half: float
    summary_final_half: float
    poses: dict[int, tuple[Quat, Vec]]
    landmarks: dict[int, Vec]


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise PublisherError(f"cannot read {label} {path}: {error}") from error


def parse_manifest(path: Path) -> Manifest:
    sensors: dict[int, Sensor] = {}
    assignments: dict[str, tuple[int, int]] = {}
    frame_sensors: set[tuple[int, int]] = set()
    text = _read_bytes(path, "rig manifest").decode("utf-8")
    for line_number, raw in enumerate(text.splitlines(), 1):
        fields = raw.strip().split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] == "S":
            if len(fields) != 16:
                raise PublisherError(f"manifest line {line_number}: sensor requires 16 fields")
            index = _uint(fields[1], "sensor index")
            camera_id = _uint(fields[2], "sensor camera id")
            width = _uint(fields[3], "sensor width")
            height = _uint(fields[4], "sensor height")
            params = tuple(_finite(x, "sensor intrinsic") for x in fields[5:9])
            if width == 0 or height == 0 or params[0] <= 0 or params[1] <= 0:
                raise PublisherError(f"manifest line {line_number}: invalid sensor calibration")
            sensor = Sensor(index, camera_id, width, height, params, _unit_quat(fields[9:13], "sensor quaternion"), _vec(fields[13:16], "sensor translation"))
            if index in sensors:
                raise PublisherError(f"manifest line {line_number}: duplicate sensor {index}")
            sensors[index] = sensor
        elif fields[0] == "F":
            if len(fields) != 4:
                raise PublisherError(f"manifest line {line_number}: frame requires 4 fields")
            frame = _uint(fields[1], "frame id")
            name = fields[2]
            sensor = _uint(fields[3], "frame sensor")
            if not name or name in assignments or (frame, sensor) in frame_sensors:
                raise PublisherError(f"manifest line {line_number}: duplicate frame/image assignment")
            assignments[name] = (frame, sensor)
            frame_sensors.add((frame, sensor))
        else:
            raise PublisherError(f"manifest line {line_number}: unknown row {fields[0]!r}")
    if not sensors or not assignments or 0 not in sensors:
        raise PublisherError("manifest must contain sensors, assignments, and sensor 0")
    if sorted(sensors) != list(range(len(sensors))):
        raise PublisherError("manifest sensor indices must be contiguous from zero")
    if any(sensor not in sensors for _, sensor in assignments.values()):
        raise PublisherError("manifest assignment references unknown sensor")
    centers = []
    for sensor in sensors.values():
        centers.append(_qrotate(_qconj(sensor.q), (-sensor.t[0], -sensor.t[1], -sensor.t[2])))
    baseline = max(math.dist(left, right) for left in centers for right in centers)
    if not math.isfinite(baseline) or baseline <= 1.0e-9:
        raise PublisherError("manifest has no nonzero sensor baseline")
    return Manifest(sensors, assignments)


def _parse_camera_line(fields: list[str], line_number: int) -> Camera:
    if len(fields) != 8 or fields[1] != "PINHOLE":
        raise PublisherError(f"cameras.txt line {line_number}: only PINHOLE with four parameters is accepted")
    camera_id = _uint(fields[0], "camera id")
    width = _uint(fields[2], "camera width")
    height = _uint(fields[3], "camera height")
    params = tuple(_finite(x, "camera parameter") for x in fields[4:8])
    if width == 0 or height == 0 or params[0] <= 0 or params[1] <= 0:
        raise PublisherError(f"cameras.txt line {line_number}: invalid camera")
    return Camera(camera_id, fields[1], width, height, params)  # type: ignore[arg-type]


def _parse_points2d(line: str, line_number: int) -> list[Keypoint]:
    fields = line.split()
    if len(fields) % 3:
        raise PublisherError(f"images.txt line {line_number}: malformed POINTS2D row")
    result = []
    for offset in range(0, len(fields), 3):
        x = _finite(fields[offset], "POINTS2D x")
        y = _finite(fields[offset + 1], "POINTS2D y")
        try:
            point_id = int(fields[offset + 2], 10)
        except ValueError as error:
            raise PublisherError(f"images.txt line {line_number}: invalid POINT3D_ID") from error
        if point_id < -1:
            raise PublisherError(f"images.txt line {line_number}: POINT3D_ID below -1")
        result.append(Keypoint(x, y, point_id))
    return result


def parse_model(path: Path) -> Model:
    try:
        model_metadata = path.lstat()
    except OSError as error:
        raise PublisherError(f"model is unavailable: {path}: {error}") from error
    if not stat.S_ISDIR(model_metadata.st_mode) or stat.S_ISLNK(model_metadata.st_mode):
        raise PublisherError(f"model is not a real directory: {path}")
    camera_path = path / "cameras.txt"
    image_path = path / "images.txt"
    point_path = path / "points3D.txt"
    for required in (camera_path, image_path, point_path):
        try:
            required_metadata = required.lstat()
        except OSError as error:
            raise PublisherError(f"model file is unavailable: {required}: {error}") from error
        if not stat.S_ISREG(required_metadata.st_mode) or stat.S_ISLNK(required_metadata.st_mode):
            raise PublisherError(f"model file is missing or is a symlink: {required}")
    cameras_bytes = _read_bytes(camera_path, "cameras.txt")
    camera_text = cameras_bytes.decode("utf-8")
    camera_lines = camera_text.splitlines(keepends=True)
    cameras: dict[int, Camera] = {}
    for line_number, raw in enumerate(camera_lines, 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        camera = _parse_camera_line(raw.split(), line_number)
        if camera.camera_id in cameras:
            raise PublisherError(f"cameras.txt line {line_number}: duplicate camera id")
        cameras[camera.camera_id] = camera
    if not cameras:
        raise PublisherError("cameras.txt is empty")

    image_lines = _read_bytes(image_path, "images.txt").decode("utf-8").splitlines(keepends=True)
    images: list[Image] = []
    image_ids: set[int] = set()
    names: set[str] = set()
    index = 0
    while index < len(image_lines):
        raw = image_lines[index]
        if not raw.strip() or raw.lstrip().startswith("#"):
            index += 1
            continue
        fields = raw.split()
        if len(fields) != 10:
            raise PublisherError(f"images.txt line {index + 1}: malformed image header")
        image_id = _uint(fields[0], "image id")
        q = _unit_quat(fields[1:5], "image quaternion")
        t = _vec(fields[5:8], "image translation")
        camera_id = _uint(fields[8], "image camera id")
        name = fields[9]
        if image_id in image_ids or name in names or camera_id not in cameras:
            raise PublisherError(f"images.txt line {index + 1}: duplicate image or unknown camera")
        if index + 1 >= len(image_lines):
            raise PublisherError(f"images.txt line {index + 1}: missing POINTS2D row")
        point_line = image_lines[index + 1]
        if point_line.strip().startswith("#"):
            raise PublisherError(f"images.txt line {index + 2}: POINTS2D row cannot be a comment")
        keypoints = _parse_points2d(point_line, index + 2)
        images.append(Image(image_id, q, t, camera_id, name, keypoints, index, index + 1))
        image_ids.add(image_id)
        names.add(name)
        index += 2
    if not images or len(images) > MAX_POSES * 2:
        raise PublisherError("images.txt has no images or exceeds bounded image cap")

    point_lines = _read_bytes(point_path, "points3D.txt").decode("utf-8").splitlines(keepends=True)
    points: list[Point] = []
    point_ids: set[int] = set()
    for line_number, raw in enumerate(point_lines, 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        fields = raw.split()
        if len(fields) < 8 or (len(fields) - 8) % 2:
            raise PublisherError(f"points3D.txt line {line_number}: malformed point/track row")
        point_id = _uint(fields[0], "point id")
        if point_id in point_ids:
            raise PublisherError(f"points3D.txt line {line_number}: duplicate point id")
        xyz = _vec(fields[1:4], "point coordinates")
        rgb = fields[4:7]
        for value in rgb:
            try:
                color = int(value, 10)
            except ValueError as error:
                raise PublisherError(f"points3D.txt line {line_number}: invalid RGB") from error
            if not 0 <= color <= 255:
                raise PublisherError(f"points3D.txt line {line_number}: RGB outside u8")
        error_value = _finite(fields[7], "point ERROR")
        if error_value < 0.0:
            raise PublisherError(f"points3D.txt line {line_number}: negative ERROR")
        track = []
        for offset in range(8, len(fields), 2):
            image_id = _uint(fields[offset], "track image id")
            keypoint = _uint(fields[offset + 1], "track keypoint index")
            track.append((image_id, keypoint))
        if not track:
            raise PublisherError(f"points3D.txt line {line_number}: unsupported point has empty track")
        points.append(Point(point_id, xyz, (rgb[0], rgb[1], rgb[2]), error_value, track))
        point_ids.add(point_id)
    if not points or len(points) > MAX_POINTS:
        raise PublisherError("points3D.txt has no points or exceeds bounded point cap")

    by_image = {image.image_id: image for image in images}
    referenced: dict[tuple[int, int], int] = {}
    for point in points:
        for image_id, keypoint in point.track:
            image = by_image.get(image_id)
            if image is None or keypoint >= len(image.keypoints):
                raise PublisherError(f"point {point.point_id}: track references unknown keypoint")
            key = (image_id, keypoint)
            if key in referenced:
                raise PublisherError(f"duplicate observation reference {key}")
            observed = image.keypoints[keypoint].point_id
            if observed != point.point_id:
                raise PublisherError(f"point {point.point_id}: reverse POINTS2D reference mismatch")
            referenced[key] = point.point_id
    for image in images:
        for keypoint_index, keypoint in enumerate(image.keypoints):
            key = (image.image_id, keypoint_index)
            if keypoint.point_id >= 0 and referenced.get(key) != keypoint.point_id:
                raise PublisherError(f"image {image.image_id} keypoint {keypoint_index}: missing reverse track")
    observation_count = sum(len(point.track) for point in points)
    if observation_count == 0 or observation_count > MAX_OBSERVATIONS:
        raise PublisherError("model observation count is outside bounded range")
    return Model(path, cameras_bytes, camera_lines, image_lines, point_lines, cameras, images, points)


def parse_fixture(path: Path) -> Fixture:
    lines = _read_bytes(path, "fixture").decode("utf-8").splitlines()
    if not lines or lines[0].strip() != "VISLOC_BA_ORACLE_FIXTURE 1":
        raise PublisherError("unsupported fixture magic/version")
    index = 1
    hashes: dict[str, str] = {}

    def header(expected: str) -> list[str]:
        nonlocal index
        if index >= len(lines):
            raise PublisherError(f"fixture missing {expected}")
        fields = lines[index].split()
        index += 1
        if len(fields) != 2 or fields[0] != expected:
            raise PublisherError(f"fixture expected {expected}")
        if expected.startswith("SOURCE_SHA256"):
            value = fields[1].lower()
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise PublisherError(f"fixture {expected} is not SHA256")
            if expected in hashes:
                raise PublisherError(f"fixture duplicates {expected}")
            hashes[expected] = value
        return fields

    for key in SHA_FIELDS:
        header(key)
    initial_fields = header("INITIAL_COST")
    initial_cost = _finite(initial_fields[1], "fixture INITIAL_COST")
    if initial_cost < 0:
        raise PublisherError("fixture INITIAL_COST is negative")
    bits = _uint(header("INITIAL_COST_BITS")[1], "fixture INITIAL_COST_BITS")
    if _bits(initial_cost) != bits:
        raise PublisherError("fixture INITIAL_COST_BITS mismatch")

    def count(key: str, cap: int) -> int:
        value = _uint(header(key)[1], f"fixture {key}")
        if value == 0 or value > cap:
            raise PublisherError(f"fixture {key} outside bounded range")
        return value

    camera_count = count("CAMERA_COUNT", 32)
    pose_count = count("POSE_COUNT", MAX_POSES)
    point_count = count("LANDMARK_COUNT", MAX_POINTS)
    obs_count = count("OBSERVATION_COUNT", MAX_OBSERVATIONS)
    cameras: dict[int, FixtureCamera] = {}
    for _ in range(camera_count):
        fields = lines[index].split() if index < len(lines) else []
        index += 1
        if len(fields) != 10 or fields[0] != "CAMERA" or fields[2] != "PINHOLE" or fields[5] != "4":
            raise PublisherError("fixture malformed CAMERA row")
        camera_id = _uint(fields[1], "fixture camera id")
        if camera_id in cameras:
            raise PublisherError("fixture duplicate camera id")
        width = _uint(fields[3], "fixture camera width")
        height = _uint(fields[4], "fixture camera height")
        params = tuple(_finite(value, "fixture camera parameter") for value in fields[6:10])
        cameras[camera_id] = FixtureCamera(camera_id, fields[2], width, height, params)  # type: ignore[arg-type]
    poses: dict[int, tuple[Quat, Vec]] = {}
    for _ in range(pose_count):
        fields = lines[index].split() if index < len(lines) else []
        index += 1
        if len(fields) != 9 or fields[0] != "POSE":
            raise PublisherError("fixture malformed POSE row")
        pose_id = _uint(fields[1], "fixture pose id")
        if pose_id in poses:
            raise PublisherError("fixture duplicate pose id")
        poses[pose_id] = (_unit_quat(fields[2:6], "fixture pose quaternion"), _vec(fields[6:9], "fixture pose translation"))
    fields = lines[index].split() if index < len(lines) else []
    index += 1
    if len(fields) != 2 or fields[0] != "FIXED_POSE":
        raise PublisherError("fixture missing FIXED_POSE")
    fixed_pose = _uint(fields[1], "fixture fixed pose")
    if fixed_pose != 0 or fixed_pose not in poses:
        raise PublisherError("fixture requires existing FIXED_POSE 0")
    landmarks: dict[int, Vec] = {}
    for _ in range(point_count):
        fields = lines[index].split() if index < len(lines) else []
        index += 1
        if len(fields) != 5 or fields[0] != "LANDMARK":
            raise PublisherError("fixture malformed LANDMARK row")
        point_id = _uint(fields[1], "fixture landmark id")
        if point_id in landmarks:
            raise PublisherError("fixture duplicate landmark id")
        landmarks[point_id] = _vec(fields[2:5], "fixture landmark")
    observations = []
    sensor_by_camera: dict[int, tuple[Quat, Vec]] = {}
    for _ in range(obs_count):
        fields = lines[index].split() if index < len(lines) else []
        index += 1
        if len(fields) != 13 or fields[0] != "RIG_OBSERVATION":
            raise PublisherError("fixture malformed RIG_OBSERVATION row")
        frame = _uint(fields[1], "fixture observation frame")
        point = _uint(fields[2], "fixture observation point")
        x = _finite(fields[3], "fixture observation x")
        y = _finite(fields[4], "fixture observation y")
        camera = _uint(fields[5], "fixture observation camera")
        q = _unit_quat(fields[6:10], "fixture sensor quaternion")
        t = _vec(fields[10:13], "fixture sensor translation")
        if frame not in poses or point not in landmarks or camera not in cameras:
            raise PublisherError("fixture observation references an unknown record")
        previous = sensor_by_camera.setdefault(camera, (q, t))
        if not _quat_close(previous[0], q, 1.0e-12, 1.0e-12) or not _vec_close(previous[1], t, 1.0e-12, 1.0e-12):
            raise PublisherError("fixture sensor transform changes for one camera")
        observations.append((frame, point, x, y, camera, q, t))
    if index >= len(lines) or lines[index].strip() != "END":
        raise PublisherError("fixture missing END")
    if any(line.strip() for line in lines[index + 1:]):
        raise PublisherError("fixture has records after END")
    if set(sensor_by_camera) != set(cameras):
        raise PublisherError("fixture does not provide a sensor transform for every camera")
    if len(cameras) < 2:
        raise PublisherError("fixture requires a multi-sensor rig")
    centers = [_qrotate(_qconj(q), (-t[0], -t[1], -t[2])) for q, t in sensor_by_camera.values()]
    if max(math.dist(left, right) for left in centers for right in centers) <= 1.0e-9:
        raise PublisherError("fixture has no nonzero sensor baseline")
    return Fixture(hashes, initial_cost, bits, cameras, poses, fixed_pose, landmarks, observations)


def _parse_state_float(fields: list[str], label: str) -> float:
    if len(fields) != 2:
        raise PublisherError(f"state malformed {label}")
    return _finite(fields[1], f"state {label}")


def parse_state(path: Path) -> State:
    lines = _read_bytes(path, "solve state").decode("utf-8").splitlines()
    if not lines or lines[0].strip() != "VISLOC_BA_CERES_SOLVE_STATE 1":
        raise PublisherError("unsupported state magic/version")
    single: dict[str, list[str]] = {}
    poses: dict[int, tuple[Quat, Vec]] = {}
    landmarks: dict[int, Vec] = {}
    options: set[str] = set()
    index = 1
    section: str | None = None
    report_end: str | None = None
    sections_seen: set[str] = set()
    reports_seen: set[str] = set()
    iteration_rows = 0
    while index < len(lines):
        raw = lines[index]
        index += 1
        fields = raw.split()
        if section == "options":
            if fields == ["OPTIONS_END"]:
                section = None
            elif len(fields) >= 2:
                if fields[0] in options:
                    raise PublisherError(f"state duplicate option {fields[0]}")
                options.add(fields[0])
            else:
                raise PublisherError("state malformed option row")
            continue
        if section == "pose":
            if fields == ["POSE_STATE_END"]:
                section = None
                continue
            if len(fields) != 9 or fields[0] != "POSE":
                raise PublisherError("state malformed POSE row")
            pose_id = _uint(fields[1], "state pose id")
            if pose_id in poses:
                raise PublisherError("state duplicate pose id")
            poses[pose_id] = (_unit_quat(fields[2:6], "state pose quaternion"), _vec(fields[6:9], "state pose translation"))
            continue
        if section == "landmark":
            if fields == ["LANDMARK_STATE_END"]:
                section = None
                continue
            if len(fields) != 5 or fields[0] != "LANDMARK":
                raise PublisherError("state malformed LANDMARK row")
            point_id = _uint(fields[1], "state landmark id")
            if point_id in landmarks:
                raise PublisherError("state duplicate landmark id")
            landmarks[point_id] = _vec(fields[2:5], "state landmark")
            continue
        if report_end is not None:
            if raw == report_end:
                report_end = None
            continue
        if not fields:
            continue
        if fields[0] in ("OPTIONS_BEGIN", "POSE_STATE_BEGIN", "LANDMARK_STATE_BEGIN"):
            if section is not None:
                raise PublisherError("state nested section")
            if fields[0] in sections_seen:
                raise PublisherError(f"state duplicate section {fields[0]}")
            sections_seen.add(fields[0])
            section = {"OPTIONS_BEGIN": "options", "POSE_STATE_BEGIN": "pose", "LANDMARK_STATE_BEGIN": "landmark"}[fields[0]]
            continue
        if fields[0] in ("CERES_SUMMARY_MESSAGE_BEGIN", "CERES_SUMMARY_FULL_REPORT_BEGIN"):
            if report_end is not None:
                raise PublisherError("state nested report")
            if fields[0] in reports_seen:
                raise PublisherError(f"state duplicate report {fields[0]}")
            reports_seen.add(fields[0])
            report_end = fields[0].replace("_BEGIN", "_END")
            continue
        if fields[0] == "ITERATION":
            if len(fields) != 17:
                raise PublisherError("state malformed ITERATION row")
            iteration_rows += 1
            for value in fields[1:]:
                if value not in ("0", "1"):
                    _finite(value, "state iteration field")
            continue
        if fields[0] == "END":
            if len(fields) != 1:
                raise PublisherError("state malformed END")
            if section is not None or report_end is not None:
                raise PublisherError("state ended inside a section/report")
            if index < len(lines) and any(line.strip() for line in lines[index:]):
                raise PublisherError("state has records after END")
            break
        if len(fields) < 2:
            raise PublisherError(f"state malformed row {raw!r}")
        key = fields[0]
        if key in single:
            raise PublisherError(f"state duplicate field {key}")
        single[key] = fields
    else:
        raise PublisherError("state missing END")
    if section is not None or report_end is not None:
        raise PublisherError("state has unterminated section/report")
    if sections_seen != {"OPTIONS_BEGIN", "POSE_STATE_BEGIN", "LANDMARK_STATE_BEGIN"}:
        raise PublisherError("state is missing an explicit required section")
    if reports_seen != {"CERES_SUMMARY_MESSAGE_BEGIN", "CERES_SUMMARY_FULL_REPORT_BEGIN"}:
        raise PublisherError("state is missing an explicit Ceres report section")
    required = set(SHA_FIELDS) | {
        "CERES_VERSION", "FIXED_POSE", "CAMERA_COUNT", "POSE_COUNT", "LANDMARK_COUNT", "OBSERVATION_COUNT",
        "DECLARED_INITIAL_COST", "DECLARED_INITIAL_COST_BITS", "SOLVER_MODE", "COST_CONVENTION",
        "INITIAL_FULL_SQUARED_COST", "INITIAL_FULL_SQUARED_COST_BITS", "INITIAL_OBSERVATIONS", "INITIAL_ALL_OBSERVATIONS_VALID",
        "INITIAL_POSITIVE_DEPTH", "INITIAL_MIN_DEPTH", "INITIAL_MAX_DEPTH", "FINAL_FULL_SQUARED_COST", "FINAL_FULL_SQUARED_COST_BITS",
        "FINAL_OBSERVATIONS", "FINAL_ALL_OBSERVATIONS_VALID", "FINAL_POSITIVE_DEPTH", "FINAL_MIN_DEPTH", "FINAL_MAX_DEPTH",
        "CERES_SUMMARY_INITIAL_HALF_COST", "CERES_SUMMARY_FINAL_HALF_COST", "CERES_SUMMARY_TERMINATION_TYPE",
        "CERES_SUMMARY_IS_SOLUTION_USABLE", "ITERATION_COUNT",
        "INITIAL_CERES_EIGEN_MAX_RESIDUAL_ABS_DIFF", "INITIAL_CERES_EIGEN_MAX_DEPTH_ABS_DIFF",
        "FINAL_CERES_EIGEN_MAX_RESIDUAL_ABS_DIFF", "FINAL_CERES_EIGEN_MAX_DEPTH_ABS_DIFF",
    }
    missing = sorted(required - set(single))
    if missing:
        raise PublisherError(f"state missing required fields: {', '.join(missing)}")
    hashes = {key: single[key][1].lower() for key in SHA_FIELDS}
    for key, value in hashes.items():
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise PublisherError(f"state {key} is not SHA256")
    if single["CERES_VERSION"][1] != "2.2.0" or single["SOLVER_MODE"][1] != "CERES_STANDALONE_REFERENCE":
        raise PublisherError("state is not the frozen Ceres 2.2.0 reference")
    if single["COST_CONVENTION"][1] != "CERES_HALF_SQUARED_INTERNAL_FULL_SQUARED_REPORTED":
        raise PublisherError("state has an unsupported cost convention")
    def int_field(key: str) -> int:
        fields = single[key]
        if len(fields) != 2:
            raise PublisherError(f"state malformed {key}")
        return _uint(fields[1], f"state {key}")
    counts = {key: int_field(key) for key in ("CAMERA_COUNT", "POSE_COUNT", "LANDMARK_COUNT", "OBSERVATION_COUNT")}
    limits = {"CAMERA_COUNT": 32, "POSE_COUNT": MAX_POSES, "LANDMARK_COUNT": MAX_POINTS, "OBSERVATION_COUNT": MAX_OBSERVATIONS}
    if any(value == 0 or value > limits[key] for key, value in counts.items()):
        raise PublisherError("state count is outside bounded range")
    fixed_pose = int_field("FIXED_POSE")
    declared_initial = _parse_state_float(single["DECLARED_INITIAL_COST"], "DECLARED_INITIAL_COST")
    declared_bits = int_field("DECLARED_INITIAL_COST_BITS")
    initial_cost = _parse_state_float(single["INITIAL_FULL_SQUARED_COST"], "INITIAL_FULL_SQUARED_COST")
    initial_bits = int_field("INITIAL_FULL_SQUARED_COST_BITS")
    final_cost = _parse_state_float(single["FINAL_FULL_SQUARED_COST"], "FINAL_FULL_SQUARED_COST")
    final_bits = int_field("FINAL_FULL_SQUARED_COST_BITS")
    if any(_bits(value) != bits for value, bits in ((declared_initial, declared_bits), (initial_cost, initial_bits), (final_cost, final_bits))):
        raise PublisherError("state cost bit field mismatch")
    if declared_initial < 0.0 or initial_cost < 0.0 or final_cost < 0.0:
        raise PublisherError("state cost is negative")
    summary_initial = _parse_state_float(single["CERES_SUMMARY_INITIAL_HALF_COST"], "CERES_SUMMARY_INITIAL_HALF_COST")
    summary_final = _parse_state_float(single["CERES_SUMMARY_FINAL_HALF_COST"], "CERES_SUMMARY_FINAL_HALF_COST")
    if int_field("CERES_SUMMARY_IS_SOLUTION_USABLE") != 1 or int_field("INITIAL_ALL_OBSERVATIONS_VALID") != 1 or int_field("FINAL_ALL_OBSERVATIONS_VALID") != 1:
        raise PublisherError("state reports an unusable or partially evaluated solution")
    if int_field("INITIAL_OBSERVATIONS") != counts["OBSERVATION_COUNT"] or int_field("FINAL_OBSERVATIONS") != counts["OBSERVATION_COUNT"]:
        raise PublisherError("state observation count does not match its declared count")
    if int_field("INITIAL_POSITIVE_DEPTH") != counts["OBSERVATION_COUNT"] or int_field("FINAL_POSITIVE_DEPTH") != counts["OBSERVATION_COUNT"]:
        raise PublisherError("state does not report positive depth for every observation")
    for key in ("INITIAL_MIN_DEPTH", "INITIAL_MAX_DEPTH", "FINAL_MIN_DEPTH", "FINAL_MAX_DEPTH", "INITIAL_CERES_EIGEN_MAX_RESIDUAL_ABS_DIFF", "INITIAL_CERES_EIGEN_MAX_DEPTH_ABS_DIFF", "FINAL_CERES_EIGEN_MAX_RESIDUAL_ABS_DIFF", "FINAL_CERES_EIGEN_MAX_DEPTH_ABS_DIFF"):
        if _parse_state_float(single[key], key) < 0.0 and "DEPTH" not in key:
            raise PublisherError(f"state {key} is negative")
    if _parse_state_float(single["INITIAL_MIN_DEPTH"], "INITIAL_MIN_DEPTH") <= 0.0 or _parse_state_float(single["FINAL_MIN_DEPTH"], "FINAL_MIN_DEPTH") <= 0.0:
        raise PublisherError("state reports nonpositive minimum depth")
    if int_field("ITERATION_COUNT") != iteration_rows:
        raise PublisherError("state iteration count does not match iteration rows")
    if abs(summary_initial * 2.0 - initial_cost) > COST_TOL_ABS + COST_TOL_REL * abs(initial_cost) or abs(summary_final * 2.0 - final_cost) > COST_TOL_ABS + COST_TOL_REL * abs(final_cost):
        raise PublisherError("state Ceres half-cost summary disagrees with full cost")
    if len(poses) != counts["POSE_COUNT"] or len(landmarks) != counts["LANDMARK_COUNT"]:
        raise PublisherError("state pose/landmark rows do not match declared counts")
    return State(hashes, fixed_pose, counts, declared_initial, declared_bits, initial_cost, initial_bits, final_cost, final_bits, summary_initial, summary_final, poses, landmarks)


def _resolved(path: Path) -> Path:
    try:
        return path.expanduser().absolute().resolve(strict=False)
    except OSError as error:
        raise PublisherError(f"cannot resolve path {path}: {error}") from error


def _overlap(left: Path, right: Path) -> bool:
    left_resolved = _resolved(left)
    right_resolved = _resolved(right)
    return left_resolved == right_resolved or left_resolved in right_resolved.parents or right_resolved in left_resolved.parents


def _require_real_file(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise PublisherError(f"{label} is unavailable: {path}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise PublisherError(f"{label} must be a regular non-symlink file: {path}")


def _compare_hashes(fixture: Fixture, state: State, model: Model, manifest_path: Path) -> dict[str, str]:
    files = {
        "SOURCE_SHA256_CAMERAS": model.path / "cameras.txt",
        "SOURCE_SHA256_IMAGES": model.path / "images.txt",
        "SOURCE_SHA256_POINTS": model.path / "points3D.txt",
        "SOURCE_SHA256_MANIFEST": manifest_path,
    }
    hashes = {key: _sha256(path) for key, path in files.items()}
    hashes["SOURCE_SHA256"] = _combined_sha(hashes)
    for owner, source in (("fixture", fixture.hashes), ("state", state.hashes)):
        for key, actual in hashes.items():
            if source.get(key) != actual:
                raise PublisherError(f"{owner} {key} does not match source input")
    if fixture.hashes != state.hashes:
        raise PublisherError("fixture and state source SHA fields differ")
    return hashes


def _validate_manifest_model(model: Model, manifest: Manifest, fixture: Fixture) -> dict[int, tuple[int, int]]:
    if len(model.images) != len(manifest.assignments):
        raise PublisherError("source image count differs from rig manifest")
    by_sensor_cameras: dict[int, set[int]] = collections.defaultdict(set)
    for image in model.images:
        assignment = manifest.assignments.get(image.name)
        if assignment is None:
            raise PublisherError(f"source image {image.name!r} is absent from manifest")
        frame, sensor = assignment
        by_sensor_cameras[sensor].add(image.camera_id)
        spec = manifest.sensors[sensor]
        camera = model.cameras[image.camera_id]
        if image.camera_id != spec.camera_id or camera.width != spec.width or camera.height != spec.height or any(not _close(a, b, 1.0e-8, 1.0e-8) for a, b in zip(camera.params, spec.params)):
            raise PublisherError(f"image {image.name!r} camera calibration disagrees with rig manifest")
        if image.camera_id not in fixture.cameras:
            raise PublisherError(f"image {image.name!r} camera is absent from fixture")
        fixture_camera = fixture.cameras[image.camera_id]
        if fixture_camera.width != camera.width or fixture_camera.height != camera.height or any(not _close(a, b, 1.0e-8, 1.0e-8) for a, b in zip(fixture_camera.params, camera.params)):
            raise PublisherError(f"image {image.name!r} camera calibration disagrees with fixture")
    if set(manifest.assignments) != {image.name for image in model.images}:
        raise PublisherError("manifest/source image name sets differ")
    sensor_to_camera: dict[int, int] = {}
    for sensor, camera_ids in by_sensor_cameras.items():
        if len(camera_ids) != 1:
            raise PublisherError(f"sensor {sensor} maps to multiple source cameras")
        sensor_to_camera[sensor] = next(iter(camera_ids))
    if set(sensor_to_camera) != set(manifest.sensors) or set(sensor_to_camera.values()) != set(fixture.cameras):
        raise PublisherError("manifest/source/fixture sensor-camera mapping is incomplete")
    for sensor, spec in manifest.sensors.items():
        if sensor_to_camera[sensor] != spec.camera_id:
            raise PublisherError(f"sensor {sensor} source camera does not match manifest camera")
    return {image.image_id: manifest.assignments[image.name] for image in model.images}


def _validate_support(model: Model, fixture: Fixture, assignments: dict[int, tuple[int, int]]) -> None:
    by_id = {image.image_id: image for image in model.images}
    expected = fixture.observations
    index = 0
    for point in model.points:
        for image_id, keypoint_index in point.track:
            if index >= len(expected):
                raise PublisherError("source has more observations than fixture")
            image = by_id[image_id]
            frame, _ = assignments[image_id]
            keypoint = image.keypoints[keypoint_index]
            actual = (frame, point.point_id, keypoint.x, keypoint.y, image.camera_id)
            wanted = expected[index]
            candidate = (wanted[0], wanted[1], wanted[2], wanted[3], wanted[4])
            if actual[:2] != candidate[:2] or actual[4] != candidate[4] or _bits(actual[2]) != _bits(candidate[2]) or _bits(actual[3]) != _bits(candidate[3]):
                raise PublisherError(f"source observation order/identity differs at index {index}")
            index += 1
    if index != len(expected):
        raise PublisherError("source has fewer observations than fixture")


def _fixture_sensor_map(fixture: Fixture) -> dict[int, tuple[Quat, Vec]]:
    result: dict[int, tuple[Quat, Vec]] = {}
    for _, _, _, _, camera, q, t in fixture.observations:
        previous = result.setdefault(camera, (q, t))
        if not _quat_close(previous[0], q, 1.0e-12, 1.0e-12) or not _vec_close(previous[1], t, 1.0e-12, 1.0e-12):
            raise PublisherError("fixture sensor transform is not constant")
    return result


def _derive_rig_pose(image: Image, sensor: tuple[Quat, Vec]) -> tuple[Quat, Vec]:
    sensor_q, sensor_t = sensor
    inverse_q = _qconj(sensor_q)
    rig_q = _qnormalize(_qmul(inverse_q, image.q))
    rig_t = _qrotate(inverse_q, _vsub(image.t, sensor_t))
    return rig_q, rig_t


def _validate_source_poses(model: Model, manifest: Manifest, fixture: Fixture, assignments: dict[int, tuple[int, int]]) -> None:
    sensor_by_camera = _fixture_sensor_map(fixture)
    frame_pose: dict[int, tuple[Quat, Vec]] = {}
    for image in model.images:
        frame, sensor = assignments[image.image_id]
        derived = _derive_rig_pose(image, sensor_by_camera[image.camera_id])
        if frame in frame_pose and (not _quat_close(frame_pose[frame][0], derived[0], 1.0e-7, 1.0e-7) or not _vec_close(frame_pose[frame][1], derived[1], 1.0e-7, 1.0e-7)):
            raise PublisherError(f"source sensor poses disagree within rig frame {frame}")
        frame_pose.setdefault(frame, derived)
    if set(frame_pose) != set(fixture.poses):
        raise PublisherError("source rig frames differ from fixture pose IDs")
    for frame, pose in frame_pose.items():
        expected = fixture.poses[frame]
        if not _quat_close(pose[0], expected[0], 1.0e-6, 1.0e-6) or not _vec_close(pose[1], expected[1], 1.0e-6, 1.0e-6):
            raise PublisherError(f"source rig pose differs from fixture pose {frame}")


def _evaluate_model(model: Model, assignments: dict[int, tuple[int, int]], fixture: Fixture, poses: dict[int, tuple[Quat, Vec]], points: dict[int, Vec]) -> tuple[float, dict[int, float], int]:
    sensor_by_camera = _fixture_sensor_map(fixture)
    image_by_id = {image.image_id: image for image in model.images}
    point_sums: dict[int, list[float]] = {point_id: [0.0, 0.0] for point_id in points}
    total = 0.0
    observations = 0
    for point in model.points:
        point_xyz = points.get(point.point_id)
        if point_xyz is None:
            raise PublisherError(f"state is missing point {point.point_id}")
        for image_id, keypoint_index in point.track:
            image = image_by_id[image_id]
            frame, _ = assignments[image_id]
            pose = poses.get(frame)
            if pose is None:
                raise PublisherError(f"state is missing frame {frame}")
            keypoint = image.keypoints[keypoint_index]
            camera = model.cameras[image.camera_id]
            projected_x, projected_y, _ = _project(camera, pose[0], pose[1], point_xyz, *sensor_by_camera[image.camera_id])
            error = math.hypot(projected_x - keypoint.x, projected_y - keypoint.y)
            if not math.isfinite(error):
                raise PublisherError("reprojection error is non-finite")
            total += (projected_x - keypoint.x) ** 2 + (projected_y - keypoint.y) ** 2
            point_sums[point.point_id][0] += error
            point_sums[point.point_id][1] += 1.0
            observations += 1
    if not math.isfinite(total) or observations == 0:
        raise PublisherError("model evaluation is empty or non-finite")
    means = {point_id: values[0] / values[1] for point_id, values in point_sums.items()}
    if any(not math.isfinite(value) for value in means.values()):
        raise PublisherError("point mean ERROR is non-finite")
    return total, means, observations


def _validate_state_against_inputs(model: Model, manifest: Manifest, fixture: Fixture, state: State, assignments: dict[int, tuple[int, int]]) -> None:
    if state.fixed_pose != fixture.fixed_pose or state.fixed_pose != 0:
        raise PublisherError("state fixed pose does not match FIXED_POSE 0")
    expected_counts = {
        "CAMERA_COUNT": len(fixture.cameras),
        "POSE_COUNT": len(fixture.poses),
        "LANDMARK_COUNT": len(fixture.landmarks),
        "OBSERVATION_COUNT": len(fixture.observations),
    }
    if state.counts != expected_counts or len(model.points) != expected_counts["LANDMARK_COUNT"]:
        raise PublisherError("state/model/fixture counts differ")
    if set(state.poses) != set(fixture.poses) or set(state.landmarks) != set(fixture.landmarks):
        raise PublisherError("state IDs differ from fixture IDs")
    if not _quat_close(state.poses[0][0], fixture.poses[0][0], 1.0e-12, 1.0e-12) or not _vec_close(state.poses[0][1], fixture.poses[0][1], 1.0e-12, 1.0e-12):
        raise PublisherError("fixed anchor pose changed")
    source_poses = {frame: pose for frame, pose in fixture.poses.items()}
    source_cost, _, source_observations = _evaluate_model(model, assignments, fixture, source_poses, {point.point_id: point.xyz for point in model.points})
    if source_observations != len(fixture.observations) or not _close(source_cost, fixture.initial_cost, COST_TOL_ABS, COST_TOL_REL):
        raise PublisherError(f"source model cost disagrees with fixture ({source_cost} vs {fixture.initial_cost})")
    if not _close(state.initial_cost, source_cost, COST_TOL_ABS, COST_TOL_REL) or not _close(state.declared_initial_cost, fixture.initial_cost, COST_TOL_ABS, COST_TOL_REL):
        raise PublisherError("state initial cost disagrees with source/fixture")
    final_cost, _, observations = _evaluate_model(model, assignments, fixture, state.poses, state.landmarks)
    if observations != state.counts["OBSERVATION_COUNT"] or not _close(final_cost, state.final_cost, COST_TOL_ABS, COST_TOL_REL):
        raise PublisherError(f"state final cost disagrees with published model geometry ({final_cost} vs {state.final_cost})")
    if state.final_cost > state.initial_cost + COST_TOL_ABS + COST_TOL_REL * abs(state.initial_cost):
        raise PublisherError("state final cost is greater than its initial cost")


def _pose_tokens(pose: tuple[Quat, Vec]) -> list[str]:
    q, t = pose
    return [f"{value:.17g}" for value in (*q, *t)]


def _build_output(model: Model, assignments: dict[int, tuple[int, int]], fixture: Fixture, state: State) -> tuple[bytes, bytes, bytes]:
    image_lines = list(model.image_lines)
    sensor_by_camera = _fixture_sensor_map(fixture)
    for image in model.images:
        frame, _ = assignments[image.image_id]
        rig_q, rig_t = state.poses[frame]
        sensor_q, sensor_t = sensor_by_camera[image.camera_id]
        pose = (_qnormalize(_qmul(sensor_q, rig_q)), _vadd(_qrotate(sensor_q, rig_t), sensor_t))
        fields = image_lines[image.header_index].split()
        fields[1:8] = _pose_tokens(pose)
        line_ending = "\n" if image_lines[image.header_index].endswith("\n") else ""
        image_lines[image.header_index] = " ".join(fields) + line_ending
    _, means, _ = _evaluate_model(model, assignments, fixture, state.poses, state.landmarks)
    point_lines = list(model.point_lines)
    point_by_id = {point.point_id: point for point in model.points}
    for line_index, raw in enumerate(point_lines):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        fields = raw.split()
        point_id = int(fields[0], 10)
        if point_id not in point_by_id:
            raise PublisherError(f"cannot update unknown point {point_id}")
        xyz = state.landmarks[point_id]
        line_ending = "\n" if raw.endswith("\n") else ""
        point_fields = [fields[0], *(f"{value:.17g}" for value in xyz), *fields[4:7], f"{means[point_id]:.17g}", *fields[8:]]
        point_lines[line_index] = " ".join(point_fields) + line_ending
    return model.cameras_bytes, "".join(image_lines).encode("utf-8"), "".join(point_lines).encode("utf-8")


def _evaluate_serialized_model(model: Model) -> tuple[float, dict[int, float], int]:
    image_by_id = {image.image_id: image for image in model.images}
    point_sums: dict[int, list[float]] = {point.point_id: [0.0, 0.0] for point in model.points}
    total = 0.0
    observations = 0
    for point in model.points:
        for image_id, keypoint_index in point.track:
            image = image_by_id[image_id]
            keypoint = image.keypoints[keypoint_index]
            camera = model.cameras[image.camera_id]
            projected_x, projected_y, _ = _project(camera, image.q, image.t, point.xyz, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            dx = projected_x - keypoint.x
            dy = projected_y - keypoint.y
            error = math.hypot(dx, dy)
            total += dx * dx + dy * dy
            point_sums[point.point_id][0] += error
            point_sums[point.point_id][1] += 1.0
            observations += 1
    if not math.isfinite(total) or observations == 0:
        raise PublisherError("serialized model evaluation is empty or non-finite")
    return total, {point_id: value[0] / value[1] for point_id, value in point_sums.items()}, observations


def _validate_staged_model(stage: Path, source: Model, fixture: Fixture, state: State, assignments: dict[int, tuple[int, int]], expected_files: dict[str, bytes]) -> None:
    staged = parse_model(stage)
    if staged.cameras_bytes != expected_files["cameras.txt"]:
        raise PublisherError("staged camera bytes changed")
    _validate_support(staged, fixture, assignments)
    if len(staged.images) != len(source.images) or len(staged.points) != len(source.points):
        raise PublisherError("staged model record counts changed")
    for source_image, staged_image in zip(source.images, staged.images):
        if (source_image.image_id, source_image.name, source_image.camera_id) != (staged_image.image_id, staged_image.name, staged_image.camera_id):
            raise PublisherError(f"staged image identity/order changed for {source_image.image_id}")
        if len(source_image.keypoints) != len(staged_image.keypoints):
            raise PublisherError(f"staged POINTS2D count changed for {source_image.image_id}")
        for source_keypoint, staged_keypoint in zip(source_image.keypoints, staged_image.keypoints):
            if _bits(source_keypoint.x) != _bits(staged_keypoint.x) or _bits(source_keypoint.y) != _bits(staged_keypoint.y) or source_keypoint.point_id != staged_keypoint.point_id:
                raise PublisherError(f"staged POINTS2D tokens changed for {source_image.image_id}")
    for source_point, staged_point in zip(source.points, staged.points):
        if (source_point.point_id, source_point.rgb, source_point.track) != (staged_point.point_id, staged_point.rgb, staged_point.track):
            raise PublisherError(f"staged point identity/order changed for {source_point.point_id}")
    source_by_id = {image.image_id: image for image in source.images}
    staged_by_id = {image.image_id: image for image in staged.images}
    sensor_by_camera = _fixture_sensor_map(fixture)
    for image_id, source_image in source_by_id.items():
        staged_image = staged_by_id.get(image_id)
        if staged_image is None or staged_image.name != source_image.name or staged_image.camera_id != source_image.camera_id:
            raise PublisherError(f"staged image identity changed for {image_id}")
        frame, _ = assignments[image_id]
        rig_q, rig_t = state.poses[frame]
        sensor_q, sensor_t = sensor_by_camera[source_image.camera_id]
        expected_q = _qnormalize(_qmul(sensor_q, rig_q))
        expected_t = _vadd(_qrotate(sensor_q, rig_t), sensor_t)
        if not _quat_close(staged_image.q, expected_q, 1.0e-12, 1.0e-12) or not _vec_close(staged_image.t, expected_t, 1.0e-12, 1.0e-12):
            raise PublisherError(f"staged image pose does not equal sensor/rig composition for {image_id}")
    final_cost, means, observations = _evaluate_serialized_model(staged)
    if observations != state.counts["OBSERVATION_COUNT"] or not _close(final_cost, state.final_cost, COST_TOL_ABS, COST_TOL_REL):
        raise PublisherError("serialized model cost disagrees with Ceres final cost")
    for point in staged.points:
        expected = state.landmarks[point.point_id]
        if not _vec_close(point.xyz, expected, 1.0e-12, 1.0e-12) or not _close(point.error, means[point.point_id], 1.0e-10, 1.0e-10):
            raise PublisherError(f"staged landmark {point.point_id} changed or has invalid ERROR")


def _publish_files(out_dir: Path, files: dict[str, bytes], validator=None) -> None:
    parent = out_dir.parent
    try:
        metadata = parent.lstat()
    except OSError as error:
        raise PublisherError(f"output parent is unavailable: {parent}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise PublisherError("output parent must be a real directory")
    if os.path.lexists(out_dir):
        raise PublisherError(f"output path already exists: {out_dir}")
    stage = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.staging-", dir=parent))
    owns_stage = True
    final_owned = False
    published: list[tuple[Path, int, int]] = []
    try:
        for name, data in files.items():
            path = stage / name
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                stream = os.fdopen(fd, "wb")
                fd = None
                with stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException:
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                raise
        # Reparse the complete staged text model before reserving the final
        # path.  This is a bounded identity/finite check, not a second model
        # retained after publication, and catches serializer regressions.
        parse_model(stage)
        if validator is not None:
            validator(stage)
        if os.path.lexists(out_dir):
            raise PublisherError(f"output path appeared during staging: {out_dir}")
        try:
            os.mkdir(out_dir, 0o700)
        except FileExistsError as error:
            raise PublisherError(f"output path appeared during publication: {out_dir}") from error
        final_owned = True
        for name in files:
            source = stage / name
            destination = out_dir / name
            os.link(source, destination)
            source_stat = os.stat(source)
            published.append((destination, source_stat.st_dev, source_stat.st_ino))
        shutil.rmtree(stage)
        owns_stage = False
    except BaseException:
        if final_owned:
            for destination, device, inode in reversed(published):
                try:
                    metadata = destination.stat()
                    if metadata.st_dev == device and metadata.st_ino == inode:
                        destination.unlink()
                except OSError:
                    pass
            try:
                out_dir.rmdir()
            except OSError:
                pass
        if owns_stage:
            shutil.rmtree(stage, ignore_errors=True)
        raise


def publish_model(*, fixture_path: Path, state_path: Path, model_path: Path, rig_manifest_path: Path, out_dir: Path) -> None:
    input_paths = (fixture_path, state_path, model_path, rig_manifest_path)
    for path in (fixture_path, state_path, rig_manifest_path):
        _require_real_file(path, "input")
    try:
        model_metadata = model_path.lstat()
    except OSError as error:
        raise PublisherError(f"model input is unavailable: {model_path}: {error}") from error
    if stat.S_ISLNK(model_metadata.st_mode) or not stat.S_ISDIR(model_metadata.st_mode):
        raise PublisherError("model input must be a real directory")
    if out_dir.exists() or os.path.lexists(out_dir):
        raise PublisherError(f"output must be a new directory: {out_dir}")
    for left in input_paths:
        if _overlap(out_dir, left):
            raise PublisherError("output overlaps an input path")
    fixture = parse_fixture(fixture_path)
    state = parse_state(state_path)
    model = parse_model(model_path)
    manifest = parse_manifest(rig_manifest_path)
    _compare_hashes(fixture, state, model, rig_manifest_path)
    assignments = _validate_manifest_model(model, manifest, fixture)
    _validate_support(model, fixture, assignments)
    _validate_source_poses(model, manifest, fixture, assignments)
    _validate_state_against_inputs(model, manifest, fixture, state, assignments)
    camera_bytes, image_bytes, point_bytes = _build_output(model, assignments, fixture, state)
    files = {"cameras.txt": camera_bytes, "images.txt": image_bytes, "points3D.txt": point_bytes}
    _publish_files(
        out_dir,
        files,
        validator=lambda stage: _validate_staged_model(stage, model, fixture, state, assignments, files),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rig-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        publish_model(fixture_path=args.fixture, state_path=args.state, model_path=args.model, rig_manifest_path=args.rig_manifest, out_dir=args.out_dir)
    except (OSError, PublisherError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
