#!/usr/bin/env python3
"""Compare temporal track support and reprojection residuals in COLMAP text models."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator


IMAGE_NUMBER = re.compile(r"_(\d+)\.[^.]+$")


def track_span_class(span_frames: int) -> str:
    if span_frames == 0:
        return "same-frame"
    if span_frames <= 7:
        return "1-7"
    if span_frames <= 15:
        return "8-15"
    if span_frames <= 31:
        return "16-31"
    if span_frames <= 127:
        return "32-127"
    return "128+"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def quaternion_matrix(qw: float, qx: float, qy: float, qz: float) -> tuple[tuple[float, ...], ...]:
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid image quaternion")
    w, x, y, z = qw / norm, qx / norm, qy / norm, qz / norm
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def image_records(path: Path) -> Iterator[tuple[list[str], list[str]]]:
    with path.open(encoding="utf-8") as handle:
        while True:
            header = handle.readline()
            if not header:
                return
            if not header.strip() or header.startswith("#"):
                continue
            observations = handle.readline()
            if not observations:
                raise ValueError(f"missing POINTS2D line after image header in {path}")
            yield header.split(), observations.split()


def frame_from_name(name: str, aliases: dict[str, str] | None = None) -> int:
    if aliases is not None:
        name = aliases.get(name, name)
    match = IMAGE_NUMBER.search(name)
    if match is None:
        raise ValueError(f"image name has no numeric suffix: {name}")
    return int(match.group(1)) // 2


def read_cameras(path: Path) -> dict[int, tuple[str, list[float]]]:
    cameras: dict[int, tuple[str, list[float]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            cameras[int(fields[0])] = (fields[1], [float(value) for value in fields[4:]])
    return cameras


def read_image_metadata(path: Path, aliases: dict[str, str] | None = None) -> dict[int, dict[str, object]]:
    images: dict[int, dict[str, object]] = {}
    for fields, _ in image_records(path):
        if len(fields) < 10:
            raise ValueError(f"malformed image header in {path}")
        rotation = quaternion_matrix(*(float(value) for value in fields[1:5]))
        translation = tuple(float(value) for value in fields[5:8])
        centre = tuple(-sum(rotation[row][column] * translation[row] for row in range(3)) for column in range(3))
        images[int(fields[0])] = {
            "rotation": rotation,
            "translation": translation,
            "camera_id": int(fields[8]),
            "name": fields[9],
            "frame": frame_from_name(fields[9], aliases),
            "centre": centre,
        }
    return images


def project(camera: tuple[str, list[float]], image: dict[str, object], point: tuple[float, float, float]) -> tuple[float, float] | None:
    rotation = image["rotation"]
    translation = image["translation"]
    assert isinstance(rotation, tuple) and isinstance(translation, tuple)
    camera_point = tuple(sum(rotation[row][column] * point[column] for column in range(3)) + translation[row] for row in range(3))
    if camera_point[2] <= 1.0e-12 or not all(math.isfinite(value) for value in camera_point):
        return None
    model, params = camera
    if model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    elif model == "SIMPLE_PINHOLE":
        focal, cx, cy = params[:3]
        fx = fy = focal
    else:
        raise ValueError(f"unsupported camera model: {model}")
    return fx * camera_point[0] / camera_point[2] + cx, fy * camera_point[1] / camera_point[2] + cy


def summarize(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "median": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
    }


def analyze_component(
    model: Path, bin_frames: int, aliases: dict[str, str] | None = None
) -> dict[str, object]:
    cameras = read_cameras(model / "cameras.txt")
    images_path = model / "images.txt"
    images = read_image_metadata(images_path, aliases)
    points: dict[int, tuple[float, float, float]] = {}
    point_span_classes: dict[int, str] = {}
    track_bins: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    points_path = model / "points3D.txt"
    with points_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            point_id = int(fields[0])
            point = tuple(float(value) for value in fields[1:4])
            points[point_id] = point
            track_images = [images[int(fields[index])] for index in range(8, len(fields), 2) if int(fields[index]) in images]
            if len(track_images) < 2:
                continue
            frames = sorted(int(image["frame"]) for image in track_images)
            point_span_classes[point_id] = track_span_class(frames[-1] - frames[0])
            anchor_bin = frames[len(frames) // 2] // bin_frames
            first = min(track_images, key=lambda image: int(image["frame"]))
            last = max(track_images, key=lambda image: int(image["frame"]))
            rays = []
            for image in (first, last):
                centre = image["centre"]
                assert isinstance(centre, tuple)
                ray = tuple(point[index] - centre[index] for index in range(3))
                length = math.sqrt(sum(value * value for value in ray))
                rays.append(tuple(value / length for value in ray) if length > 1.0e-12 else (0.0, 0.0, 0.0))
            cosine = max(-1.0, min(1.0, sum(rays[0][index] * rays[1][index] for index in range(3))))
            track_bins[anchor_bin]["observations"].append(float(len(track_images)))
            track_bins[anchor_bin]["span_frames"].append(float(frames[-1] - frames[0]))
            track_bins[anchor_bin]["endpoint_angle_deg"].append(math.degrees(math.acos(cosine)))

    residual_bins: dict[int, list[float]] = defaultdict(list)
    residual_span_classes: dict[str, list[float]] = defaultdict(list)
    registered_bins: dict[int, int] = defaultdict(int)
    invalid_projections = 0
    for fields, observations in image_records(images_path):
        image = images[int(fields[0])]
        frame_bin = int(image["frame"]) // bin_frames
        registered_bins[frame_bin] += 1
        camera = cameras[int(image["camera_id"])]
        for index in range(0, len(observations), 3):
            point_id = int(observations[index + 2])
            if point_id < 0 or point_id not in points:
                continue
            prediction = project(camera, image, points[point_id])
            if prediction is None:
                invalid_projections += 1
                continue
            dx = prediction[0] - float(observations[index])
            dy = prediction[1] - float(observations[index + 1])
            residual = math.hypot(dx, dy)
            residual_bins[frame_bin].append(residual)
            if point_id in point_span_classes:
                residual_span_classes[point_span_classes[point_id]].append(residual)

    bins = []
    for frame_bin in sorted(set(registered_bins) | set(track_bins) | set(residual_bins)):
        tracks = track_bins[frame_bin]
        bins.append({
            "frame_start": frame_bin * bin_frames,
            "frame_end_exclusive": (frame_bin + 1) * bin_frames,
            "registered_images": registered_bins[frame_bin],
            "reprojection_px": summarize(residual_bins[frame_bin]),
            "tracks_anchored": len(tracks["observations"]),
            "track_observations": summarize(tracks["observations"]),
            "track_span_frames": summarize(tracks["span_frames"]),
            "endpoint_angle_deg": summarize(tracks["endpoint_angle_deg"]),
        })
    all_residuals = [value for values in residual_bins.values() for value in values]
    return {
        "model": str(model),
        "registered_images": len(images),
        "points": len(points),
        "invalid_projections": invalid_projections,
        "reprojection_px": summarize(all_residuals),
        "reprojection_by_track_span": {
            span_class: summarize(residual_span_classes[span_class])
            for span_class in ("same-frame", "1-7", "8-15", "16-31", "32-127", "128+")
        },
        "bins": bins,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--image-aliases", action="append", default=[], metavar="LABEL=TSV")
    parser.add_argument("--bin-frames", type=int, default=500)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.bin_frames <= 0:
        parser.error("--bin-frames must be positive")
    groups: dict[str, list[Path]] = defaultdict(list)
    for specification in args.model:
        if "=" not in specification:
            parser.error("--model must be LABEL=DIR")
        label, directory = specification.split("=", 1)
        groups[label].append(Path(directory))
    aliases_by_label: dict[str, dict[str, str]] = {}
    for specification in args.image_aliases:
        if "=" not in specification:
            parser.error("--image-aliases must be LABEL=TSV")
        label, filename = specification.split("=", 1)
        aliases: dict[str, str] = {}
        with Path(filename).open(encoding="utf-8") as handle:
            next(handle, None)
            for line in handle:
                flat_name, model_name = line.rstrip("\n").split("\t")
                aliases[model_name] = flat_name
        aliases_by_label[label] = aliases
    result = {
        "schema": "visloc_sfm_temporal_support_diagnostic_v1",
        "bin_frames": args.bin_frames,
        "ground_truth_used": False,
        "groups": {
            label: [
                analyze_component(model, args.bin_frames, aliases_by_label.get(label))
                for model in models
            ]
            for label, models in groups.items()
        },
    }
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
