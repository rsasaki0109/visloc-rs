#!/usr/bin/env python3
"""Score OpenLORIS COLMAP text models against timestamp-interpolated official GT.

Ground truth is consumed only by this post-mapping scorer.  The official GT
describes ``base_link``; official ``trans_matrix.yaml`` supplies the camera
lever arms.  Each disconnected reconstruction has its own gauge, so it is
Sim(3)-aligned independently and the published aggregate is observation-
weighted across components.  Per-component scores remain in the result to
make that convention explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("score_openloris_model.py requires numpy") from exc


class ScoreError(RuntimeError):
    """An OpenLORIS input or reconstruction cannot be scored."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if not math.isfinite(norm) or norm == 0:
        raise ScoreError("pose has a zero or non-finite quaternion")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    return np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def _slerp_xyzw(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    dot = float(left @ right)
    if dot < 0:
        right = -right
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        result = left + fraction * (right - left)
        return result / np.linalg.norm(result)
    angle = math.acos(dot)
    denominator = math.sin(angle)
    return (
        math.sin((1.0 - fraction) * angle) / denominator * left
        + math.sin(fraction * angle) / denominator * right
    )


def load_manifest(path: Path) -> dict[str, tuple[int, float]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["images"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as exc:
        raise ScoreError(f"cannot read OpenLORIS manifest {path}: {exc}") from exc
    result: dict[str, tuple[int, float]] = {}
    if payload.get("schema") != "visloc_openloris_corridor_manifest_v1" or not isinstance(rows, list):
        raise ScoreError(f"unsupported OpenLORIS manifest schema in {path}")
    for row in rows:
        try:
            name, camera, timestamp = row["name"], int(row["camera"]), float(row["timestamp"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ScoreError(f"malformed image row in {path}") from exc
        if not isinstance(name, str) or not name or name in result or camera not in (1, 2):
            raise ScoreError(f"invalid or duplicate image in {path}: {name!r}")
        result[name] = camera, timestamp
    if not result:
        raise ScoreError(f"OpenLORIS manifest is empty: {path}")
    return result


def load_ground_truth(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps, positions, quaternions = [], [], []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ScoreError(f"cannot read OpenLORIS ground truth {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 8:
            raise ScoreError(f"malformed ground-truth row {path}:{line_number}")
        try:
            values = [float(value) for value in fields]
        except ValueError as exc:
            raise ScoreError(f"non-numeric ground-truth row {path}:{line_number}") from exc
        timestamps.append(values[0])
        positions.append(values[1:4])
        quaternions.append(values[4:8])  # x, y, z, w
    times = np.asarray(timestamps, dtype=float)
    if len(times) < 2 or not np.all(np.diff(times) > 0):
        raise ScoreError("ground-truth timestamps must be strictly increasing")
    return times, np.asarray(positions, dtype=float), np.asarray(quaternions, dtype=float)


def load_camera_extrinsics(path: Path) -> dict[int, np.ndarray]:
    try:
        payload = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ScoreError(f"cannot read official transform matrix {path}: {exc}") from exc
    pattern = re.compile(
        r"parent_frame:\s*(\S+)\s+child_frame:\s*(\S+)\s+matrix:.*?"
        r"data:\s*\[(.*?)\]",
        re.DOTALL,
    )
    transforms: dict[tuple[str, str], np.ndarray] = {}
    for match in pattern.finditer(payload):
        parent, child, raw_values = match.groups()
        try:
            values = [float(value.strip()) for value in raw_values.split(",") if value.strip()]
        except ValueError as exc:
            raise ScoreError(f"non-numeric transform {parent} -> {child} in {path}") from exc
        if len(values) != 16:
            raise ScoreError(f"malformed transform {parent} -> {child} in {path}")
        transforms[parent, child] = np.asarray(values, dtype=float).reshape(4, 4)
    try:
        base_cam1 = transforms["base_link", "t265_fisheye1_optical_frame"]
        cam1_cam2 = transforms[
            "t265_fisheye1_optical_frame", "t265_fisheye2_optical_frame"
        ]
    except KeyError as exc:
        raise ScoreError(f"required T265 transform is absent from {path}: {exc}") from exc
    return {1: base_cam1, 2: base_cam1 @ cam1_cam2}


def interpolate_camera_centres(
    manifest: dict[str, tuple[int, float]],
    ground_truth: tuple[np.ndarray, np.ndarray, np.ndarray],
    extrinsics: dict[int, np.ndarray],
    *,
    max_gap_seconds: float,
) -> dict[str, np.ndarray]:
    times, positions, quaternions = ground_truth
    result: dict[str, np.ndarray] = {}
    for name, (camera, timestamp) in manifest.items():
        right = int(np.searchsorted(times, timestamp, side="left"))
        if right == 0 or right == len(times):
            continue
        left = right - 1
        gap = float(times[right] - times[left])
        if gap <= 0 or gap > max_gap_seconds:
            continue
        fraction = float((timestamp - times[left]) / gap)
        position = positions[left] + fraction * (positions[right] - positions[left])
        quaternion = _slerp_xyzw(quaternions[left], quaternions[right], fraction)
        rotation = _rotation(quaternion[3], quaternion[0], quaternion[1], quaternion[2])
        result[name] = position + rotation @ extrinsics[camera][:3, 3]
    return result


def load_model_centres(path: Path) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        # Pose rows have exactly ten fields. POINTS2D rows contain triples.
        if len(fields) != 10:
            continue
        try:
            int(fields[0])
            qw, qx, qy, qz = (float(value) for value in fields[1:5])
            translation = np.asarray([float(value) for value in fields[5:8]])
            int(fields[8])
        except ValueError as exc:
            raise ScoreError(f"non-numeric COLMAP pose row {path}:{line_number}") from exc
        name = fields[9]
        if name in result:
            raise ScoreError(f"duplicate image {name!r} in {path}")
        result[name] = -_rotation(qw, qx, qy, qz).T @ translation
    if not result:
        raise ScoreError(f"COLMAP model contains no poses: {path}")
    return result


def load_colmap_aliases(path: Path) -> dict[str, str]:
    """Load ``original_name<TAB>colmap_name`` and return COLMAP -> original."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ScoreError(f"cannot read image aliases {path}: {exc}") from exc
    aliases: dict[str, str] = {}
    for line_number, raw in enumerate(lines, 1):
        if not raw.strip() or raw.startswith("#"):
            continue
        fields = raw.split("\t")
        if fields in (["original_name", "colmap_name"], ["flat_name", "colmap_name"]):
            continue
        if len(fields) != 2 or not all(fields):
            raise ScoreError(f"malformed image alias row {path}:{line_number}")
        original, colmap_name = fields
        if colmap_name in aliases or any(char.isspace() for char in colmap_name):
            raise ScoreError(f"duplicate or invalid COLMAP alias {colmap_name!r} in {path}")
        aliases[colmap_name] = original
    if not aliases:
        raise ScoreError(f"image alias map is empty: {path}")
    return aliases


def umeyama(source: np.ndarray, destination: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean, destination_mean = source.mean(axis=0), destination.mean(axis=0)
    centered_source, centered_destination = source - source_mean, destination - destination_mean
    covariance = centered_destination.T @ centered_source / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[2, 2] = -1
    rotation = u @ correction @ vt
    variance = float((centered_source**2).sum() / len(source))
    if not math.isfinite(variance) or variance <= 1e-12:
        raise ScoreError("query camera centres are degenerate")
    scale = float(np.trace(np.diag(singular) @ correction) / variance)
    return scale, rotation, destination_mean - scale * rotation @ source_mean


def score_component(
    path: Path,
    reference: dict[str, np.ndarray],
    aliases: dict[str, str] | None = None,
) -> tuple[dict[str, Any], np.ndarray, list[str]]:
    query_raw = load_model_centres(path)
    query = {aliases.get(name, name) if aliases else name: centre for name, centre in query_raw.items()}
    if len(query) != len(query_raw):
        raise ScoreError(f"image aliases collapse multiple model names in {path}")
    common = sorted(set(query) & set(reference))
    if len(common) < 3:
        raise ScoreError(f"model {path} has only {len(common)} GT-scored images")
    source = np.asarray([query[name] for name in common])
    destination = np.asarray([reference[name] for name in common])
    scale, rotation, translation = umeyama(source, destination)
    aligned = scale * (rotation @ source.T).T + translation
    errors = np.linalg.norm(aligned - destination, axis=1)
    extent = float(np.linalg.norm(destination.max(axis=0) - destination.min(axis=0)))
    return {
        "images_txt": str(path.resolve()),
        "images_txt_sha256": sha256_file(path),
        "registered": len(query_raw),
        "gt_scored": len(common),
        "sim3_scale": scale,
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "median_m": float(np.median(errors)),
        "p95_m": float(np.percentile(errors, 95)),
        "max_m": float(np.max(errors)),
        "reference_extent_m": extent,
        "relative_rmse": float(np.sqrt(np.mean(errors**2)) / extent) if extent > 0 else None,
    }, errors, common


def temporal_error_segments(
    scored: list[tuple[str, float]],
    manifest: dict[str, tuple[int, float]],
    *,
    bins: int = 10,
) -> list[dict[str, Any]]:
    """Summarize aligned errors in deterministic equal-image temporal bins."""
    if bins <= 0:
        raise ScoreError("temporal segment count must be positive")
    ordered = sorted(scored, key=lambda item: (manifest[item[0]][1], item[0]))
    timestamps_all = np.asarray([manifest[name][1] for name, _ in ordered], dtype=float)
    boundaries = np.quantile(timestamps_all, np.linspace(0.0, 1.0, bins + 1))
    grouped: list[list[tuple[str, float]]] = [[] for _ in range(bins)]
    for row in ordered:
        timestamp = manifest[row[0]][1]
        index = int(np.searchsorted(boundaries[1:-1], timestamp, side="left"))
        grouped[index].append(row)
    segments: list[dict[str, Any]] = []
    for index, rows in enumerate(grouped):
        if not rows:
            continue
        errors = np.asarray([error for _, error in rows], dtype=float)
        timestamps = [manifest[name][1] for name, _ in rows]
        segments.append(
            {
                "segment": index,
                "images": len(rows),
                "timestamp_start": min(timestamps),
                "timestamp_end": max(timestamps),
                "rmse_m": float(np.sqrt(np.mean(errors**2))),
                "median_m": float(np.median(errors)),
                "p95_m": float(np.percentile(errors, 95)),
                "max_m": float(np.max(errors)),
            }
        )
    return segments


def score(
    model_paths: list[Path], manifest_path: Path, ground_truth_path: Path,
    transform_path: Path, *, alias_path: Path | None = None, max_gap_seconds: float = 0.1,
) -> dict[str, Any]:
    if max_gap_seconds <= 0:
        raise ScoreError("max interpolation gap must be positive")
    manifest = load_manifest(manifest_path)
    reference = interpolate_camera_centres(
        manifest, load_ground_truth(ground_truth_path), load_camera_extrinsics(transform_path),
        max_gap_seconds=max_gap_seconds,
    )
    aliases = load_colmap_aliases(alias_path) if alias_path else None
    components, all_errors, scored_errors, registered_names = [], [], [], set()
    for path in model_paths:
        raw_names = set(load_model_centres(path))
        names = {aliases.get(name, name) if aliases else name for name in raw_names}
        if len(names) != len(raw_names):
            raise ScoreError(f"image aliases collapse multiple model names in {path}")
        duplicates = registered_names & names
        if duplicates:
            raise ScoreError(f"images occur in multiple models; first duplicate={min(duplicates)!r}")
        registered_names.update(names)
        component, errors, scored_names = score_component(path, reference, aliases)
        component["trajectory_segments"] = temporal_error_segments(
            list(zip(scored_names, (float(value) for value in errors))), manifest
        )
        components.append(component)
        all_errors.append(errors)
        scored_errors.extend(zip(scored_names, (float(value) for value in errors)))
    if not components:
        raise ScoreError("no COLMAP text models were supplied")
    errors = np.concatenate(all_errors)
    return {
        "schema": "visloc_openloris_model_score_v1",
        "scorer_sha256": sha256_file(Path(__file__).resolve()),
        "score_convention": "per-component Sim(3), image-weighted aggregate",
        "max_interpolation_gap_seconds": max_gap_seconds,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "ground_truth": str(ground_truth_path.resolve()),
        "ground_truth_sha256": sha256_file(ground_truth_path),
        "transform_matrix": str(transform_path.resolve()),
        "transform_matrix_sha256": sha256_file(transform_path),
        "image_aliases": str(alias_path.resolve()) if alias_path else None,
        "image_aliases_sha256": sha256_file(alias_path) if alias_path else None,
        "manifest_images": len(manifest),
        "gt_interpolated_images": len(reference),
        "registered_images": len(registered_names),
        "gt_scored_images": int(len(errors)),
        "models": len(components),
        "component_weighted_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "component_weighted_median_m": float(np.median(errors)),
        "component_weighted_p95_m": float(np.percentile(errors, 95)),
        "component_weighted_max_m": float(np.max(errors)),
        "trajectory_segments": temporal_error_segments(scored_errors, manifest),
        "components": components,
        "ground_truth_used_only_for_post_mapping_score": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-images", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--transform-matrix", type=Path, required=True)
    parser.add_argument("--image-aliases", type=Path)
    parser.add_argument("--max-gap-seconds", type=float, default=0.1)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = score(
            [path.resolve() for path in args.model_images], args.manifest.resolve(),
            args.ground_truth.resolve(), args.transform_matrix.resolve(),
            alias_path=args.image_aliases.resolve() if args.image_aliases else None,
            max_gap_seconds=args.max_gap_seconds,
        )
        payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0
    except (OSError, ValueError, ScoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
