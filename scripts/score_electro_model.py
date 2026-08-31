#!/usr/bin/env python3
"""Score an ETH3D ``electro`` COLMAP model by camera centre.

This is a post-mapping score only.  It maps both flat visloc/COLMAP names
(``cam4_<timestamp>.png``) and official ETH3D names
(``images_rig_cam4_undistorted/<timestamp>.png``) to the key
``(camera_number, timestamp)`` and then Umeyama-aligns the query centres to
the official reference centres.  The reference is never consumed by
candidate generation, matching, or mapping.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only on minimal hosts
    raise SystemExit("score_electro_model.py requires numpy") from exc


IMAGE_KEY_RE = re.compile(
    r"^(?:images_rig_)?cam(?P<camera>[0-9]+)(?:_undistorted)?[_/](?P<timestamp>[0-9]+)\.(?P<suffix>[^./]+)$"
)


class ScoreError(RuntimeError):
    """A reference or query model could not be scored."""


REFERENCE_EXTENT_EPSILON_M = 1e-9


def image_key(name: str) -> tuple[int, int]:
    normalized = name.replace("\\", "/")
    match = IMAGE_KEY_RE.fullmatch(normalized)
    if match is None:
        # The official path has the camera token in a directory and the flat
        # path has it at the beginning.  Keep this fallback strict enough to
        # avoid accidentally pairing unrelated numeric filenames.
        match = re.search(
            r"(?:^|/)images_rig_cam(?P<camera>[0-9]+)_undistorted/(?P<timestamp>[0-9]+)\.[^/]+$",
            normalized,
        )
    if match is None:
        match = re.search(r"(?:^|/)cam(?P<camera>[0-9]+)_(?P<timestamp>[0-9]+)\.[^/]+$", normalized)
    if match is None:
        raise ScoreError(f"unsupported electro image name: {name!r}")
    return int(match.group("camera")), int(match.group("timestamp"))


def _rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if not math.isfinite(norm) or norm == 0:
        raise ScoreError("COLMAP pose has a zero or non-finite quaternion")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def load_centres(path: Path) -> dict[tuple[int, int], np.ndarray]:
    """Load camera centres from a COLMAP text ``images.txt``."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ScoreError(f"cannot read COLMAP images model {path}: {exc}") from exc
    centres: dict[tuple[int, int], np.ndarray] = {}
    expect_points = False
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if expect_points:
            # COLMAP stores one POINTS2D row immediately after every pose.
            # It is intentionally ignored.  If a model omits that row, the
            # next pose would be skipped; valid text models always include it.
            expect_points = False
            continue
        fields = line.split()
        if len(fields) < 10:
            raise ScoreError(f"COLMAP pose row {path}:{line_number} is malformed")
        try:
            int(fields[0])
            qw, qx, qy, qz = (float(value) for value in fields[1:5])
            t = np.array([float(value) for value in fields[5:8]], dtype=float)
            int(fields[8])
        except ValueError as exc:
            raise ScoreError(f"COLMAP pose row {path}:{line_number} is not numeric") from exc
        name = fields[9]
        key = image_key(name)
        if key in centres:
            raise ScoreError(f"COLMAP model contains duplicate electro image key {key}: {path}")
        rotation = _rotation(qw, qx, qy, qz)
        centres[key] = -rotation.T @ t
        expect_points = True
    if not centres:
        raise ScoreError(f"COLMAP model contains no registered electro images: {path}")
    return centres


def umeyama(source: np.ndarray, destination: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return scale, rotation, translation mapping source to destination."""

    if len(source) < 3:
        raise ScoreError("at least three common camera centres are required")
    source_mean = source.mean(axis=0)
    destination_mean = destination.mean(axis=0)
    centered_source = source - source_mean
    centered_destination = destination - destination_mean
    covariance = centered_destination.T @ centered_source / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[2, 2] = -1
    rotation = u @ correction @ vt
    variance = float((centered_source**2).sum() / len(source))
    if not math.isfinite(variance) or variance <= 0:
        raise ScoreError("common query camera centres are degenerate")
    scale = float(np.trace(np.diag(singular) @ correction) / variance)
    translation = destination_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def validate_reference_geometry(reference: dict[tuple[int, int], np.ndarray]) -> float:
    """Reject calibration-only/identity pose files before Sim(3) scoring.

    ETH3D's staging calibration helper can intentionally emit identity dummy
    poses for intrinsics lookup.  Treating that file as a reference would
    make a query with the same dummy poses appear to score perfectly.  A real
    electro reference has a non-zero camera-centre extent, so fail closed on
    an all-collapsed reference before looking at query overlap.
    """

    centres = np.asarray(list(reference.values()), dtype=float)
    if centres.ndim != 2 or centres.shape[1] != 3 or not np.isfinite(centres).all():
        raise ScoreError("reference camera centres contain non-finite or malformed values")
    extent = float(np.linalg.norm(centres.max(axis=0) - centres.min(axis=0)))
    if not math.isfinite(extent) or extent <= REFERENCE_EXTENT_EPSILON_M:
        raise ScoreError(
            "reference camera centres are degenerate (near-zero extent); "
            "pass the official electro rig poses, not an identity staging calibration"
        )
    return extent


def score(reference_path: Path, query_path: Path) -> dict[str, Any]:
    reference = load_centres(reference_path)
    reference_extent = validate_reference_geometry(reference)
    query = load_centres(query_path)
    common = sorted(set(reference) & set(query))
    if len(common) < 3:
        raise ScoreError(
            f"only {len(common)} common electro cameras; at least three are required for Sim(3)"
        )
    source = np.array([query[key] for key in common], dtype=float)
    destination = np.array([reference[key] for key in common], dtype=float)
    scale, rotation, translation = umeyama(source, destination)
    aligned = (scale * (rotation @ source.T).T) + translation
    errors = np.linalg.norm(aligned - destination, axis=1)
    extent = reference_extent
    by_camera: dict[str, dict[str, Any]] = {}
    for camera in sorted({key[0] for key in common}):
        camera_errors = errors[[key[0] == camera for key in common]]
        by_camera[str(camera)] = {
            "common": int(len(camera_errors)),
            "rmse_m": float(np.sqrt(np.mean(camera_errors**2))),
            "median_m": float(np.median(camera_errors)),
            "p95_m": float(np.percentile(camera_errors, 95)),
            "max_m": float(np.max(camera_errors)),
        }
    return {
        "schema": "visloc_electro_model_score_v1",
        "reference_images": str(reference_path.resolve()),
        "query_images": str(query_path.resolve()),
        "reference_registered": len(reference),
        "query_registered": len(query),
        "common": len(common),
        "missing_query": len(set(reference) - set(query)),
        "missing_reference": len(set(query) - set(reference)),
        "sim3_scale": scale,
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "median_m": float(np.median(errors)),
        "p95_m": float(np.percentile(errors, 95)),
        "max_m": float(np.max(errors)),
        "reference_extent_m": extent,
        "relative_rmse": float(np.sqrt(np.mean(errors**2)) / extent) if extent > 0 else None,
        "by_camera": by_camera,
        "reference_used_only_for_post_mapping_score": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="official electro model images.txt")
    parser.add_argument("query", type=Path, help="completed COLMAP model images.txt")
    parser.add_argument("--output-json", type=Path, help="write the score object to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = score(args.reference.resolve(), args.query.resolve())
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except ScoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
