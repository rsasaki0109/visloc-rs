#!/usr/bin/env python3
"""Validate or rerun the reproducible high-resolution courtyard control.

The benchmark's large inputs and model outputs intentionally live outside the
repository.  This runner validates their manifests before doing any work,
then optionally runs the normal visloc mapper and scores its *completed*
model against the supplied calibration model.  The calibration model is a
score reference only; it is never passed to the mapper as ground truth.

The default is the fast, read-only ``--verify-only`` mode.  Use ``--full``
for a fresh mapping run.  Progress and child-process output go to stderr/log
files, while the final stdout value is deterministic JSON suitable for CI or
another program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "benchmarks" / "courtyard" / "exhaustive_control.json"
DEFAULT_ARTIFACT_ROOT_ENV = "COURTYARD_ARTIFACT_ROOT"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
SCORE_MATCH_RE = re.compile(r"matched=(\d+)/(\d+)\s+est_registered=(\d+)")
SCORE_RMSE_RE = re.compile(r"^rmse_m=([0-9eE+.-]+)", re.MULTILINE)
CANDIDATE_MANIFEST_MAGIC = "visloc_candidate_manifest_v1"
VIEW_GRAPH_RE = re.compile(r"view graph:\s+(\d+)\s+candidate pairs")
VERIFIED_RE = re.compile(r"verified\s+(\d+)\s*/\s*(\d+)\s+pairs,\s+(\d+)\s+inlier correspondences")
RECONSTRUCTION_RE = re.compile(
    r"reconstruction .*?:\s+(\d+)\s*/\s*(\d+)\s+images registered,\s+(\d+)\s+tracks,\s+mean reproj\s+([0-9eE+.-]+)\s+px"
)
WRITTEN_MODEL_RE = re.compile(
    r"wrote COLMAP model to\s+(.+?)\s+\((\d+)\s+images,\s+(\d+)\s+points,\s+(\d+)\s+observations\)"
)
TRANSITIVE_RE = re.compile(r"transitive expansion:\s+(\d+)\s+new pairs")
RUNTIME_RE = re.compile(r"^elapsed=([0-9eE+.-]+)\s+maxrss=([0-9]+)", re.MULTILINE)


class ValidationError(RuntimeError):
    """An input or result failed a reproducibility gate."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all in memory."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValidationError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def _required_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def validate_hashed_file(path: Path, expected_sha256: str, label: str) -> str:
    """Check existence and exact digest, returning the verified digest."""

    if not path.is_file():
        raise ValidationError(f"{label} is missing: {path}; provide the durable artifact or override its root")
    expected = _required_sha(expected_sha256, f"{label} expected hash")
    actual = sha256_file(path)
    if actual != expected:
        raise ValidationError(
            f"{label} hash mismatch for {path}: expected {expected}, got {actual}; "
            "use the artifact version recorded by the benchmark manifest"
        )
    return actual


def load_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate a benchmark manifest."""

    if not path.is_file():
        raise ValidationError(f"benchmark config not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot parse benchmark config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationError(f"benchmark config must contain a JSON object: {path}")
    if data.get("schema_version") != 1:
        raise ValidationError(f"unsupported benchmark config schema_version: {data.get('schema_version')!r}")
    if not isinstance(data.get("benchmark"), dict) or not data["benchmark"].get("id"):
        raise ValidationError("benchmark config is missing benchmark.id")
    for section in ("inputs", "models", "mapping", "visuals"):
        if not isinstance(data.get(section), dict):
            raise ValidationError(f"benchmark config is missing object section {section!r}")
    return data


def resolve_artifact_root(config: dict[str, Any], override: Path | None) -> Path:
    configured = config.get("artifact_root")
    value: str | Path | None = override
    if value is None:
        value = os.environ.get(DEFAULT_ARTIFACT_ROOT_ENV) or configured
    if value is None:
        raise ValidationError(
            f"no artifact root configured; pass --artifact-root or set {DEFAULT_ARTIFACT_ROOT_ENV}"
        )
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise ValidationError(
            f"artifact root does not exist: {root}; pass --artifact-root pointing at the durable courtyard artifact"
        )
    return root


def resolve_path(root: Path, value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    # Relative paths in the manifest are rooted at the external artifact.
    return (root / path).resolve()


def _safe_manifest_relative(value: str, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not value or any(part in ("", ".", "..") for part in path.parts):
        raise ValidationError(f"{label} must be a simple relative path, got {value!r}")
    return path


def _noncomment_lines(path: Path) -> list[str]:
    try:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(f"cannot read text file {path}: {exc}") from exc


def parse_feature_manifest(path: Path) -> list[dict[str, Any]]:
    """Parse a ``file rows sha256`` feature manifest."""

    if not path.is_file():
        raise ValidationError(f"feature manifest is missing: {path}")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(f"cannot read feature manifest {path}: {exc}") from exc
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValidationError(f"feature manifest {path}:{line_number} must contain file, rows, sha256")
        relative = _safe_manifest_relative(fields[0], f"feature manifest {path}:{line_number} path")
        name = relative.as_posix()
        if name in seen:
            raise ValidationError(f"feature manifest repeats {name!r} at {path}:{line_number}")
        seen.add(name)
        try:
            rows = int(fields[1])
        except ValueError as exc:
            raise ValidationError(f"feature manifest {path}:{line_number} has non-integer row count") from exc
        if rows < 0:
            raise ValidationError(f"feature manifest {path}:{line_number} has negative row count")
        entries.append({"path": relative, "rows": rows, "sha256": _required_sha(fields[2], f"feature manifest {path}:{line_number}")})
    if not entries:
        raise ValidationError(f"feature manifest is empty: {path}")
    return entries


def _count_data_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return sum(1 for line in stream if line.strip() and not line.lstrip().startswith("#"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(f"cannot count rows in {path}: {exc}") from exc


def validate_features(root: Path, spec: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    manifest_spec = spec.get("manifest")
    if not isinstance(manifest_spec, dict) or not isinstance(manifest_spec.get("path"), str):
        raise ValidationError("inputs.features.manifest.path is required")
    manifest_path = resolve_path(root, manifest_spec["path"], label="feature manifest")
    validate_hashed_file(manifest_path, manifest_spec.get("sha256"), "feature manifest")
    entries = parse_feature_manifest(manifest_path)
    expected_count = spec.get("file_count")
    if expected_count is not None and len(entries) != expected_count:
        raise ValidationError(f"feature file count mismatch: expected {expected_count}, got {len(entries)}")
    expected_total = spec.get("total_rows")
    total_rows = 0
    row_counts: dict[int, int] = {}
    feature_names: list[str] = []
    suffix = "_features.txt"
    for entry in entries:
        path = (manifest_path.parent / entry["path"]).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ValidationError(f"feature manifest entry escapes artifact root: {entry['path']}") from exc
        label = f"feature file {entry['path']}"
        validate_hashed_file(path, entry["sha256"], label)
        rows = _count_data_rows(path)
        if rows != entry["rows"]:
            raise ValidationError(f"{label} row count mismatch: manifest {entry['rows']}, actual {rows}")
        total_rows += rows
        row_counts[len(feature_names)] = rows
        filename = Path(entry["path"]).name
        if not filename.endswith(suffix):
            raise ValidationError(f"feature file does not use {suffix!r}: {filename}")
        feature_names.append(filename[: -len(suffix)])
    if expected_total is not None and total_rows != expected_total:
        raise ValidationError(f"feature row total mismatch: expected {expected_total}, got {total_rows}")
    return feature_names, {"file_count": len(entries), "total_rows": total_rows, "manifest_sha256": manifest_spec["sha256"], "row_counts": row_counts}


def _expected_image_names(feature_stems: Iterable[str], image_suffix: str) -> list[str]:
    return [f"{stem}{image_suffix}" for stem in feature_stems]


def validate_images(directory: Path, spec: dict[str, Any], expected_names: list[str]) -> dict[str, Any]:
    if not directory.is_dir():
        raise ValidationError(f"image directory is missing: {directory}; pass --images-dir to the official source images")
    suffix = spec.get("suffix", ".JPG")
    hashes = spec.get("sha256")
    if not isinstance(hashes, dict):
        raise ValidationError("inputs.images.sha256 must map exact image names to hashes")
    expected_set = set(hashes)
    if set(expected_names) != expected_set:
        raise ValidationError("feature manifest image names do not match the configured image hash manifest")
    actual_names = {path.name for path in directory.iterdir() if path.is_file() and path.name.endswith(suffix)}
    if actual_names != expected_set:
        missing = sorted(expected_set - actual_names)
        extra = sorted(actual_names - expected_set)
        raise ValidationError(f"image set mismatch in {directory}: missing={missing[:5]} extra={extra[:5]}")
    for name in expected_names:
        validate_hashed_file(directory / name, hashes[name], f"source image {name}")
    expected_count = spec.get("file_count")
    if expected_count is not None and len(actual_names) != expected_count:
        raise ValidationError(f"source image count mismatch: expected {expected_count}, got {len(actual_names)}")
    return {"directory": str(directory), "file_count": len(actual_names), "suffix": suffix}


def _next_nonempty(lines: Iterable[str], label: str) -> str:
    for raw in lines:
        line = raw.strip()
        if line and not line.startswith("#"):
            return line
    raise ValidationError(f"{label} is truncated")


def validate_matches(path: Path, spec: dict[str, Any], feature_stems: list[str], image_suffix: str, row_counts: dict[int, int] | None = None) -> dict[str, Any]:
    expected_sha = spec.get("sha256")
    validate_hashed_file(path, expected_sha, "raw match import")
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"cannot open raw match import {path}: {exc}") from exc
    with stream:
        lines = (line for line in stream)
        try:
            image_count = int(_next_nonempty(lines, "raw match import image count"))
        except ValueError as exc:
            raise ValidationError("raw match import image count is not an integer") from exc
        names = [_next_nonempty(lines, "raw match import image names") for _ in range(image_count)]
        expected_names = _expected_image_names(feature_stems, image_suffix)
        if names != expected_names:
            raise ValidationError("raw match import image names/order differ from the feature manifest")
        try:
            pair_count = int(_next_nonempty(lines, "raw match import pair count"))
        except ValueError as exc:
            raise ValidationError("raw match import pair count is not an integer") from exc
        seen_pairs: set[tuple[int, int]] = set()
        raw_matches = 0
        for pair_number in range(pair_count):
            header = _next_nonempty(lines, f"raw match import pair {pair_number} header").split()
            if len(header) != 3:
                raise ValidationError(f"raw match import pair {pair_number} header must be i j count")
            try:
                first, second, count = (int(value) for value in header)
            except ValueError as exc:
                raise ValidationError(f"raw match import pair {pair_number} header is not numeric") from exc
            if not (0 <= first < image_count and 0 <= second < image_count) or first == second:
                raise ValidationError(f"raw match import pair {pair_number} has invalid image indices")
            pair = tuple(sorted((first, second)))
            if pair in seen_pairs:
                raise ValidationError(f"raw match import repeats pair {pair}")
            seen_pairs.add(pair)
            if count < 0:
                raise ValidationError(f"raw match import pair {pair} has negative match count")
            for match_number in range(count):
                fields = _next_nonempty(lines, f"raw match import pair {pair} match {match_number}").split()
                if len(fields) != 2:
                    raise ValidationError(f"raw match import pair {pair} contains a malformed correspondence")
                try:
                    first_index = int(fields[0])
                    second_index = int(fields[1])
                except ValueError as exc:
                    raise ValidationError(f"raw match import pair {pair} contains a non-numeric correspondence") from exc
                if row_counts is not None:
                    if first not in row_counts or second not in row_counts:
                        raise ValidationError(f"raw match import pair {pair} has no matching feature row-count entry")
                    if not 0 <= first_index < row_counts[first] or not 0 <= second_index < row_counts[second]:
                        raise ValidationError(f"raw match import pair {pair} contains a feature index outside its manifest row count")
            raw_matches += count
        trailing = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
        if trailing:
            raise ValidationError(f"raw match import has unexpected trailing data after {pair_count} pairs")
    expected_pairs = spec.get("pair_count")
    if expected_pairs is not None and pair_count != expected_pairs:
        raise ValidationError(f"raw match pair count mismatch: expected {expected_pairs}, got {pair_count}")
    expected_raw = spec.get("raw_match_count")
    if expected_raw is not None and raw_matches != expected_raw:
        raise ValidationError(f"raw match correspondence count mismatch: expected {expected_raw}, got {raw_matches}")
    if spec.get("candidate_semantics", "").startswith("all unordered"):
        expected_set = set(combinations(range(image_count), 2))
        if seen_pairs != expected_set:
            raise ValidationError("raw match import is not the complete unordered candidate set")
    return {"pair_count": pair_count, "raw_match_count": raw_matches, "sha256": expected_sha}


def parse_candidate_manifest(path: Path, expected_names: list[str]) -> dict[str, Any]:
    """Parse the image-name-bound candidate schedule emitted by the Rust demo.

    Candidate manifests deliberately contain no matches or verification data;
    they are safe to select before the expensive local matcher runs.  Keep this
    parser structurally identical to the Rust reader so a manifest can be
    validated in a cheap Python-only job before it is handed to the mapper.
    """

    lines = _noncomment_lines(path)
    cursor = 0

    def next_line(label: str) -> str:
        nonlocal cursor
        if cursor >= len(lines):
            raise ValidationError(f"candidate manifest {path} is truncated while reading {label}")
        line = lines[cursor]
        cursor += 1
        return line

    if next_line("header") != CANDIDATE_MANIFEST_MAGIC:
        raise ValidationError(f"candidate manifest {path} has unsupported header")
    image_header = next_line("image count").split()
    if len(image_header) != 2 or image_header[0] != "images":
        raise ValidationError(f"candidate manifest {path} image count must be images N")
    try:
        image_count = int(image_header[1])
    except ValueError as exc:
        raise ValidationError(f"candidate manifest {path} image count is not numeric") from exc
    if image_count != len(expected_names):
        raise ValidationError(
            f"candidate manifest {path} image count {image_count} differs from loaded {len(expected_names)}"
        )
    for expected_index, expected_name in enumerate(expected_names):
        fields = next_line("image entry").split(maxsplit=2)
        if len(fields) != 3 or fields[0] != "image":
            raise ValidationError(f"candidate manifest {path} image entry must be image INDEX NAME")
        try:
            index = int(fields[1])
        except ValueError as exc:
            raise ValidationError(f"candidate manifest {path} image index is not numeric") from exc
        if index != expected_index or fields[2] != expected_name:
            raise ValidationError(
                f"candidate manifest {path} image entry {expected_index} does not match loaded image order"
            )

    pair_header = next_line("pair count").split()
    if len(pair_header) != 2 or pair_header[0] != "pairs":
        raise ValidationError(f"candidate manifest {path} pair count must be pairs N")
    try:
        pair_count = int(pair_header[1])
    except ValueError as exc:
        raise ValidationError(f"candidate manifest {path} pair count is not numeric") from exc
    if pair_count < 0:
        raise ValidationError(f"candidate manifest {path} has a negative pair count")
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pair_number in range(pair_count):
        fields = next_line("pair entry").split()
        if len(fields) != 3 or fields[0] != "pair":
            raise ValidationError(f"candidate manifest {path} pair {pair_number} must be pair I J")
        try:
            first, second = int(fields[1]), int(fields[2])
        except ValueError as exc:
            raise ValidationError(f"candidate manifest {path} pair {pair_number} is not numeric") from exc
        if not (0 <= first < second < image_count):
            raise ValidationError(
                f"candidate manifest {path} pair {pair_number} must satisfy 0 <= I < J < {image_count}"
            )
        pair = (first, second)
        if pair in seen:
            raise ValidationError(f"candidate manifest {path} repeats pair {pair}")
        seen.add(pair)
        pairs.append(pair)
    if cursor != len(lines):
        raise ValidationError(f"candidate manifest {path} has unexpected trailing data")
    return {"path": str(path), "pair_count": pair_count, "pairs": pairs, "sha256": sha256_file(path)}


def validate_candidate_manifest(
    path: Path,
    expected_names: list[str],
    expected_pair_count: int | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    if expected_sha256 is not None:
        validate_hashed_file(path, expected_sha256, "candidate manifest")
    result = parse_candidate_manifest(path, expected_names)
    if expected_pair_count is not None and result["pair_count"] != expected_pair_count:
        raise ValidationError(
            f"candidate manifest pair count mismatch: expected {expected_pair_count}, got {result['pair_count']}"
        )
    # Do not expose the full pair list in the benchmark summary; its digest and
    # count are sufficient for an independently reproducible schedule.
    return {key: value for key, value in result.items() if key != "pairs"}


def parse_mapping_log(path: Path) -> dict[str, Any]:
    """Extract machine-readable candidate/verification/mapping counters."""

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValidationError(f"cannot read mapping log {path}: {exc}") from exc
    graph = VIEW_GRAPH_RE.findall(text)
    verified = VERIFIED_RE.findall(text)
    recon = RECONSTRUCTION_RE.findall(text)
    written = WRITTEN_MODEL_RE.findall(text)
    expansions = [int(value) for value in TRANSITIVE_RE.findall(text)]
    runtime = RUNTIME_RE.findall(text)
    if not graph:
        raise ValidationError(f"mapping log has no candidate-pair summary: {path}")
    result: dict[str, Any] = {
        "candidate_pairs": int(graph[-1]),
        "candidate_pairs_base": int(graph[-1]),
        "adaptive_expansion_pairs": sum(expansions),
    }
    if expansions:
        result["candidate_pairs_total"] = int(graph[-1]) + sum(expansions)
    if verified:
        matched, attempted, inliers = verified[-1]
        result.update(
            {
                "verified_pairs": int(matched),
                "verification_pairs": int(attempted),
                "inlier_correspondences": int(inliers),
            }
        )
    if recon:
        registered, total, tracks, reprojection = recon[-1]
        result.update(
            {
                "registered": int(registered),
                "total_images": int(total),
                "tracks": int(tracks),
                "reprojection_px": float(reprojection),
            }
        )
    if written:
        model_path, images, points, observations = written[-1]
        result["written_model"] = {
            "path": model_path,
            "images": int(images),
            "points": int(points),
            "observations": int(observations),
        }
    if runtime:
        elapsed, maxrss = runtime[-1]
        result["process_elapsed_s"] = float(elapsed)
        result["peak_rss_kb"] = int(maxrss)
    return result


def parse_colmap_image_names(path: Path) -> list[str]:
    try:
        lines = [line.rstrip("\r\n") for line in path.read_text(encoding="utf-8").splitlines(keepends=True) if not line.lstrip().startswith("#")]
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(f"cannot read COLMAP images.txt {path}: {exc}") from exc
    while lines and not lines[0].strip():
        lines.pop(0)
    if len(lines) % 2:
        raise ValidationError(f"COLMAP images.txt has an unmatched pose/points2D row: {path}")
    names: list[str] = []
    for row_number in range(0, len(lines), 2):
        fields = lines[row_number].split()
        if len(fields) < 10:
            raise ValidationError(f"COLMAP images.txt pose row {row_number + 1} is malformed: {path}")
        try:
            int(fields[0])
            [float(value) for value in fields[1:8]]
        except ValueError as exc:
            raise ValidationError(f"COLMAP images.txt pose row {row_number + 1} is not numeric: {path}") from exc
        names.append(fields[9])
    if len(set(names)) != len(names):
        raise ValidationError(f"COLMAP images.txt contains duplicate image names: {path}")
    return names


def parse_colmap_point_stats(path: Path) -> tuple[int, int]:
    points = _noncomment_lines(path)
    observations = 0
    for row_number, line in enumerate(points, 1):
        fields = line.split()
        if len(fields) < 8:
            raise ValidationError(f"COLMAP points3D.txt row {row_number} is malformed: {path}")
        try:
            [float(value) for value in fields[1:8]]
        except ValueError as exc:
            raise ValidationError(f"COLMAP points3D.txt row {row_number} is not numeric: {path}") from exc
        if (len(fields) - 8) % 2:
            raise ValidationError(f"COLMAP points3D.txt row {row_number} has an incomplete track: {path}")
        observations += (len(fields) - 8) // 2
    return len(points), observations


def validate_model(root: Path, spec: dict[str, Any], label: str, *, expected_files: bool = True) -> dict[str, Any]:
    if not isinstance(spec, dict) or not isinstance(spec.get("path"), str):
        raise ValidationError(f"models.{label}.path is required")
    model_dir = resolve_path(root, spec["path"], label=f"{label} model")
    if not model_dir.is_dir():
        raise ValidationError(f"{label} model directory is missing: {model_dir}")
    files = spec.get("files", {})
    if expected_files:
        if not isinstance(files, dict):
            raise ValidationError(f"models.{label}.files must be an object")
        for filename, expected_sha in files.items():
            validate_hashed_file(model_dir / filename, expected_sha, f"{label} model {filename}")
    image_path = model_dir / "images.txt"
    point_path = model_dir / "points3D.txt"
    names = parse_colmap_image_names(image_path)
    points, observations = parse_colmap_point_stats(point_path)
    expected_registered = spec.get("registered")
    if expected_registered is not None and len(names) != expected_registered:
        raise ValidationError(f"{label} registered-camera count mismatch: expected {expected_registered}, got {len(names)}")
    for key, actual in (("tracks", points), ("points", points), ("observations", observations)):
        if key in spec and spec[key] != actual:
            raise ValidationError(f"{label} {key} mismatch: expected {spec[key]}, got {actual}")
    result: dict[str, Any] = {
        "path": str(model_dir),
        "registered": len(names),
        "points": points,
        "observations": observations,
    }
    if "tracks" in spec:
        result["tracks"] = points
    if "reprojection_px" in spec:
        result["reprojection_px"] = spec["reprojection_px"]
    if files:
        result["file_hashes"] = dict(sorted((str(key), str(value).lower()) for key, value in files.items()))
    return result


def parse_score_file(path: Path) -> dict[str, float | int]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(f"cannot read visloc score file {path}: {exc}") from exc
    match = SCORE_MATCH_RE.search(text)
    rmse = SCORE_RMSE_RE.search(text)
    if match is None or rmse is None:
        raise ValidationError(f"visloc score file has an unsupported format: {path}")
    return {
        "matched": int(match.group(1)),
        "reference_registered": int(match.group(2)),
        "registered": int(match.group(3)),
        "rmse_m": float(rmse.group(1)),
    }


def validate_score_file(root: Path, spec: dict[str, Any], model_spec: dict[str, Any], max_rmse_m: float) -> dict[str, Any]:
    score_spec = model_spec.get("score_file")
    if not isinstance(score_spec, dict):
        raise ValidationError("models.visloc.score_file is required for verify-only mode")
    path = resolve_path(root, score_spec.get("path"), label="visloc score")
    validate_hashed_file(path, score_spec.get("sha256"), "visloc score file")
    score = parse_score_file(path)
    expected_registered = int(model_spec.get("registered", 0))
    if score["registered"] != expected_registered or score["matched"] != expected_registered:
        raise ValidationError(
            f"stored visloc score is incomplete: expected {expected_registered}/{expected_registered}, "
            f"got {score['matched']}/{score['reference_registered']} and registered={score['registered']}"
        )
    expected_rmse = model_spec.get("score_rmse_m")
    if expected_rmse is not None and abs(float(score["rmse_m"]) - float(expected_rmse)) > 1e-12:
        raise ValidationError(f"stored visloc RMSE differs from manifest: expected {expected_rmse}, got {score['rmse_m']}")
    if float(score["rmse_m"]) > max_rmse_m:
        raise ValidationError(f"visloc centre RMSE {score['rmse_m']:.6f} m exceeds threshold {max_rmse_m:.6f} m")
    return {**score, "score_file": str(path), "sha256": score_spec["sha256"]}


def _png_dimensions(path: Path) -> tuple[int, int]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValidationError(f"cannot read PNG asset {path}: {exc}") from exc
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValidationError(f"invalid PNG signature/header: {path}")
    return struct.unpack(">II", data[16:24])


def _gif_dimensions_and_frames(path: Path) -> tuple[int, int, int]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValidationError(f"cannot read GIF asset {path}: {exc}") from exc
    if len(data) < 13 or data[:6] not in (b"GIF87a", b"GIF89a"):
        raise ValidationError(f"invalid GIF header: {path}")
    width, height = struct.unpack_from("<HH", data, 6)
    packed = data[10]
    offset = 13
    if packed & 0x80:
        offset += 3 * (1 << ((packed & 0x07) + 1))
    frames = 0
    while offset < len(data):
        introducer = data[offset]
        offset += 1
        if introducer == 0x3B:
            break
        if introducer == 0x2C:
            if offset + 9 > len(data):
                raise ValidationError(f"truncated GIF image descriptor: {path}")
            local_packed = data[offset + 8]
            offset += 9
            if local_packed & 0x80:
                offset += 3 * (1 << ((local_packed & 0x07) + 1))
            if offset >= len(data):
                raise ValidationError(f"truncated GIF LZW header: {path}")
            offset += 1
            while True:
                if offset >= len(data):
                    raise ValidationError(f"truncated GIF image data: {path}")
                size = data[offset]
                offset += 1
                if size == 0:
                    break
                offset += size
                if offset > len(data):
                    raise ValidationError(f"truncated GIF sub-block: {path}")
            frames += 1
            continue
        if introducer == 0x21:
            if offset >= len(data):
                raise ValidationError(f"truncated GIF extension: {path}")
            offset += 1  # extension label
            while True:
                if offset >= len(data):
                    raise ValidationError(f"truncated GIF extension data: {path}")
                size = data[offset]
                offset += 1
                if size == 0:
                    break
                offset += size
                if offset > len(data):
                    raise ValidationError(f"truncated GIF extension sub-block: {path}")
            continue
        raise ValidationError(f"unknown GIF block 0x{introducer:02x}: {path}")
    return width, height, frames


def validate_visuals(repo_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    readme_value = spec.get("readme", "README.md")
    readme = (repo_root / readme_value).resolve()
    if not readme.is_file():
        raise ValidationError(f"README for benchmark visual validation is missing: {readme}")
    readme_text = readme.read_text(encoding="utf-8")
    for needle in spec.get("readme_needles", []):
        if needle not in readme_text:
            raise ValidationError(f"README is missing the measured courtyard claim/reference: {needle!r}")
    assets_result: dict[str, Any] = {}
    assets = spec.get("assets")
    if not isinstance(assets, dict):
        raise ValidationError("visuals.assets must be an object")
    for name, asset in sorted(assets.items()):
        if not isinstance(asset, dict) or not isinstance(asset.get("path"), str):
            raise ValidationError(f"visuals.assets.{name}.path is required")
        path = (repo_root / asset["path"]).resolve()
        validate_hashed_file(path, asset.get("sha256"), f"README {name} asset")
        if name == "png":
            width, height = _png_dimensions(path)
            frames = None
        elif name == "gif":
            width, height, frames = _gif_dimensions_and_frames(path)
        else:
            raise ValidationError(f"unsupported visual asset kind {name!r}")
        if (width, height) != (asset.get("width"), asset.get("height")):
            raise ValidationError(f"README {name} dimensions differ: expected {(asset.get('width'), asset.get('height'))}, got {(width, height)}")
        if frames is not None and asset.get("frames") is not None and frames != asset["frames"]:
            raise ValidationError(f"README GIF frame count differs: expected {asset['frames']}, got {frames}")
        assets_result[name] = {"path": str(path), "sha256": asset["sha256"], "width": width, "height": height, **({"frames": frames} if frames is not None else {})}
    return {"readme": str(readme), "assets": assets_result}


def validate_artifact_root_manifest(root: Path, config: dict[str, Any]) -> None:
    manifest_spec = config.get("artifact_manifest")
    if isinstance(manifest_spec, dict):
        path = resolve_path(root, manifest_spec.get("path"), label="artifact checksum manifest")
        validate_hashed_file(path, manifest_spec.get("sha256"), "artifact checksum manifest")


def validate_inputs(config: dict[str, Any], root: Path, *, images_override: Path | None, calibration_override: Path | None, max_rmse_m: float, include_colmap: bool = True) -> dict[str, Any]:
    validate_artifact_root_manifest(root, config)
    inputs = config["inputs"]
    mapping = config["mapping"]
    image_suffix = str(mapping.get("image_suffix", ".JPG"))
    feature_stems, feature_stats = validate_features(root, inputs["features"])
    images_spec = inputs.get("images")
    images_summary: dict[str, Any] | None = None
    if isinstance(images_spec, dict):
        configured_dir = Path(images_spec.get("directory", "")) if images_spec.get("directory") else None
        image_dir = (images_override or configured_dir)
        if image_dir is None:
            raise ValidationError("source image directory is required; pass --images-dir or configure inputs.images.directory")
        images_summary = validate_images(image_dir.expanduser().resolve(), images_spec, _expected_image_names(feature_stems, image_suffix))
    matches_spec = inputs["matches"]
    matches_path = resolve_path(root, matches_spec.get("path"), label="raw matches")
    matches_summary = validate_matches(matches_path, matches_spec, feature_stems, image_suffix, feature_stats.get("row_counts"))

    calibration_spec = inputs.get("calibration")
    calibration_summary: dict[str, Any] | None = None
    calibration_dir: Path | None = None
    if isinstance(calibration_spec, dict):
        configured_calibration = Path(calibration_spec.get("model_dir", "")) if calibration_spec.get("model_dir") else None
        calibration_dir = (calibration_override or configured_calibration)
        if calibration_dir is None:
            raise ValidationError("calibration model is required; pass --calibration-model or configure inputs.calibration.model_dir")
        calibration_dir = calibration_dir.expanduser().resolve()
        if not calibration_dir.is_dir():
            raise ValidationError(f"calibration model directory is missing: {calibration_dir}")
        for filename, expected_sha in calibration_spec.get("files", {}).items():
            validate_hashed_file(calibration_dir / filename, expected_sha, f"calibration {filename}")
        calibration_names = parse_colmap_image_names(calibration_dir / "images.txt")
        expected_registered = calibration_spec.get("registered")
        if expected_registered is not None and len(calibration_names) != expected_registered:
            raise ValidationError(f"calibration registered-camera count mismatch: expected {expected_registered}, got {len(calibration_names)}")
        calibration_summary = {"path": str(calibration_dir), "registered": len(calibration_names), "file_hashes": dict(sorted(calibration_spec.get("files", {}).items()))}

    model_summary: dict[str, Any] = {}
    visloc_spec = config["models"].get("visloc")
    if not isinstance(visloc_spec, dict):
        raise ValidationError("models.visloc is required")
    model_summary["visloc"] = validate_model(root, visloc_spec, "visloc")
    if include_colmap:
        colmap_spec = config["models"].get("colmap")
        if not isinstance(colmap_spec, dict):
            raise ValidationError("models.colmap is required when --colmap-control is validate or run")
        model_summary["colmap"] = validate_model(root, colmap_spec, "COLMAP")
    result = {
        "artifact_root": str(root),
        "features": {key: value for key, value in feature_stats.items() if key != "row_counts"},
        "feature_names": feature_stems,
        "matches": matches_summary,
        "images": images_summary,
        "calibration": calibration_summary,
        "models": model_summary,
        "calibration_dir": str(calibration_dir) if calibration_dir is not None else None,
        "matches_path": str(matches_path),
        "features_dir": str((resolve_path(root, inputs["features"]["manifest"]["path"], label="feature manifest")).parent),
    }
    return result


def _tail(path: Path, lines: int = 30) -> str:
    try:
        values = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(log unavailable)"
    return "\n".join(values[-lines:])


def run_command(command: list[str], *, cwd: Path, log_path: Path, env: dict[str, str] | None = None) -> float:
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {shlex.join(command)}", file=sys.stderr)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=cwd, env=merged_env, stdout=log, stderr=subprocess.STDOUT, check=False)
    except OSError as exc:
        raise ValidationError(f"cannot execute {command[0]!r}: {exc}; see intended command above") from exc
    if result.returncode != 0:
        raise ValidationError(f"command failed with exit code {result.returncode}: {shlex.join(command)}\nlast log lines:\n{_tail(log_path)}")
    return time.monotonic() - started


def _prepare_output(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise ValidationError(f"output directory already exists: {path}; choose a new --output-dir or pass --force")
        if not path.is_dir():
            raise ValidationError(f"--output-dir exists but is not a directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def resolve_candidate_schedule(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Resolve a named, pre-match candidate schedule from the benchmark config."""

    schedules = config.get("candidate_schedules", {})
    if not isinstance(schedules, dict):
        raise ValidationError("candidate_schedules must be an object")
    value = schedules.get(name)
    if value is None:
        # Explicit strategy names are useful for a one-off local run while
        # named entries keep the published benchmark reproducible.
        value = {"strategy": name}
    if not isinstance(value, dict):
        raise ValidationError(f"candidate schedule {name!r} must be an object")
    schedule = dict(value)
    strategy = schedule.get("strategy")
    if strategy not in {"exhaustive", "sequential", "vlad", "vlad-mutual", "vlad-union", "adaptive", "manifest"}:
        raise ValidationError(
            f"candidate schedule {name!r} has unsupported strategy {strategy!r}; "
            "expected exhaustive|sequential|vlad|vlad-mutual|vlad-union|adaptive|manifest"
        )
    schedule["name"] = name
    if strategy in {"vlad", "vlad-mutual", "vlad-union"}:
        topk = schedule.get("retrieval_topk")
        if not isinstance(topk, int) or topk <= 0:
            raise ValidationError(f"candidate schedule {name!r} retrieval_topk must be a positive integer")
    if strategy == "sequential":
        window = schedule.get("stem_window")
        if not isinstance(window, int) or window <= 0:
            raise ValidationError(f"candidate schedule {name!r} stem_window must be a positive integer")
    if strategy == "vlad-union":
        window = schedule.get("local_stem_window")
        if not isinstance(window, int) or window <= 0:
            raise ValidationError(f"candidate schedule {name!r} local_stem_window must be a positive integer")
    if strategy == "adaptive":
        count = schedule.get("vocab_tree_num_images")
        if not isinstance(count, int) or count <= 0:
            raise ValidationError(f"candidate schedule {name!r} vocab_tree_num_images must be a positive integer")
    budget = schedule.get("candidate_budget")
    if budget is not None and (not isinstance(budget, int) or budget <= 0):
        raise ValidationError(f"candidate schedule {name!r} candidate_budget must be a positive integer")
    if "candidate_count" in schedule and (not isinstance(schedule["candidate_count"], int) or schedule["candidate_count"] <= 0):
        raise ValidationError(f"candidate schedule {name!r} candidate_count must be a positive integer")
    return schedule


def resolve_schedule_manifest(root: Path, schedule: dict[str, Any]) -> tuple[Path, str | None] | None:
    """Resolve an optional pre-generated manifest referenced by a schedule."""

    value = schedule.get("candidate_manifest")
    if value is None:
        return None
    if isinstance(value, str):
        return resolve_path(root, value, label="candidate manifest"), None
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ValidationError("candidate schedule candidate_manifest must be a path or {path, sha256}")
    return resolve_path(root, value["path"], label="candidate manifest"), value.get("sha256")


def build_mapping_command(
    config: dict[str, Any],
    *,
    binary: Path,
    features_dir: Path,
    images_dir: Path | None,
    calibration_dir: Path,
    matches_path: Path,
    output_model: Path,
    candidate_schedule: dict[str, Any] | None = None,
    candidate_manifest: Path | None = None,
    include_match_import: bool = True,
) -> list[str]:
    mapping = config["mapping"]
    command = [str(binary), "--feature-extractor", "files", "--features-dir", str(features_dir)]
    command.extend(["--feature-suffix", str(mapping.get("feature_suffix", "_features.txt")), "--image-suffix", str(mapping.get("image_suffix", ".JPG"))])
    if images_dir is not None:
        command.extend(["--images-dir", str(images_dir)])
    command.extend(["--input-colmap-calibration", str(calibration_dir)])
    if include_match_import:
        command.extend(["--import-matches-file", str(matches_path)])
    schedule = candidate_schedule or {"strategy": "exhaustive", "name": "exhaustive"}
    strategy = schedule["strategy"]
    if candidate_manifest is not None:
        command.extend(["--candidate-manifest", str(candidate_manifest)])
    elif strategy == "exhaustive":
        command.append("--exhaustive")
    elif strategy == "sequential":
        command.extend(["--exhaustive", "--pair-stem-window", str(schedule["stem_window"])])
    elif strategy in {"vlad", "vlad-mutual"}:
        command.extend(["--pair-source", strategy, "--retrieval-topk", str(schedule["retrieval_topk"])])
    elif strategy == "vlad-union":
        command.extend(
            [
                "--pair-source",
                "vlad-union",
                "--local-stem-window",
                str(schedule["local_stem_window"]),
                "--retrieval-topk",
                str(schedule["retrieval_topk"]),
            ]
        )
    elif strategy == "adaptive":
        command.extend(
            [
                "--pair-source",
                "transitive",
                "--vocab-tree-num-images",
                str(schedule["vocab_tree_num_images"]),
            ]
        )
    elif strategy == "manifest":
        raise ValidationError("candidate strategy 'manifest' requires --candidate-manifest or a manifest path in the schedule")
    else:  # pragma: no cover - resolve_candidate_schedule validates this
        raise ValidationError(f"unsupported candidate strategy {strategy!r}")
    if candidate_manifest is None and schedule.get("candidate_budget") is not None:
        if strategy != "vlad-union":
            raise ValidationError("candidate_budget is currently supported only by vlad-union")
        command.extend(["--candidate-budget", str(schedule["candidate_budget"])])
    command.extend([
        "--min-matches", str(mapping.get("min_matches", 20)),
        "--match-ratio", str(mapping.get("match_ratio", 0.8)),
        "--verification-mode", str(mapping.get("verification_mode", "full")),
        "--mapper", str(mapping.get("mapper", "incremental")),
        "--pnp-max-iterations", str(mapping.get("pnp_max_iterations", 100000)),
        "--min-pnp-inliers", str(mapping.get("min_pnp_inliers", 8)),
    ])
    max_mapper_matches = mapping.get("max_mapper_matches_per_pair")
    if max_mapper_matches is not None:
        if not isinstance(max_mapper_matches, int) or max_mapper_matches <= 0:
            raise ValidationError(
                "mapping.max_mapper_matches_per_pair must be a positive integer"
            )
        command.extend(
            ["--max-mapper-matches-per-pair", str(max_mapper_matches)]
        )
    for key, flag in (("geometry_guided_conflict_recovery", "--geometry-guided-conflict-recovery"), ("post_refinement_registration", "--post-refinement-registration"), ("final_iterative_refinement", "--final-iterative-refinement")):
        if mapping.get(key, False):
            command.append(flag)
    if mapping.get("next_image_policy"):
        command.extend(["--next-image-policy", str(mapping["next_image_policy"])])
    command.extend(["--out-colmap", str(output_model)])
    return command


def score_model(query_model: Path, reference_model: Path) -> dict[str, Any]:
    try:
        import numpy as np
        import compare_sfm_sim3
    except ImportError as exc:
        raise ValidationError(f"full score requires numpy and scripts/compare_sfm_sim3.py: {exc}") from exc
    try:
        reference = compare_sfm_sim3.load_centers(str(reference_model / "images.txt"))
        query = compare_sfm_sim3.load_centers(str(query_model / "images.txt"))
    except (OSError, ValueError, ZeroDivisionError) as exc:
        raise ValidationError(f"cannot parse COLMAP model for Sim(3) score: {exc}") from exc
    common = sorted(set(reference) & set(query))
    if len(common) < 3:
        raise ValidationError(f"cannot score model: only {len(common)} common camera names")
    source = np.asarray([query[index] for index in common], dtype=float)
    destination = np.asarray([reference[index] for index in common], dtype=float)
    scale, rotation, translation = compare_sfm_sim3.umeyama(source, destination)
    aligned = (scale * (rotation @ source.T).T) + translation
    errors = np.linalg.norm(aligned - destination, axis=1)
    return {
        "registered": len(query),
        "reference_registered": len(reference),
        "common": len(common),
        "scale": float(scale),
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "median_m": float(np.median(errors)),
        "max_m": float(np.max(errors)),
    }


def run_colmap_control(config: dict[str, Any], root: Path, images_dir: Path, output_dir: Path, calibration_dir: Path) -> dict[str, Any]:
    control = config.get("colmap_control", {})
    database_spec = control.get("database", "exhaustive/database_calibrated.db")
    database = resolve_path(root, database_spec, label="COLMAP database")
    if not database.is_file():
        raise ValidationError(f"COLMAP control database is missing: {database}")
    model_dir = output_dir / "colmap_model"
    command = ["colmap", "mapper", "--database_path", str(database), "--image_path", str(images_dir), "--output_path", str(model_dir)]
    run_command(command, cwd=REPO_ROOT, log_path=output_dir / "colmap_mapper.log")
    candidates = sorted(path for path in model_dir.rglob("images.txt") if path.is_file())
    if not candidates:
        raise ValidationError(f"COLMAP mapper completed but wrote no images.txt under {model_dir}")
    selected = candidates[0].parent
    model = validate_model(root, {"path": str(selected), "registered": None}, "generated COLMAP", expected_files=False)
    score = score_model(selected, calibration_dir)
    return {"command": command, "model": model, "score": score}


def regenerate_visuals(config: dict[str, Any], root: Path, output_dir: Path, calibration_dir: Path) -> dict[str, Any]:
    static_visloc = resolve_path(root, config["models"]["visloc"]["path"], label="visloc model")
    generated_visloc = output_dir / "visloc_model"
    visloc_model = generated_visloc if (generated_visloc / "images.txt").is_file() else static_visloc
    static_colmap = resolve_path(root, config["models"]["colmap"]["path"], label="COLMAP model")
    visual_dir = output_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "generate_courtyard_readme_visuals.py"),
        "--visloc-model", str(visloc_model),
        "--colmap-model", str(static_colmap),
        "--reference-model", str(calibration_dir),
        "--output-dir", str(visual_dir),
    ]
    run_command(command, cwd=REPO_ROOT, log_path=output_dir / "visuals.log")
    generated_spec = {
        "readme": "README.md",
        "readme_needles": [],
        "assets": config["visuals"]["assets"],
    }
    # The generator's output is in a temporary directory, so validate its
    # dimensions and deterministic hashes without requiring those files to be
    # copied into tracked docs/assets.
    result: dict[str, Any] = {"command": command, "output_dir": str(visual_dir), "assets": {}}
    for name, asset in sorted(config["visuals"]["assets"].items()):
        generated = visual_dir / Path(asset["path"]).name
        if not generated.is_file():
            raise ValidationError(f"visual generator did not produce {generated}")
        actual = sha256_file(generated)
        if actual != asset["sha256"]:
            raise ValidationError(f"regenerated README {name} hash differs: expected {asset['sha256']}, got {actual}; check plotting dependency versions")
        if name == "png":
            width, height = _png_dimensions(generated)
            frames = None
        else:
            width, height, frames = _gif_dimensions_and_frames(generated)
        if (width, height) != (asset["width"], asset["height"]):
            raise ValidationError(f"regenerated README {name} dimensions differ")
        if frames is not None and frames != asset.get("frames"):
            raise ValidationError(f"regenerated README GIF frame count differs")
        result["assets"][name] = {"path": str(generated), "sha256": actual, "width": width, "height": height, **({"frames": frames} if frames is not None else {})}
    del generated_spec
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.is_dir():
        raise ValidationError(f"summary path is a directory: {path}")
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        raise ValidationError(f"cannot write summary JSON {path}: {exc}") from exc


def default_summary_path(args: argparse.Namespace) -> Path:
    if args.summary_json is not None:
        return Path(args.summary_json).expanduser()
    if args.full:
        return Path(args.output_dir).expanduser().resolve() / "summary.json"
    return (REPO_ROOT / "target" / "benchmark_courtyard" / "verify-summary.json").resolve()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true", help="validate hashes/models/README without building or mapping (default)")
    mode.add_argument("--full", action="store_true", help="run the visloc mapper, score the output, and write logs/artifacts")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=f"benchmark JSON manifest (default: {DEFAULT_CONFIG})")
    parser.add_argument("--artifact-root", type=Path, help=f"external artifact root (or {DEFAULT_ARTIFACT_ROOT_ENV})")
    parser.add_argument("--images-dir", type=Path, help="override the exact source image directory")
    parser.add_argument("--calibration-model", type=Path, help="override the calibration model directory (hashes must still match)")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "target" / "benchmark_courtyard", help="fresh full-run output directory")
    parser.add_argument("--summary-json", type=str, help="summary path; use '-' to print only to stdout (default writes target/.../summary.json)")
    parser.add_argument("--force", action="store_true", help="remove only an existing --output-dir before a full run")
    parser.add_argument("--no-build", action="store_true", help="full mode: do not run cargo build before mapping")
    parser.add_argument("--colmap-control", choices=("validate", "run", "skip"), default="validate", help="validate the durable COLMAP model, run COLMAP, or skip the control")
    parser.add_argument("--visuals", choices=("check", "regenerate", "skip"), default="check", help="check tracked README assets, regenerate into the full output, or skip")
    parser.add_argument("--max-rmse-m", type=float, help="maximum accepted visloc centre RMSE in metres (default: manifest threshold)")
    parser.add_argument(
        "--candidate-strategy",
        default="exhaustive",
        help="named pre-match candidate schedule from the config (default: exhaustive)",
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        help="use an image-name-bound candidate manifest instead of generating candidates",
    )
    parser.add_argument(
        "--write-candidate-manifest",
        type=Path,
        help="full mode: generate the selected pre-match schedule and atomically export its manifest before mapping",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="full mode: report a reduced schedule even when it does not register every reference image",
    )
    parser.add_argument(
        "--max-mapper-matches-per-pair",
        type=int,
        help="override the mapper correspondence cap without changing matching/verification",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    if args.max_mapper_matches_per_pair is not None:
        if args.max_mapper_matches_per_pair <= 0:
            raise ValidationError("--max-mapper-matches-per-pair must be positive")
        config["mapping"]["max_mapper_matches_per_pair"] = (
            args.max_mapper_matches_per_pair
        )
    root = resolve_artifact_root(config, args.artifact_root)
    schedule = resolve_candidate_schedule(config, args.candidate_strategy)
    configured_manifest = resolve_schedule_manifest(root, schedule)
    cli_manifest = args.candidate_manifest.expanduser().resolve() if args.candidate_manifest is not None else None
    if cli_manifest is not None and configured_manifest is not None and cli_manifest != configured_manifest[0]:
        raise ValidationError("--candidate-manifest differs from the manifest configured by the candidate schedule")
    selected_manifest = cli_manifest or (configured_manifest[0] if configured_manifest is not None else None)
    selected_manifest_sha = configured_manifest[1] if configured_manifest is not None else None
    # A CLI manifest is an explicit replay override.  In particular, the
    # documented `--candidate-strategy exhaustive --candidate-manifest ...`
    # form must not insist that the replayed file contain 703 pairs.
    manifest_expected_pairs = schedule.get("candidate_count")
    if cli_manifest is not None and schedule["strategy"] == "exhaustive":
        manifest_expected_pairs = None
    expected_max = config.get("expected", {}).get("max_rmse_m", config["models"]["visloc"].get("score_rmse_m", 0.01))
    max_rmse_m = float(args.max_rmse_m if args.max_rmse_m is not None else expected_max)
    if max_rmse_m <= 0:
        raise ValidationError("--max-rmse-m must be positive")
    full = bool(args.full)
    if not full and (args.colmap_control == "run" or args.visuals == "regenerate"):
        raise ValidationError("--colmap-control run and --visuals regenerate require --full")
    if args.force and not full:
        raise ValidationError("--force is only valid with --full")
    if args.no_build and not full:
        raise ValidationError("--no-build is only valid with --full")
    if args.allow_incomplete and not full:
        raise ValidationError("--allow-incomplete is only valid with --full")
    if args.candidate_manifest is not None and args.write_candidate_manifest is not None:
        raise ValidationError("--candidate-manifest and --write-candidate-manifest are mutually exclusive")
    if args.write_candidate_manifest is not None and schedule["strategy"] == "manifest":
        raise ValidationError("candidate strategy 'manifest' cannot generate a new candidate manifest")
    if selected_manifest is None and schedule["strategy"] == "manifest":
        raise ValidationError("candidate strategy 'manifest' requires --candidate-manifest or a configured manifest")
    inputs = validate_inputs(
        config,
        root,
        images_override=args.images_dir,
        calibration_override=args.calibration_model,
        max_rmse_m=max_rmse_m,
        include_colmap=args.colmap_control != "skip",
    )
    candidate_manifest_summary: dict[str, Any] | None = None
    if selected_manifest is not None and args.write_candidate_manifest is None:
        candidate_manifest_summary = validate_candidate_manifest(
            selected_manifest,
            _expected_image_names(inputs["feature_names"], config["mapping"].get("image_suffix", ".JPG")),
            manifest_expected_pairs,
            selected_manifest_sha,
        )
    visual_summary: dict[str, Any] | None = None
    if args.visuals == "check":
        visual_summary = validate_visuals(REPO_ROOT, config["visuals"])
    elif args.visuals == "skip":
        visual_summary = {"status": "skipped"}

    summary: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "mode": "full" if full else "verify-only",
        "artifact_root": str(root),
        "mapping": {"performed": False, "ground_truth_used": False, "reference_used_only_for_score": True},
        "inputs": {key: value for key, value in inputs.items() if key not in {"calibration_dir", "matches_path", "features_dir"}},
        "candidate_schedule": {key: value for key, value in schedule.items() if key != "name"},
        "candidate_manifest": candidate_manifest_summary,
        "future_scale": config.get("future_scale", {}),
        "visuals": visual_summary,
    }
    if not full:
        score = validate_score_file(root, config.get("expected", {}), config["models"]["visloc"], max_rmse_m)
        summary["models"] = {"visloc": inputs["models"]["visloc"], "visloc_score": score}
        if args.colmap_control == "skip":
            summary["colmap_control"] = {"status": "skipped"}
        else:
            summary["colmap_control"] = {"status": "validated", "model": inputs["models"].get("colmap")}
        return summary

    output_dir = args.output_dir.expanduser().resolve()
    _prepare_output(output_dir, args.force)
    features_dir = Path(inputs["features_dir"])
    images_dir = Path(inputs["images"]["directory"]) if inputs.get("images") else None
    calibration_dir = Path(inputs["calibration_dir"]) if inputs.get("calibration_dir") else None
    if calibration_dir is None:
        raise ValidationError("full mode requires a calibration model")
    binary = REPO_ROOT / "target" / "release" / "examples" / "unordered_sfm_demo"
    if not args.no_build:
        run_command(["cargo", "build", "--release", "--example", "unordered_sfm_demo", "--features", "image-io"], cwd=REPO_ROOT, log_path=output_dir / "cargo_build.log")
    if not binary.is_file():
        raise ValidationError(f"visloc example binary is missing: {binary}; remove --no-build or build it with --features image-io")
    export_elapsed: float | None = None
    if args.write_candidate_manifest is not None:
        selected_manifest = args.write_candidate_manifest.expanduser().resolve()
        if selected_manifest.exists() and not args.force:
            raise ValidationError(
                f"candidate manifest already exists: {selected_manifest}; choose a new path or pass --force"
            )
        export_command = build_mapping_command(
            config,
            binary=binary,
            features_dir=features_dir,
            images_dir=images_dir,
            calibration_dir=calibration_dir,
            matches_path=Path(inputs["matches_path"]),
            output_model=output_dir / "candidate_export_unused",
            candidate_schedule=schedule,
            include_match_import=False,
        )
        export_command.extend(["--export-candidate-manifest", str(selected_manifest)])
        export_elapsed = run_command(
            export_command,
            cwd=REPO_ROOT,
            log_path=output_dir / "candidate_manifest.log",
            env={"RAYON_NUM_THREADS": str(config.get("runtime", {}).get("rayon_threads", "1"))},
        )
        candidate_manifest_summary = validate_candidate_manifest(
            selected_manifest,
            _expected_image_names(inputs["feature_names"], config["mapping"].get("image_suffix", ".JPG")),
            manifest_expected_pairs,
        )
        summary["candidate_manifest"] = {
            **candidate_manifest_summary,
            "generated": True,
            "generation_command": export_command,
            "generation_elapsed_s": export_elapsed,
        }
    output_model = output_dir / "visloc_model"
    mapping_command = build_mapping_command(
        config,
        binary=binary,
        features_dir=features_dir,
        images_dir=images_dir,
        calibration_dir=calibration_dir,
        matches_path=Path(inputs["matches_path"]),
        output_model=output_model,
        candidate_schedule=schedule,
        candidate_manifest=selected_manifest,
    )
    mapping_elapsed = run_command(
        mapping_command,
        cwd=REPO_ROOT,
        log_path=output_dir / "visloc_mapper.log",
        env={"RAYON_NUM_THREADS": str(config.get("runtime", {}).get("rayon_threads", "1"))},
    )
    diagnostics = parse_mapping_log(output_dir / "visloc_mapper.log")
    expected_candidates = manifest_expected_pairs
    if expected_candidates is not None and diagnostics["candidate_pairs"] != expected_candidates:
        raise ValidationError(
            f"candidate schedule {schedule['name']!r} produced {diagnostics['candidate_pairs']} pairs, "
            f"manifest expects {expected_candidates}"
        )
    expected_registered = config["models"]["visloc"].get("registered")
    generated = validate_model(
        root,
        {"path": str(output_model), "registered": None if args.allow_incomplete else expected_registered},
        "generated visloc",
        expected_files=False,
    )
    score: dict[str, Any] | None
    score_error: str | None = None
    try:
        score = score_model(output_model, calibration_dir)
    except ValidationError as exc:
        if not args.allow_incomplete:
            raise
        score = None
        score_error = str(exc)
    if score is None:
        gate_reasons = [score_error or "model could not be scored"]
    else:
        gate_reasons: list[str] = []
        if score["registered"] != expected_registered or score["common"] != expected_registered:
            gate_reasons.append(
                f"registered/common={score['registered']}/{score['common']} expected {expected_registered}/{expected_registered}"
            )
        if score["rmse_m"] > max_rmse_m:
            gate_reasons.append(f"centre RMSE {score['rmse_m']:.6f} m exceeds {max_rmse_m:.6f} m")
    gate_passed = not gate_reasons
    if not gate_passed and not args.allow_incomplete:
        raise ValidationError("candidate schedule failed benchmark gate: " + "; ".join(gate_reasons))
    summary["mapping"] = {
        "performed": True,
        "ground_truth_used": False,
        "reference_used_only_for_score": True,
        "matching_mode": "imported raw matches; local matching was not recomputed",
        "command": mapping_command,
        "elapsed_s": mapping_elapsed,
        "candidate_generation_elapsed_s": export_elapsed,
        "diagnostics": diagnostics,
        "model": generated,
        "score": score,
        "gate_passed": gate_passed,
        "gate_reasons": gate_reasons,
    }
    if args.colmap_control == "skip":
        summary["colmap_control"] = {"status": "skipped"}
    elif args.colmap_control == "validate":
        summary["colmap_control"] = {"status": "validated", "model": inputs["models"].get("colmap")}
    else:
        if images_dir is None:
            raise ValidationError("--colmap-control run requires --images-dir or inputs.images.directory")
        summary["colmap_control"] = run_colmap_control(config, root, images_dir, output_dir, calibration_dir)
    if args.visuals == "regenerate":
        summary["visuals"] = regenerate_visuals(config, root, output_dir, calibration_dir)
    summary["output_dir"] = str(output_dir)
    return summary


def _summary_path_for_failure(args: argparse.Namespace) -> Path | None:
    if args.summary_json == "-":
        return None
    try:
        return default_summary_path(args)
    except (AttributeError, TypeError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        summary = run(args)
        rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.summary_json == "-":
            print(rendered, end="")
        else:
            path = default_summary_path(args)
            _write_json(path, summary)
            print(rendered, end="")
            print(f"summary: {path}", file=sys.stderr)
        return 0
    except ValidationError as exc:
        failure = {"schema_version": 1, "status": "failed", "error": str(exc)}
        path = _summary_path_for_failure(args)
        if path is not None:
            try:
                _write_json(path, failure)
            except ValidationError:
                pass
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
