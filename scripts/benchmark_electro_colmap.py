#!/usr/bin/env python3
"""Prepare and run an official COLMAP control for ETH3D ``electro``.

The control uses COLMAP's own CPU SIFT extractor and ``matches_importer``
with ``match_type=pairs``.  The latter is important: it makes COLMAP compute
and geometrically verify exactly the image pairs from the visloc candidate
manifest rather than silently switching to an exhaustive or vocabulary-tree
pair schedule.  The four camera folders are extracted separately so each
folder receives its official PINHOLE intrinsics, while the stored COLMAP
image names remain the flat ``camN_timestamp.png`` names used by visloc.

``--prepare`` is read-only with respect to the dataset and writes only a
small external plan, pair list, and hashes.  ``--run`` is intentionally an
explicit opt-in because feature extraction, matching, and mapping are CPU
heavy.  Every phase is wrapped in ``/usr/bin/time -v`` and gets its own log
and resource file, so the result is comparable with the visloc phase logs.
The reference model is accepted only by the separate scoring script; it is
never present in a feature, match, or mapper command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR.parent / "benchmarks" / "electro" / "colmap_1200_v1.json"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_electro import (  # noqa: E402
    ValidationError,
    parse_candidate_manifest_with_metadata,
    sha256_file,
    validate_candidate_shards,
)


PLAN_SCHEMA = "visloc_electro_colmap_control_plan_v1"
CONFIG_SCHEMA = "visloc_electro_colmap_control_config_v1"
CAMERA_PREFIX_RE = re.compile(r"^cam(?P<camera>[0-9]+)_(?P<timestamp>[0-9]+)\.(?P<suffix>[^.]+)$")
TIME_RSS_RE = re.compile(r"^\s*Maximum resident set size \(kbytes\):\s*([0-9]+)\s*$", re.MULTILINE)
TIME_WALL_RE = re.compile(
    r"^\s*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)\s*$",
    re.MULTILINE,
)
TIME_USER_RE = re.compile(r"^\s*User time \(seconds\):\s*([0-9.eE+-]+)\s*$", re.MULTILINE)
TIME_SYSTEM_RE = re.compile(r"^\s*System time \(seconds\):\s*([0-9.eE+-]+)\s*$", re.MULTILINE)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise ValidationError(f"cannot atomically install {path}: {exc}") from exc


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode())


def _safe_relative(value: str, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not value or any(part in ("", ".", "..") for part in path.parts):
        raise ValidationError(f"{label} must be a simple relative path: {value!r}")
    return path


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{label} {path} must contain a JSON object")
    return value


def parse_pinhole_cameras(path: Path) -> dict[int, dict[str, Any]]:
    """Read official COLMAP PINHOLE camera rows keyed by CAMERA_ID."""

    cameras: dict[int, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(f"cannot read calibration cameras {path}: {exc}") from exc
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8:
            raise ValidationError(f"calibration cameras {path}:{line_number} is malformed")
        try:
            camera_id = int(fields[0])
            width, height = int(fields[2]), int(fields[3])
            params = [float(value) for value in fields[4:]]
        except ValueError as exc:
            raise ValidationError(f"calibration cameras {path}:{line_number} is not numeric") from exc
        if fields[1] != "PINHOLE" or len(params) != 4:
            raise ValidationError(
                f"calibration cameras {path}:{line_number} must be PINHOLE with four parameters"
            )
        if camera_id in cameras:
            raise ValidationError(f"calibration cameras repeats CAMERA_ID {camera_id}")
        cameras[camera_id] = {
            "id": camera_id,
            "model": fields[1],
            "width": width,
            "height": height,
            "params": params,
        }
    if not cameras:
        raise ValidationError(f"calibration cameras has no camera rows: {path}")
    return cameras


def parse_image_camera_assignments(path: Path) -> dict[str, int]:
    """Read the image-to-camera assignments from a COLMAP text model."""

    assignments: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(f"cannot read calibration images {path}: {exc}") from exc
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        # An images.txt pose row has IMAGE_ID, qvec, tvec, CAMERA_ID, NAME.
        if len(fields) < 10:
            continue
        try:
            int(fields[0])
            camera_id = int(fields[8])
            float(fields[1])
            float(fields[2])
            float(fields[3])
            float(fields[4])
            float(fields[5])
            float(fields[6])
            float(fields[7])
        except ValueError:
            continue
        name = fields[9]
        if name in assignments and assignments[name] != camera_id:
            raise ValidationError(f"calibration image {name!r} has conflicting camera assignments")
        assignments[name] = camera_id
    if not assignments:
        raise ValidationError(f"calibration images has no pose rows: {path}")
    return assignments


def _camera_for_flat_name(name: str, assignments: dict[str, int]) -> int | None:
    """Resolve a flat camN_timestamp name to its calibration CAMERA_ID."""

    if name in assignments:
        return assignments[name]
    match = CAMERA_PREFIX_RE.fullmatch(name)
    if match is None:
        return None
    camera = int(match.group("camera"))
    # The staging calibration uses flat names, but accepting the official
    # basename here makes the helper useful with a source calibration model.
    official = f"images_rig_cam{camera}_undistorted/{match.group('timestamp')}.{match.group('suffix')}"
    return assignments.get(official)


def _format_params(params: Iterable[float]) -> str:
    return ",".join(format(value, ".17g") for value in params)


def _candidate_source_from_index(candidate_index: Path) -> tuple[Path, dict[str, Any]]:
    candidate = validate_candidate_shards(candidate_index)
    source_value = candidate["index"].get("source_manifest")
    if not isinstance(source_value, str) or not source_value:
        raise ValidationError("candidate index source_manifest is missing")
    source = Path(source_value).expanduser().resolve()
    if not source.is_file():
        raise ValidationError(f"candidate source manifest is missing: {source}")
    names, pairs, _ = parse_candidate_manifest_with_metadata(source)
    if names != candidate["image_names"] or len(pairs) != candidate["pair_count"]:
        raise ValidationError("candidate source does not match its validated shard index")
    # Validate the ordered pair stream against the index, not just its count.
    indexed_pairs: list[tuple[int, int]] = []
    for shard in candidate["shards"]:
        shard_path = candidate_index.parent / _safe_relative(shard["path"], "candidate shard path")
        shard_names, shard_pairs, _ = parse_candidate_manifest_with_metadata(shard_path)
        if shard_names != names:
            raise ValidationError("candidate shard image order differs from source manifest")
        indexed_pairs.extend(shard_pairs)
    if indexed_pairs != pairs:
        raise ValidationError("candidate shards do not preserve source pair order")
    return source, candidate


def _identity_aliases(names: list[str]) -> dict[str, str]:
    return {name: name for name in names}


def parse_alias_tsv(path: Path) -> dict[str, str]:
    """Read ``flat_name<TAB>official_name[<TAB>camera_id]`` aliases."""

    aliases: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(f"cannot read image alias map {path}: {exc}") from exc
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if fields and fields[0] == "flat_name":
            continue
        if len(fields) < 2 or not fields[0] or not fields[1] or any(" " in value for value in fields[:2]):
            raise ValidationError(f"image alias map {path}:{line_number} is malformed")
        if fields[0] in aliases and aliases[fields[0]] != fields[1]:
            raise ValidationError(f"image alias map repeats {fields[0]!r} with different targets")
        aliases[fields[0]] = fields[1]
    if not aliases:
        raise ValidationError(f"image alias map has no entries: {path}")
    return aliases


def write_pair_list(
    candidate_manifest: Path,
    output: Path,
    *,
    aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write a deterministic COLMAP image-pair list bound to a candidate file."""

    names, pairs, metadata = parse_candidate_manifest_with_metadata(candidate_manifest)
    aliases = aliases or _identity_aliases(names)
    mapped_names: list[str] = []
    for name in names:
        mapped = aliases.get(name)
        if not mapped:
            raise ValidationError(f"image alias map has no target for candidate image {name!r}")
        if any(char.isspace() for char in mapped):
            raise ValidationError(f"COLMAP image name cannot contain whitespace: {mapped!r}")
        mapped_names.append(mapped)
    if len(set(mapped_names)) != len(mapped_names):
        raise ValidationError("candidate image aliases are not one-to-one")
    lines = [f"{mapped_names[first]} {mapped_names[second]}" for first, second in pairs]
    _atomic_bytes(output, ("\n".join(lines) + ("\n" if lines else "")).encode())
    return {
        "candidate_manifest": str(candidate_manifest.resolve()),
        "candidate_manifest_sha256": sha256_file(candidate_manifest),
        "candidate_metadata": metadata,
        "image_count": len(names),
        "pair_count": len(pairs),
        "candidate_image_names": names,
        "colmap_image_names": mapped_names,
        "pair_list": str(output.resolve()),
        "pair_list_sha256": sha256_file(output),
    }


def _image_dir_for_camera(camera_root: Path, camera_number: int) -> Path:
    candidates = (
        camera_root / f"cam{camera_number}" / "images",
        camera_root / f"cam{camera_number}",
    )
    for path in candidates:
        if path.is_dir():
            return path.resolve()
    raise ValidationError(
        f"camera image directory for cam{camera_number} is missing under {camera_root}"
    )


def build_commands(
    *,
    colmap_binary: Path,
    database: Path,
    image_root: Path,
    camera_root: Path,
    cameras: dict[int, dict[str, Any]],
    pair_list: Path,
    output_model: Path,
    threads: int = 8,
    max_num_features: int = 2048,
    max_image_size: int = 5000,
    first_octave: int = -1,
    peak_threshold: float = 0.0066666667,
    max_num_orientations: int = 2,
    max_ratio: float = 0.8,
    cross_check: bool = True,
    min_num_inliers: int = 8,
    max_error: float = 4.0,
) -> dict[str, Any]:
    """Build per-camera feature, exact-pair matching, and mapper commands."""

    if threads <= 0 or max_num_features <= 0 or min_num_inliers <= 0:
        raise ValidationError("threads, max_num_features, and min_num_inliers must be positive")
    feature: list[list[str]] = []
    # Use physical camera number for a stable, human-readable phase order;
    # official CAMERA_ID values are not guaranteed to be cam4, cam5, ...
    for camera_id in sorted(cameras, key=lambda value: cameras[value]["camera_number"]):
        camera = cameras[camera_id]
        camera_number = camera.get("camera_number")
        if not isinstance(camera_number, int):
            raise ValidationError(f"camera {camera_id} is missing its camN mapping")
        feature.append(
            [
                str(colmap_binary),
                "feature_extractor",
                "--database_path",
                str(database),
                "--image_path",
                str(_image_dir_for_camera(camera_root, camera_number)),
                "--ImageReader.camera_model",
                "PINHOLE",
                "--ImageReader.single_camera",
                "1",
                "--ImageReader.camera_params",
                _format_params(camera["params"]),
                "--SiftExtraction.use_gpu",
                "0",
                "--SiftExtraction.num_threads",
                str(threads),
                "--SiftExtraction.max_num_features",
                str(max_num_features),
                "--SiftExtraction.max_image_size",
                str(max_image_size),
                "--SiftExtraction.first_octave",
                str(first_octave),
                "--SiftExtraction.peak_threshold",
                format(peak_threshold, ".17g"),
                "--SiftExtraction.max_num_orientations",
                str(max_num_orientations),
            ]
        )
    match = [
        str(colmap_binary),
        "matches_importer",
        "--database_path",
        str(database),
        "--match_list_path",
        str(pair_list),
        "--match_type",
        "pairs",
        "--SiftMatching.use_gpu",
        "0",
        "--SiftMatching.num_threads",
        str(threads),
        "--SiftMatching.max_ratio",
        format(max_ratio, ".17g"),
        "--SiftMatching.cross_check",
        "1" if cross_check else "0",
        "--TwoViewGeometry.min_num_inliers",
        str(min_num_inliers),
        "--TwoViewGeometry.max_error",
        format(max_error, ".17g"),
        "--TwoViewGeometry.multiple_models",
        "1",
    ]
    mapper = [
        str(colmap_binary),
        "mapper",
        "--image_path",
        str(image_root),
        "--database_path",
        str(database),
        "--output_path",
        str(output_model),
        "--Mapper.multiple_models",
        "0",
        "--Mapper.min_model_size",
        "10",
        "--Mapper.num_threads",
        str(threads),
        "--Mapper.abs_pose_min_num_inliers",
        str(min_num_inliers),
        "--Mapper.ba_refine_focal_length",
        "0",
        "--Mapper.ba_refine_principal_point",
        "0",
        "--Mapper.ba_refine_extra_params",
        "0",
    ]
    return {"feature_extractor": feature, "matches_importer": match, "mapper": mapper}


def _parse_time(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    result: dict[str, Any] = {}
    rss = TIME_RSS_RE.search(text)
    if rss:
        result["max_rss_kb"] = int(rss.group(1))
    wall = TIME_WALL_RE.search(text)
    if wall:
        fields = wall.group(1).split(":")
        try:
            if len(fields) == 3:
                result["gnu_elapsed_s"] = float(fields[0]) * 3600 + float(fields[1]) * 60 + float(fields[2])
            elif len(fields) == 2:
                result["gnu_elapsed_s"] = float(fields[0]) * 60 + float(fields[1])
            elif len(fields) == 1:
                result["gnu_elapsed_s"] = float(fields[0])
        except ValueError:
            pass
    for key, pattern in (("user_s", TIME_USER_RE), ("system_s", TIME_SYSTEM_RE)):
        match = pattern.search(text)
        if match:
            try:
                result[key] = float(match.group(1))
            except ValueError:
                pass
    return result


def run_timed(
    command: list[str],
    log_path: Path,
    time_path: Path,
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one phase and return wall time plus GNU-time measurements."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    time_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    environment_value = None
    if environment:
        environment_value = os.environ.copy()
        environment_value.update(environment)
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            ["/usr/bin/time", "-v", "-o", str(time_path), *command],
            cwd=str(cwd) if cwd else None,
            env=environment_value,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - started
    result = {"elapsed_s": elapsed, "returncode": completed.returncode}
    result.update(_parse_time(time_path))
    if completed.returncode != 0:
        raise ValidationError(
            f"COLMAP phase failed ({completed.returncode}); inspect {log_path}"
        )
    return result


def _validate_plan_artifacts(plan: dict[str, Any], plan_path: Path) -> None:
    """Reject a moved or tampered plan before starting any CPU-heavy phase."""

    root = plan_path.resolve().parent
    colmap = plan.get("colmap")
    if not isinstance(colmap, dict):
        raise ValidationError("COLMAP control plan has no COLMAP binding")
    binary_value = colmap.get("binary")
    binary_hash = colmap.get("binary_sha256")
    if not isinstance(binary_value, str) or not isinstance(binary_hash, str):
        raise ValidationError("COLMAP binary binding is incomplete")
    binary = Path(binary_value).expanduser().resolve()
    if not binary.is_file() or sha256_file(binary) != binary_hash:
        raise ValidationError("COLMAP binary hash mismatch; regenerate the control plan")
    protocol_config = plan.get("protocol_config")
    if not isinstance(protocol_config, dict):
        raise ValidationError("COLMAP control plan has no protocol config binding")
    config_value = protocol_config.get("path")
    config_hash = protocol_config.get("sha256")
    if not isinstance(config_value, str) or not isinstance(config_hash, str):
        raise ValidationError("COLMAP protocol config binding is incomplete")
    config_path = Path(config_value).expanduser().resolve()
    if not config_path.is_file() or sha256_file(config_path) != config_hash:
        raise ValidationError("COLMAP protocol config hash mismatch; regenerate the control plan")
    candidate = plan.get("candidate")
    if not isinstance(candidate, dict):
        raise ValidationError("COLMAP control plan has no candidate binding")

    def check_hash(value: Any, label: str) -> None:
        if not isinstance(value, str) or len(value) != 64:
            raise ValidationError(f"{label} hash is missing from the plan")
        path_value = candidate.get(label)
        if not isinstance(path_value, str):
            raise ValidationError(f"{label} path is missing from the plan")
        path = Path(path_value).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != value:
            raise ValidationError(f"{label} hash mismatch; regenerate the COLMAP plan")

    check_hash(candidate.get("candidate_manifest_sha256"), "candidate_manifest")
    pair_list = candidate.get("pair_list")
    pair_list_hash = candidate.get("pair_list_sha256")
    if not isinstance(pair_list, str) or not isinstance(pair_list_hash, str):
        raise ValidationError("COLMAP control plan pair-list binding is incomplete")
    pair_path = Path(pair_list).expanduser().resolve()
    if not pair_path.is_file() or sha256_file(pair_path) != pair_list_hash:
        raise ValidationError("candidate pair-list hash mismatch; regenerate the COLMAP plan")
    candidate_index = candidate.get("index")
    candidate_index_hash = candidate.get("index_sha256")
    if candidate_index is not None:
        if not isinstance(candidate_index, str) or not isinstance(candidate_index_hash, str):
            raise ValidationError("candidate index binding is incomplete")
        index_path = Path(candidate_index).expanduser().resolve()
        if not index_path.is_file() or sha256_file(index_path) != candidate_index_hash:
            raise ValidationError("candidate index hash mismatch; regenerate the COLMAP plan")
        validated_index = validate_candidate_shards(index_path)
        source_in_index = Path(validated_index["index"]["source_manifest"]).expanduser().resolve()
        source_in_plan = Path(candidate["candidate_manifest"]).expanduser().resolve()
        if source_in_index != source_in_plan:
            raise ValidationError("candidate source path differs between index and COLMAP plan")

    inputs = plan.get("inputs")
    if not isinstance(inputs, dict):
        raise ValidationError("COLMAP control plan has no input binding")
    for key, filename in (
        ("calibration_cameras_sha256", "cameras.txt"),
        ("calibration_images_sha256", "images.txt"),
    ):
        directory = inputs.get("calibration_dir")
        expected = inputs.get(key)
        if not isinstance(directory, str) or not isinstance(expected, str):
            raise ValidationError(f"COLMAP plan input binding {key} is incomplete")
        path = Path(directory).expanduser().resolve() / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise ValidationError(f"calibration {filename} hash mismatch; regenerate the COLMAP plan")

    source_value = candidate.get("candidate_manifest")
    source_names = candidate.get("candidate_image_names")
    if not isinstance(source_value, str) or not isinstance(source_names, list):
        raise ValidationError("COLMAP plan source image envelope is malformed")
    source_path = Path(source_value).expanduser().resolve()
    parsed_names, parsed_pairs, _ = parse_candidate_manifest_with_metadata(source_path)
    if parsed_names != source_names:
        raise ValidationError("candidate image names differ between source and COLMAP plan")
    names = candidate.get("colmap_image_names")
    count = candidate.get("pair_count")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names) or not isinstance(count, int):
        raise ValidationError("COLMAP plan candidate envelope is malformed")
    try:
        actual_lines = [line.strip() for line in pair_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(f"cannot read candidate pair list {pair_path}: {exc}") from exc
    if len(actual_lines) != count:
        raise ValidationError("candidate pair list count differs from the plan")
    if any(len(line.split()) != 2 for line in actual_lines):
        raise ValidationError("candidate pair list contains a malformed line")
    known = set(names)
    if any(value not in known for line in actual_lines for value in line.split()):
        raise ValidationError("candidate pair list contains an image name outside the plan")
    inputs_alias = inputs.get("alias_tsv")
    inputs_alias_hash = inputs.get("alias_tsv_sha256")
    aliases = parse_alias_tsv(Path(inputs_alias).expanduser().resolve()) if inputs_alias else _identity_aliases(source_names)
    if inputs_alias:
        if not isinstance(inputs_alias_hash, str):
            raise ValidationError("COLMAP plan alias-map binding is incomplete")
        alias_path = Path(inputs_alias).expanduser().resolve()
        if not alias_path.is_file() or sha256_file(alias_path) != inputs_alias_hash:
            raise ValidationError("candidate alias-map hash mismatch; regenerate the COLMAP plan")
    expected_names = [aliases.get(name) for name in source_names]
    if expected_names != names:
        raise ValidationError("candidate image aliases differ between source and COLMAP plan")
    expected_lines = [
        f"{expected_names[first]} {expected_names[second]}" for first, second in parsed_pairs
    ]
    if len(parsed_pairs) != count:
        raise ValidationError("candidate source pair count differs from the plan")
    if actual_lines != expected_lines:
        raise ValidationError("candidate pair list differs from the frozen candidate manifest")


def _camera_specs(
    calibration_dir: Path,
    image_names: list[str],
) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    cameras = parse_pinhole_cameras(calibration_dir / "cameras.txt")
    assignments = parse_image_camera_assignments(calibration_dir / "images.txt")
    camera_numbers: dict[int, int] = {}
    for name in image_names:
        camera_id = _camera_for_flat_name(name, assignments)
        if camera_id is None:
            raise ValidationError(f"no official calibration camera assignment for {name!r}")
        match = CAMERA_PREFIX_RE.fullmatch(name)
        if match is None:
            raise ValidationError(f"candidate image is not camN_timestamp.png: {name!r}")
        camera_number = int(match.group("camera"))
        previous = camera_numbers.setdefault(camera_id, camera_number)
        if previous != camera_number:
            raise ValidationError(
                f"camera id {camera_id} maps to both cam{previous} and cam{camera_number}"
            )
        if camera_id not in cameras:
            raise ValidationError(f"image {name!r} refers to missing CAMERA_ID {camera_id}")
    for camera_id, camera_number in camera_numbers.items():
        cameras[camera_id]["camera_number"] = camera_number
    return {camera_id: cameras[camera_id] for camera_id in sorted(camera_numbers)}, assignments


def make_plan(
    *,
    candidate_index: Path | None,
    candidate_manifest: Path | None,
    output_root: Path,
    image_root: Path,
    camera_root: Path,
    calibration_dir: Path,
    colmap_binary: Path,
    alias_tsv: Path | None = None,
    threads: int = 8,
    max_num_features: int = 2048,
    max_image_size: int = 5000,
    first_octave: int = -1,
    peak_threshold: float = 0.0066666667,
    max_num_orientations: int = 2,
    max_ratio: float = 0.8,
    min_num_inliers: int = 8,
    max_error: float = 4.0,
    colmap_library_path: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    if (candidate_index is None) == (candidate_manifest is None):
        raise ValidationError("pass exactly one of --candidate-index and --candidate-manifest")
    if candidate_index is not None:
        source, candidate = _candidate_source_from_index(candidate_index.resolve())
        candidate_index_hash = candidate["index_sha256"]
    else:
        source = candidate_manifest.resolve()
        if not source.is_file():
            raise ValidationError(f"candidate manifest is missing: {source}")
        names, pairs, metadata = parse_candidate_manifest_with_metadata(source)
        candidate = {
            "index_sha256": None,
            "image_names": names,
            "pair_count": len(pairs),
            "candidate_manifest_metadata": metadata,
        }
        candidate_index_hash = None
    names = candidate["image_names"]
    aliases = parse_alias_tsv(alias_tsv.resolve()) if alias_tsv else _identity_aliases(names)
    output_root = output_root.resolve()
    protocol_config_path = (config_path or DEFAULT_CONFIG).resolve()
    protocol_config = _load_json(protocol_config_path, "COLMAP protocol config")
    if protocol_config.get("schema") != CONFIG_SCHEMA:
        raise ValidationError(
            f"COLMAP protocol config has unsupported schema: {protocol_config.get('schema')!r}"
        )
    runtime_env: dict[str, str] = {}
    if colmap_library_path is not None:
        library_path = colmap_library_path.resolve()
        if not library_path.is_dir():
            raise ValidationError(f"COLMAP library directory is missing: {library_path}")
        runtime_env["LD_LIBRARY_PATH"] = str(library_path)
    pair_list = output_root / "candidate_pairs.txt"
    pair_info = write_pair_list(source, pair_list, aliases=aliases)
    if pair_info["image_count"] != len(names) or pair_info["pair_count"] != candidate["pair_count"]:
        raise ValidationError("pair list envelope differs from candidate index")
    cameras, _ = _camera_specs(calibration_dir.resolve(), names)
    database = output_root / "database.db"
    model = output_root / "models"
    commands = build_commands(
        colmap_binary=colmap_binary.resolve(),
        database=database,
        image_root=image_root.resolve(),
        camera_root=camera_root.resolve(),
        cameras=cameras,
        pair_list=pair_list,
        output_model=model,
        threads=threads,
        max_num_features=max_num_features,
        max_image_size=max_image_size,
        first_octave=first_octave,
        peak_threshold=peak_threshold,
        max_num_orientations=max_num_orientations,
        max_ratio=max_ratio,
        min_num_inliers=min_num_inliers,
        max_error=max_error,
    )
    plan = {
        "schema": PLAN_SCHEMA,
        "protocol_config": {
            "path": str(protocol_config_path),
            "sha256": sha256_file(protocol_config_path),
        },
        "candidate": {
            "index": str(candidate_index.resolve()) if candidate_index else None,
            "index_sha256": candidate_index_hash,
            **pair_info,
        },
        "inputs": {
            "image_root": str(image_root.resolve()),
            "camera_root": str(camera_root.resolve()),
            "calibration_dir": str(calibration_dir.resolve()),
            "calibration_cameras_sha256": sha256_file(calibration_dir / "cameras.txt"),
            "calibration_images_sha256": sha256_file(calibration_dir / "images.txt"),
            "alias_tsv": str(alias_tsv.resolve()) if alias_tsv else None,
            "alias_tsv_sha256": sha256_file(alias_tsv) if alias_tsv else None,
        },
        "colmap": {
            "binary": str(colmap_binary.resolve()),
            "binary_sha256": sha256_file(colmap_binary),
            "runtime_env": runtime_env,
            "commands": commands,
            "cameras": cameras,
            "database": str(database),
            "model": str(model),
            "feature_settings": {
                "max_num_features": max_num_features,
                "max_image_size": max_image_size,
                "first_octave": first_octave,
                "peak_threshold": peak_threshold,
                "max_num_orientations": max_num_orientations,
            },
            "matching_settings": {
                "mode": "matches_importer",
                "match_type": "pairs",
                "max_ratio": max_ratio,
                "cross_check": True,
                "min_num_inliers": min_num_inliers,
                "max_error": max_error,
            },
            "mapper_settings": {
                "multiple_models": False,
                "min_model_size": 10,
                "num_threads": threads,
                "abs_pose_min_num_inliers": min_num_inliers,
                "ba_refine_focal_length": False,
                "ba_refine_principal_point": False,
                "ba_refine_extra_params": False,
            },
        },
        "ground_truth_used_for_selection_or_mapping": False,
    }
    atomic_json(output_root / "plan.json", plan)
    return plan


def run_plan(plan_path: Path) -> dict[str, Any]:
    plan = _load_json(plan_path.resolve(), "COLMAP control plan")
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValidationError(f"unsupported COLMAP control plan schema: {plan.get('schema')!r}")
    _validate_plan_artifacts(plan, plan_path)
    root = plan_path.resolve().parent
    commands = plan.get("colmap", {}).get("commands")
    if not isinstance(commands, dict):
        raise ValidationError("COLMAP control plan has no commands")
    runtime_env = plan.get("colmap", {}).get("runtime_env", {})
    if not isinstance(runtime_env, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in runtime_env.items()
    ):
        raise ValidationError("COLMAP control plan runtime_env is malformed")
    model_value = plan.get("colmap", {}).get("model")
    if not isinstance(model_value, str) or not model_value:
        raise ValidationError("COLMAP control plan model output path is missing")
    # COLMAP 3.9 expects mapper --output_path to already exist.  Creating the
    # external run directory is safe and makes an interrupted prepare/run
    # restartable without a hand-written shell prelude.
    Path(model_value).expanduser().resolve().mkdir(parents=True, exist_ok=True)
    phases: dict[str, Any] = {}
    camera_specs = plan.get("colmap", {}).get("cameras", {})
    camera_numbers = [
        camera.get("camera_number")
        for _, camera in sorted(
            camera_specs.items(), key=lambda item: item[1].get("camera_number", 0)
        )
        if isinstance(camera, dict)
    ]
    for index, command in enumerate(commands.get("feature_extractor", [])):
        if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
            raise ValidationError(f"feature command {index} is malformed")
        camera_label = camera_numbers[index] if index < len(camera_numbers) else index
        phases[f"feature_extractor_cam{camera_label}"] = run_timed(
            command,
            root / "logs" / f"feature_cam{camera_label}.log",
            root / "timing" / f"feature_cam{camera_label}.time.txt",
            environment=runtime_env,
        )
    for phase in ("matches_importer", "mapper"):
        command = commands.get(phase)
        if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
            raise ValidationError(f"COLMAP {phase} command is malformed")
        phases[phase] = run_timed(
            command,
            root / "logs" / f"{phase}.log",
            root / "timing" / f"{phase}.time.txt",
            environment=runtime_env,
        )
    result = {
        "schema": "visloc_electro_colmap_control_result_v1",
        "plan": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
        "phases": phases,
        "ground_truth_used_for_selection_or_mapping": False,
    }
    atomic_json(root / "result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="write a pair list and external execution plan")
    mode.add_argument("--run", action="store_true", help="execute an existing plan (CPU-heavy, explicit opt-in)")
    parser.add_argument("--plan", type=Path, help="existing plan.json for --run")
    parser.add_argument("--candidate-index", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--camera-root", type=Path)
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--colmap-binary", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--colmap-library-path",
        type=Path,
        help="optional directory containing COLMAP's bundled shared libraries (recorded as LD_LIBRARY_PATH)",
    )
    parser.add_argument("--alias-tsv", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-num-features", type=int, default=2048)
    parser.add_argument("--max-image-size", type=int, default=5000)
    parser.add_argument("--first-octave", type=int, default=-1)
    parser.add_argument("--peak-threshold", type=float, default=0.0066666667)
    parser.add_argument("--max-num-orientations", type=int, default=2)
    parser.add_argument("--max-ratio", type=float, default=0.8)
    parser.add_argument("--min-num-inliers", type=int, default=8)
    parser.add_argument("--max-error", type=float, default=4.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.run:
            if args.plan is None:
                raise ValidationError("--run requires --plan")
            print(json.dumps(run_plan(args.plan), sort_keys=True, indent=2))
            return 0
        required = {
            "--output-root": args.output_root,
            "--image-root": args.image_root,
            "--camera-root": args.camera_root,
            "--calibration-dir": args.calibration_dir,
            "--colmap-binary": args.colmap_binary,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValidationError(f"--prepare requires {', '.join(missing)}")
        plan = make_plan(
            candidate_index=args.candidate_index,
            candidate_manifest=args.candidate_manifest,
            output_root=args.output_root,
            image_root=args.image_root,
            camera_root=args.camera_root,
            calibration_dir=args.calibration_dir,
            colmap_binary=args.colmap_binary,
            colmap_library_path=args.colmap_library_path,
            config_path=args.config,
            alias_tsv=args.alias_tsv,
            threads=args.threads,
            max_num_features=args.max_num_features,
            max_image_size=args.max_image_size,
            first_octave=args.first_octave,
            peak_threshold=args.peak_threshold,
            max_num_orientations=args.max_num_orientations,
            max_ratio=args.max_ratio,
            min_num_inliers=args.min_num_inliers,
            max_error=args.max_error,
        )
        print(json.dumps(plan, sort_keys=True, indent=2))
        return 0
    except (ValidationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
