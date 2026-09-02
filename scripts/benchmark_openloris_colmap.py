#!/usr/bin/env python3
"""Prepare and run a hash-bound OpenLORIS COLMAP control in Docker.

The first protocol intentionally reuses the frozen visloc candidate manifest:
COLMAP extracts its own CPU SIFT features, matches exactly those pairs, and
runs its ordinary incremental mapper.  Ground truth is never mounted into the
container.  Every heavy phase runs through ``docker_process_metrics.sh`` so
the result records both process VmHWM and the whole-container cgroup peak.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_electro import ValidationError, sha256_file  # noqa: E402
from benchmark_electro_colmap import (  # noqa: E402
    _candidate_source_from_index,
    _camera_specs,
    atomic_json,
    write_pair_list,
)


PLAN_SCHEMA = "visloc_openloris_colmap_control_plan_v1"
RESULT_SCHEMA = "visloc_openloris_colmap_control_result_v1"
METRICS_SCHEMA = "visloc_docker_process_metrics_v1"
IMAGE_RE = re.compile(r"^cam(?P<camera>[0-9]+)_[0-9]+\.[^.]+$")


def docker_image_identity(image: str) -> dict[str, Any]:
    try:
        payload = subprocess.check_output(
            ["docker", "image", "inspect", image], text=True
        )
        records = json.loads(payload)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot inspect Docker image {image!r}: {exc}") from exc
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
        raise ValidationError(f"Docker image inspect returned an invalid record for {image!r}")
    record = records[0]
    image_id = record.get("Id")
    digests = record.get("RepoDigests")
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise ValidationError(f"Docker image {image!r} has no immutable image id")
    if not isinstance(digests, list) or not all(isinstance(value, str) for value in digests):
        raise ValidationError(f"Docker image {image!r} has malformed RepoDigests")
    return {"reference": image, "id": image_id, "repo_digests": sorted(digests)}


def parse_metrics(path: Path) -> dict[str, Any]:
    try:
        rows = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(f"cannot read Docker metrics {path}: {exc}") from exc
    values: dict[str, str] = {}
    for line_number, row in enumerate(rows, 1):
        fields = row.split("\t")
        if len(fields) != 2 or not fields[0] or fields[0] in values:
            raise ValidationError(f"Docker metrics {path}:{line_number} is malformed")
        values[fields[0]] = fields[1]
    if values.get("schema") != METRICS_SCHEMA:
        raise ValidationError(f"Docker metrics {path} has unsupported schema")
    result: dict[str, Any] = {"schema": values["schema"]}
    for key in ("status", "wall_ns", "peak_process_hwm_kib", "cgroup_peak_bytes"):
        try:
            result[key] = int(values[key])
        except (KeyError, ValueError) as exc:
            raise ValidationError(f"Docker metrics {path} has invalid {key}") from exc
    try:
        result["poll_seconds"] = float(values["poll_seconds"])
    except (KeyError, ValueError) as exc:
        raise ValidationError(f"Docker metrics {path} has invalid poll_seconds") from exc
    result["wall_seconds"] = result["wall_ns"] / 1_000_000_000.0
    result["cgroup_peak_kib"] = result["cgroup_peak_bytes"] // 1024
    return result


def _log_diagnostics(path: Path) -> dict[str, int]:
    """Count retained COLMAP severities and the recurrent sparse-solver failure."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise ValidationError(f"cannot read COLMAP log {path}: {exc}") from exc
    return {
        "warning_lines": sum(line.startswith("W") for line in lines),
        "error_lines": sum(line.startswith("E") for line in lines),
        "linear_solver_failure_lines": sum("Linear solver failure" in line for line in lines),
    }


def _stage_images(image_root: Path, staging_root: Path, names: list[str]) -> dict[str, Any]:
    all_root = staging_root / "all"
    all_root.mkdir(parents=True, exist_ok=True)
    cameras: dict[int, int] = {}
    for name in names:
        match = IMAGE_RE.fullmatch(name)
        if match is None:
            raise ValidationError(f"OpenLORIS image name is not camN_index: {name!r}")
        camera = int(match.group("camera"))
        source = (image_root / name).resolve()
        if not source.is_file():
            raise ValidationError(f"OpenLORIS image is missing: {source}")
        for destination in (all_root / name, staging_root / f"cam{camera}" / name):
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if not destination.samefile(source):
                    raise ValidationError(f"staged image does not match source: {destination}")
            else:
                try:
                    os.link(source, destination)
                except OSError as exc:
                    raise ValidationError(
                        f"cannot hard-link {source} to {destination}; keep output on the same filesystem: {exc}"
                    ) from exc
        cameras[camera] = cameras.get(camera, 0) + 1
    return {"images": len(names), "cameras": {str(key): cameras[key] for key in sorted(cameras)}}


def _atomic_text(path: Path, payload: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _load_tier_records(path: Path, names: list[str]) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["images"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as exc:
        raise ValidationError(f"cannot read OpenLORIS tier manifest {path}: {exc}") from exc
    if payload.get("schema") != "visloc_openloris_corridor_manifest_v1" or not isinstance(rows, list):
        raise ValidationError(f"unsupported OpenLORIS tier manifest schema: {path}")
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        try:
            name = row["name"]
            camera = int(row["camera"])
            timestamp = str(row["timestamp"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"malformed OpenLORIS tier row in {path}") from exc
        if not isinstance(name, str) or name in records or camera not in (1, 2):
            raise ValidationError(f"invalid or duplicate OpenLORIS tier image {name!r}")
        records[name] = {"camera": camera, "timestamp": timestamp}
    if set(records) != set(names):
        raise ValidationError("tier manifest image envelope differs from candidate manifest")
    return records


def _stage_rig_images(
    image_root: Path,
    staging_root: Path,
    names: list[str],
    records: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    aliases: dict[str, str] = {}
    cameras: dict[int, int] = {}
    frame_cameras: dict[str, set[int]] = {}
    for name in names:
        record = records[name]
        camera, timestamp = record["camera"], record["timestamp"]
        alias = f"rig/camera{camera}/{timestamp}.png"
        if alias in aliases.values():
            raise ValidationError(f"rig staging alias is not unique: {alias}")
        source = (image_root / name).resolve()
        destination = staging_root / alias
        if not source.is_file():
            raise ValidationError(f"OpenLORIS image is missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not destination.samefile(source):
                raise ValidationError(f"staged rig image does not match source: {destination}")
        else:
            try:
                os.link(source, destination)
            except OSError as exc:
                raise ValidationError(
                    f"cannot hard-link {source} to {destination}; keep output on the same filesystem: {exc}"
                ) from exc
        aliases[name] = alias
        cameras[camera] = cameras.get(camera, 0) + 1
        frame_cameras.setdefault(timestamp, set()).add(camera)
    incomplete = [timestamp for timestamp, values in frame_cameras.items() if values != {1, 2}]
    if incomplete:
        raise ValidationError(f"rig staging has incomplete frames; first={min(incomplete)}")
    return (
        {
            "images": len(names),
            "frames": len(frame_cameras),
            "cameras": {str(key): cameras[key] for key in sorted(cameras)},
            "layout": "rig/cameraN/TIMESTAMP.png",
        },
        aliases,
    )


def _inverse_relative_rig_pose(path: Path) -> tuple[list[float], list[float]]:
    """Return cam2_from_cam1 quaternion (wxyz) and translation."""
    try:
        payload = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(f"cannot read OpenLORIS transform matrix {path}: {exc}") from exc
    pattern = re.compile(
        r"parent_frame:\s*t265_fisheye1_optical_frame\s+"
        r"child_frame:\s*t265_fisheye2_optical_frame\s+matrix:.*?"
        r"data:\s*\[(.*?)\]",
        re.DOTALL,
    )
    match = pattern.search(payload)
    if match is None:
        raise ValidationError("OpenLORIS transform matrix lacks fisheye1 -> fisheye2")
    try:
        values = [float(value.strip()) for value in match.group(1).split(",") if value.strip()]
    except ValueError as exc:
        raise ValidationError("OpenLORIS fisheye rig transform is non-numeric") from exc
    if len(values) != 16:
        raise ValidationError("OpenLORIS fisheye rig transform is malformed")
    if any(abs(values[12 + index] - expected) > 1e-9 for index, expected in enumerate((0, 0, 0, 1))):
        raise ValidationError("OpenLORIS fisheye rig transform has an invalid homogeneous row")
    rotation = [[values[4 * row + column] for column in range(3)] for row in range(3)]
    for left in range(3):
        for right in range(3):
            dot = sum(rotation[row][left] * rotation[row][right] for row in range(3))
            if abs(dot - (1.0 if left == right else 0.0)) > 1e-5:
                raise ValidationError("OpenLORIS fisheye rig rotation is not orthonormal")
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(determinant - 1.0) > 1e-5:
        raise ValidationError("OpenLORIS fisheye rig rotation is not proper")
    centre = [values[3], values[7], values[11]]
    inverse_rotation = [[rotation[column][row] for column in range(3)] for row in range(3)]
    translation = [
        -sum(inverse_rotation[row][column] * centre[column] for column in range(3))
        for row in range(3)
    ]
    trace = sum(inverse_rotation[index][index] for index in range(3))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = [
            0.25 * scale,
            (inverse_rotation[2][1] - inverse_rotation[1][2]) / scale,
            (inverse_rotation[0][2] - inverse_rotation[2][0]) / scale,
            (inverse_rotation[1][0] - inverse_rotation[0][1]) / scale,
        ]
    else:
        axis = max(range(3), key=lambda index: inverse_rotation[index][index])
        first, second = (axis + 1) % 3, (axis + 2) % 3
        scale = math.sqrt(
            1.0 + inverse_rotation[axis][axis]
            - inverse_rotation[first][first] - inverse_rotation[second][second]
        ) * 2.0
        vector = [0.0, 0.0, 0.0]
        vector[axis] = 0.25 * scale
        vector[first] = (inverse_rotation[first][axis] + inverse_rotation[axis][first]) / scale
        vector[second] = (inverse_rotation[second][axis] + inverse_rotation[axis][second]) / scale
        quaternion = [
            (inverse_rotation[second][first] - inverse_rotation[first][second]) / scale,
            *vector,
        ]
    norm = math.sqrt(sum(value * value for value in quaternion))
    return [value / norm for value in quaternion], translation


def _rig_container_commands(
    cameras: dict[int, dict[str, Any]],
    *,
    rotation: list[float],
    translation: list[float],
    threads: int,
    max_num_features: int,
    max_ratio: float,
    min_num_inliers: int,
    max_error: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_number = {camera["camera_number"]: camera for camera in cameras.values()}
    if set(by_number) != {1, 2}:
        raise ValidationError("the OpenLORIS rig control requires exactly camera 1 and camera 2")

    rig_cameras: list[dict[str, Any]] = []
    for number in (1, 2):
        camera = by_number[number]
        params = list(camera["params"])
        params[2] += 0.5
        params[3] += 0.5
        entry: dict[str, Any] = {
            "image_prefix": f"rig/camera{number}/",
            "camera_model_name": "PINHOLE",
            "camera_params": params,
        }
        if number == 1:
            entry["ref_sensor"] = True
        else:
            entry["cam_from_rig_rotation"] = rotation
            entry["cam_from_rig_translation"] = translation
        rig_cameras.append(entry)

    commands = _container_commands(
        cameras,
        threads=threads,
        max_num_features=max_num_features,
        max_ratio=max_ratio,
        min_num_inliers=min_num_inliers,
        max_error=max_error,
    )
    commands["feature_extractor"] = [
        {
            "phase": "feature_extractor_rig",
            "camera_number": "rig",
            "command": [
                "colmap", "feature_extractor",
                "--database_path", "/output/database.db",
                "--image_path", "/output/staged-images",
                "--ImageReader.single_camera_per_folder", "1",
                "--FeatureExtraction.use_gpu", "0",
                "--FeatureExtraction.num_threads", str(threads),
                "--FeatureExtraction.max_image_size", "848",
                "--SiftExtraction.max_num_features", str(max_num_features),
                "--SiftExtraction.first_octave", "-1",
                "--SiftExtraction.peak_threshold", "0.0066666667",
                "--SiftExtraction.max_num_orientations", "2",
            ],
        }
    ]
    commands["rig_configurator"] = [
        "colmap", "rig_configurator",
        "--database_path", "/output/database.db",
        "--rig_config_path", "/output/rig_config.json",
    ]
    mapper = commands["mapper"]
    mapper[mapper.index("/output/staged-images/all")] = "/output/staged-images"
    mapper.extend(["--Mapper.ba_refine_sensor_from_rig", "0"])
    return commands, [{"cameras": rig_cameras}]


def _container_commands(
    cameras: dict[int, dict[str, Any]],
    *,
    threads: int,
    max_num_features: int,
    max_ratio: float,
    min_num_inliers: int,
    max_error: float,
) -> dict[str, Any]:
    feature: list[dict[str, Any]] = []
    for camera_id, camera in sorted(
        cameras.items(), key=lambda item: item[1]["camera_number"]
    ):
        number = camera["camera_number"]
        # OpenLORIS staging intrinsics use OpenCV's integer pixel-centre
        # convention. COLMAP's documented convention places the top-left
        # pixel centre at (0.5, 0.5), hence the principal-point shift.
        params = list(camera["params"])
        params[2] += 0.5
        params[3] += 0.5
        feature.append(
            {
                "camera_id": camera_id,
                "camera_number": number,
                "command": [
                    "colmap",
                    "feature_extractor",
                    "--database_path",
                    "/output/database.db",
                    "--image_path",
                    f"/output/staged-images/cam{number}",
                    "--ImageReader.camera_model",
                    "PINHOLE",
                    "--ImageReader.single_camera",
                    "1",
                    "--ImageReader.camera_params",
                    ",".join(format(value, ".17g") for value in params),
                    "--FeatureExtraction.use_gpu",
                    "0",
                    "--FeatureExtraction.num_threads",
                    str(threads),
                    "--FeatureExtraction.max_image_size",
                    "848",
                    "--SiftExtraction.max_num_features",
                    str(max_num_features),
                    "--SiftExtraction.first_octave",
                    "-1",
                    "--SiftExtraction.peak_threshold",
                    "0.0066666667",
                    "--SiftExtraction.max_num_orientations",
                    "2",
                ],
            }
        )
    return {
        "feature_extractor": feature,
        "matches_importer": [
            "colmap",
            "matches_importer",
            "--database_path",
            "/output/database.db",
            "--match_list_path",
            "/output/candidate_pairs.txt",
            "--match_type",
            "pairs",
            "--FeatureMatching.use_gpu",
            "0",
            "--FeatureMatching.num_threads",
            str(threads),
            "--FeatureMatching.guided_matching",
            "0",
            "--SiftMatching.max_ratio",
            format(max_ratio, ".17g"),
            "--SiftMatching.cross_check",
            "1",
            "--TwoViewGeometry.min_num_inliers",
            str(min_num_inliers),
            "--TwoViewGeometry.max_error",
            format(max_error, ".17g"),
            "--TwoViewGeometry.multiple_models",
            "1",
            "--TwoViewGeometry.random_seed",
            "0",
        ],
        "mapper": [
            "colmap",
            "mapper",
            "--database_path",
            "/output/database.db",
            "--image_path",
            "/output/staged-images/all",
            "--output_path",
            "/output/models",
            "--Mapper.multiple_models",
            "1",
            "--Mapper.max_num_models",
            "50",
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
            "--Mapper.ba_use_gpu",
            "0",
            "--Mapper.random_seed",
            "0",
        ],
    }


def prepare_plan(
    *,
    candidate_index: Path,
    image_root: Path,
    calibration_dir: Path,
    output_root: Path,
    docker_image: str,
    tier_manifest: Path | None = None,
    rig_transform_matrix: Path | None = None,
    docker_identity: dict[str, Any] | None = None,
    threads: int = 8,
    max_num_features: int = 256,
    max_ratio: float = 0.8,
    min_num_inliers: int = 8,
    max_error: float = 4.0,
) -> dict[str, Any]:
    if threads <= 0 or max_num_features <= 0 or min_num_inliers <= 0:
        raise ValidationError("threads, max_num_features, and min_num_inliers must be positive")
    candidate_index = candidate_index.resolve()
    image_root = image_root.resolve()
    calibration_dir = calibration_dir.resolve()
    output_root = output_root.resolve()
    if (tier_manifest is None) != (rig_transform_matrix is None):
        raise ValidationError("--tier-manifest and --rig-transform-matrix must be passed together")
    rig_aware = tier_manifest is not None
    if tier_manifest is not None and rig_transform_matrix is not None:
        tier_manifest = tier_manifest.resolve()
        rig_transform_matrix = rig_transform_matrix.resolve()
    source, candidate = _candidate_source_from_index(candidate_index)
    output_root.mkdir(parents=True, exist_ok=False)
    aliases: dict[str, str] | None = None
    if rig_aware:
        assert tier_manifest is not None
        records = _load_tier_records(tier_manifest, candidate["image_names"])
        staging, aliases = _stage_rig_images(
            image_root, output_root / "staged-images", candidate["image_names"], records
        )
        _atomic_text(
            output_root / "image_aliases.tsv",
            "flat_name\tcolmap_name\n"
            + "".join(f"{name}\t{aliases[name]}\n" for name in candidate["image_names"]),
        )
    else:
        staging = _stage_images(image_root, output_root / "staged-images", candidate["image_names"])
    pair_info = write_pair_list(source, output_root / "candidate_pairs.txt", aliases=aliases)
    if pair_info["candidate_image_names"] != candidate["image_names"]:
        raise ValidationError("candidate pair-list names differ from validated index")
    cameras, _ = _camera_specs(calibration_dir, candidate["image_names"])
    identity = docker_identity or docker_image_identity(docker_image)
    if identity.get("reference") != docker_image:
        raise ValidationError("Docker identity reference differs from requested image")
    if rig_aware:
        assert rig_transform_matrix is not None
        rotation, translation = _inverse_relative_rig_pose(rig_transform_matrix)
        commands, rig_config = _rig_container_commands(
            cameras,
            rotation=rotation,
            translation=translation,
            threads=threads,
            max_num_features=max_num_features,
            max_ratio=max_ratio,
            min_num_inliers=min_num_inliers,
            max_error=max_error,
        )
        atomic_json(output_root / "rig_config.json", rig_config)
    else:
        commands = _container_commands(
            cameras,
            threads=threads,
            max_num_features=max_num_features,
            max_ratio=max_ratio,
            min_num_inliers=min_num_inliers,
            max_error=max_error,
        )
    plan = {
        "schema": PLAN_SCHEMA,
        "protocol": (
            "same-candidate-calibrated-stereo-rig-cpu-sift-incremental-v3"
            if rig_aware
            else "same-candidate-cpu-sift-incremental-v2"
        ),
        "candidate": {
            "index": str(candidate_index),
            "index_sha256": sha256_file(candidate_index),
            "source_manifest": str(source),
            "source_manifest_sha256": sha256_file(source),
            "pair_count": candidate["pair_count"],
            "pair_list_sha256": sha256_file(output_root / "candidate_pairs.txt"),
            "image_names": candidate["image_names"],
        },
        "inputs": {
            "image_root": str(image_root),
            "calibration_dir": str(calibration_dir),
            "calibration_cameras_sha256": sha256_file(calibration_dir / "cameras.txt"),
            "calibration_images_sha256": sha256_file(calibration_dir / "images.txt"),
            "staging": staging,
            "tier_manifest": str(tier_manifest) if tier_manifest else None,
            "tier_manifest_sha256": sha256_file(tier_manifest) if tier_manifest else None,
            "rig_transform_matrix": str(rig_transform_matrix) if rig_transform_matrix else None,
            "rig_transform_matrix_sha256": (
                sha256_file(rig_transform_matrix) if rig_transform_matrix else None
            ),
            "image_aliases_sha256": (
                sha256_file(output_root / "image_aliases.tsv") if rig_aware else None
            ),
            "rig_config_sha256": sha256_file(output_root / "rig_config.json") if rig_aware else None,
        },
        "output_root": str(output_root),
        "docker": identity,
        "software": {
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "metrics_wrapper": str((REPO / "scripts" / "docker_process_metrics.sh").resolve()),
            "metrics_wrapper_sha256": sha256_file(REPO / "scripts" / "docker_process_metrics.sh"),
        },
        "settings": {
            "threads": threads,
            "max_num_features": max_num_features,
            "max_ratio": max_ratio,
            "min_num_inliers": min_num_inliers,
            "max_error": max_error,
            "gpu": False,
            "opencv_to_colmap_principal_point_shift_px": 0.5,
            "fixed_calibrated_stereo_rig": rig_aware,
        },
        "commands": commands,
        "ground_truth_used_for_selection_or_mapping": False,
    }
    atomic_json(output_root / "plan.json", plan)
    return plan


def _validate_plan(plan: dict[str, Any], plan_path: Path) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValidationError(f"unsupported OpenLORIS COLMAP plan schema: {plan.get('schema')!r}")
    root = plan_path.resolve().parent
    if Path(plan.get("output_root", "")).resolve() != root:
        raise ValidationError("OpenLORIS COLMAP plan moved from its bound output root")
    candidate = plan.get("candidate")
    inputs = plan.get("inputs")
    if not isinstance(candidate, dict) or not isinstance(inputs, dict):
        raise ValidationError("OpenLORIS COLMAP plan input bindings are malformed")
    checks = [
        (Path(candidate["index"]), candidate["index_sha256"], "candidate index"),
        (Path(candidate["source_manifest"]), candidate["source_manifest_sha256"], "candidate manifest"),
        (root / "candidate_pairs.txt", candidate["pair_list_sha256"], "candidate pair list"),
        (Path(inputs["calibration_dir"]) / "cameras.txt", inputs["calibration_cameras_sha256"], "calibration cameras"),
        (Path(inputs["calibration_dir"]) / "images.txt", inputs["calibration_images_sha256"], "calibration images"),
    ]
    settings = plan.get("settings")
    if not isinstance(settings, dict):
        raise ValidationError("OpenLORIS COLMAP plan settings are malformed")
    rig_aware = bool(settings.get("fixed_calibrated_stereo_rig"))
    software = plan.get("software")
    if software is not None:
        if not isinstance(software, dict):
            raise ValidationError("OpenLORIS COLMAP plan software bindings are malformed")
        checks.extend(
            [
                (Path(software["runner"]), software["runner_sha256"], "control runner"),
                (
                    Path(software["metrics_wrapper"]),
                    software["metrics_wrapper_sha256"],
                    "metrics wrapper",
                ),
            ]
        )
    elif rig_aware:
        raise ValidationError("calibrated rig plans must bind the runner and metrics wrapper")
    if rig_aware:
        checks.extend(
            [
                (Path(inputs["tier_manifest"]), inputs["tier_manifest_sha256"], "tier manifest"),
                (
                    Path(inputs["rig_transform_matrix"]),
                    inputs["rig_transform_matrix_sha256"],
                    "rig transform matrix",
                ),
                (root / "image_aliases.tsv", inputs["image_aliases_sha256"], "image aliases"),
                (root / "rig_config.json", inputs["rig_config_sha256"], "rig configuration"),
            ]
        )
    for path, expected, label in checks:
        if not path.is_file() or sha256_file(path) != expected:
            raise ValidationError(f"{label} hash mismatch: {path}")
    names = candidate.get("image_names")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValidationError("candidate image-name envelope is malformed")
    records = _load_tier_records(Path(inputs["tier_manifest"]), names) if rig_aware else None
    for name in names:
        source = Path(inputs["image_root"]) / name
        if records is not None:
            record = records[name]
            staged = (
                root / "staged-images" / "rig" / f"camera{record['camera']}"
                / f"{record['timestamp']}.png"
            )
        else:
            staged = root / "staged-images" / "all" / name
        if not source.is_file() or not staged.is_file() or not staged.samefile(source):
            raise ValidationError(f"staged image binding differs for {name}")
    docker = plan.get("docker")
    if not isinstance(docker, dict):
        raise ValidationError("Docker image binding is missing")
    current = docker_image_identity(docker.get("reference", ""))
    if current.get("id") != docker.get("id"):
        raise ValidationError("Docker image id changed; prepare a new control plan")


def run_docker_phase(
    *,
    image: str,
    output_root: Path,
    phase: str,
    command: list[str],
    poll_seconds: float = 0.05,
) -> dict[str, Any]:
    wrapper = REPO / "scripts" / "docker_process_metrics.sh"
    logs = output_root / "logs"
    timing = output_root / "timing"
    logs.mkdir(exist_ok=True)
    timing.mkdir(exist_ok=True)
    metrics_path = timing / f"{phase}.tsv"
    docker_command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--entrypoint",
        "/bin/bash",
        "--env",
        "NVIDIA_VISIBLE_DEVICES=void",
        "--env",
        f"VISLOC_METRICS_OUTPUT=/output/timing/{phase}.tsv",
        "--env",
        f"VISLOC_METRICS_POLL_SECONDS={poll_seconds}",
        "--mount",
        f"type=bind,src={output_root},dst=/output",
        "--mount",
        f"type=bind,src={wrapper},dst=/metrics-wrapper.sh,readonly",
        image,
        "/metrics-wrapper.sh",
        *command,
    ]
    started = time.monotonic()
    with (logs / f"{phase}.log").open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            docker_command, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    host_wall_seconds = time.monotonic() - started
    metrics = parse_metrics(metrics_path)
    metrics["host_wall_seconds"] = host_wall_seconds
    metrics["docker_command"] = docker_command
    metrics["colmap_command"] = command
    metrics["log_diagnostics"] = _log_diagnostics(logs / f"{phase}.log")
    if completed.returncode != metrics["status"]:
        raise ValidationError(f"Docker/client status differs from wrapped command in phase {phase}")
    if completed.returncode != 0:
        raise ValidationError(f"COLMAP phase {phase} failed; inspect {logs / f'{phase}.log'}")
    return metrics


def _registered_names(images_txt: Path) -> set[str]:
    names: set[str] = set()
    for line in images_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) != 10:
            continue
        try:
            int(fields[0])
            [float(value) for value in fields[1:8]]
            int(fields[8])
        except ValueError:
            continue
        names.add(fields[9])
    return names


def _registered_frame_keys(names: set[str], *, rig_aware: bool) -> set[str]:
    if not rig_aware:
        return set()
    keys: set[str] = set()
    for name in names:
        path = Path(name)
        if len(path.parts) != 3 or path.parts[0] != "rig" or path.parts[1] not in {
            "camera1", "camera2",
        }:
            raise ValidationError(f"registered rig image has an unexpected name: {name!r}")
        keys.add(path.name)
    return keys


def _point_stats(points_txt: Path) -> tuple[int, int, float | None]:
    points = 0
    observations = 0
    weighted_error = 0.0
    for line in points_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8 or (len(fields) - 8) % 2:
            continue
        error = float(fields[7])
        count = (len(fields) - 8) // 2
        points += 1
        observations += count
        weighted_error += error * count
    return points, observations, weighted_error / observations if observations else None


def _database_stats(database: Path) -> dict[str, Any]:
    """Return auditable counts for the exact-pair COLMAP database."""
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        images = connection.execute("SELECT image_id, name FROM images").fetchall()
        keypoints = connection.execute(
            "SELECT COUNT(*) AS records, COALESCE(SUM(rows), 0) AS features, "
            "COALESCE(MAX(rows), 0) AS max_per_image FROM keypoints"
        ).fetchone()
        descriptors = connection.execute(
            "SELECT COUNT(*) AS records, COALESCE(SUM(rows), 0) AS features FROM descriptors"
        ).fetchone()
        raw = connection.execute(
            "SELECT COUNT(*) AS records, COALESCE(SUM(rows), 0) AS correspondences, "
            "COALESCE(SUM(CASE WHEN rows > 0 THEN 1 ELSE 0 END), 0) AS nonempty "
            "FROM matches"
        ).fetchone()
        verified_rows = connection.execute(
            "SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0"
        ).fetchall()
        table_names = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        rig_counts = None
        if {"cameras", "rigs", "rig_sensors", "frames", "frame_data"} <= table_names:
            rig_counts = {
                "cameras": int(connection.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]),
                "rigs": int(connection.execute("SELECT COUNT(*) FROM rigs").fetchone()[0]),
                "rig_sensors": int(
                    connection.execute("SELECT COUNT(*) FROM rig_sensors").fetchone()[0]
                ),
                "frames": int(connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0]),
                "frame_data": int(
                    connection.execute("SELECT COUNT(*) FROM frame_data").fetchone()[0]
                ),
            }
            frame_size = connection.execute(
                "SELECT MIN(size), MAX(size) FROM "
                "(SELECT COUNT(*) AS size FROM frame_data GROUP BY frame_id)"
            ).fetchone()
            rig_counts["min_images_per_frame"] = int(frame_size[0]) if frame_size[0] else 0
            rig_counts["max_images_per_frame"] = int(frame_size[1]) if frame_size[1] else 0
    except sqlite3.Error as exc:
        raise ValidationError(f"cannot inspect COLMAP database {database}: {exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()

    image_ids = {int(row["image_id"]) for row in images}
    parent = {image_id: image_id for image_id in image_ids}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    max_image_id = 2_147_483_647
    verified_inliers = 0
    for row in verified_rows:
        pair_id = int(row["pair_id"])
        image_id2 = pair_id % max_image_id
        image_id1 = (pair_id - image_id2) // max_image_id
        if image_id1 not in image_ids or image_id2 not in image_ids:
            raise ValidationError(f"COLMAP database contains invalid pair id {pair_id}")
        union(image_id1, image_id2)
        verified_inliers += int(row["rows"])

    component_sizes: dict[int, int] = {}
    for image_id in image_ids:
        root = find(image_id)
        component_sizes[root] = component_sizes.get(root, 0) + 1
    return {
        "images": len(images),
        "keypoint_records": int(keypoints["records"]),
        "keypoints": int(keypoints["features"]),
        "max_keypoints_per_image": int(keypoints["max_per_image"]),
        "descriptor_records": int(descriptors["records"]),
        "descriptors": int(descriptors["features"]),
        "candidate_pair_records": int(raw["records"]),
        "candidate_pairs_with_raw_matches": int(raw["nonempty"]),
        "raw_correspondences": int(raw["correspondences"]),
        "verified_pairs": len(verified_rows),
        "verified_inliers": verified_inliers,
        "verified_component_sizes": sorted(component_sizes.values(), reverse=True),
        "rig_configuration": rig_counts,
    }


def run_plan(plan_path: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _validate_plan(plan, plan_path)
    root = plan_path.parent
    (root / "models").mkdir(exist_ok=True)
    image = plan["docker"]["id"]
    phases: dict[str, Any] = {}
    for feature in plan["commands"]["feature_extractor"]:
        label = feature.get("phase", f"feature_extractor_cam{feature['camera_number']}")
        phases[label] = run_docker_phase(
            image=image, output_root=root, phase=label, command=feature["command"]
        )
    for label in ("rig_configurator", "matches_importer", "mapper"):
        if label not in plan["commands"]:
            continue
        phases[label] = run_docker_phase(
            image=image, output_root=root, phase=label, command=plan["commands"][label]
        )
    models: list[dict[str, Any]] = []
    registered_union: set[str] = set()
    rig_aware = bool(plan["settings"].get("fixed_calibrated_stereo_rig"))
    for model_dir in sorted(path for path in (root / "models").iterdir() if path.is_dir()):
        text_dir = root / "models-text" / model_dir.name
        text_dir.mkdir(parents=True, exist_ok=True)
        run_docker_phase(
            image=image,
            output_root=root,
            phase=f"convert_model_{model_dir.name}",
            command=[
                "colmap",
                "model_converter",
                "--input_path",
                f"/output/models/{model_dir.name}",
                "--output_path",
                f"/output/models-text/{model_dir.name}",
                "--output_type",
                "TXT",
            ],
        )
        names = _registered_names(text_dir / "images.txt")
        frame_keys = _registered_frame_keys(names, rig_aware=rig_aware)
        points, observations, mean_error = _point_stats(text_dir / "points3D.txt")
        registered_union.update(names)
        models.append(
            {
                "model": model_dir.name,
                "registered_images": len(names),
                "registered_frames": len(frame_keys) if rig_aware else None,
                "points": points,
                "observations": observations,
                "mean_reprojection_px_observation_weighted": mean_error,
                "cameras_txt_sha256": sha256_file(text_dir / "cameras.txt"),
                "images_txt_sha256": sha256_file(text_dir / "images.txt"),
                "points3D_txt_sha256": sha256_file(text_dir / "points3D.txt"),
            }
        )
    measured = [value for key, value in phases.items() if not key.startswith("convert_")]
    registered_frames = _registered_frame_keys(registered_union, rig_aware=rig_aware)
    expected_images = len(plan["candidate"]["image_names"])
    expected_frames = plan["inputs"]["staging"].get("frames") if rig_aware else None
    result = {
        "schema": RESULT_SCHEMA,
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "phases": phases,
        "summary": {
            "models": len(models),
            "unique_registered_images": len(registered_union),
            "registered_image_fraction": len(registered_union) / expected_images,
            "unique_registered_frames": len(registered_frames) if rig_aware else None,
            "registered_frame_fraction": (
                len(registered_frames) / expected_frames if expected_frames else None
            ),
            "phase_wall_seconds": sum(value["wall_seconds"] for value in measured),
            "peak_process_hwm_kib": max((value["peak_process_hwm_kib"] for value in measured), default=0),
            "peak_container_cgroup_kib": max((value["cgroup_peak_kib"] for value in measured), default=0),
        },
        "models": models,
        "database": _database_stats(root / "database.db"),
        "ground_truth_used_for_selection_or_mapping": False,
    }
    atomic_json(root / "result.json", result)
    return result


def summarize_existing_plan(plan_path: Path) -> dict[str, Any]:
    """Rebuild result.json from completed phase artifacts without rerunning SfM."""
    plan_path = plan_path.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _validate_plan(plan, plan_path)
    root = plan_path.parent
    phase_commands = {
        feature.get("phase", f"feature_extractor_cam{feature['camera_number']}"): feature["command"]
        for feature in plan["commands"]["feature_extractor"]
    }
    phase_commands.update(
        {
            label: plan["commands"][label]
            for label in ("rig_configurator", "matches_importer", "mapper")
            if label in plan["commands"]
        }
    )
    phases: dict[str, Any] = {}
    for label, command in phase_commands.items():
        metrics = parse_metrics(root / "timing" / f"{label}.tsv")
        if metrics["status"] != 0:
            raise ValidationError(f"cannot summarize unsuccessful phase {label}")
        metrics["colmap_command"] = command
        metrics["log_diagnostics"] = _log_diagnostics(root / "logs" / f"{label}.log")
        phases[label] = metrics

    models: list[dict[str, Any]] = []
    registered_union: set[str] = set()
    rig_aware = bool(plan["settings"].get("fixed_calibrated_stereo_rig"))
    for model_dir in sorted(path for path in (root / "models").iterdir() if path.is_dir()):
        text_dir = root / "models-text" / model_dir.name
        required = [text_dir / name for name in ("cameras.txt", "images.txt", "points3D.txt")]
        if not all(path.is_file() for path in required):
            text_dir.mkdir(parents=True, exist_ok=True)
            run_docker_phase(
                image=plan["docker"]["id"],
                output_root=root,
                phase=f"convert_model_{model_dir.name}",
                command=[
                    "colmap",
                    "model_converter",
                    "--input_path",
                    f"/output/models/{model_dir.name}",
                    "--output_path",
                    f"/output/models-text/{model_dir.name}",
                    "--output_type",
                    "TXT",
                ],
            )
        names = _registered_names(text_dir / "images.txt")
        frame_keys = _registered_frame_keys(names, rig_aware=rig_aware)
        points, observations, mean_error = _point_stats(text_dir / "points3D.txt")
        registered_union.update(names)
        models.append(
            {
                "model": model_dir.name,
                "registered_images": len(names),
                "registered_frames": len(frame_keys) if rig_aware else None,
                "points": points,
                "observations": observations,
                "mean_reprojection_px_observation_weighted": mean_error,
                "cameras_txt_sha256": sha256_file(text_dir / "cameras.txt"),
                "images_txt_sha256": sha256_file(text_dir / "images.txt"),
                "points3D_txt_sha256": sha256_file(text_dir / "points3D.txt"),
            }
        )
    registered_frames = _registered_frame_keys(registered_union, rig_aware=rig_aware)
    expected_images = len(plan["candidate"]["image_names"])
    expected_frames = plan["inputs"]["staging"].get("frames") if rig_aware else None
    result = {
        "schema": RESULT_SCHEMA,
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "phases": phases,
        "summary": {
            "models": len(models),
            "unique_registered_images": len(registered_union),
            "registered_image_fraction": len(registered_union) / expected_images,
            "unique_registered_frames": len(registered_frames) if rig_aware else None,
            "registered_frame_fraction": (
                len(registered_frames) / expected_frames if expected_frames else None
            ),
            "phase_wall_seconds": sum(value["wall_seconds"] for value in phases.values()),
            "peak_process_hwm_kib": max(
                (value["peak_process_hwm_kib"] for value in phases.values()), default=0
            ),
            "peak_container_cgroup_kib": max(
                (value["cgroup_peak_kib"] for value in phases.values()), default=0
            ),
        },
        "models": models,
        "database": _database_stats(root / "database.db"),
        "ground_truth_used_for_selection_or_mapping": False,
    }
    atomic_json(root / "result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--summarize-existing", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--candidate-index", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--tier-manifest", type=Path)
    parser.add_argument("--rig-transform-matrix", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--docker-image", default="colmap/colmap:latest")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-num-features", type=int, default=256)
    parser.add_argument("--max-ratio", type=float, default=0.8)
    parser.add_argument("--min-num-inliers", type=int, default=8)
    parser.add_argument("--max-error", type=float, default=4.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.run or args.summarize_existing:
            if args.plan is None:
                raise ValidationError("--run/--summarize-existing requires --plan")
            result = (
                run_plan(args.plan)
                if args.run
                else summarize_existing_plan(args.plan)
            )
        else:
            required = {
                "--candidate-index": args.candidate_index,
                "--image-root": args.image_root,
                "--calibration-dir": args.calibration_dir,
                "--output-root": args.output_root,
            }
            missing = [key for key, value in required.items() if value is None]
            if missing:
                raise ValidationError(f"--prepare requires {', '.join(missing)}")
            result = prepare_plan(
                candidate_index=args.candidate_index,
                image_root=args.image_root,
                calibration_dir=args.calibration_dir,
                output_root=args.output_root,
                docker_image=args.docker_image,
                tier_manifest=args.tier_manifest,
                rig_transform_matrix=args.rig_transform_matrix,
                threads=args.threads,
                max_num_features=args.max_num_features,
                max_ratio=args.max_ratio,
                min_num_inliers=args.min_num_inliers,
                max_error=args.max_error,
            )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
