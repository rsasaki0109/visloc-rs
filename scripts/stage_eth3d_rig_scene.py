#!/usr/bin/env python3
"""Stage one official ETH3D low-res rig scene without leaking reference poses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys


SCHEMA = "visloc_eth3d_rig_staging_v1"


class StageError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def parse_images(path: Path) -> list[tuple[int, str]]:
    records: list[tuple[int, str]] = []
    expect_image = True
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if stripped.startswith("#"):
            continue
        if expect_image:
            if not stripped:
                continue
            fields = stripped.split()
            if len(fields) < 10:
                raise StageError(f"malformed COLMAP image record: {stripped[:120]}")
            try:
                camera_id = int(fields[8])
            except ValueError as error:
                raise StageError("invalid CAMERA_ID in images.txt") from error
            records.append((camera_id, fields[9]))
            expect_image = False
        else:
            expect_image = True
    if not expect_image:
        raise StageError("images.txt ends before the final POINTS2D line")
    if not records:
        raise StageError("images.txt contains no images")
    return records


def flattened_name(source_name: str) -> str:
    value = PurePosixPath(source_name)
    if len(value.parts) != 2 or not value.name.lower().endswith(".png"):
        raise StageError(f"unexpected ETH3D rig image path: {source_name}")
    directory = value.parts[0]
    prefix = "images_rig_"
    suffix = "_undistorted"
    if not directory.startswith(prefix) or not directory.endswith(suffix):
        raise StageError(f"unexpected ETH3D rig camera directory: {directory}")
    camera = directory[len(prefix) : -len(suffix)]
    if not camera.startswith("cam"):
        raise StageError(f"unexpected ETH3D rig camera label: {camera}")
    return f"{camera}_{value.name}"


def stage(scene_dir: Path, output_root: Path) -> dict:
    scene_dir = scene_dir.resolve()
    output_root = output_root.resolve()
    official_calibration = scene_dir / "rig_calibration_undistorted"
    cameras_source = official_calibration / "cameras.txt"
    images_source = official_calibration / "images.txt"
    if not cameras_source.is_file() or not images_source.is_file():
        raise StageError("official rig calibration is missing")

    records = []
    seen: set[str] = set()
    for camera_id, source_name in parse_images(images_source):
        flat = flattened_name(source_name)
        if flat in seen:
            raise StageError(f"duplicate flattened image name: {flat}")
        seen.add(flat)
        source = scene_dir / "images" / PurePosixPath(source_name)
        if not source.is_file():
            raise StageError(f"source image is missing: {source_name}")
        records.append((flat, camera_id, source_name, source))
    records.sort(key=lambda item: item[0])

    images_dir = output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_records = []
    calibration_lines = [
        "# Generated from official ETH3D rig calibration.\n",
        "# Intrinsics and CAMERA_ID assignments only; reference poses/points omitted.\n",
        "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n",
        "# POINTS2D[]\n",
    ]
    for image_id, (flat, camera_id, source_name, source) in enumerate(records, start=1):
        destination = images_dir / flat
        if destination.is_symlink():
            if destination.resolve() != source:
                raise StageError(f"stale image link has a different target: {flat}")
        elif destination.exists():
            raise StageError(f"non-symlink staging image already exists: {flat}")
        else:
            temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
            os.symlink(source, temporary)
            os.replace(temporary, destination)
        image_sha = sha256_file(source)
        manifest_records.append(
            {
                "camera_id": camera_id,
                "flat_name": flat,
                "source_name": source_name,
                "source_sha256": image_sha,
            }
        )
        calibration_lines.append(f"{image_id} 1 0 0 0 0 0 0 {camera_id} {flat}\n\n")

    calibration_dir = output_root / "calibration"
    cameras_bytes = cameras_source.read_bytes()
    atomic_write(calibration_dir / "cameras.txt", cameras_bytes)
    atomic_write(calibration_dir / "images.txt", "".join(calibration_lines).encode())
    atomic_write(calibration_dir / "points3D.txt", b"# Reference points intentionally omitted.\n")
    index = {
        "cameras_sha256": hashlib.sha256(cameras_bytes).hexdigest(),
        "ground_truth_used_for_selection_or_mapping": False,
        "image_count": len(records),
        "images": manifest_records,
        "scene": scene_dir.name,
        "schema": SCHEMA,
    }
    index_bytes = canonical_json(index)
    atomic_write(output_root / "index.json", index_bytes)
    return {
        "image_count": len(records),
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "scene": scene_dir.name,
        "schema": SCHEMA,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        summary = stage(args.scene_dir, args.output_root)
    except (OSError, StageError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
