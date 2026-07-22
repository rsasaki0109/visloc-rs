#!/usr/bin/env python3
"""Safely extract only the held-out EuRoC sequences from verified archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from download_ssfm_heldout_euroc import pid_exists


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int)
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_relative_path(member: str, selected: set[str]) -> Path | None:
    pure = PurePosixPath(member)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe archive member: {member}")
    matches = [index for index, part in enumerate(pure.parts) if part in selected]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"ambiguous selected sequence path: {member}")
    relative = Path(*pure.parts[matches[0] :])
    if not relative.parts:
        return None
    return relative


def extract_archive(archive: Path, sequences: list[str], out_dir: Path) -> dict:
    selected = set(sequences)
    member_count = 0
    file_count = 0
    extracted_bytes = 0
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            relative = selected_relative_path(info.filename, selected)
            if relative is None:
                continue
            member_count += 1
            destination = (out_dir / relative).resolve()
            if out_dir.resolve() not in destination.parents and destination != out_dir.resolve():
                raise ValueError(f"archive member escapes output: {info.filename}")
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if unix_mode and (unix_mode & 0o170000) == 0o120000:
                raise ValueError(f"archive symlink is not allowed: {info.filename}")
            if destination.exists():
                raise FileExistsError(f"duplicate or existing output: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as input_stream, destination.open("xb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream, length=8 * 1024 * 1024)
            file_count += 1
            extracted_bytes += info.file_size
    if member_count == 0:
        raise ValueError(f"no selected sequences found in {archive}")
    return {
        "archive": str(archive.resolve()),
        "selected_sequences": sequences,
        "selected_members": member_count,
        "extracted_files": file_count,
        "extracted_bytes": extracted_bytes,
    }


def validate_sequence(root: Path, sequence: str) -> dict:
    sequence_root = root / sequence
    required = [
        sequence_root / "mav0" / "cam0" / "sensor.yaml",
        sequence_root / "mav0" / "cam1" / "sensor.yaml",
        sequence_root / "mav0" / "cam0" / "data.csv",
        sequence_root / "mav0" / "cam1" / "data.csv",
        sequence_root / "mav0" / "state_groundtruth_estimate0" / "data.csv",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    cam0 = len(list((sequence_root / "mav0" / "cam0" / "data").glob("*.png")))
    cam1 = len(list((sequence_root / "mav0" / "cam1" / "data").glob("*.png")))
    if cam0 == 0 or cam1 == 0:
        raise ValueError(f"empty camera stream in {sequence}")
    return {
        "path": str(sequence_root.resolve()),
        "cam0_images": cam0,
        "cam1_images": cam1,
        "cam0_sensor_sha256": sha256(required[0]),
        "cam1_sensor_sha256": sha256(required[1]),
        "cam0_index_sha256": sha256(required[2]),
        "cam1_index_sha256": sha256(required[3]),
        "ground_truth_sha256": sha256(required[4]),
    }


def main() -> int:
    args = parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    if args.wait_pid is not None:
        print(f"waiting for PID {args.wait_pid} before extraction", flush=True)
        while pid_exists(args.wait_pid):
            time.sleep(15.0)

    download_manifest_path = args.download_dir / "download_manifest.json"
    download_manifest = json.loads(download_manifest_path.read_text(encoding="utf-8"))
    if download_manifest["status"] != "success":
        raise RuntimeError("download manifest is not successful")
    if download_manifest["protocol_sha256"] != protocol_sha256:
        raise ValueError("download/protocol hash mismatch")
    downloaded = {entry["name"]: entry for entry in download_manifest["archives"]}

    args.out_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "status": "in_progress",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "started_utc": timestamp(),
        "download_manifest": {
            "path": str(download_manifest_path.resolve()),
            "sha256": sha256(download_manifest_path),
        },
        "archives": [],
        "sequences": {},
    }
    manifest_path = args.out_dir / "extraction_manifest.json"
    try:
        for archive_spec in protocol["inputs"]["official_archives"]:
            entry = downloaded[archive_spec["name"]]
            archive = Path(entry["path"])
            if archive.stat().st_size != archive_spec["size_bytes"]:
                raise ValueError(f"archive size changed: {archive}")
            manifest["archives"].append(
                extract_archive(
                    archive,
                    archive_spec["selected_sequences"],
                    args.out_dir,
                )
            )
        for sequence in protocol["selection"]["held_out_sequences"]:
            manifest["sequences"][sequence] = validate_sequence(args.out_dir, sequence)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["finished_utc"] = timestamp()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        raise
    manifest["status"] = "success"
    manifest["finished_utc"] = timestamp()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
