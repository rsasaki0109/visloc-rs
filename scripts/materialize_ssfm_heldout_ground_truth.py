#!/usr/bin/env python3
"""Materialize one held-out EuRoC GT file only after timed engines exit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--hierarchical-manifest", type=Path, required=True)
    parser.add_argument("--colmap-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def engine_exit_evidence(hierarchical_path: Path, colmap_path: Path) -> dict:
    hierarchical = read_json(hierarchical_path)
    colmap = read_json(colmap_path)
    if hierarchical.get("protocol", {}).get("ground_truth_read") is not False:
        raise ValueError("hierarchical manifest does not prove GT isolation")
    if colmap.get("ground_truth_read") is not False:
        raise ValueError("COLMAP manifest does not prove GT isolation")
    return {
        "hierarchical": {
            "path": str(hierarchical_path.resolve()),
            "sha256": sha256(hierarchical_path),
        },
        "colmap": {
            "path": str(colmap_path.resolve()),
            "sha256": sha256(colmap_path),
        },
    }


def find_ground_truth_member(archive: zipfile.ZipFile, sequence: str) -> zipfile.ZipInfo:
    matches = []
    expected_suffix = (sequence, "mav0", "state_groundtruth_estimate0", "data.csv")
    for info in archive.infolist():
        pure = PurePosixPath(info.filename)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"unsafe archive member: {info.filename}")
        if tuple(pure.parts[-len(expected_suffix) :]) == expected_suffix:
            matches.append(info)
    if len(matches) != 1:
        raise ValueError(
            f"expected one GT data.csv for {sequence}, found {len(matches)}"
        )
    if matches[0].is_dir():
        raise ValueError("ground-truth archive member is a directory")
    return matches[0]


def main() -> int:
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    if args.sequence not in protocol["selection"]["held_out_sequences"]:
        raise ValueError(f"not a frozen held-out sequence: {args.sequence}")

    engines = engine_exit_evidence(
        args.hierarchical_manifest,
        args.colmap_manifest,
    )
    download_manifest_path = args.download_dir / "download_manifest.json"
    download = read_json(download_manifest_path)
    if download.get("status") != "success":
        raise ValueError("download manifest is not successful")
    if download.get("protocol_sha256") != protocol_sha256:
        raise ValueError("download/protocol hash mismatch")

    archive_specs = [
        spec
        for spec in protocol["inputs"]["official_archives"]
        if args.sequence in spec["selected_sequences"]
    ]
    if len(archive_specs) != 1:
        raise ValueError("sequence must belong to exactly one official archive")
    spec = archive_specs[0]
    entries = [entry for entry in download["archives"] if entry["name"] == spec["name"]]
    if len(entries) != 1:
        raise ValueError("download manifest archive entry mismatch")
    entry = entries[0]
    if entry["size_bytes"] != spec["size_bytes"]:
        raise ValueError("downloaded archive size evidence mismatch")
    if entry["checksum"].lower() != spec["checksum"].lower():
        raise ValueError("downloaded archive checksum evidence mismatch")
    archive_path = Path(entry["path"])
    if archive_path.stat().st_size != spec["size_bytes"]:
        raise ValueError("official archive size changed after verification")

    args.out_dir.mkdir(parents=True)
    ground_truth_path = args.out_dir / "data.csv"
    with zipfile.ZipFile(archive_path) as archive:
        member = find_ground_truth_member(archive, args.sequence)
        with archive.open(member) as source, ground_truth_path.open("xb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)

    output = {
        "schema_version": 1,
        "status": "success",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "sequence": args.sequence,
        "materialized_utc": timestamp(),
        "ground_truth_first_read_after_all_timed_engines_exited": True,
        "engine_exit_evidence": engines,
        "download_manifest": {
            "path": str(download_manifest_path.resolve()),
            "sha256": sha256(download_manifest_path),
        },
        "archive": {
            "path": str(archive_path.resolve()),
            "name": spec["name"],
            "size_bytes": spec["size_bytes"],
            "checksum_algorithm": spec["checksum_algorithm"],
            "checksum": spec["checksum"],
            "member": member.filename,
            "member_crc32": f"{member.CRC:08x}",
            "member_size_bytes": member.file_size,
        },
        "ground_truth": {
            "path": str(ground_truth_path.resolve()),
            "sha256": sha256(ground_truth_path),
        },
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
