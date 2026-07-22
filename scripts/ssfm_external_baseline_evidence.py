#!/usr/bin/env python3
"""Shared schema validation for pre-GT external SSfM baseline evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path


EXTERNAL_ENGINES = ("gluemap", "instantsfm")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_external_baseline_manifest(
    manifest: dict,
    *,
    sequence: str,
    heldout_protocol_sha256: str,
    external_protocol_sha256: str,
    manifest_dir: Path,
) -> dict[str, dict]:
    if manifest.get("sequence") != sequence:
        raise ValueError("external baseline sequence mismatch")
    if manifest.get("heldout_protocol_sha256") != heldout_protocol_sha256:
        raise ValueError("external baseline held-out protocol mismatch")
    if manifest.get("external_protocol_sha256") != external_protocol_sha256:
        raise ValueError("external baseline companion protocol mismatch")
    if manifest.get("ground_truth_read") is not False:
        raise ValueError("external baseline manifest does not prove GT isolation")
    if not manifest.get("all_engine_processes_exited"):
        raise ValueError("external baseline processes have not all exited")
    results = manifest.get("results", {})
    if set(results) != set(EXTERNAL_ENGINES):
        raise ValueError("missing or unexpected external baseline cells")

    for engine, cell in results.items():
        status = cell.get("status")
        if status not in ("success", "dnf"):
            raise ValueError(f"invalid {engine} status: {status}")
        if not cell.get("source_revision"):
            raise ValueError(f"{engine} cell lacks source revision/outage identity")
        if not cell.get("attempt", {}).get("command"):
            raise ValueError(f"{engine} cell lacks attempted command")
        if "returncode" not in cell["attempt"]:
            raise ValueError(f"{engine} cell lacks attempted return code")
        if status == "dnf":
            if not cell.get("reason"):
                raise ValueError(f"{engine} DNF lacks a reason")
            if cell.get("trajectory") is not None:
                raise ValueError(f"{engine} DNF unexpectedly publishes a trajectory")
            continue

        trajectory = cell.get("trajectory")
        if not isinstance(trajectory, dict):
            raise ValueError(f"{engine} success lacks trajectory evidence")
        trajectory_path = Path(trajectory["path"])
        if not trajectory_path.is_absolute():
            trajectory_path = manifest_dir / trajectory_path
        if not trajectory_path.is_file():
            raise FileNotFoundError(trajectory_path)
        if sha256(trajectory_path) != trajectory.get("sha256"):
            raise ValueError(f"{engine} trajectory hash mismatch")
        for key in (
            "registered_images",
            "registration_rate",
            "total_wall_seconds",
            "peak_process_tree_rss_bytes",
        ):
            if cell.get(key) is None:
                raise ValueError(f"{engine} success lacks {key}")
    return results
