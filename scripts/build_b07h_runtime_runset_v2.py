#!/usr/bin/env python3
"""Materialize the E:-resident B07-H runtime runset v2.

The frozen v1 runset is read only as a protocol/input manifest.  This command
does not run an engine or inspect any dataset/ground-truth member.  It writes
only the explicitly selected output beneath the declared E: candidate root.
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import Any, Sequence

sys.dont_write_bytecode = True

import run_b07h_runtime_driver_v6 as storage  # noqa: E402


V1_RUNSET_SHA256 = storage.FROZEN_V1_RUNSET_SHA256


def _rebase_string(value: str, old_root: Path, new_root: Path) -> str:
    old_text = str(old_root).replace("/", "\\").rstrip("\\")
    new_text = str(new_root).replace("/", "\\").rstrip("\\")
    pattern = re.compile(re.escape(old_text) + r"(?=\\|$)", re.IGNORECASE)
    return pattern.sub(lambda _match: new_text, value.replace("/", "\\"))


def rebase_candidate_paths(value: Any, old_root: Path, new_root: Path) -> Any:
    """Recursively rebase absolute old-candidate paths without touching IDs."""

    if isinstance(value, dict):
        return {key: rebase_candidate_paths(item, old_root, new_root) for key, item in value.items()}
    if isinstance(value, list):
        return [rebase_candidate_paths(item, old_root, new_root) for item in value]
    if isinstance(value, str):
        return _rebase_string(value, old_root, new_root)
    return value


def _contains_text(value: Any, needle: str) -> bool:
    if isinstance(value, dict):
        return any(_contains_text(key, needle) or _contains_text(item, needle) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_text(item, needle) for item in value)
    return isinstance(value, str) and needle.lower() in value.lower()


def _reject_reparse_chain(path: Path, label: str) -> None:
    current = path
    while True:
        if storage._is_reparse(current):
            raise storage.DriverError(f"{label} contains a symlink/reparse component: {current}")
        if current.parent == current:
            return
        current = current.parent


def _fixed_tool_path(item: Any, root: Path, label: str) -> Path:
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        raise storage.DriverError(f"{label} metadata is malformed")
    raw_path = item["path"]
    storage._reject_path_syntax(raw_path, label)
    raw = Path(raw_path)
    if raw.is_absolute():
        _reject_reparse_chain(raw, label)
    try:
        path = storage.require_regular_candidate_file(raw_path, root, label)
    except storage.DriverError:
        path = storage._validate_runset_tool_path(
            raw_path,
            root,
            label,
            allow_c_readonly=item.get("read_only") is True,
        )
    if storage._is_reparse(path) or not path.is_file():
        raise storage.DriverError(f"{label} is not a regular non-reparse file")
    return path


def _enrich_fixed_tools(value: Any, root: Path) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != set(storage.FIXED_TOOL_KEYS):
        raise storage.DriverError("v2 runset fixed_tools must contain the exact tool inventory")
    enriched: dict[str, dict[str, Any]] = {}
    for name in storage.FIXED_TOOL_KEYS:
        item = value.get(name)
        path = _fixed_tool_path(item, root, f"v2 fixed tool {name}")
        actual_sha = storage.digest(path)
        actual_bytes = path.stat().st_size
        if isinstance(item, dict) and "sha256" in item:
            declared_sha = item["sha256"]
            if not isinstance(declared_sha, str) or declared_sha.upper() != actual_sha.upper():
                raise storage.DriverError(f"v2 fixed tool {name} SHA mismatch")
        if isinstance(item, dict) and "bytes" in item:
            declared_bytes = item["bytes"]
            if type(declared_bytes) is not int or declared_bytes != actual_bytes:
                raise storage.DriverError(f"v2 fixed tool {name} byte-size mismatch")
        normalized = dict(item)
        normalized["sha256"] = actual_sha
        normalized["bytes"] = actual_bytes
        enriched[name] = normalized
    return enriched


def build_runset_v2(candidate_root: Path, frozen_v1: Path, output: Path) -> Path:
    root = storage.require_e_root(candidate_root, "v2 runset candidate root")
    frozen_path = storage.require_regular_allowed_e_file(frozen_v1, "frozen v1 runset")
    frozen_sha = storage.digest(frozen_path)
    if frozen_sha.upper() != V1_RUNSET_SHA256:
        raise storage.DriverError("frozen v1 runset SHA does not match the pinned protocol")
    old = storage.read_json(frozen_path)
    if old.get("schema") != storage.RUNSET_V1_SCHEMA or old.get("status") != "fixed_preflight_only":
        raise storage.DriverError("input is not the fixed v1 runset")
    storage.reject_gt(old, "frozen v1 runset")
    old_root = storage.require_e_root(old.get("candidate_root"), "frozen v1 declared candidate root")
    try:
        frozen_path.relative_to(old_root)
    except ValueError as error:
        raise storage.DriverError("frozen v1 runset is not resident under its declared candidate root") from error
    value: dict[str, Any] = rebase_candidate_paths(copy.deepcopy(old), old_root, root)
    value["schema"] = storage.RUNSET_SCHEMA
    value["supersedes_schema"] = storage.RUNSET_V1_SCHEMA
    value["supersedes_sha256"] = V1_RUNSET_SHA256
    value["storage_policy"] = {
        "allowed_roots": ["E:/visloc_archive", "E:/datasets"],
        "candidate_root_must_match_declared": True,
        "c_workspace_monitor_root": str(storage.DEFAULT_C_WORKSPACE),
        "c_target_temp_cache_rejected": True,
        "atomic_result_and_ledger_sidecars": True,
    }
    value["candidate_root"] = str(root)
    value["ground_truth_read"] = False
    value["ground_truth_materialized"] = False
    value["ground_truth_argument_present_anywhere"] = False
    if old_root != root and _contains_text(value, str(old_root)):
        raise storage.DriverError("v2 runset retains an old candidate-root string")
    value["fixed_tools"] = _enrich_fixed_tools(value.get("fixed_tools"), root)
    oracle_path = storage.AMBIENT_ORACLE_RELATIVE_PATH
    oracle_file = storage.require_regular_candidate_file(oracle_path, root, "v2 ambient oracle")
    value["ambient_oracle"] = {
        "path": str(oracle_path),
        "sha256": storage.AMBIENT_ORACLE_SHA256,
        "sidecar": str(Path(str(oracle_path) + ".sha256")),
        "bytes": oracle_file.stat().st_size,
    }
    storage.validate_runset_value(value, root)
    output_path = storage.candidate_path(output, root, "v2 runset output")
    storage.atomic_json(output_path, value, root, replace=False)
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--frozen-v1-runset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runsets") / "B07H_GT_FREE_RUNTIME_RUNSET_V2.json")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(build_runset_v2(args.candidate_root, args.frozen_v1_runset, args.output))
