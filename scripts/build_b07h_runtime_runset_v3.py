#!/usr/bin/env python3
"""Materialize the E:-resident B07-H ambient-recorded runset v3.

The v2 runset is consumed as an immutable contract/input manifest.  This
builder rebases its candidate-owned paths, removes the strict v6 ambient
oracle, and adds a finite ambient telemetry policy.  It never runs a mapper,
opens a dataset member, or materializes ground truth.
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.dont_write_bytecode = True
import run_b07h_runtime_driver_v7 as storage  # noqa: E402


def _rebase_string(value: str, old_root: Path, new_root: Path) -> str:
    old_text = str(old_root).replace("/", "\\").rstrip("\\")
    new_text = str(new_root).replace("/", "\\").rstrip("\\")
    pattern = re.compile(re.escape(old_text) + r"(?=\\|$)", re.IGNORECASE)
    return pattern.sub(lambda _match: new_text, value.replace("/", "\\"))


def rebase_candidate_paths(value: Any, old_root: Path, new_root: Path) -> Any:
    """Recursively rebase absolute candidate paths without changing labels."""

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
    if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
        raise storage.DriverError(f"{label} metadata is malformed")
    raw_path = item["path"]
    storage._reject_path_syntax(raw_path, label)
    raw = Path(raw_path)
    if raw.is_absolute():
        _reject_reparse_chain(raw, label)
    try:
        path = storage.require_regular_candidate_file(raw_path, root, label)
    except storage.DriverError:
        path = storage._validate_runset_tool_path(raw_path, root, label, allow_c_readonly=item.get("read_only") is True)
    if storage._is_reparse(path) or not path.is_file():
        raise storage.DriverError(f"{label} is not a regular non-reparse file")
    return path


def _enrich_fixed_tools(value: Any, root: Path) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != set(storage.FIXED_TOOL_KEYS):
        raise storage.DriverError("v3 runset fixed_tools must contain the exact tool inventory")
    enriched: dict[str, dict[str, Any]] = {}
    for name in storage.FIXED_TOOL_KEYS:
        item = value.get(name)
        path = _fixed_tool_path(item, root, f"v3 fixed tool {name}")
        actual_sha = storage.digest(path)
        actual_bytes = path.stat().st_size
        if isinstance(item, Mapping) and "sha256" in item:
            declared = item["sha256"]
            if not isinstance(declared, str) or declared.upper() != actual_sha.upper():
                raise storage.DriverError(f"v3 fixed tool {name} SHA mismatch")
        if isinstance(item, Mapping) and "bytes" in item:
            declared_bytes = item["bytes"]
            if type(declared_bytes) is not int or declared_bytes != actual_bytes:
                raise storage.DriverError(f"v3 fixed tool {name} byte-size mismatch")
        normalized = dict(item)
        normalized["sha256"] = actual_sha
        normalized["bytes"] = actual_bytes
        enriched[name] = normalized
    return enriched


def build_runset_v3(candidate_root: Path, frozen_v2: Path, output: Path) -> Path:
    root = storage.require_e_root(candidate_root, "v3 runset candidate root")
    frozen_path = storage.require_regular_allowed_e_file(frozen_v2, "frozen v2 runset")
    old = storage.read_json(frozen_path)
    if old.get("schema") != storage.RUNSET_V2_SCHEMA or old.get("status") != "fixed_preflight_only":
        raise storage.DriverError("input is not the fixed v2 runset")
    storage.reject_gt(old, "frozen v2 runset")
    old_root = storage.require_e_root(old.get("candidate_root"), "frozen v2 declared candidate root")
    try:
        frozen_path.relative_to(old_root)
    except ValueError as error:
        raise storage.DriverError("frozen v2 runset is not resident under its declared candidate root") from error
    _reject_reparse_chain(frozen_path, "frozen v2 runset")
    frozen_sha = storage.validate_sidecar(frozen_path, old_root, "frozen v2 runset")
    if not re.fullmatch(r"[0-9A-Fa-f]{64}", frozen_sha):
        raise storage.DriverError("frozen v2 runset digest is malformed")
    value: dict[str, Any] = rebase_candidate_paths(copy.deepcopy(old), old_root, root)
    value.pop("ambient_oracle", None)
    value["schema"] = storage.RUNSET_SCHEMA
    value["supersedes_schema"] = storage.RUNSET_V2_SCHEMA
    value["supersedes_sha256"] = frozen_sha
    value["ambient_policy"] = storage.AMBIENT_POLICY
    value["ambient_recording"] = {
        "finite_window": True,
        "default_samples": storage.DEFAULT_AMBIENT_SAMPLES,
        "default_sample_seconds": storage.DEFAULT_AMBIENT_SAMPLE_SECONDS,
        "noise_is_informational": True,
        "noise_fields": ["cpu", "search_indexer", "gpu"],
        "logs_wsl": True,
        "hard_blockers": ["target_processes", "c_workspace", "e_free_threshold"],
        "start_gate": "hard_blockers_only",
        "history_namespace": "B07H_v4_ambient_recorded",
    }
    value["storage_policy"] = {
        "allowed_roots": ["E:/visloc_archive", "E:/datasets"],
        "candidate_root_must_match_declared": True,
        "c_workspace_monitor_root": str(storage.DEFAULT_C_WORKSPACE),
        "c_target_temp_cache_rejected": True,
        "atomic_result_and_ledger_sidecars": True,
        "ambient_policy": storage.AMBIENT_POLICY,
        "strict_v6_namespace_disjoint": True,
    }
    value["candidate_root"] = str(root)
    value["ground_truth_read"] = False
    value["ground_truth_materialized"] = False
    value["ground_truth_argument_present_anywhere"] = False
    if old_root != root and _contains_text(value, str(old_root)):
        raise storage.DriverError("v3 runset retains an old candidate-root string")
    value["fixed_tools"] = _enrich_fixed_tools(value.get("fixed_tools"), root)
    storage.validate_runset_value(value, root)
    output_path = storage.candidate_path(output, root, "v3 runset output")
    storage.atomic_json(output_path, value, root, replace=False)
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--frozen-v2-runset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runsets") / "B07H_GT_FREE_RUNTIME_RUNSET_V3.json")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(build_runset_v3(args.candidate_root, args.frozen_v2_runset, args.output))
