#!/usr/bin/env python3
"""GT-free B07-H runtime driver v7 (ambient telemetry is recorded, not gated).

This lane is intentionally independent from the v6 strict-quiet lane.  The
v7 runset, result artifacts, ambient history, manifest, and ledger all carry
``ambient_policy: recorded`` and use v7-only schema/path names.  CPU,
SearchIndexer, and GPU observations are useful context for interpreting a
run, but never prevent a start.  Only target-process conflicts, recursive C:
workspace contamination, and the E: free-space floor are start blockers.

The execution contract otherwise remains the frozen B07-H contract: exactly
six serial invocations, exactly nine result cells, fixed source/protocol/tool
hashes, GT-free commands, regular non-reparse E:-resident artifacts, and
failure-inclusive terminal accounting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

sys.dont_write_bytecode = True

# The v6 module is used only as a read-only library of already audited path,
# hash, command, and E:-environment primitives.  No v6 result/history/ledger
# path is read or written by this module, and no v6 ambient gate is called.
import run_b07h_runtime_driver_v6 as _strict  # noqa: E402


RUNSET_SCHEMA = "B07H_GT_FREE_RUNTIME_RUNSET_V3"
RUNSET_V2_SCHEMA = "B07H_GT_FREE_RUNTIME_RUNSET_V2"
EXPECTED_RUNSET_SCHEMA = RUNSET_SCHEMA
EXPECTED_RUNSET_V2_SCHEMA = RUNSET_V2_SCHEMA
RESULT_SCHEMA = "B07H_RUNTIME_DRIVER_RESULT_V4"
DEFERRED_SCHEMA = "B07H_RUNTIME_DRIVER_DEFERRED_V4"
LEDGER_SCHEMA = "B07H_RUNTIME_DRIVER_LEDGER_V4"
AMBIENT_HISTORY_SCHEMA = "B07H_RUNTIME_DRIVER_AMBIENT_RECORDED_V1"
AMBIENT_MANIFEST_SCHEMA = "B07H_RUNTIME_DRIVER_AMBIENT_RECORDED_MANIFEST_V1"
DRIVER_VERSION = "B07H_RUNTIME_DRIVER_V7"
AMBIENT_POLICY = "recorded"

EXPECTED_SOURCE_SHA256 = _strict.EXPECTED_SOURCE_SHA256
EXPECTED_PROTOCOL_SHA256 = _strict.EXPECTED_PROTOCOL_SHA256
STOP_FREE_BYTES = _strict.STOP_FREE_BYTES
TOTAL_INVOCATIONS = _strict.TOTAL_INVOCATIONS
TOTAL_RESULT_CELLS = _strict.TOTAL_RESULT_CELLS
SERIAL_ORDER = _strict.SERIAL_ORDER
RESULT_CELLS = _strict.RESULT_CELLS
INVOCATION_CELLS = _strict.INVOCATION_CELLS
ALLOWED_E_ROOTS = _strict.ALLOWED_E_ROOTS
DEFAULT_C_WORKSPACE = _strict.DEFAULT_C_WORKSPACE
E_ENV_SUFFIXES = _strict.E_ENV_SUFFIXES
ENV_KEYS_TO_SCRUB = _strict.ENV_KEYS_TO_SCRUB
ENV_SCRUB_MARKERS = _strict.ENV_SCRUB_MARKERS
FORBIDDEN_C_CACHE_NAMES = _strict.FORBIDDEN_C_CACHE_NAMES
FIXED_TOOL_KEYS = _strict.FIXED_TOOL_KEYS
OUTPUT_PATH_FLAGS = _strict.OUTPUT_PATH_FLAGS
C_READONLY_PATH_FLAGS = _strict.C_READONLY_PATH_FLAGS
CPU_SETTLE_LIMIT_PERCENT = _strict.CPU_SETTLE_LIMIT_PERCENT
SEARCH_INDEXER_SETTLE_LIMIT_PERCENT = _strict.SEARCH_INDEXER_SETTLE_LIMIT_PERCENT
GPU_MEMORY_GROWTH_TOLERANCE_MIB = _strict.GPU_MEMORY_GROWTH_TOLERANCE_MIB
DEFAULT_AMBIENT_SAMPLES = 5
DEFAULT_AMBIENT_SAMPLE_SECONDS = 2.0
LEDGER_RELATIVE_PATH = Path("logs/B07H_v4_ambient_recorded_ledger.json")
AMBIENT_LEDGER_RELATIVE_PATH = LEDGER_RELATIVE_PATH
DEFAULT_LEDGER_PATH = LEDGER_RELATIVE_PATH
TARGET_PROCESS_NAMES = frozenset({
    "cargo", "cargo.exe", "rustc", "rustc.exe", "colmap", "colmap.exe",
    "sequential_sfm_demo", "sequential_sfm_demo.exe", "robosim", "robosim.exe",
})
SEARCH_INDEXER_NAMES = frozenset({"searchindexer", "searchindexer.exe"})


class DriverError(ValueError, RuntimeError):
    """A deterministic v7 preflight, storage, or accounting failure."""


# Read-only aliases keep the v7 contract's public helpers familiar to the v6
# tests while making all v7 writes below explicit and independently named.
digest = _strict.digest
read_json = _strict.read_json
_resolved = _strict._resolved
_is_reparse = _strict._is_reparse
_same_path = _strict._same_path
_sha_claim = _strict._sha_claim
command_contains_gt = _strict.command_contains_gt
_reject_path_syntax = _strict._reject_path_syntax


def _reject_reparse_components(path: Path, root: Path | None, label: str) -> None:
    """Reject every existing reparse component using the lexical path.

    ``Path.resolve`` follows junctions/symlinks, so checking only a resolved
    path can hide the alias that escaped the candidate boundary.  Callers
    invoke this before and after resolution; the second check also catches a
    component replaced during the short validation window.
    """

    current = Path(os.path.normpath(str(path)))
    stop = Path(os.path.normpath(str(root))) if root is not None else None
    while True:
        if _is_reparse(current):
            raise DriverError(f"{label} contains a symlink/reparse component: {current}")
        if stop is not None and _same_path(current, stop):
            return
        if current.parent == current:
            if stop is not None:
                raise DriverError(f"{label} escaped candidate root")
            return
        current = current.parent


def _lexical_candidate(value: Path | str, root: Path, label: str) -> Path:
    raw = _reject_path_syntax(value, label)
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    lexical = Path(os.path.normpath(str(path)))
    if lexical.drive.upper() != "E:":
        raise DriverError(f"{label} must be on E:")
    try:
        lexical.relative_to(root)
    except ValueError as error:
        raise DriverError(f"{label} escapes declared candidate root: {lexical}") from error
    return lexical


def require_e_root(path: Path | str, label: str = "candidate root") -> Path:
    _reject_path_syntax(path, label)
    lexical = Path(os.path.normpath(str(path)))
    if not lexical.is_absolute():
        lexical = Path(os.path.abspath(str(lexical)))
    if lexical.drive.upper() != "E:":
        raise DriverError(f"{label} must be on E:, got {lexical}")
    _reject_reparse_components(lexical, None, label)
    resolved = _resolved(lexical)
    if resolved.drive.upper() != "E:":
        raise DriverError(f"{label} must be on E:, got {resolved}")
    allowed = False
    for base in ALLOWED_E_ROOTS:
        base_resolved = _resolved(base)
        try:
            resolved.relative_to(base_resolved)
        except ValueError:
            continue
        allowed = True
        break
    if not allowed:
        raise DriverError(f"{label} must be below E:\\visloc_archive or E:\\datasets: {resolved}")
    _reject_reparse_components(resolved, None, label)
    return resolved


def require_allowed_e_path(path: Path | str, label: str = "E path") -> Path:
    _reject_path_syntax(path, label)
    lexical = Path(os.path.normpath(str(path)))
    if not lexical.is_absolute():
        lexical = Path(os.path.abspath(str(lexical)))
    if lexical.drive.upper() != "E:":
        raise DriverError(f"{label} must be below E:\\visloc_archive or E:\\datasets")
    _reject_reparse_components(lexical, None, label)
    resolved = _resolved(lexical)
    if resolved.drive.upper() != "E:":
        raise DriverError(f"{label} must be below E:\\visloc_archive or E:\\datasets")
    if not any((lambda base_resolved: _same_path(resolved, base_resolved) or _is_relative_to(resolved, base_resolved))(_resolved(base)) for base in ALLOWED_E_ROOTS):
        raise DriverError(f"{label} must be below E:\\visloc_archive or E:\\datasets")
    _reject_reparse_components(resolved, None, label)
    return resolved


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def candidate_path(value: Path | str, root: Path, label: str) -> Path:
    root = require_e_root(root)
    lexical = _lexical_candidate(value, root, label)
    # Check the path before resolution so an alias cannot disappear from the
    # validation view.  Resolve only after the lexical boundary is clean.
    _reject_reparse_components(lexical, root, label)
    resolved = _resolved(lexical)
    if resolved.drive.upper() != "E:" or not _is_relative_to(resolved, root):
        raise DriverError(f"{label} escapes declared candidate root: {resolved}")
    _reject_reparse_components(resolved, root, label)
    return resolved


def require_regular_allowed_e_file(path: Path | str, label: str = "E file") -> Path:
    resolved = require_allowed_e_path(path, label)
    if _is_reparse(resolved) or not resolved.is_file():
        raise DriverError(f"{label} is not a regular E: file")
    return resolved


def require_regular_candidate_file(value: Path | str, root: Path, label: str) -> Path:
    path = candidate_path(value, root, label)
    if _is_reparse(path) or not path.is_file():
        raise DriverError(f"{label} is missing or not a regular file: {path}")
    return path


def require_c_readonly_file(value: Path | str, label: str) -> Path:
    _reject_path_syntax(value, label)
    lexical = Path(os.path.normpath(str(value)))
    if not lexical.is_absolute() or lexical.drive.upper() != "C:":
        raise DriverError(f"{label} is not an explicit C: read-only path")
    _reject_reparse_components(lexical, None, label)
    resolved = _resolved(lexical)
    workspace = _resolved(DEFAULT_C_WORKSPACE)
    if not _is_relative_to(resolved, workspace):
        raise DriverError(f"{label} is outside the fixed C workspace")
    _reject_reparse_components(resolved, workspace, label)
    if _is_reparse(resolved) or not resolved.is_file():
        raise DriverError(f"{label} is not a regular C read-only file")
    return resolved


# The command parser and environment constructor remain read-only helpers;
# all artifact paths supplied to them are checked again by the v7 candidate
# and atomic-write guards above/below.
_validate_runset_tool_path = _strict._validate_runset_tool_path
_validate_invocation_command = _strict._validate_invocation_command
build_runtime_environment = _strict.build_runtime_environment


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def reject_gt(value: Any, label: str) -> None:
    """Reuse the strict recursive GT guard, with a v7-specific error type."""

    try:
        _strict.reject_gt(value, label)
    except _strict.DriverError as error:
        raise DriverError(str(error)) from error


def _reject_strict_artifact(path: Path, label: str) -> None:
    """Prevent accidental reuse of v6 strict evidence destinations."""

    text = str(path).replace("\\", "/").lower()
    strict_tokens = (
        "b07h_v3_ledger", "b07h_v3_invocation", "runtime_driver_result_v3",
        "runtime_driver_ledger_v3", "runtime_driver_deferred_v3",
    )
    if any(token in text for token in strict_tokens):
        raise DriverError(f"{label} points at the strict v6 evidence namespace")


def _path_overlaps(left: Path, right: Path) -> bool:
    """Return true when two lexical artifact paths alias or nest."""

    if _same_path(left, right):
        return True
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _validate_disjoint_artifacts(root: Path, artifacts: Mapping[str, Path]) -> None:
    """Reject file/sidecar/manifest/output/runtime path collisions."""

    checked = [(name, candidate_path(path, root, f"v7 {name}")) for name, path in artifacts.items()]
    for index, (left_name, left_path) in enumerate(checked):
        for right_name, right_path in checked[index + 1:]:
            if _path_overlaps(left_path, right_path):
                # A normal invocation owns exactly one child manifest at the
                # root of its output directory.  This is the sole intentional
                # file/directory nesting exception; aliases, nested manifests,
                # and cross-invocation output collisions remain forbidden.
                names = {left_name, right_name}
                if names in ({"current_manifest", "output_dir"}, {"current_manifest_sidecar", "output_dir"}):
                    manifest_name = "current_manifest" if "current_manifest" in names else "current_manifest_sidecar"
                    manifest_path = left_path if left_name == manifest_name else right_path
                    output_path = left_path if left_name == "output_dir" else right_path
                    owned_manifest = candidate_path(output_path / "manifest.json", root, "v7 owned child manifest")
                    if manifest_name == "current_manifest_sidecar":
                        owned_manifest = candidate_path(Path(str(owned_manifest) + ".sha256"), root, "v7 owned child manifest sidecar")
                    if _same_path(manifest_path, owned_manifest):
                        continue
                raise DriverError(f"v7 artifact paths overlap: {left_name}={left_path} and {right_name}={right_path}")


def _atomic_bytes(path: Path, data: bytes, root: Path, *, replace: bool) -> None:
    root = require_e_root(root)
    lexical = _lexical_candidate(path, root, "v7 atomic output")
    _reject_reparse_components(lexical, root, "v7 atomic output")
    target = candidate_path(lexical, root, "v7 atomic output")
    target_parent = lexical.parent
    target_parent.mkdir(parents=True, exist_ok=True)
    _reject_reparse_components(lexical, root, "v7 atomic output")
    if not target_parent.is_dir() or _is_reparse(target_parent):
        raise DriverError(f"v7 atomic output parent is not a regular directory: {target_parent}")
    parent_stat = target_parent.stat()
    parent_identity = (getattr(parent_stat, "st_dev", None), getattr(parent_stat, "st_ino", None))
    if not replace and (os.path.lexists(str(lexical)) or _sidecar_path(target, root).exists()):
        raise FileExistsError(f"refusing to overwrite E-only v7 artifact: {target}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{lexical.name}.", suffix=".tmp", dir=str(target_parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # Recheck the lexical chain, parent identity, and destination just
        # before replacement.  A junction/symlink swap after the initial
        # check must fail closed instead of redirecting the atomic write.
        _reject_reparse_components(Path(temporary), root, "v7 atomic temporary")
        _reject_reparse_components(lexical, root, "v7 atomic output")
        if not target_parent.is_dir() or _is_reparse(target_parent):
            raise DriverError(f"v7 atomic output parent changed to a reparse/non-directory: {target_parent}")
        current_stat = target_parent.stat()
        current_identity = (getattr(current_stat, "st_dev", None), getattr(current_stat, "st_ino", None))
        if current_identity != parent_identity:
            raise DriverError(f"v7 atomic output parent changed during write: {target_parent}")
        if _is_reparse(lexical):
            raise DriverError(f"v7 atomic output destination is a symlink/reparse point: {lexical}")
        resolved = _resolved(lexical)
        if resolved.drive.upper() != "E:" or not _is_relative_to(resolved, root):
            raise DriverError(f"v7 atomic output escaped candidate root: {resolved}")
        _reject_reparse_components(resolved, root, "v7 atomic output")
        os.replace(temporary, lexical)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _sidecar_path(path: Path, root: Path) -> Path:
    return candidate_path(Path(str(path) + ".sha256"), root, "v7 hash sidecar")


def validate_sidecar(path: Path, root: Path, label: str) -> str:
    path = require_regular_candidate_file(path, root, label)
    actual = digest(path)
    sidecar = _sidecar_path(path, root)
    sidecar = require_regular_candidate_file(sidecar, root, f"{label} sidecar")
    try:
        fields = sidecar.read_text(encoding="ascii").strip().split()
    except (OSError, UnicodeError) as error:
        raise DriverError(f"{label} sidecar is unreadable") from error
    if len(fields) != 2 or not re.fullmatch(r"[0-9A-Fa-f]{64}", fields[0]) or fields[1] != path.name or fields[0].upper() != actual.upper():
        raise DriverError(f"{label} sidecar hash mismatch")
    return actual


def atomic_json(path: Path, value: Mapping[str, Any], root: Path, *, replace: bool) -> str:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_bytes(path, data, root, replace=replace)
    actual = digest(candidate_path(path, root, "v7 atomic output"))
    sidecar = _sidecar_path(candidate_path(path, root, "v7 atomic output"), root)
    _atomic_bytes(sidecar, f"{actual}  {Path(path).name}\n".encode("ascii"), root, replace=True)
    return actual


def workspace_state(workspace_root: Path | str) -> dict[str, Any]:
    """Recursively report C target/temp/cache/bytecode contamination."""

    root = _resolved(workspace_root)
    if not root.is_dir():
        return {"workspace_root": str(root), "missing": True, "clean": False, "forbidden": [str(root)]}
    forbidden: list[str] = []
    for directory, directories, files in os.walk(root):
        current = Path(directory)
        relative_parts = {part.lower() for part in current.relative_to(root).parts}
        if ".git" in relative_parts:
            directories[:] = []
            continue
        for name in list(directories):
            if name.lower() in {"target", "temp", *FORBIDDEN_C_CACHE_NAMES}:
                forbidden.append(str(current / name))
        for name in files:
            if name.lower().endswith((".pyc", ".pyo")):
                forbidden.append(str(current / name))
    return {
        "workspace_root": str(root),
        "target_present": any(Path(item).name.lower() == "target" for item in forbidden),
        "temp_present": any(Path(item).name.lower() == "temp" for item in forbidden),
        "cache_present": any(Path(item).name.lower() in {name.lower() for name in FORBIDDEN_C_CACHE_NAMES} or item.lower().endswith((".pyc", ".pyo")) for item in forbidden),
        "forbidden": sorted(set(forbidden)),
        "clean": not forbidden,
    }


def c_workspace_state(workspace_root: Path | str = DEFAULT_C_WORKSPACE) -> dict[str, Any]:
    root = _resolved(workspace_root)
    # The production default is the fixed C: workspace.  An explicitly
    # supplied workspace is also accepted for isolated E:-resident tests;
    # ``main`` always uses the default and therefore remains C:-fail-closed.
    if root.drive.upper() != "C:" and _same_path(root, _resolved(DEFAULT_C_WORKSPACE)):
        return {"workspace_root": str(root), "missing": True, "clean": False, "forbidden": [str(root)]}
    return workspace_state(root)


def require_workspace_clean(workspace_root: Path | str) -> dict[str, Any]:
    state = workspace_state(workspace_root)
    if not state["clean"]:
        raise DriverError("C workspace target/temp/cache artifacts are present: " + ", ".join(state["forbidden"][:8]))
    return state


def require_c_workspace_clean(workspace_root: Path | str = DEFAULT_C_WORKSPACE) -> dict[str, Any]:
    return require_workspace_clean(workspace_root)


def _fixed_tool_paths(value: Any, root: Path) -> dict[str, Path]:
    if not isinstance(value, Mapping) or set(value) != set(FIXED_TOOL_KEYS):
        raise DriverError("v7 runset fixed_tools must contain the exact tool inventory")
    checked: dict[str, Path] = {}
    for key in FIXED_TOOL_KEYS:
        item = value.get(key)
        if not isinstance(item, Mapping):
            raise DriverError(f"v7 fixed tool {key} declaration is malformed")
        try:
            path = _validate_runset_tool_path(item.get("path"), root, f"v7 fixed tool {key}", allow_c_readonly=item.get("read_only") is True)
            claimed = _sha_claim(item.get("sha256"), f"v7 fixed tool {key}")
        except _strict.DriverError as error:
            raise DriverError(str(error)) from error
        if digest(path).upper() != claimed:
            raise DriverError(f"v7 fixed tool {key} SHA mismatch")
        if path.drive.upper() == "C:" and item.get("read_only") is not True:
            raise DriverError(f"v7 fixed tool {key} C: path must be explicitly read-only")
        checked[key] = path
    return checked


def _fixed_invocation(index: int, raw: Mapping[str, Any]) -> None:
    expected_id, expected_engine, expected_sequence, expected_cells = INVOCATION_CELLS[index - 1]
    if raw.get("id") != expected_id or raw.get("engine") != expected_engine or raw.get("sequence") != expected_sequence:
        raise DriverError(f"v7 runset invocation {index} identity/order mismatch")
    if raw.get("result_cells") != list(expected_cells) or raw.get("ground_truth_argument_present") is not False:
        raise DriverError(f"v7 runset invocation {index} cell/GT contract mismatch")
    command = raw.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command) or command_contains_gt(command):
        raise DriverError(f"v7 runset invocation {index} command is malformed or GT-bearing")


def _validate_ambient_policy(value: Mapping[str, Any], label: str = "v7 ambient policy") -> None:
    if value.get("ambient_policy") != AMBIENT_POLICY:
        raise DriverError(f"{label} must be exactly ambient_policy=recorded")


def validate_runset_value(value: Mapping[str, Any], root: Path) -> dict[str, Any]:
    root = require_e_root(root)
    if value.get("schema") != RUNSET_SCHEMA:
        raise DriverError("v7 driver requires B07H_GT_FREE_RUNTIME_RUNSET_V3")
    if value.get("status") != "fixed_preflight_only":
        raise DriverError("v7 runset status is not fixed_preflight_only")
    if require_e_root(value.get("candidate_root"), "v7 runset declared candidate root") != root:
        raise DriverError("v7 runset declared candidate root mismatch")
    if value.get("supersedes_schema") != RUNSET_V2_SCHEMA or not re.fullmatch(r"[0-9A-Fa-f]{64}", str(value.get("supersedes_sha256", ""))):
        raise DriverError("v7 runset does not bind a v2 runset")
    _validate_ambient_policy(value)
    recording = value.get("ambient_recording")
    if not isinstance(recording, Mapping):
        raise DriverError("v7 runset ambient_recording metadata is required")
    if recording.get("finite_window") is not True or recording.get("noise_is_informational") is not True:
        raise DriverError("v7 runset ambient recording policy is malformed")
    if recording.get("hard_blockers") != ["target_processes", "c_workspace", "e_free_threshold"]:
        raise DriverError("v7 runset ambient hard-blocker inventory mismatch")
    source = value.get("source")
    protocol = value.get("protocol")
    if not isinstance(source, Mapping) or not isinstance(source.get("path"), str) or str(source.get("sha256", "")).upper() != EXPECTED_SOURCE_SHA256:
        raise DriverError("v7 runset source binding mismatch")
    if not isinstance(protocol, Mapping) or not isinstance(protocol.get("path"), str) or str(protocol.get("sha256", "")).upper() != EXPECTED_PROTOCOL_SHA256:
        raise DriverError("v7 runset protocol binding mismatch")
    source_path = require_regular_candidate_file(source["path"], root, "v7 runset source")
    protocol_path = require_regular_candidate_file(protocol["path"], root, "v7 runset protocol")
    if digest(source_path).upper() != EXPECTED_SOURCE_SHA256 or digest(protocol_path).upper() != EXPECTED_PROTOCOL_SHA256:
        raise DriverError("v7 runset source/protocol bytes do not match the frozen hashes")
    if "ambient_oracle" in value:
        raise DriverError("v7 runset must not carry the strict v6 ambient oracle")
    storage_gate = value.get("storage_gate")
    if not isinstance(storage_gate, Mapping) or storage_gate.get("stop_threshold_bytes") != STOP_FREE_BYTES or storage_gate.get("stop_threshold_gib") != 250 or storage_gate.get("check_before_each_invocation") is not True or storage_gate.get("unstarted_cells_if_below_threshold") != "DNF and preserve denominator 9":
        raise DriverError("v7 runset storage gate must preserve the 250 GiB/9-cell contract")
    environment = value.get("environment")
    if isinstance(environment, Mapping):
        for key, nested in environment.items():
            if not isinstance(nested, str) or not re.match(r"^[A-Za-z]:[\\/]", nested):
                continue
            upper_key = str(key).upper()
            if upper_key in E_ENV_SUFFIXES or any(marker in upper_key for marker in ("CACHE", "TEMP", "TMP", "CARGO", "RUSTUP", "TORCH", "NUMBA", "HUGGINGFACE", "XDG")):
                if not nested[:1].upper() == "E":
                    raise DriverError(f"v7 runset environment {key} is not E:-resident")
    prepared = value.get("prepared_inputs")
    if prepared is not None:
        if not isinstance(prepared, list):
            raise DriverError("v7 runset prepared input inventory is malformed")
        for index, item in enumerate(prepared, 1):
            if not isinstance(item, Mapping) or not isinstance(item.get("prepared_dir"), str):
                raise DriverError(f"v7 runset prepared input {index} path is missing")
            candidate_path(item["prepared_dir"], root, f"v7 prepared input {index}")
    tools = _fixed_tool_paths(value.get("fixed_tools"), root)
    invocations = value.get("invocations")
    if not isinstance(invocations, list) or len(invocations) != TOTAL_INVOCATIONS:
        raise DriverError("v7 runset invocation denominator mismatch")
    flattened: list[str] = []
    expected_serial_order = [
        "MH_01_easy visloc", "MH_01_easy colmap (incremental + global cells)",
        "MH_03_medium visloc", "MH_03_medium colmap (incremental + global cells)",
        "MH_05_difficult visloc", "MH_05_difficult colmap (incremental + global cells)",
    ]
    for index, raw in enumerate(invocations, 1):
        if not isinstance(raw, Mapping):
            raise DriverError(f"v7 runset invocation {index} is malformed")
        _fixed_invocation(index, raw)
        flattened.extend(raw["result_cells"])
        output = raw.get("output")
        if not isinstance(output, str):
            raise DriverError(f"v7 runset invocation {index} output missing")
        candidate_path(output, root, f"v7 invocation {index} output")
        try:
            _validate_invocation_command(index, raw, root, tools, protocol_path)
        except _strict.DriverError as error:
            raise DriverError(str(error)) from error
    if flattened != list(RESULT_CELLS) or value.get("serial_order") != expected_serial_order:
        raise DriverError("v7 runset serial/cell order mismatch")
    policy = value.get("runtime_policy")
    required_false = ("mapping_executed", "gt_opened", "performance_claim", "ground_truth_argument_present_anywhere")
    if not isinstance(policy, Mapping) or any(policy.get(key) is not False for key in required_false) or policy.get("output_paths_preflight_absent") is not True or policy.get("serial_only") is not True or policy.get("total_invocations") != 6 or policy.get("total_result_cells") != 9:
        raise DriverError("v7 runset runtime policy mismatch")
    reject_gt(value, "v7 runset")
    return dict(value)


def validate_runset(path: Path, root: Path, expected_sha256: str) -> dict[str, Any]:
    root = require_e_root(root)
    path = candidate_path(path, root, "v7 runset")
    actual = validate_sidecar(path, root, "v7 runset")
    if actual.upper() != expected_sha256.upper():
        raise DriverError(f"v7 runset SHA mismatch: expected {expected_sha256}, got {actual}")
    return validate_runset_value(read_json(path), root)


def _free_bytes(root: Path) -> int:
    return int(shutil.disk_usage(root).free)


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _process_target_entries(value: Any) -> list[Any]:
    if not isinstance(value, Mapping):
        return []
    entries: list[Any] = []
    for key in ("target_processes", "conflicts", "processes"):
        nested = value.get(key)
        if isinstance(nested, list):
            entries.extend(nested)
    return entries


def _name_of_process(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("name") or value.get("process_name") or value.get("comm") or "").lower()
    return str(value).lower()


def _target_conflicts(process: Any, wsl: Any) -> list[Any]:
    conflicts: list[Any] = []
    for item in [*_process_target_entries(process), *_process_target_entries(wsl)]:
        name = _name_of_process(item)
        basename = re.split(r"[\\/]", name)[-1]
        if name in SEARCH_INDEXER_NAMES or basename in SEARCH_INDEXER_NAMES:
            continue
        if name in TARGET_PROCESS_NAMES or basename in TARGET_PROCESS_NAMES:
            conflicts.append(item)
    return conflicts


def _default_process_sampler() -> dict[str, Any]:
    """Best-effort Windows sampler; missing optional psutil is non-fatal."""

    try:
        import psutil  # type: ignore
    except Exception:
        return {"total_processor_percent": None, "search_indexer_percent": None, "target_processes": [], "sampler_error": "psutil unavailable"}
    target: list[dict[str, Any]] = []
    search_percent = 0.0
    for process in psutil.process_iter(["pid", "name"]):
        try:
            name = str(process.info.get("name") or "")
            lowered = name.lower()
            cpu = _as_float(process.cpu_percent(interval=None)) or 0.0
            if lowered in SEARCH_INDEXER_NAMES:
                search_percent += cpu
            if lowered in TARGET_PROCESS_NAMES:
                target.append({"pid": process.info.get("pid"), "name": name, "cpu_percent": cpu})
        except (psutil.Error, OSError):
            continue
    return {
        "total_processor_percent": _as_float(psutil.cpu_percent(interval=None)),
        "search_indexer_percent": search_percent,
        "target_processes": target,
    }


def _default_wsl_sampler() -> dict[str, Any]:
    command = ["wsl.exe", "-e", "ps", "-eo", "pid=,comm="]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=3.0, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        return {"status": "unavailable", "target_processes": [], "sampler_error": f"{type(error).__name__}: {error}"}
    target: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2 and fields[1].lower() in TARGET_PROCESS_NAMES:
            target.append({"pid": fields[0], "name": fields[1]})
    return {"status": "idle" if completed.returncode == 0 else "error", "returncode": completed.returncode, "target_processes": target}


def _default_gpu_sampler() -> dict[str, Any]:
    command = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=3.0, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "utilization_percent": None, "memory_used_mib": None, "sampler_error": f"{type(error).__name__}: {error}"}
    rows = []
    for line in completed.stdout.splitlines():
        fields = [item.strip() for item in line.split(",", 1)]
        if len(fields) == 2:
            rows.append({"utilization_percent": _as_float(fields[0]), "memory_used_mib": _as_float(fields[1])})
    if not rows:
        return {"available": False, "utilization_percent": None, "memory_used_mib": None, "returncode": completed.returncode}
    return {
        "available": True,
        "utilization_percent": max((_as_float(item["utilization_percent"]) or 0.0) for item in rows),
        "memory_used_mib": sum((_as_float(item["memory_used_mib"]) or 0.0) for item in rows),
    }


def ambient_sample(
    root: Path,
    workspace_root: Path | str = DEFAULT_C_WORKSPACE,
    *,
    process_sampler: Callable[[], Mapping[str, Any]] | None = None,
    wsl_sampler: Callable[[], Mapping[str, Any]] | None = None,
    gpu_sampler: Callable[[], Mapping[str, Any]] | None = None,
    free_bytes_fn: Callable[[Path], int] | None = None,
    now_fn: Callable[[], str] = utc_now,
) -> dict[str, Any]:
    """Take one bounded ambient observation.

    ``cpu_settled``, ``search_settled``, and ``gpu_settled`` are deliberately
    informational.  They are retained in ``noise`` and ``checks`` for audit
    context but do not affect ``start_allowed``.
    """

    root = require_e_root(root)
    workspace_root = _resolved(workspace_root)
    process = dict((process_sampler or _default_process_sampler)())
    wsl = dict((wsl_sampler or _default_wsl_sampler)())
    gpu = dict((gpu_sampler or _default_gpu_sampler)())
    free_fn = free_bytes_fn or _free_bytes
    try:
        free_bytes = int(free_fn(root))
    except Exception as error:
        free_bytes = -1
        free_error = f"{type(error).__name__}: {error}"
    else:
        free_error = None
    c_state = c_workspace_state(workspace_root)
    conflicts = _target_conflicts(process, wsl)
    cpu = _as_float(process.get("total_processor_percent", process.get("total_cpu_percent", process.get("cpu_percent"))))
    search = _as_float(process.get("search_indexer_percent", process.get("search_indexer_cpu_percent")))
    gpu_util = _as_float(gpu.get("utilization_percent"))
    checks = {
        "target_processes_clear": not conflicts,
        "c_workspace_clean": c_state.get("clean") is True,
        "e_free_threshold": free_bytes >= STOP_FREE_BYTES,
        "cpu_settled": cpu is None or cpu <= CPU_SETTLE_LIMIT_PERCENT,
        "search_settled": search is None or search <= SEARCH_INDEXER_SETTLE_LIMIT_PERCENT,
        "gpu_settled": gpu_util is None or gpu_util <= 0.0,
    }
    return {
        "schema": AMBIENT_HISTORY_SCHEMA,
        "ambient_policy": AMBIENT_POLICY,
        "status": "recorded",
        "timestamp_utc": now_fn(),
        "checks": checks,
        "hard_blockers": {
            "target_processes": conflicts,
            "c_workspace": c_state,
            "e_free_threshold": {"free_bytes": free_bytes, "required_bytes": STOP_FREE_BYTES, "error": free_error},
        },
        "noise": {
            "cpu_percent": cpu,
            "search_indexer_percent": search,
            "gpu": gpu,
            "cpu_informational": True,
            "search_indexer_informational": True,
            "gpu_informational": True,
        },
        "process": process,
        "wsl": wsl,
        "gpu": gpu,
        "free_bytes": free_bytes,
        "start_allowed": bool(checks["target_processes_clear"] and checks["c_workspace_clean"] and checks["e_free_threshold"]),
    }


def _validate_history_event(value: Mapping[str, Any], label: str = "v7 ambient history event") -> None:
    _validate_ambient_policy(value, label)
    if value.get("schema") not in {AMBIENT_HISTORY_SCHEMA, DEFERRED_SCHEMA}:
        raise DriverError(f"{label} schema mismatch")
    # A history event may carry a deferred status, but no GT-bearing material
    # is permitted in either the observation or deferred sidecar stream.
    for key, item in value.items():
        if key != "status":
            reject_gt({key: item}, label)


def _validate_history_manifest(root: Path, history_path: Path, history_sha: str) -> None:
    manifest = candidate_path(Path(str(history_path) + ".manifest"), root, "v7 ambient manifest")
    require_regular_candidate_file(manifest, root, "v7 ambient manifest")
    validate_sidecar(manifest, root, "v7 ambient manifest")
    value = read_json(manifest)
    if value.get("schema") != AMBIENT_MANIFEST_SCHEMA:
        raise DriverError("v7 ambient manifest schema mismatch")
    _validate_ambient_policy(value, "v7 ambient manifest")
    claimed_path = candidate_path(value.get("path"), root, "v7 ambient manifest path")
    if not _same_path(claimed_path, history_path):
        raise DriverError("v7 ambient manifest path mismatch")
    if _sha_claim(value.get("sha256"), "v7 ambient manifest history SHA") != history_sha.upper():
        raise DriverError("v7 ambient manifest history SHA mismatch")


def _validate_existing_history(root: Path, history_path: Path) -> Path:
    history_path = candidate_path(history_path, root, "v7 ambient history")
    _reject_strict_artifact(history_path, "v7 ambient history")
    sidecar = _sidecar_path(history_path, root)
    manifest = candidate_path(Path(str(history_path) + ".manifest"), root, "v7 ambient manifest")
    manifest_sidecar = _sidecar_path(manifest, root)
    for item, label in ((history_path, "v7 ambient history"), (sidecar, "v7 ambient history sidecar"), (manifest, "v7 ambient manifest"), (manifest_sidecar, "v7 ambient manifest sidecar")):
        _reject_reparse_components(item, root, label)
    if not history_path.exists():
        if sidecar.exists() or manifest.exists() or manifest_sidecar.exists():
            raise DriverError("v7 ambient history seal exists without history")
        return history_path
    require_regular_candidate_file(history_path, root, "v7 ambient history")
    if not sidecar.exists() or not manifest.exists() or not manifest_sidecar.exists():
        raise DriverError("v7 ambient history seal is incomplete")
    history_sha = validate_sidecar(history_path, root, "v7 ambient history")
    _validate_history_manifest(root, history_path, history_sha)
    try:
        lines = history_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if line.strip():
                parsed = json.loads(line)
                if not isinstance(parsed, Mapping):
                    raise DriverError("v7 ambient history event is not an object")
                _validate_history_event(parsed)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DriverError("v7 ambient history is not valid JSONL") from error
    return history_path


def _seal_history(root: Path, history_path: Path) -> str | None:
    history_path = candidate_path(history_path, root, "v7 ambient history")
    if not history_path.exists():
        return None
    require_regular_candidate_file(history_path, root, "v7 ambient history")
    history_sha = digest(history_path)
    sidecar = _sidecar_path(history_path, root)
    _atomic_bytes(sidecar, f"{history_sha}  {history_path.name}\n".encode("ascii"), root, replace=True)
    manifest = candidate_path(Path(str(history_path) + ".manifest"), root, "v7 ambient manifest")
    atomic_json(manifest, {"schema": AMBIENT_MANIFEST_SCHEMA, "ambient_policy": AMBIENT_POLICY, "path": str(history_path), "sha256": history_sha}, root, replace=True)
    _validate_history_manifest(root, history_path, history_sha)
    return history_sha


def _append_history_event(root: Path, history_path: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    history_path = _validate_existing_history(root, history_path)
    event_value = dict(event)
    _validate_history_event(event_value)
    previous = history_path.read_bytes() if history_path.exists() else b""
    encoded = (json.dumps(event_value, sort_keys=True) + "\n").encode("utf-8")
    _atomic_bytes(history_path, previous + encoded, root, replace=True)
    if _seal_history(root, history_path) is None:
        raise DriverError("v7 ambient history disappeared while sealing")
    return event_value


def append_deferred(root: Path, history_path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    event = {**dict(payload), "schema": DEFERRED_SCHEMA, "ambient_policy": AMBIENT_POLICY, "status": "deferred", "deferred_cells": list(payload.get("deferred_cells", [])), "deferred_diagnostic_path": str(history_path)}
    return _append_history_event(root, history_path, event)


def record_ambient_window(
    root: Path,
    history_path: Path,
    cells: Sequence[str] | None = None,
    *,
    workspace_root: Path | str = DEFAULT_C_WORKSPACE,
    window_seconds: float = DEFAULT_AMBIENT_SAMPLE_SECONDS * DEFAULT_AMBIENT_SAMPLES,
    sample_seconds: float = DEFAULT_AMBIENT_SAMPLE_SECONDS,
    samples: int | None = None,
    process_sampler: Callable[[], Mapping[str, Any]] | None = None,
    wsl_sampler: Callable[[], Mapping[str, Any]] | None = None,
    gpu_sampler: Callable[[], Mapping[str, Any]] | None = None,
    free_bytes_fn: Callable[[Path], int] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], str] = utc_now,
) -> dict[str, Any]:
    """Record a deterministic finite observation window and return its gate.

    The number of samples is fixed by ``samples`` when supplied; otherwise it
    is derived from the finite ``window_seconds`` and ``sample_seconds``.  A
    zero interval is valid for tests and CI.  There is deliberately no loop
    that waits for CPU/SearchIndexer/GPU to become quiet.
    """

    root = require_e_root(root)
    if sample_seconds < 0 or window_seconds < 0:
        raise DriverError("ambient observation durations must be non-negative")
    if samples is None:
        samples = max(1, int(math.ceil(window_seconds / sample_seconds))) if sample_seconds > 0 else 1
    if type(samples) is not int or samples < 1 or samples > 3600:
        raise DriverError("ambient observation sample count must be between 1 and 3600")
    observations: list[dict[str, Any]] = []
    for index in range(samples):
        observation = ambient_sample(root, workspace_root, process_sampler=process_sampler, wsl_sampler=wsl_sampler, gpu_sampler=gpu_sampler, free_bytes_fn=free_bytes_fn, now_fn=now_fn)
        observation["sample_index"] = index + 1
        observation["window_samples"] = samples
        _append_history_event(root, history_path, observation)
        observations.append(observation)
        if index + 1 < samples and sample_seconds > 0:
            sleep_fn(sample_seconds)
    blocker_names = ("target_processes", "c_workspace", "e_free_threshold")
    blockers = [name for name in blocker_names if any(not item["checks"][{"target_processes": "target_processes_clear", "c_workspace": "c_workspace_clean", "e_free_threshold": "e_free_threshold"}[name]] for item in observations)]
    return {
        "schema": AMBIENT_HISTORY_SCHEMA,
        "ambient_policy": AMBIENT_POLICY,
        "status": "recorded",
        "reason": "recorded" if not blockers else "hard_blocker",
        "samples": len(observations),
        "window_seconds": window_seconds,
        "sample_seconds": sample_seconds,
        "observations": observations,
        "hard_blockers": blockers,
        "start_allowed": not blockers,
        "noise_is_informational": True,
        "observed_cells": list(cells or []),
        "noise_observations": observations,
        "last_sample": observations[-1],
    }


# Descriptive aliases make the non-gating policy explicit to callers while
# keeping one implementation and one finite-loop semantics.
observe_ambient = ambient_sample
observe_ambient_window = record_ambient_window
record_ambient = record_ambient_window


def _cells_for(index: int) -> list[str]:
    if not 1 <= index <= TOTAL_INVOCATIONS:
        raise DriverError("invocation-index must be between 1 and 6")
    return list(INVOCATION_CELLS[index - 1][3])


def _dnf_cells(cells: Sequence[str], reason: str, status: str = "dnf") -> list[dict[str, Any]]:
    if status not in {"success", "dnf"}:
        raise DriverError("attempted to create a non-terminal cell status")
    return [{"id": cell, "status": status, "reason": reason} for cell in cells]


def normalize_terminal_status(raw_status: Any, returncode: int | None, reason: Any = None) -> tuple[str, str | None]:
    raw = str(raw_status or "").lower()
    if raw == "success" and returncode == 0:
        return "success", None
    text = str(reason or raw or "child result status is missing/unknown")
    return "dnf", text


def _empty_ledger() -> dict[str, Any]:
    return {
        "schema": LEDGER_SCHEMA,
        "ambient_policy": AMBIENT_POLICY,
        "total_result_cells": TOTAL_RESULT_CELLS,
        "expected_cells": list(RESULT_CELLS),
        "results": [],
        "cells": {},
        "denominator": {
            "total_cells": TOTAL_RESULT_CELLS,
            "completed_cells": [],
            "completed_count": 0,
            "remaining_cells": list(RESULT_CELLS),
            "remaining_count": TOTAL_RESULT_CELLS,
        },
        "updated_utc": utc_now(),
    }


def _validate_result_manifest(root: Path, value: Mapping[str, Any], label: str) -> None:
    manifest = value.get("manifest")
    if not isinstance(manifest, Mapping) or manifest.get("ambient_policy") != AMBIENT_POLICY:
        raise DriverError(f"{label} manifest policy is not recorded")
    manifest_path_claim = manifest.get("path")
    manifest_sha_claim = manifest.get("sha256")
    if manifest_path_claim is None:
        if manifest_sha_claim is not None:
            raise DriverError(f"{label} missing manifest path with a SHA claim")
        return
    manifest_path = require_regular_candidate_file(manifest_path_claim, root, f"{label} child manifest")
    if not isinstance(manifest_sha_claim, str) or digest(manifest_path).upper() != manifest_sha_claim.upper():
        raise DriverError(f"{label} child manifest hash mismatch")


def _declared_result_artifacts(root: Path, value: Mapping[str, Any], label: str) -> dict[str, Path]:
    """Return every result-declared evidence path for collision checking."""

    artifacts: dict[str, Path] = {}
    manifest = value.get("manifest")
    if isinstance(manifest, Mapping) and manifest.get("path") is not None:
        manifest_path = require_regular_candidate_file(manifest["path"], root, f"{label} child manifest")
        artifacts["manifest"] = manifest_path
        artifacts["manifest_sidecar"] = Path(str(manifest_path) + ".sha256")
    observation = value.get("ambient_observation")
    if isinstance(observation, Mapping) and observation.get("history_path") is not None:
        history_path = require_regular_candidate_file(observation["history_path"], root, f"{label} ambient history")
        artifacts["history"] = history_path
        artifacts["history_sidecar"] = Path(str(history_path) + ".sha256")
        history_manifest = Path(str(history_path) + ".manifest")
        artifacts["history_manifest"] = history_manifest
        artifacts["history_manifest_sidecar"] = Path(str(history_manifest) + ".sha256")
    return artifacts


def _validate_terminal_result_artifact(
    root: Path,
    record: Mapping[str, Any],
    index: int,
    expected: tuple[str, str, str, tuple[str, ...]],
    ledger_path: Path,
) -> tuple[dict[str, Any], str]:
    label = f"v7 ledger invocation {index} result"
    result_path_claim = record.get("result_path")
    if not isinstance(result_path_claim, str):
        raise DriverError(f"{label} path is missing")
    result_path = require_regular_candidate_file(result_path_claim, root, label)
    result_sidecar = Path(str(result_path) + ".sha256")
    _validate_disjoint_artifacts(root, {
        "result": result_path,
        "result_sidecar": result_sidecar,
        "ledger": ledger_path,
        "ledger_sidecar": Path(str(ledger_path) + ".sha256"),
    })
    actual_sha = validate_sidecar(result_path, root, label)
    claimed_sha = _sha_claim(record.get("result_sha256"), f"{label} SHA")
    if actual_sha.upper() != claimed_sha:
        raise DriverError(f"{label} SHA mismatch")
    value = read_json(result_path)
    if value.get("schema") != RESULT_SCHEMA or value.get("ambient_policy") != AMBIENT_POLICY or value.get("terminal") is not True or value.get("attempt_terminal") is not True or not isinstance(value.get("finished_utc"), str):
        raise DriverError(f"{label} schema/policy/terminal flags are malformed")
    expected_id, expected_engine, expected_sequence, expected_cells = expected
    if value.get("invocation_index") != index or value.get("invocation") != expected_id or value.get("engine") != expected_engine or value.get("sequence") != expected_sequence:
        raise DriverError(f"{label} invocation identity mismatch")
    if value.get("result_cells") != list(expected_cells):
        raise DriverError(f"{label} result-cell order mismatch")
    cell_results = value.get("cell_results")
    if not isinstance(cell_results, list) or len(cell_results) != len(expected_cells):
        raise DriverError(f"{label} cell inventory is malformed")
    statuses: list[str] = []
    for cell, item in zip(expected_cells, cell_results):
        if not isinstance(item, Mapping) or item.get("id") != cell or item.get("status") not in {"success", "dnf"}:
            raise DriverError(f"{label} cell identity/status mismatch")
        statuses.append(str(item["status"]))
    expected_status = "dnf" if "dnf" in statuses else "success"
    if value.get("status") != expected_status or record.get("status") != expected_status:
        raise DriverError(f"{label} top-level status mismatch")
    if value.get("mapping_started") is not True:
        raise DriverError(f"{label} mapping-start evidence is missing")
    for key in ("gt_opened", "ground_truth_read", "ground_truth_materialized", "ground_truth_argument_present_anywhere"):
        if value.get(key) is not False:
            raise DriverError(f"{label} {key} must be false")
    if not isinstance(value.get("runset_sha256"), str) or not re.fullmatch(r"[0-9A-Fa-f]{64}", value["runset_sha256"]):
        raise DriverError(f"{label} runset SHA is malformed")
    if str(value.get("source_sha256", "")).upper() != EXPECTED_SOURCE_SHA256 or str(value.get("protocol_sha256", "")).upper() != EXPECTED_PROTOCOL_SHA256:
        raise DriverError(f"{label} source/protocol hash binding mismatch")
    _validate_result_manifest(root, value, label)
    _validate_ambient_result_metadata(root, value, label)
    reject_gt(value, label)
    if record.get("result_cells") != list(expected_cells) or record.get("invocation_index") != index or record.get("invocation") != expected_id or not isinstance(record.get("finished_utc"), str):
        raise DriverError(f"{label} ledger record identity mismatch")
    return value, actual_sha.upper()


def _validate_ambient_result_metadata(root: Path, value: Mapping[str, Any], label: str) -> None:
    observation = value.get("ambient_observation")
    if not isinstance(observation, Mapping) or observation.get("ambient_policy") != AMBIENT_POLICY:
        raise DriverError(f"{label} ambient observation policy is missing")
    history_claim = observation.get("history_path")
    if history_claim is not None:
        history_path = candidate_path(history_claim, root, f"{label} ambient history")
        _validate_existing_history(root, history_path)
        history_sha_claim = observation.get("history_sha256")
        if not isinstance(history_sha_claim, str) or digest(history_path).upper() != history_sha_claim.upper():
            raise DriverError(f"{label} ambient history SHA mismatch")


def read_ledger(path: Path | None, root: Path) -> dict[str, Any]:
    path = candidate_path(path or LEDGER_RELATIVE_PATH, root, "v7 ambient-recorded ledger")
    _reject_strict_artifact(path, "v7 ledger")
    if not path.exists():
        if _sidecar_path(path, root).exists():
            raise DriverError("v7 ledger hash sidecar exists without its ledger")
        return _empty_ledger()
    validate_sidecar(path, root, "v7 ambient-recorded ledger")
    value = read_json(path)
    if value.get("schema") != LEDGER_SCHEMA or value.get("ambient_policy") != AMBIENT_POLICY or value.get("expected_cells") != list(RESULT_CELLS) or value.get("total_result_cells") != TOTAL_RESULT_CELLS:
        raise DriverError("v7 ledger schema/policy/denominator mismatch")
    if not isinstance(value.get("results"), list) or not isinstance(value.get("cells"), dict) or not set(value["cells"]).issubset(set(RESULT_CELLS)):
        raise DriverError("v7 ledger inventory malformed")
    if len(value["results"]) > TOTAL_INVOCATIONS or not all(isinstance(item, Mapping) for item in value["results"]):
        raise DriverError("v7 ledger invocation records are not a strict serial prefix")
    indexes = [item.get("invocation_index") for item in value["results"]]
    if indexes != list(range(1, len(value["results"]) + 1)):
        raise DriverError("v7 ledger invocation records are not a strict serial prefix")
    expected_cell_records: dict[str, tuple[int, str, str]] = {}
    ledger_artifacts: dict[str, Path] = {
        "ledger": path,
        "ledger_sidecar": Path(str(path) + ".sha256"),
    }
    for index, record in enumerate(value["results"], 1):
        expected = INVOCATION_CELLS[index - 1]
        if record.get("ambient_policy") != AMBIENT_POLICY or record.get("invocation_index") != index:
            raise DriverError(f"v7 ledger invocation {index} identity/policy is malformed")
        result_value, result_sha = _validate_terminal_result_artifact(root, record, index, expected, path)
        result_path = require_regular_candidate_file(record["result_path"], root, f"v7 ledger invocation {index} result")
        ledger_artifacts[f"result_{index}"] = result_path
        ledger_artifacts[f"result_{index}_sidecar"] = Path(str(result_path) + ".sha256")
        for name, artifact in _declared_result_artifacts(root, result_value, f"v7 ledger invocation {index} result").items():
            ledger_artifacts[f"{name}_{index}"] = artifact
        for cell, item in zip(expected[3], result_value["cell_results"]):
            if cell in expected_cell_records:
                raise DriverError(f"v7 ledger cell appears in multiple invocation records: {cell}")
            expected_cell_records[cell] = (index, str(item["status"]), result_sha)
    _validate_disjoint_artifacts(root, ledger_artifacts)
    if set(value["cells"]) != set(expected_cell_records):
        raise DriverError("v7 ledger cell inventory does not match invocation result cells")
    for cell in expected_cell_records:
        item = value["cells"].get(cell)
        expected_index, expected_status, expected_sha = expected_cell_records[cell]
        if not isinstance(item, Mapping) or item.get("ambient_policy") != AMBIENT_POLICY or item.get("status") != expected_status or item.get("invocation_index") != expected_index or str(item.get("result_sha256", "")).upper() != expected_sha:
            raise DriverError(f"v7 ledger cell {cell} identity/status/policy/hash is malformed")
        expected_invocation = INVOCATION_CELLS[expected_index - 1]
        if cell not in expected_invocation[3] or item.get("result_path") != value["results"][expected_index - 1].get("result_path"):
            raise DriverError(f"v7 ledger cell {cell} is assigned to the wrong invocation/result")
    denominator = value.get("denominator")
    completed = [cell for cell in RESULT_CELLS if cell in value["cells"]]
    expected_denominator = {
        "total_cells": TOTAL_RESULT_CELLS,
        "completed_cells": completed,
        "completed_count": len(completed),
        "remaining_cells": [cell for cell in RESULT_CELLS if cell not in value["cells"]],
        "remaining_count": TOTAL_RESULT_CELLS - len(completed),
    }
    if denominator != expected_denominator:
        raise DriverError("v7 ledger denominator is inconsistent with its cell inventory")
    reject_gt(value, "v7 ledger")
    return value


def record_result(
    root: Path,
    result_path: Path,
    payload: Mapping[str, Any],
    *,
    ledger_path: Path | None = None,
    history_path: Path | None = None,
    driver_log: Path | None = None,
    output_dir: Path | None = None,
    runtime_temp: Path | None = None,
) -> dict[str, Any]:
    root = require_e_root(root)
    result_path = candidate_path(result_path, root, "v7 result")
    _reject_strict_artifact(result_path, "v7 result")
    ledger_path = candidate_path(ledger_path or LEDGER_RELATIVE_PATH, root, "v7 ambient-recorded ledger")
    _reject_strict_artifact(ledger_path, "v7 ledger")
    _validate_disjoint_artifacts(root, {
        "result": result_path,
        "result_sidecar": Path(str(result_path) + ".sha256"),
        "ledger": ledger_path,
        "ledger_sidecar": Path(str(ledger_path) + ".sha256"),
    })
    if payload.get("ambient_policy") != AMBIENT_POLICY:
        raise DriverError("v7 terminal result must set ambient_policy=recorded")
    runset_sha = payload.get("runset_sha256")
    if not isinstance(runset_sha, str) or not re.fullmatch(r"[0-9A-Fa-f]{64}", runset_sha):
        raise DriverError("v7 terminal result runset SHA is malformed")
    if str(payload.get("source_sha256", "")).upper() != EXPECTED_SOURCE_SHA256 or str(payload.get("protocol_sha256", "")).upper() != EXPECTED_PROTOCOL_SHA256:
        raise DriverError("v7 terminal result source/protocol hash binding mismatch")
    cells = payload.get("cell_results")
    if not isinstance(cells, list) or not cells or not all(isinstance(item, Mapping) for item in cells):
        raise DriverError("terminal result has no cells")
    expected_index = int(payload.get("invocation_index", 0))
    ids = [str(item.get("id")) for item in cells]
    if ids != _cells_for(expected_index) or len(ids) != len(set(ids)) or any(item.get("status") not in {"success", "dnf"} for item in cells):
        raise DriverError("terminal result cell order/identity/status mismatch")
    expected_invocation = INVOCATION_CELLS[expected_index - 1]
    if payload.get("invocation") != expected_invocation[0] or payload.get("engine") != expected_invocation[1] or payload.get("sequence") != expected_invocation[2]:
        raise DriverError("terminal result invocation identity mismatch")
    for key in ("gt_opened", "ground_truth_read", "ground_truth_materialized", "ground_truth_argument_present_anywhere"):
        if payload.get(key) is not False:
            raise DriverError(f"terminal result {key} must be false")
    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping) or manifest.get("ambient_policy") != AMBIENT_POLICY:
        raise DriverError("v7 terminal result manifest must set ambient_policy=recorded")
    manifest_path_claim = manifest.get("path")
    manifest_sha_claim = manifest.get("sha256")
    if manifest_path_claim is not None:
        manifest_path = require_regular_candidate_file(manifest_path_claim, root, "v7 terminal result child manifest")
        if not isinstance(manifest_sha_claim, str) or digest(manifest_path).upper() != manifest_sha_claim.upper():
            raise DriverError("v7 terminal result child manifest hash mismatch")
    observation = payload.get("ambient_observation")
    if observation is not None and (not isinstance(observation, Mapping) or observation.get("ambient_policy") != AMBIENT_POLICY):
        raise DriverError("v7 terminal result ambient observation policy is malformed")
    ledger = read_ledger(ledger_path, root)
    current_value_for_paths = {
        **dict(payload),
        "ambient_observation": dict(observation) if isinstance(observation, Mapping) else {"ambient_policy": AMBIENT_POLICY},
    }
    _validate_ambient_result_metadata(root, current_value_for_paths, "v7 current result")
    evidence_paths: dict[str, Path] = {
        "current_result": result_path,
        "current_result_sidecar": Path(str(result_path) + ".sha256"),
        "ledger": ledger_path,
        "ledger_sidecar": Path(str(ledger_path) + ".sha256"),
    }
    for name, artifact in _declared_result_artifacts(root, current_value_for_paths, "v7 current result").items():
        evidence_paths[f"current_{name}"] = artifact
    optional_paths = {
        "history": history_path,
        "driver_log": driver_log,
        "output_dir": output_dir,
        "runtime_temp": runtime_temp,
    }
    for name, artifact in optional_paths.items():
        if artifact is not None:
            if name == "history" and "current_history" in evidence_paths and _same_path(artifact, evidence_paths["current_history"]):
                continue
            evidence_paths[name] = artifact
    for prior_index, prior_record in enumerate(ledger["results"], 1):
        prior_result = require_regular_candidate_file(prior_record["result_path"], root, f"v7 prior result {prior_index}")
        evidence_paths[f"prior_{prior_index}_result"] = prior_result
        evidence_paths[f"prior_{prior_index}_result_sidecar"] = Path(str(prior_result) + ".sha256")
        prior_value = read_json(prior_result)
        for name, artifact in _declared_result_artifacts(root, prior_value, f"v7 prior result {prior_index}").items():
            evidence_paths[f"prior_{prior_index}_{name}"] = artifact
    _validate_disjoint_artifacts(root, evidence_paths)
    next_index = len(ledger["results"]) + 1
    if expected_index != next_index:
        raise DriverError(f"v7 ledger requires strict serial invocation {next_index}, got {expected_index}")
    existing = set(str(item) for item in ledger["cells"])
    if existing.intersection(ids):
        raise FileExistsError(f"v7 result cells already recorded: {sorted(existing.intersection(ids))}")
    status_values = [str(item["status"]) for item in cells]
    status = "dnf" if "dnf" in status_values else "success"
    if payload.get("status") != status:
        raise DriverError("terminal top-level status does not match cell statuses")
    result_value = {**dict(payload), "schema": RESULT_SCHEMA, "ambient_policy": AMBIENT_POLICY, "status": status, "terminal": True, "attempt_terminal": True, "ambient_observation": dict(observation) if isinstance(observation, Mapping) else {"ambient_policy": AMBIENT_POLICY}, "finished_utc": str(payload.get("finished_utc") or utc_now())}
    reject_gt(result_value, "v7 result")
    if result_path.exists() or _sidecar_path(result_path, root).exists():
        raise FileExistsError(f"refusing to overwrite v7 result: {result_path}")
    result_sha = atomic_json(result_path, result_value, root, replace=False)
    record = {"ambient_policy": AMBIENT_POLICY, "invocation_index": expected_index, "invocation": payload.get("invocation"), "result_path": str(result_path), "result_sha256": result_sha, "result_cells": ids, "status": status, "finished_utc": result_value["finished_utc"]}
    ledger["results"].append(record)
    for item in cells:
        ledger["cells"][str(item["id"])] = {"ambient_policy": AMBIENT_POLICY, "status": item["status"], "invocation_index": expected_index, "result_path": str(result_path), "result_sha256": result_sha}
    completed = [cell for cell in RESULT_CELLS if cell in ledger["cells"]]
    ledger["denominator"] = {"total_cells": TOTAL_RESULT_CELLS, "completed_cells": completed, "completed_count": len(completed), "remaining_cells": [cell for cell in RESULT_CELLS if cell not in ledger["cells"]], "remaining_count": TOTAL_RESULT_CELLS - len(completed)}
    ledger["updated_utc"] = utc_now()
    atomic_json(ledger_path, ledger, root, replace=True)
    return result_value


def _read_child_manifest(root: Path, output_dir: Path) -> tuple[Path, dict[str, Any], str | None]:
    try:
        manifest_candidate = _lexical_candidate(output_dir / "manifest.json", root, "v7 child manifest")
        _reject_reparse_components(manifest_candidate, root, "v7 child manifest")
        if _is_reparse(manifest_candidate):
            raise DriverError(f"v7 child manifest is a symlink/reparse point: {manifest_candidate}")
        if not manifest_candidate.exists():
            return manifest_candidate, {}, None
        manifest_path = require_regular_candidate_file(manifest_candidate, root, "v7 child manifest")
        return manifest_path, read_json(manifest_path), None
    except Exception as error:
        # Return a candidate-owned fallback only for diagnostics.  The caller
        # must treat any non-None error as DNF and must not hash this path.
        fallback = Path(os.path.normpath(str(output_dir / "manifest.json")))
        return fallback, {}, f"child manifest validation/read exception: {type(error).__name__}: {error}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runset", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--expected-runset-sha256", required=True)
    parser.add_argument("--invocation-index", type=int, required=True)
    parser.add_argument("--runtime-temp", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--driver-log", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--ambient-window-seconds", type=float, default=DEFAULT_AMBIENT_SAMPLE_SECONDS * DEFAULT_AMBIENT_SAMPLES)
    parser.add_argument("--sample-seconds", type=float, default=DEFAULT_AMBIENT_SAMPLE_SECONDS)
    parser.add_argument("--ambient-samples", type=int)
    args = parser.parse_args(argv)
    root = require_e_root(args.candidate_root)
    if not 1 <= args.invocation_index <= TOTAL_INVOCATIONS:
        raise DriverError("invocation-index must be between 1 and 6")
    runset_path = candidate_path(args.runset, root, "v7 runset")
    runset = validate_runset(runset_path, root, args.expected_runset_sha256)
    invocation = runset["invocations"][args.invocation_index - 1]
    cells = list(invocation["result_cells"])
    result_path = candidate_path(args.result or Path("logs") / f"B07H_v4_ambient_recorded_invocation_{args.invocation_index:02d}.json", root, "v7 result")
    history_path = candidate_path(args.history or Path("logs") / f"B07H_v4_ambient_recorded_invocation_{args.invocation_index:02d}.jsonl", root, "v7 ambient history")
    driver_log = candidate_path(args.driver_log or Path("logs") / f"B07H_v4_ambient_recorded_invocation_{args.invocation_index:02d}.log", root, "v7 driver log")
    ledger_path = candidate_path(args.ledger or LEDGER_RELATIVE_PATH, root, "v7 ambient-recorded ledger")
    _reject_strict_artifact(result_path, "v7 result")
    _reject_strict_artifact(history_path, "v7 history")
    _reject_strict_artifact(driver_log, "v7 driver log")
    _reject_strict_artifact(ledger_path, "v7 ledger")
    output_dir = candidate_path(invocation["output"], root, "v7 invocation output")
    runtime_temp = candidate_path(args.runtime_temp, root, "v7 runtime temp")
    history_manifest = candidate_path(Path(str(history_path) + ".manifest"), root, "v7 ambient manifest")
    artifacts = {
        "runset": runset_path,
        "runset_sidecar": Path(str(runset_path) + ".sha256"),
        "result": result_path,
        "result_sidecar": Path(str(result_path) + ".sha256"),
        "history": history_path,
        "history_sidecar": Path(str(history_path) + ".sha256"),
        "history_manifest": history_manifest,
        "history_manifest_sidecar": Path(str(history_manifest) + ".sha256"),
        "driver_log": driver_log,
        "ledger": ledger_path,
        "ledger_sidecar": Path(str(ledger_path) + ".sha256"),
        "output_dir": output_dir,
        "runtime_temp": runtime_temp,
    }
    _validate_disjoint_artifacts(root, artifacts)
    for artifact, label in ((result_path, "v7 result"), (history_path, "v7 ambient history"), (driver_log, "v7 driver log"), (ledger_path, "v7 ledger"), (output_dir, "v7 invocation output")):
        _reject_reparse_components(artifact, root, label)
    if output_dir.exists():
        raise DriverError(f"v7 invocation output already exists; refusing stale evidence: {output_dir}")
    if result_path.exists() or _sidecar_path(result_path, root).exists():
        raise DriverError(f"v7 invocation result already exists: {result_path}")
    ledger = read_ledger(ledger_path, root)
    expected_index = len(ledger["results"]) + 1
    if args.invocation_index != expected_index:
        raise DriverError(f"v7 ledger requires strict serial invocation {expected_index}, got {args.invocation_index}")
    if set(cells).intersection(ledger.get("cells", {})):
        raise DriverError(f"v7 invocation cells are already present in the ledger: {cells}")
    # This is the only pre-window hard check.  CPU/SearchIndexer/GPU are not
    # consulted for admission and therefore cannot make this call wait.
    require_c_workspace_clean()
    if args.validation_only:
        print(json.dumps({"schema": "B07H_RUNTIME_DRIVER_VALIDATION_V4", "ambient_policy": AMBIENT_POLICY, "status": "validation_only", "passed": True, "invocation_index": args.invocation_index, "result_cells": cells, "candidate_root": str(root), "runset_sha256": args.expected_runset_sha256.upper()}, sort_keys=True))
        return 0
    ambient = record_ambient_window(root, history_path, cells, window_seconds=args.ambient_window_seconds, sample_seconds=args.sample_seconds, samples=args.ambient_samples)
    if not ambient["start_allowed"]:
        append_deferred(root, history_path, {"invocation": invocation["id"], "deferred_cells": cells, "reason": "hard start blocker: " + ",".join(ambient["hard_blockers"]), "ambient": ambient})
        print(history_path)
        return 4
    env, locations = build_runtime_environment(root, args.invocation_index, runtime_temp)
    if driver_log.exists():
        raise FileExistsError(f"refusing to overwrite v7 driver log: {driver_log}")
    driver_log.parent.mkdir(parents=True, exist_ok=True)
    returncode: int | None = None
    launch_error: str | None = None
    with driver_log.open("x", encoding="utf-8") as stream:
        try:
            process = subprocess.Popen([str(item) for item in invocation["command"]], cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
            returncode = process.wait()
        except Exception as error:
            launch_error = f"child launch exception: {type(error).__name__}: {error}"
    manifest_path, child_manifest, manifest_error = _read_child_manifest(root, output_dir)
    try:
        reject_gt(child_manifest, "v7 child manifest")
    except Exception as error:
        child_manifest = {}
        manifest_error = manifest_error or f"child manifest validation exception: {type(error).__name__}: {error}"
    raw_status = "failure" if launch_error or manifest_error else child_manifest.get("status")
    reason_hint = launch_error or manifest_error or child_manifest.get("reason")
    status, reason = normalize_terminal_status(raw_status, returncode, reason_hint)
    cell_results = _dnf_cells(cells, reason or "terminal success", status) if status == "dnf" else [{"id": cell, "status": "success"} for cell in cells]
    manifest_valid = manifest_error is None and manifest_path.is_file()
    manifest = {"ambient_policy": AMBIENT_POLICY, "path": str(manifest_path) if manifest_valid else None, "sha256": digest(manifest_path) if manifest_valid else None}
    payload = {"schema": RESULT_SCHEMA, "ambient_policy": AMBIENT_POLICY, "status": status, "mapping_started": True, "invocation_index": args.invocation_index, "invocation": invocation["id"], "engine": invocation["engine"], "sequence": invocation["sequence"], "result_cells": cells, "cell_results": cell_results, "runset_sha256": args.expected_runset_sha256.upper(), "source_sha256": EXPECTED_SOURCE_SHA256, "protocol_sha256": EXPECTED_PROTOCOL_SHA256, "gt_opened": False, "ground_truth_read": False, "ground_truth_materialized": False, "ground_truth_argument_present_anywhere": False, "manifest": manifest, "runtime_environment": locations, "ambient_observation": {"ambient_policy": AMBIENT_POLICY, "history_path": str(history_path), "history_sha256": digest(history_path) if history_path.is_file() else None}, "finished_utc": utc_now()}
    record_result(root, result_path, payload, ledger_path=ledger_path, history_path=history_path, driver_log=driver_log, output_dir=output_dir, runtime_temp=runtime_temp)
    print(result_path)
    return 0 if status == "success" else 2


__all__ = [
    "ALLOWED_E_ROOTS", "DEFAULT_C_WORKSPACE", "DRIVER_VERSION", "RUNSET_SCHEMA", "RUNSET_V2_SCHEMA", "EXPECTED_RUNSET_SCHEMA", "EXPECTED_RUNSET_V2_SCHEMA", "RESULT_SCHEMA", "DEFERRED_SCHEMA", "LEDGER_SCHEMA", "AMBIENT_POLICY", "RESULT_CELLS", "INVOCATION_CELLS", "STOP_FREE_BYTES", "CPU_SETTLE_LIMIT_PERCENT", "SEARCH_INDEXER_SETTLE_LIMIT_PERCENT", "GPU_MEMORY_GROWTH_TOLERANCE_MIB", "TARGET_PROCESS_NAMES", "LEDGER_RELATIVE_PATH", "AMBIENT_LEDGER_RELATIVE_PATH", "DEFAULT_LEDGER_PATH", "DriverError", "digest", "candidate_path", "require_e_root", "require_allowed_e_path", "require_regular_candidate_file", "require_c_readonly_file", "reject_gt", "validate_runset", "validate_runset_value", "build_runtime_environment", "workspace_state", "c_workspace_state", "require_workspace_clean", "require_c_workspace_clean", "ambient_sample", "observe_ambient", "record_ambient_window", "observe_ambient_window", "record_ambient", "normalize_terminal_status", "record_result", "append_deferred", "read_ledger", "atomic_json", "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
