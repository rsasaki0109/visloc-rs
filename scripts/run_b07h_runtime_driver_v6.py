#!/usr/bin/env python3
"""GT-free B07-H runtime driver v6 with a strict E:-only storage boundary.

This is a versioned successor to the frozen v5 driver.  It intentionally does
not edit or import the active candidate's result files.  A v6 runset must be
``B07H_GT_FREE_RUNTIME_RUNSET_V2`` and must bind its exact bytes with
  ``--expected-runset-sha256``.  The six-invocation/nine-result-cell accounting is
fixed at the original six invocations and nine result cells.

The driver never opens a held-out dataset or ground-truth member.  Deferred
ambient/resource diagnostics are append-only and never enter the terminal
ledger denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

sys.dont_write_bytecode = True

RUNSET_SCHEMA = "B07H_GT_FREE_RUNTIME_RUNSET_V2"
RUNSET_V1_SCHEMA = "B07H_GT_FREE_RUNTIME_RUNSET_V1"
RESULT_SCHEMA = "B07H_RUNTIME_DRIVER_RESULT_V3"
DEFERRED_SCHEMA = "B07H_RUNTIME_DRIVER_DEFERRED_V3"
LEDGER_SCHEMA = "B07H_RUNTIME_DRIVER_LEDGER_V3"
DRIVER_VERSION = "B07H_RUNTIME_DRIVER_V6"
EXPECTED_SOURCE_SHA256 = "38A704369AF7EC4898307D2EA61016260834DA7CFD452CB17B51FBAD621CCA8D"
EXPECTED_PROTOCOL_SHA256 = "4CE9B2306E5559325A42BDB3E46C51B1991691B5C0D2FC67F48E41F76B70BB7F"
FROZEN_V1_RUNSET_SHA256 = "99265390BFA15C4086FAC91713F46C24F3EE2B7838F1525CADD461E8CF00BD4F"
AMBIENT_ORACLE_RELATIVE_PATH = Path("scripts/run_b07h_runtime_driver_v5.py")
AMBIENT_ORACLE_SHA256 = "EB49541B6CB7A3FF784E7AA5F56184A670D919B5A2C5B6813779CAB0DC79384E"
# These values are part of the frozen v5 ambient contract.  v6 exposes them
# for runset/test consumers, but deliberately does not reimplement the gate:
# the verified v5 ``settle_ambient`` remains the production oracle.
STOP_FREE_BYTES = 250 * 1024**3
CPU_SETTLE_LIMIT_PERCENT = 15.0
SEARCH_INDEXER_SETTLE_LIMIT_PERCENT = 10.0
GPU_MEMORY_GROWTH_TOLERANCE_MIB = 64.0
DEFAULT_CONSECUTIVE_SAMPLES = 5
TOTAL_INVOCATIONS = 6
TOTAL_RESULT_CELLS = 9

SERIAL_ORDER = (
    "visloc_MH_01_easy",
    "colmap_MH_01_easy",
    "visloc_MH_03_medium",
    "colmap_MH_03_medium",
    "visloc_MH_05_difficult",
    "colmap_MH_05_difficult",
)
RESULT_CELLS = (
    "visloc_MH_01_easy",
    "colmap_inc_MH_01_easy",
    "colmap_global_MH_01_easy",
    "visloc_MH_03_medium",
    "colmap_inc_MH_03_medium",
    "colmap_global_MH_03_medium",
    "visloc_MH_05_difficult",
    "colmap_inc_MH_05_difficult",
    "colmap_global_MH_05_difficult",
)
INVOCATION_CELLS = (
    ("visloc_MH_01_easy", "visloc", "MH_01_easy", ("visloc_MH_01_easy",)),
    ("colmap_MH_01_easy", "colmap", "MH_01_easy", ("colmap_inc_MH_01_easy", "colmap_global_MH_01_easy")),
    ("visloc_MH_03_medium", "visloc", "MH_03_medium", ("visloc_MH_03_medium",)),
    ("colmap_MH_03_medium", "colmap", "MH_03_medium", ("colmap_inc_MH_03_medium", "colmap_global_MH_03_medium")),
    ("visloc_MH_05_difficult", "visloc", "MH_05_difficult", ("visloc_MH_05_difficult",)),
    ("colmap_MH_05_difficult", "colmap", "MH_05_difficult", ("colmap_inc_MH_05_difficult", "colmap_global_MH_05_difficult")),
)
ALLOWED_E_ROOTS = (Path("E:/visloc_archive"), Path("E:/datasets"))
DEFAULT_C_WORKSPACE = Path("C:/Users/rsasa/Workspace/visloc-rs")
FORBIDDEN_C_CACHE_NAMES = ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache")
TARGET_PROCESS_NAMES = frozenset({
    "cargo", "cargo.exe", "rustc", "rustc.exe", "colmap", "colmap.exe",
    "sequential_sfm_demo", "sequential_sfm_demo.exe", "robosim", "robosim.exe",
    "searchindexer", "searchindexer.exe",
})
E_ENV_SUFFIXES = {
    "TEMP": "temp",
    "TMP": "temp",
    "TMPDIR": "temp",
    "PYTHONPYCACHEPREFIX": "pycache",
    "CARGO_TARGET_DIR": "cargo-target",
    "CARGO_HOME": "cargo-home",
    "RUSTUP_HOME": "rustup",
    "TORCH_HOME": "torch",
    "XDG_CACHE_HOME": "xdg-cache",
    "XDG_CONFIG_HOME": "xdg-config",
    "XDG_STATE_HOME": "xdg-state",
    "HF_HOME": "huggingface",
    "HF_DATASETS_CACHE": "huggingface/datasets",
    "TRANSFORMERS_CACHE": "huggingface/transformers",
    "MPLCONFIGDIR": "matplotlib",
    "NUMBA_CACHE_DIR": "numba",
    "PIP_CACHE_DIR": "pip",
    "UV_CACHE_DIR": "uv",
    "POETRY_CACHE_DIR": "poetry",
    "CUDA_CACHE_PATH": "cuda-cache",
    "TRITON_CACHE_DIR": "triton",
    "TORCH_EXTENSIONS_DIR": "torch-extensions",
    "RAY_TMPDIR": "ray",
    "JOBLIB_TEMP_FOLDER": "joblib",
    "NVDIFRAST_CACHE_DIR": "nvdiffrast",
    "CUDNN_CACHE_PATH": "cudnn-cache",
    "PYTHONUSERBASE": "python-user",
    "HOME": "home",
    "USERPROFILE": "userprofile",
    "APPDATA": "appdata",
    "LOCALAPPDATA": "localappdata",
}
ENV_KEYS_TO_SCRUB = frozenset(E_ENV_SUFFIXES) | frozenset({
    "HOMEDRIVE", "HOMEPATH", "RUSTC_WRAPPER", "SCCACHE_DIR", "CCACHE_DIR",
    "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM",
})
ENV_SCRUB_MARKERS = ("CACHE", "CARGO", "RUSTUP", "RUSTC_WRAPPER", "CCACHE", "SCCACHE", "PYTHONPYCACHE", "TMP", "TEMP", "TORCH", "NUMBA", "HUGGINGFACE", "XDG")
GT_KEY_PARTS = ("ground_truth", "groundtruth", "gt_opened", "gt_read", "gt_materialized", "gt_path")
GT_VALUE_PARTS = ("ground_truth", "groundtruth", "state_groundtruth", "gt_path")
FIXED_TOOL_KEYS = ("python", "hierarchical_runner", "hierarchical_executable", "colmap_runner", "colmap")
OUTPUT_PATH_FLAGS = frozenset({"--out-dir", "--output", "--output-path", "--result", "--result-path", "--runtime-temp", "--history", "--driver-log", "--ledger", "--manifest"})
C_READONLY_PATH_FLAGS = frozenset({"--source", "--script"})


class DriverError(ValueError, RuntimeError):
    """A deterministic v6 preflight/accounting failure."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest().upper()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DriverError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(value, dict):
        raise DriverError(f"JSON object expected: {path}")
    return value


def _resolved(path: Path | str) -> Path:
    if not isinstance(path, (str, Path)):
        raise DriverError(f"path value is not a path: {path!r}")
    return Path(path).expanduser().resolve(strict=False)


def _reject_path_syntax(value: Path | str, label: str) -> str:
    raw = str(value)
    if not raw or "\x00" in raw:
        raise DriverError(f"{label} is empty or contains NUL")
    # A single leading backslash is a Windows rooted path (relative to the
    # current drive), while two are UNC.  Both are absolute from a storage
    # boundary's point of view.  Drive-relative forms such as ``E:foo`` are
    # also rejected: they silently resolve against a process-specific CWD.
    if raw.startswith(("/", "\\", "//")) and not re.match(r"^[A-Za-z]:[\\/]", raw):
        raise DriverError(f"{label} must not be POSIX-absolute or UNC")
    if re.match(r"^[A-Za-z]:", raw) and not re.match(r"^[A-Za-z]:[\\/]", raw):
        raise DriverError(f"{label} must not be drive-relative")
    if any(part == ".." for part in re.split(r"[\\/]", raw)):
        raise DriverError(f"{label} contains traversal")
    return raw


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
    root = _resolved(path)
    if root.drive.upper() != "E:":
        raise DriverError(f"{label} must be on E:, got {root}")
    if not any(root == base or base in root.parents for base in (_resolved(item) for item in ALLOWED_E_ROOTS)):
        raise DriverError(f"{label} must be below E:\\visloc_archive or E:\\datasets: {root}")
    return root


def require_allowed_e_path(path: Path | str, label: str = "E path") -> Path:
    _reject_path_syntax(path, label)
    resolved = _resolved(path)
    if resolved.drive.upper() != "E:" or not any(resolved == base or base in resolved.parents for base in (_resolved(item) for item in ALLOWED_E_ROOTS)):
        raise DriverError(f"{label} must be below E:\\visloc_archive or E:\\datasets")
    return resolved


def require_regular_allowed_e_file(path: Path | str, label: str = "E file") -> Path:
    resolved = require_allowed_e_path(path, label)
    if _is_reparse(resolved) or not resolved.is_file():
        raise DriverError(f"{label} is not a regular E: file")
    return resolved


def candidate_path(value: Path | str, root: Path, label: str) -> Path:
    root = require_e_root(root)
    lexical = _lexical_candidate(value, root, label)
    resolved = _resolved(lexical)
    if resolved.drive.upper() != "E:":
        raise DriverError(f"{label} must be on E:")
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise DriverError(f"{label} escapes declared candidate root: {resolved}") from error
    return resolved


def _is_reparse(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISLNK(metadata.st_mode) or bool(getattr(metadata, "st_file_attributes", 0) & 0x400)


def _reject_reparse_components(path: Path, root: Path, label: str) -> None:
    current = path
    while True:
        if _is_reparse(current):
            raise DriverError(f"{label} contains a symlink/reparse component: {current}")
        if current == root:
            return
        if current.parent == current:
            raise DriverError(f"{label} escaped candidate root")
        current = current.parent


def require_regular_candidate_file(value: Path | str, root: Path, label: str) -> Path:
    """Require a non-reparse regular file whose lexical path stays in root."""

    root = require_e_root(root)
    path = _lexical_candidate(value, root, label)
    _reject_reparse_components(path, root, label)
    if not path.is_file():
        raise DriverError(f"{label} is missing or not a regular file: {path}")
    return path


def require_c_readonly_file(value: Path | str, label: str) -> Path:
    """Allow only an exact, non-reparse read-only source file under C workspace."""

    _reject_path_syntax(value, label)
    path = Path(value)
    if not path.is_absolute():
        raise DriverError(f"{label} must be an explicit C: read-only path")
    lexical = Path(os.path.normpath(str(path)))
    if lexical.drive.upper() != "C:":
        raise DriverError(f"{label} is not a C: read-only path")
    workspace = _resolved(DEFAULT_C_WORKSPACE)
    try:
        lexical.relative_to(workspace)
    except ValueError as error:
        raise DriverError(f"{label} is outside the fixed C workspace") from error
    if _is_reparse(lexical) or not lexical.is_file():
        raise DriverError(f"{label} is not a regular C read-only file")
    return lexical


def reject_gt(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in GT_KEY_PARTS):
                if nested is not False:
                    raise DriverError(f"{label} has a non-false GT field: {key}")
            if isinstance(nested, str) and any(part in nested.lower().replace("/", "\\") for part in GT_VALUE_PARTS):
                raise DriverError(f"{label} contains a GT token in field {key}")
            if isinstance(nested, str) and lowered == "status" and nested.lower() in {"deferred", "ambient_timeout"}:
                raise DriverError(f"{label} contains deferred status")
            reject_gt(nested, label)
    elif isinstance(value, list):
        for nested in value:
            reject_gt(nested, label)
    elif isinstance(value, str) and any(part in value.lower().replace("/", "\\") for part in GT_VALUE_PARTS):
        raise DriverError(f"{label} contains a GT token")


def command_contains_gt(command: Iterable[str]) -> bool:
    tokens = (str(item).lower().replace("/", "\\") for item in command)
    return any(any(part in token for part in ("ground_truth", "groundtruth", "state_groundtruth", "gt_path")) for token in tokens)


def _validate_runset_command_storage(command: Sequence[str], root: Path, label: str) -> None:
    """Reject absolute C:/other-drive command paths while allowing E:\\tools."""

    for token in command:
        match = re.search(r"([A-Za-z]):[\\/]", str(token))
        if match is None or match.group(1).upper() == "E":
            continue
        raise DriverError(f"{label} contains a non-E absolute path: {token}")


def _sha_claim(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9A-Fa-f]{64}", value):
        raise DriverError(f"{label} is not a SHA-256 claim")
    return value.upper()


def _validate_runset_tool_path(value: Any, root: Path, label: str, *, allow_c_readonly: bool = False) -> Path:
    if not isinstance(value, str):
        raise DriverError(f"{label} path is missing")
    _reject_path_syntax(value, label)
    path = Path(value)
    if not path.is_absolute():
        return require_regular_candidate_file(path, root, label)
    resolved = _resolved(path)
    if resolved.drive.upper() == "C:":
        if not allow_c_readonly:
            raise DriverError(f"{label} C: path is not an explicitly allowed read-only source")
        return require_c_readonly_file(resolved, label)
    if resolved.drive.upper() != "E:":
        raise DriverError(f"{label} must be E:-resident")
    try:
        return require_regular_candidate_file(resolved, root, label)
    except DriverError:
        tools = _resolved(Path("E:/tools"))
        try:
            resolved.relative_to(tools)
        except ValueError as error:
            raise DriverError(f"{label} absolute path must be below candidate root or E:\\tools") from error
        if _is_reparse(resolved) or not resolved.is_file():
            raise DriverError(f"{label} is not a regular E:\\tools file")
        return resolved


def _validate_fixed_tools(value: Any, root: Path) -> dict[str, Path]:
    if not isinstance(value, Mapping) or set(value) != set(FIXED_TOOL_KEYS):
        raise DriverError("v6 runset fixed_tools must contain the exact tool inventory")
    result: dict[str, Path] = {}
    for key in FIXED_TOOL_KEYS:
        item = value.get(key)
        if not isinstance(item, Mapping):
            raise DriverError(f"v6 fixed tool {key} declaration is malformed")
        path = _validate_runset_tool_path(item.get("path"), root, f"v6 fixed tool {key}", allow_c_readonly=item.get("read_only") is True)
        claimed = _sha_claim(item.get("sha256"), f"v6 fixed tool {key}")
        actual = digest(path)
        if actual.upper() != claimed:
            raise DriverError(f"v6 fixed tool {key} SHA mismatch")
        if path.drive.upper() == "C:" and item.get("read_only") is not True:
            raise DriverError(f"v6 fixed tool {key} C: path must be explicitly read-only")
        result[key] = path
    return result


def _command_resolved_path(value: str, root: Path, label: str, *, readonly_flag: str | None = None, allow_c_tool: bool = False, strict_file: bool = False) -> Path:
    _reject_path_syntax(value, label)
    path = Path(value)
    if re.match(r"^[A-Za-z]:[\\/]", value):
        if value[:1].upper() == "C":
            if not allow_c_tool and readonly_flag not in C_READONLY_PATH_FLAGS:
                raise DriverError(f"{label} C: path is not read-only input")
            return require_c_readonly_file(value, label)
        if value[:1].upper() != "E":
            raise DriverError(f"{label} must be E:-resident")
        try:
            return require_regular_candidate_file(value, root, label) if strict_file else candidate_path(value, root, label)
        except DriverError:
            tools = _resolved(Path("E:/tools"))
            resolved = _resolved(value)
            try:
                resolved.relative_to(tools)
            except ValueError as error:
                raise DriverError(f"{label} escapes candidate root/E:\\tools") from error
            if _is_reparse(resolved) or not resolved.is_file():
                raise DriverError(f"{label} is not a regular E:\\tools file")
            return resolved
    return require_regular_candidate_file(path, root, label) if strict_file or readonly_flag in C_READONLY_PATH_FLAGS else candidate_path(path, root, label)


def _same_path(left: Path, right: Path) -> bool:
    return str(left).replace("/", "\\").lower() == str(right).replace("/", "\\").lower()


def _validate_invocation_command(index: int, raw: Mapping[str, Any], root: Path, tools: Mapping[str, Path], protocol_path: Path) -> None:
    command = raw.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) and item for item in command) or command_contains_gt(command):
        raise DriverError(f"runset invocation {index} command is malformed or GT-bearing")
    # Validate syntax on every token, including unknown flags.  A path hidden
    # in an unclassified argument must not smuggle ``..`` or a drive-relative
    # component past the specific flag checks below.
    for position, token in enumerate(command):
        _reject_path_syntax(token.replace("=", "/"), f"runset invocation {index} command token {position}")
    expected_runner = "hierarchical_runner" if raw.get("engine") == "visloc" else "colmap_runner"
    first = _command_resolved_path(command[0], root, f"runset invocation {index} executable", allow_c_tool=True, strict_file=True)
    second = _command_resolved_path(command[1], root, f"runset invocation {index} runner", allow_c_tool=True, strict_file=True) if len(command) > 1 else None
    if not _same_path(first, tools["python"]) or second is None or not _same_path(second, tools[expected_runner]):
        raise DriverError(f"runset invocation {index} executable/runner is not fixed-tool bound")
    expected_execution_flag = "--exe" if raw.get("engine") == "visloc" else "--colmap"
    forbidden_execution_flag = "--colmap" if raw.get("engine") == "visloc" else "--exe"

    def flag_occurrences(flag: str) -> list[int]:
        prefix = flag + "="
        return [position for position, token in enumerate(command) if token == flag or token.startswith(prefix)]

    expected_positions = flag_occurrences(expected_execution_flag)
    if len(expected_positions) != 1 or command[expected_positions[0]] != expected_execution_flag:
        raise DriverError(f"runset invocation {index} must contain exactly one {expected_execution_flag} flag with a separate value")
    if flag_occurrences(forbidden_execution_flag):
        raise DriverError(f"runset invocation {index} contains the wrong engine execution flag")
    execution_position = expected_positions[0]
    if execution_position + 1 >= len(command):
        raise DriverError(f"runset invocation {index} {expected_execution_flag} value is missing")
    expected_execution_tool = tools["hierarchical_executable"] if expected_execution_flag == "--exe" else tools["colmap"]
    bound_execution_tool = _command_resolved_path(
        command[execution_position + 1],
        root,
        f"runset invocation {index} execution tool",
        allow_c_tool=True,
        strict_file=True,
    )
    if not _same_path(bound_execution_tool, expected_execution_tool) or digest(bound_execution_tool).upper() != digest(expected_execution_tool).upper():
        raise DriverError(f"runset invocation {index} execution tool is not fixed-tool bound")
    previous_flag: str | None = None
    for position, token in enumerate(command[2:], 2):
        flag = token.split("=", 1)[0] if token.startswith("--") else None
        raw_value = token.split("=", 1)[1] if flag is not None and "=" in token else token
        if flag in (OUTPUT_PATH_FLAGS | {"--exe", "--colmap", "--protocol", "--features-dir", "--timestamps", "--prepared-dir", "--config", "--source", "--script"}) and "=" not in token:
            previous_flag = flag
            continue
        if flag in OUTPUT_PATH_FLAGS or previous_flag in OUTPUT_PATH_FLAGS:
            output = candidate_path(raw_value, root, f"runset invocation {index} output argument {position}")
            if output.drive.upper() != "E:":
                raise DriverError(f"runset invocation {index} output argument is not E:")
        elif flag in {"--exe", "--colmap"} or previous_flag in {"--exe", "--colmap"}:
            expected_tool = tools["hierarchical_executable"] if (flag or previous_flag) == "--exe" else tools["colmap"]
            bound = _command_resolved_path(raw_value, root, f"runset invocation {index} execution tool", allow_c_tool=True, strict_file=True)
            if not _same_path(bound, expected_tool) or digest(bound).upper() != digest(expected_tool).upper():
                raise DriverError(f"runset invocation {index} execution tool is not fixed-tool bound")
        elif flag in {"--protocol", "--features-dir", "--timestamps", "--prepared-dir", "--config", "--source", "--script"} or previous_flag in {"--protocol", "--features-dir", "--timestamps", "--prepared-dir", "--config", "--source", "--script"}:
            bound = _command_resolved_path(raw_value, root, f"runset invocation {index} input argument {position}", readonly_flag=flag or previous_flag)
            if (flag or previous_flag) == "--protocol" and not _same_path(bound, protocol_path):
                raise DriverError(f"runset invocation {index} protocol path is not source-bound")
        elif re.search(r"[A-Za-z]:[\\/]", token) or token.startswith(("/", "\\\\", "//")):
            raise DriverError(f"runset invocation {index} has an unclassified absolute path: {token}")
        previous_flag = flag if flag is not None else None


def _sidecar_path(path: Path, root: Path) -> Path:
    return candidate_path(Path(str(path) + ".sha256"), root, "hash sidecar")


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


def _atomic_bytes(path: Path, data: bytes, root: Path, *, replace: bool) -> None:
    lexical = _lexical_candidate(path, require_e_root(root), "atomic output")
    _reject_reparse_components(lexical, require_e_root(root), "atomic output")
    path = candidate_path(lexical, root, "atomic output")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace and (path.exists() or _sidecar_path(path, root).exists()):
        raise FileExistsError(f"refusing to overwrite E-only artifact: {path}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def atomic_json(path: Path, value: Mapping[str, Any], root: Path, *, replace: bool) -> str:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_bytes(path, data, root, replace=replace)
    actual = digest(candidate_path(path, root, "atomic output"))
    sidecar = _sidecar_path(candidate_path(path, root, "atomic output"), root)
    sidecar_data = f"{actual}  {Path(path).name}\n".encode("ascii")
    _atomic_bytes(sidecar, sidecar_data, root, replace=True)
    return actual


def workspace_state(workspace_root: Path | str) -> dict[str, Any]:
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
    """Return the recursive fail-closed state for the monitored workspace."""

    root = _resolved(workspace_root)
    if root.drive.upper() != "C:":
        return {"workspace_root": str(root), "missing": True, "clean": False, "forbidden": [str(root)]}
    return workspace_state(root)


def require_c_workspace_clean() -> dict[str, Any]:
    return require_workspace_clean(DEFAULT_C_WORKSPACE)


def require_workspace_clean(workspace_root: Path | str) -> dict[str, Any]:
    state = workspace_state(workspace_root)
    if not state["clean"]:
        raise DriverError("C workspace target/temp/cache artifacts are present: " + ", ".join(state["forbidden"][:8]))
    return state


def build_runtime_environment(root: Path, invocation_index: int, runtime_temp: Path) -> tuple[dict[str, str], dict[str, str]]:
    root = require_e_root(root)
    runtime_temp = candidate_path(runtime_temp, root, "runtime temp")
    if not 1 <= invocation_index <= TOTAL_INVOCATIONS:
        raise DriverError("invocation-index must be between 1 and 6")
    base = runtime_temp / f"invocation-{invocation_index:02d}"
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ENV_KEYS_TO_SCRUB and not any(marker in key.upper() for marker in ENV_SCRUB_MARKERS)
    }
    locations: dict[str, str] = {}
    for key, suffix in E_ENV_SUFFIXES.items():
        location = candidate_path(base / suffix, root, f"child environment {key}")
        location.mkdir(parents=True, exist_ok=True)
        locations[key] = str(location)
        env[key] = str(location)
    env["HOMEDRIVE"] = "E:"
    env["HOMEPATH"] = str(base).replace("/", "\\")[2:]
    env["PYTHONPATH"] = str(candidate_path("scripts", root, "child PYTHONPATH"))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    # PATH is intentionally retained for executable lookup; it is not a
    # storage destination. Every storage/cache variable above is E:-resident.
    for key in E_ENV_SUFFIXES:
        if not Path(env[key]).is_absolute() or _resolved(env[key]).drive.upper() != "E:":
            raise DriverError(f"child environment {key} is not E:-resident")
    return env, locations


def _fixed_invocation(index: int, raw: Mapping[str, Any], root: Path) -> None:
    expected_id, expected_engine, expected_sequence, expected_cells = INVOCATION_CELLS[index - 1]
    if raw.get("id") != expected_id or raw.get("engine") != expected_engine or raw.get("sequence") != expected_sequence:
        raise DriverError(f"runset invocation {index} identity/order mismatch")
    if raw.get("result_cells") != list(expected_cells) or raw.get("ground_truth_argument_present") is not False:
        raise DriverError(f"runset invocation {index} cell/GT contract mismatch")
    command = raw.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command) or command_contains_gt(command):
        raise DriverError(f"runset invocation {index} command is malformed or GT-bearing")


def validate_runset_value(value: Mapping[str, Any], root: Path) -> dict[str, Any]:
    root = require_e_root(root)
    if value.get("schema") != RUNSET_SCHEMA:
        raise DriverError("v6 driver requires B07H_GT_FREE_RUNTIME_RUNSET_V2")
    if value.get("status") != "fixed_preflight_only":
        raise DriverError("v6 runset status is not fixed_preflight_only")
    if require_e_root(value.get("candidate_root"), "runset declared candidate root") != root:
        raise DriverError("runset declared candidate root mismatch")
    if value.get("supersedes_schema") != RUNSET_V1_SCHEMA or str(value.get("supersedes_sha256", "")).upper() != FROZEN_V1_RUNSET_SHA256:
        raise DriverError("v6 runset does not bind the frozen v1 runset")
    _validate_ambient_oracle(value.get("ambient_oracle"), root)
    source = value.get("source")
    protocol = value.get("protocol")
    if not isinstance(source, Mapping) or not isinstance(source.get("path"), str) or str(source.get("sha256", "")).upper() != EXPECTED_SOURCE_SHA256:
        raise DriverError("v6 runset source binding mismatch")
    if not isinstance(protocol, Mapping) or not isinstance(protocol.get("path"), str) or str(protocol.get("sha256", "")).upper() != EXPECTED_PROTOCOL_SHA256:
        raise DriverError("v6 runset protocol binding mismatch")
    source_path = require_regular_candidate_file(source["path"], root, "v6 runset source")
    protocol_path = require_regular_candidate_file(protocol["path"], root, "v6 runset protocol")
    if digest(source_path).upper() != EXPECTED_SOURCE_SHA256:
        raise DriverError("v6 runset source artifact bytes do not match the frozen source hash")
    if digest(protocol_path).upper() != EXPECTED_PROTOCOL_SHA256:
        raise DriverError("v6 runset protocol bytes do not match the frozen protocol hash")
    storage_gate = value.get("storage_gate")
    if not isinstance(storage_gate, Mapping):
        raise DriverError("v6 runset storage gate metadata is required")
    if storage_gate.get("stop_threshold_bytes") != STOP_FREE_BYTES:
        raise DriverError("v6 runset storage stop threshold must be exactly 250 GiB")
    if storage_gate.get("stop_threshold_gib") != 250:
        raise DriverError("v6 runset storage stop threshold GiB metadata mismatch")
    if storage_gate.get("check_before_each_invocation") is not True:
        raise DriverError("v6 runset must check storage before each invocation")
    if storage_gate.get("unstarted_cells_if_below_threshold") != "DNF and preserve denominator 9":
        raise DriverError("v6 runset storage failure must preserve the nine-cell denominator")
    prepared = value.get("prepared_inputs")
    if prepared is not None:
        if not isinstance(prepared, list):
            raise DriverError("v6 runset prepared input inventory is malformed")
        for index, item in enumerate(prepared, 1):
            if not isinstance(item, Mapping) or not isinstance(item.get("prepared_dir"), str):
                raise DriverError(f"v6 runset prepared input {index} path is missing")
            candidate_path(item["prepared_dir"], root, f"v6 runset prepared input {index}")
    tools = _validate_fixed_tools(value.get("fixed_tools"), root)
    invocations = value.get("invocations")
    if not isinstance(invocations, list) or len(invocations) != TOTAL_INVOCATIONS:
        raise DriverError("v6 runset invocation denominator mismatch")
    flattened: list[str] = []
    for index, raw in enumerate(invocations, 1):
        if not isinstance(raw, Mapping):
            raise DriverError(f"v6 runset invocation {index} is malformed")
        _fixed_invocation(index, raw, root)
        flattened.extend(raw["result_cells"])
        output = raw.get("output")
        if not isinstance(output, str):
            raise DriverError(f"v6 runset invocation {index} output missing")
        candidate_path(output, root, f"v6 invocation {index} output")
        _validate_invocation_command(index, raw, root, tools, protocol_path)
    if flattened != list(RESULT_CELLS) or value.get("serial_order") != [
        "MH_01_easy visloc", "MH_01_easy colmap (incremental + global cells)",
        "MH_03_medium visloc", "MH_03_medium colmap (incremental + global cells)",
        "MH_05_difficult visloc", "MH_05_difficult colmap (incremental + global cells)",
    ]:
        raise DriverError("v6 runset serial/cell order mismatch")
    policy = value.get("runtime_policy")
    if not isinstance(policy, Mapping) or policy.get("mapping_executed") is not False or policy.get("gt_opened") is not False or policy.get("performance_claim") is not False or policy.get("output_paths_preflight_absent") is not True or policy.get("serial_only") is not True or policy.get("total_invocations") != 6 or policy.get("total_result_cells") != 9 or policy.get("ground_truth_argument_present_anywhere") is not False:
        raise DriverError("v6 runset runtime policy mismatch")
    reject_gt(value, "v6 runset")
    return dict(value)


def _validate_ambient_oracle(value: Any, root: Path) -> dict[str, Any]:
    """Validate the hash-pinned v5 ambient-gate production oracle.

    The oracle is intentionally candidate-owned code.  A runset may not point
    at a different copy, a reparse/outside path, or an unsigned sidecar.
    """

    root = require_e_root(root)
    if not isinstance(value, Mapping):
        raise DriverError("v6 runset ambient_oracle metadata is required")
    expected_path = candidate_path(AMBIENT_ORACLE_RELATIVE_PATH, root, "v6 ambient oracle")
    path_claim = value.get("path")
    oracle_path = require_regular_candidate_file(path_claim, root, "v6 ambient oracle")
    if not _same_path(oracle_path, expected_path):
        raise DriverError("v6 ambient oracle path is not the pinned candidate path")
    claimed_sha = _sha_claim(value.get("sha256"), "v6 ambient oracle SHA")
    if claimed_sha != AMBIENT_ORACLE_SHA256:
        raise DriverError("v6 ambient oracle SHA is not the pinned v5 hash")
    if digest(oracle_path).upper() != claimed_sha:
        raise DriverError("v6 ambient oracle bytes do not match the pinned v5 hash")
    claimed_bytes = value.get("bytes")
    if type(claimed_bytes) is not int or claimed_bytes != oracle_path.stat().st_size:
        raise DriverError("v6 ambient oracle byte-size metadata mismatch")
    expected_sidecar = _sidecar_path(expected_path, root)
    sidecar_claim = value.get("sidecar")
    sidecar_path = require_regular_candidate_file(sidecar_claim, root, "v6 ambient oracle sidecar")
    if not _same_path(sidecar_path, expected_sidecar):
        raise DriverError("v6 ambient oracle sidecar path is not the pinned sidecar")
    validate_sidecar(oracle_path, root, "v6 ambient oracle")
    return {
        "path": str(oracle_path),
        "sha256": claimed_sha,
        "sidecar": str(sidecar_path),
        "bytes": claimed_bytes,
    }


def validate_runset(path: Path, root: Path, expected_sha256: str) -> dict[str, Any]:
    root = require_e_root(root)
    path = candidate_path(path, root, "v6 runset")
    actual = validate_sidecar(path, root, "v6 runset")
    if actual.upper() != expected_sha256.upper():
        raise DriverError(f"v6 runset SHA mismatch: expected {expected_sha256}, got {actual}")
    return validate_runset_value(read_json(path), root)


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
    if raw in {"failure", "failed", "error", "timeout", "ambient_timeout", "dnf", "partial_or_dnf"} or returncode not in (None, 0):
        return "dnf", text
    return "dnf", text


def _empty_ledger() -> dict[str, Any]:
    return {"schema": LEDGER_SCHEMA, "total_result_cells": 9, "expected_cells": list(RESULT_CELLS), "results": [], "cells": {}, "updated_utc": utc_now()}


def read_ledger(path: Path, root: Path) -> dict[str, Any]:
    path = candidate_path(path, root, "v6 ledger")
    if not path.exists():
        if _sidecar_path(path, root).exists():
            raise DriverError("v6 ledger hash sidecar exists without its ledger")
        return _empty_ledger()
    validate_sidecar(path, root, "v6 ledger")
    value = read_json(path)
    if value.get("schema") != LEDGER_SCHEMA or value.get("expected_cells") != list(RESULT_CELLS) or value.get("total_result_cells") != 9:
        raise DriverError("v6 ledger schema/denominator mismatch")
    if not isinstance(value.get("results"), list) or not isinstance(value.get("cells"), dict):
        raise DriverError("v6 ledger inventory malformed")
    if not set(value["cells"]).issubset(set(RESULT_CELLS)):
        raise DriverError("v6 ledger contains an unknown result cell")
    result_indexes = [item.get("invocation_index") for item in value["results"] if isinstance(item, Mapping)]
    if len(value["results"]) > TOTAL_INVOCATIONS or result_indexes != list(range(1, len(value["results"]) + 1)):
        raise DriverError("v6 ledger invocation records are not a strict serial prefix")
    for index, record in enumerate(value["results"], 1):
        expected = INVOCATION_CELLS[index - 1]
        if record.get("invocation") != expected[0] or record.get("result_cells") != list(expected[3]) or record.get("status") not in {"success", "dnf"} or not isinstance(record.get("result_sha256"), str):
            raise DriverError(f"v6 ledger invocation {index} identity/status/hash is malformed")
    reject_gt(value, "v6 ledger")
    return value


def record_result(root: Path, result_path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    root = require_e_root(root)
    result_path = candidate_path(result_path, root, "v6 result")
    ledger_path = candidate_path("logs/B07H_v3_ledger.json", root, "v6 ledger")
    cells = payload.get("cell_results")
    if not isinstance(cells, list) or not cells or not all(isinstance(item, Mapping) for item in cells):
        raise DriverError("terminal result has no cells")
    ids = [str(item.get("id")) for item in cells]
    expected_index = int(payload.get("invocation_index", 0))
    if ids != _cells_for(expected_index) or len(ids) != len(set(ids)):
        raise DriverError("terminal result cell order/identity mismatch")
    if any(item.get("status") not in {"success", "dnf"} for item in cells):
        raise DriverError("terminal result contains non-success/DNF status")
    expected_invocation = INVOCATION_CELLS[expected_index - 1]
    if payload.get("invocation") != expected_invocation[0] or payload.get("engine") != expected_invocation[1] or payload.get("sequence") != expected_invocation[2]:
        raise DriverError("terminal result invocation identity mismatch")
    for key in ("gt_opened", "ground_truth_read", "ground_truth_materialized", "ground_truth_argument_present_anywhere"):
        if payload.get(key) is not False:
            raise DriverError(f"terminal result {key} must be false")
    ledger = read_ledger(ledger_path, root)
    next_index = len(ledger["results"]) + 1
    if expected_index != next_index:
        raise DriverError(f"v6 ledger requires strict serial invocation {next_index}, got {expected_index}")
    existing = set(str(item) for item in ledger["cells"])
    if existing.intersection(ids):
        raise FileExistsError(f"v6 result cells already recorded: {sorted(existing.intersection(ids))}")
    status_values = [str(item["status"]) for item in cells]
    status = "dnf" if "dnf" in status_values else "success"
    if payload.get("status") != status:
        raise DriverError("terminal top-level status does not match cell statuses")
    result_value = {**dict(payload), "schema": RESULT_SCHEMA, "status": status, "terminal": True, "attempt_terminal": True, "finished_utc": str(payload.get("finished_utc") or utc_now())}
    reject_gt(result_value, "v6 result")
    if result_path.exists() or _sidecar_path(result_path, root).exists():
        raise FileExistsError(f"refusing to overwrite v6 result: {result_path}")
    result_sha = atomic_json(result_path, result_value, root, replace=False)
    record = {"invocation_index": expected_index, "invocation": payload.get("invocation"), "result_path": str(result_path), "result_sha256": result_sha, "result_cells": ids, "status": status, "finished_utc": result_value["finished_utc"]}
    ledger["results"].append(record)
    for item in cells:
        ledger["cells"][str(item["id"])] = {"status": item["status"], "invocation_index": expected_index, "result_path": str(result_path), "result_sha256": result_sha}
    completed = [cell for cell in RESULT_CELLS if cell in ledger["cells"]]
    ledger["denominator"] = {"total_cells": 9, "completed_cells": completed, "completed_count": len(completed), "remaining_cells": [cell for cell in RESULT_CELLS if cell not in ledger["cells"]], "remaining_count": 9 - len(completed)}
    ledger["updated_utc"] = utc_now()
    atomic_json(ledger_path, ledger, root, replace=True)
    return result_value


def append_deferred(root: Path, history_path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    root = require_e_root(root)
    history_path = _validate_existing_ambient_history(root, history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    event = {**dict(payload), "schema": DEFERRED_SCHEMA, "status": "deferred", "deferred_cells": list(payload.get("deferred_cells", [])), "deferred_diagnostic_path": str(history_path)}
    # The frozen v5 retry validator accepts raw samples and attempt markers.
    # A v6 deferred marker is not itself a sample, so carry the last sample's
    # checks/clean fields at the top level.  This keeps the JSONL append-only
    # stream retryable without changing the pinned v5 oracle bytes.
    ambient = event.get("ambient")
    last_sample = ambient.get("last_sample") if isinstance(ambient, Mapping) else None
    if isinstance(last_sample, Mapping):
        if isinstance(last_sample.get("checks"), Mapping):
            event["checks"] = dict(last_sample["checks"])
        if isinstance(last_sample.get("clean"), bool):
            event["clean"] = last_sample["clean"]
    event.setdefault("checks", {})
    event.setdefault("clean", False)
    # ``deferred`` is intentionally legal only in this append-only diagnostic
    # stream.  Validate every other field with the normal recursive GT guard;
    # terminal result/gate validators still reject this status outright.
    for key, value in event.items():
        if key != "status":
            reject_gt({key: value}, "v6 deferred history")
    sidecar = _sidecar_path(history_path, root)
    if history_path.exists():
        # ``_validate_existing_ambient_history`` already checked the whole
        # seal; retain the explicit bytes read only after that validation.
        previous = history_path.read_bytes()
    elif sidecar.exists():
        raise DriverError("v6 deferred history hash sidecar exists without its history")
    else:
        previous = b""
    encoded = (json.dumps(event, sort_keys=True) + "\n").encode("utf-8")
    _atomic_bytes(history_path, previous + encoded, root, replace=True)
    history_sha = _seal_ambient_history(root, history_path)
    if history_sha is None:
        raise DriverError("v6 deferred history disappeared while sealing")
    return event


def _validate_ambient_history_manifest(
    root: Path,
    history_path: Path,
    history_sha256: str,
) -> None:
    """Validate the v6 seal that accompanies an append-only history.

    The manifest is candidate-owned JSON and therefore gets the same regular,
    non-reparse and sidecar checks as every other v6 artifact.  Checking its
    path and digest claim before a retry prevents an old or redirected
    manifest from blessing a different history stream.
    """

    root = require_e_root(root)
    manifest = candidate_path(
        Path(str(history_path) + ".manifest"),
        root,
        "v6 ambient history manifest",
    )
    require_regular_candidate_file(manifest, root, "v6 ambient history manifest")
    validate_sidecar(manifest, root, "v6 ambient history manifest")
    value = read_json(manifest)
    if value.get("schema") != "B07H_RUNTIME_DRIVER_DEFERRED_SIDECAR_V1":
        raise DriverError("v6 ambient history manifest schema mismatch")
    claimed_path = value.get("path")
    if not isinstance(claimed_path, str):
        raise DriverError("v6 ambient history manifest path is missing")
    claimed_path = candidate_path(claimed_path, root, "v6 ambient history manifest path")
    if not _same_path(claimed_path, history_path):
        raise DriverError("v6 ambient history manifest path mismatch")
    claimed_sha = _sha_claim(value.get("sha256"), "v6 ambient history manifest SHA")
    if claimed_sha != history_sha256.upper():
        raise DriverError("v6 ambient history manifest SHA mismatch")


def _validate_existing_ambient_history(root: Path, history_path: Path) -> Path:
    """Validate retry inputs before the v5 oracle is allowed to append.

    A history is appendable only when its v6 hash sidecar and manifest are
    both present and mutually consistent.  This makes a retry fail closed on
    truncation, sidecar loss, manifest tampering, and stale artifacts instead
    of silently appending to an unsealed stream.
    """

    root = require_e_root(root)
    history_path = candidate_path(history_path, root, "v6 ambient history")
    sidecar = _sidecar_path(history_path, root)
    manifest = candidate_path(Path(str(history_path) + ".manifest"), root, "v6 ambient history manifest")
    manifest_sidecar = _sidecar_path(manifest, root)
    _reject_reparse_components(history_path, root, "v6 ambient history")
    _reject_reparse_components(sidecar, root, "v6 ambient history sidecar")
    _reject_reparse_components(manifest, root, "v6 ambient history manifest")
    _reject_reparse_components(manifest_sidecar, root, "v6 ambient history manifest sidecar")
    if not history_path.exists():
        if sidecar.exists():
            raise DriverError("v6 ambient history hash sidecar exists without its history")
        if manifest.exists() or manifest_sidecar.exists():
            raise DriverError("v6 ambient history manifest exists without its history")
        return history_path
    require_regular_candidate_file(history_path, root, "v6 ambient history")
    if not sidecar.exists():
        raise DriverError("v6 ambient history hash sidecar is missing")
    history_sha = validate_sidecar(history_path, root, "v6 ambient history")
    if not manifest.exists():
        raise DriverError("v6 ambient history manifest is missing")
    if not manifest_sidecar.exists():
        raise DriverError("v6 ambient history manifest sidecar is missing")
    _validate_ambient_history_manifest(root, history_path, history_sha)
    return history_path


def _seal_ambient_history(root: Path, history_path: Path) -> str | None:
    """Atomically refresh the history hash sidecar and seal manifest."""

    root = require_e_root(root)
    history_path = candidate_path(history_path, root, "v6 ambient history")
    if not history_path.exists():
        return None
    history_path = require_regular_candidate_file(history_path, root, "v6 ambient history")
    history_sha = digest(history_path)
    sidecar = _sidecar_path(history_path, root)
    manifest = candidate_path(
        Path(str(history_path) + ".manifest"),
        root,
        "v6 ambient history manifest",
    )
    _reject_reparse_components(sidecar, root, "v6 ambient history sidecar")
    _reject_reparse_components(manifest, root, "v6 ambient history manifest")
    _atomic_bytes(sidecar, f"{history_sha}  {history_path.name}\n".encode("ascii"), root, replace=True)
    atomic_json(
        manifest,
        {"schema": "B07H_RUNTIME_DRIVER_DEFERRED_SIDECAR_V1", "path": str(history_path), "sha256": history_sha},
        root,
        replace=True,
    )
    _validate_ambient_history_manifest(root, history_path, history_sha)
    return history_sha


def _load_ambient_oracle(root: Path, metadata: Mapping[str, Any], workspace_root: Path) -> tuple[Any, dict[str, Any]]:
    """Load only the already verified, hash-pinned v5 production module."""

    checked = _validate_ambient_oracle(metadata, root)
    oracle_path = Path(checked["path"])
    module_name = "_b07h_ambient_oracle_" + hashlib.sha256(str(oracle_path).encode("utf-8")).hexdigest()[:16]
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, oracle_path)
        if spec is None or spec.loader is None:
            raise DriverError("v6 ambient oracle module cannot be loaded")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException as error:
            sys.modules.pop(module_name, None)
            raise DriverError(f"v6 ambient oracle import failed: {type(error).__name__}: {error}") from error
    settle = getattr(module, "settle_ambient", None)
    if not callable(settle):
        raise DriverError("v6 ambient oracle has no production settle_ambient")
    # The v5 production oracle remains authoritative for CPU/SearchIndexer,
    # GPU, target/WSL, free-space, timeout, and five-consecutive semantics.
    # Only its C-workspace probe is replaced with v6's recursive fail-closed
    # implementation.  The closure also makes deterministic tests possible
    # without changing the pinned oracle bytes.
    module.c_workspace_state = lambda *_args, **_kwargs: workspace_state(workspace_root)
    return module, checked


def settle_ambient(
    root: Path,
    history_path: Path,
    cells: Sequence[str],
    *,
    ambient_oracle: Mapping[str, Any],
    workspace_root: Path | str = DEFAULT_C_WORKSPACE,
    timeout_seconds: float = 7200.0,
    sample_seconds: float = 2.0,
    consecutive_samples: int = DEFAULT_CONSECUTIVE_SAMPLES,
    process_sampler: Any = None,
    wsl_sampler: Any = None,
    gpu_sampler: Any = None,
    free_bytes_fn: Any = None,
) -> dict[str, Any]:
    """Delegate the ambient gate to the verified v5 production oracle."""

    root = require_e_root(root)
    history_path = _validate_existing_ambient_history(root, history_path)
    workspace_root = _resolved(workspace_root)
    oracle, checked = _load_ambient_oracle(root, ambient_oracle, workspace_root)
    kwargs: dict[str, Any] = {}
    if process_sampler is not None:
        kwargs["process_sampler"] = process_sampler
    if wsl_sampler is not None:
        kwargs["wsl_sampler"] = wsl_sampler
    if gpu_sampler is not None:
        kwargs["gpu_sampler"] = gpu_sampler
    if free_bytes_fn is not None:
        kwargs["free_bytes_fn"] = free_bytes_fn
    try:
        result = oracle.settle_ambient(
            root,
            history_path,
            workspace_root=workspace_root,
            timeout_seconds=timeout_seconds,
            sample_seconds=sample_seconds,
            consecutive_samples=consecutive_samples,
            **kwargs,
        )
    finally:
        # The oracle owns sampling/append semantics; v6 owns the final seal
        # even when the oracle raises part-way through an attempt.
        if history_path.exists():
            _seal_ambient_history(root, history_path)
    if not isinstance(result, Mapping):
        raise DriverError("v6 ambient oracle returned a malformed result")
    return {**dict(result), "ambient_oracle": checked}


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
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--ambient-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--sample-seconds", type=float, default=2.0)
    parser.add_argument("--consecutive-samples", type=int, default=DEFAULT_CONSECUTIVE_SAMPLES)
    args = parser.parse_args(argv)
    root = require_e_root(args.candidate_root)
    if not 1 <= args.invocation_index <= TOTAL_INVOCATIONS:
        raise DriverError("invocation-index must be between 1 and 6")
    runset_path = candidate_path(args.runset, root, "v6 runset")
    runset = validate_runset(runset_path, root, args.expected_runset_sha256)
    invocation = runset["invocations"][args.invocation_index - 1]
    cells = list(invocation["result_cells"])
    result_path = candidate_path(args.result or Path("logs") / f"B07H_v3_invocation_{args.invocation_index:02d}.json", root, "v6 result")
    history_path = candidate_path(args.history or Path("logs") / f"B07H_v3_invocation_{args.invocation_index:02d}_ambient.jsonl", root, "v6 history")
    driver_log = candidate_path(args.driver_log or Path("logs") / f"B07H_v3_invocation_{args.invocation_index:02d}.log", root, "v6 driver log")
    ledger_path = candidate_path("logs/B07H_v3_ledger.json", root, "v6 ledger")
    output_dir = candidate_path(invocation["output"], root, "v6 invocation output")
    if output_dir.exists():
        raise DriverError(f"v6 invocation output already exists; refusing stale evidence: {output_dir}")
    if result_path.exists() or _sidecar_path(result_path, root).exists():
        raise DriverError(f"v6 invocation result already exists: {result_path}")
    ledger = read_ledger(ledger_path, root)
    expected_index = len(ledger["results"]) + 1
    if args.invocation_index != expected_index:
        raise DriverError(f"v6 ledger requires strict serial invocation {expected_index}, got {args.invocation_index}")
    if set(cells).intersection(ledger.get("cells", {})):
        raise DriverError(f"v6 invocation cells are already present in the ledger: {cells}")
    require_c_workspace_clean()
    if args.validation_only:
        print(json.dumps({"schema": "B07H_RUNTIME_DRIVER_VALIDATION_V3", "status": "validation_only", "passed": True, "invocation_index": args.invocation_index, "result_cells": cells, "candidate_root": str(root), "runset_sha256": args.expected_runset_sha256}, sort_keys=True))
        return 0
    ambient = settle_ambient(
        root,
        history_path,
        cells,
        ambient_oracle=runset["ambient_oracle"],
        timeout_seconds=args.ambient_timeout_seconds,
        sample_seconds=args.sample_seconds,
        consecutive_samples=args.consecutive_samples,
    )
    if ambient["reason"] != "settled":
        append_deferred(root, history_path, {"invocation": invocation["id"], "deferred_cells": cells, "reason": "ambient timeout", "ambient": ambient})
        print(history_path)
        return 4
    env, locations = build_runtime_environment(root, args.invocation_index, args.runtime_temp)
    if driver_log.exists():
        raise FileExistsError(f"refusing to overwrite v6 driver log: {driver_log}")
    driver_log.parent.mkdir(parents=True, exist_ok=True)
    returncode: int | None = None
    launch_error: str | None = None
    with driver_log.open("x", encoding="utf-8") as stream:
        try:
            process = subprocess.Popen([str(item) for item in invocation["command"]], cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
            returncode = process.wait()
        except Exception as error:
            launch_error = f"child launch exception: {type(error).__name__}: {error}"
    # Child manifests are intentionally only consumed from the candidate output path.
    output_dir = candidate_path(invocation["output"], root, "v6 invocation output")
    manifest_candidate = _lexical_candidate(output_dir / "manifest.json", root, "v6 child manifest")
    manifest_error: str | None = None
    manifest_path = manifest_candidate
    # Check the complete path chain before existence/read/hash.  A regular
    # file reached through a symlinked output directory is not candidate-owned
    # evidence even when the final directory entry itself is not a symlink.
    _reject_reparse_components(manifest_candidate, root, "v6 child manifest")
    if _is_reparse(manifest_candidate):
        raise DriverError(f"v6 child manifest is a symlink/reparse point: {manifest_candidate}")
    if manifest_candidate.exists():
        manifest_path = require_regular_candidate_file(manifest_candidate, root, "v6 child manifest")
        try:
            manifest = read_json(manifest_path)
        except Exception as error:
            manifest = {}
            manifest_error = f"child manifest parse exception: {type(error).__name__}: {error}"
    else:
        manifest = {}
    reject_gt(manifest, "v6 child manifest")
    raw_status = "failure" if launch_error or manifest_error else manifest.get("status")
    reason_hint = launch_error or manifest_error or manifest.get("reason")
    status, reason = normalize_terminal_status(raw_status, returncode, reason_hint)
    cell_results = _dnf_cells(cells, reason or "terminal success", status) if status == "dnf" else [{"id": cell, "status": "success"} for cell in cells]
    payload = {"schema": RESULT_SCHEMA, "status": status, "mapping_started": True, "invocation_index": args.invocation_index, "invocation": invocation["id"], "engine": invocation["engine"], "sequence": invocation["sequence"], "result_cells": cells, "cell_results": cell_results, "runset_sha256": args.expected_runset_sha256.upper(), "source_sha256": EXPECTED_SOURCE_SHA256, "protocol_sha256": EXPECTED_PROTOCOL_SHA256, "gt_opened": False, "ground_truth_read": False, "ground_truth_materialized": False, "ground_truth_argument_present_anywhere": False, "manifest": {"path": str(manifest_path), "sha256": digest(manifest_path) if manifest_path.is_file() else None}, "runtime_environment": locations, "finished_utc": utc_now()}
    record_result(root, result_path, payload)
    print(result_path)
    return 0 if status == "success" else 2


__all__ = ["ALLOWED_E_ROOTS", "DEFAULT_C_WORKSPACE", "DRIVER_VERSION", "RESULT_SCHEMA", "DEFERRED_SCHEMA", "LEDGER_SCHEMA", "RESULT_CELLS", "INVOCATION_CELLS", "STOP_FREE_BYTES", "CPU_SETTLE_LIMIT_PERCENT", "SEARCH_INDEXER_SETTLE_LIMIT_PERCENT", "GPU_MEMORY_GROWTH_TOLERANCE_MIB", "DEFAULT_CONSECUTIVE_SAMPLES", "DriverError", "candidate_path", "require_e_root", "require_allowed_e_path", "require_regular_candidate_file", "require_c_readonly_file", "reject_gt", "validate_runset", "validate_runset_value", "build_runtime_environment", "workspace_state", "c_workspace_state", "require_workspace_clean", "require_c_workspace_clean", "normalize_terminal_status", "record_result", "append_deferred", "settle_ambient", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
