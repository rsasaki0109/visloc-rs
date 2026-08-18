#!/usr/bin/env python3
"""Safely drive the frozen B07-H six-invocation runtime runset.

This controller is deliberately separate from the frozen runners and runset.
It owns only preflight, the ambient-settle barrier, E-only runtime locations,
and failure-inclusive result accounting.  The command stored in the runset is
passed to the child process byte-for-byte (as a list of argv tokens).

The driver is safe to invoke from the workspace when ``--candidate-root`` is
the archived E: candidate, or from a copied driver below that candidate.  A
validation-only invocation never starts a child mapper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


EXPECTED_RUNSET_SHA256 = (
    "99265390BFA15C4086FAC91713F46C24F3EE2B7838F1525CADD461E8CF00BD4F"
)
EXPECTED_SOURCE_SHA256 = (
    "38A704369AF7EC4898307D2EA61016260834DA7CFD452CB17B51FBAD621CCA8D"
)
EXPECTED_RUNSET_SCHEMA = "B07H_GT_FREE_RUNTIME_RUNSET_V1"
STOP_FREE_BYTES = 250 * 1024**3
TOTAL_INVOCATIONS = 6
TOTAL_RESULT_CELLS = 9
DEFERRED_DIAGNOSTIC_RELATIVE_PATH = Path("logs") / "B07H_v2_deferred.jsonl"

# These checks describe operator-controlled resources rather than an invalid
# runset/input.  A failed resource check must leave the invocation's result
# cells available for a later retry; all other failed preflight checks remain
# terminal and are recorded as failure-inclusive DNF rows.
DEFERRED_PREFLIGHT_CHECKS = frozenset(
    {
        "e_free_threshold",
        "c_workspace_target_absent",
        "c_workspace_temp_absent",
    }
)

# The order is a protocol value, not a convenience sort.  Never derive this
# list from filesystem order or from a user-provided invocation id.
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

TARGET_PROCESS_NAMES = {
    "cargo.exe",
    "rustc.exe",
    "colmap.exe",
    "sequential_sfm_demo.exe",
}
# WSL exposes Linux executable names without the Windows ``.exe`` suffix.
# Keep this list separate from TARGET_PROCESS_NAMES so the Windows process
# query remains byte-for-byte compatible with the frozen driver contract.
WSL_TARGET_PROCESS_NAMES = frozenset(
    {
        "cargo",
        "rustc",
        "colmap",
        "sequential_sfm_demo",
    }
)
WSL_QUERY_TIMEOUT_SECONDS = 30
# A resident CUDA service may reserve a few GiB while doing no work.  The
# settle gate therefore compares later samples with the first valid idle
# sample, rather than imposing an absolute memory ceiling.  64 MiB is large
# enough for ordinary allocator/reporting jitter but small enough to reject a
# newly started GPU workload.  Utilization must still be exactly zero.
GPU_MEMORY_GROWTH_TOLERANCE_MIB = 64.0
GT_TOKENS = (
    "ground-truth",
    "ground_truth",
    "groundtruth",
    "ground truth",
    "state_groundtruth",
    "gt_path",
    "groundtruth_estimate",
)

# Every variable which is known to be used as a temporary/cache/config
# location is overwritten.  Keeping this explicit prevents a user profile on
# C: from silently becoming part of a supposedly E-only run.
E_ONLY_ENV_SUFFIXES: Mapping[str, str] = {
    "TEMP": "tmp",
    "TMP": "tmp",
    "TMPDIR": "tmp",
    "PYTHONPYCACHEPREFIX": "pycache",
    "CARGO_TARGET_DIR": "cargo-target",
    "CARGO_HOME": ".cargo",
    "RUSTUP_HOME": "temp/rustup",
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
    "CUDA_CACHE_MAXSIZE": "2147483648",
    "TRITON_CACHE_DIR": "triton",
    "TORCH_EXTENSIONS_DIR": "torch-extensions",
    "RAY_TMPDIR": "ray",
    "JOBLIB_TEMP_FOLDER": "joblib",
    "NVDIFRAST_CACHE_DIR": "nvdiffrast",
}

DEFAULT_C_WORKSPACE = Path(r"C:\Users\rsasa\Workspace\visloc-rs")


class DriverError(ValueError, RuntimeError):
    """A deterministic preflight or accounting error."""


@dataclass(frozen=True)
class InvocationContext:
    candidate_root: Path
    runset_path: Path
    runset: dict[str, Any]
    invocation_index: int
    invocation: dict[str, Any]
    command: list[str]
    out_dir: Path
    prepared_dir: Path
    prepared_manifest_path: Path
    prepared_manifest: dict[str, Any]
    protocol_path: Path
    source_path: Path

    @property
    def result_cells(self) -> list[str]:
        return [str(cell) for cell in self.invocation["result_cells"]]

    @property
    def id(self) -> str:
        return str(self.invocation["id"])


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    """Return an uppercase SHA-256 digest for a regular file."""

    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest().upper()


# Compatibility alias used by a few existing benchmark scripts.
sha256 = digest


def json_no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DriverError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def read_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=json_no_duplicate_keys
    )
    if not isinstance(parsed, dict):
        raise DriverError(f"JSON object expected: {path}")
    return parsed


def resolve_root(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def candidate_path(value: str | Path, candidate_root: Path | None = None) -> Path:
    """Resolve a path and reject values which escape the E candidate.

    ``candidate_root`` is optional for compatibility with the old H1a helper;
    callers in this driver always pass it explicitly.
    """

    root = resolve_root(candidate_root or Path(__file__).resolve().parents[1])
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise DriverError(f"path escapes B07 candidate: {path}") from error
    return path


def _external_or_candidate_path(value: str | Path, candidate_root: Path) -> Path:
    """Resolve a frozen tool path while allowing fixed external tools.

    Inputs, outputs, protocols and scripts use ``candidate_path``.  Only the
    fixed Python/Colmap executables may live outside the candidate.
    """

    path = Path(value)
    if not path.is_absolute():
        return candidate_path(path, candidate_root)
    return path.resolve(strict=False)


def command_contains_gt(command: Iterable[str]) -> bool:
    for raw in command:
        normalized = str(raw).lower().replace("/", "\\")
        if any(token in normalized for token in GT_TOKENS):
            return True
    return False


def _option_values(command: Sequence[str], option: str) -> list[str]:
    values: list[str] = []
    for index, token in enumerate(command[:-1]):
        if token == option:
            values.append(command[index + 1])
    return values


def _assert_candidate_option_paths(
    command: Sequence[str], candidate_root: Path, options: Sequence[str]
) -> None:
    for option in options:
        values = _option_values(command, option)
        if len(values) != 1:
            raise DriverError(f"frozen command must contain exactly one {option}")
        candidate_path(values[0], candidate_root)


def _require_digest(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = digest(path)
    if actual.upper() != str(expected).upper():
        raise DriverError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")


def _validate_fixed_tools(runset: Mapping[str, Any], candidate_root: Path) -> None:
    fixed = runset.get("fixed_tools")
    if not isinstance(fixed, dict):
        raise DriverError("runset.fixed_tools is missing")
    required = (
        "python",
        "hierarchical_runner",
        "hierarchical_executable",
        "colmap_runner",
        "colmap",
    )
    for name in required:
        meta = fixed.get(name)
        if not isinstance(meta, dict) or not meta.get("path"):
            raise DriverError(f"fixed tool metadata is missing: {name}")
        path = _external_or_candidate_path(str(meta["path"]), candidate_root)
        if not path.is_file():
            raise FileNotFoundError(f"fixed tool is missing: {path}")
        expected = meta.get("sha256")
        if expected:
            _require_digest(path, str(expected), f"fixed tool {name}")


def _validate_invocation_contract(
    invocation: Mapping[str, Any], candidate_root: Path, expected_index: int
) -> None:
    if invocation.get("id") != SERIAL_ORDER[expected_index - 1]:
        raise DriverError(
            f"invocation {expected_index} is not in fixed serial order: {invocation.get('id')}"
        )
    command_value = invocation.get("command")
    if not isinstance(command_value, list) or not command_value or not all(
        isinstance(token, str) for token in command_value
    ):
        raise DriverError(f"invocation {expected_index} has no frozen command list")
    if command_contains_gt(command_value):
        raise DriverError("frozen command contains a GT token/path")
    if invocation.get("ground_truth_argument_present") is not False:
        raise DriverError("runset marks a ground-truth argument as present")
    result_cells = invocation.get("result_cells")
    if not isinstance(result_cells, list) or not result_cells or not all(
        isinstance(cell, str) for cell in result_cells
    ):
        raise DriverError(f"invocation {expected_index} has invalid result_cells")
    # This check binds the cell names as well as their order.  In particular,
    # COLMAP must contribute exactly the incremental and global cells.
    expected_cells = {
        "visloc_MH_01_easy": ["visloc_MH_01_easy"],
        "colmap_MH_01_easy": [
            "colmap_inc_MH_01_easy",
            "colmap_global_MH_01_easy",
        ],
        "visloc_MH_03_medium": ["visloc_MH_03_medium"],
        "colmap_MH_03_medium": [
            "colmap_inc_MH_03_medium",
            "colmap_global_MH_03_medium",
        ],
        "visloc_MH_05_difficult": ["visloc_MH_05_difficult"],
        "colmap_MH_05_difficult": [
            "colmap_inc_MH_05_difficult",
            "colmap_global_MH_05_difficult",
        ],
    }[str(invocation["id"])]
    if result_cells != expected_cells:
        raise DriverError(
            f"result_cells for {invocation['id']} do not match frozen cells"
        )
    _assert_candidate_option_paths(
        command_value,
        candidate_root,
        (
            "--features-dir",
            "--timestamps",
            "--out-dir",
            "--exe",
        )
        if invocation.get("engine") == "visloc"
        else ("--prepared-dir", "--out-dir", "--protocol"),
    )


def validate_runset(
    runset_path: Path,
    candidate_root: Path,
    *,
    expected_runset_sha256: str = EXPECTED_RUNSET_SHA256,
) -> dict[str, Any]:
    """Validate the immutable runset and return its decoded object."""

    root = resolve_root(candidate_root)
    path = candidate_path(runset_path, root)
    _require_digest(path, expected_runset_sha256, "B07H runset")
    runset = read_json(path)
    if runset.get("schema") != EXPECTED_RUNSET_SCHEMA:
        raise DriverError("unexpected B07H runset schema")
    if runset.get("candidate_root"):
        declared_root = resolve_root(Path(str(runset["candidate_root"])))
        if declared_root != root:
            raise DriverError(
                f"runset candidate_root mismatch: {declared_root} != {root}"
            )
    invocations = runset.get("invocations")
    if not isinstance(invocations, list) or len(invocations) != TOTAL_INVOCATIONS:
        raise DriverError("frozen runset must contain exactly six invocations")
    serial = runset.get("serial_order")
    if serial != [
        "MH_01_easy visloc",
        "MH_01_easy colmap (incremental + global cells)",
        "MH_03_medium visloc",
        "MH_03_medium colmap (incremental + global cells)",
        "MH_05_difficult visloc",
        "MH_05_difficult colmap (incremental + global cells)",
    ]:
        raise DriverError("runset serial_order is not the frozen B07-H order")
    for index, invocation in enumerate(invocations, 1):
        if not isinstance(invocation, dict):
            raise DriverError(f"invocation {index} is not an object")
        _validate_invocation_contract(invocation, root, index)
    invocation_ids = [str(item["id"]) for item in invocations]
    if invocation_ids != list(SERIAL_ORDER):
        raise DriverError("runset invocation ids are not in fixed serial order")
    cells = [str(cell) for item in invocations for cell in item["result_cells"]]
    if len(cells) != TOTAL_RESULT_CELLS or len(set(cells)) != TOTAL_RESULT_CELLS:
        raise DriverError("runset result cells are not a unique nine-cell denominator")
    if cells != list(RESULT_CELLS):
        raise DriverError("runset result cells do not match the frozen cell order")
    storage = runset.get("storage_gate", {})
    if storage.get("stop_threshold_bytes") != STOP_FREE_BYTES:
        raise DriverError("runset storage stop threshold is not 250 GiB")
    runtime_policy = runset.get("runtime_policy", {})
    if runtime_policy.get("serial_only") is not True:
        raise DriverError("runset does not enforce serial-only execution")
    if runtime_policy.get("total_invocations") != TOTAL_INVOCATIONS:
        raise DriverError("runset invocation denominator changed")
    if runtime_policy.get("total_result_cells") != TOTAL_RESULT_CELLS:
        raise DriverError("runset result-cell denominator changed")
    if runtime_policy.get("ground_truth_argument_present_anywhere") is not False:
        raise DriverError("runset is not globally GT-free")
    protocol_meta = runset.get("protocol")
    source_meta = runset.get("source")
    if not isinstance(protocol_meta, dict) or not isinstance(source_meta, dict):
        raise DriverError("runset protocol/source metadata is missing")
    protocol_path = candidate_path(str(protocol_meta["path"]), root)
    _require_digest(protocol_path, str(protocol_meta["sha256"]), "B07H protocol")
    source_path = candidate_path(str(source_meta["path"]), root)
    _require_digest(source_path, EXPECTED_SOURCE_SHA256, "B07H source")
    if str(source_meta.get("sha256", "")).upper() != EXPECTED_SOURCE_SHA256:
        raise DriverError("runset source SHA is not the frozen source SHA")
    _validate_fixed_tools(runset, root)
    # All generated result locations are frozen candidate-relative paths.
    for invocation in invocations:
        candidate_path(str(invocation["output"]), root)
    return runset


def _prepared_entry(runset: Mapping[str, Any], sequence: str) -> dict[str, Any]:
    for item in runset.get("prepared_inputs", []):
        if isinstance(item, dict) and item.get("sequence") == sequence:
            return item
    raise DriverError(f"prepared input is missing for {sequence}")


def load_invocation_context(
    runset_path: Path, invocation_index: int, candidate_root: Path
) -> InvocationContext:
    """Validate immutable artifacts for one positional invocation."""

    if invocation_index < 1 or invocation_index > TOTAL_INVOCATIONS:
        raise DriverError("invocation-index must be between 1 and 6")
    root = resolve_root(candidate_root)
    runset_path = candidate_path(runset_path, root)
    runset = validate_runset(runset_path, root)
    invocation = dict(runset["invocations"][invocation_index - 1])
    command = [str(token) for token in invocation["command"]]
    prepared = _prepared_entry(runset, str(invocation["sequence"]))
    prepared_dir = candidate_path(str(prepared["prepared_dir"]), root)
    prepared_manifest_path = prepared_dir / "manifest.json"
    _require_digest(
        prepared_manifest_path,
        str(prepared["manifest_sha256"]),
        f"prepared manifest {invocation['sequence']}",
    )
    prepared_manifest = read_json(prepared_manifest_path)
    if prepared_manifest.get("sequence") != invocation["sequence"]:
        raise DriverError("prepared manifest sequence mismatch")
    if prepared_manifest.get("ground_truth_read") is not False:
        raise DriverError("prepared manifest is not GT-free")
    if str(prepared_manifest.get("protocol_sha256", "")).upper() != str(
        runset["protocol"]["sha256"]
    ).upper():
        raise DriverError("prepared manifest protocol SHA mismatch")
    expected_frames = int(prepared.get("expected_frames", 0))
    if int(prepared_manifest.get("expected_frames", -1)) != expected_frames:
        raise DriverError("prepared manifest frame count mismatch")
    out_dir = candidate_path(str(invocation["output"]), root)
    protocol_path = candidate_path(str(runset["protocol"]["path"]), root)
    source_path = candidate_path(str(runset["source"]["path"]), root)
    return InvocationContext(
        candidate_root=root,
        runset_path=runset_path,
        runset=runset,
        invocation_index=invocation_index,
        invocation=invocation,
        command=command,
        out_dir=out_dir,
        prepared_dir=prepared_dir,
        prepared_manifest_path=prepared_manifest_path,
        prepared_manifest=prepared_manifest,
        protocol_path=protocol_path,
        source_path=source_path,
    )


def _drive_free_bytes(candidate_root: Path) -> int:
    return int(shutil.disk_usage(candidate_root).free)


def c_workspace_state(workspace_root: Path = DEFAULT_C_WORKSPACE) -> dict[str, Any]:
    root = resolve_root(workspace_root)
    target = root / "target"
    temp = root / "temp"
    return {
        "workspace_root": str(root),
        "target": str(target),
        "temp": str(temp),
        "target_present": target.exists(),
        "temp_present": temp.exists(),
        "clean": not target.exists() and not temp.exists(),
    }


def _wsl_output_text(value: Any) -> str:
    """Decode WSL command output across the Windows console encodings."""

    if value is None:
        return ""
    if isinstance(value, bytes):
        raw = value
        # ``wsl.exe`` has emitted UTF-16LE for list commands on some Windows
        # builds.  Prefer a BOM/null-byte signal before the normal UTF-8 path.
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in raw:
            for encoding in ("utf-16", "utf-16-le", "utf-16-be"):
                try:
                    return raw.decode(encoding).replace("\x00", "")
                except UnicodeDecodeError:
                    continue
        for encoding in ("utf-8-sig", "utf-8", "oem", "mbcs"):
            try:
                return raw.decode(encoding)
            except (LookupError, UnicodeDecodeError):
                continue
        return raw.decode("utf-8", errors="replace")
    # Tests and ``subprocess.run(..., text=True)`` normally provide str.  A
    # null-byte cleanup also handles a UTF-16 stream decoded with a legacy
    # Windows code page before it reaches this function.
    return str(value).replace("\x00", "")


def _wsl_lines(value: Any) -> list[str]:
    return [
        line.strip().lstrip("\ufeff").strip()
        for line in _wsl_output_text(value).splitlines()
        if line.strip().lstrip("\ufeff").strip()
    ]


def _parse_running_wsl_distros(value: Any) -> list[str]:
    """Parse ``wsl --list --running --quiet`` without treating diagnostics as distros."""

    ignored_prefixes = (
        "name",
        "state",
        "version",
        "there are no",
        "no running",
        "no installed",
        "windows subsystem for linux",
        "the requested operation",
    )
    distros: list[str] = []
    for line in _wsl_lines(value):
        name = line.lstrip("*").strip()
        if not name or name.casefold().startswith(ignored_prefixes):
            continue
        if name not in distros:
            distros.append(name)
    return distros


def _parse_verbose_running_wsl_distros(value: Any) -> list[str]:
    """Parse running distro names from ``wsl --list --verbose`` output."""

    distros: list[str] = []
    for line in _wsl_lines(value):
        # The default distro is prefixed with ``*``.  A distro name may be
        # multi-word, so locate the state column rather than taking token 0.
        fields = line.lstrip("*").strip().split()
        if len(fields) < 2:
            continue
        state_index = next(
            (
                index
                for index, field in enumerate(fields[1:], 1)
                if field.casefold() == "running"
            ),
            None,
        )
        if state_index is None:
            continue
        name = " ".join(fields[:state_index]).strip()
        if name and name.casefold() not in {"name", "state", "version"} and name not in distros:
            distros.append(name)
    return distros


def _wsl_failure_text(completed: Any, fallback: str) -> str:
    stderr = _wsl_output_text(getattr(completed, "stderr", "")).strip()
    stdout = _wsl_output_text(getattr(completed, "stdout", "")).strip()
    return stderr or stdout or fallback


def _wsl_result(
    *,
    status: str,
    available: bool,
    running_distros: Sequence[str] = (),
    target_processes: Sequence[Mapping[str, Any]] = (),
    query_error: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "available": bool(available),
        "running_distros": [str(name) for name in running_distros],
        "target_processes": [dict(item) for item in target_processes],
        "query_error": query_error,
    }


def _wsl_process_name_matches(command: str, target_names: frozenset[str]) -> str | None:
    """Return a target executable basename found in a WSL ps command line."""

    # Matching complete path/argv components catches both ``comm`` and an
    # absolute executable path while avoiding names such as ``cargo-helper``.
    for token in re.split(r"[\s]+", command.strip()):
        token = token.strip("'\"(),[]")
        if not token:
            continue
        basename = token.replace("\\", "/").rsplit("/", 1)[-1].casefold()
        if basename.endswith(".exe"):
            basename = basename[:-4]
        if basename in target_names:
            return basename
    return None


def _parse_wsl_target_processes(value: Any, distro: str) -> list[dict[str, Any]]:
    """Parse ``ps -eo pid=,comm=,args=`` output for target executables."""

    processes: list[dict[str, Any]] = []
    for line in _wsl_lines(value):
        fields = line.split(None, 2)
        if len(fields) < 2:
            continue
        pid_text, comm = fields[0], fields[1]
        try:
            pid: int | None = int(pid_text)
        except ValueError:
            pid = None
        command = fields[2] if len(fields) > 2 else comm
        match = _wsl_process_name_matches(
            f"{comm} {command}", WSL_TARGET_PROCESS_NAMES
        )
        if match is None:
            continue
        processes.append(
            {
                "name": match,
                "pid": pid,
                "distro": distro,
                "command": command,
            }
        )
    return processes


def _wsl_unavailable(error: str | None = None) -> dict[str, Any]:
    return _wsl_result(
        status="unavailable",
        available=False,
        query_error=error,
    )


def wsl_process_sample() -> dict[str, Any]:
    """Read-only WSL process sample used by the ambient gate.

    The running-distro list is queried before entering any distro, so this
    helper never starts a stopped distro or executes a command in the default
    distro.  A process-list failure for a distro already reported as running
    is represented as a synthetic target process; the ambient gate therefore
    fails closed for that sample.  Missing WSL and an empty running-distro
    list are non-blocking states.
    """

    list_command = ["wsl.exe", "--list", "--running", "--quiet"]
    try:
        running = subprocess.run(
            list_command,
            capture_output=True,
            text=False,
            check=False,
            timeout=WSL_QUERY_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        return _wsl_unavailable("wsl.exe is unavailable")
    except (OSError, subprocess.TimeoutExpired) as error:
        # No running distro has been established, so an unavailable WSL
        # service must not turn into a permanent benchmark block.
        return _wsl_unavailable(f"WSL running-distro query failed: {error}")

    running_distros = _parse_running_wsl_distros(getattr(running, "stdout", ""))
    returncode = int(getattr(running, "returncode", 1))
    if returncode != 0 and not running_distros:
        # Older WSL builds return non-zero for an empty running list.  The
        # verbose query tells us whether that was an idle state or an error
        # while a distro was in fact running.
        try:
            verbose = subprocess.run(
                ["wsl.exe", "--list", "--verbose"],
                capture_output=True,
                text=False,
                check=False,
                timeout=WSL_QUERY_TIMEOUT_SECONDS,
            )
        except FileNotFoundError:
            return _wsl_unavailable("wsl.exe is unavailable")
        except (OSError, subprocess.TimeoutExpired) as error:
            return _wsl_unavailable(f"WSL distro-state query failed: {error}")
        running_distros = _parse_verbose_running_wsl_distros(
            getattr(verbose, "stdout", "")
        )
        if not running_distros:
            # A non-zero command with no running distro is either an idle WSL
            # service or an unavailable service.  Both are explicitly
            # non-blocking; retain diagnostics for the append-only history.
            failure = _wsl_failure_text(
                running, "WSL reports no running distributions"
            )
            if int(getattr(verbose, "returncode", 1)) == 0:
                return _wsl_result(status="idle", available=True)
            return _wsl_unavailable(failure)

    if not running_distros:
        return _wsl_result(status="idle", available=True)

    targets: list[dict[str, Any]] = []
    errors: list[str] = []
    for distro in running_distros:
        process_command = [
            "wsl.exe",
            "--distribution",
            distro,
            "--exec",
            "ps",
            "-eo",
            "pid=,comm=,args=",
        ]
        try:
            completed = subprocess.run(
                process_command,
                capture_output=True,
                text=False,
                check=False,
                timeout=WSL_QUERY_TIMEOUT_SECONDS,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as error:
            message = f"{distro}: WSL process query failed: {error}"
            errors.append(message)
            targets.append(
                {
                    "name": "wsl-query-error",
                    "pid": None,
                    "distro": distro,
                    "error": message,
                }
            )
            continue
        if int(getattr(completed, "returncode", 1)) != 0:
            message = f"{distro}: {_wsl_failure_text(completed, 'WSL process query failed')}"
            errors.append(message)
            targets.append(
                {
                    "name": "wsl-query-error",
                    "pid": None,
                    "distro": distro,
                    "error": message,
                }
            )
            continue
        targets.extend(_parse_wsl_target_processes(getattr(completed, "stdout", ""), distro))

    if errors:
        return _wsl_result(
            status="error",
            available=True,
            running_distros=running_distros,
            target_processes=targets,
            query_error="; ".join(errors),
        )
    return _wsl_result(
        status="running",
        available=True,
        running_distros=running_distros,
        target_processes=targets,
    )


def powershell_sample() -> dict[str, Any]:
    """Sample target processes, total CPU and SearchIndexer CPU on Windows."""

    names = sorted(TARGET_PROCESS_NAMES)
    script = rf"""
$names = @({','.join(repr(name) for name in names)})
$procs = @(Get-CimInstance Win32_Process | Where-Object {{ $names -contains $_.Name.ToLowerInvariant() }} | ForEach-Object {{ [pscustomobject]@{{name=$_.Name;pid=$_.ProcessId}} }})
try {{ $total = [double]((Get-Counter '\Processor(_Total)\% Processor Time' -ErrorAction Stop).CounterSamples[0].CookedValue) }} catch {{ $total = $null }}
try {{ $search = [double]((Get-Counter '\Process(SearchIndexer*)\% Processor Time' -ErrorAction Stop).CounterSamples | Measure-Object CookedValue -Sum).Sum }} catch {{ $search = 0.0 }}
[pscustomobject]@{{target_processes=$procs;total_processor_percent=$total;search_indexer_percent=$search}} | ConvertTo-Json -Compress
"""
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        raise DriverError(f"PowerShell ambient sample failed: {completed.stderr.strip()}")
    text = completed.stdout.strip()
    if not text:
        raise DriverError("PowerShell ambient sample was empty")
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise DriverError("PowerShell ambient sample was not an object")
    # Keep the Windows and WSL observations together in each ambient sample.
    # A WSL process-list error is represented by wsl_process_sample as a
    # synthetic target, so merging it here makes the existing target gate fail
    # closed without changing the frozen Windows process query.
    wsl = wsl_process_sample()
    windows_targets = parsed.get("target_processes", [])
    if not isinstance(windows_targets, list):
        windows_targets = [windows_targets]
    parsed["wsl"] = wsl
    parsed["target_processes"] = windows_targets + list(wsl["target_processes"])
    return parsed


def gpu_sample() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        # No driver/tool, or a timed-out query, is an unavailable sample.  It
        # must fail the ambient gate and remain retryable rather than escaping
        # as a terminal driver error.
        return {
            "available": False,
            "utilization_percent": None,
            "memory_used_mib": None,
        }
    if completed.returncode != 0 or not completed.stdout.strip():
        return {
            "available": False,
            "utilization_percent": None,
            "memory_used_mib": None,
        }
    values: list[tuple[float, float]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2:
            continue
        try:
            values.append((float(fields[0]), float(fields[1])))
        except ValueError:
            continue
    if not values:
        return {
            "available": False,
            "utilization_percent": None,
            "memory_used_mib": None,
        }
    return {
        "available": True,
        "utilization_percent": max(value[0] for value in values),
        "memory_used_mib": max(value[1] for value in values),
    }


def _finite_nonnegative(value: Any) -> float | None:
    """Return a finite, non-negative numeric GPU field or ``None``."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0.0:
        return None
    return parsed


def _gpu_observation(
    gpu: Any,
    baseline_memory_mib: float | None,
    *,
    baseline_sample: bool = False,
) -> dict[str, Any]:
    """Normalize one GPU observation and evaluate it against an attempt baseline.

    ``nvidia-smi`` is an external probe, so every field is treated as
    untrusted input.  A missing/unparseable value is represented in evidence
    and fails closed instead of escaping as an exception from the settle loop.
    The baseline is intentionally supplied by ``settle_ambient`` and is scoped
    to one settle attempt; an old append-only history can never silently
    become the new baseline.
    """

    available = isinstance(gpu, Mapping) and gpu.get("available") is True
    utilization = _finite_nonnegative(
        gpu.get("utilization_percent") if isinstance(gpu, Mapping) else None
    )
    memory = _finite_nonnegative(
        gpu.get("memory_used_mib") if isinstance(gpu, Mapping) else None
    )
    baseline = _finite_nonnegative(baseline_memory_mib)
    valid = available and utilization is not None and memory is not None
    idle = valid and utilization == 0.0
    growth = None if baseline is None or memory is None else memory - baseline
    growth_within_tolerance = bool(
        valid
        and baseline is not None
        and growth is not None
        and growth <= GPU_MEMORY_GROWTH_TOLERANCE_MIB
    )
    if not available:
        reason = "unavailable"
    elif not valid:
        reason = "unparseable"
    elif not idle:
        reason = "nonzero_utilization"
    elif baseline is None:
        reason = "baseline_pending"
    elif not growth_within_tolerance:
        reason = "memory_growth"
    elif baseline_sample:
        reason = "baseline_established"
    else:
        reason = "clean"
    settled = bool(idle and growth_within_tolerance and not baseline_sample)
    return {
        "available": available,
        "valid": valid,
        "utilization_percent": utilization,
        "memory_used_mib": memory,
        "baseline_memory_used_mib": baseline,
        "memory_growth_mib": growth,
        "memory_growth_tolerance_mib": GPU_MEMORY_GROWTH_TOLERANCE_MIB,
        "memory_within_tolerance": growth_within_tolerance,
        "idle": idle,
        "settled": settled,
        "baseline_sample": baseline_sample,
        "reason": reason,
    }


def _with_gpu_evidence(
    sample: Mapping[str, Any],
    baseline_memory_mib: float | None,
    *,
    baseline_sample: bool = False,
) -> dict[str, Any]:
    """Attach normalized GPU baseline evidence and refresh the clean checks."""

    result = dict(sample)
    observation = _gpu_observation(
        result.get("gpu"),
        baseline_memory_mib,
        baseline_sample=baseline_sample,
    )
    result["gpu_observation"] = observation
    result["gpu_baseline_memory_mib"] = observation["baseline_memory_used_mib"]
    result["gpu_observed_memory_mib"] = observation["memory_used_mib"]
    result["gpu_memory_growth_mib"] = observation["memory_growth_mib"]
    result["gpu_memory_growth_tolerance_mib"] = (
        GPU_MEMORY_GROWTH_TOLERANCE_MIB
    )
    checks = dict(result.get("checks", {}))
    checks["gpu_settled"] = bool(observation["settled"])
    result["checks"] = checks
    result["clean"] = all(checks.values())
    return result


def ambient_sample(
    candidate_root: Path | None = None,
    workspace_root: Path = DEFAULT_C_WORKSPACE,
    *,
    process_sampler: Callable[[], dict[str, Any]] = powershell_sample,
    wsl_sampler: Callable[[], dict[str, Any]] | None = None,
    gpu_sampler: Callable[[], dict[str, Any]] = gpu_sample,
    gpu_baseline_memory_mib: float | None = None,
    free_bytes_fn: Callable[[Path], int] = _drive_free_bytes,
) -> dict[str, Any]:
    root = resolve_root(candidate_root or Path(__file__).resolve().parents[1])
    process = process_sampler()
    # The production PowerShell sampler already includes its WSL observation.
    # Keeping an explicit sampler hook makes the WSL gate independently
    # testable and lets callers provide a deterministic read-only probe.
    wsl = wsl_sampler() if wsl_sampler is not None else process.get("wsl")
    gpu = gpu_sampler()
    free_bytes = int(free_bytes_fn(root))
    workspace = c_workspace_state(workspace_root)
    target_processes = process.get("target_processes", [])
    if not isinstance(target_processes, list):
        target_processes = [target_processes]
    if isinstance(wsl, Mapping):
        wsl_targets = wsl.get("target_processes", [])
        if not isinstance(wsl_targets, list):
            wsl_targets = [wsl_targets]
        # Avoid duplicating observations when a caller supplies both a
        # pre-combined process sample and an explicit WSL sampler.
        if wsl_sampler is not None:
            target_processes = target_processes + wsl_targets
    total = process.get("total_processor_percent")
    search = process.get("search_indexer_percent", 0.0)
    try:
        target_clear = not bool(target_processes)
        cpu_clear = total is not None and float(total) <= 15.0
        search_clear = search is not None and float(search) <= 10.0
    except (TypeError, ValueError):
        target_clear = cpu_clear = search_clear = False
    gpu_observation = _gpu_observation(gpu, gpu_baseline_memory_mib)
    gpu_clear = bool(gpu_observation["settled"])
    checks = {
        "target_processes_clear": target_clear,
        "cpu_settled": cpu_clear,
        "search_settled": search_clear,
        "gpu_settled": gpu_clear,
        "e_free_threshold": free_bytes >= STOP_FREE_BYTES,
        "c_workspace_clean": workspace["clean"],
    }
    return {
        "utc": utc_now(),
        "target_processes": target_processes,
        "wsl": wsl,
        "total_processor_percent": total,
        "search_indexer_percent": search,
        "gpu": gpu,
        "gpu_observation": gpu_observation,
        "gpu_baseline_memory_mib": gpu_observation["baseline_memory_used_mib"],
        "gpu_observed_memory_mib": gpu_observation["memory_used_mib"],
        "gpu_memory_growth_mib": gpu_observation["memory_growth_mib"],
        "gpu_memory_growth_tolerance_mib": GPU_MEMORY_GROWTH_TOLERANCE_MIB,
        "e_free_bytes": free_bytes,
        "c_workspace": workspace,
        "c_target_present": workspace["target_present"],
        "c_temp_present": workspace["temp_present"],
        "checks": checks,
        "clean": all(checks.values()),
    }


def _ambient_history_is_valid(path: Path, candidate_root: Path) -> bool:
    """Return whether an existing ambient history can be appended safely."""

    path = candidate_path(path, candidate_root)
    if not path.is_file():
        return False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return False
        for line in lines:
            value = json.loads(line)
            if not isinstance(value, dict):
                return False
            # Accept histories produced by both the original v2 driver (raw
            # ambient samples) and the attempt-bounded append-only format.
            if value.get("event") in {"attempt_start", "sample", "attempt_end"}:
                continue
            if "checks" in value and "clean" in value:
                continue
            return False
        return True
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def settle_ambient(
    candidate_root: Path,
    history_path: Path,
    workspace_root: Path = DEFAULT_C_WORKSPACE,
    *,
    timeout_seconds: float = 7200.0,
    sample_seconds: float = 2.0,
    consecutive_samples: int = 5,
    process_sampler: Callable[[], dict[str, Any]] = powershell_sample,
    wsl_sampler: Callable[[], dict[str, Any]] | None = None,
    gpu_sampler: Callable[[], dict[str, Any]] = gpu_sample,
    free_bytes_fn: Callable[[Path], int] = _drive_free_bytes,
) -> dict[str, Any]:
    """Wait for CPU/Search/GPU/storage/workspace to be clean for N samples.

    The first valid idle GPU sample establishes an attempt-local memory
    baseline and is evidence only; it never contributes to the consecutive
    clean count.  Every later sample must have zero utilization and stay no
    more than ``GPU_MEMORY_GROWTH_TOLERANCE_MIB`` above that baseline.  A
    growth violation resets the consecutive count while retaining the
    baseline, so a newly allocated workload cannot be accepted merely because
    it becomes stable at its higher reservation.
    """

    if timeout_seconds < 0:
        raise DriverError("ambient timeout must be non-negative")
    if sample_seconds < 0:
        raise DriverError("ambient sample interval must be non-negative")
    if consecutive_samples < 1:
        raise DriverError("consecutive-samples must be positive")
    history_path = candidate_path(history_path, candidate_root)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    append_history = history_path.exists()
    if append_history and not _ambient_history_is_valid(history_path, candidate_root):
        raise FileExistsError(f"refusing to overwrite ambient history: {history_path}")
    started = time.monotonic()
    samples = 0
    consecutive = 0
    gpu_baseline_memory_mib: float | None = None
    gpu_baseline_sample: int | None = None
    gpu_observations: list[dict[str, Any]] = []
    reason = "timeout"
    last: dict[str, Any] | None = None
    attempt_id = f"{datetime.now(timezone.utc).isoformat()}-{os.getpid()}"
    mode = "a" if append_history else "x"
    with history_path.open(mode, encoding="utf-8") as history:
        history.write(
            json.dumps(
                {
                    "event": "attempt_start",
                    "attempt_id": attempt_id,
                    "utc": utc_now(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        history.flush()
        error_text: str | None = None
        try:
            while True:
                last = ambient_sample(
                    candidate_root,
                    workspace_root,
                    process_sampler=process_sampler,
                    wsl_sampler=wsl_sampler,
                    gpu_sampler=gpu_sampler,
                    gpu_baseline_memory_mib=gpu_baseline_memory_mib,
                    free_bytes_fn=free_bytes_fn,
                )
                samples += 1
                baseline_sample = False
                if gpu_baseline_memory_mib is None:
                    candidate_observation = _gpu_observation(last.get("gpu"), None)
                    if candidate_observation["valid"] and candidate_observation["idle"]:
                        # The baseline is only established from a valid idle
                        # observation.  A busy, unavailable, or malformed
                        # first sample cannot bless a later reservation.
                        gpu_baseline_memory_mib = float(
                            candidate_observation["memory_used_mib"]
                        )
                        gpu_baseline_sample = samples
                        baseline_sample = True
                last = _with_gpu_evidence(
                    last,
                    gpu_baseline_memory_mib,
                    baseline_sample=baseline_sample,
                )
                observation = last["gpu_observation"]
                gpu_observations.append({"sample": samples, **observation})
                sample = {
                    **last,
                    "event": "sample",
                    "attempt_id": attempt_id,
                }
                history.write(json.dumps(sample, sort_keys=True) + "\n")
                history.flush()
                if last["clean"]:
                    consecutive += 1
                    if consecutive >= consecutive_samples:
                        reason = "settled"
                        break
                else:
                    consecutive = 0
                if time.monotonic() - started >= timeout_seconds:
                    break
                if sample_seconds:
                    time.sleep(sample_seconds)
        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"
            raise
        finally:
            end: dict[str, Any] = {
                "event": "attempt_end",
                "attempt_id": attempt_id,
                "utc": utc_now(),
                "reason": reason if error_text is None else "error",
                "samples": samples,
                "consecutive": consecutive,
            }
            if error_text is not None:
                end["error"] = error_text
            history.write(json.dumps(end, sort_keys=True) + "\n")
            history.flush()
    return {
        "reason": reason,
        "samples": samples,
        "consecutive": consecutive,
        "history": str(history_path),
        "last_sample": last,
        "attempt_id": attempt_id,
        "gpu_baseline_memory_mib": gpu_baseline_memory_mib,
        "gpu_baseline_sample": gpu_baseline_sample,
        "gpu_memory_growth_tolerance_mib": GPU_MEMORY_GROWTH_TOLERANCE_MIB,
        "gpu_observations": gpu_observations,
    }


def build_runtime_environment(candidate_root: Path, invocation_index: int) -> tuple[dict[str, str], dict[str, str]]:
    """Build an E-only child environment and return it with its path map."""

    if invocation_index < 1 or invocation_index > TOTAL_INVOCATIONS:
        raise DriverError("invocation-index must be between 1 and 6")
    root = resolve_root(candidate_root)
    base = root / "temp" / "runtime-b07h-v2" / f"invocation-{invocation_index:02d}"
    env = os.environ.copy()
    locations: dict[str, str] = {}
    for key, suffix in E_ONLY_ENV_SUFFIXES.items():
        # CUDA_CACHE_MAXSIZE is a numeric limit, not a path despite its
        # historical proximity to CUDA_CACHE_PATH.
        if key == "CUDA_CACHE_MAXSIZE":
            env[key] = suffix
            continue
        location = (base / suffix).resolve(strict=False)
        candidate_path(location, root)
        locations[key] = str(location)
        env[key] = str(location)
    env["PYTHONPATH"] = str((root / "scripts").resolve(strict=False))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["NO_PROXY"] = env.get("NO_PROXY", "")
    # Make all configured path values real before launch.  This operation is
    # intentionally delayed until after the ambient and storage gate.
    for location in locations.values():
        Path(location).mkdir(parents=True, exist_ok=True)
    return env, locations


def default_result_path(candidate_root: Path, context: InvocationContext) -> Path:
    safe_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in context.id)
    return candidate_path(
        Path("logs") / f"B07H_v2_invocation_{context.invocation_index:02d}_{safe_id}.json",
        candidate_root,
    )


def default_history_path(candidate_root: Path, context: InvocationContext) -> Path:
    safe_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in context.id)
    return candidate_path(
        Path("logs")
        / f"B07H_v2_invocation_{context.invocation_index:02d}_{safe_id}_ambient.jsonl",
        candidate_root,
    )


def default_driver_log_path(candidate_root: Path, context: InvocationContext) -> Path:
    safe_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in context.id)
    return candidate_path(
        Path("logs") / f"B07H_v2_invocation_{context.invocation_index:02d}_{safe_id}.log",
        candidate_root,
    )


def deferred_diagnostic_path(candidate_root: Path) -> Path:
    """Return the append-only E-candidate path for deferred attempts."""

    return candidate_path(DEFERRED_DIAGNOSTIC_RELATIVE_PATH, candidate_root)


def ledger_path(candidate_root: Path) -> Path:
    return candidate_path(Path("logs") / "B07H_v2_ledger.json", candidate_root)


def _empty_ledger() -> dict[str, Any]:
    return {
        "schema": "B07H_RUNTIME_DRIVER_LEDGER_V2",
        "total_result_cells": TOTAL_RESULT_CELLS,
        "expected_cells": list(RESULT_CELLS),
        "results": [],
        "cells": {},
        "updated_utc": utc_now(),
    }


def read_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_ledger()
    ledger = read_json(path)
    if ledger.get("schema") != "B07H_RUNTIME_DRIVER_LEDGER_V2":
        raise DriverError(f"unexpected B07H v2 ledger schema: {path}")
    if ledger.get("expected_cells") != list(RESULT_CELLS):
        raise DriverError("ledger denominator cells changed")
    cells = ledger.get("cells")
    if not isinstance(cells, dict):
        raise DriverError("ledger cells are not an object")
    if len(cells) != len(set(cells)) or not set(cells) <= set(RESULT_CELLS):
        raise DriverError("ledger has duplicate or unknown result cells")
    return ledger


def denominator(
    completed_cells: Iterable[str], expected_cells: Sequence[str] = RESULT_CELLS
) -> dict[str, Any]:
    """Compute a failure-inclusive denominator without silently de-duplicating."""

    expected = list(expected_cells)
    if len(expected) != len(set(expected)):
        raise DriverError("expected result cells contain duplicates")
    completed = list(completed_cells)
    if len(completed) != len(set(completed)):
        raise DriverError("completed result cells contain duplicates")
    unknown = sorted(set(completed) - set(expected))
    if unknown:
        raise DriverError(f"completed result cells are unknown: {unknown}")
    completed_ordered = [cell for cell in expected if cell in set(completed)]
    remaining = [cell for cell in expected if cell not in set(completed)]
    return {
        "total_cells": len(expected),
        "completed_cells": completed_ordered,
        "completed_count": len(completed_ordered),
        "remaining_cells": remaining,
        "remaining_count": len(remaining),
    }


def compute_remaining_cells(
    completed_cells: Iterable[str], expected_cells: Sequence[str] = RESULT_CELLS
) -> list[str]:
    return list(denominator(completed_cells, expected_cells)["remaining_cells"])


def _atomic_replace_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def write_new_json(path: Path, value: Mapping[str, Any], candidate_root: Path) -> None:
    """Write a result once; never overwrite an existing result/report."""

    path = candidate_path(path, candidate_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite result: {path}")
    # ``x`` gives the result artifact an overwrite-resistant creation step.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_deferred_diagnostic(
    candidate_root: Path,
    payload: Mapping[str, Any],
    path: Path | None = None,
) -> dict[str, Any]:
    """Append one prelaunch defer event without creating a result artifact.

    Deferred attempts are expected to be retried after an operator fixes the
    resource/ambient condition.  A JSONL diagnostic therefore records every
    attempt while leaving the result ledger and result path untouched.  The
    path is always candidate-relative (and thus E:-resident in production).
    """

    root = resolve_root(candidate_root)
    diagnostic_path = candidate_path(path or deferred_diagnostic_path(root), root)
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        **dict(payload),
        "schema": "B07H_RUNTIME_DRIVER_DEFERRED_V2",
        "deferred_diagnostic": str(diagnostic_path),
        "deferred_diagnostic_path": str(diagnostic_path),
    }
    # Append mode is intentional: a retry must never overwrite an earlier
    # operator/resource diagnosis, and each line remains independently valid
    # JSON if a later write is interrupted.
    with diagnostic_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return event


def record_result(
    candidate_root: Path,
    result_path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Append one invocation's cells to the E-only ledger exactly once."""

    root = resolve_root(candidate_root)
    result_path = candidate_path(result_path, root)
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite result: {result_path}")
    path = ledger_path(root)
    ledger = read_ledger(path)
    cell_results = payload.get("cell_results")
    if not isinstance(cell_results, list) or not cell_results:
        raise DriverError("terminal result has no cell_results")
    ids: list[str] = []
    for item in cell_results:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise DriverError("invalid cell result")
        cell_id = str(item["id"])
        ids.append(cell_id)
        if cell_id not in RESULT_CELLS:
            raise DriverError(f"unknown result cell: {cell_id}")
    if len(ids) != len(set(ids)):
        raise DriverError("one invocation reports a duplicate result cell")
    declared_ids = payload.get("result_cells")
    if declared_ids is not None and list(declared_ids) != ids:
        raise DriverError("result_cells and cell_results disagree")
    invocation_index = payload.get("invocation_index")
    if isinstance(invocation_index, int) and 1 <= invocation_index <= TOTAL_INVOCATIONS:
        expected_ids = cells_for_invocation(invocation_index)
        if ids != expected_ids:
            raise DriverError(
                f"result cells do not match invocation {invocation_index}: {ids}"
            )
    existing = set(str(cell) for cell in ledger["cells"])
    duplicate = sorted(existing.intersection(ids))
    if duplicate:
        raise FileExistsError(f"result cells already recorded: {duplicate}")
    all_completed = [*existing, *ids]
    # Validate before touching either artifact.  Order in the ledger is the
    # protocol order, while the result record preserves invocation order.
    denom = denominator(all_completed)
    record = {
        "invocation_index": payload.get("invocation_index"),
        "invocation": payload.get("invocation"),
        "result_path": str(result_path),
        "result_sha256": None,
        "result_cells": ids,
        "status": payload.get("status"),
        "finished_utc": payload.get("finished_utc", utc_now()),
    }
    # Write result first.  If ledger persistence fails, the preserved result
    # and its cells make a later retry fail closed instead of duplicating work.
    result_value = {
        **dict(payload),
        "denominator": denom,
        # Keep the explicit aliases easy to consume while deriving all three
        # from the same ordered denominator object.
        "completed_cells": denom["completed_cells"],
        "remaining_cells": denom["remaining_cells"],
        "pending_cells": denom["remaining_cells"],
    }
    write_new_json(result_path, result_value, root)
    record["result_sha256"] = digest(result_path)
    ledger["results"].append(record)
    for item in cell_results:
        ledger["cells"][str(item["id"])] = {
            "status": item.get("status"),
            "invocation_index": payload.get("invocation_index"),
            "result_path": str(result_path),
        }
    ledger["updated_utc"] = utc_now()
    ledger["denominator"] = denominator(ledger["cells"].keys())
    _atomic_replace_json(path, ledger)
    return result_value


def _status_from_value(value: Any, returncode: int | None) -> tuple[str, str | None]:
    if not isinstance(value, dict):
        return "dnf", "missing child result evidence"
    raw = str(value.get("status", "")).lower()
    if raw == "success" and returncode == 0:
        return "success", None
    if raw in {"failure", "failed", "error"}:
        return "failure", value.get("reason") or value.get("error")
    if raw in {"dnf", "partial_or_dnf", "timeout", "ambient_timeout"}:
        return "dnf", value.get("reason") or raw
    if raw == "success" and returncode not in (None, 0):
        return "failure", f"child process returned {returncode}"
    return "dnf", value.get("reason") or "child result status is missing/unknown"


def extract_colmap_cells(
    invocation: Mapping[str, Any], manifest: Mapping[str, Any], returncode: int | None
) -> list[dict[str, Any]]:
    """Extract incremental and global cells from the frozen COLMAP manifest."""

    results = manifest.get("results")
    if not isinstance(results, dict):
        return [
            {"id": cell, "status": "dnf", "reason": "COLMAP manifest.results missing"}
            for cell in invocation.get("result_cells", [])
        ]
    extracted: list[dict[str, Any]] = []
    for engine, cell in zip(
        ("incremental", "global"), invocation.get("result_cells", [])
    ):
        status, reason = _status_from_value(results.get(engine), returncode)
        row: dict[str, Any] = {
            "id": str(cell),
            "engine": engine,
            "status": status,
            "evidence": results.get(engine),
        }
        if reason:
            row["reason"] = str(reason)
        extracted.append(row)
    return extracted


def extract_visloc_cell(
    invocation: Mapping[str, Any], manifest: Mapping[str, Any], returncode: int | None
) -> dict[str, Any]:
    value = manifest.get("mapper")
    if str(manifest.get("status", "")).lower() in {"dnf", "partial_or_dnf"}:
        status, reason = "dnf", manifest.get("reason") or manifest.get("status")
    elif str(manifest.get("status", "")).lower() in {"failure", "failed", "error"}:
        status, reason = "failure", manifest.get("reason") or manifest.get("error")
    elif isinstance(value, dict) and value.get("returncode") not in (None, 0):
        status, reason = "failure", f"mapper returned {value.get('returncode')}"
    elif returncode == 0 and (isinstance(value, dict) or manifest.get("status") == "success"):
        status, reason = "success", None
    else:
        status, reason = "dnf", "incomplete hierarchical manifest"
    row: dict[str, Any] = {
        "id": str(invocation["result_cells"][0]),
        "engine": "visloc",
        "status": status,
        "evidence": value,
    }
    if reason:
        row["reason"] = str(reason)
    return row


def overall_status(cell_results: Sequence[Mapping[str, Any]]) -> str:
    statuses = [str(item.get("status")) for item in cell_results]
    if any(status == "dnf" for status in statuses):
        return "dnf"
    if any(status == "failure" for status in statuses):
        return "failure"
    if statuses and all(status == "success" for status in statuses):
        return "success"
    return "dnf"


def _manifest_record(path: Path) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if not path.is_file():
        return None, None, "child manifest is missing"
    manifest_sha = digest(path)
    try:
        manifest = read_json(path)
    except Exception as error:  # malformed child output is an explicit DNF
        return None, manifest_sha, f"child manifest is invalid: {type(error).__name__}: {error}"
    return manifest, manifest_sha, None


def manifest_is_gt_free(manifest: Mapping[str, Any]) -> bool:
    """Accept the frozen runners' two GT-free manifest layouts.

    The hierarchical runner records the flag under ``protocol`` while the
    COLMAP runner records it at the top level.  Neither layout may be absent
    or truthy for a B07-H runtime result.
    """

    if manifest.get("ground_truth_read") is False:
        return True
    protocol = manifest.get("protocol")
    return isinstance(protocol, dict) and protocol.get("ground_truth_read") is False


def _base_payload(
    context: InvocationContext,
    *,
    status: str,
    result_cells: Sequence[Mapping[str, Any]],
    mapping_started: bool,
    reason: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "B07H_RUNTIME_DRIVER_RESULT_V2",
        "status": status,
        "mapping_started": mapping_started,
        "invocation_index": context.invocation_index,
        "invocation": context.id,
        "engine": context.invocation.get("engine"),
        "sequence": context.invocation.get("sequence"),
        "result_cells": [str(item["id"]) for item in result_cells],
        "cell_results": [dict(item) for item in result_cells],
        "runset_sha256": EXPECTED_RUNSET_SHA256,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "protocol_sha256": str(context.runset["protocol"]["sha256"]).upper(),
        "prepared_manifest_sha256": digest(context.prepared_manifest_path),
        "prepared_manifest": str(context.prepared_manifest_path),
        "out_dir": str(context.out_dir),
        "gt_opened": False,
        "ground_truth_read": False,
        "finished_utc": utc_now(),
    }
    if reason:
        payload["reason"] = reason
    return payload


def _dnf_cells(context: InvocationContext, status: str, reason: str) -> list[dict[str, Any]]:
    return [
        {"id": cell, "status": status, "reason": reason}
        for cell in context.result_cells
    ]


def _deferred_payload(
    context: InvocationContext,
    result_path: Path,
    reason: str,
) -> dict[str, Any]:
    """Build a diagnostic payload while leaving its cells uncommitted."""

    payload = _base_payload(
        context,
        status="deferred",
        # Preserve the intended cells in the diagnostic for operator
        # visibility.  ``append_deferred_diagnostic`` is deliberately not
        # ``record_result``, so these cells do not enter the denominator.
        result_cells=_dnf_cells(context, "deferred", reason),
        mapping_started=False,
        reason=reason,
    )
    payload["deferred_cells"] = context.result_cells
    payload["result_path"] = str(result_path)
    return payload


def serial_progress(context: InvocationContext) -> dict[str, Any]:
    """Check that this positional invocation is the next serial invocation.

    A previous invocation may be terminal ``dnf``; it still counts as
    complete for ordering.  Seeing a future cell is itself terminal evidence
    that the candidate was attempted out of order, so this gate fails closed.
    """

    prior_cells = [
        str(cell)
        for invocation in context.runset["invocations"][: context.invocation_index - 1]
        for cell in invocation["result_cells"]
    ]
    future_cells = [
        str(cell)
        for invocation in context.runset["invocations"][context.invocation_index :]
        for cell in invocation["result_cells"]
    ]
    ledger = read_ledger(ledger_path(context.candidate_root))
    completed = {str(cell) for cell in ledger.get("cells", {})}
    missing_prior = [cell for cell in prior_cells if cell not in completed]
    future_present = [cell for cell in future_cells if cell in completed]
    return {
        "passed": not missing_prior and not future_present,
        "prior_cells": prior_cells,
        "missing_prior_cells": missing_prior,
        "future_cells_present": future_present,
        "completed_cells": [cell for cell in RESULT_CELLS if cell in completed],
    }


def preflight_checks(
    context: InvocationContext,
    result_path: Path,
    history_path: Path,
    workspace_root: Path,
    *,
    free_bytes_fn: Callable[[Path], int] = _drive_free_bytes,
) -> dict[str, Any]:
    """Evaluate every static/storage/path gate without starting a mapper."""

    root = context.candidate_root
    free_bytes = int(free_bytes_fn(root))
    workspace = c_workspace_state(workspace_root)
    checks: dict[str, Any] = {
        "runset_sha256": digest(context.runset_path) == EXPECTED_RUNSET_SHA256,
        "source_sha256": digest(context.source_path) == EXPECTED_SOURCE_SHA256,
        "prepared_manifest_sha256": digest(context.prepared_manifest_path)
        == str(
            _prepared_entry(context.runset, str(context.invocation["sequence"]))[
                "manifest_sha256"
            ]
        ).upper(),
        "prepared_manifest_gt_free": context.prepared_manifest.get("ground_truth_read")
        is False,
        "command_gt_free": not command_contains_gt(context.command),
        "output_absent": not context.out_dir.exists(),
        "result_absent": not result_path.exists(),
        "history_absent": not history_path.exists(),
        "history_valid": (
            not history_path.exists()
            or _ambient_history_is_valid(history_path, root)
        ),
        "e_free_bytes": free_bytes,
        "e_free_threshold": free_bytes >= STOP_FREE_BYTES,
        "c_workspace_target_absent": not workspace["target_present"],
        "c_workspace_temp_absent": not workspace["temp_present"],
    }
    serial = serial_progress(context)
    checks["serial_order"] = serial
    checks["serial_order_passed"] = serial["passed"]
    failed = [
        key
        for key, value in checks.items()
        if isinstance(value, bool) and not value
    ]
    if checks["e_free_threshold"] is False:
        failed.append("e_free_threshold")
    checks["c_workspace"] = workspace
    failed = list(dict.fromkeys(failed))
    # A valid history is an append-only diagnostic from an earlier deferred
    # attempt, not a consumed result or a static protocol violation.  Keep the
    # legacy ``history_absent`` observation in the report, but do not fail the
    # launch gate for it when the history is valid and appendable.
    if checks["history_valid"]:
        failed = [key for key in failed if key != "history_absent"]
    checks["failed"] = failed
    checks["resource_failed"] = [
        key for key in failed if key in DEFERRED_PREFLIGHT_CHECKS
    ]
    checks["static_failed"] = [
        key for key in failed if key not in DEFERRED_PREFLIGHT_CHECKS
    ]
    # ``deferred`` is true only when the resource gate is the sole reason the
    # invocation cannot launch.  A simultaneous protocol/path violation must
    # remain terminal DNF even if the machine is also short on space.
    checks["deferred"] = bool(checks["resource_failed"]) and not bool(
        checks["static_failed"]
    )
    checks["passed"] = not failed
    return checks


def _validation_report(
    *,
    candidate_root: Path,
    invocation_index: int,
    result_path: Path,
    context: InvocationContext | None,
    preflight: Mapping[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    completed: list[str] = []
    try:
        completed = list(read_ledger(ledger_path(candidate_root))["cells"])
    except Exception:
        completed = []
    report: dict[str, Any] = {
        "schema": "B07H_RUNTIME_DRIVER_VALIDATION_V2",
        "status": "validation_only",
        # Keep validation-only a non-mapping operation while making a failed
        # capacity/workspace gate explicit to callers.  In particular, a
        # report with ``passed: false`` must not be mistaken for a terminal
        # result row or a consumed denominator cell.
        "passed": bool(preflight and preflight.get("passed")) and error is None,
        "deferred": bool(preflight and preflight.get("deferred")) and error is None,
        "mapping_started": False,
        "invocation_index": invocation_index,
        "driver_sha256": digest(Path(__file__)),
        "runset_sha256": EXPECTED_RUNSET_SHA256,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "result_cells": [] if context is None else context.result_cells,
        "denominator": denominator(completed),
        "preflight": dict(preflight or {}),
        "finished_utc": utc_now(),
    }
    if context is not None:
        report.update(
            {
                "invocation": context.id,
                "sequence": context.invocation.get("sequence"),
                "command": list(context.command),
                "command_gt_free": not command_contains_gt(context.command),
                "out_dir": str(context.out_dir),
            }
        )
    if error:
        report["error"] = error
    return report


def run_invocation(
    context: InvocationContext,
    result_path: Path,
    history_path: Path,
    driver_log_path: Path,
    workspace_root: Path,
    *,
    ambient_timeout_seconds: float = 7200.0,
    sample_seconds: float = 2.0,
    consecutive_samples: int = 5,
    execution_timeout_seconds: float = 7200.0,
    process_sampler: Callable[[], dict[str, Any]] = powershell_sample,
    wsl_sampler: Callable[[], dict[str, Any]] | None = None,
    gpu_sampler: Callable[[], dict[str, Any]] = gpu_sample,
    free_bytes_fn: Callable[[Path], int] = _drive_free_bytes,
) -> tuple[str, dict[str, Any]]:
    """Run one positional invocation, deferring operator/resource gates."""

    root = context.candidate_root
    result_path = candidate_path(result_path, root)
    history_path = candidate_path(history_path, root)
    driver_log_path = candidate_path(driver_log_path, root)
    checks = preflight_checks(
        context,
        result_path,
        history_path,
        workspace_root,
        free_bytes_fn=free_bytes_fn,
    )
    if not checks["passed"]:
        reason = "preflight failed: " + ", ".join(checks["failed"])
        if checks.get("deferred"):
            payload = _deferred_payload(context, result_path, reason)
            payload["preflight"] = checks
            payload["ambient_gate"] = None
            return "deferred", append_deferred_diagnostic(root, payload)
        payload = _base_payload(
            context,
            status="dnf",
            result_cells=_dnf_cells(context, "dnf", reason),
            mapping_started=False,
            reason=reason,
        )
        payload["preflight"] = checks
        payload["ambient_gate"] = None
        return "dnf", record_result(root, result_path, payload)

    ambient = settle_ambient(
        root,
        history_path,
        workspace_root,
        timeout_seconds=ambient_timeout_seconds,
        sample_seconds=sample_seconds,
        consecutive_samples=consecutive_samples,
        process_sampler=process_sampler,
        wsl_sampler=wsl_sampler,
        gpu_sampler=gpu_sampler,
        free_bytes_fn=free_bytes_fn,
    )
    if ambient["reason"] != "settled":
        reason = "ambient CPU/Search/GPU settle timeout"
        payload = _deferred_payload(context, result_path, reason)
        payload["preflight"] = checks
        payload["ambient_gate"] = ambient
        return "deferred", append_deferred_diagnostic(root, payload)

    # The settle sample is not a reservation.  Check storage and C: one more
    # time immediately before process launch, so a compression job cannot
    # cross the threshold between the gate and Popen.
    free_before = int(free_bytes_fn(root))
    workspace = c_workspace_state(workspace_root)
    if free_before < STOP_FREE_BYTES or not workspace["clean"]:
        reason = "post-settle storage/workspace gate failed"
        payload = _deferred_payload(context, result_path, reason)
        payload["preflight"] = checks
        payload["ambient_gate"] = ambient
        payload["e_free_bytes_before"] = free_before
        payload["c_workspace"] = workspace
        return "deferred", append_deferred_diagnostic(root, payload)

    try:
        env, locations = build_runtime_environment(root, context.invocation_index)
    except Exception as error:
        reason = f"E-only environment setup failed: {type(error).__name__}: {error}"
        if isinstance(error, DriverError) and "path escapes B07 candidate" in str(error):
            # A candidate escape is a deterministic static-security failure;
            # never turn it into a retryable infrastructure diagnosis.
            payload = _base_payload(
                context,
                status="dnf",
                result_cells=_dnf_cells(context, "dnf", reason),
                mapping_started=False,
                reason=reason,
            )
            payload["preflight"] = checks
            payload["ambient_gate"] = ambient
            return "dnf", record_result(root, result_path, payload)
        payload = _deferred_payload(context, result_path, reason)
        payload["preflight"] = checks
        payload["ambient_gate"] = ambient
        payload["environment_error"] = reason
        return "deferred", append_deferred_diagnostic(root, payload)

    if driver_log_path.exists():
        raise FileExistsError(f"refusing to overwrite driver log: {driver_log_path}")
    driver_log_path.parent.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    returncode: int | None = None
    launched = False
    timed_out = False
    launch_error: str | None = None
    with driver_log_path.open("x", encoding="utf-8") as stream:
        try:
            process = subprocess.Popen(
                context.command,
                cwd=root,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            launched = True
            try:
                returncode = process.wait(timeout=execution_timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.terminate()
                try:
                    returncode = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    returncode = process.wait(timeout=30)
        except Exception as error:
            launch_error = f"{type(error).__name__}: {error}"
    finished = utc_now()
    manifest_path = context.out_dir / "manifest.json"
    manifest, manifest_sha, manifest_error = _manifest_record(manifest_path)
    if timed_out:
        status = "dnf"
        cells = _dnf_cells(context, "dnf", "mapping execution timeout")
    elif launch_error:
        status = "dnf"
        cells = _dnf_cells(context, "dnf", f"mapping launch failed: {launch_error}")
    elif manifest_error:
        status = "dnf"
        cells = _dnf_cells(context, "dnf", manifest_error)
    elif manifest is None:
        status = "dnf"
        cells = _dnf_cells(context, "dnf", "mapping manifest missing")
    elif not manifest_is_gt_free(manifest):
        status = "dnf"
        cells = _dnf_cells(context, "dnf", "child manifest is not GT-free")
    elif context.invocation.get("engine") == "colmap":
        cells = extract_colmap_cells(context.invocation, manifest, returncode)
        status = overall_status(cells)
    else:
        cells = [extract_visloc_cell(context.invocation, manifest, returncode)]
        status = overall_status(cells)
    payload = _base_payload(
        context,
        status=status,
        result_cells=cells,
        mapping_started=launched,
        reason=None,
    )
    payload.update(
        {
            "start_utc": started,
            "finish_utc": finished,
            "exit_code": returncode,
            "driver_log": str(driver_log_path),
            "driver_log_sha256": digest(driver_log_path),
            "manifest": {
                "path": str(manifest_path),
                "sha256": manifest_sha,
                "status": manifest.get("status") if manifest else None,
            },
            "ambient_gate": ambient,
            "preflight": checks,
            "runtime_environment": locations,
            "e_free_bytes_before": free_before,
            "e_free_bytes_after": int(free_bytes_fn(root)),
            "c_workspace": c_workspace_state(workspace_root),
        }
    )
    if launch_error:
        payload["reason"] = launch_error
    if timed_out:
        payload["reason"] = "mapping execution timeout"
    return status, record_result(root, result_path, payload)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runset", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--invocation-index", type=int, required=True)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_C_WORKSPACE)
    parser.add_argument("--ambient-timeout-seconds", type=float, default=7200.0)
    parser.add_argument(
        "--timeout-seconds",
        dest="ambient_timeout_seconds",
        type=float,
        help="compatibility alias for --ambient-timeout-seconds",
    )
    parser.add_argument("--execution-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--sample-seconds", type=float, default=2.0)
    parser.add_argument("--consecutive-samples", type=int, default=5)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--driver-log", type=Path)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args(argv)
    if args.ambient_timeout_seconds is None:
        args.ambient_timeout_seconds = 7200.0
    return args


def _infer_candidate_root(args: argparse.Namespace) -> Path:
    if args.candidate_root is not None:
        return resolve_root(args.candidate_root)
    runset = Path(args.runset)
    if runset.is_absolute():
        return resolve_root(runset.parent)
    return resolve_root(Path(__file__).resolve().parents[1])


def _fallback_cells(index: int) -> list[str]:
    if 1 <= index <= TOTAL_INVOCATIONS:
        starts = (0, 1, 3, 4, 6, 7)
        lengths = (1, 2, 1, 2, 1, 2)
        start = starts[index - 1]
        return list(RESULT_CELLS[start : start + lengths[index - 1]])
    return []


def cells_for_invocation(index: int) -> list[str]:
    """Return the immutable result-cell slice for a positional invocation."""

    cells = _fallback_cells(index)
    if not cells:
        raise DriverError("invocation-index must be between 1 and 6")
    return cells


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    candidate_root = _infer_candidate_root(args)
    report_path: Path | None = None
    context: InvocationContext | None = None
    try:
        context = load_invocation_context(
            args.runset, args.invocation_index, candidate_root
        )
        result_path = candidate_path(
            args.result
            if args.result is not None
            else default_result_path(candidate_root, context),
            candidate_root,
        )
        if args.validation_only:
            report_path = candidate_path(
                args.result
                if args.result is not None
                else candidate_path(
                    Path("logs")
                    / f"B07H_v2_validation_invocation_{args.invocation_index:02d}.json",
                    candidate_root,
                ),
                candidate_root,
            )
            checks = preflight_checks(
                context,
                result_path,
                candidate_path(
                    args.history
                    if args.history is not None
                    else default_history_path(candidate_root, context),
                    candidate_root,
                ),
                args.workspace_root,
            )
            report = _validation_report(
                candidate_root=candidate_root,
                invocation_index=args.invocation_index,
                result_path=report_path,
                context=context,
                preflight=checks,
                error=None,
            )
            write_new_json(report_path, report, candidate_root)
            print(report_path)
            return 0
        history_path = candidate_path(
            args.history
            if args.history is not None
            else default_history_path(candidate_root, context),
            candidate_root,
        )
        driver_log_path = candidate_path(
            args.driver_log
            if args.driver_log is not None
            else default_driver_log_path(candidate_root, context),
            candidate_root,
        )
        status, payload = run_invocation(
            context,
            result_path,
            history_path,
            driver_log_path,
            args.workspace_root,
            ambient_timeout_seconds=args.ambient_timeout_seconds,
            sample_seconds=args.sample_seconds,
            consecutive_samples=args.consecutive_samples,
            execution_timeout_seconds=args.execution_timeout_seconds,
        )
        # A deferred invocation has no result artifact yet; expose its
        # append-only diagnostic path to the caller instead of printing a
        # path which was intentionally not created.
        print(payload.get("deferred_diagnostic", result_path) if status == "deferred" else result_path)
        return {
            "success": 0,
            "failure": 1,
            "dnf": 2,
            # Keep the old key available for callers which report historical
            # statuses, while new prelaunch gates use the distinct deferred
            # status/exit code.
            "ambient_timeout": 3,
            "deferred": 4,
        }[status]
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        if args.validation_only:
            try:
                report_path = candidate_path(
                    args.result
                    if args.result is not None
                    else Path("logs")
                    / f"B07H_v2_validation_invocation_{args.invocation_index:02d}.json",
                    candidate_root,
                )
                report = _validation_report(
                    candidate_root=candidate_root,
                    invocation_index=args.invocation_index,
                    result_path=report_path,
                    context=context,
                    preflight=None,
                    error=message,
                )
                write_new_json(report_path, report, candidate_root)
                print(report_path, file=sys.stderr)
            except Exception as report_error:
                print(f"{message}; validation report failed: {report_error}", file=sys.stderr)
            return 1
        # A valid invocation should leave a denominator row even when static
        # validation fails.  If the runset itself was corrupt, do not invent a
        # result cell mapping: the runset hash failure is terminal and the
        # caller must repair the archive rather than execute anything.
        if context is not None:
            try:
                result_path = candidate_path(
                    args.result
                    if args.result is not None
                    else default_result_path(candidate_root, context),
                    candidate_root,
                )
                cells = _dnf_cells(context, "dnf", message)
                payload = _base_payload(
                    context,
                    status="dnf",
                    result_cells=cells,
                    mapping_started=False,
                    reason=message,
                )
                payload["preflight"] = {"passed": False, "failed": [message]}
                record_result(candidate_root, result_path, payload)
                print(result_path, file=sys.stderr)
            except Exception as result_error:
                print(f"{message}; result preservation failed: {result_error}", file=sys.stderr)
        else:
            # If immutable validation itself failed, the invocation object is
            # intentionally unavailable.  The fixed protocol still gives us
            # the positional cell mapping, so preserve a terminal DNF row
            # rather than silently shrinking the nine-cell denominator.
            if 1 <= args.invocation_index <= TOTAL_INVOCATIONS:
                try:
                    fallback_result = candidate_path(
                        args.result
                        if args.result is not None
                        else Path("logs")
                        / f"B07H_v2_invocation_{args.invocation_index:02d}_"
                        f"{SERIAL_ORDER[args.invocation_index - 1]}.json",
                        candidate_root,
                    )
                    cells = _fallback_cells(args.invocation_index)
                    payload = {
                        "schema": "B07H_RUNTIME_DRIVER_RESULT_V2",
                        "status": "dnf",
                        "mapping_started": False,
                        "invocation_index": args.invocation_index,
                        "invocation": SERIAL_ORDER[args.invocation_index - 1],
                        "result_cells": cells,
                        "cell_results": [
                            {"id": cell, "status": "dnf", "reason": message}
                            for cell in cells
                        ],
                        "runset_sha256": EXPECTED_RUNSET_SHA256,
                        "source_sha256": EXPECTED_SOURCE_SHA256,
                        "gt_opened": False,
                        "ground_truth_read": False,
                        "reason": message,
                        "finished_utc": utc_now(),
                    }
                    record_result(candidate_root, fallback_result, payload)
                    print(fallback_result, file=sys.stderr)
                except Exception as result_error:
                    print(
                        f"{message}; result preservation failed: {result_error}",
                        file=sys.stderr,
                    )
            else:
                print(message, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
