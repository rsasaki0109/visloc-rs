#!/usr/bin/env python3
"""Recover a valid B07-H invocation from immutable v7 child evidence.

The v7 launcher required a top-level ``status`` field which the frozen
mapper manifest does not provide.  This recovery lane never edits v7 files:
it verifies the sealed v7 result/ledger, the runner's schema-version-1
manifest, and the regular output evidence, then writes a distinct v8 result
and ledger with explicit provenance to all original artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.dont_write_bytecode = True

import run_b07h_runtime_driver_v7 as v7  # noqa: E402


DRIVER_VERSION = "B07H_RUNTIME_DRIVER_V8_RECOVERY"
RESULT_SCHEMA = "B07H_RUNTIME_DRIVER_RESULT_V5_RECOVERED"
LEDGER_SCHEMA = "B07H_RUNTIME_DRIVER_LEDGER_V5_RECOVERED"
RECOVERY_SCHEMA = "B07H_RUNTIME_DRIVER_V8_RECOVERY_V1"
AMBIENT_POLICY = v7.AMBIENT_POLICY
TOTAL_RESULT_CELLS = v7.TOTAL_RESULT_CELLS
RESULT_CELLS = v7.RESULT_CELLS
INVOCATION_CELLS = v7.INVOCATION_CELLS
EXPECTED_SOURCE_SHA256 = v7.EXPECTED_SOURCE_SHA256
EXPECTED_PROTOCOL_SHA256 = v7.EXPECTED_PROTOCOL_SHA256
DriverError = v7.DriverError
digest = v7.digest
read_json = v7.read_json
candidate_path = v7.candidate_path
require_e_root = v7.require_e_root
require_regular_candidate_file = v7.require_regular_candidate_file
validate_sidecar = v7.validate_sidecar
atomic_json = v7.atomic_json
reject_gt = v7.reject_gt


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path, root: Path, label: str) -> str:
    return validate_sidecar(path, root, label).upper()


def _nonempty(path: Path, root: Path, label: str) -> tuple[Path, str]:
    checked = require_regular_candidate_file(path, root, label)
    if checked.stat().st_size <= 0:
        raise DriverError(f"{label} is empty")
    return checked, digest(checked).upper()


def _line_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            if line.strip() and not line.lstrip().startswith("#"):
                count += 1
    return count


def _expected_frames(runset: Mapping[str, Any]) -> int:
    invocations = runset.get("invocations")
    if not isinstance(invocations, list) or len(invocations) != 6:
        raise DriverError("v8 runset invocation inventory is malformed")
    first = invocations[0]
    if not isinstance(first, Mapping) or first.get("id") != INVOCATION_CELLS[0][0]:
        raise DriverError("v8 runset invocation-1 identity mismatch")
    command = first.get("command")
    if not isinstance(command, list):
        raise DriverError("v8 runset invocation-1 command is malformed")
    try:
        value = int(command[command.index("--expected-frames") + 1])
    except (ValueError, IndexError, TypeError) as error:
        raise DriverError("v8 runset invocation-1 expected-frame contract is missing") from error
    if value <= 0:
        raise DriverError("v8 expected frame count is not positive")
    return value


def _validate_original(root: Path, result_path: Path, ledger_path: Path, runset_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, str, str]:
    result_path = require_regular_candidate_file(result_path, root, "v8 original v7 result")
    ledger_path = require_regular_candidate_file(ledger_path, root, "v8 original v7 ledger")
    result_sha = _sha(result_path, root, "v8 original v7 result")
    ledger_sha = _sha(ledger_path, root, "v8 original v7 ledger")
    result = read_json(result_path)
    ledger = read_json(ledger_path)
    if result.get("schema") != v7.RESULT_SCHEMA or result.get("ambient_policy") != AMBIENT_POLICY or result.get("status") != "dnf" or result.get("terminal") is not True:
        raise DriverError("v8 recovery requires a sealed v7 DNF result")
    if result.get("invocation_index") != 1 or result.get("invocation") != INVOCATION_CELLS[0][0] or result.get("result_cells") != list(INVOCATION_CELLS[0][3]):
        raise DriverError("v8 original result is not invocation 1")
    if result.get("cell_results") != [{"id": INVOCATION_CELLS[0][3][0], "reason": "child result status is missing/unknown", "status": "dnf"}]:
        raise DriverError("v8 original result cell is not the known manifest-status DNF")
    for key in ("gt_opened", "ground_truth_read", "ground_truth_materialized", "ground_truth_argument_present_anywhere"):
        if result.get(key) is not False:
            raise DriverError(f"v8 original result {key} is not false")
    if result.get("source_sha256", "").upper() != EXPECTED_SOURCE_SHA256 or result.get("protocol_sha256", "").upper() != EXPECTED_PROTOCOL_SHA256:
        raise DriverError("v8 original result immutable hash binding mismatch")
    manifest = result.get("manifest")
    if not isinstance(manifest, Mapping) or manifest.get("path") is None:
        raise DriverError("v8 original result has no child manifest provenance")
    manifest_path = require_regular_candidate_file(manifest["path"], root, "v8 original child manifest")
    manifest_sha = digest(manifest_path).upper()
    if str(manifest.get("sha256", "")).upper() != manifest_sha:
        raise DriverError("v8 original child manifest SHA mismatch")
    if ledger.get("schema") != v7.LEDGER_SCHEMA or ledger.get("ambient_policy") != AMBIENT_POLICY:
        raise DriverError("v8 original ledger schema/policy mismatch")
    cell = ledger.get("cells", {}).get(INVOCATION_CELLS[0][3][0]) if isinstance(ledger.get("cells"), Mapping) else None
    if not isinstance(cell, Mapping) or cell.get("status") != "dnf" or cell.get("result_path") != str(result_path) or str(cell.get("result_sha256", "")).upper() != result_sha:
        raise DriverError("v8 original ledger is not bound to the sealed v7 result")
    runset_path = require_regular_candidate_file(runset_path, root, "v8 source runset")
    runset_sha = _sha(runset_path, root, "v8 source runset")
    runset = read_json(runset_path)
    if runset.get("schema") != v7.RUNSET_SCHEMA or runset_sha != str(result.get("runset_sha256", "")).upper():
        raise DriverError("v8 source runset binding mismatch")
    v7.validate_runset(runset_path, root, runset_sha)
    reject_gt(result, "v8 original result")
    return result, ledger, runset, result_sha, ledger_sha, manifest_sha


def _validate_child(root: Path, manifest_path: Path, runset: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = require_regular_candidate_file(manifest_path, root, "v8 mapper child manifest")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise DriverError("v8 mapper manifest schema_version must be 1")
    mapper = manifest.get("mapper")
    protocol = manifest.get("protocol")
    executable = manifest.get("executable")
    if not isinstance(mapper, Mapping) or mapper.get("returncode") != 0 or not isinstance(protocol, Mapping) or not isinstance(executable, Mapping):
        raise DriverError("v8 mapper manifest success contract is incomplete")
    frames = _expected_frames(runset)
    for key in ("input_feature_frames", "timestamp_rows", "expected_frames"):
        if protocol.get(key) != frames:
            raise DriverError(f"v8 mapper manifest {key} does not match runset")
    if mapper.get("registered_images") != frames or not isinstance(mapper.get("points3d"), int) or mapper.get("points3d") <= 0:
        raise DriverError("v8 mapper result counts do not prove success")
    if not isinstance(mapper.get("wall_seconds"), (int, float)) or mapper.get("wall_seconds") <= 0:
        raise DriverError("v8 mapper wall time is invalid")
    exe_path, exe_sha = _nonempty(executable.get("path"), root, "v8 mapper executable")
    if str(executable.get("sha256", "")).upper() != exe_sha:
        raise DriverError("v8 mapper executable SHA mismatch")
    tools = runset.get("fixed_tools")
    expected_tool = tools.get("hierarchical_executable") if isinstance(tools, Mapping) else None
    if not isinstance(expected_tool, Mapping) or exe_sha != str(expected_tool.get("sha256", "")).upper():
        raise DriverError("v8 mapper executable is not the runset-bound tool")
    output = candidate_path(manifest_path.parent, root, "v8 mapper output")
    expected_output = candidate_path(runset["invocations"][0]["output"], root, "v8 runset output")
    if output != expected_output:
        raise DriverError("v8 mapper manifest is outside the declared invocation output")
    model = output / "model"
    evidence: dict[str, Any] = {}
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        path, sha = _nonempty(model / name, root, f"v8 mapper evidence {name}")
        evidence[name] = {"path": str(path), "sha256": sha, "bytes": path.stat().st_size}
    trajectory, trajectory_sha = _nonempty(output / "trajectory.tum", root, "v8 mapper trajectory")
    mapping_log, mapping_sha = _nonempty(output / "mapping.log", root, "v8 mapper mapping log")
    evidence["trajectory.tum"] = {"path": str(trajectory), "sha256": trajectory_sha, "bytes": trajectory.stat().st_size}
    evidence["mapping.log"] = {"path": str(mapping_log), "sha256": mapping_sha, "bytes": mapping_log.stat().st_size}
    if _line_count(model / "images.txt") < frames * 2 or _line_count(model / "points3D.txt") != mapper["points3d"]:
        raise DriverError("v8 mapper model line counts do not match manifest counts")
    reject_gt(manifest, "v8 mapper manifest")
    return manifest, evidence


def recover_invocation_one(root: Path, *, original_result: Path, original_ledger: Path, runset: Path, output_result: Path, output_ledger: Path) -> tuple[Path, Path]:
    root = require_e_root(root)
    original_result = candidate_path(original_result, root, "v8 original result")
    original_ledger = candidate_path(original_ledger, root, "v8 original ledger")
    output_result = candidate_path(output_result, root, "v8 recovered result")
    output_ledger = candidate_path(output_ledger, root, "v8 recovered ledger")
    if len({str(original_result), str(original_ledger), str(output_result), str(output_ledger)}) != 4:
        raise DriverError("v8 recovery artifacts must be distinct")
    original, ledger, runset_value, original_result_sha, original_ledger_sha, _ = _validate_original(root, original_result, original_ledger, candidate_path(runset, root, "v8 runset"))
    manifest_path = candidate_path(original["manifest"]["path"], root, "v8 child manifest")
    manifest, evidence = _validate_child(root, manifest_path, runset_value)
    recovered = {"schema": RESULT_SCHEMA, "recovery_schema": RECOVERY_SCHEMA, "ambient_policy": AMBIENT_POLICY, "status": "success", "terminal": True, "attempt_terminal": True, "mapping_started": True, "invocation_index": 1, "invocation": INVOCATION_CELLS[0][0], "engine": "visloc", "sequence": "MH_01_easy", "result_cells": list(INVOCATION_CELLS[0][3]), "cell_results": [{"id": INVOCATION_CELLS[0][3][0], "status": "success", "reason": None}], "runset_sha256": digest(candidate_path(runset, root, "v8 runset")).upper(), "source_sha256": EXPECTED_SOURCE_SHA256, "protocol_sha256": EXPECTED_PROTOCOL_SHA256, "gt_opened": False, "ground_truth_read": False, "ground_truth_materialized": False, "ground_truth_argument_present_anywhere": False, "manifest": {"ambient_policy": AMBIENT_POLICY, "path": str(manifest_path), "sha256": digest(manifest_path).upper()}, "recovery": {"schema": RECOVERY_SCHEMA, "original_result_path": str(original_result), "original_result_sha256": original_result_sha, "original_ledger_path": str(original_ledger), "original_ledger_sha256": original_ledger_sha, "original_status": original["status"], "child_manifest_path": str(manifest_path), "child_manifest_sha256": digest(manifest_path).upper(), "child_manifest": manifest, "output_evidence": evidence}, "finished_utc": _utc()}
    reject_gt(recovered, "v8 recovered result")
    result_sha = atomic_json(output_result, recovered, root, replace=False)
    recovered_ledger = {"schema": LEDGER_SCHEMA, "recovery_schema": RECOVERY_SCHEMA, "ambient_policy": AMBIENT_POLICY, "total_result_cells": TOTAL_RESULT_CELLS, "expected_cells": list(RESULT_CELLS), "results": [{"ambient_policy": AMBIENT_POLICY, "invocation_index": 1, "invocation": INVOCATION_CELLS[0][0], "result_cells": list(INVOCATION_CELLS[0][3]), "status": "success", "result_path": str(output_result), "result_sha256": result_sha, "provenance": recovered["recovery"]}], "cells": {INVOCATION_CELLS[0][3][0]: {"ambient_policy": AMBIENT_POLICY, "invocation_index": 1, "result_path": str(output_result), "result_sha256": result_sha, "status": "success"}}, "denominator": {"total_cells": TOTAL_RESULT_CELLS, "completed_cells": list(INVOCATION_CELLS[0][3]), "completed_count": 1, "remaining_cells": [cell for cell in RESULT_CELLS if cell not in INVOCATION_CELLS[0][3]], "remaining_count": TOTAL_RESULT_CELLS - 1}, "provenance": {"original_result_path": str(original_result), "original_result_sha256": original_result_sha, "original_ledger_path": str(original_ledger), "original_ledger_sha256": original_ledger_sha}, "updated_utc": _utc()}
    reject_gt(recovered_ledger, "v8 recovered ledger")
    atomic_json(output_ledger, recovered_ledger, root, replace=False)
    return output_result, output_ledger


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--original-result", type=Path, required=True)
    parser.add_argument("--original-ledger", type=Path, required=True)
    parser.add_argument("--runset", type=Path, required=True)
    parser.add_argument("--output-result", type=Path, required=True)
    parser.add_argument("--output-ledger", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result, ledger = recover_invocation_one(args.candidate_root, original_result=args.original_result, original_ledger=args.original_ledger, runset=args.runset, output_result=args.output_result, output_ledger=args.output_ledger)
    print(json.dumps({"schema": RECOVERY_SCHEMA, "status": "recovered", "result": str(result), "ledger": str(ledger)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DRIVER_VERSION", "RESULT_SCHEMA", "LEDGER_SCHEMA", "RECOVERY_SCHEMA", "recover_invocation_one", "parse_args", "main"]
