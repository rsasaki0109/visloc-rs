#!/usr/bin/env python3
"""Cumulative, non-overwriting importer for sealed B07-H v7 results.

V9 starts from the separately sealed v8 invocation-1 recovery and appends
invocations 2--6 one at a time.  Every append reads an immutable v7 result,
revalidates its child manifest/output contract, and writes a new result and a
new cumulative ledger path; no prior V7/V8/V9 artifact is replaced.
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
import run_b07h_runtime_driver_v8 as v8  # noqa: E402


DRIVER_VERSION = "B07H_RUNTIME_DRIVER_V9_CUMULATIVE_IMPORT"
RESULT_SCHEMA = "B07H_RUNTIME_DRIVER_RESULT_V6_CUMULATIVE"
V10_RESULT_SCHEMA = "B07H_RUNTIME_DRIVER_RESULT_V7_RELAXED"
LEDGER_SCHEMA = "B07H_RUNTIME_DRIVER_LEDGER_V6_CUMULATIVE"
CHAIN_SCHEMA = "B07H_RUNTIME_DRIVER_V9_CHAIN_V1"
AMBIENT_POLICY = v7.AMBIENT_POLICY
RESULT_CELLS = v7.RESULT_CELLS
INVOCATION_CELLS = v7.INVOCATION_CELLS
TOTAL_RESULT_CELLS = v7.TOTAL_RESULT_CELLS
EXPECTED_SOURCE_SHA256 = v7.EXPECTED_SOURCE_SHA256
EXPECTED_PROTOCOL_SHA256 = v7.EXPECTED_PROTOCOL_SHA256
DriverError = v8.DriverError


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path, root: Path, label: str) -> tuple[dict[str, Any], str, Path]:
    checked = v7.require_regular_candidate_file(path, root, label)
    sha = v7.validate_sidecar(checked, root, label).upper()
    return v7.read_json(checked), sha, checked


def _expected(root: Path, runset_path: Path, index: int) -> tuple[dict[str, Any], int]:
    runset, _, _ = _load(runset_path, root, "v9 runset")
    if runset.get("schema") not in {v7.RUNSET_SCHEMA, "B07H_GT_FREE_RUNTIME_RUNSET_V4"}:
        raise DriverError("v9 requires runset V3 or the versioned V4 relaxed runset")
    invocations = runset.get("invocations")
    if not isinstance(invocations, list) or len(invocations) != 6:
        raise DriverError("v9 runset invocation inventory is malformed")
    invocation = invocations[index - 1]
    if not isinstance(invocation, Mapping):
        raise DriverError("v9 invocation declaration is malformed")
    command = invocation.get("command")
    try:
        frames = int(command[command.index("--expected-frames") + 1])
    except (AttributeError, ValueError, IndexError, TypeError) as error:
        raise DriverError("v9 invocation expected-frame contract is missing") from error
    return dict(invocation), frames


def _validate_prior(root: Path, ledger_path: Path) -> tuple[dict[str, Any], str, Path]:
    ledger, sha, checked = _load(ledger_path, root, "v9 prior ledger")
    if ledger.get("schema") != LEDGER_SCHEMA or ledger.get("chain_schema") != CHAIN_SCHEMA or ledger.get("ambient_policy") != AMBIENT_POLICY:
        raise DriverError("v9 prior ledger schema/policy mismatch")
    records = ledger.get("results")
    cells = ledger.get("cells")
    if not isinstance(records, list) or not isinstance(cells, Mapping) or len(cells) != sum(len(record.get("result_cells", [])) for record in records if isinstance(record, Mapping)):
        raise DriverError("v9 prior ledger inventory is malformed")
    if len(records) < 1 or len(records) > 5:
        raise DriverError("v9 prior ledger must contain invocation 1 through 5")
    seen_paths: set[str] = set()
    for index, record in enumerate(records, 1):
        if not isinstance(record, Mapping) or record.get("invocation_index") != index or record.get("result_cells") != list(INVOCATION_CELLS[index - 1][3]):
            raise DriverError(f"v9 prior invocation {index} order mismatch")
        path = v7.require_regular_candidate_file(record.get("result_path"), root, f"v9 prior result {index}")
        path_key = str(path).lower()
        if path_key in seen_paths or path_key == str(checked).lower():
            raise DriverError("v9 prior evidence path alias detected")
        seen_paths.add(path_key)
        actual = v7.validate_sidecar(path, root, f"v9 prior result {index}").upper()
        if actual != str(record.get("result_sha256", "")).upper():
            raise DriverError(f"v9 prior result {index} hash mismatch")
        value = v7.read_json(path)
        if value.get("schema") not in {v8.RESULT_SCHEMA, RESULT_SCHEMA} or value.get("ambient_policy") != AMBIENT_POLICY or value.get("terminal") is not True:
            raise DriverError(f"v9 prior result {index} schema mismatch")
        if value.get("runset_sha256", "").upper() != str(ledger.get("runset_sha256", value.get("runset_sha256", ""))).upper():
            raise DriverError(f"v9 prior result {index} runset binding mismatch")
        for cell in record["result_cells"]:
            binding = cells.get(cell)
            if not isinstance(binding, Mapping) or binding.get("invocation_index") != index or binding.get("result_path") != str(path) or str(binding.get("result_sha256", "")).upper() != actual:
                raise DriverError(f"v9 prior cell {cell} binding mismatch")
    if ledger.get("denominator", {}).get("completed_count") != len(cells) or ledger.get("denominator", {}).get("total_cells") != TOTAL_RESULT_CELLS:
        raise DriverError("v9 prior denominator mismatch")
    return ledger, sha, checked


def initialize_from_v8(root: Path, *, v8_ledger: Path, output_ledger: Path) -> Path:
    root = v7.require_e_root(root)
    source, source_sha, source_path = _load(v8_ledger, root, "v9 source v8 ledger")
    if source.get("schema") != v8.LEDGER_SCHEMA or source.get("recovery_schema") != v8.RECOVERY_SCHEMA or len(source.get("results", [])) != 1:
        raise DriverError("v9 initialization requires the sealed one-result v8 ledger")
    record = source["results"][0]
    result_path = v7.require_regular_candidate_file(record.get("result_path"), root, "v9 v8 recovered result")
    result_sha = v7.validate_sidecar(result_path, root, "v9 v8 recovered result").upper()
    if result_sha != str(record.get("result_sha256", "")).upper() or v7.read_json(result_path).get("schema") != v8.RESULT_SCHEMA:
        raise DriverError("v9 v8 recovery result binding mismatch")
    output = v7.candidate_path(output_ledger, root, "v9 initialized ledger")
    if output == source_path:
        raise DriverError("v9 initialization would overwrite v8 ledger")
    cumulative = {"schema": LEDGER_SCHEMA, "chain_schema": CHAIN_SCHEMA, "ambient_policy": AMBIENT_POLICY, "runset_sha256": v7.read_json(result_path).get("runset_sha256"), "total_result_cells": TOTAL_RESULT_CELLS, "expected_cells": list(RESULT_CELLS), "results": [{**dict(record), "source_schema": v8.RESULT_SCHEMA, "source_ledger_path": str(source_path), "source_ledger_sha256": source_sha}], "cells": {cell: {"ambient_policy": AMBIENT_POLICY, "invocation_index": 1, "result_path": str(result_path), "result_sha256": result_sha, "status": source["cells"][cell]["status"]} for cell in record["result_cells"]}, "denominator": {"total_cells": TOTAL_RESULT_CELLS, "completed_cells": list(record["result_cells"]), "completed_count": 1, "remaining_cells": [cell for cell in RESULT_CELLS if cell not in record["result_cells"]], "remaining_count": TOTAL_RESULT_CELLS - 1}, "provenance": {"parent_ledger_path": str(source_path), "parent_ledger_sha256": source_sha}, "updated_utc": _now()}
    v7.atomic_json(output, cumulative, root, replace=False)
    return output


def _manifest_cells(root: Path, manifest_path: Path | None, invocation: Mapping[str, Any], frames: int) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str | None, str | None]:
    cells = list(invocation.get("result_cells", []))
    if manifest_path is None:
        return [{"id": cell, "status": "dnf", "reason": "child manifest is missing"} for cell in cells], None, None, None
    checked = v7.require_regular_candidate_file(manifest_path, root, "v9 child manifest")
    manifest_sha = v7.digest(checked).upper()
    manifest = v7.read_json(checked)
    if manifest.get("schema_version") != 1 or manifest.get("ground_truth_read") is not False:
        return [{"id": cell, "status": "dnf", "reason": "child manifest is partial or not GT-free"} for cell in cells], manifest, manifest_sha, str(checked)
    if invocation.get("engine") == "visloc":
        mapper = manifest.get("mapper")
        protocol = manifest.get("protocol")
        if isinstance(mapper, Mapping) and mapper.get("returncode") == 0 and isinstance(protocol, Mapping) and protocol.get("input_feature_frames") == frames and protocol.get("timestamp_rows") == frames and protocol.get("expected_frames") == frames and mapper.get("registered_images") == frames and isinstance(mapper.get("points3d"), int) and mapper.get("points3d") > 0:
            # Full model/evidence validation is shared with v8, using a local
            # invocation-shaped runset so v8 remains the audited checker.
            synthetic = {"invocations": [dict(invocation, command=["python", "--expected-frames", str(frames)], output=str(Path(str(checked)).parent.relative_to(root)))] * 6, "fixed_tools": {"hierarchical_executable": manifest.get("executable", {})}}
            try:
                v8._validate_child(root, checked, synthetic)
                return [{"id": cells[0], "status": "success", "reason": None}], manifest, manifest_sha, str(checked)
            except DriverError:
                pass
        reason = f"visloc child manifest did not prove success (returncode={mapper.get('returncode') if isinstance(mapper, Mapping) else None})"
        return [{"id": cells[0], "status": "dnf", "reason": reason}], manifest, manifest_sha, str(checked)
    results = manifest.get("results")
    if not isinstance(results, Mapping) or len(cells) != 2:
        return [{"id": cell, "status": "dnf", "reason": "COLMAP manifest.results missing"} for cell in cells], manifest, manifest_sha, str(checked)
    extracted: list[dict[str, Any]] = []
    for engine, cell in zip(("incremental", "global"), cells):
        row = results.get(engine)
        if not isinstance(row, Mapping) or row.get("status") != "success" or row.get("registered_images") != frames or not isinstance(row.get("points3d"), int) or row.get("points3d") <= 0:
            extracted.append({"id": cell, "status": "dnf", "reason": f"COLMAP {engine} result is missing/partial"})
        else:
            model = row.get("model")
            try:
                model_path = v7.candidate_path(model, root, f"v9 COLMAP {engine} model")
                for name in ("cameras.txt", "images.txt", "points3D.txt"):
                    v8._nonempty(model_path / name, root, f"v9 COLMAP {engine} {name}")
                extracted.append({"id": cell, "status": "success", "reason": None})
            except (DriverError, TypeError):
                extracted.append({"id": cell, "status": "dnf", "reason": f"COLMAP {engine} model evidence is invalid"})
    return extracted, manifest, manifest_sha, str(checked)


def import_v7_result(root: Path, *, prior_ledger: Path, v7_result: Path, runset: Path, output_result: Path, output_ledger: Path) -> tuple[Path, Path]:
    root = v7.require_e_root(root)
    ledger, prior_sha, prior_path = _validate_prior(root, prior_ledger)
    next_index = len(ledger["results"]) + 1
    invocation, frames = _expected(root, runset, next_index)
    runset_path = v7.require_regular_candidate_file(runset, root, "v9 runset")
    runset_sha = v7.validate_sidecar(runset_path, root, "v9 runset").upper()
    source, source_sha, source_path = _load(v7_result, root, f"v9 sealed v7 result {next_index}")
    if ledger.get("runset_sha256", "").upper() != runset_sha:
        transition = source.get("runset_transition") if isinstance(source, Mapping) else None
        if not isinstance(transition, Mapping) or transition.get("from_runset_sha256", "").upper() != str(ledger.get("runset_sha256", "")).upper() or transition.get("to_runset_sha256", "").upper() != runset_sha:
            raise DriverError("v9 prior ledger/runset mismatch without an immutable V3-to-V4 transition")
    if source.get("schema") not in {v7.RESULT_SCHEMA, V10_RESULT_SCHEMA} or source.get("ambient_policy") not in {AMBIENT_POLICY, "relaxed_recorded_robosim"} or source.get("invocation_index") != next_index or source.get("invocation") != invocation.get("id") or source.get("result_cells") != list(invocation.get("result_cells", [])) or source.get("runset_sha256", "").upper() != runset_sha or source.get("source_sha256", "").upper() != EXPECTED_SOURCE_SHA256 or source.get("protocol_sha256", "").upper() != EXPECTED_PROTOCOL_SHA256 or source.get("terminal") is not True:
        raise DriverError("v9 sealed v7 result identity/binding mismatch")
    for key in ("gt_opened", "ground_truth_read", "ground_truth_materialized", "ground_truth_argument_present_anywhere"):
        if source.get(key) is not False:
            raise DriverError(f"v9 sealed v7 result {key} is not false")
    prior_paths = {str(v7.candidate_path(item["result_path"], root, "v9 prior path")).lower() for item in ledger["results"]}
    if str(source_path).lower() in prior_paths or str(source_path).lower() in {str(v7.candidate_path(prior_path, root, "v9 prior ledger")).lower()}:
        raise DriverError("v9 sealed result aliases prior evidence")
    manifest_claim = source.get("manifest")
    manifest_path = None
    if isinstance(manifest_claim, Mapping) and manifest_claim.get("path") is not None:
        manifest_path = v7.require_regular_candidate_file(manifest_claim["path"], root, "v9 sealed child manifest")
        if str(manifest_claim.get("sha256", "")).upper() != v7.digest(manifest_path).upper():
            raise DriverError("v9 sealed child manifest SHA mismatch")
        expected_output = v7.candidate_path(invocation.get("output"), root, "v9 invocation output")
        if manifest_path.parent != expected_output:
            raise DriverError("v9 child manifest is outside declared output")
    cell_results, manifest, manifest_sha, manifest_text_path = _manifest_cells(root, manifest_path, invocation, frames)
    status = "dnf" if any(item["status"] == "dnf" for item in cell_results) else "success"
    result_output = v7.candidate_path(output_result, root, "v9 normalized result")
    ledger_output = v7.candidate_path(output_ledger, root, "v9 cumulative ledger")
    prior_artifacts = {prior_path, prior_path.with_name(prior_path.name + ".sha256")}
    for prior_record in ledger["results"]:
        prior_result_path = v7.candidate_path(prior_record["result_path"], root, "v9 prior result")
        prior_artifacts.update({prior_result_path, prior_result_path.with_name(prior_result_path.name + ".sha256")})
    if result_output in prior_artifacts or ledger_output in prior_artifacts or result_output == ledger_output or ledger_output == prior_path:
        raise DriverError("v9 output aliases immutable prior chain")
    normalized = {"schema": RESULT_SCHEMA, "chain_schema": CHAIN_SCHEMA, "ambient_policy": AMBIENT_POLICY, "status": status, "terminal": True, "attempt_terminal": True, "mapping_started": source.get("mapping_started") is True, "invocation_index": next_index, "invocation": invocation["id"], "engine": invocation["engine"], "sequence": invocation["sequence"], "result_cells": list(invocation["result_cells"]), "cell_results": cell_results, "runset_sha256": runset_sha, "source_sha256": EXPECTED_SOURCE_SHA256, "protocol_sha256": EXPECTED_PROTOCOL_SHA256, "gt_opened": False, "ground_truth_read": False, "ground_truth_materialized": False, "ground_truth_argument_present_anywhere": False, "manifest": {"ambient_policy": AMBIENT_POLICY, "path": manifest_text_path, "sha256": manifest_sha}, "provenance": {"source_schema": source.get("schema"), "source_result_path": str(source_path), "source_result_sha256": source_sha, "source_ledger_path": str(prior_path), "source_ledger_sha256": prior_sha}, "finished_utc": _now()}
    v8.reject_gt(normalized, "v9 normalized result")
    result_sha = v7.atomic_json(result_output, normalized, root, replace=False)
    records = list(ledger["results"]) + [{"schema": RESULT_SCHEMA, "chain_schema": CHAIN_SCHEMA, "invocation_index": next_index, "invocation": invocation["id"], "result_cells": list(invocation["result_cells"]), "status": status, "result_path": str(result_output), "result_sha256": result_sha, "provenance": normalized["provenance"]}]
    cell_map = dict(ledger["cells"])
    for item in cell_results:
        cell_map[item["id"]] = {"ambient_policy": AMBIENT_POLICY, "invocation_index": next_index, "result_path": str(result_output), "result_sha256": result_sha, "status": item["status"]}
    cumulative = {**ledger, "results": records, "cells": cell_map, "denominator": {"total_cells": TOTAL_RESULT_CELLS, "completed_cells": list(cell_map), "completed_count": len(cell_map), "remaining_cells": [cell for cell in RESULT_CELLS if cell not in cell_map], "remaining_count": TOTAL_RESULT_CELLS - len(cell_map)}, "provenance": {"parent_ledger_path": str(prior_path), "parent_ledger_sha256": prior_sha, "imported_source_path": str(source_path), "imported_source_sha256": source_sha}, "updated_utc": _now()}
    v7.atomic_json(ledger_output, cumulative, root, replace=False)
    return result_output, ledger_output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--prior-ledger", type=Path, required=True)
    parser.add_argument("--v7-result", type=Path, required=True)
    parser.add_argument("--runset", type=Path, required=True)
    parser.add_argument("--output-result", type=Path, required=True)
    parser.add_argument("--output-ledger", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result, ledger = import_v7_result(args.candidate_root, prior_ledger=args.prior_ledger, v7_result=args.v7_result, runset=args.runset, output_result=args.output_result, output_ledger=args.output_ledger)
    print(json.dumps({"schema": CHAIN_SCHEMA, "status": "imported", "result": str(result), "ledger": str(ledger)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DRIVER_VERSION", "RESULT_SCHEMA", "LEDGER_SCHEMA", "CHAIN_SCHEMA", "initialize_from_v8", "import_v7_result", "parse_args", "main"]
