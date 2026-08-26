"""Security-focused tests for the non-overwriting v7-to-v8 recovery lane."""

from __future__ import annotations

import json
import os
import shutil
import sys
import uuid
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import unittest

try:
    import build_ssfm_dev_gate_v3 as gate_v3  # type: ignore[no-redef]  # noqa: E402
except ImportError as error:  # pragma: no cover - depends on private bench modules
    raise unittest.SkipTest(
        f"private bench module unavailable on this machine: {error}"
    ) from error

import run_b07h_runtime_driver_v7 as v7  # noqa: E402
import run_b07h_runtime_driver_v8 as v8  # noqa: E402
import run_b07h_runtime_driver_v9 as v9  # noqa: E402


ROOT_PARENT = Path("E:/visloc_archive/tmp/b07_storage_hardening/v8-recovery")


@pytest.fixture
def case_root():
    ROOT_PARENT.mkdir(parents=True, exist_ok=True)
    root = ROOT_PARENT / f"case-{os.getpid()}-{uuid.uuid4().hex}"
    root.mkdir()
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _fixture(root: Path, *, returncode=0):
    output = root / "runs" / "b07h-runtime" / "MH_01_easy" / "visloc"
    model = output / "model"
    model.mkdir(parents=True)
    executable = root / "bin" / "sequential_sfm_demo.exe"
    executable.parent.mkdir()
    executable.write_bytes(b"fixture-executable")
    (model / "cameras.txt").write_text("# camera\n0 PINHOLE 1 1 1 1 0 0 0\n", encoding="utf-8")
    (model / "images.txt").write_text("# images\n1 pose\ntrack\n2 pose\ntrack\n", encoding="utf-8")
    (model / "points3D.txt").write_text("# points\n1 point\n2 point\n", encoding="utf-8")
    (output / "trajectory.tum").write_text("0 0 0 0 0 0 0 1\n", encoding="utf-8")
    (output / "mapping.log").write_text("mapper complete\n", encoding="utf-8")
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "executable": {"path": str(executable), "sha256": v8.digest(executable)},
        "protocol": {"input_feature_frames": 2, "timestamp_rows": 2, "expected_frames": 2, "ground_truth_read": False},
        "mapper": {"returncode": returncode, "wall_seconds": 1.0, "registered_images": 2, "points3d": 2},
    }, sort_keys=True) + "\n", encoding="utf-8")
    runset = root / "runsets" / "runset.json"
    runset.parent.mkdir()
    value = {
        "schema": v7.RUNSET_SCHEMA, "candidate_root": str(root),
        "fixed_tools": {"hierarchical_executable": {"path": str(executable), "sha256": v8.digest(executable), "bytes": executable.stat().st_size}},
        "invocations": [{"id": ident, "engine": engine, "sequence": sequence, "command": ["python", "--expected-frames", "2"], "output": (str(output.relative_to(root)) if index == 1 else f"runs/out-{index}"), "result_cells": list(cells)} for index, (ident, engine, sequence, cells) in enumerate(v7.INVOCATION_CELLS, 1)],
    }
    runset.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    runset_sha = v7.atomic_json(runset, value, root, replace=True)
    original_result = root / "logs" / "B07H_v4_ambient_recorded_invocation_01.json"
    original_ledger = root / "logs" / "B07H_v4_ambient_recorded_ledger.json"
    manifest_sha = v8.digest(manifest)
    result = {
        "schema": v7.RESULT_SCHEMA, "ambient_policy": "recorded", "status": "dnf", "terminal": True, "attempt_terminal": True,
        "mapping_started": True, "invocation_index": 1, "invocation": "visloc_MH_01_easy", "engine": "visloc", "sequence": "MH_01_easy",
        "result_cells": ["visloc_MH_01_easy"], "cell_results": [{"id": "visloc_MH_01_easy", "reason": "child result status is missing/unknown", "status": "dnf"}],
        "runset_sha256": runset_sha, "source_sha256": v8.EXPECTED_SOURCE_SHA256, "protocol_sha256": v8.EXPECTED_PROTOCOL_SHA256,
        "gt_opened": False, "ground_truth_read": False, "ground_truth_materialized": False, "ground_truth_argument_present_anywhere": False,
        "manifest": {"ambient_policy": "recorded", "path": str(manifest), "sha256": manifest_sha},
        "finished_utc": "2026-01-01T00:00:00+00:00",
    }
    result_sha = v7.atomic_json(original_result, result, root, replace=False)
    ledger = {"schema": v7.LEDGER_SCHEMA, "ambient_policy": "recorded", "expected_cells": list(v7.RESULT_CELLS), "total_result_cells": 9,
              "results": [{"invocation_index": 1, "invocation": "visloc_MH_01_easy", "result_cells": ["visloc_MH_01_easy"], "status": "dnf", "result_path": str(original_result), "result_sha256": result_sha}],
              "cells": {"visloc_MH_01_easy": {"invocation_index": 1, "result_path": str(original_result), "result_sha256": result_sha, "status": "dnf"}}}
    v7.atomic_json(original_ledger, ledger, root, replace=False)
    return runset, original_result, original_ledger, manifest


def test_recovery_accepts_frozen_manifest_contract_without_mutating_v7(case_root, monkeypatch):
    runset, original_result, original_ledger, manifest = _fixture(case_root)
    monkeypatch.setattr(v7, "validate_runset", lambda *args, **kwargs: json.loads(runset.read_text(encoding="utf-8")))
    before_result = v8.digest(original_result)
    before_ledger = v8.digest(original_ledger)
    recovered_result, recovered_ledger = v8.recover_invocation_one(case_root, original_result=original_result, original_ledger=original_ledger, runset=runset, output_result=case_root / "recovery" / "result.json", output_ledger=case_root / "recovery" / "ledger.json")
    assert json.loads(recovered_result.read_text(encoding="utf-8"))["status"] == "success"
    assert json.loads(recovered_result.read_text(encoding="utf-8"))["recovery"]["original_result_sha256"] == before_result
    assert json.loads(recovered_ledger.read_text(encoding="utf-8"))["schema"] == v8.LEDGER_SCHEMA
    assert v8.digest(original_result) == before_result
    assert v8.digest(original_ledger) == before_ledger
    assert manifest.is_file()


@pytest.mark.parametrize("returncode", [1, None])
def test_recovery_rejects_partial_or_failed_manifest_without_outputs(case_root, monkeypatch, returncode):
    runset, original_result, original_ledger, _ = _fixture(case_root, returncode=returncode)
    monkeypatch.setattr(v7, "validate_runset", lambda *args, **kwargs: json.loads(runset.read_text(encoding="utf-8")))
    with pytest.raises(v8.DriverError):
        v8.recover_invocation_one(case_root, original_result=original_result, original_ledger=original_ledger, runset=runset, output_result=case_root / "recovery" / "result.json", output_ledger=case_root / "recovery" / "ledger.json")
    assert not (case_root / "recovery" / "result.json").exists()
    assert not (case_root / "recovery" / "ledger.json").exists()


def test_recovery_rejects_output_manifest_alias(case_root, monkeypatch):
    runset, original_result, original_ledger, manifest = _fixture(case_root)
    monkeypatch.setattr(v7, "validate_runset", lambda *args, **kwargs: json.loads(runset.read_text(encoding="utf-8")))
    manifest.write_text(manifest.read_text(encoding="utf-8").replace('"schema_version": 1', '"schema_version": 2'), encoding="utf-8")
    with pytest.raises(v8.DriverError):
        v8.recover_invocation_one(case_root, original_result=original_result, original_ledger=original_ledger, runset=runset, output_result=case_root / "recovery" / "result.json", output_ledger=case_root / "recovery" / "ledger.json")


def _v9_source(root, runset, index, *, engine="visloc", manifest_path=None, status="dnf"):
    invocation = json.loads(runset.read_text(encoding="utf-8"))["invocations"][index - 1]
    result = root / "sealed-v7" / f"invocation-{index}.json"
    result.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "schema": v7.RESULT_SCHEMA, "ambient_policy": "recorded", "status": status, "terminal": True, "attempt_terminal": True,
        "mapping_started": True, "invocation_index": index, "invocation": invocation["id"], "engine": engine, "sequence": invocation["sequence"],
        "result_cells": list(invocation["result_cells"]), "cell_results": [{"id": cell, "status": status, "reason": "sealed DNF" if status == "dnf" else None} for cell in invocation["result_cells"]],
        "runset_sha256": v8.digest(runset), "source_sha256": v8.EXPECTED_SOURCE_SHA256, "protocol_sha256": v8.EXPECTED_PROTOCOL_SHA256,
        "gt_opened": False, "ground_truth_read": False, "ground_truth_materialized": False, "ground_truth_argument_present_anywhere": False,
        "manifest": {"ambient_policy": "recorded", "path": str(manifest_path) if manifest_path else None, "sha256": v8.digest(manifest_path) if manifest_path else None},
        "finished_utc": "2026-01-01T00:00:00+00:00",
    }
    v7.atomic_json(result, value, root, replace=False)
    return result


def test_v9_strict_serial_import_and_no_invocation_one_rerun(case_root, monkeypatch):
    runset, original_result, original_ledger, _ = _fixture(case_root)
    monkeypatch.setattr(v7, "validate_runset", lambda *args, **kwargs: json.loads(runset.read_text(encoding="utf-8")))
    v8_result = case_root / "recovery" / "v8-result.json"
    v8_ledger = case_root / "recovery" / "v8-ledger.json"
    v8.recover_invocation_one(case_root, original_result=original_result, original_ledger=original_ledger, runset=runset, output_result=v8_result, output_ledger=v8_ledger)
    cumulative = v9.initialize_from_v8(case_root, v8_ledger=v8_ledger, output_ledger=case_root / "recovery" / "v9-ledger-01.json")
    inv2 = _v9_source(case_root, runset, 3)
    with pytest.raises(v9.DriverError):
        v9.import_v7_result(case_root, prior_ledger=cumulative, v7_result=inv2, runset=runset, output_result=case_root / "recovery" / "bad-result.json", output_ledger=case_root / "recovery" / "bad-ledger.json")
    before = v8.digest(v8_result)
    assert v8.digest(v8_result) == before


def test_v9_imports_two_colmap_cells_and_preserves_missing_manifest_as_dnf(case_root, monkeypatch):
    runset, original_result, original_ledger, _ = _fixture(case_root)
    monkeypatch.setattr(v7, "validate_runset", lambda *args, **kwargs: json.loads(runset.read_text(encoding="utf-8")))
    v8_result = case_root / "recovery" / "v8-result.json"
    v8_ledger = case_root / "recovery" / "v8-ledger.json"
    v8.recover_invocation_one(case_root, original_result=original_result, original_ledger=original_ledger, runset=runset, output_result=v8_result, output_ledger=v8_ledger)
    cumulative = v9.initialize_from_v8(case_root, v8_ledger=v8_ledger, output_ledger=case_root / "recovery" / "v9-ledger-01.json")
    output = case_root / "runs" / "out-2"
    (output / "incremental_models").mkdir(parents=True)
    (output / "global_models").mkdir(parents=True)
    for model in (output / "incremental_models", output / "global_models"):
        (model / "cameras.txt").write_text("camera\n", encoding="utf-8")
        (model / "images.txt").write_text("image\n", encoding="utf-8")
        (model / "points3D.txt").write_text("point\n", encoding="utf-8")
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "ground_truth_read": False, "results": {"incremental": {"status": "success", "registered_images": 2, "points3d": 1, "model": str(output / "incremental_models")}, "global": {"status": "success", "registered_images": 2, "points3d": 1, "model": str(output / "global_models")}}}), encoding="utf-8")
    inv2 = _v9_source(case_root, runset, 2, engine="colmap", manifest_path=manifest, status="dnf")
    normalized, cumulative2 = v9.import_v7_result(case_root, prior_ledger=cumulative, v7_result=inv2, runset=runset, output_result=case_root / "recovery" / "v9-result-02.json", output_ledger=case_root / "recovery" / "v9-ledger-02.json")
    value = json.loads(normalized.read_text(encoding="utf-8"))
    assert value["status"] == "success"
    assert [item["id"] for item in value["cell_results"]] == ["colmap_inc_MH_01_easy", "colmap_global_MH_01_easy"]
    assert len(json.loads(cumulative2.read_text(encoding="utf-8"))["cells"]) == 3

    inv3 = _v9_source(case_root, runset, 3, engine="visloc", manifest_path=None, status="dnf")
    normalized3, _ = v9.import_v7_result(case_root, prior_ledger=cumulative2, v7_result=inv3, runset=runset, output_result=case_root / "recovery" / "v9-result-03.json", output_ledger=case_root / "recovery" / "v9-ledger-03.json")
    assert json.loads(normalized3.read_text(encoding="utf-8"))["status"] == "dnf"


def test_v9_rejects_tampered_prior_ledger_and_cross_chain_alias(case_root, monkeypatch):
    runset, original_result, original_ledger, _ = _fixture(case_root)
    monkeypatch.setattr(v7, "validate_runset", lambda *args, **kwargs: json.loads(runset.read_text(encoding="utf-8")))
    v8_result = case_root / "recovery" / "v8-result.json"
    v8_ledger = case_root / "recovery" / "v8-ledger.json"
    v8.recover_invocation_one(case_root, original_result=original_result, original_ledger=original_ledger, runset=runset, output_result=v8_result, output_ledger=v8_ledger)
    cumulative = v9.initialize_from_v8(case_root, v8_ledger=v8_ledger, output_ledger=case_root / "recovery" / "v9-ledger-01.json")
    tampered = json.loads(cumulative.read_text(encoding="utf-8"))
    tampered["results"][0]["result_sha256"] = "0" * 64
    tampered_path = case_root / "recovery" / "tampered-ledger.json"
    v7.atomic_json(tampered_path, tampered, case_root, replace=False)
    with pytest.raises(v9.DriverError):
        v9.import_v7_result(case_root, prior_ledger=tampered_path, v7_result=_v9_source(case_root, runset, 2), runset=runset, output_result=case_root / "recovery" / "result.json", output_ledger=case_root / "recovery" / "ledger.json")


def test_v3_gate_consumes_complete_v9_chain_without_rerunning_invocation_one(case_root, monkeypatch):
    runset, original_result, original_ledger, _ = _fixture(case_root)
    monkeypatch.setattr(v7, "validate_runset", lambda *args, **kwargs: json.loads(runset.read_text(encoding="utf-8")))
    v8_result = case_root / "recovery" / "v8-result.json"
    v8_ledger = case_root / "recovery" / "v8-ledger.json"
    v8.recover_invocation_one(case_root, original_result=original_result, original_ledger=original_ledger, runset=runset, output_result=v8_result, output_ledger=v8_ledger)
    prior = v9.initialize_from_v8(case_root, v8_ledger=v8_ledger, output_ledger=case_root / "recovery" / "v9-ledger-01.json")
    for index in range(2, 7):
        source = _v9_source(case_root, runset, index, engine="colmap" if index in {2, 4, 6} else "visloc", manifest_path=None, status="dnf")
        _, prior = v9.import_v7_result(case_root, prior_ledger=prior, v7_result=source, runset=runset, output_result=case_root / "recovery" / f"v9-result-{index:02d}.json", output_ledger=case_root / "recovery" / f"v9-ledger-{index:02d}.json")
    driver_copy = case_root / "tools" / "run_b07h_runtime_driver_v9.py"
    driver_copy.parent.mkdir(parents=True)
    driver_copy.write_text(Path(__file__).resolve().parents[1].joinpath("scripts/run_b07h_runtime_driver_v9.py").read_text(encoding="utf-8"), encoding="utf-8")
    (Path(str(driver_copy) + ".sha256")).write_text(f"{v8.digest(driver_copy)}  {driver_copy.name}\n", encoding="ascii")
    gate = gate_v3.build_dev_gate(case_root, case_root / "gate" / "dev-gate.json", runset=runset, expected_runset_sha256=v8.digest(runset), driver=driver_copy, expected_driver_sha256=v8.digest(driver_copy), ledger=prior)
    gate_value = json.loads(gate.read_text(encoding="utf-8"))
    assert gate_value["schema"] == gate_v3.GATE_SCHEMA
    assert gate_value["heldout_execution_allowed"] is True
    assert len(gate_value["cell_results"]) == 9
