"""Focused tests for the isolated B07-H v7 ambient-recorded lane."""

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

import pytest  # noqa: E402

import build_b07h_runtime_runset_v3 as runset_builder  # noqa: E402
import run_b07h_runtime_driver_v7 as driver  # noqa: E402


ARCHIVE = Path("E:/visloc_archive")
FIXTURE_PARENT = ARCHIVE / "tmp" / "b07_storage_hardening" / "ambient-recorded-v7"


@pytest.fixture
def case_root():
    FIXTURE_PARENT.mkdir(parents=True, exist_ok=True)
    root = FIXTURE_PARENT / f"case-{os.getpid()}-{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _process(*, cpu=0.0, search=0.0, targets=None):
    return {
        "total_processor_percent": cpu,
        "search_indexer_percent": search,
        "target_processes": list(targets or []),
    }


def _wsl(*, targets=None):
    return {"status": "idle", "target_processes": list(targets or [])}


def _gpu(*, utilization=0.0, memory=0.0):
    return {"available": True, "utilization_percent": utilization, "memory_used_mib": memory}


def test_v7_contract_is_disjoint_and_preserves_exact_order():
    assert driver.RUNSET_SCHEMA == "B07H_GT_FREE_RUNTIME_RUNSET_V3"
    assert driver.RESULT_SCHEMA != "B07H_RUNTIME_DRIVER_RESULT_V3"
    assert driver.LEDGER_SCHEMA != "B07H_RUNTIME_DRIVER_LEDGER_V3"
    assert driver.AMBIENT_POLICY == "recorded"
    assert [item[0] for item in driver.INVOCATION_CELLS] == list(driver.SERIAL_ORDER)
    assert [cell for item in driver.INVOCATION_CELLS for cell in item[3]] == list(driver.RESULT_CELLS)
    assert driver.TOTAL_INVOCATIONS == 6
    assert driver.TOTAL_RESULT_CELLS == 9
    assert "searchindexer" not in driver.TARGET_PROCESS_NAMES


def test_high_searchindexer_and_cpu_noise_is_recorded_without_start_block(case_root):
    workspace = case_root / "workspace"
    workspace.mkdir()
    history = case_root / "logs" / "ambient.jsonl"
    process = lambda: _process(cpu=195.0, search=195.0)
    window = driver.record_ambient_window(
        case_root,
        history,
        workspace_root=workspace,
        samples=3,
        sample_seconds=0.0,
        process_sampler=process,
        wsl_sampler=lambda: _wsl(),
        gpu_sampler=lambda: _gpu(utilization=99.0, memory=3072.0),
        free_bytes_fn=lambda _root: driver.STOP_FREE_BYTES,
    )
    assert window["ambient_policy"] == "recorded"
    assert window["samples"] == 3
    assert window["start_allowed"] is True
    assert all(item["checks"]["cpu_settled"] is False for item in window["observations"])
    assert all(item["checks"]["search_settled"] is False for item in window["observations"])
    assert all(item["checks"]["gpu_settled"] is False for item in window["observations"])
    assert all(item["checks"]["target_processes_clear"] for item in window["observations"])
    events = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 3
    assert all(item["ambient_policy"] == "recorded" for item in events)
    manifest = history.with_name(history.name + ".manifest")
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_value["ambient_policy"] == "recorded"
    assert driver.validate_sidecar(history, case_root, "ambient")
    assert not list(case_root.rglob("target"))
    assert not list(case_root.rglob("temp"))
    assert not list(case_root.rglob("__pycache__"))


@pytest.mark.parametrize(
    "process,wsl,free,workspace_contamination,expected",
    [
        (_process(targets=[{"name": "cargo", "pid": 11}]), _wsl(), driver.STOP_FREE_BYTES, False, "target_processes"),
        (_process(), _wsl(targets=[{"name": "robosim", "pid": 12}]), driver.STOP_FREE_BYTES, False, "target_processes"),
        (_process(), _wsl(), driver.STOP_FREE_BYTES - 1, False, "e_free_threshold"),
        (_process(), _wsl(), driver.STOP_FREE_BYTES, True, "c_workspace"),
    ],
)
def test_hard_start_gates_block_but_keep_noise_non_gating(case_root, process, wsl, free, workspace_contamination, expected):
    workspace = case_root / "workspace"
    workspace.mkdir()
    if workspace_contamination:
        (workspace / "target").mkdir()
    observation = driver.ambient_sample(
        case_root,
        workspace,
        process_sampler=lambda: process,
        wsl_sampler=lambda: wsl,
        gpu_sampler=lambda: _gpu(utilization=100.0),
        free_bytes_fn=lambda _root: free,
    )
    assert observation["ambient_policy"] == "recorded"
    assert observation["start_allowed"] is False
    assert expected in [
        "target_processes" if not observation["checks"]["target_processes_clear"] else "",
        "c_workspace" if not observation["checks"]["c_workspace_clean"] else "",
        "e_free_threshold" if not observation["checks"]["e_free_threshold"] else "",
    ]
    assert observation["checks"]["gpu_settled"] is False
    assert not list(case_root.glob("*.pyc"))


def test_deferred_history_and_ledger_are_v7_policy_bound_and_never_v6(case_root):
    history = case_root / "logs" / "deferred.jsonl"
    deferred = driver.append_deferred(
        case_root,
        history,
        {"invocation": driver.INVOCATION_CELLS[0][0], "deferred_cells": list(driver.INVOCATION_CELLS[0][3]), "reason": "hard start blocker"},
    )
    assert deferred["ambient_policy"] == "recorded"
    assert deferred["status"] == "deferred"
    ledger = driver.read_ledger(None, case_root)
    assert ledger["ambient_policy"] == "recorded"
    assert driver.LEDGER_RELATIVE_PATH.name == "B07H_v4_ambient_recorded_ledger.json"
    assert not (case_root / "logs" / "B07H_v3_ledger.json").exists()


def test_terminal_result_requires_recorded_policy_and_writes_separate_ledger(case_root):
    result = case_root / "runs" / "ambient" / "invocation-01.json"
    payload = {
        "ambient_policy": "recorded",
        "status": "success",
        "mapping_started": True,
        "invocation_index": 1,
        "invocation": driver.INVOCATION_CELLS[0][0],
        "engine": driver.INVOCATION_CELLS[0][1],
        "sequence": driver.INVOCATION_CELLS[0][2],
        "result_cells": list(driver.INVOCATION_CELLS[0][3]),
        "cell_results": [{"id": driver.RESULT_CELLS[0], "status": "success"}],
        "runset_sha256": "A" * 64,
        "source_sha256": driver.EXPECTED_SOURCE_SHA256,
        "protocol_sha256": driver.EXPECTED_PROTOCOL_SHA256,
        "gt_opened": False,
        "ground_truth_read": False,
        "ground_truth_materialized": False,
        "ground_truth_argument_present_anywhere": False,
        "manifest": {"ambient_policy": "recorded", "path": None, "sha256": None},
    }
    checked = driver.record_result(case_root, result, payload)
    assert checked["schema"] == driver.RESULT_SCHEMA
    assert checked["ambient_policy"] == "recorded"
    ledger_path = case_root / driver.LEDGER_RELATIVE_PATH
    assert ledger_path.is_file()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["ambient_policy"] == "recorded"
    assert ledger["results"][0]["ambient_policy"] == "recorded"
    assert not (case_root / "logs" / "B07H_v3_ledger.json").exists()
    bad = dict(payload)
    bad["ambient_policy"] = "strict_quiet"
    with pytest.raises(driver.DriverError):
        driver.record_result(case_root, case_root / "runs" / "ambient" / "bad.json", bad)


def _make_v2_fixture(root: Path) -> Path:
    active = Path("E:/visloc_archive/sota_slice13_replay_protocol_r5_20260816")
    source = root / "pipelines" / "slam" / "src" / "hierarchical_sfm.rs"
    protocol = root / "B07E_DEV_REPLAY_PROTOCOL.json"
    source.parent.mkdir(parents=True)
    shutil.copyfile(active / "pipelines" / "slam" / "src" / "hierarchical_sfm.rs", source)
    shutil.copyfile(active / "B07E_DEV_REPLAY_PROTOCOL.json", protocol)
    tool_names = {
        "python": "python.exe",
        "hierarchical_runner": "hierarchical_runner.py",
        "hierarchical_executable": "hierarchical_executable.exe",
        "colmap_runner": "colmap_runner.py",
        "colmap": "colmap.exe",
    }
    fixed_tools = {}
    for key, name in tool_names.items():
        tool = root / "tools" / name
        tool.parent.mkdir(parents=True, exist_ok=True)
        tool.write_bytes((key + " fixture\n").encode("ascii"))
        fixed_tools[key] = {"path": str(tool.relative_to(root)), "sha256": driver.digest(tool), "bytes": tool.stat().st_size}
    invocations = []
    for invocation_id, engine, sequence, cells in driver.INVOCATION_CELLS:
        command = ["tools/python.exe", "tools/hierarchical_runner.py" if engine == "visloc" else "tools/colmap_runner.py"]
        if engine == "visloc":
            command.extend(["--exe", "tools/hierarchical_executable.exe"])
        else:
            command.extend(["--protocol", "B07E_DEV_REPLAY_PROTOCOL.json", "--colmap", "tools/colmap.exe"])
        command.extend(["--out-dir", f"runs/b07h-runtime/{sequence}/{engine}"])
        invocations.append({"id": invocation_id, "engine": engine, "sequence": sequence, "command": command, "output": f"runs/b07h-runtime/{sequence}/{engine}", "result_cells": list(cells), "ground_truth_argument_present": False})
    value = {
        "schema": "B07H_GT_FREE_RUNTIME_RUNSET_V2",
        "status": "fixed_preflight_only",
        "candidate_root": str(root),
        "supersedes_schema": "B07H_GT_FREE_RUNTIME_RUNSET_V1",
        "supersedes_sha256": "A" * 64,
        "ambient_oracle": {"path": "scripts/run_b07h_runtime_driver_v5.py", "sha256": "B" * 64, "bytes": 1, "sidecar": "scripts/run_b07h_runtime_driver_v5.py.sha256"},
        "source": {"path": str(source.relative_to(root)), "sha256": driver.EXPECTED_SOURCE_SHA256},
        "protocol": {"path": str(protocol.relative_to(root)), "sha256": driver.EXPECTED_PROTOCOL_SHA256},
        "fixed_tools": fixed_tools,
        "serial_order": ["MH_01_easy visloc", "MH_01_easy colmap (incremental + global cells)", "MH_03_medium visloc", "MH_03_medium colmap (incremental + global cells)", "MH_05_difficult visloc", "MH_05_difficult colmap (incremental + global cells)"],
        "runtime_policy": {"mapping_executed": False, "gt_opened": False, "performance_claim": False, "serial_only": True, "total_invocations": 6, "total_result_cells": 9, "ground_truth_argument_present_anywhere": False, "output_paths_preflight_absent": True},
        "storage_gate": {"free_bytes_at_preflight": driver.STOP_FREE_BYTES, "free_gib_at_preflight": 250, "stop_threshold_bytes": driver.STOP_FREE_BYTES, "stop_threshold_gib": 250, "check_before_each_invocation": True, "unstarted_cells_if_below_threshold": "DNF and preserve denominator 9"},
        "invocations": invocations,
        "ground_truth_read": False,
        "ground_truth_materialized": False,
        "ground_truth_argument_present_anywhere": False,
    }
    path = root / "runsets" / "B07H_GT_FREE_RUNTIME_RUNSET_V2.json"
    driver.atomic_json(path, value, root, replace=False)
    return path


def test_v3_builder_removes_v6_oracle_and_binds_ambient_recording(case_root):
    frozen = _make_v2_fixture(case_root)
    output = runset_builder.build_runset_v3(case_root, frozen, Path("runsets/B07H_GT_FREE_RUNTIME_RUNSET_V3.json"))
    value = driver.validate_runset(output, case_root, driver.digest(output))
    assert value["schema"] == driver.RUNSET_SCHEMA
    assert value["supersedes_schema"] == "B07H_GT_FREE_RUNTIME_RUNSET_V2"
    assert value["ambient_policy"] == "recorded"
    assert value["ambient_recording"]["finite_window"] is True
    assert "ambient_oracle" not in value
    assert value["storage_gate"]["stop_threshold_gib"] == 250
    assert [item["id"] for item in value["invocations"]] == list(driver.SERIAL_ORDER)
    assert [cell for item in value["invocations"] for cell in item["result_cells"]] == list(driver.RESULT_CELLS)


def test_main_launches_with_high_searchindexer_and_records_ambient_window(case_root, monkeypatch):
    frozen = _make_v2_fixture(case_root)
    runset = runset_builder.build_runset_v3(case_root, frozen, Path("runsets/B07H_GT_FREE_RUNTIME_RUNSET_V3.json"))
    calls = []

    class Completed:
        def wait(self):
            return 1  # absent child manifest becomes an explicit DNF cell

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return Completed()

    monkeypatch.setattr(driver.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(driver, "require_c_workspace_clean", lambda: {"clean": True, "forbidden": []})
    monkeypatch.setattr(driver, "c_workspace_state", lambda _root=driver.DEFAULT_C_WORKSPACE: {"clean": True, "forbidden": []})
    monkeypatch.setattr(driver, "_default_process_sampler", lambda: _process(cpu=195.0, search=195.0))
    monkeypatch.setattr(driver, "_default_wsl_sampler", lambda: _wsl())
    monkeypatch.setattr(driver, "_default_gpu_sampler", lambda: _gpu(utilization=100.0, memory=4096.0))
    monkeypatch.setattr(driver, "_free_bytes", lambda _root: driver.STOP_FREE_BYTES)
    result = driver.main([
        "--runset", str(runset),
        "--candidate-root", str(case_root),
        "--expected-runset-sha256", driver.digest(runset),
        "--invocation-index", "1",
        "--runtime-temp", "temp/v7-runtime",
        "--ambient-window-seconds", "0",
        "--sample-seconds", "0",
        "--ambient-samples", "2",
    ])
    assert result == 2
    assert len(calls) == 1
    history = case_root / "logs" / "B07H_v4_ambient_recorded_invocation_01.jsonl"
    events = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 2
    assert all(item["ambient_policy"] == "recorded" for item in events)
    assert all(item["noise"]["search_indexer_percent"] == 195.0 for item in events)
    assert (case_root / driver.LEDGER_RELATIVE_PATH).is_file()
    assert not (case_root / "logs" / "B07H_v3_ledger.json").exists()


def _built_v3_fixture(case_root: Path) -> Path:
    frozen = _make_v2_fixture(case_root)
    return runset_builder.build_runset_v3(case_root, frozen, Path("runsets/B07H_GT_FREE_RUNTIME_RUNSET_V3.json"))


def test_main_rejects_colliding_evidence_sidecars_output_and_runtime_paths(case_root, monkeypatch):
    runset = _built_v3_fixture(case_root)
    monkeypatch.setattr(driver, "require_c_workspace_clean", lambda: {"clean": True, "forbidden": []})
    common = [
        "--runset", str(runset),
        "--candidate-root", str(case_root),
        "--expected-runset-sha256", driver.digest(runset),
        "--invocation-index", "1",
        "--runtime-temp", "temp/v7-runtime",
        "--validation-only",
    ]
    collisions = [
        ("--result", "logs/same", "--history", "logs/same"),
        ("--result", "logs/evidence", "--history", "logs/evidence.sha256"),
        ("--result", "runs/b07h-runtime/MH_01_easy/visloc/result.json", "--history", "logs/history.jsonl"),
        ("--result", "logs/result.json", "--history", "logs/history.jsonl", "--runtime-temp", "logs"),
    ]
    for mutation in collisions:
        args = list(common)
        for index in range(0, len(mutation), 2):
            flag, value = mutation[index], mutation[index + 1]
            if flag == "--runtime-temp":
                args[args.index("--runtime-temp") + 1] = value
            else:
                args.extend([flag, value])
        with pytest.raises(driver.DriverError):
            driver.main(args)


def test_read_ledger_rejects_cell_reassignment_and_tampered_result_chain(case_root):
    result = case_root / "runs" / "ambient" / "invocation-01.json"
    payload = {
        "ambient_policy": "recorded",
        "status": "success",
        "mapping_started": True,
        "invocation_index": 1,
        "invocation": driver.INVOCATION_CELLS[0][0],
        "engine": driver.INVOCATION_CELLS[0][1],
        "sequence": driver.INVOCATION_CELLS[0][2],
        "result_cells": list(driver.INVOCATION_CELLS[0][3]),
        "cell_results": [{"id": driver.RESULT_CELLS[0], "status": "success"}],
        "runset_sha256": "A" * 64,
        "source_sha256": driver.EXPECTED_SOURCE_SHA256,
        "protocol_sha256": driver.EXPECTED_PROTOCOL_SHA256,
        "gt_opened": False,
        "ground_truth_read": False,
        "ground_truth_materialized": False,
        "ground_truth_argument_present_anywhere": False,
        "manifest": {"ambient_policy": "recorded", "path": None, "sha256": None},
    }
    driver.record_result(case_root, result, payload)
    ledger_path = case_root / driver.LEDGER_RELATIVE_PATH
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["cells"][driver.RESULT_CELLS[0]]["invocation_index"] = 2
    driver.atomic_json(ledger_path, ledger, case_root, replace=True)
    with pytest.raises(driver.DriverError):
        driver.read_ledger(ledger_path, case_root)
    # Restore the mapping, then rewrite the result bytes with a valid new
    # sidecar.  The ledger's old result SHA must still be required.
    ledger["cells"][driver.RESULT_CELLS[0]]["invocation_index"] = 1
    driver.atomic_json(ledger_path, ledger, case_root, replace=True)
    result_value = json.loads(result.read_text(encoding="utf-8"))
    result_value["ambient_policy"] = "recorded"
    result_value["status"] = "dnf"
    result_value["cell_results"][0] = {"id": driver.RESULT_CELLS[0], "status": "dnf", "reason": "tampered"}
    driver.atomic_json(result, result_value, case_root, replace=True)
    with pytest.raises(driver.DriverError):
        driver.read_ledger(ledger_path, case_root)


def test_invocation_two_manifest_cannot_alias_prior_result_sidecar_or_ledger(case_root):
    manifest_file = case_root / "manifests" / "child-01.json"
    manifest_file.parent.mkdir(parents=True)
    manifest_file.write_text("{}\n", encoding="utf-8")

    def payload(index, result_cells, manifest_path, manifest_sha):
        invocation_id, engine, sequence, _ = driver.INVOCATION_CELLS[index - 1]
        return {
            "ambient_policy": "recorded", "status": "success", "mapping_started": True,
            "invocation_index": index, "invocation": invocation_id, "engine": engine, "sequence": sequence,
            "result_cells": list(result_cells),
            "cell_results": [{"id": cell, "status": "success"} for cell in result_cells],
            "runset_sha256": "A" * 64, "source_sha256": driver.EXPECTED_SOURCE_SHA256,
            "protocol_sha256": driver.EXPECTED_PROTOCOL_SHA256, "gt_opened": False,
            "ground_truth_read": False, "ground_truth_materialized": False,
            "ground_truth_argument_present_anywhere": False,
            "manifest": {"ambient_policy": "recorded", "path": str(manifest_path), "sha256": manifest_sha},
        }

    result1 = case_root / "runs" / "ambient" / "invocation-01.json"
    driver.record_result(
        case_root,
        result1,
        payload(1, driver.INVOCATION_CELLS[0][3], manifest_file, driver.digest(manifest_file)),
    )
    ledger_path = case_root / driver.LEDGER_RELATIVE_PATH
    result1_sidecar = result1.with_name(result1.name + ".sha256")
    collisions = [
        (result1, driver.digest(result1)),
        (result1_sidecar, driver.digest(result1_sidecar)),
        (ledger_path, driver.digest(ledger_path)),
    ]
    for collision_path, collision_sha in collisions:
        with pytest.raises(driver.DriverError):
            driver.record_result(
                case_root,
                case_root / "runs" / "ambient" / f"invocation-02-{collision_path.name}.json",
                payload(2, driver.INVOCATION_CELLS[1][3], collision_path, collision_sha),
            )
    manifest_two = case_root / "manifests" / "child-02.json"
    manifest_two.write_text("{}\n", encoding="utf-8")
    result2 = case_root / "runs" / "ambient" / "invocation-02-valid.json"
    driver.record_result(
        case_root,
        result2,
        payload(2, driver.INVOCATION_CELLS[1][3], manifest_two, driver.digest(manifest_two)),
    )
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    result2_value = json.loads(result2.read_text(encoding="utf-8"))
    result2_value["manifest"] = {"ambient_policy": "recorded", "path": str(result1), "sha256": driver.digest(result1)}
    result2_sha = driver.atomic_json(result2, result2_value, case_root, replace=True)
    ledger["results"][1]["result_sha256"] = result2_sha
    for cell in driver.INVOCATION_CELLS[1][3]:
        ledger["cells"][cell]["result_sha256"] = result2_sha
    driver.atomic_json(ledger_path, ledger, case_root, replace=True)
    with pytest.raises(driver.DriverError):
        driver.read_ledger(ledger_path, case_root)


def test_normal_success_allows_owned_output_manifest_but_rejects_other_output_overlap(case_root):
    output_dir = case_root / "runs" / "b07h-runtime" / "MH_01_easy" / "visloc"
    output_dir.mkdir(parents=True)
    manifest = output_dir / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    result = case_root / "runs" / "ambient" / "invocation-01.json"
    invocation_id, engine, sequence, cells = driver.INVOCATION_CELLS[0]
    payload = {
        "ambient_policy": "recorded", "status": "success", "mapping_started": True,
        "invocation_index": 1, "invocation": invocation_id, "engine": engine, "sequence": sequence,
        "result_cells": list(cells), "cell_results": [{"id": cell, "status": "success"} for cell in cells],
        "runset_sha256": "A" * 64, "source_sha256": driver.EXPECTED_SOURCE_SHA256,
        "protocol_sha256": driver.EXPECTED_PROTOCOL_SHA256, "gt_opened": False,
        "ground_truth_read": False, "ground_truth_materialized": False,
        "ground_truth_argument_present_anywhere": False,
        "manifest": {"ambient_policy": "recorded", "path": str(manifest), "sha256": driver.digest(manifest)},
    }
    checked = driver.record_result(case_root, result, payload, output_dir=output_dir)
    assert checked["status"] == "success"

    unrelated = output_dir / "nested" / "manifest.json"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("{}\n", encoding="utf-8")
    bad = dict(payload)
    bad["manifest"] = {"ambient_policy": "recorded", "path": str(unrelated), "sha256": driver.digest(unrelated)}
    with pytest.raises(driver.DriverError):
        driver.record_result(case_root, case_root / "runs" / "ambient" / "invocation-02.json", bad, output_dir=output_dir)

    other_output = case_root / "runs" / "b07h-runtime" / "MH_01_easy" / "colmap"
    other_output.mkdir(parents=True)
    bad["manifest"] = {"ambient_policy": "recorded", "path": str(manifest), "sha256": driver.digest(manifest)}
    with pytest.raises(driver.DriverError):
        # Treating an existing output file as a directory creates an
        # unrelated manifest/output overlap and must fail closed.
        driver.record_result(case_root, case_root / "runs" / "ambient" / "invocation-02-cross-output.json", bad, output_dir=manifest)

def test_candidate_and_atomic_writer_reject_lexical_reparse_alias(case_root):
    outside = case_root.parent / f"outside-{uuid.uuid4().hex}"
    outside.mkdir()
    alias = case_root / "alias"
    try:
        os.symlink(outside, alias, target_is_directory=True)
    except (OSError, NotImplementedError):
        shutil.rmtree(outside, ignore_errors=True)
        pytest.skip("directory symlink creation is unavailable")
    try:
        with pytest.raises(driver.DriverError):
            driver.candidate_path(alias / "result.json", case_root, "lexical alias")
        with pytest.raises(driver.DriverError):
            driver.atomic_json(alias / "result.json", {"ambient_policy": "recorded"}, case_root, replace=False)
    finally:
        alias.unlink(missing_ok=True)
        shutil.rmtree(outside, ignore_errors=True)


def test_manifest_directory_error_becomes_dnf_and_sealed_result(case_root, monkeypatch):
    runset = _built_v3_fixture(case_root)
    output_dir = case_root / "runs" / "b07h-runtime" / "MH_01_easy" / "visloc"

    class Child:
        def wait(self):
            manifest = output_dir / "manifest.json"
            manifest.mkdir(parents=True)
            return 0

    def fake_ambient(root, history, *args, **kwargs):
        driver._append_history_event(root, history, {"schema": driver.AMBIENT_HISTORY_SCHEMA, "ambient_policy": "recorded", "status": "recorded"})
        return {"ambient_policy": "recorded", "start_allowed": True, "hard_blockers": [], "samples": 1}

    monkeypatch.setattr(driver, "require_c_workspace_clean", lambda: {"clean": True, "forbidden": []})
    monkeypatch.setattr(driver, "record_ambient_window", fake_ambient)
    monkeypatch.setattr(driver.subprocess, "Popen", lambda *args, **kwargs: Child())
    rc = driver.main([
        "--runset", str(runset),
        "--candidate-root", str(case_root),
        "--expected-runset-sha256", driver.digest(runset),
        "--invocation-index", "1",
        "--runtime-temp", "temp/v7-runtime",
        "--ambient-window-seconds", "0",
        "--sample-seconds", "0",
        "--ambient-samples", "1",
    ])
    assert rc == 2
    result_path = case_root / "logs" / "B07H_v4_ambient_recorded_invocation_01.json"
    ledger_path = case_root / driver.LEDGER_RELATIVE_PATH
    assert result_path.is_file() and ledger_path.is_file()
    result_value = json.loads(result_path.read_text(encoding="utf-8"))
    assert result_value["status"] == "dnf"
    assert result_value["manifest"]["ambient_policy"] == "recorded"
    checked = driver.read_ledger(ledger_path, case_root)
    assert checked["cells"][driver.RESULT_CELLS[0]]["status"] == "dnf"
