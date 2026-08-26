"""Focused storage-boundary tests for the versioned B07-H v6 path.

The test fixture deliberately lives on E:. It never runs a mapper, held-out
controller, RoboSim, SearchIndexer, or any GT-bearing operation.
"""

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
    import build_ssfm_dev_gate_v2 as builder  # type: ignore[no-redef]  # noqa: E402
except ImportError as error:  # pragma: no cover - depends on private bench modules
    raise unittest.SkipTest(
        f"private bench module unavailable on this machine: {error}"
    ) from error
import build_b07h_runtime_runset_v2 as runset_builder  # noqa: E402
import run_b07h_runtime_driver_v6 as driver  # noqa: E402
import run_ssfm_heldout_suite_v5_pre_gt as controller  # noqa: E402


ARCHIVE = Path("E:/visloc_archive")
FIXTURE_PARENT = ARCHIVE / "tmp" / "b07_storage_hardening" / "ambient-parity"
ACTIVE = ARCHIVE / "sota_slice13_replay_protocol_r5_20260816"


def _write_sidecar(path: Path) -> str:
    value = driver.digest(path)
    path.with_name(path.name + ".sha256").write_text(f"{value}  {path.name}\n", encoding="ascii")
    return value


def _copy_hashed(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return _write_sidecar(target)


def _fixture_runset(root: Path) -> tuple[Path, str, Path, Path]:
    source = root / "pipelines" / "slam" / "src" / "hierarchical_sfm.rs"
    protocol = root / "B07E_DEV_REPLAY_PROTOCOL.json"
    oracle = root / "scripts" / "run_b07h_runtime_driver_v5.py"
    _copy_hashed(ACTIVE / "pipelines" / "slam" / "src" / "hierarchical_sfm.rs", source)
    _copy_hashed(ACTIVE / "B07E_DEV_REPLAY_PROTOCOL.json", protocol)
    oracle_sha = _copy_hashed(ACTIVE / "scripts" / "run_b07h_runtime_driver_v5.py", oracle)

    tool_bytes = {
        "python.exe": b"fixture-python-tool\n",
        "hierarchical_runner.py": b"fixture-hierarchical-runner\n",
        "hierarchical_executable.exe": b"fixture-hierarchical-executable\n",
        "colmap_runner.py": b"fixture-colmap-runner\n",
        "colmap.exe": b"fixture-colmap-executable\n",
    }
    fixed_tools = {}
    for name, contents in tool_bytes.items():
        tool = root / "tools" / name
        tool.parent.mkdir(parents=True, exist_ok=True)
        tool.write_bytes(contents)
        fixed_tools[name.split(".", 1)[0] if name != "hierarchical_executable.exe" else "hierarchical_executable"] = {"path": str(tool.relative_to(root)), "sha256": driver.digest(tool)}
    fixed_tools = {
        "python": fixed_tools["python"],
        "hierarchical_runner": {"path": "tools/hierarchical_runner.py", "sha256": driver.digest(root / "tools" / "hierarchical_runner.py")},
        "hierarchical_executable": fixed_tools["hierarchical_executable"],
        "colmap_runner": {"path": "tools/colmap_runner.py", "sha256": driver.digest(root / "tools" / "colmap_runner.py")},
        "colmap": {"path": "tools/colmap.exe", "sha256": driver.digest(root / "tools" / "colmap.exe")},
    }

    invocations = []
    for invocation_id, engine, sequence, cells in driver.INVOCATION_CELLS:
        command = [
            "tools/python.exe",
            "tools/hierarchical_runner.py" if engine == "visloc" else "tools/colmap_runner.py",
        ]
        if engine == "visloc":
            command.extend(["--exe", "tools/hierarchical_executable.exe"])
        else:
            command.extend(["--protocol", "B07E_DEV_REPLAY_PROTOCOL.json", "--colmap", "tools/colmap.exe"])
        command.extend(["--out-dir", f"runs/b07h-runtime/{sequence}/{engine}"])
        invocations.append({
            "id": invocation_id,
            "engine": engine,
            "sequence": sequence,
            "command": command,
            "output": f"runs/b07h-runtime/{sequence}/{engine}",
            "result_cells": list(cells),
            "ground_truth_argument_present": False,
        })
    value = {
        "schema": driver.RUNSET_SCHEMA,
        "status": "fixed_preflight_only",
        "candidate_root": str(root),
        "supersedes_schema": driver.RUNSET_V1_SCHEMA,
        "supersedes_sha256": driver.FROZEN_V1_RUNSET_SHA256,
        "protocol": {"path": str(protocol.relative_to(root)), "sha256": driver.EXPECTED_PROTOCOL_SHA256},
        "source": {"path": str(source.relative_to(root)), "sha256": driver.EXPECTED_SOURCE_SHA256},
        "ambient_oracle": {
            "path": str(oracle.relative_to(root)),
            "sha256": oracle_sha,
            "sidecar": str(Path(str(oracle.relative_to(root)) + ".sha256")),
            "bytes": oracle.stat().st_size,
        },
        "fixed_tools": fixed_tools,
        "serial_order": [
            "MH_01_easy visloc", "MH_01_easy colmap (incremental + global cells)",
            "MH_03_medium visloc", "MH_03_medium colmap (incremental + global cells)",
            "MH_05_difficult visloc", "MH_05_difficult colmap (incremental + global cells)",
        ],
        "runtime_policy": {
            "mapping_executed": False,
            "gt_opened": False,
            "performance_claim": False,
            "serial_only": True,
            "total_invocations": 6,
            "total_result_cells": 9,
            "ground_truth_argument_present_anywhere": False,
            "output_paths_preflight_absent": True,
        },
        "storage_gate": {
            "free_bytes_at_preflight": driver.STOP_FREE_BYTES,
            "free_gib_at_preflight": 250,
            "stop_threshold_bytes": driver.STOP_FREE_BYTES,
            "stop_threshold_gib": 250,
            "check_before_each_invocation": True,
            "unstarted_cells_if_below_threshold": "DNF and preserve denominator 9",
        },
        "invocations": invocations,
        "ground_truth_read": False,
        "ground_truth_materialized": False,
        "ground_truth_argument_present_anywhere": False,
    }
    path = root / "runsets" / "B07H_GT_FREE_RUNTIME_RUNSET_V2.json"
    runset_sha = driver.atomic_json(path, value, root, replace=False)
    return path, runset_sha, source, protocol


def _fixture_results(root: Path, runset_sha: str) -> Path:
    statuses = (
        ("success",),
        ("success", "dnf"),
        ("success",),
        ("success", "success"),
        ("success",),
        ("dnf", "success"),
    )
    for index, ((invocation_id, engine, sequence, cells), cell_statuses) in enumerate(zip(driver.INVOCATION_CELLS, statuses), 1):
        cell_results = []
        for cell, status in zip(cells, cell_statuses):
            cell_results.append({"id": cell, "status": status, **({"reason": "fixture DNF"} if status == "dnf" else {})})
        status = "dnf" if "dnf" in cell_statuses else "success"
        payload = {
            "status": status,
            "mapping_started": True,
            "invocation_index": index,
            "invocation": invocation_id,
            "engine": engine,
            "sequence": sequence,
            "result_cells": list(cells),
            "cell_results": cell_results,
            "runset_sha256": runset_sha,
            "source_sha256": driver.EXPECTED_SOURCE_SHA256,
            "protocol_sha256": driver.EXPECTED_PROTOCOL_SHA256,
            "gt_opened": False,
            "ground_truth_read": False,
            "ground_truth_materialized": False,
            "ground_truth_argument_present_anywhere": False,
            "finished_utc": f"2026-08-17T00:00:0{index}+00:00",
        }
        result = root / "runs" / "b07h-runtime" / sequence / f"{engine}.json"
        driver.record_result(root, result, payload)
    return root / "logs" / "B07H_v3_ledger.json"


@pytest.fixture
def case_root():
    FIXTURE_PARENT.mkdir(parents=True, exist_ok=True)
    root = FIXTURE_PARENT / f"case-{os.getpid()}-{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def evidence(case_root: Path):
    runset, runset_sha, _, _ = _fixture_runset(case_root)
    ledger = _fixture_results(case_root, runset_sha)
    driver_copy = case_root / "scripts" / "run_b07h_runtime_driver_v6.py"
    driver_sha = _copy_hashed(Path(__file__).resolve().parents[1] / "scripts" / "run_b07h_runtime_driver_v6.py", driver_copy)
    output = case_root / "evidence" / "ssfm_heldout_v4_dev_gate_v2.json"
    builder.build_dev_gate(
        case_root,
        output,
        runset=runset,
        expected_runset_sha256=runset_sha,
        driver=driver_copy,
        expected_driver_sha256=driver_sha,
        ledger=ledger,
    )
    return {
        "root": case_root,
        "runset": runset,
        "runset_sha": runset_sha,
        "ledger": ledger,
        "driver": driver_copy,
        "driver_sha": driver_sha,
        "gate": output,
    }


def test_six_invocations_nine_cells_mixed_colmap_and_source_chain(evidence):
    checked = builder.validate_dev_gate_v2(evidence["gate"], evidence["root"], expected_runset_sha256=evidence["runset_sha"])
    assert [item["id"] for item in checked["result_cells"]] == list(driver.RESULT_CELLS)
    assert [item["status"] for item in checked["result_cells"]] == [
        "success", "success", "dnf", "success", "success", "success", "success", "dnf", "success"
    ]
    assert all(Path(item["path"]).drive.upper() == "E:" for item in checked["result_cells"])


def test_deferred_evidence_never_consumes_cells_or_validates_as_terminal(evidence):
    before = json.loads(evidence["ledger"].read_text(encoding="utf-8"))["cells"]
    event = driver.append_deferred(
        evidence["root"],
        evidence["root"] / "logs" / "ambient.jsonl",
        {"reason": "ambient timeout", "deferred_cells": list(driver.RESULT_CELLS)},
    )
    assert event["status"] == "deferred"
    after = json.loads(evidence["ledger"].read_text(encoding="utf-8"))["cells"]
    assert after == before
    assert (evidence["root"] / "logs" / "ambient.jsonl.sha256").is_file()

    deferred = evidence["root"] / "logs" / "deferred-result.json"
    driver.atomic_json(deferred, {
        "schema": driver.DEFERRED_SCHEMA,
        "status": "deferred",
        "invocation_index": 1,
        "invocation": driver.INVOCATION_CELLS[0][0],
        "result_cells": [driver.RESULT_CELLS[0]],
        "cell_results": [{"id": driver.RESULT_CELLS[0], "status": "deferred"}],
        "deferred_cells": [driver.RESULT_CELLS[0]],
    }, evidence["root"], replace=False)
    with pytest.raises(driver.DriverError):
        builder._validate_source_artifact(
            deferred, driver.digest(deferred), evidence["root"], 1, driver.INVOCATION_CELLS[0], evidence["runset_sha"]
        )


def test_source_artifact_tamper_is_rejected(evidence):
    ledger = json.loads(evidence["ledger"].read_text(encoding="utf-8"))
    result_path = Path(ledger["results"][0]["result_path"])
    result_path.write_text(result_path.read_text(encoding="utf-8").replace('"success"', '"dnf"', 1), encoding="utf-8")
    with pytest.raises(driver.DriverError):
        builder.validate_dev_gate_v2(evidence["gate"], evidence["root"], expected_runset_sha256=evidence["runset_sha"])


def test_normalized_cell_tamper_is_rejected(evidence):
    gate = json.loads(evidence["gate"].read_text(encoding="utf-8"))
    cell_path = Path(gate["cell_results"][0]["result_artifact_path"])
    cell_path.write_text(cell_path.read_text(encoding="utf-8").replace('"success"', '"dnf"', 1), encoding="utf-8")
    with pytest.raises(driver.DriverError):
        builder.validate_dev_gate_v2(evidence["gate"], evidence["root"], expected_runset_sha256=evidence["runset_sha"])


def test_missing_source_chain_is_rejected(evidence):
    gate = json.loads(evidence["gate"].read_text(encoding="utf-8"))
    gate.pop("source_artifact_chain")
    driver.atomic_json(evidence["gate"], gate, evidence["root"], replace=True)
    with pytest.raises(driver.DriverError):
        builder.validate_dev_gate_v2(evidence["gate"], evidence["root"], expected_runset_sha256=evidence["runset_sha"])


def test_top_level_gt_flag_is_rejected(evidence):
    gate = json.loads(evidence["gate"].read_text(encoding="utf-8"))
    gate["ground_truth_read"] = True
    driver.atomic_json(evidence["gate"], gate, evidence["root"], replace=True)
    with pytest.raises(driver.DriverError):
        builder.validate_dev_gate_v2(evidence["gate"], evidence["root"], expected_runset_sha256=evidence["runset_sha"])


@pytest.mark.parametrize("mutation", ["reverse", "duplicate", "missing"])
def test_wrong_order_duplicate_or_missing_cells_rejected(evidence, mutation):
    gate = json.loads(evidence["gate"].read_text(encoding="utf-8"))
    cells = gate["cell_results"]
    if mutation == "reverse":
        cells[:] = list(reversed(cells))
    elif mutation == "duplicate":
        cells[1] = dict(cells[0])
    else:
        cells.pop()
    driver.atomic_json(evidence["gate"], gate, evidence["root"], replace=True)
    with pytest.raises(driver.DriverError):
        builder.validate_dev_gate_v2(evidence["gate"], evidence["root"], expected_runset_sha256=evidence["runset_sha"])


def test_nested_gt_flag_and_c_paths_are_rejected():
    with pytest.raises(driver.DriverError):
        driver.reject_gt({"nested": [{"metadata": {"ground_truth_read": True}}]}, "test")
    with pytest.raises(driver.DriverError):
        driver.require_e_root(Path("C:/Users/rsasa/Workspace/visloc-rs"))
    with pytest.raises(driver.DriverError):
        driver.candidate_path(Path("C:/temp/result.json"), Path("E:/visloc_archive/case"), "test path")
    with pytest.raises(driver.DriverError):
        driver._validate_runset_command_storage(["C:/tools/python.exe", "--out-dir", "C:/temp"], Path("E:/visloc_archive/case"), "test command")


def test_recursive_gt_string_tokens_are_rejected_but_false_flags_are_allowed():
    driver.reject_gt({"flags": {"ground_truth_read": False}, "nested": [{"gt_opened": False}]}, "test")
    with pytest.raises(driver.DriverError):
        driver.reject_gt({"nested": ["contains groundtruth token"]}, "test")
    with pytest.raises(driver.DriverError):
        driver.reject_gt({"nested": {"description": "state_groundtruth_estimate0"}}, "test")


def _mutate_runset(evidence, mutation):
    value = json.loads(evidence["runset"].read_text(encoding="utf-8"))
    mutation(value)
    return driver.atomic_json(evidence["runset"], value, evidence["root"], replace=True)


def test_fixed_tool_sha_and_command_binding_are_required(evidence):
    def bad_sha(value):
        value["fixed_tools"]["python"]["sha256"] = "0" * 64

    sha = _mutate_runset(evidence, bad_sha)
    with pytest.raises(driver.DriverError):
        driver.validate_runset(evidence["runset"], evidence["root"], sha)

    def bad_runner(value):
        value["invocations"][0]["command"][1] = "tools/colmap_runner.py"

    sha = _mutate_runset(evidence, bad_runner)
    with pytest.raises(driver.DriverError):
        driver.validate_runset(evidence["runset"], evidence["root"], sha)


def test_execution_tool_flags_are_required_unique_and_engine_bound(evidence):
    baseline = json.loads(evidence["runset"].read_text(encoding="utf-8"))

    def assert_rejected(index, mutate):
        value = json.loads(json.dumps(baseline))
        mutate(value["invocations"][index]["command"])
        sha = driver.atomic_json(evidence["runset"], value, evidence["root"], replace=True)
        with pytest.raises(driver.DriverError):
            driver.validate_runset(evidence["runset"], evidence["root"], sha)

    assert_rejected(0, lambda command: (command.remove("--exe"), command.remove("tools/hierarchical_executable.exe")))
    assert_rejected(1, lambda command: (command.remove("--colmap"), command.remove("tools/colmap.exe")))
    assert_rejected(0, lambda command: command.extend(["--exe", "tools/hierarchical_executable.exe"]))
    assert_rejected(1, lambda command: command.extend(["--colmap", "tools/colmap.exe"]))
    assert_rejected(0, lambda command: command.__setitem__(command.index("--exe") + 1, "tools/colmap.exe"))
    assert_rejected(1, lambda command: command.__setitem__(command.index("--colmap") + 1, "tools/hierarchical_executable.exe"))
    assert_rejected(0, lambda command: command.extend(["--colmap", "tools/colmap.exe"]))
    assert_rejected(1, lambda command: command.extend(["--exe", "tools/hierarchical_executable.exe"]))


@pytest.mark.parametrize("bad_path", ["C:/outside/result", "\\\\server\\share\\result", "/tmp/result", "runs/../escape"])
def test_unc_posix_c_and_traversal_output_paths_are_rejected(evidence, bad_path):
    sha = _mutate_runset(evidence, lambda value: value["invocations"][0].update(output=bad_path))
    with pytest.raises(driver.DriverError):
        driver.validate_runset(evidence["runset"], evidence["root"], sha)


def test_manifest_must_be_regular_and_non_reparse(case_root):
    manifest_dir = case_root / "manifest-dir"
    manifest_dir.mkdir()
    with pytest.raises(driver.DriverError):
        driver.require_regular_candidate_file(manifest_dir, case_root, "manifest")
    outside = case_root.parent / "outside-manifest.json"
    outside.write_text("{}", encoding="utf-8")
    link = case_root / "manifest-link.json"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        return
    with pytest.raises(driver.DriverError):
        driver.require_regular_candidate_file(link, case_root, "manifest")


@pytest.mark.parametrize("mutation", ["gap", "duplicate", "reorder"])
def test_ledger_requires_strict_serial_prefix(evidence, mutation):
    ledger = json.loads(evidence["ledger"].read_text(encoding="utf-8"))
    if mutation == "gap":
        ledger["results"][0]["invocation_index"] = 2
    elif mutation == "duplicate":
        ledger["results"][1]["invocation_index"] = 1
    else:
        ledger["results"] = list(reversed(ledger["results"]))
    driver.atomic_json(evidence["ledger"], ledger, evidence["root"], replace=True)
    with pytest.raises(driver.DriverError):
        driver.read_ledger(evidence["ledger"], evidence["root"])


def test_runtime_environment_is_e_only_and_c_workspace_cache_is_fail_closed(case_root):
    assert driver.workspace_state(case_root)["clean"] is True
    driver.require_workspace_clean(case_root)
    env, locations = driver.build_runtime_environment(case_root, 1, case_root / "runtime_temp")
    for key in driver.E_ENV_SUFFIXES:
        assert Path(env[key]).drive.upper() == "E:"
        assert Path(env[key]).is_relative_to(case_root)
        assert Path(locations[key]).is_relative_to(case_root)
    (case_root / "nested" / "target").mkdir(parents=True)
    (case_root / "nested" / ".pytest_cache").mkdir(parents=True)
    (case_root / "nested" / "generated.pyc").write_bytes(b"cache")
    (case_root / ".git" / "target").mkdir(parents=True)
    state = driver.workspace_state(case_root)
    assert state["clean"] is False
    assert any("nested" in item for item in state["forbidden"])
    assert not any(".git" in item for item in state["forbidden"])
    with pytest.raises(driver.DriverError):
        driver.require_workspace_clean(case_root)


def test_failure_statuses_normalize_to_terminal_dnf():
    assert driver.normalize_terminal_status("failure", 1) == ("dnf", "failure")
    assert driver.normalize_terminal_status("error", 0) == ("dnf", "error")
    assert driver.normalize_terminal_status("success", 0) == ("success", None)


def test_heldout_v5_overrides_inherited_c_storage_paths(case_root):
    inherited = {
        "CARGO_HOME": "C:/cargo-home",
        "RUSTUP_HOME": "C:/rustup",
        "PYTHONPYCACHEPREFIX": "C:/pycache",
        "TEMP": "C:/temp",
        "TMP": "C:/temp",
        "CARGO_TARGET_DIR": "C:/target",
        "CCACHE_DIR": "C:/ccache",
        "UNKNOWN_CACHE_ROOT": "C:/unknown-cache",
        "SOME_TEMP_DIR": "C:/unknown-temp",
        "TORCH_EXTENSIONS_DIR": "C:/torch-extensions",
        "PYTHONPATH": "C:/workspace/scripts",
        "PATH": "C:/Windows/System32",
    }
    env = controller.build_hardened_environment(inherited, case_root, case_root / "runtime_temp")
    for key in controller.E_CHILD_ENV_SUFFIXES:
        assert Path(env[key]).drive.upper() == "E:"
        assert Path(env[key]).is_relative_to(case_root)
    assert env["PATH"] == inherited["PATH"]
    assert "CCACHE_DIR" not in env
    assert "UNKNOWN_CACHE_ROOT" not in env
    assert "SOME_TEMP_DIR" not in env
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["PYTHONPATH"] == str(case_root / "scripts")
    assert env["PATH"] == inherited["PATH"]


def test_runset_rebase_removes_old_candidate_strings():
    old_root = Path("E:/visloc_archive/old-candidate")
    new_root = Path("E:/visloc_archive/new-candidate")
    value = {
        "candidate_root": str(old_root),
        "output": str(old_root / "runs" / "out"),
        "command": ["--out-dir", str(old_root / "runs" / "out")],
        "label": "old-candidate-id",
    }
    rebased = runset_builder.rebase_candidate_paths(value, old_root, new_root)
    assert str(new_root) in rebased["output"]
    assert str(old_root) not in json.dumps(rebased)
    assert rebased["label"] == value["label"]


def _fixed_tool_metadata(root: Path) -> dict[str, dict[str, object]]:
    names = {
        "python": "python.exe",
        "hierarchical_runner": "hierarchical_runner.py",
        "hierarchical_executable": "hierarchical_executable.exe",
        "colmap_runner": "colmap_runner.py",
        "colmap": "colmap.exe",
    }
    result = {}
    for key, name in names.items():
        path = root / "tools" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{key}-fixture\n".encode("ascii"))
        result[key] = {"path": str(path.relative_to(root)), "sha256": driver.digest(path), "bytes": path.stat().st_size}
    return result


def test_runset_builder_enriches_legacy_fixed_tool_metadata(case_root):
    fixed = _fixed_tool_metadata(case_root)
    fixed["python"].pop("sha256")
    fixed["python"].pop("bytes")
    enriched = runset_builder._enrich_fixed_tools(fixed, case_root)
    for item in enriched.values():
        path = case_root / item["path"]
        assert item["sha256"] == driver.digest(path)
        assert item["bytes"] == path.stat().st_size


@pytest.mark.parametrize("field,bad_value", [("sha256", "0" * 64), ("bytes", 0)])
def test_runset_builder_rejects_existing_fixed_tool_metadata_mismatch(case_root, field, bad_value):
    fixed = _fixed_tool_metadata(case_root)
    fixed["python"][field] = bad_value
    with pytest.raises(driver.DriverError):
        runset_builder._enrich_fixed_tools(fixed, case_root)


def test_runset_builder_rejects_fixed_tool_symlink_and_outside_path(case_root):
    fixed = _fixed_tool_metadata(case_root)
    outside = case_root.parent / "outside-fixed-tool.exe"
    outside.write_bytes(b"outside\n")
    link = case_root / "tools" / "python-link.exe"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        outside.unlink(missing_ok=True)
        pytest.skip("symlink creation is unavailable")
    try:
        fixed["python"]["path"] = str(link.relative_to(case_root))
        with pytest.raises(driver.DriverError):
            runset_builder._enrich_fixed_tools(fixed, case_root)
        fixed["python"]["path"] = str(outside)
        with pytest.raises(driver.DriverError):
            runset_builder._enrich_fixed_tools(fixed, case_root)
    finally:
        link.unlink(missing_ok=True)
        outside.unlink(missing_ok=True)


def test_cli_relative_default_is_candidate_rooted_from_c_cwd(evidence, monkeypatch):
    monkeypatch.chdir(Path("C:/"))
    args = builder.parse_args([
        "--candidate-root", str(evidence["root"]),
        "--runset", str(evidence["runset"]),
        "--expected-runset-sha256", evidence["runset_sha"],
        "--driver", str(evidence["driver"]),
        "--expected-driver-sha256", evidence["driver_sha"],
        "--ledger", str(evidence["ledger"]),
    ])
    assert args.output == Path("evidence") / "ssfm_heldout_v4_dev_gate_v2.json"
    with pytest.raises(driver.DriverError):
        builder.build_dev_gate(
            evidence["root"], Path("C:/candidate-escape.json"),
            runset=evidence["runset"], expected_runset_sha256=evidence["runset_sha"],
            driver=evidence["driver"], expected_driver_sha256=evidence["driver_sha"], ledger=evidence["ledger"],
        )


def _ambient_oracle_fixture(evidence):
    workspace = evidence["root"] / "ambient-workspace"
    workspace.mkdir()
    oracle, checked = driver._load_ambient_oracle(
        evidence["root"],
        json.loads(evidence["runset"].read_text(encoding="utf-8"))["ambient_oracle"],
        workspace,
    )
    return oracle, checked, workspace


def _clean_process(*, targets=None, cpu=0.0, search=0.0):
    return {
        "target_processes": list(targets or []),
        "total_processor_percent": cpu,
        "search_indexer_percent": search,
    }


def _idle_gpu(memory=0.0, utilization=0.0):
    return {
        "available": True,
        "utilization_percent": utilization,
        "memory_used_mib": memory,
    }


def test_v5_ambient_contract_each_false_check_is_fail_closed(evidence):
    oracle, _, workspace = _ambient_oracle_fixture(evidence)

    def sample(process, *, wsl=None, gpu=None, free=driver.STOP_FREE_BYTES):
        return oracle.ambient_sample(
            evidence["root"],
            workspace,
            process_sampler=lambda: process,
            wsl_sampler=lambda: wsl or {"status": "idle", "target_processes": []},
            gpu_sampler=lambda: gpu or _idle_gpu(),
            free_bytes_fn=lambda _root: free,
        )["checks"]

    assert sample(_clean_process(targets=[{"name": "cargo"}]))["target_processes_clear"] is False
    assert sample(_clean_process(cpu=driver.CPU_SETTLE_LIMIT_PERCENT + 0.01))["cpu_settled"] is False
    assert sample(_clean_process(search=driver.SEARCH_INDEXER_SETTLE_LIMIT_PERCENT + 0.01))["search_settled"] is False
    assert sample(_clean_process(), gpu=_idle_gpu(utilization=1.0))["gpu_settled"] is False
    assert sample(_clean_process(), free=driver.STOP_FREE_BYTES - 1)["e_free_threshold"] is False
    (workspace / "target").mkdir()
    try:
        assert sample(_clean_process())["c_workspace_clean"] is False
    finally:
        (workspace / "target").rmdir()
    assert sample(
        _clean_process(),
        wsl={"status": "running", "target_processes": [{"name": "cargo", "pid": 7}]},
    )["target_processes_clear"] is False


def test_v5_gpu_growth_resets_consecutive_and_requires_exact_five(evidence):
    oracle, checked, workspace = _ambient_oracle_fixture(evidence)
    values = iter(
        [
            2307.0,
            2307.0 + driver.GPU_MEMORY_GROWTH_TOLERANCE_MIB + 1.0,
            2307.0,
            2307.0,
            2307.0,
            2307.0,
            2307.0,
        ]
    )
    process = lambda: _clean_process()
    result = oracle.settle_ambient(
        evidence["root"],
        evidence["root"] / "logs" / "oracle-growth.jsonl",
        workspace_root=workspace,
        timeout_seconds=1.0,
        sample_seconds=0.0,
        consecutive_samples=5,
        process_sampler=process,
        wsl_sampler=lambda: {"status": "idle", "target_processes": []},
        gpu_sampler=lambda: _idle_gpu(next(values)),
        free_bytes_fn=lambda _root: driver.STOP_FREE_BYTES,
    )
    assert result["reason"] == "settled"
    assert result["samples"] == 7
    assert result["consecutive"] == 5
    observations = result["gpu_observations"]
    assert observations[0]["baseline_sample"] is True
    assert observations[1]["reason"] == "memory_growth"
    assert observations[1]["memory_growth_mib"] > driver.GPU_MEMORY_GROWTH_TOLERANCE_MIB


def test_v6_settle_finally_seals_history_and_retry_append_is_consistent(evidence):
    _, checked, workspace = _ambient_oracle_fixture(evidence)
    history = evidence["root"] / "logs" / "retry-ambient.jsonl"
    process = lambda: _clean_process()
    timeout = driver.settle_ambient(
        evidence["root"],
        history,
        [driver.RESULT_CELLS[0]],
        ambient_oracle=checked,
        workspace_root=workspace,
        timeout_seconds=0.0,
        sample_seconds=0.0,
        consecutive_samples=1,
        process_sampler=process,
        wsl_sampler=lambda: {"status": "idle", "target_processes": []},
        gpu_sampler=lambda: {"available": False, "utilization_percent": None, "memory_used_mib": None},
        free_bytes_fn=lambda _root: driver.STOP_FREE_BYTES,
    )
    assert timeout["reason"] == "timeout"
    sidecar = history.with_name(history.name + ".sha256")
    manifest = history.with_name(history.name + ".manifest")
    manifest_sidecar = manifest.with_name(manifest.name + ".sha256")
    assert sidecar.is_file() and manifest.is_file() and manifest_sidecar.is_file()
    history_sha = driver.validate_sidecar(history, evidence["root"], "ambient history")
    assert driver.read_json(manifest)["sha256"] == history_sha
    driver.validate_sidecar(manifest, evidence["root"], "ambient history manifest")

    before = history.read_bytes()
    driver.append_deferred(
        evidence["root"],
        history,
        {"reason": "ambient timeout", "deferred_cells": [driver.RESULT_CELLS[0]]},
    )
    assert len(history.read_bytes()) > len(before)
    driver.validate_sidecar(history, evidence["root"], "ambient history")
    driver.settle_ambient(
        evidence["root"],
        history,
        [driver.RESULT_CELLS[0]],
        ambient_oracle=checked,
        workspace_root=workspace,
        timeout_seconds=1.0,
        sample_seconds=0.0,
        consecutive_samples=1,
        process_sampler=process,
        wsl_sampler=lambda: {"status": "idle", "target_processes": []},
        gpu_sampler=lambda: _idle_gpu(),
        free_bytes_fn=lambda _root: driver.STOP_FREE_BYTES,
    )
    assert driver.read_json(manifest)["sha256"] == driver.digest(history)


@pytest.mark.parametrize("mutation", ["missing", "oracle_tamper", "sidecar_tamper", "bytes_tamper"])
def test_v6_ambient_oracle_metadata_tamper_or_missing_is_rejected(evidence, mutation):
    value = json.loads(evidence["runset"].read_text(encoding="utf-8"))
    oracle = evidence["root"] / value["ambient_oracle"]["path"]
    sidecar = oracle.with_name(oracle.name + ".sha256")
    if mutation == "missing":
        value.pop("ambient_oracle")
        sha = driver.atomic_json(evidence["runset"], value, evidence["root"], replace=True)
    elif mutation == "oracle_tamper":
        oracle.write_bytes(oracle.read_bytes() + b"tamper")
        sha = evidence["runset_sha"]
    elif mutation == "sidecar_tamper":
        sidecar.write_text("0" * 64 + "  " + oracle.name + "\n", encoding="ascii")
        sha = evidence["runset_sha"]
    else:
        value["ambient_oracle"]["bytes"] += 1
        sha = driver.atomic_json(evidence["runset"], value, evidence["root"], replace=True)
    with pytest.raises(driver.DriverError):
        driver.validate_runset(evidence["runset"], evidence["root"], sha)


def test_v6_ambient_history_retry_rejects_missing_or_tampered_seal(evidence):
    _, checked, workspace = _ambient_oracle_fixture(evidence)
    history = evidence["root"] / "logs" / "sealed-history.jsonl"
    driver.settle_ambient(
        evidence["root"], history, [], ambient_oracle=checked, workspace_root=workspace,
        timeout_seconds=0.0, sample_seconds=0.0, consecutive_samples=1,
        process_sampler=lambda: _clean_process(),
        wsl_sampler=lambda: {"status": "idle", "target_processes": []},
        gpu_sampler=lambda: {"available": False, "utilization_percent": None, "memory_used_mib": None},
        free_bytes_fn=lambda _root: driver.STOP_FREE_BYTES,
    )
    history.with_name(history.name + ".manifest").write_text(
        json.dumps({"schema": "B07H_RUNTIME_DRIVER_DEFERRED_SIDECAR_V1", "path": str(history), "sha256": "0" * 64}),
        encoding="utf-8",
    )
    with pytest.raises(driver.DriverError):
        driver.settle_ambient(
            evidence["root"], history, [], ambient_oracle=checked, workspace_root=workspace,
            timeout_seconds=0.0, sample_seconds=0.0, consecutive_samples=1,
            process_sampler=lambda: _clean_process(),
            wsl_sampler=lambda: {"status": "idle", "target_processes": []},
            gpu_sampler=lambda: _idle_gpu(), free_bytes_fn=lambda _root: driver.STOP_FREE_BYTES,
        )
