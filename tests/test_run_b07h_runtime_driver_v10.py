"""Admission-policy tests for the V10 relaxed RoboSim lane."""
from __future__ import annotations
import json, os, shutil, sys, uuid
from pathlib import Path
os.environ["PYTHONDONTWRITEBYTECODE"]="1"; sys.dont_write_bytecode=True; sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import pytest
import run_b07h_runtime_driver_v10 as driver
import run_b07h_runtime_driver_v7 as v7
import run_b07h_runtime_driver_v8 as v8
import run_b07h_runtime_driver_v9 as v9
from test_run_b07h_runtime_driver_v8 import _fixture, _v9_source

PARENT=Path("E:/visloc_archive/tmp/b07_storage_hardening/v10-policy")
@pytest.fixture
def root(monkeypatch):
    PARENT.mkdir(parents=True,exist_ok=True); value=PARENT/f"case-{os.getpid()}-{uuid.uuid4().hex}"; value.mkdir()
    monkeypatch.setattr(driver.v7,"c_workspace_state",lambda _: {"clean":True,"forbidden":[]})
    try: yield value
    finally: shutil.rmtree(value,ignore_errors=True)

def proc(*names): return {"total_processor_percent":199,"search_indexer_percent":199,"target_processes":[{"name":n} for n in names]}
def wsl(*names): return {"status":"busy","target_processes":[{"comm":n} for n in names]}

def test_unrelated_robosim_wsl_cargo_rustc_is_recorded_not_blocking(root):
    item=driver.ambient_sample(root,process_sampler=lambda:proc("cargo","rustc","robosim"),wsl_sampler=lambda:wsl("cargo","rustc","robosim"),gpu_sampler=lambda:{"utilization_percent":100},free_bytes_fn=lambda _:driver.STOP_FREE_BYTES)
    assert item["start_allowed"] is True
    assert item["hard_blockers"]["processes"] == []
    assert len(item["informational_processes"]) == 6
    assert item["noise"]["informational"] is True

@pytest.mark.parametrize("name",["sequential_sfm_demo.exe","colmap.exe","run_b07h_runtime_driver_v10.py"])
def test_actual_sfm_colmap_or_this_driver_blocks(root,name):
    item=driver.ambient_sample(root,process_sampler=lambda:proc(name),wsl_sampler=lambda:wsl(),free_bytes_fn=lambda _:driver.STOP_FREE_BYTES)
    assert item["start_allowed"] is False
    assert item["hard_blockers"]["processes"]

def test_resources_and_c_workspace_remain_hard_blockers(root,monkeypatch):
    low=driver.ambient_sample(root,process_sampler=lambda:proc("robosim"),wsl_sampler=lambda:wsl("cargo"),free_bytes_fn=lambda _:driver.STOP_FREE_BYTES-1)
    assert low["start_allowed"] is False
    monkeypatch.setattr(driver.v7,"c_workspace_state",lambda _: {"clean":False,"forbidden":["target"]})
    dirty=driver.ambient_sample(root,process_sampler=lambda:proc("robosim"),wsl_sampler=lambda:wsl("rustc"),free_bytes_fn=lambda _:driver.STOP_FREE_BYTES)
    assert dirty["start_allowed"] is False

def test_v10_record_result_has_distinct_policy_and_namespace(root):
    result=root/"logs"/"B07H_v7_relaxed_invocation_01.json"; ledger=root/"logs"/"B07H_v7_relaxed_ledger.json"
    ident,engine,sequence,cells=driver.INVOCATION_CELLS[0]
    payload={"ambient_policy":driver.AMBIENT_POLICY,"admission_policy":driver.ADMISSION_POLICY,"status":"success","invocation_index":1,"invocation":ident,"engine":engine,"sequence":sequence,"result_cells":list(cells),"cell_results":[{"id":cells[0],"status":"success"}],"runset_sha256":"A"*64,"source_sha256":driver.EXPECTED_SOURCE_SHA256,"protocol_sha256":driver.EXPECTED_PROTOCOL_SHA256,"gt_opened":False,"ground_truth_read":False,"ground_truth_materialized":False,"ground_truth_argument_present_anywhere":False,"manifest":{"ambient_policy":driver.AMBIENT_POLICY,"admission_policy":driver.ADMISSION_POLICY,"path":None,"sha256":None}}
    prior=root/"logs"/"B07H_v7_relaxed_prior.json"; v7.atomic_json(prior,{"schema":driver.LEDGER_SCHEMA,"ambient_policy":driver.AMBIENT_POLICY,"admission_policy":driver.ADMISSION_POLICY,"runset_sha256":"A"*64,"expected_cells":list(driver.RESULT_CELLS),"total_result_cells":9,"results":[],"cells":{}},root,replace=False)
    value=driver.record_result(root,result,payload,prior_ledger=prior,output_ledger=ledger)
    assert value["schema"]==driver.RESULT_SCHEMA and value["admission_policy"]==driver.ADMISSION_POLICY
    assert json.loads(ledger.read_text())["schema"]==driver.LEDGER_SCHEMA
    assert not (root/"logs"/"B07H_v4_ambient_recorded_ledger.json").exists()

def test_v8_to_v4_bootstrap_then_v10_invocation_two_and_v9_import(root,monkeypatch):
    v3_runset,v7_result,v7_ledger,_=_fixture(root)
    v3_value=json.loads(v3_runset.read_text(encoding="utf-8")); v4_value=json.loads(json.dumps(v3_value)); v3_sha=v8.digest(v3_runset)
    v4_value["schema"]=driver.RUNSET_SCHEMA; v4_value["supersedes_schema"]=v7.RUNSET_SCHEMA; v4_value["supersedes_sha256"]=v3_sha; v4_value["admission_policy"]=driver.ADMISSION_POLICY; v4_value["ambient_policy"]=driver.ADMISSION_POLICY; v4_value["ambient_recording"]={"finite_window":True,"noise_is_informational":True,"hard_blockers":["visloc_sfm_colmap_driver_processes","c_workspace","e_free_threshold"],"informational_processes":["robosim","cargo","rustc"],"robosim_wsl_processes_are_informational":True}; v4_value.setdefault("storage_policy",{})["ambient_policy"]=driver.ADMISSION_POLICY
    v4_runset=root/"runsets"/"v4.json"; v7.atomic_json(v4_runset,v4_value,root,replace=False)
    monkeypatch.setattr(driver,"validate_runset",lambda path,*args,**kwargs: v4_value); monkeypatch.setattr(v8.v7,"validate_runset",lambda path,*args,**kwargs: v3_value)
    v8_result=root/"recovery"/"v8-result.json"; v8_ledger=root/"recovery"/"v8-ledger.json"; v8.recover_invocation_one(root,original_result=v7_result,original_ledger=v7_ledger,runset=v3_runset,output_result=v8_result,output_ledger=v8_ledger)
    bootstrap=root/"recovery"/"v10-ledger-01.json"; driver.bootstrap_v8_to_v10(root,v8_ledger=v8_ledger,v3_runset=v3_runset,v4_runset=v4_runset,output_ledger=bootstrap)
    inv2=v3_value["invocations"][1]; v10_result=root/"recovery"/"v10-result-02.json"; payload={"ambient_policy":driver.AMBIENT_POLICY,"admission_policy":driver.ADMISSION_POLICY,"status":"dnf","invocation_index":2,"invocation":inv2["id"],"engine":inv2["engine"],"sequence":inv2["sequence"],"result_cells":list(inv2["result_cells"]),"cell_results":[{"id":cell,"status":"dnf","reason":"mock child manifest absent"} for cell in inv2["result_cells"]],"runset_sha256":v8.digest(v4_runset),"source_sha256":driver.EXPECTED_SOURCE_SHA256,"protocol_sha256":driver.EXPECTED_PROTOCOL_SHA256,"gt_opened":False,"ground_truth_read":False,"ground_truth_materialized":False,"ground_truth_argument_present_anywhere":False,"manifest":{"ambient_policy":driver.AMBIENT_POLICY,"admission_policy":driver.ADMISSION_POLICY,"path":None,"sha256":None}}
    output_ledger=root/"recovery"/"v10-ledger-02.json"; driver.record_result(root,v10_result,payload,prior_ledger=bootstrap,output_ledger=output_ledger)
    cumulative=v9.initialize_from_v8(root,v8_ledger=v8_ledger,output_ledger=root/"recovery"/"v9-ledger-01.json"); imported,_=v9.import_v7_result(root,prior_ledger=cumulative,v7_result=v10_result,runset=v4_runset,output_result=root/"recovery"/"v9-result-02.json",output_ledger=root/"recovery"/"v9-ledger-02.json")
    assert json.loads(imported.read_text())["invocation_index"]==2
    assert json.loads(bootstrap.read_text())["results"][0]["invocation_index"]==1

def test_v10_prior_output_ledger_is_non_overwriting_and_sidecar_bound(root):
    ident,engine,sequence,cells=driver.INVOCATION_CELLS[0]; prior=root/"prior.json"; output=root/"output.json"; result=root/"result.json"
    base={"schema":driver.LEDGER_SCHEMA,"ambient_policy":driver.AMBIENT_POLICY,"admission_policy":driver.ADMISSION_POLICY,"runset_sha256":"B"*64,"expected_cells":list(driver.RESULT_CELLS),"total_result_cells":9,"results":[],"cells":{}}
    v7.atomic_json(prior,base,root,replace=False)
    payload={"ambient_policy":driver.AMBIENT_POLICY,"admission_policy":driver.ADMISSION_POLICY,"invocation_index":1,"invocation":ident,"engine":engine,"sequence":sequence,"result_cells":list(cells),"cell_results":[{"id":cells[0],"status":"dnf","reason":"mock"}],"runset_sha256":"B"*64,"source_sha256":driver.EXPECTED_SOURCE_SHA256,"protocol_sha256":driver.EXPECTED_PROTOCOL_SHA256,"gt_opened":False,"ground_truth_read":False,"ground_truth_materialized":False,"ground_truth_argument_present_anywhere":False,"manifest":{"ambient_policy":driver.AMBIENT_POLICY,"admission_policy":driver.ADMISSION_POLICY,"path":None,"sha256":None}}
    with pytest.raises(driver.DriverError): driver.record_result(root,result,payload,prior_ledger=prior,output_ledger=prior)
    output.write_text("stale",encoding="utf-8")
    with pytest.raises(driver.DriverError): driver.record_result(root,result,payload,prior_ledger=prior,output_ledger=output)
    output.unlink(); Path(str(prior)+".sha256").unlink()
    with pytest.raises(driver.DriverError): driver.record_result(root,result,payload,prior_ledger=prior,output_ledger=output)
