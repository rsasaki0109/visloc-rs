#!/usr/bin/env python3
"""B07-H V10 relaxed admission lane; RoboSim noise is informational only."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
sys.dont_write_bytecode = True
import run_b07h_runtime_driver_v7 as v7  # noqa: E402
import run_b07h_runtime_driver_v9 as v9  # noqa: E402

RUNSET_SCHEMA = "B07H_GT_FREE_RUNTIME_RUNSET_V4"
RESULT_SCHEMA = "B07H_RUNTIME_DRIVER_RESULT_V7_RELAXED"
LEDGER_SCHEMA = "B07H_RUNTIME_DRIVER_LEDGER_V7_RELAXED"
HISTORY_SCHEMA = "B07H_RUNTIME_DRIVER_AMBIENT_RELAXED_V1"
DRIVER_VERSION = "B07H_RUNTIME_DRIVER_V10_RELAXED"
ADMISSION_POLICY = "relaxed_recorded_robosim"
AMBIENT_POLICY = ADMISSION_POLICY
RESULT_CELLS, INVOCATION_CELLS = v7.RESULT_CELLS, v7.INVOCATION_CELLS
TOTAL_RESULT_CELLS, STOP_FREE_BYTES = v7.TOTAL_RESULT_CELLS, v7.STOP_FREE_BYTES
EXPECTED_SOURCE_SHA256, EXPECTED_PROTOCOL_SHA256 = v7.EXPECTED_SOURCE_SHA256, v7.EXPECTED_PROTOCOL_SHA256
DriverError = v7.DriverError

HARD_NAMES = frozenset({"colmap", "colmap.exe", "sequential_sfm_demo", "sequential_sfm_demo.exe", "visloc", "sfm", "run_b07h_runtime_driver_v10", "run_b07h_runtime_driver_v10.py"})
INFO_NAMES = frozenset({"robosim", "robosim.exe", "cargo", "cargo.exe", "rustc", "rustc.exe", "searchindexer", "searchindexer.exe"})

def _name(item: Any) -> str:
    if isinstance(item, Mapping): return str(item.get("name") or item.get("process_name") or item.get("comm") or "").lower().replace("\\", "/").rsplit("/", 1)[-1]
    return str(item).lower().replace("\\", "/").rsplit("/", 1)[-1]

def _entries(value: Any) -> list[Any]:
    if not isinstance(value, Mapping): return []
    out=[]
    for key in ("target_processes", "conflicts", "processes"):
        if isinstance(value.get(key), list): out.extend(value[key])
    return out

def _hard_conflicts(process: Mapping[str, Any], wsl: Mapping[str, Any]) -> list[Any]:
    return [item for item in [*_entries(process), *_entries(wsl)] if _name(item) in HARD_NAMES or any(token in _name(item) for token in ("sequential_sfm", "colmap", "b07h_runtime_driver_v10"))]

def ambient_sample(root: Path, workspace_root: Path | str = v7.DEFAULT_C_WORKSPACE, *, process_sampler: Callable[[], Mapping[str, Any]] | None = None, wsl_sampler: Callable[[], Mapping[str, Any]] | None = None, gpu_sampler: Callable[[], Mapping[str, Any]] | None = None, free_bytes_fn: Callable[[Path], int] | None = None, now_fn: Callable[[], str] = v7.utc_now) -> dict[str, Any]:
    root=v7.require_e_root(root); process=dict((process_sampler or v7._default_process_sampler)()); wsl=dict((wsl_sampler or v7._default_wsl_sampler)()); gpu=dict((gpu_sampler or v7._default_gpu_sampler)()); conflicts=_hard_conflicts(process,wsl); informational=[item for item in [*_entries(process),*_entries(wsl)] if _name(item) in INFO_NAMES]; free=int((free_bytes_fn or v7._free_bytes)(root)); c_state=v7.c_workspace_state(workspace_root)
    checks={"hard_processes_clear":not conflicts,"c_workspace_clean":c_state.get("clean") is True,"e_free_threshold":free>=STOP_FREE_BYTES,"cpu_settled":True,"search_settled":True,"gpu_settled":True}
    return {"schema":HISTORY_SCHEMA,"ambient_policy":AMBIENT_POLICY,"admission_policy":ADMISSION_POLICY,"timestamp_utc":now_fn(),"checks":checks,"hard_blockers":{"processes":conflicts,"c_workspace":c_state,"e_free_threshold":{"free_bytes":free,"required_bytes":STOP_FREE_BYTES}},"informational_processes":informational,"noise":{"cpu":process.get("total_processor_percent"),"search_indexer":process.get("search_indexer_percent"),"gpu":gpu,"informational":True},"process":process,"wsl":wsl,"start_allowed":all(checks[key] for key in ("hard_processes_clear","c_workspace_clean","e_free_threshold"))}

def record_ambient_window(root: Path, history: Path, cells: Sequence[str], *, samples: int = 5, sample_seconds: float = 2.0, **kwargs: Any) -> dict[str, Any]:
    observations=[]
    for index in range(samples):
        item=ambient_sample(root, process_sampler=kwargs.get("process_sampler"), wsl_sampler=kwargs.get("wsl_sampler"), gpu_sampler=kwargs.get("gpu_sampler"), free_bytes_fn=kwargs.get("free_bytes_fn")); item["sample_index"]=index+1; observations.append(item)
        if index+1<samples and sample_seconds>0: time.sleep(sample_seconds)
    history=v7.candidate_path(history,root,"v10 history"); history.parent.mkdir(parents=True,exist_ok=True)
    if history.exists(): raise DriverError("v10 history already exists")
    history.write_text("".join(json.dumps(item,sort_keys=True)+"\n" for item in observations),encoding="utf-8")
    return {"schema":HISTORY_SCHEMA,"ambient_policy":AMBIENT_POLICY,"admission_policy":ADMISSION_POLICY,"samples":len(observations),"observations":observations,"hard_blockers":sorted({key for item in observations for key,failed in (("processes",not item["checks"]["hard_processes_clear"]),("c_workspace",not item["checks"]["c_workspace_clean"]),("e_free_threshold",not item["checks"]["e_free_threshold"])) if failed}),"start_allowed":all(item["start_allowed"] for item in observations),"observed_cells":list(cells),"noise_is_informational":True}

def validate_runset_value(value: Mapping[str,Any], root: Path) -> dict[str,Any]:
    if value.get("schema")!=RUNSET_SCHEMA or value.get("ambient_policy")!=ADMISSION_POLICY or value.get("admission_policy")!=ADMISSION_POLICY: raise DriverError("v10 runset policy/schema mismatch")
    recording=value.get("ambient_recording")
    if not isinstance(recording,Mapping) or recording.get("robosim_wsl_processes_are_informational") is not True or recording.get("hard_blockers")!=["visloc_sfm_colmap_driver_processes","c_workspace","e_free_threshold"]: raise DriverError("v10 admission policy is malformed")
    shadow=dict(value); shadow["schema"]=v7.RUNSET_SCHEMA; shadow["supersedes_schema"]=v7.RUNSET_V2_SCHEMA; shadow["ambient_policy"]="recorded"; shadow["ambient_recording"]={"finite_window":True,"noise_is_informational":True,"hard_blockers":["target_processes","c_workspace","e_free_threshold"]}; v7.validate_runset_value(shadow,root); return dict(value)

def validate_runset(path: Path, root: Path, expected_sha256: str) -> dict[str,Any]:
    path=v7.candidate_path(path,root,"v10 runset"); actual=v7.validate_sidecar(path,root,"v10 runset");
    if actual.upper()!=expected_sha256.upper(): raise DriverError("v10 runset SHA mismatch")
    return validate_runset_value(v7.read_json(path),root)

def _ledger(root: Path, path: Path) -> dict[str,Any]:
    path=v7.candidate_path(path,root,"v10 prior ledger")
    if not path.is_file(): raise DriverError("v10 prior ledger is missing")
    v7.validate_sidecar(path,root,"v10 prior ledger")
    value=v7.read_json(path)
    if value.get("schema")!=LEDGER_SCHEMA or value.get("ambient_policy")!=AMBIENT_POLICY or value.get("admission_policy")!=ADMISSION_POLICY: raise DriverError("v10 ledger schema/policy mismatch")
    return value

def _contract(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.get(key) for key in ("candidate_root", "fixed_tools", "invocations", "prepared_inputs", "source", "protocol")}

def bootstrap_v8_to_v10(root: Path, *, v8_ledger: Path, v3_runset: Path, v4_runset: Path, output_ledger: Path) -> Path:
    """Create a V10 ledger whose first record is sealed V8, never rerun."""
    root=v7.require_e_root(root)
    old=v7.candidate_path(v3_runset,root,"v10 V3 runset"); old_value=v7.read_json(old); old_sha=v7.validate_sidecar(old,root,"v10 V3 runset").upper()
    new=v7.candidate_path(v4_runset,root,"v10 V4 runset"); new_value=validate_runset(new,root,v7.validate_sidecar(new,root,"v10 V4 runset")); new_sha=v7.validate_sidecar(new,root,"v10 V4 runset").upper()
    if new_value.get("supersedes_schema") != v7.RUNSET_SCHEMA or str(new_value.get("supersedes_sha256","")).upper()!=old_sha or _contract(old_value)!=_contract(new_value): raise DriverError("v10 V3/V4 immutable command contract transition mismatch")
    source=v7.candidate_path(v8_ledger,root,"v10 V8 ledger"); source_value=v7.read_json(source); source_sha=v7.validate_sidecar(source,root,"v10 V8 ledger").upper()
    if source_value.get("schema")!= "B07H_RUNTIME_DRIVER_LEDGER_V5_RECOVERED" or len(source_value.get("results",[]))!=1: raise DriverError("v10 bootstrap requires sealed one-result V8 ledger")
    record=source_value["results"][0]; result=v7.candidate_path(record.get("result_path"),root,"v10 sealed V8 result"); result_sha=v7.validate_sidecar(result,root,"v10 sealed V8 result").upper(); result_value=v7.read_json(result)
    if result_sha!=str(record.get("result_sha256","")).upper() or result_value.get("runset_sha256","").upper()!=old_sha or result_value.get("schema")!="B07H_RUNTIME_DRIVER_RESULT_V5_RECOVERED": raise DriverError("v10 bootstrap V8 provenance mismatch")
    output=v7.candidate_path(output_ledger,root,"v10 bootstrap ledger")
    if output in {source,result,old,new}: raise DriverError("v10 bootstrap would overwrite sealed evidence")
    transition={"schema":"B07H_RUNTIME_DRIVER_V3_TO_V4_TRANSITION_V1","from_runset_schema":v7.RUNSET_SCHEMA,"from_runset_sha256":old_sha,"to_runset_schema":RUNSET_SCHEMA,"to_runset_sha256":new_sha,"immutable_contract_sha256":hashlib.sha256(json.dumps(_contract(old_value),sort_keys=True,separators=(",",":")).encode()).hexdigest().upper(),"source_v8_ledger_path":str(source),"source_v8_ledger_sha256":source_sha}
    cells=list(result_value["result_cells"]); cumulative={"schema":LEDGER_SCHEMA,"ambient_policy":AMBIENT_POLICY,"admission_policy":ADMISSION_POLICY,"runset_sha256":new_sha,"total_result_cells":9,"expected_cells":list(RESULT_CELLS),"results":[{"schema":result_value["schema"],"invocation_index":1,"invocation":result_value["invocation"],"result_cells":cells,"status":result_value["status"],"result_path":str(result),"result_sha256":result_sha,"source_schema":result_value["schema"],"provenance":{"source_v8_ledger_path":str(source),"source_v8_ledger_sha256":source_sha}}],"cells":{cell:{"invocation_index":1,"result_path":str(result),"result_sha256":result_sha,"status":result_value["cell_results"][0]["status"]} for cell in cells},"denominator":{"total_cells":9,"completed_cells":cells,"completed_count":1,"remaining_cells":[cell for cell in RESULT_CELLS if cell not in cells],"remaining_count":8},"runset_transition":transition,"provenance":{"parent_ledger_path":str(source),"parent_ledger_sha256":source_sha},"updated_utc":v7.utc_now()}
    v7.atomic_json(output,cumulative,root,replace=False); return output

def record_result(root: Path, result_path: Path, payload: Mapping[str,Any], *, prior_ledger: Path, output_ledger: Path) -> dict[str,Any]:
    root=v7.require_e_root(root); result_path=v7.candidate_path(result_path,root,"v10 result"); prior_path=v7.candidate_path(prior_ledger,root,"v10 prior ledger"); output_path=v7.candidate_path(output_ledger,root,"v10 output ledger")
    if output_path == prior_path or output_path.exists() or Path(str(output_path)+".sha256").exists(): raise DriverError("v10 output ledger already exists or aliases prior ledger")
    v7._validate_disjoint_artifacts(root,{"result":result_path,"result_sidecar":Path(str(result_path)+".sha256"),"prior_ledger":prior_path,"prior_ledger_sidecar":Path(str(prior_path)+".sha256"),"output_ledger":output_path,"output_ledger_sidecar":Path(str(output_path)+".sha256")})
    ledger=_ledger(root,prior_path); index=int(payload.get("invocation_index",0)); expected=INVOCATION_CELLS[index-1]; cells=payload.get("cell_results")
    if payload.get("ambient_policy") != AMBIENT_POLICY or payload.get("admission_policy") != ADMISSION_POLICY: raise DriverError("v10 result policy mismatch")
    if index!=len(ledger["results"])+1 or payload.get("invocation")!=expected[0] or payload.get("result_cells")!=list(expected[3]) or not isinstance(cells,list): raise DriverError("v10 strict serial invocation/cell mismatch")
    if str(payload.get("runset_sha256","")).upper()!=str(ledger.get("runset_sha256","")).upper(): raise DriverError("v10 result/prior ledger runset binding mismatch")
    status="dnf" if any(item.get("status")=="dnf" for item in cells) else "success"; value={**dict(payload),"schema":RESULT_SCHEMA,"ambient_policy":AMBIENT_POLICY,"admission_policy":ADMISSION_POLICY,"status":status,"terminal":True,"attempt_terminal":True,"runset_transition":ledger.get("runset_transition"),"finished_utc":str(payload.get("finished_utc") or v7.utc_now())}; v7.reject_gt(value,"v10 result"); result_sha=v7.atomic_json(result_path,value,root,replace=False)
    ledger["results"].append({"schema":RESULT_SCHEMA,"invocation_index":index,"invocation":expected[0],"result_cells":list(expected[3]),"status":status,"result_path":str(result_path),"result_sha256":result_sha});
    for item in cells: ledger["cells"][item["id"]]={"invocation_index":index,"result_path":str(result_path),"result_sha256":result_sha,"status":item["status"]}
    ledger["denominator"]={"total_cells":9,"completed_cells":list(ledger["cells"]),"completed_count":len(ledger["cells"]),"remaining_cells":[cell for cell in RESULT_CELLS if cell not in ledger["cells"]],"remaining_count":9-len(ledger["cells"])}; ledger["updated_utc"]=v7.utc_now(); v7.atomic_json(output_path,ledger,root,replace=False); return value

def run_invocation(root: Path, *, runset: Path, expected_runset_sha256: str, invocation_index: int, runtime_temp: Path, result_path: Path, history_path: Path, prior_ledger: Path, output_ledger: Path) -> int:
    root=v7.require_e_root(root); runset_value=validate_runset(runset,root,expected_runset_sha256); invocation=runset_value["invocations"][invocation_index-1]; output=v7.candidate_path(invocation["output"],root,"v10 invocation output"); v7.require_c_workspace_clean(); ambient=record_ambient_window(root,history_path,invocation["result_cells"])
    if not ambient["start_allowed"]: return 4
    env,_=v7.build_runtime_environment(root,invocation_index,runtime_temp)
    log=v7.candidate_path(Path("logs") / f"B07H_v10_invocation_{invocation_index:02d}.log",root,"v10 driver log"); log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("x",encoding="utf-8") as stream:
        process=subprocess.Popen([str(item) for item in invocation["command"]],cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT); returncode=process.wait()
    manifest_path=output / "manifest.json" if (output / "manifest.json").is_file() else None
    frames=int(invocation["command"][invocation["command"].index("--expected-frames")+1])
    cell_results,_,manifest_sha,manifest_text_path=v9._manifest_cells(root,manifest_path,invocation,frames)
    status="dnf" if any(item["status"]=="dnf" for item in cell_results) else "success"
    payload={"ambient_policy":AMBIENT_POLICY,"admission_policy":ADMISSION_POLICY,"status":status,"mapping_started":True,"invocation_index":invocation_index,"invocation":invocation["id"],"engine":invocation["engine"],"sequence":invocation["sequence"],"result_cells":list(invocation["result_cells"]),"cell_results":cell_results,"runset_sha256":expected_runset_sha256.upper(),"source_sha256":EXPECTED_SOURCE_SHA256,"protocol_sha256":EXPECTED_PROTOCOL_SHA256,"gt_opened":False,"ground_truth_read":False,"ground_truth_materialized":False,"ground_truth_argument_present_anywhere":False,"manifest":{"ambient_policy":AMBIENT_POLICY,"admission_policy":ADMISSION_POLICY,"path":manifest_text_path,"sha256":manifest_sha},"ambient_observation":ambient,"finished_utc":v7.utc_now()}
    record_result(root,result_path,payload,prior_ledger=prior_ledger,output_ledger=output_ledger); return 0 if status=="success" and returncode==0 else 2

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--candidate-root",type=Path,required=True); p.add_argument("--bootstrap-v8",action="store_true"); p.add_argument("--v8-ledger",type=Path); p.add_argument("--v3-runset",type=Path); p.add_argument("--v4-runset",type=Path); p.add_argument("--output-ledger",type=Path); p.add_argument("--runset",type=Path); p.add_argument("--expected-runset-sha256"); p.add_argument("--invocation-index",type=int); p.add_argument("--runtime-temp",type=Path); p.add_argument("--result",type=Path); p.add_argument("--history",type=Path); p.add_argument("--prior-ledger",type=Path); return p.parse_args(argv)

def main(argv: Sequence[str] | None = None) -> int:
    a=parse_args(argv)
    if a.bootstrap_v8:
        required=(a.v8_ledger,a.v3_runset,a.v4_runset,a.output_ledger)
        if any(item is None for item in required): raise DriverError("bootstrap requires --v8-ledger --v3-runset --v4-runset --output-ledger")
        print(bootstrap_v8_to_v10(a.candidate_root,v8_ledger=a.v8_ledger,v3_runset=a.v3_runset,v4_runset=a.v4_runset,output_ledger=a.output_ledger)); return 0
    required=(a.runset,a.expected_runset_sha256,a.invocation_index,a.runtime_temp,a.result,a.history,a.prior_ledger,a.output_ledger)
    if any(item is None for item in required): raise DriverError("run mode requires --runset --expected-runset-sha256 --invocation-index --runtime-temp --result --history --prior-ledger --output-ledger")
    return run_invocation(a.candidate_root,runset=a.runset,expected_runset_sha256=a.expected_runset_sha256,invocation_index=a.invocation_index,runtime_temp=a.runtime_temp,result_path=a.result,history_path=a.history,prior_ledger=a.prior_ledger,output_ledger=a.output_ledger)

if __name__ == "__main__": raise SystemExit(main())

__all__=["RUNSET_SCHEMA","RESULT_SCHEMA","LEDGER_SCHEMA","HISTORY_SCHEMA","DRIVER_VERSION","ADMISSION_POLICY","AMBIENT_POLICY","ambient_sample","record_ambient_window","validate_runset","validate_runset_value","bootstrap_v8_to_v10","record_result","run_invocation","parse_args","main","DriverError"]
