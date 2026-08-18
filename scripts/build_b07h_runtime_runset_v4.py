#!/usr/bin/env python3
"""Create the V10 relaxed-admission runset from immutable V3 bytes."""
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path
from typing import Sequence
sys.dont_write_bytecode = True
import run_b07h_runtime_driver_v7 as v7  # noqa: E402

RUNSET_SCHEMA = "B07H_GT_FREE_RUNTIME_RUNSET_V4"
POLICY = "relaxed_recorded_robosim"

def build(source: Path, output: Path, root: Path, expected_source_sha256: str) -> Path:
    root = v7.require_e_root(root)
    source = v7.require_regular_candidate_file(source, root, "v4 source runset")
    source_sha = v7.validate_sidecar(source, root, "v4 source runset").upper()
    value = v7.read_json(source)
    if value.get("schema") != v7.RUNSET_SCHEMA or source_sha != expected_source_sha256.upper():
        raise v7.DriverError("v4 builder requires the exact immutable V3 runset")
    result = copy.deepcopy(value)
    result["schema"] = RUNSET_SCHEMA
    result["supersedes_schema"] = v7.RUNSET_SCHEMA
    result["supersedes_sha256"] = source_sha
    result["admission_policy"] = POLICY
    result["ambient_policy"] = POLICY
    result["ambient_recording"] = {"finite_window": True, "noise_is_informational": True, "hard_blockers": ["visloc_sfm_colmap_driver_processes", "c_workspace", "e_free_threshold"], "informational_processes": ["robosim", "cargo", "rustc", "search_indexer"], "robosim_wsl_processes_are_informational": True, "start_gate": "relaxed-hard-blockers-only"}
    if isinstance(result.get("storage_policy"), dict): result["storage_policy"]["ambient_policy"] = POLICY
    result["runtime_policy"]["performance_claim"] = False
    v7.reject_gt(result, "v4 runset")
    output = v7.candidate_path(output, root, "v4 runset output")
    if output == source:
        raise v7.DriverError("v4 runset would overwrite V3")
    v7.atomic_json(output, result, root, replace=False)
    return output

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--candidate-root", type=Path, required=True); p.add_argument("--source", type=Path, required=True); p.add_argument("--expected-source-sha256", required=True); p.add_argument("--output", type=Path, required=True); return p.parse_args(argv)

def main(argv: Sequence[str] | None = None) -> int:
    a = parse_args(argv); print(build(a.source, a.output, a.candidate_root, a.expected_source_sha256)); return 0

if __name__ == "__main__": raise SystemExit(main())
