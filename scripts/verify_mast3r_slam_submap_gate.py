#!/usr/bin/env python3
"""Independently verify an R1e MASt3R-SLAM submap gate transaction."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from run_mast3r_slam_submap_gate import (
    MIN_INLIER_RATIO,
    parse_pass_metrics,
    path_for_probe,
    scale_gate_passes,
    sha256,
    validate_export_manifest,
    validate_probe_source_revision,
    verify_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def command_options(command: list[str]) -> dict[str, str]:
    if not isinstance(command, list) or len(command) < 3 or len(command) % 2 == 0:
        raise ValueError("R1e probe command is not executable plus flag/value pairs")
    options = {}
    for flag, value in zip(command[1::2], command[2::2]):
        if not isinstance(flag, str) or not flag.startswith("--"):
            raise ValueError("R1e probe command contains a malformed flag")
        if flag in options:
            raise ValueError(f"R1e probe command repeats {flag}")
        options[flag] = str(value)
    return options


def verify_gate_manifest(path: Path) -> dict:
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("schema_version") != 1:
        raise ValueError("unsupported R1e gate manifest schema")
    if gate.get("status") != "success":
        raise ValueError("R1e gate transaction did not complete successfully")
    if gate.get("ground_truth_read") is not False:
        raise ValueError("R1e gate does not prove ground-truth isolation")
    if gate.get("backend_writeback") is not False:
        raise ValueError("R1e measurement transaction performed backend writeback")
    if gate.get("returncode") != 0:
        raise ValueError("R1e probe exited unsuccessfully")
    if gate.get("changed_inputs") != []:
        raise ValueError("R1e gate records changed frozen inputs")

    export_path = verify_evidence(gate["export_manifest"], "R1e export manifest")
    export, _, old_points, new_points = validate_export_manifest(export_path)
    verify_evidence(gate["probe_executable"], "R1e probe executable")
    frozen_inputs = gate.get("frozen_inputs")
    if not isinstance(frozen_inputs, list) or not frozen_inputs:
        raise ValueError("R1e gate lacks frozen input evidence")
    for index, item in enumerate(frozen_inputs):
        verify_evidence(item, f"R1e gate frozen input {index}")

    source = gate.get("probe_source")
    if not isinstance(source, dict):
        raise ValueError("R1e gate lacks probe source provenance")
    verified_source = validate_probe_source_revision(source.get("build_revision", ""))
    if source != verified_source:
        raise ValueError("R1e probe source provenance differs from committed source")

    log_path = verify_evidence(gate["probe_log"], "R1e probe log")
    metrics = parse_pass_metrics(log_path.read_text(encoding="utf-8"))
    recomputed_pass = scale_gate_passes(metrics)
    recorded = gate.get("same_side_scale_transfer_gate")
    if not isinstance(recorded, dict):
        raise ValueError("R1e gate result is missing")
    if recorded.get("minimum_inlier_ratio") != MIN_INLIER_RATIO:
        raise ValueError("R1e gate changed the frozen minimum inlier ratio")
    if recorded.get("metrics") != metrics:
        raise ValueError("R1e recorded metrics differ from probe log")
    if recorded.get("passed") is not recomputed_pass:
        raise ValueError("R1e recorded verdict differs from recomputed verdict")

    options = command_options(gate.get("command"))
    command = gate["command"]
    probe_path = Path(gate["probe_executable"]["path"])
    if Path(command[0]).resolve() != probe_path.resolve():
        raise ValueError("R1e probe command executable differs from frozen evidence")
    required_flags = {
        "--dump-dir",
        "--lightglue-dir",
        "--learned-old-points",
        "--learned-new-points",
        "--old-anchor",
        "--new-anchor",
        "--radius",
    }
    if set(options) != required_flags:
        raise ValueError("R1e probe command flag set differs from frozen transaction")
    expected_scalars = {
        "--old-anchor": str(export["sides"]["old"]["anchor_arrival"]),
        "--new-anchor": str(export["sides"]["new"]["anchor_arrival"]),
        "--radius": str(export["radius"]),
    }
    for flag, expected in expected_scalars.items():
        if options[flag] != expected:
            raise ValueError(f"R1e probe command {flag} differs from export manifest")
    lightglue_directory = gate.get("lightglue_directory")
    if not isinstance(lightglue_directory, str):
        raise ValueError("R1e gate lacks LightGlue directory binding")
    expected_paths = (
        ("--dump-dir", Path(export["descriptor_manifest"]).parent),
        ("--lightglue-dir", Path(lightglue_directory)),
        ("--learned-old-points", old_points),
        ("--learned-new-points", new_points),
    )
    for flag, expected_path in expected_paths:
        expected = path_for_probe(expected_path, probe_path)
        if options[flag].replace("\\", "/").casefold() != expected.replace(
            "\\", "/"
        ).casefold():
            raise ValueError(f"R1e probe command {flag} differs from export evidence")

    return {
        "gate_passed": recomputed_pass,
        "metrics": metrics,
        "export_manifest": gate["export_manifest"],
        "probe_executable": gate["probe_executable"],
        "probe_source": verified_source,
        "probe_log": gate["probe_log"],
    }


def main() -> int:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    verifier_path = Path(__file__).resolve()
    verifier_sha256 = sha256(verifier_path)
    audit = verify_gate_manifest(args.gate_manifest)
    if sha256(verifier_path) != verifier_sha256:
        raise ValueError("R1e release verifier changed during verification")
    output = {
        "schema_version": 1,
        "status": "verified",
        "verified_utc": timestamp(),
        "verifier": {"path": str(verifier_path), "sha256": verifier_sha256},
        "gate_manifest": {
            "path": str(args.gate_manifest.resolve()),
            "sha256": sha256(args.gate_manifest),
        },
        **audit,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out, output)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
