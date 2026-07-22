#!/usr/bin/env python3
"""Verify the complete frozen held-out SSfM evidence chain for release."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ssfm_external_baseline_evidence import validate_external_baseline_manifest


REPO = Path(__file__).resolve().parents[1]
ENGINES = (
    "visloc_hierarchical",
    "colmap_incremental",
    "colmap_global",
    "gluemap",
    "instantsfm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_file_evidence(evidence: dict, label: str) -> Path:
    path = Path(evidence["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    actual = sha256(path)
    if actual != evidence["sha256"]:
        raise ValueError(f"{label} hash mismatch: {actual}")
    return path


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def require_sampling_cadence(record: dict, label: str) -> float:
    value = record.get("resource_poll_seconds")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} lacks numeric resource_poll_seconds")
    value = float(value)
    if not 0.1 <= value <= 10.0:
        raise ValueError(f"{label} has invalid resource_poll_seconds={value}")
    return value


def require_uniform_sampling_cadence(cadences: list[tuple[str, float]]) -> float:
    if not cadences:
        raise ValueError("resource sampling cadence evidence is empty")
    distinct = {value for _, value in cadences}
    if len(distinct) != 1:
        rendered = ", ".join(f"{label}={value}" for label, value in cadences)
        raise ValueError(f"resource sampling cadence mismatch: {rendered}")
    return next(iter(distinct))


def verify_success_sequence(
    sequence_root: Path,
    runner: dict,
    protocol_sha256: str,
) -> dict:
    if not runner.get("ground_truth_materialized_only_after_timed_engines_exited"):
        raise ValueError("successful runner lacks deferred-GT proof")
    if runner.get("status") != "success":
        raise ValueError("finalized sequence runner is not successful")
    for stage in (
        "hierarchical",
        "colmap",
        "external",
        "materialize_ground_truth",
        "finalize",
    ):
        if stage not in runner["commands"] or stage not in runner["returncodes"]:
            raise ValueError(f"successful runner is missing stage {stage}")
    if any(runner["returncodes"][stage] != 0 for stage in runner["returncodes"]):
        raise ValueError("successful runner contains a nonzero stage")

    prepared_path = sequence_root / "prepared" / "manifest.json"
    hierarchical_path = sequence_root / "hierarchical" / "manifest.json"
    colmap_path = sequence_root / "colmap" / "manifest.json"
    external_path = sequence_root / "external" / "manifest.json"
    ground_truth_path = sequence_root / "ground_truth" / "manifest.json"
    final_path = sequence_root / "final" / "manifest.json"
    prepared = read_json(prepared_path)
    hierarchical = read_json(hierarchical_path)
    colmap = read_json(colmap_path)
    external = read_json(external_path)
    ground_truth = read_json(ground_truth_path)
    final = read_json(final_path)
    if ground_truth["protocol_sha256"] != protocol_sha256:
        raise ValueError("materialized GT protocol mismatch")
    if final["protocol_sha256"] != protocol_sha256:
        raise ValueError("final result protocol mismatch")
    if set(final["results"]) != set(ENGINES):
        raise ValueError("final result engine set mismatch")
    if hierarchical["protocol"].get("ground_truth_read") is not False:
        raise ValueError("hierarchical engine read GT")
    if colmap.get("ground_truth_read") is not False:
        raise ValueError("COLMAP engine read GT")
    sampling_cadences = []
    for stage_name, stage in prepared["stages"].items():
        label = f"prepared stage {stage_name}"
        sampling_cadences.append((label, require_sampling_cadence(stage, label)))
    sampling_cadences.append(
        (
            "hierarchical mapper",
            require_sampling_cadence(hierarchical["mapper"], "hierarchical mapper"),
        )
    )
    for stage_name, stage in colmap["stages"].items():
        label = f"COLMAP stage {stage_name}"
        sampling_cadences.append((label, require_sampling_cadence(stage, label)))
    for engine, result in external["results"].items():
        if result["status"] == "success":
            label = f"external engine {engine}"
            sampling_cadences.append((label, require_sampling_cadence(result, label)))
    for engine, result in final["results"].items():
        if result["status"] != "success":
            continue
        cadence = result.get("resource_poll_seconds")
        if isinstance(cadence, dict):
            for stage_name, value in cadence.items():
                label = f"final engine {engine} stage {stage_name}"
                sampling_cadences.append(
                    (
                        label,
                        require_sampling_cadence(
                            {"resource_poll_seconds": value}, label
                        ),
                    )
                )
        else:
            label = f"final engine {engine}"
            sampling_cadences.append((label, require_sampling_cadence(result, label)))
    require_uniform_sampling_cadence(sampling_cadences)
    validate_external_baseline_manifest(
        external,
        sequence=runner["sequence"],
        heldout_protocol_sha256=protocol_sha256,
        external_protocol_sha256=runner["external_protocol"]["sha256"],
        manifest_dir=external_path.parent,
    )

    engine_evidence = ground_truth["engine_exit_evidence"]
    if verify_file_evidence(
        engine_evidence["hierarchical"], "GT hierarchical exit evidence"
    ).resolve() != hierarchical_path.resolve():
        raise ValueError("GT hierarchical exit evidence path mismatch")
    if verify_file_evidence(
        engine_evidence["colmap"], "GT COLMAP exit evidence"
    ).resolve() != colmap_path.resolve():
        raise ValueError("GT COLMAP exit evidence path mismatch")
    if verify_file_evidence(
        engine_evidence["external"], "GT external exit evidence"
    ).resolve() != external_path.resolve():
        raise ValueError("GT external exit evidence path mismatch")
    materialized_utc = parse_timestamp(ground_truth["materialized_utc"])
    for label, manifest in (
        ("hierarchical", hierarchical),
        ("colmap", colmap),
        ("external", external),
    ):
        if materialized_utc < parse_timestamp(manifest["finished_utc"]):
            raise ValueError(f"GT materialized before {label} exited")

    for label, evidence in final["input_manifests"].items():
        verify_file_evidence(evidence, f"final input {label}")
    gt_csv = verify_file_evidence(final["ground_truth"], "final ground truth")
    if gt_csv.resolve() != (sequence_root / "ground_truth" / "data.csv").resolve():
        raise ValueError("final ground-truth path mismatch")
    if runner["final_manifest_sha256"] != sha256(final_path):
        raise ValueError("runner/final manifest hash mismatch")
    return {
        "kind": "success",
        "final_manifest_sha256": sha256(final_path),
        "ground_truth_manifest_sha256": sha256(ground_truth_path),
        "result_statuses": {
            engine: final["results"][engine]["status"] for engine in ENGINES
        },
    }


def compare_regenerated_summary(
    protocol: Path,
    suite_root: Path,
    recorded: dict,
    summarizer: Path,
) -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        regenerated_path = Path(raw_tmp) / "summary.json"
        subprocess.run(
            [
                sys.executable,
                str(summarizer),
                "--protocol",
                str(protocol),
                "--suite-root",
                str(suite_root),
                "--out",
                str(regenerated_path),
            ],
            cwd=REPO,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        regenerated = read_json(regenerated_path)
    recorded_without_time = {key: value for key, value in recorded.items() if key != "generated_utc"}
    regenerated_without_time = {
        key: value for key, value in regenerated.items() if key != "generated_utc"
    }
    if recorded_without_time != regenerated_without_time:
        raise ValueError("recorded summary differs from deterministic regeneration")


def main() -> int:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    verifier_path = Path(__file__).resolve()
    verifier_sha256 = sha256(verifier_path)
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    sequences = protocol["selection"]["held_out_sequences"]
    suite_manifest_path = args.suite_root / "suite_manifest.json"
    suite = read_json(suite_manifest_path)
    if suite.get("status") != "complete":
        raise ValueError("suite execution is not complete")
    if suite.get("protocol_sha256") != protocol_sha256:
        raise ValueError("suite/protocol hash mismatch")
    if suite.get("sequence_order") != sequences:
        raise ValueError("suite sequence order mismatch")
    if suite.get("serial_execution") is not True:
        raise ValueError("suite does not assert serial execution")
    if suite.get("post_result_tuning_permitted") is not False:
        raise ValueError("suite permits post-result tuning")

    for label, evidence in suite["frozen_evidence"]["all_frozen_files"].items():
        verify_file_evidence(evidence, f"frozen file {label}")
    if list(suite["runs"]) != sequences:
        raise ValueError("suite run set/order mismatch")

    sequence_audit = {}
    previous_finished = None
    for sequence in sequences:
        run = suite["runs"][sequence]
        started = parse_timestamp(run["started_utc"])
        finished = parse_timestamp(run["finished_utc"])
        if finished < started:
            raise ValueError(f"negative run duration for {sequence}")
        if previous_finished is not None and started < previous_finished:
            raise ValueError("suite sequence runs overlap")
        previous_finished = finished
        runner_path = verify_file_evidence(
            run["runner_manifest"], f"runner manifest {sequence}"
        )
        expected_runner_path = args.suite_root / sequence / "manifest.json"
        if runner_path.resolve() != expected_runner_path.resolve():
            raise ValueError("runner manifest escaped its sequence directory")
        runner = read_json(runner_path)
        if runner["sequence"] != sequence:
            raise ValueError("runner sequence mismatch")
        if runner["protocol_sha256"] != protocol_sha256:
            raise ValueError("runner protocol mismatch")
        final_path = args.suite_root / sequence / "final" / "manifest.json"
        if final_path.is_file():
            sequence_audit[sequence] = verify_success_sequence(
                args.suite_root / sequence,
                runner,
                protocol_sha256,
            )
        else:
            if runner.get("status") != "failed":
                raise ValueError("sequence lacks final result and explicit failure")
            sequence_audit[sequence] = {
                "kind": "runner_failure",
                "reason": runner.get("failure_reason"),
            }

    summary_path = verify_file_evidence(suite["summary"], "suite summary")
    recorded_summary = read_json(summary_path)
    summarizer = Path(
        suite["frozen_evidence"]["all_frozen_files"]["summarizer"]["path"]
    )
    compare_regenerated_summary(
        args.protocol,
        args.suite_root,
        recorded_summary,
        summarizer,
    )
    output = {
        "schema_version": 1,
        "status": "verified",
        "verified_utc": timestamp(),
        "verifier": {
            "path": str(verifier_path),
            "sha256": verifier_sha256,
        },
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "suite_manifest": {
            "path": str(suite_manifest_path.resolve()),
            "sha256": sha256(suite_manifest_path),
        },
        "summary": {"path": str(summary_path.resolve()), "sha256": sha256(summary_path)},
        "sequence_audit": sequence_audit,
    }
    if sha256(verifier_path) != verifier_sha256:
        raise ValueError("release verifier changed during verification")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
