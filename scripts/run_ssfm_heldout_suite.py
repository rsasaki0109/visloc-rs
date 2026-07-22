#!/usr/bin/env python3
"""Run the frozen three-sequence SSfM suite serially and retain every failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--external-protocol", type=Path, required=True)
    parser.add_argument("--external-setup-manifest", type=Path, required=True)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hierarchical-exe", type=Path, required=True)
    parser.add_argument("--hierarchical-build-revision", required=True)
    parser.add_argument("--colmap", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], log: Path) -> int:
    with log.open("w", encoding="utf-8") as stream:
        stream.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
        stream.flush()
        completed = subprocess.run(
            command,
            cwd=REPO,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def changed_frozen_files(files: dict[str, dict]) -> list[str]:
    changed = []
    for label, evidence in files.items():
        path = Path(evidence["path"])
        if not path.is_file() or sha256(path) != evidence["sha256"]:
            changed.append(label)
    return changed


def validate_extraction(
    extracted_root: Path,
    sequences: list[str],
    protocol_sha256: str,
) -> dict:
    manifest_path = extracted_root / "extraction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "success":
        raise ValueError("extraction manifest is not successful")
    if manifest.get("protocol_sha256") != protocol_sha256:
        raise ValueError("extraction/protocol hash mismatch")
    if manifest.get("ground_truth_read") is not False:
        raise ValueError("extraction manifest does not prove GT isolation")
    if set(manifest.get("sequences", {})) != set(sequences):
        raise ValueError("extraction manifest sequence set mismatch")
    for sequence in sequences:
        evidence = manifest["sequences"][sequence]
        if evidence.get("ground_truth_materialized") is not False:
            raise ValueError(f"GT was materialized before suite start: {sequence}")
        mav0 = extracted_root / sequence / "mav0"
        if not mav0.is_dir():
            raise FileNotFoundError(mav0)
        if (mav0 / "state_groundtruth_estimate0").exists():
            raise ValueError(f"GT directory exists before suite start: {sequence}")
    return {
        "path": str(manifest_path.resolve()),
        "sha256": sha256(manifest_path),
    }


def synthetic_runner_failure(
    path: Path,
    protocol: dict,
    protocol_sha256: str,
    sequence: str,
    reason: str,
    command: list[str],
    returncode: int,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        path / "manifest.json",
        {
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_sha256,
            "sequence": sequence,
            "status": "failed",
            "failure_reason": reason,
            "source": "run_ssfm_heldout_suite.py synthetic failure evidence",
            "command": command,
            "returncode": returncode,
            "finished_utc": timestamp(),
        },
    )


def main() -> int:
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    external_protocol = json.loads(args.external_protocol.read_text(encoding="utf-8"))
    if external_protocol["heldout_protocol"]["sha256"] != protocol_sha256:
        raise ValueError("external protocol does not bind held-out protocol")
    sequences = protocol["selection"]["held_out_sequences"]
    if len(sequences) != 3 or len(set(sequences)) != 3:
        raise ValueError("protocol must bind exactly three unique sequences")
    if args.hierarchical_build_revision != protocol["policy"]["source_revision"]:
        raise ValueError("hierarchical build revision does not match frozen policy")
    for executable in (args.python, args.hierarchical_exe, args.colmap):
        if not executable.is_file():
            raise FileNotFoundError(executable)
    if not args.download_dir.is_dir():
        raise FileNotFoundError(args.download_dir)
    if not args.external_setup_manifest.is_file():
        raise FileNotFoundError(args.external_setup_manifest)
    extraction = validate_extraction(
        args.extracted_root,
        sequences,
        protocol_sha256,
    )

    sequence_runner = REPO / "scripts" / "run_ssfm_heldout_sequence.py"
    summarizer = REPO / "scripts" / "summarize_ssfm_heldout_suite.py"
    frozen_paths = {
        "protocol": args.protocol,
        "external_protocol": args.external_protocol,
        "external_setup_manifest": args.external_setup_manifest,
        "extraction_manifest": Path(extraction["path"]),
        "download_manifest": args.download_dir / "download_manifest.json",
        "hierarchical_executable": args.hierarchical_exe,
        "colmap_executable": args.colmap,
        "python_executable": args.python,
        "superpoint_checkpoint": (
            Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "superpoint_v1.pth"
        ),
        "suite_runner": Path(__file__),
        "sequence_runner": sequence_runner,
        "preparer": REPO / "scripts" / "prepare_ssfm_heldout_euroc_inputs.py",
        "rectifier": REPO / "scripts" / "rectify_euroc_stereo.py",
        "feature_exporter": REPO / "scripts" / "export_superpoint_lightglue.py",
        "hierarchical_runner": REPO / "scripts" / "run_hierarchical_sfm_frozen.py",
        "colmap_runner": REPO / "scripts" / "run_colmap_ssfm_frozen.py",
        "external_runner": (
            REPO / "scripts" / "run_external_ssfm_baselines_frozen.py"
        ),
        "external_evidence_schema": (
            REPO / "scripts" / "ssfm_external_baseline_evidence.py"
        ),
        "gt_materializer": (
            REPO / "scripts" / "materialize_ssfm_heldout_ground_truth.py"
        ),
        "finalizer": REPO / "scripts" / "finalize_ssfm_heldout_sequence.py",
        "trajectory_evaluator": REPO / "scripts" / "evaluate_euroc_trajectory.py",
        "process_monitor": REPO / "scripts" / "benchmark_process_metrics.py",
        "summarizer": summarizer,
    }
    for label, path in frozen_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"frozen input {label}: {path}")
    frozen_files = {
        label: {"path": str(path.resolve()), "sha256": sha256(path)}
        for label, path in frozen_paths.items()
    }
    frozen_evidence = {
        "protocol": {
            "path": str(args.protocol.resolve()),
            "sha256": protocol_sha256,
        },
        "extraction_manifest": extraction,
        "hierarchical_executable": {
            "path": str(args.hierarchical_exe.resolve()),
            "sha256": sha256(args.hierarchical_exe),
            "build_revision": args.hierarchical_build_revision,
        },
        "colmap_executable": {
            "path": str(args.colmap.resolve()),
            "sha256": sha256(args.colmap),
        },
        "python_executable": {
            "path": str(args.python.resolve()),
            "sha256": sha256(args.python),
        },
        "scripts": {
            "sequence_runner": {
                "path": str(sequence_runner.resolve()),
                "sha256": sha256(sequence_runner),
            },
            "summarizer": {
                "path": str(summarizer.resolve()),
                "sha256": sha256(summarizer),
            },
        },
        "all_frozen_files": frozen_files,
    }

    args.out_dir.mkdir(parents=True)
    logs = args.out_dir / "_suite_logs"
    logs.mkdir()
    suite_manifest_path = args.out_dir / "suite_manifest.json"
    suite_manifest = {
        "schema_version": 1,
        "status": "in_progress",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "started_utc": timestamp(),
        "sequence_order": sequences,
        "serial_execution": True,
        "post_result_tuning_permitted": False,
        "frozen_evidence": frozen_evidence,
        "runs": {},
    }
    write_json_atomic(suite_manifest_path, suite_manifest)

    def reject_if_frozen_files_changed(boundary: str) -> bool:
        changed = changed_frozen_files(frozen_files)
        if not changed:
            return False
        suite_manifest["status"] = "invalidated"
        suite_manifest["invalidation"] = {
            "boundary": boundary,
            "changed_files": changed,
            "reason": "frozen evidence changed during the held-out suite",
        }
        suite_manifest["finished_utc"] = timestamp()
        write_json_atomic(suite_manifest_path, suite_manifest)
        print(suite_manifest_path)
        return True

    for sequence in sequences:
        if reject_if_frozen_files_changed(f"before:{sequence}"):
            return 2
        sequence_out = args.out_dir / sequence
        command = [
            str(args.python),
            str(sequence_runner),
            "--protocol",
            str(args.protocol),
            "--external-protocol",
            str(args.external_protocol),
            "--external-setup-manifest",
            str(args.external_setup_manifest),
            "--sequence",
            sequence,
            "--mav0",
            str(args.extracted_root / sequence / "mav0"),
            "--download-dir",
            str(args.download_dir),
            "--out-dir",
            str(sequence_out),
            "--hierarchical-exe",
            str(args.hierarchical_exe),
            "--hierarchical-build-revision",
            args.hierarchical_build_revision,
            "--colmap",
            str(args.colmap),
            "--python",
            str(args.python),
            "--device",
            args.device,
        ]
        started_utc = timestamp()
        returncode = run(command, logs / f"{sequence}.log")
        runner_manifest = sequence_out / "manifest.json"
        if not runner_manifest.is_file():
            synthetic_runner_failure(
                sequence_out,
                protocol,
                protocol_sha256,
                sequence,
                "sequence runner exited without a manifest",
                command,
                returncode,
            )
        suite_manifest["runs"][sequence] = {
            "started_utc": started_utc,
            "finished_utc": timestamp(),
            "command": command,
            "returncode": returncode,
            "runner_manifest": {
                "path": str(runner_manifest.resolve()),
                "sha256": sha256(runner_manifest),
            },
        }
        write_json_atomic(suite_manifest_path, suite_manifest)
        if reject_if_frozen_files_changed(f"after:{sequence}"):
            return 2

    if reject_if_frozen_files_changed("before:summary"):
        return 2
    summary_path = args.out_dir / "summary.json"
    summary_command = [
        str(args.python),
        str(summarizer),
        "--protocol",
        str(args.protocol),
        "--suite-root",
        str(args.out_dir),
        "--out",
        str(summary_path),
    ]
    summary_returncode = run(summary_command, logs / "summary.log")
    suite_manifest["summary"] = {
        "command": summary_command,
        "returncode": summary_returncode,
        "path": str(summary_path.resolve()) if summary_path.is_file() else None,
        "sha256": sha256(summary_path) if summary_path.is_file() else None,
    }
    if reject_if_frozen_files_changed("after:summary"):
        return 2
    suite_manifest["status"] = (
        "complete" if summary_returncode == 0 and summary_path.is_file() else "failed"
    )
    suite_manifest["finished_utc"] = timestamp()
    write_json_atomic(suite_manifest_path, suite_manifest)
    print(suite_manifest_path)
    return 0 if suite_manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
