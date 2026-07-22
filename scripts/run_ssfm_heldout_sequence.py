#!/usr/bin/env python3
"""Run one frozen held-out SSfM sequence, deferring GT until all engines exit."""

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
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--mav0", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hierarchical-exe", type=Path, required=True)
    parser.add_argument("--hierarchical-build-revision", required=True)
    parser.add_argument("--colmap", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def parse_pinhole(calib: Path) -> tuple[float, float, float, float]:
    line = next(
        line
        for line in calib.read_text(encoding="utf-8").splitlines()
        if line.startswith("P0:")
    )
    values = [float(value) for value in line.split()[1:]]
    return values[0], values[5], values[2], values[6]


def main() -> int:
    args = parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    if args.sequence not in protocol["selection"]["held_out_sequences"]:
        raise ValueError(f"not a frozen held-out sequence: {args.sequence}")
    if args.mav0.parent.name != args.sequence:
        raise ValueError("mav0/sequence mismatch")
    if args.hierarchical_build_revision != protocol["policy"]["source_revision"]:
        raise ValueError("hierarchical build revision does not match frozen policy")
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    for path in (args.python, args.hierarchical_exe, args.colmap):
        if not path.is_file():
            raise FileNotFoundError(path)

    args.out_dir.mkdir(parents=True)
    logs = args.out_dir / "logs"
    logs.mkdir()
    prepared = args.out_dir / "prepared"
    hierarchical = args.out_dir / "hierarchical"
    colmap = args.out_dir / "colmap"
    final = args.out_dir / "final"
    scripts = {
        "prepare": REPO / "scripts" / "prepare_ssfm_heldout_euroc_inputs.py",
        "hierarchical": REPO / "scripts" / "run_hierarchical_sfm_frozen.py",
        "colmap": REPO / "scripts" / "run_colmap_ssfm_frozen.py",
        "finalize": REPO / "scripts" / "finalize_ssfm_heldout_sequence.py",
    }
    hierarchical_executable = {
        "path": str(args.hierarchical_exe.resolve()),
        "sha256": sha256(args.hierarchical_exe),
        "build_revision": args.hierarchical_build_revision,
    }
    colmap_executable = {
        "path": str(args.colmap.resolve()),
        "sha256": sha256(args.colmap),
    }
    python_executable = {
        "path": str(args.python.resolve()),
        "sha256": sha256(args.python),
    }
    runner_scripts = {
        name: {"path": str(path.resolve()), "sha256": sha256(path)}
        for name, path in scripts.items()
    }
    commands = {}
    returncodes = {}
    started_utc = timestamp()

    def write_manifest(status: str, failure_reason: str | None = None) -> Path:
        final_manifest = final / "manifest.json"
        manifest = {
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "sequence": args.sequence,
            "status": status,
            "failure_reason": failure_reason,
            "started_utc": started_utc,
            "finished_utc": timestamp(),
            "ground_truth_path_disclosed_only_to_finalizer": (
                "finalize" in commands
            ),
            "hierarchical_executable": hierarchical_executable,
            "colmap_executable": colmap_executable,
            "python_executable": python_executable,
            "runner_scripts": runner_scripts,
            "commands": commands,
            "returncodes": returncodes,
            "final_manifest": (
                str(final_manifest.resolve()) if final_manifest.is_file() else None
            ),
            "final_manifest_sha256": (
                sha256(final_manifest) if final_manifest.is_file() else None
            ),
        }
        manifest_path = args.out_dir / "manifest.json"
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)
        return manifest_path

    def fail(reason: str, returncode: int = 1) -> int:
        manifest_path = write_manifest("failed", reason)
        print(f"ERROR: {reason}", file=sys.stderr)
        print(manifest_path)
        return returncode if returncode != 0 else 1

    commands["prepare"] = [
        str(args.python),
        str(scripts["prepare"]),
        "--protocol",
        str(args.protocol),
        "--sequence",
        args.sequence,
        "--mav0",
        str(args.mav0),
        "--out-dir",
        str(prepared),
        "--python",
        str(args.python),
        "--device",
        args.device,
    ]
    returncodes["prepare"] = run(commands["prepare"], logs / "prepare.log")
    if returncodes["prepare"] != 0:
        return fail(
            "held-out input preparation failed",
            returncodes["prepare"],
        )
    prepared_manifest_path = prepared / "manifest.json"
    if not prepared_manifest_path.is_file():
        return fail("input preparation exited without a manifest")
    try:
        prepared_manifest = json.loads(
            prepared_manifest_path.read_text(encoding="utf-8")
        )
        expected_frames = int(prepared_manifest["expected_frames"])
        fx, fy, cx, cy = parse_pinhole(prepared / "rect" / "calib.txt")
    except (OSError, ValueError, KeyError, StopIteration, json.JSONDecodeError) as error:
        return fail(f"invalid prepared input evidence: {error}")

    commands["hierarchical"] = [
        str(args.python),
        str(scripts["hierarchical"]),
        "--exe",
        str(args.hierarchical_exe),
        "--features-dir",
        str(prepared / "features"),
        "--timestamps",
        str(prepared / "rect" / "timestamps.txt"),
        "--out-dir",
        str(hierarchical),
        "--expected-frames",
        str(expected_frames),
        "--build-git-revision",
        args.hierarchical_build_revision,
        "--fx",
        str(fx),
        "--fy",
        str(fy),
        "--cx",
        str(cx),
        "--cy",
        str(cy),
    ]
    returncodes["hierarchical"] = run(
        commands["hierarchical"], logs / "hierarchical.log"
    )
    if not (hierarchical / "manifest.json").is_file():
        return fail(
            "hierarchical runner exited without a DNF/success manifest",
            returncodes["hierarchical"],
        )

    commands["colmap"] = [
        str(args.python),
        str(scripts["colmap"]),
        "--protocol",
        str(args.protocol),
        "--sequence",
        args.sequence,
        "--prepared-dir",
        str(prepared),
        "--out-dir",
        str(colmap),
        "--expected-frames",
        str(expected_frames),
        "--colmap",
        str(args.colmap),
    ]
    returncodes["colmap"] = run(commands["colmap"], logs / "colmap.log")
    if not (colmap / "manifest.json").is_file():
        return fail(
            "COLMAP runner exited without a DNF/success manifest",
            returncodes["colmap"],
        )

    # This is deliberately the first command that receives a GT path. Every
    # timed mapper process above has exited, including failed/DNF engines.
    commands["finalize"] = [
        str(args.python),
        str(scripts["finalize"]),
        "--protocol",
        str(args.protocol),
        "--sequence",
        args.sequence,
        "--prepared-dir",
        str(prepared),
        "--hierarchical-dir",
        str(hierarchical),
        "--colmap-dir",
        str(colmap),
        "--ground-truth-csv",
        str(args.mav0 / "state_groundtruth_estimate0" / "data.csv"),
        "--out-dir",
        str(final),
    ]
    returncodes["finalize"] = run(commands["finalize"], logs / "finalize.log")
    if returncodes["finalize"] != 0:
        return fail(
            "held-out suite finalization failed",
            returncodes["finalize"],
        )
    if not (final / "manifest.json").is_file():
        return fail("finalizer exited without a manifest")

    manifest_path = write_manifest("success")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
