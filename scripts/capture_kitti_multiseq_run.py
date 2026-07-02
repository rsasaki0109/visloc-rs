#!/usr/bin/env python3
"""Capture a KITTI multi-sequence VO run into the benchmark registry.

This is the full-run companion to `capture_kitti_loop_retrieval_recall.py`.
It expects the trajectory, VO log, and `evaluation.json` emitted by
`scripts/run_kitti_multiseq_benchmark.sh`, then records ATE, loop count,
command/config, and key artifacts in `benchmarks/registry/runs/kitti/`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ID = "kitti-multiseq"
BENCHMARK_NAME = "KITTI multi-sequence full-stack"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, help="KITTI odometry sequence, e.g. 02")
    parser.add_argument("--evaluation-json", type=Path, required=True)
    parser.add_argument("--vo-log", type=Path, required=True)
    parser.add_argument("--vo-poses", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--dataset-version", default="KITTI odometry grayscale")
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--registry-dir", type=Path, default=Path("benchmarks/registry/runs/kitti"))
    parser.add_argument("--run-id")
    parser.add_argument("--command")
    parser.add_argument("--claim-scope", choices=["headline", "supporting", "exploratory", "negative"], default="supporting")
    parser.add_argument("--status", choices=["success", "dnf", "failure"], default="success")
    parser.add_argument("--failure-reason")
    parser.add_argument("--config", action="append", default=[], help="KEY=VALUE, JSON values accepted")
    parser.add_argument("--notes")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def load_evaluation(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = ["sequence", "frames", "ate_rmse_se3_m", "ate_rmse_sim3_m"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"evaluation JSON missing required key(s): {', '.join(missing)}")
    return data


def parse_verified_loops(path: Path) -> int | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"verified_loops=(\d+)", text)
    if not match:
        return None
    return int(match.group(1))


def optional_artifacts(out_dir: Path | None, features_dir: Path | None) -> list[tuple[str, Path]]:
    artifacts: list[tuple[str, Path]] = []
    if out_dir is not None:
        for kind, name in [
            ("summary", "summary.txt"),
            ("vo_csv", "vo.csv"),
            ("frontend_pair_diagnostics", "frontend_pair_diagnostics.csv"),
            ("frontend_depth_gate_diagnostics", "frontend_depth_gate_diagnostics.csv"),
            ("loop_candidates", "loop_candidates.csv"),
            ("loop_candidate_verifications", "loop_candidate_verifications.csv"),
        ]:
            path = out_dir / name
            if path.exists():
                artifacts.append((kind, path))
    if features_dir is not None and features_dir.exists():
        artifacts.append(("features_dir", features_dir))
    return artifacts


def metric_args(evaluation: dict[str, Any], verified_loops: int | None) -> list[str]:
    args = [
        "--primary-metric",
        "ate_rmse_se3_m",
        "--metric",
        f"ate_rmse_se3_m={evaluation['ate_rmse_se3_m']}:m",
        "--metric",
        f"ate_rmse_sim3_m={evaluation['ate_rmse_sim3_m']}:m",
        "--metric",
        f"frames={evaluation['frames']}:count",
    ]
    if verified_loops is not None:
        args.extend(["--metric", f"verified_loops={verified_loops}:count"])
    return args


def build_capture_cmd(
    args: argparse.Namespace,
    *,
    run_id: str,
    evaluation: dict[str, Any],
    verified_loops: int | None,
) -> list[str]:
    command = args.command or f"scripts/run_kitti_multiseq_benchmark.sh --sequence {args.sequence}"
    cmd = [
        sys.executable,
        "scripts/benchmark_registry.py",
        "capture",
        "--out",
        str(args.registry_dir / f"{run_id}.json"),
        "--run-id",
        run_id,
        "--benchmark-id",
        BENCHMARK_ID,
        "--benchmark-name",
        BENCHMARK_NAME,
        "--script",
        "scripts/run_kitti_multiseq_benchmark.sh",
        "--protocol",
        "full-stack stereo VO with online BA and loop closure; Umeyama SE(3)/Sim(3) ATE RMSE against KITTI odometry poses",
        "--docs",
        "docs/kitti_multiseq_benchmark.md",
        "--dataset-name",
        "KITTI odometry",
        "--dataset-sequence",
        args.sequence,
        "--dataset-version",
        args.dataset_version,
        "--dataset-path",
        str(args.dataset_path or args.poses.parent),
        "--result-kind",
        "visloc_run",
        "--claim-scope",
        args.claim_scope,
        "--status",
        args.status,
        "--command",
        command,
        "--feature",
        "image-io",
        "--profile",
        "release",
        "--config",
        f"sequence={args.sequence}",
        "--config",
        f"frames={evaluation['frames']}",
        "--artifact",
        f"evaluation_json={args.evaluation_json}",
        "--artifact",
        f"trajectory={args.vo_poses}",
        "--artifact",
        f"ground_truth={args.poses}",
        "--artifact",
        f"vo_log={args.vo_log}",
    ]
    if args.failure_reason:
        cmd.extend(["--failure-reason", args.failure_reason])
    for item in args.config:
        cmd.extend(["--config", item])
    for kind, path in optional_artifacts(args.out_dir, args.features_dir):
        cmd.extend(["--artifact", f"{kind}={path}"])
    if args.notes:
        cmd.extend(["--notes", args.notes])
    cmd.extend(metric_args(evaluation, verified_loops))
    return cmd


def run_command(cmd: list[str], *, dry_run: bool) -> int:
    printable = shlex.join(cmd)
    if dry_run:
        print(printable)
        return 0
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        missing = [path for path in [args.evaluation_json, args.vo_log, args.vo_poses, args.poses] if not path.exists()]
        if missing:
            joined = "\n".join(f"  {path}" for path in missing)
            raise SystemExit(f"missing input file(s):\n{joined}")

    evaluation = load_evaluation(args.evaluation_json)
    verified_loops = parse_verified_loops(args.vo_log)
    run_id = args.run_id or f"{BENCHMARK_ID}-seq{args.sequence}-{utc_stamp()}"
    cmd = build_capture_cmd(args, run_id=run_id, evaluation=evaluation, verified_loops=verified_loops)
    return run_command(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
