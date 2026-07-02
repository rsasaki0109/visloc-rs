#!/usr/bin/env python3
"""Generate EuRoC descriptor-retrieval candidates and capture recall evidence.

This is the recovery-free companion to
`capture_euroc_relocalization_retrieval_recall.py`. It starts from a
`frame_appearance_descriptors.csv` file, uses `keyframe_decisions.csv` or an
explicit database-id list to produce a candidate CSV, then evaluates recall
against `frame_groundtruth.csv` and captures the result in the benchmark
registry.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ID = "euroc-descriptor-retrieval-recall"
BENCHMARK_NAME = "EuRoC descriptor retrieval recall"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, help="EuRoC sequence, e.g. MH_03_medium")
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--frame-groundtruth", type=Path, required=True)
    parser.add_argument("--keyframe-decisions", type=Path)
    parser.add_argument("--database-ids", help="comma/space-separated ids or a one-id-per-line file")
    parser.add_argument("--query-ids", help="comma/space-separated ids or a one-id-per-line file")
    parser.add_argument("--run-dir", type=Path, help="demo output directory containing summary/artifacts")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--dataset-version", default="ASL EuRoC MAV dataset")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--exclude-recent-frame-gap", type=int, default=30)
    parser.add_argument("--min-similarity", type=float)
    parser.add_argument("--frontend", default="descriptor_cosine")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--distance-threshold-m", type=float, default=1.0)
    parser.add_argument("--min-temporal-gap", type=int, default=30)
    parser.add_argument("--min-path-length-m", type=float)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument(
        "--all-pose-queries",
        action="store_true",
        help="evaluate every pose row instead of only query_frame_id values in generated candidates",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--registry-dir", type=Path, default=Path("benchmarks/registry/runs/euroc"))
    parser.add_argument("--run-id")
    parser.add_argument("--command", help="original demo command to record in the registry manifest")
    parser.add_argument("--profile", default="release")
    parser.add_argument("--feature", action="append", default=["image-io"])
    parser.add_argument("--result-kind", default="visloc_run")
    parser.add_argument("--claim-scope", default="exploratory")
    parser.add_argument("--status", default="success")
    parser.add_argument("--failure-reason")
    parser.add_argument("--config", action="append", default=[], help="extra KEY=VALUE config entries")
    parser.add_argument("--notes")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-capture-registry", action="store_true")
    parser.add_argument("--primary-recall-k", type=int)
    parser.add_argument("--dnf-if-recall-at", action="append", default=[])
    return parser.parse_args()


def utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def run_command(cmd: list[str], *, dry_run: bool) -> int:
    printable = shlex.join(cmd)
    if dry_run:
        print(printable)
        return 0
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode


def build_export_cmd(args: argparse.Namespace, candidates_csv: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/export_retrieval_candidates_from_descriptors.py",
        "--descriptors",
        str(args.descriptors),
        "--top-k",
        str(args.top_k),
        "--exclude-recent-frame-gap",
        str(args.exclude_recent_frame_gap),
        "--frontend",
        args.frontend,
        "--out",
        str(candidates_csv),
    ]
    if args.keyframe_decisions:
        cmd.extend(["--keyframe-decisions", str(args.keyframe_decisions)])
    if args.database_ids:
        cmd.extend(["--database-ids", args.database_ids])
    if args.query_ids:
        cmd.extend(["--query-ids", args.query_ids])
    if args.min_similarity is not None:
        cmd.extend(["--min-similarity", str(args.min_similarity)])
    if args.no_normalize:
        cmd.append("--no-normalize")
    return cmd


def build_capture_cmd(
    args: argparse.Namespace,
    *,
    candidates_csv: Path,
    run_id: str,
    out_dir: Path,
    export_cmd: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/capture_euroc_relocalization_retrieval_recall.py",
        "--sequence",
        args.sequence,
        "--candidates",
        str(candidates_csv),
        "--frame-groundtruth",
        str(args.frame_groundtruth),
        "--input-kind",
        "descriptor_retrieval_candidates",
        "--distance-threshold-m",
        str(args.distance_threshold_m),
        "--min-temporal-gap",
        str(args.min_temporal_gap),
        "--ks",
        *[str(k) for k in sorted(set(args.ks))],
        "--out-dir",
        str(out_dir),
        "--registry-dir",
        str(args.registry_dir),
        "--run-id",
        run_id,
        "--benchmark-id",
        BENCHMARK_ID,
        "--benchmark-name",
        BENCHMARK_NAME,
        "--protocol",
        "pose-derived true-revisit recall@K over offline descriptor-cosine EuRoC retrieval candidates; no recovery PnP score",
        "--profile",
        args.profile,
        "--result-kind",
        args.result_kind,
        "--claim-scope",
        args.claim_scope,
        "--status",
        args.status,
        "--command",
        args.command or shlex.join(export_cmd),
        "--config",
        f"descriptor_csv={args.descriptors}",
        "--config",
        f"top_k={args.top_k}",
        "--config",
        f"exclude_recent_frame_gap={args.exclude_recent_frame_gap}",
        "--config",
        f"frontend={args.frontend}",
        "--extra-artifact",
        f"frame_appearance_descriptors={args.descriptors}",
    ]
    if args.run_dir:
        cmd.extend(["--run-dir", str(args.run_dir)])
    if args.dataset_path:
        cmd.extend(["--dataset-path", str(args.dataset_path)])
    if args.dataset_version:
        cmd.extend(["--dataset-version", args.dataset_version])
    if args.min_path_length_m is not None:
        cmd.extend(["--min-path-length-m", str(args.min_path_length_m)])
    if args.query_ids:
        cmd.extend(["--query-ids", args.query_ids])
    if args.all_pose_queries:
        cmd.append("--all-pose-queries")
    if args.database_ids:
        cmd.extend(["--config", f"database_ids={args.database_ids}"])
    if args.keyframe_decisions:
        cmd.extend(["--config", f"keyframe_decisions={args.keyframe_decisions}"])
        cmd.extend(["--extra-artifact", f"keyframe_decisions={args.keyframe_decisions}"])
    if args.min_similarity is not None:
        cmd.extend(["--config", f"min_similarity={args.min_similarity}"])
    if args.no_normalize:
        cmd.extend(["--config", "normalize_descriptors=false"])
    for feature in args.feature:
        cmd.extend(["--feature", feature])
    for item in args.config:
        cmd.extend(["--config", item])
    for gate in args.dnf_if_recall_at:
        cmd.extend(["--dnf-if-recall-at", gate])
    if args.primary_recall_k is not None:
        cmd.extend(["--primary-recall-k", str(args.primary_recall_k)])
    if args.failure_reason:
        cmd.extend(["--failure-reason", args.failure_reason])
    if args.notes:
        cmd.extend(["--notes", args.notes])
    if args.no_capture_registry:
        cmd.append("--no-capture-registry")
    return cmd


def main() -> int:
    args = parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be > 0")
    if args.exclude_recent_frame_gap < 0:
        raise SystemExit("--exclude-recent-frame-gap must be >= 0")
    if args.distance_threshold_m <= 0:
        raise SystemExit("--distance-threshold-m must be > 0")
    if args.min_temporal_gap < 0:
        raise SystemExit("--min-temporal-gap must be >= 0")
    if args.keyframe_decisions and args.database_ids:
        raise SystemExit("--keyframe-decisions and --database-ids are mutually exclusive")
    if not args.keyframe_decisions and not args.database_ids:
        raise SystemExit("one of --keyframe-decisions or --database-ids is required")
    args.ks = sorted(set(args.ks))
    if not args.ks or any(k <= 0 for k in args.ks):
        raise SystemExit("--ks values must be positive")

    stamp = utc_stamp()
    run_id = args.run_id or f"{BENCHMARK_ID}-{args.sequence}-{stamp}"
    out_dir = args.out_dir or Path("target/euroc_descriptor_retrieval_recall") / args.sequence / stamp
    candidates_csv = out_dir / "descriptor_retrieval_candidates.csv"

    if not args.dry_run:
        missing = [path for path in [args.descriptors, args.frame_groundtruth] if not path.exists()]
        if args.keyframe_decisions and not args.keyframe_decisions.exists():
            missing.append(args.keyframe_decisions)
        if args.run_dir and not args.run_dir.exists():
            missing.append(args.run_dir)
        if missing:
            joined = "\n".join(f"  {path}" for path in missing)
            raise SystemExit(f"missing input file(s):\n{joined}")
        out_dir.mkdir(parents=True, exist_ok=True)

    export_cmd = build_export_cmd(args, candidates_csv)
    export_code = run_command(export_cmd, dry_run=args.dry_run)
    if export_code != 0:
        return export_code
    capture_cmd = build_capture_cmd(
        args,
        candidates_csv=candidates_csv,
        run_id=run_id,
        out_dir=out_dir,
        export_cmd=export_cmd,
    )
    return run_command(capture_cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
