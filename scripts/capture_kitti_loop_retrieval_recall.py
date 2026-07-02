#!/usr/bin/env python3
"""Evaluate KITTI loop retrieval recall and capture a registry manifest.

This is a post-run evidence helper. It expects `loop_candidates.csv` from
`examples/stereo_vo_external_deep_files --loop-closure` (or an equivalent raw
retrieval CSV) plus KITTI odometry poses, runs
`scripts/eval_loop_retrieval_recall.py`, then registers the recall JSON/Markdown,
input CSVs, and sibling loop-candidate verifier diagnostics (when present) in
`benchmarks/registry/runs/kitti/`.

Example:

    python scripts/capture_kitti_loop_retrieval_recall.py \
      --sequence 02 \
      --candidates target/kitti_seq02_full/loop_candidates.csv \
      --poses ~/datasets/kitti_seq02_full/poses_02.txt \
      --distance-threshold-m 10 \
      --min-temporal-gap 50 \
      --min-path-length-m 5 \
      --ks 1 5 20 \
      --dnf-if-recall-at 20=0.01
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
BENCHMARK_ID = "kitti-loop-retrieval-recall"
BENCHMARK_NAME = "KITTI loop retrieval recall"


def parse_gate(raw: str) -> tuple[int, float]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("gate must be K=VALUE, e.g. 20=0.01")
    k_raw, value_raw = raw.split("=", 1)
    try:
        k = int(k_raw)
        value = float(value_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("gate must be K=VALUE, e.g. 20=0.01") from exc
    if k <= 0 or not (0.0 <= value <= 1.0):
        raise argparse.ArgumentTypeError("gate K must be > 0 and VALUE must be in [0, 1]")
    return k, value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, help="KITTI odometry sequence, e.g. 02")
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--dataset-version", default="KITTI odometry grayscale")
    parser.add_argument("--input-kind", default="raw_appearance_candidates")
    parser.add_argument("--distance-threshold-m", type=float, default=10.0)
    parser.add_argument("--min-temporal-gap", type=int, default=50)
    parser.add_argument("--min-path-length-m", type=float)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--query-ids")
    parser.add_argument(
        "--query-ids-from-candidates",
        action="store_true",
        help="pass through to eval_loop_retrieval_recall.py and scope queries to candidate CSV rows",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--registry-dir", type=Path, default=Path("benchmarks/registry/runs/kitti"))
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-capture-registry", action="store_true")
    parser.add_argument(
        "--dnf-if-recall-at",
        action="append",
        type=parse_gate,
        default=[],
        help="mark manifest DNF when any frontend has recall@K below threshold",
    )
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


def build_eval_cmd(args: argparse.Namespace, out_json: Path, out_md: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/eval_loop_retrieval_recall.py",
        "--candidates",
        *[str(path) for path in args.candidates],
        "--poses",
        str(args.poses),
        "--input-kind",
        args.input_kind,
        "--distance-threshold-m",
        str(args.distance_threshold_m),
        "--min-temporal-gap",
        str(args.min_temporal_gap),
        "--ks",
        *[str(k) for k in sorted(set(args.ks))],
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]
    if args.min_path_length_m is not None:
        cmd.extend(["--min-path-length-m", str(args.min_path_length_m)])
    if args.query_ids:
        cmd.extend(["--query-ids", args.query_ids])
    if getattr(args, "query_ids_from_candidates", False):
        cmd.append("--query-ids-from-candidates")
    return cmd


def sanitize_metric_prefix(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value or "frontend"


def optional_metric_args(name: str, value: Any, unit: str | None = None) -> list[str]:
    if value is None:
        return []
    suffix = f":{unit}" if unit else ""
    return ["--metric", f"{name}={value}{suffix}"]


def metric_args(result: dict[str, Any], primary_metric: str | None) -> list[str]:
    args: list[str] = []
    for frontend in result.get("frontends", []):
        prefix = sanitize_metric_prefix(str(frontend.get("frontend", "frontend")))
        args.extend(optional_metric_args(f"{prefix}_candidate_count", frontend.get("candidate_count"), "count"))
        args.extend(optional_metric_args(f"{prefix}_eligible_query_count", frontend.get("eligible_query_count"), "count"))
        args.extend(optional_metric_args(f"{prefix}_mrr", frontend.get("mrr"), "ratio"))
        args.extend(
            optional_metric_args(
                f"{prefix}_mean_first_relevant_rank",
                frontend.get("mean_first_relevant_rank"),
                "rank",
            )
        )
        args.extend(
            optional_metric_args(
                f"{prefix}_top1_false_positive_rate",
                frontend.get("top1_false_positive_rate"),
                "ratio",
            )
        )
        for k, value in sorted(frontend.get("recall_at_k", {}).items(), key=lambda item: int(item[0])):
            args.extend(optional_metric_args(f"{prefix}_recall_at_{k}", value, "ratio"))
        for k, value in sorted(frontend.get("mean_precision_at_k", {}).items(), key=lambda item: int(item[0])):
            args.extend(optional_metric_args(f"{prefix}_mean_precision_at_{k}", value, "ratio"))
    if primary_metric:
        args.extend(["--primary-metric", primary_metric])
    return args


def choose_primary_metric(result: dict[str, Any], ks: list[int]) -> str | None:
    frontends = result.get("frontends", [])
    if not frontends:
        return None
    prefix = sanitize_metric_prefix(str(frontends[0].get("frontend", "frontend")))
    return f"{prefix}_recall_at_{max(ks)}"


def discover_verification_diagnostics(candidates: list[Path]) -> list[Path]:
    diagnostics: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        path = candidate.with_name("loop_candidate_verifications.csv")
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        diagnostics.append(path)
    return diagnostics


def status_from_result(
    eval_returncode: int,
    result: dict[str, Any] | None,
    dnf_gates: list[tuple[int, float]],
) -> tuple[str, str | None]:
    if eval_returncode != 0:
        return "failure", f"eval_loop_retrieval_recall.py exited with status {eval_returncode}"
    if result is None:
        return "failure", "recall JSON was not produced"
    frontends = result.get("frontends", [])
    if not frontends:
        return "dnf", "no frontend rows in recall result"
    if all(int(frontend.get("eligible_query_count") or 0) == 0 for frontend in frontends):
        return "dnf", "no eligible true-revisit queries under configured gates"
    failures: list[str] = []
    for k, threshold in dnf_gates:
        for frontend in frontends:
            value = frontend.get("recall_at_k", {}).get(str(k))
            if value is None or value < threshold:
                failures.append(
                    f"{frontend.get('frontend', 'frontend')} recall@{k}={value} below {threshold}"
                )
    if failures:
        return "dnf", "; ".join(failures)
    return "success", None


def build_capture_cmd(
    args: argparse.Namespace,
    *,
    run_id: str,
    eval_cmd: list[str],
    result: dict[str, Any] | None,
    out_json: Path,
    out_md: Path,
    status: str,
    failure_reason: str | None,
) -> list[str]:
    primary_metric = choose_primary_metric(result or {}, sorted(set(args.ks)))
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
        "scripts/eval_loop_retrieval_recall.py",
        "--protocol",
        "pose-derived true-revisit recall@K over gated loop retrieval candidates; no PnP/PGO score",
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
        "exploratory",
        "--claim-scope",
        "exploratory",
        "--status",
        status,
        "--command",
        shlex.join(eval_cmd),
        "--config",
        f"input_kind={args.input_kind}",
        "--config",
        f"distance_threshold_m={args.distance_threshold_m}",
        "--config",
        f"min_temporal_gap={args.min_temporal_gap}",
        "--config",
        f"min_path_length_m={args.min_path_length_m}",
        "--config",
        f"ks={json.dumps(sorted(set(args.ks)))}",
        "--artifact",
        f"poses={args.poses}",
        "--artifact",
        f"recall_json={out_json}",
        "--artifact",
        f"recall_markdown={out_md}",
    ]
    for index, path in enumerate(args.candidates):
        cmd.extend(["--artifact", f"candidate_csv_{index}={path}"])
    for index, path in enumerate(discover_verification_diagnostics(args.candidates)):
        cmd.extend(["--artifact", f"verification_diagnostics_csv_{index}={path}"])
    if args.query_ids:
        cmd.extend(["--config", f"query_ids={args.query_ids}"])
    if getattr(args, "query_ids_from_candidates", False):
        cmd.extend(["--config", "query_ids_from_candidates=true"])
    if failure_reason:
        cmd.extend(["--failure-reason", failure_reason])
    cmd.extend(metric_args(result or {}, primary_metric))
    return cmd


def main() -> int:
    args = parse_args()
    if args.distance_threshold_m <= 0:
        raise SystemExit("--distance-threshold-m must be > 0")
    if args.min_temporal_gap < 0:
        raise SystemExit("--min-temporal-gap must be >= 0")
    if args.query_ids and args.query_ids_from_candidates:
        raise SystemExit("--query-ids and --query-ids-from-candidates are mutually exclusive")
    args.ks = sorted(set(args.ks))
    if not args.ks or any(k <= 0 for k in args.ks):
        raise SystemExit("--ks values must be positive")

    stamp = utc_stamp()
    run_id = args.run_id or f"{BENCHMARK_ID}-seq{args.sequence}-{stamp}"
    out_dir = args.out_dir or Path("target/kitti_loop_retrieval_recall") / f"seq{args.sequence}" / stamp
    out_json = out_dir / "retrieval_recall.json"
    out_md = out_dir / "retrieval_recall.md"

    if not args.dry_run:
        missing = [path for path in [*args.candidates, args.poses] if not path.exists()]
        if missing:
            joined = "\n".join(f"  {path}" for path in missing)
            raise SystemExit(f"missing input file(s):\n{joined}")
        out_dir.mkdir(parents=True, exist_ok=True)

    eval_cmd = build_eval_cmd(args, out_json, out_md)
    eval_code = run_command(eval_cmd, dry_run=args.dry_run)
    result = None
    if args.dry_run:
        status, failure_reason = "success", None
    elif out_json.exists():
        result = json.loads(out_json.read_text(encoding="utf-8"))
        status, failure_reason = status_from_result(eval_code, result, args.dnf_if_recall_at)
    else:
        status, failure_reason = status_from_result(eval_code, result, args.dnf_if_recall_at)

    if args.no_capture_registry:
        return eval_code
    capture_cmd = build_capture_cmd(
        args,
        run_id=run_id,
        eval_cmd=eval_cmd,
        result=result,
        out_json=out_json,
        out_md=out_md,
        status=status,
        failure_reason=failure_reason,
    )
    capture_code = run_command(capture_cmd, dry_run=args.dry_run)
    return eval_code or capture_code


if __name__ == "__main__":
    raise SystemExit(main())
