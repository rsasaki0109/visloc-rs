#!/usr/bin/env python3
"""Evaluate EuRoC relocalization retrieval recall and capture a registry manifest.

This is a post-run evidence helper for
`examples/euroc_online_slam_vi_image_demo --relocalization-appearance-*` runs.
It expects `relocalization_appearance_candidates.csv` plus the demo's
all-frame `frame_groundtruth.csv`, runs `scripts/eval_loop_retrieval_recall.py`,
then records the recall JSON/Markdown, candidate CSVs, run summary, and common
trajectory diagnostics in `benchmarks/registry/runs/euroc/`.

By default the recall denominator is the set of query frames present in the
candidate CSV. Use `--all-pose-queries` only when you intentionally want every
pose row in `frame_groundtruth.csv` to be an evaluated query.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ID = "euroc-relocalization-appearance-store"
BENCHMARK_NAME = "EuRoC relocalization appearance retrieval recall"


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
    parser.add_argument("--sequence", required=True, help="EuRoC sequence, e.g. MH_03_medium")
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--poses",
        "--frame-groundtruth",
        dest="poses",
        type=Path,
        required=True,
        help="EuRoC frame_groundtruth.csv with frame_idx,gt_px,gt_py,gt_pz",
    )
    parser.add_argument("--run-dir", type=Path, help="demo output directory containing summary/artifacts")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--dataset-version", default="ASL EuRoC MAV dataset")
    parser.add_argument("--input-kind", default="relocalization_appearance_candidates")
    parser.add_argument("--distance-threshold-m", type=float, default=1.0)
    parser.add_argument("--min-temporal-gap", type=int, default=30)
    parser.add_argument("--min-path-length-m", type=float)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--query-ids")
    parser.add_argument(
        "--all-pose-queries",
        action="store_true",
        help="evaluate every pose row instead of only query_frame_id values in the candidate CSVs",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--registry-dir", type=Path, default=Path("benchmarks/registry/runs/euroc"))
    parser.add_argument("--run-id")
    parser.add_argument(
        "--skip-candidate-diagnostics",
        action="store_true",
        help="do not run diagnose_relocalization_candidates.py alongside recall evaluation",
    )
    parser.add_argument("--benchmark-id", default=BENCHMARK_ID)
    parser.add_argument("--benchmark-name", default=BENCHMARK_NAME)
    parser.add_argument(
        "--protocol",
        default="pose-derived true-revisit recall@K over EuRoC relocalization appearance candidates; no recovery PnP score",
    )
    parser.add_argument("--command", help="original demo command; defaults to the recall evaluator command")
    parser.add_argument("--profile", default="release")
    parser.add_argument("--feature", action="append", default=["image-io"])
    parser.add_argument(
        "--result-kind",
        choices=["visloc_run", "external_published", "external_rerun", "exploratory"],
        default="visloc_run",
    )
    parser.add_argument(
        "--claim-scope",
        choices=["headline", "supporting", "exploratory", "negative"],
        default="negative",
    )
    parser.add_argument("--status", choices=["success", "dnf", "failure"], default="success")
    parser.add_argument("--failure-reason")
    parser.add_argument("--config", action="append", default=[], help="extra KEY=VALUE config entries")
    parser.add_argument("--extra-artifact", action="append", default=[], help="extra KIND=PATH artifact entries")
    parser.add_argument("--notes")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-capture-registry", action="store_true")
    parser.add_argument(
        "--dnf-if-recall-at",
        action="append",
        type=parse_gate,
        default=[],
        help="mark manifest DNF when any frontend has recall@K below threshold",
    )
    parser.add_argument(
        "--primary-recall-k",
        type=int,
        help="recall@K metric to mark primary; default is the smallest requested K",
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
    elif not args.all_pose_queries:
        cmd.append("--query-ids-from-candidates")
    return cmd


def build_diagnostic_cmd(
    args: argparse.Namespace,
    out_json: Path,
    out_md: Path,
    out_csv: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/diagnose_relocalization_candidates.py",
        "--candidates",
        *[str(path) for path in args.candidates],
        "--poses",
        str(args.poses),
        "--distance-threshold-m",
        str(args.distance_threshold_m),
        "--min-temporal-gap",
        str(args.min_temporal_gap),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
        "--out-csv",
        str(out_csv),
    ]
    if args.min_path_length_m is not None:
        cmd.extend(["--min-path-length-m", str(args.min_path_length_m)])
    return cmd


def parse_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def parse_number(raw: str | None) -> float | int | None:
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "None":
        return None
    if raw.startswith("Some(") and raw.endswith(")"):
        raw = raw[5:-1].strip()
    try:
        as_int = int(raw)
    except ValueError:
        pass
    else:
        return as_int
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def optional_metric_args(name: str, value: Any, unit: str | None = None) -> list[str]:
    if value is None:
        return []
    suffix = f":{unit}" if unit else ""
    return ["--metric", f"{name}={value}{suffix}"]


def summary_metric_args(summary: dict[str, str]) -> list[str]:
    specs = [
        ("tracking_success_rate", "ratio"),
        ("frames_recorded", "count"),
        ("map_keyframes", "count"),
        ("map_landmarks", "count"),
        ("relocalization_successes", "count"),
        ("relocalization_attempts", "count"),
        ("relocalization_gate_passes", "count"),
        ("relocalization_descriptor_store_landmark_count_mean", "count"),
        ("relocalization_appearance_descriptor_store_tried_frames", "count"),
        ("relocalization_appearance_descriptor_store_used_frames", "count"),
        ("relocalization_appearance_candidate_keyframe_count_mean", "count"),
        ("relocalization_appearance_best_similarity_mean", "ratio"),
        ("relocalization_budget_skips", "count"),
        ("ate_rigid_rmse_m", "m"),
        ("ate_similarity_rmse_m", "m"),
        ("ate_similarity_scale", "ratio"),
    ]
    args: list[str] = []
    for name, unit in specs:
        args.extend(optional_metric_args(name, parse_number(summary.get(name)), unit))
    return args


def retrieval_metric_args(result: dict[str, Any], primary_metric: str | None) -> list[str]:
    frontends = result.get("frontends", [])
    if not frontends:
        return []
    frontend = frontends[0]
    args: list[str] = []
    specs = [
        ("retrieval_candidate_count", frontend.get("candidate_count"), "count"),
        ("retrieval_query_count", frontend.get("query_count"), "count"),
        ("retrieval_queries_with_candidates", frontend.get("queries_with_candidates"), "count"),
        ("retrieval_eligible_query_count", frontend.get("eligible_query_count"), "count"),
        ("retrieval_mrr", frontend.get("mrr"), "ratio"),
        ("retrieval_mean_first_relevant_rank", frontend.get("mean_first_relevant_rank"), "rank"),
        ("retrieval_top1_false_positive_rate", frontend.get("top1_false_positive_rate"), "ratio"),
    ]
    for name, value, unit in specs:
        args.extend(optional_metric_args(name, value, unit))
    for k, value in sorted(frontend.get("recall_at_k", {}).items(), key=lambda item: int(item[0])):
        args.extend(optional_metric_args(f"retrieval_recall_at_{k}", value, "ratio"))
    for k, value in sorted(frontend.get("mean_precision_at_k", {}).items(), key=lambda item: int(item[0])):
        args.extend(optional_metric_args(f"retrieval_mean_precision_at_{k}", value, "ratio"))
    if primary_metric:
        args.extend(["--primary-metric", primary_metric])
    return args


def diagnostic_metric_args(result: dict[str, Any] | None) -> list[str]:
    if not result:
        return []
    frontends = result.get("frontends", [])
    if not frontends:
        return []
    attempt_count = sum(int(frontend.get("attempt_count") or 0) for frontend in frontends)
    recovery_known_count = sum(int(frontend.get("recovery_status_known_count") or 0) for frontend in frontends)
    gate_known_count = sum(int(frontend.get("gate_status_known_count") or 0) for frontend in frontends)
    success_count = sum(int(frontend.get("success_count") or 0) for frontend in frontends)
    gate_pass_count = sum(int(frontend.get("gate_pass_count") or 0) for frontend in frontends)
    top1_relevant_count = sum(int(frontend.get("top1_relevant_count") or 0) for frontend in frontends)
    any_relevant_count = sum(int(frontend.get("any_relevant_count") or 0) for frontend in frontends)
    top1_acceptance_known_count = sum(
        int(frontend.get("top1_relevant_acceptance_known_count") or 0) for frontend in frontends
    )
    top1_rejected_count = sum(
        int(frontend.get("top1_relevant_rejected_count") or 0) for frontend in frontends
    )
    args: list[str] = []
    for name, value, unit in [
        ("candidate_diag_attempt_count", attempt_count, "count"),
        ("candidate_diag_recovery_status_known_count", recovery_known_count, "count"),
        ("candidate_diag_gate_status_known_count", gate_known_count, "count"),
        ("candidate_diag_success_count", success_count, "count"),
        ("candidate_diag_gate_pass_count", gate_pass_count, "count"),
        ("candidate_diag_top1_relevant_count", top1_relevant_count, "count"),
        ("candidate_diag_any_relevant_count", any_relevant_count, "count"),
        ("candidate_diag_top1_relevant_acceptance_known_count", top1_acceptance_known_count, "count"),
        ("candidate_diag_top1_relevant_rejected_count", top1_rejected_count, "count"),
    ]:
        args.extend(optional_metric_args(name, value, unit))
    if recovery_known_count:
        args.extend(
            optional_metric_args("candidate_diag_success_rate", success_count / recovery_known_count, "ratio")
        )
    if gate_known_count:
        args.extend(optional_metric_args("candidate_diag_gate_pass_rate", gate_pass_count / gate_known_count, "ratio"))
    if attempt_count:
        args.extend(
            optional_metric_args("candidate_diag_top1_relevant_rate", top1_relevant_count / attempt_count, "ratio")
        )
    if top1_acceptance_known_count:
        args.extend(
            optional_metric_args(
                "candidate_diag_top1_relevant_rejected_rate",
                top1_rejected_count / top1_acceptance_known_count,
                "ratio",
            )
        )
    return args


def choose_primary_metric(args: argparse.Namespace) -> str:
    k = args.primary_recall_k if args.primary_recall_k is not None else min(args.ks)
    return f"retrieval_recall_at_{k}"


def optional_run_artifacts(run_dir: Path | None) -> list[tuple[str, Path]]:
    if run_dir is None:
        return []
    names = [
        ("summary", "summary.txt"),
        ("trajectory", "slam_trajectory.csv"),
        ("errors", "slam_errors.csv"),
        ("frame_groundtruth", "frame_groundtruth.csv"),
        ("vi_init_log", "vi_init_log.txt"),
        ("motion_vi_init_log", "motion_vi_init_log.txt"),
        ("keyframe_decisions", "keyframe_decisions.csv"),
        ("covisibility_ba_log", "covisibility_ba_log.txt"),
        ("relocalization_attempts", "relocalization_attempts.csv"),
        ("relocalization_appearance_candidates", "relocalization_appearance_candidates.csv"),
    ]
    return [(kind, run_dir / name) for kind, name in names if (run_dir / name).exists()]


def append_artifact(
    cmd: list[str],
    seen: set[Path],
    kind: str,
    path: Path,
) -> None:
    resolved = path.resolve()
    if resolved in seen:
        return
    seen.add(resolved)
    cmd.extend(["--artifact", f"{kind}={path}"])


def status_from_result(
    eval_returncode: int,
    result: dict[str, Any] | None,
    default_status: str,
    default_failure_reason: str | None,
    dnf_gates: list[tuple[int, float]],
) -> tuple[str, str | None]:
    if default_status in {"dnf", "failure"}:
        return default_status, default_failure_reason
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
    return default_status, default_failure_reason


def build_capture_cmd(
    args: argparse.Namespace,
    *,
    run_id: str,
    eval_cmd: list[str],
    result: dict[str, Any] | None,
    summary: dict[str, str],
    out_json: Path,
    out_md: Path,
    status: str,
    failure_reason: str | None,
    diag_result: dict[str, Any] | None = None,
    diag_json: Path | None = None,
    diag_md: Path | None = None,
    diag_csv: Path | None = None,
) -> list[str]:
    command = args.command or shlex.join(eval_cmd)
    query_scope = "all_poses" if args.all_pose_queries else "candidate_queries"
    if args.query_ids:
        query_scope = "explicit_query_ids"
    cmd = [
        sys.executable,
        "scripts/benchmark_registry.py",
        "capture",
        "--out",
        str(args.registry_dir / f"{run_id}.json"),
        "--run-id",
        run_id,
        "--benchmark-id",
        args.benchmark_id,
        "--benchmark-name",
        args.benchmark_name,
        "--script",
        "scripts/eval_loop_retrieval_recall.py",
        "--protocol",
        args.protocol,
        "--docs",
        "docs/motion_based_vi_alignment.md",
        "--dataset-name",
        "EuRoC MAV",
        "--dataset-sequence",
        args.sequence,
        "--dataset-version",
        args.dataset_version,
        "--dataset-path",
        str(args.dataset_path or summary.get("euroc_dir") or args.poses.parent),
        "--result-kind",
        args.result_kind,
        "--claim-scope",
        args.claim_scope,
        "--status",
        status,
        "--command",
        command,
        "--profile",
        args.profile,
        "--config",
        f"sequence={args.sequence}",
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
        "--config",
        f"query_scope={query_scope}",
        "--artifact",
        f"frame_groundtruth={args.poses}",
        "--artifact",
        f"recall_json={out_json}",
        "--artifact",
        f"recall_markdown={out_md}",
    ]
    for feature in args.feature:
        cmd.extend(["--feature", feature])
    for item in args.config:
        cmd.extend(["--config", item])
    if args.query_ids:
        cmd.extend(["--config", f"query_ids={args.query_ids}"])
    if args.all_pose_queries:
        cmd.extend(["--config", "all_pose_queries=true"])
    if failure_reason:
        cmd.extend(["--failure-reason", failure_reason])
    if args.notes:
        cmd.extend(["--notes", args.notes])

    seen: set[Path] = {args.poses.resolve(), out_json.resolve(), out_md.resolve()}
    for kind, path in [
        ("candidate_diagnostics_json", diag_json),
        ("candidate_diagnostics_markdown", diag_md),
        ("candidate_diagnostics_csv", diag_csv),
    ]:
        if path is not None:
            append_artifact(cmd, seen, kind, path)
    for index, path in enumerate(args.candidates):
        append_artifact(cmd, seen, f"candidate_csv_{index}", path)
    for kind, path in optional_run_artifacts(args.run_dir):
        append_artifact(cmd, seen, kind, path)
    for artifact in args.extra_artifact:
        cmd.extend(["--artifact", artifact])

    primary_metric = choose_primary_metric(args)
    cmd.extend(summary_metric_args(summary))
    cmd.extend(retrieval_metric_args(result or {}, primary_metric))
    cmd.extend(diagnostic_metric_args(diag_result))
    return cmd


def main() -> int:
    args = parse_args()
    if args.distance_threshold_m <= 0:
        raise SystemExit("--distance-threshold-m must be > 0")
    if args.min_temporal_gap < 0:
        raise SystemExit("--min-temporal-gap must be >= 0")
    if args.query_ids and args.all_pose_queries:
        raise SystemExit("--query-ids and --all-pose-queries are mutually exclusive")
    args.ks = sorted(set(args.ks))
    if not args.ks or any(k <= 0 for k in args.ks):
        raise SystemExit("--ks values must be positive")
    if args.primary_recall_k is not None and args.primary_recall_k not in args.ks:
        raise SystemExit("--primary-recall-k must be one of --ks")

    stamp = utc_stamp()
    run_id = args.run_id or f"{BENCHMARK_ID}-{args.sequence}-{stamp}"
    out_dir = args.out_dir or Path("target/euroc_relocalization_retrieval_recall") / args.sequence / stamp
    out_json = out_dir / "retrieval_recall.json"
    out_md = out_dir / "retrieval_recall.md"
    diag_json = out_dir / "candidate_diagnostics.json"
    diag_md = out_dir / "candidate_diagnostics.md"
    diag_csv = out_dir / "candidate_diagnostics.csv"
    summary_path = args.run_dir / "summary.txt" if args.run_dir else None
    summary = parse_summary(summary_path) if summary_path else {}

    if not args.dry_run:
        missing = [path for path in [*args.candidates, args.poses] if not path.exists()]
        if args.run_dir and not args.run_dir.exists():
            missing.append(args.run_dir)
        if missing:
            joined = "\n".join(f"  {path}" for path in missing)
            raise SystemExit(f"missing input file(s):\n{joined}")
        out_dir.mkdir(parents=True, exist_ok=True)

    eval_cmd = build_eval_cmd(args, out_json, out_md)
    eval_code = run_command(eval_cmd, dry_run=args.dry_run)
    diag_code = 0
    diag_result = None
    if not args.skip_candidate_diagnostics:
        diag_cmd = build_diagnostic_cmd(args, diag_json, diag_md, diag_csv)
        diag_code = run_command(diag_cmd, dry_run=args.dry_run)
        if not args.dry_run and diag_json.exists():
            diag_result = json.loads(diag_json.read_text(encoding="utf-8"))
    result = None
    if args.dry_run:
        status, failure_reason = args.status, args.failure_reason
    elif out_json.exists():
        result = json.loads(out_json.read_text(encoding="utf-8"))
        status, failure_reason = status_from_result(
            eval_code,
            result,
            args.status,
            args.failure_reason,
            args.dnf_if_recall_at,
        )
    else:
        status, failure_reason = status_from_result(
            eval_code,
            result,
            args.status,
            args.failure_reason,
            args.dnf_if_recall_at,
        )

    if args.no_capture_registry:
        return eval_code or diag_code
    capture_cmd = build_capture_cmd(
        args,
        run_id=run_id,
        eval_cmd=eval_cmd,
        result=result,
        summary=summary,
        out_json=out_json,
        out_md=out_md,
        diag_result=diag_result,
        diag_json=None if args.skip_candidate_diagnostics else diag_json,
        diag_md=None if args.skip_candidate_diagnostics else diag_md,
        diag_csv=None if args.skip_candidate_diagnostics else diag_csv,
        status=status,
        failure_reason=failure_reason,
    )
    capture_code = run_command(capture_cmd, dry_run=args.dry_run)
    return eval_code or diag_code or capture_code


if __name__ == "__main__":
    raise SystemExit(main())
