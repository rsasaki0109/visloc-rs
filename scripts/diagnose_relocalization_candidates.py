#!/usr/bin/env python3
"""Summarize relocalization candidate quality versus recovery acceptance.

This is a post-run diagnostic for candidate CSVs such as
`relocalization_appearance_candidates.csv`. It answers a narrower question than
`eval_loop_retrieval_recall.py`: when a recovery attempt emitted candidates, was
the top candidate pose-near, and did the downstream recovery gates accept it?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval_loop_retrieval_recall import (
    Candidate,
    cumulative_path_lengths,
    distance,
    is_temporally_valid,
    read_candidates,
    read_pose_centres,
)


@dataclass(frozen=True)
class AttemptDiagnostic:
    frontend: str
    query_frame_id: int
    candidate_count: int
    top1_keyframe_id: int | None
    top1_score: float | None
    top1_rank: int | None
    top1_distance_m: float | None
    top1_temporal_gap: int | None
    top1_relevant: bool
    any_relevant: bool
    first_relevant_rank: int | None
    recovery_attempted: bool | None
    recovery_succeeded: bool | None
    passed_acceptance_gates: bool | None
    used_appearance_store: bool | None
    used_broader_fallback: bool | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--poses",
        type=Path,
        required=True,
        help="KITTI poses.txt or CSV with frame_id,x,y,z or frame_idx,gt_px,gt_py,gt_pz",
    )
    parser.add_argument("--distance-threshold-m", type=float, default=1.0)
    parser.add_argument("--min-temporal-gap", type=int, default=30)
    parser.add_argument("--min-path-length-m", type=float)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--out-csv", type=Path)
    return parser.parse_args()


def parse_bool(raw: str | None) -> bool | None:
    if raw is None or raw == "":
        return None
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    return None


def parse_rank(candidate: Candidate) -> int | None:
    raw = candidate.raw.get("rank")
    if raw is None or raw == "":
        return None
    try:
        rank = int(raw)
    except ValueError:
        return None
    return rank if rank > 0 else None


def candidate_sort_key(candidate: Candidate) -> tuple[int, float]:
    rank = parse_rank(candidate)
    if rank is not None:
        return (rank, -candidate.score)
    return (2**31 - 1, -candidate.score)


def candidate_relevant(
    candidate: Candidate,
    poses: dict[int, Any],
    path_lengths: dict[int, float],
    *,
    distance_threshold_m: float,
    min_temporal_gap: int,
    min_path_length_m: float | None,
) -> bool:
    query = candidate.query_frame_id
    matched = candidate.matched_keyframe_id
    if query not in poses or matched not in poses:
        return False
    if not is_temporally_valid(
        matched,
        query,
        min_temporal_gap=min_temporal_gap,
        path_lengths=path_lengths,
        min_path_length_m=min_path_length_m,
    ):
        return False
    return distance(poses[query], poses[matched]) <= distance_threshold_m


def diagnose_attempt(
    frontend: str,
    query_frame_id: int,
    rows: list[Candidate],
    poses: dict[int, Any],
    path_lengths: dict[int, float],
    *,
    distance_threshold_m: float,
    min_temporal_gap: int,
    min_path_length_m: float | None,
) -> AttemptDiagnostic:
    rows = sorted(rows, key=candidate_sort_key)
    top = rows[0] if rows else None
    first_relevant_rank: int | None = None
    for index, candidate in enumerate(rows, start=1):
        if candidate_relevant(
            candidate,
            poses,
            path_lengths,
            distance_threshold_m=distance_threshold_m,
            min_temporal_gap=min_temporal_gap,
            min_path_length_m=min_path_length_m,
        ):
            first_relevant_rank = parse_rank(candidate) or index
            break

    top_distance = None
    top_gap = None
    top_relevant = False
    if top is not None and top.query_frame_id in poses and top.matched_keyframe_id in poses:
        top_distance = distance(poses[top.query_frame_id], poses[top.matched_keyframe_id])
        top_gap = top.query_frame_id - top.matched_keyframe_id
        top_relevant = candidate_relevant(
            top,
            poses,
            path_lengths,
            distance_threshold_m=distance_threshold_m,
            min_temporal_gap=min_temporal_gap,
            min_path_length_m=min_path_length_m,
        )

    raw = top.raw if top is not None else {}
    return AttemptDiagnostic(
        frontend=frontend,
        query_frame_id=query_frame_id,
        candidate_count=len(rows),
        top1_keyframe_id=top.matched_keyframe_id if top is not None else None,
        top1_score=top.score if top is not None else None,
        top1_rank=parse_rank(top) if top is not None else None,
        top1_distance_m=top_distance,
        top1_temporal_gap=top_gap,
        top1_relevant=top_relevant,
        any_relevant=first_relevant_rank is not None,
        first_relevant_rank=first_relevant_rank,
        recovery_attempted=parse_bool(raw.get("recovery_attempted")),
        recovery_succeeded=parse_bool(raw.get("recovery_succeeded")),
        passed_acceptance_gates=parse_bool(raw.get("passed_acceptance_gates")),
        used_appearance_store=parse_bool(raw.get("used_appearance_store")),
        used_broader_fallback=parse_bool(raw.get("used_broader_fallback")),
    )


def mean(values: list[float | int]) -> float | None:
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def summarize(frontend: str, attempts: list[AttemptDiagnostic]) -> dict[str, Any]:
    top1_relevant = [attempt for attempt in attempts if attempt.top1_relevant]
    any_relevant = [attempt for attempt in attempts if attempt.any_relevant]
    known_recovery = [attempt for attempt in attempts if attempt.recovery_succeeded is not None]
    known_gates = [attempt for attempt in attempts if attempt.passed_acceptance_gates is not None]
    successes = [attempt for attempt in known_recovery if attempt.recovery_succeeded is True]
    gate_passes = [attempt for attempt in known_gates if attempt.passed_acceptance_gates is True]
    top1_relevant_with_acceptance = [
        attempt
        for attempt in top1_relevant
        if attempt.recovery_succeeded is not None or attempt.passed_acceptance_gates is not None
    ]
    top1_relevant_rejected = [
        attempt
        for attempt in top1_relevant_with_acceptance
        if attempt.recovery_succeeded is False or attempt.passed_acceptance_gates is False
    ]
    distances = [
        attempt.top1_distance_m
        for attempt in attempts
        if attempt.top1_distance_m is not None and math.isfinite(attempt.top1_distance_m)
    ]
    return {
        "frontend": frontend,
        "attempt_count": len(attempts),
        "recovery_status_known_count": len(known_recovery),
        "gate_status_known_count": len(known_gates),
        "success_count": len(successes),
        "gate_pass_count": len(gate_passes),
        "top1_relevant_count": len(top1_relevant),
        "any_relevant_count": len(any_relevant),
        "top1_relevant_acceptance_known_count": len(top1_relevant_with_acceptance),
        "top1_relevant_rejected_count": len(top1_relevant_rejected),
        "top1_relevant_rate": (len(top1_relevant) / len(attempts)) if attempts else None,
        "success_rate": (len(successes) / len(known_recovery)) if known_recovery else None,
        "gate_pass_rate": (len(gate_passes) / len(known_gates)) if known_gates else None,
        "top1_relevant_rejected_rate": (
            len(top1_relevant_rejected) / len(top1_relevant_with_acceptance)
        )
        if top1_relevant_with_acceptance
        else None,
        "top1_distance_mean_m": mean(distances),
        "top1_distance_max_m": max(distances) if distances else None,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    poses = read_pose_centres(args.poses)
    path_lengths = cumulative_path_lengths(poses)
    grouped: dict[tuple[str, int], list[Candidate]] = defaultdict(list)
    candidate_files: list[str] = []
    for path in args.candidates:
        candidate_files.append(str(path))
        for candidate in read_candidates(path):
            grouped[(candidate.frontend, candidate.query_frame_id)].append(candidate)

    attempts = [
        diagnose_attempt(
            frontend,
            query_frame_id,
            rows,
            poses,
            path_lengths,
            distance_threshold_m=args.distance_threshold_m,
            min_temporal_gap=args.min_temporal_gap,
            min_path_length_m=args.min_path_length_m,
        )
        for (frontend, query_frame_id), rows in sorted(grouped.items())
    ]
    frontends = sorted({attempt.frontend for attempt in attempts})
    return {
        "schema_version": 1,
        "pose_file": str(args.poses),
        "pose_count": len(poses),
        "candidate_files": candidate_files,
        "distance_threshold_m": args.distance_threshold_m,
        "min_temporal_gap": args.min_temporal_gap,
        "min_path_length_m": args.min_path_length_m,
        "frontends": [
            summarize(frontend, [attempt for attempt in attempts if attempt.frontend == frontend])
            for frontend in frontends
        ],
        "attempts": [attempt.__dict__ for attempt in attempts],
    }


def format_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_text(result: dict[str, Any]) -> str:
    lines = [
        "# Relocalization Candidate Diagnostics",
        (
            f"poses={result['pose_count']} distance_threshold_m={result['distance_threshold_m']} "
            f"min_temporal_gap={result['min_temporal_gap']} "
            f"min_path_length_m={result['min_path_length_m']}"
        ),
        "",
        "| frontend | attempts | acceptance known | success | gate pass | top1 true | any true | top1 true rejected | top1 dist mean m | top1 dist max m |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for frontend in result["frontends"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    frontend["frontend"],
                    str(frontend["attempt_count"]),
                    str(frontend["top1_relevant_acceptance_known_count"]),
                    str(frontend["success_count"]),
                    str(frontend["gate_pass_count"]),
                    str(frontend["top1_relevant_count"]),
                    str(frontend["any_relevant_count"]),
                    str(frontend["top1_relevant_rejected_count"]),
                    format_optional(frontend["top1_distance_mean_m"]),
                    format_optional(frontend["top1_distance_max_m"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_attempt_csv(path: Path, attempts: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frontend",
        "query_frame_id",
        "candidate_count",
        "top1_keyframe_id",
        "top1_score",
        "top1_rank",
        "top1_distance_m",
        "top1_temporal_gap",
        "top1_relevant",
        "any_relevant",
        "first_relevant_rank",
        "recovery_attempted",
        "recovery_succeeded",
        "passed_acceptance_gates",
        "used_appearance_store",
        "used_broader_fallback",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for attempt in attempts:
            writer.writerow({field: attempt.get(field) for field in fields})


def main() -> int:
    args = parse_args()
    if args.distance_threshold_m <= 0:
        raise SystemExit("--distance-threshold-m must be > 0")
    if args.min_temporal_gap < 0:
        raise SystemExit("--min-temporal-gap must be >= 0")

    result = evaluate(args)
    text = render_text(result)
    print(text, end="")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(text, encoding="utf-8")
    if args.out_csv:
        write_attempt_csv(args.out_csv, result["attempts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
