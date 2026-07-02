#!/usr/bin/env python3
"""Evaluate loop-retrieval recall against pose-derived true revisits.

This is intentionally a retrieval-stage metric, not an end-to-end SLAM score.
Given a candidate CSV with rows like

    matched_keyframe_id,query_frame_id,score

it ranks candidates per query by descending score and asks whether a
pose-near true revisit appears in the top K. The same evaluator can be used on
raw appearance candidates or on post-geometry `candidates.csv`; the input kind
is recorded in the output so the two are not confused.

For online-SLAM recovery diagnostics, prefer a pose CSV that covers every query
frame (for example the EuRoC demo's `frame_groundtruth.csv`). A success-only
error file such as `slam_errors.csv` can undercount failed recovery queries.

Example:

    python scripts/eval_loop_retrieval_recall.py \
      --candidates target/kitti_revisit/candidates.csv \
      --poses data/kitti/poses/02.txt \
      --distance-threshold-m 10 \
      --min-temporal-gap 50 \
      --ks 1 5 20 \
      --out-json target/kitti_revisit/retrieval_recall.json

Use `--query-ids-from-candidates` when the candidate CSV is already scoped to
frames where retrieval was attempted, such as recovery diagnostics. This keeps
the recall denominator tied to the emitted query set without maintaining a
separate one-id-per-line file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PoseCentre:
    frame_id: int
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Candidate:
    query_frame_id: int
    matched_keyframe_id: int
    score: float
    frontend: str
    source: str
    raw: dict[str, str]


@dataclass(frozen=True)
class QueryEvaluation:
    query_frame_id: int
    eligible: bool
    relevant_count: int
    candidate_count: int
    first_relevant_rank: int | None
    top1_relevant: bool


def read_kitti_pose_centres(path: Path) -> dict[int, PoseCentre]:
    """Read KITTI odometry pose text: one 3x4 row-major matrix per frame."""

    centres: dict[int, PoseCentre] = {}
    with path.open(encoding="utf-8") as handle:
        for frame_id, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            parts = [float(value) for value in stripped.split()]
            if len(parts) != 12:
                raise ValueError(
                    f"{path}:{frame_id + 1}: expected 12 KITTI pose values, got {len(parts)}"
                )
            centres[frame_id] = PoseCentre(frame_id, parts[3], parts[7], parts[11])
    return centres


def read_pose_csv(path: Path) -> dict[int, PoseCentre]:
    """Read a pose CSV with frame plus x/y/z, tx/ty/tz, or EuRoC gt_p* columns."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV header is required")
        fields = {name.lower(): name for name in reader.fieldnames}
        frame_col = first_existing(fields, ["frame_id", "frame", "id", "idx", "frame_idx"])
        x_col = first_existing(fields, ["gt_px", "gt_x", "x", "tx", "t_x", "px", "est_px", "est_x"])
        y_col = first_existing(fields, ["gt_py", "gt_y", "y", "ty", "t_y", "py", "est_py", "est_y"])
        z_col = first_existing(fields, ["gt_pz", "gt_z", "z", "tz", "t_z", "pz", "est_pz", "est_z"])
        missing = [
            name
            for name, value in [
                ("frame_id", frame_col),
                ("x/tx", x_col),
                ("y/ty", y_col),
                ("z/tz", z_col),
            ]
            if value is None
        ]
        if missing:
            raise ValueError(f"{path}: missing required pose column(s): {', '.join(missing)}")
        centres: dict[int, PoseCentre] = {}
        for row_num, row in enumerate(reader, start=2):
            frame_id = int(row[frame_col])
            centres[frame_id] = PoseCentre(
                frame_id,
                float(row[x_col]),
                float(row[y_col]),
                float(row[z_col]),
            )
            if not all(math.isfinite(v) for v in (centres[frame_id].x, centres[frame_id].y, centres[frame_id].z)):
                raise ValueError(f"{path}:{row_num}: non-finite pose centre")
    return centres


def first_existing(fields: dict[str, str], options: list[str]) -> str | None:
    for option in options:
        if option in fields:
            return fields[option]
    return None


def read_pose_centres(path: Path) -> dict[int, PoseCentre]:
    if path.suffix.lower() == ".csv":
        return read_pose_csv(path)
    return read_kitti_pose_centres(path)


def read_candidates(path: Path, *, default_frontend: str = "all") -> list[Candidate]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV header is required")
        fields = {name.lower(): name for name in reader.fieldnames}
        query_col = first_existing(fields, ["query_frame_id", "query", "to", "newer"])
        match_col = first_existing(fields, ["matched_keyframe_id", "matched", "from", "older", "db"])
        score_col = first_existing(fields, ["score", "similarity", "retrieval_score"])
        frontend_col = first_existing(fields, ["frontend", "retriever", "variant"])
        missing = [
            name
            for name, value in [
                ("query_frame_id", query_col),
                ("matched_keyframe_id", match_col),
                ("score/similarity", score_col),
            ]
            if value is None
        ]
        if missing:
            raise ValueError(f"{path}: missing required candidate column(s): {', '.join(missing)}")

        out: list[Candidate] = []
        for row_num, row in enumerate(reader, start=2):
            try:
                frontend = row[frontend_col] if frontend_col else default_frontend
                out.append(
                    Candidate(
                        query_frame_id=int(row[query_col]),
                        matched_keyframe_id=int(row[match_col]),
                        score=float(row[score_col]),
                        frontend=frontend or default_frontend,
                        source=str(path),
                        raw=row,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{row_num}: invalid candidate row: {row}") from exc
        return out


def distance(a: PoseCentre, b: PoseCentre) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def cumulative_path_lengths(poses: dict[int, PoseCentre]) -> dict[int, float]:
    lengths: dict[int, float] = {}
    total = 0.0
    prev: PoseCentre | None = None
    for frame_id in sorted(poses):
        current = poses[frame_id]
        if prev is not None:
            total += distance(prev, current)
        lengths[frame_id] = total
        prev = current
    return lengths


def is_temporally_valid(
    older: int,
    query: int,
    *,
    min_temporal_gap: int,
    path_lengths: dict[int, float],
    min_path_length_m: float | None,
) -> bool:
    if older >= query:
        return False
    if query - older < min_temporal_gap:
        return False
    if min_path_length_m is not None:
        if older not in path_lengths or query not in path_lengths:
            return False
        if path_lengths[query] - path_lengths[older] < min_path_length_m:
            return False
    return True


def relevant_db_frames(
    query: int,
    poses: dict[int, PoseCentre],
    *,
    distance_threshold_m: float,
    min_temporal_gap: int,
    path_lengths: dict[int, float],
    min_path_length_m: float | None,
) -> set[int]:
    qpose = poses.get(query)
    if qpose is None:
        return set()
    relevant: set[int] = set()
    for older, opose in poses.items():
        if not is_temporally_valid(
            older,
            query,
            min_temporal_gap=min_temporal_gap,
            path_lengths=path_lengths,
            min_path_length_m=min_path_length_m,
        ):
            continue
        if distance(opose, qpose) <= distance_threshold_m:
            relevant.add(older)
    return relevant


def ranked_candidates(candidates: list[Candidate]) -> dict[tuple[str, int], list[Candidate]]:
    grouped: dict[tuple[str, int], list[Candidate]] = {}
    for candidate in candidates:
        grouped.setdefault((candidate.frontend, candidate.query_frame_id), []).append(candidate)
    for key, rows in grouped.items():
        rows.sort(key=lambda c: (-c.score, c.matched_keyframe_id, c.query_frame_id))
        deduped: list[Candidate] = []
        seen: set[int] = set()
        for row in rows:
            if row.matched_keyframe_id in seen:
                continue
            deduped.append(row)
            seen.add(row.matched_keyframe_id)
        grouped[key] = deduped
    return grouped


def evaluate_frontend(
    frontend: str,
    candidates: list[Candidate],
    poses: dict[int, PoseCentre],
    *,
    ks: list[int],
    distance_threshold_m: float,
    min_temporal_gap: int,
    min_path_length_m: float | None,
    query_ids: set[int] | None = None,
) -> dict[str, Any]:
    path_lengths = cumulative_path_lengths(poses)
    by_query = ranked_candidates(candidates)
    candidate_query_ids = {query for candidate_frontend, query in by_query if candidate_frontend == frontend}
    queries = sorted(query_ids if query_ids is not None else poses.keys())
    evaluations: list[QueryEvaluation] = []

    for query in queries:
        relevant = relevant_db_frames(
            query,
            poses,
            distance_threshold_m=distance_threshold_m,
            min_temporal_gap=min_temporal_gap,
            path_lengths=path_lengths,
            min_path_length_m=min_path_length_m,
        )
        rows = by_query.get((frontend, query), [])
        first_rank = None
        for rank, candidate in enumerate(rows, start=1):
            if candidate.matched_keyframe_id in relevant:
                first_rank = rank
                break
        evaluations.append(
            QueryEvaluation(
                query_frame_id=query,
                eligible=bool(relevant),
                relevant_count=len(relevant),
                candidate_count=len(rows),
                first_relevant_rank=first_rank,
                top1_relevant=bool(rows and rows[0].matched_keyframe_id in relevant),
            )
        )

    eligible = [item for item in evaluations if item.eligible]
    eligible_count = len(eligible)
    recall = {}
    precision = {}
    for k in ks:
        hits = sum(
            1
            for item in eligible
            if item.first_relevant_rank is not None and item.first_relevant_rank <= k
        )
        recall[str(k)] = hits / eligible_count if eligible_count else None
        precision[str(k)] = mean(
            [
                precision_at_k(
                    by_query.get((frontend, item.query_frame_id), []),
                    relevant_db_frames(
                        item.query_frame_id,
                        poses,
                        distance_threshold_m=distance_threshold_m,
                        min_temporal_gap=min_temporal_gap,
                        path_lengths=path_lengths,
                        min_path_length_m=min_path_length_m,
                    ),
                    k,
                )
                for item in eligible
            ]
        )

    reciprocal_ranks = [
        (1.0 / item.first_relevant_rank) if item.first_relevant_rank is not None else 0.0
        for item in eligible
    ]
    ranks = [
        item.first_relevant_rank
        for item in eligible
        if item.first_relevant_rank is not None
    ]
    top1_false = sum(1 for item in eligible if item.candidate_count > 0 and not item.top1_relevant)
    top1_count = sum(1 for item in eligible if item.candidate_count > 0)
    return {
        "frontend": frontend,
        "candidate_count": len(candidates),
        "query_count": len(queries),
        "eligible_query_count": eligible_count,
        "queries_with_candidates": len(candidate_query_ids),
        "recall_at_k": recall,
        "mean_precision_at_k": precision,
        "mrr": mean(reciprocal_ranks),
        "mean_first_relevant_rank": mean(ranks),
        "top1_false_positive_rate": (top1_false / top1_count) if top1_count else None,
        "queries": [item.__dict__ for item in evaluations],
    }


def precision_at_k(rows: list[Candidate], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top = rows[:k]
    if not top:
        return 0.0
    return sum(1 for row in top if row.matched_keyframe_id in relevant) / k


def mean(values: list[float | int]) -> float | None:
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def parse_query_ids(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    path = Path(raw)
    if path.exists():
        values: set[int] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                values.add(int(stripped))
        return values
    return {int(part) for part in raw.replace(",", " ").split() if part}


def candidate_query_ids(candidates: list[Candidate]) -> set[int]:
    return {candidate.query_frame_id for candidate in candidates}


def resolve_query_scope(
    *,
    raw_query_ids: str | None,
    query_ids_from_candidates: bool,
    candidates: list[Candidate],
    pose_count: int,
) -> tuple[set[int] | None, str, int]:
    explicit = parse_query_ids(raw_query_ids)
    if explicit is not None and query_ids_from_candidates:
        raise ValueError("--query-ids and --query-ids-from-candidates are mutually exclusive")
    if query_ids_from_candidates:
        values = candidate_query_ids(candidates)
        return values, "candidate_queries", len(values)
    if explicit is not None:
        return explicit, "explicit_query_ids", len(explicit)
    return None, "all_poses", pose_count


def parse_gate(raw: str) -> tuple[int, float]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("gate must be K=VALUE, e.g. 5=0.8")
    k_raw, value_raw = raw.split("=", 1)
    try:
        k = int(k_raw)
        value = float(value_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("gate must be K=VALUE, e.g. 5=0.8") from exc
    if k <= 0 or not (0.0 <= value <= 1.0):
        raise argparse.ArgumentTypeError("gate K must be > 0 and VALUE must be in [0, 1]")
    return k, value


def render_text(result: dict[str, Any], ks: list[int]) -> str:
    lines = [
        "# Loop Retrieval Recall",
        (
            f"input_kind={result['input_kind']} poses={result['pose_count']} "
            f"query_scope={result['query_scope']} "
            f"distance_threshold_m={result['distance_threshold_m']} "
            f"min_temporal_gap={result['min_temporal_gap']} "
            f"min_path_length_m={result['min_path_length_m']}"
        ),
        "",
        "| frontend | candidates | eligible queries | MRR | mean rank | top1 FP | "
        + " | ".join(f"recall@{k}" for k in ks)
        + " |",
        "| --- | ---: | ---: | ---: | ---: | ---: | "
        + " | ".join("---:" for _ in ks)
        + " |",
    ]
    for frontend in result["frontends"]:
        values = [
            frontend["frontend"],
            str(frontend["candidate_count"]),
            str(frontend["eligible_query_count"]),
            format_optional(frontend["mrr"]),
            format_optional(frontend["mean_first_relevant_rank"]),
            format_optional(frontend["top1_false_positive_rate"]),
        ]
        values.extend(format_optional(frontend["recall_at_k"][str(k)]) for k in ks)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def format_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    poses = read_pose_centres(args.poses)
    candidates: list[Candidate] = []
    for path in args.candidates:
        candidates.extend(read_candidates(path))
    frontends = sorted({candidate.frontend for candidate in candidates})
    query_ids, query_scope, query_scope_count = resolve_query_scope(
        raw_query_ids=args.query_ids,
        query_ids_from_candidates=bool(getattr(args, "query_ids_from_candidates", False)),
        candidates=candidates,
        pose_count=len(poses),
    )
    return {
        "schema_version": 1,
        "input_kind": args.input_kind,
        "pose_file": str(args.poses),
        "pose_count": len(poses),
        "query_scope": query_scope,
        "query_scope_count": query_scope_count,
        "candidate_files": [str(path) for path in args.candidates],
        "distance_threshold_m": args.distance_threshold_m,
        "min_temporal_gap": args.min_temporal_gap,
        "min_path_length_m": args.min_path_length_m,
        "ks": args.ks,
        "frontends": [
            evaluate_frontend(
                frontend,
                [candidate for candidate in candidates if candidate.frontend == frontend],
                poses,
                ks=args.ks,
                distance_threshold_m=args.distance_threshold_m,
                min_temporal_gap=args.min_temporal_gap,
                min_path_length_m=args.min_path_length_m,
                query_ids=query_ids,
            )
            for frontend in frontends
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--poses",
        type=Path,
        required=True,
        help="KITTI poses.txt or CSV with frame_id,x,y,z or frame_idx,gt_px,gt_py,gt_pz",
    )
    parser.add_argument("--input-kind", default="retrieval_candidates")
    parser.add_argument("--distance-threshold-m", type=float, default=10.0)
    parser.add_argument("--min-temporal-gap", type=int, default=50)
    parser.add_argument("--min-path-length-m", type=float)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--query-ids", help="comma/space-separated query ids or a one-id-per-line file")
    parser.add_argument(
        "--query-ids-from-candidates",
        action="store_true",
        help="evaluate only query_frame_id values present in the candidate CSVs",
    )
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--require-recall-at", action="append", type=parse_gate, default=[])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.distance_threshold_m <= 0:
        raise SystemExit("--distance-threshold-m must be > 0")
    if args.min_temporal_gap < 0:
        raise SystemExit("--min-temporal-gap must be >= 0")
    if args.query_ids and args.query_ids_from_candidates:
        raise SystemExit("--query-ids and --query-ids-from-candidates are mutually exclusive")
    args.ks = sorted(set(args.ks))
    if not args.ks or any(k <= 0 for k in args.ks):
        raise SystemExit("--ks values must be positive")

    result = evaluate(args)
    text = render_text(result, args.ks)
    print(text, end="")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(text, encoding="utf-8")

    failures: list[str] = []
    for k, threshold in args.require_recall_at:
        for frontend in result["frontends"]:
            value = frontend["recall_at_k"].get(str(k))
            if value is None or value < threshold:
                failures.append(
                    f"{frontend['frontend']} recall@{k}={format_optional(value)} < {threshold:.4f}"
                )
    if failures:
        raise SystemExit("recall gate failed:\n  - " + "\n  - ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
