#!/usr/bin/env python3
"""Export loop/relocalization retrieval candidates from per-frame descriptors.

This helper isolates candidate generation from recovery PnP. Given a descriptor
CSV with one global descriptor per frame and either `keyframe_decisions.csv` or
an explicit database-id list, it ranks older database frames for each query by
cosine similarity and writes a candidate CSV consumable by
`scripts/eval_loop_retrieval_recall.py`.

Descriptor CSV formats:

    frame_idx,d0,d1,d2
    10,0.1,0.2,0.3

or:

    frame_idx,descriptor
    10,"0.1 0.2 0.3"
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


METADATA_COLUMNS = {
    "frame",
    "frame_id",
    "frame_idx",
    "id",
    "idx",
    "timestamp",
    "timestamp_ns",
    "time",
}


@dataclass(frozen=True)
class Candidate:
    query_frame_id: int
    matched_keyframe_id: int
    score: float
    rank: int


def first_existing(fields: dict[str, str], options: list[str]) -> str | None:
    for option in options:
        if option in fields:
            return fields[option]
    return None


def vector_norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def normalize(values: list[float]) -> list[float]:
    norm = vector_norm(values)
    if norm == 0.0:
        return values
    return [value / norm for value in values]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def parse_descriptor_cell(raw: str) -> list[float]:
    return [float(part) for part in raw.replace(",", " ").split() if part]


def read_descriptors(path: Path, *, normalize_vectors: bool = True) -> dict[int, list[float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV header is required")
        fields = {name.lower(): name for name in reader.fieldnames}
        frame_col = first_existing(fields, ["frame_idx", "frame_id", "frame", "id", "idx"])
        descriptor_col = first_existing(fields, ["descriptor", "global_descriptor", "embedding"])
        if frame_col is None:
            raise ValueError(f"{path}: missing frame id column")
        descriptor_cols = [
            name
            for name in reader.fieldnames
            if name != descriptor_col and name != frame_col and name.lower() not in METADATA_COLUMNS
        ]
        if descriptor_col is None and not descriptor_cols:
            raise ValueError(f"{path}: missing descriptor columns")

        out: dict[int, list[float]] = {}
        dim: int | None = None
        for row_num, row in enumerate(reader, start=2):
            frame_id = int(row[frame_col])
            if descriptor_col is not None:
                descriptor = parse_descriptor_cell(row[descriptor_col])
            else:
                descriptor = [float(row[name]) for name in descriptor_cols]
            if not descriptor:
                raise ValueError(f"{path}:{row_num}: descriptor is empty")
            if not all(math.isfinite(value) for value in descriptor):
                raise ValueError(f"{path}:{row_num}: descriptor contains non-finite values")
            if dim is None:
                dim = len(descriptor)
            elif len(descriptor) != dim:
                raise ValueError(f"{path}:{row_num}: descriptor dimension {len(descriptor)} != {dim}")
            out[frame_id] = normalize(descriptor) if normalize_vectors else descriptor
    return out


def parse_ids(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    path = Path(raw)
    if path.exists():
        return {
            int(line.strip())
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    return {int(part) for part in raw.replace(",", " ").split() if part}


def read_keyframe_ids(path: Path) -> set[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV header is required")
        fields = {name.lower(): name for name in reader.fieldnames}
        frame_col = first_existing(fields, ["frame_idx", "frame_id", "frame", "id", "idx"])
        selected_col = first_existing(fields, ["selected", "is_keyframe", "keyframe"])
        if frame_col is None:
            raise ValueError(f"{path}: missing frame id column")
        if selected_col is None:
            raise ValueError(f"{path}: missing selected/is_keyframe column")
        out: set[int] = set()
        for row_num, row in enumerate(reader, start=2):
            raw_selected = row[selected_col].strip().lower()
            if raw_selected in {"1", "true", "yes", "y"}:
                out.add(int(row[frame_col]))
            elif raw_selected in {"0", "false", "no", "n", ""}:
                continue
            else:
                raise ValueError(f"{path}:{row_num}: invalid selected value {row[selected_col]!r}")
    return out


def resolve_database_ids(args: argparse.Namespace) -> set[int]:
    explicit = parse_ids(args.database_ids)
    if explicit is not None and args.keyframe_decisions is not None:
        raise ValueError("--database-ids and --keyframe-decisions are mutually exclusive")
    if explicit is not None:
        return explicit
    if args.keyframe_decisions is None:
        raise ValueError("one of --database-ids or --keyframe-decisions is required")
    return read_keyframe_ids(args.keyframe_decisions)


def generate_candidates(
    descriptors: dict[int, list[float]],
    database_ids: Iterable[int],
    query_ids: Iterable[int],
    *,
    top_k: int,
    exclude_recent_frame_gap: int,
    min_similarity: float | None,
) -> list[Candidate]:
    database = sorted(set(database_ids))
    out: list[Candidate] = []
    for query in sorted(set(query_ids)):
        qdesc = descriptors.get(query)
        if qdesc is None:
            continue
        scored: list[tuple[float, int]] = []
        for db in database:
            if db >= query:
                continue
            if query - db < exclude_recent_frame_gap:
                continue
            ddesc = descriptors.get(db)
            if ddesc is None:
                continue
            score = dot(qdesc, ddesc)
            if min_similarity is not None and score < min_similarity:
                continue
            scored.append((score, db))
        scored.sort(key=lambda item: (-item[0], item[1]))
        for rank, (score, db) in enumerate(scored[:top_k], start=1):
            out.append(
                Candidate(
                    query_frame_id=query,
                    matched_keyframe_id=db,
                    score=score,
                    rank=rank,
                )
            )
    return out


def write_candidates(path: Path, candidates: list[Candidate], *, frontend: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frontend", "query_frame_id", "matched_keyframe_id", "score", "rank"])
        for candidate in candidates:
            writer.writerow(
                [
                    frontend,
                    candidate.query_frame_id,
                    candidate.matched_keyframe_id,
                    f"{candidate.score:.9g}",
                    candidate.rank,
                ]
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--keyframe-decisions", type=Path)
    parser.add_argument("--database-ids", help="comma/space-separated ids or a one-id-per-line file")
    parser.add_argument("--query-ids", help="comma/space-separated ids or a one-id-per-line file")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--exclude-recent-frame-gap", type=int, default=30)
    parser.add_argument("--min-similarity", type=float)
    parser.add_argument("--frontend", default="descriptor_cosine")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be > 0")
    if args.exclude_recent_frame_gap < 0:
        raise SystemExit("--exclude-recent-frame-gap must be >= 0")
    if args.min_similarity is not None and not math.isfinite(args.min_similarity):
        raise SystemExit("--min-similarity must be finite")

    descriptors = read_descriptors(args.descriptors, normalize_vectors=not args.no_normalize)
    database_ids = resolve_database_ids(args)
    query_ids = parse_ids(args.query_ids) or set(descriptors)
    candidates = generate_candidates(
        descriptors,
        database_ids,
        query_ids,
        top_k=args.top_k,
        exclude_recent_frame_gap=args.exclude_recent_frame_gap,
        min_similarity=args.min_similarity,
    )
    write_candidates(args.out, candidates, frontend=args.frontend)
    print(
        f"wrote {len(candidates)} candidates for {len(set(query_ids))} queries "
        f"against {len(database_ids)} database frames to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
