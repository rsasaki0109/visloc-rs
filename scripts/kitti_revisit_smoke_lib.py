"""Small reusable helpers for the KITTI revisit smoke runner.

The command-line runner owns orchestration (fetch, cargo, report rendering).
This module owns the pure validation logic so headline regression checks can be
tested without running Rust or downloading KITTI data.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RevisitExpectations:
    min_candidates: int | None = None
    strongest_from: int | None = None
    strongest_to: int | None = None
    min_strongest_inliers: int | None = None
    min_strongest_ratio: float | None = None


README_HEADLINE_EXPECTATIONS = RevisitExpectations(
    min_candidates=41,
    strongest_from=49,
    strongest_to=4501,
    min_strongest_inliers=57,
    min_strongest_ratio=0.6,
)


def read_candidates_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def strongest_candidate(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        raise ValueError("candidates.csv has no accepted candidates")
    return max(rows, key=lambda row: float(row["score"]))


def validate_expectations(
    rows: list[dict[str, str]],
    expectations: RevisitExpectations,
) -> None:
    strongest = strongest_candidate(rows)
    failures: list[str] = []
    if (
        expectations.min_candidates is not None
        and len(rows) < expectations.min_candidates
    ):
        failures.append(
            f"expected at least {expectations.min_candidates} candidates, got {len(rows)}"
        )
    if expectations.strongest_from is not None:
        actual = int(strongest["matched_keyframe_id"])
        if actual != expectations.strongest_from:
            failures.append(
                f"expected strongest_from={expectations.strongest_from}, got {actual}"
            )
    if expectations.strongest_to is not None:
        actual = int(strongest["query_frame_id"])
        if actual != expectations.strongest_to:
            failures.append(
                f"expected strongest_to={expectations.strongest_to}, got {actual}"
            )
    if expectations.min_strongest_inliers is not None:
        actual = int(strongest["inliers"])
        if actual < expectations.min_strongest_inliers:
            failures.append(
                f"expected strongest inliers >= {expectations.min_strongest_inliers}, got {actual}"
            )
    if expectations.min_strongest_ratio is not None:
        actual = float(strongest["inlier_ratio"])
        if actual < expectations.min_strongest_ratio:
            failures.append(
                f"expected strongest ratio >= {expectations.min_strongest_ratio}, got {actual:.6f}"
            )
    if failures:
        joined = "\n  - ".join(failures)
        raise ValueError(f"KITTI revisit expectation check failed:\n  - {joined}")


def row(
    *,
    score: float,
    matched_keyframe_id: int,
    query_frame_id: int,
    inliers: int,
    inlier_ratio: float,
) -> dict[str, str]:
    """Build a minimal candidate row for tests and small call sites."""
    return {
        "score": str(score),
        "matched_keyframe_id": str(matched_keyframe_id),
        "query_frame_id": str(query_frame_id),
        "inliers": str(inliers),
        "inlier_ratio": str(inlier_ratio),
    }
