"""Report parsing helpers for the KITTI revisit README asset renderer."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

Candidate = dict[str, Any]
SvgLine = tuple[float, float, float, float]


def read_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for token in path.read_text(encoding="utf-8").replace("\n", " ").split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def read_candidates(path: Path) -> list[Candidate]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("score", "inlier_ratio", "mean_sampson_error"):
            row[key] = float(row[key])
        for key in ("matched_keyframe_id", "query_frame_id", "matches", "inliers"):
            row[key] = int(row[key])
    return rows


def strongest_candidate(rows: list[Candidate]) -> Candidate:
    if not rows:
        raise SystemExit("candidates.csv has no accepted candidates")
    return max(rows, key=lambda row: row["score"])


def find_pair_images(report_dir: Path, candidate: Candidate) -> tuple[Path, Path]:
    assets = report_dir / "assets"
    from_id = candidate["matched_keyframe_id"]
    to_id = candidate["query_frame_id"]
    from_matches = sorted(assets.glob(f"*_from_{from_id}.*"))
    to_matches = sorted(assets.glob(f"*_to_{to_id}.*"))
    if not from_matches or not to_matches:
        raise SystemExit(
            f"missing copied pair images under {assets} for frames {from_id} -> {to_id}"
        )
    return from_matches[0], to_matches[0]


def find_overlay_svg(report_dir: Path, candidate: Candidate) -> Path | None:
    assets = report_dir / "assets"
    from_id = candidate["matched_keyframe_id"]
    to_id = candidate["query_frame_id"]
    matches = sorted(assets.glob(f"*_matches_{from_id}_{to_id}.svg"))
    return matches[0] if matches else None


def parse_svg_lines(path: Path | None) -> list[SvgLine]:
    if path is None or not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r'<line x1="([0-9.]+)" y1="([0-9.]+)" x2="([0-9.]+)" y2="([0-9.]+)"'
    )
    return [tuple(float(value) for value in match.groups()) for match in pattern.finditer(text)]


def first_summary_int(summary: dict[str, str], key: str, fallback: int) -> int:
    value = summary.get(key)
    if value is None:
        return fallback
    match = re.match(r"^-?\d+", value)
    return int(match.group(0)) if match else fallback
