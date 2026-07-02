#!/usr/bin/env python3
"""Run LightGlue on loop-candidate pairs using cached SuperPoint features.

The normal stereo VO export stores adjacent temporal matches only. This helper
replays LightGlue on non-adjacent loop-candidate pairs so seq02-style failures
can be separated into:

  * retrieval did not propose a true revisit;
  * descriptor matching failed on a true revisit;
  * matching produced enough pairs, but PnP/geometric verification rejected it.

Input features use the `scripts/export_superpoint_lightglue.py` text format:

    frame_000000_left_features.txt  # X Y SCORE D0 D1 ...

Output matches use the visloc-rs match format with query = current/query frame
and train = older matched keyframe:

    QUERY_IDX TRAIN_IDX CONFIDENCE DISTANCE
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CandidateRow:
    older: int
    newer: int
    score: float
    raw: dict[str, str]


@dataclass(frozen=True)
class CachedFeatures:
    keypoints: list[tuple[float, float]]
    scores: list[float]
    descriptors: list[list[float]]


def parse_image_size(raw: str) -> tuple[int, int]:
    parts = raw.lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("image size must be WIDTHxHEIGHT, e.g. 1241x376")
    try:
        width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("image size must be WIDTHxHEIGHT, e.g. 1241x376") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("image dimensions must be positive")
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--image-size", type=parse_image_size, default=(1241, 376))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument(
        "--candidate-filter",
        choices=("all", "attempted", "not-attempted", "verified", "failed"),
        default="all",
    )
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--max-keypoints", type=int, default=None)
    parser.add_argument("--filter-threshold", type=float, default=0.1)
    parser.add_argument("--write-matches", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.skip < 0:
        parser.error("--skip must be non-negative")
    if args.max_keypoints is not None and args.max_keypoints <= 0:
        parser.error("--max-keypoints must be positive")
    if not (0.0 <= args.filter_threshold <= 1.0):
        parser.error("--filter-threshold must be in [0, 1]")
    return args


def first_existing(fields: dict[str, str], options: list[str]) -> str | None:
    for option in options:
        if option in fields:
            return fields[option]
    return None


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"true", "1", "yes"}


def read_candidates(path: Path, candidate_filter: str, min_score: float | None) -> list[CandidateRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV header is required")
        fields = {name.lower(): name for name in reader.fieldnames}
        older_col = first_existing(fields, ["matched_keyframe_id", "matched", "older", "from", "db"])
        newer_col = first_existing(fields, ["query_frame_id", "query", "newer", "to"])
        score_col = first_existing(fields, ["score", "similarity", "retrieval_score"])
        attempted_col = first_existing(fields, ["attempted"])
        verified_col = first_existing(fields, ["verified"])
        failure_col = first_existing(fields, ["failure_reason"])
        missing = [
            name
            for name, value in [
                ("matched_keyframe_id", older_col),
                ("query_frame_id", newer_col),
                ("score", score_col),
            ]
            if value is None
        ]
        if missing:
            raise ValueError(f"{path}: missing required column(s): {', '.join(missing)}")

        out: list[CandidateRow] = []
        for row_num, row in enumerate(reader, start=2):
            attempted = truthy(row.get(attempted_col)) if attempted_col else False
            verified = truthy(row.get(verified_col)) if verified_col else False
            failure = (row.get(failure_col) or "").strip() if failure_col else ""
            if candidate_filter == "attempted" and not attempted:
                continue
            if candidate_filter == "not-attempted" and attempted:
                continue
            if candidate_filter == "verified" and not verified:
                continue
            if candidate_filter == "failed" and (not attempted or verified or not failure):
                continue
            try:
                score = float(row[score_col])
                if min_score is not None and score < min_score:
                    continue
                out.append(
                    CandidateRow(
                        older=int(row[older_col]),
                        newer=int(row[newer_col]),
                        score=score,
                        raw=row,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{row_num}: invalid candidate row: {row}") from exc
    out.sort(key=lambda candidate: (-candidate.score, candidate.older, candidate.newer))
    return out


def feature_path(features_dir: Path, frame_id: int) -> Path:
    return features_dir / f"frame_{frame_id:06}_left_features.txt"


def stereo_matches_path(features_dir: Path, frame_id: int) -> Path:
    return features_dir / f"frame_{frame_id:06}_stereo_matches.txt"


def read_features(path: Path, max_keypoints: int | None = None) -> CachedFeatures:
    keypoints: list[tuple[float, float]] = []
    scores: list[float] = []
    descriptors: list[list[float]] = []
    with path.open(encoding="utf-8") as handle:
        for row_num, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = [float(value) for value in stripped.split()]
            if len(parts) < 4:
                raise ValueError(f"{path}:{row_num}: expected X Y SCORE D0 ...")
            keypoints.append((parts[0], parts[1]))
            scores.append(parts[2])
            descriptors.append(parts[3:])
            if max_keypoints is not None and len(keypoints) >= max_keypoints:
                break
    if not keypoints:
        raise ValueError(f"{path}: no features")
    dim = len(descriptors[0])
    if dim == 0 or any(len(descriptor) != dim for descriptor in descriptors):
        raise ValueError(f"{path}: inconsistent descriptor dimensions")
    return CachedFeatures(keypoints, scores, descriptors)


def read_stereo_left_indices(path: Path) -> set[int]:
    indices: set[int] = set()
    if not path.exists():
        return indices
    with path.open(encoding="utf-8") as handle:
        for row_num, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(f"{path}:{row_num}: expected QUERY_IDX TRAIN_IDX ...")
            indices.add(int(parts[0]))
    return indices


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def as_lightglue_features(
    features: CachedFeatures,
    image_size: tuple[int, int],
    device: str,
) -> dict[str, Any]:
    import torch

    descriptors = torch.tensor(features.descriptors, dtype=torch.float32, device=device)
    descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
    return {
        "keypoints": torch.tensor(features.keypoints, dtype=torch.float32, device=device)[None],
        "keypoint_scores": torch.tensor(features.scores, dtype=torch.float32, device=device)[None],
        "descriptors": descriptors[None],
        "image_size": torch.tensor(image_size, dtype=torch.float32, device=device)[None],
    }


def squeeze_batch(value: Any) -> Any:
    if hasattr(value, "dim") and value.dim() > 0 and value.shape[0] == 1:
        return value[0]
    return value


def match_rows(match_output: dict[str, Any]) -> list[tuple[int, int, float]]:
    matches = squeeze_batch(match_output.get("matches"))
    scores = squeeze_batch(match_output.get("scores", match_output.get("matching_scores")))
    if matches is None or scores is None:
        raise ValueError("LightGlue output missing matches/scores")
    matches = matches.detach().cpu().tolist()
    scores = scores.detach().cpu().tolist()
    return [(int(pair[0]), int(pair[1]), float(score)) for pair, score in zip(matches, scores)]


def write_matches(path: Path, rows: list[tuple[int, int, float]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# QUERY_IDX TRAIN_IDX CONFIDENCE DISTANCE\n")
        for query_index, train_index, confidence in rows:
            handle.write(
                f"{query_index} {train_index} {confidence:.9g} {1.0 - confidence:.9g}\n"
            )


def optional_int(row: dict[str, str], name: str) -> str:
    value = row.get(name, "")
    return value if value != "" else ""


def optional_float(row: dict[str, str], name: str) -> str:
    value = row.get(name, "")
    return value if value != "" else ""


def main() -> int:
    args = parse_args()
    try:
        from lightglue import LightGlue
        from lightglue.utils import rbd
        import torch
    except ImportError as error:
        print("missing optional LightGlue Python stack; install lightglue/torch first", file=sys.stderr)
        print(f"import error: {error}", file=sys.stderr)
        return 2

    candidates = read_candidates(args.candidates, args.candidate_filter, args.min_score)
    if args.skip:
        candidates = candidates[args.skip :]
    if args.limit is not None:
        candidates = candidates[: args.limit]
    if not candidates:
        raise SystemExit("no candidates selected")

    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    matches_dir = args.out_dir / "matches"
    if args.write_matches:
        matches_dir.mkdir(parents=True, exist_ok=True)

    matcher = LightGlue(features="superpoint", filter_threshold=args.filter_threshold).eval().to(device)
    feature_cache: dict[int, CachedFeatures] = {}
    stereo_cache: dict[int, set[int]] = {}

    def frame_features(frame_id: int) -> CachedFeatures:
        if frame_id not in feature_cache:
            feature_cache[frame_id] = read_features(
                feature_path(args.features_dir, frame_id),
                args.max_keypoints,
            )
        return feature_cache[frame_id]

    def stereo_indices(frame_id: int) -> set[int]:
        if frame_id not in stereo_cache:
            stereo_cache[frame_id] = read_stereo_left_indices(stereo_matches_path(args.features_dir, frame_id))
        return stereo_cache[frame_id]

    summary_path = args.out_dir / "loop_lightglue_pair_diagnostics.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "matched_keyframe_id",
                "query_frame_id",
                "retrieval_score",
                "input_attempted",
                "input_verified",
                "input_failure_reason",
                "input_match_count",
                "input_pnp_correspondence_count",
                "input_inlier_count",
                "lightglue_match_count",
                "lightglue_stereo_left_overlap_count",
                "lightglue_mean_confidence",
                "lightglue_max_confidence",
                "matches_path",
            ]
        )
        for index, candidate in enumerate(candidates, start=1):
            newer_features = as_lightglue_features(frame_features(candidate.newer), args.image_size, device)
            older_features = as_lightglue_features(frame_features(candidate.older), args.image_size, device)
            with torch.no_grad():
                output = rbd(matcher({"image0": newer_features, "image1": older_features}))
            rows = match_rows(output)
            stereo_left = stereo_indices(candidate.older)
            stereo_overlap = sum(1 for _, train_index, _ in rows if train_index in stereo_left)
            confidences = [confidence for _, _, confidence in rows if math.isfinite(confidence)]
            mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
            max_conf = max(confidences) if confidences else 0.0
            match_path = ""
            if args.write_matches:
                path = matches_dir / f"loop_{candidate.older:06}_{candidate.newer:06}_matches.txt"
                write_matches(path, rows)
                match_path = str(path)
            writer.writerow(
                [
                    candidate.older,
                    candidate.newer,
                    f"{candidate.score:.8f}",
                    candidate.raw.get("attempted", ""),
                    candidate.raw.get("verified", ""),
                    candidate.raw.get("failure_reason", ""),
                    optional_int(candidate.raw, "match_count"),
                    optional_int(candidate.raw, "pnp_correspondence_count"),
                    optional_int(candidate.raw, "inlier_count"),
                    len(rows),
                    stereo_overlap,
                    f"{mean_conf:.6f}",
                    f"{max_conf:.6f}",
                    match_path,
                ]
            )
            print(
                f"{index}/{len(candidates)} {candidate.older}->{candidate.newer} "
                f"matches={len(rows)} stereo_overlap={stereo_overlap}",
                flush=True,
            )

    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
