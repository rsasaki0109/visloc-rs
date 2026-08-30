#!/usr/bin/env python3
"""Compare corresponding 128-D external SIFT descriptor rows.

The Rust fixed-keypoint probe and COLMAP export retain the same keypoint row
order. This small, dependency-free diagnostic reports descriptor cosine/L2,
nonzero-bin overlap, quantized-byte equality, and the best fixed D4-spatial ×
circular-orientation permutation (including a global sign). It is an analysis
tool only; it does not modify either feature directory.

Example:
    python3 scripts/compare_sift_descriptors.py \
        --reference-dir /path/to/colmap_features_export \
        --candidate-dir /tmp/fixed_vlfeat_probe \
        --stems DSC_0305,DSC_0306,DSC_0307
"""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path


DIM = 128
CELLS = 4
BINS = 8


def load_descriptors(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 3 + DIM:
            raise ValueError(f"{path}: expected {DIM} descriptor values")
        rows.append([float(value) for value in fields[3 : 3 + DIM]])
    return rows


def cosine(lhs: list[float], rhs: list[float]) -> float:
    dot = sum(a * b for a, b in zip(lhs, rhs))
    lhs_norm = math.sqrt(sum(a * a for a in lhs))
    rhs_norm = math.sqrt(sum(b * b for b in rhs))
    return dot / (lhs_norm * rhs_norm) if lhs_norm and rhs_norm else 0.0


def quantized(value: float) -> int:
    # COLMAP's descriptors are non-negative, and FeatureDescriptorsToUnsignedByte
    # uses round(512*d), i.e. an away-from-zero half tie for this domain.
    return max(0, min(255, math.floor(value + 0.5)))


def spatial_transforms() -> list[tuple[str, object]]:
    return [
        ("identity", lambda r, c: (r, c)),
        ("rot90", lambda r, c: (c, CELLS - 1 - r)),
        ("rot180", lambda r, c: (CELLS - 1 - r, CELLS - 1 - c)),
        ("rot270", lambda r, c: (CELLS - 1 - c, r)),
        ("flip-x", lambda r, c: (r, CELLS - 1 - c)),
        ("flip-y", lambda r, c: (CELLS - 1 - r, c)),
        ("transpose", lambda r, c: (c, r)),
        ("anti-transpose", lambda r, c: (CELLS - 1 - c, CELLS - 1 - r)),
    ]


def transform(
    row: list[float], spatial: object, reverse: bool, shift: int, sign: int
) -> list[float]:
    out = [0.0] * DIM
    for r in range(CELLS):
        for c in range(CELLS):
            rr, cc = spatial(r, c)  # type: ignore[misc]
            for orientation in range(BINS):
                oo = (shift + (-orientation if reverse else orientation)) % BINS
                out[(rr * CELLS + cc) * BINS + oo] = sign * row[(r * CELLS + c) * BINS + orientation]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--stems", default="", help="comma-separated stems (default: intersection)")
    parser.add_argument(
        "--max-transform-rows",
        type=int,
        default=2048,
        help="rows used for the fixed-permutation search (default: 2048; all rows still get summary statistics)",
    )
    args = parser.parse_args()

    if args.stems:
        stems = [stem.strip() for stem in args.stems.split(",") if stem.strip()]
    else:
        stems = sorted(
            path.name.removesuffix("_features.txt")
            for path in args.reference_dir.glob("*_features.txt")
            if (args.candidate_dir / path.name).is_file()
        )
    if not stems:
        parser.error("no common feature files")

    all_reference: list[list[float]] = []
    all_candidate: list[list[float]] = []
    for stem in stems:
        reference = load_descriptors(args.reference_dir / f"{stem}_features.txt")
        candidate = load_descriptors(args.candidate_dir / f"{stem}_features.txt")
        if len(reference) != len(candidate):
            raise ValueError(f"{stem}: row count {len(reference)} != {len(candidate)}")
        if not reference:
            continue
        cosines = [cosine(a, b) for a, b in zip(reference, candidate)]
        l2 = [math.sqrt(sum((a - b) ** 2 for a, b in zip(lhs, rhs))) for lhs, rhs in zip(reference, candidate)]
        jaccard = []
        exact = 0
        for lhs, rhs in zip(reference, candidate):
            lhs_nonzero = {i for i, value in enumerate(lhs) if value > 0.0}
            rhs_nonzero = {i for i, value in enumerate(rhs) if value > 0.0}
            union = lhs_nonzero | rhs_nonzero
            jaccard.append(len(lhs_nonzero & rhs_nonzero) / len(union) if union else 1.0)
            exact += int(all(quantized(a) == quantized(b) for a, b in zip(lhs, rhs)))
        print(
            f"{stem}: rows={len(reference)} cosine_mean={statistics.fmean(cosines):.6f} "
            f"cosine_median={statistics.median(cosines):.6f} l2_mean={statistics.fmean(l2):.6f} "
            f"nonzero_jaccard={statistics.fmean(jaccard):.6f} quantized_exact={exact / len(reference):.6f}"
        )
        all_reference.extend(reference)
        all_candidate.extend(candidate)

    if not all_reference:
        return 0
    base_cos = [cosine(a, b) for a, b in zip(all_reference, all_candidate)]
    print(
        f"all: rows={len(all_reference)} cosine_mean={statistics.fmean(base_cos):.6f} "
        f"cosine_median={statistics.median(base_cos):.6f}"
    )

    transform_reference = all_reference[: max(args.max_transform_rows, 0)]
    transform_candidate = all_candidate[: len(transform_reference)]
    best = (-float("inf"), "")
    for spatial_name, spatial in spatial_transforms():
        for reverse in (False, True):
            for shift in range(BINS):
                for sign in (1, -1):
                    score = statistics.fmean(
                        cosine(reference, transform(candidate, spatial, reverse, shift, sign))
                        for reference, candidate in zip(transform_reference, transform_candidate)
                    )
                    label = f"{spatial_name},reverse={int(reverse)},shift={shift},sign={sign:+d}"
                    if score > best[0]:
                        best = (score, label)
    print(
        f"best_fixed_transform: rows={len(transform_reference)} "
        f"cosine_mean={best[0]:.6f} ({best[1]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
