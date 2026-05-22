"""CLI parsing for the KITTI revisit smoke runner."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from kitti_revisit_smoke_lib import README_HEADLINE_EXPECTATIONS


def env_default(name: str, default: str) -> str:
    return os.environ.get(name, default)


def optional_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else None


def optional_float_env(name: str) -> float | None:
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else None


def parse_args(description: str | None = None) -> argparse.Namespace:
    home = Path.home()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--start-dir",
        type=Path,
        default=Path(
            env_default(
                "KITTI_REVISIT_START_DIR",
                str(home / "datasets" / "kitti_seq00_start_50"),
            )
        ),
        help="Start segment directory",
    )
    parser.add_argument(
        "--revisit-dir",
        type=Path,
        default=Path(
            env_default(
                "KITTI_REVISIT_DIR",
                str(home / "datasets" / "kitti_seq00_revisit_4500"),
            )
        ),
        help="Revisit segment directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(env_default("KITTI_REVISIT_OUT_DIR", "target/kitti_revisit_deep_smoke")),
        help="Output directory",
    )
    parser.add_argument(
        "--start-frames",
        type=int,
        default=int(env_default("KITTI_REVISIT_START_FRAMES", "50")),
    )
    parser.add_argument(
        "--revisit-start-frame",
        type=int,
        default=int(env_default("KITTI_REVISIT_START_FRAME", "4500")),
    )
    parser.add_argument(
        "--revisit-frames",
        type=int,
        default=int(env_default("KITTI_REVISIT_FRAMES", "30")),
    )
    parser.add_argument("--workers", type=int, default=int(env_default("KITTI_REVISIT_WORKERS", "8")))
    parser.add_argument("--frontend", default=env_default("KITTI_REVISIT_FRONTEND", "deep"))
    parser.add_argument(
        "--max-features",
        type=int,
        default=int(env_default("KITTI_REVISIT_MAX_FEATURES", "200")),
    )
    parser.add_argument("--min-matches", type=int, default=int(env_default("KITTI_REVISIT_MIN_MATCHES", "30")))
    parser.add_argument("--min-inliers", type=int, default=int(env_default("KITTI_REVISIT_MIN_INLIERS", "12")))
    parser.add_argument(
        "--min-inlier-ratio",
        type=float,
        default=float(env_default("KITTI_REVISIT_MIN_INLIER_RATIO", "0.4")),
    )
    parser.add_argument(
        "--max-mean-sampson-error",
        type=float,
        default=float(env_default("KITTI_REVISIT_MAX_MEAN_SAMPSON_ERROR", "0.005")),
    )
    parser.add_argument("--skip-fetch", action="store_true", help="Reuse already-fetched subsets")
    parser.add_argument(
        "--readme-headline-gate",
        action="store_true",
        help="Apply the expected README headline values for KITTI 00 quick run",
    )
    parser.add_argument(
        "--cargo-profile",
        choices=("release", "dev"),
        default=env_default("KITTI_REVISIT_CARGO_PROFILE", "release"),
        help="Cargo profile to run",
    )
    readme_asset_default = os.environ.get("KITTI_REVISIT_README_ASSET_OUT")
    parser.add_argument(
        "--readme-asset-out",
        type=Path,
        default=Path(readme_asset_default) if readme_asset_default else None,
        help="Optional JPEG path to render from the generated report",
    )
    parser.add_argument(
        "--expect-min-candidates",
        type=int,
        default=optional_int_env("KITTI_REVISIT_EXPECT_MIN_CANDIDATES"),
        help="Fail if candidates.csv has fewer accepted candidates",
    )
    parser.add_argument(
        "--expect-strongest-from",
        type=int,
        default=optional_int_env("KITTI_REVISIT_EXPECT_STRONGEST_FROM"),
        help="Fail if the strongest candidate starts from a different KITTI frame",
    )
    parser.add_argument(
        "--expect-strongest-to",
        type=int,
        default=optional_int_env("KITTI_REVISIT_EXPECT_STRONGEST_TO"),
        help="Fail if the strongest candidate targets a different KITTI frame",
    )
    parser.add_argument(
        "--expect-min-strongest-inliers",
        type=int,
        default=optional_int_env("KITTI_REVISIT_EXPECT_MIN_STRONGEST_INLIERS"),
        help="Fail if the strongest candidate has fewer verifier inliers",
    )
    parser.add_argument(
        "--expect-min-strongest-ratio",
        type=float,
        default=optional_float_env("KITTI_REVISIT_EXPECT_MIN_STRONGEST_RATIO"),
        help="Fail if the strongest candidate has a lower verifier inlier ratio",
    )
    args = parser.parse_args()
    if args.readme_headline_gate:
        apply_readme_headline_gate(args)
    return args


def apply_readme_headline_gate(args: argparse.Namespace) -> None:
    defaults = README_HEADLINE_EXPECTATIONS
    if args.expect_min_candidates is None:
        args.expect_min_candidates = defaults.min_candidates
    if args.expect_strongest_from is None:
        args.expect_strongest_from = defaults.strongest_from
    if args.expect_strongest_to is None:
        args.expect_strongest_to = defaults.strongest_to
    if args.expect_min_strongest_inliers is None:
        args.expect_min_strongest_inliers = defaults.min_strongest_inliers
    if args.expect_min_strongest_ratio is None:
        args.expect_min_strongest_ratio = defaults.min_strongest_ratio
