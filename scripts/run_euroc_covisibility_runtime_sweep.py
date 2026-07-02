#!/usr/bin/env python3
"""Run a EuRoC covisibility-local-BA runtime sweep.

This wrapper varies `--covisibility-local-ba-max-landmarks` while keeping the
rest of the EuRoC online-SLAM command fixed. Each point is captured through
`scripts/run_euroc_covisibility_local_ba_ab.py --only enabled`, then the
registry-backed Markdown summary is regenerated.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQUENCES = ["MH_03_medium"]
DEFAULT_LANDMARK_CAPS = [100, 200, 400]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--euroc-root",
        type=Path,
        default=Path(os.environ.get("EUROC", "old_~2026/simple_visual_slam/datasets/euroc")),
        help="directory containing EuRoC sequence dirs; default env EUROC or old_~2026/.../euroc",
    )
    parser.add_argument(
        "--sequence",
        action="append",
        default=None,
        help="EuRoC sequence name under --euroc-root; may be repeated",
    )
    parser.add_argument(
        "--euroc-dir",
        action="append",
        type=Path,
        default=[],
        help="explicit EuRoC sequence directory; may be repeated",
    )
    parser.add_argument(
        "--landmark-cap",
        action="append",
        type=int,
        default=None,
        help="max-landmarks cap to sweep; may be repeated",
    )
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--profile", choices=["dev", "release"], default="dev")
    parser.add_argument("--out-root", type=Path, default=Path("target/euroc_covisibility_runtime_sweep"))
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=Path("benchmarks/registry/runs/euroc"),
        help="where run manifests are written",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("docs/generated/euroc_covisibility_runtime_sweep.md"),
        help="where the rendered Markdown sweep summary is written",
    )
    parser.add_argument("--min-active-observations", type=int, default=20)
    parser.add_argument("--fallback-min-boundary-observations", default="none")
    parser.add_argument("--min-keyframes", type=int, default=3)
    parser.add_argument("--trigger-every", type=int, default=1)
    parser.add_argument("--max-neighbor-keyframes", type=int, default=10)
    parser.add_argument("--min-shared", type=int, default=15)
    parser.add_argument("--max-boundary-keyframes", type=int, default=10)
    parser.add_argument("--min-boundary-observations", type=int, default=5)
    parser.add_argument("--outlier-threshold-px", default="5.0")
    parser.add_argument(
        "--max-outlier-observation-ratio",
        type=float,
        default=None,
        help="optional covisibility-BA write-back quality gate ratio",
    )
    parser.add_argument(
        "--boundary-support-min-optimized-keyframes",
        type=int,
        default=None,
        help="optional pre-solve boundary support gate threshold",
    )
    parser.add_argument(
        "--boundary-support-min-fixed-keyframes",
        type=int,
        default=0,
        help="minimum fixed boundary keyframes when boundary support gate is enabled",
    )
    parser.add_argument(
        "--base-demo-args",
        default=(
            "--gravity 0,0,-9.81 --feature-extractor hog --cross-check-matcher "
            "--keyframe-min-translation 0.1 --max-pose-jump-meters 0.2 "
            "--stereo-bootstrap-strict"
        ),
        help="extra demo args shared by all sweep runs",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-capture-registry", action="store_true")
    args = parser.parse_args()
    if args.sequence is None and not args.euroc_dir:
        args.sequence = DEFAULT_SEQUENCES.copy()
    elif args.sequence is None:
        args.sequence = []
    if args.landmark_cap is None:
        args.landmark_cap = DEFAULT_LANDMARK_CAPS.copy()
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.max_frames < 0:
        raise SystemExit("--max-frames must be >= 0")
    if not args.landmark_cap:
        raise SystemExit("at least one --landmark-cap is required")
    for cap in args.landmark_cap:
        if cap < 1:
            raise SystemExit("--landmark-cap values must be >= 1")
    if args.min_active_observations < 1:
        raise SystemExit("--min-active-observations must be >= 1")
    if args.min_keyframes < 1 or args.trigger_every < 1 or args.min_shared < 1:
        raise SystemExit("--min-keyframes, --trigger-every, and --min-shared must be >= 1")
    if args.max_outlier_observation_ratio is not None and not (
        0.0 <= args.max_outlier_observation_ratio <= 1.0
    ):
        raise SystemExit("--max-outlier-observation-ratio must be in [0, 1]")
    if args.boundary_support_min_optimized_keyframes is not None:
        if args.boundary_support_min_optimized_keyframes < 1:
            raise SystemExit("--boundary-support-min-optimized-keyframes must be >= 1")
        if args.boundary_support_min_fixed_keyframes < 1:
            raise SystemExit(
                "--boundary-support-min-fixed-keyframes must be >= 1 when boundary support gate is enabled"
            )
    elif args.boundary_support_min_fixed_keyframes != 0:
        raise SystemExit(
            "--boundary-support-min-fixed-keyframes requires --boundary-support-min-optimized-keyframes"
        )


def covisibility_runner_cmd(args: argparse.Namespace, cap: int, *, skip_build: bool) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_euroc_covisibility_local_ba_ab.py",
        "--euroc-root",
        str(args.euroc_root),
        "--max-frames",
        str(args.max_frames),
        "--profile",
        args.profile,
        "--out-root",
        str(args.out_root / f"landmarks{cap}"),
        "--registry-dir",
        str(args.registry_dir),
        "--only",
        "enabled",
        "--min-keyframes",
        str(args.min_keyframes),
        "--trigger-every",
        str(args.trigger_every),
        "--max-neighbor-keyframes",
        str(args.max_neighbor_keyframes),
        "--min-shared",
        str(args.min_shared),
        "--max-boundary-keyframes",
        str(args.max_boundary_keyframes),
        "--min-boundary-observations",
        str(args.min_boundary_observations),
        "--fallback-min-boundary-observations",
        str(args.fallback_min_boundary_observations),
        "--max-landmarks",
        str(cap),
        "--min-active-observations",
        str(args.min_active_observations),
        "--outlier-threshold-px",
        str(args.outlier_threshold_px),
        "--demo-args",
        args.base_demo_args,
    ]
    if args.max_outlier_observation_ratio is not None:
        cmd.extend(
            [
                "--max-outlier-observation-ratio",
                str(args.max_outlier_observation_ratio),
            ]
        )
    if args.boundary_support_min_optimized_keyframes is not None:
        cmd.extend(
            [
                "--boundary-support-min-optimized-keyframes",
                str(args.boundary_support_min_optimized_keyframes),
                "--boundary-support-min-fixed-keyframes",
                str(args.boundary_support_min_fixed_keyframes),
            ]
        )
    for sequence in args.sequence:
        cmd.extend(["--sequence", sequence])
    for euroc_dir in args.euroc_dir:
        cmd.extend(["--euroc-dir", str(euroc_dir)])
    if skip_build:
        cmd.append("--skip-build")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.no_capture_registry:
        cmd.append("--no-capture-registry")
    return cmd


def summarizer_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/summarize_euroc_covisibility_runtime_sweep.py",
        "--registry-dir",
        str(args.registry_dir),
        "--out",
        str(args.summary_out),
        "--max-frames",
        str(args.max_frames),
        "--neighbor-keyframes",
        str(args.max_neighbor_keyframes),
        "--boundary-keyframes",
        str(args.max_boundary_keyframes),
        "--min-active-observations",
        str(args.min_active_observations),
        "--fallback",
        str(args.fallback_min_boundary_observations),
    ]
    if args.max_outlier_observation_ratio is not None:
        cmd.extend(
            [
                "--max-outlier-observation-ratio",
                str(args.max_outlier_observation_ratio),
            ]
        )
    if args.boundary_support_min_optimized_keyframes is not None:
        cmd.extend(
            [
                "--boundary-support-min-optimized-keyframes",
                str(args.boundary_support_min_optimized_keyframes),
                "--boundary-support-min-fixed-keyframes",
                str(args.boundary_support_min_fixed_keyframes),
            ]
        )
    for sequence in args.sequence:
        cmd.extend(["--sequence", sequence])
    for euroc_dir in args.euroc_dir:
        cmd.extend(["--sequence", euroc_dir.name])
    for cap in args.landmark_cap:
        cmd.extend(["--landmark-cap", str(cap)])
    return cmd


def run(cmd: list[str], dry_run: bool) -> int:
    printable = shlex.join(cmd)
    if dry_run:
        print(printable)
        return 0
    return subprocess.run(cmd, cwd=ROOT).returncode


def main() -> int:
    args = parse_args()
    validate_args(args)
    failures = 0
    for index, cap in enumerate(args.landmark_cap):
        skip_build = args.skip_build or index > 0
        code = run(covisibility_runner_cmd(args, cap, skip_build=skip_build), args.dry_run)
        if code != 0:
            failures += 1
    if failures:
        return 1
    return run(summarizer_cmd(args), args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
