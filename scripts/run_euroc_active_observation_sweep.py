#!/usr/bin/env python3
"""Run the EuRoC covisibility-BA active-observation sweep.

This is a thin reproducibility wrapper around
`scripts/run_euroc_keyframe_policy_ab.py`: for each requested active-observation
floor it runs the fixed-vs-tracked-drop A/B, captures benchmark-registry
manifests, then regenerates the active-observation sweep Markdown summary.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQUENCES = ["MH_01_easy", "MH_03_medium", "MH_05_difficult"]
DEFAULT_ACTIVE_FLOORS = [20, 50]


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
        "--active-floor",
        action="append",
        type=int,
        default=None,
        help="min-active-observations floor to sweep; may be repeated",
    )
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--profile", choices=["dev", "release"], default="release")
    parser.add_argument("--out-root", type=Path, default=Path("target/euroc_active_observation_sweep"))
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=Path("benchmarks/registry/runs/euroc"),
        help="where run manifests are written",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("docs/generated/euroc_active_observation_sweep.md"),
        help="where the rendered Markdown sweep summary is written",
    )
    parser.add_argument("--tracked-landmark-ratio", type=float, default=0.9)
    parser.add_argument("--min-tracked-landmarks", type=int, default=20)
    parser.add_argument("--ba-min-keyframes", type=int, default=3)
    parser.add_argument("--ba-trigger-every", type=int, default=1)
    parser.add_argument("--ba-max-landmarks", default="200")
    parser.add_argument("--ba-outlier-threshold-px", default="5.0")
    parser.add_argument("--fallback-min-boundary-observations", default="none")
    parser.add_argument(
        "--base-demo-args",
        default="--gravity 0,0,-9.81 --keyframe-min-translation 0.1",
        help="extra demo args shared by all sweep runs before covisibility-BA knobs",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-capture-registry", action="store_true")
    args = parser.parse_args()
    if args.sequence is None:
        args.sequence = DEFAULT_SEQUENCES.copy()
    if args.active_floor is None:
        args.active_floor = DEFAULT_ACTIVE_FLOORS.copy()
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.max_frames < 0:
        raise SystemExit("--max-frames must be >= 0")
    if not args.active_floor:
        raise SystemExit("at least one --active-floor is required")
    for floor in args.active_floor:
        if floor < 1:
            raise SystemExit("--active-floor values must be >= 1")
    if args.ba_min_keyframes < 1:
        raise SystemExit("--ba-min-keyframes must be >= 1")
    if args.ba_trigger_every < 1:
        raise SystemExit("--ba-trigger-every must be >= 1")
    if args.min_tracked_landmarks < 1:
        raise SystemExit("--min-tracked-landmarks must be >= 1")


def demo_args_for_floor(args: argparse.Namespace, floor: int) -> str:
    parts = shlex.split(args.base_demo_args) if args.base_demo_args else []
    parts.extend(
        [
            "--covisibility-local-ba",
            "--covisibility-local-ba-min-keyframes",
            str(args.ba_min_keyframes),
            "--covisibility-local-ba-trigger-every",
            str(args.ba_trigger_every),
            "--covisibility-local-ba-max-landmarks",
            str(args.ba_max_landmarks),
            "--covisibility-local-ba-outlier-threshold-px",
            str(args.ba_outlier_threshold_px),
            "--covisibility-local-ba-min-active-observations",
            str(floor),
            "--covisibility-local-ba-fallback-min-boundary-observations",
            str(args.fallback_min_boundary_observations),
        ]
    )
    return shlex.join(parts)


def keyframe_runner_cmd(args: argparse.Namespace, floor: int, *, skip_build: bool) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_euroc_keyframe_policy_ab.py",
        "--euroc-root",
        str(args.euroc_root),
        "--max-frames",
        str(args.max_frames),
        "--profile",
        args.profile,
        "--out-root",
        str(args.out_root / f"active{floor}"),
        "--registry-dir",
        str(args.registry_dir),
        "--tracked-landmark-ratio",
        str(args.tracked_landmark_ratio),
        "--min-tracked-landmarks",
        str(args.min_tracked_landmarks),
        "--demo-args",
        demo_args_for_floor(args, floor),
    ]
    for sequence in args.sequence:
        cmd.extend(["--sequence", sequence])
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
        "scripts/summarize_euroc_active_observation_sweep.py",
        "--registry-dir",
        str(args.registry_dir),
        "--out",
        str(args.summary_out),
        "--max-frames",
        str(args.max_frames),
        "--fallback",
        str(args.fallback_min_boundary_observations),
    ]
    for sequence in args.sequence:
        cmd.extend(["--sequence", sequence])
    for floor in args.active_floor:
        cmd.extend(["--active-floor", str(floor)])
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
    for index, floor in enumerate(args.active_floor):
        skip_build = args.skip_build or index > 0
        code = run(keyframe_runner_cmd(args, floor, skip_build=skip_build), args.dry_run)
        if code != 0:
            failures += 1
    if failures:
        return 1
    return run(summarizer_cmd(args), args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
