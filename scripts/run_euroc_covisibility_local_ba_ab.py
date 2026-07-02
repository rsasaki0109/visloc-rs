#!/usr/bin/env python3
"""Run EuRoC online-SLAM covisibility-local-BA A/B and register artifacts.

The runner executes `examples/euroc_online_slam_vi_image_demo` twice per
sequence:

* `disabled`: baseline, `OnlineSlamConfig::covisibility_local_ba = None`
* `enabled`: same command plus `--covisibility-local-ba` and its window knobs

Each run gets an output directory under `target/euroc_covisibility_local_ba/`
and, by default, a benchmark registry manifest under
`benchmarks/registry/runs/euroc/`.

Example:

    python scripts/run_euroc_covisibility_local_ba_ab.py \
      --euroc-root /datasets/euroc \
      --sequence MH_01_easy --sequence V1_01_easy --sequence V2_01_easy \
      --max-frames 1500 \
      --demo-args "--gravity 0,0,-9.81 --feature-extractor hog --cross-check-matcher \
        --keyframe-min-translation 0.1 --max-pose-jump-meters 0.2 \
        --motion-model adaptive-imu-pose --pnp-pose-prior-warm-start \
        --vi-init-gyro-std-limit 0.5 --vi-init-accel-std-limit 5.0 \
        --vi-init-try-initialize-on-every-frame --vi-init-min-stationary-window-seconds 1.5 \
        --local-vi-ba --run-local-vi-ba-at-vi-init-promotion \
        --keep-pre-promotion-imu-factors --stereo-bootstrap-strict"
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = "euroc_online_slam_vi_image_demo"
BENCHMARK_ID = "euroc-covisibility-local-ba"
BENCHMARK_NAME = "EuRoC online SLAM covisibility local BA A/B"


def cargo_executable() -> str:
    env_cargo = os.environ.get("CARGO")
    if env_cargo:
        return env_cargo
    cargo = shutil.which("cargo")
    if cargo:
        return cargo
    if os.name == "nt":
        candidate = Path.home() / ".cargo" / "bin" / "cargo.exe"
        if candidate.exists():
            return str(candidate)
    return "cargo"


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
        default=[],
        help="EuRoC sequence name under --euroc-root; may be repeated",
    )
    parser.add_argument(
        "--euroc-dir",
        action="append",
        type=Path,
        default=[],
        help="explicit EuRoC sequence directory; may be repeated",
    )
    parser.add_argument("--out-root", type=Path, default=Path("target/euroc_covisibility_local_ba"))
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=Path("benchmarks/registry/runs/euroc"),
        help="where run manifests are written",
    )
    parser.add_argument("--max-frames", type=int, default=1500)
    parser.add_argument("--profile", choices=["dev", "release"], default="release")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing")
    parser.add_argument("--no-capture-registry", action="store_true")
    parser.add_argument(
        "--only",
        choices=["disabled", "enabled"],
        default=None,
        help="run only one side of the A/B",
    )
    parser.add_argument(
        "--demo-args",
        default="",
        help="extra flags passed to both demo runs, parsed with shlex",
    )
    parser.add_argument("--min-keyframes", type=int, default=3)
    parser.add_argument("--trigger-every", type=int, default=1)
    parser.add_argument("--max-neighbor-keyframes", type=int, default=10)
    parser.add_argument("--min-shared", type=int, default=15)
    parser.add_argument("--max-boundary-keyframes", type=int, default=10)
    parser.add_argument("--min-boundary-observations", type=int, default=5)
    parser.add_argument(
        "--fallback-min-boundary-observations",
        default=None,
        help="optional lower boundary-observation floor used only when strict selection has no local landmarks; pass 'none' to disable",
    )
    parser.add_argument("--max-landmarks", type=int, default=None)
    parser.add_argument("--min-active-observations", type=int, default=1)
    parser.add_argument("--outlier-threshold-px", default="5.0")
    parser.add_argument("--remove-outliers", action="store_true")
    parser.add_argument(
        "--max-outlier-observation-ratio",
        type=float,
        default=None,
        help="optional write-back quality gate; reject BA when post-BA outlier observations exceed this ratio",
    )
    parser.add_argument(
        "--boundary-support-min-optimized-keyframes",
        type=int,
        default=None,
        help="optional pre-solve gate; large optimized windows require fixed boundary support",
    )
    parser.add_argument(
        "--boundary-support-min-fixed-keyframes",
        type=int,
        default=0,
        help="minimum fixed boundary keyframes required when the boundary support gate is enabled",
    )
    parser.add_argument(
        "--dnf-if-tracking-success-below",
        type=float,
        default=None,
        help="mark command-success runs as DNF when tracking_success_rate is below this value",
    )
    return parser.parse_args()


def utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def run_command(cmd: list[str], log_path: Path | None, dry_run: bool) -> int:
    printable = shlex.join(cmd)
    if dry_run:
        print(printable)
        if log_path:
            print(f"# log: {log_path}")
        return 0
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        return proc.returncode
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode


def parse_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def parse_number(raw: str | None) -> float | int | None:
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "None":
        return None
    if raw.startswith("Some(") and raw.endswith(")"):
        raw = raw[5:-1].strip()
    try:
        as_int = int(raw)
    except ValueError:
        pass
    else:
        return as_int
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def metric_args(summary: dict[str, str]) -> list[str]:
    specs = [
        ("ate_rigid_rmse_m", "m"),
        ("ate_similarity_rmse_m", "m"),
        ("tracking_success_rate", "ratio"),
        ("frames_recorded", "count"),
        ("map_keyframes", "count"),
        ("map_landmarks", "count"),
        ("covisibility_local_ba_triggers", "count"),
        ("covisibility_local_ba_successes", "count"),
        ("covisibility_local_ba_failures", "count"),
        ("covisibility_local_ba_active_observation_gate_failures", "count"),
        ("covisibility_local_ba_boundary_fallback_active_gate_failures", "count"),
        ("covisibility_local_ba_quality_gate_failures", "count"),
        ("covisibility_local_ba_boundary_support_failures", "count"),
        ("covisibility_local_ba_no_local_landmarks_failures", "count"),
        ("covisibility_local_ba_no_observations_failures", "count"),
        ("covisibility_local_ba_solver_failures", "count"),
        ("covisibility_local_ba_other_failures", "count"),
        ("covisibility_local_ba_boundary_fallback_successes", "count"),
        ("covisibility_local_ba_mean_reprojection_before_px", "px"),
        ("covisibility_local_ba_mean_reprojection_after_px", "px"),
        ("covisibility_local_ba_elapsed_ms_total", "ms"),
        ("covisibility_local_ba_elapsed_ms_mean", "ms"),
        ("covisibility_local_ba_elapsed_ms_max", "ms"),
    ]
    args: list[str] = []
    for name, unit in specs:
        value = parse_number(summary.get(name))
        if value is None:
            continue
        args.extend(["--metric", f"{name}={value}:{unit}"])
    if parse_number(summary.get("ate_rigid_rmse_m")) is not None:
        args.extend(["--primary-metric", "ate_rigid_rmse_m"])
    return args


def sequences(args: argparse.Namespace) -> list[tuple[str, Path]]:
    seqs = [(seq, args.euroc_root / seq) for seq in args.sequence]
    seqs.extend((path.name, path) for path in args.euroc_dir)
    if not seqs:
        seqs = [(seq, args.euroc_root / seq) for seq in ["MH_01_easy", "V1_01_easy", "V2_01_easy"]]
    return seqs


def build_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [cargo_executable(), "build", "--features", "image-io", "--example", EXAMPLE]
    if args.profile == "release":
        cmd.insert(2, "--release")
    return cmd


def demo_base_cmd(args: argparse.Namespace, euroc_dir: Path, out_dir: Path) -> list[str]:
    cmd = [cargo_executable(), "run", "--features", "image-io", "--example", EXAMPLE]
    if args.profile == "release":
        cmd.insert(2, "--release")
    cmd.extend([
        "--",
        "--euroc-dir",
        str(euroc_dir),
        "--out-dir",
        str(out_dir),
        "--max-frames",
        str(args.max_frames),
    ])
    if args.demo_args:
        cmd.extend(shlex.split(args.demo_args))
    return cmd


def enabled_flags(args: argparse.Namespace) -> list[str]:
    flags = [
        "--covisibility-local-ba",
        "--covisibility-local-ba-min-keyframes",
        str(args.min_keyframes),
        "--covisibility-local-ba-trigger-every",
        str(args.trigger_every),
        "--covisibility-local-ba-max-neighbor-keyframes",
        str(args.max_neighbor_keyframes),
        "--covisibility-local-ba-min-shared",
        str(args.min_shared),
        "--covisibility-local-ba-max-boundary-keyframes",
        str(args.max_boundary_keyframes),
        "--covisibility-local-ba-min-boundary-observations",
        str(args.min_boundary_observations),
        "--covisibility-local-ba-fallback-min-boundary-observations",
        str(args.fallback_min_boundary_observations or "none"),
        "--covisibility-local-ba-min-active-observations",
        str(args.min_active_observations),
        "--covisibility-local-ba-outlier-threshold-px",
        str(args.outlier_threshold_px),
    ]
    if args.max_landmarks is not None:
        flags.extend(["--covisibility-local-ba-max-landmarks", str(args.max_landmarks)])
    if args.remove_outliers:
        flags.append("--covisibility-local-ba-remove-outliers")
    if args.max_outlier_observation_ratio is not None:
        flags.extend(
            [
                "--covisibility-local-ba-max-outlier-observation-ratio",
                str(args.max_outlier_observation_ratio),
            ]
        )
    if args.boundary_support_min_optimized_keyframes is not None:
        flags.extend(
            [
                "--covisibility-local-ba-boundary-support-min-optimized-keyframes",
                str(args.boundary_support_min_optimized_keyframes),
                "--covisibility-local-ba-boundary-support-min-fixed-keyframes",
                str(args.boundary_support_min_fixed_keyframes),
            ]
        )
    return flags


def status_from_result(returncode: int, summary: dict[str, str], args: argparse.Namespace) -> tuple[str, str | None]:
    if returncode != 0:
        return "failure", f"demo exited with status {returncode}"
    if not summary:
        return "failure", "summary.txt missing or empty"
    if args.dnf_if_tracking_success_below is not None:
        rate = parse_number(summary.get("tracking_success_rate"))
        if isinstance(rate, (int, float)) and rate < args.dnf_if_tracking_success_below:
            return "dnf", (
                f"tracking_success_rate {rate} below "
                f"{args.dnf_if_tracking_success_below}"
            )
    return "success", None


def capture_manifest(
    args: argparse.Namespace,
    *,
    sequence: str,
    euroc_dir: Path,
    variant: str,
    out_dir: Path,
    command: list[str],
    status: str,
    failure_reason: str | None,
    summary: dict[str, str],
    stamp: str,
) -> int:
    manifest_path = args.registry_dir / f"{BENCHMARK_ID}-{sequence}-{variant}-{stamp}.json"
    cmd = [
        sys.executable,
        "scripts/benchmark_registry.py",
        "capture",
        "--out",
        str(manifest_path),
        "--run-id",
        f"{BENCHMARK_ID}-{sequence}-{variant}-{stamp}",
        "--benchmark-id",
        BENCHMARK_ID,
        "--benchmark-name",
        BENCHMARK_NAME,
        "--script",
        f"examples/{EXAMPLE}.rs",
        "--protocol",
        "disabled/enabled OnlineSlamConfig::covisibility_local_ba A/B on identical EuRoC image/IMU command",
        "--docs",
        "docs/motion_based_vi_alignment.md",
        "--dataset-name",
        "EuRoC MAV",
        "--dataset-sequence",
        sequence,
        "--dataset-version",
        "ASL EuRoC MAV dataset",
        "--dataset-path",
        str(euroc_dir),
        "--status",
        status,
        "--command",
        shlex.join(command),
        "--feature",
        "image-io",
        "--profile",
        args.profile,
        "--config",
        f"variant={variant}",
        "--config",
        f"max_frames={args.max_frames}",
        "--config",
        f"demo_args={args.demo_args}",
        "--artifact",
        f"trajectory={out_dir / 'slam_trajectory.csv'}",
        "--artifact",
        f"errors={out_dir / 'slam_errors.csv'}",
        "--artifact",
        f"summary={out_dir / 'summary.txt'}",
        "--artifact",
        f"covisibility_ba_log={out_dir / 'covisibility_ba_log.txt'}",
    ]
    if variant == "enabled":
        for key, value in {
            "covisibility_local_ba_min_keyframes": args.min_keyframes,
            "covisibility_local_ba_trigger_every": args.trigger_every,
            "covisibility_local_ba_max_neighbor_keyframes": args.max_neighbor_keyframes,
            "covisibility_local_ba_min_shared": args.min_shared,
            "covisibility_local_ba_max_boundary_keyframes": args.max_boundary_keyframes,
            "covisibility_local_ba_min_boundary_observations": args.min_boundary_observations,
            "covisibility_local_ba_fallback_min_boundary_observations": args.fallback_min_boundary_observations,
            "covisibility_local_ba_max_landmarks": args.max_landmarks,
            "covisibility_local_ba_min_active_observations": args.min_active_observations,
            "covisibility_local_ba_outlier_threshold_px": args.outlier_threshold_px,
            "covisibility_local_ba_remove_outliers": args.remove_outliers,
            "covisibility_local_ba_max_outlier_observation_ratio": args.max_outlier_observation_ratio,
            "covisibility_local_ba_boundary_support_min_optimized_keyframes": args.boundary_support_min_optimized_keyframes,
            "covisibility_local_ba_boundary_support_min_fixed_keyframes": args.boundary_support_min_fixed_keyframes,
        }.items():
            cmd.extend(["--config", f"{key}={value}"])
    if failure_reason:
        cmd.extend(["--failure-reason", failure_reason])
    cmd.extend(metric_args(summary))
    if args.dry_run:
        print(shlex.join(cmd))
        return 0
    return run_command(cmd, None, False)


def run_variant(
    args: argparse.Namespace,
    sequence: str,
    euroc_dir: Path,
    variant: str,
    stamp: str,
) -> int:
    out_dir = args.out_root / sequence / variant
    log_path = args.out_root / sequence / f"{variant}.log"
    cmd = demo_base_cmd(args, euroc_dir, out_dir)
    if variant == "enabled":
        cmd.extend(enabled_flags(args))
    print(f"=== {sequence} {variant} -> {out_dir} ===")
    returncode = run_command(cmd, log_path, args.dry_run)
    if args.dry_run:
        if not args.no_capture_registry:
            manifest_path = (
                args.registry_dir / f"{BENCHMARK_ID}-{sequence}-{variant}-{stamp}.json"
            )
            print(f"# registry capture after run: {manifest_path}")
        return 0
    summary = {} if args.dry_run else parse_summary(out_dir / "summary.txt")
    status, failure_reason = status_from_result(returncode, summary, args)
    if not args.no_capture_registry:
        capture_code = capture_manifest(
            args,
            sequence=sequence,
            euroc_dir=euroc_dir,
            variant=variant,
            out_dir=out_dir,
            command=cmd,
            status=status,
            failure_reason=failure_reason,
            summary=summary,
            stamp=stamp,
        )
        if capture_code != 0:
            return capture_code
    return returncode


def main() -> int:
    args = parse_args()
    if args.max_frames < 0:
        raise SystemExit("--max-frames must be >= 0")
    if args.min_keyframes < 1 or args.trigger_every < 1 or args.min_shared < 1:
        raise SystemExit("--min-keyframes, --trigger-every, and --min-shared must be >= 1")
    if args.min_active_observations < 1:
        raise SystemExit("--min-active-observations must be >= 1")
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
    if (
        args.fallback_min_boundary_observations is not None
        and args.fallback_min_boundary_observations.lower() != "none"
    ):
        try:
            fallback_min = int(args.fallback_min_boundary_observations)
        except ValueError as exc:
            raise SystemExit("--fallback-min-boundary-observations must be an integer or 'none'") from exc
        if fallback_min < 1:
            raise SystemExit("--fallback-min-boundary-observations must be >= 1 or 'none'")

    seqs = sequences(args)
    if not args.dry_run:
        missing = [path for _, path in seqs if not path.exists()]
        if missing:
            joined = "\n".join(f"  {path}" for path in missing)
            raise SystemExit(f"missing EuRoC sequence directory/directories:\n{joined}")

    if not args.skip_build:
        code = run_command(build_cmd(args), None, args.dry_run)
        if code != 0:
            return code

    stamp = utc_stamp()
    variants = [args.only] if args.only else ["disabled", "enabled"]
    failures = 0
    for sequence, euroc_dir in seqs:
        for variant in variants:
            code = run_variant(args, sequence, euroc_dir, variant, stamp)
            if code != 0:
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
