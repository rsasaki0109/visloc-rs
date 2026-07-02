#!/usr/bin/env python3
"""Run EuRoC online-SLAM keyframe-policy A/B and register artifacts.

The runner executes `examples/euroc_online_slam_vi_image_demo` twice per
sequence:

* `fixed`: baseline `SimpleKeyframePolicy` with any caller-supplied
  `--keyframe-min-translation` demo arg
* `tracked_drop`: same command plus `--keyframe-tracked-landmark-ratio`
  and `--keyframe-min-tracked-landmarks-for-ratio`

Each run gets an output directory under `target/euroc_keyframe_policy_ab/`
and, by default, a benchmark registry manifest under
`benchmarks/registry/runs/euroc/`.

Example:

    python scripts/run_euroc_keyframe_policy_ab.py \
      --euroc-root /datasets/euroc \
      --sequence MH_01_easy --sequence V1_01_easy --sequence V2_01_easy \
      --max-frames 1500 \
      --tracked-landmark-ratio 0.9 \
      --demo-args "--gravity 0,0,-9.81 --feature-extractor hog --cross-check-matcher \
        --keyframe-min-translation 0.1 --max-pose-jump-meters 0.2 \
        --motion-model adaptive-imu-pose --pnp-pose-prior-warm-start \
        --vi-init-gyro-std-limit 0.5 --vi-init-accel-std-limit 5.0 \
        --stereo-bootstrap-strict"
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = "euroc_online_slam_vi_image_demo"
BENCHMARK_ID = "euroc-keyframe-tracked-landmark-drop"
BENCHMARK_NAME = "EuRoC online SLAM tracked-landmark keyframe policy A/B"


def cargo_exe() -> str:
    configured = os.environ.get("CARGO")
    if configured:
        return configured
    userprofile = os.environ.get("USERPROFILE")
    if userprofile:
        candidate = Path(userprofile) / ".cargo" / "bin" / "cargo.exe"
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
    parser.add_argument("--out-root", type=Path, default=Path("target/euroc_keyframe_policy_ab"))
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
        choices=["fixed", "tracked_drop"],
        default=None,
        help="run only one side of the A/B",
    )
    parser.add_argument(
        "--demo-args",
        default="",
        help="extra flags passed to both demo runs, parsed with shlex",
    )
    parser.add_argument("--tracked-landmark-ratio", type=float, default=0.9)
    parser.add_argument("--min-tracked-landmarks", type=int, default=20)
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
        ("imu_factors_staged", "count"),
        ("local_vi_ba_triggers", "count"),
        ("covisibility_local_ba_triggers", "count"),
        ("covisibility_local_ba_successes", "count"),
        ("covisibility_local_ba_failures", "count"),
        ("covisibility_local_ba_active_observation_gate_failures", "count"),
        ("covisibility_local_ba_boundary_fallback_active_gate_failures", "count"),
        ("covisibility_local_ba_quality_gate_failures", "count"),
        ("covisibility_local_ba_no_local_landmarks_failures", "count"),
        ("covisibility_local_ba_no_observations_failures", "count"),
        ("covisibility_local_ba_solver_failures", "count"),
        ("covisibility_local_ba_other_failures", "count"),
        ("covisibility_local_ba_boundary_fallback_successes", "count"),
        ("covisibility_local_ba_elapsed_ms_total", "ms"),
        ("covisibility_local_ba_elapsed_ms_mean", "ms"),
        ("covisibility_local_ba_elapsed_ms_max", "ms"),
        ("covisibility_local_map_used_frames", "count"),
        ("covisibility_local_map_mean_size", "count"),
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


def keyframe_decision_metric_args(out_dir: Path) -> list[str]:
    path = out_dir / "keyframe_decisions.csv"
    if not path.exists():
        return []

    counts: dict[str, int] = {}
    selected = 0
    rows = 0
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows += 1
            reason = row.get("reason", "")
            counts[reason] = counts.get(reason, 0) + 1
            if row.get("selected") == "1":
                selected += 1

    specs = [
        ("keyframe_decision_rows", rows),
        ("keyframe_selected_count", selected),
        ("keyframe_tracked_landmark_drop_count", counts.get("TrackedLandmarkDrop", 0)),
        ("keyframe_thresholds_met_count", counts.get("ThresholdsMet", 0)),
        ("keyframe_relocalized_count", counts.get("Relocalized", 0)),
        ("keyframe_translation_too_small_count", counts.get("TranslationTooSmall", 0)),
        ("keyframe_frame_gap_too_small_count", counts.get("FrameIdGapTooSmall", 0)),
        ("keyframe_no_mapping_count", counts.get("NoMapping", 0)),
    ]
    args: list[str] = []
    for name, value in specs:
        args.extend(["--metric", f"{name}={value}:count"])
    return args


def sequences(args: argparse.Namespace) -> list[tuple[str, Path]]:
    seqs = [(seq, args.euroc_root / seq) for seq in args.sequence]
    seqs.extend((path.name, path) for path in args.euroc_dir)
    if not seqs:
        seqs = [(seq, args.euroc_root / seq) for seq in ["MH_01_easy", "V1_01_easy", "V2_01_easy"]]
    return seqs


def build_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [cargo_exe(), "build", "--features", "image-io", "--example", EXAMPLE]
    if args.profile == "release":
        cmd.insert(2, "--release")
    return cmd


def demo_base_cmd(args: argparse.Namespace, euroc_dir: Path, out_dir: Path) -> list[str]:
    cmd = [cargo_exe(), "run", "--features", "image-io", "--example", EXAMPLE]
    if args.profile == "release":
        cmd.insert(2, "--release")
    cmd.extend(
        [
            "--",
            "--euroc-dir",
            str(euroc_dir),
            "--out-dir",
            str(out_dir),
            "--max-frames",
            str(args.max_frames),
        ]
    )
    if args.demo_args:
        cmd.extend(shlex.split(args.demo_args))
    return cmd


def variant_flags(args: argparse.Namespace, variant: str) -> list[str]:
    if variant == "fixed":
        return []
    return [
        "--keyframe-tracked-landmark-ratio",
        str(args.tracked_landmark_ratio),
        "--keyframe-min-tracked-landmarks-for-ratio",
        str(args.min_tracked_landmarks),
    ]


def status_from_result(
    returncode: int,
    summary: dict[str, str],
    args: argparse.Namespace,
) -> tuple[str, str | None]:
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
        "fixed/tracked-drop SimpleKeyframePolicy A/B on identical EuRoC image/IMU command",
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
        "--config",
        f"tracked_landmark_ratio={args.tracked_landmark_ratio}",
        "--config",
        f"min_tracked_landmarks={args.min_tracked_landmarks}",
        "--artifact",
        f"trajectory={out_dir / 'slam_trajectory.csv'}",
        "--artifact",
        f"errors={out_dir / 'slam_errors.csv'}",
        "--artifact",
        f"summary={out_dir / 'summary.txt'}",
        "--artifact",
        f"keyframe_decisions={out_dir / 'keyframe_decisions.csv'}",
        "--artifact",
        f"covisibility_ba_log={out_dir / 'covisibility_ba_log.txt'}",
    ]
    if failure_reason:
        cmd.extend(["--failure-reason", failure_reason])
    cmd.extend(metric_args(summary))
    cmd.extend(keyframe_decision_metric_args(out_dir))
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
    cmd.extend(variant_flags(args, variant))
    print(f"=== {sequence} {variant} -> {out_dir} ===")
    returncode = run_command(cmd, log_path, args.dry_run)
    if args.dry_run:
        if not args.no_capture_registry:
            manifest_path = args.registry_dir / f"{BENCHMARK_ID}-{sequence}-{variant}-{stamp}.json"
            print(f"# registry capture after run: {manifest_path}")
        return 0
    summary = parse_summary(out_dir / "summary.txt")
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
    if not math.isfinite(args.tracked_landmark_ratio) or not 0.0 <= args.tracked_landmark_ratio <= 1.0:
        raise SystemExit("--tracked-landmark-ratio must be in [0, 1]")
    if args.min_tracked_landmarks < 1:
        raise SystemExit("--min-tracked-landmarks must be >= 1")

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
    variants = [args.only] if args.only else ["fixed", "tracked_drop"]
    failures = 0
    for sequence, euroc_dir in seqs:
        for variant in variants:
            code = run_variant(args, sequence, euroc_dir, variant, stamp)
            if code != 0:
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
