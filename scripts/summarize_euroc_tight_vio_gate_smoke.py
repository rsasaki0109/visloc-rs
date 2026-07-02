#!/usr/bin/env python3
"""Summarize EuRoC tight-VIO local-BA writeback gate smoke manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ID = "euroc-tight-vio-local-ba-gates"
DEFAULT_SEQUENCES = ["MH_01_easy", "MH_03_medium", "MH_05_difficult"]
DEFAULT_VARIANTS = [
    "baseline",
    "adaptive_velocity",
    "gated_10mps",
    "gated_20mps",
    "velocity_tripwire_1mps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=Path("benchmarks/registry/runs/euroc"),
        help="directory containing EuRoC run manifests",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/generated/euroc_tight_vio_gate_smoke.md"),
        help="Markdown output path",
    )
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument(
        "--sequence",
        action="append",
        default=None,
        help="sequence to include; may be repeated",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help="variant to include; may be repeated",
    )
    args = parser.parse_args()
    if args.sequence is None:
        args.sequence = DEFAULT_SEQUENCES.copy()
    if args.variant is None:
        args.variant = DEFAULT_VARIANTS.copy()
    return args


def metric_map(manifest: dict[str, Any]) -> dict[str, Any]:
    return {metric["name"]: metric.get("value") for metric in manifest.get("metrics", [])}


def load_latest_runs(args: argparse.Namespace) -> dict[tuple[str, str], dict[str, Any]]:
    registry_dir = args.registry_dir if args.registry_dir.is_absolute() else ROOT / args.registry_dir
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    variants = set(args.variant)
    sequences = set(args.sequence)
    for path in sorted(registry_dir.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("benchmark", {}).get("id") != BENCHMARK_ID:
            continue
        if manifest.get("status") != "success":
            continue
        params = manifest.get("config", {}).get("params", {})
        if params.get("max_frames") != args.max_frames:
            continue
        sequence = manifest.get("dataset", {}).get("sequence")
        if sequence not in sequences:
            continue
        variant = str(params.get("variant", ""))
        if variant not in variants:
            continue
        key = (sequence, variant)
        previous = selected.get(key)
        if previous is None or manifest.get("created_utc", "") > previous.get("created_utc", ""):
            selected[key] = manifest
    return selected


def expected_keys(args: argparse.Namespace) -> list[tuple[str, str]]:
    return [(sequence, variant) for sequence in args.sequence for variant in args.variant]


def missing_expected_runs(
    args: argparse.Namespace,
    runs: dict[tuple[str, str], dict[str, Any]],
) -> list[tuple[str, str]]:
    return [key for key in expected_keys(args) if key not in runs]


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def metric(manifest: dict[str, Any] | None, name: str) -> Any:
    if manifest is None:
        return None
    return metric_map(manifest).get(name)


def param(manifest: dict[str, Any] | None, name: str) -> Any:
    if manifest is None:
        return None
    return manifest.get("config", {}).get("params", {}).get(name)


def as_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def delta(left: Any, right: Any) -> float | None:
    left_number = as_number(left)
    right_number = as_number(right)
    if left_number is None or right_number is None:
        return None
    return left_number - right_number


def velocity_cap(manifest: dict[str, Any] | None) -> str:
    cap = param(manifest, "local_vi_ba_reject_velocity_above_mps")
    if cap is None:
        return "none"
    return fmt(cap, 1)


def cost_ratio_cap(manifest: dict[str, Any] | None) -> str:
    cap = param(manifest, "local_vi_ba_reject_writeback_above")
    if cap is None:
        return "none"
    return fmt(cap, 2)


def verdict(
    baseline: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
    variant: str,
) -> str:
    if manifest is None:
        return "missing"
    if variant == "baseline":
        return "baseline"
    if baseline is None:
        return "missing baseline"

    tracking_delta = delta(metric(manifest, "tracking_success_rate"), metric(baseline, "tracking_success_rate"))
    rigid_delta = delta(metric(manifest, "ate_rigid_rmse_m"), metric(baseline, "ate_rigid_rmse_m"))
    if tracking_delta is None or rigid_delta is None:
        return "incomplete"
    if abs(tracking_delta) <= 0.0005 and abs(rigid_delta) <= 0.0005:
        return "non-interfering"
    if tracking_delta >= 0.0 and rigid_delta <= 0.0:
        return "win"
    if tracking_delta < 0.0 and rigid_delta > 0.0:
        return "regress"
    return "mixed"


def render(args: argparse.Namespace, runs: dict[tuple[str, str], dict[str, Any]]) -> str:
    lines = [
        "# EuRoC Tight VIO Gate Smoke",
        "",
        "Generated from benchmark-registry run manifests. This table compares",
        f"`--max-frames {args.max_frames}` local VI-BA smoke runs with the same HOG/cross-check",
        "front-end and motion-IMU warm start; only local VI-BA writeback gates change.",
        "",
        "Recommendation from this smoke: prefer `adaptive_velocity` over a raw fixed",
        "velocity cap. The adaptive gate is non-interfering on the recorded MH rows",
        "and avoids the fixed 10 m/s false rejection on MH_03. The 20 m/s fixed cap is",
        "also non-interfering here, but remains a scene-scale safety ceiling rather than",
        "a primary policy. The 1 m/s cap is an intentional tripwire.",
        "",
        "| sequence | variant | velocity cap m/s | adaptive threshold m/s | cost-ratio cap | rejects | velocity rejects | adaptive rejects | mirrors | tracking | tracking delta | map KF | rigid ATE m | rigid delta m | sim ATE m | sim scale | verdict | run id |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for sequence in args.sequence:
        baseline = runs.get((sequence, "baseline"))
        for variant in args.variant:
            manifest = runs.get((sequence, variant))
            if manifest is None:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            sequence,
                            variant,
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "missing",
                            "missing",
                        ]
                    )
                    + " |"
                )
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        sequence,
                        variant,
                        velocity_cap(manifest),
                        fmt(
                            metric(
                                manifest,
                                "local_vi_ba_last_adaptive_velocity_threshold_mps",
                            ),
                            3,
                        ),
                        cost_ratio_cap(manifest),
                        fmt(metric(manifest, "local_vi_ba_quality_gate_rejections"), 0),
                        fmt(metric(manifest, "local_vi_ba_velocity_gate_rejections"), 0),
                        fmt(
                            metric(
                                manifest,
                                "local_vi_ba_adaptive_velocity_gate_rejections",
                            ),
                            0,
                        ),
                        fmt(metric(manifest, "local_vi_ba_mirrors_into_imu_motion_model"), 0),
                        fmt(metric(manifest, "tracking_success_rate"), 3),
                        fmt(
                            delta(
                                metric(manifest, "tracking_success_rate"),
                                metric(baseline, "tracking_success_rate"),
                            ),
                            3,
                        ),
                        fmt(metric(manifest, "map_keyframes"), 0),
                        fmt(metric(manifest, "ate_rigid_rmse_m"), 4),
                        fmt(
                            delta(
                                metric(manifest, "ate_rigid_rmse_m"),
                                metric(baseline, "ate_rigid_rmse_m"),
                            ),
                            4,
                        ),
                        fmt(metric(manifest, "ate_similarity_rmse_m"), 4),
                        fmt(metric(manifest, "ate_similarity_scale"), 6),
                        verdict(baseline, manifest, variant),
                        str(manifest.get("run_id")),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- Negative `rigid delta` means the gated run reduced rigid ATE versus the per-sequence baseline.",
            "- `rejects` is the combined local VI-BA writeback quality gate counter; `velocity rejects` is the subset triggered by the velocity cap.",
            "- `adaptive threshold` is the final per-trigger adaptive velocity threshold from the run summary when that gate was enabled.",
            "- `mirrors` counts accepted local VI-BA velocity/bias writebacks mirrored into the IMU motion model.",
            "- Missing cells mean that optional exploratory cap was not run for that sequence yet.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    runs = load_latest_runs(args)
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(args, runs), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
