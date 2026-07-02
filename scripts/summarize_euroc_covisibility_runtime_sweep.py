#!/usr/bin/env python3
"""Summarize EuRoC covisibility-local-BA runtime sweep manifests.

This reads benchmark-registry manifests from
`scripts/run_euroc_covisibility_local_ba_ab.py` and renders a compact table for
landmark-cap A/B runs at a fixed keyframe-window budget. The table is
intentionally registry-backed so BA runtime evidence stays tied to the exact
command, commit, feature set, and artifacts that produced it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ID = "euroc-covisibility-local-ba"


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
        default=Path("docs/generated/euroc_covisibility_runtime_sweep.md"),
        help="Markdown output path",
    )
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument(
        "--sequence",
        action="append",
        default=None,
        help="sequence to include; may be repeated",
    )
    parser.add_argument(
        "--landmark-cap",
        action="append",
        type=int,
        default=None,
        help="covisibility-local-BA max-landmarks cap to include; may be repeated",
    )
    parser.add_argument("--neighbor-keyframes", type=int, default=10)
    parser.add_argument("--boundary-keyframes", type=int, default=10)
    parser.add_argument("--min-active-observations", type=int, default=20)
    parser.add_argument("--fallback", default="none")
    parser.add_argument("--remove-outliers", action="store_true")
    parser.add_argument("--max-outlier-observation-ratio", default="none")
    parser.add_argument("--boundary-support-min-optimized-keyframes", default="none")
    parser.add_argument("--boundary-support-min-fixed-keyframes", type=int, default=0)
    args = parser.parse_args()
    if args.sequence is None:
        args.sequence = ["MH_03_medium"]
    if args.landmark_cap is None:
        args.landmark_cap = [100, 200, 400]
    return args


def metric_map(manifest: dict[str, Any]) -> dict[str, Any]:
    return {metric["name"]: metric.get("value") for metric in manifest.get("metrics", [])}


def normalize_optional(value: Any) -> str:
    if value is None:
        return "none"
    return str(value)


def manifest_landmark_cap(manifest: dict[str, Any]) -> int | None:
    params = manifest.get("config", {}).get("params", {})
    return int_param(params, "covisibility_local_ba_max_landmarks")


def int_param(params: dict[str, Any], key: str) -> int | None:
    value = params.get(key)
    if value in {None, "None", "none"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def int_param_or(params: dict[str, Any], key: str, default: int) -> int:
    value = int_param(params, key)
    return default if value is None else value


def bool_param(params: dict[str, Any], key: str) -> bool:
    value = params.get(key)
    if isinstance(value, bool):
        return value
    if value in {None, "None", "none"}:
        return False
    return str(value).lower() == "true"


def load_latest_runs(args: argparse.Namespace) -> dict[tuple[str, int], dict[str, Any]]:
    registry_dir = args.registry_dir if args.registry_dir.is_absolute() else ROOT / args.registry_dir
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(registry_dir.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("benchmark", {}).get("id") != BENCHMARK_ID:
            continue
        if manifest.get("status") != "success":
            continue
        params = manifest.get("config", {}).get("params", {})
        if params.get("variant") != "enabled":
            continue
        if params.get("max_frames") != args.max_frames:
            continue
        sequence = manifest.get("dataset", {}).get("sequence")
        if sequence not in args.sequence:
            continue
        cap = manifest_landmark_cap(manifest)
        if cap not in args.landmark_cap:
            continue
        if params.get("covisibility_local_ba_min_active_observations") != args.min_active_observations:
            continue
        if bool_param(params, "covisibility_local_ba_remove_outliers") != args.remove_outliers:
            continue
        fallback = normalize_optional(params.get("covisibility_local_ba_fallback_min_boundary_observations"))
        if fallback != args.fallback:
            continue
        max_outlier_ratio = normalize_optional(
            params.get("covisibility_local_ba_max_outlier_observation_ratio")
        )
        if max_outlier_ratio != args.max_outlier_observation_ratio:
            continue
        boundary_min_optimized = normalize_optional(
            params.get("covisibility_local_ba_boundary_support_min_optimized_keyframes")
        )
        if boundary_min_optimized != args.boundary_support_min_optimized_keyframes:
            continue
        if (
            int_param_or(
                params,
                "covisibility_local_ba_boundary_support_min_fixed_keyframes",
                0,
            )
            != args.boundary_support_min_fixed_keyframes
        ):
            continue
        if int_param(params, "covisibility_local_ba_max_neighbor_keyframes") != args.neighbor_keyframes:
            continue
        if int_param(params, "covisibility_local_ba_max_boundary_keyframes") != args.boundary_keyframes:
            continue
        key = (sequence, cap)
        previous = selected.get(key)
        if previous is None or manifest.get("created_utc", "") > previous.get("created_utc", ""):
            selected[key] = manifest
    return selected


def expected_keys(args: argparse.Namespace) -> list[tuple[str, int]]:
    return [
        (sequence, cap)
        for sequence in args.sequence
        for cap in sorted(args.landmark_cap)
    ]


def missing_expected_runs(
    args: argparse.Namespace,
    runs: dict[tuple[str, int], dict[str, Any]],
) -> list[tuple[str, int]]:
    return [key for key in expected_keys(args) if key not in runs]


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render(args: argparse.Namespace, runs: dict[tuple[str, int], dict[str, Any]]) -> str:
    lines = [
        "# EuRoC Covisibility BA Runtime Sweep",
        "",
        "Generated from benchmark-registry run manifests. This smoke sweep keeps",
        f"`--max-frames {args.max_frames}`,",
        f"`--covisibility-local-ba-max-neighbor-keyframes {args.neighbor_keyframes}`,",
        f"`--covisibility-local-ba-max-boundary-keyframes {args.boundary_keyframes}`,",
        f"`--covisibility-local-ba-min-active-observations {args.min_active_observations}`,",
        f"remove-outliers `{args.remove_outliers}`,",
        f"fallback boundary selection `{args.fallback}`,",
        f"max-outlier observation ratio `{args.max_outlier_observation_ratio}`,",
        "and boundary support gate "
        f"`{args.boundary_support_min_optimized_keyframes}/"
        f"{args.boundary_support_min_fixed_keyframes}`; only the local BA",
        "`--covisibility-local-ba-max-landmarks` cap changes.",
        "",
        "| sequence | max landmarks | tracking | rigid ATE m | sim ATE m | BA success | BA fail | quality reject | boundary support | mean ms | max ms | run id |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for sequence, cap in expected_keys(args):
        manifest = runs.get((sequence, cap))
        if manifest is None:
            lines.append(f"| {sequence} | {cap} |  |  |  |  |  |  |  |  |  | missing |")
            continue
        metrics = metric_map(manifest)
        lines.append(
            "| "
            + " | ".join(
                [
                    sequence,
                    str(cap),
                    fmt(metrics.get("tracking_success_rate"), 3),
                    fmt(metrics.get("ate_rigid_rmse_m"), 4),
                    fmt(metrics.get("ate_similarity_rmse_m"), 4),
                    fmt(metrics.get("covisibility_local_ba_successes"), 0),
                    fmt(metrics.get("covisibility_local_ba_failures"), 0),
                    fmt(metrics.get("covisibility_local_ba_quality_gate_failures"), 0),
                    fmt(metrics.get("covisibility_local_ba_boundary_support_failures"), 0),
                    fmt(metrics.get("covisibility_local_ba_elapsed_ms_mean"), 3),
                    fmt(metrics.get("covisibility_local_ba_elapsed_ms_max"), 3),
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
            "- Runtime is wall-clock milliseconds measured inside `OnlineSlamPipeline` around each triggered covisibility local BA attempt.",
            "- `mean ms` averages every trigger, including selection failures; `max ms` exposes single-frame BA spikes.",
            "- This is smoke evidence for choosing an opt-in BA window budget, not a headline benchmark claim.",
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
