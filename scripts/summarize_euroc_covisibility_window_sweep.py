#!/usr/bin/env python3
"""Summarize EuRoC covisibility-local-BA window-cap sweep manifests.

This reads benchmark-registry manifests from
`scripts/run_euroc_covisibility_local_ba_ab.py` and renders a table for
neighbor/boundary keyframe cap A/B runs. It complements the landmark-cap
runtime sweep: landmark budget stays fixed while the selected keyframe window
changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ID = "euroc-covisibility-local-ba"
DEFAULT_WINDOWS = [(5, 5), (10, 10), (15, 15)]
DEFAULT_SEQUENCES = ["MH_01_easy", "MH_03_medium", "MH_05_difficult"]


def parse_window_cap(raw: str) -> tuple[int, int]:
    normalized = raw.replace(",", ":")
    if ":" not in normalized:
        raise argparse.ArgumentTypeError("window cap must be NEIGHBOR:BOUNDARY")
    left, right = normalized.split(":", 1)
    try:
        neighbor = int(left)
        boundary = int(right)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("window cap values must be integers") from exc
    if neighbor < 1 or boundary < 1:
        raise argparse.ArgumentTypeError("window cap values must be >= 1")
    return neighbor, boundary


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
        default=Path("docs/generated/euroc_covisibility_window_sweep.md"),
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
        "--window-cap",
        action="append",
        type=parse_window_cap,
        default=None,
        help="neighbor:boundary keyframe cap pair to include; may be repeated",
    )
    parser.add_argument("--landmark-cap", type=int, default=200)
    parser.add_argument("--min-keyframes", type=int, default=3)
    parser.add_argument("--trigger-every", type=int, default=1)
    parser.add_argument("--min-active-observations", type=int, default=20)
    parser.add_argument("--fallback", default="none")
    parser.add_argument("--remove-outliers", action="store_true")
    parser.add_argument("--max-outlier-observation-ratio", default="none")
    parser.add_argument("--boundary-support-min-optimized-keyframes", default="none")
    parser.add_argument("--boundary-support-min-fixed-keyframes", type=int, default=0)
    args = parser.parse_args()
    if args.sequence is None:
        args.sequence = DEFAULT_SEQUENCES.copy()
    if args.window_cap is None:
        args.window_cap = DEFAULT_WINDOWS.copy()
    return args


def metric_map(manifest: dict[str, Any]) -> dict[str, Any]:
    return {metric["name"]: metric.get("value") for metric in manifest.get("metrics", [])}


def normalize_optional(value: Any) -> str:
    if value is None:
        return "none"
    return str(value)


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


def load_latest_runs(args: argparse.Namespace) -> dict[tuple[str, int, int], dict[str, Any]]:
    registry_dir = args.registry_dir if args.registry_dir.is_absolute() else ROOT / args.registry_dir
    selected: dict[tuple[str, int, int], dict[str, Any]] = {}
    wanted_windows = set(args.window_cap)
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
        if int_param(params, "covisibility_local_ba_max_landmarks") != args.landmark_cap:
            continue
        if int_param(params, "covisibility_local_ba_min_keyframes") != args.min_keyframes:
            continue
        if int_param(params, "covisibility_local_ba_trigger_every") != args.trigger_every:
            continue
        if int_param(params, "covisibility_local_ba_min_active_observations") != args.min_active_observations:
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
        neighbor = int_param(params, "covisibility_local_ba_max_neighbor_keyframes")
        boundary = int_param(params, "covisibility_local_ba_max_boundary_keyframes")
        if neighbor is None or boundary is None:
            continue
        if (neighbor, boundary) not in wanted_windows:
            continue
        key = (sequence, neighbor, boundary)
        previous = selected.get(key)
        if previous is None or manifest.get("created_utc", "") > previous.get("created_utc", ""):
            selected[key] = manifest
    return selected


def expected_keys(args: argparse.Namespace) -> list[tuple[str, int, int]]:
    return [
        (sequence, neighbor, boundary)
        for sequence in args.sequence
        for neighbor, boundary in sorted(args.window_cap)
    ]


def missing_expected_runs(
    args: argparse.Namespace,
    runs: dict[tuple[str, int, int], dict[str, Any]],
) -> list[tuple[str, int, int]]:
    return [key for key in expected_keys(args) if key not in runs]


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render(args: argparse.Namespace, runs: dict[tuple[str, int, int], dict[str, Any]]) -> str:
    lines = [
        "# EuRoC Covisibility BA Window Sweep",
        "",
        "Generated from benchmark-registry run manifests. This sweep keeps",
        f"`--max-frames {args.max_frames}`,",
        f"`--covisibility-local-ba-max-landmarks {args.landmark_cap}`,",
        f"`--covisibility-local-ba-min-keyframes {args.min_keyframes}`,",
        f"`--covisibility-local-ba-trigger-every {args.trigger_every}`,",
        f"`--covisibility-local-ba-min-active-observations {args.min_active_observations}`,",
        f"remove-outliers `{args.remove_outliers}`,",
        f"fallback boundary selection `{args.fallback}`,",
        f"max-outlier observation ratio `{args.max_outlier_observation_ratio}`,",
        "and boundary support gate "
        f"`{args.boundary_support_min_optimized_keyframes}/"
        f"{args.boundary_support_min_fixed_keyframes}`; only the local BA",
        "neighbor/boundary keyframe caps change.",
        "",
        "| sequence | neighbor KF | boundary KF | tracking | rigid ATE m | sim ATE m | BA success | BA fail | quality reject | boundary support | mean ms | max ms | run id |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for sequence, neighbor, boundary in expected_keys(args):
        manifest = runs.get((sequence, neighbor, boundary))
        if manifest is None:
            lines.append(
                f"| {sequence} | {neighbor} | {boundary} |  |  |  |  |  |  |  |  |  | missing |"
            )
            continue
        metrics = metric_map(manifest)
        lines.append(
            "| "
            + " | ".join(
                [
                    sequence,
                    str(neighbor),
                    str(boundary),
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
            "- The landmark cap is fixed, so deltas mainly reflect keyframe-window selection and observation count.",
            "- This is registry-backed evidence for choosing an opt-in BA window budget, not a headline benchmark claim.",
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
