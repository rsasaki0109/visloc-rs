#!/usr/bin/env python3
"""Summarize EuRoC covisibility-local-BA disabled/enabled A/B manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ID = "euroc-covisibility-local-ba"
DEFAULT_SEQUENCES = ["MH_01_easy", "MH_03_medium", "MH_05_difficult"]


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
        default=Path("docs/generated/euroc_covisibility_ab_400.md"),
        help="Markdown output path",
    )
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument(
        "--sequence",
        action="append",
        default=None,
        help="sequence to include; may be repeated",
    )
    parser.add_argument("--enabled-neighbor-keyframes", type=int, default=10)
    parser.add_argument("--enabled-boundary-keyframes", type=int, default=10)
    parser.add_argument("--enabled-min-keyframes", type=int, default=3)
    parser.add_argument("--enabled-trigger-every", type=int, default=1)
    parser.add_argument("--enabled-landmark-cap", type=int, default=200)
    parser.add_argument("--enabled-min-active-observations", type=int, default=20)
    parser.add_argument("--enabled-fallback", default="none")
    parser.add_argument("--enabled-remove-outliers", action="store_true")
    parser.add_argument("--enabled-max-outlier-observation-ratio", default="none")
    parser.add_argument("--enabled-boundary-support-min-optimized-keyframes", default="none")
    parser.add_argument("--enabled-boundary-support-min-fixed-keyframes", type=int, default=0)
    args = parser.parse_args()
    if args.sequence is None:
        args.sequence = DEFAULT_SEQUENCES.copy()
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


def enabled_matches(args: argparse.Namespace, params: dict[str, Any]) -> bool:
    if int_param(params, "covisibility_local_ba_max_neighbor_keyframes") != args.enabled_neighbor_keyframes:
        return False
    if int_param(params, "covisibility_local_ba_max_boundary_keyframes") != args.enabled_boundary_keyframes:
        return False
    if int_param(params, "covisibility_local_ba_min_keyframes") != args.enabled_min_keyframes:
        return False
    if int_param(params, "covisibility_local_ba_trigger_every") != args.enabled_trigger_every:
        return False
    if int_param(params, "covisibility_local_ba_max_landmarks") != args.enabled_landmark_cap:
        return False
    if int_param(params, "covisibility_local_ba_min_active_observations") != args.enabled_min_active_observations:
        return False
    if bool_param(params, "covisibility_local_ba_remove_outliers") != args.enabled_remove_outliers:
        return False
    fallback = normalize_optional(params.get("covisibility_local_ba_fallback_min_boundary_observations"))
    if fallback != args.enabled_fallback:
        return False
    max_outlier_ratio = normalize_optional(
        params.get("covisibility_local_ba_max_outlier_observation_ratio")
    )
    if max_outlier_ratio != args.enabled_max_outlier_observation_ratio:
        return False
    boundary_min_optimized = normalize_optional(
        params.get("covisibility_local_ba_boundary_support_min_optimized_keyframes")
    )
    if boundary_min_optimized != args.enabled_boundary_support_min_optimized_keyframes:
        return False
    return (
        int_param_or(params, "covisibility_local_ba_boundary_support_min_fixed_keyframes", 0)
        == args.enabled_boundary_support_min_fixed_keyframes
    )


def load_latest_runs(args: argparse.Namespace) -> dict[tuple[str, str], dict[str, Any]]:
    registry_dir = args.registry_dir if args.registry_dir.is_absolute() else ROOT / args.registry_dir
    selected: dict[tuple[str, str], dict[str, Any]] = {}
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
        if sequence not in args.sequence:
            continue
        variant = params.get("variant")
        if variant not in {"disabled", "enabled"}:
            continue
        if variant == "enabled" and not enabled_matches(args, params):
            continue
        key = (sequence, variant)
        previous = selected.get(key)
        if previous is None or manifest.get("created_utc", "") > previous.get("created_utc", ""):
            selected[key] = manifest
    return selected


def expected_keys(args: argparse.Namespace) -> list[tuple[str, str]]:
    return [(sequence, variant) for sequence in args.sequence for variant in ("disabled", "enabled")]


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


def delta(left: Any, right: Any) -> float | None:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) - float(right)
    return None


def verdict(disabled: dict[str, Any] | None, enabled: dict[str, Any] | None) -> str:
    if disabled is None or enabled is None:
        return "missing"
    disabled_tracking = metric(disabled, "tracking_success_rate")
    enabled_tracking = metric(enabled, "tracking_success_rate")
    disabled_rigid = metric(disabled, "ate_rigid_rmse_m")
    enabled_rigid = metric(enabled, "ate_rigid_rmse_m")
    tracking_delta = delta(enabled_tracking, disabled_tracking)
    rigid_improvement = delta(disabled_rigid, enabled_rigid)
    if tracking_delta is None or rigid_improvement is None:
        return "incomplete"
    if tracking_delta >= 0.0 and rigid_improvement >= 0.0:
        return "win"
    if tracking_delta < 0.0 and rigid_improvement < 0.0:
        return "regress"
    return "mixed"


def render(args: argparse.Namespace, runs: dict[tuple[str, str], dict[str, Any]]) -> str:
    lines = [
        "# EuRoC Covisibility BA A/B",
        "",
        "Generated from benchmark-registry run manifests. This comparison keeps",
        f"`--max-frames {args.max_frames}` and compares the disabled baseline against",
        f"enabled covisibility local BA with neighbor/boundary `{args.enabled_neighbor_keyframes}/{args.enabled_boundary_keyframes}`,",
        f"min-keyframes `{args.enabled_min_keyframes}`,",
        f"trigger-every `{args.enabled_trigger_every}`,",
        f"landmark cap `{args.enabled_landmark_cap}`,",
        f"active-observation floor `{args.enabled_min_active_observations}`,",
        f"remove-outliers `{args.enabled_remove_outliers}`,",
        f"fallback boundary selection `{args.enabled_fallback}`,",
        f"max-outlier observation ratio `{args.enabled_max_outlier_observation_ratio}`,",
        "and boundary support gate "
        f"`{args.enabled_boundary_support_min_optimized_keyframes}/"
        f"{args.enabled_boundary_support_min_fixed_keyframes}`.",
        "",
        "| sequence | disabled tracking | enabled tracking | tracking delta | disabled rigid ATE m | enabled rigid ATE m | rigid improvement m | disabled sim ATE m | enabled sim ATE m | BA success | BA fail | quality reject | boundary support | mean ms | verdict | run ids |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for sequence in args.sequence:
        disabled = runs.get((sequence, "disabled"))
        enabled = runs.get((sequence, "enabled"))
        disabled_tracking = metric(disabled, "tracking_success_rate")
        enabled_tracking = metric(enabled, "tracking_success_rate")
        disabled_rigid = metric(disabled, "ate_rigid_rmse_m")
        enabled_rigid = metric(enabled, "ate_rigid_rmse_m")
        disabled_sim = metric(disabled, "ate_similarity_rmse_m")
        enabled_sim = metric(enabled, "ate_similarity_rmse_m")
        run_ids = []
        if disabled is not None:
            run_ids.append(str(disabled.get("run_id")))
        if enabled is not None:
            run_ids.append(str(enabled.get("run_id")))
        lines.append(
            "| "
            + " | ".join(
                [
                    sequence,
                    fmt(disabled_tracking, 3),
                    fmt(enabled_tracking, 3),
                    fmt(delta(enabled_tracking, disabled_tracking), 3),
                    fmt(disabled_rigid, 4),
                    fmt(enabled_rigid, 4),
                    fmt(delta(disabled_rigid, enabled_rigid), 4),
                    fmt(disabled_sim, 4),
                    fmt(enabled_sim, 4),
                    fmt(metric(enabled, "covisibility_local_ba_successes"), 0),
                    fmt(metric(enabled, "covisibility_local_ba_failures"), 0),
                    fmt(metric(enabled, "covisibility_local_ba_quality_gate_failures"), 0),
                    fmt(metric(enabled, "covisibility_local_ba_boundary_support_failures"), 0),
                    fmt(metric(enabled, "covisibility_local_ba_elapsed_ms_mean"), 3),
                    verdict(disabled, enabled),
                    "<br>".join(run_ids) if run_ids else "missing",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- Positive `tracking delta` means enabled BA tracked more frames than the disabled baseline.",
            "- Positive `rigid improvement` means enabled BA reduced rigid ATE RMSE.",
            "- This is scoped A/B evidence for the opt-in covisibility local BA path, not a headline benchmark claim.",
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
