#!/usr/bin/env python3
"""Summarize MH_05 covisibility-local-BA mitigation manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ID = "euroc-covisibility-local-ba"
DEFAULT_CONFIGS = [
    ("enabled min3/every1", 3, 1),
    ("enabled min6/every3", 6, 3),
    ("enabled min10/every5", 10, 5),
]


def parse_config(raw: str) -> tuple[str, int, int]:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("config must be LABEL:MIN_KEYFRAMES:TRIGGER_EVERY")
    label, min_keyframes, trigger_every = parts
    if not label:
        raise argparse.ArgumentTypeError("config label must be non-empty")
    try:
        parsed_min = int(min_keyframes)
        parsed_every = int(trigger_every)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("config min/trigger values must be integers") from exc
    if parsed_min < 1 or parsed_every < 1:
        raise argparse.ArgumentTypeError("config min/trigger values must be >= 1")
    return label, parsed_min, parsed_every


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
        default=Path("docs/generated/euroc_covisibility_mh05_mitigation.md"),
        help="Markdown output path",
    )
    parser.add_argument("--sequence", default="MH_05_difficult")
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--neighbor-keyframes", type=int, default=10)
    parser.add_argument("--boundary-keyframes", type=int, default=10)
    parser.add_argument("--landmark-cap", type=int, default=200)
    parser.add_argument("--min-active-observations", type=int, default=20)
    parser.add_argument("--fallback", default="none")
    parser.add_argument("--remove-outliers", action="store_true")
    parser.add_argument("--max-outlier-observation-ratio", default="none")
    parser.add_argument("--boundary-support-min-optimized-keyframes", default="none")
    parser.add_argument("--boundary-support-min-fixed-keyframes", type=int, default=0)
    parser.add_argument(
        "--config",
        action="append",
        type=parse_config,
        default=None,
        help="enabled config as LABEL:MIN_KEYFRAMES:TRIGGER_EVERY; may be repeated",
    )
    args = parser.parse_args()
    if args.config is None:
        args.config = DEFAULT_CONFIGS.copy()
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


def enabled_matches(args: argparse.Namespace, params: dict[str, Any], config: tuple[str, int, int]) -> bool:
    _, min_keyframes, trigger_every = config
    checks = {
        "covisibility_local_ba_max_neighbor_keyframes": args.neighbor_keyframes,
        "covisibility_local_ba_max_boundary_keyframes": args.boundary_keyframes,
        "covisibility_local_ba_min_keyframes": min_keyframes,
        "covisibility_local_ba_trigger_every": trigger_every,
        "covisibility_local_ba_max_landmarks": args.landmark_cap,
        "covisibility_local_ba_min_active_observations": args.min_active_observations,
    }
    for key, expected in checks.items():
        if int_param(params, key) != expected:
            return False
    if bool_param(params, "covisibility_local_ba_remove_outliers") != args.remove_outliers:
        return False
    fallback = normalize_optional(params.get("covisibility_local_ba_fallback_min_boundary_observations"))
    if fallback != args.fallback:
        return False
    max_outlier_ratio = normalize_optional(
        params.get("covisibility_local_ba_max_outlier_observation_ratio")
    )
    if max_outlier_ratio != args.max_outlier_observation_ratio:
        return False
    boundary_min_optimized = normalize_optional(
        params.get("covisibility_local_ba_boundary_support_min_optimized_keyframes")
    )
    if boundary_min_optimized != args.boundary_support_min_optimized_keyframes:
        return False
    return (
        int_param_or(params, "covisibility_local_ba_boundary_support_min_fixed_keyframes", 0)
        == args.boundary_support_min_fixed_keyframes
    )


def load_latest_runs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    registry_dir = args.registry_dir if args.registry_dir.is_absolute() else ROOT / args.registry_dir
    selected: dict[str, dict[str, Any]] = {}
    configs_by_label = {label: config for label, *config in args.config}
    for path in sorted(registry_dir.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("benchmark", {}).get("id") != BENCHMARK_ID:
            continue
        if manifest.get("status") != "success":
            continue
        if manifest.get("dataset", {}).get("sequence") != args.sequence:
            continue
        params = manifest.get("config", {}).get("params", {})
        if params.get("max_frames") != args.max_frames:
            continue
        label = None
        if params.get("variant") == "disabled":
            label = "disabled"
        elif params.get("variant") == "enabled":
            for candidate in args.config:
                candidate_label = candidate[0]
                if enabled_matches(args, params, candidate):
                    label = candidate_label
                    break
        if label is None or (label != "disabled" and label not in configs_by_label):
            continue
        previous = selected.get(label)
        if previous is None or manifest.get("created_utc", "") > previous.get("created_utc", ""):
            selected[label] = manifest
    return selected


def expected_labels(args: argparse.Namespace) -> list[str]:
    return ["disabled", *[label for label, _, _ in args.config]]


def missing_expected_runs(args: argparse.Namespace, runs: dict[str, dict[str, Any]]) -> list[str]:
    return [label for label in expected_labels(args) if label not in runs]


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


def render(args: argparse.Namespace, runs: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# EuRoC MH_05 Covisibility BA Mitigation",
        "",
        "Generated from benchmark-registry run manifests. This targeted sweep keeps",
        f"`--max-frames {args.max_frames}`, neighbor/boundary `{args.neighbor_keyframes}/{args.boundary_keyframes}`,",
        f"landmark cap `{args.landmark_cap}`, active-observation floor `{args.min_active_observations}`,",
        f"remove-outliers `{args.remove_outliers}`, fallback boundary selection `{args.fallback}`,",
        f"max-outlier observation ratio `{args.max_outlier_observation_ratio}`,",
        "and boundary support gate "
        f"`{args.boundary_support_min_optimized_keyframes}/"
        f"{args.boundary_support_min_fixed_keyframes}`; only BA start/trigger cadence changes.",
        "",
        "| config | tracking | rigid ATE m | sim ATE m | map keyframes | BA triggers | BA success | BA fail | quality reject | boundary support | no-local-landmarks | mean ms | run id |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for label in expected_labels(args):
        manifest = runs.get(label)
        if manifest is None:
            lines.append(f"| {label} |  |  |  |  |  |  |  |  |  |  |  | missing |")
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    fmt(metric(manifest, "tracking_success_rate"), 3),
                    fmt(metric(manifest, "ate_rigid_rmse_m"), 4),
                    fmt(metric(manifest, "ate_similarity_rmse_m"), 4),
                    fmt(metric(manifest, "map_keyframes"), 0),
                    fmt(metric(manifest, "covisibility_local_ba_triggers"), 0),
                    fmt(metric(manifest, "covisibility_local_ba_successes"), 0),
                    fmt(metric(manifest, "covisibility_local_ba_failures"), 0),
                    fmt(metric(manifest, "covisibility_local_ba_quality_gate_failures"), 0),
                    fmt(metric(manifest, "covisibility_local_ba_boundary_support_failures"), 0),
                    fmt(metric(manifest, "covisibility_local_ba_no_local_landmarks_failures"), 0),
                    fmt(metric(manifest, "covisibility_local_ba_elapsed_ms_mean"), 3),
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
            "- `min3/every1` is the original 400-frame 10/10 enabled row that regressed MH_05.",
            "- Later starts and less frequent triggers recover tracking stability but do not beat the disabled baseline yet.",
            "- This is diagnostic evidence for the opt-in covisibility BA path, not a default-policy claim.",
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
