#!/usr/bin/env python3
"""Summarize MH_05 covisibility-BA boundary-support gate manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ID = "euroc-covisibility-local-ba"
DEFAULT_GATES = [
    ("quality-gate only", "none", 0),
    ("boundary7/2", "7", 2),
    ("boundary10/2", "10", 2),
]


def parse_gate(raw: str) -> tuple[str, str, int]:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("gate must be LABEL:MIN_OPTIMIZED:MIN_FIXED")
    label, min_optimized, min_fixed = parts
    if not label:
        raise argparse.ArgumentTypeError("gate label must be non-empty")
    min_optimized = normalize_optional(min_optimized)
    try:
        parsed_fixed = int(min_fixed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("gate min-fixed value must be an integer") from exc
    if min_optimized == "none":
        if parsed_fixed != 0:
            raise argparse.ArgumentTypeError("gate min-fixed must be 0 when min-optimized is none")
    else:
        try:
            parsed_min_optimized = int(min_optimized)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("gate min-optimized must be an integer or none") from exc
        if parsed_min_optimized < 1:
            raise argparse.ArgumentTypeError("gate min-optimized must be >= 1 or none")
        if parsed_fixed < 1:
            raise argparse.ArgumentTypeError("gate min-fixed must be >= 1 when min-optimized is set")
    return label, min_optimized, parsed_fixed


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
        default=Path("docs/generated/euroc_covisibility_mh05_boundary_support_gate_sweep.md"),
        help="Markdown output path",
    )
    parser.add_argument("--sequence", default="MH_05_difficult")
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--neighbor-keyframes", type=int, default=10)
    parser.add_argument("--boundary-keyframes", type=int, default=10)
    parser.add_argument("--landmark-cap", type=int, default=200)
    parser.add_argument("--min-keyframes", type=int, default=3)
    parser.add_argument("--trigger-every", type=int, default=1)
    parser.add_argument("--min-active-observations", type=int, default=20)
    parser.add_argument("--fallback", default="none")
    parser.add_argument("--remove-outliers", action="store_true")
    parser.add_argument("--max-outlier-observation-ratio", default="0.3")
    parser.add_argument(
        "--gate",
        action="append",
        type=parse_gate,
        default=None,
        help="enabled gate row as LABEL:MIN_OPTIMIZED:MIN_FIXED; use none:0 for quality-gate-only",
    )
    args = parser.parse_args()
    if args.gate is None:
        args.gate = DEFAULT_GATES.copy()
    return args


def metric_map(manifest: dict[str, Any]) -> dict[str, Any]:
    return {metric["name"]: metric.get("value") for metric in manifest.get("metrics", [])}


def normalize_optional(value: Any) -> str:
    if value in {None, "None", "none"}:
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


def gate_matches(params: dict[str, Any], gate: tuple[str, str, int]) -> bool:
    _, min_optimized, min_fixed = gate
    actual_min_optimized = normalize_optional(
        params.get("covisibility_local_ba_boundary_support_min_optimized_keyframes")
    )
    if actual_min_optimized != min_optimized:
        return False
    return (
        int_param_or(params, "covisibility_local_ba_boundary_support_min_fixed_keyframes", 0)
        == min_fixed
    )


def enabled_matches(args: argparse.Namespace, params: dict[str, Any], gate: tuple[str, str, int]) -> bool:
    checks = {
        "covisibility_local_ba_max_neighbor_keyframes": args.neighbor_keyframes,
        "covisibility_local_ba_max_boundary_keyframes": args.boundary_keyframes,
        "covisibility_local_ba_min_keyframes": args.min_keyframes,
        "covisibility_local_ba_trigger_every": args.trigger_every,
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
    return max_outlier_ratio == args.max_outlier_observation_ratio and gate_matches(params, gate)


def load_latest_runs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    registry_dir = args.registry_dir if args.registry_dir.is_absolute() else ROOT / args.registry_dir
    selected: dict[str, dict[str, Any]] = {}
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
            for gate in args.gate:
                candidate_label = gate[0]
                if enabled_matches(args, params, gate):
                    label = candidate_label
                    break
        if label is None:
            continue
        previous = selected.get(label)
        if previous is None or manifest.get("created_utc", "") > previous.get("created_utc", ""):
            selected[label] = manifest
    return selected


def expected_labels(args: argparse.Namespace) -> list[str]:
    return ["disabled", *[label for label, _, _ in args.gate]]


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


def delta(value: Any, baseline: Any) -> float | None:
    if isinstance(value, (int, float)) and isinstance(baseline, (int, float)):
        return float(value) - float(baseline)
    return None


def gate_label(args: argparse.Namespace, label: str) -> str:
    if label == "disabled":
        return "disabled"
    for candidate_label, min_optimized, min_fixed in args.gate:
        if candidate_label == label:
            return f"{min_optimized}/{min_fixed}"
    return ""


def verdict(label: str, tracking_delta: float | None, mean_ms_delta: float | None) -> str:
    if label == "disabled":
        return ""
    if tracking_delta is None or mean_ms_delta is None:
        return "incomplete"
    if tracking_delta < 0.0:
        return "reject"
    if mean_ms_delta < 0.0:
        return "candidate"
    return "neutral"


def render(args: argparse.Namespace, runs: dict[str, dict[str, Any]]) -> str:
    quality_baseline = runs.get(args.gate[0][0]) if args.gate else None
    baseline_tracking = metric(quality_baseline, "tracking_success_rate")
    baseline_mean_ms = metric(quality_baseline, "covisibility_local_ba_elapsed_ms_mean")
    lines = [
        "# EuRoC MH_05 Boundary Support Gate Sweep",
        "",
        "Generated from benchmark-registry run manifests. This diagnostic table keeps",
        f"`--max-frames {args.max_frames}`, min-keyframes `{args.min_keyframes}`, trigger-every `{args.trigger_every}`,",
        f"neighbor/boundary `{args.neighbor_keyframes}/{args.boundary_keyframes}`,",
        f"landmark cap `{args.landmark_cap}`, active-observation floor `{args.min_active_observations}`,",
        f"max-outlier observation ratio `{args.max_outlier_observation_ratio}`,",
        "and varies only the pre-solve boundary-support gate.",
        "",
        "| config | gate min opt/fixed | tracking | tracking delta vs qg | rigid ATE m | sim ATE m | BA success | BA fail | quality reject | boundary support | no-local-landmarks | mean ms | mean ms delta vs qg | verdict | run id |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for label in expected_labels(args):
        manifest = runs.get(label)
        if manifest is None:
            lines.append(f"| {label} | {gate_label(args, label)} |  |  |  |  |  |  |  |  |  |  |  | missing | missing |")
            continue
        tracking = metric(manifest, "tracking_success_rate")
        mean_ms = metric(manifest, "covisibility_local_ba_elapsed_ms_mean")
        tracking_delta = None if label == "disabled" else delta(tracking, baseline_tracking)
        mean_ms_delta = None if label == "disabled" else delta(mean_ms, baseline_mean_ms)
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    gate_label(args, label),
                    fmt(tracking, 3),
                    fmt(tracking_delta, 3),
                    fmt(metric(manifest, "ate_rigid_rmse_m"), 4),
                    fmt(metric(manifest, "ate_similarity_rmse_m"), 4),
                    fmt(metric(manifest, "covisibility_local_ba_successes"), 0),
                    fmt(metric(manifest, "covisibility_local_ba_failures"), 0),
                    fmt(metric(manifest, "covisibility_local_ba_quality_gate_failures"), 0),
                    fmt(metric(manifest, "covisibility_local_ba_boundary_support_failures"), 0),
                    fmt(metric(manifest, "covisibility_local_ba_no_local_landmarks_failures"), 0),
                    fmt(mean_ms, 3),
                    fmt(mean_ms_delta, 3),
                    verdict(label, tracking_delta, mean_ms_delta),
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
            "- `qg` is the quality-gate-only enabled row with no pre-solve boundary-support gate.",
            "- Negative `mean ms delta vs qg` means the pre-solve gate reduced average covisibility-BA trigger time.",
            "- `candidate` preserves quality-gate-only tracking while reducing mean trigger time; `reject` loses tracking.",
            "- This is diagnostic evidence for one MH_05 failure mode, not a default-policy claim.",
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
