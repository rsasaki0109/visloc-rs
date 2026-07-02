#!/usr/bin/env python3
"""Summarize EuRoC covisibility-BA active-observation sweep manifests.

The input is the benchmark registry produced by
`scripts/run_euroc_keyframe_policy_ab.py`. The script intentionally reads only
registered manifests, not ad-hoc target directories, so the rendered table stays
connected to reproducible commands and artifact hashes.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ID = "euroc-keyframe-tracked-landmark-drop"
ACTIVE_RE = re.compile(r"--covisibility-local-ba-min-active-observations\s+(\d+)")
FALLBACK_RE = re.compile(
    r"--covisibility-local-ba-fallback-min-boundary-observations\s+(\S+)"
)


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
        default=Path("docs/generated/euroc_active_observation_sweep.md"),
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
        "--active-floor",
        action="append",
        type=int,
        default=None,
        help="min-active-observations floor to include; may be repeated",
    )
    parser.add_argument(
        "--fallback",
        default="none",
        help="fallback boundary setting to include",
    )
    args = parser.parse_args()
    if args.sequence is None:
        args.sequence = ["MH_01_easy", "MH_03_medium", "MH_05_difficult"]
    if args.active_floor is None:
        args.active_floor = [20, 50]
    return args


def metric_map(manifest: dict[str, Any]) -> dict[str, Any]:
    return {metric["name"]: metric.get("value") for metric in manifest.get("metrics", [])}


def demo_args(manifest: dict[str, Any]) -> str:
    params = manifest.get("config", {}).get("params", {})
    value = params.get("demo_args", "")
    return str(value)


def active_floor(manifest: dict[str, Any]) -> int | None:
    match = ACTIVE_RE.search(demo_args(manifest))
    return int(match.group(1)) if match else None


def fallback_setting(manifest: dict[str, Any]) -> str | None:
    match = FALLBACK_RE.search(demo_args(manifest))
    return match.group(1) if match else None


def load_latest_runs(args: argparse.Namespace) -> dict[tuple[int, str, str], dict[str, Any]]:
    registry_dir = args.registry_dir if args.registry_dir.is_absolute() else ROOT / args.registry_dir
    selected: dict[tuple[int, str, str], dict[str, Any]] = {}
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
        variant = str(params.get("variant", ""))
        if variant not in {"fixed", "tracked_drop"}:
            continue
        floor = active_floor(manifest)
        if floor not in args.active_floor:
            continue
        fallback = fallback_setting(manifest)
        if fallback != args.fallback:
            continue
        key = (floor, sequence, variant)
        previous = selected.get(key)
        if previous is None or manifest.get("created_utc", "") > previous.get("created_utc", ""):
            selected[key] = manifest
    return selected


def expected_keys(args: argparse.Namespace) -> list[tuple[int, str, str]]:
    return [
        (floor, sequence, variant)
        for floor in sorted(args.active_floor)
        for sequence in args.sequence
        for variant in ["fixed", "tracked_drop"]
    ]


def missing_expected_runs(
    args: argparse.Namespace,
    runs: dict[tuple[int, str, str], dict[str, Any]],
) -> list[tuple[int, str, str]]:
    return [key for key in expected_keys(args) if key not in runs]


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render(args: argparse.Namespace, runs: dict[tuple[int, str, str], dict[str, Any]]) -> str:
    lines = [
        "# EuRoC Covisibility BA Active-Observation Sweep",
        "",
        "Generated from benchmark-registry run manifests. The sweep keeps covisibility BA enabled,",
        "`--covisibility-local-ba-max-landmarks 200`, fallback boundary selection disabled,",
        "`--keyframe-tracked-landmark-ratio 0.9`, and `--max-frames 400`; only",
        "`--covisibility-local-ba-min-active-observations` changes.",
        "",
        "Recommendation from this sweep: use `20` as the MH smoke-run opt-in value.",
        "`50` is more conservative, but it drops MH_05 tracked-drop continuity while not",
        "improving the cross-sequence picture enough to justify the stricter gate.",
        "",
        "| floor | sequence | variant | tracking | rigid ATE m | sim ATE m | map KF | BA success | BA fail | active gate | no local | solver fail | run id |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for floor, sequence, variant in expected_keys(args):
        manifest = runs.get((floor, sequence, variant))
        if manifest is None:
            lines.append(
                f"| {floor} | {sequence} | {variant} |  |  |  |  |  |  |  |  |  | missing |"
            )
            continue
        metrics = metric_map(manifest)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(floor),
                    sequence,
                    variant,
                    fmt(metrics.get("tracking_success_rate"), 3),
                    fmt(metrics.get("ate_rigid_rmse_m"), 4),
                    fmt(metrics.get("ate_similarity_rmse_m"), 4),
                    fmt(metrics.get("map_keyframes"), 0),
                    fmt(metrics.get("covisibility_local_ba_successes"), 0),
                    fmt(metrics.get("covisibility_local_ba_failures"), 0),
                    fmt(
                        metrics.get("covisibility_local_ba_active_observation_gate_failures"),
                        0,
                    ),
                    fmt(
                        metrics.get("covisibility_local_ba_no_local_landmarks_failures"),
                        0,
                    ),
                    fmt(metrics.get("covisibility_local_ba_solver_failures"), 0),
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
            "- `fixed` leaves tracked-landmark keyframe promotion disabled.",
            "- `tracked_drop` enables `--keyframe-tracked-landmark-ratio 0.9` with a count floor of 20.",
            "- `no local` means the selected local BA window had no eligible landmarks after the strict boundary threshold.",
            "- `solver fail` stayed at zero in the recorded sweep; the failures are selection/gating diagnostics, not optimizer crashes.",
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
