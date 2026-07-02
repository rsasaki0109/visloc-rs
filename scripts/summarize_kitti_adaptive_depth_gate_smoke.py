#!/usr/bin/env python3
"""Summarize KITTI adaptive/fixed stereo depth-gate smoke manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ID = "kitti-adaptive-depth-gate-smoke"
DEFAULT_VARIANTS = ["adaptive", "fixed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=Path("benchmarks/registry/runs/kitti"),
        help="directory containing KITTI run manifests",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/generated/kitti_adaptive_depth_gate_smoke.md"),
        help="Markdown output path",
    )
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--max-frames", type=int, default=2)
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        choices=DEFAULT_VARIANTS,
        help="variant to include; may be repeated",
    )
    args = parser.parse_args()
    if args.variant is None:
        args.variant = DEFAULT_VARIANTS.copy()
    return args


def metric_map(manifest: dict[str, Any]) -> dict[str, Any]:
    return {metric["name"]: metric.get("value") for metric in manifest.get("metrics", [])}


def depth_gate_variant(manifest: dict[str, Any]) -> str | None:
    value = manifest.get("config", {}).get("params", {}).get("depth_gate")
    if value == "adaptive":
        return "adaptive"
    if value == "fixed":
        return "fixed"
    return None


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
        variant = depth_gate_variant(manifest)
        if variant not in args.variant:
            continue
        previous = selected.get(variant)
        if previous is None or manifest.get("created_utc", "") > previous.get("created_utc", ""):
            selected[variant] = manifest
    return selected


def load_failure_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    registry_dir = args.registry_dir if args.registry_dir.is_absolute() else ROOT / args.registry_dir
    failures: list[dict[str, Any]] = []
    for path in sorted(registry_dir.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("benchmark", {}).get("id") != BENCHMARK_ID:
            continue
        if manifest.get("status") not in {"dnf", "failure"}:
            continue
        if manifest.get("dataset", {}).get("sequence") != args.sequence:
            continue
        failures.append(manifest)
    return sorted(failures, key=lambda item: item.get("created_utc", ""))


def expected_variants(args: argparse.Namespace) -> list[str]:
    return list(args.variant)


def missing_expected_runs(args: argparse.Namespace, runs: dict[str, dict[str, Any]]) -> list[str]:
    return [variant for variant in expected_variants(args) if variant not in runs]


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def artifact_exists(manifest: dict[str, Any], kind: str) -> str:
    for artifact in manifest.get("artifacts", []):
        if artifact.get("kind") == kind:
            return "yes" if artifact.get("exists") else "no"
    return ""


def dataset_checksum_lines(
    runs: dict[str, dict[str, Any]],
    failures: list[dict[str, Any]],
) -> list[str]:
    for manifest in [*runs.values(), *failures]:
        dataset = manifest.get("dataset", {})
        checksum = dataset.get("checksum")
        if checksum:
            method = dataset.get("checksum_method") or "checksum"
            return [
                "",
                f"Dataset checksum: `{method} {checksum}`.",
            ]
    return []


def render_failure_rows(failures: list[dict[str, Any]]) -> list[str]:
    if not failures:
        return [
            "",
            "## Recorded Failures",
            "",
            "No DNF/failure manifests are registered for this smoke.",
        ]
    lines = [
        "",
        "## Recorded Failures",
        "",
        "| variant | frames requested | status | failure reason | run id |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for manifest in failures:
        metrics = metric_map(manifest)
        params = manifest.get("config", {}).get("params", {})
        variant = depth_gate_variant(manifest) or str(params.get("depth_gate", ""))
        frames = metrics.get("frames_requested", params.get("max_frames"))
        reason = " ".join(str(manifest.get("failure_reason", "")).split())
        lines.append(
            "| "
            + " | ".join(
                [
                    variant,
                    fmt(frames, 0),
                    str(manifest.get("status", "")),
                    reason,
                    manifest.get("run_id", ""),
                ]
            )
            + " |"
        )
    return lines


def render(
    args: argparse.Namespace,
    runs: dict[str, dict[str, Any]],
    failures: list[dict[str, Any]] | None = None,
) -> str:
    lines = [
        "# KITTI Adaptive Depth Gate Smoke",
        "",
        "Generated from benchmark-registry run manifests. This is a diagnostic",
        "A/B smoke for the rectified-stereo depth gate, not a trajectory benchmark",
        "claim. The run uses a stride-20 KITTI seq00 subset, `--max-frames 2`,",
        "`--frontend deep`, `--deep-max-features 300`, and disables stereo BA.",
    ]
    lines.extend(dataset_checksum_lines(runs, failures or []))
    lines.extend(
        [
            "",
            "| variant | frames | effective min depth m | candidates mean | accepted mean | depth quantile mean m | VO ATE RMSE m | diagnostics artifact | run id |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for variant in expected_variants(args):
        manifest = runs.get(variant)
        if manifest is None:
            lines.append(f"| {variant} |  |  |  |  |  |  | missing | missing |")
            continue
        metrics = metric_map(manifest)
        lines.append(
            "| "
            + " | ".join(
                [
                    variant,
                    fmt(metrics.get("frames"), 0),
                    fmt(metrics.get("effective_min_depth_m_mean"), 3),
                    fmt(metrics.get("candidates_mean"), 1),
                    fmt(metrics.get("accepted_mean"), 1),
                    fmt(metrics.get("depth_quantile_m_mean"), 3),
                    fmt(metrics.get("vo_ate_rmse_m"), 4),
                    artifact_exists(manifest, "depth_gate_diagnostics"),
                    manifest.get("run_id", ""),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation: on this far-field KITTI smoke subset, the adaptive",
            "policy remains at the bounded `3 m` effective lower-depth floor, so it",
            "matches the legacy fixed-3m replay while still recording per-frame",
            "adaptive diagnostics for audit.",
        ]
    )
    lines.extend(render_failure_rows(failures or []))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    runs = load_latest_runs(args)
    failures = load_failure_runs(args)
    text = render(args, runs, failures)
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    missing = missing_expected_runs(args, runs)
    if missing:
        print("missing KITTI adaptive depth-gate smoke run(s):", file=sys.stderr)
        for variant in missing:
            print(f"  variant={variant}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
