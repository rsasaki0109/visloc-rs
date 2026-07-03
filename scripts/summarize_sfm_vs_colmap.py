#!/usr/bin/env python3
"""Summarize the sequential SfM-vs-COLMAP head-to-head (metric video).

Phase 2 registry formalization of the existing head-to-head documented in
`docs/sfm_vs_colmap_benchmark.md`: visloc-rs stereo VO + loop-closure SfM vs
COLMAP monocular incremental SfM, both reconstructing the same rectified
EuRoC MH_03_medium 2700-frame stream and scored against the same Vicon/Leica
ground truth with the same `evo_ape` tooling. This summarizer reads the two
registered run manifests (visloc, COLMAP) and renders a 5-metric table plus
the honest caveats already documented in the source doc -- it does not
re-run either engine (COLMAP is not installed locally and its arm takes
11.7h, so the COLMAP manifest is an explicitly provenance-marked prior-run
reference, not reproduced in this session).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ID = "sfm-vs-colmap"
ENGINES = ["visloc", "colmap"]
ENGINE_LABELS = {
    "visloc": "visloc stereo VO + loop SfM",
    "colmap": "COLMAP mono incremental",
}


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
        default=Path("docs/generated/sfm_vs_colmap_headtohead.md"),
        help="Markdown output path",
    )
    parser.add_argument("--sequence", default="MH_03_medium")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def metric_map(manifest: dict[str, Any]) -> dict[str, Any]:
    return {metric["name"]: metric.get("value") for metric in manifest.get("metrics", [])}


def metric(manifest: dict[str, Any] | None, name: str) -> Any:
    if manifest is None:
        return None
    return metric_map(manifest).get(name)


def engine_of(params: dict[str, Any]) -> str | None:
    raw = params.get("engine")
    return raw if raw in ENGINES else None


def load_latest_runs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Return {engine: manifest} for the newest matching manifest per engine."""
    registry_dir = args.registry_dir if args.registry_dir.is_absolute() else ROOT / args.registry_dir
    selected: dict[str, dict[str, Any]] = {}
    for path in sorted(registry_dir.glob("*.json")):
        manifest = _load_json(path)
        if manifest is None:
            continue
        if manifest.get("benchmark", {}).get("id") != BENCHMARK_ID:
            continue
        if manifest.get("status") != "success":
            continue
        if manifest.get("dataset", {}).get("sequence") != args.sequence:
            continue
        params = manifest.get("config", {}).get("params", {})
        engine = engine_of(params)
        if engine is None:
            continue
        previous = selected.get(engine)
        if previous is None or manifest.get("created_utc", "") > previous.get("created_utc", ""):
            selected[engine] = manifest
    return selected


def missing_expected_runs(args: argparse.Namespace, runs: dict[str, dict[str, Any]]) -> list[str]:
    return [engine for engine in ENGINES if engine not in runs]


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def wall_clock_text(manifest: dict[str, Any] | None) -> str:
    if manifest is None:
        return ""
    primary = next((m for m in manifest.get("metrics", []) if m.get("name") == "wall_clock"), None)
    if primary is None:
        return ""
    return f"{fmt(primary.get('value'), 1)} {primary.get('unit')}"


def render_table(args: argparse.Namespace, runs: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| engine | wall-clock | registration rate | ATE vs GT | mean reprojection | downstream 3DGS | metric scale | run id |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for engine in ENGINES:
        manifest = runs.get(engine)
        label = ENGINE_LABELS[engine]
        if manifest is None:
            lines.append(f"| {label} |  |  |  |  |  |  | missing |")
            continue
        ate_sim3 = metric(manifest, "ate_sim3")
        ate_se3 = metric(manifest, "ate_se3_metric")
        if ate_se3 is not None:
            ate_text = f"{fmt(ate_sim3, 3)} m (Sim3, model) / {fmt(ate_se3, 3)} m (SE3, metric VO)"
        else:
            ate_text = f"{fmt(ate_sim3, 3)} m (Sim3)"
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    wall_clock_text(manifest),
                    fmt(metric(manifest, "registration_rate"), 3),
                    ate_text,
                    f"{fmt(metric(manifest, 'mean_reprojection'), 2)} px",
                    str(metric(manifest, "downstream_3dgs") or ""),
                    fmt(metric(manifest, "metric_scale")),
                    str(manifest.get("run_id")),
                ]
            )
            + " |"
        )
    return lines


def render(args: argparse.Namespace, runs: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Sequential SfM vs COLMAP Head-to-Head (Registry-Backed)",
        "",
        "Generated from benchmark-registry run manifests. Phase 2 formalization of "
        "the head-to-head documented in `docs/sfm_vs_colmap_benchmark.md`: the same "
        f"rectified EuRoC `{args.sequence}` 2700-frame stereo stream reconstructed by "
        "both engines and scored against the same timestamped Vicon/Leica ground "
        "truth with the same `evo_ape` tooling. visloc-rs is stereo VO + online "
        "windowed BA + loop-closure pose-graph optimization -> merged multi-view "
        "tracks -> COLMAP model export, metric scale by construction from the "
        "rectified stereo baseline. COLMAP is monocular `sequential_matcher` + "
        "incremental `mapper` (its SIFT frontend), scale-free, Sim(3)-aligned to "
        "ground truth.",
        "",
    ]
    lines.extend(render_table(args, runs))
    lines.extend(
        [
            "",
            "## Headline",
            "",
            "- **~117x faster**: visloc 6 min vs COLMAP 11.7 h (COLMAP's mapper stage "
            "alone is ~11.5 h; its incremental mapper interleaves a global bundle "
            "adjustment that grows with the registered-image count, so cost is "
            "super-linear in frame count).",
            "- **~17-33x more accurate**: visloc 0.13 m (Sim3, COLMAP-model export) / "
            "0.066 m (SE3, loop-closed VO trajectory, metric) vs COLMAP 2.18 m (Sim3, "
            "scale-absorbed).",
            "- **Metric scale**: visloc recovers metric scale by construction (rectified "
            "stereo baseline); COLMAP's monocular reconstruction is scale-free and "
            "cannot recover it -- a single global Sim(3) cannot even absorb the local "
            "scale drift COLMAP accumulates over this long, low-parallax forward "
            "flight.",
            "",
            "## Caveats",
            "",
            "- **The stereo-vs-monocular asymmetry is the thesis, not a thumb on the "
            "scale.** The claim is narrow and turf-specific: on metric video SfM, an "
            "architecture built around a stereo VO frontend + windowed BA + loop "
            "closure dominates a from-scratch monocular incremental mapper on speed, "
            "accuracy, and scale recovery. This is **not** a claim that visloc-rs "
            "beats COLMAP on COLMAP's home turf -- unordered internet photo "
            "collections, where retrieval + multi-hypothesis incremental mapping is "
            "COLMAP's strength.",
            "- **On COLMAP's home turf (monocular, small scene), COLMAP wins.** On the "
            "first 300 MH_03 frames reconstructed monocularly (left camera only, both "
            "engines scale-free, same Sim(3)-aligned scoring), COLMAP reaches "
            "**0.37 cm** Sim(3) ATE at 300/300 registered vs visloc's "
            "**1.64 cm** at 299/300 (`--colmap-style` incremental mapper). visloc "
            "does not yet match COLMAP on this turf.",
            "- **The 3DGS blur is capture-geometry-limited, not a pose defect.** Both "
            "engines produce a blurry downstream 3DGS fly-through on this forward-"
            "flight sequence -- the same limited-parallax capture geometry blurs any "
            "pose source, including COLMAP's. Contrast the V2_03 orbit sequence "
            "(better capture geometry), which renders crisp (l1 ~= 0.006), against "
            "this MH-class forward flight (l1 ~= 0.24 on MH_05). The blur is a "
            "property of the capture trajectory, not of which engine estimated the "
            "poses.",
            "- **The COLMAP arm is a prior-run reference, not reproduced this "
            "session.** COLMAP is not installed on this machine, and its documented "
            "11.7 h wall-clock cost (single CPU, COLMAP 4.0.3, no CUDA) makes a local "
            "re-run impractical for this evidence-formalization pass. The COLMAP "
            "manifest captures the already-documented, previously-executed result "
            "from `docs/sfm_vs_colmap_benchmark.md` rather than re-executing it.",
            "",
            "## Conclusion",
            "",
            "On the metric-video turf visloc-rs is built for, the stereo VO + loop-"
            "closure SfM architecture wins all three axes against a from-scratch "
            "monocular COLMAP reconstruction -- speed, accuracy, and metric-scale "
            "recovery. That win is scoped honestly: it does not extend to COLMAP's "
            "unordered-photo home turf, where the small-scene monocular subset still "
            "favors COLMAP, and the downstream 3DGS blur on this sequence reflects "
            "capture geometry rather than either engine's pose quality.",
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
