#!/usr/bin/env python3
"""Summarize the counterbalanced full-sequence tracking-cliff matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path


RUN_RE = re.compile(
    r"^(MH_01_easy|MH_03_medium|MH_05_difficult)_(control|candidate)_r(\d{2})$"
)
SEQUENCES = ("MH_01_easy", "MH_03_medium", "MH_05_difficult")
VARIANTS = ("control", "candidate")
HASH_FILES = (
    "slam_trajectory.csv",
    "keyframe_trajectory.csv",
    "tracking_diagnostics.csv",
    "final_keyframe_errors.csv",
    "loop_constraints.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--markdown-out", required=True, type=Path)
    return parser.parse_args()


def read_summary(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def option_float(value: str) -> float | None:
    if value in ("None", "", None):
        return None
    match = re.fullmatch(r"Some\(([-+0-9.eE]+)\)", value)
    return float(match.group(1) if match else value)


def longest_continuity(path: Path) -> tuple[int, int]:
    longest = current = segments = 0
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["success"] == "1":
                if current == 0:
                    segments += 1
                current += 1
                longest = max(longest, current)
            else:
                current = 0
    return longest, segments


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_run(run_dir: Path, sequence: str, variant: str, repetition: int) -> dict:
    summary = read_summary(run_dir / "summary.txt")
    longest, segments = longest_continuity(run_dir / "tracking_diagnostics.csv")
    triggers = int(summary["local_vi_ba_triggers"])
    rejected = int(summary["local_vi_ba_quality_gate_rejections"])
    return {
        "sequence": sequence,
        "variant": variant,
        "repetition": repetition,
        "directory": str(run_dir),
        "frames": int(summary["frames_recorded"]),
        "tracking": float(summary["tracking_success_rate"]),
        "longest_continuity": longest,
        "tracking_segments": segments,
        "rigid_ate_m": float(summary["ate_rigid_rmse_m"]),
        "final_rigid_ate_m": float(summary["final_keyframe_ate_rigid_rmse_m"]),
        "rpe_d1_translation_m": option_float(
            summary["final_keyframe_rpe_delta1_translation_rmse_m"]
        ),
        "rpe_d1_rotation_deg": option_float(
            summary["final_keyframe_rpe_delta1_rotation_rmse_deg"]
        ),
        "rpe_d10_translation_m": option_float(
            summary["final_keyframe_rpe_delta10_translation_rmse_m"]
        ),
        "rpe_d10_rotation_deg": option_float(
            summary["final_keyframe_rpe_delta10_rotation_rmse_deg"]
        ),
        "loop_precision": option_float(
            summary["pose_graph_refinement_gt_precision_0p5m_10deg"]
        ),
        "runtime_s": float(summary["wall_clock_seconds"]),
        "visual_override_count": int(summary["pose_prior_visual_override_count"]),
        "motion_vi_initialized": summary["motion_vi_init_succeeded_frame"] != "None",
        "motion_vi_init_frame": summary["motion_vi_init_succeeded_frame"],
        "motion_vi_scale": option_float(summary["motion_vi_init_recovered_scale"]),
        "covis_ba_triggers": int(summary["covisibility_local_ba_triggers"]),
        "covis_ba_successes": int(summary["covisibility_local_ba_successes"]),
        "local_vi_triggers": triggers,
        "local_vi_accepted": triggers - rejected,
        "local_vi_rejected": rejected,
        "local_vi_imu_nis_rejected": int(summary["local_vi_ba_imu_nis_gate_rejections"]),
        "local_vi_pose_rejected": int(summary["local_vi_ba_pose_correction_gate_rejections"]),
        "local_vi_marginalization_successes": int(
            summary["local_vi_ba_marginalization_successes"]
        ),
        "local_vi_mirrors": int(summary["local_vi_ba_mirrors_into_imu_motion_model"]),
        "hashes": {name: sha256(run_dir / name) for name in HASH_FILES},
    }


def median_summary(runs: list[dict]) -> dict:
    fields = (
        "tracking",
        "longest_continuity",
        "tracking_segments",
        "rigid_ate_m",
        "final_rigid_ate_m",
        "rpe_d1_translation_m",
        "rpe_d1_rotation_deg",
        "rpe_d10_translation_m",
        "rpe_d10_rotation_deg",
        "loop_precision",
        "runtime_s",
    )
    result = {}
    for field in fields:
        values = [run[field] for run in runs]
        result[field] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    result["repeat_hashes_identical"] = all(
        len({run["hashes"][name] for run in runs}) == 1 for name in HASH_FILES
    )
    return result


def relative_delta(candidate: float, control: float) -> float:
    return (candidate / control - 1.0) * 100.0


def main() -> None:
    args = parse_args()
    manifest = json.loads(
        (args.root / "tracking_cliff_tight_vi_manifest.json").read_text(encoding="utf-8-sig")
    )
    runs = []
    for child in args.root.iterdir():
        match = RUN_RE.match(child.name)
        if match and (child / "summary.txt").is_file():
            runs.append(load_run(child, match.group(1), match.group(2), int(match.group(3))))
    runs.sort(key=lambda run: (run["sequence"], run["variant"], run["repetition"]))
    if len(runs) != 18:
        raise SystemExit(f"expected 18 complete runs, found {len(runs)}")

    grouped = {}
    comparisons = {}
    gate_tolerance = 1.02
    for sequence in SEQUENCES:
        grouped[sequence] = {}
        for variant in VARIANTS:
            selected = [r for r in runs if r["sequence"] == sequence and r["variant"] == variant]
            if len(selected) != 3:
                raise SystemExit(f"expected 3 runs for {sequence}/{variant}")
            grouped[sequence][variant] = median_summary(selected)
        control = grouped[sequence]["control"]
        candidate = grouped[sequence]["candidate"]
        lower_is_better = (
            "rigid_ate_m",
            "final_rigid_ate_m",
            "rpe_d1_translation_m",
            "rpe_d1_rotation_deg",
            "rpe_d10_translation_m",
            "rpe_d10_rotation_deg",
        )
        checks = {
            "tracking_non_regression": candidate["tracking"]["median"] >= control["tracking"]["median"],
            "continuity_non_regression": candidate["longest_continuity"]["median"]
            >= control["longest_continuity"]["median"],
            "loop_precision_non_regression": candidate["loop_precision"]["median"]
            >= control["loop_precision"]["median"],
        }
        for field in lower_is_better:
            checks[f"{field}_within_2pct"] = candidate[field]["median"] <= (
                control[field]["median"] * gate_tolerance
            )
        comparisons[sequence] = {
            "checks": checks,
            "gate_pass": all(checks.values()),
            "deltas_percent": {
                field: relative_delta(candidate[field]["median"], control[field]["median"])
                for field in ("tracking", "longest_continuity", *lower_is_better, "runtime_s")
            },
        }

    cliff_improved = any(
        comparisons[sequence]["deltas_percent"]["tracking"] > 0
        and comparisons[sequence]["deltas_percent"]["longest_continuity"] > 0
        for sequence in ("MH_03_medium", "MH_05_difficult")
    )
    report = {
        "schema_version": 1,
        "root": str(args.root),
        "manifest": str(args.root / "tracking_cliff_tight_vi_manifest.json"),
        "artifact_hashes": {
            "executable": manifest["executable_sha256"],
            "model": manifest["superpoint_model_sha256"],
            "onnx_runtime": manifest["ort_dylib_sha256"],
        },
        "run_count": len(runs),
        "gate_policy": {
            "accuracy_relative_tolerance": 0.02,
            "tracking_and_continuity_tolerance": 0.0,
            "runtime": "report-only",
            "requires_cliff_improvement": True,
        },
        "groups": grouped,
        "comparisons": comparisons,
        "cliff_improved": cliff_improved,
        "overall_gate_pass": all(value["gate_pass"] for value in comparisons.values())
        and cliff_improved,
        "runs": runs,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Tracking-cliff tight-VI full-sequence matrix",
        "",
        f"Runs: {len(runs)} (3 sequences × 2 variants × 3 counterbalanced repetitions).",
        f"Overall gate: **{'PASS' if report['overall_gate_pass'] else 'FAIL'}**.",
        "",
        "| Sequence | Variant | Tracking | Longest | Rigid ATE m | d1 m / deg | d10 m / deg | Loop precision | Runtime s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sequence in SEQUENCES:
        for variant in VARIANTS:
            data = grouped[sequence][variant]
            lines.append(
                f"| {sequence} | {variant} | {data['tracking']['median']:.3f} | "
                f"{data['longest_continuity']['median']:.0f} | {data['rigid_ate_m']['median']:.4f} | "
                f"{data['rpe_d1_translation_m']['median']:.4f} / {data['rpe_d1_rotation_deg']['median']:.4f} | "
                f"{data['rpe_d10_translation_m']['median']:.4f} / {data['rpe_d10_rotation_deg']['median']:.4f} | "
                f"{data['loop_precision']['median']:.3f} | {data['runtime_s']['median']:.1f} "
                f"[{data['runtime_s']['min']:.1f}, {data['runtime_s']['max']:.1f}] |"
            )
    lines.extend(["", "## Candidate delta from control", ""])
    for sequence in SEQUENCES:
        delta = comparisons[sequence]["deltas_percent"]
        lines.append(
            f"- {sequence}: tracking {delta['tracking']:+.2f}%, longest {delta['longest_continuity']:+.2f}%, "
            f"ATE {delta['rigid_ate_m']:+.2f}%, d1 trans/rot "
            f"{delta['rpe_d1_translation_m']:+.2f}%/{delta['rpe_d1_rotation_deg']:+.2f}%, "
            f"d10 trans/rot {delta['rpe_d10_translation_m']:+.2f}%/"
            f"{delta['rpe_d10_rotation_deg']:+.2f}%."
        )
    lines.extend(
        [
            "",
            "Gate: no tracking/continuity/loop-precision regression; each ATE/RPE metric within 2%; "
            "runtime is report-only; at least one cliff sequence must improve tracking and longest continuity.",
            "",
        ]
    )
    args.markdown_out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
