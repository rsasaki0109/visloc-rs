#!/usr/bin/env python3
"""Summarize reproducible EuRoC online-SLAM runs into one Markdown table.

Usage:
  python scripts/summarize_visual_slam_evaluation.py \
    --run no_loop=E:/runs/mh01_no_loop \
    --run region=E:/runs/mh01_region \
    --output docs/generated/mh01_region_ab.md

Each run directory must contain ``summary.txt`` and ``slam_trajectory.csv``.
``loop_constraints.csv`` is optional for a no-loop baseline.  Loop precision
uses the demo's recorded 0.5 m / 10 degree ground-truth classification.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class RunSummary:
    label: str
    directory: str
    frames: int
    tracking_coverage: float | None
    longest_continuous_frames: int
    rigid_ate_m: float | None
    similarity_ate_m: float | None
    similarity_scale: float | None
    final_keyframe_rigid_ate_m: float | None
    final_keyframe_similarity_ate_m: float | None
    final_keyframe_similarity_scale: float | None
    rpe_delta1_translation_rmse_m: float | None
    rpe_delta1_rotation_rmse_deg: float | None
    rpe_delta10_translation_rmse_m: float | None
    rpe_delta10_rotation_rmse_deg: float | None
    accepted_loops: int
    evaluated_loops: int
    correct_loops: int
    loop_precision: float | None
    wall_clock_ms_per_frame: float | None
    frame_processing_ms_p95: float | None
    sampled_peak_working_set_mb: float | None


@dataclass
class AggregateSummary:
    sequence: str
    variant: str
    runs: int
    tracking_coverage_mean: float | None
    tracking_coverage_std: float | None
    longest_continuous_frames_mean: float | None
    longest_continuous_frames_std: float | None
    rigid_ate_m_mean: float | None
    rigid_ate_m_std: float | None
    similarity_ate_m_mean: float | None
    similarity_ate_m_std: float | None
    similarity_scale_mean: float | None
    similarity_scale_std: float | None
    final_keyframe_rigid_ate_m_mean: float | None
    final_keyframe_rigid_ate_m_std: float | None
    rpe_delta1_translation_rmse_m_mean: float | None
    rpe_delta1_translation_rmse_m_std: float | None
    rpe_delta1_rotation_rmse_deg_mean: float | None
    rpe_delta1_rotation_rmse_deg_std: float | None
    rpe_delta10_translation_rmse_m_mean: float | None
    rpe_delta10_translation_rmse_m_std: float | None
    rpe_delta10_rotation_rmse_deg_mean: float | None
    rpe_delta10_rotation_rmse_deg_std: float | None
    accepted_loops: int
    evaluated_loops: int
    correct_loops: int
    pooled_loop_precision: float | None
    wall_clock_ms_per_frame_mean: float | None
    wall_clock_ms_per_frame_std: float | None
    frame_processing_ms_p95_mean: float | None
    frame_processing_ms_p95_std: float | None
    sampled_peak_working_set_mb_mean: float | None
    sampled_peak_working_set_mb_std: float | None


@dataclass
class PromotionVerdict:
    sequence: str
    status: str
    candidate_worst_final_keyframe_rigid_ate_m: float | None
    baseline_best_final_keyframe_rigid_ate_m: float | None
    candidate_worst_live_rigid_ate_m: float | None
    baseline_best_live_rigid_ate_m: float | None
    accepted_loops: int
    evaluated_loops: int
    correct_loops: int
    reasons: list[str]


@dataclass(frozen=True)
class RunIdentity:
    sequence: str
    variant: str
    repetition: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        metavar="LABEL=DIR",
        help="Run label and output directory; repeat for an A/B table.",
    )
    parser.add_argument(
        "--matrix-root",
        type=Path,
        help="Load completed run_manifest.json children produced by the A/B runner.",
    )
    parser.add_argument("--output", type=Path, help="Markdown output; stdout when omitted.")
    parser.add_argument("--json", type=Path, help="Optional machine-readable JSON output.")
    args = parser.parse_args()
    if not args.run and args.matrix_root is None:
        parser.error("at least one --run or --matrix-root is required")
    return args


def parse_summary(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.strip().split("=", 1)
        values[key] = value
    return values


def optional_float(value: str | None) -> float | None:
    if value is None or value in {"None", "nan", "NaN"}:
        return None
    if value.startswith("Some(") and value.endswith(")"):
        value = value[5:-1]
    return float(value)


def tracking_continuity(path: Path) -> tuple[int, int]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    longest = current = 0
    for row in rows:
        success = row.get("tracking_success", "0").strip().lower() in {"1", "true"}
        current = current + 1 if success else 0
        longest = max(longest, current)
    return len(rows), longest


def loop_counts(path: Path) -> tuple[int, int, int, float | None]:
    if not path.exists():
        return 0, 0, 0, None
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    evaluated = 0
    correct = 0
    for row in rows:
        value = row.get("gt_correct_0p5m_10deg", "").strip().lower()
        if value not in {"true", "false"}:
            continue
        evaluated += 1
        correct += int(value == "true")
    precision = correct / evaluated if evaluated else None
    return len(rows), evaluated, correct, precision


def load_run(spec: str) -> RunSummary:
    if "=" not in spec:
        raise ValueError(f"--run expects LABEL=DIR, got {spec!r}")
    label, raw_directory = spec.split("=", 1)
    directory = Path(raw_directory)
    values = parse_summary(directory / "summary.txt")
    trajectory_frames, longest = tracking_continuity(directory / "slam_trajectory.csv")
    accepted, evaluated, correct, precision = loop_counts(directory / "loop_constraints.csv")
    manifest_path = directory / "run_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if manifest_path.exists()
        else {}
    )
    peak_working_set_bytes = manifest.get("sampled_peak_working_set_bytes")
    frames = int(values.get("frames_recorded", trajectory_frames))
    return RunSummary(
        label=label,
        directory=str(directory),
        frames=frames,
        tracking_coverage=optional_float(values.get("tracking_success_rate")),
        longest_continuous_frames=longest,
        rigid_ate_m=optional_float(values.get("ate_rigid_rmse_m")),
        similarity_ate_m=optional_float(values.get("ate_similarity_rmse_m")),
        similarity_scale=optional_float(values.get("ate_similarity_scale")),
        final_keyframe_rigid_ate_m=optional_float(
            values.get("final_keyframe_ate_rigid_rmse_m")
        ),
        final_keyframe_similarity_ate_m=optional_float(
            values.get("final_keyframe_ate_similarity_rmse_m")
        ),
        final_keyframe_similarity_scale=optional_float(
            values.get("final_keyframe_ate_similarity_scale")
        ),
        rpe_delta1_translation_rmse_m=optional_float(
            values.get("final_keyframe_rpe_delta1_translation_rmse_m")
        ),
        rpe_delta1_rotation_rmse_deg=optional_float(
            values.get("final_keyframe_rpe_delta1_rotation_rmse_deg")
        ),
        rpe_delta10_translation_rmse_m=optional_float(
            values.get("final_keyframe_rpe_delta10_translation_rmse_m")
        ),
        rpe_delta10_rotation_rmse_deg=optional_float(
            values.get("final_keyframe_rpe_delta10_rotation_rmse_deg")
        ),
        accepted_loops=accepted,
        evaluated_loops=evaluated,
        correct_loops=correct,
        loop_precision=precision,
        wall_clock_ms_per_frame=optional_float(values.get("wall_clock_ms_per_frame")),
        frame_processing_ms_p95=optional_float(values.get("frame_processing_ms_p95")),
        sampled_peak_working_set_mb=(
            float(peak_working_set_bytes) / (1024 * 1024)
            if peak_working_set_bytes is not None
            else None
        ),
    )


def number(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def mean_std(values: list[float | int | None]) -> tuple[float | None, float | None]:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None, None
    return statistics.mean(present), statistics.stdev(present) if len(present) > 1 else 0.0


def validate_matrix_protocol(root: Path) -> tuple[int, int]:
    experiment_path = root / "experiment_manifest.json"
    if not experiment_path.exists():
        raise ValueError(f"matrix is missing {experiment_path}")
    experiment = json.loads(experiment_path.read_text(encoding="utf-8-sig"))
    if experiment.get("schema_version") != 3:
        raise ValueError(
            "matrix must use schema_version=3; older runner protocols did not "
            "prove both a genuinely loop-free control and resource-valid runs"
        )
    protocol = experiment.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("matrix experiment manifest is missing its argument protocol")
    common = protocol.get("common_arguments")
    variants = protocol.get("variant_arguments")
    if not isinstance(common, list) or not isinstance(variants, dict):
        raise ValueError("matrix argument protocol has an invalid shape")
    no_loop = variants.get("no_loop")
    appearance = variants.get("appearance_loop")
    if not isinstance(no_loop, list) or not isinstance(appearance, list):
        raise ValueError("matrix protocol is missing no_loop or appearance_loop arguments")
    if any(str(arg).startswith("--pose-graph-refinement") for arg in common + no_loop):
        raise ValueError("no_loop protocol enables pose-graph refinement")
    required_candidate_switches = {
        "--pose-graph-refinement",
        "--pose-graph-refinement-appearance-loops",
        "--pose-graph-refinement-gnc",
        "--pose-graph-refinement-pcm",
        "--pose-graph-refinement-pcm-pairwise-only",
    }
    missing = required_candidate_switches.difference(map(str, appearance))
    if missing:
        raise ValueError(
            "appearance_loop protocol is missing required switches: "
            + ", ".join(sorted(missing))
        )
    resource_gate = protocol.get("resource_gate")
    if not isinstance(resource_gate, dict):
        raise ValueError("matrix protocol is missing its resource gate")
    minimum_available = resource_gate.get("minimum_available_physical_bytes")
    minimum_commit = resource_gate.get("minimum_commit_headroom_bytes")
    sample_interval = resource_gate.get("sample_interval_seconds")
    if not isinstance(minimum_available, int) or minimum_available <= 0:
        raise ValueError("matrix resource gate has no positive physical-memory reserve")
    if not isinstance(minimum_commit, int) or minimum_commit <= 0:
        raise ValueError("matrix resource gate has no positive commit-headroom reserve")
    if not isinstance(sample_interval, int) or sample_interval < 1:
        raise ValueError("matrix resource gate has an invalid sample interval")
    return minimum_available, minimum_commit


def load_matrix(root: Path) -> tuple[list[RunSummary], dict[str, RunIdentity]]:
    required_available, required_commit = validate_matrix_protocol(root)
    runs: list[RunSummary] = []
    groups: dict[str, RunIdentity] = {}
    identities: set[RunIdentity] = set()
    for manifest_path in sorted(root.glob("*/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if manifest.get("exit_code") != 0:
            continue
        if manifest.get("validation_error") not in {None, ""}:
            raise ValueError(
                f"{manifest_path.parent}: successful run records validation_error="
                f"{manifest.get('validation_error')!r}"
            )
        minimum_available = manifest.get("minimum_available_physical_bytes")
        minimum_commit = manifest.get("minimum_commit_headroom_bytes")
        if not isinstance(minimum_available, int) or minimum_available < required_available:
            raise ValueError(
                f"{manifest_path.parent}: physical-memory minimum does not satisfy "
                "the experiment resource gate"
            )
        if not isinstance(minimum_commit, int) or minimum_commit < required_commit:
            raise ValueError(
                f"{manifest_path.parent}: commit-headroom minimum does not satisfy "
                "the experiment resource gate"
            )
        run_dir = manifest_path.parent
        if not (run_dir / "summary.txt").exists():
            continue
        label = str(manifest["name"])
        identity = RunIdentity(
            sequence=str(manifest["sequence"]),
            variant=str(manifest["variant"]),
            repetition=int(manifest["repetition"]),
        )
        if identity.variant not in {"no_loop", "appearance_loop"}:
            raise ValueError(f"{run_dir}: unsupported matrix variant {identity.variant!r}")
        summary_values = parse_summary(run_dir / "summary.txt")
        expected_pose_graph = "true" if identity.variant == "appearance_loop" else "false"
        actual_pose_graph = summary_values.get("pose_graph_refinement")
        if actual_pose_graph != expected_pose_graph:
            raise ValueError(
                f"{run_dir}: pose_graph_refinement={actual_pose_graph!r}, "
                f"expected {expected_pose_graph!r} for {identity.variant}"
            )
        if identity in identities:
            raise ValueError(f"duplicate completed matrix run identity: {identity}")
        identities.add(identity)
        runs.append(load_run(f"{label}={run_dir}"))
        groups[label] = identity
    if not runs:
        raise ValueError(f"no completed runs found below {root}")
    return runs, groups


def aggregate(
    runs: list[RunSummary], groups: dict[str, RunIdentity]
) -> list[AggregateSummary]:
    grouped: dict[tuple[str, str], list[RunSummary]] = {}
    for run in runs:
        if run.label in groups:
            identity = groups[run.label]
            grouped.setdefault((identity.sequence, identity.variant), []).append(run)

    summaries: list[AggregateSummary] = []
    for (sequence, variant), members in sorted(grouped.items()):
        tracking = mean_std([run.tracking_coverage for run in members])
        longest = mean_std([run.longest_continuous_frames for run in members])
        rigid_ate = mean_std([run.rigid_ate_m for run in members])
        similarity_ate = mean_std([run.similarity_ate_m for run in members])
        similarity_scale = mean_std([run.similarity_scale for run in members])
        final_ate = mean_std([run.final_keyframe_rigid_ate_m for run in members])
        rpe1 = mean_std([run.rpe_delta1_translation_rmse_m for run in members])
        rpe1_rotation = mean_std([run.rpe_delta1_rotation_rmse_deg for run in members])
        rpe10 = mean_std([run.rpe_delta10_translation_rmse_m for run in members])
        rpe10_rotation = mean_std([run.rpe_delta10_rotation_rmse_deg for run in members])
        runtime = mean_std([run.wall_clock_ms_per_frame for run in members])
        p95 = mean_std([run.frame_processing_ms_p95 for run in members])
        memory = mean_std([run.sampled_peak_working_set_mb for run in members])
        evaluated = sum(run.evaluated_loops for run in members)
        correct = sum(run.correct_loops for run in members)
        summaries.append(
            AggregateSummary(
                sequence=sequence,
                variant=variant,
                runs=len(members),
                tracking_coverage_mean=tracking[0],
                tracking_coverage_std=tracking[1],
                longest_continuous_frames_mean=longest[0],
                longest_continuous_frames_std=longest[1],
                rigid_ate_m_mean=rigid_ate[0],
                rigid_ate_m_std=rigid_ate[1],
                similarity_ate_m_mean=similarity_ate[0],
                similarity_ate_m_std=similarity_ate[1],
                similarity_scale_mean=similarity_scale[0],
                similarity_scale_std=similarity_scale[1],
                final_keyframe_rigid_ate_m_mean=final_ate[0],
                final_keyframe_rigid_ate_m_std=final_ate[1],
                rpe_delta1_translation_rmse_m_mean=rpe1[0],
                rpe_delta1_translation_rmse_m_std=rpe1[1],
                rpe_delta1_rotation_rmse_deg_mean=rpe1_rotation[0],
                rpe_delta1_rotation_rmse_deg_std=rpe1_rotation[1],
                rpe_delta10_translation_rmse_m_mean=rpe10[0],
                rpe_delta10_translation_rmse_m_std=rpe10[1],
                rpe_delta10_rotation_rmse_deg_mean=rpe10_rotation[0],
                rpe_delta10_rotation_rmse_deg_std=rpe10_rotation[1],
                accepted_loops=sum(run.accepted_loops for run in members),
                evaluated_loops=evaluated,
                correct_loops=correct,
                pooled_loop_precision=correct / evaluated if evaluated else None,
                wall_clock_ms_per_frame_mean=runtime[0],
                wall_clock_ms_per_frame_std=runtime[1],
                frame_processing_ms_p95_mean=p95[0],
                frame_processing_ms_p95_std=p95[1],
                sampled_peak_working_set_mb_mean=memory[0],
                sampled_peak_working_set_mb_std=memory[1],
            )
        )
    return summaries


def mean_plus_minus(mean: float | None, std: float | None, digits: int) -> str:
    if mean is None or std is None:
        return "n/a"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def aggregate_markdown(groups: list[AggregateSummary]) -> str:
    lines = [
        "\n## Repeated-run aggregate (mean +/- sample standard deviation)\n",
        "| sequence | variant | n | tracking | longest | rigid ATE m | sim ATE m / scale | final KF ATE m | RPE d1 m/deg | RPE d10 m/deg | loops correct/eval | precision | mean ms/frame | p95 ms | peak MB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in groups:
        precision = number(group.pooled_loop_precision, 3)
        lines.append(
            f"| {group.sequence} | {group.variant} | {group.runs} | "
            f"{mean_plus_minus(group.tracking_coverage_mean, group.tracking_coverage_std, 3)} | "
            f"{mean_plus_minus(group.longest_continuous_frames_mean, group.longest_continuous_frames_std, 1)} | "
            f"{mean_plus_minus(group.rigid_ate_m_mean, group.rigid_ate_m_std, 4)} | "
            f"{mean_plus_minus(group.similarity_ate_m_mean, group.similarity_ate_m_std, 4)} / "
            f"{mean_plus_minus(group.similarity_scale_mean, group.similarity_scale_std, 4)} | "
            f"{mean_plus_minus(group.final_keyframe_rigid_ate_m_mean, group.final_keyframe_rigid_ate_m_std, 4)} | "
            f"{mean_plus_minus(group.rpe_delta1_translation_rmse_m_mean, group.rpe_delta1_translation_rmse_m_std, 4)} / "
            f"{mean_plus_minus(group.rpe_delta1_rotation_rmse_deg_mean, group.rpe_delta1_rotation_rmse_deg_std, 2)} | "
            f"{mean_plus_minus(group.rpe_delta10_translation_rmse_m_mean, group.rpe_delta10_translation_rmse_m_std, 4)} / "
            f"{mean_plus_minus(group.rpe_delta10_rotation_rmse_deg_mean, group.rpe_delta10_rotation_rmse_deg_std, 2)} | "
            f"{group.correct_loops}/{group.evaluated_loops} ({group.accepted_loops} accepted) | "
            f"{precision} | "
            f"{mean_plus_minus(group.wall_clock_ms_per_frame_mean, group.wall_clock_ms_per_frame_std, 1)} | "
            f"{mean_plus_minus(group.frame_processing_ms_p95_mean, group.frame_processing_ms_p95_std, 1)} | "
            f"{mean_plus_minus(group.sampled_peak_working_set_mb_mean, group.sampled_peak_working_set_mb_std, 1)} |"
        )
    return "\n".join(lines) + "\n"


def promotion_verdicts(
    runs: list[RunSummary], groups: dict[str, RunIdentity]
) -> list[PromotionVerdict]:
    by_group: dict[tuple[str, str], list[RunSummary]] = {}
    for run in runs:
        if run.label in groups:
            identity = groups[run.label]
            by_group.setdefault((identity.sequence, identity.variant), []).append(run)

    sequences = sorted({sequence for sequence, _ in by_group})
    verdicts: list[PromotionVerdict] = []
    for sequence in sequences:
        baseline = by_group.get((sequence, "no_loop"), [])
        candidate = by_group.get((sequence, "appearance_loop"), [])
        reasons: list[str] = []
        if not baseline or not candidate:
            reasons.append("missing baseline or candidate runs")
        if len(baseline) != len(candidate):
            reasons.append("unbalanced repetition count")
        if len(baseline) < 3 or len(candidate) < 3:
            reasons.append("fewer than 3 paired repetitions")
        baseline_repetitions = {
            groups[run.label].repetition for run in baseline if run.label in groups
        }
        candidate_repetitions = {
            groups[run.label].repetition for run in candidate if run.label in groups
        }
        if baseline_repetitions != candidate_repetitions:
            reasons.append("baseline and candidate repetition IDs are not paired")

        baseline_final_ates = [
            run.final_keyframe_rigid_ate_m
            for run in baseline
            if run.final_keyframe_rigid_ate_m is not None
        ]
        candidate_final_ates = [
            run.final_keyframe_rigid_ate_m
            for run in candidate
            if run.final_keyframe_rigid_ate_m is not None
        ]
        baseline_best_final = (
            min(baseline_final_ates)
            if len(baseline_final_ates) == len(baseline) and baseline
            else None
        )
        candidate_worst_final = (
            max(candidate_final_ates)
            if len(candidate_final_ates) == len(candidate) and candidate
            else None
        )
        if baseline_best_final is None or candidate_worst_final is None:
            reasons.append("missing final-keyframe rigid ATE")
        elif candidate_worst_final >= baseline_best_final:
            reasons.append(
                "candidate worst final-keyframe rigid ATE does not beat baseline best"
            )

        # Live per-frame CSVs are causal and cannot be retroactively corrected
        # when a late loop closes. Use them only as a regression guard; the
        # final optimized keyframe trajectory above is the primary loop-SLAM
        # accuracy criterion.
        baseline_live_ates = [
            run.rigid_ate_m for run in baseline if run.rigid_ate_m is not None
        ]
        candidate_live_ates = [
            run.rigid_ate_m for run in candidate if run.rigid_ate_m is not None
        ]
        baseline_best_live = (
            min(baseline_live_ates)
            if len(baseline_live_ates) == len(baseline) and baseline
            else None
        )
        candidate_worst_live = (
            max(candidate_live_ates)
            if len(candidate_live_ates) == len(candidate) and candidate
            else None
        )
        if baseline_best_live is None or candidate_worst_live is None:
            reasons.append("missing live rigid ATE")
        elif candidate_worst_live > baseline_best_live * 1.01:
            reasons.append("live rigid ATE regresses by more than 1%")

        accepted = sum(run.accepted_loops for run in candidate)
        evaluated = sum(run.evaluated_loops for run in candidate)
        correct = sum(run.correct_loops for run in candidate)
        if accepted == 0:
            reasons.append("zero accepted loops")
        if evaluated != accepted:
            reasons.append("not every accepted loop has ground-truth evaluation")
        if correct != accepted:
            reasons.append("accepted-loop precision is below 100%")

        baseline_tracking = [
            run.tracking_coverage for run in baseline if run.tracking_coverage is not None
        ]
        candidate_tracking = [
            run.tracking_coverage for run in candidate if run.tracking_coverage is not None
        ]
        if len(baseline_tracking) != len(baseline) or len(candidate_tracking) != len(candidate):
            reasons.append("missing tracking coverage")
        elif baseline and candidate and min(candidate_tracking) < min(baseline_tracking) - 0.005:
            reasons.append("tracking coverage regresses by more than 0.005")

        if baseline and candidate:
            baseline_longest = min(run.longest_continuous_frames for run in baseline)
            candidate_longest = min(run.longest_continuous_frames for run in candidate)
            if candidate_longest < 0.99 * baseline_longest:
                reasons.append("longest continuous run regresses by more than 1%")

        if any(run.wall_clock_ms_per_frame is None for run in baseline + candidate):
            reasons.append("missing runtime measurement")
        elif baseline and candidate:
            baseline_best_runtime = min(
                run.wall_clock_ms_per_frame for run in baseline
            )
            candidate_worst_runtime = max(
                run.wall_clock_ms_per_frame for run in candidate
            )
            if candidate_worst_runtime > baseline_best_runtime * 1.25:
                reasons.append("runtime regresses by more than 25%")
        if any(run.sampled_peak_working_set_mb is None for run in baseline + candidate):
            reasons.append("missing peak working-set measurement")
        elif baseline and candidate:
            baseline_best_memory = min(
                run.sampled_peak_working_set_mb for run in baseline
            )
            candidate_worst_memory = max(
                run.sampled_peak_working_set_mb for run in candidate
            )
            if candidate_worst_memory > baseline_best_memory * 1.25:
                reasons.append("peak working-set regresses by more than 25%")

        incomplete_markers = {
            "missing baseline or candidate runs",
            "unbalanced repetition count",
            "fewer than 3 paired repetitions",
            "baseline and candidate repetition IDs are not paired",
            "missing final-keyframe rigid ATE",
            "missing live rigid ATE",
            "missing tracking coverage",
            "missing runtime measurement",
            "missing peak working-set measurement",
        }
        status = "PROMOTE" if not reasons else (
            "INCOMPLETE" if any(reason in incomplete_markers for reason in reasons) else "REJECT"
        )
        verdicts.append(
            PromotionVerdict(
                sequence=sequence,
                status=status,
                candidate_worst_final_keyframe_rigid_ate_m=candidate_worst_final,
                baseline_best_final_keyframe_rigid_ate_m=baseline_best_final,
                candidate_worst_live_rigid_ate_m=candidate_worst_live,
                baseline_best_live_rigid_ate_m=baseline_best_live,
                accepted_loops=accepted,
                evaluated_loops=evaluated,
                correct_loops=correct,
                reasons=reasons,
            )
        )
    return verdicts


def promotion_markdown(verdicts: list[PromotionVerdict]) -> str:
    lines = [
        "\n## Conservative promotion gate\n",
        "| sequence | verdict | candidate worst final-KF rigid ATE m | baseline best final-KF rigid ATE m | candidate worst live rigid ATE m | baseline best live rigid ATE m | loops correct/eval/accepted | reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for verdict in verdicts:
        reason = "; ".join(verdict.reasons) if verdict.reasons else "all gates passed"
        lines.append(
            f"| {verdict.sequence} | {verdict.status} | "
            f"{number(verdict.candidate_worst_final_keyframe_rigid_ate_m)} | "
            f"{number(verdict.baseline_best_final_keyframe_rigid_ate_m)} | "
            f"{number(verdict.candidate_worst_live_rigid_ate_m)} | "
            f"{number(verdict.baseline_best_live_rigid_ate_m)} | "
            f"{verdict.correct_loops}/{verdict.evaluated_loops}/{verdict.accepted_loops} | "
            f"{reason} |"
        )
    return "\n".join(lines) + "\n"


def markdown(runs: list[RunSummary]) -> str:
    lines = [
        "| run | frames | tracking | longest | rigid ATE m | sim ATE m / scale | final KF ATE m | "
        "RPE d1 m/deg | RPE d10 m/deg | loops correct/eval | precision | mean ms/frame | p95 ms | peak MB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        tracking = "n/a" if run.tracking_coverage is None else f"{run.tracking_coverage:.3f}"
        precision = "n/a" if run.loop_precision is None else f"{run.loop_precision:.3f}"
        lines.append(
            f"| {run.label} | {run.frames} | {tracking} | {run.longest_continuous_frames} | "
            f"{number(run.rigid_ate_m)} | {number(run.similarity_ate_m)}/{number(run.similarity_scale, 4)} | "
            f"{number(run.final_keyframe_rigid_ate_m)} | "
            f"{number(run.rpe_delta1_translation_rmse_m)}/{number(run.rpe_delta1_rotation_rmse_deg, 2)} | "
            f"{number(run.rpe_delta10_translation_rmse_m)}/{number(run.rpe_delta10_rotation_rmse_deg, 2)} | "
            f"{run.correct_loops}/{run.evaluated_loops} ({run.accepted_loops} accepted) | "
            f"{precision} | {number(run.wall_clock_ms_per_frame, 1)} | "
            f"{number(run.frame_processing_ms_p95, 1)} | "
            f"{number(run.sampled_peak_working_set_mb, 1)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    runs = [load_run(spec) for spec in (args.run or [])]
    matrix_groups: dict[str, RunIdentity] = {}
    if args.matrix_root is not None:
        matrix_runs, matrix_groups = load_matrix(args.matrix_root)
        runs.extend(matrix_runs)
    aggregates = aggregate(runs, matrix_groups)
    verdicts = promotion_verdicts(runs, matrix_groups)
    output = markdown(runs)
    if aggregates:
        output += aggregate_markdown(aggregates)
    if verdicts:
        output += promotion_markdown(verdicts)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(output, end="")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "runs": [asdict(run) for run in runs],
                    "aggregates": [asdict(group) for group in aggregates],
                    "promotion_verdicts": [asdict(verdict) for verdict in verdicts],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
