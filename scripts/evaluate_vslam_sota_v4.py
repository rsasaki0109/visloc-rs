#!/usr/bin/env python3
"""Evaluate a failure-inclusive frozen EuRoC V4 SOTA matrix.

The evaluator deliberately separates the local EuRoC engineering target from
the public-frontier claim. Missing, failed, malformed, or out-of-policy runs
remain in the denominator and make the claim gate false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def parse_summary(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.strip().split("=", 1)
            values[key] = value
    return values


def finite_float(values: dict[str, str], key: str) -> float:
    try:
        value = float(values[key])
    except (KeyError, ValueError) as error:
        raise ValueError(f"summary has no numeric {key}") from error
    if not math.isfinite(value):
        raise ValueError(f"summary has non-finite {key}")
    return value


def nonnegative_int(values: dict[str, str], key: str) -> int:
    value = values.get(key)
    if value is None or not value.isascii() or not value.isdecimal():
        raise ValueError(f"summary has no non-negative integer {key}")
    return int(value)


def valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise ValueError("V4 protocol must have schema_version=1")
    sequences = protocol.get("sequences")
    if not isinstance(sequences, list) or len(sequences) != 11 or len(set(sequences)) != 11:
        raise ValueError("V4 protocol must contain exactly 11 unique EuRoC sequences")
    frame_counts = protocol.get("full_sequence_frame_counts")
    if not isinstance(frame_counts, dict) or set(frame_counts) != set(sequences):
        raise ValueError("V4 protocol frame counts do not match its sequences")
    if any(not isinstance(value, int) or value < 1 for value in frame_counts.values()):
        raise ValueError("V4 protocol frame counts must be positive integers")
    if protocol.get("repetitions") != 3:
        raise ValueError("V4 protocol requires exactly three repetitions")
    gates = protocol.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("V4 protocol has no gates")
    for key in (
        "mean_sequence_sim3_ate_rmse_m_max",
        "min_tracked_fraction",
        "max_committed_abs_log_scale",
        "max_ms_per_frame_total",
        "max_sampled_peak_working_set_bytes",
        "max_sampled_peak_gpu_memory_bytes",
    ):
        value = gates.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"V4 protocol has invalid gate {key}")
    bounds = gates.get("queue_bounds")
    if not isinstance(bounds, dict):
        raise ValueError("V4 protocol has no queue bounds")
    for key in ("inactive_edge_cap", "max_free_poses", "long_loop_max_indexed_frames"):
        if not isinstance(bounds.get(key), int) or bounds[key] < 1:
            raise ValueError(f"V4 protocol has invalid queue bound {key}")


def public_frontier_verdict(path: Path | None) -> tuple[bool, list[str]]:
    if path is None:
        return False, ["no public benchmark evidence was supplied"]
    evidence = load_json(path)
    reasons: list[str] = []
    if evidence.get("schema_version") != 1:
        reasons.append("public evidence schema_version is not 1")
    if evidence.get("status") != "verified_public_frontier":
        reasons.append("public evidence is not marked verified_public_frontier")
    if evidence.get("benchmark") not in {"ETH3D SLAM", "ORBIT"}:
        reasons.append("public evidence benchmark is not ETH3D SLAM or ORBIT")
    result_url = evidence.get("result_url")
    if not isinstance(result_url, str) or not result_url.startswith("https://"):
        reasons.append("public evidence has no HTTPS result URL")
    if not valid_digest(evidence.get("released_artifact_sha256")):
        reasons.append("public evidence has no released-artifact SHA-256")
    return not reasons, reasons


def evaluate(
    matrix_root: Path, protocol_path: Path, public_evidence: Path | None
) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    validate_protocol(protocol)
    experiment_path = matrix_root / "experiment_manifest.json"
    experiment = load_json(experiment_path)
    if experiment.get("schema_version") != 1:
        raise ValueError("V4 experiment manifest must have schema_version=1")
    if experiment.get("protocol_sha256", "").lower() != sha256(protocol_path):
        raise ValueError("experiment protocol SHA-256 does not match the frozen protocol")
    for key in (
        "executable_sha256",
        "model_bundle_sha256",
        "configuration_sha256",
        "ort_dylib_sha256",
    ):
        if not valid_digest(experiment.get(key)):
            raise ValueError(f"experiment has invalid {key}")

    sequences: list[str] = protocol["sequences"]
    repetitions: int = protocol["repetitions"]
    expected = {(sequence, repetition) for sequence in sequences for repetition in range(1, repetitions + 1)}
    manifests: dict[tuple[str, int], Path] = {}
    for manifest_path in sorted(matrix_root.glob("*/run_manifest.json")):
        manifest = load_json(manifest_path)
        identity = (manifest.get("sequence"), manifest.get("repetition"))
        if identity not in expected:
            raise ValueError(f"{manifest_path}: unexpected run identity {identity!r}")
        if identity in manifests:
            raise ValueError(f"duplicate run identity {identity!r}")
        manifests[identity] = manifest_path

    gates = protocol["gates"]
    queue_bounds = gates["queue_bounds"]
    run_rows: list[dict[str, Any]] = []
    successful_ates: dict[str, list[float]] = {sequence: [] for sequence in sequences}
    max_scale = 0.0
    scale_rejections = 0
    for sequence, repetition in sorted(expected):
        row: dict[str, Any] = {
            "sequence": sequence,
            "repetition": repetition,
            "success": False,
            "reasons": [],
        }
        manifest_path = manifests.get((sequence, repetition))
        if manifest_path is None:
            row["reasons"].append("missing run manifest")
            run_rows.append(row)
            continue
        manifest = load_json(manifest_path)
        if manifest.get("exit_code") != 0:
            row["reasons"].append(f"exit_code={manifest.get('exit_code')!r}")
            run_rows.append(row)
            continue
        if str(manifest.get("protocol_sha256", "")).lower() != sha256(protocol_path):
            row["reasons"].append("protocol_sha256 differs from frozen protocol")
        for key in (
            "executable_sha256",
            "model_bundle_sha256",
            "configuration_sha256",
            "ort_dylib_sha256",
        ):
            if str(manifest.get(key, "")).lower() != str(experiment[key]).lower():
                row["reasons"].append(f"{key} differs from experiment")
        summary_path = manifest_path.parent / "summary.txt"
        expected_summary_hash = manifest.get("summary_sha256")
        if not summary_path.is_file() or not valid_digest(expected_summary_hash):
            row["reasons"].append("missing summary or summary SHA-256")
            run_rows.append(row)
            continue
        if sha256(summary_path) != str(expected_summary_hash).lower():
            raise ValueError(f"{summary_path}: SHA-256 differs from run manifest")
        values = parse_summary(summary_path)
        try:
            ate = finite_float(values, "ate_similarity_rmse_m")
            tracked_fraction = finite_float(values, "tracked_fraction")
            elapsed_ms = finite_float(values, "ms_per_frame_total")
            configured_scale_gate = finite_float(
                values, "sim3_backend_max_abs_log_scale_correction"
            )
            committed_scale = finite_float(
                values, "sim3_backend_max_committed_abs_log_scale"
            )
            rejections = nonnegative_int(
                values, "sim3_backend_scale_jump_rejections_total"
            )
            frames = nonnegative_int(values, "frames_requested")
            inactive_edges = nonnegative_int(
                values, "global_ba_inactive_edges_retained"
            )
            max_free_poses = nonnegative_int(values, "global_ba_max_free_pose_count")
            indexed_frames = nonnegative_int(values, "long_loop_frames_indexed")
        except ValueError as error:
            row["reasons"].append(str(error))
            run_rows.append(row)
            continue

        manifest_bounds = manifest.get("queue_bounds")
        if manifest_bounds != queue_bounds:
            row["reasons"].append("run queue bounds differ from frozen protocol")
        if frames != protocol["full_sequence_frame_counts"][sequence]:
            row["reasons"].append("run did not consume the frozen full sequence")
        if tracked_fraction < gates["min_tracked_fraction"]:
            row["reasons"].append("tracking coverage is below the frozen minimum")
        if elapsed_ms > gates["max_ms_per_frame_total"]:
            row["reasons"].append("input-rate budget exceeded")
        peak_memory = manifest.get("sampled_peak_working_set_bytes")
        if (
            not isinstance(peak_memory, int)
            or peak_memory < 0
            or peak_memory > gates["max_sampled_peak_working_set_bytes"]
        ):
            row["reasons"].append("working-set memory bound exceeded or unreported")
        peak_gpu_memory = manifest.get("sampled_peak_gpu_memory_bytes")
        if (
            not isinstance(peak_gpu_memory, int)
            or peak_gpu_memory < 0
            or peak_gpu_memory > gates["max_sampled_peak_gpu_memory_bytes"]
        ):
            row["reasons"].append("GPU memory bound exceeded or unreported")
        if not math.isclose(
            configured_scale_gate,
            gates["max_committed_abs_log_scale"],
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            row["reasons"].append("configured scale gate differs from frozen protocol")
        if committed_scale > gates["max_committed_abs_log_scale"]:
            row["reasons"].append("committed scale-cliff threshold exceeded")
        if inactive_edges > queue_bounds["inactive_edge_cap"]:
            row["reasons"].append("inactive-edge queue bound exceeded")
        if max_free_poses > queue_bounds["max_free_poses"]:
            row["reasons"].append("global-BA free-pose bound exceeded")
        if indexed_frames > queue_bounds["long_loop_max_indexed_frames"]:
            row["reasons"].append("long-loop index bound exceeded")

        row.update(
            sim3_ate_rmse_m=ate,
            tracked_fraction=tracked_fraction,
            ms_per_frame_total=elapsed_ms,
            sampled_peak_working_set_bytes=peak_memory,
            sampled_peak_gpu_memory_bytes=peak_gpu_memory,
            max_committed_abs_log_scale=committed_scale,
            scale_jump_rejections_total=rejections,
        )
        row["success"] = not row["reasons"]
        if row["success"]:
            successful_ates[sequence].append(ate)
            max_scale = max(max_scale, committed_scale)
            scale_rejections += rejections
        run_rows.append(row)

    successful_runs = sum(row["success"] for row in run_rows)
    all_runs_successful = successful_runs == len(expected)
    sequence_means = {
        sequence: statistics.mean(values)
        for sequence, values in successful_ates.items()
        if len(values) == repetitions
    }
    all_sequences_successful = len(sequence_means) == len(sequences)
    mean_ate = statistics.mean(sequence_means.values()) if all_sequences_successful else None
    median_ate = statistics.median(sequence_means.values()) if all_sequences_successful else None
    worst_ate = max(sequence_means.values()) if all_sequences_successful else None
    euroc_gate = (
        all_runs_successful
        and all_sequences_successful
        and mean_ate is not None
        and mean_ate <= gates["mean_sequence_sim3_ate_rmse_m_max"]
    )
    public_gate, public_reasons = public_frontier_verdict(public_evidence)
    return {
        "schema_version": 1,
        "protocol_sha256": sha256(protocol_path),
        "expected_runs": len(expected),
        "successful_runs": successful_runs,
        "failed_or_missing_runs": len(expected) - successful_runs,
        "sequence_sim3_ate_rmse_m_mean": sequence_means,
        "all_sequence_mean_sim3_ate_rmse_m": mean_ate,
        "all_sequence_median_sim3_ate_rmse_m": median_ate,
        "all_sequence_worst_sim3_ate_rmse_m": worst_ate,
        "max_committed_abs_log_scale": max_scale if successful_runs else None,
        "scale_jump_rejections_total": scale_rejections,
        "euroc_engineering_gate_pass": euroc_gate,
        "public_frontier_gate_pass": public_gate,
        "public_frontier_reasons": public_reasons,
        "claimable_sota": euroc_gate and public_gate,
        "runs": run_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--public-evidence", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.matrix_root, args.protocol, args.public_evidence)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if report["claimable_sota"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
