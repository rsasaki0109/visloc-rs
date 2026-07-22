#!/usr/bin/env python3
"""Aggregate the frozen three-sequence SSfM suite without hiding failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


ENGINES = (
    "visloc_hierarchical",
    "colmap_incremental",
    "colmap_global",
    "gluemap",
    "instantsfm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def success_metrics(cell: dict) -> dict:
    evaluation = cell["evaluation"]
    return {
        "ate_sim3_rmse_m": evaluation["ate_translation_sim3_m"]["rmse"],
        "rpe_translation_rmse_m": evaluation["rpe_translation_consecutive_m"]["rmse"],
        "rpe_rotation_rmse_deg": evaluation["rpe_rotation_consecutive_deg"]["rmse"],
        "registration_rate": cell["registration_rate"],
        "total_wall_seconds": cell["total_wall_seconds"],
        "peak_process_tree_rss_bytes": cell["peak_process_tree_rss_bytes"],
        "peak_global_gpu_memory_mib": cell.get("peak_global_gpu_memory_mib"),
    }


def distribution(values: list[float], higher_is_worse: bool = True) -> dict | None:
    if not values:
        return None
    return {
        "median": statistics.median(values),
        "worst": max(values) if higher_is_worse else min(values),
    }


def dominates(left: dict, right: dict) -> bool:
    minimizing = (
        "ate_sim3_rmse_m",
        "total_wall_seconds",
        "peak_process_tree_rss_bytes",
    )
    no_worse = all(left[key] <= right[key] for key in minimizing)
    no_worse = no_worse and left["registration_rate"] >= right["registration_rate"]
    strictly_better = any(left[key] < right[key] for key in minimizing)
    strictly_better = strictly_better or left["registration_rate"] > right["registration_rate"]
    return no_worse and strictly_better


def load_sequence_results(
    suite_root: Path,
    sequence: str,
    protocol_sha256: str,
) -> tuple[dict, dict]:
    sequence_root = suite_root / sequence
    final_path = sequence_root / "final" / "manifest.json"
    runner_path = sequence_root / "manifest.json"
    if final_path.is_file():
        path = final_path
        kind = "final"
    elif runner_path.is_file():
        path = runner_path
        kind = "runner_failure"
    else:
        raise FileNotFoundError(
            f"missing final or explicit runner-failure manifest for {sequence}"
        )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["sequence"] != sequence:
        raise ValueError(f"sequence mismatch in {path}")
    if manifest["protocol_sha256"] != protocol_sha256:
        raise ValueError(f"protocol mismatch in {path}")

    evidence = {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "kind": kind,
    }
    if kind == "final":
        if set(manifest["results"]) != set(ENGINES):
            raise ValueError(f"missing or unexpected engine cells in {path}")
        return evidence, manifest["results"]

    if manifest.get("status") != "failed":
        raise ValueError(f"runner manifest without final result is not failed: {path}")
    reason = manifest.get("failure_reason") or "unspecified sequence-runner failure"
    return evidence, {
        engine: {
            "status": "dnf",
            "reason": f"sequence runner failed before finalization: {reason}",
            "registered_images": 0,
            "registration_rate": 0.0,
        }
        for engine in ENGINES
    }


def main() -> int:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    sequences = protocol["selection"]["held_out_sequences"]
    if len(sequences) != 3 or len(set(sequences)) != 3:
        raise ValueError("protocol must bind exactly three unique held-out sequences")

    manifests = {}
    cells = {}
    frontier = {}
    for sequence in sequences:
        manifests[sequence], cells[sequence] = load_sequence_results(
            args.suite_root,
            sequence,
            protocol_sha256,
        )

        successful = {
            engine: success_metrics(cell)
            for engine, cell in cells[sequence].items()
            if cell["status"] == "success"
        }
        frontier[sequence] = [
            engine
            for engine, metrics in successful.items()
            if not any(
                other != engine and dominates(other_metrics, metrics)
                for other, other_metrics in successful.items()
            )
        ]

    aggregate = {}
    for engine in ENGINES:
        engine_cells = [cells[sequence][engine] for sequence in sequences]
        successful = [success_metrics(cell) for cell in engine_cells if cell["status"] == "success"]
        failures = [
            {"sequence": sequence, "reason": cells[sequence][engine].get("reason")}
            for sequence in sequences
            if cells[sequence][engine]["status"] != "success"
        ]
        aggregate[engine] = {
            "success_count": len(successful),
            "required_count": len(sequences),
            "all_sequences_successful": not failures,
            "worst_outcome": "dnf" if failures else "success",
            "failures": failures,
            "frontier_sequence_count": sum(engine in frontier[sequence] for sequence in sequences),
            "success_only_metrics": {
                "ate_sim3_rmse_m": distribution(
                    [metrics["ate_sim3_rmse_m"] for metrics in successful]
                ),
                "rpe_translation_rmse_m": distribution(
                    [metrics["rpe_translation_rmse_m"] for metrics in successful]
                ),
                "rpe_rotation_rmse_deg": distribution(
                    [metrics["rpe_rotation_rmse_deg"] for metrics in successful]
                ),
                "registration_rate": distribution(
                    [metrics["registration_rate"] for metrics in successful],
                    higher_is_worse=False,
                ),
                "total_wall_seconds": distribution(
                    [metrics["total_wall_seconds"] for metrics in successful]
                ),
                "peak_process_tree_rss_bytes": distribution(
                    [metrics["peak_process_tree_rss_bytes"] for metrics in successful]
                ),
            },
        }

    output = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "generated_utc": timestamp(),
        "sequence_manifests": manifests,
        "per_sequence_results": cells,
        "per_sequence_reproduced_frontier": frontier,
        "aggregate": aggregate,
        "internal_colmap_frontier_gate": {
            "passed": aggregate["visloc_hierarchical"]["all_sequences_successful"]
            and aggregate["visloc_hierarchical"]["frontier_sequence_count"] == len(sequences),
            "scope": "Only the reproduced COLMAP incremental/global baselines in this artifact.",
        },
        "external_baseline_completeness_gate": {
            "passed": all(
                aggregate[engine]["success_count"]
                + len(aggregate[engine]["failures"])
                == len(sequences)
                for engine in ("gluemap", "instantsfm")
            ),
            "scope": (
                "Both external engines have a success or explicit DNF cell "
                "for every frozen sequence."
            ),
        },
        "claimable_sota_gate": {
            "passed": False,
            "reason": "ORBIT and remaining release requirements are separate mandatory evidence.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
