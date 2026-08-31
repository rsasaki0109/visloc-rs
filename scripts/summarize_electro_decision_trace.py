#!/usr/bin/env python3
"""Convert unordered-SfM debug/timing output into a compact JSON trace.

Run the mapper with ``VISLOC_SFM_DEBUG=1`` and ``VISLOC_SFM_TIMING=1``, then
pass its stderr to this script.  The result intentionally contains decisions
and counters, not wall-clock ordering metadata, so repeated deterministic runs
can be compared directly after removing the optional timing fields.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PNP_RE = re.compile(
    r"^sfm-debug: PnP attempt #(\d+) on image (\d+) "
    r"(succeeded|failed) \((\d+) corrs -> (\d+) inliers"
)
POST_RE = re.compile(
    r"^sfm-debug: post-refinement "
    r"(?:(registered) image (\d+) \((\d+) corrs, (\d+) inliers\)|"
    r"PnP on image (\d+) failed \((\d+) corrs -> (none|\d+) inliers)"
)
EXHAUSTED_RE = re.compile(
    r"^sfm-debug:\s+image (\d+): trials exhausted "
    r"\((\d+)/(\d+), (\d+) corrs available\)"
)
SEED_RE = re.compile(
    r"^sfm-timing-seed-trial: index=(\d+) pair=\((\d+), (\d+)\) "
    r"reach=(\d+) elapsed=([0-9.]+)s"
)
SEED_SUMMARY_RE = re.compile(
    r"^sfm-timing-seed-summary: candidates=(\d+) attempted=(\d+) "
    r"zero_reach=(\d+) successful=(\d+) winner_reach=(\d+) "
    r"elapsed=([0-9.]+)s"
)
BA_RE = re.compile(
    r"^sfm-timing-ba: registered=(\d+) landmarks=(\d+) observations=(\d+) "
    r"warm_start=([0-9.]+)s assemble=([0-9.]+)s solve=([0-9.]+)s "
    r"writeback=([0-9.]+)s total=([0-9.]+)s iterations=(\d+) accepted=(\d+)"
)
TOTAL_RE = re.compile(
    r"^sfm-timing: total=([0-9.]+)s track_build=([0-9.]+)s "
    r"seed_growth=([0-9.]+)s final_refinement=([0-9.]+)s "
    r"geometry_recovery=([0-9.]+)s structureless=([0-9.]+)s "
    r"assembly=([0-9.]+)s"
)


def parse_trace(text: str, *, include_timing: bool = True) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "schema": "visloc_sfm_decision_trace_v1",
        "growth_pnp": [],
        "post_refinement_pnp": [],
        "exhausted_images": [],
        "seed_trials": [],
        "refinement_rounds": [],
    }
    for line in text.splitlines():
        if match := PNP_RE.match(line):
            attempt, image, outcome, corrs, inliers = match.groups()
            trace["growth_pnp"].append(
                {
                    "image": int(image),
                    "attempt": int(attempt),
                    "correspondences": int(corrs),
                    "inliers": int(inliers),
                    "accepted": outcome == "succeeded",
                }
            )
        elif match := POST_RE.match(line):
            registered, ok_image, ok_corrs, ok_inliers, fail_image, fail_corrs, fail_inliers = (
                match.groups()
            )
            accepted = registered is not None
            trace["post_refinement_pnp"].append(
                {
                    "image": int(ok_image if accepted else fail_image),
                    "correspondences": int(ok_corrs if accepted else fail_corrs),
                    "inliers": (
                        int(ok_inliers)
                        if accepted
                        else None if fail_inliers == "none" else int(fail_inliers)
                    ),
                    "accepted": accepted,
                }
            )
        elif match := EXHAUSTED_RE.match(line):
            image, trials, limit, corrs = match.groups()
            trace["exhausted_images"].append(
                {
                    "image": int(image),
                    "trials": int(trials),
                    "trial_limit": int(limit),
                    "available_correspondences": int(corrs),
                }
            )
        elif match := SEED_RE.match(line):
            index, left, right, reach, elapsed = match.groups()
            event = {
                "index": int(index),
                "pair": [int(left), int(right)],
                "reach": int(reach),
            }
            if include_timing:
                event["elapsed_s"] = float(elapsed)
            trace["seed_trials"].append(event)
        elif match := SEED_SUMMARY_RE.match(line):
            candidates, attempted, zero_reach, successful, winner_reach, elapsed = match.groups()
            summary = {
                "candidates": int(candidates),
                "attempted": int(attempted),
                "zero_reach": int(zero_reach),
                "successful": int(successful),
                "winner_reach": int(winner_reach),
            }
            if include_timing:
                summary["elapsed_s"] = float(elapsed)
            trace["seed_summary"] = summary
        elif match := BA_RE.match(line):
            (
                registered,
                landmarks,
                observations,
                warm_start,
                assemble,
                solve,
                writeback,
                total,
                iterations,
                accepted,
            ) = match.groups()
            event = {
                "registered_images": int(registered),
                "landmarks": int(landmarks),
                "observations": int(observations),
                "iterations": int(iterations),
                "accepted_steps": int(accepted),
            }
            if include_timing:
                event["timing_s"] = {
                    "warm_start": float(warm_start),
                    "assemble": float(assemble),
                    "solve": float(solve),
                    "writeback": float(writeback),
                    "total": float(total),
                }
            trace["refinement_rounds"].append(event)
        elif match := TOTAL_RE.match(line):
            if include_timing:
                total, track_build, seed_growth, final, recovery, structureless, assembly = (
                    match.groups()
                )
                trace["timing_s"] = {
                    "total": float(total),
                    "track_build": float(track_build),
                    "seed_growth": float(seed_growth),
                    "final_refinement": float(final),
                    "geometry_recovery": float(recovery),
                    "structureless": float(structureless),
                    "assembly": float(assembly),
                }

    trace["summary"] = {
        "growth_attempts": len(trace["growth_pnp"]),
        "growth_accepted": sum(event["accepted"] for event in trace["growth_pnp"]),
        "post_attempts": len(trace["post_refinement_pnp"]),
        "post_accepted": sum(event["accepted"] for event in trace["post_refinement_pnp"]),
        "exhausted": len(trace["exhausted_images"]),
        "refinement_rounds": len(trace["refinement_rounds"]),
    }
    return trace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stderr", type=Path, help="captured mapper stderr")
    parser.add_argument("--output", type=Path, required=True, help="JSON output path")
    parser.add_argument(
        "--decisions-only",
        action="store_true",
        help="omit non-deterministic wall-clock measurements",
    )
    args = parser.parse_args()
    trace = parse_trace(
        args.stderr.read_text(encoding="utf-8"), include_timing=not args.decisions_only
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
