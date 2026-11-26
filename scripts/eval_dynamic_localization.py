#!/usr/bin/env python3
"""Score `basalt_euroc_localize_demo` runs for the dynamic-obstacle benchmark.

For each run directory (its `--out-dir`), reads

  per_frame.csv        every attempt: success, inliers, latency, correction
  localization.tum     the accepted localizations (IMU poses, map frame)
  trajectory_map.tum   with --vio / --odometry: the map pose of every frame
  diagnostics.csv      with --diagnostics: inlier cells, expected landmarks

and the ground truth (TUM, seconds) in the map's frame. The map must be in
the ground truth's frame already (e.g. `tum_rgbd_save_map --keep-world` along
the ground truth): nothing is fitted to the scored poses.

Prints one Markdown row per run, and with --json writes every number.

  python3 scripts/eval_dynamic_localization.py --gt GT.tum --label walking_xyz RUN_DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_left
from pathlib import Path


def read_tum(path: Path) -> list[tuple[float, tuple[float, float, float]]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        values = line.split()
        rows.append((float(values[0]), (float(values[1]), float(values[2]), float(values[3]))))
    rows.sort()
    return rows


class GroundTruth:
    def __init__(self, rows, max_gap_s: float = 0.05):
        self.stamps = [stamp for stamp, _ in rows]
        self.positions = [position for _, position in rows]
        self.max_gap_s = max_gap_s

    def at(self, stamp: float):
        i = bisect_left(self.stamps, stamp)
        if i == 0 or i == len(self.stamps):
            return None
        t0, t1 = self.stamps[i - 1], self.stamps[i]
        if t1 - t0 > self.max_gap_s:
            return None
        a = (stamp - t0) / (t1 - t0)
        p0, p1 = self.positions[i - 1], self.positions[i]
        return tuple(p0[k] + a * (p1[k] - p0[k]) for k in range(3))


def errors_cm(poses, truth: GroundTruth) -> list[float]:
    out = []
    for stamp, position in poses:
        reference = truth.at(stamp)
        if reference is not None:
            out.append(100.0 * math.dist(position, reference))
    return out


def percentile(values, q):
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def rmse(values):
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else float("nan")


def longest_gap_s(all_stamps, good_stamps) -> float:
    """Longest stretch of the sequence without a good stamp, from its first
    to its last attempt."""
    if not all_stamps:
        return float("nan")
    points = [all_stamps[0]] + sorted(good_stamps) + [all_stamps[-1]]
    return max(b - a for a, b in zip(points, points[1:]))


def score(run: Path, truth: GroundTruth, wrong_cm: float) -> dict:
    attempts = list(csv.DictReader((run / "per_frame.csv").open()))
    stamps = [int(row["timestamp_ns"]) * 1e-9 for row in attempts]
    fixes = read_tum(run / "localization.tum")
    fix_errors = errors_cm(fixes, truth)
    result = {
        "run": str(run),
        "attempts": len(attempts),
        "localized": len(fixes),
        "localized_share": len(fixes) / len(attempts) if attempts else float("nan"),
        "longest_gap_s": longest_gap_s(stamps, [stamp for stamp, _ in fixes]),
        "fix_rmse_cm": rmse(fix_errors),
        "fix_median_cm": percentile(fix_errors, 0.5),
        "fix_max_cm": max(fix_errors) if fix_errors else float("nan"),
        "wrong_fixes": sum(error > wrong_cm for error in fix_errors),
        "latency_median_ms": percentile([float(row["latency_ms"]) for row in attempts], 0.5),
    }
    corrections = [row.get("correction", "-") for row in attempts]
    result["used_fixes"] = sum(c in ("accepted", "initialized", "reset", "weak-accepted", "weak-initialized") for c in corrections)
    result["resets"] = sum(c == "reset" for c in corrections)
    # With --vio / --odometry: the map pose of every frame.
    trajectory = run / "trajectory_map.tum"
    if trajectory.exists():
        poses = read_tum(trajectory)
        map_errors = errors_cm(poses, truth)
        used = {
            int(row["timestamp_ns"]) * 1e-9
            for row, c in zip(attempts, corrections)
            if c in ("accepted", "initialized", "reset", "weak-accepted", "weak-initialized")
        }
        bad_used = sum(
            1
            for stamp, position in fixes
            if any(abs(stamp - u) < 1e-6 for u in used)
            and (reference := truth.at(stamp)) is not None
            and 100.0 * math.dist(position, reference) > wrong_cm
        )
        result.update(
            {
                "map_poses": len(poses),
                "map_rmse_cm": rmse(map_errors),
                "map_median_cm": percentile(map_errors, 0.5),
                "map_max_cm": max(map_errors) if map_errors else float("nan"),
                "wrong_fixes_used": bad_used,
                "longest_gap_used_s": longest_gap_s(stamps, sorted(used)),
            }
        )
    diagnostics = run / "diagnostics.csv"
    if diagnostics.exists():
        rows = list(csv.DictReader(diagnostics.open()))
        accepted = [row for row in rows if row["success"] == "true"]
        viewed = [row for row in rows if row["view_pose"] != "-" and int(row["expected"]) > 0]
        result.update(
            {
                "inlier_cells_median_accepted": percentile([int(r["inlier_cells"]) for r in accepted], 0.5),
                "expected_inlier_share_median": percentile(
                    [int(r["expected_inliers"]) / int(r["expected"]) for r in viewed], 0.5
                ),
                "missing_cells_median": percentile([int(r["missing_cells"]) for r in viewed], 0.5),
                "matched_cell_share_median": percentile(
                    [
                        1 - int(r["missing_cells"]) / int(r["expected_cells"])
                        for r in viewed
                        if int(r["expected_cells"]) > 0
                    ],
                    0.5,
                ),
                "diag_ms_median": percentile([float(r["diag_ms"]) for r in rows], 0.5),
            }
        )
        masked = [r for r in rows if r["mask_inliers"] != ""]
        if masked:
            total = sum(int(r["inliers"]) for r in masked)
            result["mask_inlier_share"] = sum(int(r["mask_inliers"]) for r in masked) / total if total else 0.0
    return result


def fmt(value, digits=1):
    if isinstance(value, float):
        return "—" if math.isnan(value) else f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--wrong-cm", type=float, default=10.0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    truth = GroundTruth(read_tum(args.gt))
    results = []
    for run in args.runs:
        result = score(run, truth, args.wrong_cm)
        result["label"] = args.label or run.name
        results.append(result)
        line = (
            f"| {result['label']} | {result['localized']}/{result['attempts']} "
            f"({100 * result['localized_share']:.1f}%) | {fmt(result['longest_gap_s'])} s | "
            f"{fmt(result['fix_rmse_cm'])} / {fmt(result['fix_median_cm'])} / {fmt(result['fix_max_cm'])} cm | "
            f"{result['wrong_fixes']} |"
        )
        if "map_poses" in result:
            line += (
                f" {fmt(result['map_rmse_cm'])} / {fmt(result['map_max_cm'])} cm |"
                f" {result['wrong_fixes_used']} | {result['resets']} |"
            )
        if "inlier_cells_median_accepted" in result:
            line += (
                f" {fmt(result['inlier_cells_median_accepted'], 0)} |"
                f" {fmt(100 * result['expected_inlier_share_median'], 0)}% |"
                f" {fmt(100 * result['matched_cell_share_median'], 0)}% |"
                f" {fmt(result['missing_cells_median'], 0)} |"
            )
        if "mask_inlier_share" in result:
            line += f" {100 * result['mask_inlier_share']:.1f}% |"
        line += f" {fmt(result['latency_median_ms'])} ms |"
        print(line)
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
