#!/usr/bin/env python3
"""Metric scale of a EuRoC trajectory: overall, over time, and when it settles.

One run (prints a JSON object, and writes it with --out):

    python scripts/euroc_scale_report.py run --sequence MH_01_easy --mode mono \
        --ground-truth-csv MH_01_easy/mav0/state_groundtruth_estimate0/data.csv \
        --trajectory out/mono/MH_01_easy/trajectory.tum --out mono_MH_01_easy.json

A Markdown table from several runs:

    python scripts/euroc_scale_report.py summarize results/*.json

Uses the ground-truth loader, association and Umeyama alignment of
evaluate_euroc_trajectory.py, as euroc_scale_diagnostic.py does. The scale
error is how much larger the estimate is than the truth (+2 % = 2 % too large):
1 / s - 1, where s maps the estimate onto the ground truth (gt ~ s R est + t).

Over time, the scale is fitted (Sim3) on sliding windows of --window seconds
that move at least --min-path metres. The trajectory has settled at the start
of the first window from which every later window is within --tolerance.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("evaluate_euroc_trajectory", _HERE / "evaluate_euroc_trajectory.py")
_eval = importlib.util.module_from_spec(_spec)
sys.modules["evaluate_euroc_trajectory"] = _eval
_spec.loader.exec_module(_eval)


def rms(a, b):
    return float(np.sqrt(np.mean(np.sum(np.square(a - b), axis=1))))


def sim3_scale_error(est, gt):
    scale, _, _ = _eval.umeyama(est, gt, True)
    return 1.0 / scale - 1.0


def windows(times, est, gt, window_s, step_s, min_path_m):
    """(start, scale error) per sliding window that moves at least `min_path_m`."""
    out = []
    start = times[0]
    while start + window_s <= times[-1] + 1e-9:
        sel = (times >= start) & (times < start + window_s)
        if sel.sum() >= 3:
            path = float(np.sum(np.linalg.norm(np.diff(gt[sel], axis=0), axis=1)))
            if path >= min_path_m:
                out.append((float(start - times[0]), sim3_scale_error(est[sel], gt[sel])))
        start += step_s
    return out


def settled_at(window_errors, tolerance):
    """Start of the first window from which every window is within `tolerance`."""
    settled = None
    for start, error in reversed(window_errors):
        if abs(error) > tolerance:
            break
        settled = start
    return settled


def report(ground_truth_csv, trajectory, max_diff_ns=10_000_000, window_s=10.0, step_s=2.0,
           min_path_m=1.0, tolerance=0.05, tum_time_unit="s"):
    ground_truth = _eval.load_ground_truth(Path(ground_truth_csv))
    estimate = _eval.load_estimate(Path(trajectory), tum_time_unit=tum_time_unit)
    pairs = _eval.associate(ground_truth, estimate, max_diff_ns)
    if len(pairs) < 3:
        raise ValueError("fewer than 3 associated poses")
    times = np.asarray([p[1][0] for p in pairs], dtype=float) * 1e-9
    gt = np.asarray([p[0][1] for p in pairs])
    est = np.asarray([p[1][1] for p in pairs])

    _, rotation, translation = _eval.umeyama(est, gt, False)
    se3 = rms((rotation @ est.T).T + translation, gt)
    scale, rotation, translation = _eval.umeyama(est, gt, True)
    sim3 = rms(scale * (rotation @ est.T).T + translation, gt)

    per_window = windows(times, est, gt, window_s, step_s, min_path_m)
    settled = settled_at(per_window, tolerance)
    after = [abs(e) for s, e in per_window if settled is not None and s >= settled]
    return {
        "poses": len(pairs),
        "duration_s": float(times[-1] - times[0]),
        "path_m": float(np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1))),
        "se3_ate_m": se3,
        "sim3_ate_m": sim3,
        "scale_error": 1.0 / scale - 1.0,
        "window_s": window_s,
        "tolerance": tolerance,
        "windows": [{"start_s": s, "scale_error": e} for s, e in per_window],
        "settled_s": settled,
        "median_abs_window_error_after_settling": float(np.median(after)) if after else None,
        "max_abs_window_error_after_settling": float(np.max(after)) if after else None,
    }


def fmt(value, spec, missing="–"):
    return missing if value is None else format(value, spec)


def summarize(paths):
    runs = [json.loads(Path(p).read_text()) for p in paths]
    modes = sorted({r["mode"] for r in runs})
    sequences = sorted({r["sequence"] for r in runs})
    by = {(r["sequence"], r["mode"]): r for r in runs}
    lines = [
        "| Sequence | Mode | SE3 ATE (m) | Sim3 ATE (m) | Scale error | Settled (s, ±5 %) | Window error after |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for seq in sequences:
        for mode in modes:
            r = by.get((seq, mode))
            if r is None:
                lines.append(f"| {seq} | {mode} | – | – | – | – | – |")
                continue
            if "error" in r:
                lines.append(f"| {seq} | {mode} | failed: {r['error']} | | | | |")
                continue
            after = r["median_abs_window_error_after_settling"]
            lines.append(
                f"| {seq} | {mode} | {r['se3_ate_m']:.3f} | {r['sim3_ate_m']:.3f} | "
                f"{100 * r['scale_error']:+.1f} % | {fmt(r['settled_s'], '.0f', 'never')} | "
                f"{'–' if after is None else f'{100 * after:.1f} %'} |"
            )
    for mode in modes:
        ok = [by[(s, mode)] for s in sequences if (s, mode) in by and "error" not in by[(s, mode)]]
        if ok:
            errors = [abs(r["scale_error"]) for r in ok]
            lines.append(
                f"\n**{mode}**: {len(ok)} runs, median |scale error| {100 * float(np.median(errors)):.1f} %, "
                f"max {100 * max(errors):.1f} %, settled within ±5 % in "
                f"{sum(r['settled_s'] is not None for r in ok)}/{len(ok)}."
            )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="score one trajectory")
    run.add_argument("--sequence", required=True)
    run.add_argument("--mode", required=True, help="label, e.g. mono or stereo")
    run.add_argument("--ground-truth-csv", type=Path, required=True)
    run.add_argument("--trajectory", type=Path, required=True)
    run.add_argument("--out", type=Path)
    run.add_argument("--max-diff-ns", type=int, default=10_000_000)
    run.add_argument("--window", type=float, default=10.0, help="seconds")
    run.add_argument("--step", type=float, default=2.0, help="seconds")
    run.add_argument("--min-path", type=float, default=1.0, help="metres a window must move")
    run.add_argument("--tolerance", type=float, default=0.05)
    run.add_argument("--tum-time-unit", choices=["s", "ns"], default="s",
                     help="trajectory timestamps (basalt_euroc_vio_demo writes seconds)")
    summary = sub.add_parser("summarize", help="Markdown table of run JSONs")
    summary.add_argument("runs", nargs="+", type=Path)
    args = p.parse_args()

    if args.command == "summarize":
        print(summarize(args.runs))
        return 0
    result = {"sequence": args.sequence, "mode": args.mode}
    try:
        result.update(report(args.ground_truth_csv, args.trajectory, args.max_diff_ns, args.window,
                             args.step, args.min_path, args.tolerance, args.tum_time_unit))
    except (OSError, ValueError) as e:  # a failed run still gets its row
        result["error"] = str(e)
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "windows"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
