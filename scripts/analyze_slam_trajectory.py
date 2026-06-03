#!/usr/bin/env python3
"""Diagnose a SLAM / VO run from its slam_errors.csv: is it tracking, is it
continuous, is it accurate?

The EuRoC / KITTI online demos write one slam_errors.csv row per
tracking-success frame (timestamp, frame_idx, ground-truth and estimated
position, per-frame error). This tool turns that into an at-a-glance quality
report so "the trajectory looks broken" becomes a measured statement instead of
a guess. It is the analysis-kit companion to the animation scripts.

It reports, and plots as a 4-panel PNG:
  1. Top-down (X, Y) trajectory, ground truth vs estimate (rigidly aligned),
     with tracking-dropout gaps drawn as dotted bridges so intermittent
     tracking is visible rather than hidden by a connect-the-dots line.
  2. Tracking timeline: which frames produced a pose vs dropped out, the
     coverage fraction, and the longest continuous run.
  3. Per-frame motion: the estimated step size vs the ground-truth step size -
     VO jitter shows up as estimated steps far above the true motion (the usual
     "why is it so noisy" culprit when the platform is nearly stationary).
  4. Localization error over time (rigid-aligned), with the RMSE line.

Usage:
    python3 scripts/analyze_slam_trajectory.py \\
        --errors-csv target/euroc_phase26_MH_01_easy_strict_superpoint/slam_errors.csv \\
        --output     docs/assets/mh01_superpoint_diagnostic.png \\
        --label      'EuRoC MH_01 - SuperPoint strict'

Pass --output - (or omit it) to print the text report only. Asset/diagnostic
tool, not part of CI.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--errors-csv", type=Path, required=True)
    p.add_argument("--output", type=str, default="-", help="PNG path, or '-' for text report only")
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--stationary-thresh", type=float, default=0.01, help="gt step [m] below which a frame counts as 'platform nearly still'")
    return p.parse_args()


def umeyama(src, dst, with_scale: bool):
    import numpy as np

    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = s_c.T @ d_c / len(src)
    u, sig, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    scale = ((sig * np.array([1.0, 1.0, d])).sum() / ((s_c ** 2).sum() / len(src))) if with_scale else 1.0
    t = mu_d - scale * rot @ mu_s
    return (scale * (rot @ src.T)).T + t


def analyze(errors_csv: Path, stationary_thresh: float):
    import numpy as np

    rows = list(csv.DictReader(errors_csv.open()))
    gt = np.array([[float(r["gt_px"]), float(r["gt_py"]), float(r["gt_pz"])] for r in rows])
    est = np.array([[float(r["est_px"]), float(r["est_py"]), float(r["est_pz"])] for r in rows])
    fidx = np.array([int(r["frame_idx"]) for r in rows])

    est_rigid = umeyama(est, gt, False)
    est_sim = umeyama(est, gt, True)
    err = np.linalg.norm(est_rigid - gt, axis=1)
    err_sim = np.linalg.norm(est_sim - gt, axis=1)

    span = int(fidx[-1] - fidx[0] + 1)
    coverage = len(fidx) / span if span else 1.0

    # contiguous (consecutive frame_idx) runs
    gaps = np.diff(fidx)
    breaks = np.flatnonzero(gaps > 1)
    seg_bounds = np.concatenate([[0], breaks + 1, [len(fidx)]])
    segs = [(seg_bounds[i], seg_bounds[i + 1]) for i in range(len(seg_bounds) - 1)]
    seg_lens = [b - a for a, b in segs]
    longest = max(seg_lens) if seg_lens else 0

    est_step = np.linalg.norm(np.diff(est_rigid, axis=0), axis=1)
    gt_step = np.linalg.norm(np.diff(gt, axis=0), axis=1)
    # only count steps inside a contiguous segment (a step across a gap is meaningless)
    contiguous_step = gaps == 1
    still = (gt_step < stationary_thresh) & contiguous_step
    jitter = float(np.median(est_step[still])) if still.any() else float("nan")

    return {
        "rows": rows, "gt": gt, "est_rigid": est_rigid, "fidx": fidx,
        "err": err, "err_sim": err_sim, "gt_step": gt_step, "est_step": est_step,
        "contiguous_step": contiguous_step, "segs": segs,
        "n": len(fidx), "span": span, "coverage": coverage,
        "longest_contiguous": longest, "n_gaps": int(len(breaks)),
        "max_gap": int(gaps.max()) if len(gaps) else 0,
        "rmse_rigid": float(np.sqrt((err ** 2).mean())),
        "rmse_sim": float(np.sqrt((err_sim ** 2).mean())),
        "jitter_still": jitter,
        "gt_path": float(gt_step.sum()), "est_path": float(est_step[contiguous_step].sum()),
    }


def print_report(a, label):
    print(f"== {label} ==")
    print(f"tracked frames        : {a['n']}  (frame_idx span {a['span']}, coverage {a['coverage'] * 100:.1f} %)")
    print(f"tracking continuity   : {a['n_gaps']} dropouts, longest continuous run {a['longest_contiguous']} frames, max gap {a['max_gap']}")
    print(f"rigid ATE / sim ATE   : {a['rmse_rigid'] * 100:.1f} cm / {a['rmse_sim'] * 100:.1f} cm")
    print(f"ground-truth path     : {a['gt_path']:.2f} m   estimate path (in-segment): {a['est_path']:.2f} m   ratio {a['est_path'] / max(a['gt_path'], 1e-9):.2f}")
    j = a["jitter_still"]
    print(f"VO jitter when still  : median est step {j * 100:.1f} cm/frame" if j == j else "VO jitter when still  : n/a (platform never near-still)")
    verdict = (
        "INTERMITTENT - tracks in bursts (low coverage); a connect-the-dots plot looks discontinuous"
        if a["coverage"] < 0.5 else "mostly continuous tracking"
    )
    print(f"verdict               : {verdict}")


def main() -> int:
    args = parse_args()
    label = args.label or args.errors_csv.parent.name
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        print("need numpy", file=sys.stderr)
        return 2

    a = analyze(args.errors_csv, args.stationary_thresh)
    print_report(a, label)

    if args.output in ("-", "", None):
        return 0

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available; text report only", file=sys.stderr)
        return 0

    gt, est, fidx = a["gt"], a["est_rigid"], a["fidx"]
    gt_c, est_c = "#3a3a3a", "#d23737"

    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.suptitle(
        f"{label}\ncoverage {a['coverage'] * 100:.1f} %   longest continuous {a['longest_contiguous']} fr   "
        f"rigid ATE {a['rmse_rigid'] * 100:.1f} cm   still-jitter {a['jitter_still'] * 100:.1f} cm",
        fontsize=11,
    )

    # 1: top-down with gap bridges
    ax = axs[0, 0]
    for s0, s1 in a["segs"]:
        ax.plot(gt[s0:s1, 0], gt[s0:s1, 1], color=gt_c, lw=2.0)
        ax.plot(est[s0:s1, 0], est[s0:s1, 1], color=est_c, lw=1.3, ls="--")
    for i in np.flatnonzero(np.diff(fidx) > 1):  # dropout bridges on the estimate
        ax.plot(est[i:i + 2, 0], est[i:i + 2, 1], color=est_c, lw=0.7, ls=":", alpha=0.5)
    ax.plot([], [], color=gt_c, lw=2.0, label="ground truth")
    ax.plot([], [], color=est_c, lw=1.3, ls="--", label="estimate")
    ax.plot([], [], color=est_c, lw=0.7, ls=":", label="tracking dropout")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8); ax.set_title("Top-down trajectory")

    # 2: tracking timeline
    ax = axs[0, 1]
    full = np.arange(fidx[0], fidx[-1] + 1)
    tracked = np.isin(full, fidx)
    ax.fill_between(full, 0, tracked.astype(float), step="mid", color="#16c79a", alpha=0.8)
    ax.set_ylim(0, 1.2); ax.set_yticks([0, 1]); ax.set_yticklabels(["dropout", "tracked"])
    ax.set_xlabel("frame index"); ax.set_title(f"Tracking timeline ({a['coverage'] * 100:.0f} % coverage)")
    ax.grid(True, alpha=0.3)

    # 3: per-frame motion
    ax = axs[1, 0]
    contig = a["contiguous_step"]
    x = np.arange(len(a["est_step"]))
    ax.plot(x[contig], a["est_step"][contig] * 100, color=est_c, lw=1.0, label="estimate step")
    ax.plot(x[contig], a["gt_step"][contig] * 100, color=gt_c, lw=1.2, label="ground-truth step")
    ax.set_xlabel("step index (within-segment)"); ax.set_ylabel("per-frame step [cm]")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Per-frame motion (VO jitter = est >> gt)")

    # 4: error over time
    ax = axs[1, 1]
    ax.plot(np.arange(a["n"]), a["err"] * 100, color=est_c, lw=1.1)
    ax.axhline(a["rmse_rigid"] * 100, color=gt_c, lw=1.0, ls=":", label=f"RMSE {a['rmse_rigid'] * 100:.1f} cm")
    ax.set_xlabel("tracked-frame index"); ax.set_ylabel("rigid-aligned error [cm]")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8); ax.set_title("Localization error over time")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
