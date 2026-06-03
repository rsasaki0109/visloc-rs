#!/usr/bin/env python3
"""Frontend health over time: feature-match quality per frame, overlaid on where
the tracker actually kept or dropped the pose.

The trajectory diagnostic (analyze_slam_trajectory.py) shows *where* tracking
drops out; this shows *why*. It matches each consecutive pair of exported
SuperPoint frames (the same mutual-NN + Lowe-ratio matcher the animation uses)
and plots, frame by frame:
  * feature count and surviving-match count,
  * match ratio (matches / features) and median match strength (cosine),
  * median match displacement in pixels (apparent motion / blur proxy),
with the tracking-success timeline from slam_errors.csv shaded underneath - so
whether a tracking dropout coincides with a match-ratio collapse, or whether the
frontend stays healthy and the dropout came from a geometry / scale gate
instead, is visible at a glance. (On EuRoC MH_01 strict-stereo it is the latter:
~480 matches at ratio ~0.8 throughout, no worse during dropouts.)

Usage:
    python3 scripts/analyze_match_quality.py \\
        --features-dir target/euroc_phase26_superpoint/MH_01_easy/cam0 \\
        --frame-start 22 --frame-end 200 \\
        --errors-csv target/euroc_phase26_MH_01_easy_strict_superpoint/slam_errors.csv \\
        --output docs/assets/mh01_match_quality.png \\
        --label 'EuRoC MH_01 - SuperPoint frontend health'

Analysis tool, not part of CI.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from animate_euroc_match_track import load_features, match  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--frame-start", type=int, required=True)
    p.add_argument("--frame-end", type=int, required=True)
    p.add_argument("--errors-csv", type=Path, default=None, help="optional, to shade tracking-success frames")
    p.add_argument("--output", type=str, default="-")
    p.add_argument("--label", type=str, default="frontend match quality")
    p.add_argument("--top-features", type=int, default=600)
    p.add_argument("--ratio", type=float, default=0.85)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import numpy as np
    except ImportError:
        print("need numpy", file=sys.stderr)
        return 2

    frames = list(range(args.frame_start, args.frame_end + 1))
    feat = {}
    for f in frames:
        path = args.features_dir / f"frame_{f:06d}_features.txt"
        feat[f] = load_features(path, args.top_features) if path.exists() else (np.zeros((0, 2)), np.zeros((0, 1)))

    idx, n_feat, n_match, ratio, disp = [], [], [], [], []
    for f in frames[1:]:
        xy_a, da = feat[f]
        xy_b, db = feat[f - 1]
        if len(da) < 2 or len(db) < 2:
            idx.append(f); n_feat.append(len(da)); n_match.append(0); ratio.append(0.0); disp.append(np.nan)
            continue
        cur, prev = match(xy_a, da, xy_b, db, args.ratio)
        m = len(cur)
        idx.append(f); n_feat.append(len(da)); n_match.append(m)
        ratio.append(m / max(len(da), 1))
        disp.append(float(np.median(np.linalg.norm(cur - prev, axis=1))) if m else np.nan)

    idx = np.array(idx)
    n_feat = np.array(n_feat); n_match = np.array(n_match)
    ratio = np.array(ratio); disp = np.array(disp)

    tracked = None
    if args.errors_csv and args.errors_csv.exists():
        tf = {int(r["frame_idx"]) for r in csv.DictReader(args.errors_csv.open())}
        tracked = np.array([f in tf for f in idx])

    # text report
    print(f"== {args.label} ==")
    print(f"frames {args.frame_start}..{args.frame_end}  ({len(idx)} pairs)")
    print(f"features/frame  : median {int(np.median(n_feat))}")
    print(f"matches/frame   : median {int(np.median(n_match))}  min {int(n_match.min())}  max {int(n_match.max())}")
    print(f"match ratio     : median {np.median(ratio):.2f}  min {ratio.min():.2f}")
    print(f"median displace : median {np.nanmedian(disp):.1f} px  max {np.nanmax(disp):.1f} px")
    if tracked is not None:
        lo = ratio[tracked].mean() if tracked.any() else float("nan")
        hi = ratio[~tracked].mean() if (~tracked).any() else float("nan")
        print(f"match ratio when tracked {lo:.2f}  vs dropped {hi:.2f}  (coverage {tracked.mean() * 100:.0f} %)")

    if args.output in ("-", "", None):
        return 0

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; text report only", file=sys.stderr)
        return 0

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(args.label, fontsize=12)

    def shade(ax):
        if tracked is None:
            return
        ax.fill_between(idx, 0, 1, where=tracked, transform=ax.get_xaxis_transform(),
                        color="#16c79a", alpha=0.12, step="mid", label="tracking success")

    ax = axs[0]; shade(ax)
    ax.plot(idx, n_feat, color="#888", lw=1.0, label="features")
    ax.plot(idx, n_match, color="#d23737", lw=1.3, label="matches")
    ax.set_ylabel("count"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Feature & surviving-match count", fontsize=10)

    ax = axs[1]; shade(ax)
    ax.plot(idx, ratio, color="#d23737", lw=1.3)
    ax.set_ylabel("matches / features"); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
    ax.set_title("Match ratio (frontend confidence) - flat here = dropouts are NOT a match failure", fontsize=10)

    ax = axs[2]; shade(ax)
    ax.plot(idx, disp, color="#3a3a3a", lw=1.1)
    ax.set_ylabel("median displacement [px]"); ax.set_xlabel("frame index"); ax.grid(True, alpha=0.3)
    ax.set_title("Apparent feature motion (fast motion / blur proxy)", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
