#!/usr/bin/env python3
"""Scan many SLAM / VO runs and rank them in one table: coverage, tracking
continuity, accuracy, jitter - across sequences, frontends, and configs.

This is the fleet view on top of ``analyze_slam_trajectory.py`` (whose
per-run metrics it reuses). The motivating question is the practical one:
"which frontend / config actually tracks best on this sequence?" - the kind of
thing that otherwise takes manual archaeology across dozens of ``target/*``
directories. It glob-scans ``slam_errors.csv`` files, pulls the frontend and
tracking-success rate from each run's sibling ``summary.txt`` when present, and
emits a sorted Markdown table (+ optional CSV and a coverage-vs-ATE scatter).

Usage:
    # rank every run under target/, best rigid ATE first, needs >=60 frames
    python3 scripts/compare_slam_runs.py --root target --min-frames 60 --sort rigid_ate

    # only EuRoC MH_01 runs, write CSV + scatter
    python3 scripts/compare_slam_runs.py --root target --filter MH_01 \\
        --output-csv target/slam_run_comparison.csv \\
        --scatter docs/assets/slam_coverage_vs_ate.png

Analysis tool, not part of CI.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_slam_trajectory import analyze  # noqa: E402

SEQ_RE = re.compile(r"(MH_0\d|V[12]_0\d)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("target"), help="directory to scan for slam_errors.csv")
    p.add_argument("--filter", type=str, default=None, help="only runs whose path contains this substring")
    p.add_argument("--min-frames", type=int, default=40, help="skip runs with fewer tracked frames")
    p.add_argument("--sort", choices=["rigid_ate", "coverage", "continuity", "frames"], default="rigid_ate")
    p.add_argument("--top", type=int, default=40, help="rows to print")
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--scatter", type=Path, default=None, help="write a coverage-vs-rigid-ATE scatter PNG")
    return p.parse_args()


def _summary_text(run_dir: Path) -> str:
    summary = run_dir / "summary.txt"
    return summary.read_text(errors="ignore") if summary.exists() else ""


def detect_frontend(run_dir: Path) -> str:
    text = _summary_text(run_dir)
    if not text:
        return "?"
    if re.search(r"superpoint_features_dir=(?!None)\S", text):
        return "superpoint"
    m = re.search(r"feature_extractor=(\w+)", text)
    return m.group(1) if m else "?"


def detect_sequence(run_dir: Path, path_str: str) -> str:
    # path naming first (MH_01 / V1_01 ...), then the run's own euroc_dir.
    m = SEQ_RE.search(path_str)
    if m:
        return m.group(1)
    m = re.search(r"euroc_dir=\S*?(MH_0\d|V[12]_0\d)", _summary_text(run_dir))
    return m.group(1) if m else "?"


def collect(args):
    rows = []
    for csv_path in sorted(args.root.rglob("slam_errors.csv")):
        rel = str(csv_path)
        try:
            a = analyze(csv_path, stationary_thresh=0.01)
        except Exception:
            continue
        if a["n"] < args.min_frames:
            continue
        seq = detect_sequence(csv_path.parent, rel)
        # --filter matches the path or the detected sequence (so 'MH_01' also
        # catches runs whose dir is named 'mh01' but whose euroc_dir is MH_01).
        if args.filter and args.filter not in rel and args.filter not in seq:
            continue
        rows.append({
            "run": csv_path.parent.relative_to(args.root).as_posix(),
            "seq": seq,
            "frontend": detect_frontend(csv_path.parent),
            "frames": a["n"],
            "coverage": a["coverage"],
            "longest_cont": a["longest_contiguous"],
            "max_gap": a["max_gap"],
            "rigid_ate": a["rmse_rigid"],
            "sim_ate": a["rmse_sim"],
            "jitter": a["jitter_still"],
            "gt_path": a["gt_path"],
        })
    return rows


def sort_key(args):
    if args.sort == "rigid_ate":
        return lambda r: r["rigid_ate"]
    if args.sort == "coverage":
        return lambda r: -r["coverage"]
    if args.sort == "continuity":
        return lambda r: -r["longest_cont"]
    return lambda r: -r["frames"]


def fmt_table(rows):
    head = "| run | seq | frontend | frames | cov % | cont | maxgap | rigid ATE | sim ATE | still-jit | gt path |"
    sep = "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines = [head, sep]
    for r in rows:
        jit = f"{r['jitter'] * 100:.1f}cm" if r["jitter"] == r["jitter"] else "-"
        lines.append(
            f"| {r['run']} | {r['seq']} | {r['frontend']} | {r['frames']} | "
            f"{r['coverage'] * 100:.0f} | {r['longest_cont']} | {r['max_gap']} | "
            f"{r['rigid_ate'] * 100:.1f}cm | {r['sim_ate'] * 100:.1f}cm | {jit} | {r['gt_path']:.2f}m |"
        )
    return "\n".join(lines)


def best_per_sequence(rows, min_cov=0.4):
    by_seq = {}
    for r in rows:
        if r["coverage"] < min_cov:
            continue
        cur = by_seq.get(r["seq"])
        if cur is None or r["rigid_ate"] < cur["rigid_ate"]:
            by_seq[r["seq"]] = r
    return by_seq


def main() -> int:
    args = parse_args()
    rows = collect(args)
    if not rows:
        print("no runs matched", file=sys.stderr)
        return 1
    rows.sort(key=sort_key(args))

    print(f"# SLAM run comparison ({len(rows)} runs, sorted by {args.sort})\n")
    print(fmt_table(rows[: args.top]))

    best = best_per_sequence(rows)
    if best:
        print(f"\n## Best run per sequence (coverage >= 40 %, lowest rigid ATE)\n")
        print("| seq | run | frontend | cov % | cont | rigid ATE |")
        print("| --- | --- | --- | ---: | ---: | ---: |")
        for seq in sorted(best):
            r = best[seq]
            print(f"| {seq} | {r['run']} | {r['frontend']} | {r['coverage'] * 100:.0f} | {r['longest_cont']} | {r['rigid_ate'] * 100:.1f}cm |")

    if args.output_csv:
        import csv as _csv

        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.output_csv}", file=sys.stderr)

    if args.scatter:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            seqs = sorted({r["seq"] for r in rows})
            cmap = plt.get_cmap("tab10")
            color = {s: cmap(i % 10) for i, s in enumerate(seqs)}
            fig, ax = plt.subplots(figsize=(7.5, 5))
            for s in seqs:
                sub = [r for r in rows if r["seq"] == s]
                ax.scatter([r["coverage"] * 100 for r in sub], [r["rigid_ate"] * 100 for r in sub],
                           s=28, color=color[s], label=s, alpha=0.8, edgecolors="none")
            ax.set_xlabel("tracking coverage [%]")
            ax.set_ylabel("rigid ATE [cm]")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3, which="both")
            ax.legend(fontsize=8, title="sequence")
            ax.set_title("SLAM runs: tracking coverage vs accuracy\n(upper-left = accurate but sparse, lower-right = the goal)")
            fig.tight_layout()
            args.scatter.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.scatter, dpi=130)
            print(f"wrote {args.scatter}", file=sys.stderr)
        except ImportError:
            print("matplotlib not available; scatter skipped", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
