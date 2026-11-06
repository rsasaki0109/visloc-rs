#!/usr/bin/env python3
"""How well two sets of LaMAria estimates agree with the recordings' own GPS, a check without ground truth.

    python scripts/lamaria_gps_agreement.py OLD_DIR NEW_DIR GPS_DIR SEQ [SEQ ...]

Each estimate (DIR/SEQ.txt, nanoseconds) is fitted to its fixes (GPS_DIR/SEQ.csv from lamaria_gps_extract.py, those
with accuracy up to --max-acc metres within the estimate's time span) by a 2D similarity, robustly (Cauchy weights,
5 m), and the horizontal residuals at the fixes are summarized: per recording the median and the 90th percentile,
old -> new, and over all recordings the median of each and how many got better or worse. The fixes themselves are
off by about 3 m in the city, so this tells metre-level changes apart, not decimetre ones.
"""
import argparse

import numpy as np

from lamaria_gps_steps import local_en


def residuals(est, g, iters=10, cauchy=5.0):
    """Horizontal residuals (m) of the estimate's x, y at the fixes after a robust 2D similarity fit."""
    src = np.stack([np.interp(g["t_ns"], est[:, 0], est[:, 1 + i]) for i in range(2)], 1)
    en = local_en(g["lat"], g["lon"])
    wt = np.ones(len(en))
    for _ in range(iters):
        ms, md = wt @ src / wt.sum(), wt @ en / wt.sum()
        a, b = src - ms, en - md
        u, sv, vt = np.linalg.svd((b * wt[:, None]).T @ a)
        dd = np.diag([1, np.sign(np.linalg.det(u @ vt))])
        r = u @ dd @ vt
        sc = np.trace(np.diag(sv) @ dd) / (wt @ (a ** 2).sum(1))
        res = np.linalg.norm(sc * a @ r.T - b, axis=1)
        wt = 1 / (1 + (res / cauchy) ** 2)
    return res


def compare(old_dir, new_dir, gps_dir, seqs, max_acc=10.0, log=print):
    """Per recording (sequence, fixes, median old, median new, 90 % old, 90 % new)."""
    rows = []
    for s in seqs:
        a, b = np.loadtxt(f"{old_dir}/{s}.txt"), np.loadtxt(f"{new_dir}/{s}.txt")
        g = np.genfromtxt(f"{gps_dir}/{s}.csv", delimiter=",", names=True, dtype=None, encoding=None)
        g = g[(g["accuracy"] <= max_acc) & (g["t_ns"] >= a[0, 0]) & (g["t_ns"] <= a[-1, 0])]
        r0, r1 = residuals(a, g), residuals(b, g)
        rows.append((s, len(g), np.median(r0), np.median(r1), np.percentile(r0, 90), np.percentile(r1, 90)))
        log(f"{s}: {len(g)} fixes, median {rows[-1][2]:.2f} -> {rows[-1][3]:.2f} m, 90 % {rows[-1][4]:.2f} -> "
            f"{rows[-1][5]:.2f} m")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("gps")
    p.add_argument("seqs", nargs="+")
    p.add_argument("--max-acc", type=float, default=10.0, help="m, the fixes kept")
    a = p.parse_args()
    m = np.array([r[2:] for r in compare(a.old, a.new, a.gps, a.seqs, a.max_acc)])
    print(f"all: median of medians {np.median(m[:, 0]):.2f} -> {np.median(m[:, 1]):.2f} m, of 90 % "
          f"{np.median(m[:, 2]):.2f} -> {np.median(m[:, 3]):.2f} m; better median in {(m[:, 1] < m[:, 0]).sum()} of "
          f"{len(m)}, worse by over 0.5 m in {(m[:, 1] > m[:, 0] + 0.5).sum()}")


if __name__ == "__main__":
    main()
