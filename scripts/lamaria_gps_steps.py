#!/usr/bin/env python3
"""Find where a trajectory jumps against GPS and bridge it there, without being told where the rides are.

    python scripts/lamaria_gps_steps.py work/gps/sequence_5_5.csv slam/sequence_5_5.txt bridged/sequence_5_5.txt

A missed or shortened ride on the Moving platform track leaves the trajectory in the right shape but displaced
after the ride, by tens to hundreds of metres. Against the GPS fixes (lamaria_gps_extract.py; those with accuracy up
to --max-acc), the trajectory then fits as pieces that share one rotation and scale but not their translation:
en ~ c R xy + t(t), with t piecewise constant and changing linearly over a ramp between two pieces (the ride). The
ramps are added one at a time: in each piece between the ramps so far, the ramp (centre on a 10 s grid, length up to
--max-ramp) whose whole-model fit (a robust Cauchy least squares, the rotation, scale and levels fitted again) costs
least, if its two levels differ by --min-step metres or more and each side keeps --min-piece seconds and
--min-fixes fixes. The change of level at each ramp, taken back to the trajectory's frame, is spread linearly over
the ramp and added to the positions after it. Only x and y move; the timestamps, z and the rotations stay.

The 25 m used for the test submission is twice the largest step found on the eight training sequences, which have
none (docs/lamaria_gps.md). The output keeps the input's format: nanosecond estimates (slam/<seq>.txt) or TUM with
timestamps in seconds. A sequence without a step, or without enough fixes, is written as it is. No ground truth is
used.
"""
import argparse

import numpy as np

R_EARTH = 6378137.0
LENGTHS = (0, 30, 60, 90, 120, 180, 240)


def local_en(lat_deg, lon_deg):
    """Latitude and longitude in degrees -> east and north in metres around their mean."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.c_[R_EARTH * np.cos(lat.mean()) * (lon - lon.mean()), R_EARTH * (lat - lat.mean())]


def load(gps_path, est_path, max_acc):
    """The trajectory and its x, y at the fixes: (raw rows, ns timestamps?, t (s), fix t (s), fix xy, fix en)."""
    raw = np.loadtxt(est_path, comments="#")
    ns = raw[0, 0] >= 1e7
    te = raw[:, 0].astype(np.int64) if ns else (raw[:, 0] * 1e9).round().astype(np.int64)
    ts = (te - te[0]) / 1e9
    g = np.genfromtxt(gps_path, delimiter=",", names=True, dtype=None, encoding=None)
    g = g[(g["t_ns"] >= te[0]) & (g["t_ns"] <= te[-1]) & (g["accuracy"] <= max_acc)]
    sg = (g["t_ns"] - te[0]) / 1e9
    xy = np.stack([np.interp(sg, ts, raw[:, 1 + i]) for i in range(2)], 1)
    en = local_en(g["lat"], g["lon"]) if len(g) else np.zeros((0, 2))
    return raw, ns, ts, sg, xy, en


def fit(sg, xy, en, ramps, cauchy=5.0, iters=10):
    """en ~ c R xy + t(sg), t piecewise constant with linear ramps; returns c, theta, the levels, residuals, cost."""
    k = len(ramps)
    # Per fix, its weight on each level: 1 on its piece, split linearly inside a ramp.
    lev = np.zeros((len(sg), k + 1))
    lev[np.arange(len(sg)), np.searchsorted(np.array([e for _, e in ramps]), sg, side="right")] = 1.0
    for q, (s0, e0) in enumerate(ramps):
        inside = (sg >= s0) & (sg < e0)
        f = (sg[inside] - s0) / max(e0 - s0, 1e-6)
        lev[inside] = 0
        lev[inside, q], lev[inside, q + 1] = 1 - f, f
    A = np.zeros((2 * len(sg), 2 + 2 * (k + 1)))
    A[0::2, 0], A[0::2, 1] = xy[:, 0], -xy[:, 1]
    A[1::2, 0], A[1::2, 1] = xy[:, 1], xy[:, 0]
    A[0::2, 2::2], A[1::2, 3::2] = lev, lev
    b = en.reshape(-1)
    w = np.ones(len(sg))
    for _ in range(iters):
        sw = np.sqrt(np.repeat(w, 2))
        sol = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)[0]
        res = (A @ sol - b).reshape(-1, 2)
        w = 1 / (1 + (np.linalg.norm(res, axis=1) / cauchy) ** 2)
    cost = np.log1p((np.linalg.norm(res, axis=1) / cauchy) ** 2).sum()
    return np.hypot(*sol[:2]), np.arctan2(sol[1], sol[0]), sol[2:].reshape(-1, 2), res, cost


def find_ramps(sg, xy, en, min_step=15.0, min_piece=60.0, min_fixes=30, max_ramp=300.0, cauchy=5.0, log=print):
    """The ramps (start, end in s), added one at a time while one of --min-step or more lowers the cost."""
    _, _, _, _, cost = fit(sg, xy, en, [], cauchy)
    ramps = []
    lengths = [L for L in LENGTHS if L <= max_ramp]
    while True:
        bounds = [sg[0] - 1] + [v for rp in ramps for v in rp] + [sg[-1] + 1]
        best = None
        biggest = 0.0
        for k in range(len(ramps) + 1):
            lo, hi = bounds[2 * k], bounds[2 * k + 1]
            for mid in np.arange(lo + min_piece, hi - min_piece, 10.0):
                for L in lengths:
                    s0, e0 = mid - L / 2, mid + L / 2
                    left, right = (sg > lo) & (sg < s0), (sg > e0) & (sg < hi)
                    if left.sum() < min_fixes or right.sum() < min_fixes:
                        continue
                    if np.ptp(sg[left]) < min_piece or np.ptp(sg[right]) < min_piece:
                        continue
                    trial = sorted(ramps + [(s0, e0)])
                    _, _, lv, _, tc = fit(sg, xy, en, trial, cauchy, iters=6)
                    q = trial.index((s0, e0))
                    step = np.linalg.norm(lv[q + 1] - lv[q])
                    biggest = max(biggest, step)
                    if step >= min_step and (best is None or tc < best[0]):
                        best = (tc, step, s0, e0)
        if best is None or best[0] >= cost:
            log(f"no further step of {min_step:g} m or more (largest candidate {biggest:.1f} m)")
            return ramps
        tc, step, s0, e0 = best
        log(f"step {step:.1f} m over {s0:.0f}-{e0:.0f} s (cost {cost - tc:.1f} lower)")
        ramps = sorted(ramps + [(s0, e0)])
        cost = tc


def bridge(ts, xy_all, ramps, c, th, levels):
    """The trajectory's x, y with each ramp's change of level (in its frame) spread over the ramp; the moves (m)."""
    rot_inv = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]]) / c
    shift = np.zeros((len(ts), 2))
    moves = []
    for q, (s0, e0) in enumerate(ramps):
        corr = rot_inv @ (levels[q + 1] - levels[q])
        f = np.clip((ts - s0) / max(e0 - s0, 1e-6), 0, 1)
        shift += f[:, None] * corr
        moves.append(float(np.linalg.norm(corr)))
    return xy_all + shift, moves


def write(path, traj, ns):
    if ns:
        with open(path, "w") as f:
            for r in traj:
                f.write(f"{int(round(r[0]))} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} "
                        f"{r[4]:.9f} {r[5]:.9f} {r[6]:.9f} {r[7]:.9f}\n")
    else:
        np.savetxt(path, traj, fmt="%.9f", header="timestamp tx ty tz qx qy qz qw")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("gps", help="lamaria_gps_extract.py's CSV")
    p.add_argument("est", help="the trajectory: slam/<seq>.txt (ns) or TUM (s)")
    p.add_argument("out")
    p.add_argument("--max-acc", type=float, default=10.0, help="m, the fixes kept")
    p.add_argument("--cauchy", type=float, default=5.0, help="m, the scale of the robust loss")
    p.add_argument("--min-step", type=float, default=25.0, help="m between the levels on either side of a ramp")
    p.add_argument("--min-piece", type=float, default=60.0, help="s a piece spans on either side of a ramp")
    p.add_argument("--min-fixes", type=int, default=30, help="fixes on either side of a ramp")
    p.add_argument("--max-ramp", type=float, default=300.0, help="s the longest ride a ramp may spread over")
    a = p.parse_args()

    raw, ns, ts, sg, xy, en = load(a.gps, a.est, a.max_acc)
    if len(sg) < 2 * a.min_fixes:
        print(f"{len(sg)} fixes with accuracy <= {a.max_acc:g} m: too few, the trajectory is left as it is")
        write(a.out, raw, ns)
        return
    c, _, _, res0, _ = fit(sg, xy, en, [], a.cauchy)
    print(f"{len(sg)} fixes (accuracy <= {a.max_acc:g} m), scale {c:.4f}, residual median "
          f"{np.median(np.linalg.norm(res0, axis=1)):.2f} m")
    ramps = find_ramps(sg, xy, en, a.min_step, a.min_piece, a.min_fixes, a.max_ramp, a.cauchy)
    if not ramps:
        print("no step: the trajectory is left as it is")
        write(a.out, raw, ns)
        return
    c, th, levels, res, _ = fit(sg, xy, en, ramps, a.cauchy)
    print(f"bridged: scale {c:.4f}, residual median {np.median(np.linalg.norm(res, axis=1)):.2f} m "
          f"(one piece: {np.median(np.linalg.norm(res0, axis=1)):.2f} m)")
    out = raw.copy()
    out[:, 1:3], moves = bridge(ts, raw[:, 1:3], ramps, c, th, levels)
    for (s0, e0), m in zip(ramps, moves):
        print(f"ride {s0:.0f}-{e0:.0f} s: moved by {m:.1f} m")
    write(a.out, out, ns)


if __name__ == "__main__":
    main()
