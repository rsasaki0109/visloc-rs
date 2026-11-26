#!/usr/bin/env python3
"""Bend a trajectory's slow heading and scale drift toward its recording's GPS, only where the drift is large.

    python scripts/lamaria_gps_bend.py work/gps/sequence_1_2.csv slam/sequence_1_2.txt bent/sequence_1_2.txt

A monocular-inertial trajectory can drift in heading and scale by more than the GPS errs (docs/lamaria_mono.md),
which no single similarity takes out. Here the trajectory's x, y are resampled every --node-s seconds and solved as a
2D pose graph: between two nodes the trajectory's own step, turned by a heading offset and stretched by a scale
offset that each follow a random walk (--heading-rw degrees and --scale-rw per cent per square-root minute), with
--step-sigma metres of slack; at each node the GPS fixes near it (accuracy up to --max-acc) pull with a Cauchy loss of
--cauchy metres. The fixes' frame is fitted first (a robust 2D similarity), so the
solve starts from the trajectory as it is.

The bend is kept only when the drift is large against the GPS's own errors (docs/lamaria_gps.md: 2-8 m at the median,
correlated over about 30 s): when the trajectory's median residual after the similarity is --min-residual metres or
more (and the bend moves some node by --min-move metres or more). On a trajectory that does not drift the bend only
adds the GPS's own errors: ungated it costs the training monos 5 to 13 CP points, while their residuals are 1.8 to
2.9 m; the default 4 m leaves them all as they are and bends every injected drift of 0.5 deg and 0.5 % per minute or
more that reaches it (docs/lamaria_mono.md). Otherwise the trajectory is written as it is. Every
pose moves by the bend of its time (interpolated between nodes): x, y by the node's correction, the rotation by its
heading offset about the vertical. z and the timestamps stay. The output keeps the input's format (nanosecond
estimates or TUM in seconds). No ground truth is used.
"""
import argparse

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from lamaria_gps_steps import load, write


def similarity(src, dst, w):
    """dst ~ c R src + t, weighted: (c, R, t)."""
    ms, md = w @ src / w.sum(), w @ dst / w.sum()
    a, b = src - ms, dst - md
    u, sv, vt = np.linalg.svd((b * w[:, None]).T @ a)
    d = np.diag([1, np.sign(np.linalg.det(u @ vt))])
    r = u @ d @ vt
    c = np.trace(np.diag(sv) @ d) / (w @ (a ** 2).sum(1))
    return c, r, md - c * r @ ms


def bend(ts, xy, sg, en, node_s=5.0, heading_rw=1.0, scale_rw=1.0, step_sigma=0.05, cauchy=5.0):
    """The bend: node times, the nodes' corrected x, y in the GPS frame, their heading offsets (rad), and the
    similarity (c, R, t) that took the trajectory to the GPS frame first."""
    src, w = xy_at(ts, xy, sg), np.ones(len(sg))
    for _ in range(10):  # robust similarity, as lamaria_gps_agreement.residuals
        c, r, t = similarity(src, en, w)
        res = np.linalg.norm(c * src @ r.T + t - en, axis=1)
        w = 1 / (1 + (res / cauchy) ** 2)
    tn = np.arange(ts[0], ts[-1] + 1e-9, node_s)
    pn = (c * xy_at(ts, xy, tn) @ r.T) + t  # the nodes in the GPS frame, before the bend
    n = len(tn)
    dp = np.diff(pn, axis=0)
    # the fixes near each node: the nearest node within half a node gap
    k = np.clip(np.round((sg - ts[0]) / node_s).astype(int), 0, n - 1)
    near = np.abs(tn[k] - sg) <= node_s / 2
    k, gen = k[near], en[near]
    s_h = np.radians(heading_rw) * np.sqrt(node_s / 60.0)
    s_s = scale_rw / 100.0 * np.sqrt(node_s / 60.0)

    # unknowns: x, y per node, then heading offset and log scale per node
    def unpack(v):
        return v[:2 * n].reshape(n, 2), v[2 * n:3 * n], v[3 * n:]

    def fun(v):
        q, h, s = unpack(v)
        cs, sn = np.cos(h[:-1]), np.sin(h[:-1])
        e = np.exp(s[:-1])
        rot = np.c_[cs * dp[:, 0] - sn * dp[:, 1], sn * dp[:, 0] + cs * dp[:, 1]] * e[:, None]
        step = ((q[1:] - q[:-1]) - rot) / step_sigma
        return np.concatenate([step.ravel(), np.diff(h) / s_h, np.diff(s) / s_s, (q[k] - gen).ravel() / cauchy])

    m_step, m_rw = 2 * (n - 1), n - 1
    rows = m_step + 2 * m_rw + 2 * len(k)
    jac = lil_matrix((rows, 4 * n), dtype=bool)
    for i in range(n - 1):
        for d in range(2):
            jac[2 * i + d, [2 * i + d, 2 * (i + 1) + d, 2 * n + i, 3 * n + i]] = True
        jac[m_step + i, [2 * n + i, 2 * n + i + 1]] = True
        jac[m_step + m_rw + i, [3 * n + i, 3 * n + i + 1]] = True
    for j, i in enumerate(k):
        for d in range(2):
            jac[m_step + 2 * m_rw + 2 * j + d, 2 * i + d] = True

    v = np.concatenate([pn.ravel(), np.zeros(2 * n)])
    w = np.ones(len(k))
    for _ in range(6):  # iteratively reweighted: Cauchy weights on the fixes
        def weighted(v, w=w):
            f = fun(v)
            f[m_step + 2 * m_rw:] *= np.repeat(np.sqrt(w), 2)
            return f
        v = least_squares(weighted, v, jac_sparsity=jac, method="trf", x_scale="jac").x
        q = unpack(v)[0]
        rk = np.linalg.norm(q[k] - gen, axis=1)
        w = 1 / (1 + (rk / cauchy) ** 2)
    q, h, _ = unpack(v)
    return tn, pn, q, h, (c, r, t), res


def xy_at(ts, xy, t):
    return np.stack([np.interp(t, ts, xy[:, i]) for i in range(2)], 1)


def apply(raw, ts, tn, pn, q, h, sim):
    """Every pose moved by the bend at its time, back in the trajectory's frame."""
    c, r, t = sim
    out = raw.copy()
    d_gps = xy_at(tn, q - pn, ts)  # the node corrections, in the GPS frame, at each pose's time
    out[:, 1:3] = raw[:, 1:3] + (d_gps @ r) / c  # back through the similarity: R^T d / c
    hh = np.interp(ts, tn, h)
    yaw = Rotation.from_euler("z", hh[:, None])
    out[:, 4:8] = (yaw * Rotation.from_quat(raw[:, 4:8])).as_quat()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("gps", help="lamaria_gps_extract.py's CSV")
    p.add_argument("estimate", help="the trajectory: slam/<seq>.txt (ns) or TUM (s)")
    p.add_argument("out")
    p.add_argument("--max-acc", type=float, default=10.0, help="fixes with this accuracy (m) or better")
    p.add_argument("--node-s", type=float, default=5.0)
    p.add_argument("--heading-rw", type=float, default=1.0, help="degrees per square-root minute")
    p.add_argument("--scale-rw", type=float, default=1.0, help="per cent per square-root minute")
    p.add_argument("--step-sigma", type=float, default=0.05, help="metres of slack per node step")
    p.add_argument("--cauchy", type=float, default=5.0, help="metres")
    p.add_argument("--min-residual", type=float, default=4.0,
                   help="bend only a trajectory whose median residual to the fixes is this or more (m)")
    p.add_argument("--min-move", type=float, default=0.0, help="keep the bend only if it moves a node this far (m)")
    p.add_argument("--min-fixes", type=int, default=60)
    a = p.parse_args()

    raw, ns, ts, sg, _, en = load(a.gps, a.estimate, a.max_acc)
    if len(sg) < a.min_fixes:
        print(f"{len(sg)} fixes: written as it is")
        write(a.out, raw, ns)
        return
    tn, pn, q, h, sim, res = bend(ts, raw[:, 1:3], sg, en, a.node_s, a.heading_rw, a.scale_rw, a.step_sigma, a.cauchy)
    move = np.linalg.norm(q - pn, axis=1).max()
    keep = np.median(res) >= a.min_residual and move >= a.min_move
    print(f"{len(sg)} fixes, median residual {np.median(res):.1f} m, the bend moves up to {move:.1f} m and turns "
          f"{np.degrees(np.ptp(h)):.1f} deg: {'bent' if keep else 'written as it is'}")
    write(a.out, apply(raw, ts, tn, pn, q, h, sim) if keep else raw, ns)


if __name__ == "__main__":
    main()
