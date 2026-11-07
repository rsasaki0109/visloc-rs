#!/usr/bin/env python3
"""Refine many LaMAria recordings at once: a 4-DoF pose graph over every recording's keyframes, joined by the cross
edges of lamaria_xs_match.py.

    python scripts/lamaria_xs_graph.py WORK OUT SEQ [SEQ ...] --edges EDGES.npy [EDGES.npy ...]

The nodes are each recording's pose graph keyframes (ba/features.npz, 2 Hz), with x, y, z and the heading about the
IMU's most horizontal axis; roll and pitch stay the recording's own. Odometry edges join a recording's consecutive
keyframes (the trajectory's relative motion in the earlier keyframe's heading frame, sigma 5 mm + --sig-odo-t of the
step, and --sig-odo-yaw degrees). Cross edges place a keyframe of one recording relative to a keyframe of another
(sigmas --sig-x-t, --sig-x-yaw), those with --min-edge-inliers PnP inliers or more. GPS only gives the start: each
trajectory is placed by a rough 2D fit to its fixes. The solve:
1. each recording as a rigid whole (x, y, z, heading) against the cross edges (soft L1), the first one fixed;
2. Gauss-Newton on the sparse normal equations, Cauchy weights (--cauchy sigmas) on the cross edges;
3. --vce rounds of per-recording odometry weights from how far each recording had to bend (variance components,
   no ground truth): the sigmas scale by the root of the recording's mean squared odometry residual over the median
   one (clipped to 0.2..10), and step 2 runs again.
Each trajectory is then corrected between its keyframes by the interpolated change of heading and position and
written as OUT/SEQ.tum (the input's timestamps). The defaults are submission 8's settings, chosen on the six
training recordings solved together with the 49 test walks (docs/lamaria_multisession.md); on a sparse graph of
the training recordings alone they do worse than --sig-odo-t 0.01 --sig-x-t 0.3 --sig-x-yaw 0.5 --vce 0
--min-edge-inliers 100.
"""
import argparse
import os

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, identity
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation as R

from lamaria_xs_common import gps_fit, gps_origin, heading_axis, load_gps, load_traj, rz, wrap, yaw_of


def load_recording(work, seq):
    """A recording's trajectory, its pose graph keyframes and its GPS fixes."""
    w = f"{work}/{seq}"
    kf_ns = np.load(f"{w}/ba/features.npz")["kf_ns"]
    traj, tns = load_traj(w)
    idx = np.searchsorted(tns, kf_ns)
    assert np.all(tns[np.minimum(idx, len(tns) - 1)] == kf_ns), f"{seq}: keyframes without a pose"
    rot = R.from_quat(traj[:, 4:8]).as_matrix()
    return {"kf_ns": kf_ns, "traj": traj, "tns": tns, "p": traj[idx, 1:4], "rot": rot[idx], "rot_all": rot,
            "gps": load_gps(f"{w}/gps.csv", tns[0], tns[-1])}


class Graph:
    """The edges (i, j, measured relative pose, sigmas), the odometry ones first, and each node's recording."""

    def __init__(self, recs, edges, axis, sig_odo_t, sig_odo_yaw, sig_x_t, sig_x_yaw):
        self.names = list(recs)
        self.base, n = {}, 0
        for s in self.names:
            self.base[s] = n
            n += len(recs[s]["kf_ns"])
        self.n = n
        I, J, DP, DPSI, ST, SY = [], [], [], [], [], []
        for s, d in recs.items():
            psi = yaw_of(d["rot"], axis)
            for k in range(len(d["kf_ns"]) - 1):
                dp = rz(psi[k]).T @ (d["p"][k + 1] - d["p"][k])
                I.append(self.base[s] + k); J.append(self.base[s] + k + 1); DP.append(dp)
                DPSI.append(wrap(psi[k + 1] - psi[k]))
                ST.append(0.005 + sig_odo_t * np.linalg.norm(dp)); SY.append(np.radians(sig_odo_yaw))
        self.n_odo = len(I)
        for sA, ka, sB, kb, dp, dpsi, _ in edges:
            I.append(self.base[sA] + ka); J.append(self.base[sB] + kb); DP.append(dp); DPSI.append(dpsi)
            ST.append(sig_x_t); SY.append(np.radians(sig_x_yaw))
        self.I, self.J, self.DP, self.DPSI, self.ST, self.SY = map(np.array, (I, J, DP, DPSI, ST, SY))
        self.m = len(self.I)
        self.seq_of = np.zeros(n, int)
        for q, s in enumerate(self.names):
            self.seq_of[self.base[s]:self.base[s] + len(recs[s]["kf_ns"])] = q
        self.has_x = np.zeros(len(self.names), bool)
        for sA, _, sB, *_ in edges:
            self.has_x[self.names.index(sA)] = self.has_x[self.names.index(sB)] = True

    def residuals(self, x):
        """Per edge: the translation residual in i's heading frame (3) and the heading residual, both over their
        sigmas; and the Jacobian pieces (cos, sin of i's heading, d loc / d psi_i)."""
        pi, pj, si, sj = x[self.I, :3], x[self.J, :3], x[self.I, 3], x[self.J, 3]
        c, s_ = np.cos(si), np.sin(si)
        d = pj - pi
        loc = np.c_[c * d[:, 0] + s_ * d[:, 1], -s_ * d[:, 0] + c * d[:, 1], d[:, 2]]
        dloc = np.c_[-s_ * d[:, 0] + c * d[:, 1], -c * d[:, 0] - s_ * d[:, 1], np.zeros(len(d))]
        return (loc - self.DP) / self.ST[:, None], wrap(sj - si - self.DPSI) / self.SY, c, s_, dloc

    def cross_mismatch(self, x):
        """The cross edges' translation (m) and heading (deg) mismatch."""
        no = self.n_odo
        rt, ry, *_ = self.residuals(x)
        return np.linalg.norm(rt[no:] * self.ST[no:, None], axis=1), np.degrees(np.abs(ry[no:] * self.SY[no:]))


def initial_state(recs, axis):
    """Each recording's keyframes placed by its rough GPS fit: (n, 4) x, y, z, heading."""
    lat0, lon0 = gps_origin([d["gps"] for d in recs.values()])
    rows = []
    for d in recs.values():
        r2, t2, _ = gps_fit(d["traj"], d["tns"], d["gps"], lat0, lon0)
        th0 = np.arctan2(r2[1, 0], r2[0, 0])
        x = np.zeros((len(d["kf_ns"]), 4))
        x[:, :3] = d["p"] @ rz(th0).T + np.r_[t2, 0]
        x[:, 3] = yaw_of(d["rot"], axis) + th0
        rows.append(x)
    return np.concatenate(rows)


def rigid_placement(g, x0):
    """Each recording moved as a whole (x, y, z, heading about the origin) to meet the cross edges; the first stays."""
    no = g.n_odo

    def rigid(v):
        v = np.r_[np.zeros(4), v].reshape(-1, 4)
        off = v[g.seq_of]
        c, s_ = np.cos(off[:, 3]), np.sin(off[:, 3])
        xx = x0.copy()
        xx[:, 0], xx[:, 1] = c * x0[:, 0] - s_ * x0[:, 1] + off[:, 0], s_ * x0[:, 0] + c * x0[:, 1] + off[:, 1]
        xx[:, 2] += off[:, 2]
        xx[:, 3] += off[:, 3]
        return xx

    def res(v):
        rt, ry, *_ = g.residuals(rigid(v))
        return np.c_[rt[no:] * g.ST[no:, None] / 1.0, ry[no:] * g.SY[no:] / np.radians(2.0)].ravel()

    sol = least_squares(res, np.zeros(4 * (len(g.names) - 1)), loss="soft_l1", f_scale=1.0)
    return rigid(sol.x)


def gauss_newton(g, x, iters=15, cauchy=3.0):
    """The whole graph by Gauss-Newton; the cross edges' weights 1 / (1 + (r / cauchy)^2) of their normalized
    residual r, the first node held. Returns the nodes, the edge weights and the last iteration."""
    m, n, no, I, J = g.m, g.n, g.n_odo, g.I, g.J
    anchor = x[0].copy()
    x = x.copy()
    wt = np.ones(m)
    for it in range(iters):
        rt, ry, c, s_, dloc = g.residuals(x)
        if m > no:
            rn = np.sqrt((rt[no:] ** 2).sum(1) + ry[no:] ** 2)
            wt[no:] = 1 / (1 + (rn / cauchy) ** 2)
        rows, cols, vals = [], [], []
        e = np.arange(m)
        inv = 1 / g.ST
        # translation rows 3e+k: d loc_k / d p_j = Rz^T row k, d / d p_i = -that, d / d psi_i = dloc_k
        rzt = [np.c_[c, s_, np.zeros(m)], np.c_[-s_, c, np.zeros(m)], np.c_[np.zeros(m), np.zeros(m), np.ones(m)]]
        for k in range(3):
            for q in range(3):
                rows += [3 * e + k, 3 * e + k]
                cols += [4 * J + q, 4 * I + q]
                vals += [rzt[k][:, q] * inv, -rzt[k][:, q] * inv]
            rows.append(3 * e + k); cols.append(4 * I + 3); vals.append(dloc[:, k] * inv)
        rows += [3 * m + e, 3 * m + e]; cols += [4 * J + 3, 4 * I + 3]; vals += [1 / g.SY, -1 / g.SY]
        for q in range(4):
            rows.append(np.array([4 * m + q])); cols.append(np.array([q])); vals.append(np.array([1e3]))
        Jm = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(4 * m + 4, 4 * n)).tocsr()
        r = np.concatenate([rt.ravel(), ry, (x[0] - anchor) * 1e3])
        w = np.concatenate([np.repeat(wt, 3), wt, np.ones(4)])
        JtW = Jm.T.multiply(w).tocsr()
        H = (JtW @ Jm + 1e-9 * identity(4 * n)).tocsc()
        dx = spsolve(H, -(JtW @ r))
        x = x + dx.reshape(-1, 4)
        if np.abs(dx).max() < 1e-6:
            break
    return x, wt, it


def solve(g, x0, vce=1, iters=15, cauchy=3.0, log=print):
    """Rigid placement, then Gauss-Newton with `vce` rounds of per-recording odometry weights.

    Returns the nodes, the edge weights, the last iteration and each recording's odometry sigma scale."""
    no = g.n_odo
    if g.m > no and len(g.names) > 1:
        x0 = rigid_placement(g, x0)
        mt, my = g.cross_mismatch(x0)
        log(f"rigid placement: cross edges {np.median(mt):.2f} m, {np.median(my):.2f} deg at the median")
    ST0, SY0 = g.ST.copy(), g.SY.copy()
    odo_seq = g.seq_of[g.I[:no]]
    scale = np.ones(len(g.names))
    for v in range(vce + 1):
        g.ST[:no], g.SY[:no] = ST0[:no] * scale[odo_seq], SY0[:no] * scale[odo_seq]
        x, wt, it = gauss_newton(g, x0, iters, cauchy)
        if v == vce or not g.has_x.any():
            break
        rt, ry, *_ = g.residuals(x)
        chi = (rt[:no] ** 2).sum(1) + ry[:no] ** 2
        var = np.array([chi[odo_seq == q].mean() for q in range(len(g.names))])
        ref = np.median(var[g.has_x])
        scale = np.where(g.has_x, np.clip(scale * np.sqrt(var / ref), 0.2, 10.0), scale)
        log(f"  odometry scale after round {v + 1}: "
            + ", ".join(f"{s[9:]} {scale[q]:.2f}" for q, s in enumerate(g.names)))
    return x, wt, it, scale


def corrected(d, xs, axis):
    """The recording's whole trajectory (TUM rows) moved by its keyframes' change of heading and position, both
    interpolated in time between the keyframes; and the heading change at the keyframes."""
    psi_old = yaw_of(d["rot"], axis)
    dpsi = np.unwrap(wrap(xs[:, 3] - psi_old))
    tcorr = xs[:, :3] - np.einsum("kij,kj->ki", rz(dpsi), d["p"])
    tt, kt = d["tns"].astype(float), d["kf_ns"].astype(float)
    dps = np.interp(tt, kt, dpsi)
    tc = np.stack([np.interp(tt, kt, tcorr[:, i]) for i in range(3)], 1)
    rzz = rz(dps)
    pos = np.einsum("nij,nj->ni", rzz, d["traj"][:, 1:4]) + tc
    rot = np.einsum("nij,njk->nik", rzz, d["rot_all"])
    return np.c_[d["traj"][:, 0], pos, R.from_matrix(rot).as_quat()], dpsi


def load_edges(paths, min_inliers=0, exclude=()):
    """Cross edges (recording A, keyframe, recording B, keyframe, B in A's heading frame (3), heading, inliers) of
    one or more lamaria_xs_match.py runs, those with min_inliers or more and neither recording excluded."""
    edges = [tuple(e) for f in paths for e in np.load(f, allow_pickle=True)]
    return [e for e in edges if e[6] >= min_inliers and e[0] not in exclude and e[2] not in exclude]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("work", help="the recordings' work directories, WORK/SEQ")
    p.add_argument("out")
    p.add_argument("seqs", nargs="+")
    p.add_argument("--edges", nargs="+", required=True, help="lamaria_xs_match.py's edges.npy, one or more")
    p.add_argument("--min-edge-inliers", type=int, default=30)
    p.add_argument("--edge-exclude", default="", help="recordings whose cross edges are left out, comma-separated")
    p.add_argument("--sig-odo-yaw", type=float, default=0.005, help="deg per keyframe step")
    p.add_argument("--sig-odo-t", type=float, default=0.04, help="fraction of the step (plus 5 mm)")
    p.add_argument("--sig-x-t", type=float, default=0.15, help="m")
    p.add_argument("--sig-x-yaw", type=float, default=0.25, help="deg")
    p.add_argument("--vce", type=int, default=1, help="rounds of per-recording odometry weights")
    p.add_argument("--iters", type=int, default=15, help="Gauss-Newton iterations at most")
    p.add_argument("--cauchy", type=float, default=3.0, help="the cross edges' Cauchy scale, in sigmas")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    recs = {}
    for s in a.seqs:
        recs[s] = load_recording(a.work, s)
        print(f"{s}: {len(recs[s]['kf_ns'])} keyframes, {len(recs[s]['gps'])} fixes")
    axis, vert = heading_axis([d["rot"] for d in recs.values()])
    print(f"world-z component of the IMU axes x, y, z: {vert.round(2)}; heading from axis {'xyz'[axis]}")
    excl = set(a.edge_exclude.split(",")) if a.edge_exclude else set()
    edges = load_edges(a.edges, a.min_edge_inliers, excl)
    print(f"{len(edges)} cross edges kept (inliers >= {a.min_edge_inliers}"
          + (f", without {sorted(excl)}" if excl else "") + ")")
    g = Graph(recs, edges, axis, a.sig_odo_t, a.sig_odo_yaw, a.sig_x_t, a.sig_x_yaw)
    x, wt, it, _ = solve(g, initial_state(recs, axis), a.vce, a.iters, a.cauchy)
    if g.m > g.n_odo:
        mt, my = g.cross_mismatch(x)
        print(f"solved in {it + 1} iterations: cross edges {np.median(mt):.2f} m, {np.median(my):.2f} deg at the "
              f"median; {np.mean(wt[g.n_odo:] < 0.5) * 100:.0f} % of them down-weighted below a half")
    for s, d in recs.items():
        out, dpsi = corrected(d, x[g.base[s]:g.base[s] + len(d["kf_ns"])], axis)
        np.savetxt(f"{a.out}/{s}.tum", out, fmt="%.9f", header="timestamp tx ty tz qx qy qz qw")
        ne = sum(1 for e in edges if e[0] == s or e[2] == s)
        print(f"{s}: {ne} cross edges, heading change {np.degrees(dpsi.min() - dpsi.mean()):+.2f} .. "
              f"{np.degrees(dpsi.max() - dpsi.mean()):+.2f} deg about its mean")


if __name__ == "__main__":
    main()
