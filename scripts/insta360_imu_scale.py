#!/usr/bin/env python3
"""Metric scale of a 360-camera SfM model from the camera's own IMU.

    insta360_imu_scale.py scene SCENE_DIR OUT_JSON
    insta360_imu_scale.py summary RESULT_JSON...

Re-implements the per-run visual-inertial linear alignment of Cuttini et al.,
arXiv 2610.35554, on the OverMaps360_50 layout: SCENE_DIR holds the
perspective model's `images.bin` (COLMAP 3.12+ binary,
`colmap_perspective/sparse/0`), `insta360_imu.csv` and
`smartphone_poses_*.csv`. Docs: docs/insta360_imu_scale.md.

1. Frames: images `<2 * i + lens>_perspective_<view>.jpg`, view 0 of each
   fisheye lens; i is the frame's index in time.
2. IMU units: the dataset CSV holds raw sensor counts. The gyro is taken as
   16.4 counts per deg/s (refined below), the accelerometer as the nominal
   sensitivity (1000, 1024, 2048 or 4096 counts per g) closest to the mean
   specific force over 20 s windows.
3. Timing t_i = a i + b and the camera-from-IMU rotation R_CB: a coarse grid
   on the frame-to-frame rotation angles, a finer grid within +-0.4% of a,
   then Nelder-Mead on the median residual of the Kabsch fit
   r_C = R_CB r_B. The gyro scale is refined from the same fit. Drift check:
   b re-estimated on the first and last quarters.
4. Scale: per run of frames at most 1.2 s apart (at least 8 intervals),
   unknowns s, g, b_a and one velocity per frame, with IMU preintegration
   using R_WB from the model propagated by the gyro:
     s (C_{k+1} - C_k) - v_k dt - g dt^2 / 2 + J_k b_a = dp_k
     v_{k+1} - v_k - g dt + K_k b_a = dv_k
   Intervals with residuals above five times the median are dropped and the
   run refitted; runs with |g| outside [9.3, 10.3] m/s^2 are discarded; the
   scale is the mean over runs. Added to the paper: runs also break where
   the model's speed exceeds three times its median (a misregistered frame).
   The joint variant fits one s, g and b_a to all runs.
5. Reference: the ARKit scale of the model, a Sim(3) from the model's phone
   images to the ARKit poses (and per quarter of the phone path).
"""

from __future__ import annotations

import csv
import glob
import json
import os
import struct
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

GYRO_COUNTS_PER_DPS = 16.4
ACCEL_COUNTS_PER_G = (1000, 1024, 2048, 4096)
STANDARD_G = 9.80665
MAX_GAP_S = 1.2
MIN_INTERVALS = 8
G_RANGE = (9.3, 10.3)


def rotation(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def read_images_bin(path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Image name -> (world-from-camera rotation, camera centre)."""
    out = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            _, qw, qx, qy, qz, tx, ty, tz, _ = struct.unpack("<I7dI", f.read(64))
            name = bytearray()
            while (c := f.read(1)) != b"\0":
                name += c
            (npts,) = struct.unpack("<Q", f.read(8))
            f.seek(24 * npts, 1)
            r = rotation(np.array([qw, qx, qy, qz]))
            out[name.decode()] = (r.T, -r.T @ np.array([tx, ty, tz]))
    return out


def lens_frames(images) -> tuple[bool, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Per lens: frame time index, world-from-camera rotations, centres.

    Frame numbers alternate between the two lenses when consecutive frames
    look 180 degrees apart; otherwise every frame is one lens."""
    per = {}
    for name, pose in images.items():
        if "_perspective_" in name:
            frame, _, view = name[: -len(".jpg")].split("_")
            if int(view) == 0:
                per[int(frame)] = pose
    frames = sorted(per)
    angles = [
        Rot.from_matrix(per[f][0].T @ per[f + 1][0]).magnitude()
        for f in frames
        if f % 2 == 0 and f + 1 in per
    ]
    interleaved = len(angles) > 10 and np.degrees(np.median(angles)) > 90
    lenses = {}
    for lens in (0, 1) if interleaved else (0,):
        sel = [f for f in frames if not interleaved or f % 2 == lens]
        lenses[lens] = (
            np.array([f // 2 if interleaved else f for f in sel]),
            np.array([per[f][0] for f in sel]),
            np.array([per[f][1] for f in sel]),
        )
    return interleaved, lenses


def umeyama(x: np.ndarray, y: np.ndarray) -> float:
    """Scale of the Sim(3) taking points x onto y."""
    xc, yc = x - x.mean(0), y - y.mean(0)
    u, s, vt = np.linalg.svd(yc.T @ xc / len(x))
    d = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    return float(np.trace(np.diag(s) @ d) / (xc**2).sum(1).mean())


def arkit_scale(images, poses_csv: str) -> tuple[int, float, list[float]]:
    with open(poses_csv) as f:
        ark = {
            row["imageName"]: np.array([float(row[k]) for k in ("pos_x", "pos_y", "pos_z")])
            for row in csv.DictReader(f)
        }
    common = sorted(set(ark) & set(images))
    x = np.array([images[n][1] for n in common])
    y = np.array([ark[n] for n in common])
    q = len(common) // 4
    quarters = [umeyama(x[i * q : (i + 1) * q], y[i * q : (i + 1) * q]) for i in range(4)]
    return len(common), umeyama(x, y), quarters


def accel_counts_per_g(acc: np.ndarray, window: int = 20000) -> tuple[int, float]:
    """The nominal sensitivity closest to the mean specific force over 20 s."""
    m = len(acc) // window
    one_g = float(np.median(np.linalg.norm(acc[: m * window].reshape(m, window, 3).mean(1), axis=1)))
    return min(ACCEL_COUNTS_PER_G, key=lambda v: abs(np.log(one_g / v))), one_g


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a.T
    bx, by, bz, bw = b.T
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=1,
    )


class GyroIntegral:
    """Body rotation P(t) integrated from the gyro (rad/s, midpoint rule);
    the rotation from ti to tj is P(ti)^-1 P(tj)."""

    def __init__(self, t: np.ndarray, gyro: np.ndarray):
        self.t = t
        self.w = gyro
        mid = 0.5 * (gyro[:-1] + gyro[1:])
        q = np.vstack([[0, 0, 0, 1.0], Rot.from_rotvec(mid * np.diff(t)[:, None]).as_quat()])
        step = 1
        while step < len(q):  # inclusive scan of the quaternion products
            q[step:] = _qmul(q[:-step], q[step:])
            step *= 2
        self.P = q

    def at(self, tq: np.ndarray) -> Rot:
        k = np.clip(np.searchsorted(self.t, tq, side="right") - 1, 0, len(self.t) - 2)
        return Rot.from_quat(self.P[k]) * Rot.from_rotvec(self.w[k] * (tq - self.t[k])[..., None])

    def rel(self, ti: np.ndarray, tj: np.ndarray) -> Rot:
        return self.at(ti).inv() * self.at(tj)


def kabsch(rb: np.ndarray, rc: np.ndarray) -> np.ndarray:
    """R minimising sum |rc - R rb|^2."""
    u, _, vt = np.linalg.svd(rb.T @ rc)
    return vt.T @ np.diag([1, 1, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T


def estimate_timing(t, gyro_counts, idx, r_wc):
    """Gyro scale k (rad/s per count), frame timing (a, b), the gyro integral,
    the median rotation residual (deg) and the drift check (ms)."""
    pair = np.flatnonzero(np.diff(idx) <= 3)
    i0, i1 = idx[pair], idx[pair + 1]
    d_rc = Rot.from_matrix(np.einsum("nji,njk->nik", r_wc[pair], r_wc[pair + 1]))
    rc = d_rc.as_rotvec()
    angle_c = np.linalg.norm(rc, axis=1)

    def coarse(g, a_grid, b_grid):
        best = (np.inf, 0.0, 0.0)
        for a in a_grid:
            ti = a * i0[None, :] + b_grid[:, None]
            tj = a * i1[None, :] + b_grid[:, None]
            ok = (ti.min(1) >= t[0]) & (tj.max(1) <= t[-1])
            if not ok.any():
                continue
            qi = g.P[np.searchsorted(t, ti[ok])]
            qj = g.P[np.searchsorted(t, tj[ok])]
            angle_b = 2 * np.arccos(np.clip(np.abs((qi * qj).sum(-1)), 0, 1))
            cost = np.median(np.abs(angle_b - angle_c[None, :]), axis=1)
            j = int(np.argmin(cost))
            if cost[j] < best[0]:
                best = (float(cost[j]), float(a), float(b_grid[ok][j]))
        return best

    def fit(g, a, b, sel=slice(None)):
        ti, tj = a * i0[sel] + b, a * i1[sel] + b
        if ti.min() < t[0] or tj.max() > t[-1]:
            return np.inf, None, 1.0
        rb = g.rel(ti, tj).as_rotvec()
        r = kabsch(rb, rc[sel])
        pred = Rot.from_matrix(r) * Rot.from_rotvec(rb) * Rot.from_matrix(r.T)
        resid = (d_rc[sel].inv() * pred).magnitude()
        return float(np.median(resid)), r, float((rc[sel] * (rb @ r.T)).sum() / (rb**2).sum())

    a0 = (t[-1] - t[0]) / idx[-1]
    a_grid = np.arange(a0 * 0.9, a0 * 1.1, a0 * 0.0012)
    b_grid = np.arange(-30, 30, 0.05)
    candidates = []
    for k in (np.radians(1 / GYRO_COUNTS_PER_DPS), np.radians(1.0)):  # counts, or deg/s
        g = GyroIntegral(t, gyro_counts * k)
        candidates.append((*coarse(g, a_grid, b_grid), k, g))
    _, a, b, k, g = min(candidates, key=lambda c: c[0])
    _, a, b = coarse(g, np.linspace(a * 0.996, a * 1.004, 81), np.arange(b - 0.3, b + 0.3, 0.005))
    for _ in range(3):
        a, b = minimize(
            lambda x: fit(g, x[0], x[1])[0],
            [a, b],
            method="Nelder-Mead",
            options=dict(
                xatol=1e-7,
                fatol=1e-8,
                maxiter=300,
                initial_simplex=[[a, b], [a * 1.0005, b], [a, b + 0.02]],
            ),
        ).x
        residual, _, rotvec_scale = fit(g, a, b)
        if abs(rotvec_scale - 1) < 1e-3:
            break
        k *= rotvec_scale
        g = GyroIntegral(t, gyro_counts * k)
    drift = []
    q = len(pair) // 4
    for sel in (slice(0, q), slice(3 * q, None)):
        bq = minimize(
            lambda x: fit(g, a, x[0], sel)[0],
            [b],
            method="Nelder-Mead",
            options=dict(xatol=1e-5, initial_simplex=[[b], [b + 0.01]]),
        ).x[0]
        drift.append(float((bq - b) * 1e3))
    return k, float(a), float(b), g, float(np.degrees(residual)), drift


def preintegrate(t, acc, g, r_wb0, t0, t1):
    """dp, dv, J, K over [t0, t1] (Euler steps at the IMU rate), world frame."""
    n0, n1 = np.searchsorted(t, [t0, t1], side="right")
    ts = np.r_[t0, t[n0:n1], t1]
    dts = np.diff(ts)
    rw = r_wb0 @ (g.at(t0).inv() * g.at(ts[:-1])).as_matrix()
    aw = np.einsum("nij,nj->ni", rw, acc[n0 - 1 : n1])
    dv = aw * dts[:, None]
    v_before = np.vstack([np.zeros(3), np.cumsum(dv, 0)[:-1]])
    ks = rw * dts[:, None, None]
    k_before = np.concatenate([np.zeros((1, 3, 3)), np.cumsum(ks, 0)[:-1]])
    return (
        (v_before * dts[:, None] + 0.5 * aw * dts[:, None] ** 2).sum(0),
        dv.sum(0),
        (k_before * dts[:, None, None] + 0.5 * rw * dts[:, None, None] ** 2).sum(0),
        ks.sum(0),
        t1 - t0,
    )


def solve(runs, pre, centres, reject=True):
    """Least squares for one s, g, b_a over the given runs; returns (s, |g|)."""
    nv = sum(len(r) for r in runs)
    rows, rhs, vo = [], [], 7
    for r in runs:
        for j, i in enumerate(r[:-1]):
            dp, dv, jm, km, dt = pre[i]
            a = np.zeros((3, 7 + 3 * nv))
            a[:, 0] = centres[i + 1] - centres[i]
            a[:, 1:4] = -0.5 * dt * dt * np.eye(3)
            a[:, 4:7] = jm
            a[:, vo + 3 * j : vo + 3 * j + 3] = -dt * np.eye(3)
            rows.append(a)
            rhs.append(dp)
            a = np.zeros((3, 7 + 3 * nv))
            a[:, 1:4] = -dt * np.eye(3)
            a[:, 4:7] = km
            a[:, vo + 3 * j : vo + 3 * j + 3] = -np.eye(3)
            a[:, vo + 3 * j + 3 : vo + 3 * j + 6] = np.eye(3)
            rows.append(a)
            rhs.append(dv)
        vo += 3 * len(r)
    a, y = np.vstack(rows), np.concatenate(rhs)
    x = np.linalg.lstsq(a, y, rcond=None)[0]
    if reject:
        e = np.linalg.norm((a @ x - y).reshape(-1, 6), axis=1)
        keep = np.repeat(e <= 5 * np.median(e), 6)
        x = np.linalg.lstsq(a[keep], y[keep], rcond=None)[0]
    return float(x[0]), float(np.linalg.norm(x[1:4]))


def align(t, acc, g, a, b, frames):
    """Per-run and joint scale for one lens (R_CB fitted for that lens)."""
    idx, r_wc, centres = frames
    tk = a * idx + b
    p = np.flatnonzero(np.diff(idx) <= 3)
    rc = Rot.from_matrix(np.einsum("nji,njk->nik", r_wc[p], r_wc[p + 1])).as_rotvec()
    r_cb = kabsch(g.rel(tk[p], tk[p + 1]).as_rotvec(), rc)
    r_wb = r_wc @ r_cb
    speed = np.linalg.norm(np.diff(centres, axis=0), axis=1) / np.diff(tk)
    jump = speed > 3 * np.median(speed)
    ok = (np.diff(tk) <= MAX_GAP_S) & (tk[:-1] >= t[0]) & (tk[1:] <= t[-1]) & ~jump
    runs, cur = [], [0]
    for i, step_ok in enumerate(ok):
        if step_ok:
            cur.append(i + 1)
        else:
            runs.append(cur)
            cur = [i + 1]
    runs.append(cur)
    runs = [r for r in runs if len(r) - 1 >= MIN_INTERVALS]
    pre = {i: preintegrate(t, acc, g, r_wb[i], tk[i], tk[i + 1]) for r in runs for i in r[:-1]}
    detail = []
    for r in runs:
        s, gn = solve([r], pre, centres)
        detail.append(dict(t0=float(tk[r[0]]), t1=float(tk[r[-1]]), intervals=len(r) - 1, s=s, g=gn))
    kept = [d for d in detail if G_RANGE[0] <= d["g"] <= G_RANGE[1]]
    s_runs = np.array([d["s"] for d in kept])
    s_joint, g_joint = solve(runs, pre, centres, reject=False) if runs else (np.nan, np.nan)
    return dict(
        speed_jumps=int(jump.sum()),
        runs=len(runs),
        runs_kept=len(kept),
        s_per_run=float(s_runs.mean()) if len(kept) else None,
        s_per_run_std=float(s_runs.std()) if len(kept) else None,
        s_per_run_median=float(np.median(s_runs)) if len(kept) else None,
        g_median=float(np.median([d["g"] for d in kept])) if len(kept) else None,
        s_joint=s_joint,
        g_joint=g_joint,
        detail=detail,
    )


def analyse_scene(scene: str, log=print) -> dict:
    images = read_images_bin(os.path.join(scene, "images.bin"))
    interleaved, lenses = lens_frames(images)
    n_phone, s_ref, quarters = arkit_scale(images, glob.glob(os.path.join(scene, "smartphone_poses_*.csv"))[0])
    log(f"{len(images)} images, lenses interleaved: {interleaved}; ARKit scale {s_ref:.4f} from {n_phone} phone images")
    d = np.loadtxt(os.path.join(scene, "insta360_imu.csv"), delimiter=",", skiprows=1)
    t, gyro_counts, acc_counts = d[:, 0], d[:, 1:4], d[:, 4:7]
    counts_per_g, one_g = accel_counts_per_g(acc_counts)
    acc = acc_counts / counts_per_g * STANDARD_G
    k, a, b, g, residual, drift = estimate_timing(t, gyro_counts, *lenses[0][:2])
    log(
        f"IMU {t[-1] - t[0]:.1f} s; gyro {np.radians(1) / k:.3f} counts per deg/s, accel {counts_per_g} counts per g; "
        f"frames every {a:.6f} s from {b:+.4f} s, rotation residual {residual:.3f} deg, drift {drift[0]:+.1f} / {drift[1]:+.1f} ms"
    )
    out = dict(
        scene=os.path.basename(os.path.normpath(scene)),
        lens_interleaved=bool(interleaved),
        phone_images=n_phone,
        s_ref=s_ref,
        s_ref_quarters=quarters,
        imu_seconds=float(t[-1] - t[0]),
        accel_one_g_counts=one_g,
        accel_counts_per_g=counts_per_g,
        gyro_counts_per_dps=float(np.radians(1) / k),
        frame_period=a,
        frame_offset=b,
        rotation_residual_deg=residual,
        drift_ms=drift,
    )
    for lens, frames in lenses.items():
        r = align(t, acc, g, a, b, frames)
        out[f"lens{lens}"] = r
        if r["s_per_run"] is None:
            log(f"lens {lens}: no run passed the gravity check")
            continue
        log(
            f"lens {lens}: {r['runs_kept']}/{r['runs']} runs ({r['speed_jumps']} speed jumps), "
            f"per-run scale {r['s_per_run']:.4f} ({100 * (r['s_per_run'] / s_ref - 1):+.2f}% vs ARKit), "
            f"|g| {r['g_median']:.3f}; joint {r['s_joint']:.4f} ({100 * (r['s_joint'] / s_ref - 1):+.2f}%)"
        )
    return out


def scene_errors(result: dict) -> dict[str, float] | None:
    """Scale errors (%) against ARKit, averaged over the lenses."""
    lenses = [result[k] for k in ("lens0", "lens1") if k in result and result[k]["s_per_run"] is not None]
    if not lenses:
        return None
    s_ref = result["s_ref"]
    return {
        key: float(np.mean([100 * (r[key] / s_ref - 1) for r in lenses]))
        for key in ("s_per_run", "s_per_run_median", "s_joint")
    }


def summary(paths: list[str]) -> None:
    rows = []
    for path in sorted(paths):
        with open(path) as f:
            r = json.load(f)
        e = scene_errors(r)
        if e is None:
            print(f"{r['scene'][:8]}  no usable run")
            continue
        rows.append(e)
        q = r["s_ref_quarters"]
        print(
            f"{r['scene'][:8]}  phone {r['phone_images']:3d}  ARKit quarters {min(q):.3f}-{max(q):.3f}  "
            f"per-run {e['s_per_run']:+6.2f}%  joint {e['s_joint']:+6.2f}%  "
            f"|g| {r['lens0']['g_median'] or float('nan'):.3f}  gyro {r['gyro_counts_per_dps']:.2f}  "
            f"accel {r['accel_counts_per_g']}  rot {r['rotation_residual_deg']:.3f} deg  "
            f"drift {max(map(abs, r['drift_ms'])):.1f} ms"
        )
    for key, label in (("s_per_run", "per-run"), ("s_per_run_median", "median of runs"), ("s_joint", "joint")):
        v = np.array([e[key] for e in rows])
        print(
            f"{label:15s} {len(v)} scenes: mean {v.mean():+.2f}%  MAE {np.abs(v).mean():.2f}%  "
            f"SD {v.std(ddof=1) if len(v) > 1 else 0:.2f}%  ratio mean {1 + v.mean() / 100:.4f}"
        )


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "scene":
        result = analyse_scene(sys.argv[2], log=lambda m: print(m, flush=True))
        with open(sys.argv[3], "w") as f:
            json.dump(result, f, indent=1)
    elif len(sys.argv) >= 3 and sys.argv[1] == "summary":
        summary(sys.argv[2:])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
