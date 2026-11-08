"""Shared pieces of the LaMAria multi-session scripts (lamaria_xs_*.py): headings about the IMU's most horizontal axis,
the recordings' files under a work directory, and the rough GPS placement that proposes the pairs.

A recording's work directory WORK/SEQ holds ba/ba3.tum (or ba/online.tum; TUM, seconds), ba/features.npz (kf_ns: the
pose graph's keyframes), ba_aliked/features.npz (the matched keyframes' ALIKED keypoints), gps.csv
(lamaria_gps_extract.py) and, once built, xs_map.npz (lamaria_xs_map.py); the calibration is
WORK/calibrations/SEQ_calib_ba.json (lamaria_to_basalt_calib.py). The descriptors live apart, as
XS_DESC/SEQ/aliked_desc.npy (float16, memory-mapped; default ~/lamaria_xs).
"""
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

R_EARTH = 6378137.0


def wrap(x):
    """Angles to [-pi, pi)."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def rz(t):
    """Rotations about z by t (any shape) -> (..., 3, 3)."""
    c, s = np.cos(t), np.sin(t)
    out = np.zeros(np.shape(t) + (3, 3))
    out[..., 0, 0], out[..., 0, 1], out[..., 1, 0], out[..., 1, 1], out[..., 2, 2] = c, -s, s, c, 1
    return out


def heading_axis(rotations):
    """The IMU axis that stays closest to horizontal over all recordings' keyframe rotations (world from IMU).

    On Aria it is y: the world-z components of x, y and z average 0.68, 0.27 and 0.42 on the walks."""
    vert = np.mean([np.abs(r[:, 2, :]).mean(0) for r in rotations], axis=0)
    return int(np.argmin(vert)), vert


def yaw_of(rm, axis):
    """Heading of the body's axis `axis`; turning the world about z by d adds d to it."""
    return np.arctan2(rm[..., 1, axis], rm[..., 0, axis])


def desc_dir():
    return os.path.expanduser(os.environ.get("XS_DESC", "~/lamaria_xs"))


def features_path(w):
    """The ALIKED keypoints of a recording's work directory (the chain's pass, its plain run, or the first trial)."""
    return next((q for q in (f"{w}/ba_aliked_pass/features.npz", f"{w}/ba_aliked/features.npz",
                             f"{w}/exp_aliked1k/features.npz") if os.path.exists(q)), None)


def load_traj(w):
    """The recording's fixed trajectory (TUM rows, seconds) and its timestamps in nanoseconds."""
    traj = np.loadtxt(f"{w}/ba/ba3.tum" if os.path.exists(f"{w}/ba/ba3.tum") else f"{w}/ba/online.tum")
    return traj, np.round(traj[:, 0] * 1e9).astype(np.int64)


def load_gps(path, t0_ns, t1_ns, max_acc=10.0):
    """The fixes of lamaria_gps_extract.py's CSV within [t0, t1] and with accuracy up to max_acc metres."""
    g = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    return g[(g["accuracy"] <= max_acc) & (g["t_ns"] >= t0_ns) & (g["t_ns"] <= t1_ns)]


def load_cams(path):
    """Both cameras of a Basalt calibration: (R_imu_cam, p_imu_cam, K) each."""
    cal = json.load(open(path))["value0"]
    cams = []
    for c in range(2):
        e = cal["T_imu_cam"][c]
        intr = cal["intrinsics"][c]["intrinsics"]
        cams.append((R.from_quat([e["qx"], e["qy"], e["qz"], e["qw"]]).as_matrix(),
                     np.array([e["px"], e["py"], e["pz"]]),
                     np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]])))
    return cams


def gps_origin(fixes):
    """A common origin (radians) for the recordings' fixes: the mean of their mean latitudes and longitudes."""
    lat0 = np.radians(np.mean([g["lat"].mean() for g in fixes if len(g)]))
    lon0 = np.radians(np.mean([g["lon"].mean() for g in fixes if len(g)]))
    return lat0, lon0


def gps_fit(traj, tns, g, lat0, lon0, iters=8, scale=5.0):
    """The trajectory's x, y fitted to the fixes by a 2D rotation and translation (Cauchy weights, `scale` metres).

    Returns the rotation (2, 2), the translation and the residuals at the fixes. Rough on purpose: it only says
    which keyframes of two recordings pass the same place."""
    en = np.c_[R_EARTH * np.cos(lat0) * (np.radians(g["lon"]) - lon0), R_EARTH * (np.radians(g["lat"]) - lat0)]
    src = np.stack([np.interp(g["t_ns"], tns, traj[:, 1 + i]) for i in range(2)], 1)
    return rigid2d(src, en, iters, scale)


def rigid2d(src, dst, iters=8, scale=5.0):
    """The 2D rotation (2, 2) and translation taking points src (n, 2) to dst (Cauchy weights, `scale` metres), and
    the residuals."""
    wt = np.ones(len(dst))
    for _ in range(iters):
        ms, md = wt @ src / wt.sum(), wt @ dst / wt.sum()
        u, _, vt = np.linalg.svd(((dst - md) * wt[:, None]).T @ (src - ms))
        r2 = u @ np.diag([1, np.sign(np.linalg.det(u @ vt))]) @ vt
        t2 = md - r2 @ ms
        res = np.linalg.norm(src @ r2.T + t2 - dst, axis=1)
        wt = 1 / (1 + (res / scale) ** 2)
    return r2, t2, res


def nearest_index(sorted_ns, ns):
    """For each of `ns`, the index of the nearest entry of `sorted_ns` (two or more entries)."""
    j = np.clip(np.searchsorted(sorted_ns, ns), 1, len(sorted_ns) - 1)
    return np.where(np.abs(sorted_ns[j - 1] - ns) < np.abs(sorted_ns[j] - ns), j - 1, j)


def map_blocks(z):
    """An own map's observations (lamaria_xs_map.py) grouped by (keyframe, camera), one per keypoint.

    Returns the keypoint and landmark index arrays and {(keyframe, camera): (start, end)} into them."""
    okf, ocam, ofi, olm = z["obs_kf"], z["obs_cam"], z["obs_fi"], z["obs_lm"]
    order = np.lexsort((ofi, ocam, okf))
    okf, ocam, ofi, olm = okf[order], ocam[order], ofi[order], olm[order]
    first = np.r_[True, (np.diff(okf) != 0) | (np.diff(ocam) != 0) | (np.diff(ofi) != 0)]
    okf, ocam, ofi, olm = okf[first], ocam[first], ofi[first], olm[first]
    bounds = np.flatnonzero(np.r_[True, (np.diff(okf) != 0) | (np.diff(ocam) != 0), True])
    return ofi, olm, {(okf[b0], ocam[b0]): (b0, b1) for b0, b1 in zip(bounds[:-1], bounds[1:])}
