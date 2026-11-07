#!/usr/bin/env python3
"""A LaMAria recording's own map for lamaria_xs_match.py, from its fixed trajectory.

    XS_DESC=~/lamaria_xs python scripts/lamaria_xs_map.py WORK SEQ [--steps 1,2,4] [--max-px 2]

The ALIKED keypoints of every matched keyframe are matched by LightGlue to the other camera of the same keyframe
and to the same camera --steps keyframes on, and triangulated with the trajectory's poses (the midpoint of the two
rays). A point is kept in front of both cameras (0.3 m to --max-depth), within --max-px of both observations and
with --min-parallax degrees or more between the rays. Writes WORK/SEQ/xs_map.npz: the keyframe times and IMU poses,
the points, and their observations (keyframe, camera, keypoint index into ba_aliked/features.npz, point). Needs
torch, LightGlue and a GPU.
"""
import argparse
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from lamaria_xs_common import desc_dir, features_path, load_cams, load_traj


def cam_pose(r_wi, t_wi, cam):
    """World-to-camera rotation and translation, the camera centre, and K, of an IMU pose and a camera."""
    rc, tc, K = cam
    r_wc = r_wi @ rc
    p_wc = r_wi @ tc + t_wi
    return r_wc.T, -r_wc.T @ p_wc, p_wc, K


def triangulate(pose_a, pose_b, xa, xb, max_px=2.0, min_parallax=1.0, max_depth=60.0):
    """Midpoint triangulation of matched pixels xa, xb (N, 2) seen by two cam_pose()s: the points (N, 3) and
    which of them to keep."""
    ra, ta, ca_c, Ka = pose_a
    rb, tb, cb_c, Kb = pose_b
    da = (ra.T @ np.linalg.solve(Ka, np.c_[xa, np.ones(len(xa))].T)).T
    db = (rb.T @ np.linalg.solve(Kb, np.c_[xb, np.ones(len(xb))].T)).T
    da /= np.linalg.norm(da, axis=1, keepdims=True)
    db /= np.linalg.norm(db, axis=1, keepdims=True)
    w0 = ca_c - cb_c
    aa, bb, cc = (da * da).sum(1), (da * db).sum(1), (db * db).sum(1)
    dd, ee = (da * w0).sum(1), (db * w0).sum(1)
    den = aa * cc - bb ** 2
    ok = den > 1e-9
    s_ = np.where(ok, (bb * ee - cc * dd) / np.where(ok, den, 1), 0)
    t_ = np.where(ok, (aa * ee - bb * dd) / np.where(ok, den, 1), 0)
    X = 0.5 * ((ca_c + s_[:, None] * da) + (cb_c + t_[:, None] * db))
    par = np.degrees(np.arccos(np.clip(bb, -1, 1)))
    ok &= (s_ > 0.3) & (t_ > 0.3) & (s_ < max_depth) & (par >= min_parallax)
    for (rr, tt, K, x) in ((ra, ta, Ka, xa), (rb, tb, Kb, xb)):
        pc = X @ rr.T + tt
        proj = (pc @ K.T)[:, :2] / np.maximum(pc[:, 2:3], 1e-9)
        ok &= (pc[:, 2] > 0.3) & (np.linalg.norm(proj - x, axis=1) < max_px)
    return X, ok


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("work", help="the recordings' work directories, WORK/SEQ")
    p.add_argument("seq")
    p.add_argument("--steps", default="1,2,4", help="keyframe gaps matched within each camera")
    p.add_argument("--max-px", type=float, default=2.0)
    p.add_argument("--min-parallax", type=float, default=1.0, help="deg")
    p.add_argument("--max-depth", type=float, default=60.0, help="m")
    a = p.parse_args()
    import torch
    from lightglue import LightGlue

    w = f"{a.work}/{a.seq}"
    fpath = features_path(w)
    f = np.load(fpath)
    dpath = f"{desc_dir()}/{a.seq}/aliked_desc.npy"
    if not os.path.exists(dpath):  # the chain's features.npz holds them; memory-map them from a separate file
        os.makedirs(os.path.dirname(dpath), exist_ok=True)
        np.save(dpath, f["desc"])
    desc = np.load(dpath, mmap_mode="r")
    xy, offsets, kf_ns, wh = f["xy"], f["offsets"], f["kf_ns"], f["image_wh"]
    traj, tns = load_traj(w)
    idx = np.searchsorted(tns, kf_ns)
    assert np.all(tns[np.minimum(idx, len(tns) - 1)] == kf_ns), "keyframes without a pose"
    r_wi = R.from_quat(traj[idx, 4:8]).as_matrix()
    t_wi = traj[idx, 1:4]
    cams = load_cams(f"{a.work}/calibrations/{a.seq}_calib_ba.json")

    dev = "cuda"
    lg = LightGlue(features="aliked", filter_threshold=0.1).eval().to(dev)
    cache = {}

    def feats(k, c):
        if (k, c) not in cache:
            if len(cache) > 64:
                cache.pop(next(iter(cache)))
            o0, o1 = offsets[2 * k + c], offsets[2 * k + c + 1]
            whv = wh[2 * k + c]
            cache[(k, c)] = {"keypoints": torch.from_numpy(np.ascontiguousarray(xy[o0:o1], np.float32))[None].to(dev),
                             "descriptors": torch.from_numpy(np.asarray(desc[o0:o1], np.float32))[None].to(dev),
                             "image_size": torch.tensor([[float(whv[0]), float(whv[1])]], device=dev)}
        return cache[(k, c)]

    steps = [int(s) for s in a.steps.split(",")]
    pts, obs = [], []
    t0 = time.time()
    n = len(kf_ns)
    with torch.inference_mode():
        for k in range(n):
            pairs = [((k, 0), (k, 1))] + [((k, c), (k + s, c)) for s in steps for c in (0, 1) if k + s < n]
            for (ka, ca), (kb, cb) in pairs:
                fa, fb = feats(ka, ca), feats(kb, cb)
                if fa["keypoints"].shape[1] < 20 or fb["keypoints"].shape[1] < 20:
                    continue
                m = lg({"image0": fa, "image1": fb})["matches"][0].cpu().numpy()
                if len(m) < 8:
                    continue
                oa, ob = offsets[2 * ka + ca], offsets[2 * kb + cb]
                X, ok = triangulate(cam_pose(r_wi[ka], t_wi[ka], cams[ca]), cam_pose(r_wi[kb], t_wi[kb], cams[cb]),
                                    xy[oa + m[:, 0]].astype(np.float64), xy[ob + m[:, 1]].astype(np.float64),
                                    a.max_px, a.min_parallax, a.max_depth)
                for q in np.flatnonzero(ok):
                    lm = len(pts)
                    pts.append(X[q])
                    obs.append((ka, ca, oa + m[q, 0], lm))
                    obs.append((kb, cb, ob + m[q, 1], lm))
            if k % 500 == 0:
                print(f"  keyframe {k} of {n}: {len(pts)} points ({time.time() - t0:.0f} s)", flush=True)
    obs = np.array(obs, dtype=np.int64)
    np.savez(f"{w}/xs_map.npz", kf_ns=kf_ns, r_wi=r_wi, t_wi=t_wi, pts=np.array(pts), obs_kf=obs[:, 0],
             obs_cam=obs[:, 1], obs_fi=obs[:, 2], obs_lm=obs[:, 3], features=fpath)
    print(f"{a.seq}: {len(pts)} points, {len(obs)} observations from {n} keyframes in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
