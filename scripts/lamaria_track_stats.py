#!/usr/bin/env python3
"""How often the batch BA's landmarks are found again: track statistics of a solve.

    python scripts/lamaria_track_stats.py --states work/ba1_states.npz --calib calib.json --work work

For every landmark of a solve (`lamaria_batch_ba.py solve --save-states`) it projects the solved point into
both cameras of every keyframe within --window keyframes of its observed span (the solved poses, camera
rotations and focal lengths). A keyframe counts when the point lands inside one of its images, in front of
the camera; of those keyframes, the share where the landmark is observed is the re-observation rate
(docs/lamaria_batch_ba.md#where-the-second-sequences-heading-drift-comes-from: 8.9 % on sequence_3_18 with
SIFT). With --work (the features.npz of the solve) it also counts the images where a keypoint lies within
--near-px of the projection, and where that keypoint's descriptor is as close to one of the landmark's own
as the matches the solve kept (their --similar-pct percentile).

It also prints the track lengths: the keyframes each landmark is observed in, and its span.
No ground truth is used.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lamaria_batch_ba import load_calib  # noqa: E402


def lengths_table(name, v):
    q = np.percentile(v, [50, 75, 90, 99])
    bins = [2, 3, 5, 9, 17, 33, 65, 129, 1 << 30]
    hist = np.histogram(v, bins)[0] / len(v) * 100
    labels = ["2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128", ">128"]
    print(f"{name}: mean {v.mean():.2f}, median {q[0]:.0f}, 75 % {q[1]:.0f}, 90 % {q[2]:.0f}, "
          f"99 % {q[3]:.0f}, max {v.max()}")
    print("  " + ", ".join(f"{lab}: {h:.1f} %" for lab, h in zip(labels, hist)))
    return {"mean": float(v.mean()), "median": float(q[0]), "p75": float(q[1]), "p90": float(q[2]),
            "p99": float(q[3]), "max": int(v.max()), "hist": dict(zip(labels, hist.round(2).tolist()))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", required=True, help="<out>_states.npz of lamaria_batch_ba.py solve --save-states")
    p.add_argument("--calib", required=True, help="the solve's Basalt calibration")
    p.add_argument("--work", default="", help="the solve's work directory (features.npz): nearby keypoints")
    p.add_argument("--window", type=int, default=32, help="keyframes before and after the observed span")
    p.add_argument("--near-px", type=float, default=3.0)
    p.add_argument("--similar-pct", type=float, default=90.0,
                   help="percentile of the descriptor distances within the solve's tracks that counts as similar")
    p.add_argument("--similar-max", type=float, default=0.0,
                   help="a descriptor distance that counts as similar instead (e.g. 300, the SIFT matching's limit)")
    p.add_argument("--chunk", type=int, default=20000, help="landmarks per vectorized batch")
    p.add_argument("--json", default="", help="also write the numbers here")
    a = p.parse_args()

    st = np.load(a.states)
    cams, _ = load_calib(a.calib)
    with open(a.calib) as f:
        res = np.array(json.load(f)["value0"]["resolution"], np.float64)  # [[w, h], [w, h]]
    k0, kf_ns, r_wi, t_wi, r_ic, pts = (st[k] for k in ("k0", "kf_ns", "r_wi", "t_wi", "r_ic", "pts"))
    k0, n = int(k0), len(kf_ns)
    n_seg = int(st["ext_segments"]) if "ext_segments" in st else 1
    seg_of_kf = np.minimum(np.arange(n) * n_seg // n, n_seg - 1)  # as in solve
    foc = st["foc"] if "foc" in st else np.zeros(2 * n_seg)
    t_ic = np.array([c["t_ic"] for c in cams])
    kcal = np.array([[c["K"][0, 0], c["K"][1, 1], c["K"][0, 2], c["K"][1, 2]] for c in cams])
    act = st["active"]
    okf, ocam, olm, oz = st["obs_kf"][act], st["obs_cam"][act], st["obs_lm"][act], st["obs_z"][act]
    used = np.unique(olm)
    print(f"{n} keyframes, {len(used)} landmarks with active observations, {len(okf)} observations", flush=True)

    # track lengths: distinct keyframes per landmark, and the span
    kf_key = np.unique(olm.astype(np.int64) * n + okf)
    kf_count = np.bincount(kf_key // n, minlength=len(pts))[used]
    first = np.full(len(pts), n, np.int64)
    last = np.full(len(pts), -1, np.int64)
    np.minimum.at(first, olm, okf)
    np.maximum.at(last, olm, okf)
    span = (last - first + 1)[used]
    out = {"keyframes": n, "landmarks": int(len(used)), "observations": int(len(okf)),
           "obs_per_keyframe": float(len(okf) / n)}
    out["track_keyframes"] = lengths_table("keyframes observed per landmark", kf_count)
    out["track_span"] = lengths_table("span per landmark (keyframes)", span)

    # world -> camera for every (keyframe, camera)
    rot = np.zeros((n, 2, 3, 3))
    ctr = np.zeros((n, 2, 3))
    kk = np.zeros((n, 2, 4))
    for c in range(2):
        r_c = r_ic[c * n_seg + seg_of_kf]  # (n, 3, 3) imu_from_cam
        rot[:, c] = np.einsum("nij,njk->nik", r_wi, r_c)  # world_from_cam
        ctr[:, c] = np.einsum("nij,j->ni", r_wi, t_ic[c]) + t_wi
        s = np.exp(foc[c * n_seg + seg_of_kf])
        kk[:, c] = kcal[c][None] * np.c_[s, s, np.ones(n), np.ones(n)]

    feats = None
    if a.work:
        f = np.load(os.path.join(a.work, "features.npz"))
        offsets, xy, desc = f["offsets"], f["xy"], f["desc"]  # rows become float32 only where they are compared
        # feature index of every observation: the same pixel in the same image
        img = 2 * (okf.astype(np.int64) + k0) + ocam
        fimg = np.repeat(np.arange(len(offsets) - 1, dtype=np.int64), np.diff(offsets))
        xbits = np.ascontiguousarray(xy[:, 0], np.float32).view(np.uint32).astype(np.int64)
        fkey = (fimg << 32) | xbits  # image and x; y is checked below
        order = np.argsort(fkey, kind="stable")
        okey = (img << 32) | np.ascontiguousarray(oz[:, 0], np.float32).view(np.uint32).astype(np.int64)
        pos = np.minimum(np.searchsorted(fkey[order], okey), len(order) - 1)
        fidx = order[pos]
        ok = (fkey[fidx] == okey) & (xy[fidx, 1].astype(np.float32) == oz[:, 1].astype(np.float32))
        if not ok.all():  # several keypoints at the same x: look them up one by one
            for i in np.flatnonzero(~ok):
                s, e = offsets[img[i]], offsets[img[i] + 1]
                d = np.abs(xy[s:e] - oz[i].astype(np.float32)).sum(1)
                fidx[i] = s + int(np.argmin(d))
        # descriptor distances between the observations of one landmark (consecutive in keyframe order)
        o = np.lexsort((okf, olm))
        same = olm[o][1:] == olm[o][:-1]
        dd = np.linalg.norm(desc[fidx[o][1:][same]].astype(np.float32) - desc[fidx[o][:-1][same]], axis=1)
        thr = a.similar_max or float(np.percentile(dd, a.similar_pct))
        print(f"descriptor distance within the tracks: median {np.median(dd):.3f}, "
              f"{a.similar_pct:.0f}th percentile {np.percentile(dd, a.similar_pct):.3f}; similar below {thr:.3f}",
              flush=True)
        feats = (offsets, xy, desc, fidx, thr)
        out["desc_within_tracks_median"] = float(np.median(dd))
        out["desc_similar_threshold"] = thr

    obs_set = np.unique(olm.astype(np.int64) * n + okf)  # (landmark, keyframe) observed
    obs_img = np.unique((olm.astype(np.int64) * n + okf) * 2 + ocam)  # (landmark, image) observed
    lm_obs_order = np.argsort(olm, kind="stable")
    lm_obs_start = np.r_[0, np.cumsum(np.bincount(olm, minlength=len(pts)))]
    tot = {"kf_visible": 0, "kf_observed": 0, "img_visible": 0, "img_observed": 0, "img_near": 0,
           "img_near_similar": 0}
    for s in range(0, len(used), a.chunk):
        lms = used[s:s + a.chunk]
        lo = np.maximum(first[lms] - a.window, 0)
        hi = np.minimum(last[lms] + a.window, n - 1)
        cnt = hi - lo + 1
        lm_rep = np.repeat(lms, cnt)
        k_rep = np.repeat(lo, cnt) + np.arange(cnt.sum()) - np.repeat(np.r_[0, np.cumsum(cnt)[:-1]], cnt)
        vis_any = np.zeros(len(k_rep), bool)
        for c in range(2):
            pc = np.einsum("nji,nj->ni", rot[k_rep, c], pts[lm_rep] - ctr[k_rep, c])
            z = pc[:, 2]
            u = kk[k_rep, c, 0] * pc[:, 0] / np.maximum(z, 1e-9) + kk[k_rep, c, 2]
            v = kk[k_rep, c, 1] * pc[:, 1] / np.maximum(z, 1e-9) + kk[k_rep, c, 3]
            vis = (z > 0.1) & (u >= 0) & (v >= 0) & (u < res[c, 0]) & (v < res[c, 1])
            vis_any |= vis
            key_img = (lm_rep * n + k_rep) * 2 + c
            seen = np.isin(key_img, obs_img)
            tot["img_visible"] += int(vis.sum())
            tot["img_observed"] += int((vis & seen).sum())
            if feats is not None:  # images where the landmark is visible but not observed: a keypoint nearby?
                offsets, xy, desc, fidx, thr = feats
                for i in np.flatnonzero(vis & ~seen):
                    im = 2 * (k_rep[i] + k0) + c
                    a0, a1 = offsets[im], offsets[im + 1]
                    d2 = (xy[a0:a1, 0] - u[i]) ** 2 + (xy[a0:a1, 1] - v[i]) ** 2
                    near = np.flatnonzero(d2 < a.near_px ** 2)
                    if not len(near):
                        continue
                    tot["img_near"] += 1
                    lm = lm_rep[i]
                    own = fidx[lm_obs_order[lm_obs_start[lm]:lm_obs_start[lm + 1]]]
                    dmin = np.linalg.norm(desc[a0 + near].astype(np.float32)[:, None] - desc[own][None], axis=2).min()
                    tot["img_near_similar"] += int(dmin <= thr)
        tot["kf_visible"] += int(vis_any.sum())
        tot["kf_observed"] += int((vis_any & np.isin(lm_rep * n + k_rep, obs_set)).sum())
        print(f"  {min(s + a.chunk, len(used))}/{len(used)} landmarks", flush=True)
    rate_kf = tot["kf_observed"] / max(tot["kf_visible"], 1)
    rate_img = tot["img_observed"] / max(tot["img_visible"], 1)
    print(f"keyframes within {a.window} of a landmark's span that see it: {tot['kf_visible']}, "
          f"observed in {tot['kf_observed']} ({rate_kf * 100:.1f} %)")
    print(f"images that see it: {tot['img_visible']}, observed in {tot['img_observed']} ({rate_img * 100:.1f} %)")
    out.update(tot, rate_kf=rate_kf, rate_img=rate_img)
    if feats is not None:
        unobs = tot["img_visible"] - tot["img_observed"]
        print(f"of the images that see it but do not observe it, a keypoint within {a.near_px:g} px: "
              f"{tot['img_near'] / max(unobs, 1) * 100:.1f} % ({tot['img_near'] / max(tot['img_visible'], 1) * 100:.1f}"
              f" % of all), with a similar descriptor: {tot['img_near_similar'] / max(tot['img_visible'], 1) * 100:.1f}"
              f" % of all")
        out["near_rate_img"] = tot["img_near"] / max(tot["img_visible"], 1)
        out["near_similar_rate_img"] = tot["img_near_similar"] / max(tot["img_visible"], 1)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
