#!/usr/bin/env python3
"""Cross edges between LaMAria recordings that pass the same place, for lamaria_xs_graph.py.

    XS_DESC=~/lamaria_xs python scripts/lamaria_xs_match.py WORK OUT SEQ [SEQ ...] [--every 4] [--max-maps 2]

GPS proposes the pairs: each trajectory is placed by a rough 2D fit to its fixes (recordings with fewer than 30
fixes are left out; the --loose ones take fixes up to --loose-acc metres and need 10, and --place moves them by an
earlier run's edges with 100+ inliers to the others, so a second run searches where those edges put them), and
every --every-th matched keyframe of a recording (the query) tries its --max-maps nearest other recordings with a
keyframe within --radius metres (--map-from skips the nearest ones, --offset shifts the queries: both add to an
earlier run; so does --only, which keeps the pairs with one of the recordings it names).
Against a map (lamaria_xs_map.py), the query's ALIKED keypoints, each camera in turn, are matched by LightGlue to
the landmark keypoints of the map's --window nearest keyframes within --yaw degrees of heading (both cameras), and
PnP RANSAC (AP3P, --reproj px, then Levenberg-Marquardt on the inliers) places the query camera in the map; the
camera with more inliers wins, with --min-inliers or more. The edge is the query IMU pose relative to the map
keyframe nearest to it: its position in that keyframe's heading frame and the heading difference, with the inlier
count.

The maps are loaded one at a time, so 55 recordings fit in 11 GB. Resumable: the maps done are listed in
OUT/done.txt and their edges kept in OUT/edges.npy. Needs torch, LightGlue and a GPU.
"""
import argparse
import os
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from lamaria_xs_common import (desc_dir, features_path, gps_fit, gps_origin, heading_axis, load_cams, load_gps,
                               load_traj, map_blocks, nearest_index, rigid2d, rz, wrap, yaw_of)


def load_recording(work, seq, max_acc=10.0):
    """A recording's matched keyframes (keypoints, memory-mapped descriptors, poses), cameras and fixes."""
    w = f"{work}/{seq}"
    f = np.load(features_path(w))
    akf = f["kf_ns"]
    gkf = np.load(f"{w}/ba/features.npz")["kf_ns"]
    traj, tns = load_traj(w)
    i = np.clip(np.searchsorted(tns, akf), 0, len(tns) - 1)
    d = {"akf": akf, "gkf": gkf, "a2g": nearest_index(gkf, akf), "offsets": f["offsets"],
         "wh": f["image_wh"], "xy": f["xy"], "p": traj[i, 1:4], "rot": R.from_quat(traj[i, 4:8]).as_matrix(),
         "traj": traj, "tns": tns, "gps": load_gps(f"{w}/gps.csv", tns[0], tns[-1], max_acc),
         "cams": load_cams(f"{work}/calibrations/{seq}_calib_ba.json"),
         "desc": np.load(f"{desc_dir()}/{seq}/aliked_desc.npy", mmap_mode="r")}
    return d


def place(d, r2, t2, axis):
    """The recording's keyframes moved into the common east-north frame by a 2D rotation and translation."""
    d["r2"], d["t2"] = r2, t2
    d["en"] = d["p"][:, :2] @ r2.T + t2
    d["yaw_en"] = wrap(yaw_of(d["rot"], axis) + np.arctan2(r2[1, 0], r2[0, 0]))
    d["tree"] = cKDTree(d["en"])


def place_by_gps(recs, axis, min_fixes=30, log=print, loose=(), loose_min_fixes=10):
    """Each recording's keyframes in a common east-north frame (en, yaw_en, a k-d tree); those with too few fixes
    (loose_min_fixes for the recordings in `loose`) are dropped."""
    lat0, lon0 = gps_origin([d["gps"] for d in recs.values()])
    for s, d in list(recs.items()):
        if len(d["gps"]) < (loose_min_fixes if s in loose else min_fixes):
            log(f"{s}: {len(d['gps'])} fixes, left out")
            del recs[s]
            continue
        r2, t2, _ = gps_fit(d["traj"], d["tns"], d["gps"], lat0, lon0)
        place(d, r2, t2, axis)


def graph_pose(d, k, axis):
    """Position and heading of the recording's pose graph keyframe k, in its own frame."""
    i = min(np.searchsorted(d["tns"], d["gkf"][k]), len(d["tns"]) - 1)
    return d["traj"][i, 1:3], yaw_of(R.from_quat(d["traj"][i, 4:8]).as_matrix(), axis)


def place_by_edges(recs, seqs, edges, axis, min_inliers=100, log=print):
    """The recordings `seqs` placed by earlier cross edges to recordings placed otherwise: each edge says where the
    query keyframe sits in the map's own frame, so one end placed gives the other end's point in the common frame,
    and a 2D rotation and translation (Cauchy, 3 m) is fitted to these points."""
    for s in seqs:
        if s not in recs:
            continue
        src, dst = [], []
        for sA, ka, sB, kb, dp, _, n_in in edges:
            if n_in < min_inliers or (sA == s) == (sB == s) or (sB if sA == s else sA) in seqs:
                continue
            if sA not in recs or sB not in recs:
                continue
            pa, psa = graph_pose(recs[sA], ka, axis)
            qa = pa + rz(psa)[:2, :2] @ dp[:2]  # the query in the map's own frame
            qb, _ = graph_pose(recs[sB], kb, axis)  # the query in its own frame
            if sB == s:
                src.append(qb); dst.append(qa @ recs[sA]["r2"].T + recs[sA]["t2"])
            else:
                src.append(qa); dst.append(qb @ recs[sB]["r2"].T + recs[sB]["t2"])
        if len(src) < 5:
            log(f"{s}: {len(src)} edges to place it, kept as GPS placed it")
            continue
        r2, t2, res = rigid2d(np.array(src), np.array(dst), 8, 3.0)
        old = recs[s].get("en")
        place(recs[s], r2, t2, axis)
        log(f"{s}: placed by {len(src)} edges, residual {np.median(res):.1f} m at the median"
            + ("" if old is None else
               f"; {np.median(np.linalg.norm(recs[s]['en'] - old, axis=1)):.1f} m from where GPS put it"))


def plan_queries(recs, every=4, offset=0, radius=12.0, max_maps=4, map_from=0):
    """{map: {query recording: [query keyframes]}}: each query keyframe goes to its nearest maps
    (map_from..max_maps-1 by distance) with a keyframe within radius."""
    names = list(recs)
    plan = {s: {} for s in names}
    for sB in names:
        B = recs[sB]
        for kb in range(offset, len(B["akf"]), every):
            cand = []
            for sA in names:
                if sA == sB:
                    continue
                dd, _ = recs[sA]["tree"].query(B["en"][kb])
                if dd < radius:
                    cand.append((dd, sA))
            for _, sA in sorted(cand)[map_from:max_maps]:
                plan[sA].setdefault(sB, []).append(kb)
    return plan


def imu_from_pnp(rvec, tvec, cam):
    """World-from-IMU rotation and position from PnP's world-to-camera pose and the camera's extrinsics."""
    import cv2
    rc, tc, _ = cam
    r_cw = cv2.Rodrigues(rvec)[0]
    r_wc, p_wc = r_cw.T, -r_cw.T @ tvec[:, 0]
    r_wi = r_wc @ rc.T
    return r_wi, p_wc - r_wi @ tc


def edge(sA, ka_graph, sB, kb_graph, r_wi_a, p_wi_a, r_wi_b, p_wi_b, n_in, axis):
    """The cross edge: B's position in A's heading frame, the heading difference, the inlier count."""
    psa = yaw_of(r_wi_a, axis)
    return (sA, int(ka_graph), sB, int(kb_graph), rz(psa).T @ (p_wi_b - p_wi_a), wrap(yaw_of(r_wi_b, axis) - psa), n_in)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("work", help="the recordings' work directories, WORK/SEQ")
    p.add_argument("out")
    p.add_argument("seqs", nargs="+")
    p.add_argument("--radius", type=float, default=12.0, help="m between a query and a map keyframe (GPS-placed)")
    p.add_argument("--yaw", type=float, default=60.0, help="deg between their headings")
    p.add_argument("--every", type=int, default=4, help="query every n-th matched keyframe")
    p.add_argument("--offset", type=int, default=0, help="first query keyframe (with --every, to add queries to a run)")
    p.add_argument("--window", type=int, default=3, help="map keyframes a query is matched to")
    p.add_argument("--max-maps", type=int, default=4, help="maps a query keyframe tries, the nearest first")
    p.add_argument("--map-from", type=int, default=0, help="skip the nearest n maps (to add maps to a run)")
    p.add_argument("--min-inliers", type=int, default=30)
    p.add_argument("--reproj", type=float, default=2.5, help="px, PnP RANSAC")
    p.add_argument("--only", default="", help="keep the pairs with one of these recordings, comma-separated")
    p.add_argument("--loose", default="", help="recordings with poor GPS, comma-separated: they take fixes up to "
                   "--loose-acc metres and need 10 of them")
    p.add_argument("--loose-acc", type=float, default=50.0)
    p.add_argument("--place", nargs="*", default=[], help="edges.npy of an earlier run that place the --loose "
                   "recordings instead of their GPS (edges with 100+ inliers to other recordings)")
    a = p.parse_args()
    import cv2
    import torch
    from lightglue import LightGlue

    os.makedirs(a.out, exist_ok=True)
    loose = set(a.loose.split(",")) if a.loose else set()
    recs = {s: load_recording(a.work, s, a.loose_acc if s in loose else 10.0) for s in a.seqs}
    axis, _ = heading_axis([d["rot"] for d in recs.values()])
    place_by_gps(recs, axis, loose=loose)
    if a.place:
        place_by_edges(recs, loose, [tuple(e) for f in a.place for e in np.load(f, allow_pickle=True)], axis)
    names = list(recs)
    print(f"{len(names)} recordings; heading axis {'xyz'[axis]}", flush=True)
    plan = plan_queries(recs, a.every, a.offset, a.radius, a.max_maps, a.map_from)
    if a.only:
        only = set(a.only.split(","))
        plan = {sA: {sB: kbs for sB, kbs in m.items() if sA in only or sB in only} for sA, m in plan.items()}
    print("queries planned: " + str(sum(len(v) for m in plan.values() for v in m.values())), flush=True)

    dev = "cuda"
    lg = LightGlue(features="aliked", filter_threshold=0.1).eval().to(dev)

    def feats(xy, desc, whv):
        return {"keypoints": torch.from_numpy(np.ascontiguousarray(xy, np.float32))[None].to(dev),
                "descriptors": torch.from_numpy(np.ascontiguousarray(desc, np.float32))[None].to(dev),
                "image_size": torch.tensor([[float(whv[0]), float(whv[1])]], device=dev)}

    edges, done = [], set()
    if os.path.exists(f"{a.out}/done.txt"):
        done = set(open(f"{a.out}/done.txt").read().split())
        edges = [tuple(e) for e in np.load(f"{a.out}/edges.npy", allow_pickle=True)]
        print(f"resuming: {len(done)} maps done, {len(edges)} edges", flush=True)
    t0 = time.time()
    for sA in names:
        if not plan[sA] or sA in done:
            continue
        A = recs[sA]
        z = np.load(f"{a.work}/{sA}/xs_map.npz")
        ofi, olm, mb = map_blocks(z)
        pts, r_wi, t_wi = z["pts"], z["r_wi"], z["t_wi"]
        del z
        for sB, kbs in plan[sA].items():
            B = recs[sB]
            tried = found = 0
            for kb in kbs:
                near = A["tree"].query_ball_point(B["en"][kb], a.radius)
                near = [k for k in near if abs(wrap(A["yaw_en"][k] - B["yaw_en"][kb])) < np.radians(a.yaw)]
                if not near:
                    continue
                near = sorted(near, key=lambda k: np.linalg.norm(A["en"][k] - B["en"][kb]))[:a.window]
                tried += 1
                best = None
                for c in range(2):
                    o0, o1 = B["offsets"][2 * kb + c], B["offsets"][2 * kb + c + 1]
                    if o1 - o0 < 50:
                        continue
                    fq = feats(B["xy"][o0:o1], B["desc"][o0:o1], B["wh"][2 * kb + c])
                    q2lm = {}  # query keypoint -> landmark, the first match kept
                    with torch.inference_mode():
                        for ka in near:
                            for cm in range(2):
                                r = mb.get((ka, cm))
                                if r is None or r[1] - r[0] < 30:
                                    continue
                                fi = ofi[r[0]:r[1]]
                                fm = feats(A["xy"][fi], A["desc"][fi], A["wh"][2 * ka + cm])
                                mt = lg({"image0": fq, "image1": fm})["matches"][0].cpu().numpy()
                                for qi_, mi_ in mt:
                                    q2lm.setdefault(int(qi_), int(olm[r[0] + mi_]))
                    if len(q2lm) < a.min_inliers:
                        continue
                    qi = np.fromiter(q2lm.keys(), int)
                    obj = pts[np.fromiter(q2lm.values(), int)]
                    img = B["xy"][o0 + qi].astype(np.float64)
                    K = B["cams"][c][2]
                    ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, iterationsCount=1000,
                                                             reprojectionError=a.reproj, confidence=0.999,
                                                             flags=cv2.SOLVEPNP_AP3P)
                    if not ok or inl is None or len(inl) < a.min_inliers:
                        continue
                    inl = inl[:, 0]
                    rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, None, rvec, tvec)
                    rb_wi, pb_wi = imu_from_pnp(rvec, tvec, B["cams"][c])
                    if best is None or len(inl) > best[0]:
                        best = (len(inl), rb_wi, pb_wi)
                if best is None:
                    continue
                n_in, rb_wi, pb_wi = best
                k_star = min(near, key=lambda k: np.linalg.norm(t_wi[k] - pb_wi))
                if np.linalg.norm(pb_wi - t_wi[k_star]) > 2 * a.radius:
                    continue
                edges.append(edge(sA, A["a2g"][k_star], sB, B["a2g"][kb], r_wi[k_star], t_wi[k_star], rb_wi, pb_wi,
                                  n_in, axis))
                found += 1
            print(f"  {sB} -> map {sA}: {tried} queries, {found} localized ({time.time() - t0:.0f} s)", flush=True)
        del ofi, olm, mb, pts, r_wi, t_wi
        np.save(f"{a.out}/edges.npy", np.array(edges, dtype=object), allow_pickle=True)
        with open(f"{a.out}/done.txt", "a") as fd:
            fd.write(sA + "\n")
    print(f"{len(edges)} cross edges in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
