#!/usr/bin/env python3
"""Offline visual-inertial bundle adjustment of a LaMAria VIO trajectory.

The Basalt VIO keeps a sliding window of short KLT tracks, so its heading and
scale drift slowly over a sequence. This tool refines the VIO trajectory
afterwards with one batch optimization over the whole sequence:

  features  SIFT on every --kf-step-th VIO frame, both cameras (or learned
            keypoints on the GPU, --extractor aliked|disk|superpoint).
  match     epipolar matching between keyframes up to 256 keyframes (128 s
            at the default 2 Hz) and 200 m apart, guided by the VIO's
            relative poses (learned features: LightGlue's matches within
            the same epipolar band).
  solve     union-find tracks -> midpoint triangulation -> Levenberg-Marquardt
            over every keyframe's pose, velocity and IMU biases, every
            landmark, and each camera's IMU-camera rotation (one per eighth
            of the sequence, --ext-segments), with GTSAM's
            preintegrated IMU factors and Huber reprojection terms. The
            landmarks are eliminated by the Schur complement; because every
            landmark is seen within --band keyframes, the reduced system is
            banded and is factored by LAPACK's banded Cholesky, with the
            camera rotations as a dense border (with --loops, conjugate
            gradients on the whole system, preconditioned by that factor).

The output is the VIO's per-frame trajectory with each frame moved by the
correction of the surrounding keyframes (TUM, seconds, world_from_imu), plus a
JSON summary with the estimated camera rotations. No ground truth is used.

Requirements: numpy, scipy, opencv-python (SIFT), gtsam (IMU factors); numba is optional
and makes the matching about 30 times faster. The learned features need torch and
LightGlue (https://github.com/cvg/LightGlue).
The cameras must be pinhole (the ASL release's undistorted images; a Basalt
double-sphere model with xi = alpha = 0).

Revisits (optional): loops matches places seen again; loop-correct turns those
matches into PnP relative poses and closes them in a pose graph over a solved
trajectory; a second solve started from the corrected trajectory then adds the
revisit matches themselves.

Online: online runs the solve causally, on windows of the most recent
keyframes as the data arrives, each one starting from the one before.

usage:
  lamaria_batch_ba.py features --mav0 DIR --vio TUM --work DIR [--kf-step 10]
  lamaria_batch_ba.py match --work DIR --vio TUM --calib JSON
  lamaria_batch_ba.py solve --work DIR --imu CSV --calib JSON --vio TUM --out TUM
with the revisits:
  lamaria_batch_ba.py loops --work DIR --vio TUM --calib JSON
  lamaria_batch_ba.py solve ... --vio TUM --out BA1.tum --save-states
  lamaria_batch_ba.py loop-correct --work DIR --calib JSON --states BA1_states.npz --tum BA1.tum --out LC.tum
  lamaria_batch_ba.py solve ... --vio LC.tum --out TUM --loops
online:
  lamaria_batch_ba.py online ... --vio TUM --out TUM [--loops] [--window 480] [--step 60]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

DOF = 15  # per keyframe: pose 6 (rotation, translation), velocity 3, bias 6 (accel, gyro)


# ----------------------------------------------------------------------------- io

def quat_to_rot(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_calib(path):
    """Pinhole intrinsics, IMU-camera extrinsics and IMU noise of a Basalt calibration."""
    with open(path) as f:
        v = json.load(f)["value0"]
    cams = []
    for c in range(len(v["T_imu_cam"])):
        e = v["T_imu_cam"][c]
        intr = v["intrinsics"][c]["intrinsics"]
        if abs(intr.get("xi", 0.0)) > 0 or abs(intr.get("alpha", 0.0)) > 0:
            sys.exit("only pinhole cameras (double sphere with xi = alpha = 0) are supported")
        cams.append({
            "R_ic": quat_to_rot(np.array([e["qx"], e["qy"], e["qz"], e["qw"]])),
            "t_ic": np.array([e["px"], e["py"], e["pz"]]),
            "K": np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]]),
        })
    if len(cams) != 2:
        sys.exit("a two-camera calibration is expected")
    noise = {k: float(v[k][0]) for k in ("accel_noise_std", "gyro_noise_std", "accel_bias_std", "gyro_bias_std")}
    return cams, noise


def load_tum(path):
    d = np.loadtxt(path)
    ns = np.round(d[:, 0] * 1e9).astype(np.int64)
    rot = np.array([quat_to_rot(q) for q in d[:, 4:8]])
    return ns, d[:, 1:4].copy(), rot


def load_features(work, desc=True):
    f = np.load(os.path.join(work, "features.npz"))  # lazy: the descriptors are read only when asked for
    return f["kf_ns"], f["offsets"], f["xy"], f["desc"] if desc else None


def feature_kind(work):
    """The extractor of a features.npz ("sift" for those written before it was recorded) and its image sizes."""
    f = np.load(os.path.join(work, "features.npz"))
    return (str(f["extractor"]) if "extractor" in f else "sift"), (f["image_wh"] if "image_wh" in f else None)


def keyframe_poses(kf_ns, vio):
    vio_ns, vio_p, vio_r = load_tum(vio)
    idx = np.searchsorted(vio_ns, kf_ns)
    if np.any(idx >= len(vio_ns)) or np.any(vio_ns[np.minimum(idx, len(vio_ns) - 1)] != kf_ns):
        sys.exit("every keyframe must have a VIO pose with the same timestamp")
    return vio_p[idx], vio_r[idx]


# ----------------------------------------------------------------------------- features

def _sift_image(path, clahe, denoise=0.0, denoise_window=21):
    import cv2
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        sys.exit(f"cannot read {path}")
    if denoise > 0:  # low light: non-local means first, so that the equalization does not amplify the noise
        im = cv2.fastNlMeansDenoising(im, None, denoise, 7, denoise_window)
    if clahe:  # low light: local contrast equalization before detection
        im = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(im)
    return im


def _sift_worker(task):
    import cv2
    path, nfeat, clahe, denoise, denoise_window, rootsift, contrast, edge = task
    cv2.setNumThreads(1)
    sift = cv2.SIFT_create(nfeatures=nfeat, contrastThreshold=contrast, edgeThreshold=edge)
    kp, des = sift.detectAndCompute(_sift_image(path, clahe, denoise, denoise_window), None)
    xy = np.array([k.pt for k in kp], dtype=np.float32).reshape(-1, 2)
    size = np.array([k.size for k in kp], dtype=np.float32)
    if des is None:
        des = np.zeros((0, 128), np.uint8)
    elif rootsift:  # RootSIFT (Arandjelović and Zisserman, 2012): L1-normalized, square-rooted, unit L2 norm
        des = np.sqrt(des / np.maximum(des.sum(1, keepdims=True), 1e-9))
        des = np.clip(np.round(des * 512), 0, 255).astype(np.uint8)
    else:
        des = np.clip(des, 0, 255).astype(np.uint8)
    return xy, des, size


# learned keypoints and descriptors (LightGlue's extractors, on the GPU), matched by LightGlue
LEARNED = ("aliked", "disk", "superpoint")


def _device(name):
    import torch
    return torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))


def _learned_features(a, paths):
    """Keypoints (float32), descriptors (float16) and image sizes of a learned extractor; a thread pool reads
    and preprocesses the images while the GPU runs."""
    from concurrent.futures import ThreadPoolExecutor

    import torch
    from lightglue import ALIKED, DISK, SuperPoint
    device = _device(a.device)
    conf = {"max_num_keypoints": a.num_features}
    if a.detection_threshold is not None:
        conf["detection_threshold"] = a.detection_threshold
    ext = {"aliked": ALIKED, "disk": DISK, "superpoint": SuperPoint}[a.extractor](**conf).eval().to(device)
    res = []
    with ThreadPoolExecutor(a.jobs) as pool:
        for k, im in enumerate(pool.map(lambda p: _sift_image(p, a.clahe, a.denoise, a.denoise_window), paths)):
            x = torch.from_numpy(im).to(device)[None, None].float() / 255
            with torch.inference_mode():
                f = ext.extract(x, resize=a.resize or None)
            res.append((f["keypoints"][0].cpu().numpy().astype(np.float32),
                        f["descriptors"][0].cpu().numpy().astype(np.float16), im.shape[::-1]))
            if k % 2000 == 0:
                print(f"  {k}/{len(paths)} images", flush=True)
    return res


def cmd_features(a):
    os.makedirs(a.work, exist_ok=True)
    vio_ns, _, _ = load_tum(a.vio)
    frames = []
    for c in range(2):
        names = os.listdir(os.path.join(a.mav0, f"cam{c}", "data"))
        frames.append({int(n[:-4]): n for n in names if n.endswith(".png")})
    vio_set = set(vio_ns.tolist())
    usable = [t for t in sorted(set(frames[0]) & set(frames[1])) if t in vio_set]
    kf_ns = np.array(usable[::a.kf_step], dtype=np.int64)
    print(f"{len(usable)} stereo frames with a VIO pose, {len(kf_ns)} keyframes", flush=True)
    paths = [os.path.join(a.mav0, f"cam{c}", "data", frames[c][int(t)]) for t in kf_ns for c in range(2)]
    t0 = time.time()
    extra = {}
    if a.extractor == "sift":
        with mp.Pool(a.jobs) as pool:
            res = pool.map(_sift_worker, [(p, a.num_features, a.clahe, a.denoise, a.denoise_window, a.rootsift,
                                           a.contrast_threshold, a.edge_threshold) for p in paths],
                           chunksize=8)
    else:  # no keypoint size: the solve's --sigma-size leaves these observations at --pixel-sigma
        res = [(xy, des, np.full(len(xy), np.nan, np.float32), wh) for xy, des, wh in _learned_features(a, paths)]
        extra = {"image_wh": np.array([r[3] for r in res], np.int32)}
    if a.rootsift:
        extra["rootsift"] = np.array(True)
    counts = np.array([len(r[0]) for r in res], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    np.savez(os.path.join(a.work, "features.npz"), kf_ns=kf_ns, offsets=offsets,
             xy=np.concatenate([r[0] for r in res]), desc=np.concatenate([r[1] for r in res]),
             size=np.concatenate([r[2] for r in res]), extractor=np.array(a.extractor), **extra)
    print(f"features ({a.extractor}): {offsets[-1]} keypoints, {counts.mean():.0f}/image, {time.time() - t0:.0f} s",
          flush=True)


# ----------------------------------------------------------------------------- flow
FLOW_DESC = 255  # descriptor of a keypoint the flow added: far from every SIFT descriptor, so it never matches


def _flow_chunk(job):
    """Lucas-Kanade flow of one camera's keypoints through the frames between consecutive keyframes ks[0]..ks[-1].

    Each keyframe's keypoints (its own and those the flow brought in) are tracked frame by frame to the next
    keyframe and back; a point that returns within fb_max px and lands within assoc px of one of the next
    keyframe's keypoints links to it, one further than min_dist px from all of them becomes a keypoint of the next
    keyframe itself. A link that contradicts a descriptor match between the two keyframes is left out.
    Returns the links (keyframe, index in it, index in the next) and the added points per next keyframe."""
    import cv2
    from scipy.spatial import cKDTree
    cam, ks, kf_ns, frames, orig, sizes, dmatch, fb_max, assoc, min_dist = job
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    frame_ns = np.array(sorted(frames), np.int64)
    links, added = [], {}
    cur, cur_size = orig[0].astype(np.float32), sizes[0]
    for n, k in enumerate(ks[:-1]):
        nxt, nxt_size = orig[n + 1], sizes[n + 1]
        ts = frame_ns[(frame_ns >= kf_ns[n]) & (frame_ns <= kf_ns[n + 1])]
        if len(cur) == 0 or len(ts) < 2 or ts[0] != kf_ns[n] or ts[-1] != kf_ns[n + 1]:
            cur, cur_size = nxt.astype(np.float32), nxt_size
            continue
        ims = [cv2.imread(frames[t], cv2.IMREAD_GRAYSCALE) for t in ts]
        h, w = ims[0].shape
        p, ok = cur.reshape(-1, 1, 2), np.ones(len(cur), bool)
        for i in range(1, len(ims)):
            p, st, _ = cv2.calcOpticalFlowPyrLK(ims[i - 1], ims[i], p, None, **lk)
            ok &= st[:, 0] == 1
        q = p.copy()
        for i in range(len(ims) - 1, 0, -1):
            q, st, _ = cv2.calcOpticalFlowPyrLK(ims[i], ims[i - 1], q, None, **lk)
            ok &= st[:, 0] == 1
        end = p[:, 0]
        ok &= np.linalg.norm(q[:, 0] - cur, axis=1) < fb_max
        ok &= (end[:, 0] >= 2) & (end[:, 0] < w - 2) & (end[:, 1] >= 2) & (end[:, 1] < h - 2)
        src = np.flatnonzero(ok)
        fwd, bwd = dmatch[n]
        new_pts, new_size = [], []
        if len(nxt) and len(src):
            d, j = cKDTree(nxt).query(end[src])
            # the nearest keypoint, each taken by the closest endpoint only
            near = d < assoc
            order = np.argsort(d)
            taken = set()
            for i in order[near[order]]:
                jj = int(j[i])
                s = int(src[i])
                if jj in taken or fwd.get(s, jj) != jj or bwd.get(jj, s) != s:
                    continue
                taken.add(jj)
                links.append((k, s, jj))
            far = src[~near & (d > min_dist)] if len(nxt) else src
        else:
            far = src
        if len(far):  # new keypoints, one per min_dist cell
            _, first = np.unique(np.floor(end[far] / min_dist).astype(np.int64), axis=0, return_index=True)
            keep = far[np.sort(first)]
            base = len(nxt)
            for m, i in enumerate(keep):
                links.append((k, int(i), base + m))
            new_pts, new_size = end[keep], cur_size[keep]
        added[k + 1] = (np.asarray(new_pts, np.float32).reshape(-1, 2), np.asarray(new_size, np.float32))
        cur = np.concatenate([nxt.astype(np.float32), added[k + 1][0]])
        cur_size = np.concatenate([nxt_size, added[k + 1][1]])
    return cam, links, added


def cmd_flow(a):
    """Longer tracks in low light: links consecutive keyframes by Lucas-Kanade flow through the frames between
    them, adds the points it carries over as keypoints (descriptor FLOW_DESC, never matched by descriptor) to
    features.npz and writes the links as flow.npz (matches.npz's format), which solve and online add with
    --flow. Run after match: a link that contradicts a descriptor match is left out."""
    path = os.path.join(a.work, "features.npz")
    f = dict(np.load(path))
    if "n_orig" in f:
        sys.exit("features.npz already holds the flow's keypoints")
    kf_ns, offsets = f["kf_ns"], f["offsets"]
    n_kf = len(kf_ns)
    frames = []
    for c in range(2):
        d = os.path.join(a.mav0, f"cam{c}", "data")
        frames.append({int(n[:-4]): os.path.join(d, n) for n in os.listdir(d) if n.endswith(".png")})
    dm = [[({}, {}) for _ in range(n_kf)] for _ in range(2)]
    mp_ = os.path.join(a.work, "matches.npz")
    if os.path.exists(mp_):
        m = np.load(mp_)
        P, M = m["pairs"], m["matches"]
        sel = np.flatnonzero((P[:, 2] == P[:, 0] + 1) & (P[:, 1] == P[:, 3]))
        for r in sel:
            fwd, bwd = dm[P[r, 1]][P[r, 0]]
            fwd[int(M[r, 0])], bwd[int(M[r, 1])] = int(M[r, 1]), int(M[r, 0])
    xy, size = f["xy"], f["size"]
    img = lambda k, c: slice(offsets[2 * k + c], offsets[2 * k + c + 1])
    edges = np.linspace(0, n_kf - 1, max(2, min(n_kf - 1, a.jobs * 4) + 1)).round().astype(int)
    jobs = []
    for c in range(2):
        for s, e in zip(edges[:-1], edges[1:]):
            ks = list(range(s, e + 1))
            jobs.append((c, ks, kf_ns[s:e + 1], {t: frames[c][t] for t in frames[c]
                                                 if kf_ns[s] <= t <= kf_ns[e]},
                         [xy[img(k, c)] for k in ks], [size[img(k, c)] for k in ks], dm[c][s:e + 1],
                         a.fb_max, a.assoc, a.min_dist))
    t0 = time.time()
    with mp.Pool(a.jobs) as pool:
        res = pool.map(_flow_chunk, jobs, chunksize=1)
    add = {}
    pairs, matches = [], []
    for c, links, added in res:
        for k, (p, s) in added.items():
            add[2 * k + c] = (p, s)
        for k, i, j in links:
            pairs.append((k, c, k + 1, c))
            matches.append((i, j))
    n_img = 2 * n_kf
    n_orig = np.diff(offsets)
    counts = n_orig + np.array([len(add[i][0]) if i in add else 0 for i in range(n_img)])
    new_off = np.concatenate([[0], np.cumsum(counts)])
    nx = np.zeros((new_off[-1], 2), np.float32)
    nd = np.full((new_off[-1], f["desc"].shape[1]), FLOW_DESC, f["desc"].dtype)
    ns = np.zeros(new_off[-1], np.float32)
    for i in range(n_img):
        o, n0 = new_off[i], n_orig[i]
        nx[o:o + n0], nd[o:o + n0], ns[o:o + n0] = xy[offsets[i]:offsets[i + 1]], \
            f["desc"][offsets[i]:offsets[i + 1]], size[offsets[i]:offsets[i + 1]]
        if i in add:
            nx[o + n0:new_off[i + 1]], ns[o + n0:new_off[i + 1]] = add[i]
    f.update(offsets=new_off, xy=nx, desc=nd, size=ns, n_orig=n_orig)
    np.savez(path + ".tmp.npz", **f)
    os.replace(path + ".tmp.npz", path)
    pairs = np.array(pairs, np.int32).reshape(-1, 4)
    matches = np.array(matches, np.int32).reshape(-1, 2)
    np.savez(os.path.join(a.work, "flow.npz"), pairs=pairs, matches=matches)
    print(f"flow: {len(matches)} links ({len(matches) - (counts - n_orig).sum()} to keypoints), "
          f"{(counts - n_orig).sum()} keypoints added, {time.time() - t0:.0f} s", flush=True)


# ----------------------------------------------------------------------------- matching

_G = {}  # read-only matching state shared with the forked workers


def _cam_pose(k, c):
    """world_from_camera of keyframe k, camera c."""
    r_wi, t_wi = _G["kf_r"][k], _G["kf_p"][k]
    cam = _G["cams"][c]
    return r_wi @ cam["R_ic"], r_wi @ cam["t_ic"] + t_wi


def _image(k, c):
    i = 2 * k + c
    s, e = _G["offsets"][i], _G["offsets"][i + 1]
    d = _G["desc"][s:e]
    return _G["xy"][s:e], d.astype(np.float32) if d.dtype == np.float16 else d


def _best_two(keys, vals, n):
    """Per key: index (into keys/vals) of the smallest value, and the second smallest value."""
    order = np.lexsort((vals, keys))
    k_sorted = keys[order]
    first = np.flatnonzero(np.r_[True, k_sorted[1:] != k_sorted[:-1]])
    best = np.full(n, -1, np.int64)
    second_val = np.full(n, np.inf, np.float32)
    best[k_sorted[first]] = order[first]
    nxt = np.minimum(first + 1, len(order) - 1)
    same = (first + 1 < len(order)) & (k_sorted[nxt] == k_sorted[first])
    second_val[k_sorted[first[same]]] = vals[order[nxt[same]]]
    return best, second_val



try:  # optional: the same matching rules ~30x faster
    from numba import njit, prange
except ImportError:
    njit, prange = None, range

if njit is not None:
    @njit(cache=True, nogil=True)
    def _match_kernel(xa, xb, da, db, mat, use_rot, tau_n, ratio, max_desc):
        n, m = xa.shape[0], xb.shape[0]
        inf = np.float32(np.inf)
        best_j = np.full(n, -1, np.int64)
        best_d = np.full(n, inf, np.float32)
        second_d = np.full(n, inf, np.float32)
        bbest_i = np.full(m, -1, np.int64)
        bbest_d = np.full(m, inf, np.float32)
        for i in range(n):
            l0 = mat[0, 0] * xa[i, 0] + mat[0, 1] * xa[i, 1] + mat[0, 2] * xa[i, 2]
            l1 = mat[1, 0] * xa[i, 0] + mat[1, 1] * xa[i, 1] + mat[1, 2] * xa[i, 2]
            l2 = mat[2, 0] * xa[i, 0] + mat[2, 1] * xa[i, 1] + mat[2, 2] * xa[i, 2]
            if not use_rot:  # epipolar line in b
                nrm = np.sqrt(l0 * l0 + l1 * l1)
                l0, l1, l2 = l0 / nrm, l1 / nrm, l2 / nrm
            else:  # rotation-only prediction in b
                l2 = max(l2, np.float32(1e-3))
                l0, l1 = l0 / l2, l1 / l2
            for j in range(m):
                if not use_rot:
                    dist = abs(l0 * xb[j, 0] + l1 * xb[j, 1] + l2 * xb[j, 2])
                else:
                    dx, dy = l0 - xb[j, 0], l1 - xb[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                if dist < tau_n:
                    acc = 0
                    for k in range(128):
                        d = np.int32(da[i, k]) - np.int32(db[j, k])
                        acc += d * d
                    dd = np.sqrt(np.float32(acc))
                    if dd < best_d[i]:
                        second_d[i] = best_d[i]
                        best_d[i] = dd
                        best_j[i] = j
                    elif dd < second_d[i]:
                        second_d[i] = dd
                    if dd < bbest_d[j]:
                        bbest_d[j] = dd
                        bbest_i[j] = i
        cnt = 0
        out = np.empty((n, 2), np.int32)
        for i in range(n):
            j = best_j[i]
            if j >= 0 and best_d[i] < ratio * second_d[i] and best_d[i] < max_desc and bbest_i[j] == i:
                out[cnt, 0], out[cnt, 1] = i, j
                cnt += 1
        return out[:cnt]

def _match_worker(pair):
    """Mutual, ratio-tested SIFT matches restricted to the VIO's epipolar band."""
    ka, ca, kb, cb, tau = pair
    xya, da = _image(ka, ca)
    xyb, db = _image(kb, cb)
    empty = np.zeros((0, 2), np.int32)
    if len(xya) == 0 or len(xyb) == 0:
        return pair, empty
    r_wa, t_wa = _cam_pose(ka, ca)
    r_wb, t_wb = _cam_pose(kb, cb)
    r_ba = r_wb.T @ r_wa
    t_ba = r_wb.T @ (t_wa - t_wb)
    fb = _G["cams"][cb]["K"][0, 0]
    xa = (np.c_[xya, np.ones(len(xya))] @ np.linalg.inv(_G["cams"][ca]["K"]).T).astype(np.float32)
    xb = (np.c_[xyb, np.ones(len(xyb))] @ np.linalg.inv(_G["cams"][cb]["K"]).T).astype(np.float32)
    use_rot = np.linalg.norm(t_ba) <= 0.05
    if njit is not None and da.dtype == np.uint8:  # SIFT; learned descriptors take the numpy path below
        mat = r_ba if use_rot else np.array([[0, -t_ba[2], t_ba[1]], [t_ba[2], 0, -t_ba[0]],
                                             [-t_ba[1], t_ba[0], 0]]) @ r_ba
        return pair, _match_kernel(xa, xb, np.ascontiguousarray(da), np.ascontiguousarray(db),
                                   mat.astype(np.float32), use_rot, np.float32(tau / fb),
                                   np.float32(_G["ratio"]), np.float32(_G["max_desc"]))
    if not use_rot:
        e = np.array([[0, -t_ba[2], t_ba[1]], [t_ba[2], 0, -t_ba[0]], [-t_ba[1], t_ba[0], 0]]) @ r_ba
        lines = xa @ e.T.astype(np.float32)
        lines /= np.hypot(lines[:, 0], lines[:, 1])[:, None]
        dist = np.abs(lines @ xb.T)
    else:  # no baseline: predict with the rotation alone
        pred = xa @ r_ba.T.astype(np.float32)
        pred = pred[:, :2] / np.maximum(pred[:, 2:3], 1e-3)
        dist = np.sqrt(((pred[:, None, :] - xb[None, :, :2]) ** 2).sum(2))
    ia, ib = np.nonzero(dist < tau / fb)
    if len(ia) == 0:
        return pair, empty
    if da.dtype == np.uint8:
        diff = da[ia].astype(np.int32) - db[ib].astype(np.int32)
        dd = np.sqrt((diff * diff).sum(1).astype(np.float32))
    else:
        diff = da[ia] - db[ib]
        dd = np.sqrt((diff * diff).sum(1))
    best_a, second_a = _best_two(ia, dd, len(xya))
    best_b, _ = _best_two(ib, dd, len(xyb))
    rows = np.flatnonzero(best_a >= 0)
    c = best_a[rows]
    d1 = dd[c]
    j = ib[c]
    ok = (d1 < _G["ratio"] * second_a[rows]) & (d1 < _G["max_desc"]) & (ia[best_b[j]] == rows)
    return pair, np.c_[rows[ok], j[ok]].astype(np.int32)


def _matcher(choice, extractor):
    """--matcher auto: LightGlue for learned features, the ratio test for SIFT."""
    if choice == "auto":
        return "ratio" if extractor == "sift" else "lightglue"
    if choice == "lightglue" and extractor == "sift":
        sys.exit("LightGlue matches the learned features only (features --extractor)")
    return choice


ROOTSIFT_MAX_DESC = 225.0  # RootSIFT's nearest-neighbour distances run at about 0.75 times SIFT's


def _max_desc(value, extractor, work=None):
    """--max-desc: SIFT's L2 distance of 300 by default (225 for RootSIFT); no limit on the unit-length learned
    descriptors."""
    if value is not None:
        return value
    if extractor != "sift":
        return np.inf
    rootsift = work is not None and bool(np.load(os.path.join(work, "features.npz")).get("rootsift", False))
    return ROOTSIFT_MAX_DESC if rootsift else 300.0


def _band_dist(ka, ca, kb, cb, xya, xyb):
    """Distance of each correspondence from the VIO's epipolar line in b, or from the rotation-only prediction
    when the cameras are within 5 cm (normalized image units of b, as in _match_worker)."""
    r_wa, t_wa = _cam_pose(ka, ca)
    r_wb, t_wb = _cam_pose(kb, cb)
    r_ba = r_wb.T @ r_wa
    t_ba = r_wb.T @ (t_wa - t_wb)
    xa = np.c_[xya, np.ones(len(xya))] @ np.linalg.inv(_G["cams"][ca]["K"]).T
    xb = np.c_[xyb, np.ones(len(xyb))] @ np.linalg.inv(_G["cams"][cb]["K"]).T
    if np.linalg.norm(t_ba) <= 0.05:
        pred = xa @ r_ba.T
        return np.linalg.norm(pred[:, :2] / np.maximum(pred[:, 2:3], 1e-3) - xb[:, :2], axis=1)
    lines = xa @ (skew(t_ba) @ r_ba).T
    lines /= np.hypot(lines[:, 0], lines[:, 1])[:, None]
    return np.abs((lines * xb).sum(1))


def _lightglue_batch(lg, fa, fb, mixed):
    """LightGlue on a batch of image pairs: all nine layers (no early stopping or point pruning). One pair per
    call takes about 27 ms on an 8 GB GPU whatever the number of keypoints (512-1,024): the per-call latency,
    which a batch amortizes. Images with the same number of keypoints on each side (a learned extractor's top
    --num-features, features --detection-threshold 0) batch as they are: 16 pairs of 1,024 keypoints take
    about 9 ms per pair in half precision. Otherwise the padding is masked out, and the masked attention is as
    slow as one pair per call. Returns each pair's matches."""
    import torch
    from lightglue.lightglue import filter_matches, normalize_keypoints, sigmoid_log_double_softmax
    na = [f["keypoints"].shape[1] for f in fa]
    nb = [f["keypoints"].shape[1] for f in fb]
    dev = fa[0]["keypoints"].device

    def stack(fs, key, n):
        out = fs[0][key].new_zeros((len(fs), n) + fs[0][key].shape[2:])
        for i, f in enumerate(fs):
            out[i, :f[key].shape[1]] = f[key][0]
        return out

    padded = len(set(na)) > 1 or len(set(nb)) > 1
    m_a = torch.arange(max(na), device=dev)[None] < torch.tensor(na, device=dev)[:, None]
    m_b = torch.arange(max(nb), device=dev)[None] < torch.tensor(nb, device=dev)[:, None]
    size_a = torch.cat([f["image_size"] for f in fa]) if "image_size" in fa[0] else None
    size_b = torch.cat([f["image_size"] for f in fb]) if "image_size" in fb[0] else None
    with torch.inference_mode(), torch.autocast(device_type=dev.type, enabled=mixed):
        da, db = lg.input_proj(stack(fa, "descriptors", max(na))), lg.input_proj(stack(fb, "descriptors", max(nb)))
        ea = lg.posenc(normalize_keypoints(stack(fa, "keypoints", max(na)), size_a))
        eb = lg.posenc(normalize_keypoints(stack(fb, "keypoints", max(nb)), size_b))
        for layer in lg.transformers:  # masks B x 1 x M x 1: they broadcast over the attention heads
            da, db = (layer.masked_forward(da, db, ea, eb, m_a[:, None, :, None], m_b[:, None, :, None]) if padded
                      else layer(da, db, ea, eb))
        assign = lg.log_assignment[-1]
        pa, pb = assign.final_proj(da), assign.final_proj(db)
        sim = torch.einsum("bmd,bnd->bmn", pa, pb) / pa.shape[-1] ** 0.5
        if padded:
            sim = sim.masked_fill(~(m_a[:, :, None] & m_b[:, None, :]), -1e4)  # the padding takes no share
        scores = sigmoid_log_double_softmax(sim, assign.matchability(da), assign.matchability(db)).float()
    out = []
    for i in range(len(fa)):  # row na[i] and column nb[i] are padding or the dustbins: filter_matches drops them
        m0 = filter_matches(scores[i:i + 1, :na[i] + 1, :nb[i] + 1], lg.conf.filter_threshold)[0][0]
        valid = m0 > -1
        out.append(torch.stack([torch.nonzero(valid)[:, 0], m0[valid]], -1))
    return [o.cpu().numpy().astype(np.int32) for o in out]


def _lightglue_pairs(tasks, a, cache_images=1200):
    """LightGlue's matches of every image pair (ka, ca, kb, cb, ...), in order, --lg-batch pairs per call. The
    features of the last cache_images images used stay on the GPU."""
    from collections import OrderedDict
    from itertools import islice

    import torch
    from lightglue import LightGlue
    device = _device(a.device)
    mixed = a.lg_mp and device.type == "cuda"
    lg = LightGlue(features=_G["extractor"], filter_threshold=a.lg_threshold).eval().to(device)
    cache = OrderedDict()

    def feats(i):
        if i in cache:
            cache.move_to_end(i)
            return cache[i]
        s, e = _G["offsets"][i], _G["offsets"][i + 1]
        f = {"keypoints": torch.from_numpy(np.ascontiguousarray(_G["xy"][s:e])).to(device)[None],
             "descriptors": torch.from_numpy(np.asarray(_G["desc"][s:e], np.float16 if mixed else np.float32))
             .to(device)[None]}
        if _G["image_wh"] is not None:
            f["image_size"] = torch.tensor(_G["image_wh"][i], dtype=torch.float32, device=device)[None]
        cache[i] = f
        if len(cache) > cache_images:
            cache.popitem(last=False)
        return f

    tasks = iter(tasks)
    while chunk := list(islice(tasks, a.lg_batch)):
        fa = [feats(2 * t[0] + t[1]) for t in chunk]
        fb = [feats(2 * t[2] + t[3]) for t in chunk]
        out = [np.zeros((0, 2), np.int32)] * len(chunk)
        ok = [i for i in range(len(chunk)) if fa[i]["keypoints"].shape[1] and fb[i]["keypoints"].shape[1]]
        if ok:
            for i, m in zip(ok, _lightglue_batch(lg, [fa[i] for i in ok], [fb[i] for i in ok], mixed)):
                out[i] = m
        yield from zip(chunk, out)


def _lightglue_guided(pairs, a, count):
    """LightGlue's matches within the VIO's epipolar band of each pair (the band of _match_worker)."""
    off = _G["offsets"]
    for pair, m in _lightglue_pairs(pairs, a):
        count[0] += len(m)
        if len(m):
            ka, ca, kb, cb, tau = pair
            xya, xyb = _G["xy"][off[2 * ka + ca]:], _G["xy"][off[2 * kb + cb]:]
            m = m[_band_dist(ka, ca, kb, cb, xya[m[:, 0]], xyb[m[:, 1]]) < tau / _G["cams"][cb]["K"][0, 0]]
        count[1] += len(m)
        yield pair, m


def _lightglue_essential(pairs, a, count, chunk=512):
    """LightGlue's matches of each pair that an essential matrix explains, checked as a revisit is (_loop_worker:
    RANSAC at 1 px, --min-inliers, its rotation within --max-rot of the VIO's): the VIO's translation, and so
    its drift, no longer picks the matches. The essential matrices are fitted on --jobs threads (OpenCV
    releases the GIL), chunk pairs at a time."""
    from concurrent.futures import ThreadPoolExecutor
    from itertools import islice

    def verify(item):
        return _loop_worker(item[0][:4], item[1])[1]

    found = _lightglue_pairs(pairs, a)
    with ThreadPoolExecutor(max(1, a.jobs)) as pool:
        while batch := list(islice(found, chunk)):
            for (pair, m), kept in zip(batch, pool.map(verify, batch)):
                count[0] += len(m)
                count[1] += len(kept)
                yield pair, kept


def cmd_match(a):
    from contextlib import nullcontext
    kf_ns, offsets, xy, desc = load_features(a.work)
    extractor, image_wh = feature_kind(a.work)
    matcher = _matcher(a.matcher, extractor)
    if a.verify == "essential" and matcher != "lightglue":
        sys.exit("match --verify essential takes LightGlue's matches (learned features)")
    cams, _ = load_calib(a.calib)
    kf_p, kf_r = keyframe_poses(kf_ns, a.vio)
    _G.update(offsets=offsets, xy=xy, desc=desc, cams=cams, kf_p=kf_p, kf_r=kf_r,
              ratio=a.ratio, max_desc=_max_desc(a.max_desc, extractor, a.work), extractor=extractor, image_wh=image_wh,
              min_inliers=a.min_inliers, max_rot=a.max_rot)
    offs = [int(x) for x in a.offsets.split(",")]
    cos_max = np.cos(np.radians(a.max_axis_angle))
    pairs = []
    n = len(kf_ns)
    for ka in range(n):
        if a.stereo:  # the two cameras of one keyframe: the rig's fixed baseline, whatever their axes' angle
            pairs.append((ka, 0, ka, 1, a.stereo_tau))
        for d in offs:
            kb = ka + d
            if kb >= n:
                break
            for ca in range(2):
                ra, ta = _cam_pose(ka, ca)
                for cb in range(2):
                    rb, tb = _cam_pose(kb, cb)
                    if ra[:, 2] @ rb[:, 2] < cos_max or np.linalg.norm(ta - tb) > a.max_dist:
                        continue
                    pairs.append((ka, ca, kb, cb, min(a.tau0 + a.tau_slope * d, a.tau_max)))
    print(f"{n} keyframes, {len(pairs)} image pairs to match ({extractor}, {matcher})", flush=True)
    t0 = time.time()
    out_pairs, out_matches, count = [], [], [0, 0]
    lightglue = _lightglue_essential if a.verify == "essential" else _lightglue_guided
    with mp.Pool(a.jobs) if matcher == "ratio" else nullcontext() as pool:
        results = pool.imap(_match_worker, pairs, chunksize=16) if pool else lightglue(pairs, a, count)
        for i, (pair, m) in enumerate(results):
            if len(m):
                out_pairs.append(np.array(pair[:4], np.int32)[None].repeat(len(m), 0))
                out_matches.append(m)
            if i % 20000 == 0:
                print(f"  {i}/{len(pairs)} pairs, {time.time() - t0:.0f} s", flush=True)
    np.savez(os.path.join(a.work, "matches.npz"), pairs=np.concatenate(out_pairs),
             matches=np.concatenate(out_matches))
    if matcher == "lightglue":
        kept = "explained by an essential matrix" if a.verify == "essential" else "within the epipolar band"
        print(f"  LightGlue: {count[0]} matches, {count[1]} {kept}", flush=True)
    print(f"matches: {sum(len(m) for m in out_matches)} over {len(out_pairs)} pairs, "
          f"{time.time() - t0:.0f} s", flush=True)


# ----------------------------------------------------------------------------- revisits

def _unguided_matches(da, db, ratio, max_desc, method=None):
    """Mutual nearest neighbours with the ratio test, over all features.

    method: "cv2" (OpenCV's brute-force matcher, the default when OpenCV is there), "numba" or "numpy".
    """
    if method is None:
        try:
            import cv2  # noqa: F401
            method = "cv2"
        except ImportError:  # the numba kernel takes SIFT's bytes only
            method = "numba" if njit is not None and da.dtype == np.uint8 else "numpy"
    if method == "cv2":
        import cv2
        fa, fb = np.ascontiguousarray(da, np.float32), np.ascontiguousarray(db, np.float32)
        if len(fa) < 2 or len(fb) < 2:
            return np.zeros((0, 2), np.int32)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        best_a = np.full(len(fb), -1)
        for m in bf.match(fb, fa):
            best_a[m.queryIdx] = m.trainIdx
        out = [(i, m0.trainIdx) for i, (m0, m1) in enumerate(bf.knnMatch(fa, fb, k=2))
               if m0.distance < ratio * m1.distance and m0.distance < max_desc and best_a[m0.trainIdx] == i]
        return np.array(out, np.int32).reshape(-1, 2)
    if method == "numba":
        ones = np.zeros((len(da), 3), np.float32)
        return _match_kernel(ones, np.zeros((len(db), 3), np.float32), np.ascontiguousarray(da),
                             np.ascontiguousarray(db), np.eye(3, dtype=np.float32), True, np.float32(1e9),
                             np.float32(ratio), np.float32(max_desc))
    fa, fb = da.astype(np.float32), db.astype(np.float32)
    d2 = np.maximum((fa * fa).sum(1)[:, None] + (fb * fb).sum(1)[None, :] - 2 * fa @ fb.T, 0)
    two = np.argpartition(d2, 1, axis=1)[:, :2]
    rows = np.arange(len(fa))
    d_two = d2[rows[:, None], two]
    o = np.argsort(d_two, axis=1)
    j = two[rows, o[:, 0]]
    d1, d2nd = np.sqrt(d_two[rows, o[:, 0]]), np.sqrt(d_two[rows, o[:, 1]])
    ok = (d1 < ratio * d2nd) & (d1 < max_desc) & (np.argmin(d2, axis=0)[j] == rows)
    return np.c_[rows[ok], j[ok]].astype(np.int32)


def _loop_worker(task, m=None):
    """A revisit candidate: unguided matching (or LightGlue's matches m), essential-matrix RANSAC, and a
    rotation check."""
    import cv2
    cv2.setNumThreads(1)  # one thread per pool process
    ka, ca, kb, cb = task
    xya, da = _image(ka, ca)
    xyb, db = _image(kb, cb)
    empty = np.zeros((0, 2), np.int32)
    if len(xya) < _G["min_inliers"] or len(xyb) < _G["min_inliers"]:
        return task, empty, 0
    if m is None:
        m = _unguided_matches(da, db, _G["ratio"], _G["max_desc"])
    if len(m) < _G["min_inliers"]:
        return task, empty, 0
    pa = (np.c_[xya[m[:, 0]], np.ones(len(m))] @ np.linalg.inv(_G["cams"][ca]["K"]).T)[:, :2]
    pb = (np.c_[xyb[m[:, 1]], np.ones(len(m))] @ np.linalg.inv(_G["cams"][cb]["K"]).T)[:, :2]
    e, mask = cv2.findEssentialMat(pa, pb, np.eye(3), method=cv2.RANSAC, prob=0.999,
                                   threshold=1.0 / _G["cams"][cb]["K"][0, 0])
    if e is None or mask is None or e.shape != (3, 3) or mask.sum() < _G["min_inliers"]:
        return task, empty, 0
    mask = mask.ravel().astype(bool)
    _, r_est, _, _ = cv2.recoverPose(e, pa[mask], pb[mask], np.eye(3))
    r_ba = _cam_pose(kb, cb)[0].T @ _cam_pose(ka, ca)[0]
    angle = np.degrees(np.arccos(np.clip((np.trace(r_est.T @ r_ba) - 1) / 2, -1, 1)))
    if angle > _G["max_rot"]:
        return task, empty, -1
    return task, m[mask], int(mask.sum())


def cmd_loops(a):
    from contextlib import nullcontext
    kf_ns, offsets, xy, desc = load_features(a.work)
    extractor, image_wh = feature_kind(a.work)
    matcher = _matcher(a.matcher, extractor)
    cams, _ = load_calib(a.calib)
    kf_p, kf_r = keyframe_poses(kf_ns, a.vio)
    _G.update(offsets=offsets, xy=xy, desc=desc, cams=cams, kf_p=kf_p, kf_r=kf_r, ratio=a.ratio,
              max_desc=_max_desc(a.max_desc, extractor, a.work), min_inliers=a.min_inliers, max_rot=a.max_rot,
              extractor=extractor, image_wh=image_wh)
    n = len(kf_ns)
    axes = np.array([[_cam_pose(k, c)[0][:, 2] for c in range(2)] for k in range(n)])
    ks = np.arange(0, n, a.step)
    dist = np.linalg.norm(kf_p[ks][:, None, :2] - kf_p[ks][None, :, :2], axis=2)
    ia, ib = np.nonzero((ks[None, :] - ks[:, None] > a.min_gap) & (dist < a.max_dist))
    cos_max = np.cos(np.radians(a.max_axis_angle))
    tasks = [(int(ks[i]), ca, int(ks[j]), cb) for i, j in zip(ia, ib) for ca in range(2) for cb in range(2)
             if axes[ks[i], ca] @ axes[ks[j], cb] >= cos_max]
    print(f"{len(ia)} revisit keyframe pairs, {len(tasks)} image pairs to verify ({extractor}, {matcher})",
          flush=True)
    t0 = time.time()
    out_pairs, out_matches, by_rotation = [], [], 0
    with mp.Pool(a.jobs) if matcher == "ratio" else nullcontext() as pool:
        results = (pool.imap(_loop_worker, tasks, chunksize=8) if pool else
                   (_loop_worker(task, m) for task, m in _lightglue_pairs(tasks, a)))
        for task, m, ninl in results:
            by_rotation += ninl < 0
            if len(m):
                out_pairs.append(np.array(task, np.int32)[None].repeat(len(m), 0))
                out_matches.append(m)
    pairs = np.concatenate(out_pairs) if out_pairs else np.zeros((0, 4), np.int32)
    matches = np.concatenate(out_matches) if out_matches else np.zeros((0, 2), np.int32)
    np.savez(os.path.join(a.work, "loops.npz"), pairs=pairs, matches=matches)
    print(f"loops: {len(matches)} matches over {len(out_pairs)} verified image pairs "
          f"({len(np.unique(pairs[:, [0, 2]], axis=0))} keyframe pairs, {by_rotation} rejected by rotation), "
          f"{time.time() - t0:.0f} s", flush=True)


# ----------------------------------------------------------------------------- tracks

def build_tracks(offsets, pairs, matches, max_offset):
    """Connected components of the match graph; components with two features in one image are dropped."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    keep = (pairs[:, 2] - pairs[:, 0]) <= max_offset
    pairs, matches = pairs[keep], matches[keep]
    ga = offsets[2 * pairs[:, 0] + pairs[:, 1]] + matches[:, 0]
    gb = offsets[2 * pairs[:, 2] + pairs[:, 3]] + matches[:, 1]
    n = int(offsets[-1])
    _, roots = connected_components(coo_matrix((np.ones(len(ga), np.int8), (ga, gb)), shape=(n, n)),
                                    directed=False)
    image_of = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    order = np.argsort(roots, kind="stable")
    r_sorted = roots[order]
    starts = np.flatnonzero(np.r_[True, r_sorted[1:] != r_sorted[:-1]])
    ends = np.r_[starts[1:], len(order)]
    tracks, conflicts = [], 0
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        feats = order[s:e]
        imgs = image_of[feats]
        if len(np.unique(imgs)) != len(imgs):
            conflicts += 1
            continue
        tracks.append(feats)
    return tracks, image_of, conflicts


def triangulate(tracks, image_of, xy, kf_p, kf_r, cams, max_err, min_angle_deg):
    """Midpoint triangulation with the VIO poses; drops observations off by more than max_err pixels.

    Every track gets up to three rounds: triangulate it from its observations, drop those behind the camera
    or more than max_err off, and keep it once a round drops nothing, with at least two observations and a
    triangulation angle of min_angle_deg. All tracks at once; returns the points and the kept features.
    """
    nimg = len(kf_p) * 2
    centers = np.zeros((nimg, 3))
    rots = np.zeros((nimg, 3, 3))
    for k in range(len(kf_p)):
        for c in range(2):
            rots[2 * k + c] = kf_r[k] @ cams[c]["R_ic"]
            centers[2 * k + c] = kf_r[k] @ cams[c]["t_ic"] + kf_p[k]
    kmat = np.array([cams[c]["K"] for c in range(2)])
    kinv = np.linalg.inv(kmat)
    n_t = len(tracks)
    lens = np.array([len(t) for t in tracks], np.int64)
    feats = np.concatenate(tracks).astype(np.int64) if n_t else np.zeros(0, np.int64)
    tid = np.repeat(np.arange(n_t), lens)
    imgs = image_of[feats]
    cam = imgs % 2
    rays = np.einsum("nij,nj->ni", kinv[cam], np.c_[xy[feats], np.ones(len(feats))])
    rays = np.einsum("nij,nj->ni", rots[imgs], rays)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    proj = (np.eye(3)[None] - rays[:, :, None] * rays[:, None, :]).reshape(-1, 9)
    proj_c = np.einsum("nij,nj->ni", proj.reshape(-1, 3, 3), centers[imgs])
    active = np.ones(len(feats), bool)
    state = np.zeros(n_t, np.int8)  # 0 open, 1 kept, 2 dropped
    x = np.zeros((n_t, 3))
    for _ in range(3):
        state[(state == 0) & (np.bincount(tid[active], minlength=n_t) < 2)] = 2
        sel = np.flatnonzero(active & (state[tid] == 0))
        if not len(sel):
            break
        a_m = np.stack([np.bincount(tid[sel], proj[sel, i], n_t) for i in range(9)], 1).reshape(-1, 3, 3)
        b_v = np.stack([np.bincount(tid[sel], proj_c[sel, i], n_t) for i in range(3)], 1)
        open_t = np.flatnonzero(state == 0)
        singular = np.linalg.det(a_m[open_t]) == 0
        state[open_t[singular]] = 2
        open_t = open_t[~singular]
        x[open_t] = np.linalg.solve(a_m[open_t], b_v[open_t][..., None])[..., 0]
        o = sel[state[tid[sel]] == 0]
        pc = np.einsum("nji,nj->ni", rots[imgs[o]], x[tid[o]] - centers[imgs[o]])
        px = np.einsum("nij,nj->ni", kmat[cam[o]], pc)
        err = np.linalg.norm(px[:, :2] / np.maximum(px[:, 2:3], 1e-9) - xy[feats[o]], axis=1)
        bad = o[~((pc[:, 2] > 0.2) & (err < max_err))]
        clean = np.ones(n_t, bool)
        clean[tid[bad]] = False
        state[open_t[clean[open_t]]] = 1
        active[bad] = False
    # the widest pair of a kept track's rays must reach min_angle_deg
    o = np.flatnonzero(active & (state[tid] == 1))
    if not len(o):
        return np.zeros((0, 3)), []
    t_o = tid[o]
    first = np.r_[0, np.flatnonzero(np.diff(t_o)) + 1]
    cnt = np.diff(np.r_[first, len(o)])
    m = np.repeat(cnt, cnt)
    pa = np.repeat(np.arange(len(o)), m)
    pb = np.repeat(first, cnt)[pa] + np.arange(len(pa)) - np.repeat(np.r_[0, np.cumsum(m)[:-1]], m)
    min_cos = np.minimum.reduceat(np.einsum("ni,ni->n", rays[o[pa]], rays[o[pb]]), np.r_[0, np.cumsum(m)[:-1]][first])
    wide = min_cos <= np.cos(np.radians(min_angle_deg))
    kept_t = t_o[first][wide]
    if not len(kept_t):
        return np.zeros((0, 3)), []
    keep_o = o[np.isin(t_o, kept_t)]
    split = np.cumsum(np.bincount(tid[keep_o], minlength=n_t)[kept_t])[:-1]
    return x[kept_t], np.split(feats[keep_o], split)


# ----------------------------------------------------------------------------- solver

def skew(v):
    o = np.zeros(v.shape[:-1] + (3, 3))
    o[..., 0, 1], o[..., 0, 2] = -v[..., 2], v[..., 1]
    o[..., 1, 0], o[..., 1, 2] = v[..., 2], -v[..., 0]
    o[..., 2, 0], o[..., 2, 1] = -v[..., 1], v[..., 0]
    return o


def so3_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3) + skew(w[None])[0]
    k = skew((w / th)[None])[0]
    return np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * k @ k


def so3_log(r):
    """Rotation vector of a rotation matrix (not near 180 degrees)."""
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / 2  # sin(angle) * axis
    s = np.linalg.norm(w)
    return w if s < 1e-12 else w * (np.arctan2(s, (np.trace(r) - 1) / 2) / s)


def so3_log_deg(r):
    return np.degrees(so3_log(r))


def segment_prior(r_ic, pairs, slot, n_ext, w):
    """Prior 0.5 w |Log(R_a^T R_b)|^2 between consecutive time segments (a, b) of a camera's rotation.

    Returns the cost and its Gauss-Newton normal equations H phi = b in the increments
    R <- R Exp(phi) of the estimated rotations, stacked by slot: Log(R_a^T R_b) ~ e0 - phi_a + phi_b.
    """
    h, b, cost = np.zeros((3 * n_ext, 3 * n_ext)), np.zeros(3 * n_ext), 0.0
    ws = np.broadcast_to(np.asarray(w, dtype=np.float64), (len(pairs),))  # one weight, or one per pair
    for (ca, cb), w in zip(pairs, ws):
        e0 = so3_log(r_ic[ca].T @ r_ic[cb])
        cost += 0.5 * w * float(e0 @ e0)
        ia, ib = slot[ca] * 3 + np.arange(3), slot[cb] * 3 + np.arange(3)
        h[ia, ia] += w
        h[ib, ib] += w
        h[ia, ib] -= w
        h[ib, ia] -= w
        b[ia] += w * e0
        b[ib] -= w * e0
    return cost, h, b


def stereo_translation_prior(t_ref, translations, baseline, walk_sigma, baseline_sigma, linearize=False):
    """Random walk on the stereo offset and a factory baseline-length prior, in metres."""
    offsets = translations - t_ref
    walk = np.diff(offsets, axis=0) / walk_sigma
    lengths = np.linalg.norm(offsets, axis=1)
    length_error = (lengths - baseline) / baseline_sigma
    cost = 0.5 * (np.sum(walk ** 2) + length_error @ length_error)
    if not linearize:
        return cost

    n = len(translations)
    direction = offsets / np.where(lengths > 0, lengths, 1.0)[:, None] / baseline_sigma
    h = np.zeros((n, 3, n, 3))
    i = np.arange(n)
    h[i, :, i, :] = direction[:, :, None] * direction[:, None, :]
    b = -direction * length_error[:, None]
    w = np.eye(3) / walk_sigma ** 2
    h[i[:-1], :, i[:-1], :] += w
    h[i[1:], :, i[1:], :] += w
    h[i[:-1], :, i[1:], :] -= w
    h[i[1:], :, i[:-1], :] -= w
    b[:-1] += walk / walk_sigma
    b[1:] -= walk / walk_sigma
    return h.reshape(n * 3, n * 3), b.ravel()


def focal_prior(foc, pairs, slot, n_par, n_ext, w_pair, w_abs, w_mean=0.0, groups=()):
    """Cost, Hessian and right-hand side of the priors on the focal length scales s (the last parameter of
    each estimated camera's slot): s ~ N(0, 1 / w_abs), s_b - s_a ~ N(0, 1 / w_pair) for each pair, and the
    mean of each group of cameras (a physical camera's segments) ~ N(0, 1 / w_mean)."""
    ne = n_par * n_ext
    h, b, cost = np.zeros((ne, ne)), np.zeros(ne), 0.0
    for g in groups:
        if w_mean <= 0 or not len(g):
            continue
        idx = np.array([slot[c] * n_par + n_par - 1 for c in g])
        m = np.mean(foc[list(g)])
        h[np.ix_(idx, idx)] += w_mean / len(g) ** 2
        b[idx] -= w_mean * m / len(g)
        cost += 0.5 * w_mean * m ** 2
    for c in np.flatnonzero(slot >= 0):
        i = slot[c] * n_par + n_par - 1
        h[i, i] += w_abs
        b[i] -= w_abs * foc[c]
        cost += 0.5 * w_abs * foc[c] ** 2
    wps = np.broadcast_to(np.asarray(w_pair, dtype=np.float64), (len(pairs),))  # one weight, or one per pair
    for (ca, cb), w_pair in zip(pairs, wps):
        i, j = slot[ca] * n_par + n_par - 1, slot[cb] * n_par + n_par - 1
        r = foc[cb] - foc[ca]
        h[i, i] += w_pair
        h[j, j] += w_pair
        h[i, j] -= w_pair
        h[j, i] -= w_pair
        b[i] += w_pair * r
        b[j] -= w_pair * r
        cost += 0.5 * w_pair * r ** 2
    return cost, h, b


def _row_dot(x, a, i, y, b, j):
    """sum_k x[a, i, k] y[b, j, k] over the landmark's 3 coordinates."""
    return x[a, i, 0] * y[b, j, 0] + x[a, i, 1] * y[b, j, 1] + x[a, i, 2] * y[b, j, 2]


if njit is not None:
    _row_dot = njit(inline="always")(_row_dot)


def _schur_add(m_o, h_xl, h_xx, g_x, g_l, kf, slot, lm, lm_start, band, nb, ab, border, corner, rhs):
    """Eliminates the landmarks into the reduced system, one landmark at a time (plain loops, for numba).

    Every ordered pair (a, b) of a landmark's observations adds -M_a X_b^T (M = H_xl H_ll^-1, X = H_xl,
    (6 + P) x 3 each: pose 6, then the P camera parameters of the observation's slot): its pose-pose
    block when kf_a >= kf_b within the band (lower banded storage ab[i - j, j]), its pose-camera block to
    the border and its camera-camera block to the corner. Every observation then adds its own H_xx
    blocks and its reduced gradient -g_x + M g_l.
    """
    npar = h_xl.shape[1] - 6
    for l in range(lm_start.shape[0] - 1):
        s0, s1 = lm_start[l], lm_start[l + 1]
        for a in range(s0, s1):
            ra, sa = kf[a] * DOF, slot[a]
            for b in range(s0, s1):
                cb, sb = kf[b] * DOF, slot[b]
                if kf[a] >= kf[b] and kf[a] - kf[b] <= band:
                    for i in range(6):
                        for j in range(6):
                            if ra + i >= cb + j:
                                ab[ra + i - cb - j, cb + j] -= _row_dot(m_o, a, i, h_xl, b, j)
                if sb >= 0:
                    for i in range(6):
                        for j in range(npar):
                            border[ra + i, sb * npar + j] -= _row_dot(m_o, a, i, h_xl, b, 6 + j)
                    if sa >= 0:
                        for i in range(npar):
                            for j in range(npar):
                                corner[sa * npar + i, sb * npar + j] -= _row_dot(m_o, a, 6 + i, h_xl, b, 6 + j)
    for o in range(kf.shape[0]):
        r0, so = kf[o] * DOF, slot[o]
        for i in range(6):
            for j in range(i + 1):
                ab[i - j, r0 + j] += h_xx[o, i, j]
        gl = g_l[lm[o]]
        for i in range(6 + npar):
            red = -g_x[o, i] + m_o[o, i, 0] * gl[0] + m_o[o, i, 1] * gl[1] + m_o[o, i, 2] * gl[2]
            if i < 6:
                rhs[r0 + i] += red
            elif so >= 0:
                rhs[nb + so * npar + i - 6] += red
        if so >= 0:
            for i in range(6):
                for j in range(npar):
                    border[r0 + i, so * npar + j] += h_xx[o, i, 6 + j]
            for i in range(npar):
                for j in range(npar):
                    corner[so * npar + i, so * npar + j] += h_xx[o, 6 + i, 6 + j]


# compiled per run (about 3 s for the three kernels): a cache written by another import context can fail to load
_schur_add_jit = njit(nogil=True)(_schur_add) if njit is not None else None


def _fish(a, b, kc):
    """The projection's distortion at the normalized point (a, b) = (X/Z, Y/Z): (s, q) with the image point
    f s (a, b) + c and d(s (a, b))/d(a, b) = s I + q (a, b)(a, b)^T. A pinhole camera (kc[4] == 0) has s = 1,
    q = 0; a fisheye (kc[4] == 1) the factory radial model r(theta) = theta + k0 theta^3 + ... + k5 theta^13
    (kc[5:11]), s = r(atan rho) / rho with rho = |(a, b)|."""
    if kc[4] == 0.0:
        return 1.0, 0.0
    rho2 = a * a + b * b
    if rho2 < 1e-12:
        return 1.0, 2.0 * (kc[5] - 1.0 / 3.0)
    rho = np.sqrt(rho2)
    th = np.arctan(rho)
    t2 = th * th
    r, dr, tp = th, 1.0, th
    for i in range(6):
        tp = tp * t2  # theta^(2i+3)
        r += kc[5 + i] * tp
        dr += kc[5 + i] * (2 * i + 3) * tp / th
    sv = r / rho
    return sv, (dr / (1.0 + rho2) - sv) / rho2


_fish_jit = njit(nogil=True)(_fish) if njit is not None else _fish


def _normal_eqs(r_wi, t_wi, r_ic, t_ic, pts, kf, cam, lm, z, kcam, est_rot, est_trans, est_foc, active, sigma, huber,
                h_xx, h_xl, g_x, h_ll_o, h_ll, g_l, o0=0, o1=-1):
    """Reprojection.linearize and its normal equations in one pass (plain loops, for numba).

    Each active observation in front of its camera adds its Huber-weighted blocks H_xx (n x n), H_xl (n x 3),
    g_x (n) and H_ll (3 x 3) at its own index, and H_ll and g_l to its landmark; the arrays start at zero.
    The columns are pose, camera rotation, optional camera translation, then optional focal scale.
    Observations o0 to o1 (default: all).
    """
    nx = h_xx.shape[1]
    jpi, jr, jx, jl = np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, nx)), np.zeros((2, 3))
    p_i, p_c, d = np.zeros(3), np.zeros(3), np.zeros(3)
    for o in range(o0, kf.shape[0] if o1 < 0 else o1):
        k, c, l = kf[o], cam[o], lm[o]
        for a in range(3):
            d[a] = pts[l, a] - t_wi[k, a]
        for a in range(3):  # p_i = R_wi^T (X - t_wi)
            p_i[a] = r_wi[k, 0, a] * d[0] + r_wi[k, 1, a] * d[1] + r_wi[k, 2, a] * d[2]
        for a in range(3):
            d[a] = p_i[a] - t_ic[c, a]
        for a in range(3):  # p_c = R_ic^T (p_i - t_ic)
            p_c[a] = r_ic[c, 0, a] * d[0] + r_ic[c, 1, a] * d[1] + r_ic[c, 2, a] * d[2]
        if not active[o] or p_c[2] <= 0.05:
            continue
        zs, sg = p_c[2], sigma[o]
        na, nb_ = p_c[0] / zs, p_c[1] / zs
        sv, q = _fish_jit(na, nb_, kcam[c])
        ga, gb = sv * na, sv * nb_
        res0 = (kcam[c, 0] * ga + kcam[c, 2] - z[o, 0]) / sg
        res1 = (kcam[c, 1] * gb + kcam[c, 3] - z[o, 1]) / sg
        e = np.sqrt(res0 * res0 + res1 * res1)
        w = (1.0 if e <= huber else huber / max(e, 1e-12)) if huber > 0 else 1.0 / (1.0 + (e / huber) ** 2)
        # d(g)/d(a, b) times d(a, b)/d(p_c) = [[1, 0, -a], [0, 1, -b]] / Z, scaled by f / sigma
        m00, m01, m11 = sv + q * na * na, q * na * nb_, sv + q * nb_ * nb_
        f0, f1 = kcam[c, 0] / zs / sg, kcam[c, 1] / zs / sg
        jpi[0, 0], jpi[0, 1], jpi[0, 2] = f0 * m00, f0 * m01, -f0 * (m00 * na + m01 * nb_)
        jpi[1, 0], jpi[1, 1], jpi[1, 2] = f1 * m01, f1 * m11, -f1 * (m01 * na + m11 * nb_)
        for r in range(2):
            for b in range(3):  # jr = jpi R_ic^T
                jr[r, b] = jpi[r, 0] * r_ic[c, b, 0] + jpi[r, 1] * r_ic[c, b, 1] + jpi[r, 2] * r_ic[c, b, 2]
            # pose rotation jr [p_i]x, pose translation -jr, camera rotation jpi [p_c]x, landmark jr R_wi^T
            jx[r, 0] = jr[r, 1] * p_i[2] - jr[r, 2] * p_i[1]
            jx[r, 1] = jr[r, 2] * p_i[0] - jr[r, 0] * p_i[2]
            jx[r, 2] = jr[r, 0] * p_i[1] - jr[r, 1] * p_i[0]
            for b in range(3):
                jx[r, 3 + b] = -jr[r, b]
            if est_rot[c]:
                jx[r, 6] = jpi[r, 1] * p_c[2] - jpi[r, 2] * p_c[1]
                jx[r, 7] = jpi[r, 2] * p_c[0] - jpi[r, 0] * p_c[2]
                jx[r, 8] = jpi[r, 0] * p_c[1] - jpi[r, 1] * p_c[0]
            else:
                jx[r, 6] = jx[r, 7] = jx[r, 8] = 0.0
            if nx >= 12:
                for b in range(3):
                    jx[r, 9 + b] = -jr[r, b] if est_trans[c] else 0.0
            if nx == 10 or nx == 13:
                jx[r, nx - 1] = kcam[c, r] * (ga if r == 0 else gb) / sg if est_foc[c] else 0.0
            for b in range(3):
                jl[r, b] = jr[r, 0] * r_wi[k, b, 0] + jr[r, 1] * r_wi[k, b, 1] + jr[r, 2] * r_wi[k, b, 2]
        for i in range(nx):
            for j in range(nx):
                h_xx[o, i, j] = w * (jx[0, i] * jx[0, j] + jx[1, i] * jx[1, j])
            for j in range(3):
                h_xl[o, i, j] = w * (jx[0, i] * jl[0, j] + jx[1, i] * jl[1, j])
            g_x[o, i] = w * (jx[0, i] * res0 + jx[1, i] * res1)
        for i in range(3):
            for j in range(3):
                v = w * (jl[0, i] * jl[0, j] + jl[1, i] * jl[1, j])
                h_ll_o[o, i, j] = v
                h_ll[l, i, j] += v
            g_l[l, i] += w * (jl[0, i] * res0 + jl[1, i] * res1)


def _reproj_cost(r_wi, t_wi, r_ic, t_ic, pts, kf, cam, lm, z, kcam, active, sigma, huber, e, o0=0, o1=-1):
    """Reprojection.cost in one pass (plain loops, for numba): fills e, returns the summed Huber cost."""
    total = 0.0
    d, p_i = np.zeros(3), np.zeros(3)
    for o in range(o0, kf.shape[0] if o1 < 0 else o1):
        k, c, l = kf[o], cam[o], lm[o]
        for a in range(3):
            d[a] = pts[l, a] - t_wi[k, a]
        for a in range(3):
            p_i[a] = r_wi[k, 0, a] * d[0] + r_wi[k, 1, a] * d[1] + r_wi[k, 2, a] * d[2]
        for a in range(3):
            d[a] = p_i[a] - t_ic[c, a]
        x = r_ic[c, 0, 0] * d[0] + r_ic[c, 1, 0] * d[1] + r_ic[c, 2, 0] * d[2]
        y = r_ic[c, 0, 1] * d[0] + r_ic[c, 1, 1] * d[1] + r_ic[c, 2, 1] * d[2]
        zc = r_ic[c, 0, 2] * d[0] + r_ic[c, 1, 2] * d[1] + r_ic[c, 2, 2] * d[2]
        valid = zc > 0.05
        zs = zc if valid else 1.0
        sv, _ = _fish_jit(x / zs, y / zs, kcam[c])
        r0 = (kcam[c, 0] * sv * x / zs + kcam[c, 2] - z[o, 0]) / sigma[o]
        r1 = (kcam[c, 1] * sv * y / zs + kcam[c, 3] - z[o, 1]) / sigma[o]
        e[o] = np.sqrt(r0 * r0 + r1 * r1)
        if valid and active[o]:
            if huber < 0:  # Cauchy at scale -huber
                total += 0.5 * huber * huber * np.log1p((e[o] / huber) ** 2)
            else:
                total += 0.5 * e[o] * e[o] if e[o] <= huber else huber * (e[o] - 0.5 * huber)
    return total


def _landmark_m(h_xl, h_ll_inv, lm, m_o):
    """m_o = H_xl H_ll^-1 of each observation's landmark (plain loops, for numba)."""
    for o in range(h_xl.shape[0]):
        l = lm[o]
        for i in range(h_xl.shape[1]):
            for j in range(3):
                m_o[o, i, j] = (h_xl[o, i, 0] * h_ll_inv[l, 0, j] + h_xl[o, i, 1] * h_ll_inv[l, 1, j]
                                + h_xl[o, i, 2] * h_ll_inv[l, 2, j])


_normal_eqs_jit = njit(nogil=True)(_normal_eqs) if njit is not None else None
_landmark_m_jit = njit(nogil=True)(_landmark_m) if njit is not None else None
_reproj_cost_jit = njit(nogil=True)(_reproj_cost) if njit is not None else None


def _normal_eqs_chunks(bounds, r_wi, t_wi, r_ic, t_ic, pts, kf, cam, lm, z, kcam, est_rot, est_trans, est_foc, active, sigma, huber,
                       h_xx, h_xl, g_x, h_ll_o, h_ll, g_l):
    """_normal_eqs on chunks of whole landmarks in parallel (bounds: the first observation of each chunk)."""
    for ch in prange(bounds.shape[0] - 1):
        _normal_eqs_jit(r_wi, t_wi, r_ic, t_ic, pts, kf, cam, lm, z, kcam, est_rot, est_trans, est_foc, active, sigma, huber,
                        h_xx, h_xl, g_x, h_ll_o, h_ll, g_l, bounds[ch], bounds[ch + 1])


def _reproj_cost_chunks(bounds, r_wi, t_wi, r_ic, t_ic, pts, kf, cam, lm, z, kcam, active, sigma, huber, e):
    """_reproj_cost on chunks in parallel; the chunk sums add up in order, so the total is reproducible."""
    part = np.zeros(bounds.shape[0] - 1)
    for ch in prange(bounds.shape[0] - 1):
        part[ch] = _reproj_cost_jit(r_wi, t_wi, r_ic, t_ic, pts, kf, cam, lm, z, kcam, active, sigma, huber, e,
                                    bounds[ch], bounds[ch + 1])
    total = 0.0
    for ch in range(part.shape[0]):
        total += part[ch]
    return total


# the threads come from NUMBA_NUM_THREADS (default: all cores)
_normal_eqs_par = njit(nogil=True, parallel=True)(_normal_eqs_chunks) if njit is not None else None
_reproj_cost_par = njit(nogil=True, parallel=True)(_reproj_cost_chunks) if njit is not None else None


class Reprojection:
    """All reprojection observations, sorted by landmark, with the fixed Schur pair structure.

    Each observation connects the keyframe pose (GTSAM Pose3 retract), the camera's rotation
    (right perturbation), optional translation (additive in the IMU frame), optional log focal
    scale and the landmark (world frame). Fixed camera parameters have zero columns.
    """

    def __init__(self, obs_kf, obs_cam, obs_lm, obs_z, n_lm, cams, est_cams, sigma, huber, band, chunk=400_000,
                 numpy_pairs=None, est_focal=False, fixed_rotations=(), translation_cams=()):
        order = np.lexsort((obs_kf, obs_lm))
        self.kf, self.cam, self.lm, self.z = obs_kf[order], obs_cam[order], obs_lm[order], obs_z[order]
        self.n_lm, self.huber, self.chunk = n_lm, huber, chunk
        # pixel noise of each observation (sigma may be one value for all)
        self.sigma = np.broadcast_to(np.asarray(sigma, np.float64), order.shape)[order].copy()
        self.active = np.ones(len(self.kf), bool)  # observations not rejected as outliers
        for name, i, j in (("fx", 0, 0), ("fy", 1, 1), ("cx", 0, 2), ("cy", 1, 2)):
            setattr(self, name, np.array([c["K"][i, j] for c in cams])[self.cam])
        # fx, fy, cx, cy, then 1 and the radial coefficients k0..k5 for a fisheye ("fisheye_k"), else 0
        self.kcam = np.array([[c["K"][0, 0], c["K"][1, 1], c["K"][0, 2], c["K"][1, 2],
                               1.0 if "fisheye_k" in c else 0.0, *c.get("fisheye_k", np.zeros(6))] for c in cams])
        self.kcam0 = self.kcam.copy()
        self.ext_slot = np.full(len(cams), -1)
        for i, c in enumerate(est_cams):
            self.ext_slot[c] = i
        self.n_ext = len(est_cams)
        # A camera slot contains rotation, optional translation and optional log focal scale.
        camera_ids = np.arange(len(cams))
        estimated = self.ext_slot >= 0
        self.est_rot = estimated & ~np.isin(camera_ids, fixed_rotations)
        self.est_trans = np.isin(camera_ids, translation_cams)
        self.est_foc = estimated & est_focal
        self.n_par = 3 + 3 * bool(len(translation_cams)) + est_focal
        self.foc = np.zeros(len(cams))
        self.band = band
        counts = np.bincount(self.lm, minlength=n_lm)
        start = np.r_[0, np.cumsum(counts)[:-1]]
        self.lm_start = np.r_[0, np.cumsum(counts)].astype(np.int64)  # observations of landmark l: [l], [l + 1]
        # chunks of whole landmarks for the parallel kernels
        self.chunks = self.lm_start[np.linspace(0, n_lm, min(64, max(n_lm, 1)) + 1).astype(np.int64)]
        # every ordered pair (a, b) of observations of the same landmark
        m = counts[self.lm]
        pa = np.repeat(np.arange(len(self.lm)), m)
        pb = start[self.lm[pa]] + np.arange(len(pa)) - np.repeat(np.r_[0, np.cumsum(m)[:-1]], m)
        print(f"  {len(self.lm)} observations, {len(pa)} Schur pairs", flush=True)
        ka, kb = self.kf[pa], self.kf[pb]
        # pose-pose pairs further apart than --band keyframes (revisits) stay outside the banded factor
        off = np.flatnonzero((ka >= kb) & (ka - kb > band))
        self.off_keys, oinv = np.unique(ka[off].astype(np.int64) * 1_000_000 + kb[off], return_inverse=True)
        s = np.argsort(oinv, kind="stable")
        self.off_a, self.off_b, self.off_bid = pa[off][s].astype(np.int32), pb[off][s].astype(np.int32), oinv[s]
        self.off_ka, self.off_kb = self.off_keys // 1_000_000, self.off_keys % 1_000_000
        if len(off):
            print(f"  {len(self.off_keys)} keyframe blocks outside the band (revisits)", flush=True)
        # lower-band pose-pose pairs (kf_a >= kf_b), sorted by their (kf_a, kf_b) block
        low = np.flatnonzero((ka >= kb) & (ka - kb <= band))
        self.band_kf = int((ka - kb)[low].max()) if len(low) else 1
        # the pair lists below serve the numpy elimination only (without numba)
        self.numpy_pairs = _schur_add_jit is None if numpy_pairs is None else numpy_pairs
        if not self.numpy_pairs:
            return
        self.blk_keys, inv = np.unique(ka[low].astype(np.int64) * 1_000_000 + kb[low], return_inverse=True)
        s = np.argsort(inv, kind="stable")
        self.low_a, self.low_b, self.low_bid = pa[low][s].astype(np.int32), pb[low][s].astype(np.int32), inv[s]
        self.blk_a, self.blk_b = self.blk_keys // 1_000_000, self.blk_keys % 1_000_000
        # border pairs (pose of a x extrinsic of b), sorted by (kf_a, slot_b); corner pairs (both extrinsic)
        slot = self.ext_slot[self.cam]
        if self.n_ext:
            hb = np.flatnonzero(slot[pb] >= 0)
            key = self.kf[pa[hb]].astype(np.int64) * self.n_ext + slot[pb[hb]]
            o = np.argsort(key, kind="stable")
            self.bd_a, self.bd_b, self.bd_key = pa[hb][o].astype(np.int32), pb[hb][o].astype(np.int32), key[o]
            both = np.flatnonzero((slot[pa] >= 0) & (slot[pb] >= 0))
            self.cn_a, self.cn_b = pa[both].astype(np.int32), pb[both].astype(np.int32)
        # direct (same-observation) pose-pose terms, reduced per keyframe
        self.obs_sort = np.argsort(self.kf, kind="stable")
        self.obs_starts = np.flatnonzero(np.r_[True, np.diff(self.kf[self.obs_sort]) != 0])
        self.obs_kfs = self.kf[self.obs_sort][self.obs_starts]

    def set_focal(self, foc):
        """Focal length scales s per camera (f = f_calib exp(s))."""
        self.foc = np.asarray(foc, np.float64).copy()
        self.kcam = self.kcam0.copy()
        self.kcam[:, 0:2] *= np.exp(self.foc)[:, None]
        self.fx, self.fy = self.kcam[self.cam, 0], self.kcam[self.cam, 1]

    def _project(self, r_wi, t_wi, r_ic, t_ic, pts):
        rw = r_wi[self.kf]
        p_i = np.einsum("nji,nj->ni", rw, pts[self.lm] - t_wi[self.kf])
        rci = np.transpose(r_ic[self.cam], (0, 2, 1))
        p_c = np.einsum("nij,nj->ni", rci, p_i - t_ic[self.cam])
        valid = p_c[:, 2] > 0.05
        zs = np.where(valid, p_c[:, 2], 1.0)
        nab = p_c[:, :2] / zs[:, None]
        sq = np.array([_fish(na, nb_, self.kcam[c]) for (na, nb_), c in zip(nab, self.cam)]).reshape(-1, 2)
        self._sq = sq
        g = nab * sq[:, :1]
        res = np.stack([self.fx * g[:, 0] + self.cx - self.z[:, 0],
                        self.fy * g[:, 1] + self.cy - self.z[:, 1]], 1) / self.sigma[:, None]
        return rw, p_i, rci, p_c, zs, valid, res

    def _robust(self, res, valid):
        e = np.linalg.norm(res, axis=1)
        k = self.huber
        valid = valid & self.active
        if k < 0:  # Cauchy at scale -k
            w = np.where(valid, 1.0 / (1.0 + (e / k) ** 2), 0.0)
            cost = np.where(valid, 0.5 * k * k * np.log1p((e / k) ** 2), 0.0)
        else:
            w = np.where(valid, np.where(e <= k, 1.0, k / np.maximum(e, 1e-12)), 0.0)
            cost = np.where(valid, np.where(e <= k, 0.5 * e * e, k * (e - 0.5 * k)), 0.0)
        return e, w, cost

    def cost(self, r_wi, t_wi, r_ic, t_ic, pts):
        if _reproj_cost_par is not None:
            e = np.empty(len(self.kf))
            total = _reproj_cost_par(self.chunks, r_wi, t_wi, r_ic, t_ic, pts, self.kf, self.cam, self.lm, self.z,
                                     self.kcam, self.active, self.sigma, self.huber, e)
            return float(total), e
        *_, valid, res = self._project(r_wi, t_wi, r_ic, t_ic, pts)
        e, _, cost = self._robust(res, valid)
        return float(cost.sum()), e

    def linearize(self, r_wi, t_wi, r_ic, t_ic, pts):
        rw, p_i, rci, p_c, zs, valid, res = self._project(r_wi, t_wi, r_ic, t_ic, pts)
        na, nb_ = p_c[:, 0] / zs, p_c[:, 1] / zs
        sv, q = self._sq[:, 0], self._sq[:, 1]
        m00, m01, m11 = sv + q * na * na, q * na * nb_, sv + q * nb_ * nb_
        f0, f1 = self.fx / zs / self.sigma, self.fy / zs / self.sigma
        jpi = np.zeros((len(zs), 2, 3))
        jpi[:, 0, 0], jpi[:, 0, 1], jpi[:, 0, 2] = f0 * m00, f0 * m01, -f0 * (m00 * na + m01 * nb_)
        jpi[:, 1, 0], jpi[:, 1, 1], jpi[:, 1, 2] = f1 * m01, f1 * m11, -f1 * (m01 * na + m11 * nb_)
        jr = jpi @ rci
        jx = np.zeros((len(zs), 2, 6 + self.n_par))
        jx[:, :, 0:3] = jr @ skew(p_i)          # pose rotation
        jx[:, :, 3:6] = -jr                     # pose translation
        jx[:, :, 6:9] = jpi @ skew(p_c)         # extrinsic rotation
        jx[~self.est_rot[self.cam], :, 6:9] = 0.0
        if self.n_par >= 6:
            jx[:, :, 9:12] = -jr * self.est_trans[self.cam, None, None]
        if self.n_par in (4, 7):
            jx[:, 0, -1] = self.fx * sv * na / self.sigma * self.est_foc[self.cam]
            jx[:, 1, -1] = self.fy * sv * nb_ / self.sigma * self.est_foc[self.cam]
        jl = jr @ np.transpose(rw, (0, 2, 1))   # landmark
        _, w, _ = self._robust(res, valid)
        return res, jx, jl, w

    def normal_equations(self, r_wi, t_wi, r_ic, t_ic, pts):
        """Per observation H_xx, H_xl, g_x and H_ll (weighted), and per landmark H_ll and g_l."""
        n_o, nx = len(self.kf), 6 + self.n_par
        h_xx, h_xl, g_x = np.zeros((n_o, nx, nx)), np.zeros((n_o, nx, 3)), np.zeros((n_o, nx))
        h_ll_o, h_ll, g_l = np.zeros((n_o, 3, 3)), np.zeros((self.n_lm, 3, 3)), np.zeros((self.n_lm, 3))
        if _normal_eqs_par is not None:
            _normal_eqs_par(self.chunks, r_wi, t_wi, r_ic, t_ic, pts, self.kf, self.cam, self.lm, self.z, self.kcam,
                            self.est_rot, self.est_trans, self.est_foc, self.active, self.sigma, self.huber,
                            h_xx, h_xl, g_x, h_ll_o, h_ll, g_l)
            return h_xx, h_xl, g_x, h_ll_o, h_ll, g_l
        res, jx, jl, w = self.linearize(r_wi, t_wi, r_ic, t_ic, pts)
        wjx, wjl = jx * w[:, None, None], jl * w[:, None, None]
        h_xx = np.einsum("nki,nkj->nij", wjx, jx)
        h_xl = np.einsum("nki,nkj->nij", wjx, jl)
        g_x = np.einsum("nki,nk->ni", wjx, res)
        h_ll_o = np.einsum("nki,nkj->nij", wjl, jl)
        np.add.at(h_ll, self.lm, h_ll_o)
        np.add.at(g_l, self.lm, np.einsum("nki,nk->ni", wjl, res))
        return h_xx, h_xl, g_x, h_ll_o, h_ll, g_l


def _imu_graph(kf_ns, kf_p, kf_r, vio, imu_path, noise, noise_scale=1.0, accel_rw_scale=1.0, gyro_rw_scale=1.0,
               prior_rot_sigma=1e-4, omega_coriolis=None, init=None, warmup=(1.0, 300.0, None), gyro_m=None, accel_m=None,
               temp=None, gyro_temp=None, time_offsets=None, bias_chunk=None):
    """GTSAM values and factors for poses, velocities, biases, the IMU and the gauge priors.

    The first keyframe's position is pinned (1e-4 m); its rotation gets prior_rot_sigma (rad). Gravity
    makes roll and pitch observable, so a loose rotation prior lets the solve fix a tilt the VIO got
    wrong at its start instead of pinning it there. Velocities start from the trajectory's finite
    differences and biases from zero, or from init (kf_ns, vel, bias of an earlier solve) where it has the
    keyframe; keyframes after its last one start from its last bias. warmup (scale, seconds, start_ns) loosens the
    gyro bias random walk at the start of the recording, by 1 + (scale - 1) exp(-t / seconds) at t seconds after
    start_ns (default: the first keyframe), while the device warms up. imu_path is a csv path or the loaded
    array; gyro_m (3 x 3) is the gyroscope's response error M: the rate is (I + M) times the measured one.
    temp (one per IMU sample, degrees from a reference) and gyro_temp (6: the bias's change per degree c, then each
    axis's scale change per degree s) make the rate (I + M) w + dT (s * w) - dT c, less the bias.
    time_offsets (2, seconds): the IMU's clock against the cameras' (a sample stamped t was taken at t + d0) and
    the accelerometer's against the gyroscope's (its reading at t is the one stamped t + d1, linearly
    interpolated). bias_chunk (seconds, accelerometer sigma, gyroscope sigma) holds the biases nearly constant
    within chunks of that many seconds from the first keyframe and lets them step by the sigmas between chunks
    (SLAM Dunk's bias per 5 s chunk), in place of the calibration's continuous walk and its warm-up.
    """
    import gtsam
    from gtsam.symbol_shorthand import B, V, X
    vio_ns, vio_p, _ = vio
    imu = np.loadtxt(imu_path, delimiter=",", comments="#") if isinstance(imu_path, (str, os.PathLike)) \
        else np.array(imu_path, dtype=np.float64)
    if gyro_m is not None:
        imu[:, 1:4] = imu[:, 1:4] @ (np.eye(3) + np.asarray(gyro_m).reshape(3, 3)).T
    if accel_m is not None:
        imu[:, 4:7] = imu[:, 4:7] @ (np.eye(3) + np.asarray(accel_m).reshape(3, 3)).T
    if temp is not None and gyro_temp is not None:
        gt_ = np.asarray(gyro_temp, dtype=np.float64)
        imu[:, 1:4] += temp[:, None] * (imu[:, 1:4] * gt_[3:6] - gt_[0:3])
    imu_ns = imu[:, 0].astype(np.int64)
    if time_offsets is not None and np.any(time_offsets):
        d_cam, d_acc = float(time_offsets[0]), float(time_offsets[1])
        if d_acc:
            tf = (imu_ns - imu_ns[0]).astype(np.float64)
            for col in range(4, 7):
                imu[:, col] = np.interp(tf + d_acc * 1e9, tf, imu[:, col])
        imu_ns = imu_ns + int(round(d_cam * 1e9))

    def make_params(gyro_rw, accel_rw=None):
        accel_rw = accel_rw_scale if accel_rw is None else accel_rw
        params = gtsam.PreintegrationCombinedParams.MakeSharedU(9.81)
        params.setAccelerometerCovariance(np.eye(3) * (noise_scale * noise["accel_noise_std"]) ** 2)
        params.setGyroscopeCovariance(np.eye(3) * (noise_scale * noise["gyro_noise_std"]) ** 2)
        params.setIntegrationCovariance(np.eye(3) * 1e-8)
        params.setBiasAccCovariance(np.eye(3) * (accel_rw * noise["accel_bias_std"]) ** 2)
        params.setBiasOmegaCovariance(np.eye(3) * (gyro_rw * noise["gyro_bias_std"]) ** 2)
        if omega_coriolis is not None:  # Earth's rotation in the world frame, which the gyroscope also measures
            params.setOmegaCoriolis(np.asarray(omega_coriolis, float))
        return params

    warm_scale, warm_s, warm_t0 = warmup[:3]
    warm_accel = warmup[3] if len(warmup) > 3 else 1.0  # the accelerometer's bias walk scale at the start
    warm_t0 = kf_ns[0] if warm_t0 is None else warm_t0
    by_scale = {(1.0, 1.0): make_params(gyro_rw_scale)}

    by_rw = {}

    def chunk_params(ta, tb):  # the bias walk of a factor that stays inside a chunk or crosses into the next
        sec, s_acc, s_gyr = bias_chunk
        dt = max((tb - ta) * 1e-9, 1e-3)
        step = int((tb - kf_ns[0]) // round(sec * 1e9)) != int((ta - kf_ns[0]) // round(sec * 1e9))
        f = 1.0 if step else 1e-3  # inside a chunk: a thousandth of the step's sigma
        key = (round(f * s_acc / np.sqrt(dt), 12), round(f * s_gyr / np.sqrt(dt), 14))
        if key not in by_rw:
            params = make_params(1.0)
            params.setBiasAccCovariance(np.eye(3) * key[0] ** 2)
            params.setBiasOmegaCovariance(np.eye(3) * key[1] ** 2)
            by_rw[key] = params
        return by_rw[key]

    def factor_params(t_ns):  # the bias random walks of the warm-up at t_ns, rounded to share the params
        decay = np.exp(-(t_ns - warm_t0) * 1e-9 / warm_s)
        f = max(round(1.0 + (warm_scale - 1.0) * decay, 2), 1.0)
        fa = max(round(1.0 + (warm_accel - 1.0) * decay, 2), 1.0)
        if (f, fa) not in by_scale:
            by_scale[(f, fa)] = make_params(f * gyro_rw_scale, fa * accel_rw_scale)
        return by_scale[(f, fa)]
    graph, values = gtsam.NonlinearFactorGraph(), gtsam.Values()
    imu_factors = []
    zero_bias = gtsam.imuBias.ConstantBias()
    n = len(kf_ns)
    for k in range(n):
        i = np.searchsorted(vio_ns, kf_ns[k])
        i0, i1 = max(i - 1, 0), min(i + 1, len(vio_ns) - 1)
        vel = (vio_p[i1] - vio_p[i0]) / ((vio_ns[i1] - vio_ns[i0]) * 1e-9)
        bias0 = zero_bias
        if init is not None and kf_ns[k] >= init["kf_ns"][0]:
            j = min(np.searchsorted(init["kf_ns"], kf_ns[k]), len(init["kf_ns"]) - 1)
            if init["kf_ns"][j] == kf_ns[k]:
                vel = init["vel"][j]
            bias0 = gtsam.imuBias.ConstantBias(init["bias"][j, 0:3], init["bias"][j, 3:6])
        values.insert(X(k), gtsam.Pose3(gtsam.Rot3(kf_r[k]), kf_p[k]))
        values.insert(V(k), vel)
        values.insert(B(k), bias0)
        if k == 0:
            graph.add(gtsam.PriorFactorPose3(X(0), values.atPose3(X(0)), gtsam.noiseModel.Diagonal.Sigmas(
                np.r_[np.full(3, prior_rot_sigma), np.full(3, 1e-4)])))
            graph.add(gtsam.PriorFactorVector(V(0), vel, gtsam.noiseModel.Isotropic.Sigma(3, 0.1)))
            graph.add(gtsam.PriorFactorConstantBias(B(0), zero_bias, gtsam.noiseModel.Diagonal.Sigmas(
                np.array([0.1, 0.1, 0.1, 1e-3, 1e-3, 1e-3]))))
    for k in range(n - 1):
        ta, tb = kf_ns[k], kf_ns[k + 1]
        pim = gtsam.PreintegratedCombinedMeasurements(
            chunk_params(ta, tb) if bias_chunk else factor_params(ta), zero_bias)
        s = max(np.searchsorted(imu_ns, ta, side="right") - 1, 0)
        e = np.searchsorted(imu_ns, tb, side="left")
        for i in range(s, min(e, len(imu_ns) - 1)):
            lo, hi = max(imu_ns[i], ta), min(imu_ns[i + 1], tb)
            if hi > lo:
                pim.integrateMeasurement(imu[i, 4:7], imu[i, 1:4], (hi - lo) * 1e-9)
        imu_factors.append(gtsam.CombinedImuFactor(X(k), V(k), X(k + 1), V(k + 1), B(k), B(k + 1), pim))
        graph.add(imu_factors[-1])
    index = {}
    for k in range(n):
        index[X(k)], index[V(k)], index[B(k)] = (k * DOF, 6), (k * DOF + 6, 3), (k * DOF + 9, 6)
    return graph, values, index, imu_factors


def expand_modes(spec):
    """The --modes list; yawN stands for N pieces yaw:0:N .. yaw:N-1:N of the heading drift."""
    out = []
    for m in (x for x in spec.split(",") if x):
        if m.startswith("yaw") and m[3:].isdigit():
            out += [f"yaw:{j}:{m[3:]}" for j in range(int(m[3:]))]
        else:
            out.append(m)
    return out


def _mode_profile(mode, kf_s):
    """Each keyframe's share s of a mode's amplitude: from 0 at the first keyframe to 1 at the last, or
    for the piece yaw:j:N, from 0 to 1 over the j-th of N equal time pieces (0 before, 1 after)."""
    s = (kf_s - kf_s[0]) / max(kf_s[-1] - kf_s[0], 1e-9)
    if mode.startswith("yaw:"):
        j, npieces = (int(x) for x in mode.split(":")[1:])
        s = np.clip(s * npieces - j, 0.0, 1.0)
    return s


def mode_vectors(modes, kf_s, r_wi, t_wi, vel, pts, lm_first):
    """Tangent directions of slow deformations of the whole solution, per unit amplitude at the end.

    yaw: the heading turns by s_k rad at keyframe k (s from 0 at the first keyframe to 1 at the
    last); positions are re-integrated from the turned displacements, velocities turn, landmarks
    move with their first keyframe and the gyro bias takes one offset, the mean body-frame vertical,
    so that the preintegrated rotations still agree. scale: displacements stretch by 1 + s_k.
    yaw:j:N: the heading turns over the j-th of N time pieces only. The gyro bias takes the same
    offset as for yaw, the piece's turn spread over the whole recording: the pieces' mean rate is
    what a constant bias absorbs, their differences the gyroscope sees.
    Returns V_x (m, n, DOF) in the tangent space of each keyframe and V_l (m, n_lm, 3).
    """
    up = np.array([0.0, 0.0, 1.0])
    dur = max(kf_s[-1] - kf_s[0], 1e-9)
    dp = np.diff(t_wi, axis=0)
    out_x, out_l = [], []
    for mode in modes:
        s = _mode_profile(mode, kf_s)
        s_mid = 0.5 * (s[1:] + s[:-1])
        v = np.zeros((len(kf_s), DOF))
        if mode == "yaw" or mode.startswith("yaw:"):
            shift = np.vstack([np.zeros(3), np.cumsum(np.cross(up, dp) * s_mid[:, None], axis=0)])
            up_body = np.einsum("nji,j->ni", r_wi, up)
            v[:, 0:3] = s[:, None] * up_body
            v[:, 6:9] = s[:, None] * np.cross(up, vel)
            v[:, 12:15] = -up_body.mean(0) / dur
            v_l = shift[lm_first] + s[lm_first, None] * np.cross(up, pts - t_wi[lm_first])
        elif mode == "scale":
            shift = np.vstack([np.zeros(3), np.cumsum(dp * s_mid[:, None], axis=0)])
            v[:, 6:9] = s[:, None] * vel
            v_l = shift[lm_first] + s[lm_first, None] * (pts - t_wi[lm_first])
        else:
            raise ValueError(f"unknown mode {mode}")
        v[:, 3:6] = np.einsum("nji,nj->ni", r_wi, shift)
        out_x.append(v)
        out_l.append(v_l)
    return np.array(out_x), np.array(out_l)


def deform(mode, beta, kf_s, r_wi, t_wi, vel, bias, pts, lm_first):
    """The finite deformation whose tangent mode_vectors gives, applied with amplitude beta.

    Applied exactly instead of through the pose retraction: GTSAM's Pose3 retraction couples
    rotation and translation, and a heading turn of 0.05 rad would move far keyframes by decimetres.
    """
    dur = max(kf_s[-1] - kf_s[0], 1e-9)
    s = _mode_profile(mode, kf_s)
    s_mid = 0.5 * (s[1:] + s[:-1])
    dp = np.diff(t_wi, axis=0)
    k = lm_first
    bias = bias.copy()
    if mode == "yaw" or mode.startswith("yaw:"):
        def rz(angle):
            c, sn, zero, one = np.cos(angle), np.sin(angle), np.zeros_like(angle), np.ones_like(angle)
            return np.stack([np.stack([c, -sn, zero], -1), np.stack([sn, c, zero], -1),
                             np.stack([zero, zero, one], -1)], -2)
        turn, turn_mid = rz(beta * s), rz(beta * s_mid)
        t_new = np.vstack([t_wi[:1], t_wi[:1] + np.cumsum(np.einsum("nij,nj->ni", turn_mid, dp), axis=0)])
        r_new = np.einsum("nij,njk->nik", turn, r_wi)
        vel_new = np.einsum("nij,nj->ni", turn, vel)
        bias[:, 3:6] -= beta * r_wi[:, 2, :].mean(0) / dur  # R^T z: the body-frame vertical
        pts_new = t_new[k] + np.einsum("nij,nj->ni", turn[k], pts - t_wi[k])
    elif mode == "scale":
        t_new = np.vstack([t_wi[:1], t_wi[:1] + np.cumsum(dp * (1 + beta * s_mid[:, None]), axis=0)])
        r_new = r_wi
        vel_new = vel * (1 + beta * s[:, None])
        pts_new = t_new[k] + (1 + beta * s[k, None]) * (pts - t_wi[k])
    else:
        raise ValueError(f"unknown mode {mode}")
    return r_new, t_new, vel_new, bias, pts_new


def _band_matvec(ab, x):
    """Symmetric banded matrix (lower storage) times a vector."""
    nb = len(x)
    y = ab[0] * x
    for d in range(1, ab.shape[0]):
        y[d:] += ab[d, :nb - d] * x[:nb - d]
        y[:nb - d] += ab[d, :nb - d] * x[d:]
    return y


def temperature_corrected(xy, offsets, kf_ns, cams, path, coef):
    """The keypoints as the calibrated focal lengths would see them, when a camera's focal length is
    (1 + coef (T - T_median)) times the calibration at its temperature T (lamaria_temperature.py's CSV; the
    sparse samples interpolated). Images alternate cam0, cam1 per keyframe."""
    tab = np.loadtxt(path, delimiter=",", skiprows=1)
    xy = xy.copy()
    for c in range(2):
        temp = np.interp(kf_ns.astype(np.float64), tab[:, 0], tab[:, 1 + c])
        scale = 1.0 / (1.0 + coef * (temp - np.median(tab[:, 1 + c])))
        pp = cams[c]["K"][:2, 2]
        for k in range(len(kf_ns)):
            o0, o1 = offsets[2 * k + c], offsets[2 * k + c + 1]
            xy[o0:o1] = pp + (xy[o0:o1] - pp) * scale[k]
    print(f"  keypoints corrected for the temperature ({coef:g} per degree)", flush=True)
    return xy


def fisheye_corrected(xy, offsets, cams, aria_calib, eps):
    """The keypoints of the rectified pinhole images as they would be if each fisheye's focal length were
    (1 + eps) times the factory's: the factory radial model r(theta) = theta + k0 theta^3 + ... + k5 theta^13
    (Aria's json), a keypoint's ray angle theta moved to where r(theta') = r(theta) / (1 + eps). A pinhole
    focal length scale would move it by tan(theta) instead, half as far as this at the image's edge.
    Images alternate cam0, cam1 per keyframe."""
    calib = json.load(open(aria_calib))
    xy = xy.copy()
    cam = np.zeros(len(xy), np.int64)
    for i in range(len(offsets) - 1):
        cam[offsets[i]:offsets[i + 1]] = i % 2
    for c in range(2):
        if eps[c] == 0.0:
            continue
        k = np.asarray(calib[f"cam{c}"]["params"][4:10], np.float64)
        pw = 2 * np.arange(len(k)) + 3
        r = lambda t: t + (k[None] * t[:, None] ** pw[None]).sum(1)
        dr = lambda t: 1.0 + (k[None] * pw[None] * t[:, None] ** (pw[None] - 1)).sum(1)
        sel = cam == c
        f, pp = cams[c]["K"][0, 0], cams[c]["K"][:2, 2]
        d = xy[sel] - pp
        rho = np.maximum(np.linalg.norm(d, axis=1) / f, 1e-12)
        th = np.arctan(rho)
        target = r(th) / (1.0 + eps[c])
        t = th.copy()
        for _ in range(4):
            t -= (r(t) - target) / dr(t)
        xy[sel] = pp + d * (np.tan(t) / rho)[:, None]
    print(f"  keypoints corrected for the fisheye focal lengths x(1 + {eps})", flush=True)
    return xy


def to_fisheye(cams, n_seg, obs_cam, z, sigma, aria_calib):
    """The observations of the rectified pinhole images on each camera's factory fisheye model (LaMAria's
    aria_calibrations json: fx, fy, cx, cy, k0..k5, ...): a keypoint's ray from the pinhole K, projected with
    the radial model r(theta) = theta + k0 theta^3 + ... + k5 theta^13; its sigma scaled by the local
    magnification (the geometric mean of the radial and the tangential one). The cameras (per time segment,
    camera c in segment s at c * n_seg + s) take the fisheye's K and coefficients, so their focal lengths are
    the fisheye's (SLAM Dunk's camera model, without the tangential and thin-prism terms)."""
    calib = json.load(open(aria_calib))
    z, sigma = z.astype(np.float64).copy(), np.asarray(sigma, np.float64).copy()
    phys = obs_cam // n_seg
    out = []
    for c in range(2):
        fp = np.asarray(calib[f"cam{c}"]["params"], np.float64)
        kd = fp[4:10]
        kp = cams[c * n_seg]["K"]
        sel = phys == c
        ab = (z[sel] - kp[:2, 2]) / np.diag(kp)[:2]
        rho = np.maximum(np.linalg.norm(ab, axis=1), 1e-12)
        th = np.arctan(rho)
        pw = 2 * np.arange(6) + 3
        r = th + (kd[None] * th[:, None] ** pw[None]).sum(1)
        dr = 1.0 + (kd[None] * pw[None] * th[:, None] ** (pw[None] - 1)).sum(1)
        sv = np.where(rho > 1e-6, r / rho, 1.0)
        z[sel] = fp[None, 2:4] + fp[None, 0:2] * sv[:, None] * ab
        f_ratio = np.sqrt(fp[0] * fp[1] / (kp[0, 0] * kp[1, 1]))
        sigma[sel] *= f_ratio * np.sqrt(sv * dr / (1.0 + rho ** 2))
        kf_ = np.array([[fp[0], 0.0, fp[2]], [0.0, fp[1], fp[3]], [0.0, 0.0, 1.0]])
        out += [dict(cams[c * n_seg + s_], K=kf_, fisheye_k=kd.copy()) for s_ in range(n_seg)]
    print(f"  observations on the fisheye model: sigma median {np.median(sigma):.2f} px", flush=True)
    return out, z, sigma


def camera_segments(kf_ns, count, seconds=0.0, time_chunks=False):
    """Return the segment count and each keyframe's camera segment."""
    n = len(kf_ns)
    if not np.isfinite(seconds):
        raise SystemExit("--ext-seg-seconds must be finite")
    if seconds > 0:
        if time_chunks:
            chunk_ns = round(seconds * 1e9)
            if chunk_ns < 1:
                raise SystemExit("--ext-seg-seconds must span at least one nanosecond")
            seg = (kf_ns - kf_ns[0]) // chunk_ns
            return int(seg[-1]) + 1, seg
        count = int(np.ceil((kf_ns[-1] - kf_ns[0]) * 1e-9 / seconds))
    count = max(count, 1)
    return count, np.arange(n) * count // n


def saved_camera_segments(states, kf_ns):
    """Use the saved assignment, with the original equal-keyframe assignment for older states."""
    count = int(states["ext_segments"]) if "ext_segments" in states else 1
    if "seg_of_kf" in states:
        return count, states["seg_of_kf"]
    return camera_segments(kf_ns, count)


def cmd_solve(a):
    import gtsam
    from gtsam.symbol_shorthand import B, V, X
    from scipy.linalg import cho_solve_banded, cholesky_banded

    t0 = time.time()
    kf_ns, offsets, xy, _ = load_features(a.work, desc=False)
    kf_ns_all = kf_ns
    cams, noise = load_calib(a.calib)
    for key, opt in (("accel_noise_std", "accel_noise"), ("gyro_noise_std", "gyro_noise"),
                     ("accel_bias_std", "accel_bias_rw"), ("gyro_bias_std", "gyro_bias_rw")):
        if getattr(a, opt, None):  # the calibration's IMU noise overridden (e.g. SLAM Dunk's densities)
            noise[key] = float(getattr(a, opt))
    vio = load_tum(a.vio)
    kf_p, kf_r = keyframe_poses(kf_ns, a.vio)
    if a.focal_temp and a.focal_temp_coef:
        xy = temperature_corrected(xy, offsets, kf_ns, cams, a.focal_temp, a.focal_temp_coef)
    if a.focal_fisheye:
        xy = fisheye_corrected(xy, offsets, cams, a.fisheye_calib, [float(v) for v in a.focal_fisheye.split(",")])
    m = np.load(os.path.join(a.work, "matches.npz"))
    k0, k1 = (int(x) for x in a.kf_range.split(":")) if a.kf_range else (0, len(kf_ns))
    k1 = min(k1, len(kf_ns))
    sel = (m["pairs"][:, 0] >= k0) & (m["pairs"][:, 2] < k1) & (m["pairs"][:, 2] - m["pairs"][:, 0] <= a.max_offset)
    pairs, matches = m["pairs"][sel], m["matches"][sel]
    if a.loops:  # verified revisit matches from the loops stage
        lm = np.load(os.path.join(a.work, "loops.npz"))
        lsel = (lm["pairs"][:, 0] >= k0) & (lm["pairs"][:, 2] < k1)
        pairs, matches = np.concatenate([pairs, lm["pairs"][lsel]]), np.concatenate([matches, lm["matches"][lsel]])
        print(f"  + {lsel.sum()} revisit matches", flush=True)
    if a.flow:  # the flow's links between consecutive keyframes (cmd_flow)
        fl = np.load(a.flow)
        fsel = (fl["pairs"][:, 0] >= k0) & (fl["pairs"][:, 2] < k1)
        pairs, matches = np.concatenate([pairs, fl["pairs"][fsel]]), np.concatenate([matches, fl["matches"][fsel]])
        print(f"  + {fsel.sum()} flow links", flush=True)
    tracks, image_of, conflicts = build_tracks(offsets, pairs, matches, 1 << 30)
    pts, kept = triangulate(tracks, image_of, xy, kf_p, kf_r, cams, a.max_reproj, a.min_angle)
    kf_ns, kf_p, kf_r = kf_ns[k0:k1], kf_p[k0:k1], kf_r[k0:k1]
    # a landmark keeps the observations within --band keyframes of its first one; with --loops all of them
    obs_kf, obs_cam, obs_lm, obs_z, obs_f = [], [], [], [], []
    for j, feats in enumerate(kept):
        k = image_of[feats] // 2
        keep = (k - k.min() <= a.band) | bool(a.loops)
        if keep.sum() >= 2:
            obs_kf.append(k[keep] - k0)
            obs_cam.append(image_of[feats[keep]] % 2)
            obs_lm.append(np.full(keep.sum(), j))
            obs_z.append(xy[feats[keep]].astype(np.float64))
            obs_f.append(feats[keep])
    obs_lm = np.concatenate(obs_lm)
    obs_sigma = np.full(len(obs_lm), a.pixel_sigma)
    if a.sigma_size > 0:  # a keypoint larger than --sigma-size is located less precisely
        size = np.load(os.path.join(a.work, "features.npz"))["size"][np.concatenate(obs_f)]
        size = np.where(np.isfinite(size), size, a.sigma_size)
        obs_sigma *= np.maximum(size / a.sigma_size, 1.0) ** a.sigma_size_pow
        print(f"  pixel sigma by keypoint size: median {np.median(obs_sigma):.2f}, "
              f"90th percentile {np.percentile(obs_sigma, 90):.2f}", flush=True)
    used = np.unique(obs_lm)
    remap = np.full(len(kept), -1)
    remap[used] = np.arange(len(used))
    pts = pts[used].copy()
    print(f"tracks {len(tracks)} (dropped {conflicts} with conflicts), landmarks {len(pts)}, "
          f"{time.time() - t0:.0f} s", flush=True)
    chosen_cams = phys_est = [int(ch) for ch in a.est_extrinsic]
    reference = getattr(a, "stereo_reference", None)
    fix_translation = getattr(a, "fix_stereo_translation", False)
    if fix_translation and reference is None:
        raise SystemExit("--fix-stereo-translation needs --stereo-reference")
    # --fix-camera-rotations: both cameras keep the factory rotation, only the focal lengths vary per segment
    fix_rotations = getattr(a, "fix_camera_rotations", False)
    if fix_rotations and (reference is None or not a.est_focal):
        raise SystemExit("--fix-camera-rotations needs --stereo-reference and --est-focal")
    fixed_phys = set() if reference is None else {0, 1} if fix_rotations else {reference}
    if reference is not None:
        if 1 - reference not in phys_est:
            raise SystemExit("--stereo-reference requires the other camera in --est-extrinsic")
        if not all(np.isfinite(v) and v > 0 for v in (a.stereo_walk_sigma, a.stereo_baseline_sigma)):
            raise SystemExit("stereo prior sigmas must be finite and positive")
        if not a.est_focal:
            phys_est = [1 - reference]
    est_cams = phys_est
    n_seg, seg_of_kf = camera_segments(kf_ns, a.ext_segments, a.ext_seg_seconds, time_chunks=reference is not None)
    # --focal-seg-seconds: the focal lengths per time chunk, each camera's rotation still in --ext-segments equal
    # keyframe segments (the chunks of one segment tied by --rot-tie-sigma)
    rot_group = None
    focal_seconds = getattr(a, "focal_seg_seconds", 0.0) or 0.0
    if focal_seconds > 0:
        if reference is not None or a.ext_seg_seconds > 0 or not a.est_focal:
            raise SystemExit("--focal-seg-seconds needs --est-focal, without --stereo-reference or --ext-seg-seconds")
        n_seg, seg_of_kf = camera_segments(kf_ns, 1, focal_seconds, time_chunks=True)
        kf_group = np.arange(len(kf_ns)) * max(a.ext_segments, 1) // len(kf_ns)
        rot_group = kf_group[np.minimum(np.searchsorted(seg_of_kf, np.arange(n_seg)), len(kf_ns) - 1)]
    fixed = None  # the calibration a solve before ended with (--init-states), held as it is with --fix-calib
    if getattr(a, "init_states_file", None):
        st = np.load(a.init_states_file)
        a = argparse.Namespace(**vars(a))
        a.init_states = {"kf_ns": st["kf_ns"], "vel": st["vel"], "bias": st["bias"], "r_ic": st["r_ic"],
                         "foc": st["foc"], "n_seg": int(st["ext_segments"]),
                         "t_ic": st["t_ic"] if "t_ic" in st else None,
                         **({k: st["gm"][sl] for k, sl in (("gyro_m", slice(0, 9)), ("accel_m", slice(9, 18)),
                                                           ("gyro_temp", slice(18, 24)),
                                                           ("time_offsets", slice(24, 26)))
                             if len(st["gm"]) >= sl.stop} if "gm" in st else {})}
        if a.fix_calib:
            fixed = a.init_states
            if not np.array_equal(fixed["kf_ns"], kf_ns):
                raise SystemExit("--fix-calib needs the states of a solve over the same keyframes")
            # nothing of the calibration is estimated; the time segments are the states' own
            phys_est, est_cams = [], []
            n_seg, seg_of_kf = saved_camera_segments(st, kf_ns)
            a.est_focal = a.est_gyro_m = a.est_accel_m = a.est_gyro_bias_temp = a.est_gyro_scale_temp = False
    elif getattr(a, "fix_calib", False):
        raise SystemExit("--fix-calib needs --init-states")
    obs_kf, obs_cam = np.concatenate(obs_kf), np.concatenate(obs_cam)
    if n_seg > 1:
        # an IMU-camera rotation per time segment: camera c in segment s becomes camera c * n_seg + s
        obs_cam = obs_cam * n_seg + seg_of_kf[obs_kf]
        cams = [cams[c] for c in range(2) for _ in range(n_seg)]
        est_cams = [c * n_seg + s for c in phys_est for s in range(n_seg)]
        print(f"  {n_seg} time segments per camera rotation", flush=True)
    ref_cams = [] if reference is None else list(range(reference * n_seg, (reference + 1) * n_seg))
    fixed_rot_cams = [c * n_seg + s for c in sorted(fixed_phys) for s in range(n_seg)]
    trans_cams = [] if reference is None or fixed is not None or fix_translation else list(
        range((1 - reference) * n_seg, (2 - reference) * n_seg))
    obs_zc = np.concatenate(obs_z)
    if getattr(a, "fisheye_model", ""):  # observations and focal lengths on the factory fisheye model
        cams, obs_zc, obs_sigma = to_fisheye(cams, n_seg, obs_cam, obs_zc, obs_sigma, a.fisheye_model)
    rp = Reprojection(obs_kf, obs_cam, remap[obs_lm], obs_zc, len(pts), cams, est_cams,
                      obs_sigma, a.huber, a.band, est_focal=a.est_focal,
                      fixed_rotations=fixed_rot_cams, translation_cams=trans_cams)
    omega = [float(v) for v in a.omega_coriolis.split(",")] if a.omega_coriolis else None
    imu_data = np.loadtxt(a.imu, delimiter=",", comments="#")
    init_st = getattr(a, "init_states", None)
    # the gyroscope's response error M (rate = (I + M) measured - bias), row-major: given, carried over from the
    # solve before, or zero; --est-gyro-m estimates it (docs/lamaria_batch_ba.md#the-gyroscopes-response)
    # and the accelerometer's (specific force = (I + M_a) measured - bias), --est-accel-m, then the gyroscope's
    # bias change per degree and scale change per degree (--est-gyro-bias-temp, --est-gyro-scale-temp; SLAM Dunk's
    # thermal model, docs/lamaria_batch_ba.md#the-gyroscope-and-the-temperature): one vector of 24, of which the
    # estimated entries `act` join the border; then the IMU's clock against the cameras' and the accelerometer's
    # against the gyroscope's, in seconds (--est-time-offsets, SLAM Dunk's clock offsets)
    gm = np.zeros(26)
    imu_temp = None
    if a.imu_temp:  # the SLAM cameras' mean temperature at every IMU sample, from the recording's median
        tab = np.loadtxt(a.imu_temp, delimiter=",", skiprows=1)
        tc = tab[:, 1:3].mean(axis=1)
        imu_temp = np.interp(imu_data[:, 0], tab[:, 0], tc) - np.median(tc)
    elif a.est_gyro_bias_temp or a.est_gyro_scale_temp:
        raise SystemExit("--est-gyro-bias-temp and --est-gyro-scale-temp need --imu-temp")
    for name, sl in (("gyro_m", slice(0, 9)), ("accel_m", slice(9, 18)), ("gyro_temp", slice(18, 24)),
                     ("time_offsets", slice(24, 26))):
        if init_st is not None and init_st.get(name) is not None:
            gm[sl] = np.array(init_st[name], dtype=np.float64).ravel()
        if getattr(a, name, ""):
            gm[sl] = [float(v) for v in getattr(a, name).split(",")]
    act = np.r_[np.arange(9) if a.est_gyro_m else [], np.arange(9, 18) if a.est_accel_m else [],
                np.arange(18, 21) if a.est_gyro_bias_temp else [],
                np.arange(21, 24) if a.est_gyro_scale_temp else [],
                np.arange(24, 26) if getattr(a, "est_time_offsets", False) else []].astype(np.int64)
    msig = np.r_[np.full(9, a.gyro_m_sigma), np.full(9, a.accel_m_sigma), np.full(3, a.gyro_bias_temp_sigma),
                 np.full(3, a.gyro_scale_temp_sigma), np.full(2, getattr(a, "time_offset_sigma", 1e-3))][act]
    ng = len(act)

    def build_imu(m_vec):
        return _imu_graph(kf_ns, kf_p, kf_r, vio, imu_data, noise, a.imu_noise_scale, a.accel_rw_scale,
                          a.gyro_rw_scale, a.prior_rot_sigma, omega, init_st,
                          (a.gyro_rw_warmup, a.gyro_rw_warmup_s, kf_ns_all[0], getattr(a, "accel_rw_warmup", 1.0)),
                          gyro_m=m_vec[:9].reshape(3, 3) if np.any(m_vec[:9]) else None,
                          accel_m=m_vec[9:18].reshape(3, 3) if np.any(m_vec[9:18]) else None,
                          temp=imu_temp, gyro_temp=m_vec[18:24] if np.any(m_vec[18:24]) else None,
                          time_offsets=m_vec[24:26] if np.any(m_vec[24:26]) else None,
                          bias_chunk=(a.bias_chunk_seconds, a.bias_chunk_accel, a.bias_chunk_gyro)
                          if getattr(a, "bias_chunk_seconds", 0) else None)

    graph, values, index, imu_factors = build_imu(gm)
    gm_pass = gm.copy()  # the M the IMU factors were preintegrated with

    def gyro_jac(vals, m_vec, h=1e-4):
        """d(whitened error)/dM of every IMU factor at vals, by preintegrating again with each entry moved by h
        (at most its prior sigma)."""
        e0 = np.array([f.whitenedError(vals) for f in imu_factors])
        jm_ = np.zeros((len(imu_factors), e0.shape[1], ng))
        for jj, j in enumerate(act):
            mp_ = m_vec.copy()
            h = min(1e-4, msig[jj])
            mp_[j] += h
            fac = build_imu(mp_)[3]
            jm_[:, :, jj] = (np.array([f.whitenedError(vals) for f in fac]) - e0) / h
        return jm_

    jm = gyro_jac(values, gm) if ng else None

    def gyro_cost(vals, m_vec):  # the IMU factors' change by M's move since the preintegration (first order), and M's prior
        if not ng:
            return 0.0
        d = (m_vec - gm_pass)[act]
        ew = np.array([f.whitenedError(vals) for f in imu_factors])
        return 0.5 * (np.sum((ew + jm @ d) ** 2) - np.sum(ew ** 2)) + 0.5 * np.sum((m_vec[act] / msig) ** 2)

    n = len(kf_ns)
    nb, ne, npar = n * DOF, rp.n_par * rp.n_ext, rp.n_par
    modes = expand_modes(a.modes)
    nm = len(modes)
    if ng and nm:
        raise SystemExit("--est-gyro-m and --est-accel-m do not combine with --modes")
    nz = ne + nm + ng  # border: the camera rotations, the deformation modes (undamped), then the gyroscope's M
    order = np.argsort(rp.lm, kind="stable")
    lm_first = np.zeros(rp.n_lm, np.int64)
    lm_first[rp.lm[order][::-1]] = rp.kf[order][::-1]  # first keyframe of every landmark
    kf_s = kf_ns.astype(np.float64) * 1e-9
    u = (max(rp.band_kf, 1) + 1) * DOF - 1  # scalar bandwidth of the reduced system
    print(f"  reduced system: {nb} banded dims (bandwidth {u}) + {ne} border dims, {time.time() - t0:.0f} s",
          flush=True)
    r_ic = np.array([c["R_ic"] for c in cams])
    t_factory = np.array([c["t_ic"] for c in cams])
    t_ic = t_factory.copy()
    baseline = np.linalg.norm(t_factory[n_seg] - t_factory[0])
    if trans_cams and (not np.isfinite(baseline) or baseline <= 0):
        raise SystemExit("--stereo-reference needs a nonzero factory stereo baseline")
    init = getattr(a, "init_states", None)
    if init is not None:  # the estimated cameras start where the solve before ended (its last time segment)
        for c_ in phys_est:
            last = c_ * init["n_seg"] + init["n_seg"] - 1
            segment = slice(c_ * n_seg, (c_ + 1) * n_seg)
            if c_ not in fixed_phys:
                r_ic[segment] = init["r_ic"][last]
            if rp.est_trans[c_ * n_seg] and init.get("t_ic") is not None:
                t_ic[segment] = init["t_ic"][last]
            if a.est_focal:
                rp.foc[c_ * n_seg:(c_ + 1) * n_seg] = init["foc"][last]
        if fixed is not None:  # every camera and time segment as the solve before ended
            r_ic[:] = fixed["r_ic"]
            rp.foc[:] = fixed["foc"]
            if fixed["t_ic"] is not None:
                t_ic[:] = fixed["t_ic"]
        rp.set_focal(rp.foc)
    # consecutive time segments of a camera rotate alike, within --ext-seg-sigma
    seg_pairs = [(c * n_seg + s, c * n_seg + s + 1) for c in phys_est for s in range(n_seg - 1)]
    seg_w = 1.0 / np.radians(a.ext_seg_sigma) ** 2
    rot_pairs = [(ca, cb) for ca, cb in seg_pairs if ca // n_seg not in fixed_phys]
    if a.ext_seg_sigma_cams:
        sig = dict(zip(chosen_cams, (float(v) for v in a.ext_seg_sigma_cams.split(","))))
        seg_w = np.array([1.0 / np.radians(sig[ca // n_seg]) ** 2 for ca, _ in rot_pairs])
    if rot_group is not None and fixed is None:  # chunks of one rotation segment rotate together
        tie = 1.0 / np.radians(a.rot_tie_sigma) ** 2
        base = np.broadcast_to(np.asarray(seg_w, dtype=np.float64), (len(rot_pairs),))
        seg_w = np.where([rot_group[ca % n_seg] == rot_group[cb % n_seg] for ca, cb in rot_pairs], tie, base)
    # the rotation's three parameters within each camera's slot of the border
    rot_idx = (np.arange(rp.n_ext)[:, None] * npar + np.arange(3)[None, :]).ravel()
    foc_pair_w = 1.0 / a.focal_seg_sigma ** 2
    focal_seg_sigma_px = getattr(a, "focal_seg_sigma_px", None)
    if focal_seg_sigma_px is not None:
        if not np.isfinite(focal_seg_sigma_px) or focal_seg_sigma_px <= 0 or a.focal_seg_sigma_cams:
            raise SystemExit("--focal-seg-sigma-px needs a positive finite value and excludes --focal-seg-sigma-cams")
        # Each axis contributes its pixel prior to the shared log scale at the factory focal lengths.
        foc_pair_w = np.array([np.sum(np.diag(cams[ca]["K"])[:2] ** 2) / focal_seg_sigma_px ** 2
                               for ca, _ in seg_pairs])
    elif a.focal_seg_sigma_cams:
        sig = dict(zip(phys_est, (float(v) for v in a.focal_seg_sigma_cams.split(","))))
        foc_pair_w = np.array([1.0 / sig[ca // n_seg] ** 2 for ca, _ in seg_pairs])
    foc_w = (foc_pair_w, 1.0 / a.focal_sigma ** 2,
             1.0 / a.focal_mean_sigma ** 2 if a.focal_mean_sigma > 0 else 0.0,
             [[c * n_seg + s for s in range(n_seg)] for c in phys_est])

    # With a fixed reference, isotropic walks on the body-frame translations and on relative
    # stereo translations are equivalent. Put the prior on the length of the stereo offset.
    trans_idx = (rp.ext_slot[trans_cams][:, None] * npar + np.arange(3, 6)).ravel()
    trans_prior_args = (baseline, getattr(a, "stereo_walk_sigma", 0.0005),
                        getattr(a, "stereo_baseline_sigma", 0.0005))

    def seg_cost(ric, tic):
        c_ = segment_prior(ric, rot_pairs, rp.ext_slot, rp.n_ext, seg_w)[0]
        if a.est_focal:
            c_ += focal_prior(rp.foc, seg_pairs, rp.ext_slot, npar, rp.n_ext, *foc_w)[0]
        if trans_cams:
            c_ += stereo_translation_prior(tic[ref_cams[0]], tic[trans_cams], *trans_prior_args)
        return c_

    def poses(vals):
        return (np.array([vals.atPose3(X(k)).rotation().matrix() for k in range(n)]),
                np.array([vals.atPose3(X(k)).translation() for k in range(n)]))

    r_wi, t_wi = poses(values)
    cost = rp.cost(r_wi, t_wi, r_ic, t_ic, pts)[0] + graph.error(values) + seg_cost(r_ic, t_ic) + gyro_cost(values, gm)
    print(f"initial cost {cost:.6g}", flush=True)
    passes = 0
    lam = a.lambda0
    mu = 0.0  # damping of the modes relative to their own curvature: none until a step with them fails
    for it in range(a.iterations):
        ti = time.time()
        if ng and it > 0 and it % a.gyro_m_pass == 0 and np.abs(gm - gm_pass).max() > 1e-6:
            # preintegrate again with the current M (the first-order model is exact there)
            graph, _, index, imu_factors = build_imu(gm)
            gm_pass = gm.copy()
            jm = gyro_jac(values, gm)
            passes += 1
            cost = rp.cost(r_wi, t_wi, r_ic, t_ic, pts)[0] + graph.error(values) + seg_cost(r_ic, t_ic) + gyro_cost(values, gm)
            print(f"  IMU preintegrated again with M, cost {cost:.6g}", flush=True)
        if a.outlier_px > 0:
            # drop observations that moved far off, then landmarks left with a single observation;
            # from --outlier-start2 on with the tighter --outlier-px2
            thr = a.outlier_px2 if 0 < a.outlier_px2 and it >= a.outlier_start2 else a.outlier_px
            bad = rp.active & (rp.cost(r_wi, t_wi, r_ic, t_ic, pts)[1] * rp.sigma > thr)
            if bad.any():
                rp.active &= ~bad
                lone = rp.active & (np.bincount(rp.lm[rp.active], minlength=rp.n_lm)[rp.lm] < 2)
                rp.active &= ~lone
                cost = rp.cost(r_wi, t_wi, r_ic, t_ic, pts)[0] + graph.error(values) + seg_cost(r_ic, t_ic) \
                    + gyro_cost(values, gm)
                print(f"  outliers: {bad.sum()} observations over {thr} px (+{lone.sum()} left alone), "
                      f"{rp.active.sum()} active, cost {cost:.6g}", flush=True)
        h_xx, h_xl, g_x, h_ll_o, h_ll, g_l = rp.normal_equations(r_wi, t_wi, r_ic, t_ic, pts)
        if trans_cams:
            h_t, b_t = stereo_translation_prior(t_ic[ref_cams[0]], t_ic[trans_cams], *trans_prior_args,
                                                linearize=True)
        # IMU and prior factors, linearized by GTSAM: ||A dx - b||^2
        ab_imu, rhs_imu = np.zeros((u + 1, nb)), np.zeros(nb)
        lin = graph.linearize(values)
        imu_first = lin.size() - len(imu_factors)  # the IMU factors come last in the graph
        if ng:
            g_border, g_corner = np.zeros((nb, ng)), np.diag(1.0 / msig ** 2)
            g_rhs = -gm[act] / msig ** 2
            dgm = (gm - gm_pass)[act]
        for fi in range(lin.size()):
            jf = lin.at(fi)
            A, b = jf.jacobian()
            cols = np.concatenate([np.arange(index[key][0], index[key][0] + index[key][1]) for key in jf.keys()])
            if ng and fi >= imu_first:  # ||A dx + J (M - M_pass + dM) - b||^2
                jmk = jm[fi - imu_first]
                b = b - jmk @ dgm
                np.add.at(g_border, cols, A.T @ jmk)
                g_corner += jmk.T @ jmk
                g_rhs += jmk.T @ b
            ci, cj = np.meshgrid(cols, cols, indexing="ij")
            low = ci >= cj
            np.add.at(ab_imu, ((ci - cj)[low], cj[low]), (A.T @ A)[low])
            np.add.at(rhs_imu, cols, A.T @ b)
        slot = rp.ext_slot[rp.cam]
        has = slot >= 0
        if rp.numpy_pairs:
            direct = np.add.reduceat(h_xx[rp.obs_sort][:, 0:6, 0:6], rp.obs_starts, axis=0)
        modes_on = nm > 0 and it >= a.modes_start
        if modes_on:
            # H v for every mode before the landmarks are eliminated (undamped)
            vel = np.array([values.atVector(V(k)) for k in range(n)])
            v_x, v_l = mode_vectors(modes, kf_s, r_wi, t_wi, vel, pts, lm_first)
            vxf = v_x.reshape(nm, nb)
            hv_x, hv_l, hv_e = np.zeros((nm, nb)), np.zeros((nm, rp.n_lm, 3)), np.zeros((nm, ne))
            has_o = rp.ext_slot[rp.cam] >= 0
            for mi in range(nm):
                v6, vlo = v_x[mi][rp.kf, 0:6], v_l[mi][rp.lm]
                hv_x[mi] = _band_matvec(ab_imu, vxf[mi])
                np.add.at(hv_x[mi], (rp.kf * DOF)[:, None] + np.arange(6)[None, :],
                          np.einsum("nij,nj->ni", h_xx[:, 0:6, 0:6], v6) + np.einsum("nij,nj->ni", h_xl[:, 0:6], vlo))
                np.add.at(hv_l[mi], rp.lm, np.einsum("nji,nj->ni", h_xl[:, 0:6], v6) + np.einsum("nij,nj->ni", h_ll_o, vlo))
                if ne:
                    pe = (np.einsum("nji,nj->ni", h_xx[has_o][:, 0:6, 6:], v6[has_o])
                          + np.einsum("nij,nj->ni", h_xl[has_o, 6:], vlo[has_o]))
                    for sl in range(rp.n_ext):
                        hv_e[mi, sl * npar:sl * npar + npar] = pe[rp.ext_slot[rp.cam[has_o]] == sl].sum(0)
            vhv = vxf @ hv_x.T + np.einsum("mlk,nlk->mn", v_l, hv_l)
            b_x = rhs_imu.copy()
            np.add.at(b_x, (rp.kf * DOF)[:, None] + np.arange(6)[None, :], -g_x[:, 0:6])
            b_v = vxf @ b_x - np.einsum("mlk,lk->m", v_l, g_l)
        accepted = False
        # landmark damping: the diagonal, floored relative to the landmark's largest entry, so that a
        # landmark with a poorly observed direction (low parallax) cannot take a huge step
        d_l = np.diagonal(h_ll, axis1=1, axis2=2)
        d_l = np.maximum(d_l, a.lm_damp_floor * d_l.max(axis=1, keepdims=True))[:, :, None] * np.eye(3)[None]
        for _ in range(8):
            h_ll_inv = np.linalg.inv(h_ll + lam * d_l + 1e-9 * np.eye(3)[None])
            if _landmark_m_jit is not None:
                m_o = np.empty_like(h_xl)
                _landmark_m_jit(h_xl, h_ll_inv, rp.lm, m_o)
            else:
                m_o = np.einsum("nij,njk->nik", h_xl, h_ll_inv[rp.lm])
            ab, rhs = ab_imu.copy(), np.zeros(nb + nz)
            rhs[:nb] = rhs_imu
            border, corner = np.zeros((nb, nz)), np.zeros((nz, nz))
            if rp.numpy_pairs:
                _schur_add_numpy(rp, m_o, h_xl, h_xx, g_x, g_l, direct, slot, has, n, nb, ne, ab, border, corner, rhs)
            else:  # numba: one pass over the landmarks
                _schur_add_jit(m_o, h_xl, h_xx, g_x, g_l, rp.kf, slot, rp.lm, rp.lm_start, rp.band, nb, ab, border,
                               corner, rhs)
            if ne and rot_pairs:
                _, h_s, b_s = segment_prior(r_ic, rot_pairs, rp.ext_slot, rp.n_ext, seg_w)
                corner[np.ix_(rot_idx, rot_idx)] += h_s
                rhs[nb + rot_idx] += b_s
            if ne and a.est_focal:
                _, h_f, b_f = focal_prior(rp.foc, seg_pairs, rp.ext_slot, npar, rp.n_ext, *foc_w)
                corner[:ne, :ne] += h_f
                rhs[nb:nb + ne] += b_f
            if trans_cams:
                corner[np.ix_(trans_idx, trans_idx)] += h_t
                rhs[nb + trans_idx] += b_t
            ab[0] += lam * ab[0] + 1e-9
            if ne:
                corner[np.arange(ne), np.arange(ne)] += lam * np.diag(corner)[:ne] + 1e-9
            if ng:
                gi = np.arange(ne + nm, nz)
                border[:, gi] = g_border
                corner[np.ix_(gi, gi)] = g_corner + np.diag(lam * np.diag(g_corner) + 1e-9)
                rhs[nb + gi] = g_rhs
            if nm and not modes_on:
                corner[ne:, ne:] = np.eye(nm)  # inert until --modes-start
            if modes_on:  # the modes' column after eliminating the landmarks; no damping
                q = np.einsum("lij,mlj->mli", h_ll_inv, hv_l)
                for mi in range(nm):
                    border[:, ne + mi] = hv_x[mi]
                    np.add.at(border[:, ne + mi], (rp.kf * DOF)[:, None] + np.arange(6)[None, :],
                              -np.einsum("nij,nj->ni", h_xl[:, 0:6], q[mi][rp.lm]))
                if ne:
                    pe = np.einsum("nij,mnj->mni", h_xl[has, 6:], q[:, rp.lm[has]])
                    ce = hv_e.copy()
                    for sl in range(rp.n_ext):
                        ce[:, sl * npar:sl * npar + npar] -= pe[:, rp.ext_slot[rp.cam[has]] == sl].sum(1)
                    corner[:ne, ne:], corner[ne:, :ne] = ce.T, ce
                cmm = vhv - np.einsum("mlk,nlk->mn", hv_l, q)
                corner[ne:, ne:] = cmm + np.diag((mu + 1e-12) * np.abs(np.diag(cmm)) + 1e-12)
                rhs[nb + ne:] = b_v + np.einsum("mlk,lk->m", hv_l, np.einsum("lij,lj->li", h_ll_inv, g_l))
            try:
                cfac = cholesky_banded(ab, lower=True)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            if nz:  # block elimination of the border, with zb = L^-1 border (A = L L^T)
                zb = _forward_border(cfac, border)
                schur = corner - zb.T @ zb
                s_inv = np.linalg.inv(schur)
                if modes_on and len(rp.off_keys) == 0:
                    # trust region: damp the modes further until each step is within its cap
                    rz = rhs[nb:] - zb.T @ _tri_banded(cfac, rhs[:nb])
                    cap = np.array([a.mode_cap_yaw if m_.startswith("yaw") else a.mode_cap_scale for m_ in modes])
                    extra = 0.0
                    for _ in range(30):
                        if np.all(np.abs((s_inv @ rz)[ne:]) <= cap):
                            break
                        extra = 4 * extra if extra else 1e-3
                        damped = schur.copy()
                        damped[ne:, ne:] += np.diag(extra * np.abs(np.diag(cmm)))
                        s_inv = np.linalg.inv(damped)

            def precond(r_):
                if not nz:
                    return cho_solve_banded((cfac, True), r_[:nb])
                w0 = _tri_banded(cfac, r_[:nb])  # L^-1 r
                de0 = s_inv @ (r_[nb:] - zb.T @ w0)
                return np.r_[_tri_banded(cfac, w0 - zb @ de0, "T"), de0]

            sol = precond(rhs) if len(rp.off_keys) == 0 else _pcg(rp, m_o, h_xl, ab, u, nb, nz, border, corner,
                                                                    precond, rhs, a.pcg_tol, a.pcg_iters)
            dx, de, dk = sol[:nb], sol[nb:nb + ne], sol[nb + ne:nb + ne + nm]
            new_gm = gm.copy()
            new_gm[act] += sol[nb + ne + nm:]
            dxo = np.zeros((len(rp.kf), 6 + npar))
            dxo[:, 0:6] = dx.reshape(n, DOF)[rp.kf, 0:6]
            if ne:
                dxo[has, 6:] = de.reshape(-1, npar)[slot[has]]
            back = np.zeros((rp.n_lm, 3))
            np.add.at(back, rp.lm, np.einsum("nji,nj->ni", h_xl, dxo))
            if modes_on:
                back += np.einsum("mlk,m->lk", hv_l, dk)
            dl = -np.einsum("nij,nj->ni", h_ll_inv, g_l + back)
            vv = gtsam.VectorValues()
            for key, (col, dim) in index.items():
                vv.insert(key, dx[col:col + dim])
            new_values = values.retract(vv)
            new_ric, new_tic, foc = r_ic.copy(), t_ic.copy(), rp.foc.copy()
            new_foc = foc.copy()
            for c, inc in zip(est_cams, de.reshape(-1, npar)):
                if rp.est_rot[c]:
                    new_ric[c] = r_ic[c] @ so3_exp(inc[:3])
                if rp.est_trans[c]:
                    new_tic[c] += inc[3:6]
                if a.est_focal:
                    new_foc[c] += inc[-1]
            rp.set_focal(new_foc)
            new_pts = pts + dl
            nr, nt = poses(new_values)
            if modes_on:  # the modes' share of the step, as exact deformations
                st = (nr, nt, np.array([new_values.atVector(V(k)) for k in range(n)]),
                      np.array([np.r_[new_values.atConstantBias(B(k)).accelerometer(),
                                      new_values.atConstantBias(B(k)).gyroscope()] for k in range(n)]), new_pts)
                for mo, be in zip(modes, dk):
                    st = deform(mo, be, kf_s, *st, lm_first)
                new_values = gtsam.Values()
                for k in range(n):
                    new_values.insert(X(k), gtsam.Pose3(gtsam.Rot3(st[0][k]), st[1][k]))
                    new_values.insert(V(k), st[2][k])
                    new_values.insert(B(k), gtsam.imuBias.ConstantBias(st[3][k, 0:3], st[3][k, 3:6]))
                nr, nt, new_pts = st[0], st[1], st[4]
            new_cost = rp.cost(nr, nt, new_ric, new_tic, new_pts)[0] + graph.error(new_values) + seg_cost(new_ric, new_tic) \
                + gyro_cost(new_values, new_gm)
            if new_cost >= cost:
                rp.set_focal(foc)  # rejected: the focal lengths stay
            if new_cost < cost:
                rel = (cost - new_cost) / cost
                values, r_ic, pts, r_wi, t_wi, cost, gm = new_values, new_ric, new_pts, nr, nt, new_cost, new_gm
                t_ic = new_tic
                lam = max(lam / 3, 1e-10)
                mu = mu / 10 if mu > 1e-4 else 0.0
                accepted = True
                break
            lam *= 5
            if modes_on:  # a rejected step also shrinks the modes' share
                mu = max(mu * 10, 1e-3)
        msg = f"iter {it:2d}: cost {cost:.6g} lambda {lam:.1e} {time.time() - ti:.0f} s"
        if modes_on and accepted:
            msg += " | " + " ".join(f"{mo} {d_:+.2e}" for mo, d_ in zip(modes, dk))
        for c in phys_est:  # change from the calibration; with segments, the first and the last one
            ch = [np.round(so3_log_deg(cams[c * n_seg]["R_ic"].T @ r_ic[c * n_seg + s]), 3) for s in (0, n_seg - 1)]
            msg += f" | cam{c} rotation {ch[0]}" + (f" .. {ch[1]}" if n_seg > 1 else "")
            if a.est_focal:
                msg += f" focal {100 * np.expm1(rp.foc[c * n_seg]):+.3f} .. {100 * np.expm1(rp.foc[c * n_seg + n_seg - 1]):+.3f} %"
        if ng:
            msg += f" | gyro M diag {np.round(100 * gm[[0, 4, 8]], 3)} % accel {np.round(100 * gm[[9, 13, 17]], 3)} %"
            if np.any(gm[18:24]):
                msg += f" | per degree: bias {np.round(gm[18:21] * 1e6, 2)} urad/s, scale {np.round(gm[21:24] * 1e6, 1)} ppm"
            if np.any(gm[24:26]):
                msg += f" | clock: IMU {gm[24] * 1e3:+.3f} ms, accel {gm[25] * 1e3:+.3f} ms"
        print(msg, flush=True)
        if not accepted or rel < a.rel_tol:
            if ng and accepted and np.abs(gm - gm_pass).max() > 1e-5 and passes < 5:
                continue  # converged with the first-order M: preintegrate again at the next iteration
            break
    reproj, err = rp.cost(r_wi, t_wi, r_ic, t_ic, pts)
    err_px = (err * rp.sigma)[rp.active]
    comp = np.zeros(5)
    for fct in imu_factors:  # whitened IMU residual per component: rotation, position, velocity, biases
        ew = fct.whitenedError(values)
        comp += [np.sum(ew[i:i + 3] ** 2) for i in (0, 3, 6, 9, 12)]
    comp /= max(len(imu_factors), 1)
    summary = {
        "keyframes": n, "landmarks": int(rp.n_lm), "observations": len(rp.kf),
        "active_observations": int(rp.active.sum()),
        "final_cost": cost, "reprojection_cost": reproj, "imu_prior_cost": graph.error(values),
        "segment_prior_cost": seg_cost(r_ic, t_ic),
        "median_reprojection_px": float(np.median(err_px)),
        "reprojection_over_4px_percent": float(np.mean(err_px > 4) * 100),
        "imu_residual_per_factor": dict(zip(("rotation", "position", "velocity", "accel_bias", "gyro_bias"),
                                            comp.round(3).tolist())),
        "focal_change_percent": {f"cam{c}": (100 * np.expm1(rp.foc[c * n_seg:c * n_seg + n_seg])).round(4).tolist()
                                 for c in phys_est},
        "camera_rotation_change_deg": {  # per time segment with --ext-segments
            f"cam{c}": [so3_log_deg(cams[c * n_seg]["R_ic"].T @ r_ic[c * n_seg + s]).tolist() for s in range(n_seg)]
            if n_seg > 1 else so3_log_deg(cams[c]["R_ic"].T @ r_ic[c]).tolist() for c in phys_est},
        "keyframe_displacement_m": {"median": float(np.median(np.linalg.norm(t_wi - kf_p, axis=1))),
                                    "max": float(np.linalg.norm(t_wi - kf_p, axis=1).max())},
        "biases": {str(k): {"accel": values.atConstantBias(B(k)).accelerometer().tolist(),
                            "gyro": values.atConstantBias(B(k)).gyroscope().tolist()} for k in (0, n // 2, n - 1)},
        "gyro_m": gm[:9].reshape(3, 3).tolist(),
        "accel_m": gm[9:18].reshape(3, 3).tolist(),
        "gyro_bias_per_degree": gm[18:21].tolist(),
        "gyro_scale_per_degree": gm[21:24].tolist(),
        "time_offsets_ms": {"imu_camera": gm[24] * 1e3, "accel_gyro": gm[25] * 1e3},
        "seconds": time.time() - t0,
    }
    if rot_group is not None:  # how far the tied chunks of a rotation segment ended apart
        summary["rotation_segment_of_chunk"] = rot_group.tolist()
        summary["rotation_tie_spread_deg"] = max(
            (float(np.degrees(np.linalg.norm(so3_log(r_ic[c * n_seg + i].T @ r_ic[c * n_seg + j]))))
             for c in phys_est for i in range(n_seg) for j in [int(np.flatnonzero(rot_group == rot_group[i])[0])]),
            default=0.0)
    if reference is not None or (fixed is not None and not np.array_equal(t_ic, t_factory)):
        summary["stereo_segment_keyframes"] = np.bincount(seg_of_kf, minlength=n_seg).tolist()
        offsets = t_ic[n_seg:] - t_ic[:n_seg]
        summary["stereo_baseline_mm"] = (1000 * np.linalg.norm(offsets, axis=1)).tolist()
        summary["stereo_offset_change_imu_mm"] = (1000 * (offsets - (t_factory[n_seg:] - t_factory[:n_seg]))).tolist()
    print(json.dumps(summary, indent=1), flush=True)
    with open(os.path.splitext(a.out)[0] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    write_corrected(a.out, vio, kf_ns, kf_p, kf_r, t_wi, r_wi)
    bias = np.array([np.r_[values.atConstantBias(B(k)).accelerometer(), values.atConstantBias(B(k)).gyroscope()]
                     for k in range(n)])
    vel = np.array([values.atVector(V(k)) for k in range(n)])
    if a.save_states:  # what loop-correct needs: keyframe poses, landmarks and their observations
        # Camera extrinsics are ordered by camera, then by time segment.
        np.savez(os.path.splitext(a.out)[0] + "_states.npz", k0=k0, kf_ns=kf_ns, r_wi=r_wi, t_wi=t_wi, r_ic=r_ic, t_ic=t_ic,
                 ext_segments=n_seg, seg_of_kf=seg_of_kf, pts=pts, obs_kf=rp.kf, obs_cam=rp.cam // n_seg,
                 obs_lm=rp.lm, obs_z=rp.z,
                 active=rp.active, vel=vel, bias=bias, foc=rp.foc, sigma=rp.sigma, huber=a.huber, gm=gm)
    return {"kf_ns": kf_ns, "vel": vel, "bias": bias, "r_ic": r_ic, "t_ic": t_ic, "foc": rp.foc.copy(), "n_seg": n_seg,
            "gyro_m": gm[:9].copy(), "accel_m": gm[9:18].copy(), "gyro_temp": gm[18:24].copy(),
            "time_offsets": gm[24:26].copy()}


def cmd_online(a):
    """Causal mode: every --step keyframes, solve the last --window keyframes (solve --kf-range) from the
    trajectory so far, and carry the correction to the frames after them.

    A window uses only the data up to its last keyframe (a match or a revisit between keyframes i < j needs the
    VIO poses of i and j only), so the result is what a system running alongside the VIO would have had. The
    window's first keyframe stays at its earlier estimate (the solve's gauge prior), which ties the windows,
    and its biases, velocities, camera rotations and focal lengths start from the window before: the biases
    take a solve many iterations to settle from zero (on sequence_1_19 a 480-keyframe window needs about 30).
    """
    import contextlib
    import copy
    import shutil
    n = len(np.load(os.path.join(a.work, "features.npz"))["kf_ns"])
    shutil.copy(a.vio, a.out)
    base = os.path.splitext(a.out)[0]
    t0 = time.time()
    states = None  # each window starts from the biases and velocities of the one before
    with open(base + "_windows.log", "w") as log:
        for k1 in list(range(min(a.step, n), n, a.step)) + [n]:
            w = copy.copy(a)
            w.init_states = states
            w.kf_range, w.vio, w.save_states = f"{max(0, k1 - a.window)}:{k1}", a.out, False
            w.out = base + "_window.tum"
            ti = time.time()
            with contextlib.redirect_stdout(log):
                states = cmd_solve(w)
            os.replace(w.out, a.out)
            print(f"window {w.kf_range}: {time.time() - ti:.1f} s, {time.time() - t0:.0f} s in all", flush=True)
    os.remove(base + "_window_summary.json")


def _tri_banded(cfac, b, trans="N"):
    """L^-1 b (trans "N") or L^-T b ("T") for the lower banded Cholesky factor cfac (LAPACK storage)."""
    from scipy.linalg.lapack import dtbtrs
    x, info = dtbtrs(cfac, b, uplo="L", trans=trans)
    if info != 0:
        raise np.linalg.LinAlgError(f"dtbtrs info {info}")
    return x


def _band_forward(cfac, z, starts):
    """z <- L^-1 z in place for the lower banded factor cfac (LAPACK storage). The columns of z are sorted by
    starts, the first row that may be nonzero (rows above stay zero).

    Column-oriented over L and over every right-hand side started so far, so that each band entry is
    read once per row (plain loops, for numba).
    """
    kd, n, m = cfac.shape[0] - 1, cfac.shape[1], z.shape[1]
    na = 0
    for j in range(starts[0] if m else n, n):
        while na < m and starts[na] <= j:
            na += 1
        d0 = cfac[0, j]
        for c in range(na):
            z[j, c] /= d0
        for d in range(1, min(kd, n - 1 - j) + 1):
            lv = cfac[d, j]
            if lv != 0.0:
                for c in range(na):
                    z[j + d, c] -= lv * z[j, c]


_band_forward_jit = njit(nogil=True)(_band_forward) if njit is not None else None


def _forward_border(cfac, border):
    """L^-1 border, each column solved from the keyframe of its first nonzero row on (the rows above stay 0)."""
    nonzero = border != 0
    first = np.where(nonzero.any(0), nonzero.argmax(0) // DOF * DOF, border.shape[0])
    if _band_forward_jit is not None:  # every column in one pass, in the order of their first rows
        order = np.argsort(first, kind="stable")
        z = np.ascontiguousarray(border[:, order])
        _band_forward_jit(cfac, z, first[order].astype(np.int64))
        out = np.empty_like(z)
        out[:, order] = z
        return out
    z = np.zeros_like(border)
    for r0 in np.unique(first[first < border.shape[0]]):
        cols = np.flatnonzero(first == r0)
        z[r0:, cols] = _tri_banded(cfac[:, r0:], border[r0:, cols])
    return z


def _pcg(rp, m_o, h_xl, ab, u, nb, ne, border, corner, precond, rhs, tol=1e-4, max_iter=200):
    """Conjugate gradients on the full reduced system; the bordered banded factor preconditions it."""
    blocks = np.zeros((len(rp.off_keys), 6, 6))
    for s in range(0, len(rp.off_a), rp.chunk):
        e = min(s + rp.chunk, len(rp.off_a))
        prod = -np.einsum("nij,nkj->nik", m_o[rp.off_a[s:e], 0:6, :], h_xl[rp.off_b[s:e], 0:6, :])
        bid = rp.off_bid[s:e]
        st = np.flatnonzero(np.r_[True, bid[1:] != bid[:-1]])
        blocks[bid[st]] += np.add.reduceat(prod, st, axis=0)
    from scipy.linalg.blas import dsbmv
    rows_a = ((rp.off_ka * DOF)[:, None] + np.arange(6)).ravel()
    rows_b = ((rp.off_kb * DOF)[:, None] + np.arange(6)).ravel()
    ab_f = np.asfortranarray(ab)

    def operator(x):
        xb = x[:nb]
        y = dsbmv(u, 1.0, ab_f, xb, lower=1)  # the banded part (symmetric, lower storage)
        y += np.bincount(rows_a, np.einsum("nij,nj->ni", blocks, xb[rows_b].reshape(-1, 6)).ravel(), nb)
        y += np.bincount(rows_b, np.einsum("nji,nj->ni", blocks, xb[rows_a].reshape(-1, 6)).ravel(), nb)
        if not ne:
            return y
        return np.r_[y + border @ x[nb:], border.T @ xb + corner @ x[nb:]]

    x = np.zeros(nb + ne)
    r = rhs.copy()
    z = precond(r)
    p = z.copy()
    rz = r @ z
    bnorm = np.linalg.norm(rhs)
    for it in range(max_iter):
        ap = operator(p)
        alpha = rz / (p @ ap)
        x += alpha * p
        r -= alpha * ap
        if np.linalg.norm(r) < tol * bnorm:
            break
        z = precond(r)
        rz, rz_old = r @ z, rz
        p = z + (rz / rz_old) * p
    print(f"    pcg: {it + 1} iterations, relative residual {np.linalg.norm(r) / bnorm:.1e}", flush=True)
    return x


def _schur_add_numpy(rp, m_o, h_xl, h_xx, g_x, g_l, direct, slot, has, n, nb, ne, ab, border, corner, rhs):
    """The numpy version of _schur_add (without numba): the same sums over the precomputed pair lists."""
    nz = border.shape[1]
    # Schur complement of the landmarks: -M_a X_b^T per (kf_a, kf_b) block, in chunks
    blocks = np.zeros((len(rp.blk_keys), 6, 6))
    for s in range(0, len(rp.low_a), rp.chunk):
        e = min(s + rp.chunk, len(rp.low_a))
        prod = -np.einsum("nij,nkj->nik", m_o[rp.low_a[s:e], 0:6, :], h_xl[rp.low_b[s:e], 0:6, :])
        bid = rp.low_bid[s:e]
        st = np.flatnonzero(np.r_[True, bid[1:] != bid[:-1]])
        blocks[bid[st]] += np.add.reduceat(prod, st, axis=0)
    for rows, cols, vals in ((rp.blk_a * DOF, rp.blk_b * DOF, blocks), (rp.obs_kfs * DOF, rp.obs_kfs * DOF, direct)):
        ii, jj = np.broadcast_arrays(rows[:, None, None] + np.arange(6)[None, :, None],
                                     cols[:, None, None] + np.arange(6)[None, None, :])
        low = ii >= jj
        ab[(ii - jj)[low], jj[low]] += vals[low]
    if ne:
        P = rp.n_par
        bsum = np.zeros((n * rp.n_ext, 6, P))
        for s in range(0, len(rp.bd_a), rp.chunk):
            e = min(s + rp.chunk, len(rp.bd_a))
            prod = -np.einsum("nij,nkj->nik", m_o[rp.bd_a[s:e], 0:6, :], h_xl[rp.bd_b[s:e], 6:, :])
            key = rp.bd_key[s:e]
            st = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
            bsum[key[st]] += np.add.reduceat(prod, st, axis=0)
        np.add.at(bsum, rp.kf[has].astype(np.int64) * rp.n_ext + slot[has], h_xx[has][:, 0:6, 6:])
        for sl in range(rp.n_ext):
            border.reshape(n, DOF, nz)[:, 0:6, sl * P:sl * P + P] += bsum[sl::rp.n_ext]
        csum = np.zeros((rp.n_ext, rp.n_ext, P, P))
        for s in range(0, len(rp.cn_a), rp.chunk):
            e = min(s + rp.chunk, len(rp.cn_a))
            prod = -np.einsum("nij,nkj->nik", m_o[rp.cn_a[s:e], 6:, :], h_xl[rp.cn_b[s:e], 6:, :])
            np.add.at(csum, (slot[rp.cn_a[s:e]], slot[rp.cn_b[s:e]]), prod)
        np.add.at(csum, (slot[has], slot[has]), h_xx[has][:, 6:, 6:])
        for sa in range(rp.n_ext):
            for sb in range(rp.n_ext):
                corner[sa * P:sa * P + P, sb * P:sb * P + P] = csum[sa, sb]
    red = -g_x + np.einsum("nij,nj->ni", m_o, g_l[rp.lm])
    np.add.at(rhs, (rp.kf * DOF)[:, None] + np.arange(6)[None, :], red[:, 0:6])
    if ne:
        np.add.at(rhs, nb + (slot[has] * rp.n_par)[:, None] + np.arange(rp.n_par)[None, :], red[has, 6:])


def write_corrected(path, vio, kf_ns, kf_p, kf_r, opt_p, opt_r):
    """Each VIO frame moves with the world-frame correction of the surrounding keyframes, interpolated."""
    from scipy.spatial.transform import Rotation, Slerp
    vio_ns, vio_p, vio_r = vio
    corr_r = np.einsum("nij,nkj->nik", opt_r, kf_r)
    corr_t = opt_p - np.einsum("nij,nj->ni", corr_r, kf_p)
    tq = kf_ns.astype(np.float64)
    t = np.clip(vio_ns.astype(np.float64), tq[0], tq[-1])
    rc = Slerp(tq, Rotation.from_matrix(corr_r))(t).as_matrix()
    tc = np.stack([np.interp(t, tq, corr_t[:, i]) for i in range(3)], axis=1)
    out_r = np.einsum("nij,njk->nik", rc, vio_r)
    out_p = np.einsum("nij,nj->ni", rc, vio_p) + tc
    q = Rotation.from_matrix(out_r).as_quat()
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for i in range(len(vio_ns)):
            f.write(f"{vio_ns[i] * 1e-9:.9f} {out_p[i, 0]:.6f} {out_p[i, 1]:.6f} {out_p[i, 2]:.6f} "
                    f"{q[i, 0]:.9f} {q[i, 1]:.9f} {q[i, 2]:.9f} {q[i, 3]:.9f}\n")
    print(f"wrote {path}", flush=True)


# ----------------------------------------------------------------------------- loop correction

def pnp_body_pose(p3, p2, kmat, r_ic, t_ic, r_wb0, t_wb0, min_inliers):
    """World pose (R_wb, t_wb) of the body whose camera (R_ic, t_ic in the body) sees the world points p3 at
    the pixels p2, by PnP with RANSAC (3 px) from the guess (R_wb0, t_wb0); None below min_inliers."""
    import cv2
    r_wc = r_wb0 @ r_ic
    c_w = r_wb0 @ t_ic + t_wb0
    rvec, tvec = cv2.Rodrigues(r_wc.T)[0], (-r_wc.T @ c_w).reshape(3, 1)
    found, rvec, tvec, inl = cv2.solvePnPRansac(p3, p2, kmat, None, rvec, tvec, useExtrinsicGuess=True,
                                                iterationsCount=200, reprojectionError=3.0, confidence=0.999)
    if not found or inl is None or len(inl) < min_inliers:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(p3[inl[:, 0]], p2[inl[:, 0]], kmat, None, rvec, tvec)
    r_wc = cv2.Rodrigues(rvec)[0].T
    r_wb = r_wc @ r_ic.T
    return r_wb, -r_wc @ tvec.ravel() - r_wb @ t_ic


def saved_camera_translations(states, cams, n_seg):
    """Read per-segment translations, or use the factory calibration for older states."""
    if "t_ic" in states:
        return states["t_ic"]
    return np.repeat([c["t_ic"] for c in cams], n_seg, axis=0)


def cmd_loop_correct(a):
    """Close the revisits of a finished solve with a pose graph.

    A solve with the revisit matches cannot close a loop the VIO drifted on: the triangulation and the
    outlier rejection drop exactly the revisit observations that disagree by a metre or more. Here
    the revisits become relative poses instead. For each revisit image pair (a, b), a's features that
    are observations of a solved landmark give 3D points of a's visit; PnP with RANSAC places b's
    camera against them (and a's against b's visit). Each such edge between keyframes a and b joins
    a pose graph whose odometry edges are the solve's own consecutive relative poses; an edge whose
    disagreement with the solve is implausible for the time between the visits (--gate) is dropped,
    and the rest are robust (Cauchy). The solved graph's keyframe corrections are spread over every
    frame of the solve's output trajectory.
    """
    import gtsam
    from gtsam.symbol_shorthand import X
    from scipy.spatial.transform import Rotation

    st = np.load(a.states)
    cams, _ = load_calib(a.calib)
    k0, kf_ns, r_wi, t_wi, r_ic, pts = (st[k] for k in ("k0", "kf_ns", "r_wi", "t_wi", "r_ic", "pts"))
    k0, n = int(k0), len(kf_ns)
    n_seg, seg_of_kf = saved_camera_segments(st, kf_ns)
    t_ic = saved_camera_translations(st, cams, n_seg)
    foc = st["foc"] if "foc" in st else np.zeros(2 * n_seg)  # focal length scales of solve --est-focal
    act = st["active"]
    okf, ocam, olm, oz = st["obs_kf"][act], st["obs_cam"][act], st["obs_lm"][act], st["obs_z"][act]
    _, offsets, xy, _ = load_features(a.work, desc=False)
    lp = np.load(os.path.join(a.work, "loops.npz"))
    sel = (lp["pairs"][:, 0] >= k0) & (lp["pairs"][:, 2] < k0 + n)
    pairs, matches = lp["pairs"][sel].astype(np.int64), lp["matches"][sel]
    pairs[:, [0, 2]] -= k0
    # landmark of a feature, found by its pixel among the solve's active observations
    lookup = {(int(2 * k + c), round(float(z[0]), 3), round(float(z[1]), 3)): int(j)
              for k, c, z, j in zip(okf, ocam, oz, olm)}

    def edge(k3, c3, f3, k2, c2, f2):
        """Body pose of keyframe k2 placed by PnP against the landmarks seen by features f3 of (k3, c3)."""
        lms = np.array([lookup.get((2 * k3 + c3, round(float(xy[i, 0]), 3), round(float(xy[i, 1]), 3)), -1)
                        for i in f3])
        ok = lms >= 0
        if ok.sum() < a.min_inliers:
            return None
        kmat = cams[c2]["K"].copy()
        camera = c2 * n_seg + seg_of_kf[k2]
        kmat[[0, 1], [0, 1]] *= np.exp(foc[camera])
        placed = pnp_body_pose(pts[lms[ok]], xy[f2[ok]].astype(np.float64), kmat,
                               r_ic[camera], t_ic[camera],
                               r_wi[k2], t_wi[k2], a.min_inliers)
        if placed is None:
            return None
        r_wb, t_wb = placed
        rel_r, rel_t = r_wi[k3].T @ r_wb, r_wi[k3].T @ (t_wb - t_wi[k3])
        dev_t = np.linalg.norm(rel_t - r_wi[k3].T @ (t_wi[k2] - t_wi[k3]))
        dev_r = np.degrees(np.linalg.norm(Rotation.from_matrix((r_wi[k3].T @ r_wi[k2]).T @ rel_r).as_rotvec()))
        return k3, k2, rel_r, rel_t, dev_r, dev_t

    key = pairs[:, 0] * 10_000_000 + pairs[:, 1] * 1_000_000 + pairs[:, 2] * 10 + pairs[:, 3]
    _, inv = np.unique(key, return_inverse=True)
    t_kf = (kf_ns - kf_ns[0]) * 1e-9
    base, rate, max_rot = (float(v) for v in a.gate.split(","))
    edges, gated = [], 0
    for j in range(inv.max() + 1 if len(inv) else 0):
        rows = np.flatnonzero(inv == j)
        ka, ca, kb, cb = pairs[rows[0]]
        fa = offsets[2 * (ka + k0) + ca] + matches[rows, 0]
        fb = offsets[2 * (kb + k0) + cb] + matches[rows, 1]
        for e in (edge(ka, ca, fa, kb, cb, fb), edge(kb, cb, fb, ka, ca, fa)):
            if e is None:
                continue
            if e[5] > base + rate * abs(t_kf[e[1]] - t_kf[e[0]]) or e[4] > max_rot:
                gated += 1
                continue
            edges.append(e)
    print(f"{inv.max() + 1 if len(inv) else 0} revisit image pairs: {len(edges)} loop edges, {gated} more over the "
          f"drift gate", flush=True)
    if not edges:  # nothing to close: the solve's trajectory as it is
        import shutil
        shutil.copyfile(a.tum, a.out)
        print("no loop edges: trajectory unchanged", flush=True)
        return
    dev = np.array([e[5] for e in edges])
    print(f"  disagreement with the solve: median {np.median(dev):.2f} m, p90 {np.percentile(dev, 90):.2f} m",
          flush=True)
    graph, vals = gtsam.NonlinearFactorGraph(), gtsam.Values()
    for k in range(n):
        vals.insert(X(k), gtsam.Pose3(gtsam.Rot3(r_wi[k]), t_wi[k]))
    graph.add(gtsam.PriorFactorPose3(X(0), vals.atPose3(X(0)), gtsam.noiseModel.Isotropic.Sigma(6, 1e-6)))
    odo = gtsam.noiseModel.Diagonal.Sigmas(np.r_[np.full(3, a.odo_rot_sigma), np.full(3, a.odo_pos_sigma)])
    for k in range(n - 1):
        graph.add(gtsam.BetweenFactorPose3(X(k), X(k + 1), vals.atPose3(X(k)).between(vals.atPose3(X(k + 1))), odo))
    loop_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Cauchy.Create(1.0),
        gtsam.noiseModel.Diagonal.Sigmas(np.r_[np.full(3, a.loop_rot_sigma), np.full(3, a.loop_pos_sigma)]))
    for k3, k2, rel_r, rel_t, _, _ in edges:
        graph.add(gtsam.BetweenFactorPose3(X(k3), X(k2), gtsam.Pose3(gtsam.Rot3(rel_r), rel_t), loop_noise))
    res = gtsam.LevenbergMarquardtOptimizer(graph, vals, gtsam.LevenbergMarquardtParams()).optimize()
    new_r = np.array([res.atPose3(X(k)).rotation().matrix() for k in range(n)])
    new_t = np.array([res.atPose3(X(k)).translation() for k in range(n)])
    moved = np.linalg.norm(new_t - t_wi, axis=1)
    print(f"pose graph: error {graph.error(vals):.4g} -> {graph.error(res):.4g}; keyframes moved by "
          f"{np.median(moved):.2f} m at the median, {moved.max():.2f} m at most", flush=True)
    # the per-keyframe corrections, interpolated over the solve's per-frame output
    write_corrected(a.out, load_tum(a.tum), kf_ns, t_wi, r_wi, new_t, new_r)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features", help="SIFT on the keyframes")
    f.add_argument("--mav0", required=True, help="ASL mav0 directory (cam0/data, cam1/data)")
    f.add_argument("--vio", required=True, help="VIO trajectory (TUM, seconds); keyframes need a pose")
    f.add_argument("--work", required=True)
    f.add_argument("--kf-step", type=int, default=10, help="every Nth stereo frame is a keyframe")
    f.add_argument("--num-features", type=int, default=1500)
    f.add_argument("--jobs", type=int, default=2)
    f.add_argument("--clahe", action="store_true", help="contrast-limited histogram equalization (low light)")
    f.add_argument("--denoise", type=float, default=0.0,
                   help="non-local means denoising strength h before detection (low light, e.g. 7; 0: off)")
    f.add_argument("--denoise-window", type=int, default=21,
                   help="its search window (px); 11 takes about 40 %% less time than 21")
    f.add_argument("--contrast-threshold", type=float, default=0.01, help="SIFT: OpenCV's contrastThreshold")
    f.add_argument("--edge-threshold", type=float, default=10.0, help="SIFT: OpenCV's edgeThreshold")
    f.add_argument("--rootsift", action="store_true",
                   help="SIFT: RootSIFT descriptors, compared by their Euclidean distance like SIFT's")
    f.add_argument("--extractor", choices=("sift", *LEARNED), default="sift",
                   help="SIFT (OpenCV), or learned keypoints on the GPU (LightGlue's extractors; --num-features "
                        "keypoints at most); match and loops then use LightGlue")
    f.add_argument("--detection-threshold", type=float, default=None,
                   help="learned extractors: keypoint score threshold (default: the extractor's)")
    f.add_argument("--resize", type=int, default=0,
                   help="learned extractors: resize the longer image side to this before detection (0: as recorded)")
    f.add_argument("--device", default="", help="torch device of the learned extractors (default: cuda if there)")

    def lightglue_args(s):  # match and loops with learned features
        s.add_argument("--matcher", choices=("auto", "ratio", "lightglue"), default="auto",
                       help="auto: LightGlue for learned features, the ratio test for SIFT")
        s.add_argument("--device", default="", help="torch device of LightGlue (default: cuda if there)")
        s.add_argument("--lg-threshold", type=float, default=0.1, help="LightGlue's match score threshold")
        s.add_argument("--lg-batch", type=int, default=16, help="image pairs per LightGlue call")
        s.add_argument("--lg-mp", action=argparse.BooleanOptionalAction, default=True,
                       help="LightGlue in mixed precision on the GPU")
    m = sub.add_parser("match", help="VIO-guided epipolar matching between keyframes")
    m.add_argument("--work", required=True)
    m.add_argument("--vio", required=True)
    m.add_argument("--calib", required=True, help="Basalt calibration (pinhole)")
    m.add_argument("--offsets", default="1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256",
                   help="keyframe offsets to match")
    m.add_argument("--stereo", action="store_true",
                   help="also match the two cameras of each keyframe (their axes are 75.2 degrees apart, beyond "
                        "--max-axis-angle), within the epipolar band of the rig's calibration")
    m.add_argument("--stereo-tau", type=float, default=2.0,
                   help="--stereo: the epipolar band (px) of the two cameras of a keyframe (the factory rig)")
    m.add_argument("--max-axis-angle", type=float, default=75.0, help="skip pairs whose optical axes differ more")
    m.add_argument("--max-dist", type=float, default=200.0, help="skip pairs whose cameras are further apart (m)")
    m.add_argument("--tau0", type=float, default=2.0, help="epipolar band (px) at offset 0")
    m.add_argument("--tau-slope", type=float, default=0.1, help="band growth per keyframe offset (px)")
    m.add_argument("--tau-max", type=float, default=6.0)
    m.add_argument("--ratio", type=float, default=0.8, help="descriptor ratio test within the band")
    m.add_argument("--max-desc", type=float, default=None,
                   help="maximum descriptor L2 distance of the ratio test (default: 300 for SIFT, none for learned)")
    m.add_argument("--verify", choices=("band", "essential"), default="band",
                   help="band: keep the matches within the VIO's epipolar band; essential (LightGlue): keep those "
                        "an essential matrix explains, as loops does, whatever the VIO's translation")
    m.add_argument("--min-inliers", type=int, default=20, help="--verify essential: inliers to keep a pair")
    m.add_argument("--max-rot", type=float, default=10.0,
                   help="--verify essential: largest disagreement with the VIO rotation (deg)")
    m.add_argument("--jobs", type=int, default=2)
    lightglue_args(m)
    lp = sub.add_parser("loops", help="verified matches between revisits of the same place")
    lp.add_argument("--work", required=True)
    lp.add_argument("--vio", required=True)
    lp.add_argument("--calib", required=True)
    lp.add_argument("--min-gap", type=int, default=64, help="keyframes between the two visits")
    lp.add_argument("--max-dist", type=float, default=15.0, help="horizontal distance of the two keyframes (m)")
    lp.add_argument("--max-axis-angle", type=float, default=60.0)
    lp.add_argument("--step", type=int, default=2, help="candidate keyframe stride")
    lp.add_argument("--ratio", type=float, default=0.8)
    lp.add_argument("--max-desc", type=float, default=None,
                    help="maximum descriptor L2 distance of the ratio test (default: 300 for SIFT, none for learned)")
    lp.add_argument("--min-inliers", type=int, default=25, help="essential-matrix inliers to accept a pair")
    lp.add_argument("--max-rot", type=float, default=10.0, help="largest disagreement with the VIO rotation (deg)")
    lp.add_argument("--jobs", type=int, default=2)
    lightglue_args(lp)
    def solve_args(s):  # shared by solve and online
        s.add_argument("--work", required=True)
        s.add_argument("--imu", required=True, help="IMU csv (ns, gyro, accel) as the VIO used it")
        s.add_argument("--calib", required=True)
        s.add_argument("--vio", required=True)
        s.add_argument("--out", required=True, help="corrected trajectory (TUM)")
        s.add_argument("--est-extrinsic", default="01", help="cameras whose IMU-camera rotation is estimated")
        s.add_argument("--ext-segments", type=int, default=8,
                       help="estimate each such rotation in this many equal time segments of the keyframes "
                            "(1: one rotation per camera)")
        s.add_argument("--ext-seg-seconds", type=float, default=0.0,
                       help="segment duration in seconds (0: use --ext-segments); with --stereo-reference, "
                            "boundaries follow keyframe timestamps")
        s.add_argument("--ext-seg-sigma", type=float, default=0.05,
                       help="prior on the rotation change between consecutive segments (deg)")
        s.add_argument("--ext-seg-sigma-cams", default="",
                       help="--ext-seg-sigma per estimated camera, comma-separated in --est-extrinsic's order")
        s.add_argument("--stereo-reference", type=int, choices=(0, 1),
                       help="fix this camera's transform to the IMU and estimate the other camera's rotation "
                            "and translation in each segment (both focal scales can still vary with --est-focal)")
        s.add_argument("--fix-camera-rotations", action="store_true",
                       help="with --stereo-reference and --est-focal, keep both cameras' factory rotations and "
                            "estimate only the focal lengths per segment")
        s.add_argument("--fix-stereo-translation", action="store_true",
                       help="with --stereo-reference, keep the factory translation while allowing rotation drift")
        s.add_argument("--stereo-walk-sigma", type=float, default=0.0005,
                       help="stereo translation random-walk sigma per segment (m)")
        s.add_argument("--stereo-baseline-sigma", type=float, default=0.0005,
                       help="sigma for the factory stereo baseline length in each segment (m)")
        s.add_argument("--focal-seg-sigma-cams", default="",
                       help="--focal-seg-sigma per estimated camera, comma-separated in --est-extrinsic's order")
        s.add_argument("--max-offset", type=int, default=256, help="use matches up to this keyframe offset")
        s.add_argument("--band", type=int, default=64, help="landmark observations within this many keyframes")
        s.add_argument("--max-reproj", type=float, default=4.0, help="triangulation inlier threshold (px)")
        s.add_argument("--min-angle", type=float, default=1.0, help="minimum triangulation angle (deg)")
        s.add_argument("--pixel-sigma", type=float, default=1.0)
        s.add_argument("--sigma-size", type=float, default=0.0,
                       help="keypoint size (px) up to which an observation has --pixel-sigma; a larger keypoint's "
                       "sigma grows as (size / this) ** --sigma-size-pow (0: off)")
        s.add_argument("--sigma-size-pow", type=float, default=1.0)
        s.add_argument("--est-focal", action="store_true",
                       help="also estimate each estimated camera's focal length scale, per time segment")
        s.add_argument("--focal-seg-sigma", type=float, default=0.0005,
                       help="prior on the focal length change between consecutive segments (relative)")
        s.add_argument("--focal-seg-seconds", type=float, default=0.0,
                       help="estimate the focal lengths in time chunks of this many seconds, keeping each camera's "
                            "rotation in --ext-segments segments (needs --est-focal; 0: off)")
        s.add_argument("--rot-tie-sigma", type=float, default=1e-4,
                       help="with --focal-seg-seconds, prior on the rotation change between chunks of one rotation "
                            "segment (deg)")
        s.add_argument("--focal-seg-sigma-px", type=float,
                       help="focal drift sigma in pixels per segment, converted to the log-scale prior")
        s.add_argument("--focal-sigma", type=float, default=0.01, help="prior on each focal length scale (relative)")
        s.add_argument("--focal-mean-sigma", type=float, default=0.0,
                       help="prior on the mean of each camera's focal length scales over the segments (0: none); "
                       "a tight one keeps the calibration's focal length on average and lets only its course change")
        s.add_argument("--pcg-tol", type=float, default=1e-4,
                       help="relative residual at which conjugate gradients stop (with revisit matches)")
        s.add_argument("--pcg-iters", type=int, default=200, help="most conjugate-gradient iterations per step")
        s.add_argument("--huber", type=float, default=2.0,
                       help="Huber threshold in sigmas; negative: a Cauchy loss at that scale instead")
        s.add_argument("--iterations", type=int, default=60)
        s.add_argument("--rel-tol", type=float, default=1e-7, help="stop once an iteration lowers the cost less")
        s.add_argument("--lambda0", type=float, default=1e-4)
        s.add_argument("--outlier-px", type=float, default=10.0,
                       help="each iteration drops observations whose error exceeds this (0: keep all)")
        s.add_argument("--outlier-px2", type=float, default=3.0,
                       help="tighter threshold (px) from --outlier-start2 on, once the solution has settled (0: off)")
        s.add_argument("--outlier-start2", type=int, default=8)
        s.add_argument("--omega-coriolis", default="", help="x,y,z: Earth's rotation (rad/s) in the VIO's world frame, "
                       "e.g. w_e (0, cos lat, sin lat) turned into it; needs the world frame's north")
        s.add_argument("--prior-rot-sigma", type=float, default=1e-4,
                       help="prior on the first keyframe's rotation (rad); its position stays pinned")
        s.add_argument("--lm-damp-floor", type=float, default=0.01,
                       help="landmark damping floor, relative to the landmark's largest diagonal entry")
        s.add_argument("--kf-range", default="", help="START:END keyframe window (default: all)")
        s.add_argument("--imu-noise-scale", type=float, default=1.0,
                       help="multiplies the calibration's accelerometer and gyroscope noise densities")
        s.add_argument("--accel-rw-scale", type=float, default=1.0, help="multiplies the accel bias random walk")
        s.add_argument("--gyro-rw-scale", type=float, default=1.0, help="multiplies the gyro bias random walk")
        s.add_argument("--gyro-rw-warmup", type=float, default=1.0,
                       help="the gyro bias random walk is this many times looser at the recording's first keyframe, "
                            "decaying to 1 over --gyro-rw-warmup-s (the device warming up)")
        s.add_argument("--gyro-rw-warmup-s", type=float, default=300.0)
        s.add_argument("--accel-rw-warmup", type=float, default=1.0,
                       help="the accelerometer's bias random walk at the start, times the walk, decaying to 1 over "
                            "--gyro-rw-warmup-s as the gyroscope's (the device warming up)")
        s.add_argument("--loops", action="store_true", help="add the revisit matches of the loops stage")
        s.add_argument("--flow", default="", help="flow.npz of the flow step: its links join the matches")
        s.add_argument("--modes", default="", help="undamped deformation modes of the whole solution: yaw, scale "
                       "(comma-separated); they let LM move along slow drifts the damped steps cannot follow. "
                       "yawN splits the heading drift into N time pieces")
        s.add_argument("--modes-start", type=int, default=3, help="first iteration that uses the modes")
        s.add_argument("--mode-cap-yaw", type=float, default=0.01, help="largest yaw-mode step per iteration (rad)")
        s.add_argument("--mode-cap-scale", type=float, default=0.01, help="largest scale-mode step per iteration")
        s.add_argument("--est-gyro-m", action="store_true",
                       help="estimate the gyroscope's response error M, one 3 x 3 matrix for the recording: the rate "
                            "is (I + M) times the measured one, less the bias")
        s.add_argument("--gyro-m-sigma", type=float, default=0.002, help="prior on each entry of M")
        s.add_argument("--gyro-m-pass", type=int, default=10,
                       help="iterations between preintegrating the IMU again with the current M")
        s.add_argument("--gyro-m", default="", help="9 comma-separated values: M to start from (row-major)")
        s.add_argument("--focal-temp", default="",
                       help="the SLAM cameras' temperature (lamaria_temperature.py) for --focal-temp-coef")
        s.add_argument("--focal-fisheye", default="",
                       help="EPS0,EPS1: correct the keypoints for fisheye focal lengths (1 + EPS) times the factory's "
                            "(needs --fisheye-calib)")
        s.add_argument("--fisheye-calib", default="", help="the Aria factory calibration json for --focal-fisheye")
        s.add_argument("--fisheye-model", default="",
                       help="LaMAria's aria_calibrations json: solve on each camera's factory fisheye model, its "
                            "focal lengths (--est-focal) the fisheye's")
        s.add_argument("--focal-temp-coef", type=float, default=0.0,
                       help="relative focal length change per degree Celsius, about the recording's median")
        s.add_argument("--est-accel-m", action="store_true",
                       help="estimate the accelerometer's response error M_a likewise (scales, misalignment)")
        s.add_argument("--accel-m-sigma", type=float, default=0.003, help="prior on each entry of M_a")
        s.add_argument("--accel-m", default="", help="9 comma-separated values: M_a to start from (row-major)")
        s.add_argument("--imu-temp", default="",
                       help="the SLAM cameras' temperature (lamaria_temperature.py) for the gyroscope's thermal model")
        s.add_argument("--est-gyro-bias-temp", action="store_true",
                       help="estimate the gyroscope bias's change per degree (rad/s per degree, one per axis)")
        s.add_argument("--gyro-bias-temp-sigma", type=float, default=5e-5, help="its prior (rad/s per degree)")
        s.add_argument("--est-gyro-scale-temp", action="store_true",
                       help="estimate each gyroscope axis's scale change per degree")
        s.add_argument("--gyro-scale-temp-sigma", type=float, default=3e-5, help="its prior (per degree)")
        s.add_argument("--accel-noise", type=float, default=None,
                       help="accelerometer noise density (m/s^2/sqrt(Hz)) instead of the calibration's")
        s.add_argument("--gyro-noise", type=float, default=None,
                       help="gyroscope noise density (rad/s/sqrt(Hz)) instead of the calibration's")
        s.add_argument("--accel-bias-rw", type=float, default=None,
                       help="accelerometer bias random walk (m/s^2/sqrt(s)) instead of the calibration's")
        s.add_argument("--gyro-bias-rw", type=float, default=None,
                       help="gyroscope bias random walk (rad/s/sqrt(s)) instead of the calibration's "
                            "(--gyro-rw-scale still multiplies it)")
        s.add_argument("--bias-chunk-seconds", type=float, default=0.0,
                       help="biases constant within chunks of this many seconds, stepping between them by "
                            "--bias-chunk-accel / --bias-chunk-gyro (SLAM Dunk's 5 s; 0: the continuous walk)")
        s.add_argument("--bias-chunk-accel", type=float, default=0.005, help="accelerometer bias step (m/s^2)")
        s.add_argument("--bias-chunk-gyro", type=float, default=2e-5, help="gyroscope bias step (rad/s)")
        s.add_argument("--est-time-offsets", action="store_true",
                       help="estimate the IMU's clock against the cameras' and the accelerometer's against the "
                            "gyroscope's (SLAM Dunk's clock offsets)")
        s.add_argument("--time-offset-sigma", type=float, default=1e-3, help="their prior (s)")
        s.add_argument("--save-states", action="store_true",
                       help="also write <out>_states.npz (keyframes, landmarks, observations) for loop-correct")
        s.add_argument("--init-states", dest="init_states_file", default=None,
                       help="<out>_states.npz of a solve before: start from its velocities, biases, camera "
                            "rotations, focal lengths and IMU response")
        s.add_argument("--fix-calib", action="store_true",
                       help="with --init-states, hold that calibration as it is (cameras per time segment, IMU "
                            "response) and estimate the trajectory only: SLAM Dunk's final refinement")

    solve_args(sub.add_parser("solve", help="batch VI bundle adjustment"))
    o = sub.add_parser("online", help="causal mode: solve windows of the most recent keyframes as the data arrives")
    solve_args(o)
    o.add_argument("--window", type=int, default=480, help="keyframes per window (the most recent ones)")
    o.add_argument("--step", type=int, default=60, help="keyframes between two windows")
    o.set_defaults(iterations=10, ext_segments=2)  # per window
    fl = sub.add_parser("flow", help="link consecutive keyframes by Lucas-Kanade flow through the frames between "
                        "them (after match)")
    fl.add_argument("--mav0", required=True, help="ASL mav0 directory (cam0/data, cam1/data), every frame")
    fl.add_argument("--work", required=True)
    fl.add_argument("--jobs", type=int, default=2)
    fl.add_argument("--fb-max", type=float, default=0.5, help="largest forward-backward error (px)")
    fl.add_argument("--assoc", type=float, default=1.5, help="link to a keypoint within this distance (px)")
    fl.add_argument("--min-dist", type=float, default=3.0,
                    help="add a point as a keypoint when no keypoint is within this distance (px)")
    c = sub.add_parser("loop-correct", help="close the revisits of a solve with a pose graph (after solve "
                       "--save-states and loops)")
    c.add_argument("--work", required=True)
    c.add_argument("--calib", required=True)
    c.add_argument("--states", required=True, help="<out>_states.npz of the solve")
    c.add_argument("--tum", required=True, help="the solve's output trajectory")
    c.add_argument("--out", required=True, help="corrected trajectory (TUM)")
    c.add_argument("--min-inliers", type=int, default=15, help="PnP inliers for a loop edge")
    c.add_argument("--gate", default="0.3,0.01,2.0",
                   help="drop an edge disagreeing with the solve by more than BASE + RATE * (time between the visits)"
                        " metres, or by more than MAX_ROT degrees: BASE,RATE,MAX_ROT")
    c.add_argument("--odo-rot-sigma", type=float, default=2e-4, help="rad, per consecutive keyframe pair")
    c.add_argument("--odo-pos-sigma", type=float, default=0.02, help="m, per consecutive keyframe pair")
    c.add_argument("--loop-rot-sigma", type=float, default=0.005, help="rad")
    c.add_argument("--loop-pos-sigma", type=float, default=0.1, help="m")
    a = p.parse_args()
    {"features": cmd_features, "match": cmd_match, "loops": cmd_loops, "solve": cmd_solve, "online": cmd_online,
     "loop-correct": cmd_loop_correct, "flow": cmd_flow}[a.cmd](a)


if __name__ == "__main__":
    main()
