#!/usr/bin/env python3
"""Lay out a gsplat-ready `<out>/undistorted/{images,sparse}` from a visloc-rs
unordered-SfM COLMAP model plus the original (distorted) source photos.

The SfM demo exports an *ideal-pinhole* COLMAP model (cameras.txt = PINHOLE) whose
poses/points are gauge-free monocular; the gsplat trainer needs the matching
*undistorted* images. This script:

  1. undistorts each source photo from its real intrinsics (SIMPLE_PINHOLE /
     SIMPLE_RADIAL / RADIAL / PINHOLE / OPENCV in sparse/cameras.txt) to the ideal
     pinhole, optionally downscaling (caps trainer VRAM), writing PNGs;
  2. writes cameras.txt = the (scaled) PINHOLE, images.txt with the SfM poses and
     NAME rewritten .JPG/.jpg -> .png, and points3D.txt recoloured by projecting
     every track point into the images that observe it (the SfM exporter writes
     white points; a real colour gives the SH DC term a good init).

Usage:
  scripts/prepare_3dgs_from_colmap.py \
    --source-images ~/datasets/south-building/south-building/images \
    --source-cameras ~/datasets/south-building/south-building/sparse/cameras.txt \
    --colmap-model target/colmap_sfm_benchmark/south-building/colmap \
    --out /tmp/sb_3dgs --scale 0.5
"""
import argparse, os, math, numpy as np, cv2
from PIL import Image


def read_source_cam(path):
    for l in open(path):
        if l.startswith("#") or not l.strip():
            continue
        v = l.split(); model = v[1]; W, H = int(v[2]), int(v[3]); p = list(map(float, v[4:]))
        if model in ("SIMPLE_PINHOLE",):
            f, cx, cy = p[0], p[1], p[2]; return model, W, H, f, f, cx, cy, [0, 0, 0, 0, 0]
        if model in ("SIMPLE_RADIAL", "RADIAL"):
            f, cx, cy, k1 = p[0], p[1], p[2], p[3]
            k2 = p[4] if model == "RADIAL" and len(p) > 4 else 0.0
            return model, W, H, f, f, cx, cy, [k1, k2, 0, 0, 0]
        if model in ("PINHOLE",):
            return model, W, H, p[0], p[1], p[2], p[3], [0, 0, 0, 0, 0]
        if model in ("OPENCV", "FULL_OPENCV"):
            fx, fy, cx, cy, k1, k2, p1, p2 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]
            return model, W, H, fx, fy, cx, cy, [k1, k2, p1, p2, p[8] if len(p) > 8 else 0.0]
        raise SystemExit(f"unsupported camera model {model}")
    raise SystemExit("no camera in " + path)


def Q2R(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z); w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-images", required=True)
    ap.add_argument("--source-cameras", required=True)
    ap.add_argument("--colmap-model", required=True, help="visloc SfM out dir (cameras/images/points3D.txt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args()

    _, W0, H0, fx, fy, cx, cy, dist = read_source_cam(a.source_cameras)
    K0 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], np.float64)
    m1, m2 = cv2.initUndistortRectifyMap(K0, np.array(dist), None, K0, (W0, H0), cv2.CV_16SC2)
    s = a.scale; Wn, Hn = int(round(W0 * s)), int(round(H0 * s))
    outimg = os.path.join(a.out, "undistorted", "images"); outsp = os.path.join(a.out, "undistorted", "sparse")
    os.makedirs(outimg, exist_ok=True); os.makedirs(outsp, exist_ok=True)

    # which source images the SfM model references, and their poses
    poses = {}
    for l in open(os.path.join(a.colmap_model, "images.txt")):
        if l.startswith("#"):
            continue
        v = l.split()
        if len(v) >= 10 and v[9].lower().endswith((".jpg", ".jpeg", ".png")):
            poses[int(v[0])] = (Q2R(*map(float, v[1:5])), np.array(list(map(float, v[5:8]))), v[9])
    names = {p[2] for p in poses.values()}
    print(f"{len(names)} images referenced by the SfM model")

    imcache = {}
    for nm in sorted(names):
        img = cv2.imread(os.path.join(a.source_images, nm))
        if img is None:
            print("MISSING", nm); continue
        und = cv2.remap(img, m1, m2, cv2.INTER_LINEAR)
        small = cv2.resize(und, (Wn, Hn), interpolation=cv2.INTER_AREA)
        png = os.path.splitext(nm)[0] + ".png"
        cv2.imwrite(os.path.join(outimg, png), small)
        imcache[nm] = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    print(f"undistorted + scaled {len(imcache)} images -> {Wn}x{Hn}")

    with open(os.path.join(outsp, "cameras.txt"), "w") as f:
        f.write("# Camera list\n")
        f.write(f"0 PINHOLE {Wn} {Hn} {fx*s:.6f} {fy*s:.6f} {cx*s:.6f} {cy*s:.6f}\n")
    with open(os.path.join(a.colmap_model, "images.txt")) as fi, \
         open(os.path.join(outsp, "images.txt"), "w") as fo:
        for l in fi:
            if l.startswith("#"):
                fo.write(l); continue
            v = l.split()
            if len(v) >= 10 and v[9].lower().endswith((".jpg", ".jpeg", ".png")):
                v[9] = os.path.splitext(v[9])[0] + ".png"; fo.write(" ".join(v) + "\n")
            else:
                fo.write(l)

    # recolour points by projecting into observing images (scaled pinhole)
    Ks = np.array([[fx * s, 0, cx * s], [0, fy * s, cy * s], [0, 0, 1.0]])
    out = []; colored = 0; total = 0
    for l in open(os.path.join(a.colmap_model, "points3D.txt")):
        if l.startswith("#"):
            out.append(l); continue
        v = l.split()
        if len(v) < 8:
            out.append(l); continue
        total += 1
        X = np.array(list(map(float, v[1:4]))); track = v[8:]; cols = []
        for k in range(0, len(track) - 1, 2):
            iid = int(track[k])
            if iid not in poses:
                continue
            R, t, nm = poses[iid]
            if nm not in imcache:
                continue
            Xc = R @ X + t
            if Xc[2] <= 1e-6:
                continue
            u = Ks[0, 0] * Xc[0] / Xc[2] + Ks[0, 2]; w_ = Ks[1, 1] * Xc[1] / Xc[2] + Ks[1, 2]
            ui, vi = int(round(u)), int(round(w_)); im = imcache[nm]
            if 0 <= vi < im.shape[0] and 0 <= ui < im.shape[1]:
                cols.append(im[vi, ui].astype(np.float32))
        c = np.clip(np.mean(cols, 0), 0, 255).astype(int) if cols else np.array([128, 128, 128])
        if cols:
            colored += 1
        v[4], v[5], v[6] = str(c[0]), str(c[1]), str(c[2])
        out.append(" ".join(v) + "\n")
    open(os.path.join(outsp, "points3D.txt"), "w").writelines(out)
    print(f"recoloured {colored}/{total} points; layout ready at {a.out}/undistorted")


if __name__ == "__main__":
    main()
