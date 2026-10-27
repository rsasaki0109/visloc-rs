#!/usr/bin/env python3
"""Move a COLMAP text model into a reference model's frame.

    align_colmap_model.py EST_DIR REF_IMAGES_TXT OUT_DIR [--names IMAGE_NAMES_TXT]
                          [--centres]

Fits the Sim(3) taking EST_DIR's cameras onto the reference images with
the same file names (compared without directories) and writes the model
with its poses and points transformed to OUT_DIR. Observations and
tracks are kept as they are, so a dense reconstruction from OUT_DIR lands
in the reference frame (ETH3D's laser scans, for example). With
``--names`` (a ``gsplat_photos`` run's ``image_names.txt``: working name,
original path), images are matched and written under their original file
names.

The camera centres are fitted first, refitted once without the cameras
more than three times the median error off. The rotation then comes from
the cameras' orientations (the chordal mean of R_ref^T R_est over the
cameras kept by the centre fit, refitted once without those more than
three times the median angle off), and the scale and translation from the
centres under it, fitted on the cameras the orientations keep and refitted
once without centres beyond three times the median error: centres nearly
in one plane, as on ETH3D's meadow, fix the tilt poorly, and one camera a
few centimetres off can tilt the cloud by a degree. ``--centres`` keeps
the centres' fit, the alignment before the orientations were used
(``--rotations``, their old switch, is accepted). Prints the fit: matched cameras, scale, centre errors in
reference units and the rotation's change. Needs numpy.
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np


def quat_to_r(qw, qx, qy, qz):
    # Normalized: ETH3D's quaternions carry six digits, and the angles taken
    # by arccos near the identity magnify their norm's error to 0.1 deg.
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def r_to_quat(r):
    w = np.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2
    x = np.copysign(x, r[2, 1] - r[1, 2])
    y = np.copysign(y, r[0, 2] - r[2, 0])
    z = np.copysign(z, r[1, 0] - r[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def read_images(path):
    """(header tokens, observation line) per image, in file order."""
    images = []
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f if not l.startswith("#")]
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        tokens = lines[i].split()
        obs = lines[i + 1] if i + 1 < len(lines) else ""
        images.append((tokens, obs))
        i += 2
    return images


def centre(tokens):
    r = quat_to_r(*map(float, tokens[1:5]))
    t = np.array(list(map(float, tokens[5:8])))
    return -r.T @ t


def sim3(ref, est):
    """(scale, rotation, translation) with ref ≈ s R est + t."""
    mr, me = ref.mean(0), est.mean(0)
    r0, e0 = ref - mr, est - me
    u, s, vt = np.linalg.svd(r0.T @ e0)
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    rot = u @ d @ vt
    scale = (s * np.diag(d)).sum() / (e0**2).sum()
    return scale, rot, mr - scale * rot @ me


def mean_rotation(rs):
    """The rotation nearest the mean of `rs` (chordal L2 mean)."""
    u, _, vt = np.linalg.svd(sum(rs))
    return u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt


def angle_deg(r):
    return np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0)))


def main():
    args = sys.argv[1:]
    rotations = "--centres" not in args
    args = [a for a in args if a not in ("--centres", "--rotations")]
    original = {}
    if "--names" in args:
        k = args.index("--names")
        with open(args[k + 1]) as f:
            for line in f:
                if line.strip():
                    work, path = line.rstrip("\n").split(" ", 1)
                    original[work] = os.path.basename(path)
        del args[k:k + 2]
    if len(args) != 3:
        raise SystemExit(__doc__)
    est_dir, ref_path, out_dir = args
    est = read_images(os.path.join(est_dir, "images.txt"))
    for tokens, _ in est:
        name = " ".join(tokens[9:])
        tokens[9:] = [original.get(name, name)]
    ref_images = read_images(ref_path)
    ref = {os.path.basename(" ".join(t[9:])): centre(t) for t, _ in ref_images}
    ref_rot = {os.path.basename(" ".join(t[9:])): quat_to_r(*map(float, t[1:5])) for t, _ in ref_images}
    names = [os.path.basename(" ".join(t[9:])) for t, _ in est]
    common = [k for k, n in enumerate(names) if n in ref]
    if len(common) < 3:
        raise SystemExit(f"only {len(common)} images in common with the reference")
    a = np.array([ref[names[k]] for k in common])
    b = np.array([centre(est[k][0]) for k in common])
    s, rot, t = sim3(a, b)
    err = np.linalg.norm(a - (s * b @ rot.T + t), axis=1)
    keep = err <= 3 * np.median(err) + 1e-12
    if not keep.all() and keep.sum() >= 3:
        s, rot, t = sim3(a[keep], b[keep])
        err = np.linalg.norm(a - (s * b @ rot.T + t), axis=1)
    note = ""
    if rotations:
        # Camera k: R_ref = R_est rot^T, so rot = R_ref^T R_est.
        rs = [ref_rot[names[k]].T @ quat_to_r(*map(float, est[k][0][1:5])) for k in common]
        used = [i for i in range(len(common)) if keep[i]]
        mean = mean_rotation([rs[i] for i in used])
        off = np.array([angle_deg(rs[i] @ mean.T) for i in used])
        if (off > 3 * np.median(off) + 1e-9).any():
            used = [i for i, o in zip(used, off) if o <= 3 * np.median(off) + 1e-9]
            mean = mean_rotation([rs[i] for i in used])
        change = angle_deg(mean @ rot.T)
        rot = mean

        def scale_translation(k):
            ma, mb = a[k].mean(0), b[k].mean(0)
            rb = (b[k] - mb) @ rot.T
            sk = ((a[k] - ma) * rb).sum() / (rb**2).sum()
            return sk, ma - sk * rot @ mb

        # Scale and translation from the cameras whose orientations agree,
        # then refitted once without centres beyond three times the median
        # error: a part of the model bent away from the rest (the far end of
        # ETH3D's facade) would otherwise pull the whole cloud along.
        keep = np.zeros(len(common), bool)
        keep[used] = True
        s, t = scale_translation(keep)
        err = np.linalg.norm(a - (s * b @ rot.T + t), axis=1)
        keep = err <= 3 * np.median(err) + 1e-12
        if keep.sum() >= 3:
            s, t = scale_translation(keep)
            err = np.linalg.norm(a - (s * b @ rot.T + t), axis=1)
        note = f"; rotation from {len(used)} cameras' orientations, {change:.3f} deg from the centres' fit"

    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(os.path.join(est_dir, "cameras.txt"), os.path.join(out_dir, "cameras.txt"))
    with open(os.path.join(out_dir, "images.txt"), "w") as f:
        f.write("# Image list (aligned)\n")
        for tokens, obs in est:
            r = quat_to_r(*map(float, tokens[1:5]))
            c = s * rot @ centre(tokens) + t
            r_new = r @ rot.T
            q = r_to_quat(r_new)
            t_new = -r_new @ c
            f.write(" ".join([tokens[0], *(f"{v:.17g}" for v in q), *(f"{v:.17g}" for v in t_new), *tokens[8:]]) + "\n")
            f.write(obs + "\n")
    with open(os.path.join(est_dir, "points3D.txt")) as src, open(os.path.join(out_dir, "points3D.txt"), "w") as f:
        for line in src:
            if line.startswith("#") or not line.strip():
                f.write(line)
                continue
            tokens = line.split()
            x = s * rot @ np.array(list(map(float, tokens[1:4]))) + t
            f.write(" ".join([tokens[0], *(f"{v:.17g}" for v in x), *tokens[4:]]) + "\n")
    print(f"aligned {len(common)} of {len(est)} images to the reference ({len(ref)}): "
          f"scale {s:.6g}, centre error median {np.median(err):.4f}, max {err.max():.4f} "
          f"({int((~keep).sum())} left out of the fit){note}")


if __name__ == "__main__":
    main()
