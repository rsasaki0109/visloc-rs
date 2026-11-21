#!/usr/bin/env python3
"""Put an SfM model of TUM RGB-D frames into metres, z up (docs/navigation.md).

    python3 ros2/visloc-ros2/gs/align_sfm_tum.py --model runs/gs_tum/sfm/sparse/0 \
        --data runs/gs_tum/data --out runs/gs_tum/metric

Photo SfM has no scale and no up. The TUM sequences' motion-capture
trajectories have both (metres, z up, the floor near z = 0), so this fits the
similarity (scale, rotation, translation) that takes the SfM camera centres
onto them, by Umeyama's method, and applies it to the whole model. The
ground truth is used only for those 7 numbers; every pose stays the SfM's.
The residual is how far the SfM's own trajectory is from the ground truth.

Image names must be `<sequence>_<timestamp>.png`, where the sequence is the
part after `rgbd_dataset_freiburg2_pioneer_` (`360`, `slam`, `slam3`).

Writes OUT/sparse/0 (the model in metres, COLMAP text), OUT/images (links to
the model's images), OUT/<sequence>.tum (each sequence's SfM camera poses, TUM
format, for `tum_rgbd_save_map --keep-world`) and OUT/alignment.json.
"""

import argparse
import bisect
import json
import os

import numpy as np


def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rot2qvec(R):
    K = np.array([[R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
                  [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
                  [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], 0],
                  [R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0], R[0, 0] + R[1, 1] + R[2, 2]]]) / 3
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]
    return q if q[0] >= 0 else -q


def umeyama(src, dst):
    """s, R, t minimizing |dst - (s R src + t)|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


def read_images(path):
    images, lines = {}, open(path, encoding='utf-8').read().splitlines()
    rows = [line for line in lines if line and not line.startswith('#')]
    for header, points in zip(rows[0::2], rows[1::2]):
        f = header.split()
        images[int(f[0])] = dict(q=np.array([float(v) for v in f[1:5]]), t=np.array([float(v) for v in f[5:8]]),
                                 cam=int(f[8]), name=f[9], points=points)
    return images


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--model', required=True, help='COLMAP text model (cameras/images/points3D.txt)')
    p.add_argument('--images', default='', help="the model's image folder (default: ../../images of --model)")
    p.add_argument('--data', required=True, help='folder of the rgbd_dataset_freiburg2_pioneer_* sequences')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    images = read_images(os.path.join(args.model, 'images.txt'))
    gt = {}
    for name in os.listdir(args.data):
        if name.startswith('rgbd_dataset_freiburg2_pioneer_'):
            g = np.loadtxt(os.path.join(args.data, name, 'groundtruth.txt'))
            gt[name[len('rgbd_dataset_freiburg2_pioneer_'):]] = g
    src, dst, ids = [], [], []
    for iid, im in images.items():
        seq, ts = im['name'][:-4].split('_', 1)
        g = gt[seq]
        k = bisect.bisect_left(g[:, 0], float(ts))
        k = min(range(max(k - 1, 0), min(k + 1, len(g) - 1) + 1), key=lambda j: abs(g[j, 0] - float(ts)))
        if abs(g[k, 0] - float(ts)) > 0.02:
            continue
        R = qvec2rot(im['q'])
        src.append(-R.T @ im['t'])
        dst.append(g[k, 1:4])
        ids.append(iid)
    src, dst = np.array(src), np.array(dst)
    s, R, t = umeyama(src, dst)
    err = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
    # Robust refit on the inliers (a few badly registered frames must not tilt the floor).
    keep = err < max(3 * np.median(err), 0.05)
    s, R, t = umeyama(src[keep], dst[keep])
    err = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
    print(f'{len(images)} images, {len(src)} with ground truth; scale {s:.4f} m/unit; '
          f'camera centre error RMSE {np.sqrt((err ** 2).mean()) * 100:.1f} cm, median {np.median(err) * 100:.1f} cm, '
          f'max {err.max() * 100:.1f} cm')

    out_model = os.path.join(args.out, 'sparse', '0')
    os.makedirs(out_model, exist_ok=True)
    with open(os.path.join(args.model, 'cameras.txt'), encoding='utf-8') as f:
        cams = f.read()
    with open(os.path.join(out_model, 'cameras.txt'), 'w', encoding='utf-8') as f:
        f.write(cams)
    tum = {}
    with open(os.path.join(out_model, 'images.txt'), 'w', encoding='utf-8') as f:
        for iid in sorted(images):
            im = images[iid]
            Rcw = qvec2rot(im['q'])
            c = s * R @ (-Rcw.T @ im['t']) + t      # centre in metres
            Rcw_new = Rcw @ R.T
            tcw = -Rcw_new @ c
            q = rot2qvec(Rcw_new)
            f.write(f"{iid} {' '.join(f'{v:.9f}' for v in q)} {' '.join(f'{v:.9f}' for v in tcw)} "
                    f"{im['cam']} {im['name']}\n{im['points']}\n")
            seq, ts = im['name'][:-4].split('_', 1)
            qwc = rot2qvec(Rcw_new.T)
            tum.setdefault(seq, []).append(f"{ts} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f} "
                                           f"{qwc[1]:.8f} {qwc[2]:.8f} {qwc[3]:.8f} {qwc[0]:.8f}\n")
    with open(os.path.join(args.model, 'points3D.txt'), encoding='utf-8') as fin, \
            open(os.path.join(out_model, 'points3D.txt'), 'w', encoding='utf-8') as fout:
        for line in fin:
            if line.startswith('#') or not line.strip():
                fout.write(line)
                continue
            f = line.split()
            x = s * R @ np.array([float(v) for v in f[1:4]]) + t
            fout.write(' '.join([f[0], *(f'{v:.6f}' for v in x), *f[4:]]) + '\n')
    for seq, lines in tum.items():
        with open(os.path.join(args.out, f'{seq}.tum'), 'w', encoding='utf-8') as f:
            f.writelines(sorted(lines, key=lambda line: float(line.split()[0])))
    src_images = args.images or os.path.abspath(os.path.join(args.model, '..', '..', 'images'))
    link = os.path.join(args.out, 'images')
    if not os.path.exists(link):
        os.symlink(os.path.abspath(src_images), link)
    with open(os.path.join(args.out, 'alignment.json'), 'w', encoding='utf-8') as f:
        json.dump({'scale': s, 'rotation': R.tolist(), 'translation': t.tolist(), 'images': len(images),
                   'with_ground_truth': len(src), 'rmse_m': float(np.sqrt((err ** 2).mean())),
                   'median_m': float(np.median(err)), 'max_m': float(err.max())}, f, indent=2)


if __name__ == '__main__':
    main()
