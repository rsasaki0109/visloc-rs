#!/usr/bin/env python3
"""OpenLORIS-Scene's D435i colour + aligned depth as a TUM RGB-D sequence and a EuRoC query set.

    python3 scripts/openloris_to_tum_rgbd.py --sequence data/office1-1 --out data/office1-1_rgbd [--stride 1]

`--sequence` is an extracted sequence directory with `color/`,
`aligned_depth/`, `color.txt`, `aligned_depth.txt`, `groundtruth.txt`,
`sensors.yaml` and `trans_matrix.yaml`. Writes, under `--out`:

- `rgb.txt`, `depth.txt`, `rgb/`, `depth/` (links to the images): the TUM
  RGB-D layout `tum_rgbd_save_map` reads. Each colour image is paired with
  the nearest aligned depth image (within 20 ms). The depth is in
  millimetres, so pass `--depth-scale 0.001`.
- `camera.tum`: the colour camera's optical frame in the motion-capture
  frame at each colour stamp. It is `groundtruth.txt`'s `base_link`
  interpolated at the stamp (linearly, slerp for the rotation), composed
  with `trans_matrix.yaml`'s `base_link -> d400_color_optical_frame`.
  Stamps outside the ground truth are left out.
- `mav0/cam0/data/` and `data.csv`: the same colour images, grey, for
  `basalt_euroc_localize_demo`, every `--stride`-th one.
- `intrinsics.txt`: `fx fy cx cy` of `d400_color_optical_frame` (the
  dataset gives no distortion for it).

All the office sequences share one motion-capture frame, so a map built
from one sequence and queries from another can be scored against
`camera.tum` directly.
"""

import argparse
import os
import re

import numpy as np


def read_rows(path):
    return [line.split() for line in open(path, encoding='utf-8') if line.strip() and not line.startswith('#')]


def quat_to_rot(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rot_to_quat(R):
    """x y z w (Shepperd)."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q


def slerp(q0, q1, t):
    d = float(q0 @ q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


def base_to_color(path):
    text = open(path, encoding='utf-8').read()
    m = re.search(r'parent_frame:\s*base_link\s*child_frame:\s*d400_color_optical_frame\s*matrix:.*?data:\s*\[([^\]]*)\]',
                  text, re.S)
    return np.array([float(v) for v in m.group(1).split(',') if v.strip()]).reshape(4, 4)


def intrinsics(path):
    text = open(path, encoding='utf-8').read()
    start = text.index('d400_color_optical_frame:')
    m = re.search(r'intrinsics:.*?data:\s*\[([^\]]*)\]', text[start:], re.S)
    fx, cx, fy, cy = (float(v) for v in m.group(1).split(',') if v.strip())   # the file's order: fx cx fy cy
    return fx, fy, cx, cy


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--sequence', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--stride', type=int, default=1, help='every N-th image in the EuRoC query set')
    args = p.parse_args()
    from PIL import Image  # noqa: PLC0415
    seq = os.path.abspath(args.sequence)
    out = args.out
    os.makedirs(os.path.join(out, 'mav0', 'cam0', 'data'), exist_ok=True)
    for name in ('rgb', 'depth'):
        if not os.path.exists(os.path.join(out, name)):
            os.symlink(os.path.join(seq, 'color' if name == 'rgb' else 'aligned_depth'), os.path.join(out, name))
    fx, fy, cx, cy = intrinsics(os.path.join(seq, 'sensors.yaml'))
    with open(os.path.join(out, 'intrinsics.txt'), 'w', encoding='utf-8') as f:
        f.write(f'{fx} {fy} {cx} {cy}\n')
    T_bc = base_to_color(os.path.join(seq, 'trans_matrix.yaml'))
    gt = np.array([[float(v) for v in r[:8]] for r in read_rows(os.path.join(seq, 'groundtruth.txt'))])
    colour = [(float(r[0]), r[0], r[1]) for r in read_rows(os.path.join(seq, 'color.txt'))]
    depth = [(float(r[0]), r[0], r[1]) for r in read_rows(os.path.join(seq, 'aligned_depth.txt'))]
    dstamps = np.array([d[0] for d in depth])
    rgb_rows, depth_rows, traj_rows, csv_rows = [], [], [], []
    for n, (t, ts, rel) in enumerate(colour):
        k = int(np.searchsorted(gt[:, 0], t))
        if k == 0 or k >= len(gt):
            continue
        a, b = gt[k - 1], gt[k]
        u = (t - a[0]) / max(b[0] - a[0], 1e-9)
        pos = (1 - u) * a[1:4] + u * b[1:4]
        q = slerp(a[4:8] / np.linalg.norm(a[4:8]), b[4:8] / np.linalg.norm(b[4:8]), u)
        T_wb = np.eye(4)
        T_wb[:3, :3] = quat_to_rot(*q)
        T_wb[:3, 3] = pos
        T_wc = T_wb @ T_bc
        qc = rot_to_quat(T_wc[:3, :3])
        j = int(np.argmin(np.abs(dstamps - t)))
        if abs(dstamps[j] - t) > 0.02:
            continue
        rgb_rows.append(f'{ts} rgb/{os.path.basename(rel)}\n')
        depth_rows.append(f'{ts} depth/{os.path.basename(depth[j][2])}\n')
        traj_rows.append(f'{ts} {T_wc[0, 3]:.6f} {T_wc[1, 3]:.6f} {T_wc[2, 3]:.6f} '
                         f'{qc[0]:.8f} {qc[1]:.8f} {qc[2]:.8f} {qc[3]:.8f}\n')
        if n % args.stride == 0:
            ns = int(round(t * 1e9))
            Image.open(os.path.join(seq, rel)).convert('L').save(os.path.join(out, 'mav0', 'cam0', 'data', f'{ns}.png'))
            csv_rows.append(f'{ns},{ns}.png\n')
    for name, rows in (('rgb.txt', rgb_rows), ('depth.txt', depth_rows)):
        with open(os.path.join(out, name), 'w', encoding='utf-8') as f:
            f.write('# OpenLORIS-Scene D435i\n# timestamp filename\n# \n')
            f.writelines(rows)
    with open(os.path.join(out, 'camera.tum'), 'w', encoding='utf-8') as f:
        f.writelines(traj_rows)
    with open(os.path.join(out, 'mav0', 'cam0', 'data.csv'), 'w', encoding='utf-8') as f:
        f.write('#timestamp [ns],filename\n')
        f.writelines(csv_rows)
    print(f'{len(rgb_rows)} RGB-D frames, {len(csv_rows)} query images, fx {fx:.1f} fy {fy:.1f} cx {cx:.1f} cy {cy:.1f}')


if __name__ == '__main__':
    main()
