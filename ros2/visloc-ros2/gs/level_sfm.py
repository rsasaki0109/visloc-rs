#!/usr/bin/env python3
"""Level a metric SfM model: z up, the floor at z = 0 (docs/navigation.md).

    python3 ros2/visloc-ros2/gs/level_sfm.py --model runs/om360/sparse/0 --out runs/om360/level

For a model already in metres (e.g. `gsplat_photos --insv`, scaled by the
camera's IMU) of a walk or drive over a flat floor:

1. **Up.** The cameras' mean "up" (their -y axis), over the cameras that
   look roughly level, refined by the plane through the camera centres
   (the camera height barely changes on a walk); the sign follows the
   cameras' up.
2. **Floor.** RANSAC over the SfM points below the cameras for the largest
   plane within 15 degrees of up; the model is turned so its normal is z and
   shifted so it is at z = 0.
3. **Origin.** x, y of the first camera; its view direction is +x.

Writes OUT/sparse/0 (COLMAP text), OUT/images (link) and OUT/level.json
(the rotation, the shift, the camera height above the floor and the floor
plane's inlier share).
"""

import argparse
import json
import os

import numpy as np

from align_sfm_tum import qvec2rot, read_images, rot2qvec


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--model', required=True)
    p.add_argument('--images', default='', help="the model's image folder (default: ../../images of --model)")
    p.add_argument('--out', required=True)
    p.add_argument('--floor-tolerance', type=float, default=0.03, help='RANSAC inlier distance, m')
    args = p.parse_args()
    images = read_images(os.path.join(args.model, 'images.txt'))
    Rs = {i: qvec2rot(im['q']) for i, im in images.items()}
    centres = np.array([-Rs[i].T @ images[i]['t'] for i in sorted(images)])
    ups = np.array([-Rs[i][1] for i in sorted(images)])        # camera -y in the world
    fwd = np.array([Rs[i][2] for i in sorted(images)])
    # 1. up: mean camera up of the level-looking cameras, then the path plane
    up = ups.sum(0)
    up /= np.linalg.norm(up)
    level = np.abs(fwd @ up) < 0.3
    up = ups[level].sum(0)
    up /= np.linalg.norm(up)
    c = centres - centres.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    normal = vt[2] * np.sign(vt[2] @ up)
    spread = np.linalg.svd(c, compute_uv=False)
    if spread[2] < 0.2 * spread[1] and normal @ up > np.cos(np.radians(15)):
        up = normal
    # 2. floor: RANSAC on points below the cameras
    pts, lines = [], []
    with open(os.path.join(args.model, 'points3D.txt'), encoding='utf-8') as f:
        for line in f:
            lines.append(line)
            if line.startswith('#') or not line.strip():
                continue
            pts.append([float(v) for v in line.split()[1:4]])
    pts = np.array(pts)
    h_cam = np.median(centres @ up)
    below = pts[(pts @ up) < h_cam - 0.3]
    rng = np.random.default_rng(0)
    best, best_n = None, None
    for _ in range(3000):
        s = below[rng.choice(len(below), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        n *= np.sign(n @ up)
        if n @ up < np.cos(np.radians(15)):
            continue
        inl = np.abs((below - s[0]) @ n) < args.floor_tolerance
        if best is None or inl.sum() > best.sum():
            best, best_n = inl, n
    fl = below[best]
    centroid = fl.mean(0)
    _, _, vt = np.linalg.svd(fl - centroid, full_matrices=False)
    n = vt[2] * np.sign(vt[2] @ best_n)
    # 3. rotation taking n to z, the first camera's view to +x
    first = sorted(images)[0]
    f0 = fwd[0] - (fwd[0] @ n) * n
    f0 /= np.linalg.norm(f0)
    R = np.stack([f0, np.cross(n, f0), n])                     # rows: new x, y, z in old coords
    origin = R @ centres[0]
    shift = np.array([-origin[0], -origin[1], -(R @ centroid)[2]])
    cam_height = float(np.median((centres @ R.T + shift)[:, 2]))
    out_model = os.path.join(args.out, 'sparse', '0')
    os.makedirs(out_model, exist_ok=True)
    with open(os.path.join(args.model, 'cameras.txt'), encoding='utf-8') as f, \
            open(os.path.join(out_model, 'cameras.txt'), 'w', encoding='utf-8') as g:
        g.write(f.read())
    with open(os.path.join(out_model, 'images.txt'), 'w', encoding='utf-8') as g:
        for i in sorted(images):
            im = images[i]
            Rcw = Rs[i] @ R.T
            c_new = R @ (-Rs[i].T @ im['t']) + shift
            q, t = rot2qvec(Rcw), -Rcw @ c_new
            g.write(f"{i} {' '.join(f'{v:.9f}' for v in q)} {' '.join(f'{v:.9f}' for v in t)} "
                    f"{im['cam']} {im['name']}\n{im['points']}\n")
    with open(os.path.join(out_model, 'points3D.txt'), 'w', encoding='utf-8') as g:
        for line in lines:
            if line.startswith('#') or not line.strip():
                g.write(line)
                continue
            f = line.split()
            x = R @ np.array([float(v) for v in f[1:4]]) + shift
            g.write(' '.join([f[0], *(f'{v:.6f}' for v in x), *f[4:]]) + '\n')
    src = args.images or os.path.abspath(os.path.join(args.model, '..', '..', 'images'))
    link = os.path.join(args.out, 'images')
    if not os.path.exists(link):
        os.symlink(os.path.abspath(src), link)
    info = {'rotation': R.tolist(), 'shift': shift.tolist(), 'camera_height_m': cam_height,
            'floor_inliers': int(best.sum()), 'points_below_cameras': int(len(below)),
            'up_vs_mean_camera_up_deg': float(np.degrees(np.arccos(np.clip(n @ up, -1, 1))))}
    with open(os.path.join(args.out, 'level.json'), 'w', encoding='utf-8') as g:
        json.dump(info, g, indent=2)
    print(f"camera height {cam_height:.2f} m above the floor; floor plane {best.sum()} of {len(below)} points "
          f"below the cameras; floor normal {info['up_vs_mean_camera_up_deg']:.1f} deg from the cameras' up")


if __name__ == '__main__':
    main()
