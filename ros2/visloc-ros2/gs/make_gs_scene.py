#!/usr/bin/env python3
"""Make gs_sim.py's scene from a splat trained in metres, z up (docs/navigation.md).

    python3 ros2/visloc-ros2/gs/make_gs_scene.py --splat runs/gs_tum/scene.ply \
        --model runs/gs_tum/metric/sparse/0 --out runs/gs_tum/scene.json \
        --camera "FULL_OPENCV 640 480 520.9 521.0 325.1 249.7 0.2312 -0.7849 -0.0033 -0.0001 0.9172 0 0 0" \
        --camera-height 0.55 --start -2.0,-0.6,1.2

The splat has no collision geometry, so the robot's world is made from it:

- **What the robot can hit**: Gaussians whose centres are between `--z-min`
  and `--z-max` above the floor, opaque enough (`--min-opacity`) and small
  enough (`--max-scale`) to be surfaces rather than haze, binned into a 2D
  grid; a cell needs `--min-count` of them. Isolated cells are dropped.
- **Where the splat is not to be trusted**: a splat only looks right near
  the photos it was trained on. Cells farther than `--fence-m` from every
  training camera are marked occupied too, a fence the robot (and its scan)
  sees as a wall, so it drives where its camera images mean something.

Writes the grid (`<out stem>_grid.npy`, 1 = occupied, row j = y) and the
scene JSON (splat, camera, grid frame, start pose, robot radius), plus a PNG
of the grid with the training cameras.
"""

import argparse
import json
import os

import numpy as np


def read_ply(path):
    with open(path, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('ascii').strip()
            header.append(line)
            if line == 'end_header':
                break
        n = next(int(h.split()[2]) for h in header if h.startswith('element vertex'))
        props = [h.split()[2] for h in header if h.startswith('property float')]
        if any(h.startswith('property') and not h.startswith('property float') for h in header):
            raise SystemExit('only float vertex properties are supported')
        data = np.frombuffer(f.read(n * 4 * len(props)), dtype='<f4').reshape(n, len(props))
    return {p: data[:, i] for i, p in enumerate(props)}


def camera_centres(model):
    centres = []
    rows = [line for line in open(os.path.join(model, 'images.txt'), encoding='utf-8')
            if line.strip() and not line.startswith('#')]
    for header in rows[0::2]:
        f = header.split()
        w, x, y, z = (float(v) for v in f[1:5])
        t = np.array([float(v) for v in f[5:8]])
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        centres.append(-R.T @ t)
    return np.array(centres)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--splat', required=True)
    p.add_argument('--model', required=True, help='the COLMAP model the splat was trained on (for the fence)')
    p.add_argument('--out', required=True, help='scene JSON to write')
    p.add_argument('--camera', required=True, help='COLMAP camera row without the id (the rendered camera)')
    p.add_argument('--camera-height', type=float, required=True, help='camera height above the floor, m')
    p.add_argument('--camera-forward', type=float, default=0.1, help='camera ahead of base_link, m')
    p.add_argument('--start', required=True, help='x,y,yaw of the robot at start')
    p.add_argument('--floor-z', type=float, default=0.0)
    p.add_argument('--z-min', type=float, default=0.06)
    p.add_argument('--z-max', type=float, default=0.8)
    p.add_argument('--min-opacity', type=float, default=0.5)
    p.add_argument('--max-scale', type=float, default=0.1, help='largest Gaussian axis, m')
    p.add_argument('--min-count', type=int, default=3)
    p.add_argument('--fence-m', type=float, default=1.5)
    p.add_argument('--resolution', type=float, default=0.05)
    p.add_argument('--robot-radius', type=float, default=0.2)
    args = p.parse_args()
    from scipy import ndimage  # noqa: PLC0415

    g = read_ply(args.splat)
    xyz = np.stack([g['x'], g['y'], g['z']], 1)
    opacity = 1 / (1 + np.exp(-g['opacity']))
    scale = np.exp(np.stack([g['scale_0'], g['scale_1'], g['scale_2']], 1)).max(1)
    h = xyz[:, 2] - args.floor_z
    solid = (h >= args.z_min) & (h <= args.z_max) & (opacity >= args.min_opacity) & (scale <= args.max_scale)
    cams = camera_centres(args.model)
    res = args.resolution
    x0, y0 = np.floor(cams[:, :2].min(0) - args.fence_m - 0.5)
    x1, y1 = np.ceil(cams[:, :2].max(0) + args.fence_m + 0.5)
    W, H = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    count = np.zeros((H, W), np.int32)
    i = ((xyz[solid, 0] - x0) / res).astype(int)
    j = ((xyz[solid, 1] - y0) / res).astype(int)
    ok = (i >= 0) & (i < W) & (j >= 0) & (j < H)
    np.add.at(count, (j[ok], i[ok]), 1)
    occ = count >= args.min_count
    # drop isolated cells (floaters), then close small gaps in surfaces
    occ &= ndimage.uniform_filter(occ.astype(float), 3) * 9 >= 2
    occ = ndimage.binary_closing(occ, iterations=1)
    # the fence: distance to the nearest training camera
    seen = np.zeros((H, W), bool)
    ci = ((cams[:, 0] - x0) / res).astype(int)
    cj = ((cams[:, 1] - y0) / res).astype(int)
    seen[cj.clip(0, H - 1), ci.clip(0, W - 1)] = True
    dist = ndimage.distance_transform_edt(~seen) * res
    fence = dist > args.fence_m
    grid = (occ | fence).astype(np.uint8)
    stem = os.path.splitext(args.out)[0]
    np.save(stem + '_grid.npy', grid)
    sx, sy, syaw = (float(v) for v in args.start.split(','))
    width, height = (int(v) for v in args.camera.split()[1:3])
    out_dir = os.path.dirname(os.path.abspath(args.out))
    scene = {
        'splat': os.path.relpath(os.path.abspath(args.splat), out_dir),
        'camera': {'model': args.camera, 'width': width, 'height': height},
        'camera_height': args.camera_height, 'camera_forward': args.camera_forward,
        'grid': {'file': os.path.basename(stem + '_grid.npy'), 'x0': float(x0), 'y0': float(y0),
                 'resolution': res},
        'start': [sx, sy, syaw], 'robot_radius': args.robot_radius,
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(scene, f, indent=2)
    try:
        from PIL import Image  # noqa: PLC0415
        rgb = np.full((H, W, 3), 245, np.uint8)
        rgb[fence] = (200, 200, 205)
        rgb[occ] = (40, 40, 40)
        rgb[cj.clip(0, H - 1), ci.clip(0, W - 1)] = (220, 60, 50)
        Image.fromarray(rgb[::-1]).resize((W * 3, H * 3), Image.NEAREST).save(stem + '_grid.png')
    except ImportError:
        pass
    print(f'grid {W}x{H} at {res} m from ({x0}, {y0}); {int(occ.sum())} obstacle cells from '
          f'{int(solid.sum())} of {len(xyz)} Gaussians; {int(fence.sum())} fence cells')
    if grid[int((sy - y0) / res), int((sx - x0) / res)]:
        print('warning: the start pose is in an occupied cell')


if __name__ == '__main__':
    main()
