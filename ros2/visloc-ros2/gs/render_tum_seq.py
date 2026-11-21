#!/usr/bin/env python3
"""Render a TUM RGB-D sequence from a splat along a trajectory (docs/navigation.md).

    python3 ros2/visloc-ros2/gs/render_tum_seq.py --splat runs/gs_tum/scene.ply \
        --trajectory runs/gs_tum/metric/slam.tum --out runs/gs_tum/render_slam \
        --camera "FULL_OPENCV 640 480 520.9 521.0 325.1 249.7 0.2312 -0.7849 -0.0033 -0.0001 0.9172 0 0 0"

For each pose of the trajectory (TUM: time, camera centre, world <- camera
quaternion x y z w, the optical frame), `gsplat_render_server --depth`
renders the colour image and the median depth of the splat. They are written
as a TUM sequence (`rgb/`, `depth/` as 16-bit PNG at 5000 per metre, `rgb.txt`,
`depth.txt`), so `tum_rgbd_save_map --keep-world` builds a localization map of
the splat itself: what a robot would map if it drove through the simulator.
"""

import argparse
import os
import subprocess

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))


def quat_to_rot(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--splat', required=True)
    p.add_argument('--trajectory', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--camera', required=True, help='COLMAP camera row without the id')
    p.add_argument('--server', default=os.path.join(REPO, 'target', 'release', 'examples', 'gsplat_render_server'))
    p.add_argument('--max-depth', type=float, default=10.0, help='deeper is written as 0 (no depth)')
    args = p.parse_args()
    from PIL import Image  # noqa: PLC0415
    w, h = (int(v) for v in args.camera.split()[1:3])
    os.makedirs(os.path.join(args.out, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'depth'), exist_ok=True)
    srv = subprocess.Popen([args.server, '--ply', args.splat, '--camera', args.camera, '--depth'],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    n = w * h
    rows = np.loadtxt(args.trajectory, ndmin=2)
    rgb_list, depth_list = [], []
    for r in rows:
        ts = f'{r[0]:.6f}'
        R_wc = quat_to_rot(*r[4:8])
        R_cw = R_wc.T
        t_cw = -R_cw @ r[1:4]
        srv.stdin.write((' '.join(f'{v:.7f}' for v in [*R_cw.reshape(-1), *t_cw]) + '\n').encode())
        srv.stdin.flush()
        buf = srv.stdout.read(n * 7)
        rgb = np.frombuffer(buf[:n * 3], np.uint8).reshape(h, w, 3)
        depth = np.frombuffer(buf[n * 3:], '<f4').reshape(h, w)
        d16 = np.where((depth > 0) & (depth < args.max_depth), depth * 5000, 0).astype(np.uint16)
        Image.fromarray(rgb).save(os.path.join(args.out, 'rgb', ts + '.png'))
        Image.fromarray(d16).save(os.path.join(args.out, 'depth', ts + '.png'))
        rgb_list.append(f'{ts} rgb/{ts}.png\n')
        depth_list.append(f'{ts} depth/{ts}.png\n')
    srv.stdin.close()
    srv.wait()
    for name, lines in (('rgb.txt', rgb_list), ('depth.txt', depth_list)):
        with open(os.path.join(args.out, name), 'w', encoding='utf-8') as f:
            f.write('# rendered by render_tum_seq.py\n# timestamp filename\n# \n')
            f.writelines(lines)
    print(f'rendered {len(rows)} RGB-D frames into {args.out}')


if __name__ == '__main__':
    main()
