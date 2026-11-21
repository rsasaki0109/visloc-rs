#!/usr/bin/env python3
"""Undistort TUM RGB-D fr2 pioneer frames for SfM, or crop a metric model for training.

    python3 ros2/visloc-ros2/gs/prep_tum_images.py undistort --data DIR --out DIR [--step 5]
    python3 ros2/visloc-ros2/gs/prep_tum_images.py crop --model DIR --images DIR --out DIR [--margin 20]

`undistort` takes every `--step`-th RGB frame of the pioneer_360,
pioneer_slam and pioneer_slam3 sequences under `--data`, undistorts it with
the Kinect's fr2 calibration (the same K out, so `PINHOLE 640 480 520.9 521.0
325.1 249.7`) and writes `<sequence>_<timestamp>.png`, the names
`align_sfm_tum.py` reads.

`crop` writes a training dataset (`OUT/sparse/0`, `OUT/images`) from a model
(`align_sfm_tum.py`'s metric one) and its images, `--margin` pixels cut off
each side so no undistortion border is left, with the principal point moved
to match. Needs OpenCV (`cv2`).
"""

import argparse
import os
import shutil

import cv2
import numpy as np

K = np.array([[520.9, 0, 325.1], [0, 521.0, 249.7], [0, 0, 1]])
DIST = np.array([0.2312, -0.7849, -0.0033, -0.0001, 0.9172])


def undistort(args):
    os.makedirs(args.out, exist_ok=True)
    m1, m2 = cv2.initUndistortRectifyMap(K, DIST, None, K, (640, 480), cv2.CV_32FC1)
    n = 0
    for seq in ('360', 'slam', 'slam3'):
        d = os.path.join(args.data, f'rgbd_dataset_freiburg2_pioneer_{seq}', 'rgb')
        for f in sorted(os.listdir(d))[::args.step]:
            im = cv2.imread(os.path.join(d, f))
            cv2.imwrite(os.path.join(args.out, f'{seq}_{f[:-4]}.png'), cv2.remap(im, m1, m2, cv2.INTER_LINEAR))
            n += 1
    print(f'{n} undistorted images in {args.out}')


def crop(args):
    m = args.margin
    out_model = os.path.join(args.out, 'sparse', '0')
    out_images = os.path.join(args.out, 'images')
    os.makedirs(out_model, exist_ok=True)
    os.makedirs(out_images, exist_ok=True)
    for name in ('images.txt', 'points3D.txt'):
        shutil.copy(os.path.join(args.model, name), out_model)
    rows = [r for r in open(os.path.join(args.model, 'cameras.txt'), encoding='utf-8') if r.strip() and not r.startswith('#')]
    with open(os.path.join(out_model, 'cameras.txt'), 'w', encoding='utf-8') as f:
        for r in rows:
            c = r.split()
            if c[1] != 'PINHOLE':
                raise SystemExit(f'crop expects PINHOLE cameras, got {c[1]}')
            w, h, fx, fy, cx, cy = int(c[2]), int(c[3]), *map(float, c[4:8])
            f.write(f'{c[0]} PINHOLE {w - 2 * m} {h - 2 * m} {fx} {fy} {cx - m} {cy - m}\n')
    n = 0
    for name in os.listdir(args.images):
        im = cv2.imread(os.path.join(args.images, name))
        cv2.imwrite(os.path.join(out_images, name), im[m:im.shape[0] - m, m:im.shape[1] - m])
        n += 1
    print(f'{n} images cropped by {m} px into {args.out}')


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    u = sub.add_parser('undistort')
    u.add_argument('--data', required=True, help='folder with the rgbd_dataset_freiburg2_pioneer_* sequences')
    u.add_argument('--out', required=True)
    u.add_argument('--step', type=int, default=5)
    c = sub.add_parser('crop')
    c.add_argument('--model', required=True)
    c.add_argument('--images', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--margin', type=int, default=20)
    args = p.parse_args()
    (undistort if args.cmd == 'undistort' else crop)(args)


if __name__ == '__main__':
    main()
