#!/usr/bin/env python3
"""Copy a EuRoC-layout replay with its camera images changed, to test localization under appearance change.

    python3 ros2/visloc-ros2/gs/perturb_replay.py --replay runs/street/replay --out runs/street/replay_night \
        --kind night [--strength 1.0] [--seed 0]

Kinds (each applied to every image of mav0/cam0/data):

- `dark`: intensity times `1 - 0.85 * strength` (an overcast evening, a
  dim street);
- `gamma`: gamma `1 + 1.5 * strength` (deeper shadows, brighter highlights
  stay);
- `contrast`: contrast towards mid-grey by `1 - 0.7 * strength`;
- `noise`: Gaussian noise of `25 * strength` grey levels (a cheap sensor in
  low light);
- `blur`: Gaussian blur of sigma `3 * strength` px (motion, focus);
- `night`: dark and gamma together, with noise and a little blur, as a camera
  sees a street at night. It is lit evenly, though, with no lamps or shadows
  of its own, so it is a lower bound on how hard real night is;
- `patches`: rectangles over `40 * strength` percent of the image replaced
  by other images' content (changed signs, posters, parked vehicles), the
  same rectangles for an image wherever the scene repeats.

The odometry and the image lists are linked unchanged.
"""

import argparse
import os
import shutil

import numpy as np
from PIL import Image, ImageFilter


def perturb(a, kind, s, rng, donors):
    x = a.astype(np.float32)
    if kind == 'dark':
        x *= 1 - 0.85 * s
    elif kind == 'gamma':
        x = 255 * (x / 255) ** (1 + 1.5 * s)
    elif kind == 'contrast':
        x = 128 + (x - 128) * (1 - 0.7 * s)
    elif kind == 'noise':
        x += rng.normal(0, 25 * s, x.shape)
    elif kind == 'blur':
        x = np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(3 * s)), np.float32)
    elif kind == 'night':
        x = 255 * (x / 255) ** (1 + 1.2 * s) * (1 - 0.8 * s)
        x = np.asarray(Image.fromarray(np.clip(x, 0, 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(0.8 * s)), np.float32)
        x += rng.normal(0, 6 * s, x.shape)
    elif kind == 'patches':
        h, w = a.shape
        target = 0.4 * s * h * w
        covered = 0
        while covered < target:
            ph, pw = rng.integers(h // 12, h // 4), rng.integers(w // 12, w // 4)
            y0, x0 = rng.integers(0, h - ph), rng.integers(0, w - pw)
            d = donors[rng.integers(len(donors))]
            dy, dx = rng.integers(0, h - ph), rng.integers(0, w - pw)
            x[y0:y0 + ph, x0:x0 + pw] = d[dy:dy + ph, dx:dx + pw]
            covered += ph * pw
    else:
        raise SystemExit(f'unknown kind {kind}')
    return np.clip(x, 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--replay', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--kind', required=True)
    p.add_argument('--strength', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    src = os.path.join(args.replay, 'mav0', 'cam0')
    dst = os.path.join(args.out, 'mav0', 'cam0', 'data')
    os.makedirs(dst, exist_ok=True)
    shutil.copy(os.path.join(src, 'data.csv'), os.path.join(args.out, 'mav0', 'cam0', 'data.csv'))
    for name in os.listdir(args.replay):
        if name != 'mav0' and not os.path.exists(os.path.join(args.out, name)):
            os.symlink(os.path.abspath(os.path.join(args.replay, name)), os.path.join(args.out, name))
    names = sorted(os.listdir(os.path.join(src, 'data')))
    donors = []
    if args.kind == 'patches':
        for n in rng.choice(names, size=min(40, len(names)), replace=False):
            donors.append(np.asarray(Image.open(os.path.join(src, 'data', n)).convert('L')))
    for n in names:
        a = np.asarray(Image.open(os.path.join(src, 'data', n)).convert('L'))
        Image.fromarray(perturb(a, args.kind, args.strength, rng, donors)).save(os.path.join(dst, n))
    print(f'{len(names)} images ({args.kind}, strength {args.strength}) in {args.out}')


if __name__ == '__main__':
    main()
