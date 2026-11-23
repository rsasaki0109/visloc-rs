#!/usr/bin/env python3
"""Keep the robot to the walk: an occupancy map free only along the path the video was walked.

    python3 ros2/visloc-ros2/gs/lane_map.py --occupancy runs/street/map/occupancy.yaml \
        --walk runs/street/faces.tum --out runs/street/map/occupancy_lane [--half-width 1.0]

The 360-degree video was walked along a pavement, so the camera's track is
that pavement's line. Nav2 planning on the full occupancy map is free to cut
across the lawn or wander onto the road wherever the map is open. This map
keeps the cells within --half-width metres of the walk (unless the original
map has them occupied) and marks every other cell occupied, so the global
plan, and the robot, stay on the pavement. Cells near the walk that the
original map left unknown become free: someone walked there. Lone occupied
cells there (fewer than two occupied neighbours, stray map points) are
cleared too.

`--walk` is a TUM trajectory (`t x y z qx qy qz qw`, e.g. `real2sim.py`'s
`faces.tum`); only x and y are used. Where the walk wanders off and comes
back (a walker crossing a side street halfway down it rather than at its
mouth), the lane follows the straight line instead (`--detour-ratio`,
`--max-chord`); that walk is written to OUT_walk.txt, so goals can be put
on it. Writes OUT.pgm and OUT.yaml (the original's resolution and origin);
pass OUT.yaml to `gs_nav.launch.py` as `occupancy:=`.
"""

import argparse
import os

import numpy as np


def read_yaml(path):
    meta = {}
    for line in open(path, encoding='utf-8'):
        if ':' in line:
            k, v = line.split(':', 1)
            meta[k.strip()] = v.strip()
    return meta


def lane_mask(shape, res, origin, walk, half_width):
    """Boolean (rows, cols) of the cells within half_width of the walk polyline (row 0 at the top)."""
    h, w = shape
    xs = origin[0] + (np.arange(w) + 0.5) * res
    ys = origin[1] + (h - 1 - np.arange(h) + 0.5) * res
    near = np.zeros(shape, bool)
    for a, b in zip(walk[:-1], walk[1:]):
        lo = np.minimum(a, b) - half_width
        hi = np.maximum(a, b) + half_width
        ci = np.nonzero((xs >= lo[0]) & (xs <= hi[0]))[0]
        ri = np.nonzero((ys >= lo[1]) & (ys <= hi[1]))[0]
        if not len(ci) or not len(ri):
            continue
        X, Y = np.meshgrid(xs[ci], ys[ri])
        d = b - a
        L2 = max(float(d @ d), 1e-12)
        t = np.clip(((X - a[0]) * d[0] + (Y - a[1]) * d[1]) / L2, 0, 1)
        dist = np.hypot(X - (a[0] + t * d[0]), Y - (a[1] + t * d[1]))
        near[np.ix_(ri, ci)] |= dist <= half_width
    return near


def straighten(walk, blocked, ratio=1.6, max_chord=20.0, step=0.25):
    """The walk with its detours cut: wherever it wanders off and comes back (the arc between
    two points at least `ratio` times the straight line between them, the line at most
    `max_chord` m and clear of blocked(x, y) cells), the straight line instead. A walker
    crossing a side street halfway down it, or stepping round something, becomes a straight
    crossing at the street's mouth."""
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(walk, axis=0), axis=1))]
    out, i = [walk[0]], 0
    while i < len(walk) - 1:
        best = None
        for j in range(i + 2, len(walk)):
            if s[j] - s[i] > max_chord * ratio * 2:
                break
            chord = float(np.linalg.norm(walk[j] - walk[i]))
            if chord > max_chord or s[j] - s[i] < ratio * max(chord, 1e-6) or s[j] - s[i] < 3.0:
                continue
            n = max(2, int(chord / step))
            pts = walk[i] + np.linspace(0, 1, n)[:, None] * (walk[j] - walk[i])
            if not any(blocked(x, y) for x, y in pts):
                best = (j, pts)
        if best is None:
            out.append(walk[i + 1])
            i += 1
        else:
            j, pts = best
            out.extend(pts[1:])
            i = j
    return np.array(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--occupancy', required=True, help="the map's occupancy.yaml")
    p.add_argument('--walk', required=True, help='TUM trajectory of the walk')
    p.add_argument('--out', required=True, help='output prefix (writes OUT.pgm and OUT.yaml)')
    p.add_argument('--half-width', type=float, default=1.0, help='m either side of the walk kept free')
    p.add_argument('--detour-ratio', type=float, default=1.6,
                   help='cut a detour whose length is at least this many times the straight line (0: keep the walk)')
    p.add_argument('--max-chord', type=float, default=20.0, help='longest straight line that replaces a detour (m)')
    args = p.parse_args()
    from PIL import Image  # noqa: PLC0415
    meta = read_yaml(args.occupancy)
    img = np.array(Image.open(os.path.join(os.path.dirname(args.occupancy), meta['image'])))
    res = float(meta['resolution'])
    origin = [float(v) for v in meta['origin'].strip('[]').split(',')[:2]]
    walk = np.loadtxt(args.walk, ndmin=2)[:, 1:3]
    keep = np.r_[True, np.linalg.norm(np.diff(walk, axis=0), axis=1) > 0.05]   # drop repeats
    walk = walk[keep]
    negate = int(float(meta.get('negate', 0)))
    occupied = (img < 100) if not negate else (img > 155)
    if args.detour_ratio > 0:
        h = img.shape[0]

        def blocked(x, y):
            c = int((x - origin[0]) / res)
            r = h - 1 - int((y - origin[1]) / res)
            r0, r1, c0, c1 = max(r - 3, 0), min(r + 4, h), max(c - 3, 0), min(c + 4, img.shape[1])
            return occupied[r0:r1, c0:c1].sum() >= 3          # a wall within 15 cm, not a stray point
        n0 = len(walk)
        walk = straighten(walk, blocked, args.detour_ratio, args.max_chord)
        np.savetxt(args.out + '_walk.txt', walk, fmt='%.3f', header='x y: the walk with its detours cut')
        print(f'walk: {n0} points, detours cut -> {len(walk)} ({args.out}_walk.txt)')
    near = lane_mask(img.shape, res, origin, walk, args.half_width)
    # A lone occupied cell is a stray map point, not a wall: on a 2 m lane, inflated, it
    # would narrow the way for nothing.
    pad = np.pad(occupied, 1)
    neighbours = sum(np.roll(np.roll(pad, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1))[1:-1, 1:-1] - occupied
    lone = occupied & (neighbours < 2)
    occupied = occupied & ~(lone & near)
    out = np.where(near & ~occupied, 0 if negate else 254, 254 if negate else 0).astype(np.uint8)
    Image.fromarray(out).save(args.out + '.pgm')
    with open(args.out + '.yaml', 'w', encoding='utf-8') as f:
        f.write(f'image: {os.path.basename(args.out)}.pgm\nmode: trinary\nresolution: {res}\n'
                f'origin: [{origin[0]}, {origin[1]}, 0.0]\nnegate: {negate}\n'
                f"occupied_thresh: {meta.get('occupied_thresh', 0.65)}\nfree_thresh: {meta.get('free_thresh', 0.196)}\n")
    print(f'{int((near & ~occupied).sum() * res * res)} m^2 free along {len(walk)} walk points '
          f'(half width {args.half_width} m) -> {args.out}.yaml')


if __name__ == '__main__':
    main()
