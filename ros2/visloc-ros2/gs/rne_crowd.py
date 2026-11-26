#!/usr/bin/env python3
"""Write a street's crowd for Robot Native Engine: who starts where, and what they look like.

    python3 ros2/visloc-ros2/gs/rne_crowd.py --scene runs/street/sim/scene.json \
        --people-json runs/street/sim/nav.sim.csv.people.json --out runs/street/rne \
        [--pedestrians 4 --cyclists 2 --dog-walkers 1 --strollers 1 --cars 3 --seed 0 --crossing]

The bundle's `crowd.json` and `crowd/*.ply` (rne_bundle.py writes the rest of the bundle).
RNE moves the crowd itself (example 163's port of `people.Crowd.step`). This file gives it the
same start as `gs_sim.py` with the same seed: `people.Crowd` is built here and its state written
after its own start-up (nobody placed inside anybody). It also gives the looks: every agent's
gaussians at each of `PHASES` gait phases (a car's: its wheel turn and its shadow), as 3DGS PLYs
in the agent's own frame (x forward, y left, z up, on the ground at 0), each padded with
transparent gaussians to the agent's largest phase so RNE can swap phases in place. Then the
crossing's stripes and light at each light state, in the map frame.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from people import (FOOT, LANE_HALF, NARROW, PHASES, RADIUS, Car, Crowd,  # noqa: E402
                    mesh_splats)

C0 = 0.28209479177387814           # degree-0 spherical harmonic
PROPS = (['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2'] + [f'f_rest_{i}' for i in range(45)]
         + ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'])


class Grid:
    """gs_sim.py's collision grid (scene.json's 'grid'): solid cells, outside is solid."""

    def __init__(self, path, x0, y0, res):
        self.g = np.load(path).astype(bool)
        self.x0, self.y0, self.res = x0, y0, res

    def occupied(self, x, y):
        i, j = int((x - self.x0) / self.res), int((y - self.y0) / self.res)
        return not (0 <= j < self.g.shape[0] and 0 <= i < self.g.shape[1]) or bool(self.g[j, i])

    def hits(self, x, y, r):
        n = max(8, int(2 * math.pi * r / self.res))
        return self.occupied(x, y) or any(
            self.occupied(x + r * math.cos(2 * math.pi * k / n), y + r * math.sin(2 * math.pi * k / n))
            for k in range(n))


def write_ply(path, rows, pad_to=None):
    """people.mesh_splats rows (x y z, log scales, quaternion w x y z, opacity logit, r g b) as a
    3DGS PLY, padded with transparent gaussians to pad_to."""
    rows = np.asarray(rows, np.float32)
    n = len(rows) if pad_to is None else pad_to
    a = np.zeros((n, len(PROPS)), np.float32)
    k = len(rows)
    a[:k, 0:3] = rows[:, 0:3]
    a[:k, 6:9] = (rows[:, 11:14] - 0.5) / C0
    a[:k, 54] = rows[:, 10]
    a[:k, 55:58] = rows[:, 3:6]
    a[:k, 58:62] = rows[:, 6:10]
    a[k:, 54] = -20.0                       # padding: transparent, tiny, at the origin
    a[k:, 55:58] = math.log(1e-4)
    a[k:, 58] = 1.0
    with open(path, 'wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\n' % n
                 + ''.join(f'property float {p}\n' for p in PROPS) + 'end_header\n').encode())
        a.tofile(f)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--scene', required=True)
    p.add_argument('--people-json', dest='people_json', required=True,
                   help="a gs_sim.py run's PREFIX.sim.csv.people.json (road, pavement, walk)")
    p.add_argument('--out', required=True, help='the street bundle directory')
    p.add_argument('--pedestrians', type=int, default=4)
    p.add_argument('--cyclists', type=int, default=2)
    p.add_argument('--dog-walkers', dest='dog_walkers', type=int, default=1)
    p.add_argument('--strollers', type=int, default=1)
    p.add_argument('--cars', type=int, default=3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-crossing', dest='crossing', action='store_false')
    args = p.parse_args()
    sdir = os.path.dirname(os.path.abspath(args.scene))
    scene = json.load(open(args.scene))
    sp = lambda f: f if os.path.isabs(f) else os.path.join(sdir, f)  # noqa: E731
    meta = json.load(open(args.people_json))
    g = scene['grid']
    grid = Grid(sp(g['file']), g['x0'], g['y0'], g['resolution'])
    route = scene.get('route') or [scene['start']] + scene.get('goals', [])
    road = meta.get('road_offset')
    road = [tuple(r) for r in road] if isinstance(road, list) else road
    crowd = Crowd(route, args.pedestrians, grid, seed=args.seed, cyclists=args.cyclists,
                  dog_walkers=args.dog_walkers, strollers=args.strollers, cars=args.cars,
                  road_offset=road, road_lines=meta.get('road_lines'),
                  crossing=args.crossing and bool(meta.get('crossing', True)),
                  pavement=meta.get('pavement'), walk=meta.get('walk'), narrow=meta.get('narrow'))
    os.makedirs(os.path.join(args.out, 'crowd'), exist_ok=True)
    agents = []
    for i, fig in enumerate(crowd.figures):
        if isinstance(fig, Car):
            frames = [np.concatenate([mesh_splats(m, fill=0.5), fig.shadow]) for m in fig.frames]
        else:
            frames = [mesh_splats(m) for m in fig.frames]
        most = max(len(f) for f in frames)
        files = []
        for k, rows in enumerate(frames):
            name = f'crowd/agent{i:02d}_phase{k:02d}.ply'
            write_ply(os.path.join(args.out, name), rows, pad_to=most)
            files.append(name)
        agents.append({'kind': int(crowd.kind[i]), 's': float(crowd.s[i]), 'dir': float(crowd.dir[i]),
                       'v': float(crowd.v[i]), 'side': float(crowd.side[i]), 'phase': float(crowd.phase[i]),
                       'foot': [list(map(float, f)) for f in crowd.foot[i]], 'phases': files})
    decor = {}
    if crowd.crossing_s is not None:
        gz = np.load(sp(scene['ground'])) if scene.get('ground') else None

        def ground_z(x, y):
            if gz is None:
                return 0.0
            r = min(max(int((y - g['y0']) / g['resolution']), 0), gz.shape[0] - 1)
            c = min(max(int((x - g['x0']) / g['resolution']), 0), gz.shape[1] - 1)
            return float(gz[r, c])
        for state in ('green', 'amber', 'red'):
            crowd.clock = {'green': 0.0, 'amber': 15.0, 'red': 18.0}[state]
            rows = crowd.decor(ground_z)
            name = f'crowd/crossing_{state}.ply'
            write_ply(os.path.join(args.out, name), rows)
            decor[state] = name
        crowd.clock = 0.0
    out = {'format': 'visloc_rs.crowd', 'version': 1, 'seed': args.seed,
           'phases': PHASES, 'radius_m': RADIUS, 'lane_half_m': LANE_HALF, 'narrow_m': NARROW,
           'yield_m': crowd.yield_m, 'route': np.asarray(route, float)[:, :2].tolist(),
           'pavement': None if crowd.pavement is None else crowd.pavement.tolist(),
           'walk_offsets': None if crowd.walk is None else crowd.walk.tolist(),
           'kerb_spans': [list(r) for r in crowd.narrow],
           'crossing_s': crowd.crossing_s, 'decor': decor, 'agents': agents,
           'default_foot': {str(k): [list(f) for f in v] for k, v in FOOT.items()}}
    if args.cars:
        out.update({'road': [list(r) for r in crowd.road], 'road_s': list(crowd.road_s),
                    'road_lines': crowd.road_lines,
                    'lanes': {'1': crowd.lanes[1.0].tolist(), '-1': crowd.lanes[-1.0].tolist()}})
    with open(os.path.join(args.out, 'crowd.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    kinds = np.bincount(crowd.kind, minlength=5)
    print(f'crowd -> {args.out}: {kinds[0]} walkers, {kinds[1]} cyclists, {kinds[2]} dog walkers, '
          f'{kinds[3]} strollers, {kinds[4]} cars, {PHASES} phases each'
          + (f', crossing at s {crowd.crossing_s:.1f}' if crowd.crossing_s is not None else ''))


if __name__ == '__main__':
    main()
