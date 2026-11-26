#!/usr/bin/env python3
"""Checks rne_bundle.py on a small synthetic street: python3 ros2/visloc-ros2/gs/rne_bundle_test.py"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    from PIL import Image  # noqa: PLC0415
    d = tempfile.mkdtemp(prefix='rne_bundle_test_')
    solid = np.zeros((20, 30), np.uint8)
    solid[5, 7] = 1                                    # one post at x = 0.35 + 0.025, y = 0.25 + 0.025
    np.save(os.path.join(d, 'grid.npy'), solid)
    np.save(os.path.join(d, 'ground.npy'), np.full((20, 30), -0.1, np.float32))
    pgm = np.full((10, 20), 25, np.uint8)              # off the lane: occupancy 0.90
    pgm[4:6, :] = 254                                  # the lane, two rows near the middle
    pgm[0, 0] = 0                                      # a wall, top left (the highest y)
    Image.fromarray(pgm).save(os.path.join(d, 'lane.pgm'))
    with open(os.path.join(d, 'lane.yaml'), 'w', encoding='utf-8') as f:
        f.write('image: lane.pgm\nmode: scale\nresolution: 0.1\norigin: [-1.0, -0.5, 0.0]\nnegate: 0\n')
    open(os.path.join(d, 'scene.ply'), 'w').close()
    os.makedirs(os.path.join(d, 'map'))
    scene = {'splat': 'scene.ply', 'grid': {'file': 'grid.npy', 'x0': 0.0, 'y0': 0.0, 'resolution': 0.05},
             'ground': 'ground.npy', 'start': [0.0, 0.0, 0.0], 'goals': [[1.0, 0.0, 0.0]],
             'walk': [[0.0, 0.0], [1.0, 0.0]], 'kerbs': [], 'robot_radius': 0.2, 'camera_height': 1.2}
    json.dump(scene, open(os.path.join(d, 'scene.json'), 'w'))
    people = {'road_offset': [[0, 1, 3.0]], 'road_lines': None, 'pavement': [[0.0, 1.0, 1.0]],
              'walk': [[0.0, 0.0], [1.0, 0.0]], 'narrow': [[0.2, 0.4]], 'crossing': True}
    json.dump(people, open(os.path.join(d, 'people.json'), 'w'))
    out = os.path.join(d, 'out')
    subprocess.run([sys.executable, os.path.join(HERE, 'rne_bundle.py'), '--scene', os.path.join(d, 'scene.json'),
                    '--occupancy', os.path.join(d, 'lane.yaml'), '--map', os.path.join(d, 'map'),
                    '--people-json', os.path.join(d, 'people.json'), '--out', out], check=True)

    def load(name):
        m = json.load(open(os.path.join(out, name)))
        assert m['format'] == 'rne.map' and m['version'] == 1
        lo = np.array(m['log_odds'], float).reshape(m['height'], m['width'])
        return m, 1 / (1 + np.exp(-lo / 1000.0))

    m, p = load('solid.rne.map')
    assert (m['width'], m['height']) == (30, 20)
    assert abs(m['origin']['x_m'] - 0.025) < 1e-9 and abs(m['origin']['y_m'] - 0.025) < 1e-9
    assert p[5, 7] > 0.99 and (p > 0.5).sum() == 1         # row j is y: the post stays where it was
    m, p = load('lane.rne.map')
    assert (m['width'], m['height']) == (20, 10)
    assert p[9, 0] > 0.99                                  # PGM row 0 (top) is the last row here
    assert (p[4:6] < 0.01).all()                           # the lane is free (PGM rows 4-5 -> rows 4-5)
    assert abs(p[0, 5] - 230 / 255) < 0.01                 # off the lane stays 0.90, costly not lethal
    street = json.load(open(os.path.join(out, 'street.json')))
    assert street['kerb_spans'] == [[0.2, 0.4]] and street['crossing'] is True
    assert street['credit'].startswith('Map data: OverMaps360')
    g = np.fromfile(os.path.join(out, 'ground.f32'), '<f4').reshape(street['ground']['height'],
                                                                    street['ground']['width'])
    assert g.shape == (20, 30) and np.allclose(g, -0.1)
    toml = open(os.path.join(out, 'street.rne.splat.toml')).read()
    assert 'rotation_xyzw = [-0.7071067811865476, 0.0, 0.0, 0.7071067811865476]' in toml
    print('rne_bundle_test: ok')


if __name__ == '__main__':
    main()
