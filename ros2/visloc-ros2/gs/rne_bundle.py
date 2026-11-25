#!/usr/bin/env python3
"""Write a street bundle: a real2sim street for Robot Native Engine's simulator (docs/rne_migration.md).

    python3 ros2/visloc-ros2/gs/rne_bundle.py --scene runs/street/sim/scene.json \
        --occupancy runs/street/map/occupancy_lane.yaml --map runs/street/map \
        [--people-json runs/street/sim/nav.sim.csv.people.json] --out runs/street/rne

visloc-rs builds the street; RNE simulates it. This writes, once per street,
everything RNE needs and nothing it would have to work out from the splat's
colours. All coordinates stay in the visloc-rs map frame (z up, metres).

- `street.rne.splat.toml`: RNE's splat manifest. `rotation_xyzw` turns the
  z-up splat into RNE's y-up world: RNE (x, y, z) = map (x, z, -y).
- `street.json`: start, goals, robot radius, camera, the walk, the kerbs,
  and, from a run's `people.json` (`--people-json`), the road beside the
  route, the road's centre lines, the pavement edges, the walk's offsets
  and the kerb spans (see people.py). The crowd's route is start then
  goals.
- `solid.rne.map`: the sim's solid cells (scene.json's grid: walls, posts,
  trunks), as an `rne_nav` occupancy map (`rne.map` JSON, log-odds x 1000,
  row 0 at the bottom).
- `lane.rne.map`: Nav2's map (lane_map.py's): occupancy probability as in
  its PGM, so free on the lane, 0.90 off it (costly, not lethal), 1 at walls.
- `ground.f32` and its shape in `street.json`: the ground height grid
  (float32, little-endian, row j = y0 + j * resolution).
- `robot.ply`: the delivery robot drawn in the videos (robot_model.py), as gaussians in its
  own frame (x forward, y left, z up, on the ground at 0), for the chase view.
- `map`: a link to the visloc-rs localization map, and `CREDIT.txt`.
"""

import argparse
import json
import math
import os

import numpy as np

LOG_ODDS_SCALE = 1000.0          # rne_nav's fixed point
CREDIT = 'Map data: OverMaps360 (OverTheReality), CC BY-NC 4.0.'
# -90 degrees about x: map (x, y, z) -> RNE (x, z, -y)
Z_UP_TO_Y_UP_XYZW = [-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]


def rne_map(prob, x0, y0, resolution):
    """`rne.map` JSON text for a (rows, cols) occupancy probability grid, row 0 at y0."""
    p = np.clip(np.asarray(prob, float), 0.001, 0.999)
    lo = np.round(np.log(p / (1 - p)) * LOG_ODDS_SCALE).astype(int)
    h, w = p.shape
    # rne_nav: cell (0, 0)'s centre at the origin
    origin = {'x_m': float(x0 + resolution / 2), 'y_m': float(y0 + resolution / 2), 'yaw_rad': 0.0}
    return json.dumps({'format': 'rne.map', 'version': 1, 'width': int(w), 'height': int(h),
                       'resolution_m': float(resolution), 'origin': origin,
                       'log_odds': lo.ravel().tolist()}, separators=(',', ':'))


def read_pgm_map(yaml_path):
    """(probability grid with row 0 at the bottom, x0, y0, resolution) of a ROS map yaml + PGM."""
    from PIL import Image  # noqa: PLC0415
    meta = {}
    for line in open(yaml_path, encoding='utf-8'):
        if ':' in line:
            k, v = line.split(':', 1)
            meta[k.strip()] = v.strip()
    img = np.array(Image.open(os.path.join(os.path.dirname(yaml_path), meta['image'])), float)
    if int(float(meta.get('negate', 0))):
        img = 255 - img
    prob = (255.0 - img) / 255.0               # ROS: white free, black occupied
    origin = [float(v) for v in meta['origin'].strip('[]').split(',')[:2]]
    return prob[::-1], origin[0], origin[1], float(meta['resolution'])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--scene', required=True, help="gs_sim.py's scene.json")
    p.add_argument('--occupancy', required=True, help="Nav2's map yaml (lane_map.py's)")
    p.add_argument('--map', required=True, help='the visloc-rs localization map directory')
    p.add_argument('--people-json', dest='people_json', default='',
                   help="a run's PREFIX.sim.csv.people.json: the road, pavement and walk gs_sim.py worked out")
    p.add_argument('--out', required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sdir = os.path.dirname(os.path.abspath(args.scene))
    scene = json.load(open(args.scene))
    sp = lambda f: f if os.path.isabs(f) else os.path.normpath(os.path.join(sdir, f))  # noqa: E731

    ply = sp(scene['splat'])
    with open(os.path.join(args.out, 'street.rne.splat.toml'), 'w', encoding='utf-8') as f:
        f.write('kind = "rne_gaussian_splat_environment"\nschema_version = 1\n'
                f'environment_id = "visloc_rs.real2sim.{os.path.basename(os.path.abspath(args.out))}"\n'
                f'ply_path = "{os.path.abspath(ply)}"\ntranslation_m = [0.0, 0.0, 0.0]\n'
                f'rotation_xyzw = [{", ".join(repr(v) for v in Z_UP_TO_Y_UP_XYZW)}]\nscale = 1.0\n'
                'renderer_identity = "rne.gaussian_splat.v1"\n'
                f'coordinate_note = "visloc-rs real2sim street, z-up map frame turned into RNE y-up. {CREDIT}"\n')

    g = scene['grid']
    solid = np.load(sp(g['file'])).astype(float)           # row j = y
    with open(os.path.join(args.out, 'solid.rne.map'), 'w', encoding='utf-8') as f:
        f.write(rne_map(np.where(solid > 0, 0.999, 0.001), g['x0'], g['y0'], g['resolution']))
    prob, lx0, ly0, lres = read_pgm_map(args.occupancy)
    with open(os.path.join(args.out, 'lane.rne.map'), 'w', encoding='utf-8') as f:
        f.write(rne_map(prob, lx0, ly0, lres))

    street = {'format': 'visloc_rs.street', 'version': 1, 'frame': 'visloc-rs map, z up, metres',
              'credit': CREDIT,
              'start': scene['start'], 'goals': scene.get('goals', []),
              'robot_radius_m': scene.get('robot_radius', 0.2),
              'camera': scene.get('camera'), 'camera_height_m': scene.get('camera_height'),
              'camera_forward_m': scene.get('camera_forward', 0.0),
              'walk': scene.get('walk'), 'kerbs': scene.get('kerbs', [])}
    if scene.get('ground'):
        ground = np.load(sp(scene['ground'])).astype('<f4')
        ground.tofile(os.path.join(args.out, 'ground.f32'))
        street['ground'] = {'file': 'ground.f32', 'x0': g['x0'], 'y0': g['y0'], 'resolution_m': g['resolution'],
                            'width': int(ground.shape[1]), 'height': int(ground.shape[0])}
    if args.people_json:
        meta = json.load(open(args.people_json))
        street.update({'road_offset': meta.get('road_offset'), 'road_lines': meta.get('road_lines'),
                       'pavement': meta.get('pavement'), 'walk_offsets': meta.get('walk'),
                       'kerb_spans': meta.get('narrow'), 'crossing': bool(meta.get('crossing'))})
    import robot_model  # noqa: PLC0415
    from people import mesh_splats  # noqa: PLC0415
    from rne_crowd import write_ply  # noqa: PLC0415
    write_ply(os.path.join(args.out, 'robot.ply'),
              mesh_splats(robot_model.Model(camera_height=scene.get('camera_height', 1.2))))
    street['robot_actor'] = 'robot.ply'
    with open(os.path.join(args.out, 'street.json'), 'w', encoding='utf-8') as f:
        json.dump(street, f, indent=1)

    link = os.path.join(args.out, 'map')
    if os.path.islink(link):
        os.remove(link)
    if not os.path.exists(link):
        os.symlink(os.path.abspath(args.map), link)
    with open(os.path.join(args.out, 'CREDIT.txt'), 'w', encoding='utf-8') as f:
        f.write(CREDIT + '\nThe splat, the maps and the street data come from that footage.\n')
    print(f'street bundle -> {args.out}: {solid.shape[1]}x{solid.shape[0]} solid cells, '
          f'lane map {prob.shape[1]}x{prob.shape[0]}, {len(street["goals"])} goals'
          + (', with the road and pavement' if args.people_json else ''))


if __name__ == '__main__':
    main()
