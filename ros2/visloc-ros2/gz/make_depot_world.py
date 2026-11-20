#!/usr/bin/env python3
"""Put the Gazebo robot in the public Depot warehouse (docs/navigation.md).

    curl -L -o depot.zip https://fuel.gazebosim.org/1.0/OpenRobotics/models/Depot.zip
    unzip depot.zip -d runs/gz_depot/depot_model
    python3 ros2/visloc-ros2/gz/make_depot_world.py --depot runs/gz_depot/depot_model \
        --out runs/gz_depot/world [--rgb-twin] [--chase] [--gt-grid]

The Depot (OpenRobotics on Gazebo Fuel, CC BY 4.0) is a warehouse with PBR
materials: brick walls, steel racks, crates, a concrete floor. Its only
collision geometry is the floor, so the robot does not bump into anything:
`--gt-grid` projects the meshes (pycollada) into a 2D grid of what the robot
can hit (`gt_grid.npy`, 5 cm cells, with `scene.json`), which the route,
the goals and the contact count use. The robot is the one of make_world.py
(stereo, IMU, lidar; no wheel odometry for the visloc nodes).

Writes OUT/depot.sdf, OUT/calibration.json and OUT/scene.json (start, the
mapping route, the goals, the racks, the blackout line, the grid's frame).
`--rgb-twin` adds a colour camera at cam0's pose (cam0_rgb/image, 5 Hz)
for a 3DGS of the place; nothing in the visloc stack reads it. `--chase`
adds a camera following the robot (chase/image) for videos.
"""

import argparse
import json
import math
import os

from make_world import calibration, robot

# Racks (x0, y0, x1, y1): their shelves start above 1 m, so the mesh grid
# shows only their legs; the route and Nav2's goals keep out of them.
RACKS = [(1.4, 0.0, 2.7, 5.15), (5.3, 0.0, 6.55, 5.15), (9.05, 0.0, 10.3, 5.15)]
START = (-12.0, 0.0, 0.0)
# Every part of the hall, twice through the middle, back to the start.
ROUTE = [(-12.0, 0.0), (-12.0, 6.5), (0.0, 6.5), (4.0, 6.5), (4.0, -0.8), (8.4, -0.8),
         (8.4, 6.5), (14.0, 6.5), (14.5, 1.2), (14.0, -0.8), (14.0, -6.8), (-12.0, -6.8),
         (-12.0, -1.0), (-4.0, 0.5), (-4.0, 4.8), (-12.0, 4.8), (-12.0, 0.0)]
GOALS = [(-12.0, 6.0, 1.57), (4.0, 3.0, -1.57), (13.8, 6.0, 0.0), (13.8, -6.8, -1.57),
         (-11.0, -6.0, 3.14)]
# The blackout starts where the robot first crosses this x (on the way to goal 2).
BLACKOUT_X = -4.0
GRID = {'x0': -16.0, 'y0': -8.5, 'resolution': 0.05, 'z_min': 0.03, 'z_max': 0.6}

TWIN = '''<sensor name="cam0_rgb" type="camera">
          <pose>0.1 0.06 0 0 0 0</pose><always_on>1</always_on><update_rate>5</update_rate>
          <topic>cam0_rgb/image</topic><gz_frame_id>cam0_optical</gz_frame_id>
          <camera><horizontal_fov>1.5707963267948966</horizontal_fov>
            <image><width>640</width><height>480</height><format>R8G8B8</format></image>
            <clip><near>0.05</near><far>40</far></clip>
          </camera>
        </sensor>
        '''
CHASE = '''<sensor name="chase" type="camera">
          <pose>-1.6 0 1.9 0 0.55 0</pose><always_on>1</always_on><update_rate>10</update_rate>
          <topic>chase/image</topic><gz_frame_id>chase</gz_frame_id>
          <camera><horizontal_fov>1.3</horizontal_fov>
            <image><width>640</width><height>360</height><format>R8G8B8</format></image>
            <clip><near>0.1</near><far>60</far></clip>
          </camera>
        </sensor>
        '''


def gt_grid(depot, out):
    """Rasterize the meshes between z_min and z_max (metres) into GRID."""
    import collada  # noqa: PLC0415 (only for --gt-grid)
    import numpy as np  # noqa: PLC0415
    tris = []
    for name in ('Depot.dae', 'Crates.dae'):
        mesh = collada.Collada(os.path.join(depot, 'meshes', name), ignore=[
            collada.common.DaeUnsupportedError, collada.common.DaeBrokenRefError])
        for geom in mesh.scene.objects('geometry'):
            for prim in geom.primitives():
                t = prim.triangleset() if hasattr(prim, 'triangleset') else prim
                tris.append(t.vertex[t.vertex_index].reshape(-1, 3, 3))
    # The model scales the centimetre meshes by 0.6.
    tris = np.concatenate(tris) * 0.01 * 0.6
    lo, hi = GRID['z_min'], GRID['z_max']
    tris = tris[(tris[:, :, 2].max(1) >= lo) & (tris[:, :, 2].min(1) <= hi)]
    rng = np.random.default_rng(0)
    res = GRID['resolution']
    width, height = int(-2 * GRID['x0'] / res), int(-2 * GRID['y0'] / res)
    grid = np.zeros((height, width), np.uint8)
    for a, b, c in tris:
        n = int(min(20000, max(3, np.linalg.norm(np.cross(b - a, c - a)) / 2 / 0.02 ** 2)))
        u, v = rng.random(n), rng.random(n)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        p = a + u[:, None] * (b - a) + v[:, None] * (c - a)
        p = p[(p[:, 2] >= lo) & (p[:, 2] <= hi)]
        i = ((p[:, 0] - GRID['x0']) / res).astype(int)
        j = ((p[:, 1] - GRID['y0']) / res).astype(int)
        ok = (i >= 0) & (i < width) & (j >= 0) & (j < height)
        grid[j[ok], i[ok]] = 1
    np.save(os.path.join(out, 'gt_grid.npy'), grid)
    return int(grid.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--depot', required=True, help='the unzipped Fuel model (model.sdf, meshes/)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--rgb-twin', action='store_true')
    parser.add_argument('--chase', action='store_true')
    parser.add_argument('--gt-grid', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    depot = os.path.abspath(args.depot)
    bot = robot(*START)
    extra = (TWIN if args.rgb_twin else '') + (CHASE if args.chase else '')
    if extra:
        bot = bot.replace('<sensor name="cam1"', extra + '<sensor name="cam1"', 1)
    sdf = f'''<?xml version="1.0"?>
<!-- Generated by ros2/visloc-ros2/gz/make_depot_world.py; see docs/navigation.md. -->
<sdf version="1.9">
  <world name="depot">
    <physics name="1ms" type="ignored"><max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <scene><ambient>0.4 0.4 0.4 1</ambient><background>0.7 0.7 0.7 1</background><shadows>false</shadows></scene>
    <light type="directional" name="sun"><cast_shadows>false</cast_shadows><direction>-0.4 0.2 -0.9</direction>
      <diffuse>0.9 0.9 0.9 1</diffuse><specular>0.1 0.1 0.1 1</specular></light>
    <include><uri>{depot}</uri><name>depot</name><pose>0 0 0 0 0 0</pose></include>
{bot}
  </world>
</sdf>
'''
    with open(os.path.join(args.out, 'depot.sdf'), 'w', encoding='utf-8') as f:
        f.write(sdf)
    with open(os.path.join(args.out, 'calibration.json'), 'w', encoding='utf-8') as f:
        json.dump(calibration(), f, indent=2)
    scene = {'world': 'depot', 'start': START, 'route': ROUTE, 'goals': GOALS, 'racks': RACKS,
             'blackout_x': BLACKOUT_X, 'grid': dict(GRID, file='gt_grid.npy')}
    with open(os.path.join(args.out, 'scene.json'), 'w', encoding='utf-8') as f:
        json.dump(scene, f, indent=2)
    msg = f'wrote {args.out}/depot.sdf'
    if args.gt_grid:
        msg += f', gt_grid.npy ({gt_grid(depot, args.out)} occupied cells)'
    print(msg)


if __name__ == '__main__':
    main()
