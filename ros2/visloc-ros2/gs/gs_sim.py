#!/usr/bin/env python3
"""A robot simulator whose camera sees a 3D Gaussian splat (docs/navigation.md).

    python3 ros2/visloc-ros2/gs/gs_sim.py --scene runs/gs_tum/scene.json [--record run.csv]

The world is what a splat trained on real photos shows. The robot is a
differential drive driven by `cmd_vel`. It publishes:

- `odom` and `/tf` `odom -> base_link`: wheel odometry, with scale errors
  on the distance and the turn rate and a small yaw-rate bias, so it drifts
  as a real robot's does;
- `camera/image` (`sensor_msgs/Image`, rgb8, or mono8 with `--mono`): the splat rendered at the
  camera's pose by `gsplat_render_server`, at `camera_hz`;
- `/tf_static` `base_link -> camera_optical`;
- `scan` (`sensor_msgs/LaserScan`, frame `base_link`): a 2D lidar, ray-cast
  in the scene's collision grid (what the robot can hit, from the splat's
  mesh or from a ground-truth scan of the place);
- `ground_truth` (`nav_msgs/Odometry`, the true pose in the map frame) and
  `collisions` (`std_msgs/Int32`, steps a wall stopped the robot);
- with `--pedestrians N` (and `--cyclists`, `--dog-walkers`, `--strollers`),
  people walk (ride bicycles, walk dogs, push pushchairs) along the scene's
  route (its start and goals) back and forth (`people.py`). Walkers now and
  then stand still a while, and people on foot drift across to the other
  side. `--cars` drive both ways along a road beside the route, keeping
  right; the road is `--road-offset` metres to the left of the route, or
  estimated from the splat's grey ground (stretches whose renders are
  blurred are left out). `--crossing` adds a zebra crossing with a traffic
  light: cars wait unless green, walkers cross on red; They show in the camera and in the
  scan, so Nav2 has to steer round them, and they wait while the robot is
  just in front of them. In the camera they are gaussians rendered with the
  scene by default (`--people-style splat`, `gsplat_render_server
  --dynamic`), or meshes drawn over it by depth (`mesh`). `--record` also
  writes their positions to `<record>.people.csv`;
- `/clock`: the simulation owns time. Each step waits for the step's camera
  image, so a slow renderer slows the clock, not the robot: every node runs
  on simulation time, as in Gazebo.

`scene.json` (written by `make_gs_scene.py`) gives the splat, the camera
(`camera`: its COLMAP row without the id as `model`, e.g. `FULL_OPENCV 640 480
fx fy cx cy k1 k2 p1 p2 k3 0 0 0`, and its width and height; `camera_height` and
`camera_forward`: where it sits on base_link, metres),
the collision grid (`.npy`, 1 = occupied; its x0, y0, resolution), the
start pose, the robot's radius and optionally `ground` (a height map in the
grid's frame, for a sloping street). The splat is in the map's frame: metres,
z up, the floor at z = 0.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Int32
from tf2_msgs.msg import TFMessage

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
DT = 0.02  # physics step (s)


def stamp(t):
    ns = int(round(t * 1e9))
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def quat_yaw(yaw):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


class Grid:
    def __init__(self, path, x0, y0, res):
        self.g = np.load(path).astype(bool)
        self.x0, self.y0, self.res = x0, y0, res

    def occupied(self, x, y):
        i, j = int((x - self.x0) / self.res), int((y - self.y0) / self.res)
        if not (0 <= j < self.g.shape[0] and 0 <= i < self.g.shape[1]):
            return True
        return bool(self.g[j, i])

    def hits(self, x, y, r):
        n = max(8, int(2 * math.pi * r / self.res))
        return self.occupied(x, y) or any(
            self.occupied(x + r * math.cos(2 * math.pi * k / n), y + r * math.sin(2 * math.pi * k / n))
            for k in range(n))

    def scan(self, x, y, yaw, n, rmax):
        # March all rays together, one cell at a time.
        a = yaw + np.linspace(-math.pi, math.pi, n, endpoint=False)
        steps = np.arange(0.0, rmax, self.res * 0.5)
        px = x + np.cos(a)[:, None] * steps[None, :]
        py = y + np.sin(a)[:, None] * steps[None, :]
        i = ((px - self.x0) / self.res).astype(int)
        j = ((py - self.y0) / self.res).astype(int)
        inside = (i >= 0) & (i < self.g.shape[1]) & (j >= 0) & (j < self.g.shape[0])
        hit = np.ones_like(inside)
        hit[inside] = self.g[j[inside], i[inside]]
        first = np.where(hit.any(1), hit.argmax(1), -1)
        return np.where(first >= 0, steps[np.maximum(first, 0)], np.inf)


class Renderer:
    """`gsplat_render_server` behind a pipe: a pose in, an image out."""

    def __init__(self, server, ply, cam, depth=False, dynamic=0):
        self.w, self.h = cam['width'], cam['height']
        self.depth = depth
        self.p = subprocess.Popen([server, '--ply', ply, '--camera', cam['model']] + (['--depth'] if depth else [])
                                  + (['--dynamic', str(dynamic)] if dynamic else []),
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def render(self, R_cw, t_cw, splats=None):
        """splats: moving objects' gaussians (K x 14 float32) for --dynamic."""
        line = ' '.join(f'{v:.7f}' for v in list(R_cw.reshape(-1)) + list(t_cw)) + '\n'
        if splats is not None:
            self.p.stdin.write(f'dyn {len(splats)}\n'.encode() + np.ascontiguousarray(splats, '<f4').tobytes())
        self.p.stdin.write(line.encode())
        self.p.stdin.flush()
        n = self.w * self.h * (7 if self.depth else 3)
        buf = b''
        while len(buf) < n:
            chunk = self.p.stdout.read(n - len(buf))
            if not chunk:
                raise RuntimeError('gsplat_render_server exited')
            buf += chunk
        return buf


class Sim(Node):
    def __init__(self, args, scene):
        super().__init__('gs_sim')
        self.args, self.scene = args, scene
        cam = scene['camera']
        self.cam = cam
        here = os.path.dirname(os.path.abspath(args.scene))
        sp = lambda p: p if os.path.isabs(p) else os.path.join(here, p)  # noqa: E731
        # Splat people are rendered with the scene; mesh people are drawn over it by depth.
        movers = args.pedestrians + args.cyclists + args.dog_walkers + args.strollers + args.cars
        self.splat_people = movers > 0 and args.people_style == 'splat'
        self.renderer = Renderer(args.server, sp(scene['splat']), cam,
                                 depth=movers > 0 and not self.splat_people,
                                 dynamic=(3000 * args.pedestrians + 6000 * args.cyclists + 4500 * args.dog_walkers
                                          + 5500 * args.strollers + 7000 * args.cars) if self.splat_people else 0)
        g = scene['grid']
        self.grid = Grid(sp(g['file']), g['x0'], g['y0'], g['resolution'])
        self.ground = np.load(sp(scene['ground'])) if scene.get('ground') else None
        from kerbs import Kerbs  # noqa: PLC0415
        self.kerbs = Kerbs(scene.get('kerbs'))   # roads the robot crosses, between kerbs
        self.crowd = None
        if movers > 0:
            from people import Crowd  # noqa: PLC0415
            route = scene.get('route') or [scene['start']] + scene.get('goals', [])
            if len(route) < 2:
                raise SystemExit('--pedestrians needs a route: the scene has no goals')
            road_offset = args.road_offset
            if args.road:
                road_offset = [tuple(r) for r in json.load(open(args.road))]
            road_lines = None
            walk = None
            if scene.get('walk'):                 # the line the video was walked: people follow it
                from people import walk_offsets  # noqa: PLC0415
                walk = walk_offsets(route, scene['walk'])
                w = np.abs([v for _, v in walk])
                self.get_logger().info(f'people: along the walk, {w.max():.1f} m from the route at most')
            from people import kerb_spans  # noqa: PLC0415
            narrow = kerb_spans(route, scene.get('kerbs'))      # crossing a street: people keep to the walk
            if args.cars:
                from make_gs_scene import read_ply  # noqa: PLC0415
                ply = read_ply(sp(scene['splat']))
                xyz = np.stack([ply['x'], ply['y'], ply['z']], 1)
                opacity = 1 / (1 + np.exp(-ply['opacity']))
                rgb = np.clip(np.stack([ply['f_dc_0'], ply['f_dc_1'], ply['f_dc_2']], 1) * 0.28209479 + 0.5, 0, 1)
            if args.cars and road_offset is None:
                from people import estimate_road  # noqa: PLC0415
                road_offset = estimate_road(xyz, rgb, opacity, route, self.ground_z)
                from people import check_road  # noqa: PLC0415

                def grey(R_cw, t_cw):
                    buf = self.renderer.render(R_cw, t_cw)[:cam['width'] * cam['height'] * 3]
                    a_ = np.frombuffer(buf, np.uint8).reshape(cam['height'], cam['width'], 3).astype(float)
                    return a_ @ [0.299, 0.587, 0.114]
                road_offset, sharp = check_road(road_offset, route, self.ground_z, grey)
                self.get_logger().info('cars: road views sharpness per stretch (kept at >= 4 and 40% of the sharpest): '
                                       + ', '.join('-' if v is None else f'{v:.0f}' for v in sharp))
                if all(o is None for _, _, o in road_offset):
                    raise SystemExit('--cars: no road found beside the route; pass --road-offset')
                self.get_logger().info('cars: road beside the route (estimated, s from-to: offset m): ' + ', '.join(
                    f'{a:.0f}-{b:.0f}: {"-" if o is None else f"{o:+.1f}"}' for a, b, o in road_offset))
            if args.cars and not np.isscalar(road_offset) and not args.no_road_fit:
                from people import fit_road_lines  # noqa: PLC0415
                road_lines = fit_road_lines(xyz, opacity, route, self.ground_z, road_offset)
                for s0, s1, px, py, ux, uy in road_lines:
                    self.get_logger().info(f'cars: road centre line {s0:.0f}-{s1:.0f}: through ({px:.1f}, {py:.1f}), '
                                           f'heading {math.degrees(math.atan2(uy, ux)):.1f} deg')
            pavement = None
            if args.cars and not np.isscalar(road_offset):
                # where the pavement ends on the road side: people keep off the road
                from people import pavement_edges  # noqa: PLC0415
                pavement = pavement_edges(xyz, rgb, opacity, route, self.ground_z, road_offset, walk=walk)
                if pavement:
                    e = np.array([(v - (np.interp(s_, *np.array(walk).T) if walk else 0.0)) * side
                                  for s_, v, side in pavement])
                    self.get_logger().info(f'people: pavement edge on the road side at s {pavement[0][0]:.0f}-'
                                           f'{pavement[-1][0]:.0f}: {e.min():.2f}-{e.max():.2f} m from the walk '
                                           f'(median {np.median(e):.2f})')
            self.crowd = Crowd(route, args.pedestrians, self.grid, seed=args.seed, speed=args.pedestrian_speed,
                               cyclists=args.cyclists, cyclist_speed=args.cyclist_speed,
                               dog_walkers=args.dog_walkers, strollers=args.strollers,
                               cars=args.cars, road_offset=road_offset, car_speed=args.car_speed,
                               road_lines=road_lines, pavement=pavement, walk=walk, narrow=narrow,
                               crossing=args.crossing)
            self.crowd.robot_radius = scene.get("robot_radius", 0.2)
        self.radius = scene.get('robot_radius', 0.2)
        x, y, yaw = scene['start']
        self.pose = [x, y, yaw]          # true pose in the map frame
        self.odom = [0.0, 0.0, 0.0]      # wheel odometry's pose in its odom frame
        self.v = self.w = 0.0
        self.cmd = (0.0, 0.0, -1.0)
        self.t = 0.0
        self.collisions = 0
        rng = np.random.default_rng(args.seed)
        self.v_scale = 1.0 + args.odom_scale_error * rng.choice([-1, 1])
        self.w_bias = args.odom_yaw_bias * rng.choice([-1, 1])
        self.w_scale = 1.0 + args.odom_yaw_scale_error * rng.choice([-1, 1])
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.p_clock = self.create_publisher(Clock, '/clock', 10)
        self.p_odom = self.create_publisher(Odometry, 'odom', 10)
        self.p_gt = self.create_publisher(Odometry, 'ground_truth', 10)
        self.p_tf = self.create_publisher(TFMessage, '/tf', 50)
        self.p_tfs = self.create_publisher(TFMessage, '/tf_static', latched)
        self.p_img = self.create_publisher(Image, 'camera/image', 5)
        self.p_scan = self.create_publisher(LaserScan, 'scan', 5)
        self.p_col = self.create_publisher(Int32, 'collisions', 10)
        self.create_subscription(Twist, 'cmd_vel', self.on_cmd, 10)
        self.publish_static()
        self.record = open(args.record, 'w') if args.record else None
        if self.record:
            self.record.write('time_s,true_x,true_y,true_yaw,odom_x,odom_y,odom_yaw,collisions\n')
        self.people = open(args.record + '.people.csv', 'w') if args.record and self.crowd else None
        if self.people:
            self.people.write('time_s,id,x,y,yaw,phase,kind\n')
            json.dump({'road_offset': getattr(self.crowd, 'road_offset', None),  # number or segments
                       'road_lines': getattr(self.crowd, 'road_lines', None),
                       'pavement': None if self.crowd.pavement is None else self.crowd.pavement.tolist(),
                       'walk': None if self.crowd.walk is None else self.crowd.walk.tolist(),
                       'narrow': self.crowd.narrow,
                       'crossing': args.crossing},
                      open(args.record + '.people.json', 'w'))

    def on_cmd(self, msg):
        self.cmd = (msg.linear.x, msg.angular.z, self.t)

    def publish_static(self):
        cam = self.cam
        tf = TransformStamped()
        tf.header.stamp = stamp(0.0)
        tf.header.frame_id, tf.child_frame_id = 'base_link', 'camera_optical'
        tf.transform.translation.x = self.scene['camera_forward']
        tf.transform.translation.z = self.scene['camera_height']
        # optical (x right, y down, z forward) in base_link (x forward, z up)
        tf.transform.rotation.x, tf.transform.rotation.y = -0.5, 0.5
        tf.transform.rotation.z, tf.transform.rotation.w = -0.5, 0.5
        self.p_tfs.publish(TFMessage(transforms=[tf]))

    def step(self):
        a = self.args
        v_cmd, w_cmd, t_cmd = self.cmd
        if self.t - t_cmd > a.cmd_timeout:
            v_cmd = w_cmd = 0.0
        dv = max(-a.max_accel * DT, min(a.max_accel * DT, v_cmd - self.v))
        dw = max(-a.max_alpha * DT, min(a.max_alpha * DT, w_cmd - self.w))
        self.v, self.w = self.v + dv, self.w + dw
        x, y, yaw = self.pose
        if self.kerbs and self.kerbs.on_edge(x, y, yaw):     # a wheel on a kerb: crawl over it
            self.v = max(-a.kerb_speed, min(a.kerb_speed, self.v))
        nyaw = yaw + self.w * DT
        nx = x + self.v * DT * math.cos(yaw + self.w * DT / 2)
        ny = y + self.v * DT * math.sin(yaw + self.w * DT / 2)
        if self.crowd:
            self.crowd.step(DT, (x, y))
        blocked = lambda px, py: self.grid.hits(px, py, self.radius) or bool(  # noqa: E731
            self.crowd and self.crowd.hits(px, py, self.radius))
        if self.v != 0 and blocked(nx, ny) and not blocked(x, y):
            self.collisions += 1
            nx, ny, self.v = x, y, 0.0
        moved = math.hypot(nx - x, ny - y) * (1 if self.v >= 0 else -1)
        self.pose = [nx, ny, math.atan2(math.sin(nyaw), math.cos(nyaw))]
        # Wheel odometry: what the encoders report, with their errors.
        ov, ow = moved / DT * self.v_scale, self.w * self.w_scale + (self.w_bias if abs(self.v) + abs(self.w) > 1e-3 else 0.0)
        ox, oy, oyaw = self.odom
        self.odom = [ox + ov * DT * math.cos(oyaw + ow * DT / 2), oy + ov * DT * math.sin(oyaw + ow * DT / 2),
                     math.atan2(math.sin(oyaw + ow * DT), math.cos(oyaw + ow * DT))]
        self.t += DT
        return ov, ow

    def publish(self, ov, ow, k):
        t = stamp(self.t)
        self.p_clock.publish(Clock(clock=t))
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = t, 'odom', 'base_link'
        odom.pose.pose.position.x, odom.pose.pose.position.y = self.odom[0], self.odom[1]
        q = quat_yaw(self.odom[2])
        o = odom.pose.pose.orientation
        o.x, o.y, o.z, o.w = q
        odom.twist.twist.linear.x, odom.twist.twist.angular.z = ov, ow
        self.p_odom.publish(odom)
        tf = TransformStamped()
        tf.header = odom.header
        tf.child_frame_id = 'base_link'
        tf.transform.translation.x, tf.transform.translation.y = self.odom[0], self.odom[1]
        r = tf.transform.rotation
        r.x, r.y, r.z, r.w = q
        self.p_tf.publish(TFMessage(transforms=[tf]))
        gt = Odometry()
        gt.header.stamp, gt.header.frame_id, gt.child_frame_id = t, 'map', 'base_link'
        gt.pose.pose.position.x, gt.pose.pose.position.y = self.pose[0], self.pose[1]
        o = gt.pose.pose.orientation
        o.x, o.y, o.z, o.w = quat_yaw(self.pose[2])
        self.p_gt.publish(gt)
        if k % int(round(1 / (DT * self.args.scan_hz))) == 0:
            self.publish_scan(t)
        if k % int(round(1 / (DT * self.args.camera_hz))) == 0:
            self.publish_image(t)
        if k % 5 == 0:
            self.p_col.publish(Int32(data=self.collisions))
            if self.record:
                self.record.write(f'{self.t:.3f},{self.pose[0]:.4f},{self.pose[1]:.4f},{self.pose[2]:.4f},'
                                  f'{self.odom[0]:.4f},{self.odom[1]:.4f},{self.odom[2]:.4f},{self.collisions}\n')
            if self.people:
                for i, (p, yw, ph, kd) in enumerate(zip(self.crowd.xy, self.crowd.yaw, self.crowd.phase,
                                                        self.crowd.kind)):
                    self.people.write(f'{self.t:.3f},{i},{p[0]:.4f},{p[1]:.4f},{yw:.4f},{ph:.3f},{kd}\n')

    def ground_z(self, x, y):
        """The ground, with the road between kerbs lowered."""
        return self.ground_raw(x, y) - (self.kerbs.drop_at(x, y) if self.kerbs else 0.0)

    def ground_raw(self, x, y):
        if self.ground is None:
            return 0.0
        g = self.grid
        i = min(max(int((x - g.x0) / g.res), 0), self.ground.shape[1] - 1)
        j = min(max(int((y - g.y0) / g.res), 0), self.ground.shape[0] - 1)
        return float(self.ground[j, i])

    def publish_scan(self, t):
        n = self.args.scan_rays
        r = self.grid.scan(self.pose[0], self.pose[1], self.pose[2], n, self.args.scan_range)
        if self.crowd:
            r = self.crowd.scan(self.pose[0], self.pose[1], self.pose[2], n, r)
        msg = LaserScan()
        msg.header.stamp, msg.header.frame_id = t, 'base_link'
        msg.angle_min, msg.angle_increment = -math.pi, 2 * math.pi / n
        msg.angle_max = msg.angle_min + (n - 1) * msg.angle_increment
        msg.range_min, msg.range_max = 0.05, self.args.scan_range
        msg.ranges = [float(v) for v in r]
        self.p_scan.publish(msg)

    def publish_image(self, t):
        cam = self.cam
        x, y, yaw = self.pose
        c, s = math.cos(yaw), math.sin(yaw)
        fwd = self.scene['camera_forward']
        # the body on its wheels: down a kerb with one pair, it pitches and the camera with it
        zmid, pitch = self.kerbs.pose(x, y, yaw) if self.kerbs else (0.0, 0.0)
        f = np.array([c * math.cos(pitch), s * math.cos(pitch), math.sin(pitch)])
        up = np.array([-c * math.sin(pitch), -s * math.sin(pitch), math.cos(pitch)])
        base = np.array([x, y, self.ground_raw(x, y) + zmid])
        centre = base + fwd * f + self.scene['camera_height'] * up
        # camera axes in the map: x right = -left, y down, z forward
        R_wc = np.stack([np.array([s, -c, 0.0]), -up, f], 1)
        R_cw = R_wc.T
        splats = self.crowd.splats(centre, self.ground_z) if self.splat_people else None
        buf = self.renderer.render(R_cw, -R_cw @ centre, splats)
        if self.crowd and not self.splat_people:
            n = cam['width'] * cam['height']
            rgb = np.frombuffer(buf[:n * 3], np.uint8).reshape(cam['height'], cam['width'], 3).astype(float)
            depth = np.frombuffer(buf[n * 3:], '<f4').reshape(cam['height'], cam['width'])
            fx = float(cam['model'].split()[3])
            rgb = self.crowd.draw(rgb, depth, R_wc, centre, fx, self.ground_z, fast=True)
            buf = np.clip(rgb, 0, 255).astype(np.uint8).tobytes()
        img = Image()
        img.header.stamp, img.header.frame_id = t, 'camera_optical'
        img.height, img.width = cam['height'], cam['width']
        if self.args.mono:
            a = np.frombuffer(buf, np.uint8).reshape(-1, 3).astype(np.uint16)
            grey = ((a[:, 0] * 77 + a[:, 1] * 150 + a[:, 2] * 29) >> 8).astype(np.uint8)
            img.encoding, img.step, img.data = 'mono8', cam['width'], grey.tobytes()
        else:
            img.encoding, img.step, img.data = 'rgb8', cam['width'] * 3, buf
        self.p_img.publish(img)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--scene', required=True)
    p.add_argument('--server', default=os.path.join(REPO, 'target', 'release', 'examples', 'gsplat_render_server'))
    p.add_argument('--record', default='')
    p.add_argument('--camera-hz', dest='camera_hz', type=float, default=10.0)
    p.add_argument('--scan-hz', dest='scan_hz', type=float, default=10.0)
    p.add_argument('--scan-rays', dest='scan_rays', type=int, default=360)
    p.add_argument('--scan-range', dest='scan_range', type=float, default=8.0)
    p.add_argument('--rtf', type=float, default=1.0, help='at most this many simulated seconds per second')
    p.add_argument('--odom-scale-error', dest='odom_scale_error', type=float, default=0.02)
    p.add_argument('--odom-yaw-bias', dest='odom_yaw_bias', type=float, default=0.002, help='rad/s while moving')
    p.add_argument('--odom-yaw-scale-error', dest='odom_yaw_scale_error', type=float, default=0.03)
    p.add_argument('--max-accel', dest='max_accel', type=float, default=1.0)
    p.add_argument('--max-alpha', dest='max_alpha', type=float, default=3.0)
    p.add_argument('--cmd-timeout', dest='cmd_timeout', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--mono', action='store_true',
                   help='publish mono8 (a third of the bytes; large rgb8 images can stall some DDS stacks)')
    p.add_argument('--pedestrians', type=int, default=0, help='people walking the route (people.py)')
    p.add_argument('--cyclists', type=int, default=0, help='people riding bicycles along the route (people.py)')
    p.add_argument('--dog-walkers', dest='dog_walkers', type=int, default=0, help='people walking a dog on a lead')
    p.add_argument('--strollers', type=int, default=0, help='people pushing a pushchair')
    p.add_argument('--cars', type=int, default=0, help='cars driving along the road beside the route')
    p.add_argument('--kerb-speed', dest='kerb_speed', type=float, default=0.12,
                   help="m/s at most while a wheel is on a kerb (the scene's 'kerbs')")
    p.add_argument('--road-offset', dest='road_offset', type=float, default=None,
                   help='the road centre line, m to the left of the route (default: estimated from the splat)')
    p.add_argument('--road', default='',
                   help='JSON [[s_start, s_end, offset or null], ...]: the road beside the route per stretch '
                        '(as estimated and logged, edited where the splat misleads the estimate)')
    p.add_argument('--no-road-fit', dest='no_road_fit', action='store_true',
                   help='lanes at the offsets from the route instead of along the road centre line fitted '
                        'to the splat (people.fit_road_lines)')
    p.add_argument('--crossing', action='store_true',
                   help='with --cars: a zebra crossing and a traffic light in the middle of the road')
    p.add_argument('--car-speed', dest='car_speed', type=float, default=6.0, help='m/s, each +-20%%')
    p.add_argument('--cyclist-speed', dest='cyclist_speed', type=float, default=4.0, help='m/s, each +-20%%')
    p.add_argument('--people-style', dest='people_style', choices=['splat', 'mesh'], default='splat',
                   help='splat: the people are gaussians rendered with the scene; mesh: drawn over it')
    p.add_argument('--pedestrian-speed', dest='pedestrian_speed', type=float, default=1.1, help='m/s, each +-20%%')
    p.add_argument('--duration', type=float, default=0.0, help='stop after this many simulated seconds (0: never)')
    args = p.parse_args()
    with open(args.scene, encoding='utf-8') as f:
        scene = json.load(f)
    rclpy.init()
    sim = Sim(args, scene)
    wall0, k = time.time(), 0
    try:
        while rclpy.ok() and (args.duration <= 0 or sim.t < args.duration):
            rclpy.spin_once(sim, timeout_sec=0.0)
            ov, ow = sim.step()
            k += 1
            sim.publish(ov, ow, k)
            ahead = sim.t / args.rtf - (time.time() - wall0)
            if ahead > 0:
                time.sleep(ahead)
    finally:
        if sim.record:
            sim.record.close()
        sim.renderer.p.stdin.close()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
