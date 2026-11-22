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
- with `--pedestrians N`, N people walk the scene's route (its start and
  goals) back and forth (`people.py`): they show in the camera, drawn over
  the splat by depth, and in the scan, so Nav2 has to steer round them;
  they wait while the robot is just in front of them. `--record` also writes
  their positions to `<record>.people.csv`;
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

    def __init__(self, server, ply, cam, depth=False):
        self.w, self.h = cam['width'], cam['height']
        self.depth = depth
        self.p = subprocess.Popen([server, '--ply', ply, '--camera', cam['model']] + (['--depth'] if depth else []),
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def render(self, R_cw, t_cw):
        line = ' '.join(f'{v:.7f}' for v in list(R_cw.reshape(-1)) + list(t_cw)) + '\n'
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
        self.renderer = Renderer(args.server, sp(scene['splat']), cam, depth=args.pedestrians > 0)
        g = scene['grid']
        self.grid = Grid(sp(g['file']), g['x0'], g['y0'], g['resolution'])
        self.crowd = None
        if args.pedestrians > 0:
            from people import Crowd  # noqa: PLC0415
            route = scene.get('route') or [scene['start']] + scene.get('goals', [])
            if len(route) < 2:
                raise SystemExit('--pedestrians needs a route: the scene has no goals')
            self.crowd = Crowd(route, args.pedestrians, self.grid, seed=args.seed, speed=args.pedestrian_speed)
        self.ground = np.load(sp(scene['ground'])) if scene.get('ground') else None
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
            self.people.write('time_s,id,x,y,yaw,phase\n')

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
                for i, (p, yw, ph) in enumerate(zip(self.crowd.xy, self.crowd.yaw, self.crowd.phase)):
                    self.people.write(f'{self.t:.3f},{i},{p[0]:.4f},{p[1]:.4f},{yw:.4f},{ph:.3f}\n')

    def ground_z(self, x, y):
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
        centre = np.array([x + c * fwd, y + s * fwd, self.scene['camera_height'] + self.ground_z(x, y)])
        # camera axes in the map: x right = -left, y down, z forward
        R_wc = np.array([[s, 0, c], [-c, 0, s], [0, -1, 0]], dtype=float)
        R_cw = R_wc.T
        buf = self.renderer.render(R_cw, -R_cw @ centre)
        if self.crowd:
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
