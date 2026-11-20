#!/usr/bin/env python3
"""Send Nav2 goals in the Gazebo apartment and score the drive (docs/navigation.md).

    python3 ros2/visloc-ros2/gz/nav_goals.py --world-from-map runs/gz_nav/map/world_from_map.json \
        --out runs/gz_nav/nav.json [--extra-box 4.6,4.0,5.2,5.0]

The goals are in the world (rooms of the apartment) and are sent in the map
frame through `world_from_map.json` (align_map.py), as a person would click
them on the map. Per goal: Nav2's result, the simulated time it took, and
the robot's true distance to it at the end (`/ground_truth`); a goal is
reached when Nav2 succeeds and the robot is truly within --tolerance. Over
the drive: the error of `visloc_map_localize_node`'s `/map_pose` against the
ground truth, and contacts (the true pose closer than the robot's radius to
a wall or a piece of furniture). Runs on simulation time.
"""

import argparse
import json
import math
import os
import sys

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from nav_msgs.msg import Odometry
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_world import THICK, apartment  # noqa: E402

GOALS = [(3.0, 6.0, 0.0), (8.5, 6.0, 1.57), (8.0, 1.0, -1.57), (4.0, 1.0, 3.14), (1.5, 2.5, 0.0)]
RADIUS = 0.2


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def segment_distance(p, a, b):
    ab = (b[0] - a[0], b[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) /
                     max(ab[0] ** 2 + ab[1] ** 2, 1e-12)))
    return math.hypot(p[0] - a[0] - t * ab[0], p[1] - a[1] - t * ab[1])


def obstacles(extra_box):
    walls, boxes = apartment()
    segments = [(a, b, THICK / 2) for a, b, _ in walls]
    if extra_box:
        x0, y0, x1, y1 = (float(v) for v in extra_box.split(',')[:4])
        boxes = boxes + [((x0, y0), (x1, y1), 1.0)]
    for (x0, y0), (x1, y1), _ in boxes:
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        segments += [(corners[i], corners[(i + 1) % 4], 0.0) for i in range(4)]
    return segments


class Monitor:
    def __init__(self, node, world_from_map, segments):
        self.t = world_from_map
        self.segments = segments
        self.truth = None
        self.errors = []
        self.contacts = 0
        self.in_contact = False
        self.rows = []  # time, true x y yaw, believed x y (world)
        node.create_subscription(Odometry, '/ground_truth', self.on_truth, qos_profile_sensor_data)
        node.create_subscription(PoseWithCovarianceStamped, '/map_pose', self.on_estimate, 10)

    def to_world(self, x, y):
        c, s = math.cos(self.t['yaw']), math.sin(self.t['yaw'])
        return c * x - s * y + self.t['x'], s * x + c * y + self.t['y']

    def on_truth(self, msg):
        p = msg.pose.pose
        self.truth = (p.position.x, p.position.y, yaw_of(p.orientation))
        clearance = min(segment_distance(self.truth, a, b) - half for a, b, half in self.segments)
        touching = clearance < RADIUS - 0.01
        if touching and not self.in_contact:
            self.contacts += 1
        self.in_contact = touching

    def on_estimate(self, msg):
        if self.truth is None:
            return
        x, y = self.to_world(msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.errors.append(math.hypot(x - self.truth[0], y - self.truth[1]))
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.rows.append((stamp, *self.truth, x, y))


def map_goal(navigator, t, x, y, yaw):
    """A world goal expressed in the map frame."""
    c, s = math.cos(t['yaw']), math.sin(t['yaw'])
    dx, dy = x - t['x'], y - t['y']
    mx, my, myaw = c * dx + s * dy, -s * dx + c * dy, yaw - t['yaw']
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = navigator.get_clock().now().to_msg()
    pose.pose.position.x, pose.pose.position.y = mx, my
    pose.pose.orientation.z, pose.pose.orientation.w = math.sin(myaw / 2), math.cos(myaw / 2)
    return pose


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--world-from-map', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--extra-box', default='')
    parser.add_argument('--timeout', type=float, default=240.0, help='simulated seconds per goal')
    parser.add_argument('--tolerance', type=float, default=0.3)
    parser.add_argument('--label', default='')
    parser.add_argument('--record', default='',
                        help='CSV of true and believed poses (scripts/plot_nav_sim.py reads it)')
    args = parser.parse_args()
    with open(args.world_from_map, encoding='utf-8') as f:
        t = json.load(f)

    rclpy.init()
    navigator = BasicNavigator()
    navigator.set_parameters([Parameter('use_sim_time', value=True)])
    monitor = Monitor(navigator, t, obstacles(args.extra_box))
    tf_buffer = Buffer()
    TransformListener(tf_buffer, navigator)
    navigator.waitUntilNav2Active(localizer='robot_localization')
    # Nav2 rejects goals until it can look up the robot in the map: wait for
    # visloc's first map -> odom on /tf, not only for its /map_pose.
    while not (monitor.errors and tf_buffer.can_transform('map', 'base_link', Time())):
        rclpy.spin_once(navigator, timeout_sec=0.1)
    clock = navigator.get_clock()
    sim_s = lambda: clock.now().nanoseconds * 1e-9  # noqa: E731
    started = sim_s()
    results = []
    for x, y, yaw in GOALS:
        t0 = sim_s()
        navigator.goToPose(map_goal(navigator, t, x, y, yaw))
        while not navigator.isTaskComplete():
            rclpy.spin_once(navigator, timeout_sec=0.05)
            if sim_s() - t0 > args.timeout:
                navigator.cancelTask()
        result = navigator.getResult()
        for _ in range(10):
            rclpy.spin_once(navigator, timeout_sec=0.05)
        distance = math.hypot(monitor.truth[0] - x, monitor.truth[1] - y)
        entry = {
            'goal': [x, y, yaw],
            'nav2_result': {TaskResult.SUCCEEDED: 'succeeded', TaskResult.CANCELED: 'canceled',
                            TaskResult.FAILED: 'failed'}.get(result, str(result)),
            'sim_time_s': round(sim_s() - t0, 1),
            'true_distance_m': round(distance, 3),
            'reached': result == TaskResult.SUCCEEDED and distance <= args.tolerance,
        }
        results.append(entry)
        navigator.get_logger().info(f'goal {entry}')
    errors = sorted(monitor.errors)
    report = {
        'label': args.label,
        'goals': results,
        'reached': sum(r['reached'] for r in results),
        'goal_count': len(results),
        'contacts': monitor.contacts,
        'drive_sim_time_s': round(sim_s() - started, 1),
        'pose_error_m': {
            'samples': len(errors),
            'rmse': round(math.sqrt(sum(e * e for e in errors) / len(errors)), 4),
            'median': round(errors[len(errors) // 2], 4),
            'max': round(errors[-1], 4),
        },
        'world_from_map': t,
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    if args.record:
        with open(args.record, 'w', encoding='utf-8') as f:
            f.write('time_s,true_x,true_y,true_yaw,est_x,est_y,est_yaw,collisions\n')
            for t, tx, ty, tyaw, ex, ey in monitor.rows:
                f.write(f'{t:.3f},{tx:.4f},{ty:.4f},{tyaw:.4f},{ex:.4f},{ey:.4f},,0\n')
    print(json.dumps(report, indent=2))
    rclpy.shutdown()


if __name__ == '__main__':
    main()
