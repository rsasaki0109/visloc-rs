#!/usr/bin/env python3
"""Send Nav2 a sequence of goals and score the drive (docs/navigation.md).

    python3 ros2/visloc-ros2/tools/nav_sim_goals.py --out runs/nav_sim/visloc.json

Run with ROS 2 (rclpy, nav2_simple_commander) sourced, against
`nav_sim.launch.py`. For each goal: Nav2's result, the time it took, and the
true distance to the goal at the end (from `ground_truth`), so a goal Nav2
reports reached while the robot is elsewhere (wrong localization) counts as
missed. Over the whole drive: the believed-pose error (`map_pose` against
`ground_truth`) and the collisions (`collisions`). Writes a JSON report.
"""

import argparse
import json
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Int32

# Apartment goals (x, y, yaw): bedroom 1, bedroom 2, kitchen, living room, start.
DEFAULT_GOALS = [
    (3.0, 6.0, 0.0),
    (8.5, 6.0, 1.57),
    (8.0, 1.0, -1.57),
    (4.0, 1.0, 3.14),
    (1.5, 2.5, 0.0),
]


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Monitor:
    def __init__(self, node):
        self.truth = None
        self.estimate = None
        self.collisions = 0
        self.errors = []
        node.create_subscription(Odometry, 'ground_truth', self.on_truth, qos_profile_sensor_data)
        node.create_subscription(PoseStamped, 'map_pose', self.on_estimate, qos_profile_sensor_data)
        node.create_subscription(Int32, 'collisions', self.on_collisions, qos_profile_sensor_data)

    def on_truth(self, msg):
        p = msg.pose.pose
        self.truth = (p.position.x, p.position.y, yaw_of(p.orientation))

    def on_estimate(self, msg):
        p = msg.pose
        self.estimate = (p.position.x, p.position.y, yaw_of(p.orientation))
        if self.truth is not None:
            self.errors.append(math.hypot(self.estimate[0] - self.truth[0],
                                          self.estimate[1] - self.truth[1]))

    def on_collisions(self, msg):
        self.collisions = msg.data


def goal_pose(navigator, x, y, yaw):
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = navigator.get_clock().now().to_msg()
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.orientation.z = math.sin(yaw / 2.0)
    pose.pose.orientation.w = math.cos(yaw / 2.0)
    return pose


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', required=True, help='JSON report path')
    parser.add_argument('--timeout', type=float, default=180.0, help='seconds per goal')
    parser.add_argument('--tolerance', type=float, default=0.3,
                        help='a goal is reached when the true pose ends this close (m)')
    parser.add_argument('--label', default='', help='free text stored in the report')
    args = parser.parse_args()

    rclpy.init()
    navigator = BasicNavigator()
    monitor = Monitor(navigator)
    navigator.waitUntilNav2Active(localizer='robot_localization')
    # Wait until the robot is localized (map_pose published).
    deadline = time.monotonic() + 60.0
    while monitor.estimate is None and time.monotonic() < deadline:
        rclpy.spin_once(navigator, timeout_sec=0.1)
    started = time.monotonic()
    results = []
    for x, y, yaw in DEFAULT_GOALS:
        t0 = time.monotonic()
        navigator.goToPose(goal_pose(navigator, x, y, yaw))
        while not navigator.isTaskComplete():
            rclpy.spin_once(navigator, timeout_sec=0.05)
            if time.monotonic() - t0 > args.timeout:
                navigator.cancelTask()
        result = navigator.getResult()
        for _ in range(10):
            rclpy.spin_once(navigator, timeout_sec=0.05)
        truth = monitor.truth or (math.nan, math.nan, math.nan)
        distance = math.hypot(truth[0] - x, truth[1] - y)
        entry = {
            'goal': [x, y, yaw],
            'nav2_result': {TaskResult.SUCCEEDED: 'succeeded', TaskResult.CANCELED: 'canceled',
                            TaskResult.FAILED: 'failed'}.get(result, str(result)),
            'time_s': round(time.monotonic() - t0, 1),
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
        'collisions': monitor.collisions,
        'drive_time_s': round(time.monotonic() - started, 1),
        'pose_error_m': {
            'samples': len(errors),
            'rmse': round(math.sqrt(sum(e * e for e in errors) / len(errors)), 4) if errors else None,
            'median': round(errors[len(errors) // 2], 4) if errors else None,
            'max': round(errors[-1], 4) if errors else None,
        },
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    navigator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
