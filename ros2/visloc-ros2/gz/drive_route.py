#!/usr/bin/env python3
"""Drive the Gazebo robot along the apartment's mapping route on its true pose.

    python3 ros2/visloc-ros2/gz/drive_route.py [--scene runs/gz_depot/world/scene.json]

The route is `visloc_navigation::sim::apartment_mapping_route` (every room,
start and end at (1.5, 2.5)); the controller turns toward the next waypoint
and drives while facing it, as `sim::waypoint_command` does. It steers on
`/ground_truth`, standing in for a person driving the robot while it maps.
`--scene` takes the route of another world instead (make_depot_world.py's
scene.json). Exits when the route is done.
"""

import argparse
import json
import math
import sys

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter

ROUTE = [(1.5, 2.5), (2.8, 1.8), (5.0, 1.6), (5.3, 3.5), (3.0, 3.5), (0.7, 2.6), (0.7, 1.7),
         (2.5, 3.5), (2.5, 4.5), (1.5, 4.5), (1.5, 6.0), (3.8, 6.1), (1.5, 6.0), (1.5, 4.5),
         (8.5, 4.5), (8.5, 5.6), (6.0, 5.5), (9.3, 6.4), (8.5, 5.6), (8.5, 4.5), (7.5, 4.5),
         (7.5, 3.0), (8.3, 2.2), (7.0, 0.9), (6.6, 2.0), (5.0, 1.8), (3.0, 1.7), (1.5, 2.5)]
MAX_V, MAX_W = 0.3, 0.8


class Driver(Node):
    def __init__(self):
        super().__init__('drive_route', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/ground_truth', self.on_pose, 10)
        self.next = 1
        self.done = False

    def on_pose(self, msg):
        p = msg.pose.pose
        x, y = p.position.x, p.position.y
        q = p.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        while self.next < len(ROUTE) and math.hypot(ROUTE[self.next][0] - x, ROUTE[self.next][1] - y) < 0.15:
            self.next += 1
            self.get_logger().info(f'waypoint {self.next - 1}/{len(ROUTE) - 1}')
        cmd = Twist()
        if self.next >= len(ROUTE):
            self.done = True
        else:
            tx, ty = ROUTE[self.next]
            bearing = math.atan2(ty - y, tx - x) - yaw
            bearing = math.atan2(math.sin(bearing), math.cos(bearing))
            cmd.angular.z = max(-MAX_W, min(MAX_W, 1.5 * bearing))
            if abs(bearing) < 0.4:
                cmd.linear.x = min(MAX_V * (1 - abs(bearing) / 0.4), math.hypot(tx - x, ty - y))
        self.pub.publish(cmd)


def main():
    global ROUTE  # noqa: PLW0603
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scene', default='', help="a world's scene.json (its 'route')")
    args = parser.parse_args()
    if args.scene:
        with open(args.scene, encoding='utf-8') as f:
            ROUTE = [tuple(p) for p in json.load(f)['route']]
    rclpy.init()
    node = Driver()
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.pub.publish(Twist())
    node.get_logger().info('route done')
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
