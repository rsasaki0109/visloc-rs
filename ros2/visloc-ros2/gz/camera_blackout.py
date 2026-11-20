#!/usr/bin/env python3
"""Blind the Gazebo robot's cameras for a while, as a covered lens would (docs/navigation.md).

    python3 ros2/visloc-ros2/gz/camera_blackout.py --at 4.5,4.5 --duration 5 \
        --log runs/gz_nav/blackout.json

Relays `/cam0/image_raw` and `/cam1/image_raw` to `/cam0/image` and
`/cam1/image` (launch `nav.launch.py` with `camera_suffix:=image` to feed
the visloc nodes from them). The first time the robot's true pose
(`/ground_truth`) comes within --radius of --at (world frame), every image
of the next --duration seconds of simulation time is replaced by a uniform
one (--value): the VIO and the map localization see no features at all.
Writes the blackout's start and end stamps to --log.
"""

import argparse
import json
import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image


class Blackout(Node):
    def __init__(self, args):
        super().__init__('camera_blackout')
        self.args = args
        self.at = tuple(float(v) for v in args.at.split(','))
        self.start_ns = None
        self.end_ns = None
        self.blanked = 0
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self.pubs = {}
        for cam in ('cam0', 'cam1'):
            self.pubs[cam] = self.create_publisher(Image, f'/{cam}/image', qos)
            self.create_subscription(Image, f'/{cam}/image_raw',
                                     lambda msg, cam=cam: self.on_image(cam, msg), qos)
        self.create_subscription(Odometry, '/ground_truth', self.on_truth, qos_profile_sensor_data)
        self.last_stamp_ns = 0

    def on_truth(self, msg):
        if self.start_ns is not None:
            return
        p = msg.pose.pose.position
        if math.hypot(p.x - self.at[0], p.y - self.at[1]) <= self.args.radius and self.last_stamp_ns:
            self.start_ns = self.last_stamp_ns
            self.end_ns = self.start_ns + int(self.args.duration * 1e9)
            self.get_logger().info(
                f'blackout from {self.start_ns * 1e-9:.2f} s for {self.args.duration} s '
                f'at ({p.x:.2f}, {p.y:.2f})')
            self.write_log()

    def on_image(self, cam, msg):
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.last_stamp_ns = max(self.last_stamp_ns, stamp_ns)
        if self.start_ns is not None and self.start_ns <= stamp_ns < self.end_ns:
            msg.data = bytes([self.args.value]) * len(msg.data)
            self.blanked += 1
        self.pubs[cam].publish(msg)

    def write_log(self):
        if self.args.log:
            with open(self.args.log, 'w', encoding='utf-8') as f:
                json.dump({'at': self.at, 'duration_s': self.args.duration,
                           'start_s': self.start_ns * 1e-9, 'end_s': self.end_ns * 1e-9}, f)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--at', required=True, help='x,y in the world: where the blackout starts')
    parser.add_argument('--radius', type=float, default=0.3)
    parser.add_argument('--duration', type=float, default=5.0, help='simulated seconds')
    parser.add_argument('--value', type=int, default=16, help='grey level of the blank images')
    parser.add_argument('--log', default='')
    args = parser.parse_args()
    rclpy.init()
    node = Blackout(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
