#!/usr/bin/env python3
"""Record the Gazebo robot's stereo images, IMU and ground truth as EuRoC.

    python3 ros2/visloc-ros2/gz/record_euroc.py --out runs/gz_nav/mapping

Writes OUT/mav0/{cam0,cam1}/data/<ns>.png + data.csv, OUT/mav0/imu0/data.csv
and OUT/mav0/state_groundtruth_estimate0/data.csv (the IMU's true pose:
the robot's ground truth raised to imu_link, 0.5 m up on its axis), the
layout `basalt_euroc_online_slam_demo` reads. Stamps are simulation time.
Stops on SIGINT, or with --duration seconds of simulation time.
"""

import argparse
import os

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from PIL import Image as PilImage
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, Imu

IMU_HEIGHT = 0.5


def stamp_ns(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


class Recorder(Node):
    def __init__(self, out):
        super().__init__('record_euroc', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.mav0 = os.path.join(out, 'mav0')
        self.files = {}
        for cam in ('cam0', 'cam1'):
            os.makedirs(os.path.join(self.mav0, cam, 'data'), exist_ok=True)
            f = open(os.path.join(self.mav0, cam, 'data.csv'), 'w', encoding='utf-8')
            f.write('#timestamp [ns],filename\n')
            self.files[cam] = f
        for name, header in (('imu0', '#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],'
                                       'w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n'),
                             ('state_groundtruth_estimate0', '#timestamp,p_RS_R_x [m],p_RS_R_y [m],'
                              'p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],v_RS_R_x [m s^-1],'
                              'v_RS_R_y [m s^-1],v_RS_R_z [m s^-1],b_w_RS_S_x [rad s^-1],b_w_RS_S_y [rad s^-1],'
                              'b_w_RS_S_z [rad s^-1],b_a_RS_S_x [m s^-2],b_a_RS_S_y [m s^-2],b_a_RS_S_z [m s^-2]\n')):
            os.makedirs(os.path.join(self.mav0, name), exist_ok=True)
            f = open(os.path.join(self.mav0, name, 'data.csv'), 'w', encoding='utf-8')
            f.write(header)
            self.files[name] = f
        qos = QoSProfile(depth=200, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE)
        for cam in ('cam0', 'cam1'):
            self.create_subscription(Image, f'/{cam}/image_raw', lambda m, c=cam: self.on_image(c, m), qos)
        self.create_subscription(Imu, '/imu', self.on_imu, QoSProfile(depth=2000))
        self.create_subscription(Odometry, '/ground_truth', self.on_truth, qos)
        self.counts = {'cam0': 0, 'cam1': 0, 'imu0': 0, 'gt': 0}
        self.first_ns = None

    def on_image(self, cam, msg):
        ns = stamp_ns(msg)
        self.first_ns = self.first_ns or ns
        pixels = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)[:, :msg.width]
        PilImage.fromarray(pixels).save(os.path.join(self.mav0, cam, 'data', f'{ns}.png'))
        self.files[cam].write(f'{ns},{ns}.png\n')
        self.counts[cam] += 1

    def on_imu(self, msg):
        w, a = msg.angular_velocity, msg.linear_acceleration
        self.files['imu0'].write(f'{stamp_ns(msg)},{w.x},{w.y},{w.z},{a.x},{a.y},{a.z}\n')
        self.counts['imu0'] += 1

    def on_truth(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.files['state_groundtruth_estimate0'].write(
            f'{stamp_ns(msg)},{p.x},{p.y},{p.z + IMU_HEIGHT},{q.w},{q.x},{q.y},{q.z}' + ',0' * 9 + '\n')
        self.counts['gt'] += 1

    def close(self):
        for f in self.files.values():
            f.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--duration', type=float, default=0.0, help='simulation seconds (0: until SIGINT)')
    args = parser.parse_args()
    rclpy.init()
    node = Recorder(args.out)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            now = node.get_clock().now().nanoseconds
            if args.duration and node.first_ns and now - node.first_ns > args.duration * 1e9:
                break
    except KeyboardInterrupt:
        pass
    node.close()
    print(f'recorded {node.counts}')


if __name__ == '__main__':
    main()
