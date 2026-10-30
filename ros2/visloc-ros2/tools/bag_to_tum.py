#!/usr/bin/env python3
"""Write the poses of one topic of a ROS 2 bag as a TUM trajectory.

Reads `geometry_msgs/PoseWithCovarianceStamped` (e.g. `map_pose`),
`nav_msgs/Odometry` (`odom`) or `geometry_msgs/PoseStamped` (`pose`) and
writes `stamp x y z qx qy qz qw` lines sorted by header stamp, for
`scripts/evaluate_map_localization.py`. Needs a sourced ROS 2 environment.

usage: bag_to_tum.py <bag> <topic> <out.tum>
"""

import sys

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    bag, topic, out = sys.argv[1:]
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag), rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {metadata.name: metadata.type for metadata in reader.get_all_topics_and_types()}
    if topic not in types:
        sys.exit(f"{topic} is not in {bag}: {sorted(types)}")
    message_type = get_message(types[topic])
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    lines = []
    while reader.has_next():
        _, data, _ = reader.read_next()
        message = deserialize_message(data, message_type)
        # PoseWithCovarianceStamped and Odometry nest the pose one level deeper.
        pose = message.pose.pose if hasattr(message.pose, "pose") else message.pose
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        p, q = pose.position, pose.orientation
        lines.append((stamp, f"{stamp:.9f} {p.x:.9f} {p.y:.9f} {p.z:.9f} "
                             f"{q.x:.9f} {q.y:.9f} {q.z:.9f} {q.w:.9f}\n"))
    with open(out, "w") as handle:
        handle.writelines(line for _, line in sorted(lines))
    print(f"{len(lines)} poses")


if __name__ == "__main__":
    main()
