#!/usr/bin/env python3
"""Convert a EuRoC MAV sequence (ASL folder layout) to a ROS 2 bag.

Writes `sensor_msgs/Image` (mono8) for cam0 and cam1 and `sensor_msgs/Imu`
for imu0, stamped with the dataset's nanosecond timestamps, under the topic
names of the official EuRoC ROS bags (`/cam0/image_raw`, `/cam1/image_raw`,
`/imu0`). Needs a sourced ROS 2 environment (`rosbag2_py`, message packages)
and Pillow.

usage: euroc_to_rosbag2.py <sequence dir containing mav0/> <output bag dir>
       [--storage mcap|sqlite3] [--max-frames N]

Play it into the nodes with, e.g.:
  ros2 bag play <bag>
  visloc_vio_node ... -r left/image_raw:=/cam0/image_raw \\
      -r right/image_raw:=/cam1/image_raw -r imu:=/imu0
"""

import argparse
import csv
from pathlib import Path

from PIL import Image as PilImage
import rosbag2_py
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image, Imu

TOPICS = {
    "/cam0/image_raw": "sensor_msgs/msg/Image",
    "/cam1/image_raw": "sensor_msgs/msg/Image",
    "/imu0": "sensor_msgs/msg/Imu",
}


def rows(path):
    with open(path, newline="") as handle:
        for row in csv.reader(handle):
            if row and not row[0].startswith("#"):
                yield row


def stamp(message, nanoseconds):
    message.header.stamp.sec = nanoseconds // 1_000_000_000
    message.header.stamp.nanosec = nanoseconds % 1_000_000_000


def image_message(path, nanoseconds, frame_id):
    pixels = PilImage.open(path).convert("L")
    message = Image()
    stamp(message, nanoseconds)
    message.header.frame_id = frame_id
    message.width, message.height = pixels.size
    message.encoding = "mono8"
    message.step = message.width
    message.data = pixels.tobytes()
    return message


def imu_message(row):
    message = Imu()
    stamp(message, int(row[0]))
    message.header.frame_id = "imu0"
    # Orientation is not measured (REP-145: covariance[0] = -1).
    message.orientation_covariance[0] = -1.0
    gyro, accel = message.angular_velocity, message.linear_acceleration
    gyro.x, gyro.y, gyro.z = (float(v) for v in row[1:4])
    accel.x, accel.y, accel.z = (float(v) for v in row[4:7])
    return message


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sequence", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--storage", default="mcap", choices=("mcap", "sqlite3"))
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    mav0 = args.sequence / "mav0"

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(args.output), storage_id=args.storage),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    for topic_id, (name, message_type) in enumerate(TOPICS.items()):
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                id=topic_id, name=name, type=message_type, serialization_format="cdr"
            )
        )

    # Merge the three streams in time order, as a recorder would see them.
    events = []
    for camera in ("cam0", "cam1"):
        frames = list(rows(mav0 / camera / "data.csv"))[: args.max_frames]
        events += [(int(r[0]), f"/{camera}/image_raw", camera, r[1]) for r in frames]
    last_image = max(event[0] for event in events)
    events += [(int(r[0]), "/imu0", None, r) for r in rows(mav0 / "imu0" / "data.csv")
               if int(r[0]) <= last_image]
    events.sort(key=lambda event: (event[0], event[1]))

    counts = dict.fromkeys(TOPICS, 0)
    for nanoseconds, topic, camera, payload in events:
        if camera is None:
            message = imu_message(payload)
        else:
            message = image_message(mav0 / camera / "data" / payload, nanoseconds, camera)
        writer.write(topic, serialize_message(message), nanoseconds)
        counts[topic] += 1
    del writer
    print(" ".join(f"{topic}={count}" for topic, count in counts.items()))


if __name__ == "__main__":
    main()
