"""Map localization on a robot: robot description, VIO, map localization, RViz.

Starts
- robot_state_publisher with `urdf` (base_link -> IMU -> cameras on /tf_static;
  `tools/calib_to_urdf.py` writes one from a Basalt calibration),
- visloc_vio_node, publishing odom -> `base_frame` (unless `use_vio:=false`,
  when your own odometry, e.g. wheel odometry, publishes `odom_topic`),
- visloc_map_localize_node with the camera extrinsics from /tf_static,
  publishing map -> odom and `map_pose`,
- RViz with `rviz/map_localization.rviz` (`rviz:=true`).

The nodes are plain binaries, not a colcon package: `bin_dir` is where
`cargo build --release` put them. For example, with EuRoC MH_02 played from
a bag and a map saved from MH_01:

    ros2 launch ros2/visloc-ros2/launch/map_localization.launch.py map:=runs/mh01_map rviz:=true
    ros2 bag play mh02_bag --delay 3
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(CRATE))
EUROC = os.path.join(REPO, "configs", "basalt", "variants", "official_euroc_ds")

ARGUMENTS = [
    ("map", None, "map directory from basalt_euroc_online_slam_demo --save-map or basalt_map_merge"),
    ("bin_dir", os.path.join(CRATE, "target", "release"), "directory of the node binaries"),
    ("calibration", os.path.join(EUROC, "euroc_ds_calib.json"), "Basalt calibration"),
    ("config", os.path.join(EUROC, "euroc_config.json"), "Basalt VIO config"),
    ("urdf", os.path.join(HERE, "euroc_rig.urdf"), "robot description"),
    ("left_topic", "/cam0/image_raw", "left camera (the map's camera)"),
    ("right_topic", "/cam1/image_raw", "right camera"),
    ("imu_topic", "/imu0", "IMU"),
    ("odom_topic", "/odom", "odometry the map localization anchors"),
    ("imu_frame", "imu0", "the IMU's link in the URDF"),
    ("base_frame", "base_link", "the robot's base link"),
    ("use_vio", "true", "run visloc_vio_node (false: another node publishes odom_topic)"),
    ("rviz", "false", "start RViz"),
    ("rviz_config", os.path.join(HERE, "rviz", "map_localization.rviz"), "RViz config"),
]


def launch_setup(context):
    arg = {name: context.launch_configurations[name] for name, _, _ in ARGUMENTS}
    with open(arg["urdf"]) as handle:
        description = handle.read()
    actions = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": description}],
            output="screen",
        )
    ]
    if arg["use_vio"].lower() == "true":
        actions.append(
            ExecuteProcess(
                name="visloc_vio_node",
                cmd=[
                    os.path.join(arg["bin_dir"], "visloc_vio_node"),
                    "--ros-args",
                    "-p", f"calibration:={arg['calibration']}",
                    "-p", f"config:={arg['config']}",
                    "-p", f"body_frame:={arg['imu_frame']}",
                    "-p", f"base_frame:={arg['base_frame']}",
                    "-r", f"left/image_raw:={arg['left_topic']}",
                    "-r", f"right/image_raw:={arg['right_topic']}",
                    "-r", f"imu:={arg['imu_topic']}",
                    "-r", f"odom:={arg['odom_topic']}",
                ],
                output="screen",
            )
        )
    actions.append(
        ExecuteProcess(
            name="visloc_map_localize_node",
            cmd=[
                os.path.join(arg["bin_dir"], "visloc_map_localize_node"),
                "--ros-args",
                "-p", f"map:={arg['map']}",
                "-p", "extrinsics:=tf",
                "-r", f"image:={arg['left_topic']}",
                "-r", f"odom:={arg['odom_topic']}",
            ],
            output="screen",
        )
    )
    if arg["rviz"].lower() == "true":
        actions.append(
            Node(package="rviz2", executable="rviz2", arguments=["-d", arg["rviz_config"]])
        )
    return actions


def generate_launch_description():
    declared = [
        DeclareLaunchArgument(name, description=description)
        if default is None
        else DeclareLaunchArgument(name, default_value=default, description=description)
        for name, default, description in ARGUMENTS
    ]
    return LaunchDescription([*declared, OpaqueFunction(function=launch_setup)])
