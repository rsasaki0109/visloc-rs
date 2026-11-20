"""Nav2 in Gazebo on visloc-rs's real robot nodes (docs/navigation.md).

    ros2 launch ros2/visloc-ros2/gz/nav.launch.py world:=runs/gz_nav/world/apartment.sdf \
        map:=runs/gz_nav/map nodes:=ros2/visloc-ros2/target/release

Starts Gazebo (headless, rendering on the CPU is fine: Nav2 runs on
simulation time), the ros_gz bridge (`bridge.yaml`), `visloc_vio_node` on the
stereo pair and IMU (`odom -> imu_link`), `visloc_map_localize_node` on the
left camera against the saved map (`map -> odom`), a static `imu_link ->
base_link`, Nav2's map_server on the map's `occupancy.yaml`, and Nav2's
navigation servers (`nav2_params.yaml`). No AMCL and no wheel odometry: the
robot's odometry is the VIO, its localization visloc's.

Arguments: world, map (a `--save-map` directory with `occupancy.yaml`),
nodes (directory of the visloc ROS 2 binaries), calibration (Basalt
calibration of the rig; default `calibration.json` next to the world),
config (Basalt VIO config), params_file.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
SIM = {'use_sim_time': True}


def _setup(context):
    get = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    world, map_dir, nodes = get('world'), get('map'), get('nodes')
    calibration = get('calibration') or os.path.join(os.path.dirname(world), 'calibration.json')
    return [
        ExecuteProcess(cmd=['gz', 'sim', '-s', '-r', '--headless-rendering', world],
                       cwd=os.path.dirname(world), output='log', name='gazebo'),
        Node(package='ros_gz_bridge', executable='parameter_bridge', name='bridge', output='log',
             parameters=[{'config_file': os.path.join(HERE, 'bridge.yaml')}, SIM]),
        Node(package='tf2_ros', executable='static_transform_publisher', name='imu_to_base',
             arguments=['--z', '-0.5', '--frame-id', 'imu_link', '--child-frame-id', 'base_link'],
             parameters=[SIM]),
        ExecuteProcess(cmd=[os.path.join(nodes, 'visloc_vio_node'), '--calibration', calibration,
                            '--config', get('config'),
                            '--remap', 'left/image_raw=/cam0/image_raw',
                            '--remap', 'right/image_raw=/cam1/image_raw', '--remap', 'imu=/imu',
                            '--publish_path', 'false'],
                       output='screen', name='visloc_vio_node'),
        ExecuteProcess(cmd=[os.path.join(nodes, 'visloc_map_localize_node'), '--map', map_dir,
                            '--remap', 'image=/cam0/image_raw', '--publish_map_cloud', 'false'],
                       output='screen', name='visloc_map_localize_node'),
        Node(package='nav2_map_server', executable='map_server', name='map_server', output='log',
             parameters=[{'yaml_filename': os.path.join(map_dir, 'occupancy.yaml')}, SIM]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_map', output='log',
             parameters=[{'autostart': True, 'node_names': ['map_server']}, SIM]),
    ]


def generate_launch_description():
    nav2_launch = os.path.join(get_package_share_directory('nav2_bringup'), 'launch',
                               'navigation_launch.py')
    return LaunchDescription([
        DeclareLaunchArgument('world'),
        DeclareLaunchArgument('map'),
        DeclareLaunchArgument('nodes', default_value=os.path.join(REPO, 'ros2', 'visloc-ros2',
                                                                  'target', 'release')),
        DeclareLaunchArgument('calibration', default_value=''),
        DeclareLaunchArgument('config', default_value=os.path.join(
            REPO, 'configs', 'basalt', 'variants', 'official_euroc_ds', 'euroc_config.json')),
        DeclareLaunchArgument('params_file', default_value=os.path.join(HERE, 'nav2_params.yaml')),
        DeclareLaunchArgument('log_level', default_value='info', description="Nav2's log level"),
        OpaqueFunction(function=_setup),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_launch),
            launch_arguments={'params_file': LaunchConfiguration('params_file'),
                              'use_sim_time': 'True', 'autostart': 'True',
                              'log_level': LaunchConfiguration('log_level')}.items()),
    ])
