"""Nav2 in a 3D Gaussian splat of a real place, on visloc localization (docs/navigation.md).

    ros2 launch ros2/visloc-ros2/gs/gs_nav.launch.py scene:=runs/gs_tum/scene.json \
        map:=runs/gs_tum/map nodes:=ros2/visloc-ros2/target/release

Starts `gs_sim.py` (the robot: wheel odometry `odom -> base_link`, the
camera rendered from the splat, a 2D scan, `/clock`),
`visloc_map_localize_node` on that camera against the visual map
(`map -> odom`, extrinsics from `/tf_static`), Nav2's map_server on the
map's `occupancy.yaml`, and Nav2's navigation servers (the Gazebo setup's
`nav2_params.yaml`). No AMCL: the robot's localization is visloc's.

Arguments: scene, map, nodes, params_file, record (CSV of the true and
odometry poses), rtf (largest real-time factor), log_level, and
localize_args (extra `visloc_map_localize_node` arguments, space-separated).
"""

import os
import sys

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
    sim = [sys.executable, os.path.join(HERE, 'gs_sim.py'), '--scene', get('scene'), '--rtf', get('rtf')]
    if get('record'):
        sim += ['--record', get('record')]
    return [
        ExecuteProcess(cmd=sim, output='screen', name='gs_sim'),
        ExecuteProcess(cmd=[os.path.join(get('nodes'), 'visloc_map_localize_node'), '--map', get('map'),
                            '--remap', 'image=/camera/image', '--extrinsics', 'tf',
                            '--publish_map_cloud', 'false', *get('localize_args').split()],
                       output='screen', name='visloc_map_localize_node'),
        Node(package='nav2_map_server', executable='map_server', name='map_server', output='log',
             parameters=[{'yaml_filename': os.path.join(get('map'), 'occupancy.yaml')}, SIM]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_map', output='log',
             parameters=[{'autostart': True, 'node_names': ['map_server']}, SIM]),
    ]


def generate_launch_description():
    nav2_launch = os.path.join(get_package_share_directory('nav2_bringup'), 'launch',
                               'navigation_launch.py')
    return LaunchDescription([
        DeclareLaunchArgument('scene'),
        DeclareLaunchArgument('map'),
        DeclareLaunchArgument('nodes', default_value=os.path.join(REPO, 'ros2', 'visloc-ros2',
                                                                  'target', 'release')),
        DeclareLaunchArgument('params_file', default_value=os.path.join(HERE, '..', 'gz', 'nav2_params.yaml')),
        DeclareLaunchArgument('record', default_value=''),
        DeclareLaunchArgument('rtf', default_value='1.0'),
        DeclareLaunchArgument('localize_args', default_value=''),
        DeclareLaunchArgument('log_level', default_value='info', description="Nav2's log level"),
        OpaqueFunction(function=_setup),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_launch),
            launch_arguments={'params_file': LaunchConfiguration('params_file'),
                              'use_sim_time': 'True', 'autostart': 'True',
                              'log_level': LaunchConfiguration('log_level')}.items()),
    ])
