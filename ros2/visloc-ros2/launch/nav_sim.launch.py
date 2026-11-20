"""Nav2 driving the simulated robot that visloc-rs localizes (docs/navigation.md).

    ros2 launch ros2/visloc-ros2/launch/nav_sim.launch.py \
        map_dir:=runs/nav_sim/map \
        node:=ros2/visloc-ros2/target/release/visloc_nav_sim_node \
        mode:=visloc

Starts `visloc_nav_sim_node` (robot, wheel odometry, scan, and map -> odom
from visloc's localization), Nav2's map_server on the map's occupancy grid
(`nav_sim_build_map` writes `occupancy.yaml` next to the visual map), and
Nav2's navigation servers with `nav_sim_nav2_params.yaml`. No AMCL: visloc is
the localization.

Arguments: map_dir, node (the node binary), mode (visloc | odometry |
ground_truth), extra_box (x0,y0,x1,y1,height: an obstacle the map lacks;
empty for none), record (CSV path; empty for none), params_file.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))


def _sim_node(context):
    get = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    command = [get('node'), '--map_dir', get('map_dir'), '--mode', get('mode')]
    if get('extra_box'):
        command += ['--extra_box', get('extra_box')]
    if get('record'):
        command += ['--record', get('record')]
    return [ExecuteProcess(cmd=command, output='screen', name='visloc_nav_sim_node')]


def _map_server(context):
    yaml_filename = os.path.join(LaunchConfiguration('map_dir').perform(context), 'occupancy.yaml')
    return [
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'yaml_filename': yaml_filename, 'use_sim_time': False}],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_map',
            output='screen',
            parameters=[{'autostart': True, 'node_names': ['map_server'], 'use_sim_time': False}],
        ),
    ]


def generate_launch_description():
    nav2_launch = os.path.join(
        get_package_share_directory('nav2_bringup'), 'launch', 'navigation_launch.py'
    )
    return LaunchDescription([
        DeclareLaunchArgument('map_dir', description='map saved by nav_sim_build_map'),
        DeclareLaunchArgument('node', default_value='visloc_nav_sim_node',
                              description='path of the visloc_nav_sim_node binary'),
        DeclareLaunchArgument('mode', default_value='visloc'),
        DeclareLaunchArgument('extra_box', default_value=''),
        DeclareLaunchArgument('record', default_value=''),
        DeclareLaunchArgument('params_file',
                              default_value=os.path.join(HERE, 'nav_sim_nav2_params.yaml')),
        OpaqueFunction(function=_sim_node),
        OpaqueFunction(function=_map_server),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_launch),
            launch_arguments={
                'params_file': LaunchConfiguration('params_file'),
                'use_sim_time': 'False',
                'autostart': 'True',
            }.items(),
        ),
    ])
