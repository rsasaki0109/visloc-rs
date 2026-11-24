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
odometry poses), occupancy (Nav2's map yaml instead of the map's
`occupancy.yaml`, e.g. `lane_map.py`'s, which keeps the robot to the walk), rtf (largest real-time factor), log_level, sim_args (extra
`gs_sim.py` arguments, e.g. `--mono`), and
localize_args (extra `visloc_map_localize_node` arguments, space-separated),
and robot_radius (m: Nav2's footprint circle instead of the params file's, with
the inflation 0.25 m beyond it; e.g. 0.45 for the delivery robot drawn in the
chase video, 0.87 x 0.70 m, so it does not squeeze between people).
"""

import os
import re
import sys
import tempfile

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
    sim += get('sim_args').split()
    params = get('params_file')
    if get('robot_radius'):
        r = float(get('robot_radius'))
        text = open(params, encoding='utf-8').read()
        text = re.sub(r'robot_radius: [0-9.]+', f'robot_radius: {r}', text)
        text = re.sub(r'inflation_radius: [0-9.]+', f'inflation_radius: {r + 0.25:.2f}', text)
        params = os.path.join(tempfile.mkdtemp(prefix='gs_nav_'), 'nav2_params.yaml')
        open(params, 'w', encoding='utf-8').write(text)
    nav2_launch = os.path.join(get_package_share_directory('nav2_bringup'), 'launch', 'navigation_launch.py')
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_launch),
            launch_arguments={'params_file': params, 'use_sim_time': 'True', 'autostart': 'True',
                              'log_level': get('log_level')}.items()),
        ExecuteProcess(cmd=sim, output='screen', name='gs_sim'),
        ExecuteProcess(cmd=[os.path.join(get('nodes'), 'visloc_map_localize_node'), '--map', get('map'),
                            '--remap', 'image=/camera/image', '--extrinsics', 'tf',
                            '--publish_map_cloud', 'false', *get('localize_args').split()],
                       output='screen', name='visloc_map_localize_node'),
        Node(package='nav2_map_server', executable='map_server', name='map_server', output='log',
             parameters=[{'yaml_filename': get('occupancy') or os.path.join(get('map'), 'occupancy.yaml')}, SIM]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_map', output='log',
             parameters=[{'autostart': True, 'node_names': ['map_server']}, SIM]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('scene'),
        DeclareLaunchArgument('map'),
        DeclareLaunchArgument('nodes', default_value=os.path.join(REPO, 'ros2', 'visloc-ros2',
                                                                  'target', 'release')),
        DeclareLaunchArgument('params_file', default_value=os.path.join(HERE, '..', 'gz', 'nav2_params.yaml')),
        DeclareLaunchArgument('record', default_value=''),
        DeclareLaunchArgument('occupancy', default_value='',
                              description="Nav2's map yaml (default: the map's occupancy.yaml; lane_map.py's "
                                          'keeps the robot to the walk)'),
        DeclareLaunchArgument('rtf', default_value='1.0'),
        DeclareLaunchArgument('localize_args', default_value=''),
        DeclareLaunchArgument('sim_args', default_value='', description='extra gs_sim.py arguments'),
        DeclareLaunchArgument('log_level', default_value='info', description="Nav2's log level"),
        DeclareLaunchArgument('robot_radius', default_value='',
                              description="m: Nav2's footprint circle (default: the params file's)"),
        OpaqueFunction(function=_setup),
    ])
