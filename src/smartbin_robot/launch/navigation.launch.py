import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    pkg_path = get_package_share_directory('smartbin_robot')
    nav2_bringup_share = get_package_share_directory('nav2_bringup')

    # Path to map and custom parameters
    map_yaml_file = os.path.join(pkg_path, 'map', 'smartbin_world.yaml')
    nav2_params_file = os.path.join(pkg_path, 'config', 'nav2_params.yaml')

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup_share, 'launch', 'bringup_launch.py')
            ),
            launch_arguments={
                'map': map_yaml_file,
                'params_file': nav2_params_file,
                'use_sim_time': 'True',
                'autostart': 'True',
            }.items()
        )
    ])