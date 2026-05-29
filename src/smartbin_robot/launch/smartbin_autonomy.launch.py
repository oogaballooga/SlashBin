import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_path = get_package_share_directory('smartbin_robot')

    # 1. Include the Simulation Launch (Spawns Gazebo + Robot + Bridges)
    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_path, 'launch', 'spawn_robot.launch.py')
        )
    )

    # 2. Include the Navigation Launch (Brings up Nav2 + Map Server)
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_path, 'launch', 'navigation.launch.py')
        )
    )

    # 3. Run Custom Audio Wake-Phrase Python Node
    audio_node = Node(
        package='smartbin_robot',
        executable='audio_trigger_node.py',
        name='audio_trigger_node',
        output='screen'
    )

    # 4. Run Custom YOLOv8 Human Tracker Position Projector Node
    detector_node = Node(
        package='smartbin_robot',
        executable='human_detector_node.py',
        name='human_detector_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        simulation_launch,
        navigation_launch,
        audio_node,
        detector_node
    ])