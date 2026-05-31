import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    pkg_path = get_package_share_directory('smartbin_robot')

    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=[os.path.join(pkg_path, '..')]
    )

    xacro_file = os.path.join(pkg_path, 'urdf', 'main.urdf.xacro')
    robot_description_raw = xacro.process_file(xacro_file).toxml()

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_raw, 'use_sim_time': True}]
    )

    world_file = os.path.join(pkg_path, 'worlds', 'smartbin_world.sdf')
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')]),
        launch_arguments={'gz_args': f'-r {world_file}'}.items(),
    )

    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', 'robot_description', '-name', 'smartbin_robot'],
    )

    bridge_params = os.path.join(pkg_path, 'config', 'gz_bridge.yaml')
    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['--ros-args', '-p', f'config_file:={bridge_params}'],
    )

    joint_state_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"]
    )

    # Single unified controller replaces the three separate wheel controllers
    omni_wheel_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["omni_wheel_controller"]
    )

    kinematics_node = Node(
        package='smartbin_robot',
        executable='kinematics_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        set_gz_resource_path,
        node_robot_state_publisher,
        gazebo_launch,
        gz_spawn_entity,
        bridge_node,
        joint_state_spawner,
        omni_wheel_spawner,
        kinematics_node,
    ])