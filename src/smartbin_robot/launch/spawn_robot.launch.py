import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    pkg_path = get_package_share_directory('smartbin_robot')

    # Fix for invisible meshes
    resource_path = os.path.join(pkg_path, '..')
    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=[resource_path]
    )

    # Robot State Publisher
    xacro_file = os.path.join(pkg_path, 'urdf', 'main.urdf.xacro')
    robot_description_raw = xacro.process_file(xacro_file).toxml()
    
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_raw, 'use_sim_time': True}]
    )

    # Gazebo Sim
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')]),
        launch_arguments={'gz_args': '-r empty.sdf'}.items(),
    )

    # Spawn Entity
    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', 'robot_description', '-name', 'smartbin_robot'],
    )

    # Bridge for ROS 2 <-> Gazebo
    bridge_params = os.path.join(pkg_path, 'config', 'gz_bridge.yaml')
    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['--ros-args', '-p', f'config_file:={bridge_params}'],
    )

    # Controllers
    joint_state_spawner = Node(package="controller_manager", executable="spawner", arguments=["joint_state_broadcaster"])
    wheel1_spawner = Node(package="controller_manager", executable="spawner", arguments=["wheel1_controller"])
    wheel2_spawner = Node(package="controller_manager", executable="spawner", arguments=["wheel2_controller"])
    wheel3_spawner = Node(package="controller_manager", executable="spawner", arguments=["wheel3_controller"])

    # Kinematics Node to bridge cmd_vel to wheels
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
        wheel1_spawner,
        wheel2_spawner,
        wheel3_spawner,
        kinematics_node
    ])