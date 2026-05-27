cmake_minimum_required(VERSION 3.8)
project(smartbin_robot)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(urdf REQUIRED)
find_package(xacro REQUIRED)
find_package(robot_state_publisher REQUIRED)
find_package(ros_gz_bridge REQUIRED)
find_package(ros_gz_sim REQUIRED)

# Added for Kinematics
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Build the kinematics node
include_directories(${EIGEN3_INCLUDE_DIRS})
add_executable(kinematics_node src/kinematics.cpp)
ament_target_dependencies(kinematics_node 
  rclcpp std_msgs geometry_msgs nav_msgs sensor_msgs tf2 tf2_ros
)

install(TARGETS kinematics_node DESTINATION lib/${PROJECT_NAME})

# Install configuration and support files to share
install(
  DIRECTORY 
    config
    launch
    urdf
    meshes
    worlds
  DESTINATION share/${PROJECT_NAME}
)

# Install executable scripts to lib
install(
  PROGRAMS
    scripts/audio_trigger_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()