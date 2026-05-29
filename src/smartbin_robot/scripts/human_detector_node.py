#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import os
import yaml

import torch
from functools import partial
torch.load = partial(torch.load, weights_only=False)

from ultralytics import YOLO
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, PointStamped, Twist
from cv_bridge import CvBridge

from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

from ament_index_python.packages import get_package_share_directory


class HumanDetectorNode(Node):
    def __init__(self):
        super().__init__('human_detector_node')
        self.bridge = CvBridge()

        # YOLO model
        torch.serialization.add_safe_globals([dict, list, set, torch.nn.Module, torch.Tensor])
        pkg_share = get_package_share_directory('smartbin_robot')
        model_path = os.path.join(pkg_share, 'models', 'yolov8n.pt')
        self.model = YOLO(model_path)

        # Home pose - loaded from config/home_pose.yaml
        home_pose_file = os.path.join(pkg_share, 'config', 'home_pose.yaml')
        try:
            with open(home_pose_file, 'r') as f:
                home_cfg = yaml.safe_load(f)
            self.home_x   = float(home_cfg['home_pose']['x'])
            self.home_y   = float(home_cfg['home_pose']['y'])
            self.home_yaw = float(home_cfg['home_pose']['yaw'])
            self.get_logger().info(
                f"Home pose loaded: x={self.home_x}, y={self.home_y}, yaw={self.home_yaw}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to load home_pose.yaml: {e}. Defaulting to origin.")
            self.home_x = self.home_y = self.home_yaw = 0.0

        # State tracking
        self.current_state    = "IDLE"
        self.target_published = False  # Ensures we only send one human goal per SEARCH

        # Camera intrinsics
        self.img_width = 640
        self.img_height = 480
        self.fov = 1.5   # radians (horizontal FOV)
        self.focal_length = (self.img_width / 2) / math.tan(self.fov / 2)
        self.avg_human_height = 1.7  # metres

        # TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers & subscriptions
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.state_pub = self.create_publisher(String, '/smartbin/robot_state', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_subscription(String, '/smartbin/robot_state', self.state_callback, 10)
        self.create_subscription(Image, '/camera', self.image_callback, 10)

        self.spin_timer = self.create_timer(0.1, self.spin_callback)

        self.get_logger().info("Human Detector Node ready. Waiting for SEARCH or GOHOME state...")

    # State machine
    def state_callback(self, msg):
        new_state = msg.data

        if new_state == "SEARCH" and self.current_state != "SEARCH":
            self.current_state = "SEARCH"
            self.target_published = False
            self.get_logger().info("SEARCH state - spinning to find human.")

        elif new_state == "GOTARGET" and self.current_state != "GOTARGET":
            self.current_state = "GOTARGET"
            self.get_logger().info("GOTARGET state - navigating to human.")

        elif new_state == "GOHOME" and self.current_state != "GOHOME":
            self.current_state = "GOHOME"
            self.target_published = False
            self.get_logger().info("GOHOME state - navigating to home pose.")
            self._go_home()

        elif new_state == "IDLE":
            self.current_state    = "IDLE"
            self.target_published = False
            self.get_logger().info("IDLE state - standing by.")

    # Spin Logic (SEARCH state)
    def spin_callback(self):
        """Continuously spins the robot while in SEARCH state."""
        if self.current_state == "SEARCH":
            twist = Twist()
            twist.angular.z = 1  # Spin speed (rad/s)
            self.cmd_vel_pub.publish(twist)

    # Human detection
    def image_callback(self, msg):
        if self.current_state != "SEARCH" or self.target_published:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results  = self.model(cv_image, classes=[0], verbose=False)

        for r in results:
            if len(r.boxes) == 0:
                continue

            box = r.boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            center_x = (x1 + x2) / 2
            box_height = y2 - y1

            distance = (self.avg_human_height * self.focal_length) / box_height
            yaw_angle = ((self.img_width / 2) - center_x) * (self.fov / self.img_width)
            lateral_offset = distance * math.tan(yaw_angle)

            # Stop 1m in front of the human
            target_distance = max(0.0, distance - 1.0)

            self.get_logger().info(
                f"Human detected! Distance: {distance:.2f} m, Yaw: {yaw_angle:.2f} rad"
            )

            self._publish_state("GOTARGET")
            
            self.publish_goal(target_distance, lateral_offset, msg.header.stamp)
            break

    def publish_goal(self, x_dist, y_dist, timestamp):
        """Transform a camera-frame point to map frame and send as Nav2 goal."""
        point_camera = PointStamped()
        point_camera.header.frame_id = 'camera_link_1'
        point_camera.header.stamp    = timestamp
        point_camera.point.x = x_dist
        point_camera.point.y = y_dist
        point_camera.point.z = 0.0

        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'camera_link_1', rclpy.time.Time()
            )
            point_map = tf2_geometry_msgs.do_transform_point(point_camera, transform)

            goal_msg = PoseStamped()
            goal_msg.header.frame_id = 'map'
            goal_msg.header.stamp    = self.get_clock().now().to_msg()
            goal_msg.pose.position.x = point_map.point.x
            goal_msg.pose.position.y = point_map.point.y
            goal_msg.pose.position.z = 0.0
            goal_msg.pose.orientation.w = 1.0  # Face the direction of travel

            self.goal_pub.publish(goal_msg)
            self.get_logger().info(
                f"Nav2 goal sent: x={point_map.point.x:.2f}, y={point_map.point.y:.2f}"
            )
            self.target_published = True  # Don't re-publish until next SEARCH

        except Exception as e:
            self.get_logger().error(f"TF transform failed: {e}")

    # Go home (GOHOME state)
    def _go_home(self):
        """Send a Nav2 goal to the home pose, then publish IDLE."""
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = self.home_x
        goal_msg.pose.position.y = self.home_y
        goal_msg.pose.position.z = 0.0

        # Convert yaw to quaternion (rotation around Z axis only)
        goal_msg.pose.orientation.x = 0.0
        goal_msg.pose.orientation.y = 0.0
        goal_msg.pose.orientation.z = math.sin(self.home_yaw / 2.0)
        goal_msg.pose.orientation.w = math.cos(self.home_yaw / 2.0)

        self.goal_pub.publish(goal_msg)
        self.get_logger().info(
            f"Returning home: x={self.home_x}, y={self.home_y}, yaw={self.home_yaw}"
        )

        self._publish_state("IDLE")

    def _publish_state(self, state: str):
        self.current_state = state
        msg = String()
        msg.data = state
        self.state_pub.publish(msg)
        self.get_logger().info(f"State published: {state}")


def main(args=None):
    rclpy.init(args=args)
    node = HumanDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()