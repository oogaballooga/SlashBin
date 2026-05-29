#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
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
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge

from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

from ament_index_python.packages import get_package_share_directory


SEARCH_SPIN_SPEED = 0.6  # rad/s while scanning

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

        # Using the action client means we get a result callback when goal is reached
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._active_goal_handle = None   # Track so we can cancel on state change

        # State tracking
        self.current_state = "IDLE"
        self.target_published = False # Ensures we only send one human goal per SEARCH

        # Camera intrinsics
        self.img_width = 640
        self.img_height = 480
        self.fov = 1.5 # radians
        self.focal_length = (self.img_width / 2) / math.tan(self.fov / 2)
        self.avg_human_height = 1.7  # metres

        # TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers & subscriptions
        self.state_pub   = self.create_publisher(String, '/smartbin/robot_state', 10)
        self.cmd_vel_pub = self.create_publisher(Twist,  '/cmd_vel', 10)

        self.create_subscription(String, '/smartbin/robot_state', self.state_callback, 10)
        self.create_subscription(Image, '/camera', self.image_callback, 10)

        self.create_timer(0.1, self._spin_tick)

        self.get_logger().info("Human Detector Node ready. Waiting for SEARCH or GOHOME state...")

    # State machine
    def state_callback(self, msg):
        new_state = msg.data
        if new_state == self.current_state:
            return
 
        prev = self.current_state
        self.current_state = new_state
        self.get_logger().info(f"State: {prev} -> {new_state}")
 
        if new_state == "SEARCH":
            self._human_goal_sent = False
            self.get_logger().info(f"Spinning at {SEARCH_SPIN_SPEED} rad/s, scanning for humans...")
 
        elif new_state == "GOTARGET":
            # Spinning stopped automatically because state != SEARCH;
            # publish one explicit zero-vel so the robot doesn't coast
            self._stop_spinning()
 
        elif new_state == "GOHOME":
            self._stop_spinning()
            # Cancel whatever Nav2 was doing before heading home
            self._cancel_active_goal(then=self._go_home)
 
        elif new_state == "IDLE":
            self._stop_spinning()

    # Spin Logic (SEARCH state)
    def _spin_tick(self):
        if self.current_state != "SEARCH":
            return
        twist = Twist()
        twist.angular.z = SEARCH_SPIN_SPEED
        self.cmd_vel_pub.publish(twist)
 
    def _stop_spinning(self):
        self.cmd_vel_pub.publish(Twist())  # All-zero Twist

    # Human detection
    def image_callback(self, msg):
        if self.current_state != "SEARCH" or self._human_goal_sent:
            return
 
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(cv_image, classes=[0], verbose=False)
 
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
            target_distance = max(0.0, distance - 1.0)
 
            self.get_logger().info(
                f"Human detected! dist={distance:.2f} m  yaw={yaw_angle:.2f} rad"
            )
 
            # Mark sent BEFORE the async goal so a second camera frame can't
            # sneak through while the action client is still connecting
            self._human_goal_sent = True
            self._transition_to("GOTARGET")
            self._send_human_goal(target_distance, lateral_offset, msg.header.stamp)
            break

    def _send_human_goal(self, x_dist, y_dist, timestamp):
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
        except Exception as e:
            self.get_logger().error(f"TF transform failed: {e} — dropping back to SEARCH.")
            self._human_goal_sent = False
            self._transition_to("SEARCH")
            return
 
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = point_map.point.x
        pose.pose.position.y = point_map.point.y
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0 # Nav2 chooses approach heading
 
        self.get_logger().info(
            f"Sending GOTARGET goal: x={pose.pose.position.x:.2f}, y={pose.pose.position.y:.2f}"
        )
        # For the human goal we don't need a strict result, the user dismisses
        # with voice ("robot go home"). So we send via action but only log result.
        self._send_nav_goal(pose, result_cb=self._on_human_goal_result)
 
    def _on_human_goal_result(self, status):
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Reached human target.")
        else:
            self.get_logger().warn(f"Human navigation ended with status {status}.")
        # No automatic state transition here, the user drives this with voice
 
    # GOHOME state
    def _go_home(self):
        """Called only after any prior goal has been cancelled (or there was none)."""
        if self.current_state != "GOHOME":
            return  # State changed again while we were cancelling; do nothing
 
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = self.home_x
        pose.pose.position.y = self.home_y
        pose.pose.position.z = 0.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(self.home_yaw / 2.0)
        pose.pose.orientation.w = math.cos(self.home_yaw / 2.0)
 
        self.get_logger().info(
            f"Heading home: x={self.home_x}, y={self.home_y}, yaw={self.home_yaw:.3f} rad"
        )
        self._send_nav_goal(pose, result_cb=self._on_home_goal_result)
 
    def _on_home_goal_result(self, status):
        """
        Called by the action client when the GOHOME navigation finishes.
        This is the correct place to transition to IDLE — only AFTER the
        robot has physically arrived, not when the goal was merely sent.
        """
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Reached home. Transitioning to IDLE.")
        else:
            self.get_logger().warn(
                f"Home navigation ended with status {status}. Resetting to IDLE anyway."
            )
        # Transition to IDLE regardless of success/failure so the system
        # never gets stuck in GOHOME
        self._transition_to("IDLE")
 
    # Nav2 action client helpers 
    def _send_nav_goal(self, pose: PoseStamped, result_cb):
        """
        Send a NavigateToPose goal asynchronously.
        result_cb(status: int) is called when the action finishes.
        """
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("navigate_to_pose action server not available!")
            return
 
        goal_msg      = NavigateToPose.Goal()
        goal_msg.pose = pose
 
        send_future = self._nav_client.send_goal_async(goal_msg)
        # Capture result_cb in a closure so we know which callback to fire
        send_future.add_done_callback(
            lambda f: self._on_goal_accepted(f, result_cb)
        )
 
    def _on_goal_accepted(self, future, result_cb):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Nav2 rejected the goal.")
            return
 
        self._active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f: self._on_goal_result(f, result_cb)
        )
 
    def _on_goal_result(self, future, result_cb):
        self._active_goal_handle = None
        status = future.result().status
        result_cb(status)
 
    def _cancel_active_goal(self, then=None):
        """
        Cancel the currently active Nav2 goal if there is one, then call `then`.
        If no goal is active, calls `then` immediately.
        """
        if self._active_goal_handle is None:
            if then:
                then()
            return
 
        self.get_logger().info("Cancelling active Nav2 goal...")
        cancel_future = self._active_goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(
            lambda f: self._on_goal_cancelled(f, then)
        )
 
    def _on_goal_cancelled(self, future, then):
        self._active_goal_handle = None
        self.get_logger().info("Active goal cancelled.")
        if then:
            then()
 
    # Helpers 
    def _transition_to(self, new_state: str):
        """Publish a state change so all nodes stay in sync."""
        self.current_state = new_state
        msg      = String()
        msg.data = new_state
        self.state_pub.publish(msg)
        self.get_logger().info(f"Published state: {new_state}")
 
 
def main(args=None):
    rclpy.init(args=args)
    node = HumanDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()