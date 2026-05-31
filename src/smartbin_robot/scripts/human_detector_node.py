#!/usr/bin/env python3
import rclpy
import math
import os
import yaml
import numpy as np
import torch
from functools import partial

from rclpy.node import Node
from rclpy.action import ActionClient
from ultralytics import YOLO
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, PointStamped, Twist
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from ament_index_python.packages import get_package_share_directory
import message_filters

torch.load = partial(torch.load, weights_only=False)
torch.serialization.add_safe_globals([dict, list, set, torch.nn.Module, torch.Tensor])

SEARCH_SPIN_SPEED = 1.2
# How long (seconds) to wait after cancelling the current goal before sending
# the home goal. This lets the global costmap run ~3 update cycles at 5 Hz so
# nearby obstacles (e.g. the human) are marked before the planner runs.
COSTMAP_SETTLE_S  = 0.7
HUMAN_STANDOFF_M  = 0.6
MIN_CONFIDENCE       = 0.80
DEPTH_SAMPLE_HALF    = 10
# Number of consecutive frames a detection must appear in before we act on it.
# Kills first-frame flukes and single-frame chair misdetections.
CONFIRM_FRAMES_REQUIRED = 3


class HumanDetectorNode(Node):
    def __init__(self):
        super().__init__('human_detector_node')
        self.bridge   = CvBridge()
        pkg_share     = get_package_share_directory('smartbin_robot')

        self.model = YOLO(os.path.join(pkg_share, 'models', 'yolov8s.pt'))
        # Warm up YOLO so the first real detection isn't a slow/unreliable
        # lazily-initialised inference. A blank frame is enough to trigger
        # CUDA kernel compilation without affecting any state.
        self.get_logger().info("Warming up YOLO model...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model(dummy, classes=[0], verbose=False)
        self.get_logger().info("YOLO warmup done.")

        try:
            with open(os.path.join(pkg_share, 'config', 'home_pose.yaml'), 'r') as f:
                cfg = yaml.safe_load(f)['home_pose']
                self.home_pose = (float(cfg['x']), float(cfg['y']), float(cfg['yaw']))
        except Exception as e:
            self.get_logger().error(f"Home pose load failed: {e}. Defaulting to 0,0,0.")
            self.home_pose = (0.0, 0.0, 0.0)

        self._nav_client          = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._active_goal_handle  = None
        self.current_state        = "IDLE"
        self._human_goal_sent     = False
        self._settle_timer        = None
        self._detection_streak    = 0   # consecutive frames with a valid high-conf human

        self.fx        = None
        self.cx_center = None

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.state_pub   = self.create_publisher(String, '/smartbin/robot_state', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_subscription(String,      '/smartbin/robot_state', self.state_callback,       10)
        self.create_subscription(CameraInfo,  '/camera_info',          self.camera_info_callback, 10)

        rgb_sub   = message_filters.Subscriber(self, Image, '/camera')
        depth_sub = message_filters.Subscriber(self, Image, '/depth')
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=5, slop=0.05
        )
        self.sync.registerCallback(self.synced_image_callback)

        self.create_timer(0.1, self._spin_tick)
        self.get_logger().info("Human Detector Node ready.")

    def camera_info_callback(self, msg):
        self.fx        = msg.k[0]
        self.cx_center = msg.k[2]

    def state_callback(self, msg):
        if msg.data == self.current_state:
            return
        self.current_state = msg.data
        self.get_logger().info(f"State transitioned to: {self.current_state}")

        if self.current_state == "SEARCH":
            self._human_goal_sent  = False
            self._detection_streak = 0
        elif self.current_state in ["GOTARGET", "IDLE"]:
            self.cmd_vel_pub.publish(Twist())
        elif self.current_state == "GOHOME":
            self.cmd_vel_pub.publish(Twist())
            # Cancel any active goal, then wait COSTMAP_SETTLE_S seconds before
            # sending the home goal. This gives the global costmap time to mark
            # the human (who is still standing nearby) so the planner routes
            # around them instead of straight through them.
            self._cancel_active_goal(then=self._deferred_go_home)

    def _spin_tick(self):
        if self.current_state == "SEARCH":
            t = Twist()
            t.angular.z = SEARCH_SPIN_SPEED
            self.cmd_vel_pub.publish(t)

    def synced_image_callback(self, rgb_msg, depth_msg):
        if self.current_state != "SEARCH" or self._human_goal_sent:
            return
        if self.fx is None:
            return

        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f"Depth decode error: {e}")
            return

        cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        results  = self.model(cv_image, classes=[0], verbose=False)

        for r in results:
            valid_boxes = [b for b in r.boxes if float(b.conf[0]) >= MIN_CONFIDENCE]
            if not valid_boxes:
                self._detection_streak = 0
                continue

            # Pick the detection the model is most sure about, not the closest/largest
            best_box = max(valid_boxes, key=lambda b: float(b.conf[0]))
            x1, y1, x2, y2 = best_box.xyxy[0].tolist()
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            distance = self._depth_at(depth_image, cx, cy)
            if not distance:
                self._detection_streak = 0
                continue

            # Require CONFIRM_FRAMES_REQUIRED consecutive confident detections
            # before committing — eliminates first-frame flukes
            self._detection_streak += 1
            self.get_logger().debug(
                f"Human detection streak: {self._detection_streak}/{CONFIRM_FRAMES_REQUIRED} "
                f"(conf={float(best_box.conf[0]):.2f}, dist={distance:.2f}m)"
            )
            if self._detection_streak < CONFIRM_FRAMES_REQUIRED:
                continue

            yaw_angle      = math.atan2((self.cx_center - cx), self.fx)
            lateral_offset = distance * math.tan(yaw_angle)

            self.get_logger().info(f"Target locked. Distance: {distance:.2f}m")
            self._human_goal_sent = True
            self._transition_to("GOTARGET")
            self._send_human_goal(distance, lateral_offset, rgb_msg.header.stamp)
            break

    def _depth_at(self, depth_image, cx, cy):
        h, w = depth_image.shape[:2]
        x0 = max(0, cx - DEPTH_SAMPLE_HALF)
        x1 = min(w - 1, cx + DEPTH_SAMPLE_HALF)
        y0 = max(0, cy - DEPTH_SAMPLE_HALF)
        y1 = min(h - 1, cy + DEPTH_SAMPLE_HALF)

        patch = depth_image[y0:y1+1, x0:x1+1].astype(np.float32)
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        return float(np.median(valid)) if valid.size >= 3 else None

    def _send_human_goal(self, distance, lateral_offset, timestamp):
        pt = PointStamped()
        pt.header.frame_id = 'camera_link_1'
        pt.header.stamp    = timestamp
        pt.point.x = distance
        pt.point.y = lateral_offset
        pt.point.z = 0.0

        try:
            cam_to_map  = self.tf_buffer.lookup_transform('map', 'camera_link_1', rclpy.time.Time())
            base_to_map = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            human_map   = tf2_geometry_msgs.do_transform_point(pt, cam_to_map)
        except Exception as e:
            self.get_logger().error(f"TF failed: {e}")
            self._human_goal_sent = False
            return self._transition_to("SEARCH")

        dx = human_map.point.x - base_to_map.transform.translation.x
        dy = human_map.point.y - base_to_map.transform.translation.y
        approach_dist = math.hypot(dx, dy)

        if approach_dist <= HUMAN_STANDOFF_M:
            return

        goal_x = human_map.point.x - (dx / approach_dist) * HUMAN_STANDOFF_M
        goal_y = human_map.point.y - (dy / approach_dist) * HUMAN_STANDOFF_M
        yaw    = math.atan2(dy, dx)

        self._send_nav_goal(goal_x, goal_y, yaw)

    def _deferred_go_home(self):
        """
        Wait COSTMAP_SETTLE_S seconds for the global costmap to mark nearby
        obstacles (the human still standing there) before asking the planner
        for a home path. Uses a one-shot ROS timer so we stay non-blocking.
        """
        self.get_logger().info(
            f"Waiting {COSTMAP_SETTLE_S}s for costmap to settle before planning home..."
        )
        self._settle_timer = self.create_timer(COSTMAP_SETTLE_S, self._go_home_once)

    def _go_home_once(self):
        """One-shot callback — cancel the timer immediately, then navigate home."""
        self._settle_timer.cancel()
        self._settle_timer = None
        self._go_home()

    def _go_home(self):
        self.get_logger().info("Heading to home pose...")
        self._send_nav_goal(*self.home_pose, home=True)

    def _send_nav_goal(self, x, y, yaw, home=False):
        pose = PoseStamped()
        pose.header.frame_id    = 'map'
        pose.header.stamp       = self.get_clock().now().to_msg()
        pose.pose.position.x    = x
        pose.pose.position.y    = y
        pose.pose.orientation.z = math.sin(yaw / 2)
        pose.pose.orientation.w = math.cos(yaw / 2)

        self._nav_client.wait_for_server()
        goal_future = self._nav_client.send_goal_async(NavigateToPose.Goal(pose=pose))
        goal_future.add_done_callback(lambda f: self._on_goal_accepted(f, home=home))

    def _on_goal_accepted(self, future, home=False):
        result = future.result()
        if not result.accepted:
            self.get_logger().warn("Goal rejected by Nav2.")
            self._on_goal_failed(home=home)
            return
        self._active_goal_handle = result
        result.get_result_async().add_done_callback(
            lambda f: self._on_nav_result(f, home=home)
        )

    def _on_nav_result(self, future, home=False):
        self._active_goal_handle = None
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            if home:
                self.get_logger().info("Reached home.")
                self._transition_to("IDLE")
            # Non-home success: robot is near the human, stay in GOTARGET and wait for dismiss
        else:
            # Nav2 aborted, timed out, or was cancelled by a recovery behaviour
            status_name = {
                GoalStatus.STATUS_ABORTED:   "ABORTED",
                GoalStatus.STATUS_CANCELED:  "CANCELLED",
            }.get(status, f"status={status}")
            self.get_logger().warn(f"Navigation goal {status_name}.")
            self._on_goal_failed(home=home)

    def _on_goal_failed(self, home=False):
        """
        Called when Nav2 rejects, aborts, or cancels a goal.
        Falls back to a recoverable state so voice commands can re-trigger.
        - Human goal failed  → back to SEARCH (spin and try to re-acquire)
        - Home goal failed   → back to IDLE so 'robot come here' works again
        """
        if home:
            self.get_logger().warn("Failed to reach home — returning to IDLE.")
            self._transition_to("IDLE")
        else:
            self.get_logger().warn("Failed to reach human — returning to SEARCH.")
            self._human_goal_sent = False
            self._transition_to("SEARCH")

    def _cancel_active_goal(self, then=None):
        if self._active_goal_handle is None:
            if then:
                then()
            return
        handle = self._active_goal_handle
        self._active_goal_handle = None
        cancel_future = handle.cancel_goal_async()
        if then:
            cancel_future.add_done_callback(lambda _: then())

    def _transition_to(self, new_state: str):
        self.current_state = new_state
        self.state_pub.publish(String(data=new_state))


def main(args=None):
    rclpy.init(args=args)
    node = HumanDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()