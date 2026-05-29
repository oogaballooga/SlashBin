#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import os

import torch
from functools import partial
torch.load = partial(torch.load, weights_only=False)

from ultralytics import YOLO
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, PointStamped
from cv_bridge import CvBridge

# TF2 for frame transformations
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

from ament_index_python.packages import get_package_share_directory

class HumanDetectorNode(Node):
    def __init__(self):
        super().__init__('human_detector_node')
        self.bridge = CvBridge()
        
        # Load YOLOv8 model using absolute path
        torch.serialization.add_safe_globals([dict, list, set, torch.nn.Module, torch.Tensor])
        pkg_share = get_package_share_directory('smartbin_robot')
        model_path = os.path.join(pkg_share, 'models', 'yolov8n.pt')
        self.model = YOLO(model_path) 
        
        self.enabled = False
        self.target_published = False # Ensures we only send the goal once per search
        
        # Camera Intrinsics
        self.img_width = 640
        self.img_height = 480
        self.fov = 1.5 # radians
        # Focal length estimate: f = (width / 2) / tan(FOV / 2)
        self.focal_length = (self.img_width / 2) / math.tan(self.fov / 2)
        self.avg_human_height = 1.7 # meters
        
        # TF2 Setup for coordinate transformation
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Publishers and Subscriptions
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.create_subscription(String, '/smartbin/robot_state', self.state_callback, 10)
        self.create_subscription(Image, '/camera', self.image_callback, 10)
        
        self.get_logger().info("Human Detector Node Initialized. Waiting for SEARCH state...")

    def state_callback(self, msg):
        if msg.data == "SEARCH" and not self.enabled:
            self.enabled = True
            self.target_published = False
            self.get_logger().info("SEARCH state detected. Human detection enabled.")
        elif msg.data == "IDLE":
            self.enabled = False

    def image_callback(self, msg):
        if not self.enabled or self.target_published:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(cv_image, classes=[0], verbose=False)
        
        for r in results:
            boxes = r.boxes
            if len(boxes) > 0:
                # Take the first detected person
                box = boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                center_x = (x1 + x2) / 2
                box_height = y2 - y1
                
                # 1. Estimate Distance (Z) and lateral offset (Y)
                distance = (self.avg_human_height * self.focal_length) / box_height
                yaw_angle = ((self.img_width / 2) - center_x) * (self.fov / self.img_width)
                lateral_offset = distance * math.tan(yaw_angle)
                
                # Stop a safe distance (1.0m) away from the human
                target_distance = max(0.0, distance - 1.0)
                
                self.get_logger().info(f"Human detected! Distance: {distance:.2f}m, Yaw: {yaw_angle:.2f} rad")
                self.publish_goal(target_distance, lateral_offset, msg.header.stamp)
                break # Only process one person

    def publish_goal(self, x_dist, y_dist, timestamp):
        # Create a point in the camera's local frame
        point_camera = PointStamped()
        point_camera.header.frame_id = 'camera_link_1'
        point_camera.header.stamp = timestamp
        
        # Standard ROS camera frames: X is forward, Y is left
        point_camera.point.x = x_dist
        point_camera.point.y = y_dist
        point_camera.point.z = 0.0 

        try:
            # Transform the local coordinate to the absolute 'map' frame
            transform = self.tf_buffer.lookup_transform('map', 'camera_link_1', rclpy.time.Time())
            point_map = tf2_geometry_msgs.do_transform_point(point_camera, transform)
            
            # Create Nav2 Goal
            goal_msg = PoseStamped()
            goal_msg.header.frame_id = 'map'
            goal_msg.header.stamp = self.get_clock().now().to_msg()
            
            goal_msg.pose.position.x = point_map.point.x
            goal_msg.pose.position.y = point_map.point.y
            goal_msg.pose.position.z = 0.0
            
            # Simplified orientation (pointing in the direction of the target)
            goal_msg.pose.orientation.w = 1.0 
            
            self.goal_pub.publish(goal_msg)
            self.get_logger().info(f"Published Nav2 Goal: X={point_map.point.x:.2f}, Y={point_map.point.y:.2f}")
            
            # Stop detecting to let Nav2 take over
            self.target_published = True 
            
        except Exception as e:
            self.get_logger().error(f"Could not transform target to map frame: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = HumanDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()