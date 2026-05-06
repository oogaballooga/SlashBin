#include <chrono>
#include <memory>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <stdexcept>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <Eigen/Dense>

using namespace std::chrono_literals;
using std::placeholders::_1;
using namespace std;

#define WHEEL_RADIUS        0.03
#define ROBOT_RADIUS        0.088

class OmniKinematics : public rclcpp::Node
{
public:
  OmniKinematics(int num_wheels_, double robot_radius_, double wheel_radius_)
  : Node("omni_kinematics")
  {
    N = num_wheels_;
    R = robot_radius_;
    r = wheel_radius_;

    tM = init_transform_matrix(N);
    tMI = pseudo_inverse(tM);
    tMI = Eigen::MatrixXd(tMI.block(0, 0, 2, tMI.cols()));
    
    wheel_joint_map_index["wheel1_joint"] = 0;
    wheel_joint_map_index["wheel2_joint"] = 1;
    wheel_joint_map_index["wheel3_joint"] = 2;

    for(int i = 0; i < N; i++){
      string topic_name = "wheel" + to_string(i+1) + "_controller/commands";
      auto pub = this->create_publisher<std_msgs::msg::Float64MultiArray>(topic_name, 10);
      pub_wheels.push_back(pub);
    }
    pub_odometry = this->create_publisher<nav_msgs::msg::Odometry>("odom", 10);

    sub_cmd_vel = this->create_subscription<geometry_msgs::msg::Twist>("cmd_vel", 10, std::bind(&OmniKinematics::cmd_vel_callback, this, _1));
    sub_join_states = this->create_subscription<sensor_msgs::msg::JointState>("joint_states", 10, std::bind(&OmniKinematics::join_states_callback, this, _1));

    tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(*this);
    last_time = this->get_clock()->now();
  }

private:
  vector<rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr> pub_wheels;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odometry;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_cmd_vel;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_join_states;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

  Eigen::MatrixXd tM; 
  Eigen::MatrixXd tMI; 
  unordered_map<string, int> wheel_joint_map_index; 

  int N; 
  double r; 
  double R; 
  double pos_x = 0;
  double pos_y = 0;
  double yaw = 0;
  rclcpp::Time last_time;

  void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
    Eigen::Vector3d v(msg->linear.x, msg->linear.y, msg->angular.z);
    Eigen::VectorXd M = tM * v;
    for(int i = 0; i < (int)M.size(); i++) {
      auto message = std_msgs::msg::Float64MultiArray();
      message.data = {M(i)};
      pub_wheels[i]->publish(message);
    }
  }

  void join_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        Eigen::VectorXd w_vel(N);
        bool found_any = false;
        for (size_t i = 0; i < msg->name.size(); ++i) {
          if (wheel_joint_map_index.count(msg->name[i])) {
              w_vel(wheel_joint_map_index[msg->name[i]]) = msg->velocity[i];
              found_any = true;
          }
        }
        if(!found_any) return;
        
        rclcpp::Time current_time = this->get_clock()->now();
        double dt = (current_time - last_time).seconds();
        last_time = current_time;
        
        Eigen::Matrix2d rM;
        rM << cos(yaw), -sin(yaw), sin(yaw), cos(yaw);
        Eigen::MatrixXd dp = rM * tMI * w_vel * dt;
        
        pos_x += dp(0);
        pos_y += dp(1);
        publish_odom(current_time);
  }

  void publish_odom(rclcpp::Time current_time) {
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = current_time;
    odom_msg.header.frame_id = "odom";
    odom_msg.child_frame_id = "base_footprint";
    odom_msg.pose.pose.position.x = pos_x;
    odom_msg.pose.pose.position.y = pos_y;

    tf2::Quaternion q;
    q.setRPY(0, 0, yaw);
    odom_msg.pose.pose.orientation.x = q.x();
    odom_msg.pose.pose.orientation.y = q.y();
    odom_msg.pose.pose.orientation.z = q.z();
    odom_msg.pose.pose.orientation.w = q.w();
    pub_odometry->publish(odom_msg);

    geometry_msgs::msg::TransformStamped odom_tf;
    odom_tf.header.stamp = current_time;
    odom_tf.header.frame_id = "odom";
    odom_tf.child_frame_id = "base_footprint";
    odom_tf.transform.translation.x = pos_x;
    odom_tf.transform.translation.y = pos_y;
    odom_tf.transform.rotation = odom_msg.pose.pose.orientation;
    tf_broadcaster->sendTransform(odom_tf);
  }

  Eigen::MatrixXd init_transform_matrix(int n_wheels) {
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n_wheels, 3);
    double del_angle = 360.0 / n_wheels;
    for(int i = 0; i < n_wheels; i++){
      M(i,0) = -sin((del_angle * i) * M_PI / 180.0)/r;
      M(i,1) = cos((del_angle * i) * M_PI / 180.0)/r;
      M(i,2) = R/r;
    }
    return M;
  }

  Eigen::MatrixXd pseudo_inverse(const Eigen::MatrixXd& A) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd inv_sv = svd.singularValues();
    for (int i = 0; i < inv_sv.size(); ++i) {
      if (inv_sv(i) > 1e-8) inv_sv(i) = 1.0 / inv_sv(i);
      else inv_sv(i) = 0;
    }
    return svd.matrixV() * inv_sv.asDiagonal() * svd.matrixU().transpose();
  }
};

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OmniKinematics>(3, ROBOT_RADIUS, WHEEL_RADIUS));
  rclcpp::shutdown();
  return 0;
}