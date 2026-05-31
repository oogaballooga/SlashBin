#include <chrono>
#include <memory>
#include <cmath>
#include <vector>
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
using std::string;
using std::vector;
using std::unordered_map;

#define WHEEL_RADIUS  0.03
#define ROBOT_RADIUS  0.1025

class OmniKinematics : public rclcpp::Node
{
public:
  OmniKinematics(int num_wheels, double robot_radius, double wheel_radius)
  : Node("omni_kinematics"), N(num_wheels), R(robot_radius), r(wheel_radius)
  {
    tM  = init_transform_matrix(N);
    tMI = pseudo_inverse(tM);

    wheel_joint_map_index["omni_wheel_joint_1"] = 0;
    wheel_joint_map_index["omni_wheel_joint_2"] = 1;
    wheel_joint_map_index["omni_wheel_joint_3"] = 2;

    // Single publisher for all three wheels — matches the unified controller
    pub_wheels = this->create_publisher<std_msgs::msg::Float64MultiArray>(
      "omni_wheel_controller/commands", 10);

    pub_odometry = this->create_publisher<nav_msgs::msg::Odometry>("odom", 10);

    sub_cmd_vel = this->create_subscription<geometry_msgs::msg::Twist>(
      "cmd_vel", 10, std::bind(&OmniKinematics::cmd_vel_callback, this, _1));

    sub_joint_states = this->create_subscription<sensor_msgs::msg::JointState>(
      "joint_states", 10, std::bind(&OmniKinematics::joint_states_callback, this, _1));

    tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(*this);
    last_time = this->get_clock()->now();
  }

private:
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_wheels;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odometry;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_cmd_vel;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joint_states;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

  Eigen::MatrixXd tM;
  Eigen::MatrixXd tMI;
  unordered_map<string, int> wheel_joint_map_index;

  int N;
  double r, R;
  double pos_x = 0, pos_y = 0, yaw = 0;
  double vx_body = 0, vy_body = 0, vyaw = 0;  // last commanded body-frame velocity
  rclcpp::Time last_time;

  void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    vx_body = msg->linear.x;
    vy_body = msg->linear.y;
    vyaw    = msg->angular.z;

    Eigen::Vector3d v(vx_body, vy_body, vyaw);
    Eigen::VectorXd wheel_vels = tM * v;

    auto message = std_msgs::msg::Float64MultiArray();
    for (int i = 0; i < N; i++) {
      message.data.push_back(wheel_vels(i));
    }
    pub_wheels->publish(message);
  }

  void joint_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    Eigen::VectorXd w_vel = Eigen::VectorXd::Zero(N);
    bool found_any = false;

    for (size_t i = 0; i < msg->name.size(); ++i) {
      auto it = wheel_joint_map_index.find(msg->name[i]);
      if (it != wheel_joint_map_index.end()) {
        w_vel(it->second) = msg->velocity[i];
        found_any = true;
      }
    }
    if (!found_any) return;

    rclcpp::Time current_time = this->get_clock()->now();
    double dt = (current_time - last_time).seconds();
    last_time = current_time;

    // Body-frame velocity from wheel states
    Eigen::VectorXd body_vel = tMI * w_vel;

    // Rotate body-frame dx/dy into world frame
    Eigen::Matrix3d rM = Eigen::Matrix3d::Identity();
    rM(0, 0) =  cos(yaw);  rM(0, 1) = -sin(yaw);
    rM(1, 0) =  sin(yaw);  rM(1, 1) =  cos(yaw);

    Eigen::VectorXd dp = rM * body_vel * dt;

    pos_x += dp(0);
    pos_y += dp(1);
    yaw   += dp(2);

    publish_odom(current_time, body_vel);
  }

  void publish_odom(rclcpp::Time current_time, const Eigen::VectorXd& body_vel)
  {
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp    = current_time;
    odom_msg.header.frame_id = "odom";
    odom_msg.child_frame_id  = "base_footprint";

    odom_msg.pose.pose.position.x = pos_x;
    odom_msg.pose.pose.position.y = pos_y;

    tf2::Quaternion q;
    q.setRPY(0, 0, yaw);
    odom_msg.pose.pose.orientation.x = q.x();
    odom_msg.pose.pose.orientation.y = q.y();
    odom_msg.pose.pose.orientation.z = q.z();
    odom_msg.pose.pose.orientation.w = q.w();

    // Populate body-frame velocity so Nav2/AMCL have a proper twist
    odom_msg.twist.twist.linear.x  = body_vel(0);
    odom_msg.twist.twist.linear.y  = body_vel(1);
    odom_msg.twist.twist.angular.z = body_vel(2);

    // Diagonal covariance — gives AMCL a realistic noise model
    odom_msg.pose.covariance[0]  = 0.001;   // x
    odom_msg.pose.covariance[7]  = 0.001;   // y
    odom_msg.pose.covariance[35] = 0.001;   // yaw
    odom_msg.twist.covariance[0]  = 0.001;
    odom_msg.twist.covariance[7]  = 0.001;
    odom_msg.twist.covariance[35] = 0.001;

    pub_odometry->publish(odom_msg);

    geometry_msgs::msg::TransformStamped odom_tf;
    odom_tf.header.stamp         = current_time;
    odom_tf.header.frame_id      = "odom";
    odom_tf.child_frame_id       = "base_footprint";
    odom_tf.transform.translation.x = pos_x;
    odom_tf.transform.translation.y = pos_y;
    odom_tf.transform.rotation       = odom_msg.pose.pose.orientation;
    tf_broadcaster->sendTransform(odom_tf);
  }

  Eigen::MatrixXd init_transform_matrix(int n_wheels)
  {
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n_wheels, 3);
    double del_angle = 360.0 / n_wheels;
    for (int i = 0; i < n_wheels; i++) {
      double a = (del_angle * i) * M_PI / 180.0;
      M(i, 0) = -sin(a) / r;
      M(i, 1) =  cos(a) / r;
      M(i, 2) =  R / r;
    }
    return M;
  }

  Eigen::MatrixXd pseudo_inverse(const Eigen::MatrixXd& A)
  {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd inv_sv = svd.singularValues();
    for (int i = 0; i < inv_sv.size(); ++i) {
      inv_sv(i) = (inv_sv(i) > 1e-8) ? 1.0 / inv_sv(i) : 0.0;
    }
    return svd.matrixV() * inv_sv.asDiagonal() * svd.matrixU().transpose();
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OmniKinematics>(3, ROBOT_RADIUS, WHEEL_RADIUS));
  rclcpp::shutdown();
  return 0;
}