// Copyright (c) 2025 Fumiya Ohnishi
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NAV2_REGULATED_PURE_PURSUIT_CONTROLLER__DYNAMIC_WINDOW_PURE_PURSUIT_FUNCTIONS_HPP_
#define NAV2_REGULATED_PURE_PURSUIT_CONTROLLER__DYNAMIC_WINDOW_PURE_PURSUIT_FUNCTIONS_HPP_

#include <string>
#include <vector>
#include <algorithm>
#include <tuple>
#include <utility>
#include <limits>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <ctime>
#include <sstream>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "sensor_msgs/msg/battery_state.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "tf2/utils.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

namespace nav2_regulated_pure_pursuit_controller
{

namespace dynamic_window_pure_pursuit
{

struct DynamicWindowBounds
{
  double max_linear_vel;
  double min_linear_vel;
  double max_angular_vel;
  double min_angular_vel;
};

struct Transform2DData
{
  bool valid;
  double x;
  double y;
  double yaw;
};

struct CsvLogState
{
  std::ofstream csv_stream;
  bool initialized = false;
  bool header_written = false;
  std::string csv_path;
};

inline CsvLogState & getCsvLogState()
{
  static CsvLogState state;
  return state;
}

inline std::filesystem::path getLogDir()
{
  static bool fallback_warned = false;
  try {
    const auto pkg_share =
      ament_index_cpp::get_package_share_directory("dwpp_test_simulation");
    std::filesystem::path log_dir = std::filesystem::path(pkg_share) / "data";
    std::error_code ec;
    std::filesystem::create_directories(log_dir, ec);
    return log_dir;
  } catch (const std::exception & e) {
    if (!fallback_warned) {
      RCLCPP_WARN(
        rclcpp::get_logger("dynamic_window_pure_pursuit"),
        "Failed to locate dwpp_test_simulation package (%s). Falling back to /tmp.",
        e.what());
      fallback_warned = true;
    }
    return std::filesystem::path("/tmp");
  }
}

inline std::string makeTimestampedCsvPath(const rclcpp::Time & stamp)
{
  const auto seconds = static_cast<std::time_t>(stamp.seconds());
  const auto nsec = static_cast<uint32_t>(stamp.nanoseconds() % 1000000000LL);
  std::tm tm_snapshot{};
  if (const std::tm * tm_ptr = std::localtime(&seconds)) {
    tm_snapshot = *tm_ptr;
  }

  std::ostringstream name;
  name << "dynamic_window_pure_pursuit_log_"
       << std::put_time(&tm_snapshot, "%Y%m%d_%H%M%S")
       << "_" << std::setw(9) << std::setfill('0') << nsec
       << ".csv";
  return (getLogDir() / name.str()).string();
}

inline void requestNewLogFile(const rclcpp::Time & stamp)
{
  auto & state = getCsvLogState();
  if (state.csv_stream.is_open()) {
    state.csv_stream.close();
  }
  state.csv_path = makeTimestampedCsvPath(stamp);
  state.initialized = false;
  state.header_written = false;
}

/**
 * @brief Compute the dynamic window (feasible velocity bounds) based on the current speed and the given velocity and acceleration constraints.
 * @param current_speed     Current linear and angular velocity of the robot
 * @param max_linear_vel    Maximum allowable linear velocity
 * @param min_linear_vel    Minimum allowable linear velocity
 * @param max_angular_vel   Maximum allowable angular velocity
 * @param min_angular_vel   Minimum allowable angular velocity
 * @param max_linear_accel  Maximum allowable linear acceleration
 * @param max_linear_decel  Maximum allowable linear deceleration
 * @param max_angular_accel Maximum allowable angular acceleration
 * @param max_angular_decel Maximum allowable angular deceleration
 * @param dt                Control duration
 * @return                  Computed dynamic window's velocity bounds
 */
inline DynamicWindowBounds computeDynamicWindow(
  const geometry_msgs::msg::Twist & current_speed,
  const double & max_linear_vel,
  const double & min_linear_vel,
  const double & max_angular_vel,
  const double & min_angular_vel,
  const double & max_linear_accel,
  const double & max_linear_decel,
  const double & max_angular_accel,
  const double & max_angular_decel,
  const double & dt
)
{
  DynamicWindowBounds dynamic_window;
  constexpr double Eps = 1e-3;

  // function to compute dynamic window for a single dimension
  auto compute_window = [&](const double & current_vel, const double & max_vel,
    const double & min_vel, const double & max_accel, const double & max_decel)
    {
      double candidate_max_vel = 0.0;
      double candidate_min_vel = 0.0;

      if (current_vel > Eps) {
      // if the current velocity is positive, acceleration means an increase in speed
        candidate_max_vel = current_vel + max_accel * dt;
        candidate_min_vel = current_vel + max_decel * dt;
      } else if (current_vel < -Eps) {
      // if the current velocity is negative, acceleration means a decrease in speed
        candidate_max_vel = current_vel - max_decel * dt;
        candidate_min_vel = current_vel - max_accel * dt;
      } else {
      // if the current velocity is zero, allow acceleration in both directions.
        candidate_max_vel = current_vel + max_accel * dt;
        candidate_min_vel = current_vel - max_accel * dt;
      }

    // clip to max/min velocity limits
      double dynamic_window_max_vel = std::min(candidate_max_vel, max_vel);
      double dynamic_window_min_vel = std::max(candidate_min_vel, min_vel);
      return std::make_tuple(dynamic_window_max_vel, dynamic_window_min_vel);
    };

  // linear velocity
  std::tie(dynamic_window.max_linear_vel,
        dynamic_window.min_linear_vel) = compute_window(current_speed.linear.x,
                 max_linear_vel, min_linear_vel,
                 max_linear_accel, max_linear_decel);

  // angular velocity
  std::tie(dynamic_window.max_angular_vel,
        dynamic_window.min_angular_vel) = compute_window(current_speed.angular.z,
                 max_angular_vel, min_angular_vel,
                 max_angular_accel, max_angular_decel);

  return dynamic_window;
}

inline bool evaluateVelocityConstraints(
  const geometry_msgs::msg::Twist & next_cmd_vel,
  const geometry_msgs::msg::Twist & current_cmd_vel,
  const double & max_linear_vel,
  const double & min_linear_vel,
  const double & max_angular_vel,
  const double & min_angular_vel,
  const double & max_linear_accel,
  const double & max_linear_decel,
  const double & max_angular_accel,
  const double & max_angular_decel,
  const double & dt)
{
  constexpr double Eps = 1e-2;

  // function to evaluate velocity constraints for a single dimension
  auto evaluate_velocity =
    [&](const double & current_vel, const double & last_vel, const double & max_vel,
    const double & min_vel,
    const double & max_accel, const double & max_decel)
    {
      double candidate_max_vel = 0.0;
      double candidate_min_vel = 0.0;

      if (last_vel > Eps) {
        // if the last velocity is positive, acceleration means an increase in speed
        candidate_max_vel = last_vel + max_accel * dt;
        candidate_min_vel = last_vel + max_decel * dt;
      } else if (last_vel < -Eps) {
        // if the last velocity is negative, acceleration means a decrease in speed
        candidate_max_vel = last_vel - max_decel * dt;
        candidate_min_vel = last_vel - max_accel * dt;
      } else {
        // if the last velocity is zero, allow acceleration in both directions.
        candidate_max_vel = last_vel + max_accel * dt;
        candidate_min_vel = last_vel - max_accel * dt;
      }

      // clip to max/min velocity limits
      candidate_max_vel = std::min(candidate_max_vel, max_vel);
      candidate_min_vel = std::max(candidate_min_vel, min_vel);

      // check whether current_vel is within [candidate_min_vel, candidate_max_vel]
      if (current_vel > candidate_max_vel + Eps || current_vel < candidate_min_vel - Eps) {
        return true;  // violation
      } else {
        return false;  // no violation
      }
    };
  // linear velocity
  bool linear_violation = evaluate_velocity(
    next_cmd_vel.linear.x,
    current_cmd_vel.linear.x,
    max_linear_vel, min_linear_vel,
    max_linear_accel, max_linear_decel);
  // angular velocity
  bool angular_violation = evaluate_velocity(
    next_cmd_vel.angular.z,
    current_cmd_vel.angular.z,
    max_angular_vel, min_angular_vel,
    max_angular_accel, max_angular_decel);

  return linear_violation || angular_violation;
}

/**
 * @brief                        Apply regulated linear velocity to the dynamic window
 * @param regulated_linear_vel   Regulated linear velocity
 * @param dynamic_window         Dynamic window to be regulated
 */
inline void applyRegulationToDynamicWindow(
  const double & regulated_linear_vel,
  DynamicWindowBounds & dynamic_window)
{
  double regulated_dynamic_window_max_linear_vel;
  double regulated_dynamic_window_min_linear_vel;

  // Extract the portion of the dynamic window that lies within the range [0, regulated_linear_vel]
  if (regulated_linear_vel >= 0.0) {
    regulated_dynamic_window_max_linear_vel = std::min(
      dynamic_window.max_linear_vel, regulated_linear_vel);
    regulated_dynamic_window_min_linear_vel = std::max(
      dynamic_window.min_linear_vel, 0.0);
  } else {
    regulated_dynamic_window_max_linear_vel = std::min(
      dynamic_window.max_linear_vel, 0.0);
    regulated_dynamic_window_min_linear_vel = std::max(
      dynamic_window.min_linear_vel, regulated_linear_vel);
  }

  if (regulated_dynamic_window_max_linear_vel < regulated_dynamic_window_min_linear_vel) {
    // No valid portion of the dynamic window remains after applying the regulation
    if (regulated_dynamic_window_min_linear_vel > 0.0) {
      // If the dynamic window is entirely in the positive range,
      // collapse both bounds to dynamic_window_min_linear_vel
      regulated_dynamic_window_max_linear_vel = regulated_dynamic_window_min_linear_vel;
    } else {
      // If the dynamic window is entirely in the negative range,
      // collapse both bounds to dynamic_window_max_linear_vel
      regulated_dynamic_window_min_linear_vel = regulated_dynamic_window_max_linear_vel;
    }
  }

  dynamic_window.max_linear_vel = regulated_dynamic_window_max_linear_vel;
  dynamic_window.min_linear_vel = regulated_dynamic_window_min_linear_vel;
}


/**
 * @brief                Compute the optimal velocity to follow the path within the dynamic window
 * @param dynamic_window Dynamic window defining feasible velocity bounds
 * @param curvature      Curvature of the path to follow
 * @param sign           Velocity sign (forward or backward)
 * @return               Optimal linear and angular velocity
 */
inline std::tuple<double, double> computeOptimalVelocityWithinDynamicWindow(
  const DynamicWindowBounds & dynamic_window,
  const double & curvature,
  const double & sign
)
{
  double optimal_linear_vel;
  double optimal_angular_vel;

  // consider linear_vel - angular_vel space (horizontal and vertical axes respectively)
  // Select the closest point to the line
  // angular_vel = curvature * linear_vel within the dynamic window.
  // If multiple points are equally close, select the one with the largest linear_vel.

  // When curvature == 0, the line is angular_vel = 0
  if (abs(curvature) < 1e-3) {
    // linear velocity
    if (sign >= 0.0) {
      // If moving forward, select the max linear vel
      optimal_linear_vel = dynamic_window.max_linear_vel;
    } else {
      // If moving backward, select the min linear vel
      optimal_linear_vel = dynamic_window.min_linear_vel;
    }

    // angular velocity
    // If the line angular_vel = 0 intersects the dynamic window,angular_vel = 0.0
    if (dynamic_window.min_angular_vel <= 0.0 && 0.0 <= dynamic_window.max_angular_vel) {
      optimal_angular_vel = 0.0;
    } else {
      // If not, select angular vel within dynamic window closest to 0
      if (std::abs(dynamic_window.min_angular_vel) <= std::abs(dynamic_window.max_angular_vel)) {
        optimal_angular_vel = dynamic_window.min_angular_vel;
      } else {
        optimal_angular_vel = dynamic_window.max_angular_vel;
      }
    }
    return std::make_tuple(optimal_linear_vel, optimal_angular_vel);
  }

  // When the dynamic window and the line angular_vel = curvature * linear_vel intersect,
  // select the intersection point that yields the highest linear velocity.

  // List the four candidate intersection points
  std::pair<double, double> candidates[] = {
    {dynamic_window.min_linear_vel, curvature * dynamic_window.min_linear_vel},
    {dynamic_window.max_linear_vel, curvature * dynamic_window.max_linear_vel},
    {dynamic_window.min_angular_vel / curvature, dynamic_window.min_angular_vel},
    {dynamic_window.max_angular_vel / curvature, dynamic_window.max_angular_vel}
  };

  double best_linear_vel = -std::numeric_limits<double>::infinity() * sign;
  double best_angular_vel = 0.0;

  for (auto [linear_vel, angular_vel] : candidates) {
    // Check whether the candidate lies within the dynamic window
    if (linear_vel >= dynamic_window.min_linear_vel &&
      linear_vel <= dynamic_window.max_linear_vel &&
      angular_vel >= dynamic_window.min_angular_vel &&
      angular_vel <= dynamic_window.max_angular_vel)
    {
      // Select the candidate with the largest linear velocity (considering moving direction)
      if (linear_vel * sign > best_linear_vel * sign) {
        best_linear_vel = linear_vel;
        best_angular_vel = angular_vel;
      }
    }
  }

  // If best_linear_vel was updated, it means that a valid intersection exists
  if (best_linear_vel != -std::numeric_limits<double>::infinity() * sign) {
    optimal_linear_vel = best_linear_vel;
    optimal_angular_vel = best_angular_vel;
    return std::make_tuple(optimal_linear_vel, optimal_angular_vel);
  }

  // When the dynamic window and the line angular_vel = curvature * linear_vel have no intersection,
  // select the point within the dynamic window that is closest to the line.

  // Because the dynamic window is a convex region,
  // the closest point must be one of its four corners.
  const std::array<std::array<double, 2>, 4> corners = {{
    {dynamic_window.min_linear_vel, dynamic_window.min_angular_vel},
    {dynamic_window.min_linear_vel, dynamic_window.max_angular_vel},
    {dynamic_window.max_linear_vel, dynamic_window.min_angular_vel},
    {dynamic_window.max_linear_vel, dynamic_window.max_angular_vel}
  }};

  // Compute the distance from a point (linear_vel, angular_vel)
  // to the line angular_vel = curvature * linear_vel
  const double denom = std::sqrt(curvature * curvature + 1.0);
  auto compute_dist = [&](const std::array<double, 2> & corner) -> double {
      return std::abs(curvature * corner[0] - corner[1]) / denom;
    };

  double closest_dist = std::numeric_limits<double>::infinity();
  best_linear_vel = -std::numeric_limits<double>::infinity() * sign;
  best_angular_vel = 0.0;

  for (const auto & corner : corners) {
    const double dist = compute_dist(corner);
    // Update if this corner is closer to the line,
    // or equally close but has a larger linear velocity (considering moving direction)
    if (dist < closest_dist ||
      (std::abs(dist - closest_dist) <= 1e-3 && corner[0] * sign > best_linear_vel * sign))
    {
      closest_dist = dist;
      best_linear_vel = corner[0];
      best_angular_vel = corner[1];
    }
  }

  optimal_linear_vel = best_linear_vel;
  optimal_angular_vel = best_angular_vel;

  return std::make_tuple(optimal_linear_vel, optimal_angular_vel);
}

inline void recordData(
  const double & curvature,
  const geometry_msgs::msg::Twist & current_cmd_vel,
  const geometry_msgs::msg::Twist & next_cmd_vel,
  const double & regulated_linear_vel,
  DynamicWindowBounds & dynamic_window,
  const double & max_linear_vel,
  const double & min_linear_vel,
  const double & max_angular_vel,
  const double & min_angular_vel,
  const double & max_linear_accel,
  const double & max_linear_decel,
  const double & max_angular_accel,
  const double & max_angular_decel,
  const double & dt,
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & speed,
  const bool & constraints_violation_flag,
  const sensor_msgs::msg::BatteryState::SharedPtr battery_state,
  const sensor_msgs::msg::Imu::SharedPtr imu,
  const Transform2DData & map_to_odom,
  const Transform2DData & map_to_base
)
{
  constexpr double Eps = 1e-3;

  // calc actual velocity
  auto calc_actual_velocity =
    [&](const double & current_vel, const double & last_vel,
    const double & max_vel, const double & min_vel,
    const double & max_accel, const double & max_decel)
    {
      double candidate_max_vel = 0.0;
      double candidate_min_vel = 0.0;

      if (last_vel > Eps) {
        // if the last velocity is positive, acceleration means an increase in speed
        candidate_max_vel = last_vel + max_accel * dt;
        candidate_min_vel = last_vel + max_decel * dt;
      } else if (last_vel < -Eps) {
        // if the last velocity is negative, acceleration means a decrease in speed
        candidate_max_vel = last_vel - max_decel * dt;
        candidate_min_vel = last_vel - max_accel * dt;
      } else {
        // if the last velocity is zero, allow acceleration in both directions.
        candidate_max_vel = last_vel + max_accel * dt;
        candidate_min_vel = last_vel - max_accel * dt;
      }

      // clip to max/min velocity limits
      candidate_max_vel = std::min(candidate_max_vel, max_vel);
      candidate_min_vel = std::max(candidate_min_vel, min_vel);

      // check whether current_vel is within [candidate_min_vel, candidate_max_vel]
      if (current_vel > candidate_max_vel + Eps) {
        return candidate_max_vel;  // violation
      } else if (current_vel < candidate_min_vel - Eps) {
        return candidate_min_vel;  // violation
      } else {
        return current_vel;  // no violation
      }
    };

  // linear velocity
  double actual_linear_vel = calc_actual_velocity(
    next_cmd_vel.linear.x,
    current_cmd_vel.linear.x,
    max_linear_vel, min_linear_vel,
    max_linear_accel, max_linear_decel);
  // angular velocity
  double actual_angular_vel = calc_actual_velocity(
    next_cmd_vel.angular.z,
    current_cmd_vel.angular.z,
    max_angular_vel, min_angular_vel,
    max_angular_accel, max_angular_decel);

  // record by csv
  auto & log_state = getCsvLogState();
  if (!log_state.initialized) {
    log_state.initialized = true;
    if (log_state.csv_path.empty()) {
      log_state.csv_path = (getLogDir() / "dynamic_window_pure_pursuit_log.csv").string();
    }
    log_state.csv_stream.open(log_state.csv_path, std::ios::app);
    if (!log_state.csv_stream.is_open()) {
      RCLCPP_WARN(
        rclcpp::get_logger("dynamic_window_pure_pursuit"),
        "Failed to open %s for logging.", log_state.csv_path.c_str());
      return;
    }
    log_state.header_written = log_state.csv_stream.tellp() > 0;
  }

  if (!log_state.csv_stream.is_open()) {
    return;
  }

  if (!log_state.header_written) {
    log_state.csv_stream << "sec,nsec,odom_base_x,odom_base_y,odom_base_yaw,"
      "map_odom_x,map_odom_y,map_odom_yaw,"
      "map_base_x,map_base_y,map_base_yaw,"
      "v_real,w_real,v_now,w_now,v_cmd,w_cmd,v_nav,w_nav,"
      "velocity_violation,"
      "battery_v,battery_i,battery_percent,"
      "imu_ax,imu_ay,imu_az,imu_vx,imu_vy,imu_vz,"
      "curvature,dw_v_max,dw_v_min,dw_w_max,dw_w_min,v_reg\n";
    log_state.header_written = true;
  }

  // get time object
  const auto now = rclcpp::Clock(RCL_SYSTEM_TIME).now();
  const int32_t sec = static_cast<int32_t>(now.seconds());
  const uint32_t nsec = static_cast<uint32_t>(now.nanoseconds() % 1000000000LL);

  log_state.csv_stream  << sec << ","
                        << nsec << ","
                        << std::fixed << std::setprecision(6)
                        << pose.pose.position.x << ','
                        << pose.pose.position.y << ','
                        << tf2::getYaw(pose.pose.orientation) << ','
                        << (map_to_odom.valid ? map_to_odom.x :
  std::numeric_limits<double>::quiet_NaN()) << ','
                        << (map_to_odom.valid ? map_to_odom.y :
  std::numeric_limits<double>::quiet_NaN()) << ','
                        << (map_to_odom.valid ? map_to_odom.yaw :
  std::numeric_limits<double>::quiet_NaN()) << ','
                        << (map_to_base.valid ? map_to_base.x :
  std::numeric_limits<double>::quiet_NaN()) << ','
                        << (map_to_base.valid ? map_to_base.y :
  std::numeric_limits<double>::quiet_NaN()) << ','
                        << (map_to_base.valid ? map_to_base.yaw :
  std::numeric_limits<double>::quiet_NaN()) << ','
                        << speed.linear.x << ',' << speed.angular.z << ','
                        << current_cmd_vel.linear.x << ',' << current_cmd_vel.angular.z << ','
                        << next_cmd_vel.linear.x << ',' << next_cmd_vel.angular.z << ','
                        << actual_linear_vel << ',' << actual_angular_vel << ','
                        << (constraints_violation_flag ? 1 : 0) << ','
                        << (battery_state ? battery_state->voltage :
  std::numeric_limits<double>::quiet_NaN()) << ','
                        << (battery_state ? battery_state->current :
  std::numeric_limits<double>::quiet_NaN()) << ','
                        << (battery_state ? battery_state->percentage :
  std::numeric_limits<double>::quiet_NaN()) << ','
                        << (imu ? imu->linear_acceleration.x :
  std::numeric_limits<double>::quiet_NaN()) << ','
                        << (imu ? imu->linear_acceleration.y :
  std::numeric_limits<double>::quiet_NaN()) << ','
                        << (imu ? imu->linear_acceleration.z :
  std::numeric_limits<double>::quiet_NaN()) << ','
                        << (imu ? imu->angular_velocity.x :
  std::numeric_limits<double>::quiet_NaN()) << ','
                        << (imu ? imu->angular_velocity.y :
  std::numeric_limits<double>::quiet_NaN()) << ','
                        << (imu ? imu->angular_velocity.z :
  std::numeric_limits<double>::quiet_NaN()) << ','
                        << curvature << ','
                        << dynamic_window.max_linear_vel << ',' <<
    dynamic_window.min_linear_vel << ','
                        << dynamic_window.max_angular_vel << ',' <<
    dynamic_window.min_angular_vel << ','
                        << regulated_linear_vel
                        << '\n';
  log_state.csv_stream.flush();
}

}  // namespace dynamic_window_pure_pursuit

}  // namespace nav2_regulated_pure_pursuit_controller

#endif  // NAV2_REGULATED_PURE_PURSUIT_CONTROLLER__DYNAMIC_WINDOW_PURE_PURSUIT_FUNCTIONS_HPP_
