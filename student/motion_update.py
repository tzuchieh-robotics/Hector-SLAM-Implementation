import numpy as np
from core.pose import Pose, normalize_angle
from config.params import *

def motion_update(prev_pose: Pose, imu_data: dict) -> Pose:
    """
    Update robot pose based on IMU measurements.

    Args:
        prev_pose: Previous pose estimate (Pose object)
        imu_data: Dictionary containing:
            - 'linear_velocity': [vx, vy] in robot frame (m/s)
            - 'angular_velocity': omega in world frame (rad/s)
            - 'dt': time elapsed (seconds)

    Returns:
        Updated pose estimate (Pose object)

    TODO: Implement motion update using constant velocity model
    Hint: Update x, y based on linear velocity
          Update theta based on angular velocity
    """
    new_pose = prev_pose.copy()

    # YOUR CODE HERE
    # Update new_pose.x, new_pose.y, new_pose.theta


    vx, vy = imu_data['linear_velocity']
    angular_velocity = imu_data['angular_velocity']
    dt = imu_data['dt']

    new_pose.x += vx * np.cos(prev_pose.theta) * dt
    new_pose.y += vx * np.sin(prev_pose.theta) * dt
    
    new_pose.x -= vy * np.sin(prev_pose.theta) * dt
    new_pose.y += vy * np.cos(prev_pose.theta) * dt
    
    new_pose.theta += angular_velocity * dt
    new_pose.theta = normalize_angle(new_pose.theta)

    return new_pose
