import numpy as np
from core.pose import Pose
from core.map import OccupancyGrid
from core.scan import LaserScan
from config.params import *

def lidar_measurement_update(real_scan: LaserScan, predicted_scan: LaserScan,
                              pose: Pose, occupancy_map: OccupancyGrid) -> tuple:
    """
    Compare real and predicted scans to compute likelihood and matching points.

    Args:
        real_scan: Actual laser scan from simulator
        predicted_scan: Expected scan from ray casting
        pose: Current pose estimate
        occupancy_map: Occupancy grid

    Returns:
        Tuple of:
        - likelihood: float, log-likelihood of this pose given the scan
        - matched_points: Nx2 array of cartesian points from real scan

    TODO:
        1. Convert real scan to cartesian points (use real_scan.to_cartesian)
        2. For each point, find distance to nearest obstacle in map
        3. Compute likelihood as gaussian over these distances
        4. Return both likelihood and matched points
    """
    # Convert scan to cartesian points
    matched_points = real_scan.to_cartesian(pose)

    likelihood = 0.0
    # YOUR CODE HERE
    # For each matched point, calculate its contribution to likelihood

    matched_points = []
    for i in range(real_scan.num_rays):
        r_real = real_scan.ranges[i]
        r_pred = predicted_scan.ranges[i]
        if abs(r_real - r_pred) < 0.5:  # threshold
            a = real_scan.angles[i]
            wx = pose.x + r_real * np.cos(a)
            wy = pose.y + r_real * np.sin(a)
            matched_points.append([wx, wy])

    matched_points = np.array(matched_points) if matched_points else np.empty((0, 2))

    sigma = 0.1
    for px, py in matched_points:
        val = occupancy_map.get_value(px, py)
        prob = occupancy_map.to_probability(val)
        likelihood += np.log(prob + 1e-6)

    return likelihood, matched_points
