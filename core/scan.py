import numpy as np
from config.params import *
from core.pose import Pose

class LaserScan:
    """Laser scan data in polar coordinates"""

    def __init__(self, ranges: np.ndarray, angles: np.ndarray = None,
                 angle_min: float = -np.pi, angle_max: float = np.pi):
        self.ranges = ranges  # distance measurements
        self.num_rays = len(ranges)

        if angles is None:
            # Generate angles from angle_min to angle_max
            self.angles = np.linspace(angle_min, angle_max, self.num_rays)
        else:
            self.angles = angles

    def to_cartesian(self, pose: Pose) -> np.ndarray:
        """
        Convert laser scan to cartesian points in world frame
        Returns: Nx2 array of [x, y] points
        """
        points = []
        for range_val, angle in zip(self.ranges, self.angles):
            if LASER_RANGE_MIN <= range_val <= LASER_RANGE_MAX:
                # Point in sensor frame
                x_sensor = range_val * np.cos(angle)
                y_sensor = range_val * np.sin(angle)

                # Transform to world frame
                p_world = pose.transform_point(np.array([x_sensor, y_sensor]))
                points.append(p_world)

        return np.array(points) if points else np.array([]).reshape(0, 2)

    def to_json(self) -> dict:
        return {
            'ranges': self.ranges.tolist(),
            'angles': self.angles.tolist(),
            'num_rays': int(self.num_rays)
        }
