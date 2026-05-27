import numpy as np
from config.params import *

class Pose:
    """Represent robot pose in world frame"""
    def __init__(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        self.x = x
        self.y = y
        self.theta = theta

    def to_transform_matrix(self) -> np.ndarray:
        """Convert to 3x3 homogeneous transformation matrix"""
        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        T = np.array([
            [cos_t, -sin_t, self.x],
            [sin_t,  cos_t, self.y],
            [0,      0,     1]
        ])
        return T

    @staticmethod
    def from_transform_matrix(T: np.ndarray) -> 'Pose':
        """Create Pose from 3x3 transformation matrix"""
        x = T[0, 2]
        y = T[1, 2]
        theta = np.arctan2(T[1, 0], T[0, 0])
        return Pose(x, y, theta)

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """Transform a point from robot frame to world frame"""
        # point: [x, y] in robot frame
        T = self.to_transform_matrix()
        p_homo = np.array([point[0], point[1], 1])
        p_world = T @ p_homo
        return p_world[:2]

    def inverse_transform_point(self, point: np.ndarray) -> np.ndarray:
        """Transform a point from world frame to robot frame"""
        # point: [x, y] in world frame
        T = self.to_transform_matrix()
        T_inv = np.linalg.inv(T)
        p_homo = np.array([point[0], point[1], 1])
        p_robot = T_inv @ p_homo
        return p_robot[:2]

    def __repr__(self):
        return f"Pose(x={self.x:.3f}, y={self.y:.3f}, theta={self.theta:.3f})"

    def copy(self):
        return Pose(self.x, self.y, self.theta)

def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle
