import numpy as np
from config.params import *
from core.pose import Pose
from core.scan import LaserScan

class LaserSimulator:
    def __init__(self):
        self.obstacles = []
        self._setup_simple_world()

    def _setup_simple_world(self):
        self.obstacles.append(('wall_left',   -5,  0, -5, 10))
        self.obstacles.append(('wall_right',   5,  0,  5, 10))
        self.obstacles.append(('wall_front',  -5, 10,  5, 10))
        self.obstacles.append(('wall_back',   -5,  0,  5,  0))
        self.obstacles.append(('obstacle_1',  -1,  2,  1,  2))
        self.obstacles.append(('obstacle_2',   0,  5,  2,  5))
        self.obstacles.append(('obstacle_3',   3,  3,  3,  7))
        # 新增
        self.obstacles.append(('obstacle_4',  -4,  4, -2,  4))
        self.obstacles.append(('obstacle_5',  -3,  6, -3,  9))
        self.obstacles.append(('obstacle_6',   1,  7,  4,  7))
        self.obstacles.append(('obstacle_7',  -2,  1, -2,  3))
        self.obstacles.append(('obstacle_8',   2,  1,  4,  1))
        self.obstacles.append(('obstacle_9',   1,  3,  3,  3))
        self.obstacles.append(('obstacle_10', -4,  8, -1,  8))

        
    def get_scan(self, pose: Pose, num_rays: int = NUM_RAYS,
                 fov: float = LASER_FOV) -> LaserScan:
        angle_min = pose.theta - fov / 2
        angle_max = pose.theta + fov / 2
        angles = np.linspace(angle_min, angle_max, num_rays)

        ray_dx = np.cos(angles)  # (num_rays,)
        ray_dy = np.sin(angles)

        ranges = np.full(num_rays, LASER_RANGE_MAX)

        for _, x1, y1, x2, y2 in self.obstacles:
            seg_dx = x2 - x1
            seg_dy = y2 - y1

            denom = ray_dx * seg_dy - ray_dy * seg_dx
            safe = np.abs(denom) > 1e-6

            t = np.where(safe,
                ((x1 - pose.x) * seg_dy - (y1 - pose.y) * seg_dx) / np.where(safe, denom, 1),
                -1.0)
            s = np.where(safe,
                ((x1 - pose.x) * ray_dy - (y1 - pose.y) * ray_dx) / np.where(safe, denom, 1),
                -1.0)

            hit = safe & (t > LASER_RANGE_MIN) & (s >= 0) & (s <= 1)
            ranges = np.where(hit & (t < ranges), t, ranges)

        noise = np.random.normal(0, IMU_NOISE_LINEAR, num_rays)
        return LaserScan(np.clip(ranges + noise, LASER_RANGE_MIN, LASER_RANGE_MAX), angles=angles)