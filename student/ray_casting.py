import numpy as np
from core.pose import Pose
from core.map import OccupancyGrid
from core.scan import LaserScan
from config.params import *

def ray_casting(pose: Pose, occupancy_map: OccupancyGrid,
                num_rays: int = NUM_RAYS,
                fov: float = LASER_FOV) -> LaserScan:
    """
    Cast rays from the map to generate expected laser scan at given pose.

    Args:
        pose: Robot pose (Pose object)
        occupancy_map: Occupancy grid of the map
        num_rays: Number of rays to cast
        fov: Field of view in radians

    Returns:
        LaserScan object with predicted ranges

    TODO: For each ray angle:
          1. Calculate ray origin in world frame
          2. Cast ray and find distance to first occupied cell
          3. Store distance in ranges array
    """

    ranges = []

    angle_min = pose.theta - fov / 2
    angle_max = pose.theta + fov / 2
    angles = np.linspace(angle_min, angle_max, num_rays)

    gm = occupancy_map
    gx0, gy0 = gm.world_to_grid(pose.x, pose.y)
    gx0, gy0 = int(gx0), int(gy0)
    steps = int(LASER_RANGE_MAX / gm.resolution)

    t = np.arange(1, steps + 1)
    gxs = np.round(gx0 + np.outer(np.cos(angles), t)).astype(int)  # (num_rays, steps)
    gys = np.round(gy0 + np.outer(np.sin(angles), t)).astype(int)

    gxs = np.clip(gxs, 0, gm.width - 1)
    gys = np.clip(gys, 0, gm.height - 1)

    vals = gm.data[gys, gxs]  # (num_rays, steps)

    hit_mask = vals > 0.5
    first_hit = np.argmax(hit_mask, axis=1)
    has_hit = hit_mask.any(axis=1)
    ranges = np.where(has_hit, (first_hit + 1) * gm.resolution, LASER_RANGE_MAX)

    return LaserScan(ranges, angles=angles)

    return LaserScan(np.array(ranges), angles=angles)
