import numpy as np
from core.pose import Pose
from core.map import OccupancyGrid
from core.scan import LaserScan
from config.params import *

def map_update(occupancy_map: OccupancyGrid, pose: Pose,
               real_scan: LaserScan) -> OccupancyGrid:
    gm = occupancy_map

    for i in range(real_scan.num_rays):
        r = real_scan.ranges[i]
        a_rel = real_scan.angles[i]
        if r < LASER_RANGE_MIN:
            continue

        # 相對角度加上當前 pose.theta
        a = a_rel + pose.theta

        x_hit = pose.x + r * np.cos(a)
        y_hit = pose.y + r * np.sin(a)

        gm._expand_if_needed(*gm.world_to_grid(pose.x, pose.y))
        gm._expand_if_needed(*gm.world_to_grid(x_hit, y_hit))

        gx0, gy0 = gm.world_to_grid(pose.x, pose.y)
        gx_end, gy_end = gm.world_to_grid(x_hit, y_hit)

        steps = max(abs(gx_end - gx0), abs(gy_end - gy0), 1)
        gxs = np.round(np.linspace(gx0, gx_end, steps + 1)).astype(int)
        gys = np.round(np.linspace(gy0, gy_end, steps + 1)).astype(int)

        valid = (gxs >= 0) & (gxs < gm.width) & (gys >= 0) & (gys < gm.height)
        gxs, gys = gxs[valid], gys[valid]

        if len(gxs) == 0:
            continue

        gm.data[gys[:-1], gxs[:-1]] = np.clip(
            gm.data[gys[:-1], gxs[:-1]] + MISS_LOG_ODDS, LOG_ODDS_MIN, LOG_ODDS_MAX)

        if r < LASER_RANGE_MAX:
            gm.data[gys[-1], gxs[-1]] = np.clip(
                gm.data[gys[-1], gxs[-1]] + HIT_LOG_ODDS, LOG_ODDS_MIN, LOG_ODDS_MAX)

    return gm