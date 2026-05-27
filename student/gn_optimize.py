import numpy as np
from core.pose import Pose, normalize_angle
from core.map import OccupancyGrid
from config.params import *

def gn_optimize(pose_initial: Pose, occupancy_map: OccupancyGrid,
                matched_points, max_iterations: int = GN_MAX_ITERATIONS) -> Pose:


    pose = pose_initial.copy()
    res = occupancy_map.resolution


    # 把 LaserScan 轉成世界座標點
    world_pts = np.array([
        [pose_initial.x + r * np.cos(a), pose_initial.y + r * np.sin(a)]
        for r, a in zip(matched_points.ranges, matched_points.angles)
        if r < LASER_RANGE_MAX - 0.01
    ])


    if len(world_pts) == 0:
        return pose_initial.copy()

    # 轉到 robot frame（固定，不隨迭代變）
    robot_pts = np.array([
        pose_initial.inverse_transform_point(pt) for pt in world_pts
    ])


    count = 0
    for i in range(max_iterations):
        H = np.zeros((3, 3))
        b = np.zeros(3)

        for r, a_rel in zip(matched_points.ranges, matched_points.angles):
            if r >= LASER_RANGE_MAX - 0.01:
                continue
            a = a_rel + pose.theta  # 加上當前 slam pose 的 theta
            wx_cur = pose.x + r * np.cos(a)
            wy_cur = pose.y + r * np.sin(a)

            v   = occupancy_map.get_probability(wx_cur, wy_cur)
            vdx = occupancy_map.get_probability(wx_cur + res, wy_cur)
            vdy = occupancy_map.get_probability(wx_cur, wy_cur + res)

            dMdx = (vdx - v) / res
            dMdy = (vdy - v) / res

            # scan 點在 robot frame 的座標
            dWdTheta_x = -r * np.sin(a)
            dWdTheta_y =  r * np.cos(a)
            
            J = np.array([dMdx, dMdy, dMdx * dWdTheta_x + dMdy * dWdTheta_y])
            H += np.outer(J, J)
            b += J * (1.0 - v)

        try:
            delta = np.linalg.solve(H + 1e-6 * np.eye(3), b)
        except np.linalg.LinAlgError:
            break

        # clamp
        trans = np.linalg.norm(delta[:2])
        clamp_val = 0.5
        if trans > clamp_val:
            delta[:2] *= clamp_val / trans
        delta[2] = np.clip(delta[2], -clamp_val, clamp_val)

        pose.x    += delta[0]
        pose.y    += delta[1]
        pose.theta = normalize_angle(pose.theta + delta[2])


        if np.linalg.norm(delta) < GN_CONVERGENCE_THRESHOLD:
            break

        print(f"  iter={i} | delta_norm={np.linalg.norm(delta):.6f} | "
              f"H_det={np.linalg.det(H):.4f} | "
              f"b_norm={np.linalg.norm(b):.6f} | "
              f"mean_v={np.mean([occupancy_map.get_probability(*pose.transform_point(pr)) for pr in robot_pts]):.3f}")

    pose.theta = normalize_angle(pose.theta)

    print("Hello from gn_optimization5")

    return pose, count