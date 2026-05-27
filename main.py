"""
Hector SLAM Local Simulation
W/S: forward/back  A/D: rotate  R: reset  ESC: quit
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.dirname(__file__))
from core import Pose, OccupancyGrid, LaserSimulator
from config.params import (
    KEYBOARD_SCALE, ROTATION_SCALE, FINE_RESOLUTION, MAP_ORIGIN,
    MIN_TRANSLATION_FOR_MAP_UPDATE, MIN_ROTATION_FOR_MAP_UPDATE,
    COARSE_GN_MAX_ITERATIONS, GN_MAX_ITERATIONS,
    IMU_NOISE_LINEAR, IMU_NOISE_ANGULAR
)

# ── Try importing student functions (ok if not implemented) ───────────────────
def _not_implemented(*args, **kwargs):
    return args[0] if args else None

try:
    from student.map_update import map_update
except Exception:
    map_update = None

try:
    from student.down_sampling import down_sample
except Exception:
    down_sample = None

try:
    from student.gn_optimize import gn_optimize
except Exception:
    gn_optimize = None

try:
    from student.ray_casting import ray_casting
except Exception:
    ray_casting = None

try:
    from student.lidar_measurement import lidar_measurement_update
except Exception:
    lidar_measurement_update = None

# ── World ─────────────────────────────────────────────────────────────────────
WALLS = [
    # 外牆
    (-5, 0,  -5, 10),
    ( 5, 0,   5, 10),
    (-5, 10,  5, 10),
    (-5, 0,   5,  0),
    # 內部障礙物（加更多）
    (-1, 2,   1,  2),
    ( 0, 5,   2,  5),
    ( 3, 3,   3,  7),
    # 新增
    (-4, 4,  -2,  4),   # 左邊橫牆
    (-3, 6,  -3,  9),   # 左邊縱牆
    ( 1, 7,   4,  7),   # 右上橫牆
    (-2, 1,  -2,  3),   # 左下縱牆
    ( 2, 1,   4,  1),   # 右下橫牆
    ( 1, 3,   3,  3),   # 中間橫牆
    (-4, 8,  -1,  8),   # 左上橫牆
]

# ── State ─────────────────────────────────────────────────────────────────────
gt_pose    = Pose(0, 2, 0)   # ground truth (keyboard)
slam_pose  = Pose(0, 2, 0)   # SLAM estimate (noisy + GN corrected)
prev_pose  = Pose(0, 2, 0)   # last map update pose
fine_map   = OccupancyGrid(FINE_RESOLUTION, origin=MAP_ORIGIN)
coarse_map = fine_map.downsample()
simulator  = LaserSimulator()
step       = 0
keys       = set()

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax_map, ax_info) = plt.subplots(1, 2, figsize=(13, 7))
fig.patch.set_facecolor('#080d1a')
fig.canvas.manager.set_window_title('Hector SLAM — Local Simulation')

for ax in (ax_map, ax_info):
    ax.set_facecolor('#080d1a')
    ax.tick_params(colors='#00ffe0')
    for sp in ax.spines.values():
        sp.set_edgecolor('#00ffe0'); sp.set_alpha(0.3)

ax_map.set_aspect('equal')
ax_map.set_title('Map', color='#00ffe0')
ax_map.grid(True, color='white', alpha=0.04)
ax_map.set_xlim(-8, 8)
ax_map.set_ylim(-2, 12)
ax_info.axis('off')
scan_scatter = ax_map.scatter([], [], s=2, c="#ff0000", alpha=0.4, zorder=3)

# GT walls
for x1, y1, x2, y2 in WALLS:
    ax_map.plot([x1, x2], [y1, y2], color='#00ffe0', lw=2, alpha=0.5, zorder=1)

# Occupancy map image
from matplotlib.colors import LinearSegmentedColormap

colors = [
    (0.85, 0.85, 0.85),  # 淡灰（free）
    (0.5, 0.5, 0.5),     # 中灰（unknown）
    (0.0, 0.0, 0.0),     # 黑（occupied）
]
cmap = LinearSegmentedColormap.from_list('custom', colors)

map_img = ax_map.imshow(
    np.zeros((200, 200)), origin='lower',
    extent=[-8, 8, -2, 12], vmin=-2, vmax=3.5,
    cmap=cmap, alpha=0.85, interpolation='none', zorder=2
)


# GT robot (green)
gt_dot, = ax_map.plot([], [], 'o', color='#00ff88', ms=7,
                      markeredgecolor='white', mew=1, zorder=4, label='GT')
gt_arrow = ax_map.annotate('', xy=(0.3, 0), xytext=(0, 0),
    arrowprops=dict(arrowstyle='->', color='#00ff88', lw=1.5), zorder=4)

# SLAM robot (red)
slam_dot, = ax_map.plot([], [], 'o', color='#ff4444', ms=7,
                        markeredgecolor='white', mew=1, zorder=5, label='SLAM')
slam_arrow = ax_map.annotate('', xy=(0.3, 0), xytext=(0, 0),
    arrowprops=dict(arrowstyle='->', color='#ff4444', lw=1.5), zorder=5)

ax_map.legend(loc='upper right', facecolor='#0a1020', edgecolor='#00ffe0',
              labelcolor='white', fontsize=8)

# Info
info = ax_info.text(0.05, 0.95, '', transform=ax_info.transAxes,
    color='#00ffe0', fontsize=10, va='top', fontfamily='monospace',
    bbox=dict(boxstyle='round', fc='#0a1020', ec='#00ffe0', alpha=0.9))

status = ax_info.text(0.05, 0.45, '', transform=ax_info.transAxes,
    color='#888', fontsize=9, va='top', fontfamily='monospace',
    bbox=dict(boxstyle='round', fc='#0a1020', ec='#333', alpha=0.8))

ax_info.text(0.05, 0.18,
    "CONTROLS\n─────────────\nW/S  forward/back\nA/D  rotate\nR    reset\nESC  quit",
    transform=ax_info.transAxes, color='#5affdc', fontsize=10,
    va='top', fontfamily='monospace',
    bbox=dict(boxstyle='round', fc='#0a1020', ec='#005544', alpha=0.8))

# ── Update loop ───────────────────────────────────────────────────────────────
def update(frame):
    global gt_arrow, slam_arrow, gt_pose, slam_pose, prev_pose
    global fine_map, coarse_map, step

    fwd = (1 if 'w' in keys else 0) - (1 if 's' in keys else 0)
    rot = (1 if 'a' in keys else 0) - (1 if 'd' in keys else 0)

    if fwd or rot:
        step += 1
        dt = 0.05
        vx    = fwd * KEYBOARD_SCALE / dt
        omega = rot * ROTATION_SCALE / dt
        c, s  = np.cos(gt_pose.theta), np.sin(gt_pose.theta)

        # 1. Ground truth motion (clean)
        gt_pose.x     += c * vx * dt
        gt_pose.y     += s * vx * dt
        gt_pose.theta  = (gt_pose.theta + omega * dt + np.pi) % (2*np.pi) - np.pi

        # 2. SLAM motion (noisy) — 用 slam_pose 自己的 theta
        cs = np.cos(slam_pose.theta)
        ss = np.sin(slam_pose.theta)
        slam_pose.x     += cs * vx * dt + np.random.normal(0, IMU_NOISE_LINEAR)
        slam_pose.y     += ss * vx * dt + np.random.normal(0, IMU_NOISE_LINEAR)
        slam_pose.theta  = (slam_pose.theta + omega * dt + np.random.normal(0, IMU_NOISE_ANGULAR) + np.pi) % (2*np.pi) - np.pi

        # 3. LiDAR scan from GT pose
        real_scan = simulator.get_scan(gt_pose)

        # 建立相對角度的 scan 給 GN 用
        from core.scan import LaserScan
        # 外面改成
        relative_scan = LaserScan(
            ranges=real_scan.ranges,
            angles=real_scan.angles - slam_pose.theta  # ← 用 slam_pose.theta
        )

        # 4. Ray casting (if implemented)
        predicted_scan = None
        if ray_casting is not None:
            try:
                predicted_scan = ray_casting(slam_pose, coarse_map)
            except Exception:
                pass
        
        from config.params import LASER_RANGE_MAX
        pts = []
        for r, a_rel in zip(relative_scan.ranges, relative_scan.angles):
            if r >= LASER_RANGE_MAX - 0.01:
                continue
            a = a_rel + slam_pose.theta
            x = slam_pose.x + r * np.cos(a)
            y = slam_pose.y + r * np.sin(a)
            pts.append([x, y])
        scan_scatter.set_offsets(np.array(pts))

        # 6. GN optimize (if implemented, only after map has content)
        has_map = (step > 20 and
           np.sum(fine_map.data > 0.5) > 100)


        if gn_optimize is not None and has_map:
            try:
                pose_coarse, count = gn_optimize(slam_pose, coarse_map, relative_scan, COARSE_GN_MAX_ITERATIONS)
                slam_pose, count   = gn_optimize(pose_coarse, fine_map, relative_scan, GN_MAX_ITERATIONS)

            except Exception as e:
                import traceback
                traceback.print_exc()

        # 7. Map update when moved enough (if implemented)
        dx     = slam_pose.x - prev_pose.x
        dy     = slam_pose.y - prev_pose.y
        dtheta = abs(slam_pose.theta - prev_pose.theta)
        moved  = np.sqrt(dx**2 + dy**2) > MIN_TRANSLATION_FOR_MAP_UPDATE or \
                 dtheta > MIN_ROTATION_FOR_MAP_UPDATE

        if moved:
            if map_update is not None:
                try:
                    fine_map = map_update(fine_map, slam_pose, relative_scan)
                except Exception:
                    pass
            if down_sample is not None:
                try:
                    coarse_map = down_sample(fine_map)
                except Exception:
                    pass
            prev_pose = slam_pose.copy()

            # Update map image
            map_img.set_data(fine_map.data)
            map_img.set_extent([fine_map.origin_x, fine_map.origin_x + fine_map.width  * fine_map.resolution,
                                 fine_map.origin_y, fine_map.origin_y + fine_map.height * fine_map.resolution])
            ax_map.set_xlim(fine_map.origin_x, fine_map.origin_x + fine_map.width * fine_map.resolution)
            ax_map.set_ylim(fine_map.origin_y, fine_map.origin_y + fine_map.height * fine_map.resolution)


    # Update GT robot marker
    gx, gy, gth = gt_pose.x, gt_pose.y, gt_pose.theta
    gt_dot.set_data([gx], [gy])
    gt_arrow.remove()
    gt_arrow = ax_map.annotate('',
        xy=(gx + 0.5*np.cos(gth), gy + 0.5*np.sin(gth)),
        xytext=(gx, gy),
        arrowprops=dict(arrowstyle='->', color='#00ff88', lw=1.5), zorder=4)

    # Update SLAM robot marker
    sx, sy, sth = slam_pose.x, slam_pose.y, slam_pose.theta
    slam_dot.set_data([sx], [sy])
    slam_arrow.remove()
    slam_arrow = ax_map.annotate('',
        xy=(sx + 0.5*np.cos(sth), sy + 0.5*np.sin(sth)),
        xytext=(sx, sy),
        arrowprops=dict(arrowstyle='->', color='#ff4444', lw=1.5), zorder=5)

    # Status of each module
    def check(fn, name):
        return f"  {name:20s} {'✓' if fn is not None else '○ not implemented'}"

    status.set_text(
        "MODULES\n" +
        check(map_update,              'map_update') + '\n' +
        check(down_sample,             'down_sample') + '\n' +
        check(ray_casting,             'ray_casting') + '\n' +
        check(gn_optimize,             'gn_optimize') + '\n' +
        check(lidar_measurement_update,'lidar_measurement')
    )

    info.set_text(
        f"GT POSE\n"
        f"  x : {gx:+.3f} m\n"
        f"  y : {gy:+.3f} m\n"
        f"  θ : {np.degrees(gth):+.1f}°\n\n"
        f"SLAM POSE\n"
        f"  x : {sx:+.3f} m\n"
        f"  y : {sy:+.3f} m\n"
        f"  θ : {np.degrees(sth):+.1f}°\n\n"
        f"  step : {step}"
    )

    return gt_dot, slam_dot, info, status

# ── Key handlers ──────────────────────────────────────────────────────────────
plt.rcParams['keymap.save']    = []
plt.rcParams['keymap.quit']    = []
plt.rcParams['keymap.home']    = []
plt.rcParams['keymap.back']    = []
plt.rcParams['keymap.forward'] = []
plt.rcParams['keymap.pan']     = []
plt.rcParams['keymap.zoom']    = []

def on_press(event):
    global gt_pose, slam_pose, prev_pose, fine_map, coarse_map, step
    if event.key == 'escape':
        plt.close(); sys.exit(0)
    if event.key == 'r':
        gt_pose    = Pose(0, 6, 0)
        slam_pose  = Pose(0, 6, 0)
        prev_pose  = Pose(0, 6, 0)
        fine_map   = OccupancyGrid(FINE_RESOLUTION, origin=MAP_ORIGIN)
        coarse_map = fine_map.downsample()
        step       = 0
    keys.add(event.key.lower() if event.key else '')

def on_release(event):
    keys.discard(event.key.lower() if event.key else '')

fig.canvas.mpl_connect('key_press_event', on_press)
fig.canvas.mpl_connect('key_release_event', on_release)

plt.tight_layout()
print("Click the window first, then use keys to move.")
print("Green = GT pose,  Red = SLAM pose (noisy)")

ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
try:
    plt.show()
except KeyboardInterrupt:
    sys.exit(0)