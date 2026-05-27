import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from core.pose import Pose
from core.map import OccupancyGrid
from core.scan import LaserScan
from config.params import HIT_LOG_ODDS, MISS_LOG_ODDS, LOG_ODDS_MIN, LOG_ODDS_MAX
from student.map_update import map_update


def make_map(res=0.1, origin=(-5.0, -5.0), size=100):
    return OccupancyGrid(res, origin=origin, initial_size=size)


def single_ray_scan(range_val, angle, num_rays=1):
    """One-ray scan at the given range and absolute angle."""
    return LaserScan(np.array([range_val]), angles=np.array([angle]))


# ── return type ────────────────────────────────────────────────────────────────
def test_returns_occupancy_grid():
    gm = make_map()
    scan = single_ray_scan(1.0, 0.0)
    result = map_update(gm, Pose(0, 0, 0), scan)
    assert isinstance(result, OccupancyGrid)


# ── hit cell increases ─────────────────────────────────────────────────────────
def test_hit_cell_increases():
    gm = make_map(res=0.1, origin=(-5, -5))
    pose = Pose(0, 0, 0)
    # Ray along +x at range 1.0 → hit at (1.0, 0.0)
    scan = single_ray_scan(1.0, 0.0)
    map_update(gm, pose, scan)
    val = gm.get_value(1.0, 0.0)
    assert val > 0, f"Expected positive log-odds at hit cell, got {val}"


def test_hit_cell_gets_hit_log_odds():
    gm = make_map()
    pose = Pose(0, 0, 0)
    scan = single_ray_scan(1.0, 0.0)
    map_update(gm, pose, scan)
    assert gm.get_value(1.0, 0.0) == pytest.approx(HIT_LOG_ODDS, abs=1e-4)


# ── free cells decrease ────────────────────────────────────────────────────────
def test_free_cells_decrease():
    gm = make_map()
    pose = Pose(0, 0, 0)
    # Ray at range 2.0 → cells at (0.5, 0), (1.0, 0), (1.5, 0) should be free
    scan = single_ray_scan(2.0, 0.0)
    map_update(gm, pose, scan)
    val = gm.get_value(0.5, 0.0)
    assert val < 0, f"Expected negative log-odds at free cell, got {val}"


def test_free_cell_gets_miss_log_odds():
    gm = make_map()
    pose = Pose(0, 0, 0)
    scan = single_ray_scan(2.0, 0.0)
    map_update(gm, pose, scan)
    # Cell halfway along the ray
    assert gm.get_value(1.0, 0.0) == pytest.approx(MISS_LOG_ODDS, abs=1e-4)


# ── clamping ──────────────────────────────────────────────────────────────────
def test_log_odds_clamped_at_max():
    gm = make_map()
    pose = Pose(0, 0, 0)
    scan = single_ray_scan(1.0, 0.0)
    # Repeat updates many times
    for _ in range(100):
        map_update(gm, pose, scan)
    assert gm.get_value(1.0, 0.0) <= LOG_ODDS_MAX + 1e-6


def test_log_odds_clamped_at_min():
    gm = make_map()
    pose = Pose(0, 0, 0)
    scan = single_ray_scan(2.0, 0.0)
    for _ in range(100):
        map_update(gm, pose, scan)
    assert gm.get_value(1.0, 0.0) >= LOG_ODDS_MIN - 1e-6


# ── geometry: angle ───────────────────────────────────────────────────────────
def test_hit_at_45_degrees():
    gm = make_map(res=0.05)
    pose = Pose(0, 0, 0)
    r = np.sqrt(2)  # 45° ray, range sqrt(2) → hit near (1, 1)
    scan = single_ray_scan(r, np.pi / 4)
    map_update(gm, pose, scan)
    # Hit cell should have positive log-odds
    val = gm.get_value(1.0, 1.0)
    assert val > 0


def test_hit_behind_robot():
    gm = make_map()
    pose = Pose(0, 0, 0)
    # Ray at angle pi (behind), range 1.0 → hit at (-1, 0)
    scan = single_ray_scan(1.0, np.pi)
    map_update(gm, pose, scan)
    assert gm.get_value(-1.0, 0.0) > 0


# ── max-range rays are not marked as hits ────────────────────────────────────
def test_max_range_ray_no_hit():
    from config.params import LASER_RANGE_MAX
    gm = make_map()
    pose = Pose(0, 0, 0)
    scan = single_ray_scan(LASER_RANGE_MAX, 0.0)
    map_update(gm, pose, scan)
    # The endpoint should NOT be marked as hit (max-range → no obstacle)
    end_x = LASER_RANGE_MAX
    val = gm.get_value(end_x, 0.0)
    assert val <= 0


# ── multi-ray scan ────────────────────────────────────────────────────────────
def test_multi_ray_scan():
    gm = make_map()
    pose = Pose(0, 0, 0)
    angles = np.linspace(-np.pi, np.pi, 36)
    ranges = np.full(36, 2.0)
    scan = LaserScan(ranges, angles=angles)
    result = map_update(gm, pose, scan)
    # At least some cells should be occupied
    assert np.any(result.data > 0)


# ── non-zero pose ─────────────────────────────────────────────────────────────
def test_map_update_offset_pose():
    gm = make_map()
    pose = Pose(1.0, 1.0, 0)
    scan = single_ray_scan(1.0, 0.0)
    map_update(gm, pose, scan)
    # Hit should be at (2.0, 1.0) = pose.x + r*cos(0), pose.y + r*sin(0)
    assert gm.get_value(2.0, 1.0) > 0
