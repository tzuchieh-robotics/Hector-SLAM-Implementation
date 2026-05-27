import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from core.pose import Pose
from core.map import OccupancyGrid
from core.scan import LaserScan
from config.params import LASER_RANGE_MAX, LASER_RANGE_MIN, NUM_RAYS, LASER_FOV
from student.ray_casting import ray_casting


def empty_map(res=0.1, origin=(-10.0, -10.0), size=200):
    return OccupancyGrid(res, origin=origin, initial_size=size)


def map_with_wall(wx, wy, res=0.1, origin=(-10.0, -10.0), size=200):
    """Place a single occupied cell at world position (wx, wy)."""
    gm = empty_map(res, origin, size)
    gm.set_value(wx, wy, 3.0)
    return gm


# ── return type ────────────────────────────────────────────────────────────────
def test_returns_laser_scan():
    gm = empty_map()
    result = ray_casting(Pose(0, 0, 0), gm)
    assert isinstance(result, LaserScan)


def test_correct_number_of_rays():
    gm = empty_map()
    result = ray_casting(Pose(0, 0, 0), gm)
    assert len(result.ranges) == NUM_RAYS
    assert len(result.angles) == NUM_RAYS


# ── empty map → max range ─────────────────────────────────────────────────────
def test_empty_map_returns_max_range():
    gm = empty_map()
    result = ray_casting(Pose(0, 0, 0), gm)
    np.testing.assert_allclose(result.ranges, LASER_RANGE_MAX, atol=1e-3)


def test_empty_map_various_poses():
    gm = empty_map()
    for pose in [Pose(0, 0, 0), Pose(1, 1, np.pi / 3), Pose(-2, 3, np.pi)]:
        result = ray_casting(pose, gm)
        assert np.all(result.ranges >= LASER_RANGE_MIN - 1e-6)
        assert np.all(result.ranges <= LASER_RANGE_MAX + 1e-6)


# ── obstacle detection ────────────────────────────────────────────────────────
def test_obstacle_ahead_detected():
    """Wall directly ahead (angle 0) → range much smaller than max."""
    gm = map_with_wall(2.0, 0.0, res=0.1, origin=(-5.0, -5.0))
    result = ray_casting(Pose(0, 0, 0), gm)
    # Find the ray closest to angle=0
    idx = np.argmin(np.abs(result.angles))
    assert result.ranges[idx] < LASER_RANGE_MAX


def test_obstacle_range_is_approx_correct():
    """Obstacle at ~2m should give range ~2m."""
    res = 0.1
    gm = map_with_wall(2.0, 0.0, res=res, origin=(-5.0, -5.0))
    result = ray_casting(Pose(0, 0, 0), gm)
    idx = np.argmin(np.abs(result.angles))
    assert result.ranges[idx] == pytest.approx(2.0, abs=res * 2)


def test_no_obstacle_behind_detected_from_front():
    """Obstacle behind robot → forward rays see max range."""
    gm = map_with_wall(-2.0, 0.0, res=0.1, origin=(-5.0, -5.0))
    result = ray_casting(Pose(0, 0, 0), gm)
    idx = np.argmin(np.abs(result.angles))  # forward ray
    assert result.ranges[idx] == pytest.approx(LASER_RANGE_MAX, abs=0.5)


# ── angle convention ──────────────────────────────────────────────────────────
def test_angles_span_full_fov():
    gm = empty_map()
    result = ray_casting(Pose(0, 0, 0), gm)
    span = result.angles[-1] - result.angles[0]
    assert span == pytest.approx(LASER_FOV * (1 - 1.0 / NUM_RAYS), rel=0.01)


def test_angles_centered_on_pose_theta():
    """With theta=pi/2, angles should be centered around pi/2."""
    gm = empty_map()
    result = ray_casting(Pose(0, 0, np.pi / 2), gm)
    center = (result.angles[0] + result.angles[-1]) / 2
    assert center == pytest.approx(np.pi / 2, abs=0.1)


# ── ranges always in bounds ───────────────────────────────────────────────────
def test_ranges_within_bounds_with_obstacle():
    gm = map_with_wall(1.0, 0.0, res=0.1, origin=(-5.0, -5.0))
    result = ray_casting(Pose(0, 0, 0), gm)
    assert np.all(result.ranges >= LASER_RANGE_MIN - 1e-6)
    assert np.all(result.ranges <= LASER_RANGE_MAX + 1e-6)


# ── obstacle distance scaling ─────────────────────────────────────────────────
def test_closer_obstacle_gives_smaller_range():
    """Obstacle at 1m vs 3m: 1m should give smaller range on the same ray."""
    gm_near = map_with_wall(1.0, 0.0, res=0.1, origin=(-5.0, -5.0))
    gm_far  = map_with_wall(3.0, 0.0, res=0.1, origin=(-5.0, -5.0))
    near = ray_casting(Pose(0, 0, 0), gm_near)
    far  = ray_casting(Pose(0, 0, 0), gm_far)
    idx = np.argmin(np.abs(near.angles))
    assert near.ranges[idx] < far.ranges[idx]
