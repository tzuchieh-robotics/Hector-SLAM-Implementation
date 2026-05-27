import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from core.pose import Pose
from core.scan import LaserScan
from core.simulator import LaserSimulator
from config.params import LASER_RANGE_MAX, LASER_RANGE_MIN, NUM_RAYS, LASER_FOV


@pytest.fixture
def sim():
    return LaserSimulator()


# ─── return type ──────────────────────────────────────────────────────────────

def test_get_scan_returns_laser_scan(sim):
    result = sim.get_scan(Pose(0, 2, 0))
    assert isinstance(result, LaserScan)


def test_scan_has_correct_num_rays(sim):
    result = sim.get_scan(Pose(0, 2, 0))
    assert len(result.ranges) == NUM_RAYS
    assert len(result.angles) == NUM_RAYS


# ─── angle convention ─────────────────────────────────────────────────────────

def test_angles_span_full_fov(sim):
    """Angles should cover exactly LASER_FOV radians (minus endpoint overlap)."""
    result = sim.get_scan(Pose(0, 2, 0))
    span = result.angles[-1] - result.angles[0]
    # linspace with num_rays points spans FOV * (1 - 1/num_rays)
    expected = LASER_FOV * (1 - 1.0 / NUM_RAYS)
    assert span == pytest.approx(expected, rel=0.01)


def test_angles_centered_on_theta(sim):
    """Angle array center should equal pose.theta."""
    for theta in [0, np.pi / 4, np.pi / 2, np.pi, -np.pi / 3]:
        result = sim.get_scan(Pose(0, 2, theta))
        center = (result.angles[0] + result.angles[-1]) / 2
        assert center == pytest.approx(theta, abs=0.01), (
            f"theta={theta:.3f}: center={center:.3f}"
        )


def test_angles_are_absolute(sim):
    """Angles should be world-frame (absolute), not relative to robot."""
    # Robot at theta=pi/2: first angle ≈ pi/2 - pi = -pi/2
    result = sim.get_scan(Pose(0, 2, np.pi / 2))
    assert result.angles[0] == pytest.approx(-np.pi / 2, abs=0.05)


# ─── range bounds ─────────────────────────────────────────────────────────────

def test_ranges_within_bounds(sim):
    for theta in [0, np.pi / 4, np.pi]:
        result = sim.get_scan(Pose(0, 2, theta))
        assert np.all(result.ranges >= LASER_RANGE_MIN - 1e-4)
        assert np.all(result.ranges <= LASER_RANGE_MAX + 1e-4)


# ─── wall detection ───────────────────────────────────────────────────────────

def test_left_wall_detected(sim):
    """Robot at (0,5,0): left wall at x=-5, distance should be ~5m."""
    result = sim.get_scan(Pose(0, 5, 0))
    # Angle pi (left) → should hit left wall at distance ~5
    idx = np.argmin(np.abs(result.angles - np.pi))
    assert result.ranges[idx] == pytest.approx(5.0, abs=0.5)


def test_right_wall_detected(sim):
    """Robot at (0,8,0): forward ray clears obstacle_3 (ends at y=7), hits right wall at x=5."""
    result = sim.get_scan(Pose(0, 8, 0))
    idx = np.argmin(np.abs(result.angles - 0.0))
    assert result.ranges[idx] == pytest.approx(5.0, abs=0.5)


def test_back_wall_detected(sim):
    """Robot at (-3,5,0): back ray (-y) misses obstacle_1 (x=-1 to 1), hits back wall y=0."""
    result = sim.get_scan(Pose(-3, 5, 0))
    idx = np.argmin(np.abs(result.angles - (-np.pi / 2)))
    assert result.ranges[idx] == pytest.approx(5.0, abs=0.5)


def test_closer_position_gives_smaller_range(sim):
    """Robot closer to a wall should report smaller range on that ray."""
    r_far  = sim.get_scan(Pose(0, 5, 0))
    r_near = sim.get_scan(Pose(4, 5, 0))  # closer to right wall (x=5)
    idx = np.argmin(np.abs(r_far.angles - 0.0))
    assert r_near.ranges[idx] < r_far.ranges[idx]


def test_no_zero_range(sim):
    """Robot should never report range 0 (not inside a wall)."""
    result = sim.get_scan(Pose(0, 5, 0))
    assert np.all(result.ranges > 0)


# ─── noise ───────────────────────────────────────────────────────────────────

def test_scan_is_noisy(sim):
    """Two scans from same pose should differ slightly (noise)."""
    pose = Pose(0, 5, 0)
    s1 = sim.get_scan(pose)
    s2 = sim.get_scan(pose)
    assert not np.allclose(s1.ranges, s2.ranges), "Scans are identical — noise not added"


def test_noise_is_small(sim):
    """Mean absolute difference between two scans should be small."""
    pose = Pose(0, 5, 0)
    diffs = []
    for _ in range(20):
        s1 = sim.get_scan(pose)
        s2 = sim.get_scan(pose)
        diffs.append(np.mean(np.abs(s1.ranges - s2.ranges)))
    assert np.mean(diffs) < 0.05  # noise should be < 5 cm on average


# ─── edge positions ───────────────────────────────────────────────────────────

def test_scan_near_wall_corner(sim):
    """Robot close to a corner should still produce a valid scan."""
    result = sim.get_scan(Pose(-4.5, 0.5, 0))
    assert np.all(result.ranges >= LASER_RANGE_MIN - 1e-4)
    assert np.all(result.ranges <= LASER_RANGE_MAX + 1e-4)


def test_all_walls_visible_from_center(sim):
    """From room center, at least some rays should hit walls."""
    result = sim.get_scan(Pose(0, 5, 0))
    assert np.any(result.ranges < LASER_RANGE_MAX - 0.5)
