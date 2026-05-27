import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from core.pose import Pose
from core.map import OccupancyGrid
from core.scan import LaserScan
from config.params import LASER_RANGE_MAX, NUM_RAYS
from student.lidar_measurement import lidar_measurement_update


def make_map(res=0.1, origin=(-5.0, -5.0), size=100):
    gm = OccupancyGrid(res, origin=origin, initial_size=size)
    return gm


def identical_scans(n=NUM_RAYS, range_val=2.0, theta=0.0):
    """Real and predicted scans with the same ranges → all rays match."""
    angles = np.linspace(theta - np.pi, theta + np.pi, n)
    ranges = np.full(n, range_val)
    real = LaserScan(ranges.copy(), angles=angles)
    pred = LaserScan(ranges.copy(), angles=angles)
    return real, pred


def mismatched_scans(n=NUM_RAYS, theta=0.0):
    """Predicted ranges are very different from real → no rays match."""
    angles = np.linspace(theta - np.pi, theta + np.pi, n)
    real = LaserScan(np.full(n, 2.0), angles=angles)
    pred = LaserScan(np.full(n, 8.0), angles=angles)  # > 0.5 difference
    return real, pred


# ── return type ────────────────────────────────────────────────────────────────
def test_returns_tuple():
    gm = make_map()
    real, pred = identical_scans()
    result = lidar_measurement_update(real, pred, Pose(0, 0, 0), gm)
    assert isinstance(result, tuple) and len(result) == 2


def test_matched_points_is_ndarray():
    gm = make_map()
    real, pred = identical_scans()
    _, pts = lidar_measurement_update(real, pred, Pose(0, 0, 0), gm)
    assert isinstance(pts, np.ndarray)


def test_matched_points_shape():
    gm = make_map()
    real, pred = identical_scans(n=36)
    _, pts = lidar_measurement_update(real, pred, Pose(0, 0, 0), gm)
    assert pts.ndim == 2 and pts.shape[1] == 2


# ── matching logic ────────────────────────────────────────────────────────────
def test_identical_scans_all_matched():
    """When real==predicted, all valid rays should be returned."""
    gm = make_map()
    real, pred = identical_scans(n=36)
    _, pts = lidar_measurement_update(real, pred, Pose(0, 0, 0), gm)
    # All 36 rays are within threshold (diff=0)
    assert len(pts) == 36


def test_mismatched_scans_no_points():
    """When real differs from predicted by > threshold, no points returned."""
    gm = make_map()
    real, pred = mismatched_scans(n=36)
    _, pts = lidar_measurement_update(real, pred, Pose(0, 0, 0), gm)
    assert len(pts) == 0


def test_partial_match():
    """Half identical, half mismatched → roughly half returned."""
    n = 36
    angles = np.linspace(-np.pi, np.pi, n)
    real_ranges = np.full(n, 2.0)
    pred_ranges = np.full(n, 2.0)
    pred_ranges[:n // 2] = 8.0  # first half doesn't match
    real = LaserScan(real_ranges, angles=angles)
    pred = LaserScan(pred_ranges, angles=angles)
    gm = make_map()
    _, pts = lidar_measurement_update(real, pred, Pose(0, 0, 0), gm)
    assert len(pts) == n // 2


# ── world frame correctness ───────────────────────────────────────────────────
def test_matched_point_world_position():
    """Single ray at angle=0, range=2 from pose (1,1,0) → point at (3,1)."""
    angles = np.array([0.0])  # forward ray
    real = LaserScan(np.array([2.0]), angles=angles)
    pred = LaserScan(np.array([2.0]), angles=angles)
    gm = make_map()
    pose = Pose(1.0, 1.0, 0.0)
    _, pts = lidar_measurement_update(real, pred, pose, gm)
    assert len(pts) == 1
    assert pts[0, 0] == pytest.approx(3.0, abs=1e-6)
    assert pts[0, 1] == pytest.approx(1.0, abs=1e-6)


def test_matched_point_angle_90():
    """Single ray at angle=pi/2, range=2 from (0,0,0) → point at (0,2)."""
    angles = np.array([np.pi / 2])
    real = LaserScan(np.array([2.0]), angles=angles)
    pred = LaserScan(np.array([2.0]), angles=angles)
    gm = make_map()
    _, pts = lidar_measurement_update(real, pred, Pose(0, 0, 0), gm)
    assert pts[0, 0] == pytest.approx(0.0, abs=1e-5)
    assert pts[0, 1] == pytest.approx(2.0, abs=1e-5)


# ── likelihood ────────────────────────────────────────────────────────────────
def test_likelihood_is_float():
    gm = make_map()
    real, pred = identical_scans(n=10)
    likelihood, _ = lidar_measurement_update(real, pred, Pose(0, 0, 0), gm)
    assert isinstance(float(likelihood), float)


def test_likelihood_higher_on_occupied_map():
    """Occupied map should give higher (less negative) likelihood than empty."""
    gm_empty = make_map()
    gm_occ = make_map()
    real, pred = identical_scans(n=10, range_val=1.0)
    pose = Pose(0, 0, 0)
    # Mark the hit cells as occupied
    angles = np.linspace(-np.pi, np.pi, 10)
    for a in angles:
        gm_occ.set_value(1.0 * np.cos(a), 1.0 * np.sin(a), 3.0)

    l_empty, _ = lidar_measurement_update(real, pred, pose, gm_empty)
    l_occ,   _ = lidar_measurement_update(real, pred, pose, gm_occ)
    assert l_occ > l_empty


# ── filtering is based on real vs predicted difference ────────────────────────
def test_large_discrepancy_filtered():
    """Rays where |real - pred| > threshold should be excluded."""
    angles = np.array([0.0, np.pi / 2])
    # First ray: discrepancy = 6 (well above 0.5 threshold)
    # Second ray: discrepancy = 0 → should be included
    real = LaserScan(np.array([2.0, 2.0]), angles=angles)
    pred = LaserScan(np.array([8.0, 2.0]), angles=angles)
    gm = make_map()
    _, pts = lidar_measurement_update(real, pred, Pose(0, 0, 0), gm)
    assert len(pts) == 1
