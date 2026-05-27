import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from core.pose import Pose
from core.map import OccupancyGrid
from core.scan import LaserScan
from student.gn_optimize import gn_optimize
from student.map_update import map_update


# ─── helpers ──────────────────────────────────────────────────────────────────

def scan_from_pose(pose, ranges, angles):
    """World-frame scan points from a given pose + absolute angles."""
    pts = []
    for r, a in zip(ranges, angles):
        pts.append([pose.x + r * np.cos(a), pose.y + r * np.sin(a)])
    return np.array(pts)


def box_scan(pose, room_size=5.0, n=36):
    """Synthetic scan inside a square room, returning (ranges, angles)."""
    angles = np.linspace(pose.theta - np.pi, pose.theta + np.pi, n, endpoint=False)
    ranges = []
    for a in angles:
        cx, cy = np.cos(a), np.sin(a)
        dists = []
        if abs(cx) > 1e-9:
            for wall_x in [0.0, room_size]:
                t = (wall_x - pose.x) / cx
                if t > 0:
                    hy = pose.y + t * cy
                    if 0 <= hy <= room_size:
                        dists.append(t)
        if abs(cy) > 1e-9:
            for wall_y in [0.0, room_size]:
                t = (wall_y - pose.y) / cy
                if t > 0:
                    hx = pose.x + t * cx
                    if 0 <= hx <= room_size:
                        dists.append(t)
        ranges.append(min(dists) if dists else 10.0)
    return np.array(ranges), angles


def build_test_map(true_pose, ranges, angles, res=0.3, n_updates=10):
    """Build map using map_update (creates both hit and free/miss cells)."""
    size = int(14 / res) + 1
    gm = OccupancyGrid(res, origin=(-7.0, -7.0), initial_size=size)
    scan = LaserScan(ranges.copy(), angles=angles.copy())
    for _ in range(n_updates):
        gm = map_update(gm, true_pose, scan)
    return gm


# ─── return type ──────────────────────────────────────────────────────────────

def test_returns_pose():
    true_pose = Pose(2.5, 2.5, 0)
    ranges, angles = box_scan(true_pose)
    gm = build_test_map(true_pose, ranges, angles)
    pts = scan_from_pose(true_pose, ranges, angles)
    result = gn_optimize(true_pose, gm, pts)
    assert isinstance(result, Pose)


def test_theta_normalized():
    """Output theta should be in [-pi, pi]."""
    true_pose = Pose(2.5, 2.5, 0)
    ranges, angles = box_scan(true_pose)
    gm = build_test_map(true_pose, ranges, angles)
    pts = scan_from_pose(true_pose, ranges, angles)
    result = gn_optimize(Pose(2.5, 2.5, 3.5), gm, pts)
    assert -np.pi <= result.theta <= np.pi


# ─── empty matched_points ─────────────────────────────────────────────────────

def test_empty_matched_points_returns_initial():
    true_pose = Pose(2.5, 2.5, 0)
    ranges, angles = box_scan(true_pose)
    gm = build_test_map(true_pose, ranges, angles)
    result = gn_optimize(true_pose, gm, np.empty((0, 2)))
    assert isinstance(result, Pose)


# ─── already-aligned pose does not move much ──────────────────────────────────

def test_aligned_pose_stable():
    """Scan from true_pose with world-pts from same pose → small correction."""
    true_pose = Pose(2.5, 2.5, 0)
    ranges, angles = box_scan(true_pose)
    gm = build_test_map(true_pose, ranges, angles)
    # World pts from same pose → scan lands on hit cells, residual ≈ 0
    pts = scan_from_pose(true_pose, ranges, angles)
    result = gn_optimize(true_pose, gm, pts)
    assert abs(result.x - true_pose.x) < 0.3
    assert abs(result.y - true_pose.y) < 0.3


# ─── correction direction ─────────────────────────────────────────────────────
# SLAM scenario: ranges come from GT sensor, world-pts use noisy SLAM position.
# Noisy pose UNDER-estimates position → scan lands in miss zone (inside room),
# giving a nonzero gradient toward occupied wall cells.

def test_x_correction_direction():
    """noisy_pose.x < true_pose.x → scan inside room → GN increases x."""
    true_pose  = Pose(2.5, 2.5, 0)
    noisy_pose = Pose(2.2, 2.5, 0)  # -0.3 m x error (scan ends up inside room)

    ranges, angles = box_scan(true_pose)
    gm  = build_test_map(true_pose, ranges, angles)
    pts = scan_from_pose(noisy_pose, ranges, angles)

    result = gn_optimize(noisy_pose, gm, pts)
    assert result.x > noisy_pose.x, (
        f"Expected x > {noisy_pose.x:.3f} (correction toward wall), got {result.x:.4f}"
    )


def test_y_correction_direction():
    """noisy_pose.y < true_pose.y → GN increases y."""
    true_pose  = Pose(2.5, 2.5, 0)
    noisy_pose = Pose(2.5, 2.2, 0)

    ranges, angles = box_scan(true_pose)
    gm  = build_test_map(true_pose, ranges, angles)
    pts = scan_from_pose(noisy_pose, ranges, angles)

    result = gn_optimize(noisy_pose, gm, pts)
    assert result.y > noisy_pose.y


# ─── convergence ──────────────────────────────────────────────────────────────

def test_converges_closer_than_initial():
    """After GN, pose should be closer to the true pose than the noisy start."""
    true_pose  = Pose(2.5, 2.5, 0)
    noisy_pose = Pose(2.1, 2.2, 0)

    ranges, angles = box_scan(true_pose)
    gm  = build_test_map(true_pose, ranges, angles)
    pts = scan_from_pose(noisy_pose, ranges, angles)

    result = gn_optimize(noisy_pose, gm, pts)

    dist_before = np.hypot(noisy_pose.x - true_pose.x, noisy_pose.y - true_pose.y)
    dist_after  = np.hypot(result.x     - true_pose.x, result.y     - true_pose.y)
    assert dist_after < dist_before, (
        f"GN moved AWAY from true pose: before={dist_before:.4f}, after={dist_after:.4f}"
    )


def test_max_iterations_respected():
    """Function must not hang; finish in finite time."""
    true_pose = Pose(2.5, 2.5, 0)
    ranges, angles = box_scan(true_pose)
    gm = build_test_map(true_pose, ranges, angles)
    pts = scan_from_pose(true_pose, ranges, angles)
    gn_optimize(true_pose, gm, pts, max_iterations=1)
    gn_optimize(true_pose, gm, pts, max_iterations=20)


# ─── step-size safety ────────────────────────────────────────────────────────

def test_no_wild_jump_on_sparse_map():
    """On a nearly empty map, GN should not fling the pose far away."""
    gm = OccupancyGrid(0.05, origin=(-10.0, -10.0), initial_size=400)
    pts = np.random.uniform(-3, 3, (30, 2))
    result = gn_optimize(Pose(0, 0, 0), gm, pts)
    assert abs(result.x) < 5.0
    assert abs(result.y) < 5.0
