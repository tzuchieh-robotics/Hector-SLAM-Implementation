import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from core.pose import Pose, normalize_angle
from student.motion_update import motion_update


def make_imu(vx=0.0, vy=0.0, omega=0.0, dt=0.1):
    return {'linear_velocity': [vx, vy], 'angular_velocity': omega, 'dt': dt}


# ── return type ────────────────────────────────────────────────────────────────
def test_returns_pose():
    result = motion_update(Pose(0, 0, 0), make_imu())
    assert isinstance(result, Pose)


def test_does_not_mutate_input():
    p = Pose(1.0, 2.0, 0.5)
    motion_update(p, make_imu(vx=1.0, omega=0.5))
    assert p.x == 1.0 and p.y == 2.0 and p.theta == 0.5


# ── pure translation ───────────────────────────────────────────────────────────
def test_forward_motion_theta0():
    """vx=1 m/s, theta=0 → x increases by vx*dt"""
    p = Pose(0, 0, 0)
    result = motion_update(p, make_imu(vx=1.0, dt=0.5))
    assert result.x == pytest.approx(0.5, abs=1e-6)
    assert result.y == pytest.approx(0.0, abs=1e-6)
    assert result.theta == pytest.approx(0.0, abs=1e-6)


def test_forward_motion_theta_pi2():
    """vx=1 m/s, theta=pi/2 → y increases"""
    p = Pose(0, 0, np.pi / 2)
    result = motion_update(p, make_imu(vx=1.0, dt=1.0))
    assert result.x == pytest.approx(0.0, abs=1e-6)
    assert result.y == pytest.approx(1.0, abs=1e-6)


def test_lateral_motion():
    """vy=1 m/s, theta=0 → y increases (left-strafe)"""
    p = Pose(0, 0, 0)
    result = motion_update(p, make_imu(vy=1.0, dt=1.0))
    assert result.x == pytest.approx(0.0, abs=1e-6)
    assert result.y == pytest.approx(1.0, abs=1e-6)


def test_lateral_motion_theta_pi2():
    """vy=1 m/s, theta=pi/2 → -x direction"""
    p = Pose(0, 0, np.pi / 2)
    result = motion_update(p, make_imu(vy=1.0, dt=1.0))
    assert result.x == pytest.approx(-1.0, abs=1e-6)
    assert result.y == pytest.approx(0.0, abs=1e-6)


# ── pure rotation ──────────────────────────────────────────────────────────────
def test_pure_rotation():
    p = Pose(0, 0, 0)
    result = motion_update(p, make_imu(omega=1.0, dt=1.0))
    assert result.x == pytest.approx(0.0, abs=1e-6)
    assert result.y == pytest.approx(0.0, abs=1e-6)
    assert result.theta == pytest.approx(1.0, abs=1e-6)


def test_angle_normalization_overflow():
    p = Pose(0, 0, 3.0)
    result = motion_update(p, make_imu(omega=1.0, dt=1.0))
    assert -np.pi <= result.theta <= np.pi


def test_angle_normalization_underflow():
    p = Pose(0, 0, -3.0)
    result = motion_update(p, make_imu(omega=-1.0, dt=1.0))
    assert -np.pi <= result.theta <= np.pi


# ── combined motion ────────────────────────────────────────────────────────────
def test_combined_motion():
    """Forward + rotation: position uses prev_pose.theta, not updated theta."""
    p = Pose(0, 0, 0)
    imu = make_imu(vx=1.0, omega=np.pi, dt=1.0)
    result = motion_update(p, imu)
    # theta should be pi (normalized), position from theta=0
    assert result.x == pytest.approx(1.0, abs=1e-6)
    assert result.theta == pytest.approx(np.pi, abs=1e-6)


def test_zero_dt():
    p = Pose(1.0, 2.0, 0.5)
    result = motion_update(p, make_imu(vx=10.0, omega=10.0, dt=0.0))
    assert result.x == pytest.approx(1.0, abs=1e-6)
    assert result.y == pytest.approx(2.0, abs=1e-6)
    assert result.theta == pytest.approx(0.5, abs=1e-6)


def test_negative_velocity():
    p = Pose(1.0, 0, 0)
    result = motion_update(p, make_imu(vx=-1.0, dt=1.0))
    assert result.x == pytest.approx(0.0, abs=1e-6)
