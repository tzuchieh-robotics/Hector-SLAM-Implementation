import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from core.map import OccupancyGrid
from student.down_sampling import down_sample


def make_map(res=0.05, origin=(0.0, 0.0), size=10):
    return OccupancyGrid(res, origin=origin, initial_size=size)


# ── return type ────────────────────────────────────────────────────────────────
def test_returns_occupancy_grid():
    gm = make_map()
    result = down_sample(gm)
    assert isinstance(result, OccupancyGrid)


# ── resolution ────────────────────────────────────────────────────────────────
def test_resolution_doubled():
    gm = make_map(res=0.05)
    result = down_sample(gm)
    assert result.resolution == pytest.approx(0.10, abs=1e-9)


def test_resolution_doubled_arbitrary():
    gm = make_map(res=0.20)
    result = down_sample(gm)
    assert result.resolution == pytest.approx(0.40, abs=1e-9)


# ── dimensions ────────────────────────────────────────────────────────────────
def test_width_halved_even():
    gm = make_map(size=10)
    result = down_sample(gm)
    assert result.width == 5


def test_height_halved_even():
    gm = make_map(size=10)
    result = down_sample(gm)
    assert result.height == 5


def test_width_halved_odd():
    gm = make_map(size=9)
    result = down_sample(gm)
    assert result.width == 5  # ceil(9/2)


def test_data_shape_matches_dimensions():
    gm = make_map(size=8)
    result = down_sample(gm)
    assert result.data.shape == (result.height, result.width)


# ── origin preserved ──────────────────────────────────────────────────────────
def test_origin_preserved():
    gm = OccupancyGrid(0.05, origin=(-3.0, -1.0), initial_size=10)
    result = down_sample(gm)
    assert result.origin_x == pytest.approx(-3.0, abs=1e-9)
    assert result.origin_y == pytest.approx(-1.0, abs=1e-9)


# ── value averaging ───────────────────────────────────────────────────────────
def test_uniform_map_preserved():
    """All cells at constant value → coarse cells equal the same value."""
    gm = make_map(size=4)
    gm.data[:] = 2.0
    result = down_sample(gm)
    np.testing.assert_allclose(result.data, 2.0, atol=1e-5)


def test_two_by_two_block_average():
    """Single 2×2 block → coarse value = mean of the 4 fine cells."""
    gm = make_map(size=2)
    gm.data[0, 0] = 1.0
    gm.data[0, 1] = 3.0
    gm.data[1, 0] = 0.0
    gm.data[1, 1] = 4.0
    result = down_sample(gm)
    assert result.data[0, 0] == pytest.approx(2.0, abs=1e-5)


def test_checkerboard_pattern():
    """Alternating 0/4 → coarse value = 2 everywhere."""
    gm = make_map(size=4)
    for r in range(4):
        for c in range(4):
            gm.data[r, c] = 4.0 if (r + c) % 2 == 0 else 0.0
    result = down_sample(gm)
    np.testing.assert_allclose(result.data, 2.0, atol=1e-5)


def test_zero_map_stays_zero():
    gm = make_map(size=6)
    gm.data[:] = 0.0
    result = down_sample(gm)
    np.testing.assert_allclose(result.data, 0.0, atol=1e-9)


# ── larger map ────────────────────────────────────────────────────────────────
def test_larger_map_dimensions():
    gm = make_map(size=200)
    result = down_sample(gm)
    assert result.width == 100
    assert result.height == 100
    assert result.resolution == pytest.approx(0.10, abs=1e-9)


def test_larger_map_value_mean():
    gm = make_map(size=200)
    gm.data[0:2, 0:2] = np.array([[1.0, 3.0], [5.0, 7.0]])
    result = down_sample(gm)
    assert result.data[0, 0] == pytest.approx(4.0, abs=1e-5)
