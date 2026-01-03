import numpy as np
import pytest

from refmvo.core.constraints import normalize_bounds


def test_long_only_enforced():
    bounds = np.array([[-0.2, 0.5], [0.0, 0.6]])
    out = normalize_bounds(bounds, long_only=True)
    assert np.all(out[:, 0] >= 0.0)


def test_global_cap_clamps():
    bounds = np.array([[0.0, 0.9], [0.0, 0.9]])
    out = normalize_bounds(bounds, long_only=True, global_max=0.6)
    assert np.allclose(out[:, 1], 0.6)


def test_lb_gt_ub_raises():
    bounds = np.array([[0.6, 0.4], [0.0, 0.5]])
    with pytest.raises(ValueError, match="lower bound exceeds upper bound"):
        normalize_bounds(bounds, long_only=True)


def test_sum_lb_gt_one_raises():
    bounds = np.array([[0.6, 1.0], [0.5, 1.0]])
    with pytest.raises(ValueError, match="sum of lower bounds exceeds 1"):
        normalize_bounds(bounds, long_only=True)


def test_sum_ub_lt_one_raises():
    bounds = np.array([[0.0, 0.4], [0.0, 0.4]])
    with pytest.raises(ValueError, match="sum of upper bounds below 1"):
        normalize_bounds(bounds, long_only=True)
