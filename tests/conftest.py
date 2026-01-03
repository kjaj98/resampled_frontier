import numpy as np
import pytest

from refmvo.utils.constants import PERIODS_PER_YEAR


def periods_per_year(freq: str) -> int:
    key = (freq or "").strip().lower()
    return PERIODS_PER_YEAR.get(key, 1)


def assert_simplex_and_bounds(w: np.ndarray, lb: np.ndarray, ub: np.ndarray, tol: float = 1e-6) -> None:
    assert np.isfinite(w).all()
    assert np.isclose(np.sum(w), 1.0, atol=tol)
    assert np.all(w >= lb - tol)
    assert np.all(w <= ub + tol)


@pytest.fixture
def toy_case_2_assets():
    mu = np.array([0.02, 0.08])
    cov = np.array([[0.01, 0.002], [0.002, 0.04]])
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    return mu, cov, bounds


@pytest.fixture
def case_4_assets_moderate_corr():
    mu = np.array([0.03, 0.05, 0.07, 0.06])
    vols = np.array([0.05, 0.10, 0.15, 0.12])
    corr = np.array(
        [
            [1.0, 0.2, 0.3, 0.25],
            [0.2, 1.0, 0.4, 0.35],
            [0.3, 0.4, 1.0, 0.5],
            [0.25, 0.35, 0.5, 1.0],
        ]
    )
    cov = np.diag(vols) @ corr @ np.diag(vols)
    bounds = np.array([[0.0, 0.7], [0.0, 0.7], [0.0, 0.7], [0.0, 0.7]])
    return mu, cov, bounds


def has_solver():
    try:
        import cvxpy as cp
    except Exception:
        return False
    installed = set(cp.installed_solvers())
    return bool(installed.intersection({"OSQP", "ECOS", "SCS"}))


@pytest.fixture
def require_solver():
    if not has_solver():
        pytest.skip("No supported CVXPY solver installed.")
