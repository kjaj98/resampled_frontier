from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ConstraintsSpec:
    """Constraint settings for portfolio optimization."""

    asset_bounds: np.ndarray
    long_only: bool = True
    global_min: Optional[float] = None
    global_max: Optional[float] = None
    tol: float = 1e-8


def normalize_bounds(
    asset_bounds: np.ndarray,
    long_only: bool = True,
    global_min: Optional[float] = None,
    global_max: Optional[float] = None,
    tol: float = 1e-8,
) -> np.ndarray:
    """Apply global caps and validate bounds for feasibility."""
    bounds = np.asarray(asset_bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("asset_bounds must be an (N,2) array")
    if not np.all(np.isfinite(bounds)):
        raise ValueError("asset_bounds must be finite")

    lb = bounds[:, 0].copy()
    ub = bounds[:, 1].copy()

    if global_min is not None:
        lb = np.maximum(lb, float(global_min))
    if global_max is not None:
        ub = np.minimum(ub, float(global_max))
    if long_only:
        lb = np.maximum(lb, 0.0)

    if np.any(lb > ub + tol):
        raise ValueError("lower bound exceeds upper bound")
    if lb.sum() > 1.0 + tol:
        raise ValueError("sum of lower bounds exceeds 1")
    if ub.sum() < 1.0 - tol:
        raise ValueError("sum of upper bounds below 1")

    return np.vstack([lb, ub]).T


def constraints_summary(bounds: np.ndarray) -> dict:
    """Summarize bounds for diagnostics."""
    bounds = np.asarray(bounds, dtype=float)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    return {
        "lb_min": float(lb.min()) if len(lb) else 0.0,
        "lb_max": float(lb.max()) if len(lb) else 0.0,
        "ub_min": float(ub.min()) if len(ub) else 0.0,
        "ub_max": float(ub.max()) if len(ub) else 0.0,
        "lb_sum": float(lb.sum()),
        "ub_sum": float(ub.sum()),
    }


__all__ = ["ConstraintsSpec", "normalize_bounds", "constraints_summary"]
