from typing import Dict, Optional, Sequence

import numpy as np

from refmvo.core.solvers import SOLVER_ORDER, solve_extreme_return, solve_target_qp


def compute_targets(
    mu: np.ndarray,
    bounds: np.ndarray,
    points: int,
    solvers: Sequence[str] = SOLVER_ORDER,
    tol: float = 1e-8,
) -> Dict[str, object]:
    """Compute a fixed global target grid based on base moments and constraints."""
    mu = np.asarray(mu, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    lo = solve_extreme_return(mu, bounds, maximize=False, solvers=solvers)
    hi = solve_extreme_return(mu, bounds, maximize=True, solvers=solvers)
    if lo["status"] != "OK" or hi["status"] != "OK":
        raise RuntimeError("Failed to compute target grid endpoints from base inputs.")

    ret_min = float(lo["ret"])
    ret_max = float(hi["ret"])
    if ret_max - ret_min < tol:
        targets = np.array([ret_min], dtype=float)
        degenerate = True
    else:
        targets = np.linspace(ret_min, ret_max, max(int(points), 2))
        degenerate = False

    return {
        "targets": targets,
        "ret_min": ret_min,
        "ret_max": ret_max,
        "degenerate": degenerate,
    }


def compute_target_bands(targets: np.ndarray, min_band: float = 1e-6) -> np.ndarray:
    """Compute symmetric return bands using half the grid spacing."""
    targets = np.asarray(targets, dtype=float)
    if targets.size <= 1:
        return np.array([float(min_band)], dtype=float)
    diffs = np.diff(targets)
    if np.any(diffs <= 0):
        raise ValueError("Targets must be strictly increasing to compute bands.")
    bands = np.empty_like(targets)
    bands[0] = 0.5 * diffs[0]
    bands[-1] = 0.5 * diffs[-1]
    if targets.size > 2:
        bands[1:-1] = 0.5 * np.minimum(diffs[:-1], diffs[1:])
    bands = np.maximum(bands, float(min_band))
    return bands


def efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    r_f: float,
    bounds: np.ndarray,
    targets: np.ndarray,
    eps: float = 1e-6,
    target_bands: Optional[np.ndarray] = None,
    solvers: Sequence[str] = SOLVER_ORDER,
) -> Dict[str, object]:
    """Solve a constrained frontier for a fixed target grid."""
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    targets = np.asarray(targets, dtype=float)
    target_bands = None if target_bands is None else np.asarray(target_bands, dtype=float)
    n = len(mu)
    k = len(targets)
    if target_bands is not None and target_bands.size != k:
        raise ValueError("target_bands length must match targets length")
    W = np.full((k, n), np.nan)
    rets = np.full(k, np.nan)
    risks = np.full(k, np.nan)
    ok_mask = np.zeros(k, dtype=bool)
    statuses = []
    solver_names = []
    use_bands = target_bands is not None

    for i, R in enumerate(targets):
        if use_bands:
            eps_i = float(target_bands[i])
            res = solve_target_qp(mu, cov, bounds, float(R), eps=eps_i, eps_rel=0.0, solvers=solvers)
        else:
            res = solve_target_qp(mu, cov, bounds, float(R), eps=eps, solvers=solvers)
        statuses.append(res.status)
        solver_names.append(res.solver_name)
        if res.status == "OK" and res.w is not None:
            W[i, :] = res.w
            rets[i] = res.achieved_return
            risks[i] = np.sqrt(res.achieved_var)
            ok_mask[i] = True

    sharpe = np.full(k, -np.inf)
    if np.any(ok_mask):
        sharpe[ok_mask] = (rets[ok_mask] - r_f) / np.maximum(risks[ok_mask], 1e-12)
        tangency_idx = int(np.nanargmax(sharpe))
    else:
        tangency_idx = None

    return {
        "targets": targets,
        "W": W,
        "rets": rets,
        "risks": risks,
        "ok_mask": ok_mask,
        "statuses": statuses,
        "solver_names": solver_names,
        "tangency_index": tangency_idx,
    }


__all__ = [
    "compute_targets",
    "compute_target_bands",
    "efficient_frontier",
    "solve_target_qp",
    "SOLVER_ORDER",
]
