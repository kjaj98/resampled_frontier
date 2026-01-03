from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import cvxpy as cp
import numpy as np

SOLVER_ORDER = ("OSQP", "ECOS", "SCS")
DEFAULT_TOLS = {"sum": 1e-6, "bounds": 1e-6, "ret": 1e-6}
DEFAULT_EPS_REL = 1e-4
DEFAULT_EPS_FLOOR = 1e-3


@dataclass
class SolveResult:
    status: str
    w: Optional[np.ndarray]
    achieved_return: float
    achieved_var: float
    solver_name: str
    message: str


def effective_eps(
    target: float,
    eps_min: float,
    eps_rel: float = DEFAULT_EPS_REL,
    scale_floor: float = DEFAULT_EPS_FLOOR,
) -> float:
    """Return a scale-aware epsilon for target-return bands."""
    if eps_rel <= 0.0:
        return float(eps_min)
    scale = max(abs(float(target)), float(scale_floor))
    return max(float(eps_min), float(eps_rel) * scale)


def _check_simplex_bounds(
    w: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    tol_sum: float = DEFAULT_TOLS["sum"],
    tol_bounds: float = DEFAULT_TOLS["bounds"],
) -> tuple[bool, Dict[str, float]]:
    w = np.asarray(w, dtype=float).ravel()
    if not np.all(np.isfinite(w)):
        return False, {"sum_abs": float("nan")}
    sum_abs = float(abs(np.sum(w) - 1.0))
    min_violation = float(max(0.0, np.max(lb - w)))
    max_violation = float(max(0.0, np.max(w - ub)))
    ok = sum_abs <= tol_sum and min_violation <= tol_bounds and max_violation <= tol_bounds
    return ok, {
        "sum_abs": sum_abs,
        "min_bound_violation": min_violation,
        "max_bound_violation": max_violation,
    }


def _check_target_band(
    w: np.ndarray,
    mu: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    target: float,
    eps: float,
    tol_sum: float = DEFAULT_TOLS["sum"],
    tol_bounds: float = DEFAULT_TOLS["bounds"],
    tol_ret: float = DEFAULT_TOLS["ret"],
) -> tuple[bool, Dict[str, float]]:
    ok, res = _check_simplex_bounds(w, lb, ub, tol_sum=tol_sum, tol_bounds=tol_bounds)
    achieved = float(mu @ w)
    low_violation = float(max(0.0, (target - eps) - achieved))
    high_violation = float(max(0.0, achieved - (target + eps)))
    res.update(
        {
            "achieved_return": achieved,
            "target_low_violation": low_violation,
            "target_high_violation": high_violation,
        }
    )
    ok = ok and low_violation <= tol_ret and high_violation <= tol_ret
    return ok, res


def solve_extreme_return(
    mu: np.ndarray,
    bounds: np.ndarray,
    maximize: bool,
    solvers: Sequence[str] = SOLVER_ORDER,
) -> Dict[str, object]:
    """Solve a linear program to get the min/max feasible return."""
    n = len(mu)
    w = cp.Variable(n)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    cons = [cp.sum(w) == 1.0, w >= lb, w <= ub]
    obj = cp.Maximize(mu @ w) if maximize else cp.Minimize(mu @ w)
    prob = cp.Problem(obj, cons)

    last_err = ""
    last_status = None
    for solver in solvers:
        try:
            prob.solve(solver=solver, verbose=False)
        except cp.SolverError as exc:
            last_err = str(exc)
            continue
        last_status = prob.status
        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and w.value is not None:
            w_val = np.asarray(w.value).ravel()
            ok, _ = _check_simplex_bounds(w_val, lb, ub)
            if not ok:
                last_err = f"{solver}: residual check failed"
                continue
            return {
                "status": "OK",
                "w": w_val,
                "ret": float(mu @ w_val),
                "solver_name": prob.solver_stats.solver_name if prob.solver_stats else str(solver),
                "message": str(prob.status),
            }

    return {
        "status": "FAIL",
        "w": None,
        "ret": float("nan"),
        "solver_name": "",
        "message": last_err or str(last_status),
    }


def solve_target_qp(
    mu: np.ndarray,
    cov: np.ndarray,
    bounds: np.ndarray,
    target: float,
    eps: float = 1e-6,
    eps_rel: float = DEFAULT_EPS_REL,
    scale_floor: float = DEFAULT_EPS_FLOOR,
    solvers: Sequence[str] = SOLVER_ORDER,
) -> SolveResult:
    """Solve min-variance portfolio for a target return band."""
    n = len(mu)
    w = cp.Variable(n)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    eps_eff = effective_eps(target, eps_min=eps, eps_rel=eps_rel, scale_floor=scale_floor)
    cons = [
        cp.sum(w) == 1.0,
        w >= lb,
        w <= ub,
        mu @ w >= target - eps_eff,
        mu @ w <= target + eps_eff,
    ]
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cov)), cons)

    last_err = ""
    last_status = None
    for solver in solvers:
        try:
            prob.solve(solver=solver, verbose=False)
        except cp.SolverError as exc:
            last_err = str(exc)
            continue
        last_status = prob.status
        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and w.value is not None:
            w_val = np.asarray(w.value).ravel()
            ok, _ = _check_target_band(w_val, mu, lb, ub, target, eps_eff)
            if not ok:
                last_err = f"{solver}: residual check failed"
                continue
            achieved_var = float(w_val @ cov @ w_val)
            return SolveResult(
                status="OK",
                w=w_val,
                achieved_return=float(mu @ w_val),
                achieved_var=achieved_var,
                solver_name=prob.solver_stats.solver_name if prob.solver_stats else str(solver),
                message=str(prob.status),
            )

    return SolveResult(
        status="FAIL",
        w=None,
        achieved_return=float("nan"),
        achieved_var=float("nan"),
        solver_name="",
        message=last_err or str(last_status),
    )


__all__ = ["SOLVER_ORDER", "SolveResult", "solve_extreme_return", "solve_target_qp"]
