from typing import Dict, Optional, Sequence

import numpy as np

from refmvo.core.diagnostics import build_diagnostics_df
from refmvo.core.moments import nearest_psd, shrink_cov_from_samples
from refmvo.core.solvers import SOLVER_ORDER, solve_target_qp
from refmvo.utils.constants import PERIODS_PER_YEAR


def _ensure_psd(cov: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    cov = 0.5 * (cov + cov.T)
    eig_min = float(np.min(np.linalg.eigvalsh(cov)))
    if eig_min < -tol:
        cov = nearest_psd(cov)
    return cov


def _sample_parametric(rng: np.random.Generator, mu: np.ndarray, cov: np.ndarray, T: int) -> np.ndarray:
    return rng.multivariate_normal(mu, cov, size=T)


def _sample_nonparametric(
    rng: np.random.Generator, returns: np.ndarray, T: int
) -> np.ndarray:
    idx = rng.integers(0, returns.shape[0], size=T)
    return returns[idx, :]


def resample_frontier_with_coverage(
    mu_base: np.ndarray,
    cov_base: np.ndarray,
    bounds: np.ndarray,
    targets: np.ndarray,
    resamples: int,
    resample_years: float,
    frequency: str,
    seed: int,
    eps: float = 1e-6,
    target_bands: Optional[np.ndarray] = None,
    min_valid_bootstraps: int = 10,
    returns: Optional[np.ndarray] = None,
    mode: str = "Parametric",
    shrinkage: bool = False,
    solvers: Sequence[str] = SOLVER_ORDER,
) -> Dict[str, object]:
    """Resample frontiers and compute coverage by target return."""
    rng = np.random.default_rng(seed)
    mu_base = np.asarray(mu_base, dtype=float)
    cov_base = np.asarray(cov_base, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    targets = np.asarray(targets, dtype=float)
    target_bands = None if target_bands is None else np.asarray(target_bands, dtype=float)
    n = len(mu_base)
    k = len(targets)
    periods = PERIODS_PER_YEAR.get((frequency or "").strip().lower(), 1)
    T = int(round(float(resample_years) * periods))
    T = max(T, 2)

    W_samples = np.full((resamples, k, n), np.nan)
    ok_mask = np.zeros((resamples, k), dtype=bool)
    achieved_return = np.full((resamples, k), np.nan)
    achieved_risk = np.full((resamples, k), np.nan)

    use_nonparam = mode.strip().lower().startswith("nonparam")
    if use_nonparam and returns is None:
        raise ValueError("Nonparametric resampling requested but no returns data supplied.")
    if use_nonparam:
        returns = np.asarray(returns, dtype=float)
    else:
        cov_base = _ensure_psd(cov_base)

    use_bands = target_bands is not None and target_bands.size == k
    for b in range(resamples):
        if use_nonparam:
            X = _sample_nonparametric(rng, returns, T)
        else:
            X = _sample_parametric(rng, mu_base, cov_base, T)

        mu_b = X.mean(axis=0)
        if shrinkage:
            cov_b = shrink_cov_from_samples(X)
        else:
            cov_b = np.cov(X, rowvar=False, ddof=1)
        cov_b = nearest_psd(cov_b)

        for j, R in enumerate(targets):
            if use_bands:
                eps_j = float(target_bands[j])
                res = solve_target_qp(mu_b, cov_b, bounds, float(R), eps=eps_j, eps_rel=0.0, solvers=solvers)
            else:
                res = solve_target_qp(mu_b, cov_b, bounds, float(R), eps=eps, solvers=solvers)
            if res.status == "OK" and res.w is not None:
                W_samples[b, j, :] = res.w
                ok_mask[b, j] = True
                achieved_return[b, j] = res.achieved_return
                achieved_risk[b, j] = np.sqrt(res.achieved_var)

    n_valid = ok_mask.sum(axis=0)
    coverage = n_valid / max(int(resamples), 1)

    W_bar = np.full((k, n), np.nan)
    for j in range(k):
        if n_valid[j] >= int(min_valid_bootstraps):
            W_bar[j, :] = np.nanmean(W_samples[:, j, :], axis=0)

    ret_bar = np.full(k, np.nan)
    risk_bar = np.full(k, np.nan)
    if np.any(np.isfinite(W_bar)):
        ret_bar = (W_bar @ mu_base).ravel()
        risk_bar = np.sqrt(np.einsum("ij,jk,ik->i", W_bar, cov_base, W_bar))

    diagnostics_df = build_diagnostics_df(
        targets=targets,
        coverage=coverage,
        n_valid=n_valid,
        resampled_ret=ret_bar,
        resampled_risk=risk_bar,
    )

    return {
        "targets": np.asarray(targets, dtype=float),
        "W_bar": W_bar,
        "rets": ret_bar,
        "risks": risk_bar,
        "W_samples": W_samples,
        "ok_mask": ok_mask,
        "coverage": coverage,
        "n_valid": n_valid,
        "achieved_return": achieved_return,
        "achieved_risk": achieved_risk,
        "T": T,
        "diagnostics_df": diagnostics_df,
    }


__all__ = ["resample_frontier_with_coverage"]
