import numpy as np
from scipy.optimize import brentq
from sklearn.covariance import LedoitWolf


def nearest_psd(A: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Higham (2002) eigenvalue clipping to enforce PSD."""
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, eps)
    return (vecs * vals) @ vecs.T


def build_covariance(vols: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Construct covariance from volatilities and correlation matrix."""
    vols = np.asarray(vols, dtype=float)
    corr = np.asarray(corr, dtype=float)
    return np.diag(vols) @ corr @ np.diag(vols)


def shrink_cov_from_samples(X: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance from samples."""
    return LedoitWolf().fit(X).covariance_


def geometric_to_arithmetic(G: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Convert geometric mean returns to arithmetic mean returns."""
    G = np.asarray(G, dtype=float)
    s = np.asarray(s, dtype=float)
    mu = np.log1p(G)
    y = (1.0 + np.sqrt(1.0 + 4.0 * (s**2) * np.exp(-2.0 * mu))) / 2.0
    m = (1.0 + G) * np.sqrt(y) - 1.0

    bad = ~np.isfinite(m)
    if np.any(bad):
        for i in np.where(bad)[0]:
            m[i] = _arith_from_geom_brent(G[i], s[i])
    return m


def _g_from_m(m: float, s: float) -> float:
    A = 1.0 + m
    return A / np.sqrt(1.0 + (s**2) / (A**2)) - 1.0


def _arith_from_geom_brent(G: float, s: float) -> float:
    f = lambda m: _g_from_m(m, s) - G
    lo, hi = -0.99, 3.0
    try:
        return brentq(f, lo, hi, maxiter=500)
    except Exception:
        return G + 0.5 * (s**2) / (1.0 + G)


def portfolio_mapping_to_geometric(m_p: float, s_p: float) -> float:
    """Given arithmetic mean and stdev, return compound geometric mean."""
    A = 1.0 + m_p
    y = 1.0 + (s_p**2) / (A**2)
    return A / np.sqrt(y) - 1.0


__all__ = [
    "nearest_psd",
    "build_covariance",
    "shrink_cov_from_samples",
    "geometric_to_arithmetic",
    "portfolio_mapping_to_geometric",
]
