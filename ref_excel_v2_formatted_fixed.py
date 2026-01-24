#!/usr/bin/env python3
"""
Resampled Efficient Frontier (Michaud 1998)
Excel-based implementation with forward forecast resampling and illiquidity penalty

Key Features:
- Forward-looking forecast perturbation (NOT time-series simulation)
- Bayesian posterior sampling option
- Michaud illiquidity penalty
- Risk-ranked resampling
- Geometric/arithmetic conversion
- Excel I/O with embedded charts

Usage:
    python ref_excel_v2.py scenario.xlsx
    python ref_excel_v2.py scenario.xlsx -o results.xlsx
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import time
import json

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import brentq
import cvxpy as cp

# Suppress CVXPY warnings
warnings.filterwarnings('ignore', message='Solution may be inaccurate')
warnings.filterwarnings('ignore', category=UserWarning, module='cvxpy')


# =============================================================================
# Settings and Configuration
# =============================================================================

@dataclass
class FrontierSettings:
    """Settings for frontier computation."""
    # Basic settings
    risk_free_rate: float = 0.035
    long_only: bool = True
    global_min_weight: float = 0.0
    global_max_weight: float = 1.0
    frontier_points: int = 40
    resamples: int = 200
    random_seed: int = 123
    
    # Resampling approach (NEW)
    use_forward_forecasts: bool = True  # Recommended: perturb forecasts directly
    forecast_uncertainty_pct: float = 0.02  # ±2% std error on forecasts
    perturb_volatility: bool = False  # Also perturb volatilities (±10%)
    perturb_correlation: bool = False  # Also perturb correlations (±0.05)
    
    # Alternative: Bayesian posterior sampling
    use_bayesian_resampling: bool = False  # Alternative to direct perturbation
    bayesian_kappa: float = 20.0  # Equivalent sample size (precision)
    
    # Alternative: Historical time-series bootstrap (NEW)
    use_historical_data: bool = False  # Bootstrap from actual historical returns
    historical_block_size: int = 1  # Block size: 1=standard, 3-6=monthly blocks
    
    # Deprecated (only for parametric time-series approach)
    resample_years: int = 30  # Only used if all flags above are False
    
    # Illiquidity penalty (Michaud) (NEW)
    use_illiquidity_penalty: bool = False  # Enable quadratic penalty
    illiquidity_penalty_lambda: float = 1.0  # Penalty strength (0=none, higher=stronger)
    
    # Other settings
    shrinkage: Optional[str] = None  # None or 'LedoitWolf'
    frequency: str = 'Annual'
    use_compound_return: bool = True
    target_band_multiplier: float = 0.5
    min_target_band: float = 1e-6
    min_valid_bootstraps: int = 10
    coverage_threshold: float = 0.80
    

@dataclass
class RuntimeInfo:
    """Runtime information for diagnostics."""
    start_time: float = field(default_factory=time.time)
    load_time: float = 0.0
    classic_time: float = 0.0
    resample_time: float = 0.0
    write_time: float = 0.0
    total_time: float = 0.0
    solver_used: str = "Unknown"
    n_solver_failures: int = 0


# =============================================================================
# Moments and Conversions
# =============================================================================

def geometric_to_arithmetic(g: np.ndarray, sigma: np.ndarray, 
                           threshold: float = 0.40) -> np.ndarray:
    """
    Convert geometric (compound) returns to arithmetic (simple) means.
    
    For σ < threshold: uses approximation m ≈ g + 0.5σ²/(1+g)
    For σ ≥ threshold: uses exact formula via equation solving
    
    Parameters
    ----------
    g : array_like
        Geometric (compound) returns
    sigma : array_like
        Volatilities (standard deviations)
    threshold : float
        Volatility threshold for approximation vs exact
        
    Returns
    -------
    m : np.ndarray
        Arithmetic means
    """
    g = np.asarray(g, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Check for NaN or invalid values
    if np.any(np.isnan(g)):
        raise ValueError(f"Geometric returns contain NaN values. Check your GeometricReturns column.")
    if np.any(np.isnan(sigma)):
        raise ValueError(f"Volatilities contain NaN values. Check your Volatilities column.")
    if np.any(g <= -1):
        raise ValueError(f"Geometric returns must be > -100% (found: {g[g <= -1]})")
    
    m = np.zeros_like(g)
    
    # Simple approximation for low volatility
    low_vol = sigma < threshold
    m[low_vol] = g[low_vol] + 0.5 * sigma[low_vol]**2 / (1 + g[low_vol])
    
    # Exact formula for high volatility
    high_vol = ~low_vol
    for i in np.where(high_vol)[0]:
        mu_log = np.log(1 + g[i])
        
        # Solve: (1+g)²·y - (1+g)² - σ²·exp(-2μ) = 0
        def equation(y):
            return (1+g[i])**2 * y - (1+g[i])**2 - sigma[i]**2 * np.exp(-2*mu_log)
        
        try:
            y_sol = brentq(equation, 0.5, 3.0)
            m[i] = (1 + g[i]) * np.sqrt(y_sol) - 1
        except ValueError:
            warnings.warn(f"Root finding failed for asset {i}, using approximation")
            m[i] = g[i] + 0.5 * sigma[i]**2 / (1 + g[i])
    
    return m


def arithmetic_to_geometric(m: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Convert arithmetic means to geometric returns.
    
    Uses exact formula: g = (1+m)/sqrt(1 + σ²/(1+m)²) - 1
    """
    m = np.asarray(m, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    variance_ratio = sigma**2 / (1 + m)**2
    g = (1 + m) / np.sqrt(1 + variance_ratio) - 1
    
    return g


def portfolio_arithmetic_to_geometric(mu_p: float, sigma_p: float) -> float:
    """Convert portfolio arithmetic return to geometric."""
    return (1 + mu_p) / np.sqrt(1 + sigma_p**2 / (1 + mu_p)**2) - 1


def portfolio_geometric_return_from_weights(
    weights: np.ndarray,
    geometric_returns: np.ndarray,
    covariance: np.ndarray
) -> Tuple[float, float]:
    """
    Compute portfolio geometric return and volatility from weights.
    
    This is the EXACT way to compute what an investor actually receives.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    geometric_returns : np.ndarray
        Asset geometric (compound) returns
    covariance : np.ndarray
        Asset covariance matrix
        
    Returns
    -------
    g_p : float
        Portfolio geometric return (what investor receives)
    sigma_p : float
        Portfolio volatility
        
    Notes
    -----
    The portfolio geometric return is:
        (1 + g_p) = ∏(1 + g_i)^w_i
    
    Taking logs:
        ln(1 + g_p) = Σ w_i · ln(1 + g_i)
        
    Therefore:
        g_p = exp(Σ w_i · ln(1 + g_i)) - 1
    """
    # Portfolio volatility
    sigma_p = np.sqrt(weights @ covariance @ weights)
    
    # Portfolio geometric return (log-weighted)
    log_returns = np.log(1 + geometric_returns)
    log_portfolio_return = weights @ log_returns
    g_p = np.exp(log_portfolio_return) - 1
    
    return g_p, sigma_p


def build_covariance_from_correlation(vols: np.ndarray, 
                                     corr: np.ndarray) -> np.ndarray:
    """
    Build covariance matrix from volatilities and correlation.
    
    Enforces positive semi-definite via eigenvalue truncation.
    """
    D = np.diag(vols)
    S = D @ corr @ D
    
    # Ensure PSD
    eigvals, eigvecs = linalg.eigh(S)
    
    if np.min(eigvals) < 0:
        min_eigval = np.min(eigvals)
        pct_adjustment = -min_eigval / np.max(eigvals)
        
        if pct_adjustment > 0.01:
            warnings.warn(
                f"Correlation matrix not PSD, truncating negative eigenvalues "
                f"(adjustment: {pct_adjustment:.2%})"
            )
        
        eigvals = np.maximum(eigvals, 0)
        S = eigvecs @ np.diag(eigvals) @ eigvecs.T
        S = (S + S.T) / 2
    
    return S


# =============================================================================
# Constraints
# =============================================================================

def normalize_bounds(bounds: np.ndarray, 
                    long_only: bool, 
                    global_min: float,
                    global_max: float) -> np.ndarray:
    """
    Normalize bounds array with global constraints.
    """
    bounds = np.asarray(bounds, dtype=float).copy()
    
    # Apply global constraints
    bounds[:, 0] = np.maximum(bounds[:, 0], global_min)
    bounds[:, 1] = np.minimum(bounds[:, 1], global_max)
    
    # Long-only constraint
    if long_only:
        bounds[:, 0] = np.maximum(bounds[:, 0], 0.0)
    
    # Ensure lower <= upper
    if np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("Invalid bounds: lower bound exceeds upper bound")
    
    return bounds


# =============================================================================
# Frontier Computation
# =============================================================================

def compute_target_returns(mu: np.ndarray, 
                          bounds: np.ndarray,
                          n_points: int) -> Dict[str, Any]:
    """
    Compute target return grid for frontier.
    """
    # Check for NaN values in mu
    if np.any(np.isnan(mu)):
        raise ValueError(f"Expected returns (mu) contain NaN values. Check your input data.")
    
    lb, ub = bounds[:, 0], bounds[:, 1]
    
    # Minimum return: allocate to lowest-return assets
    idx_sorted = np.argsort(mu)
    w_min = np.zeros(len(mu))
    remaining = 1.0
    for i in idx_sorted:
        allocation = min(ub[i], remaining)
        w_min[i] = allocation
        remaining -= allocation
        if remaining <= 1e-10:
            break
    
    w_min = w_min / w_min.sum()
    min_return = mu @ w_min
    
    # Maximum return: allocate to highest-return assets
    w_max = np.zeros(len(mu))
    remaining = 1.0
    for i in idx_sorted[::-1]:
        allocation = min(ub[i], remaining)
        w_max[i] = allocation
        remaining -= allocation
        if remaining <= 1e-10:
            break
    
    w_max = w_max / w_max.sum()
    max_return = mu @ w_max
    
    # Create grid
    targets = np.linspace(min_return, max_return, n_points)
    
    return {
        'targets': targets,
        'min_return': min_return,
        'max_return': max_return,
        'w_min': w_min,
        'w_max': w_max
    }


def compute_target_bands(targets: np.ndarray,
                        multiplier: float = 0.5,
                        min_band: float = 1e-6) -> np.ndarray:
    """
    Compute target bands (tolerance around each target).
    """
    if len(targets) < 2:
        return np.full(len(targets), min_band)
    
    spacing = np.diff(targets)
    bands = multiplier * spacing
    bands = np.maximum(bands, min_band)
    
    # First and last points use their neighbor's band
    bands = np.concatenate([[bands[0]], bands])
    
    return bands


class FrontierSolver:
    """
    Clean interface for efficient frontier QP solving with illiquidity penalty.
    """
    
    SOLVER_PREFERENCES = ['OSQP', 'CLARABEL', 'ECOS', 'SCS']
    
    def __init__(self, solver_prefs: Optional[list] = None):
        self.solver_prefs = solver_prefs or self.SOLVER_PREFERENCES
        self.available_solvers = self._check_available()
        self.solver_stats = {s: {'calls': 0, 'successes': 0, 'failures': 0} 
                           for s in self.solver_prefs}
    
    def _check_available(self) -> list:
        """Check which solvers are actually available."""
        available = []
        for solver_name in self.solver_prefs:
            if cp.installed_solvers() and solver_name in cp.installed_solvers():
                available.append(solver_name)
        return available
    
    def solve_target(self, 
                    mu: np.ndarray,
                    S: np.ndarray, 
                    bounds: np.ndarray,
                    target: float,
                    band: float,
                    eps: float = 1e-6,
                    illiquidity_penalty: float = 0.0,
                    illiquidity_scores: Optional[np.ndarray] = None,
                    groups: Optional[list] = None,
                    group_constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Solve minimum variance portfolio for target return range.
        
        With illiquidity penalty (Michaud):
            minimize    w' Σ w + λ × Σ(illiq_i × w_i²)
            subject to  1' w = 1
                        lb <= w <= ub  
                        target - band <= μ' w <= target + band
                        group constraints (if provided)
        
        NEW: Group constraints allow limiting total allocation to categories:
            Σ(w_i where group[i]=="Illiquid") <= max_illiquid
            Σ(w_i where group[i]=="Equities") <= max_equities
            etc.
        
        Parameters
        ----------
        mu : np.ndarray
            Expected returns
        S : np.ndarray
            Covariance matrix
        bounds : np.ndarray
            Weight bounds (n_assets × 2)
        target : float
            Target return
        band : float
            Tolerance around target
        eps : float
            Tolerance for constraint satisfaction
        illiquidity_penalty : float
            Penalty strength λ (0 = no penalty, higher = stronger)
        illiquidity_scores : np.ndarray, optional
            Illiquidity score per asset (0=liquid, 1=illiquid)
        groups : list, optional
            Group label for each asset (e.g., ["Liquid", "Equities", "Illiquid", ...])
        group_constraints : dict, optional
            Dict of (group_name, constraint_type) -> value
            Example: {("Illiquid", "Max"): 0.40, ("Liquid", "Min"): 0.20}
            
        Returns
        -------
        result : dict
            - weights: optimal weights (or None if infeasible)
            - volatility: portfolio volatility
            - return: portfolio return
            - sharpe: Sharpe ratio
            - feasible: bool
            - status: solver status
            - solver: solver used
        """
        n_assets = len(mu)
        w = cp.Variable(n_assets)
        
        # Objective: minimize variance + illiquidity penalty
        objective = cp.quad_form(w, S)
        
        if illiquidity_penalty > 0 and illiquidity_scores is not None:
            # Add quadratic penalty: λ × Σ(illiq_i × w_i²)
            # Ensure no NaN values and create symmetric matrix
            clean_scores = np.nan_to_num(illiquidity_scores, nan=0.0)
            illiq_matrix = np.diag(clean_scores)
            # Force symmetry (diagonal matrices should be symmetric, but cvxpy is strict)
            illiq_matrix = (illiq_matrix + illiq_matrix.T) / 2
            penalty_term = illiquidity_penalty * cp.quad_form(w, illiq_matrix)
            objective = objective + penalty_term
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= bounds[:, 0],
            w <= bounds[:, 1],
            mu @ w >= target - band,
            mu @ w <= target + band
        ]
        
        # NEW: Add group constraints
        if groups and group_constraints:
            unique_groups = set(groups)
            for group_name in unique_groups:
                # Create boolean mask for this group
                mask = np.array([g == group_name for g in groups], dtype=float)
                
                # Check for Max constraint
                if (group_name, 'Max') in group_constraints:
                    max_val = group_constraints[(group_name, 'Max')]
                    # Sum of weights in this group <= max_val
                    constraints.append(mask @ w <= max_val)
                
                # Check for Min constraint
                if (group_name, 'Min') in group_constraints:
                    min_val = group_constraints[(group_name, 'Min')]
                    # Sum of weights in this group >= min_val
                    constraints.append(mask @ w >= min_val)
        
        problem = cp.Problem(cp.Minimize(objective), constraints)
        
        # Try solvers in preference order
        for solver_name in self.available_solvers:
            try:
                self.solver_stats[solver_name]['calls'] += 1
                problem.solve(solver=solver_name, verbose=False)
                
                if problem.status in ['optimal', 'optimal_inaccurate']:
                    weights = w.value
                    
                    # Validate solution
                    if weights is None or not np.all(np.isfinite(weights)):
                        continue
                    
                    # Check constraints
                    budget_error = abs(weights.sum() - 1.0)
                    bounds_ok = np.all(weights >= bounds[:, 0] - eps) and \
                               np.all(weights <= bounds[:, 1] + eps)
                    ret = mu @ weights
                    target_ok = (ret >= target - band - eps) and (ret <= target + band + eps)
                    
                    if budget_error < eps and bounds_ok and target_ok:
                        vol = np.sqrt(weights @ S @ weights)
                        sharpe = (ret - 0.0) / vol if vol > 0 else 0.0
                        
                        self.solver_stats[solver_name]['successes'] += 1
                        
                        return {
                            'weights': weights,
                            'volatility': vol,
                            'return': ret,
                            'sharpe': sharpe,
                            'feasible': True,
                            'status': problem.status,
                            'solver': solver_name
                        }
            
            except Exception:
                self.solver_stats[solver_name]['failures'] += 1
                continue
        
        # All solvers failed
        return {
            'weights': None,
            'volatility': np.nan,
            'return': np.nan,
            'sharpe': np.nan,
            'feasible': False,
            'status': 'failed',
            'solver': 'none'
        }
    
    def get_stats(self) -> Dict:
        """Get solver statistics."""
        return self.solver_stats.copy()


def compute_efficient_frontier(
    mu: np.ndarray,
    S: np.ndarray,
    bounds: np.ndarray,
    targets: np.ndarray,
    target_bands: np.ndarray,
    risk_free_rate: float = 0.0,
    illiquidity_penalty: float = 0.0,
    illiquidity_scores: Optional[np.ndarray] = None,
    groups: Optional[list] = None,
    group_constraints: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Compute efficient frontier with optional illiquidity penalty and group constraints.
    
    Parameters
    ----------
    ... (standard parameters)
    illiquidity_penalty : float
        Penalty strength (0 = no penalty)
    illiquidity_scores : np.ndarray, optional
        Illiquidity score per asset
    groups : list, optional
        Group label for each asset
    group_constraints : dict, optional
        Dict of (group_name, constraint_type) -> value
        
    Returns
    -------
    results : dict
        Frontier results including weights, returns, volatilities, Sharpe ratios
    """
    n_targets = len(targets)
    n_assets = len(mu)
    
    weights = np.full((n_targets, n_assets), np.nan)
    returns = np.full(n_targets, np.nan)
    volatilities = np.full(n_targets, np.nan)
    sharpe_ratios = np.full(n_targets, np.nan)
    feasible = np.zeros(n_targets, dtype=bool)
    
    solver = FrontierSolver()
    
    for k, (target, band) in enumerate(zip(targets, target_bands)):
        result = solver.solve_target(
            mu, S, bounds, target, band,
            illiquidity_penalty=illiquidity_penalty,
            illiquidity_scores=illiquidity_scores,
            groups=groups,
            group_constraints=group_constraints
        )
        
        if result['feasible']:
            weights[k] = result['weights']
            returns[k] = result['return']
            volatilities[k] = result['volatility']
            sharpe_ratios[k] = (result['return'] - risk_free_rate) / result['volatility']
            feasible[k] = True
    
    return {
        'targets': targets,
        'weights': weights,
        'returns': returns,
        'volatilities': volatilities,
        'sharpe_ratios': sharpe_ratios,
        'feasible': feasible,
        'solver_stats': solver.get_stats()
    }


# =============================================================================
# Resampling Functions (NEW - Forward Forecast Approaches)
# =============================================================================

def resample_forecast_perturbation(
    geometric_returns: np.ndarray,
    volatilities: np.ndarray,
    correlation: np.ndarray,
    forecast_uncertainty: float = 0.02,
    perturb_volatility: bool = False,
    perturb_correlation: bool = False,
    random_state: Optional[np.random.RandomState] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample by perturbing forward-looking forecasts.
    
    This is the CORRECT approach for forward-looking forecasts.
    
    Parameters
    ----------
    geometric_returns : np.ndarray
        Forecasted geometric returns (e.g., 10-year forecasts)
    volatilities : np.ndarray
        Forecasted volatilities
    correlation : np.ndarray
        Forecasted correlation matrix
    forecast_uncertainty : float
        Standard error of return forecasts (e.g., 0.02 for ±2%)
        - 0.01-0.015: Very confident
        - 0.02-0.025: Moderately confident (recommended)
        - 0.03-0.04: Uncertain
    perturb_volatility : bool
        If True, also perturb volatilities (±10% relative)
    perturb_correlation : bool
        If True, also perturb correlations (±0.05)
    random_state : np.random.RandomState, optional
        Random state
        
    Returns
    -------
    mu_perturbed : np.ndarray
        Perturbed arithmetic means
    S_perturbed : np.ndarray
        Perturbed covariance matrix (PSD enforced)
    """
    if random_state is None:
        random_state = np.random.RandomState()
    
    n_assets = len(geometric_returns)
    
    # Perturb geometric returns
    g_perturbed = geometric_returns + random_state.normal(
        0, forecast_uncertainty, size=n_assets
    )
    
    # Perturb volatilities if requested
    if perturb_volatility:
        vol_multipliers = 1 + random_state.normal(0, 0.10, size=n_assets)
        vol_multipliers = np.clip(vol_multipliers, 0.5, 2.0)
        sigma_perturbed = volatilities * vol_multipliers
        sigma_perturbed = np.maximum(sigma_perturbed, 0.001)
    else:
        sigma_perturbed = volatilities.copy()
    
    # Perturb correlations if requested
    if perturb_correlation:
        corr_perturbed = correlation.copy()
        n = len(correlation)
        for i in range(n):
            for j in range(i+1, n):
                delta = random_state.normal(0, 0.05)
                new_corr = np.clip(correlation[i,j] + delta, -0.95, 0.95)
                corr_perturbed[i,j] = new_corr
                corr_perturbed[j,i] = new_corr
        
        # Ensure PSD
        eigvals, eigvecs = linalg.eigh(corr_perturbed)
        eigvals = np.maximum(eigvals, 0.01)
        corr_perturbed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        D_inv = np.diag(1.0 / np.sqrt(np.diag(corr_perturbed)))
        corr_perturbed = D_inv @ corr_perturbed @ D_inv
    else:
        corr_perturbed = correlation.copy()
    
    # Convert to arithmetic
    mu_perturbed = geometric_to_arithmetic(g_perturbed, sigma_perturbed)
    
    # Build covariance
    S_perturbed = build_covariance_from_correlation(sigma_perturbed, corr_perturbed)
    
    return mu_perturbed, S_perturbed


def resample_forecast_bayesian(
    geometric_returns: np.ndarray,
    volatilities: np.ndarray,
    correlation: np.ndarray,
    kappa: float = 20.0,
    random_state: Optional[np.random.RandomState] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample using Bayesian posterior distribution.
    
    Assumes forecasts are posterior means with precision kappa.
    
    Parameters
    ----------
    geometric_returns : np.ndarray
        Forecasted geometric returns (posterior mean)
    volatilities : np.ndarray
        Forecasted volatilities
    correlation : np.ndarray
        Forecasted correlation matrix
    kappa : float
        Equivalent sample size (precision of forecasts)
        - kappa = 5-10: Very uncertain (subjective)
        - kappa = 15-20: Moderately confident
        - kappa = 30-50: Very confident
    random_state : np.random.RandomState, optional
        Random state
        
    Returns
    -------
    mu_sample : np.ndarray
        Sampled arithmetic means
    S_sample : np.ndarray
        Sampled covariance
    """
    if random_state is None:
        random_state = np.random.RandomState()
    
    # Convert to arithmetic and build covariance
    mu_forecast = geometric_to_arithmetic(geometric_returns, volatilities)
    Sigma_forecast = build_covariance_from_correlation(volatilities, correlation)
    
    # Posterior covariance of the mean estimate
    # μ ~ N(μ_forecast, Σ_forecast / kappa)
    Sigma_posterior_mean = Sigma_forecast / kappa
    
    # Sample from posterior
    mu_sample = random_state.multivariate_normal(mu_forecast, Sigma_posterior_mean)
    
    # For covariance: keep fixed (could sample from inverse-Wishart)
    S_sample = Sigma_forecast.copy()
    
    return mu_sample, S_sample


def resample_returns(mu: np.ndarray,
                    S: np.ndarray,
                    n_periods: int,
                    random_state: Optional[np.random.RandomState] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample returns from multivariate normal and reestimate moments.
    
    NOTE: This is the DEPRECATED parametric time-series approach.
    Use resample_forecast_perturbation() for forward-looking forecasts.
    Use resample_from_historical_data() for actual historical time-series.
    
    This function simulates having only n_periods of data and seeing
    how parameter estimates would vary.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    
    # Simulate returns
    returns = random_state.multivariate_normal(mu, S, size=n_periods)
    
    # Estimate moments
    mu_resample = returns.mean(axis=0)
    S_resample = np.cov(returns, rowvar=False, ddof=1)
    
    # Ensure PSD
    eigvals, eigvecs = linalg.eigh(S_resample)
    if np.min(eigvals) < 0:
        eigvals = np.maximum(eigvals, 0)
        S_resample = eigvecs @ np.diag(eigvals) @ eigvecs.T
        S_resample = (S_resample + S_resample.T) / 2
    
    return mu_resample, S_resample


def resample_from_historical_data(
    historical_returns: np.ndarray,
    block_size: int = 1,
    random_state: Optional[np.random.RandomState] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap resample from actual historical return data (NEW).
    
    Use this when you have ACTUAL historical returns data (monthly, quarterly, annual)
    rather than forward-looking forecasts.
    
    Parameters
    ----------
    historical_returns : np.ndarray
        Historical returns matrix (T × n_assets), arithmetic returns
        T = number of time periods (e.g., 120 for 10 years monthly)
        Each row = returns for one period across all assets
        
        IMPORTANT: Provide arithmetic (simple) returns, not geometric!
        Example row: [0.01, -0.02, 0.03, ...] for [1%, -2%, 3%, ...]
        
    block_size : int, default=1
        Block size for block bootstrap
        - 1: Standard bootstrap (sample periods independently)
        - 3-6: Block bootstrap for monthly data (preserves short-term correlation)
        - 4: Quarterly data
        - 1: Annual data (already independent)
        
    random_state : np.random.RandomState, optional
        Random state for reproducibility
        
    Returns
    -------
    mu_resample : np.ndarray
        Resampled arithmetic mean returns
    S_resample : np.ndarray
        Resampled covariance matrix (PSD enforced)
        
    Examples
    --------
    >>> # Monthly returns for 10 years (120 months)
    >>> historical = np.array([
    ...     [0.02, 0.01, 0.03],  # Month 1
    ...     [-0.01, 0.005, -0.02],  # Month 2
    ...     # ... 118 more months
    ... ])
    >>> mu, S = resample_from_historical_data(historical, block_size=3)
    
    Notes
    -----
    This implements standard bootstrap resampling:
    - block_size=1: Sample T periods with replacement (standard bootstrap)
    - block_size>1: Sample overlapping blocks to preserve serial correlation
    
    The resampled means and covariances represent one possible dataset
    you might have observed given the underlying data-generating process.
    """
    
    if random_state is None:
        random_state = np.random.RandomState()
    
    T, n_assets = historical_returns.shape
    
    if T < 20:
        warnings.warn(f"Only {T} observations - estimates will be very unstable")
    
    if block_size == 1:
        # Standard bootstrap: sample periods with replacement
        indices = random_state.choice(T, size=T, replace=True)
        resampled_returns = historical_returns[indices, :]
        
    else:
        # Block bootstrap: sample overlapping blocks
        n_blocks = int(np.ceil(T / block_size))
        block_indices = []
        
        for _ in range(n_blocks):
            # Random starting point for block
            start = random_state.randint(0, max(1, T - block_size + 1))
            block_indices.extend(range(start, min(start + block_size, T)))
        
        # Truncate to original length
        block_indices = block_indices[:T]
        resampled_returns = historical_returns[block_indices, :]
    
    # Estimate parameters from resampled data
    mu_resample = resampled_returns.mean(axis=0)
    S_resample = np.cov(resampled_returns, rowvar=False, ddof=1)
    
    # Ensure PSD
    eigvals, eigvecs = linalg.eigh(S_resample)
    if np.min(eigvals) < 0:
        eigvals = np.maximum(eigvals, 0)
        S_resample = eigvecs @ np.diag(eigvals) @ eigvecs.T
        S_resample = (S_resample + S_resample.T) / 2
    
    return mu_resample, S_resample


def resample_frontier_with_coverage(
    mu: np.ndarray,
    S: np.ndarray,
    bounds: np.ndarray,
    targets: np.ndarray,
    target_bands: np.ndarray,
    n_resamples: int,
    n_periods: int,  # May be unused with new approach
    random_seed: int,
    risk_free_rate: float = 0.0,
    min_valid_bootstraps: int = 10,
    shrinkage: Optional[str] = None,
    progress_callback: Optional[callable] = None,
    # NEW PARAMETERS:
    settings: Optional[FrontierSettings] = None,
    geometric_returns: Optional[np.ndarray] = None,
    volatilities: Optional[np.ndarray] = None,
    correlation: Optional[np.ndarray] = None,
    illiquidity_scores: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute Michaud Resampled Efficient Frontier with coverage tracking.
    
    Now supports THREE resampling approaches:
    1. Forward forecast perturbation (RECOMMENDED for forecasts)
    2. Bayesian posterior sampling (Alternative for forecasts)
    3. Time-series simulation (Original, for historical data)
    
    All approaches produce 200 bootstrap frontiers, then average by risk rank.
    
    Parameters
    ----------
    mu, S : np.ndarray
        Original moments (arithmetic)
    bounds : np.ndarray
        Weight bounds
    targets : np.ndarray
        Target returns
    target_bands : np.ndarray
        Tolerance bands for targets
    n_resamples : int
        Number of bootstrap samples (e.g., 200)
    n_periods : int
        Periods per bootstrap (only used for time-series approach)
    random_seed : int
        Random seed
    risk_free_rate : float
        Risk-free rate
    min_valid_bootstraps : int
        Minimum feasible bootstraps required per target
    shrinkage : str, optional
        Covariance shrinkage method
    progress_callback : callable, optional
        Progress reporting function
    settings : FrontierSettings, optional
        Settings object (NEW)
    geometric_returns : np.ndarray, optional
        Original geometric returns (NEW)
    volatilities : np.ndarray, optional
        Original volatilities (NEW)
    correlation : np.ndarray, optional
        Original correlation matrix (NEW)
    illiquidity_scores : np.ndarray, optional
        Illiquidity scores (NEW)
        
    Returns
    -------
    results : dict
        - W_bar: average weights (n_targets × n_assets)
        - returns: average returns
        - volatilities: average volatilities  
        - sharpe_ratios: average Sharpe ratios
        - coverage: coverage per target (fraction feasible)
        - n_valid: count of feasible bootstraps per target
        - feasible: boolean array
        - W_samples: all bootstrap weights
    """
    n_targets = len(targets)
    n_assets = len(mu)
    
    # Storage for bootstrap results
    W_samples = np.full((n_resamples, n_targets, n_assets), np.nan)
    feasible_matrix = np.zeros((n_resamples, n_targets), dtype=bool)
    
    # Run bootstraps
    rng = np.random.RandomState(random_seed)
    
    # Determine illiquidity penalty
    if settings and settings.use_illiquidity_penalty:
        illiq_penalty = settings.illiquidity_penalty_lambda
    else:
        illiq_penalty = 0.0
    
    solver = FrontierSolver()
    
    for b in range(n_resamples):
        if progress_callback:
            progress_callback(b + 1, n_resamples, 
                            f"Bootstrap {b+1}/{n_resamples}")
        
        # RESAMPLE using appropriate method
        if settings and settings.use_historical_data:
            # Method 1: Bootstrap from actual historical data (NEW)
            if not hasattr(settings, '_historical_returns') or settings._historical_returns is None:
                raise ValueError("UseHistoricalData=TRUE but no historical returns provided")
            mu_b, S_b = resample_from_historical_data(
                settings._historical_returns,
                settings.historical_block_size,
                rng
            )
        elif settings and settings.use_forward_forecasts:
            # Method 2: Direct perturbation of forecasts (RECOMMENDED for forecasts)
            mu_b, S_b = resample_forecast_perturbation(
                geometric_returns,
                volatilities,
                correlation,
                settings.forecast_uncertainty_pct,
                settings.perturb_volatility,
                settings.perturb_correlation,
                rng
            )
        elif settings and settings.use_bayesian_resampling:
            # Method 3: Bayesian posterior sampling
            mu_b, S_b = resample_forecast_bayesian(
                geometric_returns,
                volatilities,
                correlation,
                settings.bayesian_kappa,
                rng
            )
        else:
            # Method 4: Parametric time-series simulation (original, deprecated)
            mu_b, S_b = resample_returns(mu, S, n_periods, rng)
        
        # Optional shrinkage (mostly for backwards compatibility)
        if shrinkage:
            # Simplified - full implementation would require raw returns
            pass
        
        # Solve frontier for this bootstrap
        for k, (target, band) in enumerate(zip(targets, target_bands)):
            result = solver.solve_target(
                mu_b, S_b, bounds, target, band,
                illiquidity_penalty=illiq_penalty,
                illiquidity_scores=illiquidity_scores,
                groups=settings._groups if hasattr(settings, '_groups') else None,
                group_constraints=settings._group_constraints if hasattr(settings, '_group_constraints') else None
            )
            
            if result['feasible']:
                W_samples[b, k, :] = result['weights']
                feasible_matrix[b, k] = True
    
    # Aggregate results
    n_valid = feasible_matrix.sum(axis=0)
    coverage = n_valid / n_resamples
    
    # =============================================================================
    # RISK-RANK AVERAGING (Michaud's original method)
    # =============================================================================
    
    # Average weights (only over feasible bootstraps)
    W_bar_rank = np.full((n_targets, n_assets), np.nan)
    returns_bar_rank = np.full(n_targets, np.nan)
    vols_bar_rank = np.full(n_targets, np.nan)
    sharpe_bar_rank = np.full(n_targets, np.nan)
    
    # Also compute portfolio metrics for each bootstrap for later analysis
    returns_samples = np.full((n_resamples, n_targets), np.nan)
    vols_samples = np.full((n_resamples, n_targets), np.nan)
    
    for b in range(n_resamples):
        for k in range(n_targets):
            if feasible_matrix[b, k]:
                w = W_samples[b, k, :]
                # Use ORIGINAL moments to compute metrics
                returns_samples[b, k] = mu @ w
                vols_samples[b, k] = np.sqrt(w @ S @ w)
    
    for k in range(n_targets):
        if n_valid[k] >= min_valid_bootstraps:
            # Average over feasible bootstraps
            valid_weights = W_samples[feasible_matrix[:, k], k, :]
            W_bar_rank[k] = valid_weights.mean(axis=0)
            
            # Compute portfolio metrics with averaged weights using ORIGINAL moments
            returns_bar_rank[k] = mu @ W_bar_rank[k]
            vols_bar_rank[k] = np.sqrt(W_bar_rank[k] @ S @ W_bar_rank[k])
            sharpe_bar_rank[k] = (returns_bar_rank[k] - risk_free_rate) / vols_bar_rank[k]
    
    # =============================================================================
    # RISK-BUCKET AVERAGING (Alternative method)
    # =============================================================================
    
    # Define risk buckets: use the rank-averaged volatilities as bucket centers
    # Bucket width: 0.5% (can be adjusted)
    bucket_width = 0.005
    
    W_bar_bucket = np.full((n_targets, n_assets), np.nan)
    returns_bar_bucket = np.full(n_targets, np.nan)
    vols_bar_bucket = np.full(n_targets, np.nan)
    sharpe_bar_bucket = np.full(n_targets, np.nan)
    bucket_counts = np.zeros(n_targets, dtype=int)
    
    for k in range(n_targets):
        if not np.isnan(vols_bar_rank[k]):
            # Use rank-averaged vol as bucket center
            target_vol = vols_bar_rank[k]
            
            # Find all portfolios (across all bootstraps) within this bucket
            in_bucket = (vols_samples >= target_vol - bucket_width) & \
                       (vols_samples <= target_vol + bucket_width) & \
                       ~np.isnan(vols_samples)
            
            if in_bucket.sum() >= min_valid_bootstraps:
                # Collect weights of all portfolios in bucket
                weights_in_bucket = []
                for b in range(n_resamples):
                    for k_target in range(n_targets):
                        if in_bucket[b, k_target]:
                            weights_in_bucket.append(W_samples[b, k_target, :])
                
                if weights_in_bucket:
                    weights_in_bucket = np.array(weights_in_bucket)
                    W_bar_bucket[k] = weights_in_bucket.mean(axis=0)
                    bucket_counts[k] = len(weights_in_bucket)
                    
                    # Compute metrics using ORIGINAL moments
                    returns_bar_bucket[k] = mu @ W_bar_bucket[k]
                    vols_bar_bucket[k] = np.sqrt(W_bar_bucket[k] @ S @ W_bar_bucket[k])
                    sharpe_bar_bucket[k] = (returns_bar_bucket[k] - risk_free_rate) / vols_bar_bucket[k]
    
    # Mark targets as feasible if they have enough valid bootstraps
    feasible_targets = n_valid >= min_valid_bootstraps
    
    return {
        # Risk-rank averaging (original Michaud)
        'W_bar': W_bar_rank,
        'returns': returns_bar_rank,
        'volatilities': vols_bar_rank,
        'sharpe_ratios': sharpe_bar_rank,
        
        # Risk-bucket averaging (alternative)
        'W_bar_bucket': W_bar_bucket,
        'returns_bucket': returns_bar_bucket,
        'volatilities_bucket': vols_bar_bucket,
        'sharpe_ratios_bucket': sharpe_bar_bucket,
        'bucket_counts': bucket_counts,
        
        # Metadata
        'coverage': coverage,
        'n_valid': n_valid,
        'feasible': feasible_targets,
        
        # Individual bootstrap data (for analysis and plotting)
        'W_samples': W_samples,
        'returns_samples': returns_samples,
        'vols_samples': vols_samples,
        'feasible_matrix': feasible_matrix,
        
        'solver_stats': solver.get_stats()
    }


# =============================================================================
# I/O - Excel Reading
# =============================================================================

PERIODS_PER_YEAR = {
    'annual': 1,
    'monthly': 12,
    'weekly': 52,
    'daily': 252
}

def read_excel_inputs(filepath: str) -> Dict[str, Any]:
    """
    Read scenario inputs from Excel file.

    Expected sheets:
        - Assets: asset characteristics (with optional Group, Currency, Fees, IlliquidityScore)
        - Correlation: correlation matrix
        - Settings: frontier settings (optional)
        - HistoricalReturns: optional, required only if UseHistoricalData=TRUE

    Returns dictionary with all inputs including illiquidity scores.
    """
    excel_file = pd.ExcelFile(filepath, engine='openpyxl')

    # -------------------------------------------------------------------------
    # Settings (read FIRST so defaults are available everywhere)
    # -------------------------------------------------------------------------
    # Helper function to safely convert boolean
    def to_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return bool(val)
        if isinstance(val, str):
            return val.strip().upper() in ['TRUE', 'YES', 'Y', '1']
        return bool(val)

    if 'Settings' in excel_file.sheet_names:
        settings_df = pd.read_excel(excel_file, sheet_name='Settings')
        if {'Key', 'Value'}.issubset(set(settings_df.columns)):
            settings_dict = dict(zip(settings_df['Key'], settings_df['Value']))
        else:
            settings_dict = {}
            print("Note: Settings sheet found but missing 'Key'/'Value' columns; using defaults")
    else:
        settings_dict = {}
        print("Note: No Settings sheet found, using defaults")

    # -------------------------------------------------------------------------
    # Assets
    # -------------------------------------------------------------------------
    assets_df = pd.read_excel(excel_file, sheet_name='Assets')
    required_cols = ['Asset', 'GeometricReturn', 'Volatility', 'MinWeight', 'MaxWeight']
    if not all(col in assets_df.columns for col in required_cols):
        raise ValueError(f"Assets sheet must contain columns: {required_cols}")

    asset_names = assets_df['Asset'].tolist()
    geometric_returns = assets_df['GeometricReturn'].values.astype(float)
    volatilities = assets_df['Volatility'].values.astype(float)
    bounds = assets_df[['MinWeight', 'MaxWeight']].values.astype(float)

    # Optional Group column
    if 'Group' in assets_df.columns:
        groups = assets_df['Group'].tolist()
    else:
        groups = ['Uncategorized'] * len(assets_df)
        print("Note: No Group column found, all assets uncategorized")

    # Optional Currency column
    if 'Currency' in assets_df.columns:
        currencies = assets_df['Currency'].tolist()
    else:
        currencies = ['Unknown'] * len(assets_df)
        print("Note: No Currency column found")

    # Optional Fees column -> net returns
    if 'Fees' in assets_df.columns:
        fees = assets_df['Fees'].values.astype(float)
        geometric_returns_net = geometric_returns - fees
        geometric_returns_gross = geometric_returns.copy()
        print("  Found Fees column: returns will be adjusted to net-of-fee")
    else:
        fees = np.zeros(len(assets_df), dtype=float)
        geometric_returns_net = geometric_returns.copy()
        geometric_returns_gross = geometric_returns.copy()
        print("Note: No Fees column found, using gross returns")

    # Optional IlliquidityScore column
    if 'IlliquidityScore' in assets_df.columns:
        illiquidity_scores = assets_df['IlliquidityScore'].values.astype(float)
    else:
        illiquidity_scores = np.zeros(len(assets_df), dtype=float)
        print("Note: No IlliquidityScore column found, assuming all assets equally liquid")

    # Optional ENEB column (Equity-Normalized Equivalent Beta)
    if 'ENEB' in assets_df.columns:
        eneb_factors = assets_df['ENEB'].values.astype(float)
        print("  Found ENEB column: will calculate equity-normalized risk for portfolios")
    else:
        eneb_factors = np.ones(len(assets_df), dtype=float)
        print("Note: No ENEB column found, using 1.0 for all assets")

    # -------------------------------------------------------------------------
    # Correlation
    # -------------------------------------------------------------------------
    corr_df = pd.read_excel(excel_file, sheet_name='Correlation')
    corr_df = corr_df.set_index(corr_df.columns[0])
    correlation = corr_df.values.astype(float)

    # Validate correlation
    if correlation.shape[0] != len(asset_names) or correlation.shape[1] != len(asset_names):
        raise ValueError("Correlation matrix size doesn't match number of assets")
    if not np.allclose(correlation, correlation.T):
        warnings.warn("Correlation matrix not symmetric, symmetrizing")
        correlation = (correlation + correlation.T) / 2
    if not np.allclose(np.diag(correlation), 1.0):
        warnings.warn("Correlation diagonal not all 1.0")

    # Build covariance (PSD enforced)
    covariance = build_covariance_from_correlation(volatilities, correlation)

    # -------------------------------------------------------------------------
    # Decide whether to use net or gross returns (defaults to net if available)
    # -------------------------------------------------------------------------
    use_net_returns = to_bool(settings_dict.get('UseNetReturns', True))
    geometric_returns_for_opt = geometric_returns_net if use_net_returns else geometric_returns_gross

    # Convert to arithmetic returns
    mu_arithmetic = geometric_to_arithmetic(geometric_returns_for_opt, volatilities)

    if use_net_returns and fees.sum() > 0:
        avg_fee = float((geometric_returns_gross - geometric_returns_net).mean())
        print(f"  Using net-of-fee returns (avg fee drag: {avg_fee:.2%})")

    # -------------------------------------------------------------------------
    # Group constraints
    # -------------------------------------------------------------------------
    group_constraints = {}
    use_group_constraints = to_bool(settings_dict.get('UseGroupConstraints', False))
    if use_group_constraints:
        for key, val in settings_dict.items():
            if isinstance(key, str) and key.startswith('GroupConstraint_'):
                # Format: GroupConstraint_Illiquid_Max or GroupConstraint_Equities_Min
                parts = key.replace('GroupConstraint_', '').split('_')
                if len(parts) == 2:
                    group_name, constraint_type = parts
                    if constraint_type in ['Min', 'Max']:
                        try:
                            group_constraints[(group_name, constraint_type)] = float(val)
                        except Exception:
                            raise ValueError(f"Invalid group constraint value for {key}: {val}")

        if group_constraints:
            print(f"  Found {len(group_constraints)} group constraints:")
            for (grp, typ), val in group_constraints.items():
                print(f"    {grp} {typ}: {val:.1%}")

    # -------------------------------------------------------------------------
    # Build settings object
    # -------------------------------------------------------------------------
    settings = FrontierSettings(
        risk_free_rate=float(settings_dict.get('RiskFreeRate', 0.035)),
        long_only=to_bool(settings_dict.get('LongOnly', True)),
        global_min_weight=float(settings_dict.get('GlobalMinWeight', 0.0)),
        global_max_weight=float(settings_dict.get('GlobalMaxWeight', 1.0)),
        frontier_points=int(settings_dict.get('FrontierPoints', 40)),
        resamples=int(settings_dict.get('Resamples', 200)),
        random_seed=int(settings_dict.get('RandomSeed', 123)),

        # Forward forecast settings
        use_forward_forecasts=to_bool(settings_dict.get('UseForwardForecasts', True)),
        forecast_uncertainty_pct=float(settings_dict.get('ForecastUncertaintyPct', 0.02)),
        perturb_volatility=to_bool(settings_dict.get('PerturbVolatility', False)),
        perturb_correlation=to_bool(settings_dict.get('PerturbCorrelation', False)),

        # Bayesian settings
        use_bayesian_resampling=to_bool(settings_dict.get('UseBayesianResampling', False)),
        bayesian_kappa=float(settings_dict.get('BayesianKappa', 20.0)),

        # Historical data settings
        use_historical_data=to_bool(settings_dict.get('UseHistoricalData', False)),
        historical_block_size=int(settings_dict.get('HistoricalBlockSize', 1)),

        # Illiquidity penalty
        use_illiquidity_penalty=to_bool(settings_dict.get('UseIlliquidityPenalty', False)),
        illiquidity_penalty_lambda=float(settings_dict.get('IlliquidityPenaltyLambda', 1.0)),

        # Backwards compatibility
        resample_years=int(settings_dict.get('ResampleYears', 30)),

        # Other settings
        shrinkage=settings_dict.get('Shrinkage', None) if settings_dict.get('Shrinkage') != 'None' else None,
        frequency=str(settings_dict.get('Frequency', 'Annual')),
        use_compound_return=to_bool((settings_dict.get('FrontierYAxis', 'Compound') == 'Compound'))
    )

    # Historical returns (required only if UseHistoricalData is True)
    if settings.use_historical_data:
        if 'HistoricalReturns' in excel_file.sheet_names:
            hist_df = pd.read_excel(excel_file, sheet_name='HistoricalReturns')
            # First column is date, rest are returns
            hist_df = hist_df.iloc[:, 1:]  # Drop date column
            if hist_df.shape[1] != len(asset_names):
                raise ValueError(
                    f"HistoricalReturns has {hist_df.shape[1]} columns but "
                    f"Assets has {len(asset_names)} assets"
                )
            historical_returns = hist_df.values.astype(float)
            print(f"  Loaded {len(historical_returns)} periods of historical returns")
            settings._historical_returns = historical_returns
        else:
            raise ValueError(
                "UseHistoricalData=TRUE but no 'HistoricalReturns' sheet found. "
                "Add a HistoricalReturns sheet with columns: Date, Asset1, Asset2, ... "
                "Each row = one period's returns (arithmetic, e.g., 0.02 for 2%)"
            )

    # Store groups and group constraints on settings for access in bootstrap loop
    settings._groups = groups
    settings._group_constraints = group_constraints

    return {
        'assets_df': assets_df,
        'asset_names': asset_names,
        'geometric_returns': geometric_returns_for_opt,  # Net or gross depending on setting
        'geometric_returns_gross': geometric_returns_gross,
        'geometric_returns_net': geometric_returns_net,
        'volatilities': volatilities,
        'correlation': correlation,
        'covariance': covariance,
        'mu_arithmetic': mu_arithmetic,
        'bounds': bounds,
        'illiquidity_scores': illiquidity_scores,
        'eneb_factors': eneb_factors,
        'groups': groups,
        'currencies': currencies,
        'fees': fees,
        'use_net_returns': use_net_returns,
        'group_constraints': group_constraints,
        'settings': settings
    }


# =============================================================================
# Main Analysis Class
# =============================================================================

class ResampledFrontierAnalysis:
    """
    Main analysis class for Resampled Efficient Frontier.
    """
    
    def __init__(self, input_filepath: str, verbose: bool = True):
        self.input_filepath = input_filepath
        self.verbose = verbose
        self.inputs = None
        self.classic_results = None
        self.resampled_results = None
        self.runtime = RuntimeInfo()
    
    def log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(message)
    
    def load(self):
        """Load inputs from Excel file."""
        self.log(f"Loading inputs from {self.input_filepath}...")
        start = time.time()
        
        self.inputs = read_excel_inputs(self.input_filepath)
        
        self.runtime.load_time = time.time() - start
        self.log(f"  Loaded {len(self.inputs['asset_names'])} assets "
                f"({self.runtime.load_time:.2f}s)")
        
        # Report resampling mode
        if self.inputs['settings'].use_historical_data:
            block = self.inputs['settings'].historical_block_size
            self.log(f"  Using historical data bootstrap (block size={block})")
        elif self.inputs['settings'].use_forward_forecasts:
            unc = self.inputs['settings'].forecast_uncertainty_pct
            self.log(f"  Using forward forecast perturbation (±{unc:.1%} uncertainty)")
        elif self.inputs['settings'].use_bayesian_resampling:
            kappa = self.inputs['settings'].bayesian_kappa
            self.log(f"  Using Bayesian posterior sampling (κ={kappa:.0f})")
        else:
            years = self.inputs['settings'].resample_years
            self.log(f"  Using parametric time-series simulation ({years} years)")
        
        # Report illiquidity penalty
        if self.inputs['settings'].use_illiquidity_penalty:
            lam = self.inputs['settings'].illiquidity_penalty_lambda
            self.log(f"  Using illiquidity penalty (λ={lam:.1f})")
    
    def run(self):
        """Run the complete analysis."""
        if self.inputs is None:
            self.load()
        
        inputs = self.inputs
        settings = inputs['settings']
        mu = inputs['mu_arithmetic']
        S = inputs['covariance']
        bounds = normalize_bounds(
            inputs['bounds'],
            settings.long_only,
            settings.global_min_weight,
            settings.global_max_weight
        )
        
        # Compute target grid
        self.log("Computing target return grid...")
        target_info = compute_target_returns(mu, bounds, settings.frontier_points)
        targets = target_info['targets']
        target_bands = compute_target_bands(
            targets,
            settings.target_band_multiplier,
            settings.min_target_band
        )
        
        self.log(f"  Target range: [{targets[0]:.2%}, {targets[-1]:.2%}]")
        
        # Determine illiquidity penalty
        illiq_penalty = (settings.illiquidity_penalty_lambda 
                        if settings.use_illiquidity_penalty else 0.0)
        
        # Compute classical frontier
        self.log(f"Computing classical efficient frontier ({settings.frontier_points} points)...")
        start = time.time()
        
        self.classic_results = compute_efficient_frontier(
            mu, S, bounds, targets, target_bands, 
            settings.risk_free_rate,
            illiquidity_penalty=illiq_penalty,
            illiquidity_scores=inputs['illiquidity_scores'],
            groups=inputs.get('groups'),
            group_constraints=inputs.get('group_constraints')
        )
        
        self.runtime.classic_time = time.time() - start
        n_feasible = self.classic_results['feasible'].sum()
        self.log(f"  Classical frontier complete: {n_feasible}/{settings.frontier_points} "
                f"points feasible ({self.runtime.classic_time:.2f}s)")
        
        # Compute resampled frontier
        self.log(f"Computing resampled frontier ({settings.resamples} bootstraps)...")
        start = time.time()
        
        periods_per_year = PERIODS_PER_YEAR.get(settings.frequency.lower(), 1)
        n_periods = int(settings.resample_years * periods_per_year)
        
        def progress_callback(current, total, message):
            if current % 50 == 0 or current == total:
                pct = 100 * current / total
                print(f"\r  [{pct:5.1f}%] {message}", end='', flush=True)
                if current == total:
                    print()
        
        self.resampled_results = resample_frontier_with_coverage(
            mu, S, bounds, targets, target_bands,
            settings.resamples,
            n_periods,
            settings.random_seed,
            settings.risk_free_rate,
            settings.min_valid_bootstraps,
            settings.shrinkage,
            progress_callback if self.verbose else None,
            # NEW PARAMETERS:
            settings=settings,
            geometric_returns=inputs['geometric_returns'],
            volatilities=inputs['volatilities'],
            correlation=inputs['correlation'],
            illiquidity_scores=inputs['illiquidity_scores']
        )
        
        self.runtime.resample_time = time.time() - start
        n_feasible_ref = self.resampled_results['feasible'].sum()
        avg_coverage = np.nanmean(self.resampled_results['coverage'])
        self.log(f"  Resampled frontier complete: {n_feasible_ref}/{settings.frontier_points} portfolios constructed")
        self.log(f"  Success rate: {avg_coverage:.1%} of bootstrap scenarios were feasible on average")
        self.log(f"  ({self.runtime.resample_time:.2f}s)")
        
        self.runtime.total_time = (self.runtime.load_time + 
                                  self.runtime.classic_time + 
                                  self.runtime.resample_time)
    
    def get_summary_dict(self) -> Dict[str, Any]:
        """Get summary dictionary for reporting."""
        if self.inputs is None or self.classic_results is None:
            raise ValueError("Must run analysis first")
        
        settings = self.inputs['settings']
        
        # Find tangency portfolio (max Sharpe) in resampled frontier
        resampled_sharpe = self.resampled_results['sharpe_ratios']
        feasible = self.resampled_results['feasible']
        
        if np.any(feasible & np.isfinite(resampled_sharpe)):
            idx_tangency = np.nanargmax(np.where(feasible, resampled_sharpe, -np.inf))
            tangency_weights = self.resampled_results['W_bar'][idx_tangency]
            tangency_return = self.resampled_results['returns'][idx_tangency]
            tangency_vol = self.resampled_results['volatilities'][idx_tangency]
            tangency_sharpe = self.resampled_results['sharpe_ratios'][idx_tangency]
        else:
            tangency_weights = None
            tangency_return = np.nan
            tangency_vol = np.nan
            tangency_sharpe = np.nan
        
        return {
            'n_assets': len(self.inputs['asset_names']),
            'n_frontier_points': settings.frontier_points,
            'n_resamples': settings.resamples,
            'resampling_method': (
                'forecast_perturbation' if settings.use_forward_forecasts 
                else 'bayesian' if settings.use_bayesian_resampling 
                else 'time_series'
            ),
            'forecast_uncertainty': (
                settings.forecast_uncertainty_pct if settings.use_forward_forecasts
                else settings.bayesian_kappa if settings.use_bayesian_resampling
                else settings.resample_years
            ),
            'uses_illiquidity_penalty': settings.use_illiquidity_penalty,
            'illiquidity_penalty_lambda': (
                settings.illiquidity_penalty_lambda if settings.use_illiquidity_penalty else 0.0
            ),
            'classic_feasible': self.classic_results['feasible'].sum(),
            'resampled_feasible': self.resampled_results['feasible'].sum(),
            'avg_coverage': float(np.nanmean(self.resampled_results['coverage'])),
            'tangency_return': float(tangency_return) if not np.isnan(tangency_return) else None,
            'tangency_volatility': float(tangency_vol) if not np.isnan(tangency_vol) else None,
            'tangency_sharpe': float(tangency_sharpe) if not np.isnan(tangency_sharpe) else None,
            'tangency_weights': tangency_weights.tolist() if tangency_weights is not None else None,
            'runtime_seconds': {
                'load': self.runtime.load_time,
                'classical': self.runtime.classic_time,
                'resampled': self.runtime.resample_time,
                'total': self.runtime.total_time
            }
        }


# =============================================================================
# Excel Chart Creation Functions
# =============================================================================

def create_frontier_chart_data(
    classic_results: Dict[str, Any],
    resampled_results: Dict[str, Any],
    geometric_returns: np.ndarray,
    covariance: np.ndarray
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for frontier charts using geometric returns.
    """
    # Classical frontier geometric returns
    classic_weights = classic_results['weights']
    classic_feasible = classic_results['feasible']
    
    classic_data = []
    for idx in range(len(classic_weights)):
        if classic_feasible[idx]:
            w = classic_weights[idx]
            g_p, sigma_p = portfolio_geometric_return_from_weights(
                w, geometric_returns, covariance
            )
            classic_data.append({'Volatility': sigma_p, 'Return': g_p})
    
    classic_df = pd.DataFrame(classic_data)
    
    # Resampled frontier geometric returns
    resampled_weights = resampled_results['W_bar']
    resampled_feasible = resampled_results['feasible']
    resampled_coverage = resampled_results['coverage']
    
    resampled_data = []
    for idx in range(len(resampled_weights)):
        if resampled_feasible[idx] and np.all(np.isfinite(resampled_weights[idx])):
            w = resampled_weights[idx]
            g_p, sigma_p = portfolio_geometric_return_from_weights(
                w, geometric_returns, covariance
            )
            resampled_data.append({
                'Volatility': sigma_p, 
                'Return': g_p,
                'Coverage': resampled_coverage[idx]
            })
    
    resampled_df = pd.DataFrame(resampled_data)
    
    # Bootstrap portfolios - sample to avoid too many points
    W_samples = resampled_results['W_samples']
    n_resamples, n_targets, n_assets = W_samples.shape
    
    bootstrap_data = []
    for b in range(0, n_resamples, 5):  # Sample every 5th
        for k in range(0, n_targets, 2):  # Sample every other target
            w = W_samples[b, k, :]
            if np.all(np.isfinite(w)):
                g_p, sigma_p = portfolio_geometric_return_from_weights(
                    w, geometric_returns, covariance
                )
                bootstrap_data.append({'Volatility': sigma_p, 'Return': g_p})
    
    bootstrap_df = pd.DataFrame(bootstrap_data)
    
    return classic_df, resampled_df, bootstrap_df


def add_frontier_chart_to_sheet(wb, sheet_name: str, 
                                classic_df: pd.DataFrame,
                                resampled_df: pd.DataFrame,
                                bootstrap_df: pd.DataFrame):
    """
    Create frontier comparison chart in Excel.
    """
    from openpyxl.chart import ScatterChart, Reference, Series
    
    ws = wb[sheet_name]

    # Write chart data far to the right (out of normal view)
    # Start at column AZ (column 52) to keep it away from main data
    chart_data_col = 52
    start_row = 2
    
    # Write chart data
    current_col = chart_data_col
    
    # Bootstrap data
    ws.cell(start_row, current_col, "Boot_Vol")
    ws.cell(start_row, current_col + 1, "Boot_Ret")
    for i, (idx, row) in enumerate(bootstrap_df.iterrows()):
        ws.cell(start_row + 1 + i, current_col, row['Volatility'])
        ws.cell(start_row + 1 + i, current_col + 1, row['Return'])
    bootstrap_end_row = start_row + len(bootstrap_df)
    
    # Classical data
    current_col += 3
    ws.cell(start_row, current_col, "Classic_Vol")
    ws.cell(start_row, current_col + 1, "Classic_Ret")
    for i, (idx, row) in enumerate(classic_df.iterrows()):
        ws.cell(start_row + 1 + i, current_col, row['Volatility'])
        ws.cell(start_row + 1 + i, current_col + 1, row['Return'])
    classic_col = current_col
    classic_end_row = start_row + len(classic_df)
    
    # Resampled data
    current_col += 3
    ws.cell(start_row, current_col, "Resamp_Vol")
    ws.cell(start_row, current_col + 1, "Resamp_Ret")
    for i, (idx, row) in enumerate(resampled_df.iterrows()):
        ws.cell(start_row + 1 + i, current_col, row['Volatility'])
        ws.cell(start_row + 1 + i, current_col + 1, row['Return'])
    resamp_col = current_col
    resamp_end_row = start_row + len(resampled_df)
    
    # Create scatter chart
    chart = ScatterChart()
    chart.title = "Efficient Frontier - Geometric Returns"
    chart.style = 2
    chart.x_axis.title = 'Volatility'
    chart.y_axis.title = 'Geometric Return'
    chart.height = 10
    chart.width = 16
    
    # Add series
    xvalues = Reference(ws, min_col=chart_data_col, min_row=start_row+1, max_row=bootstrap_end_row)
    yvalues = Reference(ws, min_col=chart_data_col+1, min_row=start_row+1, max_row=bootstrap_end_row)
    series = Series(yvalues, xvalues, title="Bootstraps")
    chart.series.append(series)
    
    xvalues = Reference(ws, min_col=classic_col, min_row=start_row+1, max_row=classic_end_row)
    yvalues = Reference(ws, min_col=classic_col+1, min_row=start_row+1, max_row=classic_end_row)
    series = Series(yvalues, xvalues, title="Classical")
    chart.series.append(series)
    
    xvalues = Reference(ws, min_col=resamp_col, min_row=start_row+1, max_row=resamp_end_row)
    yvalues = Reference(ws, min_col=resamp_col+1, min_row=start_row+1, max_row=resamp_end_row)
    series = Series(yvalues, xvalues, title="Resampled")
    chart.series.append(series)
    
    chart_position = f"A{max(bootstrap_end_row, classic_end_row, resamp_end_row) + 3}"
    ws.add_chart(chart, chart_position)
    
    return ws


def add_allocation_chart_to_sheet(wb, sheet_name: str, 
                                  weights: np.ndarray,
                                  volatilities: np.ndarray,
                                  feasible: np.ndarray,
                                  asset_names: list,
                                  chart_title: str,
                                  max_points: int = 20):
    """
    Create stacked bar chart of allocations in Excel.
    """
    from openpyxl.chart import BarChart, Reference
    
    ws = wb[sheet_name]
    
    idx = np.where(feasible)[0]
    if len(idx) > max_points:
        step = max(1, len(idx) // max_points)
        idx = idx[::step][:max_points]
    
    if len(idx) == 0:
        return ws
    
    last_col = len(ws[1])
    data_start_col = last_col + 3
    data_start_row = 2
    
    ws.cell(data_start_row, data_start_col, "Volatility")
    for i, point_idx in enumerate(idx):
        ws.cell(data_start_row + 1 + i, data_start_col, f"{volatilities[point_idx]:.1%}")
    
    for j, asset_name in enumerate(asset_names):
        ws.cell(data_start_row, data_start_col + 1 + j, asset_name)
        for i, point_idx in enumerate(idx):
            ws.cell(data_start_row + 1 + i, data_start_col + 1 + j, weights[point_idx, j])
    
    chart = BarChart()
    chart.type = "col"
    chart.grouping = "stacked"
    chart.overlap = 100
    chart.title = chart_title
    chart.style = 10
    chart.y_axis.title = 'Weight'
    chart.x_axis.title = 'Volatility'
    chart.height = 15  # Taller for better visibility
    chart.width = 22  # Wider to accommodate legend on side

    # Position legend to the right side in clear space (not overlapping bars)
    chart.legend.position = 'r'  # 'r' = right
    chart.legend.overlay = False  # Prevent legend from overlapping chart

    cats = Reference(ws, min_col=data_start_col,
                    min_row=data_start_row+1,
                    max_row=data_start_row+len(idx))
    chart.set_categories(cats)

    for j in range(len(asset_names)):
        data = Reference(ws, min_col=data_start_col + 1 + j,
                        min_row=data_start_row,
                        max_row=data_start_row + len(idx))
        chart.add_data(data, titles_from_data=True)

    # Position chart at AJ44 (column 36, row 44)
    chart_position = "AJ44"
    ws.add_chart(chart, chart_position)
    
    return ws


def add_rank_vs_bucket_comparison_chart(wb, sheet_name: str):
    """
    Add comparison chart showing Rank-based vs Bucket-based averaging results.
    
    Shows volatility and return differences between the two methods.
    """
    from openpyxl.chart import ScatterChart, Reference, Series
    
    ws = wb[sheet_name]
    
    # Find data columns
    # Expected columns: Rank, RankBased_Return, RankBased_Volatility, BucketBased_Return, BucketBased_Volatility
    
    # Create chart comparing the two methods on risk-return plot
    chart = ScatterChart()
    chart.title = "Rank-Based vs Bucket-Based Averaging Comparison"
    chart.style = 2
    chart.x_axis.title = 'Volatility'
    chart.y_axis.title = 'Return'
    chart.height = 10
    chart.width = 16
    
    # Find last row with data
    max_row = ws.max_row
    
    # Add Rank-Based series (columns C and B)
    xvalues_rank = Reference(ws, min_col=3, min_row=2, max_row=max_row)  # RankBased_Volatility
    yvalues_rank = Reference(ws, min_col=2, min_row=2, max_row=max_row)  # RankBased_Return
    series_rank = Series(yvalues_rank, xvalues_rank, title="Rank-Based")
    series_rank.marker.symbol = "circle"
    series_rank.marker.size = 7
    series_rank.marker.graphicalProperties.solidFill = "4472C4"  # Blue
    series_rank.graphicalProperties.line.noFill = True
    chart.series.append(series_rank)
    
    # Add Bucket-Based series (columns F and E)
    xvalues_bucket = Reference(ws, min_col=6, min_row=2, max_row=max_row)  # BucketBased_Volatility
    yvalues_bucket = Reference(ws, min_col=5, min_row=2, max_row=max_row)  # BucketBased_Return
    series_bucket = Series(yvalues_bucket, xvalues_bucket, title="Bucket-Based")
    series_bucket.marker.symbol = "triangle"
    series_bucket.marker.size = 7
    series_bucket.marker.graphicalProperties.solidFill = "ED7D31"  # Orange
    series_bucket.graphicalProperties.line.noFill = True
    chart.series.append(series_bucket)
    
    # Position chart to the right of data
    chart_position = "N2"
    ws.add_chart(chart, chart_position)
    
    return ws


def add_bootstrap_frontiers_chart(wb, sheet_name: str, classic, resampled):
    """
    Add chart showing all 200 individual bootstrap efficient frontiers.
    
    This creates a "spaghetti chart" with 200 lines showing the variation
    in efficient frontiers across bootstrap samples.
    """
    from openpyxl.chart import LineChart, ScatterChart, Reference, Series
    from openpyxl.chart.marker import Marker
    
    ws = wb[sheet_name]
    
    # The data structure is:
    # Bootstrap | Rank | Return | Volatility
    # We want to plot one line per bootstrap (200 lines)
    # Each line connects the 40 rank points for that bootstrap
    
    # Create scatter chart with lines
    chart = ScatterChart()
    chart.title = "All 200 Bootstrap Efficient Frontiers"
    chart.style = 2
    chart.x_axis.title = 'Volatility'
    chart.y_axis.title = 'Return'
    chart.height = 12
    chart.width = 20
    
    # Find max row
    max_row = ws.max_row
    
    # Get unique bootstrap numbers (1-200)
    n_bootstraps = 200
    n_ranks = 40
    
    # We'll add a series for each bootstrap
    # This creates 200 lines on the chart
    # Due to Excel limitations, we'll sample every 5th bootstrap (40 lines instead of 200)
    # This is still visually effective and much faster
    
    sample_every = 5  # Show every 5th bootstrap (40 lines total)
    
    for b in range(1, n_bootstraps + 1, sample_every):
        # Find rows for this bootstrap
        # Data starts at row 2
        # Each bootstrap has up to 40 rows
        start_row = 2 + (b - 1) * n_ranks
        end_row = start_row + n_ranks - 1
        
        if end_row <= max_row:
            # X values: Volatility (column D)
            xvalues = Reference(ws, min_col=4, min_row=start_row, max_row=end_row)
            # Y values: Return (column C)
            yvalues = Reference(ws, min_col=3, min_row=start_row, max_row=end_row)
            
            series = Series(yvalues, xvalues, title=f"Boot {b}")
            
            # Make lines thin and semi-transparent looking (light blue/gray)
            series.graphicalProperties.line.solidFill = "B4C7E7"  # Light blue
            series.graphicalProperties.line.width = 15000  # Thin line (EMUs)
            
            # No markers for cleaner look
            series.marker = Marker('none')
            
            chart.series.append(series)
    
    # Add the resampled frontier on top (thicker, different color)
    if resampled and 'volatilities' in resampled and 'returns' in resampled:
        # We need to add this data to the sheet first
        # Add resampled frontier data below the bootstrap data
        
        resampled_start_row = max_row + 3
        ws.cell(row=resampled_start_row, column=1, value="Resampled")
        ws.cell(row=resampled_start_row, column=2, value="Frontier")
        ws.cell(row=resampled_start_row + 1, column=3, value="Return")
        ws.cell(row=resampled_start_row + 1, column=4, value="Volatility")
        
        for k, (ret, vol) in enumerate(zip(resampled['returns'], resampled['volatilities'])):
            if not np.isnan(ret) and not np.isnan(vol):
                ws.cell(row=resampled_start_row + 2 + k, column=3, value=ret)
                ws.cell(row=resampled_start_row + 2 + k, column=4, value=vol)
        
        # Add series for resampled frontier
        resampled_data_rows = resampled_start_row + 2
        resampled_end_row = resampled_data_rows + np.sum(~np.isnan(resampled['returns'])) - 1
        
        xvalues_res = Reference(ws, min_col=4, min_row=resampled_data_rows, max_row=resampled_end_row)
        yvalues_res = Reference(ws, min_col=3, min_row=resampled_data_rows, max_row=resampled_end_row)
        
        series_res = Series(yvalues_res, xvalues_res, title="Resampled Frontier")
        series_res.graphicalProperties.line.solidFill = "FF0000"  # Red
        series_res.graphicalProperties.line.width = 40000  # Thick line
        series_res.marker = Marker('none')
        
        chart.series.append(series_res)
    
    # Add classical frontier too (if available)
    if classic and 'volatilities' in classic and 'returns' in classic:
        classic_start_row = max_row + 50
        ws.cell(row=classic_start_row, column=1, value="Classical")
        ws.cell(row=classic_start_row, column=2, value="Frontier")
        ws.cell(row=classic_start_row + 1, column=3, value="Return")
        ws.cell(row=classic_start_row + 1, column=4, value="Volatility")
        
        for k, (ret, vol) in enumerate(zip(classic['returns'], classic['volatilities'])):
            if not np.isnan(ret) and not np.isnan(vol):
                ws.cell(row=classic_start_row + 2 + k, column=3, value=ret)
                ws.cell(row=classic_start_row + 2 + k, column=4, value=vol)
        
        classic_data_rows = classic_start_row + 2
        classic_end_row = classic_data_rows + np.sum(~np.isnan(classic['returns'])) - 1
        
        xvalues_cl = Reference(ws, min_col=4, min_row=classic_data_rows, max_row=classic_end_row)
        yvalues_cl = Reference(ws, min_col=3, min_row=classic_data_rows, max_row=classic_end_row)
        
        series_cl = Series(yvalues_cl, xvalues_cl, title="Classical Frontier")
        series_cl.graphicalProperties.line.solidFill = "0070C0"  # Dark blue
        series_cl.graphicalProperties.line.width = 40000  # Thick line
        series_cl.marker = Marker('none')
        
        chart.series.append(series_cl)
    
    # Position chart at top
    chart_position = "F2"
    ws.add_chart(chart, chart_position)
    
    return ws


# =============================================================================
# Michaud Rebalancing Test
# =============================================================================

def michaud_rebalancing_test(
    current_portfolio: np.ndarray,
    target_rank: int,
    resampled_results: Dict[str, Any],
    covariance: np.ndarray,
    confidence_level: float = 0.95,
    metric: str = 'relative_variance'
) -> Dict[str, Any]:
    """
    Michaud rebalancing test: Determine if current portfolio needs rebalancing.
    
    Tests whether the current portfolio is statistically different from the
    resampled efficient portfolio at the target risk rank. Uses the distribution
    of bootstrap portfolios to define "statistical equivalence".
    
    Decision Rule:
        If current portfolio's distance is beyond the 95th percentile of
        bootstrap distances, rebalance. Otherwise, current portfolio is
        statistically equivalent to optimal, so don't rebalance.
    
    Parameters
    ----------
    current_portfolio : np.ndarray
        Current portfolio weights (n_assets,)
        Must sum to 1.0
    target_rank : int
        Which risk rank on resampled frontier to test against (0-based)
        E.g., target_rank=19 tests against the 20th portfolio
    resampled_results : dict
        Results from resample_frontier_with_coverage()
        Must contain 'W_samples' and 'W_bar'
    covariance : np.ndarray
        Covariance matrix (n_assets × n_assets)
    confidence_level : float
        Confidence level for rebalancing (default: 0.95)
        Higher = only rebalance when very far from optimal
        Typical values: 0.90 (aggressive), 0.95 (standard), 0.99 (conservative)
    metric : str
        Distance metric to use:
        - 'relative_variance': (P-P0)' Σ (P-P0)  [recommended, default]
        - 'mahalanobis': (P-P0)' Σ^-1 (P-P0)  [alternative]
        
    Returns
    -------
    results : dict
        - need_to_rebalance : bool
            True if should rebalance, False if current is statistically equivalent
        - rebalance_probability : float
            Percentile of current portfolio in bootstrap distribution (0-1)
            E.g., 0.97 means current is farther than 97% of bootstraps
        - current_distance : float
            Distance metric value for current portfolio
        - confidence_threshold : float
            Distance threshold at specified confidence level
        - bootstrap_distribution : np.ndarray
            All bootstrap distances (for plotting/analysis)
        - optimal_portfolio : np.ndarray
            Resampled efficient portfolio weights at target rank
        - weight_differences : np.ndarray
            current_portfolio - optimal_portfolio
        - tracking_error_pct : float
            Tracking error between current and optimal (annualized %)
        - n_bootstraps_used : int
            Number of valid bootstrap portfolios used
        - recommendation : str
            Human-readable recommendation
    
    Examples
    --------
    >>> # Current portfolio slightly drifted
    >>> current = np.array([0.08, 0.19, 0.44, 0.09, 0.12, 0.08])
    >>> result = michaud_rebalancing_test(
    ...     current, rank=20, resampled_results, covariance
    ... )
    >>> print(result['need_to_rebalance'])
    False  # Within normal variation, don't rebalance
    >>> print(f"{result['rebalance_probability']:.1%}")
    72.5%  # Current portfolio at 72.5th percentile
    
    References
    ----------
    Michaud & Michaud (2008), "Efficient Asset Management", Chapter 6
    US Patent 6,003,018: "Portfolio optimization by means of resampled 
                          efficient frontiers"
    """
    
    # Validate inputs
    if not np.isclose(current_portfolio.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Current portfolio weights must sum to 1.0 (got {current_portfolio.sum():.6f})")
    
    # Get optimal (resampled) portfolio at target rank
    P_optimal = resampled_results['W_bar'][target_rank]
    
    if not np.all(np.isfinite(P_optimal)):
        raise ValueError(f"Rank {target_rank} is not feasible in resampled frontier")
    
    # Get all bootstrap portfolios at this rank
    W_samples = resampled_results['W_samples']
    n_resamples, n_targets, n_assets = W_samples.shape
    
    # Compute distance for each bootstrap portfolio vs optimal
    bootstrap_distances = []
    
    for b in range(n_resamples):
        P_boot = W_samples[b, target_rank, :]
        
        if np.all(np.isfinite(P_boot)):
            diff = P_boot - P_optimal
            
            if metric == 'relative_variance':
                # Relative variance: (P-P0)' Σ (P-P0)
                distance = diff @ covariance @ diff
            elif metric == 'mahalanobis':
                # Mahalanobis distance squared: (P-P0)' Σ^-1 (P-P0)
                try:
                    Sigma_inv = linalg.inv(covariance)
                except:
                    Sigma_inv = linalg.pinv(covariance)
                distance = diff @ Sigma_inv @ diff
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            bootstrap_distances.append(distance)
    
    bootstrap_distances = np.array(bootstrap_distances)
    
    if len(bootstrap_distances) < 10:
        warnings.warn(f"Only {len(bootstrap_distances)} valid bootstraps at rank {target_rank}")
    
    # Compute distance for current portfolio
    diff_current = current_portfolio - P_optimal
    
    if metric == 'relative_variance':
        current_distance = diff_current @ covariance @ diff_current
    elif metric == 'mahalanobis':
        try:
            Sigma_inv = linalg.inv(covariance)
        except:
            Sigma_inv = linalg.pinv(covariance)
        current_distance = diff_current @ Sigma_inv @ diff_current
    
    # Percentile in bootstrap distribution
    percentile = (bootstrap_distances < current_distance).sum() / len(bootstrap_distances)
    
    # Threshold at confidence level
    threshold = np.percentile(bootstrap_distances, confidence_level * 100)
    
    # Decision
    need_to_rebalance = percentile > confidence_level
    
    # Tracking error (annualized, %)
    # TE² = (P-P0)' Σ (P-P0), so TE = sqrt(TE²)
    tracking_error_variance = diff_current @ covariance @ diff_current
    tracking_error_pct = np.sqrt(tracking_error_variance) * 100
    
    # Generate recommendation text
    if need_to_rebalance:
        recommendation = (
            f"REBALANCE RECOMMENDED: Current portfolio is at {percentile:.1%} percentile, "
            f"exceeding {confidence_level:.0%} threshold. Distance ({current_distance:.6f}) "
            f"is beyond normal estimation uncertainty."
        )
    else:
        recommendation = (
            f"NO REBALANCING NEEDED: Current portfolio is at {percentile:.1%} percentile, "
            f"within {confidence_level:.0%} threshold. Distance ({current_distance:.6f}) "
            f"is within normal estimation uncertainty."
        )
    
    return {
        'need_to_rebalance': need_to_rebalance,
        'rebalance_probability': percentile,
        'current_distance': current_distance,
        'confidence_threshold': threshold,
        'bootstrap_distribution': bootstrap_distances,
        'optimal_portfolio': P_optimal,
        'weight_differences': diff_current,
        'tracking_error_pct': tracking_error_pct,
        'n_bootstraps_used': len(bootstrap_distances),
        'recommendation': recommendation,
        'metric_used': metric,
        'confidence_level': confidence_level
    }


def test_multiple_ranks(
    current_portfolio: np.ndarray,
    resampled_results: Dict[str, Any],
    covariance: np.ndarray,
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Test current portfolio against all feasible ranks on resampled frontier.
    
    Useful to find which rank the current portfolio is closest to.
    
    Parameters
    ----------
    current_portfolio : np.ndarray
        Current portfolio weights
    resampled_results : dict
        Resampled frontier results
    covariance : np.ndarray
        Covariance matrix
    confidence_level : float
        Confidence level for rebalancing test
        
    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with columns:
        - Rank: Risk rank (0-based)
        - Feasible: Whether this rank is feasible
        - Distance: Distance from optimal at this rank
        - Percentile: Percentile in bootstrap distribution
        - NeedRebalance: Whether rebalancing recommended
        - TrackingError: Tracking error (%)
    """
    
    feasible = resampled_results['feasible']
    n_targets = len(feasible)
    
    results = []
    
    for rank in range(n_targets):
        if not feasible[rank]:
            results.append({
                'Rank': rank,
                'Feasible': False,
                'Distance': np.nan,
                'Percentile': np.nan,
                'NeedRebalance': False,
                'TrackingError': np.nan
            })
            continue
        
        try:
            test_result = michaud_rebalancing_test(
                current_portfolio, rank, resampled_results, 
                covariance, confidence_level
            )
            
            results.append({
                'Rank': rank,
                'Feasible': True,
                'Distance': test_result['current_distance'],
                'Percentile': test_result['rebalance_probability'],
                'NeedRebalance': test_result['need_to_rebalance'],
                'TrackingError': test_result['tracking_error_pct']
            })
        except Exception as e:
            results.append({
                'Rank': rank,
                'Feasible': False,
                'Distance': np.nan,
                'Percentile': np.nan,
                'NeedRebalance': False,
                'TrackingError': np.nan
            })
    
    return pd.DataFrame(results)


# =============================================================================
# Excel Writer with Detailed Risk Rank Analysis
# =============================================================================

def write_detailed_risk_rank_tabs(
    writer,
    resampled_results: Dict[str, Any],
    classic_results: Dict[str, Any],
    inputs: Dict[str, Any],
    max_tabs: int = 40
):
    """
    Write detailed analysis tabs for each risk rank.
    
    Each tab contains:
    - All 200 bootstrap portfolios at that rank
    - Statistics (mean, median, min, max, std) for returns, vols, weights
    - The averaged (resampled) portfolio
    - The classical portfolio (if available)
    
    Parameters
    ----------
    writer : pd.ExcelWriter
        Excel writer object
    resampled_results : dict
        Resampled frontier results
    classic_results : dict
        Classical frontier results
    inputs : dict
        Input data including asset names, geometric returns
    max_tabs : int
        Maximum number of risk rank tabs to create
    """
    from openpyxl.utils.dataframe import dataframe_to_rows
    from openpyxl.chart import ScatterChart, Reference, Series
    
    asset_names = inputs['asset_names']
    n_assets = len(asset_names)
    geometric_returns = inputs['geometric_returns']
    covariance = inputs['covariance']
    eneb_factors = inputs['eneb_factors']

    W_samples = resampled_results['W_samples']  # (n_resamples, n_targets, n_assets)
    n_resamples, n_targets, _ = W_samples.shape
    
    # Limit number of tabs
    n_tabs = min(n_targets, max_tabs)
    
    print(f"\nCreating {n_tabs} detailed portfolio analysis tabs (one per risk level)...")

    for rank in range(n_tabs):
        if rank % 10 == 0:
            print(f"  Progress: tabs {rank+1}-{min(rank+10, n_tabs)}...")
        
        # Get all bootstrap portfolios at this rank
        rank_portfolios = []
        
        for b in range(n_resamples):
            w = W_samples[b, rank, :]
            
            if np.all(np.isfinite(w)):
                # Compute geometric return and volatility
                g_p, sigma_p = portfolio_geometric_return_from_weights(
                    w, geometric_returns, covariance
                )

                # Compute ENEB
                eneb_p = np.dot(w, eneb_factors)

                # Create row
                row = {
                    'Bootstrap': b + 1,
                    'Return': g_p,
                    'Volatility': sigma_p,
                    'ENEB': eneb_p
                }

                # Add weights
                for i, name in enumerate(asset_names):
                    row[f'Weight_{name}'] = w[i]

                rank_portfolios.append(row)
        
        if len(rank_portfolios) == 0:
            continue  # Skip if no valid portfolios
        
        # Create DataFrame
        portfolios_df = pd.DataFrame(rank_portfolios)
        
        # Get resampled (averaged) portfolio
        w_resampled = resampled_results['W_bar'][rank]
        
        if np.all(np.isfinite(w_resampled)):
            g_resamp, sigma_resamp = portfolio_geometric_return_from_weights(
                w_resampled, geometric_returns, covariance
            )
            eneb_resamp = np.dot(w_resampled, eneb_factors)

            resampled_row = {
                'Bootstrap': 'RESAMPLED',
                'Return': g_resamp,
                'Volatility': sigma_resamp,
                'ENEB': eneb_resamp
            }
            for i, name in enumerate(asset_names):
                resampled_row[f'Weight_{name}'] = w_resampled[i]
        else:
            resampled_row = None

        # Get classical portfolio (if feasible)
        if classic_results['feasible'][rank]:
            w_classic = classic_results['weights'][rank]
            g_classic, sigma_classic = portfolio_geometric_return_from_weights(
                w_classic, geometric_returns, covariance
            )
            eneb_classic = np.dot(w_classic, eneb_factors)

            classic_row = {
                'Bootstrap': 'CLASSICAL',
                'Return': g_classic,
                'Volatility': sigma_classic,
                'ENEB': eneb_classic
            }
            for i, name in enumerate(asset_names):
                classic_row[f'Weight_{name}'] = w_classic[i]
        else:
            classic_row = None
        
        # Compute statistics
        stats_data = []
        
        # Return statistics
        returns = portfolios_df['Return'].values
        stats_data.append({
            'Metric': 'Return',
            'Mean': np.mean(returns),
            'Median': np.median(returns),
            'StdDev': np.std(returns),
            'Min': np.min(returns),
            'Max': np.max(returns),
            'Resampled': resampled_row['Return'] if resampled_row else np.nan,
            'Classical': classic_row['Return'] if classic_row else np.nan
        })
        
        # Volatility statistics
        vols = portfolios_df['Volatility'].values
        stats_data.append({
            'Metric': 'Volatility',
            'Mean': np.mean(vols),
            'Median': np.median(vols),
            'StdDev': np.std(vols),
            'Min': np.min(vols),
            'Max': np.max(vols),
            'Resampled': resampled_row['Volatility'] if resampled_row else np.nan,
            'Classical': classic_row['Volatility'] if classic_row else np.nan
        })

        # ENEB statistics
        enebs = portfolios_df['ENEB'].values
        stats_data.append({
            'Metric': 'ENEB',
            'Mean': np.mean(enebs),
            'Median': np.median(enebs),
            'StdDev': np.std(enebs),
            'Min': np.min(enebs),
            'Max': np.max(enebs),
            'Resampled': resampled_row['ENEB'] if resampled_row else np.nan,
            'Classical': classic_row['ENEB'] if classic_row else np.nan
        })

        # Weight statistics for each asset
        for i, name in enumerate(asset_names):
            col_name = f'Weight_{name}'
            weights = portfolios_df[col_name].values
            stats_data.append({
                'Metric': f'Weight_{name}',
                'Mean': np.mean(weights),
                'Median': np.median(weights),
                'StdDev': np.std(weights),
                'Min': np.min(weights),
                'Max': np.max(weights),
                'Resampled': resampled_row[col_name] if resampled_row else np.nan,
                'Classical': classic_row[col_name] if classic_row else np.nan
            })
        
        stats_df = pd.DataFrame(stats_data)

        # Calculate dynamic row positions for special portfolios
        # Stats section has: Return + Volatility + Weight_Asset1...N = 2 + n_assets rows
        n_assets = len(asset_names)
        stats_rows = len(stats_df)  # This is 2 + n_assets

        # Position special portfolios BELOW stats section to avoid overlap
        resampled_row_position = stats_rows + 2  # 2 rows gap after stats
        classical_row_position = stats_rows + 4  # 2 rows gap after resampled

        # Write to Excel
        sheet_name = f'Rank{rank+1:02d}'

        # Section 1: Summary statistics (top)
        stats_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0, startcol=0)

        # Section 2: All bootstrap portfolios (below stats)
        start_row = len(stats_df) + 3
        portfolios_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row, startcol=0)

        # Section 3: Special portfolios (to the right)
        special_start_col = len(portfolios_df.columns) + 2

        if resampled_row:
            resampled_df = pd.DataFrame([resampled_row])
            resampled_df.to_excel(writer, sheet_name=sheet_name, index=False,
                                 startrow=resampled_row_position, startcol=special_start_col)

        if classic_row:
            classic_df = pd.DataFrame([classic_row])
            classic_df.to_excel(writer, sheet_name=sheet_name, index=False,
                               startrow=classical_row_position, startcol=special_start_col)


def add_charts_to_risk_rank_tabs(
    output_path: str,
    resampled_results: Dict[str, Any],
    classic_results: Dict[str, Any],
    inputs: Dict[str, Any],
    max_tabs: int = 40
):
    """
    Add scatter charts to risk rank tabs showing portfolio distributions.
    
    Each chart shows:
    - Gray dots: All 200 bootstrap portfolios
    - Orange diamond: Resampled (averaged) portfolio
    - Blue circle: Classical efficient portfolio
    """
    from openpyxl import load_workbook
    from openpyxl.chart import ScatterChart, Reference, Series
    
    print("\nAdding charts to risk rank tabs...")
    
    wb = load_workbook(output_path)
    
    asset_names = inputs['asset_names']
    n_assets = len(asset_names)
    
    W_samples = resampled_results['W_samples']
    n_resamples, n_targets, _ = W_samples.shape
    
    n_tabs = min(n_targets, max_tabs)
    
    for rank in range(n_tabs):
        sheet_name = f'Rank{rank+1:02d}'
        
        if sheet_name not in wb.sheetnames:
            continue
        
        ws = wb[sheet_name]
        
        # Find where the portfolio data starts
        # Stats at top, then blank row, then portfolios
        stats_rows = 2 + n_assets  # Return, Volatility, + each asset weight
        data_start_row = stats_rows + 4  # Skip header

        # Calculate where special portfolios are located
        # These match the positions used in write_risk_rank_detail_tabs()
        # Note: pandas startrow is 0-indexed, Excel rows are 1-indexed
        resampled_row_position = stats_rows + 2  # pandas startrow (0-indexed)
        classical_row_position = stats_rows + 4  # pandas startrow (0-indexed)

        # Convert to Excel 1-indexed data row (+1 for indexing, +1 for header)
        resampled_data_row = resampled_row_position + 2  # = stats_rows + 4
        classical_data_row = classical_row_position + 2  # = stats_rows + 6

        # Count how many portfolios
        n_portfolios = 0
        for row in range(data_start_row + 1, data_start_row + n_resamples + 2):
            if ws.cell(row, 1).value is not None:
                n_portfolios += 1
            else:
                break

        if n_portfolios == 0:
            continue

        # Create scatter chart
        chart = ScatterChart()
        chart.title = f"Risk Rank {rank+1} - Portfolio Distribution"
        chart.style = 2
        chart.x_axis.title = 'Volatility'
        chart.y_axis.title = 'Geometric Return'
        chart.height = 8
        chart.width = 12

        # Series 1: Bootstrap portfolios (column C = Volatility, B = Return)
        xvalues = Reference(ws, min_col=3, min_row=data_start_row+1,
                           max_row=data_start_row+n_portfolios)
        yvalues = Reference(ws, min_col=2, min_row=data_start_row+1,
                           max_row=data_start_row+n_portfolios)
        series = Series(yvalues, xvalues, title="Bootstrap Portfolios")
        chart.series.append(series)

        # Series 2: Resampled portfolio (to the right of main data)
        # Column calculation: Bootstrap(1) + Return(1) + Volatility(1) + n_assets weights + 2 gap
        # pandas writes at startcol = 3 + n_assets + 2 (0-indexed)
        # Excel column (1-indexed) = pandas startcol + 1
        special_col_pandas = 3 + n_assets + 2  # 0-indexed column for pandas
        special_col = special_col_pandas + 1   # Convert to 1-indexed for Excel

        if ws.cell(resampled_data_row, special_col).value is not None:  # Check if resampled data exists
            xvalues = Reference(ws, min_col=special_col+2,
                               min_row=resampled_data_row, max_row=resampled_data_row)
            yvalues = Reference(ws, min_col=special_col+1,
                               min_row=resampled_data_row, max_row=resampled_data_row)
            series = Series(yvalues, xvalues, title="Resampled (Avg)")
            chart.series.append(series)

        # Series 3: Classical portfolio
        if ws.cell(classical_data_row, special_col).value is not None:  # Check if classical data exists
            xvalues = Reference(ws, min_col=special_col+2,
                               min_row=classical_data_row, max_row=classical_data_row)
            yvalues = Reference(ws, min_col=special_col+1,
                               min_row=classical_data_row, max_row=classical_data_row)
            series = Series(yvalues, xvalues, title="Classical")
            chart.series.append(series)
        
        # Position chart to the right of all data
        # Ensure it's beyond special columns (which end at special_col + 2 for Volatility)
        chart_col = special_col + 5  # 5 columns after special data (1-indexed)
        ws.add_chart(chart, f'{chr(64 + chart_col)}2')
    
    wb.save(output_path)
    print(f"  ✓ Charts added to {n_tabs} tabs")


# =============================================================================
# Excel Writer (Updated)
# =============================================================================

def write_results_to_excel(analysis: ResampledFrontierAnalysis, 
                          output_path: str,
                          create_charts: bool = True,
                          include_rank_details: bool = True,
                          max_rank_tabs: int = 40):
    """
    Write analysis results to Excel file with embedded charts.
    
    Parameters
    ----------
    analysis : ResampledFrontierAnalysis
        Completed analysis
    output_path : str
        Output file path
    create_charts : bool
        Create embedded charts in summary sheets
    include_rank_details : bool
        Create detailed tabs for each risk rank (NEW)
    max_rank_tabs : int
        Maximum number of risk rank detail tabs (default: 40)
    """
    from openpyxl import load_workbook
    
    inputs = analysis.inputs
    classic = analysis.classic_results
    resampled = analysis.resampled_results
    settings = inputs['settings']
    asset_names = inputs['asset_names']
    
    print(f"\nWriting results to {output_path}...")
    
    # Write data with pandas
    writer = pd.ExcelWriter(output_path, engine='openpyxl')
    
    # Sheet 1: Summary
    summary = analysis.get_summary_dict()
    summary_df = pd.DataFrame([
        {'Metric': k, 'Value': v}
        for k, v in summary.items()
        if not isinstance(v, (dict, list))
    ])
    summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    # Sheet 2: Input Assets
    inputs['assets_df'].to_excel(writer, sheet_name='InputAssets', index=False)
    
    # Sheet 3: Classical Frontier
    # Calculate ENEB for each portfolio
    eneb_factors = inputs['eneb_factors']
    classic_eneb = np.array([
        np.dot(classic['weights'][i], eneb_factors) if classic['feasible'][i] else np.nan
        for i in range(len(classic['targets']))
    ])

    classic_df = pd.DataFrame({
        'Target': classic['targets'],
        'Return': classic['returns'],
        'Volatility': classic['volatilities'],
        'ENEB': classic_eneb,
        'Sharpe': classic['sharpe_ratios'],
        'Feasible': classic['feasible']
    })
    for i, name in enumerate(asset_names):
        classic_df[f'Weight_{name}'] = classic['weights'][:, i]
    classic_df.to_excel(writer, sheet_name='ClassicFrontier', index=False, startrow=0, startcol=0)
    
    # Sheet 4: Resampled Frontier
    # Calculate ENEB for each resampled portfolio
    resampled_eneb = np.array([
        np.dot(resampled['W_bar'][i], eneb_factors) if resampled['feasible'][i] else np.nan
        for i in range(len(classic['targets']))
    ])

    resampled_df = pd.DataFrame({
        'Target': classic['targets'],
        'Return': resampled['returns'],
        'Volatility': resampled['volatilities'],
        'ENEB': resampled_eneb,
        'Sharpe': resampled['sharpe_ratios'],
        'Coverage': resampled['coverage'],
        'NumValid': resampled['n_valid'],
        'Feasible': resampled['feasible']
    })
    for i, name in enumerate(asset_names):
        resampled_df[f'Weight_{name}'] = resampled['W_bar'][:, i]
    resampled_df.to_excel(writer, sheet_name='ResampledFrontier', index=False, startrow=0, startcol=0)
    
    # Sheet 5: Weight Comparison
    comparison_data = []
    for k in range(len(classic['targets'])):
        if classic['feasible'][k] and resampled['feasible'][k]:
            row = {'Target': classic['targets'][k]}
            for i, name in enumerate(asset_names):
                row[f'Classic_{name}'] = classic['weights'][k, i]
                row[f'Resampled_{name}'] = resampled['W_bar'][k, i]
                row[f'Diff_{name}'] = resampled['W_bar'][k, i] - classic['weights'][k, i]
            comparison_data.append(row)
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_excel(writer, sheet_name='WeightComparison', index=False)
    
    # Sheet 6: Correlation Matrix
    corr_df = pd.DataFrame(
        inputs['correlation'],
        index=asset_names,
        columns=asset_names
    )
    corr_df.to_excel(writer, sheet_name='Correlation')
    
    # Sheet 7: Covariance Matrix
    cov_df = pd.DataFrame(
        inputs['covariance'],
        index=asset_names,
        columns=asset_names
    )
    cov_df.to_excel(writer, sheet_name='Covariance')
    
    # Sheet 8: NEW - Rank vs Bucket Comparison
    if 'W_bar_bucket' in resampled and resampled['W_bar_bucket'] is not None:
        rank_bucket_data = []
        for k in range(len(classic['targets'])):
            if resampled['feasible'][k]:
                row = {
                    'Rank': k + 1,
                    'RankBased_Return': resampled['returns'][k],
                    'RankBased_Volatility': resampled['volatilities'][k],
                    'RankBased_Sharpe': resampled['sharpe_ratios'][k],
                    'BucketBased_Return': resampled.get('returns_bucket', [np.nan]*len(classic['targets']))[k],
                    'BucketBased_Volatility': resampled.get('volatilities_bucket', [np.nan]*len(classic['targets']))[k],
                    'BucketBased_Sharpe': resampled.get('sharpe_ratios_bucket', [np.nan]*len(classic['targets']))[k],
                    'BucketCount': resampled.get('bucket_counts', [0]*len(classic['targets']))[k],
                    'ReturnDiff': resampled.get('returns_bucket', [np.nan]*len(classic['targets']))[k] - resampled['returns'][k],
                    'VolatilityDiff': resampled.get('volatilities_bucket', [np.nan]*len(classic['targets']))[k] - resampled['volatilities'][k],
                }
                # Add weight differences
                for i, name in enumerate(asset_names):
                    if not np.isnan(resampled.get('W_bar_bucket', np.full_like(resampled['W_bar'], np.nan))[k, i]):
                        row[f'WeightDiff_{name}'] = resampled['W_bar_bucket'][k, i] - resampled['W_bar'][k, i]
                    else:
                        row[f'WeightDiff_{name}'] = np.nan
                rank_bucket_data.append(row)
        
        if rank_bucket_data:
            rank_bucket_df = pd.DataFrame(rank_bucket_data)
            rank_bucket_df.to_excel(writer, sheet_name='RankVsBucket', index=False)
    
    # Sheet 9: NEW - Bootstrap Frontiers (for plotting 200 lines)
    # Store returns and volatilities for each bootstrap's frontier
    if 'returns_samples' in resampled and 'vols_samples' in resampled:
        bootstrap_frontiers_data = []
        returns_samples = resampled['returns_samples']
        vols_samples = resampled['vols_samples']
        
        for b in range(returns_samples.shape[0]):  # Each bootstrap
            for k in range(returns_samples.shape[1]):  # Each rank
                if not np.isnan(returns_samples[b, k]) and not np.isnan(vols_samples[b, k]):
                    bootstrap_frontiers_data.append({
                        'Bootstrap': b + 1,
                        'Rank': k + 1,
                        'Return': returns_samples[b, k],
                        'Volatility': vols_samples[b, k]
                    })
        
        if bootstrap_frontiers_data:
            bootstrap_frontiers_df = pd.DataFrame(bootstrap_frontiers_data)
            bootstrap_frontiers_df.to_excel(writer, sheet_name='BootstrapFrontiers', index=False)
    
    # Sheets 10+: Detailed risk rank tabs
    if include_rank_details:
        write_detailed_risk_rank_tabs(
            writer, resampled, classic, inputs, max_rank_tabs
        )
        print(f"  ✓ {max_rank_tabs} detailed portfolio tabs created")
    
    writer.close()
    print(f"  ✓ Excel data written")

    # Apply formatting using openpyxl
    print("\nFormatting Excel output (percentages with 1 decimal place)...")
    wb = load_workbook(output_path)
    from openpyxl.utils import get_column_letter
    
    # Format ClassicFrontier sheet
    if 'ClassicFrontier' in wb.sheetnames:
        ws = wb['ClassicFrontier']
        n_assets = len(asset_names)

        # Format percentages (1 decimal place): Return, Volatility, ENEB, Weights
        # Column layout: A=Target, B=Return, C=Volatility, D=ENEB, E=Sharpe, F=Feasible, G...=Weights
        for row in range(2, ws.max_row + 1):
            # Return (column B)
            ws[f'B{row}'].number_format = '0.0%'
            # Volatility (column C)
            ws[f'C{row}'].number_format = '0.0%'
            # ENEB (column D)
            ws[f'D{row}'].number_format = '0.0%'
            # Weights start at column G (7)
            for col_idx in range(7, 7 + n_assets):
                col_letter = get_column_letter(col_idx)
                ws[f'{col_letter}{row}'].number_format = '0.0%'

        print("  ✓ ClassicFrontier formatted (all values 1 decimal %)")
    
    # Format ResampledFrontier sheet
    if 'ResampledFrontier' in wb.sheetnames:
        ws = wb['ResampledFrontier']
        n_assets = len(asset_names)

        # Format percentages (1 decimal place): Return, Volatility, ENEB, Weights
        # Column layout: A=Target, B=Return, C=Volatility, D=ENEB, E=Sharpe, F=Coverage, G=NumValid, H=Feasible, I...=Weights
        for row in range(2, ws.max_row + 1):
            # Return (column B)
            ws[f'B{row}'].number_format = '0.0%'
            # Volatility (column C)
            ws[f'C{row}'].number_format = '0.0%'
            # ENEB (column D)
            ws[f'D{row}'].number_format = '0.0%'
            # Weights start at column I (9)
            for col_idx in range(9, 9 + n_assets):
                col_letter = get_column_letter(col_idx)
                ws[f'{col_letter}{row}'].number_format = '0.0%'

        print("  ✓ ResampledFrontier formatted (all values 1 decimal %)")

    # Format WeightComparison sheet
    if 'WeightComparison' in wb.sheetnames:
        ws = wb['WeightComparison']
        n_assets = len(asset_names)

        # Format all return, volatility, and weight columns as percentages
        for row in range(2, ws.max_row + 1):
            # Columns depend on structure, but typically:
            # Return_Classic, Vol_Classic, Weights_Classic, Return_Resampled, Vol_Resampled, Weights_Resampled
            for col in range(1, ws.max_column + 1):
                cell_value = ws.cell(1, col).value
                if cell_value and (
                    'Return' in str(cell_value) or
                    'Vol' in str(cell_value) or
                    'Weight' in str(cell_value) or
                    'Sharpe' in str(cell_value)
                ):
                    ws.cell(row, col).number_format = '0.0%'

        print("  ✓ WeightComparison formatted (all values 1 decimal %)")

    # Format RankVsBucket sheet
    if 'RankVsBucket' in wb.sheetnames:
        ws = wb['RankVsBucket']

        for row in range(2, ws.max_row + 1):
            for col in range(2, ws.max_column + 1):  # Skip Rank column
                ws.cell(row, col).number_format = '0.0%'

        print("  ✓ RankVsBucket formatted (all values 1 decimal %)")

    # Format BootstrapFrontiers sheet
    if 'BootstrapFrontiers' in wb.sheetnames:
        ws = wb['BootstrapFrontiers']

        for row in range(2, ws.max_row + 1):
            # Return column (usually C)
            ws[f'C{row}'].number_format = '0.0%'
            # Volatility column (usually D)
            ws[f'D{row}'].number_format = '0.0%'

        print("  ✓ BootstrapFrontiers formatted (all values 1 decimal %)")

    # Format detailed rank tabs (Rank01-Rank40)
    rank_sheets = [name for name in wb.sheetnames if name.startswith('Rank')]
    if rank_sheets:
        n_assets = len(asset_names)

        for sheet_name in rank_sheets:
            ws = wb[sheet_name]

            # Format all numeric cells as percentages with 1 decimal
            for row in range(1, ws.max_row + 1):
                for col in range(1, ws.max_column + 1):
                    cell = ws.cell(row, col)
                    header_cell = ws.cell(1, col)

                    # Skip header row and Bootstrap column
                    if row == 1 or header_cell.value == 'Bootstrap':
                        continue

                    # Check if header indicates this should be a percentage
                    if header_cell.value and (
                        'Return' in str(header_cell.value) or
                        'Volatility' in str(header_cell.value) or
                        'ENEB' in str(header_cell.value) or
                        'Weight' in str(header_cell.value) or
                        'Mean' == str(header_cell.value) or
                        'Median' == str(header_cell.value) or
                        'StdDev' == str(header_cell.value) or
                        'Min' == str(header_cell.value) or
                        'Max' == str(header_cell.value) or
                        'Resampled' == str(header_cell.value) or
                        'Classical' == str(header_cell.value)
                    ):
                        # Check if cell contains a number
                        if isinstance(cell.value, (int, float)):
                            cell.number_format = '0.0%'

        print(f"  ✓ {len(rank_sheets)} rank tabs formatted (all values 1 decimal %)")

    wb.save(output_path)
    print("  ✓ Formatting applied and saved")
    
    # Reload for chart creation
    if create_charts:
        print("\nCreating embedded Excel charts...")
        
        wb = load_workbook(output_path)
        
        classic_chart_df, resampled_chart_df, bootstrap_df = create_frontier_chart_data(
            classic, resampled,
            inputs['geometric_returns'],
            inputs['covariance']
        )
        
        add_frontier_chart_to_sheet(wb, 'ClassicFrontier', 
                                    classic_chart_df, resampled_chart_df, bootstrap_df)
        print("  ✓ Frontier chart added to ClassicFrontier sheet")
        
        add_allocation_chart_to_sheet(wb, 'ClassicFrontier',
                                      classic['weights'],
                                      classic['volatilities'],
                                      classic['feasible'],
                                      asset_names,
                                      "Classical Frontier - Asset Allocations",
                                      max_points=settings.frontier_points)
        print("  ✓ Classical allocation chart added")

        add_allocation_chart_to_sheet(wb, 'ResampledFrontier',
                                      resampled['W_bar'],
                                      resampled['volatilities'],
                                      resampled['feasible'],
                                      asset_names,
                                      "Resampled Frontier - Asset Allocations",
                                      max_points=settings.frontier_points)
        print("  ✓ Resampled allocation chart added")
        
        # NEW: Add Rank vs Bucket comparison chart
        if 'RankVsBucket' in wb.sheetnames:
            add_rank_vs_bucket_comparison_chart(wb, 'RankVsBucket')
            print("  ✓ Rank vs Bucket comparison chart added")
        
        # NEW: Add 200 Bootstrap Frontiers chart
        if 'BootstrapFrontiers' in wb.sheetnames:
            add_bootstrap_frontiers_chart(wb, 'BootstrapFrontiers', classic, resampled)
            print("  ✓ Bootstrap frontiers chart added (200 individual frontiers)")
        
        wb.save(output_path)
        
        # Add charts to risk rank tabs
        if include_rank_details:
            add_charts_to_risk_rank_tabs(
                output_path, resampled, classic, inputs, max_rank_tabs
            )
    
    print(f"\n✓ All outputs saved to {output_path}")
    
    # Print sheet summary
    wb = load_workbook(output_path)
    print(f"\n📊 Output Excel file created with {len(wb.sheetnames)} worksheets")
    print("="*60)
    print("\nMain Analysis Sheets:")
    print("  • Summary - Overall analysis results and settings")
    print("  • InputAssets - Your asset class inputs")
    print("  • ClassicFrontier - Traditional Markowitz efficient frontier")
    print("  • ResampledFrontier - Michaud resampled frontier (recommended)")
    print("  • WeightComparison - Side-by-side comparison of portfolios")
    print("  • Correlation, Covariance - Input correlation/covariance matrices")
    print("  • RankVsBucket - Comparison of averaging methods")
    print("  • BootstrapFrontiers - All bootstrap scenario data")
    if include_rank_details:
        print(f"\nDetailed Portfolio Tabs (Rank01-Rank{max_rank_tabs:02d}):")
        print(f"  Each tab shows one risk level with:")
        print(f"    • Summary statistics across all {resampled['W_samples'].shape[0]} bootstrap scenarios")
        print(f"    • Complete data for all bootstrap portfolios")
        print(f"    • Resampled (averaged) portfolio - RECOMMENDED for actual investing")
        print(f"    • Classical (single-scenario) portfolio - for comparison only")
        if create_charts:
            print(f"    • Risk/return scatter chart showing all scenarios")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Resampled Efficient Frontier Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ref_excel_v2.py sample_scenario.xlsx
  python ref_excel_v2.py my_portfolio.xlsx -o results.xlsx
  python ref_excel_v2.py scenario.xlsx --no-charts
  python ref_excel_v2.py scenario.xlsx --max-rank-tabs 20  # Only first 20 ranks
        """
    )
    
    parser.add_argument('input', help='Input Excel file path')
    parser.add_argument('-o', '--output', help='Output Excel file path (default: input_output.xlsx)')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress messages')
    parser.add_argument('--no-charts', action='store_true', help='Skip creating embedded charts in Excel')
    parser.add_argument('--no-rank-details', action='store_true', 
                       help='Skip creating detailed risk rank tabs (saves time for large analyses)')
    parser.add_argument('--max-rank-tabs', type=int, default=40,
                       help='Maximum number of risk rank detail tabs to create (default: 40)')
    
    args = parser.parse_args()
    
    # Default output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = input_path.stem + '_output.xlsx'
    
    try:
        # Run analysis
        analysis = ResampledFrontierAnalysis(args.input, verbose=not args.quiet)
        analysis.run()
        
        # Write results
        write_results_to_excel(
            analysis, 
            args.output, 
            create_charts=not args.no_charts,
            include_rank_details=not args.no_rank_details,
            max_rank_tabs=args.max_rank_tabs
        )
        
        # Print summary
        if not args.quiet:
            print("\n" + "="*60)
            print("ANALYSIS SUMMARY")
            print("="*60)
            summary = analysis.get_summary_dict()

            print("\nPortfolio Setup:")
            print(f"  • Number of asset classes: {summary['n_assets']}")
            print(f"  • Frontier portfolios calculated: {summary['n_frontier_points']}")
            print(f"  • Bootstrap scenarios tested: {summary['n_resamples']}")

            print("\nResampling Configuration:")
            method_names = {
                'forecast_perturbation': 'Forward forecast perturbation',
                'bayesian': 'Bayesian posterior sampling',
                'historical': 'Historical data bootstrap',
                'parametric': 'Parametric time-series'
            }
            method_display = method_names.get(summary['resampling_method'], summary['resampling_method'])
            print(f"  • Method: {method_display}")
            if summary['uses_illiquidity_penalty']:
                print(f"  • Illiquidity penalty: λ={summary['illiquidity_penalty_lambda']:.1f}")

            print("\nRobustness Metrics:")
            print(f"  • Bootstrap success rate: {summary['avg_coverage']:.1%}")
            print(f"    (On average, {summary['avg_coverage']:.1%} of {summary['n_resamples']} scenarios")
            print(f"     successfully constructed feasible portfolios at each risk level)")

            if summary['tangency_sharpe']:
                print("\nOptimal Portfolio (Maximum Sharpe Ratio):")
                print(f"  • Expected return: {summary['tangency_return']:.2%} per year")
                print(f"  • Volatility (risk): {summary['tangency_volatility']:.2%}")
                print(f"  • Sharpe ratio: {summary['tangency_sharpe']:.3f}")

            print(f"\nTotal analysis time: {summary['runtime_seconds']['total']:.1f}s")
            print("="*60)
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
