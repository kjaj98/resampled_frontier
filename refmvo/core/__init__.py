"""Core logic for the REF MVO project."""

from refmvo.core.constraints import ConstraintsSpec, constraints_summary, normalize_bounds
from refmvo.core.diagnostics import build_diagnostics_df, simulate_paths
from refmvo.core.frontier import compute_target_bands, compute_targets, efficient_frontier, solve_target_qp
from refmvo.core.io import MonteCarloParams, Settings, ensure_scaffold, read_scenario_xlsx
from refmvo.core.moments import (
    build_covariance,
    geometric_to_arithmetic,
    nearest_psd,
    portfolio_mapping_to_geometric,
    shrink_cov_from_samples,
)
from refmvo.core.resampling import resample_frontier_with_coverage

__all__ = [
    "ConstraintsSpec",
    "constraints_summary",
    "normalize_bounds",
    "build_diagnostics_df",
    "simulate_paths",
    "compute_targets",
    "compute_target_bands",
    "efficient_frontier",
    "solve_target_qp",
    "MonteCarloParams",
    "Settings",
    "ensure_scaffold",
    "read_scenario_xlsx",
    "build_covariance",
    "geometric_to_arithmetic",
    "nearest_psd",
    "portfolio_mapping_to_geometric",
    "shrink_cov_from_samples",
    "resample_frontier_with_coverage",
]
