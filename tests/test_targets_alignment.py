import numpy as np

from refmvo.core.frontier import compute_targets
from refmvo.core.resampling import resample_frontier_with_coverage


def test_targets_monotonic(case_4_assets_moderate_corr, require_solver):
    mu, cov, bounds = case_4_assets_moderate_corr
    cov = cov * 4.0
    info = compute_targets(mu, bounds, points=12)
    targets = info["targets"]
    assert np.all(np.diff(targets) >= -1e-12)
    assert targets[0] >= info["ret_min"] - 1e-10
    assert targets[-1] <= info["ret_max"] + 1e-10


def test_resampling_targets_fixed_and_infeasible(case_4_assets_moderate_corr, require_solver):
    mu, cov, bounds = case_4_assets_moderate_corr
    cov = cov * 4.0
    bounds = np.minimum(bounds, 0.30)
    info = compute_targets(mu, bounds, points=10)
    targets = info["targets"]

    res = resample_frontier_with_coverage(
        mu,
        cov,
        bounds,
        targets,
        resamples=25,
        resample_years=2,
        frequency="Annual",
        seed=123,
        min_valid_bootstraps=1,
    )

    diag = res["diagnostics_df"]
    assert np.allclose(diag["target_return"].values, targets)
    assert np.any(res["n_valid"] < 25)
    mid = len(targets) // 2
    assert res["coverage"][-1] <= res["coverage"][mid] + 1e-8
