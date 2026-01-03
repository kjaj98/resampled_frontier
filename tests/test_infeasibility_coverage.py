import numpy as np

from refmvo.core.frontier import compute_targets
from refmvo.core.resampling import resample_frontier_with_coverage


def test_infeasibility_coverage(case_4_assets_moderate_corr, require_solver):
    mu, cov, bounds = case_4_assets_moderate_corr
    cov = cov * 4.0
    bounds = np.minimum(bounds, 0.30)
    targets = compute_targets(mu, bounds, points=12)["targets"]
    res = resample_frontier_with_coverage(
        mu,
        cov,
        bounds,
        targets,
        resamples=20,
        resample_years=2,
        frequency="Annual",
        seed=321,
        min_valid_bootstraps=15,
    )

    assert np.all(res["coverage"] >= 0.0)
    assert np.all(res["coverage"] <= 1.0)
    mid = len(targets) // 2
    assert res["coverage"][-1] <= res["coverage"][mid] + 1e-8
    assert np.any(res["n_valid"] != res["n_valid"][0])

    invalid = np.where(res["n_valid"] < 15)[0]
    assert len(invalid) > 0
    for idx in invalid:
        assert np.all(~np.isfinite(res["W_bar"][idx]))
