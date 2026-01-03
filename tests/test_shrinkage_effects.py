import numpy as np

from refmvo.core.frontier import compute_targets
from refmvo.core.resampling import resample_frontier_with_coverage


def test_shrinkage_changes_risk(case_4_assets_moderate_corr, require_solver):
    mu, cov, bounds = case_4_assets_moderate_corr
    targets = compute_targets(mu, bounds, points=8)["targets"]

    res_plain = resample_frontier_with_coverage(
        mu,
        cov,
        bounds,
        targets,
        resamples=12,
        resample_years=2,
        frequency="Annual",
        seed=11,
        min_valid_bootstraps=1,
        shrinkage=False,
    )
    res_shrink = resample_frontier_with_coverage(
        mu,
        cov,
        bounds,
        targets,
        resamples=12,
        resample_years=2,
        frequency="Annual",
        seed=11,
        min_valid_bootstraps=1,
        shrinkage=True,
    )

    mask = np.isfinite(res_plain["risks"]) & np.isfinite(res_shrink["risks"])
    assert mask.any()
    assert not np.allclose(res_plain["risks"][mask], res_shrink["risks"][mask])
