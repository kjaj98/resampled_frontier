import numpy as np

from refmvo.core.frontier import compute_targets
from refmvo.core.resampling import resample_frontier_with_coverage


def test_resampling_modes_differ(case_4_assets_moderate_corr, require_solver):
    mu, cov, bounds = case_4_assets_moderate_corr
    rng = np.random.default_rng(7)
    X_hist = rng.multivariate_normal(mu, cov, size=120)

    targets = compute_targets(mu, bounds, points=8)["targets"]

    res_param = resample_frontier_with_coverage(
        mu,
        cov,
        bounds,
        targets,
        resamples=15,
        resample_years=3,
        frequency="Annual",
        seed=1,
        min_valid_bootstraps=1,
        mode="Parametric",
    )
    res_nonparam = resample_frontier_with_coverage(
        mu,
        cov,
        bounds,
        targets,
        resamples=15,
        resample_years=3,
        frequency="Annual",
        seed=1,
        min_valid_bootstraps=1,
        mode="Nonparametric",
        returns=X_hist,
    )

    mask_param = np.isfinite(res_param["rets"])
    mask_nonparam = np.isfinite(res_nonparam["rets"])
    assert mask_param.sum() >= 4
    assert mask_nonparam.sum() >= 4
    mask = mask_param & mask_nonparam
    assert mask.any()
    assert not np.allclose(res_param["rets"][mask], res_nonparam["rets"][mask])
