from refmvo.core.frontier import compute_targets
from refmvo.core.resampling import resample_frontier_with_coverage


def test_diagnostics_dataframe_columns(case_4_assets_moderate_corr, require_solver):
    mu, cov, bounds = case_4_assets_moderate_corr
    targets = compute_targets(mu, bounds, points=6)["targets"]
    res = resample_frontier_with_coverage(
        mu,
        cov,
        bounds,
        targets,
        resamples=6,
        resample_years=2,
        frequency="Annual",
        seed=5,
        min_valid_bootstraps=1,
    )
    diag = res["diagnostics_df"]
    for col in ["target_return", "coverage", "n_valid"]:
        assert col in diag.columns
