import numpy as np

from refmvo.core.frontier import compute_targets, efficient_frontier


def test_frontier_endpoints_behave(toy_case_2_assets, require_solver):
    mu, cov, bounds = toy_case_2_assets
    targets = compute_targets(mu, bounds, points=5)["targets"]
    ef = efficient_frontier(mu, cov, r_f=0.0, bounds=bounds, targets=targets)
    ok = ef["ok_mask"]
    assert ok[0] and ok[-1]
    w_min = ef["W"][0]
    w_max = ef["W"][-1]
    assert w_min[1] < 0.2
    assert w_max[1] > 0.8


def test_risk_non_decreasing(toy_case_2_assets, require_solver):
    mu, cov, bounds = toy_case_2_assets
    targets = compute_targets(mu, bounds, points=7)["targets"]
    ef = efficient_frontier(mu, cov, r_f=0.0, bounds=bounds, targets=targets)
    ok = ef["ok_mask"]
    risks = ef["risks"][ok]
    if len(risks) < 3:
        return
    min_idx = int(np.nanargmin(risks))
    diffs = np.diff(risks[min_idx:])
    assert np.all(diffs >= -1e-6)
