import json
import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from refmvo.core.io import MonteCarloParams
from refmvo.core.moments import annualize_moments, portfolio_mapping_to_geometric
from refmvo.plots.mc_plots import fan_chart_from_quantiles, histogram_with_kde


def build_diagnostics_df(
    targets: np.ndarray,
    coverage: np.ndarray,
    n_valid: np.ndarray,
    resampled_ret: Optional[np.ndarray] = None,
    resampled_risk: Optional[np.ndarray] = None,
    classic_ret: Optional[np.ndarray] = None,
    classic_risk: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Assemble a diagnostics table for coverage and frontier points."""
    data = {
        "target_return": np.asarray(targets, dtype=float),
        "coverage": np.asarray(coverage, dtype=float),
        "n_valid": np.asarray(n_valid, dtype=int),
    }
    if resampled_ret is not None:
        data["resampled_return"] = np.asarray(resampled_ret, dtype=float)
    if resampled_risk is not None:
        data["resampled_risk"] = np.asarray(resampled_risk, dtype=float)
    if classic_ret is not None:
        data["classic_return"] = np.asarray(classic_ret, dtype=float)
    if classic_risk is not None:
        data["classic_risk"] = np.asarray(classic_risk, dtype=float)
    return pd.DataFrame(data)


def simulate_paths(
    mu: np.ndarray,
    cov: np.ndarray,
    w: np.ndarray,
    params: MonteCarloParams,
    rule_row: pd.Series,
    inflation: float,
    downloads_dir: str,
    periods_per_year: int = 1,
) -> Dict[str, object]:
    """Monte Carlo with lognormal portfolio returns and optional AR(1)."""
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    w = np.asarray(w, dtype=float)
    periods = int(periods_per_year)
    if periods <= 0:
        raise ValueError("periods_per_year must be a positive integer.")
    if periods != 1:
        mu, cov = annualize_moments(mu, cov, periods)
    T = int(params.HorizonYears)
    P = int(params.NumPaths)
    rng = np.random.default_rng(params.Seed)

    mu_p = float(mu @ w)
    sigma_p = float(np.sqrt(w @ cov @ w))

    drift_stress = params.DriftStressBps / 10_000.0
    mgmt_drag = params.MgmtFeeBps / 10_000.0
    tax_rate = float(getattr(params, "WithdrawalTaxRate", 0.0))
    if tax_rate < 0.0 or tax_rate >= 1.0:
        raise ValueError("WithdrawalTaxRate must be in [0, 1).")

    mu_p_net = mu_p - drift_stress - mgmt_drag
    if mu_p_net <= -1.0:
        raise ValueError("Net mean return <= -100%; lognormal parameters undefined.")

    geom_p = float(portfolio_mapping_to_geometric(mu_p_net, sigma_p))
    gross_geom = 1.0 + geom_p
    if gross_geom <= 0.0:
        raise ValueError("Geometric mean <= -100%; lognormal parameters undefined.")

    log_mu = float(np.log(gross_geom))
    y = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * (sigma_p**2) * np.exp(-2.0 * log_mu)))
    log_var = float(np.log(y))
    log_sigma = float(np.sqrt(max(log_var, 0.0)))

    eps = rng.standard_normal(size=(P, T))

    if params.SerialCorrelation:
        rho = float(params.AR1_Rho)
        scale = log_sigma * np.sqrt(max(1e-12, 1.0 - rho**2))
        log_r = np.empty((P, T), dtype=float)
        log_r[:, 0] = log_mu + scale * eps[:, 0]
        for t in range(1, T):
            log_r[:, t] = log_mu + rho * (log_r[:, t - 1] - log_mu) + scale * eps[:, t]
    else:
        log_r = log_mu + log_sigma * eps

    rule = str(rule_row.get("Rule"))
    S0 = float(rule_row.get("S0") or 0.0)
    gamma = float(rule_row.get("Gamma") or 0.0)
    alpha = float(rule_row.get("Alpha") or 0.0)
    floor_pct = float(rule_row.get("FloorPct") or 0.0)
    ceil_pct = float(rule_row.get("CeilingPct") or 9e9)
    lower_wr = float(rule_row.get("LowerWR") or 0.0)
    upper_wr = float(rule_row.get("UpperWR") or 9e9)
    delta = float(rule_row.get("Delta") or 0.0)

    wealth = np.full((P, T + 1), float(params.StartWealth), dtype=float)
    spend = np.zeros((P, T), dtype=float)

    start_y = int(params.StartSpendingYear)

    for t in range(T):
        prev_spend = spend[:, t - 1] if t > 0 else np.full(P, S0, dtype=float)

        if t < start_y:
            S_nom = np.zeros(P, dtype=float)
        else:
            if rule.lower() == "constantreal":
                S_nom = np.full(P, S0 * ((1.0 + inflation) ** t), dtype=float)
            elif rule.lower() == "percentofportfolio":
                S_nom = gamma * wealth[:, t]
            elif rule.lower() == "endowmenthybrid":
                S_nom = alpha * (gamma * wealth[:, t]) + (1.0 - alpha) * prev_spend * (1.0 + inflation)
                S_nom = np.maximum(S_nom, floor_pct * prev_spend)
                S_nom = np.minimum(S_nom, ceil_pct * prev_spend)
            elif rule.lower() == "guardrails":
                S_nom = prev_spend * (1.0 + inflation)
                wr = np.divide(S_nom, np.maximum(wealth[:, t], 1e-12))
                inc_mask = wr < lower_wr
                dec_mask = wr > upper_wr
                S_nom[inc_mask] = S_nom[inc_mask] * (1.0 + delta)
                S_nom[dec_mask] = S_nom[dec_mask] * (1.0 - delta)
                S_nom = np.maximum(S_nom, floor_pct * prev_spend)
                S_nom = np.minimum(S_nom, ceil_pct * prev_spend)
            else:
                raise ValueError(f"Unknown spending rule: {rule}")

        max_net = wealth[:, t] * (1.0 - tax_rate)
        S_net = np.minimum(S_nom, max_net)
        spend[:, t] = S_net

        if tax_rate > 0.0:
            S_gross = S_net / (1.0 - tax_rate)
        else:
            S_gross = S_net
        wealth_after = wealth[:, t] - S_gross
        gross = np.exp(log_r[:, t])
        wealth[:, t + 1] = np.maximum(0.0, wealth_after * gross)

    infl_vec = (1.0 + inflation) ** np.arange(T + 1)
    wealth_real = wealth / infl_vec[None, :]

    terminal = wealth[:, -1]
    ruin_prob = float(np.mean(wealth.min(axis=1) <= 1.0))
    kpis = dict(
        ruin_probability=ruin_prob,
        median_terminal_wealth=float(np.median(terminal)),
        p5_terminal=float(np.percentile(terminal, 5)),
        p25_terminal=float(np.percentile(terminal, 25)),
        p75_terminal=float(np.percentile(terminal, 75)),
        p95_terminal=float(np.percentile(terminal, 95)),
    )

    os.makedirs(downloads_dir, exist_ok=True)

    qtiles = [5, 25, 50, 75, 95]

    def qtile_df(arr: np.ndarray, idx_prefix: str) -> pd.DataFrame:
        qs = np.percentile(arr, qtiles, axis=0)
        df = pd.DataFrame(qs.T, columns=[f"{idx_prefix}_p{q}" for q in qtiles])
        df["t"] = np.arange(df.shape[0])
        return df[["t"] + [c for c in df.columns if c != "t"]]

    wealth_nom_df = qtile_df(wealth, "wealth_nominal")
    wealth_real_df = qtile_df(wealth_real, "wealth_real")
    spend_nom_df = qtile_df(spend, "spend_nominal")

    wealth_nom_df.to_csv(os.path.join(downloads_dir, "wealth_quantiles_nominal.csv"), index=False)
    wealth_real_df.to_csv(os.path.join(downloads_dir, "wealth_quantiles_real.csv"), index=False)
    spend_nom_df.to_csv(os.path.join(downloads_dir, "spend_quantiles_nominal.csv"), index=False)
    with open(os.path.join(downloads_dir, "mc_kpis.json"), "w", encoding="utf-8") as f:
        json.dump(kpis, f, indent=2)

    try:
        fig_nom = fan_chart_from_quantiles(wealth_nom_df, title="Nominal Wealth Fan Chart")
        fig_nom.savefig(os.path.join(downloads_dir, "fan_nominal.png"), dpi=200)
        plt.close(fig_nom)

        fig_real = fan_chart_from_quantiles(wealth_real_df, title="Real Wealth Fan Chart")
        fig_real.savefig(os.path.join(downloads_dir, "fan_real.png"), dpi=200)
        plt.close(fig_real)

        hist_fig_wealth = histogram_with_kde(terminal, title="Terminal Wealth Distribution")
        hist_fig_wealth.savefig(os.path.join(downloads_dir, "terminal_wealth_hist.png"), dpi=200)
        plt.close(hist_fig_wealth)

        term_spend = spend[:, -1]
        hist_fig_spend = histogram_with_kde(term_spend, title="Terminal Spending Distribution")
        hist_fig_spend.savefig(os.path.join(downloads_dir, "terminal_spending_hist.png"), dpi=200)
        plt.close(hist_fig_spend)
    except Exception as e:
        print(f"[WARN] PNG export failed: {e}")

    return dict(
        wealth=wealth,
        wealth_real=wealth_real,
        spend=spend,
        kpis=kpis,
        wealth_nom_q=wealth_nom_df,
        wealth_real_q=wealth_real_df,
        spend_nom_q=spend_nom_df,
    )


__all__ = ["build_diagnostics_df", "simulate_paths"]
