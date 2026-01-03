# Resampled Efficient Frontier + Spending Rule Simulator
# -----------------------------------------------------
# Launch with:
#   streamlit run app.py
#
# This app:
#  - Loads scenario inputs from Excel (data/inputs/*.xlsx)
#  - Converts geometric to arithmetic returns for optimization
#  - Builds the classical efficient frontier with CVXPY
#  - Implements Michaud (1998) Resampled Efficient Frontier (REF)
#  - Runs a Monte Carlo spending simulator with 4 rules
#  - Exports CSV/PNG results to data/outputs/

import os
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.linalg import eigh

from refmvo.core.constraints import normalize_bounds
from refmvo.core.diagnostics import simulate_paths
from refmvo.core.frontier import compute_targets, compute_target_bands, efficient_frontier
from refmvo.core.io import Settings, MonteCarloParams, ensure_scaffold, read_scenario_xlsx
from refmvo.core.moments import geometric_to_arithmetic, portfolio_mapping_to_geometric
from refmvo.core.resampling import resample_frontier_with_coverage
from refmvo.plots.frontier_plots import (
    plot_coverage,
    plot_frontier_with_coverage,
    plot_valid_counts,
    plot_weights_by_risk_stack,
)
from refmvo.plots.mc_plots import fan_chart_from_quantiles
from refmvo.utils.constants import PERIODS_PER_YEAR

# Streamlit caching compatibility (older Streamlit lacks cache_data)
cache_data = st.cache_data if hasattr(st, "cache_data") else st.cache

TARGET_EPS = 1e-6

# Pydantic v1/v2 compatibility helpers
def _model_dump(model):
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()

def _model_copy(model, **kwargs):
    return model.model_copy(**kwargs) if hasattr(model, "model_copy") else model.copy(**kwargs)

# ------------------------
# Bootstrap project folders
# ------------------------

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_DIR = os.path.join(DATA_DIR, "inputs")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
STREAMLIT_DIR = os.path.join(PROJECT_ROOT, ".streamlit")

def _ensure_scaffold():
    sample_xlsx = os.path.join(INPUT_DIR, "sample_scenario.xlsx")
    ensure_scaffold(INPUT_DIR, OUTPUT_DIR, STREAMLIT_DIR, sample_xlsx)

    # Print a concise run guide to server console (first execution)
    if not os.environ.get("REF_APP_BOOTSTRAP_PRINTED"):
        print("\n=== Streamlit REF Simulator ===")
        print("Install:    pip install -r requirements.txt")
        print("Run:        streamlit run app.py  (opens http://localhost:8501)")
        print("Inputs:     data/inputs/sample_scenario.xlsx")
        print("Outputs:    data/outputs/")
        print("Config:     .streamlit/config.toml")
        print("===============================\n")
        os.environ["REF_APP_BOOTSTRAP_PRINTED"] = "1"

# ------------------------
# Streamlit UI
# ------------------------

@cache_data(show_spinner=False)
def _cache_read(path: str) -> Dict[str, Any]:
    return read_scenario_xlsx(path)

@cache_data(show_spinner=True)
def _cache_targets(mu, bounds, points):
    return compute_targets(np.asarray(mu, dtype=float), np.asarray(bounds, dtype=float), points)

@cache_data(show_spinner=True)
def _cache_frontier(mu, S, r_f, bounds, targets, eps, target_bands):
    return efficient_frontier(
        np.asarray(mu, dtype=float),
        np.asarray(S, dtype=float),
        float(r_f),
        np.asarray(bounds, dtype=float),
        np.asarray(targets, dtype=float),
        eps=float(eps),
        target_bands=None if target_bands is None else np.asarray(target_bands, dtype=float),
    )

@cache_data(show_spinner=True)
def _cache_resampled(mu, S, bounds, targets, resamples, resample_years, frequency, seed, eps,
                     target_bands, min_valid_bootstraps, returns, mode, shrinkage):
    return resample_frontier_with_coverage(
        np.asarray(mu, dtype=float),
        np.asarray(S, dtype=float),
        np.asarray(bounds, dtype=float),
        np.asarray(targets, dtype=float),
        int(resamples),
        float(resample_years),
        str(frequency),
        int(seed),
        eps=float(eps),
        target_bands=None if target_bands is None else np.asarray(target_bands, dtype=float),
        min_valid_bootstraps=int(min_valid_bootstraps),
        returns=None if returns is None else np.asarray(returns, dtype=float),
        mode=str(mode),
        shrinkage=bool(shrinkage),
    )

@cache_data(show_spinner=True)
def _cache_mc(mu, S, w, mc_params_dict, rule_row_dict, inflation, downloads_dir, periods_per_year):
    mc_params = MonteCarloParams(**mc_params_dict)
    rule_row = pd.Series(rule_row_dict)
    return simulate_paths(
        np.asarray(mu, dtype=float),
        np.asarray(S, dtype=float),
        np.asarray(w, dtype=float),
        mc_params,
        rule_row,
        float(inflation),
        str(downloads_dir),
        int(periods_per_year),
    )

def main():
    _ensure_scaffold()

    st.set_page_config(page_title="Resampled Efficient Frontier & Spending Simulator",
                       layout="wide")

    st.title("Resampled Efficient Frontier (REF) + Monte Carlo Spending Simulator")

    # Sidebar: scenario selection
    st.sidebar.header("Scenario & Settings")
    default_xlsx = os.path.join(INPUT_DIR, "sample_scenario.xlsx")
    xlsx_path = st.sidebar.text_input("Excel scenario path", value=default_xlsx)
    if not os.path.exists(xlsx_path):
        st.sidebar.error("Excel file not found. The app will create a sample workbook on run.", icon="🚩")

    # Load data
    data = _cache_read(xlsx_path)
    assets = data["assets"]
    names = data["names"]
    S = data["cov"]
    mu = data["mu_arith"]
    bounds = data["bounds"]
    settings: Settings = data["settings"]
    mc: MonteCarloParams = data["mc"]
    rules_df = data["rules"]
    returns_df = data.get("returns")
    returns_info = data.get("returns_info", {})

    # Controls
    st.sidebar.subheader("Optimization")
    long_only = st.sidebar.checkbox("Long-only", value=settings.LongOnly)
    r_f = st.sidebar.number_input("Risk-free rate", value=float(settings.RiskFreeRate), step=0.001)
    points = st.sidebar.slider("Frontier points", min_value=10, max_value=200, value=int(settings.FrontierPoints), step=10)
    resamples = st.sidebar.slider("Resamples", min_value=20, max_value=500, value=int(settings.Resamples), step=20)
    resample_years = st.sidebar.slider("Resample years", min_value=5, max_value=60, value=int(settings.ResampleYears), step=5)
    rng_seed = st.sidebar.number_input("Random seed", value=int(settings.RandomSeed), step=1)
    yaxis_mode = st.sidebar.selectbox("Frontier Y-axis", options=["Compound","Arithmetic"], index=0 if settings.FrontierYAxis=="Compound" else 1)

    st.sidebar.subheader("Resampling & Coverage")
    coverage_threshold = st.sidebar.slider("Coverage threshold", min_value=0.5, max_value=1.0, value=0.8, step=0.05)
    min_valid_bootstraps = st.sidebar.number_input(
        "Min valid bootstraps",
        min_value=1,
        max_value=int(resamples),
        value=min(10, int(resamples)),
        step=1,
    )
    use_resample_shrinkage = st.sidebar.checkbox("Shrinkage inside resamples", value=False)

    returns_ok = bool(returns_info.get("available")) and returns_df is not None and len(returns_df) >= 2
    if returns_ok:
        resample_mode_label = st.sidebar.selectbox(
            "Resampling mode",
            options=["Parametric (MVN)", "Nonparametric (Bootstrap)"],
            index=0,
        )
    else:
        resample_mode_label = "Parametric (MVN)"
        st.sidebar.caption("Returns sheet not available or too short; using parametric resampling.")

    periods = PERIODS_PER_YEAR.get((settings.Frequency or "").strip().lower(), 1)
    resample_periods = int(round(resample_years * periods))
    st.sidebar.caption(f"Resample periods: {resample_periods} ({settings.Frequency})")

    # Effective bounds with global caps and long-only
    try:
        bounds_eff = normalize_bounds(bounds, long_only, settings.GlobalMinWeight, settings.GlobalMaxWeight)
    except ValueError as exc:
        st.error(f"Invalid bounds after applying caps: {exc}", icon="🚩")
        st.stop()

    # Global target grid (fixed for all resamples)
    target_info = _cache_targets(tuple(mu), bounds_eff, points)
    targets = target_info["targets"]
    target_bands = compute_target_bands(targets, min_band=TARGET_EPS)
    points_eff = int(len(targets))
    if target_info["degenerate"]:
        st.warning("Constraints imply a near-degenerate frontier (min ~ max return). Using a single target.", icon="⚠️")

    # Frontier computations
    ef = _cache_frontier(tuple(mu), S, r_f, bounds_eff, targets, TARGET_EPS, tuple(target_bands))
    resample_mode = "Nonparametric" if resample_mode_label.startswith("Nonparametric") else "Parametric"
    ref = _cache_resampled(
        tuple(mu),
        S,
        bounds_eff,
        targets,
        resamples,
        resample_years,
        settings.Frequency,
        rng_seed,
        TARGET_EPS,
        tuple(target_bands),
        min_valid_bootstraps,
        None if not returns_ok else returns_df.values,
        resample_mode,
        use_resample_shrinkage,
    )

    # Layout with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Data", "Frontier", "Feasibility / Coverage", "Monte Carlo", "Validation"]
    )

    with tab1:
        st.subheader("Assets")
        st.dataframe(assets)
        st.subheader("Correlation")
        st.dataframe(data["corr"])
        st.subheader("Settings")
        st.json(_model_dump(settings))
        st.subheader("Monte Carlo")
        st.json(_model_dump(mc))
        st.subheader("Spending Rules")
        st.dataframe(rules_df)
        st.subheader("Returns (optional)")
        if returns_info.get("available"):
            st.caption(
                f"Returns rows: {returns_info.get('rows', 0)} "
                f"(dropped {returns_info.get('dropped_rows', 0)} rows with NaNs)."
            )
            st.dataframe(returns_df.head(10))
        else:
            st.caption(returns_info.get("reason", "No returns data provided."))

    with tab2:
        st.subheader("Efficient Frontier (classical vs. Michaud REF)")
        # Classical frontier (arithmetic mean)
        ef_mu = ef["rets"]
        ef_sig = ef["risks"]
        ef_geom = np.full_like(ef_mu, np.nan, dtype=float)
        ef_mask = np.isfinite(ef_mu) & np.isfinite(ef_sig)
        ef_geom[ef_mask] = [portfolio_mapping_to_geometric(ef_mu[i], ef_sig[i]) for i in np.where(ef_mask)[0]]

        # Resampled (coverage-aware)
        ref_mu = ref["rets"]
        ref_sig = ref["risks"]
        ref_geom = np.full_like(ref_mu, np.nan, dtype=float)
        ref_mask = np.isfinite(ref_mu) & np.isfinite(ref_sig)
        ref_geom[ref_mask] = [portfolio_mapping_to_geometric(ref_mu[i], ref_sig[i]) for i in np.where(ref_mask)[0]]

        if yaxis_mode == "Compound":
            y1, y2 = ef_geom, ref_geom
            ylab = "Expected compound (geometric) return"
        else:
            y1, y2 = ef_mu, ref_mu
            ylab = "Expected arithmetic return"

        fig = plot_frontier_with_coverage(
            classic_sig=ef_sig,
            classic_ret=y1,
            classic_ok=ef.get("ok_mask"),
            ref_sig=ref_sig,
            ref_ret=y2,
            coverage=ref["coverage"],
            threshold=coverage_threshold,
            yaxis_title=ylab,
            title="Efficient Frontier (classic vs. resampled)",
        )
        st.pyplot(fig)


        # Show weights at selected target
        idx = st.slider("Select frontier point (index)", 0, points_eff - 1, int(points_eff // 2))
        target_ret = float(targets[idx])
        coverage_k = float(ref["coverage"][idx])
        n_valid_k = int(ref["n_valid"][idx])
        st.caption(
            f"Target return: {target_ret:.4f} | Coverage: {coverage_k:.0%} ({n_valid_k}/{resamples})"
        )

        w_classic = ef["W"][idx] if ef.get("ok_mask", np.ones(points_eff, dtype=bool))[idx] else np.full(len(names), np.nan)
        w_ref = ref["W_bar"][idx]
        if not np.all(np.isfinite(w_ref)):
            st.warning("Selected resampled point is invalid or below minimum coverage.", icon="⚠️")
        sig_classic = ef_sig[idx] if np.isfinite(ef_sig[idx]) else float("nan")
        sig_ref = ref_sig[idx] if np.isfinite(ref_sig[idx]) else float("nan")
        st.subheader(f"Weights at selected target (σ classic={sig_classic:.4f}, σ ref={sig_ref:.4f})")
        w_df = pd.DataFrame({"Asset": names, "ClassicEF": w_classic, "REF": w_ref})
        st.dataframe(w_df.set_index("Asset"))

        dldir = settings.DownloadsPath or "data/outputs/"
        os.makedirs(dldir, exist_ok=True)

        weights_mask = (
            np.isfinite(ref_sig)
            & np.all(np.isfinite(ref["W_bar"]), axis=1)
            & (ref["coverage"] >= coverage_threshold)
        )
        expected_weights = int(
            np.sum(
                (ef.get("ok_mask", np.ones(points_eff, dtype=bool)))
                & (ref["coverage"] >= coverage_threshold)
                & np.isfinite(ref_sig)
            )
        )
        fig_weights = plot_weights_by_risk_stack(
            ref_sig,
            ref["W_bar"],
            names,
            title=f"Resampled weights by risk (coverage ≥ {coverage_threshold:.0%})",
            mask=weights_mask,
            coverage=ref["coverage"],
            coverage_threshold=coverage_threshold,
            min_width=0.002,
            expected_count=expected_weights,
        )
        st.pyplot(fig_weights)
        try:
            fig_weights.savefig(os.path.join(dldir, "weights_by_risk.png"), dpi=200)
        except Exception as e:
            st.warning(f"Could not save weights-by-risk PNG: {e}", icon="⚠️")
        plt.close(fig_weights)

        # Save frontier CSVs
        front_df = pd.DataFrame({
            "target_return": targets,
            "coverage": ref["coverage"],
            "n_valid": ref["n_valid"],
            "sigma_classic": ef_sig,
            "mu_arith_classic": ef_mu,
            "mu_geom_classic": ef_geom,
            "sigma_ref": ref_sig,
            "mu_arith_ref": ref_mu,
            "mu_geom_ref": ref_geom,
        })
        front_df.to_csv(os.path.join(dldir, "frontiers.csv"), index=False)
        weights_ref_df = pd.DataFrame(ref["W_bar"], columns=names)
        weights_ref_df.to_csv(os.path.join(dldir, "ref_weights.csv"), index=False)
        # Save frontier plot PNG for offline validation
        if fig is not None:
            try:
                fig.savefig(os.path.join(dldir, "frontier.png"), dpi=200)
                plt.close(fig)
            except Exception as e:
                st.warning(f"Could not save frontier PNG: {e}", icon="⚠️")

        st.success(f"Frontier CSVs exported to {dldir}", icon="✅")

    with tab3:
        st.subheader("Feasibility / Coverage")
        st.write(
            "Coverage is the fraction of resampled worlds in which a portfolio meeting the "
            "target return (within tolerance) exists under your constraints."
        )
        fig_cov = plot_coverage(targets, ref["coverage"], coverage_threshold)
        st.pyplot(fig_cov)
        try:
            fig_cov.savefig(os.path.join(dldir, "coverage.png"), dpi=200)
        except Exception as e:
            st.warning(f"Could not save coverage PNG: {e}", icon="⚠️")
        plt.close(fig_cov)

        fig_counts = plot_valid_counts(targets, ref["n_valid"])
        st.pyplot(fig_counts)
        try:
            fig_counts.savefig(os.path.join(dldir, "valid_counts.png"), dpi=200)
        except Exception as e:
            st.warning(f"Could not save valid-counts PNG: {e}", icon="⚠️")
        plt.close(fig_counts)

        low_cov = ref["coverage"] < coverage_threshold
        if np.any(low_cov):
            st.warning(
                f"{int(np.sum(low_cov))} target(s) fall below the coverage threshold "
                f"({coverage_threshold:.0%}). These points are fragile under uncertainty.",
                icon="⚠️",
            )

        status = []
        for k in range(points_eff):
            if int(ref["n_valid"][k]) < int(min_valid_bootstraps):
                status.append("Insufficient valid resamples")
            elif not np.isfinite(ref["rets"][k]) or not np.isfinite(ref["risks"][k]):
                status.append("Invalid resampled")
            elif not ef.get("ok_mask", np.ones(points_eff, dtype=bool))[k]:
                status.append("Classic infeasible")
            else:
                status.append("OK")

        diag_df = ref["diagnostics_df"].copy()
        diag_df["CoveragePct"] = diag_df["coverage"] * 100.0
        diag_df["ClassicReturn"] = ef["rets"]
        diag_df["ClassicRisk"] = ef["risks"]
        diag_df["Status"] = status
        diag_df = diag_df[
            [
                "target_return",
                "CoveragePct",
                "n_valid",
                "ClassicReturn",
                "ClassicRisk",
                "resampled_return",
                "resampled_risk",
                "Status",
            ]
        ].rename(
            columns={
                "target_return": "TargetReturn",
                "n_valid": "NumValid",
                "resampled_return": "ResampledReturn",
                "resampled_risk": "ResampledRisk",
            }
        )
        st.dataframe(diag_df)

        with st.expander("Advanced: Coverage heatmap (bootstraps x targets)"):
            heat = ref["ok_mask"].astype(int)
            fig_heat, ax_heat = plt.subplots()
            extent = [float(targets.min()), float(targets.max()), 1, heat.shape[0]]
            ax_heat.imshow(
                heat,
                aspect="auto",
                cmap="Greys",
                origin="lower",
                interpolation="nearest",
                extent=extent,
            )
            ax_heat.set_xlabel("Target return")
            ax_heat.set_ylabel("Bootstrap index")
            ax_heat.set_title("Coverage heatmap")
            fig_heat.tight_layout()
            st.pyplot(fig_heat)
            try:
                fig_heat.savefig(os.path.join(dldir, "coverage_heatmap.png"), dpi=200)
            except Exception as e:
                st.warning(f"Could not save coverage heatmap PNG: {e}", icon="⚠️")
            plt.close(fig_heat)

    with tab4:
        st.subheader("Monte Carlo Spending Simulation")
        rule_name = st.selectbox("Spending rule", options=rules_df["Rule"].tolist(), index=0)
        rule_row = rules_df[rules_df["Rule"]==rule_name].iloc[0]

        # Choose weights: use REF at selected idx by default
        st.caption("Using REF weights at the selected frontier index from the Frontier tab.")
        w = ref["W_bar"][idx]
        weights_ok = np.all(np.isfinite(w))
        if not weights_ok:
            st.warning("Selected resampled weights are unavailable. Pick a point with higher coverage.", icon="⚠️")

        mc_params = _model_copy(mc, update={})
        # Overrides via UI
        mc_params.HorizonYears = st.slider("Horizon (years)", 5, 60, mc_params.HorizonYears, step=1)
        mc_params.NumPaths = st.slider("Number of paths", 500, 20000, mc_params.NumPaths, step=500)
        mc_params.InflationRate = st.number_input("Inflation rate", value=float(mc_params.InflationRate), step=0.005, format="%.3f")
        mc_params.WithdrawalTaxRate = st.number_input(
            "Withdrawal tax rate",
            min_value=0.0,
            max_value=0.95,
            value=float(getattr(mc_params, "WithdrawalTaxRate", 0.24)),
            step=0.01,
            format="%.3f",
        )
        mc_params.ReturnDistribution = "LogNormal"
        st.caption("Return distribution: LogNormal (log-returns, non-negative wealth).")
        mc_params.SerialCorrelation = st.checkbox("AR(1) serial correlation at portfolio level", value=mc_params.SerialCorrelation)
        if mc_params.SerialCorrelation:
            mc_params.AR1_Rho = st.slider("AR(1) ρ", min_value=0.0, max_value=0.95, value=float(mc_params.AR1_Rho), step=0.05)
        mc_params.DriftStressBps = st.number_input("Drift stress (bps)", value=float(mc_params.DriftStressBps), step=5.0)
        mc_params.MgmtFeeBps = st.number_input("Mgmt fee (bps)", value=float(mc_params.MgmtFeeBps), step=5.0)
        mc_params.TxnCostBps = st.number_input("Txn cost (bps)", value=float(mc_params.TxnCostBps), step=5.0)

        if weights_ok:
            out = _cache_mc(
                tuple(mu),
                S,
                w,
                _model_dump(mc_params),
                rule_row.to_dict(),
                mc_params.InflationRate,
                settings.DownloadsPath,
                periods,
            )
            st.success("Monte Carlo complete.", icon="✅")

            # Fan charts
            st.pyplot(fan_chart_from_quantiles(out["wealth_nom_q"], title="Nominal Wealth (fan)"))
            st.pyplot(fan_chart_from_quantiles(out["wealth_real_q"], title="Real Wealth (fan)"))

            # KPIs
            st.subheader("KPIs")
            st.json(out["kpis"])

            st.info(f"CSVs and PNGs written to {settings.DownloadsPath}", icon="💾")

    with tab5:
        st.subheader("Automated Validation")
        st.write("Validation runs automatically on first load.")
        if "validation_ok" in st.session_state:
            if st.session_state["validation_ok"]:
                st.success(st.session_state.get("validation_msg", "Validation passed."), icon="✅")
            else:
                st.error(st.session_state.get("validation_msg", "Validation failed."), icon="🚩")

    # Auto-run validation on first load
    if "validated_once" not in st.session_state:
        st.session_state["validated_once"] = True
        try:
            ok = run_validation(default_xlsx, settings)
            st.session_state["validation_ok"] = bool(ok)
            st.session_state["validation_msg"] = "Validation passed."
            if hasattr(st, "toast"):
                st.toast("Validation ran on first load.", icon="✅")
            else:
                st.info("Validation ran on first load.")
        except Exception as e:
            st.session_state["validation_ok"] = False
            st.session_state["validation_msg"] = f"Validation failed: {e}"
            st.error(f"Validation failed on first load: {e}")

def run_validation(xlsx_path: str, settings: Settings) -> bool:
    logs = []

    def log(msg):
        logs.append(msg)
        st.write(msg)

    # 1) Excel ingestion
    data = read_scenario_xlsx(xlsx_path)
    assets = data["assets"]; corr = data["corr"]; S = data["cov"]
    mu = data["mu_arith"]; G = data["g_inputs"]; vols = data["vols"]
    bounds = data["bounds"]

    assert set(["Assets","Correlation","Settings","MonteCarlo","SpendingRules"]).issubset(
        set(pd.ExcelFile(xlsx_path, engine="openpyxl").sheet_names)), "Missing sheets"
    assert assets.shape[0] == corr.shape[0], "Assets / Correlation size mismatch"
    # PSD check
    ev = eigh(corr.values, eigvals_only=True)
    assert np.min(ev) > -1e-6, "Correlation not PSD"
    log("Excel ingestion ✔️")

    # 2) Conversion: arithmetic > geometric elementwise
    m_test = geometric_to_arithmetic(G, vols)
    assert np.all(m_test > G - 1e-12), "Arithmetic means not larger than geometric"
    log("Geometric→arithmetic conversion ✔️")

    # 3) Frontier + REF (coverage-aware)
    bounds_eff = normalize_bounds(bounds, settings.LongOnly, settings.GlobalMinWeight, settings.GlobalMaxWeight)
    target_info = compute_targets(mu, bounds_eff, int(settings.FrontierPoints))
    targets = target_info["targets"]
    target_bands = compute_target_bands(targets, min_band=TARGET_EPS)
    ef = efficient_frontier(
        mu,
        S,
        settings.RiskFreeRate,
        bounds_eff,
        targets,
        eps=TARGET_EPS,
        target_bands=target_bands,
    )
    ref = resample_frontier_with_coverage(
        mu, S, bounds_eff, targets,
        int(settings.Resamples), float(settings.ResampleYears), settings.Frequency,
        int(settings.RandomSeed),
        eps=TARGET_EPS,
        target_bands=target_bands,
        min_valid_bootstraps=10,
    )
    # Check averaging property for valid points
    W_bar = ref["W_bar"]; W_samples = ref["W_samples"]
    for k in range(len(targets)):
        if ref["n_valid"][k] >= 10:
            avg = np.nanmean(W_samples[:, k, :], axis=0)
            assert np.allclose(W_bar[k], avg, atol=1e-6, equal_nan=True), "Resampled weights not mean of valid samples"
    assert np.all((ref["coverage"] >= 0.0) & (ref["coverage"] <= 1.0)), "Coverage outside [0,1]"
    # Default plotting in compound
    assert settings.FrontierYAxis == "Compound", "Default y-axis should be 'Compound'"
    log("Frontier (40pt) + REF (200×30) ✔️")

    # 4) Monte Carlo quick run
    rule_row = data["rules"][data["rules"]["Rule"]=="ConstantReal"].iloc[0]
    mc_small = _model_copy(data["mc"], update=dict(NumPaths=1000, HorizonYears=10, Seed=999))
    valid_idx = np.where(np.all(np.isfinite(ref["W_bar"]), axis=1))[0]
    assert len(valid_idx) > 0, "No valid resampled weights for Monte Carlo test"
    w0 = ref["W_bar"][valid_idx[len(valid_idx)//2]]
    periods = PERIODS_PER_YEAR.get((settings.Frequency or "").strip().lower(), 1)
    out = simulate_paths(
        mu,
        S,
        w0,
        mc_small,
        rule_row,
        mc_small.InflationRate,
        settings.DownloadsPath,
        periods,
    )
    assert os.path.exists(os.path.join(settings.DownloadsPath, "wealth_quantiles_nominal.csv"))
    assert os.path.exists(os.path.join(settings.DownloadsPath, "wealth_quantiles_real.csv"))
    assert "median_terminal_wealth" in out["kpis"]
    log("Monte Carlo (1k paths, 10y) ✔️")

    # 5) Outputs check
    for fn in ["frontiers.csv", "ref_weights.csv", "wealth_quantiles_nominal.csv", "wealth_quantiles_real.csv"]:
        assert os.path.exists(os.path.join(settings.DownloadsPath, fn)), f"Missing export: {fn}"
    log("Exports present ✔️")

    return True

if __name__ == "__main__":
    main()
