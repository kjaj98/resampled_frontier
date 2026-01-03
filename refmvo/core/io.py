import os
import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from refmvo.core.moments import build_covariance, geometric_to_arithmetic, nearest_psd
from refmvo.core.moments import shrink_cov_from_samples

# Pydantic v1/v2 compatibility
try:
    from pydantic import BaseModel, ConfigDict, field_validator as _field_validator

    _has_field_validator = True
    _model_config = ConfigDict(extra="ignore")
except ImportError:  # pragma: no cover - legacy fallback
    from pydantic import BaseModel, validator as _validator  # type: ignore

    _has_field_validator = False
    _field_validator = None  # type: ignore
    _model_config = None


class Settings(BaseModel):
    RiskFreeRate: float = 0.0
    LongOnly: bool = True
    GlobalMinWeight: float = 0.0
    GlobalMaxWeight: float = 1.0
    FrontierPoints: int = 40
    Resamples: int = 200
    ResampleYears: int = 30
    RandomSeed: int = 123
    Shrinkage: Optional[str] = "None"  # None|LedoitWolf
    Frequency: str = "Annual"
    FrontierYAxis: str = "Compound"  # Compound|Arithmetic
    DownloadsPath: str = "data/outputs/"

    if _has_field_validator:

        @_field_validator("FrontierYAxis")
        @classmethod
        def _chk_y(cls, v):
            assert v in ["Compound", "Arithmetic"]
            return v

    else:

        @_validator("FrontierYAxis")
        def _chk_y(cls, v):
            assert v in ["Compound", "Arithmetic"]
            return v


class MonteCarloParams(BaseModel):
    if _model_config is not None:
        model_config = _model_config
    else:  # pragma: no cover - legacy fallback
        class Config:
            extra = "ignore"

    HorizonYears: int = 30
    NumPaths: int = 10000
    StartWealth: float = 1_000_000.0
    InflationRate: float = 0.02
    StartSpendingYear: int = 0
    UseRealTerms: str = "Both"  # Real|Nominal|Both
    RebalanceFrequency: str = "Annual"
    ReturnDistribution: str = "LogNormal"  # LogNormal
    SerialCorrelation: bool = False
    AR1_Rho: float = 0.15
    DriftStressBps: float = 0.0
    MgmtFeeBps: float = 0.0
    TxnCostBps: float = 0.0
    Seed: int = 123


def ensure_scaffold(
    input_dir: str,
    output_dir: str,
    streamlit_dir: str,
    sample_xlsx: str,
) -> None:
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(streamlit_dir, exist_ok=True)
    cfg = os.path.join(streamlit_dir, "config.toml")
    if not os.path.exists(cfg):
        with open(cfg, "w", encoding="utf-8") as f:
            f.write("[server]\nheadless = false\nrunOnSave = true\n")

    if not os.path.exists(sample_xlsx):
        write_sample_workbook(sample_xlsx)


def write_sample_workbook(path: str) -> None:
    assets = pd.DataFrame(
        {
            "Asset": [
                "US Equity",
                "Dev ex-US Equity",
                "Emerging Equity",
                "Global IG Bonds",
                "EM Debt (HC)",
                "Global REITs",
            ],
            "GeometricReturn": [0.075, 0.065, 0.085, 0.025, 0.040, 0.060],
            "Volatility": [0.18, 0.16, 0.22, 0.06, 0.10, 0.18],
            "MinWeight": [0.00, 0.00, 0.00, 0.10, 0.00, 0.00],
            "MaxWeight": [0.60, 0.60, 0.40, 0.80, 0.30, 0.30],
            "Group": ["Equity", "Equity", "Equity", "Fixed Income", "Fixed Income", "Real Assets"],
            "Currency": ["USD"] * 6,
            "FeeBps": [5, 5, 10, 3, 8, 10],
        }
    )

    corr_vals = np.array(
        [
            [1.00, 0.80, 0.75, 0.20, 0.35, 0.60],
            [0.80, 1.00, 0.78, 0.18, 0.32, 0.58],
            [0.75, 0.78, 1.00, 0.15, 0.30, 0.55],
            [0.20, 0.18, 0.15, 1.00, 0.30, 0.25],
            [0.35, 0.32, 0.30, 0.30, 1.00, 0.40],
            [0.60, 0.58, 0.55, 0.25, 0.40, 1.00],
        ]
    )
    corr = pd.DataFrame(corr_vals, columns=assets["Asset"], index=assets["Asset"])

    settings = pd.DataFrame(
        {
            "Key": [
                "RiskFreeRate",
                "LongOnly",
                "GlobalMinWeight",
                "GlobalMaxWeight",
                "FrontierPoints",
                "Resamples",
                "ResampleYears",
                "RandomSeed",
                "Shrinkage",
                "Frequency",
                "FrontierYAxis",
                "DownloadsPath",
            ],
            "Value": [0.0, "True", 0.0, 1.0, 40, 200, 30, 123, "None", "Annual", "Compound", "data/outputs/"],
        }
    )

    monte = pd.DataFrame(
        {
            "Key": [
                "HorizonYears",
                "NumPaths",
                "StartWealth",
                "InflationRate",
                "StartSpendingYear",
                "UseRealTerms",
                "RebalanceFrequency",
                "ReturnDistribution",
                "SerialCorrelation",
                "AR1_Rho",
                "DriftStressBps",
                "MgmtFeeBps",
                "TxnCostBps",
                "Seed",
            ],
            "Value": [30, 10000, 1000000, 0.02, 0, "Both", "Annual", "LogNormal", "False", 0.15, 0, 0, 0, 123],
        }
    )

    rules = pd.DataFrame(
        {
            "Rule": ["ConstantReal", "PercentOfPortfolio", "EndowmentHybrid", "Guardrails"],
            "S0": [40000, None, None, 40000],
            "Gamma": [None, 0.04, 0.20, None],
            "Alpha": [None, None, 0.70, None],
            "FloorPct": [None, None, 0.80, 0.80],
            "CeilingPct": [None, None, 1.20, 1.20],
            "LowerWR": [None, None, None, 0.03],
            "UpperWR": [None, None, None, 0.07],
            "Delta": [None, None, None, 0.10],
        }
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        assets.to_excel(writer, sheet_name="Assets", index=False)
        corr.to_excel(writer, sheet_name="Correlation")
        settings.to_excel(writer, sheet_name="Settings", index=False)
        monte.to_excel(writer, sheet_name="MonteCarlo", index=False)
        rules.to_excel(writer, sheet_name="SpendingRules", index=False)


def read_scenario_xlsx(path: str, autofill: bool = True) -> Dict[str, Any]:
    try:
        xl = pd.ExcelFile(path, engine="openpyxl")
        sheets = xl.sheet_names
    except Exception as e:
        raise RuntimeError(f"Failed to open Excel at {path}: {e}")

    required = ["Assets", "Correlation", "Settings", "MonteCarlo", "SpendingRules"]
    missing = [s for s in required if s not in sheets]
    if missing:
        if autofill:
            warnings.warn(
                f"Missing sheets {missing}. Loading built-in synthetic demo.",
                RuntimeWarning,
            )
            write_sample_workbook(path)
            xl = pd.ExcelFile(path, engine="openpyxl")
            sheets = xl.sheet_names
        else:
            raise ValueError(f"Missing required sheets: {missing}")

    assets = xl.parse("Assets")
    corr = xl.parse("Correlation", index_col=0)
    settings_df = xl.parse("Settings")
    mc_df = xl.parse("MonteCarlo")
    rules_df = xl.parse("SpendingRules")

    def _norm(v):
        if pd.isna(v):
            return None
        if isinstance(v, str):
            lv = v.strip().lower()
            if lv in ["true", "false"]:
                return lv == "true"
            if lv in ["none", "nan", ""]:
                return None
        return v

    settings = Settings(**{r.Key: _norm(r.Value) for _, r in settings_df.iterrows()})
    mc_params = MonteCarloParams(**{r.Key: _norm(r.Value) for _, r in mc_df.iterrows()})

    for col in ["Asset", "GeometricReturn", "Volatility", "MinWeight", "MaxWeight"]:
        if col not in assets.columns:
            raise ValueError(f"Assets sheet missing column: {col}")
    n = len(assets)
    if corr.shape != (n, n):
        raise ValueError(f"Correlation must be {n}x{n}, got {corr.shape}")
    if not np.allclose(corr.values, corr.values.T, atol=1e-10):
        raise ValueError("Correlation not symmetric")
    if not np.allclose(np.diag(corr.values), np.ones(n), atol=1e-10):
        raise ValueError("Correlation diagonal must be 1's")

    eigvals = np.linalg.eigvalsh(corr.values)
    if np.min(eigvals) < -1e-8:
        warnings.warn("Correlation not PSD; projecting to nearest PSD.", RuntimeWarning)
        corr = pd.DataFrame(nearest_psd(corr.values), index=assets["Asset"], columns=assets["Asset"])

    vols = assets["Volatility"].astype(float).values
    cov = build_covariance(vols, corr.values)

    if (settings.Shrinkage or "").lower() == "ledoitwolf":
        rng = np.random.default_rng(settings.RandomSeed)
        X = rng.multivariate_normal(np.zeros(n), cov, size=500)
        cov = shrink_cov_from_samples(X)

    G = assets["GeometricReturn"].astype(float).values
    m = geometric_to_arithmetic(G, vols)
    fee_bps = assets.get("FeeBps", pd.Series([0] * n)).astype(float).values
    m_net = m - fee_bps / 10_000.0

    bounds = np.vstack([assets["MinWeight"].values, assets["MaxWeight"].values]).T
    names = assets["Asset"].tolist()

    returns = None
    returns_info = {"available": False, "reason": "No Returns sheet provided."}
    if "Returns" in sheets:
        returns_raw = xl.parse("Returns")
        if set(names).issubset(returns_raw.columns):
            returns = returns_raw[names].apply(pd.to_numeric, errors="coerce")
            before = returns.shape[0]
            returns = returns.dropna(how="any")
            returns_info = {
                "available": True,
                "rows": int(returns.shape[0]),
                "cols": int(returns.shape[1]),
                "dropped_rows": int(before - returns.shape[0]),
            }
        else:
            missing_cols = sorted(set(names) - set(returns_raw.columns))
            returns_info = {
                "available": False,
                "reason": f"Returns sheet missing asset columns: {missing_cols}",
            }

    return dict(
        assets=assets,
        names=names,
        corr=corr,
        cov=cov,
        mu_arith=m_net,
        mu_arith_gross=m,
        g_inputs=G,
        vols=vols,
        bounds=bounds,
        settings=settings,
        mc=mc_params,
        rules=rules_df,
        returns=returns,
        returns_info=returns_info,
    )


__all__ = ["Settings", "MonteCarloParams", "ensure_scaffold", "write_sample_workbook", "read_scenario_xlsx"]
