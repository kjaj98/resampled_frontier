# Resampled Efficient Frontier (REF) + Spending Rule Simulator

A production-quality Streamlit app that runs locally in your browser and implements:

- Michaud (1998) **Resampled Efficient Frontier (REF)**
- A **Monte Carlo spending-rule simulator** (4 common rules)
- Full **Excel (.xlsx)** input pipeline
- Automated **local project scaffolding** (including a sample workbook)

## Quick start

```bash
# 1) Create a fresh virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app (opens http://localhost:8501)
streamlit run app.py
```

> On first run the app prints a short install/run guide to the console, creates `data/` folders,
> and materializes a sample Excel file at `data/inputs/sample_scenario.xlsx` if missing.

## Workflow (what users actually do)

1) Edit `data/inputs/sample_scenario.xlsx` (or another workbook in `data/inputs/`).
2) Run the app: `streamlit run app.py`.
3) Adjust **Settings** in the Excel file or sidebar; the app recomputes automatically.
4) Outputs (CSVs/PNGs) are written to `data/outputs/` without any manual “Run” buttons.

The app is intentionally “settings‑driven”: users change inputs and the app runs end‑to‑end.

## Project structure (what lives where)

```
app.py                      Streamlit entrypoint
refmvo/
  core/
    constraints.py          Bounds logic, validation
    frontier.py             Target grid, classic frontier solver
    resampling.py           Michaud resampling + coverage aggregation
    moments.py              Covariance utilities + mean conversions
    diagnostics.py          Monte Carlo engine + diagnostics tables
    io.py                   Excel I/O + settings models
  plots/
    frontier_plots.py       Frontier + coverage plots
    mc_plots.py             Fan charts + histograms
tests/                      Pytest suite
data/inputs/                Excel inputs
data/outputs/               CSV/PNG exports
```

## Tests

```bash
pytest
```

Solver availability affects optimization tests. If no OSQP/ECOS/SCS solver is installed, tests
that depend on CVXPY will skip automatically.

## Excel input schema

The app defaults to reading from `data/inputs/sample_scenario.xlsx`. It must contain **five** sheets:

### Sheet 1: `Assets`

| Asset            | GeometricReturn | Volatility | MinWeight | MaxWeight | Group        | Currency | FeeBps |
| ---------------- | --------------- | ---------- | --------- | --------- | ------------ | -------- | ------ |
| US Equity        | 0.075           | 0.18       | 0.00      | 0.60      | Equity       | USD      | 5      |
| Dev ex-US Equity | 0.065           | 0.16       | 0.00      | 0.60      | Equity       | USD      | 5      |
| Emerging Equity  | 0.085           | 0.22       | 0.00      | 0.40      | Equity       | USD      | 10     |
| Global IG Bonds  | 0.025           | 0.06       | 0.10      | 0.80      | Fixed Income | USD      | 3      |
| EM Debt (HC)     | 0.040           | 0.10       | 0.00      | 0.30      | Fixed Income | USD      | 8      |
| Global REITs     | 0.060           | 0.18       | 0.00      | 0.30      | Real Assets  | USD      | 10     |

### Sheet 2: `Correlation`

6×6 symmetric matrix with diagonal = 1. Example:

```
1.00  0.80  0.75  0.20  0.35  0.60
0.80  1.00  0.78  0.18  0.32  0.58
0.75  0.78  1.00  0.15  0.30  0.55
0.20  0.18  0.15  1.00  0.30  0.25
0.35  0.32  0.30  0.30  1.00  0.40
0.60  0.58  0.55  0.25  0.40  1.00
```

### Sheet 3: `Settings`

| Key             | Value         |
| --------------- | ------------- |
| RiskFreeRate    | 0.0           |
| LongOnly        | True          |
| GlobalMinWeight | 0.0           |
| GlobalMaxWeight | 1.0           |
| FrontierPoints  | 40            |
| Resamples       | 200           |
| ResampleYears   | 30            |
| RandomSeed      | 123           |
| Shrinkage       | None          |
| Frequency       | Annual        |
| FrontierYAxis   | Compound      |
| DownloadsPath   | data/outputs/ |

### Sheet 4: `MonteCarlo`

| Key                | Value   |
| ------------------ | ------- |
| HorizonYears       | 30      |
| NumPaths           | 10000   |
| StartWealth        | 1000000 |
| InflationRate      | 0.02    |
| StartSpendingYear  | 0       |
| UseRealTerms       | Both    |
| RebalanceFrequency | Annual  |
| ReturnDistribution | LogNormal |
| SerialCorrelation  | False   |
| AR1_Rho            | 0.15    |
| DriftStressBps     | 0       |
| MgmtFeeBps         | 0       |
| TxnCostBps         | 0       |
| Seed               | 123     |

### Sheet 5: `SpendingRules`

| Rule               | S0    | Gamma | Alpha | FloorPct | CeilingPct | LowerWR | UpperWR | Delta |
| ------------------ | ----- | ----- | ----- | -------- | ---------- | ------- | ------- | ----- |
| ConstantReal       | 40000 |       |       |          |            |         |         |       |
| PercentOfPortfolio |       | 0.04  |       |          |            |         |         |       |
| EndowmentHybrid    |       | 0.20  | 0.70  | 0.80     | 1.20       |         |         |       |
| Guardrails         | 40000 |       |       | 0.80     | 1.20       | 0.03    | 0.07    | 0.10  |

## Settings reference (what each field does)

**Settings (Sheet 3)**
- `RiskFreeRate`: used for tangency/Sharpe calculation (frontier plot only).
- `LongOnly`: if True, clamps lower bounds at 0.
- `GlobalMinWeight` / `GlobalMaxWeight`: clamp per‑asset bounds before optimization.
- `FrontierPoints`: number of target returns (K).
- `Resamples`: number of bootstrap worlds (B).
- `ResampleYears`: years per bootstrap sample.
- `RandomSeed`: RNG seed for resampling.
- `Shrinkage`: `None` or `LedoitWolf` (applied to covariance estimates).
- `Frequency`: Annual / Monthly / etc; sets periods per year in resampling.
- `FrontierYAxis`: `Compound` or `Arithmetic` for plotting only.
- `DownloadsPath`: output directory for CSV/PNG artifacts.

**MonteCarlo (Sheet 4)**
- `HorizonYears`: simulation horizon (years).
- `NumPaths`: number of Monte Carlo paths.
- `StartWealth`: initial wealth level.
- `InflationRate`: used to compute real wealth series.
- `StartSpendingYear`: when spending begins.
- `UseRealTerms`: retained for compatibility (outputs include both nominal and real).
- `RebalanceFrequency`: retained for compatibility; not used (annual rebalance assumed).
- `ReturnDistribution`: fixed to `LogNormal` (other values ignored).
- `SerialCorrelation`: AR(1) on **log returns** if True.
- `AR1_Rho`: AR(1) coefficient for log returns.
- `DriftStressBps`: reduces log‑return drift (bps per period).
- `MgmtFeeBps`: reduces log‑return drift (bps per period).
- `TxnCostBps`: retained for compatibility; not used in current MC engine.
- `Seed`: RNG seed for Monte Carlo.

**SpendingRules (Sheet 5)**
- `ConstantReal`: fixed real spending level.
- `PercentOfPortfolio`: spend a fixed % of current wealth.
- `EndowmentHybrid`: blend of prior spending and % of wealth with floor/ceiling.
- `Guardrails`: inflation‑adjusted spending with raise/cut bands.

## App-only controls (sidebar)

- **Coverage threshold**: used to flag/grey low‑coverage frontier points.
- **Min valid bootstraps**: minimum feasible resamples required to accept a target.
- **Shrinkage inside resamples**: applies Ledoit‑Wolf shrinkage per bootstrap.

## Optional `Returns` sheet (for nonparametric resampling)

If provided, the app can bootstrap historical returns instead of simulating MVN.
The sheet must include all asset columns with clean numeric data.

## Arithmetic vs. geometric

Optimization uses **arithmetic means** derived from the input **geometric** means and volatilities.
For asset *i* with geometric return \(G_i\) and volatility \(s_i\):

\[
\mu_i = \ln(1+G_i),\qquad
y_i = \frac{1+\sqrt{1+4 s_i^2 e^{-2\mu_i}}}{2},\qquad
m_i = (1+G_i)\sqrt{y_i}-1.
\]

Diagnostics include a Brent root-finding fallback and the small-σ approximation
\(m_i \approx G_i + \tfrac{1}{2}\, s_i^2/(1+G_i)\).

REF plots **compound (geometric) portfolio returns** by default, while optimization and constraints
always use the arithmetic means and the covariance matrix.

## How the code works (full flow)

1) **Load Excel inputs**: assets, correlations, settings, spending rules.
2) **Build moments**:
   - Convert geometric means to arithmetic means (used for optimization).
   - Construct covariance from volatilities and correlation; enforce PSD.
3) **Apply constraints**: long‑only, per‑asset bounds, global caps.
4) **Compute fixed target grid** (base moments + constraints).
5) **Define target bands**: half the grid spacing per target (minimum `1e-6`).
6) **Classic frontier**: solve the Markowitz QP at each target return band.
7) **Resampled frontier (Michaud)**:
   - Resample returns (parametric MVN or optional bootstrap if returns are provided).
   - Re‑estimate μ/Σ each resample, enforce PSD, solve the same targets.
   - Exclude infeasible points; aggregate weights only across valid resamples.
   - Compute coverage = fraction of feasible resamples per target.
8) **Frontier selection**: user chooses a target index; weights are shown.
9) **Monte Carlo**: simulate portfolio wealth using lognormal returns and spending rules.
10) **Exports**: CSV/PNG outputs are written to `data/outputs/`.

## Optimization details

For each target return `R_k`, the app solves:

```
minimize    w' Σ w
subject to  1' w = 1
            lb <= w <= ub
            μ' w ∈ [R_k - band_k, R_k + band_k]
```

Key implementation notes:
- The **target grid** is computed once (base μ and constraints).
- **Bands** are half the grid spacing (with a small floor) and are used for both classic and resampled solves.
- **Solvers**: CVXPY with OSQP → ECOS → SCS fallback.
- **Residual checks** are applied so a solver “OK” means constraints are actually satisfied.

## Michaud resampling details

- **Parametric (default)**: simulate `T` draws from MVN(μ_base, Σ_base).
  `T = ResampleYears × periods_per_year(Frequency)`.
- **Nonparametric (optional)**: bootstrap rows from the `Returns` sheet (if present).
- **Per resample**: recompute μ_b, Σ_b; apply shrinkage (if enabled); project to PSD.
- **Aggregation**: weights are averaged only across feasible bootstraps at each target.
- **Coverage**: `coverage_k = n_valid_k / B`.

## Feasibility vs. coverage

- **Feasible** at (b, k): solver finds weights that meet constraints and the target band.
- **Coverage** at k: fraction of bootstraps feasible at that target.
- **Exclusion rule**: infeasible points are excluded from averaging; if too few are valid, the resampled point is marked invalid.

## Monte Carlo (lognormal only)

- Distribution: Lognormal (log-returns, non-negative wealth)
- `ReturnDistribution` is fixed to `LogNormal`; other values are ignored.
- Log-return parameters are derived from the portfolio arithmetic mean and volatility
  to preserve the geometric mean implied by the inputs.
- Optional AR(1) **serial correlation** at the **portfolio** level (applied to log returns)
- 4 spending rules: ConstantReal, PercentOfPortfolio, EndowmentHybrid, Guardrails
- Rebalancing (annual), management & transaction costs
- Fan charts (real & nominal), histograms, KPIs and CSV/PNG exports in `data/outputs/`

## Validation

Validation runs **automatically on first load** to check:

1. Excel ingestion and PSD checks
2. Geometric→arithmetic conversion sanity
3. Frontier (40 points) and Michaud REF (200×30)
4. Monte Carlo sanity run (1000 paths × 10 years)
5. Exports exist and are well-formed

## Outputs (written to `data/outputs/`)

- `frontiers.csv`: target grid, classic/ref metrics, coverage
- `ref_weights.csv`: resampled weights by target
- `wealth_quantiles_nominal.csv`, `wealth_quantiles_real.csv`
- `spend_quantiles_nominal.csv`
- `frontier.png`, `fan_nominal.png`, `fan_real.png`
- `terminal_wealth_hist.png`, `terminal_spending_hist.png`
- `mc_kpis.json`

## Assumptions and limitations

- Inputs are **annualized** and consistent in units.
- Lognormal MC assumes continuous compounding and non‑negative wealth.
- Coverage is sensitive to constraints and target band width; tight caps reduce feasible targets.
