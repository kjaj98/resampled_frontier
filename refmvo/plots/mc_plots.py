import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde


def fan_chart_from_quantiles(df: pd.DataFrame, title: str) -> Figure:
    # expects columns like wealth_nominal_p5,... with a 't' column
    t = df["t"].values
    cols = [c for c in df.columns if c != "t"]
    prefix = cols[0].split("_p")[0]
    p5 = df[f"{prefix}_p5"].values
    p25 = df[f"{prefix}_p25"].values
    p50 = df[f"{prefix}_p50"].values
    p75 = df[f"{prefix}_p75"].values
    p95 = df[f"{prefix}_p95"].values

    fig, ax = plt.subplots()
    ax.plot(t, p95, label="p95", linewidth=1)
    ax.plot(t, p75, label="p75", linewidth=1)
    ax.plot(t, p50, label="median", linewidth=2)
    ax.plot(t, p25, label="p25", linewidth=1)
    ax.plot(t, p5, label="p5", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def histogram_with_kde(data: np.ndarray, title: str, nbins: int = 50) -> Figure:
    """Histogram with KDE overlay for terminal distributions."""
    x = np.asarray(data).ravel()
    fig, ax = plt.subplots()
    ax.hist(x, bins=nbins, density=True, alpha=0.6, label="Histogram")
    if len(x) > 1 and np.std(x) > 1e-12:
        try:
            kde = gaussian_kde(x)
            xs = np.linspace(x.min(), x.max(), 200)
            ax.plot(xs, kde(xs), linewidth=2, label="KDE")
        except Exception:
            pass
    else:
        const_val = float(x[0]) if len(x) else 0.0
        ax.axvline(const_val, linewidth=2, label="Value")
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig
