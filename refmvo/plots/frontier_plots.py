from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_coverage(targets: np.ndarray, coverage: np.ndarray, threshold: float = 0.8) -> Figure:
    fig, ax = plt.subplots()
    ax.plot(targets, coverage, marker="o", linewidth=1.5, label="Coverage")
    ax.axhline(threshold, linestyle="--", color="gray", linewidth=1.0, label=f"Threshold {threshold:.0%}")
    ax.set_title("Bootstrap feasibility coverage by target return")
    ax.set_xlabel("Target return")
    ax.set_ylabel("Coverage")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_valid_counts(targets: np.ndarray, n_valid: np.ndarray) -> Figure:
    fig, ax = plt.subplots()
    ax.plot(targets, n_valid, marker="o", linewidth=1.5, label="Valid bootstraps")
    ax.set_title("Valid bootstrap count by target return")
    ax.set_xlabel("Target return")
    ax.set_ylabel("Num valid")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_frontier_with_coverage(
    classic_sig: np.ndarray,
    classic_ret: np.ndarray,
    classic_ok: Optional[np.ndarray],
    ref_sig: np.ndarray,
    ref_ret: np.ndarray,
    coverage: np.ndarray,
    threshold: float = 0.8,
    title: str = "Efficient Frontier (classic vs. resampled)",
    yaxis_title: str = "Expected return",
) -> Figure:
    fig, ax = plt.subplots()

    classic_mask = np.isfinite(classic_sig) & np.isfinite(classic_ret)
    if classic_ok is not None:
        classic_mask = classic_mask & classic_ok

    if np.any(classic_mask):
        ax.plot(
            classic_sig[classic_mask],
            classic_ret[classic_mask],
            marker="o",
            linestyle="--",
            linewidth=1.5,
            label="Classical EF",
        )

    ref_mask = np.isfinite(ref_sig) & np.isfinite(ref_ret)
    high_cov = ref_mask & (coverage >= threshold)
    low_cov = ref_mask & ~high_cov

    if np.any(high_cov):
        ax.plot(
            ref_sig[high_cov],
            ref_ret[high_cov],
            marker="o",
            linewidth=1.5,
            label=f"Resampled EF (coverage >= {threshold:.0%})",
        )

    if np.any(low_cov):
        ax.scatter(
            ref_sig[low_cov],
            ref_ret[low_cov],
            marker="x",
            color="gray",
            label="Resampled EF (low coverage)",
            zorder=10
        )

    ax.set_title(title)
    ax.set_xlabel("Volatility (sigma)")
    ax.set_ylabel(yaxis_title)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_weights_by_risk_stack(
    risks: np.ndarray,
    weights: np.ndarray,
    asset_names: Iterable[str],
    title: str = "Portfolio weights by risk",
    mask: Optional[np.ndarray] = None,
) -> Figure:
    """Stacked bar chart of weights across the risk axis."""
    risks = np.asarray(risks, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2:
        raise ValueError("weights must be a (K, N) array")
    asset_names = list(asset_names)
    if weights.shape[1] != len(asset_names):
        raise ValueError("asset_names length must match weights columns")

    base_mask = np.isfinite(risks) & np.all(np.isfinite(weights), axis=1)
    if mask is not None:
        base_mask = base_mask & np.asarray(mask, dtype=bool)

    fig, ax = plt.subplots()
    if not np.any(base_mask):
        ax.text(0.5, 0.5, "No valid points to plot", ha="center", va="center")
        ax.set_axis_off()
        return fig

    x = risks[base_mask]
    W = weights[base_mask]
    order = np.argsort(x)
    x = x[order]
    W = W[order]

    if len(x) > 1:
        diffs = np.diff(x)
        pos_diffs = diffs[diffs > 0]
        if pos_diffs.size:
            width = 0.8 * float(np.min(pos_diffs))
        else:
            width = max(1e-4, float(x[0]) * 0.1)
    else:
        width = max(1e-4, float(x[0]) * 0.1)

    colors = plt.cm.tab20(np.linspace(0, 1, max(W.shape[1], 1)))
    bottom = np.zeros(len(x))
    for i in range(W.shape[1]):
        ax.bar(
            x,
            W[:, i],
            bottom=bottom,
            width=width,
            color=colors[i % len(colors)],
            label=asset_names[i],
        )
        bottom += W[:, i]

    ax.set_title(title)
    ax.set_xlabel("Volatility (sigma)")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    return fig
