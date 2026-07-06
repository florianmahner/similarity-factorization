"""Generate THINGS figure, 180mm Nature double-column width.

Layout (1 row x 3 cols):
  a) Task overview (placeholder -- assembled in Affinity)
  b) 48-word scatter (true vs predicted similarity)
  c) Pairwise recovery (dimension-wise correlation)

Data sources:
  - 48-word: experiments/analyses/things_behavior/similarity_48/outputs/rank35/
  - Pairwise: experiments/analyses/things_behavior/pairwise/outputs/

Usage:
    poetry run python experiments/figures/plot_things/plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from src.colors import GRAY, GRAY_LIGHT, ROSE, TEAL, setup_style
from src.utils.figure_theme import despine

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSES = PROJECT_ROOT / "experiments" / "analyses" / "things_behavior"
OUTPUT = Path(__file__).resolve().parent / "outputs"

FIG_WIDTH_MM = 180
FIG_WIDTH_IN = FIG_WIDTH_MM / 25.4

THINGS_IMAGES = Path("/SSD/datasets/things/behav1854")


def _nature_rc(font_size: float) -> dict:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 1,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "axes.linewidth": 0.4,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "lines.linewidth": 0.8,
        "lines.markersize": 3,
        "lines.markeredgewidth": 0.3,
    }


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_48word() -> pd.DataFrame:
    return pd.read_csv(ANALYSES / "similarity_48" / "outputs" / "rank35" / "48_performance.csv")


def _load_pairwise() -> pd.DataFrame:
    return pd.read_csv(ANALYSES / "pairwise" / "outputs" / "pairwise_reconstruction.csv")


# ---------------------------------------------------------------------------
# Panel functions
# ---------------------------------------------------------------------------

def _panel_overview(ax: plt.Axes) -> None:
    """Task schematic placeholder -- will be replaced in Affinity."""
    ax.set_axis_off()
    ax.text(0.5, 0.5, "(assembled in Affinity)",
            ha="center", va="center", fontsize=5, color=GRAY_LIGHT,
            transform=ax.transAxes)


def _panel_scatter(ax: plt.Axes, df: pd.DataFrame, fs: float) -> None:
    srf = df[(df["model"] == "SRF") & (df["seed"] == 0)]
    true_vals = srf["true_similarity"].values
    pred_vals = srf["predicted_similarity"].values

    true_norm = (true_vals - true_vals.min()) / (true_vals.max() - true_vals.min())
    pred_norm = (pred_vals - pred_vals.min()) / (pred_vals.max() - pred_vals.min())
    r = np.corrcoef(true_norm, pred_norm)[0, 1]

    ax.scatter(
        true_norm, pred_norm, color=TEAL, alpha=0.4, s=4,
        linewidths=0.2, edgecolors="white", zorder=2,
    )
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=0.6, zorder=1)
    ax.text(0.05, 0.92, f"r = {r:.3f}", transform=ax.transAxes,
            fontsize=fs, fontweight="bold", va="top")
    ax.set_xlabel("True similarity")
    ax.set_ylabel("Predicted similarity (SRF)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    despine(ax)


def _panel_pairwise(ax: plt.Axes, df: pd.DataFrame, fs: float) -> None:
    snr1 = df[df["snr"] == 1.0]
    ann = fs - 0.5

    seeds = sorted(snr1["seed"].unique())
    sorted_corrs_per_seed = []
    for seed in seeds:
        corrs = snr1[snr1["seed"] == seed]["correlation"].values
        sorted_corrs_per_seed.append(np.sort(corrs)[::-1])

    all_sorted = np.array(sorted_corrs_per_seed)
    median_corr = np.median(all_sorted)
    n_dims = all_sorted.shape[1]

    for corrs in sorted_corrs_per_seed:
        alpha = 0.25 if len(seeds) > 3 else 0.5
        ax.scatter(np.arange(n_dims), corrs, color=ROSE, alpha=alpha, s=2.5,
                   linewidths=0, zorder=2)

    ax.axhline(median_corr, color=GRAY, linestyle=(0, (4, 3)), linewidth=0.5, zorder=1)
    ax.text(n_dims - 1, median_corr + 0.02, f"median = {median_corr:.2f}",
            va="bottom", ha="right", fontsize=ann, color=GRAY)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Best pairwise match (r)")
    ax.set_xlim(-1, n_dims)
    ax.set_ylim(-0.05, 1.05)
    despine(ax)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT / f"{name}.pdf"
    plt.rcParams["savefig.bbox"] = "standard"
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    setup_style()

    fs = 5
    plt.rcParams.update(_nature_rc(fs))

    perf_48 = _load_48word()
    pairwise = _load_pairwise()

    ROW_H = 40 / 25.4  # 40mm per row
    fig = plt.figure(figsize=(FIG_WIDTH_IN, ROW_H + 0.3))

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    ax_overview = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[0, 1])
    ax_pairwise = fig.add_subplot(gs[0, 2])

    # Panel labels
    for label, ax in [("a", ax_overview), ("b", ax_scatter), ("c", ax_pairwise)]:
        x_off = -0.02 if ax is ax_overview else -0.15
        ax.text(x_off, 1.12, label, transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="right")

    _panel_overview(ax_overview)
    _panel_scatter(ax_scatter, perf_48, fs)
    _panel_pairwise(ax_pairwise, pairwise, fs)

    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.16, top=0.90)
    _save(fig, "things")


if __name__ == "__main__":
    main()
