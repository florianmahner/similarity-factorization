"""Rank detection scatter plots: detected vs true rank.

Two separate PDF figures:
1. rank_detection_vary_complexity.pdf -- fixed SNR=0.7, dots colored by alpha
2. rank_detection_vary_snr.pdf       -- fixed alpha=2.0, dots colored by SNR

Each figure has 4 panels (one per method), with a shared colorbar showing
the varying parameter.

Usage:
    poetry run python experiments/simulation/plot_kappa_rank_detection.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.utils.figure_theme import CMAP, GRAY, apply_theme, despine, save_figure

PROJECT_ROOT = Path(__file__).parents[2]
DATA_PATH = PROJECT_ROOT / "outputs/experiments/simulation/kappa_rank_detection/kappa_rank_detection.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/simulation/kappa_rank_detection"

METHODS = [
    ("Coherence ($\\kappa$)", "rank_kappa"),
    ("Parallel analysis", "rank_parallel"),
    ("Cophenetic", "rank_cophenetic"),
    ("Elbow", "rank_elbow"),
]


def _draw_panel(ax, df, rank_col, color_col, cmap, norm, title, show_ylabel):
    """Scatter plot with dots colored by a continuous variable."""
    ranks = sorted(df["true_rank"].unique())
    lo, hi = 0, max(ranks) + 5

    ax.plot([lo, hi], [lo, hi], color=GRAY["light"], lw=0.8, zorder=0)

    rng = np.random.default_rng(42)
    jx = rng.uniform(-0.8, 0.8, len(df))
    jy = rng.uniform(-0.8, 0.8, len(df))

    ax.scatter(
        df["true_rank"].values + jx,
        df[rank_col].values + jy,
        c=df[color_col].values,
        cmap=cmap, norm=norm,
        s=14, alpha=0.5, edgecolors="none", zorder=2,
    )

    # Median line (black, method-agnostic)
    med = df.groupby("true_rank")[rank_col].median()
    ax.plot(
        med.index, med.values,
        "o-", color="black", lw=1.5, ms=4,
        markeredgecolor="white", markeredgewidth=0.5,
        zorder=4,
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks(ranks)
    ax.set_yticks(ranks)
    ax.set_xlabel("True rank")
    if show_ylabel:
        ax.set_ylabel("Detected rank")
    else:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    despine(ax)


def plot_vary_complexity(df, output_dir):
    """Fixed SNR=0.7, all alphas pooled, colored by alpha."""
    apply_theme()
    sub = df[df["snr"] == 0.7].copy()

    alphas = sorted(sub["alpha"].unique())
    norm = mcolors.LogNorm(vmin=min(alphas), vmax=max(alphas))
    cmap = "viridis"

    ncols = len(METHODS)
    fig, axes = plt.subplots(1, ncols, figsize=(2.8 * ncols, 3.0),
                              gridspec_kw={"wspace": 0.25})

    for i, (label, col) in enumerate(METHODS):
        _draw_panel(axes[i], sub, col, "alpha", cmap, norm, label, show_ylabel=(i == 0))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.8, pad=0.02)
    cbar.set_label("Complexity ($\\alpha$)", fontsize=8)
    cbar.set_ticks(alphas)
    cbar.set_ticklabels([str(a) for a in alphas])

    save_figure(fig, output_dir / "rank_detection_vary_complexity.pdf")
    plt.close(fig)
    print(f"Saved rank_detection_vary_complexity.pdf ({len(sub)} points)")


def plot_vary_snr(df, output_dir):
    """Fixed alpha=2.0, all SNRs pooled, colored by SNR."""
    apply_theme()
    sub = df[df["alpha"] == 2.0].copy()

    snrs = sorted(sub["snr"].unique())
    norm = mcolors.Normalize(vmin=min(snrs), vmax=max(snrs))
    cmap = "plasma"

    ncols = len(METHODS)
    fig, axes = plt.subplots(1, ncols, figsize=(2.8 * ncols, 3.0),
                              gridspec_kw={"wspace": 0.25})

    for i, (label, col) in enumerate(METHODS):
        _draw_panel(axes[i], sub, col, "snr", cmap, norm, label, show_ylabel=(i == 0))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.8, pad=0.02)
    cbar.set_label("Signal-to-noise ratio", fontsize=8)
    cbar.set_ticks(snrs)
    cbar.set_ticklabels([str(s) for s in snrs])

    save_figure(fig, output_dir / "rank_detection_vary_snr.pdf")
    plt.close(fig)
    print(f"Saved rank_detection_vary_snr.pdf ({len(sub)} points)")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")

    plot_vary_complexity(df, OUTPUT_DIR)
    plot_vary_snr(df, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
