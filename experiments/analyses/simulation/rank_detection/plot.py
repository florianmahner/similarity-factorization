"""Rank detection plots: detected vs true rank across methods.

Two figures, each 1x3 panels (one per method):
1. rank_detection_vary_complexity.pdf -- all SNRs pooled, dots colored by alpha
2. rank_detection_vary_snr.pdf       -- all alphas pooled, dots colored by SNR

Usage:
    poetry run python experiments/analyses/simulation/rank_detection/plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.colors import ROSE, TEAL, INDIGO, GRAY_DARK, GRAY_LIGHT, lighten, setup_style
from src.utils.figure_theme import despine

TASK_DIR = Path(__file__).parent
DATA_PATH = TASK_DIR / "outputs" / "rank_detection.csv"
OUTPUT_DIR = TASK_DIR / "outputs"

METHODS = [
    ("Coherence ($\\kappa$)", "rank_kappa"),
    ("Parallel analysis", "rank_parallel"),
    ("Scree test", "rank_elbow"),
]


def _draw_panel(ax, df, rank_col, color_vals, cmap, norm, title, show_ylabel, jitter_seed=42):
    ranks = sorted(df["true_rank"].unique())
    lo, hi = 0, max(ranks) + 3

    ax.plot([lo, hi], [lo, hi], color=GRAY_LIGHT, lw=0.6, zorder=0)

    rng = np.random.default_rng(jitter_seed)
    jx = rng.uniform(-0.6, 0.6, len(df))
    jy = rng.uniform(-0.6, 0.6, len(df))

    ax.scatter(
        df["true_rank"].values + jx,
        df[rank_col].values + jy,
        c=color_vals,
        cmap=cmap,
        norm=norm,
        s=10,
        alpha=0.5,
        edgecolors="white",
        linewidths=0.15,
        zorder=2,
    )

    med = df.groupby("true_rank")[rank_col].median()
    ax.plot(
        med.index, med.values,
        "o-", color="black", lw=1.5, ms=3,
        markeredgecolor="white", markeredgewidth=0.5, zorder=5,
    )

    mae = np.mean(np.abs(df[rank_col] - df["true_rank"]))
    ax.text(
        0.03, 0.97, f"MAE = {mae:.1f}",
        transform=ax.transAxes, fontsize=6.5, ha="left", va="top",
        color=GRAY_DARK,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
    )

    ticks = [r for r in ranks if r % 10 == 0 or r == ranks[0]]
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("True rank")
    if show_ylabel:
        ax.set_ylabel("Detected rank")
    else:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=9, fontweight="normal")
    ax.set_aspect("equal")
    despine(ax)


def plot_vary_complexity(df, output_dir):
    """All SNRs pooled, dots colored by alpha (log scale)."""
    setup_style()

    alphas = sorted(df["alpha"].unique())
    norm = mcolors.LogNorm(vmin=min(alphas), vmax=max(alphas))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "complexity", [lighten(ROSE, 0.7), ROSE, INDIGO], N=256
    )

    ncols = len(METHODS)
    fig, axes = plt.subplots(
        1, ncols, figsize=(2.8 * ncols + 0.6, 3.0),
        gridspec_kw={"wspace": 0.08, "right": 0.88},
    )

    for i, (label, col) in enumerate(METHODS):
        _draw_panel(axes[i], df, col, df["alpha"].values, cmap, norm, label, show_ylabel=(i == 0), jitter_seed=42)

    cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Complexity ($\\alpha$)", fontsize=7.5)
    cbar.set_ticks(alphas)
    cbar.set_ticklabels([str(a) for a in alphas])
    cbar.ax.tick_params(labelsize=6.5)

    fig.savefig(output_dir / "rank_detection_vary_complexity.pdf", format="pdf")
    plt.close(fig)
    print(f"Saved rank_detection_vary_complexity.pdf ({len(df)} points)")


def plot_vary_snr(df, output_dir):
    """All alphas pooled, dots colored by SNR."""
    setup_style()

    snrs = sorted(df["snr"].unique())
    norm = mcolors.Normalize(vmin=min(snrs), vmax=max(snrs))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "snr", [lighten(TEAL, 0.7), TEAL, INDIGO], N=256
    )

    ncols = len(METHODS)
    fig, axes = plt.subplots(
        1, ncols, figsize=(2.8 * ncols + 0.6, 3.0),
        gridspec_kw={"wspace": 0.08, "right": 0.88},
    )

    for i, (label, col) in enumerate(METHODS):
        _draw_panel(axes[i], df, col, df["snr"].values, cmap, norm, label, show_ylabel=(i == 0), jitter_seed=99)

    cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Signal-to-noise ratio", fontsize=7.5)
    cbar.set_ticks(snrs)
    cbar.set_ticklabels([str(s) for s in snrs])
    cbar.ax.tick_params(labelsize=6.5)

    fig.savefig(output_dir / "rank_detection_vary_snr.pdf", format="pdf")
    plt.close(fig)
    print(f"Saved rank_detection_vary_snr.pdf ({len(df)} points)")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")

    plot_vary_complexity(df, OUTPUT_DIR)
    plot_vary_snr(df, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
