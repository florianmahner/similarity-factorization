"""
Publication-quality rank detection plots comparing kappa, BIC, and elbow methods.

Produces two PDF figures:
1. rank_detection_vary_alpha.pdf  -- fixed SNR=1.0, panels for alpha in {0.1, 1.0, 5.0}
2. rank_detection_vary_snr.pdf   -- fixed alpha=5.0, panels for SNR in {0.3, 0.5, 0.7, 1.0}

Usage:
    poetry run python experiments/simulation/plot_kappa_rank_detection.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.figure_theme import CMAP, GRAY, apply_theme, despine, save_figure

PROJECT_ROOT = Path(__file__).parents[2]
DATA_PATH = PROJECT_ROOT / "outputs/experiments/simulation/kappa_rank_detection/kappa_rank_detection.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/simulation/kappa_rank_detection"

TRUE_RANKS = [5, 10, 15, 20, 25, 30, 35, 40]
MAX_RANK = 40
AXIS_LIM = (0, MAX_RANK + 5)
AXIS_TICKS = TRUE_RANKS

METHODS = [
    ("Kappa (ours)", "rank_kappa", CMAP[1], "o"),
    ("BIC", "rank_bic", CMAP[0], "s"),
    ("Elbow", "rank_elbow", CMAP[2], "D"),
]

JITTER_SEEDS = {"rank_kappa": 0, "rank_bic": 1, "rank_elbow": 2}
JITTER_SCALE = 0.35
SCATTER_ALPHA = 0.3
SCATTER_SIZE = 13
MEDIAN_LW = 1.5
MEDIAN_MS = 5


def _add_identity(ax: plt.Axes) -> None:
    lo, hi = AXIS_LIM
    ax.plot([lo, hi], [lo, hi], color=GRAY["light"], lw=1.0, zorder=0)


def _scatter_with_jitter(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    marker: str,
    rng: np.random.Generator,
) -> None:
    jitter = rng.uniform(-JITTER_SCALE, JITTER_SCALE, size=len(x))
    ax.scatter(
        x + jitter,
        y,
        color=color,
        marker=marker,
        s=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
        linewidths=0,
        zorder=2,
    )


def _median_line(
    ax: plt.Axes,
    df: pd.DataFrame,
    rank_col: str,
    color: str,
    marker: str,
    label: str,
) -> None:
    medians = df.groupby("true_rank")[rank_col].median().reset_index()
    ax.plot(
        medians["true_rank"],
        medians[rank_col],
        marker=marker,
        color=color,
        linewidth=MEDIAN_LW,
        markersize=MEDIAN_MS,
        label=label,
        zorder=3,
    )


def _format_panel(ax: plt.Axes, title: str, show_ylabel: bool) -> None:
    ax.set_xlim(*AXIS_LIM)
    ax.set_ylim(*AXIS_LIM)
    ax.set_xticks(AXIS_TICKS)
    ax.set_yticks(AXIS_TICKS)
    ax.set_aspect("equal")
    ax.set_xlabel("True rank", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Detected rank", fontsize=9)
    else:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=9)
    despine(ax)


def _draw_panel(
    ax: plt.Axes,
    subset: pd.DataFrame,
    title: str,
    show_ylabel: bool,
    show_legend: bool,
) -> None:
    _add_identity(ax)
    for label, rank_col, color, marker in METHODS:
        rng = np.random.default_rng(JITTER_SEEDS[rank_col])
        _scatter_with_jitter(ax, subset["true_rank"].values, subset[rank_col].values, color, marker, rng)
        _median_line(ax, subset, rank_col, color, marker, label)
    _format_panel(ax, title, show_ylabel)
    if show_legend:
        ax.legend(frameon=False, fontsize=7, loc="upper left")


def plot_vary_alpha(df: pd.DataFrame, output_dir: Path) -> None:
    """Figure 1: Vary alpha, fixed SNR=1.0."""
    alphas = [0.1, 1.0, 5.0]
    titles = [r"$\alpha$ = 0.1", r"$\alpha$ = 1.0", r"$\alpha$ = 5.0"]
    subset_snr = df[df["snr"] == 1.0]

    ncols = len(alphas)
    panel_w = 2.8
    panel_h = 2.8

    apply_theme()
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(panel_w * ncols, panel_h),
        gridspec_kw={"wspace": 0.35},
    )

    for i, (alpha, title) in enumerate(zip(alphas, titles)):
        subset = subset_snr[subset_snr["alpha"] == alpha]
        _draw_panel(
            axes[i],
            subset,
            title,
            show_ylabel=(i == 0),
            show_legend=(i == 0),
        )

    save_figure(fig, output_dir / "rank_detection_vary_alpha.pdf")


def plot_vary_snr(df: pd.DataFrame, output_dir: Path) -> None:
    """Figure 2: Vary SNR, fixed alpha=5.0."""
    snrs = [0.3, 0.5, 0.7, 1.0]
    titles = ["SNR = 0.3", "SNR = 0.5", "SNR = 0.7", "SNR = 1.0"]
    subset_alpha = df[df["alpha"] == 5.0]

    ncols = len(snrs)
    panel_w = 2.8
    panel_h = 2.8

    apply_theme()
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(panel_w * ncols, panel_h),
        gridspec_kw={"wspace": 0.35},
    )

    for i, (snr, title) in enumerate(zip(snrs, titles)):
        subset = subset_alpha[subset_alpha["snr"] == snr]
        _draw_panel(
            axes[i],
            subset,
            title,
            show_ylabel=(i == 0),
            show_legend=(i == 0),
        )

    save_figure(fig, output_dir / "rank_detection_vary_snr.pdf")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    plot_vary_alpha(df, OUTPUT_DIR)
    plot_vary_snr(df, OUTPUT_DIR)


if __name__ == "__main__":
    main()
