"""Final rank detection plots for paper.

1. Violin by SNR - shows noise effect
2. MAE vs k with alpha - shows both effects together

Usage:
    poetry run python sandbox/rank_detection_viz/final_plots.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.colors import ROSE, TEAL, CYAN, SAND, GRAY, GRAY_LIGHT
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

PROJECT_ROOT = Path(__file__).parents[2]
DATA_PATH = PROJECT_ROOT / "outputs/experiments/simulation/data/rank_detection.csv"
OUTPUT_DIR = get_output_dir()

ALPHAS = [0.1, 1.0, 5.0]
SNRS = [0.4, 0.6, 1.0]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def plot_violin_by_snr(df: pd.DataFrame, output_dir: Path) -> None:
    """Three-panel violin plots, one per SNR."""
    df = df[df["snr"].isin(SNRS)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), sharey=True)

    snr_labels = {0.4: "SNR = 0.4", 0.6: "SNR = 0.6", 1.0: "SNR = 1.0"}
    snr_colors = {0.4: GRAY, 0.6: CYAN, 1.0: ROSE}

    for ax, snr in zip(axes, SNRS):
        subset = df[df["snr"] == snr]
        ranks = sorted(subset["true_rank"].unique())
        data = [subset[subset["true_rank"] == r]["selected_rank"].values for r in ranks]

        parts = ax.violinplot(data, positions=ranks, widths=1.8,
                              showmeans=False, showmedians=True, showextrema=False)

        for pc in parts["bodies"]:
            pc.set_facecolor(snr_colors[snr])
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(1.5)

        ax.plot([0, 32], [0, 32], "--", color=GRAY, lw=1.5, zorder=0)

        ax.set_xlabel("True rank")
        ax.set_title(snr_labels[snr])
        ax.set_xlim(0, 32)
        ax.set_ylim(-2, 45)
        ax.set_xticks([2, 10, 20, 30])
        despine(ax)

    axes[0].set_ylabel("Selected rank")

    plt.tight_layout()
    fig.savefig(output_dir / "violin_by_snr.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_mae_vs_k_by_alpha(df: pd.DataFrame, output_dir: Path) -> None:
    """MAE vs true rank k, with lines for each alpha value."""
    df = df[df["alpha"].isin(ALPHAS)].copy()

    summary = df.groupby(["true_rank", "alpha"]).agg(
        mae=("abs_error", "mean"),
        sem=("abs_error", "sem"),
    ).reset_index()

    fig, ax = create_figure("single")

    alpha_colors = {0.1: ROSE, 1.0: TEAL, 5.0: CYAN}

    for alpha in ALPHAS:
        subset = summary[summary["alpha"] == alpha].sort_values("true_rank")
        ax.plot(subset["true_rank"], subset["mae"], "o-",
                color=alpha_colors[alpha], markersize=4, linewidth=1.5,
                label=f"{alpha}")
        ax.fill_between(
            subset["true_rank"],
            subset["mae"] - subset["sem"],
            subset["mae"] + subset["sem"],
            color=alpha_colors[alpha], alpha=0.15, linewidth=0,
        )

    ax.axhline(0, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=0)

    ax.set_xlabel("True rank")
    ax.set_ylabel("Mean absolute error")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 6)
    ax.legend(frameon=False, loc="upper left", fontsize=7, title=r"$\alpha$",
              title_fontsize=8)
    despine(ax)

    save_figure(fig, output_dir / "mae_vs_k_by_alpha.pdf")


def plot_mae_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap: MAE for each (k, alpha) combination."""
    df = df[df["alpha"].isin(ALPHAS)].copy()

    summary = df.groupby(["true_rank", "alpha"])["abs_error"].mean().reset_index()
    pivot = summary.pivot(index="alpha", columns="true_rank", values="abs_error")
    pivot = pivot.sort_index(ascending=False)

    fig, ax = create_figure("wide")

    # Custom colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#2ecc71", "#f1c40f", "#e74c3c"]
    cmap = LinearSegmentedColormap.from_list("mae", colors)

    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0, vmax=5)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{a}" for a in pivot.index])

    ax.set_xlabel("True rank")
    ax.set_ylabel(r"$\alpha$")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("MAE")

    save_figure(fig, output_dir / "mae_heatmap_k_alpha.pdf")


def plot_combined_k_alpha_snr(df: pd.DataFrame, output_dir: Path) -> None:
    """Two-panel: MAE vs k colored by alpha (left), MAE vs k colored by SNR (right)."""
    fig, axes = create_figure("full_width", nrows=1, ncols=2)

    # Left: by alpha
    ax = axes[0]
    df_alpha = df[df["alpha"].isin(ALPHAS)]
    summary = df_alpha.groupby(["true_rank", "alpha"]).agg(
        mae=("abs_error", "mean"),
        sem=("abs_error", "sem"),
    ).reset_index()

    alpha_colors = {0.1: ROSE, 1.0: TEAL, 5.0: CYAN}

    for alpha in ALPHAS:
        subset = summary[summary["alpha"] == alpha].sort_values("true_rank")
        ax.plot(subset["true_rank"], subset["mae"], "o-",
                color=alpha_colors[alpha], markersize=3, linewidth=1.2,
                label=f"{alpha}")
        ax.fill_between(
            subset["true_rank"],
            subset["mae"] - subset["sem"],
            subset["mae"] + subset["sem"],
            color=alpha_colors[alpha], alpha=0.15, linewidth=0,
        )

    ax.set_xlabel("True rank")
    ax.set_ylabel("Mean absolute error")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 6)
    ax.legend(frameon=False, loc="upper left", fontsize=7, title=r"$\alpha$",
              title_fontsize=8)
    despine(ax)

    # Right: by SNR
    ax = axes[1]
    df_snr = df[df["snr"].isin([0.4, 0.6, 0.8, 1.0])]
    summary = df_snr.groupby(["true_rank", "snr"]).agg(
        mae=("abs_error", "mean"),
        sem=("abs_error", "sem"),
    ).reset_index()

    snr_colors = {0.4: GRAY, 0.6: CYAN, 0.8: TEAL, 1.0: ROSE}

    for snr in [0.4, 0.6, 0.8, 1.0]:
        subset = summary[summary["snr"] == snr].sort_values("true_rank")
        ax.plot(subset["true_rank"], subset["mae"], "o-",
                color=snr_colors[snr], markersize=3, linewidth=1.2,
                label=f"{snr}")
        ax.fill_between(
            subset["true_rank"],
            subset["mae"] - subset["sem"],
            subset["mae"] + subset["sem"],
            color=snr_colors[snr], alpha=0.15, linewidth=0,
        )

    ax.set_xlabel("True rank")
    ax.set_ylabel("Mean absolute error")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 6)
    ax.legend(frameon=False, loc="upper left", fontsize=7, title="SNR",
              title_fontsize=8)
    despine(ax)

    save_figure(fig, output_dir / "mae_vs_k_combined.pdf")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data()

    print("\nGenerating final plots...")

    plot_violin_by_snr(df, OUTPUT_DIR)
    print("  violin_by_snr.pdf")

    plot_mae_vs_k_by_alpha(df, OUTPUT_DIR)
    print("  mae_vs_k_by_alpha.pdf")

    plot_mae_heatmap(df, OUTPUT_DIR)
    print("  mae_heatmap_k_alpha.pdf")

    plot_combined_k_alpha_snr(df, OUTPUT_DIR)
    print("  mae_vs_k_combined.pdf")

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
