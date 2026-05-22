"""Violin panel plots for rank detection - separate for alpha and SNR.

Usage:
    poetry run python sandbox/rank_detection_viz/mockup_v6.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.colors import ROSE, TEAL, CYAN, GRAY
from src.utils.figure_theme import despine, save_figure
from src.utils import get_output_dir

PROJECT_ROOT = Path(__file__).parents[2]
DATA_PATH = PROJECT_ROOT / "outputs/experiments/simulation/data/rank_detection.csv"
OUTPUT_DIR = get_output_dir()

ALPHAS = [0.1, 1.0, 5.0]
SNRS = [0.4, 0.6, 1.0]
ALPHA_COLORS = {0.1: ROSE, 1.0: TEAL, 5.0: CYAN}
SNR_COLORS = {0.4: GRAY, 0.6: CYAN, 1.0: ROSE}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def plot_violin_by_alpha(df: pd.DataFrame, output_dir: Path) -> None:
    """Three-panel violin plots, one per alpha."""
    df = df[df["alpha"].isin(ALPHAS)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), sharey=True)

    for ax, alpha in zip(axes, ALPHAS):
        subset = df[df["alpha"] == alpha]
        ranks = sorted(subset["true_rank"].unique())
        data = [subset[subset["true_rank"] == r]["selected_rank"].values for r in ranks]

        parts = ax.violinplot(data, positions=ranks, widths=1.8,
                              showmeans=False, showmedians=True, showextrema=False)

        for pc in parts["bodies"]:
            pc.set_facecolor(ALPHA_COLORS[alpha])
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(1.5)

        ax.plot([0, 32], [0, 32], "--", color=GRAY, lw=1.5, zorder=0)

        ax.set_xlabel("True rank")
        ax.set_title(rf"$\alpha = {alpha}$")
        ax.set_xlim(0, 32)
        ax.set_ylim(-2, 45)
        ax.set_xticks([2, 10, 20, 30])
        despine(ax)

    axes[0].set_ylabel("Selected rank")

    plt.tight_layout()
    fig.savefig(output_dir / "v6_violin_by_alpha.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_violin_by_snr(df: pd.DataFrame, output_dir: Path) -> None:
    """Three-panel violin plots, one per SNR."""
    df = df[df["snr"].isin(SNRS)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), sharey=True)

    snr_labels = {0.4: "SNR = 0.4", 0.6: "SNR = 0.6", 1.0: "SNR = 1.0"}

    for ax, snr in zip(axes, SNRS):
        subset = df[df["snr"] == snr]
        ranks = sorted(subset["true_rank"].unique())
        data = [subset[subset["true_rank"] == r]["selected_rank"].values for r in ranks]

        parts = ax.violinplot(data, positions=ranks, widths=1.8,
                              showmeans=False, showmedians=True, showextrema=False)

        for pc in parts["bodies"]:
            pc.set_facecolor(SNR_COLORS[snr])
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
    fig.savefig(output_dir / "v6_violin_by_snr.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data()

    print("\nGenerating violin panels...")

    plot_violin_by_alpha(df, OUTPUT_DIR)
    print("  v6_violin_by_alpha.pdf")

    plot_violin_by_snr(df, OUTPUT_DIR)
    print("  v6_violin_by_snr.pdf")

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
