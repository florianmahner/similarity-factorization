"""Plot rank detection results for n=900 SNR simulation.

Creates a combined violin plot showing all SNR levels together.

Usage:
    poetry run python sandbox/rank_detection_viz/plot_n900_snr.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.colors import ROSE, TEAL, CYAN, SAND, GRAY, GRAY_LIGHT
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

DATA_PATH = Path(__file__).parent / "outputs/rank_detection_n900_snr.csv"
OUTPUT_DIR = get_output_dir()


def plot_violin_combined(df: pd.DataFrame, output_dir: Path) -> None:
    """Single violin plot with all SNRs colored differently."""
    fig, ax = create_figure("wide")

    true_ranks = sorted(df["true_rank"].unique())
    snrs = sorted(df["snr"].unique())

    # Color map for SNRs
    snr_colors = {
        0.2: GRAY,
        0.4: CYAN,
        0.6: SAND,
        0.8: TEAL,
        1.0: ROSE,
    }

    # Width and offset for grouped violins
    width = 0.15
    offsets = np.linspace(-0.3, 0.3, len(snrs))

    for snr, offset in zip(snrs, offsets):
        subset = df[df["snr"] == snr]
        data = [subset[subset["true_rank"] == r]["selected_rank"].values for r in true_ranks]
        positions = [r + offset for r in true_ranks]

        parts = ax.violinplot(data, positions=positions, widths=width * 2,
                              showmeans=False, showmedians=True, showextrema=False)

        for pc in parts["bodies"]:
            pc.set_facecolor(snr_colors[snr])
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(1)

    # Identity line
    ax.plot([0, 35], [0, 35], "--", color=GRAY, lw=1.5, zorder=0)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=snr_colors[snr], alpha=0.7, label=f"{snr}")
                       for snr in snrs]
    ax.legend(handles=legend_elements, frameon=False, loc="upper left",
              fontsize=7, title="SNR", title_fontsize=8)

    ax.set_xlabel("True rank")
    ax.set_ylabel("Selected rank")
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 45)
    ax.set_xticks(true_ranks)
    despine(ax)

    save_figure(fig, output_dir / "violin_n900_snr_combined.pdf")
    print("  violin_n900_snr_combined.pdf")


def plot_mae_by_snr(df: pd.DataFrame, output_dir: Path) -> None:
    """MAE vs true rank, with lines for each SNR."""
    fig, ax = create_figure("single")

    snrs = sorted(df["snr"].unique())
    snr_colors = {
        0.2: GRAY,
        0.4: CYAN,
        0.6: SAND,
        0.8: TEAL,
        1.0: ROSE,
    }

    summary = df.groupby(["true_rank", "snr"]).agg(
        mae=("abs_error", "mean"),
        sem=("abs_error", "sem"),
    ).reset_index()

    for snr in snrs:
        subset = summary[summary["snr"] == snr].sort_values("true_rank")
        ax.plot(subset["true_rank"], subset["mae"], "o-",
                color=snr_colors[snr], markersize=4, linewidth=1.5,
                label=f"{snr}")
        ax.fill_between(
            subset["true_rank"],
            subset["mae"] - subset["sem"],
            subset["mae"] + subset["sem"],
            color=snr_colors[snr], alpha=0.15, linewidth=0,
        )

    ax.axhline(0, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=0)

    ax.set_xlabel("True rank")
    ax.set_ylabel("Mean absolute error")
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 10)
    ax.legend(frameon=False, loc="upper left", fontsize=7, title="SNR",
              title_fontsize=8)
    despine(ax)

    save_figure(fig, output_dir / "mae_vs_k_by_snr_n900.pdf")
    print("  mae_vs_k_by_snr_n900.pdf")


def plot_violin_panels_by_snr(df: pd.DataFrame, output_dir: Path) -> None:
    """Three-panel violin plot for selected SNRs."""
    snrs_to_plot = [0.4, 0.6, 1.0]
    df = df[df["snr"].isin(snrs_to_plot)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), sharey=True)

    snr_labels = {0.4: "SNR = 0.4", 0.6: "SNR = 0.6", 1.0: "SNR = 1.0"}
    snr_colors = {0.4: GRAY, 0.6: CYAN, 1.0: ROSE}

    true_ranks = sorted(df["true_rank"].unique())

    for ax, snr in zip(axes, snrs_to_plot):
        subset = df[df["snr"] == snr]
        data = [subset[subset["true_rank"] == r]["selected_rank"].values for r in true_ranks]

        parts = ax.violinplot(data, positions=true_ranks, widths=3,
                              showmeans=False, showmedians=True, showextrema=False)

        for pc in parts["bodies"]:
            pc.set_facecolor(snr_colors[snr])
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(1.5)

        ax.plot([0, 35], [0, 35], "--", color=GRAY, lw=1.5, zorder=0)

        ax.set_xlabel("True rank")
        ax.set_title(snr_labels[snr])
        ax.set_xlim(0, 35)
        ax.set_ylim(-2, 45)
        ax.set_xticks(true_ranks)
        despine(ax)

    axes[0].set_ylabel("Selected rank")

    plt.tight_layout()
    fig.savefig(output_dir / "violin_n900_snr_panels.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  violin_n900_snr_panels.pdf")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    print(f"\nData summary:")
    print(f"  n = {df['n'].iloc[0]}")
    print(f"  True ranks: {sorted(df['true_rank'].unique())}")
    print(f"  SNRs: {sorted(df['snr'].unique())}")
    print(f"  Seeds per condition: {df.groupby(['true_rank', 'snr']).size().iloc[0]}")

    print("\nMAE by SNR:")
    print(df.groupby("snr")["abs_error"].mean().round(2))

    print("\nGenerating plots...")
    plot_violin_combined(df, OUTPUT_DIR)
    plot_mae_by_snr(df, OUTPUT_DIR)
    plot_violin_panels_by_snr(df, OUTPUT_DIR)

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
