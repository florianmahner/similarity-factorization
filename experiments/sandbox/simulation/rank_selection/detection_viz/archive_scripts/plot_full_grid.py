"""Plot rank detection results from full grid simulation.

Usage:
    poetry run python sandbox/rank_detection_viz/plot_full_grid.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.colors import ROSE, TEAL, CYAN, SAND, GRAY, GRAY_LIGHT
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

DATA_PATH = Path(__file__).parent / "outputs/rank_detection_full_grid.csv"
OUTPUT_DIR = get_output_dir()


def plot_violin_by_alpha(df: pd.DataFrame, output_dir: Path) -> None:
    """Violin plot by alpha."""
    alphas = sorted(df["alpha"].unique())
    fig, axes = plt.subplots(1, len(alphas), figsize=(7.5, 2.8), sharey=True)

    alpha_colors = {0.5: CYAN, 1.0: TEAL, 5.0: ROSE}
    true_ranks = sorted(df["true_rank"].unique())

    for ax, alpha in zip(axes, alphas):
        subset = df[df["alpha"] == alpha]
        data = [subset[subset["true_rank"] == r]["selected_rank"].values for r in true_ranks]

        parts = ax.violinplot(data, positions=true_ranks, widths=3,
                              showmeans=False, showmedians=True, showextrema=False)

        for pc in parts["bodies"]:
            pc.set_facecolor(alpha_colors[alpha])
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(1.5)

        # Identity line
        ax.plot([0, 25], [0, 25], "--", color=GRAY, lw=1.5, zorder=0)

        ax.set_xlabel("True rank")
        ax.set_title(f"α = {alpha}")
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 30)
        ax.set_xticks(true_ranks)
        despine(ax)

    axes[0].set_ylabel("Selected rank")

    plt.tight_layout()
    fig.savefig(output_dir / "full_grid_violin_by_alpha.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  full_grid_violin_by_alpha.pdf")


def plot_violin_by_snr(df: pd.DataFrame, output_dir: Path) -> None:
    """Violin plot by SNR."""
    snrs = [0.2, 0.6, 1.0]  # Select 3 for panels
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), sharey=True)

    snr_colors = {0.2: GRAY, 0.6: CYAN, 1.0: ROSE}
    true_ranks = sorted(df["true_rank"].unique())

    for ax, snr in zip(axes, snrs):
        subset = df[df["snr"] == snr]
        data = [subset[subset["true_rank"] == r]["selected_rank"].values for r in true_ranks]

        parts = ax.violinplot(data, positions=true_ranks, widths=3,
                              showmeans=False, showmedians=True, showextrema=False)

        for pc in parts["bodies"]:
            pc.set_facecolor(snr_colors[snr])
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(1.5)

        ax.plot([0, 25], [0, 25], "--", color=GRAY, lw=1.5, zorder=0)

        ax.set_xlabel("True rank")
        ax.set_title(f"SNR = {snr}")
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 30)
        ax.set_xticks(true_ranks)
        despine(ax)

    axes[0].set_ylabel("Selected rank")

    plt.tight_layout()
    fig.savefig(output_dir / "full_grid_violin_by_snr.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  full_grid_violin_by_snr.pdf")


def plot_mae_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """MAE heatmap: alpha x SNR."""
    pivot = df.groupby(["alpha", "snr"])["abs_error"].mean().unstack()

    fig, ax = create_figure("single")
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", origin="lower")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.1f}" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{y:.1f}" for y in pivot.index])

    ax.set_xlabel("SNR")
    ax.set_ylabel("Alpha")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("MAE")

    # Add values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if val > pivot.values.mean() else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=8)

    save_figure(fig, output_dir / "full_grid_mae_heatmap.pdf")
    print("  full_grid_mae_heatmap.pdf")


def plot_mae_lines(df: pd.DataFrame, output_dir: Path) -> None:
    """MAE vs true rank, lines by alpha and SNR."""
    fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))

    # By alpha
    ax = axes[0]
    alpha_colors = {0.5: CYAN, 1.0: TEAL, 5.0: ROSE}
    summary = df.groupby(["true_rank", "alpha"]).agg(
        mae=("abs_error", "mean"),
        sem=("abs_error", "sem"),
    ).reset_index()

    for alpha in sorted(df["alpha"].unique()):
        subset = summary[summary["alpha"] == alpha].sort_values("true_rank")
        ax.plot(subset["true_rank"], subset["mae"], "o-",
                color=alpha_colors[alpha], markersize=4, label=f"{alpha}")
        ax.fill_between(subset["true_rank"], subset["mae"] - subset["sem"],
                        subset["mae"] + subset["sem"],
                        color=alpha_colors[alpha], alpha=0.15, linewidth=0)

    ax.set_xlabel("True rank")
    ax.set_ylabel("MAE")
    ax.legend(frameon=False, title="α", fontsize=7, title_fontsize=8)
    ax.set_ylim(0, None)
    despine(ax)

    # By SNR
    ax = axes[1]
    snr_colors = {0.2: GRAY, 0.4: CYAN, 0.6: SAND, 0.8: TEAL, 1.0: ROSE}
    summary = df.groupby(["true_rank", "snr"]).agg(
        mae=("abs_error", "mean"),
        sem=("abs_error", "sem"),
    ).reset_index()

    for snr in sorted(df["snr"].unique()):
        subset = summary[summary["snr"] == snr].sort_values("true_rank")
        ax.plot(subset["true_rank"], subset["mae"], "o-",
                color=snr_colors[snr], markersize=4, label=f"{snr}")

    ax.set_xlabel("True rank")
    ax.set_ylabel("MAE")
    ax.legend(frameon=False, title="SNR", fontsize=7, title_fontsize=8)
    ax.set_ylim(0, None)
    despine(ax)

    plt.tight_layout()
    fig.savefig(output_dir / "full_grid_mae_lines.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  full_grid_mae_lines.pdf")


def plot_signed_error_by_snr(df: pd.DataFrame, output_dir: Path) -> None:
    """Signed error distribution by SNR to see over/underestimation."""
    fig, ax = create_figure("wide")

    snrs = sorted(df["snr"].unique())
    snr_colors = {0.2: GRAY, 0.4: CYAN, 0.6: SAND, 0.8: TEAL, 1.0: ROSE}

    positions = np.arange(len(snrs))
    for i, snr in enumerate(snrs):
        data = df[df["snr"] == snr]["signed_error"].values
        parts = ax.violinplot([data], positions=[i], widths=0.7,
                              showmeans=False, showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(snr_colors[snr])
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("white")

    ax.axhline(0, color=GRAY_LIGHT, linestyle="--", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{s}" for s in snrs])
    ax.set_xlabel("SNR")
    ax.set_ylabel("Signed error (selected - true)")
    despine(ax)

    save_figure(fig, output_dir / "full_grid_signed_error_by_snr.pdf")
    print("  full_grid_signed_error_by_snr.pdf")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    print(f"\nData summary:")
    print(f"  n = {df['n'].iloc[0]}")
    print(f"  True ranks: {sorted(df['true_rank'].unique())}")
    print(f"  Alphas: {sorted(df['alpha'].unique())}")
    print(f"  SNRs: {sorted(df['snr'].unique())}")

    print("\nMAE by alpha:")
    print(df.groupby("alpha")["abs_error"].mean().round(2))

    print("\nMAE by SNR:")
    print(df.groupby("snr")["abs_error"].mean().round(2))

    print("\nMean signed error by SNR (positive=overestimate):")
    print(df.groupby("snr")["signed_error"].mean().round(2))

    print("\nGenerating plots...")
    plot_violin_by_alpha(df, OUTPUT_DIR)
    plot_violin_by_snr(df, OUTPUT_DIR)
    plot_mae_heatmap(df, OUTPUT_DIR)
    plot_mae_lines(df, OUTPUT_DIR)
    plot_signed_error_by_snr(df, OUTPUT_DIR)

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
