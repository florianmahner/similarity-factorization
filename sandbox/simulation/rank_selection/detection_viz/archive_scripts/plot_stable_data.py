"""Plot rank detection from stable experiments data.

Uses alpha=1.0 which shows correct SNR behavior.

Usage:
    poetry run python sandbox/rank_detection_viz/plot_stable_data.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.colors import ROSE, TEAL, CYAN, SAND, GRAY, GRAY_LIGHT
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

DATA_PATH = Path("outputs/experiments/simulation/data/rank_detection.csv")
OUTPUT_DIR = get_output_dir()


def plot_violin_by_snr_alpha1(df: pd.DataFrame, output_dir: Path) -> None:
    """Violin plot by SNR for alpha=1.0."""
    df = df[df["alpha"] == 1.0].copy()
    snrs = [0.2, 0.6, 1.0]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), sharey=True)

    snr_colors = {0.2: GRAY, 0.6: CYAN, 1.0: ROSE}
    true_ranks = sorted(df["true_rank"].unique())

    for ax, snr in zip(axes, snrs):
        subset = df[df["snr"] == snr]
        data = [subset[subset["true_rank"] == r]["selected_rank"].values for r in true_ranks]

        parts = ax.violinplot(data, positions=true_ranks, widths=2,
                              showmeans=False, showmedians=True, showextrema=False)

        for pc in parts["bodies"]:
            pc.set_facecolor(snr_colors[snr])
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(1.5)

        # Identity line
        ax.plot([0, 35], [0, 35], "--", color=GRAY, lw=1.5, zorder=0)

        ax.set_xlabel("True rank")
        ax.set_title(f"SNR = {snr}")
        ax.set_xlim(0, 35)
        ax.set_ylim(0, 40)
        despine(ax)

    axes[0].set_ylabel("Selected rank")

    plt.tight_layout()
    fig.savefig(output_dir / "stable_violin_by_snr_alpha1.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  stable_violin_by_snr_alpha1.pdf")


def plot_violin_by_alpha_snr1(df: pd.DataFrame, output_dir: Path) -> None:
    """Violin plot by alpha for SNR=1.0."""
    df = df[df["snr"] == 1.0].copy()
    alphas = [0.1, 1.0, 10.0]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), sharey=True)

    alpha_colors = {0.1: CYAN, 1.0: TEAL, 10.0: ROSE}
    true_ranks = sorted(df["true_rank"].unique())

    for ax, alpha in zip(axes, alphas):
        subset = df[df["alpha"] == alpha]
        data = [subset[subset["true_rank"] == r]["selected_rank"].values for r in true_ranks]

        parts = ax.violinplot(data, positions=true_ranks, widths=2,
                              showmeans=False, showmedians=True, showextrema=False)

        for pc in parts["bodies"]:
            pc.set_facecolor(alpha_colors[alpha])
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(1.5)

        ax.plot([0, 35], [0, 35], "--", color=GRAY, lw=1.5, zorder=0)

        ax.set_xlabel("True rank")
        ax.set_title(f"α = {alpha}")
        ax.set_xlim(0, 35)
        ax.set_ylim(0, 40)
        despine(ax)

    axes[0].set_ylabel("Selected rank")

    plt.tight_layout()
    fig.savefig(output_dir / "stable_violin_by_alpha_snr1.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  stable_violin_by_alpha_snr1.pdf")


def plot_mae_lines_alpha1(df: pd.DataFrame, output_dir: Path) -> None:
    """MAE vs true rank for alpha=1.0, lines by SNR."""
    df = df[df["alpha"] == 1.0].copy()
    fig, ax = create_figure("single")

    snr_colors = {0.2: GRAY, 0.4: CYAN, 0.6: SAND, 0.8: TEAL, 1.0: ROSE}

    summary = df.groupby(["true_rank", "snr"]).agg(
        mae=("abs_error", "mean"),
        sem=("abs_error", "sem"),
    ).reset_index()

    for snr in sorted(df["snr"].unique()):
        subset = summary[summary["snr"] == snr].sort_values("true_rank")
        ax.plot(subset["true_rank"], subset["mae"], "o-",
                color=snr_colors[snr], markersize=4, label=f"{snr}")

    ax.axhline(0, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=0)

    ax.set_xlabel("True rank")
    ax.set_ylabel("Mean absolute error")
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 6)
    ax.legend(frameon=False, title="SNR", fontsize=7, title_fontsize=8)
    despine(ax)

    save_figure(fig, output_dir / "stable_mae_by_snr_alpha1.pdf")
    print("  stable_mae_by_snr_alpha1.pdf")


def plot_accuracy_by_snr(df: pd.DataFrame, output_dir: Path) -> None:
    """Accuracy bar plot by SNR for alpha=1.0."""
    df = df[df["alpha"] == 1.0].copy()
    fig, ax = create_figure("single")

    snrs = sorted(df["snr"].unique())
    acc = df.groupby("snr")["is_correct"].mean()

    colors = [GRAY, CYAN, SAND, TEAL, ROSE]
    bars = ax.bar(range(len(snrs)), [acc[s] for s in snrs], color=colors)

    ax.set_xticks(range(len(snrs)))
    ax.set_xticklabels([f"{s}" for s in snrs])
    ax.set_xlabel("SNR")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    despine(ax)

    save_figure(fig, output_dir / "stable_accuracy_by_snr_alpha1.pdf")
    print("  stable_accuracy_by_snr_alpha1.pdf")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading stable experiments data...")
    df = pd.read_csv(DATA_PATH)

    print(f"\nData: {len(df)} records")
    print(f"Alphas: {sorted(df['alpha'].unique())}")
    print(f"SNRs: {sorted(df['snr'].unique())}")
    print(f"True ranks: {sorted(df['true_rank'].unique())}")

    print("\n=== Alpha=1.0 Summary ===")
    df1 = df[df["alpha"] == 1.0]
    print("MAE by SNR:")
    print(df1.groupby("snr")["abs_error"].mean().round(2))
    print("\nAccuracy by SNR:")
    print(df1.groupby("snr")["is_correct"].mean().round(2))

    print("\nGenerating plots...")
    plot_violin_by_snr_alpha1(df, OUTPUT_DIR)
    plot_violin_by_alpha_snr1(df, OUTPUT_DIR)
    plot_mae_lines_alpha1(df, OUTPUT_DIR)
    plot_accuracy_by_snr(df, OUTPUT_DIR)

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
