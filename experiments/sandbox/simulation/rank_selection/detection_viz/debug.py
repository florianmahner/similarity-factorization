"""Debug rank detection - clean analysis of stable data.

Usage:
    poetry run python sandbox/rank_detection_viz/debug.py
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.colors import TEAL, CYAN, ROSE, GRAY, GRAY_DARK
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

DATA_PATH = Path("outputs/experiments/simulation/data/rank_detection.csv")
OUTPUT_DIR = get_output_dir()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    df = pd.read_csv(DATA_PATH)

    # === Alpha=1.0, SNR=1.0: The best case ===
    df_best = df[(df["alpha"] == 1.0) & (df["snr"] == 1.0)]

    print("="*60)
    print("BEST CASE: Alpha=1.0, SNR=1.0")
    print("="*60)

    summary = df_best.groupby("true_rank").agg(
        selected_mean=("selected_rank", "mean"),
        mae=("abs_error", "mean"),
        accuracy=("is_correct", "mean"),
        bias=("signed_error", "mean"),
    ).round(2)
    print(summary)
    print()
    print(f"Overall MAE: {df_best['abs_error'].mean():.2f}")
    print(f"Overall accuracy: {df_best['is_correct'].mean():.1%}")

    # === Plot 1: Violin for alpha=1.0, SNR=1.0 ===
    fig, ax = create_figure("wide")
    true_ranks = sorted(df_best["true_rank"].unique())
    data = [df_best[df_best["true_rank"] == r]["selected_rank"].values for r in true_ranks]

    parts = ax.violinplot(data, positions=true_ranks, widths=1.5,
                          showmeans=False, showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(TEAL)
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(1.5)

    ax.plot([0, 35], [0, 35], "--", color=GRAY, lw=1.5, zorder=0, label="Identity")
    ax.set_xlabel("True rank")
    ax.set_ylabel("Selected rank")
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 40)
    ax.set_title("α = 1.0, SNR = 1.0")
    despine(ax)
    save_figure(fig, OUTPUT_DIR / "debug_violin_alpha1_snr1.pdf")
    print("\nSaved: debug_violin_alpha1_snr1.pdf")

    # === Plot 2: Compare alphas at SNR=1.0 ===
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), sharey=True)
    alphas = [0.1, 1.0, 10.0]
    alpha_colors = {0.1: CYAN, 1.0: TEAL, 10.0: ROSE}

    for ax, alpha in zip(axes, alphas):
        subset = df[(df["alpha"] == alpha) & (df["snr"] == 1.0)]
        data = [subset[subset["true_rank"] == r]["selected_rank"].values for r in true_ranks]

        parts = ax.violinplot(data, positions=true_ranks, widths=1.5,
                              showmeans=False, showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(alpha_colors[alpha])
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("white")

        ax.plot([0, 35], [0, 35], "--", color=GRAY, lw=1.5, zorder=0)
        ax.set_xlabel("True rank")
        mae = subset["abs_error"].mean()
        ax.set_title(f"α = {alpha} (MAE={mae:.1f})")
        ax.set_xlim(0, 35)
        ax.set_ylim(0, 45)
        despine(ax)

    axes[0].set_ylabel("Selected rank")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "debug_compare_alphas_snr1.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: debug_compare_alphas_snr1.pdf")

    # === Plot 3: SNR effect for alpha=1.0 ===
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), sharey=True)
    snrs = [0.2, 0.6, 1.0]
    snr_colors = {0.2: GRAY, 0.6: CYAN, 1.0: TEAL}

    for ax, snr in zip(axes, snrs):
        subset = df[(df["alpha"] == 1.0) & (df["snr"] == snr)]
        data = [subset[subset["true_rank"] == r]["selected_rank"].values for r in true_ranks]

        parts = ax.violinplot(data, positions=true_ranks, widths=1.5,
                              showmeans=False, showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(snr_colors[snr])
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("white")

        ax.plot([0, 35], [0, 35], "--", color=GRAY, lw=1.5, zorder=0)
        ax.set_xlabel("True rank")
        mae = subset["abs_error"].mean()
        ax.set_title(f"SNR = {snr} (MAE={mae:.1f})")
        ax.set_xlim(0, 35)
        ax.set_ylim(0, 45)
        despine(ax)

    axes[0].set_ylabel("Selected rank")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "debug_compare_snrs_alpha1.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: debug_compare_snrs_alpha1.pdf")

    # === Summary table ===
    print("\n" + "="*60)
    print("SUMMARY: MAE by Alpha and SNR")
    print("="*60)
    pivot = df.pivot_table(values="abs_error", index="snr", columns="alpha", aggfunc="mean")
    print(pivot.round(2))


if __name__ == "__main__":
    main()
