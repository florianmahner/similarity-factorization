"""Clean publication plot showing rank detection works.

Usage:
    poetry run python sandbox/rank_detection_viz/plot_rank_detection_works.py
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.colors import TEAL, CYAN, GRAY, GRAY_LIGHT, GRAY_DARK
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

DATA_PATH = Path("outputs/experiments/simulation/data/rank_detection.csv")
OUTPUT_DIR = get_output_dir()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    df = pd.read_csv(DATA_PATH)

    # === Plot 1: Selected vs True rank (best case: alpha=1, SNR=1) ===
    df_best = df[(df["alpha"] == 1.0) & (df["snr"] == 1.0)]

    fig, ax = create_figure("single")

    true_ranks = sorted(df_best["true_rank"].unique())

    # Aggregate
    agg = df_best.groupby("true_rank").agg(
        selected_mean=("selected_rank", "mean"),
        selected_std=("selected_rank", "std"),
    ).reset_index()

    ax.errorbar(agg["true_rank"], agg["selected_mean"], yerr=agg["selected_std"],
                fmt="o", color=TEAL, capsize=3, markersize=5, lw=1.5)

    # Identity line
    ax.plot([0, 35], [0, 35], "--", color=GRAY, lw=1.5, zorder=0)

    ax.set_xlabel("True rank")
    ax.set_ylabel("Selected rank")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 35)
    ax.set_aspect("equal")
    despine(ax)

    save_figure(fig, OUTPUT_DIR / "selected_vs_true.pdf")
    print("Saved: selected_vs_true.pdf")

    # === Plot 2: Violin plot showing distribution ===
    fig, ax = create_figure("wide")

    data = [df_best[df_best["true_rank"] == r]["selected_rank"].values for r in true_ranks]

    parts = ax.violinplot(data, positions=true_ranks, widths=1.8,
                          showmeans=False, showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(TEAL)
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(1.5)

    ax.plot([0, 32], [0, 32], "--", color=GRAY, lw=1.5, zorder=0)

    ax.set_xlabel("True rank")
    ax.set_ylabel("Selected rank")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 38)
    despine(ax)

    save_figure(fig, OUTPUT_DIR / "violin_selected_vs_true.pdf")
    print("Saved: violin_selected_vs_true.pdf")

    # === Plot 3: Accuracy by rank ===
    fig, ax = create_figure("single")

    agg = df_best.groupby("true_rank").agg(
        accuracy=("is_correct", "mean"),
        sem=("is_correct", "sem"),
    ).reset_index()

    ax.bar(agg["true_rank"], agg["accuracy"] * 100, width=1.8,
           color=TEAL, alpha=0.8, edgecolor="white", linewidth=0.5)

    ax.axhline(50, color=GRAY_LIGHT, linestyle="--", lw=0.8)

    ax.set_xlabel("True rank")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 105)
    despine(ax)

    save_figure(fig, OUTPUT_DIR / "accuracy_by_rank.pdf")
    print("Saved: accuracy_by_rank.pdf")

    # === Plot 4: Effect of SNR (alpha=1.0) ===
    fig, ax = create_figure("single")

    df_alpha1 = df[df["alpha"] == 1.0]
    snr_colors = {0.2: GRAY_DARK, 0.6: CYAN, 1.0: TEAL}

    for snr in [0.2, 0.6, 1.0]:
        subset = df_alpha1[df_alpha1["snr"] == snr]
        agg = subset.groupby("true_rank").agg(
            accuracy=("is_correct", "mean"),
        ).reset_index()

        ax.plot(agg["true_rank"], agg["accuracy"] * 100, "o-",
                color=snr_colors[snr], label=f"SNR = {snr}",
                markersize=4, lw=1.5)

    ax.axhline(50, color=GRAY_LIGHT, linestyle="--", lw=0.8)

    ax.set_xlabel("True rank")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    despine(ax)

    save_figure(fig, OUTPUT_DIR / "accuracy_by_snr.pdf")
    print("Saved: accuracy_by_snr.pdf")

    # === Summary stats ===
    print()
    print("="*50)
    print("Summary (alpha=1.0, SNR=1.0):")
    print("="*50)
    print(f"Overall accuracy: {df_best['is_correct'].mean():.1%}")
    print(f"Mean absolute error: {df_best['abs_error'].mean():.2f}")
    print(f"Accuracy for k ≤ 14: {df_best[df_best['true_rank'] <= 14]['is_correct'].mean():.1%}")

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
