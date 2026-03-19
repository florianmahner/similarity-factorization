"""Show rank detection is robust to noise (SNR) and structure (alpha).

Focus on k ≤ 15 where the method works well.

Usage:
    poetry run python sandbox/rank_detection_viz/plot_robustness.py
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.colors import TEAL, CYAN, ROSE, SAND, GRAY, GRAY_LIGHT
from src.utils.figure_theme import despine, save_figure
from src.utils import get_output_dir

DATA_PATH = Path("outputs/experiments/simulation/data/rank_detection.csv")
OUTPUT_DIR = get_output_dir()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    df = pd.read_csv(DATA_PATH)

    # Focus on k ≤ 14 where method works well
    df_good = df[df["true_rank"] <= 14].copy()

    print("Focusing on k ≤ 14 (reliable regime)")
    print(f"Total samples: {len(df_good)}")
    print()

    # === Plot 1: Accuracy heatmap (alpha × SNR) ===
    pivot = df_good.pivot_table(
        values="is_correct",
        index="snr",
        columns="alpha",
        aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    im = ax.imshow(pivot.values * 100, cmap="Blues", aspect="auto",
                   vmin=50, vmax=100)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{a}" for a in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{s}" for s in pivot.index])

    ax.set_xlabel("α (Dirichlet concentration)")
    ax.set_ylabel("SNR")

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j] * 100
            color = "white" if val > 75 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    color=color, fontsize=8)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy (%)", fontsize=8)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "accuracy_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: accuracy_heatmap.pdf")
    print(f"  Mean accuracy: {df_good['is_correct'].mean():.1%}")

    # === Plot 2: Line plot - Accuracy by SNR for each alpha ===
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    alpha_colors = {0.1: CYAN, 1.0: TEAL, 5.0: SAND, 10.0: ROSE}

    for alpha in sorted(df_good["alpha"].unique()):
        subset = df_good[df_good["alpha"] == alpha]
        agg = subset.groupby("snr").agg(
            acc=("is_correct", "mean"),
            sem=("is_correct", "sem"),
        ).reset_index()

        ax.errorbar(agg["snr"], agg["acc"] * 100, yerr=agg["sem"] * 100,
                    fmt="o-", color=alpha_colors[alpha], label=f"α = {alpha}",
                    capsize=3, markersize=5, lw=1.5)

    ax.axhline(90, color=GRAY_LIGHT, linestyle="--", lw=0.8)
    ax.set_xlabel("SNR")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(0.1, 1.1)
    ax.set_ylim(50, 105)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    despine(ax)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "accuracy_by_snr_alpha.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: accuracy_by_snr_alpha.pdf")

    # === Plot 3: Two-panel - SNR effect and Alpha effect ===
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2))

    # Left: SNR effect (averaged over alpha)
    ax = axes[0]
    agg_snr = df_good.groupby("snr").agg(
        acc=("is_correct", "mean"),
        sem=("is_correct", "sem"),
    ).reset_index()

    ax.errorbar(agg_snr["snr"], agg_snr["acc"] * 100, yerr=agg_snr["sem"] * 100,
                fmt="o-", color=TEAL, capsize=3, markersize=6, lw=2)

    ax.axhline(90, color=GRAY_LIGHT, linestyle="--", lw=0.8)
    ax.set_xlabel("SNR")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(0.1, 1.1)
    ax.set_ylim(70, 100)
    ax.set_title("Effect of noise", fontsize=9)
    despine(ax)

    # Right: Alpha effect (averaged over SNR)
    ax = axes[1]
    agg_alpha = df_good.groupby("alpha").agg(
        acc=("is_correct", "mean"),
        sem=("is_correct", "sem"),
    ).reset_index()

    ax.errorbar(agg_alpha["alpha"], agg_alpha["acc"] * 100, yerr=agg_alpha["sem"] * 100,
                fmt="o-", color=ROSE, capsize=3, markersize=6, lw=2)

    ax.axhline(90, color=GRAY_LIGHT, linestyle="--", lw=0.8)
    ax.set_xlabel("α (Dirichlet)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xscale("log")
    ax.set_xlim(0.05, 15)
    ax.set_ylim(70, 100)
    ax.set_title("Effect of structure", fontsize=9)
    despine(ax)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "robustness_two_panel.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: robustness_two_panel.pdf")

    # === Summary stats ===
    print()
    print("="*50)
    print("Summary (k ≤ 14):")
    print("="*50)
    print(f"Overall accuracy: {df_good['is_correct'].mean():.1%}")
    print(f"Mean absolute error: {df_good['abs_error'].mean():.2f}")
    print()
    print("Accuracy by SNR:")
    print(agg_snr.round(2))
    print()
    print("Accuracy by alpha:")
    print(agg_alpha.round(2))

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
