"""Separate rank detection plots for each SNR level.

Usage:
    poetry run python sandbox/rank_detection_viz/plot_rank_detection_by_snr.py
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.colors import TEAL, CYAN, GRAY, GRAY_DARK
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

DATA_PATH = Path("outputs/experiments/simulation/data/rank_detection.csv")
OUTPUT_DIR = get_output_dir()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    df = pd.read_csv(DATA_PATH)
    df_alpha1 = df[df["alpha"] == 1.0]

    true_ranks = sorted(df_alpha1["true_rank"].unique())
    snrs = [1.0, 0.6, 0.2]
    snr_colors = {1.0: TEAL, 0.6: CYAN, 0.2: GRAY_DARK}

    # === Individual violin plots for each SNR ===
    for snr in snrs:
        subset = df_alpha1[df_alpha1["snr"] == snr]

        fig, ax = create_figure("wide")

        data = [subset[subset["true_rank"] == r]["selected_rank"].values for r in true_ranks]

        parts = ax.violinplot(data, positions=true_ranks, widths=1.8,
                              showmeans=False, showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(snr_colors[snr])
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(1.5)

        ax.plot([0, 32], [0, 32], "--", color=GRAY, lw=1.5, zorder=0)

        ax.set_xlabel("True rank")
        ax.set_ylabel("Selected rank")
        ax.set_xlim(0, 32)
        ax.set_ylim(0, 38)
        ax.set_title(f"SNR = {snr}", fontsize=9)
        despine(ax)

        save_figure(fig, OUTPUT_DIR / f"violin_snr_{snr}.pdf")
        print(f"Saved: violin_snr_{snr}.pdf")

        # Stats
        acc = subset["is_correct"].mean()
        mae = subset["abs_error"].mean()
        print(f"  SNR={snr}: accuracy={acc:.1%}, MAE={mae:.2f}")

    # === Multi-panel figure ===
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5), sharey=True)

    for ax, snr in zip(axes, snrs):
        subset = df_alpha1[df_alpha1["snr"] == snr]
        data = [subset[subset["true_rank"] == r]["selected_rank"].values for r in true_ranks]

        parts = ax.violinplot(data, positions=true_ranks, widths=1.8,
                              showmeans=False, showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(snr_colors[snr])
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(1.5)

        ax.plot([0, 32], [0, 32], "--", color=GRAY, lw=1.5, zorder=0)

        ax.set_xlabel("True rank")
        ax.set_xlim(0, 32)
        ax.set_ylim(0, 38)

        acc = subset["is_correct"].mean()
        ax.set_title(f"SNR = {snr} (acc: {acc:.0%})", fontsize=9)
        despine(ax)

    axes[0].set_ylabel("Selected rank")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "violin_all_snrs.pdf", bbox_inches="tight")
    plt.close(fig)
    print("\nSaved: violin_all_snrs.pdf")

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
