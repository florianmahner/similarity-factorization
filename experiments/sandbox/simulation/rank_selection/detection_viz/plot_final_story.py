"""Final rank detection plot: works on distributed clusters, robust to noise.

Story: Rank detection works well for distributed clusters (alpha=1.0)
and is robust across noise levels (SNR 0.2 to 1.0).

Usage:
    poetry run python sandbox/rank_detection_viz/plot_final_story.py
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.colors import TEAL, GRAY, GRAY_LIGHT, GRAY_PALE
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

DATA_PATH = Path("outputs/experiments/simulation/data/rank_detection.csv")
OUTPUT_DIR = get_output_dir()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    df = pd.read_csv(DATA_PATH)
    df = df[df["true_rank"] <= 14]  # Focus on reliable regime

    # === Main figure: Accuracy by SNR for different alpha ===
    fig, ax = create_figure("single")

    # Show alpha=1.0 prominently, others as context
    alpha_styles = {
        0.1: {"color": GRAY_LIGHT, "lw": 1.0, "marker": "s", "ms": 3},
        1.0: {"color": TEAL, "lw": 2.0, "marker": "o", "ms": 5},
        5.0: {"color": GRAY_LIGHT, "lw": 1.0, "marker": "^", "ms": 3},
        10.0: {"color": GRAY_LIGHT, "lw": 1.0, "marker": "d", "ms": 3},
    }

    for alpha in [0.1, 5.0, 10.0, 1.0]:  # Plot 1.0 last to be on top
        subset = df[df["alpha"] == alpha]
        agg = subset.groupby("snr").agg(
            acc=("is_correct", "mean"),
            sem=("is_correct", "sem"),
        ).reset_index()

        style = alpha_styles[alpha]
        label = f"α = {alpha}" if alpha == 1.0 else None
        ax.plot(agg["snr"], agg["acc"] * 100,
                marker=style["marker"], markersize=style["ms"],
                color=style["color"], lw=style["lw"], label=label)

    ax.axhline(90, color=GRAY_PALE, linestyle="-", lw=8, zorder=0)
    ax.set_xlabel("Signal-to-noise ratio")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(0.1, 1.1)
    ax.set_ylim(0, 105)

    # Add annotation for alpha=1.0
    ax.annotate("α = 1.0\n(distributed)", xy=(0.6, 96), fontsize=7,
                color=TEAL, ha="center")

    despine(ax)

    save_figure(fig, OUTPUT_DIR / "rank_detection_accuracy.pdf")
    print("Saved: rank_detection_accuracy.pdf")

    # === Supplementary: violin plot for alpha=1.0 ===
    df_a1 = df[df["alpha"] == 1.0]
    true_ranks = sorted(df_a1["true_rank"].unique())

    fig, ax = create_figure("wide")

    data = [df_a1[df_a1["true_rank"] == r]["selected_rank"].values for r in true_ranks]

    parts = ax.violinplot(data, positions=true_ranks, widths=1.5,
                          showmeans=False, showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(TEAL)
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(1.5)

    ax.plot([0, 16], [0, 16], "--", color=GRAY, lw=1.5, zorder=0)

    ax.set_xlabel("True rank")
    ax.set_ylabel("Selected rank")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 18)
    despine(ax)

    save_figure(fig, OUTPUT_DIR / "rank_detection_violin.pdf")
    print("Saved: rank_detection_violin.pdf")

    # === Summary ===
    print()
    print("="*50)
    print("Summary (k ≤ 14, alpha=1.0):")
    print("="*50)
    print(f"Accuracy: {df_a1['is_correct'].mean():.1%}")
    print(f"MAE: {df_a1['abs_error'].mean():.2f}")
    print()
    print("By SNR:")
    print(df_a1.groupby("snr")["is_correct"].mean().round(2))

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
