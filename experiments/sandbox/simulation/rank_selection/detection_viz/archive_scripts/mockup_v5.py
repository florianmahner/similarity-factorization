"""Single grouped boxplot for rank detection.

Usage:
    poetry run python sandbox/rank_detection_viz/mockup_v5.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.colors import ROSE, TEAL, CYAN, GRAY, GRAY_LIGHT
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

PROJECT_ROOT = Path(__file__).parents[2]
DATA_PATH = PROJECT_ROOT / "outputs/experiments/simulation/data/rank_detection.csv"
OUTPUT_DIR = get_output_dir()

ALPHAS = [0.1, 1.0, 5.0]
ALPHA_COLORS = {0.1: ROSE, 1.0: TEAL, 5.0: CYAN}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df[df["alpha"].isin(ALPHAS)].copy()
    return df


def plot_grouped_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
    """Single plot with grouped boxplots - 3 boxes per true rank."""
    fig, ax = create_figure("wide")

    ranks = sorted(df["true_rank"].unique())
    n_alphas = len(ALPHAS)
    width = 0.55
    offsets = np.array([-1, 0, 1]) * width

    for i, alpha in enumerate(ALPHAS):
        subset = df[df["alpha"] == alpha]
        positions = [r + offsets[i] for r in ranks]
        data = [subset[subset["true_rank"] == r]["selected_rank"].values for r in ranks]

        bp = ax.boxplot(data, positions=positions, widths=width * 0.85,
                        patch_artist=True, showfliers=False, whis=[10, 90])

        for patch in bp["boxes"]:
            patch.set_facecolor(ALPHA_COLORS[alpha])
            patch.set_alpha(0.7)
            patch.set_edgecolor(ALPHA_COLORS[alpha])
        for median in bp["medians"]:
            median.set_color("white")
            median.set_linewidth(1.2)
        for whisker in bp["whiskers"]:
            whisker.set_color(ALPHA_COLORS[alpha])
            whisker.set_alpha(0.7)
        for cap in bp["caps"]:
            cap.set_color(ALPHA_COLORS[alpha])
            cap.set_alpha(0.7)

    # Identity line
    ax.plot([0, 32], [0, 32], "--", color=GRAY_LIGHT, lw=1.5, zorder=0)
    ax.text(29, 27, "identity", fontsize=7, color=GRAY, ha="right")

    ax.set_xlabel("True rank")
    ax.set_ylabel("Selected rank")
    ax.set_xlim(0, 32)
    ax.set_ylim(-2, 42)
    ax.set_xticks(ranks[::2])

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, fc=ALPHA_COLORS[a], alpha=0.7) for a in ALPHAS]
    ax.legend(handles, [f"{a}" for a in ALPHAS], title=r"$\alpha$",
              frameon=False, loc="upper left", fontsize=7, title_fontsize=8)

    despine(ax)
    save_figure(fig, output_dir / "v5_grouped_boxplot.pdf")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data for alpha = 0.1, 1.0, 5.0...")
    df = load_data()

    print("\nGenerating grouped boxplot...")
    plot_grouped_boxplot(df, OUTPUT_DIR)
    print("  v5_grouped_boxplot.pdf")

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
