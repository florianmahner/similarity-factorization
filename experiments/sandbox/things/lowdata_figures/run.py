"""Prototype lowdata comparison figures for the THINGS panel.

Creates several candidate designs for replacing the accuracy + rank subpanels.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.colors import ROSE, TEAL, INDIGO, GRAY, GRAY_LIGHT, GRAY_DARK, soft
from src.utils import get_output_dir
from src.utils.figure_theme import create_figure, despine, save_figure

OUTPUT_DIR = get_output_dir()

PCTS = [5, 10, 20, 50, 100]
NOISE_CEILING = 0.6694

# Definitive data (alpha=0, kappa ranks, train_90.txt, ReLU for VICE)
SRF = {
    "acc":  [0.5929, 0.6132, 0.6249, 0.6371, 0.6372],
    "rank": [5, 9, 13, 22, 24],
}
VICE = {
    "acc":  [0.5931, 0.6187, 0.6327, 0.6447, 0.6418],
    "rank": [10, 19, 31, 53, 67],
}
SPOSE_ACC = 0.6490
SPOSE_DIM = 66


def figure_bars_annotated():
    """Grouped bars with dimension annotations -- honest, clean."""
    fig, ax = create_figure("wide")
    x = np.arange(len(PCTS))
    w = 0.32

    bars_srf = ax.bar(x - w/2, [a * 100 for a in SRF["acc"]], w,
                       color=soft(TEAL), edgecolor=TEAL, linewidth=0.5, label="SRF", zorder=3)
    bars_vice = ax.bar(x + w/2, [a * 100 for a in VICE["acc"]], w,
                        color=soft(ROSE), edgecolor=ROSE, linewidth=0.5, label="VICE", zorder=3)

    for i, (bar, k) in enumerate(zip(bars_srf, SRF["rank"])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{k}", ha="center", va="bottom", fontsize=4.5, color=TEAL, fontweight="bold")

    for i, (bar, k) in enumerate(zip(bars_vice, VICE["rank"])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{k}", ha="center", va="bottom", fontsize=4.5, color=ROSE, fontweight="bold")

    ax.axhline(NOISE_CEILING * 100, color=GRAY_LIGHT, linestyle=":", linewidth=0.5, zorder=1)
    ax.text(len(PCTS) - 0.6, NOISE_CEILING * 100 + 0.2, "noise ceiling",
            fontsize=4.5, color=GRAY_LIGHT, va="bottom", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}%" for p in PCTS])
    ax.set_xlabel("Training data")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(55, 69)
    ax.legend(fontsize=5, loc="lower right")
    despine(ax)
    save_figure(fig, OUTPUT_DIR / "bars_annotated.png")
    plt.close()


def figure_scatter_efficiency():
    """Accuracy vs dimensions scatter -- shows efficiency tradeoff."""
    fig, ax = create_figure("wide")

    for i, pct in enumerate(PCTS):
        ax.scatter(SRF["rank"][i], SRF["acc"][i] * 100, color=TEAL, s=25, zorder=5,
                   edgecolors="white", linewidths=0.4)
        ax.scatter(VICE["rank"][i], VICE["acc"][i] * 100, color=ROSE, s=25, zorder=5,
                   edgecolors="white", linewidths=0.4)

        # Connect SRF-VICE pairs with a thin line
        ax.plot([SRF["rank"][i], VICE["rank"][i]],
                [SRF["acc"][i] * 100, VICE["acc"][i] * 100],
                color=GRAY_LIGHT, linewidth=0.4, zorder=2)

        # Label with percentage
        mid_x = (SRF["rank"][i] + VICE["rank"][i]) / 2
        mid_y = (SRF["acc"][i] + VICE["acc"][i]) / 2 * 100
        ax.text(mid_x, mid_y + 0.4, f"{pct}%", ha="center", va="bottom",
                fontsize=4, color=GRAY_DARK)

    ax.scatter([], [], color=TEAL, s=20, label="SRF")
    ax.scatter([], [], color=ROSE, s=20, label="VICE")

    ax.axhline(NOISE_CEILING * 100, color=GRAY_LIGHT, linestyle=":", linewidth=0.5)

    ax.set_xlabel("Number of dimensions")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(0, 75)
    ax.set_ylim(55, 69)
    ax.legend(fontsize=5, loc="lower right")
    despine(ax)
    save_figure(fig, OUTPUT_DIR / "scatter_efficiency.png")
    plt.close()


def figure_dual_axis():
    """Lines for accuracy + shaded area showing dimension ratio."""
    fig, ax1 = create_figure("wide")
    x = np.arange(len(PCTS))

    ax1.plot(x, [a * 100 for a in SRF["acc"]], color=TEAL, linewidth=0.8,
             marker="o", markersize=3.5, markeredgecolor="white", markeredgewidth=0.3,
             label="SRF", zorder=5)
    ax1.plot(x, [a * 100 for a in VICE["acc"]], color=ROSE, linewidth=0.8,
             marker="o", markersize=3.5, markeredgecolor="white", markeredgewidth=0.3,
             label="VICE", zorder=5)

    ax1.axhline(NOISE_CEILING * 100, color=GRAY_LIGHT, linestyle=":", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{p}%" for p in PCTS])
    ax1.set_xlabel("Training data")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(55, 69)
    ax1.legend(fontsize=5, loc="lower right")
    despine(ax1)

    # Dimension annotations along each line
    for i in range(len(PCTS)):
        ax1.annotate(f"k={SRF['rank'][i]}", (x[i], SRF["acc"][i]*100),
                     textcoords="offset points", xytext=(-8, -8),
                     fontsize=3.5, color=TEAL, ha="center")
        ax1.annotate(f"k={VICE['rank'][i]}", (x[i], VICE["acc"][i]*100),
                     textcoords="offset points", xytext=(8, 6),
                     fontsize=3.5, color=ROSE, ha="center")

    save_figure(fig, OUTPUT_DIR / "dual_annotated.png")
    plt.close()


def figure_bar_with_dim_panel():
    """Two-row figure: top=accuracy bars, bottom=dimension bars (inverted)."""
    fig, (ax_acc, ax_dim) = plt.subplots(2, 1, figsize=(3.4, 3.0),
                                          height_ratios=[3, 1.5], sharex=True)
    x = np.arange(len(PCTS))
    w = 0.32

    # Top: accuracy
    ax_acc.bar(x - w/2, [a * 100 for a in SRF["acc"]], w,
               color=soft(TEAL), edgecolor=TEAL, linewidth=0.5, label="SRF", zorder=3)
    ax_acc.bar(x + w/2, [a * 100 for a in VICE["acc"]], w,
               color=soft(ROSE), edgecolor=ROSE, linewidth=0.5, label="VICE", zorder=3)
    ax_acc.axhline(NOISE_CEILING * 100, color=GRAY_LIGHT, linestyle=":", linewidth=0.5)
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(55, 69)
    ax_acc.legend(fontsize=5, loc="lower right")
    despine(ax_acc)

    # Bottom: dimensions (shows SRF is much more compact)
    ax_dim.bar(x - w/2, SRF["rank"], w,
               color=soft(TEAL), edgecolor=TEAL, linewidth=0.5, zorder=3)
    ax_dim.bar(x + w/2, VICE["rank"], w,
               color=soft(ROSE), edgecolor=ROSE, linewidth=0.5, zorder=3)

    ax_dim.axhline(SPOSE_DIM, color=GRAY, linestyle=(0, (4, 3)), linewidth=0.5)
    ax_dim.text(len(PCTS) - 0.6, SPOSE_DIM + 1, "SPoSE", fontsize=4.5, color=GRAY, ha="right")

    ax_dim.set_xticks(x)
    ax_dim.set_xticklabels([f"{p}%" for p in PCTS])
    ax_dim.set_xlabel("Training data")
    ax_dim.set_ylabel("Dimensions")
    ax_dim.set_ylim(0, 80)
    despine(ax_dim)

    fig.tight_layout(h_pad=0.5)
    fig.savefig(OUTPUT_DIR / "bar_with_dim_panel.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def figure_lollipop():
    """Lollipop chart: accuracy on y, data fraction on x, stem length = dimensions."""
    fig, ax = create_figure("wide")
    x = np.arange(len(PCTS))

    for i, pct in enumerate(PCTS):
        # SRF: circle at accuracy, stem length proportional to dims
        ax.plot([x[i] - 0.15, x[i] - 0.15], [55, SRF["acc"][i] * 100],
                color=TEAL, linewidth=1.5, solid_capstyle="round", zorder=3, alpha=0.6)
        ax.scatter(x[i] - 0.15, SRF["acc"][i] * 100, color=TEAL, s=SRF["rank"][i] * 2.5,
                   zorder=5, edgecolors="white", linewidths=0.3)

        # VICE
        ax.plot([x[i] + 0.15, x[i] + 0.15], [55, VICE["acc"][i] * 100],
                color=ROSE, linewidth=1.5, solid_capstyle="round", zorder=3, alpha=0.6)
        ax.scatter(x[i] + 0.15, VICE["acc"][i] * 100, color=ROSE, s=VICE["rank"][i] * 2.5,
                   zorder=5, edgecolors="white", linewidths=0.3)

    ax.scatter([], [], color=TEAL, s=30, label="SRF")
    ax.scatter([], [], color=ROSE, s=30, label="VICE")
    ax.plot([], [], color=GRAY, linewidth=0.5, label="circle size = dims")

    ax.axhline(NOISE_CEILING * 100, color=GRAY_LIGHT, linestyle=":", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}%" for p in PCTS])
    ax.set_xlabel("Training data")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(55, 69)
    ax.legend(fontsize=4.5, loc="lower right")
    despine(ax)
    save_figure(fig, OUTPUT_DIR / "lollipop.png")
    plt.close()


def main():
    figure_bars_annotated()
    figure_scatter_efficiency()
    figure_dual_annotated()
    figure_bar_with_dim_panel()
    figure_lollipop()
    print(f"Saved 5 figures to {OUTPUT_DIR}")


def figure_dual_annotated():
    """Renamed wrapper."""
    figure_dual_axis()


if __name__ == "__main__":
    main()
