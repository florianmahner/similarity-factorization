"""Draft panel: triplet accuracy (%) vs rank curve.

Shows SRF accuracy as a function of rank with SPoSE horizontal line,
noise ceiling, chance level, and kappa k* vertical marker.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import get_output_dir
from src.colors import GRAY, GRAY_LIGHT, PURPLE, TEAL, setup_style
from src.utils.figure_theme import create_figure, despine, save_figure

OUTPUT_DIR = get_output_dir()

RESULTS = (
    Path(__file__).resolve().parents[4]
    / "experiments"
    / "analyses"
    / "things_behavior"
    / "rank_sweep"
    / "outputs"
    / "results.csv"
)

CHANCE = 100 / 3
NOISE_CEILING = 67.22
KAPPA_K_STAR = 24


def main():
    setup_style()
    df = pd.read_csv(RESULTS)

    srf = df[(df["model"] == "SRF") & (df["alpha"] == 0)]
    summary = srf.groupby("rank")["val_acc"].agg(["mean", "std"])
    ranks = summary.index.values
    means = summary["mean"].values * 100
    stds = summary["std"].values * 100

    spose_acc = df[df["model"] == "SPoSE"]["val_acc"].values[0] * 100

    fig, ax = create_figure("single")

    ax.fill_between(ranks, means - stds, means + stds, color=TEAL, alpha=0.2)
    ax.plot(ranks, means, color=TEAL, linewidth=1.2, zorder=3)

    ax.axhline(spose_acc, color=PURPLE, linestyle="--", linewidth=0.8, zorder=2)
    ax.text(ranks[-1] + 1, spose_acc, "SPoSE", fontsize=5, color=PURPLE,
            ha="left", va="center")

    ax.axhline(NOISE_CEILING, color=GRAY, linestyle=":", linewidth=0.6, zorder=1)
    ax.text(ranks[-1] + 1, NOISE_CEILING, "ceiling", fontsize=5, color=GRAY,
            ha="left", va="center")

    ax.axvline(KAPPA_K_STAR, color=GRAY, linestyle="--", linewidth=0.6, zorder=1)
    ax.text(KAPPA_K_STAR + 1, means.min() - 0.3, "k*", fontsize=5, color=GRAY,
            ha="left", va="top")

    ax.set_xlabel("Rank (k)")
    ax.set_ylabel("Triplet accuracy (%)")
    ax.set_xlim(0, ranks[-1] + 8)
    ax.set_ylim(55, NOISE_CEILING + 1)
    despine(ax)

    fig.savefig(OUTPUT_DIR / "rank_curve.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUTPUT_DIR / 'rank_curve.png'}")


if __name__ == "__main__":
    main()
