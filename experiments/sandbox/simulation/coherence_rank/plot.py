"""Plot coherence kappa rank detection -- single detected-vs-true scatter."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

from src.colors import ROSE, TEAL, CYAN, SAND, GRAY, GRAY_LIGHT, GRAY_DARK, setup_style, CYCLE
from src.utils.figure_theme import despine

OUTPUT_DIR = Path("sandbox/simulation/coherence_rank/outputs/latest")
df = pd.read_csv(OUTPUT_DIR / "coherence_rank_detection.csv")


def main():
    setup_style()

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    true_ranks = sorted(df["true_rank"].unique())
    lo, hi = 0, max(true_ranks) + 5

    # Identity line
    ax.plot([lo, hi], [lo, hi], "-", color=GRAY_LIGHT, linewidth=0.8, zorder=0)

    # Encode alpha as color, SNR as marker
    alpha_to_color = {0.1: TEAL, 0.5: CYAN, 2.0: SAND, 5.0: ROSE}
    snr_to_marker = {1.0: "o", 0.7: "s", 0.4: "D"}

    rng = np.random.default_rng(7)

    for _, row in df.iterrows():
        jx = rng.uniform(-0.7, 0.7)
        jy = rng.uniform(-0.5, 0.5)
        ax.scatter(
            row["true_rank"] + jx,
            row["k_kappa"] + jy,
            s=14,
            color=alpha_to_color[row["alpha"]],
            marker=snr_to_marker[row["snr"]],
            alpha=0.6,
            edgecolors="none",
            zorder=3,
        )

    # Median per true_rank
    g = df.groupby("true_rank")["k_kappa"].median()
    ax.plot(g.index, g.values, "-", color=GRAY_DARK, linewidth=1, alpha=0.4, zorder=2)

    # Legends -- top-right corner, stacked
    alpha_handles = [
        Line2D([], [], marker="o", linestyle="none", color=alpha_to_color[a],
               markersize=4, markeredgecolor="none", label=f"{a}")
        for a in [0.1, 0.5, 2.0, 5.0]
    ]
    snr_handles = [
        Line2D([], [], marker=snr_to_marker[s], linestyle="none", color=GRAY_DARK,
               markersize=3.5, markeredgecolor="none", label=f"{s}")
        for s in [1.0, 0.7, 0.4]
    ]

    leg1 = ax.legend(
        handles=alpha_handles, title="$\\alpha$",
        loc="upper left", fontsize=6.5, title_fontsize=7,
        handletextpad=0.2, borderpad=0.3, labelspacing=0.2,
        bbox_to_anchor=(0.01, 0.99),
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=snr_handles, title="SNR",
        loc="lower right", fontsize=6.5, title_fontsize=7,
        handletextpad=0.2, borderpad=0.3, labelspacing=0.2,
        bbox_to_anchor=(0.99, 0.01),
    )

    # Accuracy annotation -- bottom left, out of the way
    acc = df["correct"].mean()
    n_total = len(df)
    n_correct = int(df["correct"].sum())
    ax.text(
        0.5, 0.02,
        f"{n_correct}/{n_total} correct ({acc:.0%})",
        transform=ax.transAxes, fontsize=6.5,
        ha="center", va="bottom", color=GRAY,
    )

    ax.set_xlabel("True rank $k$")
    ax.set_ylabel("Detected rank $\\hat{k}$")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    despine(ax)

    fig.savefig(OUTPUT_DIR / "rank_detection.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / "rank_detection.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
