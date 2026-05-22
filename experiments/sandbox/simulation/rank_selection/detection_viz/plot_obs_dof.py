"""Plot rank detection accuracy vs obs/dof ratio.

Analogous to imputation plot but for rank detection.

Usage:
    poetry run python sandbox/rank_detection_viz/plot_obs_dof.py
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator

from src.colors import TEAL, CYAN, ROSE, SAND, GRAY, GRAY_LIGHT, GRAY_DARK
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

DATA_PATH = Path("outputs/experiments/simulation/data/rank_detection.csv")
OUTPUT_DIR = get_output_dir()

N_SAMPLES = 300  # From stable experiment


def compute_obs_dof(n: int, k: int) -> float:
    """Compute observations per degree of freedom."""
    obs = n * (n - 1) / 2
    dof = n * k
    return obs / dof


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    df = pd.read_csv(DATA_PATH)
    df["obs_per_dof"] = df["true_rank"].apply(lambda k: compute_obs_dof(N_SAMPLES, k))

    # === Plot 1: Accuracy vs obs/dof by alpha (SNR=1.0) ===
    df_snr1 = df[df["snr"] == 1.0].copy()

    fig, ax = create_figure("single")

    alpha_colors = {0.1: CYAN, 1.0: TEAL, 5.0: SAND, 10.0: ROSE}

    for alpha in sorted(df_snr1["alpha"].unique()):
        subset = df_snr1[df_snr1["alpha"] == alpha]
        agg = subset.groupby("obs_per_dof").agg(
            acc=("is_correct", "mean"),
            sem=("is_correct", "sem"),
        ).reset_index().sort_values("obs_per_dof")

        ax.plot(agg["obs_per_dof"], agg["acc"] * 100, "o-",
                color=alpha_colors[alpha], label=f"α={alpha}", markersize=4, lw=1.5)
        ax.fill_between(agg["obs_per_dof"],
                        (agg["acc"] - agg["sem"]) * 100,
                        (agg["acc"] + agg["sem"]) * 100,
                        color=alpha_colors[alpha], alpha=0.15, linewidth=0)

    ax.axhline(50, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("Samples per degree of freedom")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(4, 100)
    ax.set_ylim(0, 105)
    ax.xaxis.set_major_locator(FixedLocator([5, 10, 20, 50]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["5", "10", "20", "50"])
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    despine(ax)

    save_figure(fig, OUTPUT_DIR / "accuracy_vs_obs_dof_by_alpha.pdf")
    print("Saved: accuracy_vs_obs_dof_by_alpha.pdf")

    # === Plot 2: Accuracy vs obs/dof by SNR (alpha=1.0) ===
    df_alpha1 = df[df["alpha"] == 1.0].copy()

    fig, ax = create_figure("single")

    snr_colors = {0.2: GRAY, 0.4: CYAN, 0.6: SAND, 0.8: TEAL, 1.0: ROSE}

    for snr in sorted(df_alpha1["snr"].unique()):
        subset = df_alpha1[df_alpha1["snr"] == snr]
        agg = subset.groupby("obs_per_dof").agg(
            acc=("is_correct", "mean"),
            sem=("is_correct", "sem"),
        ).reset_index().sort_values("obs_per_dof")

        ax.plot(agg["obs_per_dof"], agg["acc"] * 100, "o-",
                color=snr_colors[snr], label=f"SNR={snr}", markersize=4, lw=1.5)

    ax.axhline(50, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("Samples per degree of freedom")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(4, 100)
    ax.set_ylim(0, 105)
    ax.xaxis.set_major_locator(FixedLocator([5, 10, 20, 50]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["5", "10", "20", "50"])
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    despine(ax)

    save_figure(fig, OUTPUT_DIR / "accuracy_vs_obs_dof_by_snr.pdf")
    print("Saved: accuracy_vs_obs_dof_by_snr.pdf")

    # === Plot 3: Bias vs obs/dof by alpha (SNR=1.0) ===
    fig, ax = create_figure("single")

    for alpha in sorted(df_snr1["alpha"].unique()):
        subset = df_snr1[df_snr1["alpha"] == alpha]
        agg = subset.groupby("obs_per_dof").agg(
            bias=("signed_error", "mean"),
            sem=("signed_error", "sem"),
        ).reset_index().sort_values("obs_per_dof")

        ax.plot(agg["obs_per_dof"], agg["bias"], "o-",
                color=alpha_colors[alpha], label=f"α={alpha}", markersize=4, lw=1.5)
        ax.fill_between(agg["obs_per_dof"],
                        agg["bias"] - agg["sem"],
                        agg["bias"] + agg["sem"],
                        color=alpha_colors[alpha], alpha=0.15, linewidth=0)

    ax.axhline(0, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("Samples per degree of freedom")
    ax.set_ylabel("Bias (selected − true)")
    ax.set_xlim(4, 100)
    ax.xaxis.set_major_locator(FixedLocator([5, 10, 20, 50]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["5", "10", "20", "50"])
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    despine(ax)

    save_figure(fig, OUTPUT_DIR / "bias_vs_obs_dof_by_alpha.pdf")
    print("Saved: bias_vs_obs_dof_by_alpha.pdf")

    # === Plot 4: Score gap vs obs/dof (SNR=1.0, alpha=1.0) ===
    df_best = df[(df["alpha"] == 1.0) & (df["snr"] == 1.0)].copy()
    df_best["rel_gap"] = df_best["score_gap"] / df_best["best_score"] * 100

    fig, ax = create_figure("single")

    agg = df_best.groupby("obs_per_dof").agg(
        gap=("rel_gap", "mean"),
        sem=("rel_gap", "sem"),
    ).reset_index().sort_values("obs_per_dof")

    ax.plot(agg["obs_per_dof"], agg["gap"], "o-",
            color=TEAL, markersize=5, lw=2)
    ax.fill_between(agg["obs_per_dof"],
                    agg["gap"] - agg["sem"],
                    agg["gap"] + agg["sem"],
                    color=TEAL, alpha=0.2, linewidth=0)

    # Threshold line
    ax.axhline(10, color=ROSE, linestyle="--", linewidth=1, zorder=0)
    ax.text(80, 12, "10% threshold", fontsize=7, color=ROSE, ha="right")

    ax.set_xscale("log")
    ax.set_xlabel("Samples per degree of freedom")
    ax.set_ylabel("Relative score gap (%)")
    ax.set_xlim(4, 100)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(FixedLocator([5, 10, 20, 50]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["5", "10", "20", "50"])
    despine(ax)

    save_figure(fig, OUTPUT_DIR / "score_gap_vs_obs_dof.pdf")
    print("Saved: score_gap_vs_obs_dof.pdf")

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
