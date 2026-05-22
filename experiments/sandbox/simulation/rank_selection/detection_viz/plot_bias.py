"""Publication-quality bias vs obs/dof plot for rank detection.

Two-panel figure: alpha effect (left) and SNR effect (right).

Usage:
    poetry run python sandbox/rank_detection_viz/plot_bias.py
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator

from src.colors import TEAL, CYAN, ROSE, SAND, GRAY, GRAY_LIGHT, GRAY_DARK, GRAY_PALE
from src.utils.figure_theme import despine, save_figure
from src.utils import get_output_dir

DATA_PATH = Path("outputs/experiments/simulation/data/rank_detection.csv")
OUTPUT_DIR = get_output_dir()

N_SAMPLES = 300


def compute_obs_dof(n: int, k: int) -> float:
    obs = n * (n - 1) / 2
    dof = n * k
    return obs / dof


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    df = pd.read_csv(DATA_PATH)
    df["obs_per_dof"] = df["true_rank"].apply(lambda k: compute_obs_dof(N_SAMPLES, k))

    # === Two-panel figure ===
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2), sharey=True)

    # --- Left panel: Alpha effect (SNR=1.0) ---
    ax = axes[0]
    df_snr1 = df[df["snr"] == 1.0]

    alpha_colors = {0.1: CYAN, 1.0: TEAL, 5.0: SAND, 10.0: ROSE}
    alpha_labels = {0.1: "0.1", 1.0: "1.0", 5.0: "5.0", 10.0: "10.0"}

    for alpha in [0.1, 1.0, 5.0, 10.0]:
        subset = df_snr1[df_snr1["alpha"] == alpha]
        agg = subset.groupby("obs_per_dof").agg(
            bias=("signed_error", "mean"),
            sem=("signed_error", "sem"),
        ).reset_index().sort_values("obs_per_dof")

        ax.plot(agg["obs_per_dof"], agg["bias"], "o-",
                color=alpha_colors[alpha], label=alpha_labels[alpha],
                markersize=3, lw=1.2)
        ax.fill_between(agg["obs_per_dof"],
                        agg["bias"] - agg["sem"],
                        agg["bias"] + agg["sem"],
                        color=alpha_colors[alpha], alpha=0.12, linewidth=0)

    # Shaded "hard" region (obs/dof < 10)
    ax.axvspan(1, 10, color=GRAY_PALE, zorder=0)
    ax.axvline(10, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=1)

    ax.axhline(0, color=GRAY, linestyle="-", linewidth=0.8, zorder=0)

    ax.set_xscale("log")
    ax.set_xlabel("Samples per degree of freedom")
    ax.set_ylabel("Bias (selected − true rank)")
    ax.set_xlim(4, 100)
    ax.set_ylim(-10, 4)
    ax.xaxis.set_major_locator(FixedLocator([5, 10, 20, 50]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["5", "10", "20", "50"])

    ax.legend(frameon=False, fontsize=6, loc="lower left", title="α", title_fontsize=7)
    ax.set_title("SNR = 1.0", fontsize=8)
    despine(ax)

    # --- Right panel: SNR effect (alpha=1.0) ---
    ax = axes[1]
    df_alpha1 = df[df["alpha"] == 1.0]

    snr_colors = {0.2: GRAY_DARK, 0.4: CYAN, 0.6: SAND, 0.8: TEAL, 1.0: ROSE}

    for snr in [0.2, 0.4, 0.6, 0.8, 1.0]:
        subset = df_alpha1[df_alpha1["snr"] == snr]
        agg = subset.groupby("obs_per_dof").agg(
            bias=("signed_error", "mean"),
            sem=("signed_error", "sem"),
        ).reset_index().sort_values("obs_per_dof")

        ax.plot(agg["obs_per_dof"], agg["bias"], "o-",
                color=snr_colors[snr], label=f"{snr}",
                markersize=3, lw=1.2)
        ax.fill_between(agg["obs_per_dof"],
                        agg["bias"] - agg["sem"],
                        agg["bias"] + agg["sem"],
                        color=snr_colors[snr], alpha=0.12, linewidth=0)

    # Shaded "hard" region (obs/dof < 10)
    ax.axvspan(1, 10, color=GRAY_PALE, zorder=0)
    ax.axvline(10, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=1)

    ax.axhline(0, color=GRAY, linestyle="-", linewidth=0.8, zorder=0)

    ax.set_xscale("log")
    ax.set_xlabel("Samples per degree of freedom")
    ax.set_xlim(4, 100)
    ax.xaxis.set_major_locator(FixedLocator([5, 10, 20, 50]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["5", "10", "20", "50"])

    ax.legend(frameon=False, fontsize=6, loc="lower left", title="SNR", title_fontsize=7)
    ax.set_title("α = 1.0", fontsize=8)
    despine(ax)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "bias_two_panel.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: bias_two_panel.pdf")

    # === Single combined plot (alternative) ===
    # Show alpha=1.0 with all SNRs - the "reference" case
    fig, ax = plt.subplots(figsize=(3.2, 2.4))

    for snr in [0.2, 0.6, 1.0]:
        subset = df_alpha1[df_alpha1["snr"] == snr]
        agg = subset.groupby("obs_per_dof").agg(
            bias=("signed_error", "mean"),
            sem=("signed_error", "sem"),
        ).reset_index().sort_values("obs_per_dof")

        lw = 1.8 if snr == 1.0 else 1.2
        ax.plot(agg["obs_per_dof"], agg["bias"], "o-",
                color=snr_colors[snr], label=f"SNR = {snr}",
                markersize=4, lw=lw)
        ax.fill_between(agg["obs_per_dof"],
                        agg["bias"] - agg["sem"],
                        agg["bias"] + agg["sem"],
                        color=snr_colors[snr], alpha=0.15, linewidth=0)

    # Shaded region (obs/dof < 10)
    ax.axvspan(1, 10, color=GRAY_PALE, zorder=0)
    ax.axvline(10, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=1)

    ax.axhline(0, color=GRAY, linestyle="-", linewidth=0.8, zorder=0)

    ax.set_xscale("log")
    ax.set_xlabel("Samples per degree of freedom")
    ax.set_ylabel("Bias (selected − true rank)")
    ax.set_xlim(4, 100)
    ax.set_ylim(-2, 4)
    ax.xaxis.set_major_locator(FixedLocator([5, 10, 20, 50]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["5", "10", "20", "50"])

    ax.legend(frameon=False, fontsize=7, loc="upper right")
    despine(ax)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "bias_single.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: bias_single.pdf")

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
