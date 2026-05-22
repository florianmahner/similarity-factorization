"""Combined bias plot: new full-range data + stable experiment effects.

Usage:
    poetry run python sandbox/rank_detection_viz/plot_combined.py
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

# Data paths
NEW_DATA = Path("sandbox/rank_detection_viz/outputs/251217/111105/rank_detection_obs_dof_range.csv")
STABLE_DATA = Path("outputs/experiments/simulation/data/rank_detection.csv")
OUTPUT_DIR = get_output_dir()

N_STABLE = 300  # n from stable experiment


def compute_obs_dof(n: int, k: int) -> float:
    return (n * (n - 1) / 2) / (n * k)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Load new full-range data
    df_new = pd.read_csv(NEW_DATA)
    agg_new = df_new.groupby("obs_per_dof").agg(
        bias=("signed_error", "mean"),
        sem=("signed_error", "sem"),
    ).reset_index().sort_values("obs_per_dof")

    # Load stable data
    df_stable = pd.read_csv(STABLE_DATA)
    df_stable["obs_per_dof"] = df_stable["true_rank"].apply(lambda k: compute_obs_dof(N_STABLE, k))

    # === Main figure: Two-panel ===
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2))

    # --- Left: Full range (new data) ---
    ax = axes[0]

    ax.plot(agg_new["obs_per_dof"], agg_new["bias"], "o-",
            color=TEAL, markersize=4, lw=1.5)
    ax.fill_between(agg_new["obs_per_dof"],
                    agg_new["bias"] - agg_new["sem"],
                    agg_new["bias"] + agg_new["sem"],
                    color=TEAL, alpha=0.2, linewidth=0)

    ax.axvspan(0.5, 10, color=GRAY_PALE, zorder=0)
    ax.axvline(10, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=1)
    ax.axhline(0, color=GRAY, linestyle="-", linewidth=0.8, zorder=0)

    ax.set_xscale("log")
    ax.set_xlabel("Samples per degree of freedom")
    ax.set_ylabel("Bias (selected − true rank)")
    ax.set_xlim(0.8, 60)
    ax.set_ylim(-8, 2)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
    ax.set_title("n = 200, α = 1, SNR = 1", fontsize=8)
    despine(ax)

    # --- Right: SNR effect (stable data) ---
    ax = axes[1]
    df_alpha1 = df_stable[df_stable["alpha"] == 1.0]

    snr_colors = {0.2: GRAY_DARK, 0.4: CYAN, 0.6: SAND, 0.8: TEAL, 1.0: ROSE}

    for snr in [0.2, 0.6, 1.0]:
        subset = df_alpha1[df_alpha1["snr"] == snr]
        agg = subset.groupby("obs_per_dof").agg(
            bias=("signed_error", "mean"),
            sem=("signed_error", "sem"),
        ).reset_index().sort_values("obs_per_dof")

        ax.plot(agg["obs_per_dof"], agg["bias"], "o-",
                color=snr_colors[snr], label=f"SNR = {snr}",
                markersize=3, lw=1.2)
        ax.fill_between(agg["obs_per_dof"],
                        agg["bias"] - agg["sem"],
                        agg["bias"] + agg["sem"],
                        color=snr_colors[snr], alpha=0.12, linewidth=0)

    ax.axvspan(1, 10, color=GRAY_PALE, zorder=0)
    ax.axvline(10, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=1)
    ax.axhline(0, color=GRAY, linestyle="-", linewidth=0.8, zorder=0)

    ax.set_xscale("log")
    ax.set_xlabel("Samples per degree of freedom")
    ax.set_xlim(4, 100)
    ax.set_ylim(-3, 4)
    ax.xaxis.set_major_locator(FixedLocator([5, 10, 20, 50]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["5", "10", "20", "50"])
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    ax.set_title("n = 300, α = 1", fontsize=8)
    despine(ax)

    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR / "bias_combined.pdf", tight=False)
    print("Saved: bias_combined.pdf")

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
