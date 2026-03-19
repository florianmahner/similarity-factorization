"""Plot bias vs obs/dof for the full range simulation.

Usage:
    poetry run python sandbox/rank_detection_viz/plot_obs_dof_range.py
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator

from src.colors import TEAL, GRAY, GRAY_LIGHT, GRAY_PALE
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

DATA_PATH = Path("sandbox/rank_detection_viz/outputs/251217/111105/rank_detection_obs_dof_range.csv")
OUTPUT_DIR = get_output_dir()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    df = pd.read_csv(DATA_PATH)

    # Aggregate by obs_per_dof
    agg = df.groupby("obs_per_dof").agg(
        true_rank=("true_rank", "first"),
        bias=("signed_error", "mean"),
        sem=("signed_error", "sem"),
        mae=("abs_error", "mean"),
        acc=("is_correct", "mean"),
    ).reset_index().sort_values("obs_per_dof")

    print("Summary by obs/dof:")
    print(agg.round(2))

    # === Main plot: Bias vs obs/dof ===
    fig, ax = create_figure("single")

    ax.plot(agg["obs_per_dof"], agg["bias"], "o-",
            color=TEAL, markersize=5, lw=1.8)
    ax.fill_between(agg["obs_per_dof"],
                    agg["bias"] - agg["sem"],
                    agg["bias"] + agg["sem"],
                    color=TEAL, alpha=0.2, linewidth=0)

    # Shaded "hard" region (obs/dof < 10)
    ax.axvspan(0.5, 10, color=GRAY_PALE, zorder=0)
    ax.axvline(10, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=1)

    # Zero line
    ax.axhline(0, color=GRAY, linestyle="-", linewidth=0.8, zorder=0)

    ax.set_xscale("log")
    ax.set_xlabel("Samples per degree of freedom")
    ax.set_ylabel("Bias (selected − true rank)")
    ax.set_xlim(0.8, 60)
    ax.set_ylim(-8, 2)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
    despine(ax)

    save_figure(fig, OUTPUT_DIR / "bias_vs_obs_dof_full_range.pdf")
    print("\nSaved: bias_vs_obs_dof_full_range.pdf")

    # === Accuracy plot ===
    fig, ax = create_figure("single")

    ax.plot(agg["obs_per_dof"], agg["acc"] * 100, "o-",
            color=TEAL, markersize=5, lw=1.8)

    # Shaded "hard" region
    ax.axvspan(0.5, 10, color=GRAY_PALE, zorder=0)
    ax.axvline(10, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=1)

    # 50% line
    ax.axhline(50, color=GRAY_LIGHT, linestyle="--", linewidth=0.8, zorder=0)

    ax.set_xscale("log")
    ax.set_xlabel("Samples per degree of freedom")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(0.8, 60)
    ax.set_ylim(-5, 105)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
    despine(ax)

    save_figure(fig, OUTPUT_DIR / "accuracy_vs_obs_dof_full_range.pdf")
    print("Saved: accuracy_vs_obs_dof_full_range.pdf")

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
