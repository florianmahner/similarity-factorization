"""Plot comparison of p methods for rank detection.

Usage:
    poetry run python sandbox/rank_detection_viz/plot_p_comparison.py
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator

from src.colors import TEAL, CYAN, ROSE, GRAY, GRAY_LIGHT, GRAY_PALE
from src.utils.figure_theme import despine, save_figure
from src.utils import get_output_dir

DATA_PATH = Path("sandbox/rank_detection_viz/outputs/251217/131405/p_comparison.csv")
OUTPUT_DIR = get_output_dir()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    df = pd.read_csv(DATA_PATH)

    # Colors for each method
    method_colors = {
        "bounds": TEAL,
        "0.5": CYAN,
        "0.8": ROSE,
    }
    method_labels = {
        "bounds": "Bounds-estimated p",
        "0.5": "Fixed p = 0.5",
        "0.8": "Fixed p = 0.8",
    }

    # === Plot 1: Accuracy comparison ===
    fig, ax = plt.subplots(figsize=(4, 3))

    for method in ["bounds", "0.5", "0.8"]:
        subset = df[df["p_method"] == method]
        agg = subset.groupby("obs_per_dof").agg(
            acc=("is_correct", "mean"),
            sem=("is_correct", "sem"),
        ).reset_index().sort_values("obs_per_dof")

        ax.plot(agg["obs_per_dof"], agg["acc"] * 100, "o-",
                color=method_colors[method], label=method_labels[method],
                markersize=4, lw=1.5)
        ax.fill_between(agg["obs_per_dof"],
                        (agg["acc"] - agg["sem"]) * 100,
                        (agg["acc"] + agg["sem"]) * 100,
                        color=method_colors[method], alpha=0.15, linewidth=0)

    ax.axhline(50, color=GRAY_LIGHT, linestyle="--", lw=0.8)
    ax.axvspan(1, 10, color=GRAY_PALE, zorder=0)
    ax.axvline(10, color=GRAY_LIGHT, linestyle="--", lw=0.8)

    ax.set_xscale("log")
    ax.set_xlabel("Observations per degree of freedom")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(1.5, 40)
    ax.set_ylim(-5, 105)
    ax.xaxis.set_major_locator(FixedLocator([2, 5, 10, 20]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["2", "5", "10", "20"])
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    despine(ax)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "accuracy_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: accuracy_comparison.pdf")

    # === Plot 2: Bias comparison ===
    fig, ax = plt.subplots(figsize=(4, 3))

    for method in ["bounds", "0.5", "0.8"]:
        subset = df[df["p_method"] == method]
        agg = subset.groupby("obs_per_dof").agg(
            bias=("signed_error", "mean"),
            sem=("signed_error", "sem"),
        ).reset_index().sort_values("obs_per_dof")

        ax.plot(agg["obs_per_dof"], agg["bias"], "o-",
                color=method_colors[method], label=method_labels[method],
                markersize=4, lw=1.5)
        ax.fill_between(agg["obs_per_dof"],
                        agg["bias"] - agg["sem"],
                        agg["bias"] + agg["sem"],
                        color=method_colors[method], alpha=0.15, linewidth=0)

    ax.axhline(0, color=GRAY, linestyle="-", lw=0.8)
    ax.axvspan(1, 10, color=GRAY_PALE, zorder=0)
    ax.axvline(10, color=GRAY_LIGHT, linestyle="--", lw=0.8)

    ax.set_xscale("log")
    ax.set_xlabel("Observations per degree of freedom")
    ax.set_ylabel("Bias (selected − true rank)")
    ax.set_xlim(1.5, 40)
    ax.xaxis.set_major_locator(FixedLocator([2, 5, 10, 20]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["2", "5", "10", "20"])
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    despine(ax)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "bias_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: bias_comparison.pdf")

    # === Plot 3: Two-panel summary ===
    fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))

    # Left: Accuracy
    ax = axes[0]
    for method in ["bounds", "0.5", "0.8"]:
        subset = df[df["p_method"] == method]
        agg = subset.groupby("obs_per_dof").agg(
            acc=("is_correct", "mean"),
        ).reset_index().sort_values("obs_per_dof")

        ax.plot(agg["obs_per_dof"], agg["acc"] * 100, "o-",
                color=method_colors[method], label=method_labels[method],
                markersize=4, lw=1.5)

    ax.axhline(50, color=GRAY_LIGHT, linestyle="--", lw=0.8)
    ax.axvspan(1, 10, color=GRAY_PALE, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("obs/dof")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(1.5, 40)
    ax.set_ylim(-5, 105)
    ax.xaxis.set_major_locator(FixedLocator([2, 5, 10, 20]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["2", "5", "10", "20"])
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    ax.set_title("(a) Accuracy", fontsize=9)
    despine(ax)

    # Right: Bias
    ax = axes[1]
    for method in ["bounds", "0.5", "0.8"]:
        subset = df[df["p_method"] == method]
        agg = subset.groupby("obs_per_dof").agg(
            bias=("signed_error", "mean"),
        ).reset_index().sort_values("obs_per_dof")

        ax.plot(agg["obs_per_dof"], agg["bias"], "o-",
                color=method_colors[method], label=method_labels[method],
                markersize=4, lw=1.5)

    ax.axhline(0, color=GRAY, linestyle="-", lw=0.8)
    ax.axvspan(1, 10, color=GRAY_PALE, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("obs/dof")
    ax.set_ylabel("Bias")
    ax.set_xlim(1.5, 40)
    ax.xaxis.set_major_locator(FixedLocator([2, 5, 10, 20]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["2", "5", "10", "20"])
    ax.set_title("(b) Bias", fontsize=9)
    despine(ax)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "p_comparison_two_panel.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: p_comparison_two_panel.pdf")

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
