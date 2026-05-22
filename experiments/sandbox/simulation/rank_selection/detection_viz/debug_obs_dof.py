"""Debug plots: How obs/dof, p_mean, and rank detection interact.

Usage:
    poetry run python sandbox/rank_detection_viz/debug_obs_dof.py
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.colors import TEAL, ROSE, GRAY, GRAY_LIGHT, GRAY_PALE
from src.utils.figure_theme import despine, save_figure
from src.utils import get_output_dir

DATA_PATH = Path("sandbox/rank_detection_viz/outputs/251217/111105/rank_detection_obs_dof_range.csv")
OUTPUT_DIR = get_output_dir()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    df = pd.read_csv(DATA_PATH)

    # Compute additional quantities
    df["theoretical_obs_dof"] = df["obs_per_dof"]  # This is (n-1)/(2k)
    df["effective_obs_dof"] = df["p_mean"] * df["obs_per_dof"]

    # Aggregate by true_rank
    agg = df.groupby("true_rank").agg(
        theoretical_obs_dof=("theoretical_obs_dof", "first"),
        p_mean=("p_mean", "mean"),
        p_std=("p_mean", "std"),
        effective_obs_dof=("effective_obs_dof", "mean"),
        bias=("signed_error", "mean"),
        bias_sem=("signed_error", "sem"),
        mae=("abs_error", "mean"),
        accuracy=("is_correct", "mean"),
    ).reset_index().sort_values("true_rank", ascending=False)

    print("Summary by true rank:")
    print(agg.round(3).to_string(index=False))
    print()

    # === Plot 1: p_mean vs true rank ===
    fig, ax = plt.subplots(figsize=(4, 3))

    ax.errorbar(agg["true_rank"], agg["p_mean"], yerr=agg["p_std"],
                fmt="o-", color=TEAL, capsize=3, markersize=5)

    ax.axhline(0.5, color=GRAY_LIGHT, linestyle="--", lw=0.8)
    ax.set_xlabel("True rank (k)")
    ax.set_ylabel("p_mean (sampling fraction)")
    ax.set_title("Bounds estimation vs rank")
    ax.set_ylim(0, 0.7)
    despine(ax)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "1_pmean_vs_rank.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: 1_pmean_vs_rank.pdf")

    # === Plot 2: p_mean vs theoretical obs/dof ===
    fig, ax = plt.subplots(figsize=(4, 3))

    ax.errorbar(agg["theoretical_obs_dof"], agg["p_mean"], yerr=agg["p_std"],
                fmt="o-", color=TEAL, capsize=3, markersize=5)

    ax.axhline(0.5, color=GRAY_LIGHT, linestyle="--", lw=0.8)
    ax.axvspan(0.5, 10, color=GRAY_PALE, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("Theoretical obs/dof = (n-1)/(2k)")
    ax.set_ylabel("p_mean (sampling fraction)")
    ax.set_title("Bounds estimation vs theoretical obs/dof")
    ax.set_xlim(0.8, 60)
    ax.set_ylim(0, 0.7)
    despine(ax)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "2_pmean_vs_theoretical_obs_dof.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: 2_pmean_vs_theoretical_obs_dof.pdf")

    # === Plot 3: Effective obs/dof vs theoretical obs/dof ===
    fig, ax = plt.subplots(figsize=(4, 3))

    ax.plot(agg["theoretical_obs_dof"], agg["effective_obs_dof"], "o-",
            color=TEAL, markersize=5)

    # Identity line
    x = np.array([0.5, 60])
    ax.plot(x, x, "--", color=GRAY, lw=1, label="Identity (p=1)")
    ax.plot(x, 0.5*x, ":", color=GRAY_LIGHT, lw=1, label="p=0.5")

    ax.axvspan(0.5, 10, color=GRAY_PALE, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Theoretical obs/dof = (n-1)/(2k)")
    ax.set_ylabel("Effective obs/dof = p × (n-1)/(2k)")
    ax.set_title("Effective vs theoretical obs/dof")
    ax.set_xlim(0.8, 60)
    ax.set_ylim(0.01, 60)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    despine(ax)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "3_effective_vs_theoretical_obs_dof.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: 3_effective_vs_theoretical_obs_dof.pdf")

    # === Plot 4: Bias vs theoretical obs/dof ===
    fig, ax = plt.subplots(figsize=(4, 3))

    ax.errorbar(agg["theoretical_obs_dof"], agg["bias"], yerr=agg["bias_sem"],
                fmt="o-", color=TEAL, capsize=3, markersize=5)

    ax.axhline(0, color=GRAY, linestyle="-", lw=0.8)
    ax.axvspan(0.5, 10, color=GRAY_PALE, zorder=0)
    ax.axvline(10, color=GRAY_LIGHT, linestyle="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Theoretical obs/dof = (n-1)/(2k)")
    ax.set_ylabel("Bias (selected − true)")
    ax.set_title("Bias vs theoretical obs/dof")
    ax.set_xlim(0.8, 60)
    despine(ax)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "4_bias_vs_theoretical_obs_dof.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: 4_bias_vs_theoretical_obs_dof.pdf")

    # === Plot 5: Bias vs effective obs/dof ===
    fig, ax = plt.subplots(figsize=(4, 3))

    ax.errorbar(agg["effective_obs_dof"], agg["bias"], yerr=agg["bias_sem"],
                fmt="o-", color=ROSE, capsize=3, markersize=5)

    ax.axhline(0, color=GRAY, linestyle="-", lw=0.8)
    ax.axvspan(0.01, 10, color=GRAY_PALE, zorder=0)
    ax.axvline(10, color=GRAY_LIGHT, linestyle="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Effective obs/dof = p × (n-1)/(2k)")
    ax.set_ylabel("Bias (selected − true)")
    ax.set_title("Bias vs effective obs/dof")
    ax.set_xlim(0.01, 60)
    despine(ax)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "5_bias_vs_effective_obs_dof.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: 5_bias_vs_effective_obs_dof.pdf")

    # === Plot 6: Accuracy vs both obs/dof measures ===
    fig, ax = plt.subplots(figsize=(4, 3))

    ax.plot(agg["theoretical_obs_dof"], agg["accuracy"] * 100, "o-",
            color=TEAL, markersize=5, label="Theoretical")
    ax.plot(agg["effective_obs_dof"], agg["accuracy"] * 100, "s--",
            color=ROSE, markersize=5, label="Effective")

    ax.axhline(50, color=GRAY_LIGHT, linestyle="--", lw=0.8)
    ax.axvspan(0.01, 10, color=GRAY_PALE, zorder=0)
    ax.axvline(10, color=GRAY_LIGHT, linestyle="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("obs/dof")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs obs/dof (both measures)")
    ax.set_xlim(0.01, 60)
    ax.set_ylim(-5, 105)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    despine(ax)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "6_accuracy_vs_both_obs_dof.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: 6_accuracy_vs_both_obs_dof.pdf")

    # === Plot 7: Combined 2x2 summary ===
    fig, axes = plt.subplots(2, 2, figsize=(6, 5))

    # Top-left: p_mean vs rank
    ax = axes[0, 0]
    ax.plot(agg["true_rank"], agg["p_mean"], "o-", color=TEAL, markersize=4)
    ax.axhline(0.5, color=GRAY_LIGHT, linestyle="--", lw=0.8)
    ax.set_xlabel("True rank (k)")
    ax.set_ylabel("p_mean")
    ax.set_title("(a) Sampling fraction", fontsize=9)
    despine(ax)

    # Top-right: effective vs theoretical
    ax = axes[0, 1]
    ax.plot(agg["theoretical_obs_dof"], agg["effective_obs_dof"], "o-",
            color=TEAL, markersize=4)
    x = np.array([0.5, 60])
    ax.plot(x, x, "--", color=GRAY, lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Theoretical obs/dof")
    ax.set_ylabel("Effective obs/dof")
    ax.set_title("(b) Effective vs theoretical", fontsize=9)
    ax.set_xlim(0.8, 60)
    ax.set_ylim(0.01, 60)
    despine(ax)

    # Bottom-left: bias vs theoretical
    ax = axes[1, 0]
    ax.plot(agg["theoretical_obs_dof"], agg["bias"], "o-", color=TEAL, markersize=4)
    ax.axhline(0, color=GRAY, linestyle="-", lw=0.8)
    ax.axvspan(0.5, 10, color=GRAY_PALE, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("Theoretical obs/dof")
    ax.set_ylabel("Bias")
    ax.set_title("(c) Bias vs theoretical", fontsize=9)
    ax.set_xlim(0.8, 60)
    despine(ax)

    # Bottom-right: bias vs effective
    ax = axes[1, 1]
    ax.plot(agg["effective_obs_dof"], agg["bias"], "o-", color=ROSE, markersize=4)
    ax.axhline(0, color=GRAY, linestyle="-", lw=0.8)
    ax.axvspan(0.01, 10, color=GRAY_PALE, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("Effective obs/dof")
    ax.set_ylabel("Bias")
    ax.set_title("(d) Bias vs effective", fontsize=9)
    ax.set_xlim(0.01, 60)
    despine(ax)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "7_summary_2x2.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: 7_summary_2x2.pdf")

    # === Plot 8: Individual seed scatter ===
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # Left: bias vs theoretical (all seeds)
    ax = axes[0]
    ax.scatter(df["theoretical_obs_dof"], df["signed_error"],
               alpha=0.5, s=20, color=TEAL)
    ax.axhline(0, color=GRAY, linestyle="-", lw=0.8)
    ax.axvspan(0.5, 10, color=GRAY_PALE, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("Theoretical obs/dof")
    ax.set_ylabel("Signed error (per seed)")
    ax.set_title("Individual seeds: theoretical")
    ax.set_xlim(0.8, 60)
    despine(ax)

    # Right: bias vs effective (all seeds)
    ax = axes[1]
    ax.scatter(df["effective_obs_dof"], df["signed_error"],
               alpha=0.5, s=20, color=ROSE)
    ax.axhline(0, color=GRAY, linestyle="-", lw=0.8)
    ax.axvspan(0.01, 10, color=GRAY_PALE, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("Effective obs/dof")
    ax.set_ylabel("Signed error (per seed)")
    ax.set_title("Individual seeds: effective")
    ax.set_xlim(0.01, 60)
    despine(ax)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "8_scatter_all_seeds.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: 8_scatter_all_seeds.pdf")

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
