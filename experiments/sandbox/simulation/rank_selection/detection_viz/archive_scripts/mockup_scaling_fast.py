"""Fast scaling visualization using existing data + theoretical curve.

Usage:
    poetry run python sandbox/rank_detection_viz/mockup_scaling_fast.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.colors import ROSE, TEAL, GRAY, GRAY_LIGHT, GRAY_DARK, GRAY_PALE
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

PROJECT_ROOT = Path(__file__).parents[2]
DATA_PATH = PROJECT_ROOT / "outputs/experiments/simulation/data/rank_detection.csv"
OUTPUT_DIR = get_output_dir()

# The simulation used n=300
N_SIMULATION = 300


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df[df["alpha"].isin([0.1, 1.0, 5.0])].copy()
    # Compute obs/dof
    df["obs_per_dof"] = (N_SIMULATION * (N_SIMULATION - 1) / 2) / (N_SIMULATION * df["true_rank"])
    return df


def plot_mae_vs_obs_dof(df: pd.DataFrame, output_dir: Path) -> None:
    """MAE vs observations per DOF from existing data."""
    summary = df.groupby("true_rank").agg(
        mae=("abs_error", "mean"),
        sem=("abs_error", "sem"),
        obs_per_dof=("obs_per_dof", "first"),
    ).reset_index()

    fig, ax = create_figure("single")

    # Underdetermined region
    ax.axvspan(0, 5, color=GRAY_PALE, zorder=0)
    ax.axvline(5, color=GRAY_LIGHT, linestyle="--", linewidth=1, zorder=1)
    ax.text(2.5, 0.3, "underdetermined", fontsize=7, color=GRAY,
            ha="center", rotation=90)

    # Plot MAE vs obs/dof
    ax.errorbar(summary["obs_per_dof"], summary["mae"], yerr=summary["sem"],
                fmt="o-", color=TEAL, markersize=6, linewidth=1.5, capsize=3)

    # Annotate some points with k values
    for _, row in summary.iterrows():
        if row["true_rank"] in [2, 10, 20, 30]:
            ax.annotate(f"k={int(row['true_rank'])}",
                       (row["obs_per_dof"], row["mae"]),
                       textcoords="offset points", xytext=(5, 5),
                       fontsize=7, color=GRAY_DARK)

    ax.set_xlabel("Observations per degree of freedom")
    ax.set_ylabel("Mean absolute error")
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 5)
    despine(ax)

    save_figure(fig, output_dir / "mae_vs_obs_dof.pdf")


def plot_required_samples(output_dir: Path) -> None:
    """Show required n for different k to achieve reliable detection."""
    fig, ax = create_figure("single")

    ks = np.arange(5, 51, 1)

    # For obs/dof = r: n(n-1)/2 = r*n*k => n ≈ 2*r*k + 1
    n_for_ratio_5 = 2 * 5 * ks + 1
    n_for_ratio_8 = 2 * 8 * ks + 1

    ax.fill_between(ks, n_for_ratio_5, n_for_ratio_8, color=TEAL, alpha=0.3,
                    label="Reliable (obs/dof 5-8)")
    ax.plot(ks, n_for_ratio_5, "-", color=TEAL, linewidth=1.5)
    ax.plot(ks, n_for_ratio_8, "--", color=TEAL, linewidth=1.5)

    # Shade underdetermined region
    ax.fill_between(ks, 0, n_for_ratio_5, color=GRAY_PALE, alpha=0.5)
    ax.text(30, 80, "underdetermined", fontsize=8, color=GRAY,
            ha="center", rotation=22)

    # Mark the simulation point (n=300)
    ax.axhline(300, color=ROSE, linestyle=":", linewidth=1.5, zorder=5)
    ax.text(48, 310, "n=300", fontsize=7, color=ROSE, ha="right")

    # Find where n=300 crosses the threshold
    k_threshold = 300 / (2 * 5)  # obs/dof = 5 threshold
    ax.axvline(k_threshold, color=ROSE, linestyle=":", linewidth=1, alpha=0.5)

    ax.set_xlabel("True rank (k)")
    ax.set_ylabel("Required samples (n)")
    ax.set_xlim(5, 50)
    ax.set_ylim(0, 600)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    despine(ax)

    save_figure(fig, output_dir / "required_samples.pdf")


def plot_combined_scaling(df: pd.DataFrame, output_dir: Path) -> None:
    """Two-panel figure: MAE vs obs/dof (left), required samples (right)."""
    fig, axes = create_figure("full_width", nrows=1, ncols=2)

    # Left: MAE vs obs/dof
    ax = axes[0]
    summary = df.groupby("true_rank").agg(
        mae=("abs_error", "mean"),
        sem=("abs_error", "sem"),
        obs_per_dof=("obs_per_dof", "first"),
    ).reset_index()

    ax.axvspan(0, 5, color=GRAY_PALE, zorder=0)
    ax.axvline(5, color=GRAY_LIGHT, linestyle="--", linewidth=1, zorder=1)

    ax.errorbar(summary["obs_per_dof"], summary["mae"], yerr=summary["sem"],
                fmt="o-", color=TEAL, markersize=5, linewidth=1.5, capsize=2)

    for _, row in summary.iterrows():
        if row["true_rank"] in [2, 10, 20, 30]:
            ax.annotate(f"k={int(row['true_rank'])}",
                       (row["obs_per_dof"], row["mae"]),
                       textcoords="offset points", xytext=(4, 4),
                       fontsize=6, color=GRAY_DARK)

    ax.set_xlabel("Observations / degrees of freedom")
    ax.set_ylabel("Mean absolute error")
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 5)
    despine(ax)

    # Right: Required samples
    ax = axes[1]
    ks = np.arange(5, 51, 1)
    n_for_ratio_5 = 2 * 5 * ks + 1
    n_for_ratio_8 = 2 * 8 * ks + 1

    ax.fill_between(ks, n_for_ratio_5, n_for_ratio_8, color=TEAL, alpha=0.3)
    ax.plot(ks, n_for_ratio_5, "-", color=TEAL, linewidth=1.5)
    ax.plot(ks, n_for_ratio_8, "--", color=TEAL, linewidth=1.5)
    ax.fill_between(ks, 0, n_for_ratio_5, color=GRAY_PALE, alpha=0.5)

    ax.axhline(300, color=ROSE, linestyle=":", linewidth=1.5, zorder=5)
    ax.text(48, 315, "n=300", fontsize=7, color=ROSE, ha="right")

    ax.set_xlabel("True rank (k)")
    ax.set_ylabel("Required samples (n)")
    ax.set_xlim(5, 50)
    ax.set_ylim(0, 600)
    despine(ax)

    save_figure(fig, output_dir / "scaling_combined.pdf")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading existing data...")
    df = load_data()

    print("\nGenerating scaling plots...")

    plot_mae_vs_obs_dof(df, OUTPUT_DIR)
    print("  mae_vs_obs_dof.pdf")

    plot_required_samples(OUTPUT_DIR)
    print("  required_samples.pdf")

    plot_combined_scaling(df, OUTPUT_DIR)
    print("  scaling_combined.pdf")

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
