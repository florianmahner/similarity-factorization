"""Honest scaling visualization - theoretical vs empirical.

Usage:
    poetry run python sandbox/rank_detection_viz/mockup_scaling_honest.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.colors import ROSE, TEAL, GRAY, GRAY_DARK, GRAY_PALE
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

PROJECT_ROOT = Path(__file__).parents[2]
DATA_PATH = PROJECT_ROOT / "outputs/experiments/simulation/data/rank_detection.csv"
OUTPUT_DIR = get_output_dir()

N_SIMULATION = 300


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df[df["alpha"].isin([0.1, 1.0, 5.0])].copy()
    df["obs_per_dof"] = (N_SIMULATION * (N_SIMULATION - 1) / 2) / (N_SIMULATION * df["true_rank"])
    return df


def plot_mae_vs_obs_dof_honest(df: pd.DataFrame, output_dir: Path) -> None:
    """MAE vs obs/dof with only theoretical minimum marked."""
    summary = df.groupby("true_rank").agg(
        mae=("abs_error", "mean"),
        sem=("abs_error", "sem"),
        obs_per_dof=("obs_per_dof", "first"),
    ).reset_index()

    fig, ax = create_figure("single")

    # Only mark the theoretical minimum (obs/dof = 1)
    ax.axvline(1, color=GRAY, linestyle="--", linewidth=1.5, zorder=1)
    ax.text(1.5, 2.8, "theoretical\nminimum", fontsize=7, color=GRAY, ha="left")

    # Plot data
    ax.errorbar(summary["obs_per_dof"], summary["mae"], yerr=summary["sem"],
                fmt="o-", color=TEAL, markersize=6, linewidth=1.5, capsize=3)

    # Annotate k values
    for _, row in summary.iterrows():
        if row["true_rank"] in [2, 10, 20, 30]:
            offset = (5, 5) if row["true_rank"] != 2 else (5, -10)
            ax.annotate(f"k={int(row['true_rank'])}",
                       (row["obs_per_dof"], row["mae"]),
                       textcoords="offset points", xytext=offset,
                       fontsize=7, color=GRAY_DARK)

    ax.set_xlabel("Observations per degree of freedom")
    ax.set_ylabel("Mean absolute error")
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 4)
    despine(ax)

    save_figure(fig, output_dir / "mae_vs_obs_dof_honest.pdf")


def plot_required_samples_honest(output_dir: Path) -> None:
    """Required samples plot with only theoretical minimum."""
    fig, ax = create_figure("single")

    ks = np.arange(2, 51, 1)

    # Theoretical minimum: obs/dof = 1
    # n(n-1)/2 = n*k  =>  n = 2k + 1
    n_theoretical = 2 * ks + 1

    # Shade below theoretical minimum
    ax.fill_between(ks, 0, n_theoretical, color=GRAY_PALE, alpha=0.7,
                    label="Underdetermined (obs/dof < 1)")
    ax.plot(ks, n_theoretical, "-", color=GRAY, linewidth=2,
            label="Theoretical minimum (obs/dof = 1)")

    # Mark n=300
    ax.axhline(300, color=ROSE, linestyle=":", linewidth=1.5, zorder=5)
    ax.text(48, 315, "n=300", fontsize=8, color=ROSE, ha="right")

    # Where does n=300 cross theoretical minimum?
    # 300 = 2k + 1  =>  k = 149.5 (way beyond our range, so n=300 is always above theoretical min)

    ax.set_xlabel("True rank (k)")
    ax.set_ylabel("Required samples (n)")
    ax.set_xlim(2, 50)
    ax.set_ylim(0, 400)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    despine(ax)

    save_figure(fig, output_dir / "required_samples_honest.pdf")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data()

    print("\nGenerating honest scaling plots...")

    plot_mae_vs_obs_dof_honest(df, OUTPUT_DIR)
    print("  mae_vs_obs_dof_honest.pdf")

    plot_required_samples_honest(OUTPUT_DIR)
    print("  required_samples_honest.pdf")

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
