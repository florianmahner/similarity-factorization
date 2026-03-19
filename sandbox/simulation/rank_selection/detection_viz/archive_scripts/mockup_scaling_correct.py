"""Correct scaling visualization using effective obs/dof (accounting for CV sampling).

Usage:
    poetry run python sandbox/rank_detection_viz/mockup_scaling_correct.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.colors import TEAL, GRAY, GRAY_DARK
from src.utils import get_output_dir
from src.utils.figure_theme import create_figure, despine, save_figure

PROJECT_ROOT = Path(__file__).parents[2]
DATA_PATH = PROJECT_ROOT / "outputs/experiments/simulation/data/rank_detection.csv"
OUTPUT_DIR = get_output_dir()

N_SIMULATION = 300


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df[df["alpha"].isin([0.1, 1.0, 5.0])].copy()
    total_obs = N_SIMULATION * (N_SIMULATION - 1) / 2
    df["obs_per_dof_effective"] = (
        df["p_mean"] * total_obs / (N_SIMULATION * df["true_rank"])
    )
    return df


def plot_mae_vs_effective_obs_dof(df: pd.DataFrame, output_dir: Path) -> None:
    """MAE vs effective obs/dof (the correct metric)."""
    summary = (
        df.groupby("true_rank")
        .agg(
            mae=("abs_error", "mean"),
            sem=("abs_error", "sem"),
            obs_dof_eff=("obs_per_dof_effective", "mean"),
        )
        .reset_index()
    )

    fig, ax = create_figure("single")

    # Theoretical minimum
    ax.axvline(1, color=GRAY, linestyle="--", linewidth=1.5, zorder=1)
    ax.text(
        1.2, 2.8, "theoretical\nminimum", fontsize=7, color=GRAY, ha="left"
    )

    # Plot data
    ax.errorbar(
        summary["obs_dof_eff"],
        summary["mae"],
        yerr=summary["sem"],
        fmt="o-",
        color=TEAL,
        markersize=6,
        linewidth=1.5,
        capsize=3,
    )

    # Annotate k values
    for _, row in summary.iterrows():
        if row["true_rank"] in [2, 10, 20, 30]:
            offset = (5, 3)
            ax.annotate(
                f"k={int(row['true_rank'])}",
                (row["obs_dof_eff"], row["mae"]),
                textcoords="offset points",
                xytext=offset,
                fontsize=7,
                color=GRAY_DARK,
            )

    ax.set_xlabel("Effective observations / degrees of freedom")
    ax.set_ylabel("Mean absolute error")
    ax.set_xlim(0, 45)
    ax.set_ylim(0, 4)
    despine(ax)

    save_figure(fig, output_dir / "mae_vs_effective_obs_dof.pdf")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data()

    print("\nGenerating correct scaling plot...")

    plot_mae_vs_effective_obs_dof(df, OUTPUT_DIR)
    print("  mae_vs_effective_obs_dof.pdf")

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
