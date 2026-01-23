"""Plotting for VICE low-data experiment results.

Usage:
    poetry run python experiments/things_behavior/vice_lowdata_plot.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.figure_theme import (
    CMAP,
    GRAY,
    add_reference_line,
    create_figure,
    despine,
    save_figure,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "things_behavior" / "vice_lowdata"
RESULTS_PATH = OUTPUT_DIR / "results.csv"
FIGURES_DIR = OUTPUT_DIR / "figures"


def compute_ci(data: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Compute confidence interval for mean."""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - h, mean + h


def load_results() -> pd.DataFrame:
    """Load and validate results."""
    df = pd.read_csv(RESULTS_PATH)
    print(f"Loaded {len(df)} results from {RESULTS_PATH}")
    print(f"Percentages: {sorted(df['percentage'].unique())}")
    print(f"Partitions per %: {df.groupby('percentage')['partition'].nunique().to_dict()}")
    print(f"Seeds per partition: {df.groupby(['percentage', 'partition'])['seed'].count().mean():.1f}")
    return df


def plot_accuracy_vs_percentage(df: pd.DataFrame) -> None:
    """Plot accuracy vs data percentage with 95% CI."""
    fig, ax = create_figure("single")

    percentages = sorted(df["percentage"].unique())
    means = []
    ci_low = []
    ci_high = []

    for pct in percentages:
        data = df[df["percentage"] == pct]["accuracy"].values
        mean = np.mean(data)
        low, high = compute_ci(data)
        means.append(mean)
        ci_low.append(low)
        ci_high.append(high)

    means = np.array(means)
    ci_low = np.array(ci_low)
    ci_high = np.array(ci_high)

    ax.fill_between(percentages, ci_low, ci_high, alpha=0.3, color=CMAP[1])
    ax.plot(percentages, means, "o-", color=CMAP[1], markersize=6, label="VICE")

    add_reference_line(ax, 1/3, orientation="horizontal", color=GRAY["light"])
    ax.text(percentages[-1] + 2, 1/3, "chance", va="center", fontsize=8, color=GRAY["medium"])

    ax.set_xlabel("Training data (%)")
    ax.set_ylabel("Triplet prediction accuracy")
    ax.set_xlim(0, max(percentages) + 5)
    ax.set_ylim(0.3, 0.7)
    ax.set_xticks(percentages)
    despine(ax)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(fig, FIGURES_DIR / "accuracy_vs_percentage.pdf")
    print(f"Saved accuracy plot to {FIGURES_DIR / 'accuracy_vs_percentage.pdf'}")


def plot_dimensions_vs_percentage(df: pd.DataFrame) -> None:
    """Plot inferred dimensions vs data percentage with 95% CI."""
    fig, ax = create_figure("single")

    percentages = sorted(df["percentage"].unique())
    means = []
    ci_low = []
    ci_high = []

    for pct in percentages:
        data = df[df["percentage"] == pct]["n_dimensions"].values
        mean = np.mean(data)
        low, high = compute_ci(data)
        means.append(mean)
        ci_low.append(low)
        ci_high.append(high)

    means = np.array(means)
    ci_low = np.array(ci_low)
    ci_high = np.array(ci_high)

    ax.fill_between(percentages, ci_low, ci_high, alpha=0.3, color=CMAP[1])
    ax.plot(percentages, means, "o-", color=CMAP[1], markersize=6, label="VICE")

    ax.set_xlabel("Training data (%)")
    ax.set_ylabel("Number of dimensions")
    ax.set_xlim(0, max(percentages) + 5)
    ax.set_xticks(percentages)
    despine(ax)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(fig, FIGURES_DIR / "dimensions_vs_percentage.pdf")
    print(f"Saved dimensions plot to {FIGURES_DIR / 'dimensions_vs_percentage.pdf'}")


def main():
    if not RESULTS_PATH.exists():
        print(f"Results file not found: {RESULTS_PATH}")
        print("Run --aggregate first to create results.csv")
        return

    df = load_results()
    plot_accuracy_vs_percentage(df)
    plot_dimensions_vs_percentage(df)

    print("\nSummary statistics:")
    summary = df.groupby("percentage").agg({
        "accuracy": ["mean", "std"],
        "n_dimensions": ["mean", "std"],
    }).round(3)
    print(summary)


if __name__ == "__main__":
    main()
