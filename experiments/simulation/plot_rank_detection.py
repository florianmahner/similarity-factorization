"""
Rank detection plots for simulation experiments.

Creates 4 publication-quality plots:
1. Rank detection by true rank (fixed alpha=0.1, SNR=1.0)
2. Rank detection by alpha (fixed k=20, SNR=1.0)
3. Rank detection by SNR (fixed k=20, alpha=0.1)
4. Method comparison (mean absolute error)

Usage:
    poetry run python experiments/simulation/plot_rank_detection.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.colors import ROSE, TEAL, CYAN, SAND, GRAY, GRAY_LIGHT, GRAY_DARK, GRAY_PALE
from src.utils.figure_theme import (
    create_figure,
    despine,
    save_figure,
)

PROJECT_ROOT = Path(__file__).parents[2]
DATA_PATH = PROJECT_ROOT / "outputs/experiments/simulation/data/rank_detection/rank_detection.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/simulation/plots"


def load_data() -> pd.DataFrame:
    """Load and preprocess rank detection results."""
    df = pd.read_csv(DATA_PATH)
    return df


def plot_by_true_rank(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 1: Selected rank vs true rank (alpha=0.1, SNR=1.0)."""
    subset = df[(df["alpha"] == 0.1) & (df["snr"] == 1.0)].copy()

    agg = (
        subset.groupby("true_rank")["rank_cv"]
        .agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
        .reset_index()
    )
    agg.columns = ["true_rank", "median", "q25", "q75"]

    fig, ax = create_figure("single")

    # Identity line
    lims = [5, 45]
    ax.plot(lims, lims, "--", color=GRAY_LIGHT, lw=1.5, zorder=0)
    ax.text(42, 39, "identity", fontsize=7, color=GRAY, ha="right", va="top")

    # CV results
    ax.plot(
        agg["true_rank"],
        agg["median"],
        "o-",
        color=TEAL,
        markersize=5,
        linewidth=1.5,
        zorder=3,
    )
    ax.fill_between(
        agg["true_rank"],
        agg["q25"],
        agg["q75"],
        color=TEAL,
        alpha=0.2,
        linewidth=0,
        zorder=2,
    )

    ax.set_xlabel("True rank")
    ax.set_ylabel("Selected rank (CV)")
    ax.set_xlim(5, 45)
    ax.set_ylim(5, 45)
    ax.set_xticks([10, 15, 20, 25, 30, 35, 40])
    ax.set_yticks([10, 15, 20, 25, 30, 35, 40])
    ax.set_aspect("equal")
    despine(ax)

    save_figure(fig, output_dir / "rank_detection_by_true_rank.pdf")
    print(f"  Saved: rank_detection_by_true_rank.pdf")


def plot_by_alpha(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 2: Selected rank vs alpha (k=20, SNR=1.0)."""
    subset = df[(df["true_rank"] == 20) & (df["snr"] == 1.0)].copy()

    agg = (
        subset.groupby("alpha")["rank_cv"]
        .agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
        .reset_index()
    )
    agg.columns = ["alpha", "median", "q25", "q75"]

    fig, ax = create_figure("single")

    # True rank reference
    ax.axhline(20, color=GRAY_LIGHT, linestyle="--", linewidth=1.5, zorder=0)
    ax.text(0.12, 20.8, "true rank", fontsize=7, color=GRAY, ha="left")

    # CV results
    ax.plot(
        agg["alpha"],
        agg["median"],
        "o-",
        color=ROSE,
        markersize=6,
        linewidth=1.5,
        zorder=3,
    )
    ax.fill_between(
        agg["alpha"],
        agg["q25"],
        agg["q75"],
        color=ROSE,
        alpha=0.2,
        linewidth=0,
        zorder=2,
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"Concentration $\alpha$")
    ax.set_ylabel("Selected rank (CV)")
    ax.set_xlim(0.08, 7)
    ax.set_ylim(15, 35)

    # Custom x-ticks
    ax.set_xticks([0.1, 0.5, 2.0, 5.0])
    ax.set_xticklabels(["0.1", "0.5", "2.0", "5.0"])

    despine(ax)

    save_figure(fig, output_dir / "rank_detection_by_alpha.pdf")
    print(f"  Saved: rank_detection_by_alpha.pdf")


def plot_by_snr(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 3: Selected rank vs SNR (k=20, alpha=0.1)."""
    subset = df[(df["true_rank"] == 20) & (df["alpha"] == 0.1)].copy()

    agg = (
        subset.groupby("snr")["rank_cv"]
        .agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
        .reset_index()
    )
    agg.columns = ["snr", "median", "q25", "q75"]

    fig, ax = create_figure("single")

    # Low SNR region shading
    ax.axvspan(-0.05, 0.5, color=GRAY_PALE, zorder=0)
    ax.text(0.25, 25, "low SNR", fontsize=7, color=GRAY, ha="center")

    # True rank reference
    ax.axhline(20, color=GRAY_LIGHT, linestyle="--", linewidth=1.5, zorder=1)

    # CV results
    ax.plot(
        agg["snr"],
        agg["median"],
        "o-",
        color=CYAN,
        markersize=5,
        linewidth=1.5,
        zorder=3,
    )
    ax.fill_between(
        agg["snr"],
        agg["q25"],
        agg["q75"],
        color=CYAN,
        alpha=0.2,
        linewidth=0,
        zorder=2,
    )

    ax.set_xlabel("Signal-to-noise ratio (SNR)")
    ax.set_ylabel("Selected rank (CV)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(18, 26)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    despine(ax)

    save_figure(fig, output_dir / "rank_detection_by_snr.pdf")
    print(f"  Saved: rank_detection_by_snr.pdf")


def plot_method_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 4: Method comparison - mean absolute error."""
    methods = {
        "CV": "rank_cv",
        "Kaiser": "rank_kaiser",
        "Variance": "rank_variance",
        "BIC": "rank_bic",
        "AIC": "rank_aic",
    }

    # Calculate absolute errors
    errors = {}
    for name, col in methods.items():
        abs_error = np.abs(df[col] - df["true_rank"])
        errors[name] = {"mean": abs_error.mean(), "sem": abs_error.sem()}

    # Sort by mean error
    sorted_methods = sorted(errors.keys(), key=lambda x: errors[x]["mean"])

    fig, ax = create_figure("single")

    y_pos = np.arange(len(sorted_methods))
    means = [errors[m]["mean"] for m in sorted_methods]
    sems = [errors[m]["sem"] for m in sorted_methods]

    # Color CV differently
    colors = [TEAL if m == "CV" else GRAY for m in sorted_methods]

    bars = ax.barh(y_pos, means, xerr=sems, color=colors, edgecolor="white", height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_methods)
    ax.set_xlabel("Mean absolute error (ranks)")
    ax.set_xlim(0, None)
    ax.invert_yaxis()

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    save_figure(fig, output_dir / "rank_detection_methods.pdf")
    print(f"  Saved: rank_detection_methods.pdf")


def plot_method_comparison_by_condition(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 5: Method comparison broken down by experimental condition."""
    methods = ["CV", "Kaiser", "Variance", "BIC", "AIC"]
    method_cols = {
        "CV": "rank_cv",
        "Kaiser": "rank_kaiser",
        "Variance": "rank_variance",
        "BIC": "rank_bic",
        "AIC": "rank_aic",
    }

    # Define conditions
    conditions = {
        "Vary rank\n(α=0.1, SNR=1)": (df["alpha"] == 0.1) & (df["snr"] == 1.0),
        "Vary α\n(k=20, SNR=1)": (df["true_rank"] == 20) & (df["snr"] == 1.0),
        "Vary SNR\n(k=20, α=0.1)": (df["true_rank"] == 20) & (df["alpha"] == 0.1),
    }

    fig, ax = create_figure("wide")

    x = np.arange(len(conditions))
    width = 0.15
    offsets = np.array([-2, -1, 0, 1, 2]) * width

    colors = {
        "CV": TEAL,
        "Kaiser": ROSE,
        "Variance": CYAN,
        "BIC": SAND,
        "AIC": GRAY,
    }

    for i, method in enumerate(methods):
        col = method_cols[method]
        means = []
        sems = []
        for cond_mask in conditions.values():
            subset = df[cond_mask]
            abs_error = np.abs(subset[col] - subset["true_rank"])
            means.append(abs_error.mean())
            sems.append(abs_error.sem())

        ax.bar(
            x + offsets[i],
            means,
            width,
            yerr=sems,
            label=method,
            color=colors[method],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(conditions.keys(), fontsize=8)
    ax.set_ylabel("Mean absolute error")
    ax.set_ylim(0, None)
    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=7,
        ncol=2,
    )
    despine(ax)

    save_figure(fig, output_dir / "rank_detection_methods_by_condition.pdf")
    print(f"  Saved: rank_detection_methods_by_condition.pdf")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} rows loaded")

    print("\nCreating plots...")
    plot_by_true_rank(df, OUTPUT_DIR)
    plot_by_alpha(df, OUTPUT_DIR)
    plot_by_snr(df, OUTPUT_DIR)
    plot_method_comparison(df, OUTPUT_DIR)
    plot_method_comparison_by_condition(df, OUTPUT_DIR)

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
