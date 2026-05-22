"""Plot SWOW behavioral prediction results.

Generates:
  - method_comparison.pdf: bar chart comparing Ridge/Lasso/Projection
  - projection_scatter_grid.pdf: scatter grid of predicted vs actual ratings
  - {property}_prediction.pdf: individual prediction scatters

Wordclouds are in experiments/datasets/visualize/ (run with wordclouds=true).

Usage:
    poetry run python experiments/analyses/swow/plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.colors import CYCLE, GRAY_LIGHT, INDIGO, ROSE, TEAL
from src.utils.figure_theme import create_figure, despine, save_figure

TASK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TASK_DIR / "outputs"


# ---------------------------------------------------------------------------
# Panel functions (reusable, accept ax)
# ---------------------------------------------------------------------------

def panel_method_comparison(ax: plt.Axes, results: pd.DataFrame) -> None:
    """Bar chart comparing correlations across dimensions and methods."""
    dimensions = sorted(results["dimension"].dropna().unique())
    methods = sorted(results["method"].dropna().unique())
    colors = {m: c for m, c in zip(methods, CYCLE)}

    n_groups = len(methods)
    x = np.arange(len(dimensions))
    width = 0.8 / n_groups

    for i, method in enumerate(methods):
        group = results[results["method"] == method]
        heights = []
        for d in dimensions:
            vals = group[group["dimension"] == d]["correlation"].values
            heights.append(vals[0] if len(vals) > 0 else 0)
        offset = (i - (n_groups - 1) / 2) * width
        ax.bar(x + offset, heights, width * 0.9, label=method, color=colors[method])

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in dimensions], rotation=35, ha="right")
    ax.set_ylabel("Spearman correlation")
    ax.legend(frameon=False)
    despine(ax)


def panel_projection_scatter_grid(
    axes: np.ndarray,
    projections: pd.DataFrame,
    ratings: pd.DataFrame,
) -> None:
    """Scatter grid: ground truth (x) vs predicted (y) per property."""
    merged = projections.merge(ratings, on="word", how="inner")
    dimensions = [col for col in projections.columns if col != "word"]

    for idx, dimension in enumerate(dimensions):
        if idx >= axes.size:
            break
        ax = axes.flat[idx]
        valid = merged[[dimension + "_x", dimension + "_y"]].dropna()
        if len(valid) < 10:
            ax.text(0.5, 0.5, f"Insufficient data\nfor {dimension}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(dimension.title())
            continue

        x = valid[dimension + "_y"].values
        y = valid[dimension + "_x"].values
        r, p = spearmanr(x, y)

        ax.scatter(x, y, alpha=0.35, s=14, color=GRAY_LIGHT, edgecolors="none")
        z = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "-", color=INDIGO, linewidth=2.0)
        ax.set_xlabel("Actual rating")
        ax.set_ylabel("Predicted")
        ax.set_title(dimension.title())
        ax.text(0.04, 0.96, fr"$\rho$ = {r:.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=10)
        despine(ax)

    for idx in range(len(dimensions), axes.size):
        axes.flat[idx].axis("off")


# ---------------------------------------------------------------------------
# Backward-compatible wrappers (called by run.py)
# ---------------------------------------------------------------------------

def plot_correlation_bars(results: pd.DataFrame, output_path: Path) -> None:
    fig, ax = create_figure("wide")
    panel_method_comparison(ax, results)
    save_figure(fig, output_path)


def plot_projection_scatter_grid(
    projections: pd.DataFrame, ratings: pd.DataFrame, output_path: Path,
) -> None:
    dimensions = [col for col in projections.columns if col != "word"]
    n_dims = len(dimensions)
    n_cols = 3
    n_rows = (n_dims + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = np.atleast_2d(axes)
    panel_projection_scatter_grid(axes, projections, ratings)

    plt.suptitle("Ridge Encoding: Ground Truth vs Predicted", fontsize=14, y=1.00)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------

def main():
    results_csv = OUTPUT_DIR / "results.csv"
    if not results_csv.exists():
        print(f"No results at {results_csv}")
        return

    results = pd.read_csv(results_csv)
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = create_figure("wide")
    panel_method_comparison(ax, results)
    save_figure(fig, plots_dir / "method_comparison.pdf")
    print("Saved method_comparison.pdf")


if __name__ == "__main__":
    main()
