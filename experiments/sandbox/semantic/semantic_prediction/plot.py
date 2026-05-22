"""
Create publication-quality plots for semantic prediction results.

Usage:
    poetry run python sandbox/semantic_prediction/plot.py
"""

from __future__ import annotations

from pathlib import Path

from src.utils import get_output_dir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import project theme
from src.colors import ROSE, TEAL, CYAN, SAND, PURPLE, GRAY, GRAY_LIGHT, GRAY_DARK, GRAY_PALE, setup_style
from src.utils.figure_theme import despine, save_figure, add_reference_line

OUTPUT_DIR = get_output_dir()


def plot_correlations_by_dataset(results: pd.DataFrame) -> plt.Figure:
    """Create grouped bar plot of correlations by dataset."""
    setup_style()

    # Dataset colors
    dataset_colors = {
        "warriner": ROSE,
        "lancaster": TEAL,
        "glasgow": CYAN,
        "brysbaert": SAND,
        "things": PURPLE,
    }

    # Sort by correlation within each dataset
    results = results.sort_values(["dataset", "correlation"], ascending=[True, False])

    fig, ax = plt.subplots(figsize=(5.5, 6))

    # Create grouped horizontal bar plot
    datasets = results["dataset"].unique()
    y_positions = []
    y_labels = []
    colors = []

    y = 0
    gap = 0.5

    for dataset in sorted(datasets):
        subset = results[results["dataset"] == dataset].sort_values("correlation", ascending=True)
        for _, row in subset.iterrows():
            y_positions.append(y)
            y_labels.append(row["dimension"])
            colors.append(dataset_colors[dataset])
            y += 1
        y += gap

    bars = ax.barh(y_positions, results.sort_values(["dataset", "correlation"])["correlation"],
                   color=colors, edgecolor="white", linewidth=0.5, height=0.8)

    # Add dataset labels
    y = 0
    for dataset in sorted(datasets):
        n_dims = len(results[results["dataset"] == dataset])
        y_mid = y + n_dims / 2 - 0.5
        ax.text(-0.05, y_mid, dataset.upper(), fontsize=9, fontweight="bold",
                ha="right", va="center", transform=ax.get_yaxis_transform())
        y += n_dims + gap

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Spearman correlation (ρ)")
    ax.set_xlim(0, 0.9)

    add_reference_line(ax, 0.5, orientation="vertical", color=GRAY_LIGHT)
    add_reference_line(ax, 0.7, orientation="vertical", color=GRAY_PALE)

    despine(ax)
    ax.set_title("Predicting semantic ratings from SWOW embeddings", fontsize=11, pad=10)

    plt.tight_layout()
    return fig


def plot_top_dimensions(results: pd.DataFrame) -> plt.Figure:
    """Create focused bar plot of top 15 dimensions."""
    setup_style()

    dataset_colors = {
        "warriner": ROSE,
        "lancaster": TEAL,
        "glasgow": CYAN,
        "brysbaert": SAND,
        "things": PURPLE,
    }

    top = results.nlargest(15, "correlation").sort_values("correlation")

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    colors = [dataset_colors[d] for d in top["dataset"]]
    bars = ax.barh(range(len(top)), top["correlation"],
                   color=colors, edgecolor="white", linewidth=0.5)

    # Add dimension labels
    labels = [f"{row['dimension']} ({row['dataset'][:3]})" for _, row in top.iterrows()]
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_xlabel("Spearman correlation (ρ)")
    ax.set_xlim(0.6, 0.85)

    # Add correlation values
    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(row["correlation"] + 0.005, i, f"{row['correlation']:.2f}",
                va="center", fontsize=8, color=GRAY_DARK)

    despine(ax)
    ax.set_title("Top 15 predicted dimensions", fontsize=11, pad=10)

    plt.tight_layout()
    return fig


def plot_dataset_summary(results: pd.DataFrame) -> plt.Figure:
    """Create summary plot by dataset."""
    setup_style()

    dataset_colors = {
        "warriner": ROSE,
        "lancaster": TEAL,
        "glasgow": CYAN,
        "brysbaert": SAND,
        "things": PURPLE,
    }

    # Calculate summary stats
    summary = results.groupby("dataset").agg({
        "correlation": ["mean", "std", "count"],
        "n_samples": "mean"
    }).reset_index()
    summary.columns = ["dataset", "mean", "std", "count", "n_samples"]
    summary = summary.sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(4, 3.5))

    colors = [dataset_colors[d] for d in summary["dataset"]]
    x = range(len(summary))

    # Bar plot with error bars
    bars = ax.bar(x, summary["mean"], color=colors, edgecolor="white",
                  linewidth=0.5, width=0.7)
    ax.errorbar(x, summary["mean"], yerr=summary["std"], fmt="none",
                color=GRAY_DARK, capsize=3, linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in summary["dataset"]],
                       fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Mean correlation (ρ)")
    ax.set_ylim(0, 0.85)

    # Add sample size annotations
    for i, row in summary.iterrows():
        idx = list(summary["dataset"]).index(row["dataset"])
        ax.text(idx, row["mean"] + row["std"] + 0.03,
                f"n={int(row['count'])}", ha="center", fontsize=7, color=GRAY)

    add_reference_line(ax, 0.5, color=GRAY_LIGHT)
    add_reference_line(ax, 0.7, color=GRAY_PALE)

    despine(ax)
    ax.set_title("Prediction performance by dataset", fontsize=11, pad=10)

    plt.tight_layout()
    return fig


def plot_scatter_best(predictions_path: Path) -> plt.Figure:
    """Create scatter plots for best predicted dimensions."""
    setup_style()

    data = np.load(predictions_path, allow_pickle=True)

    # Best dimensions to plot
    dims_to_plot = [
        ("glasgow_valence", "Valence (Glasgow)"),
        ("warriner_valence", "Valence (Warriner)"),
        ("glasgow_concreteness", "Concreteness (Glasgow)"),
        ("brysbaert_concreteness", "Concreteness (Brysbaert)"),
        ("lancaster_haptic", "Haptic (Lancaster)"),
        ("things_pleasant", "Pleasant (THINGS)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7, 5))
    axes = axes.flatten()

    for ax, (key, title) in zip(axes, dims_to_plot):
        try:
            item = data[key].item()
            true = item["true"]
            pred = item["pred"]

            # Subsample for clarity if needed
            if len(true) > 2000:
                idx = np.random.RandomState(42).choice(len(true), 2000, replace=False)
                true, pred = true[idx], pred[idx]

            ax.scatter(true, pred, alpha=0.3, s=8, c=TEAL, edgecolor="none")

            # Add regression line
            z = np.polyfit(true, pred, 1)
            p = np.poly1d(z)
            x_line = np.linspace(true.min(), true.max(), 100)
            ax.plot(x_line, p(x_line), color=ROSE, linewidth=1.5, linestyle="--")

            # Add identity line
            lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
            ax.plot(lims, lims, color=GRAY_LIGHT, linewidth=1, linestyle=":")

            ax.set_xlabel("True rating", fontsize=8)
            ax.set_ylabel("Predicted", fontsize=8)
            ax.set_title(title, fontsize=9)
            despine(ax)

        except (KeyError, TypeError):
            ax.text(0.5, 0.5, "Data not available", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()

    plt.tight_layout()
    return fig


def main():
    # Load results
    results = pd.read_csv(OUTPUT_DIR / "prediction_results.csv")

    # Plot 1: All correlations by dataset
    fig1 = plot_correlations_by_dataset(results)
    save_figure(fig1, OUTPUT_DIR / "correlations_by_dataset.pdf")
    print("Saved: correlations_by_dataset.pdf")

    # Plot 2: Top dimensions
    fig2 = plot_top_dimensions(results)
    save_figure(fig2, OUTPUT_DIR / "top_dimensions.pdf")
    print("Saved: top_dimensions.pdf")

    # Plot 3: Dataset summary
    fig3 = plot_dataset_summary(results)
    save_figure(fig3, OUTPUT_DIR / "dataset_summary.pdf")
    print("Saved: dataset_summary.pdf")

    # Plot 4: Scatter plots for best dimensions
    predictions_path = OUTPUT_DIR / "predictions.npz"
    if predictions_path.exists():
        fig4 = plot_scatter_best(predictions_path)
        save_figure(fig4, OUTPUT_DIR / "scatter_best.pdf")
        print("Saved: scatter_best.pdf")

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
