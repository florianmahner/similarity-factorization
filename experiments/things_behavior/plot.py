"""
Create all things_behavior experiment plots.

Usage:
    poetry run python experiments/things_behavior/plot.py
    poetry run python experiments/things_behavior/plot.py --dims 66
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.colors import ROSE, TEAL, CYAN, GRAY, GRAY_LIGHT, GRAY_DARK, CYCLE
from src.utils.figure_theme import (
    create_figure,
    despine,
    save_figure,
)

PROJECT_ROOT = Path(__file__).parents[2]
THINGS_DIR = PROJECT_ROOT / "outputs/experiments/things_behavior"
DATA_DIR = THINGS_DIR / "data"
PLOT_DIR = THINGS_DIR / "plots"


def _plot_pairwise_reconstruction(df: pd.DataFrame, output_path: Path) -> None:
    """Scatter plot of pairwise dimension correlations."""
    fig, ax = create_figure("single")

    # Plot all points from all seeds
    sns.scatterplot(
        data=df,
        x="dimension",
        y="correlation",
        alpha=0.3,
        s=10,
        color=TEAL,
        edgecolor="none",
        ax=ax,
        legend=False,
    )

    ax.set_xlabel("Dimension")
    ax.set_ylabel("Best pairwise match (r)")
    ax.set_xlim(-2, df["dimension"].max() + 2)
    ax.set_ylim(-0.05, 1.05)

    despine(ax)
    save_figure(fig, output_path)


def _plot_low_data_accuracy(
    df: pd.DataFrame, accuracy_df: pd.DataFrame | None, output_path: Path
) -> None:
    """Plot accuracy vs training data percentage with baseline references."""
    fig, ax = create_figure("single")

    grouped = (
        df.groupby("data_percentage")["accuracy"].agg(["mean", "std"]).reset_index()
    )
    grouped["mean"] = grouped["mean"] * 100
    grouped["std"] = grouped["std"] * 100

    x_positions = np.linspace(0, 1, len(grouped))

    ax.errorbar(
        x_positions,
        grouped["mean"],
        yerr=grouped["std"],
        marker="o",
        markersize=5,
        color=TEAL,
        capsize=2,
        capthick=0.8,
        linewidth=1.2,
        label="SRF",
    )

    # Reference lines
    ax.axhline(
        y=33.33,
        color=GRAY_LIGHT,
        linestyle=":",
        linewidth=1,
        label="Chance",
        zorder=0,
    )
    ax.axhline(
        y=66.67,
        color=GRAY,
        linestyle=":",
        linewidth=1,
        label="Noise ceiling",
        zorder=0,
    )

    # Get VICE and SPoSE accuracies from accuracy_comparison if available
    if accuracy_df is not None:
        vice_acc = accuracy_df[accuracy_df["model"] == "VICE"]["accuracy"].mean() * 100
        spose_acc = (
            accuracy_df[accuracy_df["model"] == "SPoSE"]["accuracy"].mean() * 100
        )
        ax.axhline(
            y=vice_acc,
            color=ROSE,
            linestyle="--",
            linewidth=1,
            label=f"VICE",
            zorder=0,
        )
        ax.axhline(
            y=spose_acc,
            color=CYAN,
            linestyle="--",
            linewidth=1,
            label=f"SPoSE",
            zorder=0,
        )

    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Training data (%)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{p*100:.0f}" for p in grouped["data_percentage"]])
    ax.set_ylim(30, 68)

    despine(ax)
    ax.legend(frameon=False, fontsize=7, loc="lower right", ncol=2)

    save_figure(fig, output_path)


def _plot_lowdata_comparison(
    df: pd.DataFrame, accuracy_df: pd.DataFrame | None, output_path: Path
) -> None:
    """Plot SRF vs VICE accuracy across training data percentages.

    Same style as _plot_low_data_accuracy but with both SRF and VICE curves.
    """
    fig, ax = create_figure("single")

    percentages = sorted(df["pct"].unique())
    x_positions = np.linspace(0, 1, len(percentages))

    # SRF curve
    srf_df = df[df["model"] == "SRF"]
    srf_grouped = srf_df.groupby("pct")["val_acc"].agg(["mean", "std"]).reset_index()
    srf_grouped = srf_grouped.sort_values("pct")
    ax.errorbar(
        x_positions,
        srf_grouped["mean"] * 100,
        yerr=srf_grouped["std"] * 100,
        marker="o",
        markersize=5,
        color=TEAL,
        capsize=2,
        capthick=0.8,
        linewidth=1.2,
        label="SRF",
    )

    # VICE curve
    vice_df = df[df["model"] == "VICE"]
    vice_grouped = vice_df.groupby("pct")["val_acc"].agg(["mean", "std"]).reset_index()
    vice_grouped = vice_grouped.sort_values("pct")
    ax.errorbar(
        x_positions,
        vice_grouped["mean"] * 100,
        yerr=vice_grouped["std"] * 100,
        marker="o",
        markersize=5,
        color=ROSE,
        capsize=2,
        capthick=0.8,
        linewidth=1.2,
        label="VICE",
    )

    # Reference lines
    ax.axhline(
        y=33.33,
        color=GRAY_LIGHT,
        linestyle=":",
        linewidth=1,
        label="Chance",
        zorder=0,
    )
    ax.axhline(
        y=66.67,
        color=GRAY,
        linestyle=":",
        linewidth=1,
        label="Noise ceiling",
        zorder=0,
    )

    # SPoSE baseline (100% data)
    if accuracy_df is not None:
        spose_acc = (
            accuracy_df[accuracy_df["model"] == "SPoSE"]["accuracy"].mean() * 100
        )
        ax.axhline(
            y=spose_acc,
            color=CYAN,
            linestyle="--",
            linewidth=1,
            label="SPoSE",
            zorder=0,
        )

    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Training data (%)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{p:.0f}" for p in percentages])
    ax.set_ylim(30, 68)

    despine(ax)
    ax.legend(frameon=False, fontsize=7, loc="lower right", ncol=2)

    save_figure(fig, output_path)


def _plot_accuracy_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Bar plot comparing model accuracy and correlation."""
    df = df.copy()
    df["accuracy"] = df["accuracy"] * 100

    fig, axes = create_figure("single", ncols=2)

    models = ["SRF", "SPoSE", "VICE"]
    colors = [TEAL, CYAN, ROSE]

    # Accuracy
    ax = axes[0]
    means = [df[df["model"] == m]["accuracy"].mean() for m in models]
    stds = [df[df["model"] == m]["accuracy"].std() for m in models]
    bars = ax.bar(
        models,
        means,
        yerr=stds,
        color=colors,
        capsize=3,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axhline(y=33.33, color=GRAY_LIGHT, linestyle="--", linewidth=1)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 70)
    despine(ax)

    # Correlation
    ax = axes[1]
    means = [df[df["model"] == m]["correlation"].mean() for m in models]
    stds = [df[df["model"] == m]["correlation"].std() for m in models]
    ax.bar(
        models,
        means,
        yerr=stds,
        color=colors,
        capsize=3,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_ylabel("Correlation")
    ax.set_ylim(0.85, 0.92)
    despine(ax)

    fig.tight_layout()
    save_figure(fig, output_path)


def _plot_predicted_similarity(df: pd.DataFrame, output_path: Path) -> None:
    """Scatter plot of true vs predicted similarity."""
    # Filter for SRF model and seed 2
    srf_data = df[(df["model"] == "SRF") & (df["seed"] == 2)]

    if srf_data.empty:
        srf_data = df[df["model"] == "SRF"]
        if not srf_data.empty:
            srf_data = srf_data[srf_data["seed"] == srf_data["seed"].iloc[0]]

    if srf_data.empty:
        return

    true_sim = srf_data["true_similarity"].values
    pred_sim = srf_data["predicted_similarity"].values

    # Normalize each to [0, 1] by dividing by max
    true_norm = true_sim / true_sim.max()
    pred_norm = pred_sim / pred_sim.max()

    fig, ax = create_figure("single")

    ax.scatter(
        true_norm,
        pred_norm,
        alpha=0.6,
        color=TEAL,
        s=8,
        edgecolors="white",
        linewidth=0.3,
    )

    # Identity line
    ax.plot([-0.05, 1.05], [-0.05, 1.05], "--", color=GRAY_DARK, linewidth=1, zorder=0)

    correlation = np.corrcoef(true_norm, pred_norm)[0, 1]
    ax.text(0.05, 0.90, f"r = {correlation:.3f}", transform=ax.transAxes, fontsize=10)

    ax.set_xlabel("True similarity")
    ax.set_ylabel("Predicted similarity (SRF)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    despine(ax)
    save_figure(fig, output_path)


def _plot_dimension_reliability(df: pd.DataFrame, output_path: Path, n_runs: int = 20) -> None:
    """Bar plot of dimension reliability across random restarts."""
    fig, ax = create_figure("wide")

    # Sort by reliability
    df_sorted = df.sort_values("Reliability", ascending=False).reset_index(drop=True)

    colors = [TEAL if r >= 0.9 else GRAY for r in df_sorted["Reliability"]]

    ax.bar(
        range(len(df_sorted)),
        df_sorted["Reliability"],
        color=colors,
        edgecolor="none",
        width=0.8,
    )

    ax.axhline(y=0.9, color=ROSE, linestyle="--", linewidth=1, zorder=0)

    mean_rel = df["Reliability"].mean()
    ax.axhline(y=mean_rel, color="black", linestyle=":", linewidth=1, zorder=0)

    ax.set_xlabel("Dimension (sorted)")
    ax.set_ylabel("Reliability (r)")
    ax.set_xlim(-1, len(df_sorted))
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Dimension stability ({n_runs} restarts, mean r = {mean_rel:.2f})", fontsize=9)

    despine(ax)
    save_figure(fig, output_path)


def _plot_cross_validation_by_fraction(df: pd.DataFrame, output_path: Path) -> None:
    """Line plot of CV scores by rank, colored by observed fraction."""
    if "observed_fraction" not in df.columns:
        return

    df = df.copy()
    df["observed_fraction"] = df["observed_fraction"].round(2)

    fig, ax = create_figure("single")

    fractions = sorted(df["observed_fraction"].unique())
    for i, frac in enumerate(fractions):
        subset = df[df["observed_fraction"] == frac].sort_values("rank")
        color = CYCLE[i % len(CYCLE)]
        ax.plot(
            subset["rank"],
            subset["score"],
            marker="o",
            markersize=4,
            linewidth=1.0,
            label=f"{frac:.2f}",
            color=color,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Rank")
    ax.set_ylabel("MSE")

    despine(ax)
    ax.legend(title="Obs. fraction", frameon=False, fontsize=7, ncol=2)

    save_figure(fig, output_path)


def _plot_optimal_rank_by_fraction(df: pd.DataFrame, output_path: Path) -> None:
    """Bar plot of optimal rank for each observed fraction."""
    if "observed_fraction" not in df.columns:
        return

    df = df.copy()
    df["observed_fraction"] = df["observed_fraction"].round(2)

    optimal_ranks = (
        df.groupby("observed_fraction")
        .apply(lambda x: x.loc[x["score"].idxmin(), "rank"], include_groups=False)
        .reset_index()
    )
    optimal_ranks.columns = ["observed_fraction", "optimal_rank"]

    fig, ax = create_figure("single")

    n_bars = len(optimal_ranks)
    colors = [CYCLE[i % len(CYCLE)] for i in range(n_bars)]

    ax.bar(
        range(n_bars),
        optimal_ranks["optimal_rank"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_xlabel("Observed fraction")
    ax.set_ylabel("Optimal rank")
    ax.set_xticks(range(n_bars))
    ax.set_xticklabels(
        [f"{f:.2f}" for f in optimal_ranks["observed_fraction"]], fontsize=7
    )

    despine(ax)
    save_figure(fig, output_path)


def _plot_cross_validation(df: pd.DataFrame, output_path: Path) -> None:
    """Line plot of cross-validation scores by rank (filtered to rank >= 20)."""
    df = df.copy()
    df = df[df["rank"] >= 20]

    if df.empty:
        return

    # Average across observed fractions if present
    if "observed_fraction" in df.columns:
        df = df.groupby("rank")["score"].mean().reset_index()

    fig, ax = create_figure("single")

    ax.plot(
        df["rank"],
        df["score"],
        color="black",
        marker="o",
        linewidth=1.2,
        markersize=5,
    )

    best_rank = df.loc[df["score"].idxmin(), "rank"]

    ax.axvline(
        x=best_rank,
        color=ROSE,
        linestyle="--",
        linewidth=1,
        label=f"Best: {best_rank}",
    )

    ax.set_xlabel("Rank")
    ax.set_ylabel("MSE")

    despine(ax)
    ax.legend(frameon=False, fontsize=8)

    save_figure(fig, output_path)


def main():
    parser = argparse.ArgumentParser(description="Create things_behavior plots")
    parser.add_argument("--output-dir", type=Path, default=PLOT_DIR)
    parser.add_argument(
        "--dims",
        type=int,
        default=None,
        help="Specific dimensionality to plot (e.g., 49, 66)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dims_to_process = [args.dims] if args.dims else [49, 66]

    for dims in dims_to_process:
        dim_data_dir = DATA_DIR / str(dims)
        dim_plot_dir = output_dir / str(dims)
        dim_plot_dir.mkdir(parents=True, exist_ok=True)

        # Load accuracy comparison for reference lines
        accuracy_df = None
        acc_csv = dim_data_dir / "accuracy_comparison.csv"
        if acc_csv.exists():
            accuracy_df = pd.read_csv(acc_csv)

        # Pairwise reconstruction
        csv = dim_data_dir / "pairwise_reconstruction.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            _plot_pairwise_reconstruction(
                df, dim_plot_dir / "pairwise_reconstruction.pdf"
            )
            print(f"dims={dims}: pairwise_reconstruction")

        # Low data accuracy
        csv = dim_data_dir / "low_data.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            _plot_low_data_accuracy(
                df, accuracy_df, dim_plot_dir / "low_data_accuracy.pdf"
            )
            print(f"dims={dims}: low_data_accuracy")

        # Accuracy comparison
        if accuracy_df is not None:
            _plot_accuracy_comparison(
                accuracy_df, dim_plot_dir / "accuracy_comparison.pdf"
            )
            print(f"dims={dims}: accuracy_comparison")

        # Predicted similarity (48 performance)
        csv = dim_data_dir / "48_performance.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            _plot_predicted_similarity(df, dim_plot_dir / "48_predicted.pdf")
            print(f"dims={dims}: 48_predicted")

        # Dimension reliability
        csv = dim_data_dir / "dimension_reliability.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            _plot_dimension_reliability(
                df, dim_plot_dir / "dimension_reliability.pdf", n_runs=20
            )
            print(f"dims={dims}: dimension_reliability")

    # Low-data comparison (SRF vs VICE)
    csv = DATA_DIR / "lowdata_comparison.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        # Load accuracy_comparison for SPoSE baseline (use dims=66)
        acc_csv = DATA_DIR / "66" / "accuracy_comparison.csv"
        acc_df = pd.read_csv(acc_csv) if acc_csv.exists() else None
        _plot_lowdata_comparison(df, acc_df, output_dir / "lowdata_comparison.pdf")
        print("lowdata_comparison: SRF vs VICE")

    # Cross-validation plots (not per-dims)
    csv = DATA_DIR / "spose_cross_validation.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        _plot_cross_validation_by_fraction(
            df, output_dir / "cross_validation_by_fraction.pdf"
        )
        _plot_optimal_rank_by_fraction(df, output_dir / "optimal_rank_by_fraction.pdf")
        _plot_cross_validation(df, output_dir / "cross_validation.pdf")
        print("cross_validation: 3 plots")

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
