"""
Standalone plotting script for simulation experiments.

Loads CSV outputs from each task and creates publication-quality plots.

Usage:
    poetry run python experiments/simulation/tasks/plot.py
    poetry run python experiments/simulation/tasks/plot.py --output-dir plots/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SIMULATION_OUTPUTS = PROJECT_ROOT / "experiments/simulation/outputs"


def _save_figure(fig, output_dir: Path, filename: str) -> None:
    """Save figure as PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Interpretability plots - scatter plots (x: cluster_overlap, y: metric)
# =============================================================================


def plot_cophenetic_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter: Cophenetic correlation vs cluster overlap."""
    fig, ax = plt.subplots(figsize=(5, 4))

    sns.scatterplot(
        data=df,
        x="cluster_overlap",
        y="cophenetic_corr",
        hue="snr",
        size="snr",
        sizes=(60, 250),
        palette="viridis",
        alpha=0.75,
        ax=ax,
    )

    ax.set_xlabel("Cluster Overlap (log α)", fontsize=11)
    ax.set_ylabel("Cophenetic Correlation", fontsize=11)
    ax.legend(title="SNR", frameon=False, fontsize=9)
    ax.set_ylim([0.8, 1.01])
    sns.despine()

    plt.tight_layout()
    _save_figure(fig, output_dir, "cophenetic_correlation.pdf")


def plot_consensus_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter: Consensus dispersion vs cluster overlap."""
    fig, ax = plt.subplots(figsize=(5, 4))

    sns.scatterplot(
        data=df,
        x="cluster_overlap",
        y="consensus_dispersion",
        hue="snr",
        size="snr",
        sizes=(60, 250),
        palette="plasma",
        alpha=0.75,
        ax=ax,
    )

    ax.set_xlabel("Cluster Overlap (log α)", fontsize=11)
    ax.set_ylabel("Consensus Dispersion", fontsize=11)
    ax.legend(title="SNR", frameon=False, fontsize=9)
    sns.despine()

    plt.tight_layout()
    _save_figure(fig, output_dir, "consensus_dispersion.pdf")


def plot_reconstruction_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter: Reconstruction R2 vs cluster overlap."""
    fig, ax = plt.subplots(figsize=(5, 4))

    sns.scatterplot(
        data=df,
        x="cluster_overlap",
        y="reconstruction_r2",
        hue="snr",
        size="snr",
        sizes=(60, 250),
        palette="viridis",
        alpha=0.75,
        ax=ax,
    )

    ax.set_xlabel("Cluster Overlap (log α)", fontsize=11)
    ax.set_ylabel(r"Reconstruction $R^2$", fontsize=11)
    ax.legend(title="SNR", frameon=False, fontsize=9)
    sns.despine()

    plt.tight_layout()
    _save_figure(fig, output_dir, "reconstruction_fidelity.pdf")


def plot_kl_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter: KL divergence vs cluster overlap."""
    fig, ax = plt.subplots(figsize=(5, 4))

    sns.scatterplot(
        data=df,
        x="cluster_overlap",
        y="kl_mean",
        hue="snr",
        size="snr",
        sizes=(60, 250),
        palette="magma",
        alpha=0.75,
        ax=ax,
    )

    ax.set_xlabel("Cluster Overlap (log α)", fontsize=11)
    ax.set_ylabel("KL Divergence", fontsize=11)
    ax.legend(title="SNR", frameon=False, fontsize=9)
    sns.despine()

    plt.tight_layout()
    _save_figure(fig, output_dir, "kl_divergence.pdf")


def plot_entropy_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter: Factor entropy vs cluster overlap."""
    fig, ax = plt.subplots(figsize=(5, 4))

    sns.scatterplot(
        data=df,
        x="cluster_overlap",
        y="entropy_mean",
        hue="snr",
        size="snr",
        sizes=(60, 250),
        palette="viridis",
        alpha=0.75,
        ax=ax,
    )

    ax.set_xlabel("Cluster Overlap (log α)", fontsize=11)
    ax.set_ylabel("Factor Entropy", fontsize=11)
    ax.legend(title="SNR", frameon=False, fontsize=9)
    sns.despine()

    plt.tight_layout()
    _save_figure(fig, output_dir, "factor_entropy.pdf")


def plot_sparsity_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter: Factor sparsity vs cluster overlap."""
    fig, ax = plt.subplots(figsize=(5, 4))

    sns.scatterplot(
        data=df,
        x="cluster_overlap",
        y="sparsity_mean",
        hue="snr",
        size="snr",
        sizes=(60, 250),
        palette="cividis",
        alpha=0.75,
        ax=ax,
    )

    ax.set_xlabel("Cluster Overlap (log α)", fontsize=11)
    ax.set_ylabel("Factor Sparsity", fontsize=11)
    ax.legend(title="SNR", frameon=False, fontsize=9)
    sns.despine()

    plt.tight_layout()
    _save_figure(fig, output_dir, "factor_sparsity.pdf")


# =============================================================================
# Interpretability plots - heatmaps
# =============================================================================


def plot_reconstruction_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap: (alpha, SNR) -> Reconstruction R2."""
    fig, ax = plt.subplots(figsize=(4, 3.5))

    pivot = df.pivot(index="alpha", columns="snr", values="reconstruction_r2")
    pivot = pivot.sort_index(ascending=False)

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        cbar_kws={"label": "$R^2$", "shrink": 0.8},
    )

    ax.set_xlabel("SNR", fontsize=11)
    ax.set_ylabel(r"$\alpha$ (Dirichlet)", fontsize=11)

    plt.tight_layout()
    _save_figure(fig, output_dir, "reconstruction_r2_heatmap.pdf")


def plot_factor_recovery_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap: (alpha, SNR) -> Factor recovery (1 - KL)."""
    fig, ax = plt.subplots(figsize=(4, 3.5))

    pivot = df.pivot(index="alpha", columns="snr", values="kl_mean")
    recovery = 1 - pivot
    recovery = recovery.sort_index(ascending=False)

    sns.heatmap(
        recovery,
        ax=ax,
        cmap="viridis",
        vmin=0.9,
        vmax=1.0,
        annot=True,
        fmt=".3f",
        annot_kws={"size": 7},
        cbar_kws={"label": "1 - KL", "shrink": 0.8},
    )

    ax.set_xlabel("SNR", fontsize=11)
    ax.set_ylabel(r"$\alpha$ (Dirichlet)", fontsize=11)

    plt.tight_layout()
    _save_figure(fig, output_dir, "factor_recovery_heatmap.pdf")


def plot_cophenetic_lineplot(df: pd.DataFrame, output_dir: Path) -> None:
    """Line plot: Cophenetic correlation vs alpha, colored by SNR."""
    fig, ax = plt.subplots(figsize=(4, 3))

    sns.lineplot(
        data=df,
        x="alpha",
        y="cophenetic_corr",
        hue="snr",
        marker="o",
        ax=ax,
        palette="viridis",
        markersize=5,
    )

    ax.set_xlabel(r"$\alpha$ (Dirichlet)", fontsize=11)
    ax.set_ylabel("Cophenetic Correlation", fontsize=11)
    ax.set_xscale("log")
    ax.legend(title="SNR", frameon=False, fontsize=9)
    sns.despine()

    plt.tight_layout()
    _save_figure(fig, output_dir, "cophenetic_vs_alpha.pdf")


# =============================================================================
# Rank detection plots
# =============================================================================


def plot_rank_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter: True rank vs Selected rank (jittered, by SNR)."""
    rng = np.random.default_rng(42)
    jitter = 0.8
    df = df.copy()
    df["true_rank_j"] = df["true_rank"] + rng.uniform(-jitter, jitter, len(df))
    df["selected_rank_j"] = df["selected_rank"] + rng.uniform(-jitter, jitter, len(df))

    fig, ax = plt.subplots(figsize=(4, 3))

    snr_values = sorted(df["snr"].unique())
    markers = ["o", "s", "^", "D", "v", "P"]
    colors = sns.color_palette("tab10", n_colors=len(snr_values))

    for i, snr in enumerate(snr_values):
        subset = df[df["snr"] == snr]
        ax.scatter(
            subset["true_rank_j"],
            subset["selected_rank_j"],
            s=25,
            alpha=0.6,
            marker=markers[i % len(markers)],
            color=colors[i],
            label=f"{snr}",
            edgecolors="white",
            linewidths=0.3,
        )

    min_r, max_r = df["true_rank"].min(), df["true_rank"].max()
    ax.plot([min_r, max_r], [min_r, max_r], "--", color="gray", lw=1, zorder=0)

    ax.set_xlabel("True rank", fontsize=11)
    ax.set_ylabel("Selected rank", fontsize=11)
    ax.legend(title="SNR", loc="upper left", frameon=False, fontsize=9)
    sns.despine()

    plt.tight_layout()
    _save_figure(fig, output_dir, "rank_detection_scatter.pdf")


def plot_rank_accuracy(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart: Rank detection accuracy by SNR."""
    fig, ax = plt.subplots(figsize=(4, 3))

    accuracy = df.groupby("snr")["is_correct"].mean()
    ax.bar([str(x) for x in accuracy.index], accuracy.values, color="steelblue")

    ax.set_xlabel("SNR", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1)
    sns.despine()

    plt.tight_layout()
    _save_figure(fig, output_dir, "rank_accuracy_by_snr.pdf")


# =============================================================================
# Imputation plots
# =============================================================================


def plot_imputation_r2(df: pd.DataFrame, output_dir: Path) -> None:
    """Line plot: R2 vs fraction retained by method."""
    agg = (
        df.groupby(["fraction_retained", "method"])["r2"]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg["pct"] = agg["fraction_retained"] * 100
    agg = agg[agg["method"] != "Mean"]

    colors = {"Median": "#E74C3C", "KNN": "#3498DB", "SRF": "#2ECC71"}

    fig, ax = plt.subplots(figsize=(4, 3))

    fractions = sorted(agg["pct"].unique())
    x = np.arange(len(fractions))

    for method in agg["method"].unique():
        data = agg[agg["method"] == method].sort_values("pct")
        c = colors.get(method, "#95A5A6")
        ax.plot(x, data["mean"], label=method, color=c, marker="o", lw=2,
                markersize=7, markerfacecolor="white", markeredgewidth=1.5,
                markeredgecolor=c)
        ax.fill_between(x, data["mean"] - data["std"], data["mean"] + data["std"],
                        color=c, alpha=0.15)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(f)}%" for f in fractions])
    ax.set_xlabel("Percentage retained", fontsize=11)
    ax.set_ylabel(r"Validation $R^2$", fontsize=11)
    ax.legend(frameon=False)
    sns.despine()

    plt.tight_layout()
    _save_figure(fig, output_dir, "imputation_r2.pdf")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Create simulation plots from CSV data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SIMULATION_OUTPUTS / "plots",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    sns.set_theme(style="whitegrid", font_scale=1.0)

    # Interpretability plots
    interp_csv = SIMULATION_OUTPUTS / "interpretability/interpretability_summary.csv"
    if interp_csv.exists():
        print(f"Loading {interp_csv}")
        df_interp = pd.read_csv(interp_csv)
        # Scatter plots (x: cluster_overlap, y: metric)
        plot_cophenetic_scatter(df_interp, output_dir)
        plot_consensus_scatter(df_interp, output_dir)
        plot_reconstruction_scatter(df_interp, output_dir)
        plot_kl_scatter(df_interp, output_dir)
        plot_entropy_scatter(df_interp, output_dir)
        plot_sparsity_scatter(df_interp, output_dir)
        # Heatmaps
        plot_reconstruction_heatmap(df_interp, output_dir)
        plot_factor_recovery_heatmap(df_interp, output_dir)
        plot_cophenetic_lineplot(df_interp, output_dir)
        print("  Created: 6 scatter plots + 3 heatmaps/lineplots")
    else:
        print(f"Skipping interpretability (no data at {interp_csv})")

    # Rank detection plots
    rank_csv = SIMULATION_OUTPUTS / "rank_detection/rank_detection_results.csv"
    if rank_csv.exists():
        print(f"Loading {rank_csv}")
        df_rank = pd.read_csv(rank_csv)
        plot_rank_scatter(df_rank, output_dir)
        plot_rank_accuracy(df_rank, output_dir)
        print("  Created: rank_detection_scatter.pdf, rank_accuracy_by_snr.pdf")
    else:
        print(f"Skipping rank detection (no data at {rank_csv})")

    # Imputation plots
    impute_csv = SIMULATION_OUTPUTS / "imputation/imputation_results.csv"
    if impute_csv.exists():
        print(f"Loading {impute_csv}")
        df_impute = pd.read_csv(impute_csv)
        plot_imputation_r2(df_impute, output_dir)
        print("  Created: imputation_r2.pdf")
    else:
        print(f"Skipping imputation (no data at {impute_csv})")

    print(f"\nDone. Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
