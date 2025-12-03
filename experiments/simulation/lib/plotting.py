"""Plotting utilities for simulation interpretability analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save_figure(fig, output_dir: Path, filename: str) -> None:
    """Save figure as PDF with consistent settings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)


def create_cophenetic_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Create cophenetic correlation vs difficulty plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

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

    ax.set_xlabel("Cluster Overlap (Entropy)", fontsize=13)
    ax.set_ylabel("Cophenetic Correlation", fontsize=13)
    ax.set_title("Cluster Stability vs Difficulty", fontsize=14, fontweight="bold")
    ax.legend(title="SNR", frameon=False, fontsize=11)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_ylim([0.8, 1.01])
    sns.despine()

    plt.tight_layout()
    _save_figure(fig, output_dir, "cophenetic_correlation.pdf")


def create_consensus_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Create consensus dispersion plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

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

    ax.set_xlabel("Cluster Overlap (Entropy)", fontsize=13)
    ax.set_ylabel("Consensus Dispersion", fontsize=13)
    ax.set_title("Consensus Stability", fontsize=14, fontweight="bold")
    ax.legend(title="SNR", frameon=False, fontsize=11)
    ax.grid(alpha=0.3, linestyle="--")
    sns.despine()

    plt.tight_layout()
    _save_figure(fig, output_dir, "consensus_dispersion.pdf")


def create_reconstruction_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Create reconstruction fidelity plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

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

    ax.set_xlabel("Cluster Overlap (Entropy)", fontsize=13)
    ax.set_ylabel("Reconstruction $R^2$", fontsize=13)
    ax.set_title("Reconstruction Fidelity", fontsize=14, fontweight="bold")
    ax.legend(title="SNR", frameon=False, fontsize=11)
    ax.grid(alpha=0.3, linestyle="--")
    sns.despine()

    plt.tight_layout()
    _save_figure(fig, output_dir, "reconstruction_fidelity.pdf")


def create_kl_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Create KL divergence (factor recovery error) plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

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

    ax.set_xlabel("Cluster Overlap (Entropy)", fontsize=13)
    ax.set_ylabel("KL Divergence", fontsize=13)
    ax.set_title("Factor Recovery Error", fontsize=14, fontweight="bold")
    ax.legend(title="SNR", frameon=False, fontsize=11)
    ax.grid(alpha=0.3, linestyle="--")
    sns.despine()

    plt.tight_layout()
    _save_figure(fig, output_dir, "kl_divergence.pdf")


def create_entropy_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Create factor entropy plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

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

    ax.set_xlabel("Cluster Overlap (Entropy)", fontsize=13)
    ax.set_ylabel("Factor Entropy", fontsize=13)
    ax.set_title("Factor Interpretability", fontsize=14, fontweight="bold")
    ax.legend(title="SNR", frameon=False, fontsize=11)
    ax.grid(alpha=0.3, linestyle="--")
    sns.despine()

    plt.tight_layout()
    _save_figure(fig, output_dir, "factor_entropy.pdf")


def create_sparsity_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Create factor sparsity plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

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

    ax.set_xlabel("Cluster Overlap (Entropy)", fontsize=13)
    ax.set_ylabel("Factor Sparsity", fontsize=13)
    ax.set_title("Factor Sparsity", fontsize=14, fontweight="bold")
    ax.legend(title="SNR", frameon=False, fontsize=11)
    ax.grid(alpha=0.3, linestyle="--")
    sns.despine()

    plt.tight_layout()
    _save_figure(fig, output_dir, "factor_sparsity.pdf")


def create_imputation_r2_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Create imputation R² vs fraction retained plot."""
    import numpy as np

    # Aggregate by fraction_retained and method
    agg_df = (
        df.groupby(["fraction_retained", "method"])["r2"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Convert to percentage
    agg_df["fraction_retained_pct"] = agg_df["fraction_retained"] * 100

    # Filter out Mean method
    agg_df = agg_df[agg_df["method"] != "Mean"]

    # Define colors
    colors = {
        "Median": "#E74C3C",
        "KNN": "#3498DB",
        "SRF": "#2ECC71",
    }

    fig, ax = plt.subplots(figsize=(4, 3))

    # Get unique fractions and create x positions
    unique_fractions = sorted(agg_df["fraction_retained_pct"].unique())
    x_positions = np.arange(len(unique_fractions))

    # Plot each method
    for method in agg_df["method"].unique():
        data = agg_df[agg_df["method"] == method].sort_values("fraction_retained_pct")

        ax.plot(
            x_positions,
            data["mean"],
            label=method,
            color=colors.get(method, "#95A5A6"),
            marker="o",
            linewidth=2,
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.5,
            markeredgecolor=colors.get(method, "#95A5A6"),
        )

        # Add shaded error region
        ax.fill_between(
            x_positions,
            data["mean"] - data["std"],
            data["mean"] + data["std"],
            color=colors.get(method, "#95A5A6"),
            alpha=0.15,
        )

    # Format x-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{int(x)}%" for x in unique_fractions])
    ax.set_xlim(-0.5, len(unique_fractions) - 0.5)

    ax.set_xlabel("Percentage retained", fontsize=11)
    ax.set_ylabel("Validation R²", fontsize=11)
    ax.legend(frameon=False)
    sns.despine(ax=ax)

    plt.tight_layout()
    _save_figure(fig, output_dir, "imputation_r2.pdf")


def create_rank_detection_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Create rank detection scatter plot with jitter."""
    import numpy as np

    # Add jitter
    rng = np.random.default_rng(42)
    jitter_amount = 0.8
    df = df.copy()
    df["true_rank_jittered"] = df["true_rank"] + rng.uniform(
        -jitter_amount, jitter_amount, size=len(df)
    )
    df["selected_rank_jittered"] = df["selected_rank"] + rng.uniform(
        -jitter_amount, jitter_amount, size=len(df)
    )

    fig, ax = plt.subplots(figsize=(4, 3))

    # Define markers and colors for each SNR
    snr_values = sorted(df["snr"].unique())
    markers = ["o", "s", "^", "D", "v", "P"]
    colors = sns.color_palette("tab10", n_colors=len(snr_values))

    # Plot each SNR with different marker
    for i, snr in enumerate(snr_values):
        subset = df[df["snr"] == snr]
        ax.scatter(
            subset["true_rank_jittered"],
            subset["selected_rank_jittered"],
            s=25,
            alpha=0.6,
            marker=markers[i % len(markers)],
            color=colors[i],
            label=f"{snr}",
            edgecolors="white",
            linewidths=0.3,
        )

    # Add diagonal reference line
    min_rank = df["true_rank"].min()
    max_rank = df["true_rank"].max()
    ax.plot(
        [min_rank, max_rank],
        [min_rank, max_rank],
        linestyle="--",
        color="gray",
        linewidth=1.0,
        zorder=0,
    )

    ax.set_xlabel("True rank", fontsize=11)
    ax.set_ylabel("Selected rank", fontsize=11)
    ax.set_xlim(min_rank - 1, max_rank + 1)
    ax.set_ylim(min_rank - 1, max_rank + 1)

    ax.legend(title="SNR", loc="upper left", frameon=False, fontsize=9, title_fontsize=10)
    sns.despine(ax=ax, top=True, right=True)

    plt.tight_layout()
    _save_figure(fig, output_dir, "rank_detection.pdf")
