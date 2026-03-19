"""Create optimal visualizations for rank detection.

Produces publication-quality figures showing:
1. The fundamental challenge (score gap vs rank)
2. Best accuracy display (heatmap + line)
3. Detection errors clearly visible
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "figure.dpi": 150,
})


def save_figure(fig, path: Path, filename: str) -> None:
    fig.savefig(path / filename, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {filename}")


def plot_main_figure(df: pd.DataFrame, output_dir: Path) -> None:
    """Main figure: 2-panel showing accuracy and discriminability."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    ax = axes[0]
    summary = df.groupby(["true_rank", "snr"])["is_correct"].mean().reset_index()

    for snr in sorted(df["snr"].unique()):
        subset = summary[summary["snr"] == snr].sort_values("true_rank")
        ax.plot(subset["true_rank"], subset["is_correct"],
                marker="o", markersize=5, label=f"SNR={snr}", lw=1.5)

    ax.set_xlabel("True Rank")
    ax.set_ylabel("Detection Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(3, 32)
    ax.legend(frameon=False, loc="lower left")
    ax.axhline(0.5, color="gray", linestyle="--", lw=0.8, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("(a) Rank Detection Accuracy", loc="left", fontweight="bold")

    ax = axes[1]
    gap_summary = df.groupby(["true_rank", "snr"])["score_gap"].agg(["median", "std"]).reset_index()

    for snr in sorted(df["snr"].unique()):
        subset = gap_summary[gap_summary["snr"] == snr].sort_values("true_rank")
        ax.semilogy(subset["true_rank"], subset["median"],
                    marker="s", markersize=4, label=f"SNR={snr}", lw=1.5)

    ax.set_xlabel("True Rank")
    ax.set_ylabel("Score Gap (log scale)")
    ax.set_xlim(3, 32)
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("(b) CV Score Discriminability", loc="left", fontweight="bold")

    plt.tight_layout()
    save_figure(fig, output_dir, "rank_detection_main.pdf")


def plot_accuracy_heatmap_clean(df: pd.DataFrame, output_dir: Path) -> None:
    """Clean accuracy heatmap."""
    summary = df.groupby(["true_rank", "snr"])["is_correct"].mean().reset_index()
    pivot = summary.pivot(index="true_rank", columns="snr", values="is_correct")
    pivot = pivot.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(3.5, 3))

    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(
        pivot, ax=ax, cmap=cmap, vmin=0, vmax=1,
        annot=True, fmt=".0%", annot_kws={"size": 8, "weight": "bold"},
        cbar_kws={"label": "Accuracy", "shrink": 0.8},
        linewidths=0.5, linecolor="white"
    )

    ax.set_xlabel("SNR")
    ax.set_ylabel("True Rank")

    plt.tight_layout()
    save_figure(fig, output_dir, "accuracy_heatmap_clean.pdf")


def plot_scatter_publication(df: pd.DataFrame, output_dir: Path) -> None:
    """Publication-quality scatter plot."""
    rng = np.random.default_rng(42)
    jitter = 0.5
    df = df.copy()
    df["x"] = df["true_rank"] + rng.uniform(-jitter, jitter, len(df))
    df["y"] = df["selected_rank"] + rng.uniform(-jitter, jitter, len(df))

    fig, ax = plt.subplots(figsize=(4, 3.5))

    correct = df[df["is_correct"]]
    errors = df[~df["is_correct"]]

    ax.scatter(correct["x"], correct["y"], c="#2ecc71", alpha=0.5, s=20,
               label="Correct", edgecolors="none")
    ax.scatter(errors["x"], errors["y"], c="#e74c3c", alpha=0.7, s=25,
               label="Error", edgecolors="white", linewidths=0.3, marker="X")

    min_r, max_r = df["true_rank"].min(), df["true_rank"].max()
    ax.plot([min_r-2, max_r+2], [min_r-2, max_r+2], "--", color="gray", lw=1, zorder=0)

    ax.fill_between([min_r-2, max_r+2], [min_r-7, max_r-3], [min_r+3, max_r+7],
                    color="gray", alpha=0.1, zorder=0, label="±5 tolerance")

    ax.set_xlabel("True Rank")
    ax.set_ylabel("Selected Rank")
    ax.set_xlim(min_r - 2, max_r + 2)
    ax.set_ylim(min_r - 8, max_r + 12)
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_figure(fig, output_dir, "scatter_publication.pdf")


def plot_boxplot_clean(df: pd.DataFrame, output_dir: Path) -> None:
    """Clean boxplot faceted by SNR."""
    fig, axes = plt.subplots(2, 2, figsize=(7, 5), sharex=True)
    axes = axes.flatten()
    snrs = sorted(df["snr"].unique())

    palette = sns.color_palette("Blues", n_colors=6)

    for ax, snr in zip(axes, snrs):
        subset = df[df["snr"] == snr]

        positions = sorted(subset["true_rank"].unique())
        data = [subset[subset["true_rank"] == tr]["selected_rank"].values for tr in positions]

        bp = ax.boxplot(data, positions=range(len(positions)), patch_artist=True,
                        widths=0.6, showfliers=True)

        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(palette[i])
            patch.set_edgecolor("black")
            patch.set_linewidth(0.5)

        ax.plot(range(len(positions)), positions, "r--", lw=1.5, label="Ideal", zorder=5)

        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(positions)
        ax.set_title(f"SNR = {snr}", fontsize=10)

        if ax in [axes[0], axes[2]]:
            ax.set_ylabel("Selected Rank")
        if ax in [axes[2], axes[3]]:
            ax.set_xlabel("True Rank")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_figure(fig, output_dir, "boxplot_clean.pdf")


def plot_error_by_rank(df: pd.DataFrame, output_dir: Path) -> None:
    """Show mean absolute error by rank with confidence intervals."""
    summary = df.groupby("true_rank").agg(
        accuracy=("is_correct", "mean"),
        mae=("abs_error", "mean"),
        mae_std=("abs_error", "std"),
        n=("abs_error", "count")
    ).reset_index()

    summary["mae_sem"] = summary["mae_std"] / np.sqrt(summary["n"])

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.bar(summary["true_rank"], summary["mae"], yerr=summary["mae_sem"],
           capsize=3, color="steelblue", edgecolor="black", linewidth=0.5, alpha=0.8)

    ax.set_xlabel("True Rank")
    ax.set_ylabel("Mean Absolute Error")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, row in summary.iterrows():
        ax.text(row["true_rank"], row["mae"] + row["mae_sem"] + 0.1,
                f"{row['accuracy']:.0%}", ha="center", fontsize=7, color="gray")

    plt.tight_layout()
    save_figure(fig, output_dir, "error_by_rank.pdf")


def main():
    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(exist_ok=True)

    csv_path = Path("/LOCAL/fmahner/similarity-factorization/outputs/experiments/simulation/rank_detection/rank_detection_results.csv")

    if not csv_path.exists():
        log.error(f"Results file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    log.info(f"Loaded {len(df)} results")

    plot_main_figure(df, output_dir)
    plot_accuracy_heatmap_clean(df, output_dir)
    plot_scatter_publication(df, output_dir)
    plot_boxplot_clean(df, output_dir)
    plot_error_by_rank(df, output_dir)

    overall_acc = df["is_correct"].mean()
    low_rank_acc = df[df["true_rank"] <= 10]["is_correct"].mean()
    high_rank_acc = df[df["true_rank"] >= 25]["is_correct"].mean()

    log.info("\n=== Summary ===")
    log.info(f"Overall accuracy: {overall_acc:.1%}")
    log.info(f"Low rank (≤10): {low_rank_acc:.1%}")
    log.info(f"High rank (≥25): {high_rank_acc:.1%}")


if __name__ == "__main__":
    main()
