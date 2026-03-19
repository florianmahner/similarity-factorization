"""Visualization for rank selection analysis.

Usage:
    poetry run python sandbox/rank_selection_analysis/plot.py <results_dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.colors import CYCLE, GRAY
from src.utils.figure_theme import create_figure, despine, save_figure


def plot_main_results(df: pd.DataFrame, output_dir: Path):
    """Figure 1: Main accuracy results."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # (a) Heatmap: accuracy by obs_dof bin × true_k
    ax = axes[0, 0]
    pivot = df.pivot_table(values="is_correct", index="obs_dof_bin", columns="true_k", aggfunc="mean")
    sns.heatmap(pivot, annot=True, fmt=".0%", cmap="RdYlGn", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Accuracy"})
    ax.set_title("(a) Accuracy by obs/dof and true rank")
    ax.set_xlabel("True rank (k)")
    ax.set_ylabel("obs/dof")

    # (b) Line plot: accuracy vs obs_dof by true_k
    ax = axes[0, 1]
    for i, k in enumerate(sorted(df["true_k"].unique())):
        subset = df[df["true_k"] == k]
        grouped = subset.groupby("obs_dof_bin")["is_correct"].agg(["mean", "std", "count"])
        x = range(len(grouped))
        ax.errorbar(x, grouped["mean"], yerr=grouped["std"] / np.sqrt(grouped["count"]),
                    marker="o", label=f"k={k}", color=CYCLE[i % len(CYCLE)], linewidth=2, markersize=8)
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped.index)
    ax.set_xlabel("obs/dof")
    ax.set_ylabel("Accuracy")
    ax.set_title("(b) Accuracy vs obs/dof")
    ax.legend()
    ax.axhline(0.8, color=GRAY, linestyle="--", alpha=0.7)
    ax.set_ylim(-0.05, 1.05)
    despine(ax)

    # (c) Box plot: selection error by obs_dof bin
    ax = axes[1, 0]
    df_plot = df[df["selection_error"] <= 6]  # exclude outliers for viz
    sns.boxplot(data=df_plot, x="obs_dof_bin", y="selection_error", hue="true_k", ax=ax, palette=[CYCLE[i % len(CYCLE)] for i in range(3)])
    ax.set_xlabel("obs/dof")
    ax.set_ylabel("Selection error |selected - true|")
    ax.set_title("(c) Selection error distribution")
    ax.legend(title="true k")
    despine(ax)

    # (d) Scatter: obs_dof vs selection error
    ax = axes[1, 1]
    for i, k in enumerate(sorted(df["true_k"].unique())):
        subset = df[df["true_k"] == k]
        ax.scatter(subset["obs_dof"], subset["selection_error"], alpha=0.5, s=30,
                   label=f"k={k}", color=CYCLE[i % len(CYCLE)])
    ax.set_xlabel("obs/dof")
    ax.set_ylabel("Selection error")
    ax.set_title("(d) Selection error vs obs/dof")
    ax.axvline(5, color="red", linestyle="--", label="obs/dof=5")
    ax.legend()
    despine(ax)

    plt.tight_layout()
    save_figure(fig, output_dir / "fig1_main_results.pdf")
    print(f"Saved: fig1_main_results.pdf")


def plot_cv_curves(df: pd.DataFrame, output_dir: Path):
    """Figure 2: CV score curves for different regimes."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    true_ks = sorted(df["true_k"].unique())
    obs_dof_ranges = [("<3", lambda x: x < 3), ("3-5", lambda x: (x >= 3) & (x < 5)), (">5", lambda x: x >= 5)]

    for row, k in enumerate(true_ks):
        for col, (label, condition) in enumerate(obs_dof_ranges):
            ax = axes[row, col]

            # Get subset
            subset = df[(df["true_k"] == k) & condition(df["obs_dof"])]

            if len(subset) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"k={k}, obs/dof {label}")
                continue

            # Aggregate scores across seeds
            all_scores = {}
            for _, row_data in subset.iterrows():
                scores = row_data["scores"]
                for rank, score in scores.items():
                    if rank not in all_scores:
                        all_scores[rank] = []
                    all_scores[rank].append(score)

            ranks = sorted(all_scores.keys())
            means = [np.mean(all_scores[r]) for r in ranks]
            stds = [np.std(all_scores[r]) for r in ranks]

            # Normalize
            min_mean, max_mean = min(means), max(means)
            means_norm = [(m - min_mean) / (max_mean - min_mean + 1e-10) for m in means]
            stds_norm = [s / (max_mean - min_mean + 1e-10) for s in stds]

            ax.errorbar(ranks, means_norm, yerr=stds_norm, fmt="o-", capsize=4,
                        color="steelblue", markersize=8, linewidth=2)
            ax.axvline(k, color="red", linestyle="--", linewidth=2, label=f"True k={k}")

            # Mark selected (mode)
            selected = subset["selected_rank"].mode().iloc[0] if len(subset) > 0 else k
            ax.axvline(selected, color="green", linestyle=":", linewidth=2, label=f"Selected={selected}")

            accuracy = subset["is_correct"].mean()
            ax.set_title(f"k={k}, obs/dof {label}\nAccuracy: {accuracy:.0%}")
            ax.set_xlabel("Rank")
            ax.set_ylabel("Normalized CV Score")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.suptitle("CV Score Curves by True Rank and obs/dof Regime", fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir / "fig2_cv_curves.pdf")
    print(f"Saved: fig2_cv_curves.pdf")


def plot_eigenvalues(df: pd.DataFrame, output_dir: Path):
    """Figure 3: Eigenvalue spectra."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Generate example matrices for each k
    from sandbox.rank_selection_analysis.run import simulation_block

    n = 500  # representative size
    for idx, k in enumerate([10, 20, 30]):
        ax = axes[idx]

        S = simulation_block(n, k, seed=42)
        eigenvalues = np.linalg.eigvalsh(S)[::-1]

        # Plot eigenvalues
        x = np.arange(min(50, len(eigenvalues)))
        ax.bar(x, eigenvalues[:len(x)], color="steelblue", alpha=0.7)
        ax.axvline(k - 0.5, color="red", linestyle="--", linewidth=2, label=f"True k={k}")

        gap = eigenvalues[k - 1] - eigenvalues[k] if k < len(eigenvalues) else 0
        ax.set_title(f"k={k}\nSpectral gap: {gap:.1f}")
        ax.set_xlabel("Component")
        ax.set_ylabel("Eigenvalue")
        ax.legend()
        despine(ax)

    plt.suptitle(f"Eigenvalue Spectra (n={n})", fontsize=12, y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir / "fig3_eigenvalues.pdf")
    print(f"Saved: fig3_eigenvalues.pdf")


def plot_sharpness(df: pd.DataFrame, output_dir: Path):
    """Figure 4: Score ratio (CV curve sharpness) analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # (a) Scatter: score_ratio vs obs_dof, color = accuracy
    ax = axes[0]
    df_valid = df[df["score_ratio"].notna() & (df["score_ratio"] < 100)]

    scatter = ax.scatter(df_valid["obs_dof"], df_valid["score_ratio"],
                         c=df_valid["is_correct"].astype(int), cmap="RdYlGn",
                         alpha=0.6, s=40, vmin=0, vmax=1)
    ax.set_xlabel("obs/dof")
    ax.set_ylabel("Score ratio: err(k-2) / err(k)")
    ax.set_title("(a) CV Curve Sharpness vs obs/dof")
    ax.axhline(3, color="gray", linestyle="--", alpha=0.7, label="Ratio=3")
    ax.axvline(5, color="red", linestyle="--", alpha=0.7, label="obs/dof=5")
    ax.legend()
    plt.colorbar(scatter, ax=ax, label="Correct")
    despine(ax)

    # (b) Line: mean score_ratio vs obs_dof by k
    ax = axes[1]
    for i, k in enumerate(sorted(df["true_k"].unique())):
        subset = df_valid[df_valid["true_k"] == k]
        grouped = subset.groupby("obs_dof_bin")["score_ratio"].mean()
        ax.plot(range(len(grouped)), grouped.values, marker="o", label=f"k={k}",
                color=CYCLE[i % len(CYCLE)], linewidth=2, markersize=8)
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped.index)
    ax.set_xlabel("obs/dof")
    ax.set_ylabel("Mean score ratio")
    ax.set_title("(b) Mean CV Curve Sharpness")
    ax.axhline(3, color="gray", linestyle="--", alpha=0.7)
    ax.legend()
    despine(ax)

    plt.tight_layout()
    save_figure(fig, output_dir / "fig4_sharpness.pdf")
    print(f"Saved: fig4_sharpness.pdf")


def plot_recommendation(df: pd.DataFrame, output_dir: Path):
    """Figure 5: Summary recommendation plot."""
    fig, ax = create_figure("wide")

    # Compute accuracy threshold (80%) for each k
    results = []
    for k in sorted(df["true_k"].unique()):
        for obs_dof_thresh in np.arange(1, 8, 0.5):
            subset = df[(df["true_k"] == k) & (df["obs_dof"] >= obs_dof_thresh)]
            if len(subset) >= 5:
                accuracy = subset["is_correct"].mean()
                results.append({"true_k": k, "obs_dof_min": obs_dof_thresh, "accuracy": accuracy})

    df_thresh = pd.DataFrame(results)

    # Find minimum obs_dof for 80% accuracy
    for i, k in enumerate(sorted(df["true_k"].unique())):
        subset = df_thresh[df_thresh["true_k"] == k]
        ax.plot(subset["obs_dof_min"], subset["accuracy"], marker="o", label=f"k={k}",
                color=CYCLE[i % len(CYCLE)], linewidth=2, markersize=6)

    ax.axhline(0.8, color="gray", linestyle="--", linewidth=2, label="80% accuracy threshold")
    ax.fill_between([1, 8], [0, 0], [0.8, 0.8], color="red", alpha=0.1, label="Unreliable zone")
    ax.fill_between([1, 8], [0.8, 0.8], [1, 1], color="green", alpha=0.1, label="Reliable zone")

    ax.set_xlabel("Minimum obs/dof")
    ax.set_ylabel("Accuracy")
    ax.set_title("Rank Selection: Required obs/dof for Reliable Recovery")
    ax.legend(loc="lower right")
    ax.set_xlim(1, 8)
    ax.set_ylim(0, 1.05)
    despine(ax)

    # Add annotation
    ax.annotate(
        "For reliable rank selection:\n• k=10: obs/dof ≥ 3\n• k=20: obs/dof ≥ 4\n• k=30: obs/dof ≥ 5",
        xy=(6, 0.3), fontsize=10, bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray")
    )

    save_figure(fig, output_dir / "fig5_recommendation.pdf")
    print(f"Saved: fig5_recommendation.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path, help="Directory with results")
    args = parser.parse_args()

    results_dir = args.results_dir
    pkl_file = results_dir / "rank_selection_results.pkl"

    if not pkl_file.exists():
        print(f"Error: {pkl_file} not found")
        sys.exit(1)

    df = pd.read_pickle(pkl_file)
    print(f"Loaded {len(df)} results")

    # Create obs_dof bins
    df["obs_dof_bin"] = pd.cut(
        df["obs_dof"],
        bins=[0, 2, 3, 4, 5, 10],
        labels=["<2", "2-3", "3-4", "4-5", ">5"]
    )

    # Generate all plots
    plot_main_results(df, results_dir)
    plot_cv_curves(df, results_dir)
    plot_eigenvalues(df, results_dir)
    plot_sharpness(df, results_dir)
    plot_recommendation(df, results_dir)

    print(f"\nAll plots saved to {results_dir}")


if __name__ == "__main__":
    main()
