"""Visualize rank detection scaling with sample size.

Key insight: Detection accuracy depends on observations per degree of freedom.
obs = n(n-1)/2, dof = n*k, ratio = obs/dof ≈ (n-1)/(2k)

Usage:
    poetry run python sandbox/rank_detection_viz/mockup_scaling.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.colors import ROSE, TEAL, CYAN, SAND, GRAY, GRAY_LIGHT, GRAY_DARK, GRAY_PALE
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

PROJECT_ROOT = Path(__file__).parents[2]
OUTPUT_DIR = get_output_dir()


def run_single_detection(n: int, k: int, alpha: float, seed: int) -> dict:
    """Run single rank detection test."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from pysrf import SRF
    from pysrf.cross_validation import cross_val_score
    from utils.simulation import simulation_dirichlet

    rng = np.random.default_rng(seed)
    w = simulation_dirichlet(n=n, k=k, alpha=alpha, rng=rng)
    S = w @ w.T

    ranks = list(range(max(2, k - 6), k + 8, 2))
    if k not in ranks:
        ranks.append(k)
    ranks = sorted(ranks)

    result = cross_val_score(
        S,
        param_grid={"rank": ranks},
        n_repeats=3,
        estimate_sampling_fraction=True,
        verbose=0,
        n_jobs=1,
    )
    selected = result.best_params_["rank"]

    obs = n * (n - 1) // 2
    dof = n * k
    ratio = obs / dof

    return {
        "n": n,
        "k": k,
        "alpha": alpha,
        "seed": seed,
        "selected": selected,
        "abs_error": abs(selected - k),
        "obs_per_dof": ratio,
    }


def run_scaling_experiment() -> pd.DataFrame:
    """Run experiment varying n and k."""
    # Grid of conditions
    ns = [50, 75, 100, 150, 200, 300, 500]
    ks = [5, 10, 15, 20, 25, 30]
    alpha = 1.0
    n_seeds = 5

    conditions = [
        (n, k, alpha, seed)
        for n in ns
        for k in ks
        if k < n  # k must be less than n
        for seed in range(n_seeds)
    ]

    print(f"Running {len(conditions)} conditions...")

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_single_detection)(n, k, alpha, seed)
        for n, k, alpha, seed in conditions
    )

    return pd.DataFrame(results)


def plot_mae_vs_obs_dof(df: pd.DataFrame, output_dir: Path) -> None:
    """MAE vs observations per DOF - the universal curve."""
    summary = df.groupby(["n", "k", "obs_per_dof"]).agg(
        mae=("abs_error", "mean"),
        sem=("abs_error", "sem"),
    ).reset_index()

    fig, ax = create_figure("single")

    # Color by k
    ks = sorted(summary["k"].unique())
    colors = {5: ROSE, 10: TEAL, 15: CYAN, 20: SAND,
              25: GRAY, 30: GRAY_DARK}

    for k in ks:
        subset = summary[summary["k"] == k].sort_values("obs_per_dof")
        color = colors.get(k, GRAY)
        ax.plot(subset["obs_per_dof"], subset["mae"], "o-",
                color=color, markersize=5, linewidth=1.5, label=f"k={k}")

    # Threshold region
    ax.axvspan(0, 5, color=GRAY_PALE, zorder=0, label="Underdetermined")
    ax.axvline(5, color=GRAY_LIGHT, linestyle="--", linewidth=1, zorder=1)

    ax.set_xlabel("Observations per degree of freedom")
    ax.set_ylabel("Mean absolute error")
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 10)
    ax.legend(frameon=False, loc="upper right", fontsize=7, ncol=2)
    despine(ax)

    save_figure(fig, output_dir / "mae_vs_obs_dof.pdf")


def plot_heatmap_phase(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap showing detection accuracy in (n, k) space."""
    summary = df.groupby(["n", "k"])["abs_error"].mean().reset_index()
    pivot = summary.pivot(index="k", columns="n", values="abs_error")
    pivot = pivot.sort_index(ascending=False)

    fig, ax = create_figure("single")

    # Custom colormap: green (good) to red (bad)
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#2ecc71", "#f1c40f", "#e74c3c"]
    cmap = LinearSegmentedColormap.from_list("mae", colors)

    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0, vmax=6)

    # Add obs/dof = 5 contour line
    ns = pivot.columns.values
    ks = pivot.index.values
    for i, k in enumerate(ks):
        for j, n in enumerate(ns):
            ratio = (n * (n - 1) / 2) / (n * k)
            if 4.5 < ratio < 5.5:
                ax.plot(j, i, "ko", markersize=8, markerfacecolor="none", linewidth=2)

    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels(ns)
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels(ks)

    ax.set_xlabel("Number of samples (n)")
    ax.set_ylabel("True rank (k)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("MAE")

    save_figure(fig, output_dir / "phase_diagram.pdf")


def plot_collapsed_curve(df: pd.DataFrame, output_dir: Path) -> None:
    """All data collapsed onto obs/dof curve."""
    fig, ax = create_figure("single")

    # Bin by obs/dof
    df["obs_dof_bin"] = pd.cut(df["obs_per_dof"], bins=[0, 2, 4, 6, 8, 10, 15, 25])

    summary = df.groupby("obs_dof_bin", observed=True).agg(
        mae=("abs_error", "mean"),
        sem=("abs_error", "sem"),
        obs_dof_mid=("obs_per_dof", "mean"),
    ).reset_index()

    ax.errorbar(summary["obs_dof_mid"], summary["mae"], yerr=summary["sem"],
                fmt="o-", color=TEAL, markersize=6, linewidth=2, capsize=3)

    # Threshold
    ax.axvspan(0, 5, color=GRAY_PALE, zorder=0)
    ax.axvline(5, color=GRAY_LIGHT, linestyle="--", linewidth=1, zorder=1)
    ax.text(2.5, 0.5, "underdetermined", fontsize=7, color=GRAY,
            ha="center", va="bottom")

    ax.set_xlabel("Observations per degree of freedom")
    ax.set_ylabel("Mean absolute error")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 8)
    despine(ax)

    save_figure(fig, output_dir / "collapsed_curve.pdf")


def plot_required_samples(output_dir: Path) -> None:
    """Show required n for different k to achieve reliable detection."""
    fig, ax = create_figure("single")

    ks = np.arange(5, 51, 1)

    # For obs/dof = 5: n(n-1)/2 = 5*n*k => n ≈ 10k + 1
    # For obs/dof = 8: n ≈ 16k + 1
    n_for_ratio_5 = 10 * ks + 1
    n_for_ratio_8 = 16 * ks + 1

    ax.fill_between(ks, n_for_ratio_5, n_for_ratio_8, color=TEAL, alpha=0.3,
                    label="Reliable zone")
    ax.plot(ks, n_for_ratio_5, "-", color=TEAL, linewidth=1.5,
            label="obs/dof = 5")
    ax.plot(ks, n_for_ratio_8, "--", color=TEAL, linewidth=1.5,
            label="obs/dof = 8")

    # Shade underdetermined region
    ax.fill_between(ks, 0, n_for_ratio_5, color=GRAY_PALE, alpha=0.5)
    ax.text(30, 100, "underdetermined", fontsize=8, color=GRAY,
            ha="center", rotation=25)

    ax.set_xlabel("True rank (k)")
    ax.set_ylabel("Required samples (n)")
    ax.set_xlim(5, 50)
    ax.set_ylim(0, 600)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    despine(ax)

    save_figure(fig, output_dir / "required_samples.pdf")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # First, create the theoretical plot (no simulation needed)
    print("Creating theoretical required samples plot...")
    plot_required_samples(OUTPUT_DIR)
    print("  required_samples.pdf")

    # Check if we have cached results
    cache_path = OUTPUT_DIR / "scaling_results.csv"

    if cache_path.exists():
        print(f"\nLoading cached results from {cache_path}")
        df = pd.read_csv(cache_path)
    else:
        print("\nRunning scaling experiment (this takes a few minutes)...")
        df = run_scaling_experiment()
        df.to_csv(cache_path, index=False)
        print(f"  Saved to {cache_path}")

    print("\nGenerating plots...")

    plot_mae_vs_obs_dof(df, OUTPUT_DIR)
    print("  mae_vs_obs_dof.pdf")

    plot_heatmap_phase(df, OUTPUT_DIR)
    print("  phase_diagram.pdf")

    plot_collapsed_curve(df, OUTPUT_DIR)
    print("  collapsed_curve.pdf")

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
