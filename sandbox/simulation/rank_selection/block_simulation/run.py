"""Test Ka Chun's block simulation for rank detection.

This simulation creates:
1. Block-structured latent factors W with controlled overlap
2. RBF kernel similarity S from W

Key parameters:
- n: number of items
- k: number of blocks (true rank)
- overlap: number of items shared between adjacent blocks (difficulty)

Usage:
    poetry run python sandbox/block_simulation_test/run.py
"""

from __future__ import annotations

import sys
from datetime import datetime
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pysrf import SRF
from pysrf.bounds import estimate_sampling_bounds_ultra
from pysrf.cross_validation import cross_val_score

from src.colors import TEAL, CYAN, SAND, ROSE, GRAY, GRAY_LIGHT
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()


def rbf_kernel(X: np.ndarray, bw: float = 1.0) -> np.ndarray:
    """Compute RBF (Gaussian) kernel similarity matrix."""
    D2 = cdist(X, X, metric="sqeuclidean")
    return np.exp(-D2 / (2.0 * bw**2))


def simulation_block(
    n: int,
    k: int,
    overlap: int | None = None,
    density: float = 0.9,
    seed: int = 0,
) -> np.ndarray:
    """Generate block-structured latent factors.

    This is Ka Chun's simulation rewritten for clarity.

    Parameters
    ----------
    n : int
        Number of items (rows)
    k : int
        Number of blocks/factors (columns, true rank)
    overlap : int, optional
        Number of items shared between adjacent blocks.
        Default: n // (4 * k) for moderate overlap.
    density : float
        Probability of non-zero entry within a block (default 0.9)
    seed : int
        Random seed

    Returns
    -------
    W : np.ndarray
        (n, k) latent factor matrix with block structure
    """
    rng = np.random.default_rng(seed)

    if overlap is None:
        overlap = max(1, n // (4 * k))

    block_size = n // k
    W = np.zeros((n, k))

    for block_idx in range(k):
        # Block starts at block_idx * block_size
        start = block_idx * block_size
        # Block ends at start + block_size + overlap (but capped at n)
        end = min(start + block_size + overlap, n)

        for row in range(start, end):
            # Each entry is Uniform(0.5, 1.0) with probability `density`
            if rng.random() < density:
                W[row, block_idx] = rng.uniform(0.5, 1.0)

    # Clip to [0, 1]
    W = np.clip(W, 0, 1)

    return W


def make_similarity_block(
    n: int,
    k: int,
    overlap: int | None = None,
    bw: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate similarity matrix from block-structured factors.

    Parameters
    ----------
    n : int
        Number of items
    k : int
        Number of blocks (true rank)
    overlap : int, optional
        Overlap between adjacent blocks
    bw : float
        RBF kernel bandwidth
    seed : int
        Random seed

    Returns
    -------
    S : np.ndarray
        (n, n) similarity matrix
    W : np.ndarray
        (n, k) latent factors
    """
    W = simulation_block(n, k, overlap=overlap, seed=seed)
    S = rbf_kernel(W, bw=bw)
    return S, W


def compute_obs_dof(n: int, k: int) -> float:
    """Compute observations per degree of freedom."""
    return (n * (n - 1) / 2) / (n * k)


def test_bounds_stability(output_dir: Path) -> pd.DataFrame:
    """Test bounds estimation stability across parameters."""
    print("Testing bounds estimation stability...")

    results = []
    n = 200

    for k in [5, 10, 15, 20]:
        for overlap_frac in [0.0, 0.25, 0.5, 0.75]:
            overlap = int(overlap_frac * (n // k))

            for seed in range(5):
                S, W = make_similarity_block(n, k, overlap=overlap, seed=seed)

                pmin, pmax, _ = estimate_sampling_bounds_ultra(S, verbose=False)
                p_mean = (pmin + pmax) / 2

                # Eigenvalue analysis
                eigvals = np.linalg.eigvalsh(S)[::-1]
                eigvals_pos = eigvals[eigvals > 1e-10]
                p_norm = eigvals_pos / eigvals_pos.sum()
                eff_rank = np.exp(-np.sum(p_norm * np.log(p_norm + 1e-10)))

                results.append({
                    "n": n,
                    "k": k,
                    "overlap": overlap,
                    "overlap_frac": overlap_frac,
                    "seed": seed,
                    "pmin": pmin,
                    "pmax": pmax,
                    "p_mean": p_mean,
                    "eff_rank": eff_rank,
                    "obs_dof": compute_obs_dof(n, k),
                })

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "bounds_stability.csv", index=False)

    # Summary
    print("\nBounds stability by (k, overlap_frac):")
    summary = df.groupby(["k", "overlap_frac"]).agg(
        p_mean_avg=("p_mean", "mean"),
        p_mean_std=("p_mean", "std"),
        pmax_avg=("pmax", "mean"),
        eff_rank=("eff_rank", "mean"),
    ).round(3)
    print(summary)

    return df


def test_rank_detection(output_dir: Path) -> pd.DataFrame:
    """Test rank detection accuracy with block simulation."""
    print("\nTesting rank detection...")

    results = []
    n = 200

    for k in [5, 10, 15]:
        for overlap_frac in [0.0, 0.25, 0.5]:
            overlap = int(overlap_frac * (n // k))

            for seed in range(3):
                print(f"  k={k}, overlap_frac={overlap_frac:.2f}, seed={seed}")

                S, W = make_similarity_block(n, k, overlap=overlap, seed=seed)

                # Bounds estimation
                pmin, pmax, _ = estimate_sampling_bounds_ultra(S, verbose=False)
                p_mean = (pmin + pmax) / 2
                if p_mean <= 0.05 or p_mean >= 0.95:
                    p_mean = 0.5

                # Candidate ranks
                ranks = list(range(max(2, k - 6), min(k + 8, n // 2), 2))
                if k not in ranks:
                    ranks.append(k)
                ranks = sorted(set(ranks))

                # CV
                try:
                    result = cross_val_score(
                        S,
                        param_grid={"rank": ranks},
                        sampling_fraction=p_mean,
                        n_repeats=5,
                        n_jobs=-1,
                        verbose=0,
                    )
                    selected_rank = result.best_params_["rank"]
                except Exception as e:
                    print(f"    CV failed: {e}")
                    selected_rank = -1

                results.append({
                    "n": n,
                    "k": k,
                    "overlap": overlap,
                    "overlap_frac": overlap_frac,
                    "seed": seed,
                    "p_mean": p_mean,
                    "selected_rank": selected_rank,
                    "abs_error": abs(selected_rank - k),
                    "signed_error": selected_rank - k,
                    "is_correct": selected_rank == k,
                })

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "rank_detection.csv", index=False)

    # Summary
    print("\nRank detection by (k, overlap_frac):")
    summary = df.groupby(["k", "overlap_frac"]).agg(
        accuracy=("is_correct", "mean"),
        bias=("signed_error", "mean"),
        p_mean=("p_mean", "mean"),
    ).round(2)
    print(summary)

    return df


def plot_simulation_overview(output_dir: Path) -> None:
    """Plot overview of the block simulation."""
    print("\nPlotting simulation overview...")

    n, k = 200, 10

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    for col, overlap_frac in enumerate([0.0, 0.25, 0.5, 0.75]):
        overlap = int(overlap_frac * (n // k))
        S, W = make_similarity_block(n, k, overlap=overlap, seed=42)

        # Top row: W matrix
        ax = axes[0, col]
        im = ax.imshow(W, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"overlap={overlap_frac:.0%}")
        if col == 0:
            ax.set_ylabel("W (latent factors)")
        ax.set_xlabel("Factor")

        # Bottom row: S matrix
        ax = axes[1, col]
        ax.imshow(S, cmap="Blues", vmin=0, vmax=1)
        if col == 0:
            ax.set_ylabel("S (similarity)")

        # Compute stats
        eigvals = np.linalg.eigvalsh(S)[::-1]
        eigvals_pos = eigvals[eigvals > 1e-10]
        p_norm = eigvals_pos / eigvals_pos.sum()
        eff_rank = np.exp(-np.sum(p_norm * np.log(p_norm + 1e-10)))
        ax.set_xlabel(f"eff_rank={eff_rank:.1f}")

    plt.tight_layout()
    fig.savefig(output_dir / "simulation_overview.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  simulation_overview.pdf")


def plot_eigenvalue_spectrum(output_dir: Path) -> None:
    """Plot eigenvalue spectrum for different overlap values."""
    print("\nPlotting eigenvalue spectrum...")

    n, k = 200, 10

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    colors = {0.0: TEAL, 0.25: CYAN, 0.5: SAND, 0.75: ROSE}

    for overlap_frac in [0.0, 0.25, 0.5, 0.75]:
        overlap = int(overlap_frac * (n // k))
        S, W = make_similarity_block(n, k, overlap=overlap, seed=42)

        eigvals = np.linalg.eigvalsh(S)[::-1]

        # Left: eigenvalue spectrum
        ax = axes[0]
        ax.semilogy(range(1, len(eigvals) + 1), eigvals, "-",
                   color=colors[overlap_frac], lw=1.5,
                   label=f"overlap={overlap_frac:.0%}")

    ax = axes[0]
    ax.axvline(k, color=GRAY, ls="--", lw=1, label=f"true k={k}")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue (log)")
    ax.set_xlim(0, 30)
    ax.legend(frameon=False, fontsize=8)
    despine(ax)

    # Right: cumulative variance
    ax = axes[1]
    for overlap_frac in [0.0, 0.25, 0.5, 0.75]:
        overlap = int(overlap_frac * (n // k))
        S, W = make_similarity_block(n, k, overlap=overlap, seed=42)
        eigvals = np.linalg.eigvalsh(S)[::-1]
        cumvar = np.cumsum(eigvals) / eigvals.sum()
        ax.plot(range(1, len(cumvar) + 1), cumvar * 100, "-",
               color=colors[overlap_frac], lw=1.5)

    ax.axvline(k, color=GRAY, ls="--", lw=1)
    ax.axhline(99, color=GRAY_LIGHT, ls=":", lw=1)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative variance (%)")
    ax.set_xlim(0, 30)
    ax.set_ylim(90, 100.5)
    despine(ax)

    plt.tight_layout()
    fig.savefig(output_dir / "eigenvalue_spectrum.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  eigenvalue_spectrum.pdf")


def plot_bounds_stability(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot bounds estimation stability results."""
    print("\nPlotting bounds stability...")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    # Left: p_mean by k and overlap
    ax = axes[0]
    for k in df["k"].unique():
        subset = df[df["k"] == k].groupby("overlap_frac")["p_mean"].mean()
        ax.plot(subset.index, subset.values, "o-", label=f"k={k}", markersize=5)

    ax.set_xlabel("Overlap fraction")
    ax.set_ylabel("p_mean")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(0, 0.6)
    despine(ax)

    # Middle: pmax stability (std)
    ax = axes[1]
    pivot = df.groupby(["k", "overlap_frac"])["pmax"].std().unstack()
    for k in pivot.index:
        ax.plot(pivot.columns, pivot.loc[k], "o-", label=f"k={k}", markersize=5)

    ax.set_xlabel("Overlap fraction")
    ax.set_ylabel("pmax std (across seeds)")
    ax.legend(frameon=False, fontsize=8)
    despine(ax)

    # Right: effective rank
    ax = axes[2]
    for k in df["k"].unique():
        subset = df[df["k"] == k].groupby("overlap_frac")["eff_rank"].mean()
        ax.plot(subset.index, subset.values, "o-", label=f"k={k}", markersize=5)
        ax.axhline(k, color=GRAY_LIGHT, ls=":", lw=0.8)

    ax.set_xlabel("Overlap fraction")
    ax.set_ylabel("Effective rank")
    ax.legend(frameon=False, fontsize=8)
    despine(ax)

    plt.tight_layout()
    fig.savefig(output_dir / "bounds_stability.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  bounds_stability.pdf")


def plot_rank_detection_results(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot rank detection results."""
    print("\nPlotting rank detection results...")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    # Left: accuracy by k and overlap
    ax = axes[0]
    pivot = df.groupby(["k", "overlap_frac"])["is_correct"].mean().unstack()
    for k in pivot.index:
        ax.plot(pivot.columns, pivot.loc[k] * 100, "o-", label=f"k={k}", markersize=6)

    ax.axhline(100, color=GRAY_LIGHT, ls="--", lw=0.8)
    ax.set_xlabel("Overlap fraction")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(-5, 110)
    ax.legend(frameon=False, fontsize=8)
    despine(ax)

    # Middle: bias by k and overlap
    ax = axes[1]
    pivot = df.groupby(["k", "overlap_frac"])["signed_error"].mean().unstack()
    for k in pivot.index:
        ax.plot(pivot.columns, pivot.loc[k], "o-", label=f"k={k}", markersize=6)

    ax.axhline(0, color=GRAY, ls="-", lw=0.8)
    ax.set_xlabel("Overlap fraction")
    ax.set_ylabel("Bias (selected - true)")
    ax.legend(frameon=False, fontsize=8)
    despine(ax)

    # Right: scatter of selected vs true rank
    ax = axes[2]
    for overlap_frac in df["overlap_frac"].unique():
        subset = df[df["overlap_frac"] == overlap_frac]
        ax.scatter(subset["k"], subset["selected_rank"],
                  alpha=0.6, s=30, label=f"overlap={overlap_frac:.0%}")

    ax.plot([0, 20], [0, 20], "--", color=GRAY, lw=1)
    ax.set_xlabel("True rank")
    ax.set_ylabel("Selected rank")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    despine(ax)

    plt.tight_layout()
    fig.savefig(output_dir / "rank_detection.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  rank_detection.pdf")


def plot_comparison_with_dirichlet(output_dir: Path) -> None:
    """Compare block simulation with Dirichlet simulation for bounds."""
    print("\nComparing with Dirichlet simulation...")

    from src.utils.simulation import simulation_dirichlet

    n, k = 200, 10

    results = []

    # Block simulation
    for overlap_frac in [0.0, 0.25, 0.5]:
        overlap = int(overlap_frac * (n // k))
        for seed in range(5):
            S, _ = make_similarity_block(n, k, overlap=overlap, seed=seed)
            pmin, pmax, _ = estimate_sampling_bounds_ultra(S, verbose=False)
            results.append({
                "method": "block",
                "param": overlap_frac,
                "seed": seed,
                "pmin": pmin,
                "pmax": pmax,
                "p_mean": (pmin + pmax) / 2,
            })

    # Dirichlet simulation
    for alpha in [0.1, 0.5, 1.0]:
        for seed in range(5):
            rng = np.random.default_rng(seed)
            W = simulation_dirichlet(n, k, alpha=alpha, rng=rng)
            S = W @ W.T
            pmin, pmax, _ = estimate_sampling_bounds_ultra(S, verbose=False)
            results.append({
                "method": "dirichlet",
                "param": alpha,
                "seed": seed,
                "pmin": pmin,
                "pmax": pmax,
                "p_mean": (pmin + pmax) / 2,
            })

    df = pd.DataFrame(results)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Left: Block simulation
    ax = axes[0]
    block_df = df[df["method"] == "block"]
    for param in block_df["param"].unique():
        subset = block_df[block_df["param"] == param]
        ax.scatter([param] * len(subset), subset["p_mean"], alpha=0.7, s=50)

    summary = block_df.groupby("param")["p_mean"].agg(["mean", "std"])
    ax.errorbar(summary.index, summary["mean"], yerr=summary["std"],
                fmt="s-", color="black", capsize=4, markersize=8, lw=2)

    ax.set_xlabel("Overlap fraction")
    ax.set_ylabel("p_mean")
    ax.set_title("Block simulation")
    ax.set_ylim(0, 0.6)
    despine(ax)

    # Right: Dirichlet simulation
    ax = axes[1]
    dir_df = df[df["method"] == "dirichlet"]
    for param in dir_df["param"].unique():
        subset = dir_df[dir_df["param"] == param]
        ax.scatter([param] * len(subset), subset["p_mean"], alpha=0.7, s=50)

    summary = dir_df.groupby("param")["p_mean"].agg(["mean", "std"])
    ax.errorbar(summary.index, summary["mean"], yerr=summary["std"],
                fmt="s-", color="black", capsize=4, markersize=8, lw=2)

    ax.set_xlabel("α (Dirichlet)")
    ax.set_ylabel("p_mean")
    ax.set_title("Dirichlet simulation")
    ax.set_ylim(0, 0.6)
    ax.set_xscale("log")
    despine(ax)

    plt.tight_layout()
    fig.savefig(output_dir / "comparison_with_dirichlet.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  comparison_with_dirichlet.pdf")

    # Print summary
    print("\nBlock simulation p_mean:")
    print(block_df.groupby("param")[["pmin", "pmax", "p_mean"]].agg(["mean", "std"]).round(3))
    print("\nDirichlet simulation p_mean:")
    print(dir_df.groupby("param")[["pmin", "pmax", "p_mean"]].agg(["mean", "std"]).round(3))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # 1. Simulation overview
    plot_simulation_overview(OUTPUT_DIR)

    # 2. Eigenvalue spectrum analysis
    plot_eigenvalue_spectrum(OUTPUT_DIR)

    # 3. Bounds stability test
    bounds_df = test_bounds_stability(OUTPUT_DIR)
    plot_bounds_stability(bounds_df, OUTPUT_DIR)

    # 4. Comparison with Dirichlet
    plot_comparison_with_dirichlet(OUTPUT_DIR)

    # 5. Rank detection test (this takes longer)
    rank_df = test_rank_detection(OUTPUT_DIR)
    plot_rank_detection_results(rank_df, OUTPUT_DIR)

    print(f"\nAll outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
