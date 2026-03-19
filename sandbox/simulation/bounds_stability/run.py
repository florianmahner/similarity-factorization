"""Test how simulation parameters affect bounds estimation stability.

Goal: Find parameter combinations where pmin/pmax are stable across seeds.

Usage:
    poetry run python sandbox/bounds_stability/run.py
"""

from __future__ import annotations

import sys
from datetime import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pysrf import estimate_sampling_bounds_fast

from src.utils.simulation import simulation_dirichlet
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()

# Parameters to test
N_VALUES = [100, 200, 300, 400]
K_VALUES = [5, 10, 15, 20, 30, 50]
ALPHA_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
N_SEEDS = 20


def compute_obs_dof(n: int, k: int) -> float:
    return (n * (n - 1) / 2) / (n * k)


def run_single(n: int, k: int, alpha: float, seed: int) -> dict:
    """Generate matrix and estimate bounds."""
    if k >= n:
        return None

    rng = np.random.default_rng(seed)
    w = simulation_dirichlet(n=n, k=k, alpha=alpha, rng=rng)
    similarity = w @ w.T

    pmin, pmax, _ = estimate_sampling_bounds_fast(similarity, verbose=False)
    p_mean = (pmin + pmax) / 2

    # Also compute eigenvalue spectrum info
    eigenvalues = np.linalg.eigvalsh(similarity)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending

    # Effective rank (participation ratio)
    eigenvalues_pos = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues_pos) > 0:
        p_norm = eigenvalues_pos / eigenvalues_pos.sum()
        effective_rank = np.exp(-np.sum(p_norm * np.log(p_norm + 1e-10)))
    else:
        effective_rank = 0

    # Spectral gap at true rank
    if k < len(eigenvalues):
        gap_at_k = eigenvalues[k-1] - eigenvalues[k] if k < len(eigenvalues) else 0
        ratio_at_k = eigenvalues[k-1] / (eigenvalues[k] + 1e-10) if k < len(eigenvalues) else 0
    else:
        gap_at_k = 0
        ratio_at_k = 0

    return {
        "n": n,
        "k": k,
        "alpha": alpha,
        "seed": seed,
        "obs_per_dof": compute_obs_dof(n, k),
        "pmin": pmin,
        "pmax": pmax,
        "p_mean": p_mean,
        "effective_rank": effective_rank,
        "gap_at_k": gap_at_k,
        "ratio_at_k": ratio_at_k,
        "top_eigenvalue": eigenvalues[0],
        "eigenvalue_at_k": eigenvalues[k-1] if k <= len(eigenvalues) else 0,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Build conditions
    conditions = [
        (n, k, alpha, seed)
        for n in N_VALUES
        for k in K_VALUES
        for alpha in ALPHA_VALUES
        for seed in range(N_SEEDS)
        if k < n  # Skip invalid combinations
    ]

    print(f"Testing {len(conditions)} conditions...")
    print(f"  n: {N_VALUES}")
    print(f"  k: {K_VALUES}")
    print(f"  alpha: {ALPHA_VALUES}")
    print(f"  seeds: {N_SEEDS}")
    print()

    results = Parallel(n_jobs=32, verbose=10)(
        delayed(run_single)(n, k, alpha, seed)
        for n, k, alpha, seed in conditions
    )

    # Filter None results
    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "bounds_stability.csv", index=False)

    # Aggregate by (n, k, alpha)
    agg = df.groupby(["n", "k", "alpha"]).agg(
        obs_dof=("obs_per_dof", "first"),
        p_mean_avg=("p_mean", "mean"),
        p_mean_std=("p_mean", "std"),
        p_mean_min=("p_mean", "min"),
        p_mean_max=("p_mean", "max"),
        pmin_avg=("pmin", "mean"),
        pmin_std=("pmin", "std"),
        effective_rank_avg=("effective_rank", "mean"),
        gap_at_k_avg=("gap_at_k", "mean"),
    ).reset_index()

    # Flag unstable conditions (high std or low p_mean)
    agg["cv_p_mean"] = agg["p_mean_std"] / (agg["p_mean_avg"] + 1e-10)
    agg["is_stable"] = (agg["cv_p_mean"] < 0.3) & (agg["p_mean_avg"] > 0.2)

    agg.to_csv(OUTPUT_DIR / "bounds_stability_summary.csv", index=False)

    print("\n" + "="*80)
    print("STABLE CONDITIONS (cv < 0.3 and p_mean > 0.2):")
    print("="*80)
    stable = agg[agg["is_stable"]].sort_values(["k", "n", "alpha"])
    print(stable[["n", "k", "alpha", "obs_dof", "p_mean_avg", "p_mean_std", "cv_p_mean"]].round(3).to_string(index=False))

    print("\n" + "="*80)
    print("UNSTABLE CONDITIONS (high variance or low p_mean):")
    print("="*80)
    unstable = agg[~agg["is_stable"]].sort_values(["k", "n", "alpha"])
    if len(unstable) > 30:
        print(f"Showing first 30 of {len(unstable)}:")
        print(unstable[["n", "k", "alpha", "obs_dof", "p_mean_avg", "p_mean_std", "cv_p_mean"]].head(30).round(3).to_string(index=False))
    else:
        print(unstable[["n", "k", "alpha", "obs_dof", "p_mean_avg", "p_mean_std", "cv_p_mean"]].round(3).to_string(index=False))

    print("\n" + "="*80)
    print("SUMMARY BY ALPHA (averaged over n, k):")
    print("="*80)
    alpha_summary = agg.groupby("alpha").agg(
        stable_pct=("is_stable", "mean"),
        p_mean_avg=("p_mean_avg", "mean"),
        cv_p_mean_avg=("cv_p_mean", "mean"),
    ).round(3)
    print(alpha_summary)

    print("\n" + "="*80)
    print("SUMMARY BY K (averaged over n, alpha):")
    print("="*80)
    k_summary = agg.groupby("k").agg(
        stable_pct=("is_stable", "mean"),
        p_mean_avg=("p_mean_avg", "mean"),
        cv_p_mean_avg=("cv_p_mean", "mean"),
    ).round(3)
    print(k_summary)

    print(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
