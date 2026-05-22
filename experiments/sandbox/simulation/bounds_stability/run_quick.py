"""Quick test of bounds estimation stability.

Smaller parameter grid for faster results.

Usage:
    poetry run python sandbox/bounds_stability/run_quick.py
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

# Focused parameter grid
N_VALUES = [200, 300]
K_VALUES = [10, 20, 30]
ALPHA_VALUES = [0.1, 0.5, 1.0, 5.0]
N_SEEDS = 10


def compute_obs_dof(n: int, k: int) -> float:
    return (n * (n - 1) / 2) / (n * k)


def run_single(n: int, k: int, alpha: float, seed: int) -> dict:
    """Generate matrix and estimate bounds."""
    rng = np.random.default_rng(seed)
    w = simulation_dirichlet(n=n, k=k, alpha=alpha, rng=rng)
    similarity = w @ w.T

    pmin, pmax, _ = estimate_sampling_bounds_fast(similarity, verbose=False)
    p_mean = (pmin + pmax) / 2

    # Eigenvalue spectrum
    eigenvalues = np.linalg.eigvalsh(similarity)
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Effective rank
    eigenvalues_pos = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues_pos) > 0:
        p_norm = eigenvalues_pos / eigenvalues_pos.sum()
        effective_rank = np.exp(-np.sum(p_norm * np.log(p_norm + 1e-10)))
    else:
        effective_rank = 0

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
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    conditions = [
        (n, k, alpha, seed)
        for n in N_VALUES
        for k in K_VALUES
        for alpha in ALPHA_VALUES
        for seed in range(N_SEEDS)
    ]

    print(f"Testing {len(conditions)} conditions...")

    results = Parallel(n_jobs=32, verbose=10)(
        delayed(run_single)(n, k, alpha, seed)
        for n, k, alpha, seed in conditions
    )

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "bounds_quick.csv", index=False)

    # Aggregate
    agg = df.groupby(["n", "k", "alpha"]).agg(
        obs_dof=("obs_per_dof", "first"),
        p_mean_avg=("p_mean", "mean"),
        p_mean_std=("p_mean", "std"),
        p_mean_min=("p_mean", "min"),
        p_mean_max=("p_mean", "max"),
        pmin_avg=("pmin", "mean"),
        pmin_std=("pmin", "std"),
        eff_rank=("effective_rank", "mean"),
    ).reset_index()

    agg["cv"] = agg["p_mean_std"] / (agg["p_mean_avg"] + 1e-10)
    agg["is_stable"] = (agg["cv"] < 0.3) & (agg["p_mean_avg"] > 0.15)

    print("\n" + "="*90)
    print("ALL CONDITIONS:")
    print("="*90)
    print(agg[["n", "k", "alpha", "obs_dof", "p_mean_avg", "p_mean_std", "cv", "eff_rank", "is_stable"]].round(3).to_string(index=False))

    print("\n" + "="*90)
    print("BY ALPHA:")
    print("="*90)
    print(df.groupby("alpha").agg(
        p_mean=("p_mean", "mean"),
        p_mean_std=("p_mean", "std"),
        pmin=("pmin", "mean"),
    ).round(3))

    print("\n" + "="*90)
    print("BY K:")
    print("="*90)
    print(df.groupby("k").agg(
        p_mean=("p_mean", "mean"),
        p_mean_std=("p_mean", "std"),
        pmin=("pmin", "mean"),
    ).round(3))

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
