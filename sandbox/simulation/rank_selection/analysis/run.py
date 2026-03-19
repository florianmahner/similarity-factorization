"""Comprehensive rank selection analysis.

Shows when/how CV-based rank selection works or fails.
Key finding: bounds give p for completion (obs/dof~2-3), but
rank selection needs obs/dof >= 5 for reliability.

Usage:
    poetry run python sandbox/rank_selection_analysis/run.py
"""

from __future__ import annotations

import sys
from datetime import datetime
import os
from pathlib import Path

from src.utils import get_output_dir

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pysrf import SRF

OUTPUT_DIR = get_output_dir()

# Parameter grid
N_VALUES = [300, 500, 700, 900]
TRUE_K_VALUES = [10, 20, 30]
P_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
N_SEEDS = 10


def simulation_block(n: int, k: int, offdiag_prob: float = 0.2, seed: int = 0) -> np.ndarray:
    """Generate block-structured similarity matrix."""
    rng = np.random.default_rng(seed)
    block_size = n // k
    W = np.zeros((n, k))

    # Primary block structure
    for b in range(k):
        start = b * block_size
        end = min(start + block_size + 3, n)
        for i in range(start, end):
            if rng.random() < 0.9:
                W[i, b] = rng.uniform(0.5, 1.0)

    # Off-diagonal entries (weak membership in other factors)
    for i in range(n):
        primary = min(i // block_size, k - 1)
        for j in range(k):
            if j != primary and W[i, j] == 0 and rng.random() < offdiag_prob:
                W[i, j] = rng.uniform(0.05, 0.3)

    return W @ W.T


def create_masks(n: int, p: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create train/test masks for upper triangle."""
    rng = np.random.default_rng(seed)

    # Upper triangle mask (excluding diagonal)
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    n_upper = upper.sum()

    # Random split
    train_idx = rng.random(n_upper) < p

    train_mask = np.zeros((n, n), dtype=bool)
    test_mask = np.zeros((n, n), dtype=bool)
    train_mask[upper] = train_idx
    test_mask[upper] = ~train_idx

    # Symmetrize
    train_mask = train_mask | train_mask.T
    test_mask = test_mask | test_mask.T

    return train_mask, test_mask


def fit_and_score(S: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray,
                  rank: int, seed: int) -> float:
    """Fit SRF and return test error."""
    S_train = S.copy()
    S_train[~train_mask] = np.nan
    np.fill_diagonal(S_train, np.diag(S))

    model = SRF(rank=rank, random_state=seed, verbose=False, max_outer=100)
    model.fit(S_train)
    S_recon = model.reconstruct()

    test_error = np.mean((S[test_mask] - S_recon[test_mask]) ** 2)
    return test_error


def evaluate_condition(n: int, true_k: int, p: float, seed: int) -> dict:
    """Evaluate rank selection for one condition."""
    # Generate data
    S = simulation_block(n, true_k, seed=seed)
    train_mask, test_mask = create_masks(n, p, seed=seed + 1000)

    # Candidate ranks
    candidate_ranks = [true_k - 4, true_k - 2, true_k, true_k + 2, true_k + 4]
    candidate_ranks = [r for r in candidate_ranks if r >= 2]

    # Compute obs/dof for max candidate rank
    k_max = max(candidate_ranks)
    obs_dof = p * (n - 1) / (2 * k_max)

    # Fit all candidate ranks
    scores = {}
    for rank in candidate_ranks:
        scores[rank] = fit_and_score(S, train_mask, test_mask, rank, seed)

    # Find best rank
    selected_rank = min(scores, key=scores.get)
    selection_error = abs(selected_rank - true_k)
    is_correct = selected_rank == true_k

    # Compute eigenvalue metrics
    eigenvalues = np.linalg.eigvalsh(S)[::-1]
    spectral_gap = eigenvalues[true_k - 1] - eigenvalues[true_k] if true_k < len(eigenvalues) else 0

    # Effective rank
    eig_pos = eigenvalues[eigenvalues > 1e-10]
    p_norm = eig_pos / eig_pos.sum()
    effective_rank = np.exp(-np.sum(p_norm * np.log(p_norm + 1e-10)))

    # Score ratio (sharpness of CV curve)
    score_at_k = scores.get(true_k, np.nan)
    score_at_k_minus_2 = scores.get(true_k - 2, np.nan)
    score_ratio = score_at_k_minus_2 / (score_at_k + 1e-10) if not np.isnan(score_at_k_minus_2) else np.nan

    return {
        "n": n,
        "true_k": true_k,
        "p": p,
        "seed": seed,
        "obs_dof": obs_dof,
        "selected_rank": selected_rank,
        "selection_error": selection_error,
        "is_correct": is_correct,
        "spectral_gap": spectral_gap,
        "effective_rank": effective_rank,
        "score_ratio": score_ratio,
        "score_at_k": score_at_k,
        "scores": scores,  # dict of all scores for plotting
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Build all conditions
    conditions = [
        (n, true_k, p, seed)
        for n in N_VALUES
        for true_k in TRUE_K_VALUES
        for p in P_VALUES
        for seed in range(N_SEEDS)
    ]

    print(f"Running {len(conditions)} conditions...")
    print(f"  n: {N_VALUES}")
    print(f"  true_k: {TRUE_K_VALUES}")
    print(f"  p: {P_VALUES}")
    print(f"  seeds: {N_SEEDS}")
    print()

    # Run in parallel
    results = Parallel(n_jobs=32, verbose=10)(
        delayed(evaluate_condition)(n, true_k, p, seed)
        for n, true_k, p, seed in conditions
    )

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save raw results (excluding scores dict for CSV)
    df_save = df.drop(columns=["scores"])
    df_save.to_csv(OUTPUT_DIR / "rank_selection_results.csv", index=False)

    # Save full results with scores as pickle for plotting
    df.to_pickle(OUTPUT_DIR / "rank_selection_results.pkl")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY BY obs_dof BINS")
    print("=" * 70)

    df["obs_dof_bin"] = pd.cut(df["obs_dof"], bins=[0, 2, 3, 4, 5, 10], labels=["<2", "2-3", "3-4", "4-5", ">5"])

    summary = df.groupby(["true_k", "obs_dof_bin"]).agg(
        accuracy=("is_correct", "mean"),
        mean_error=("selection_error", "mean"),
        n_samples=("is_correct", "count"),
    ).round(3)

    print(summary.to_string())

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
