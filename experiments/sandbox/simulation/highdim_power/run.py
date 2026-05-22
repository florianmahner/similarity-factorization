"""
RSA vs SRF Power Analysis: High-Dimensional Noise Model.

Tests the hypothesis that SRF outperforms RSA when noise is added in a
high-dimensional space (p >> k), because NMF filters out noise from
orthogonal dimensions.

Generative Model:
1. X (N × k): Ground truth factorial design
2. M (k × p): Mixing matrix projecting to p-dimensional "voxel" space
3. Y_clean = X @ M: Signal lives in k-dim subspace of p-dim space
4. Y_noisy = add_positive_noise_with_snr(Y_clean, snr): Noise in all p dims
5. S = Y_noisy @ Y_noisy.T: RSM (positive by construction)

Key insight:
- When p = k: Noise is in same subspace as signal → SRF ≈ RSA
- When p >> k: Noise spread across p dims, signal in k dims → SRF > RSA

Tests (with restricted permutation for factorial):
- RSA: Mantel test (hypothesis RSM vs measured RSM)
- SRF: Hungarian alignment (X columns vs W columns)
"""

from datetime import datetime
from pathlib import Path
import itertools as it

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform

from pysrf import SRF
from src.utils.helpers import add_positive_noise_with_snr
from src.colors import ROSE, TEAL, GRAY_DARK
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir


# --- Restricted Mantel Test ---


def get_stratified_indices(n: int, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Get permutation indices that only shuffle within strata groups."""
    idx = np.arange(n)
    unique_strata = np.unique(strata, axis=0)
    for group in unique_strata:
        mask = (strata == group).all(axis=1)
        indices = np.where(mask)[0]
        idx[indices] = rng.permutation(indices)
    return idx


def _mantel_worker(
    model_rsm: np.ndarray, data_rsm_flat: np.ndarray, strata: np.ndarray, seed: int
) -> float:
    """Permutes Model RSM rows/cols restricted by strata, then correlates with Data."""
    rng = np.random.default_rng(seed)
    n = model_rsm.shape[0]
    idx = get_stratified_indices(n, strata, rng)
    model_perm = model_rsm[idx, :][:, idx]
    model_flat = squareform(model_perm, checks=False)
    return np.corrcoef(model_flat, data_rsm_flat)[0, 1]


def mantel_restricted_test(
    model_rsm: np.ndarray,
    data_rsm: np.ndarray,
    strata: np.ndarray,
    permutations: int = 1000,
    n_jobs: int = 1,
    seed_base: int = 0,
) -> tuple[float, float]:
    """Mantel test with restricted permutation for factorial designs."""
    model_flat = squareform(model_rsm, checks=False)
    data_flat = squareform(data_rsm, checks=False)
    obs_stat = np.corrcoef(model_flat, data_flat)[0, 1]

    seeds = range(seed_base, seed_base + permutations)
    null_stats = Parallel(n_jobs=n_jobs)(
        delayed(_mantel_worker)(model_rsm, data_flat, strata, s) for s in seeds
    )
    null_stats = np.array(null_stats)
    p_value = (np.sum(null_stats >= obs_stat) + 1) / (permutations + 1)
    return p_value, obs_stat


# --- SRF Structural Test (Hungarian Alignment) ---


def fast_alignment_score(X_std: np.ndarray, W_std: np.ndarray) -> float:
    """Vectorized correlation + Hungarian matching."""
    n = X_std.shape[0]
    corr_matrix = (X_std.T @ W_std) / n
    row_ind, col_ind = linear_sum_assignment(-np.abs(corr_matrix))
    return np.abs(corr_matrix[row_ind, col_ind]).sum()


def stratified_shuffle(
    X_sub: np.ndarray, strata: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Permute rows of X_sub within each stratum group."""
    X_perm = np.empty_like(X_sub)
    unique_strata = np.unique(strata, axis=0)
    for group in unique_strata:
        mask = (strata == group).all(axis=1)
        indices = np.where(mask)[0]
        perm_indices = rng.permutation(indices)
        X_perm[indices] = X_sub[perm_indices]
    return X_perm


def _srf_perm_worker(
    X_std: np.ndarray, W_std: np.ndarray, strata: np.ndarray, seed: int
) -> float:
    """Worker: Restricted permutation and alignment score."""
    rng = np.random.default_rng(seed)
    X_perm = stratified_shuffle(X_std, strata, rng)
    return fast_alignment_score(X_perm, W_std)


def srf_permutation_test(
    X_sub: np.ndarray,
    W: np.ndarray,
    strata: np.ndarray,
    permutations: int = 1000,
    n_jobs: int = 1,
    seed_base: int = 0,
) -> tuple[float, float]:
    """SRF structural test with restricted permutation."""
    X_std = (X_sub - X_sub.mean(0)) / (X_sub.std(0) + 1e-9)
    W_std = (W - W.mean(0)) / (W.std(0) + 1e-9)
    obs_stat = fast_alignment_score(X_std, W_std)

    seeds = range(seed_base, seed_base + permutations)
    null_stats = Parallel(n_jobs=n_jobs)(
        delayed(_srf_perm_worker)(X_std, W_std, strata, s) for s in seeds
    )
    null_stats = np.array(null_stats)
    p_value = (np.sum(null_stats >= obs_stat) + 1) / (permutations + 1)
    return p_value, obs_stat


# --- Factorial Data ---


def create_factorial_data(
    levels: dict[str, list[str]],
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Create factorial design matrix with column ranges per factor."""
    items = list(it.product(*levels.values()))
    col_ranges, features, col_start = {}, [], 0

    for f_idx, (factor, lvls) in enumerate(levels.items()):
        idxs = [lvls.index(item[f_idx]) for item in items]
        Z = np.eye(len(lvls))[idxs]
        features.append(Z)
        col_ranges[factor] = (col_start, col_start + len(lvls))
        col_start += len(lvls)

    X = np.concatenate(features, axis=1)
    return X, col_ranges


# --- High-Dimensional Generative Model ---


def generate_mixing_matrix(k: int, p: int, rng: np.random.Generator) -> np.ndarray:
    """Generate random positive mixing matrix M (k × p)."""
    # Use absolute value of Gaussian to ensure positivity
    M = np.abs(rng.standard_normal((k, p)))
    # Normalize columns to unit norm for stability
    M = M / (np.linalg.norm(M, axis=0, keepdims=True) + 1e-9)
    return M


def generate_high_dim_rsm(
    X: np.ndarray, p: int, snr: float, rng: np.random.Generator | int
) -> np.ndarray:
    """
    Generate RSM from high-dimensional noisy data.

    1. X (N × k) → Y_clean = X @ M (N × p)
    2. Y_noisy = add_positive_noise_with_snr(Y_clean, snr)
    3. S = Y_noisy @ Y_noisy.T (positive RSM)
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    n, k = X.shape

    # Generate mixing matrix
    M = generate_mixing_matrix(k, p, rng)

    # Project to high-dimensional space
    Y_clean = X @ M

    # Add noise in high-dimensional space (clips to positive)
    Y_noisy = add_positive_noise_with_snr(Y_clean, snr, rng=rng)

    # Compute RSM (positive by construction since Y_noisy >= 0)
    S = Y_noisy @ Y_noisy.T

    return S


# --- Main Experiment ---


def run_one_condition(
    snr: float,
    p: int,
    repeat: int,
    X: np.ndarray,
    col_ranges: dict,
    n_permutations: int,
) -> list[dict]:
    """Run all tests for one (snr, p, repeat) combination."""
    k = X.shape[1]
    seed_base = int(snr * 1000 + p + repeat)
    rng = np.random.default_rng(seed_base)

    # Generate RSM from high-dimensional noisy data
    measured_rsm = generate_high_dim_rsm(X, p, snr, rng)

    # Fit SRF
    model = SRF(rank=k, random_state=seed_base, verbose=False)
    W = model.fit_transform(measured_rsm)

    results = []
    for i, (factor_name, (col_start, col_end)) in enumerate(col_ranges.items()):
        X_sub = X[:, col_start:col_end]

        # Strata: all columns except current factor
        non_factor_cols = list(range(0, col_start)) + list(range(col_end, X.shape[1]))
        strata = X[:, non_factor_cols]

        # RSA: Mantel test with restricted permutation
        hypothesis_rsm = X_sub @ X_sub.T
        p_rsa, score_rsa = mantel_restricted_test(
            hypothesis_rsm,
            measured_rsm,
            strata,
            permutations=n_permutations,
            n_jobs=1,
            seed_base=seed_base + 100 * i,
        )

        # SRF: Hungarian alignment with restricted permutation
        p_srf, score_srf = srf_permutation_test(
            X_sub,
            W,
            strata,
            permutations=n_permutations,
            n_jobs=1,
            seed_base=seed_base + 200 * i,
        )

        results.extend([
            {
                "snr": snr,
                "p_dim": p,
                "repeat": repeat,
                "factor": factor_name,
                "method": "RSA",
                "p_value": p_rsa,
                "score": score_rsa,
            },
            {
                "snr": snr,
                "p_dim": p,
                "repeat": repeat,
                "factor": factor_name,
                "method": "SRF",
                "p_value": p_srf,
                "score": score_srf,
            },
        ])

    return results


def main():
    OUTPUT_DIR = get_output_dir()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Factorial design
    levels = {
        "animacy": ["animate", "inanimate"],
        "size": ["small", "medium", "large"],
        "shape": ["round", "angular"],
        "color": ["red", "green", "blue", "yellow"],
    }
    X, col_ranges = create_factorial_data(levels)
    k = X.shape[1]  # k = 11
    n_items = X.shape[0]  # N = 48

    print(f"Factorial: {n_items} items, {k} dims, factors: {list(col_ranges.keys())}")

    # Parameters
    snrs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    p_dims = [k, 50, 100, 500]  # k=11, then increasing dimensionality
    n_repeats = 50
    n_permutations = 1000

    # All conditions
    conditions = [
        (snr, p, rep) for snr in snrs for p in p_dims for rep in range(n_repeats)
    ]
    print(
        f"Running {len(conditions)} conditions "
        f"({len(snrs)} SNRs × {len(p_dims)} p_dims × {n_repeats} repeats × {n_permutations} perms)..."
    )

    # Parallel execution
    all_results_nested = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_one_condition)(snr, p, rep, X, col_ranges, n_permutations)
        for snr, p, rep in conditions
    )

    # Flatten and save
    all_results = [r for batch in all_results_nested for r in batch]
    df = pd.DataFrame(all_results)
    df["significant"] = (df["p_value"] < 0.05).astype(int)
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)

    # Compute power
    power_df = (
        df.groupby(["snr", "p_dim", "method", "factor"])["significant"]
        .mean()
        .reset_index()
    )
    power_df["power"] = power_df["significant"] * 100

    # --- PLOT: Power by p_dim (one subplot per p_dim) ---
    fig, axes = create_figure("full_width", ncols=len(p_dims))

    for idx, p_dim in enumerate(p_dims):
        ax = axes[idx]
        sub = power_df[power_df["p_dim"] == p_dim]

        # Average across factors
        avg = sub.groupby(["snr", "method"])["power"].mean().reset_index()

        for method, color, marker in [
            ("RSA", ROSE, "o"),
            ("SRF", TEAL, "s"),
        ]:
            data = avg[avg["method"] == method].sort_values("snr")
            ax.plot(
                data["snr"],
                data["power"],
                f"{marker}-",
                color=color,
                label=method,
                markersize=6,
                linewidth=1.5,
            )

        ax.axhline(5, color=GRAY_DARK, linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("SNR")
        ax.set_ylabel("Power (%)" if idx == 0 else "")
        ax.set_title(f"p = {p_dim}" + (" (= k)" if p_dim == k else ""))
        ax.set_ylim(-5, 105)
        ax.set_xlim(-0.02, 0.52)
        if idx == 0:
            ax.legend(fontsize=7, loc="lower right", frameon=False)
        despine(ax)

    save_figure(fig, OUTPUT_DIR / "power_by_pdim.pdf")

    # Print summary
    print("\n" + "=" * 70)
    print("POWER SUMMARY (% significant at alpha=0.05, averaged across factors)")
    print("=" * 70)
    summary = (
        power_df.groupby(["p_dim", "snr", "method"])["power"]
        .mean()
        .unstack("method")
    )
    print(summary.round(1))
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
