"""
RSA vs SRF Power Comparison.

Adds noise directly to factorial embedding X, computes similarity,
then compares RSA (Mantel) vs SRF (LOO correlation) power.
"""

from datetime import datetime
from pathlib import Path
import itertools as it

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr

from pysrf import SRF
from tools.rsa import compute_similarity
from src.utils.helpers import add_positive_noise_with_snr
from src.colors import ROSE, TEAL, GRAY_LIGHT
from src.utils.figure_theme import create_figure, despine, save_figure
from src.utils import get_output_dir


def create_factorial_design(levels: dict[str, list[str]]) -> tuple[np.ndarray, dict, list]:
    """Create factorial design matrix (one-hot encoded)."""
    items = list(it.product(*levels.values()))
    col_ranges, features, col_start = {}, [], 0

    for f_idx, (factor, lvls) in enumerate(levels.items()):
        idxs = [lvls.index(item[f_idx]) for item in items]
        Z = np.eye(len(lvls))[idxs]
        features.append(Z)
        col_ranges[factor] = (col_start, col_start + len(lvls))
        col_start += len(lvls)

    X = np.concatenate(features, axis=1)
    return X, col_ranges, items


def hungarian_match(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Find optimal column permutation to align W to X."""
    k_w, k_x = W.shape[1], X.shape[1]
    corr_matrix = np.zeros((k_w, k_x))
    for i in range(k_w):
        for j in range(k_x):
            c = pearsonr(W[:, i], X[:, j]).statistic
            corr_matrix[i, j] = c if not np.isnan(c) else 0

    row_ind, col_ind = linear_sum_assignment(-np.abs(corr_matrix))
    perm = np.zeros(k_x, dtype=int)
    for w_col, x_col in zip(row_ind, col_ind):
        perm[x_col] = w_col
    return perm


def loo_correlation(W: np.ndarray, X: np.ndarray) -> float:
    """LOO matching correlation (PI's simpler approach).

    For each item i:
    1. Hungarian match W[~i] to X[~i]
    2. Apply permutation to W[i] and save

    Final: correlate reconstructed W_test with X.
    """
    N, k = X.shape
    W_test = np.zeros((N, k))

    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        perm = hungarian_match(W[mask], X[mask])
        W_test[i] = W[i, perm]

    return pearsonr(W_test.ravel(), X.ravel()).statistic


def rsa_mantel_test(
    H: np.ndarray,
    RSM: np.ndarray,
    n_perms: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """RSA Mantel test with permutation null."""
    from scipy.spatial.distance import squareform

    h_flat = squareform(H, checks=False)
    rsm_flat = squareform(RSM, checks=False)

    obs = pearsonr(h_flat, rsm_flat).statistic

    nulls = []
    for _ in range(n_perms):
        perm = rng.permutation(H.shape[0])
        H_perm = H[perm][:, perm]
        h_perm_flat = squareform(H_perm, checks=False)
        nulls.append(pearsonr(h_perm_flat, rsm_flat).statistic)

    p_value = (np.sum(np.array(nulls) >= obs) + 1) / (n_perms + 1)
    return p_value, obs


def srf_loo_test(
    RSM: np.ndarray,
    X: np.ndarray,
    rank: int,
    n_perms: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """SRF LOO correlation test with permutation null."""
    model = SRF(rank=rank, random_state=int(rng.integers(10000)), verbose=False)
    W = model.fit_transform(RSM)

    obs_corr = loo_correlation(W, X)

    nulls = []
    for _ in range(n_perms):
        perm = rng.permutation(X.shape[0])
        nulls.append(loo_correlation(W, X[perm]))

    p_value = (np.sum(np.array(nulls) >= obs_corr) + 1) / (n_perms + 1)
    return p_value, obs_corr


def run_one_condition(
    snr: float,
    repeat: int,
    X: np.ndarray,
    k: int,
    n_perms: int,
) -> dict:
    """Run RSA and SRF tests for one SNR and repeat."""
    seed = int(snr * 10000 + repeat)
    rng = np.random.default_rng(seed)

    # Add noise to X directly
    X_noisy = add_positive_noise_with_snr(X, snr, rng=seed)

    # Compute similarity from noisy embedding
    RSM = compute_similarity(X_noisy, X_noisy, metric="linear")

    # Hypothesis RSM (from clean X)
    H = X @ X.T

    # RSA test
    p_rsa, corr_rsa = rsa_mantel_test(H, RSM, n_perms, rng)

    # SRF test (LOO correlation)
    p_srf, corr_srf = srf_loo_test(RSM, X, k, n_perms, rng)

    return {
        'snr': snr,
        'repeat': repeat,
        'p_rsa': p_rsa,
        'p_srf': p_srf,
        'corr_rsa': corr_rsa,
        'corr_srf': corr_srf,
    }


def main():
    OUTPUT_DIR = get_output_dir()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Factorial design
    levels = {
        'animacy': ['animate', 'inanimate'],
        'size': ['small', 'medium', 'large'],
        'shape': ['round', 'angular'],
        'color': ['red', 'green', 'blue', 'yellow'],
    }
    X, col_ranges, _ = create_factorial_design(levels)
    N, k = X.shape

    print(f"Factorial design: {N} items, {k} dimensions")
    print(f"Factors: {', '.join(f'{f} ({e-s} levels)' for f, (s, e) in col_ranges.items())}")

    # Parameters
    snrs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_repeats = 50
    n_perms = 500

    conditions = [(snr, rep) for snr in snrs for rep in range(n_repeats)]
    print(f"\nRunning {len(conditions)} conditions ({n_perms} perms each)")

    # Run
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_one_condition)(snr, rep, X, k, n_perms)
        for snr, rep in conditions
    )

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / 'results.csv', index=False)

    # Compute power
    df['sig_rsa'] = df['p_rsa'] < 0.05
    df['sig_srf'] = df['p_srf'] < 0.05

    power_df = df.groupby('snr').agg({
        'sig_rsa': 'mean',
        'sig_srf': 'mean',
        'corr_rsa': 'mean',
        'corr_srf': 'mean',
    }).reset_index()
    power_df.columns = ['snr', 'power_rsa', 'power_srf', 'corr_rsa', 'corr_srf']

    # Plot: Power vs SNR
    fig, ax = create_figure('single')

    ax.plot(power_df['snr'], power_df['power_rsa'] * 100,
            'o-', color=ROSE, label='RSA (Mantel)', markersize=6, linewidth=1.5)
    ax.plot(power_df['snr'], power_df['power_srf'] * 100,
            's-', color=TEAL, label='SRF (LOO)', markersize=6, linewidth=1.5)

    ax.axhline(5, color=GRAY_LIGHT, linestyle=':', linewidth=1)
    ax.set_xlabel('SNR')
    ax.set_ylabel('Power (%)')
    ax.set_ylim(-5, 105)
    ax.set_xlim(-0.05, 1.05)
    ax.legend(fontsize=8, loc='lower right', frameon=False)
    despine(ax)

    save_figure(fig, OUTPUT_DIR / 'power_vs_snr.pdf')

    # Plot: Correlations
    fig, ax = create_figure('single')

    ax.plot(power_df['snr'], power_df['corr_rsa'],
            'o-', color=ROSE, label='RSA', markersize=6, linewidth=1.5)
    ax.plot(power_df['snr'], power_df['corr_srf'],
            's-', color=TEAL, label='SRF (LOO)', markersize=6, linewidth=1.5)

    ax.set_xlabel('SNR')
    ax.set_ylabel('Correlation')
    ax.set_xlim(-0.05, 1.05)
    ax.legend(fontsize=8, loc='lower right', frameon=False)
    despine(ax)

    save_figure(fig, OUTPUT_DIR / 'correlation_vs_snr.pdf')

    # Print summary
    print("\n" + "=" * 60)
    print("POWER SUMMARY")
    print("=" * 60)
    print(power_df[['snr', 'power_rsa', 'power_srf']].to_string(index=False))

    print("\n" + "=" * 60)
    print("CORRELATIONS")
    print("=" * 60)
    print(power_df[['snr', 'corr_rsa', 'corr_srf']].to_string(index=False))

    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
