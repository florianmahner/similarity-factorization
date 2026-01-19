from __future__ import annotations

import itertools as it
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score

from pysrf import SRF
from statsmodels.stats.multitest import multipletests

from scipy.spatial.distance import squareform

from tools.rsa import compute_similarity
from src.utils.helpers import add_positive_noise_with_snr


# --- Restricted Mantel Test (Parallelized) ---


def get_stratified_indices(
    n: int, strata: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
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

    # 1. Get stratified shuffle indices
    idx = get_stratified_indices(n, strata, rng)

    # 2. Reorder Model RSM (Row i -> Row idx[i], Col j -> Col idx[j])
    model_perm = model_rsm[idx, :][:, idx]

    # 3. Correlate (using flattened upper triangle)
    model_flat = squareform(model_perm, checks=False)

    return np.corrcoef(model_flat, data_rsm_flat)[0, 1]


def mantel_restricted_test(
    model_rsm: np.ndarray,
    data_rsm: np.ndarray,
    strata: np.ndarray,
    permutations: int = 1000,
    n_jobs: int = -1,
    seed_base: int = 0,
) -> tuple[float, float]:
    """Parallelized Mantel test with restricted permutation for factorial designs."""
    # Pre-calculate observed
    model_flat = squareform(model_rsm, checks=False)
    data_flat = squareform(data_rsm, checks=False)
    obs_stat = np.corrcoef(model_flat, data_flat)[0, 1]

    # Run Parallel Permutations
    seeds = range(seed_base, seed_base + permutations)
    null_stats = Parallel(n_jobs=n_jobs)(
        delayed(_mantel_worker)(model_rsm, data_flat, strata, s) for s in seeds
    )

    null_stats = np.array(null_stats)
    p_value = (np.sum(null_stats >= obs_stat) + 1) / (permutations + 1)
    return p_value, obs_stat


# --- Optimized Core Functions ---


def fast_alignment_score(X_std: np.ndarray, W_std: np.ndarray) -> float:
    """
    Vectorized correlation + Hungarian matching.
    """
    n = X_std.shape[0]
    # Fast Vectorized Correlation: (1/n) * X.T @ W
    corr_matrix = (X_std.T @ W_std) / n
    # Hungarian Algo on Negative Abs Correlation
    row_ind, col_ind = linear_sum_assignment(-np.abs(corr_matrix))
    return np.abs(corr_matrix[row_ind, col_ind]).sum()


def stratified_shuffle(
    X_sub: np.ndarray, strata: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    Permutes rows of X_sub SEPARATELY within each group defined by strata.
    Preserves orthogonality with respect to the strata (other factors).
    """
    X_perm = np.empty_like(X_sub)
    # Get unique groups (e.g., Animacy levels)
    unique_strata = np.unique(strata, axis=0)

    for group in unique_strata:
        # Find indices belonging to this group
        # (Using broadcasting for multi-column strata)
        mask = (strata == group).all(axis=1)
        indices = np.where(mask)[0]

        # Shuffle these indices among themselves
        perm_indices = rng.permutation(indices)
        X_perm[indices] = X_sub[perm_indices]

    return X_perm


def _perm_worker(
    X_std: np.ndarray, W_std: np.ndarray, strata: np.ndarray, seed: int
) -> float:
    """
    Worker: Performs RESTRICTED permutation and computes score.
    """
    rng = np.random.default_rng(seed)

    # Critical: Permute X only within the fixed levels of other factors
    X_perm = stratified_shuffle(X_std, strata, rng)

    return fast_alignment_score(X_perm, W_std)


def srf_permutation_test_parallel(
    X_sub: np.ndarray,
    W: np.ndarray,
    strata: np.ndarray,
    permutations: int = 1000,
    n_jobs: int = -1,
    seed_base: int = 0,
) -> tuple[float, float]:
    """
    Parallelized Permutation Test with Stratification.
    """
    # Standardize inputs once
    X_std = (X_sub - X_sub.mean(0)) / (X_sub.std(0) + 1e-9)
    W_std = (W - W.mean(0)) / (W.std(0) + 1e-9)

    # Observed Score
    obs_stat = fast_alignment_score(X_std, W_std)

    # Parallel Restricted Null Distribution
    seeds = range(seed_base, seed_base + permutations)
    null_stats = Parallel(n_jobs=n_jobs)(
        delayed(_perm_worker)(X_std, W_std, strata, s) for s in seeds
    )

    null_stats = np.array(null_stats)
    p_value = (np.sum(null_stats >= obs_stat) + 1) / (permutations + 1)

    return p_value, obs_stat


def create_factorial_data(
    levels: dict[str, list[str]],
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
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


def run(cfg: DictConfig) -> None:
    levels = OmegaConf.to_container(cfg.levels, resolve=True)
    X, col_ranges = create_factorial_data(levels)
    rank = X.shape[1]
    seed_base = int(cfg.snrs * 100 + cfg.seed)

    # 1. Add Noise & Compute Similarity
    noisy_data = add_positive_noise_with_snr(X, cfg.snrs, rng=seed_base)
    measured_similarity = compute_similarity(
        noisy_data, noisy_data, metric=cfg.similarity_metric
    )

    # 2. Fit SRF
    model = SRF(rank=rank, random_state=0, verbose=False)
    W = model.fit_transform(measured_similarity)

    rows = []

    # 3. Iterate Factors
    for i, (factor_name, (col_start, col_end)) in enumerate(col_ranges.items()):
        # Current Factor (e.g., Color)
        X_sub = X[:, col_start:col_end]

        # Strata: All columns of X *except* the current factor
        # This defines the "blocks" we must respect during permutation
        non_factor_cols = list(range(0, col_start)) + list(range(col_end, X.shape[1]))
        strata = X[:, non_factor_cols]

        # --- A. RSA: Mantel Test (restricted permutation, parallelized) ---
        hypothesis_rsm = X_sub @ X_sub.T
        p_rsa, obs_rsa = mantel_restricted_test(
            hypothesis_rsm,
            measured_similarity,
            strata=strata,
            permutations=cfg.n_permutations,
            n_jobs=-1,
            seed_base=seed_base + 100 * i,
        )

        # --- B. SRF: Structural Recovery (Stratified) ---
        p_srf, obs_srf = srf_permutation_test_parallel(
            X_sub,
            W,
            strata=strata,  # Pass the other factors as constraints
            permutations=cfg.n_permutations,
            n_jobs=-1,
            seed_base=seed_base + 200 * i,
        )

        # --- C. Decoding (LOO) ---
        y_labels = np.argmax(X_sub, axis=1)
        clf = KNeighborsClassifier(n_neighbors=1, metric="correlation")
        loo_acc = np.mean(
            cross_val_score(clf, W, y_labels, cv=LeaveOneOut(), scoring="accuracy")
        )

        base_res = {
            "snr": float(cfg.snrs),
            "repeat": int(cfg.seed),
            "factor": factor_name,
        }
        rows.append(
            {
                **base_res,
                "method": "RSA",
                "p_val": float(p_rsa),
                "score": float(obs_rsa),
                "type": "structure",
            }
        )
        rows.append(
            {
                **base_res,
                "method": "SRF_Struct",
                "p_val": float(p_srf),
                "score": float(obs_srf),
                "type": "structure",
            }
        )
        rows.append(
            {
                **base_res,
                "method": "SRF_Decod",
                "p_val": np.nan,
                "score": float(loo_acc),
                "type": "decoding",
            }
        )

    # Save
    output_file = Path.cwd() / "results.json"
    with open(output_file, "w") as f:
        json.dump(rows, f)
