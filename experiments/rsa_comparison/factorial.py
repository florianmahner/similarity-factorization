from __future__ import annotations

import itertools as it
import json
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr

from pysrf import SRF
from statsmodels.stats.multitest import multipletests

from tools.rsa import mantel_test, compute_similarity
from src.utils.helpers import add_positive_noise_with_snr


def create_factorial_data(
    levels: dict[str, list[str]],
) -> tuple[np.ndarray, dict[str, tuple[int, int]], list[tuple[str, ...]]]:
    """Create factorial design matrix with column ranges per factor."""
    items = list(it.product(*levels.values()))

    col_ranges: dict[str, tuple[int, int]] = {}
    features: list[np.ndarray] = []
    col_start = 0

    for f_idx, (factor, lvls) in enumerate(levels.items()):
        idxs = [lvls.index(item[f_idx]) for item in items]
        Z = np.eye(len(lvls))[idxs]
        features.append(Z)
        col_ranges[factor] = (col_start, col_start + len(lvls))
        col_start += len(lvls)

    X = np.concatenate(features, axis=1)
    return X, col_ranges, items


def do_linear_assignment(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Match W columns to X columns. Returns column indices."""
    rank = X.shape[1]
    cost = np.zeros((rank, rank))
    for i in range(rank):
        for j in range(rank):
            c = pearsonr(X[:, i], W[:, j]).statistic
            cost[i, j] = -abs(c) if not np.isnan(c) else 0
    _, col_ind = linear_sum_assignment(cost)
    return col_ind


def factor_correlation(
    X: np.ndarray, W: np.ndarray, col_ind: np.ndarray, col_start: int, col_end: int
) -> float:
    """Correlation for one factor's columns after assignment."""
    corrs = []
    for i in range(col_start, col_end):
        c = pearsonr(X[:, i], W[:, col_ind[i]]).statistic
        corrs.append(abs(c) if not np.isnan(c) else 0)
    return np.mean(corrs)


def srf_permutation_test(
    X: np.ndarray,
    W: np.ndarray,
    col_start: int,
    col_end: int,
    permutations: int = 1000,
    random_state: int | None = None,
) -> tuple[float, float]:
    """
    Permutation test for one factor using linear assignment.

    Observed: assign W to X, compute factor correlation
    Null: permute X rows, assign W to permuted X, compute factor correlation
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]

    # Observed
    col_ind = do_linear_assignment(X, W)
    obs_corr = factor_correlation(X, W, col_ind, col_start, col_end)

    # Null distribution
    nulls = np.zeros(permutations)
    for i in range(permutations):
        perm = rng.permutation(n)
        X_perm = X[perm]
        col_ind_perm = do_linear_assignment(X_perm, W)
        nulls[i] = factor_correlation(X_perm, W, col_ind_perm, col_start, col_end)

    p_value = (np.sum(nulls >= obs_corr) + 1) / (permutations + 1)
    return p_value, obs_corr


def run(cfg: DictConfig) -> None:
    """
    Compare RSA vs SRF on factorial design.

    Both methods test the same 4 factors:
    - RSA: Mantel test (factor RSM vs measured similarity)
    - SRF: Permutation test (linear assignment + factor correlation)
    """
    levels = OmegaConf.to_container(cfg.levels, resolve=True)
    X, col_ranges, _ = create_factorial_data(levels)
    rank = X.shape[1]

    # Seed for this condition
    seed_base = int(cfg.snrs * 100 + cfg.seed)

    # Add noise and compute similarity
    noisy_data = add_positive_noise_with_snr(X, cfg.snrs, rng=seed_base)
    measured_similarity = compute_similarity(
        noisy_data, noisy_data, metric=cfg.similarity_metric
    )

    # Fit SRF
    model = SRF(rank=rank, random_state=0, verbose=False)
    W = model.fit_transform(measured_similarity)

    # Test each factor
    rows = []
    for i, (factor_name, (col_start, col_end)) in enumerate(col_ranges.items()):
        # Hypothesis RSM for this factor
        Z = X[:, col_start:col_end]
        hypothesis_rsm = Z @ Z.T

        # RSA: Mantel test
        p_rsa, _, obs_rsa = mantel_test(
            hypothesis_rsm,
            measured_similarity,
            permutations=cfg.n_permutations,
            random_state=seed_base + 100 * i,
            two_sided=True,
        )

        # SRF: Permutation test with linear assignment
        p_srf, obs_srf = srf_permutation_test(
            X,
            W,
            col_start,
            col_end,
            permutations=cfg.n_permutations,
            random_state=seed_base + 200 * i,
        )

        rows.append(
            {
                "snr": float(cfg.snrs),
                "repeat": int(cfg.seed),
                "factor": factor_name,
                "method": "RSA",
                "raw_p": float(p_rsa),
                "correlation": float(obs_rsa),
            }
        )
        rows.append(
            {
                "snr": float(cfg.snrs),
                "repeat": int(cfg.seed),
                "factor": factor_name,
                "method": "SRF",
                "raw_p": float(p_srf),
                "correlation": float(obs_srf),
            }
        )

    # FDR correction within each method
    for method in ["RSA", "SRF"]:
        method_rows = [r for r in rows if r["method"] == method]
        raw_ps = [r["raw_p"] for r in method_rows]
        reject, corr_ps = multipletests(raw_ps, cfg.alpha, method="fdr_bh")[:2]
        for r, corrected_p, rej in zip(method_rows, corr_ps, reject):
            r["corrected_p"] = float(corrected_p)
            r["significant"] = bool(rej)

    # Save results
    output_file = Path.cwd() / "results.json"
    with open(output_file, "w") as f:
        json.dump(rows, f)
