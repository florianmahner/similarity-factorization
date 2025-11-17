from __future__ import annotations

import itertools as it

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pysrf import SRF
from statsmodels.stats.multitest import multipletests

from analyses.rsa_testing import mantel_test, permutation_test
from utils.helpers import add_positive_noise_with_snr, align_latent_dimensions
from tools.rsa import compute_similarity


def create_factorial_data(
    levels: dict[str, list[str]], max_objects: int | None = None
) -> tuple[np.ndarray, dict[str, np.ndarray], list[tuple[str, ...]]]:
    items = list(it.product(*levels.values()))
    n = len(items)

    if max_objects is not None and n > max_objects:
        rng = np.random.default_rng(42)
        selected_indices = rng.choice(n, size=max_objects, replace=False)
        items = [items[i] for i in selected_indices]

    blocks: dict[str, np.ndarray] = {}
    features: list[np.ndarray] = []
    for f_idx, (factor, lvls) in enumerate(levels.items()):
        idxs = [lvls.index(item[f_idx]) for item in items]
        Z = np.eye(len(lvls))[idxs]
        blocks[factor] = Z
        features.append(Z)

    X = np.concatenate(features, axis=1)
    return X, blocks, items


def _evaluate_factorial_condition(
    model: SRF,
    true_data: np.ndarray,
    hypotheses: list[np.ndarray],
    snr: float,
    repeat: int,
    n_permutations: int,
    alpha: float,
    similarity_metric: str,
):
    seed = int(snr * 100 + repeat)
    noisy_data = add_positive_noise_with_snr(true_data, snr, rng=seed)

    measured_similarity = compute_similarity(
        noisy_data, noisy_data, metric=similarity_metric
    )
    w = model.fit_transform(measured_similarity)

    rank = true_data.shape[1]
    w_aligned = align_latent_dimensions(true_data, w)

    rsa_tests = [
        mantel_test(
            h,
            measured_similarity,
            permutations=n_permutations,
            random_state=seed + 100 * i,
            two_sided=True,
        )
        for i, h in enumerate(hypotheses)
    ]

    latent_tests = [
        permutation_test(
            true_data[:, i],
            w_aligned[:, i],
            permutations=n_permutations,
            random_state=seed + 200 * i,
            two_sided=True,
        )
        for i in range(rank)
    ]

    test_results = {"RSA": rsa_tests, "SRF": latent_tests}
    rows = []
    for method, tests in test_results.items():
        raw_ps = [t[0] for t in tests]
        reject, corr_ps = multipletests(raw_ps, alpha, method="fdr_bh")[:2]
        for i, (test, corrected_p, rej) in enumerate(zip(tests, corr_ps, reject)):
            rows.append(
                {
                    "snr": snr,
                    "repeat": repeat,
                    "hypothesis": i + 1,
                    "method": method,
                    "raw_p": test[0],
                    "corrected_p": corrected_p,
                    "significant": bool(rej),
                }
            )

    return rows


def run_factorial_experiment(
    snrs: list[float],
    n_repeats: int,
    n_permutations: int,
    max_jobs: int,
    similarity_metric: str,
    levels: dict[str, list[str]],
    alpha: float = 0.05,
) -> pd.DataFrame:
    X, blocks, _ = create_factorial_data(levels)
    rank = X.shape[1]
    hypotheses = [
        compute_similarity(Z, Z, metric=similarity_metric) for Z in blocks.values()
    ]
    model = SRF(rank=rank, random_state=0)

    tasks = list(it.product(snrs, range(n_repeats)))
    results = Parallel(n_jobs=max_jobs, verbose=10)(
        delayed(_evaluate_factorial_condition)(
            model,
            X,
            hypotheses,
            snr,
            repeat,
            n_permutations,
            alpha,
            similarity_metric,
        )
        for snr, repeat in tasks
    )

    return pd.DataFrame([row for sub in results for row in sub])
