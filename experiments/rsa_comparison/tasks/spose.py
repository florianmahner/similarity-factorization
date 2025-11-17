from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

from pysrf import SRF

from analyses.rsa_testing import mantel_test, permutation_test
from utils.helpers import add_positive_noise_with_snr, align_latent_dimensions
from utils.io import load_spose_embedding


def _evaluate_condition(
    model: SRF,
    data: np.ndarray,
    hypotheses: list[np.ndarray],
    n_objects: int,
    snr: float,
    repeat: int,
    n_permutations: int,
    alpha: float,
) -> list[dict]:
    seed_base = 10000 + 97 * (repeat + 1)
    noisy_data = add_positive_noise_with_snr(data, snr, rng=seed_base)

    measured_similarity = noisy_data @ noisy_data.T
    w = model.fit_transform(measured_similarity)

    rank = data.shape[1]
    w_aligned = align_latent_dimensions(data, w)

    rsa_tests = [
        mantel_test(
            h,
            measured_similarity,
            permutations=n_permutations,
            random_state=seed_base + 100 * i,
            two_sided=True,
        )
        for i, h in enumerate(hypotheses)
    ]
    latent_tests = [
        permutation_test(
            data[:, i],
            w_aligned[:, i],
            permutations=n_permutations,
            random_state=seed_base + 200 * i,
            two_sided=True,
        )
        for i in range(rank)
    ]

    rows = []
    for method, tests in {"RSA": rsa_tests, "SRF": latent_tests}.items():
        raw_ps = [t[0] for t in tests]
        reject, corr_ps = multipletests(raw_ps, alpha, method="fdr_bh")[:2]
        for i, (test, corrected_p, rej) in enumerate(zip(tests, corr_ps, reject)):
            rows.append(
                {
                    "n_objects": n_objects,
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


def _process_single_condition(
    full_data: np.ndarray,
    dims: np.ndarray,
    selected_objects: np.ndarray,
    snr: float,
    repeat: int,
    n_permutations: int,
) -> list[dict]:
    data = full_data[selected_objects][:, dims]

    hypotheses = []
    rank = len(dims)
    for i in range(rank):
        x_i = data[:, [i]]
        hypotheses.append(x_i @ x_i.T)

    model = SRF(rank=rank, verbose=False, tol=0.0, random_state=repeat)
    return _evaluate_condition(
        model,
        data,
        hypotheses,
        len(selected_objects),
        snr,
        repeat,
        n_permutations,
        alpha=0.05,
    )


def run_spose_experiment(
    object_counts: list[int],
    num_dims: int,
    snrs: list[float],
    n_repeats: int,
    n_permutations: int,
    max_jobs: int,
) -> pd.DataFrame:
    full_data = load_spose_embedding(num_dims=49)
    rng = np.random.default_rng(0)
    dims = rng.choice(full_data.shape[1], size=num_dims, replace=False)

    selected_objects = {
        n_objects: rng.choice(full_data.shape[0], size=n_objects, replace=False)
        for n_objects in object_counts
    }

    tasks = [
        (n_objects, snr, repeat)
        for n_objects in object_counts
        for snr in snrs
        for repeat in range(n_repeats)
    ]

    results = Parallel(n_jobs=max_jobs, verbose=10)(
        delayed(_process_single_condition)(
            full_data,
            dims,
            selected_objects[n_objects],
            snr,
            repeat,
            n_permutations,
        )
        for n_objects, snr, repeat in tasks
    )

    return pd.DataFrame([row for sub in results for row in sub])

