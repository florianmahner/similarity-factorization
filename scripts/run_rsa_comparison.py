#!/usr/bin/env python3
"""
Hypothesis testing benchmark for comparing RSA, NMF, and Latent space methods
across different signal-to-noise ratios and object counts.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

from utils.helpers import add_noise_with_snr, align_latent_dimensions
from utils.io import load_spose_embedding
from models.admm import ADMM
from tools.rsa import compute_similarity

# TODO make condition class probably.


def mantel_test(A, B, permutations=10000, random_state=None, two_sided=False):
    """Mantel test for correlation between two distance/similarity matrices."""
    if random_state is not None:
        np.random.seed(random_state)

    idx_upper = np.triu_indices_from(A, k=1)
    sim1, sim2 = A[idx_upper], B[idx_upper]
    obs = (
        np.abs(pearsonr(sim1, sim2).statistic)
        if two_sided
        else pearsonr(sim1, sim2).statistic
    )

    nulls = np.zeros(permutations)
    for i in range(permutations):
        perm = np.random.permutation(B.shape[0])
        Bp = B[perm][:, perm]
        sc = pearsonr(sim1, Bp[idx_upper]).statistic
        nulls[i] = np.abs(sc) if two_sided else sc

    return (np.sum(nulls >= obs) + 1) / (permutations + 1), nulls, obs


def permutation_test(A, B, permutations=10000, random_state=None, two_sided=False):
    """Permutation test for correlation between two vectors."""
    if random_state is not None:
        np.random.seed(random_state)

    observed_corr = pearsonr(A, B).statistic
    if two_sided:
        observed_corr = np.abs(observed_corr)

    greater = 0
    null_corrs = np.zeros(permutations)
    for i in range(permutations):
        perm = np.random.permutation(B)
        perm_corr = pearsonr(A, perm).statistic
        if two_sided:
            perm_corr = np.abs(perm_corr)
        if perm_corr >= observed_corr:
            greater += 1
        null_corrs[i] = perm_corr

    return (greater + 1) / (permutations + 1), null_corrs, observed_corr


def evaluate_condition(
    model,
    data,
    hypotheses,
    n_objects,
    snr,
    repeat,
    n_permutations,
    alpha=0.05,
    similarity_metric="linear",
):
    """Process a single experimental run."""
    seed = int(n_objects * 1000 + snr * 100 + repeat)
    noisy_data = add_noise_with_snr(data, snr, rng=seed)

    measured_similarity = compute_similarity(noisy_data, noisy_data, similarity_metric)
    w = model.fit_transform(measured_similarity)
    denoised_similarity = model.reconstruct()

    rank = data.shape[1]
    w_aligned = align_latent_dimensions(data, w)
    rsa_tests = [
        mantel_test(h, measured_similarity, permutations=n_permutations)
        for h in hypotheses
    ]
    latent_tests = [
        permutation_test(data[:, i], w_aligned[:, i], permutations=n_permutations)
        for i in range(rank)
    ]

    test_results = {"RSA": rsa_tests, "NMF": latent_tests}

    results = []
    for method, tests in test_results.items():
        raw_ps = [t[0] for t in tests]
        reject, corr_ps = multipletests(raw_ps, alpha, method="fdr_bh")[:2]

        for i, (test, corrected_p, rej) in enumerate(zip(tests, corr_ps, reject)):
            results.append(
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

    return results


# TODO merge this into one function.
def process_single_condition(
    full_data,
    dims,
    n_objects,
    snr,
    repeat,
    n_permutations,
    similarity_metric="linear",
):
    """Process a single experimental condition."""
    seed = int(n_objects * 10000 + snr * 100 + repeat)
    np.random.seed(seed)

    selected_objects = np.random.choice(
        full_data.shape[0], size=n_objects, replace=False
    )
    data = full_data[selected_objects][:, dims]

    hypotheses = []
    rank = len(dims)
    for i in range(rank):
        x_i = data[:, [i]]
        s_i = compute_similarity(x_i, x_i, similarity_metric)
        hypotheses.append(s_i)

    model = ADMM(rank=rank, verbose=False, tol=0.0)
    return evaluate_condition(
        model, data, hypotheses, n_objects, snr, repeat, n_permutations
    )


def run_experiment(
    object_counts,
    num_dims,
    snrs,
    n_repeats,
    n_permutations,
    similarity_metric,
    max_jobs,
    output_path,
):
    """Main experiment orchestrator."""

    full_data = load_spose_embedding(max_objects=None, num_dims=49)
    np.random.seed(42)
    dims = np.random.choice(full_data.shape[1], size=num_dims, replace=False)

    all_tasks = [
        (n_objects, snr, repeat)
        for n_objects in object_counts
        for snr in snrs
        for repeat in range(n_repeats)
    ]

    print(f"Running {len(all_tasks)} conditions in parallel")
    results = Parallel(n_jobs=max_jobs)(
        delayed(process_single_condition)(
            full_data, dims, n_objects, snr, repeat, n_permutations, similarity_metric
        )
        for n_objects, snr, repeat in all_tasks
    )

    results_df = pd.DataFrame([item for sublist in results for item in sublist])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    return results_df


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Hypothesis testing benchmark for RSA, NMF, and Latent space methods"
    )

    parser.add_argument("--object-counts", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--num-dims", type=int, default=10)
    parser.add_argument(
        "--snrs", nargs="+", type=float, default=np.linspace(0.0, 1.0, 10)
    )
    parser.add_argument("--n-repeats", type=int, default=500)
    parser.add_argument("--n-permutations", type=int, default=5_000)
    parser.add_argument(
        "--similarity-metric",
        type=str,
        default="linear",
        choices=["linear", "cosine", "euclidean"],
    )
    parser.add_argument("--max-jobs", type=int, default=140)
    parser.add_argument(
        "--output-path",
        type=str,
        default="./results/rsa_comparison.csv",
    )

    args = parser.parse_args()

    run_experiment(
        args.object_counts,
        args.num_dims,
        args.snrs,
        args.n_repeats,
        args.n_permutations,
        args.similarity_metric,
        args.max_jobs,
        args.output_path,
    )


if __name__ == "__main__":
    main()
