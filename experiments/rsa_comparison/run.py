#!/usr/bin/env python3
"""
RSA Comparison and Hypothesis Testing.

Compares RSA, NMF, and Latent space methods across different experimental designs:
- SPOSE: Varying object counts and dimensions with SPOSE embeddings
- Factorial: Factorial design with specific factors (animacy, size, etc.)

Usage:
    ./run rsa_comparison --mode spose
    ./run rsa_comparison --mode factorial
"""

import argparse
import itertools as it
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

from src.analysis.rsa_testing import mantel_test, permutation_test
from src.utils.helpers import add_positive_noise_with_snr, align_latent_dimensions
from src.utils.io import load_spose_embedding
from tools.rsa import compute_similarity
from pysrf import SRF


def create_factorial_data(levels: dict, max_objects: int | None = None):
    """
    Create factorial design data.

    Parameters
    ----------
    levels : dict
        Dictionary mapping factor names to their levels
    max_objects : int, optional
        Maximum number of objects to sample

    Returns
    -------
    X : ndarray
        Feature matrix (n_objects, n_features)
    blocks : dict
        One-hot encodings for each factor
    items : list
        List of factorial combinations
    """
    items = list(it.product(*levels.values()))
    n = len(items)

    if max_objects is not None and n > max_objects:
        np.random.seed(42)
        selected_indices = np.random.choice(n, size=max_objects, replace=False)
        items = [items[i] for i in selected_indices]

    blocks = {}
    features = []
    for f_idx, f in enumerate(levels.keys()):
        lvls = levels[f]
        Z = np.eye(len(lvls))[[lvls.index(item[f_idx]) for item in items]]
        blocks[f] = Z
        features.append(Z)

    X = np.concatenate(features, axis=1)
    return X, blocks, items


def evaluate_factorial_condition(
    model,
    true_data: np.ndarray,
    hypotheses: list[np.ndarray],
    snr: float,
    repeat: int,
    n_permutations: int,
    alpha: float = 0.05,
    similarity_metric: str = "linear",
):
    """Evaluate single factorial condition."""
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

    results = []
    for method, tests in test_results.items():
        raw_ps = [t[0] for t in tests]
        reject, corr_ps = multipletests(raw_ps, alpha, method="fdr_bh")[:2]

        for i, (test, corrected_p, rej) in enumerate(zip(tests, corr_ps, reject)):
            results.append(
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

    return results


def evaluate_condition(
    model,
    data,
    hypotheses,
    n_objects,
    snr,
    repeat,
    n_permutations,
    alpha=0.05,
):
    """Process a single experimental run."""
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

    test_results = {"RSA": rsa_tests, "SRF": latent_tests}

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


def process_single_condition(
    full_data,
    dims,
    selected_objects,
    snr,
    repeat,
    n_permutations,
):
    """Process a single experimental condition."""
    data = full_data[selected_objects][:, dims]

    hypotheses = []
    rank = len(dims)
    for i in range(rank):
        x_i = data[:, [i]]
        s_i = x_i @ x_i.T
        hypotheses.append(s_i)

    seed_base = 10000 + 97 * (repeat + 1)
    model = SRF(rank=rank, verbose=False, tol=0.0, random_state=seed_base)
    return evaluate_condition(
        model,
        data,
        hypotheses,
        len(selected_objects),
        snr,
        repeat,
        n_permutations,
    )


def run_factorial_experiment(
    snrs: list[float],
    n_repeats: int,
    n_permutations: int,
    max_jobs: int,
    output_path: str,
    levels: dict,
    similarity_metric: str = "linear",
):
    """Run factorial design experiment."""
    X, blocks, items = create_factorial_data(levels)
    rank = X.shape[1]

    hypotheses = [
        compute_similarity(Z, Z, metric=similarity_metric) for Z in blocks.values()
    ]

    model = SRF(rank=rank, random_state=0)

    all_tasks = list(it.product(snrs, range(n_repeats)))

    results = Parallel(n_jobs=max_jobs, verbose=10)(
        delayed(evaluate_factorial_condition)(
            model, X, hypotheses, snr, repeat, n_permutations, 0.05, similarity_metric
        )
        for snr, repeat in all_tasks
    )

    # Flatten results (each task returns a list of dicts)
    flattened_results = [item for sublist in results for item in sublist]
    results_df = pd.DataFrame(flattened_results)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")
    return results_df


def run_spose_experiment(
    object_counts: list[int],
    num_dims: int,
    snrs: list[float],
    n_repeats: int,
    n_permutations: int,
    max_jobs: int,
    output_path: str,
):
    """Run RSA comparison experiment with SPOSE embeddings."""
    full_data = load_spose_embedding(num_dims=49)
    rng = np.random.default_rng(0)
    dims = rng.choice(full_data.shape[1], size=num_dims, replace=False)

    # Pre-select objects for each object count
    selected_objects_dict = {}
    for n_objects in object_counts:
        selected_objects_dict[n_objects] = rng.choice(
            full_data.shape[0], size=n_objects, replace=False
        )

    all_tasks = [
        (n_objects, snr, repeat)
        for n_objects in object_counts
        for snr in snrs
        for repeat in range(n_repeats)
    ]

    print(f"Running {len(all_tasks)} conditions in parallel")
    results = Parallel(n_jobs=max_jobs, verbose=10)(
        delayed(process_single_condition)(
            full_data,
            dims,
            selected_objects_dict[n_objects],
            snr,
            repeat,
            n_permutations,
        )
        for n_objects, snr, repeat in all_tasks
    )

    results_df = pd.DataFrame([item for sublist in results for item in sublist])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")
    return results_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--mode",
        type=str,
        choices=["spose", "factorial"],
        default="spose",
        help="Experiment mode: spose (SPOSE embeddings) or factorial (factorial design)",
    )
    parser.add_argument("--n-repeats", type=int, default=100)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--max-jobs", type=int, default=140)

    # SPOSE mode arguments
    parser.add_argument("--object-counts", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--num-dims", type=int, default=10)

    # Factorial mode arguments
    parser.add_argument(
        "--similarity-metric",
        type=str,
        default="linear",
        choices=["linear", "cosine", "euclidean"],
    )

    # Common arguments
    parser.add_argument(
        "--snrs",
        nargs="+",
        type=float,
        default=None,
        help="SNR values to test (defaults depend on mode)",
    )

    args = parser.parse_args()

    # Set defaults based on mode
    if args.snrs is None:
        args.snrs = (
            np.linspace(0.0, 1.0, 10)
            if args.mode == "spose"
            else np.linspace(0.0, 0.6, 10)
        )

    # Map mode names to output file names
    output_names = {
        "spose": "spose.csv",
        "factorial": "factorial.csv",
    }
    output_path = f"outputs/{output_names[args.mode]}"

    if args.mode == "factorial":
        levels = {
            "animacy": ["animate", "inanimate"],
            "size": ["small", "medium", "large"],
            "curvature": ["straight", "curved"],
            "color": ["red", "blue", "green", "yellow"],
        }
        run_factorial_experiment(
            args.snrs,
            args.n_repeats,
            args.n_permutations,
            args.max_jobs,
            output_path,
            levels,
            args.similarity_metric,
        )
    else:
        run_spose_experiment(
            args.object_counts,
            args.num_dims,
            args.snrs,
            args.n_repeats,
            args.n_permutations,
            args.max_jobs,
            output_path,
        )


if __name__ == "__main__":
    main()
