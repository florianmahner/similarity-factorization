"""
Hypothesis testing benchmark for comparing RSA, NMF, and Latent space methods
across different signal-to-noise ratios and object counts.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

from utils.helpers import (
    add_noise_with_snr,
    align_latent_dimensions,
)
from utils.io import load_spose_embedding
from models.admm import ADMM
from tools.rsa import compute_similarity

# PARAMETERS

# Data parameters
OBJECT_COUNTS = [25, 50, 100, 200]  # Different object counts to test
NUM_DIMS = 10
SELECTED_DIMS = [3, 5, 8, 12, 14]  # Subsample of the SPOSE dimensions

# Experiment parameters
SNR_LIST = np.linspace(0, 0.6, 11)

N_REPEATS = 100  # Reduced from 500 since we're adding another factor
N_PERMUTATIONS = 1_000
ALPHA = 0.05

# Model parameters
SIMILARITY_METRIC = "linear"

# Processing parameters
MAX_JOBS = 140

# Output parameters
OUTPUT_PATH = "/LOCAL/fmahner/similarity-factorization/results/benchmarks/hypothesis_tests_objects.csv"

# STATISTICAL TESTS


def mantel_test(A, B, permutations=10000, random_state=None, two_sided=False):
    """
    Mantel test for correlation between two distance/similarity matrices.

    Args:
        A, B: Square matrices to compare
        permutations: Number of permutations for null distribution
        random_state: Random seed for reproducibility
        two_sided: Whether to use two-sided test

    Returns:
        p_value, null_distribution, observed_statistic
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Extract upper triangular elements
    idx_upper = np.triu_indices_from(A, k=1)
    sim1 = A[idx_upper]
    sim2 = B[idx_upper]

    # Observed correlation
    obs = pearsonr(sim1, sim2).statistic
    if two_sided:
        obs = abs(obs)

    # Generate null distribution
    nulls = np.zeros(permutations)
    for i in range(permutations):
        perm = np.random.permutation(B.shape[0])
        Bp = B[perm][:, perm]
        sc = pearsonr(sim1, Bp[idx_upper]).statistic
        if two_sided:
            sc = abs(sc)
        nulls[i] = sc

    # Calculate p-value
    p = (np.sum(nulls >= obs) + 1) / (permutations + 1)
    return p, nulls, obs


def permutation_test(A, B, permutations=10000, random_state=None, two_sided=False):
    """
    Permutation test for correlation between two vectors.

    Args:
        A, B: Vectors to compare
        permutations: Number of permutations for null distribution
        random_state: Random seed for reproducibility
        two_sided: Whether to use two-sided test

    Returns:
        p_value, null_distribution, observed_correlation
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Observed correlation
    observed_corr = pearsonr(A, B).statistic
    if two_sided:
        observed_corr = np.abs(observed_corr)

    # Generate null distribution
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

    # Calculate p-value with correction to avoid p=0
    p_value = (greater + 1) / (permutations + 1)
    return p_value, null_corrs, observed_corr


# EXPERIMENT PROCESSING


def process_single_run(
    model, data, hypotheses, n_objects, snr, repeat, n_permutations, alpha=ALPHA
):
    """
    Process a single experimental run with given object count, SNR and repeat number.

    Args:
        model: ADMM model instance
        data: Original clean data
        hypotheses: List of hypothesis matrices
        n_objects: Number of objects in this run
        snr: Signal-to-noise ratio
        repeat: Repeat number for this condition
        n_permutations: Number of permutations for statistical tests
        alpha: Significance level for multiple testing correction

    Returns:
        List of result dictionaries
    """
    # Generate reproducible noise
    seed = int(n_objects * 1000 + snr * 100 + repeat)
    noisy_data = add_noise_with_snr(data, snr, rng=seed)

    # Compute similarities and fit model
    measured_similarity = compute_similarity(noisy_data, noisy_data, SIMILARITY_METRIC)
    w = model.fit_transform(measured_similarity)
    denoised_similarity = w @ w.T

    # Align latent dimensions and compute reconstruction error
    rank = data.shape[1]
    w_aligned = align_latent_dimensions(data, w)
    rec_error = np.linalg.norm(
        measured_similarity - denoised_similarity
    ) / np.linalg.norm(measured_similarity)

    # Run statistical tests
    rsa_tests = [
        mantel_test(h, measured_similarity, permutations=n_permutations)
        for h in hypotheses
    ]
    nmf_tests = [
        mantel_test(h, denoised_similarity, permutations=n_permutations)
        for h in hypotheses
    ]
    latent_tests = [
        permutation_test(data[:, i], w_aligned[:, i], permutations=n_permutations)
        for i in range(rank)
    ]

    # Organize test results
    test_results = {
        "RSA": rsa_tests,
        "NMF Denoised": nmf_tests,
        "NMF": latent_tests,
    }

    # Apply multiple testing correction and format results
    results = []
    for method, tests in test_results.items():
        raw_ps = [t[0] for t in tests]
        reject, corr_ps = multipletests(raw_ps, alpha, method="fdr_bh")[:2]

        for i, (test, corrected_p, rej) in enumerate(zip(tests, corr_ps, reject)):
            results.append(
                {
                    "N_objects": n_objects,
                    "SNR": snr,
                    "Repeat": repeat,
                    "Hypothesis": i + 1,
                    "Method": method,
                    "Raw_p": test[0],
                    "Corrected_p": corrected_p,
                    "Significant": bool(rej),
                    "Rec_error": rec_error,
                }
            )

    return results


def main():
    """Main execution function."""
    print("Loading full SPOSE embedding...")
    full_data = load_spose_embedding(max_objects=None, num_dims=49)  # Load all objects

    # Select dimensions once for consistency across object counts
    np.random.seed(42)  # Fix seed for reproducible dimension selection
    dims = np.random.choice(full_data.shape[1], size=NUM_DIMS, replace=False)

    print(f"Full data shape: {full_data.shape}")
    print(f"Selected dimensions: {dims}")

    # Generate all task combinations
    all_tasks = []
    for n_objects in OBJECT_COUNTS:
        for snr in SNR_LIST:
            for repeat in range(N_REPEATS):
                all_tasks.append((n_objects, snr, repeat))

    print(
        f"Total tasks: {len(all_tasks)} = {len(OBJECT_COUNTS)} objects x {len(SNR_LIST)} SNRs x {N_REPEATS} repeats"
    )

    # Run parallel processing
    print("Starting parallel processing...")
    results = Parallel(n_jobs=MAX_JOBS, verbose=10)(
        delayed(process_single_condition)(
            full_data, dims, n_objects, snr, repeat, N_PERMUTATIONS
        )
        for n_objects, snr, repeat in all_tasks
    )

    # Combine and save results
    print("Combining results...")
    results_df = pd.DataFrame([item for sublist in results for item in sublist])

    # Ensure output directory exists
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    print(f"Final results shape: {results_df.shape}")


def process_single_condition(full_data, dims, n_objects, snr, repeat, n_permutations):
    """Process a single experimental condition."""
    # Create reproducible object selection for this condition
    seed = int(n_objects * 10000 + snr * 100 + repeat)
    np.random.seed(seed)

    # Select subset of objects and dimensions
    selected_objects = np.random.choice(
        full_data.shape[0], size=n_objects, replace=False
    )
    data = full_data[selected_objects][:, dims]

    # Create hypotheses for this object count
    hypotheses = []
    rank = len(dims)
    for i in range(rank):
        x_i = data[:, [i]]
        s_i = compute_similarity(x_i, x_i, SIMILARITY_METRIC)
        hypotheses.append(s_i)

    # Initialize model for this rank
    model = ADMM(rank=rank, verbose=False, tol=0.0)

    # Process the run
    return process_single_run(
        model, data, hypotheses, n_objects, snr, repeat, n_permutations
    )


if __name__ == "__main__":
    main()
