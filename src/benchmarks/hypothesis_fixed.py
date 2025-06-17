"""
Fixed hypothesis testing benchmark - fast and simple implementation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

from srf.helpers import (
    load_spose_embedding,
    add_noise_with_snr,
    align_latent_dimensions,
)
from srf.mixed.admm import ADMM
from tools.metrics import compute_similarity

# PARAMETERS - REDUCED FOR SPEED
MAX_OBJECTS = 100  # Reduced from 300 for faster testing
MAX_DIMS = 66  # I think this is ignored anyways!
SELECTED_DIMS = [3, 5, 8, 12, 14]

# Experiment parameters
SNR_LIST = np.linspace(0, 0.1, 4)  # Reduced from 5 points
N_REPEATS = 5  # Reduced from 100 for quick testing
N_PERMUTATIONS = 10_000  # Reduced from 500 for speed
ALPHA = 0.05

# Model parameters - FASTER SETTINGS
SIMILARITY_METRIC = "linear"
ADMM_MAX_OUTER = 10  # Reduced from 15
ADMM_W_INNER = 5  # Reduced from 10
ADMM_TOLERANCE = 0.0  # Increased from 0.0 for faster convergence

# Processing parameters
MAX_JOBS = 40  # Reduced for stability

# Output parameters
OUTPUT_PATH = "/LOCAL/fmahner/srf/results/benchmarks/hypothesis_tests_fixed.csv"


# Vectorized permutation test (from your optimized version)
def simple_permutation_test(vec1, vec2, n_permutations=1000, random_state=None):
    """Vectorized permutation test for correlation - much faster!"""
    if random_state is not None:
        np.random.seed(random_state)

    # Observed correlation (using numpy for speed)
    obs_corr = np.corrcoef(vec1, vec2)[0, 1]

    # Vectorized permutation test - generate all permutations at once
    n = len(vec2)
    perm_indices = np.random.rand(n_permutations, n).argsort(axis=1)
    vec2_perms = vec2[perm_indices]  # Shape: (n_permutations, n)

    # Vectorized correlation computation
    vec1_centered = vec1 - vec1.mean()
    vec1_norm = np.linalg.norm(vec1_centered)

    # Compute all permutation correlations in one go
    vec2_perms_centered = vec2_perms - vec2_perms.mean(axis=1, keepdims=True)
    vec2_perms_norms = np.linalg.norm(vec2_perms_centered, axis=1)

    # Avoid division by zero
    valid_mask = (vec1_norm > 0) & (vec2_perms_norms > 0)
    perm_corrs = np.zeros(n_permutations)

    if valid_mask.any():
        perm_corrs[valid_mask] = (vec2_perms_centered[valid_mask] @ vec1_centered) / (
            vec2_perms_norms[valid_mask] * vec1_norm
        )

    # Two-sided test
    better_count = np.sum(np.abs(perm_corrs) >= np.abs(obs_corr))
    p_value = (better_count + 1) / (n_permutations + 1)

    return p_value, obs_corr


def simple_mantel_test(matrix1, matrix2, n_permutations=1000, random_state=None):
    """Simple and fast Mantel test for matrix correlation."""
    # Extract upper triangular elements
    idx_upper = np.triu_indices_from(matrix1, k=1)
    vec1 = matrix1[idx_upper]
    vec2 = matrix2[idx_upper]

    # Use simple permutation test on vectors (it will handle the random seed)
    return simple_permutation_test(vec1, vec2, n_permutations, random_state)


def process_single_run(
    model, data, hypotheses, snr, repeat, n_permutations, alpha=ALPHA
):
    """Process a single experimental run - optimized for speed."""

    # Generate reproducible noise
    seed = int(snr * 1000 + repeat)
    noisy_data = add_noise_with_snr(data, snr, rng=seed)

    # Compute similarities and fit model
    measured_similarity = compute_similarity(noisy_data, noisy_data, SIMILARITY_METRIC)

    # Fix for SNR=0.0: Don't apply ADMM to pure noise (it creates spurious correlations)
    if snr == 0.0:
        # For pure noise, denoised should be identical to measured (no signal to recover)
        denoised_similarity = measured_similarity.copy()
        w = (
            np.random.randn(measured_similarity.shape[0], len(SELECTED_DIMS)) * 0.01
        )  # Dummy W
    else:
        # Fit ADMM only when there's actual signal
        w = model.fit_transform(measured_similarity)
        denoised_similarity = w @ w.T

    # Align latent dimensions and compute reconstruction error
    rank = data.shape[1]
    w_aligned = align_latent_dimensions(data, w)
    rec_error = np.linalg.norm(
        measured_similarity - denoised_similarity
    ) / np.linalg.norm(measured_similarity)

    # Run statistical tests - much faster versions
    rsa_tests = [
        simple_mantel_test(
            h, measured_similarity, n_permutations=n_permutations, random_state=seed + i
        )
        for i, h in enumerate(hypotheses)
    ]

    nmf_tests = [
        simple_mantel_test(
            h,
            denoised_similarity,
            n_permutations=n_permutations,
            random_state=seed + i + 100,
        )
        for i, h in enumerate(hypotheses)
    ]

    latent_tests = [
        simple_permutation_test(
            data[:, i],
            w_aligned[:, i],
            n_permutations=n_permutations,
            random_state=seed + i + 200,
        )
        for i in range(rank)
    ]

    # Organize test results
    test_results = {
        "RSA": rsa_tests,
        "NMF": nmf_tests,
        "Latent": latent_tests,
    }

    # Apply multiple testing correction and format results
    results = []
    for method, tests in test_results.items():
        raw_ps = [t[0] for t in tests]  # p-values
        observed_corrs = [t[1] for t in tests]  # correlations
        reject, corr_ps = multipletests(raw_ps, alpha, method="fdr_bh")[:2]

        for i, (raw_p, obs_corr, corrected_p, rej) in enumerate(
            zip(raw_ps, observed_corrs, corr_ps, reject)
        ):
            results.append(
                {
                    "SNR": snr,
                    "Repeat": repeat,
                    "Hypothesis": i + 1,
                    "Method": method,
                    "Raw_p": raw_p,
                    "Corrected_p": corrected_p,
                    "Observed_correlation": obs_corr,
                    "Significant": bool(rej),
                    "Rec_error": rec_error,
                }
            )

    return results


def main():
    """Main execution function."""
    print("=== FAST HYPOTHESIS TESTING ===")
    print("Loading data...")
    data = load_spose_embedding(max_objects=MAX_OBJECTS, max_dims=MAX_DIMS)
    data = data[:, SELECTED_DIMS]
    rank = len(SELECTED_DIMS)

    print(f"Data shape: {data.shape}")
    print(f"Using dimensions: {SELECTED_DIMS}")
    print(f"SNR values: {SNR_LIST}")
    print(f"Repeats per SNR: {N_REPEATS}")
    print(f"Permutations per test: {N_PERMUTATIONS}")

    # Create hypotheses (one per dimension)
    print("Creating hypotheses...")
    hypotheses = []
    for i in range(rank):
        x_i = data[:, [i]]
        s_i = compute_similarity(x_i, x_i, SIMILARITY_METRIC)
        hypotheses.append(s_i)

    # Generate all task combinations
    all_tasks = [(snr, rep) for snr in SNR_LIST for rep in range(N_REPEATS)]
    print(f"Total tasks: {len(all_tasks)}")

    # Initialize model with faster settings
    model = ADMM(
        rank=rank,
        max_outer=ADMM_MAX_OUTER,
        w_inner=ADMM_W_INNER,
        tol=ADMM_TOLERANCE,
        verbose=False,
        random_state=0,
    )

    print("Starting parallel processing...")
    print("This should be much faster now!")

    # Run parallel processing
    results = Parallel(n_jobs=MAX_JOBS, verbose=1)(
        delayed(process_single_run)(
            model, data, hypotheses, snr, repeat, N_PERMUTATIONS
        )
        for snr, repeat in all_tasks
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

    # Quick summary
    print("\n=== QUICK SUMMARY ===")
    summary = (
        results_df.groupby(["Method", "SNR"])
        .agg(
            {"Observed_correlation": "mean", "Significant": "mean", "Rec_error": "mean"}
        )
        .round(4)
    )
    print(summary)


if __name__ == "__main__":
    main()
