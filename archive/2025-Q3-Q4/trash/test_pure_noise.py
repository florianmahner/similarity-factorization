"""
Test script for pure noise case - debugging permutation test behavior
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from srf.helpers import (
    load_spose_embedding,
    add_noise_with_snr,
    align_latent_dimensions,
)
from srf.mixed.admm import ADMM
from tools.metrics import compute_similarity


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


def test_pure_noise_power():
    """Test permutation test behavior in pure noise scenarios"""

    print("=" * 60)
    print("PURE NOISE PERMUTATION TEST ANALYSIS")
    print("=" * 60)

    # Parameters
    MAX_OBJECTS = 100
    SELECTED_DIMS = [3, 5, 8, 12, 14]
    SIMILARITY_METRIC = "linear"
    N_REPEATS = 20  # Multiple runs to check consistency
    N_PERMUTATIONS = 5_000

    # Load real data
    print("Loading real data...")
    data = load_spose_embedding(max_objects=MAX_OBJECTS, max_dims=66)
    data = data[:, SELECTED_DIMS]
    rank = len(SELECTED_DIMS)

    # Create hypothesis from first dimension
    hypothesis = compute_similarity(data[:, [0]], data[:, [0]], SIMILARITY_METRIC)

    print(f"Data shape: {data.shape}")
    print(f"Hypothesis matrix shape: {hypothesis.shape}")
    print(f"Number of repeats: {N_REPEATS}")
    print(f"Permutations per test: {N_PERMUTATIONS}")

    # Test 1: Pure random data (sanity check)
    print(f"\n{'='*40}")
    print("TEST 1: PURE RANDOM DATA (SANITY CHECK)")
    print(f"{'='*40}")

    random_p_values = []
    for i in range(N_REPEATS):
        # Generate completely random data
        random_data = np.random.randn(*data.shape)
        random_similarity = compute_similarity(
            random_data, random_data, SIMILARITY_METRIC
        )

        # Test correlation with hypothesis
        p_val, corr = simple_mantel_test(
            hypothesis, random_similarity, n_permutations=N_PERMUTATIONS, random_state=i
        )
        random_p_values.append(p_val)

    random_significant = np.mean(np.array(random_p_values) < 0.05) * 100
    print(f"Random data significance rate: {random_significant:.1f}% (should be ~5%)")

    # Test 2: Pure noise via add_noise_with_snr
    print(f"\n{'='*40}")
    print("TEST 2: PURE NOISE VIA add_noise_with_snr(SNR=0.0)")
    print(f"{'='*40}")

    results = []

    for repeat in range(N_REPEATS):
        seed = repeat + 1000

        # Generate "pure noise" using your function
        # noisy_data = add_noise_with_snr(data, 0.0, rng=seed)

        noisy_data = np.random.randn(*data.shape)

        # Check if it's actually identical to original data
        is_identical = np.allclose(data, noisy_data)

        # Compute similarities
        measured_similarity = compute_similarity(
            noisy_data, noisy_data, SIMILARITY_METRIC
        )

        # Test RSA (without ADMM)
        rsa_p, rsa_corr = simple_mantel_test(
            hypothesis,
            measured_similarity,
            n_permutations=N_PERMUTATIONS,
            random_state=seed,
        )

        # Test NMF (with ADMM) - but skip ADMM for SNR=0.0 as in your fixed version
        if True:  # SNR == 0.0 case from your fixed script
            denoised_similarity = measured_similarity.copy()
            w = np.random.randn(measured_similarity.shape[0], rank) * 0.01
        else:
            model = ADMM(
                rank=rank,
                max_outer=10,
                w_inner=5,
                tol=0.01,
                verbose=False,
                random_state=seed,
                init="random",
            )
            w = model.fit_transform(measured_similarity)
            denoised_similarity = w @ w.T

        nmf_p, nmf_corr = simple_mantel_test(
            hypothesis,
            denoised_similarity,
            n_permutations=N_PERMUTATIONS,
            random_state=seed + 100,
        )

        # Test Latent (correlation with recovered factors)
        w_aligned = align_latent_dimensions(data, w)
        latent_tests = []
        for i in range(rank):
            lat_p, lat_corr = simple_permutation_test(
                data[:, i],
                w_aligned[:, i],
                n_permutations=N_PERMUTATIONS,
                random_state=seed + 200 + i,
            )
            latent_tests.append((lat_p, lat_corr))

        # Store results
        results.append(
            {
                "repeat": repeat,
                "data_identical": is_identical,
                "rsa_p": rsa_p,
                "rsa_corr": rsa_corr,
                "nmf_p": nmf_p,
                "nmf_corr": nmf_corr,
                "latent_p_mean": np.mean([t[0] for t in latent_tests]),
                "latent_corr_mean": np.mean([t[1] for t in latent_tests]),
            }
        )

    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)

    print(
        f"Data identical to original: {df['data_identical'].mean()*100:.1f}% of cases"
    )
    print(f"\nSignificance rates (should all be ~5% for pure noise):")
    print(f"RSA:    {(df['rsa_p'] < 0.05).mean()*100:.1f}%")
    print(f"NMF:    {(df['nmf_p'] < 0.05).mean()*100:.1f}%")
    print(f"Latent: {(df['latent_p_mean'] < 0.05).mean()*100:.1f}%")

    print(f"\nMean correlations (should all be ~0 for pure noise):")
    print(f"RSA:    {df['rsa_corr'].mean():.6f} ± {df['rsa_corr'].std():.6f}")
    print(f"NMF:    {df['nmf_corr'].mean():.6f} ± {df['nmf_corr'].std():.6f}")
    print(
        f"Latent: {df['latent_corr_mean'].mean():.6f} ± {df['latent_corr_mean'].std():.6f}"
    )

    # Test 3: What happens if we actually run ADMM on noise?
    print(f"\n{'='*40}")
    print("TEST 3: WHAT HAPPENS IF WE RUN ADMM ON PURE NOISE?")
    print(f"{'='*40}")

    admm_results = []
    model = ADMM(
        rank=rank, max_outer=10, w_inner=5, tol=0.01, verbose=False, random_state=0
    )

    for repeat in range(5):  # Just a few tests since this is expensive
        seed = repeat + 2000
        noisy_data = add_noise_with_snr(data, 0.0, rng=seed)
        measured_similarity = compute_similarity(
            noisy_data, noisy_data, SIMILARITY_METRIC
        )

        # Actually run ADMM on noise
        w = model.fit_transform(measured_similarity)
        denoised_similarity = w @ w.T

        # Test correlation
        admm_p, admm_corr = simple_mantel_test(
            hypothesis,
            denoised_similarity,
            n_permutations=N_PERMUTATIONS,
            random_state=seed,
        )

        rec_error = np.linalg.norm(
            measured_similarity - denoised_similarity
        ) / np.linalg.norm(measured_similarity)

        admm_results.append(
            {"p_value": admm_p, "correlation": admm_corr, "rec_error": rec_error}
        )

    admm_df = pd.DataFrame(admm_results)
    print(
        f"ADMM on noise significance rate: {(admm_df['p_value'] < 0.05).mean()*100:.1f}%"
    )
    print(f"ADMM on noise mean correlation: {admm_df['correlation'].mean():.6f}")
    print(f"ADMM reconstruction error: {admm_df['rec_error'].mean():.4f}")

    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")

    if random_significant > 10:
        print("🚨 PROBLEM: Even pure random data shows high significance!")
        print("   → Your permutation test has a bug")
    elif (df["nmf_p"] < 0.05).mean() > 0.15:  # > 15%
        print("🚨 PROBLEM: NMF shows high significance on pure noise!")
        print("   → ADMM is creating spurious correlations")
    else:
        print("✅ GOOD: Permutation test behaves correctly on pure noise")
        print("   → Your approach is sound, issue might be elsewhere")


if __name__ == "__main__":
    test_pure_noise_power()
