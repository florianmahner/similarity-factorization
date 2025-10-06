import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


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

    return p_value, obs_corr, perm_corrs


def test_permutation_test_correctness():
    """Test if permutation test has correct Type I error rate"""

    print("Testing permutation test correctness...")
    print("=" * 50)

    n_tests = 500  # Number of independent tests
    n_points = 50  # Points per test
    n_permutations = 1000
    alpha = 0.05

    p_values = []
    correlations = []

    for i in range(n_tests):
        # Generate independent random data (should have ~5% false positive rate)
        vec1 = np.random.randn(n_points)
        vec2 = np.random.randn(n_points)

        p_val, obs_corr, perm_corrs = simple_permutation_test(
            vec1, vec2, n_permutations=n_permutations, random_state=i
        )

        p_values.append(p_val)
        correlations.append(obs_corr)

        if i % 100 == 0:
            print(f"Test {i}: p={p_val:.4f}, r={obs_corr:.4f}")

    p_values = np.array(p_values)
    correlations = np.array(correlations)

    # Check Type I error rate
    false_positive_rate = np.mean(p_values < alpha)

    print(f"\nResults:")
    print(f"False positive rate: {false_positive_rate:.3f} (expected: {alpha:.3f})")
    print(f"Mean correlation: {correlations.mean():.6f} (expected: ~0)")
    print(f"Std correlation: {correlations.std():.6f}")
    print(f"P-value range: [{p_values.min():.4f}, {p_values.max():.4f}]")

    # Plot histogram of p-values (should be uniform under null)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(p_values, bins=20, alpha=0.7, edgecolor="black")
    plt.axhline(
        y=len(p_values) / 20, color="red", linestyle="--", label="Expected (uniform)"
    )
    plt.xlabel("P-value")
    plt.ylabel("Frequency")
    plt.title("P-value distribution\n(should be uniform under null)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(correlations, bins=30, alpha=0.7, edgecolor="black")
    plt.xlabel("Correlation")
    plt.ylabel("Frequency")
    plt.title("Correlation distribution\n(should be centered at 0)")
    plt.axvline(x=0, color="red", linestyle="--")

    plt.tight_layout()
    plt.savefig("permtest_debug.png", dpi=150, bbox_inches="tight")
    plt.show()

    return false_positive_rate


if __name__ == "__main__":
    false_pos_rate = test_permutation_test_correctness()

    if abs(false_pos_rate - 0.05) < 0.02:  # Within 2% is reasonable
        print(
            f"\n✅ PASS: False positive rate {false_pos_rate:.3f} is close to expected 0.05"
        )
    else:
        print(
            f"\n❌ FAIL: False positive rate {false_pos_rate:.3f} is too far from expected 0.05"
        )
        print("There is likely a bug in the permutation test implementation!")
