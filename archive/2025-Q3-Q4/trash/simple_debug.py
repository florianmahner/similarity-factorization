import numpy as np
from scipy.stats import pearsonr


def simple_permutation_test(vec1, vec2, n_permutations=1000, random_state=None):
    """Simple permutation test for correlation"""
    if random_state is not None:
        np.random.seed(random_state)

    # Observed correlation
    obs_corr = np.corrcoef(vec1, vec2)[0, 1]

    # Generate permutation correlations
    n = len(vec2)
    perm_indices = np.random.rand(n_permutations, n).argsort(axis=1)
    vec2_perms = vec2[perm_indices]

    # Vectorized correlation computation
    vec1_centered = vec1 - vec1.mean()
    vec1_norm = np.linalg.norm(vec1_centered)

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
    """Extract upper triangular and run permutation test"""
    idx_upper = np.triu_indices_from(matrix1, k=1)
    vec1 = matrix1[idx_upper]
    vec2 = matrix2[idx_upper]
    return simple_permutation_test(vec1, vec2, n_permutations, random_state)


print("DEBUGGING PERMUTATION TEST")
print("=" * 50)

# Test 1: Pure random vectors (should be ~5%)
print("\nTest 1: Random vectors")
n_tests = 100
p_values = []
for i in range(n_tests):
    vec1 = np.random.randn(50)
    vec2 = np.random.randn(50)
    p_val, _ = simple_permutation_test(vec1, vec2, n_permutations=1000, random_state=i)
    p_values.append(p_val)

print(
    f"Significance rate: {np.mean(np.array(p_values) < 0.05)*100:.1f}% (should be ~5%)"
)

# Test 2: Random matrices (should be ~5%)
print("\nTest 2: Random matrices")
p_values = []
for i in range(n_tests):
    mat1 = np.random.randn(20, 20)
    mat2 = np.random.randn(20, 20)
    p_val, _ = simple_mantel_test(mat1, mat2, n_permutations=1000, random_state=i)
    p_values.append(p_val)

print(
    f"Significance rate: {np.mean(np.array(p_values) < 0.05)*100:.1f}% (should be ~5%)"
)

# Test 3: What happens with STRUCTURED vs RANDOM?
print("\nTest 3: Structured matrix vs random")
# Create a structured matrix (like your hypothesis)
x = np.random.randn(20, 1)  # Some data
structured_matrix = x @ x.T  # Similarity matrix (structured!)

p_values = []
for i in range(n_tests):
    random_matrix = np.random.randn(20, 20)  # Random matrix
    p_val, _ = simple_mantel_test(
        structured_matrix, random_matrix, n_permutations=1000, random_state=i
    )
    p_values.append(p_val)

print(
    f"Significance rate: {np.mean(np.array(p_values) < 0.05)*100:.1f}% (should be ~5%)"
)

# Test 4: What if BOTH are structured but independent?
print("\nTest 4: Two independent structured matrices")
p_values = []
for i in range(n_tests):
    x1 = np.random.randn(20, 1)
    x2 = np.random.randn(20, 1)
    mat1 = x1 @ x1.T  # Structured
    mat2 = x2 @ x2.T  # Also structured but independent
    p_val, _ = simple_mantel_test(mat1, mat2, n_permutations=1000, random_state=i)
    p_values.append(p_val)

print(
    f"Significance rate: {np.mean(np.array(p_values) < 0.05)*100:.1f}% (should be ~5%)"
)

print("\n" + "=" * 50)
print("CONCLUSION: If Test 3 or 4 show high significance rates,")
print("the problem is using STRUCTURED matrices in your tests!")
