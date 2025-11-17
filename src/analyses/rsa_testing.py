"""RSA testing utilities for hypothesis testing and comparison."""

import numpy as np
from scipy.stats import pearsonr

ndarray = np.ndarray


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

    p_value = (np.sum(nulls >= obs) + 1) / (permutations + 1)
    return p_value, nulls, obs


def permutation_test(A, B, permutations=10000, random_state=None, two_sided=True):
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

    p_value = (greater + 1) / (permutations + 1)
    return p_value, null_corrs, observed_corr
