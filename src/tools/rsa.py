#!/usr/bin/env python3

import numpy as np

from numba import njit, prange
from .metrics import compute_similarity, compute_distance
from .stats import compute_correlation_coeff
from scipy.stats import pearsonr

Array = np.ndarray


def compute_rdm(x: Array, metric: str = "pearson") -> Array:
    return compute_distance(x, x, metric)


def compute_rsm(x: Array, metric: str = "pearson") -> Array:
    return compute_similarity(x, x, metric)


def correlate_rsms(
    x: Array, y: Array, corr_type: str = "pearson", return_pval: bool = False
) -> float | tuple[float, float]:
    """Correlate the upper triangular parts of two rsms."""
    if corr_type not in ["pearson", "spearman"]:
        raise ValueError("Correlation must be 'pearson' or 'spearman'")

    x = x.copy()
    y = y.copy()
    np.fill_diagonal(x, 1)
    np.fill_diagonal(y, 1)
    triu_inds = np.triu_indices(len(x), k=1)
    x_triu = x[triu_inds]
    y_triu = y[triu_inds]
    corr, p = compute_correlation_coeff(x_triu, y_triu, corr_type)

    return (corr, p) if return_pval else corr


@njit(parallel=True, fastmath=True)
def matmul(x: Array, y: Array) -> Array:
    n_i, n_k = x.shape
    _, n_j = y.shape
    f = np.zeros((n_i, n_j))
    for i in prange(n_i):
        for j in prange(n_j):
            for k in prange(n_k):
                f[i, j] += x[i, k] * y[k, j]
    return f


@njit(parallel=True, fastmath=True)
def reconstruct_rsm(w: Array) -> Array:
    n = len(w)
    s = matmul(w, w.T)
    s_e = np.exp(s)
    rsm = np.zeros((n, n))
    for i in prange(n):
        for j in prange(i + 1, n):
            for k in prange(n):
                if k != i and k != j:
                    rsm[i, j] += s_e[i, j] / (s_e[i, j] + s_e[i, k] + s_e[j, k])

    rsm /= n - 2
    rsm += rsm.T
    np.fill_diagonal(rsm, 1)
    return rsm


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
