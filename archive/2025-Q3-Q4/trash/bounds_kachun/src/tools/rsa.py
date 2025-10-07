#!/usr/bin/env python3

import numpy as np

from numba import njit, prange
from .metrics import compute_similarity, compute_distance
from .stats import compute_correlation_coeff


Array = np.ndarray


def compute_rdm(x: Array, metric: str = "pearson") -> Array:
    return compute_distance(x, x, metric)


def compute_rsm(x: Array, metric: str = "pearson") -> Array:
    return compute_similarity(x, x, metric)


def correlate_rsms(
    x: Array, y: Array, corr_type: str = "pearson", return_pval: bool = False
) -> float | tuple[float, float]:
    """Correlate the lower triangular parts of two rsms."""
    if corr_type not in ["pearson", "spearman"]:
        raise ValueError("Correlation must be 'pearson' or 'spearman'")

    np.fill_diagonal(x, 1)
    np.fill_diagonal(y, 1)
    triu_inds = np.triu_indices(len(x), k=1)
    x_triu = x[triu_inds]
    y_triu = y[triu_inds]
    corr, p = compute_correlation_coeff(x_triu, y_triu, corr_type)

    return (corr, p) if return_pval else corr


@njit(parallel=False, fastmath=False)
def matmul(x: Array, y: Array) -> Array:
    i, k = x.shape
    k, j = y.shape
    f = np.zeros((i, j))
    for i in prange(i):
        for j in prange(j):
            for k in prange(k):
                f[i, j] += x[i, k] * y[k, j]
    return f


@njit(parallel=False, fastmath=False)
def reconstruct_rsm(w: Array) -> Array:
    """Reconstructs a RSM from an embedding matrix W"""
    n = len(w)
    s = matmul(w, w.T)
    s_e = np.exp(s)  # exponentiate all elements in the inner product matrix s
    rsm = np.zeros((n, n))
    for i in prange(n):
        for j in prange(i + 1, n):
            for k in prange(n):
                if k != i and k != j:
                    rsm[i, j] += s_e[i, j] / (s_e[i, j] + s_e[i, k] + s_e[j, k])

    rsm /= n - 2
    rsm += rsm.T  # make similarity matrix symmetric
    np.fill_diagonal(rsm, 1)
    return rsm
