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


# =============================================================================
# Restricted Mantel Test (for factorial designs)
# =============================================================================


def _get_stratified_permutation(
    n: int, strata: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Get permutation indices that only shuffle within strata groups.

    Parameters
    ----------
    n : int
        Number of items
    strata : np.ndarray
        Array of shape (n, n_other_factors) defining strata membership.
        Items with identical rows are in the same stratum.
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    np.ndarray
        Permutation indices of length n
    """
    idx = np.arange(n)
    unique_strata = np.unique(strata, axis=0)
    for group in unique_strata:
        mask = (strata == group).all(axis=1)
        group_indices = np.where(mask)[0]
        idx[group_indices] = rng.permutation(group_indices)
    return idx


def mantel_test_restricted(
    model_rsm: np.ndarray,
    data_rsm: np.ndarray,
    strata: np.ndarray,
    permutations: int = 1000,
    random_state: int | None = None,
) -> tuple[float, float, np.ndarray]:
    """Mantel test with restricted permutation for factorial designs.

    In factorial designs, when testing one factor's effect, we must control
    for other factors. This is done by only permuting items within strata
    defined by the levels of the other factors.

    Parameters
    ----------
    model_rsm : np.ndarray
        Hypothesis RSM for the factor being tested (n x n)
    data_rsm : np.ndarray
        Observed/measured RSM (n x n)
    strata : np.ndarray
        Strata membership array (n x n_other_factors). Items with the same
        row values are in the same stratum and can be permuted together.
    permutations : int
        Number of permutations for null distribution
    random_state : int | None
        Random seed for reproducibility

    Returns
    -------
    p_value : float
        One-sided p-value (proportion of null >= observed)
    r_obs : float
        Observed correlation
    r_null : np.ndarray
        Null distribution of correlations
    """
    rng = np.random.default_rng(random_state)
    n = model_rsm.shape[0]

    idx_upper = np.triu_indices(n, k=1)
    data_flat = data_rsm[idx_upper]
    model_flat = model_rsm[idx_upper]

    r_obs = pearsonr(data_flat, model_flat).statistic

    r_null = np.zeros(permutations)
    for i in range(permutations):
        perm = _get_stratified_permutation(n, strata, rng)
        model_perm = model_rsm[perm][:, perm]
        model_perm_flat = model_perm[idx_upper]
        r_null[i] = pearsonr(data_flat, model_perm_flat).statistic

    p_value = (np.sum(r_null >= r_obs) + 1) / (permutations + 1)
    return p_value, r_obs, r_null


# =============================================================================
# LOO Column Alignment Test (for SRF embeddings)
# =============================================================================


def _find_column_permutation(
    X: np.ndarray, W: np.ndarray
) -> np.ndarray:
    """Find optimal column permutation to align W to X via Hungarian algorithm.

    Parameters
    ----------
    X : np.ndarray
        Target matrix (n x k), e.g., one-hot factorial design
    W : np.ndarray
        Source matrix (n x k), e.g., SRF embedding

    Returns
    -------
    np.ndarray
        Column indices such that W[:, perm] best aligns with X
    """
    from scipy.optimize import linear_sum_assignment

    k = X.shape[1]
    # Correlation matrix between X columns and W columns
    # Standardize for correlation
    X_std = (X - X.mean(0)) / (X.std(0) + 1e-9)
    W_std = (W - W.mean(0)) / (W.std(0) + 1e-9)
    corr = (X_std.T @ W_std) / X.shape[0]  # (k x k)

    # Hungarian algorithm to maximize absolute correlation
    _, col_perm = linear_sum_assignment(-np.abs(corr))
    return col_perm


def loo_alignment(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Leave-one-out column alignment of W to X.

    For each item i, finds the optimal column permutation using only
    the other N-1 items, then applies that permutation to item i.
    This prevents overfitting when evaluating alignment quality.

    Parameters
    ----------
    W : np.ndarray
        SRF embedding matrix (n x k)
    X : np.ndarray
        Target matrix (n x k), e.g., one-hot factorial design

    Returns
    -------
    np.ndarray
        W with columns permuted using LOO alignment (n x k)
    """
    n, k = W.shape
    W_aligned = np.zeros_like(W)

    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False

        col_perm = _find_column_permutation(X[train_mask], W[train_mask])
        W_aligned[i] = W[i, col_perm]

    return W_aligned


def loo_alignment_test(
    W: np.ndarray,
    X: np.ndarray,
    columns: slice | int | None = None,
    permutations: int = 1000,
    random_state: int | None = None,
) -> tuple[float, float, np.ndarray]:
    """LOO alignment test for SRF embedding recovery of structure.

    Tests whether the SRF embedding W recovers the structure in X
    using leave-one-out column alignment to prevent overfitting.

    Works for both:
    - Factorial designs: X is one-hot, columns=slice for factor groups
    - SPOSE-style: X is continuous, columns=int for single dimension

    Parameters
    ----------
    W : np.ndarray
        SRF embedding matrix (n x k)
    X : np.ndarray
        Target matrix (n x k). Can be one-hot (factorial) or continuous (SPOSE).
    columns : slice | int | None
        Which columns to test:
        - slice: test a group of columns (e.g., slice(0,3) for a factor)
        - int: test a single column/dimension (e.g., 0 for first dim)
        - None: test all columns
    permutations : int
        Number of permutations for null distribution
    random_state : int | None
        Random seed for reproducibility

    Returns
    -------
    p_value : float
        One-sided p-value
    r_obs : float
        Observed correlation between aligned W and X
    r_null : np.ndarray
        Null distribution
    """
    rng = np.random.default_rng(random_state)
    n = W.shape[0]

    W_aligned = loo_alignment(W, X)

    if columns is not None:
        if isinstance(columns, int):
            W_test = W_aligned[:, columns]
            X_test = X[:, columns]
        else:
            W_test = W_aligned[:, columns]
            X_test = X[:, columns]
    else:
        W_test = W_aligned
        X_test = X

    r_obs = pearsonr(W_test.ravel(), X_test.ravel()).statistic

    r_null = np.zeros(permutations)
    for i in range(permutations):
        perm = rng.permutation(n)
        X_perm = X_test[perm] if X_test.ndim > 1 else X_test[perm]
        r_null[i] = pearsonr(W_test.ravel(), X_perm.ravel()).statistic

    p_value = (np.sum(r_null >= r_obs) + 1) / (permutations + 1)
    return p_value, r_obs, r_null


def loo_alignment_test_multi(
    W: np.ndarray,
    X: np.ndarray,
    permutations: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> dict:
    """LOO alignment test for all dimensions with FDR correction.

    Tests each dimension independently using LOO alignment, then applies
    FDR (Benjamini-Hochberg) correction for multiple comparisons.

    Parameters
    ----------
    W : np.ndarray
        SRF embedding matrix (n x k)
    X : np.ndarray
        Target matrix (n x k), e.g., SPOSE dimensions or factorial design
    permutations : int
        Number of permutations per dimension
    alpha : float
        Significance level for FDR correction
    random_state : int | None
        Random seed for reproducibility

    Returns
    -------
    dict with keys:
        - r_obs: array of observed correlations per dimension
        - raw_p: array of raw p-values per dimension
        - corrected_p: array of FDR-corrected p-values
        - significant: boolean array of significant dimensions
        - W_aligned: LOO-aligned W matrix
    """
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(random_state)
    n, k = W.shape

    W_aligned = loo_alignment(W, X)

    raw_ps = []
    r_obs_all = []

    for dim in range(k):
        W_dim = W_aligned[:, dim]
        X_dim = X[:, dim]

        r_obs = pearsonr(W_dim, X_dim).statistic
        r_obs_all.append(r_obs)

        r_null = np.zeros(permutations)
        for i in range(permutations):
            perm = rng.permutation(n)
            r_null[i] = pearsonr(W_dim, X_dim[perm]).statistic

        p_value = (np.sum(r_null >= r_obs) + 1) / (permutations + 1)
        raw_ps.append(p_value)

    raw_ps = np.array(raw_ps)
    r_obs_all = np.array(r_obs_all)

    reject, corrected_ps, _, _ = multipletests(raw_ps, alpha=alpha, method="fdr_bh")

    return {
        "r_obs": r_obs_all,
        "raw_p": raw_ps,
        "corrected_p": corrected_ps,
        "significant": reject,
        "W_aligned": W_aligned,
    }
