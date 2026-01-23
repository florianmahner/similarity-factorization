#!/usr/bin/env python3
"""RSA and alignment tests for similarity analysis."""

import numpy as np
from numba import njit, prange
from scipy.stats import pearsonr

from .metrics import compute_similarity, compute_distance
from .stats import compute_correlation_coeff

Array = np.ndarray


def _correlation(a: np.ndarray, b: np.ndarray, two_sided: bool = True) -> float:
    """Pearson correlation, absolute if two_sided."""
    r = pearsonr(a, b).statistic
    return np.abs(r) if two_sided else r


def _pvalue(obs: float, null: np.ndarray) -> float:
    """Permutation p-value."""
    return (np.sum(null >= obs) + 1) / (len(null) + 1)


def compute_rdm(x: Array, metric: str = "pearson") -> Array:
    return compute_distance(x, x, metric)


def compute_rsm(x: Array, metric: str = "pearson") -> Array:
    return compute_similarity(x, x, metric)


def correlate_rsms(
    x: Array, y: Array, corr_type: str = "pearson", return_pval: bool = False
) -> float | tuple[float, float]:
    """Correlate upper triangular parts of two RSMs."""
    if corr_type not in ["pearson", "spearman"]:
        raise ValueError("Correlation must be 'pearson' or 'spearman'")

    x, y = x.copy(), y.copy()
    np.fill_diagonal(x, 1)
    np.fill_diagonal(y, 1)
    idx = np.triu_indices(len(x), k=1)
    corr, p = compute_correlation_coeff(x[idx], y[idx], corr_type)
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
    """Reconstruct RSM from embedding using softmax normalization."""
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


def mantel_test(
    a: np.ndarray,
    b: np.ndarray,
    permutations: int = 1000,
    two_sided: bool = True,
    random_state: int | None = None,
) -> tuple[float, np.ndarray, float]:
    """Mantel test for RSM correlation."""
    rng = np.random.default_rng(random_state)
    idx = np.triu_indices_from(a, k=1)
    a_flat, b_flat = a[idx], b[idx]

    r_obs = _correlation(a_flat, b_flat, two_sided)

    null = np.zeros(permutations)
    for i in range(permutations):
        perm = rng.permutation(b.shape[0])
        null[i] = _correlation(a_flat, b[perm][:, perm][idx], two_sided)

    return _pvalue(r_obs, null), null, r_obs


def permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    permutations: int = 1000,
    two_sided: bool = True,
    random_state: int | None = None,
) -> tuple[float, np.ndarray, float]:
    """Permutation test for vector correlation."""
    rng = np.random.default_rng(random_state)
    r_obs = _correlation(a, b, two_sided)

    null = np.zeros(permutations)
    for i in range(permutations):
        null[i] = _correlation(a, rng.permutation(b), two_sided)

    return _pvalue(r_obs, null), null, r_obs


def _get_stratified_permutation(n: int, strata: np.ndarray, rng) -> np.ndarray:
    """Permute only within strata groups."""
    idx = np.arange(n)
    for group in np.unique(strata, axis=0):
        mask = (strata == group).all(axis=1)
        group_idx = np.where(mask)[0]
        idx[group_idx] = rng.permutation(group_idx)
    return idx


def mantel_test_restricted(
    model_rsm: np.ndarray,
    data_rsm: np.ndarray,
    strata: np.ndarray,
    permutations: int = 1000,
    random_state: int | None = None,
) -> tuple[float, float, np.ndarray]:
    """Mantel test with restricted permutation within strata."""
    rng = np.random.default_rng(random_state)
    n = model_rsm.shape[0]
    idx = np.triu_indices(n, k=1)

    r_obs = pearsonr(data_rsm[idx], model_rsm[idx]).statistic

    r_null = np.zeros(permutations)
    for i in range(permutations):
        perm = _get_stratified_permutation(n, strata, rng)
        r_null[i] = pearsonr(data_rsm[idx], model_rsm[perm][:, perm][idx]).statistic

    return _pvalue(r_obs, r_null), r_obs, r_null


def rsa_test(
    x: np.ndarray,
    s: np.ndarray,
    two_sided: bool = True,
    permutations: int = 1000,
    alpha: float = 0.05,
    fdr: bool = True,
    random_state: int | None = None,
) -> dict:
    """RSA test for each column of x against similarity matrix s.

    Returns dict with: r_obs, raw_p, corrected_p (if fdr), significant (if fdr)
    """
    k = x.shape[1]
    r_obs = np.zeros(k)
    raw_p = np.zeros(k)

    for i in range(k):
        h = x[:, [i]] @ x[:, [i]].T
        seed = random_state + i if random_state is not None else None
        raw_p[i], _, r_obs[i] = mantel_test(h, s, permutations=permutations, two_sided=two_sided, random_state=seed)

    result = {"r_obs": r_obs, "raw_p": raw_p}

    if fdr:
        from statsmodels.stats.multitest import multipletests
        reject, corrected_p, _, _ = multipletests(raw_p, alpha=alpha, method="fdr_bh")
        result["corrected_p"] = corrected_p
        result["significant"] = reject

    return result


def _find_column_permutation(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Hungarian algorithm to align w columns to x using signed correlation."""
    from scipy.optimize import linear_sum_assignment

    k = x.shape[1]
    # Vectorized correlation: corrcoef returns (2k x 2k), we want top-right block
    combined = np.corrcoef(x.T, w.T)
    corr = combined[:k, k:]
    _, col_perm = linear_sum_assignment(-corr)
    return col_perm


def global_alignment(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Align w to x using all data."""
    return w[:, _find_column_permutation(x, w)]


def loo_alignment(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Leave-one-out alignment (unbiased)."""
    n = w.shape[0]
    w_aligned = np.zeros_like(w)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        perm = _find_column_permutation(x[mask], w[mask])
        w_aligned[i] = w[i, perm]
    return w_aligned


def alignment_test(
    w: np.ndarray,
    x: np.ndarray,
    alignment: str = "global",
    two_sided: bool = True,
    permutations: int = 1000,
    alpha: float = 0.05,
    fdr: bool = True,
    random_state: int | None = None,
) -> dict:
    """Test SRF embedding recovery of latent structure.

    Returns dict with: r_obs, raw_p, corrected_p (if fdr), significant (if fdr), w_aligned
    """
    rng = np.random.default_rng(random_state)
    n, k = w.shape
    align_fn = global_alignment if alignment == "global" else loo_alignment

    w_aligned = align_fn(w, x)
    r_obs = np.array([_correlation(w_aligned[:, d], x[:, d], two_sided) for d in range(k)])

    # Null distribution
    r_null = np.zeros((permutations, k))
    for i in range(permutations):
        perm = rng.permutation(n)
        x_perm = x[perm]
        w_perm = align_fn(w, x_perm)
        for d in range(k):
            r_null[i, d] = _correlation(w_perm[:, d], x_perm[:, d], two_sided)

    raw_p = np.array([_pvalue(r_obs[d], r_null[:, d]) for d in range(k)])
    result = {"r_obs": r_obs, "raw_p": raw_p, "w_aligned": w_aligned}

    if fdr:
        from statsmodels.stats.multitest import multipletests
        reject, corrected_p, _, _ = multipletests(raw_p, alpha=alpha, method="fdr_bh")
        result["corrected_p"] = corrected_p
        result["significant"] = reject

    return result


# Legacy aliases
def loo_alignment_test_multi(w, x, permutations=1000, alpha=0.05, random_state=None):
    return alignment_test(
        w, x, "loo", permutations=permutations, alpha=alpha, random_state=random_state
    )


def global_alignment_test_multi(w, x, permutations=1000, alpha=0.05, random_state=None):
    return alignment_test(
        w,
        x,
        "global",
        permutations=permutations,
        alpha=alpha,
        random_state=random_state,
    )
