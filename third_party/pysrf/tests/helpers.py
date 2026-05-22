"""Shared test data generators."""

import numpy as np


def make_symmetric_matrix(n=30, rank=5, noise_level=0.01, seed=42):
    """Generate a low-rank symmetric similarity matrix with noise."""
    rng = np.random.default_rng(seed)
    w = np.abs(rng.standard_normal((n, rank)))
    s = w @ w.T
    s += noise_level * rng.standard_normal((n, n))
    s = (s + s.T) / 2
    return s


def make_missing_matrix(s, fraction=0.3, seed=42):
    """Add symmetric NaN entries to a similarity matrix."""
    rng = np.random.default_rng(seed)
    s_missing = s.copy()
    n = s.shape[0]
    triu_i, triu_j = np.triu_indices(n, k=1)
    n_entries = len(triu_i)
    n_missing = int(n_entries * fraction)
    idx = rng.choice(n_entries, size=n_missing, replace=False)
    for k in idx:
        i, j = triu_i[k], triu_j[k]
        s_missing[i, j] = np.nan
        s_missing[j, i] = np.nan
    return s_missing


def make_bsum_data(n=100, r=10, seed=0):
    """Generate symmetric matrix and initial W for BSUM solver tests."""
    rng = np.random.default_rng(seed)
    m = rng.random((n, n))
    m = (m + m.T) * 0.5
    w0 = rng.random((n, r))
    return m, w0
