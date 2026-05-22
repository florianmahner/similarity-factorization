"""Masked-bootstrap engine for eigenspace coherence.

Given a symmetric similarity matrix ``s`` and a grid of sampling
probabilities, draws Bernoulli-masked + rescaled replicates of ``s``,
takes their top eigenvectors, and reports two cumulative statistics
versus a reference subspace:

- ``overlap``: how much of the reference subspace the bootstrap
  recovers at each rank;
- ``projected_trace``: how much spectral mass of ``s`` the bootstrap
  subspace captures at each rank.

Signal eigenvectors are stable under masking; noise eigenvectors are
not. Downstream modules turn these statistics into a rank estimate and
a calibrated sampling fraction.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh


def symmetrize(s: np.ndarray) -> np.ndarray:
    """Symmetrize a matrix while preserving NaN semantics.

    For each pair (i, j): average if both finite, copy the finite value
    if only one side is finite, keep NaN if both missing. Diagonal NaNs
    are replaced with zero.
    """
    s = np.asarray(s, dtype=np.float64)
    st = s.T
    a = np.isfinite(s)
    b = np.isfinite(st)

    out = np.full_like(s, np.nan)
    both = a & b
    out[both] = 0.5 * (s[both] + st[both])
    only_a = a & ~b
    out[only_a] = s[only_a]
    only_b = ~a & b
    out[only_b] = st[only_b]

    d = np.diag(out).copy()
    d[~np.isfinite(d)] = 0.0
    np.fill_diagonal(out, d)
    return out


def observation_mask(s_sym: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Build observation mask from a symmetrized matrix.

    Returns
    -------
    s_filled : ndarray of shape (n, n)
        Input with NaN replaced by zero.
    mask : ndarray of shape (n, n)
        Float 0/1 observation mask, symmetrized via ``(W + W.T) > 0``
        with the diagonal forced to 1.
    observed_rate : float
        Off-diagonal fraction of observed entries, clipped to
        ``[1e-12, 1.0]``.
    """
    n = s_sym.shape[0]
    mask = np.isfinite(s_sym).astype(np.float64)
    mask = ((mask + mask.T) > 0.0).astype(np.float64)
    np.fill_diagonal(mask, 1.0)
    s_filled = np.nan_to_num(s_sym, nan=0.0)

    upper = np.triu_indices(n, k=1)
    observed_rate = float(mask[upper].mean()) if upper[0].size else 1.0
    return s_filled, mask, float(np.clip(observed_rate, 1e-12, 1.0))


def _top_eigenpairs(a: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k eigenpairs of a symmetric matrix, descending eigenvalue order.

    Tries scipy.sparse.linalg.eigsh for efficiency, falls back to dense eigh.
    """
    n = a.shape[0]
    if k < n:
        try:
            values, vectors = eigsh(a, k=k, which="LA", tol=1e-6)
            order = np.argsort(values)[::-1]
            return values[order], vectors[:, order]
        except Exception:
            pass
    values, vectors = eigh(a)
    order = np.argsort(values)[::-1][:k]
    return values[order], vectors[:, order]


def _randomized_top_eigenpairs(
    a: np.ndarray,
    k: int,
    oversample: int = 10,
    n_iter: int = 2,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k eigenpairs via randomized subspace iteration.

    Used when the input has missing entries and a direct top-k
    eigendecomposition on the zero-filled matrix would be biased.
    """
    n = a.shape[0]
    width = min(n, k + oversample)
    rng = np.random.default_rng(int(random_state) & 0xFFFFFFFF)
    y = a @ rng.standard_normal((n, width))
    for _ in range(n_iter):
        y = a @ (a @ y)
    q, _ = np.linalg.qr(y, mode="reduced")
    small = q.T @ a @ q
    values, vectors_small = eigh(0.5 * (small + small.T))
    order = np.argsort(values)[::-1][:k]
    return values[order], q @ vectors_small[:, order]


def _dense_top_eigenpairs(a: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k eigenpairs via dense ``eigh``, descending eigenvalue order."""
    values, vectors = eigh(a)
    order = np.argsort(values)[::-1][:k]
    return values[order], vectors[:, order]


def reference_eigenpairs(
    s_filled: np.ndarray,
    mask: np.ndarray,
    observed_rate: float,
    k_max: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k_max reference eigenpairs of the observed matrix.

    Fully observed: dense ``eigh`` on the symmetrized matrix, sliced
    top-k descending. Partially observed: randomized subspace iteration
    on the observed-only proxy ``mask * s_filled`` with the diagonal
    preserved.
    """
    if observed_rate >= 1.0 - 1e-15:
        return _dense_top_eigenpairs(0.5 * (s_filled + s_filled.T), k_max)

    proxy = mask * s_filled
    np.fill_diagonal(proxy, np.diag(s_filled))
    return _randomized_top_eigenpairs(
        0.5 * (proxy + proxy.T), k_max, random_state=random_state
    )


def _bernoulli_replicate(
    s_filled: np.ndarray,
    mask: np.ndarray,
    p: float,
    rng: np.random.Generator,
    upper: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """One symmetric Bernoulli-masked, 1/p-rescaled replicate of ``s_filled``.

    Off-diagonal entries are kept with probability ``p`` and rescaled by
    ``1/p`` so the replicate is an unbiased estimate of ``s_filled``.
    The diagonal is preserved, and unobserved entries (``mask == 0``)
    stay at zero. Multiplication by the precomputed ``1/p`` scale
    matches the reference's order of operations exactly.
    """
    n = s_filled.shape[0]
    scale = 1.0 / max(p, 1e-12)
    keep = (rng.random(upper[0].size) < p).astype(np.float64)
    values = keep * mask[upper] * s_filled[upper] * scale

    a = np.zeros((n, n), dtype=np.float64)
    a[upper] = values
    a[(upper[1], upper[0])] = values
    np.fill_diagonal(a, np.diag(s_filled))
    return 0.5 * (a + a.T)


def _cumulative_overlap(u_boot: np.ndarray, u_ref: np.ndarray) -> np.ndarray:
    """Cumulative squared overlap with the reference subspace per rank.

    Entry ``r-1`` is ``sum_{j<=r} (u_boot[:, j]^T u_ref[:, j])^2``,
    clipped to [0, 1].
    """
    g = u_boot.T @ u_ref
    k_max = g.shape[0]
    cumulative = np.cumsum(g * g, axis=0)[np.arange(k_max), np.arange(k_max)]
    return np.clip(cumulative, 0.0, 1.0)


def _cumulative_projected_trace(s: np.ndarray, u_boot: np.ndarray) -> np.ndarray:
    """Cumulative ``tr(P_r S)`` per rank.

    Entry ``r-1`` is ``sum_{j<=r} u_j^T S u_j``, the trace of ``S``
    projected onto the top-r bootstrap subspace.
    """
    per_column = np.einsum("ij,ij->j", u_boot, s @ u_boot)
    return np.cumsum(per_column)


def _bootstrap_one_grid_point(args: tuple) -> tuple[int, np.ndarray, np.ndarray]:
    """Worker: collect ``n_bootstrap`` replicate statistics at one sampling rate."""
    i, p, s_filled, mask, u_ref, k_max, n_bootstrap, seed_base = args
    n = s_filled.shape[0]
    upper = np.triu_indices(n, k=1)

    overlap = np.empty((k_max, n_bootstrap), dtype=np.float64)
    projected = np.empty((k_max, n_bootstrap), dtype=np.float64)
    for b in range(n_bootstrap):
        seed = (seed_base + 1_000_003 * i + 9176 * b) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        a = _bernoulli_replicate(s_filled, mask, p, rng, upper)
        _, u_boot = _top_eigenpairs(a, k_max)
        overlap[:, b] = _cumulative_overlap(u_boot, u_ref)
        projected[:, b] = _cumulative_projected_trace(s_filled, u_boot)
    return i, overlap, projected


def bootstrap_coherence(
    s_filled: np.ndarray,
    mask: np.ndarray,
    u_ref: np.ndarray,
    k_max: int,
    sampling_grid: np.ndarray,
    n_bootstrap: int,
    random_state: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap eigenspace coherence across a grid of sampling probabilities.

    Returns
    -------
    overlap : ndarray of shape (k_max, P, B)
        Cumulative overlap with the reference subspace per (rank, p,
        replicate).
    projected_trace : ndarray of shape (k_max, P, B)
        Cumulative ``tr(P_r S)`` per (rank, p, replicate).
    """
    tasks = [
        (i, float(sampling_grid[i]), s_filled, mask, u_ref, k_max, n_bootstrap, random_state)
        for i in range(len(sampling_grid))
    ]
    if n_jobs == 1:
        results = [_bootstrap_one_grid_point(t) for t in tasks]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_bootstrap_one_grid_point)(t) for t in tasks
        )
    results.sort(key=lambda r: r[0])
    overlap = np.stack([r[1] for r in results], axis=1)
    projected = np.stack([r[2] for r in results], axis=1)
    return overlap, projected
