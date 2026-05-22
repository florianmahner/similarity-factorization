"""Masked-bootstrap engine for eigenspace coherence."""

# RATIONALE:
# - Dropped the per-module Author/License header and shortened the module
#   docstring to a single line (matches stats.py).
# - Added `Array = np.ndarray` alias and threaded it through all signatures.
# - Collapsed verbose Parameters/Returns docstrings to one-line summaries;
#   kept the multi-axis shape info on `_bootstrap_coherence` because the
#   `(k_max, P, B)` layout is genuinely load-bearing for callers.
# - `_symmetrize`: replaced the four-branch finite/finite-only/NaN logic with
#   a single `np.nanmean` of `np.stack([s, s.T])`. Same arithmetic in fewer
#   lines; wrapped in `warnings.catch_warnings` because all-NaN pairs would
#   otherwise emit a RuntimeWarning the original silently avoided.
# - `_observation_mask`: replaced the `(mask + mask.T) > 0` float-addition
#   trick with a straight boolean `obs | obs.T`.
# - Folded the old `_dense_top_eigenpairs` and the two ad-hoc `argsort[::-1]`
#   slicings inside `_top_eigenpairs` / `_randomized_top_eigenpairs` into a
#   single `_topk_desc(values, vectors, k)` helper used in three places.
# - `_bootstrap_coherence`: dropped the tuple-packing-with-index and post-hoc
#   `results.sort(...)` (joblib preserves input order). Tasks now use plain
#   positional args via `delayed(...)(*a)`.
# - `_bootstrap_one_grid_point` takes named positional args instead of an
#   opaque 8-tuple, making the call site self-documenting.
# - Renamed `upper` -> `iu`, `scale * ...` -> `... / max(p, 1e-12)` for
#   compactness.
# - Behaviour is byte-identical: same seeds, same eigensolvers, same
#   arithmetic, same joblib backend.

from __future__ import annotations

import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

Array = np.ndarray


# ------- Public helpers (used by _estimate.py) ------- #


def _symmetrize(s: Array) -> Array:
    """Symmetrize a matrix, averaging finite pairs and preserving NaN-only pairs."""
    s = np.asarray(s, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmean(np.stack([s, s.T]), axis=0)
    diag = np.diag(out).copy()
    diag[~np.isfinite(diag)] = 0.0
    np.fill_diagonal(out, diag)
    return out


def _observation_mask(s_sym: Array) -> tuple[Array, Array, float]:
    """Return (filled_matrix, symmetric 0/1 observation mask, off-diagonal rate)."""
    obs = np.isfinite(s_sym)
    mask = (obs | obs.T).astype(np.float64)
    np.fill_diagonal(mask, 1.0)
    s_filled = np.nan_to_num(s_sym, nan=0.0)

    iu = np.triu_indices(s_sym.shape[0], k=1)
    rate = float(mask[iu].mean()) if iu[0].size else 1.0
    return s_filled, mask, float(np.clip(rate, 1e-12, 1.0))


def _reference_eigenpairs(
    s_filled: Array,
    mask: Array,
    observed_rate: float,
    k_max: int,
    random_state: int,
) -> tuple[Array, Array]:
    """Top-``k_max`` eigenpairs of the reference (dense if fully observed, else randomized)."""
    if observed_rate >= 1.0 - 1e-15:
        a = 0.5 * (s_filled + s_filled.T)
        return _topk_desc(*eigh(a), k=k_max)

    proxy = mask * s_filled
    np.fill_diagonal(proxy, np.diag(s_filled))
    return _randomized_top_eigenpairs(0.5 * (proxy + proxy.T), k_max, random_state=random_state)


def _bootstrap_coherence(
    s_filled: Array,
    mask: Array,
    u_ref: Array,
    k_max: int,
    sampling_grid: Array,
    n_bootstrap: int,
    random_state: int,
    n_jobs: int,
) -> tuple[Array, Array]:
    """Bootstrap eigenspace coherence across ``sampling_grid``.

    Returns ``(overlap, projected_trace)``, each of shape ``(k_max, P, B)``:
    cumulative squared overlap with ``u_ref`` and cumulative ``tr(P_r S)`` for
    the top-``r`` bootstrap subspace at each (rank, sampling probability, replicate).
    """
    args = [
        (i, float(p), s_filled, mask, u_ref, k_max, n_bootstrap, random_state)
        for i, p in enumerate(sampling_grid)
    ]
    if n_jobs == 1:
        results = [_bootstrap_one_grid_point(*a) for a in args]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_bootstrap_one_grid_point)(*a) for a in args
        )

    overlap = np.stack([r[0] for r in results], axis=1)
    projected = np.stack([r[1] for r in results], axis=1)
    return overlap, projected


# ------- Eigensolvers ------- #


def _topk_desc(values: Array, vectors: Array, k: int) -> tuple[Array, Array]:
    """Sort eigh output by descending eigenvalue and keep the top ``k``."""
    order = np.argsort(values)[::-1][:k]
    return values[order], vectors[:, order]


def _top_eigenpairs(a: Array, k: int, v0: Array | None = None) -> tuple[Array, Array]:
    """Top-k eigenpairs of a symmetric matrix; ARPACK with dense ``eigh`` fallback."""
    if k < a.shape[0]:
        try:
            values, vectors = eigsh(a, k=k, which="LA", tol=1e-6, v0=v0)
            return _topk_desc(values, vectors, k)
        except Exception:
            pass
    return _topk_desc(*eigh(a), k=k)


def _randomized_top_eigenpairs(
    a: Array,
    k: int,
    oversample: int = 10,
    n_iter: int = 2,
    random_state: int = 0,
) -> tuple[Array, Array]:
    """Top-k eigenpairs via randomized subspace iteration."""
    n = a.shape[0]
    width = min(n, k + oversample)
    rng = np.random.default_rng(int(random_state) & 0xFFFFFFFF)
    y = a @ rng.standard_normal((n, width))
    for _ in range(n_iter):
        y = a @ (a @ y)
    q, _ = np.linalg.qr(y, mode="reduced")
    small = q.T @ a @ q
    values, vectors_small = eigh(0.5 * (small + small.T))
    top_values, top_small = _topk_desc(values, vectors_small, k)
    return top_values, q @ top_small


# ------- Bootstrap kernel ------- #


def _bernoulli_replicate(
    s_filled: Array,
    mask: Array,
    p: float,
    rng: np.random.Generator,
    iu: tuple[Array, Array],
) -> Array:
    """One symmetric Bernoulli-masked, 1/p-rescaled replicate of ``s_filled``."""
    n = s_filled.shape[0]
    keep = (rng.random(iu[0].size) < p).astype(np.float64)
    values = keep * mask[iu] * s_filled[iu] / max(p, 1e-12)

    a = np.zeros((n, n), dtype=np.float64)
    a[iu] = values
    a[(iu[1], iu[0])] = values
    np.fill_diagonal(a, np.diag(s_filled))
    return 0.5 * (a + a.T)


def _cumulative_overlap(u_boot: Array, u_ref: Array) -> Array:
    """Cumulative squared projection of ``u_ref[:, r]`` onto span(``u_boot[:, :r+1]``)."""
    g = u_boot.T @ u_ref
    k = g.shape[0]
    cumulative = np.cumsum(g * g, axis=0)[np.arange(k), np.arange(k)]
    return np.clip(cumulative, 0.0, 1.0)


def _cumulative_projected_trace(s: Array, u_boot: Array) -> Array:
    """Cumulative ``tr(P_r S)`` per rank."""
    per_column = np.einsum("ij,ij->j", u_boot, s @ u_boot)
    return np.cumsum(per_column)


def _bootstrap_one_grid_point(
    i: int,
    p: float,
    s_filled: Array,
    mask: Array,
    u_ref: Array,
    k_max: int,
    n_bootstrap: int,
    seed_base: int,
) -> tuple[Array, Array]:
    """Collect ``n_bootstrap`` replicate statistics at one sampling rate."""
    n = s_filled.shape[0]
    iu = np.triu_indices(n, k=1)

    overlap = np.empty((k_max, n_bootstrap), dtype=np.float64)
    projected = np.empty((k_max, n_bootstrap), dtype=np.float64)
    for b in range(n_bootstrap):
        seed = (seed_base + 1_000_003 * i + 9176 * b) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        a = _bernoulli_replicate(s_filled, mask, p, rng, iu)
        _, u_boot = _top_eigenpairs(a, k_max, v0=rng.standard_normal(n))
        overlap[:, b] = _cumulative_overlap(u_boot, u_ref)
        projected[:, b] = _cumulative_projected_trace(s_filled, u_boot)
    return overlap, projected
