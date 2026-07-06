"""Eigenspace coherence for dimensionality estimation.

Estimates the number of signal dimensions (k*) in a symmetric
similarity matrix by measuring how stable each eigenspace
dimension is under random entry masking.

The method works by:
1. Computing a reference eigenspace from the full matrix
2. Repeatedly masking entries at rate p and re-computing eigenvectors
3. Measuring overlap (Iproj) between bootstrap and reference eigenvectors
4. Estimating scaled leakage (kappa) per dimension
5. Finding the changepoint where kappa jumps from signal to noise
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.linalg as la

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer 1: Matrix preparation
# ---------------------------------------------------------------------------


def _symmetrize(s: np.ndarray) -> np.ndarray:
    """Symmetrize a matrix while preserving NaN semantics.

    For each pair (i, j):
    - average if both finite
    - copy the finite value if only one side is finite
    - keep NaN if both missing

    Diagonal NaNs are replaced with zero.
    """
    s = np.asarray(s, dtype=float)
    st = s.T
    a = np.isfinite(s)
    b = np.isfinite(st)

    out = np.full_like(s, np.nan, dtype=float)

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


def _observation_mask(
    s_sym: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build observation mask from a symmetrized matrix.

    Parameters
    ----------
    s_sym : (n, n) array
        Symmetrized similarity matrix (may contain NaN).

    Returns
    -------
    s_filled : (n, n) array
        Input with NaN replaced by zero.
    mask : (n, n) array
        Symmetric 0/1 observation mask with diagonal forced to 1.
    obs_rate : float
        Off-diagonal observation rate, clipped to [eps, 1].
    """
    n = s_sym.shape[0]
    mask = np.isfinite(s_sym).astype(float)
    mask = ((mask + mask.T) > 0).astype(float)
    np.fill_diagonal(mask, 1.0)

    s_filled = np.nan_to_num(s_sym, nan=0.0)

    if n <= 1:
        obs_rate = 1.0
    else:
        iu = np.triu_indices(n, k=1)
        obs_rate = float(mask[iu].mean()) if iu[0].size else 1.0
    obs_rate = float(np.clip(obs_rate, eps, 1.0))

    return s_filled, mask, obs_rate


# ---------------------------------------------------------------------------
# Layer 2: Reference eigenspace
# ---------------------------------------------------------------------------


def _reference_eigenpairs(
    s: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k eigenpairs of a symmetric matrix in descending order.

    Parameters
    ----------
    s : (n, n) array
        Symmetric similarity matrix (fully observed, no NaN).
    k : int
        Number of top eigenpairs to return.

    Returns
    -------
    evals : (k,) array
        Top-k eigenvalues, descending.
    evecs : (n, k) array
        Corresponding eigenvectors with deterministic sign convention.
    """
    s = 0.5 * (s + s.T)
    evals_all, evecs_all = la.eigh(s)
    idx = np.argsort(evals_all)[::-1][:k]
    evals = evals_all[idx]
    evecs = evecs_all[:, idx]
    evecs = _sign_normalize(evecs)
    return evals, evecs


def _sign_normalize(q: np.ndarray) -> np.ndarray:
    """Fix sign ambiguity: largest-magnitude entry in each column is positive."""
    for j in range(q.shape[1]):
        i = int(np.argmax(np.abs(q[:, j])))
        if q[i, j] < 0.0:
            q[:, j] *= -1.0
    return q


# ---------------------------------------------------------------------------
# Layer 3: Bootstrap coherence engine
# ---------------------------------------------------------------------------


def _bootstrap_sample(
    s_filled: np.ndarray,
    mask: np.ndarray,
    p: float,
    rng: np.random.Generator,
    iu: tuple[np.ndarray, np.ndarray],
    eps: float = 1e-12,
) -> np.ndarray:
    """One symmetric Bernoulli-masked bootstrap replicate.

    Off-diagonal entries sampled with probability p, rescaled by 1/p.
    Missing entries suppressed through the observation mask.
    Diagonal is preserved.
    """
    n = s_filled.shape[0]
    p = float(p)
    scale = 1.0 / max(p, eps)

    bern = (rng.random(iu[0].size) < p).astype(float)
    vals = bern * mask[iu] * s_filled[iu] * scale

    a = np.zeros((n, n), dtype=float)
    a[iu] = vals
    a[(iu[1], iu[0])] = vals
    np.fill_diagonal(a, np.diag(s_filled))
    a = 0.5 * (a + a.T)
    return a


def _eigenspace_overlap(
    u_boot: np.ndarray,
    u_ref: np.ndarray,
) -> np.ndarray:
    """Incremental projection overlap between bootstrap and reference eigenvectors.

    Computes Iproj[k] = sum_{j<=k} (u_boot_j . u_ref_j)^2 for each k.
    """
    g = u_boot.T @ u_ref
    g2 = g * g
    k_max = g2.shape[0]
    iproj = np.cumsum(g2, axis=0)[np.arange(k_max), np.arange(k_max)]
    return np.clip(iproj, 0.0, 1.0)


def _top_eigenvectors(a: np.ndarray, k: int) -> np.ndarray:
    """Top-k eigenvectors of a symmetric matrix.

    Tries scipy.sparse.linalg.eigsh for efficiency; falls back to dense eigh.
    """
    n = a.shape[0]
    eigsh = _try_eigsh()
    if eigsh is not None and k < n:
        try:
            vals, vecs = eigsh(a, k=k, which="LA", tol=1e-6)
            idx = np.argsort(vals)[::-1]
            return vecs[:, idx]
        except Exception:
            pass

    vals, vecs = la.eigh(a)
    idx = np.argsort(vals)[::-1][:k]
    return vecs[:, idx]


def _try_eigsh():
    """Return scipy.sparse.linalg.eigsh if available, else None."""
    try:
        from scipy.sparse.linalg import eigsh

        return eigsh
    except Exception:
        return None


def _worker_one_p(args: tuple) -> tuple[int, np.ndarray]:
    """Bootstrap coherence for one masking rate p.

    Returns (i, iproj_p) where iproj_p has shape (k_max, n_boot).
    """
    (i, p, s_filled, mask, u_ref, k_max, n_boot, seed_base) = args

    n = s_filled.shape[0]
    iu = np.triu_indices(n, k=1)
    iproj_p = np.zeros((k_max, n_boot), dtype=float)

    for b in range(n_boot):
        seed = (seed_base + 1000003 * i + 9176 * b) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)

        a = _bootstrap_sample(s_filled, mask, p, rng, iu)
        u_boot = _top_eigenvectors(a, k_max)
        iproj_p[:, b] = _eigenspace_overlap(u_boot, u_ref)

    return i, iproj_p


def _bootstrap_coherence(
    s: np.ndarray,
    k_max: int,
    p_list: np.ndarray,
    n_boot: int = 20,
    random_state: int = 0,
    n_jobs: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute bootstrap eigenspace coherence across masking rates.

    Parameters
    ----------
    s : (n, n) array
        Symmetric similarity matrix (may contain NaN).
    k_max : int
        Maximum number of dimensions to test.
    p_list : (n_p,) array
        Masking probabilities in (0, 1].
    n_boot : int
        Number of bootstrap replicates per masking rate.
    random_state : int
        Seed for reproducibility.
    n_jobs : int or None
        Parallel workers (None = cpu_count - 1).

    Returns
    -------
    iproj_boot : (k_max, n_p, n_boot) array
        Bootstrap overlap values.
    evals_ref : (k_max,) array
        Reference eigenvalues.
    """
    import os
    from joblib import Parallel, delayed

    s_sym = _symmetrize(s)
    s_filled, mask, _ = _observation_mask(s_sym)
    evals_ref, u_ref = _reference_eigenpairs(s_filled, k_max)

    p_list = np.asarray(p_list, dtype=float)
    n_p = len(p_list)

    cpu = os.cpu_count() or 2
    if n_jobs is None:
        n_jobs_eff = max(1, cpu - 1)
    else:
        n_jobs_eff = max(1, min(int(n_jobs), cpu - 1))

    tasks = [
        (i, float(p_list[i]), s_filled, mask, u_ref, k_max, n_boot, random_state)
        for i in range(n_p)
    ]

    if n_jobs_eff == 1:
        results = [_worker_one_p(t) for t in tasks]
    else:
        results = Parallel(n_jobs=n_jobs_eff, backend="loky")(
            delayed(_worker_one_p)(t) for t in tasks
        )

    results.sort(key=lambda x: x[0])
    iproj_boot = np.zeros((k_max, n_p, n_boot), dtype=float)
    for i, iproj_p in results:
        iproj_boot[:, i, :] = iproj_p

    return iproj_boot, evals_ref


# ---------------------------------------------------------------------------
# Layer 4: Kappa estimation
# ---------------------------------------------------------------------------


def _scaled_leakage(
    iproj_median: np.ndarray,
    p_list: np.ndarray,
    hi_quantile: float = 0.85,
) -> np.ndarray:
    """Estimate kappa per dimension from the high-p band.

    For each dimension k, the scaled leakage is:
        ell_k(p) = (1 - Iproj_median[k, p]) * p / (1 - p)

    Kappa is the median of ell_k over masking rates in the high-p band.
    Signal dimensions have low kappa; noise dimensions leak and have high kappa.

    Parameters
    ----------
    iproj_median : (k_max, n_p) array
        Median Iproj across bootstrap replicates.
    p_list : (n_p,) array
        Masking probabilities.
    hi_quantile : float
        Quantile of p_list defining the high-p band (default 0.85).

    Returns
    -------
    kappa : (k_max,) array
        Scaled leakage per dimension.
    """
    p_list = np.asarray(p_list, dtype=float)
    p_hi = float(np.quantile(p_list, hi_quantile))
    hi_idx = np.where(p_list >= p_hi)[0]
    if hi_idx.size == 0:
        hi_idx = np.array([len(p_list) - 1], dtype=int)

    scale = p_list[hi_idx] / np.maximum(1.0 - p_list[hi_idx], 1e-12)
    ell = (1.0 - iproj_median[:, hi_idx]) * scale[np.newaxis, :]
    return np.median(ell, axis=1).astype(float)


def _smooth_median(x: np.ndarray, w: int) -> np.ndarray:
    """Running median smoother."""
    if w <= 1:
        return x.copy()
    n = x.size
    out = np.empty(n, dtype=float)
    half = w // 2
    for i in range(n):
        out[i] = np.median(x[max(0, i - half) : min(n, i + half + 1)])
    return out


def _largest_jump(
    kappa: np.ndarray,
    k_list: np.ndarray,
    smooth_window: int = 5,
    min_k: int = 2,
) -> int:
    """Find the changepoint where kappa jumps from signal to noise.

    Returns k_cut: dimensions {k <= k_cut} are signal.

    Parameters
    ----------
    kappa : (K,) array
        Scaled leakage per dimension.
    k_list : (K,) array
        Dimension indices.
    smooth_window : int
        Window for median smoothing before differentiation.
    min_k : int
        Minimum k to consider for the changepoint.

    Returns
    -------
    k_cut : int
        Last signal dimension.
    """
    kappa = np.asarray(kappa, dtype=float)
    k_list = np.asarray(k_list, dtype=int)

    if kappa.size < 3:
        return int(k_list[-1])

    sm = _smooth_median(kappa, max(1, smooth_window))
    d = sm[1:] - sm[:-1]

    valid = np.where(k_list[:-1] >= min_k)[0]
    if valid.size == 0:
        valid = np.arange(kappa.size - 1)

    i_star = int(valid[np.argmax(d[valid])])
    return int(k_list[i_star])


# ---------------------------------------------------------------------------
# Layer 5: Operating point (p*) via signal-noise lift-off
# ---------------------------------------------------------------------------


def _estimate_p_star(
    iproj_boot: np.ndarray,
    k_star: int,
    p_list: np.ndarray,
    ci_level: float = 0.90,
    noise_quantile: float = 0.90,
    require_consecutive: int = 3,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Find the smallest masking rate where signal lifts off from noise.

    Compares the median Iproj for the boundary dimension k* against
    the upper envelope of noise dimensions (k > k*). The operating
    point p* is the first p where the signal clears the noise for
    ``require_consecutive`` grid points.

    Parameters
    ----------
    iproj_boot : (k_max, n_p, n_boot) array
        Bootstrap overlap values.
    k_star : int
        Boundary dimension (1-indexed).
    p_list : (n_p,) array
        Masking probabilities.
    ci_level : float
        Confidence level for signal/noise envelopes (default 0.90).
    noise_quantile : float
        Quantile across noise dimensions for the reference curve.
    require_consecutive : int
        Number of consecutive p-grid points where the gap must hold.

    Returns
    -------
    p_star : float
        Smallest masking rate where signal lifts off from noise.
    signal_curve : (n_p,) array
        Median Iproj for the boundary dimension.
    noise_curve : (n_p,) array
        Upper envelope of noise dimensions.
    """
    k_max, n_p, _ = iproj_boot.shape
    alpha = (1.0 - ci_level) / 2.0

    # Signal: median of boundary dimension k*
    signal_curve = np.median(iproj_boot[k_star - 1], axis=1)

    # Noise: upper CI of dimensions k > k*, aggregated at noise_quantile
    if k_star < k_max:
        noise_hi = np.quantile(iproj_boot[k_star:], 1.0 - alpha, axis=2)
        noise_curve = np.quantile(noise_hi, noise_quantile, axis=0)
    else:
        noise_curve = np.zeros(n_p, dtype=float)

    # Lift-off: first p where signal > noise for consecutive points
    gap = signal_curve - noise_curve
    p_star = _first_consecutive_above(p_list, gap, 0.0, require_consecutive)

    return p_star, signal_curve, noise_curve


def _first_consecutive_above(
    p_list: np.ndarray,
    curve: np.ndarray,
    threshold: float,
    run: int,
) -> float:
    """First p where curve >= threshold for ``run`` consecutive grid points."""
    above = curve >= threshold
    streak = 0
    for i, ok in enumerate(above):
        streak = streak + 1 if ok else 0
        if streak >= run:
            return float(p_list[i - run + 1])
    return float(p_list[-1])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_rank(
    s: np.ndarray,
    k_max: int,
    p_list: np.ndarray | None = None,
    n_boot: int = 20,
    random_state: int = 0,
    n_jobs: int | None = None,
    hi_quantile: float = 0.85,
    smooth_window: int = 5,
    ci_level: float = 0.90,
    noise_quantile: float = 0.90,
    require_consecutive: int = 3,
) -> dict:
    """Estimate the number of signal dimensions and operating point.

    Computes bootstrap eigenspace coherence, estimates scaled leakage
    (kappa) per dimension, finds the changepoint where kappa jumps
    from signal to noise (k*), and determines the minimum masking rate
    where the boundary dimension lifts off from the noise floor (p*).

    Parameters
    ----------
    s : (n, n) array
        Symmetric similarity matrix (may contain NaN for missing entries).
    k_max : int
        Maximum number of dimensions to test.
    p_list : (n_p,) array or None
        Masking probabilities in (0, 1]. If None, uses linspace(0.1, 0.95, 20).
    n_boot : int
        Number of bootstrap replicates per masking rate.
    random_state : int
        Seed for reproducibility.
    n_jobs : int or None
        Parallel workers (None = cpu_count - 1).
    hi_quantile : float
        Quantile of p_list defining the high-p band for kappa estimation.
    smooth_window : int
        Window for median smoothing in changepoint detection.
    ci_level : float
        Confidence level for signal/noise envelopes in p* estimation.
    noise_quantile : float
        Quantile across noise dimensions for the noise reference curve.
    require_consecutive : int
        Number of consecutive p-grid points for lift-off detection.

    Returns
    -------
    result : dict
        k_star : int
            Estimated number of signal dimensions.
        p_star : float
            Operating point: smallest masking rate where the boundary
            dimension's lower CI lifts off above the noise envelope.
        kappa : (k_max,) array
            Scaled leakage per dimension.
        iproj_median : (k_max, n_p) array
            Median bootstrap overlap per dimension and masking rate.
        iproj_boot : (k_max, n_p, n_boot) array
            Raw bootstrap overlap values.
        evals_ref : (k_max,) array
            Reference eigenvalues.
        k_list : (k_max,) array
            Dimension indices 1..k_max.
        p_list : (n_p,) array
            Masking probabilities used.
        signal_curve : (n_p,) array
            Median Iproj for boundary dimension k*.
        noise_curve : (n_p,) array
            Upper envelope of noise dimensions.
    """
    if p_list is None:
        p_list = np.linspace(0.1, 0.95, 20)
    p_list = np.asarray(p_list, dtype=float)

    k_list = np.arange(1, k_max + 1)

    iproj_boot, evals_ref = _bootstrap_coherence(
        s,
        k_max=k_max,
        p_list=p_list,
        n_boot=n_boot,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    iproj_median = np.median(iproj_boot, axis=2)
    kappa = _scaled_leakage(iproj_median, p_list, hi_quantile=hi_quantile)
    k_star = _largest_jump(kappa, k_list, smooth_window=smooth_window)
    p_star, signal_curve, noise_curve = _estimate_p_star(
        iproj_boot,
        k_star,
        p_list,
        ci_level=ci_level,
        noise_quantile=noise_quantile,
        require_consecutive=require_consecutive,
    )

    return {
        "k_star": k_star,
        "p_star": p_star,
        "kappa": kappa,
        "iproj_median": iproj_median,
        "iproj_boot": iproj_boot,
        "evals_ref": evals_ref,
        "k_list": k_list,
        "p_list": p_list,
        "signal_curve": signal_curve,
        "noise_curve": noise_curve,
    }
