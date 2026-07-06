import os

import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
import contextlib
import warnings

import numpy as np
import numpy.linalg as la
from functools import lru_cache

import plotly.graph_objects as go

# ==============================================================================
# Missing-entry aware helpers (S has np.nan off-diagonal)
# ==============================================================================


@lru_cache(maxsize=1)
def _try_import_eigsh():
    """Return ``scipy.sparse.linalg.eigsh`` when SciPy is available, else ``None``."""
    try:
        from scipy.sparse.linalg import eigsh

        return eigsh
    except Exception:
        return None


def _symmetrize_with_nan(S):
    """
    Symmetrize a matrix while preserving missingness semantics.

    For each pair ``(i, j)``:
    - average if both entries are finite,
    - copy the finite value if only one side is finite,
    - keep ``NaN`` if both are missing.
    The diagonal is forced finite by replacing missing values with zero.
    """
    S = np.asarray(S, float)
    St = S.T
    a = np.isfinite(S)
    b = np.isfinite(St)

    out = np.full_like(S, np.nan, dtype=float)

    both = a & b
    out[both] = 0.5 * (S[both] + St[both])

    only_a = a & (~b)
    out[only_a] = S[only_a]

    only_b = (~a) & b
    out[only_b] = St[only_b]

    # keep diagonal as is; if diagonal is NaN, set to 0 by default
    d = np.diag(out).copy()
    d[~np.isfinite(d)] = 0.0
    np.fill_diagonal(out, d)
    return out


def _prepare_observation_mask(S_sym, keep_diag=True, eps=1e-12):
    """
    Build dense value/mask tensors used by masked bootstrap sampling.

    Returns ``(S0, W, q)`` where:
    - ``S0`` is ``S_sym`` with ``NaN`` replaced by zero,
    - ``W`` is a symmetric 0/1 observation mask,
    - ``q`` is the off-diagonal observation rate clipped to ``[eps, 1]``.
    """
    S_sym = np.asarray(S_sym, float)
    n = S_sym.shape[0]

    W = np.isfinite(S_sym).astype(float)
    # enforce symmetry on W (it should already be symmetric after _symmetrize_with_nan)
    W = (W + W.T) > 0.0
    W = W.astype(float)

    if keep_diag:
        np.fill_diagonal(W, 1.0)

    S0 = np.nan_to_num(S_sym, nan=0.0)

    if n <= 1:
        q = 1.0
    else:
        iu = np.triu_indices(n, k=1)
        off_obs = W[iu]
        q = float(off_obs.mean()) if off_obs.size else 1.0
    q = float(np.clip(q, eps, 1.0))

    return S0, W, q


def _apply_unbiased_missingness_scaling(S0, W, q, keep_diag=True):
    """Apply MCAR correction to observed entries and return a symmetric estimate."""
    S_hat = (W * S0) / q
    if keep_diag:
        np.fill_diagonal(S_hat, np.diag(S0))
    S_hat = 0.5 * (S_hat + S_hat.T)
    return S_hat


def _masked_unbiased_spd_missing(S0, W, q, p, rng, keep_diag=True, eps=1e-12, iu=None):
    """
    Sample a symmetric Bernoulli-masked matrix used in one bootstrap replicate.

    Off-diagonal entries are sampled with probability ``p`` and rescaled by ``1/p``.
    Missing entries are suppressed through ``W``. The diagonal is either preserved
    (``keep_diag=True``) or sampled/rescaled consistently with ``p``.
    """
    n = S0.shape[0]
    p = float(p)

    M = np.zeros((n, n), dtype=float)
    if iu is None:
        iu = np.triu_indices(n, k=1)

    # Bernoulli(p) mask on off-diagonal (stored via upper-triangle indices).
    mask_u = (rng.random(iu[0].size) < p).astype(float)
    M[iu] = mask_u
    M[(iu[1], iu[0])] = mask_u

    # combine with observation mask
    MW = M * W

    A = MW * S0
    scale_off = 1.0 / max(p, eps)
    A[iu] *= scale_off
    A[(iu[1], iu[0])] *= scale_off

    if keep_diag:
        np.fill_diagonal(A, np.diag(S0))
    else:
        dmask = (rng.random(n) < p).astype(float)
        np.fill_diagonal(A, dmask * np.diag(S0) / max(p, eps))

    A = 0.5 * (A + A.T)
    return A


def _topk_eigenvectors(A, k, eigsh=None, tol=1e-6, maxiter=None):
    """Return top-``k`` eigenpairs of a symmetric matrix in descending order."""
    n = A.shape[0]
    k = int(k)
    if k <= 0:
        return np.array([], float), np.zeros((n, 0), float)

    if eigsh is not None and k < n:
        vals, vecs = eigsh(A, k=k, which="LA", tol=tol, maxiter=maxiter)
        idx = np.argsort(vals)[::-1]
        return vals[idx], vecs[:, idx]
    else:
        vals, vecs = la.eigh(A)
        idx = np.argsort(vals)[::-1][:k]
        return vals[idx], vecs[:, idx]


def _randomized_topk_eigenspace_symmetric(
    A,
    k,
    oversample=10,
    n_iter=2,
    random_state=0,
):
    """Compute an approximate top-``k`` symmetric eigenspace via random projection."""
    n = A.shape[0]
    k = int(k)
    n_sketch = int(min(n, k + int(oversample)))

    rng = np.random.default_rng(int(random_state) & 0xFFFFFFFF)
    Omega = rng.standard_normal((n, n_sketch))
    Y = A @ Omega

    # power iterations to improve spectral separation (optional)
    for _ in range(int(n_iter)):
        Y = A @ (A @ Y)

    Q, _ = la.qr(Y, mode="reduced")  # n x n_sketch
    B = Q.T @ A @ Q  # n_sketch x n_sketch (small)
    B = 0.5 * (B + B.T)

    evals, evecs_small = la.eigh(B)
    idx = np.argsort(evals)[::-1][:k]
    evals_top = evals[idx]
    evecs_top = Q @ evecs_small[:, idx]
    return evals_top, evecs_top


def _orthonormalize_columns(X):
    """Return a QR-orthonormal basis with deterministic column signs."""
    X = np.asarray(X, float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    n, k = X.shape
    if k == 0:
        return np.zeros((n, 0), float)

    Q, _ = la.qr(X, mode="reduced")
    for j in range(Q.shape[1]):
        idx = int(np.argmax(np.abs(Q[:, j])))
        if Q[idx, j] < 0.0:
            Q[:, j] *= -1.0
    return Q


@lru_cache(maxsize=1)
def _try_import_lobpcg():
    """Return ``scipy.sparse.linalg.lobpcg`` when SciPy is available, else ``None``."""
    try:
        from scipy.sparse.linalg import lobpcg

        return lobpcg
    except Exception:
        return None


def _masked_unbiased_spd_missing_from_uniform(
    S0,
    W,
    p,
    edge_u,
    iu,
    keep_diag=True,
    diag_u=None,
    eps=1e-12,
):
    """
    Build one symmetric bootstrap matrix from pre-sampled uniforms.

    Reusing the same ``edge_u`` across increasing ``p`` yields a nested mask path.
    For each fixed ``p`` the marginal bootstrap law remains exactly Bernoulli(``p``).
    """
    n = S0.shape[0]
    p = float(p)
    scale = 1.0 / max(p, eps)

    vals = ((edge_u < p).astype(float) * W[iu] * S0[iu]) * scale
    A = np.zeros((n, n), dtype=float)
    A[iu] = vals
    A[(iu[1], iu[0])] = vals

    if keep_diag:
        np.fill_diagonal(A, np.diag(S0))
    else:
        if diag_u is None:
            raise ValueError("diag_u is required when keep_diag=False.")
        dmask = (diag_u < p).astype(float)
        np.fill_diagonal(A, dmask * np.diag(S0) * scale)

    A = 0.5 * (A + A.T)
    return A


def _topk_eigenvectors_exact_warm(
    A,
    k,
    X_init=None,
    solver="auto",
    tol=1e-10,
    maxiter=200,
    residual_tol=1e-8,
):
    """
    Exact top-``k`` eigensolve with a warm-start fast path and dense fallback.

    The returned eigenpairs are validated by the residual norm. If the warm-started
    iterative solve does not meet the requested residual tolerance, the routine
    falls back to dense ``eigh``.
    """
    A = 0.5 * (np.asarray(A, float) + np.asarray(A, float).T)
    n = A.shape[0]
    k = int(k)
    if k <= 0:
        return (
            np.array([], float),
            np.zeros((n, 0), float),
            {
                "solver": "none",
                "fallback": False,
                "max_residual": 0.0,
            },
        )

    lobpcg = _try_import_lobpcg()
    use_lobpcg = (
        solver in ("auto", "lobpcg") and lobpcg is not None and k < n and n > (8 * k)
    )

    if use_lobpcg:
        if X_init is None or np.shape(X_init) != (n, k):
            rng = np.random.default_rng(0)
            X = rng.standard_normal((n, k))
        else:
            X = np.array(X_init, dtype=float, copy=True)
        X = _orthonormalize_columns(X)

        try:
            with warnings.catch_warnings(record=True) as wlog:
                warnings.simplefilter("always")
                vals, vecs = lobpcg(
                    A,
                    X,
                    largest=True,
                    tol=float(tol),
                    maxiter=int(maxiter),
                    verbosityLevel=0,
                )
            order = np.argsort(vals)[::-1]
            vals = np.asarray(vals, float)[order]
            vecs = np.asarray(vecs, float)[:, order]
            vecs = _orthonormalize_columns(vecs)
            resid = la.norm(A @ vecs - vecs * vals[np.newaxis, :], axis=0)
            max_resid = float(np.max(resid)) if resid.size else 0.0
            warning_messages = [str(w.message) for w in wlog]
            warning_flag = any(msg for msg in warning_messages)
            if (
                np.isfinite(vals).all()
                and (not warning_flag)
                and max_resid <= float(residual_tol)
            ):
                return (
                    vals,
                    vecs,
                    {
                        "solver": "lobpcg",
                        "fallback": False,
                        "max_residual": max_resid,
                        "warning_count": 0,
                    },
                )
        except Exception:
            pass

    evals_all, evecs_all = la.eigh(A)
    idx = np.argsort(evals_all)[::-1][:k]
    vals = np.asarray(evals_all[idx], float)
    vecs = _orthonormalize_columns(np.asarray(evecs_all[:, idx], float))
    resid = la.norm(A @ vecs - vecs * vals[np.newaxis, :], axis=0)
    max_resid = float(np.max(resid)) if resid.size else 0.0
    return (
        vals,
        vecs,
        {
            "solver": "eigh",
            "fallback": bool(use_lobpcg),
            "max_residual": max_resid,
            "warning_count": 1 if use_lobpcg else 0,
        },
    )


def _fast_iproj_worker_one_boot(args):
    """Compute exact Iproj bootstrap curves for one replicate using warm starts."""
    (
        b,
        p_list,
        S0,
        W,
        U_ref_K,
        k_idx,
        random_seed_base,
        keep_diag,
        solver,
        solver_tol,
        solver_maxiter,
        residual_tol,
        use_nested_masks,
        show_inner_progress,
    ) = args

    n = S0.shape[0]
    Kmax = U_ref_K.shape[1]
    K_count = len(k_idx)
    p_list = np.asarray(p_list, float)
    P = len(p_list)
    iu = np.triu_indices(n, k=1)

    rng = np.random.default_rng((int(random_seed_base) + 9176 * int(b)) & 0xFFFFFFFF)
    shared_edge_u = rng.random(iu[0].size) if use_nested_masks else None
    shared_diag_u = (None if keep_diag else rng.random(n)) if use_nested_masks else None

    Iproj_boot_b = np.zeros((K_count, P), float)
    X_init = np.array(U_ref_K, copy=True)
    solver_counts = {"lobpcg": 0, "eigh": 0, "none": 0}
    fallback_count = 0
    warning_count = 0
    max_residual_seen = 0.0

    p_iter = enumerate(p_list)
    if show_inner_progress:
        p_iter = enumerate(
            tqdm(p_list, total=P, desc=f"fast p-sweep b={b + 1}", leave=False)
        )

    for j, p in p_iter:
        if use_nested_masks:
            edge_u = shared_edge_u
            diag_u = shared_diag_u
        else:
            edge_u = rng.random(iu[0].size)
            diag_u = None if keep_diag else rng.random(n)

        A = _masked_unbiased_spd_missing_from_uniform(
            S0,
            W,
            float(p),
            edge_u,
            iu,
            keep_diag=keep_diag,
            diag_u=diag_u,
        )

        _, U_k, info = _topk_eigenvectors_exact_warm(
            A,
            Kmax,
            X_init=X_init,
            solver=solver,
            tol=solver_tol,
            maxiter=solver_maxiter,
            residual_tol=residual_tol,
        )

        solver_name = info.get("solver", "eigh")
        solver_counts[solver_name] = solver_counts.get(solver_name, 0) + 1
        fallback_count += int(info.get("fallback", False))
        warning_count += int(info.get("warning_count", 0))
        max_residual_seen = max(max_residual_seen, float(info.get("max_residual", 0.0)))

        X_init = np.array(U_k, copy=True)
        G = U_k.T @ U_ref_K
        G2 = G * G
        Iproj_all = np.cumsum(G2, axis=0)[np.arange(Kmax), np.arange(Kmax)]
        Iproj_boot_b[:, j] = np.clip(Iproj_all[k_idx], 0.0, 1.0)

    return (
        b,
        Iproj_boot_b,
        {
            "solver_counts": solver_counts,
            "fallback_count": int(fallback_count),
            "warning_count": int(warning_count),
            "max_residual_seen": float(max_residual_seen),
        },
    )


# ==============================================================================
# Worker (modified to accept S0, W, q)
# ==============================================================================


def _eig_worker_one_p(args):
    """Compute all bootstrap outputs for one masking level ``p``."""
    (
        i,
        p,
        S0,
        W,
        q,
        evals_ref,
        U_ref_K,
        k_list,
        B,
        random_seed_base,
        keep_diag,
        eigsh_tol,
        eigsh_maxiter,
        collect_worker_stats,
        compute_null,
        B_null,
        null_dist,
    ) = args

    n = S0.shape[0]
    p = float(p)
    Kmax = U_ref_K.shape[1]
    k_list = np.asarray(k_list, int)
    K_count = len(k_list)

    eigsh = _try_import_eigsh()
    iu = np.triu_indices(n, k=1)
    k_idx = k_list - 1
    ranks = np.arange(1, Kmax + 1, dtype=float)

    Iproj_boot_p = np.zeros((K_count, B), float)
    C_boot_p = np.zeros((K_count, B), float)
    I_boot_p = np.zeros((K_count, B), float)
    lambda_topk_boot_p = np.full((Kmax, B), np.nan, float)

    null_boot_p = None
    if compute_null and B_null > 0:
        null_boot_p = np.zeros((K_count, B, B_null), float)

    stats = {"eigsh_used": int(eigsh is not None), "fail_count": 0}

    for b in range(B):
        rng = np.random.default_rng(
            (random_seed_base + 1000003 * i + 9176 * b) & 0xFFFFFFFF
        )

        A = _masked_unbiased_spd_missing(S0, W, q, p, rng, keep_diag=keep_diag, iu=iu)

        try:
            vals_k, U_k = _topk_eigenvectors(
                A, Kmax, eigsh=eigsh, tol=eigsh_tol, maxiter=eigsh_maxiter
            )
        except Exception:
            stats["fail_count"] += 1
            vals_k, U_k = _topk_eigenvectors(A, Kmax, eigsh=None)

        lambda_topk_boot_p[: len(vals_k), b] = vals_k

        # overlaps: G = U_k^T U_ref_K  (Kmax x Kmax)
        G = U_k.T @ U_ref_K
        G2 = G * G

        Iproj_all = np.cumsum(G2, axis=0)[np.arange(Kmax), np.arange(Kmax)]
        C_all = np.cumsum(Iproj_all) / ranks
        I_all = (ranks * C_all) - np.concatenate(([0.0], ranks[:-1] * C_all[:-1]))

        Iproj_boot_p[:, b] = np.clip(Iproj_all[k_idx], 0.0, 1.0)
        C_boot_p[:, b] = np.clip(C_all[k_idx], 0.0, 1.0)
        I_boot_p[:, b] = I_all[k_idx]

        # null calibration: v^T P_k v = ||U_{:,:k}^T v||^2
        if null_boot_p is not None:
            if null_dist == "gaussian":
                V = rng.standard_normal((n, B_null))
            elif null_dist == "rademacher":
                V = rng.integers(0, 2, size=(n, B_null)).astype(float)
                V[V == 0.0] = -1.0
            else:
                raise ValueError(f"Unknown null_dist={null_dist}")

            V /= np.maximum(np.sqrt(np.sum(V * V, axis=0, keepdims=True)), 1e-12)
            Acoef = U_k.T @ V
            cumsq = np.cumsum(Acoef * Acoef, axis=0)
            null_boot_p[:, b, :] = cumsq[k_idx, :]

    worker_stats = stats if collect_worker_stats else None
    return (
        i,
        p,
        Iproj_boot_p,
        C_boot_p,
        I_boot_p,
        lambda_topk_boot_p,
        worker_stats,
        null_boot_p,
    )


# ==============================================================================
# Main (modified reference eigenspace estimation via randomized method if NaNs exist)
# ==============================================================================


def compute_incremental_coherence_multi_k_eig_anisotropic(
    S,
    k_list,
    p_list,
    B=20,
    random_state=0,
    keep_diag=True,
    # eigsh controls
    eigsh_tol=1e-6,
    eigsh_maxiter=None,
    # parallel
    n_jobs=None,
    max_nbytes="64M",
    prefer="processes",
    show_progress=True,
    collect_worker_stats=False,
    # visualization
    visualize=True,
    visualize_mode="plotly",
    # tau + CI diagnostics ===
    compute_null=True,
    B_null=30,
    null_dist="gaussian",  # "gaussian" or "rademacher"
    alpha_tau=0.95,  # tau quantile (null)
    ci_level=0.95,  # CI for Iproj across bootstrap masks
    use_baseline_correction=False,
    clip_baseline=True,
    plot_tau=True,  # overlay tau curves
    mark_activation=True,  # activation points
    # NEW: randomized reference eigenspace options
    ref_oversample=10,
    ref_n_iter=2,
):
    """
    Estimate incremental eigenspace coherence across multiple ``k`` and mask rates ``p``.

    Workflow:
    1. Symmetrize ``S`` with missing-value awareness.
    2. Build a reference top-``K`` eigenspace (exact when fully observed, randomized when missing).
    3. For each ``p`` and bootstrap replicate, mask/rescale entries, solve top eigenvectors,
       and accumulate ``Iproj``, ``C``, ``I``, optional null calibration, and diagnostics.
    """

    # ---- symmetrize while respecting NaNs ----
    S_sym = _symmetrize_with_nan(S)
    n = S_sym.shape[0]

    k_list = np.asarray(sorted(set(k_list)), int)
    if np.any(k_list <= 0) or np.any(k_list > n):
        raise ValueError("All k in k_list must be in {1, ..., n}.")

    p_list = np.asarray(p_list, float)
    if np.any(p_list <= 0.0) or np.any(p_list > 1.0):
        raise ValueError("All p in p_list must be in (0, 1].")

    Kmax = int(k_list.max())
    K_count = int(len(k_list))
    P = int(len(p_list))

    cpu = os.cpu_count() or 2
    if n_jobs is None:
        n_jobs_eff = max(1, cpu - 1)
    else:
        n_jobs_eff = int(max(1, min(int(n_jobs), max(1, cpu - 1))))

    # ---- build observation mask + unbiased missingness scaling base ----
    S0, W, q = _prepare_observation_mask(S_sym, keep_diag=keep_diag)

    # ---- reference eigendecomposition (exact if fully observed; randomized otherwise) ----
    has_missing_offdiag = q < (1.0 - 1e-15)
    if not has_missing_offdiag:
        # fully observed: use exact eigh (your original behavior)
        S_full = 0.5 * (S0 + S0.T)
        evals_full, evecs_full = la.eigh(S_full)
        idx = np.argsort(evals_full)[::-1]
        evals_ref = evals_full[idx[:Kmax]]
        U_ref_K = evecs_full[:, idx[:Kmax]]
    else:
        # missing off-diagonal: observed-only reference + randomized eigenspace
        S_hat = W * S0
        if keep_diag:
            np.fill_diagonal(S_hat, np.diag(S0))
        S_hat = 0.5 * (S_hat + S_hat.T)
        evals_ref, U_ref_K = _randomized_topk_eigenspace_symmetric(
            S_hat,
            k=Kmax,
            oversample=ref_oversample,
            n_iter=ref_n_iter,
            random_state=random_state,
        )

    # ---- allocate outputs ----
    Iproj_boot = np.zeros((K_count, P, B), float)
    C_boot = np.zeros((K_count, P, B), float)
    I_boot = np.zeros((K_count, P, B), float)
    lambda_topk_boot = np.full((Kmax, P, B), np.nan, float)
    worker_stats_all = [None] * P if collect_worker_stats else None

    null_boot = None
    if compute_null and B_null > 0:
        null_boot = np.zeros((K_count, P, B, B_null), float)

    tasks = [
        (
            i,
            float(p_list[i]),
            S0,
            W,
            q,  # <<< changed
            evals_ref,
            U_ref_K,
            k_list,
            int(B),
            int(random_state or 0),
            bool(keep_diag),
            float(eigsh_tol),
            eigsh_maxiter,
            collect_worker_stats,
            bool(compute_null),
            int(B_null),
            str(null_dist),
        )
        for i in range(P)
    ]

    if n_jobs_eff == 1:
        if show_progress:
            results = []
            for t in tqdm(tasks, total=P, desc="eig-based serial p", leave=True):
                results.append(_eig_worker_one_p(t))
        else:
            results = [_eig_worker_one_p(t) for t in tasks]
    elif show_progress:
        pbar = tqdm(
            total=P, desc=f"eig-based parallel p (n_jobs={n_jobs_eff})", leave=True
        )

        class _TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):
                pbar.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)

        @contextlib.contextmanager
        def _cm():
            old_cb = joblib.parallel.BatchCompletionCallBack
            joblib.parallel.BatchCompletionCallBack = _TqdmBatchCompletionCallback
            try:
                yield pbar
            finally:
                joblib.parallel.BatchCompletionCallBack = old_cb
                pbar.close()

        with _cm():
            results = Parallel(
                n_jobs=n_jobs_eff,
                backend="loky",
                prefer=prefer,
                max_nbytes=max_nbytes,
            )(delayed(_eig_worker_one_p)(t) for t in tasks)
    else:
        results = Parallel(
            n_jobs=n_jobs_eff,
            backend="loky",
            prefer=prefer,
            max_nbytes=max_nbytes,
        )(delayed(_eig_worker_one_p)(t) for t in tasks)

    results.sort(key=lambda x: x[0])

    for i, p, Iproj_p, C_p, I_p, lam_topk_p, wstats, null_p in results:
        Iproj_boot[:, i, :] = Iproj_p
        C_boot[:, i, :] = C_p
        I_boot[:, i, :] = I_p
        lambda_topk_boot[:, i, :] = lam_topk_p
        if collect_worker_stats:
            worker_stats_all[i] = wstats
        if null_boot is not None and null_p is not None:
            null_boot[:, i, :, :] = null_p

    # means (raw, mostly for summary)
    Iproj_mean = Iproj_boot.mean(axis=2)
    C_mean = C_boot.mean(axis=2)
    I_mean = I_boot.mean(axis=2)

    # ---- CI for Iproj across bootstrap masks ----
    ci_lo_q = (1.0 - float(ci_level)) / 2.0
    ci_hi_q = 1.0 - ci_lo_q

    if use_baseline_correction:
        x_boot = _baseline_correct_array(
            Iproj_boot, k_list, n, clip=clip_baseline
        )  # (K,P,B)
    else:
        x_boot = np.clip(Iproj_boot, 0.0, 1.0)

    x_mean = x_boot.mean(axis=2)  # (K,P)
    x_median = np.median(x_boot, axis=2)  # (K,P)
    x_ci_lo = np.quantile(x_boot, ci_lo_q, axis=2)  # (K,P)
    x_ci_hi = np.quantile(x_boot, ci_hi_q, axis=2)  # (K,P)

    # ---- tau_k(p) per (k,p) from null_boot ----
    tau_kp = None
    if null_boot is not None:
        if use_baseline_correction:
            null_x = _baseline_correct_array(
                null_boot, k_list, n, clip=clip_baseline
            )  # (K,P,B,B_null)
        else:
            null_x = np.clip(null_boot, 0.0, 1.0)

        null_flat = null_x.reshape(K_count, P, -1)  # pool over (B * B_null)
        tau_kp = np.quantile(null_flat, float(alpha_tau), axis=2)  # (K,P)

    # ---- activation (count-based, using LOWER CI) ----
    activation_idx = None
    activation_p = None
    activation_idx_component = None
    activation_p_component = None

    if (tau_kp is not None) and mark_activation:
        A = x_ci_lo >= tau_kp  # (K_count, P)

        activation_idx_component = np.full(K_count, -1, int)
        activation_p_component = np.full(K_count, np.nan, float)
        for kk in range(K_count):
            idxs = np.where(A[kk, :])[0]
            if len(idxs) > 0:
                activation_idx_component[kk] = int(idxs[0])
                activation_p_component[kk] = float(p_list[idxs[0]])

        N = A.sum(axis=0)
        activation_idx = np.full(K_count, -1, int)
        activation_p = np.full(K_count, np.nan, float)
        for r in range(K_count):
            idxs = np.where(N >= (r + 1))[0]
            if len(idxs) > 0:
                activation_idx[r] = int(idxs[0])
                activation_p[r] = float(p_list[idxs[0]])

    # summary
    summary = {
        "k": k_list.copy(),
        "Iproj_mean_over_p": Iproj_mean.mean(axis=1),
        "Iproj_value_at_max_p": Iproj_mean[:, -1],
        "C_mean_over_p": C_mean.mean(axis=1),
        "C_value_at_max_p": C_mean[:, -1],
        "I_mean_over_p": I_mean.mean(axis=1),
        "I_value_at_max_p": I_mean[:, -1],
        "n_jobs_used": int(n_jobs_eff),
        "method": "eigsh_topK_masked_unbiased_missingness_aware",
        "keep_diag": bool(keep_diag),
        "B": int(B),
        "q_offdiag_obs_rate": float(q),
        "ref_used_randomized": bool(has_missing_offdiag),
        "ref_oversample": int(ref_oversample),
        "ref_n_iter": int(ref_n_iter),
    }
    if collect_worker_stats:
        summary["worker_stats"] = worker_stats_all

    fig = None
    if visualize and visualize_mode == "plotly":
        fig = go.Figure()
        has_tau = (tau_kp is not None) and bool(plot_tau)
        has_act = (activation_idx is not None) and bool(mark_activation)
        traces_per_k = 1 + 2 + (1 if has_tau else 0) + (1 if has_act else 0)

        def _add_trace_mean(kk, k):
            fig.add_trace(
                go.Scatter(
                    x=p_list,
                    y=x_mean[kk],
                    mode="lines+markers",
                    name=f"Iproj k={k}" if not use_baseline_correction else f"x k={k}",
                    line=dict(width=2),
                    opacity=0.9,
                    visible=True,
                    hovertemplate=f"k={k}<br>p=%{{x:.4g}}<br>mean=%{{y:.4g}}<extra></extra>",
                )
            )

        def _add_trace_ci(kk, k):
            fig.add_trace(
                go.Scatter(
                    x=p_list,
                    y=x_ci_lo[kk],
                    mode="lines",
                    line=dict(width=1, dash="dot"),
                    name=f"CI low k={k}",
                    showlegend=False,
                    opacity=0.9,
                    visible=False,
                    hovertemplate=f"k={k}<br>p=%{{x:.4g}}<br>CI_low=%{{y:.4g}}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=p_list,
                    y=x_ci_hi[kk],
                    mode="lines",
                    line=dict(width=1, dash="dot"),
                    name=f"CI high k={k}",
                    showlegend=False,
                    opacity=0.9,
                    visible=False,
                    hovertemplate=f"k={k}<br>p=%{{x:.4g}}<br>CI_high=%{{y:.4g}}<extra></extra>",
                )
            )

        def _add_trace_tau(kk, k):
            if not has_tau:
                return
            fig.add_trace(
                go.Scatter(
                    x=p_list,
                    y=tau_kp[kk],
                    mode="lines",
                    line=dict(width=1.5, dash="dash"),
                    name=f"tau k={k}",
                    showlegend=False,
                    opacity=0.75,
                    visible=False,
                    hovertemplate=f"k={k}<br>p=%{{x:.4g}}<br>tau=%{{y:.4g}}<extra></extra>",
                )
            )

        def _add_trace_act(kk, k):
            if not has_act:
                return
            j = int(activation_idx[kk]) if activation_idx[kk] is not None else -1
            if j is None or j < 0:
                fig.add_trace(
                    go.Scatter(
                        x=[],
                        y=[],
                        mode="markers",
                        marker=dict(size=9, symbol="circle-open"),
                        name=f"act k={k}",
                        showlegend=False,
                        visible=False,
                    )
                )
                return

            ht = (
                f"k={k}"
                f"<br>p_act={p_list[j]:.4g}"
                f"<br>median={x_median[kk, j]:.4g}"
                f"<br>mean={x_mean[kk, j]:.4g}"
                f"<br>CI_low={x_ci_lo[kk, j]:.4g}"
            )
            if tau_kp is not None:
                ht += f"<br>tau={tau_kp[kk, j]:.4g}"
            ht += "<extra></extra>"

            fig.add_trace(
                go.Scatter(
                    x=[p_list[j]],
                    y=[x_median[kk, j]],
                    mode="markers",
                    marker=dict(size=10, symbol="circle-open"),
                    name=f"act k={k}",
                    showlegend=False,
                    opacity=1.0,
                    visible=False,
                    hovertemplate=ht,
                )
            )

        for kk, k in enumerate(k_list):
            _add_trace_mean(kk, k)
            _add_trace_ci(kk, k)
            _add_trace_tau(kk, k)
            _add_trace_act(kk, k)

        n_traces = len(fig.data)

        def _mask_all_false():
            return [False] * n_traces

        def _vis_means_only():
            vis = _mask_all_false()
            for kk in range(len(k_list)):
                vis[kk * traces_per_k + 0] = True
            return vis

        def _vis_means_plus_ci():
            vis = _mask_all_false()
            for kk in range(len(k_list)):
                base = kk * traces_per_k
                vis[base + 0] = True
                vis[base + 1] = True
                vis[base + 2] = True
            return vis

        def _vis_all():
            vis = _vis_means_plus_ci()
            if has_tau:
                for kk in range(len(k_list)):
                    base = kk * traces_per_k
                    vis[base + 3] = True
                if has_act:
                    for kk in range(len(k_list)):
                        base = kk * traces_per_k
                        vis[base + (traces_per_k - 1)] = True
            else:
                if has_act:
                    for kk in range(len(k_list)):
                        base = kk * traces_per_k
                        vis[base + (traces_per_k - 1)] = True
            return vis

        def _vis_focus_k(kk_focus, show_ci=True, show_tau=True, show_act=True):
            vis = _mask_all_false()
            base = kk_focus * traces_per_k
            vis[base + 0] = True
            if show_ci:
                vis[base + 1] = True
                vis[base + 2] = True
            offset = 3
            if has_tau:
                if show_tau:
                    vis[base + offset] = True
                offset += 1
            if has_act and show_act:
                vis[base + (traces_per_k - 1)] = True
            return vis

        buttons = []
        buttons.append(
            dict(
                label="Means only",
                method="update",
                args=[
                    {"visible": _vis_means_only()},
                    {"title": "Iproj trends (means only)"},
                ],
            )
        )
        buttons.append(
            dict(
                label="Means + CI",
                method="update",
                args=[
                    {"visible": _vis_means_plus_ci()},
                    {"title": f"Iproj trends with {int(ci_level * 100)}% CI (lines)"},
                ],
            )
        )
        if tau_kp is not None:
            buttons.append(
                dict(
                    label="Means + CI + tau + activation",
                    method="update",
                    args=[
                        {"visible": _vis_all()},
                        {
                            "title": f"Iproj trends with CI, tau (q={alpha_tau:.2f}), and activation points"
                        },
                    ],
                )
            )
        for kk, k in enumerate(k_list):
            buttons.append(
                dict(
                    label=f"Focus k={k}",
                    method="update",
                    args=[
                        {
                            "visible": _vis_focus_k(
                                kk, show_ci=True, show_tau=True, show_act=True
                            )
                        },
                        {"title": f"Iproj: focus view (k={k})"},
                    ],
                )
            )

        ylab = (
            "Iproj_k(p)"
            if not use_baseline_correction
            else "x_k(p) (baseline-corrected)"
        )

        fig.update_layout(
            template="plotly_white",
            width=700,
            height=700,
            title="Iproj trends (means only)",
            xaxis=dict(
                title=dict(text="p (masking probability)", standoff=18), automargin=True
            ),
            yaxis=dict(title=dict(text=ylab, standoff=14), automargin=True),
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0.0),
            margin=dict(l=70, r=25, t=80, b=160),
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    x=0.995,
                    y=0.995,
                    xanchor="right",
                    yanchor="top",
                    showactive=True,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                )
            ],
        )
        fig.show()

    diagnostics = {
        "use_baseline_correction": bool(use_baseline_correction),
        "ci_level": float(ci_level),
        "alpha_tau": float(alpha_tau),
        "x_boot": x_boot,  # (K,P,B)
        "x_mean": x_mean,  # (K,P)
        "x_median": x_median,  # (K,P)
        "x_ci_lo": x_ci_lo,  # (K,P)
        "x_ci_hi": x_ci_hi,  # (K,P)
        "tau_kp": tau_kp,  # (K,P) or None
        "activation_idx": activation_idx,
        "activation_p": activation_p,
        "null_boot": null_boot,
        # NEW diagnostics
        "q_offdiag_obs_rate": float(q),
        "ref_used_randomized": bool(has_missing_offdiag),
    }

    return {
        "p": p_list,
        "k_list": k_list,
        "C_boot": C_boot,
        "C_mean": C_mean,
        "I_boot": I_boot,
        "I_mean": I_mean,
        "Iproj_boot": Iproj_boot,
        "Iproj_mean": Iproj_mean,
        "summary": summary,
        "fig": fig,
        "evals_ref": evals_ref,
        "U_ref_K": U_ref_K,
        "lambda_topk_boot": lambda_topk_boot,
        "diagnostics": diagnostics,
    }


def test_fast_coherence(
    S,
    k_list,
    p_list,
    B=50,
    random_state=0,
    keep_diag=True,
    solver="auto",
    solver_tol=1e-10,
    solver_maxiter=200,
    residual_tol=1e-8,
    use_nested_masks=True,
    n_jobs=None,
    max_nbytes="64M",
    prefer="threads",
    show_progress=True,
    visualize=True,
    visualize_mode="plotly",
    ci_level=0.95,
):
    """
    Compute exact bootstrap ``Iproj`` curves with warm starts and p-continuation.

    This routine is intended as a faster alternative when downstream analysis only
    needs ``Iproj_boot`` (for example ``analyze_iproj_signal_alignment_v3``).

    Speedups used here:
    1. Compute the reference top-``Kmax`` eigenspace of ``S`` once.
    2. For each bootstrap replicate, sweep ``p_list`` in increasing order and warm
       start each eigensolve from the previous one.
    3. Reuse one uniform edge mask across the p-sweep to form a nested Bernoulli
       path. For each fixed ``p`` the marginal bootstrap law remains exact; only
       the dependence across different ``p`` values changes.

    Returns a compact ``result_dict`` with the keys needed by downstream ``Iproj``
    analyses: ``p``, ``k_list``, ``Iproj_boot``, ``Iproj_mean``, ``evals_ref``,
    ``U_ref_K``, ``summary``, ``fig`` (always ``None``), and ``diagnostics``.
    """
    S_sym = _symmetrize_with_nan(S)
    n = S_sym.shape[0]

    k_list = np.asarray(k_list, int)
    if k_list.ndim != 1 or k_list.size == 0:
        raise ValueError("k_list must be a non-empty 1D array.")
    if np.any(k_list <= 0) or np.any(k_list > n):
        raise ValueError("All k in k_list must be in {1, ..., n}.")
    if np.any(np.diff(k_list) <= 0):
        raise ValueError("k_list must be strictly increasing.")

    p_list = np.asarray(p_list, float)
    if p_list.ndim != 1 or p_list.size == 0:
        raise ValueError("p_list must be a non-empty 1D array.")
    if np.any((p_list <= 0.0) | (p_list > 1.0)):
        raise ValueError("All p in p_list must be in (0, 1].")
    if np.any(np.diff(p_list) < 0.0):
        raise ValueError("p_list must be sorted increasing.")

    B = int(B)
    if B <= 0:
        raise ValueError("B must be positive.")

    Kmax = int(k_list.max())
    K_count = int(len(k_list))
    P = int(len(p_list))
    k_idx = k_list - 1

    cpu = os.cpu_count() or 2
    if n_jobs is None:
        n_jobs_eff = max(1, cpu - 1)
    else:
        n_jobs_eff = int(max(1, min(int(n_jobs), max(1, cpu - 1))))

    S0, W, q = _prepare_observation_mask(S_sym, keep_diag=keep_diag)
    has_missing_offdiag = q < (1.0 - 1e-15)

    if has_missing_offdiag:
        S_ref = W * S0
        if keep_diag:
            np.fill_diagonal(S_ref, np.diag(S0))
        S_ref = 0.5 * (S_ref + S_ref.T)
    else:
        S_ref = 0.5 * (S0 + S0.T)

    evals_all, evecs_all = la.eigh(S_ref)
    idx_ref = np.argsort(evals_all)[::-1][:Kmax]
    evals_ref = np.asarray(evals_all[idx_ref], float)
    U_ref_K = _orthonormalize_columns(np.asarray(evecs_all[:, idx_ref], float))

    Iproj_boot = np.zeros((K_count, P, B), float)

    tasks = [
        (
            b,
            p_list,
            S0,
            W,
            U_ref_K,
            k_idx,
            int(random_state or 0),
            bool(keep_diag),
            str(solver),
            float(solver_tol),
            int(solver_maxiter),
            float(residual_tol),
            bool(use_nested_masks),
            False,
        )
        for b in range(B)
    ]

    backend = "threading" if str(prefer).lower().startswith("thread") else "loky"

    if n_jobs_eff == 1:
        if show_progress:
            results = []
            for t in tqdm(
                tasks, total=B, desc="fast exact Iproj bootstrap", leave=True
            ):
                tt = list(t)
                tt[-1] = True
                results.append(_fast_iproj_worker_one_boot(tuple(tt)))
        else:
            results = [_fast_iproj_worker_one_boot(t) for t in tasks]
    elif show_progress:
        pbar = tqdm(
            total=B,
            desc=f"fast exact Iproj bootstrap (n_jobs={n_jobs_eff})",
            leave=True,
        )

        class _TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):
                pbar.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)

        @contextlib.contextmanager
        def _cm():
            old_cb = joblib.parallel.BatchCompletionCallBack
            joblib.parallel.BatchCompletionCallBack = _TqdmBatchCompletionCallback
            try:
                yield pbar
            finally:
                joblib.parallel.BatchCompletionCallBack = old_cb
                pbar.close()

        with _cm():
            results = Parallel(
                n_jobs=n_jobs_eff,
                backend=backend,
                prefer=prefer,
                max_nbytes=max_nbytes,
            )(delayed(_fast_iproj_worker_one_boot)(t) for t in tasks)
    else:
        results = Parallel(
            n_jobs=n_jobs_eff,
            backend=backend,
            prefer=prefer,
            max_nbytes=max_nbytes,
        )(delayed(_fast_iproj_worker_one_boot)(t) for t in tasks)

    results.sort(key=lambda x: x[0])

    solver_counts_total = {}
    fallback_count_total = 0
    warning_count_total = 0
    max_residual_seen = 0.0
    worker_stats = []

    for b, Iproj_b, stats_b in results:
        Iproj_boot[:, :, b] = Iproj_b
        worker_stats.append(stats_b)
        for name, count in stats_b["solver_counts"].items():
            solver_counts_total[name] = solver_counts_total.get(name, 0) + int(count)
        fallback_count_total += int(stats_b["fallback_count"])
        warning_count_total += int(stats_b.get("warning_count", 0))
        max_residual_seen = max(max_residual_seen, float(stats_b["max_residual_seen"]))

    Iproj_mean = Iproj_boot.mean(axis=2)
    ci_lo_q = (1.0 - float(ci_level)) / 2.0
    ci_hi_q = 1.0 - ci_lo_q
    Iproj_ci_lo = np.quantile(Iproj_boot, ci_lo_q, axis=2)
    Iproj_ci_hi = np.quantile(Iproj_boot, ci_hi_q, axis=2)

    summary = {
        "k": k_list.copy(),
        "Iproj_mean_over_p": Iproj_mean.mean(axis=1),
        "Iproj_value_at_max_p": Iproj_mean[:, -1],
        "n_jobs_used": int(n_jobs_eff),
        "method": "exact_warmstart_continuation_iproj_only",
        "keep_diag": bool(keep_diag),
        "B": int(B),
        "q_offdiag_obs_rate": float(q),
        "ref_used_observed_only": bool(has_missing_offdiag),
        "use_nested_masks": bool(use_nested_masks),
        "solver": str(solver),
        "solver_tol": float(solver_tol),
        "solver_maxiter": int(solver_maxiter),
        "residual_tol": float(residual_tol),
        "solver_counts": solver_counts_total,
        "fallback_count": int(fallback_count_total),
        "warning_count": int(warning_count_total),
        "max_residual_seen": float(max_residual_seen),
        "worker_stats": worker_stats,
    }

    diagnostics = {
        "q_offdiag_obs_rate": float(q),
        "use_nested_masks": bool(use_nested_masks),
        "solver_counts": solver_counts_total,
        "fallback_count": int(fallback_count_total),
        "warning_count": int(warning_count_total),
        "max_residual_seen": float(max_residual_seen),
        "Iproj_ci_lo": Iproj_ci_lo,
        "Iproj_ci_hi": Iproj_ci_hi,
    }

    fig = None
    if visualize and visualize_mode == "plotly":
        fig = go.Figure()
        for kk, k in enumerate(k_list):
            fig.add_trace(
                go.Scatter(
                    x=p_list,
                    y=Iproj_mean[kk],
                    mode="lines+markers",
                    name=f"Iproj k={k}",
                    line=dict(width=2),
                    hovertemplate=f"k={k}<br>p=%{{x:.4g}}<br>mean=%{{y:.4g}}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=p_list,
                    y=Iproj_ci_lo[kk],
                    mode="lines",
                    line=dict(width=1, dash="dot"),
                    name=f"CI low k={k}",
                    showlegend=False,
                    visible=False,
                    hovertemplate=f"k={k}<br>p=%{{x:.4g}}<br>CI_low=%{{y:.4g}}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=p_list,
                    y=Iproj_ci_hi[kk],
                    mode="lines",
                    line=dict(width=1, dash="dot"),
                    name=f"CI high k={k}",
                    showlegend=False,
                    visible=False,
                    hovertemplate=f"k={k}<br>p=%{{x:.4g}}<br>CI_high=%{{y:.4g}}<extra></extra>",
                )
            )

        n_traces = len(fig.data)

        def _mask_all_false():
            return [False] * n_traces

        def _vis_means_only():
            vis = _mask_all_false()
            for kk in range(len(k_list)):
                vis[3 * kk + 0] = True
            return vis

        def _vis_means_plus_ci():
            vis = _mask_all_false()
            for kk in range(len(k_list)):
                base = 3 * kk
                vis[base + 0] = True
                vis[base + 1] = True
                vis[base + 2] = True
            return vis

        buttons = [
            dict(
                label="Means only",
                method="update",
                args=[
                    {"visible": _vis_means_only()},
                    {"title": "Fast exact Iproj trends (means only)"},
                ],
            ),
            dict(
                label="Means + CI",
                method="update",
                args=[
                    {"visible": _vis_means_plus_ci()},
                    {
                        "title": f"Fast exact Iproj trends with {int(ci_level * 100)}% CI"
                    },
                ],
            ),
        ]

        fig.update_layout(
            template="plotly_white",
            width=700,
            height=700,
            title="Fast exact Iproj trends (means only)",
            xaxis=dict(
                title=dict(text="p (masking probability)", standoff=18), automargin=True
            ),
            yaxis=dict(title=dict(text="Iproj_k(p)", standoff=14), automargin=True),
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0.0),
            margin=dict(l=70, r=25, t=80, b=160),
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    x=0.995,
                    y=0.995,
                    xanchor="right",
                    yanchor="top",
                    showactive=True,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                )
            ],
        )
        fig.show()

    return {
        "p": p_list,
        "k_list": k_list,
        "Iproj_boot": Iproj_boot,
        "Iproj_mean": Iproj_mean,
        "summary": summary,
        "fig": fig,
        "evals_ref": evals_ref,
        "U_ref_K": U_ref_K,
        "diagnostics": diagnostics,
    }


def _rankdata_average_1d(x):
    """Average-rank transform for one 1D array (ties receive average rank)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return x.copy()

    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks_sorted = np.empty(n, dtype=float)

    start = 0
    while start < n:
        end = start + 1
        while end < n and xs[end] == xs[start]:
            end += 1
        avg_rank = 0.5 * ((start + 1) + end)  # 1-based average rank
        ranks_sorted[start:end] = avg_rank
        start = end

    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def _pearson_corr_matrix(X, eps=1e-12):
    """Vectorized Pearson correlation matrix for rows of X (shape K x P)."""
    X = np.asarray(X, dtype=float)
    K, P = X.shape
    if P < 2:
        return np.eye(K, dtype=float)

    Xc = X - X.mean(axis=1, keepdims=True)
    norm = np.sqrt(np.sum(Xc * Xc, axis=1))
    denom = np.outer(norm, norm)

    C = Xc @ Xc.T
    out = np.zeros((K, K), dtype=float)
    valid = denom > eps
    out[valid] = C[valid] / denom[valid]
    np.fill_diagonal(out, 1.0)
    np.clip(out, -1.0, 1.0, out=out)
    return out


def _spearman_corr_matrix(X):
    """Spearman correlation via rank transform followed by Pearson."""
    X = np.asarray(X, dtype=float)
    R = np.vstack([_rankdata_average_1d(row) for row in X])
    return _pearson_corr_matrix(R)


def _mutual_info_matrix(X, n_bins=10, eps=1e-12):
    """
    Symmetric normalized mutual information matrix for rows of X.

    Uses discrete binning per row, then:
      NMI(i,j) = MI(i,j) / sqrt(H(i) H(j))
    with diagonal fixed to 1.
    """
    X = np.asarray(X, dtype=float)
    K, P = X.shape
    if P < 2:
        return np.eye(K, dtype=float)

    n_bins = int(max(2, n_bins))
    D = np.zeros((K, P), dtype=np.int32)
    H = np.zeros(K, dtype=float)
    valid_row = np.zeros(K, dtype=bool)

    for i in range(K):
        xi = X[i]
        if np.max(xi) - np.min(xi) < eps:
            continue
        edges = np.quantile(xi, np.linspace(0.0, 1.0, n_bins + 1))
        edges = np.maximum.accumulate(edges)
        if edges[-1] - edges[0] < eps:
            continue
        di = np.searchsorted(edges[1:-1], xi, side="right").astype(np.int32)
        D[i] = di
        valid_row[i] = True

        cnt = np.bincount(di, minlength=n_bins).astype(float)
        p = cnt / cnt.sum()
        nz = p > 0
        H[i] = -np.sum(p[nz] * np.log(p[nz] + eps))

    M = np.zeros((K, K), dtype=float)
    np.fill_diagonal(M, 1.0)

    for i in range(K):
        if not valid_row[i]:
            continue
        bi = D[i]
        for j in range(i + 1, K):
            if not valid_row[j]:
                continue
            bj = D[j]
            joint = (
                np.bincount(
                    (bi * n_bins + bj).astype(np.int64),
                    minlength=n_bins * n_bins,
                )
                .reshape(n_bins, n_bins)
                .astype(float)
            )
            pxy = joint / np.sum(joint)
            px = np.sum(pxy, axis=1, keepdims=True)
            py = np.sum(pxy, axis=0, keepdims=True)
            nz = pxy > 0
            mi = np.sum(pxy[nz] * np.log((pxy[nz] + eps) / ((px @ py)[nz] + eps)))
            denom = np.sqrt(max(H[i] * H[j], eps))
            val = float(mi / denom) if denom > eps else 0.0
            M[i, j] = val
            M[j, i] = val

    return M


def _compute_all_trend_similarity_matrices(x_trend, mi_bins=10):
    """Return Pearson, Spearman, and MI similarity matrices for trend rows."""
    pearson = _pearson_corr_matrix(x_trend)
    spearman = _spearman_corr_matrix(x_trend)
    mutual_info = _mutual_info_matrix(x_trend, n_bins=mi_bins)
    return {
        "pearson": pearson,
        "spearman": spearman,
        "mutual_info": mutual_info,
    }


def _baseline_correct_array(arr_kpb, k_list, n, clip=True):
    """
    Baseline correction with leading-k broadcasting:
      (arr - k/n) / (1 - k/n)
    """
    arr = np.asarray(arr_kpb, dtype=float)
    k = np.asarray(k_list, dtype=float)
    reshape = (k.size,) + (1,) * (arr.ndim - 1)
    k_over_n = (k / float(n)).reshape(reshape)
    denom = np.maximum(1.0 - k_over_n, 1e-12)

    out = (arr - k_over_n) / denom
    if clip:
        np.clip(out, 0.0, 1.0, out=out)
    return out


def compute_trend_correlation(
    Iproj_boot,
    k_list,
    p_list,
    use_baseline_correction=False,
    n=None,
    clip_baseline=True,
    aggregation="median",
    mi_bins=10,
):
    """
    Compute row-wise trend similarity across k using three metrics:
    Pearson, Spearman, and normalized mutual information.
    """
    K_count, P, _ = Iproj_boot.shape
    k_list = np.asarray(k_list)
    p_list = np.asarray(p_list)

    if use_baseline_correction:
        if n is None:
            raise ValueError("n must be provided when use_baseline_correction=True")
        x_boot = _baseline_correct_array(Iproj_boot, k_list, n, clip=clip_baseline)
    else:
        x_boot = np.clip(Iproj_boot, 0.0, 1.0)

    if aggregation == "mean":
        x_trend = x_boot.mean(axis=2)
    elif aggregation == "median":
        x_trend = np.median(x_boot, axis=2)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    corr_matrices = _compute_all_trend_similarity_matrices(x_trend, mi_bins=mi_bins)

    return {
        "corr_matrices": corr_matrices,
        "rho_matrix": corr_matrices["pearson"],  # backward-compatible alias
        "x_trend": x_trend,
        "x_boot": x_boot,
        "k_list": k_list,
        "p_list": p_list,
        "use_baseline_correction": bool(use_baseline_correction),
        "aggregation": aggregation,
        "mi_bins": int(mi_bins),
    }


def compute_cumulative_trend_correlation(
    Iproj_boot,
    k_list,
    p_list,
    use_baseline_correction=False,
    n=None,
    clip_baseline=True,
    aggregation="mean",
    mi_bins=10,
):
    """
    Compute trend similarity up to each p index (prefix windows).

    Returns full trajectories for Pearson/Spearman/MI matrices.
    """
    K_count, P, _ = Iproj_boot.shape
    k_list = np.asarray(k_list)
    p_list = np.asarray(p_list)

    if use_baseline_correction:
        if n is None:
            raise ValueError("n must be provided when use_baseline_correction=True")
        x_boot = _baseline_correct_array(Iproj_boot, k_list, n, clip=clip_baseline)
    else:
        x_boot = np.clip(Iproj_boot, 0.0, 1.0)

    if aggregation == "mean":
        x_trend = x_boot.mean(axis=2)
    elif aggregation == "median":
        x_trend = np.median(x_boot, axis=2)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    corr_matrices = {
        "pearson": np.zeros((P, K_count, K_count), dtype=float),
        "spearman": np.zeros((P, K_count, K_count), dtype=float),
        "mutual_info": np.zeros((P, K_count, K_count), dtype=float),
    }

    for p_idx in range(P):
        if p_idx < 1:
            corr_matrices["pearson"][p_idx] = np.eye(K_count, dtype=float)
            corr_matrices["spearman"][p_idx] = np.eye(K_count, dtype=float)
            corr_matrices["mutual_info"][p_idx] = np.eye(K_count, dtype=float)
            continue

        x_prefix = x_trend[:, : p_idx + 1]
        mats = _compute_all_trend_similarity_matrices(x_prefix, mi_bins=mi_bins)
        corr_matrices["pearson"][p_idx] = mats["pearson"]
        corr_matrices["spearman"][p_idx] = mats["spearman"]
        corr_matrices["mutual_info"][p_idx] = mats["mutual_info"]

    return {
        "corr_matrices": corr_matrices,
        "rho_matrices": corr_matrices["pearson"],  # backward-compatible alias
        "x_trend": x_trend,
        "x_boot": x_boot,
        "p_list": p_list,
        "k_list": k_list,
        "use_baseline_correction": bool(use_baseline_correction),
        "aggregation": aggregation,
        "mi_bins": int(mi_bins),
    }


def _pearson_corr(a, b, eps=1e-12):
    """Pearson correlation for two 1D arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ac = a - a.mean()
    bc = b - b.mean()
    sa = np.sqrt(np.mean(ac * ac))
    sb = np.sqrt(np.mean(bc * bc))
    if sa < eps or sb < eps:
        return 0.0
    return float(np.mean((ac / sa) * (bc / sb)))


def _spearman_corr(a, b, eps=1e-12):
    """Spearman correlation for two 1D arrays."""
    return _pearson_corr(_rankdata_average_1d(a), _rankdata_average_1d(b), eps=eps)


def _fd_bins(x):
    """Freedman-Diaconis bin rule with robust clipping."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 2
    q25, q75 = np.quantile(x, [0.25, 0.75])
    iqr = q75 - q25
    if iqr <= 0:
        return 5
    bw = 2.0 * iqr * (x.size ** (-1.0 / 3.0))
    if bw <= 0:
        return 5
    nb = int(np.ceil((x.max() - x.min()) / bw))
    return int(np.clip(nb, 2, 50))


def _mutual_info_discrete(a, b, bins="fd", normalize="sqrt", eps=1e-12):
    """Discrete mutual information between two 1D arrays with optional normalization."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.size != b.size:
        raise ValueError("a and b must have the same length.")
    if a.size < 2:
        return 0.0

    if isinstance(bins, str) and bins.lower() == "fd":
        nb_a = _fd_bins(a)
        nb_b = _fd_bins(b)
    else:
        nb_a = nb_b = int(max(2, bins))

    a_edges = np.linspace(a.min(), a.max() + eps, nb_a + 1)
    b_edges = np.linspace(b.min(), b.max() + eps, nb_b + 1)
    ai = np.clip(np.digitize(a, a_edges) - 1, 0, nb_a - 1)
    bi = np.clip(np.digitize(b, b_edges) - 1, 0, nb_b - 1)

    joint = np.zeros((nb_a, nb_b), dtype=float)
    np.add.at(joint, (ai, bi), 1.0)
    joint /= float(a.size)

    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    papb = pa @ pb
    mask = joint > 0
    mi = float(np.sum(joint[mask] * np.log((joint[mask] + eps) / (papb[mask] + eps))))

    if normalize is None:
        return mi

    def _entropy(p):
        m = p > 0
        return float(-np.sum(p[m] * np.log(p[m] + eps)))

    Ha = _entropy(pa[:, 0])
    Hb = _entropy(pb[0, :])

    if normalize == "sqrt":
        denom = np.sqrt(max(Ha * Hb, eps))
    elif normalize == "min":
        denom = max(min(Ha, Hb), eps)
    else:
        raise ValueError("mi_normalize must be 'sqrt', 'min', or None")

    return float(np.clip(mi / denom, 0.0, 1.0))


def _mutual_info_matrix_pairwise(X, bins="fd", normalize="sqrt", eps=1e-12):
    """Pairwise mutual-information similarity matrix for rows of X."""
    X = np.asarray(X, dtype=float)
    K = X.shape[0]
    M = np.eye(K, dtype=float)
    for i in range(K):
        for j in range(i + 1, K):
            val = _mutual_info_discrete(
                X[i], X[j], bins=bins, normalize=normalize, eps=eps
            )
            M[i, j] = val
            M[j, i] = val
    return M


def _get_pair_metric(metric, mi_bins="fd", mi_normalize="sqrt", eps=1e-12):
    """Return a pairwise metric callable and expected value range."""
    if callable(metric):
        return metric, (None, None)

    m = str(metric).lower()
    if m in ("pearson", "corr", "correlation"):
        return (lambda a, b: _pearson_corr(a, b, eps=eps)), (-1.0, 1.0)
    if m in ("spearman", "rank", "rankcorr"):
        return (lambda a, b: _spearman_corr(a, b, eps=eps)), (-1.0, 1.0)
    if m in ("mi", "mutual_info", "mutualinformation", "mutual-information"):

        def fn(a, b):
            return _mutual_info_discrete(
                a, b, bins=mi_bins, normalize=mi_normalize, eps=eps
            )

        return fn, (0.0, 1.0) if mi_normalize is not None else (None, None)
    raise ValueError(
        "metric must be 'pearson', 'spearman', 'mi', or a callable(a,b)->float"
    )


def analyze_cluster_consensus_across_p(
    result_dict,
    use_baseline_correction=True,
    n=None,
    clip_baseline=True,
    aggregation="median",
    metric="spearman",
    mi_bins="fd",
    mi_normalize="sqrt",
    eps=1e-12,
    sim_threshold=0.85,
    k_window=1,
    min_cluster_size=1,
    cluster_on_diff=False,
    use_abs_for_corr=True,
    min_prefix_points=5,
    prefix_stride=1,
    persistence_level=0.8,
    require_consecutive=3,
    make_plots=True,
    width=1250,
    height=820,
):
    """
    Analyze contiguous-k cluster stability over p-prefixes from coherence output.

    Expects output dictionary from `compute_incremental_coherence_multi_k_eig_anisotropic`.
    """
    if make_plots:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except Exception as e:
            raise ImportError(
                "plotly is required when make_plots=True. pip install plotly"
            ) from e

    required = ("Iproj_boot", "k_list", "p", "U_ref_K")
    missing = [k for k in required if k not in result_dict]
    if missing:
        raise KeyError(f"result_dict missing keys: {missing}")

    Iproj_boot = result_dict["Iproj_boot"]
    k_list = np.asarray(result_dict["k_list"])
    p_list = np.asarray(result_dict["p"])

    if n is None:
        n = int(result_dict["U_ref_K"].shape[0])

    K, P, _ = Iproj_boot.shape
    if k_list.shape[0] != K:
        raise ValueError("k_list length does not match Iproj_boot first dimension.")
    if p_list.shape[0] != P:
        raise ValueError("p length does not match Iproj_boot second dimension.")

    if use_baseline_correction:
        x_boot = _baseline_correct_array(Iproj_boot, k_list, n, clip=clip_baseline)
    else:
        x_boot = np.clip(Iproj_boot, 0.0, 1.0)

    if aggregation == "mean":
        x_trend = x_boot.mean(axis=2)
    elif aggregation == "median":
        x_trend = np.median(x_boot, axis=2)
    else:
        raise ValueError("aggregation must be 'mean' or 'median'")

    pair_metric, metric_range = _get_pair_metric(
        metric=metric,
        mi_bins=mi_bins,
        mi_normalize=mi_normalize,
        eps=eps,
    )
    zmin, zmax = metric_range

    metric_is_callable = callable(metric)
    metric_name = None if metric_is_callable else str(metric).lower()
    dk = np.abs(k_list[:, None] - k_list[None, :])

    def _similarity_matrix_rows(X_kp):
        if metric_is_callable:
            S = np.eye(K, dtype=float)
            row_std = np.std(X_kp, axis=1)
            for i in range(K):
                ai = X_kp[i]
                si = row_std[i]
                for j in range(i + 1, K):
                    bj = X_kp[j]
                    sj = row_std[j]
                    if si < eps or sj < eps:
                        val = 0.0 if (zmin is None or zmin < 0.0) else zmin
                    else:
                        val = float(pair_metric(ai, bj))
                    S[i, j] = val
                    S[j, i] = val
            if zmin is not None and zmax is not None:
                np.clip(S, zmin, zmax, out=S)
            return S

        if metric_name in ("pearson", "corr", "correlation"):
            return _pearson_corr_matrix(X_kp, eps=eps)
        if metric_name in ("spearman", "rank", "rankcorr"):
            return _spearman_corr_matrix(X_kp)
        if metric_name in (
            "mi",
            "mutual_info",
            "mutualinformation",
            "mutual-information",
        ):
            return _mutual_info_matrix_pairwise(
                X_kp, bins=mi_bins, normalize=mi_normalize, eps=eps
            )

        S = np.eye(K, dtype=float)
        row_std = np.std(X_kp, axis=1)
        for i in range(K):
            ai = X_kp[i]
            si = row_std[i]
            for j in range(i + 1, K):
                bj = X_kp[j]
                sj = row_std[j]
                if si < eps or sj < eps:
                    val = 0.0 if (zmin is None or zmin < 0.0) else zmin
                else:
                    val = float(pair_metric(ai, bj))
                S[i, j] = val
                S[j, i] = val
        if zmin is not None and zmax is not None:
            np.clip(S, zmin, zmax, out=S)
        return S

    def _cluster_from_similarity(S):
        if metric_is_callable:
            adj_base = np.abs(S) if use_abs_for_corr else S
        elif metric_name in (
            "pearson",
            "spearman",
            "corr",
            "correlation",
            "rank",
            "rankcorr",
        ):
            adj_base = np.abs(S) if use_abs_for_corr else S
        else:
            adj_base = S

        adj = (adj_base >= float(sim_threshold)) & (dk <= int(k_window))
        np.fill_diagonal(adj, True)

        seen = np.zeros(K, dtype=bool)
        clusters = []
        for i in range(K):
            if seen[i]:
                continue
            stack = [i]
            seen[i] = True
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                neighbors = np.where(adj[u])[0]
                for v in neighbors:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            comp = np.array(sorted(comp), dtype=int)
            if comp.size >= int(min_cluster_size):
                clusters.append(comp)

        if not clusters:
            clusters = [np.arange(K, dtype=int)]

        mk = np.array([np.mean(k_list[c]) for c in clusters], dtype=float)
        order = np.argsort(mk)
        clusters = [clusters[i] for i in order]

        labels = -np.ones(K, dtype=int)
        for cid, c in enumerate(clusters):
            labels[c] = cid
        return clusters, labels

    prefix_ts = list(range(max(int(min_prefix_points), 2), P + 1, int(prefix_stride)))
    if len(prefix_ts) == 0:
        prefix_ts = [P]

    consensus = np.zeros((K, K), dtype=float)
    labels_over_prefix = []
    eval_prefix_ts = []

    for t in prefix_ts:
        Xpref = x_trend[:, :t]
        if cluster_on_diff:
            Xpref = np.diff(Xpref, axis=1)
            if Xpref.shape[1] < 2:
                continue

        S = _similarity_matrix_rows(Xpref)
        _, labels_t = _cluster_from_similarity(S)
        labels_over_prefix.append(labels_t)
        eval_prefix_ts.append(t)
        consensus += (labels_t[:, None] == labels_t[None, :]).astype(float)

    denom = float(max(len(labels_over_prefix), 1))
    consensus /= denom
    np.fill_diagonal(consensus, 1.0)

    Xfull = x_trend if not cluster_on_diff else np.diff(x_trend, axis=1)
    Sfull = _similarity_matrix_rows(Xfull)
    ref_clusters, ref_labels = _cluster_from_similarity(Sfull)
    C = len(ref_clusters)

    persistence_curves = np.full((C, P), np.nan, dtype=float)

    def _jaccard(a, b):
        a = set(map(int, a))
        b = set(map(int, b))
        inter = len(a & b)
        union = len(a | b)
        return 0.0 if union == 0 else inter / union

    def _clusters_from_labels(labels):
        labs = np.unique(labels)
        labs = labs[labs >= 0]
        return [np.where(labels == lab)[0] for lab in labs]

    for idx, t in enumerate(eval_prefix_ts):
        labels_t = labels_over_prefix[idx]
        cls_t = _clusters_from_labels(labels_t)
        col = t - 1
        for ci, cref in enumerate(ref_clusters):
            best = 0.0
            for g in cls_t:
                best = max(best, _jaccard(cref, g))
            persistence_curves[ci, col] = best

    for ci in range(C):
        last = 0.0
        for j in range(P):
            if np.isnan(persistence_curves[ci, j]):
                persistence_curves[ci, j] = last
            else:
                last = persistence_curves[ci, j]

    L = max(1, int(require_consecutive))
    emergence_idx = -np.ones(C, dtype=int)
    for ci in range(C):
        ok = persistence_curves[ci] >= float(persistence_level)
        for j in range(0, P - L + 1):
            if np.all(ok[j : j + L]):
                emergence_idx[ci] = j
                break
    emergence_p = np.array(
        [p_list[j] if j >= 0 else np.nan for j in emergence_idx], dtype=float
    )

    fig = None
    if make_plots:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                f"Consensus co-clustering across p (metric={metric})",
                "Reference clusters at full p (membership by k)",
                "Persistence of each reference cluster across p (best Jaccard match)",
                "Emergence p of each reference cluster",
            ),
            horizontal_spacing=0.12,
            vertical_spacing=0.14,
        )

        fig.add_trace(
            go.Heatmap(
                z=consensus,
                x=k_list,
                y=k_list,
                zmin=0.0,
                zmax=1.0,
                colorbar=dict(title="P(same cluster)"),
                hovertemplate="k_i=%{y}<br>k_j=%{x}<br>consensus=%{z:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.update_xaxes(title_text="k_j", row=1, col=1)
        fig.update_yaxes(title_text="k_i", row=1, col=1)

        fig.add_trace(
            go.Heatmap(
                z=ref_labels.reshape(1, -1),
                x=k_list,
                y=["ref cluster"],
                zmin=-1,
                zmax=max(C - 1, 0),
                colorbar=dict(title="ref id"),
                hovertemplate="k=%{x}<br>ref_cluster=%{z}<extra></extra>",
                showscale=True,
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="k", row=1, col=2)
        fig.update_yaxes(title_text="", row=1, col=2)

        for ci in range(C):
            fig.add_trace(
                go.Scatter(
                    x=p_list,
                    y=persistence_curves[ci],
                    mode="lines",
                    name=f"ref cluster {ci} (|k|={len(ref_clusters[ci])})",
                    showlegend=(ci < min(C, 12)),
                ),
                row=2,
                col=1,
            )
        fig.add_hline(
            y=float(persistence_level),
            line_dash="dash",
            annotation_text=f"level={persistence_level}",
            row=2,
            col=1,
        )
        fig.update_xaxes(title_text="p", row=2, col=1)
        fig.update_yaxes(
            title_text="best Jaccard(ref, clusters(p))",
            row=2,
            col=1,
            range=[-0.02, 1.02],
        )

        fig.add_trace(
            go.Scatter(
                x=emergence_p,
                y=[f"ref {ci}" for ci in range(C)],
                mode="markers",
                marker=dict(size=10),
                hovertemplate="cluster=%{y}<br>emergence p=%{x:.4g}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        fig.update_xaxes(title_text="emergence p", row=2, col=2)
        fig.update_yaxes(title_text="reference cluster", row=2, col=2)

        fig.update_layout(
            width=width,
            height=height,
            template="plotly_white",
            title=(
                "Consensus of Clusters Across p "
                f"(threshold={sim_threshold}, k_window={k_window}, metric={metric})"
            ),
        )

    return {
        "consensus_matrix": consensus,
        "ref_clusters": ref_clusters,
        "ref_labels": ref_labels,
        "persistence_curves": persistence_curves,
        "emergence_idx": emergence_idx,
        "emergence_p": emergence_p,
        "prefix_ts": prefix_ts,
        "evaluated_prefix_ts": eval_prefix_ts,
        "fig_consensus": fig,
        "settings": dict(
            metric=metric,
            sim_threshold=float(sim_threshold),
            k_window=int(k_window),
            min_cluster_size=int(min_cluster_size),
            cluster_on_diff=bool(cluster_on_diff),
            min_prefix_points=int(min_prefix_points),
            prefix_stride=int(prefix_stride),
            persistence_level=float(persistence_level),
            require_consecutive=int(require_consecutive),
            use_baseline_correction=bool(use_baseline_correction),
            aggregation=str(aggregation),
        ),
    }


def plot_consensus_from_p_slider(
    result_dict,
    use_baseline_correction=True,
    n=None,
    clip_baseline=True,
    aggregation="median",
    metric="spearman",
    mi_bins="fd",
    mi_normalize="sqrt",
    eps=1e-12,
    sim_threshold=0.85,
    k_window=1,
    min_cluster_size=1,
    cluster_on_diff=False,
    use_abs_for_corr=True,
    min_prefix_points=5,
    prefix_stride=1,
    width=900,
    height=800,
    make_plot=True,
):
    """
    Build a cumulative-from-start consensus tensor and optional slider figure.

    For each end index e in [1, ..., P-1], compute consensus clustering over
    windows X[:, :t] for t from 2..(e+1) (controlled by min_prefix_points and
    prefix_stride). The first matrix is therefore based on p0..p1.
    """
    if make_plot:
        try:
            import plotly.graph_objects as go
        except Exception as e:
            raise ImportError(
                "plotly is required when make_plot=True. pip install plotly"
            ) from e

    required = ("Iproj_boot", "k_list", "p", "U_ref_K")
    missing = [k for k in required if k not in result_dict]
    if missing:
        raise KeyError(f"result_dict missing keys: {missing}")

    Iproj_boot = result_dict["Iproj_boot"]
    k_list = np.asarray(result_dict["k_list"])
    p_list = np.asarray(result_dict["p"])

    if n is None:
        n = int(result_dict["U_ref_K"].shape[0])

    K, P, _ = Iproj_boot.shape
    if k_list.shape[0] != K:
        raise ValueError("k_list length does not match Iproj_boot first dimension.")
    if p_list.shape[0] != P:
        raise ValueError("p length does not match Iproj_boot second dimension.")

    if use_baseline_correction:
        x_boot = _baseline_correct_array(Iproj_boot, k_list, n, clip=clip_baseline)
    else:
        x_boot = np.clip(Iproj_boot, 0.0, 1.0)

    if aggregation == "mean":
        x_trend = x_boot.mean(axis=2)
    elif aggregation == "median":
        x_trend = np.median(x_boot, axis=2)
    else:
        raise ValueError("aggregation must be 'mean' or 'median'")

    pair_metric, metric_range = _get_pair_metric(
        metric=metric,
        mi_bins=mi_bins,
        mi_normalize=mi_normalize,
        eps=eps,
    )
    zmin, zmax = metric_range

    metric_is_callable = callable(metric)
    metric_name = None if metric_is_callable else str(metric).lower()
    dk = np.abs(k_list[:, None] - k_list[None, :])

    def _similarity_matrix_rows(X_kp):
        if metric_is_callable:
            S = np.eye(K, dtype=float)
            row_std = np.std(X_kp, axis=1)
            for i in range(K):
                ai = X_kp[i]
                si = row_std[i]
                for j in range(i + 1, K):
                    bj = X_kp[j]
                    sj = row_std[j]
                    if si < eps or sj < eps:
                        val = 0.0 if (zmin is None or zmin < 0.0) else zmin
                    else:
                        val = float(pair_metric(ai, bj))
                    S[i, j] = val
                    S[j, i] = val
            if zmin is not None and zmax is not None:
                np.clip(S, zmin, zmax, out=S)
            return S

        if metric_name in ("pearson", "corr", "correlation"):
            return _pearson_corr_matrix(X_kp, eps=eps)
        if metric_name in ("spearman", "rank", "rankcorr"):
            return _spearman_corr_matrix(X_kp)
        if metric_name in (
            "mi",
            "mutual_info",
            "mutualinformation",
            "mutual-information",
        ):
            return _mutual_info_matrix_pairwise(
                X_kp, bins=mi_bins, normalize=mi_normalize, eps=eps
            )

        S = np.eye(K, dtype=float)
        row_std = np.std(X_kp, axis=1)
        for i in range(K):
            ai = X_kp[i]
            si = row_std[i]
            for j in range(i + 1, K):
                bj = X_kp[j]
                sj = row_std[j]
                if si < eps or sj < eps:
                    val = 0.0 if (zmin is None or zmin < 0.0) else zmin
                else:
                    val = float(pair_metric(ai, bj))
                S[i, j] = val
                S[j, i] = val
        if zmin is not None and zmax is not None:
            np.clip(S, zmin, zmax, out=S)
        return S

    def _cluster_labels_from_similarity(S):
        if metric_is_callable:
            adj_base = np.abs(S) if use_abs_for_corr else S
        elif metric_name in (
            "pearson",
            "spearman",
            "corr",
            "correlation",
            "rank",
            "rankcorr",
        ):
            adj_base = np.abs(S) if use_abs_for_corr else S
        else:
            adj_base = S

        adj = (adj_base >= float(sim_threshold)) & (dk <= int(k_window))
        np.fill_diagonal(adj, True)

        seen = np.zeros(K, dtype=bool)
        clusters = []
        for i in range(K):
            if seen[i]:
                continue
            stack = [i]
            seen[i] = True
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in np.where(adj[u])[0]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            comp = np.array(sorted(comp), dtype=int)
            if comp.size >= int(min_cluster_size):
                clusters.append(comp)

        if not clusters:
            clusters = [np.arange(K, dtype=int)]

        mk = np.array([np.mean(k_list[c]) for c in clusters], dtype=float)
        order = np.argsort(mk)
        clusters = [clusters[i] for i in order]

        labels = -np.ones(K, dtype=int)
        for cid, c in enumerate(clusters):
            labels[c] = cid
        return labels

    if P < 2:
        raise ValueError("Need at least two p points to build p0->p consensus.")

    end_indices = np.arange(1, P, dtype=int)  # e = 1..P-1
    n_end = end_indices.size
    consensus_cube = np.zeros((n_end, K, K), dtype=float)
    evaluated_counts = np.zeros(n_end, dtype=int)
    effective_prefix_ts = [[] for _ in range(n_end)]

    min_t = max(int(min_prefix_points), 2)
    stride = max(int(prefix_stride), 1)

    for ii, e in enumerate(end_indices):
        prefix_ts = list(range(min_t, e + 2, stride))  # t is inclusive length
        if len(prefix_ts) == 0:
            prefix_ts = [e + 1]

        cons_e = np.zeros((K, K), dtype=float)
        labels_count = 0

        for t in prefix_ts:
            Xwin = x_trend[:, :t]  # always starts at p0
            if cluster_on_diff:
                Xwin = np.diff(Xwin, axis=1)
                if Xwin.shape[1] < 2:
                    continue
            elif Xwin.shape[1] < 2:
                continue

            S = _similarity_matrix_rows(Xwin)
            labels = _cluster_labels_from_similarity(S)
            cons_e += (labels[:, None] == labels[None, :]).astype(float)
            labels_count += 1
            effective_prefix_ts[ii].append(int(t))

        if labels_count == 0:
            cons_e = np.eye(K, dtype=float)
            labels_count = 1
        else:
            cons_e /= float(labels_count)
            np.fill_diagonal(cons_e, 1.0)

        consensus_cube[ii] = cons_e
        evaluated_counts[ii] = int(labels_count)

    fig = None
    if make_plot:
        traces = []
        for ii, e in enumerate(end_indices):
            traces.append(
                go.Heatmap(
                    z=consensus_cube[ii],
                    x=k_list,
                    y=k_list,
                    zmin=0.0,
                    zmax=1.0,
                    colorbar=dict(title="P(same cluster)") if ii == 0 else None,
                    hovertemplate=(
                        "end p=%{customdata:.4g}<br>"
                        "k_i=%{y}<br>k_j=%{x}<br>consensus=%{z:.3f}<extra></extra>"
                    ),
                    customdata=np.full((K, K), p_list[e]),
                    visible=(ii == 0),
                )
            )

        fig = go.Figure(data=traces)
        steps = []
        for ii, e in enumerate(end_indices):
            vis = [False] * n_end
            vis[ii] = True
            title = (
                "Consensus From p0 To p "
                f"(p_end={p_list[e]:.4g}, metric={metric}, "
                f"windows={evaluated_counts[ii]})"
            )
            steps.append(
                dict(
                    method="update",
                    args=[{"visible": vis}, {"title": title}],
                    label=f"{p_list[e]:.3g}",
                )
            )

        fig.update_layout(
            width=width,
            height=height,
            template="plotly_white",
            title=(
                "Consensus From p0 To p "
                f"(p_end={p_list[1]:.4g}, metric={metric}, windows={evaluated_counts[0]})"
            ),
            sliders=[
                dict(
                    active=0,
                    currentvalue={"prefix": "end p: "},
                    pad={"t": 40},
                    steps=steps,
                )
            ],
        )
        fig.update_xaxes(title_text="k_j")
        fig.update_yaxes(title_text="k_i")

    return {
        "consensus_cube": consensus_cube,  # (P-1, K, K), index = end-index in end_indices
        "end_indices": end_indices,
        "end_p_list": p_list[end_indices],
        "p_list": p_list,
        "k_list": k_list,
        "evaluated_counts": evaluated_counts,  # number of windows used per end p
        "effective_prefix_ts": effective_prefix_ts,
        "x_trend": x_trend,
        "fig": fig,
        "settings": dict(
            metric=metric,
            sim_threshold=float(sim_threshold),
            k_window=int(k_window),
            min_cluster_size=int(min_cluster_size),
            cluster_on_diff=bool(cluster_on_diff),
            min_prefix_points=int(min_prefix_points),
            prefix_stride=int(prefix_stride),
            use_baseline_correction=bool(use_baseline_correction),
            aggregation=str(aggregation),
        ),
    }


## Analysis of Iproj_boot


from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt


def _quantile(a, q, axis=None):
    """Convenience wrapper around np.quantile."""
    return np.quantile(a, q, axis=axis)


def _safe_log(x: float) -> float:
    """Log with floor to avoid -inf."""
    return float(np.log(max(x, 1e-300)))


def _bh_fdr_reject(pvals: np.ndarray, q: float) -> np.ndarray:
    """
    Benjamini–Hochberg FDR procedure across a 1D array of p-values.

    Parameters
    ----------
    pvals : (m,) array
        P-values to test.
    q : float
        Target FDR level in (0,1).

    Returns
    -------
    reject : (m,) bool array
        True where the null is rejected at FDR q.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = (np.arange(1, m + 1) / m) * float(q)
    below = ranked <= thresh
    if not np.any(below):
        return np.zeros(m, dtype=bool)
    kmax = int(np.max(np.where(below)[0]))
    cutoff = ranked[kmax]
    return pvals <= cutoff


def _find_first_p_ge_threshold(
    p_list: np.ndarray, curve: np.ndarray, thr: float
) -> Optional[float]:
    """Return smallest p where curve >= thr; if never crosses, return None."""
    idx = np.where(curve >= thr)[0]
    return None if idx.size == 0 else float(p_list[int(idx[0])])


def _default_select_k_for_curves(k_list: np.ndarray, max_curves: int = 8) -> np.ndarray:
    """Pick a few k's to plot (roughly evenly spaced)."""
    k_list = np.asarray(k_list, dtype=int)
    if k_list.size <= max_curves:
        return k_list
    idx = np.linspace(0, k_list.size - 1, max_curves).round().astype(int)
    idx = np.unique(idx)
    return k_list[idx]


def _try_beta_cdf():
    """
    Try to load a Beta CDF implementation.
    We prefer scipy.stats.beta.cdf; fallback to scipy.special.betainc.
    """
    try:
        from scipy.stats import beta as _beta_dist

        return ("stats", _beta_dist.cdf)
    except Exception:
        try:
            from scipy.special import betainc

            def _cdf(x, a, b):
                return betainc(a, b, x)

            return ("special", _cdf)
        except Exception:
            return (None, None)


_BETA_CDF_KIND, _BETA_CDF = _try_beta_cdf()


def _estimate_kappa_hat(
    I_med: np.ndarray,
    p_list: np.ndarray,
    hi_band_quantile: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Estimate kappa_hat[k] from the high-p band using scaled leakage:
        ell_k(p) = (1 - I_med[k,p]) * p/(1-p)
        kappa_hat[k] = median_{p in high band} ell_k(p)

    Returns
    -------
    kappa_hat : (K,) array
    info : dict with band indices, etc.
    """
    p_list = np.asarray(p_list, dtype=float)

    # Determine "high-p band": p >= quantile(p_list, hi_band_quantile)
    p_hi = float(np.quantile(p_list, float(hi_band_quantile)))
    hi_idx = np.where(p_list >= p_hi)[0]
    if hi_idx.size == 0:
        hi_idx = np.array([len(p_list) - 1], dtype=int)

    # Scale factor p/(1-p); protect against division by 0 at p=1
    scale = p_list[hi_idx] / np.maximum(1.0 - p_list[hi_idx], 1e-12)

    # ell_k(p) computed using MEDIAN curve; shape becomes (K, |hi_idx|)
    ell = (1.0 - I_med[:, hi_idx]) * scale[None, :]

    # Robust aggregation across the band
    kappa_hat = np.median(ell, axis=1)

    info = dict(p_hi=p_hi, hi_idx=hi_idx.tolist(), hi_band_quantile=hi_band_quantile)
    return kappa_hat.astype(float), info


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-12)


def _smooth_median(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if w <= 1:
        return x.copy()
    n = x.size
    out = np.zeros_like(x)
    half = w // 2
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        out[i] = np.median(x[a:b])
    return out


def kappa_changepoint(
    kappa_hat: np.ndarray,
    k_list: np.ndarray,
    *,
    smooth_window: int = 5,
    changepoint_mode: str = "largest_jump",  # "largest_jump" or "first_above_quantile"
    jump_metric: str = "abs",  # "abs" or "rel"
    jump_quantile: float = 0.90,  # used only for first_above_quantile
    min_k: int = 2,
) -> tuple[int, dict]:
    """
    Returns k_cut, with signals defined as {k <= k_cut}.

    largest_jump: chooses argmax jump (or relative jump) after smoothing.
    first_above_quantile: chooses first jump exceeding quantile threshold.
    """
    kappa_hat = np.asarray(kappa_hat, dtype=float)
    k_list = np.asarray(k_list, dtype=int)
    K = kappa_hat.size
    if K < 3:
        return int(k_list[-1]), dict(reason="K<3")

    sm = _smooth_median(kappa_hat, int(max(1, smooth_window)))
    d = sm[1:] - sm[:-1]

    # Restrict to k >= min_k
    valid = np.where(k_list[:-1] >= int(min_k))[0]
    if valid.size == 0:
        valid = np.arange(K - 1)

    if jump_metric == "rel":
        denom = np.maximum(sm[:-1], 1e-12)
        score = d / denom
    else:
        score = d

    if changepoint_mode == "largest_jump":
        i_star = int(valid[np.argmax(score[valid])])
        k_cut = int(k_list[i_star])
        info = dict(
            mode="largest_jump",
            jump_metric=jump_metric,
            smoothed=sm,
            diffs=d,
            score=score,
            i_star=i_star,
            k_cut=k_cut,
        )
        return k_cut, info

    elif changepoint_mode == "first_above_quantile":
        pos = score[valid][score[valid] > 0]
        if pos.size == 0:
            i_star = int(valid[np.argmax(score[valid])])
            k_cut = int(k_list[i_star])
            return k_cut, dict(
                mode="first_above_quantile",
                reason="no positive score; used argmax",
                smoothed=sm,
                diffs=d,
                score=score,
                i_star=i_star,
                k_cut=k_cut,
            )

        thr = float(np.quantile(pos, float(jump_quantile)))
        cand = valid[np.where(score[valid] >= thr)[0]]
        if cand.size == 0:
            i_star = int(valid[np.argmax(score[valid])])
            k_cut = int(k_list[i_star])
            return k_cut, dict(
                mode="first_above_quantile",
                reason="no cand; used argmax",
                thr=thr,
                smoothed=sm,
                diffs=d,
                score=score,
                i_star=i_star,
                k_cut=k_cut,
            )

        i_star = int(cand[0])
        k_cut = int(k_list[i_star])
        return k_cut, dict(
            mode="first_above_quantile",
            thr=thr,
            smoothed=sm,
            diffs=d,
            score=score,
            i_star=i_star,
            k_cut=k_cut,
        )

    else:
        raise ValueError(
            "changepoint_mode must be 'largest_jump' or 'first_above_quantile'."
        )


def eps_from_signal_quantile(
    I_med: np.ndarray,
    I_lo: np.ndarray,
    p_list: np.ndarray,
    sig_mask: np.ndarray,
    *,
    p_op: float,
    q: float = 0.10,
    use_lower_ci: bool = True,
) -> tuple[float, dict]:
    """
    Cap-free epsilon: choose threshold as a low-quantile of signal overlaps at p_op.
      threshold = Quantile( {Y_k}, q ),  Y_k = I_lo(k,p_op) or I_med(k,p_op) over k in signals
      eps = 1 - threshold
    """
    p_list = np.asarray(p_list, dtype=float)
    j = int(np.argmin(np.abs(p_list - float(p_op))))
    Y = (I_lo if use_lower_ci else I_med)[:, j]
    pool = Y[np.asarray(sig_mask, dtype=bool)]
    if pool.size == 0:
        pool = Y  # fallback: use all k
        pool_desc = "fallback_all_k"
    else:
        pool_desc = "signals_only"

    thr = float(np.quantile(pool, float(q)))
    eps = float(np.clip(1.0 - thr, 1e-6, 1.0 - 1e-6))  # no arbitrary 0.45 cap
    info = dict(
        p_op_used=float(p_list[j]),
        q=float(q),
        pool_desc=pool_desc,
        threshold=thr,
        eps=eps,
        use_lower_ci=bool(use_lower_ci),
    )
    return eps, info


# ==============================================================================
# Masked Parallel Analysis (B2 rank estimation)
# ==============================================================================


def _mpa_worker(args):
    """Worker for masked parallel analysis: eigenvalues of one masked matrix.

    Each worker reconstructs (or permutes) the matrix from shared arrays,
    applies nested Bernoulli masks across the full p-grid, and returns
    eigenvalues for every p in a single call.  This avoids dispatching
    one task per (replicate, p) pair and drastically reduces scheduling
    overhead.
    """
    idx, p_list, is_null, diag, triu_vals, n, k_max, seed_base = args
    from scipy.linalg import eigh as _eigh

    rng = np.random.default_rng(seed_base)
    iu = np.triu_indices(n, k=1)
    m = len(iu[0])

    if is_null:
        vals = rng.permutation(triu_vals)
    else:
        vals = triu_vals

    A = np.zeros((n, n), dtype=np.float64)
    A[iu] = vals
    A = A + A.T
    np.fill_diagonal(A, diag)

    U = rng.random(m)

    p_list = np.asarray(p_list, dtype=np.float64)
    out = np.empty((k_max, len(p_list)), dtype=np.float64)
    for j, p in enumerate(p_list):
        mask = U < p
        A_masked = np.zeros((n, n), dtype=np.float64)
        np.fill_diagonal(A_masked, diag)
        inv_p = 1.0 / p
        masked_vals = vals[mask] * inv_p
        A_masked[iu[0][mask], iu[1][mask]] = masked_vals
        A_masked[iu[1][mask], iu[0][mask]] = masked_vals

        evals = _eigh(
            A_masked,
            subset_by_index=[n - k_max, n - 1],
            eigvals_only=True,
        )
        out[:, j] = evals[::-1]

    return idx, is_null, out


def masked_parallel_analysis(
    S,
    k_max,
    p_list,
    B=50,
    J=100,
    alpha=0.05,
    random_state=42,
    n_jobs=None,
    show_progress=True,
):
    """Masked parallel analysis for principled rank estimation.

    Compares eigenvalues of bootstrap-masked ``S`` against a permutation null
    where off-diagonal entries are randomly reassigned (destroying structure
    while preserving marginals).  The estimated rank ``k*`` is the largest
    dimension whose observed eigenvalue exceeds the null at every masking rate
    ``p >= median(p_list)``.

    Parameters
    ----------
    S : (n, n) array
        Symmetric similarity matrix (may contain ``NaN`` off-diagonal).
    k_max : int
        Maximum rank to test.
    p_list : array-like
        Grid of masking fractions in (0, 1].
    B : int
        Bootstrap replicates for observed eigenvalue aggregation.
    J : int
        Null permutation replicates.
    alpha : float
        Significance level.
    random_state : int
        Seed for reproducibility.
    n_jobs : int or None
        Parallel workers (``None`` uses cpu_count - 1).
    show_progress : bool
        Show tqdm progress bar.

    Returns
    -------
    dict with keys:
        k_star, pvalues, evals_observed, evals_null, thresholds,
        evals_ref, p_list, params.
    """
    S_sym = _symmetrize_with_nan(S)
    n = S_sym.shape[0]
    k_max = min(k_max, n - 1)
    p_list = np.sort(np.asarray(p_list, dtype=np.float64))

    S0 = np.where(np.isfinite(S_sym), S_sym, 0.0)
    diag = np.diag(S0).copy()
    iu = np.triu_indices(n, k=1)
    triu_vals = S0[iu].copy()

    from scipy.linalg import eigh as _eigh

    evals_ref = _eigh(S0, subset_by_index=[n - k_max, n - 1], eigvals_only=True)
    evals_ref = evals_ref[::-1].copy()

    n_jobs_eff = n_jobs if n_jobs and n_jobs > 0 else max(1, (os.cpu_count() or 2) - 1)

    rng_master = np.random.default_rng(random_state)
    obs_seeds = rng_master.integers(0, 2**63, size=B)
    null_seeds = rng_master.integers(0, 2**63, size=J)

    tasks = []
    for b in range(B):
        tasks.append((b, p_list, False, diag, triu_vals, n, k_max, int(obs_seeds[b])))
    for j in range(J):
        tasks.append((j, p_list, True, diag, triu_vals, n, k_max, int(null_seeds[j])))

    P = len(p_list)
    results = Parallel(n_jobs=n_jobs_eff, prefer="processes")(
        delayed(_mpa_worker)(t)
        for t in tqdm(tasks, desc="MPA", disable=not show_progress)
    )

    evals_obs_all = np.empty((k_max, P, B), dtype=np.float64)
    evals_null_all = np.empty((k_max, P, J), dtype=np.float64)

    for idx, is_null, out in results:
        if is_null:
            evals_null_all[:, :, idx] = out
        else:
            evals_obs_all[:, :, idx] = out

    evals_observed = np.median(evals_obs_all, axis=2)

    thresholds = np.quantile(evals_null_all, 1.0 - alpha, axis=2)

    p_median = float(np.median(p_list))
    high_p_mask = p_list >= p_median
    high_p_idx = np.where(high_p_mask)[0]

    pvalues = np.ones(k_max, dtype=np.float64)
    for k in range(k_max):
        worst_pval = 0.0
        for pi in high_p_idx:
            obs_val = evals_observed[k, pi]
            null_vals = evals_null_all[k, pi, :]
            pval = (1.0 + np.sum(null_vals >= obs_val)) / (1.0 + J)
            if pval > worst_pval:
                worst_pval = pval
        pvalues[k] = worst_pval

    significant = np.where(pvalues < alpha)[0]
    k_star = int(significant[-1] + 1) if significant.size > 0 else 0

    return dict(
        k_star=k_star,
        pvalues=pvalues,
        evals_observed=evals_observed,
        evals_null=evals_null_all,
        thresholds=thresholds,
        evals_ref=evals_ref,
        p_list=p_list,
        params=dict(
            k_max=k_max,
            B=B,
            J=J,
            alpha=alpha,
            random_state=random_state,
            n=n,
        ),
    )


@dataclass
class IprojAnalysisResult:
    summary: Dict[str, Any]
    p_star: Dict[str, Any]
    figures: Dict[str, Any]


def _find_first_run_ge_threshold(
    p_list: np.ndarray, curve: np.ndarray, thr: float, run: int = 1
) -> Optional[float]:
    """Return the first p where curve stays >= thr for `run` consecutive grid points."""
    p_list = np.asarray(p_list, dtype=float)
    curve = np.asarray(curve, dtype=float)
    run = int(max(1, run))
    mask = curve >= float(thr)
    if run == 1:
        idx = np.where(mask)[0]
        return None if idx.size == 0 else float(p_list[int(idx[0])])
    streak = 0
    for i, ok in enumerate(mask):
        streak = streak + 1 if ok else 0
        if streak >= run:
            return float(p_list[int(i - run + 1)])
    return None


def analyze_iproj_signal_alignment_v3(
    Iproj_boot: np.ndarray,
    k_list: Union[List[int], np.ndarray],
    p_list: Union[List[float], np.ndarray],
    *,
    ci: float = 0.90,
    K_sig_mode: str = "kappa_changepoint",
    hi_band_quantile: float = 0.85,
    smooth_window: int = 3,
    changepoint_mode: str = "largest_jump",
    jump_metric: str = "abs",
    jump_quantile: float = 0.85,
    min_k_for_jump: int = 2,
    n_dim: Optional[int] = None,
    p_cap: float = 0.90,
    fdr_q: float = 0.10,
    signal_curve_mode: str = "lower_ci",  # "lower_ci" or "median"
    noise_curve_mode: str = "upper_ci",  # "upper_ci" or "median"
    signal_reference: str = "boundary",  # "boundary" or "signal_quantile"
    signal_quantile: float = 0.10,
    noise_quantile: float = 0.90,
    lift_margin: float = 0.0,
    require_consecutive: int = 1,
    make_plots: bool = True,
    max_curve_plots: int = 8,
    figsize_heatmap: Tuple[int, int] = (10, 6),
    figsize_curves: Tuple[int, int] = (10, 6),
    figsize_kappa: Tuple[int, int] = (10, 4),
    figsize_liftoff: Tuple[int, int] = (10, 5),
) -> IprojAnalysisResult:
    """
    Variant of v2 with data-driven p* defined by signal-vs-noise lift-off.

    Main idea
    ---------
    1) detect the signal set K_sig (default: kappa changepoint)
    2) build a noise reference curve from k > k_cut
    3) define p*_k as the first p where the signal curve for k clears the noise
       reference by `lift_margin`

    This removes the need for p_op / eps calibration in v2.
    """
    Iproj_boot = np.asarray(Iproj_boot, dtype=float)
    k_list = np.asarray(k_list, dtype=int)
    p_list = np.asarray(p_list, dtype=float)

    if Iproj_boot.ndim != 3:
        raise ValueError("Iproj_boot must have shape (K_count, P_count, B).")
    K_count, P_count, B = Iproj_boot.shape
    if k_list.size != K_count:
        raise ValueError("k_list length must match Iproj_boot.shape[0].")
    if p_list.size != P_count:
        raise ValueError("p_list length must match Iproj_boot.shape[1].")
    if np.any(np.diff(p_list) < 0):
        raise ValueError("p_list must be sorted increasing.")

    alpha_ci = 1.0 - float(ci)
    q_lo = alpha_ci / 2.0
    q_hi = 1.0 - alpha_ci / 2.0

    I_med = np.median(Iproj_boot, axis=2)
    I_lo = _quantile(Iproj_boot, q_lo, axis=2)
    I_hi = _quantile(Iproj_boot, q_hi, axis=2)

    kappa_hat, kappa_info = _estimate_kappa_hat(I_med, p_list, hi_band_quantile)

    K_sig_mask = np.zeros(K_count, dtype=bool)
    K_sig_details: Dict[str, Any] = dict(mode=K_sig_mode)
    mode = str(K_sig_mode).lower()

    if mode == "kappa_changepoint":
        k_cut, cp_info = kappa_changepoint(
            kappa_hat,
            k_list,
            smooth_window=smooth_window,
            changepoint_mode=changepoint_mode,
            jump_metric=jump_metric,
            jump_quantile=jump_quantile,
            min_k=min_k_for_jump,
        )
        K_sig_mask = k_list <= k_cut
        K_sig_details.update(
            k_cut=int(k_cut),
            changepoint_info=cp_info,
            interpretation="Signals defined as {k <= k_cut} where k_cut is the kappa jump location.",
        )
    elif mode == "beta_null":
        if n_dim is None:
            raise ValueError("beta_null mode requires n_dim (matrix dimension n).")
        if _BETA_CDF is None:
            raise ValueError(
                "beta_null mode requires SciPy (stats.beta.cdf or special.betainc)."
            )
        n = int(n_dim)
        cap_idx = np.where(p_list <= float(p_cap))[0]
        if cap_idx.size == 0:
            cap_idx = np.array([0], dtype=int)
        mcap = int(cap_idx.size)
        pi = np.ones((K_count, P_count), dtype=float)
        for i, k in enumerate(k_list):
            a = 0.5
            b = 0.5 * (n - int(k))
            if b <= 0:
                continue
            for t in range(P_count):
                x = float(np.clip(I_med[i, t], 0.0, 1.0))
                cdf = float(_BETA_CDF(x, a, b))
                pi[i, t] = float(np.clip(1.0 - cdf, 0.0, 1.0))
        pi_curve = np.minimum(1.0, mcap * np.min(pi[:, cap_idx], axis=1))
        K_sig_mask = _bh_fdr_reject(pi_curve, q=float(fdr_q))
        K_sig_details.update(
            n=n,
            p_cap=float(p_cap),
            p_cap_count=mcap,
            fdr_q=float(fdr_q),
            beta_backend=_BETA_CDF_KIND,
            pvals_curve=pi_curve,
        )
    elif mode == "none":
        K_sig_mask[:] = True
        K_sig_details.update(note="All k treated as signals (debug mode).")
    else:
        raise ValueError(
            "K_sig_mode must be one of: 'kappa_changepoint', 'beta_null', 'none'."
        )

    K_sig_indices = np.where(K_sig_mask)[0]
    K_noise_mask = ~K_sig_mask
    K_noise_indices = np.where(K_noise_mask)[0]
    K_sig_k_values = k_list[K_sig_mask].tolist()
    K_noise_k_values = k_list[K_noise_mask].tolist()

    if signal_curve_mode == "lower_ci":
        signal_curves = I_lo
    elif signal_curve_mode == "median":
        signal_curves = I_med
    else:
        raise ValueError("signal_curve_mode must be 'lower_ci' or 'median'.")

    if noise_curve_mode == "upper_ci":
        noise_curves = I_hi
    elif noise_curve_mode == "median":
        noise_curves = I_med
    else:
        raise ValueError("noise_curve_mode must be 'upper_ci' or 'median'.")

    noise_pool_desc = "noise_only"
    if len(K_noise_indices) > 0:
        noise_pool = noise_curves[K_noise_mask]
    else:
        start = max(K_count // 2, 1)
        noise_pool = noise_curves[start:]
        noise_pool_desc = "fallback_upper_half_k"
    noise_ref_curve = np.quantile(noise_pool, float(noise_quantile), axis=0)

    p_star_by_k: List[Optional[float]] = []
    liftoff_gap_by_k = []
    for i in range(K_count):
        gap = signal_curves[i] - noise_ref_curve
        liftoff_gap_by_k.append(gap)
        p_star_by_k.append(
            _find_first_run_ge_threshold(
                p_list,
                gap,
                float(lift_margin),
                run=require_consecutive,
            )
        )
    liftoff_gap_by_k = np.asarray(liftoff_gap_by_k, dtype=float)

    ref_info: Dict[str, Any] = dict(signal_reference=signal_reference)
    if signal_reference == "boundary":
        if len(K_sig_indices) > 0:
            ref_idx = int(K_sig_indices[-1])
        else:
            ref_idx = int(np.argmin(kappa_hat))
            ref_info["fallback"] = "argmin_kappa_hat"
        signal_ref_curve = signal_curves[ref_idx]
        signal_ref_label = f"boundary k={int(k_list[ref_idx])}"
        ref_info.update(ref_k=int(k_list[ref_idx]), ref_index=ref_idx)
    elif signal_reference == "signal_quantile":
        if len(K_sig_indices) > 0:
            pool = signal_curves[K_sig_mask]
            pool_desc = "signals_only"
        else:
            pool = signal_curves
            pool_desc = "fallback_all_k"
        signal_ref_curve = np.quantile(pool, float(signal_quantile), axis=0)
        signal_ref_label = f"signal q={float(signal_quantile):.2f}"
        ref_info.update(signal_quantile=float(signal_quantile), pool_desc=pool_desc)
    else:
        raise ValueError("signal_reference must be 'boundary' or 'signal_quantile'.")

    ref_gap_curve = signal_ref_curve - noise_ref_curve
    p_star_reference = _find_first_run_ge_threshold(
        p_list,
        ref_gap_curve,
        float(lift_margin),
        run=require_consecutive,
    )

    p_star_global_all = None
    if all(v is not None for v in p_star_by_k):
        p_star_global_all = float(np.max(np.asarray(p_star_by_k, dtype=float)))

    p_star_global_signals = None
    if len(K_sig_indices) > 0:
        vals = [p_star_by_k[i] for i in K_sig_indices]
        if all(v is not None for v in vals):
            p_star_global_signals = float(np.max(np.asarray(vals, dtype=float)))

    recommended_global = (
        p_star_reference if p_star_reference is not None else p_star_global_signals
    )

    summary = dict(
        I_median=I_med,
        I_ci_low=I_lo,
        I_ci_high=I_hi,
        ci_level=float(ci),
        kappa_hat=kappa_hat,
        kappa_info=kappa_info,
        K_sig_mode=K_sig_mode,
        K_sig_details=K_sig_details,
        K_sig_indices=K_sig_indices.tolist(),
        K_noise_indices=K_noise_indices.tolist(),
        K_sig_k_values=K_sig_k_values,
        K_noise_k_values=K_noise_k_values,
        p_star_definition="signal_vs_noise_liftoff",
        signal_curve_mode=signal_curve_mode,
        noise_curve_mode=noise_curve_mode,
        signal_reference=signal_reference,
        signal_quantile=float(signal_quantile),
        noise_quantile=float(noise_quantile),
        lift_margin=float(lift_margin),
        require_consecutive=int(require_consecutive),
        noise_ref_curve=noise_ref_curve,
        signal_ref_curve=signal_ref_curve,
        ref_gap_curve=ref_gap_curve,
        reference_info=ref_info,
        noise_pool_desc=noise_pool_desc,
    )

    p_star = dict(
        by_k=p_star_by_k,
        global_all=p_star_global_all,
        global_signals=p_star_global_signals,
        reference_signal=p_star_reference,
        recommended_global=recommended_global,
        K_sig_k_values=K_sig_k_values,
    )

    figures: Dict[str, Any] = {}
    if make_plots:
        fig_hm, ax_hm = plt.subplots(figsize=figsize_heatmap)
        im = ax_hm.imshow(
            I_med,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[p_list[0], p_list[-1], 0, K_count],
        )
        ax_hm.set_title("Median $I^{\\mathrm{proj}}_k(p)$ across bootstrap")
        ax_hm.set_xlabel("$p$")
        ax_hm.set_ylabel("index in k_list (0..K-1)")
        plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
        figures["heatmap_median"] = fig_hm

        sel_k = _default_select_k_for_curves(k_list, max_curves=max_curve_plots)
        sel_idx = [int(np.where(k_list == kk)[0][0]) for kk in sel_k]
        fig_curves, ax_curves = plt.subplots(figsize=figsize_curves)
        for i in sel_idx:
            ax_curves.plot(p_list, I_med[i], label=f"k={k_list[i]}")
            ax_curves.fill_between(p_list, I_lo[i], I_hi[i], alpha=0.12)
        ax_curves.plot(
            p_list,
            noise_ref_curve,
            linestyle="--",
            linewidth=2,
            label=f"noise q={float(noise_quantile):.2f}",
        )
        ax_curves.plot(p_list, signal_ref_curve, linewidth=2.5, label=signal_ref_label)
        ax_curves.axhline(0.0, alpha=0.0)
        ax_curves.set_title(
            "Selected $I^{\\mathrm{proj}}_k(p)$ curves with signal/noise references"
        )
        ax_curves.set_xlabel("$p$")
        ax_curves.set_ylabel("$I^{\\mathrm{proj}}$")
        ax_curves.legend(ncols=2, fontsize=9)
        figures["curves_selected"] = fig_curves

        fig_kappa, ax_kappa = plt.subplots(figsize=figsize_kappa)
        ax_kappa.plot(
            k_list, kappa_hat, marker=".", linestyle="-", label="$\\hat\\kappa_k$"
        )
        if mode == "kappa_changepoint" and "k_cut" in K_sig_details:
            k_cut = int(K_sig_details["k_cut"])
            ax_kappa.axvline(
                k_cut + 0.5, linestyle="--", linewidth=1.5, label=f"cut at k={k_cut}"
            )
        if len(K_sig_indices) > 0:
            ax_kappa.scatter(
                k_list[K_sig_mask],
                kappa_hat[K_sig_mask],
                s=60,
                marker="s",
                label=r"$k\in\mathcal{K}_{\mathrm{sig}}$",
            )
        ax_kappa.set_title("High-$p$ scaled leakage difficulty $\\hat\\kappa_k$")
        ax_kappa.set_xlabel("$k$")
        ax_kappa.set_ylabel("$\\hat\\kappa_k$")
        ax_kappa.legend(fontsize=9)
        figures["kappa_hat"] = fig_kappa

        fig_lift, ax_lift = plt.subplots(figsize=figsize_liftoff)
        ax_lift.plot(p_list, signal_ref_curve, linewidth=2.5, label=signal_ref_label)
        ax_lift.plot(
            p_list,
            noise_ref_curve,
            linestyle="--",
            linewidth=2,
            label=f"noise q={float(noise_quantile):.2f}",
        )
        ax_lift.plot(
            p_list, ref_gap_curve, linestyle=":", linewidth=2, label="signal - noise"
        )
        ax_lift.axhline(
            float(lift_margin),
            color="k",
            linestyle="-.",
            linewidth=1.2,
            label=f"margin={float(lift_margin):.3f}",
        )
        if p_star_reference is not None:
            ax_lift.axvline(
                float(p_star_reference),
                color="k",
                linestyle="--",
                linewidth=1.2,
                label=f"reference p*={float(p_star_reference):.3f}",
            )
        ax_lift.set_title("Reference signal lift-off against noise band")
        ax_lift.set_xlabel("$p$")
        ax_lift.set_ylabel("curve / gap value")
        ax_lift.legend(fontsize=9)
        figures["liftoff_reference"] = fig_lift

        fig_ps, ax_ps = plt.subplots(figsize=figsize_curves)
        vals = np.array([np.nan if v is None else v for v in p_star_by_k], dtype=float)
        ax_ps.plot(k_list, vals, marker="o", linestyle="-", label="$p_k^*$")
        if len(K_sig_indices) > 0:
            ax_ps.scatter(
                k_list[K_sig_mask],
                vals[K_sig_mask],
                s=60,
                marker="s",
                label=r"$k\in\mathcal{K}_{\mathrm{sig}}$",
            )
        if p_star_reference is not None:
            ax_ps.axhline(
                float(p_star_reference),
                linestyle="--",
                linewidth=1.2,
                label=f"reference p*={float(p_star_reference):.3f}",
            )
        ax_ps.set_title("$p_k^*$ by $k$ under signal-vs-noise lift-off")
        ax_ps.set_xlabel("$k$")
        ax_ps.set_ylabel("$p_k^*$")
        ax_ps.set_ylim(0, 1.02)
        ax_ps.legend(fontsize=9)
        figures["pstar_vs_k"] = fig_ps

    return IprojAnalysisResult(summary=summary, p_star=p_star, figures=figures)
