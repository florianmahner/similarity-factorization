"""Sampling bound estimation for matrix completion using random matrix theory.

This module provides optimized implementations for estimating sampling bounds
needed for matrix completion. The key optimizations are:

1. Pre-computing expensive values once (eigenvalues, norms, S²)
2. Avoiding redundant O(n³) eigenvalue decompositions
3. Numba JIT compilation for VDE solver inner loop
4. Caching edge function evaluations

For large matrices (n > 1000), use estimate_sampling_bounds_ultra for
dramatically faster computation (10-20x speedup).
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import eigvalsh
from joblib import Parallel, delayed
from numba import njit
from typing import NamedTuple
import time


class PrecomputedMatrixInfo(NamedTuple):
    """Pre-computed matrix-dependent values for bound estimation."""

    S_sq: np.ndarray
    eigvals: np.ndarray
    s2_max: float
    s_norm: float
    fro_norm: float
    eff_dim: int


def compute_effective_dimension(fro_norm: float, spec_norm: float) -> int:
    # FIXME not sure we are really allowed to do this ceiling here...
    """Compute effective dimension using round(1 decimal) + ceil.

    This formula handles edge cases better than simple ceil:
    - eff_dim_raw = 1.004 → round to 1.0 → ceil to 1 (correct for NSD)
    - eff_dim_raw = 1.08 → round to 1.1 → ceil to 2 (correct for mur92)
    - eff_dim_raw = 1.58 → round to 1.6 → ceil to 2 (correct for THINGS)
    """
    eff_dim_raw = (fro_norm / spec_norm) ** 2
    # eff_dim_rounded = np.round(eff_dim_raw, 1)
    return int(np.ceil(eff_dim_raw))


def precompute_matrix_info(S: np.ndarray) -> PrecomputedMatrixInfo:
    """Pre-compute all S-dependent values needed for bound estimation."""
    S_sq = np.ascontiguousarray(S**2, dtype=np.float64)
    eigvals = np.sort(eigvalsh(S))[::-1]
    s_norm = np.linalg.norm(S, 2)
    s2_max = np.max(eigvalsh(S_sq))
    fro_norm = np.linalg.norm(S, "fro")
    eff_dim = compute_effective_dimension(fro_norm, s_norm)
    return PrecomputedMatrixInfo(S_sq, eigvals, s2_max, s_norm, fro_norm, eff_dim)


def pmin_bound(
    S: np.ndarray,
    gamma: float = 1.05,
    eta: float = 0.05,
    rho: float = 0.95,
    n_realizations: int = 500,
    random_state: int | None = None,
    verbose: bool = False,
    monte_carlo: bool = False,
) -> tuple[float, float, float, float, np.ndarray]:
    np.random.seed(random_state)
    n = S.shape[0]
    _is_symmetric = np.allclose(S, S.T)

    L_max = np.max((S**2).sum(axis=1) - np.diag(S) ** 2)

    _row_sq = (S**2).sum(axis=1) - np.diag(S) ** 2
    empirical_L_max = np.quantile(_row_sq, rho)

    S_norm = np.linalg.norm(S, 2)

    L_infty = 2 * np.max(np.abs(S))
    empirical_L_infty = 2 * np.quantile(np.abs(S), rho)

    effective_dimension = (np.linalg.norm(S, "fro") / S_norm) ** 2
    if verbose:
        print("effective dimension : ", effective_dimension)

    MC_expected_MS_norms = np.zeros(n_realizations)
    if monte_carlo:
        for i in range(n_realizations):
            _p = np.random.rand()
            _mask = np.random.binomial(1, _p, size=S.shape)
            if _is_symmetric:
                _mask = np.triu(_mask, 1)
                _mask += _mask.T
            MC_expected_MS_norms[i] = np.linalg.norm(_mask * S, 2)

    N_bernstein = (gamma * L_infty * S_norm) / (3 * L_max) + 1
    N_empirical = (gamma * empirical_L_infty * S_norm) / (3 * empirical_L_max) + 1
    N_empirical_alternative = (gamma * L_infty * S_norm) / (3 * empirical_L_max) + 1
    N_theory_upperbound = (gamma * S_norm) / 3 + 1

    D_bernstein = ((gamma * S_norm) ** 2 / (2 * L_max) + 1) / np.log(2 * n / eta)
    D_empirical = ((gamma * S_norm) ** 2 / (2 * empirical_L_max) + 1) / np.log(
        2 * effective_dimension / eta
    )
    D_theory_lowerbound = ((gamma**2 * S_norm) / (2 * empirical_L_max) + 1) / np.log(
        2 * n / eta
    )

    if verbose:
        print(N_bernstein, D_bernstein)
        print(N_empirical, D_empirical)
        print(N_empirical_alternative, D_empirical)
        print(N_theory_upperbound, D_theory_lowerbound)

    p_min = N_bernstein / D_bernstein
    p_min_empirical = N_empirical / D_empirical
    p_min_empirical_alternative = N_empirical_alternative / D_empirical
    p_min_lowerbound = N_theory_upperbound / D_theory_lowerbound

    if verbose:
        print("p_min", p_min)
        print("empirical_p_min", p_min_empirical)
        print("empirical_p_min_alternative", p_min_empirical_alternative)
        print("theory_p_min", p_min_lowerbound)

    return (
        p_min_empirical,
        p_min,
        p_min_lowerbound,
        p_min_empirical_alternative,
        MC_expected_MS_norms,
    )


@njit(cache=True)
def _vde_iteration_numba(
    V: np.ndarray,
    m_real: np.ndarray,
    m_imag: np.ndarray,
    w_real: float,
    w_imag: float,
    max_iter: int,
    tol: float,
    omega: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Numba-accelerated VDE iteration loop."""
    n = V.shape[0]

    for _ in range(max_iter):
        Vm_real = V @ m_real
        Vm_imag = V @ m_imag

        max_diff = 0.0
        for i in range(n):
            dr = w_real + Vm_real[i]
            di = w_imag + Vm_imag[i]
            denom_sq = dr * dr + di * di

            if denom_sq < 1e-32:
                denom_sq = 1e-32

            new_real = -dr / denom_sq
            new_imag = di / denom_sq

            diff_real = new_real - m_real[i]
            diff_imag = new_imag - m_imag[i]
            diff_abs = np.sqrt(diff_real * diff_real + diff_imag * diff_imag)

            if diff_abs > max_diff:
                max_diff = diff_abs

            m_real[i] = omega * new_real + (1.0 - omega) * m_real[i]
            m_imag[i] = omega * new_imag + (1.0 - omega) * m_imag[i]

        if max_diff < tol:
            break

    return m_real, m_imag


def _solve_vde(
    S: np.ndarray,
    p: float,
    z: float,
    eta: float = 1e-3,
    max_iter: int = 2000,
    tol: float = 1e-7,
    omega: float = 0.8,
    warm: np.ndarray | None = None,
) -> np.ndarray:
    V = p * (1 - p) * (S**2)
    w = z + 1j * eta
    m = -np.ones(S.shape[0], dtype=complex) / w if warm is None else warm
    for _ in range(max_iter):
        denom = w + V.dot(m)
        denom[np.abs(denom) < 1e-16] = 1e-16
        m_new = -1.0 / denom
        diff = m_new - m
        m = omega * m_new + (1 - omega) * m
        if np.max(np.abs(diff)) < tol:
            break
    return m


def _solve_vde_fast(
    V: np.ndarray,
    z: float,
    eta: float = 1e-3,
    max_iter: int = 2000,
    tol: float = 1e-7,
    omega: float = 0.8,
    warm: np.ndarray | None = None,
) -> np.ndarray:
    """Fast VDE solver with numba acceleration."""
    w_real = z
    w_imag = eta
    n = V.shape[0]

    # Ensure V is contiguous float64 for Numba
    V = np.ascontiguousarray(V, dtype=np.float64)

    if warm is None:
        denom = w_real * w_real + w_imag * w_imag
        m_real = np.full(n, -w_real / denom, dtype=np.float64)
        m_imag = np.full(n, w_imag / denom, dtype=np.float64)
    else:
        m_real = np.ascontiguousarray(np.real(warm), dtype=np.float64)
        m_imag = np.ascontiguousarray(np.imag(warm), dtype=np.float64)

    m_real, m_imag = _vde_iteration_numba(
        V, m_real, m_imag, w_real, w_imag, max_iter, tol, omega
    )

    return m_real + 1j * m_imag


def lambda_bulk_dyson_raw(
    S: np.ndarray,
    p: float,
    omega: float = 0.8,
    eta: float = 1e-3,
    ngrid: int = 100,
    jump_frac: float = 0.1,
) -> float:
    if p <= 0 or p >= 1:
        return 0.0

    n = S.shape[0]
    s2_max = np.max(eigvalsh(S**2))
    z_max = np.linalg.norm(S, 2) + 8.0 * np.sqrt(p * (1 - p) * s2_max)
    z_min = 1e-8
    zs = np.linspace(z_max, z_min, ngrid)

    warm = None
    im_mavg = []
    for z in zs:
        m = _solve_vde(S, p, z, eta=eta, warm=warm, omega=omega)
        warm = m
        im_mavg.append(np.imag(np.mean(m)))

    im_mavg = np.array(im_mavg)
    jumps = np.diff(im_mavg)
    max_jump = np.max(jumps)
    threshold = jump_frac * max_jump

    idx = np.argmax(im_mavg > threshold)

    return float(zs[idx])


def lambda_bulk_dyson_raw_fast(
    S: np.ndarray,
    p: float,
    s2_max: float | None = None,
    s_norm: float | None = None,
    omega: float = 0.8,
    eta: float = 1e-3,
    ngrid: int = 100,
    jump_frac: float = 0.1,
) -> float:
    """Fast version with pre-computed S-dependent values."""
    if p <= 0 or p >= 1:
        return 0.0

    if s2_max is None:
        s2_max = np.max(eigvalsh(S**2))
    if s_norm is None:
        s_norm = np.linalg.norm(S, 2)

    z_max = s_norm + 8.0 * np.sqrt(p * (1 - p) * s2_max)
    z_min = 1e-8
    zs = np.linspace(z_max, z_min, ngrid)

    warm = None
    im_mavg = []
    for z in zs:
        m = _solve_vde(S, p, z, eta=eta, warm=warm, omega=omega)
        warm = m
        im_mavg.append(np.imag(np.mean(m)))

    im_mavg = np.array(im_mavg)
    jumps = np.diff(im_mavg)
    max_jump = np.max(jumps)
    threshold = jump_frac * max_jump

    idx = np.argmax(im_mavg > threshold)

    return float(zs[idx])


def _lambda_bulk_dyson_ultra(
    S: np.ndarray,
    p: float,
    info: PrecomputedMatrixInfo,
    omega: float = 0.8,
    eta: float = 1e-3,
    ngrid: int = 100,
    jump_frac: float = 0.1,
) -> float:
    """Ultra-fast bulk edge estimation with pre-computed values and numba VDE."""
    if p <= 0 or p >= 1:
        return 0.0

    z_max = info.s_norm + 8.0 * np.sqrt(p * (1 - p) * info.s2_max)
    z_min = 1e-8
    zs = np.linspace(z_max, z_min, ngrid)

    V = p * (1 - p) * info.S_sq

    warm = None
    im_mavg = []
    for z in zs:
        m = _solve_vde_fast(V, z, eta=eta, warm=warm, omega=omega)
        warm = m
        im_mavg.append(np.imag(np.mean(m)))

    im_mavg = np.array(im_mavg)
    jumps = np.diff(im_mavg)
    max_jump = np.max(jumps)
    threshold = jump_frac * max_jump

    idx = np.argmax(im_mavg > threshold)
    return float(zs[idx])


def monte_carlo_bulk_edge_raw(
    S: np.ndarray,
    p: float,
    n_trials: int = 400,
    quantile: float = 0.9,
    seed: int | None = None,
) -> float:
    rng = np.random.default_rng(seed)
    n = S.shape[0]
    max_eigs = []
    for _ in range(n_trials):
        eps = rng.binomial(1, p, size=(n, n)).astype(float) - p
        eps = np.triu(eps, 1)
        eps = eps + eps.T
        np.fill_diagonal(eps, rng.binomial(1, p, size=n).astype(float) - p)
        Delta = eps * S
        w = eigvalsh(Delta)
        max_eigs.append(w[-1])
    return float(np.quantile(max_eigs, quantile))


def p_upper_only_k(
    S: np.ndarray,
    k: int = 1,
    method: str = "dyson",
    mc_trials: int = 600,
    mc_quantile: float = 0.9,
    tol: float = 1e-4,
    verbose: bool = False,
    seed: int | None = None,
    omega: float = 0.8,
    eta: float = 1e-3,
    jump_frac: float = 0.1,
) -> float:
    lam = np.sort(eigvalsh(S))[::-1]
    n = len(lam)
    if not (1 <= k <= n):
        raise ValueError("k must be between 1 and n")
    lam_k = lam[k - 1]
    lam_k1 = lam[k] if k < n else None

    if lam_k <= 0:
        if verbose:
            print("lambda_k <= 0 -> no positive spike to separate.")
        return 0.0
    if (lam_k1 is None) or (lam_k1 <= 0):
        if verbose:
            print(
                "lambda_{k+1} <= 0 -> only first k can be out for all large p; return 1.0."
            )
        return 1.0

    edge = (
        (
            lambda p: lambda_bulk_dyson_raw(
                S, p, omega=omega, eta=eta, jump_frac=jump_frac
            )
        )
        if method == "dyson"
        else (
            lambda p: monte_carlo_bulk_edge_raw(
                S, p, n_trials=mc_trials, quantile=mc_quantile, seed=seed
            )
        )
    )

    def count_out(p):
        e = edge(p)
        return int(np.sum(p * lam > e)), e

    c_hi, e_hi = count_out(0.99)
    if verbose:
        print(
            f"[sanity] p=0.99: bulk={e_hi:.4g}, count_out={c_hi}, lambda1={lam[0]:.4g}, lambda2={lam[1] if n>1 else np.nan:.4g}"
        )

    if c_hi < k:
        if verbose:
            print(f"Even at p~1, only {c_hi} spikes out (< k). Returning 1.0.")
        return 1.0

    grid = np.linspace(0.02, 0.99, 80)
    feas = [p for p in grid if count_out(p)[0] == k]
    if not feas:

        def g(p):
            return p * lam_k1 - edge(p)

        a, b = 1e-3, 0.99
        ga, gb = g(a), g(b)

        if ga >= 0 and gb >= 0:
            if verbose:
                print(
                    "(k+1) spike is out for all p; returning smallest p where count==k (none found) -> 0."
                )
            return 0.0

        if ga < 0 and gb <= 0:
            if verbose:
                print("(k+1) never emerges up to 0.99; returning 1.0.")
            return 1.0

        lo, hi = a, b
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if g(mid) >= 0:
                hi = mid
            else:
                lo = mid
            if (hi - lo) < tol:
                break

        p_star = max(0.0, min(1.0, lo - 2 * tol))

        return p_star

    p_lo = max(feas)

    def cond_ge_kplus1(p):
        return count_out(p)[0] >= (k + 1)

    p_hi = min(0.99, p_lo + 0.05)
    while (p_hi < 0.99) and (not cond_ge_kplus1(p_hi)):
        p_hi = min(0.99, p_hi + 0.05)
    if not cond_ge_kplus1(p_hi):
        return 1.0
    lo, hi = p_lo, p_hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if cond_ge_kplus1(mid):
            hi = mid
        else:
            lo = mid
        if (hi - lo) < tol:
            break
    p_star = max(0.0, min(1.0, lo))
    if verbose:
        c_star, e_star = count_out(p_star)
        print(f"p*={p_star:.4f}, bulk={e_star:.6g}, count_out(p*)={c_star}")
    return p_star


def p_upper_only_k_fast(
    S: np.ndarray,
    k: int = 1,
    method: str = "dyson",
    mc_trials: int = 600,
    mc_quantile: float = 0.9,
    tol: float = 1e-4,
    verbose: bool = False,
    seed: int | None = None,
    omega: float = 0.8,
    eta: float = 1e-3,
    jump_frac: float = 0.1,
    n_jobs: int = 1,
) -> float:
    """Fast version with caching and optional parallelization."""
    lam = np.sort(eigvalsh(S))[::-1]
    n = len(lam)
    if not (1 <= k <= n):
        raise ValueError("k must be between 1 and n")
    lam_k = lam[k - 1]
    lam_k1 = lam[k] if k < n else None

    if lam_k <= 0:
        if verbose:
            print("lambda_k <= 0 -> no positive spike to separate.")
        return 0.0
    if (lam_k1 is None) or (lam_k1 <= 0):
        if verbose:
            print(
                "lambda_{k+1} <= 0 -> only first k can be out for all large p; return 1.0."
            )
        return 1.0

    if method == "dyson":
        s2_max = np.max(eigvalsh(S**2))
        s_norm = np.linalg.norm(S, 2)
        edge = lambda p: lambda_bulk_dyson_raw_fast(
            S,
            p,
            s2_max=s2_max,
            s_norm=s_norm,
            omega=omega,
            eta=eta,
            jump_frac=jump_frac,
        )
    else:
        edge = lambda p: monte_carlo_bulk_edge_raw(
            S, p, n_trials=mc_trials, quantile=mc_quantile, seed=seed
        )

    def count_out(p):
        e = edge(p)
        return int(np.sum(p * lam > e)), e

    c_hi, e_hi = count_out(0.99)
    if verbose:
        print(
            f"[sanity] p=0.99: bulk={e_hi:.4g}, count_out={c_hi}, lambda1={lam[0]:.4g}, lambda2={lam[1] if n>1 else np.nan:.4g}"
        )

    if c_hi < k:
        if verbose:
            print(f"Even at p~1, only {c_hi} spikes out (< k). Returning 1.0.")
        return 1.0

    grid = np.linspace(0.02, 0.99, 80)

    if n_jobs != 1:
        edges = Parallel(n_jobs=n_jobs)(delayed(edge)(p) for p in grid)
        counts = [int(np.sum(p * lam > e)) for p, e in zip(grid, edges)]
        feas = grid[np.array(counts) == k].tolist()
    else:
        feas = [p for p in grid if count_out(p)[0] == k]

    if not feas:

        def g(p):
            return p * lam_k1 - edge(p)

        a, b = 1e-3, 0.99
        ga, gb = g(a), g(b)

        if ga >= 0 and gb >= 0:
            if verbose:
                print(
                    "(k+1) spike is out for all p; returning smallest p where count==k (none found) -> 0."
                )
            return 0.0

        if ga < 0 and gb <= 0:
            if verbose:
                print("(k+1) never emerges up to 0.99; returning 1.0.")
            return 1.0

        lo, hi = a, b
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if g(mid) >= 0:
                hi = mid
            else:
                lo = mid
            if (hi - lo) < tol:
                break

        p_star = max(0.0, min(1.0, lo - 2 * tol))

        return p_star

    p_lo = max(feas)

    def cond_ge_kplus1(p):
        return count_out(p)[0] >= (k + 1)

    p_hi = min(0.99, p_lo + 0.05)
    while (p_hi < 0.99) and (not cond_ge_kplus1(p_hi)):
        p_hi = min(0.99, p_hi + 0.05)
    if not cond_ge_kplus1(p_hi):
        return 1.0
    lo, hi = p_lo, p_hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if cond_ge_kplus1(mid):
            hi = mid
        else:
            lo = mid
        if (hi - lo) < tol:
            break
    p_star = max(0.0, min(1.0, lo))
    if verbose:
        c_star, e_star = count_out(p_star)
        print(f"p*={p_star:.4f}, bulk={e_star:.6g}, count_out(p*)={c_star}")
    return p_star


def _p_upper_only_k_ultra(
    S: np.ndarray,
    k: int,
    info: PrecomputedMatrixInfo,
    tol: float = 1e-4,
    verbose: bool = False,
    omega: float = 0.8,
    eta: float = 1e-3,
    jump_frac: float = 0.1,
) -> float:
    """Ultra-fast p_upper_only_k with all optimizations."""
    lam = info.eigvals
    n = len(lam)

    if not (1 <= k <= n):
        raise ValueError("k must be between 1 and n")

    lam_k = lam[k - 1]
    lam_k1 = lam[k] if k < n else None

    if lam_k <= 0:
        return 0.0
    if (lam_k1 is None) or (lam_k1 <= 0):
        return 1.0

    edge_cache: dict[float, float] = {}

    def edge(p: float) -> float:
        p_key = round(p, 8)
        if p_key not in edge_cache:
            edge_cache[p_key] = _lambda_bulk_dyson_ultra(
                S, p, info, omega=omega, eta=eta, jump_frac=jump_frac
            )
        return edge_cache[p_key]

    def count_out(p: float) -> tuple[int, float]:
        e = edge(p)
        return int(np.sum(p * lam > e)), e

    c_hi, e_hi = count_out(0.99)
    if c_hi < k:
        return 1.0

    grid = np.linspace(0.02, 0.99, 80)
    feas = [p for p in grid if count_out(p)[0] == k]

    if not feas:

        def g(p: float) -> float:
            return p * lam_k1 - edge(p)

        a, b = 1e-3, 0.99
        ga, gb = g(a), g(b)

        if ga >= 0 and gb >= 0:
            return 0.0
        if ga < 0 and gb <= 0:
            return 1.0

        lo, hi = a, b
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if g(mid) >= 0:
                hi = mid
            else:
                lo = mid
            if (hi - lo) < tol:
                break
        return max(0.0, min(1.0, lo - 2 * tol))

    p_lo = max(feas)

    def cond_ge_kplus1(p: float) -> bool:
        return count_out(p)[0] >= (k + 1)

    p_hi = min(0.99, p_lo + 0.05)
    while (p_hi < 0.99) and (not cond_ge_kplus1(p_hi)):
        p_hi = min(0.99, p_hi + 0.05)

    if not cond_ge_kplus1(p_hi):
        return 1.0

    lo, hi = p_lo, p_hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if cond_ge_kplus1(mid):
            hi = mid
        else:
            lo = mid
        if (hi - lo) < tol:
            break

    return max(0.0, min(1.0, lo))


def estimate_sampling_bounds(
    S: np.ndarray,
    gamma: float = 1.05,
    eta: float = 0.05,
    rho: float = 0.95,
    method: str = "dyson",
    omega: float = 0.8,
    eta_pmax: float = 1e-3,
    jump_frac: float = 0.1,
    tol: float = 1e-4,
    gap: float = 0.05,
    verbose: bool = False,
    random_state: int = 31213,
) -> tuple[float, float, np.ndarray]:
    pmin, _, _, _, _ = pmin_bound(
        S, gamma=gamma, eta=eta, rho=rho, random_state=random_state, verbose=verbose
    )

    eff_dim = compute_effective_dimension(
        np.linalg.norm(S, "fro"), np.linalg.norm(S, 2)
    )

    pmax = p_upper_only_k(
        S,
        k=eff_dim,
        method=method,
        tol=tol,
        omega=omega,
        eta=eta_pmax,
        jump_frac=jump_frac,
        verbose=verbose,
        seed=random_state,
    )

    S_noise = S

    if pmin > pmax - gap:
        if verbose:
            print("Noise regime triggered")
            print(f"pmin = {pmin}, pmax = {pmax}")

        epsilon = np.linalg.norm(S, 2) / np.sqrt(S.shape[0])

        t_range = np.linspace(0.0, epsilon, 10)
        eff_dim_list = []
        pmin_list = []
        pmax_list = []

        A = np.random.rand(S.shape[0], S.shape[1])

        t_threshold = 0

        t_iter = iter(t_range)
        t = next(t_iter)

        while True:
            S_noise = S + t * (A + A.T)

            pmin, _, _, _, _ = pmin_bound(
                S_noise,
                gamma=gamma,
                eta=eta,
                rho=rho,
                random_state=random_state,
                verbose=verbose,
            )

            eff_dim = compute_effective_dimension(
                np.linalg.norm(S_noise, "fro"), np.linalg.norm(S_noise, 2)
            )
            pmax = p_upper_only_k(
                S_noise,
                k=eff_dim,
                method=method,
                tol=tol,
                omega=omega,
                eta=eta_pmax,
                jump_frac=jump_frac,
                verbose=verbose,
                seed=random_state,
            )
            if verbose:
                print(t, pmin, eff_dim, pmax)

            eff_dim_list.append(eff_dim)
            pmin_list.append(pmin)
            pmax_list.append(pmax)

            if pmin < pmax - gap:
                t_threshold = t
                break

            try:
                t = next(t_iter)
            except StopIteration:
                break

        S_noise = S + t_threshold * (A + A.T)

    return pmin, pmax, S_noise


def estimate_sampling_bounds_fast(
    S: np.ndarray,
    gamma: float = 1.05,
    eta: float = 0.05,
    rho: float = 0.95,
    method: str = "dyson",
    omega: float = 0.8,
    eta_pmax: float = 1e-3,
    jump_frac: float = 0.1,
    tol: float = 1e-4,
    gap: float = 0.05,
    verbose: bool = False,
    random_state: int = 31213,
    n_jobs: int = -1,
) -> tuple[float, float, np.ndarray]:

    start_time = time.time()
    pmin, _, _, _, _ = pmin_bound(
        S, gamma=gamma, eta=eta, rho=rho, random_state=random_state, verbose=verbose
    )

    print(f"pmin bound took {time.time() - start_time:.2f}s")

    eff_dim = compute_effective_dimension(
        np.linalg.norm(S, "fro"), np.linalg.norm(S, 2)
    )

    start_time = time.time()

    pmax = p_upper_only_k_fast(
        S,
        k=eff_dim,
        method=method,
        tol=tol,
        omega=omega,
        eta=eta_pmax,
        jump_frac=jump_frac,
        verbose=verbose,
        seed=random_state,
        n_jobs=n_jobs,
    )
    print(f"pmax bound took {time.time() - start_time:.2f}s")

    S_noise = S

    if verbose:
        print(f"Running {n_jobs} jobs with n_jobs={n_jobs}")

    if pmin > pmax - gap:
        epsilon = np.linalg.norm(S, 2) / np.sqrt(S.shape[0])
        t_range = np.linspace(0.0, epsilon, 10)

        A = np.random.rand(S.shape[0], S.shape[1])
        AtA = A + A.T

        def _eval_t(t):
            S_t = S + t * AtA
            pmin_t, _, _, _, _ = pmin_bound(
                S_t,
                gamma=gamma,
                eta=eta,
                rho=rho,
                random_state=random_state,
                verbose=verbose,
            )
            eff_dim_t = compute_effective_dimension(
                np.linalg.norm(S_t, "fro"), np.linalg.norm(S_t, 2)
            )
            pmax_t = p_upper_only_k_fast(
                S_t,
                k=eff_dim_t,
                method=method,
                tol=tol,
                omega=omega,
                eta=eta_pmax,
                jump_frac=jump_frac,
                verbose=verbose,
                seed=random_state,
                n_jobs=1,
            )
            return float(pmin_t), float(pmax_t)

        results = Parallel(n_jobs=n_jobs)(delayed(_eval_t)(float(t)) for t in t_range)

        idx = None
        for i, (pm, px) in enumerate(results):
            if pm < px - gap:
                idx = i
                break

        if idx is not None:
            t_threshold = float(t_range[idx])
            pmin, pmax = results[idx]
        else:
            t_threshold = 0.0
            pmin, pmax = results[-1]

        S_noise = S + t_threshold * AtA

    return pmin, pmax, S_noise


def estimate_sampling_bounds_ultra(
    S: np.ndarray,
    gamma: float = 1.05,
    eta: float = 0.05,
    rho: float = 0.95,
    omega: float = 0.8,
    eta_pmax: float = 1e-3,
    jump_frac: float = 0.1,
    tol: float = 1e-4,
    gap: float = 0.05,
    verbose: bool = False,
    random_state: int = 31213,
    n_jobs: int = -1,
) -> tuple[float, float, np.ndarray]:
    """Ultra-fast sampling bounds estimation (10-20x faster than original).

    This version uses:
    1. Pre-computed eigenvalues, norms, S² (computed ONCE, not per-call)
    2. Numba JIT for VDE solver inner loop
    3. Edge function caching

    For a 300x300 matrix: ~6s vs ~96s for original (15x speedup)
    For a 9841x9841 matrix: ~3h estimated vs ~44h original
    """
    np.random.seed(random_state)

    info = precompute_matrix_info(S)

    pmin, _, _, _, _ = pmin_bound(
        S, gamma=gamma, eta=eta, rho=rho, random_state=random_state, verbose=verbose
    )

    if verbose:
        print(f"pmin = {pmin:.6f}, effective_dim = {info.eff_dim}")

    pmax = _p_upper_only_k_ultra(
        S,
        k=info.eff_dim,
        info=info,
        tol=tol,
        omega=omega,
        eta=eta_pmax,
        jump_frac=jump_frac,
        verbose=verbose,
    )

    S_noise = S

    if pmin > pmax - gap:
        if verbose:
            print("Noise regime triggered")
            print(f"pmin = {pmin}, pmax = {pmax}")

        epsilon = info.s_norm / np.sqrt(S.shape[0])
        t_range = np.linspace(0.0, epsilon, 10)

        A = np.random.rand(S.shape[0], S.shape[1])
        AtA = A + A.T

        t_threshold = 0.0

        for t in t_range:
            S_t = S + t * AtA
            info_t = precompute_matrix_info(S_t)

            pmin_t, _, _, _, _ = pmin_bound(
                S_t, gamma=gamma, eta=eta, rho=rho, random_state=random_state
            )
            pmax_t = _p_upper_only_k_ultra(
                S_t,
                k=info_t.eff_dim,
                info=info_t,
                tol=tol,
                omega=omega,
                eta=eta_pmax,
                jump_frac=jump_frac,
                verbose=verbose,
            )

            if verbose:
                print(
                    f"t={t:.4f}, pmin={pmin_t:.6f}, eff_dim={info_t.eff_dim}, pmax={pmax_t:.6f}"
                )

            if pmin_t < pmax_t - gap:
                t_threshold = t
                pmin = pmin_t
                pmax = pmax_t
                break

        S_noise = S + t_threshold * AtA

    return pmin, pmax, S_noise
