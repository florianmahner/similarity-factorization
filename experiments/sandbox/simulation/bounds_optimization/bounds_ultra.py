"""Ultra-optimized sampling bound estimation for matrix completion.

This module provides dramatically faster versions of the bound estimation
algorithms by:
1. Pre-computing all S-dependent quantities once (eigenvalues, norms, S²)
2. Avoiding redundant eigenvalue decompositions
3. Caching edge function evaluations
4. Pre-computing V matrix once per p value
5. Numba JIT compilation for VDE solver inner loop

The VDE solver is the main computational bottleneck (called ~100 times per
lambda_bulk_dyson call, which is called ~80+ times per p_upper_only_k call).
Numba provides significant speedup for this inner loop.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import eigvalsh
from numba import njit
from joblib import Parallel, delayed
from typing import NamedTuple


class PrecomputedMatrixInfo(NamedTuple):
    """Pre-computed matrix-dependent values for bound estimation."""
    S_sq: np.ndarray       # S**2, shape (n, n)
    eigvals: np.ndarray    # eigenvalues of S, sorted descending
    s2_max: float          # max eigenvalue of S**2 = max(eigvals**2)
    s_norm: float          # spectral norm of S = max(|eigvals|)
    fro_norm: float        # Frobenius norm of S
    eff_dim: int           # effective dimension


def precompute_matrix_info(S: np.ndarray) -> PrecomputedMatrixInfo:
    """Pre-compute all S-dependent values needed for bound estimation.

    This function computes expensive values ONCE:
    - S² (element-wise squaring) - reused in VDE solver
    - eigvalsh(S) - for eigenvalue-based checks
    - eigvalsh(S²) - for s2_max (note: S² is element-wise, NOT matrix product)
    - spectral norm ||S||_2

    Important: S**2 is element-wise squaring (Hadamard), not matrix multiplication.
    So eigvalsh(S²) ≠ eigvalsh(S)². We must compute eigvalsh(S²) separately.
    """
    S_sq = S ** 2
    eigvals = np.sort(eigvalsh(S))[::-1]

    s_norm = np.linalg.norm(S, 2)
    s2_max = np.max(eigvalsh(S_sq))
    fro_norm = np.linalg.norm(S, 'fro')
    eff_dim = int(np.ceil((fro_norm / s_norm) ** 2))

    return PrecomputedMatrixInfo(
        S_sq=S_sq,
        eigvals=eigvals,
        s2_max=s2_max,
        s_norm=s_norm,
        fro_norm=fro_norm,
        eff_dim=eff_dim,
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
    """Numba-accelerated VDE iteration loop.

    Operates on real/imaginary parts separately for numba compatibility.
    """
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
    V: np.ndarray,
    z: float,
    eta: float = 1e-3,
    max_iter: int = 2000,
    tol: float = 1e-7,
    omega: float = 0.8,
    warm: np.ndarray | None = None,
) -> np.ndarray:
    """Solve the Vector Dyson Equation (numerically identical to original).

    This version uses numba for the inner iteration loop while maintaining
    exact numerical equivalence to pysrf.bounds._solve_vde.
    """
    w_real = z
    w_imag = eta
    n = V.shape[0]

    if warm is None:
        denom = w_real * w_real + w_imag * w_imag
        m_real = np.full(n, -w_real / denom, dtype=np.float64)
        m_imag = np.full(n, w_imag / denom, dtype=np.float64)
    else:
        m_real = np.real(warm).copy()
        m_imag = np.imag(warm).copy()

    m_real, m_imag = _vde_iteration_numba(
        V, m_real, m_imag, w_real, w_imag, max_iter, tol, omega
    )

    return m_real + 1j * m_imag


def _solve_vde_python(
    V: np.ndarray,
    z: float,
    eta: float = 1e-3,
    max_iter: int = 2000,
    tol: float = 1e-7,
    omega: float = 0.8,
    warm: np.ndarray | None = None,
) -> np.ndarray:
    """Pure Python VDE solver (for verification, identical to original)."""
    w = z + 1j * eta
    n = V.shape[0]
    m = -np.ones(n, dtype=complex) / w if warm is None else warm.copy()

    for _ in range(max_iter):
        denom = w + V.dot(m)
        denom[np.abs(denom) < 1e-16] = 1e-16
        m_new = -1.0 / denom
        diff = m_new - m
        m = omega * m_new + (1 - omega) * m
        if np.max(np.abs(diff)) < tol:
            break

    return m


def lambda_bulk_dyson_ultra(
    S: np.ndarray,
    p: float,
    info: PrecomputedMatrixInfo,
    omega: float = 0.8,
    eta: float = 1e-3,
    ngrid: int = 100,
    jump_frac: float = 0.1,
) -> float:
    """Ultra-fast bulk edge estimation using pre-computed values.

    Key optimizations over lambda_bulk_dyson_raw:
    - Uses pre-computed s2_max and s_norm (no eigvalsh calls)
    - Pre-computes V = p(1-p)S² once (not in original)
    - Otherwise identical algorithm for numerical equivalence
    """
    if p <= 0 or p >= 1:
        return 0.0

    z_max = info.s_norm + 8.0 * np.sqrt(p * (1 - p) * info.s2_max)
    z_min = 1e-8
    zs = np.linspace(z_max, z_min, ngrid)

    V = p * (1 - p) * info.S_sq

    warm = None
    im_mavg = []

    for z in zs:
        m = _solve_vde(V, z, eta=eta, warm=warm, omega=omega)
        warm = m
        im_mavg.append(np.imag(np.mean(m)))

    im_mavg = np.array(im_mavg)
    jumps = np.diff(im_mavg)
    max_jump = np.max(jumps)
    threshold = jump_frac * max_jump

    idx = np.argmax(im_mavg > threshold)
    return float(zs[idx])


def pmin_bound_ultra(
    S: np.ndarray,
    info: PrecomputedMatrixInfo,
    gamma: float = 1.05,
    eta: float = 0.05,
    rho: float = 0.95,
) -> tuple[float, float, float, float, np.ndarray]:
    """Ultra-fast pmin bound using pre-computed values.

    Returns the same values as pmin_bound for compatibility.
    Note: Uses raw effective_dimension (without ceiling) to match original.
    """
    n = S.shape[0]

    L_max = np.max(info.S_sq.sum(axis=1) - np.diag(S) ** 2)
    row_sq = info.S_sq.sum(axis=1) - np.diag(S) ** 2
    empirical_L_max = np.quantile(row_sq, rho)

    L_infty = 2 * np.max(np.abs(S))
    empirical_L_infty = 2 * np.quantile(np.abs(S), rho)

    effective_dimension = (info.fro_norm / info.s_norm) ** 2

    N_bernstein = (gamma * L_infty * info.s_norm) / (3 * L_max) + 1
    N_empirical = (gamma * empirical_L_infty * info.s_norm) / (3 * empirical_L_max) + 1
    N_empirical_alternative = (gamma * L_infty * info.s_norm) / (3 * empirical_L_max) + 1
    N_theory_upperbound = (gamma * info.s_norm) / 3 + 1

    D_bernstein = ((gamma * info.s_norm) ** 2 / (2 * L_max) + 1) / np.log(2 * n / eta)
    D_empirical = ((gamma * info.s_norm) ** 2 / (2 * empirical_L_max) + 1) / np.log(
        2 * effective_dimension / eta
    )
    D_theory_lowerbound = ((gamma**2 * info.s_norm) / (2 * empirical_L_max) + 1) / np.log(
        2 * n / eta
    )

    p_min = N_bernstein / D_bernstein
    p_min_empirical = N_empirical / D_empirical
    p_min_empirical_alternative = N_empirical_alternative / D_empirical
    p_min_lowerbound = N_theory_upperbound / D_theory_lowerbound

    return (
        p_min_empirical,
        p_min,
        p_min_lowerbound,
        p_min_empirical_alternative,
        np.zeros(0),
    )


def p_upper_only_k_ultra(
    S: np.ndarray,
    k: int,
    info: PrecomputedMatrixInfo,
    tol: float = 1e-4,
    verbose: bool = False,
    omega: float = 0.8,
    eta: float = 1e-3,
    jump_frac: float = 0.1,
    n_jobs: int = -1,
) -> float:
    """Ultra-fast p_upper_only_k with pre-computed values and parallel grid search.

    Key optimizations:
    - Uses pre-computed eigenvalues (no eigvalsh call)
    - Uses pre-computed s2_max and s_norm
    - Parallel evaluation of grid points
    - Caches edge function evaluations for bisection
    """
    lam = info.eigvals
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
            print("lambda_{k+1} <= 0 -> first k out for all large p; return 1.0.")
        return 1.0

    edge_cache: dict[float, float] = {}

    def edge(p: float) -> float:
        p_key = round(p, 8)
        if p_key not in edge_cache:
            edge_cache[p_key] = lambda_bulk_dyson_ultra(
                S, p, info, omega=omega, eta=eta, jump_frac=jump_frac
            )
        return edge_cache[p_key]

    def count_out(p: float) -> tuple[int, float]:
        e = edge(p)
        return int(np.sum(p * lam > e)), e

    c_hi, e_hi = count_out(0.99)
    if verbose:
        print(
            f"[sanity] p=0.99: bulk={e_hi:.4g}, count_out={c_hi}, "
            f"lambda1={lam[0]:.4g}, lambda2={lam[1] if n>1 else np.nan:.4g}"
        )

    if c_hi < k:
        if verbose:
            print(f"Even at p~1, only {c_hi} spikes out (< k). Returning 1.0.")
        return 1.0

    grid = np.linspace(0.02, 0.99, 80)

    if n_jobs != 1:
        edges = Parallel(n_jobs=n_jobs)(
            delayed(lambda_bulk_dyson_ultra)(S, p, info, omega, eta, 100, jump_frac)
            for p in grid
        )
        for p, e in zip(grid, edges):
            edge_cache[round(p, 8)] = e
        counts = [int(np.sum(p * lam > e)) for p, e in zip(grid, edges)]
        feas = [p for p, c in zip(grid, counts) if c == k]
    else:
        feas = [p for p in grid if count_out(p)[0] == k]

    if not feas:
        def g(p: float) -> float:
            return p * lam_k1 - edge(p)

        a, b = 1e-3, 0.99
        ga, gb = g(a), g(b)

        if ga >= 0 and gb >= 0:
            if verbose:
                print("(k+1) spike is out for all p; returning 0.")
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

    p_star = max(0.0, min(1.0, lo))
    if verbose:
        c_star, e_star = count_out(p_star)
        print(f"p*={p_star:.4f}, bulk={e_star:.6g}, count_out(p*)={c_star}")

    return p_star


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
    """Ultra-fast sampling bounds estimation.

    This is dramatically faster than estimate_sampling_bounds by:
    1. Pre-computing all S-dependent values once (eigenvalues, norms, S²)
    2. Avoiding redundant eigvalsh(S²) calls
    3. Parallel grid search over p-values
    4. Numba JIT for VDE solver inner loop
    5. Caching edge function evaluations

    Parameters
    ----------
    n_jobs : int, default=-1
        Number of parallel jobs for grid search. -1 uses all cores.
    """
    np.random.seed(random_state)

    info = precompute_matrix_info(S)

    pmin, _, _, _, _ = pmin_bound_ultra(S, info, gamma=gamma, eta=eta, rho=rho)

    if verbose:
        print(f"pmin = {pmin:.6f}, effective_dim = {info.eff_dim}")

    pmax = p_upper_only_k_ultra(
        S,
        k=info.eff_dim,
        info=info,
        tol=tol,
        omega=omega,
        eta=eta_pmax,
        jump_frac=jump_frac,
        verbose=verbose,
        n_jobs=n_jobs,
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

            pmin_t, _, _, _, _ = pmin_bound_ultra(S_t, info_t, gamma=gamma, eta=eta, rho=rho)
            pmax_t = p_upper_only_k_ultra(
                S_t,
                k=info_t.eff_dim,
                info=info_t,
                tol=tol,
                omega=omega,
                eta=eta_pmax,
                jump_frac=jump_frac,
                verbose=verbose,
                n_jobs=n_jobs,
            )

            if verbose:
                print(f"t={t:.4f}, pmin={pmin_t:.6f}, eff_dim={info_t.eff_dim}, pmax={pmax_t:.6f}")

            if pmin_t < pmax_t - gap:
                t_threshold = t
                pmin = pmin_t
                pmax = pmax_t
                break

        S_noise = S + t_threshold * AtA

    return pmin, pmax, S_noise
