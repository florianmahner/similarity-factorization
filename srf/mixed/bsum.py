"""
Implementation of the BSUM algorithm.
See TABLE 1 in 'Inexact Block Coordinate Descent Methods For Symmetric Nonnegative Matrix Factorization.'
"""

import numpy as np
import pyximport

pyximport.install()
from srf.mixed._bsum_fast import update_bsum_cython_fast
import math


def _cubic_root(a, b, c, d):
    """
    solve min_{x >= 0} g(x) = a/4 x^4 + b/3 x^3 + c/2 x^2 + d x (a == 4), e.g a four-order polynomial.
    See eq. (11)'
    """
    p = (3.0 * a * c - b**2) / (3.0 * a**2)
    q = (9.0 * a * b * c - 27.0 * a**2 * d - 2.0 * b**3) / (27.0 * a**3)

    # closed‑form minimiser of the 1‑D quartic upper‑bound
    if c > b**2 / (3.0 * a):  # three real roots → pick the one in (0, inf)
        delta = math.sqrt(q**2 / 4.0 + p**3 / 27.0)
        x_new = np.cbrt(q / 2.0 - delta) + np.cbrt(q / 2.0 + delta)
    else:  # one real root
        stmp = b**3 / (27.0 * a**3) - d / a
        x_new = np.cbrt(stmp)

    if x_new < 0.0:  # non neg constraint
        x_new = 0.0

    return x_new


def update_bsum(
    m: np.ndarray,
    x0: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
) -> np.ndarray:
    x = x0.copy()
    n, r = x.shape
    xtx = x.T @ x
    diag = np.einsum("ij,ij->i", x, x)
    a = 4.0

    for it in range(max_iter):
        max_delta = 0.0
        for i in range(n):
            for j in range(r):
                old = x[i, j]
                b = 12 * old
                c = 4 * ((diag[i] - m[i, i]) + xtx[j, j] + old * old)
                d = 4 * (x[i] @ xtx[:, j]) - 4 * (m[i] @ x[:, j])

                new = _cubic_root(a, b, c, d)
                delta = new - old
                if abs(delta) > max_delta:
                    max_delta = abs(delta)

                # steps 7 - 13 in TABLE 1
                diag[i] += new * new - old * old
                update_row = delta * x[i]
                xtx[j, :] += update_row
                xtx[:, j] += update_row
                xtx[j, j] += delta * delta
                x[i, j] = new

        if verbose:
            evar = 1 - (np.linalg.norm(m - x @ x.T, "fro") / np.linalg.norm(m, "fro"))
            print(f"it {it:3d}  evar {evar:.6f}", end="\r")

        if max_delta < tol and tol > 0.0:
            break

    return x


def update_bsum_fast(z, x0, max_iter=100, rng=None):
    """
    Python-side driver that prepares memory-contiguous buffers and calls the
    Cython core.
    """
    z = np.ascontiguousarray(z, dtype=np.float64)
    x = np.ascontiguousarray(x0, dtype=np.float64).copy()  # will be modified
    xxt = x @ x.T  # (n,n)
    vtmp = np.einsum("ij,ij->j", x, x)  # (r,)

    rng = np.random.default_rng() if rng is None else rng
    perm = np.vstack([rng.permutation(x.shape[0]) for _ in range(max_iter)]).astype(
        np.intp
    )

    violation = update_bsum_cython_fast(z, x, xxt, vtmp, perm, max_iter)
    return x, violation
