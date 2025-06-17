"""ADMM - exact same algorithm, simple structure."""

# TODO rewrite loss only based on nan mask entries and then print this explained variance or so / also determine convergence.
# TODO I probably need to do the correc train/val/test split so that we can then afterwards get a true estimate on the entire matrix?
# TODO I need to actually also still solve the median splitting problem!
# TODO when we do median splitting we already hold out certain entries. this is a mask and we need to mask it furtherr!
# TODO outsource different computations again, eg rank selection etc.

import numpy as np
import math
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from .utils import init_factor
from collections import defaultdict

try:
    import pyximport

    pyximport.install(language_level=3, setup_args={"include_dirs": np.get_include()})
    from .bsum_cython import update_w as update_w_cython

    USE_CYTHON = True
except ImportError:
    warnings.warn(
        "Cython implementation not available, falling back to Python implementation. "
        "Performance may be significantly slower.",
        UserWarning,
    )
    USE_CYTHON = False


def _quartic_root(a, b, c, d):
    """
    solve min_{x >= 0} g(x) = a/4 x^4 + b/3 x^3 + c/2 x^2 + d x (a == 4), e.g a four-order polynomial.
    See eq. (11)'
    """
    p = (3.0 * a * c - b**2) / (3.0 * a**2)
    q = (9.0 * a * b * c - 27.0 * a**2 * d - 2.0 * b**3) / (27.0 * a**3)

    # closed‑form minimiser of the 1‑D quartic upper‑bound
    if c > b**2 / (3.0 * a):  # three real roots, pick the one in (0, inf)
        delta = math.sqrt(q**2 / 4.0 + p**3 / 27.0)
        x_new = np.cbrt(q / 2.0 - delta) + np.cbrt(q / 2.0 + delta)
    else:  # one real root
        stmp = b**3 / (27.0 * a**3) - d / a
        x_new = np.cbrt(stmp)

    if x_new < 0.0:  # non neg constraint
        x_new = 0.0

    return x_new


def update_w(
    m: np.ndarray,
    x0: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
) -> np.ndarray:
    """Block successive upper bound minimization (Shi et al., 2016). Implementation of the algorithm.
    See TABLE 1 in 'Inexact Block Coordinate Descent Methods For Symmetric Nonnegative Matrix Factorization.'
    """
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

                new = _quartic_root(a, b, c, d)
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


def _get_update_w_function():
    """Return the appropriate update_w function (Cython or Python fallback)."""
    if USE_CYTHON:
        return update_w_cython
    else:
        return update_w


def update_v(m, s, w, lam, rho, min_val, max_val):
    ww = w @ w.T  # current reconstruction
    v = ww - lam / rho  # default (missing-entry) formula

    observed = m.astype(bool)
    v[observed] = (s[observed] + rho * ww[observed] - lam[observed]) / (1.0 + rho)

    # since this optimization problem is linear we can do a projection step here to respect the bounds
    if min_val is not None:
        v[v < min_val] = min_val
    if max_val is not None:
        v[v > max_val] = max_val

    return v


def update_lambda(lam, v, w, rho):
    return lam + rho * (v - w @ w.T)


class ADMM(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        rank=10,
        rho=1.0,
        max_outer=15,
        max_inner=40,
        tol=1e-4,
        verbose=False,
        init="random_sqrt",
        random_state=None,
    ):
        self.rank = rank
        self.rho = rho
        self.max_outer = max_outer
        self.max_inner = max_inner
        self.tol = tol
        self.verbose = verbose
        self.init = init
        self.random_state = random_state
        self._update_w_func = _get_update_w_function()

    def fit(self, x, y=None, mask=None, bounds=None):
        x = self._validate_data(x, ensure_2d=True)

        if mask is None:
            mask = np.ones_like(x)

        w = init_factor(x, self.rank, self.init, self.random_state)
        lam = np.zeros_like(x)
        min_val, max_val = bounds if bounds is not None else (None, None)

        history = defaultdict(list)

        for i in range(1, self.max_outer + 1):
            v = update_v(mask, x, w, lam, self.rho, min_val, max_val)
            t = v + lam / self.rho
            w = self._update_w_func(t, w, max_iter=self.max_inner, tol=self.tol)
            lam = update_lambda(lam, v, w, self.rho)

            # Exact same objective computation
            data_fit = np.linalg.norm(mask * (x - v), "fro") ** 2
            penalty = (self.rho / 2) * np.linalg.norm(v - w @ w.T, "fro") ** 2
            lagrangian = np.sum(lam * (v - w @ w.T))
            total_obj = data_fit + penalty + lagrangian

            evar = 1 - np.linalg.norm(mask * (x - w @ w.T), "fro") / np.linalg.norm(
                x * mask, "fro"
            )
            history["evar"].append(evar)
            history["data_fit"].append(data_fit)
            history["penalty"].append(penalty)
            history["lagrangian"].append(lagrangian)
            history["total_objective"].append(total_obj)
            history["rec_error"].append(np.linalg.norm((x - w @ w.T) * mask, "fro"))

            if np.linalg.norm(v - w @ w.T, "fro") < self.tol and self.tol > 0.0:
                break

        # Store results
        self.w_ = w
        self.components_ = w
        self.n_iter_ = i
        self.history_ = history

        return self

    def transform(self, x):
        check_is_fitted(self)
        x = self._validate_data(x, reset=False)
        return x @ self.components_

    def fit_transform(self, x, y=None, mask=None, bounds=None):
        """Returns w."""
        self.fit(x, y, mask, bounds)
        return self.w_

    def fit_w(self, s):
        # TODO intergrate into the fit transform somehow!
        x0 = init_factor(s, self.rank, self.init, self.random_state)
        self.w_ = self._update_w_func(s, x0, max_iter=self.max_inner, tol=self.tol)
        self.s_hat_ = self.w_ @ self.w_.T
        self.history_ = defaultdict(list)
        self.history_["rec_error"] = np.linalg.norm(s - self.s_hat_, "fro")
        return self.w_
