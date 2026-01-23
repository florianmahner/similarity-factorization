"""
SRF-based Symmetric Non-negative Matrix Factorization

This module implements the Alternating Direction Method of Multipliers (SRF)
for symmetric non-negative matrix factorization, with support for missing entries
and optional bounded constraints.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import (
    check_is_fitted,
    check_symmetric,
    validate_data,
)
from sklearn.utils._param_validation import Interval, StrOptions, Integral, Real
from scipy.optimize import brentq

try:
    from ._bsum import update_w as _update_w_impl
except ImportError:
    import warnings

    warnings.warn(
        "Cython implementation not available. Using Python implementation "
        "which is significantly slower. Compile Cython to improve performance.",
        RuntimeWarning,
        stacklevel=2,
    )
    from .model import update_w_python as _update_w_impl


def _solve_quartic_minimization(a: float, b: float, c: float, d: float) -> float:
    """
    Find x >= 0 minimizing quartic polynomial g(x) = a/4 x^4 + b/3 x^3 + c/2 x^2 + d x.

    This is a key subroutine in the BSUM algorithm for updating individual
    elements of the factor matrix W.

    Parameters
    ----------
    a, b, c, d : float
        Polynomial coefficients

    Returns
    -------
    root : float
        Non-negative minimizer of g(x)

    References
    ----------
    Shi et al. (2016), "Inexact Block Coordinate Descent Methods For
    Symmetric Nonnegative Matrix Factorization", Equation (11)
    """
    a, b, c, d = float(a), float(b), float(c), float(d)
    bb = b * b
    a2 = a * a
    p = (3.0 * a * c - bb) / (3.0 * a2)

    q = (9.0 * a * b * c - 27.0 * a2 * d - 2.0 * b * bb) / (27.0 * a2 * a)

    if c > bb / (3.0 * a):
        delta = math.sqrt(q * q * 0.25 + p * p * p / 27.0)
        root = math.cbrt(q * 0.5 - delta) + math.cbrt(q * 0.5 + delta)
    else:
        tmp = b * bb / (27.0 * a2 * a) - d / a
        root = math.cbrt(tmp)

    return max(0.0, root)


def _dot(a: np.ndarray, b: np.ndarray) -> float:
    """Compute dot product of two arrays. We use the naive dot-product to ensure
    numerical equivalence to the Cython implementation."""
    return sum(x * y for x, y in zip(a, b))


def _is_nan_marker(missing_values: float | None) -> bool:
    """Check if missing_values represents NaN (handles identity issues after cloning)."""
    return missing_values is np.nan or (
        isinstance(missing_values, float) and np.isnan(missing_values)
    )


def _get_missing_mask(x: np.ndarray, missing_values: float | None) -> np.ndarray:
    if _is_nan_marker(missing_values) or missing_values is None:
        return np.isnan(x)
    else:
        return x == missing_values


def _initialize_w(
    x: np.ndarray,
    rank: int,
    method: str = "random_sqrt",
    random_state: int | np.random.RandomState | None = None,
    eps: float = 1e-6,
    observed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Initialize factor matrix W for symmetric NMF.
    
    Parameters
    ----------
    x : ndarray of shape (n_samples, n_samples)
        Symmetric input matrix (used for scaling in some methods)
    rank : int
        Number of components (columns in w)
    method : {'random', 'random_sqrt', 'nndsvd', 'nndsvda', 'nndsvdar'}, \
             default='random_sqrt'
        Initialization strategy:
        
        - 'random' : Random Gaussian, scaled by sqrt(X.mean() / n_components)
        - 'random_sqrt' : Element-wise sqrt(|N(0,1)|), popular for SymNMF
        - 'nndsvd' : Non-Negative Double SVD (deterministic, sparse)
        - 'nndsvda' : NNDSVD with zeros filled by column average
        - 'nndsvdar' : NNDSVD with zeros filled by small random values
    random_state : int, RandomState or None
        Controls random number generation
    eps : float, default=1e-6
        Small constant added to avoid exact zeros
    observed_mask : ndarray of shape (n_samples, n_samples) or None, default=None
        Binary mask indicating observed entries in the similarity matrix.
        If None, all entries are considered observed.
    
    Returns
    -------
    w : ndarray of shape (n_samples, n_components)
        Initialized non-negative factor matrix
    """
    rng = np.random.RandomState(random_state)
    n_samples = x.shape[0]

    if method == "random":
        w = 0.1 * rng.rand(n_samples, rank)
    elif method == "random_sqrt":
        # Use nanmean to handle missing data
        observed_mean = (
            np.nanmean(x[observed_mask]) if observed_mask is not None else np.nanmean(x)
        )
        avg = 2.0 * np.sqrt(observed_mean / rank)
        w = rng.rand(n_samples, rank) * avg

    elif method in ("nndsvd", "nndsvda", "nndsvdar"):
        # Leverage sklearn's implementation
        from sklearn.decomposition._nmf import _initialize_nmf

        w, _ = _initialize_nmf(x, rank, init=method, random_state=random_state, eps=eps)

    else:
        raise ValueError(
            f"Invalid initialization method '{method}'. "
            f"Choose from: 'random', 'random_sqrt', 'nndsvd', 'nndsvda', 'nndsvdar'"
        )

    return w


def _validate_bounds(val) -> None:
    if val is None:
        return
    if not (isinstance(val, tuple) and len(val) == 2):
        raise ValueError("bounds must be a tuple (lower, upper) of floats or None")
    lo, up = val
    # allow None for either end
    if lo is None or up is None:
        return
    try:
        lo_f = float(lo)
        up_f = float(up)
    except (TypeError, ValueError):
        raise ValueError("bounds must be numeric or None")
    if lo_f > up_f:
        raise ValueError("lower bound cannot be greater than upper bound")


def update_w(
    m: np.ndarray,
    w0: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Block successive upper bound minimization (Shi et al., 2016).
    Implementation of the algorithm. See TABLE 1 in "Inexact Block Coordinate
    Descent Methods For Symmetric Nonnegative Matrix Factorization".

    Args:
        m: Target symmetric matrix to factorize
        x0: Initial factor matrix
        max_iter: Maximum number of iterations
        tol: Convergence tolerance

    Returns:
        Optimized factor matrix
    """
    w = w0.copy()
    n, r = w.shape
    xtx = w.T @ w
    diag = np.einsum("ij,ij->i", w, w)
    a = 4.0

    for it in range(max_iter):
        max_delta = 0.0
        for i in range(n):
            for j in range(r):
                old = w[i, j]
                b = 12.0 * old
                c = 4.0 * ((diag[i] - m[i, i]) + xtx[j, j] + old * old)
                d = 4.0 * _dot(w[i], xtx[:, j]) - 4.0 * _dot(m[i], w[:, j])
                new = _solve_quartic_minimization(a, b, c, d)

                delta = new - old
                if abs(delta) > max_delta:
                    max_delta = abs(delta)

                # steps 7 - 13 in TABLE 1
                diag[i] += new * new - old * old
                update_row = delta * w[i]
                xtx[j, :] += update_row
                xtx[:, j] += update_row
                xtx[j, j] += delta * delta
                w[i, j] = new

        if max_delta < tol and tol > 0.0:
            break

    return w


def update_v_frobenius_(
    observed_mask: np.ndarray,
    x: np.ndarray,
    x_hat: np.ndarray,
    lam: np.ndarray,
    rho: float,
    bound_min: float,
    bound_max: float,
    v: np.ndarray,
) -> None:
    """
    Update auxiliary variable v in SRF algorithm.

    Args:
        observed_mask: Binary observation mask
        x: Original similarity matrix
        x_hat: Current estimate of the similarity matrix
        lam: Lagrange multipliers
        rho: Penalty parameter
        bound_min: Lower bound constraint
        bound_max: Upper bound constraint
        v: Auxiliary variable to be updated in place
    """
    v[:] = x_hat - (lam / rho)
    idx = observed_mask
    if idx.any():
        v[idx] = (x[idx] + rho * x_hat[idx] - lam[idx]) / (1.0 + rho)

    if bound_min > -np.inf or bound_max < np.inf:
        np.clip(v, bound_min, bound_max, out=v)

    v += v.T
    v *= 0.5


def update_v_kullback_leibler_(
    observed_mask: np.ndarray,
    x: np.ndarray,
    x_hat: np.ndarray,
    lam: np.ndarray,
    rho: float,
    bound_min: float,
    bound_max: float,
    v: np.ndarray,
) -> None:
    """
    Update V for Generalized Kullback-Leibler Divergence.
    Solves: min D_KL(X || V) + rho/2 ||V - Beta||^2
    Result is a quadratic equation: rho*v^2 + (1 - rho*Beta)*v - x = 0
    """
    # --- 1. Unobserved Entries (Standard ADMM projection) ---
    # v = beta = x_hat - lam/rho
    v[:] = x_hat - (lam / rho)

    # --- 2. Observed Entries (Quadratic Solution) ---
    idx = observed_mask
    if idx.any():
        x_obs = x[idx]
        lam_obs = lam[idx]
        x_hat_obs = x_hat[idx]

        # beta = x_hat - lam/rho
        beta = x_hat_obs - (lam_obs / rho)

        # Quadratic coeffs for: rho*v^2 + (1 - rho*beta)*v - x = 0
        # a = rho (always positive)
        b = 1.0 - rho * beta
        c = -x_obs

        # Determinant: sqrt(b^2 - 4ac)
        # Since a>0 and c<=0 (because x>=0), 4ac <= 0, so delta >= b^2 >= 0.
        # Roots are real. We need the positive root.
        delta = np.sqrt(b * b - 4.0 * rho * c)

        # Root = (-b + delta) / 2a
        v[idx] = (-b + delta) / (2.0 * rho)

    # --- 3. Symmetrize (KL is usually unbounded or >0) ---
    # KL requires v > 0. The quadratic solution naturally provides v >= 0 if x >= 0.
    # We clamp to epsilon to be safe against log(0) later.
    np.clip(v, max(bound_min, 1e-12), bound_max, out=v)
    v += v.T
    v *= 0.5


def update_v_bce_(
    observed_mask: np.ndarray,
    x: np.ndarray,
    x_hat: np.ndarray,
    lam: np.ndarray,
    rho: float,
    bound_min: float,
    bound_max: float,
    v: np.ndarray,
) -> None:
    """
    Robust Vectorized Newton-Raphson for BCE.
    Solves f'(v) = -x/v + (1-x)/(1-v) + rho*(v - beta) = 0
    """
    # 1. Unobserved entries: Standard ADMM projection (Frobenius penalty only)
    v[:] = x_hat - (lam / rho)

    # 2. Observed entries: Newton solver
    idx = observed_mask
    if idx.any():
        x_obs = x[idx]
        beta = x_hat[idx] - (lam[idx] / rho)

        # Initial Guess: 0.5 is safe for BCE (avoid boundaries)
        # Or projected beta clamped to (eps, 1-eps)
        v_k = np.clip(beta, 0.01, 0.99)

        # 5 iterations of Newton-Raphson is usually sufficient for machine precision
        # f(v) = rho(v - beta) - x/v + (1-x)/(1-v) (derivative of Lagrangian)
        # f'(v) = rho + x/v^2 + (1-x)/(1-v)^2      (Hessian of Lagrangian)

        for _ in range(5):
            # Safe calculation to prevent zero division
            v_k_inv = 1.0 / v_k
            v_k_inv_sq = v_k_inv * v_k_inv

            v_comp = 1.0 - v_k
            v_comp_inv = 1.0 / v_comp
            v_comp_inv_sq = v_comp_inv * v_comp_inv

            # Gradient value (f)
            grad = rho * (v_k - beta) - (x_obs * v_k_inv) + ((1.0 - x_obs) * v_comp_inv)

            # Hessian value (f_prime)
            hess = rho + (x_obs * v_k_inv_sq) + ((1.0 - x_obs) * v_comp_inv_sq)

            # Newton step
            diff = grad / hess
            v_k -= diff

            # Safety clamp between steps to ensure v remains in (0, 1)
            # This is crucial for BCE validity
            np.clip(v_k, 1e-9, 1.0 - 1e-9, out=v_k)

        v[idx] = v_k

    # 3. Global Bounds and Symmetry
    np.clip(v, bound_min, bound_max, out=v)
    v += v.T
    v *= 0.5


def update_lambda_(
    lam: np.ndarray, v: np.ndarray, x_hat: np.ndarray, rho: float
) -> None:
    """
    Update Lagrange multipliers in SRF algorithm.

    Args:
        lam: Current Lagrange multipliers
        v: Auxiliary variable
        x_hat: Current matrix estimate (w @ w.T)
        rho: Penalty parameter
    """
    lam += rho * (v - x_hat)


class SRF(TransformerMixin, BaseEstimator):
    """
    Symmetric Non-negative Matrix Factorization using SRF.

    This class implements symmetric non-negative matrix factorization (SymNMF) using
    the Alternating Direction Method of Multipliers (SRF). It can handle missing
    entries and optional bound constraints on the factorization.

    The algorithm solves: min_{w>=0,v} ||M o (S - v)||^2_F + rho/2 ||v - ww^T||^2_F
    subject to optional bounds on v, where M is an observation mask.

    Parameters
    ----------
    rank : int, default=10
        Number of factors (dimensionality of the latent space)
    rho : float, default=3.0
        SRF penalty parameter controlling constraint enforcement
    max_outer : int, default=10
        Maximum number of SRF outer iterations
    max_inner : int, default=30
        Maximum iterations for w-subproblem per outer iteration
    tol : float, default=1e-4
        Convergence tolerance for constraint violation
    verbose : int, default=0
        Whether to print optimization progress
    init : str, default='random_sqrt'
        Method for factor initialization ('random', 'random_sqrt', 'nndsvd',
        'nndsvdar', 'eigenspectrum')
    random_state : int or None, default=None
        Random seed for reproducible initialization
    missing_values : float or None, default=np.nan
        Values to be treated as missing to mask the matrix
    bounds : tuple of (float, float) or None, default=(None, None)
        Tuple of (lower, upper) bounds for the auxiliary variable v.
        If None, the bounds are inferred from the data.
        In practice, one can also pass the expected bounds of the matrix
        (e.g. (0, 1) for cosine similarity)

    Attributes
    ----------
    w_ : np.ndarray of shape (n_samples, rank)
        Learned factor matrix w
    components_ : np.ndarray of shape (n_samples, rank)
        Alias for w_ (sklearn compatibility)
    n_iter_ : int
        Number of SRF iterations performed
    history_ : dict
        Dictionary containing optimization metrics per iteration

    Examples
    --------
    >>> # Basic usage with complete data
    >>> from pysrf import SRF
    >>> model = SRF(rank=10, random_state=42)
    >>> w = model.fit_transform(similarity_matrix)
    >>> reconstruction = w @ w.T

    >>> # Usage with missing data (NaN values)
    >>> similarity_matrix[mask] = np.nan
    >>> model = SRF(rank=10, missing_values=np.nan)
    >>> w = model.fit_transform(similarity_matrix)
    """

    _parameter_constraints = {
        "rank": [Interval(Integral, 1, None, closed="left")],
        "rho": [Interval(Real, 0.0, None, closed="left")],
        "max_outer": [Interval(Integral, 1, None, closed="left")],
        "max_inner": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0.0, None, closed="left")],
        "init": [
            StrOptions({"random", "random_sqrt", "nndsvd", "nndsvda", "nndsvdar"})
        ],
        "verbose": [Interval(Integral, 0, None, closed="left")],
        "random_state": ["random_state"],  # sklearn's special validator
        "missing_values": [None, Real, np.nan],
        "bounds": [None, tuple],
        "loss": [StrOptions({"frobenius", "kullback-leibler", "bce"})],
    }

    def __init__(
        self,
        rank: int = 10,
        rho: float = 3.0,
        max_outer: int = 30,
        max_inner: int = 20,
        tol: float = 1e-4,
        verbose: int = 0,
        init: str = "random_sqrt",
        random_state: int | None = None,
        missing_values: float | None = np.nan,
        bounds: tuple[float, float] | None = (None, None),
        loss: str = "frobenius",
    ) -> None:
        self.rank = rank
        self.rho = rho
        self.max_outer = max_outer
        self.max_inner = max_inner
        self.tol = tol
        self.verbose = verbose
        self.init = init
        self.random_state = random_state
        self.missing_values = missing_values
        self.bounds = bounds
        self.loss = loss

    def _print_metrics(self, iteration: int, metrics: dict):
        """Print metrics in a formatted way."""
        if self.verbose > 0:
            print(
                f"Iter {iteration:4d}/{self.max_outer}: "
                f"Obj={metrics['data_fit']:.2f} "
                f"Primal={metrics['primal_residual']:.2f} "
                f"Dual={metrics['dual_residual']:.2f} "
                f"MSE={metrics['mse']:.4f} "
                f"R\u00b2={metrics['quality']:.3f}",  # FIXME check if disabling or not
                end="\r",
            )

    def _compute_metrics(
        self,
        x: np.ndarray,
        v: np.ndarray,
        x_hat: np.ndarray,
        lam: np.ndarray,
        v_old: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute optimization metrics for monitoring and convergence."""
        mask = self._observation_mask
        primal_residual = v - x_hat

        x_obs = x[mask]
        x_hat_obs = x_hat[mask]

        # Loss-specific data fit term (use x_hat for objective, v for ADMM constraint)
        if self.loss == "frobenius":
            data_fit = np.sum((x_obs - x_hat_obs) ** 2)
        elif self.loss == "kullback-leibler":
            v_obs = np.clip(v[mask], 1e-10, np.inf)
            data_fit = np.sum(x_obs * np.log(x_obs / v_obs) - x_obs + v_obs)
        elif self.loss == "bce":
            # Standard BCE loss: -[s*log(v) + (1-s)*log(1-v)]
            # v is a probability in (0, 1)

            # Clip v_obs to a safe range for logging
            eps = 1e-10
            v_obs = np.clip(v[mask], eps, 1.0 - eps)

            data_fit = -np.sum(
                x_obs * np.log(v_obs) + (1.0 - x_obs) * np.log(1.0 - v_obs)
            )
        # Universal metrics (same for both losses)
        mse = np.mean((x_obs - x_hat_obs) ** 2)
        total_var = np.var(x_obs)
        quality = 1.0 - mse / total_var if total_var > 0 else 0.0

        primal_res = np.linalg.norm(primal_residual)
        dual_res = self.rho * np.linalg.norm(v - v_old) if v_old is not None else np.inf

        return {
            "data_fit": data_fit,
            "mse": mse,
            "quality": quality,
            "primal_residual": primal_res,
            "dual_residual": dual_res,
        }

    def _check_convergence(
        self,
        metrics: dict[str, float],
        v: np.ndarray,
        x_hat: np.ndarray,
        lam: np.ndarray | None,
    ) -> bool:
        """Check ADMM convergence using primal and dual residuals."""
        eps_abs = self.tol
        eps_rel = 1e-4
        n = v.size

        eps_pri = np.sqrt(n) * eps_abs + eps_rel * max(
            np.linalg.norm(v), np.linalg.norm(x_hat)
        )

        if lam is not None:
            eps_dual = np.sqrt(n) * eps_abs + eps_rel * np.linalg.norm(lam)
            return (
                metrics["primal_residual"] <= eps_pri
                and metrics["dual_residual"] <= eps_dual
            )
        else:
            return metrics["primal_residual"] <= eps_pri

    def _fit_complete_data(self, x: np.ndarray) -> SRF:
        """Fit model with complete data (no missing values)."""
        w = _initialize_w(
            x, self.rank, self.init, self.random_state, observed_mask=self._observation_mask
        )
        history = defaultdict(list)

        for i in range(1, self.max_outer + 1):
            # Update w and compute x_hat
            w = _update_w_impl(x, w, max_iter=self.max_inner, tol=self.tol)
            x_hat = w @ w.T

            # For complete data: v=x (original), so primal_residual = x - x_hat
            metrics = self._compute_metrics(x, x, x_hat, lam=None)

            for key, value in metrics.items():
                history[key].append(value)

            self._print_metrics(i, metrics)

            if self._check_convergence(metrics, x, x_hat, lam=None):
                if self.verbose > 0:
                    print(f"\nConverged at iteration {i}")
                break

        self.w_ = w
        self.components_ = w
        self.n_iter_ = i
        self.history_ = history

        return self

    def _fit_missing_data(self, x: np.ndarray) -> SRF:
        """Fit model with missing data using ADMM."""
        # Set bounds for KL loss or default to data bounds
        # Bounds setup
        if self.loss == "kullback-leibler":
            default_min, default_max = 1e-10, np.inf
        else:
            default_min = np.nanmin(x) if self._missing_mask.any() else x.min()
            default_max = np.nanmax(x) if self._missing_mask.any() else x.max()

        if self.bounds is None:
            bound_min, bound_max = default_min, default_max
        else:
            bound_min = self.bounds[0] or default_min
            bound_max = self.bounds[1] or default_max

        if self.loss == "bce":
            # For BCE, V is a probability and must be in (eps, 1-eps)
            # Use a small epsilon for numerical stability in the solver.
            eps = 1e-8
            bound_min = eps
            bound_max = 1.0 - eps

            # Check if user-provided bounds are valid
            if self.bounds is not None:
                if self.bounds[0] is not None:
                    bound_min = max(bound_min, self.bounds[0])
                if self.bounds[1] is not None:
                    bound_max = min(bound_max, self.bounds[1])

        history = defaultdict(list)

        w = _initialize_w(
            x, self.rank, self.init, self.random_state, observed_mask=self._observation_mask
        )
        lam = np.zeros_like(x)
        x_hat = w @ w.T

        v = x_hat.copy()
        v = 0.5 * (v + v.T)
        v[self._observation_mask] = x[self._observation_mask]

        update_v_impl = {
            "frobenius": update_v_frobenius_,
            "kullback-leibler": update_v_kullback_leibler_,
            "bce": update_v_bce_,
        }[self.loss]

        for i in range(1, self.max_outer + 1):
            v_old = v.copy()

            # Update w and x_hat
            w = _update_w_impl(
                v + lam / self.rho, w, max_iter=self.max_inner, tol=self.tol
            )
            np.matmul(w, w.T, out=x_hat)

            update_v_impl(
                self._observation_mask,
                x,
                x_hat,
                lam,
                self.rho,
                bound_min,
                bound_max,
                v,
            )
            update_lambda_(lam, v, x_hat, self.rho)

            # Compute metrics using _compute_metrics
            metrics = self._compute_metrics(x, v, x_hat, lam, v_old)

            # Store metrics in history
            for key, value in metrics.items():
                history[key].append(value)

            # Print verbose output
            self._print_metrics(i, metrics)

            # Convergence check
            if i > 1 and self._check_convergence(metrics, v, x_hat, lam):
                if self.verbose > 0:
                    print(f"\nConverged at iteration {i}")
                break

        self.w_ = w
        self.components_ = w
        self.n_iter_ = i
        self.history_ = history

        return self

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> SRF:
        """
        Fit the symmetric NMF model to the data.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_samples)
            Symmetric similarity matrix. Missing values are allowed and should
            be marked according to the missing_values parameter.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._validate_params()
        _validate_bounds(self.bounds)

        x = validate_data(
            self,
            x,
            reset=True,
            ensure_all_finite=(
                "allow-nan" if _is_nan_marker(self.missing_values) else True
            ),
            ensure_2d=True,
            dtype=np.float64,
            copy=True,
        )

        self._missing_mask = _get_missing_mask(x, self.missing_values)
        if np.all(self._missing_mask):
            raise ValueError(
                "No observed entries found in the data. All values are missing."
            )

        check_symmetric(self._missing_mask, raise_exception=True)
        self._observation_mask = ~self._missing_mask
        x[self._missing_mask] = 0.0
        x = check_symmetric(x, raise_exception=True, tol=1e-10)

        if np.all(self._observation_mask):
            return self._fit_complete_data(x)
        else:
            return self._fit_missing_data(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Project data onto the learned factor space.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_samples)
            Symmetric matrix to transform

        Returns
        -------
        w : array-like of shape (n_samples, rank)
            Transformed data
        """
        check_is_fitted(self)
        return self.w_

    def fit_transform(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """
        Fit the model and return the learned factors.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_samples)
            Symmetric similarity matrix
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        w : array-like of shape (n_samples, rank)
            Learned factor matrix
        """
        return self.fit(x, y).transform(x)

    def reconstruct(self, w: np.ndarray | None = None) -> np.ndarray:
        """
        Reconstruct the similarity matrix from factors.

        Parameters
        ----------
        w : array-like of shape (n_samples, rank) or None
            Factor matrix to use for reconstruction.
            If None, uses the fitted factors.

        Returns
        -------
        s_hat : array-like of shape (n_samples, n_samples)
            Reconstructed similarity matrix
        """
        if w is None:
            check_is_fitted(self)
            w = self.w_

        return w @ w.T

    def score(self, x: np.ndarray, y: np.ndarray | None = None) -> float:
        """
        Score the model using reconstruction error on observed entries only.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_samples)
            Symmetric similarity matrix. Missing values are allowed and should
            be marked according to the missing_values parameter.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        mse : float
            Mean squared error of the reconstruction on observed entries.
        """
        check_is_fitted(self)

        x = validate_data(
            self,
            x,
            reset=False,
            ensure_2d=True,
            dtype=np.float64,
            ensure_all_finite="allow-nan" if self.missing_values is np.nan else True,
        )
        observation_mask = ~_get_missing_mask(x, self.missing_values)
        reconstruction = self.reconstruct()
        mse = np.mean((x[observation_mask] - reconstruction[observation_mask]) ** 2)
        return -mse
