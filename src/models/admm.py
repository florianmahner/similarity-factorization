"""
ADMM-based Symmetric Non-negative Matrix Factorization

This module implements the Alternating Direction Method of Multipliers (ADMM)
for symmetric non-negative matrix factorization, with support for missing entries
and optional bounded constraints.

"""

import math
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_symmetric, validate_data
from .utils import init_factor

NDArray = np.ndarray
OptionalFloat = float | None
OptionalArray = NDArray | None


def _get_missing_mask(x: NDArray, missing_values) -> NDArray:
    """
    Create a boolean mask indicating missing values.

    Similar to sklearn's _get_mask but adapted for our use case.
    """
    if missing_values is np.nan:
        return np.isnan(x)
    elif missing_values is None:
        # Handle None as missing value
        return pd.isna(x) if hasattr(x, "isna") else np.isnan(x)
    else:
        # Handle specific values (like 0.0, -999, etc.)
        return x == missing_values


def _validate_missing_values(missing_values: float | None):
    """Validate the missing_values parameter."""
    if missing_values is not np.nan and missing_values is not None:
        if not isinstance(missing_values, (int, float)):
            raise ValueError(
                f"missing_values must be np.nan, None, or a numeric value, "
                f"got {type(missing_values)}"
            )


def _quartic_root(a: float, b: float, c: float, d: float) -> float:
    """
    Solve min_{x >= 0} g(x) = a/4 x^4 + b/3 x^3 + c/2 x^2 + d x (a == 4),
    e.g a four-order polynomial. See eq. (11) in Shi et al. (2016).

    Args:
        a, b, c, d: Coefficients of the quartic polynomial

    Returns:
        Non-negative minimizer of the quartic polynomial
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
    m: NDArray,
    x0: NDArray,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
) -> NDArray:
    """
    Block successive upper bound minimization (Shi et al., 2016).
    Implementation of the algorithm. See TABLE 1 in "Inexact Block Coordinate
    Descent Methods For Symmetric Nonnegative Matrix Factorization".

    Args:
        m: Target symmetric matrix to factorize
        x0: Initial factor matrix
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        verbose: Whether to print progress

    Returns:
        Optimized factor matrix
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
    try:
        import pyximport

        pyximport.install(
            language_level=3, setup_args={"include_dirs": np.get_include()}
        )
        from .bsum import update_w

        return update_w
    except ImportError:
        # Silently fall back to Python implementation for multiprocessing compatibility
        return update_w


def update_v(
    observed_mask: NDArray,
    s: NDArray,
    w: NDArray,
    lam: NDArray,
    rho: float,
    bound_min: float,
    bound_max: float,
) -> NDArray:
    """
    Update auxiliary variable v in ADMM algorithm.

    Args:
        observed_mask: Binary observation mask
        s: Original data matrix
        w: Current factor matrix
        lam: Lagrange multipliers
        rho: Penalty parameter
        bound_min: Lower bound constraint
        bound_max: Upper bound constraint

    Returns:
        Updated auxiliary variable v
    """
    ww = w @ w.T  # current reconstruction
    v = ww - lam / rho  # default (missing-entry) formula

    observed = observed_mask.astype(bool)
    v[observed] = (s[observed] + rho * ww[observed] - lam[observed]) / (1.0 + rho)

    # since this optimization problem is linear we can do a projection step here to respect the bounds
    np.clip(v, bound_min, bound_max, out=v)

    return v


def update_lambda(lam: NDArray, v: NDArray, w: NDArray, rho: float) -> NDArray:
    """
    Update Lagrange multipliers in ADMM algorithm.

    Args:
        lam: Current Lagrange multipliers
        v: Auxiliary variable
        w: Factor matrix
        rho: Penalty parameter

    Returns:
        Updated Lagrange multipliers
    """
    return lam + rho * (v - w @ w.T)


class ADMM(TransformerMixin, BaseEstimator):
    """
    Symmetric Non-negative Matrix Factorization using ADMM.

    This class implements symmetric non-negative matrix factorization (SymNMF) using
    the Alternating Direction Method of Multipliers (ADMM). It can handle missing
    entries and optional bound constraints on the factorization.

    The algorithm solves: min_{w>=0,v} ||M o (S - v)||^2_F + rho/2 ||v - ww^T||^2_F
    subject to optional bounds on v, where M is an observation mask.

    Parameters:
        rank: Number of factors (dimensionality of the latent space)
        rho: ADMM penalty parameter controlling constraint enforcement
        max_outer: Maximum number of ADMM outer iterations
        max_inner: Maximum iterations for w-subproblem per outer iteration
        tol: Convergence tolerance for constraint violation
        verbose: Whether to print optimization progress
        init: Method for factor initialization ('random', 'random_sqrt', 'nndsvd', 'nndsvdar')
        random_state: Random seed for reproducible initialization
        missing_values: values to be treated as missing to mask the matrix (default: np.nan)

    Attributes:
        w_: Learned factor matrix w of shape (n_samples, rank)
        components_: Alias for w_ (sklearn compatibility)
        n_iter_: Number of ADMM iterations performed
        history_: Dictionary containing optimization metrics per iteration

    Examples:
        >>> # Basic usage with complete data
        >>> model = ADMM(rank=10, random_state=42)
        >>> w = model.fit_transform(similarity_matrix)
        >>> reconstruction = w @ w.T

        >>> # Usage with missing data (NaN values)
        >>> similarity_matrix[mask] = np.nan
        >>> model = ADMM(rank=10, missing_values=np.nan)
        >>> w = model.fit_transform(similarity_matrix)
    """

    def __init__(
        self,
        rank: int = 10,
        rho: float = 1.0,
        max_outer: int = 15,
        max_inner: int = 40,
        tol: float = 1e-4,
        verbose: bool = False,
        init: str = "random_sqrt",
        random_state: int | None = None,
        missing_values: float | None = np.nan,
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

    def _validate_parameters(self):
        """Validate input parameters."""
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        if self.rho <= 0:
            raise ValueError(f"rho must be positive, got {self.rho}")
        if self.max_outer <= 0:
            raise ValueError(f"max_outer must be positive, got {self.max_outer}")
        if self.max_inner <= 0:
            raise ValueError(f"max_inner must be positive, got {self.max_inner}")
        if self.tol < 0:
            raise ValueError(f"tol must be non-negative, got {self.tol}")
        _validate_missing_values(self.missing_values)

    def _compute_metrics(
        self, x: NDArray, v: NDArray, w: NDArray, lam: NDArray
    ) -> dict[str, float]:
        """Compute comprehensive optimization metrics for monitoring."""
        # Use the stored observation mask for metrics computation
        observed_mask = self._observation_mask

        data_fit = np.sum((observed_mask * (x - v)) ** 2)

        constraint_violation = v - w @ w.T
        penalty = (self.rho / 2.0) * np.linalg.norm(constraint_violation, "fro") ** 2
        lagrangian = np.sum(lam * constraint_violation)
        total_obj = data_fit + penalty + lagrangian
        rec_error = np.sqrt(np.sum((observed_mask * (x - w @ w.T)) ** 2))

        # Explained variance on observed entries only
        if observed_mask.any():
            observed_data = x[observed_mask]
            observed_reconstruction = (w @ w.T)[observed_mask]
            total_var = np.var(observed_data) * len(observed_data)
            if total_var > 0:
                evar = (
                    1.0
                    - np.sum((observed_data - observed_reconstruction) ** 2) / total_var
                )
            else:
                evar = 0.0
        else:
            evar = 0.0

        return {
            "total_objective": total_obj,
            "data_fit": data_fit,
            "penalty": penalty,
            "lagrangian": lagrangian,
            "rec_error": rec_error,
            "evar": evar,
            "n_observed": observed_mask.sum(),
            "n_missing": (~observed_mask).sum(),
        }

    def _fit_complete_data(self, x: NDArray):
        """Optimized fitting for complete data (no missing entries)."""
        x0 = init_factor(x, self.rank, self.init, self.random_state)
        total_iter = self.max_inner * self.max_outer

        update_w_func = _get_update_w_function()
        self.w_ = update_w_func(x, x0, max_iter=total_iter, tol=self.tol)

        self.components_ = self.w_
        self.n_iter_ = 1
        self.history_ = defaultdict(list)
        self.history_["rec_error"] = [np.linalg.norm(x - self.w_ @ self.w_.T, "fro")]
        evar = 1 - np.linalg.norm(x - self.w_ @ self.w_.T, "fro") / np.linalg.norm(
            x, "fro"
        )
        self.history_["evar"] = [evar]

        return self

    def _fit_missing_data(self, x: NDArray):
        """ADMM fitting for data with missing entries."""
        w = init_factor(x, self.rank, self.init, self.random_state)
        lam = np.zeros_like(x)
        bound_min, bound_max = x.min(), x.max()
        history = defaultdict(list)

        update_w_func = _get_update_w_function()

        for i in range(1, self.max_outer + 1):
            # Update auxiliary variable v
            v = update_v(
                self._observation_mask, x, w, lam, self.rho, bound_min, bound_max
            )

            # Update factor matrix w
            t = v + lam / self.rho
            w = update_w_func(t, w, max_iter=self.max_inner, tol=self.tol)

            # Update Lagrange multipliers
            lam = update_lambda(lam, v, w, self.rho)

            metrics = self._compute_metrics(x, v, w, lam)
            for key, value in metrics.items():
                history[key].append(value)

            if self.tol > 0.0 and np.linalg.norm(v - w @ w.T, "fro") < self.tol:
                break

            if self.verbose:
                print(
                    f"Iteration {i}/{self.max_outer}, "
                    f"Objective: {metrics['total_objective']:.3f}, "
                    f"Rec Error: {metrics['rec_error']:.3f}, "
                    f"Evar: {metrics['evar']:.3f}, "
                    f"Observed: {metrics['n_observed']}/{metrics['n_observed'] + metrics['n_missing']}"
                )

        self.w_ = w
        self.components_ = w
        self.n_iter_ = i
        self.history_ = history

        return self

    def fit(self, x, y=None):
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
        self._validate_parameters()

        x = validate_data(
            self,
            x,
            reset=True,
            ensure_all_finite="allow-nan" if self.missing_values is np.nan else True,
        )

        missing_mask = _get_missing_mask(x, self.missing_values)
        if np.all(missing_mask):
            raise ValueError(
                "No observed entries found in the data. All values are missing."
            )

        check_symmetric(missing_mask, raise_exception=True)
        observed_mask = ~missing_mask
        x[missing_mask] = 0.0

        x = check_symmetric(x, raise_exception=True)
        self._observation_mask = observed_mask
        self._missing_mask = missing_mask

        if np.all(observed_mask):
            return self._fit_complete_data(x)
        else:
            return self._fit_missing_data(x)

    def transform(self, x):
        """
        Project data onto the learned factor space.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_samples)
            Symmetric matrix to transform

        Returns
        -------
        w : array-like of shape (n_samples, rank)
            Transformed data of shape (n_samples, rank)
        """
        check_is_fitted(self)
        return self.w_

    def fit_transform(self, x, y=None):
        """
        Fit the model and return the learned factors.

        Args:
            x: Symmetric similarity matrix
            y: Ignored (present for sklearn compatibility)

        Returns:
            Learned factor matrix w of shape (n_samples, rank)
        """
        return self.fit(x, y).transform(x)

    def reconstruct(self, w=None):
        """
        Reconstruct the similarity matrix from factors.

        Args:
            w: Factor matrix to use for reconstruction.
               If None, uses the fitted factors.

        Returns:
            s_hat: Reconstructed similarity matrix
        """
        if w is None:
            check_is_fitted(self)
            w = self.w_

        return w @ w.T

    def score(self, x, y=None):
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
        reconstruction = self.reconstruct()
        mse = np.mean(
            (x[self._observation_mask] - reconstruction[self._observation_mask]) ** 2
        )
        return mse
