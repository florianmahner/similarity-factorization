"""
Symmetric Non-negative Matrix Factorization using Alternating Non-negative Least Squares

This module implements symmetric NMF using the Alternating Non-negative Least Squares
(ANLS) algorithm with block coordinate descent for solving the constrained normal equations.

References:
    Kim, J., & Park, H. (2011). Fast nonnegative matrix factorization: an active-set-like
    method and comparisons. SIAM Journal on Scientific Computing, 33(6), 3261-3281.
"""

from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_symmetric

from ._nnls_block import nnlsm_blockpivot
from .utils import frobenius_norm, init_factor

# Type aliases for better readability
Array = np.ndarray
OptionalFloat = Optional[float]


def solve_stacked_normal_eqs(
    s: Array, fixed: Array, alpha: float, update: Array
) -> Array:
    """
    Solve the stacked normal equations for symmetric NMF using non-negative least squares.

    This function constructs and solves the regularized non-negative least squares problem:
    min_{X≥0} ||AX - B||²_F where A and B are constructed from the current iterate and
    regularization parameter.

    Args:
        s: Symmetric data matrix of shape (n_samples, n_samples)
        fixed: Fixed factor matrix (either W or H) of shape (n_samples, rank)
        alpha: Regularization parameter controlling the symmetric constraint
        update: Current factor matrix to be updated of shape (n_samples, rank)

    Returns:
        Updated factor matrix of shape (n_samples, rank)

    Note:
        The stacking approach incorporates the symmetric regularization term
        α||W - H||²_F into the least squares formulation.
    """
    sqrt_alpha = np.sqrt(alpha)
    rank = fixed.shape[1]

    # Construct the stacked system: [S; √α·H^T] and [H; √α·I]
    a_stack = np.vstack([s.T, sqrt_alpha * fixed.T])
    c_stack = np.vstack([fixed, sqrt_alpha * np.eye(rank)])

    # Solve the normal equations: C^T C X^T = C^T A
    left = c_stack.T @ c_stack
    right = c_stack.T @ a_stack

    # Use block pivot NNLS to solve for X^T
    x_t = nnlsm_blockpivot(left, right, True, update.T)[0]

    return x_t.T


def _compute_gradients(
    x: Array, w: Array, h: Array, alpha: float
) -> Tuple[Array, Array]:
    """
    Compute the gradients of the symmetric NMF objective function.

    The objective function is:
    f(W,H) = ||X - WH^T||²_F + α||W - H||²_F

    Args:
        x: Symmetric data matrix of shape (n_samples, n_samples)
        w: Factor matrix W of shape (n_samples, rank)
        h: Factor matrix H of shape (n_samples, rank)
        alpha: Regularization parameter for symmetric constraint

    Returns:
        Tuple of (grad_w, grad_h) where each is of shape (n_samples, rank)

    Note:
        The gradients include both the reconstruction term and the symmetry
        regularization term α(W - H).
    """
    # Gradient w.r.t. W: ∇_W f = W(H^TH) - XH - α(H - W)
    left_w = w.T @ w
    right_w = x @ w
    regularization = alpha * (h - w)
    grad_h = h @ left_w - right_w + regularization

    # Gradient w.r.t. H: ∇_H f = H(W^TW) - X^TW + α(H - W)
    left_h = h.T @ h
    right_h = x @ h
    grad_w = w @ left_h - right_h - regularization

    return grad_w, grad_h


def _projected_grad_norm(grad_w: Array, w: Array, grad_h: Array, h: Array) -> float:
    """
    Compute the projected gradient norm for convergence assessment.

    The projected gradient accounts for the non-negativity constraints by only
    considering gradient components that are either negative or correspond to
    positive variables (i.e., the active set).

    Args:
        grad_w: Gradient w.r.t. W of shape (n_samples, rank)
        w: Current factor matrix W of shape (n_samples, rank)
        grad_h: Gradient w.r.t. H of shape (n_samples, rank)
        h: Current factor matrix H of shape (n_samples, rank)

    Returns:
        Combined projected gradient norm (scalar)

    Note:
        The projected gradient norm is a standard convergence criterion for
        constrained optimization problems with box constraints.
    """
    # Active set: components where gradient is negative OR variable is positive
    mask_w = (grad_w <= 0) | (w > 0)
    mask_h = (grad_h <= 0) | (h > 0)

    # Compute norms only over the active sets
    grad_w_norm = np.linalg.norm(grad_w[mask_w])
    grad_h_norm = np.linalg.norm(grad_h[mask_h])

    return np.sqrt(grad_w_norm**2 + grad_h_norm**2)


class SymmetricNMF(BaseEstimator, TransformerMixin):
    """
    Symmetric Non-negative Matrix Factorization using Alternating Non-negative Least Squares.

    This class implements symmetric NMF using the ANLS algorithm, which alternately solves
    regularized non-negative least squares problems for the factor matrices W and H while
    enforcing the symmetric constraint ||W - H||²_F.

    The algorithm minimizes:
    f(W,H) = ||X - WH^T||²_F + α||W - H||²_F
    subject to W ≥ 0, H ≥ 0

    The symmetric regularization term encourages W ≈ H, making the factorization
    approximately symmetric: X ≈ WW^T.

    Parameters:
        rank: Number of factors (dimensionality of the latent space)
        alpha: Regularization parameter for symmetric constraint. If None, set to max(X)²
        max_iter: Maximum number of alternating iterations
        tol: Relative tolerance for projected gradient norm convergence criterion
        random_state: Random seed for reproducible factor initialization
        init: Factor initialization method ('random', 'random_sqrt', 'nndsvd', 'nndsvdar')
        verbose: Whether to print convergence progress

    Attributes:
        w_: Learned factor matrix W of shape (n_samples, rank)
        h_: Learned factor matrix H of shape (n_samples, rank)
        components_: Alias for h_ (sklearn compatibility)
        n_iter_: Number of iterations performed
        reconstruction_err_: Final reconstruction error ||X - WH^T||_F
        history_: Dictionary containing optimization metrics per iteration

    Examples:
        >>> # Basic usage
        >>> model = SymmetricNMF(rank=10, random_state=42)
        >>> w = model.fit_transform(similarity_matrix)
        >>> reconstruction = w @ model.h_.T

        >>> # With custom regularization
        >>> model = SymmetricNMF(rank=5, alpha=1.0, verbose=True)
        >>> w = model.fit_transform(similarity_matrix)

    References:
        Kim, J., & Park, H. (2011). Fast nonnegative matrix factorization: an active-set-like
        method and comparisons. SIAM Journal on Scientific Computing, 33(6), 3261-3281.
    """

    def __init__(
        self,
        rank: int = 10,
        alpha: OptionalFloat = None,
        max_iter: int = 1000,
        tol: float = 1e-5,
        random_state: Optional[int] = None,
        init: str = "random",
        verbose: bool = False,
    ):
        self.rank = rank
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.verbose = verbose

    def _validate_parameters(self):
        """Validate input parameters."""
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        if self.alpha is not None and self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {self.max_iter}")
        if self.tol < 0:
            raise ValueError(f"tol must be non-negative, got {self.tol}")

    def _validate_input(self, x: Array) -> Array:
        """Validate and prepare input data."""
        x = check_symmetric(x, raise_exception=True)

        if x.shape[0] < self.rank:
            raise ValueError(
                f"rank ({self.rank}) cannot be larger than the number of samples ({x.shape[0]})"
            )

        return x

    def _initialize_factors(self, x: Array) -> Tuple[Array, Array, float]:
        """Initialize factor matrices and regularization parameter."""
        # Initialize factors
        w = init_factor(x, self.rank, self.init, self.random_state)
        h = w.copy()  # Start with symmetric initialization

        # Set regularization parameter
        if self.alpha is None:
            alpha = np.max(x) ** 2
        else:
            alpha = self.alpha

        return w, h, alpha

    def _update_factors(
        self, x: Array, w: Array, h: Array, alpha: float
    ) -> Tuple[Array, Array]:
        """Perform one iteration of alternating factor updates."""
        # Update W given fixed H
        w = solve_stacked_normal_eqs(x, fixed=h, alpha=alpha, update=w)

        # Update H given fixed W
        h = solve_stacked_normal_eqs(x, fixed=w, alpha=alpha, update=h)

        return w, h

    def _compute_convergence_metrics(
        self,
        x: Array,
        w: Array,
        h: Array,
        alpha: float,
        iter_num: int,
        initgrad: OptionalFloat,
    ) -> Tuple[float, float, float]:
        """Compute metrics for convergence assessment and monitoring."""
        # Compute gradients
        grad_w, grad_h = _compute_gradients(x, w, h, alpha)

        # Projected gradient norm for convergence
        projnorm = _projected_grad_norm(grad_w, w, grad_h, h)

        # Reconstruction error
        rec_error = frobenius_norm(x, w @ h.T)

        # Relative gradient change (for monitoring)
        if initgrad is not None and initgrad > 0:
            rel_grad_change = projnorm / initgrad
        else:
            rel_grad_change = np.inf

        return projnorm, rec_error, rel_grad_change

    def _check_convergence(self, projnorm: float, initgrad: OptionalFloat) -> bool:
        """Check if convergence criterion is satisfied."""
        if self.tol <= 0 or initgrad is None or initgrad == 0:
            return False
        return projnorm < self.tol * initgrad

    def fit(self, x: Array, y: Array = None) -> "SymmetricNMF":
        """
        Fit the symmetric NMF model to the data.

        Args:
            x: Symmetric data matrix of shape (n_samples, n_samples)
            y: Ignored (present for sklearn compatibility)

        Returns:
            self: Fitted estimator

        Raises:
            ValueError: If input validation fails or parameters are invalid
        """
        # Validate parameters and input
        self._validate_parameters()
        x = self._validate_input(x)

        # Initialize factors and parameters
        w, h, alpha = self._initialize_factors(x)

        # Initialize convergence tracking
        initgrad = None
        history = defaultdict(list)

        # Main optimization loop
        for iter_num in range(1, self.max_iter + 1):
            # Update factor matrices
            w, h = self._update_factors(x, w, h, alpha)

            # Compute convergence metrics
            projnorm, rec_error, rel_grad_change = self._compute_convergence_metrics(
                x, w, h, alpha, iter_num, initgrad
            )

            # Initialize reference gradient norm on first iteration
            if iter_num == 1:
                initgrad = projnorm
                continue

            # Record optimization history
            history["rec_error"].append(rec_error)
            history["diff"].append(rel_grad_change)
            history["projnorm"].append(projnorm)

            # Progress reporting
            if self.verbose:
                print(
                    f"Iteration {iter_num:4d}: Rec Error = {rec_error:.6f}, "
                    f"Rel Grad Change = {rel_grad_change:.6f}"
                )

            # Check convergence
            if self._check_convergence(projnorm, initgrad):
                if self.verbose:
                    print(f"Converged after {iter_num} iterations")
                break

        # Store final results
        self.w_ = w
        self.h_ = h
        self.components_ = h
        self.n_iter_ = iter_num
        self.reconstruction_err_ = rec_error
        self.history_ = dict(history)

        return self

    def transform(self, x: Array) -> Array:
        """
        Project data onto the learned factor space.

        Args:
            x: Symmetric matrix to transform of shape (n_samples, n_samples)

        Returns:
            Transformed data of shape (n_samples, rank)

        Note:
            This projects the input using x @ H^T where H is the learned component matrix.
        """
        check_is_fitted(self)
        x = check_symmetric(x, raise_exception=True)
        return x @ self.components_.T

    def fit_transform(self, x: Array, y: Array = None) -> Array:
        """
        Fit the model and return the learned W factor matrix.

        Args:
            x: Symmetric data matrix of shape (n_samples, n_samples)
            y: Ignored (present for sklearn compatibility)

        Returns:
            Learned factor matrix W of shape (n_samples, rank)
        """
        self.fit(x, y)
        return self.w_

    def reconstruct(self, w: Array = None, h: Array = None) -> Array:
        """
        Reconstruct the data matrix from factor matrices.

        Args:
            w: Factor matrix W. If None, uses the fitted W
            h: Factor matrix H. If None, uses the fitted H

        Returns:
            Reconstructed data matrix of shape (n_samples, n_samples)
        """
        if w is None:
            check_is_fitted(self)
            w = self.w_
        if h is None:
            check_is_fitted(self)
            h = self.h_

        return w @ h.T
