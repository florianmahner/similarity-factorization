import numpy as np
from .base import BaseNMF
from .metrics import explained_variance
from .nnls_block import nnlsm_blockpivot
from typing import Any

Array = np.ndarray

# TODO Consider implementing ADMM for alternating minimization.
# TODO Consider warming alpha every iteration eg alpha -> 1.01 alpha. In the limit W and H are then the same.
# TODO Implement an augmented matrix for the normal equations.


def update_a(s: Array, w: Array, h: Array, epsilon: float = 1e-12) -> Array:
    """
    update rule for a given s, w, and h.

    we have the optimality condition:
         (w.t @ w) * a * (h.t @ h) = w.t @ s @ h.

    the closed-form solution for a is:
         a = (w.t @ w)^{-1} * (w.t @ s @ h) * (h.t @ h)^{-1}.

    to compute a efficiently and stably without explicitly inverting large matrices,
    we first solve (w.t @ w) * x = w.t @ s @ h for x (i.e., x = (w.t @ w)^{-1} w.t @ s @ h),
    then obtain a by right-multiplying x with (h.t @ h)^{-1}.
    """
    # compute gram matrices, adding a small regularization to avoid singularity
    wtw = w.T @ w + epsilon * np.eye(w.shape[1])  # (r, r)
    hth = h.T @ h + epsilon * np.eye(h.shape[1])  # (r, r)
    wts = w.T @ s @ h  # (r, r)

    wts = w.T @ s @ h  # (r, r)

    # First, solve for the intermediate matrix X: (W.T @ W) * X = W.T @ S @ H
    x = np.linalg.solve(wtw, wts)
    # Then, A = X * (H.T @ H)^{-1}
    a = x @ np.linalg.inv(hth)
    return a


def update_w(
    s: np.ndarray, w: np.ndarray, h: np.ndarray, a: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Update step for W using stacked normal equations and NNLS.

    Solves: min_W ||S - WAH^T||_F^2 + alpha * ||W - H||_F^2
    """
    sqrt_alpha = np.sqrt(alpha)
    r = h.shape[1]

    # Design matrix shape (n + r, r)
    A_aug = np.vstack([h @ a.T, sqrt_alpha * np.eye(r)])  # (n, r)  # (r, r)

    # Target matrix shape (n + r, n)
    B_aug = np.vstack([s.T, sqrt_alpha * h.T])  # (n, n)  # (r, n)

    # Normal equations
    left = A_aug.T @ A_aug  # (r x r)
    right = A_aug.T @ B_aug  # (r x n)

    x_t = nnlsm_blockpivot(left, right, is_input_prod=True, init=w.T)[0]
    return x_t.T


def update_h(
    s: np.ndarray, w: np.ndarray, h: np.ndarray, a: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Update step for H using stacked normal equations and NNLS.

    Solves: min_H ||S - WAH^T||_F^2 + alpha * ||H - W||_F^2
    """
    sqrt_alpha = np.sqrt(alpha)
    r = w.shape[1]

    # Design matrix C and target A for stacked system
    A_aug = np.vstack([w @ a, sqrt_alpha * np.eye(r)])  # (n, r)  # (r, r)

    B_aug = np.vstack([s.T, sqrt_alpha * w.T])  # (n, n)  # (r, n)
    # Normal equations
    left = A_aug.T @ A_aug  # (r x r)
    right = A_aug.T @ B_aug  # (r x n)

    x_t = nnlsm_blockpivot(left, right, is_input_prod=True, init=h.T)[0]
    return x_t.T


def _normalize_factors_column(w: Array, h: Array) -> tuple[Array, Array]:
    """
    Column normalization: for matrices where both W and H are (n x r),
    normalize each column by scaling them using the geometric mean of the
    column norms.

    This ensures that the product of the column norms is preserved.
    """
    norms_w = np.linalg.norm(w, axis=0)
    norms_h = np.linalg.norm(h, axis=0)
    norms = np.sqrt(norms_w * norms_h)
    norms_w = np.where(norms_w == 0, 1, norms_w)
    norms_h = np.where(norms_h == 0, 1, norms_h)
    w = w * (norms / norms_w)
    h = h * (norms / norms_h)
    return w, h


class TrifactorCD(BaseNMF):
    """
    Tri-factor coordinate descent of the form  min ||S - W A H.T||_F^2 + alpha * ||W - H||_F^2
    """

    def __init__(
        self,
        rank: int,
        alpha: float | None = None,
        max_iter: int = 1000,
        tol: float = 1e-5,
        random_state: int | None = None,
        init: str = "random_sqrt",
        verbose: bool = False,
        eval_every: int = 100,
        eps: float = np.finfo(float).eps,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            rank, max_iter, tol, random_state, init, verbose, eval_every, eps
        )
        self.alpha = alpha

    def make_symmetric_(self, a: Array) -> None:
        a += a.T
        a *= 0.5

    def init_mixing(self) -> Array:
        """Initialize a symmetric mixing matrix with small random values"""
        a = 0.01 * np.random.randn(self.rank, self.rank)
        self.make_symmetric_(a)
        return a

    def normalize_factors_(self, w: Array, h: Array, a: Array) -> None:
        """normalization step to resolve scaling ambiguity:
        compute the l2 norm of each column of w (or h)
        # FIXME: this is not working right now. think about theory how to remove scaling ambiguity
        """
        norms = np.linalg.norm(w, axis=0) + 1e-10
        # normalize the columns of w and h
        w /= norms
        h /= norms  # since the regularizer pushes w and h to be similar, they get the same normalization
        a[:] = np.diag(norms) @ a

    def fit(self, s: Array) -> "TrifactorCD":
        if self.alpha is None:
            self.alpha = np.max(s) ** 2

        w = self.init_factor(s)
        h = w.copy()
        a = self.init_mixing()

        if self.alpha is None:
            self.alpha = np.max(s) ** 2

        for it in range(1, self.max_iter + 1):
            w = update_w(s, w, h, a, self.alpha)
            h = update_h(s, w, h, a, self.alpha)

            a = update_a(s, w, h)

            w, h = _normalize_factors_column(w, h)
            a = np.diag(np.linalg.norm(w, axis=0)) @ a

            self.make_symmetric_(a)

            # Compute the reconstruction error.
            s_hat = w @ a @ h.T
            obj = np.linalg.norm(s - s_hat, "fro") ** 2

            if self.verbose:
                explained_var = explained_variance(s, s_hat)
                print(
                    f"Iteration {it}, objective: {obj:.3f}, explained variance: {explained_var:.3f}",
                    end="\r",
                )
            if obj < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {it} with objective {obj:.4e}")
                break

        self.w_ = w
        self.h_ = h
        self.a_ = a
        self.s_hat_ = s_hat
        self.iter_ = it
        return self

    def fit_transform(self, s: Array) -> Array:
        self.fit(s)
        return self.w_
