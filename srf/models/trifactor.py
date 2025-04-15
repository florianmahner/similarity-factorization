import numpy as np
from .base import BaseNMF
from .metrics import explained_variance
from .nnls_block import nnlsm_blockpivot
from typing import Any

Array = np.ndarray

# TODO Consider implementing ADMM for alternating minimization.
# TODO Consider warming alpha every iteration eg alpha -> 1.01 alpha. In the limit W and H are then the same.


def update_a(s: Array, w: Array, h: Array) -> Array:
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
    wtw = w.T @ w  # (r, r)
    hth = h.T @ h  # (r, r)
    wts = w.T @ s @ h  # (r, r)

    # First, solve for the intermediate matrix X: (W.T @ W) * X = W.T @ S @ H
    x = np.linalg.solve(wtw, wts)
    # Then, A = X * (H.T @ H)^{-1}
    a = x @ np.linalg.inv(hth)
    return a


def update_w(s: Array, w: Array, h: Array, a: Array, alpha: float) -> Array:
    """
    update rule for w (with fixed a and h).

    we aim to minimize:
         ||s - w a h.t||_f^2 + alpha * ||w - h||_f^2
    with the constraint w >= 0.

    by differentiating (and not ignoring the nonnegativity), we get the normal
    eqn:
         w * (a (h.t @ h) a.t + alpha * i) = s @ h @ a.t + alpha * h

    let:
         g_w = a (h.t @ h) a.t + alpha * i   (r x r)
         f_w = s @ h @ a.t + alpha * h         (n x r)

    so we have: w * g_w = f_w.

    to solve this nnls problem, we rewrite it in the form:
         g_w.t * x = f_w.t,
    where x = w.t.

    we then use nnlsm_blockpivot to solve for x (with x >= 0, which
    enforces nonnegativity on w) and finally set w = x.t.

    note: we pass g_w.t and f_w.t with is_input_prod=True since these matrices
    are the precomputed products (like a^t a and a^t b) needed by the solver.
    """
    r = w.shape[1]
    g_w = a @ (h.T @ h) @ a.T + alpha * np.eye(r)
    f_w = s @ h @ a.T + alpha * h

    x, _ = nnlsm_blockpivot(g_w.T, f_w.T, is_input_prod=True)
    return x.T


def update_h(s: Array, w: Array, h: Array, a: Array, alpha: float) -> Array:
    """
    update rule for h (with fixed a and w).

    we aim to minimize:
         ||s - w a h.t||_f^2 + alpha * ||w - h||_f^2
    with the constraint h >= 0.
    normal eqn:
        h * (a.t (w.t @ w) a + alpha * i) = s @ w @ a + alpha * w

    and solve like before for w
    """
    r = h.shape[1]

    g_h = a.T @ (w.T @ w) @ a + alpha * np.eye(r)
    f_h = s @ w @ a + alpha * w
    x, _ = nnlsm_blockpivot(g_h.T, f_h.T, is_input_prod=True)
    return x.T


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
