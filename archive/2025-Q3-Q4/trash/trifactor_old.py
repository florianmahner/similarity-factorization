import numpy as np
from ..srf.models.base import BaseNMF
from ..srf.models.metrics import explained_variance
from ..srf.models.nnls_block import nnlsm_blockpivot
from scipy.linalg import cho_factor, cho_solve, solve, eigh
from typing import Any

Array = np.ndarray

# TODO Consider implementing ADMM for alternating minimization.
# TODO Consider warming alpha every iteration eg alpha -> 1.01 alpha. In the limit W and H are then the same.
# TODO Implement an augmented matrix for the normal equations.


def update_a(s: Array, w: Array, h: Array, epsilon: float = 1e-12) -> Array:
    # NOTE we can update it differently without the inv, eg the cholesky way.
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


def update_a_ridge(
    s: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    lam: float = 1e-1,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Fast exact minimiser of
        ||S - W A Hᵀ||_F² + λ·||A - I||_F²
    using symmetric eigendecomposition.

    Parameters
    ----------
    s, w, h :  float64 arrays  (n × n, n × r, n × r)
    lam     :  ridge weight λ
    eps     :  tiny diagonal loading for safety.
    damp    :  0<damp≤1  — optional under-relaxation to stop oscillations.
    A_prev  :  previous A  (for under-relaxation).

    Returns
    -------
    a       :  (r × r) ndarray
    """
    r = w.shape[1]
    L = w.T @ w + eps * np.eye(r)
    R = h.T @ h + eps * np.eye(r)

    # eigendecompose once (SPD → real eigenpairs)
    d, p = eigh(l)  # eigh returns (eigvals, eigvecs)
    e, q = eigh(r)

    # right-hand side
    c = w.T @ s @ h + lam * np.eye(r)
    g = p.T @ c @ q  #   g = pᵀ c q

    # element-wise solve
    denom = d[:, None] * e[None, :] + lam  # (r × r) broadcast
    a_hat = g / denom

    # back-rotate
    a_new = p @ a_hat @ q.T

    return a_new


def update_k_ridge(s, w, h, lam=1e-1, eps=1e-12):
    """Fast exact minimiser of ||S-WK KᵀHᵀ||² + λ||K-I||²."""
    r = w.shape[1]

    L = w.T @ w + eps * np.eye(r)
    R = h.T @ h + eps * np.eye(r)
    Z = w.T @ s @ h

    # eigen-pairs
    d, P = eigh(L)  # L = P diag(d) Pᵀ
    e, Q = eigh(R)  # R = Q diag(e) Qᵀ

    rhs_hat = P.T @ (L @ Z @ R + lam * np.eye(r)) @ Q  # Pᵀ RHS Q
    denom = d[:, None] * e[None, :] + lam  # element-wise denominator
    K_hat = rhs_hat / denom  # solve element-wise
    K_new = P @ K_hat @ Q.T  # back-rotate
    return K_new


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
        mask: Array | None = None,
        update_a: bool = True,  # if false, we do not update a, so we have a = I
        a_update_scheme: str = "free",  # if true, we do not update a, so we have a = I
        freeze_a: int | None = None,  # update a after this many iterations
        lam: float = 1e-1,  # λ = how hard you pull A toward I
        **kwargs: Any,
    ) -> None:
        super().__init__(
            rank, max_iter, tol, random_state, init, verbose, eval_every, eps
        )
        self.alpha = alpha
        self.mask = mask
        self.update_a = update_a
        self.a_update_scheme = a_update_scheme
        self.lam = lam
        self.freeze_a = freeze_a

    def make_symmetric_(self, a: Array) -> None:
        a += a.T
        a *= 0.5

    def normalize_factors_(
        self, w: Array, h: Array, a: Array, eps: float = 1e-12
    ) -> None:
        """
        Remove the scale ambiguity:

            * column-wise max of W → 1, compensate on A rows
            * column-wise max of H → 1, compensate on A columns
        """
        k = w.shape[1]

        # rescale rows of A  (⇔ columns of W)
        for r in range(k):
            s = max(w[:, r].max(), eps)
            w[:, r] /= s
            a[r, :] *= s

        # rescale columns of A  (⇔ columns of H)
        for r in range(k):
            s = max(h[:, r].max(), eps)
            h[:, r] /= s
            a[:, r] *= s

    def fit(self, s: Array) -> "TrifactorCD":
        if self.alpha is None:
            self.alpha = np.max(s) ** 2

        w = self.init_factor(s)
        h = w.copy()
        a = np.eye(self.rank)

        if self.alpha is None:
            self.alpha = np.max(s) ** 2

        for it in range(1, self.max_iter + 1):

            if self.mask is not None:
                s_hat = w @ a @ h.T
                s_imp = self.mask * s + (1 - self.mask) * s_hat
            else:
                s_imp = s

            if self.update_a and (self.freeze_a is None or it >= self.freeze_a):
                if self.a_update_scheme == "free":
                    a = update_a(s_imp, w, h)
                elif self.a_update_scheme == "ridge":
                    # NOTE here we drive a towards the identity matrix with a ridge penalty
                    a = update_a_ridge(s_imp, w, h, self.lam)
                    # k = update_k_ridge(s_imp, w, h, self.lam)
                    # a = k @ k.T
                else:
                    raise ValueError("unknown a_update_scheme")

            w = update_w(s_imp, w, h, a, self.alpha)
            h = update_h(s_imp, w, h, a, self.alpha)

            d = np.sqrt(np.clip(np.diag(a), 1e-12, None))
            a /= d[:, None] * d[None, :]  #  (C-1)
            w *= d[None, :]
            h *= d[None, :]

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
