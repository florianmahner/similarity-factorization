import numpy as np
from .base import BaseNMF
from .metrics import sse, frobenius_norm
from .nnls_block import nnlsm_blockpivot

Array = np.ndarray


def solve_stacked_normal_eqs(
    s: Array, fixed: Array, alpha: float, update: Array
) -> Array:
    """
    We stack the matrices and rearrange the normal equations.
    see eq. 16 and 17 in Kuang et al. (2014)
    """
    sqrt_alpha = np.sqrt(alpha)
    rank = fixed.shape[1]
    # Build the stacked matrices:
    # a_stack has A^T on top and sqrt(alpha)*fixed.T on the bottom.
    a_stack = np.vstack([s.T, sqrt_alpha * fixed.T])
    # c_stack has fixed on top and sqrt(alpha)*I on the bottom.
    c_stack = np.vstack([fixed, sqrt_alpha * np.eye(rank)])

    # solve for x (which approximates the updated factor's transpose) via NNLS.
    # this basically solves the normal equations for the stacked matrices
    left = c_stack.T @ c_stack
    right = c_stack.T @ a_stack
    x_t = nnlsm_blockpivot(left, right, True, update.T)[0]
    # transpose back to get the updated factor.
    return x_t.T


def _compute_gradients(
    x: Array, w: Array, h: Array, alpha: float
) -> tuple[Array, Array]:
    """Gradients for the objective function wrt. the regularization term"""
    left_w = w.T @ w
    right_w = x @ w
    regularization = alpha * (h - w)
    grad_h = h @ left_w - right_w + regularization

    left_h = h.T @ h
    right_h = x @ h
    grad_w = w @ left_h - right_h - regularization
    return grad_w, grad_h


def _compute_objective(x: Array, w: Array, h: Array) -> float:
    # Compute objective functions following the paper by Kuang et al.
    frob_x = np.linalg.norm(x, "fro") ** 2
    trace_wh = np.trace(w.T @ (x @ h))
    trace_ww = np.trace((w.T @ w) @ (h.T @ h))
    obj = frob_x - 2 * trace_wh + trace_ww
    return obj


def _projected_grad_norm(grad_w: Array, w: Array, grad_h: Array, h: Array) -> float:
    mask_w = (grad_w <= 0) | (w > 0)
    mask_h = (grad_h <= 0) | (h > 0)
    grad_w = np.linalg.norm(grad_w[mask_w])
    grad_h = np.linalg.norm(grad_h[mask_h])
    return np.sqrt(grad_w**2 + grad_h**2)


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


class SymmetricCD(BaseNMF):
    def __init__(
        self,
        rank: int,
        alpha: float | None = None,
        max_iter: int = 1000,
        tol: float = 1e-5,
        random_state: int | None = None,
        init: str = "random",
        verbose: bool = False,
        eval_every: int = 100,
        eps: float = np.finfo(float).eps,
    ) -> None:
        super().__init__(
            rank, max_iter, tol, random_state, init, verbose, eval_every, eps
        )
        self.alpha = alpha

    def fit(
        self,
        s: Array,
    ) -> "SymmetricCD":
        """
        Implementation of the coordinate descent algorithm for Symmetric NMF by (Kuang et al., 2014)
        """
        self.init_progress_bar(self.max_iter)

        if self.alpha is None:
            self.alpha = np.max(s) ** 2

        w = self.init_factor(s)
        h = w.copy()

        initgrad = None

        for iter in range(1, self.max_iter + 1):
            # solve the non negative least squares problem for w with h fixed
            w = solve_stacked_normal_eqs(s, fixed=h, alpha=self.alpha, update=w)
            h = solve_stacked_normal_eqs(s, fixed=w, alpha=self.alpha, update=h)

            grad_w, grad_h = _compute_gradients(s, w, h, self.alpha)

            if iter == 1:
                initgrad = _projected_grad_norm(grad_w, w, grad_h, h)
                continue
            else:
                projnorm = _projected_grad_norm(grad_w, w, grad_h, h)

            obj = frobenius_norm(s, w @ h.T)
            if projnorm < self.tol * initgrad and self.tol > 0:
                break
            else:
                self.print_progress(
                    **{
                        "Diff": f"{projnorm:.4f}",
                        "Obj": f"{obj:.4f}",
                        "Diff": f"{projnorm / initgrad:.4f}",
                    }
                )

        # Handle alpha == 0 case
        if self.alpha == 0:
            w, h = _normalize_factors_column(w, h)

        self.objective_ = _compute_objective(s, w, h)
        self.iter_ = iter
        self.h_ = h
        self.w_ = w
        self.s_hat_ = w @ h.T

        self.close_progress_bar()

        return self

    def fit_transform(self, s: Array) -> Array:
        return self.fit(s).w_
