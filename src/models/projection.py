import numpy as np
from .base import BaseNMF
from .metrics import frobenius_norm
from dataclasses import dataclass

Array = np.ndarray


@dataclass(kw_only=True)
class ProjectionNMF(BaseNMF):
    max_iter: int = 1000
    tol: float = 1e-5
    random_state: int | None = None
    init: str = "random"
    verbose: bool = False
    eval_every: int = 100
    eps: float = np.finfo(float).eps

    def fit(
        self,
        x: Array,
    ) -> "ProjectionNMF":

        self.init_progress_bar(self.max_iter)
        w = self.init_w(x)
        error_init = frobenius_norm(x, w @ w.T @ x)
        previous_error = error_init
        xx = x @ x.T

        for iteration in range(1, self.max_iter + 1):
            w_prev = w.copy()
            # numerator of the multiplicative update
            a = 2 * xx @ w  # (m, r)

            # denominator of the multiplicative update (eg.(5) in linear and nonlinear paper, simplified version would be eq.(41))
            b = w @ (w.T @ a) + xx @ (w @ (w.T @ w))

            np.maximum(b, self.eps, out=b)  # (m, r)

            # multiplicative update
            w *= a / b  # (m, r)

            # Normalize W by the norm, eg same as the square root of the largest eigenvalue of W^T W.
            # w /= np.linalg.norm(w, ord=2)

            # Alternative normalization
            wxxw = w.T @ xx @ w
            # Compute normalization factor = sqrt( trace(WXXW) / trace(WXXW @ (w.T @ w)) )
            # this is important otherwise the algorithm oscillates
            norm_factor = np.sqrt(np.trace(wxxw) / np.trace(wxxw @ (w.T @ w)))
            w *= norm_factor

            error = frobenius_norm(x, w @ w.T @ x)
            diff = (previous_error - error) / error_init

            if diff < self.tol:
                break
            previous_error = error

            obj = error

            self.print_progress(
                **{
                    "Iteration": iteration,
                    "Objective": f"{obj:.5f}",
                    # "Diff": f"{diff:.5f}",
                }
            )

        self.sort_by_sum_(w)
        self.w_ = w
        self.x_hat_ = w @ w.T @ x
        self.frobenius_norm_ = obj

        return self

    def fit_transform(self, x: Array) -> Array:
        return self.fit(x).w_


@dataclass(kw_only=True)
class ProjectionKernelNMF(BaseNMF):
    """
    Projection NMF using a kernel matrix K where K(i, j) = <x_i, x_j> using
    some kernel function.
    """

    max_iter: int = 1000
    tol: float = 1e-5
    random_state: int | None = None
    init: str = "random"
    verbose: bool = False
    eval_every: int = 100
    eps: float = np.finfo(float).eps

    def separate_pos_neg(self, x: Array) -> tuple[Array, Array]:
        pos = (abs(x) + x) / 2
        neg = (abs(x) - x) / 2
        return pos, neg

    def fit(
        self,
        k: Array,
    ) -> "ProjectionKernelNMF":
        """
        k is the kernel matrix K where K(i, j) = <x_i, x_j> using
        some kernel function.
        """

        self.init_progress_bar(self.max_iter)
        # w = self.init_w(k)
        w = np.random.rand(k.shape[0], self.rank)

        error_init = frobenius_norm(k, w @ w.T @ k)
        previous_error = error_init

        for iteration in range(1, self.max_iter + 1):

            w_prev = w.copy()

            # while both are in theory the same the computations of pos neg is much
            # slower than if no negatives are present
            if k.min() < 0:
                pos, neg = self.separate_pos_neg(k)

                # Instead of the following we reorder to avoid n x n intermediate:
                ww = w @ w.T
                a = pos @ w + ww @ neg @ w
                b = neg @ w + ww @ pos @ w
                # a = pos @ w + w @ ((w.T @ neg) @ w)
                # b = neg @ w + w @ ((w.T @ pos) @ w)

            else:
                # (48) from Linear and nonlinear paper with orthogonality of W
                a = k @ w
                b = w @ (w.T @ a)
            np.maximum(b, 1e-12, out=b)

            # multiplicative update
            w *= a
            w /= b

            # Equation (54) from the paper
            # Alternatively we could normalize by the norm of w, same as sqrt of largest eigenvalue of w^T w, eg:
            # w /= np.linalg.norm(w, ord=2)
            wxw = w.T @ k @ w
            norm_factor = np.sqrt(np.trace(wxw) / np.trace(wxw @ (w.T @ w)))
            w *= norm_factor

            error = frobenius_norm(k, w @ w.T @ k)
            diff = frobenius_norm(w_prev, w) / np.linalg.norm(w_prev, ord="fro")

            if diff < self.tol and self.tol > 0:
                break

            self.print_progress(
                **{
                    "Iteration": iteration,
                    "Objective": f"{error:.5f}",
                    "Diff": f"{diff:.5f}",
                }
            )

        self.sort_by_sum_(w)
        self.w_ = w
        self.k_hat_ = w @ w.T @ k
        self.frobenius_norm_ = error
        self.close_progress_bar()

        return self

    def fit_transform(self, k: Array) -> Array:
        return self.fit(k).w_

    def transform(self, k: Array) -> Array:
        return self.w_ @ self.w_.T @ k
