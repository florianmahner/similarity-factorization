"""Symmetric NMF - exact same algorithm, simple structure."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from .utils import init_factor, frobenius_norm
from ._nnls_block import nnlsm_blockpivot
from collections import defaultdict


def solve_stacked_normal_eqs(s, fixed, alpha, update):
    """Keep exact same function from symmetric.py."""
    sqrt_alpha = np.sqrt(alpha)
    rank = fixed.shape[1]
    a_stack = np.vstack([s.T, sqrt_alpha * fixed.T])
    c_stack = np.vstack([fixed, sqrt_alpha * np.eye(rank)])
    left = c_stack.T @ c_stack
    right = c_stack.T @ a_stack
    x_t = nnlsm_blockpivot(left, right, True, update.T)[0]
    return x_t.T


def _compute_gradients(x, w, h, alpha):
    """Keep exact same from symmetric.py."""
    left_w = w.T @ w
    right_w = x @ w
    regularization = alpha * (h - w)
    grad_h = h @ left_w - right_w + regularization
    left_h = h.T @ h
    right_h = x @ h
    grad_w = w @ left_h - right_h - regularization
    return grad_w, grad_h


def _projected_grad_norm(grad_w, w, grad_h, h):
    """Keep exact same from symmetric.py."""
    mask_w = (grad_w <= 0) | (w > 0)
    mask_h = (grad_h <= 0) | (h > 0)
    grad_w_norm = np.linalg.norm(grad_w[mask_w])
    grad_h_norm = np.linalg.norm(grad_h[mask_h])
    return np.sqrt(grad_w_norm**2 + grad_h_norm**2)


class SymmetricNMF(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        rank=10,
        alpha=None,
        max_iter=1000,
        tol=1e-5,
        random_state=None,
        init="random",
        verbose=False,
    ):
        self.rank = rank
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.verbose = verbose

    def fit(self, x, y=None):
        """Exact same algorithm from ANLS class."""
        x = self._validate_data(x, ensure_2d=True)

        if self.alpha is None:
            alpha = np.max(x) ** 2
        else:
            alpha = self.alpha

        w = init_factor(x, self.rank, self.init, self.random_state)
        h = w.copy()

        initgrad = None
        history = defaultdict(list)

        for iter_num in range(1, self.max_iter + 1):
            w = solve_stacked_normal_eqs(x, fixed=h, alpha=alpha, update=w)
            h = solve_stacked_normal_eqs(x, fixed=w, alpha=alpha, update=h)

            grad_w, grad_h = _compute_gradients(x, w, h, alpha)

            if iter_num == 1:
                initgrad = _projected_grad_norm(grad_w, w, grad_h, h)
                continue
            else:
                projnorm = _projected_grad_norm(grad_w, w, grad_h, h)

            obj = frobenius_norm(x, w @ h.T)
            history["rec_error"].append(obj)
            history["diff"].append(projnorm / initgrad)

            if projnorm < self.tol * initgrad and self.tol > 0:
                break

        # Store results
        self.w_ = w
        self.h_ = h
        self.components_ = h
        self.n_iter_ = iter_num
        self.reconstruction_err_ = obj
        self.history_ = history

        return self

    def transform(self, x):
        check_is_fitted(self)
        x = self._validate_data(x, reset=False)
        return x @ self.components_.T

    def fit_transform(self, x, y=None):
        """Returns w."""
        self.fit(x, y)
        return self.w_
