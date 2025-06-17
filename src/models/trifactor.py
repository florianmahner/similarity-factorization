"""TriFactor - exact same algorithm, simple structure."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from .utils import init_factor
from ._nnls_block import nnlsm_blockpivot
from scipy.linalg import cho_factor, cho_solve
from collections import defaultdict


# Keep all the exact same functions from trifactor.py
def update_a_cholesky(s, w, h, eps=1e-12):
    """Exact same from trifactor.py."""
    wtw = w.T @ w + eps * np.eye(w.shape[1])
    hth = h.T @ h + eps * np.eye(h.shape[1])
    m = w.T @ s @ h
    c_w, lower_w = cho_factor(wtw, overwrite_a=False, check_finite=False)
    c_h, lower_h = cho_factor(hth, overwrite_a=False, check_finite=False)
    x = cho_solve((c_w, lower_w), m, overwrite_b=False, check_finite=False)
    a = cho_solve((c_h, lower_h), x.T, overwrite_b=False, check_finite=False).T
    return a


def update_w(s, w, h, a, alpha):
    """Exact same from trifactor.py."""
    sqrt_alpha = np.sqrt(alpha)
    r = h.shape[1]
    a_aug = np.vstack([h @ a.T, sqrt_alpha * np.eye(r)])
    b_aug = np.vstack([s.T, sqrt_alpha * h.T])
    left = a_aug.T @ a_aug
    right = a_aug.T @ b_aug
    x_t = nnlsm_blockpivot(left, right, is_input_prod=True, init=w.T)[0]
    return x_t.T


def update_h(s, w, h, a, alpha):
    """Exact same from trifactor.py."""
    sqrt_alpha = np.sqrt(alpha)
    r = w.shape[1]
    a_aug = np.vstack([w @ a, sqrt_alpha * np.eye(r)])
    b_aug = np.vstack([s.T, sqrt_alpha * w.T])
    left = a_aug.T @ a_aug
    right = a_aug.T @ b_aug
    x_t = nnlsm_blockpivot(left, right, is_input_prod=True, init=h.T)[0]
    return x_t.T


class TriFactor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        rank=10,
        alpha=1.0,
        init="random_sqrt",
        max_iter=300,
        verbose=False,
        random_state=None,
        lam=0.01,
        update_a=True,
    ):
        self.rank = rank
        self.alpha = alpha
        self.init = init
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state
        self.lam = lam
        self.update_a = update_a

    def fit(self, x, y=None):
        """Exact same algorithm from TriFactor class."""
        x = self._validate_data(x, ensure_2d=True)

        w = init_factor(x, self.rank, self.init, self.random_state)
        h = init_factor(x, self.rank, self.init, self.random_state)
        a = np.eye(self.rank)

        history = defaultdict(list)

        for iter_num in range(1, self.max_iter + 1):
            if self.update_a:
                a = update_a_cholesky(x, w, h)

            w = update_w(x, w, h, a, self.alpha)
            h = update_h(x, w, h, a, self.alpha)

            reconstruction = w @ a @ h.T
            error = np.linalg.norm(x - reconstruction, "fro")
            history["rec_error"].append(error)

            if error < 1e-6:  # Simple convergence
                break

        # Store results
        self.w_ = w
        self.h_ = h
        self.a_ = a
        self.components_ = h
        self.n_iter_ = iter_num
        self.reconstruction_err_ = error
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
