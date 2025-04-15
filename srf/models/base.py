import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_random_state
from tqdm import tqdm
from .metrics import frobenius_norm
from sklearn.utils.extmath import randomized_svd, squared_norm

Array = np.ndarray


def norm(x: Array) -> Array:
    return np.sqrt(squared_norm(x))


def nndsvd(
    x: Array, rank: int, eps: float = np.finfo(float).eps, random_state: int = None
) -> tuple[Array, Array]:
    """adapted from https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/_nmf.py"""

    u, s, v = randomized_svd(x, rank, random_state=random_state)
    w = np.zeros_like(u)
    h = np.zeros_like(v)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    w[:, 0] = np.sqrt(s[0]) * np.abs(u[:, 0])
    h[0, :] = np.sqrt(s[0]) * np.abs(v[0, :])

    for j in range(1, rank):
        x, y = u[:, j], v[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u_update = x_p / x_p_nrm
            v_update = y_p / y_p_nrm
            sigma = m_p
        else:
            u_update = x_n / x_n_nrm
            v_update = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(s[j] * sigma)
        w[:, j] = lbd * u_update
        h[j, :] = lbd * v_update

    w[w < eps] = 0
    h[h < eps] = 0

    np.abs(w, out=w)
    np.abs(h, out=h)
    return w, h


class BaseNMF(BaseEstimator, TransformerMixin, ABC):
    def __init__(
        self,
        rank=10,
        max_iter=1000,
        tol=1e-5,
        random_state=None,
        init="random",
        verbose=False,
        eval_every=100,
        eps=np.finfo(float).eps,
    ):
        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.eval_every = eval_every
        self.eps = eps
        self.init = init

    @abstractmethod
    def fit(self, X, y=None):
        """Fit the NMF model."""
        pass


    def init_factor(self, s: Array) -> Array:
        rng = check_random_state(self.random_state)
        if self.init == "random":
            factor = 0.01 * rng.rand(s.shape[0], self.rank)
        elif self.init == "random_sqrt":
            avg = np.sqrt(s.mean() / self.rank)
            factor = rng.rand(s.shape[0], self.rank) * avg
        elif self.init == "nndsvd":
            factor, _ = nndsvd(s, self.rank, self.eps, self.random_state)
        elif self.init == "nndsvdar":
            factor, _ = nndsvd(s, self.rank, self.eps, self.random_state)
            avg = s.mean()
            factor[factor == 0] = abs(avg * rng.standard_normal(size=len(factor[factor == 0])) / 100)
        else:
            raise ValueError(f"Invalid initialization method: {self.init}")

        return factor

    def relative_error(self, x: Array, x_hat: Array) -> float:
        return frobenius_norm(x, x_hat) / (np.linalg.norm(x, ord="fro") + self.eps)

    def init_progress_bar(self, total=None):
        """Initialize a progress bar to track iterations."""
        if self.verbose:
            self.pbar = tqdm(
                total=total or self.max_iter,
                desc="Optimizing:",
                ascii=[" ", "="],  # ' ' for empty, '=' for fill
                bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} iterations",
                ncols=200,  # Optional fixed width
                leave=False,
            )

    def print_progress(
        self,
        **kwargs,
    ):
        if self.verbose:
            self.pbar.set_postfix({k.capitalize(): v for k, v in kwargs.items()})
            self.pbar.update(1)

    def close_progress_bar(self):
        if self.verbose:
            self.pbar.close()

    def sort_by_sum_(self, w: Array) -> None:
        order = np.argsort(-np.sum(w, axis=0))
        w[:] = w[:, order]
