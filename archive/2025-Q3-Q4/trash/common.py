import numpy as np
from sklearn.utils.extmath import randomized_svd, squared_norm
from typing import Optional

Array = np.ndarray


def norm(x: Array) -> Array:
    return np.sqrt(squared_norm(x))


def nndsvd(
    x: Array,
    rank: int,
    eps: float = np.finfo(float).eps,
    random_state: Optional[int] = None,
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


def explained_variance(s: Array, s_hat: Array, center: bool = False) -> float:
    """
    Compute the explained variance of the model based on the upper triangle
    (excluding the diagonal) of the similarity matrices.
    """

    s_upper = s[np.triu_indices(s.shape[0], k=1)]
    s_hat_upper = s_hat[np.triu_indices(s_hat.shape[0], k=1)]

    rss = np.sum((s_upper - s_hat_upper) ** 2)

    if center:
        tss = np.sum((s_upper - np.mean(s_upper)) ** 2)
    else:
        tss = np.sum(s_upper**2)

    if tss == 0:
        # Handle case where the variance of the original data is zero
        return 1.0 if rss == 0 else 0.0

    return 1 - (rss / tss)


def sse(s: Array, s_hat: Array) -> float:
    s_upper = s[np.triu_indices(s.shape[0], k=1)]
    s_hat_upper = s_hat[np.triu_indices(s_hat.shape[0], k=1)]
    return np.sum((s_upper - s_hat_upper) ** 2)


def frobenius_norm(s: Array, s_hat: Array) -> float:
    return np.linalg.norm(s - s_hat, ord="fro")


def relative_error(s: Array, s_hat: Array, eps: float = np.finfo(float).eps) -> float:
    return frobenius_norm(s, s_hat) / (np.linalg.norm(s, ord="fro") + eps)


def sort_by_sum_(w: Array) -> None:
    order = np.argsort(-np.sum(w, axis=0))
    w[:] = w[:, order]
