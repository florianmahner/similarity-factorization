from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_random_state

array = np.ndarray


def get_missing_mask(x: array, missing_values: float | None) -> array:
    if missing_values is np.nan:
        return np.isnan(x)
    elif missing_values is None:
        return np.isnan(x)
    else:
        return x == missing_values


def validate_missing_values(missing_values: float | None) -> None:
    """Validate the missing_values parameter."""
    if missing_values is not np.nan and missing_values is not None:
        if not isinstance(missing_values, (int, float)):
            raise ValueError(
                f"missing_values must be np.nan, None, or a numeric value, got {type(missing_values)}"
            )


def init_factor(
    s: array,
    rank: int,
    init: str,
    random_state: int | None = None,
    eps: float = np.finfo(float).eps,
) -> array:
    """Initialize factor matrix using specified method."""
    rng = check_random_state(random_state)
    if init == "random":
        factor = 0.1 * rng.rand(s.shape[0], rank)
    elif init == "random_sqrt":
        avg = np.sqrt(s.mean() / rank)
        factor = rng.rand(s.shape[0], rank) * avg
    elif init == "nndsvd":
        factor, _ = nndsvd(s, rank, eps, random_state)
    elif init == "nndsvdar":
        factor, _ = nndsvd(s, rank, eps, random_state)
        avg = s.mean()
        factor[factor == 0] = abs(
            avg * rng.standard_normal(size=len(factor[factor == 0])) / 100
        )
    elif init == "eigenspectrum":
        factor = eigenspectrum_initialization(s, rank, random_state)
    else:
        raise ValueError(f"Invalid initialization method: {init}")
    return factor


def eigenspectrum_initialization(
    x: array, rank: int, random_state: int | None = None
) -> array:
    """Initialize factors using eigenspectrum decomposition."""
    eigvals, eigvecs = np.linalg.eigh(x)
    idx = np.argsort(np.abs(eigvals))[::-1][:rank]
    eigenvalues_sorted = eigvals[idx]
    eigenvectors_sorted = eigvecs[:, idx]
    factor = eigenvectors_sorted @ np.diag(np.sqrt(np.abs(eigenvalues_sorted)))
    factor[factor < 0] = 0
    return factor


def nndsvd(
    x: array,
    rank: int,
    eps: float = np.finfo(float).eps,
    random_state: int | None = None,
) -> tuple[array, array]:
    """Non-Negative Double Singular Value Decomposition initialization."""
    from sklearn.utils.extmath import randomized_svd, squared_norm

    def norm(x: array) -> float:
        return np.sqrt(squared_norm(x))

    u, s, v = randomized_svd(x, rank, random_state=random_state)
    w = np.zeros_like(u)
    h = np.zeros_like(v)

    w[:, 0] = np.sqrt(s[0]) * np.abs(u[:, 0])
    h[0, :] = np.sqrt(s[0]) * np.abs(v[0, :])

    for j in range(1, rank):
        x_col, y_col = u[:, j], v[j, :]
        x_p, y_p = np.maximum(x_col, 0), np.maximum(y_col, 0)
        x_n, y_n = np.abs(np.minimum(x_col, 0)), np.abs(np.minimum(y_col, 0))
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)
        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

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


def explained_variance(x: array, x_hat: array, center: bool = False) -> float:
    x_upper = x[np.triu_indices(x.shape[0], k=1)]
    x_hat_upper = x_hat[np.triu_indices(x_hat.shape[0], k=1)]

    rss = np.sum((x_upper - x_hat_upper) ** 2)

    if center:
        tss = np.sum((x_upper - np.mean(x_upper)) ** 2)
    else:
        tss = np.sum(x_upper**2)

    if tss == 0:
        return 1.0 if rss == 0 else 0.0

    return 1 - (rss / tss)


def sse(x: array, x_hat: array) -> float:
    x_upper = x[np.triu_indices(x.shape[0], k=1)]
    x_hat_upper = x_hat[np.triu_indices(x_hat.shape[0], k=1)]
    return np.sum((x_upper - x_hat_upper) ** 2)


def frobenius_norm(x: array, x_hat: array) -> float:
    return np.linalg.norm(x - x_hat, ord="fro")
