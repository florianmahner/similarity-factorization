import numpy as np

Array = np.ndarray


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
