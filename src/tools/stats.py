#!/usr/bin/env python3

"""Statistical utilities"""

import numpy as np
from scipy.stats import pearsonr, spearmanr

Array = np.ndarray


# ------- Helper Functions for Array Transformations ------- #


def apply_transform(
    x: np.ndarray, transform: str | list[str], axis: int = 0, verbose: bool = False
) -> np.ndarray:
    transform_to_func = {
        "center": center,
        "standardize": standardize,
        "normalize_l2": normalize_l2,
        "normalize_l1": normalize_l1,
        "positive_shift": positive_shift,
        "softmax": softmax,
        "log_softmax": log_softmax,
        "log": log,
    }

    if isinstance(transform, str):
        if transform not in transform_to_func:
            raise ValueError(
                f"Transform {transform} not found. Available transforms: {transform_to_func.keys()}"
            )
        transform = [transform]

    for t in transform:
        if verbose:
            print(f"Applying {t}")
        x = transform_to_func[t](x, axis=axis)
    return x


def center(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Center the input array by subtracting the mean along the specified axis."""
    x = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(x, axis=axis, keepdims=True)
    return x - mean


def standardize(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Standardize the input array by centering and scaling to unit variance."""
    x = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(x, axis=axis, keepdims=True)
    std = np.nanstd(x, axis=axis, keepdims=True, ddof=0)
    std[std == 0] = 1  # Prevent division by zero
    return (x - mean) / std


def relu(x: Array) -> Array:
    """Apply the Rectified Linear Unit (ReLU) function element-wise."""
    return np.maximum(0, x)


def normalize_l2(x: Array, axis: int = 0, eps: float = 1e-8) -> Array:
    """Normalize the input array to have unit L2 norm."""
    x = np.asarray(x)
    norms = np.sqrt(np.sum(x**2, axis=axis, keepdims=True))
    return x / (norms + eps)


def normalize_l1(x: Array, axis: int = 0, eps: float = 1e-8) -> Array:
    """Normalize the input array to have unit L1 norm."""
    x = np.asarray(x)
    norms = np.sum(np.abs(x), axis=axis, keepdims=True)
    return x / (norms + eps)


def positive_shift(x: Array, axis: int = 0) -> Array:
    """Shift all values in the array to be non-negative."""
    return x - np.min(x)


def softmax(x: Array, axis: int = 1) -> Array:
    """Apply the softmax function along axis 1."""
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


def log(x: Array) -> Array:
    """Log transform the input array."""
    return np.log(x)


def log_softmax(x: Array, axis: int = 1) -> Array:
    """Apply the log softmax function along axis 1."""
    return np.log(softmax(x, axis=axis))


def one_hot(x: Array, n_classes: int) -> Array:
    """Convert integer labels to one-hot encoded vectors."""
    return np.eye(n_classes)[x]


def log_likelihood(x: Array, mean: Array, std: Array) -> Array:
    """Calculate the log likelihood of the input array under a normal distribution."""
    return -0.5 * np.log(2 * np.pi * std**2) - ((x - mean) ** 2) / (2 * std**2)


def rss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the residual sum of squares (RSS)."""
    return np.sum((y_true - y_pred) ** 2)


def aic_linear_regression(
    y_true: np.ndarray, y_pred: np.ndarray, num_predictors: int
) -> float:
    """
    Calculate the Akaike Information Criterion (AIC) for linear regression.
    """
    n = len(y_true)
    k = num_predictors + 1  # +1 for the intercept
    aic = n * np.log(rss(y_true, y_pred) / n) + 2 * k
    return aic


# ------- Helper Function for Variance Testing  ------- #


def effective_dimensionality(X: Array) -> float:
    """Calculate the effective dimensionality of an input array. It measure the
    effective number of dimensions that are relevant in the data (ie the number
    of dimensions needed to explain the variance in the data). This is done by
    calculating the ratio of the sum of the eigenvalues to the sum of the squared
    eigenvalues of the covariance matrix (ie PCA explained variance ratio)."""
    if X.ndim > 2:
        raise ValueError("Input array must be two-dimensional; " f"X.shape = {X.shape}")
    if X.ndim == 1:
        X = X[:, None]
    cov = np.cov(X.T, ddof=1)
    eigenvalues = np.linalg.eigvals(cov)
    return np.sum(eigenvalues) ** 2 / np.sum(eigenvalues**2)


# ------- Helper Function for Significance and Effect Size Testing  ------- #


def compute_correlation_coeff(x: Array, y: Array, method: str = "pearson"):
    """Compute the correlation coefficient and p-value between two matrices."""
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    # Dictionary mapping method names to functions
    correlation_methods = {
        "pearson": pearsonr,
        "spearman": spearmanr,
    }

    # Validate the method
    if method not in correlation_methods:
        raise ValueError(f"Method {method} not supported.")

    # Compute the correlation using the selected method
    corr, pval = correlation_methods[method](x.flatten(), y.flatten())
    return corr, pval


def vectorized_pearsonr(x: Array, y: Array) -> float | Array:
    """Alterntive to scipy.stats.pearsonr that is vectorized over the first dimension for
    fast pairwise correlation calculation."""
    if x.shape != y.shape:
        raise ValueError(
            "Input arrays must have the same dimensions; "
            f"x.shape = {x.shape}, y.shape = {y.shape}"
        )
    if x.ndim < 2:
        x = x[:, None]
    if y.ndim < 2:
        y = y[:, None]
    n = x.shape[1]
    covariance = np.cov(x.T, y.T, ddof=1)
    x_std = np.sqrt(covariance[:n, :n].diagonal())
    y_std = np.sqrt(covariance[n:, n:].diagonal())
    pearson_r = covariance[:n, n:] / np.outer(x_std, y_std)
    return pearson_r


def spearman_brown_correction(reliability: float, split_factor: int) -> float:
    """Compute the Spearman-Brown prophecy formula.
    Args:
        reliability (float): The reliability of the original test.
        split_factor (int): The factor of the split (e.g. 2 for split-half reliability).
    """
    return split_factor * reliability / (1 + (split_factor - 1) * reliability)


def split_half_reliability(data: list | Array, num_splits: int = 1000) -> float | Array:
    """Split-half reliability by randomly splitting the data into two halves and
    calculating the pearson correlation between the two halves.
    Args:
        data (list | Array): The array to calculate split half reliability for.
        num_splits (int): The number of random splits for calculating reliability."""
    if isinstance(data, list):
        data = np.array(data)
    if data.ndim > 1:
        raise ValueError(
            "Input array must be one-dimensional; " f"data.shape = {data.shape}"
        )
    split_reliablity = np.zeros(num_splits)
    for n in range(num_splits):
        mask = np.random.choice([True, False], size=data.shape)
        split_reliablity[n] = vectorized_pearsonr(data[mask], data[~mask])

    average_reliability = average_pearson_r(split_reliablity)
    corrected_reliability = spearman_brown_correction(average_reliability, 2)
    return corrected_reliability


def dimension_reliability(
    embeddings: Array,
    n_splits: int = 100,
    random_state: int = 42,
) -> Array:
    """Cross-validated split-half reliability per dimension across multiple runs.

    For each random split of objects into halves A and B:
      - Match dimensions between all pairs of runs using A-objects (argmax corr)
      - Evaluate matched dimensions on held-out B-objects
    This prevents overfitting in the dimension matching step.

    Parameters
    ----------
    embeddings : (n_runs, n_objects, n_dims) array
        Aligned embeddings from multiple SRF runs.
    n_splits : int
        Number of random object splits.
    random_state : int
        Random seed.

    Returns
    -------
    reliability : (n_dims,) array
        Cross-validated per-dimension reliability (Spearman-Brown corrected).
    """
    n_runs, n_objects, n_dims = embeddings.shape
    rng = np.random.default_rng(random_state)
    emb = embeddings.astype(np.float64)

    # Precompute all unique pairs
    pairs_i, pairs_j = np.triu_indices(n_runs, k=1)
    n_pairs = len(pairs_i)

    all_split_reliabilities = np.zeros((n_splits, n_dims))

    for s in range(n_splits):
        mask_a = rng.random(n_objects) < 0.5
        mask_b = ~mask_a

        # Normalize all runs on each half once: (n_runs, n_half, k) -> (n_runs, k, n_half)
        emb_a = _normalize_cols(emb[:, mask_a])  # (n_runs, n_a, k)
        emb_b = _normalize_cols(emb[:, mask_b])  # (n_runs, n_b, k)

        pair_per_dim = np.zeros((n_pairs, n_dims))

        for p in range(n_pairs):
            i, j = pairs_i[p], pairs_j[p]
            # Match on A: (k x k) corr matrix via normalized dot product
            corr_a = emb_a[i].T @ emb_a[j]  # (k, k)
            best_j = np.argmax(np.abs(corr_a), axis=1)  # (k,)

            # Evaluate on B
            corr_b = emb_b[i].T @ emb_b[j]  # (k, k)
            pair_per_dim[p] = np.abs(corr_b[np.arange(n_dims), best_j])

        z = np.arctanh(np.clip(pair_per_dim, -0.999, 0.999))
        all_split_reliabilities[s] = np.tanh(np.mean(z, axis=0))

    # Average across splits in Fisher-z space
    z_splits = np.arctanh(np.clip(all_split_reliabilities, -0.999, 0.999))
    raw_reliability = np.tanh(np.mean(z_splits, axis=0))

    # Spearman-Brown correction (split-half -> full)
    return 2 * raw_reliability / (1 + raw_reliability)


def _normalize_cols(x: Array) -> Array:
    """Center and L2-normalize columns for each matrix in a batch.

    Input: (batch, n_samples, k) -> Output: (batch, n_samples, k)
    After normalization, col_i.T @ col_j = Pearson r between columns i and j.
    """
    x_c = x - x.mean(axis=-2, keepdims=True)
    norms = np.sqrt(np.sum(x_c ** 2, axis=-2, keepdims=True))
    return x_c / np.maximum(norms, 1e-12)


def _fast_col_corr(a: Array, b: Array) -> Array:
    """Fast (k x k) Pearson correlation between columns of a and b.

    Both a and b are (n_samples, k). Returns (k, k) correlation matrix.
    """
    a_c = a - a.mean(axis=0, keepdims=True)
    b_c = b - b.mean(axis=0, keepdims=True)
    a_n = np.sqrt(np.sum(a_c ** 2, axis=0, keepdims=True))
    b_n = np.sqrt(np.sum(b_c ** 2, axis=0, keepdims=True))
    a_normed = a_c / np.maximum(a_n, 1e-12)
    b_normed = b_c / np.maximum(b_n, 1e-12)
    return a_normed.T @ b_normed


def fisher_z_transform(pearson_r: Array) -> Array:
    """Perform Fisher Z-transform on Pearson r values, ensuring valid input range."""
    pearson_r = np.asarray(pearson_r, dtype=np.float64)  # Ensure floating point type
    eps = 1e-7  # Slightly larger than before for better numerical stability

    # Replace invalid values (e.g., NaNs or infinite)
    pearson_r = np.where(np.isfinite(pearson_r), pearson_r, np.nan)
    pearson_r = np.clip(pearson_r, -1 + eps, 1 - eps)

    return np.arctanh(pearson_r)


def average_pearson_r(pearson_r: Array, axis: int = 0) -> Array:
    """Compute the average Pearson r after Fisher Z-transform and inverse transformation."""
    fisher_z = fisher_z_transform(pearson_r)
    mean_z = np.nanmean(fisher_z, axis=axis)

    return np.tanh(mean_z)


# ------- Helper Functions for Probability Densities  ------- #


def _normal_pdf_numpy(x: Array, mean: Array, std: Array) -> Array:
    """Compute the probability density function of a normal distribution."""
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
