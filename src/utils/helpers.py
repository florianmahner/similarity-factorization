import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
from numpy.random import Generator
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy

SPOSE_66_PATH = (
    "/LOCAL/fmahner/similarity-factorization/data/things/spose_embedding_66d.txt"
)
SPOSE_49_PATH = (
    "/LOCAL/fmahner/similarity-factorization/data/things/spose_embedding_49d.txt"
)


def estimate_effective_rank(rsm: np.ndarray) -> float:
    """
    Compute the effective rank of a square matrix
    following Roy &Vetterli (2007).
    """
    singular_values = np.linalg.svd(rsm, compute_uv=False)
    probabilities = singular_values / singular_values.sum()
    entropy = -np.sum(probabilities * np.log(probabilities, where=probabilities > 0))
    return float(np.exp(entropy))


def compute_subsampling_fraction_symmetric(
    s: np.ndarray, constant: float = 1.0
) -> float:
    n = len(s)

    # For symmetric matrices, we observe upper triangular + diagonal
    effective_rank = estimate_effective_rank(s)

    # For symmetric matrices: c n r log(n)**2
    required_samples = constant * effective_rank * n * (np.log(n) ** 2)
    degrees_of_freedom = n * (n + 1) // 2
    print(f"Effective rank: {effective_rank}")
    print(f"Required samples: {required_samples}")
    print(f"Degrees of freedom: {degrees_of_freedom}")
    print(f"Fraction of entries to sample: {required_samples / degrees_of_freedom}")
    return min(required_samples / degrees_of_freedom, 1.0)


@dataclass
class CVParams:
    n_splits: int = 5
    n_repeats: int = 5
    max_iter: int = 500
    init: str = "random"
    random_state: int = 0
    candidate_ranks: range = range(2, 7)
    verbose: int = 10


def procrustes_alignment(x, w):
    # find best orthogonal alignment X->W.
    q, _ = orthogonal_procrustes(w, x)

    w_rot = w @ q  # rotate into Xs frame
    return w_rot


def safe_cross_corr(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Returns the (d1 x d2) Pearson correlation matrix between columns of gt and pred,
    but never divides by zero or emits warnings.
    """
    # 1) stack and compute per-column means & stds
    X = np.concatenate([gt, pred], axis=1)  # shape (n, d1+d2)
    means = X.mean(axis=0, keepdims=True)  # (1, d1+d2)
    stds = X.std(axis=0, ddof=1, keepdims=True)  # (1, d1+d2)

    # 2) guard zero‐variance columns
    stds[stds == 0] = 1.0

    # 3) z-score
    Xz = (X - means) / stds  # (n, d1+d2)

    # 4) slice out the cross‐block
    d1 = gt.shape[1]
    C_full = (Xz.T @ Xz) / (gt.shape[0] - 1)  # exact same formula as corrcoef
    return C_full[:d1, d1:]


def best_pairwise_match(
    gt: np.ndarray, pred: np.ndarray
) -> list[tuple[int, int, float]]:
    """
    Finds the optimal one-to-one matching between columns of gt and pred
    that maximizes the sum of Pearson correlations.
    Returns a list of (i, j, corr) tuples.
    """

    C = safe_cross_corr(gt, pred)

    # 3) solve assignment on -C to maximize C
    row_ind, col_ind = linear_sum_assignment(-C)

    # 4) collect results
    matches = []
    for i, j in zip(row_ind, col_ind):
        matches.append((int(i), int(j), float(C[i, j])))
    # sort by descending correlation if you like
    matches.sort(key=lambda x: -x[2])
    corrs = [c[2] for c in matches]
    return corrs


def align_latent_dimensions(x, w, return_correlations=False):
    """
    Align the columns of w to match x as closely as possible (in absolute correlation).

    Returns:
        w_aligned     : w with its columns permuted so that column i is the best match to x[:, i].
        aligned_corrs : list of correlations between each matched pair of columns.
        mean_corr     : mean of those matched correlations.
    """
    x = np.asarray(x)
    w = np.asarray(w)
    rank = x.shape[1]

    # Compute all pairwise correlations between columns of x and columns of w
    corr_matrix = np.zeros((rank, rank))
    for i in range(rank):
        for j in range(rank):
            corr_matrix[i, j] = np.corrcoef(x[:, i], w[:, j])[0, 1]

    # We want to maximize sum of |corr|, but linear_sum_assignment does a min-cost match.
    # So we pass negative absolute correlations as "cost."
    cost_matrix = -np.abs(corr_matrix)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # row_ind will be [0, 1, 2, ..., rank-1] in sorted order
    # col_ind is the best matching column j for each row i

    # Reorder w so that w_aligned[:, i] is best matched to x[:, i]
    # The i-th column of w_aligned is the column col_ind[i] of the original w
    w_aligned = w[:, col_ind]

    # Gather the actual correlations (with sign) for each matched pair
    aligned_corrs = [corr_matrix[i, col_ind[i]] for i in row_ind]
    mean_corr = np.mean(aligned_corrs)

    if return_correlations:
        return w_aligned, aligned_corrs
    else:
        return w_aligned


def add_noise_with_snr(
    x: np.ndarray, snr: float, rng: Generator | int | None = None
) -> np.ndarray:
    if rng is None:
        rng_gen = np.random.default_rng()
    elif isinstance(rng, np.random.Generator):
        rng_gen = rng
    else:
        rng_gen = np.random.default_rng(rng)
    # Ensure snr is within [0, 1]
    snr = np.clip(snr + 1e-12, 1e-12, 1.0)
    # Compute the standard deviation of the signal X
    signal_std = np.std(x, ddof=1)
    # Generate noise with the same standard deviation as the signal
    noise = rng_gen.standard_normal(size=x.shape) * signal_std
    # Combine signal and noise using square-root mixing
    return np.sqrt(snr) * x + np.sqrt(1 - snr) * noise


def add_noise_with_snr_db(
    signal: np.ndarray,
    snr_db: float,
    rng: int | np.random.Generator | None = None,
    centre: bool = True,
) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to *signal* to achieve a target SNR.
    """
    rng_gen: np.random.Generator
    if rng is None:
        rng_gen = np.random.default_rng()
    elif isinstance(rng, np.random.Generator):
        rng_gen = rng
    else:
        rng_gen = np.random.default_rng(rng)

    sig = signal.astype(float)  # ensure float64
    if centre:
        sig_mean = sig.mean(axis=-1, keepdims=True)
        sig = sig - sig_mean

    # power = variance for zero-mean signals
    power_signal = np.mean(sig**2, axis=-1, keepdims=True)

    if power_signal == 0:
        raise ValueError("Signal power is zero; SNR is undefined.")

    snr_linear = 10 ** (snr_db / 10.0)
    power_noise = power_signal / snr_linear
    std_noise = np.sqrt(power_noise)

    noise = rng_gen.standard_normal(size=sig.shape) * std_noise
    return signal + noise


# Clustering and Evaluation Functions
def map_labels_with_hungarian(true_labels, cluster_labels):
    """Use Hungarian assignment to map cluster labels to true labels."""
    cm = confusion_matrix(true_labels, cluster_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[label] for label in cluster_labels])


def purity_score(y_true, y_pred):
    """Compute clustering purity score."""
    cm = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)


def compute_entropy(y, y_hat):
    """Compute normalized entropy of a clustering solution."""
    y, y_hat = np.array(y), np.array(y_hat)
    unique_clusters, unique_classes = np.unique(y_hat), np.unique(y)
    total_entropy = 0.0

    for cluster in unique_clusters:
        indices = np.where(y_hat == cluster)[0]
        if len(indices) == 0:
            continue
        cluster_truth = y[indices]
        counts = np.array([np.sum(cluster_truth == cls) for cls in unique_classes])
        p = counts / (len(indices) + 1e-10)
        cluster_entropy = -np.sum(p[p > 0] * np.log2(p[p > 0] + 1e-10))
        total_entropy += cluster_entropy * len(indices)

    return total_entropy / (len(y) * np.log2(len(unique_classes)))


def compute_metrics(true_labels, predicted_labels):
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    purity = purity_score(true_labels, predicted_labels)
    entropy = compute_entropy(true_labels, predicted_labels)
    return ari, nmi, purity, entropy


def compute_sparseness(matrix):
    """Compute the sparseness of a matrix."""
    if np.any(matrix < 0):
        raise ValueError(
            "Matrix contains negative values. Sparseness is defined for nonnegative matrices only."
        )
    return np.sum(matrix < np.mean(matrix)) / matrix.size


def compute_orthogonality(w):
    """Compute orthogonality score of a matrix."""
    gram_matrix = np.dot(w.T, w)
    return np.linalg.norm(
        gram_matrix - np.eye(gram_matrix.shape[0]), ord="fro"
    ) / np.linalg.norm(np.eye(gram_matrix.shape[0]), ord="fro")


def median_matrix_split(s):
    threshold = np.median(s.flatten())
    s_plus = s.copy()
    s_plus = s_plus - threshold
    s_plus[s_plus < 0] = 0
    s_minus = s.copy()
    s_minus = threshold - s_minus
    s_minus[s_minus < 0] = 0
    mask = s >= threshold
    print("fraction of positive values:", np.sum(mask) / s.size)
    return s_plus, s_minus, threshold, mask


def zero_matrix_split(s):
    s_plus = s.copy()
    s_plus[s_plus < 0] = 0  # keep positives
    s_minus = -s.copy()
    s_minus[s_minus < 0] = 0  # flip negatives to positives
    threshold = 0.0
    mask = s >= threshold
    print("fraction of positive values:", np.sum(mask) / s.size)
    return s_plus, s_minus, threshold, mask


def reconstruct_matrix(w_pos, h_pos, w_neg, h_neg, threshold):
    s_plus_hat = w_pos @ h_pos.T
    s_minus_hat = w_neg @ h_neg.T
    return s_plus_hat - s_minus_hat + threshold


def evar(gt, pred):
    return 1 - (np.linalg.norm(gt - pred, "fro") / np.linalg.norm(gt, "fro"))


def is_symmetric(x):
    return np.allclose(x, x.T)


def is_psd(x):
    return np.all(np.linalg.eigvals(x) >= 0)


def rbf_entropy_heuristic(
    x: np.ndarray,
    sigma_values: np.ndarray,
    n_bins: int = 100,
    return_entropy: bool = False,
):
    """
    Automatically select RBF sigma by maximizing Shannon entropy of off-diagonal kernel entries.
    Parameters:
        x: np.ndarray, shape (n_samples, n_features)
            Input data.
        sigma_values: np.ndarray
            Candidate sigma values to try.
        n_bins: int
            Number of bins to use for entropy histogram.
    """

    x = StandardScaler().fit_transform(x)
    d = pairwise_distances(x, metric="euclidean") ** 2
    n = d.shape[0]

    entropy_values = []
    for sigma in sigma_values:
        k = np.exp(-d / (2 * sigma**2))

        k_offdiag = k[~np.eye(n, dtype=bool)]

        # Histogram to estimate empirical distribution
        hist, _ = np.histogram(k_offdiag, bins=n_bins, range=(0, 1), density=True)
        hist = hist[hist > 0]  # Avoid log(0)

        # Shannon entropy
        h = entropy(hist, base=np.e)
        entropy_values.append(h)

    best_idx = np.argmax(entropy_values)
    best_sigma = sigma_values[best_idx]

    if return_entropy:
        return best_sigma, entropy_values
    return best_sigma
