import numpy as np
from dataclasses import dataclass
from numpy.random import Generator
from utils.helpers import add_noise_with_snr, add_positive_noise_with_snr
from tools.rsa import compute_similarity

RNG = np.random.default_rng(0)

Array = np.ndarray


def simulation(
    n: int,
    k: int,
    primary_concentration: float = 5.0,
    base_concentration: float = 1.0,
    rng: np.random.Generator = np.random.default_rng(42),
) -> np.ndarray:
    m = np.zeros((n, k))
    cluster_sizes = [n // k] * k
    leftover = n - sum(cluster_sizes)
    for i in range(leftover):
        cluster_sizes[i] += 1

    start_idx = 0
    primary_cluster = np.zeros(n, dtype=int)
    for c in range(k):
        end_idx = start_idx + cluster_sizes[c]
        m[start_idx:end_idx, c] = 1.0
        primary_cluster[start_idx:end_idx] = c
        start_idx = end_idx

    soft_m = np.zeros_like(m, dtype=float)
    for i in range(n):
        alphas = np.ones(k) * base_concentration
        alphas[primary_cluster[i]] = primary_concentration
        soft_m[i, :] = rng.dirichlet(alphas)
    return soft_m


def make_rsa_similarity(
    n_stimuli=100, K=3, block_strength=0.8, within_block_var=0.1, seed=0
):
    """
    Simulate RSA-style data with latent structure
    n_stimuli: number of stimuli (images, words, etc.)
    K: latent feature dimensions
    block_strength: how distinct the clusters are
    """
    rng = np.random.default_rng(seed)

    # Soft feature loadings (what NMF should recover)
    H = rng.dirichlet(np.ones(K) * 0.5, size=n_stimuli)  # n x K

    # Expected similarity from shared features
    S_base = H @ H.T  # This is what symmetric NMF assumes!

    # Add within-block noise (measurement error)
    S = S_base + within_block_var * rng.standard_normal((n_stimuli, n_stimuli))
    S = np.maximum(S, 0)  # Similarities are non-negative
    S = (S + S.T) / 2  # Symmetrize

    # Normalize to [0, 1] like correlation-based RDMs
    S = S / np.max(S)

    return S, H, {"story": "RSA with overlapping latent features"}


def simulation_dirichlet(
    n: int,
    k: int,
    alpha: float = 1.0,
    rng: np.random.Generator = np.random.default_rng(42),
    sort_by_cluster_size: bool = True,
) -> np.ndarray:
    """Generate soft cluster memberships from symmetric Dirichlet distribution.

    Parameters
    ----------
    n : int
        Number of items
    k : int
        Number of clusters
    alpha : float
        Dirichlet concentration parameter (controls difficulty)
        - Low alpha (0.1-0.5): Concentrated memberships (easy clustering)
        - High alpha (5-10): Diffuse memberships (hard clustering)
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    memberships : np.ndarray
        (n, k) matrix of soft cluster memberships (rows sum to 1)
    """

    # sort by cluster size
    memberships = rng.dirichlet([alpha] * k, size=n)
    # Sort by max membership for visualization
    if sort_by_cluster_size:
        max_indices = memberships.argmax(axis=1)
        sorted_indices = np.argsort(max_indices)
        sorted_memberships = memberships[sorted_indices]
        return sorted_memberships
    else:
        return memberships


@dataclass
class SimulationParams:
    n: int = 100  # number of samples
    p: int = 50  # number of features
    k: int = 4  # number of components
    snr: float = 1.0  # noise level for the data

    rng_state: int = 0

    # Dirichlet concentration parameter for the primary cluster.
    # Higher values make the primary membership stronger.
    primary_concentration: float = 5.0

    # Dirichlet concentration parameter for non-primary clusters.
    # Lower values keep these memberships small.
    base_concentration: float = 1.0

    # Sparsity of the feature matrix.
    sparsity: float | None = None

    # Type of membership to generate: 'dirichlet' or 'hard'
    membership_type: str = "dirichlet"


def generate_hard_membership_loadings(n_samples: int, n_clusters: int) -> np.ndarray:
    m = np.zeros((n_samples, n_clusters))
    start_idx = 0
    # Distribute n_samples as evenly as possible among n_clusters
    cluster_sizes = [n_samples // n_clusters] * n_clusters
    leftover = n_samples - sum(cluster_sizes)
    for i in range(leftover):
        cluster_sizes[i] += 1

    for c in range(n_clusters):
        end_idx = start_idx + cluster_sizes[c]
        # Hard assignment: for each sample in the cluster, M[i, c] = 1
        m[start_idx:end_idx, c] = 1.0
        start_idx = end_idx

    return m


def generate_dirichlet_membership_loadings(
    n_samples: int,
    n_clusters: int,
    primary_concentration: float = 5.0,
    base_concentration: float = 1.0,
    rng: Generator = RNG,
) -> np.ndarray:
    m = np.zeros((n_samples, n_clusters))
    cluster_sizes = [n_samples // n_clusters] * n_clusters
    leftover = n_samples - sum(cluster_sizes)
    for i in range(leftover):
        cluster_sizes[i] += 1

    start_idx = 0
    primary_cluster = np.zeros(n_samples, dtype=int)
    for c in range(n_clusters):
        end_idx = start_idx + cluster_sizes[c]
        m[start_idx:end_idx, c] = 1.0
        primary_cluster[start_idx:end_idx] = c
        start_idx = end_idx

    soft_m = np.zeros_like(m, dtype=float)
    for i in range(n_samples):
        # Build the Dirichlet alpha vector:
        # - Set the base concentration for all clusters
        # - Boost the primary cluster's concentration for sample i
        alphas = np.ones(n_clusters) * base_concentration
        alphas[primary_cluster[i]] = primary_concentration
        soft_m[i, :] = rng.dirichlet(alphas)
    return soft_m


def generate_feature_matrix(
    p: int,
    k: int,
    rng: Generator,
    low: float = 0.0,
    high: float = 1.0,
    sparsity: float | None = None,
) -> np.ndarray:
    """
    Sample a positive, potentially sparse feature matrix F (k x p) with uniform distribution.
    """
    if sparsity is None:
        f = rng.uniform(low, high, size=(k, p))
    else:
        mask = rng.random((k, p)) < (1 - sparsity)  # 1-sparsity for density
        f = rng.uniform(low, high, size=(k, p)) * mask

    return f


def generate_data_matrix(
    m: Array, f: Array, snr: float = 0.1, rng: Generator = RNG
) -> Array:
    """Generate noisy data matrix with a given SNR between 0 and 1."""
    x = m @ f  # Compute the clean signal
    noisy_x = add_positive_noise_with_snr(x, snr, rng)
    return noisy_x


def generate_simulation_data(
    params: SimulationParams,
) -> tuple[Array, Array, Array]:
    rng = np.random.default_rng(params.rng_state)
    if params.membership_type == "hard":
        m = generate_hard_membership_loadings(params.n, params.k)
    else:
        m = generate_dirichlet_membership_loadings(
            params.n,
            params.k,
            params.primary_concentration,
            params.base_concentration,
            rng,
        )

    f = generate_feature_matrix(params.p, params.k, rng, sparsity=params.sparsity)
    x = generate_data_matrix(m, f, params.snr, rng)

    s = x @ x.T

    return x, m, f, s
