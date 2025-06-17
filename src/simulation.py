import numpy as np
from dataclasses import dataclass
from numpy.random import Generator
from srf.helpers import add_noise_with_snr
from tools.metrics import cosine_similarity

RNG = np.random.default_rng(0)

Array = np.ndarray


@dataclass
class SimulationParams:
    n: int = 100  # number of samples
    p: int = 50  # number of features
    k: int = 4  # number of components
    snr: float = 1.0  # noise level for the data

    rng_state: int = 0

    # Dirichlet concentration parameter for the primary cluster.
    # Higher values make the primary membership stronger.
    primary_concentration: float = 10.0

    # Dirichlet concentration parameter for non-primary clusters.
    # Lower values keep these memberships small.
    base_concentration: float = 1.0


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


def generate_feature_matrix(p: int, k: int, rng: Generator = RNG) -> np.ndarray:
    f = rng.normal(size=(k, p))
    return f


def generate_data_matrix(
    m: Array, f: Array, snr: float = 0.1, rng: Generator = RNG
) -> Array:
    """Generate noisy data matrix with a given SNR between 0 and 1."""
    x = m @ f  # Compute the clean signal
    return add_noise_with_snr(x, snr, rng)


def generate_simulation_data(
    params: SimulationParams,
) -> tuple[Array, Array, Array]:
    rng = np.random.default_rng(params.rng_state)
    m = generate_dirichlet_membership_loadings(
        params.n,
        params.k,
        params.primary_concentration,
        params.base_concentration,
        rng,
    )

    f = generate_feature_matrix(params.p, params.k, rng)
    x = generate_data_matrix(m, f, params.snr, rng)
    s = x @ x.T
    return x, m, f, s
