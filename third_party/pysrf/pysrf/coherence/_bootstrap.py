from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from threadpoolctl import threadpool_limits

from .._common import RandomStateLike, as_seed_sequence


def _replicate_seed(parent: int, p_index: int, replicate_index: int) -> int:
    """Deterministic per-(p_index, replicate_index) bootstrap seed.

    A plain linear hash on the three integers, masked to 32 bits. Matches
    the seeding used by the reference implementation in update_pysrf, so
    coherence and leakage profiles are numerically identical (this is
    what reproduces the published k_cut values on borderline matrices
    like THINGS). The multipliers 1_000_003 and 9176 are arbitrary
    co-prime mixers chosen to spread bits across the index space; no
    significance beyond that.
    """
    return (int(parent) + 1_000_003 * int(p_index) + 9176 * int(replicate_index)) & 0xFFFFFFFF


def _bootstrap_parent_seed(random_state: RandomStateLike) -> int:
    """Reduce a user-supplied random_state to a 32-bit parent seed."""
    if random_state is None or isinstance(random_state, (int, np.integer)):
        return int(random_state or 0) & 0xFFFFFFFF
    return int(as_seed_sequence(random_state).generate_state(1)[0]) & 0xFFFFFFFF


def _bootstrap_subspace_stability(
    similarity: np.ndarray,
    observation_mask: np.ndarray,
    top_eigenvectors: np.ndarray,
    sampling_grid: np.ndarray,
    n_bootstrap: int,
    random_state: RandomStateLike,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    parent_seed = _bootstrap_parent_seed(random_state)

    args_list = [
        (index, float(fraction), similarity, observation_mask, top_eigenvectors,
         n_bootstrap, parent_seed)
        for index, fraction in enumerate(sampling_grid)
    ]

    # Loky backend: each replicate runs in its own process, so ARPACK calls
    # escape the GIL and the matrix is shared via joblib's memmap for free.
    # BLAS is pinned to 1 thread per worker inside `_bootstrap_at_fraction`.
    if n_jobs == 1:
        slices = [_bootstrap_at_fraction(*args) for args in args_list]
    else:
        slices = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_bootstrap_at_fraction)(*args) for args in args_list
        )

    coherence = np.stack([s[0] for s in slices], axis=1)
    recovered_mass = np.stack([s[1] for s in slices], axis=1)
    return coherence, recovered_mass


def _bootstrap_at_fraction(
    p_index: int,
    sampling_fraction: float,
    similarity: np.ndarray,
    observation_mask: np.ndarray,
    top_eigenvectors: np.ndarray,
    n_bootstrap: int,
    parent_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = similarity.shape[0]
    max_rank = top_eigenvectors.shape[1]
    triu = np.triu_indices(n, k=1)

    coherence = np.empty((max_rank, n_bootstrap), dtype=np.float64)
    recovered_mass = np.empty_like(coherence)
    # Pin BLAS to one thread inside the loky worker so n_jobs maps to
    # actual core usage instead of BLAS_threads * n_jobs.
    with threadpool_limits(limits=1):
        for replicate_index in range(n_bootstrap):
            seed = _replicate_seed(parent_seed, p_index, replicate_index)
            rng = np.random.default_rng(seed)
            replicate = _subsampled_replicate(
                similarity, observation_mask, sampling_fraction, rng, triu,
            )
            # `eigsh` is called without v0 on purpose: the bootstrap rng is
            # for the subsample, not for ARPACK's starting vector. Coupling
            # them lets the bootstrap rng influence the basis ARPACK lands
            # on inside close-eigenvalue subspaces, which inflates the
            # per-dim leakage at the changepoint on borderline matrices.
            _, replicate_eigenvectors = _top_eigenpairs(replicate, max_rank)
            coherence[:, replicate_index] = _cumulative_subspace_overlap(
                replicate_eigenvectors, top_eigenvectors,
            )
            recovered_mass[:, replicate_index] = _cumulative_recovered_mass(
                similarity, replicate_eigenvectors,
            )
    return coherence, recovered_mass


def _subsampled_replicate(
    similarity: np.ndarray,
    observation_mask: np.ndarray,
    sampling_fraction: float,
    rng: np.random.Generator,
    triu: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    n = similarity.shape[0]
    triu_i, triu_j = triu
    sampled = rng.random(triu_i.size) < sampling_fraction
    sampled &= observation_mask[triu_i, triu_j]

    replicate = np.zeros((n, n), dtype=np.float64)
    values = similarity[triu_i[sampled], triu_j[sampled]] / sampling_fraction
    replicate[triu_i[sampled], triu_j[sampled]] = values
    replicate[triu_j[sampled], triu_i[sampled]] = values
    np.fill_diagonal(replicate, np.diag(similarity))
    return replicate


def _cumulative_subspace_overlap(
    replicate_eigenvectors: np.ndarray, top_eigenvectors: np.ndarray,
) -> np.ndarray:
    squared_overlap = (replicate_eigenvectors.T @ top_eigenvectors) ** 2
    ranks = np.arange(squared_overlap.shape[0])
    cumulative = np.cumsum(squared_overlap, axis=0)[ranks, ranks]
    return np.clip(cumulative, 0.0, 1.0)


def _cumulative_recovered_mass(
    similarity: np.ndarray, replicate_eigenvectors: np.ndarray,
) -> np.ndarray:
    mass_by_dimension = np.einsum(
        "ij,ij->j", replicate_eigenvectors, similarity @ replicate_eigenvectors,
    )
    return np.cumsum(mass_by_dimension)


def _top_eigenpairs(
    a: np.ndarray, k: int, v0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if k < a.shape[0]:
        try:
            values, vectors = eigsh(a, k=k, which="LA", tol=1e-6, v0=v0)
            return _keep_top_k_descending(values, vectors, k)
        except Exception:
            pass
    return _keep_top_k_descending(*eigh(a), k=k)


def _keep_top_k_descending(
    values: np.ndarray, vectors: np.ndarray, k: int,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(values)[::-1][:k]
    return values[order], vectors[:, order]
