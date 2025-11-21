# FIXME not working wih hydra sweep for now

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig

from ..lib.common import compute_similarity_matrix_from_triplets

from ..lib.resources import load_resources
from ..lib.utils import fit_srf_model


def _fisher_z(r: float) -> float:
    r = np.clip(r, -0.999, 0.999)
    return 0.5 * np.log((1 + r) / (1 - r))


def _inverse_fisher_z(z: float) -> float:
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


def _find_best_matching_dimension(
    original_dim: np.ndarray, reference_embedding: np.ndarray
) -> float:
    corrs = []
    for idx in range(reference_embedding.shape[1]):
        ref_dim = reference_embedding[:, idx]
        corr = np.corrcoef(original_dim, ref_dim)[0, 1]
        if not np.isnan(corr):
            corrs.append(abs(corr))
    return max(corrs) if corrs else 0.0


def compute_dimension_reliability(
    similarity_matrix: np.ndarray,
    rank: int,
    n_runs: int = 20,
    n_jobs: int = -1,
) -> np.ndarray:
    embeddings = Parallel(n_jobs=n_jobs)(
        delayed(fit_srf_model)(similarity_matrix, rank=rank, seed=seed)
        for seed in range(0, n_runs + 1)
    )
    original_embedding, reference_embeddings = embeddings[0], embeddings[1:]
    reference_embeddings = [emb for emb in reference_embeddings if emb is not None]
    if original_embedding is None or not reference_embeddings:
        raise RuntimeError("SRF fitting failed for dimension reliability analysis")

    n_dims = original_embedding.shape[1]
    reliabilities = np.zeros(n_dims)
    for dim_idx in range(n_dims):
        original_dim = original_embedding[:, dim_idx]
        best_corrs = [
            _find_best_matching_dimension(original_dim, ref_embedding)
            for ref_embedding in reference_embeddings
        ]
        if best_corrs:
            mean_z = np.mean([_fisher_z(corr) for corr in best_corrs])
            reliabilities[dim_idx] = _inverse_fisher_z(mean_z)
        else:
            reliabilities[dim_idx] = 0.0
    return reliabilities


def run_dimension_reliability_analysis(
    triplets: np.ndarray,
    rank: int,
    n_runs: int = 20,
    n_jobs: int = -1,
) -> pd.DataFrame:
    similarity = compute_similarity_matrix_from_triplets(1854, triplets)
    reliabilities = compute_dimension_reliability(
        similarity, rank=rank, n_runs=n_runs, n_jobs=n_jobs
    )
    data = [
        {"Dimension": idx, "Reliability": float(value)}
        for idx, value in enumerate(reliabilities)
    ]
    return pd.DataFrame(data)


def run(cfg: DictConfig) -> None:
    resources = load_resources(cfg)

    df = run_dimension_reliability_analysis(
        resources.train_triplets,
        rank=cfg.dims,
        n_runs=cfg.n_runs,
        n_jobs=cfg.n_jobs,
    )
    output_file = Path.cwd() / "results.csv"
    df.to_csv(output_file, index=False)
