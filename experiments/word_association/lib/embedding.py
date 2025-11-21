from __future__ import annotations

import numpy as np
import pandas as pd
from pysrf import SRF


def fit_srf(similarity: np.ndarray, rank: int, max_outer: int = 1000) -> SRF:
    """Fit SRF model to similarity matrix."""
    model = SRF(
        rank=rank,
        rho=3.0,
        max_outer=max_outer,
        max_inner=50,
        tol=1e-4,
        verbose=1,
        random_state=42,
        missing_values=np.nan,
    )
    model.fit(similarity)
    return model


def compute_coherence(
    w: np.ndarray, similarity: np.ndarray, top_n: int = 50
) -> np.ndarray:
    """Compute coherence for each dimension (avg similarity among top words)."""
    n_dims = w.shape[1]
    coherence = np.zeros(n_dims)

    for dim in range(n_dims):
        top_idx = np.argsort(w[:, dim])[-top_n:]
        submatrix = similarity[np.ix_(top_idx, top_idx)]
        coherence[dim] = np.nanmean(submatrix)

    return coherence


def compute_sparsity(w: np.ndarray) -> pd.DataFrame:
    """Compute sparsity metrics for each dimension."""
    metrics = []
    for dim in range(w.shape[1]):
        weights = w[:, dim]
        pct_active = (weights > 0).mean()
        gini = np.abs(np.subtract.outer(weights, weights)).mean() / (
            2 * np.mean(weights + 1e-8)
        )
        entropy = -np.sum(weights * np.log(weights + 1e-8))
        metrics.append(
            {
                "dimension": dim,
                "pct_active": pct_active,
                "gini": gini,
                "entropy": entropy,
            }
        )
    return pd.DataFrame(metrics)


def get_top_words(
    w: np.ndarray, vocabulary: list[str], top_n: int = 20
) -> pd.DataFrame:
    """Get top N words for each dimension."""
    rows = []
    for dim in range(w.shape[1]):
        top_idx = np.argsort(w[:, dim])[-top_n:][::-1]
        for rank, idx in enumerate(top_idx):
            rows.append(
                {
                    "dimension": dim,
                    "rank": rank,
                    "word": vocabulary[idx],
                    "loading": w[idx, dim],
                }
            )
    return pd.DataFrame(rows)


def compute_reconstruction_metrics(
    w: np.ndarray, similarity: np.ndarray
) -> dict[str, float]:
    """Compute reconstruction quality metrics."""
    observed = ~np.isnan(similarity)
    reconstruction = w @ w.T
    sim_actual = similarity[observed]
    sim_pred = reconstruction[observed]

    corr = float(np.corrcoef(sim_actual, sim_pred)[0, 1])
    mse = float(np.mean((sim_actual - sim_pred) ** 2))

    return {"reconstruction_corr": corr, "reconstruction_mse": mse}
