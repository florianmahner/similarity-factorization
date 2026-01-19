from __future__ import annotations

import numpy as np
from pathlib import Path
import pandas as pd

from utils.graphs import build_directed_graph, clean_graph, graph_to_matrix
from utils.graphs import symmetrize_matrix


def _select_vocabulary(g, top_n: int | None) -> list[str]:
    if top_n is None or top_n <= 0 or top_n >= g.number_of_nodes():
        return sorted(g.nodes())

    # Using purely degree might bias towards general words (stopwords).
    # But for SWOW it's usually fine.
    strengths = g.degree(weight="weight")
    sorted_nodes = sorted(
        strengths,
        key=lambda item: (-item[1], item[0]),
    )
    selected = [node for node, _ in sorted_nodes[:top_n]]
    return sorted(selected)


def compute_ppmi(
    counts: np.ndarray, negative_as_nan: bool = False, smoothing: float = 1e-7
) -> np.ndarray:
    """
    Calculate PPMI matrix from count matrix using vectorized operations.

    Args:
        counts: Symmetric count matrix.
        negative_as_nan: If True, Negative PMI is NaN (Missing).
                         If False, Negative PMI is 0 (Standard SPPMI).
        smoothing: Small constant to prevent division by zero.
    """
    n = counts.shape[0]

    # 1. Probability Distributions
    # Ensure we don't divide by zero if the matrix has empty rows/cols
    total = np.nansum(counts)
    if total == 0:
        raise ValueError("Count matrix is empty")

    # Treat NaNs as 0 for probability calculation, but keep track of them
    counts_clean = np.nan_to_num(counts, nan=0.0)

    # P(i, j)
    p_joint = counts_clean / total

    # P(i) and P(j)
    row_sum = counts_clean.sum(axis=1)
    p_marginal = row_sum / total

    # 2. Vectorized PMI Calculation
    # Outer product: p_marginal[i] * p_marginal[j]
    p_indep = np.outer(p_marginal, p_marginal)

    # Avoid division by zero
    p_indep[p_indep < smoothing] = smoothing

    # PMI = log2( P(i,j) / (P(i)*P(j)) )
    # We use 'where' to handle cases where p_joint is 0
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2(p_joint / p_indep)

    # 3. Handling Bounds
    # If p_joint was 0, log2(0) is -inf. We generally want to mask these.
    # If we want sparse positive, we look at pmi > 0.

    ppmi = np.full((n, n), np.nan, dtype=np.float32)

    if negative_as_nan:
        # Standard User Request: Values <= 0 become NaN
        mask_positive = pmi > 0
        ppmi[mask_positive] = pmi[mask_positive]
        # Note: Implicitly, everything else remains NaN
    else:
        # Standard SPPMI: Values <= 0 become 0
        ppmi = np.maximum(pmi, 0)
        # Restore NaNs where the original data was missing (if desired)
        if np.isnan(counts).any():
            ppmi[np.isnan(counts)] = np.nan

    # Always mask diagonal for semantic similarity graphs
    np.fill_diagonal(ppmi, np.nan)

    return ppmi


def make_ppmi_graph(
    sources: list[str],
    targets: list[str],
    counts: list[float],
    symmetrization: str = "sum",  # CHANGED DEFAULT: 'geometric_mean' destroys association data
    top_n: int | None = None,
    bidirectional_only: bool = False,
) -> tuple[np.ndarray, list[str], dict]:
    """
    Build PPMI similarity graph from directed association data.
    """
    # 1. Build Directed Graph
    g = build_directed_graph(sources, targets, counts)
    g = clean_graph(g)

    # 2. Select Vocabulary
    vocabulary = _select_vocabulary(g, top_n)
    if not vocabulary:
        raise ValueError("Graph is empty after top-n filtering")

    # 3. Extract Subgraph
    subgraph = g.subgraph(vocabulary).copy()
    counts_matrix = graph_to_matrix(subgraph, vocabulary)

    # 4. Symmetrize
    # CRITICAL CHANGE: Use "sum" or "mean".
    # "geometric_mean" implies AND logic (must be present in BOTH).
    # "sum" implies OR logic (present in EITHER).
    counts_sym = symmetrize_matrix(
        counts_matrix, method=symmetrization, bidirectional_only=bidirectional_only
    )

    # 5. Compute PPMI
    # NOTE: You might want to toggle negative_as_nan based on your specific NMF loss
    ppmi_matrix = compute_ppmi(counts_sym, negative_as_nan=False)

    # 6. Stats
    ppmi_obs = ppmi_matrix[~np.isnan(ppmi_matrix)]
    n_edges_directed = subgraph.number_of_edges()
    n_edges_sym = np.sum(~np.isnan(counts_sym)) // 2

    metadata = {
        "n_words": int(len(vocabulary)),
        "n_edges_directed": int(n_edges_directed),
        "n_edges_bidirectional": int(n_edges_sym),
        "pct_missing": float(np.isnan(ppmi_matrix).sum() / ppmi_matrix.size * 100),
        "symmetrization": symmetrization,
        "ppmi_min": float(ppmi_obs.min()) if len(ppmi_obs) > 0 else 0.0,
        "ppmi_max": float(ppmi_obs.max()) if len(ppmi_obs) > 0 else 0.0,
        "ppmi_mean": float(ppmi_obs.mean()) if len(ppmi_obs) > 0 else 0.0,
    }

    return ppmi_matrix, vocabulary, metadata
