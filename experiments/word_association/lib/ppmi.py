from __future__ import annotations

import numpy as np

from utils.graphs import build_directed_graph, clean_graph, graph_to_matrix
from utils.graphs import symmetrize_matrix


def _select_vocabulary(g, top_n: int | None) -> list[str]:
    if top_n is None or top_n <= 0 or top_n >= g.number_of_nodes():
        return sorted(g.nodes())

    strengths = g.degree(weight="weight")
    sorted_nodes = sorted(
        strengths,
        key=lambda item: (-item[1], item[0]),
    )
    selected = [node for node, _ in sorted_nodes[:top_n]]
    return sorted(selected)


def compute_ppmi(counts: np.ndarray) -> np.ndarray:
    """Calculate PPMI matrix from count matrix.

    PPMI = max(0, log2(p(i,j) / (p(i) * p(j))))
    Negative PMI values are treated as missing (NaN) rather than zero.
    Returns unbounded positive values (not [0,1]).
    """
    n = counts.shape[0]
    observed = ~np.isnan(counts)
    counts_clean = np.where(observed, counts, 0)
    total = counts_clean.sum()

    if total == 0:
        raise ValueError("Count matrix is empty")

    p_joint = counts_clean / total
    p_marginal = counts_clean.sum(axis=1) / total

    ppmi = np.full((n, n), np.nan, dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            if not observed[i, j]:
                continue

            p_ij = p_joint[i, j]
            p_i, p_j = p_marginal[i], p_marginal[j]

            # importantly, we treat negative PMI values as missing (NaN) rather than zero
            # for SRF optimization on very sparse entries and not to bias towards zero
            # optimization
            if p_i > 0 and p_j > 0:
                pmi = np.log2(p_ij / (p_i * p_j + 1e-10) + 1e-10)
                if pmi > 0:
                    ppmi[i, j] = ppmi[j, i] = pmi

    np.fill_diagonal(ppmi, np.nan)
    return ppmi


def make_ppmi_graph(
    sources: list[str],
    targets: list[str],
    counts: list[float],
    symmetrization: str = "geometric_mean",
    top_n: int | None = None,
) -> tuple[np.ndarray, list[str], dict]:
    """Build PPMI similarity graph from directed association data.

    Returns:
        ppmi_matrix: Symmetric PPMI matrix (NaN for missing/diagonal)
        vocabulary: Sorted list of words
        metadata: Graph statistics
    """
    g = build_directed_graph(sources, targets, counts)
    g = clean_graph(g)
    vocabulary = _select_vocabulary(g, top_n)
    if not vocabulary:
        raise ValueError("Graph is empty after top-n filtering")

    subgraph = g.subgraph(vocabulary).copy()

    counts_matrix = graph_to_matrix(subgraph, vocabulary)
    counts_sym = symmetrize_matrix(counts_matrix, method=symmetrization)
    ppmi_matrix = compute_ppmi(counts_sym)

    ppmi_obs = ppmi_matrix[~np.isnan(ppmi_matrix)]
    n_edges_directed = subgraph.number_of_edges()
    n_edges_sym = np.sum(~np.isnan(counts_sym)) // 2

    metadata = {
        "n_words": len(vocabulary),
        "n_edges_directed": n_edges_directed,
        "n_edges_bidirectional": n_edges_sym,
        "pct_missing": float(np.sum(np.isnan(ppmi_matrix)) / ppmi_matrix.size * 100),
        "symmetrization": symmetrization,
        "ppmi_min": float(ppmi_obs.min()) if len(ppmi_obs) > 0 else 0.0,
        "ppmi_max": float(ppmi_obs.max()) if len(ppmi_obs) > 0 else 0.0,
        "ppmi_mean": float(ppmi_obs.mean()) if len(ppmi_obs) > 0 else 0.0,
    }

    return ppmi_matrix, vocabulary, metadata
