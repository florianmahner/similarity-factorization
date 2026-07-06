"""SWOW (Small World of Words) data loading and similarity computation.

Supports two similarity measures:
- PPMI (Positive Pointwise Mutual Information): Local co-occurrence measure
- RW (Random Walk): Global similarity via Katz walks, following De Deyne et al. (2019)

Reference:
    De Deyne, S., Navarro, D.J., Perfors, A. et al. (2019).
    The "Small World of Words" English word association norms for over 12,000 cue words.
    Behavior Research Methods, 51, 987-1006.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.metrics.pairwise import cosine_similarity

from utils.graphs import build_directed_graph, clean_graph, graph_to_matrix, symmetrize_matrix


def load_swow_data(data_dir: Path, use_all_responses: bool = False) -> pd.DataFrame:
    """Load SWOW strength data (cue, response, count, strength)."""
    suffix = "R123" if use_all_responses else "R1"
    df = pd.read_csv(data_dir / f"strength.SWOW-EN.{suffix}.20180827.csv", sep="\t")
    df = pd.DataFrame(
        {
            "cue": df["cue"],
            "response": df["response"],
            "count": df[suffix],
            "strength": df[f"{suffix}.Strength"],
        }
    )
    return df.dropna(subset=["response"])


def filter_by_word_length(
    cues: list[str], responses: list[str], counts: list[float], min_length: int
) -> tuple[list[str], list[str], list[float]]:
    """Keep only edges where both words meet minimum length."""
    filtered = [
        (c, r, cnt)
        for c, r, cnt in zip(cues, responses, counts)
        if isinstance(c, str) and isinstance(r, str) and len(c) >= min_length and len(r) >= min_length
    ]
    if not filtered:
        return [], [], []
    return zip(*filtered) if filtered else ([], [], [])


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


def compute_ppmi(
    counts: np.ndarray, negative_as_nan: bool = False, smoothing: float = 1e-7
) -> np.ndarray:
    """
    Calculate PPMI matrix from count matrix using vectorized operations.

    The diagonal is set to s_ii = -log2(p_i), the self-information of word i,
    which equals PMI(i,i) under the identity assumption p_ii = p_i.

    Args:
        counts: Symmetric count matrix.
        negative_as_nan: If True, Negative PMI is NaN (Missing).
                         If False, Negative PMI is 0 (Standard SPPMI).
        smoothing: Small constant to prevent division by zero.
    """
    n = counts.shape[0]

    total = np.nansum(counts)
    if total == 0:
        raise ValueError("Count matrix is empty")

    counts_clean = np.nan_to_num(counts, nan=0.0)
    p_joint = counts_clean / total
    row_sum = counts_clean.sum(axis=1)
    p_marginal = row_sum / total
    p_indep = np.outer(p_marginal, p_marginal)
    p_indep[p_indep < smoothing] = smoothing

    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2(p_joint / p_indep)

    ppmi = np.full((n, n), np.nan, dtype=np.float32)

    if negative_as_nan:
        mask_positive = pmi > 0
        ppmi[mask_positive] = pmi[mask_positive]
    else:
        ppmi = np.maximum(pmi, 0)
        if np.isnan(counts).any():
            ppmi[np.isnan(counts)] = np.nan

    self_info = -np.log2(np.maximum(p_marginal, smoothing)).astype(ppmi.dtype)
    np.fill_diagonal(ppmi, self_info)
    return ppmi


def _l1_normalize(P: np.ndarray) -> np.ndarray:
    """Row-normalize matrix so each row sums to 1."""
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return P / row_sums


def _ppmi_rw(P: np.ndarray) -> np.ndarray:
    """
    PPMI transformation for random walk (matches SWOWEN-2018 R implementation).

    R code:
        N = dim(P)[1]
        D = Diagonal(x = 1/(colSums(P)/N))
        P = P %*% D
        P@x = log2(P@x)
        P2 = pmax(P,0)
    """
    n = P.shape[0]
    col_sums = P.sum(axis=0)
    col_sums[col_sums == 0] = 1

    D_inv = n / col_sums
    P_scaled = P * D_inv

    with np.errstate(divide="ignore", invalid="ignore"):
        P_log = np.where(P_scaled > 0, np.log2(P_scaled), 0)

    return np.maximum(P_log, 0)


def _katz_walk(P: np.ndarray, alpha: float = 0.75) -> np.ndarray:
    """
    Katz walk: G_rw = (I - alpha*P)^{-1}

    Solves (I - alpha*P) @ G_rw = I for numerical stability.
    """
    n = P.shape[0]
    I = np.eye(n)
    return solve(I - alpha * P, I)


def compute_rw_similarity(
    counts: np.ndarray,
    alpha: float = 0.75,
) -> np.ndarray:
    """
    Compute random walk similarity following De Deyne et al. (2019).

    Pipeline: counts -> L1-norm -> PPMI -> L1-norm -> Katz -> PPMI -> L1-norm -> cosine

    This produces a dense, symmetric similarity matrix bounded in [0, 1].

    Parameters
    ----------
    counts : np.ndarray
        Directed adjacency matrix with association counts
    alpha : float
        Katz walk damping parameter (default 0.75, as in paper)

    Returns
    -------
    np.ndarray
        Symmetric similarity matrix (n x n) with values in [0, 1]
    """
    P = _l1_normalize(counts)
    P = _ppmi_rw(P)
    P = _l1_normalize(P)
    P = _katz_walk(P, alpha)
    P = _ppmi_rw(P)
    P = _l1_normalize(P)
    return cosine_similarity(P)


def make_ppmi_graph(
    sources: list[str],
    targets: list[str],
    counts: list[float],
    symmetrization: str = "sum",
    top_n: int | None = None,
    bidirectional_only: bool = False,
) -> tuple[np.ndarray, list[str], dict]:
    """
    Build PPMI similarity graph from directed association data.

    Parameters
    ----------
    sources : list[str]
        Source words (cues)
    targets : list[str]
        Target words (responses)
    counts : list[float]
        Association counts
    symmetrization : str
        Method to symmetrize: 'sum', 'mean', 'geometric_mean'
    top_n : int | None
        Keep only top N words by degree (None = all)
    bidirectional_only : bool
        If True, keep only bidirectional edges

    Returns
    -------
    ppmi_matrix : np.ndarray
        PPMI similarity matrix
    vocabulary : list[str]
        Sorted list of words
    metadata : dict
        Graph statistics
    """
    g = build_directed_graph(sources, targets, counts)
    g = clean_graph(g)

    vocabulary = _select_vocabulary(g, top_n)
    if not vocabulary:
        raise ValueError("Graph is empty after top-n filtering")

    subgraph = g.subgraph(vocabulary).copy()
    counts_matrix = graph_to_matrix(subgraph, vocabulary)

    counts_sym = symmetrize_matrix(
        counts_matrix, method=symmetrization, bidirectional_only=bidirectional_only
    )

    ppmi_matrix = compute_ppmi(counts_sym, negative_as_nan=False)

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


def load_swow_ppmi(
    data_dir: Path,
    use_all_responses: bool = False,
    top_n_words: int | None = None,
    min_word_length: int = 1,
    symmetrization: str = "geometric_mean",
    bidirectional_only: bool = False,
) -> tuple[np.ndarray, list[str], dict]:
    """
    Load SWOW data and compute PPMI similarity matrix.

    This is the main entry point for loading word association data.

    Parameters
    ----------
    data_dir : Path
        Directory containing SWOW data files
    use_all_responses : bool
        If True, use R123 (all responses). If False, use R1 only.
    top_n_words : int | None
        Keep only top N words by degree (None = all)
    min_word_length : int
        Minimum word length filter
    symmetrization : str
        Method to symmetrize: 'sum', 'mean', 'geometric_mean'
    bidirectional_only : bool
        If True, keep only bidirectional edges

    Returns
    -------
    ppmi_matrix : np.ndarray
        PPMI similarity matrix
    vocabulary : list[str]
        Sorted list of words
    metadata : dict
        Graph statistics
    """
    data_dir = Path(data_dir)

    df = load_swow_data(data_dir, use_all_responses)
    cues = df["cue"].tolist()
    responses = df["response"].tolist()
    counts = df["count"].tolist()

    if min_word_length > 1:
        cues, responses, counts = filter_by_word_length(
            cues, responses, counts, min_word_length
        )

    return make_ppmi_graph(
        cues,
        responses,
        counts,
        symmetrization=symmetrization,
        top_n=top_n_words,
        bidirectional_only=bidirectional_only,
    )


def load_swow_rw(
    data_dir: Path,
    use_all_responses: bool = False,
    top_n_words: int | None = None,
    min_word_length: int = 1,
    alpha: float = 0.75,
) -> tuple[np.ndarray, list[str], dict]:
    """
    Load SWOW data and compute random walk similarity matrix.

    This implements the global similarity measure from De Deyne et al. (2019),
    which uses Katz walks to capture indirect semantic relationships.

    Parameters
    ----------
    data_dir : Path
        Directory containing SWOW data files
    use_all_responses : bool
        If True, use R123 (all responses). If False, use R1 only.
    top_n_words : int | None
        Keep only top N words by degree (None = all)
    min_word_length : int
        Minimum word length filter
    alpha : float
        Katz walk damping parameter (default 0.75)

    Returns
    -------
    similarity : np.ndarray
        Random walk similarity matrix (dense, symmetric, bounded [0, 1])
    vocabulary : list[str]
        Sorted list of words
    metadata : dict
        Graph statistics
    """
    data_dir = Path(data_dir)

    df = load_swow_data(data_dir, use_all_responses)
    cues = df["cue"].tolist()
    responses = df["response"].tolist()
    counts = df["count"].tolist()

    if min_word_length > 1:
        cues, responses, counts = filter_by_word_length(
            cues, responses, counts, min_word_length
        )

    g = build_directed_graph(cues, responses, counts)
    g = clean_graph(g)

    vocabulary = _select_vocabulary(g, top_n_words)
    if not vocabulary:
        raise ValueError("Graph is empty after filtering")

    subgraph = g.subgraph(vocabulary).copy()
    counts_matrix = graph_to_matrix(subgraph, vocabulary)

    similarity = compute_rw_similarity(counts_matrix, alpha=alpha)

    metadata = {
        "n_words": int(len(vocabulary)),
        "n_edges": int(subgraph.number_of_edges()),
        "similarity_method": "rw",
        "alpha": alpha,
        "sim_min": float(similarity.min()),
        "sim_max": float(similarity.max()),
        "sim_mean": float(similarity.mean()),
    }

    return similarity, vocabulary, metadata


def load_swow_similarity(
    data_dir: Path,
    method: str = "ppmi",
    use_all_responses: bool = False,
    top_n_words: int | None = None,
    min_word_length: int = 1,
    symmetrization: str = "sum",
    bidirectional_only: bool = False,
    alpha: float = 0.75,
) -> tuple[np.ndarray, list[str], dict]:
    """
    Load SWOW data and compute similarity matrix.

    Unified interface for both PPMI and random walk similarity.

    Parameters
    ----------
    data_dir : Path
        Directory containing SWOW data files
    method : str
        Similarity method: 'ppmi' or 'rw'
    use_all_responses : bool
        If True, use R123 (all responses). If False, use R1 only.
    top_n_words : int | None
        Keep only top N words by degree (None = all)
    min_word_length : int
        Minimum word length filter
    symmetrization : str
        Method to symmetrize (PPMI only): 'sum', 'mean', 'geometric_mean'
    bidirectional_only : bool
        If True, keep only bidirectional edges (PPMI only)
    alpha : float
        Katz walk damping parameter (RW only, default 0.75)

    Returns
    -------
    similarity : np.ndarray
        Similarity matrix
    vocabulary : list[str]
        Sorted list of words
    metadata : dict
        Statistics about the computation
    """
    if method == "ppmi":
        return load_swow_ppmi(
            data_dir,
            use_all_responses=use_all_responses,
            top_n_words=top_n_words,
            min_word_length=min_word_length,
            symmetrization=symmetrization,
            bidirectional_only=bidirectional_only,
        )
    elif method == "rw":
        return load_swow_rw(
            data_dir,
            use_all_responses=use_all_responses,
            top_n_words=top_n_words,
            min_word_length=min_word_length,
            alpha=alpha,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ppmi' or 'rw'")
