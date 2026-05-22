"""
Random Walk Similarity for SWOW following De Deyne et al. (2019).

IMPLEMENTATION PLAN
===================

Input: strength.SWOW-EN.R123.csv (or R1) with columns: cue, response, R123, N, R123.Strength

Step 1: Build directed graph from cue -> response edges with counts as weights
Step 2: Clean graph (remove self-loops, out-degree=0 nodes, extract largest SCC)
Step 3: Convert to adjacency matrix

Step 4: L1-normalize rows (transition probabilities)
Step 5: Apply PPMI transformation
Step 6: L1-normalize rows again

Step 7: Katz walk: G_rw = (I - alpha*P)^{-1}  [alpha=0.75]
        - This is NOT symmetric (directed transition matrix)

Step 8: Apply PPMI again to the result
Step 9: L1-normalize rows again

Step 10: Compute cosine similarity matrix
         - cosine(row_i, row_j) IS symmetric
         - Final output is symmetric similarity matrix

KEY ANSWERS:
- Input file: Use strength.SWOW-EN.R123.csv (same as R100 aggregated)
- Symmetry: Final cosine similarity IS symmetric, intermediate Katz walk is NOT
- Alpha: 0.75 (default from paper)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.utils import get_output_dir
from utils.graphs import build_directed_graph, clean_graph, graph_to_matrix

OUTPUT_DIR = get_output_dir()

DATA_DIR = PROJECT_ROOT / "data" / "small-world-of-words"
ALPHA = 0.75


def load_swow_edges(data_dir: Path, use_r123: bool = True) -> tuple[list, list, list]:
    """Load SWOW strength data as edge list."""
    import pandas as pd

    suffix = "R123" if use_r123 else "R1"
    path = data_dir / f"strength.SWOW-EN.{suffix}.20180827.csv"
    print(f"Loading {path}")

    df = pd.read_csv(path, sep="\t")

    # Drop rows with missing cue or response
    df = df.dropna(subset=["cue", "response"])

    # Ensure cue and response are strings
    df["cue"] = df["cue"].astype(str)
    df["response"] = df["response"].astype(str)

    cues = df["cue"].tolist()
    responses = df["response"].tolist()
    counts = df[suffix].tolist()

    print(f"  Loaded {len(cues)} edges")
    print(f"  Unique cues: {df['cue'].nunique()}")
    print(f"  Unique responses: {df['response'].nunique()}")

    return cues, responses, counts


def l1_normalize(P: np.ndarray) -> np.ndarray:
    """Row-normalize matrix so each row sums to 1."""
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    return P / row_sums


def ppmi(P: np.ndarray) -> np.ndarray:
    """
    PPMI transformation matching SWOWEN-2018 R implementation.

    R code:
        N = dim(P)[1]
        D = Diagonal(x = 1/(colSums(P)/N))
        P = P %*% D
        P@x = log2(P@x)
        P2 = pmax(P,0)
    """
    n = P.shape[0]
    col_sums = P.sum(axis=0)
    col_sums[col_sums == 0] = 1  # avoid division by zero

    # D = diag(N / colSums(P))
    D_inv = n / col_sums

    # P = P %*% D (scale columns by inverse marginals)
    P_scaled = P * D_inv  # broadcasting: each column scaled

    # log2 transform (only positive values)
    with np.errstate(divide="ignore", invalid="ignore"):
        P_log = np.where(P_scaled > 0, np.log2(P_scaled), 0)

    # Clamp negatives to 0
    return np.maximum(P_log, 0)


def katz_walk(P: np.ndarray, alpha: float = 0.75) -> np.ndarray:
    """
    Katz walk: G_rw = (I - alpha*P)^{-1}

    Solves the linear system (I - alpha*P) @ G_rw = I
    This is more numerically stable than direct inversion.
    """
    n = P.shape[0]
    I = np.eye(n)
    print(f"  Computing Katz walk (n={n}, alpha={alpha})...")

    # G_rw = (I - alpha*P)^{-1} = solve(I - alpha*P, I)
    G_rw = solve(I - alpha * P, I)

    return G_rw


def compute_rw_similarity(
    counts: np.ndarray,
    alpha: float = 0.75,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute random walk similarity following De Deyne et al. (2019).

    Pipeline:
        counts -> L1-norm -> PPMI -> L1-norm -> Katz -> PPMI -> L1-norm -> cosine

    Args:
        counts: Directed adjacency matrix (cue x cue) with association counts
        alpha: Katz walk damping parameter (default 0.75)
        verbose: Print progress

    Returns:
        Symmetric similarity matrix (n x n)
    """
    if verbose:
        print("Step 1: L1-normalize (transition probabilities)")
    P = l1_normalize(counts)

    if verbose:
        print("Step 2: PPMI transformation")
    P = ppmi(P)

    if verbose:
        print("Step 3: L1-normalize after PPMI")
    P = l1_normalize(P)

    if verbose:
        print("Step 4: Katz walk (I - alpha*P)^{-1}")
    P = katz_walk(P, alpha)

    if verbose:
        print("Step 5: PPMI transformation (second)")
    P = ppmi(P)

    if verbose:
        print("Step 6: L1-normalize after second PPMI")
    P = l1_normalize(P)

    if verbose:
        print("Step 7: Cosine similarity")
    S = cosine_similarity(P)

    return S


def main():
    print("=" * 60)
    print("SWOW Random Walk Similarity")
    print("=" * 60)

    # Load edges
    cues, responses, counts = load_swow_edges(DATA_DIR, use_r123=True)

    # Build and clean graph
    print("\nBuilding directed graph...")
    g = build_directed_graph(cues, responses, counts)
    print(f"  Raw graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    print("Cleaning graph (SCC extraction)...")
    g = clean_graph(g)
    print(f"  Clean graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    # Get vocabulary and adjacency matrix
    vocabulary = sorted(g.nodes())
    print(f"\nVocabulary size: {len(vocabulary)}")

    print("Converting to adjacency matrix...")
    counts_matrix = graph_to_matrix(g, vocabulary)
    print(f"  Matrix shape: {counts_matrix.shape}")
    print(f"  Non-zero entries: {np.count_nonzero(counts_matrix)}")
    print(f"  Sparsity: {100 * (1 - np.count_nonzero(counts_matrix) / counts_matrix.size):.1f}%")

    # Compute random walk similarity
    print("\n" + "=" * 60)
    print("Computing Random Walk Similarity")
    print("=" * 60)

    S_rw = compute_rw_similarity(counts_matrix, alpha=ALPHA, verbose=True)

    # Verify symmetry
    is_symmetric = np.allclose(S_rw, S_rw.T)
    print(f"\nResult is symmetric: {is_symmetric}")

    # Statistics
    print(f"\nSimilarity matrix statistics:")
    print(f"  Shape: {S_rw.shape}")
    print(f"  Min: {S_rw.min():.4f}")
    print(f"  Max: {S_rw.max():.4f}")
    print(f"  Mean: {S_rw.mean():.4f}")
    print(f"  Diagonal mean: {np.diag(S_rw).mean():.4f}")

    # Compare with current PPMI approach
    print("\n" + "=" * 60)
    print("Comparing with current PPMI approach")
    print("=" * 60)

    from datasets.swow import compute_ppmi
    from utils.graphs import symmetrize_matrix

    counts_sym = symmetrize_matrix(counts_matrix, method="sum", bidirectional_only=False)
    S_ppmi = compute_ppmi(counts_sym, negative_as_nan=False)

    # Fill NaN with 0 for comparison
    S_ppmi_filled = np.nan_to_num(S_ppmi, nan=0.0)

    print(f"PPMI matrix statistics:")
    print(f"  Non-NaN entries: {np.count_nonzero(~np.isnan(S_ppmi))}")
    print(f"  Zero entries (after filling): {np.count_nonzero(S_ppmi_filled == 0)}")
    print(f"  Min (non-zero): {S_ppmi_filled[S_ppmi_filled > 0].min():.4f}")
    print(f"  Max: {np.nanmax(S_ppmi):.4f}")
    print(f"  Mean (non-zero): {S_ppmi_filled[S_ppmi_filled > 0].mean():.4f}")

    # Save results
    print("\n" + "=" * 60)
    print("Saving results")
    print("=" * 60)

    np.save(OUTPUT_DIR / "S_rw.npy", S_rw)
    np.save(OUTPUT_DIR / "vocabulary.npy", np.array(vocabulary))

    # Save a small sample for inspection
    import pandas as pd

    sample_words = ["dog", "cat", "car", "house", "love", "happy", "sad", "run", "eat", "think"]
    sample_indices = [vocabulary.index(w) for w in sample_words if w in vocabulary]
    sample_vocab = [vocabulary[i] for i in sample_indices]

    if len(sample_indices) > 0:
        S_sample = S_rw[np.ix_(sample_indices, sample_indices)]
        df_sample = pd.DataFrame(S_sample, index=sample_vocab, columns=sample_vocab)
        df_sample.to_csv(OUTPUT_DIR / "S_rw_sample.csv")
        print(f"\nSample similarity matrix ({len(sample_vocab)} words):")
        print(df_sample.round(3).to_string())

    print(f"\nSaved to {OUTPUT_DIR}")

    # Test bounds estimation
    print("\n" + "=" * 60)
    print("Bounds Estimation Comparison")
    print("=" * 60)

    from pysrf.bounds import pmin_bound, compute_effective_dimension

    # Prepare RW similarity for bounds (fill diagonal)
    S_rw_bounds = S_rw.copy()
    # Diagonal is already 1.0 from cosine similarity

    # Prepare PPMI similarity for bounds
    S_ppmi_bounds = S_ppmi.copy()
    np.fill_diagonal(S_ppmi_bounds, np.nanmax(S_ppmi))  # Fill diagonal with max
    S_ppmi_bounds = np.nan_to_num(S_ppmi_bounds, nan=0.0)  # Fill NaN with 0

    print("\n--- Random Walk Similarity ---")
    fro_norm_rw = np.linalg.norm(S_rw_bounds, "fro")
    spec_norm_rw = np.linalg.norm(S_rw_bounds, 2)
    eff_dim_rw = compute_effective_dimension(fro_norm_rw, spec_norm_rw)
    print(f"  Frobenius norm: {fro_norm_rw:.2f}")
    print(f"  Spectral norm: {spec_norm_rw:.2f}")
    print(f"  Effective dimension: {eff_dim_rw}")

    pmin_rw, _, _, _, _ = pmin_bound(S_rw_bounds, verbose=False)
    print(f"  pmin: {pmin_rw:.4f}")

    print("\n--- PPMI Similarity ---")
    fro_norm_ppmi = np.linalg.norm(S_ppmi_bounds, "fro")
    spec_norm_ppmi = np.linalg.norm(S_ppmi_bounds, 2)
    eff_dim_ppmi = compute_effective_dimension(fro_norm_ppmi, spec_norm_ppmi)
    print(f"  Frobenius norm: {fro_norm_ppmi:.2f}")
    print(f"  Spectral norm: {spec_norm_ppmi:.2f}")
    print(f"  Effective dimension: {eff_dim_ppmi}")

    pmin_ppmi, _, _, _, _ = pmin_bound(S_ppmi_bounds, verbose=False)
    print(f"  pmin: {pmin_ppmi:.4f}")

    print("\n--- Summary ---")
    print(f"  RW effective dimension: {eff_dim_rw} vs PPMI: {eff_dim_ppmi}")
    print(f"  RW pmin: {pmin_rw:.4f} vs PPMI: {pmin_ppmi:.4f}")
    print(f"  RW pmin is {pmin_rw / pmin_ppmi:.1f}x higher than PPMI")


if __name__ == "__main__":
    main()
