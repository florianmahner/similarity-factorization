"""
Advanced Analyses for Word Association Network
==============================================

This module provides comprehensive analyses to validate and characterize SRF dimensions:
1. Dimension properties (correlation with word norms)
2. Network community comparison
3. Cross-validation of missing associations
4. Semantic category enrichment
5. Dimension hierarchy analysis
6. Temporal stability

Usage:
    from advanced_analyses import *
    results = run_all_analyses(W, similarity, vocabulary, data_dir, output_dir)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, hypergeom
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings("ignore")


# ==============================================================================
# 1. DIMENSION PROPERTIES ANALYSIS
# ==============================================================================


def analyze_dimension_properties(W, vocabulary, data_dir):
    """
    Correlate dimension loadings with word-level properties.

    Tests whether dimensions are organized by:
    - Word frequency (common vs. rare)
    - Response diversity (entropy)
    - Network connectivity (coverage)

    Parameters
    ----------
    W : ndarray (n_words, n_dims)
        Factor matrix from SRF
    vocabulary : list
        Word labels
    data_dir : Path
        Directory containing SWOW data files

    Returns
    -------
    df_results : DataFrame
        Correlation results for each dimension
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 1: DIMENSION PROPERTIES")
    print("=" * 80)

    # Load word statistics
    cue_stats = pd.read_csv(data_dir / "cueStats.SWOW-EN.R123.20180827.csv")
    response_stats = pd.read_csv(data_dir / "responseStats.SWOW-EN.20180827.csv")

    # Match with vocabulary
    word_props = []
    for word in vocabulary:
        # Get cue properties
        cue_row = cue_stats[cue_stats["cue"] == word]
        if len(cue_row) > 0:
            coverage = cue_row["coverage"].values[0]
            entropy = cue_row["H"].values[0]
        else:
            coverage = np.nan
            entropy = np.nan

        # Get response properties
        resp_row = response_stats[response_stats["response"] == word]
        if len(resp_row) > 0:
            freq_r1 = resp_row["Freq.R1"].values[0]
            freq_r123 = resp_row["Freq.R123"].values[0]
            types_r123 = resp_row["Types.R123"].values[0]
        else:
            freq_r1 = 0
            freq_r123 = 0
            types_r123 = 0

        word_props.append(
            {
                "word": word,
                "coverage": coverage,
                "entropy": entropy,
                "freq_r1": freq_r1,
                "freq_r123": freq_r123,
                "types_r123": types_r123,
            }
        )

    df_props = pd.DataFrame(word_props)

    # Log-transform frequencies (they're heavily skewed)
    df_props["log_freq_r123"] = np.log1p(df_props["freq_r123"])
    df_props["log_types"] = np.log1p(df_props["types_r123"])

    # Compute correlations for each dimension
    results = []
    n_dims = W.shape[1]

    for dim in range(n_dims):
        loadings = W[:, dim]

        # Correlate with each property
        corr_coverage = pearsonr(
            loadings, df_props["coverage"].fillna(df_props["coverage"].mean())
        )[0]
        corr_entropy = pearsonr(
            loadings, df_props["entropy"].fillna(df_props["entropy"].mean())
        )[0]
        corr_freq = pearsonr(loadings, df_props["log_freq_r123"])[0]
        corr_types = pearsonr(loadings, df_props["log_types"])[0]

        results.append(
            {
                "dimension": dim,
                "corr_coverage": corr_coverage,
                "corr_entropy": corr_entropy,
                "corr_log_freq": corr_freq,
                "corr_log_types": corr_types,
            }
        )

    df_results = pd.DataFrame(results)

    # Summary statistics
    print("\nCorrelation Summary (mean ± std):")
    print(
        f"  Coverage:  {df_results['corr_coverage'].mean():.3f} ± {df_results['corr_coverage'].std():.3f}"
    )
    print(
        f"  Entropy:   {df_results['corr_entropy'].mean():.3f} ± {df_results['corr_entropy'].std():.3f}"
    )
    print(
        f"  Log Freq:  {df_results['corr_log_freq'].mean():.3f} ± {df_results['corr_log_freq'].std():.3f}"
    )
    print(
        f"  Log Types: {df_results['corr_log_types'].mean():.3f} ± {df_results['corr_log_types'].std():.3f}"
    )

    # Find dimensions with strong property correlations
    print("\nDimensions with strong correlations (|r| > 0.3):")
    for _, row in df_results.iterrows():
        strong_corrs = []
        if abs(row["corr_coverage"]) > 0.3:
            strong_corrs.append(f"coverage r={row['corr_coverage']:.3f}")
        if abs(row["corr_entropy"]) > 0.3:
            strong_corrs.append(f"entropy r={row['corr_entropy']:.3f}")
        if abs(row["corr_log_freq"]) > 0.3:
            strong_corrs.append(f"freq r={row['corr_log_freq']:.3f}")
        if abs(row["corr_log_types"]) > 0.3:
            strong_corrs.append(f"types r={row['corr_log_types']:.3f}")

        if strong_corrs:
            print(f"  Dim {int(row['dimension'])}: {', '.join(strong_corrs)}")

    return df_results, df_props


# ==============================================================================
# 2. NETWORK COMMUNITY COMPARISON
# ==============================================================================


def compare_with_network_communities(similarity, W, vocabulary):
    """
    Compare SRF dimensions to graph-based community detection.

    Tests whether SRF captures different structure than modularity-based clustering.

    Parameters
    ----------
    similarity : ndarray
        Symmetric association matrix (with NaN for missing)
    W : ndarray
        Factor matrix from SRF
    vocabulary : list
        Word labels

    Returns
    -------
    df_overlap : DataFrame
        Overlap statistics between dimensions and communities
    communities : list
        Detected communities (sets of words)
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 2: NETWORK COMMUNITY COMPARISON")
    print("=" * 80)

    # Build graph from observed similarities
    G = nx.Graph()
    n = len(vocabulary)

    for i in range(n):
        for j in range(i + 1, n):
            if not np.isnan(similarity[i, j]) and similarity[i, j] > 0:
                G.add_edge(vocabulary[i], vocabulary[j], weight=similarity[i, j])

    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Detect communities using Louvain
    try:
        from networkx.algorithms import community as nx_comm

        communities = list(nx_comm.louvain_communities(G, seed=42))
        print(f"Detected {len(communities)} communities (Louvain)")
    except:
        # Fallback to greedy modularity
        communities = list(nx.community.greedy_modularity_communities(G))
        print(f"Detected {len(communities)} communities (greedy modularity)")

    # Sort by size
    communities = sorted(communities, key=len, reverse=True)

    print("\nCommunity sizes:")
    for i, comm in enumerate(communities[:10]):
        print(f"  Community {i}: {len(comm)} words")

    # Compare SRF dimensions to communities
    overlap_results = []

    for dim in range(W.shape[1]):
        # Get top words for this dimension
        loadings = W[:, dim]
        top_idx = np.argsort(loadings)[-50:]
        top_words = set([vocabulary[i] for i in top_idx])

        # Find best-matching community
        overlaps = [len(top_words & comm) for comm in communities]
        best_comm_idx = np.argmax(overlaps)
        best_overlap = overlaps[best_comm_idx]
        overlap_pct = 100 * best_overlap / len(top_words)

        overlap_results.append(
            {
                "dimension": dim,
                "best_community": best_comm_idx,
                "overlap_count": best_overlap,
                "overlap_pct": overlap_pct,
                "community_size": len(communities[best_comm_idx]),
            }
        )

    df_overlap = pd.DataFrame(overlap_results)

    print(f"\nMean overlap: {df_overlap['overlap_pct'].mean():.1f}%")
    print(
        f"Overlap range: [{df_overlap['overlap_pct'].min():.1f}%, {df_overlap['overlap_pct'].max():.1f}%]"
    )

    # Dimensions with low overlap = capture different structure
    low_overlap = df_overlap[df_overlap["overlap_pct"] < 30]
    if len(low_overlap) > 0:
        print(f"\nDimensions with low community overlap (<30%): {len(low_overlap)}")
        print("These capture latent structure beyond local clustering")

    return df_overlap, communities


# ==============================================================================
# 4. SEMANTIC CATEGORY ENRICHMENT
# ==============================================================================


def test_category_enrichment(W, vocabulary, top_n=100):
    """
    Test if dimensions are enriched for semantic categories.

    Uses simple category definitions based on common semantic groupings.
    For more comprehensive analysis, integrate external semantic norms.

    Parameters
    ----------
    W : ndarray
        Factor matrix from SRF
    vocabulary : list
        Word labels
    top_n : int
        Number of top words per dimension to test

    Returns
    -------
    df_enrichment : DataFrame
        Enrichment results (hypergeometric test)
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 4: SEMANTIC CATEGORY ENRICHMENT")
    print("=" * 80)

    # Define simple semantic categories
    # (In practice, use external norms like THINGS, WordNet, etc.)
    categories = {
        "animals": [
            "dog",
            "cat",
            "bird",
            "fish",
            "mouse",
            "horse",
            "cow",
            "pig",
            "sheep",
            "chicken",
            "lion",
            "tiger",
            "bear",
            "elephant",
            "monkey",
            "snake",
            "rabbit",
            "deer",
            "fox",
            "wolf",
            "duck",
            "goose",
            "owl",
            "eagle",
            "crow",
            "penguin",
            "dolphin",
            "whale",
            "shark",
            "turtle",
        ],
        "food": [
            "apple",
            "banana",
            "orange",
            "bread",
            "cheese",
            "meat",
            "milk",
            "egg",
            "butter",
            "rice",
            "pizza",
            "sandwich",
            "cake",
            "cookie",
            "chocolate",
            "coffee",
            "tea",
            "water",
            "juice",
            "beer",
            "wine",
            "pasta",
            "salad",
            "soup",
            "chicken",
            "beef",
            "fish",
            "vegetable",
            "fruit",
            "food",
        ],
        "emotions": [
            "happy",
            "sad",
            "angry",
            "fear",
            "love",
            "hate",
            "joy",
            "pain",
            "worry",
            "anxiety",
            "excitement",
            "boredom",
            "surprise",
            "disgust",
            "pride",
            "shame",
            "guilt",
            "jealousy",
            "envy",
            "sympathy",
            "empathy",
            "compassion",
            "emotion",
            "feeling",
            "mood",
        ],
        "body_parts": [
            "head",
            "eye",
            "ear",
            "nose",
            "mouth",
            "hand",
            "foot",
            "leg",
            "arm",
            "finger",
            "toe",
            "neck",
            "shoulder",
            "chest",
            "stomach",
            "back",
            "heart",
            "brain",
            "blood",
            "bone",
            "muscle",
            "skin",
            "hair",
            "face",
            "body",
        ],
        "colors": [
            "red",
            "blue",
            "green",
            "yellow",
            "orange",
            "purple",
            "pink",
            "brown",
            "black",
            "white",
            "gray",
            "grey",
            "color",
            "colour",
            "bright",
            "dark",
            "light",
        ],
        "numbers": [
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "zero",
            "hundred",
            "thousand",
            "million",
            "number",
            "count",
            "many",
            "few",
            "several",
        ],
        "time": [
            "day",
            "night",
            "morning",
            "evening",
            "noon",
            "midnight",
            "hour",
            "minute",
            "second",
            "week",
            "month",
            "year",
            "time",
            "clock",
            "watch",
            "today",
            "tomorrow",
            "yesterday",
            "past",
            "present",
            "future",
            "now",
            "then",
            "when",
            "always",
            "never",
            "sometimes",
        ],
        "family": [
            "mother",
            "father",
            "parent",
            "child",
            "son",
            "daughter",
            "brother",
            "sister",
            "grandmother",
            "grandfather",
            "aunt",
            "uncle",
            "cousin",
            "wife",
            "husband",
            "family",
            "baby",
            "kid",
            "mom",
            "dad",
            "grandma",
            "grandpa",
        ],
    }

    print(f"\nTesting {len(categories)} categories")
    print(f"Total vocabulary: {len(vocabulary)}")

    enrichment_results = []

    for dim in range(W.shape[1]):
        loadings = W[:, dim]
        top_idx = np.argsort(loadings)[-top_n:]
        top_words = set([vocabulary[i] for i in top_idx])

        for cat_name, cat_words in categories.items():
            # Filter category words to those in vocabulary
            cat_in_vocab = set(cat_words) & set(vocabulary)

            if len(cat_in_vocab) == 0:
                continue

            # Count overlap
            overlap = len(top_words & cat_in_vocab)

            # Hypergeometric test
            M = len(vocabulary)  # population size
            n = len(cat_in_vocab)  # successes in population
            N = top_n  # sample size
            k = overlap  # observed successes

            # P(X >= k)
            p_value = hypergeom.sf(k - 1, M, n, N)

            enrichment_results.append(
                {
                    "dimension": dim,
                    "category": cat_name,
                    "overlap": overlap,
                    "category_size": len(cat_in_vocab),
                    "expected": (N * n) / M,
                    "fold_enrichment": (
                        overlap / ((N * n) / M) if (N * n) / M > 0 else 0
                    ),
                    "p_value": p_value,
                }
            )

    df_enrichment = pd.DataFrame(enrichment_results)

    # Multiple testing correction (Bonferroni)
    alpha = 0.05
    df_enrichment["significant"] = df_enrichment["p_value"] < (
        alpha / len(df_enrichment)
    )

    # Show significant enrichments
    sig = df_enrichment[df_enrichment["significant"]].sort_values("p_value")

    print(f"\nSignificant enrichments (Bonferroni-corrected α={alpha}):")
    if len(sig) > 0:
        print(f"  Found {len(sig)} significant enrichments")
        for _, row in sig.head(20).iterrows():
            print(
                f"  Dim {int(row['dimension'])}: '{row['category']}' "
                f"({int(row['overlap'])}/{int(row['category_size'])}, "
                f"{row['fold_enrichment']:.1f}x, p={row['p_value']:.2e})"
            )
    else:
        print("  None found (try larger categories or external norms)")

    return df_enrichment


def analyze_dimension_relationships(W, vocabulary):
    """
    Explore hierarchical structure among dimensions.

    Tests whether dimensions have meta-organization (e.g., abstract vs. concrete clusters).

    Parameters
    ----------
    W : ndarray
        Factor matrix from SRF
    vocabulary : list
        Word labels

    Returns
    -------
    dim_sim : ndarray
        Dimension-by-dimension similarity matrix
    """
    print("\n" + "=" * 80)
    print("DIMENSION HIERARCHY")
    print("=" * 80)

    # Dimension similarity = correlation of loading vectors
    dim_sim = np.corrcoef(W.T)
    np.fill_diagonal(dim_sim, 0)  # Remove self-correlations

    print(f"\nDimension similarity matrix: {dim_sim.shape}")
    print(
        f"  Mean similarity: {dim_sim[np.triu_indices_from(dim_sim, k=1)].mean():.3f}"
    )
    print(f"  Max similarity: {dim_sim.max():.3f}")

    # Find strongly related dimensions
    threshold = 0.5
    strong_pairs = np.argwhere(dim_sim > threshold)
    strong_pairs = strong_pairs[
        strong_pairs[:, 0] < strong_pairs[:, 1]
    ]  # Upper triangle

    print(f"\nStrongly related dimension pairs (r > {threshold}):")
    if len(strong_pairs) > 0:
        for i, j in strong_pairs[:10]:
            print(f"  Dim {i} <-> Dim {j}: r={dim_sim[i, j]:.3f}")
    else:
        print("  None found (dimensions are largely independent)")

    return dim_sim


def build_graph_and_fit(df_raw, rank):
    """Helper function to build graph and fit SRF from raw data."""
    from pysrf import SRF

    # Convert to long format
    df_long = pd.melt(
        df_raw, id_vars=["cue"], value_vars=["R1", "R2", "R3"], value_name="response"
    ).dropna(subset=["response"])

    # Count frequencies
    edge_table = df_long.groupby(["cue", "response"]).size().reset_index(name="weight")

    # Build directed graph
    import networkx as nx

    G = nx.DiGraph()
    for _, row in edge_table.iterrows():
        G.add_edge(row["cue"], row["response"], weight=row["weight"])

    # Remove outdegree=0 and extract SCC
    G.remove_nodes_from([n for n in G.nodes() if G.out_degree(n) == 0])
    G.remove_edges_from(list(nx.selfloop_edges(G)))

    largest_scc = max(nx.strongly_connected_components(G), key=len)
    G_scc = G.subgraph(largest_scc).copy()

    # Build adjacency
    vocab = sorted(G_scc.nodes())
    n = len(vocab)
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    A = np.full((n, n), np.nan, dtype=np.float32)
    for source, target, data in G_scc.edges(data=True):
        i = word_to_idx[source]
        j = word_to_idx[target]
        A[i, j] = data["weight"]

    # Normalize to strengths
    row_sums = np.nansum(A, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    A = A / row_sums

    # Symmetrize with NaN preservation
    A_sym = np.full((n, n), np.nan, dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                A_sym[i, j] = 0
                continue

            has_ij = ~np.isnan(A[i, j])
            has_ji = ~np.isnan(A[j, i])

            if has_ij and has_ji:
                A_sym[i, j] = (A[i, j] + A[j, i]) / 2
            elif has_ij:
                A_sym[i, j] = A[i, j]
            elif has_ji:
                A_sym[i, j] = A[j, i]

            A_sym[j, i] = A_sym[i, j]

    # Fit SRF
    srf = SRF(
        rank=rank,
        max_outer=30,
        max_inner=30,
        random_state=42,
        init="random_sqrt",
        tol=1e-4,
        verbose=0,
    )
    srf.fit(A_sym)

    return srf.w_, vocab


def match_dimensions(W1, W2):
    """Match dimensions between two factor matrices by correlation."""
    from scipy.optimize import linear_sum_assignment

    # Compute all pairwise correlations
    corr_matrix = np.zeros((W1.shape[1], W2.shape[1]))
    for i in range(W1.shape[1]):
        for j in range(W2.shape[1]):
            corr_matrix[i, j] = pearsonr(W1[:, i], W2[:, j])[0]

    # Use Hungarian algorithm to find best matching
    row_ind, col_ind = linear_sum_assignment(-corr_matrix)  # Maximize correlation

    matches = []
    for i, j in zip(row_ind, col_ind):
        matches.append((i, j, corr_matrix[i, j]))

    # Sort by correlation
    matches = sorted(matches, key=lambda x: x[2], reverse=True)

    return matches


# ==============================================================================
# RUN ALL ANALYSES
# ==============================================================================


def run_all_analyses(W, similarity, vocabulary, data_dir, output_dir=None):
    """
    Run all advanced analyses and save results.

    Parameters
    ----------
    W : ndarray
        Factor matrix from SRF
    similarity : ndarray
        Symmetric association matrix
    vocabulary : list
        Word labels
    data_dir : Path
        Directory containing SWOW data files
    output_dir : Path, optional
        Directory to save results

    Returns
    -------
    results : dict
        Dictionary containing all analysis results
    """
    results = {}

    # 1. Dimension properties
    try:
        df_props, df_word_props = analyze_dimension_properties(W, vocabulary, data_dir)
        results["dimension_properties"] = df_props
        results["word_properties"] = df_word_props
        if output_dir:
            df_props.to_csv(
                output_dir / "analysis_dimension_properties.csv", index=False
            )
    except Exception as e:
        print(f"\nError in dimension properties analysis: {e}")

    # 2. Network communities
    try:
        df_overlap, communities = compare_with_network_communities(
            similarity, W, vocabulary
        )
        results["community_overlap"] = df_overlap
        results["communities"] = communities
        if output_dir:
            df_overlap.to_csv(
                output_dir / "analysis_community_overlap.csv", index=False
            )
    except Exception as e:
        print(f"\nError in community comparison: {e}")

    # 4. Category enrichment
    try:
        df_enrichment = test_category_enrichment(W, vocabulary)
        results["category_enrichment"] = df_enrichment
        if output_dir:
            df_enrichment.to_csv(
                output_dir / "analysis_category_enrichment.csv", index=False
            )
    except Exception as e:
        print(f"\nError in category enrichment: {e}")

    # 5. Dimension hierarchy
    try:
        dim_sim = analyze_dimension_relationships(W, vocabulary)
        results["dimension_similarity"] = dim_sim
        if output_dir:
            np.save(output_dir / "analysis_dimension_similarity.npy", dim_sim)
    except Exception as e:
        print(f"\nError in dimension hierarchy: {e}")

    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETE")
    print("=" * 80)

    return results
